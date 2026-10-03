"""Verify the quantized ONNX model against real Fruits-360 test images."""
import io
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor

REPO_ID = "PedroSampaio/fruits-360-16-7"
PARQUET = "data/test-00000-of-00001-0d294abe3826b2e6.parquet"

processor = AutoImageProcessor.from_pretrained(REPO_ID)
with open("onnx_model/labels.json", "r", encoding="utf-8") as f:
    import json
    id2label = {int(k): v for k, v in json.load(f).items()}

sess_fp32 = ort.InferenceSession(
    "onnx_model/model_fp32.onnx", providers=["CPUExecutionProvider"]
)
sess_int8 = ort.InferenceSession(
    "onnx_model/model_int8.onnx", providers=["CPUExecutionProvider"]
)

parquet_path = hf_hub_download(
    repo_id="PedroSampaio/fruits-360", filename=PARQUET, repo_type="dataset"
)
df = pd.read_parquet(parquet_path)
sample = df.groupby("label", sort=True).head(1).sample(n=30, random_state=42)


def run(sess, image):
    inputs = processor(images=image, return_tensors="pt")
    t0 = time.time()
    logits = sess.run(None, {"pixel_values": inputs["pixel_values"].numpy()})[0]
    elapsed = time.time() - t0
    probs = torch.nn.functional.softmax(torch.tensor(logits[0]), dim=0)
    top_prob, top_idx = probs.topk(3)
    return (
        [float(p) for p in top_prob],
        [int(i) for i in top_idx],
        elapsed * 1000,
    )


correct_fp32 = 0
correct_int8 = 0
rows = []
for _, row in sample.iterrows():
    raw = row["image"]
    img_bytes = raw["bytes"] if isinstance(raw, dict) else raw
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    true_label = int(row["label"])

    p_probs, p_idx, p_ms = run(sess_fp32, image)
    i_probs, i_idx, i_ms = run(sess_int8, image)

    ok_fp32 = p_idx[0] == true_label
    ok_int8 = i_idx[0] == true_label
    correct_fp32 += ok_fp32
    correct_int8 += ok_int8
    rows.append((p_ms, i_ms))
    print(
        f"true={id2label[true_label]:<20} "
        f"fp32={id2label[p_idx[0]]:<20} ({p_probs[0]:.3f}) {'OK' if ok_fp32 else 'WRONG'} | "
        f"int8={id2label[i_idx[0]]:<20} ({i_probs[0]:.3f}) {'OK' if ok_int8 else 'WRONG'} | "
        f"fp32={p_ms:5.0f}ms int8={i_ms:5.0f}ms"
    )

n = len(rows)
print(f"\nfp32 accuracy: {correct_fp32}/{n} = {correct_fp32/n*100:.1f}%")
print(f"int8 accuracy: {correct_int8}/{n} = {correct_int8/n*100:.1f}%")
print(f"fp32 avg time: {sum(r[0] for r in rows)/n:.0f}ms")
print(f"int8 avg time: {sum(r[1] for r in rows)/n:.0f}ms")
