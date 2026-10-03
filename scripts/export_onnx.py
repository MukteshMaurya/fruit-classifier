"""Export the Swin fruit classifier to ONNX and dynamically quantize it."""
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

REPO_ID = "PedroSampaio/fruits-360-16-7"
OUT_DIR = "onnx_model"
os.makedirs(OUT_DIR, exist_ok=True)

processor = AutoImageProcessor.from_pretrained(REPO_ID)
model = AutoModelForImageClassification.from_pretrained(REPO_ID)
model.eval()
id2label = {int(k): v for k, v in model.config.id2label.items()}

dummy = torch.randn(1, 3, 224, 224)
onnx_path = os.path.join(OUT_DIR, "model_fp32.onnx")
t0 = time.time()
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)
print(f"Exported fp32 ONNX in {time.time()-t0:.1f}s -> {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

onnx.checker.check_model(onnx_path)

quant_path = os.path.join(OUT_DIR, "model_int8.onnx")
from onnxruntime.quantization import QuantType, quantize_dynamic
t0 = time.time()
quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)
print(f"Quantized int8 in {time.time()-t0:.1f}s -> {quant_path} ({os.path.getsize(quant_path)/1e6:.1f} MB)")

# Quick sanity check with ONNX Runtime
sess = ort.InferenceSession(quant_path, providers=["CPUExecutionProvider"])
img = Image.new("RGB", (224, 224), color=(200, 100, 50))
inputs = processor(images=img, return_tensors="pt")
t0 = time.time()
logits = sess.run(None, {"pixel_values": inputs["pixel_values"].numpy()})[0]
print(f"ORT inference time: {(time.time()-t0)*1000:.0f}ms")
probs = torch.nn.functional.softmax(torch.tensor(logits[0]), dim=0)
top_prob, top_idx = probs.topk(3)
for p, i in zip(top_prob, top_idx):
    print(f"  {id2label[int(i)]}: {float(p):.3f}")
