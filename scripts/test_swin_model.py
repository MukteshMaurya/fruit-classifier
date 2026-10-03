"""Verify PedroSampaio/fruits-360-16-7 (Swin-base, 113 classes) on Fruits-360 test images."""
import io
import time

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

REPO_ID = "PedroSampaio/fruits-360-16-7"
DATASET_ID = "PedroSampaio/fruits-360"
PARQUET = "data/test-00000-of-00001-0d294abe3826b2e6.parquet"

t0 = time.time()
processor = AutoImageProcessor.from_pretrained(REPO_ID)
model = AutoModelForImageClassification.from_pretrained(REPO_ID)
model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s")
id2label = {int(k): v for k, v in model.config.id2label.items()}
print(f"Classes: {len(id2label)}")

parquet_path = hf_hub_download(
    repo_id=DATASET_ID, filename=PARQUET, repo_type="dataset"
)
df = pd.read_parquet(parquet_path)
print(f"Test images: {len(df)}")

sample = df.groupby("label", sort=True).head(1).sample(n=30, random_state=42)

correct = 0
rows = []
for _, row in sample.iterrows():
    raw = row["image"]
    img_bytes = raw["bytes"] if isinstance(raw, dict) else raw
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    true_label = row["label"]
    t0 = time.time()
    inputs = processor(images=image, return_tensors="pt")
    with __import__("torch").no_grad():
        logits = model(**inputs).logits
    elapsed = time.time() - t0
    probs = __import__("torch").nn.functional.softmax(logits[0], dim=0)
    top_prob, top_idx = probs.topk(1)
    pred_label = int(top_idx.item())
    ok = pred_label == true_label
    correct += ok
    rows.append((true_label, pred_label, float(top_prob), elapsed * 1000, ok))
    print(
        f"true={id2label[true_label]:<22} pred={id2label[pred_label]:<22} "
        f"conf={float(top_prob):.3f} time={elapsed*1000:5.0f}ms {'OK' if ok else 'WRONG'}"
    )

n = len(rows)
print(f"\nAccuracy: {correct}/{n} = {correct/n*100:.1f}%")
print(f"Avg inference time: {sum(r[3] for r in rows)/n:.0f}ms")
