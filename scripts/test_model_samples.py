"""Verify the fruit classifier against real Fruits-360 test images.

The dataset (PedroSampaio/fruits-360) has 131 fine-grained classes; the model
(bhumong/fruit-classifier-efficientnet-b0) was trained on a merged 76-class
version. We map each dataset class to its merged counterpart (first-word
prefix) and measure accuracy on the mapped subset.
"""
import io
import json
import time
import urllib.request

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

from verify_model import load_model, predict

DATASET_ID = "PedroSampaio/fruits-360"
PARQUET = "data/test-00000-of-00001-0d294abe3826b2e6.parquet"

with urllib.request.urlopen(
    f"https://huggingface.co/api/datasets/{DATASET_ID}", timeout=30
) as r:
    meta = json.load(r)
names = meta["cardData"]["dataset_info"]["features"][1]["dtype"]["class_label"]["names"]
dataset_classes = {int(k): v for k, v in names.items()}
print(f"Dataset classes: {len(dataset_classes)}")

model, config, id2label = load_model()
model_classes = set(id2label.values())
print(f"Model classes: {len(model_classes)}")

mapping = {}
unmapped = []
for idx, name in dataset_classes.items():
    base = name.split()[0]
    if base in model_classes:
        mapping[idx] = base
    else:
        unmapped.append(name)
print(f"Mappable dataset classes: {len(mapping)}")
print(f"Unmapped dataset classes: {unmapped}")

parquet_path = hf_hub_download(
    repo_id=DATASET_ID, filename=PARQUET, repo_type="dataset"
)
df = pd.read_parquet(parquet_path)
df = df[df["label"].isin(mapping.keys())]
print(f"Usable test images: {len(df)}")

sample = df.groupby("label", sort=True).head(1).sample(n=30, random_state=42)

correct = 0
rows = []
for _, row in sample.iterrows():
    raw = row["image"]
    img_bytes = raw["bytes"] if isinstance(raw, dict) else raw
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    t0 = time.time()
    preds = predict(model, id2label, image, top_k=3)
    elapsed = time.time() - t0
    true_class = mapping[int(row["label"])]
    top = preds[0]
    ok = top["class"] == true_class
    correct += ok
    rows.append((true_class, top["class"], top["confidence"], elapsed * 1000, ok))
    print(
        f"true={true_class:<12} pred={top['class']:<12} "
        f"conf={top['confidence']:.3f} time={elapsed*1000:5.0f}ms {'OK' if ok else 'WRONG'}"
    )

n = len(rows)
print(f"\nAccuracy: {correct}/{n} = {correct/n*100:.1f}%")
print(f"Avg inference time: {sum(r[3] for r in rows)/n:.0f}ms")
