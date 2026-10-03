"""Check that manual numpy preprocessing matches transformers AutoImageProcessor."""
import io

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("PedroSampaio/fruits-360-16-7")

parquet_path = hf_hub_download(
    repo_id="PedroSampaio/fruits-360",
    filename="data/test-00000-of-00001-0d294abe3826b2e6.parquet",
    repo_type="dataset",
)
df = pd.read_parquet(parquet_path)
row = df.iloc[0]
raw = row["image"]
img_bytes = raw["bytes"] if isinstance(raw, dict) else raw
image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

ref = processor(images=image, return_tensors="pt")["pixel_values"].numpy()

# Manual preprocessing
img = image.resize((224, 224), Image.BICUBIC)
arr = np.asarray(img, dtype=np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
arr = (arr - mean) / std
manual = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

diff = np.abs(ref - manual).max()
print(f"max abs diff: {diff:.2e}")
print(f"ref shape: {ref.shape}, manual shape: {manual.shape}")
print("MATCH" if diff < 1e-4 else "MISMATCH")
