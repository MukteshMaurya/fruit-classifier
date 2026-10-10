"""Paired int8 vs fp32 confidence comparison on the fruit validation set."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.services.classifier import FruitClassifier  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "validation_data" / "fruit"

int8 = FruitClassifier(
    "backend/app/model_assets/model_int8.onnx", "backend/app/model_assets/labels.json"
)
fp32 = FruitClassifier("onnx_model/model_fp32.onnx", "backend/app/model_assets/labels.json")

pairs = []
for cat_dir in sorted(DATA.iterdir()):
    if not cat_dir.is_dir():
        continue
    for img_path in sorted(cat_dir.glob("*")):
        img = Image.open(img_path)
        a = int8.predict(img, top_k=2)[0]
        b = fp32.predict(img, top_k=2)[0]
        pairs.append((a["confidence"], b["confidence"], a["class"] == b["class"], cat_dir.name))

ci = np.array([p[0] for p in pairs])
cf = np.array([p[1] for p in pairs])
diff = ci - cf
print(f"n={len(pairs)}")
print(f"int8: mean={ci.mean():.4f} median={np.median(ci):.4f} min={ci.min():.4f}")
print(f"fp32: mean={cf.mean():.4f} median={np.median(cf):.4f} min={cf.min():.4f}")
print(f"diff (int8-fp32): mean={diff.mean():.4f} min={diff.min():.4f} max={diff.max():.4f}")
print(f"int8 lower than fp32 in {np.mean(diff < -0.01)*100:.1f}% of images (by >0.01)")
print(f"label disagreement: {sum(1 for p in pairs if not p[2])} images")
# largest drops
worst = sorted(pairs, key=lambda p: p[0] - p[1])[:10]
for a, b, same, cat in worst:
    print(f"  {cat:22s} int8={a:.4f} fp32={b:.4f} diff={a-b:+.4f} same_label={same}")
