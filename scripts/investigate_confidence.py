"""Investigation: model confidence on fruit samples + non-fruit probes."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.services.classifier import FruitClassifier  # noqa: E402

cls = FruitClassifier(
    "backend/app/model_assets/model_int8.onnx",
    "backend/app/model_assets/labels.json",
)

samples = Path("backend/tests/samples")
for p in sorted(samples.glob("*.png")):
    img = Image.open(p)
    res = cls.predict(img, top_k=3)
    top = res[0]
    margin = top["confidence"] - res[1]["confidence"]
    print(f"{p.name:16s} -> {top['class']:20s} conf={top['confidence']:.4f} margin={margin:.4f}")

# Synthetic non-fruit probes: solid colors, noise, gradients
print("\n--- synthetic non-fruit probes ---")
rng = np.random.default_rng(0)
probes = {
    "solid_red": np.full((300, 300, 3), (200, 30, 30), dtype=np.uint8),
    "solid_gray": np.full((300, 300, 3), 128, dtype=np.uint8),
    "noise": rng.integers(0, 256, (300, 300, 3), dtype=np.uint8),
    "gradient": np.tile(
        np.linspace(0, 255, 300, dtype=np.uint8)[None, :, None], (300, 1, 3)
    ),
}
for name, arr in probes.items():
    res = cls.predict(Image.fromarray(arr), top_k=3)
    top = res[0]
    margin = top["confidence"] - res[1]["confidence"]
    print(f"{name:16s} -> {top['class']:20s} conf={top['confidence']:.4f} margin={margin:.4f}")
