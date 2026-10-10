"""Test C probe: unsupported fruit varieties (not in the 113
supported classes) — documents the limitation honestly."""
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.services.classifier import FruitClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation_data" / "unsupported_fruit"
OUT.mkdir(parents=True, exist_ok=True)

# Fruits NOT in the 113 Fruits-360 classes supported by the model
UNSUPPORTED = ["blackberry", "jackfruit", "gooseberry"]


def commons_thumbnails(query, limit):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",
        "gsrlimit": str(limit * 2),
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": "256",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "fruit-classifier-validation/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return [
        (page.get("imageinfo") or [{}])[0]["thumburl"]
        for page in data.get("query", {}).get("pages", {}).values()
        if (page.get("imageinfo") or [{}])[0].get("thumburl")
    ]


cls = FruitClassifier(
    "backend/app/model_assets/model_int8.onnx",
    "backend/app/model_assets/labels.json",
)

CONF_T, MARGIN_T = 0.45, 0.30

for fruit in UNSUPPORTED:
    saved = 0
    for url in commons_thumbnails(fruit, 5):
        if saved >= 5:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fruit-classifier-validation/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                blob = resp.read()
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            if min(img.size) < 64:
                continue
            img.save(OUT / f"{fruit}_{saved}.png")
            d = cls.predict_detailed(img, top_k=2)
            rejected = d["confidence"] < CONF_T or d["margin"] < MARGIN_T
            verdict = "REJECTED" if rejected else "ACCEPTED as " + d["top_class"]
            print(f"{fruit:12s} conf={d['confidence']:.4f} margin={d['margin']:.4f} -> {verdict}")
            saved += 1
        except Exception as exc:  # noqa: BLE001
            print(f"{fruit}: skip ({exc})")
