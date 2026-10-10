"""Fetch a few real-world fruit photos (complex backgrounds) from
Wikimedia Commons to probe confidence on non-studio images."""
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.services.classifier import FruitClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation_data" / "realworld_fruit"
OUT.mkdir(parents=True, exist_ok=True)

QUERIES = ["apple fruit", "banana fruit", "orange fruit", "strawberry fruit"]


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

for q in QUERIES:
    cat = q.split()[0]
    saved = 0
    for url in commons_thumbnails(q, 6):
        if saved >= 6:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fruit-classifier-validation/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                blob = resp.read()
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            if min(img.size) < 64:
                continue
            path = OUT / f"{cat}_{saved}.png"
            img.save(path)
            res = cls.predict(img, top_k=2)
            top, second = res[0], res[1]
            print(f"{q:18s} -> {top['class']:18s} conf={top['confidence']:.4f} "
                  f"margin={top['confidence']-second['confidence']:.4f} (2nd: {second['class']})")
            saved += 1
        except Exception as exc:  # noqa: BLE001
            print(f"{q}: skip ({exc})")
            continue
