"""Collect a small validation set for rejection-threshold tuning.

Positives: stratified sample of the Fruits-360 test split (cached
locally in the Hugging Face hub cache; no new download).

Negatives: non-fruit photos (people, cars, phones, animals,
furniture, buildings, ...) fetched as 256px thumbnails from the
Wikimedia Commons API (public-domain / CC-licensed media, no auth
required). Only a handful of images per category are fetched.

Output: validation_data/fruit/<Class>/ and validation_data/nonfruit/<category>/
"""

import io
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation_data"

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
PARQUET_GLOBS = [
    "datasets--PedroSampaio--fruits-360/snapshots/*/data/test-*.parquet",
    "datasets--PedroSampaio--fruits-360/snapshots/*/test-*.parquet",
]

# Non-fruit categories for the negative class.
NONFRUIT_QUERIES = [
    ("person", "portrait person face"),
    ("car", "car automobile street"),
    ("mobile_phone", "smartphone mobile phone"),
    ("dog", "dog animal"),
    ("cat", "cat animal"),
    ("chair", "chair furniture"),
    ("building", "house building architecture"),
    ("bicycle", "bicycle"),
    ("airplane", "airplane"),
    ("horse", "horse animal"),
]
PER_CATEGORY = 10
PER_CLASS = 5  # fruit images per class (565 total for 113 classes)


def collect_fruits():
    parquets = []
    for pattern in PARQUET_GLOBS:
        parquets.extend(HF_CACHE.glob(pattern.replace("/", os.sep)))
    if not parquets:
        raise SystemExit("Fruits-360 test parquet not found in HF cache")
    pq_path = parquets[0]
    print(f"Reading {pq_path}")
    table = pq.read_table(pq_path)
    print("columns:", table.column_names)
    print("rows:", table.num_rows)

    col_names = table.column_names
    label_col = "label" if "label" in col_names else "category"
    image_col = "image" if "image" in col_names else "image_bytes"

    labels = table.column(label_col).to_pylist()
    images = table.column(image_col)

    from collections import defaultdict

    by_class = defaultdict(list)
    for i, lab in enumerate(labels):
        by_class[lab].append(i)

    rng = __import__("random").Random(42)
    out_dir = OUT / "fruit"
    count = 0
    for lab in sorted(by_class):
        idxs = by_class[lab]
        chosen = rng.sample(idxs, min(PER_CLASS, len(idxs)))
        class_dir = out_dir / str(lab).replace(" ", "_").replace("/", "_")
        class_dir.mkdir(parents=True, exist_ok=True)
        for j, idx in enumerate(chosen):
            raw = images[idx].as_py()
            if isinstance(raw, dict):
                raw = raw.get("bytes") or raw.get("data")
            img = Image.open(io.BytesIO(raw))
            img = img.convert("RGB")
            img.save(class_dir / f"{j}.png")
            count += 1
    print(f"Saved {count} fruit images to {out_dir}")


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
    url = (
        "https://commons.wikimedia.org/w/api.php?"
        + urllib.parse.urlencode(params)
    )
    req = urllib.request.Request(
        url, headers={"User-Agent": "fruit-classifier-validation/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    pages = data.get("query", {}).get("pages", {})
    thumbs = []
    for page in pages.values():
        ii = (page.get("imageinfo") or [{}])[0]
        if ii.get("thumburl"):
            thumbs.append(ii["thumburl"])
    return thumbs


def collect_nonfruit():
    out_dir = OUT / "nonfruit"
    total = 0
    for cat, query in NONFRUIT_QUERIES:
        cat_dir = out_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        try:
            thumbs = commons_thumbnails(query, PER_CATEGORY)
        except Exception as exc:  # noqa: BLE001
            print(f"  {cat}: search failed: {exc}")
            continue
        saved = 0
        for url in thumbs:
            if saved >= PER_CATEGORY:
                break
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "fruit-classifier-validation/1.0"},
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    blob = resp.read()
                img = Image.open(io.BytesIO(blob))
                img = img.convert("RGB")
                if min(img.size) < 64:
                    continue
                img.save(cat_dir / f"{saved}.png")
                saved += 1
            except Exception:  # noqa: BLE001
                continue
        print(f"  {cat}: {saved} images")
        total += saved
    print(f"Saved {total} non-fruit images to {out_dir}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("fruit", "all"):
        collect_fruits()
    if what in ("nonfruit", "all"):
        collect_nonfruit()
