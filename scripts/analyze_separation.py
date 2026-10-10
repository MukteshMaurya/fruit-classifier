"""Analyze confidence/margin separation between fruit and non-fruit sets."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.services.classifier import FruitClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "validation_data"


def run_set(cls, folder):
    rows = []
    for cat_dir in sorted(folder.iterdir()):
        if not cat_dir.is_dir():
            continue
        for img_path in sorted(cat_dir.glob("*")):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            try:
                img = Image.open(img_path)
                res = cls.predict(img, top_k=3)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERR {img_path}: {exc}")
                continue
            top = res[0]
            rows.append(
                {
                    "path": str(img_path.relative_to(ROOT)),
                    "category": cat_dir.name,
                    "class": top["class"],
                    "confidence": top["confidence"],
                    "margin": top["confidence"] - res[1]["confidence"],
                    "second": res[1]["class"],
                    "second_conf": res[1]["confidence"],
                }
            )
    return rows


def summarize(name, rows):
    confs = np.array([r["confidence"] for r in rows])
    margins = np.array([r["margin"] for r in rows])
    print(f"\n=== {name} ({len(rows)} images) ===")
    print(f"confidence: min={confs.min():.4f} p5={np.percentile(confs,5):.4f} "
          f"median={np.median(confs):.4f} mean={confs.mean():.4f} max={confs.max():.4f}")
    print(f"margin:     min={margins.min():.4f} p5={np.percentile(margins,5):.4f} "
          f"median={np.median(margins):.4f} mean={margins.mean():.4f} max={margins.max():.4f}")
    return confs, margins


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "backend/app/model_assets/model_int8.onnx"
    cls = FruitClassifier(model, "backend/app/model_assets/labels.json")

    fruit_rows = run_set(cls, DATA / "fruit")
    nonfruit_rows = run_set(cls, DATA / "nonfruit")

    fc, fm = summarize("FRUIT (Fruits-360 test sample)", fruit_rows)
    nc, nm = summarize("NON-FRUIT (Wikimedia Commons)", nonfruit_rows)

    print("\n--- lowest-confidence fruit images ---")
    for r in sorted(fruit_rows, key=lambda r: r["confidence"])[:15]:
        print(f"  {r['confidence']:.4f} margin={r['margin']:.4f} {r['category']} -> {r['class']}")

    print("\n--- highest-confidence non-fruit images ---")
    for r in sorted(nonfruit_rows, key=lambda r: -r["confidence"])[:20]:
        print(f"  {r['confidence']:.4f} margin={r['margin']:.4f} {r['category']} -> {r['class']} (2nd: {r['second']} {r['second_conf']:.4f})")

    # Separation analysis
    print("\n--- threshold analysis ---")
    for conf_t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        for margin_t in [0.1, 0.2, 0.3, 0.5]:
            # accept = conf >= t AND margin >= m
            fruit_accept = np.mean((fc >= conf_t) & (fm >= margin_t))
            nonfruit_accept = np.mean((nc >= conf_t) & (nm >= margin_t))
            print(f"conf>={conf_t:.2f} & margin>={margin_t:.2f}: "
                  f"fruit accepted={fruit_accept:.3f}  nonfruit accepted={nonfruit_accept:.3f}")

    out = ROOT / "scripts" / "separation_analysis.json"
    out.write_text(json.dumps({"fruit": fruit_rows, "nonfruit": nonfruit_rows}, indent=1))
    print(f"\nFull results written to {out}")
