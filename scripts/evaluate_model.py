"""Full model self-check on a stratified sample of the Fruits-360 test split.

Computes accuracy, macro/weighted precision, recall and F1, a confusion
matrix, and per-class metrics. Uses batched ONNX Runtime inference.
"""
import io
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "PedroSampaio/fruits-360-16-7"
DATASET_ID = "PedroSampaio/fruits-360"
PARQUET = "data/test-00000-of-00001-0d294abe3826b2e6.parquet"
MODEL_PATH = "backend/app/model_assets/model_int8.onnx"
LABELS_PATH = "backend/app/model_assets/labels.json"
SAMPLES_PER_CLASS = 10
BATCH_SIZE = 32
SEED = 42

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    resized = rgb.resize((224, 224), Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)


def main():
    with open(LABELS_PATH, encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    sess = ort.InferenceSession(
        MODEL_PATH, providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name

    # Use the locally cached parquet when available (avoids
    # re-downloading the 98 MB test split and any auth issues).
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cached = list(cache_root.glob(f"datasets--{DATASET_ID.replace('/', '--')}/snapshots/*/{PARQUET}"))
    if cached:
        parquet_path = str(cached[0])
        print(f"Using cached dataset: {parquet_path}")
    else:
        parquet_path = hf_hub_download(
            repo_id=DATASET_ID, filename=PARQUET, repo_type="dataset"
        )
    df = pd.read_parquet(parquet_path)

    # stratified sample: SAMPLES_PER_CLASS per class
    sample = (
        df.groupby("label", sort=True)
        .head(SAMPLES_PER_CLASS)
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )
    print(f"Evaluating {len(sample)} images "
          f"({SAMPLES_PER_CLASS} per class, {df['label'].nunique()} classes)")

    y_true = []
    y_pred = []
    t0 = time.time()

    for start in range(0, len(sample), BATCH_SIZE):
        batch = sample.iloc[start:start + BATCH_SIZE]
        tensors = []
        labels = []
        for _, row in batch.iterrows():
            raw = row["image"]
            img_bytes = raw["bytes"] if isinstance(raw, dict) else raw
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            tensors.append(preprocess(img))
            labels.append(int(row["label"]))
        x = np.stack(tensors)
        logits = sess.run(None, {input_name: x})[0]
        preds = np.argmax(logits, axis=1)
        y_true.extend(labels)
        y_pred.extend(int(p) for p in preds)
        if (start // BATCH_SIZE) % 10 == 0:
            print(f"  batch {start // BATCH_SIZE + 1}/"
                  f"{(len(sample) + BATCH_SIZE - 1) // BATCH_SIZE}")

    elapsed = time.time() - t0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --- metrics ---
    num_classes = len(id2label)
    correct = int((y_true == y_pred).sum())
    accuracy = correct / len(y_true)

    classes = sorted(id2label.keys())
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)

    macro_p = float(precision.mean())
    macro_r = float(recall.mean())
    macro_f1 = float(f1.mean())
    support = tp + fn
    weighted_p = float((precision * support).sum() / support.sum())
    weighted_r = float((recall * support).sum() / support.sum())
    weighted_f1 = float((f1 * support).sum() / support.sum())

    print("\n================ MODEL SELF-CHECK ================")
    print(f"Sample size:  {len(y_true)} images ({SAMPLES_PER_CLASS}/class)")
    print(f"Accuracy:     {correct}/{len(y_true)} = {accuracy*100:.2f}%")
    print(f"Precision (macro):     {macro_p*100:.2f}%")
    print(f"Recall (macro):        {macro_r*100:.2f}%")
    print(f"F1 (macro):            {macro_f1*100:.2f}%")
    print(f"Precision (weighted):  {weighted_p*100:.2f}%")
    print(f"Recall (weighted):     {weighted_r*100:.2f}%")
    print(f"F1 (weighted):         {weighted_f1*100:.2f}%")
    print(f"Inference time: {elapsed:.1f}s total, "
          f"{elapsed/len(y_true)*1000:.0f} ms/image (batched, CPU)")

    # per-class table
    print("\nPer-class metrics (worst 15 by F1):")
    rows = []
    for c in classes:
        rows.append(
            {
                "class": id2label[c],
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            }
        )
    rows_sorted = sorted(rows, key=lambda r: r["f1"])
    for r in rows_sorted[:15]:
        print(f"  {r['class']:<22} P={r['precision']:.3f} "
              f"R={r['recall']:.3f} F1={r['f1']:.3f} n={r['support']}")

    # confusion matrix summary: which classes get confused
    confusion = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            key = (id2label[int(t)], id2label[int(p)])
            confusion[key] = confusion.get(key, 0) + 1
    if confusion:
        print("\nTop misclassifications:")
        for (t, p), n in sorted(confusion.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {t} -> {p}: {n}")
    else:
        print("\nNo misclassifications in the sample.")

    # save detailed results
    results = {
        "sample_size": len(y_true),
        "samples_per_class": SAMPLES_PER_CLASS,
        "accuracy": accuracy,
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "f1_macro": macro_f1,
        "precision_weighted": weighted_p,
        "recall_weighted": weighted_r,
        "f1_weighted": weighted_f1,
        "inference_ms_per_image": elapsed / len(y_true) * 1000,
        "per_class": rows,
        "misclassifications": [
            {"true": t, "pred": p, "count": n}
            for (t, p), n in sorted(confusion.items(), key=lambda kv: -kv[1])
        ],
    }
    out = "scripts/evaluation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {out}")


if __name__ == "__main__":
    main()
