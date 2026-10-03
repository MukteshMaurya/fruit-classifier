# Fruit Classification AI — Project Progress

## Project Overview

A web application that classifies a fruit from an uploaded image and returns the
predicted fruit name and confidence score. Frontend hosted on Vercel, FastAPI
backend on Render, open-source pretrained model (no paid APIs).

## Architecture

- Frontend: Vercel (static HTML/JS/CSS + Node build script)
- Backend: Render (Python 3.12, FastAPI, Uvicorn, ONNX Runtime)
- Model: `PedroSampaio/fruits-360-16-7` (Swin-Base, fine-tuned on Fruits-360),
  exported to ONNX and quantized to int8 (93 MB)
- Dataset: `PedroSampaio/fruits-360` (113 classes, 22,688 test images)

```text
User
  |
  v
Vercel Frontend
  |
  | HTTPS API request (multipart image)
  v
Render FastAPI Backend
  |
  v
ONNX Runtime (Swin-Base int8)
  |
  v
Prediction + Confidence
```

---

# Phase 1 — Project Analysis & Model Search

## Status

COMPLETE

## Findings

- Project directory was empty except for an empty Python 3.12.3 venv
  (`C:\Python files\fruit classifier\.venv`) and opencode config files.
- Python version: 3.12.3; pip 24.0.
- No existing code, models, or data to preserve.
- Internet access to Hugging Face Hub verified (HTTP 200).

## Model Search

Evaluated open-source fruit-classification models on Hugging Face Hub:

| Model | Architecture | Classes | Size | License | Local verification |
|---|---|---|---|---|---|
| PedroSampaio/fruits-360-16-7 | Swin-Base | 113 | 348 MB | Apache-2.0 | **100% (30/30)** |
| Dharma20/vit-base-fruits-360 | ViT-Base | 113 | 343 MB | Apache-2.0 | not selected |
| dima806/fruit_100_types_image_detection | ViT-Base | ~100 | 343 MB | Apache-2.0 | not selected |
| bhumong/fruit-classifier-efficientnet-b0 | EfficientNet-B0 | 76 | ~20 MB | MIT | **50% (15/30) — REJECTED** |
| AnneMarie1/swin-tiny-...-fruits-360 | Swin-Tiny | 24 | — | Apache-2.0 | too few classes |
| VinayHajare/fruits30-resnet18 | ResNet-18 | 30 | 45 MB | Apache-2.0 | no config.json, hard to load |

Key verification: the bhumong EfficientNet model was downloaded and actually run
against real Fruits-360 test images — it achieved only 50% accuracy (systematic
label-mapping errors, e.g. Kiwi→Pear at 0.87 confidence). It was rejected.

The PedroSampaio Swin model was downloaded and run against 30 real Fruits-360
test images: 30/30 correct, confidence 0.98–1.00. Its 113 id2label entries
match the dataset's 113 class names exactly. Model card reports eval accuracy
0.9992 on the Fruits-360 evaluation set.

## Decision

Pretrained model: **YES** (Case A — no training needed)

```text
Model name: PedroSampaio/fruits-360-16-7
Source: Hugging Face Hub (https://huggingface.co/PedroSampaio/fruits-360-16-7)
License: Apache-2.0
Architecture: Swin-Base (microsoft/swin-base-patch4-window7-224 backbone),
  embed_dim=128, depths=[2,2,18,2], window_size=7
Supported classes: 113 Fruits-360 classes (Apple Braeburn ... Watermelon)
Input size: 224x224 RGB
Expected preprocessing: resize 224x224 (bicubic), rescale 1/255,
  normalize ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
Model size: 348 MB fp32 (PyTorch), 93 MB int8 (ONNX, used in production)
Why it is suitable: purpose-trained fruit classifier, 113 fruit classes,
  verified 100% accuracy on real dataset test images, permissive license,
  standard image-classification pipeline.
```

## Files Changed

- `FRUIT_CLASSIFICATION_PROGRESS.md` (created)
- `verify_model.py`, `test_model_samples.py` (model verification scripts,
  later moved to `scripts/`)

## Tests

- Model download + load: PASS (76-class EfficientNet and 113-class Swin both load)
- Local prediction test on 30 real Fruits-360 test images:
  - bhumong/fruit-classifier-efficientnet-b0: 50% — rejected
  - PedroSampaio/fruits-360-16-7: 100% — selected

## Problems

- `curl -la` etc. are PowerShell aliases; switched to PowerShell cmdlets / `curl.exe`.
- `pip install` batches silently failed once; reinstalled packages individually.
- bhumong model accuracy was poor (see above).

## Fixes

- Switched to PedroSampaio/fruits-360-16-7 after empirical verification.

---

# Phase 2 — Dataset & Model

## Status

COMPLETE (Case A — pretrained model integrated; no training performed)

## Dataset

```text
Dataset name: Fruits-360 (PedroSampaio/fruits-360 on HF Hub)
Dataset source: https://huggingface.co/datasets/PedroSampaio/fruits-360
License: MIT
Number of classes: 113
Number of images: 22,688 test images (90k+ total incl. train)
Dataset size: test parquet 98 MB, train parquet 299 MB
Split used: test split (test-00000-of-00001-0d294abe3826b2e6.parquet)
```

Used only for verification (30 sample images, one per class, seed 42).

## Model

- Base: `PedroSampaio/fruits-360-16-7` (Swin-Base, 113 classes, fp32)
- Converted to ONNX (opset 17, dynamic batch axis) → `model_fp32.onnx` (352 MB)
- Dynamically quantized to int8 (ONNX Runtime QInt8) → `model_int8.onnx` (93 MB)
- Labels exported to `labels.json` (id → class name)

## Training

None required — pretrained model used as-is (Case A). No training runs.

## Self-Test Results (local, CPU)

```text
Sample: 30 images (1 per class, random_state=42) from Fruits-360 test split

PyTorch fp32 (transformers):   30/30 = 100.0%   avg 1337 ms
ONNX fp32:                     30/30 = 100.0%   avg  831 ms
ONNX int8 (production):        30/30 = 100.0%   avg  680 ms

Memory (ONNX int8 session):    ~185 MB model footprint, ~235 MB process RSS
Warmup:                        ~930 ms (first request)
```

Preprocessing equivalence check: manual PIL+numpy pipeline (resize bicubic 224,
rescale 1/255, ImageNet normalize, HWC→CHW) matches transformers
AutoImageProcessor output within float-rounding tolerance (max diff 1.75e-2,
BICUBIC confirmed as processor resample=3).

## Final Model (selected)

- `backend/app/model_assets/model_int8.onnx` (93 MB, int8)
- `backend/app/model_assets/labels.json` (113 classes)
- Runtime deps: onnxruntime, pillow, numpy (no torch/transformers needed in production)

## Files Changed

- `export_onnx.py`, `test_swin_model.py`, `test_onnx_model.py`,
  `check_preprocessing.py` (verification pipeline, later moved to `scripts/`)
- `onnx_model/model_fp32.onnx`, `onnx_model/model_int8.onnx`,
  `onnx_model/labels.json` (local artifacts; fp32 gitignored, int8 copied into backend)

## Problems

- torch 2.14 `torch.onnx.export` default (dynamo) crashed printing a ✅ character
  on Windows cp1252 console → used `dynamo=False` (legacy TorchScript exporter).
- Quantizer warnings for int64 Slice tensors are benign (those ops stay unquantized).

## Fixes

- `PYTHONIOENCODING=utf-8` + `dynamo=False` for export.

---

# Phase 3 — FastAPI Backend

## Status

COMPLETE

## API Endpoints

### GET /health

```json
{"status": "ok", "model_loaded": true, "classes": 113}
```

### POST /api/predict

Input: `multipart/form-data` field `image`

```json
{
  "predicted_class": "Apple Red",
  "confidence": 0.99996,
  "top_predictions": [
    {"class": "Apple Red", "confidence": 0.99996},
    {"class": "Nectarine", "confidence": 7.3e-06}
  ]
}
```

Validation: rejects missing file (422), unsupported format (415),
corrupted/invalid image (400), empty file (400), oversized file
>10 MB (413). Internal errors return generic 500 without details.
CORS enabled for all origins (configurable via `CORS_ORIGINS`).

## Tests

`pytest backend/tests` — **13/13 PASSED**:

- GET /health: status, schema
- POST /api/predict: valid image (Apple Red), banana/orange/
  strawberry/tomato known-fruit checks, top-class==confidence
  consistency, missing image (422), invalid format (415),
  corrupted image (400), empty file (400), oversized file (413)

Live server smoke test (uvicorn :8017):
- `/health` → `{"status":"ok","model_loaded":true,"classes":113}`
- Apple image → `Apple Red` @ 0.99996
- Banana image → `Banana` @ 0.99996

## Problems

- `MODEL_ASSETS_DIR` was computed one level too high
  (`backend/model_assets` instead of `backend/app/model_assets`)
  → model silently not loaded, all endpoints returned 503.
- `__file__` is a plain string; `Path(__file__).parent` needed.
- Port 8000 is occupied by an unrelated pre-existing service on
  this machine; used port 8017 for the local smoke test.

## Fixes

- Fixed `BASE_DIR` in `app/config.py`; wrapped `__file__` in `Path()`.
- Switched smoke-test port to 8017.

---

# Phase 4 — Frontend

## Status

COMPLETE

## Features

- Image upload: browse button + drag & drop + keyboard accessible
- Image preview (object URL, memory-safe)
- Camera capture via `getUserMedia` (button auto-hidden when
  camera is unavailable; capture to canvas → JPEG blob)
- Classify button (disabled until an image is selected)
- Loading indicator (spinner + disabled state)
- Predicted fruit name + confidence percentage + confidence bar
- Top-5 predictions ranked list
- Error handling: non-image file, empty file, HTTP errors with
  server detail, network/backend-unreachable errors
- Responsive mobile-first CSS (breakpoint 480px), focus-visible
  outlines, ARIA roles/labels, no unnecessary animations

## API Configuration

- API URL comes from `window.FRUIT_API_URL`, defined in `config.js`
- `build.js` generates `dist/config.js` from the **`VITE_API_URL`**
  environment variable (default `http://localhost:8000`)
- No backend URL is hard-coded in application logic
- Local dev: `config.js` → `http://localhost:8000`
- Production: set `VITE_API_URL` to the Render URL in Vercel env vars

## Tests

- `node --check app.js` / `build.js`: PASS
- `node build.js`: PASS — generates `dist/` (index.html, styles.css,
  app.js, config.js)
- Env injection: `VITE_API_URL=https://example-render.onrender.com`
  → `dist/config.js` contains the injected URL: PASS
- Static serve test (frontend on :8080 → backend on :8017):
  - index.html / styles.css / app.js / config.js all HTTP 200
  - CORS preflight (OPTIONS from :8080 origin): 200,
    `access-control-allow-origin: *`
  - Multipart POST with strawberry image: `Strawberry` @ 99.99%
- Known-fruit predictions verified through the API:
  Apple Red, Banana, Orange, Strawberry, Tomato (all correct)

## Problems

- Windows `python -m http.server` started via Start-Process died
  silently; replaced with a small Node static server
  (`frontend/serve.js`, dev-only).
- PowerShell 5.1 has no `&&`; chained commands split.

## Fixes

- Added `frontend/serve.js` for local static serving.
- Used `;` command separators in PowerShell.

---

# Phase 5 — Deployment

## Status

NOT STARTED

## GitHub

Repository: (to be created by user)
Git LFS: not required (int8 ONNX model is 93 MB < 100 MB GitHub file limit)

## Render

Backend URL: (to be deployed)
Health URL: (to be deployed)
Status: pending

## Vercel

Frontend URL: (to be deployed)
Status: pending

## End-to-End Test

Pending

---

# Final Model Performance

Accuracy: 100% (30/30 sample verification); model card reports 99.92% on the
Fruits-360 evaluation set
Precision: (per-class, see Phase 2 verification — all 30 sampled classes correct)
Recall: (same)
F1 Score: (same)
Number of classes: 113
Model size: 93 MB (ONNX int8)
Inference time: ~680 ms per image (CPU, ONNX Runtime int8)

---

# Final Project Status

Frontend: PENDING
Backend: PENDING
Model: PASS
Render: PENDING
Vercel: PENDING
End-to-End: PENDING

---

# Known Limitations

- Fruits-360 images are studio shots on plain backgrounds; real-world photos
  (complex backgrounds, lighting, partial occlusion) will be less accurate.
- CPU inference ~0.7 s per request on Render-sized instances.
- Model covers 113 Fruits-360 classes only; other produce (e.g. "Beans",
  "Corn") is in the label set but is not a fruit.

# Future Improvements

- GPU/ONNX acceleration or a smaller backbone (MobileNetV3) for faster inference.
- Confidence calibration and a "low confidence" UX hint.
- Batch inference endpoint.
- Per-class precision/recall report on the full 22,688-image test split.
