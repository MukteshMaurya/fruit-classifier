# Fruit Classification AI

A production-ready web app that classifies fruit from an uploaded
image using an open-source pretrained model.

```text
User
  |
  v
Vercel Frontend (static HTML/JS/CSS)
  |
  | HTTPS multipart request
  v
Render FastAPI Backend
  |
  v
Swin-Base ONNX model (113 fruit classes, int8)
  |
  v
Prediction + confidence
```

## Model

- **Model:** `PedroSampaio/fruits-360-16-7` — Swin-Base fine-tuned
  on the Fruits-360 dataset (Apache-2.0)
- **Classes:** 113 fruit categories
- **Format:** ONNX int8 (93 MB), served with ONNX Runtime
- **Verified accuracy:** 100% on a 30-image sample of the
  Fruits-360 test split (model card: 99.92% eval accuracy)
- **Preprocessing:** resize 224×224 bicubic, rescale 1/255,
  ImageNet normalization

## Project structure

```text
.
├── FRUIT_CLASSIFICATION_PROGRESS.md   # full build log
├── OPENCODE_PROGRESS.md               # fix-session log (rejection, confidence, security)
├── .env.example                       # env var documentation (no secrets)
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app, CORS, lifespan
│   │   ├── config.py                  # env-based settings
│   │   ├── api/routes.py              # GET /health, POST /api/predict
│   │   ├── services/classifier.py     # ONNX inference + preprocessing
│   │   ├── models/schemas.py          # response schemas
│   │   └── model_assets/              # model_int8.onnx + labels.json
│   ├── tests/                         # pytest suite (32 tests)
│   ├── requirements.txt
│   ├── render.yaml                    # Render service definition
│   └── README.md
├── frontend/
│   ├── index.html                     # UI (upload, camera, results)
│   ├── app.js                         # frontend logic
│   ├── styles.css                     # responsive styles
│   ├── config.js                      # dev API URL
│   ├── build.js                       # bakes VITE_API_URL into dist/
│   ├── vercel.json                    # Vercel static build config
│   └── package.json
└── scripts/                           # model verification/export scripts
```

## Local development

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload   # http://localhost:8000
```

### Frontend

```bash
cd frontend
node build.js                   # generates dist/ with localhost API
node serve.js dist 8080         # static server (dev only)
```

Open http://localhost:8080, upload a fruit image, click
**Classify Fruit**.

### Tests

```bash
cd backend
pytest
```

## Deployment

### Backend (Render)

1. Push this repository to GitHub.
2. In Render, create a **Web Service** from the repository.
   - Runtime: Python 3
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/health`
   - (Alternatively import `backend/render.yaml` in Render.)
3. Note the service URL, e.g. `https://fruit-classifier.onrender.com`.

The 93 MB ONNX model is committed to the repo, so no model
download step is required at deploy time.

### Frontend (Vercel)

1. In Vercel, create a project from the `frontend/` directory.
2. Set environment variable **`VITE_API_URL`** to the Render URL
   (e.g. `https://fruit-classifier.onrender.com`).
3. Deploy. Vercel runs `node build.js` and serves `dist/`.

## API

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "classes": 113}
```

### `POST /api/predict`

Multipart form field `image` (jpg/png/webp/bmp/gif, ≤ 10 MB).

Supported fruit image:

```json
{
  "predicted_class": "Apple Red",
  "confidence": 0.99996,
  "top_predictions": [
    {"class": "Apple Red", "confidence": 0.99996},
    {"class": "Nectarine", "confidence": 0.00001}
  ]
}
```

Non-fruit image (confidence and margin below the rejection
thresholds — no misleading fruit prediction is returned):

```json
{
  "predicted_class": null,
  "confidence": null,
  "rejected": true,
  "reason": "non_fruit",
  "message": "Non-fruit image detected. Please upload an image of a supported fruit.",
  "top_predictions": []
}
```

`confidence` is the model's predicted-class probability
(softmax over the 113 fruit classes), shown by the frontend
as a percentage. It is not a calibrated probability that the
prediction is correct.

## Non-fruit rejection

The model is a closed-set classifier: without a rejection
mechanism it assigns *every* image to one of the 113 fruit
classes. The backend therefore rejects an image when the
top-class confidence **or** the top1−top2 margin falls below
`REJECTION_CONFIDENCE` (default 0.45) / `REJECTION_MARGIN`
(default 0.30). The defaults were derived from a validation
set of 565 Fruits-360 test images (fruit min confidence
0.573, min margin 0.358) and 100 non-fruit photos of people,
cars, phones, animals, furniture and buildings (non-fruit max
confidence 0.332, max margin 0.264) — see
`OPENCODE_PROGRESS.md`. Limitation: unsupported fruit varieties
that visually resemble a supported class may still be
classified (and accepted) as that class.

## Environment variables

Backend (Render) — see `backend/README.md` and `.env.example`:
`CORS_ORIGINS`, `MAX_UPLOAD_MB`, `TOP_K`, `MODEL_PATH`,
`LABELS_PATH`, `REJECTION_CONFIDENCE`, `REJECTION_MARGIN`.

Frontend (Vercel): `VITE_API_URL` — the public Render backend
URL baked into the static bundle at build time. A backend URL
is **not** a secret (the prediction API is unauthenticated),
but never put private keys or tokens in a `VITE_` variable:
they would ship to every browser. No credentials are required
by the frontend; the deploy helper scripts read Vercel/GitHub
tokens from the local Vercel CLI / git credential manager at
runtime and none are committed to the repository.

## Security notes

- No API keys, tokens, or passwords are present in the
  frontend source, the generated `dist/` build output, or the
  git history (scanned for common credential formats).
- The only value exposed to the browser is the public backend
  URL (`window.FRUIT_API_URL`), which is a frontend
  configuration value, not a secret.
- No credential rotation is required: no secret was ever
  published.

## License

- Application code: MIT
- Model: Apache-2.0 (`PedroSampaio/fruits-360-16-7`)
- Dataset: MIT (Fruits-360)
