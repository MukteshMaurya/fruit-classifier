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
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app, CORS, lifespan
│   │   ├── config.py                  # env-based settings
│   │   ├── api/routes.py              # GET /health, POST /api/predict
│   │   ├── services/classifier.py     # ONNX inference + preprocessing
│   │   ├── models/schemas.py          # response schemas
│   │   └── model_assets/              # model_int8.onnx + labels.json
│   ├── tests/                         # pytest suite (13 tests)
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

Multipart form field `image` (jpg/png/webp/bmp/gif, ≤ 10 MB):

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

## License

- Application code: MIT
- Model: Apache-2.0 (`PedroSampaio/fruits-360-16-7`)
- Dataset: MIT (Fruits-360)
