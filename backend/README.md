# Fruit Classification API

FastAPI backend for the Fruit Classification AI web app. Classifies fruit
images with a pretrained Swin-Base model (113 Fruits-360 classes) served
through ONNX Runtime.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs

## Endpoints

- `GET /health` — service and model status
- `POST /api/predict` — multipart `image` upload, returns prediction

## Configuration

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `app/model_assets/model_int8.onnx` | ONNX model file |
| `LABELS_PATH` | `app/model_assets/labels.json` | Class label map |
| `MAX_UPLOAD_MB` | `10` | Maximum upload size |
| `TOP_K` | `5` | Number of top predictions returned |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `REJECTION_CONFIDENCE` | `0.45` | Minimum top-class confidence to accept an image as a fruit |
| `REJECTION_MARGIN` | `0.30` | Minimum top1−top2 probability gap to accept an image as a fruit |

Images that fail either rejection threshold receive the
non-fruit response (`rejected: true`) instead of a fruit
prediction. Defaults are derived from a validation set of
Fruits-360 test images and non-fruit photos — see
`OPENCODE_PROGRESS.md` in the repository root.

## Deployment (Render)

See `render.yaml`. Build runs `pip install -r requirements.txt`; the
service starts with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
and exposes `GET /health` as the health check.

The ONNX int8 model (93 MB) is committed to the repository, so no
download step is needed at deploy time.

## Tests

```bash
pytest
```
