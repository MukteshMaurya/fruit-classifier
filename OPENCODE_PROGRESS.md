# OPENCODE Progress — Fruit Classifier Fixes

This log tracks the three-fix session (non-fruit rejection,
confidence correctness, API key security). The full build
history is in `FRUIT_CLASSIFICATION_PROGRESS.md`.

---

# Stage 1 — Audit (COMPLETE)

## Initial project structure

```text
.
├── backend/                     # FastAPI app (Render)
│   ├── app/
│   │   ├── main.py              # lifespan loads FruitClassifier, CORS *
│   │   ├── config.py            # env-based Settings
│   │   ├── api/routes.py        # GET /health, POST /api/predict
│   │   ├── services/classifier.py   # ONNX int8 inference + preprocessing
│   │   ├── models/schemas.py    # (defined but unused by routes)
│   │   └── model_assets/        # model_int8.onnx (93 MB) + labels.json
│   ├── tests/                   # pytest: 13 tests, all passing
│   ├── render.yaml
│   └── requirements.txt
├── frontend/                    # static site (Vercel, build: node build.js)
│   ├── index.html, app.js, styles.css, config.js
│   ├── build.js                 # bakes VITE_API_URL into dist/config.js
│   ├── vercel.json, package.json
│   └── dist/                    # generated (gitignored)
├── onnx_model/                  # local fp32/int8 artifacts (gitignored)
├── scripts/                     # model verification + deploy helper scripts
└── validation_data/             # (created this session, local only)
```

## Baseline test run

- `pytest backend/tests` → **13/13 passed** (54 s).
- Model stable across repeated runs (deterministic).

## Root causes discovered

### Problem 1 — Non-fruit images classified as fruits

**Root cause:** the Swin model is a *closed-set* classifier over
113 fruit classes. Its softmax always sums to 1, so *any* input is
forced into one of the 113 fruit classes. There is no rejection /
out-of-distribution mechanism anywhere in `classifier.py` or
`routes.py`.

**Evidence (measured this session):**

- Validation set built for threshold tuning:
  - **Fruit positives:** 565 images (5 per class, all 113 classes)
    sampled from the cached Fruits-360 test split
    (`~/.cache/huggingface/hub/datasets--PedroSampaio--fruits-360/...`,
    no new download).
  - **Non-fruit negatives:** 100 real photos (people, cars, mobile
    phones, dogs, cats, chairs, buildings, bicycles, airplanes,
    horses — 10 per category) fetched as 256 px thumbnails from the
    Wikimedia Commons API (public-domain / CC-licensed).
- Results with the production int8 model:

| Set | confidence min / median / max | margin (top1−top2) min / max |
|---|---|---|
| Fruit (565) | 0.5728 / 0.9999 / 1.0000 | 0.3582 / 0.9999 |
| Non-fruit (100) | 0.0230 / 0.0464 / **0.3316** | 0.0003 / **0.2639** |

- **Clean separation:** the *lowest* fruit confidence (0.5728) is
  well above the *highest* non-fruit confidence (0.3316); same for
  margin (0.3582 vs 0.2639). Every non-fruit image currently gets a
  fruit label (e.g. mobile phone → "Pepper Orange" 0.33).

### Problem 2 — "Low confidence" on correct predictions

**Root cause investigation (all checks passed — no math bug):**

- Model ONNX output is `logits` [batch, 113] float32 (verified from
  the graph). `classifier.py` applies softmax correctly
  (`_softmax`: shifted exp / sum). No sigmoid misuse, no
  logits-as-probabilities bug.
- Preprocessing matches training (RGB → bicubic resize 224×224 →
  /255 → ImageNet mean/std → CHW → batch) — previously verified
  against the HF `AutoImageProcessor` (max diff 1.75e-2).
- `labels.json` (all 113 entries) matches the HF model config
  `id2label` exactly — no label-mapping mismatch.
- Backend loads `backend/app/model_assets/model_int8.onnx`
  (correct path, verified by 13 passing tests + live smoke test).
- Frontend display is correct: `(confidence * 100).toFixed(1) + "%"`.
  Backend returns a float in [0,1]; no rounding/conversion bug.
- Inference is deterministic (5 identical runs).

**Actual findings:**

1. The displayed confidence is the model's predicted-class
   probability, not a calibrated P(correct). On the 565-image fruit
   validation set the model is already well calibrated (99.9%
   accurate, median confidence 0.9999) — no calibration needed.
2. int8 dynamic quantization slightly lowers confidence on the hardest
   images (min 0.57 int8 vs 0.68 fp32; mean 0.9944 vs 0.9955).
   This is honest quantization noise, not a bug. Switching to the
   352 MB fp32 model is impractical (GitHub 100 MB file limit).
3. Real-world photos (complex backgrounds) get lower confidence than
   studio shots — inherent distribution shift; the softmax honestly
   reflects it (e.g. real apple photo → "Apple Red Yellow" 0.84;
   hard ones drop to 0.2–0.5 and are usually wrong).
4. The genuinely misleading case was Problem 1: non-fruit images
   displayed a fruit name with a low confidence — fixed by the
   rejection mechanism (Stage 2), which now shows the rejection
   message instead.

**Conclusion:** no confidence-inflation or formula fix is warranted.
The fix is honest display + rejection of non-fruit inputs +
regression tests. Confidence values are never manipulated.

### Problem 3 — API key / sensitive configuration exposure

**What is actually exposed in the browser:**

- `window.FRUIT_API_URL` in `config.js` / `dist/config.js` — the
  **Render backend URL**. This is a *public frontend configuration
  value*, **not a secret**: the prediction API is unauthenticated
  (CORS `*`, no API key required), and a backend URL alone grants
  no access to anything.
- `frontend/.vercel/project.json` (local only, gitignored) — Vercel
  `projectId`/`orgId`. Public identifiers, not credentials.
- `scripts/*.ps1` (committed) hardcode the Vercel **team ID** and
  **project ID** — public identifiers. The actual Vercel API token is
  read at runtime from the local Vercel CLI `auth.json` and the
  GitHub PAT from the git credential manager; **neither token is or
  was committed**.
- Secret scan across **all git revisions** for common credential
  patterns (`sk-…`, `ghp_…`, `gho_…`, `hf_…`, `AKIA…`) → **no
  matches**. No `.env` files exist in the repo.

**Conclusion:** no real secret is exposed in frontend source, build
output, or git history. No credential rotation is required. The
Vercel project env vars could not be audited remotely (local CLI
token returns "Not authorized" — expired/revoked; documented as a
limitation).

## Files changed in this session

- `backend/app/config.py` — add `REJECTION_CONFIDENCE` /
  `REJECTION_MARGIN` settings (env-configurable, defaults
  0.45 / 0.30)
- `backend/app/services/classifier.py` — add
  `predict_detailed()` returning top class, confidence,
  runner-up and margin; `predict()` now delegates to it
  (same return value as before)
- `backend/app/api/routes.py` — `/api/predict` rejects
  images failing the confidence/margin thresholds with the
  non-fruit message; success-path response unchanged
- `frontend/app.js` — `renderResult()` shows the rejection
  message in the existing error box when `data.rejected`
  is set (only frontend change; no styling/layout change)
- `backend/tests/samples/nonfruit/` — 8 non-fruit JPEG
  regression fixtures (person, car, mobile_phone, dog, cat,
  chair, building, bicycle; 256 px Wikimedia Commons
  thumbnails, public-domain / CC-licensed)
- `backend/tests/test_rejection.py` — 19 regression tests
  (rejection of 8 non-fruit categories, non-rejection of 5
  known fruits, threshold separation, margin sanity,
  model-not-loaded)
- `backend/tests/test_security.py` — scans frontend source
  AND generated `dist/` for credential patterns; asserts
  `build.js` only reads `VITE_API_URL`
- `.env.example`, `frontend/.env.example` — env var docs
  (placeholders only, no credentials)
- `backend/render.yaml` — document the two new env vars
- `README.md`, `backend/README.md` — rejection behavior,
  confidence semantics, env var and security documentation
- `.gitignore` — ignore local `validation_data/`
- `scripts/` — analysis tooling: `collect_validation_data.py`
  (builds validation sets), `analyze_separation.py` (threshold
  analysis, writes `separation_analysis.json`),
  `probe_realworld_fruit.py`, `probe_unsupported_fruit.py`,
  `compare_int8_fp32.py` (slow, >15 min on full set),
  `audit_vercel_env.ps1` (remote env audit — could not
  authorize), `investigate_confidence.py`
- `OPENCODE_PROGRESS.md` — this file

---

# Stage 2 — Non-fruit detection (COMPLETE)

## Solution

Rejection in `backend/app/api/routes.py` using the two
signals the task prefers (calibrated confidence + prediction
margin), with thresholds derived empirically from the
validation set:

- Reject when `confidence < 0.45` **OR** `margin < 0.30`.
- Validation basis: fruit min confidence 0.5728 / min margin
  0.3582 (565 Fruits-360 test images); non-fruit max
  confidence 0.3316 / max margin 0.2639 (100 Commons photos
  of people, cars, phones, animals, furniture, buildings,
  bicycles, airplanes, horses). Both thresholds sit midway
  in the empty gap between the two distributions.
- Not confidence-only: the margin check additionally rejects
  images the model is torn about even if confidence is
  moderate.
- Thresholds are env-configurable (`REJECTION_CONFIDENCE`,
  `REJECTION_MARGIN`) so they can be tuned on Render without
  a code change.
- No new model, no training, no dataset download required —
  the existing Swin classifier is preserved unchanged.

## Response contract

- Accepted (unchanged format):
  `{"predicted_class": str, "confidence": float, "top_predictions": [...]}`
- Rejected (new, HTTP 200):
  `{"predicted_class": null, "confidence": null, "rejected": true, "reason": "non_fruit", "message": "Non-fruit image detected. Please upload an image of a supported fruit.", "top_predictions": []}`
- Frontend shows the message in the existing error box;
  no fruit name or confidence is displayed for rejections.

## Live E2E verification (uvicorn :8017)

- `GET /health` → `{"status":"ok","model_loaded":true,"classes":113}`
- apple_red.png → `Apple Red` @ 0.99996 (unchanged)
- nonfruit/car.jpg, person.jpg → rejection payload with the
  exact required message
- CORS preflight from a foreign origin → 200,
  `access-control-allow-origin: *` (frontend↔backend
  communication intact)

---

# Stage 3 — Confidence (COMPLETE — verified correct, no fix warranted)

## Investigation results (all passed)

- ONNX graph output is `logits` [batch,113] float32; the
  service applies softmax correctly. No logits-as-probability
  or sigmoid/softmax mix-up.
- Preprocessing (RGB, bicubic 224×224, /255, ImageNet
  mean/std, CHW, batch dim) matches the training pipeline
  (verified against HF `AutoImageProcessor`, max diff 1.75e-2).
- `labels.json` matches the HF model `id2label` for all 113
  classes — no label-index mapping error.
- Backend loads `backend/app/model_assets/model_int8.onnx`
  (correct path).
- Frontend display formula `(confidence * 100).toFixed(1) + "%"`
  verified in Node against sample values (0.924 → "92.4%",
  0.999964 → "100.0%"). No rounding/conversion bug.
- Inference deterministic across repeated runs.
- int8 quantization lowers confidence slightly on the hardest
  images (min 0.57 vs 0.68 fp32; mean 0.9944 vs 0.9955) —
  honest quantization noise, not a bug; switching to the
  352 MB fp32 model is impractical (GitHub 100 MB limit).

## Actions taken

- No confidence manipulation (no inflation, no calibration
  rescaling — the model is already well calibrated on the
  fruit validation set: 99.9% accurate at median confidence
  0.9999).
- Regression tests assert confidence is finite, in [0,1],
  equals `top_predictions[0].confidence`, and that top-k is
  sorted descending.
- The misleading "fruit name + low confidence" display for
  non-fruit images is eliminated by the Stage 2 rejection.
- Documented that `confidence` is the predicted-class
  probability, not a calibrated P(correct).

---

# Stage 4 — Security (COMPLETE — no real secret found)

## Findings

- The value exposed in the browser is `window.FRUIT_API_URL`
  (the Render backend URL) in `config.js` / `dist/config.js`.
  It is a **public frontend configuration value, not a
  secret** — the prediction API is unauthenticated and a
  backend URL grants no access.
- No private keys/tokens in frontend source, `dist/` build
  output, or any git revision (scanned for `sk-…`, `ghp_…`,
  `gho_…`, `ghu_…`, `hf_…`, `AKIA…`, JWT patterns — zero
  matches). No `.env` files exist.
- `scripts/*.ps1` contain only Vercel **team/project IDs**
  (public identifiers); the Vercel token is read at runtime
  from the local Vercel CLI `auth.json` and the GitHub PAT
  from the git credential manager — never committed.
- **No credential rotation is required** because no secret
  was ever published.

## Actions taken

- `.env.example` + `frontend/.env.example` document every
  variable with placeholders and an explicit warning never
  to put secrets in `VITE_` variables.
- `backend/tests/test_security.py` fails the build if a
  credential pattern ever appears in frontend source or
  generated `dist/`, and asserts `build.js` only reads
  `VITE_API_URL`.
- `backend/render.yaml` updated with the new (non-secret)
  env vars.
- The backend URL remains configurable for the Vercel
  frontend via `VITE_API_URL` (plus the existing in-app
  "API settings" override) — unchanged.

## Limitation

- Remote audit of the Vercel project's environment variables
  could not be completed: the locally stored Vercel CLI
  token returns "Not authorized" for the team (expired or
  revoked). Re-run `scripts/audit_vercel_env.ps1` with a
  fresh token to verify remotely.

---

# Stage 5 — Regression testing & deployment readiness (COMPLETE)

## Test results (actually run)

### Test A — valid supported fruit images
- `pytest backend/tests` → **32/32 passed** (13 original +
  19 new).
- Known-fruit tests: Apple Red, Banana, Orange, Strawberry,
  Tomato all predicted correctly, confidence in [0,1],
  displayed as percentages (formula verified in Node).
- Challenging images: 565-image Fruits-360 validation set —
  100% accepted (min confidence 0.5728, min margin 0.3582,
  both above thresholds). Real-world Commons fruit photos
  that the model gets right (e.g. banana 0.977, strawberry
  0.88, apple 0.84) are also above thresholds.

### Test B — non-fruit images
- 8 categories committed as fixtures (person, car,
  mobile_phone, dog, cat, chair, building, bicycle) — all
  rejected with the exact required message (8/8 parametrized
  tests pass).
- Full 100-image validation set (10 categories): 100%
  rejected (max confidence 0.3316 < 0.45).

### Test C — unsupported fruit images
- Probed blackberry, jackfruit, gooseberry (not in the 113
  classes) via Commons photos: 15/15 rejected (confidence
  0.03–0.32). Limitation: an unsupported variety that
  closely resembles a supported class in a studio-style shot
  could still be accepted as that class — the mechanism
  rejects out-of-distribution images, not taxonomically
  unknown fruits.

### Test D — confidence correctness
- Backend: softmax over logits, finite, in [0,1], top-class
  consistency asserted by tests; deterministic.
- Frontend: display formula verified against sample values;
  NaN/Infinity cannot occur (backend guarantees finite
  floats; tests assert `math.isfinite`).
- No valid fruit prediction is rejected by the thresholds
  (fruit minimums 0.5728/0.3582 vs thresholds 0.45/0.30).

### Test E — API security
- Secret scan of frontend source + `dist/`: clean.
- `build.js` only reads `VITE_API_URL`.
- CORS preflight from a foreign origin: 200 with
  `access-control-allow-origin: *`.
- Normal prediction requests verified live (E2E above).
- Remote Vercel env audit: blocked (token not authorized) —
  see Stage 4 limitation.

### Test F — deployment compatibility
- Backend: full pytest suite passes; uvicorn serves
  `/health` and `/api/predict` correctly.
- Frontend: `node --check app.js` / `build.js` pass;
  `node build.js` regenerates `dist/` (index.html,
  styles.css, app.js, config.js) with the rejection handling
  included.
- `render.yaml` valid; new env vars documented with defaults.
- Render health/prediction endpoints could not be checked
  remotely — the Render backend is not yet deployed (no
  Render credentials on this machine; deployment remains a
  manual user step, unchanged from before).

## Remaining limitations

1. Unsupported fruit varieties visually similar to a
   supported class may be accepted as that class (Test C).
2. Heavily blurred/dark fruit photos could fall below the
   0.45 confidence threshold and be rejected; thresholds are
   env-tunable if that occurs in practice.
3. Vercel env-var remote audit blocked by an expired local
   token.
4. Render deployment itself is still pending (manual user
   action, as before this session).
5. `scripts/compare_int8_fp32.py` is slow (>15 min on the
   full validation set) — aggregate int8/fp32 statistics are
   recorded above instead.

## Exact deployment steps

### Render (backend)
1. Push this repository to GitHub (the 93 MB int8 model is
   committed; do not move it).
2. In Render, create a Web Service from the repo (or import
   `backend/render.yaml`) with root directory `backend`:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/health`
   - Env vars (all optional, defaults shown):
     `CORS_ORIGINS=*`, `MAX_UPLOAD_MB=10`, `TOP_K=5`,
     `REJECTION_CONFIDENCE=0.45`, `REJECTION_MARGIN=0.30`
     (optionally `MODEL_PATH`, `LABELS_PATH`)
3. Verify `https://<service>.onrender.com/health` returns
   `{"status":"ok","model_loaded":true,"classes":113}`.

### Vercel (frontend)
1. Project root directory: `frontend` (already configured).
2. Env var `VITE_API_URL` = `https://<service>.onrender.com`
   (public backend URL — not a secret; never add secrets to
   `VITE_` variables).
3. Redeploy. `node build.js` bakes the URL into `dist/config.js`.
4. After deploying, users with a cached old `app.js` may need
   one hard refresh (Ctrl+F5) to pick up the rejection-message
   handling.
5. Optionally re-run `scripts/audit_vercel_env.ps1` with a
   fresh Vercel token to confirm no secret env vars exist.

## Manual actions still required

- Deploy the backend to Render (unchanged from before this
  session; no Render credentials exist on this machine).
- Set `VITE_API_URL` on Vercel to the Render URL and redeploy.
- No model training, dataset download, or credential rotation
  is required.
