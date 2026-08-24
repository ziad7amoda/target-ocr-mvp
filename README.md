# Omani ID card field extraction

Upload or capture a photo of an ID card, get its fields back as structured JSON
with a per-field `ok` / `review` / `missing` status for human review.

Backend: FastAPI + Qwen2.5-VL. Frontend: a single static page.

- **Design:** `docs/superpowers/specs/2026-08-24-id-card-extraction-design.md`
- **Plan:** `docs/superpowers/plans/2026-08-24-id-card-extraction.md`

## Local development (no GPU required)

The whole suite runs without torch: `requirements.txt` (and `requirements-dev.txt`,
which layers on it) installs no torch-requiring package. `app/model.py`'s
`QwenEngine` is the only thing that needs torch, transformers, accelerate, or
qwen-vl-utils, and it imports them lazily, so everything else is testable on
any machine.

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -v
```

## Colab / GPU deployment

Installing just `requirements.txt` is not enough to run the model: it
deliberately excludes the GPU-only inference stack (transformers, accelerate,
qwen-vl-utils), which lives in `requirements-gpu.txt` because `accelerate`
has a hard dependency on torch. Install both:

```bash
pip install -r requirements.txt -r requirements-gpu.txt
```

torch itself is not listed in either file - Colab preinstalls a build matched
to its CUDA driver, and `notebook/run_colab.ipynb` (which does this install
for you) relies on that. Camera capture requires HTTPS, which the notebook's
Cloudflare tunnel provides.

## Running the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/. Without a GPU the model will not load and
`/api/health` reports `"loaded": false`.

## Configuration

All settings are environment variables; see `app/config.py`. The ones that matter:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | Swap to the 7B variant without code changes |
| `SHOW_BOXES` | `true` | Kill switch for the bounding-box overlay |
| `SELF_CONSISTENCY` | `true` | **Debugging only.** Disabling removes hallucination detection |
| `DEBUG_SAVE_IMAGES` | `false` | **Development only.** Writes card images to disk |
| `ALLOWED_ORIGINS` | `["*"]` | Wide open for the demo. Narrow this for any real deployment |
