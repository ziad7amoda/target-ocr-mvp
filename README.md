# Omani ID card field extraction

Upload or capture a photo of an ID card, get its fields back as structured JSON
with a per-field `ok` / `review` / `missing` status for human review.

Backend: FastAPI + Qwen2.5-VL. Frontend: a single static page.

- **Design:** `docs/superpowers/specs/2026-08-24-id-card-extraction-design.md`
- **Plan:** `docs/superpowers/plans/2026-08-24-id-card-extraction.md`

## Local development (no GPU required)

The whole suite runs without torch. `app/model.py`'s `QwenEngine` imports torch
lazily, so everything else is testable on any machine.

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -v
```

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

## Colab

See `notebook/run_colab.ipynb`. Camera capture requires HTTPS, which the
Cloudflare tunnel provides.
