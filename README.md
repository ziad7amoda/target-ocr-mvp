# Omani ID card field extraction

Upload or capture a photo of an ID card, get its fields back as structured JSON
with a per-field `ok` / `review` / `missing` status for human review.

Backend: FastAPI + Qwen2.5-VL. Frontend: a single static page.

- **Design:** `docs/superpowers/specs/2026-08-24-id-card-extraction-design.md`
- **Plan:** `docs/superpowers/plans/2026-08-24-id-card-extraction.md`

## Local development (no GPU required)

The whole suite runs without torch: `requirements.txt` (and `requirements-dev.txt`,
which layers on it) installs no torch-requiring package. `app/model.py`'s
`QwenEngine` is the only thing that needs torch, transformers, or accelerate,
and it imports them lazily, so everything else is testable on any machine.

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -v
```

## Colab / GPU deployment

Installing just `requirements.txt` is not enough to run the model: it
deliberately excludes the GPU-only inference stack (transformers, accelerate),
which lives in `requirements-gpu.txt` because `accelerate` has a hard
dependency on torch. Install both:

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

Then open http://localhost:8000/.

**Without a GPU and `DEVICE=auto` (the default), startup fails fast** with a
`RuntimeError` raised at the top of `QwenEngine.load()`, before any
`from_pretrained` call: no multi-gigabyte download happens, and no slow CPU
load is attempted by accident. The error message explains the situation and
how to opt in anyway. If you deliberately want to run inference on CPU
(functional, but extremely slow - think minutes per request), set
`DEVICE=cpu` explicitly and the load will proceed.

**Not every HuggingFace repo naming convention works with `MODEL_ID`.** This
engine loads models through transformers' `AutoModelForImageTextToText`,
which needs a HuggingFace `config.json`. `QwenEngine.load()` checks the repo
name for known incompatible suffixes before downloading anything, and fails
fast with a `RuntimeError` explaining the problem and the fix:

| Repo suffix | Works here? | Why |
|---|---|---|
| *(plain, e.g. `Qwen/Qwen2.5-VL-3B-Instruct`)* | Yes | Standard transformers format |
| `-GGUF` | No | llama.cpp's quantised format - no `config.json`, needs a llama.cpp/ollama runtime instead. Drop the suffix, e.g. `Qwen/Qwen3-VL-2B-Instruct-GGUF` -> `Qwen/Qwen3-VL-2B-Instruct` |
| `-MLX` | No | Apple's on-device format for Apple Silicon; only runs via Apple's MLX runtime, not on a CUDA GPU. Drop the suffix |
| `-AWQ` | Only with `autoawq` installed | transformers can load AWQ, but the package isn't in `requirements.txt`/`requirements-gpu.txt` (torch-free by design). `pip install autoawq` first |
| `-GPTQ` | Only with `optimum` + `gptqmodel` installed | Same story: `pip install optimum gptqmodel` first |

If you hit a raw `ValueError: ... Should have a \`model_type\` key in its
config.json` despite a plain repo name, `load()` also wraps that error with
guidance: check whether the repo is gated/private (set `HF_TOKEN`), or
whether the installed transformers version is too old for that architecture.

## Configuration

All settings are environment variables; see `app/config.py`. The ones that matter:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | Swap to the 7B variant without code changes |
| `SHOW_BOXES` | `true` | Kill switch for the bounding-box overlay |
| `SELF_CONSISTENCY` | `true` | **Debugging only.** Disabling removes hallucination detection |
| `DEBUG_SAVE_IMAGES` | `false` | **Development only.** Writes card images to disk |
| `ALLOWED_ORIGINS` | `["*"]` | Wide open for the demo. Narrow this for any real deployment |

## Current status

**Complete and tested (no GPU needed):** the full pipeline against
`FakeEngine` - schema, config, image contrast normalisation, JSON fence
stripping/repair, field parsing, format rules and cross-field checks, status
derivation, the box geometry filter and coordinate rescaling, the
`/api/extract` and `/api/health` API contracts, retry-then-missing behaviour,
and the GPU-less startup guard added in this task (`QwenEngine.load()` fails
fast with no download when no GPU is present and `DEVICE=auto`). 131 tests
pass, 1 skipped, all runnable and verified on this machine.

**Awaiting the user's hardware** - not executable in this environment and
therefore not measured or invented:

- **A Colab T4 run** (`scripts/bringup.py` and `notebook/run_colab.ipynb`):
  cold weight load, warm-up time, warm single vs. batch-of-3 latency (the D5
  batching assumption), peak VRAM, and confirming the notebook prints a
  working HTTPS tunnel URL end to end.
- **The Noto Naskh Arabic font** for `eval/make_samples.py`'s synthetic
  cards, needed to exercise the Arabic-text rendering path before real
  photos exist.
- **Real ID card photos** for `eval/samples/` with hand-typed
  `expected.json` ground truth, needed to run `eval/run_eval.py` and produce
  the accuracy table, silent error rate, and box hit rate that decide the
  `SHOW_BOXES` verdict (spec D4).

See `docs/measurements.md` for the template these numbers land in once
produced, and its acceptance-criteria table for the current pass/pending
status of each of spec §14's eight criteria.
