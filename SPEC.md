# Build spec — ID card field extraction MVP (Qwen2.5-VL on Colab)

## What we're building

A demo tool for a client meeting: upload or capture a photo of an ID card, get the card's fields back as structured JSON, with a review screen where a human confirms each field.

**Deployment shape:** the FastAPI backend runs inside a Google Colab notebook on a T4 GPU, exposed over a Cloudflare tunnel. I open the tunnel URL in my browser and the frontend talks to that backend. My own laptop has a GTX 1650 (4GB, no tensor cores), which is why inference is not running locally.

This is an MVP for a demo, using my own ID cards as test data. It is not handling real customer data. But build it as the foundation of a real product — clean structure, no throwaway hacks — because the same code will later run on a server inside a bank with the tunnel removed.

## Starting point

I have a working browser-only demo at `ocr-demo.html` in the repo root. It has drag-drop image upload, live camera capture, a canvas overlay drawing boxes over detected text coloured by confidence, a right-hand field panel, and hover-to-highlight linking a field back to its position on the image.

Keep that interface and its visual design. Replace only the OCR engine underneath: remove Tesseract.js, call our backend instead.

## Model

`Qwen/Qwen2.5-VL-3B-Instruct` in fp16 on the T4 (16GB is plenty — no quantisation needed, and skipping it avoids a quality loss).

Make the model ID a config value. I want to A/B against the 7B variant later without code changes.

Load the model once at startup and keep it warm. Never load per request.

**Cache the weights to Google Drive.** A fresh Colab session re-downloads ~7GB otherwise, which is several minutes of dead air before I can demo. Mount Drive, set `HF_HOME` to a Drive path, and document this in the notebook.

## Repo structure

Claude Code writes a normal repo on my laptop. I push to GitHub; the notebook clones and runs it. Nothing here should assume it's running in a notebook except the notebook itself.

```
app/
  main.py            FastAPI app, CORS, static file serving
  model.py           model load, warm-up, single inference entry point
  extract.py         prompt construction, JSON parsing, retry on malformed output
  validate.py        field rules and cross-checks
  schema.py          pydantic response models
  config.py          env-driven settings
static/
  index.html         the adapted demo
eval/
  run_eval.py        batch scoring against ground truth
  samples/           test images + expected.json
notebook/
  run_colab.ipynb    clone, install, mount Drive, start server, open tunnel
requirements.txt
README.md
```

## The Colab notebook

Write `notebook/run_colab.ipynb` as a small number of clearly-labelled cells:

1. Check GPU (`nvidia-smi`) and fail loudly with a clear message if no GPU is attached
2. Mount Google Drive, set `HF_HOME` to the Drive cache path
3. Clone the repo (parameterise the URL) and install requirements
4. Start uvicorn in a background thread
5. Start a Cloudflare quick tunnel and **print the HTTPS URL in large, obvious output** — this is the URL I open during the demo
6. A cell that sends a test image through the API and prints timing, so I can confirm everything works before I share my screen

Use a Cloudflare quick tunnel rather than ngrok — no account or auth token needed, one less thing to fail live.

Note in the README that the camera needs HTTPS, which the tunnel provides.

## The extraction endpoint

`POST /api/extract` — multipart image upload. Returns:

```json
{
  "fields": {
    "full_name":     {"value": "JOHN A SMITH", "status": "ok"},
    "id_number":     {"value": "12345678",     "status": "ok"},
    "date_of_birth": {"value": "1990-04-12",   "status": "review", "reason": "day/month order ambiguous"},
    "expiry_date":   {"value": "2030-04-11",   "status": "ok"},
    "nationality":   {"value": "OMANI",        "status": "ok"},
    "sex":           {"value": "M",            "status": "ok"}
  },
  "raw_text": "...",
  "elapsed_ms": 1840,
  "model": "Qwen2.5-VL-3B-Instruct"
}
```

Status is `ok` / `review` / `missing`. Never invent a confidence percentage — see below.

Also add `GET /api/health` returning model name, device, and whether the model is loaded. I want to check this before demoing.

## Reliable structured output

Prompt the model to return JSON only: explicit field list, dates normalised to ISO, missing fields as null, no commentary, no markdown fences.

Then:
1. Strip fences and parse
2. Validate against a pydantic schema
3. On failure, retry once with the error fed back in
4. On a second failure, return raw text with all fields `missing` rather than guessing

Low temperature. Cap `max_new_tokens` — the output is short and an unbounded cap wastes seconds.

**The model must not fill in plausible values for fields it cannot read.** Instruct it explicitly to return null for anything illegible. A confidently wrong ID number is the worst failure mode in this product — far more damaging than a blank field.

## Confidence — read carefully

A VLM does not expose per-character confidence like classical OCR does. It states answers fluently whether right or wrong. Do not derive a percentage from token probabilities and label it OCR confidence — it doesn't mean what a reviewer would assume.

Derive `status` from checks that are actually meaningful:

- **Format rules** — ID number length and character class, dates parse and are plausible, expiry after date of birth, sex and nationality from closed lists
- **Self-consistency** — run inference twice with different image preprocessing (original, and contrast-normalised). Fields that agree get `ok`; fields that disagree get `review`. Strongest available signal and cheap
- **Nulls** become `missing`

Document this reasoning in code comments. If you have a better approach, propose it first — don't silently substitute a token-probability number.

## Frontend changes

- Replace the Tesseract call with `fetch` to `/api/extract`; make the API base URL configurable, since the tunnel URL changes each session
- Keep the colour language, driven by `status`: ok / review / missing
- Camera: keep capture-and-read. **Remove continuous scanning** — inference takes seconds and requests would queue and lag
- Show real progress during inference; a multi-second freeze with no feedback looks broken
- Keep hover-to-highlight if the model returns usable bounding boxes (Qwen2.5-VL supports grounded output). If boxes prove unreliable, drop the overlay rather than draw wrong boxes — and tell me, don't silently disable it
- Handle backend-unreachable with a clear message naming the likely cause (tunnel down or session expired), not a generic error

## Evaluation harness

Build `eval/run_eval.py` early, not last:

- Reads images from `eval/samples/` with a matching `expected.json`
- Runs each through the pipeline
- Reports per-field exact-match accuracy, plus `review` and `missing` counts
- **Reports silent error rate separately: fields marked `ok` that are actually wrong.** This is the number I care about most
- Prints a summary table

Generate a few synthetic test cards so the harness runs before I supply real photos.

## Data handling

Even though this is my own data on Colab, build the habits now:

- Images held in memory, never written to disk
- No image bytes or field values in logs — event, timing, and status counts only
- `DEBUG_SAVE_IMAGES` config flag, default off, gitignored output, marked development-only
- The tunnel is demo infrastructure. Make it trivial to remove for a local deployment — the app itself should know nothing about it

## How I want you to work

- Read `ocr-demo.html` first so you preserve its structure and design
- Set up repo, dependencies, README with exact commands
- Get model loading and a single hardcoded image working end to end first. Report first-inference time, warm inference time, and VRAM used on a T4
- Then the notebook, then frontend, then validation, then eval
- Ask before adding any dependency that isn't obviously required
- No auth, database, Docker, or multi-user support — out of scope, but don't architect in a way that makes them painful later

## Acceptance criteria

1. The notebook runs top to bottom and prints a working HTTPS tunnel URL
2. Opening that URL serves the demo page
3. Uploading a card photo returns schema-conforming JSON in under 5 seconds warm
4. Camera capture works through the tunnel
5. Illegible fields come back `missing`, not invented
6. `run_eval.py` prints the accuracy table including silent error rate
7. `/api/health` confirms model and device before I demo

## Where I expect trouble — flag these, don't work around them quietly

- Grounded bounding box reliability from Qwen2.5-VL
- Colab session timeouts mid-demo, and how fast recovery is
- Model download time on a cold session even with the Drive cache
- The 3B model being too weak on small print — if so, tell me what 7B costs in VRAM and latency on a T4