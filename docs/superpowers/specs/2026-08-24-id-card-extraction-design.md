# ID card field extraction — design

**Date:** 2026-08-24
**Status:** approved for implementation planning
**Source requirements:** `SPEC.md` (repo root)
**Starting point:** `ocr-demo.html` (repo root)

---

## 1. Purpose

Upload or capture a photo of an Omani ID card and get its fields back as structured
JSON, with a review screen where a human confirms each field.

The backend runs `Qwen/Qwen2.5-VL-3B-Instruct` inside a Google Colab notebook on a
T4, exposed over a Cloudflare quick tunnel. The tunnel URL is opened in a browser and
serves both the API and the demo page.

This is an MVP for a client meeting, tested on the author's own ID cards. It is not
handling real customer data. It is built as the foundation of a real product: the same
code is expected to run on a bank's own server with the tunnel removed, so nothing
outside `notebook/` may assume it is running in a notebook.

**Out of scope:** auth, database, Docker, multi-user. The design must not make these
painful to add later, but none of them get built.

---

## 2. Decisions

These were open in `SPEC.md` and are now settled. Each is load-bearing for the
sections that follow.

| # | Decision | Rationale |
|---|---|---|
| D1 | **Bilingual output.** Latin and Arabic values are both extracted. | The Arabic name is what a bank keys into its core system. Cross-pass agreement on the Arabic string is an extra validity signal. |
| D2 | **`value_ar` applies to `full_name`, `nationality`, `sex` only.** | Omani cards print dates and the civil number once, in Western numerals. Requesting Arabic for those spends decode tokens on duplicates. Assumption to verify against real cards — see R4. |
| D3 | **`raw_text` is not a transcription on the fast path.** It carries pass A's raw decoded model output. Full transcription moves to `POST /api/transcribe`, on demand. | A full card transcription costs 200–400 decode tokens ≈ 7–13s on a T4, which alone breaks the latency target. The raw model string is free (already in hand) and is the more useful debugging artifact. |
| D4 | **Grounded boxes ship with a two-layer safety net:** per-box geometric filter at runtime, plus a `SHOW_BOXES` config kill switch. | `SPEC.md` requires dropping the overlay rather than drawing wrong boxes. The runtime filter makes that decision per field; the flag makes it globally, mid-demo, without a redeploy. |
| D5 | **Self-consistency, grounding and the field pass run as one batched `generate()` call.** | Decode on a T4 is bound by weight bandwidth, not compute. A batch of three re-reads the same weights once, so passes B and C cost almost no wall-clock. This is what makes the confidence design in §5 affordable. |
| D6 | **`arabic-reshaper` + `python-bidi` + Noto Naskh Arabic are approved dependencies.** | Arabic renders as disconnected, reversed letterforms in Pillow without shaping and bidi reordering, which would make synthetic eval cards worthless for the Arabic path. |
| D7 | **The GitHub repo is public.** | The notebook does a plain `git clone`. No PAT to paste each session — one less thing to fail during a live demo. |

---

## 3. Architecture

```
app/
  main.py       FastAPI app, lifespan, CORS, static mount, 3 routes
  model.py      VLMEngine protocol, QwenEngine, FakeEngine, GPU lock
  extract.py    prompt construction, batched call, JSON repair, retry
  validate.py   format rules, cross-checks, two-pass merge, box filter
  schema.py     pydantic request/response models
  config.py     env-driven settings
static/
  index.html    the adapted demo
eval/
  run_eval.py   batch scoring against ground truth
  make_samples.py  synthetic card generator
  samples/      test images + expected.json
notebook/
  run_colab.ipynb
requirements.txt
README.md
```

### 3.1 The engine seam

The single most important structural decision: **`extract.py` never imports
`transformers`.**

```
VLMEngine (Protocol)
    generate(requests: list[GenerationRequest]) -> list[str]

  ├─ QwenEngine    real inference; needs a GPU
  └─ FakeEngine    returns scripted strings; needs nothing
```

`GenerationRequest` is `(image: PIL.Image, prompt: str)`. The engine's only job is
turning a batch of those into a batch of decoded strings. It knows nothing about ID
cards, fields, or JSON.

This puts prompt construction, fence stripping, JSON repair, the retry path,
every validation rule, status derivation, the box filter, the API surface and the
entire eval harness on the local, GPU-free side of the seam. They are developed and
tested at full speed on the author's GTX 1650. Only `QwenEngine` requires the T4.

It also delivers D5's batching for free: batching is the engine's native interface,
not a special case bolted on.

### 3.2 Model lifecycle

`QwenEngine` is constructed once in the FastAPI lifespan handler and stored on
`app.state`. Never per request.

Startup sequence:

1. Load model and processor (`torch_dtype=float16`, `device_map="cuda:0"`,
   `attn_implementation="sdpa"`).
2. Run a warm-up generation against a small synthetic image with
   `max_new_tokens=8`, discarding the result. This absorbs CUDA kernel autotuning
   and allocator warm-up — otherwise the first real request pays 20–40s.
3. Only after warm-up completes does `/api/health` report `"loaded": true`.

**T4 constraints, recorded so they are not rediscovered:** T4 is compute capability
7.5. FlashAttention-2 requires 8.0+, so SDPA is the ceiling. bf16 is unsupported;
fp16 is correct here. 16 GB is ample for a 3B model in fp16 (~6 GB weights), so no
quantisation — which also avoids its quality cost.

### 3.3 Concurrency

One GPU, one model instance. `/api/extract` acquires an `asyncio.Lock` and runs
`generate()` in a thread via `run_in_threadpool`.

The lock prevents two requests interleaving on the same CUDA context. The threadpool
keeps the event loop responsive, so `/api/health` still answers while an inference is
in flight — which matters when the demo appears to hang and needs checking.

---

## 4. Inference: one call, three sequences

Every `/api/extract` request issues exactly one batched `generate()`:

| Seq | Image | Prompt | Purpose | ~tokens out |
|---|---|---|---|---|
| A | original | fields → JSON | primary extraction | ~190 |
| B | contrast-normalised | fields → JSON | self-consistency check | ~190 |
| C | original | grounding → `bbox_2d` | overlay boxes | ~120 |

Wall-clock is set by the longest sequence, so B and C are close to free. Prefill
triples (three images) but stays under a second.

**Preprocessing for pass B** is a direct port of `enhance()` from `ocr-demo.html` —
grayscale by luma weights `0.299/0.587/0.114`, then a min/max contrast stretch to full
range — reimplemented in NumPy. Same algorithm, so the preprocessing already validated
in the browser is what feeds pass B.

Generation settings: `do_sample=False` (greedy), `max_new_tokens=256`,
`min_pixels`/`max_pixels` set on the processor to cap vision tokens.

### 4.1 Field prompt (passes A and B)

Returns a flat JSON object. Flat rather than nested — uniform to parse, and fewer
tokens than nested objects.

```
You are reading an Omani national ID card. Return ONLY a JSON object, no
markdown fences, no commentary.

Keys, all required:
  full_name, full_name_ar, id_number, date_of_birth, expiry_date,
  nationality, nationality_ar, sex, sex_ar

Rules:
- Dates as ISO YYYY-MM-DD.
- sex is exactly "M" or "F".
- Latin keys hold Latin script. Keys ending _ar hold Arabic script.
- If a value is not clearly legible, return null for it.
- Do NOT guess. Do NOT infer a value from context or from another field.
  A wrong value is far worse than null.
```

The final two lines are the most important in the system. Per `SPEC.md`, a
confidently wrong ID number is this product's worst failure mode.

### 4.2 Grounding prompt (pass C)

```
Locate each printed field on this ID card. Return ONLY a JSON object mapping
field name to bbox_2d as [x1, y1, x2, y2] in pixels.
Fields: full_name, id_number, date_of_birth, expiry_date, nationality, sex
Omit any field you cannot locate.
```

**Coordinate space, easy to get wrong:** Qwen2.5-VL emits absolute pixel coordinates
against the *resized* image the processor produced, not the original upload. Boxes are
rescaled by the processor's resize factor back to the submitted image's dimensions
before leaving the backend. The frontend then scales to canvas size, exactly as
`drawBoxes()` already does.

### 4.3 Parsing and failure

1. Strip markdown fences and any prose before the first `{` / after the last `}`.
2. `json.loads`, then validate against the pydantic model.
3. On failure, **retry once** — that sequence alone, batch of one, with the parse
   error appended to the prompt.
4. On a second failure, **return every field as `missing`** with `raw_text` set to
   the unparseable output. Never guess, never partially salvage.

Passes A, B and C fail independently. If C fails, boxes are simply absent and the
overlay does not draw; extraction still succeeds. If B fails, no field can honestly be
called `ok` — the consistency check could not be performed — so §5's rule 2 is treated
as fired for every field, with `reason: "consistency pass unavailable"`. Fields null in
A are still `missing`; a field the model could not read does not become `review` merely
because a second opinion was unavailable.

---

## 5. Status derivation

`status` is `ok` / `review` / `missing`. **No number is derived from token
probabilities.** A VLM states answers fluently whether right or wrong; a percentage
computed from logits would be read by a reviewer as OCR confidence, which it is not.

Rules are evaluated in order. First match wins.

| # | Condition | Status | `reason` |
|---|---|---|---|
| 1 | null in both A and B | `missing` | — |
| 2 | A and B differ, including one null | `review` | `"passes disagreed"` |
| 3 | Format rule fails | `review` | names the rule |
| 4 | Cross-field check fails | `review` | names the check |
| 5 | otherwise | `ok` | — |

**Rule 2 precedes rule 3 deliberately.** Disagreement between passes is the only
signal available that catches hallucination; format rules cannot. A fabricated
8-digit civil number satisfies every format rule that can be written for it. Two
passes over differently-preprocessed images rarely fabricate the *same* wrong number,
which is why this ordering matters more than it looks.

Comparison for rule 2 is on the normalised value (trimmed, case-folded for Latin,
Unicode-NFC for Arabic). `value_ar` disagreement flags the field the same as `value`
disagreement.

When rule 1 applies (null in A, and B either agrees or is unavailable), `missing` wins
over everything below it. `missing` always outranks `review`.

**`SELF_CONSISTENCY=false` is a debugging setting, not a demo setting.** With pass B
disabled, rule 2 cannot fire, so fields are graded by format and cross-field rules
alone — which cannot detect hallucination (§5's opening argument). `/api/health`
reports `"self_consistency": false` so this state is visible, and the eval harness
prints a warning banner, because a silent error rate measured without it is not
comparable to one measured with it.

### 5.1 Format rules

| Field | Rule |
|---|---|
| `id_number` | exactly 8 digits (from `SPEC.md`'s sample — see R4) |
| `date_of_birth`, `expiry_date` | parses as ISO, year within 1900–2100 |
| `sex` | in `{M, F}` |
| `nationality` | in a closed list (`app/data/nationalities.txt`) |
| `full_name` | non-empty, ≥2 characters, no digits |
| `*_ar` | non-empty and contains at least one character in the Arabic Unicode block |

### 5.2 Cross-field checks

- `expiry_date` > `date_of_birth`
- `date_of_birth` is in the past
- age at `date_of_birth` is under 120 years

Arabic values are checked for agreement across passes and for script validity only.
No Latin↔Arabic transliteration comparison — that would be guesswork dressed as
validation.

Every rule carries a code comment stating *why the check is meaningful*, per
`SPEC.md`.

### 5.3 Box sanity filter (D4)

A box is dropped — field gets `box: null`, hover does nothing — if any holds:

- coordinates fall outside image bounds by more than 2% (within 2%, clamp instead)
- `x2 <= x1` or `y2 <= y1`
- area is below 0.05% or above 40% of the image
- aspect ratio outside 0.5–30 (printed field lines are wide, never tall)
- height outside 1%–25% of image height
- the identical box is claimed by more than two fields (model collapse)

`SHOW_BOXES=false` suppresses the overlay entirely.

Box accuracy is **measured, not assumed** — see §8. If the measured hit rate is poor,
the overlay is disabled and reported, not silently degraded.

---

## 6. API

### `POST /api/extract`

Multipart image upload. Response:

```json
{
  "fields": {
    "full_name":     {"value": "JOHN A SMITH", "value_ar": "جون سميث",
                      "status": "ok", "reason": null, "box": [104, 88, 402, 121]},
    "id_number":     {"value": "12345678", "value_ar": null,
                      "status": "ok", "reason": null, "box": [104, 140, 288, 170]},
    "date_of_birth": {"value": "1990-04-12", "value_ar": null,
                      "status": "review", "reason": "passes disagreed", "box": null},
    "expiry_date":   {"value": "2030-04-11", "value_ar": null,
                      "status": "ok", "reason": null, "box": [104, 196, 288, 226]},
    "nationality":   {"value": "OMANI", "value_ar": "عماني",
                      "status": "ok", "reason": null, "box": [104, 252, 250, 280]},
    "sex":           {"value": "M", "value_ar": "ذكر",
                      "status": "ok", "reason": null, "box": [104, 306, 160, 334]}
  },
  "raw_text": "{\"full_name\": \"JOHN A SMITH\", ...}",
  "agreement": {"matched": 5, "total": 6},
  "elapsed_ms": 3180,
  "model": "Qwen2.5-VL-3B-Instruct"
}
```

`value_ar` is `null` for fields outside D2's set. `box` is `null` when grounding
failed or the filter rejected it.

### `GET /api/health`

```json
{"model": "Qwen/Qwen2.5-VL-3B-Instruct", "device": "cuda:0",
 "loaded": true, "warmup_ms": 24100, "vram_mb": 7420}
```

Checked before demoing. `loaded` stays `false` until warm-up finishes.

### `POST /api/transcribe`

Multipart image upload, returns `{"text": "...", "elapsed_ms": N}`. Full card
transcription, deliberately slow (~10s), only invoked from the disclosure panel
button. Kept off the fast path per D3.

---

## 7. Frontend

`static/index.html` is `ocr-demo.html` with the OCR engine replaced. **The CSS block
is carried over unchanged** — same IBM Plex faces, same `--scan`/`--warn`/`--low`
palette, same 3px radii, same two-column grid, tabs, stage, sweep animation and
drop-hint.

| Element | Change |
|---|---|
| Tesseract CDN loader, `ensureTesseract`, `getWorker` | deleted |
| `extractFields()`, `confFor()` | deleted — server-side now |
| `runOCR()` | → `runExtract()`: canvas → `toBlob()` → `FormData` → `POST /api/extract` |
| `enhance()` | deleted from the client — moved server-side as pass B |
| `.chip ok / mid / low` | remapped: `ok`→`.ok`, `review`→`.mid`, `missing`→`.low`. **No CSS change.** |
| Metric tile 1 | `Confidence` → **`Agreement`**, showing `5/6` |
| Metric tile 2 | `Words` → **`Fields`**, showing found/total |
| Metric tile 3 | `Time` — unchanged |
| `autoScan` checkbox | removed from markup and JS |
| `Enhance before reading` checkbox | removed — enhancement is now always applied, as pass B |
| Field rows | second line for `value_ar` with `dir="rtl"`, existing mono font |
| `drawBoxes()` | retained; fed model boxes, skipping fields with `box: null` |
| Raw text panel | retained, relabelled **`Model output`**, plus a `Load full transcription` button calling `/api/transcribe` |

**The `Confidence` tile must go.** It currently renders a mean word-confidence
percentage — precisely the invented number `SPEC.md` forbids. `Agreement` replaces it
with a real measurement and preserves the three-tile layout.

**Camera** keeps capture-and-read. Continuous scanning is removed: inference takes
seconds and requests would queue and lag.

**API base URL** defaults to same-origin (`/api/...`), since the tunnel serves both
the page and the API. Overridable via `?api=<url>` or `localStorage.apiBase`, so the
page still works opened from disk against a remote tunnel. The resolved base is shown
in the status line.

**Progress** during inference is a live elapsed-second counter plus the existing
`.sweep` animation. No fabricated percentage — the backend cannot report true
progress, and a fake bar that stalls looks worse than an honest timer.

**Backend unreachable** produces:

> `Backend unreachable — the Colab tunnel is likely down or the session expired. Base: <resolved-url>`

naming the likely cause rather than surfacing a generic fetch error.

---

## 8. Evaluation harness

`eval/run_eval.py`, built **third** — before the notebook and before the frontend, per
`SPEC.md`'s instruction to build it early.

Reads every image in `eval/samples/` with its entry in `expected.json`, runs each
through the full pipeline, and prints:

```
FIELD             EXACT   OK    REVIEW  MISSING
full_name          8/8    7      1        0
id_number          7/8    6      2        0
...

SILENT ERRORS: 2/48 (4.2%)
  card_03.jpg  id_number      got 12345679  expected 12345678
  card_07.jpg  date_of_birth  got 1990-05-04 expected 1990-04-05

BOX HIT RATE: 34/48 (70.8%)
```

**Silent error rate is the headline metric:** fields where `status == "ok"` but the
value is wrong — the model was confident and wrong, and no check caught it. Each is
listed individually, not just counted, because the aggregate does not say which
failure mode to fix.

**Box hit rate** answers D4 empirically: does the returned box contain the true field
region (IoU ≥ 0.5 against ground truth)? This is the number that decides whether the
overlay ships.

`eval/make_samples.py` generates synthetic cards with Pillow so the harness runs
before real photos exist — Latin and Arabic text, using `arabic-reshaper` +
`python-bidi` + Noto Naskh Arabic (D6), with known ground truth written straight to
`expected.json`. Synthetic cards are clean renders and will overstate accuracy; they
exist to prove the harness works, not to predict field performance.

---

## 9. Configuration

`app/config.py`, pydantic-settings, all env-overridable.

| Setting | Default | Notes |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | A/B against 7B with no code change |
| `DEVICE` | `auto` | resolves to `cuda:0` when available, else `cpu`; reported by `/api/health` |
| `TORCH_DTYPE` | `float16` | bf16 unsupported on T4 |
| `MAX_NEW_TOKENS` | `256` | output is short; unbounded wastes seconds |
| `MIN_PIXELS` / `MAX_PIXELS` | `256*28*28` / `1280*28*28` | caps vision tokens |
| `SELF_CONSISTENCY` | `true` | disable to halve prefill when debugging |
| `SHOW_BOXES` | `true` | D4 kill switch |
| `DEBUG_SAVE_IMAGES` | `false` | **development only**, gitignored output |
| `DEBUG_SAVE_DIR` | `./_debug_images` | in `.gitignore` |
| `ALLOWED_ORIGINS` | `["*"]` | demo default; README documents narrowing for deployment |

---

## 10. Data handling

Habits built now, even though this is the author's own data:

- Images are held in memory as `PIL.Image` only. Never written to disk unless
  `DEBUG_SAVE_IMAGES` is explicitly enabled.
- **No image bytes and no field values in logs.** Logged per request: event name,
  elapsed ms, status counts, retry count, and whether grounding succeeded. Nothing
  else.
- A logging filter rejects any record carrying a `value` field, so a careless
  `logger.info(f"...{value}")` added later fails loudly rather than leaking quietly.
- The tunnel is demo infrastructure. Nothing in `app/` references it, knows its URL,
  or depends on it. Removing it means not running the notebook cell that starts it.

---

## 11. The Colab notebook

`notebook/run_colab.ipynb`, six clearly-labelled cells:

1. **GPU check** — `nvidia-smi`; raise with an explicit message if no GPU is attached,
   rather than failing obscurely at model load.
2. **Drive mount + cache** — mount Google Drive, set `HF_HOME` to a Drive path.
   Markdown explains that a fresh session otherwise re-downloads ~7 GB.
3. **Clone + install** — repo URL as a parameterised variable; `pip install -r
   requirements.txt`.
4. **Start server** — uvicorn in a background thread, poll `/api/health` until
   `loaded: true`, print warm-up time and VRAM.
5. **Tunnel** — Cloudflare quick tunnel; print the HTTPS URL in large, unmissable
   output. Quick tunnel over ngrok: no account, no auth token, one less live failure.
6. **Smoke test** — send a sample image through `/api/extract`, print the parsed
   fields and timing, so everything is confirmed before the screen is shared.

README notes that camera capture requires HTTPS, which the tunnel provides.

---

## 12. Testing

**Local (no GPU), against `FakeEngine`** — the whole suite runs on the author's 1650:

- fence stripping and JSON repair, including fenced, prose-wrapped and truncated output
- retry path fires once and only once; second failure yields all-`missing`
- each format rule, each cross-field check, in isolation
- status precedence — specifically that disagreement outranks a passing format rule
- box filter accepts plausible boxes and rejects each failure class
- coordinate rescaling from processor space to original image dimensions
- `/api/extract` and `/api/health` contract tests via `TestClient`
- pass B failure downgrades every field to `review`
- pass C failure leaves extraction intact with no boxes

**On Colab, requiring the T4:** model bring-up timings (§13 step 2) and `run_eval.py`.

TDD throughout: test first, watch it fail, then implement.

---

## 13. Order of work

1. Repo skeleton, `requirements.txt`, README, `config.py`, `schema.py`, `FakeEngine`,
   and the local test suite. No GPU needed.
2. `QwenEngine` and a single hardcoded image end to end on Colab. **Stop and report
   cold-start time, warm inference time, and VRAM used**, before building further.
3. `extract.py`, `validate.py`, `eval/make_samples.py`, `eval/run_eval.py`.
4. `notebook/run_colab.ipynb`.
5. `static/index.html`.
6. Real-card run: measure silent error rate and box hit rate; deliver the D4 verdict
   on the overlay.

---

## 14. Acceptance criteria

1. The notebook runs top to bottom and prints a working HTTPS tunnel URL.
2. Opening that URL serves the demo page.
3. Uploading a card photo returns schema-conforming JSON. **Target: under 5s warm.**
   Estimated ~3.2s (§15). If step 13.2 measures otherwise, the levers are, in order:
   lower `MAX_PIXELS`; move `value_ar` to a separate on-demand call; tighten the JSON.
   The measured number is reported either way — the target is not quietly redefined.
4. Camera capture works through the tunnel.
5. Illegible fields return `missing`. Nothing is invented.
6. `run_eval.py` prints the accuracy table including silent error rate.
7. `/api/health` confirms model, device and loaded state before the demo.
8. Grounding accuracy is measured and reported; the overlay ships only if the number
   justifies it.

---

## 15. Latency budget

| Stage | Estimate (T4, warm) |
|---|---|
| Decode + contrast variant | ~0.1s |
| Prefill, 3 seqs × ~1200 vision tokens | ~0.6s |
| Decode ~190 tokens @ ~30 tok/s | ~2.4s |
| Validate + serialise | <0.05s |
| **Total** | **~3.2s** |

Derived from arithmetic, not measurement: T4 has 320 GB/s of bandwidth; a 3B fp16
model is ~6 GB of weights, giving ~50 tok/s theoretical and 25–35 tok/s realistic
decode. Prefill is compute-bound and comparatively cheap. **Decode length is the
budget** — roughly one second per 30 output tokens. Every prompt and schema decision
in this document traces back to that.

Cold first inference: 20–40s, absorbed by startup warm-up (§3.2).

---

## 16. Risks

Flagged rather than worked around quietly, per `SPEC.md`.

| # | Risk | Handling |
|---|---|---|
| R1 | **Grounded box reliability.** Qwen2.5-VL grounding is trained mainly on objects and prominent text; small print on a laminated card at an angle is the hard case. | Measured as box hit rate (§8). Filtered per box (§5.3). Killable via `SHOW_BOXES`. Reported, never silently degraded. |
| R2 | **Colab session timeout mid-demo.** Recovery means re-running the notebook, re-mounting Drive and a new tunnel URL — minutes, and the URL changes. | README documents the recovery path. Drive cache (§11 cell 2) keeps it to minutes rather than a re-download. Frontend names session expiry explicitly when fetch fails. |
| R3 | **Cold-session model download** even with the Drive cache — Drive I/O is slow and the cache can miss. | Cell 2 prints cache path and hit/miss so it is visible before the demo, not during. |
| R4 | **3B too weak on small print**, and the D2 assumption that Omani cards print dates and civil numbers in Western numerals only. | Both answered by step 13.2 and the real-card run. If 3B is too weak: 7B in fp16 is ~15 GB, which fits a 16 GB T4 only barely and leaves little KV headroom — batch-of-3 may not fit, forcing sequential passes and roughly doubling latency. Cost quantified before switching. |
| R5 | **Synthetic cards overstate accuracy.** Clean renders lack glare, skew, wear and depth of field. | Stated in §8. They validate the harness, not field performance. Real photos are the acceptance gate. |
