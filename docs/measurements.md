# Measurements

This file is a **template awaiting real data**. Every numeric slot below is
`_not yet measured_` because this repository is being developed on a
GPU-less Windows machine with no Colab session, no Noto Naskh Arabic font,
and no real ID card photos on hand. Nothing here is invented. Fill in each
slot after actually running the commands in the next section, then replace
this paragraph's caveat with the real header values.

## Header

| Field | Value |
|---|---|
| Date measured | `_not yet measured_` |
| Model ID | `_not yet measured_` (repo default: `Qwen/Qwen2.5-VL-3B-Instruct`, see `app/config.py`) |
| GPU | `_not yet measured_` (e.g. "Colab T4, 16GB") |
| Commit SHA | `_not yet measured_` (`git rev-parse HEAD` at measurement time) |

## How to produce these numbers

Both commands require a CUDA GPU (a Colab T4 is the target environment per
the design) and the GPU inference stack installed:

```bash
pip install -r requirements.txt -r requirements-gpu.txt
```

**1. Model bring-up (cold load, warm-up, warm single vs. batch-of-3 latency, VRAM):**

```bash
python scripts/bringup.py path/to/card.jpg
```

This prints weight load time, warm-up ms, warm batch-of-1 ms, warm batch-of-3
ms, peak VRAM, and the original vs. processed image size (`app/model.py`'s
`processed_size()` — if processed size equals original size on a large
photo, the grid lookup fell back and grounding boxes will be drawn in the
wrong place; check this before trusting any box hit rate below).

**2. Accuracy, review/missing/silent-error rates, box hit rate:**

Requires `eval/samples/` populated with real card photos and hand-typed
ground truth in `eval/samples/expected.json` (spec §14 step 1 — not done in
this repo yet; see "Not yet executable" in the task report).

```bash
python -m eval.run_eval
```

## Latency

| Metric | Value |
|---|---|
| Cold weight load | `_not yet measured_` |
| Warm-up (first request after load) | `_not yet measured_` |
| Warm, batch of 1 | `_not yet measured_` |
| Warm, batch of 3 | `_not yet measured_` |
| Peak VRAM | `_not yet measured_` |
| Original image size | `_not yet measured_` |
| Processed image size | `_not yet measured_` |

**Load-bearing assumption (spec D5):** the batch-of-3 figure must land
within roughly 30% of the batch-of-1 figure. Design decision D5 runs
self-consistency (passes A and B) and grounding (pass C) as a single batched
`generate()` call specifically because decode on a T4 is bound by weight
bandwidth, not compute — a batch of three should re-read the same weights
once, not three times. **If batch-of-3 is not within ~30% of batch-of-1,
that assumption is false and the latency design (and the "under 5s warm"
target in acceptance criterion 3) needs revisiting** — see the levers listed
in spec §14 item 3 (lower `MAX_PIXELS`, move `value_ar` to a separate
on-demand call, tighten the JSON).

## Accuracy (from `eval/run_eval.py`)

| Field | Exact match | OK | Review | Missing |
|---|---|---|---|---|
| `_not yet measured_` | | | | |

**Silent error rate (headline number): `_not yet measured_`**

Silent errors — `status == "ok"` but the value is wrong — must be listed
individually, not just counted, per spec §8:

```
_not yet measured_ — list each as: <sample> <field> got <value> expected <value>
```

**Box hit rate: `_not yet measured_`**

IoU ≥ 0.5 against ground truth, per spec §8. This is the number that
decides whether the bounding-box overlay ships (spec D4, §16 R1). Per the
brief: report this number to the user either way — it must never be used to
silently disable `SHOW_BOXES` without saying so. `app/config.py`'s
`SHOW_BOXES` default is currently `true` (unchanged, since no measurement
exists yet to justify changing it).

## D2 assumption check

Spec D2 assumes Omani cards print date of birth, expiry and the civil
number in Western numerals only, so only `full_name`, `nationality`, and
`sex` need `value_ar`. This must be verified against real cards:

`_not yet checked_` — if any sampled card prints Eastern Arabic numerals
(٠١٢٣٤٥٦٧٨٩) in those fields, D2 is wrong and those fields need `value_ar`
too. Report this rather than working around it.

---

## Acceptance criteria status (spec §14)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | The notebook runs top to bottom and prints a working HTTPS tunnel URL. | **Pending hardware** | Requires Colab with a GPU runtime; `notebook/run_colab.ipynb` cannot execute on this machine. Run the notebook top to bottom on Colab and confirm the tunnel URL prints. |
| 2 | Opening that URL serves the demo page. | **Pending hardware** | Depends on criterion 1's tunnel. Open the printed URL and confirm `static/index.html` loads. |
| 3 | Uploading a card photo returns schema-conforming JSON, under 5s warm. | **Pending hardware** | Requires the loaded model on a GPU. Run `python scripts/bringup.py <image>` (warm batch-of-1 line) or exercise `/api/extract` from the running server, and compare against 5s. If it fails, apply the levers in spec §14 item 3 and re-measure — do not redefine the target. |
| 4 | Camera capture works through the tunnel. | **Pending hardware** | Requires HTTPS via the Cloudflare tunnel from the notebook, and a device with a camera. Cannot be exercised without Colab. |
| 5 | Illegible fields return `missing`. Nothing is invented. | **Verified locally** | `tests/test_extract.py`, `tests/test_validate_merge.py`, and `tests/test_validate_rules.py` exercise this against `FakeEngine` (e.g. pass-B failure downgrades every field, malformed/absent fields map to `missing`, no value is fabricated when the model returns null/illegible). This covers the pipeline's *handling* of illegibility; it does not substitute for real-card confirmation that the model itself refuses to invent values, which needs the real-card eval run. |
| 6 | `run_eval.py` prints the accuracy table including silent error rate. | **Verified locally (partially)** | `eval/run_eval.py` exists and its output format (including the silent-error listing) was checked by reading the module; it has not been run end-to-end against real samples because `eval/samples/expected.json` does not exist yet (real photos are step 1 of this task's brief, unexecutable here). Producing the actual table is pending hardware + real cards. |
| 7 | `/api/health` confirms model, device and loaded state before the demo. | **Verified locally** | `tests/test_api.py`'s health-endpoint tests pass against `FakeEngine`, confirming the endpoint reports `model`, `device`, and `loaded` fields correctly for both loaded and not-loaded states. Real-GPU confirmation (that `loaded: true` actually appears after a real Colab load) is pending hardware. |
| 8 | Grounding accuracy is measured and reported; the overlay ships only if the number justifies it. | **Pending hardware** | Requires the box hit rate from `eval/run_eval.py` against real cards (see "Box hit rate" above). `SHOW_BOXES` is left at its current default (`true`) until that number exists — no verdict has been made either way, and none should be inferred from this file's placeholders. |

**Summary: 2 of 8 verified locally, 1 partially verified locally (real-data
confirmation still pending), 5 pending hardware.**
