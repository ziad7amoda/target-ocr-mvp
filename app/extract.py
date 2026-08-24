"""Extraction orchestration.

The shape of this module is dictated by one measurement: on a T4, decode is
bound by weight bandwidth, not compute. A batch re-reads the same ~6GB of
weights once regardless of how many sequences it carries, so the
self-consistency pass and the grounding pass are close to free in
wall-clock while a second sequential call would cost another full inference
(spec D5, §15).

Hence: one generate(), three sequences, and every prompt kept short because
roughly every 30 output tokens costs a second.
"""

import time

from PIL import Image

from app.boxes import filter_boxes
from app.imaging import contrast_normalise
from app.model import GenerationRequest
from app.parsing import ParseError, parse_boxes, parse_card_json
from app.schema import FIELD_NAMES, Agreement, CardFields, ExtractResponse, FieldResult
from app.validate import merge_passes

FIELD_PROMPT = """You are reading an Omani national ID card. Return ONLY a JSON object, no markdown fences, no commentary.

Keys, all required:
  full_name, full_name_ar, id_number, date_of_birth, expiry_date, nationality, nationality_ar, sex, sex_ar

Rules:
- Dates as ISO YYYY-MM-DD.
- sex is exactly "M" or "F".
- Latin keys hold Latin script. Keys ending _ar hold Arabic script.
- If a value is not clearly legible, return null for it.
- Do NOT guess. Do NOT infer a value from context or from another field. A wrong value is far worse than null."""

GROUNDING_PROMPT = """Locate each printed field on this ID card. Return ONLY a JSON object mapping field name to bbox_2d as [x1, y1, x2, y2] in pixels.
Fields: full_name, id_number, date_of_birth, expiry_date, nationality, sex
Omit any field you cannot locate."""

TRANSCRIBE_PROMPT = "Transcribe all printed text on this card, line by line, exactly as it appears."


def _all_missing(raw_text: str, model_id: str, elapsed_ms: int) -> ExtractResponse:
    """Spec §4.3: after a second parse failure, return nothing rather than a
    guess. A confidently wrong ID number is this product's worst failure
    mode - far more damaging than a blank field."""
    return ExtractResponse(
        fields={f: FieldResult(value=None, status="missing") for f in FIELD_NAMES},
        raw_text=raw_text,
        agreement=Agreement(matched=0, total=len(FIELD_NAMES)),
        elapsed_ms=elapsed_ms,
        model=model_id,
    )


def _parse_with_retry(
    text: str, image: Image.Image, engine, settings
) -> tuple[CardFields | None, str]:
    """Parse, and on failure retry ONCE with the error fed back.

    Returns (card, raw_text). card is None when both attempts failed.
    """
    try:
        return parse_card_json(text), text
    except ParseError as first:
        retry_prompt = (
            f"{FIELD_PROMPT}\n\nYour previous reply could not be parsed: {first}. "
            "Reply with the JSON object only."
        )
        retry = engine.generate([GenerationRequest(image=image, prompt=retry_prompt)])[0]
        try:
            return parse_card_json(retry), retry
        except ParseError:
            return None, retry


def extract(image: Image.Image, engine, settings, processed_size=None) -> ExtractResponse:
    t0 = time.perf_counter()

    requests = [GenerationRequest(image=image, prompt=FIELD_PROMPT)]
    if settings.SELF_CONSISTENCY:
        requests.append(
            GenerationRequest(image=contrast_normalise(image), prompt=FIELD_PROMPT)
        )
    requests.append(GenerationRequest(image=image, prompt=GROUNDING_PROMPT))

    replies = engine.generate(requests)
    primary_text = replies[0]
    secondary_text = replies[1] if settings.SELF_CONSISTENCY else None
    grounding_text = replies[-1]

    primary, primary_raw = _parse_with_retry(primary_text, image, engine, settings)
    if primary is None:
        return _all_missing(primary_raw, engine.model_id, int((time.perf_counter() - t0) * 1000))

    # A failed or disabled secondary is not fatal: merge_passes downgrades
    # every field to `review` rather than claiming an unverified `ok`.
    secondary = None
    if secondary_text is not None:
        secondary, _ = _parse_with_retry(secondary_text, image, engine, settings)

    fields, agreement = merge_passes(primary, secondary)

    if settings.SHOW_BOXES:
        boxes = filter_boxes(
            parse_boxes(grounding_text), processed_size or image.size, image.size
        )
        for name, box in boxes.items():
            fields[name].box = box

    return ExtractResponse(
        fields=fields,
        raw_text=primary_raw,
        agreement=agreement,
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
        model=engine.model_id,
    )


def transcribe(image: Image.Image, engine, settings) -> str:
    """Full card transcription. Deliberately off the fast path (spec D3): it
    costs 200-400 decode tokens, which alone breaks the latency target."""
    return engine.generate([GenerationRequest(image=image, prompt=TRANSCRIBE_PROMPT)])[0].strip()
