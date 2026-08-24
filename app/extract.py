"""Extraction orchestration.

The shape of this module is dictated by one measurement: on a T4, decode is
bound by weight bandwidth, not compute. A batch re-reads the same ~6GB of
weights once regardless of how many sequences it carries, so the
self-consistency pass is close to free in wall-clock while a second
sequential call would cost another full inference (spec D5, §15).

Hence: one generate(), and every prompt kept short because roughly every 18
output tokens costs about a second on the target hardware.

Revision 2026-08-24 (docs/superpowers/specs/2026-08-24-revision-real-card-findings.md,
R7): grounding returned no usable boxes against real cards, so SHOW_BOXES now
defaults to false and, when it is false, the grounding GenerationRequest is
never issued at all - not just filtered out afterwards. That means the
normal batch is TWO sequences (primary + consistency), not three. The
grounding reply is only ever read from `replies[-1]` when a grounding
request was actually appended; see the `grounding_requested` flag below.
Reading `replies[-1]` unconditionally would, with grounding disabled, read
pass B's field JSON as if it were box data.
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
  card_type, full_name_ar, id_number, date_of_birth, expiry_date, place_of_birth_ar

card_type: "citizen" if the card says البطاقة الشخصية / IDENTITY CARD, "resident" if it
says بطاقة مقيم / RESIDENT CARD.

full_name_ar: the الإسم line, in Arabic script. The card prints no Latin name. These
names have five to eight components (given name, father, grandfather, family names) -
transcribe EVERY component, do not stop early.

place_of_birth_ar: the مكان الميلاد line, in Arabic script. The card prints no Latin
place name.

id_number: the 8-digit civil number, digits only.

date_of_birth, expiry_date: dates are PRINTED on the card as DD/MM/YYYY. You must
convert to ISO YYYY-MM-DD, never return the card's printed format. Example: a card
printed with 11/08/1989 means 11 August 1989, so the value you return is "1989-08-11",
NOT "1989-11-08".

Rules:
- If a value is not clearly legible, return null for it.
- Do NOT guess. Do NOT infer a value from context, from another field, from the card's
  header or layout, or from the holder's photograph. A wrong value is far worse than
  null."""

GROUNDING_PROMPT = """Locate each printed field on this ID card. Return ONLY a JSON object mapping field name to bbox_2d as [x1, y1, x2, y2] in pixels.
Fields: card_type, full_name, id_number, date_of_birth, expiry_date, place_of_birth
Omit any field you cannot locate."""

TRANSCRIBE_PROMPT = "Transcribe all printed text on this card, line by line, exactly as it appears."


def _all_missing(raw_text: str, model_id: str, elapsed_ms: int) -> ExtractResponse:
    """Spec §4.3: after a second parse failure, return nothing rather than a
    guess. A confidently wrong ID number is this product's worst failure
    mode - far more damaging than a blank field."""
    return ExtractResponse(
        fields={f: FieldResult(value=None, status="missing") for f in FIELD_NAMES},
        raw_text=raw_text,
        agreement=Agreement(matched=0, compared=0, total=len(FIELD_NAMES)),
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
    contrast_image = None
    if settings.SELF_CONSISTENCY:
        contrast_image = contrast_normalise(image)
        requests.append(GenerationRequest(image=contrast_image, prompt=FIELD_PROMPT))
    # Only issue the grounding request when boxes will actually be used - a
    # discarded sequence still pays for a full prefill + decode. Because
    # this flag gates both the append below and the read of replies[-1],
    # replies[-1] can never be mistaken for the wrong pass's output: it is
    # the grounding reply whenever grounding_requested is True, and it is
    # simply never read when False.
    grounding_requested = settings.SHOW_BOXES
    if grounding_requested:
        requests.append(GenerationRequest(image=image, prompt=GROUNDING_PROMPT))

    replies = engine.generate(requests)
    primary_text = replies[0]
    secondary_text = replies[1] if settings.SELF_CONSISTENCY else None
    grounding_text = replies[-1] if grounding_requested else None

    primary, primary_raw = _parse_with_retry(primary_text, image, engine, settings)
    if primary is None:
        return _all_missing(primary_raw, engine.model_id, int((time.perf_counter() - t0) * 1000))

    # A failed or disabled secondary is not fatal: merge_passes downgrades
    # every field to `review` rather than claiming an unverified `ok`.
    secondary = None
    if secondary_text is not None:
        # The retry MUST re-read the contrast image, not the original. Decoding
        # is greedy (do_sample=False), so retrying pass B over pass A's pixels
        # would just be a second read of the SAME image: the two passes would
        # then agree by construction, rule 2 ("passes disagreed") - the only
        # check that catches a well-formed hallucination - could never fire,
        # and the field would be served `ok` unchecked.
        secondary, _ = _parse_with_retry(secondary_text, contrast_image, engine, settings)

    fields, agreement = merge_passes(primary, secondary)

    if grounding_requested and grounding_text is not None:
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
