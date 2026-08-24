"""Turning model output into typed data.

The model is instructed to emit bare JSON, and mostly does. This module
handles the "mostly": fences it was told not to use, a sentence of preamble,
an unquoted number. What it deliberately does NOT do is salvage partial or
ambiguous output - spec §4.3 requires failing to `missing` rather than
guessing, because a confidently wrong ID number is this product's worst
failure mode.
"""

import json
import re

from pydantic import ValidationError

from app.schema import CardFields

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class ParseError(Exception):
    """Model output could not be turned into typed data."""


def strip_fences(text: str) -> str:
    """Remove markdown fences and any prose outside the outermost braces."""
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned.strip()


def parse_card_json(text: str) -> CardFields:
    stripped = strip_fences(text)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ParseError(f"not valid JSON: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise ParseError(f"expected a JSON object, got {type(data).__name__}")

    # Unquoted numbers are unambiguous, so coerce rather than fail. Empty
    # strings mean the same thing as null and are normalised to it.
    for key, value in list(data.items()):
        if isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)):
            data[key] = str(value)
        elif isinstance(value, str) and not value.strip():
            data[key] = None

    try:
        return CardFields(**data)
    except ValidationError as exc:
        raise ParseError(f"does not match the card schema: {exc.errors()[0]['msg']}") from exc


def parse_boxes(text: str) -> dict[str, list[int]]:
    """Best-effort box extraction. Never raises: grounding is optional and its
    failure must leave extraction intact (spec §4.3)."""
    if not isinstance(text, str):
        return {}
    try:
        data = json.loads(strip_fences(text))
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}

    out: dict[str, list[int]] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            value = value.get("bbox_2d")
        if isinstance(value, list) and len(value) == 4:
            try:
                out[key] = [int(round(float(v))) for v in value]
            except (TypeError, ValueError):
                continue
    return out
