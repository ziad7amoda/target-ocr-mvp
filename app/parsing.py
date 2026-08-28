"""Turning model output into typed data.

The model is instructed to emit bare JSON, and mostly does. This module
handles the "mostly": fences it was told not to use, a sentence of preamble,
an unquoted number, a key spelled the way the model prefers rather than the
way the prompt asked. What it deliberately does NOT do is salvage partial or
ambiguous output - spec §4.3 requires failing to `missing` rather than
guessing, because a confidently wrong ID number is this product's worst
failure mode.

Key tolerance (added after the model-comparison runs). CardFields sets
extra="forbid", so a single unrequested key ("nationality") or a single
renamed key ("full_name" where the prompt said "full_name_ar") raised
ValidationError. The retry produced the same shape, extract() fell through
to _all_missing(), and every field came back `missing` - including the four
the model had read perfectly. Qwen2.5-VL-3B reproduces the requested key
list verbatim and so never tripped it; every other model compared against it
renames keys or volunteers extra ones, which is why they all appeared to
read nothing at all.

The tolerance is deliberately narrow, because slotting a value into the
wrong field would be worse than the bug it fixes. Keys are matched through a
fixed alias table, never fuzzily; an alias fills a slot only when the
canonical key is absent; two aliases carrying DIFFERENT values for one field
resolve to null rather than a coin-flip; and output with no recognisable key
at all is still a parse failure, not a card whose every field read blank.
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


# Alias table: normalised key -> canonical CardFields key. The canonical
# spellings normalise onto themselves, so this covers exact matches too.
# Only unambiguous synonyms belong here: a key that could plausibly name two
# different fields (a bare "number", a bare "date") is deliberately absent,
# because a value in the wrong slot is precisely the failure this module
# exists to prevent.
_KEY_ALIASES: dict[str, str] = {
    # card_type
    "cardtype": "card_type",
    "type": "card_type",
    "documenttype": "card_type",
    "cardkind": "card_type",
    "cardcategory": "card_type",
    # full_name - Arabic-only, and the suffix is the part models drop
    "fullnamear": "full_name_ar",
    "fullname": "full_name_ar",
    "fullnamearabic": "full_name_ar",
    "name": "full_name_ar",
    "namear": "full_name_ar",
    "arabicname": "full_name_ar",
    "nameinarabic": "full_name_ar",
    "holdername": "full_name_ar",
    # id_number
    "idnumber": "id_number",
    "id": "id_number",
    "idno": "id_number",
    "civilnumber": "id_number",
    "civilno": "id_number",
    "nationalid": "id_number",
    "nationalidnumber": "id_number",
    "identitynumber": "id_number",
    # date_of_birth
    "dateofbirth": "date_of_birth",
    "dob": "date_of_birth",
    "birthdate": "date_of_birth",
    "birthday": "date_of_birth",
    # expiry_date
    "expirydate": "expiry_date",
    "expiry": "expiry_date",
    "expires": "expiry_date",
    "expiration": "expiry_date",
    "expirationdate": "expiry_date",
    "dateofexpiry": "expiry_date",
    "dateofexpiration": "expiry_date",
    "validuntil": "expiry_date",
    # place_of_birth - Arabic-only
    "placeofbirthar": "place_of_birth_ar",
    "placeofbirth": "place_of_birth_ar",
    "placeofbirtharabic": "place_of_birth_ar",
    "birthplace": "place_of_birth_ar",
    "pob": "place_of_birth_ar",
}


def _normalise_key(key: str) -> str:
    """Comparison form for a JSON key.

    "Card Type", "card-type", "cardType" and "card_type" are the same key as
    far as any model is concerned, so they are the same key here.
    """
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _coerce_scalar(value):
    """Unquoted numbers are unambiguous, so coerce rather than fail. Empty
    strings mean the same thing as null and are normalised to it."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _unwrap(data):
    """Peel single-key wrapper objects: {"fields": {...}}, {"result": {...}}.

    Conservative and bounded: only a lone key whose name is not itself a card
    field is peeled, so {"id_number": {...}} is left to fail validation
    rather than being silently reinterpreted as the card.
    """
    for _ in range(3):
        if not (isinstance(data, dict) and len(data) == 1):
            return data
        ((key, value),) = data.items()
        if not isinstance(key, str) or not isinstance(value, dict):
            return data
        if _normalise_key(key) in _KEY_ALIASES:
            return data
        data = value
    return data


def map_keys(data: dict) -> dict:
    """Map a model's key spellings onto CardFields keys.

    Unrecognised keys are dropped rather than rejected: `nationality` and
    `sex` left the schema in the R1 revision because they are not printed on
    the card, so a model volunteering them is the expected case. Conflicts
    resolve to None - see the module docstring.
    """
    mapped: dict[str, object] = {}
    exact: set[str] = set()
    conflicted: set[str] = set()

    for raw_key, raw_value in data.items():
        if not isinstance(raw_key, str):
            continue
        normalised = _normalise_key(raw_key)
        canonical = _KEY_ALIASES.get(normalised)
        if canonical is None:
            continue
        value = _coerce_scalar(raw_value)

        # The canonical spelling is authoritative: if the model emitted both
        # `full_name_ar` and `full_name`, the key we actually asked for wins
        # outright, and no alias can later demote it to a conflict.
        if normalised == _normalise_key(canonical):
            mapped[canonical] = value
            exact.add(canonical)
            conflicted.discard(canonical)
            continue
        if canonical in exact:
            continue

        if canonical in mapped:
            if value is not None and mapped[canonical] is not None and mapped[canonical] != value:
                conflicted.add(canonical)
            elif mapped[canonical] is None:
                mapped[canonical] = value
            continue
        mapped[canonical] = value

    # Two aliases, two different values, no way to tell which one the card
    # actually says. Spec §4.3: return nothing rather than pick one.
    for field in conflicted:
        mapped[field] = None
    return mapped


def parse_card_json(text: str) -> CardFields:
    stripped = strip_fences(text)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ParseError(f"not valid JSON: {exc.msg}") from exc

    data = _unwrap(data)
    if not isinstance(data, dict):
        raise ParseError(f"expected a JSON object, got {type(data).__name__}")

    mapped = map_keys(data)
    if not mapped:
        # Tolerance is not credulity. Nothing here names a card field, so
        # this is output we failed to understand - not a card whose every
        # field happened to be unreadable. Reporting six `missing` fields
        # would dress a parse failure up as a reading result.
        keys = sorted(k for k in data if isinstance(k, str))[:6]
        raise ParseError(f"no recognisable card fields in {keys}")

    try:
        return CardFields(**mapped)
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
