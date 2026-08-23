"""Meaningful validity checks.

Spec §5: a VLM does not expose per-character confidence. It states answers
fluently whether right or wrong, so any percentage derived from token
probabilities would be read by a reviewer as OCR confidence while meaning
nothing of the sort. Every check in this module is instead something that
can actually be verified about the value itself.

Each rule returns a human-readable reason on failure, which travels to the
UI verbatim - a reviewer seeing "expiry precedes date of birth" knows what
to look at; "confidence 62%" does not tell them anything actionable.
"""

import re
from datetime import date
from pathlib import Path

from app.schema import ARABIC_FIELDS

_ID_NUMBER = re.compile(r"^\d{8}$")
_HAS_DIGIT = re.compile(r"\d")
# Arabic (0600-06FF), Arabic Supplement, and Arabic Presentation Forms.
_ARABIC_SCRIPT = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")

_DATE_FIELDS = ("date_of_birth", "expiry_date")
_MIN_YEAR, _MAX_YEAR = 1900, 2100
_MAX_AGE_YEARS = 120


def _nationalities() -> set[str]:
    path = Path(__file__).parent / "data" / "nationalities.txt"
    return {line.strip().upper() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _parse_iso(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def check_format(field: str, value: str | None) -> str | None:
    """Return a reason string if `value` fails its field's format rule.

    A null value is NOT a format failure - it is handled by the `missing`
    rule, which outranks everything here (spec §5).
    """
    if value is None:
        return None
    value = value.strip()

    if field == "id_number":
        # Omani civil numbers are 8 digits. Note this catches typos and
        # truncation but CANNOT catch a fabricated 8-digit number - that is
        # what the self-consistency check in merge_passes() is for.
        if not _ID_NUMBER.match(value):
            return "expected 8 digits"

    elif field in _DATE_FIELDS:
        parsed = _parse_iso(value)
        if parsed is None:
            return "not a valid ISO date (YYYY-MM-DD)"
        if not _MIN_YEAR <= parsed.year <= _MAX_YEAR:
            return f"year outside {_MIN_YEAR}-{_MAX_YEAR}"

    elif field == "sex":
        if value.upper() not in {"M", "F"}:
            return "expected M or F"

    elif field == "nationality":
        if value.upper() not in _nationalities():
            return "not in the known nationality list"

    elif field == "full_name":
        if len(value) < 2:
            return "too short to be a name"
        if _HAS_DIGIT.search(value):
            return "contains digits"

    return None


def check_arabic(field: str, value_ar: str | None) -> str | None:
    """Verify an Arabic value is actually in Arabic script.

    Deliberately does NOT compare against the Latin value: transliteration
    between the two is guesswork, and guesswork dressed as validation is
    worse than no check (spec §5.2).
    """
    if field not in ARABIC_FIELDS or value_ar is None:
        return None
    if not _ARABIC_SCRIPT.search(value_ar):
        return "expected Arabic script"
    return None


def check_cross_fields(values: dict[str, str | None]) -> dict[str, str]:
    """Checks spanning more than one field. Returns field -> reason.

    Values that fail their own format rule are skipped: they are already
    flagged, and a second complaint about the same bad value adds nothing.
    """
    out: dict[str, str] = {}
    dob = _parse_iso(values.get("date_of_birth") or "")
    exp = _parse_iso(values.get("expiry_date") or "")

    if dob and exp and exp <= dob:
        out["date_of_birth"] = "expiry precedes date of birth"
        out["expiry_date"] = "expiry precedes date of birth"

    if dob:
        today = date.today()
        if dob > today:
            out["date_of_birth"] = "date of birth is in the future"
        elif (today - dob).days > _MAX_AGE_YEARS * 365.25:
            out["date_of_birth"] = f"implies an age over {_MAX_AGE_YEARS}"

    return out
