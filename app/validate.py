"""Meaningful validity checks.

Spec §5: a VLM does not expose per-character confidence. It states answers
fluently whether right or wrong, so any percentage derived from token
probabilities would be read by a reviewer as OCR confidence while meaning
nothing of the sort. Every check in this module is instead something that
can actually be verified about the value itself.

Each rule returns a human-readable reason on failure, which travels to the
UI verbatim - a reviewer seeing "expiry precedes date of birth" knows what
to look at; "confidence 62%" does not tell them anything actionable.

Revision 2026-08-24 (docs/superpowers/specs/2026-08-24-revision-real-card-findings.md):
real cards print full_name and place_of_birth ONLY in Arabic. CardFields
therefore has no `full_name` / `place_of_birth` attribute, only the
`_ar`-suffixed ones. `_field_value()` is the single place that knows how to
pull a logical field's (value, value_ar) pair out of a CardFields instance -
every other function goes through it instead of calling getattr() directly,
so this asymmetry is handled in exactly one place.
"""

import re
import unicodedata
from datetime import date

from app.schema import ARABIC_FIELDS, FIELD_NAMES, Agreement, CardFields, FieldResult

_ID_NUMBER = re.compile(r"^\d{8}$")
# Arabic (0600-06FF), Arabic Supplement, and Arabic Presentation Forms.
# Kept byte-identical to the original - verified against real Arabic values
# during the initial build and deliberately not touched here.
_ARABIC_SCRIPT = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
# date.fromisoformat is more permissive than its name suggests on Python 3.11+:
# it also accepts basic format ("19900412") and ISO week dates ("2024-W01-1").
# The contract here is specifically YYYY-MM-DD, so anchor before delegating.
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_DATE_FIELDS = ("date_of_birth", "expiry_date")
_CARD_TYPES = {"citizen", "resident"}
_MIN_YEAR, _MAX_YEAR = 1900, 2100
_MAX_AGE_YEARS = 120

# Arabic labels printed on the card. A model that returns one of these has
# read the label instead of the value beside it - a field-assignment error,
# not a reading error, and one that no format rule would otherwise catch
# because a label IS valid Arabic text of plausible length.
_CARD_LABELS = {"مكان الميلاد", "الإسم", "الاسم", "المهنة", "الرقم المدني",
                "تاريخ الميلاد", "تاريخ الإنتهاء", "بطاقة مقيم", "البطاقة الشخصية"}


def _field_value(card: CardFields, field: str) -> tuple[str | None, str | None]:
    """Map a logical field name to its (value, value_ar) pair.

    This is the one place that knows full_name/place_of_birth are
    Arabic-only: they live at `<field>_ar` on CardFields and have no Latin
    counterpart, while every other field is Latin-only and has no `_ar`
    counterpart. Everything downstream (merge_passes, cross-field checks)
    goes through this instead of calling getattr() directly, so a card with
    an Arabic-only field never silently compares None to None (see
    test_arabic_only_field_disagreement_is_detected).
    """
    if field in ARABIC_FIELDS:
        return None, getattr(card, f"{field}_ar")
    return getattr(card, field), None


def _parse_iso(value: str) -> date | None:
    if not _ISO_DATE.match(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def check_format(field: str, value: str | None) -> str | None:
    """Return a reason string if `value` fails its field's format rule.

    A null value is NOT a format failure - it is handled by the `missing`
    rule, which outranks everything here (spec §5). Arabic-only fields
    (full_name, place_of_birth) have no Latin `value` to check here at all -
    their checks live in check_arabic().
    """
    if value is None:
        return None
    value = value.strip()

    if field == "card_type":
        # The only two card families seen on real cards (spec R2). Anything
        # else means the model invented a value or misread the header -
        # there is no third option to be lenient about.
        if value not in _CARD_TYPES:
            return "expected citizen or resident"

    elif field == "id_number":
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

    return None


def check_arabic(field: str, value_ar: str | None) -> str | None:
    """Verify an Arabic value is actually plausible Arabic-script text.

    Deliberately does NOT compare against a Latin value: transliteration
    between the two is guesswork, and guesswork dressed as validation is
    worse than no check (spec §5.2). There is no Latin value to compare
    against anyway - full_name and place_of_birth are Arabic-only.
    """
    if field not in ARABIC_FIELDS or value_ar is None:
        return None
    value_ar = value_ar.strip()

    if not _ARABIC_SCRIPT.search(value_ar):
        return "expected Arabic script"

    if _normalise(value_ar) in _NORMALISED_CARD_LABELS:
        return "this is a field label, not a value"

    # A one-character value is almost certainly a truncation or misread
    # rather than a real name or place, regardless of script.
    if len(value_ar) < 2:
        return "too short to be valid"

    if field == "full_name":
        # Real Omani names are many components (see spec §3: a seven-
        # component name truncated to three under a token ceiling). A
        # single token is a strong truncation/misread signal worth
        # surfacing to a reviewer even though the script check passed.
        if len(value_ar.split()) < 2:
            return "name has fewer than two components"

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


def _normalise(value: str | None) -> str | None:
    """Comparison form for cross-pass agreement.

    NFC first so two byte-different but identical Arabic strings compare
    equal; then case-fold and collapse whitespace so trivial formatting
    differences are not reported to a human as a disagreement.
    """
    if value is None:
        return None
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


# Normalised once, at import time, using the same comparison form as
# cross-pass agreement - so a byte-different but visually identical label
# (e.g. composed vs. decomposed Arabic) is still caught in check_arabic().
_NORMALISED_CARD_LABELS = {_normalise(label) for label in _CARD_LABELS}


def merge_passes(
    primary: CardFields, secondary: CardFields | None
) -> tuple[dict[str, FieldResult], Agreement]:
    """Combine two extraction passes into per-field results and statuses.

    `secondary=None` means the consistency pass failed or was disabled.

    Rule order (spec §5), first match wins:
        1. null in both              -> missing
        2. passes disagree           -> review, "passes disagreed"
        3. format rule fails         -> review, names the rule
        4. cross-field check fails   -> review, names the check
        5. otherwise                 -> ok

    Rule 2 precedes rule 3 deliberately. Format rules cannot catch
    hallucination: a fabricated 8-digit civil number satisfies every one of
    them. Two passes over differently-preprocessed images rarely fabricate
    the SAME wrong value, which makes disagreement the strongest signal
    available - and therefore the one a reviewer should be shown first.
    """
    latin = {f: getattr(primary, f) for f in FIELD_NAMES if f not in ARABIC_FIELDS}
    cross = check_cross_fields(latin)

    results: dict[str, FieldResult] = {}
    matched = 0
    compared = 0

    for field in FIELD_NAMES:
        value, value_ar = _field_value(primary, field)
        is_null = value is None and value_ar is None

        if secondary is not None:
            sec_value, sec_value_ar = _field_value(secondary, field)
        else:
            sec_value = sec_value_ar = None

        # Rule 1. `missing` always outranks `review`, including when the
        # consistency pass is unavailable.
        if is_null and (secondary is None or (sec_value is None and sec_value_ar is None)):
            results[field] = FieldResult(value=None, value_ar=None, status="missing")
            continue

        if secondary is None:
            results[field] = FieldResult(
                value=value,
                value_ar=value_ar,
                status="review",
                reason="consistency pass unavailable",
            )
            continue

        # This field reached an actual comparison - the denominator for the
        # Agreement tile, as opposed to `total` which counts every field.
        compared += 1

        # Rule 2. Compare whichever side is actually populated for this
        # field. An Arabic-only field has `value is None` on BOTH passes
        # always - comparing `value` there would trivially "agree" every
        # time and silently disable the hallucination check for full_name
        # and place_of_birth, the two most important fields on the card
        # (see test_arabic_only_field_disagreement_is_detected).
        agrees = _normalise(value) == _normalise(sec_value) and _normalise(value_ar) == _normalise(
            sec_value_ar
        )

        if not agrees:
            results[field] = FieldResult(
                value=value, value_ar=value_ar, status="review", reason="passes disagreed"
            )
            continue

        matched += 1

        # Rules 3 and 4.
        reason = check_format(field, value) or check_arabic(field, value_ar) or cross.get(field)
        results[field] = FieldResult(
            value=value,
            value_ar=value_ar,
            status="review" if reason else "ok",
            reason=reason,
        )

    return results, Agreement(matched=matched, compared=compared, total=len(FIELD_NAMES))
