from typing import Literal

from pydantic import BaseModel, ConfigDict

# Revision 2026-08-24 (see docs/superpowers/specs/2026-08-24-revision-real-card-findings.md):
# two real Omani cards (one citizen, one resident) print NEITHER nationality
# NOR sex. The model was inventing both - "Oman" from the "SULTANATE OF OMAN"
# header regardless of the holder's actual nationality, and sex from the
# photo or from the بنت/بن particle in the name. On the resident card `sex`
# came back "M" with status `ok`: a value that was never on the card,
# presented to the reviewer as read. No confidence or agreement mechanism
# can rescue a field that does not exist on the document, so `nationality`
# and `sex` are removed rather than hardened. Extract only what is printed.
#
# `card_type` ("citizen" | "resident") and `place_of_birth` replace them:
# both are actually printed, and place_of_birth is the honest proxy for the
# nationality question a reviewer was really asking.
FIELD_NAMES: list[str] = [
    "card_type",
    "full_name",
    "id_number",
    "date_of_birth",
    "expiry_date",
    "place_of_birth",
]

# Spec D2 (as revised): Omani cards print dates, the civil number, and
# card_type once, in Western script/numerals. Requesting Arabic for those
# spends decode tokens on duplicates. full_name and place_of_birth are the
# opposite case: they are printed ONLY in Arabic on both real cards seen so
# far - there is no Latin name or Latin place of birth to extract - so they
# carry Arabic and nothing else.
ARABIC_FIELDS: set[str] = {"full_name", "place_of_birth"}

Status = Literal["ok", "review", "missing"]


class CardFields(BaseModel):
    """Exactly what the model is asked to emit. Flat, not nested: uniform to
    parse and fewer tokens than nested objects.

    Deliberately asymmetric with FIELD_NAMES: FIELD_NAMES lists the logical
    field (`full_name`), while the keys here carry the script suffix the
    model actually emits (`full_name_ar`). `full_name` and `place_of_birth`
    have ONLY an `_ar` key - there is no Latin counterpart on the card, so
    inventing one here would just recreate the D1/D2 mistake this revision
    fixes. `card_type`, `id_number` and the two dates have no `_ar` variant
    because they are printed in Latin script / Western numerals.
    """

    model_config = ConfigDict(extra="forbid")

    card_type: str | None = None
    full_name_ar: str | None = None
    id_number: str | None = None
    date_of_birth: str | None = None
    expiry_date: str | None = None
    place_of_birth_ar: str | None = None


class FieldResult(BaseModel):
    value: str | None = None
    value_ar: str | None = None
    status: Status
    reason: str | None = None
    box: tuple[int, int, int, int] | None = None


class Agreement(BaseModel):
    matched: int
    # Fields that actually reached the cross-pass comparison - i.e. excludes
    # fields that were `missing` (never compared) or where the consistency
    # pass was unavailable entirely. `matched <= compared <= total`. A UI
    # should render matched/compared, not matched/total: a card with
    # illegible fields should not read as those fields having disagreed.
    compared: int
    total: int


class ExtractResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    fields: dict[str, FieldResult]
    raw_text: str | None = None
    agreement: Agreement
    elapsed_ms: int
    model: str
    # Which PROMPT_STYLE produced this result. The response already named
    # the model but not the prompt, and the prompt has turned out to change
    # the outcome more than the model does - a run where Qari returned null
    # for both Arabic fields and a run where it read them correctly are
    # indistinguishable in this payload without it. Every result should say
    # what produced it.
    prompt_style: str | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    device: str
    loaded: bool
    self_consistency: bool
    warmup_ms: int | None = None
    vram_mb: int | None = None


class TranscribeResponse(BaseModel):
    text: str
    elapsed_ms: int
