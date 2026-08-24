from typing import Literal

from pydantic import BaseModel, ConfigDict

FIELD_NAMES: list[str] = [
    "full_name",
    "id_number",
    "date_of_birth",
    "expiry_date",
    "nationality",
    "sex",
]

# Spec D2: Omani cards print dates and the civil number once, in Western
# numerals. Requesting Arabic for those spends decode tokens on duplicates.
ARABIC_FIELDS: set[str] = {"full_name", "nationality", "sex"}

Status = Literal["ok", "review", "missing"]


class CardFields(BaseModel):
    """Exactly what the model is asked to emit. Flat, not nested: uniform to
    parse and fewer tokens than nested objects."""

    model_config = ConfigDict(extra="forbid")

    full_name: str | None = None
    full_name_ar: str | None = None
    id_number: str | None = None
    date_of_birth: str | None = None
    expiry_date: str | None = None
    nationality: str | None = None
    nationality_ar: str | None = None
    sex: str | None = None
    sex_ar: str | None = None


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
