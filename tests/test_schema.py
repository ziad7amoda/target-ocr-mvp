import pytest
from pydantic import ValidationError

from app.schema import (
    ARABIC_FIELDS,
    FIELD_NAMES,
    Agreement,
    CardFields,
    ExtractResponse,
    FieldResult,
)


def test_field_names_are_the_six_spec_fields():
    assert FIELD_NAMES == [
        "full_name",
        "id_number",
        "date_of_birth",
        "expiry_date",
        "nationality",
        "sex",
    ]


def test_only_three_fields_carry_arabic():
    assert ARABIC_FIELDS == {"full_name", "nationality", "sex"}


def test_card_fields_accepts_all_nulls():
    c = CardFields()
    assert c.full_name is None
    assert c.sex_ar is None


def test_card_fields_rejects_unknown_key():
    with pytest.raises(ValidationError):
        CardFields(full_name="X", height="180cm")


def test_field_result_rejects_invalid_status():
    with pytest.raises(ValidationError):
        FieldResult(value="X", status="probably")


def test_extract_response_round_trips():
    r = ExtractResponse(
        fields={"sex": FieldResult(value="M", value_ar="ذكر", status="ok")},
        raw_text="{}",
        agreement=Agreement(matched=1, compared=1, total=6),
        elapsed_ms=3180,
        model="Qwen2.5-VL-3B-Instruct",
    )
    assert r.model_dump()["fields"]["sex"]["box"] is None
