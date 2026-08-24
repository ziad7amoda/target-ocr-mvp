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
        "card_type",
        "full_name",
        "id_number",
        "date_of_birth",
        "expiry_date",
        "place_of_birth",
    ]


def test_only_two_fields_carry_arabic():
    assert ARABIC_FIELDS == {"full_name", "place_of_birth"}


def test_card_fields_accepts_all_nulls():
    c = CardFields()
    assert c.card_type is None
    assert c.full_name_ar is None


def test_card_fields_rejects_unknown_key():
    with pytest.raises(ValidationError):
        CardFields(card_type="citizen", height="180cm")


def test_invented_fields_are_gone():
    """nationality and sex are not printed on Omani ID cards. The model invented
    both, and on a real resident card returned sex="M" with status ok - a value
    never on the card, presented as read. They are not extractable, so they are
    not fields."""
    assert "nationality" not in FIELD_NAMES
    assert "sex" not in FIELD_NAMES
    with pytest.raises(ValidationError):
        CardFields(nationality="OMANI")


def test_field_result_rejects_invalid_status():
    with pytest.raises(ValidationError):
        FieldResult(value="X", status="probably")


def test_extract_response_round_trips():
    r = ExtractResponse(
        fields={"card_type": FieldResult(value="citizen", status="ok")},
        raw_text="{}",
        agreement=Agreement(matched=1, compared=1, total=6),
        elapsed_ms=3180,
        model="Qwen2.5-VL-3B-Instruct",
    )
    assert r.model_dump()["fields"]["card_type"]["box"] is None
