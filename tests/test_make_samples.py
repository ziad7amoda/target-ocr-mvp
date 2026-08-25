import pytest

pytest.importorskip("arabic_reshaper")

from eval.make_samples import RECORDS, shape_arabic


def test_shaping_joins_and_reverses_arabic():
    """Without reshaping + bidi, Pillow renders Arabic as disconnected,
    reversed letterforms, which would make the samples worthless."""
    out = shape_arabic("عماني")
    assert out != "عماني"
    assert len(out) > 0


def test_shaping_leaves_latin_untouched():
    assert shape_arabic("OMANI") == "OMANI"


def test_records_cover_the_declared_fields():
    """Exactly CardFields' six keys - card_type and place_of_birth_ar
    replaced nationality/sex, and full_name is Arabic-only (spec R1/R4)."""
    for record in RECORDS:
        assert set(record) == {
            "card_type", "full_name_ar", "id_number", "date_of_birth",
            "expiry_date", "place_of_birth_ar",
        }


def test_records_include_both_card_types():
    assert {r["card_type"] for r in RECORDS} == {"citizen", "resident"}


def test_records_are_internally_valid():
    from app.validate import check_arabic, check_cross_fields, check_format

    for record in RECORDS:
        assert check_format("card_type", record["card_type"]) is None
        assert check_format("id_number", record["id_number"]) is None
        assert check_format("date_of_birth", record["date_of_birth"]) is None
        assert check_format("expiry_date", record["expiry_date"]) is None
        assert check_arabic("full_name", record["full_name_ar"]) is None
        assert check_arabic("place_of_birth", record["place_of_birth_ar"]) is None
        assert check_cross_fields(record) == {}


def test_render_returns_boxes_for_every_field(tmp_path):
    from eval.make_samples import FONT_PATH, render_card

    if not FONT_PATH.exists():
        pytest.skip("Noto Naskh Arabic not installed; see eval/make_samples.py")
    image, boxes = render_card(RECORDS[0])
    assert image.size == (1000, 630)
    assert set(boxes) == {
        "card_type", "full_name", "id_number", "date_of_birth", "expiry_date",
        "place_of_birth",
    }
