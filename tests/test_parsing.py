import pytest

from app.parsing import ParseError, parse_boxes, parse_card_json, strip_fences


def test_strips_json_fences():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strips_bare_fences():
    assert strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strips_prose_before_and_after():
    text = 'Here is the data:\n{"a": 1}\nHope that helps!'
    assert strip_fences(text) == '{"a": 1}'


def test_leaves_clean_json_untouched():
    assert strip_fences('{"a": 1}') == '{"a": 1}'


def test_parses_a_full_card():
    raw = (
        '{"full_name": "JOHN A SMITH", "full_name_ar": "جون سميث", '
        '"id_number": "12345678", "date_of_birth": "1990-04-12", '
        '"expiry_date": "2030-04-11", "nationality": "OMANI", '
        '"nationality_ar": "عماني", "sex": "M", "sex_ar": "ذكر"}'
    )
    card = parse_card_json(raw)
    assert card.full_name == "JOHN A SMITH"
    assert card.sex_ar == "ذكر"


def test_parses_fenced_output_with_nulls():
    card = parse_card_json('```json\n{"full_name": null, "id_number": "12345678"}\n```')
    assert card.full_name is None
    assert card.id_number == "12345678"


def test_coerces_a_numeric_id_number_to_string():
    """Models sometimes emit an unquoted number. That is not a reason to fail
    the whole extraction; it is unambiguous."""
    assert parse_card_json('{"id_number": 12345678}').id_number == "12345678"


def test_raises_on_truncated_json():
    with pytest.raises(ParseError):
        parse_card_json('{"full_name": "JOHN')


def test_raises_on_no_json_at_all():
    with pytest.raises(ParseError):
        parse_card_json("I cannot read this card.")


def test_raises_on_unknown_key():
    with pytest.raises(ParseError):
        parse_card_json('{"full_name": "X", "eye_colour": "brown"}')


def test_parses_boxes():
    raw = '{"full_name": [10, 20, 300, 60], "sex": [10, 90, 60, 120]}'
    assert parse_boxes(raw) == {"full_name": [10, 20, 300, 60], "sex": [10, 90, 60, 120]}


def test_parses_boxes_wrapped_in_bbox_2d():
    raw = '{"full_name": {"bbox_2d": [10, 20, 300, 60]}}'
    assert parse_boxes(raw) == {"full_name": [10, 20, 300, 60]}


def test_ignores_box_entries_that_are_not_four_numbers():
    raw = '{"full_name": [10, 20, 300], "sex": [10, 90, 60, 120]}'
    assert parse_boxes(raw) == {"sex": [10, 90, 60, 120]}


def test_box_parsing_never_raises_on_garbage():
    """Grounding is optional (spec §4.3): its failure must not fail extraction."""
    assert parse_boxes("no boxes here") == {}


def test_boolean_value_is_a_type_mismatch_not_a_number():
    """bool subclasses int, so a naive isinstance(v, (int, float)) coerces
    True into the string "True" - a confident-looking wrong answer, which is
    exactly what this module exists to prevent."""
    with pytest.raises(ParseError):
        parse_card_json('{"full_name": true}')
    with pytest.raises(ParseError):
        parse_card_json('{"sex": false}')


def test_parse_boxes_returns_empty_for_non_string_input():
    """The never-raises contract is unconditional."""
    assert parse_boxes(None) == {}
    assert parse_boxes(123) == {}


def test_empty_and_whitespace_values_normalise_to_none():
    """Documented in the module docstring but previously untested. An empty
    string must not survive as a value that later reads as a real answer."""
    card = parse_card_json('{"full_name": "", "id_number": "   "}')
    assert card.full_name is None
    assert card.id_number is None
