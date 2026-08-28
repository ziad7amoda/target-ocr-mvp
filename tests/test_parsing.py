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
        '{"card_type": "citizen", "full_name_ar": "جون سميث", '
        '"id_number": "12345678", "date_of_birth": "1990-04-12", '
        '"expiry_date": "2030-04-11", "place_of_birth_ar": "مسقط"}'
    )
    card = parse_card_json(raw)
    assert card.card_type == "citizen"
    assert card.full_name_ar == "جون سميث"


def test_parses_fenced_output_with_nulls():
    card = parse_card_json('```json\n{"full_name_ar": null, "id_number": "12345678"}\n```')
    assert card.full_name_ar is None
    assert card.id_number == "12345678"


def test_coerces_a_numeric_id_number_to_string():
    """Models sometimes emit an unquoted number. That is not a reason to fail
    the whole extraction; it is unambiguous."""
    assert parse_card_json('{"id_number": 12345678}').id_number == "12345678"


def test_raises_on_truncated_json():
    with pytest.raises(ParseError):
        parse_card_json('{"full_name_ar": "JOHN')


def test_raises_on_no_json_at_all():
    with pytest.raises(ParseError):
        parse_card_json("I cannot read this card.")


def test_an_unknown_key_no_longer_fails_the_whole_card():
    """Reversal of the original contract, deliberately.

    Rejecting the object because of one unrecognised key threw away five
    good values to punish a sixth nobody asked for, and that is what made
    every non-default model report six `missing` fields. The unknown key is
    dropped; the recognised ones survive. Output with NOTHING recognisable
    in it still raises - see test_raises_when_no_key_resembles_a_card_field.
    """
    assert parse_card_json('{"full_name_ar": "X", "eye_colour": "brown"}').full_name_ar == "X"


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
        parse_card_json('{"full_name_ar": true}')
    with pytest.raises(ParseError):
        parse_card_json('{"card_type": false}')


def test_parse_boxes_returns_empty_for_non_string_input():
    """The never-raises contract is unconditional."""
    assert parse_boxes(None) == {}
    assert parse_boxes(123) == {}


def test_empty_and_whitespace_values_normalise_to_none():
    """Documented in the module docstring but previously untested. An empty
    string must not survive as a value that later reads as a real answer."""
    card = parse_card_json('{"full_name_ar": "", "id_number": "   "}')
    assert card.full_name_ar is None
    assert card.id_number is None


# --- Key tolerance -------------------------------------------------------
#
# Every model other than Qwen2.5-VL-3B was returning `missing` for every
# field. Root cause: CardFields sets extra="forbid", so ONE unrequested key
# ("nationality") or ONE renamed key ("full_name" instead of "full_name_ar")
# raised ValidationError, the retry produced the same shape, and extract()
# fell through to _all_missing(). The six values the model DID read were
# discarded because of a key the prompt never asked for.


def test_ignores_an_extra_key_the_prompt_never_asked_for():
    """Dropping a key we do not want must not discard the ones we do.

    `nationality` and `sex` were removed in the R1 revision precisely
    because they are not printed on the card; a model volunteering them is
    the expected case, not a parse failure."""
    card = parse_card_json(
        '{"card_type": "citizen", "full_name_ar": "جون سميث", '
        '"id_number": "12345678", "nationality": "Oman", "sex": "M"}'
    )
    assert card.full_name_ar == "جون سميث"
    assert card.id_number == "12345678"


def test_accepts_the_unsuffixed_key_for_an_arabic_only_field():
    """`full_name_ar`/`place_of_birth_ar` carry a suffix the field's own
    name does not. Models routinely emit the plain name instead."""
    card = parse_card_json(
        '{"full_name": "جون سميث", "place_of_birth": "مسقط"}'
    )
    assert card.full_name_ar == "جون سميث"
    assert card.place_of_birth_ar == "مسقط"


def test_accepts_common_key_spellings():
    card = parse_card_json(
        '{"Card Type": "citizen", "fullName": "جون سميث", '
        '"civil_number": "12345678", "DOB": "1990-04-12", '
        '"expiry": "2030-04-11", "birthPlace": "مسقط"}'
    )
    assert card.card_type == "citizen"
    assert card.full_name_ar == "جون سميث"
    assert card.id_number == "12345678"
    assert card.date_of_birth == "1990-04-12"
    assert card.expiry_date == "2030-04-11"
    assert card.place_of_birth_ar == "مسقط"


def test_canonical_key_wins_over_an_alias():
    card = parse_card_json('{"full_name_ar": "جون", "full_name": "JOHN SMITH"}')
    assert card.full_name_ar == "جون"


def test_conflicting_aliases_yield_null_rather_than_a_guess():
    """Two aliases, two different values, no way to tell which is the card's.
    Spec §4.3: return nothing rather than pick one."""
    card = parse_card_json('{"full_name": "جون", "name": "أحمد"}')
    assert card.full_name_ar is None


def test_unwraps_a_single_key_wrapper_object():
    card = parse_card_json('{"fields": {"id_number": "12345678"}}')
    assert card.id_number == "12345678"


def test_unwraps_a_single_element_list():
    card = parse_card_json('[{"id_number": "12345678"}]')
    assert card.id_number == "12345678"


def test_raises_when_no_key_resembles_a_card_field():
    """Tolerance is not credulity: output with nothing recognisable in it is
    still a parse failure, not a card with six null fields."""
    with pytest.raises(ParseError):
        parse_card_json('{"eye_colour": "brown", "height_cm": 180}')
