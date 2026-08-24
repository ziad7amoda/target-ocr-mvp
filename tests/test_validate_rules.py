from app.validate import check_arabic, check_cross_fields, check_format


def test_valid_id_number_passes():
    assert check_format("id_number", "12345678") is None


def test_id_number_of_wrong_length_is_flagged():
    reason = check_format("id_number", "1234567")
    assert reason is not None and "8 digits" in reason


def test_id_number_with_letters_is_flagged():
    assert check_format("id_number", "1234567A") is not None


def test_valid_iso_date_passes():
    assert check_format("date_of_birth", "1990-04-12") is None


def test_non_iso_date_is_flagged():
    reason = check_format("date_of_birth", "12/04/1990")
    assert reason is not None and "ISO" in reason


def test_impossible_date_is_flagged():
    assert check_format("date_of_birth", "1990-02-31") is not None


def test_out_of_range_year_is_flagged():
    assert check_format("expiry_date", "1832-01-01") is not None


def test_valid_card_type_values_pass():
    assert check_format("card_type", "citizen") is None
    assert check_format("card_type", "resident") is None


def test_unexpected_card_type_is_flagged():
    reason = check_format("card_type", "national")
    assert reason is not None and "citizen or resident" in reason


def test_null_card_type_passes_format_check():
    """Nulls are handled by the missing rule, not by format rules."""
    assert check_format("card_type", None) is None


def test_null_value_is_not_a_format_failure():
    """Nulls are handled by the missing rule, not by format rules."""
    assert check_format("id_number", None) is None


def test_arabic_field_with_arabic_script_passes():
    assert check_arabic("full_name", "زياد نشأت عبد الحى ابو الوفا حموده") is None


def test_arabic_field_containing_only_latin_is_flagged():
    reason = check_arabic("full_name", "JOHN SMITH")
    assert reason is not None and "Arabic" in reason


def test_arabic_check_ignores_non_arabic_fields():
    assert check_arabic("id_number", "12345678") is None


def test_arabic_check_ignores_null_value():
    assert check_arabic("full_name", None) is None


def test_short_arabic_value_is_flagged():
    reason = check_arabic("place_of_birth", "م")
    assert reason is not None and "short" in reason


def test_single_component_name_is_flagged():
    reason = check_arabic("full_name", "زياد")
    assert reason is not None and "component" in reason


def test_multi_component_name_passes():
    assert check_arabic("full_name", "زياد نشأت") is None


def test_single_component_place_of_birth_is_not_flagged_for_component_count():
    """The two-component check is specific to full_name - place_of_birth
    values like "مسقط" are legitimately a single word."""
    assert check_arabic("place_of_birth", "مسقط") is None


def test_expiry_before_birth_is_flagged_on_both_fields():
    out = check_cross_fields({"date_of_birth": "2030-04-11", "expiry_date": "1990-04-12"})
    assert "date_of_birth" in out and "expiry_date" in out


def test_future_date_of_birth_is_flagged():
    out = check_cross_fields({"date_of_birth": "2090-01-01"})
    assert "date_of_birth" in out


def test_implausible_age_is_flagged():
    out = check_cross_fields({"date_of_birth": "1850-01-01"})
    assert "date_of_birth" in out


def test_consistent_dates_pass():
    assert check_cross_fields({"date_of_birth": "1990-04-12", "expiry_date": "2030-04-11"}) == {}


def test_cross_checks_skip_unparseable_values():
    """Format rules already flagged these; cross-checks must not crash on them."""
    assert check_cross_fields({"date_of_birth": "not-a-date", "expiry_date": "2030-04-11"}) == {}


def test_basic_format_date_without_separators_is_rejected():
    """date.fromisoformat accepts "19900412" on Python 3.11+, but the contract
    is YYYY-MM-DD and a separator-less date reaching a reviewer as `ok` is
    exactly the silent-wrong-value this module exists to prevent."""
    assert check_format("date_of_birth", "19900412") is not None


def test_iso_week_date_is_rejected():
    assert check_format("expiry_date", "2024-W01-1") is not None
