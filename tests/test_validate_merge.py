from app.schema import CardFields
from app.validate import merge_passes

# Revision R3 (see docs/superpowers/specs/2026-08-24-revision-real-card-findings.md):
# fixture values drawn from the actual resident card that motivated this
# revision - full_name and place_of_birth are Arabic-only, so GOOD carries
# `full_name_ar` / `place_of_birth_ar` and no Latin counterpart.
GOOD = dict(
    card_type="resident",
    full_name_ar="زياد نشأت عبد الحى ابو الوفا حموده",
    id_number="70011864",
    date_of_birth="2002-09-29",
    expiry_date="2027-01-25",
    place_of_birth_ar="جمهورية مصر العربية",
)


def test_agreeing_valid_passes_are_all_ok():
    fields, agreement = merge_passes(CardFields(**GOOD), CardFields(**GOOD))
    assert {f.status for f in fields.values()} == {"ok"}
    assert agreement.matched == 6 and agreement.compared == 6 and agreement.total == 6


def test_null_in_both_passes_is_missing():
    blank = CardFields(**{**GOOD, "id_number": None})
    fields, _ = merge_passes(blank, blank)
    assert fields["id_number"].status == "missing"
    assert fields["id_number"].value is None


def test_a_missing_field_lowers_compared_but_not_matched():
    """The Agreement tile must not read a missing field as a disagreement:
    compared should exclude it while matched still equals compared for the
    fields that were actually checked."""
    blank = CardFields(**{**GOOD, "id_number": None})
    fields, agreement = merge_passes(blank, blank)
    assert fields["id_number"].status == "missing"
    assert agreement.compared == 5 < agreement.total == 6
    assert agreement.matched == agreement.compared == 5


def test_disagreement_is_review():
    a = CardFields(**GOOD)
    b = CardFields(**{**GOOD, "id_number": "70011865"})
    fields, agreement = merge_passes(a, b)
    assert fields["id_number"].status == "review"
    assert fields["id_number"].reason == "passes disagreed"
    assert agreement.matched == 5
    assert agreement.compared == 6  # the field was compared - it just disagreed


def test_primary_value_is_kept_when_passes_disagree():
    a = CardFields(**GOOD)
    b = CardFields(**{**GOOD, "id_number": "70011865"})
    fields, _ = merge_passes(a, b)
    assert fields["id_number"].value == "70011864"


def test_null_in_one_pass_only_is_a_disagreement_not_missing():
    a = CardFields(**GOOD)
    b = CardFields(**{**GOOD, "id_number": None})
    fields, _ = merge_passes(a, b)
    assert fields["id_number"].status == "review"


def test_disagreement_outranks_a_format_failure():
    """Spec §5: rule 2 precedes rule 3. Disagreement is the only signal that
    catches hallucination, so it must be the reason a reviewer sees."""
    a = CardFields(**{**GOOD, "id_number": "1234567"})
    b = CardFields(**{**GOOD, "id_number": "9999999"})
    fields, _ = merge_passes(a, b)
    assert fields["id_number"].status == "review"
    assert fields["id_number"].reason == "passes disagreed"


def test_format_failure_on_agreeing_passes_is_review():
    bad = CardFields(**{**GOOD, "id_number": "1234"})
    fields, _ = merge_passes(bad, bad)
    assert fields["id_number"].status == "review"
    assert "8 digits" in fields["id_number"].reason


def test_cross_field_failure_is_review_on_both_dates():
    bad = CardFields(**{**GOOD, "date_of_birth": "2030-04-11", "expiry_date": "1990-04-12"})
    fields, _ = merge_passes(bad, bad)
    assert fields["date_of_birth"].status == "review"
    assert fields["expiry_date"].status == "review"


def test_arabic_disagreement_flags_the_field():
    a = CardFields(**GOOD)
    b = CardFields(**{**GOOD, "full_name_ar": "زياد نشأت عبد الحى ابو الوفا سالم"})
    fields, _ = merge_passes(a, b)
    assert fields["full_name"].status == "review"


def test_arabic_is_only_populated_for_arabic_fields():
    fields, _ = merge_passes(CardFields(**GOOD), CardFields(**GOOD))
    assert fields["full_name"].value_ar == GOOD["full_name_ar"]
    assert fields["id_number"].value_ar is None


def test_comparison_ignores_case_and_surrounding_whitespace():
    a = CardFields(**GOOD)
    b = CardFields(**{**GOOD, "full_name_ar": f"  {GOOD['full_name_ar']}  "})
    fields, _ = merge_passes(a, b)
    assert fields["full_name"].status == "ok"


def test_missing_secondary_downgrades_everything_to_review():
    fields, agreement = merge_passes(CardFields(**GOOD), None)
    assert {f.status for f in fields.values()} == {"review"}
    assert fields["place_of_birth"].reason == "consistency pass unavailable"
    assert agreement.matched == 0
    # Nothing was ever compared, not "everything disagreed": the UI must
    # render this as "not measured" rather than 0/6.
    assert agreement.compared == 0


def test_missing_secondary_still_reports_null_fields_as_missing():
    """Spec §4.3: a field the model could not read does not become `review`
    merely because a second opinion was unavailable."""
    card = CardFields(**{**GOOD, "id_number": None})
    fields, _ = merge_passes(card, None)
    assert fields["id_number"].status == "missing"
    assert fields["place_of_birth"].status == "review"


def test_arabic_only_field_disagreement_is_detected():
    """full_name and place_of_birth have no Latin value. If agreement compared
    only `value`, both would be None on every card, every field would trivially
    agree, and the hallucination check would be silently dead for the two most
    important fields."""
    a = CardFields(**GOOD)
    b = CardFields(**{**GOOD, "full_name_ar": "اسم مختلف تماما"})
    fields, agreement = merge_passes(a, b)
    assert fields["full_name"].status == "review"
    assert fields["full_name"].reason == "passes disagreed"


def test_arabic_only_field_carries_value_ar_not_value():
    fields, _ = merge_passes(CardFields(**GOOD), CardFields(**GOOD))
    assert fields["full_name"].value is None
    assert fields["full_name"].value_ar is not None
    assert fields["id_number"].value is not None
    assert fields["id_number"].value_ar is None


def test_single_component_name_is_flagged_as_truncated():
    """A real card's name has many components; one token means truncation."""
    card = CardFields(**{**GOOD, "full_name_ar": "زياد"})
    fields, _ = merge_passes(card, card)
    assert fields["full_name"].status == "review"


# --- disagreement where only one pass read anything ------------------------
# Observed on a real resident card: place_of_birth came back `review`,
# "passes disagreed", with value and value_ar both null. The card plainly
# prints جمهورية مصر العربية and the consistency pass had read it - the
# result told a reviewer that two readings disagreed and then showed them
# nothing to review, because the reading that succeeded was discarded in
# favour of the one that failed.


def test_a_disagreement_shows_the_pass_that_actually_read_something():
    a = CardFields(place_of_birth_ar=None)
    b = CardFields(place_of_birth_ar="جمهورية مصر العربية")
    fields, _ = merge_passes(a, b)
    assert fields["place_of_birth"].value_ar == "جمهورية مصر العربية"
    assert fields["place_of_birth"].status == "review"


def test_it_works_in_either_direction():
    """Nothing privileges the primary pass here - whichever one read the
    value is the one worth showing."""
    a = CardFields(place_of_birth_ar="جمهورية مصر العربية")
    b = CardFields(place_of_birth_ar=None)
    fields, _ = merge_passes(a, b)
    assert fields["place_of_birth"].value_ar == "جمهورية مصر العربية"


def test_a_one_sided_read_says_so_rather_than_claiming_disagreement():
    """"Passes disagreed" describes two different readings. One reading and
    one blank is a weaker thing and a reviewer should be told which it is."""
    fields, _ = merge_passes(CardFields(id_number=None), CardFields(id_number="70011864"))
    assert "one" in fields["id_number"].reason


def test_two_real_but_different_readings_still_prefer_the_primary():
    """The existing contract for a genuine disagreement is unchanged."""
    a = CardFields(id_number="70011864")
    b = CardFields(id_number="70011865")
    fields, _ = merge_passes(a, b)
    assert fields["id_number"].value == "70011864"
    assert fields["id_number"].reason == "passes disagreed"


def test_a_review_never_carries_an_empty_value():
    """The invariant the bug broke: `review` means a human should look at
    something, so there has to be something to look at. Null in both passes
    is `missing`, which is a different status with a different meaning."""
    for a_val, b_val in (("X", None), (None, "X"), ("X", "Y")):
        fields, _ = merge_passes(CardFields(id_number=a_val), CardFields(id_number=b_val))
        result = fields["id_number"]
        if result.status == "review":
            assert result.value is not None or result.value_ar is not None
