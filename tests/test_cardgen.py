"""Tests for the synthetic card generator.

The generator's real risk is not that it crashes. It is that it quietly
produces a corpus that looks fine in a thumbnail and teaches the model
something false: Arabic drawn as empty boxes, or ground truth written in
presentation forms that no real card and no real label will ever contain.
Both failures are invisible without a check, so both are checked here.
"""

import random

import pytest

from app.parsing import parse_card_json
from app.validate import check_arabic, check_format
from scripts.cardgen.content import generate as generate_content
from scripts.cardgen.degrade import degrade
from scripts.cardgen.fonts import missing_characters, usable
from scripts.cardgen.render import available_arabic_fonts, render
from scripts.cardgen.text import contains_arabic, prepare, shape_for_display

PRESENTATION_FORMS = range(0xFE70, 0xFF00)


def _content(seed=0, tries=1):
    rng = random.Random(seed)
    for _ in range(tries - 1):
        generate_content(rng)
    return generate_content(rng)


# --- text shaping ----------------------------------------------------------


def test_shaping_changes_arabic():
    """Without this the whole module is a no-op and Arabic renders unjoined
    and reversed, which is worse training data than none."""
    assert shape_for_display("زياد حموده") != "زياد حموده"


def test_shaping_emits_presentation_forms():
    shaped = shape_for_display("زياد")
    assert any(ord(c) in PRESENTATION_FORMS for c in shaped)


def test_latin_is_left_alone():
    assert prepare("RESIDENT CARD") == "RESIDENT CARD"
    assert not contains_arabic("RESIDENT CARD")


# --- fonts -----------------------------------------------------------------


def test_font_check_rejects_a_font_without_presentation_forms():
    """Simplified Arabic ships with Windows, covers the base Arabic block,
    and renders و and ر as empty boxes once the reshaper has run. It is the
    exact trap this check exists for, so it is pinned as a fixture."""
    simplified = r"C:\Windows\Fonts\simpo.ttf"
    try:
        missing = missing_characters(simplified)
    except Exception:
        pytest.skip("Simplified Arabic not installed on this machine")
    assert missing, "expected simpo.ttf to be missing presentation forms"
    assert not usable(simplified)


def test_at_least_one_usable_arabic_font_is_available():
    assert available_arabic_fonts()


# --- ground truth ----------------------------------------------------------


def test_ground_truth_is_logical_order_not_display_order():
    """The single most damaging thing this generator could do.

    Text is reshaped on its way to the renderer, and the shaped form is
    visually correct but is NOT the text - its characters are presentation
    forms in visual order. If that leaked into a label, every training pair
    would teach the model to output something no real card contains and no
    downstream consumer can match.
    """
    content, _ = _content(seed=3)
    for value in (content.full_name_ar, content.place_of_birth_ar):
        assert not any(ord(c) in PRESENTATION_FORMS for c in value)


def test_ground_truth_parses_as_the_production_schema():
    content, _ = _content(seed=5)
    import json

    card = parse_card_json(json.dumps(content.ground_truth(), ensure_ascii=False))
    assert card.full_name_ar == content.full_name_ar
    assert card.id_number == content.id_number


def test_ground_truth_passes_the_production_validators():
    """A generated label that its own validators would flag for review is a
    label that teaches the model to produce flagged output."""
    rng = random.Random(11)
    for _ in range(40):
        content, _ = generate_content(rng)
        truth = content.ground_truth()
        for field in ("card_type", "id_number", "date_of_birth", "expiry_date"):
            assert check_format(field, truth[field]) is None, (field, truth[field])
        assert check_arabic("full_name", truth["full_name_ar"]) is None
        assert check_arabic("place_of_birth", truth["place_of_birth_ar"]) is None


def test_dates_are_ordered_and_plausible():
    rng = random.Random(2)
    for _ in range(50):
        content, printed = generate_content(rng)
        assert content.expiry_date > content.date_of_birth
        # The card prints DD/MM/YYYY; the label is ISO. Confusing the two is
        # what produced 8 November for a card printed 11/08.
        day, month, year = printed["date_of_birth"].split("/")
        assert content.date_of_birth == f"{year}-{month}-{day}"


def test_names_are_weighted_towards_the_long_end():
    """Truncation showed up on a seven-component name. A uniform sample
    would spend most of the corpus where the model already succeeds."""
    rng = random.Random(9)
    counts = [len(generate_content(rng)[0].full_name_ar.split()) for _ in range(400)]
    assert sum(c >= 6 for c in counts) / len(counts) > 0.5


def test_occupation_is_printed_on_resident_cards_only():
    rng = random.Random(13)
    for _ in range(60):
        content, _ = generate_content(rng)
        if content.card_type == "citizen":
            assert content.occupation_ar is None
        else:
            assert content.occupation_ar


def test_occupation_is_never_part_of_the_label():
    """المهنة is printed but deliberately not extracted (spec §7)."""
    content, _ = _content(seed=1)
    assert "occupation" not in " ".join(content.ground_truth())


# --- rendering and degradation ---------------------------------------------


def test_renders_a_card_of_the_expected_size():
    from scripts.cardgen.layout import CARD_H, CARD_W

    rng = random.Random(4)
    content, printed = generate_content(rng)
    assert render(content, printed, rng).size == (CARD_W, CARD_H)


def test_degradation_actually_changes_the_image():
    rng = random.Random(6)
    content, printed = generate_content(rng)
    card = render(content, printed, rng)
    dirty = degrade(card, rng)
    assert dirty.size != card.size or list(dirty.getdata()) != list(card.getdata())


def test_generation_is_reproducible_from_a_seed():
    """A corpus you cannot regenerate is one you cannot debug."""
    a, _ = _content(seed=99, tries=3)
    b, _ = _content(seed=99, tries=3)
    assert a == b
