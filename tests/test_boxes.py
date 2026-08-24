from app.boxes import filter_boxes, rescale_box

# A 1000x600 processed image against a 2000x1200 original: factor of 2.
PROC = (1000, 600)
ORIG = (2000, 1200)


def test_rescales_from_processed_space_to_original():
    assert rescale_box([100, 50, 300, 90], PROC, ORIG) == (200, 100, 600, 180)


def test_rescale_is_identity_when_sizes_match():
    assert rescale_box([10, 20, 30, 40], PROC, PROC) == (10, 20, 30, 40)


def test_plausible_box_survives():
    out = filter_boxes({"full_name": [100, 50, 400, 90]}, PROC, ORIG)
    assert out["full_name"] == (200, 100, 800, 180)


def test_inverted_box_is_dropped():
    assert filter_boxes({"full_name": [400, 90, 100, 50]}, PROC, ORIG) == {}


def test_box_far_outside_bounds_is_dropped():
    assert filter_boxes({"full_name": [100, 50, 5000, 90]}, PROC, ORIG) == {}


def test_box_slightly_outside_bounds_is_clamped_not_dropped():
    out = filter_boxes({"full_name": [100, 50, 1010, 90]}, PROC, ORIG)
    assert out["full_name"][2] == 2000


def test_full_page_box_is_dropped():
    assert filter_boxes({"full_name": [0, 0, 1000, 600]}, PROC, ORIG) == {}


def test_tall_narrow_box_is_dropped():
    """Printed field lines are wide, never tall."""
    assert filter_boxes({"full_name": [100, 50, 120, 400]}, PROC, ORIG) == {}


def test_sub_pixel_speck_is_dropped():
    assert filter_boxes({"full_name": [100, 50, 102, 52]}, PROC, ORIG) == {}


def test_identical_box_claimed_by_three_fields_is_dropped_for_all():
    """Model collapse: one box repeated for everything is not grounding."""
    box = [100, 50, 400, 90]
    out = filter_boxes(
        {"full_name": box, "card_type": list(box), "place_of_birth": list(box)}, PROC, ORIG
    )
    assert out == {}


def test_identical_box_claimed_by_two_fields_survives():
    box = [100, 50, 400, 90]
    out = filter_boxes({"full_name": box, "card_type": list(box)}, PROC, ORIG)
    assert len(out) == 2


def test_unknown_field_names_are_discarded():
    out = filter_boxes({"eye_colour": [100, 50, 400, 90]}, PROC, ORIG)
    assert out == {}
