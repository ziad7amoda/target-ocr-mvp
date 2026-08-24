from app.schema import Agreement, ExtractResponse, FieldResult
from eval.run_eval import iou, score

EXPECTED = {
    "a.jpg": {
        "fields": {
            "full_name": "JOHN A SMITH", "id_number": "12345678",
            "date_of_birth": "1990-04-12", "expiry_date": "2030-04-11",
            "nationality": "OMANI", "sex": "M",
        },
        "boxes": {"full_name": [100, 100, 400, 140]},
    }
}


def _response(fields):
    return ExtractResponse(
        fields=fields, raw_text="{}", agreement=Agreement(matched=6, total=6),
        elapsed_ms=100, model="fake",
    )


def _all_ok(**overrides):
    base = {k: FieldResult(value=v, status="ok") for k, v in EXPECTED["a.jpg"]["fields"].items()}
    base.update(overrides)
    return base


def test_a_perfect_run_has_no_silent_errors():
    report = score({"a.jpg": _response(_all_ok())}, EXPECTED)
    assert report.silent_errors == []
    assert report.per_field["id_number"]["exact"] == 1


def test_an_ok_field_with_a_wrong_value_is_a_silent_error():
    """The metric that matters most: the model was confident and wrong, and
    nothing caught it."""
    wrong = _all_ok(id_number=FieldResult(value="12345679", status="ok"))
    report = score({"a.jpg": _response(wrong)}, EXPECTED)
    assert len(report.silent_errors) == 1
    err = report.silent_errors[0]
    assert err.field == "id_number" and err.got == "12345679" and err.expected == "12345678"


def test_a_reviewed_wrong_field_is_not_a_silent_error():
    """It was flagged. The system did its job; a human will catch it."""
    flagged = _all_ok(
        id_number=FieldResult(value="12345679", status="review", reason="passes disagreed")
    )
    report = score({"a.jpg": _response(flagged)}, EXPECTED)
    assert report.silent_errors == []
    assert report.per_field["id_number"]["review"] == 1


def test_a_missing_field_is_counted_but_is_not_a_silent_error():
    blanked = _all_ok(id_number=FieldResult(value=None, status="missing"))
    report = score({"a.jpg": _response(blanked)}, EXPECTED)
    assert report.silent_errors == []
    assert report.per_field["id_number"]["missing"] == 1
    assert report.per_field["id_number"]["exact"] == 0


def test_comparison_ignores_case_and_whitespace():
    lenient = _all_ok(full_name=FieldResult(value="  john a smith ", status="ok"))
    assert score({"a.jpg": _response(lenient)}, EXPECTED).per_field["full_name"]["exact"] == 1


def test_iou_of_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_box_within_tolerance_counts_as_a_hit():
    hit = _all_ok(full_name=FieldResult(value="JOHN A SMITH", status="ok", box=(105, 102, 395, 138)))
    report = score({"a.jpg": _response(hit)}, EXPECTED)
    assert report.box_hits == 1 and report.box_total == 1


def test_badly_placed_box_is_a_miss():
    miss = _all_ok(full_name=FieldResult(value="JOHN A SMITH", status="ok", box=(600, 400, 700, 440)))
    report = score({"a.jpg": _response(miss)}, EXPECTED)
    assert report.box_hits == 0 and report.box_total == 1


def test_absent_box_is_not_counted_against_the_hit_rate():
    """A box the filter dropped is a non-answer, not a wrong answer."""
    report = score({"a.jpg": _response(_all_ok())}, EXPECTED)
    assert report.box_total == 0
