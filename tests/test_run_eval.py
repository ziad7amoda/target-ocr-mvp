import json
import sys
import urllib.error
from unittest import mock

import pytest

from app.schema import Agreement, ExtractResponse, FieldResult
from eval import run_eval
from eval.run_eval import _validate_expected, iou, main, score

EXPECTED = {
    "a.jpg": {
        "fields": {
            "card_type": "citizen", "full_name": "JOHN A SMITH", "id_number": "12345678",
            "date_of_birth": "1990-04-12", "expiry_date": "2030-04-11",
            "place_of_birth": "MUSCAT",
        },
        "boxes": {"full_name": [100, 100, 400, 140]},
    }
}


def _response(fields):
    return ExtractResponse(
        fields=fields, raw_text="{}", agreement=Agreement(matched=6, compared=6, total=6),
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


def test_box_coverage_counts_fields_with_ground_truth_but_no_returned_box():
    """box_total alone is self-selecting: a filter that drops 90% of boxes
    would still report a flattering hit rate on the remaining 10%.
    box_expected is the true denominator - it must count a field that has
    ground truth even when no box came back."""
    report = score({"a.jpg": _response(_all_ok())}, EXPECTED)
    assert report.box_expected == 1
    assert report.box_total == 0


def test_box_coverage_includes_returned_boxes_too():
    hit = _all_ok(full_name=FieldResult(value="JOHN A SMITH", status="ok", box=(105, 102, 395, 138)))
    report = score({"a.jpg": _response(hit)}, EXPECTED)
    assert report.box_expected == 1
    assert report.box_total == 1


def test_wrong_value_ar_on_an_ok_field_is_a_silent_error():
    """value_ar is the field a bank actually keys in. A wrong Arabic value
    served as `ok` must show up in the silent error count, distinguishable
    from a Latin-value error by the `_ar` suffix."""
    expected_with_ar = {
        "a.jpg": {
            "fields": {**EXPECTED["a.jpg"]["fields"], "full_name_ar": "جون سميث"},
            "boxes": EXPECTED["a.jpg"]["boxes"],
        }
    }
    wrong = _all_ok(full_name=FieldResult(value="JOHN A SMITH", value_ar="جون سميثي", status="ok"))
    report = score({"a.jpg": _response(wrong)}, expected_with_ar)
    assert len(report.silent_errors) == 1
    err = report.silent_errors[0]
    assert err.field == "full_name_ar"
    assert err.got == "جون سميثي" and err.expected == "جون سميث"


def test_correct_value_ar_is_not_a_silent_error():
    expected_with_ar = {
        "a.jpg": {
            "fields": {**EXPECTED["a.jpg"]["fields"], "full_name_ar": "جون سميث"},
            "boxes": EXPECTED["a.jpg"]["boxes"],
        }
    }
    right = _all_ok(full_name=FieldResult(value="JOHN A SMITH", value_ar="جون سميث", status="ok"))
    report = score({"a.jpg": _response(right)}, expected_with_ar)
    assert report.silent_errors == []


def test_missing_ground_truth_value_ar_is_not_scored():
    """EXPECTED carries no `_ar` ground truth for most fixtures - value_ar
    should simply be skipped, not treated as a mismatch."""
    report = score({"a.jpg": _response(_all_ok())}, EXPECTED)
    assert report.silent_errors == []


# --- --api mode -------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for what urllib.request.urlopen() returns."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _fake_extract_payload(**overrides):
    fields = {
        "card_type": {"value": "citizen", "status": "ok"},
        "full_name": {"value": None, "value_ar": "زياد حموده", "status": "ok"},
        "id_number": {"value": "12345678", "status": "ok"},
        "date_of_birth": {"value": "1990-04-12", "status": "ok"},
        "expiry_date": {"value": "2030-04-11", "status": "ok"},
        "place_of_birth": {"value": None, "value_ar": "مسقط", "status": "ok"},
    }
    payload = {
        "fields": fields,
        "raw_text": "{}",
        "agreement": {"matched": 6, "compared": 6, "total": 6},
        "elapsed_ms": 9000,
        "model": "remote",
    }
    payload.update(overrides)
    return payload


def _make_sample_dir(tmp_path, image_name="a.jpg"):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / image_name).write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    expected = {
        image_name: {
            "fields": {
                "card_type": "citizen", "id_number": "12345678",
                "date_of_birth": "1990-04-12", "expiry_date": "2030-04-11",
            },
            "boxes": {},
        }
    }
    (samples / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    return samples


def test_api_extract_parses_json_response_into_extract_response(tmp_path):
    samples = _make_sample_dir(tmp_path)
    fake = _FakeResponse(_fake_extract_payload())
    with mock.patch.object(run_eval.urllib.request, "urlopen", return_value=fake) as urlopen:
        result = run_eval.api_extract("http://localhost:8000", samples / "a.jpg")

    assert isinstance(result, ExtractResponse)
    assert result.fields["id_number"].value == "12345678"
    assert result.fields["full_name"].value_ar == "زياد حموده"
    assert result.model == "remote"
    # Multipart POST, not a bare URL string - and hits the right endpoint.
    request = urlopen.call_args.args[0]
    assert request.full_url == "http://localhost:8000/api/extract"
    assert request.get_method() == "POST"
    assert b'name="image"' in request.data


def test_api_extract_scores_like_a_local_response(tmp_path):
    """A JSON response parsed via --api must feed score() exactly like a
    local extract() call does - same downstream path, no special-casing."""
    samples = _make_sample_dir(tmp_path)
    fake = _FakeResponse(_fake_extract_payload())
    with mock.patch.object(run_eval.urllib.request, "urlopen", return_value=fake):
        result = run_eval.api_extract("http://localhost:8000", samples / "a.jpg")

    expected = json.loads((samples / "expected.json").read_text(encoding="utf-8"))
    report = score({"a.jpg": result}, expected)
    assert report.silent_errors == []
    assert report.per_field["id_number"]["exact"] == 1


def test_api_extract_unreachable_server_raises_clear_message(tmp_path):
    samples = _make_sample_dir(tmp_path)
    with mock.patch.object(
        run_eval.urllib.request, "urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        with pytest.raises(SystemExit) as exc_info:
            run_eval.api_extract("http://localhost:8000", samples / "a.jpg")

    message = str(exc_info.value)
    assert "http://localhost:8000/api/extract" in message
    assert "running" in message.lower()


def test_api_health_returns_flag_from_running_server():
    fake = _FakeResponse({"model": "x", "device": "cpu", "loaded": True, "self_consistency": False})
    with mock.patch.object(run_eval.urllib.request, "urlopen", return_value=fake) as urlopen:
        assert run_eval.api_health("http://localhost:8000") is False

    url = urlopen.call_args.args[0]
    assert url == "http://localhost:8000/api/health"


def test_api_health_unreachable_returns_none_without_raising():
    with mock.patch.object(
        run_eval.urllib.request, "urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        assert run_eval.api_health("http://localhost:8000") is None


def test_main_api_mode_never_touches_qwen_engine(tmp_path, monkeypatch):
    """--api mode must not import app.model / construct QwenEngine at all -
    it is the only thing that stands between this eval run and the OOM the
    user hit (a second model on top of an already-loaded server)."""
    samples = _make_sample_dir(tmp_path)
    # A None entry in sys.modules makes any `import app.model` raise
    # ImportError - if main() tries it in --api mode, this test fails.
    monkeypatch.setitem(sys.modules, "app.model", None)
    monkeypatch.setitem(sys.modules, "app.extract", None)

    health = _FakeResponse({"model": "x", "device": "cpu", "loaded": True, "self_consistency": True})
    extract_resp = _FakeResponse(_fake_extract_payload())

    def fake_urlopen(url_or_request, timeout=None):
        if isinstance(url_or_request, str):
            return health
        return extract_resp

    with mock.patch.object(run_eval.urllib.request, "urlopen", side_effect=fake_urlopen):
        rc = main(["--api", "http://localhost:8000", "--samples", str(samples)])

    assert rc == 0


def test_main_local_mode_still_requires_app_model(tmp_path, monkeypatch):
    """Sanity check for the test above: without --api, main() DOES need
    app.model, so a stubbed-out module should surface as a real failure
    rather than the test above passing for the wrong reason."""
    samples = _make_sample_dir(tmp_path)
    monkeypatch.setitem(sys.modules, "app.model", None)

    with pytest.raises(ImportError):
        main(["--samples", str(samples)])


# --- expected.json structural validation -------------------------------


def test_validate_expected_accepts_a_well_formed_file(tmp_path):
    samples = _make_sample_dir(tmp_path)
    expected = json.loads((samples / "expected.json").read_text(encoding="utf-8"))
    _validate_expected(expected, samples)  # must not raise


def test_validate_expected_rejects_unwrapped_single_card_entry(tmp_path):
    """The exact mistake hit in the field: a single card's {fields, boxes}
    object at the top level instead of nested under an image filename."""
    samples = _make_sample_dir(tmp_path)
    unwrapped = {"fields": {"card_type": "citizen"}, "boxes": {}}
    with pytest.raises(SystemExit) as exc_info:
        _validate_expected(unwrapped, samples)
    message = str(exc_info.value)
    assert "wrap it in an object keyed by the image filename" in message


def test_validate_expected_rejects_unknown_field_key(tmp_path):
    """Catches old-schema ground truth (nationality/sex) carried over,
    which would otherwise silently score nothing."""
    samples = _make_sample_dir(tmp_path)
    expected = {
        "a.jpg": {"fields": {"card_type": "citizen", "nationality": "OMANI"}, "boxes": {}}
    }
    with pytest.raises(SystemExit) as exc_info:
        _validate_expected(expected, samples)
    message = str(exc_info.value)
    assert "nationality" in message
    assert "card_type" in message  # names a valid key too


def test_validate_expected_rejects_missing_image_file(tmp_path):
    samples = _make_sample_dir(tmp_path)
    expected = {"does_not_exist.jpg": {"fields": {}, "boxes": {}}}
    with pytest.raises(SystemExit) as exc_info:
        _validate_expected(expected, samples)
    assert "does_not_exist.jpg" in str(exc_info.value)


def test_validate_expected_allows_partial_ground_truth(tmp_path):
    """Omitting fields is legitimate - a reviewer may only have typed some
    of them in. Only an unknown key is an error."""
    samples = _make_sample_dir(tmp_path)
    expected = {"a.jpg": {"fields": {"card_type": "citizen"}, "boxes": {}}}
    _validate_expected(expected, samples)  # must not raise
