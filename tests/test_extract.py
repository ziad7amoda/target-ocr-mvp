import json

import pytest
from PIL import Image

from app.config import Settings
from app.extract import extract
from app.model import FakeEngine

# Realistic fixture from the resident card in
# docs/superpowers/specs/2026-08-24-revision-real-card-findings.md.
GOOD = {
    "card_type": "resident",
    "full_name_ar": "زياد نشأت عبد الحى ابو الوفا حموده",
    "id_number": "70011864",
    "date_of_birth": "2002-09-29",
    "expiry_date": "2027-01-25",
    "place_of_birth_ar": "جمهورية مصر العربية",
}
BOXES = {"full_name": [10, 10, 300, 50], "card_type": [10, 70, 60, 100]}


def _img():
    return Image.new("RGB", (1000, 600), "white")


def _replies(fields=None):
    """Two-sequence batch (primary + consistency), the default with
    SHOW_BOXES=False."""
    payload = json.dumps(fields if fields is not None else GOOD)
    return [payload, payload]


def _replies_with_boxes(fields=None, boxes=None):
    """Three-sequence batch (primary + consistency + grounding), for tests
    that explicitly enable SHOW_BOXES."""
    payload = json.dumps(fields if fields is not None else GOOD)
    return [payload, payload, json.dumps(boxes if boxes is not None else BOXES)]


def test_issues_exactly_one_batched_call_of_two():
    """Spec D5 (as revised, R7): self-consistency rides along in one
    generate(); grounding is off by default so the normal batch is two
    sequences, not three."""
    engine = FakeEngine(_replies())
    extract(_img(), engine, Settings())
    assert len(engine.calls) == 1
    assert len(engine.calls[0]) == 2


def test_second_request_uses_a_contrast_normalised_image():
    engine = FakeEngine(_replies())
    extract(_img(), engine, Settings())
    a, b = engine.calls[0]
    assert a.image is not b.image


def test_grounding_request_uses_the_grounding_prompt_when_boxes_enabled():
    engine = FakeEngine(_replies_with_boxes())
    extract(_img(), engine, Settings(SHOW_BOXES=True))
    assert "bbox_2d" in engine.calls[0][2].prompt


def test_returns_ok_fields_for_agreeing_passes():
    resp = extract(_img(), FakeEngine(_replies()), Settings())
    assert resp.fields["id_number"].status == "ok"
    assert resp.agreement.matched == 6


def test_attaches_filtered_boxes():
    resp = extract(_img(), FakeEngine(_replies_with_boxes()), Settings(SHOW_BOXES=True))
    assert resp.fields["full_name"].box is not None
    assert resp.fields["id_number"].box is None


def test_show_boxes_false_suppresses_every_box():
    resp = extract(_img(), FakeEngine(_replies()), Settings(SHOW_BOXES=False))
    assert all(f.box is None for f in resp.fields.values())


def test_self_consistency_false_issues_one_request_and_reviews_everything():
    engine = FakeEngine([json.dumps(GOOD)])
    resp = extract(_img(), engine, Settings(SELF_CONSISTENCY=False, SHOW_BOXES=False))
    assert len(engine.calls[0]) == 1
    assert resp.fields["place_of_birth"].status == "review"
    assert resp.fields["place_of_birth"].reason == "consistency pass unavailable"


def test_raw_text_carries_the_primary_pass_output():
    """Spec D3: raw_text is the model's own echo, not a transcription."""
    engine = FakeEngine(_replies())
    resp = extract(_img(), engine, Settings())
    assert "70011864" in resp.raw_text


def test_malformed_primary_triggers_exactly_one_retry():
    engine = FakeEngine(["not json", json.dumps(GOOD), json.dumps(GOOD)])
    resp = extract(_img(), engine, Settings())
    assert len(engine.calls) == 2
    assert len(engine.calls[1]) == 1
    assert resp.fields["id_number"].status == "ok"


def test_retry_prompt_feeds_the_parse_error_back():
    engine = FakeEngine(["not json", json.dumps(GOOD), json.dumps(GOOD)])
    extract(_img(), engine, Settings())
    assert "not valid JSON" in engine.calls[1][0].prompt


def test_two_failures_yield_all_missing_and_never_guess():
    """Spec §4.3: a second failure returns raw text with everything missing
    rather than salvaging a partial answer."""
    engine = FakeEngine(["garbage", json.dumps(GOOD), "still garbage"])
    resp = extract(_img(), engine, Settings())
    assert {f.status for f in resp.fields.values()} == {"missing"}
    assert all(f.value is None for f in resp.fields.values())
    assert "still garbage" in resp.raw_text


def test_failed_secondary_downgrades_to_review_without_failing_extraction():
    engine = FakeEngine([json.dumps(GOOD), "garbage", "garbage again"])
    resp = extract(_img(), engine, Settings())
    assert all(f.status == "review" for f in resp.fields.values())
    assert resp.fields["id_number"].value == GOOD["id_number"]


def test_failed_grounding_leaves_extraction_intact():
    engine = FakeEngine([json.dumps(GOOD), json.dumps(GOOD), "no boxes for you"])
    resp = extract(_img(), engine, Settings(SHOW_BOXES=True))
    assert resp.fields["full_name"].status == "ok"
    assert resp.fields["full_name"].box is None


def test_reports_elapsed_and_model():
    resp = extract(_img(), FakeEngine(_replies()), Settings())
    assert resp.elapsed_ms >= 0
    assert resp.model == "fake"


def test_secondary_retry_rereads_the_contrast_image_not_the_original():
    """If pass B's retry re-read the original image, greedy decoding would make
    the two passes agree by construction, rule 2 could never fire, and every
    field would be served `ok` unchecked - the exact hole the design closes."""
    engine = FakeEngine([json.dumps(GOOD), "not json", json.dumps(GOOD)])
    img = _img()
    extract(img, engine, Settings())
    retry_request = engine.calls[1][0]
    assert retry_request.image is not img
    assert retry_request.image is engine.calls[0][1].image


@pytest.mark.parametrize(
    "self_consistency,show_boxes,expected_batch_size",
    [
        (True, True, 3),
        (True, False, 2),
        (False, True, 2),
        (False, False, 1),
    ],
)
def test_batch_composition_across_settings(self_consistency, show_boxes, expected_batch_size):
    """Pins the batch size for all four combinations of SELF_CONSISTENCY x
    SHOW_BOXES so a future change to the gating logic cannot silently
    reintroduce the always-three-sequences behaviour or drop a needed one."""
    engine = FakeEngine(lambda reqs: [json.dumps(GOOD)] * len(reqs))
    extract(_img(), engine, Settings(SELF_CONSISTENCY=self_consistency, SHOW_BOXES=show_boxes))
    assert len(engine.calls[0]) == expected_batch_size


def test_batch_is_two_sequences_when_boxes_are_disabled():
    engine = FakeEngine([json.dumps(GOOD), json.dumps(GOOD)])
    extract(_img(), engine, Settings(SHOW_BOXES=False))
    assert len(engine.calls[0]) == 2


def test_grounding_reply_is_not_confused_with_pass_b_when_boxes_disabled():
    """replies[-1] is pass B's JSON when no grounding request was issued.
    Reading it as boxes would break the consistency check silently."""
    engine = FakeEngine([json.dumps(GOOD), json.dumps(GOOD)])
    resp = extract(_img(), engine, Settings(SHOW_BOXES=False))
    assert all(f.box is None for f in resp.fields.values())
    assert resp.fields["id_number"].status == "ok"
