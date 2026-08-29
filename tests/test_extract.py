import json

import pytest
from PIL import Image

from app.config import Settings
from app.extract import (
    FIELD_PROMPT_NATURAL,
    FIELD_PROMPT_STRICT,
    FIELD_PROMPT_TRANSCRIBE,
    extract,
)
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


# --- prompt style selection ------------------------------------------------
# Spec (prompt-style comparison revision): FIELD_PROMPT_STRICT was tuned
# iteratively against Qwen2.5-VL-3B and accreted a negative instruction per
# observed failure. Run against a different model (Qari-OCR-0.4.0-VL-4B) it
# produced valid, correctly-shaped JSON with the two Arabic fields
# deliberately null - the accumulated "do not guess" coaching read as license
# to decline exactly the fields it was written to fix. FIELD_PROMPT_NATURAL
# is the minimal counterpart that is meant to transfer across models.


def test_natural_prompt_is_meaningfully_shorter_than_strict():
    assert len(FIELD_PROMPT_NATURAL) < len(FIELD_PROMPT_STRICT) * 0.5


def test_natural_prompt_keeps_the_date_format_instruction():
    """Factual, not coaching: the card genuinely prints DD/MM/YYYY. Without
    this a model returns 8 November for a card printed 11/08."""
    assert "DD/MM/YYYY" in FIELD_PROMPT_NATURAL
    assert "YYYY-MM-DD" in FIELD_PROMPT_NATURAL


def test_natural_prompt_keeps_the_null_instruction():
    """Factual, not coaching: the whole status pipeline's definition of
    'not read' depends on null meaning exactly that."""
    assert "null" in FIELD_PROMPT_NATURAL


def test_natural_prompt_drops_the_label_exclusion_list():
    """The right-label/left-value layout explanation and the explicit
    Arabic label exclusion list are Qwen-specific coaching, not facts about
    the card, and must not appear in the natural prompt."""
    assert "مكان الميلاد" not in FIELD_PROMPT_NATURAL
    assert "الإسم" not in FIELD_PROMPT_NATURAL


@pytest.mark.parametrize(
    "style,expected_prompt",
    [
        ("strict", FIELD_PROMPT_STRICT),
        ("natural", FIELD_PROMPT_NATURAL),
        ("transcribe", FIELD_PROMPT_TRANSCRIBE),
    ],
)
def test_extract_sends_the_prompt_matching_prompt_style(style, expected_prompt):
    engine = FakeEngine(_replies())
    extract(_img(), engine, Settings(PROMPT_STYLE=style))
    a, b = engine.calls[0]
    assert a.prompt == expected_prompt
    assert b.prompt == expected_prompt


@pytest.mark.parametrize(
    "style,expected_prompt",
    [
        ("strict", FIELD_PROMPT_STRICT),
        ("natural", FIELD_PROMPT_NATURAL),
        ("transcribe", FIELD_PROMPT_TRANSCRIBE),
    ],
)
def test_retry_path_uses_the_prompt_matching_prompt_style(style, expected_prompt):
    engine = FakeEngine(["not json", json.dumps(GOOD), json.dumps(GOOD)])
    extract(_img(), engine, Settings(PROMPT_STYLE=style))
    retry_prompt = engine.calls[1][0].prompt
    assert retry_prompt.startswith(expected_prompt)


def test_unknown_prompt_style_raises_a_clear_error():
    engine = FakeEngine(_replies())
    with pytest.raises(ValueError, match="PROMPT_STYLE"):
        extract(_img(), engine, Settings(PROMPT_STYLE="verbose"))


def test_renamed_keys_do_not_wipe_out_the_whole_card():
    """The reported symptom: every field `missing` on every model except the
    default one.

    A model that answers with its own key spellings - `full_name` for
    `full_name_ar`, plus a `nationality` nobody asked for - used to fail
    schema validation, fail it again on retry, and land in _all_missing().
    Six `missing` fields is indistinguishable, in the UI, from a model that
    read nothing; it had in fact read everything.
    """
    reply = json.dumps(
        {
            "card_type": "resident",
            "full_name": GOOD["full_name_ar"],
            "id_number": "70011864",
            "date_of_birth": "2002-09-29",
            "expiry_date": "2027-01-25",
            "place_of_birth": GOOD["place_of_birth_ar"],
            "nationality": "Oman",
        }
    )
    engine = FakeEngine([reply, reply])
    res = extract(_img(), engine, Settings())

    assert [f.status for f in res.fields.values()].count("missing") == 0
    assert res.fields["full_name"].value_ar == GOOD["full_name_ar"]
    assert res.fields["place_of_birth"].value_ar == GOOD["place_of_birth_ar"]
    assert res.fields["id_number"].value == "70011864"
    # One generate() call: the retry path is not entered at all any more.
    assert len(engine.calls) == 1


def test_unparseable_output_still_reports_every_field_missing():
    """The tolerance must not swallow genuine failure. Prose is still prose."""
    # Four replies: the two-sequence batch, plus one retry per pass.
    engine = FakeEngine(["I cannot read this card."] * 4)
    res = extract(_img(), engine, Settings())
    assert all(f.status == "missing" for f in res.fields.values())


def test_a_parse_failure_says_so_on_every_field():
    """`missing` because we could not parse the reply, and `missing` because
    the model could not read the card, look identical in the UI and have
    nothing in common as problems. The reason distinguishes them."""
    engine = FakeEngine(["I cannot read this card."] * 4)
    res = extract(_img(), engine, Settings())
    reasons = {f.reason for f in res.fields.values()}
    assert len(reasons) == 1
    assert "could not parse model output" in reasons.pop()


# --- transcribe prompt style -----------------------------------------------
# Measured, not guessed: Qari-OCR-0.4.0-VL-4B returned null for both Arabic
# fields under the natural prompt, then transcribed the same image in the
# same session correctly via /api/transcribe. The model can read the Arabic
# at the resolution it already gets; only the framing of the task differed.


def test_transcribe_prompt_frames_the_task_as_transcription():
    """The whole point of this style. "Read ... and return" is what Qari
    declined; transcription is what it was fine-tuned to do."""
    assert FIELD_PROMPT_TRANSCRIBE.lower().startswith("transcribe")


def test_transcribe_prompt_asks_for_the_arabic_verbatim():
    assert "exactly as it is printed" in FIELD_PROMPT_TRANSCRIBE
    assert "every" in FIELD_PROMPT_TRANSCRIBE
    assert "component of the person's name" in FIELD_PROMPT_TRANSCRIBE


def test_transcribe_prompt_keeps_the_date_conversion():
    """Factual about the card, not model coaching - a card printed 11/08
    otherwise comes back as 8 November."""
    assert "DD/MM/YYYY" in FIELD_PROMPT_TRANSCRIBE
    assert "YYYY-MM-DD" in FIELD_PROMPT_TRANSCRIBE


def test_transcribe_prompt_does_not_name_the_arabic_labels():
    """Naming الإسم / مكان الميلاد in FIELD_PROMPT_STRICT is what made AIN
    return the label instead of the value printed beside it."""
    assert "مكان الميلاد" not in FIELD_PROMPT_TRANSCRIBE
    assert "الإسم" not in FIELD_PROMPT_TRANSCRIBE


def test_all_prompt_styles_request_the_same_six_keys():
    """A style that renamed or dropped a key would silently change the
    output contract rather than just the phrasing."""
    from app.extract import FIELD_PROMPTS

    keys = [
        "card_type", "full_name_ar", "id_number",
        "date_of_birth", "expiry_date", "place_of_birth_ar",
    ]
    for style, prompt in FIELD_PROMPTS.items():
        for key in keys:
            assert key in prompt, f"{style} prompt is missing {key}"


def test_response_names_the_prompt_style_that_produced_it():
    """The response named the model but not the prompt, and the prompt has
    changed the outcome more than the model has. Two runs of the same model
    - one returning both Arabic fields, one returning null for both - were
    indistinguishable in the payload, which is how a whole debugging round
    went by without anyone able to confirm which prompt had actually run."""
    engine = FakeEngine(_replies())
    res = extract(_img(), engine, Settings(PROMPT_STYLE="transcribe"))
    assert res.prompt_style == "transcribe"


def test_a_parse_failure_still_names_the_prompt_style():
    """The failure path is exactly where knowing the prompt matters most."""
    engine = FakeEngine(["not json"] * 4)
    res = extract(_img(), engine, Settings(PROMPT_STYLE="natural"))
    assert res.prompt_style == "natural"


# --- recovering Arabic from a transcription --------------------------------
# Measured (docs/measurements.md, 2026-08-29): Qari-OCR returned null for
# both Arabic fields under every prompt style tried, and transcribed the
# same image correctly in the same session. See app/transcript.py.

NO_ARABIC = {**GOOD, "full_name_ar": None, "place_of_birth_ar": None}
TRANSCRIPT = (
    "سلطنة عُمان SULTANATE OF OMAN RESIDENT CARD الرقم المدني "
    "مكان الميلاد جمهورية مصر العربية الإسم "
    "زياد نشأت عبد الحي أبو الوفا حمودة المهنة إتحاق بالأقارب"
)


def test_recovers_arabic_fields_extraction_returned_null_for():
    reply = json.dumps(NO_ARABIC)
    engine = FakeEngine([reply, reply, TRANSCRIPT])
    res = extract(_img(), engine, Settings())

    assert res.fields["full_name"].value_ar == "زياد نشأت عبد الحي أبو الوفا حمودة"
    assert res.fields["place_of_birth"].value_ar == "جمهورية مصر العربية"


def test_a_recovered_value_is_review_and_never_ok():
    """It has no second pass to be compared against, so the agreement check
    that catches hallucination cannot have run on it. Presenting it as `ok`
    would be exactly the silent error this product exists to prevent."""
    reply = json.dumps(NO_ARABIC)
    res = extract(_img(), FakeEngine([reply, reply, TRANSCRIPT]), Settings())

    for field in ("full_name", "place_of_birth"):
        assert res.fields[field].status == "review"
        assert "transcription" in res.fields[field].reason


def test_recovery_costs_nothing_when_the_model_read_the_arabic():
    """A model that works must not pay for a second inference. This is the
    whole reason recovery is conditional rather than always-on."""
    engine = FakeEngine(_replies())
    extract(_img(), engine, Settings())
    assert len(engine.calls) == 1


def test_recovery_can_be_turned_off():
    reply = json.dumps(NO_ARABIC)
    engine = FakeEngine([reply, reply])
    res = extract(_img(), engine, Settings(RECOVER_ARABIC_FROM_TRANSCRIPT=False))
    assert res.fields["full_name"].status == "missing"
    assert len(engine.calls) == 1


def test_a_recovered_value_still_has_to_pass_the_arabic_rules():
    """The capture is bounded by printed labels, so it can pick up a label
    instead of a value. check_arabic() rejects that, and the field stays
    `missing` - which is what it already was."""
    reply = json.dumps(NO_ARABIC)
    labels_only = "مكان الميلاد الإسم المهنة"
    res = extract(_img(), FakeEngine([reply, reply, labels_only]), Settings())
    assert res.fields["full_name"].status == "missing"
    assert res.fields["place_of_birth"].status == "missing"


def test_recovery_never_overwrites_a_value_the_model_actually_read():
    """Only `missing` fields are eligible. A transcript must not be able to
    replace something extraction read successfully."""
    half = {**GOOD, "place_of_birth_ar": None}
    reply = json.dumps(half)
    res = extract(_img(), FakeEngine([reply, reply, TRANSCRIPT]), Settings())
    assert res.fields["full_name"].value_ar == GOOD["full_name_ar"]
    assert res.fields["full_name"].status == "ok"


def test_a_failed_transcription_leaves_the_extraction_intact():
    """Recovery is an improvement on `missing`. It must never be able to
    turn a partial result into no result."""
    reply = json.dumps(NO_ARABIC)

    def _engine(requests):
        if "Transcribe all printed text" in requests[0].prompt:
            raise RuntimeError("CUDA out of memory")
        return [reply, reply]

    res = extract(_img(), FakeEngine(_engine), Settings())
    assert res.fields["id_number"].value == "70011864"
    assert res.fields["full_name"].status == "missing"


def test_response_records_the_vision_token_budget():
    """Same reasoning as prompt_style: it changes the result, so a result
    that does not record it cannot be compared against another run."""
    engine = FakeEngine(_replies())
    res = extract(_img(), engine, Settings(MAX_PIXELS=2560 * 28 * 28))
    assert res.vision_tokens == 2560


def test_a_failed_recovery_says_it_was_attempted():
    """A `missing` field with no reason is indistinguishable from one where
    recovery never ran, and those point at different fixes."""
    reply = json.dumps(NO_ARABIC)
    res = extract(_img(), FakeEngine([reply, reply, "nothing useful here"]), Settings())
    assert res.fields["full_name"].status == "missing"
    assert "recovery found nothing" in res.fields["full_name"].reason


def test_a_successful_recovery_does_not_claim_it_failed():
    reply = json.dumps(NO_ARABIC)
    res = extract(_img(), FakeEngine([reply, reply, TRANSCRIPT]), Settings())
    assert "found nothing" not in (res.fields["full_name"].reason or "")
