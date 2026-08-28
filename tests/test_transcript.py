"""Tests for recovering Arabic values out of a transcription.

The primary fixture is a real transcription, pasted verbatim from a Qari-OCR
run against the resident card on 2026-08-29 - the run that motivated this
module. Synthetic fixtures would have hidden the two things that actually
make this hard: the labels arrive grouped together with no values between
them, and the Latin header and a garbled civil number are interleaved with
the Arabic.
"""

from app.transcript import _MAX_NAME_COMPONENTS, recover_arabic_fields

# Verbatim /api/transcribe output. Note what is and is not in here: the name
# is complete and correct, the place of birth value never appears at all
# (its label is immediately followed by the next label), the civil number is
# the garbage "7E4001C668032A7C", and no date appears anywhere.
REAL_TRANSCRIPT = (
    "سلطنة عُمان SULTANATE OF OMAN RESIDENT CARD 7E4001C668032A7C طاقة مقيم "
    "الرقم المدني تاريخ الإنتهاء تاريخ الميلاد مكان الميلاد الإسم "
    "زياد نشأت عبد الحي أبو الوفا حمودة المهنة إتحاق بالأقارب لـ "
    "الشركة العاملة للمحاجر (ش.م.م)"
)

NAME = "زياد نشأت عبد الحي أبو الوفا حمودة"


def test_recovers_the_full_name_from_the_real_transcript():
    assert recover_arabic_fields(REAL_TRANSCRIPT)["full_name"] == NAME


def test_stops_at_the_occupation_label():
    """المهنة closes the name. Without that boundary the capture would run
    on into the occupation and return a name with an employer attached."""
    recovered = recover_arabic_fields(REAL_TRANSCRIPT)["full_name"]
    assert "المهنة" not in recovered
    assert "الشركة" not in recovered


def test_does_not_invent_a_place_of_birth_that_was_never_transcribed():
    """On this card مكان الميلاد is immediately followed by الإسم - the
    value was not read. Recovering the name that follows would put a
    person's name in the place-of-birth field, which is worse than the
    `missing` it already was."""
    assert "place_of_birth" not in recover_arabic_fields(REAL_TRANSCRIPT)


def test_drops_the_latin_header_and_the_garbled_number():
    recovered = recover_arabic_fields(REAL_TRANSCRIPT)["full_name"]
    assert "SULTANATE" not in recovered
    assert "7E4001C668032A7C" not in recovered


def test_recovers_a_place_of_birth_when_one_is_actually_printed():
    transcript = "الرقم المدني مكان الميلاد جمهورية مصر العربية الإسم زياد حمودة المهنة"
    recovered = recover_arabic_fields(transcript)
    assert recovered["place_of_birth"] == "جمهورية مصر العربية"
    assert recovered["full_name"] == "زياد حمودة"


def test_accepts_either_hamza_spelling_of_the_name_label():
    """Models are not consistent about الإسم vs الاسم."""
    for label in ("الإسم", "الاسم"):
        assert recover_arabic_fields(f"{label} زياد حمودة المهنة")["full_name"] == "زياد حمودة"


def test_refuses_a_capture_that_ran_past_a_missing_boundary_label():
    """If the closing label is not transcribed, the span keeps going into
    the next field. A too-long capture must fail, not be returned as a name
    with someone's occupation stuck to the end of it."""
    runaway = "الإسم " + " ".join(["كلمة"] * (_MAX_NAME_COMPONENTS + 1))
    assert "full_name" not in recover_arabic_fields(runaway)


def test_a_capture_at_the_ceiling_is_still_accepted():
    """The ceiling is a runaway guard, not a name-length rule - a long but
    plausible Omani name must survive it."""
    at_limit = "الإسم " + " ".join(["كلمة"] * _MAX_NAME_COMPONENTS)
    assert len(recover_arabic_fields(at_limit)["full_name"].split()) == _MAX_NAME_COMPONENTS


def test_recovers_nothing_from_a_transcript_with_no_labels():
    assert recover_arabic_fields("SULTANATE OF OMAN RESIDENT CARD") == {}


def test_survives_empty_and_non_string_input():
    """This runs after a successful extraction. It must not be able to turn
    a partial result into no result."""
    assert recover_arabic_fields("") == {}
    assert recover_arabic_fields("   \n  ") == {}
    assert recover_arabic_fields(None) == {}


def test_collapses_line_structure():
    """The transcription prompt asks for line-by-line output, and which text
    lands on which line varies by model."""
    lined = "الرقم المدني\nمكان الميلاد\nمسقط\nالإسم\nزياد حمودة\nالمهنة"
    recovered = recover_arabic_fields(lined)
    assert recovered["full_name"] == "زياد حمودة"
    assert recovered["place_of_birth"] == "مسقط"
