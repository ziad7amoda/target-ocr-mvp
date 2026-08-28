"""Recovering Arabic field values from a free-form transcription.

Measured on a resident card (docs/measurements.md, 2026-08-29): asked to
extract fields as JSON, Qari-OCR returned null for full_name_ar and
place_of_birth_ar while reading the civil number and both dates correctly.
Asked to transcribe the same image in the same session, it returned the
person's full six-component Arabic name. Three prompt styles failed to move
the extraction result; the transcription was right every time.

So this module exists because the two modes are good at opposite halves of
the card, and asking one of them to do both has not worked.

What it does NOT do is parse the whole card out of a transcript. The
transcript loses exactly what extraction gets right - on that same run the
civil number came back as `7E4001C668032A7C` and the dates were absent
entirely. Only the Arabic-only fields are recovered here, and only when
extraction returned nothing for them.

The risk being managed is a value landing in the wrong field. Three things
bound it:

  * A recovered value is only ever written to a field that is `missing`.
    It can never overwrite something the model actually read.
  * A recovered value is never `ok`. It has no second pass to be compared
    against, so it is `review` and says so - a human looks at every one.
  * The capture is bounded on both sides by known printed labels, filtered
    to Arabic script, and rejected if it runs longer than a name plausibly
    can - the failure mode of an absent closing label is a capture that
    swallows the next field, and that must not read as a successful one.

Failing to recover is fine. The field stays `missing`, which is what it
already was.
"""

from app.validate import _ARABIC_SCRIPT, _CARD_LABELS

# The labels that introduce the two Arabic-only values. Both hamza spellings
# of "the name" appear in the wild, and models are not consistent about it.
_NAME_LABELS = ("الإسم", "الاسم")
_PLACE_LABELS = ("مكان الميلاد",)

# Any printed label closes the preceding value. Sorted longest-first so
# "مكان الميلاد" is found before the shorter labels it contains, which would
# otherwise cut a span in the middle of a label.
_BOUNDARIES = tuple(sorted(_CARD_LABELS, key=len, reverse=True))

# An Omani name on these cards runs five to eight components. Ten is a
# deliberately loose ceiling: its job is not to validate the name but to
# catch a capture that ran past a missing closing label and swallowed the
# occupation line with it.
_MAX_NAME_COMPONENTS = 10


def _value_after(transcript: str, labels: tuple[str, ...]) -> str | None:
    """Return the Arabic text printed after the first matching label.

    The span ends at the next known label, or at the end of the transcript.
    Non-Arabic tokens inside the span are dropped: a transcript interleaves
    the card's Latin header and Western numerals with its Arabic, and none
    of that belongs in a name or a place.
    """
    for label in labels:
        index = transcript.find(label)
        if index == -1:
            continue

        rest = transcript[index + len(label) :]
        cut = len(rest)
        for boundary in _BOUNDARIES:
            found = rest.find(boundary)
            if found != -1:
                cut = min(cut, found)

        words = [w for w in rest[:cut].split() if _ARABIC_SCRIPT.search(w)]
        if not words:
            # The label was transcribed but nothing followed it before the
            # next label - which is exactly what a card whose place of birth
            # the model did not read looks like. Nothing to recover.
            continue
        if len(words) > _MAX_NAME_COMPONENTS:
            # Almost certainly ran past a label that was not transcribed and
            # kept going into the next field. Refuse rather than return a
            # name with someone's occupation stuck to the end of it.
            continue
        return " ".join(words)
    return None


def recover_arabic_fields(transcript: str | None) -> dict[str, str]:
    """Best-effort recovery of full_name / place_of_birth from a transcript.

    Returns a mapping of logical field name to Arabic value, containing only
    the fields actually found. Never raises: this runs after a successful
    extraction and must not be able to turn a partial result into no result.
    """
    if not isinstance(transcript, str) or not transcript.strip():
        return {}

    # Collapse the line structure. The transcription prompt asks for line by
    # line output, but which text lands on which line varies by model, and
    # the label boundaries carry the structure that actually matters here.
    flat = " ".join(transcript.split())

    out: dict[str, str] = {}
    for field, labels in (("full_name", _NAME_LABELS), ("place_of_birth", _PLACE_LABELS)):
        value = _value_after(flat, labels)
        if value is not None:
            out[field] = value
    return out
