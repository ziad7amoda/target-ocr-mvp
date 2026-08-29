"""Arabic text shaping for image rendering.

Pillow on this machine is built without raqm, so ImageDraw.text() has no
complex-text layout: it draws each codepoint in its isolated form, in
logical order. Arabic then comes out unjoined and reversed - "زياد" renders
as "دايز" - which as training data is worse than nothing, because a model
learns to read text that does not exist.

This module does what raqm would have done, in two steps, before Pillow
sees the string:

    reshape  - substitute each letter's contextual form (initial, medial,
               final, isolated) according to its neighbours
    bidi     - reorder the logical-order string into visual order, so a
               left-to-right renderer draws it right-to-left

The result is a string that is CORRECT ON SCREEN and MEANINGLESS AS TEXT.
Never write it to a label file, a manifest, or anything else that is read
back as data - ground truth must always be the original logical string.
That confusion is the one real hazard in this module, so the two directions
are kept in separately named functions rather than one flag.
"""

import arabic_reshaper
from bidi.algorithm import get_display

# Ligatures off. The card prints names as plain letter sequences; enabling
# ligature substitution would render glyph combinations the real document
# does not use, and teach the model shapes it will never see.
_RESHAPER = arabic_reshaper.ArabicReshaper(
    {"delete_harakat": False, "support_ligatures": False}
)


def shape_for_display(text: str) -> str:
    """Logical-order Arabic -> visual-order, contextually shaped glyphs.

    For DRAWING ONLY. The return value is not the same text: its characters
    are presentation forms in visual order, and comparing it to the original
    will not match.
    """
    if not text:
        return text
    return get_display(_RESHAPER.reshape(text))


def contains_arabic(text: str) -> bool:
    """True if any character is in an Arabic script block."""
    return any(
        0x0600 <= ord(c) <= 0x06FF
        or 0x0750 <= ord(c) <= 0x077F
        or 0xFB50 <= ord(c) <= 0xFDFF
        or 0xFE70 <= ord(c) <= 0xFEFF
        for c in text
    )


def prepare(text: str) -> str:
    """Shape only if there is Arabic to shape.

    Latin text passed through get_display() is unchanged in practice, but
    going through the bidi algorithm for a pure-Latin label is pointless
    work and one more thing that could surprise us later.
    """
    return shape_for_display(text) if contains_arabic(text) else text
