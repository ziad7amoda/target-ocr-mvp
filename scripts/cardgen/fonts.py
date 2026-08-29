"""Choosing fonts that can actually draw what the reshaper produces.

arabic_reshaper emits Unicode Arabic Presentation Forms (U+FE70-FEFF) - the
initial/medial/final/isolated variants - rather than the base letters. A font
that covers the base Arabic block but not the presentation forms renders those
codepoints as .notdef, and Pillow draws a tofu box. It does not raise, warn,
or fall back.

That is the dangerous failure for this project: it is silent, it looks fine in
a thumbnail, and a training set built with such a font teaches the model that
Omani names contain rectangles. Simplified Arabic (simpo.ttf), which ships
with Windows and looks like an obvious choice, fails exactly this way on
و and ر.

So fonts are verified before use, not assumed, and a font that cannot draw a
character is excluded rather than silently substituted.
"""

from pathlib import Path

from fontTools.ttLib import TTFont

from scripts.cardgen.text import shape_for_display

# Held to a real name and a real place, both taken from cards we have seen,
# so the check exercises the letters that actually appear rather than a
# synthetic alphabet.
PROBE_STRINGS = (
    "زياد نشأت عبد الحى ابو الوفا حموده",
    "جمهورية مصر العربية",
    "مسقط ظفار مسندم الداخلية البريمي الظاهرة",
    "بطاقة مقيم البطاقة الشخصية",
    "الرقم المدني تاريخ الإنتهاء تاريخ الميلاد مكان الميلاد الإسم المهنة",
)


def _covered_codepoints(font_path: str) -> set[int]:
    font = TTFont(font_path, fontNumber=0, lazy=True)
    try:
        covered: set[int] = set()
        for table in font["cmap"].tables:
            covered.update(table.cmap.keys())
        return covered
    finally:
        font.close()


def missing_characters(font_path: str, texts=PROBE_STRINGS) -> set[str]:
    """Characters the font cannot draw, AFTER shaping.

    Shaping first is the whole point: the base letters are almost always
    present, and the presentation forms are what is missing.
    """
    covered = _covered_codepoints(font_path)
    missing = set()
    for text in texts:
        for char in shape_for_display(text):
            if char.isspace():
                continue
            if ord(char) not in covered:
                missing.add(char)
    return missing


def usable(font_path: str) -> bool:
    try:
        return not missing_characters(font_path)
    except Exception:
        return False


def find_arabic_fonts(candidates: list[str]) -> list[str]:
    """Filter a candidate list down to fonts that exist and are complete.

    Returns them in the order given, so the caller's preference is kept.
    """
    return [p for p in candidates if Path(p).is_file() and usable(p)]
