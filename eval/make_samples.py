"""Synthetic bilingual ID cards so the eval harness runs before real photos.

These are clean renders. They have no glare, skew, wear or depth of field,
and they WILL overstate accuracy (spec R5). They exist to prove the harness
works end to end, not to predict field performance.

Arabic needs arabic-reshaper (contextual letter joining) and python-bidi
(right-to-left reordering). Pillow does neither, so without them the text
renders as disconnected, reversed letterforms and the Arabic half of the
eval measures nothing (spec D6).
"""

import json
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

SAMPLES_DIR = Path(__file__).parent / "samples"
FONT_DIR = Path(__file__).parent / "fonts"
FONT_PATH = FONT_DIR / "NotoNaskhArabic-Regular.ttf"
LATIN_FONT_PATH = FONT_DIR / "DejaVuSans.ttf"

FONT_HELP = f"""Missing font: {FONT_PATH}

Download Noto Naskh Arabic (SIL Open Font License) and place the Regular
TTF at that path:

    https://fonts.google.com/noto/specimen/Noto+Naskh+Arabic

Pillow ships DejaVuSans, which has no Arabic coverage, so this cannot be
substituted automatically."""

# Revision 2026-08-24 (docs/superpowers/specs/2026-08-24-revision-real-card-findings.md):
# real Omani cards print neither nationality nor sex, and the holder's name
# is Arabic-only - there is no Latin full_name to extract. Every record
# below carries exactly CardFields' six keys: card_type, full_name_ar,
# id_number, date_of_birth, expiry_date, place_of_birth_ar. At least one
# citizen and one resident card are included, since the two families print
# a different header label (spec §1).
RECORDS: list[dict] = [
    {
        "card_type": "citizen",
        "full_name_ar": "زياد نشأت عبد الحى ابو الوفا حموده",
        "id_number": "10293847", "date_of_birth": "1988-03-21",
        "expiry_date": "2031-03-20", "place_of_birth_ar": "سلطنة عمان",
    },
    {
        "card_type": "resident",
        "full_name_ar": "فاطمة على محمد عبد الرحمن السيد",
        "id_number": "20394857", "date_of_birth": "1995-11-02",
        "expiry_date": "2029-11-01", "place_of_birth_ar": "جمهورية مصر العربية",
    },
    {
        "card_type": "citizen",
        "full_name_ar": "أحمد سعيد بن راشد بن سالم الحارثي",
        "id_number": "30485712", "date_of_birth": "1979-07-14",
        "expiry_date": "2028-07-13", "place_of_birth_ar": "سلطنة عمان",
    },
    {
        "card_type": "resident",
        "full_name_ar": "مريم أحمد سالم عبيد الكعبي",
        "id_number": "40576123", "date_of_birth": "1992-01-30",
        "expiry_date": "2030-01-29", "place_of_birth_ar": "الامارات",
    },
]

# card_type isn't a labelled row on the real cards - it IS the header label
# (البطاقة الشخصية / IDENTITY CARD vs بطاقة مقيم / RESIDENT CARD), so it is
# drawn in the header band rather than as a body row (see render_card).
_CARD_TYPE_LABELS = {"citizen": "IDENTITY CARD", "resident": "RESIDENT CARD"}

# Body rows printed in Latin script / Western numerals (spec D2, R2/R5).
# Box key == the logical FIELD_NAMES entry run_eval.py scores against.
_LATIN_ROWS = [
    ("CIVIL NUMBER", "id_number"),
    ("EXPIRY DATE", "expiry_date"),
    ("DATE OF BIRTH", "date_of_birth"),
]

# Body rows printed ONLY in Arabic (spec R4): (arabic label, logical field
# name for the ground-truth box, CardFields key holding the value).
_ARABIC_ROWS = [
    ("الإسم", "full_name", "full_name_ar"),
    ("مكان الميلاد", "place_of_birth", "place_of_birth_ar"),
]


def shape_arabic(text: str) -> str:
    """Join and reorder Arabic for a renderer with no bidi support."""
    return get_display(arabic_reshaper.reshape(text))


def _fonts() -> tuple[ImageFont.FreeTypeFont, ...]:
    if not FONT_PATH.exists():
        raise FileNotFoundError(FONT_HELP)
    latin = (
        ImageFont.truetype(str(LATIN_FONT_PATH), 26)
        if LATIN_FONT_PATH.exists()
        else ImageFont.load_default(26)
    )
    label = (
        ImageFont.truetype(str(LATIN_FONT_PATH), 16)
        if LATIN_FONT_PATH.exists()
        else ImageFont.load_default(16)
    )
    arabic = ImageFont.truetype(str(FONT_PATH), 24)
    # DejaVuSans has no Arabic coverage (see FONT_HELP), so الإسم / مكان
    # الميلاد's own row labels need the Arabic font too, just smaller to
    # read as a label rather than a value.
    arabic_label = ImageFont.truetype(str(FONT_PATH), 16)
    return latin, label, arabic, arabic_label


def render_card(record: dict, size: tuple[int, int] = (1000, 630)) -> tuple[Image.Image, dict]:
    """Render one card. Returns the image and field -> ground-truth box.

    Layout mirrors the two real cards this revision is based on (spec §1):
    a dark header band naming the card type, CIVIL NUMBER / EXPIRY DATE /
    DATE OF BIRTH in Latin labels with Western numerals, and الإسم / مكان
    الميلاد in Arabic. There is no nationality or sex row - real cards do
    not print either (spec R1).
    """
    latin_font, label_font, arabic_font, arabic_label_font = _fonts()

    image = Image.new("RGB", size, (243, 246, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (size[0], 72)], fill=(14, 21, 27))
    draw.text((36, 12), "SULTANATE OF OMAN", font=label_font, fill="white")

    boxes: dict[str, tuple[int, int, int, int]] = {}

    # card_type isn't its own body row on a real card - it IS the header
    # label (البطاقة الشخصية / IDENTITY CARD vs بطاقة مقيم / RESIDENT CARD).
    card_type_label = _CARD_TYPE_LABELS[record["card_type"]]
    draw.text((36, 38), card_type_label, font=label_font, fill="white")
    x1, y1, x2, y2 = draw.textbbox((36, 38), card_type_label, font=label_font)
    boxes["card_type"] = (int(x1) - 4, int(y1) - 4, int(x2) + 4, int(y2) + 4)

    y = 118
    for label, key in _LATIN_ROWS:
        draw.text((36, y), label, font=label_font, fill=(104, 117, 126))
        value = record[key]
        draw.text((300, y - 6), value, font=latin_font, fill=(14, 21, 27))
        # Ground truth for box hit rate: the box around the drawn value -
        # draw.text and draw.textbbox always get identical arguments so the
        # box stays pixel-consistent with what was actually rendered.
        x1, y1, x2, y2 = draw.textbbox((300, y - 6), value, font=latin_font)
        boxes[key] = (int(x1) - 4, int(y1) - 4, int(x2) + 4, int(y2) + 4)
        y += 78

    for label, logical_key, ar_key in _ARABIC_ROWS:
        shaped_label = shape_arabic(label)
        label_w = draw.textlength(shaped_label, font=arabic_label_font)
        draw.text(
            (size[0] - 36 - label_w, y), shaped_label, font=arabic_label_font,
            fill=(104, 117, 126),
        )

        shaped_value = shape_arabic(record[ar_key])
        value_w = draw.textlength(shaped_value, font=arabic_font)
        value_pos = (size[0] - 36 - value_w, y + 22)
        draw.text(value_pos, shaped_value, font=arabic_font, fill=(14, 21, 27))
        x1, y1, x2, y2 = draw.textbbox(value_pos, shaped_value, font=arabic_font)
        boxes[logical_key] = (int(x1) - 4, int(y1) - 4, int(x2) + 4, int(y2) + 4)
        y += 78

    return image, boxes


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    expected: dict[str, dict] = {}

    for i, record in enumerate(RECORDS, start=1):
        name = f"synthetic_{i:02d}.jpg"
        image, boxes = render_card(record)
        image.save(SAMPLES_DIR / name, quality=95)
        expected[name] = {"fields": record, "boxes": boxes}

    (SAMPLES_DIR / "expected.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(RECORDS)} cards to {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
