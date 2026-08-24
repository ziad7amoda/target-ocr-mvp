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

RECORDS: list[dict] = [
    {
        "full_name": "AHMED SAID AL HARTHY", "full_name_ar": "أحمد سعيد الحارثي",
        "id_number": "10293847", "date_of_birth": "1988-03-21",
        "expiry_date": "2031-03-20", "nationality": "OMANI",
        "nationality_ar": "عماني", "sex": "M", "sex_ar": "ذكر",
    },
    {
        "full_name": "FATIMA ALI AL BALUSHI", "full_name_ar": "فاطمة علي البلوشي",
        "id_number": "20394857", "date_of_birth": "1995-11-02",
        "expiry_date": "2029-11-01", "nationality": "OMANI",
        "nationality_ar": "عماني", "sex": "F", "sex_ar": "أنثى",
    },
    {
        "full_name": "RAJESH KUMAR NAIR", "full_name_ar": "راجيش كومار نائير",
        "id_number": "30485712", "date_of_birth": "1979-07-14",
        "expiry_date": "2028-07-13", "nationality": "INDIAN",
        "nationality_ar": "هندي", "sex": "M", "sex_ar": "ذكر",
    },
    {
        "full_name": "MARIA SANTOS CRUZ", "full_name_ar": "ماريا سانتوس كروز",
        "id_number": "40576123", "date_of_birth": "1992-01-30",
        "expiry_date": "2030-01-29", "nationality": "FILIPINO",
        "nationality_ar": "فلبيني", "sex": "F", "sex_ar": "أنثى",
    },
]

_ROWS = [
    ("Name", "full_name", "full_name_ar"),
    ("ID Number", "id_number", None),
    ("Date of Birth", "date_of_birth", None),
    ("Expiry", "expiry_date", None),
    ("Nationality", "nationality", "nationality_ar"),
    ("Sex", "sex", "sex_ar"),
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
    return latin, label, ImageFont.truetype(str(FONT_PATH), 24)


def render_card(record: dict, size: tuple[int, int] = (1000, 630)) -> tuple[Image.Image, dict]:
    """Render one card. Returns the image and field -> ground-truth box."""
    latin_font, label_font, arabic_font = _fonts()

    image = Image.new("RGB", size, (243, 246, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (size[0], 72)], fill=(14, 21, 27))
    draw.text((36, 24), "SULTANATE OF OMAN  -  IDENTITY CARD", font=label_font, fill="white")

    boxes: dict[str, tuple[int, int, int, int]] = {}
    y = 118
    for label, key, ar_key in _ROWS:
        draw.text((36, y), label.upper(), font=label_font, fill=(104, 117, 126))
        value = record[key]
        draw.text((300, y - 6), value, font=latin_font, fill=(14, 21, 27))
        # Ground truth for box hit rate: the box around the Latin value.
        x1, y1, x2, y2 = draw.textbbox((300, y - 6), value, font=latin_font)
        boxes[key] = (int(x1) - 4, int(y1) - 4, int(x2) + 4, int(y2) + 4)

        if ar_key:
            shaped = shape_arabic(record[ar_key])
            w = draw.textlength(shaped, font=arabic_font)
            draw.text((size[0] - 36 - w, y - 4), shaped, font=arabic_font, fill=(14, 21, 27))
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
