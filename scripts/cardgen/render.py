"""Drawing a card from generated content.

Ground truth never passes through this module's output. Text is shaped for
display on the way in (see scripts/cardgen/text), which turns it into
presentation forms in visual order - correct on screen, useless as data. The
label written alongside the image always comes from CardContent, never from
anything drawn here.
"""

import random

from PIL import Image, ImageDraw, ImageFont

from scripts.cardgen.content import CardContent
from scripts.cardgen.fonts import find_arabic_fonts
from scripts.cardgen.layout import (
    CARD_H,
    CARD_W,
    COLOURS,
    GHOST_CENTRE,
    GHOST_RADIUS,
    LAYOUT,
    PORTRAIT_BOX,
    SERIAL_SIZE,
    SERIAL_TOP,
    SERIAL_X,
    background,
    box_px,
    px,
    round_corners,
)
from scripts.cardgen.text import prepare

# Preference order. Filtered at import to those that can actually draw the
# presentation forms the reshaper emits - Simplified Arabic and Arabic
# Typesetting both fail that check and are excluded automatically.
ARABIC_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\trado.ttf",
    r"C:\Windows\Fonts\majalla.ttf",
    r"C:\Windows\Fonts\andlso.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
]
LATIN_FONTS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
]
LATIN_BOLD = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\tahomabd.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
]


class FontUnavailable(RuntimeError):
    """No font on this machine can draw shaped Arabic."""


def available_arabic_fonts() -> list[str]:
    fonts = find_arabic_fonts(ARABIC_FONT_CANDIDATES)
    if not fonts:
        raise FontUnavailable(
            "No usable Arabic font found. A font must cover the Unicode Arabic "
            "Presentation Forms (U+FE70-FEFF), not just the base Arabic block, "
            "or every Arabic glyph renders as an empty box and the training "
            "set is silently ruined. Install Noto Naskh Arabic or Amiri and "
            "add it to ARABIC_FONT_CANDIDATES."
        )
    return fonts


def _portrait(draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
    """A flat block where the photograph belongs.

    Never a real face, and never a generated one: the model has no use for
    it, and a synthetic corpus of identity documents should not contain
    portraits at all.
    """
    x0, y0, x1, y1 = box_px(PORTRAIT_BOX)
    shade = rng.randint(150, 185)
    draw.rectangle([x0, y0, x1, y1], fill=(shade, shade + 6, shade + 12))
    draw.rectangle(
        [x0, y0, x1, y1], outline=(shade - 42, shade - 36, shade - 30), width=2
    )
    draw.text(
        ((x0 + x1) // 2, (y0 + y1) // 2),
        "PHOTO",
        fill=(shade - 58, shade - 52, shade - 46),
        anchor="mm",
        font=ImageFont.truetype(LATIN_FONTS[0], max(10, (x1 - x0) // 7)),
    )


def _ghost(draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
    """The scalloped secondary portrait, drawn as a plain polygon.

    Shape only. The real card's rosette is part of its security design and
    is not reproduced; what the model needs is that a text-free scalloped
    region sits here and things must be read around it.
    """
    import math

    cx, cy = GHOST_CENTRE[0] * CARD_W, GHOST_CENTRE[1] * CARD_H
    rx, ry = GHOST_RADIUS[0] * CARD_W, GHOST_RADIUS[1] * CARD_H
    points = []
    lobes = rng.choice([9, 10, 11, 12])
    for i in range(lobes * 12):
        t = (i / (lobes * 12)) * 2 * math.pi
        scale = 1.0 + 0.12 * math.cos(t * lobes)
        points.append((cx + math.cos(t) * rx * scale, cy + math.sin(t) * ry * scale))
    shade = rng.randint(158, 186)
    draw.polygon(points, fill=(shade, shade + 5, shade + 11))


def _serial(card: Image.Image, serial: str, rng: random.Random) -> None:
    """The serial runs vertically up the left of the card face.

    Drawn into its own layer and rotated, because Pillow has no vertical
    text without raqm - the same missing feature that forces the Arabic
    shaping in scripts/cardgen/text.
    """
    size = int(SERIAL_SIZE * CARD_H)
    font = ImageFont.truetype(LATIN_FONTS[0], size)
    strip = Image.new("RGBA", (int(CARD_H * 0.60), size + 8), (0, 0, 0, 0))
    ImageDraw.Draw(strip).text((0, 0), serial, font=font, fill=(52, 58, 84, 240))
    strip = strip.rotate(90, expand=True)
    card.paste(strip, (int(SERIAL_X * CARD_W), int(SERIAL_TOP * CARD_H)), strip)


def _signature(draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
    """A scribble above the SIGNATURE label."""
    x = int(0.075 * CARD_W)
    y = int(0.630 * CARD_H)
    points = [(x, y)]
    for _ in range(rng.randint(5, 9)):
        x += rng.randint(14, 34)
        points.append((x, y + rng.randint(-18, 18)))
    draw.line(points, fill=(38, 42, 68), width=rng.randint(2, 4), joint="curve")


def render(content: CardContent, printed_dates: dict, rng: random.Random) -> Image.Image:
    """Compose one clean card. Degradation happens afterwards."""
    arabic_font = rng.choice(available_arabic_fonts())
    latin_font = rng.choice(LATIN_FONTS)
    latin_bold = rng.choice(LATIN_BOLD)

    card = background(rng)
    draw = ImageDraw.Draw(card)
    _portrait(draw, rng)
    _ghost(draw, rng)

    citizen = content.card_type == "citizen"
    values = {
        "header_ar": "سلطنة عمان",
        "header_en": "SULTANATE OF OMAN",
        # Both card types print their name over two lines.
        "type_en_1": "IDENTITY" if citizen else "RESIDENT",
        "type_en_2": "CARD",
        "type_ar_1": "البطاقة" if citizen else "بطاقة",
        "type_ar_2": "الشخصية" if citizen else "مقيم",
        "label_civil_en": "CIVIL NUMBER",
        "label_expiry_en": "EXPIRY DATE",
        "label_dob_en": "DATE OF BIRTH",
        "value_civil": content.id_number,
        "value_expiry": printed_dates["expiry_date"],
        "value_dob": printed_dates["date_of_birth"],
        "label_civil_ar": "الرقم المدني",
        "label_expiry_ar": "تاريخ الإنتهاء",
        "label_dob_ar": "تاريخ الميلاد",
        "label_pob_ar": "مكان الميلاد",
        "value_pob_ar": content.place_of_birth_ar,
        "label_name_ar": "الإسم",
        "value_name_ar": content.full_name_ar,
        "label_sig_en": "SIGNATURE",
        "label_sig_ar": "التوقيع",
    }
    # المهنة is printed on resident cards only. It matters even though it is
    # never extracted: it is the row below the name, and it is what closes
    # the name when anything reads the card line by line.
    if content.occupation_ar:
        values["label_occ_ar"] = "المهنة"
        values["value_occ_ar"] = content.occupation_ar

    for key, text in values.items():
        field = LAYOUT[key]
        x, y, size = px(field)
        if field.script == "arabic":
            font = ImageFont.truetype(arabic_font, size)
        else:
            font = ImageFont.truetype(
                latin_bold if field.weight == "bold" else latin_font, size
            )
        draw.text(
            (x, y), prepare(text), font=font, fill=COLOURS[field.colour], anchor=field.anchor
        )

    _serial(card, content.serial, rng)
    _signature(draw, rng)
    return round_corners(card)
