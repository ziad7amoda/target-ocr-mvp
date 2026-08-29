"""Card geometry and background.

Coordinates were measured from a clear, straight-on photograph of a real
resident card (3403x1993, 2026-08-24). An earlier version of this file was
measured from an angled shot with glare and was wrong in ways that matter:
the Arabic labels sat too far from the right edge, every value was printed
too small, and the card type was on one line where the real card uses two.
Text size and position are what the model learns, so those were not cosmetic
errors - a corpus built from them teaches the wrong layout.

What this module deliberately does NOT reproduce is the card's security
design: no khanjar emblem, no map silhouette, no hologram, no reproduction
of the guilloche of record, no portrait real or generated. The background
here is a procedurally generated rosette and wave field that gives the model
the problem it actually has - small text over a busy, low-contrast, pale
blue ground - without being a copy of a national identity document. For
training a reader, the security artwork carries no information.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

# ID-1, the bank-card standard: 85.6 x 53.98 mm, rendered at ~300 dpi.
CARD_W, CARD_H = 1012, 638
CORNER_RADIUS = 0.055  # fraction of card height


@dataclass(frozen=True)
class Field:
    """A printed element, positioned in fractions of the card's size.

    `anchor` follows Pillow's convention. Arabic values and labels are
    right-anchored because the card's right margin is the fixed edge and
    text grows leftwards from it - a long name must extend left, never
    overflow the card.
    """

    x: float
    y: float
    size: float          # fraction of card height
    anchor: str = "la"
    script: str = "latin"
    weight: str = "regular"
    colour: str = "ink"


# Measured from the real resident card. See the module docstring.
LAYOUT: dict[str, Field] = {
    # Header, centred over the right two-thirds, printed in maroon
    "header_ar":       Field(0.539, 0.062, 0.070, "ma", "arabic", "regular", "header"),
    "header_en":       Field(0.539, 0.172, 0.046, "ma", "latin", "bold", "header"),
    # Card type runs to TWO lines in both scripts
    "type_en_1":       Field(0.341, 0.325, 0.058, "la", "latin", "bold"),
    "type_en_2":       Field(0.341, 0.383, 0.058, "la", "latin", "bold"),
    "type_ar_1":       Field(0.858, 0.330, 0.062, "ra", "arabic"),
    "type_ar_2":       Field(0.858, 0.400, 0.062, "ra", "arabic"),
    # The three Latin-labelled rows
    "label_civil_en":  Field(0.313, 0.525, 0.042, "la", "latin", "bold", "label"),
    "label_expiry_en": Field(0.313, 0.590, 0.042, "la", "latin", "bold", "label"),
    "label_dob_en":    Field(0.313, 0.654, 0.042, "la", "latin", "bold", "label"),
    "value_civil":     Field(0.752, 0.545, 0.058, "ra", "latin", "bold"),
    "value_expiry":    Field(0.746, 0.610, 0.054, "ra", "latin", "bold"),
    "value_dob":       Field(0.748, 0.673, 0.054, "ra", "latin", "bold"),
    # Arabic labels sit hard against the right margin
    "label_civil_ar":  Field(0.945, 0.530, 0.044, "ra", "arabic", "regular", "label"),
    "label_expiry_ar": Field(0.945, 0.593, 0.044, "ra", "arabic", "regular", "label"),
    "label_dob_ar":    Field(0.945, 0.655, 0.044, "ra", "arabic", "regular", "label"),
    "label_pob_ar":    Field(0.944, 0.728, 0.044, "ra", "arabic", "regular", "label"),
    "label_name_ar":   Field(0.939, 0.808, 0.044, "ra", "arabic", "regular", "label"),
    "label_occ_ar":    Field(0.939, 0.870, 0.044, "ra", "arabic", "regular", "label"),
    # The Arabic values. place_of_birth right-aligns with the value column;
    # the name extends further right, almost to the labels, because it is
    # the longest thing printed on the card.
    "value_pob_ar":    Field(0.752, 0.745, 0.052, "ra", "arabic"),
    "value_name_ar":   Field(0.854, 0.820, 0.050, "ra", "arabic"),
    # Occupation gets its OWN line below its label, not the same one.
    "value_occ_ar":    Field(0.847, 0.922, 0.046, "ra", "arabic"),
    "label_sig_en":    Field(0.043, 0.678, 0.034, "la", "latin", "bold", "label"),
    "label_sig_ar":    Field(0.245, 0.672, 0.038, "ra", "arabic", "regular", "label"),
}

PORTRAIT_BOX = (0.016, 0.048, 0.251, 0.585)
# Scalloped rosette on the real card, rendered here as a plain polygon.
GHOST_CENTRE = (0.610, 0.356)
GHOST_RADIUS = (0.064, 0.112)

# The serial runs vertically between the photograph and the card-type block,
# reading bottom to top.
SERIAL_X, SERIAL_TOP, SERIAL_SIZE = 0.290, 0.076, 0.034

COLOURS = {
    # Values: near-black navy, the highest-contrast thing on the card.
    "ink": (24, 30, 52),
    # Labels: printed lighter than the values they introduce.
    "label": (46, 58, 88),
    # The header is maroon, not navy - a detail worth getting right because
    # colour is one of the few cues separating the header from a value.
    "header": (108, 42, 74),
}


def _guilloche(w: int, h: int, rng, base: tuple[int, int, int]) -> np.ndarray:
    """A pale blue security-print ground: a central rosette over a wave field.

    Generative, not copied. The property that matters for training is fine
    low-contrast structure underneath the text, which is what makes the card
    hard to read in the first place.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.zeros((h, w), dtype=np.float32)

    # Fine directional waves across the whole face.
    for _ in range(3):
        angle = rng.uniform(0, np.pi)
        freq = rng.uniform(0.05, 0.16)
        field += np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * freq)

    # A large rosette a little right of centre, as on the real card.
    cx, cy = w * rng.uniform(0.50, 0.62), h * rng.uniform(0.52, 0.66)
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    petals = rng.choice([8, 10, 12, 16])
    field += 2.2 * np.sin(r * 0.055 + np.sin(theta * petals) * 3.0)
    field += 1.4 * np.sin(r * 0.017)

    field /= 7.0
    tint = np.array(base, dtype=np.float32)
    out = tint[None, None, :] + field[:, :, None] * 13.0
    return np.clip(out, 0, 255).astype(np.uint8)


def background(rng) -> Image.Image:
    """The blank card face, before any text. Pale blue, as the real card is."""
    base = (
        rng.randint(196, 216),
        rng.randint(220, 234),
        rng.randint(232, 244),
    )
    arr = _guilloche(CARD_W, CARD_H, rng, base)

    yy, xx = np.mgrid[0:CARD_H, 0:CARD_W].astype(np.float32)
    wash = (xx / CARD_W) * 0.6 + (yy / CARD_H) * 0.4
    arr = np.clip(arr.astype(np.float32) + (wash[:, :, None] - 0.5) * 12, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def round_corners(card: Image.Image) -> Image.Image:
    """Cut the card's rounded corners into its alpha channel.

    Square corners survive the perspective warp as square corners, which is
    a strong and entirely false cue for anything that has to find the card
    in a photograph before reading it.
    """
    radius = int(CORNER_RADIUS * CARD_H)
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, card.width - 1, card.height - 1], radius=radius, fill=255
    )
    out = card.convert("RGBA")
    out.putalpha(mask)
    return out


def px(field: Field) -> tuple[int, int, int]:
    """Field position and font size in pixels."""
    return int(field.x * CARD_W), int(field.y * CARD_H), max(8, int(field.size * CARD_H))


def box_px(box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return int(x0 * CARD_W), int(y0 * CARD_H), int(x1 * CARD_W), int(y1 * CARD_H)
