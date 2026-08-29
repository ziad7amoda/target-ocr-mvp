"""Card geometry and background.

Two things this module deliberately does NOT do.

It does not reproduce the real card's design. No state emblem, no
guilloche of record, no hologram, no ghost portrait, no official colour
match. The background here is a procedurally generated interference pattern
that gives the model the problem it actually needs to solve - small text
over a busy, low-contrast ground - without being a copy of a national
identity document. For training a reader, the security artwork is not the
part that carries information.

It does not claim these coordinates are exact. They were measured off a
photograph taken at an angle with glare across the top, so they are close
but not authoritative. Every position lives in the one dict below so that
correcting them from a flat scan is a single edit rather than a hunt.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

# ID-1, the bank-card standard: 85.6 x 53.98 mm, rendered at ~300 dpi.
CARD_W, CARD_H = 1012, 638


@dataclass(frozen=True)
class Field:
    """A printed element, positioned in fractions of the card's size.

    `anchor` follows Pillow's convention. Arabic values and labels are
    right-anchored ("ra"/"rs") because the card's right margin is the fixed
    edge and the text grows leftwards from it - a long name must extend
    left, not overflow the card.
    """

    x: float
    y: float
    size: float          # fraction of card height
    anchor: str = "la"
    script: str = "latin"
    weight: str = "regular"


# APPROXIMATE - see the module docstring. Measured from the resident card
# photograph of 2026-08-29.
LAYOUT: dict[str, Field] = {
    # Header, stacked centre-right above the card-type line
    "header_ar":       Field(0.600, 0.035, 0.058, "ma", "arabic"),
    "header_en":       Field(0.600, 0.105, 0.038, "ma", "latin", "bold"),
    # Card type, printed in both scripts on opposite sides
    "type_en":         Field(0.320, 0.290, 0.052, "la", "latin", "bold"),
    "type_ar":         Field(0.880, 0.290, 0.050, "ra", "arabic"),
    # The three Latin-labelled rows. The Arabic label is pinned to the right
    # margin and grows leftwards, so the printed value has to stop well
    # short of it - at 0.75 the two overlapped and the civil number was
    # drawn straight through مكان الميلاد.
    "label_civil_en":  Field(0.300, 0.520, 0.038, "la", "latin", "bold"),
    "label_expiry_en": Field(0.300, 0.583, 0.038, "la", "latin", "bold"),
    "label_dob_en":    Field(0.300, 0.646, 0.038, "la", "latin", "bold"),
    "value_civil":     Field(0.690, 0.520, 0.050, "ra", "latin", "bold"),
    "value_expiry":    Field(0.690, 0.583, 0.050, "ra", "latin", "bold"),
    "value_dob":       Field(0.690, 0.646, 0.050, "ra", "latin", "bold"),
    "label_civil_ar":  Field(0.885, 0.522, 0.033, "ra", "arabic"),
    "label_expiry_ar": Field(0.885, 0.585, 0.033, "ra", "arabic"),
    "label_dob_ar":    Field(0.885, 0.648, 0.033, "ra", "arabic"),
    # The Arabic-only rows: label pinned right, value growing leftwards
    "label_pob_ar":    Field(0.885, 0.718, 0.033, "ra", "arabic"),
    "value_pob_ar":    Field(0.735, 0.716, 0.044, "ra", "arabic"),
    "label_name_ar":   Field(0.885, 0.788, 0.033, "ra", "arabic"),
    "value_name_ar":   Field(0.790, 0.786, 0.044, "ra", "arabic"),
    "label_occ_ar":    Field(0.885, 0.858, 0.033, "ra", "arabic"),
    "value_occ_ar":    Field(0.790, 0.856, 0.038, "ra", "arabic"),
    "label_sig_en":    Field(0.030, 0.880, 0.030, "la", "latin", "bold"),
}

# Placeholder blocks. No real photograph is ever composited - a generated
# card must not carry anybody's face.
PORTRAIT_BOX = (0.025, 0.045, 0.180, 0.530)
# The ghost portrait sits high and right, clear of the card-type line. At
# its first position it was drawn straight over "IDENTITY CARD".
GHOST_BOX = (0.600, 0.150, 0.740, 0.470)

# The serial is printed vertically down the left edge of the card face,
# reading bottom-to-top, between the photograph and the card-type line.
SERIAL_X, SERIAL_TOP, SERIAL_SIZE = 0.205, 0.055, 0.030


def _interference(w: int, h: int, rng, base: tuple[int, int, int]) -> np.ndarray:
    """A guilloche-ish ground: overlapping sine fields at random angles.

    Chosen because it is generative rather than copied, and because it puts
    fine low-contrast structure under the text, which is the property that
    makes reading the card hard in the first place.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.zeros((h, w), dtype=np.float32)
    for _ in range(4):
        angle = rng.uniform(0, np.pi)
        freq = rng.uniform(0.02, 0.09)
        phase = rng.uniform(0, 2 * np.pi)
        field += np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * freq + phase)
    for _ in range(2):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = np.hypot(xx - cx, yy - cy)
        field += np.sin(r * rng.uniform(0.02, 0.05))

    field = field / 6.0                      # -> roughly [-1, 1]
    tint = np.array(base, dtype=np.float32)
    # Shallow modulation: security printing is low-contrast, and a strong
    # pattern would make this an easier problem than the real card.
    out = tint[None, None, :] + field[:, :, None] * 17.0
    return np.clip(out, 0, 255).astype(np.uint8)


def background(rng) -> Image.Image:
    """The blank card face, before any text."""
    # Pale cyan, the family of tone the real card sits in - close enough
    # that the model sees text on a coloured ground rather than on grey,
    # without being a colour match to the document.
    base = (
        rng.randint(198, 218),
        rng.randint(222, 236),
        rng.randint(230, 242),
    )
    arr = _interference(CARD_W, CARD_H, rng, base)

    # A soft diagonal wash, which every laminated card has under photography.
    yy, xx = np.mgrid[0:CARD_H, 0:CARD_W].astype(np.float32)
    wash = ((xx / CARD_W) * 0.6 + (yy / CARD_H) * 0.4)
    arr = np.clip(arr.astype(np.float32) + (wash[:, :, None] - 0.5) * 14, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def px(field: Field) -> tuple[int, int, int]:
    """Field position and font size in pixels."""
    return int(field.x * CARD_W), int(field.y * CARD_H), max(8, int(field.size * CARD_H))


def box_px(box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return int(x0 * CARD_W), int(y0 * CARD_H), int(x1 * CARD_W), int(y1 * CARD_H)
