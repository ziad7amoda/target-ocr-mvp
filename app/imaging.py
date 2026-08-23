"""Image preprocessing for the self-consistency pass.

This is a port of enhance() from the original browser demo (ocr-demo.html).
Keeping the same algorithm matters: the preprocessing already validated by
eye in the browser is what feeds pass B, so a disagreement between passes
reflects the model's reading, not an untested new transform.

Note: The browser implementation truncates luma to int with |0 and writes
through Uint8ClampedArray (which rounds), while this port keeps float32 until
.astype(np.uint8) truncation. Results differ by up to ~1 grey level, which is
immaterial to vision model performance.
"""

import numpy as np
from PIL import Image

# ITU-R BT.601 luma weights, matching the browser implementation.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def contrast_normalise(image: Image.Image) -> Image.Image:
    """Grayscale by luma, then stretch min/max to the full 0-255 range."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = arr @ _LUMA

    lo, hi = float(gray.min()), float(gray.max())
    # A flat image has no range to stretch; guard the division rather than
    # amplifying sensor noise into a false full-contrast image.
    span = max(1.0, hi - lo)

    stretched = np.clip((gray - lo) / span * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([stretched] * 3), "RGB")
