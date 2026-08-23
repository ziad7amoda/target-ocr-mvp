import numpy as np
from PIL import Image

from app.imaging import contrast_normalise


def test_stretches_a_low_contrast_image_to_full_range():
    # Every pixel between 100 and 150: should end up spanning 0-255.
    arr = np.linspace(100, 150, 64 * 64).reshape(64, 64).astype(np.uint8)
    src = Image.fromarray(np.dstack([arr, arr, arr]), "RGB")

    out = np.asarray(contrast_normalise(src).convert("L"))

    assert out.min() == 0
    assert out.max() == 255


def test_output_is_grayscale_but_still_rgb_mode():
    src = Image.new("RGB", (32, 32), (200, 40, 40))
    out = contrast_normalise(src)
    px = np.asarray(out)
    assert out.mode == "RGB"
    assert (px[:, :, 0] == px[:, :, 1]).all() and (px[:, :, 1] == px[:, :, 2]).all()


def test_flat_image_does_not_divide_by_zero(recwarn):
    """The max(1.0, hi-lo) guard is load-bearing: without it numpy returns nan
    (a RuntimeWarning, not an exception) and nan.astype(uint8) silently becomes 0,
    so a flat image would still 'work' while the guard was quietly gone."""
    src = Image.new("RGB", (16, 16), (128, 128, 128))
    out = np.asarray(contrast_normalise(src).convert("L"))

    assert out.shape == (16, 16)
    assert not np.isnan(out.astype(np.float32)).any()
    assert (out == 0).all()          # (128-128)/1.0*255 == 0, deterministic
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_preserves_dimensions():
    src = Image.new("RGB", (123, 45), "white")
    assert contrast_normalise(src).size == (123, 45)
