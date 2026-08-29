"""Turning a clean render into something that looks photographed.

This module is the reason the synthetic set is worth building. A clean
render teaches a model to read clean renders, which is a problem nobody has.
Everything the real card photographs showed - perspective, specular glare
across the laminate, uneven white balance, compression, a card lying on a
desk - is applied here, and it is what makes the training transfer.

The order follows the physical path of the image: the card is placed in the
world and lit, then the lens blurs it, then the sensor adds noise, and only
at the end is it compressed and resized. Applying JPEG artefacts before a
blur would smooth away the very artefacts we are trying to teach.
"""

import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _perspective_coeffs(src, dst):
    """Solve for the eight coefficients PIL's PERSPECTIVE transform wants.

    PIL maps OUTPUT coordinates back to INPUT, so the correspondence is
    given in that direction. Reversing it produces a warp in the opposite
    sense, which still looks plausible and is still wrong.
    """
    matrix = []
    for (x, y), (u, v) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.array(matrix, dtype=np.float64)
    b = np.array(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(a, b)


def perspective(card, rng, strength=0.06):
    """Tilt the card as if photographed off-axis."""
    w, h = card.size
    pad_w, pad_h = int(w * 0.14), int(h * 0.14)
    canvas = Image.new("RGBA", (w + pad_w * 2, h + pad_h * 2), (0, 0, 0, 0))
    canvas.paste(card.convert("RGBA"), (pad_w, pad_h))
    cw, ch = canvas.size

    def jitter(x, y):
        return (
            x + rng.uniform(-strength, strength) * cw,
            y + rng.uniform(-strength, strength) * ch,
        )

    src = [(0, 0), (cw, 0), (cw, ch), (0, ch)]
    dst = [jitter(*p) for p in src]
    coeffs = _perspective_coeffs(src, dst)
    return canvas.transform((cw, ch), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def on_surface(card, rng):
    """Composite the card onto a desk, table or countertop.

    A card photographed in isolation on white does not exist. The model has
    to find the card before it can read it, and a plain background makes
    that step artificially easy.
    """
    w, h = card.size
    margin = rng.uniform(0.06, 0.20)
    bw, bh = int(w * (1 + margin)), int(h * (1 + margin))

    # Surfaces a card actually gets photographed on: wood, stone, laminate,
    # fabric, a grey desk. Sampling the three channels independently gives
    # saturated primaries instead - the first version of this produced cards
    # lying on bright red and green, which teaches the model to expect a
    # background no counter in a bank has.
    tone = rng.randint(55, 195)
    warmth = rng.uniform(-0.10, 0.22)  # negative reads cool/grey, positive wood
    base = np.array(
        [tone * (1 + warmth), tone * (1 + warmth * 0.45), tone * (1 - warmth * 0.55)],
        dtype=np.float32,
    )
    base = np.clip(base, 25, 225)
    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
    grain = (
        np.sin(xx * rng.uniform(0.01, 0.12) + rng.uniform(0, 6)) * rng.uniform(3, 14)
        + np.sin(yy * rng.uniform(0.01, 0.12)) * rng.uniform(2, 9)
    )
    surface = np.clip(base[None, None, :] + grain[:, :, None], 0, 255).astype(np.uint8)
    out = Image.fromarray(surface, "RGB")

    x = (bw - w) // 2 + rng.randint(-bw // 40, bw // 40)
    y = (bh - h) // 2 + rng.randint(-bh // 40, bh // 40)

    # A contact shadow, so the card sits on the surface instead of floating.
    if card.mode == "RGBA":
        alpha = card.split()[-1]
        shadow = Image.new("L", (bw, bh), 0)
        shadow.paste(alpha, (x + rng.randint(3, 12), y + rng.randint(3, 12)))
        shadow = shadow.filter(ImageFilter.GaussianBlur(rng.uniform(4, 12)))
        dark = Image.new("RGB", (bw, bh), (20, 20, 24))
        out = Image.composite(dark, out, shadow.point(lambda v: int(v * 0.45)))
        out.paste(card, (x, y), card)
    else:
        out.paste(card, (x, y))
    return out


def glare(img, rng):
    """Specular reflection off the laminate.

    Every real photograph of this card has some. On the resident card we
    were given it washes out the top third, which is why the header and the
    serial were the least reliably read parts of that image.
    """
    w, h = img.size
    layer = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(layer)
    for _ in range(rng.randint(1, 3)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h * 0.75)
        rx = rng.uniform(w * 0.12, w * 0.55)
        ry = rng.uniform(h * 0.06, h * 0.35)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=rng.randint(70, 190))
    layer = layer.filter(ImageFilter.GaussianBlur(rng.uniform(20, 70)))

    arr = np.array(img, dtype=np.float32)
    mask = np.array(layer, dtype=np.float32)[:, :, None] / 255.0
    lifted = arr + (255 - arr) * mask * rng.uniform(0.35, 0.85)
    return Image.fromarray(np.clip(lifted, 0, 255).astype(np.uint8))


def lighting(img, rng):
    """Uneven illumination and a white-balance cast."""
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    ramp = (xx / w) * np.cos(angle) + (yy / h) * np.sin(angle)
    spread = float(np.ptp(ramp))
    ramp = (ramp - ramp.min()) / max(spread, 1e-6)

    arr = np.array(img, dtype=np.float32)
    arr *= 1.0 + (ramp[:, :, None] - 0.5) * rng.uniform(0.10, 0.45)
    # White balance moves along the blue-orange temperature axis, with a
    # small green-magenta tint. Scaling the three channels independently
    # instead - the first version here - shifted a pale blue card to green
    # or pink, which teaches a colour prior no card has. The card is a cool
    # near-neutral and must stay recognisably in that family.
    temperature = rng.uniform(-0.06, 0.06)   # positive warm, negative cool
    tint = rng.uniform(-0.02, 0.02)
    cast = np.array(
        [1 + temperature, 1 + tint, 1 - temperature], dtype=np.float32
    )
    arr *= cast[None, None, :]
    arr *= rng.uniform(0.72, 1.12)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def optics(img, rng):
    """Lens blur, and sometimes a little motion."""
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.8)))
    if rng.random() < 0.3:
        dx, dy = rng.randint(-3, 3), rng.randint(-3, 3)
        shifted = img.transform(
            img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), Image.BILINEAR
        )
        img = Image.blend(img, shifted, 0.5)
    return img


def sensor(img, rng):
    """Shot noise."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(
        0, rng.uniform(1.5, 9.0), arr.shape
    )
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def capture_resolution(img, rng):
    """Resize to a plausible photographed width.

    Spread deliberately wide. The model has to cope with a 900px frame from
    a distant shot and a 3000px close-up, and a corpus rendered at one size
    teaches it neither.
    """
    target = rng.choice([900, 1100, 1400, 1700, 2100, 2600, 3000])
    scale = target / img.width
    resample = Image.LANCZOS if scale < 1 else Image.BICUBIC
    return img.resize((target, max(1, int(img.height * scale))), resample)


def compress(img, rng):
    """JPEG, at the quality a phone or a messaging app would use."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=rng.randint(38, 92))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade(card, rng):
    """The full path from clean render to something worth training on."""
    img = perspective(card, rng)
    img = on_surface(img, rng)
    img = lighting(img, rng)
    if rng.random() < 0.75:
        img = glare(img, rng)
    img = optics(img, rng)
    img = sensor(img, rng)
    img = capture_resolution(img, rng)
    return compress(img, rng)
