"""Bounding box rescaling and sanity filtering.

Spec D4 and §5.3: a box drawn 40px off a date is worse than no box at all,
so every box must earn its place. Failures here are silent per field - that
field simply does not highlight on hover - rather than drawing something
wrong and calling it a feature.

The overall reliability question is answered empirically by box hit rate in
eval/run_eval.py, not by assumption.
"""

from app.schema import FIELD_NAMES

# Fraction of image area a legitimate field box may occupy.
_MIN_AREA_FRAC = 0.0005
_MAX_AREA_FRAC = 0.40
# Printed field lines are wide, never tall.
_MIN_ASPECT, _MAX_ASPECT = 0.5, 30.0
_MIN_HEIGHT_FRAC, _MAX_HEIGHT_FRAC = 0.01, 0.25
# Beyond this much overshoot the model is guessing, not rounding.
_BOUNDS_SLOP = 0.02
# More than this many fields sharing one box means the model collapsed.
_MAX_SHARED_CLAIMS = 2


def rescale_box(
    box: list[int], from_size: tuple[int, int], to_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Map a box from the processor's resized space into the original image.

    Qwen2.5-VL emits absolute pixels against the image the processor
    produced, not the one that was uploaded (spec §4.2). Skipping this step
    silently draws every box in the wrong place on large photos.
    """
    fx, fy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    x1, y1, x2, y2 = box
    return (int(round(x1 * fx)), int(round(y1 * fy)), int(round(x2 * fx)), int(round(y2 * fy)))


def _is_plausible(box: tuple[int, int, int, int], size: tuple[int, int]) -> bool:
    w, h = size
    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        return False

    bw, bh = x2 - x1, y2 - y1
    if not _MIN_AREA_FRAC <= (bw * bh) / (w * h) <= _MAX_AREA_FRAC:
        return False
    if not _MIN_ASPECT <= bw / bh <= _MAX_ASPECT:
        return False
    if not _MIN_HEIGHT_FRAC <= bh / h <= _MAX_HEIGHT_FRAC:
        return False
    return True


def filter_boxes(
    raw: dict[str, list[int]],
    processed_size: tuple[int, int],
    image_size: tuple[int, int],
) -> dict[str, tuple[int, int, int, int]]:
    """Rescale, then keep only boxes that could plausibly be a printed field."""
    w, h = image_size
    slop_x, slop_y = w * _BOUNDS_SLOP, h * _BOUNDS_SLOP

    # A box claimed by too many fields is model collapse, not grounding, and
    # must be dropped for every claimant - including the one that might have
    # been right, since there is no way to tell which.
    counts: dict[tuple[int, ...], int] = {}
    for field, box in raw.items():
        if field in FIELD_NAMES:
            counts[tuple(box)] = counts.get(tuple(box), 0) + 1

    out: dict[str, tuple[int, int, int, int]] = {}
    for field, box in raw.items():
        if field not in FIELD_NAMES:
            continue
        if counts[tuple(box)] > _MAX_SHARED_CLAIMS:
            continue

        scaled = rescale_box(box, processed_size, image_size)
        x1, y1, x2, y2 = scaled

        # Small overshoot is rounding and gets clamped; large overshoot means
        # the coordinates are not trustworthy at all.
        if x1 < -slop_x or y1 < -slop_y or x2 > w + slop_x or y2 > h + slop_y:
            continue
        clamped = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))

        if _is_plausible(clamped, image_size):
            out[field] = clamped

    return out
