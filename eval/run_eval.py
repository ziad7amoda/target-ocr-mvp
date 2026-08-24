"""Batch scoring against ground truth.

The headline number is the SILENT ERROR RATE: fields the system marked `ok`
that are actually wrong. Everything else in this report is context for it.

A wrong field caught as `review` is the system working. A wrong field
served as `ok` is the failure this product exists to avoid, so those are
listed individually rather than only counted - an aggregate does not say
which failure mode to fix.
"""

import argparse
import json
import sys
import unicodedata
# aliased: SilentError has an attribute called `field`, and the collision
# between that and dataclasses.field is confusing even though it is legal.
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

from PIL import Image

from app.config import get_settings
from app.schema import ARABIC_FIELDS, FIELD_NAMES, ExtractResponse

SAMPLES_DIR = Path(__file__).parent / "samples"
IOU_HIT_THRESHOLD = 0.5


@dataclass
class SilentError:
    image: str
    field: str
    got: str
    expected: str


@dataclass
class Report:
    per_field: dict[str, dict[str, int]]
    silent_errors: list[SilentError] = dc_field(default_factory=list)
    box_hits: int = 0
    box_total: int = 0
    # Fields that HAVE a ground-truth box, whether or not one was returned.
    # box_total only counts boxes that survived the filter, so it alone
    # would report a flattering hit rate on whatever fraction the filter let
    # through; box_expected is the true denominator for coverage.
    box_expected: int = 0
    elapsed_ms: list[int] = dc_field(default_factory=list)


def _norm(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union


def score(results: dict[str, ExtractResponse], expected: dict) -> Report:
    per_field = {f: {"exact": 0, "ok": 0, "review": 0, "missing": 0, "n": 0} for f in FIELD_NAMES}
    report = Report(per_field=per_field)

    for name, response in results.items():
        truth = expected[name]
        report.elapsed_ms.append(response.elapsed_ms)

        for fname in FIELD_NAMES:
            got = response.fields[fname]
            want = truth["fields"].get(fname)
            row = per_field[fname]
            row["n"] += 1
            row[got.status] += 1

            correct = _norm(got.value) == _norm(want)
            if correct:
                row["exact"] += 1
            elif got.status == "ok":
                report.silent_errors.append(
                    SilentError(name, fname, got.value or "", want or "")
                )

            # The Arabic value is scored too: it is the field a bank
            # actually keys in (spec), and a wrong value_ar served as `ok`
            # is just as silent a failure as a wrong Latin value. Only
            # checked when the ground truth actually provides it.
            if fname in ARABIC_FIELDS:
                want_ar = truth["fields"].get(f"{fname}_ar")
                if want_ar is not None and _norm(got.value_ar) != _norm(want_ar):
                    if got.status == "ok":
                        report.silent_errors.append(
                            SilentError(name, f"{fname}_ar", got.value_ar or "", want_ar)
                        )

            # box_total only scores boxes that were actually returned: a box
            # the filter dropped is a non-answer, not a wrong answer. But
            # that makes it a self-selecting rate over a shrinking
            # denominator, so box_expected also counts every field that HAS
            # a ground-truth box regardless of whether one came back -
            # print_report reports both.
            true_box = truth.get("boxes", {}).get(fname)
            if true_box is not None:
                report.box_expected += 1
            if got.box is not None and true_box is not None:
                report.box_total += 1
                if iou(tuple(got.box), tuple(true_box)) >= IOU_HIT_THRESHOLD:
                    report.box_hits += 1

    return report


def print_report(report: Report, self_consistency: bool) -> None:
    if not self_consistency:
        print("!" * 68)
        print("! SELF_CONSISTENCY IS OFF. Hallucination detection is disabled, so")
        print("! the silent error rate below is NOT comparable to a normal run.")
        print("!" * 68)
        print()

    print(f"{'FIELD':<18}{'EXACT':>8}{'OK':>7}{'REVIEW':>9}{'MISSING':>9}")
    for fname, row in report.per_field.items():
        print(
            f"{fname:<18}{row['exact']:>4}/{row['n']:<3}{row['ok']:>7}"
            f"{row['review']:>9}{row['missing']:>9}"
        )

    total = sum(r["n"] for r in report.per_field.values())
    n_silent = len(report.silent_errors)
    pct = (n_silent / total * 100) if total else 0.0

    print()
    print(f"SILENT ERRORS: {n_silent}/{total} ({pct:.1f}%)")
    for err in report.silent_errors:
        print(f"  {err.image:<20} {err.field:<16} got {err.got!r} expected {err.expected!r}")

    if report.box_expected:
        coverage = report.box_total / report.box_expected * 100
        print()
        print(
            f"BOX COVERAGE: {report.box_total}/{report.box_expected} fields "
            f"returned a box ({coverage:.1f}%)"
        )
    else:
        print()
        print("BOX COVERAGE: no fields have ground-truth boxes")

    if report.box_total:
        rate = report.box_hits / report.box_total * 100
        print()
        print(f"BOX HIT RATE: {report.box_hits}/{report.box_total} ({rate:.1f}%) at IoU>={IOU_HIT_THRESHOLD}")
        print("  (measured only over boxes actually returned - see BOX COVERAGE")
        print("  above for how many fields had no box at all; a filter that")
        print("  drops most boxes can still report a flattering hit rate here.)")
    else:
        print()
        print("BOX HIT RATE: no boxes returned")

    if report.elapsed_ms:
        ms = sorted(report.elapsed_ms)
        print()
        print(f"LATENCY: median {ms[len(ms) // 2]} ms, max {ms[-1]} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=SAMPLES_DIR)
    args = parser.parse_args()

    from app.extract import extract
    from app.model import QwenEngine

    expected = json.loads((args.samples / "expected.json").read_text(encoding="utf-8"))
    settings = get_settings()

    engine = QwenEngine(settings)
    engine.load()
    engine.warmup()

    results: dict[str, ExtractResponse] = {}
    for name in expected:
        image = Image.open(args.samples / name).convert("RGB")
        results[name] = extract(image, engine, settings, engine.processed_size(image))
        print(f"  scored {name}", flush=True)

    print()
    print_report(score(results, expected), settings.SELF_CONSISTENCY)
    return 0


if __name__ == "__main__":
    sys.exit(main())
