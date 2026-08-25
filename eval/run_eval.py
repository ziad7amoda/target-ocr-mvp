"""Batch scoring against ground truth.

The headline number is the SILENT ERROR RATE: fields the system marked `ok`
that are actually wrong. Everything else in this report is context for it.

A wrong field caught as `review` is the system working. A wrong field
served as `ok` is the failure this product exists to avoid, so those are
listed individually rather than only counted - an aggregate does not say
which failure mode to fix.

Two ways to run this:

  python -m eval.run_eval
      Loads a QwenEngine IN THIS PROCESS. Only use this when nothing else
      already holds the model - e.g. no notebook uvicorn server running.

  python -m eval.run_eval --api https://<tunnel-or-host>
      Scores against a server's /api/extract instead of loading a local
      model. This is the correct choice whenever the notebook server is
      already up: a T4 has ~14.5GB of VRAM, the 3B model alone takes ~8.6GB,
      and loading a SECOND copy for `run_eval` on top of a live server is
      exactly what raises `torch.OutOfMemoryError`. --api mode never
      constructs a QwenEngine or imports torch/transformers at all.
"""

import argparse
import json
import mimetypes
import sys
import unicodedata
import urllib.error
import urllib.request
import uuid
# aliased: SilentError has an attribute called `field`, and the collision
# between that and dataclasses.field is confusing even though it is legal.
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

from PIL import Image

from app.config import get_settings
from app.schema import ARABIC_FIELDS, FIELD_NAMES, CardFields, ExtractResponse

SAMPLES_DIR = Path(__file__).parent / "samples"
IOU_HIT_THRESHOLD = 0.5
# Inference takes ~10s/card on a T4 (spec measured 14-19s); this must be
# generous enough to never be the reason a slow-but-working server looks
# unreachable.
API_TIMEOUT_S = 180

# The keys a per-card `fields` object in expected.json may carry - exactly
# CardFields' keys, i.e. what the model is actually asked to emit. Kept in
# sync with CardFields automatically rather than duplicated by hand.
_VALID_EXPECTED_FIELD_KEYS = set(CardFields.model_fields)


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


def _validate_expected(expected: object, samples_dir: Path) -> None:
    """Fail fast on a malformed expected.json - before any engine loads or
    API calls are made.

    A person hit this in the field: they wrote a single card's entry
    `{"fields": {...}, "boxes": {}}` at the TOP level instead of nesting it
    under an image filename. `main()` then did `for name in expected:`,
    treated the key "fields" as a filename, and raised a confusing
    `FileNotFoundError` for a file literally called "fields" - AFTER a ~40s
    model load. This check makes that mistake fail in under a second,
    with a message that says what is actually wrong.
    """
    if not isinstance(expected, dict):
        raise SystemExit(
            "expected.json must be a JSON object at the top level, keyed by "
            "image filename, e.g.:\n"
            '  {"my_card.jpg": {"fields": {...}, "boxes": {}}}'
        )

    if "fields" in expected or "boxes" in expected:
        raise SystemExit(
            "expected.json looks like a single card's entry; wrap it in an "
            "object keyed by the image filename. Top-level keys must be "
            "image filenames, e.g.:\n"
            '  {"my_card.jpg": {"fields": {...}, "boxes": {}}}\n'
            "not:\n"
            '  {"fields": {...}, "boxes": {}}'
        )

    for name, entry in expected.items():
        if not (samples_dir / name).exists():
            raise SystemExit(
                f"expected.json key {name!r} is not an image file in "
                f"{samples_dir}. Top-level keys must be image filenames, "
                "e.g.:\n"
                '  {"my_card.jpg": {"fields": {...}, "boxes": {}}}'
            )

        if not isinstance(entry, dict) or "fields" not in entry:
            raise SystemExit(
                f"expected.json entry for {name!r} must be an object with a "
                '"fields" key, e.g. {"fields": {...}, "boxes": {}}.'
            )

        fields = entry["fields"]
        if not isinstance(fields, dict):
            raise SystemExit(f"expected.json entry for {name!r}: \"fields\" must be an object.")

        # Partial ground truth is legitimate - score() already handles a
        # field that was never typed in. Only an UNKNOWN key is an error:
        # it usually means old-schema ground truth (nationality/sex) was
        # carried over and would otherwise silently score nothing.
        unknown = set(fields) - _VALID_EXPECTED_FIELD_KEYS
        if unknown:
            raise SystemExit(
                f"expected.json entry for {name!r} has unknown field key(s) "
                f"{sorted(unknown)}. Valid keys are "
                f"{sorted(_VALID_EXPECTED_FIELD_KEYS)}."
            )


def _api_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _unreachable(url: str, exc: Exception) -> str:
    return f"Could not reach {url} ({exc}). Is the API server running?"


def api_health(base_url: str) -> bool | None:
    """GET <base_url>/api/health and return its `self_consistency` flag.

    Returns None (rather than raising) when the server can't be reached:
    the report still needs to run without it, just minus the banner that
    would otherwise reflect the remote server's actual setting.
    """
    url = _api_url(base_url, "/api/health")
    try:
        with urllib.request.urlopen(url, timeout=API_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return bool(payload.get("self_consistency", True))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"  warning: {_unreachable(url, exc)} Continuing without it.", flush=True)
        return None


def _multipart_body(field_name: str, filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    parts = [
        f"--{boundary}".encode(),
        (
            f'Content-Disposition: form-data; name="{field_name}"; '
            f'filename="{filename}"'
        ).encode(),
        f"Content-Type: {content_type}".encode(),
        b"",
        content,
        f"--{boundary}--".encode(),
        b"",
    ]
    return b"\r\n".join(parts), boundary


def api_extract(base_url: str, image_path: Path) -> ExtractResponse:
    """POST one card image to <base_url>/api/extract and parse the result.

    Field name is `image` to match the FastAPI parameter
    (`image: UploadFile = File(...)` in app/main.py).
    """
    url = _api_url(base_url, "/api/extract")
    body, boundary = _multipart_body("image", image_path.name, image_path.read_bytes())
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"{_unreachable(url, exc)} Start the notebook's uvicorn server, "
            "then pass its base URL via --api."
        ) from exc
    return ExtractResponse.model_validate(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples", type=Path, default=SAMPLES_DIR)
    parser.add_argument(
        "--api",
        metavar="BASE_URL",
        default=None,
        help=(
            "Score against a running server's /api/extract instead of loading "
            "a local model (e.g. --api https://xxxx.trycloudflare.com). Use "
            "this when the notebook server is already up - loading a second "
            "model copy will OOM a T4."
        ),
    )
    args = parser.parse_args(argv)

    expected = json.loads((args.samples / "expected.json").read_text(encoding="utf-8"))
    # Before constructing any engine or making any API call: a malformed
    # expected.json should fail in under a second, not after a model load
    # or a round trip to a remote server.
    _validate_expected(expected, args.samples)
    results: dict[str, ExtractResponse] = {}

    if args.api:
        self_consistency = api_health(args.api)
        if self_consistency is None:
            self_consistency = True  # unknown; don't show a false "OFF" banner
        for name in expected:
            results[name] = api_extract(args.api, args.samples / name)
            print(f"  scored {name}", flush=True)
    else:
        # Imported here, not at module level: this branch is the only one
        # that needs torch, and eval/run_eval.py must stay importable (and
        # --api usable) without it - see tests/test_import_hygiene.py.
        from app.extract import extract
        from app.model import QwenEngine

        settings = get_settings()
        self_consistency = settings.SELF_CONSISTENCY

        engine = QwenEngine(settings)
        engine.load()
        engine.warmup()

        for name in expected:
            image = Image.open(args.samples / name).convert("RGB")
            results[name] = extract(image, engine, settings, engine.processed_size(image))
            print(f"  scored {name}", flush=True)

    print()
    print_report(score(results, expected), self_consistency)
    return 0


if __name__ == "__main__":
    sys.exit(main())
