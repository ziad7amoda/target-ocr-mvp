"""Write a synthetic training corpus.

    python -m scripts.cardgen.generate --count 5000 --out data/synthetic

Every image is paired with the exact production JSON, taken from the content
object rather than from anything that was drawn - text is reshaped into
visual-order presentation forms on its way to the renderer, so reading a
label back off the image would produce mangled text that happens to look
right.

Two properties of the output are deliberate.

The corpus is marked. Every PNG carries metadata saying it is synthetic, and
the manifest records the generator's commit-independent version and seed.
These images have no security features, no emblem and no real portrait, so
they are not a copy of an identity document; the marking is there so that
nobody downstream has to work that out for themselves.

The corpus is separable. Synthetic and real cards must live in directories an
eval loader cannot silently merge, because a model that scores well on
synthetic and badly on real is the standard way this goes wrong, and you only
notice if the two were never mixed. Train mostly on this. Evaluate only on
real cards.
"""

import argparse
import json
import random
from pathlib import Path

from PIL import PngImagePlugin

from scripts.cardgen.content import generate as generate_content
from scripts.cardgen.degrade import degrade
from scripts.cardgen.render import render

GENERATOR = "target-ocr-mvp/cardgen"
MARKING = "SYNTHETIC TRAINING IMAGE - not a real document, no security features"


def _write_image(image, path: Path, seed: int, index: int) -> None:
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Software", GENERATOR)
    meta.add_text("Comment", MARKING)
    meta.add_text("Source", f"seed={seed} index={index}")
    image.save(path, "PNG", pnginfo=meta)


def build(count: int, out_dir: Path, seed: int, clean: bool = False) -> Path:
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index in range(count):
            content, printed = generate_content(rng)
            card = render(content, printed, rng)
            if not clean:
                card = degrade(card, rng)

            name = f"card_{index:06d}.png"
            _write_image(card, images / name, seed, index)
            manifest.write(
                json.dumps(
                    {
                        "image": f"images/{name}",
                        "fields": content.ground_truth(),
                        "synthetic": True,
                        "generator": GENERATOR,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Skip the degradation pass. For eyeballing the layout only - a "
            "corpus built this way trains a model to read clean renders, "
            "which is not a problem anybody has."
        ),
    )
    args = parser.parse_args()

    manifest = build(args.count, args.out, args.seed, args.clean)
    print(f"wrote {args.count} cards to {args.out}")
    print(f"manifest: {manifest}")
    if args.clean:
        print("NOTE: --clean was used. Not suitable for training.")


if __name__ == "__main__":
    main()
