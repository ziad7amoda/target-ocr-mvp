"""Task 3 measurement gate (spec §13 step 2).

Run on Colab BEFORE building the rest of the pipeline. Reports the three
numbers the latency budget in spec §15 is predicated on.

    python scripts/bringup.py path/to/card.jpg
"""

import sys
import time

from PIL import Image

from app.config import get_settings
from app.model import GenerationRequest, QwenEngine

PROMPT = (
    "Read this ID card. Return ONLY a JSON object with keys full_name, "
    "id_number, date_of_birth. Use null for anything not clearly legible."
)


def main(path: str) -> None:
    settings = get_settings()
    image = Image.open(path).convert("RGB")

    t0 = time.perf_counter()
    engine = QwenEngine(settings)
    engine.load()
    load_s = time.perf_counter() - t0

    warm_ms = engine.warmup()

    t0 = time.perf_counter()
    first = engine.generate([GenerationRequest(image=image, prompt=PROMPT)])
    single_ms = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    engine.generate([
        GenerationRequest(image=image, prompt=PROMPT),
        GenerationRequest(image=image, prompt=PROMPT),
        GenerationRequest(image=image, prompt=PROMPT),
    ])
    batch3_ms = int((time.perf_counter() - t0) * 1000)

    print("=" * 60)
    print(f"model            {settings.MODEL_ID}")
    print(f"device           {engine.device}")
    print(f"weight load      {load_s:6.1f} s")
    print(f"warm-up          {warm_ms:6d} ms")
    print(f"warm, batch of 1 {single_ms:6d} ms")
    print(f"warm, batch of 3 {batch3_ms:6d} ms   <- the D5 claim")
    print(f"peak VRAM        {engine.vram_mb():6d} MB")
    # If this equals image.size on a large photo, processed_size() fell back
    # and every grounding box will be drawn in the wrong place.
    print(f"original size    {image.size}")
    print(f"processed size   {engine.processed_size(image)}")
    print("=" * 60)
    print("model output:")
    print(first[0])


if __name__ == "__main__":
    main(sys.argv[1])
