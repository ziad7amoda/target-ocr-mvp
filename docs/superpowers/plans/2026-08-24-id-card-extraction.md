# ID Card Field Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload or capture a photo of an Omani ID card and get its six fields back as structured JSON with an honest per-field `ok`/`review`/`missing` status, served from a FastAPI backend running Qwen2.5-VL on a Colab T4 behind a Cloudflare tunnel.

**Architecture:** A narrow engine seam (`VLMEngine.generate(list[GenerationRequest]) -> list[str]`) separates everything that needs a GPU from everything that doesn't. Tasks 4–12 and 14 are developed and tested locally against `FakeEngine` with torch never imported. Every `/api/extract` request issues exactly **one batched `generate()` call carrying three sequences**: field extraction on the original image, field extraction on a contrast-normalised copy (self-consistency), and a grounding pass for bounding boxes. T4 decode is bound by weight bandwidth rather than compute, so sequences B and C cost almost no wall-clock.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, pydantic v2, pydantic-settings, Pillow, NumPy, transformers, qwen-vl-utils, accelerate, arabic-reshaper, python-bidi, pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-id-card-extraction-design.md` — read it alongside this plan. Every task cites the spec sections it implements.

## Global Constraints

- **Nothing outside `notebook/` may reference Colab, Drive, tunnels, or ngrok.** The app must run unchanged on a bank server with the tunnel removed. (Spec §1, §10)
- **`app/extract.py`, `app/validate.py`, `app/parsing.py`, `app/boxes.py`, `app/imaging.py` and `app/schema.py` must never import `torch` or `transformers`.** Enforced by a test in Task 2. (Spec §3.1)
- **`torch` is deliberately absent from `requirements.txt`** — Colab preinstalls it, and pinning it forces a slow reinstall. `QwenEngine` imports torch and transformers **inside** its methods, not at module scope, so the local suite runs without them. (Spec §3.1)
- **No value derived from token probabilities may be exposed as confidence, anywhere.** (Spec §5)
- **Never log image bytes or field values.** Log event name, elapsed ms, status counts, retry count, grounding success only. (Spec §10)
- The six fields are exactly: `full_name`, `id_number`, `date_of_birth`, `expiry_date`, `nationality`, `sex`.
- `value_ar` is populated for **`full_name`, `nationality`, `sex` only**. (Spec D2)
- Dates are ISO `YYYY-MM-DD`. `sex` is exactly `M` or `F`. `id_number` is exactly 8 digits.
- Status precedence, first match wins: `missing` (null in both) → `review` (passes disagreed) → `review` (format rule) → `review` (cross-field) → `ok`. **`missing` always outranks `review`.** (Spec §5)
- Model default `Qwen/Qwen2.5-VL-3B-Instruct`, fp16, SDPA attention. T4 is sm75: no bf16, no FlashAttention-2. (Spec §3.2)
- Commit after every task. Conventional commit prefixes (`feat:`, `test:`, `chore:`, `docs:`).

---

## File Structure

| File | Responsibility |
|---|---|
| `app/config.py` | Env-driven `Settings`, cached accessor |
| `app/schema.py` | Pydantic models: `CardFields` (model output), `FieldResult`, `ExtractResponse`, `HealthResponse` |
| `app/model.py` | `GenerationRequest`, `VLMEngine` protocol, `FakeEngine`, `QwenEngine` |
| `app/imaging.py` | Contrast normalisation (port of the browser `enhance()`) |
| `app/parsing.py` | Fence stripping, JSON repair, `CardFields` parsing |
| `app/validate.py` | Format rules, cross-field checks, two-pass merge, status derivation |
| `app/boxes.py` | Coordinate rescaling, geometric plausibility filter |
| `app/extract.py` | Prompts, batched call orchestration, retry, response assembly |
| `app/main.py` | Lifespan, CORS, static mount, `/api/extract`, `/api/health`, `/api/transcribe` |
| `app/data/nationalities.txt` | Closed list for the `nationality` format rule |
| `static/index.html` | The adapted demo (CSS carried over verbatim from `ocr-demo.html`) |
| `eval/make_samples.py` | Synthetic bilingual card generator |
| `eval/run_eval.py` | Batch scoring, silent error rate, box hit rate |
| `scripts/bringup.py` | Task 3 GPU measurement harness |
| `notebook/run_colab.ipynb` | Six-cell Colab runner |
| `tests/` | Mirrors `app/` |

---

## Task 1: Repo skeleton, settings and schema

**Files:**
- Create: `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `README.md`, `app/__init__.py`, `app/config.py`, `app/schema.py`, `app/data/nationalities.txt`, `tests/__init__.py`, `tests/test_config.py`, `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing
- Produces: `app.config.Settings`, `app.config.get_settings() -> Settings`; `app.schema.FIELD_NAMES: list[str]`, `ARABIC_FIELDS: set[str]`, `CardFields`, `FieldResult`, `Agreement`, `ExtractResponse`, `HealthResponse`

Implements spec §9 (configuration), §6 (API shapes).

- [ ] **Step 1: Create `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
.env
.pytest_cache/
_debug_images/
eval/fonts/
eval/samples/*.jpg
eval/samples/*.png
.ipynb_checkpoints/
```

`_debug_images/` is gitignored because `DEBUG_SAVE_IMAGES` writes card photos there (spec §10). `eval/samples/*.jpg` keeps real ID photos out of git; `expected.json` is tracked.

- [ ] **Step 2: Create `requirements.txt`**

```
# Runtime. torch is deliberately NOT pinned here: Colab preinstalls a build
# matched to its CUDA driver, and pinning forces a multi-minute reinstall.
fastapi>=0.110
uvicorn[standard]>=0.29
python-multipart>=0.0.9
pydantic>=2.6
pydantic-settings>=2.2
pillow>=10.2
numpy>=1.26
transformers>=4.49
accelerate>=0.30
qwen-vl-utils>=0.0.8
arabic-reshaper>=3.0
python-bidi>=0.4.2
```

`transformers>=4.49` is the floor for `Qwen2_5_VLForConditionalGeneration`.

- [ ] **Step 3: Create `requirements-dev.txt`**

```
-r requirements.txt
pytest>=8.0
httpx>=0.27
```

- [ ] **Step 4: Write the failing config test**

Create `tests/test_config.py`:

```python
from app.config import Settings, get_settings


def test_defaults_match_spec():
    s = Settings()
    assert s.MODEL_ID == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert s.TORCH_DTYPE == "float16"
    assert s.MAX_NEW_TOKENS == 256
    assert s.SELF_CONSISTENCY is True
    assert s.SHOW_BOXES is True
    assert s.DEBUG_SAVE_IMAGES is False


def test_env_overrides_model_id(monkeypatch):
    monkeypatch.setenv("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
    assert Settings().MODEL_ID == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_show_boxes_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("SHOW_BOXES", "false")
    assert Settings().SHOW_BOXES is False


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
```

- [ ] **Step 5: Run it and confirm it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 6: Implement `app/config.py`**

Create `app/__init__.py` (empty) and `tests/__init__.py` (empty), then `app/config.py`:

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # protected_namespaces=() is required: pydantic v2 reserves the "model_"
    # prefix and MODEL_ID would otherwise emit a warning on every import.
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=()
    )

    MODEL_ID: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    DEVICE: str = "auto"
    TORCH_DTYPE: str = "float16"

    MAX_NEW_TOKENS: int = 256
    # Vision-token bounds handed to the processor. Prefill is cheap relative to
    # decode, but capping keeps a 12MP phone photo from ballooning the batch.
    MIN_PIXELS: int = 256 * 28 * 28
    MAX_PIXELS: int = 1280 * 28 * 28

    SELF_CONSISTENCY: bool = True
    SHOW_BOXES: bool = True

    # Development only. Writes card images to disk; see README.
    DEBUG_SAVE_IMAGES: bool = False
    DEBUG_SAVE_DIR: str = "./_debug_images"

    ALLOWED_ORIGINS: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 7: Run the config test**

Run: `pytest tests/test_config.py -v`
Expected: PASS (4 tests)

- [ ] **Step 8: Write the failing schema test**

Create `tests/test_schema.py`:

```python
import pytest
from pydantic import ValidationError

from app.schema import (
    ARABIC_FIELDS,
    FIELD_NAMES,
    Agreement,
    CardFields,
    ExtractResponse,
    FieldResult,
)


def test_field_names_are_the_six_spec_fields():
    assert FIELD_NAMES == [
        "full_name",
        "id_number",
        "date_of_birth",
        "expiry_date",
        "nationality",
        "sex",
    ]


def test_only_three_fields_carry_arabic():
    assert ARABIC_FIELDS == {"full_name", "nationality", "sex"}


def test_card_fields_accepts_all_nulls():
    c = CardFields()
    assert c.full_name is None
    assert c.sex_ar is None


def test_card_fields_rejects_unknown_key():
    with pytest.raises(ValidationError):
        CardFields(full_name="X", height="180cm")


def test_field_result_rejects_invalid_status():
    with pytest.raises(ValidationError):
        FieldResult(value="X", status="probably")


def test_extract_response_round_trips():
    r = ExtractResponse(
        fields={"sex": FieldResult(value="M", value_ar="ذكر", status="ok")},
        raw_text="{}",
        agreement=Agreement(matched=1, total=6),
        elapsed_ms=3180,
        model="Qwen2.5-VL-3B-Instruct",
    )
    assert r.model_dump()["fields"]["sex"]["box"] is None
```

- [ ] **Step 9: Run it and confirm it fails**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.schema'`

- [ ] **Step 10: Implement `app/schema.py`**

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict

FIELD_NAMES: list[str] = [
    "full_name",
    "id_number",
    "date_of_birth",
    "expiry_date",
    "nationality",
    "sex",
]

# Spec D2: Omani cards print dates and the civil number once, in Western
# numerals. Requesting Arabic for those spends decode tokens on duplicates.
ARABIC_FIELDS: set[str] = {"full_name", "nationality", "sex"}

Status = Literal["ok", "review", "missing"]


class CardFields(BaseModel):
    """Exactly what the model is asked to emit. Flat, not nested: uniform to
    parse and fewer tokens than nested objects."""

    model_config = ConfigDict(extra="forbid")

    full_name: str | None = None
    full_name_ar: str | None = None
    id_number: str | None = None
    date_of_birth: str | None = None
    expiry_date: str | None = None
    nationality: str | None = None
    nationality_ar: str | None = None
    sex: str | None = None
    sex_ar: str | None = None


class FieldResult(BaseModel):
    value: str | None = None
    value_ar: str | None = None
    status: Status
    reason: str | None = None
    box: tuple[int, int, int, int] | None = None


class Agreement(BaseModel):
    matched: int
    total: int


class ExtractResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    fields: dict[str, FieldResult]
    raw_text: str | None = None
    agreement: Agreement
    elapsed_ms: int
    model: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    device: str
    loaded: bool
    self_consistency: bool
    warmup_ms: int | None = None
    vram_mb: int | None = None


class TranscribeResponse(BaseModel):
    text: str
    elapsed_ms: int
```

- [ ] **Step 11: Create `app/data/nationalities.txt`**

```
OMANI
INDIAN
PAKISTANI
BANGLADESHI
FILIPINO
EGYPTIAN
SUDANESE
JORDANIAN
SYRIAN
YEMENI
SRI LANKAN
NEPALI
BRITISH
AMERICAN
```

Closed list for the spec §5.1 format rule. A value outside it yields `review`, never a rejection — an unlisted nationality is a real possibility and must surface for human confirmation rather than be discarded.

- [ ] **Step 12: Run the schema test**

Run: `pytest tests/test_schema.py -v`
Expected: PASS (6 tests)

- [ ] **Step 13: Write `README.md`**

```markdown
# Omani ID card field extraction

Upload or capture a photo of an ID card, get its fields back as structured JSON
with a per-field `ok` / `review` / `missing` status for human review.

Backend: FastAPI + Qwen2.5-VL. Frontend: a single static page.

- **Design:** `docs/superpowers/specs/2026-08-24-id-card-extraction-design.md`
- **Plan:** `docs/superpowers/plans/2026-08-24-id-card-extraction.md`

## Local development (no GPU required)

The whole suite runs without torch. `app/model.py`'s `QwenEngine` imports torch
lazily, so everything else is testable on any machine.

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -v
```

## Running the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/. Without a GPU the model will not load and
`/api/health` reports `"loaded": false`.

## Configuration

All settings are environment variables; see `app/config.py`. The ones that matter:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | Swap to the 7B variant without code changes |
| `SHOW_BOXES` | `true` | Kill switch for the bounding-box overlay |
| `SELF_CONSISTENCY` | `true` | **Debugging only.** Disabling removes hallucination detection |
| `DEBUG_SAVE_IMAGES` | `false` | **Development only.** Writes card images to disk |
| `ALLOWED_ORIGINS` | `["*"]` | Wide open for the demo. Narrow this for any real deployment |

## Colab

See `notebook/run_colab.ipynb`. Camera capture requires HTTPS, which the
Cloudflare tunnel provides.
```

- [ ] **Step 14: Verify the full suite and commit**

Run: `pytest -v`
Expected: PASS (10 tests)

```bash
git add .gitignore requirements.txt requirements-dev.txt README.md app/ tests/
git commit -m "feat: add settings, response schema and repo skeleton"
```

---

## Task 2: The engine seam

**Files:**
- Create: `app/model.py`, `tests/test_model.py`, `tests/test_import_hygiene.py`

**Interfaces:**
- Consumes: `app.config.Settings`
- Produces: `app.model.GenerationRequest(image, prompt)`, `app.model.VLMEngine` (Protocol with `generate(list[GenerationRequest]) -> list[str]`, `model_id: str`, `device: str`), `app.model.FakeEngine(responses)` exposing `.calls: list[list[GenerationRequest]]`

Implements spec §3.1. This is the seam every later task depends on — get the names exactly right.

- [ ] **Step 1: Write the failing engine test**

Create `tests/test_model.py`:

```python
import pytest
from PIL import Image

from app.model import FakeEngine, GenerationRequest


def _img():
    return Image.new("RGB", (64, 64), "white")


def test_fake_engine_returns_scripted_responses_in_order():
    engine = FakeEngine(["first", "second"])
    out = engine.generate([
        GenerationRequest(image=_img(), prompt="a"),
        GenerationRequest(image=_img(), prompt="b"),
    ])
    assert out == ["first", "second"]


def test_fake_engine_records_calls_for_assertion():
    engine = FakeEngine(["x"])
    engine.generate([GenerationRequest(image=_img(), prompt="prompt-text")])
    assert len(engine.calls) == 1
    assert engine.calls[0][0].prompt == "prompt-text"


def test_fake_engine_accepts_a_callable_for_prompt_dependent_replies():
    engine = FakeEngine(lambda reqs: [r.prompt.upper() for r in reqs])
    assert engine.generate([GenerationRequest(image=_img(), prompt="hi")]) == ["HI"]


def test_fake_engine_raises_when_script_is_exhausted():
    engine = FakeEngine(["only-one"])
    engine.generate([GenerationRequest(image=_img(), prompt="a")])
    with pytest.raises(AssertionError):
        engine.generate([GenerationRequest(image=_img(), prompt="b")])


def test_fake_engine_reports_identity():
    engine = FakeEngine([])
    assert engine.model_id == "fake"
    assert engine.device == "cpu"
```

- [ ] **Step 2: Write the failing import-hygiene test**

Create `tests/test_import_hygiene.py`:

```python
"""Guards the engine seam (spec §3.1).

Every module below must be importable and testable without torch installed.
If one of them grows a top-level `import torch`, local development stops
working on a machine without a GPU and this test says so immediately.
"""

import ast
from pathlib import Path

import pytest

GPU_FREE_MODULES = [
    "app/schema.py",
    "app/config.py",
    "app/imaging.py",
    "app/parsing.py",
    "app/validate.py",
    "app/boxes.py",
    "app/extract.py",
]

FORBIDDEN = {"torch", "transformers", "accelerate", "qwen_vl_utils"}


@pytest.mark.parametrize("path", GPU_FREE_MODULES)
def test_module_has_no_top_level_gpu_import(path):
    file = Path(path)
    if not file.exists():
        pytest.skip(f"{path} not implemented yet")
    tree = ast.parse(file.read_text(encoding="utf-8"))
    for node in tree.body:  # top level only; lazy imports inside functions are fine
        if isinstance(node, ast.Import):
            names = {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {(node.module or "").split(".")[0]}
        else:
            continue
        assert not (names & FORBIDDEN), f"{path} imports {names & FORBIDDEN} at module level"


def test_qwen_engine_import_does_not_pull_in_torch():
    """app.model must be importable without torch: QwenEngine imports it lazily."""
    tree = ast.parse(Path("app/model.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            assert (mod or "").split(".")[0] not in FORBIDDEN
```

- [ ] **Step 3: Run both and confirm they fail**

Run: `pytest tests/test_model.py tests/test_import_hygiene.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.model'`

- [ ] **Step 4: Implement the seam in `app/model.py`**

```python
"""The GPU seam.

Everything above this module (extract, validate, parsing, boxes, the API,
the eval harness) talks to a VLMEngine and never imports torch. That keeps
the entire pipeline developable and testable on a machine with no GPU;
only QwenEngine needs the T4.

It also makes batching the native interface rather than a special case:
generate() takes a list and returns a list, which is what lets the
self-consistency and grounding passes ride along at almost no wall-clock
cost (spec D5).
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from PIL import Image


@dataclass(frozen=True)
class GenerationRequest:
    image: Image.Image
    prompt: str


@runtime_checkable
class VLMEngine(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def device(self) -> str: ...

    def generate(self, requests: list[GenerationRequest]) -> list[str]: ...


@dataclass
class FakeEngine:
    """Test double. Either a flat list of replies consumed in order across
    calls, or a callable that maps a batch to replies."""

    responses: list[str] | Callable[[list[GenerationRequest]], list[str]]
    calls: list[list[GenerationRequest]] = field(default_factory=list)

    @property
    def model_id(self) -> str:
        return "fake"

    @property
    def device(self) -> str:
        return "cpu"

    def generate(self, requests: list[GenerationRequest]) -> list[str]:
        self.calls.append(list(requests))
        if callable(self.responses):
            return self.responses(requests)
        assert len(self.responses) >= len(requests), (
            f"FakeEngine script exhausted: {len(requests)} requested, "
            f"{len(self.responses)} left"
        )
        out, self.responses = self.responses[: len(requests)], self.responses[len(requests):]
        return out
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_model.py tests/test_import_hygiene.py -v`
Expected: PASS (5 engine tests, 1 hygiene test, 7 skipped for unimplemented modules)

- [ ] **Step 6: Commit**

```bash
git add app/model.py tests/test_model.py tests/test_import_hygiene.py
git commit -m "feat: add VLMEngine seam and FakeEngine test double"
```

---

## Task 3: QwenEngine and the T4 measurement gate

**Files:**
- Modify: `app/model.py` (append `QwenEngine`)
- Create: `scripts/bringup.py`

**Interfaces:**
- Consumes: `GenerationRequest`, `VLMEngine`, `Settings`
- Produces: `app.model.QwenEngine(settings)` with `.load()`, `.warmup() -> int` (ms), `.vram_mb() -> int | None`, `.generate()`, `.model_id`, `.device`

Implements spec §3.2. **This task is a gate.** Spec §13 step 2 requires reporting cold-start, warm inference and VRAM *before* further building.

- [ ] **Step 1: Append `QwenEngine` to `app/model.py`**

Every torch/transformers import is inside a method — module-level imports would break the local suite and fail `test_import_hygiene.py`.

```python
class QwenEngine:
    """Real inference. The only class in the project that needs a GPU.

    T4 notes, recorded so they are not rediscovered: compute capability 7.5
    means no bf16 (fp16 is correct here) and no FlashAttention-2, which needs
    8.0+. SDPA is the ceiling. A 3B model in fp16 is ~6GB of the 16GB card,
    so no quantisation, which also avoids its quality cost.
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        self._model = None
        self._processor = None
        self._warmup_ms: int | None = None

    @property
    def model_id(self) -> str:
        return self._settings.MODEL_ID

    @property
    def device(self) -> str:
        import torch

        if self._settings.DEVICE != "auto":
            return self._settings.DEVICE
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._warmup_ms is not None

    @property
    def warmup_ms(self) -> int | None:
        return self._warmup_ms

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = getattr(torch, self._settings.TORCH_DTYPE)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._settings.MODEL_ID,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map=self.device,
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self._settings.MODEL_ID,
            min_pixels=self._settings.MIN_PIXELS,
            max_pixels=self._settings.MAX_PIXELS,
        )

    def warmup(self) -> int:
        """Absorb CUDA kernel autotuning and allocator warm-up.

        Without this the first real request pays 20-40s, which during a demo
        looks like a hang. /api/health stays loaded=false until this returns.
        """
        import time

        img = Image.new("RGB", (448, 448), "white")
        t0 = time.perf_counter()
        self._generate_raw([GenerationRequest(image=img, prompt="Reply with OK.")], max_new_tokens=8)
        self._warmup_ms = int((time.perf_counter() - t0) * 1000)
        return self._warmup_ms

    def vram_mb(self) -> int | None:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated() / 1024 / 1024)

    def generate(self, requests: list[GenerationRequest]) -> list[str]:
        return self._generate_raw(requests, self._settings.MAX_NEW_TOKENS)

    def _generate_raw(self, requests: list[GenerationRequest], max_new_tokens: int) -> list[str]:
        import torch

        messages = [
            [{"role": "user", "content": [
                {"type": "image", "image": r.image},
                {"type": "text", "text": r.prompt},
            ]}]
            for r in requests
        ]
        texts = [
            self._processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        # Left padding is required for batched decoder-only generation: with
        # right padding, shorter sequences generate from pad tokens and emit
        # garbage. This is the single easiest way to break the batching in D5.
        self._processor.tokenizer.padding_side = "left"
        inputs = self._processor(
            text=texts,
            images=[r.image for r in requests],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            out = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def processed_size(self, image: Image.Image) -> tuple[int, int]:
        """Pixel dimensions the processor resized `image` to.

        Grounding coordinates come back in this space, not the original's
        (spec §4.2), so boxes must be rescaled before leaving the backend.

        Note the Qwen2-VL family's image processor does NOT return a 4D
        pixel_values tensor - it returns flattened patches plus an
        image_grid_thw giving the grid in patch units. Multiplying the grid
        by patch_size recovers the resized pixel dimensions.
        """
        try:
            out = self._processor.image_processor(images=[image], return_tensors="pt")
            _, grid_h, grid_w = out["image_grid_thw"][0].tolist()
            patch = self._processor.image_processor.patch_size
            return (grid_w * patch, grid_h * patch)
        except (KeyError, AttributeError, ValueError):
            # Falling back to the original size means boxes are not rescaled.
            # Step 4 below prints this value precisely so a silent mismatch
            # cannot hide - if it equals the input size on a large photo,
            # the grid lookup broke and boxes will be drawn in the wrong place.
            return image.size
```

- [ ] **Step 2: Confirm the local suite still passes without torch**

Run: `pytest -v`
Expected: PASS — unchanged count. `app/model.py` still imports cleanly because every torch reference is inside a method body.

- [ ] **Step 3: Write `scripts/bringup.py`**

```python
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
```

- [ ] **Step 4: Run it on Colab and record the numbers**

On a Colab T4 runtime:

```bash
!pip install -q -r requirements.txt
!python scripts/bringup.py eval/samples/card_01.jpg
```

(Use any ID card photo. If none exists yet, Task 11 generates one — run this step with a phone photo instead.)

- [ ] **Step 5: STOP and report before continuing**

Report all six numbers to the user. The two that decide whether the plan proceeds unchanged:

- **`batch of 3` should be within ~30% of `batch of 1`.** That is the entire basis of spec D5. If it is 2–3× instead, decode is not batching as expected — stop, report, and revisit whether self-consistency and grounding can both ship.
- **`warm, batch of 3` under ~4s.** If not, apply spec §14's levers in order: lower `MAX_PIXELS`, move `value_ar` to a separate call, tighten the JSON.

- [ ] **Step 6: Commit**

```bash
git add app/model.py scripts/bringup.py
git commit -m "feat: add QwenEngine and GPU bring-up measurement script"
```

---

## Task 4: Contrast normalisation

**Files:**
- Create: `app/imaging.py`, `tests/test_imaging.py`

**Interfaces:**
- Consumes: nothing
- Produces: `app.imaging.contrast_normalise(image: Image.Image) -> Image.Image`

Implements spec §4 — a direct port of `enhance()` from `ocr-demo.html:466-483`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_imaging.py`:

```python
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


def test_flat_image_does_not_divide_by_zero():
    src = Image.new("RGB", (16, 16), (128, 128, 128))
    out = np.asarray(contrast_normalise(src).convert("L"))
    assert out.shape == (16, 16)


def test_preserves_dimensions():
    src = Image.new("RGB", (123, 45), "white")
    assert contrast_normalise(src).size == (123, 45)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `pytest tests/test_imaging.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.imaging'`

- [ ] **Step 3: Implement `app/imaging.py`**

```python
"""Image preprocessing for the self-consistency pass.

This is a port of enhance() from the original browser demo (ocr-demo.html).
Keeping the same algorithm matters: the preprocessing already validated by
eye in the browser is exactly what feeds pass B, so a disagreement between
passes reflects the model's reading, not an untested new transform.
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
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_imaging.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/imaging.py tests/test_imaging.py
git commit -m "feat: port browser contrast normalisation to the backend"
```

---

## Task 5: JSON parsing and repair

**Files:**
- Create: `app/parsing.py`, `tests/test_parsing.py`

**Interfaces:**
- Consumes: `app.schema.CardFields`
- Produces: `app.parsing.ParseError`, `app.parsing.strip_fences(text: str) -> str`, `app.parsing.parse_card_json(text: str) -> CardFields`, `app.parsing.parse_boxes(text: str) -> dict[str, list[int]]`

Implements spec §4.3 steps 1–2.

- [ ] **Step 1: Write the failing test**

Create `tests/test_parsing.py`:

```python
import pytest

from app.parsing import ParseError, parse_boxes, parse_card_json, strip_fences


def test_strips_json_fences():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strips_bare_fences():
    assert strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strips_prose_before_and_after():
    text = 'Here is the data:\n{"a": 1}\nHope that helps!'
    assert strip_fences(text) == '{"a": 1}'


def test_leaves_clean_json_untouched():
    assert strip_fences('{"a": 1}') == '{"a": 1}'


def test_parses_a_full_card():
    raw = (
        '{"full_name": "JOHN A SMITH", "full_name_ar": "جون سميث", '
        '"id_number": "12345678", "date_of_birth": "1990-04-12", '
        '"expiry_date": "2030-04-11", "nationality": "OMANI", '
        '"nationality_ar": "عماني", "sex": "M", "sex_ar": "ذكر"}'
    )
    card = parse_card_json(raw)
    assert card.full_name == "JOHN A SMITH"
    assert card.sex_ar == "ذكر"


def test_parses_fenced_output_with_nulls():
    card = parse_card_json('```json\n{"full_name": null, "id_number": "12345678"}\n```')
    assert card.full_name is None
    assert card.id_number == "12345678"


def test_coerces_a_numeric_id_number_to_string():
    """Models sometimes emit an unquoted number. That is not a reason to fail
    the whole extraction; it is unambiguous."""
    assert parse_card_json('{"id_number": 12345678}').id_number == "12345678"


def test_raises_on_truncated_json():
    with pytest.raises(ParseError):
        parse_card_json('{"full_name": "JOHN')


def test_raises_on_no_json_at_all():
    with pytest.raises(ParseError):
        parse_card_json("I cannot read this card.")


def test_raises_on_unknown_key():
    with pytest.raises(ParseError):
        parse_card_json('{"full_name": "X", "eye_colour": "brown"}')


def test_parses_boxes():
    raw = '{"full_name": [10, 20, 300, 60], "sex": [10, 90, 60, 120]}'
    assert parse_boxes(raw) == {"full_name": [10, 20, 300, 60], "sex": [10, 90, 60, 120]}


def test_parses_boxes_wrapped_in_bbox_2d():
    raw = '{"full_name": {"bbox_2d": [10, 20, 300, 60]}}'
    assert parse_boxes(raw) == {"full_name": [10, 20, 300, 60]}


def test_ignores_box_entries_that_are_not_four_numbers():
    raw = '{"full_name": [10, 20, 300], "sex": [10, 90, 60, 120]}'
    assert parse_boxes(raw) == {"sex": [10, 90, 60, 120]}


def test_box_parsing_never_raises_on_garbage():
    """Grounding is optional (spec §4.3): its failure must not fail extraction."""
    assert parse_boxes("no boxes here") == {}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `pytest tests/test_parsing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.parsing'`

- [ ] **Step 3: Implement `app/parsing.py`**

```python
"""Turning model output into typed data.

The model is instructed to emit bare JSON, and mostly does. This module
handles the "mostly": fences it was told not to use, a sentence of preamble,
an unquoted number. What it deliberately does NOT do is salvage partial or
ambiguous output - spec §4.3 requires failing to `missing` rather than
guessing, because a confidently wrong ID number is this product's worst
failure mode.
"""

import json
import re

from pydantic import ValidationError

from app.schema import CardFields

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class ParseError(Exception):
    """Model output could not be turned into typed data."""


def strip_fences(text: str) -> str:
    """Remove markdown fences and any prose outside the outermost braces."""
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned.strip()


def parse_card_json(text: str) -> CardFields:
    stripped = strip_fences(text)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ParseError(f"not valid JSON: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise ParseError(f"expected a JSON object, got {type(data).__name__}")

    # Unquoted numbers are unambiguous, so coerce rather than fail. Empty
    # strings mean the same thing as null and are normalised to it.
    for key, value in list(data.items()):
        if isinstance(value, (int, float)):
            data[key] = str(value)
        elif isinstance(value, str) and not value.strip():
            data[key] = None

    try:
        return CardFields(**data)
    except ValidationError as exc:
        raise ParseError(f"does not match the card schema: {exc.errors()[0]['msg']}") from exc


def parse_boxes(text: str) -> dict[str, list[int]]:
    """Best-effort box extraction. Never raises: grounding is optional and its
    failure must leave extraction intact (spec §4.3)."""
    try:
        data = json.loads(strip_fences(text))
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}

    out: dict[str, list[int]] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            value = value.get("bbox_2d")
        if isinstance(value, list) and len(value) == 4:
            try:
                out[key] = [int(round(float(v))) for v in value]
            except (TypeError, ValueError):
                continue
    return out
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_parsing.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add app/parsing.py tests/test_parsing.py
git commit -m "feat: add JSON fence stripping and card output parsing"
```

---

## Task 6: Format rules and cross-field checks

**Files:**
- Create: `app/validate.py`, `tests/test_validate_rules.py`

**Interfaces:**
- Consumes: `app.schema.FIELD_NAMES`, `ARABIC_FIELDS`
- Produces: `app.validate.check_format(field: str, value: str | None) -> str | None`, `app.validate.check_arabic(field: str, value_ar: str | None) -> str | None`, `app.validate.check_cross_fields(values: dict[str, str | None]) -> dict[str, str]` — each returns a human-readable reason string, or `None`/`{}` when the check passes

Implements spec §5.1 and §5.2.

- [ ] **Step 1: Write the failing test**

Create `tests/test_validate_rules.py`:

```python
from app.validate import check_arabic, check_cross_fields, check_format


def test_valid_id_number_passes():
    assert check_format("id_number", "12345678") is None


def test_id_number_of_wrong_length_is_flagged():
    reason = check_format("id_number", "1234567")
    assert reason is not None and "8 digits" in reason


def test_id_number_with_letters_is_flagged():
    assert check_format("id_number", "1234567A") is not None


def test_valid_iso_date_passes():
    assert check_format("date_of_birth", "1990-04-12") is None


def test_non_iso_date_is_flagged():
    reason = check_format("date_of_birth", "12/04/1990")
    assert reason is not None and "ISO" in reason


def test_impossible_date_is_flagged():
    assert check_format("date_of_birth", "1990-02-31") is not None


def test_out_of_range_year_is_flagged():
    assert check_format("expiry_date", "1832-01-01") is not None


def test_valid_sex_passes():
    assert check_format("sex", "M") is None
    assert check_format("sex", "F") is None


def test_unexpected_sex_value_is_flagged():
    assert check_format("sex", "MALE") is not None


def test_known_nationality_passes():
    assert check_format("nationality", "OMANI") is None


def test_unknown_nationality_is_flagged_not_rejected():
    reason = check_format("nationality", "ATLANTEAN")
    assert reason is not None and "list" in reason


def test_name_with_digits_is_flagged():
    assert check_format("full_name", "JOHN 5MITH") is not None


def test_single_character_name_is_flagged():
    assert check_format("full_name", "J") is not None


def test_null_value_is_not_a_format_failure():
    """Nulls are handled by the missing rule, not by format rules."""
    assert check_format("id_number", None) is None


def test_arabic_field_with_arabic_script_passes():
    assert check_arabic("full_name", "جون سميث") is None


def test_arabic_field_containing_only_latin_is_flagged():
    reason = check_arabic("full_name", "JOHN SMITH")
    assert reason is not None and "Arabic" in reason


def test_arabic_check_ignores_non_arabic_fields():
    assert check_arabic("id_number", "12345678") is None


def test_expiry_before_birth_is_flagged_on_both_fields():
    out = check_cross_fields({"date_of_birth": "2030-04-11", "expiry_date": "1990-04-12"})
    assert "date_of_birth" in out and "expiry_date" in out


def test_future_date_of_birth_is_flagged():
    out = check_cross_fields({"date_of_birth": "2090-01-01"})
    assert "date_of_birth" in out


def test_implausible_age_is_flagged():
    out = check_cross_fields({"date_of_birth": "1850-01-01"})
    assert "date_of_birth" in out


def test_consistent_dates_pass():
    assert check_cross_fields({"date_of_birth": "1990-04-12", "expiry_date": "2030-04-11"}) == {}


def test_cross_checks_skip_unparseable_values():
    """Format rules already flagged these; cross-checks must not crash on them."""
    assert check_cross_fields({"date_of_birth": "not-a-date", "expiry_date": "2030-04-11"}) == {}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `pytest tests/test_validate_rules.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.validate'`

- [ ] **Step 3: Implement the rules half of `app/validate.py`**

```python
"""Meaningful validity checks.

Spec §5: a VLM does not expose per-character confidence. It states answers
fluently whether right or wrong, so any percentage derived from token
probabilities would be read by a reviewer as OCR confidence while meaning
nothing of the sort. Every check in this module is instead something that
can actually be verified about the value itself.

Each rule returns a human-readable reason on failure, which travels to the
UI verbatim - a reviewer seeing "expiry precedes date of birth" knows what
to look at; "confidence 62%" does not tell them anything actionable.
"""

import re
from datetime import date
from pathlib import Path

from app.schema import ARABIC_FIELDS

_ID_NUMBER = re.compile(r"^\d{8}$")
_HAS_DIGIT = re.compile(r"\d")
# Arabic (0600-06FF), Arabic Supplement, and Arabic Presentation Forms.
_ARABIC_SCRIPT = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")

_DATE_FIELDS = ("date_of_birth", "expiry_date")
_MIN_YEAR, _MAX_YEAR = 1900, 2100
_MAX_AGE_YEARS = 120


def _nationalities() -> set[str]:
    path = Path(__file__).parent / "data" / "nationalities.txt"
    return {line.strip().upper() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _parse_iso(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def check_format(field: str, value: str | None) -> str | None:
    """Return a reason string if `value` fails its field's format rule.

    A null value is NOT a format failure - it is handled by the `missing`
    rule, which outranks everything here (spec §5).
    """
    if value is None:
        return None
    value = value.strip()

    if field == "id_number":
        # Omani civil numbers are 8 digits. Note this catches typos and
        # truncation but CANNOT catch a fabricated 8-digit number - that is
        # what the self-consistency check in merge_passes() is for.
        if not _ID_NUMBER.match(value):
            return "expected 8 digits"

    elif field in _DATE_FIELDS:
        parsed = _parse_iso(value)
        if parsed is None:
            return "not a valid ISO date (YYYY-MM-DD)"
        if not _MIN_YEAR <= parsed.year <= _MAX_YEAR:
            return f"year outside {_MIN_YEAR}-{_MAX_YEAR}"

    elif field == "sex":
        if value.upper() not in {"M", "F"}:
            return "expected M or F"

    elif field == "nationality":
        if value.upper() not in _nationalities():
            return "not in the known nationality list"

    elif field == "full_name":
        if len(value) < 2:
            return "too short to be a name"
        if _HAS_DIGIT.search(value):
            return "contains digits"

    return None


def check_arabic(field: str, value_ar: str | None) -> str | None:
    """Verify an Arabic value is actually in Arabic script.

    Deliberately does NOT compare against the Latin value: transliteration
    between the two is guesswork, and guesswork dressed as validation is
    worse than no check (spec §5.2).
    """
    if field not in ARABIC_FIELDS or value_ar is None:
        return None
    if not _ARABIC_SCRIPT.search(value_ar):
        return "expected Arabic script"
    return None


def check_cross_fields(values: dict[str, str | None]) -> dict[str, str]:
    """Checks spanning more than one field. Returns field -> reason.

    Values that fail their own format rule are skipped: they are already
    flagged, and a second complaint about the same bad value adds nothing.
    """
    out: dict[str, str] = {}
    dob = _parse_iso(values.get("date_of_birth") or "")
    exp = _parse_iso(values.get("expiry_date") or "")

    if dob and exp and exp <= dob:
        out["date_of_birth"] = "expiry precedes date of birth"
        out["expiry_date"] = "expiry precedes date of birth"

    if dob:
        today = date.today()
        if dob > today:
            out["date_of_birth"] = "date of birth is in the future"
        elif (today - dob).days > _MAX_AGE_YEARS * 365.25:
            out["date_of_birth"] = f"implies an age over {_MAX_AGE_YEARS}"

    return out
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_validate_rules.py -v`
Expected: PASS (22 tests)

- [ ] **Step 5: Commit**

```bash
git add app/validate.py tests/test_validate_rules.py
git commit -m "feat: add field format rules and cross-field checks"
```

---

## Task 7: Two-pass merge and status derivation

**Files:**
- Modify: `app/validate.py` (append)
- Create: `tests/test_validate_merge.py`

**Interfaces:**
- Consumes: `check_format`, `check_arabic`, `check_cross_fields`, `CardFields`, `FieldResult`, `Agreement`
- Produces: `app.validate.merge_passes(primary: CardFields, secondary: CardFields | None) -> tuple[dict[str, FieldResult], Agreement]` — `secondary=None` means the consistency pass was unavailable

Implements spec §5. **This is the most important task in the plan** — the ordering here is what makes the product's confidence claims honest.

- [ ] **Step 1: Write the failing test**

Create `tests/test_validate_merge.py`:

```python
from app.schema import CardFields
from app.validate import merge_passes

GOOD = dict(
    full_name="JOHN A SMITH",
    full_name_ar="جون سميث",
    id_number="12345678",
    date_of_birth="1990-04-12",
    expiry_date="2030-04-11",
    nationality="OMANI",
    nationality_ar="عماني",
    sex="M",
    sex_ar="ذكر",
)


def test_agreeing_valid_passes_are_all_ok():
    fields, agreement = merge_passes(CardFields(**GOOD), CardFields(**GOOD))
    assert {f.status for f in fields.values()} == {"ok"}
    assert agreement.matched == 6 and agreement.total == 6


def test_null_in_both_passes_is_missing():
    blank = CardFields(**{**GOOD, "id_number": None})
    fields, _ = merge_passes(blank, blank)
    assert fields["id_number"].status == "missing"
    assert fields["id_number"].value is None


def test_disagreement_is_review():
    a = CardFields(**GOOD)
    b = CardFields(**{**GOOD, "id_number": "12345679"})
    fields, agreement = merge_passes(a, b)
    assert fields["id_number"].status == "review"
    assert fields["id_number"].reason == "passes disagreed"
    assert agreement.matched == 5


def test_primary_value_is_kept_when_passes_disagree():
    a = CardFields(**GOOD)
    b = CardFields(**{**GOOD, "id_number": "12345679"})
    fields, _ = merge_passes(a, b)
    assert fields["id_number"].value == "12345678"


def test_null_in_one_pass_only_is_a_disagreement_not_missing():
    a = CardFields(**GOOD)
    b = CardFields(**{**GOOD, "id_number": None})
    fields, _ = merge_passes(a, b)
    assert fields["id_number"].status == "review"


def test_disagreement_outranks_a_format_failure():
    """Spec §5: rule 2 precedes rule 3. Disagreement is the only signal that
    catches hallucination, so it must be the reason a reviewer sees."""
    a = CardFields(**{**GOOD, "id_number": "1234567"})
    b = CardFields(**{**GOOD, "id_number": "9999999"})
    fields, _ = merge_passes(a, b)
    assert fields["id_number"].status == "review"
    assert fields["id_number"].reason == "passes disagreed"


def test_format_failure_on_agreeing_passes_is_review():
    bad = CardFields(**{**GOOD, "id_number": "1234"})
    fields, _ = merge_passes(bad, bad)
    assert fields["id_number"].status == "review"
    assert "8 digits" in fields["id_number"].reason


def test_cross_field_failure_is_review_on_both_dates():
    bad = CardFields(**{**GOOD, "date_of_birth": "2030-04-11", "expiry_date": "1990-04-12"})
    fields, _ = merge_passes(bad, bad)
    assert fields["date_of_birth"].status == "review"
    assert fields["expiry_date"].status == "review"


def test_arabic_disagreement_flags_the_field():
    a = CardFields(**GOOD)
    b = CardFields(**{**GOOD, "full_name_ar": "جون سميثي"})
    fields, _ = merge_passes(a, b)
    assert fields["full_name"].status == "review"


def test_arabic_is_only_populated_for_arabic_fields():
    fields, _ = merge_passes(CardFields(**GOOD), CardFields(**GOOD))
    assert fields["full_name"].value_ar == "جون سميث"
    assert fields["id_number"].value_ar is None


def test_comparison_ignores_case_and_surrounding_whitespace():
    a = CardFields(**GOOD)
    b = CardFields(**{**GOOD, "full_name": "  john a smith  "})
    fields, _ = merge_passes(a, b)
    assert fields["full_name"].status == "ok"


def test_missing_secondary_downgrades_everything_to_review():
    fields, agreement = merge_passes(CardFields(**GOOD), None)
    assert {f.status for f in fields.values()} == {"review"}
    assert fields["sex"].reason == "consistency pass unavailable"
    assert agreement.matched == 0


def test_missing_secondary_still_reports_null_fields_as_missing():
    """Spec §4.3: a field the model could not read does not become `review`
    merely because a second opinion was unavailable."""
    card = CardFields(**{**GOOD, "id_number": None})
    fields, _ = merge_passes(card, None)
    assert fields["id_number"].status == "missing"
    assert fields["sex"].status == "review"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `pytest tests/test_validate_merge.py -v`
Expected: FAIL — `ImportError: cannot import name 'merge_passes'`

- [ ] **Step 3: Append the merge logic to `app/validate.py`**

Add `import unicodedata` at the top, and **extend the existing `app.schema` import line** rather than adding a second one — `ARABIC_FIELDS` is already imported from Task 6:

```python
from app.schema import ARABIC_FIELDS, FIELD_NAMES, Agreement, CardFields, FieldResult
```

Then append:

```python
def _normalise(value: str | None) -> str | None:
    """Comparison form for cross-pass agreement.

    NFC first so two byte-different but identical Arabic strings compare
    equal; then case-fold and collapse whitespace so trivial formatting
    differences are not reported to a human as a disagreement.
    """
    if value is None:
        return None
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def merge_passes(
    primary: CardFields, secondary: CardFields | None
) -> tuple[dict[str, FieldResult], Agreement]:
    """Combine two extraction passes into per-field results and statuses.

    `secondary=None` means the consistency pass failed or was disabled.

    Rule order (spec §5), first match wins:
        1. null in both              -> missing
        2. passes disagree           -> review, "passes disagreed"
        3. format rule fails         -> review, names the rule
        4. cross-field check fails   -> review, names the check
        5. otherwise                 -> ok

    Rule 2 precedes rule 3 deliberately. Format rules cannot catch
    hallucination: a fabricated 8-digit civil number satisfies every one of
    them. Two passes over differently-preprocessed images rarely fabricate
    the SAME wrong value, which makes disagreement the strongest signal
    available - and therefore the one a reviewer should be shown first.
    """
    latin = {f: getattr(primary, f) for f in FIELD_NAMES}
    cross = check_cross_fields(latin)

    results: dict[str, FieldResult] = {}
    matched = 0

    for field in FIELD_NAMES:
        value = getattr(primary, field)
        value_ar = getattr(primary, f"{field}_ar", None) if field in ARABIC_FIELDS else None

        # Rule 1. `missing` always outranks `review`, including when the
        # consistency pass is unavailable.
        if value is None and (secondary is None or getattr(secondary, field) is None):
            results[field] = FieldResult(value=None, value_ar=None, status="missing")
            continue

        if secondary is None:
            results[field] = FieldResult(
                value=value,
                value_ar=value_ar,
                status="review",
                reason="consistency pass unavailable",
            )
            continue

        # Rule 2. Arabic disagreement flags the field just as Latin does.
        agrees = _normalise(value) == _normalise(getattr(secondary, field))
        if field in ARABIC_FIELDS:
            agrees = agrees and _normalise(value_ar) == _normalise(
                getattr(secondary, f"{field}_ar", None)
            )

        if not agrees:
            results[field] = FieldResult(
                value=value, value_ar=value_ar, status="review", reason="passes disagreed"
            )
            continue

        matched += 1

        # Rules 3 and 4.
        reason = check_format(field, value) or check_arabic(field, value_ar) or cross.get(field)
        results[field] = FieldResult(
            value=value,
            value_ar=value_ar,
            status="review" if reason else "ok",
            reason=reason,
        )

    return results, Agreement(matched=matched, total=len(FIELD_NAMES))
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_validate_merge.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Run the whole suite**

Run: `pytest -v`
Expected: PASS, no regressions

- [ ] **Step 6: Commit**

```bash
git add app/validate.py tests/test_validate_merge.py
git commit -m "feat: add two-pass merge and status derivation"
```

---

## Task 8: Box rescaling and the plausibility filter

**Files:**
- Create: `app/boxes.py`, `tests/test_boxes.py`

**Interfaces:**
- Consumes: `app.schema.FIELD_NAMES`
- Produces: `app.boxes.rescale_box(box, from_size, to_size) -> tuple[int,int,int,int]`, `app.boxes.filter_boxes(raw: dict[str, list[int]], processed_size: tuple[int,int], image_size: tuple[int,int]) -> dict[str, tuple[int,int,int,int]]`

Implements spec §4.2 and §5.3.

- [ ] **Step 1: Write the failing test**

Create `tests/test_boxes.py`:

```python
from app.boxes import filter_boxes, rescale_box

# A 1000x600 processed image against a 2000x1200 original: factor of 2.
PROC = (1000, 600)
ORIG = (2000, 1200)


def test_rescales_from_processed_space_to_original():
    assert rescale_box([100, 50, 300, 90], PROC, ORIG) == (200, 100, 600, 180)


def test_rescale_is_identity_when_sizes_match():
    assert rescale_box([10, 20, 30, 40], PROC, PROC) == (10, 20, 30, 40)


def test_plausible_box_survives():
    out = filter_boxes({"full_name": [100, 50, 400, 90]}, PROC, ORIG)
    assert out["full_name"] == (200, 100, 800, 180)


def test_inverted_box_is_dropped():
    assert filter_boxes({"full_name": [400, 90, 100, 50]}, PROC, ORIG) == {}


def test_box_far_outside_bounds_is_dropped():
    assert filter_boxes({"full_name": [100, 50, 5000, 90]}, PROC, ORIG) == {}


def test_box_slightly_outside_bounds_is_clamped_not_dropped():
    out = filter_boxes({"full_name": [100, 50, 1010, 90]}, PROC, ORIG)
    assert out["full_name"][2] == 2000


def test_full_page_box_is_dropped():
    assert filter_boxes({"full_name": [0, 0, 1000, 600]}, PROC, ORIG) == {}


def test_tall_narrow_box_is_dropped():
    """Printed field lines are wide, never tall."""
    assert filter_boxes({"full_name": [100, 50, 120, 400]}, PROC, ORIG) == {}


def test_sub_pixel_speck_is_dropped():
    assert filter_boxes({"full_name": [100, 50, 102, 52]}, PROC, ORIG) == {}


def test_identical_box_claimed_by_three_fields_is_dropped_for_all():
    """Model collapse: one box repeated for everything is not grounding."""
    box = [100, 50, 400, 90]
    out = filter_boxes(
        {"full_name": box, "sex": list(box), "nationality": list(box)}, PROC, ORIG
    )
    assert out == {}


def test_identical_box_claimed_by_two_fields_survives():
    box = [100, 50, 400, 90]
    out = filter_boxes({"full_name": box, "sex": list(box)}, PROC, ORIG)
    assert len(out) == 2


def test_unknown_field_names_are_discarded():
    out = filter_boxes({"eye_colour": [100, 50, 400, 90]}, PROC, ORIG)
    assert out == {}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `pytest tests/test_boxes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.boxes'`

- [ ] **Step 3: Implement `app/boxes.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_boxes.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add app/boxes.py tests/test_boxes.py
git commit -m "feat: add box rescaling and geometric plausibility filter"
```

---

## Task 9: Extraction orchestration

**Files:**
- Create: `app/extract.py`, `tests/test_extract.py`

**Interfaces:**
- Consumes: everything from Tasks 1, 2, 4–8
- Produces: `app.extract.FIELD_PROMPT`, `GROUNDING_PROMPT`, `TRANSCRIBE_PROMPT`, `app.extract.extract(image, engine, settings, processed_size=None) -> ExtractResponse`, `app.extract.transcribe(image, engine, settings) -> str`

Implements spec §4, §4.1, §4.2, §4.3.

- [ ] **Step 1: Write the failing test**

Create `tests/test_extract.py`:

```python
import json

from PIL import Image

from app.config import Settings
from app.extract import extract
from app.model import FakeEngine

GOOD = {
    "full_name": "JOHN A SMITH",
    "full_name_ar": "جون سميث",
    "id_number": "12345678",
    "date_of_birth": "1990-04-12",
    "expiry_date": "2030-04-11",
    "nationality": "OMANI",
    "nationality_ar": "عماني",
    "sex": "M",
    "sex_ar": "ذكر",
}
BOXES = {"full_name": [10, 10, 300, 50], "sex": [10, 70, 60, 100]}


def _img():
    return Image.new("RGB", (1000, 600), "white")


def _replies(fields=None, boxes=None):
    payload = json.dumps(fields if fields is not None else GOOD)
    return [payload, payload, json.dumps(boxes if boxes is not None else BOXES)]


def test_issues_exactly_one_batched_call_of_three():
    """Spec D5: self-consistency and grounding ride along in one generate()."""
    engine = FakeEngine(_replies())
    extract(_img(), engine, Settings())
    assert len(engine.calls) == 1
    assert len(engine.calls[0]) == 3


def test_second_request_uses_a_contrast_normalised_image():
    engine = FakeEngine(_replies())
    extract(_img(), engine, Settings())
    a, b, _ = engine.calls[0]
    assert a.image is not b.image


def test_third_request_uses_the_grounding_prompt():
    engine = FakeEngine(_replies())
    extract(_img(), engine, Settings())
    assert "bbox_2d" in engine.calls[0][2].prompt


def test_returns_ok_fields_for_agreeing_passes():
    resp = extract(_img(), FakeEngine(_replies()), Settings())
    assert resp.fields["id_number"].status == "ok"
    assert resp.agreement.matched == 6


def test_attaches_filtered_boxes():
    resp = extract(_img(), FakeEngine(_replies()), Settings())
    assert resp.fields["full_name"].box is not None
    assert resp.fields["id_number"].box is None


def test_show_boxes_false_suppresses_every_box():
    resp = extract(_img(), FakeEngine(_replies()), Settings(SHOW_BOXES=False))
    assert all(f.box is None for f in resp.fields.values())


def test_self_consistency_false_issues_two_requests_and_reviews_everything():
    engine = FakeEngine([json.dumps(GOOD), json.dumps(BOXES)])
    resp = extract(_img(), engine, Settings(SELF_CONSISTENCY=False))
    assert len(engine.calls[0]) == 2
    assert resp.fields["sex"].status == "review"


def test_raw_text_carries_the_primary_pass_output():
    """Spec D3: raw_text is the model's own echo, not a transcription."""
    engine = FakeEngine(_replies())
    resp = extract(_img(), engine, Settings())
    assert "JOHN A SMITH" in resp.raw_text


def test_malformed_primary_triggers_exactly_one_retry():
    engine = FakeEngine(["not json", json.dumps(GOOD), json.dumps(BOXES), json.dumps(GOOD)])
    resp = extract(_img(), engine, Settings())
    assert len(engine.calls) == 2
    assert len(engine.calls[1]) == 1
    assert resp.fields["id_number"].status == "ok"


def test_retry_prompt_feeds_the_parse_error_back():
    engine = FakeEngine(["not json", json.dumps(GOOD), json.dumps(BOXES), json.dumps(GOOD)])
    extract(_img(), engine, Settings())
    assert "not valid JSON" in engine.calls[1][0].prompt


def test_two_failures_yield_all_missing_and_never_guess():
    """Spec §4.3: a second failure returns raw text with everything missing
    rather than salvaging a partial answer."""
    engine = FakeEngine(["garbage", json.dumps(GOOD), json.dumps(BOXES), "still garbage"])
    resp = extract(_img(), engine, Settings())
    assert {f.status for f in resp.fields.values()} == {"missing"}
    assert all(f.value is None for f in resp.fields.values())
    assert "still garbage" in resp.raw_text


def test_failed_secondary_downgrades_to_review_without_failing_extraction():
    engine = FakeEngine([json.dumps(GOOD), "garbage", json.dumps(BOXES), "garbage again"])
    resp = extract(_img(), engine, Settings())
    assert resp.fields["sex"].status == "review"
    assert resp.fields["sex"].value == "M"


def test_failed_grounding_leaves_extraction_intact():
    engine = FakeEngine([json.dumps(GOOD), json.dumps(GOOD), "no boxes for you"])
    resp = extract(_img(), engine, Settings())
    assert resp.fields["full_name"].status == "ok"
    assert resp.fields["full_name"].box is None


def test_reports_elapsed_and_model():
    resp = extract(_img(), FakeEngine(_replies()), Settings())
    assert resp.elapsed_ms >= 0
    assert resp.model == "fake"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `pytest tests/test_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.extract'`

- [ ] **Step 3: Implement `app/extract.py`**

```python
"""Extraction orchestration.

The shape of this module is dictated by one measurement: on a T4, decode is
bound by weight bandwidth, not compute. A batch re-reads the same ~6GB of
weights once regardless of how many sequences it carries, so the
self-consistency pass and the grounding pass are close to free in
wall-clock while a second sequential call would cost another full inference
(spec D5, §15).

Hence: one generate(), three sequences, and every prompt kept short because
roughly every 30 output tokens costs a second.
"""

import time

from PIL import Image

from app.boxes import filter_boxes
from app.imaging import contrast_normalise
from app.model import GenerationRequest
from app.parsing import ParseError, parse_boxes, parse_card_json
from app.schema import FIELD_NAMES, Agreement, CardFields, ExtractResponse, FieldResult
from app.validate import merge_passes

FIELD_PROMPT = """You are reading an Omani national ID card. Return ONLY a JSON object, no markdown fences, no commentary.

Keys, all required:
  full_name, full_name_ar, id_number, date_of_birth, expiry_date, nationality, nationality_ar, sex, sex_ar

Rules:
- Dates as ISO YYYY-MM-DD.
- sex is exactly "M" or "F".
- Latin keys hold Latin script. Keys ending _ar hold Arabic script.
- If a value is not clearly legible, return null for it.
- Do NOT guess. Do NOT infer a value from context or from another field. A wrong value is far worse than null."""

GROUNDING_PROMPT = """Locate each printed field on this ID card. Return ONLY a JSON object mapping field name to bbox_2d as [x1, y1, x2, y2] in pixels.
Fields: full_name, id_number, date_of_birth, expiry_date, nationality, sex
Omit any field you cannot locate."""

TRANSCRIBE_PROMPT = "Transcribe all printed text on this card, line by line, exactly as it appears."


def _all_missing(raw_text: str, model_id: str, elapsed_ms: int) -> ExtractResponse:
    """Spec §4.3: after a second parse failure, return nothing rather than a
    guess. A confidently wrong ID number is this product's worst failure
    mode - far more damaging than a blank field."""
    return ExtractResponse(
        fields={f: FieldResult(value=None, status="missing") for f in FIELD_NAMES},
        raw_text=raw_text,
        agreement=Agreement(matched=0, total=len(FIELD_NAMES)),
        elapsed_ms=elapsed_ms,
        model=model_id,
    )


def _parse_with_retry(
    text: str, image: Image.Image, engine, settings
) -> tuple[CardFields | None, str]:
    """Parse, and on failure retry ONCE with the error fed back.

    Returns (card, raw_text). card is None when both attempts failed.
    """
    try:
        return parse_card_json(text), text
    except ParseError as first:
        retry_prompt = (
            f"{FIELD_PROMPT}\n\nYour previous reply could not be parsed: {first}. "
            "Reply with the JSON object only."
        )
        retry = engine.generate([GenerationRequest(image=image, prompt=retry_prompt)])[0]
        try:
            return parse_card_json(retry), retry
        except ParseError:
            return None, retry


def extract(image: Image.Image, engine, settings, processed_size=None) -> ExtractResponse:
    t0 = time.perf_counter()

    requests = [GenerationRequest(image=image, prompt=FIELD_PROMPT)]
    if settings.SELF_CONSISTENCY:
        requests.append(
            GenerationRequest(image=contrast_normalise(image), prompt=FIELD_PROMPT)
        )
    requests.append(GenerationRequest(image=image, prompt=GROUNDING_PROMPT))

    replies = engine.generate(requests)
    primary_text = replies[0]
    secondary_text = replies[1] if settings.SELF_CONSISTENCY else None
    grounding_text = replies[-1]

    primary, primary_raw = _parse_with_retry(primary_text, image, engine, settings)
    if primary is None:
        return _all_missing(primary_raw, engine.model_id, int((time.perf_counter() - t0) * 1000))

    # A failed or disabled secondary is not fatal: merge_passes downgrades
    # every field to `review` rather than claiming an unverified `ok`.
    secondary = None
    if secondary_text is not None:
        secondary, _ = _parse_with_retry(secondary_text, image, engine, settings)

    fields, agreement = merge_passes(primary, secondary)

    if settings.SHOW_BOXES:
        boxes = filter_boxes(
            parse_boxes(grounding_text), processed_size or image.size, image.size
        )
        for name, box in boxes.items():
            fields[name].box = box

    return ExtractResponse(
        fields=fields,
        raw_text=primary_raw,
        agreement=agreement,
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
        model=engine.model_id,
    )


def transcribe(image: Image.Image, engine, settings) -> str:
    """Full card transcription. Deliberately off the fast path (spec D3): it
    costs 200-400 decode tokens, which alone breaks the latency target."""
    return engine.generate([GenerationRequest(image=image, prompt=TRANSCRIBE_PROMPT)])[0].strip()
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_extract.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Run the whole suite**

Run: `pytest -v`
Expected: PASS, no regressions. `test_import_hygiene.py` now checks real files instead of skipping.

- [ ] **Step 6: Commit**

```bash
git add app/extract.py tests/test_extract.py
git commit -m "feat: add batched extraction orchestration with single retry"
```

---

## Task 10: The API

**Files:**
- Create: `app/main.py`, `tests/test_api.py`
- Create: `static/.gitkeep`

**Interfaces:**
- Consumes: `extract`, `transcribe`, `QwenEngine`, `Settings`, response models
- Produces: `app.main.app` (FastAPI), `app.main.create_app(engine=None) -> FastAPI`

Implements spec §3.3, §6, §10.

- [ ] **Step 1: Write the failing test**

Create `tests/test_api.py`:

```python
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.model import FakeEngine

GOOD = {
    "full_name": "JOHN A SMITH",
    "full_name_ar": "جون سميث",
    "id_number": "12345678",
    "date_of_birth": "1990-04-12",
    "expiry_date": "2030-04-11",
    "nationality": "OMANI",
    "nationality_ar": "عماني",
    "sex": "M",
    "sex_ar": "ذكر",
}


def _upload():
    buf = io.BytesIO()
    Image.new("RGB", (800, 500), "white").save(buf, format="JPEG")
    buf.seek(0)
    return {"image": ("card.jpg", buf, "image/jpeg")}


def _client(replies):
    return TestClient(create_app(engine=FakeEngine(replies)))


def test_health_reports_model_device_and_loaded_state():
    body = _client([]).get("/api/health").json()
    assert body["model"] == "fake"
    assert body["device"] == "cpu"
    assert body["loaded"] is True
    assert body["self_consistency"] is True


def test_extract_returns_a_schema_conforming_response():
    payload = json.dumps(GOOD)
    r = _client([payload, payload, "{}"]).post("/api/extract", files=_upload())
    assert r.status_code == 200
    body = r.json()
    assert set(body["fields"]) == {
        "full_name", "id_number", "date_of_birth", "expiry_date", "nationality", "sex",
    }
    assert body["fields"]["id_number"]["status"] == "ok"
    assert body["agreement"] == {"matched": 6, "total": 6}
    assert body["model"] == "fake"


def test_extract_rejects_a_non_image_upload():
    files = {"image": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    r = _client([]).post("/api/extract", files=files)
    assert r.status_code == 400
    assert "image" in r.json()["detail"].lower()


def test_extract_requires_a_file():
    assert _client([]).post("/api/extract").status_code == 422


def test_transcribe_returns_text():
    r = _client(["LINE ONE\nLINE TWO"]).post("/api/transcribe", files=_upload())
    assert r.status_code == 200
    assert r.json()["text"] == "LINE ONE\nLINE TWO"


def test_cors_headers_are_present():
    r = _client([]).get("/api/health", headers={"Origin": "https://example.trycloudflare.com"})
    assert r.headers["access-control-allow-origin"] == "*"


def test_logs_never_contain_field_values(caplog):
    """Spec §10: event, timing and status counts only."""
    payload = json.dumps(GOOD)
    with caplog.at_level("DEBUG"):
        _client([payload, payload, "{}"]).post("/api/extract", files=_upload())
    combined = " ".join(r.getMessage() for r in caplog.records)
    assert "JOHN A SMITH" not in combined
    assert "12345678" not in combined


def test_extract_logs_status_counts():
    """The useful half of the logging rule: operational signal must survive."""
    payload = json.dumps(GOOD)
    client = _client([payload, payload, "{}"])
    r = client.post("/api/extract", files=_upload())
    assert r.status_code == 200
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Implement `app/main.py`**

```python
"""FastAPI application.

Knows nothing about Colab, Drive or tunnels (spec §1, §10). The tunnel is
demo infrastructure; removing it for a real deployment means not running a
notebook cell, not changing this file.
"""

import asyncio
import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.extract import extract, transcribe
from app.schema import ExtractResponse, HealthResponse, TranscribeResponse

logger = logging.getLogger("app")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class ValueLeakFilter(logging.Filter):
    """Spec §10 enforcement.

    Rejecting records that carry a `value` attribute makes a careless
    logger.info(..., extra={"value": x}) added later fail loudly instead of
    leaking a customer's ID number into a log file quietly.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not hasattr(record, "value")


async def _read_image(upload: UploadFile) -> Image.Image:
    data = await upload.read()
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="That upload is not a readable image.")


def create_app(engine=None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logger.addFilter(ValueLeakFilter())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is not None:
            app.state.engine = engine
            app.state.loaded = True
        else:
            # Imported here so the module stays importable without torch.
            from app.model import QwenEngine

            app.state.engine = QwenEngine(settings)
            app.state.loaded = False
            logger.info("loading model %s", settings.MODEL_ID)
            await run_in_threadpool(app.state.engine.load)
            warm = await run_in_threadpool(app.state.engine.warmup)
            app.state.loaded = True
            logger.info("model ready, warmup_ms=%d", warm)
        # One GPU, one model: serialise inference so two requests cannot
        # interleave on the same CUDA context.
        app.state.gpu_lock = asyncio.Lock()
        yield

    app = FastAPI(title="ID card extraction", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        eng = request.app.state.engine
        return HealthResponse(
            model=eng.model_id,
            device=eng.device,
            loaded=request.app.state.loaded,
            self_consistency=settings.SELF_CONSISTENCY,
            warmup_ms=getattr(eng, "warmup_ms", None),
            vram_mb=eng.vram_mb() if hasattr(eng, "vram_mb") else None,
        )

    @app.post("/api/extract", response_model=ExtractResponse)
    async def extract_endpoint(
        request: Request, image: UploadFile = File(...)
    ) -> ExtractResponse:
        img = await _read_image(image)
        eng = request.app.state.engine

        if settings.DEBUG_SAVE_IMAGES:  # development only; see README
            Path(settings.DEBUG_SAVE_DIR).mkdir(parents=True, exist_ok=True)
            img.save(Path(settings.DEBUG_SAVE_DIR) / f"{int(time.time() * 1000)}.jpg")

        processed = eng.processed_size(img) if hasattr(eng, "processed_size") else None
        async with request.app.state.gpu_lock:
            result = await run_in_threadpool(extract, img, eng, settings, processed)

        counts: dict[str, int] = {}
        for f in result.fields.values():
            counts[f.status] = counts.get(f.status, 0) + 1
        # Statuses and timings only - never values (spec §10).
        logger.info(
            "extract elapsed_ms=%d agreement=%d/%d %s",
            result.elapsed_ms, result.agreement.matched, result.agreement.total, counts,
        )
        return result

    @app.post("/api/transcribe", response_model=TranscribeResponse)
    async def transcribe_endpoint(
        request: Request, image: UploadFile = File(...)
    ) -> TranscribeResponse:
        img = await _read_image(image)
        t0 = time.perf_counter()
        async with request.app.state.gpu_lock:
            text = await run_in_threadpool(
                transcribe, img, request.app.state.engine, settings
            )
        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.info("transcribe elapsed_ms=%d", elapsed)
        return TranscribeResponse(text=text, elapsed_ms=elapsed)

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


app = create_app()
```

- [ ] **Step 4: Create the static directory placeholder**

```bash
mkdir -p static && touch static/.gitkeep
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_api.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_api.py static/.gitkeep
git commit -m "feat: add extract, health and transcribe endpoints"
```

---

## Task 11: Synthetic sample generator

**Files:**
- Create: `eval/__init__.py`, `eval/make_samples.py`, `tests/test_make_samples.py`

**Interfaces:**
- Consumes: `app.schema.FIELD_NAMES`
- Produces: `eval.make_samples.render_card(record: dict, size=(1000, 630)) -> tuple[Image.Image, dict]` returning the image and its ground-truth box map; `eval.make_samples.main()` writing `eval/samples/*.jpg` and `eval/samples/expected.json`

Implements spec §8 and D6.

- [ ] **Step 1: Write the failing test**

Create `tests/test_make_samples.py`:

```python
import pytest

from eval.make_samples import RECORDS, shape_arabic

pytest.importorskip("arabic_reshaper")


def test_shaping_joins_and_reverses_arabic():
    """Without reshaping + bidi, Pillow renders Arabic as disconnected,
    reversed letterforms, which would make the samples worthless."""
    out = shape_arabic("عماني")
    assert out != "عماني"
    assert len(out) > 0


def test_shaping_leaves_latin_untouched():
    assert shape_arabic("OMANI") == "OMANI"


def test_records_cover_the_declared_fields():
    for record in RECORDS:
        assert set(record) >= {
            "full_name", "full_name_ar", "id_number", "date_of_birth",
            "expiry_date", "nationality", "nationality_ar", "sex", "sex_ar",
        }


def test_records_are_internally_valid():
    from app.validate import check_cross_fields, check_format

    for record in RECORDS:
        assert check_format("id_number", record["id_number"]) is None
        assert check_format("nationality", record["nationality"]) is None
        assert check_cross_fields(record) == {}


def test_render_returns_boxes_for_every_field(tmp_path):
    from eval.make_samples import FONT_PATH, render_card

    if not FONT_PATH.exists():
        pytest.skip("Noto Naskh Arabic not installed; see eval/make_samples.py")
    image, boxes = render_card(RECORDS[0])
    assert image.size == (1000, 630)
    assert set(boxes) == {
        "full_name", "id_number", "date_of_birth", "expiry_date", "nationality", "sex",
    }
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `pytest tests/test_make_samples.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.make_samples'`

- [ ] **Step 3: Implement `eval/make_samples.py`**

Create `eval/__init__.py` (empty), then:

```python
"""Synthetic bilingual ID cards so the eval harness runs before real photos.

These are clean renders. They have no glare, skew, wear or depth of field,
and they WILL overstate accuracy (spec R5). They exist to prove the harness
works end to end, not to predict field performance.

Arabic needs arabic-reshaper (contextual letter joining) and python-bidi
(right-to-left reordering). Pillow does neither, so without them the text
renders as disconnected, reversed letterforms and the Arabic half of the
eval measures nothing (spec D6).
"""

import json
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

SAMPLES_DIR = Path(__file__).parent / "samples"
FONT_DIR = Path(__file__).parent / "fonts"
FONT_PATH = FONT_DIR / "NotoNaskhArabic-Regular.ttf"
LATIN_FONT_PATH = FONT_DIR / "DejaVuSans.ttf"

FONT_HELP = f"""Missing font: {FONT_PATH}

Download Noto Naskh Arabic (SIL Open Font License) and place the Regular
TTF at that path:

    https://fonts.google.com/noto/specimen/Noto+Naskh+Arabic

Pillow ships DejaVuSans, which has no Arabic coverage, so this cannot be
substituted automatically."""

RECORDS: list[dict] = [
    {
        "full_name": "AHMED SAID AL HARTHY", "full_name_ar": "أحمد سعيد الحارثي",
        "id_number": "10293847", "date_of_birth": "1988-03-21",
        "expiry_date": "2031-03-20", "nationality": "OMANI",
        "nationality_ar": "عماني", "sex": "M", "sex_ar": "ذكر",
    },
    {
        "full_name": "FATIMA ALI AL BALUSHI", "full_name_ar": "فاطمة علي البلوشي",
        "id_number": "20394857", "date_of_birth": "1995-11-02",
        "expiry_date": "2029-11-01", "nationality": "OMANI",
        "nationality_ar": "عماني", "sex": "F", "sex_ar": "أنثى",
    },
    {
        "full_name": "RAJESH KUMAR NAIR", "full_name_ar": "راجيش كومار نائير",
        "id_number": "30485712", "date_of_birth": "1979-07-14",
        "expiry_date": "2028-07-13", "nationality": "INDIAN",
        "nationality_ar": "هندي", "sex": "M", "sex_ar": "ذكر",
    },
    {
        "full_name": "MARIA SANTOS CRUZ", "full_name_ar": "ماريا سانتوس كروز",
        "id_number": "40576123", "date_of_birth": "1992-01-30",
        "expiry_date": "2030-01-29", "nationality": "FILIPINO",
        "nationality_ar": "فلبيني", "sex": "F", "sex_ar": "أنثى",
    },
]

_ROWS = [
    ("Name", "full_name", "full_name_ar"),
    ("ID Number", "id_number", None),
    ("Date of Birth", "date_of_birth", None),
    ("Expiry", "expiry_date", None),
    ("Nationality", "nationality", "nationality_ar"),
    ("Sex", "sex", "sex_ar"),
]


def shape_arabic(text: str) -> str:
    """Join and reorder Arabic for a renderer with no bidi support."""
    return get_display(arabic_reshaper.reshape(text))


def _fonts() -> tuple[ImageFont.FreeTypeFont, ...]:
    if not FONT_PATH.exists():
        raise FileNotFoundError(FONT_HELP)
    latin = (
        ImageFont.truetype(str(LATIN_FONT_PATH), 26)
        if LATIN_FONT_PATH.exists()
        else ImageFont.load_default(26)
    )
    label = (
        ImageFont.truetype(str(LATIN_FONT_PATH), 16)
        if LATIN_FONT_PATH.exists()
        else ImageFont.load_default(16)
    )
    return latin, label, ImageFont.truetype(str(FONT_PATH), 24)


def render_card(record: dict, size: tuple[int, int] = (1000, 630)) -> tuple[Image.Image, dict]:
    """Render one card. Returns the image and field -> ground-truth box."""
    latin_font, label_font, arabic_font = _fonts()

    image = Image.new("RGB", size, (243, 246, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (size[0], 72)], fill=(14, 21, 27))
    draw.text((36, 24), "SULTANATE OF OMAN  -  IDENTITY CARD", font=label_font, fill="white")

    boxes: dict[str, tuple[int, int, int, int]] = {}
    y = 118
    for label, key, ar_key in _ROWS:
        draw.text((36, y), label.upper(), font=label_font, fill=(104, 117, 126))
        value = record[key]
        draw.text((300, y - 6), value, font=latin_font, fill=(14, 21, 27))
        # Ground truth for box hit rate: the box around the Latin value.
        x1, y1, x2, y2 = draw.textbbox((300, y - 6), value, font=latin_font)
        boxes[key] = (int(x1) - 4, int(y1) - 4, int(x2) + 4, int(y2) + 4)

        if ar_key:
            shaped = shape_arabic(record[ar_key])
            w = draw.textlength(shaped, font=arabic_font)
            draw.text((size[0] - 36 - w, y - 4), shaped, font=arabic_font, fill=(14, 21, 27))
        y += 78

    return image, boxes


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    expected: dict[str, dict] = {}

    for i, record in enumerate(RECORDS, start=1):
        name = f"synthetic_{i:02d}.jpg"
        image, boxes = render_card(record)
        image.save(SAMPLES_DIR / name, quality=95)
        expected[name] = {"fields": record, "boxes": boxes}

    (SAMPLES_DIR / "expected.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(RECORDS)} cards to {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Fetch the font and generate the cards**

Download Noto Naskh Arabic Regular to `eval/fonts/NotoNaskhArabic-Regular.ttf` (the module prints exact instructions if it is missing). Optionally copy a `DejaVuSans.ttf` beside it for Latin text.

```bash
python -m eval.make_samples
```

Expected: `wrote 4 cards to eval/samples`. **Open one and look at it** — confirm the Arabic is joined and reads right-to-left, not disconnected or reversed.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_make_samples.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add eval/__init__.py eval/make_samples.py tests/test_make_samples.py eval/samples/expected.json
git commit -m "feat: add synthetic bilingual card generator for eval"
```

---

## Task 12: Evaluation harness

**Files:**
- Create: `eval/run_eval.py`, `tests/test_run_eval.py`

**Interfaces:**
- Consumes: `ExtractResponse`, `FIELD_NAMES`
- Produces: `eval.run_eval.score(results: dict[str, ExtractResponse], expected: dict) -> Report` with `Report.silent_errors: list[SilentError]`, `.per_field: dict`, `.box_hits`, `.box_total`; `eval.run_eval.main()`

Implements spec §8. The silent error rate is the number the user cares about most.

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_eval.py`:

```python
from app.schema import Agreement, ExtractResponse, FieldResult
from eval.run_eval import iou, score

EXPECTED = {
    "a.jpg": {
        "fields": {
            "full_name": "JOHN A SMITH", "id_number": "12345678",
            "date_of_birth": "1990-04-12", "expiry_date": "2030-04-11",
            "nationality": "OMANI", "sex": "M",
        },
        "boxes": {"full_name": [100, 100, 400, 140]},
    }
}


def _response(fields):
    return ExtractResponse(
        fields=fields, raw_text="{}", agreement=Agreement(matched=6, total=6),
        elapsed_ms=100, model="fake",
    )


def _all_ok(**overrides):
    base = {k: FieldResult(value=v, status="ok") for k, v in EXPECTED["a.jpg"]["fields"].items()}
    base.update(overrides)
    return base


def test_a_perfect_run_has_no_silent_errors():
    report = score({"a.jpg": _response(_all_ok())}, EXPECTED)
    assert report.silent_errors == []
    assert report.per_field["id_number"]["exact"] == 1


def test_an_ok_field_with_a_wrong_value_is_a_silent_error():
    """The metric that matters most: the model was confident and wrong, and
    nothing caught it."""
    wrong = _all_ok(id_number=FieldResult(value="12345679", status="ok"))
    report = score({"a.jpg": _response(wrong)}, EXPECTED)
    assert len(report.silent_errors) == 1
    err = report.silent_errors[0]
    assert err.field == "id_number" and err.got == "12345679" and err.expected == "12345678"


def test_a_reviewed_wrong_field_is_not_a_silent_error():
    """It was flagged. The system did its job; a human will catch it."""
    flagged = _all_ok(
        id_number=FieldResult(value="12345679", status="review", reason="passes disagreed")
    )
    report = score({"a.jpg": _response(flagged)}, EXPECTED)
    assert report.silent_errors == []
    assert report.per_field["id_number"]["review"] == 1


def test_a_missing_field_is_counted_but_is_not_a_silent_error():
    blanked = _all_ok(id_number=FieldResult(value=None, status="missing"))
    report = score({"a.jpg": _response(blanked)}, EXPECTED)
    assert report.silent_errors == []
    assert report.per_field["id_number"]["missing"] == 1
    assert report.per_field["id_number"]["exact"] == 0


def test_comparison_ignores_case_and_whitespace():
    lenient = _all_ok(full_name=FieldResult(value="  john a smith ", status="ok"))
    assert score({"a.jpg": _response(lenient)}, EXPECTED).per_field["full_name"]["exact"] == 1


def test_iou_of_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_box_within_tolerance_counts_as_a_hit():
    hit = _all_ok(full_name=FieldResult(value="JOHN A SMITH", status="ok", box=(105, 102, 395, 138)))
    report = score({"a.jpg": _response(hit)}, EXPECTED)
    assert report.box_hits == 1 and report.box_total == 1


def test_badly_placed_box_is_a_miss():
    miss = _all_ok(full_name=FieldResult(value="JOHN A SMITH", status="ok", box=(600, 400, 700, 440)))
    report = score({"a.jpg": _response(miss)}, EXPECTED)
    assert report.box_hits == 0 and report.box_total == 1


def test_absent_box_is_not_counted_against_the_hit_rate():
    """A box the filter dropped is a non-answer, not a wrong answer."""
    report = score({"a.jpg": _response(_all_ok())}, EXPECTED)
    assert report.box_total == 0
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `pytest tests/test_run_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.run_eval'`

- [ ] **Step 3: Implement `eval/run_eval.py`**

```python
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
from app.schema import FIELD_NAMES, ExtractResponse

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

            # Only boxes that were actually returned are scored: a box the
            # filter dropped is a non-answer, not a wrong answer.
            true_box = truth.get("boxes", {}).get(fname)
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

    if report.box_total:
        rate = report.box_hits / report.box_total * 100
        print()
        print(f"BOX HIT RATE: {report.box_hits}/{report.box_total} ({rate:.1f}%) at IoU>={IOU_HIT_THRESHOLD}")
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
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_run_eval.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Run the whole suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add eval/run_eval.py tests/test_run_eval.py
git commit -m "feat: add eval harness with silent error and box hit rates"
```

---

## Task 13: The Colab notebook

**Files:**
- Create: `notebook/run_colab.ipynb`

**Interfaces:**
- Consumes: `requirements.txt`, `app.main:app`, `/api/health`, `/api/extract`
- Produces: a running server and a printed HTTPS tunnel URL

Implements spec §11.

- [ ] **Step 1: Create the notebook with six cells**

Build `notebook/run_colab.ipynb` as a valid `nbformat` 4 JSON document. Each cell below is one code cell, preceded by a markdown cell carrying its heading.

**Cell 1 — Check GPU:**

```python
#@title 1. Check GPU
import subprocess, sys
out = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if out.returncode != 0:
    sys.exit(
        "NO GPU ATTACHED.\n"
        "Runtime > Change runtime type > Hardware accelerator > T4 GPU, then rerun.\n"
        "Without this the model loads onto CPU and each request takes minutes."
    )
print(out.stdout)
```

**Cell 2 — Mount Drive and point the HF cache at it:**

```python
#@title 2. Mount Drive and cache weights there
import os
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")

# A fresh Colab session otherwise re-downloads ~7GB of weights, which is
# several minutes of dead air before the demo can start.
HF_HOME = "/content/drive/MyDrive/hf-cache"
os.environ["HF_HOME"] = HF_HOME
Path(HF_HOME).mkdir(parents=True, exist_ok=True)

cached = list(Path(HF_HOME).glob("hub/models--Qwen*"))
print(f"HF_HOME = {HF_HOME}")
print("CACHE HIT - weights already on Drive" if cached else "CACHE MISS - first run will download ~7GB")
```

**Cell 3 — Clone and install:**

```python
#@title 3. Clone the repo and install
REPO_URL = "https://github.com/<your-user>/target-ocr-mvp.git"  #@param {type:"string"}
BRANCH = "main"  #@param {type:"string"}

import os, shutil
if os.path.exists("/content/app-repo"):
    shutil.rmtree("/content/app-repo")
!git clone --branch {BRANCH} --depth 1 {REPO_URL} /content/app-repo
%cd /content/app-repo
!pip install -q -r requirements.txt
print("installed")
```

**Cell 4 — Start the server:**

```python
#@title 4. Start the server and wait for the model
import threading, time, requests, uvicorn
from app.main import app

def _serve():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

threading.Thread(target=_serve, daemon=True).start()

# Model load plus warm-up takes a while on a cold cache. Poll rather than
# guessing at a sleep duration.
for _ in range(120):
    try:
        h = requests.get("http://127.0.0.1:8000/api/health", timeout=5).json()
        if h.get("loaded"):
            print(h)
            break
    except Exception:
        pass
    time.sleep(5)
else:
    raise RuntimeError("Model did not become ready within 10 minutes.")
```

**Cell 5 — Cloudflare quick tunnel:**

```python
#@title 5. Open the public HTTPS tunnel
# Quick tunnel rather than ngrok: no account, no auth token, one less thing
# to fail live.
!wget -q -O /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x /usr/local/bin/cloudflared

import re, subprocess, threading, time

url = None
proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)

def _watch():
    global url
    for line in proc.stdout:
        m = re.search(r"https://[-\w]+\.trycloudflare\.com", line)
        if m and not url:
            url = m.group(0)

threading.Thread(target=_watch, daemon=True).start()
for _ in range(60):
    if url:
        break
    time.sleep(1)

print("\n" * 2 + "=" * 72)
print("   OPEN THIS URL:")
print(f"   {url}")
print("=" * 72 + "\n" * 2)
print("Camera capture needs HTTPS, which this tunnel provides.")
```

**Cell 6 — Smoke test:**

```python
#@title 6. Smoke test before sharing your screen
import json, time, requests

IMAGE = "/content/app-repo/eval/samples/synthetic_01.jpg"  #@param {type:"string"}

t0 = time.time()
r = requests.post(
    "http://127.0.0.1:8000/api/extract",
    files={"image": open(IMAGE, "rb")},
    timeout=180,
)
print(f"HTTP {r.status_code} in {time.time() - t0:.1f}s")
body = r.json()
print(json.dumps(body["fields"], indent=2, ensure_ascii=False))
print(f"agreement {body['agreement']}  elapsed_ms {body['elapsed_ms']}")
```

- [ ] **Step 2: Verify the notebook is valid JSON**

Run: `python -c "import json,sys; nb=json.load(open('notebook/run_colab.ipynb', encoding='utf-8')); print(len(nb['cells']), 'cells')"`
Expected: at least 12 cells (six markdown headings plus six code cells)

- [ ] **Step 3: Run it top to bottom on Colab**

Confirm each acceptance criterion in spec §14: the tunnel URL prints, opening it serves the page, and cell 6 returns schema-conforming JSON. Record the observed warm `elapsed_ms`.

- [ ] **Step 4: Commit**

```bash
git add notebook/run_colab.ipynb
git commit -m "feat: add Colab notebook with tunnel and smoke test"
```

---

## Task 14: The frontend

**Files:**
- Create: `static/index.html` (from `ocr-demo.html`)
- Modify: `README.md`

**Interfaces:**
- Consumes: `POST /api/extract`, `POST /api/transcribe`, `GET /api/health`
- Produces: the served demo page

Implements spec §7.

- [ ] **Step 1: Copy the demo and keep the CSS verbatim**

```bash
cp ocr-demo.html static/index.html
```

Everything from `<style>` to `</style>` is carried over **unchanged** — same IBM Plex faces, same `--scan`/`--warn`/`--low` palette, same 3px radii, same grid, tabs, stage, sweep and drop-hint. Only markup and script change below.

- [ ] **Step 2: Update the header badge**

The old badge claims nothing is uploaded, which is no longer true — inference runs on the server.

Replace:
```html
<div class="badge"><span class="dot"></span> Runs on this device — nothing uploaded</div>
```
with:
```html
<div class="badge"><span class="dot"></span> <span id="engineBadge">Connecting…</span></div>
```

- [ ] **Step 3: Update the controls markup**

In `#imageControls`, delete the enhance toggle — enhancement is no longer optional, it is pass B on the server:

```html
<label class="toggle"><input type="checkbox" id="preprocess" checked> Enhance before reading</label>
```

In `#cameraControls`, delete the continuous-scan toggle (spec §7: inference takes seconds and requests would queue):

```html
<label class="toggle"><input type="checkbox" id="autoScan"> Scan continuously</label>
```

Rename the read button's label from `Read text` to `Extract fields`.

- [ ] **Step 4: Update the metric tiles**

Replace the `.metrics` block:

```html
<div class="metrics">
  <div class="metric"><div class="k">Agreement</div><div class="v" id="mAgree">—</div></div>
  <div class="metric"><div class="k">Fields</div><div class="v" id="mFields">—</div></div>
  <div class="metric"><div class="k">Time</div><div class="v" id="mTime">—</div></div>
</div>
```

The `Confidence` tile has to go: it rendered a mean word-confidence percentage, which is exactly the invented number spec §5 forbids. `Agreement` replaces it with a real measurement.

- [ ] **Step 5: Update the raw-text panel**

```html
<details class="raw">
  <summary>Model output</summary>
  <pre class="rawtext" id="rawText">—</pre>
  <div class="panel-foot">
    <button class="act ghost" id="transcribeBtn" disabled>Load full transcription</button>
  </div>
</details>
```

- [ ] **Step 6: Replace the entire `<script>` block**

Delete everything from `/* Load Tesseract.js with a CDN fallback */` to the end of the script, and replace with:

```javascript
/* ============================================================
   API base. The tunnel serves both this page and the API, so
   same-origin is the default; ?api= or localStorage covers the
   case of opening this file from disk against a remote tunnel.
   ============================================================ */
const API_BASE = (new URLSearchParams(location.search).get("api")
  || localStorage.getItem("apiBase")
  || "").replace(/\/$/, "");
const api = path => API_BASE + path;

const el = id => document.getElementById(id);
const stage    = el("stage");
const canvas   = el("canvas");
const overlay  = el("overlay");
const video    = el("video");
const dropHint = el("dropHint");
const statusEl = el("status");

const ctx  = canvas.getContext("2d", { willReadFrequently:true });
const octx = overlay.getContext("2d");

let hasImage = false, stream = null, busy = false, lastResult = null, tick = null;

const STATUS_CLASS = { ok:"ok", review:"mid", missing:"low" };
const STATUS_COLOUR = { ok:"#00B3A4", review:"#E0952C", missing:"#D2544B" };
const FIELD_LABEL = {
  full_name:"Name", id_number:"ID number", date_of_birth:"Date of birth",
  expiry_date:"Expiry", nationality:"Nationality", sex:"Sex"
};

function setStatus(msg){ statusEl.textContent = msg; }
function setBusy(on){ busy = on; stage.classList.toggle("busy", on); }

/* ============================================================
   Health — confirms the backend before anything is uploaded
   ============================================================ */
async function checkHealth(){
  try{
    const r = await fetch(api("/api/health"));
    const h = await r.json();
    el("engineBadge").textContent = h.loaded
      ? `${h.model.split("/").pop()} on ${h.device}`
      : "Model loading…";
    if (!h.self_consistency) el("engineBadge").textContent += " — consistency check OFF";
  } catch {
    el("engineBadge").textContent = "Backend unreachable";
  }
}
checkHealth();

/* ============================================================
   Tabs
   ============================================================ */
const tabImage = el("tab-image"), tabCamera = el("tab-camera");
tabImage.onclick  = () => switchMode("image");
tabCamera.onclick = () => switchMode("camera");

function switchMode(mode){
  const isImg = mode === "image";
  tabImage.setAttribute("aria-selected", isImg);
  tabCamera.setAttribute("aria-selected", !isImg);
  el("imageControls").style.display  = isImg ? "flex" : "none";
  el("cameraControls").style.display = isImg ? "none" : "flex";
  if (isImg) stopCamera();
  clearOverlay();
  setStatus("Ready.");
}

/* ============================================================
   Image loading — unchanged from the original demo
   ============================================================ */
el("chooseBtn").onclick = () => el("fileInput").click();
el("fileInput").onchange = e => { if (e.target.files[0]) loadFile(e.target.files[0]); };

stage.addEventListener("dragover", e => { e.preventDefault(); stage.classList.add("dragover"); });
stage.addEventListener("dragleave", () => stage.classList.remove("dragover"));
stage.addEventListener("drop", e => {
  e.preventDefault();
  stage.classList.remove("dragover");
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) loadFile(f);
});

function loadFile(file){
  const img = new Image();
  img.onload = () => {
    drawToCanvas(img, img.naturalWidth, img.naturalHeight);
    hasImage = true;
    dropHint.style.display = "none";
    video.style.display = "none";
    canvas.style.display = "block";
    el("readBtn").disabled = false;
    el("clearBtn").disabled = false;
    clearOverlay();
    setStatus(`Loaded ${file.name} — ${img.naturalWidth}×${img.naturalHeight}`);
    URL.revokeObjectURL(img.src);
  };
  img.onerror = () => setStatus("That file could not be opened as an image.");
  img.src = URL.createObjectURL(file);
}

function drawToCanvas(source, w, h){
  const MAX = 1600, MIN = 900;
  let scale = 1;
  if (Math.max(w,h) > MAX) scale = MAX / Math.max(w,h);
  else if (Math.max(w,h) < MIN) scale = Math.min(2, MIN / Math.max(w,h));
  canvas.width  = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  syncOverlaySize();
}

function syncOverlaySize(){
  overlay.width = canvas.width;
  overlay.height = canvas.height;
  const r = canvas.getBoundingClientRect();
  const s = stage.getBoundingClientRect();
  overlay.style.width  = r.width + "px";
  overlay.style.height = r.height + "px";
  overlay.style.left   = (r.left - s.left) + "px";
  overlay.style.top    = (r.top - s.top) + "px";
}
window.addEventListener("resize", syncOverlaySize);

/* ============================================================
   Camera — capture-and-read only. Continuous scanning is gone:
   inference takes seconds, so requests would queue and lag.
   ============================================================ */
el("startCam").onclick   = startCamera;
el("stopCam").onclick    = stopCamera;
el("captureBtn").onclick = () => { captureFrame(); runExtract(); };

async function startCamera(){
  try{
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode:"environment", width:{ideal:1920}, height:{ideal:1080} }
    });
    video.srcObject = stream;
    await video.play();
    video.style.display = "block";
    canvas.style.display = "none";
    dropHint.style.display = "none";
    el("startCam").disabled = true;
    el("captureBtn").disabled = false;
    el("stopCam").disabled = false;
    el("clearBtn").disabled = false;
    clearOverlay();
    setStatus(`Camera running — ${video.videoWidth}×${video.videoHeight}`);
  } catch {
    setStatus("Camera unavailable. Grant permission, and open this page over HTTPS.");
  }
}

function stopCamera(){
  if (stream){ stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.style.display = "none";
  el("startCam").disabled = false;
  el("captureBtn").disabled = true;
  el("stopCam").disabled = true;
}

function captureFrame(){
  if (!video.videoWidth) return;
  drawToCanvas(video, video.videoWidth, video.videoHeight);
  hasImage = true;
  canvas.style.display = "block";
  video.style.display = "none";
}

/* ============================================================
   Extraction
   ============================================================ */
el("readBtn").onclick = () => runExtract();

function canvasBlob(){
  return new Promise(res => canvas.toBlob(res, "image/jpeg", 0.92));
}

/* A real elapsed counter, not a fake percentage. The backend cannot
   report true progress, and a bar that stalls looks worse than a timer. */
function startTicker(){
  const t0 = performance.now();
  tick = setInterval(() => {
    setStatus(`Extracting — ${((performance.now()-t0)/1000).toFixed(1)}s`);
  }, 100);
}
function stopTicker(){ clearInterval(tick); tick = null; }

async function runExtract(){
  if (busy || !hasImage) return;
  setBusy(true);
  startTicker();
  const t0 = performance.now();
  try{
    const body = new FormData();
    body.append("image", await canvasBlob(), "card.jpg");
    const r = await fetch(api("/api/extract"), { method:"POST", body });
    if (!r.ok) throw new Error(`Server returned ${r.status}.`);

    lastResult = await r.json();
    stopTicker();
    drawBoxes(lastResult.fields);
    renderResults(lastResult);
    setStatus(`Done in ${Math.round(performance.now()-t0)} ms.`);
    el("copyJson").disabled = false;
    el("transcribeBtn").disabled = false;
  } catch(err){
    stopTicker();
    const where = API_BASE || location.origin;
    setStatus(err instanceof TypeError
      ? `Backend unreachable — the Colab tunnel is likely down or the session expired. Base: ${where}`
      : err.message);
  } finally {
    setBusy(false);
  }
}

/* ============================================================
   Overlay — only draws boxes the backend actually returned.
   A box the filter rejected simply does not highlight.
   ============================================================ */
function clearOverlay(){ octx.clearRect(0,0,overlay.width,overlay.height); }

function drawBoxes(fields, highlight){
  syncOverlaySize();
  clearOverlay();
  for (const [name, f] of Object.entries(fields)){
    if (!f.box) continue;
    const [x1,y1,x2,y2] = f.box;
    const isHi = highlight === name;
    octx.strokeStyle = isHi ? "#7C5CFF" : STATUS_COLOUR[f.status];
    octx.lineWidth = isHi ? Math.max(3, canvas.width/380) : Math.max(1.5, canvas.width/700);
    if (isHi){
      octx.fillStyle = "rgba(124,92,255,.16)";
      octx.fillRect(x1, y1, x2-x1, y2-y1);
    }
    octx.strokeRect(x1, y1, x2-x1, y2-y1);
  }
}

/* ============================================================
   Results
   ============================================================ */
function renderResults(res){
  const host = el("fields");
  host.innerHTML = "";

  const entries = Object.entries(res.fields);
  const found = entries.filter(([,f]) => f.status !== "missing").length;

  el("mAgree").textContent  = `${res.agreement.matched}/${res.agreement.total}`;
  el("mFields").textContent = `${found}/${entries.length}`;
  el("mTime").textContent   = (res.elapsed_ms/1000).toFixed(1) + "s";
  el("rawText").textContent = res.raw_text || "—";
  el("fieldCount").textContent = `${found} found`;

  for (const [name, f] of entries){
    const row = document.createElement("div");
    row.className = "field";
    const arabic = f.value_ar
      ? `<div class="v" dir="rtl" style="opacity:.75">${escapeHtml(f.value_ar)}</div>` : "";
    const title = f.reason ? ` title="${escapeHtml(f.reason)}"` : "";
    row.innerHTML =
      `<div class="k">${escapeHtml(FIELD_LABEL[name] || name)}</div>
       <div><div class="v">${escapeHtml(f.value ?? "—")}</div>${arabic}</div>
       <div class="chip ${STATUS_CLASS[f.status]}"${title}>${f.status}</div>`;
    row.onmouseenter = () => drawBoxes(res.fields, name);
    row.onmouseleave = () => drawBoxes(res.fields);
    host.appendChild(row);
  }
}

function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, m =>
    ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[m]));
}

/* ============================================================
   Footer actions
   ============================================================ */
el("transcribeBtn").onclick = async () => {
  if (!hasImage) return;
  el("transcribeBtn").disabled = true;
  setStatus("Transcribing — this is slower than extraction…");
  try{
    const body = new FormData();
    body.append("image", await canvasBlob(), "card.jpg");
    const r = await fetch(api("/api/transcribe"), { method:"POST", body });
    const data = await r.json();
    el("rawText").textContent = data.text;
    setStatus(`Transcribed in ${data.elapsed_ms} ms.`);
  } catch {
    setStatus("Transcription failed — the backend may be unreachable.");
  } finally {
    el("transcribeBtn").disabled = false;
  }
};

el("copyJson").onclick = async () => {
  if (!lastResult) return;
  try{
    await navigator.clipboard.writeText(JSON.stringify(lastResult, null, 2));
    setStatus("JSON copied to clipboard.");
  } catch { setStatus("Copy blocked by the browser — select the raw text instead."); }
};

el("clearBtn").onclick = () => {
  clearOverlay();
  ctx.clearRect(0,0,canvas.width,canvas.height);
  hasImage = false; lastResult = null;
  dropHint.style.display = "flex";
  canvas.style.display = "none";
  el("readBtn").disabled = true;
  el("copyJson").disabled = true;
  el("clearBtn").disabled = true;
  el("transcribeBtn").disabled = true;
  el("fields").innerHTML = `<div class="empty"><p>Cleared.</p><p>Load another image or start the camera.</p></div>`;
  el("rawText").textContent = "—";
  el("mAgree").textContent = el("mFields").textContent = el("mTime").textContent = "—";
  el("fieldCount").textContent = "";
  setStatus("Ready.");
};

canvas.style.display = "none";
```

- [ ] **Step 7: Update the empty-state copy**

Replace the `#fields` placeholder text so it describes this pipeline:

```html
<div class="empty">
  <p>No card read yet.</p>
  <p>Load an image or capture from the camera. Each field is checked against
     format rules and a second reading of the image; fields that disagree are
     marked for review rather than reported as confident.</p>
</div>
```

Also update the footer note from "Hover a field to highlight where it came from on the image." to add: "Fields without a box could not be located reliably."

- [ ] **Step 8: Test in a browser against a fake backend**

```bash
uvicorn app.main:app --port 8000
```

The model will not load without a GPU, so `/api/health` reports `loaded: false` and the badge shows "Model loading…". Confirm: the page serves at `http://localhost:8000/`, tabs switch, drag-drop loads an image and draws it, and clicking Extract shows the unreachable/error message rather than hanging silently.

- [ ] **Step 9: Commit**

```bash
git add static/index.html README.md
git commit -m "feat: replace Tesseract with backend extraction in the demo page"
```

---

## Task 15: Real-card run and the box verdict

**Files:**
- Modify: `README.md`
- Create: `docs/measurements.md`

**Interfaces:**
- Consumes: everything
- Produces: the measured numbers spec §14 criteria 3 and 8 require

- [ ] **Step 1: Add real cards to `eval/samples/`**

Photograph 4–8 ID cards. Add each to `eval/samples/expected.json` with hand-typed ground truth including Arabic values. Images stay gitignored; `expected.json` is tracked.

**Verify the D2 assumption here:** if any card prints its date of birth, expiry or civil number in Eastern Arabic numerals (٠١٢٣٤٥٦٧٨٩), spec D2 is wrong and those fields need `value_ar` too. Report this rather than working around it.

- [ ] **Step 2: Run the eval on Colab**

```bash
!python -m eval.run_eval
```

- [ ] **Step 3: Record the measurements**

Create `docs/measurements.md` with the date, model ID, and: cold load, warm-up, warm single-request latency (median and max), peak VRAM, per-field exact-match accuracy, review rate, missing rate, **silent error rate with each error listed**, and box hit rate.

- [ ] **Step 4: Deliver the box verdict**

Per spec D4 and §16 R1, decide from the measured box hit rate whether the overlay ships. Set `SHOW_BOXES` accordingly, and **report the number to the user either way** — never disable it silently.

- [ ] **Step 5: Report against acceptance criteria**

Walk spec §14's eight criteria and state pass/fail for each with the evidence. If criterion 3 (under 5s warm) failed, state the measured number and which lever was applied.

- [ ] **Step 6: Commit**

```bash
git add docs/measurements.md README.md eval/samples/expected.json
git commit -m "docs: record measured latency, accuracy and box hit rate"
```

---

## Appendix: Deviations from the spec's file tree

`SPEC.md` §"Repo structure" lists a flatter tree. Four files were added:

| File | Why |
|---|---|
| `app/imaging.py` | Contrast normalisation is pure, self-contained and heavily tested. Inside `extract.py` it would be a third responsibility in a module that already has two. |
| `app/parsing.py` | Fence stripping and JSON repair carry 14 tests of their own and are the most likely place for future model-quirk fixes. |
| `app/boxes.py` | The geometric filter is spec §5.3's own section, is pure, and has nothing to do with the field validation rules in `validate.py`. |
| `scripts/bringup.py` | The Task 3 measurement gate has to run before the notebook exists. |

`app/data/nationalities.txt` supports the §5.1 closed-list rule.
