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


class QwenEngine:
    """Real inference. The only class in the project that needs a GPU.

    Loads any Qwen2-VL-family checkpoint via AutoModelForImageTextToText,
    which dispatches on the model's own `architectures` config entry rather
    than a hardcoded model class. This covers both Qwen/Qwen2.5-VL-3B-Instruct
    and derivatives such as MBZUAI/AIN (which declares
    Qwen2VLForConditionalGeneration) with no branching in this file.

    T4 notes, recorded so they are not rediscovered: compute capability 7.5
    (sm75) means no bf16 (fp16 is correct here, regardless of what a
    checkpoint's config.json declares) and no FlashAttention-2, which needs
    8.0+. SDPA is the ceiling. A 3B model in fp16 is ~6GB of the 16GB card
    and needs no quantisation; a 7B model like AIN is ~15GB in fp16, which
    does not fit, so LOAD_IN_8BIT (~7.5GB via bitsandbytes) is required for
    those.
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

        # This check must run before any from_pretrained call, i.e. before any
        # network I/O. With DEVICE=auto (the default) and no GPU, the naive
        # sequence is: download ~6GB of weights from HuggingFace, THEN
        # discover there's no CUDA device, THEN grind through a slow CPU
        # load anyway. A multi-gigabyte download triggered by a default
        # setting is a trap, not a fallback - fail fast instead, before a
        # single byte moves. DEVICE=cpu remains a deliberate opt-in escape
        # hatch for the (very slow) CPU path; only the accidental case is
        # blocked.
        if self._settings.DEVICE == "auto" and not torch.cuda.is_available():
            raise RuntimeError(
                "No GPU detected (torch.cuda.is_available() is False). "
                "Refusing to download and load the model with DEVICE=auto, "
                "since a CPU load of a multi-gigabyte VLM is impractical by "
                "accident. The model was NOT downloaded. If you really want "
                "to run on CPU, set DEVICE=cpu explicitly - this will work "
                "but is extremely slow (likely minutes per request instead "
                "of seconds)."
            )

        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = getattr(torch, self._settings.TORCH_DTYPE)
        load_kwargs = dict(
            dtype=dtype,  # transformers 5.x: `torch_dtype` is deprecated in
            # favour of `dtype` (still accepted for backwards compatibility,
            # but emits `torch_dtype is deprecated! Use dtype instead!`).
            # Confirmed by inspecting PreTrainedModel.from_pretrained in the
            # installed transformers==5.15.1: it pops both `dtype` and
            # `torch_dtype` from kwargs and prefers `dtype` when both are set.
            attn_implementation="sdpa",
            device_map=self.device,
        )
        if self._settings.LOAD_IN_8BIT:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._settings.MODEL_ID, **load_kwargs
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
