from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # protected_namespaces=() is required: pydantic v2 reserves the "model_"
    # prefix and MODEL_ID would otherwise emit a warning on every import.
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=()
    )

    MODEL_ID: str = "MBZUAI/AIN"
    DEVICE: str = "auto"
    # AIN's config.json declares bfloat16, but this deliberately overrides
    # that: the T4 (compute capability 7.5) has no bf16 support, so float16
    # is the correct choice regardless of what the checkpoint asks for.
    TORCH_DTYPE: str = "float16"
    # AIN is a 7B model (~15GB of weights in fp16), which does not fit
    # unloaded on a 16GB T4 alongside CUDA overhead and activations. Loading
    # in 8-bit (~7.5GB) is required to fit. This trades decode speed for the
    # memory saving - set to False only on a GPU with enough headroom for
    # fp16 (e.g. when falling back to the 3B Qwen2.5-VL model).
    LOAD_IN_8BIT: bool = True

    # Revision 2026-08-24: 256 truncated long Arabic names on a real card (a
    # seven-component name hit the ceiling mid-word). Raised to 320, which is
    # affordable now that nationality/sex - three redundant keys including
    # their _ar duplicates - are gone from the output contract.
    MAX_NEW_TOKENS: int = 320
    # Vision-token bounds handed to the processor. Prefill is cheap relative to
    # decode, but capping keeps a 12MP phone photo from ballooning the batch.
    MIN_PIXELS: int = 256 * 28 * 28
    MAX_PIXELS: int = 1280 * 28 * 28

    # Revision 2026-08-25 (prompt-style comparison): "strict" is the prompt
    # tuned iteratively against Qwen2.5-VL-3B - it fixes observed day/month
    # swaps, name-truncation, and label/value confusion bugs on that model,
    # but the accumulated negative coaching does not transfer: run against
    # NAMAA-Space/Qari-OCR-0.4.0-VL-4B it produced valid JSON with the two
    # Arabic fields deliberately null. "natural" is a minimal prompt that
    # asks plainly for what is wanted and is the fairer choice when
    # comparing models. Neither style is uniformly better - which to use is
    # a per-model choice, made from evidence on real cards, not a default
    # to trust blindly.
    PROMPT_STYLE: str = "natural"

    SELF_CONSISTENCY: bool = True
    # Revision 2026-08-24: grounding returned no usable boxes in the live run
    # against real cards. Per the design's own rule (D4), an overlay that
    # cannot be drawn correctly is dropped rather than drawn wrong, so this
    # now defaults off until grounding is proven to work.
    SHOW_BOXES: bool = False

    # Development only. Writes card images to disk; see README.
    DEBUG_SAVE_IMAGES: bool = False
    DEBUG_SAVE_DIR: str = "./_debug_images"

    ALLOWED_ORIGINS: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
