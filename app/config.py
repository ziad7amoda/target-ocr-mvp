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

    # Revision 2026-08-24: 256 truncated long Arabic names on a real card (a
    # seven-component name hit the ceiling mid-word). Raised to 320, which is
    # affordable now that nationality/sex - three redundant keys including
    # their _ar duplicates - are gone from the output contract.
    MAX_NEW_TOKENS: int = 320
    # Vision-token bounds handed to the processor. Prefill is cheap relative to
    # decode, but capping keeps a 12MP phone photo from ballooning the batch.
    MIN_PIXELS: int = 256 * 28 * 28
    MAX_PIXELS: int = 1280 * 28 * 28

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
