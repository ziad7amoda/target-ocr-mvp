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
        if engine is None:
            # Imported here so the module stays importable without torch.
            from app.model import QwenEngine

            app.state.engine = QwenEngine(settings)
            app.state.loaded = False
            logger.info("loading model %s", settings.MODEL_ID)
            await run_in_threadpool(app.state.engine.load)
            warm = await run_in_threadpool(app.state.engine.warmup)
            app.state.loaded = True
            logger.info("model ready, warmup_ms=%d", warm)
        yield

    app = FastAPI(title="ID card extraction", lifespan=lifespan)
    # Set eagerly (not only in lifespan) so an injected test-double engine is
    # usable via a bare TestClient() call, without requiring callers to enter
    # the client as a context manager to trigger ASGI startup. Production
    # still gets its real engine exclusively through lifespan, above.
    app.state.engine = engine
    app.state.loaded = engine is not None
    # One GPU, one model: serialise inference so two requests cannot
    # interleave on the same CUDA context.
    app.state.gpu_lock = asyncio.Lock()
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
