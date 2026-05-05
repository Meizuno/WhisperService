"""FastAPI service that wraps faster-whisper for audio transcription.

Drop-in replacement for the OpenAI Whisper API on a CPU VPS.

Endpoints
    POST /transcribe
        Multipart audio + optional language hint.
        Returns ``{text, language, language_probability, duration}``.
    GET /health
        ``{status, model, device}`` for container probes.

Configuration (env vars)
    WHISPER_MODEL
        Model name or local path. Common values: ``tiny``, ``base``,
        ``small``, ``medium``, ``large-v3`` (suffix ``.en`` for the
        English-only variants). Default: ``medium``.
    WHISPER_DEVICE
        ``cpu`` or ``cuda``. Default: ``cpu``.
    WHISPER_COMPUTE_TYPE
        ``int8`` / ``int8_float16`` / ``float16`` / ``float32``.
        Default: ``int8`` — halves memory at minor accuracy cost.
    WHISPER_BEAM_SIZE
        Beam search width (1-10). Default: ``5``.
    WHISPER_VAD
        ``1`` to enable Silero VAD (skip silences). Default: ``1``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("whisper")

MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
VAD_ENABLED = os.getenv("WHISPER_VAD", "1") not in ("0", "false", "False")

# In-memory state. Populated once at startup; the model is heavy
# (~1-3 GB depending on size) so we keep one instance for the
# process lifetime instead of loading per request.
state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info(
        "loading model %s on %s (compute=%s)",
        MODEL_NAME,
        DEVICE,
        COMPUTE_TYPE,
    )
    state["model"] = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )
    log.info("model loaded; ready")
    yield
    log.info("shutting down")


app = FastAPI(title="Whisper service", lifespan=lifespan)

# Service is intended to run on a private docker network. CORS is
# permissive so any local container can call it; do not expose the
# port to the public internet.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok" if "model" in state else "loading",
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(
        ...,
        description="Audio file (wav / mp3 / m4a / webm / ogg / flac)",
    ),
    language: str | None = Form(
        default=None,
        description=(
            "ISO 639-1 hint, e.g. 'en' or 'uk'. Omit to autodetect."
        ),
    ),
    task: str = Form(
        default="transcribe",
        description="'transcribe' or 'translate' (to English)",
    ),
) -> dict[str, Any]:
    if not audio.filename:
        raise HTTPException(400, "audio file required")
    if task not in ("transcribe", "translate"):
        raise HTTPException(400, "task must be 'transcribe' or 'translate'")
    if "model" not in state:
        raise HTTPException(503, "model still loading")

    # faster-whisper reads from a path or BytesIO. Buffer the upload
    # to a tmp file so we don't hold the whole audio in memory.
    suffix = os.path.splitext(audio.filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        path = tmp.name
        while chunk := await audio.read(1 << 20):  # 1 MiB chunks
            tmp.write(chunk)

    try:
        segments, info = state["model"].transcribe(
            path,
            language=language,
            task=task,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_ENABLED,
        )
        # ``transcribe`` returns a lazy generator that streams segments
        # as it decodes. Materialize it fully here for a single JSON
        # response.
        text = " ".join(s.text.strip() for s in segments)
        return {
            "text": text.strip(),
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
        }
    except Exception as e:
        log.exception("transcription failed")
        raise HTTPException(500, f"transcription failed: {e}") from e
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
