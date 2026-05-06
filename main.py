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
    WHISPER_LANGUAGES
        Comma-separated allow-list in priority order. Detected
        languages outside this list are coerced to the first entry,
        guarding against autodetect mistakes (e.g., quiet Ukrainian
        being labelled Russian). Empty = no restriction.
        Default: ``uk,en``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("whisper")

MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
VAD_ENABLED = os.getenv("WHISPER_VAD", "1") not in ("0", "false", "False")

# Comma-separated allow-list, priority order. Whisper's autodetect
# spans ~99 languages; we constrain it to a curated set so that
# misdetections (e.g., quiet Ukrainian getting labelled as Russian)
# fall back to the top-priority entry instead of being transcribed
# in a language the user doesn't speak.
ALLOWED_LANGUAGES = [
    s.strip().lower()
    for s in os.getenv("WHISPER_LANGUAGES", "uk,en").split(",")
    if s.strip()
]


def _choose_language(
    detected: str,
    all_probs: list[tuple[str, float]] | None = None,
) -> str:
    """Pick the language we'll use for the actual transcription.

    Logic:
      1. If autodetect picked a language we accept, keep it.
      2. Otherwise, look at the per-language probabilities the model
         produced (sorted descending) and pick the first one that's
         in the allow-list. This is "what the model thinks the audio
         is, restricted to languages we'll accept" — the smart
         fallback for near-boundary cases like Ukrainian misdetected
         as Russian.
      3. If no allowed language appears in the probabilities at all
         (or the list is missing), fall back to the first entry of
         ``WHISPER_LANGUAGES``.

    Returns an empty string when ``WHISPER_LANGUAGES`` is empty,
    meaning "no constraint" — the caller should pass ``language=None``
    to faster-whisper in that case.
    """
    if not ALLOWED_LANGUAGES:
        return ""
    if detected in ALLOWED_LANGUAGES:
        return detected
    if all_probs:
        for lang, _ in all_probs:
            if lang in ALLOWED_LANGUAGES:
                return lang
    return ALLOWED_LANGUAGES[0]

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
        # First call: autodetect language. `info` is populated eagerly
        # (one fast encoder pass + language head); the segments
        # generator is lazy so no decoding has happened yet.
        segments, info = state["model"].transcribe(
            path,
            language=None,
            task=task,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_ENABLED,
        )
        chosen = _choose_language(info.language, info.all_language_probs)
        if chosen and chosen != info.language:
            # Detected a language we don't accept → discard the lazy
            # generator (no work done yet) and re-run forced to the
            # priority language.
            segments, info = state["model"].transcribe(
                path,
                language=chosen,
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


@app.post("/transcribe-stream")
async def transcribe_stream(
    audio: UploadFile = File(
        ...,
        description="Audio file (wav / mp3 / m4a / webm / ogg / flac)",
    ),
    task: str = Form(
        default="transcribe",
        description="'transcribe' or 'translate' (to English)",
    ),
) -> StreamingResponse:
    """Same as /transcribe but emits segments as they decode.

    Audio is uploaded fully, but Whisper processes it in 30-second
    windows internally and faster-whisper yields ``Segment`` objects
    as each window finishes. We forward each segment over the wire as
    one NDJSON line so clients can show partial transcripts mid-stream
    instead of waiting for the whole file.

    Wire format (newline-delimited JSON):
        {"meta":   {"language": "uk", "language_probability": 0.99,
                    "duration": 120.5}}\\n
        {"segment":{"start": 0.0, "end": 5.4,
                    "text": "first sentence."}}\\n
        {"segment":{"start": 5.4, "end": 11.2,
                    "text": "second sentence."}}\\n
        ...
        {"done":   {"text": "first sentence. second sentence. ..."}}\\n
    """
    if not audio.filename:
        raise HTTPException(400, "audio file required")
    if task not in ("transcribe", "translate"):
        raise HTTPException(400, "task must be 'transcribe' or 'translate'")
    if "model" not in state:
        raise HTTPException(503, "model still loading")

    suffix = os.path.splitext(audio.filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        path = tmp.name
        while chunk := await audio.read(1 << 20):  # 1 MiB chunks
            tmp.write(chunk)

    def _ndjson_lines():
        try:
            # Autodetect first, then re-run forced to a priority
            # language if the detection fell outside the allow-list.
            # Both calls are essentially free until the segment
            # generator is iterated, so the wasted work in the
            # bad-detection case is just one encoder pass + the
            # language head (~100-300ms total).
            segments, info = state["model"].transcribe(
                path,
                language=None,
                task=task,
                beam_size=BEAM_SIZE,
                vad_filter=VAD_ENABLED,
            )
            chosen = _choose_language(info.language, info.all_language_probs)
            if chosen and chosen != info.language:
                segments, info = state["model"].transcribe(
                    path,
                    language=chosen,
                    task=task,
                    beam_size=BEAM_SIZE,
                    vad_filter=VAD_ENABLED,
                )
            yield json.dumps({"meta": {
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration": round(info.duration, 2),
            }}, ensure_ascii=False) + "\n"

            collected: list[str] = []
            for s in segments:
                text = s.text.strip()
                collected.append(text)
                yield json.dumps({"segment": {
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "text": text,
                }}, ensure_ascii=False) + "\n"

            yield json.dumps({"done": {
                "text": " ".join(collected).strip(),
            }}, ensure_ascii=False) + "\n"
        except Exception as e:
            log.exception("stream transcription failed")
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    return StreamingResponse(
        _ndjson_lines(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"},  # disable Nginx/CF buffering
    )
