FROM python:3.11-slim AS builder

WORKDIR /build

# Install Python deps into an isolated venv that we copy across stages.
# Keeps the runner image free of pip caches and build tools.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runner

# ffmpeg is required by faster-whisper for non-WAV inputs (mp3, m4a,
# webm, ogg). Skipping it would limit the service to PCM WAV uploads.
RUN apt-get update \
 && apt-get install --no-install-recommends -y ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY main.py .

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    WHISPER_MODEL=medium \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8 \
    HF_HOME=/cache

# Models cache here; mount a volume to /cache to persist across restarts.
RUN mkdir -p /cache

EXPOSE 8000

# Long start-period — model load is 30-90s for medium, longer for
# large-v3. Health probes during that window get a `loading` status
# from /health and return 200, so the container is "healthy" once the
# webserver is up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3); sys.exit(0 if r.status==200 else 1)" || exit 1

# `--workers 1` is intentional: each worker would load its own copy
# of the model (~1-3 GB). For CPU inference there's no benefit to
# multi-worker on a single VPS — Python's GIL doesn't matter because
# the inference runs in CTranslate2 native code.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
