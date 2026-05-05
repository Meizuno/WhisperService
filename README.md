# Whisper service

Self-hosted speech-to-text via [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper),
exposed over HTTP. Designed as a sidecar container on a private docker network.

- **One model, one process.** Loaded at startup, kept in memory.
- **CPU first.** Runs fine on a 2-vCPU / 2 GB box for `medium`. CUDA optional via env.
- **Multi-language.** English, Ukrainian, and ~95 others auto-detected.

## Endpoints

### `POST /transcribe`

Multipart form:

| field      | type | notes |
|------------|------|-------|
| `audio`    | file | wav / mp3 / m4a / webm / ogg / flac (anything ffmpeg accepts) |
| `language` | string, optional | ISO 639-1 hint (`en`, `uk`). Omit to autodetect. |
| `task`     | string, optional | `transcribe` (default) or `translate` (to English) |

Response:

```json
{
  "text": "…",
  "language": "uk",
  "language_probability": 0.997,
  "duration": 12.4
}
```

### `GET /health`

```json
{ "status": "ok", "model": "medium", "device": "cpu", "compute_type": "int8" }
```

## Configuration (env)

| var | default | meaning |
|---|---|---|
| `WHISPER_MODEL` | `medium` | `tiny` / `base` / `small` / `medium` / `large-v3` (suffix `.en` for English-only) |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` / `int8_float16` / `float16` / `float32` |
| `WHISPER_BEAM_SIZE` | `5` | beam search width |
| `WHISPER_VAD` | `1` | enable Silero VAD (skip silences) |

### Model size vs RAM (int8 on CPU, approximate)

| Model | RAM | English | Ukrainian |
|---|---|---|---|
| `tiny` | ~250 MB | usable | poor |
| `base` | ~400 MB | good | barely usable |
| `small` | ~800 MB | great | usable for clean audio |
| `medium` | ~1.8 GB | excellent | decent |
| `large-v3` | ~3.5 GB | excellent | great — recommended |

## Run

### Local (Python ≥ 3.11)

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
WHISPER_MODEL=small uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t whisper-service .
docker run --rm -p 8000:8000 \
  -e WHISPER_MODEL=medium \
  -v whisper-cache:/home/whisper/.cache/huggingface \
  whisper-service
```

### docker-compose

See `docker-compose.example.yml` for a service block to drop into your stack.

First start downloads the model into the persisted volume (~1.5 GB for `medium`,
~3 GB for `large-v3`). Subsequent starts read from the cache and are ~30 s.

## Quick test

```bash
curl -F audio=@sample.wav -F language=uk http://localhost:8000/transcribe
```

## Calling from a Nuxt / Node app

```ts
const fd = new FormData()
fd.append('audio', new Blob([buffer], { type: 'audio/webm' }), 'audio.webm')
fd.append('language', 'uk')      // optional

const res = await $fetch<{ text: string }>('http://whisper:8000/transcribe', {
  method: 'POST',
  body: fd
})
```

(`http://whisper:...` works from another container on the same docker network.)

## Performance notes

- CPU inference of `medium` runs ~1× realtime on a 2-vCPU box. `large-v3` is ~0.3-0.5×.
- VAD pre-filtering speeds up real-world audio (with silences) substantially.
- A GPU drops latency 5-10×. Set `WHISPER_DEVICE=cuda` and use a CUDA-enabled
  base image (not in this Dockerfile).
- The service uses one uvicorn worker on purpose. Multiple workers would each
  load their own copy of the model.
