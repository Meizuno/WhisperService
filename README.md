# Whisper service

Self-hosted speech-to-text via [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper),
exposed over HTTP. Designed as a sidecar container on a private docker network.

- **One model, one process.** Loaded at startup, kept in memory.
- **CPU first.** Runs fine on a 2-vCPU / 2 GB box for `medium`. CUDA optional via env.
- **Multi-language.** English, Ukrainian, and ~95 others auto-detected.

## Endpoints

### `POST /transcribe`

Multipart form:

| field   | type | notes |
|---------|------|-------|
| `audio` | file | wav / mp3 / m4a / webm / ogg / flac (anything ffmpeg accepts) |
| `task`  | string, optional | `transcribe` (default) or `translate` (to English) |

Language is autodetected, then constrained to `WHISPER_LANGUAGES` (env, default
`uk,en`) — see "Configuration" below. There is no per-request language override.

Response:

```json
{
  "text": "…",
  "language": "uk",
  "language_probability": 0.997,
  "duration": 12.4
}
```

### `POST /transcribe-stream`

Same multipart shape as `/transcribe`, but the response is `application/x-ndjson` —
one JSON object per line, sent as each segment finishes decoding. Lets the client
display partial transcripts before the full file is processed.

```text
{"meta":   {"language":"uk","language_probability":0.997,"duration":12.4}}
{"segment":{"start":0.00,"end":3.20,"text":"Перше речення."}}
{"segment":{"start":3.20,"end":7.80,"text":"Друге речення."}}
...
{"done":   {"text":"Перше речення. Друге речення. ..."}}
```

`done` carries the joined transcript so the client doesn't have to reassemble it
from segments.

Why this works: Whisper internally processes audio in 30-second windows.
faster-whisper's `model.transcribe()` returns a *generator* that yields each
segment as its window finishes — this endpoint just forwards those yields over
the wire instead of materializing them all.

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
| `WHISPER_LANGUAGES` | `uk,en` | comma-separated allow-list, priority order. Detected languages outside this list are coerced to the first entry. Set empty to allow any language. |

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

### Buffered (single response)

```ts
const fd = new FormData()
fd.append('audio', new Blob([buffer], { type: 'audio/webm' }), 'audio.webm')

const res = await $fetch<{ text: string }>('http://whisper:8000/transcribe', {
  method: 'POST',
  body: fd
})
```

### Streamed (segments as they decode)

```ts
const fd = new FormData()
fd.append('audio', new Blob([buffer], { type: 'audio/webm' }), 'audio.webm')

const res = await fetch('http://whisper:8000/transcribe-stream', {
  method: 'POST',
  body: fd,
})

const reader = res.body!.getReader()
const dec = new TextDecoder()
let buf = ''
const parts: string[] = []

while (true) {
  const { done, value } = await reader.read()
  if (done) break
  buf += dec.decode(value, { stream: true })

  let nl = buf.indexOf('\n')
  while (nl >= 0) {
    const line = buf.slice(0, nl)
    buf = buf.slice(nl + 1)
    if (line) {
      const ev = JSON.parse(line) as
        | { meta:    { language: string, duration: number } }
        | { segment: { start: number, end: number, text: string } }
        | { done:    { text: string } }
        | { error:   string }
      if ('segment' in ev) {
        parts.push(ev.segment.text)
        // emit partial transcript to UI
      }
      else if ('done' in ev) {
        // final transcript = ev.done.text
      }
      else if ('error' in ev) {
        throw new Error(ev.error)
      }
    }
    nl = buf.indexOf('\n')
  }
}
```

(`http://whisper:...` works from another container on the same docker network.)

## Performance notes

- CPU inference of `medium` runs ~1× realtime on a 2-vCPU box. `large-v3` is ~0.3-0.5×.
- VAD pre-filtering speeds up real-world audio (with silences) substantially.
- A GPU drops latency 5-10×. Set `WHISPER_DEVICE=cuda` and use a CUDA-enabled
  base image (not in this Dockerfile).
- The service uses one uvicorn worker on purpose. Multiple workers would each
  load their own copy of the model.
