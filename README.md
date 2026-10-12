# vieneu-tts-server

Text-to-speech HTTP server built on [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) v3 Turbo (48 kHz).
It exposes an OpenAI-style `/v1/audio/speech` endpoint, following `tts-api.txt`.

## Run

```sh
uv sync
cp .env.example .env      # optional: set TTS_API_KEY, HOST, PORT...
uv run python -m app.main # http://127.0.0.1:8000
```

The model weights download from Hugging Face on the first start. The server can run the model on the GPU (PyTorch/CUDA) or the CPU (ONNX Runtime). `TTS_DEFAULT_DEVICE` (auto, gpu or cpu) picks the device loaded at startup. A request can ask for the other device with `"device": "gpu"` or `"device": "cpu"`; that model loads on its first use and then stays loaded (`TTS_PRELOAD_ALL=1` loads both at startup).

## API

Requests need `Authorization: Bearer <TTS_API_KEY>` when `TTS_API_KEY` is set.

| Method | Path | Description |
|-|-|-|
| GET | `/v1/voices` | `{"object":"list","default":"<id>","data":[{"id","name","gender","description","featured","aliases","custom"}]}` |
| POST | `/v1/voices` | Clone a voice. Multipart: `name`, `file` (3-8 s of speech, WAV/FLAC/OGG, max 5 MB; longer clips are cut to 8 s), `denoise` (default true), `description`, `gender`. Returns the new voice; 409 if the name is taken |
| DELETE | `/v1/voices/{id}` | Remove a cloned voice (403 for built-in voices, 404 if unknown) |
| POST | `/v1/audio/speech` | `{"model","voice","input","response_format","stream_format","sample_rate","device"}` plus sampling fields (below) returns audio; `X-Device` says which device ran it |
| GET | `/v1/models` | Model info: `backend`, `max_streams`, `response_formats`, `stream_formats`, `sample_rates`, devices |
| GET | `/` | Browser test page: pick a voice, type text, play it streamed (pcm) or as a WAV file |
| GET | `/health` | Liveness check with `backend`, `max_streams`, `active`, `waiting`, `sample_rate` (returns 503 while the model loads) |

`response_format`:
- `wav` (the default): one complete WAV file with correct RIFF/data sizes.
- `pcm`: a stream of 16-bit little-endian mono samples. The sample rate is in the `X-Sample-Rate` header.

Other `/v1/audio/speech` fields (all optional):

| Field | Default | Meaning |
|-|-|-|
| `stream_format` | `audio` | `audio` sends raw bytes. `sse` sends `text/event-stream` events `{"type":"speech.audio.delta","audio":"<base64>"}`, then `{"type":"speech.audio.done","usage":{"output_samples","sample_rate","seconds"}}`. With `wav`, the first delta is a WAV header with unknown sizes |
| `sample_rate` | `48000` | `48000`, `24000`, `16000` or `8000`; lower rates are resampled on the server |
| `temperature` | `0.8` | 0 to 2 |
| `top_k` | `25` | 1 to 1024 |
| `top_p` | `0.95` | above 0, up to 1 |
| `repetition_penalty` | `1.2` | 1 to 2 |
| `max_chars` | `256` | 64 to 512; the longest text chunk generated in one pass |
| `speed`, `instructions` | - | Accepted for OpenAI compatibility and ignored |

Blank `input` (only spaces) returns empty audio rather than an error.

If you leave out `voice`, the server uses the default voice (`TTS_DEFAULT_VOICE`, or the model's own default). The server does not rewrite numbers, dates or times as words, so write them out (for example "mười giờ" instead of "10:00").

```sh
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Authorization: Bearer $TTS_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"vieneu-v3-turbo","voice":"Mai Anh","input":"Xin chào, đây là thông báo thử.","response_format":"wav"}' \
  --output out.wav
```

Cloned voices are saved in `TTS_VOICES_DIR` (default `./data/voices`): `custom_voices.json` holds each voice's speaker embedding and reference codes, and `clips/` keeps the trimmed reference clip. They load again at startup, on both devices. Back up that folder when you redeploy.

Errors use OpenAI's shape: `{"error":{"message","type","code"}}`. The server returns 400 for bad input or an unknown voice, 401 for a bad key, and 429 when it is busy.
