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
| GET | `/v1/voices` | `{"object":"list","default":"<id>","data":[{"id","name","gender","description"}]}` |
| POST | `/v1/audio/speech` | `{"model","voice","input","response_format","device"}` returns audio; `X-Device` says which device ran it |
| GET | `/v1/models` | Model info |
| GET | `/` | Browser test page: pick a voice, type text, play it streamed (pcm) or as a WAV file |
| GET | `/health` | Liveness check (returns 503 while the model loads) |

`response_format`:
- `wav` (the default): one complete WAV file with correct RIFF/data sizes.
- `pcm`: a stream of 16-bit little-endian mono samples. The sample rate is in the `X-Sample-Rate` header (48000).

If you leave out `voice`, the server uses the default voice (`TTS_DEFAULT_VOICE`, or the model's own default). The server does not rewrite numbers, dates or times as words, so write them out (for example "mười giờ" instead of "10:00").

```sh
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Authorization: Bearer $TTS_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"vieneu-v3-turbo","voice":"Mai Anh","input":"Xin chào, đây là thông báo thử.","response_format":"wav"}' \
  --output out.wav
```

Errors use OpenAI's shape: `{"error":{"message","type","code"}}`. The server returns 400 for bad input or an unknown voice, 401 for a bad key, and 429 when it is busy.
