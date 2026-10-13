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
| DELETE | `/v1/voices/{id}` | Remove a cloned or LoRA voice (403 for built-in voices, 404 if unknown) |
| GET, POST | `/v1/lora/sessions` | List LoRA sessions, or start one: `{"name","gender","description","mode"}`. `mode` is `record` (read the server's sentences, the default) or `import` (bring clips and transcripts) |
| GET, DELETE | `/v1/lora/sessions/{id}` | A session with its sentences, clips, `status` and training `progress`; delete it with its clips and model |
| PUT, GET, DELETE | `/v1/lora/sessions/{id}/clips/{i}` | Save (or retake) the recording of sentence `i` (multipart `file`, WAV/FLAC/OGG, 1-20 s), fetch it, or drop it (in import mode its transcript goes too) |
| POST | `/v1/lora/sessions/{id}/clips` | Import mode: add a clip, multipart `file` + `text` (its transcript) |
| PATCH | `/v1/lora/sessions/{id}/sentences/{i}` | Import mode: `{"text"}` fixes a transcript |
| POST | `/v1/lora/sessions/{id}/train` | `{"denoise"}`: queue training once every sentence is recorded (import mode: at least 5 clips) |
| POST | `/v1/audio/speech` | `{"model","voice","input","response_format","stream_format","sample_rate","device"}` plus sampling fields (below) returns audio; `X-Device` says which device ran it |
| GET | `/v1/models` | Model info: `backend`, `max_streams`, `response_formats`, `stream_formats`, `sample_rates`, devices |
| GET | `/` | Browser test page: pick a voice, type text, play it streamed (pcm) or as a WAV file |
| GET | `/lora` | LoRA page: record the sentences one by one (retake any) or import clips with transcripts, then train and try the voice |
| GET | `/guide` | API guide: endpoints, request fields and copyable curl / Python / JavaScript samples |
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

## LoRA voices

A LoRA voice is the model fine-tuned on one speaker, which tracks the voice more closely than a clone. On the `/lora` page the speaker reads 14 short sentences (about a minute). Each sentence is its own clip, so a mistake only costs one retake. Then they press Start training. The **Import files** tab takes existing recordings instead: audio files of one sentence each (1-20 s) and their transcripts, either one `.txt` per audio file with the same name, one text file with a line per audio file, or a `metadata.csv`-style `name|text` file. The page pairs them, refuses mismatched counts, and shows the pairs for checking before upload. Training follows VieNeu-TTS's [`finetune/`](https://github.com/pnnbao97/VieNeu-TTS/tree/main/finetune) pipeline: each clip is denoised and trimmed, LoRA is trained on the backbone, and the merged model is saved with the voice's speaker embedding. It runs in a separate process, one job at a time, and takes about a minute on an RTX 5060. When it finishes, the voice is listed in `/v1/voices` with `"lora": true` and is used by name like any other voice.

- LoRA voices need a CUDA GPU, for both training and speech. A request for one with `"device": "cpu"` returns 400.
- Each voice is a full model (about 250 MB on disk). It loads on the GPU on first use; `TTS_LORA_MAX_LOADED` (default 1) caps how many stay loaded.
- Sessions, clips and models live in `TTS_LORA_DIR` (default `./data/lora`). Back it up with `data/voices`. If the server restarts during training, that session is marked failed and can be trained again.
- Training settings: `TTS_LORA_EPOCHS`, `TTS_LORA_LR`, `TTS_LORA_RANK`, `TTS_LORA_BATCH` (see `.env.example`). By default the number of epochs follows the data: 25 for 14 clips, down to upstream's 3 for 100 clips or more. Upstream recommends 10-30 minutes of audio for best results; one minute is the quick start.
