"""
VieNeu-TTS v3 Turbo speech server (OpenAI-style), per tts-api.txt.

    GET    /v1/voices        -> {"object": "list", "default": "<id>",
                                 "data": [{"id", "name", "gender", "description", "featured", "aliases", "custom"}]}
    POST   /v1/voices        -> multipart {name, file (3-8 s clip), denoise, description, gender}: clone a voice
    DELETE /v1/voices/{id}   -> remove a cloned voice (presets cannot be removed)
    POST   /v1/audio/speech  -> {"model", "voice", "input", "response_format": "wav" | "pcm",
                                 "stream_format": "audio" | "sse", "sample_rate": 48000 | 24000 | 16000 | 8000,
                                 "temperature", "top_k", "top_p", "repetition_penalty", "max_chars",
                                 "speed", "instructions" (ignored), "device": "gpu" | "cpu"}
         wav: one complete file with correct RIFF/data sizes
         pcm: streamed s16le mono, rate in the X-Sample-Rate header
         sse: text/event-stream of {"type": "speech.audio.delta", "audio": <base64>} then
              {"type": "speech.audio.done", "usage": {"output_samples", "sample_rate", "seconds"}}
    Cloned voices are saved under TTS_VOICES_DIR and reloaded at startup.
    GET  /v1/models, GET /health, GET / (browser test page)

Auth: ``Authorization: Bearer $TTS_API_KEY`` when TTS_API_KEY is set.
Run:  uv run python -m app.main
"""
from __future__ import annotations

import base64
import hmac
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
import unicodedata
import uuid
import wave
import weakref
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tts-server")

MODEL_ID = "vieneu-v3-turbo"
SAMPLE_RATE = 48_000


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader; real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# TTS_GPU picks the CUDA GPU by index (as nvidia-smi lists them). It must be applied before
# torch is imported; the chosen GPU then appears to torch/VieNeu as cuda:0.
if os.environ.get("TTS_GPU", "").strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["TTS_GPU"].strip()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # match nvidia-smi numbering


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


# ── engines ──────────────────────────────────────────────────────────────────

def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


class Engine:
    """One loaded model on one device: ``kind`` is "gpu" (PyTorch/CUDA) or "cpu" (ONNX Runtime)."""

    def __init__(self, kind: str) -> None:
        from vieneu import Vieneu

        self.kind = kind
        concurrency = max(1, _env_int("TTS_MAX_CONCURRENCY", 4))
        log.info("loading VieNeu-TTS v3 Turbo on %s", kind.upper())
        t = time.perf_counter()
        if kind == "gpu":
            import torch

            self.tts = Vieneu(mode="v3turbo", backend="pytorch", device="cuda", max_streams=concurrency)
            # Inside this process the chosen GPU is always cuda:0; show nvidia-smi's index instead.
            index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
            self.device = f"GPU {index}: {torch.cuda.get_device_name(0)}"
            self.max_concurrency = concurrency
        else:
            self.tts = Vieneu(mode="v3turbo", backend="onnx", device="cpu",
                              precision=os.environ.get("VIENEU_PRECISION") or "fp32")
            self.device = "CPU (ONNX)"
            # CPU streams share the cores: one at a time keeps it real-time.
            self.max_concurrency = 1
        self.backend = self.tts.backend
        self.queue_timeout = float(os.environ.get("TTS_QUEUE_TIMEOUT") or 30)
        self._gate = threading.BoundedSemaphore(self.max_concurrency)
        # Reported by /health ("active", "waiting"), as the old build did.
        self._count_lock = threading.Lock()
        self.active = 0
        self.waiting = 0

        # Warm-up so the first client does not pay for graph capture / ONNX load.
        for _ in self.tts.infer_stream("Xin chào.", apply_watermark=False):
            pass
        log.info("%s ready in %.1fs: device=%s concurrency=%d", kind.upper(),
                 time.perf_counter() - t, self.device, self.max_concurrency)

    def acquire(self):
        """Take a synthesis slot (429 after the queue timeout); returns its release-once callable."""
        with self._count_lock:
            self.waiting += 1
        try:
            got = self._gate.acquire(timeout=self.queue_timeout)
        finally:
            with self._count_lock:
                self.waiting -= 1
        if not got:
            raise HTTPException(429, "server busy, try again later", headers={"Retry-After": "2"})
        with self._count_lock:
            self.active += 1
        lock, held = threading.Lock(), [True]

        def release() -> None:
            with lock:
                was, held[0] = held[0], False
            if was:
                with self._count_lock:
                    self.active -= 1
                self._gate.release()
        return release

    def stream(self, text: str, voice: Optional[str], rid: str, release,
               sampling: Optional[dict] = None) -> Iterator[np.ndarray]:
        """48 kHz float32 chunks; calls ``release`` (from ``acquire()``) when done.

        ``sampling``: temperature / top_k / top_p / repetition_penalty / max_chars for ``infer_stream``.
        """
        t0 = time.perf_counter()
        first: Optional[float] = None
        samples = 0
        try:
            for chunk in self.tts.infer_stream(text, voice=voice, **(sampling or {})):
                if chunk is None or len(chunk) == 0:
                    continue
                if first is None:
                    first = time.perf_counter() - t0
                samples += len(chunk)
                yield chunk
        finally:
            release()
            total = time.perf_counter() - t0
            log.info("%s done on %s: chars=%d first_audio=%s total=%.2fs audio=%.2fs", rid, self.kind,
                     len(text), f"{first * 1000:.0f}ms" if first is not None else "-", total,
                     samples / SAMPLE_RATE)


CLONE_MIN_SECONDS = 3.0
CLONE_MAX_SECONDS = 8.0
CLONE_MAX_BYTES = 5 * 1024 * 1024
# Letters (any script), digits, spaces and - _ . ( ); the name is the voice id clients send back.
_VOICE_NAME_RE = re.compile(r"^[\w .()\-]{1,40}$")


class VoiceStore:
    """Cloned voices kept on disk so they survive restarts.

    ``<dir>/custom_voices.json`` holds each voice's speaker embedding and reference codes (what
    synthesis needs, so loading is instant); ``<dir>/clips/<file>.wav`` keeps the trimmed
    reference clip, so a voice can be re-enrolled if the model changes.
    """

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.clips = directory / "clips"
        self.path = directory / "custom_voices.json"
        self._lock = threading.Lock()
        self.voices: dict[str, dict] = {}
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for name, v in (data.get("voices") or {}).items():
                    if v.get("speaker_emb") is None:
                        continue
                    self.voices[name] = v
            except Exception as e:  # noqa: BLE001
                log.error("cannot read %s, starting without cloned voices: %s", self.path, e)
        log.info("loaded %d cloned voice(s) from %s", len(self.voices), self.path)

    @staticmethod
    def preset_entry(v: dict) -> dict:
        """The in-memory form VieNeu's ``_preset_voices`` expects (same as its own JSON loader)."""
        from vieneu_utils.core_utils import strip_encoder_pad_frame

        codes = v.get("codes")
        return {
            "description": v.get("description", ""),
            "gender": v.get("gender", ""),
            "style": "tu_nhien",
            "featured": None,
            "aliases": [],
            "custom": True,
            "speaker_emb": np.asarray(v["speaker_emb"], dtype=np.float32),
            "codes": None if codes is None else strip_encoder_pad_frame(np.asarray(codes, dtype=np.int64)),
        }

    def install(self, eng: "Engine") -> None:
        """Register every stored voice on an engine (both devices share the same embeddings)."""
        for name, v in self.voices.items():
            eng.tts._preset_voices[name] = self.preset_entry(v)

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="custom_voices_", suffix=".json", dir=self.dir)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"voices": self.voices}, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def add(self, name: str, entry: dict, clip_wav: bytes) -> None:
        with self._lock:
            self.clips.mkdir(parents=True, exist_ok=True)
            clip_file = f"{uuid.uuid4().hex}.wav"
            (self.clips / clip_file).write_bytes(clip_wav)
            emb, codes = entry["speaker_emb"], entry.get("codes")
            self.voices[name] = {
                "description": entry.get("description", ""),
                "gender": entry.get("gender", ""),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "clip": clip_file,
                "speaker_emb": [round(float(x), 6) for x in np.asarray(emb).reshape(-1)],
                "codes": None if codes is None else np.asarray(codes, dtype=int).tolist(),
            }
            self._save()

    def remove(self, name: str) -> bool:
        with self._lock:
            v = self.voices.pop(name, None)
            if v is None:
                return False
            self._save()
        if v.get("clip"):
            try:
                (self.clips / v["clip"]).unlink(missing_ok=True)
            except OSError:
                pass
        return True


class Engines:
    """The default engine, loaded at startup, plus the other device, loaded on first request."""

    def __init__(self) -> None:
        self.gpu_available = _cuda_available()
        want = (os.environ.get("TTS_DEFAULT_DEVICE") or "auto").lower()
        if want == "gpu" and not self.gpu_available:
            log.warning("TTS_DEFAULT_DEVICE=gpu but no CUDA GPU is available; using CPU")
        self.default = "gpu" if want in ("auto", "gpu") and self.gpu_available else "cpu"
        self._loaded: dict[str, Engine] = {}
        self._loading: set[str] = set()
        self._lock = threading.Lock()
        voices_dir = Path(os.environ.get("TTS_VOICES_DIR") or Path(__file__).resolve().parent.parent / "data" / "voices")
        self.store = VoiceStore(voices_dir)
        primary = self.get(self.default)

        default = os.environ.get("TTS_DEFAULT_VOICE") or None
        if default and primary.tts.resolve_voice_name(default) is None:
            log.warning("TTS_DEFAULT_VOICE '%s' is unknown; using the model default", default)
            default = None
        self.default_voice = primary.tts.resolve_voice_name(default or primary.tts._default_voice)
        log.info("default device=%s default_voice=%s", self.default, self.default_voice)
        if os.environ.get("TTS_PRELOAD_ALL") == "1":
            other = "cpu" if self.default == "gpu" else "gpu"
            if other == "cpu" or self.gpu_available:
                self.get(other)

    def get(self, kind: Optional[str] = None) -> Engine:
        """The engine for ``kind`` ("gpu" | "cpu" | None = default), loading it on first use."""
        kind = (kind or self.default).lower()
        if kind not in ("gpu", "cpu"):
            raise HTTPException(400, "device must be 'gpu' or 'cpu'")
        if kind == "gpu" and not self.gpu_available:
            raise HTTPException(400, "no CUDA GPU available on this server; use device 'cpu'")
        eng = self._loaded.get(kind)
        if eng is not None:
            return eng
        with self._lock:  # one load at a time; other requests for that device wait for it
            if kind not in self._loaded:
                self._loading.add(kind)
                try:
                    eng = Engine(kind)
                    self.store.install(eng)
                    self._loaded[kind] = eng
                finally:
                    self._loading.discard(kind)
            return self._loaded[kind]

    def primary(self) -> Engine:
        return self._loaded[self.default]

    def voices(self) -> list[dict]:
        from vieneu.v3turbo import sorted_voices

        return [{"id": name, "name": name, "gender": v.get("gender", ""),
                 "description": v.get("description", ""), "featured": v.get("featured"),
                 "aliases": list(v.get("aliases") or []), "custom": bool(v.get("custom"))}
                for name, v in sorted_voices(self.primary().tts._preset_voices)]

    def clone(self, name: str, clip_path: Path, clip_wav: bytes, denoise: bool,
              description: str, gender: str) -> dict:
        """Enroll ``clip_path`` once on the primary engine, persist it, then register it on every loaded engine."""
        primary = self.primary()
        release = primary.acquire()  # enrolment runs the model: take a synthesis slot like a request
        try:
            primary.tts.add_voice(name, str(clip_path), denoise=denoise, description=description,
                                  gender=gender, save=False)
        finally:
            release()
        entry = primary.tts._preset_voices[name]
        self.store.add(name, {**entry, "description": description, "gender": gender}, clip_wav)
        stored = VoiceStore.preset_entry(self.store.voices[name])
        for eng in list(self._loaded.values()):
            eng.tts._preset_voices[name] = dict(stored)
        return {"id": name, "name": name, "gender": gender, "description": description,
                "featured": None, "aliases": [], "custom": True}

    def remove(self, name: str) -> None:
        if not self.store.remove(name):
            raise HTTPException(404, f"no cloned voice '{name}'")
        for eng in list(self._loaded.values()):
            eng.tts._preset_voices.pop(name, None)
            if eng.tts._default_voice == name:
                eng.tts._default_voice = next(iter(eng.tts._preset_voices), None)
        if self.default_voice == name:  # TTS_DEFAULT_VOICE pointed at the removed clone
            self.default_voice = self.primary().tts._default_voice

    def status(self) -> dict:
        def one(kind: str) -> dict:
            eng = self._loaded.get(kind)
            state = "loaded" if eng else "loading" if kind in self._loading else "not loaded"
            return {"available": kind == "cpu" or self.gpu_available, "state": state,
                    "name": eng.device if eng else ("CPU (ONNX)" if kind == "cpu" else "GPU (CUDA)"),
                    "max_concurrency": eng.max_concurrency if eng else None,
                    "active": eng.active if eng else 0, "waiting": eng.waiting if eng else 0}
        return {"gpu": one("gpu"), "cpu": one("cpu")}


ENGINES: Optional[Engines] = None


def engines() -> Engines:
    if ENGINES is None:
        raise HTTPException(503, "speech engine is not loaded")
    return ENGINES


# ── app ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="VieNeu-TTS server", version="0.1.0")

_origins = [o.strip() for o in (os.environ.get("ALLOWED_ORIGINS") or "").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["Authorization", "Content-Type"],
                       expose_headers=["X-Sample-Rate", "X-Request-Id"])


@app.on_event("startup")
def _startup() -> None:
    global ENGINES
    ENGINES = Engines()


def _auth(authorization: Optional[str] = Header(default=None)) -> None:
    key = os.environ.get("TTS_API_KEY")
    if key and not hmac.compare_digest((authorization or "").encode(), f"Bearer {key}".encode()):
        raise HTTPException(401, "invalid api key")


def _error(status: int, message: str, headers=None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error", "code": status}},
                        status_code=status, headers=headers)


@app.exception_handler(StarletteHTTPException)
async def _http_exc(_req: Request, exc: StarletteHTTPException):
    return _error(exc.status_code, str(exc.detail), headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def _invalid_request(_req: Request, exc: RequestValidationError):
    errs = exc.errors()
    first = errs[0] if errs else {}
    field = ".".join(str(p) for p in first.get("loc", ()) if p not in ("body", "query", "header"))
    msg = first.get("msg", "invalid request")
    return _error(400, f"{field}: {msg}" if field else msg)


# ── audio helpers ────────────────────────────────────────────────────────────

def _pcm16(x: np.ndarray) -> bytes:
    return (np.asarray(x, dtype=np.float32) * 32767).clip(-32768, 32767).astype("<i2").tobytes()


def _wav_bytes(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _streaming_wav_header(rate: int) -> bytes:
    """WAV header with unknown sizes (0xFFFFFFFF), for a WAV sent in pieces before its length is known."""
    import struct

    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", 0xFFFFFFFF))


def _resampled(chunks: Iterator[np.ndarray], rate: int) -> Iterator[np.ndarray]:
    """``chunks`` (48 kHz) at ``rate``; one stateful resampler so chunk edges do not click."""
    if rate == SAMPLE_RATE:
        yield from chunks
        return
    import soxr

    rs = soxr.ResampleStream(SAMPLE_RATE, rate, 1, dtype="float32", quality="HQ")
    for c in chunks:
        out = rs.resample_chunk(np.asarray(c, dtype=np.float32))
        if len(out):
            yield out
    tail = rs.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
    if len(tail):
        yield tail


def _sse(event: dict) -> bytes:
    # json.dumps default separators (", " / ": "): the exact bytes the old build sent.
    return f"data: {json.dumps(event)}\n\n".encode()


# ── routes ───────────────────────────────────────────────────────────────────

SAMPLE_RATES = (48000, 24000, 16000, 8000)


class SpeechRequest(BaseModel):
    """Same body as the old build (OpenAI fields + VieNeu sampling), plus ``device``."""
    model: str = MODEL_ID  # any value accepted (one model)
    input: str = Field(min_length=1, max_length=20_000)
    voice: Optional[str] = None
    response_format: str = "wav"  # "wav" | "pcm"
    stream_format: str = "audio"  # "audio" = raw bytes | "sse" = OpenAI speech.audio.delta/done events
    speed: Optional[float] = None  # accepted for OpenAI compatibility, ignored
    instructions: Optional[str] = None  # accepted for OpenAI compatibility, ignored
    sample_rate: int = SAMPLE_RATE
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(25, ge=1, le=1024)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.2, ge=1.0, le=2.0)
    max_chars: int = Field(256, ge=64, le=512)  # longest text chunk synthesized in one pass
    device: Optional[str] = None  # "gpu" | "cpu"; omitted = the server default


@app.post("/v1/audio/speech", dependencies=[Depends(_auth)])
def speech(req: SpeechRequest):
    pool = engines()
    fmt = req.response_format.lower()
    if fmt not in ("wav", "pcm"):
        raise HTTPException(400, f"response_format '{req.response_format}' not supported; use 'pcm' or 'wav'")
    stream_format = req.stream_format.lower()
    if stream_format not in ("audio", "sse"):
        raise HTTPException(400, "stream_format must be 'audio' or 'sse'")
    rate = req.sample_rate
    if rate not in SAMPLE_RATES:
        raise HTTPException(400, f"sample_rate must be one of {SAMPLE_RATES}")
    voice = pool.default_voice
    if req.voice:
        voice = pool.primary().tts.resolve_voice_name(req.voice)
        if voice is None:
            raise HTTPException(400, f"unknown voice '{req.voice}'; see GET /v1/voices")

    eng = pool.get(req.device)  # the first request for the other device loads it (can take a while)

    rid = f"tts-{uuid.uuid4().hex[:8]}"
    headers = {"X-Request-Id": rid, "X-Sample-Rate": str(rate), "X-Device": eng.kind,
               "Cache-Control": "no-store"}
    if not req.input.strip():
        # The old build answered blank input with empty audio, not an error; keep that.
        chunks: Iterator[np.ndarray] = iter(())
        release = None
    else:
        release = eng.acquire()
        sampling = {"temperature": req.temperature, "top_k": req.top_k, "top_p": req.top_p,
                    "repetition_penalty": req.repetition_penalty, "max_chars": req.max_chars}
        raw = eng.stream(req.input, voice, rid, release, sampling)
        # A generator that is never iterated (client left early) never runs its
        # finally: free the slot when the body is garbage-collected instead.
        weakref.finalize(raw, release)
        chunks = _resampled(raw, rate)

    if stream_format == "sse":
        def events() -> Iterator[bytes]:
            samples = 0
            if fmt == "wav":
                yield _sse({"type": "speech.audio.delta", "audio": base64.b64encode(_streaming_wav_header(rate)).decode()})
            for c in chunks:
                samples += len(c)
                yield _sse({"type": "speech.audio.delta", "audio": base64.b64encode(_pcm16(c)).decode()})
            yield _sse({"type": "speech.audio.done", "usage": {
                "output_samples": samples, "sample_rate": rate, "seconds": round(samples / rate, 2)}})
        return StreamingResponse(events(), media_type="text/event-stream; charset=utf-8", headers=headers)

    if fmt == "pcm":
        return StreamingResponse((_pcm16(c) for c in chunks), media_type="audio/pcm", headers=headers)

    try:
        pcm = b"".join(_pcm16(c) for c in chunks)
    except Exception as e:  # noqa: BLE001
        log.exception("%s synthesis failed", rid)
        raise HTTPException(500, f"synthesis failed: {e}")
    finally:
        if release:
            release()
    return Response(_wav_bytes(pcm, rate), media_type="audio/wav", headers=headers)


@app.get("/v1/voices", dependencies=[Depends(_auth)])
def voices():
    pool = engines()
    return {"object": "list", "default": pool.default_voice, "data": pool.voices()}


def _read_clip(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode an uploaded clip (WAV/FLAC/OGG; compressed browser formats are converted to WAV client-side)."""
    import soundfile as sf

    try:
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "cannot read the audio file; send WAV, FLAC or OGG")
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    return np.asarray(wav, dtype=np.float32), int(sr)


# Plain ``def``: FastAPI runs it in a thread, so reading the upload and enrolling do not stall the event loop.
@app.post("/v1/voices", dependencies=[Depends(_auth)])
def add_voice(name: str = Form(...), file: UploadFile = File(...), denoise: bool = Form(True),
              description: str = Form(""), gender: str = Form("")):
    pool = engines()
    name = re.sub(r"\s+", " ", unicodedata.normalize("NFC", name)).strip()
    if not _VOICE_NAME_RE.fullmatch(name):
        raise HTTPException(400, "voice name must be 1-40 letters, digits, spaces or - _ . ( )")
    if pool.primary().tts.resolve_voice_name(name) is not None:
        raise HTTPException(409, f"a voice named '{name}' already exists")
    raw = file.file.read(CLONE_MAX_BYTES + 1)
    if not raw:
        raise HTTPException(400, "the audio file is empty")
    if len(raw) > CLONE_MAX_BYTES:
        raise HTTPException(413, f"the audio file is larger than {CLONE_MAX_BYTES // (1024 * 1024)} MB")
    wav, sr = _read_clip(raw)
    seconds = len(wav) / sr if sr else 0
    if seconds < CLONE_MIN_SECONDS:
        raise HTTPException(400, f"the clip is {seconds:.1f} s; it needs at least {CLONE_MIN_SECONDS:.0f} s of speech")
    wav = wav[: int(CLONE_MAX_SECONDS * sr)]
    clip = _wav_bytes(_pcm16(wav), sr)

    fd, tmp = tempfile.mkstemp(prefix="clone_", suffix=".wav")
    os.close(fd)
    try:
        Path(tmp).write_bytes(clip)
        try:
            voice = pool.clone(name, Path(tmp), clip, denoise, description.strip()[:200], gender.strip()[:20])
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("cloning voice '%s' failed", name)
            raise HTTPException(500, f"voice enrolment failed: {e}")
    finally:
        Path(tmp).unlink(missing_ok=True)
    log.info("cloned voice '%s' from %.1fs clip (denoise=%s)", name, min(seconds, CLONE_MAX_SECONDS), denoise)
    return voice


@app.delete("/v1/voices/{name}", dependencies=[Depends(_auth)])
def delete_voice(name: str):
    pool = engines()
    name = unicodedata.normalize("NFC", name)
    if name not in pool.store.voices:
        if pool.primary().tts.resolve_voice_name(name) is not None:
            raise HTTPException(403, f"'{name}' is a built-in voice and cannot be removed")
        raise HTTPException(404, f"no cloned voice '{name}'")
    pool.remove(name)
    log.info("removed cloned voice '%s'", name)
    return {"id": name, "deleted": True}


@app.get("/v1/models", dependencies=[Depends(_auth)])
def models():
    pool = engines()
    primary = pool.primary()
    return {"object": "list", "data": [{
        "id": MODEL_ID, "object": "model", "created": 0, "owned_by": "vieneu",
        "sample_rate": SAMPLE_RATE, "backend": primary.backend, "max_streams": primary.max_concurrency,
        "response_formats": ["pcm", "wav"], "stream_formats": ["audio", "sse"],
        "sample_rates": list(SAMPLE_RATES),
        "default_device": pool.default, "devices": pool.status(),
    }]}


STATIC_DIR = Path(__file__).parent / "static"


class _RevalidatedStatic(StaticFiles):
    """Static files the browser must revalidate (cheap 304 via ETag), so edits show up on a normal reload."""

    async def get_response(self, path, scope):
        res = await super().get_response(path, scope)
        res.headers["Cache-Control"] = "no-cache"
        return res


# Browser test page and its assets; the page asks for the API key itself, so it needs no auth.
app.mount("/static", _RevalidatedStatic(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def test_page():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/health")
def health():
    if ENGINES is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    primary = ENGINES.primary()
    # backend / max_streams / active / waiting: the old build's fields, for the default device.
    return {"status": "ok", "backend": primary.backend, "max_streams": primary.max_concurrency,
            "active": primary.active, "waiting": primary.waiting, "sample_rate": SAMPLE_RATE,
            "device": primary.device, "default_device": ENGINES.default, "devices": ENGINES.status()}


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST") or "127.0.0.1"
    port = _env_int("PORT", 8000)
    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("TTS_API_KEY"):
        log.warning("listening on %s:%d without TTS_API_KEY: the API is open to anyone on the network",
                    host, port)
    # One worker: the model lives in this process.
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
