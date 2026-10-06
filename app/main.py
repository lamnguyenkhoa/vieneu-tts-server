"""
VieNeu-TTS v3 Turbo speech server (OpenAI-style), per tts-api.txt.

    GET  /v1/voices          -> {"object": "list", "default": "<id>", "data": [{"id", "name", "gender", "description"}]}
    POST /v1/audio/speech    -> {"model", "voice", "input", "response_format": "wav" | "pcm", "device": "gpu" | "cpu"}
         wav: one complete file with correct RIFF/data sizes
         pcm: streamed s16le mono, rate in the X-Sample-Rate header (48000)
    GET  /v1/models, GET /health, GET / (browser test page)

Auth: ``Authorization: Bearer $TTS_API_KEY`` when TTS_API_KEY is set.
Run:  uv run python -m app.main
"""
from __future__ import annotations

import hmac
import io
import logging
import os
import threading
import time
import uuid
import wave
import weakref
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
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
            self.device = f"GPU: {torch.cuda.get_device_name(0)}"
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

        # Warm-up so the first client does not pay for graph capture / ONNX load.
        for _ in self.tts.infer_stream("Xin chào.", apply_watermark=False):
            pass
        log.info("%s ready in %.1fs: device=%s concurrency=%d", kind.upper(),
                 time.perf_counter() - t, self.device, self.max_concurrency)

    def acquire(self):
        """Take a synthesis slot (429 after the queue timeout); returns its release-once callable."""
        if not self._gate.acquire(timeout=self.queue_timeout):
            raise HTTPException(429, "server busy, try again later", headers={"Retry-After": "2"})
        lock, held = threading.Lock(), [True]

        def release() -> None:
            with lock:
                was, held[0] = held[0], False
            if was:
                self._gate.release()
        return release

    def stream(self, text: str, voice: Optional[str], rid: str, release) -> Iterator[np.ndarray]:
        """48 kHz float32 chunks; calls ``release`` (from ``acquire()``) when done."""
        t0 = time.perf_counter()
        first: Optional[float] = None
        samples = 0
        try:
            for chunk in self.tts.infer_stream(text, voice=voice):
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
                    self._loaded[kind] = Engine(kind)
                finally:
                    self._loading.discard(kind)
            return self._loaded[kind]

    def primary(self) -> Engine:
        return self._loaded[self.default]

    def voices(self) -> list[dict]:
        from vieneu.v3turbo import sorted_voices

        return [{"id": name, "name": name, "gender": v.get("gender", ""),
                 "description": v.get("description", "")}
                for name, v in sorted_voices(self.primary().tts._preset_voices)]

    def status(self) -> dict:
        def one(kind: str) -> dict:
            eng = self._loaded.get(kind)
            state = "loaded" if eng else "loading" if kind in self._loading else "not loaded"
            return {"available": kind == "cpu" or self.gpu_available, "state": state,
                    "name": eng.device if eng else ("CPU (ONNX)" if kind == "cpu" else "GPU (CUDA)"),
                    "max_concurrency": eng.max_concurrency if eng else None}
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
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"],
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


# ── routes ───────────────────────────────────────────────────────────────────

class SpeechRequest(BaseModel):
    model: str = MODEL_ID
    input: str = Field(min_length=1, max_length=20_000)
    voice: Optional[str] = None
    response_format: str = "wav"
    speed: Optional[float] = None  # accepted for OpenAI compatibility, ignored
    device: Optional[str] = None  # "gpu" | "cpu"; omitted = the server default


@app.post("/v1/audio/speech", dependencies=[Depends(_auth)])
def speech(req: SpeechRequest):
    pool = engines()
    fmt = req.response_format.lower()
    if fmt not in ("wav", "pcm"):
        raise HTTPException(400, f"response_format '{req.response_format}' not supported; use 'wav' or 'pcm'")
    if not req.input.strip():
        raise HTTPException(400, "input is empty")
    voice = pool.default_voice
    if req.voice:
        voice = pool.primary().tts.resolve_voice_name(req.voice)
        if voice is None:
            raise HTTPException(400, f"unknown voice '{req.voice}'; see GET /v1/voices")

    eng = pool.get(req.device)  # the first request for the other device loads it (can take a while)

    rid = f"tts-{uuid.uuid4().hex[:8]}"
    headers = {"X-Request-Id": rid, "X-Sample-Rate": str(SAMPLE_RATE), "X-Device": eng.kind,
               "Cache-Control": "no-store"}
    release = eng.acquire()
    chunks = eng.stream(req.input, voice, rid, release)

    if fmt == "pcm":
        # A generator that is never iterated (client left early) never runs its
        # finally: free the slot when the body is garbage-collected instead.
        weakref.finalize(chunks, release)
        return StreamingResponse((_pcm16(c) for c in chunks), media_type="audio/pcm", headers=headers)

    try:
        pcm = b"".join(_pcm16(c) for c in chunks)
    except Exception as e:  # noqa: BLE001
        log.exception("%s synthesis failed", rid)
        raise HTTPException(500, f"synthesis failed: {e}")
    return Response(_wav_bytes(pcm, SAMPLE_RATE), media_type="audio/wav", headers=headers)


@app.get("/v1/voices", dependencies=[Depends(_auth)])
def voices():
    pool = engines()
    return {"object": "list", "default": pool.default_voice, "data": pool.voices()}


@app.get("/v1/models", dependencies=[Depends(_auth)])
def models():
    pool = engines()
    return {"object": "list", "data": [{
        "id": MODEL_ID, "object": "model", "created": 0, "owned_by": "vieneu",
        "sample_rate": SAMPLE_RATE, "response_formats": ["wav", "pcm"],
        "default_device": pool.default, "devices": pool.status(),
    }]}


@app.get("/", include_in_schema=False)
def test_page():
    """Browser test page; it asks for the API key itself, so it needs no auth."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    if ENGINES is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    primary = ENGINES.primary()
    return {"status": "ok", "device": primary.device, "default_device": ENGINES.default,
            "devices": ENGINES.status(), "sample_rate": SAMPLE_RATE}


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
