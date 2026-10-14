"""
AI-assisted LoRA voices: one long recording of one speaker becomes an import-mode LoRA session.

1. decode: ffmpeg (any format; soundfile for WAV/FLAC/OGG/MP3 without it) to mono 16-bit at 24 kHz,
   the model's own rate.
2. split: at the pauses, into clips of about 3-15 s. The speaker's long pauses (sentence ends) always
   cut; longer stretches are cut at their longest pause, then short neighbours are merged again.
3. transcribe: each clip goes to the speech-to-text server (``TTS_STT_URL``, a phostt server:
   ``POST /transcribe`` multipart {file} -> {"text"}). Clips it hears no speech in are dropped.
4. the clips are added to the session as they are transcribed; the speaker then reviews them on the
   /lora-auto page and trains the voice as usual (``app.lora``).

While this runs the session's status is "transcribing", so its clips cannot be edited or trained.
Jobs run one at a time in a thread (they use the CPU and the STT server, not the GPU).
Settings (environment): TTS_STT_URL (http://localhost:8123), TTS_AUTO_MAX_MB (500),
TTS_AUTO_MAX_MINUTES (120), TTS_FFMPEG (empty = ffmpeg on PATH).
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from app import lora
from app.lora import LoraStore

log = logging.getLogger("tts-server.autolora")


def _env_num(name: str, default, cast):
    try:
        return cast(os.environ.get(name) or default)
    except ValueError:
        return default


STT_URL = (os.environ.get("TTS_STT_URL") or "http://localhost:8123").rstrip("/")
MAX_UPLOAD_MB = _env_num("TTS_AUTO_MAX_MB", 500, int)
MAX_MINUTES = _env_num("TTS_AUTO_MAX_MINUTES", 120, float)
MIN_SOURCE_SECONDS = 10.0

SR = 24_000
HOP = SR // 100  # 10 ms analysis frames; the splitter below counts in frames
MIN_PAUSE = 15   # 0.15 s: shorter dips are inside words
HARD_PAUSE = 60  # 0.6 s: a sentence end, always cut
CLIP_MIN = 300   # 3 s: no cut leaves a piece shorter than this
CLIP_MAX = 1500  # 15 s: longer stretches are cut (the training limit is lora.CLIP_MAX_SECONDS)
MERGE_MAX = 1200  # 12 s: short clips are merged with the next one up to this length...
MERGE_GAP = 100   # ...across a pause of at most 1 s
KEEP_MIN = 100    # 1 s of speech: shorter clips are dropped
PAD = 15          # 0.15 s of the pause kept on each side of a clip (at most half the pause)

STT_THREADS = 4   # clips sent to the STT server at once
STT_TIMEOUT = 120


class AutoError(Exception):
    """A problem to show the speaker (the session keeps the clips found so far)."""


class Cancelled(Exception):
    pass


def ffmpeg_path() -> Optional[str]:
    exe = os.environ.get("TTS_FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def stt_health() -> tuple[bool, str]:
    """(reachable, detail) for the STT server."""
    try:
        with urllib.request.urlopen(f"{STT_URL}/health", timeout=3) as r:
            body = json.loads(r.read())
        if body.get("status") == "ok":
            return True, f"{body.get('model') or 'model'} on {body.get('device') or '?'}"
        return False, f"status {body.get('status')!r}"
    except (OSError, ValueError) as e:
        return False, str(getattr(e, "reason", e))


def transcribe(wav: bytes) -> str:
    """Text of one clip from the STT server ("" when it hears no speech)."""
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="clip.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n").encode() + wav + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{STT_URL}/transcribe", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    last: Exception = RuntimeError("no attempt")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=STT_TIMEOUT) as r:
                return str(json.loads(r.read()).get("text") or "")
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:  # it rejected this clip; drop it rather than the whole job
                log.warning("STT rejected a clip: HTTP %d", e.code)
                return ""
            last = e
        except (OSError, ValueError) as e:
            last = e
        time.sleep(1 + 2 * attempt)
    raise AutoError(f"The speech-to-text server ({STT_URL}) stopped answering: {getattr(last, 'reason', last)}")


def clean_text(text: str) -> str:
    """phostt writes lowercase without punctuation: capitalise and end the sentence (each clip ends at a pause)."""
    text = " ".join(unicodedata.normalize("NFC", text).split())
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".?!…" else text + "."


# ── splitting ───────────────────────────────────────────────────────────────

def frame_db(x: np.ndarray) -> np.ndarray:
    """Level of each 10 ms frame in dBFS, smoothed over 30 ms (computed in chunks to bound memory)."""
    n = len(x) // HOP
    out = np.empty(n, dtype=np.float32)
    step = 30_000  # frames (5 min) per chunk
    for i in range(0, n, step):
        f = x[i * HOP: min(n, i + step) * HOP].astype(np.float32).reshape(-1, HOP) / 32768.0
        out[i: i + len(f)] = 10 * np.log10(np.mean(f * f, axis=1) + 1e-10)
    return np.convolve(out, np.ones(3, dtype=np.float32) / 3, mode="same") if n >= 3 else out


def segment(db: np.ndarray) -> list[tuple[int, int]]:
    """Speech clips as (start, end) frame ranges, about 3-15 s each, cut at pauses."""
    n = len(db)
    if n == 0:
        return []
    floor, loud = np.percentile(db, 10), np.percentile(db, 95)
    silent = db < floor + max(6.0, 0.3 * (loud - floor))
    edges = np.flatnonzero(np.diff(np.concatenate(([0], silent.astype(np.int8), [0]))))
    pauses = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])
              if b - a >= MIN_PAUSE or a == 0 or b == n]

    spans, pos = [], 0  # speech between the pauses
    for a, b in pauses:
        if a > pos:
            spans.append((pos, a))
        pos = b
    if pos < n:
        spans.append((pos, n))
    if not spans:
        return []

    def fit(group: list[tuple[int, int]]) -> list[tuple[int, int]]:
        a, b = group[0][0], group[-1][1]
        if b - a <= CLIP_MAX:
            return [(a, b)]
        if len(group) == 1:  # no pause at all: cut at the quietest moment
            cut = a + CLIP_MIN + int(np.argmin(db[a + CLIP_MIN: b - CLIP_MIN]))
            return fit([(a, cut)]) + fit([(cut, b)])
        mid = (a + b) / 2

        def score(i: int):  # the longest pause that leaves both sides long enough, nearer the middle on a tie
            left, right = group[i][1], group[i + 1][0]
            return (left - a >= CLIP_MIN and b - right >= CLIP_MIN, right - left - abs((left + right) / 2 - mid) * 0.02)

        i = max(range(len(group) - 1), key=score)
        return fit(group[: i + 1]) + fit(group[i + 1:])

    clips, group = [], [spans[0]]
    for prev, cur in zip(spans, spans[1:]):
        if cur[0] - prev[1] >= HARD_PAUSE:
            clips += fit(group)
            group = []
        group.append(cur)
    clips += fit(group)

    merged: list[tuple[int, int]] = []
    for c in clips:
        if (merged and merged[-1][1] - merged[-1][0] < CLIP_MIN and c[0] - merged[-1][1] <= MERGE_GAP
                and c[1] - merged[-1][0] <= MERGE_MAX):
            merged[-1] = (merged[-1][0], c[1])
        else:
            merged.append(c)
    return [c for c in merged if c[1] - c[0] >= KEEP_MIN]


def padded(clips: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    """Sample ranges with up to PAD of the surrounding pauses (never past the middle of a pause)."""
    out = []
    for i, (a, b) in enumerate(clips):
        prev_end = clips[i - 1][1] if i else 0
        next_start = clips[i + 1][0] if i + 1 < len(clips) else n
        a -= min(PAD, (a - prev_end) // 2 if i else a)
        b += min(PAD, (next_start - b) // 2 if i + 1 < len(clips) else n - b)
        out.append((a * HOP, b * HOP))
    return out


def wav_bytes(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


# ── jobs ────────────────────────────────────────────────────────────────────

class AutoLora:
    """Turns uploaded recordings into clips + transcripts, one job at a time."""

    def __init__(self, store: LoraStore) -> None:
        self.store = store
        self._queue: list[str] = []
        self._cond = threading.Condition()
        self._current: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._cancelled: set[str] = set()
        threading.Thread(target=self._run, name="lora-auto", daemon=True).start()

    def submit(self, sid: str) -> None:
        with self._cond:
            self._queue.append(sid)
            self._cond.notify()

    def cancel(self, sid: str) -> None:
        """Drop a queued job or stop the running one (its session is being deleted)."""
        with self._cond:
            if sid in self._queue:
                self._queue.remove(sid)
            elif sid == self._current:
                self._cancelled.add(sid)
                if self._proc is not None:
                    self._proc.kill()
                self._cond.wait_for(lambda: self._current != sid, timeout=30)

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue:
                    self._cond.wait()
                sid = self._current = self._queue.pop(0)
            try:
                self._process(sid)
            finally:
                with self._cond:
                    self._current, self._proc = None, None
                    self._cancelled.discard(sid)
                    self._cond.notify_all()

    def _check(self, sid: str) -> dict:
        if sid in self._cancelled:
            raise Cancelled
        try:
            return self.store.get(sid)
        except KeyError:
            raise Cancelled from None

    def _progress(self, sid: str, **fields) -> None:
        s = self._check(sid)
        self.store.update(sid, auto={**s.get("auto", {}), **fields, "updated_at": time.time()})

    def _process(self, sid: str) -> None:
        try:
            src = self.store.path(sid) / self._check(sid)["auto"]["file"]
        except Cancelled:
            return
        t0 = time.perf_counter()
        try:
            self._progress(sid, stage="decoding")
            x, truncated = self._decode(sid, src)
            src.unlink(missing_ok=True)
            seconds = len(x) / SR
            if seconds < MIN_SOURCE_SECONDS:
                raise AutoError(f"The recording is only {seconds:.1f} s long.")
            self._progress(sid, stage="splitting", source_seconds=round(seconds, 1), truncated=truncated)
            clips = padded(segment(frame_db(x)), len(x) // HOP)
            if not clips:
                raise AutoError("No speech was found in the recording.")
            log.info("auto LoRA %s: %.0f s of audio split into %d clips", sid, seconds, len(clips))
            self._progress(sid, stage="transcribing", done=0, total=len(clips), kept=0, no_speech=0)
            kept = no_speech = 0
            full = False
            with ThreadPoolExecutor(STT_THREADS) as pool:
                for start in range(0, len(clips), STT_THREADS * 2):
                    self._check(sid)
                    batch = []
                    for a, b in clips[start: start + STT_THREADS * 2]:
                        pcm = x[a:b]
                        batch.append((wav_bytes(pcm), (b - a) / SR, float(np.abs(pcm).max()) / 32768.0))
                    texts = list(pool.map(transcribe, [w for w, _, _ in batch]))
                    for (wav, secs, peak), text in zip(batch, texts):
                        text = clean_text(text)[: lora.TEXT_MAX_CHARS]
                        if not text or peak < 0.01:
                            no_speech += 1
                            continue
                        if len(self._check(sid)["sentences"]) >= lora.IMPORT_MAX_CLIPS:
                            full = True
                            break
                        self.store.add_item(sid, text, wav, secs, peak)
                        kept += 1
                    self._progress(sid, done=min(len(clips), start + len(batch)), kept=kept, no_speech=no_speech)
                    if full:
                        break
            if not kept:
                raise AutoError("The speech-to-text server heard no speech in any clip.")
            note = f" Stopped at {lora.IMPORT_MAX_CLIPS} clips, the most a voice can have." if full else ""
            self._progress(sid, stage="done", message=note.strip() or None)
            self.store.update(sid, status="recording", error=None)
            log.info("auto LoRA %s: %d clips kept, %d without speech, in %.0f s",
                     sid, kept, no_speech, time.perf_counter() - t0)
        except (Cancelled, KeyError):  # KeyError: the session was deleted under the job
            log.info("auto LoRA %s cancelled", sid)
        except Exception as e:  # noqa: BLE001
            if isinstance(e, AutoError):
                msg = str(e)
                log.warning("auto LoRA %s failed: %s", sid, msg)
            else:
                msg = "Processing the recording failed; see the server log."
                log.exception("auto LoRA %s crashed", sid)
            try:
                self._progress(sid, stage="failed", message=msg)
                self.store.update(sid, status="recording", error=msg)
            except (Cancelled, KeyError):
                pass
        finally:
            src.unlink(missing_ok=True)

    def _decode(self, sid: str, path: Path) -> tuple[np.ndarray, bool]:
        """(16-bit mono samples at SR, whether the recording was cut at MAX_MINUTES)."""
        limit = int(MAX_MINUTES * 60 * SR)
        exe = ffmpeg_path()
        if exe:
            with self._cond:
                self._check(sid)
                self._proc = subprocess.Popen(
                    [exe, "-v", "error", "-nostdin", "-i", str(path), "-t", str(MAX_MINUTES * 60 + 1),
                     "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = self._proc.communicate()
            code = self._proc.returncode
            self._check(sid)
            if code != 0:
                tail = err.decode("utf-8", "replace").strip().splitlines()[-1:] or [f"exit code {code}"]
                raise AutoError(f"ffmpeg could not decode the file: {tail[0]}")
            x = np.frombuffer(out, dtype=np.int16)
        else:
            import soundfile as sf
            import soxr

            try:
                info = sf.info(str(path))
                wav, sr = sf.read(str(path), frames=int((MAX_MINUTES * 60 + 1) * info.samplerate),
                                  dtype="float32", always_2d=True)
            except Exception:  # noqa: BLE001
                raise AutoError("Cannot read this format without ffmpeg; install ffmpeg or upload WAV, FLAC, "
                                "OGG or MP3.") from None
            wav = wav.mean(axis=1)
            if sr != SR:
                wav = soxr.resample(wav, sr, SR)
            x = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
        return x[:limit], len(x) > limit
