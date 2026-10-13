"""
LoRA voices: a speaker records a fixed script sentence by sentence, then the server fine-tunes
VieNeu-TTS v3 Turbo on those clips (one LoRA = one voice).

Each session lives in ``<TTS_LORA_DIR>/<id>/``:

    session.json   name, gender, description, the script it was created with, recorded clips, status
    clips/         one WAV per sentence (``000.wav``, ``001.wav``...), overwritten on a retake
    progress.json  written by the training process (stage, step, loss...)
    train.log      the training process's output
    model/         merged model + ``voices_v3_turbo.json`` (the voice), loaded with ``backbone_repo``
    adapter/       the LoRA adapter alone (a few MB)

Training runs in a separate process (``python -m app.lora_train <session dir>``), one job at a
time, so a crash or out-of-memory error cannot take the server down and its GPU memory is freed
when it ends.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("tts-server.lora")

# About one minute of speech: natural sentences of ~20 syllables (4-6 s each), covering all six
# tones, statements, questions and exclamations. Numbers are written out, as the model reads them.
LORA_SCRIPT = [
    "Xin chào, tôi rất vui được gặp bạn hôm nay. Chúc bạn một ngày làm việc thật hiệu quả.",
    "Buổi sáng ở Hà Nội thường bắt đầu bằng một ly cà phê nóng và tiếng xe cộ ồn ào trên phố.",
    "Bạn có biết đường nào gần nhất để đến bệnh viện không? Tôi cần đến đó trước mười giờ.",
    "Mẹ tôi nấu món canh chua cá lóc ngon tuyệt, cả nhà ai cũng khen và xin thêm bát nữa.",
    "Theo dự báo thời tiết, chiều nay có mưa rào và dông rải rác, nhiệt độ giảm nhẹ về đêm.",
    "Ôi, thật bất ngờ! Mình không nghĩ là cậu lại nhớ sinh nhật của mình đâu.",
    "Quý khách vui lòng giữ trật tự và xếp hàng theo thứ tự để được phục vụ nhanh chóng.",
    "Những cánh đồng lúa chín vàng trải dài tít tắp, gió thổi qua mang theo hương thơm dịu nhẹ.",
    "Anh ấy đã luyện tập chăm chỉ suốt ba năm liền, cuối cùng cũng giành được huy chương vàng.",
    "Nếu gặp sự cố khi sử dụng dịch vụ, xin hãy gọi tổng đài để được hỗ trợ kịp thời.",
    "Cuốn sách này kể về hành trình của một cậu bé nghèo vượt qua khó khăn để theo đuổi ước mơ.",
    "Trời đã khuya rồi, con hãy tắt đèn đi ngủ sớm để sáng mai còn dậy đi học đúng giờ nhé.",
    "Giá rau củ ngoài chợ tuần này tăng khá mạnh, nhiều người nội trợ phải tính toán chi tiêu kỹ hơn.",
    "Cảm ơn bạn đã lắng nghe. Nếu có câu hỏi gì, đừng ngại nhắn tin cho tôi bất cứ lúc nào.",
]

CLIP_MIN_SECONDS = 1.0
CLIP_MAX_SECONDS = 20.0  # the training pipeline skips longer clips
CLIP_MAX_BYTES = 8 * 1024 * 1024

# "record": read LORA_SCRIPT sentence by sentence | "import": upload clips with their own transcripts
MODES = ("record", "import")
IMPORT_MIN_CLIPS = 5
IMPORT_MAX_CLIPS = 1000
TEXT_MAX_CHARS = 500

# recording: clips can change | queued / training: a job owns the session | ready: voice usable
# failed: clips can change again and training can be restarted
EDITABLE = ("recording", "failed")
BUSY = ("queued", "training")


class LoraStore:
    """Recording sessions on disk; every change rewrites ``session.json`` atomically."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self._lock = threading.RLock()
        self.sessions: dict[str, dict] = {}
        if directory.is_dir():
            for f in sorted(directory.glob("*/session.json")):
                try:
                    s = json.loads(f.read_text(encoding="utf-8"))
                except Exception as e:  # noqa: BLE001
                    log.error("cannot read %s, skipping it: %s", f, e)
                    continue
                if s.get("status") in BUSY:  # the training process died with the old server
                    s["status"], s["error"] = "failed", "The server restarted during training. Start it again."
                    self.sessions[s["id"]] = s
                    self._save(s)
                else:
                    self.sessions[s["id"]] = s
        log.info("loaded %d LoRA session(s) from %s", len(self.sessions), directory)

    def path(self, sid: str) -> Path:
        return self.dir / sid

    def _save(self, s: dict) -> None:
        d = self.path(s["id"])
        d.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="session_", suffix=".json", dir=d)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=1)
        os.replace(tmp, d / "session.json")

    def get(self, sid: str) -> dict:
        s = self.sessions.get(sid)
        if s is None:
            raise KeyError(sid)
        return s

    def find_name(self, name: str) -> Optional[dict]:
        key = name.casefold()
        return next((s for s in self.sessions.values() if s["name"].casefold() == key), None)

    def ready(self) -> list[dict]:
        return [s for s in self.sessions.values() if s["status"] == "ready"]

    def create(self, name: str, gender: str, description: str, mode: str = "record") -> dict:
        with self._lock:
            s = {"id": uuid.uuid4().hex[:12], "name": name, "gender": gender, "description": description,
                 "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "recording", "error": None,
                 "denoise": True, "mode": mode, "sentences": list(LORA_SCRIPT) if mode == "record" else [],
                 "clips": {}}
            self.sessions[s["id"]] = s
            self._save(s)
            return s

    def update(self, sid: str, **fields) -> dict:
        with self._lock:
            s = self.get(sid)
            s.update(fields)
            self._save(s)
            return s

    def set_clip(self, sid: str, index: int, wav: bytes, seconds: float, peak: float) -> dict:
        """Save the clip of sentence ``index``, replacing (and deleting) any earlier take."""
        with self._lock:
            s = self.get(sid)
            clips = self.path(sid) / "clips"
            clips.mkdir(parents=True, exist_ok=True)
            # A fresh file name each time: imported items shift down when one is removed, so a
            # name derived from the index could belong to another item.
            name = f"{index:03d}_{uuid.uuid4().hex[:8]}.wav"
            (clips / name).write_bytes(wav)
            old = s["clips"].get(str(index))
            s["clips"][str(index)] = {"file": name, "seconds": round(seconds, 2), "peak": round(peak, 3),
                                      "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            self._save(s)
            if old:
                (clips / old["file"]).unlink(missing_ok=True)
            return s

    def add_item(self, sid: str, text: str, wav: bytes, seconds: float, peak: float) -> dict:
        """Import mode: append a sentence with its clip."""
        with self._lock:
            s = self.get(sid)
            s["sentences"].append(text)
            try:
                return self.set_clip(sid, len(s["sentences"]) - 1, wav, seconds, peak)
            except Exception:
                s["sentences"].pop()
                raise

    def set_text(self, sid: str, index: int, text: str) -> dict:
        with self._lock:
            s = self.get(sid)
            s["sentences"][index] = text
            self._save(s)
            return s

    def remove_clip(self, sid: str, index: int) -> dict:
        """Drop a recording; in import mode the sentence goes too and later items move up."""
        with self._lock:
            s = self.get(sid)
            c = s["clips"].pop(str(index), None)
            if s.get("mode") == "import":
                s["sentences"].pop(index)
                s["clips"] = {str(int(k) - 1 if int(k) > index else int(k)): v for k, v in s["clips"].items()}
            self._save(s)
            if c:
                (self.path(sid) / "clips" / c["file"]).unlink(missing_ok=True)
            return s

    def delete(self, sid: str) -> dict:
        with self._lock:
            s = self.sessions.pop(sid)
        shutil.rmtree(self.path(sid), ignore_errors=True)
        return s

    def progress(self, sid: str) -> dict:
        try:
            return json.loads((self.path(sid) / "progress.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def log_tail(self, sid: str, lines: int = 30) -> str:
        try:
            text = (self.path(sid) / "train.log").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def public(self, s: dict, *, detail: bool = False, queue_position: Optional[int] = None) -> dict:
        """The API view of a session; ``detail`` adds the sentences with their clips and the training log."""
        n = len(s["sentences"])
        recorded = [int(i) for i in s["clips"]]
        out = {"id": s["id"], "name": s["name"], "gender": s["gender"], "description": s["description"],
               "created_at": s["created_at"], "mode": s.get("mode", "record"),
               "status": s["status"], "error": s.get("error"),
               "denoise": s.get("denoise", True), "sentences_total": n, "sentences_recorded": len(recorded),
               "recorded_seconds": round(sum(c["seconds"] for c in s["clips"].values()), 1)}
        if s["status"] in BUSY or s["status"] in ("ready", "failed"):
            out["progress"] = self.progress(s["id"])
        if queue_position is not None:
            out["queue_position"] = queue_position
        if detail:
            out["sentences"] = [{"index": i, "text": t, "clip": s["clips"].get(str(i))}
                                for i, t in enumerate(s["sentences"])]
            if s["status"] in BUSY or s["status"] == "failed":
                out["log"] = self.log_tail(s["id"])
        return out


class LoraTrainer:
    """Runs training jobs one at a time, each in its own process."""

    def __init__(self, store: LoraStore, on_ready: Callable[[dict], None]) -> None:
        self.store = store
        self.on_ready = on_ready
        self._queue: list[str] = []
        self._cond = threading.Condition()
        self._current: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._cancelled: set[str] = set()
        threading.Thread(target=self._run, name="lora-trainer", daemon=True).start()

    def position(self, sid: str) -> Optional[int]:
        """0 = training now, 1 = next... ``None`` if the session has no job."""
        with self._cond:
            if sid == self._current:
                return 0
            return self._queue.index(sid) + 1 if sid in self._queue else None

    def enqueue(self, sid: str) -> None:
        with self._cond:
            d = self.store.path(sid)
            (d / "progress.json").unlink(missing_ok=True)
            for leftover in ("model", "adapter"):  # from an earlier run that failed
                shutil.rmtree(d / leftover, ignore_errors=True)
            self.store.update(sid, status="queued", error=None)
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
                # Wait for the job to let go of its files, so the session folder can be removed.
                self._cond.wait_for(lambda: self._current != sid, timeout=30)

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue:
                    self._cond.wait()
                sid = self._current = self._queue.pop(0)
            try:
                self._train(sid)
            except Exception:  # noqa: BLE001
                log.exception("LoRA job %s crashed", sid)
                if sid in self.store.sessions:
                    self.store.update(sid, status="failed", error="The training job crashed; see the server log.")
            finally:
                with self._cond:
                    self._current, self._proc = None, None
                    self._cancelled.discard(sid)
                    self._cond.notify_all()

    def _train(self, sid: str) -> None:
        s = self.store.update(sid, status="training")
        d = self.store.path(sid)
        root = Path(__file__).resolve().parent.parent
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        log.info("LoRA job %s ('%s') started", sid, s["name"])
        t0 = time.perf_counter()
        with open(d / "train.log", "w", encoding="utf-8") as out:
            with self._cond:
                if sid in self._cancelled or sid not in self.store.sessions:  # deleted before it started
                    return
                self._proc = subprocess.Popen([sys.executable, "-m", "app.lora_train", str(d)], cwd=root,
                                              env=env, stdout=out, stderr=subprocess.STDOUT)
            code = self._proc.wait()
        if sid in self._cancelled or sid not in self.store.sessions:
            log.info("LoRA job %s cancelled", sid)
            return
        progress = self.store.progress(sid)
        if code != 0 or not (d / "model" / "voices_v3_turbo.json").is_file():
            err = progress.get("error") or self.store.log_tail(sid, 1) or f"training exited with code {code}"
            log.error("LoRA job %s failed after %.0fs: %s", sid, time.perf_counter() - t0, err)
            self.store.update(sid, status="failed", error=err)
            return
        log.info("LoRA job %s done in %.0fs", sid, time.perf_counter() - t0)
        s = self.store.update(sid, status="ready", error=None, trained_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        try:
            self.on_ready(s)
        except Exception:  # noqa: BLE001
            log.exception("loading LoRA voice '%s' after training failed", s["name"])
