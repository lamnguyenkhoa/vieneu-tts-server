"""
Train one LoRA voice from a recording session; the server runs this as a subprocess:

    python -m app.lora_train <session dir>

It follows VieNeu-TTS's ``finetune/`` pipeline (prepare_dataset.py, train_lora.py --merge,
make_voice.py), with settings for about a minute of speech:

1. prepare: each clip is denoised (optional), trimmed of leading/trailing silence, then encoded
   like the SDK does at inference time (phonemes, MOSS codec codes, 192-d speaker embedding).
2. train: LoRA on the backbone (``app.vieneu_lora``, vendored from upstream).
3. export: merged model in ``<session>/model`` plus ``voices_v3_turbo.json`` holding the voice
   (the mean speaker embedding of the clips, no reference codes, as the LoRA was trained).

Progress goes to ``<session>/progress.json``; everything printed goes to ``train.log``.
Settings (environment): TTS_LORA_EPOCHS (25), TTS_LORA_LR (2e-4), TTS_LORA_RANK (16),
TTS_LORA_BATCH (4).
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

BASE_REPO = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
SUBFOLDER = "update"


def _env(name: str, default, cast):
    try:
        return cast(os.environ.get(name) or default)
    except ValueError:
        return default


class Progress:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.state: dict = {"started_at": time.time()}

    def __call__(self, **fields) -> None:
        self.state.update(fields, updated_at=time.time())
        fd, tmp = tempfile.mkstemp(prefix="progress_", suffix=".json", dir=self.path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False)
        os.replace(tmp, self.path)


def phonemize(text: str) -> str:
    """Same as upstream prepare_dataset.py: the SDK's normaliser + phonemiser, unchunked."""
    from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps, phonemize_text_with_emotions

    chunks, _ = normalize_to_chunks_v3_with_gaps(text, max_chars=1_000_000)
    return " ".join(phonemize_text_with_emotions(c) for c in chunks).strip()


def trim_silence(wav: np.ndarray, sr: int, pad: float = 0.15) -> np.ndarray:
    """Cut the silence before the first and after the last word (the pause around the Record button)."""
    hop = max(1, int(sr * 0.01))
    n = len(wav) // hop
    if n < 3:
        return wav
    rms = np.sqrt(np.mean(wav[: n * hop].reshape(n, hop) ** 2, axis=1))
    loud = np.flatnonzero(rms > max(rms.max() * 0.05, 1e-3))
    if not len(loud):
        return wav
    start = max(0, loud[0] * hop - int(pad * sr))
    end = min(len(wav), (loud[-1] + 1) * hop + int(pad * sr))
    return wav[start:end]


def prepare(session: dict, sdir: Path, progress: Progress) -> tuple[list[dict], np.ndarray]:
    """Upstream encodes with the ONNX (CPU) engine; this uses the PyTorch one, as GPU inference does.

    Same codec, speaker encoder and denoiser, and it avoids loading the ONNX backbone, whose
    external-data files some onnxruntime versions refuse to open from the Hugging Face cache.
    """
    import gc

    import soundfile as sf
    import torch
    from vieneu._v3_turbo_engine import VieNeuTTSv3Turbo

    progress(stage="preparing", message="Loading the encoder", done=0, total=len(session["clips"]))
    eng = VieNeuTTSv3Turbo(checkpoint_path=BASE_REPO, model_subfolder=SUBFOLDER,
                           device="cuda" if torch.cuda.is_available() else "cpu")
    denoise = bool(session.get("denoise", True)) and eng.denoiser is not None

    rows, embs = [], []
    items = sorted(session["clips"].items(), key=lambda kv: int(kv[0]))
    for done, (idx, clip) in enumerate(items, 1):
        text = session["sentences"][int(idx)]
        wav, sr = sf.read(str(sdir / "clips" / clip["file"]), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if denoise:
            wav, sr = eng.denoiser.denoise(wav, sr), 44100  # -> float32 mono @ 44.1 kHz
        wav = trim_silence(np.asarray(wav, dtype=np.float32), sr)
        phones = phonemize(text)
        if not phones:
            print(f"  skip clip {idx}: empty phonemes")
            continue
        codes = eng._encode_ref_wav(torch.from_numpy(wav)[None], sr)            # (T, 16) int64 @ 12.5 frames/s
        emb = np.asarray(eng.speaker_encoder.embed(wav, sr), dtype=np.float32)  # (192,)
        rows.append({"file_name": clip["file"], "text": text, "phones": phones,
                     "codes": codes.astype(np.int64).tolist(), "speaker_embedding": emb.tolist(),
                     "duration": len(wav) / sr})
        embs.append(emb)
        print(f"  clip {idx}: {len(wav) / sr:.1f}s, {len(codes)} frames")
        progress(message=f"Encoded clip {done} of {len(items)}", done=done)
    del eng  # free the GPU for training
    gc.collect()
    torch.cuda.empty_cache()
    if not rows:
        raise RuntimeError("No usable clips.")
    total = sum(r["duration"] for r in rows)
    print(f"prepared {len(rows)} clips, {total:.1f}s of speech (denoise={denoise})")
    return rows, np.mean(np.stack(embs), axis=0)


def train(rows: list[dict], out_dir: Path, progress: Progress) -> None:
    """upstream train_lora.py with its defaults, tuned for a handful of clips (no eval split, no checkpoints)."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_scheduler
    from vieneu._v3_turbo_engine.hub_load_v3_turbo import load_v3_turbo_checkpoint

    from app.vieneu_lora import V3TurboLoraDataset, attach_lora, compute_loss, save_adapter
    from app.vieneu_lora.lora import export_merged, merge_lora_into, trainable_parameters

    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training needs a CUDA GPU.")
    epochs = _env("TTS_LORA_EPOCHS", 25, float)
    lr = _env("TTS_LORA_LR", 2e-4, float)
    rank = _env("TTS_LORA_RANK", 16, int)
    batch_size = _env("TTS_LORA_BATCH", 4, int)
    text_w, audio_w = 1.0, 8.0
    torch.manual_seed(42)
    np.random.default_rng(42).shuffle(rows)
    device = torch.device("cuda")

    progress(stage="training", message="Loading the model", step=0, total_steps=None)
    tokenizer = AutoTokenizer.from_pretrained(BASE_REPO, subfolder=SUBFOLDER, trust_remote_code=True)
    model = load_v3_turbo_checkpoint(BASE_REPO, subfolder=SUBFOLDER, device=device, dtype=torch.float32)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    peft_model = attach_lora(model, r=rank, alpha=rank * 2, dropout=0.05, target="backbone")
    n_train = sum(p.numel() for p in trainable_parameters(model))
    print(f"trainable params: {n_train / 1e6:.2f}M, rank {rank}, lr {lr}, epochs {epochs}, batch {batch_size}")

    ds = V3TurboLoraDataset(rows, tokenizer, model.config, max_length=1024)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=ds.collate)
    total_steps = max(1, math.ceil(epochs * len(loader)))
    optim = torch.optim.AdamW(list(trainable_parameters(model)), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = get_scheduler("cosine", optim, num_warmup_steps=math.ceil(total_steps * 0.05),
                          num_training_steps=total_steps)
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731

    model.train()
    step, t0 = 0, time.perf_counter()
    progress(message="Training", total_steps=total_steps)
    while step < total_steps:
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast():
                loss, m = compute_loss(model, batch, text_w, audio_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(trainable_parameters(model)), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            elapsed = time.perf_counter() - t0
            if step == 1 or step % 5 == 0 or step == total_steps:
                print(f"step {step}/{total_steps}  loss {loss.item():.4f}  audio {m['audio_loss']:.3f}  "
                      f"acc_cb0 {m['acc_cb0']:.3f}  {elapsed:.0f}s")
            progress(step=step, loss=round(loss.item(), 4), acc_cb0=round(m["acc_cb0"], 3),
                     eta_s=round(elapsed / step * (total_steps - step)))
            if step >= total_steps:
                break

    progress(stage="saving", message="Merging and saving the model")
    save_adapter(peft_model, out_dir / "adapter", tokenizer)
    merged = merge_lora_into(peft_model)
    export_merged(merged, tokenizer, BASE_REPO, out_dir / "model", subfolder=SUBFOLDER)


def write_voice(session: dict, model_dir: Path, emb: np.ndarray) -> None:
    """upstream make_voice.py: the voice ships with the model as an embedding without reference codes."""
    data = {"presets": {session["name"]: {
        "description": session.get("description", ""), "gender": session.get("gender", ""),
        "speaker_emb": [round(float(x), 6) for x in emb.reshape(-1)], "codes": None,
    }}, "default_voice": session["name"]}
    (model_dir / "voices_v3_turbo.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    sdir = Path(sys.argv[1]).resolve()
    session = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
    progress = Progress(sdir / "progress.json")
    t0 = time.perf_counter()
    try:
        rows, emb = prepare(session, sdir, progress)
        train(rows, sdir, progress)
        write_voice(session, sdir / "model", emb)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        progress(stage="failed", error=f"{type(e).__name__}: {e}")
        sys.exit(1)
    progress(stage="done", message=f"Finished in {time.perf_counter() - t0:.0f}s")
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
