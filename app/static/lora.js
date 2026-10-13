const $ = (id) => document.getElementById(id);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};

$("key").value = store.get("tts-key") || "";
$("key").addEventListener("change", () => { store.set("tts-key", $("key").value); route(); });

function authHeaders() {
  const key = $("key").value.trim();
  return key ? { "Authorization": `Bearer ${key}` } : {};
}

async function api(path, { method = "GET", json, form } = {}) {
  const headers = authHeaders();
  let body;
  if (json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(json); }
  if (form) body = form;
  const res = await fetch(path, { method, headers, body });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).error?.message || msg; } catch {}
    throw new Error(msg);
  }
  return res;
}

function setMsg(id, msg, isError = false) {
  $(id).textContent = msg;
  $(id).classList.toggle("error", isError);
}

const STATUS_LABEL = { recording: "Recording", queued: "Queued", training: "Training", ready: "Ready", failed: "Failed" };
const EDITABLE = ["recording", "failed"];
const BUSY = ["queued", "training"];
let clipMin = 1, clipMax = 20;

// ── list view ───────────────────────────────────────────────────────────────
async function loadList() {
  const ul = $("sessions");
  try {
    const { data, gpu_available, clip_min_seconds, clip_max_seconds, script_sentences } = await (await api("/v1/lora/sessions")).json();
    clipMin = clip_min_seconds; clipMax = clip_max_seconds;
    $("sentenceCount").textContent = script_sentences;
    $("gpuWarning").hidden = gpu_available;
    ul.innerHTML = "";
    if (!data.length) { ul.innerHTML = '<li class="hint">None yet.</li>'; return; }
    for (const s of data) {
      const li = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = [s.name, s.gender, s.description].filter(Boolean).join(" · ");
      const badge = document.createElement("span");
      badge.className = `badge ${s.status}`;
      badge.textContent = s.status === "recording" ? `${s.sentences_recorded}/${s.sentences_total} recorded` : STATUS_LABEL[s.status];
      const open = document.createElement("button");
      open.textContent = s.status === "ready" ? "Open" : s.status === "recording" ? "Continue" : "Open";
      open.addEventListener("click", () => { location.hash = s.id; });
      const left = document.createElement("span");
      left.className = "grow";
      left.append(label, " ", badge);
      li.append(left, open);
      ul.append(li);
    }
  } catch (e) {
    ul.innerHTML = "";
    const li = document.createElement("li");
    li.className = "status error";
    li.textContent = `Could not load voices: ${e.message}`;
    ul.append(li);
  }
}

$("newName").addEventListener("input", () => { $("create").disabled = !$("newName").value.trim(); });
$("create").addEventListener("click", async () => {
  $("create").disabled = true;
  try {
    const s = await (await api("/v1/lora/sessions", { method: "POST", json: {
      name: $("newName").value.trim(), gender: $("newGender").value, description: $("newDesc").value.trim(),
    } })).json();
    $("newName").value = $("newDesc").value = "";
    setMsg("createStatus", "");
    location.hash = s.id;
  } catch (e) {
    setMsg("createStatus", `Could not create the voice: ${e.message}`, true);
    $("create").disabled = false;
  }
});

// ── session view ────────────────────────────────────────────────────────────
let session = null;  // GET /v1/lora/sessions/{id}
let current = 0;     // sentence index on the recorder
let take = null;     // unsaved recording: {index, blob, url, seconds, peak}
let pollTimer = null;

async function openSession(id) {
  try {
    session = await (await api(`/v1/lora/sessions/${encodeURIComponent(id)}`)).json();
  } catch (e) {
    alert(`Could not open the voice: ${e.message}`);
    location.hash = "";
    return;
  }
  const firstMissing = session.sentences.findIndex((x) => !x.clip);
  current = firstMissing >= 0 ? firstMissing : 0;
  discardTake();
  $("denoise").checked = session.denoise;
  setMsg("trainStatus", "");
  setMsg("recStatus", "");
  render();
  schedulePoll();
}

const editable = () => session && EDITABLE.includes(session.status);

function render() {
  const s = session;
  $("sName").textContent = [s.name, s.gender, s.description].filter(Boolean).join(" · ");
  $("sBadge").className = `badge ${s.status}`;
  $("sBadge").textContent = s.status === "queued" && s.queue_position > 1
    ? `Queued (#${s.queue_position})` : STATUS_LABEL[s.status];
  $("recordedBar").style.width = `${(100 * s.sentences_recorded) / s.sentences_total}%`;
  $("recordedText").textContent = `${s.sentences_recorded} of ${s.sentences_total} sentences recorded · ${s.recorded_seconds} s of speech (aim for about 60 s)`;

  $("recorder").hidden = !editable();
  renderRecorder();
  renderSentences();
  renderTraining();
  $("tryPanel").hidden = s.status !== "ready";
}

function renderRecorder() {
  if (!session) return;
  const x = session.sentences[current];
  $("sentencePos").textContent = `Sentence ${current + 1} of ${session.sentences.length}` +
    (x.clip ? ` · saved (${x.clip.seconds.toFixed(1)} s)` : "");
  $("prompt").textContent = x.text;
  const hasTake = take && take.index === current;
  $("recBtn").textContent = recording ? "Stop" : x.clip || hasTake ? "Record again" : "Record";
  $("recBtn").classList.toggle("recording", recording);
  $("playBtn").disabled = recording || !(hasTake || x.clip);
  $("saveBtn").disabled = recording || !hasTake;
  $("retakeBtn").disabled = recording || !(hasTake || x.clip);
  $("prevBtn").disabled = recording || current === 0;
  $("nextBtn").disabled = recording || current === session.sentences.length - 1;
}

function renderSentences() {
  const ol = $("sentences");
  ol.innerHTML = "";
  session.sentences.forEach((x, i) => {
    const li = document.createElement("li");
    li.classList.toggle("active", i === current && editable());
    li.classList.toggle("done", !!x.clip);
    const text = document.createElement("span");
    text.className = "grow";
    text.lang = "vi";
    text.textContent = x.text;
    const state = document.createElement("span");
    state.className = "hint nowrap";
    const unsaved = take && take.index === i;
    state.textContent = unsaved ? "unsaved take" : x.clip ? `✓ ${x.clip.seconds.toFixed(1)} s` : "not recorded";
    li.append(text, state);
    if (x.clip) {
      const play = document.createElement("button");
      play.textContent = "Play";
      play.addEventListener("click", (e) => { e.stopPropagation(); playSaved(i); });
      li.append(play);
    }
    if (editable()) li.addEventListener("click", () => select(i));
    ol.append(li);
  });
}

function renderTraining() {
  const s = session;
  const complete = s.sentences_recorded === s.sentences_total;
  $("trainBtn").hidden = !editable();
  $("denoise").parentElement.hidden = !editable();
  $("trainBtn").disabled = !complete || recording;
  $("trainBtn").textContent = s.status === "failed" ? "Train again" : "Start training";
  $("trainHint").textContent = s.status === "ready"
    ? "Training finished. The voice is in the voice list of the test page; it runs on the GPU."
    : !editable() ? "Training runs on the server's GPU. You can leave this page and come back later."
    : complete ? "Every sentence is saved. Training runs on the server's GPU and takes a few minutes; you can leave this page and come back later."
    : `Record all ${s.sentences_total} sentences to start training (${s.sentences_total - s.sentences_recorded} left).`;

  const p = s.progress || {};
  const show = BUSY.includes(s.status) || (s.status === "ready" && p.stage);
  $("trainProgress").hidden = !show;
  if (show) {
    let pct = 0, text = "";
    if (s.status === "queued") {
      text = s.queue_position > 1 ? `Waiting for ${s.queue_position - 1} other job(s) to finish…` : "Starting…";
    } else if (s.status === "ready") {
      pct = 100;
      text = p.message || "Done.";
    } else if (p.stage === "preparing") {
      pct = 15 * (p.done || 0) / (p.total || 1);
      text = `Preparing clips: ${p.message || "…"}`;
    } else if (p.stage === "training" && p.total_steps) {
      pct = 15 + 75 * (p.step || 0) / p.total_steps;
      text = `Training: step ${p.step || 0} of ${p.total_steps}` +
        (p.loss != null ? ` · loss ${p.loss.toFixed(3)}` : "") +
        (p.eta_s != null && p.step ? ` · about ${fmtSeconds(p.eta_s)} left` : "");
    } else if (p.stage === "training") {
      pct = 15;
      text = p.message ? `${p.message}…` : "Loading the model…";
    } else if (p.stage === "saving") {
      pct = 92;
      text = "Merging and saving the model…";
    } else {
      text = "Starting…";
    }
    $("trainBar").style.width = `${pct}%`;
    $("trainText").textContent = text;
  }
  if (s.status === "failed") setMsg("trainStatus", `Training failed: ${s.error || "unknown error"}`, true);
  else if (s.status === "ready") setMsg("trainStatus", "");
  $("logBox").hidden = !s.log;
  $("trainLog").textContent = s.log || "";
}

function fmtSeconds(s) {
  return s >= 90 ? `${Math.round(s / 60)} min` : `${Math.max(1, Math.round(s))} s`;
}

function schedulePoll() {
  clearTimeout(pollTimer);
  if (!session || !BUSY.includes(session.status)) return;
  const id = session.id;
  pollTimer = setTimeout(async () => {
    if (!session || session.id !== id) return;
    try {
      const s = await (await api(`/v1/lora/sessions/${encodeURIComponent(id)}`)).json();
      if (!session || session.id !== id) return;
      session = s;
      render();
    } catch {}
    schedulePoll();
  }, 2000);
}

function select(i) {
  if (recording || i < 0 || i >= session.sentences.length) return;
  if (take && take.index !== i) discardTake();
  current = i;
  setMsg("recStatus", "");
  renderRecorder();
  renderSentences();
}

$("prevBtn").addEventListener("click", () => select(current - 1));
$("nextBtn").addEventListener("click", () => select(current + 1));

// ── microphone: raw PCM through an AudioWorklet (lossless, unlike MediaRecorder's Opus) ──
let mic = null;          // {stream, ctx, node}
let recording = false;
let chunks = [], recStart = 0, recTimer = null, level = 0;

const TAP_WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("tap", Tap);`;

async function ensureMic() {
  if (mic) return mic;
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("this browser cannot record here; recording needs https or localhost");
  // No browser processing: echo cancellation / noise suppression / AGC colour the voice the model learns.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  const ctx = new AudioContext();
  const url = URL.createObjectURL(new Blob([TAP_WORKLET], { type: "application/javascript" }));
  try { await ctx.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
  const node = new AudioWorkletNode(ctx, "tap");
  node.port.onmessage = (e) => {
    const x = e.data;
    let peak = 0;
    for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > peak) peak = a; }
    level = Math.max(peak, level * 0.97);
    if (recording) chunks.push(x);
  };
  ctx.createMediaStreamSource(stream).connect(node);
  node.connect(ctx.destination);  // keeps the node running; it outputs silence
  mic = { stream, ctx, node };
  drawLevel();
  return mic;
}

function releaseMic() {
  if (!mic) return;
  mic.stream.getTracks().forEach((t) => t.stop());
  mic.ctx.close().catch(() => {});
  mic = null;
  level = 0;
}

function drawLevel() {
  if (!mic) { $("levelBar").style.width = "0"; return; }
  // dBFS-ish scale: -60 dB = empty, 0 dB = full.
  const db = 20 * Math.log10(Math.max(level, 1e-4));
  $("levelBar").style.width = `${Math.max(0, Math.min(100, ((db + 60) / 60) * 100))}%`;
  $("levelBar").classList.toggle("hot", level >= 0.98);
  requestAnimationFrame(drawLevel);
}

async function startRecording() {
  if (!editable() || recording) return;
  try {
    await ensureMic();
    if (mic.ctx.state === "suspended") await mic.ctx.resume();
  } catch (e) {
    setMsg("recStatus", `Microphone unavailable: ${e.message}`, true);
    return;
  }
  discardTake();
  chunks = [];
  recording = true;
  recStart = performance.now();
  setMsg("recStatus", "Recording… 0.0 s");
  recTimer = setInterval(() => {
    const s = (performance.now() - recStart) / 1000;
    setMsg("recStatus", `Recording… ${s.toFixed(1)} s`);
    if (s >= clipMax) stopRecording();
  }, 100);
  renderRecorder();
  renderTraining();
}

function stopRecording() {
  if (!recording) return;
  recording = false;
  clearInterval(recTimer);
  const rate = mic.ctx.sampleRate;
  const n = chunks.reduce((a, c) => a + c.length, 0);
  const pcm = new Float32Array(n);
  let off = 0, peak = 0;
  for (const c of chunks) { pcm.set(c, off); off += c.length; }
  chunks = [];
  for (let i = 0; i < n; i++) { const a = Math.abs(pcm[i]); if (a > peak) peak = a; }
  const seconds = n / rate;
  renderRecorder();
  renderTraining();
  if (seconds < clipMin) { setMsg("recStatus", `Too short (${seconds.toFixed(1)} s). Press Record and read the whole sentence.`, true); return; }
  if (peak < 0.02) { setMsg("recStatus", "Almost silent. Check that the right microphone is selected and not muted.", true); return; }
  const blob = wavFromFloat(pcm, rate);
  take = { index: current, blob, url: URL.createObjectURL(blob), seconds, peak };
  const warn = peak >= 0.98 ? " It is clipping (too loud): move back a little and retake."
    : peak < 0.1 ? " It is quite quiet: move closer to the mic or raise its volume." : "";
  setMsg("recStatus", `Take ready: ${seconds.toFixed(1)} s. Play it back, then Save & next or Retake.${warn}`, !!warn);
  renderRecorder();
  renderSentences();
}

function discardTake() {
  if (take) URL.revokeObjectURL(take.url);
  take = null;
}

function wavFromFloat(pcm, rate) {
  const len = pcm.length * 2;
  const buf = new ArrayBuffer(44 + len), v = new DataView(buf);
  const str = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + len, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, len, true);
  for (let i = 0; i < pcm.length; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 32768 : s * 32767, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

let savedUrl = null;
async function playSaved(i) {
  try {
    const blob = await (await api(`/v1/lora/sessions/${session.id}/clips/${i}`)).blob();
    if (savedUrl) URL.revokeObjectURL(savedUrl);
    savedUrl = URL.createObjectURL(blob);
    $("takePlayer").src = savedUrl;
    await $("takePlayer").play();
  } catch (e) {
    setMsg("recStatus", `Could not play the clip: ${e.message}`, true);
  }
}

function play() {
  if (take && take.index === current) {
    $("takePlayer").src = take.url;
    $("takePlayer").play();
  } else if (session.sentences[current].clip) {
    playSaved(current);
  }
}

let saving = false;
async function saveTake() {
  if (!take || take.index !== current || saving) return;
  saving = true;
  $("saveBtn").disabled = true;
  setMsg("recStatus", "Saving…");
  const form = new FormData();
  form.append("file", take.blob, `sentence-${current + 1}.wav`);
  try {
    session = await (await api(`/v1/lora/sessions/${session.id}/clips/${current}`, { method: "PUT", form })).json();
    discardTake();
    // Move on to the next sentence that still needs a recording (after this one, then from the start).
    const order = [...session.sentences.keys()].map((k) => (current + 1 + k) % session.sentences.length);
    const next = order.find((k) => !session.sentences[k].clip);
    const savedNo = current + 1;
    if (next !== undefined) current = next;
    setMsg("recStatus", next !== undefined ? `Sentence ${savedNo} saved.` : "All sentences are recorded. You can start training below.");
    render();
  } catch (e) {
    setMsg("recStatus", `Save failed: ${e.message}`, true);
    renderRecorder();
  } finally {
    saving = false;
  }
}

$("recBtn").addEventListener("click", () => (recording ? stopRecording() : startRecording()));
$("playBtn").addEventListener("click", play);
$("saveBtn").addEventListener("click", saveTake);
$("retakeBtn").addEventListener("click", startRecording);

document.addEventListener("keydown", (e) => {
  if ($("sessionView").hidden || !editable() || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.target.closest("input, textarea, select")) return;
  if (e.key === " ") {
    e.preventDefault();  // no page scroll
    document.activeElement?.blur?.();  // or the focused button would also get a click on keyup
    recording ? stopRecording() : startRecording();
  } else if (e.key === "Enter") {
    e.preventDefault();
    saveTake();
  } else if (e.key === "r" || e.key === "R") {
    startRecording();
  } else if (e.key === "ArrowLeft") {
    select(current - 1);
  } else if (e.key === "ArrowRight") {
    select(current + 1);
  }
});

// ── training ────────────────────────────────────────────────────────────────
$("trainBtn").addEventListener("click", async () => {
  $("trainBtn").disabled = true;
  setMsg("trainStatus", "");
  try {
    session = await (await api(`/v1/lora/sessions/${session.id}/train`, { method: "POST", json: { denoise: $("denoise").checked } })).json();
    releaseMic();
    render();
    schedulePoll();
  } catch (e) {
    setMsg("trainStatus", `Could not start training: ${e.message}`, true);
    renderTraining();
  }
});

$("deleteSession").addEventListener("click", async () => {
  const what = session.status === "training" ? " Training will be stopped." : "";
  if (!confirm(`Delete "${session.name}" with its recordings and model?${what}`)) return;
  try {
    await api(`/v1/lora/sessions/${session.id}`, { method: "DELETE" });
    location.hash = "";
  } catch (e) {
    alert(`Delete failed: ${e.message}`);
  }
});

let tryUrl = null;
$("trySpeak").addEventListener("click", async () => {
  const input = $("tryText").value.trim();
  if (!input) return;
  $("trySpeak").disabled = true;
  setMsg("tryStatus", "Generating… (the first request may wait while the voice model loads)");
  const t0 = performance.now();
  try {
    const blob = await (await api("/v1/audio/speech", { method: "POST", json: {
      voice: session.name, input, response_format: "wav", device: "gpu",
    } })).blob();
    if (tryUrl) URL.revokeObjectURL(tryUrl);
    tryUrl = URL.createObjectURL(blob);
    $("tryPlayer").src = tryUrl;
    $("tryPlayer").hidden = false;
    setMsg("tryStatus", `Done in ${Math.round(performance.now() - t0)} ms.`);
    await $("tryPlayer").play();
  } catch (e) {
    setMsg("tryStatus", `Error: ${e.message}`, true);
  } finally {
    $("trySpeak").disabled = false;
  }
});

// ── routing: #<session id> opens a session ──────────────────────────────────
$("back").addEventListener("click", () => { location.hash = ""; });

function route() {
  const id = location.hash.slice(1);
  if (recording) { recording = false; clearInterval(recTimer); chunks = []; }
  clearTimeout(pollTimer);
  $("listView").hidden = !!id;
  $("sessionView").hidden = !id;
  if (id) {
    openSession(id);
  } else {
    session = null;
    discardTake();
    releaseMic();
    loadList();
  }
}
window.addEventListener("hashchange", route);
route();
