const $ = (id) => document.getElementById(id);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};

$("key").value = store.get("tts-key") || "";
$("key").addEventListener("change", () => { store.set("tts-key", $("key").value); loadInfo(); route(); });

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

function fmtDuration(s) {
  s = Math.round(s);
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60), r = s % 60;
  return r ? `${m} min ${r} s` : `${m} min`;
}

function fmtSeconds(s) {
  return s >= 90 ? `${Math.round(s / 60)} min` : `${Math.max(1, Math.round(s))} s`;
}

const STATUS_LABEL = { recording: "Review", queued: "Queued", training: "Training", ready: "Ready", failed: "Failed",
  transcribing: "Transcribing" };
const EDITABLE = ["recording", "failed"];
const BUSY = ["queued", "training"];
const POLLED = [...BUSY, "transcribing"];
let info = { max_upload_mb: 500, max_minutes: 120, import_min_clips: 5 };

// ── server info: STT server, limits ─────────────────────────────────────────
async function loadInfo() {
  try {
    info = await (await api("/v1/lora/auto")).json();
    $("gpuWarning").hidden = info.gpu_available;
    setMsg("sttStatus", info.stt_available
      ? `Speech-to-text: ${info.stt_url} (${info.stt_detail})`
      : `Speech-to-text server ${info.stt_url} is not reachable (${info.stt_detail}). Start it, or set TTS_STT_URL.`,
      !info.stt_available);
    $("fileHint").textContent = `Any common audio or video format${info.ffmpeg ? "" : " (this server has no ffmpeg: WAV, FLAC, OGG or MP3 only)"}, ` +
      `up to ${info.max_upload_mb} MB; only the first ${info.max_minutes} minutes are used.`;
  } catch (e) {
    setMsg("sttStatus", `Could not reach the server: ${e.message}`, true);
  }
  updateCreate();
}

// ── list view ───────────────────────────────────────────────────────────────
async function loadList() {
  const ul = $("sessions");
  try {
    const { data } = await (await api("/v1/lora/sessions")).json();
    const mine = data.filter((s) => s.source === "auto");
    ul.innerHTML = "";
    if (!mine.length) { ul.innerHTML = '<li class="hint">None yet.</li>'; return; }
    for (const s of mine) {
      const li = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = [s.name, s.gender, s.description].filter(Boolean).join(" · ");
      const badge = document.createElement("span");
      badge.className = `badge ${s.status}`;
      badge.textContent = s.status === "recording"
        ? `${s.sentences_recorded} clips · ${fmtDuration(s.recorded_seconds)}` : STATUS_LABEL[s.status];
      const open = document.createElement("button");
      open.textContent = s.status === "recording" ? "Review" : "Open";
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

let uploading = false;
function updateCreate() {
  $("create").disabled = uploading || !$("newName").value.trim() || !$("newFile").files.length;
}
$("newName").addEventListener("input", updateCreate);
$("newFile").addEventListener("change", () => {
  const f = $("newFile").files[0];
  setMsg("createStatus", f && f.size > info.max_upload_mb * 1024 * 1024
    ? `This file is ${(f.size / 1048576).toFixed(0)} MB; the server takes up to ${info.max_upload_mb} MB. Cut it or convert it to MP3 first.` : "",
    true);
  updateCreate();
});

// XMLHttpRequest rather than fetch: fetch cannot report upload progress, and the file can be large.
function upload(form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/v1/lora/auto");
    for (const [k, v] of Object.entries(authHeaders())) xhr.setRequestHeader(k, v);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch {}
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body?.error?.message || xhr.statusText || `HTTP ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error("the connection failed"));
    xhr.send(form);
  });
}

$("create").addEventListener("click", async () => {
  const file = $("newFile").files[0];
  if (!file || uploading) return;
  uploading = true;
  updateCreate();
  const form = new FormData();
  form.append("name", $("newName").value.trim());
  form.append("gender", $("newGender").value);
  form.append("description", $("newDesc").value.trim());
  form.append("file", file, file.name);
  $("uploadProgress").hidden = false;
  $("uploadBar").style.width = "0";
  setMsg("createStatus", "Uploading…");
  try {
    const s = await upload(form, (p) => {
      $("uploadBar").style.width = `${100 * p}%`;
      setMsg("createStatus", p < 1 ? `Uploading… ${Math.round(100 * p)}%` : "Upload done, starting…");
    });
    $("newName").value = $("newDesc").value = "";
    $("newFile").value = "";
    setMsg("createStatus", "");
    location.hash = s.id;
  } catch (e) {
    setMsg("createStatus", `Could not start: ${e.message}`, true);
  } finally {
    uploading = false;
    $("uploadProgress").hidden = true;
    updateCreate();
  }
});

// ── session view ────────────────────────────────────────────────────────────
let session = null;  // GET /v1/lora/sessions/{id}
let pollTimer = null;
let filter = "all";
const editable = () => session && EDITABLE.includes(session.status);

async function openSession(id) {
  try {
    session = await (await api(`/v1/lora/sessions/${encodeURIComponent(id)}`)).json();
  } catch (e) {
    alert(`Could not open the voice: ${e.message}`);
    location.hash = "";
    return;
  }
  $("denoise").checked = session.denoise;
  setMsg("trainStatus", "");
  setMsg("reviewStatus", "");
  render();
  schedulePoll();
}

function render() {
  const s = session;
  $("sName").textContent = [s.name, s.gender, s.description].filter(Boolean).join(" · ");
  $("sBadge").className = `badge ${s.status}`;
  $("sBadge").textContent = s.status === "queued" && s.queue_position > 1
    ? `Queued (#${s.queue_position})` : STATUS_LABEL[s.status];
  $("speechBar").style.width = `${Math.min(100, (100 * s.recorded_seconds) / 1200)}%`;
  $("speechText").textContent = `${s.sentences_recorded} clips · ${fmtDuration(s.recorded_seconds)} of speech ` +
    `(at least ${info.import_min_clips} clips; 10–30 minutes works best)`;
  renderProcessing();
  renderClips();
  renderTraining();
  $("tryPanel").hidden = s.status !== "ready";
}

function renderProcessing() {
  const a = session.auto || {};
  const running = session.status === "transcribing";
  $("procPanel").hidden = !running;
  if (running) {
    let pct = 0, text = "";
    if (a.stage === "queued") text = "Waiting for another recording to finish…";
    else if (a.stage === "decoding") { pct = 3; text = "Decoding the recording…"; }
    else if (a.stage === "splitting") { pct = 8; text = `Finding the pauses in ${fmtDuration(a.source_seconds || 0)} of audio…`; }
    else if (a.stage === "transcribing") {
      pct = 10 + 90 * (a.done || 0) / (a.total || 1);
      text = `Transcribing clip ${a.done || 0} of ${a.total} · ${a.kept || 0} kept` +
        (a.no_speech ? ` · ${a.no_speech} without speech` : "") + ". You can leave this page and come back later.";
    } else text = "Starting…";
    $("procBar").style.width = `${pct}%`;
    $("procText").textContent = text;
    setMsg("procStatus", "");
    return;
  }
  const parts = [];
  if (a.source_seconds) {
    parts.push(`${session.sentences_recorded} clips from a ${fmtDuration(a.source_seconds)} recording` +
      (a.truncated ? ` (only the first ${info.max_minutes} minutes were used)` : "") + ".");
  }
  if (a.no_speech) parts.push(`${a.no_speech} piece${a.no_speech === 1 ? "" : "s"} without speech were dropped.`);
  if (a.message) parts.push(a.message);
  setMsg("procStatus", parts.join(" "), a.stage === "failed");
}

// Flags for the review: how many syllables (Vietnamese words) per second the transcript claims.
// Normal speech is about 3-6; far fewer means missed words (or music, noise), far more a likely mishearing.
function flagOf(x) {
  if (!x.clip) return null;
  const words = x.text.split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
  const rate = words / x.clip.seconds;
  if (words <= 2) return "very short transcript: is that all that is said?";
  if (rate < 1.5) return "few words for its length: words may be missing, or there is music or silence";
  if (rate > 8) return "many words for its length: check it was heard right";
  if (x.clip.peak >= 0.99) return "clipping: the audio is distorted";
  return null;
}

let playingIndex = null;
let clipUrl = null;

function renderClips() {
  const tbody = $("clipTable").tBodies[0];
  const rows = session.sentences.map((x, i) => ({ x, i, flag: flagOf(x) }));
  const flagged = rows.filter((r) => r.flag).length;
  $("countAll").textContent = `(${rows.length})`;
  $("countFlagged").textContent = `(${flagged})`;
  const shown = filter === "flagged" ? rows.filter((r) => r.flag) : rows;
  const canEdit = editable();
  $("reviewHint").hidden = !canEdit && session.status !== "transcribing";
  tbody.innerHTML = "";
  if (!shown.length) {
    const tr = tbody.insertRow();
    const td = tr.insertCell();
    td.colSpan = 5;
    td.className = "hint";
    td.textContent = session.status === "transcribing" ? "Clips appear here as they are transcribed…"
      : filter === "flagged" ? "No flagged clips." : "No clips.";
    return;
  }
  for (const { x, i, flag } of shown) {
    const tr = tbody.insertRow();
    tr.dataset.index = i;
    tr.classList.toggle("playing", playingIndex === i);
    tr.insertCell().textContent = i + 1;

    const play = document.createElement("button");
    play.className = "icon";
    play.textContent = playingIndex === i ? "■" : "▶";
    play.title = playingIndex === i ? "Stop" : "Play";
    play.addEventListener("click", () => { if (playingIndex === i) stopPlayback(); else playClip(i); });
    tr.insertCell().append(play);

    const len = tr.insertCell();
    len.className = "nowrap";
    len.textContent = x.clip ? `${x.clip.seconds.toFixed(1)} s` : "";

    const cell = tr.insertCell();
    const ta = document.createElement("textarea");
    ta.lang = "vi";
    ta.rows = 2;
    ta.value = x.text;
    ta.disabled = !canEdit;
    ta.maxLength = info.text_max_chars || 500;
    ta.addEventListener("change", () => saveText(i, ta, note));
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ta.blur(); }  // one line; Enter saves
    });
    const note = document.createElement("span");
    note.className = flag ? "row-warn" : "row-note";
    note.textContent = flag || "";
    cell.append(ta, note);

    const actions = tr.insertCell();
    if (canEdit) {
      const del = document.createElement("button");
      del.className = "icon danger";
      del.textContent = "✕";
      del.title = "Remove this clip";
      del.addEventListener("click", () => removeClip(i, del));
      actions.append(del);
    }
  }
}

document.querySelectorAll('input[name="filter"]').forEach((r) => r.addEventListener("change", () => {
  filter = document.querySelector('input[name="filter"]:checked').value;
  renderClips();
}));

async function saveText(i, ta, note) {
  const text = ta.value.replace(/\s+/g, " ").trim();
  if (!text) { ta.value = session.sentences[i].text; return; }
  if (text === session.sentences[i].text) return;
  note.className = "row-note";
  note.textContent = "Saving…";
  try {
    session = await (await api(`/v1/lora/sessions/${session.id}/sentences/${i}`, { method: "PATCH", json: { text } })).json();
    const flag = flagOf(session.sentences[i]);
    note.className = flag ? "row-warn" : "row-note";
    note.textContent = flag || "Saved.";
    $("countFlagged").textContent = `(${session.sentences.filter(flagOf).length})`;
  } catch (e) {
    note.className = "row-error";
    note.textContent = `Not saved: ${e.message}`;
  }
}

async function removeClip(i, btn) {
  btn.disabled = true;
  if (playingIndex !== null) stopPlayback();
  try {
    session = await (await api(`/v1/lora/sessions/${session.id}/clips/${i}`, { method: "DELETE" })).json();
    setMsg("reviewStatus", `Removed clip ${i + 1}; the clips after it moved up.`);
    render();
  } catch (e) {
    btn.disabled = false;
    setMsg("reviewStatus", `Could not remove clip ${i + 1}: ${e.message}`, true);
  }
}

async function playClip(i) {
  stopPlayback();
  playingIndex = i;
  markPlaying();
  try {
    const blob = await (await api(`/v1/lora/sessions/${session.id}/clips/${i}`)).blob();
    if (playingIndex !== i) return;
    if (clipUrl) URL.revokeObjectURL(clipUrl);
    clipUrl = URL.createObjectURL(blob);
    $("clipPlayer").src = clipUrl;
    await $("clipPlayer").play();
  } catch (e) {
    if (playingIndex === i) {
      playingIndex = null;
      markPlaying();
      setMsg("reviewStatus", `Could not play clip ${i + 1}: ${e.message}`, true);
    }
  }
}

function stopPlayback() {
  $("clipPlayer").pause();
  playingIndex = null;
  markPlaying();
}

// Update the play buttons in place: re-rendering would drop focus from a text box being edited.
function markPlaying() {
  for (const tr of $("clipTable").tBodies[0].rows) {
    const i = Number(tr.dataset.index);
    const btn = tr.querySelector("button.icon");
    if (!btn || Number.isNaN(i)) continue;
    tr.classList.toggle("playing", i === playingIndex);
    btn.textContent = i === playingIndex ? "■" : "▶";
    btn.title = i === playingIndex ? "Stop" : "Play";
  }
  const row = [...$("clipTable").tBodies[0].rows].find((tr) => Number(tr.dataset.index) === playingIndex);
  row?.scrollIntoView({ block: "nearest" });
}

$("clipPlayer").addEventListener("ended", () => {
  const done = playingIndex;
  playingIndex = null;
  markPlaying();
  if (done === null || !$("autoNext").checked || !session) return;
  // the next clip shown in the table (the filter may hide some)
  const rows = [...$("clipTable").tBodies[0].rows].map((tr) => Number(tr.dataset.index)).filter((i) => !Number.isNaN(i));
  const next = rows.find((i) => i > done);
  if (next !== undefined) playClip(next);
});

function renderTraining() {
  const s = session;
  const enough = s.sentences_recorded >= info.import_min_clips;
  $("trainBtn").hidden = !editable();
  $("denoise").parentElement.hidden = !editable();
  $("trainBtn").disabled = !enough;
  $("trainBtn").textContent = s.status === "failed" ? "Train again" : "Start training";
  $("trainHint").textContent = s.status === "ready"
    ? "Training finished. The voice is in the voice list of the test page; it runs on the GPU."
    : s.status === "transcribing" ? "Training can start once the recording is transcribed and you have reviewed the clips."
    : !editable() ? "Training runs on the server's GPU. You can leave this page and come back later."
    : enough ? "Review the clips first. Training runs on the server's GPU; with many clips it can take a while, and you can leave this page and come back later."
    : `At least ${info.import_min_clips} clips are needed to train (${info.import_min_clips - s.sentences_recorded} more).`;

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

function schedulePoll() {
  clearTimeout(pollTimer);
  if (!session || !POLLED.includes(session.status)) return;
  const id = session.id;
  pollTimer = setTimeout(async () => {
    if (!session || session.id !== id) return;
    try {
      const s = await (await api(`/v1/lora/sessions/${encodeURIComponent(id)}`)).json();
      if (!session || session.id !== id) return;
      const grew = s.sentences.length !== session.sentences.length || s.status !== session.status;
      session = s;
      // Re-render the clip table only when it changed, so playback and scrolling are not disturbed.
      if (grew) render();
      else { renderProcessing(); renderTraining(); $("sBadge").textContent = STATUS_LABEL[s.status]; }
      if (grew && playingIndex !== null) markPlaying();
    } catch {}
    schedulePoll();
  }, 2000);
}

// ── training ────────────────────────────────────────────────────────────────
$("trainBtn").addEventListener("click", async () => {
  $("trainBtn").disabled = true;
  setMsg("trainStatus", "");
  stopPlayback();
  try {
    session = await (await api(`/v1/lora/sessions/${session.id}/train`, { method: "POST", json: { denoise: $("denoise").checked } })).json();
    render();
    schedulePoll();
  } catch (e) {
    setMsg("trainStatus", `Could not start training: ${e.message}`, true);
    renderTraining();
  }
});

$("deleteSession").addEventListener("click", async () => {
  const what = session.status === "training" ? " Training will be stopped."
    : session.status === "transcribing" ? " Transcription will be stopped." : "";
  if (!confirm(`Delete "${session.name}" with its clips and model?${what}`)) return;
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
  stopPlayback();
  clearTimeout(pollTimer);
  $("listView").hidden = !!id;
  $("sessionView").hidden = !id;
  if (id) {
    openSession(id);
  } else {
    session = null;
    loadList();
  }
}
window.addEventListener("hashchange", route);
loadInfo();
route();
