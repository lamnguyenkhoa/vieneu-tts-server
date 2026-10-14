const $ = (id) => document.getElementById(id);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};

$("key").value = store.get("tts-key") || "";
$("key").addEventListener("change", () => { store.set("tts-key", $("key").value); loadVoices(); });

function authHeaders() {
  const key = $("key").value.trim();
  return key ? { "Authorization": `Bearer ${key}` } : {};
}

async function api(path, { method = "GET", json } = {}) {
  const headers = authHeaders();
  let body;
  if (json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(json); }
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

let voices = [];               // cloned and LoRA voices from GET /v1/voices
const selected = new Set();    // ids ticked for "Delete selected"
let busy = false;              // a delete is running

async function loadVoices() {
  try {
    const { data } = await (await api("/v1/voices")).json();
    voices = data.filter((v) => v.custom)
      .sort((a, b) => (b.created_at || "").localeCompare(a.created_at || "") || a.name.localeCompare(b.name, "vi"));
    for (const id of [...selected]) if (!voices.some((v) => v.id === id)) selected.delete(id);
    render();
  } catch (e) {
    voices = [];
    $("voices").innerHTML = "";
    setMsg("status", `Could not load voices: ${e.message}`, true);
  }
}

function matches(v) {
  const q = $("search").value.trim().toLocaleLowerCase("vi");
  return !q || [v.name, v.description, v.gender].some((s) => (s || "").toLocaleLowerCase("vi").includes(q));
}

function fmtDate(iso) {
  if (!iso) return "";
  // created_at is "2026-10-14T09:30:00+0700"; Date needs the offset as +07:00
  const d = new Date(iso.replace(/([+-]\d\d)(\d\d)$/, "$1:$2"));
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function render() {
  const ul = $("voices");
  const shown = voices.filter(matches);
  $("count").textContent = voices.length ? `(${voices.length})` : "";
  ul.innerHTML = "";
  if (!voices.length) ul.innerHTML = '<li class="hint">No cloned or LoRA voices yet.</li>';
  else if (!shown.length) ul.innerHTML = '<li class="hint">No voice matches the search.</li>';
  for (const v of shown) {
    const li = document.createElement("li");

    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = selected.has(v.id);
    check.setAttribute("aria-label", `Select ${v.name}`);
    check.addEventListener("change", () => {
      if (check.checked) selected.add(v.id); else selected.delete(v.id);
      updateTools();
    });

    const info = document.createElement("span");
    info.className = "grow voice-info";
    const name = document.createElement("span");
    const strong = document.createElement("strong");
    strong.textContent = v.name;
    const kind = document.createElement("span");
    kind.className = `badge ${v.lora ? "ready" : ""}`;
    kind.textContent = v.lora ? "LoRA" : "Cloned";
    name.append(strong, " ", kind);
    const meta = document.createElement("span");
    meta.className = "hint";
    meta.textContent = [v.gender, v.description, fmtDate(v.created_at)].filter(Boolean).join(" · ");
    info.append(name, meta);

    const clip = document.createElement("button");
    clip.textContent = "Clip";
    clip.title = "Play the recording this voice was cloned from";
    clip.disabled = !v.has_clip;
    clip.addEventListener("click", () => play(`/v1/voices/${encodeURIComponent(v.id)}/clip`, `Reference clip of "${v.name}"`, clip));
    const open = document.createElement("a");
    open.className = "button";
    open.textContent = "Open";
    open.title = "Open its training session";
    open.href = v.lora_session ? `/${v.lora_auto ? "lora-auto" : "lora"}#${v.lora_session}` : "#";

    const tryBtn = document.createElement("button");
    tryBtn.textContent = "Try";
    tryBtn.addEventListener("click", () => trySpeak(v, tryBtn));

    const del = document.createElement("button");
    del.className = "danger";
    del.textContent = "Delete";
    del.addEventListener("click", () => removeVoices([v.id]));

    const actions = document.createElement("span");
    actions.className = "item-actions";
    actions.append(v.lora ? open : clip, tryBtn, del);
    li.append(check, info, actions);
    ul.append(li);
  }
  updateTools();
}

function updateTools() {
  const shown = voices.filter(matches);
  const n = selected.size;
  $("deleteSelected").textContent = n ? `Delete selected (${n})` : "Delete selected";
  $("deleteSelected").disabled = busy || !n;
  $("selectAll").disabled = busy || !shown.length;
  $("selectAll").checked = shown.length > 0 && shown.every((v) => selected.has(v.id));
  $("selectAll").indeterminate = !$("selectAll").checked && shown.some((v) => selected.has(v.id));
}

$("search").addEventListener("input", render);
$("selectAll").addEventListener("change", () => {
  for (const v of voices.filter(matches)) {
    if ($("selectAll").checked) selected.add(v.id); else selected.delete(v.id);
  }
  render();
});
$("deleteSelected").addEventListener("click", () => removeVoices([...selected]));

async function removeVoices(ids) {
  if (!ids.length || busy) return;
  const lora = ids.filter((id) => voices.find((v) => v.id === id)?.lora).length;
  const what = ids.length === 1 ? `the ${lora ? "LoRA" : "cloned"} voice "${ids[0]}"` : `${ids.length} voices`;
  const extra = lora ? ` ${lora === ids.length && lora === 1 ? "Its" : `${lora} of them are LoRA voices; their`} training clips and model are deleted too.` : "";
  if (!confirm(`Delete ${what}?${extra} This cannot be undone.`)) return;
  busy = true;
  updateTools();
  const failed = [];
  for (const [n, id] of ids.entries()) {
    if (ids.length > 1) setMsg("status", `Deleting ${n + 1} of ${ids.length}…`);
    try {
      await api(`/v1/voices/${encodeURIComponent(id)}`, { method: "DELETE" });
      selected.delete(id);
    } catch (e) {
      failed.push(`${id}: ${e.message}`);
    }
  }
  busy = false;
  await loadVoices();
  const done = ids.length - failed.length;
  setMsg("status", failed.length
    ? `Deleted ${done} of ${ids.length}. Failed: ${failed.join("; ")}`
    : ids.length === 1 ? `Deleted "${ids[0]}".` : `Deleted ${done} voices.`, failed.length > 0);
}

// ── listening ───────────────────────────────────────────────────────────────
let playUrl = null;

async function play(path, label, btn, options) {
  btn.disabled = true;
  setMsg("playStatus", `Loading ${label}…`);
  const t0 = performance.now();
  try {
    const blob = await (await api(path, options)).blob();
    if (playUrl) URL.revokeObjectURL(playUrl);
    playUrl = URL.createObjectURL(blob);
    $("player").src = playUrl;
    $("player").hidden = false;
    setMsg("playStatus", `${label} (${Math.round(performance.now() - t0)} ms).`);
    await $("player").play();
  } catch (e) {
    setMsg("playStatus", `Could not play ${label}: ${e.message}`, true);
  } finally {
    btn.disabled = false;
  }
}

function trySpeak(v, btn) {
  const input = $("tryText").value.trim();
  if (!input) { setMsg("playStatus", "Write some text to speak first.", true); return; }
  const json = { voice: v.id, input, response_format: "wav" };
  if (v.lora) json.device = "gpu";  // LoRA voices run on the GPU only
  play("/v1/audio/speech", `"${v.name}"`, btn, { method: "POST", json });
}

loadVoices();
