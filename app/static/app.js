const $ = (id) => document.getElementById(id);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};

$("key").value = store.get("tts-key") || "";
$("mode").value = store.get("tts-mode") || "pcm";
$("key").addEventListener("change", () => { store.set("tts-key", $("key").value); loadVoices(); });
$("mode").addEventListener("change", () => store.set("tts-mode", $("mode").value));
$("voice").addEventListener("change", () => store.set("tts-voice", $("voice").value));

function headers() {
  const h = { "Content-Type": "application/json" };
  const key = $("key").value.trim();
  if (key) h["Authorization"] = `Bearer ${key}`;
  return h;
}

function setStatus(msg, isError = false) {
  $("status").textContent = msg;
  $("status").classList.toggle("error", isError);
}

async function errorMessage(res) {
  try { return (await res.json()).error?.message || res.statusText; } catch { return res.statusText; }
}

async function loadVoices() {
  try {
    const res = await fetch("/v1/voices", { headers: headers() });
    if (!res.ok) throw new Error(await errorMessage(res));
    const { default: def, data } = await res.json();
    const saved = store.get("tts-voice");
    $("voice").innerHTML = "";
    for (const v of data) {
      const opt = new Option(`${v.name}${v.description ? " — " + v.description : ""}`, v.id);
      $("voice").add(opt);
    }
    $("voice").value = data.some((v) => v.id === saved) ? saved : def;
    setStatus(`${data.length} voices loaded.`);
    fillDocs();
  } catch (e) {
    $("voice").innerHTML = '<option value="">(default)</option>';
    setStatus(`Could not load voices: ${e.message}`, true);
  }
}

// ── device (GPU / CPU) ──────────────────────────────────────────────────────
let devices = null;  // /health "devices": {gpu: {available, state, name}, cpu: {...}}
const deviceRadios = [...document.querySelectorAll('input[name="device"]')];
const selectedDevice = () => deviceRadios.find((r) => r.checked)?.value;

function renderDevice() {
  const el = $("device");
  const kind = selectedDevice();
  const d = devices?.[kind];
  if (!d) return;
  el.textContent = d.state === "loaded" ? d.name
    : d.state === "loading" ? `${d.name}: loading…`
    : `${d.name}: loads on first use`;
  el.className = kind;
}

async function loadDevice() {
  const el = $("device");
  try {
    const res = await fetch("/health");
    const h = await res.json();
    if (!res.ok) throw new Error(h.status || res.statusText);
    if (!h.devices) {
      // Server predates device switching: show its single device and lock the toggle.
      devices = null;
      const kind = String(h.device || "").startsWith("GPU") ? "gpu" : "cpu";
      deviceRadios.forEach((r) => {
        r.checked = r.value === kind;
        r.disabled = true;
        r.parentElement.title = "Restart the server to enable GPU/CPU switching";
      });
      el.textContent = h.device || h.backend || "unknown device";
      el.className = kind;
      el.title = "Restart the server to enable GPU/CPU switching";
      fillDocs();
      return;
    }
    devices = h.devices;
    el.title = "Device the next request will run on";
    deviceRadios.forEach((r) => {
      r.disabled = !devices[r.value]?.available;
      r.parentElement.title = r.disabled ? "No CUDA GPU on this server" : "";
    });
    if (!selectedDevice() || deviceRadios.find((r) => r.checked)?.disabled) {
      const saved = store.get("tts-device");
      const pick = deviceRadios.find((r) => r.value === saved && !r.disabled)
        || deviceRadios.find((r) => r.value === h.default_device && !r.disabled)
        || deviceRadios.find((r) => !r.disabled);
      if (pick) pick.checked = true;
    }
    renderDevice();
    fillDocs();
  } catch (e) {
    el.textContent = `server: ${e.message}`;
    el.className = "";
  }
}

deviceRadios.forEach((r) => r.addEventListener("change", () => {
  store.set("tts-device", selectedDevice());
  renderDevice();
  fillDocs();
}));

// ── playback ────────────────────────────────────────────────────────────────
let ctx = null, reader = null, controller = null, wavUrl = null;
let busy = false;     // a request is in flight
let playing = false;  // audio is still audible (Web Audio schedule or <audio> element)

function updateButtons() {
  $("speak").disabled = busy;
  $("stop").disabled = !busy && !playing;
}

function wavFromPcm(chunks, rate) {
  const len = chunks.reduce((n, c) => n + c.length, 0);
  const buf = new ArrayBuffer(44 + len), v = new DataView(buf);
  const str = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + len, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, len, true);
  let off = 44;
  for (const c of chunks) { new Uint8Array(buf, off, c.length).set(c); off += c.length; }
  return new Blob([buf], { type: "audio/wav" });
}

function showWav(blob) {
  if (wavUrl) URL.revokeObjectURL(wavUrl);
  wavUrl = URL.createObjectURL(blob);
  $("download").href = wavUrl;
  $("download").hidden = false;
  $("player").src = wavUrl;
  $("player").hidden = false;
}

async function stop() {
  const c = ctx, r = reader;
  controller?.abort();
  ctx = reader = controller = null;
  playing = false;
  updateButtons();
  try { await r?.cancel(); } catch {}
  await c?.close().catch(() => {});
  $("player").pause();
}

$("player").addEventListener("play", () => { playing = true; updateButtons(); });
$("player").addEventListener("pause", () => { playing = !!ctx; updateButtons(); });
$("player").addEventListener("ended", () => { playing = !!ctx; updateButtons(); });

async function speak() {
  const input = $("text").value.trim();
  if (!input) { setStatus("Enter some text.", true); return; }
  await stop();
  busy = true;
  updateButtons();
  $("download").hidden = true;
  $("player").hidden = true;

  const fmt = $("mode").value;
  const t0 = performance.now();
  const ms = () => Math.round(performance.now() - t0);
  controller = new AbortController();
  const device = selectedDevice();
  const dev = devices?.[device];
  setStatus(dev && dev.state !== "loaded"
    ? `Loading the ${device.toUpperCase()} model (first use; can take a minute, longer if it must download)…`
    : `Generating on ${device?.toUpperCase() ?? "default device"}…`);
  if (dev && dev.state !== "loaded") { dev.state = "loading"; renderDevice(); }
  try {
    const res = await fetch("/v1/audio/speech", {
      method: "POST", headers: headers(), signal: controller.signal,
      body: JSON.stringify({ model: "vieneu-v3-turbo", voice: $("voice").value || undefined, input, response_format: fmt, device }),
    });
    loadDevice();  // refresh: the device may have just finished loading
    if (!res.ok) throw new Error(await errorMessage(res));
    const ranOn = (res.headers.get("X-Device") || device || "").toUpperCase();

    if (fmt === "wav") {
      const blob = await res.blob();
      setStatus(`${ranOn}: got ${(blob.size / 1024).toFixed(0)} KB WAV in ${ms()} ms.`);
      showWav(blob);
      await $("player").play();
      return;
    }

    const rate = Number(res.headers.get("X-Sample-Rate")) || 48000;
    const myCtx = ctx = new AudioContext();
    reader = res.body.getReader();
    playing = true;
    let lastSrc = null;
    const kept = [];
    let nextTime = 0, carry = null, first = null, bytesIn = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done || ctx !== myCtx) break;  // finished, or Stop pressed
      let bytes = value;
      if (carry !== null) { bytes = new Uint8Array([carry, ...value]); carry = null; }
      if (bytes.length % 2) { carry = bytes[bytes.length - 1]; bytes = bytes.subarray(0, -1); }
      if (!bytes.length) continue;
      kept.push(bytes.slice());
      bytesIn += bytes.length;
      if (first === null) first = ms();

      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const buffer = myCtx.createBuffer(1, bytes.length / 2, rate);
      const ch = buffer.getChannelData(0);
      for (let i = 0; i < ch.length; i++) ch[i] = view.getInt16(i * 2, true) / 32768;
      const src = myCtx.createBufferSource();
      src.buffer = buffer;
      src.connect(myCtx.destination);
      if (nextTime < myCtx.currentTime) nextTime = myCtx.currentTime + 0.12;
      src.start(nextTime);
      lastSrc = src;
      nextTime += buffer.duration;
      setStatus(`${ranOn}: streaming… first audio after ${first} ms, ${(bytesIn / 2 / rate).toFixed(1)} s received.`);
    }
    if (ctx !== myCtx) { setStatus("Stopped."); return; }
    // The server is done but scheduled audio keeps playing; free Stop only when it ends.
    const finish = () => {
      if (ctx !== myCtx) return;  // already stopped or replaced by a newer request
      myCtx.close().catch(() => {});
      ctx = null;
      playing = false;
      updateButtons();
    };
    if (lastSrc) lastSrc.onended = finish; else finish();
    setStatus(`${ranOn} done: first audio after ${first ?? "-"} ms, total ${ms()} ms, ${(bytesIn / 2 / rate).toFixed(1)} s of audio.`);
    if (kept.length) {
      const blob = wavFromPcm(kept, rate);
      if (wavUrl) URL.revokeObjectURL(wavUrl);
      wavUrl = URL.createObjectURL(blob);
      $("download").href = wavUrl;
      $("download").hidden = false;
    }
  } catch (e) {
    if (e.name !== "AbortError") { setStatus(`Error: ${e.message}`, true); await stop(); loadDevice(); }
    else setStatus("Stopped.");
  } finally {
    busy = false;
    updateButtons();
  }
}

$("speak").addEventListener("click", speak);
$("stop").addEventListener("click", stop);
$("reloadVoices").addEventListener("click", loadVoices);
$("text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) speak(); });
// ── API docs: fill code samples, tabs, copy ─────────────────────────────────
document.querySelectorAll(".docs pre, .docs .base").forEach((el) => { el.dataset.tpl = el.textContent; });

function fillDocs() {
  const vals = {
    BASE: location.origin,
    KEY: $("key").value.trim() || "<your-key>",
    VOICE: $("voice").value || "Mai Anh",
    DEVICE: selectedDevice() || "gpu",
  };
  document.querySelectorAll("[data-tpl]").forEach((el) => {
    el.textContent = el.dataset.tpl.replace(/\{\{(\w+)\}\}/g, (_, k) => vals[k] ?? "");
  });
}
$("key").addEventListener("input", fillDocs);
$("voice").addEventListener("change", fillDocs);

document.querySelectorAll(".tabs").forEach((tabs) => {
  tabs.addEventListener("click", (e) => {
    const tab = e.target.dataset.tab;
    if (!tab) return;
    tabs.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === e.target));
    document.querySelectorAll(`.code[data-group="${tabs.dataset.group}"]`).forEach((c) => {
      c.hidden = c.dataset.tab !== tab;
    });
  });
});

document.querySelectorAll(".code .copy").forEach((btn) => {
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(btn.nextElementSibling.textContent);
      btn.textContent = "Copied";
    } catch {
      btn.textContent = "Copy failed";
    }
    setTimeout(() => { btn.textContent = "Copy"; }, 1500);
  });
});

fillDocs();
loadDevice();
loadVoices();
