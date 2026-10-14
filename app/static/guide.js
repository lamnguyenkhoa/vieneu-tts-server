const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
};

// Code samples are templates: {{BASE}}, {{KEY}}, {{VOICE}}, {{DEVICE}} come from the values the
// test page saved (key, voice, device), or the server's defaults.
document.querySelectorAll(".docs pre, .docs .base").forEach((el) => { el.dataset.tpl = el.textContent; });

const vals = {
  BASE: location.origin,
  KEY: store.get("tts-key")?.trim() || "<your-key>",
  VOICE: store.get("tts-voice") || "",
  DEVICE: store.get("tts-device") || "gpu",
};

function fillDocs() {
  document.querySelectorAll("[data-tpl]").forEach((el) => {
    el.textContent = el.dataset.tpl.replace(/\{\{(\w+)\}\}/g, (_, k) => vals[k] ?? "");
  });
}

async function loadDefaults() {
  const key = store.get("tts-key")?.trim();
  const headers = key ? { "Authorization": `Bearer ${key}` } : {};
  try {
    if (!vals.VOICE) {
      const res = await fetch("/v1/voices", { headers });
      if (res.ok) vals.VOICE = (await res.json()).default || "";
    }
    if (!store.get("tts-device")) {
      const res = await fetch("/health");
      if (res.ok) vals.DEVICE = (await res.json()).default_device || vals.DEVICE;
    }
  } catch {}
  vals.VOICE ||= "Hải Đăng";
  fillDocs();
}

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

// Table of contents from the section headings; each heading gets an id (kept if it has one) and a # link.
function slug(text) {
  return text.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function buildToc() {
  const ol = document.querySelector("#toc ol");
  const used = new Set();
  document.querySelectorAll(".docs h3").forEach((h) => {
    let id = h.id || slug(h.textContent) || "section";
    for (let n = 2; used.has(id); n++) id = `${slug(h.textContent)}-${n}`;
    used.add(id);
    h.id = id;
    const self = document.createElement("a");
    self.className = "anchor";
    self.href = `#${id}`;
    self.textContent = "#";
    self.setAttribute("aria-label", `Link to ${h.textContent}`);
    h.append(" ", self);
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = `#${id}`;
    a.innerHTML = h.innerHTML.replace(/\s*<a class="anchor"[\s\S]*$/, "");  // keeps <code> in titles
    li.append(a);
    ol.append(li);
  });
  const top = document.createElement("a");
  top.href = "#";
  top.className = "to-top";
  top.textContent = "↑ Contents";
  top.addEventListener("click", (e) => { e.preventDefault(); $toc().scrollIntoView({ behavior: "smooth" }); history.replaceState(null, "", location.pathname); });
  document.body.append(top);
  const onScroll = () => { top.classList.toggle("shown", $toc().getBoundingClientRect().bottom < 0); };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();
  // the page loads before the ids exist, so jump to a #section link now
  if (location.hash) document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView();
}
const $toc = () => document.getElementById("toc");

buildToc();
fillDocs();
loadDefaults();
