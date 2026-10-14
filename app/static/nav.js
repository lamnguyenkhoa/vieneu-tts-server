// Site navigation, shared by every page: add new pages here (and their route in app/main.py PAGES).
const PAGES = [
  { href: "/", label: "Test" },
  { href: "/lora", label: "LoRA voices" },
  { href: "/lora-auto", label: "AI LoRA" },
  { href: "/guide", label: "API guide" },
];

(() => {
  const nav = document.getElementById("siteNav");
  if (!nav) return;
  const here = location.pathname.replace(/\/+$/, "") || "/";
  for (const p of PAGES) {
    const a = document.createElement("a");
    a.href = p.href;
    a.textContent = p.label;
    if (p.href === here) a.setAttribute("aria-current", "page");
    nav.append(a);
  }
})();
