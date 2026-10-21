// The exploratory image experiment (CLIP and DINOv2). Reads window.PM_VISION.
(function () {
  const V0 = window.PM_VISION;
  if (!V0) return;
  const D = { vision: V0 };
  const NS = "http://www.w3.org/2000/svg";
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (parent) parent.appendChild(e); return e; };
  const txt = (parent, attrs, s) => { const t = el("text", attrs, parent); t.textContent = s; return t; };
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const tip = document.querySelector(".tip") || (() => { const t = document.createElement("div"); t.className = "tip"; t.hidden = true; document.body.appendChild(t); return t; })();
  const show = (e, html) => { tip.innerHTML = html; tip.hidden = false; const r = tip.getBoundingClientRect(); let x = e.clientX + 14, y = e.clientY + 14; if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14; if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14; tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px"; };
  const hover = (node, html) => { node.addEventListener("mousemove", (e) => show(e, typeof html === "function" ? html() : html)); node.addEventListener("mouseleave", () => { tip.hidden = true; }); };
  const pct = (x) => Math.round(x * 100) + "%";

  function visionCounts() {
    const host = $("#v-counts"); if (!host || !D.vision) return;
    const V = D.vision.models;
    const ks = ["6", "9", "12", "15", "20"];
    const x0 = 120, x1 = 540, y0 = 40, rowH = 50;
    const H = y0 + ks.length * rowH + 30;
    const maxv = Math.max(...ks.map((k) => Math.max(V.clip.present_counts[k], V.dino.present_counts[k])));
    const X = (v) => x0 + Math.sqrt(v / maxv) * (x1 - x0);
    const svg = el("svg", { viewBox: `0 0 760 ${H}`, class: "chart", role: "img", "aria-label": "Neurons that fire on many Batman images, CLIP versus DINOv2" }, host);
    txt(svg, { x: 0, y: 18, class: "axlab" }, "Neurons firing on at least k of the 29 Batman images (bar length on a square-root scale)");
    ks.forEach((k, i) => {
      const y = y0 + i * rowH;
      txt(svg, { x: x0 - 12, y: y + 22, "text-anchor": "end", class: "barlab" }, `k = ${k}`);
      [["clip", "CLIP"], ["dino", "DINOv2"]].forEach(([m, name], j) => {
        const v = V[m].present_counts[k];
        el("rect", { x: x0, y: y + 4 + j * 18, width: Math.max(2, X(v) - x0), height: 14, rx: 3, class: j === 0 ? "bar" : "bar muted" }, svg);
        txt(svg, { x: Math.max(X(v), x0 + 2) + 6, y: y + 15 + j * 18, class: "barval" }, `${name} ${v.toLocaleString()} (chance ${V[m].chance[k]})`);
      });
    });
  }

  function visionList() {
    const host = $("#v-list"); if (!host || !D.vision) return;
    const ms = D.vision.models.clip.memes.filter((m) => m.confidence !== "low" && m.kind !== "mixed").slice(0, 14);
    host.innerHTML = `<table class="xtab"><thead><tr><th scope="col">CLIP neuron fires on</th><th scope="col">Batman images</th><th scope="col">Kind</th><th scope="col">Best DINOv2 match</th></tr></thead><tbody>${ms.map((m) =>
      `<tr><th scope="row">${esc(m.label)}${m.batman_specific ? ` <span class="minichip">Batman's own</span>` : ""}<span>${esc(m.note)}</span></th><td class="share"><span class="meter"><i style="width:${(m.n_batman / 29) * 100}%"></i></span>${m.n_batman}/29</td><td>${esc(m.kind)}</td><td class="mono">${m.dino_twin ? "r = " + m.dino_twin.r.toFixed(2) : ""}</td></tr>`).join("")}</tbody></table>`;
  }

  function probes() {
    const host = $("#v-probes"); if (!host || !D.vision) return;
    const clip = D.vision.models.clip.memes, dino = D.vision.models.dino.memes;
    const pick = [clip.find((m) => m.label === "Batman in pointed cowl"), clip.find((m) => m.label.startsWith("Masked or hooded")), dino[0]].filter(Boolean);
    const names = { cape: "Hooded figure on a gothic rooftop", skyline: "Gothic skyline at night", bats: "Bats leaving a cave", mansion: "Dark mansion on a hill", beach: "Sunny beach", kitchen: "Farmhouse kitchen", meadow: "Spring meadow" };
    host.innerHTML = Object.keys(names).map((k) => `<figure class="probe"><img src="img/probe_${k}.jpg" alt="${esc(names[k])} (generated test image)" loading="lazy"><figcaption><b>${esc(names[k])}</b>${pick.map((m, i) => {
      const v = m.probes ? m.probes[k] : null;
      return `<span class="pbar"><em>${i < 2 ? "CLIP" : "DINOv2"}: ${esc(m.label)}</em><span class="meter"><i style="width:${v ?? 0}%"></i></span><b>${v == null ? "–" : Math.round(v)}</b></span>`;
    }).join("")}</figcaption></figure>`).join("");
  }

  visionCounts(); visionList(); probes();
})();
