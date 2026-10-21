// Renderers for the Phylomemetics write-up. Reads window.PM_DATA (built by export.py).
(function () {
  const D = window.PM_DATA;
  if (!D) return;
  const NS = "http://www.w3.org/2000/svg";
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, attrs = {}, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  };
  const txt = (parent, attrs, s) => { const t = el("text", attrs, parent); t.textContent = s; return t; };
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const works = D.works;
  const byId = Object.fromEntries(works.map((w) => [w.id, w]));
  const bat = works.filter((w) => w.group === "batman").sort((a, b) => a.year - b.year || a.label.localeCompare(b.label));
  const cous = works.filter((w) => w.group === "cousin").sort((a, b) => a.year - b.year);
  const yr = (w) => (w.group === "batman" ? "’" + String(w.year).slice(2) : String(w.year));

  const MODELS = {
    "qwen3-8b": { name: "Qwen3-8B", key: null },
    "llama3.1-8b": { name: "Llama 3.1 8B", key: "llama" },
    "qwen3-1.7b": { name: "Qwen3-1.7B", key: "qwen17" },
  };
  let model = "qwen3-8b";
  try { const m = localStorage.getItem("pm-model"); if (MODELS[m]) model = m; } catch (e) {}
  const presentFor = (m) => new Set(MODELS[model].key ? m[MODELS[model].key].present : m.present);

  // Era groups -> categorical slots in fixed, chronological order. Cousins are neutral.
  const GROUPS = ["Early (1943–66)", "Grim & gritty comics (1986–96)", "Burton & Schumacher films", "Animated films",
    "Modern comics (2002–17)", "Nolan trilogy", "Arkham games", "Recent films (2016–22)", "Cousins & ancestors"];
  const groupVar = (g) => { const i = GROUPS.indexOf(g); return i === 8 ? "var(--cousin)" : `var(--s${i + 1})`; };
  const groupName = (g) => (g === "Cousins & ancestors" ? "Relatives" : g); // display name only; the data keeps its label

  // ---------- tooltip ----------
  const tip = document.createElement("div");
  tip.className = "tip"; tip.setAttribute("role", "status"); tip.hidden = true;
  document.body.appendChild(tip);
  function showTip(evt, html) {
    tip.innerHTML = html; tip.hidden = false;
    const pad = 14, r = tip.getBoundingClientRect();
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
    tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
  }
  const hideTip = () => { tip.hidden = true; };
  const hover = (node, html) => {
    node.addEventListener("mousemove", (e) => showTip(e, typeof html === "function" ? html() : html));
    node.addEventListener("mouseleave", hideTip);
    node.addEventListener("click", (e) => showTip(e, typeof html === "function" ? html() : html)); // tap on touch screens
  };

  // ---------- headline chart A: cross-model match by meme strength ----------
  function matchChart() {
    const host = $("#chart-match"); if (!host) return;
    host.innerHTML = "";
    const tag = model === "qwen3-1.7b" ? "qwen3-1.7b" : "llama3.1-8b";
    const M = D.stats.match[tag];
    const label = (b) => (b.from === 1 ? `Strongest ${b.to}` : `${b.from.toLocaleString()}–${b.to.toLocaleString()}`);
    const rows = M.bands.map((b) => ({ label: label(b), value: b.median_r, tip: `${Math.round(b.share_over_07 * 100)}% of these memes have a twin with r above 0.7` }));
    const rowH = 30, labelW = 118, plotW = 300, top = 26, W = labelW + plotW + 48, H = top + rows.length * rowH + 40;
    const X = (v) => labelW + v * plotW;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": "Median correlation with the best-matching neuron in the second model, by meme strength" }, host);
    [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
      el("line", { x1: X(t), x2: X(t), y1: top - 6, y2: H - 30, class: "grid" }, svg);
      txt(svg, { x: X(t), y: H - 14, "text-anchor": "middle", class: "tick" }, t.toFixed(2).replace(/\.00$/, "").replace(/^0/, "0"));
    });
    txt(svg, { x: 0, y: 14, class: "axlab" }, "Qwen memes, by strength");
    rows.forEach((r, k) => {
      const y = top + k * rowH;
      txt(svg, { x: labelW - 8, y: y + 17, "text-anchor": "end", class: "barlab" }, r.label);
      el("rect", { x: labelW, y: y + 6, width: Math.max(2, X(r.value) - labelW), height: 16, rx: 3, class: "bar" }, svg);
      txt(svg, { x: X(r.value) + 6, y: y + 18, class: "barval" }, r.value.toFixed(2));
      const hit = el("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" }, svg);
      hover(hit, `<b>${esc(r.label)} memes</b><br>median r = ${r.value.toFixed(2)} with their best ${esc(MODELS[tag].name)} twin<br><span class="tip-k">${esc(r.tip)}</span>`);
    });
    const ref = (v, name, cls) => {
      el("line", { x1: X(v), x2: X(v), y1: top - 6, y2: top + rows.length * rowH, class: "ref " + cls }, svg);
    };
    ref(M.random_median, "non-meme", "ref-a");
    ref(M.shuffled_median, "shuffled", "ref-b");
    host.insertAdjacentHTML("beforeend", `<div class="keyrow"><span><i class="k-a"></i>Neurons not picked as memes: ${M.random_median.toFixed(2)}</span><span><i class="k-b"></i>Chance (sentences shuffled): ${M.shuffled_median.toFixed(2)}</span></div>`);
  }

  // ---------- headline chart B: source comic top-3 hits ----------
  function sourceChart() {
    const host = $("#chart-source"); if (!host) return;
    host.innerHTML = "";
    const S = D.stats;
    const rows = [
      { label: "Word overlap", v: S.tfidf.adaptation.top3, mr: S.tfidf.adaptation.mean_rank, main: false },
      { label: "Llama 3.1 8B memes", v: S["llama3.1-8b"].adaptation.top3, mr: S["llama3.1-8b"].adaptation.mean_rank },
      { label: "Qwen3-8B memes", v: S["qwen3-8b"].adaptation.top3, mr: S["qwen3-8b"].adaptation.mean_rank },
      { label: "Qwen3-1.7B memes", v: S["qwen3-1.7b"].adaptation.top3, mr: S["qwen3-1.7b"].adaptation.mean_rank },
    ];
    const chance = 6 * 0.25 + 3 * (1 - 36 / 66);
    const rowH = 30, labelW = 132, plotW = 270, top = 26, W = labelW + plotW + 56, H = top + rows.length * rowH + 40;
    const X = (v) => labelW + (v / 9) * plotW;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": "Adaptations whose real source comic ranked in the top three" }, host);
    [0, 3, 6, 9].forEach((t) => {
      el("line", { x1: X(t), x2: X(t), y1: top - 6, y2: H - 30, class: "grid" }, svg);
      txt(svg, { x: X(t), y: H - 14, "text-anchor": "middle", class: "tick" }, `${t}`);
    });
    txt(svg, { x: 0, y: 14, class: "axlab" }, "Real source in top 3, of 9 films");
    rows.forEach((r, k) => {
      const y = top + k * rowH;
      txt(svg, { x: labelW - 8, y: y + 17, "text-anchor": "end", class: "barlab" }, r.label);
      el("rect", { x: labelW, y: y + 6, width: Math.max(2, X(r.v) - labelW), height: 16, rx: 3, class: k === 0 ? "bar bar-alt" : "bar" }, svg);
      txt(svg, { x: X(r.v) + 6, y: y + 18, class: "barval" }, `${r.v} of 9`);
      const hit = el("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" }, svg);
      hover(hit, `<b>${esc(r.label)}</b><br>Real source ranked in the top 3 for ${r.v} of 9 adaptations<br><span class="tip-k">Average rank of the real source: ${r.mr.toFixed(1)} of 12 (chance ≈ 5.8)</span>`);
    });
    el("line", { x1: X(chance), x2: X(chance), y1: top - 6, y2: top + rows.length * rowH, class: "ref ref-b" }, svg);
    host.insertAdjacentHTML("beforeend", `<div class="keyrow"><span><i class="k-b"></i>Chance: ${chance.toFixed(1)} of 9</span></div>`);
  }

  // ---------- 1. Bat-genome ----------
  function genome() {
    const host = $("#genome"); if (!host) return;
    host.innerHTML = "";
    const memes = D.memes.slice();
    const share = (m) => { const s = presentFor(m); return bat.filter((w) => s.has(w.id)).length; };
    // order rows by Qwen (stable across toggles): canon band first, then by first appearance
    const qFirst = (m) => { const ys = bat.filter((w) => m.present.includes(w.id)).map((w) => w.year); return ys.length ? Math.min(...ys) : 9999; };
    memes.sort((a, b) => (a.band === b.band ? 0 : a.band === "canon" ? -1 : 1) || qFirst(a) - qFirst(b) || b.n_batman - a.n_batman);
    const cols = [...bat, null, ...cous];
    const cw = 17, ch = 19, labelW = 214, countW = 40, top = 156;
    const left = labelW + countW;
    const W = left + cols.length * cw + 8, H = top + memes.length * ch + 34;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "genome", role: "img", "aria-label": "Grid of memes by work. A filled cell means the meme's neuron fired for that work." }, host);
    txt(svg, { x: labelW + countW - 6, y: top - 8, "text-anchor": "end", class: "colhead" }, "of 32");
    cols.forEach((w, i) => {
      if (!w) return;
      const x = left + i * cw + cw / 2;
      const t = txt(svg, { x, y: top - 10, transform: `rotate(-62 ${x} ${top - 10})`, class: "collab" + (w.group === "cousin" ? " cousin" : "") }, `${w.label} ${yr(w)}`);
      hover(t, `<b>${esc(w.title)}</b> (${w.year})<br><span class="tip-k">${esc(groupName(w.group_label))} · ${esc(w.medium)}</span>`);
      el("rect", { x: left + i * cw + 2, y: top - 6, width: cw - 4, height: 3, rx: 1, fill: groupVar(w.group_label) }, svg);
    });
    const sepX = left + bat.length * cw + cw / 2;
    el("line", { x1: sepX, x2: sepX, y1: top - 2, y2: H - 28, class: "sep" }, svg);
    txt(svg, { x: left + (bat.length * cw) / 2, y: H - 10, "text-anchor": "middle", class: "axisnote" }, "32 Batman works, oldest → newest");
    txt(svg, { x: sepX + (cous.length * cw) / 2 + cw / 2, y: H - 10, "text-anchor": "middle", class: "axisnote" }, "Relatives");
    // gap marker between 1966 and 1986
    const gapIdx = bat.findIndex((w) => w.year > 1966);
    if (gapIdx > 0) {
      const gx = left + gapIdx * cw;
      el("line", { x1: gx, x2: gx, y1: top - 2, y2: H - 28, class: "gap" }, svg);
    }
    let lastBand = null;
    memes.forEach((m, r) => {
      const y = top + r * ch;
      if (m.band !== lastBand) {
        if (lastBand !== null) el("line", { x1: 0, x2: W - 8, y1: y - 0.5, y2: y - 0.5, class: "bandrule" }, svg);
        lastBand = m.band;
      }
      const s = presentFor(m);
      const lab = txt(svg, { x: labelW - 8, y: y + ch / 2 + 4, "text-anchor": "end", class: "rowlab" }, m.label);
      hover(lab, () => memeTip(m));
      txt(svg, { x: labelW + countW - 6, y: y + ch / 2 + 4, "text-anchor": "end", class: "rowcount" }, String(share(m)));
      const twin = MODELS[model].key ? m[MODELS[model].key] : null;
      cols.forEach((w, i) => {
        if (!w) return;
        const on = s.has(w.id);
        const cell = el("rect", { x: left + i * cw + 1, y: y + 1, width: cw - 2, height: ch - 2, rx: 2, class: on ? "cell on" : "cell" }, svg);
        hover(cell, () => `<b>${esc(m.label)}</b> in <b>${esc(w.label)}</b> (${w.year})<br><span class="tip-k">${on ? "Present" : "Absent"} according to ${esc(MODELS[model].name)}${twin ? ` (its twin of the Qwen meme, r = ${twin.r.toFixed(2)})` : ""}</span>` +
          (twin ? "" : `<div class="tip-q">Strongest sentence: “${esc(m.best_sentence[w.id])}”</div>`));
      });
    });
  }
  function memeTip(m) {
    return `<b>${esc(m.label)}</b> <span class="tip-k mono">${esc(m.addr)}</span><br><span class="tip-k">${esc(m.note)}</span>` +
      m.top.slice(0, 3).map((t) => `<div class="tip-q"><span class="tip-k">${esc(t.work)}:</span> “${esc(t.text)}”</div>`).join("");
  }

  // ---------- 2. Cousins: the Billionaire Vigilante Orphans ----------
  function cousins() {
    const host = $("#cousins"); if (!host) return;
    const pick = ["Puss in Boots", "Scarlet Pimpernel", "Zorro", "Iron Man"];
    const cw = pick.map((l) => works.find((w) => w.label === l));
    const memeNames = ["The secret identity", "The origin wound", "Heir to a dead father's fortune", "The billionaire vigilante",
      "The Batcave and bat gear", "Bats", "Lucius Fox", "Fell into a vat of chemicals"];
    const memes = memeNames.map((n) => D.memes.find((m) => m.label === n)).filter(Boolean);
    let html = `<table class="xtab"><thead><tr><th scope="col">Meme</th><th scope="col">Puss in Boots<span>1697</span></th><th scope="col">Scarlet Pimpernel<span>1905</span></th><th scope="col">Zorro<span>1940</span></th><th scope="col">Batman<span>share of 32 works</span></th><th scope="col">Iron Man<span>2008</span></th></tr></thead><tbody>`;
    memes.forEach((m) => {
      const s = presentFor(m);
      const nb = bat.filter((w) => s.has(w.id)).length;
      const cell = (w) => {
        const on = s.has(w.id);
        const q = MODELS[model].key ? "" : esc(m.best_sentence[w.id]);
        return `<td class="${on ? "yes" : "no"}" data-q="${q}" data-w="${esc(w.label)}" data-m="${esc(m.label)}">${on ? "●" : "·"}<span class="sr">${on ? "present" : "absent"}</span></td>`;
      };
      html += `<tr><th scope="row">${esc(m.label)}</th>${cell(cw[0])}${cell(cw[1])}${cell(cw[2])}<td class="share"><span class="meter"><i style="width:${(nb / 32) * 100}%"></i></span>${nb}</td>${cell(cw[3])}</tr>`;
    });
    html += "</tbody></table>";
    host.innerHTML = html;
    host.querySelectorAll("td[data-w]").forEach((td) => hover(td, () =>
      `<b>${esc(td.dataset.m)}</b> in <b>${esc(td.dataset.w)}</b>: ${td.classList.contains("yes") ? "present" : "absent"}` + (td.dataset.q ? `<div class="tip-q">Strongest sentence: “${td.dataset.q}”</div>` : "")));
  }

  // ---------- 3. Family tree ----------
  function parseNewick(s) {
    let i = 0;
    function node() {
      const n = { children: [], name: "", len: 0 };
      if (s[i] === "(") {
        i++;
        n.children.push(node());
        while (s[i] === ",") { i++; n.children.push(node()); }
        i++;
      }
      let name = "";
      while (i < s.length && !",():;".includes(s[i])) name += s[i++];
      n.name = name.trim();
      if (s[i] === ":") { i++; let num = ""; while (i < s.length && !",();".includes(s[i])) num += s[i++]; n.len = Math.max(0, parseFloat(num) || 0); }
      return n;
    }
    return node();
  }
  let treeSource = null; // null = follow the model toggle; "tfidf" = word overlap
  function tree() {
    const host = $("#tree"); if (!host) return;
    host.innerHTML = "";
    const key = treeSource || model;
    document.querySelectorAll("[data-treesrc]").forEach((b) => b.setAttribute("aria-pressed", String((b.dataset.treesrc === "model" && !treeSource) || b.dataset.treesrc === treeSource)));
    const cap = $("#tree-cap"); if (cap) cap.textContent = key === "tfidf" ? "Built from word overlap between the plot summaries." : `Built from ${D.stats[key].n_memes.toLocaleString()} memes found in ${MODELS[key].name}.`;
    const root = parseNewick(D.trees[key]);
    // Cladogram: leaves aligned on the right, each split one step further left. Shows branching order, not distance.
    const leaves = [];
    (function walk(n) { if (!n.children.length) leaves.push(n); n.children.forEach(walk); })(root);
    (function height(n) { n.h = n.children.length ? 1 + Math.max(...n.children.map(height)) : 0; return n.h; })(root);
    const maxDepth = root.h || 1;
    (function setDepth(n) { n.depth = maxDepth - n.h; n.children.forEach(setDepth); })(root);
    const rowH = 18, top = 8, labelW = 240, plotW = 330, left = 8;
    const W = left + plotW + labelW, H = top + leaves.length * rowH + 8;
    leaves.forEach((l, k) => { l.y = top + k * rowH + rowH / 2; });
    (function setY(n) { if (n.children.length) { n.children.forEach(setY); n.y = (n.children[0].y + n.children[n.children.length - 1].y) / 2; } })(root);
    const X = (d) => left + (d / maxDepth) * plotW;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "tree", role: "img", "aria-label": "Family tree of Batman works and relatives" }, host);
    (function drawN(n) {
      n.children.forEach((c) => { el("path", { d: `M${X(n.depth)},${c.y} H${X(c.depth)}`, class: "branch" }, svg); drawN(c); });
      if (n.children.length) el("path", { d: `M${X(n.depth)},${n.children[0].y} V${n.children[n.children.length - 1].y}`, class: "branch" }, svg);
    })(root);
    leaves.forEach((l) => {
      const w = byId[+l.name.replace(/^w/, "")];
      const x = X(l.depth);
      el("line", { x1: x, x2: left + plotW + 4, y1: l.y, y2: l.y, class: "leader" }, svg);
      el("circle", { cx: left + plotW + 10, cy: l.y, r: 4.5, fill: groupVar(w.group_label) }, svg);
      const t = txt(svg, { x: left + plotW + 20, y: l.y + 4, class: "leaf" + (w.group === "cousin" ? " cousin" : "") }, w.label.includes("(") ? w.label : `${w.label} (${w.year})`);
      hover(t, `<b>${esc(w.title)}</b> (${w.year})<br><span class="tip-k">${esc(groupName(w.group_label))} · ${esc(w.medium)}</span>`);
    });
  }
  function legend() {
    const host = $("#tree-legend"); if (!host) return;
    host.innerHTML = GROUPS.map((g) => `<span class="lg"><i style="background:${groupVar(g)}"></i>${esc(groupName(g))}</span>`).join("");
  }

  // ---------- 4. Source comic table ----------
  function sources() {
    const host = $("#sources"); if (!host) return;
    const S = D.stats;
    const rows = S["qwen3-8b"].adaptation.rows;
    const get = (tag, t) => S[tag].adaptation.rows.find((r) => r.target === t);
    const cell = (r) => `<td class="rank ${r.rank <= 3 ? "hit" : ""}"><b>${r.rank}</b><span>closest: ${esc(r.closest)}</span></td>`;
    let html = `<table class="xtab src"><thead><tr><th scope="col">Adaptation</th><th scope="col">Documented source</th><th scope="col">Word overlap</th><th scope="col">${esc(MODELS[model].name)} memes</th></tr></thead><tbody>`;
    rows.forEach((r) => {
      const w = works.find((x) => x.label === r.target);
      html += `<tr><th scope="row">${esc(r.target)}<span>${w ? w.year : ""}</span></th><td>${r.sources.map(esc).join(" or ")}</td>${cell(get("tfidf", r.target))}${cell(get(model, r.target))}</tr>`;
    });
    html += `</tbody></table>`;
    host.innerHTML = html;
  }

  // ---------- 5. Twins ----------
  function twins() {
    const host = $("#twins"); if (!host) return;
    const tag = model === "qwen3-1.7b" ? "qwen17" : "llama";
    const name = tag === "qwen17" ? "Qwen3-1.7B" : "Llama 3.1 8B";
    const pickNames = ["The joke", "Fell into a vat of chemicals", "The butler who raised him", "The origin wound", "The secret identity", "Commissioner Gordon"];
    host.innerHTML = pickNames.map((n) => D.memes.find((m) => m.label === n)).filter(Boolean).map((m) => {
      const t = m[tag];
      const good = t.r >= 0.7;
      return `<article class="twin"><header><h3>${esc(m.label)}</h3><span class="rpill ${good ? "good" : "weak"}">r = ${t.r.toFixed(2)}</span></header>
        <div class="twin-cols"><div><span class="eyebrow">Qwen3-8B · ${esc(m.addr)}</span>${m.top.slice(0, 2).map((s) => `<p class="q"><span>${esc(s.work)}</span>${esc(s.text)}</p>`).join("")}</div>
        <div><span class="eyebrow">${esc(name)} twin · ${esc(t.addr)}</span>${t.top.slice(0, 2).map((s) => `<p class="q"><span>${esc(s.work)}</span>${esc(s.text)}</p>`).join("")}</div></div></article>`;
    }).join("");
  }

  // ---------- 6. Stats table ----------
  function statsTable() {
    const host = $("#stats"); if (!host) return;
    const S = D.stats, f2 = (x) => x.toFixed(2), p = (x) => (x < 0.001 ? "<0.001" : x.toFixed(3));
    const rows = [["Qwen3-8B memes", S["qwen3-8b"]], ["Llama 3.1 8B memes", S["llama3.1-8b"]], ["Qwen3-1.7B memes", S["qwen3-1.7b"]]];
    let html = `<table class="xtab num"><thead><tr><th scope="col">Memes from</th><th scope="col">Memes</th><th scope="col">Era ratio (p)</th><th scope="col">Nearest neighbour same era</th><th scope="col">Time signal r (p)</th><th scope="col">Tree agreement with Qwen3-8B</th></tr></thead><tbody>`;
    rows.forEach(([n, s]) => {
      html += `<tr><th scope="row">${n}</th><td>${s.n_memes.toLocaleString()}</td><td>${f2(s.era_ratio[0])} (${p(s.era_ratio[1])})</td><td>${Math.round(s.nn_same_era * 100)}%</td><td>${f2(s.mantel_year[0])} (${p(s.mantel_year[1])})</td><td>${s.mantel_vs_primary ? f2(s.mantel_vs_primary[0]) : "–"}</td></tr>`;
    });
    html += `<tr class="sub"><th scope="row">Chance</th><td></td><td>1.00</td><td>${Math.round(S.nn_chance * 100)}%</td><td>0</td><td></td></tr></tbody></table>`;
    host.innerHTML = html;
  }

  // ---------- overlap at three levels ----------
  function overlapChart() {
    const host = $("#overlap"); if (!host) return;
    const tag = model === "qwen3-1.7b" ? "qwen3-1.7b" : "llama3.1-8b";
    const other = MODELS[tag].name;
    const M = D.stats.match[tag], A = D.stats.agreement[tag].named, T = D.stats[tag].mantel_vs_primary[0];
    // matched nulls exist for the Qwen3-8B vs Llama 3.1 8B comparison only (robustness/nulls.py)
    const R = tag === "llama3.1-8b" && window.PM2 && window.PM2.robust;
    const rows = [
      { title: "Single neurons", q: `Does ${other} have a neuron that fires on the same sentences as each Qwen meme?`,
        v: M.bands[0].median_r, fmt: (x) => x.toFixed(2), unit: "median correlation, strongest 33 memes",
        refs: R ? [[R.twin_top33.sparsity, "random neurons"], [R.twin_top33.lexical, "word-like neurons"]] : [[M.random_median, "non-meme neurons"]] },
      { title: "The meme grid", q: `For the 31 named memes, does ${other}'s twin mark the same works present or absent?`,
        v: A.kappa, fmt: (x) => x.toFixed(2), unit: "agreement beyond chance (Cohen's kappa)",
        refs: R ? [[R.grid_named31.lexical, "word-like neurons"]] : [] },
      { title: "The family tree", q: `Do the two models agree on which works are close relatives?`,
        v: T, fmt: (x) => x.toFixed(2), unit: "correlation of the two trees' distances",
        refs: R ? [[R.mantel.random_floor[0], "random neurons"]] : [] },
    ];
    host.innerHTML = rows.map((r) => `<div class="ov-row"><div class="ov-text"><b>${esc(r.title)}</b><span>${esc(r.q)}</span></div>
      <div class="ov-bar"><div class="ov-num">${r.fmt(r.v)}<small>${esc(r.unit)}</small></div>
      <div class="ov-track"><i class="ov-fill" style="width:${r.v * 100}%"></i>${r.refs.map(([x, l], i) => `<span class="ov-ref r${i}${x > 0.6 ? " flip" : ""}" style="left:${x * 100}%"><em>${esc(l)} ${r.fmt(x)}</em></span>`).join("")}</div></div></div>`).join("");
  }

  // ---------- revenge: same idea, different words ----------
  function revengeTable() {
    const host = $("#revenge"); if (!host || !D.wording) return;
    const R = D.wording.revenge, raw = R.raw;
    const unitName = (tag) => { const c = R[tag].chosen; return `L${c.layer}·${c.neuron}${c.pole}`; };
    const kinds = ["Uses “revenge”", "Uses “vengeance”", "No revenge word", "Not revenge"];
    const max = { "llama3.1-8b": Math.max(...raw["llama3.1-8b"].showcase), "qwen3-8b": Math.max(...raw["qwen3-8b"].showcase) };
    const bar = (tag, i) => { const v = raw[tag].showcase[i]; return `<td class="rv"><span class="rvbar"><i style="width:${(v / max[tag]) * 100}%"></i></span><b>${v.toFixed(2)}</b></td>`; };
    const sents = R["llama3.1-8b"].chosen.showcase;
    let html = `<table class="xtab rvtab"><thead><tr><th scope="col">Sentence</th><th scope="col">Revenge word</th><th scope="col">Llama 3.1 8B<span>revenge neuron ${unitName("llama3.1-8b")}</span></th><th scope="col">Qwen3-8B<span>revenge neuron ${unitName("qwen3-8b")}</span></th></tr></thead><tbody>`;
    const shared = ["revenge", "vengeance", "none", "none"];
    sents.forEach((s, i) => {
      html += `<tr class="${i === 3 ? "ctl" : ""}"><th scope="row"><span>${esc(kinds[i])}</span>${esc(s.text)}</th><td class="small muted">${shared[i]}</td>${bar("llama3.1-8b", i)}${bar("qwen3-8b", i)}</tr>`;
    });
    html += `</tbody></table>`;
    host.innerHTML = html;
    const L = R["llama3.1-8b"].chosen, Q = R["qwen3-8b"].chosen;
    const n = L.test_revenge_pct.length;
    const sc = $("#revenge-score");
    if (sc) sc.innerHTML = `<span><b>${L.test_pairs_won} of ${n}</b> new pairs: Llama's neuron fired more on the revenge sentence</span><span><b>${Q.test_pairs_won} of ${n}</b>: Qwen's neuron did</span><span><b>${n / 2} of ${n}</b>: a coin toss</span>`;
  }

  function rewordTable() {
    const host = $("#reworded"); if (!host || !D.wording) return;
    const rows = D.wording.memes.map((m) => ({ ...m, q: m.qwen.reworded > m.qwen.neutral_max, l: m.llama.reworded > m.llama.neutral_max }));
    const rank = (r) => (r.q && r.l ? 0 : r.q ? 1 : r.l ? 2 : 3);
    rows.sort((a, b) => rank(a) - rank(b));
    const dot = (on) => `<span class="rd ${on ? "on" : ""}" aria-label="${on ? "still fires" : "does not fire"}"></span>`;
    let html = `<table class="xtab rwtab"><thead><tr><th scope="col">Meme</th><th scope="col">Reworded sentence (almost no words in common with the original)</th><th scope="col">Qwen3-8B</th><th scope="col">Llama twin</th></tr></thead><tbody>`;
    rows.forEach((r) => {
      html += `<tr data-u="${r.unit}"><th scope="row">${esc(r.label)}</th><td class="rw"><span class="orig">${esc(r.original)}</span>${esc(r.reworded)}</td><td class="c">${dot(r.q)}</td><td class="c">${dot(r.l)}</td></tr>`;
    });
    html += `</tbody></table>`;
    host.innerHTML = html;
    host.querySelectorAll("tr[data-u]").forEach((tr) => {
      const r = rows.find((x) => String(x.unit) === tr.dataset.u);
      hover(tr, `<b>${esc(r.label)}</b><div class="tip-q"><span class="tip-k">Original:</span> “${esc(r.original)}”</div><div class="tip-q"><span class="tip-k">Reworded:</span> “${esc(r.reworded)}”</div>` +
        `<div class="tip-q"><span class="tip-k">Qwen percentile: original ${r.qwen.original.toFixed(0)}, reworded ${r.qwen.reworded.toFixed(0)}, best everyday sentence ${r.qwen.neutral_max.toFixed(0)}<br>Llama twin: original ${r.llama.original.toFixed(0)}, reworded ${r.llama.reworded.toFixed(0)}, best everyday sentence ${r.llama.neutral_max.toFixed(0)}</span></div>`);
    });
    const s = $("#reword-score");
    if (s) s.innerHTML = `<span><b>${rows.filter((r) => r.q).length} of 31</b> still fire in Qwen</span><span><b>${rows.filter((r) => r.l).length} of 31</b> in Llama</span><span><b>${rows.filter((r) => r.q && r.l).length}</b> in both</span><span><b>0</b> for word counting, by design</span>`;
  }

  // ---------- toggles ----------
  function renderModelDependent() {
    const cmp = model === "qwen3-1.7b" ? "qwen3-1.7b" : "llama3.1-8b";
    document.querySelectorAll("[data-model]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.cmp ? b.dataset.model === cmp : b.dataset.model === model)));
    document.querySelectorAll("[data-modelname]").forEach((n) => { n.textContent = MODELS[model].name; });
    document.querySelectorAll("[data-twinname]").forEach((n) => { n.textContent = model === "qwen3-1.7b" ? "Qwen3-1.7B" : "Llama 3.1 8B"; });
    genome(); cousins(); tree(); sources(); twins(); matchChart(); overlapChart();
  }
  document.querySelectorAll("[data-model]").forEach((b) => b.addEventListener("click", () => {
    model = b.dataset.model;
    try { localStorage.setItem("pm-model", model); } catch (e) {}
    renderModelDependent();
  }));
  document.querySelectorAll("[data-treesrc]").forEach((b) => b.addEventListener("click", () => {
    treeSource = b.dataset.treesrc === "tfidf" ? "tfidf" : null; tree();
  }));
  const copy = $("#copy-json");
  if (copy) copy.addEventListener("click", async () => {
    const s = JSON.stringify(D, null, 1);
    try { await navigator.clipboard.writeText(s); copy.textContent = "Copied"; }
    catch (e) { const ta = $("#json-fallback"); ta.hidden = false; ta.value = s; ta.select(); copy.textContent = "Select all and copy"; }
    setTimeout(() => { copy.textContent = "Copy all data as JSON"; }, 2500);
  });

  legend(); sourceChart(); statsTable(); revengeTable(); rewordTable(); renderModelDependent();
})();
