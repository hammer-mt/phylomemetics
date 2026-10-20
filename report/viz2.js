// Phase 1 additions (canon, spread), Phase 2 (experiments), Phase 3 (images), and tabs. Reads window.PM2 + window.PM_DATA.
(function () {
  const D = window.PM2, P1 = window.PM_DATA;
  if (!D || !P1) return;
  const NS = "http://www.w3.org/2000/svg";
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (parent) parent.appendChild(e); return e; };
  const txt = (parent, attrs, s) => { const t = el("text", attrs, parent); t.textContent = s; return t; };
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const W = Object.fromEntries(P1.works.map((w) => [w.id, w]));
  const tip = document.querySelector(".tip") || (() => { const t = document.createElement("div"); t.className = "tip"; t.hidden = true; document.body.appendChild(t); return t; })();
  const show = (e, html) => { tip.innerHTML = html; tip.hidden = false; const r = tip.getBoundingClientRect(); let x = e.clientX + 14, y = e.clientY + 14; if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14; if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14; tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px"; };
  const hover = (node, html) => { node.addEventListener("mousemove", (e) => show(e, typeof html === "function" ? html() : html)); node.addEventListener("mouseleave", () => { tip.hidden = true; }); };
  const pct = (x) => Math.round(x * 100) + "%";
  const MODEL = { "qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama 3.1 8B" };

  // ---------------- tabs ----------------
  function tabs() {
    const btns = document.querySelectorAll("[data-tab]");
    const panels = document.querySelectorAll("[data-panel]");
    const open = (id, push) => {
      btns.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === id)));
      panels.forEach((p) => { p.hidden = p.dataset.panel !== id; });
      if (push) { try { history.replaceState(null, "", "#" + id); } catch (e) {} }
      try { localStorage.setItem("pm-tab", id); } catch (e) {}
    };
    btns.forEach((b) => b.addEventListener("click", () => open(b.dataset.tab, true)));
    document.querySelectorAll("[data-goto]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); open(a.dataset.goto, true); document.getElementById("tabs").scrollIntoView(); }));
    const h = (location.hash || "").replace("#", "");
    let start = ["p1", "p2", "p3"].includes(h) ? h : "p2";
    open(start, false);
  }

  // ---------------- Phase 1: canon ----------------
  function canonTiers() {
    const host = $("#canon-tiers"); if (!host) return;
    const t = D.canon.tiers;
    const rows = [
      ["Must have", "in at least 29 of 32 works", t["core|specific"] || 0, t["core|shared"] || 0],
      ["Usually there", "in 23 to 28 works", t["common|specific"] || 0, t["common|shared"] || 0],
      ["Optional", "in 6 to 22 works", t["optional|specific"] || 0, t["optional|shared"] || 0],
    ];
    host.innerHTML = rows.map(([name, sub, own, shared]) => {
      const tot = own + shared;
      return `<div class="tier"><div class="tier-h"><b>${name}</b><span>${sub}</span></div><div class="tier-n">${tot.toLocaleString()}<small>memes</small></div>
        <div class="tier-bar" role="img" aria-label="${own} Batman's own, ${shared} shared"><i style="width:${(own / tot) * 100}%"></i></div>
        <div class="tier-key"><span><i class="own"></i>${own.toLocaleString()} Batman's own</span><span><i class="shared"></i>${shared.toLocaleString()} shared with other heroes</span></div></div>`;
    }).join("");
  }

  function canonChips() {
    const host = $("#canon-chips"); if (!host) return;
    const chips = D.canon.chips;
    const col = (title, sub, list) => `<div class="cchip-col"><h3>${title}</h3><p class="small muted">${sub}</p><div class="cchips">${list.map((c) =>
      `<span class="cchip ${c.specific ? "own" : "shared"}" data-u="${c.unit}"><b>${esc(c.label)}</b><small>${c.n_batman}/32</small></span>`).join("")}</div></div>`;
    const core = chips.filter((c) => c.tier === "core");
    const common = chips.filter((c) => c.tier === "common");
    const opt = chips.filter((c) => c.tier === "optional" && c.kind !== "register").slice(0, 18);
    host.innerHTML = col("Must have", "In 29 or more of the 32 works", core) + col("Usually there", "In 23 to 28 works", common) + col("Optional", "A few of the 14,593: in 6 to 22 works", opt);
    host.querySelectorAll(".cchip").forEach((n) => {
      const c = chips.find((x) => String(x.unit) === n.dataset.u);
      hover(n, `<b>${esc(c.label)}</b><br><span class="tip-k">${esc(c.kind)} · in ${c.n_batman} of 32 Batman works · ${c.n_cousin} of 13 cousins · ${c.specific ? "Batman's own" : "shared with other heroes"}</span>${c.note ? `<div class="tip-q">${esc(c.note)}</div>` : ""}`);
    });
  }

  function completeness() {
    const host = $("#canon-time"); if (!host) return;
    const rows = D.canon.per_work;
    const Wd = 520, H = 300, x0 = 48, x1 = 505, y0 = 40, y1 = 250;
    const X = (yr) => x0 + (yr - 1940) / (2025 - 1940) * (x1 - x0), Y = (v) => y1 - (v - 0.5) / 0.5 * (y1 - y0);
    const svg = el("svg", { viewBox: `0 0 ${Wd} ${H}`, class: "chart", role: "img", "aria-label": "Share of the core canon carried by each Batman work, by year" }, host);
    [0.5, 0.6, 0.7, 0.8, 0.9, 1].forEach((v) => { el("line", { x1: x0, x2: x1, y1: Y(v), y2: Y(v), class: "grid" }, svg); txt(svg, { x: x0 - 8, y: Y(v) + 4, "text-anchor": "end", class: "tick" }, pct(v)); });
    [1940, 1960, 1980, 2000, 2020].forEach((yr) => txt(svg, { x: X(yr), y: y1 + 20, "text-anchor": "middle", class: "tick" }, String(yr)));
    el("rect", { x: X(1967), y: y0, width: X(1985) - X(1967), height: y1 - y0, class: "gapband" }, svg);
    txt(svg, { x: (X(1967) + X(1985)) / 2, y: y0 + 16, "text-anchor": "middle", class: "tick" }, "no works in corpus");
    txt(svg, { x: 0, y: 22, class: "axlab" }, "Share of the 56 must-have memes present");
    const lowLabels = new Set(["Joker (2019)", "The Killing Joke", "1943 serial", "Batman '66", "Arkham Asylum (comic)"]);
    rows.forEach((r) => {
      const c = el("circle", { cx: X(r.year), cy: Y(r.core_share), r: 5, class: r.core_share < 0.85 ? "dot accent" : "dot" }, svg);
      hover(c, `<b>${esc(r.label)}</b> (${r.year})<br>${pct(r.core_share)} of the must-have memes`);
      if (lowLabels.has(r.label)) txt(svg, r.year > 1990 ? { x: X(r.year) - 9, y: Y(r.core_share) + 4, class: "ptlab", "text-anchor": "end" } : { x: X(r.year) + 9, y: Y(r.core_share) + 4, class: "ptlab" }, `${r.label} ${pct(r.core_share)}`);
    });
  }

  function swimlane() {
    const host = $("#spread"); if (!host) return;
    const rows = D.swim;
    const rowH = 26, x0 = 230, x1 = 740, y0 = 40;
    const H = y0 + rows.length * rowH + 40;
    const X = (yr) => x0 + (yr - 1940) / (2025 - 1940) * (x1 - x0);
    const svg = el("svg", { viewBox: `0 0 760 ${H}`, width: 760, height: H, class: "swim", role: "img", "aria-label": "When each meme appears across Batman works, 1943 to 2022" }, host);
    el("rect", { x: X(1967), y: y0 - 10, width: X(1985) - X(1967), height: rows.length * rowH + 10, class: "gapband" }, svg);
    [1940, 1960, 1980, 2000, 2020].forEach((yr) => { el("line", { x1: X(yr), x2: X(yr), y1: y0 - 10, y2: y0 + rows.length * rowH, class: "grid" }, svg); txt(svg, { x: X(yr), y: y0 - 16, "text-anchor": "middle", class: "tick" }, String(yr)); });
    rows.forEach((r, i) => {
      const y = y0 + i * rowH + rowH / 2;
      txt(svg, { x: x0 - 12, y: y + 4, "text-anchor": "end", class: "rowlab" }, r.label);
      const yrs = r.batman.map((id) => W[id].year).sort((a, b) => a - b);
      if (yrs.length) el("line", { x1: X(yrs[0]), x2: X(yrs.at(-1)), y1: y, y2: y, class: "lifeline" }, svg);
      r.batman.forEach((id) => {
        const w = W[id];
        const first = w.year === yrs[0];
        const c = el("circle", { cx: X(w.year), cy: y, r: first ? 6 : 4.5, class: first ? "dot accent" : "dot" }, svg);
        hover(c, `<b>${esc(r.label)}</b> in <b>${esc(w.label)}</b> (${w.year})${first ? "<br><span class='tip-k'>first appearance in this corpus</span>" : ""}`);
      });
    });
  }

  function cousinsSpread() {
    const host = $("#cousin-spread"); if (!host) return;
    const rows = D.canon.per_cousin;
    const rowH = 26, x0 = 175, x1 = 465, y0 = 30;
    const H = y0 + rows.length * rowH + 30;
    const X = (v) => x0 + v * (x1 - x0);
    const svg = el("svg", { viewBox: `0 0 520 ${H}`, class: "chart", role: "img", "aria-label": "Share of Batman's must-have memes found in each relative" }, host);
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => { el("line", { x1: X(v), x2: X(v), y1: y0 - 6, y2: H - 24, class: "grid" }, svg); txt(svg, { x: X(v), y: H - 8, "text-anchor": "middle", class: "tick" }, pct(v)); });
    txt(svg, { x: 0, y: 16, class: "axlab" }, "Share of Batman's 56 must-have memes");
    rows.forEach((r, i) => {
      const y = y0 + i * rowH;
      const before = r.year < 1939;
      txt(svg, { x: x0 - 10, y: y + 16, "text-anchor": "end", class: "barlab" }, `${r.label.replace(/ \(1926\)$/, "")} (${r.year})`);
      const b = el("rect", { x: x0, y: y + 5, width: Math.max(2, X(r.core_share) - x0), height: 15, rx: 3, class: before ? "bar muted" : "bar" }, svg);
      txt(svg, { x: X(r.core_share) + 6, y: y + 17, class: "barval" }, pct(r.core_share));
      hover(b, `<b>${esc(r.label)}</b> (${r.year})<br>${pct(r.core_share)} of Batman's must-have memes, ${pct(r.core_specific_share)} of the ones that are Batman's own`);
    });
    const yBreak = y0 + rows.findIndex((r) => r.year >= 1939) * rowH;
    el("line", { x1: 0, x2: 520, y1: yBreak, y2: yBreak, class: "sep-dash" }, svg);
    txt(svg, { x: 515, y: yBreak - 6, "text-anchor": "end", class: "tick" }, "Batman created, 1939");
  }

  // ---------------- Phase 2 ----------------
  function e1Auc() {
    const host = $("#e1-auc"); if (!host) return;
    const groups = [["held_out_films", "Ordinary films (held out)"], ["literary_cousins", "Literary ancestors"], ["hero_film_cousins", "Superhero and masked-hero films"]];
    const tags = ["qwen3-8b", "llama3.1-8b"];
    const x0 = 250, x1 = 670, y0 = 40, rowH = 56;
    const H = y0 + groups.length * rowH + 40;
    const X = (v) => x0 + v * (x1 - x0);
    const svg = el("svg", { viewBox: `0 0 760 ${H}`, class: "chart", role: "img", "aria-label": "How well name-free canon memes tell Batman from each comparison group" }, host);
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => { el("line", { x1: X(v), x2: X(v), y1: y0 - 10, y2: H - 30, class: "grid" }, svg); txt(svg, { x: X(v), y: H - 12, "text-anchor": "middle", class: "tick" }, v.toFixed(2)); });
    el("line", { x1: X(0.5), x2: X(0.5), y1: y0 - 10, y2: H - 30, class: "ref ref-b" }, svg);
    el("line", { x1: X(0.8), x2: X(0.8), y1: y0 - 10, y2: H - 30, class: "ref ref-a" }, svg);
    txt(svg, { x: X(0.5), y: y0 - 16, "text-anchor": "middle", class: "tick" }, "chance");
    txt(svg, { x: X(0.8), y: y0 - 16, "text-anchor": "middle", class: "tick" }, "pre-registered bar");
    groups.forEach(([key, name], i) => {
      const yc = y0 + i * rowH;
      txt(svg, { x: x0 - 12, y: yc + 22, "text-anchor": "end", class: "barlab" }, name);
      tags.forEach((tag, k) => {
        const r = D.experiments[tag].E1.A_name_free[key];
        const y = yc + 12 + k * 18;
        el("line", { x1: X(r.ci95[0]), x2: X(r.ci95[1]), y1: y, y2: y, class: "whisker" }, svg);
        const c = el("circle", { cx: X(r.auc), cy: y, r: 6, class: k === 0 ? "dot accent" : "dot ring" }, svg);
        txt(svg, { x: 690, y: y + 4, class: "barval" }, `${k === 0 ? "Qwen" : "Llama"} ${r.auc.toFixed(2)}`);
        hover(c, `<b>${MODEL[tag]}</b> vs ${esc(name.toLowerCase())}<br>AUC ${r.auc.toFixed(2)} (95% CI ${r.ci95[0].toFixed(2)} to ${r.ci95[1].toFixed(2)})`);
      });
    });
  }

  function e1Cousins() {
    const host = $("#e1-cousins"); if (!host) return;
    const q = D.experiments["qwen3-8b"].E1.per_cousin_beaten_by, l = D.experiments["llama3.1-8b"].E1.per_cousin_beaten_by;
    const names = Object.keys(q).sort((a, b) => (q[b] + l[b]) - (q[a] + l[a]));
    const x0 = 130, x1 = 500, y0 = 34, rowH = 28;
    const H = y0 + names.length * rowH + 34;
    const X = (v) => x0 + v * (x1 - x0);
    const svg = el("svg", { viewBox: `0 0 520 ${H}`, class: "chart", role: "img", "aria-label": "Share of name-free Batman stories each hero film matches or beats on Batman's canon" }, host);
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => { el("line", { x1: X(v), x2: X(v), y1: y0 - 8, y2: H - 26, class: "grid" }, svg); txt(svg, { x: X(v), y: H - 8, "text-anchor": "middle", class: "tick" }, pct(v)); });
    el("line", { x1: X(0.5), x2: X(0.5), y1: y0 - 8, y2: H - 26, class: "ref ref-b" }, svg);
    txt(svg, { x: 0, y: 16, class: "axlab" }, "Name-free Batman stories this film matches or beats");
    names.forEach((n, i) => {
      const y = y0 + i * rowH + 12;
      txt(svg, { x: x0 - 12, y: y + 4, "text-anchor": "end", class: "barlab" }, n.replace(/ \(1926\)$/, " (1926)"));
      el("line", { x1: X(Math.min(q[n], l[n])), x2: X(Math.max(q[n], l[n])), y1: y, y2: y, class: "whisker" }, svg);
      const a = el("circle", { cx: X(q[n]), cy: y, r: 6, class: "dot accent" }, svg);
      const b = el("circle", { cx: X(l[n]), cy: y, r: 6, class: "dot ring" }, svg);
      hover(a, `<b>${esc(n)}</b><br>Qwen3-8B: matches or beats ${pct(q[n])} of name-free Batman stories`);
      hover(b, `<b>${esc(n)}</b><br>Llama 3.1 8B: matches or beats ${pct(l[n])} of name-free Batman stories`);
    });
  }

  function e1Core() {
    const host = $("#e1-core"); if (!host) return;
    const c = D.experiments["qwen3-8b"].E1.core, cl = D.experiments["llama3.1-8b"].E1.core;
    host.innerHTML = `<div class="corelist">${D.experiments.core_labels.filter((x) => x.label !== "No clear theme").map((x) => `<span class="cchip shared"><b>${esc(x.label)}</b></span>`).join("")}</div>
      <p class="small muted">Qwen3-8B: ${c.n_core_units} of ${c.n_canon_units} canon neurons survive without names (Llama 3.1 8B: ${cl.n_core_units} of ${cl.n_canon_units}). They are present in ${pct(c.core_present_share_in_hero_cousins)} of hero-film cousins and ${pct(c.core_present_share_in_films)} of ordinary films. The same procedure finds no name-free core at all for the cousins as a group or for random sets of 32 ordinary films.</p>`;
  }

  function e2Ranks() {
    const host = $("#e2-ranks"); if (!host) return;
    const q = D.experiments["qwen3-8b"].E2.primary.rows, l = D.experiments["llama3.1-8b"].E2.primary.rows;
    const x0 = 290, x1 = 700, y0 = 40, rowH = 30;
    const H = y0 + q.length * rowH + 40;
    const svg = el("svg", { viewBox: `0 0 760 ${H}`, class: "chart", role: "img", "aria-label": "Rank of the documented source comic among eligible comics, for each adaptation" }, host);
    txt(svg, { x: x0, y: 18, class: "axlab" }, "Where the real source ranks among eligible comics (left = closest)");
    q.forEach((r, i) => {
      const y = y0 + i * rowH + 10;
      const lr = l.find((x) => x.target === r.target);
      const n = r.n_comics, step = (x1 - x0) / 12;
      txt(svg, { x: x0 - 12, y: y + 4, "text-anchor": "end", class: "barlab" }, `${r.target} ← ${r.sources.join(" / ")}`);
      for (let k = 1; k <= n; k++) el("rect", { x: x0 + (k - 1) * step + 2, y: y - 8, width: step - 4, height: 16, rx: 3, class: "slot" }, svg);
      const a = el("circle", { cx: x0 + (r.rank - 0.5) * step, cy: y, r: 6, class: "dot accent" }, svg);
      const b = el("circle", { cx: x0 + (lr.rank - 0.5) * step, cy: y, r: 9, class: "dot ring" }, svg);
      hover(a, `<b>${esc(r.target)}</b><br>Qwen3-8B ranks ${r.sources.join(" or ")} ${r.rank} of ${n} eligible comics`);
      hover(b, `<b>${esc(r.target)}</b><br>Llama 3.1 8B ranks ${lr.sources.join(" or ")} ${lr.rank} of ${n} eligible comics`);
    });
    txt(svg, { x: x0, y: H - 12, class: "tick" }, "1st");
    txt(svg, { x: x1, y: H - 12, "text-anchor": "end", class: "tick" }, "12th");
  }

  function e2Sens() {
    const host = $("#e2-sens"); if (!host) return;
    const names = { original_upper_bound: "With names (upper bound)", raw_rate: "No hub adjustment", no_time_rule: "Later comics allowed", strong_subset: "Strong attributions only (5 films)",
      without_killing_joke: "Without the 3 Killing Joke adaptations", band_3_16: "Optional = 3 to 16 works", band_4_16: "Optional = 4 to 16 works", band_2_8: "Optional = 2 to 8 works" };
    const q = D.experiments["qwen3-8b"].E2, l = D.experiments["llama3.1-8b"].E2;
    const p = (x) => (x < 0.001 ? "<0.001" : x.toFixed(3));
    let html = `<table class="xtab num"><thead><tr><th scope="col">Variant</th><th scope="col">Qwen3-8B: comics beating the source (chance)</th><th scope="col">p</th><th scope="col">Llama 3.1 8B</th><th scope="col">p</th></tr></thead><tbody>`;
    html += `<tr><th scope="row"><b>Primary (name-free)</b></th><td>${pct(q.primary.mean_share_beating_source)} (${pct(q.primary.null_mean)})</td><td>${p(q.primary.p_one_sided)}</td><td>${pct(l.primary.mean_share_beating_source)} (${pct(l.primary.null_mean)})</td><td>${p(l.primary.p_one_sided)}</td></tr>`;
    Object.entries(names).forEach(([k, n]) => { html += `<tr><th scope="row">${n}</th><td>${pct(q.sens[k].stat)} (${pct(q.sens[k].null)})</td><td>${p(q.sens[k].p)}</td><td>${pct(l.sens[k].stat)} (${pct(l.sens[k].null)})</td><td>${p(l.sens[k].p)}</td></tr>`; });
    html += `<tr class="sub"><th scope="row">Placebo films (non-adaptations given the same sources)</th><td>${pct(q.primary.placebo_mean)}</td><td></td><td>${pct(l.primary.placebo_mean)}</td><td></td></tr></tbody></table>`;
    host.innerHTML = html;
    const s = $("#e2-seq");
    if (s) s.innerHTML = `<table class="xtab num"><thead><tr><th scope="col">Sequel</th><th scope="col">Its prequel</th><th scope="col">Qwen3-8B rank</th><th scope="col">Llama 3.1 8B rank</th></tr></thead><tbody>${q.sequels.rows.map((r, i) => { const lr = l.sequels.rows[i]; return `<tr><th scope="row">${esc(r.sequel)}</th><td>${esc(r.prequel)}</td><td class="${r.rank <= 2 ? "hitc" : ""}">${r.rank} of ${r.n}</td><td class="${lr.rank <= 2 ? "hitc" : ""}">${lr.rank} of ${lr.n}</td></tr>`; }).join("")}</tbody></table>`;
  }

  function e3Scatter() {
    const host = $("#e3-scatter"); if (!host) return;
    const P = D.experiments["qwen3-8b"].E3.primary;
    const rows = P.works;
    const xs = rows.map((r) => Math.log10(r.monthly_views)), ys = rows.map((r) => r.offspring);
    const xmin = Math.min(...xs) - 0.1, xmax = Math.max(...xs) + 0.1, ymax = Math.max(...ys.map(Math.abs)) * 1.15;
    const x0 = 70, x1 = 720, y0 = 40, y1 = 330;
    const X = (v) => x0 + (v - xmin) / (xmax - xmin) * (x1 - x0), Y = (v) => (y0 + y1) / 2 - v / ymax * (y1 - y0) / 2;
    const svg = el("svg", { viewBox: "0 0 760 380", class: "chart", role: "img", "aria-label": "Popularity of early Batman works against how much of their memes spread later" }, host);
    el("line", { x1: x0, x2: x1, y1: Y(0), y2: Y(0), class: "ref ref-b" }, svg);
    [3, 3.5, 4, 4.5, 5].filter((v) => v > xmin && v < xmax).forEach((v) => { el("line", { x1: X(v), x2: X(v), y1: y0, y2: y1, class: "grid" }, svg); txt(svg, { x: X(v), y: y1 + 18, "text-anchor": "middle", class: "tick" }, Math.round(10 ** v).toLocaleString()); });
    txt(svg, { x: (x0 + x1) / 2, y: y1 + 40, "text-anchor": "middle", class: "tick" }, "Wikipedia views per month (log scale)");
    txt(svg, { x: x0, y: 22, class: "axlab" }, "Spread of its memes into later works, above or below expectation");
    rows.forEach((r, i) => {
      const c = el("circle", { cx: X(xs[i]), cy: Y(ys[i]), r: 6, class: "dot" }, svg);
      hover(c, `<b>${esc(r.work)}</b> (${r.year})<br>${r.monthly_views.toLocaleString()} views a month<br><span class="tip-k">offspring score ${r.offspring > 0 ? "+" : ""}${(r.offspring * 1000).toFixed(2)} per thousand</span>`);
      txt(svg, { x: X(xs[i]) + 9, y: Y(ys[i]) + 4, class: "ptlab" }, r.work);
    });
  }

  // ---------------- Phase 3 ----------------
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

  function lineage() {
    const host = $("#lineage"); if (!host) return;
    const q = D.lineage["qwen3-8b"], l = D.lineage["llama3.1-8b"];
    host.innerHTML = `<ul class="clean small">
      <li><b>Pimpernel to Zorro: holds.</b> Zorro's single nearest relative is the Scarlet Pimpernel in both models (Qwen: ${q.nearest["Zorro"].slice(0, 3).join(", ")}).</li>
      <li><b>"Iron Man is Batman in California": partly holds.</b> After Spider-Man, Iron Man's nearest works are recent Batman films, mostly Nolan's (Qwen: ${q.nearest["Iron Man"].slice(0, 3).join(", ")}; Llama: ${l.nearest["Iron Man"].slice(0, 3).join(", ")}).</li>
      <li><b>Zorro to Batman: not visible.</b> Zorro sits among Batman's most distant relatives, but its summary is only 9 sentences long, so it carries few memes of any kind.</li>
      <li><b>Batman's nearest relatives are his descendants.</b> Superman, Spider-Man, Daredevil and Iron Man sit closest; Puss in Boots and Zorro sit furthest.</li></ul>`;
  }

  tabs(); canonTiers(); canonChips(); completeness(); swimlane(); cousinsSpread(); lineage();
  e1Auc(); e1Cousins(); e1Core(); e2Ranks(); e2Sens(); e3Scatter();
  visionCounts(); visionList(); probes();
})();
