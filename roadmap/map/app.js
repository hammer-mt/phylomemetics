(function () {
  const D = window.DATA;
  const NS = 'http://www.w3.org/2000/svg';
  const $ = id => document.getElementById(id);
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const k in (attrs || {})) {
      if (k === 'class') e.className = attrs[k];
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]);
    }
    for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== '') e.append(kid.nodeType ? kid : String(kid));
    return e;
  }
  function s(tag, attrs, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in (attrs || {})) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }
  const STATUS = {
    held: ['Run: held up', 'held'], mixed: ['Run: mixed', 'mixed'], failed: ['Run: did not hold', 'failed'], inconclusive: ['Run: could not tell', 'inconclusive'],
    planned: ['Planned earlier, not run', 'planned'], registered: ['Design registered', 'registered'], seed: ['Idea only', 'seed'],
  };
  const stateOf = i => i.status === 'done' ? (i.verdict || 'mixed') : i.status;
  const ORDER = { held: 0, mixed: 1, failed: 2, inconclusive: 2.5, planned: 3, registered: 4, seed: 5 };
  const regionById = Object.fromEntries(D.regions.map(r => [r.id, r]));
  const ideas = D.ideas.slice();
  let selected = null, mode = 'all', bio = false;

  /* ---------- hex chart ---------- */
  const S = 17, DX = 210, DY = 240, X0 = 135, Y0 = 122;
  const DIRS = [[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]];
  function spiral(n) {
    const out = [[0, 0]];
    for (let k = 1; out.length < n; k++) {
      let q = -k, r = k;
      for (let i = 0; i < 6; i++) for (let j = 0; j < k; j++) { out.push([q, r]); q += DIRS[i][0]; r += DIRS[i][1]; }
    }
    return out.slice(0, n);
  }
  const hexPts = (x, y) => Array.from({ length: 6 }, (_, i) => {
    const a = Math.PI / 180 * (60 * i - 30);
    return (x + S * Math.cos(a)).toFixed(1) + ',' + (y + S * Math.sin(a)).toFixed(1);
  }).join(' ');
  const hexEls = {};
  function drawChart() {
    const cols = Math.max(...D.regions.map(r => r.col)) + 1, rows = Math.max(...D.regions.map(r => r.row)) + 1;
    const W = X0 * 2 + (cols - 1) * DX + DX / 2, H = Y0 + (rows - 1) * DY + 150;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Hex chart of every experiment idea, grouped by research area' });
    svg.append(s('text', { x: 14, y: 20, class: 'axis' }, 'FOUNDATIONS AT THE TOP · USES AT THE BOTTOM · LINES SHOW WHAT RESTS ON WHAT'));
    const centre = r => [X0 + r.col * DX + (r.row % 2 ? DX / 2 : 0), Y0 + r.row * DY];
    const defs = s('defs', {});
    const mk = s('marker', { id: 'arr', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' });
    mk.append(s('path', { d: 'M0 0L10 5L0 10z', class: 'linkhead' })); defs.append(mk); svg.append(defs);
    for (const [a, b] of (D.links || [])) {
      if (!regionById[a] || !regionById[b]) continue;
      const [x1, y1] = centre(regionById[a]), [x2, y2] = centre(regionById[b]);
      const len = Math.hypot(x2 - x1, y2 - y1), ux = (x2 - x1) / len, uy = (y2 - y1) / len, pad = Math.min(96, len * 0.4);
      svg.append(s('line', { x1: x1 + ux * pad, y1: y1 + uy * pad, x2: x2 - ux * pad, y2: y2 - uy * pad, class: 'link', 'marker-end': 'url(#arr)' }));
    }
    for (const r of D.regions) {
      const [cx, cy] = centre(r);
      const list = ideas.filter(i => i.region === r.id)
        .sort((a, b) => ORDER[stateOf(a)] - ORDER[stateOf(b)] || (b.priority || 0) - (a.priority || 0));
      const pos = spiral(Math.max(list.length, 1));
      let maxY = cy;
      list.forEach((idea, n) => {
        const [q, rr] = pos[n];
        const x = cx + S * Math.sqrt(3) * (q + rr / 2), y = cy + S * 1.5 * rr;
        maxY = Math.max(maxY, y);
        const st = stateOf(idea);
        const g = s('g', {});
        const p = s('polygon', { points: hexPts(x, y), class: 'hex ' + st, tabindex: 0, role: 'button', 'aria-label': idea.id + ': ' + idea.title + ' (' + STATUS[st][0] + ')' });
        p.addEventListener('click', () => select(idea.id, false));
        p.addEventListener('keydown', ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); select(idea.id, false); } });
        g.append(p);
        const mark = { held: '✓', mixed: '~', failed: '×', inconclusive: '–', planned: '◆', seed: '?' }[st];
        if (mark) g.append(s('text', { x, y, class: 'hexmark' + (st === 'seed' ? ' onland' : '') }, mark));
        if (idea.status !== 'done' && (idea.priority || 0) >= 4.5) { g.append(s('circle', { cx: x, cy: y, r: S * 0.62, class: 'treasure' })); if (!mark) g.append(s('text', { x, y, class: 'hexmark onland' }, '★')); }
        svg.append(g);
        hexEls[idea.id] = p;
      });
      const ly = maxY + S + 20;
      const lab = s('text', { x: cx, y: ly, class: 'rlabel' }, r.name);
      lab.dataset.name = r.name; lab.dataset.bio = r.bio || r.name;
      svg.append(lab);
      svg.append(s('text', { x: cx, y: ly + 14, class: 'rsub' }, list.length + ' ideas · ' + list.filter(i => i.status === 'done').length + ' run'));
    }
    $('chart').replaceChildren(svg);
  }
  function applyMode() {
    for (const i of ideas) {
      const st = stateOf(i);
      const show = mode === 'all' || (mode === 'run' && i.status === 'done') ||
        (mode === 'next' && (i.status === 'planned' || i.status === 'registered')) ||
        (mode === 'gold' && i.status !== 'done' && (i.priority || 0) >= 4.5) || (mode === 'fog' && st === 'seed');
      hexEls[i.id] && hexEls[i.id].classList.toggle('dim', !show);
    }
    document.querySelectorAll('[data-mode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
    document.querySelectorAll('.rlabel').forEach(l => { l.textContent = bio ? l.dataset.bio : l.dataset.name; });
    $('bioToggle').setAttribute('aria-pressed', String(bio));
  }
  function row(dt, dd) { return dd ? [h('dt', {}, dt), h('dd', {}, String(dd).replace(/`/g, ''))] : []; }
  function select(id, scroll) {
    selected = id;
    for (const k in hexEls) hexEls[k].classList.toggle('sel', k === id);
    const i = ideas.find(x => x.id === id); if (!i) return;
    const st = stateOf(i), r = regionById[i.region] || {};
    $('detail').replaceChildren(
      h('div', { class: 'eyebrow' }, i.id + ' · ' + (r.name || '')),
      h('h3', {}, i.title),
      h('div', { style: 'display:flex;flex-wrap:wrap;gap:6px;margin-top:8px' },
        h('span', { class: 'pill s-' + STATUS[st][1] }, STATUS[st][0]),
        i.priority ? h('span', { class: 'pill s-seed' }, 'priority ' + Number(i.priority).toFixed(1) + ' / 5') : null,
        /^[SML]\b/.test(i.cost || '') ? h('span', { class: 'pill s-seed' }, { S: 'small', M: 'medium', L: 'large' }[i.cost[0]] + ' cost') : null),
      h('dl', {},
        row('Question', i.question), row('Result so far', i.result), row('Prediction', i.hypothesis),
        row('What would refute it', i.falsifier), row('Data', i.data), row('Method', i.method),
        row('Cheap rival it must beat', i.rivals), row('Tests which premise', i.premise),
        row('Biology analogue', i.bio), row('Cost', i.cost), row('Payoff', i.payoff), row('Main risk', i.risk),
        row('Reviewer note', i.review), row('Cross-area ranking', i.ranking), row('Overlaps with', i.overlaps), row('Note', i.note), row('Amendment, 10 November', i.update), row('Sources', (i.sources || []).join('; '))));
    if (scroll) $('chartTop').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  /* ---------- database table ---------- */
  function drawTable() {
    const q = $('q').value.trim().toLowerCase(), fr = $('fRegion').value, fs = $('fStatus').value;
    const rows = ideas.filter(i => (!fr || i.region === fr) && (!fs || stateOf(i) === fs) &&
      (!q || JSON.stringify(i).toLowerCase().includes(q)))
      .sort((a, b) => (b.priority || 0) - (a.priority || 0));
    $('count').textContent = rows.length + ' of ' + ideas.length + ' ideas';
    $('tbody').replaceChildren(...rows.map(i => {
      const st = stateOf(i);
      return h('tr', { class: 'rowlink', onclick: () => select(i.id, true) },
        h('td', { class: 'num' }, i.id), h('td', {}, h('strong', {}, i.title), h('div', { class: 'small' }, i.question || '')),
        h('td', {}, (regionById[i.region] || {}).name || ''), h('td', {}, h('span', { class: 'pill s-' + STATUS[st][1] }, STATUS[st][0])),
        h('td', { class: 'num' }, i.priority ? Number(i.priority).toFixed(1) : ''), h('td', { class: 'num' }, /^[SML]\b/.test(i.cost || '') ? i.cost[0] : ''));
    }));
  }

  /* ---------- static sections ---------- */
  function drawPremises() {
    $('premises').replaceChildren(...D.premises.map(p => h('div', { class: 'sheet stack' },
      h('div', { class: 'eyebrow' }, p.id), h('h3', {}, p.title), h('p', {}, p.claim),
      h('p', { class: 'note' }, p.verdict),
      h('div', {}, p.rungs.map(r => h('div', { class: 'rung' }, h('strong', {}, r.name),
        h('span', { class: 'pill s-' + r.status }, r.status), h('p', {}, r.text)))))));
  }
  function drawAreas() {
    $('areas').replaceChildren(...D.regions.map(r => {
      const list = ideas.filter(i => i.region === r.id), n = Math.max(list.length, 1);
      const seg = k => { const c = list.filter(i => (k === 'run' ? i.status === 'done' : stateOf(i) === k)).length; return c ? h('span', { class: 'seg ' + k, style: 'width:' + (100 * c / n) + '%', title: c + ' ' + k }) : null; };
      return h('div', { class: 'sheet stack' },
        h('div', { class: 'eyebrow' }, 'In biology: ' + (r.bio || '')), h('h3', {}, r.name), h('p', { class: 'small' }, r.blurb || ''),
        h('p', {}, r.state || ''), h('div', { class: 'bar' }, seg('run'), seg('planned'), seg('registered'), seg('seed')),
        h('div', { class: 'mono small' }, list.filter(i => i.status === 'done').length + ' run of ' + list.length + ' ideas'));
    }));
  }
  function drawList(id, items) { $(id).replaceChildren(...(items || []).map(t => h('li', {}, t))); }
  function drawNews() {
    const news = D.summary.news || [];
    $('newsWrap').hidden = !news.length;
    $('news').replaceChildren(...news.map(n => h('div', { class: 'sheet stack' },
      h('div', { class: 'eyebrow' }, n.date + ' · ' + n.kind),
      h('h3', {}, n.url ? h('a', { href: n.url, target: '_blank', rel: 'noopener' }, n.title) : n.title),
      h('p', { class: 'small' }, n.by),
      h('p', {}, n.what),
      h('ul', { class: 'stack', style: 'padding-left:18px;margin:0' }, n.points.map(t => h('li', {}, t))),
      h('p', { class: 'note' }, n.changes),
      h('div', { style: 'display:flex;flex-wrap:wrap;gap:6px' }, (n.ideas || []).filter(id => ideas.some(i => i.id === id)).map(id => h('button', { class: 'chip', onclick: () => select(id, true) }, id))))));
  }
  function drawBiology() {
    $('bioBody').replaceChildren(...D.biology.map(b => h('tr', b.idea ? { class: 'rowlink', onclick: () => select(b.idea, true) } : {},
      h('td', {}, h('strong', {}, b.tool), h('div', { class: 'small' }, b.does || '')), h('td', {}, String(b.analogue).replace(/`/g, ''), b.experiment ? h('div', { class: 'small' }, 'Try: ' + String(b.experiment).replace(/`/g, '')) : null),
      h('td', {}, b.feasibility || ''), h('td', { class: 'num' }, b.idea || ''))));
    $('landmarks').replaceChildren(...D.landmarks.map(l => h('div', { class: 'sheet' },
      h('div', { class: 'eyebrow' }, l.when), h('h3', {}, l.who), h('p', { class: 'small' }, l.gave),
      h('p', {}, l.memetic), h('div', {}, h('span', { class: 'pill s-' + (l.status || 'unknown') }, l.statusText || l.status || '')),
      l.idea ? h('button', { class: 'chip', onclick: () => select(l.idea, true) }, 'See design ' + l.idea) : null)));
  }
  function drawSources() {
    $('srcBody').replaceChildren(...D.sources.map(x => h('tr', {},
      h('td', { class: 'num' }, x.priority || ''), h('td', {}, x.url ? h('a', { href: x.url, target: '_blank', rel: 'noopener' }, x.cite) : x.cite),
      h('td', {}, x.why || ''), h('td', {}, x.topic || ''))));
  }
  function init() {
    D.regions.forEach(r => $('fRegion').append(h('option', { value: r.id }, r.name)));
    Object.keys(STATUS).forEach(k => $('fStatus').append(h('option', { value: k }, STATUS[k][0])));
    drawChart(); drawAreas(); drawPremises(); drawBiology(); drawSources();
    drawList('known', D.summary.known); drawList('unknown', D.summary.unknown); drawList('nextUp', D.summary.next); drawNews();
    document.querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => { mode = b.dataset.mode; applyMode(); }));
    $('bioToggle').addEventListener('click', () => { bio = !bio; applyMode(); });
    ['q', 'fRegion', 'fStatus'].forEach(id => $(id).addEventListener('input', drawTable));
    applyMode(); drawTable();
    const first = ideas.filter(i => i.status !== 'done').sort((a, b) => (b.priority || 0) - (a.priority || 0))[0] || ideas[0];
    if (first) select(first.id, false);
    const c = k => ideas.filter(i => stateOf(i) === k).length;
    $('tally').textContent = ideas.filter(i => i.status === 'done').length + ' run · ' + c('planned') + ' planned earlier · ' + c('registered') + ' newly registered · ' + c('seed') + ' still fog';
  }
  init();
})();
