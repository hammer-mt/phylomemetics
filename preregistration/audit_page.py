"""The author's audit page (annotation_protocol.md A7; prereg-v3). Standard library only; served on 127.0.0.1, so nothing
leaves the machine.

It reads ONE file, the blinded sample (data/human_audit_sample.json, written by `build_lists.py texts audit`): opaque
ids A01, A02, ... in presentation order, each with its block, its numbered sentences and the 5 memes asked (label and
definition). It never opens t2_stories.jsonl and has no story id, plot, version, writer or genre to show. It shows
one story at a time, with three answers per meme (happens, does not happen, unsure), and APPENDS one line per answer to
the answers file (data/human_audit.jsonl):

  {"kind": "t2_cell", "audit_id": "A07", "meme": "<meme id>", "present": true | false | null, "seconds": <since the
   story was shown>, "time": "<local time>"}

An answer can be changed: a new line is appended and the last line of a cell counts (analysis.audit_summary). A story
is done when its 5 memes have an answer; stories come in balanced blocks of 12, and the analysis scores whole blocks.

  .venv/bin/python preregistration/audit_page.py                       # main audit (after freeze F2-T2), http://127.0.0.1:8765
  .venv/bin/python preregistration/audit_page.py --sample preregistration/agent/pilot/human_audit_sample_pilot.json \\
      --out preregistration/agent/pilot/human_audit_pilot.jsonl         # P2: the author times at most 5 pilot stories
  .venv/bin/python preregistration/audit_page.py --pilot-report preregistration/agent/pilot/human_audit_pilot.jsonl   # median seconds per story
"""
import argparse
import html
import json
import statistics
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PRE = Path(__file__).resolve().parent
SAMPLE = PRE / "data" / "human_audit_sample.json"
OUT = PRE / "data" / "human_audit.jsonl"
ANSWERS = {"yes": True, "no": False, "unsure": None}
LOCK = threading.Lock()


def load_sample(path):
    doc = json.load(open(path))
    for it in doc["items"]:
        assert set(it) == {"audit_id", "block", "sentences", "memes"}, f"unexpected fields in the sample: {sorted(it)}"
    return doc


def read_answers(path):
    """Every appended row, in order."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def latest(rows):
    """The answer that counts for each (audit_id, meme): the last row."""
    out = {}
    for r in rows:
        if r.get("kind") == "t2_cell":
            out[(r["audit_id"], r["meme"])] = r["present"]
    return out


def append_answer(out_path, sample, payload):
    """Validates one answer from the page and appends it as a row. Returns the row."""
    items = {it["audit_id"]: it for it in sample["items"]}
    aid, meme, ans = payload.get("audit_id"), payload.get("meme"), payload.get("answer")
    if aid not in items or meme not in [m["id"] for m in items[aid]["memes"]] or ans not in ANSWERS:
        raise ValueError("unknown story, element or answer")
    secs = payload.get("seconds")
    row = {"kind": "t2_cell", "audit_id": aid, "meme": meme, "present": ANSWERS[ans],
           "seconds": float(secs) if isinstance(secs, (int, float)) else None, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with LOCK:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    return row


def progress(sample, rows):
    last = latest(rows)
    done = [it["audit_id"] for it in sample["items"] if all((it["audit_id"], m["id"]) in last for m in it["memes"])]
    blocks = {}
    for it in sample["items"]:
        blocks.setdefault(it["block"], []).append(it["audit_id"])
    full = [b for b, ids in blocks.items() if all(a in done for a in ids)]
    return {"stories_done": len(done), "n_stories": len(sample["items"]), "blocks_done": len(full), "n_blocks": len(blocks), "done": done}


CSS = """
:root { --bg: #fbfaf7; --fg: #1d1d1b; --muted: #6b6a66; --line: #dedbd3; --accent: #2f5d8a; --chosen: #dce8f3; }
@media (prefers-color-scheme: dark) { :root { --bg: #1a1a19; --fg: #ecebe6; --muted: #a3a29c; --line: #3a3936; --accent: #8fb6dc; --chosen: #24384b; } }
body { background: var(--bg); color: var(--fg); font: 17px/1.55 Georgia, serif; margin: 0; }
main { max-width: 760px; margin: 0 auto; padding: 24px 16px 80px; }
.top { font: 14px/1.4 -apple-system, system-ui, sans-serif; color: var(--muted); }
.bar { height: 6px; background: var(--line); border-radius: 3px; margin: 8px 0 20px; }
.bar > div { height: 6px; background: var(--accent); border-radius: 3px; }
ol.story { padding-left: 2.2em; } ol.story li { margin: 0 0 6px; }
.el { border-top: 1px solid var(--line); padding: 14px 0; }
.el h3 { font: 600 16px/1.3 -apple-system, system-ui, sans-serif; margin: 0 0 4px; }
.el p { margin: 0 0 10px; color: var(--muted); font-size: 15px; }
button { font: 14px -apple-system, system-ui, sans-serif; padding: 7px 12px; margin-right: 6px; border: 1px solid var(--line);
         background: transparent; color: var(--fg); border-radius: 6px; cursor: pointer; }
button.on { background: var(--chosen); border-color: var(--accent); }
nav { display: flex; justify-content: space-between; margin-top: 24px; font: 15px -apple-system, system-ui, sans-serif; }
a { color: var(--accent); }
"""

JS = """
const shown = Date.now();
async function answer(btn, aid, meme, ans) {
  const r = await fetch('/answer', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({audit_id: aid, meme: meme, answer: ans, seconds: (Date.now() - shown) / 1000})});
  if (!r.ok) { alert('Not saved: ' + await r.text()); return; }
  for (const b of btn.parentElement.querySelectorAll('button')) b.classList.remove('on');
  btn.classList.add('on');
  const p = await r.json();
  document.getElementById('prog').textContent = p.stories_done + ' of ' + p.n_stories + ' stories done (' + p.blocks_done + ' of ' + p.n_blocks + ' blocks)';
}
"""


def render(sample, rows, i):
    """The page for item i (0-based): instructions, progress, the story as numbered sentences and the 5 elements."""
    items = sample["items"]
    i = max(0, min(i, len(items) - 1))
    it = items[i]
    last = latest(rows)
    pr = progress(sample, rows)
    esc = html.escape
    out = ["<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
           f"<title>Story audit</title><style>{CSS}</style><script>{JS}</script></head><body><main>",
           f"<div class='top'>{esc(sample.get('instructions', ''))}<br>Story {i + 1} of {len(items)} &middot; block {it['block']} &middot; "
           f"<span id='prog'>{pr['stories_done']} of {pr['n_stories']} stories done ({pr['blocks_done']} of {pr['n_blocks']} blocks)</span></div>",
           f"<div class='bar'><div style='width:{100 * pr['stories_done'] / max(1, pr['n_stories']):.1f}%'></div></div>",
           "<ol class='story'>" + "".join(f"<li>{esc(s)}</li>" for s in it["sentences"]) + "</ol>"]
    for m in it["memes"]:
        cur = last.get((it["audit_id"], m["id"]), "none")
        btns = []
        for key, lab in (("yes", "happens"), ("no", "does not happen"), ("unsure", "unsure")):
            on = " class='on'" if cur != "none" and cur == ANSWERS[key] else ""
            btns.append(f"<button{on} onclick=\"answer(this, '{esc(it['audit_id'])}', '{esc(m['id'])}', '{key}')\">{lab}</button>")
        out.append(f"<div class='el'><h3>{esc(m['label'])}</h3><p>{esc(m['definition'])}</p><div>{''.join(btns)}</div></div>")
    prev = f"<a href='/?i={i - 1}'>&larr; previous story</a>" if i > 0 else "<span></span>"
    nxt = f"<a href='/?i={i + 1}'>next story &rarr;</a>" if i + 1 < len(items) else "<span>last story</span>"
    out.append(f"<nav>{prev}<a href='/'>first unfinished</a>{nxt}</nav></main></body></html>")
    return "".join(out)


def first_unfinished(sample, rows):
    done = set(progress(sample, rows)["done"])
    return next((k for k, it in enumerate(sample["items"]) if it["audit_id"] not in done), len(sample["items"]) - 1)


def make_server(sample_path=SAMPLE, out_path=OUT, port=8765):
    sample = load_sample(sample_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            b = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path != "/":
                return self._send(404, "not found", "text/plain")
            rows = read_answers(out_path)
            q = parse_qs(u.query)
            i = int(q["i"][0]) if q.get("i", [""])[0].lstrip("-").isdigit() else first_unfinished(sample, rows)
            self._send(200, render(sample, rows, i))

        def do_POST(self):
            if urlparse(self.path).path != "/answer":
                return self._send(404, "not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", 0))
                append_answer(out_path, sample, json.loads(self.rfile.read(n) or b"{}"))
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, str(e), "text/plain")
            p = progress(sample, read_answers(out_path))
            p.pop("done")
            self._send(200, json.dumps(p), "application/json")

    return HTTPServer(("127.0.0.1", port), Handler)


def pilot_report(path):
    """P2: the author's median seconds per pilot story (the time of a story's last answer since it was shown)."""
    per = {}
    for r in read_answers(path):
        if r.get("kind") == "t2_cell" and r.get("seconds") is not None:
            per[r["audit_id"]] = max(per.get(r["audit_id"], 0.0), r["seconds"])
    med = statistics.median(per.values()) if per else None
    return {"n_stories_timed": len(per), "seconds_per_story": per, "median_seconds": med}


def main(argv=None):
    ap = argparse.ArgumentParser(description="The author's audit page (annotation_protocol.md A7)")
    ap.add_argument("--sample", default=str(SAMPLE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pilot-report", default=None, help="print the median seconds per story of an answers file and exit")
    a = ap.parse_args(argv)
    if a.pilot_report:
        print(json.dumps(pilot_report(a.pilot_report), indent=1))
        return
    srv = make_server(a.sample, a.out, a.port)
    print(f"audit page on http://127.0.0.1:{srv.server_address[1]}  (answers appended to {a.out}; Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
