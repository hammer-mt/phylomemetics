"""One polite, cached door to the English Wikipedia API for the second corpora (run from study/).

    ../.venv/bin/python v2/second/wiki.py extracts <titles.txt>      whole article as plain text, one request a page
    ../.venv/bin/python v2/second/wiki.py leads <titles.txt>         wikitext of the lead section, 50 pages a request
    ../.venv/bin/python v2/second/wiki.py wikitexts <titles.txt>     whole wikitext, 50 pages a request
    ../.venv/bin/python v2/second/wiki.py category <Category name>   the pages and sub-categories of one category

Every answer is cached in v2/cache/wiki/ (not committed), with the time it was fetched, so nothing is asked twice.
Requests go out one at a time, at most one every INTERVAL seconds, and a file lock makes several callers queue: the
API answers 429 to bursts. A title file holds one title a line.
"""
import datetime
import fcntl
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "phylomemetics-study/0.2 (research script; reads plot summaries and article leads)"}
CACHE = Path("v2/cache/wiki")
INTERVAL = 1.5


def safe(title):
    """A file name for a title: readable, and unique even where two titles differ only in punctuation."""
    return re.sub(r"[^\w]+", "_", title)[:80] + "." + hashlib.sha1(title.encode()).hexdigest()[:8]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def api(params, tries=12):
    """One GET, spaced out, retried with a growing wait when the API says slow down."""
    params = dict(params, format="json", formatversion=2, maxlag=5)
    with open(CACHE / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stamp = CACHE / ".last"                                   # shared by every caller, so the spacing holds across processes
        for attempt in range(tries):
            last = float(stamp.read_text()) if stamp.exists() else 0.0
            gap = INTERVAL - (time.time() - last)
            if gap > 0:
                time.sleep(gap)
            try:
                r = requests.get(API, headers=HEADERS, params=params, timeout=60)
                stamp.write_text(str(time.time()))
                if r.status_code == 200:
                    d = r.json()
                    if "error" not in d:
                        return d
                    wait = float(r.headers.get("retry-after", 0) or 0) or 5
                else:
                    wait = float(r.headers.get("retry-after", 0) or 0) or min(5 * 2 ** attempt, 120)
            except (requests.RequestException, ValueError):
                wait = min(5 * 2 ** attempt, 120)
            time.sleep(wait)
    raise RuntimeError(f"no answer for {params}")


def extract(title):
    """The whole article as plain text, with its section headings kept as wiki marks."""
    path = CACHE / "extract" / (safe(title) + ".json")
    if path.exists():
        return json.loads(path.read_text())
    d = api({"action": "query", "prop": "extracts|info", "explaintext": 1, "exsectionformat": "wiki", "titles": title, "redirects": 1})
    page = d["query"]["pages"][0]
    out = {"asked": title, "title": page.get("title"), "pageid": page.get("pageid"), "missing": bool(page.get("missing")),
           "touched": page.get("touched"), "lastrevid": page.get("lastrevid"), "extract": page.get("extract") or "", "fetched_utc": now()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False))
    return out


def leads(titles):
    """Wikitext of the lead section of each title, asked 50 at a time. Returns {asked title: record}."""
    out, todo = {}, []
    for t in titles:
        path = CACHE / "lead" / (safe(t) + ".json")
        if path.exists():
            out[t] = json.loads(path.read_text())
        else:
            todo.append(t)
    (CACHE / "lead").mkdir(parents=True, exist_ok=True)
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        d = api({"action": "query", "prop": "revisions", "rvprop": "content|timestamp|ids", "rvslots": "main", "rvsection": 0,
                 "titles": "|".join(chunk), "redirects": 1})
        q = d["query"]
        via = {}                                              # asked title -> the title the API settled on
        for step in ("normalized", "redirects"):
            for m in q.get(step, []):
                via[m["from"]] = m["to"]
        pages = {p["title"]: p for p in q.get("pages", [])}
        for t in chunk:
            end = t
            for _ in range(4):
                end = via.get(end, end)
            p = pages.get(end, {})
            rev = (p.get("revisions") or [{}])[0]
            rec = {"asked": t, "title": p.get("title"), "pageid": p.get("pageid"), "missing": bool(p.get("missing", "title" not in p)),
                   "revid": rev.get("revid"), "rev_timestamp": rev.get("timestamp"),
                   "wikitext": (rev.get("slots", {}).get("main", {}) or {}).get("content", ""), "fetched_utc": now()}
            (CACHE / "lead" / (safe(t) + ".json")).write_text(json.dumps(rec, ensure_ascii=False))
            out[t] = rec
        print(f"  leads {min(i + 50, len(todo))}/{len(todo)}", flush=True)
    return out


def wikitexts(titles):
    """Whole wikitext of each title, asked 50 at a time (the API hands a large batch back in parts). Returns {asked: record}."""
    out, todo = {}, []
    for t in titles:
        path = CACHE / "wikitext" / (safe(t) + ".json")
        if path.exists():
            out[t] = json.loads(path.read_text())
        else:
            todo.append(t)
    (CACHE / "wikitext").mkdir(parents=True, exist_ok=True)
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        via, pages, cont = {}, {}, {}
        while True:
            d = api(dict({"action": "query", "prop": "revisions", "rvprop": "content|timestamp|ids", "rvslots": "main",
                          "titles": "|".join(chunk), "redirects": 1}, **cont))
            q = d["query"]
            for step in ("normalized", "redirects"):
                for m in q.get(step, []):
                    via[m["from"]] = m["to"]
            for p in q.get("pages", []):
                if p.get("revisions") or p["title"] not in pages:
                    pages[p["title"]] = p
            if "continue" not in d:
                break
            cont = d["continue"]
        for t in chunk:
            end = t
            for _ in range(4):
                end = via.get(end, end)
            p = pages.get(end, {})
            rev = (p.get("revisions") or [{}])[0]
            rec = {"asked": t, "title": p.get("title"), "pageid": p.get("pageid"), "missing": bool(p.get("missing", "title" not in p)),
                   "revid": rev.get("revid"), "rev_timestamp": rev.get("timestamp"),
                   "wikitext": (rev.get("slots", {}).get("main", {}) or {}).get("content", ""), "fetched_utc": now()}
            (CACHE / "wikitext" / (safe(t) + ".json")).write_text(json.dumps(rec, ensure_ascii=False))
            out[t] = rec
        print(f"  wikitexts {min(i + 50, len(todo))}/{len(todo)}", flush=True)
    return out


def category(name):
    """Pages (namespace 0) and sub-categories (namespace 14) of one category."""
    name = name if name.startswith("Category:") else "Category:" + name
    path = CACHE / "category" / (safe(name) + ".json")
    if path.exists():
        return json.loads(path.read_text())
    members, cont = [], {}
    while True:
        d = api(dict({"action": "query", "list": "categorymembers", "cmtitle": name, "cmlimit": 500, "cmnamespace": "0|14"}, **cont))
        members += [{"title": m["title"], "ns": m["ns"]} for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            break
        cont = {"cmcontinue": d["continue"]["cmcontinue"]}
    out = {"category": name, "pages": [m["title"] for m in members if m["ns"] == 0], "subcategories": [m["title"] for m in members if m["ns"] == 14],
           "fetched_utc": now()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False))
    return out


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1]
    if what == "category":
        c = category(" ".join(sys.argv[2:]))
        print(json.dumps(c, ensure_ascii=False, indent=1))
        return
    titles = [t.strip() for t in open(sys.argv[2]) if t.strip()]
    if what in ("leads", "wikitexts"):
        got = (leads if what == "leads" else wikitexts)(titles)
        print(len(got), what + ";", sum(r["missing"] for r in got.values()), "missing;", sum(not r["wikitext"] for r in got.values()), "empty")
    else:
        miss = 0
        for k, t in enumerate(titles):
            e = extract(t)
            miss += e["missing"]
            if k % 25 == 0 or k == len(titles) - 1:
                print(f"  extracts {k + 1}/{len(titles)} ({miss} missing) {t[:50]}", flush=True)


if __name__ == "__main__":
    main()
