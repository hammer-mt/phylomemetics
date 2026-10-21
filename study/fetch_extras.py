"""Fetch the extra data E3 needs: data/pageviews.json, total English Wikipedia pageviews per work,
Jul 2015 - Sep 2026 (Wikimedia REST API).
"""
import json
import time
import urllib.parse
from pathlib import Path

import requests

HEADERS = {"User-Agent": "phylomemetics-pilot/0.2 (research project)"}
works = [json.loads(l) for l in open("data/corpus.jsonl")]


def get(url, **kw):
    for attempt in range(8):
        r = requests.get(url, headers=HEADERS, timeout=30, **kw)
        if r.status_code == 429 or r.status_code >= 500:
            wait = int(r.headers.get("retry-after", 0) or 0) or 5 * 2 ** attempt
            print(f"  waiting {wait}s ({r.status_code})", flush=True)
            time.sleep(wait)
            continue
        return r
    raise RuntimeError(url)


def pageviews():
    out_path = Path("data/pageviews.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for w in works:
        if str(w["id"]) in out:
            continue
        t = urllib.parse.quote(w["title"].replace(" ", "_"), safe="")
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{t}/monthly/20150701/20260930"
        r = get(url)
        items = r.json().get("items", []) if r.ok else []
        out[str(w["id"])] = {"title": w["title"], "views": sum(i["views"] for i in items), "months": len(items)}
        time.sleep(0.2)
    out_path.write_text(json.dumps(out, indent=1))
    print("pageviews:", len(out), "works; zero-view works:", [v["title"] for v in out.values() if not v["views"]])


if __name__ == "__main__":
    pageviews()
