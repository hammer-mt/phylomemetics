"""Shared Wikipedia helpers for the E5 scout. Mirrors fetch_corpus.py (same API call, section rule and
sentence splitter) but caches under the scratchpad scout folder, never under the project's data/raw."""
import json
import re
import sys
import time
from pathlib import Path

import requests

SCOUT = Path("<scratchpad>/prereg/scout")
CACHE = SCOUT / "raw" / "pages"
HEADERS = {"User-Agent": "phylomemetics-scout/0.1 (https://github.com/hammer-mt/phylomemetics; research project) python-requests"}
API = "https://en.wikipedia.org/w/api.php"

# identical to fetch_corpus.py
SECTION_NAMES = re.compile(r"^(plot|synopsis|plot summary|story|summary|premise|storyline|plot and themes|narrative|overview)", re.I)
MAX_SENTENCES = 40
ABBREVS = ("Mr.", "Mrs.", "Dr.", "St.", "Jr.", "Sr.", "Ms.", "Lt.", "Sgt.", "vs.", "U.S.", "R.I.P.", "Gen.", "Capt.", "Det.")
SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”’)]))\s+(?=[A-Z\"“‘(])")


def api(params, tries=8):
    for attempt in range(tries):
        try:
            r = requests.get(API, headers=HEADERS, params={**params, "format": "json", "maxlag": 5}, timeout=30)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("retry-after", 0) or 0) + 1)
                raise ValueError("429")
            d = r.json()
            if "error" in d and d["error"].get("code") == "maxlag":
                raise ValueError("maxlag")
            return d
        except (ValueError, requests.RequestException):
            wait = 5 * 2 ** attempt
            print(f"  retry {params.get('titles') or params.get('cmtitle')} in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"could not fetch {params}")


def fetch_extract(title, pause=0.7):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (re.sub(r"[^\w]+", "_", title) + ".json")
    if path.exists():
        return json.loads(path.read_text())
    d = api({"action": "query", "prop": "extracts|categories|pageprops", "explaintext": 1, "exsectionformat": "wiki",
             "cllimit": "max", "titles": title, "redirects": 1})
    page = next(iter(d["query"]["pages"].values()))
    path.write_text(json.dumps(page))
    time.sleep(pause)
    return page


def split_sections(text):
    parts = re.split(r"^==\s*([^=].*?)\s*==\s*$", text, flags=re.M)
    return parts[0], [(parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def plot_section(page):
    """Same rule as fetch_corpus.plot_section: first top-level section whose name matches SECTION_NAMES."""
    text = page.get("extract") or ""
    lead, secs = split_sections(text)
    for name, body in secs:
        if SECTION_NAMES.match(name):
            body = re.sub(r"^===+.*?===+\s*$", "", body, flags=re.M)
            return name, body
    return None, None


def sentences(body, cap=None):
    body = re.sub(r"\s+", " ", body).strip()
    merged = []
    for piece in SPLIT.split(body):
        if merged and merged[-1].endswith(ABBREVS):
            merged[-1] += " " + piece
        else:
            merged.append(piece)
    out = [s.strip() for s in merged if len(s.strip()) > 15]
    return out[:cap] if cap else out


def category_members(cat):
    out, cont = [], {}
    while True:
        d = api({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmlimit": 500, **cont})
        out += d["query"]["categorymembers"]
        if "continue" not in d:
            return out
        cont = d["continue"]
