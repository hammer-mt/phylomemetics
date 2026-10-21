"""Shared metadata extraction for T3 scout rows (no text analysis: counts, section names, medium, year, flags)."""
import json
import re
import sys

sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import SCOUT, fetch_extract, plot_section, sentences, split_sections, MAX_SENTENCES

PROJECT = "<repo>"
DEV_TITLES = set()
for l in open(f"{PROJECT}/data/corpus.jsonl"):
    r = json.loads(l)
    DEV_TITLES.add(r["title"])
    DEV_TITLES.add(r.get("label") or r["title"])

BATMAN = re.compile(r"\b(Batman|Batgirl|Batwoman|Bat-Man|Dark Knight|Joker|Gotham|Catwoman|Harley Quinn|Robin \(|Nightwing|Penguin \(|Riddler|Arkham|Bruce Wayne|Alfred Pennyworth|Lego Batman)\b")
SKIP = re.compile(r"^(List of|Lists of|Category|Template|Portal|Timeline of|Outline of)")


def medium(lead):
    l = lead[:600].lower()
    for key, val in [("film serial", "serial"), ("serial film", "serial"), ("television film", "tv film"), ("tv film", "tv film"),
                     ("television movie", "tv film"), ("made-for-television", "tv film"), ("direct-to-video", "film"),
                     ("miniseries", "tv series"), ("television series", "tv series"), ("tv series", "tv series"),
                     ("web series", "tv series"), ("graphic novel", "graphic novel"), ("comic book", "comic"),
                     ("short story", "short story"), ("novella", "novel"), ("novel", "novel"), ("video game", "video game"),
                     ("short film", "short film"), ("animated film", "film"), ("documentary", "documentary"),
                     ("film", "film"), ("movie", "film"), ("play", "play"), ("musical", "musical"), ("character", "character"),
                     ("magazine", "magazine")]:
        if key in l:
            return val
    return "unknown"


def year_of(page, lead):
    for c in page.get("categories", []):
        m = re.match(r"Category:(\d{4}) (films|novels|American novels|British novels|short stories)", c["title"])
        if m:
            return int(m.group(1))
    m = re.search(r"\b(1[6-9]\d\d|20[0-2]\d)\b", lead[:400])
    return int(m.group(1)) if m else None


def row_for(title, pause=0.15, keep_text=True):
    page = fetch_extract(title, pause=pause)
    if "missing" in page or "invalid" in page:
        return dict(query=title, title=None, missing=True)
    text = page.get("extract") or ""
    lead, secs = split_sections(text)
    lead = re.sub(r"\s+", " ", lead).strip()
    name, body = plot_section(page)
    sents = sentences(body) if body else []
    cats = [c["title"].replace("Category:", "") for c in page.get("categories", [])]
    r = dict(query=title, title=page.get("title"), wikibase_item=page.get("pageprops", {}).get("wikibase_item"),
             shortdesc=page.get("pageprops", {}).get("wikibase-shortdesc"), medium=medium(lead), year=year_of(page, lead),
             plot_section=name, n_sentences=len(sents), n_sentences_capped=min(len(sents), MAX_SENTENCES),
             in_dev=page.get("title") in DEV_TITLES, batman_lineage=bool(BATMAN.search(page.get("title") or "")),
             top_sections=[n for n, _ in secs][:10], lead=lead[:800], categories=cats)
    if keep_text:
        r["sentences"] = sents
        r["raw_section"] = body
    return r
