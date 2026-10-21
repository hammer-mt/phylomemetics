"""Add Wikidata-documented Shakespeare derivatives (P144 based on / P941 inspired by) that the category crawl missed
to scout/films_all.jsonl, with the same metadata fields as films.py plus 'wikidata_plays'. Metadata only."""
import json
import re
import sys
from collections import defaultdict

sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import SCOUT, fetch_extract, plot_section, sentences, split_sections, MAX_SENTENCES
from films import direct_plays, lead_plays, medium, NAMES, BASED, PLAY_TITLES

wd = json.load(open(SCOUT / "shakespeare_wikidata.json"))["links"]
wd_by_title = defaultdict(set)
for l in wd:
    if l["title"]:
        wd_by_title[l["title"]].add((l["play"], l["prop"]))
rows = [json.loads(l) for l in open(SCOUT / "films_all.jsonl")]
have = {r.get("title") for r in rows} | {r.get("query") for r in rows}
# annotate existing rows
for r in rows:
    r["wikidata_plays"] = sorted({f"{p}:{pr}" for p, pr in wd_by_title.get(r.get("title"), set())})
new = [t for t in wd_by_title if t not in have and t not in {p["title"] for p in map(json.loads, open(SCOUT / "plays.jsonl"))}]
print(len(new), "new titles from Wikidata", flush=True)
for t in new:
    page = fetch_extract(t, pause=0.15)
    if "missing" in page:
        rows.append(dict(query=t, title=None, missing=True))
        continue
    text = page.get("extract") or ""
    lead, secs = split_sections(text)
    lead = re.sub(r"\s+", " ", lead).strip()
    name, body = plot_section(page)
    sents = sentences(body) if body else []
    cats = [c["title"] for c in page.get("categories", [])]
    dp, lp = direct_plays(cats), lead_plays(lead)
    wps = sorted({p for p, _ in wd_by_title[t]})
    ev = [s for s in sentences(lead) if BASED.search(s) and (lead_plays(s) or "Shakespeare" in s)][:2]
    plot_txt = " ".join(sents)
    name_hits = {p: [n for n in NAMES[p] if re.search(r"\b" + re.escape(n) + r"\b", plot_txt)] for p in sorted(set(dp) | set(lp) | set(wps))}
    rows.append(dict(query=t, title=page.get("title"), in_category_tree=False, tree_paths=[], source="wikidata",
                     direct_play_cats=dp, lead_plays=lp, wikidata_plays=sorted({f"{p}:{pr}" for p, pr in wd_by_title[t]}),
                     medium=medium(lead), lead=lead[:1200], evidence=ev, plot_section=name, n_sentences=len(sents),
                     n_sentences_capped=min(len(sents), MAX_SENTENCES), plot_names_from_play=name_hits,
                     plot_mentions_shakespeare=len(re.findall(r"Shakespeare", plot_txt)),
                     plot_mentions_play_title={p: len(re.findall(re.escape(PLAY_TITLES[p]), plot_txt)) for p in name_hits},
                     categories=cats, sentences=sents, raw_section=body))
with open(SCOUT / "films_all.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("films_all now", len(rows))
