"""Wikidata 'based on' (P144) and 'inspired by' (P941) links to each Shakespeare play, restricted to items with an
English Wikipedia article. One small query per play. Writes scout/shakespeare_wikidata.json. Metadata only."""
import json
import re
import sys
import time
from urllib.parse import unquote
sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import SCOUT
from wikidata import sparql

plays = [json.loads(l) for l in open(SCOUT / "plays.jsonl")]
qid = {}
for p in plays:
    page = json.load(open(SCOUT / "raw" / "pages" / (re.sub(r"[^\w]+", "_", p["title"]) + ".json")))
    qid[p["label"]] = page["pageprops"]["wikibase_item"]
out = []
for lab, q in qid.items():
    for prop in ("P144", "P941"):
        rows = sparql(f"""SELECT ?item ?article WHERE {{ ?item wdt:{prop} wd:{q} .
          OPTIONAL {{ ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }} }}""")
        for r in rows:
            art = r.get("article", {}).get("value")
            out.append(dict(play=lab, play_qid=q, prop=prop, qid=r["item"]["value"].rsplit("/", 1)[1],
                            title=unquote(art.rsplit("/", 1)[1]).replace("_", " ") if art else None))
        time.sleep(0.3)
    print(lab, sum(1 for o in out if o["play"] == lab), flush=True)
json.dump({"play_qids": qid, "links": out}, open(SCOUT / "shakespeare_wikidata.json", "w"), indent=1, ensure_ascii=False)
print(len(out), "links;", sum(1 for o in out if o["title"]), "with enwiki article")
