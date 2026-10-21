"""Ground truth scout: Wikidata links between Batman (and Bruce Wayne) and other items via P941 'inspired by',
P144 'based on', P1889 'different from' is ignored. Both directions. Writes scout/relatives/wikidata.json."""
import json
import sys
import time
from urllib.parse import unquote
sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import SCOUT, api
from wikidata import sparql

OUT = SCOUT / "relatives"
OUT.mkdir(exist_ok=True)
seeds = {}
for t in ["Batman", "Bruce Wayne", "Batman (comic book)", "Detective Comics 27"]:
    d = api({"action": "query", "prop": "pageprops", "titles": t, "redirects": 1})
    p = next(iter(d["query"]["pages"].values()))
    seeds[t] = (p.get("title"), p.get("pageprops", {}).get("wikibase_item"))
print(seeds)
rows = []
for name, (title, q) in seeds.items():
    if not q:
        continue
    for prop in ("P941", "P144", "P4969", "P8646"):   # inspired by, based on, derivative work, expansion of
        for direction in ("out", "in"):
            trip = f"wd:{q} wdt:{prop} ?item ." if direction == "out" else f"?item wdt:{prop} wd:{q} ."
            res = sparql(f"""SELECT ?item ?itemLabel ?article WHERE {{ {trip}
              OPTIONAL {{ ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}""")
            for r in res:
                art = r.get("article", {}).get("value")
                rows.append(dict(seed=name, seed_qid=q, prop=prop, direction=direction, qid=r["item"]["value"].rsplit("/", 1)[1],
                                 label=r.get("itemLabel", {}).get("value"), enwiki=unquote(art.rsplit("/", 1)[1]).replace("_", " ") if art else None))
            time.sleep(0.3)
json.dump({"seeds": seeds, "links": rows}, open(OUT / "wikidata.json", "w"), indent=1, ensure_ascii=False)
for r in rows:
    print(r["seed"], r["prop"], r["direction"], r["label"], "|", r["enwiki"])
