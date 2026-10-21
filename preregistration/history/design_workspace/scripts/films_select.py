"""Print a review table of fetched film articles that have a plot section of >= 8 sentences and a documented
Shakespeare source. Metadata only: no similarity between texts is computed."""
import json
import re
import sys

sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import SCOUT

rows = [json.loads(l) for l in open(SCOUT / "films_all.jsonl")]
mode = sys.argv[1] if len(sys.argv) > 1 else "table"
keep = []
for r in rows:
    if r.get("missing"):
        print("MISSING", r["query"], file=sys.stderr)
        continue
    plays = sorted(set(r["direct_play_cats"]) | set(r["lead_plays"]))
    r["plays"] = plays
    if r["n_sentences"] < 8 or not plays:
        continue
    keep.append(r)
print(len(rows), "fetched;", len(keep), "with >= 8 plot sentences and a play attribution", file=sys.stderr)
for r in sorted(keep, key=lambda r: (r["plays"][0], r["title"])):
    hits = r["plot_names_from_play"]
    ev = r["evidence"][0][:230] if r["evidence"] else ("LEAD: " + r["lead"][:230])
    print(f"{r['title']} | {r['medium']} | n={r['n_sentences']} | cat={r['direct_play_cats']} lead={r['lead_plays']} | "
          f"names={ {k: v for k, v in hits.items() if v} } | shk={r['plot_mentions_shakespeare']} title={ {k: v for k, v in r['plot_mentions_play_title'].items() if v} }")
    print("     ", ev)
