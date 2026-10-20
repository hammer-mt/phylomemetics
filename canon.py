"""Canon and spread: what a story must carry to be Batman, what is optional, and how memes spread over time.

Tiers, by the share of the 32 Batman works a meme is present in:
  core      >= 90%  (29+ works)   "must have"
  common    70-89%  (23-28)
  optional  19-69%  (6-22)        the minimum for a meme in discovery is 6 works
Batman-specific if it is present in at most 3 of the 13 cousins; otherwise shared with the genre.

Writes results/canon.json.
"""
import json
from collections import Counter

import numpy as np

works = [json.loads(l) for l in open("data/corpus.jsonl")]
W = {w["id"]: w for w in works}
bat = sorted([w for w in works if w["group"] == "batman"], key=lambda w: (w["year"], w["label"]))
cous = sorted([w for w in works if w["group"] == "cousin"], key=lambda w: w["year"])
fams = json.load(open("results/qwen3-8b/memes.json"))["families"]


def tier(n):
    return "core" if n >= 29 else "common" if n >= 23 else "optional"


rows = []
for f in fams:
    pres = set(f["present"])
    nb = sum(w["id"] in pres for w in bat)
    nc = sum(w["id"] in pres for w in cous)
    years = [w["year"] for w in bat if w["id"] in pres]
    rows.append({"unit": f["unit"], "layer": f["layer"], "n_batman": nb, "n_cousin": nc, "tier": tier(nb),
                 "specific": nc <= 3, "first_year": min(years) if years else None, "strength": round(f["strength"], 3),
                 "present": sorted(pres)})

counts = Counter((r["tier"], r["specific"]) for r in rows)
print("tier x specific:", dict(counts))

core = [r for r in rows if r["tier"] == "core"]
core_spec = [r for r in core if r["specific"]]
opt = [r for r in rows if r["tier"] == "optional"]

# completeness of the canon in each Batman work, over time
per_work = []
seen = set()
for w in bat:
    has_core = sum(w["id"] in set(r["present"]) for r in core)
    has_core_spec = sum(w["id"] in set(r["present"]) for r in core_spec)
    mine = [r for r in opt if w["id"] in set(r["present"])]
    inherited = sum(r["unit"] in seen for r in mine)
    per_work.append({"id": w["id"], "label": w["label"], "year": w["year"], "era": w["era"], "medium": w["medium"],
                     "core_share": round(has_core / len(core), 3), "core_specific_share": round(has_core_spec / max(len(core_spec), 1), 3),
                     "n_optional": len(mine), "optional_inherited": inherited,
                     "optional_new": len(mine) - inherited})
    seen |= {r["unit"] for r in mine}

# how much of Batman's core each cousin carries
per_cousin = [{"id": w["id"], "label": w["label"], "year": w["year"],
               "core_share": round(sum(w["id"] in set(r["present"]) for r in core) / len(core), 3),
               "core_specific_share": round(sum(w["id"] in set(r["present"]) for r in core_spec) / max(len(core_spec), 1), 3)}
              for w in cous]

for p in per_work:
    print(f"{p['year']} {p['label'][:24]:24s} core {p['core_share']:.2f} core-specific {p['core_specific_share']:.2f} optional {p['n_optional']:5d} (new {p['optional_new']})")
for c in per_cousin:
    print(f"   cousin {c['year']} {c['label'][:22]:22s} core {c['core_share']:.2f} specific {c['core_specific_share']:.2f}")

json.dump({"tiers": {f"{t}|{'specific' if s else 'shared'}": n for (t, s), n in counts.items()},
           "n_core": len(core), "n_core_specific": len(core_spec), "n_optional": len(opt),
           "memes": rows, "per_work": per_work, "per_cousin": per_cousin}, open("results/canon.json", "w"))
print("wrote results/canon.json")
