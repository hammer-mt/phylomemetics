"""T3 pool scout, step 4: classify fetched candidates by medium (Wikipedia short description first, lead second),
apply the eligibility rule (a film, TV film, serial, novel, novella or short story with a plot section of >= 8
sentences, not already in DEV), and count works and sentences per bucket. Metadata only. Writes
scout/t3/pool_summary.json and scout/t3/pool_eligible.jsonl (titles and metadata, no text)."""
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import SCOUT

T3 = SCOUT / "t3"


def kind(r):
    sd = (r.get("shortdesc") or "").lower()
    t = r.get("title") or ""
    if re.search(r"\b(television|tv series|tv program|miniseries|episode|web series|tv special|anime series)\b", sd):
        return "tv"
    if re.search(r"\b(video game|game)\b", sd):
        return "game"
    if re.search(r"\b(character|fictional|superhero team|team)\b", sd) and not re.search(r"\b(film|novel)\b", sd):
        return "character"
    if re.search(r"\b(comic|graphic novel|manga|comic book|storyline|limited series)\b", sd):
        return "comic"
    if re.search(r"\b(film serial|serial)\b", sd):
        return "serial"
    if re.search(r"\b(film|movie|animated feature|cartoon)\b", sd):
        return "tv film" if re.search(r"television film|tv film|made-for-tv", sd) else "film"
    if re.search(r"\b(novel|novella|book|novelization)\b", sd):
        return "novel"
    if re.search(r"\b(short story|story|pulp)\b", sd):
        return "short story"
    if re.search(r"\b(play|musical|opera)\b", sd):
        return "stage"
    if re.search(r"\(([\w-]+ )*film\)$", t):
        return "film"
    if re.search(r"\((novel|novella)\)$", t):
        return "novel"
    m = r.get("medium", "unknown")
    return {"tv film": "tv film", "film": "film", "serial": "serial", "novel": "novel", "short story": "short story",
            "tv series": "tv", "video game": "game", "character": "character", "comic": "comic", "graphic novel": "comic"}.get(m, "other")


WORKS = {"film", "tv film", "serial", "novel", "short story"}
rows = {}
for l in open(T3 / "candidates.jsonl"):
    r = json.loads(l)
    if r.get("missing") or not r.get("title"):
        continue
    rows.setdefault(r["title"], r)          # redirects can map two queries to one article
summ = {"articles_fetched": len(rows), "by_bucket": {}, "kinds": Counter()}
elig = []
for t, r in rows.items():
    k = kind(r)
    r["kind"] = k
    summ["kinds"][k] += 1
    ok = k in WORKS and r["n_sentences"] >= 8 and not r["in_dev"]
    r["eligible"] = ok
    if ok:
        elig.append(r)
b = defaultdict(Counter)
for r in rows.values():
    for bk in r["buckets"]:
        b[bk]["articles"] += 1
        b[bk][f"kind_{r['kind']}"] += 1
        if r["eligible"]:
            b[bk]["eligible"] += 1
            b[bk]["eligible_sentences_capped40"] += r["n_sentences_capped"]
            if r["batman_lineage"]:
                b[bk]["eligible_batman_lineage"] += 1
        if r["in_dev"]:
            b[bk]["in_dev"] += 1
summ["by_bucket"] = {k: dict(v) for k, v in sorted(b.items())}
summ["eligible_total"] = len(elig)
summ["eligible_sentences_capped40"] = sum(r["n_sentences_capped"] for r in elig)
summ["eligible_batman_lineage"] = sum(r["batman_lineage"] for r in elig)
summ["eligible_kinds"] = Counter(r["kind"] for r in elig)
only_vig = [r for r in elig if r["buckets"] == ["vigilante_film"]]
summ["eligible_vigilante_only"] = dict(n=len(only_vig), sentences=sum(r["n_sentences_capped"] for r in only_vig))
core = [r for r in elig if r["buckets"] != ["vigilante_film"]]
summ["eligible_without_vigilante_only"] = dict(n=len(core), sentences=sum(r["n_sentences_capped"] for r in core))
yrs = [r["year"] for r in elig if r.get("year")]
summ["eligible_decades"] = dict(sorted(Counter(y // 10 * 10 for y in yrs).items()))
summ["in_dev_titles"] = sorted(t for t, r in rows.items() if r["in_dev"])
rand = [json.loads(l) for l in open(T3 / "random_films.jsonl")]
acc = [r for r in rand if r.get("status") == "accepted"]
summ["random_sample"] = dict(tried=len(rand), status=Counter(r["status"] for r in rand), accepted=len(acc),
                            sentences_capped40=sum(r["n_sentences_capped"] for r in acc),
                            median_sentences=sorted(r["n_sentences"] for r in acc)[len(acc) // 2] if acc else None,
                            decades=dict(sorted(Counter((r.get("year") or r.get("pop_year")) // 10 * 10 for r in acc).items())),
                            population=json.load(open(T3 / "random_population.json"))["n"])
json.dump(summ, open(T3 / "pool_summary.json", "w"), indent=1, default=dict)
with open(T3 / "pool_eligible.jsonl", "w") as f:
    for r in sorted(elig, key=lambda r: r["title"]):
        f.write(json.dumps({k: r[k] for k in ("title", "kind", "shortdesc", "year", "n_sentences", "n_sentences_capped", "buckets",
                                               "batman_lineage", "wikibase_item")}, ensure_ascii=False) + "\n")
print(json.dumps(summ, indent=1, default=dict))
