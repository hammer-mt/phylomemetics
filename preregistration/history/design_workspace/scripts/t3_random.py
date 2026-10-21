"""T3 pool scout, step 3: a reproducible random sample of ~1,000 films of all kinds with a plot section.

Population: direct article members of 'Category:YYYY films' for YYYY = 1915..2024 (English Wikipedia, as listed
on the scout date). Order: sha256('phylomemetics-T3-random-v1|' + title). Walk the order, skip titles in the DEV
corpus, the T3 category pool or the Shakespeare scout, fetch, and accept a film when its plot section (same rule as
fetch_corpus.py) has >= 8 sentences; stop at 1,000 accepted. Every title tried is logged with its status, so the
sample can be rebuilt exactly from random_population.json. Raw plot sections are saved, not analysed.

usage: python t3_random.py population | sample"""
import hashlib
import json
import sys
import time

sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import SCOUT, category_members
from t3_common import row_for, SKIP, DEV_TITLES

OUT = SCOUT / "t3"
SALT = "phylomemetics-T3-random-v1|"
YEARS = range(1915, 2025)
TARGET = 1000


def population():
    pop = {}
    for y in YEARS:
        ms = [m["title"] for m in category_members(f"Category:{y} films") if m["ns"] == 0]
        for t in ms:
            pop.setdefault(t, y)
        print(y, len(ms), flush=True)
    json.dump({"retrieved": time.strftime("%Y-%m-%d"), "rule": "direct ns=0 members of Category:YYYY films, 1915-2024; first year wins",
               "salt": SALT, "n": len(pop), "titles": pop}, open(OUT / "random_population.json", "w"), ensure_ascii=False)
    print(len(pop), "films in population")


def sample():
    pop = json.load(open(OUT / "random_population.json"))["titles"]
    pool = set(json.load(open(OUT / "category_members.json"))["members"])
    shk = set(json.loads(l).get("title") for l in open(SCOUT / "films_all.jsonl"))
    order = sorted(pop, key=lambda t: hashlib.sha256((SALT + t).encode()).hexdigest())
    path = OUT / "random_films.jsonl"
    done = {}
    if path.exists():
        for l in open(path):
            r = json.loads(l)
            done[r["query"]] = r
    accepted = sum(1 for r in done.values() if r.get("status") == "accepted")
    with open(path, "a") as f:
        for rank, t in enumerate(order):
            if accepted >= TARGET:
                break
            if t in done:
                continue
            if SKIP.match(t) or t in DEV_TITLES or t in pool or t in shk:
                r = dict(query=t, rank=rank, year=pop[t], status="excluded_overlap" if not SKIP.match(t) else "excluded_list")
            else:
                r = row_for(t)
                r.update(rank=rank, pop_year=pop[t])
                if r.get("missing"):
                    r["status"] = "missing"
                elif r["n_sentences"] >= 8:
                    r["status"] = "accepted"
                    accepted += 1
                else:
                    r["status"] = "no_plot" if not r["plot_section"] else "short_plot"
                    r.pop("sentences", None)
                    r.pop("raw_section", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if rank % 100 == 0:
                print(rank, "tried;", accepted, "accepted", flush=True)
    print("accepted", accepted)


if __name__ == "__main__":
    {"population": population, "sample": sample}[sys.argv[1]]()
