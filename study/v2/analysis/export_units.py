"""One table of the explained units: what each is, and which works carry it (the data behind the genome figure).

    cd study && ../.venv/bin/python v2/analysis/export_units.py <model-tag>

For every unit chosen for explanation: its set, layer, title, kind, base rate, the Batman works and relatives where
it is present on the original texts and on the name-free texts, the counts of hero films and ordinary films that
carry it, and, once the tests are scored, how its explanation did. Presence is the primary rule (strongest sentence at
least 30% of scale, codes masked in the name-free texts). Only the chosen units' columns are read, so this is light.
Writes results/v2/<tag>/units_table.json, with the works in date order.
"""
import json
import sys

import numpy as np

import common as C

import experiments as X


def main(tag):
    d = C.RESULTS / tag
    chosen = json.load(open(d / "explain_units.json"))["units"]
    ex = json.load(open(d / "explanations.json"))["units"] if (d / "explanations.json").exists() else {}
    kinds = json.load(open(d / "kinds.json"))["units"] if (d / "kinds.json").exists() else {}
    tests = {str(r["unit"]): r for r in json.load(open(d / "explanation_tests.json"))["units"]} if (d / "explanation_tests.json").exists() else {}
    cut = int(np.ceil(C.THETA * C.STEP - 1e-6))
    units = np.array([int(c["unit"]) for c in chosen])
    take = np.sort(units)
    where = {int(u): k for k, u in enumerate(take)}
    P = {t: np.asarray(np.load(C.ACTS / tag / C.score_file(t), mmap_mode="r")[:, take]) >= cut for t in ("orig", "anon")}
    base = C.base_rate(tag)[take]
    bat, hero, lit = set(X.BAT.tolist()), X.HERO_FILM_COUSINS, X.LIT_COUSINS
    rel = np.concatenate([lit, hero])
    films = np.setdiff1d(np.arange(P["orig"].shape[0]), np.concatenate([X.BAT, rel]))
    order = [int(w) for w in sorted(X.BAT, key=lambda w: (X.year[w], w))] + [int(w) for w in sorted(rel, key=lambda w: (X.year[w], w))]
    works = [{"label": X.label[w], "year": int(X.year[w]), "group": "batman" if w in bat else "relative",
              "medium": X.works[w].get("medium"), "era": X.works[w].get("era")} for w in order]
    rows = []
    for c in chosen:
        u = int(c["unit"])
        k = where[u]
        po, pa = P["orig"][order, k].astype(int).tolist(), P["anon"][order, k].astype(int).tolist()
        e, kd, t = ex.get(str(u), {}), kinds.get(str(u), {}), tests.get(str(u), {})
        rows.append({"unit": u, "name": c["name"], "set": c["set"], "layer": c["layer"], "weak_pole": c.get("weak_pole", False),
                     "title": e.get("title"), "kind": kd.get("kind"), "detects": kd.get("detects"), "base_rate": float(base[k]),
                     "batman_works_original": int(sum(po[:32])), "batman_works_name_free": int(sum(pa[:32])),
                     "relatives_original": int(sum(po[32:])), "relatives_name_free": int(sum(pa[32:])),
                     "hero_films_original": int(P["orig"][hero, k].sum()), "hero_films_name_free": int(P["anon"][hero, k].sum()),
                     "ordinary_films_original": int(P["orig"][films, k].sum()), "ordinary_films_name_free": int(P["anon"][films, k].sum()),
                     "present_original": po, "present_name_free": pa,
                     "outcome": t.get("outcome"), "idea_bound": t.get("idea_bound"), "testable_out_of_context": t.get("testable_out_of_context")})
    by_set = {}
    for s in sorted({r["set"] for r in rows}):
        g = [r for r in rows if r["set"] == s]
        by_set[s] = {"n": len(g), **{f"{k}_median": float(np.median([r[k] for r in g])) for k in
                                     ("batman_works_original", "batman_works_name_free", "hero_films_original", "hero_films_name_free", "ordinary_films_name_free")}}
    C.save({"model": tag, "works": works, "n_batman_works": 32, "n_hero_films": int(len(hero)), "n_ordinary_films": int(len(films)), "by_set": by_set, "units": rows},
           f"{tag}/units_table.json")
    print(tag, len(rows), "units;", sum(r["title"] is not None for r in rows), "with a title")


if __name__ == "__main__":
    main(sys.argv[1])
