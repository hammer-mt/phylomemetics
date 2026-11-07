"""One table of the explained units: what each is, and which works carry it (the data behind the genome figure).

    cd study && ../.venv/bin/python v2/analysis/export_units.py <model-tag>

For every unit chosen for explanation: its set, layer, title, kind, base rate, the Batman works and relatives where
it is present on the original texts and on the name-free texts, and, once the tests are scored, how its explanation
did. Writes results/v2/<tag>/units_table.json, with the works in date order.
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
    st = C.Setting(tag, icon=False)
    col = {int(u): k for k, u in enumerate(st.units)}
    rel = np.concatenate([X.LIT_COUSINS, X.HERO_FILM_COUSINS])
    order = [int(w) for w in sorted(X.BAT, key=lambda w: (X.year[w], w))] + [int(w) for w in sorted(rel, key=lambda w: (X.year[w], w))]
    works = [{"label": X.label[w], "year": int(X.year[w]), "group": "batman" if w in set(X.BAT.tolist()) else "relative",
              "medium": X.works[w].get("medium"), "era": X.works[w].get("era")} for w in order]
    rows = []
    for c in chosen:
        u = int(c["unit"])
        k = col.get(u)
        po = st.P["orig"][order, k].astype(int).tolist() if k is not None else [0] * len(order)
        pa = st.P["anon"][order, k].astype(int).tolist() if k is not None else [0] * len(order)
        e, kd, t = ex.get(str(u), {}), kinds.get(str(u), {}), tests.get(str(u), {})
        rows.append({"unit": u, "name": c["name"], "set": c["set"], "layer": c["layer"], "weak_pole": c.get("weak_pole", False),
                     "title": e.get("title"), "kind": kd.get("kind"), "detects": kd.get("detects"),
                     "base_rate": float(st.base[k]) if k is not None else None,
                     "batman_works_original": int(sum(po[:32])), "batman_works_name_free": int(sum(pa[:32])),
                     "relatives_original": int(sum(po[32:])), "relatives_name_free": int(sum(pa[32:])),
                     "present_original": po, "present_name_free": pa,
                     "outcome": t.get("outcome"), "idea_bound": t.get("idea_bound"), "testable_out_of_context": t.get("testable_out_of_context")})
    C.save({"model": tag, "works": works, "n_batman_works": 32, "units": rows}, f"{tag}/units_table.json")
    print(tag, len(rows), "units;", sum(r["title"] is not None for r in rows), "with a title")


if __name__ == "__main__":
    main(sys.argv[1])
