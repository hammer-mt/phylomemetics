"""Choose the units to explain, by rule, before anyone reads an example (PLAN.md, Section 10).

  core            every core-canon family's representative (at least 29 of 32 Batman works, original texts), up to 60,
                  in order of the number of works and then family size
  common          30 families drawn at random from those present in 23 to 28 works
  optional        30 families drawn at random from those present in 6 to 22 works
  name_free_core  canon units still present in at least 70% of the name-free Batman works, up to 40
  discriminators  the 40 distinctive units that best separate name-free Batman works from the hero films
  inherited       20 units from E2's optional band present in both members of a documented source and adaptation pair
  band_rest       20 units from the rest of E2's optional band

Duplicates are dropped (a unit keeps the first set it was chosen for), which leaves at most 240 units.

    cd study && ../.venv/bin/python v2/analysis/select_units.py <model-tag>

Writes results/v2/<tag>/explain_units.json and explain_units.npy (each unit followed by its opposite pole).
"""
import gzip
import json
import sys

import numpy as np

import common as C

SETS = ["core", "common", "optional", "name_free_core", "discriminators", "inherited", "band_rest"]


def main(tag):
    rng = np.random.default_rng(C.SEED)
    fams = json.load(gzip.open(C.RESULTS / tag / "families.json.gz", "rt"))["families"]      # sorted by works, then family size
    pools = np.load(C.RESULTS / tag / "pools.npz")
    chosen, seen = [], set()

    def add(unit, kind, **extra):
        if int(unit) not in seen:
            seen.add(int(unit))
            chosen.append(dict(unit=int(unit), set=kind, **extra))

    for f in [f for f in fams if f["tier"] == "core"][:60]:
        add(f["unit"], "core", n_batman=f["n_batman"], n_members=f["n_members"])
    for tier in ("common", "optional"):
        pool = [f for f in fams if f["tier"] == tier]
        for i in sorted(rng.choice(len(pool), min(30, len(pool)), replace=False)) if pool else []:
            add(pool[i]["unit"], tier, n_batman=pool[i]["n_batman"], n_members=pool[i]["n_members"])
    for u in pools["name_free_core"][:40]:
        add(u, "name_free_core")
    for kind in ("discriminators", "inherited", "band_rest"):
        for u in pools[kind]:
            add(u, kind)
    L, D = C.geometry(tag)
    U = L * D
    for c in chosen:
        c["name"] = C.unit_name(tag, c["unit"])
        c["layer"] = (c["unit"] % U) // D
        c["opposite"] = c["unit"] + U if c["unit"] < U else c["unit"] - U
    by_set = {s: sum(c["set"] == s for c in chosen) for s in SETS}
    json.dump({"model": tag, "seed": C.SEED, "n": len(chosen), "by_set": by_set, "units": chosen}, open(C.RESULTS / tag / "explain_units.json", "w"), indent=1)
    np.save(C.RESULTS / tag / "explain_units.npy", np.array([x for c in chosen for x in (c["unit"], c["opposite"])], dtype=np.int64))
    print(tag, len(chosen), "units to explain:", by_set)


if __name__ == "__main__":
    main(sys.argv[1])
