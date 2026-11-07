"""Jobs for the ablation test (PLAN.md, Section 11): does the model use Batman's canon to know it is Batman?

For each of the 32 name-free Batman summaries the units of that work's E1 canon (chosen from the other 31 works) are
switched off, and so are 200 matched random sets. Every canon unit gets one partner per set: a unit of the same
layer and pole that is in no canon, as near as possible in the number of name-free Batman works where it is present
(drawn from the 20 nearest not already used in that set). Partners come from all units. Two more controls are
built the same way: 200 sets that are also matched on how active the unit is in these texts (its mean peak share and
its mean peak activation over the name-free Batman works; PLAN.md Section 18, change 14), and 50 sets drawn from
distinctive units only. The same partner stands in for its canon unit in every work. Seven hero-film relatives get
the canon of all 32 works and its partners, scored on their own hero.

    cd study && ../.venv/bin/python v2/analysis/build_ablation.py <model-tag>

Needs results/v2/canon_<tag>.npz (experiments_v2.py). Writes v2/cache/jobs/ablate_<tag>.json (the confirmatory run) and
the secondaries ablate_<tag>_all_positions.json, _original.json, _strong.json and _weak.json, with ablate_<tag>_partners.npz
(to upload), and results/v2/<tag>/ablation_design.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

import common as C

import experiments as X
from build_probe_jobs import BATMAN, HEROES, Q_HERO, text_of

N_SETS, N_SETS_DISTINCTIVE, NEAREST = 200, 50, 20


def main(tag):
    rng = np.random.default_rng(C.SEED)
    z = np.load(C.RESULTS / f"canon_{tag}.npz")
    folds = [z[f"fold_{i:02d}"].astype(np.int64) for i in range(len(X.BAT))]
    canon_all = z["canon_all32"].astype(np.int64)
    units = np.unique(np.concatenate(folds + [canon_all]))
    L, D = C.geometry(tag)
    U = L * D
    st = C.Setting(tag, icon=False)
    n_nf = np.zeros(2 * U, np.int16)
    n_nf[st.units] = st.P["anon"][X.BAT].sum(0)
    full = np.load(C.ACTS / tag / "anon_main_top1.u8.npy", mmap_mode="r")
    ms = np.asarray(full[X.BAT], dtype=np.float32).mean(0) / C.STEP            # mean peak share over the name-free Batman works, every unit
    lr = np.log(ms * C.scale(tag) + 0.05)                                       # and the log of the mean peak activation
    group = np.arange(2 * U) // D                                    # layer and pole together
    can_fire = C.scale(tag) > 0
    can_fire[units] = False
    pools = {"any": (can_fire, N_SETS), "magnitude": (can_fire, N_SETS),
             "distinctive": (can_fire & (C.base_count(tag) <= C.CAP * C.n_plots(tag) + 1e-9), N_SETS_DISTINCTIVE)}
    partners, gap = {}, {}
    for name, (ok, n_sets) in pools.items():
        partners[name] = np.zeros((n_sets, len(units)), np.int64)
        gap[name] = np.zeros((n_sets, len(units)), np.int16)
        for g in np.unique(group[units]):
            mine = np.flatnonzero(group[units] == g)
            pool = np.flatnonzero(ok[g * D:(g + 1) * D]) + g * D
            assert len(pool) >= len(mine) + NEAREST, f"too few candidates in layer group {g}"
            pn = n_nf[pool]
            for k in range(n_sets):
                free = np.ones(len(pool), bool)
                for j in rng.permutation(mine):
                    dist = np.abs(pn - n_nf[units[j]]).astype(np.float64) + rng.random(len(pool)) * 0.5      # random order among ties
                    if name == "magnitude":                 # also as active in these texts: mean peak share and mean peak activation
                        dist += 32 * np.abs(ms[pool] - ms[units[j]]) + 8 * np.abs(lr[pool] - lr[units[j]])
                    dist[~free] = np.inf
                    near = np.argpartition(dist, NEAREST)[:NEAREST]
                    c = near[rng.integers(NEAREST)]
                    free[c] = False
                    partners[name][k, j] = pool[c]
                    gap[name][k, j] = abs(int(pn[c]) - int(n_nf[units[j]]))
    out = Path("v2/cache/jobs")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"ablate_{tag}_partners.npz", units=units, partners=partners["any"], partners_magnitude=partners["magnitude"],
                        partners_distinctive=partners["distinctive"])
    pos = {int(u): i for i, u in enumerate(units)}
    members = [[pos[int(u)] for u in f] for f in folds]
    all32 = [pos[int(u)] for u in canon_all]
    conds = ["canon"] + [f"rand:{k}" for k in range(N_SETS)] + [f"magnitude:{k}" for k in range(N_SETS)] + [f"distinctive:{k}" for k in range(N_SETS_DISTINCTIVE)]
    short = ["canon"] + [f"rand:{k}" for k in range(20)] + [f"magnitude:{k}" for k in range(20)]

    def items(src, conditions):
        it = [{"id": X.label[b], "group": "batman", "text": text_of(src[b]), "answers": BATMAN, "members": members[i], "conditions": conditions}
              for i, b in enumerate(X.BAT)]
        for w in X.HERO_FILM_COUSINS:
            if X.label[w] in HEROES:
                it.append({"id": X.label[w], "group": "hero film", "text": text_of(src[w]), "answers": HEROES[X.label[w]], "members": all32, "conditions": conditions})
        return it

    base = {"question": Q_HERO, "partner_file": f"/workspace/data/jobs/ablate_{tag}_partners.npz", "reply_for": ["canon"], "reply_tokens": 8}
    json.dump(dict(base, where="story", items=items(X.neutral, conds)), open(out / f"ablate_{tag}.json", "w"), ensure_ascii=False)
    json.dump(dict(base, where="all", items=items(X.neutral, short)), open(out / f"ablate_{tag}_all_positions.json", "w"), ensure_ascii=False)
    json.dump(dict(base, where="story", items=items(X.works, short)), open(out / f"ablate_{tag}_original.json", "w"), ensure_ascii=False)
    for mode in ("strong", "weak"):
        json.dump(dict(base, where="story", mode=mode, items=items(X.neutral, short)), open(out / f"ablate_{tag}_{mode}.json", "w"), ensure_ascii=False)
    design = {"model": tag, "n_sets": N_SETS, "nearest": NEAREST, "canon_units_in_any_fold": int(len(units)), "canon_units_all32": int(len(canon_all)),
              "fold_canon_size_median": float(np.median([len(f) for f in folds])),
              "canon_units_name_free_works_mean": float(n_nf[units].mean()),
              "canon_summed_mean_peak_share": float(ms[units].sum()), "canon_summed_mean_peak_activation": float((ms * C.scale(tag))[units].sum()),
              "controls": {name: {"n_sets": int(partners[name].shape[0]), "partners_name_free_works_mean": float(n_nf[partners[name]].mean()),
                                  "summed_mean_peak_share": {"mean": float(ms[partners[name]].sum(1).mean()), "min": float(ms[partners[name]].sum(1).min()),
                                                             "max": float(ms[partners[name]].sum(1).max()),
                                                             "sets_at_or_above_the_canon": int((ms[partners[name]].sum(1) >= ms[units].sum()).sum())},
                                  "summed_mean_peak_activation": {"mean": float((ms * C.scale(tag))[partners[name]].sum(1).mean()),
                                                                  "max": float((ms * C.scale(tag))[partners[name]].sum(1).max()),
                                                                  "sets_at_or_above_the_canon": int(((ms * C.scale(tag))[partners[name]].sum(1) >= (ms * C.scale(tag))[units].sum()).sum())},
                                  "partner_gap_in_works_mean": float(gap[name].mean()), "partner_gap_in_works_p90": float(np.percentile(gap[name], 90)),
                                  "partners_base_rate_median": float(np.median(C.base_rate(tag)[partners[name]])),
                                  "distinct_partners_per_canon_unit_median": float(np.median([len(np.unique(partners[name][:, j])) for j in range(len(units))]))}
                           for name in partners},
              "canon_base_rate_median": float(np.median(C.base_rate(tag)[units])),
              "canon_units_by_layer_quarter": {f"q{q + 1}": int((((units % U) // D) * 4 // L == q).sum()) for q in range(4)},
              "secondaries": "every position, original texts, strong firings only, weak activity only: the canon and the first 20 sets of the two matched controls"}
    C.save(design, f"{tag}/ablation_design.json")
    print(tag, design)


if __name__ == "__main__":
    main(sys.argv[1])
