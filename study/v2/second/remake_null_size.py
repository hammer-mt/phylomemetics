"""Remakes decision test: how often is pure noise in the units' place rejected? (v2/second/PLAN.md Section 16, entry 2)

    cd study
    ../.venv/bin/python v2/second/remake_null_size.py [draws] [workers]     -> results/v2/second/remakes_null_size.json

Not part of the frozen plan. It was written after the freeze, with words and random numbers only, when the checks that
were still running at the freeze (random word-only measures, noisy copies) were rejected 8 to 9 times in 100 at a nominal
0.05. Here the units' column is replaced by independent standard normal noise in every cell, and put through the registered
test on the primary set in its primary frame: the 39 registered word columns (and, for comparison, the designers' 31),
penalty 3, one family left out at a time, family sums, sign flips. No model and no unit is involved.

The file gives, for each set of columns, the share of draws with p below 0.05 and below 0.01, the spread of Z, the share
of draws whose Z is above the largest Z of the stand-in panel (condition 5 of the plan), and every draw's Z in order, so
that a unit result can be placed in this distribution: `noise_p(Z)` is the share of noise draws at or above Z.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "v2/second")
import remakes as R      # noqa: E402

OUT = Path("results/v2/second/remakes_null_size.json")
SEED0 = 9_000_000


def job(seed):
    test = R.G["test"]
    rng = np.random.default_rng(seed)
    r = test.slope_test(R.synth(test, R.G["base"], 1.0, 0.0, rng), n_flips=R.SIM_FLIPS, seed=seed)
    return r["p"], r["Z"], r["T"], float((r["family_sums"] ** 2).sum())


def noise_p(z, file=OUT, columns="39 columns (registered)"):
    """The share of the pure-noise draws whose Z is at or above z (with the observed value counted as one draw)."""
    zs = np.asarray(json.load(open(file))["by_columns"][columns]["Z_sorted"])
    return float((1 + (zs >= z).sum()) / (1 + len(zs)))


def main():
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    W = json.load(open("results/v2/second/remakes_words.json"))
    wc = W["word_columns"]
    zmax = W["claim_conditions_that_words_decide"]["stand_ins_largest_Z_the_units_must_exceed"]["Z"]
    C = R.Corpus()
    M, _ = R.word_measures(C, R.CACHE, must_exist=True)
    Mx = {**M, **R.load_added(R.CACHE, wc["added_measures.npz"], wc["added_measures"])}
    out = {"what": "Pure noise in the units' place, through the registered remakes decision test on the primary set in its primary frame. "
                   "Made by v2/second/remake_null_size.py after the freeze, with no model and no unit (PLAN.md Section 16).",
           "draws": draws, "first_seed": SEED0, "sign_flips_per_draw": R.SIM_FLIPS, "stand_ins_largest_Z": zmax,
           "cells_in_the_slope": "cells the word model gives a probability under 1e-08 are left out, as in every simulation of remakes.py",
           "sha256": {"remakes.py": R.sha256("v2/second/remakes.py"), "remake_test.py": R.sha256("v2/second/remake_test.py")}, "by_columns": {}}
    for label, cols in (("31 columns (the designers')", None), ("39 columns (registered)", [tuple(c) for c in wc["columns"]])):
        an = R.Analysis(C, Mx, "primary", "set", columns=cols)
        an.keep_folds()
        R.G.update(test=an.test, base=np.zeros(an.test.S.n))
        t0 = time.time()
        with R.workers_pool(workers) as pool:
            res = np.array(pool.map(job, range(SEED0, SEED0 + draws), chunksize=8))
        p, Z, T, ss = res.T
        out["by_columns"][label] = {
            "columns": len(an.test.names), "effective_pairs_left_open": an.test.unsolved,
            "below_0.05": int((p < 0.05).sum()), "rate_0.05": float((p < 0.05).mean()),
            "below_0.01": int((p < 0.01).sum()), "rate_0.01": float((p < 0.01).mean()), "above_0.95": float((p > 0.95).mean()),
            "mean_Z": float(Z.mean()), "sd_Z": float(Z.std()), "share_Z_above_the_stand_ins_largest_Z": float((Z > zmax).mean()),
            "share_p_below_0.05_and_Z_above_the_stand_ins_largest_Z": float(((p < 0.05) & (Z > zmax)).mean()),
            "Z_quantiles": {q: float(np.quantile(Z, float(q))) for q in ("0.9", "0.95", "0.975", "0.99", "0.999")},
            "sd_T": float(T.std()), "root_mean_sum_of_squares_of_family_sums": float(np.sqrt(ss.mean())),
            "Z_sorted": [round(float(z), 4) for z in np.sort(Z)], "seconds": round(time.time() - t0)}
        print(label, json.dumps({k: v for k, v in out["by_columns"][label].items() if k != "Z_sorted"}), flush=True)
        an.drop_folds()
        json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
