"""E2 under many reasonable versions of the v2 presence rule, with v1's five nulls, jackknife and stronger nulls
(PLAN.md, Section 14, items 1 and 3).

Presence varies over: threshold (20, 30, 40, 50% of scale) x score (strongest sentence, mean of three) x base-rate cap
(none, 30, 20, 10, 5, 3, 1%), plus the primary setting on the reference-only scale and on the fifth-example scale. Inside each,
v1's choices vary as before: frequency band, hub adjustment, time rule, source rule. All the statistics are computed
by v1's robustness/e2_multiverse.py, which is imported, not edited; only the presence matrix comes from here.
The base rate behind the cap is always the strongest-sentence rate. The primary cell has no cap (PLAN.md, Section 5).

    cd study && ../.venv/bin/python v2/analysis/multiverse_v2.py [model-tag ...]

Writes results/v2/robustness/e2_multiverse_<tag>.json, e2_jackknife_<tag>.json and e2_nulls_<tag>.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import common as C

sys.path.insert(0, "robustness")
import e2_multiverse as m          # noqa: E402

THETAS, SCORES, CAPS = [0.2, 0.3, 0.4, 0.5], ["top1", "top3"], C.CAPS
INNER = {"topk": ["ext"], "thr_j": ["ext"], "lo": [2, 3, 4], "hi": [8, 16, 24], "dedup": [False], "hub": [True, False],
         "time": [True, False], "rule": ["best", "mean", "first"]}
NULLS = ["p_joint", "p_era", "p_swap", "p_swap_tc", "p_target"]


def shared_draws():
    """The same random maps and target draws as e2_multiverse.main (seed 2026, same order)."""
    rng = np.random.default_rng(2026)
    src7 = sorted({m.CPOS[m.L[s]] for v in m.ADAPTATIONS.values() for s in v})
    inj = m.injections(range(m.NC), len(src7))
    mc_maps = m.expand(inj[rng.integers(0, len(inj), 20000)], src7)
    tdraws = np.argsort(rng.random((5000, len(m.TARGET_POOL))), axis=1)[:, :len(m.ADAPT)]
    era = m.era_maps([{"srcs": tuple(src7)}])
    flat9 = (np.arange(len(m.ADAPT))[None, :] * len(m.ADAPT) + m.perms(len(m.ADAPT))).astype(np.intp)
    return mc_maps, era, tdraws, flat9


def summarise(rs):
    if not rs:
        return {}
    p = {k: np.array([r[k] for r in rs]) for k in NULLS}
    return {"n_specs": len(rs), "frac_p_joint_lt_05": float((p["p_joint"] < .05).mean()), "frac_p_joint_lt_01": float((p["p_joint"] < .01).mean()),
            "frac_p_era_lt_05": float((p["p_era"] < .05).mean()), "frac_all_five_nulls_lt_05": float(np.all([p[k] < .05 for k in NULLS], axis=0).mean()),
            "median_share": float(np.median([r["share"] for r in rs])), "median_null_joint_mean": float(np.median([r["null_joint_mean"] for r in rs])),
            "max_p_joint": float(p["p_joint"].max()), "frac_sequel_control_passes": float(np.mean([r["sequel_top2"] >= 5 for r in rs])),
            "median_units": float(np.median([r["n_units"] for r in rs]))}


def run_model(tag):
    t0 = time.time()
    draws = shared_draws()
    out = Path(C.RESULTS / "robustness")
    out.mkdir(parents=True, exist_ok=True)
    recs, jack, nulls = [], {}, {}
    cells = [(th, sc, "study") for th in THETAS for sc in SCORES] + [(C.THETA, "top1", "ref"), (C.THETA, "top1", "ref5")]
    for th, sc, which in cells:
        st = C.Setting(tag, th, sc, which, icon=False)
        for text in ("anon", "orig"):
            if (th, sc, which) == (C.THETA, "top1", "study"):
                Pb = st.P[text][m.BAT]
                jack[text] = m.jackknife(Pb)
                nulls[text] = m.stronger_nulls(Pb)
                print(f"{tag}/{text} primary: share {jack[text]['primary']['share']} exact p {jack[text]['primary']['p_joint']:.5f} "
                      f"within-era p {jack[text]['primary']['p_era']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            for cap in (CAPS if which == "study" else [None]):
                Pb = st.P[text][m.BAT][:, st.distinctive(cap)]
                Sb, bars = {"ext": Pb.astype(np.float32)}, {"ext": {"ext": np.full(Pb.shape[1], 0.5, np.float32)}}
                for r in m.multiverse_cell(tag, text, Sb, bars, INNER, *draws):
                    del r["topk"], r["thr_j"], r["dedup"]
                    r.update(theta=th, score=sc, cap=("none" if cap is None else cap), scale=which)
                    recs.append(r)
        print(f"{tag} threshold {th} {sc} scale {which}: {len(recs)} specifications so far ({time.time() - t0:.0f}s)", flush=True)
        del st
    dims = ["text", "theta", "score", "cap", "scale", "lo", "hi", "hub", "time", "rule"]
    summary = {"model": tag, "grid": {"theta": THETAS, "score": SCORES, "cap": ["none" if c is None else c for c in CAPS], "inner": INNER},
               "overall": summarise(recs), "name_free": summarise([r for r in recs if r["text"] == "anon"]),
               "original": summarise([r for r in recs if r["text"] == "orig"]),
               "by_dimension": {d: {str(v): summarise([r for r in recs if r[d] == v]) for v in sorted({r[d] for r in recs}, key=str)} for d in dims},
               "name_free_by_dimension": {d: {str(v): summarise([r for r in recs if r[d] == v and r["text"] == "anon"])
                                              for v in sorted({r[d] for r in recs}, key=str)} for d in dims if d != "text"},
               "joint_failures_name_free": sorted([{k: r[k] for k in dims + ["share", "p_joint", "p_era", "n_units"]}
                                                   for r in recs if r["text"] == "anon" and r["p_joint"] >= .05], key=lambda r: -r["p_joint"])[:40],
               "primary_cells": {text: next({k: r[k] for k in ["share", "null_joint_mean", "p_joint", "p_era", "p_swap", "p_swap_tc", "p_target",
                                                               "placebo1989_mean", "ranks", "sequel_top2", "n_units"]}
                                            for r in recs if r["text"] == text and r["theta"] == C.THETA and r["score"] == "top1" and r["cap"] == "none"
                                            and r["scale"] == "study" and r["lo"] == 2 and r["hi"] == 16 and r["hub"] and r["time"] and r["rule"] == "best")
                                 for text in ("anon", "orig")},
               "columns": dims + ["n_units", "share", "null_joint_mean", "p_joint", "p_era", "p_swap", "p_swap_tc", "p_target", "sequel_top2"]}
    summary["rows"] = [[r[c] for c in summary["columns"]] for r in recs]
    spec = "primary: strongest sentence at least 30% of scale, every unit (no base-rate cap), band 2-16, hub on, time rule on, best source; p exact unless noted"
    m.dump(summary, out / f"e2_multiverse_{tag}.json", indent=None)
    m.dump({"spec": spec, "cells": jack}, out / f"e2_jackknife_{tag}.json")
    m.dump({"spec": spec, "cells": nulls}, out / f"e2_nulls_{tag}.json")
    o = summary["name_free"]
    print(f"{tag}: {summary['overall']['n_specs']} specifications; name-free: p_joint < .05 in {o['frac_p_joint_lt_05']:.1%}, "
          f"all five nulls in {o['frac_all_five_nulls_lt_05']:.1%}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for tag in sys.argv[1:] or [t for t in C.MODELS if (C.ACTS / t / "scale.npy").exists()]:
        run_model(tag)
