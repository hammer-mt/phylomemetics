"""E1, E2, the residual test and E3 at every pre-specified version of the presence rule (PLAN.md, Section 14, items 2
and 4 to 8). E2's inner choices (band, hub adjustment, time rule, source rule) and its five nulls are in
multiverse_v2.py.

The grid: threshold (20, 30, 40, 50% of scale) x score (strongest sentence, mean of three) x base-rate cap (none, 30,
20, 10, 5, 3, 1%), the cap applied to all three experiments. The 8 uncapped cells are the settings of the residual
test's decision rule (it must pass in at least half of them); the 48 capped cells are descriptive, and there the
regressors also include word presence under the same cap (PLAN.md Section 18, change 1). Beside them, at the primary threshold and score with the primary caps
(3% for E1, none for E2 and E3): the reference-only scale, the fifth-example scale, the openings (length), the
placeholder codes left in, the token after each code masked too, the raw product with the echo regime left in, weak
poles left out (under 10% and under 25% of the neuron's stronger pole), and units scaled by the supplement left out.

    cd study && ../.venv/bin/python v2/analysis/sensitivity_v2.py [model-tag ...]        (SENS_WORKERS=3 runs three settings at a time;
    SENS_ORDER=reverse lets a second process work from the other end of the list)

Writes results/v2/sensitivity_<tag>.json.
"""
import fcntl
import hashlib
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np

import common as C

import experiments_v2 as E

THETAS, SCORES = [0.2, 0.3, 0.4, 0.5], ["top1", "top3"]


def flat(r, **extra):
    a = r["E1"]["canon_share"]["A_name_free"]
    c = r["E1"]["unit_centroid"]["A_name_free"]
    e2, rs = r["E2"]["summary"], r["E2"]["residual_decision"]
    return dict(extra, **r["setting"], n_units_present=r["n_units_present_in_some_work"],
                E1_auc=a["hero_film_cousins"]["auc"], E1_ci95=a["hero_film_cousins"]["ci95"], E1_null_p95=a["null_label_shuffle"]["p95"],
                E1_canon_units_median=a.get("canon_units_median"), E1_centroid_auc=c["hero_film_cousins"]["auc"],
                E1_centroid_null_p95=c["null_label_shuffle"]["p95"],
                E2_evaluable=r["E2"]["evaluable"]["evaluable"], E2_optional_units=r["E2"]["evaluable"]["n_optional_units"],
                E2_share=e2["share"], E2_share_ties_half=r["E2"]["share_ties_half"], E2_exact_p=e2["exact_p"], E2_within_era_p=e2["within_era_p"],
                sequels_top2=r["E2"]["sequels_top2"], residual_share=rs["share"], residual_exact_p=rs["exact_p"], residual_within_era_p=rs["within_era_p"],
                residual_registered_exact_p=r["E2"]["residual_registered_two_word_measures"]["exact_p"],
                residual_every_measure_exact_p=r["E2"]["residual_every_word_and_code_measure"]["exact_p"],
                residual_cap_matched_words_exact_p=r["E2"].get("residual_cap_matched_words", {}).get("exact_p"),
                word_presence_same_cap_exact_p=r["E2"].get("word_presence_under_the_same_cap", {}).get("exact_p"),
                E3_rho=r["E3"]["rho"], E3_p=r["E3"]["p_one_sided"], E3_units=r["E3"]["n_units"], E3_evaluable=r["E3"]["evaluable"])


EXTRA = (("reference-only scale", {"which": "ref"}), ("fifth-example scale", {"which": "ref5"}), ("openings", {"variant": "cut"}),
         ("codes left in", {"variant": "codes"}), ("token after each code masked too", {"variant": "after"}),
         ("echo regime left in", {"variant": "raw"}), ("poles under 10% of the stronger pole left out", {"min_dominance": 0.10}),
         ("poles under 25% of the stronger pole left out", {"min_dominance": 0.25}),
         ("units scaled by the supplement left out", {"drop_supplement": True}))


def block(job):
    """One threshold and score with every cap, or one other version at the primary setting. Every setting seeds its own
    random numbers (experiments_v2.run_setting), so the blocks give the same values in any order or in parallel. A
    finished block is kept in v2/cache/sens/, so an interrupted run picks up where it stopped."""
    tag, kind, a, b = job
    path = Path("v2/cache/sens") / tag / (hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16] + ".json")
    if path.exists():
        return json.load(open(path))
    lock = open(Path("v2/cache") / "heavy.lock", "w")      # one memory-heavy analysis at a time on this machine (see v2/heavy.sh)
    fcntl.flock(lock, fcntl.LOCK_EX)
    if kind == "grid":
        st = C.Setting(tag, a, b, icon=False)
        rows = [flat(E.run_setting(st, None, cap, cap, full=False, centroid="all" if cap is None else "capped"), cell="grid",
                     cap="none" if cap is None else cap) for cap in C.CAPS]
    else:
        st = C.Setting(tag, C.THETA, "top1", icon=False, **b)
        rows = [flat(E.run_setting(st, None, C.CAP, None, full=False), cell=a)]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    json.dump(rows, open(tmp, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)
    del st
    lock.close()
    return json.load(open(path))                       # read back, so a fresh run and a resumed one return the same objects


def run_model(tag):
    t0 = time.time()
    jobs = [(tag, "grid", th, sc) for th in THETAS for sc in SCORES] + [(tag, "extra", name, kw) for name, kw in EXTRA]
    workers = int(os.environ.get("SENS_WORKERS", "1"))
    todo = jobs[::-1] if os.environ.get("SENS_ORDER") == "reverse" else jobs       # a second process can work from the other end
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
            pool.map(block, todo, chunksize=1)
    else:
        for j in todo:
            block(j)
    done = [block(j) for j in jobs]                    # all cached by now
    grid = [r for j, rows in zip(jobs, done) if j[1] == "grid" for r in rows]
    extra = [r for j, rows in zip(jobs, done) if j[1] == "extra" for r in rows]
    prim = [r for r in grid if r["theta"] == C.THETA and r["score"] == "top1"]
    auc = np.array([r["E1_auc"] for r in grid])
    dec = [r for r in grid if r["cap"] == "none"]
    capped = [r for r in grid if r["cap"] != "none"]
    ok = lambda r, key: bool(r["E2_evaluable"] and r[key] is not None and r[key] < 0.05)      # noqa: E731  a setting that is not evaluable does not pass
    e3 = [r for r in grid if r["E3_evaluable"]]
    out = {"model": tag,
           "grid": "threshold (0.2, 0.3, 0.4, 0.5) x score (top1 = strongest sentence, top3 = mean of three) x cap (none, 0.30, 0.20, 0.10, 0.05, 0.03, 0.01), "
                   "the cap applied to E1, E2 and E3 alike",
           "primary_cells": {"E1": next(r for r in prim if r["cap"] == C.CAP), "E2_residual_E3": next(r for r in prim if r["cap"] == "none")},
           "n_settings": len(grid), "n_decision_settings": len(dec), "n_decision_settings_evaluable": int(sum(r["E2_evaluable"] for r in dec)),
           "residual_settings_exact_p_below_0.05": int(sum(ok(r, "residual_exact_p") for r in dec)),
           "residual_passes_in_at_least_half": bool(sum(ok(r, "residual_exact_p") for r in dec) >= len(dec) / 2),
           "residual_within_era_p_below_0.05": int(sum(ok(r, "residual_within_era_p") for r in dec)),
           "residual_registered_form_exact_p_below_0.05": int(sum(ok(r, "residual_registered_exact_p") for r in dec)),
           "n_capped_settings": len(capped), "n_capped_settings_evaluable": int(sum(r["E2_evaluable"] for r in capped)),
           "residual_capped_settings_exact_p_below_0.05": int(sum(ok(r, "residual_exact_p") for r in capped)),
           "residual_capped_settings_with_cap_matched_words_below_0.05": int(sum(ok(r, "residual_cap_matched_words_exact_p") for r in capped)),
           "E2_settings_exact_p_below_0.05": int(sum(ok(r, "E2_exact_p") for r in grid)), "E2_uncapped_settings_exact_p_below_0.05": int(sum(ok(r, "E2_exact_p") for r in dec)),
           "E2_settings_evaluable": int(sum(r["E2_evaluable"] for r in grid)),
           "E1_note": "whole-work AUCs; the label-shuffle null does not control length (see the 'openings' version and length alone)",
           "E1_auc_range": [float(auc.min()), float(auc.max())], "E1_settings_auc_at_least_0.80": int((auc >= 0.8).sum()),
           "E1_settings_above_null_p95": int(sum(r["E1_auc"] > r["E1_null_p95"] for r in grid)),
           "E3_settings_evaluable": len(e3),
           "E3_rho_range": [float(min(r["E3_rho"] for r in e3)), float(max(r["E3_rho"] for r in e3))] if e3 else None,
           "E3_settings_p_below_0.05": int(sum(r["E3_p"] < 0.05 for r in e3)),
           "other_versions_note": "the 'echo regime left in' row keeps the placeholder codes too, so it is to be compared with the 'codes left in' row, not with the primary",
           "rows": grid, "other_versions_at_the_primary_setting": extra, "seconds": round(time.time() - t0)}
    C.save(out, f"sensitivity_{tag}.json")
    print(f"{tag}: residual p < 0.05 in {out['residual_settings_exact_p_below_0.05']} of {len(dec)} uncapped settings; E2 in "
          f"{out['E2_uncapped_settings_exact_p_below_0.05']} of them; E1 AUC {out['E1_auc_range']}; E3 rho {out['E3_rho_range']}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for tag in sys.argv[1:] or [t for t in C.MODELS if (C.ACTS / t / "scale.npy").exists()]:
        run_model(tag)
