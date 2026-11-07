"""E1, E2, the residual test and E3 at every pre-specified version of the presence rule (PLAN.md, Section 14, items 2
and 4 to 8). E2's inner choices (band, hub adjustment, time rule, source rule) and its five nulls are in
multiverse_v2.py.

The grid: threshold (20, 30, 40, 50% of scale) x score (strongest sentence, mean of three) x base-rate cap (none, 30,
20, 10, 5, 3, 1%), the cap applied to all three experiments. These 56 cells are "the sensitivity settings" that the
residual test must pass in at least half of. Beside them, at the primary threshold and score with the primary caps
(3% for E1, none for E2 and E3): the reference-only scale, the fifth-example scale, the openings (length), the
placeholder codes left in, the token after each code masked too, the raw product with the echo regime left in, weak
poles left out (under 10% and under 25% of the neuron's stronger pole), and units scaled by the supplement left out.

    cd study && ../.venv/bin/python v2/analysis/sensitivity_v2.py [model-tag ...]

Writes results/v2/sensitivity_<tag>.json.
"""
import sys
import time

import numpy as np

import common as C

import experiments_v2 as E

THETAS, SCORES = [0.2, 0.3, 0.4, 0.5], ["top1", "top3"]


def flat(r, **extra):
    a = r["E1"]["canon_share"]["A_name_free"]
    c = r["E1"]["unit_centroid"]["A_name_free"]
    e2, rs = r["E2"]["summary"], r["E2"]["residual_both_word_measures"]
    return dict(extra, **r["setting"], n_units_present=r["n_units_present_in_some_work"],
                E1_auc=a["hero_film_cousins"]["auc"], E1_ci95=a["hero_film_cousins"]["ci95"], E1_null_p95=a["null_label_shuffle"]["p95"],
                E1_canon_units_median=a.get("canon_units_median"), E1_centroid_auc=c["hero_film_cousins"]["auc"],
                E1_centroid_null_p95=c["null_label_shuffle"]["p95"],
                E2_evaluable=r["E2"]["evaluable"]["evaluable"], E2_optional_units=r["E2"]["evaluable"]["n_optional_units"],
                E2_share=e2["share"], E2_share_ties_half=r["E2"]["share_ties_half"], E2_exact_p=e2["exact_p"], E2_within_era_p=e2["within_era_p"],
                sequels_top2=r["E2"]["sequels_top2"], residual_share=rs["share"], residual_exact_p=rs["exact_p"], residual_within_era_p=rs["within_era_p"],
                residual6_exact_p=r["E2"]["residual_six_word_measures"]["exact_p"],
                E3_rho=r["E3"]["rho"], E3_p=r["E3"]["p_one_sided"], E3_units=r["E3"]["n_units"])


def run_model(tag):
    t0 = time.time()
    grid, extra = [], []
    for th in THETAS:
        for sc in SCORES:
            st = C.Setting(tag, th, sc, icon=False)
            for cap in C.CAPS:
                grid.append(flat(E.run_setting(st, None, cap, cap, full=False), cell="grid", cap="none" if cap is None else cap))
            del st
    for name, kw in (("reference-only scale", {"which": "ref"}), ("fifth-example scale", {"which": "ref5"}), ("openings", {"variant": "cut"}),
                     ("codes left in", {"variant": "codes"}), ("token after each code masked too", {"variant": "after"}),
                     ("echo regime left in", {"variant": "raw"}), ("poles under 10% of the stronger pole left out", {"min_dominance": 0.10}),
                     ("poles under 25% of the stronger pole left out", {"min_dominance": 0.25}),
                     ("units scaled by the supplement left out", {"drop_supplement": True})):
        st = C.Setting(tag, C.THETA, "top1", icon=False, **kw)
        extra.append(flat(E.run_setting(st, None, C.CAP, None, full=False), cell=name))
        del st
    prim = [r for r in grid if r["theta"] == C.THETA and r["score"] == "top1"]
    auc = np.array([r["E1_auc"] for r in grid])
    out = {"model": tag,
           "grid": "threshold (0.2, 0.3, 0.4, 0.5) x score (top1 = strongest sentence, top3 = mean of three) x cap (none, 0.30, 0.20, 0.10, 0.05, 0.03, 0.01), "
                   "the cap applied to E1, E2 and E3 alike",
           "primary_cells": {"E1": next(r for r in prim if r["cap"] == C.CAP), "E2_residual_E3": next(r for r in prim if r["cap"] == "none")},
           "n_settings": len(grid),
           "residual_settings_exact_p_below_0.05": int(sum(r["residual_exact_p"] < 0.05 for r in grid)),
           "residual_passes_in_at_least_half": bool(sum(r["residual_exact_p"] < 0.05 for r in grid) >= len(grid) / 2),
           "residual_within_era_p_below_0.05": int(sum(r["residual_within_era_p"] < 0.05 for r in grid)),
           "E2_settings_exact_p_below_0.05": int(sum(r["E2_exact_p"] < 0.05 for r in grid)),
           "E2_settings_evaluable": int(sum(r["E2_evaluable"] for r in grid)),
           "E1_auc_range": [float(auc.min()), float(auc.max())], "E1_settings_auc_at_least_0.80": int((auc >= 0.8).sum()),
           "E1_settings_above_null_p95": int(sum(r["E1_auc"] > r["E1_null_p95"] for r in grid)),
           "E3_rho_range": [float(min(r["E3_rho"] for r in grid)), float(max(r["E3_rho"] for r in grid))],
           "E3_settings_p_below_0.05": int(sum(r["E3_p"] < 0.05 for r in grid)),
           "rows": grid, "other_versions_at_the_primary_setting": extra, "seconds": round(time.time() - t0)}
    C.save(out, f"sensitivity_{tag}.json")
    print(f"{tag}: residual p < 0.05 in {out['residual_settings_exact_p_below_0.05']} of {len(grid)} settings; E2 in "
          f"{out['E2_settings_exact_p_below_0.05']}; E1 AUC {out['E1_auc_range']}; E3 rho {out['E3_rho_range']}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for tag in sys.argv[1:] or [t for t in C.MODELS if (C.ACTS / t / "scale.npy").exists()]:
        run_model(tag)
