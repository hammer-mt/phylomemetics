"""What changed a result: the rule, the model, or the pipeline (PLAN.md, Section 9).

Each primary statistic for every model under both rules, beside v1's published numbers:

  v1 published     4-bit weights, 8-bit storage, MLX; v1's rule            (results/experiments.json)
  v1's rule        full precision, PyTorch; v1's rule on the raw product   (results/v2/experiments_v1rule.json)
  v2's rule        full precision; strong activations against a scale      (results/v2/experiments_v2.json)

The contrasts are descriptive. A difference between two cells is given with a bootstrap interval over works (the 32
Batman works for E1, the 9 adaptations for E2), resampled in pairs; nothing is tested.

    cd study && ../.venv/bin/python v2/analysis/contrasts.py

Writes results/v2/contrasts.json.
"""
import json

import numpy as np

import common as C

MODELS = ["qwen3-8b", "llama3.1-8b", "qwen3-32b", "llama3.3-70b"]


def cell(res, rule):
    """The primary statistics of one model under one rule, and the per-work values the bootstrap needs."""
    if rule == "v2":
        a = res["E1"]["canon_share"]
        e1, per = a["A_name_free"]["hero_film_cousins"], a["A_name_free"]["per_batman_vs_hero"]
        extra = {"E1_verdict": a["verdict"], "E1_openings_auc": a.get("openings", {}).get("hero_film_cousins", {}).get("auc"),
                 "E1_null_p95": a["A_name_free"].get("null_label_shuffle", {}).get("p95"),
                 "E1_centroid_auc": res["E1"]["unit_centroid"]["A_name_free"]["hero_film_cousins"]["auc"],
                 "E2_evaluable": res["E2"]["evaluable"]["evaluable"], "length_controlled": True}
        resid = res["E2"]["residual_test_beyond_words"]["decision_both_word_measures"]
    else:
        e1, per = res["E1"]["A_name_free"]["hero_film_cousins"], res["E1"]["A_name_free"]["per_batman_vs_hero"]
        extra = {"E1_null_p95": res["E1"].get("null_label_shuffle", {}).get("p95"), "length_controlled": False}
        resid = res["E2"]["residual_test_beyond_words"]["neurons_resid_on_both"]
    e2 = res["E2"]["primary"]
    return dict(extra, E1_auc=e1["auc"], E1_ci95=e1["ci95"], E2_share=e2["chance"]["observed_unrounded"], E2_exact_p=e2["chance"]["joint_relabelling_exact"]["p"],
                E2_within_era_p=e2["chance"]["within_era"]["p"], sequels_top2=res["E2"]["sequels_name_free"]["top2"],
                residual_share=resid["chance"]["observed_unrounded"], residual_exact_p=resid["chance"]["joint_relabelling_exact"]["p"],
                residual_within_era_p=resid["chance"]["within_era"]["p"], E3_rho=res["E3"]["primary"]["rho"], E3_p=res["E3"]["primary"]["p_one_sided"],
                _e1_by_work=per, _e2_by_adaptation={r["target"]: r["share_of_comics_beating_source"] for r in e2["rows"]})


def paired(a, b, rng, n=4000):
    keys = [k for k in a if k in b]
    d = np.array([a[k] - b[k] for k in keys])
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return {"difference": float(d.mean()), "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])], "n": len(keys)}


def main():
    rng = np.random.default_rng(C.SEED + 51)
    v1 = json.load(open("results/experiments.json"))
    r1 = json.load(open(C.RESULTS / "experiments_v1rule.json")) if (C.RESULTS / "experiments_v1rule.json").exists() else {}
    r2 = json.load(open(C.RESULTS / "experiments_v2.json"))
    cells = {}
    for m in MODELS:
        if m in v1:
            cells[f"{m} | v1 published"] = cell(v1[m], "v1")
        if m in r1:
            cells[f"{m} | full precision, v1 rule"] = cell(r1[m], "v1")
        if m in r2:
            cells[f"{m} | full precision, v2 rule"] = cell(r2[m], "v2")
    pairs = []
    for m in MODELS:
        pairs += [(f"{m} | full precision, v1 rule", f"{m} | v1 published", "pipeline (4-bit and 8-bit storage against full precision), same model and rule"),
                  (f"{m} | full precision, v2 rule", f"{m} | full precision, v1 rule", "rule, same model")]
    pairs += [("qwen3-32b | full precision, v2 rule", "qwen3-8b | full precision, v2 rule", "model size, v2 rule"),
              ("qwen3-32b | full precision, v1 rule", "qwen3-8b | full precision, v1 rule", "model size, v1 rule"),
              ("llama3.3-70b | full precision, v2 rule", "llama3.1-8b | full precision, v2 rule", "model size (Llama), v2 rule")]
    contrasts = []
    for a, b, what in pairs:
        if a in cells and b in cells:
            contrasts.append({"contrast": what, "a": a, "b": b, "E1_auc": paired(cells[a]["_e1_by_work"], cells[b]["_e1_by_work"], rng),
                              "E2_share": paired(cells[a]["_e2_by_adaptation"], cells[b]["_e2_by_adaptation"], rng),
                              "E3_rho_difference": cells[a]["E3_rho"] - cells[b]["E3_rho"]})
    new = {k: c for k, c in cells.items() if "published" not in k}
    out = {"cells": {k: {kk: vv for kk, vv in c.items() if not kk.startswith("_")} for k, c in cells.items()}, "contrasts": contrasts,
           "new_cells": len(new), "new_cells_E2_exact_p_below_0.05": int(sum(c["E2_exact_p"] < 0.05 for c in new.values())),
           "new_cells_residual_exact_p_below_0.05": int(sum(c["residual_exact_p"] < 0.05 for c in new.values())),
           "new_cells_E3_p_below_0.05": int(sum(c["E3_p"] < 0.05 for c in new.values())),
           "expected_by_chance_per_test": 0.05 * len(new),
           "note": "E1 cells under v1's rule have no openings version and are not length-controlled"}
    C.save(out, "contrasts.json")
    for k, c in out["cells"].items():
        print(f"{k:42s} E1 {c['E1_auc']:.3f} | E2 {c['E2_share']:.3f} p {c['E2_exact_p']:.4f} | residual p {c['residual_exact_p']:.4f} | seq {c['sequels_top2']} | E3 {c['E3_rho']:.2f}")


if __name__ == "__main__":
    main()
