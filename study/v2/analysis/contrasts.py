"""What changed a result: the rule, the model, or the pipeline (PLAN.md, Section 9).

Each primary statistic for every model under both rules, beside v1's published numbers:

  v1 published     4-bit weights, 8-bit storage, MLX; v1's rule            (results/experiments.json)
  v1's rule        full precision, PyTorch; v1's rule on the raw product   (results/v2/experiments_v1rule.json)
  v2's rule        full precision; strong activations against a scale      (results/v2/experiments_v2.json)

The contrasts are descriptive. A difference between two cells is given with a bootstrap interval, resampled in pairs:
over the 9 adaptations for E2, and over the 32 Batman works and the 8 hero films for E1 (PLAN.md Section 18, change
20). The files keep each work's and each film's mean, not the 32 x 8 table, so the E1 interval adds the two resampled
margins; on Qwen3-8B that agrees with resampling the full table to two decimal places. Nothing is tested.

The count of cells below 0.05 for the residual test uses v2's decision test in every cell. The cells share the 32
works, so a count is not that many confirmations; cells_joint.py gives the probability of each count under one null
applied to all cells at once.

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
        extra["E1_openings_shuffle_p"] = a.get("openings", {}).get("null_label_shuffle", {}).get("p_value")
        film = a["A_name_free"]["per_cousin_beaten_by"]
        resid = res["E2"]["residual_test_beyond_words"]["registered_both_word_measures"]      # v1's two-regressor form, the one every cell has
    else:
        e1, per, film = res["E1"]["A_name_free"]["hero_film_cousins"], res["E1"]["A_name_free"]["per_batman_vs_hero"], res["E1"]["A_name_free"]["per_cousin_beaten_by"]
        extra = {"E1_null_p95": res["E1"].get("null_label_shuffle", {}).get("p95"), "length_controlled": False}
        resid = res["E2"]["residual_test_beyond_words"]["neurons_resid_on_both"]
    dec = res["E2"]["residual_test_beyond_words"].get("decision_words_by_frequency")          # absent from v1's published file
    extra["residual_decision_exact_p"] = dec["chance"]["joint_relabelling_exact"]["p"] if dec else None
    extra["_e1_by_film"] = {k: 1 - v for k, v in film.items()}                                  # the share of Batman works that beat each film
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


def paired_two_way(a, b, rng, n=4000):
    """An E1 difference: the Batman works and the hero films are both resampled, with the same draw in both cells."""
    dr = np.array([a["_e1_by_work"][k] - b["_e1_by_work"][k] for k in a["_e1_by_work"]])
    dc = np.array([a["_e1_by_film"][k] - b["_e1_by_film"][k] for k in a["_e1_by_film"]])
    boot = [dr[rng.integers(0, len(dr), len(dr))].mean() + dc[rng.integers(0, len(dc), len(dc))].mean() - dr.mean() for _ in range(n)]
    return {"difference": float(dr.mean()), "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])], "n_works": len(dr), "n_films": len(dc),
            "interval": "Batman works and hero films resampled together, from each work's and each film's mean"}


def below(cells, key):
    v = [c[key] for c in cells.values() if c.get(key) is not None]
    return {"cells": len(v), "below_0.05": int(sum(x < 0.05 for x in v)), "expected_if_independent": round(0.05 * len(v), 2)}


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
            contrasts.append({"contrast": what, "a": a, "b": b, "E1_auc": paired_two_way(cells[a], cells[b], rng),
                              "E2_share": paired(cells[a]["_e2_by_adaptation"], cells[b]["_e2_by_adaptation"], rng),
                              "E3_rho_difference": None if None in (cells[a]["E3_rho"], cells[b]["E3_rho"]) else cells[a]["E3_rho"] - cells[b]["E3_rho"]})
    new = {k: c for k, c in cells.items() if "published" not in k}
    v2c = {k: c for k, c in new.items() if "v2 rule" in k}
    out = {"cells": {k: {kk: vv for kk, vv in c.items() if not kk.startswith("_")} for k, c in cells.items()}, "contrasts": contrasts,
           "counts_over_new_cells": {
               "E2_exact_p": below(new, "E2_exact_p"),
               "residual_decision_test": below(new, "residual_decision_exact_p"),
               "residual_registered_two_measure_form_which_can_pass_on_words_alone": below(new, "residual_exact_p"),
               "E3_p": below(new, "E3_p"),
               "E1_openings_against_their_own_label_shuffle_v2_rule_only": below(v2c, "E1_openings_shuffle_p")},
           "note": "The cells share the 32 works and are not independent: a count is not that many confirmations (cells_joint.json gives the probability of "
                   "each count under one null for all cells). E1 cells under v1's rule have no openings version and are not length-controlled. "
                   "residual_exact_p is v1's registered two-measure form; residual_decision_exact_p is v2's decision test (seven word measures)."}
    C.save(out, "contrasts.json")
    for k, c in out["cells"].items():
        dp = c["residual_decision_exact_p"]
        print(f"{k:42s} E1 {c['E1_auc']:.3f} | E2 {c['E2_share']:.3f} p {c['E2_exact_p']:.4f} | residual p {c['residual_exact_p']:.4f} "
              f"(decision {'-' if dp is None else format(dp, '.4f')}) | seq {c['sequels_top2']} | E3 {c['E3_rho']:.2f}")
    print(json.dumps(out["counts_over_new_cells"]))


if __name__ == "__main__":
    main()
