"""The six confirmatory tests in one place, with Holm-adjusted p values (PLAN.md, Section 15 and Section 18, change 20).

All six are for the primary model under the v2 rule at the primary setting:

  E1        met, not met or falsified (no p value; it is a set of criteria)
  E2        exact joint-relabelling p
  residual  the decision test; the p that enters Holm is the larger of the decision-test p and the every-measure p,
            and a pass also needs at least half of the 8 uncapped sensitivity settings
  E3        one-sided p
  ablation  the largest of its three p values (two controls and specificity)
  steering  exact binomial p on the idea units

A result below 0.05 whose Holm-adjusted p is 0.05 or more is called nominal.

    cd study && ../.venv/bin/python v2/analysis/confirmatory.py [model-tag]

Reads results/v2/experiments_v2.json, sensitivity_<tag>.json, ablation_<tag>.json and steering_<tag>.json (each part is
reported as not yet run if its file is missing). Writes results/v2/confirmatory.json (or confirmatory_<tag>.json for a
model other than Qwen3-32B, where it is a replication summary and not the confirmatory family).
"""
import json
import sys

import common as C

PRIMARY = "qwen3-32b"


def load(name):
    p = C.RESULTS / name
    return json.load(open(p)) if p.exists() else None


def main(tag=PRIMARY):
    ex = load("experiments_v2.json")[tag]
    sens, abl, steer = load(f"sensitivity_{tag}.json"), load(f"ablation_{tag}.json"), load(f"steering_{tag}.json")
    e1, e2, e3 = ex["E1"], ex["E2"], ex["E3"]["primary"]
    rs = e2["residual_summary"]
    tests = {}
    a, o = e1["canon_share"]["A_name_free"], e1["canon_share"].get("openings", {})
    tests["E1_identity"] = {"verdict": e1["canon_share"]["verdict"], "auc": a["hero_film_cousins"]["auc"], "ci95": a["hero_film_cousins"]["ci95"],
                            "openings_auc": o.get("hero_film_cousins", {}).get("auc"), "clauses": e1["canon_share"]["clauses"],
                            "centroid_of_units_verdict": e1["unit_centroid"]["verdict"], "p": None, "passes": e1["canon_share"]["verdict"] == "met"}
    tests["E2_heredity"] = {"share_of_comics_beating_the_source": e2["summary"]["share"], "p": e2["summary"]["exact_p"], "evaluable": e2["evaluable"]["evaluable"],
                            "sequel_control_passes": e2["sequel_control_passes"], "passes": bool(e2["evaluable"]["evaluable"] and e2["summary"]["exact_p"] < 0.05)}
    p_dec, p_all = rs["decision_words_by_frequency"]["exact_p"], rs["every_word_and_code_measure"]["exact_p"]
    half = None if sens is None else bool(sens["residual_passes_in_at_least_half"])
    tests["residual_beyond_words"] = {"decision_test_p": p_dec, "every_measure_p": p_all, "p": max(p_dec, p_all),
                                      "registered_two_measure_form_p": rs["registered_both_word_measures"]["exact_p"],
                                      "uncapped_settings_below_0.05": None if sens is None else sens["residual_settings_exact_p_below_0.05"],
                                      "uncapped_settings": None if sens is None else sens["n_decision_settings"],
                                      "passes_in_at_least_half_of_the_uncapped_settings": half,
                                      "passes": None if half is None else bool(p_dec < 0.05 and p_all < 0.05 and half)}
    tests["E3_selection"] = {"rho": e3["rho"], "p": e3["p_one_sided"], "evaluable": e3.get("evaluable", True),
                             "passes": bool(e3.get("evaluable", True) and e3["rho"] > 0 and e3["p_one_sided"] < 0.05)}
    if abl and "criterion" in abl:
        c = abl["criterion"]
        ps = [c["p_against_sets_matched_on_works"], c["p_against_sets_also_matched_on_activity"], c["specificity_p"]]
        tests["ablation"] = {"p_against_sets_matched_on_works": ps[0], "p_against_sets_also_matched_on_activity": ps[1], "specificity_p": ps[2],
                             "p": None if None in ps else max(ps), "evaluable": c["evaluable_with_this_question"], "passes": c["met"]}
    else:
        tests["ablation"] = {"p": None, "passes": None, "status": "not yet run"}
    if steer:
        i = steer["idea_units"]
        tests["steering_idea_units"] = {"accuracy": i["accuracy"], "n": i["n"], "correct": i["correct"], "chance": steer["chance"], "multiple": steer["multiple"],
                                        "p": i["p_above_chance"], "passes": steer["criterion_met"]}
    else:
        tests["steering_idea_units"] = {"p": None, "passes": None, "status": "not yet run"}
    # Holm over the tests that have a p (E1 is a set of criteria and has none)
    with_p = sorted([(t["p"], k) for k, t in tests.items() if t.get("p") is not None])
    m, running = len(with_p), 0.0
    for rank, (p, k) in enumerate(with_p):
        running = max(running, min(1.0, (m - rank) * p))
        tests[k]["holm_adjusted_p"] = running
        tests[k]["nominal_only"] = bool(tests[k]["passes"] and running >= 0.05)
    out = {"model": tag, "is_the_confirmatory_family": tag == PRIMARY, "tests_with_a_p_value": m, "tests": tests,
           "passes": [k for k, t in tests.items() if t["passes"]], "not_yet_run": [k for k, t in tests.items() if t.get("status") == "not yet run"]}
    C.save(out, "confirmatory.json" if tag == PRIMARY else f"confirmatory_{tag}.json")
    for k, t in tests.items():
        print(f"{k:24s} p {t.get('p')}  Holm {t.get('holm_adjusted_p')}  passes {t['passes']}" + (f"  ({t['verdict']})" if "verdict" in t else ""))


if __name__ == "__main__":
    main(*sys.argv[1:2])
