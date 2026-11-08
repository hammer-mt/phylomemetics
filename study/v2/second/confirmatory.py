"""The confirmatory tests of the two new corpora, side by side (v2/second/PLAN.md, Section 13).

    ../.venv/bin/python v2/second/confirmatory.py        -> results/v2/second/confirmatory.json

All for Qwen3-32B at the primary setting. Holmes: E1 (met or not; no p), E2, E3. Remakes: heredity, the decision test.
Each p is read at 0.05; Holm-adjusted p values are given within each corpus and over the four tests that have a p.
A test that is not evaluable or not calibrated contributes no p. The Holmes residual test is reported and is not in the
family. A result below 0.05 whose Holm p within its corpus is 0.05 or more is called nominal.
"""
import json
from pathlib import Path

OUT = Path("results/v2/second")
PRIMARY = "qwen3-32b"


def holm(ps):
    """Holm-adjusted p for each key of {name: p}; None stays None."""
    have = sorted((p, k) for k, p in ps.items() if p is not None)
    out, running = {k: None for k in ps}, 0.0
    for i, (p, k) in enumerate(have):
        running = max(running, min(1.0, (len(have) - i) * p))
        out[k] = running
    return out


def label(p, adjusted):
    if p is None:
        return "no p"
    if p >= 0.05:
        return "not below 0.05"
    return "below 0.05" if adjusted < 0.05 else "nominal: below 0.05, not after Holm's adjustment"


def main():
    H = json.load(open(OUT / f"holmes_{PRIMARY}.json"))
    R = json.load(open(OUT / f"remakes_{PRIMARY}.json"))
    c = H["confirmatory"]
    dec = R["decision"]
    not_run = dec["outcome"] in ("not evaluable", "not calibrated")
    ps = {"holmes_E2": c["E2_p"], "holmes_E3": c["E3_p"], "remakes_heredity": R["heredity"]["not_copied"]["p"],
          "remakes_decision": None if not_run else dec["slope"]["p"]}
    within = {**holm({k: v for k, v in ps.items() if k.startswith("holmes")}), **holm({k: v for k, v in ps.items() if k.startswith("remakes")})}
    overall = holm(ps)
    out = {
        "what": "The confirmatory tests of v2/second/PLAN.md Section 13, Qwen3-32B, primary setting. Made by v2/second/confirmatory.py.",
        "model": PRIMARY,
        "holmes": {
            "E1": {"verdict": c["E1_verdict"], "openings_auc": c.get("E1_openings_auc"), "openings_auc_below_0.70": c.get("E1_openings_auc_below_0.70"),
                   "not_confined_to_works_the_model_names": c.get("E1_not_confined_to_works_the_model_names")},
            "E2": {"state": c["E2"], "statistic": c["E2_statistic"], "p": ps["holmes_E2"]},
            "E3": {"p": ps["holmes_E3"]},
            "residual_test_reported_not_confirmatory": {"p_larger_of_the_two_forms": c.get("residual_p_larger_of_the_two_forms")},
        },
        "remakes": {
            "heredity": {"p": ps["remakes_heredity"], "S": R["heredity"]["not_copied"]["S"], "positive_control": R["heredity"]["positive_control"]},
            "decision": {"outcome": dec["outcome"], "p": dec["slope"]["p"], "Z": dec["slope"]["Z"], "conditions": dec["conditions"],
                         "p_enters_the_family": not not_run},
        },
        "p": ps, "holm_within_corpus": within, "holm_over_all_four": overall,
        "reading": {k: label(ps[k], within[k]) for k in ps},
        "sources": {f"holmes_{PRIMARY}.json": H.get("code", {}), f"remakes_{PRIMARY}.json": {"git_head": R.get("git_head")}},
    }
    json.dump(out, open(OUT / "confirmatory.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("p", "holm_within_corpus", "holm_over_all_four", "reading")}, indent=1))
    print("E1:", out["holmes"]["E1"]["verdict"], "| remakes decision:", dec["outcome"])


if __name__ == "__main__":
    main()
