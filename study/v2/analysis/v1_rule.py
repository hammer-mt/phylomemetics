"""v1's presence rule on the new models, to tell the effect of the model from the effect of the rule (PLAN.md, Section 7).

Runs v1's experiments.py functions, unchanged, on each work's mean of its three strongest sentences of the raw
SwiGLU product (all sign regimes), with the presence bar set by the 69 ordinary films: exactly what v1 did, at
full precision and without v1's int8 rounding.

    cd study && ../.venv/bin/python v2/analysis/v1_rule.py [model-tag ...]

Needs v2/acts/<tag>/{orig,anon,anon_icon}_raw_top3.npy. Writes results/v2/experiments_v1rule.json.
"""
import json
import sys
import time

import numpy as np

import common as C

import experiments as X
import experiments_v2 as E


def run_model(tag):
    t0 = time.time()
    d = C.ACTS / tag
    S_o = np.load(d / "orig_raw_top3.npy").astype(np.float32)
    S_a = np.load(d / "anon_raw_top3.npy").astype(np.float32)
    S_ab = np.load(d / "anon_icon_raw_top3.npy").astype(np.float32)
    X.RNG = np.random.default_rng(7)                 # the state v1's first model started from, so the film splits are v1's
    splits = X.film_splits()
    res = {"E1": X.e1(S_o, S_a, S_ab, splits)}
    del S_ab
    P_a, P_o = S_a > X.bar(S_a, X.FILMS), S_o > X.bar(S_o, X.FILMS)
    res["n_units"] = int(S_a.shape[1])
    res["units_present_per_batman_work_name_free_median"] = float(np.median(P_a[X.BAT].sum(1)))
    del S_o, S_a
    res["E2"] = {"primary": X.e2(P_a), "original_upper_bound": X.e2(P_o, n_perm=2000),
                 "sequels_name_free": X.sequel_control(P_a), "sequels_original": X.sequel_control(P_o)}
    nb = P_a[X.BAT].sum(0)
    res["E2"]["n_optional_units"] = int(((nb >= 2) & (nb <= 16)).sum())
    sim, Pw = X.lexical_reps()
    R_tf, R_wd, R_n = sim[:, X.COMICS], X.unit_rates(Pw), X.unit_rates(P_a)
    res["E2"]["residual_test_beyond_words"] = {
        "neurons_resid_on_tfidf_cosine": X.residual_test(R_n, [R_tf]), "neurons_resid_on_word_presence": X.residual_test(R_n, [R_wd]),
        "neurons_resid_on_both": X.residual_test(R_n, [R_tf, R_wd]),
        "decision_words_by_frequency": E.residual(R_n, E.regressors("decision"))}     # v2's decision test (PLAN.md Section 18, changes 6 and 20)
    pv = json.load(open("data/pageviews.json"))
    res["E3"] = {"primary": X.e3(P_a, pv), "original_text": X.e3(P_o, pv, n_perm=2000)}
    res["seconds"] = round(time.time() - t0)
    e2p = res["E2"]["primary"]
    print(f"== {tag} under v1's rule: E1 AUC {res['E1']['A_name_free']['hero_film_cousins']['auc']:.3f} | E2 share {e2p['mean_share_beating_source']:.3f} "
          f"exact p {e2p['chance']['joint_relabelling_exact']['p']:.5f} | residual p (both) {res['E2']['residual_test_beyond_words']['neurons_resid_on_both']['chance']['joint_relabelling_exact']['p']:.4f} "
          f"| residual p (decision) {res['E2']['residual_test_beyond_words']['decision_words_by_frequency']['chance']['joint_relabelling_exact']['p']:.4f} "
          f"| E3 rho {res['E3']['primary']['rho']:.3f}", flush=True)
    return res


def main():
    tags = sys.argv[1:] or [t for t in C.MODELS if (C.ACTS / t / "orig_raw_top3.npy").exists()]
    path = C.RESULTS / "experiments_v1rule.json"
    out = json.load(open(path)) if path.exists() else {}
    out["rule"] = ("v1: a unit is present in a work when the mean of its three strongest sentences (raw SwiGLU product, all sign regimes) "
                   "beats all but one of the 69 ordinary films; computed by v1's experiments.py on full-precision activations")
    for tag in tags:
        out[tag] = run_model(tag)
        C.save(out, "experiments_v1rule.json")


if __name__ == "__main__":
    main()
