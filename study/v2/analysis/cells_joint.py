"""How surprising is a count of model-by-rule cells below 0.05? (PLAN.md, Section 18, change 20.)

Every cell (a model under v1's rule or v2's) analyses the same 32 works, so the cells are not independent trials.
E2's exact null relabels the comics in all 3,991,680 ways. Here each relabelling is applied to all cells at once:
for every relabelling, each cell's statistic is turned into the exact p it would get, and the number of cells below
0.05 is counted. That gives the exact probability, under one null for all cells, of at least k cells below 0.05,
for E2 itself and for the residual decision test, beside what independence would give.

    cd study && ../.venv/bin/python v2/analysis/cells_joint.py [model-tag ...]

Needs each model's presence data (v2/acts/<tag>/) and, for v1's rule, anon_raw_top3.npy. Writes results/v2/cells_joint.json.
"""
import itertools
import math
import sys

import numpy as np

import common as C

import experiments as X
import experiments_v2 as E

BAT, COMICS, L, year = X.BAT, X.COMICS, X.L, X.year


def null_and_observed(R):
    """v1's E2 statistic (hub adjustment, time rule) under every relabelling of the comics, and its observed value."""
    Z = {}
    for a in BAT:
        for j, c in enumerate(COMICS):
            vals = np.array([R[t, j] for t in X.SCREEN_NONADAPT if t != a])
            Z[(a, c)] = (R[a, j] - vals.mean()) / (vals.std() + 1e-9)

    def stat_for(a, src):
        cands = sorted(set(c for c in COMICS if year[c] <= year[a]) | set(src))
        best = max(Z[(a, s)] for s in src)
        others = [c for c in cands if c not in src]
        return float(np.mean([Z[(a, c)] > best for c in others])) if others else 0.0

    used = sorted({L[x] for srcs in X.ADAPTATIONS.values() for x in srcs})
    col = {s: i for i, s in enumerate(used)}
    M = X.source_maps(len(used))
    tot = np.zeros(len(M))
    for t, srcs in X.ADAPTATIONS.items():
        a, s = L[t], [L[x] for x in srcs]
        T = np.full((len(COMICS),) * len(s), np.nan)
        for cmb in itertools.permutations(range(len(COMICS)), len(s)):
            T[cmb] = stat_for(a, list(COMICS[list(cmb)]))
        tot += T[tuple(M[:, col[x]] for x in s)]
    tot /= len(X.ADAPTATIONS)
    obs = float(np.mean([stat_for(L[t], [L[x] for x in srcs]) for t, srcs in X.ADAPTATIONS.items()]))
    return tot.astype(np.float32), obs


def residualised(R, lex):
    mask = np.zeros(R.shape, bool)
    mask[BAT] = True
    mask[COMICS, np.arange(len(COMICS))] = False
    A = np.column_stack([np.ones(mask.sum())] + [x[mask] for x in lex])
    beta = np.linalg.lstsq(A, R[mask], rcond=None)[0]
    return R - (np.column_stack([np.ones(R.size)] + [x.ravel() for x in lex]) @ beta).reshape(R.shape)


def main():
    tags = sys.argv[1:] or [t for t in C.MODELS if (C.ACTS / t / "anon_main_top1.u8.npy").exists()]
    rates = {}
    for tag in tags:
        st = C.Setting(tag, icon=False)
        rates[f"{tag} | full precision, v2 rule"] = X.unit_rates(st.P["anon"])
        del st
        f = C.ACTS / tag / "anon_raw_top3.npy"
        if f.exists():
            S = np.load(f).astype(np.float32)
            rates[f"{tag} | full precision, v1 rule"] = X.unit_rates(S > X.bar(S, X.FILMS))
            del S
        print("rates for", tag, flush=True)
    out = {"cells": list(rates), "n_relabellings": None, "tests": {}}
    K = len(rates)
    for name, lex in (("E2", None), ("residual_decision_test", E.regressors("decision")), ("residual_registered_two_measure_form", E.regressors("registered"))):
        B, pobs = [], {}
        for k, R in rates.items():
            tot, obs = null_and_observed(R if lex is None else residualised(R, lex))
            srt = np.sort(tot)
            B.append(np.searchsorted(srt, tot + 1e-6, side="right") / len(tot) < 0.05)       # would this relabelling give this cell p < 0.05?
            pobs[k] = float(np.mean(tot <= obs + 1e-6))
        B = np.array(B)
        cnt = B.sum(0)
        out["n_relabellings"] = int(B.shape[1])
        seen = int(sum(p < 0.05 for p in pobs.values()))
        out["tests"][name] = {
            "exact_p_by_cell": pobs, "cells_below_0.05": seen,
            "probability_of_at_least_that_many_under_one_null_for_all_cells": float((cnt >= seen).mean()) if seen else 1.0,
            "probability_of_at_least_k_cells": {str(k): float((cnt >= k).mean()) for k in range(1, K + 1)},
            "the_same_if_the_cells_were_independent": {str(k): float(sum(math.comb(K, j) * 0.05 ** j * 0.95 ** (K - j) for j in range(k, K + 1))) for k in range(1, K + 1)},
            "mean_pairwise_correlation_of_cells_under_the_null": float(np.corrcoef(B.astype(float))[np.triu_indices(K, 1)].mean()) if K > 1 else None}
        r = out["tests"][name]
        print(f"{name}: {seen} of {K} cells below 0.05; under one null for all cells that many or more has probability "
              f"{r['probability_of_at_least_that_many_under_one_null_for_all_cells']:.4f}", {k: round(v, 4) for k, v in pobs.items()}, flush=True)
    C.save(out, "cells_joint.json")


if __name__ == "__main__":
    main()
