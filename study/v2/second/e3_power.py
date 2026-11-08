"""Holmes E3: power at n = 60 when the plan's covariates are regressed out first (v2's own simulation uses none).
Real Holmes page views, strata, year, sentences and number of screen works naming the story; centrality is a random normal.
No model output is read. Run from study/:  ../.venv/bin/python v2/second/e3_power.py  -> results/v2/second/holmes_e3_power.json
(PLAN.md, Section 8.)"""
import json
import sys
import numpy as np
from scipy.stats import norm, rankdata
sys.path.insert(0, "v2/second")
import holmes as H, lib
from lib import X
S = lib.install(H.holmes_spec())
rng = np.random.default_rng(99)
pop = S.popularity(S.early)
g = norm.ppf((rankdata(pop) - 0.5) / 60)
strata = [np.flatnonzero(S.strata == s) for s in sorted(set(S.strata))]
n_sim, n_perm = 2000, 1000
idx = np.tile(np.arange(60), (n_perm, 1))
for s in strata:
    idx[:, s] = s[np.argsort(rng.random((n_perm, len(s))), axis=1)]
rp = np.argsort(np.argsort(pop))
out = {"what": "Simulated power of the Holmes selection test (one-sided permutation within novels and short stories, alpha 0.05) with the real page views, strata and covariates",
       "n_stories": 60, "n_sim": n_sim, "n_perm": n_perm, "seed": 99, "power": {}}
for rho in (0.3, 0.5):
    r = 2 * np.sin(np.pi * rho / 6)
    hits = {"no covariates": 0, "year, sentences, centrality": 0, "and the number of screen works naming the story": 0}
    for _ in range(n_sim):
        off = r * g + np.sqrt(1 - r * r) * rng.standard_normal(60)
        base = np.c_[S.year[S.early], S.n_sent[S.early], rng.standard_normal(60)].astype(float)
        for name, cov in (("no covariates", None), ("year, sentences, centrality", base), ("and the number of screen works naming the story", np.c_[base, S.covariates["screen_works_naming_the_story"]])):
            x = off if cov is None else X.resid(off, cov)
            rx = np.argsort(np.argsort(x))
            obs = (rx * rp).sum()
            nul = (rx[None, :] * rp[idx]).sum(1)
            hits[name] += ((nul >= obs).sum() + 1) / (n_perm + 1) < 0.05
    out["power"][f"rho_{rho}"] = {k: v / n_sim for k, v in hits.items()}
    print(f"rho {rho}:", {k: round(v / n_sim, 3) for k, v in hits.items()})
out["spearman_of_page_views_with_the_number_of_screen_works_naming_the_story"] = float(X.spearman(pop, S.covariates["screen_works_naming_the_story"] + 1e-9 * np.arange(60)))
print("Spearman of page views with the number of screen works naming the story:", round(out["spearman_of_page_views_with_the_number_of_screen_works_naming_the_story"], 3))
json.dump(out, open("results/v2/second/holmes_e3_power.json", "w"), indent=1)
