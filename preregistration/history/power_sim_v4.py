"""Power for prereg-v3 after the fifth review (2026-10-21): H1 and H2 with their actual intersection-union tests.

power_sim_v3.py compared the neurons with ONE rival that had no signal, so its H1/H2 figures are upper bounds: H1 takes
the larger p over 2 rivals and H2 over 3, and H2 is a non-inferiority test at a 0.05 margin. This simulation runs the
real statistics (meme-level and plot-level sign flips, p = the larger, then the intersection-union p over the rivals)
with rivals that have signal of their own and are correlated with the neurons:

  story noise      each method's story-by-meme score = sqrt(rho) x shared noise + sqrt(1 - rho) x own noise, rho = 0.5
  planted signal   on planted cells, shift = sqrt(2) x PhiInv(AUC_km) + v_plot (v_plot ~ N(0, 0.3), shared by every method
                   and meme of the plot: an easy or a hard plot); AUC_km = the method's mean AUC + c_m (shared meme
                   difficulty, SD 0.08) + h_km (method-specific, SD 0.06), clipped to [0.5, 0.99]
  twin AUC         as analysis.detection_aucs (each planted story against its own plot's control)

It also gives S-coded's power at the thresholds of the prominent Benjamini-Hochberg family of 30 tests (0.05 x R / 30
for R discoveries: 0.05 if all 30 are discoveries, 0.0167 with 10, 0.00167 with one), and the overlap test at the same
thresholds. The fast statistics below are checked against analysis.paired_meme_test and analysis.plot_flip_test on the
first replicate.

  .venv/bin/python prereg/history/power_sim_v4.py      # writes prereg/history/power_v4.json (about 10-20 minutes)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analysis as A  # noqa: E402

R = 500
N_MC_MEME, N_MC_PLOT = 4000, 2000
RHO, TAU_M, TAU_K, TAU_P = 0.5, 0.08, 0.06, 0.3
HOLM = {"holm_first_0.0167": 0.05 / 3, "holm_second_0.025": 0.05 / 2, "alpha_0.05": 0.05}
BH30 = {"bh_all_30_discoveries_0.05": 0.05, "bh_10_of_30_0.0167": 0.05 * 10 / 30, "bh_1_of_30_0.00167": 0.05 / 30}
rng = np.random.default_rng(14)
design = [{"plot": p["plot"], "version": v, "memes": p["memes"]} for p in json.load(open(HERE.parent / "planted_design.json"))["plots"]
          for v in A.VERSIONS]
memes = sorted({m for d in design for m in d["memes"]})
n, M = len(design), len(memes)
plots = sorted({d["plot"] for d in design})
pid = {p: k for k, p in enumerate(plots)}
plot_of = np.array([pid[d["plot"]] for d in design])
planted = np.array([[m in d["memes"] and d["version"] in A.PLANTED for m in memes] for d in design])


def pairs_for(condition):
    out = []
    for j, m in enumerate(memes):
        pr = A.twin_pairs(design, m, condition)
        out.append((np.array([i for i, _, _ in pr]), np.array([c for _, c, _ in pr]), np.array([pid[p] for _, _, p in pr])))
    return out


PAIRS = {"reworded": pairs_for(A.REWORDED), "coded": pairs_for({"coded"})}


def draw(aucs, z):
    """Scores of methods with mean twin AUCs `aucs` (one per method) sharing the story noise z."""
    c = rng.normal(0, TAU_M, M)
    v = rng.normal(0, TAU_P, len(plots))[plot_of][:, None]
    out = []
    for a in aucs:
        auc_m = np.clip(a + c + rng.normal(0, TAU_K, M), 0.5, 0.99)
        shift = np.sqrt(2) * norm.ppf(auc_m)
        out.append(np.sqrt(RHO) * z + np.sqrt(1 - RHO) * rng.normal(0, 1, (n, M)) + planted * (shift[None, :] + v))
    return out


def compare(Sa, Sb, cond, null_value=0.0):
    """The T2 comparison (analysis._t2_compare): meme-level and plot-level sign flips, p = the larger."""
    d, W = np.zeros(M), np.zeros((M, len(plots)))
    for j, (pi, ci, pp) in enumerate(PAIRS[cond]):
        ia = (Sa[pi, j] > Sa[ci, j]).astype(float)
        ib = (Sb[pi, j] > Sb[ci, j]).astype(float)
        d[j] = ia.mean() - ib.mean()
        np.add.at(W[j], pp, (ia - ib - null_value) / len(pi))
    dd = d - null_value
    signs = rng.choice([-1.0, 1.0], size=(N_MC_MEME, M))
    p_meme = float((np.sum((signs * dd).mean(1) >= dd.mean() - 1e-12) + 1) / (N_MC_MEME + 1))
    w = W.sum(0) / M
    sp = rng.choice([-1.0, 1.0], size=(N_MC_PLOT, len(plots)))
    p_plot = float((np.sum((sp * w).sum(1) >= w.sum() - 1e-12) + 1) / (N_MC_PLOT + 1))
    return max(p_meme, p_plot), float(d.mean()), d, w.sum() + null_value


def check_against_analysis():
    z = rng.normal(0, 1, (n, M))
    Sa, Sb = draw([0.72, 0.65], z)
    p, stat, d, wstat = compare(Sa, Sb, "reworded")
    a = A.detection_aucs(Sa, design, memes, A.REWORDED)
    b = A.detection_aucs(Sb, design, memes, A.REWORDED)
    pm = A.paired_meme_test(a, b, memes, "greater")
    pf = A.plot_flip_test(Sa, Sb, design, memes, A.REWORDED, n_mc=2000)
    assert abs(pm["stat"] - stat) < 1e-9 and abs(pf["stat"] - wstat) < 1e-9, (pm["stat"], stat, pf["stat"], wstat)
    return {"meme_stat": stat, "plot_stat": wstat, "p_fast": p, "p_analysis": max(pm["p"], pf["p"])}


def power(ps, th):
    ps = np.asarray(ps)
    return {k: float(np.mean(ps <= v)) for k, v in th.items()}


out = {"model": __doc__.split("\n\n")[1], "replicates": R, "rho": RHO, "tau_meme": TAU_M, "tau_method": TAU_K, "tau_plot": TAU_P,
       "check_against_analysis": check_against_analysis()}
print("check", out["check_against_analysis"], flush=True)

# H1: neurons against the evasion-aware query and the lexical probe (intersection-union over 2 rivals)
for rivals in ((0.65, 0.65), (0.60, 0.68)):
    for gap in (0.05, 0.075, 0.10, 0.125):
        nr = max(rivals) + gap
        piut, psingle, gaps = [], [], []
        for _ in range(R):
            z = rng.normal(0, 1, (n, M))
            Sn, *Sr = draw([nr] + list(rivals), z)
            res = [compare(Sn, s, "reworded") for s in Sr]
            piut.append(max(r[0] for r in res))
            psingle.append(res[int(np.argmax(rivals))][0])
            gaps.append(min(r[1] for r in res))
        key = f"H1_rivals{rivals[0]:.2f}_{rivals[1]:.2f}_gap{gap}"
        out[key] = {"neurons_auc": nr, "mean_min_gap": float(np.mean(gaps)), "iut_2_rivals": power(piut, HOLM),
                    "single_strongest_rival": power(psingle, HOLM)}
        print(key, out[key], flush=True)

# H2: non-inferiority (margin 0.05) against 3 strong embedding rivals (intersection-union over 3)
for gap in (-0.025, 0.0, 0.025, 0.05):
    rivals = (0.75, 0.75, 0.75)
    nr = 0.75 + gap
    piut, psingle, gaps = [], [], []
    for _ in range(R):
        z = rng.normal(0, 1, (n, M))
        Sn, *Sr = draw([nr] + list(rivals), z)
        res = [compare(Sn, s, "reworded", null_value=-A.NI_MARGIN) for s in Sr]
        piut.append(max(r[0] for r in res))
        psingle.append(res[0][0])
        gaps.append(float(np.mean([r[1] for r in res])))
    key = f"H2_noninferiority_rivals0.75_gap{gap}"
    out[key] = {"neurons_auc": nr, "mean_gap": float(np.mean(gaps)), "iut_3_rivals": power(piut, HOLM), "single_rival": power(psingle, HOLM)}
    print(key, out[key], flush=True)

# S-coded: one context-using method against one context-free method with signal, 12 coded pairs per meme
for gap in (0.05, 0.075, 0.10, 0.125):
    ps, gaps = [], []
    for _ in range(R):
        z = rng.normal(0, 1, (n, M))
        Sa, Sb = draw([0.60 + gap, 0.60], z)
        r = compare(Sa, Sb, "coded")
        ps.append(r[0])
        gaps.append(r[1])
    key = f"S_coded_rival0.60_gap{gap}"
    out[key] = {"mean_gap": float(np.mean(gaps)), **power(ps, BH30)}
    print(key, out[key], flush=True)

# overlap (as power_sim_v3.py), now also at the prominent family's BH thresholds
cells = A._overlap_cells(design, memes, A.REWORDED)
eng = A._overlap_engine(cells, M, 10, 4000, "power4")
ok = np.ones(len(cells["rows"]), bool)
lg = lambda p: np.log(p / (1 - p))
for k in (2, 4, 6):
    pa = []
    for _ in range(200):
        ra, rb = np.full(M, 0.6), np.full(M, 0.6)
        ra[:k], rb[:k] = 0.75, 0.15
        u = rng.normal(0, 0.5, cells["n_plots"])[cells["plot"]]
        da = rng.random(len(ok)) < 1 / (1 + np.exp(-(lg(ra[cells["meme"]]) + u)))
        db = rng.random(len(ok)) < 1 / (1 + np.exp(-(lg(rb[cells["meme"]]) + u)))
        pa.append(A.overlap_pair(da, db, ok, cells, eng, memes)["p_a_finds_b_misses"])
    out[f"overlap_k{k}_memes_one_way"] = power(pa, BH30)
    print("overlap", k, out[f"overlap_k{k}_memes_one_way"], flush=True)

json.dump(out, open(HERE / "power_v4.json", "w"), indent=1)
print(json.dumps(out, indent=1))
