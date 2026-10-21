"""Power for prereg-v3 (the narrowed design: H1-H3 under Holm over three, S-coded and the overlap test).

Same simulation model as power_sim_v2.py for H1/H2 and H3, now read at Holm's three thresholds for three claims
(0.05/3 for the smallest p, 0.05/2, 0.05). Adds S-coded (12 coded twin pairs per meme instead of 24 reworded pairs)
and the overlap analysis's meme-level test (overlap_pair with the plot-swap null).

  .venv/bin/python prereg/history/power_sim_v3.py      # writes prereg/history/power_v3.json (about 10-20 minutes)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analysis as A  # noqa: E402

rng = np.random.default_rng(13)
R = 200
TH = {"holm_first_0.0167": 0.05 / 3, "holm_second_0.025": 0.05 / 2, "alpha_0.05": 0.05}
design = [{"plot": p["plot"], "version": v, "memes": p["memes"]} for p in json.load(open(HERE.parent / "planted_design.json"))["plots"]
          for v in A.VERSIONS]
memes = sorted({m for d in design for m in d["memes"]})
n = len(design)
plots = sorted({d["plot"] for d in design})
out = {"model": "as power_sim_v2.py: per-story scores with plot random effects (SD 0.5), meme heterogeneity SD 0.12 in the "
                "latent shift; method B has no signal. Power = share of replicates with p below each threshold.",
       "replicates": R, "thresholds": TH}


def sim_t2(delta, condition, tau_m=0.12, tau_p=0.5):
    pe = {p: rng.normal(0, tau_p) for p in plots}
    shift = {m: max(0.0, 2.5 * (delta + rng.normal(0, tau_m))) for m in memes}
    sa = np.zeros((n, len(memes)))
    sb = np.zeros((n, len(memes)))
    for i, d in enumerate(design):
        base = rng.normal(0, 1, len(memes)) + pe[d["plot"]]
        sa[i] = base + np.array([shift[m] if (m in d["memes"] and d["version"] in A.PLANTED) else 0 for m in memes])
        sb[i] = rng.normal(0, 1, len(memes)) + pe[d["plot"]]
    a = A.detection_aucs(sa, design, memes, condition)
    b = A.detection_aucs(sb, design, memes, condition)
    pm = A.paired_meme_test(a, b, memes, "greater")
    pp = A.plot_flip_test(sa, sb, design, memes, condition, n_mc=2000)
    return max(pm["p"], pp["p"]), pm["stat"]


for label, condition in (("H1_H2_reworded", A.REWORDED), ("S_coded", {"coded"})):
    for delta in (0.05, 0.08, 0.10):
        res = [sim_t2(delta, condition) for _ in range(R)]
        ps = np.array([r[0] for r in res])
        out[f"{label}_delta{delta}"] = {"mean_auc_gap": float(np.mean([r[1] for r in res])),
                                        **{k: float(np.mean(ps < v)) for k, v in TH.items()}}
        print(label, delta, out[f"{label}_delta{delta}"], flush=True)

for eff in (0.05, 0.10, 0.15):
    ps = np.array([A.signflip(rng.normal(eff, 0.15, 11), alternative="greater")["p"] for _ in range(4 * R)])
    out[f"H3_spec_{eff}"] = {k: float(np.mean(ps < v)) for k, v in TH.items()}
    print("H3", eff, out[f"H3_spec_{eff}"], flush=True)

# overlap: two methods equally good on most memes (detection rate 0.6 at the operating point); on k memes A detects
# 0.75 and B 0.15 (and, for 'both directions', the reverse on k other memes). Shared plot effects (logit SD 0.5).
cells = A._overlap_cells(design, memes, A.REWORDED)
eng = A._overlap_engine(cells, len(memes), 10, 2000, "power")
ok = np.ones(len(cells["rows"]), bool)
lg = lambda p: np.log(p / (1 - p))
for k in (2, 4, 6, 8):
    pa, pb = [], []
    for _ in range(R):
        ra = np.full(len(memes), 0.6)
        rb = np.full(len(memes), 0.6)
        ra[:k], rb[:k] = 0.75, 0.15
        ra[k:2 * k], rb[k:2 * k] = 0.15, 0.75
        u = rng.normal(0, 0.5, cells["n_plots"])[cells["plot"]]
        da = rng.random(len(ok)) < 1 / (1 + np.exp(-(lg(ra[cells["meme"]]) + u)))
        db = rng.random(len(ok)) < 1 / (1 + np.exp(-(lg(rb[cells["meme"]]) + u)))
        r = A.overlap_pair(da, db, ok, cells, eng, memes)
        pa.append(r["p_a_finds_b_misses"])
        pb.append(r["p_b_finds_a_misses"])
    pa, pb = np.array(pa), np.array(pb)
    out[f"overlap_k{k}_memes_each_way"] = {"one_direction_power_0.05": float(np.mean(pa < 0.05)),
                                           "both_directions_power_0.05": float(np.mean((pa < 0.05) & (pb < 0.05)))}
    print("overlap", k, out[f"overlap_k{k}_memes_each_way"], flush=True)

json.dump(out, open(HERE / "power_v3.json", "w"), indent=1)
print(json.dumps(out, indent=1))
