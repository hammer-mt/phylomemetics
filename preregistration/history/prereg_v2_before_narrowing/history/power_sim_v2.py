import sys, json, numpy as np
sys.path.insert(0, "prereg")
import analysis as A
rng = np.random.default_rng(11)
R = 400
out = {}
# H1/H2: 34 memes, per-meme difference SD 0.15 (meme heterogeneity + pair noise), plus plot effects: use the real design
design = [{"plot": p["plot"], "version": v, "memes": p["memes"]} for p in json.load(open("preregistration/planted_design.json"))["plots"] for v in A.VERSIONS]
memes = sorted({m for d in design for m in d["memes"]}); n = len(design)
plots = sorted({d["plot"] for d in design})
def sim_t2(delta, null_value=0.0, tau_m=0.12, tau_p=0.5):
    # latent scores: method A = signal (delta translated to a score shift) + meme and plot effects; B = no signal
    # we simulate per-story scores so that twin AUC differences have realistic correlation across memes sharing plots
    pe = {p: rng.normal(0, tau_p) for p in plots}
    sa = np.zeros((n, len(memes))); sb = np.zeros((n, len(memes)))
    shift = {m: max(0.0, 2.5 * (delta + rng.normal(0, tau_m))) for m in memes}
    for i, d in enumerate(design):
        base = rng.normal(0, 1, len(memes)) + pe[d["plot"]]
        sa[i] = base + np.array([shift[m] if (m in d["memes"] and d["version"] in A.PLANTED) else 0 for m in memes])
        sb[i] = rng.normal(0, 1, len(memes)) + pe[d["plot"]] + np.array([2.5 * 0.0 if True else 0 for m in memes])
    a = A.detection_aucs(sa, design, memes, A.REWORDED); b = A.detection_aucs(sb, design, memes, A.REWORDED)
    pm = A.paired_meme_test(a, b, memes, "greater", null_value=null_value)
    pp = A.plot_flip_test(sa, sb, design, memes, A.REWORDED, null_value=null_value, n_mc=2000)
    return max(pm["p"], pp["p"]), pm["stat"]
for delta in (0.0, 0.05, 0.08, 0.10):
    res = [sim_t2(delta) for _ in range(R // 4)]
    ps = np.array([r[0] for r in res]); st = np.mean([r[1] for r in res])
    out[f"H1_true_auc_gap_{delta}"] = {"mean_auc_gap": float(st), "power_05": float(np.mean(ps < .05)), "power_01": float(np.mean(ps < .01))}
    print("H1", delta, out[f"H1_true_auc_gap_{delta}"], flush=True)
# H3: 11 units, specificity SD 0.15, permutation ~ sign-flip
for eff in (0.05, 0.10, 0.15):
    ps = [A.signflip(rng.normal(eff, 0.15, 11), alternative="greater")["p"] for _ in range(R)]
    out[f"H3_spec_{eff}"] = {"power_05": float(np.mean(np.array(ps) < .05)), "power_01": float(np.mean(np.array(ps) < .01))}
# H4: 21 clusters (8 Batman source clusters + 13 plays), per-cluster SD 0.25, IUT over 2 rivals with correlated d (r = 0.7)
for eff in (0.05, 0.08, 0.10, 0.15):
    ps = []
    for _ in range(R):
        z1 = rng.normal(0, 1, 21); z2 = 0.7 * z1 + np.sqrt(1 - 0.49) * rng.normal(0, 1, 21)
        p1 = A.signflip(eff + 0.25 * z1, alternative="greater", n_mc=4000)["p"]; p2 = A.signflip(eff + 0.25 * z2, alternative="greater", n_mc=4000)["p"]
        ps.append(max(p1, p2))
    out[f"H4_adv_{eff}"] = {"power_05": float(np.mean(np.array(ps) < .05)), "power_01": float(np.mean(np.array(ps) < .01))}
    print("H4", eff, out[f"H4_adv_{eff}"], flush=True)
# H5: 13 plays with the actual H5 counts; per-film SD 0.22, between-play SD 0.10 -> per-play SD sqrt(0.01 + 0.0484/n)
counts = list(json.load(open("preregistration/works_shakespeare.json"))["counts"]["h5_by_play"].values())
sd = np.sqrt(0.10 ** 2 + 0.22 ** 2 / np.array(counts))
for eff in (0.05, 0.08, 0.12, 0.15):
    ps = []
    for _ in range(R):
        z1 = rng.normal(0, 1, len(counts)); z2 = 0.7 * z1 + np.sqrt(1 - 0.49) * rng.normal(0, 1, len(counts))
        ps.append(max(A.signflip(eff + sd * z1, alternative="greater")["p"], A.signflip(eff + sd * z2, alternative="greater")["p"]))
    out[f"H5_adv_{eff}"] = {"power_05": float(np.mean(np.array(ps) < .05)), "power_01": float(np.mean(np.array(ps) < .01))}
out["H5_per_play_sd"] = dict(zip(map(str, counts), sd.round(3).tolist()))
json.dump(out, open("power.json", "w"), indent=1)
print(json.dumps(out, indent=1))
