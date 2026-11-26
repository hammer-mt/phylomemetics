"""python summarize.py   -> results/scale/summary.json and a printed table. Applies the README's decision rule."""
import json
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent.parent / "results/scale"
rng = np.random.default_rng(0)
tf = json.load(open(R / "baseline_tfidf.json"))
MEMES = list(tf["logreg"])
HARD = {m: np.argsort(tf["pair_margin"][m])[:4] for m in MEMES}
M = {}                                                              # method -> [memes, 8]
for f in sorted(R.glob("*_features.json")):
    d = json.load(open(f))
    for kind, b in d["B"].items():
        for k, v in b.items():
            if k.startswith(("m1", "m5", "proto", "probe")) and "mismatch" not in k:
                M[f"{d['tag']} {kind} {k}"] = np.array([v[m] for m in MEMES])
for f in sorted(R.glob("baseline_*.json")):
    for k, v in json.load(open(f)).items():
        if "mismatch" not in k and k != "pair_margin":
            M[f"{f.stem[9:]} {k}"] = np.array([v[m] for m in MEMES])
for tag in ("qwen3-8b", "qwen3-32b"):
    sub = [k for k in M if k.startswith(tag) and "subset" in k]
    if sub:
        M[f"{tag} neuron m1 8B-size subsets (mean of {len(sub)})"] = np.mean([M.pop(k) for k in sub], 0)


def hard(a):
    return np.array([a[i, HARD[m]].mean() for i, m in enumerate(MEMES)])


def flip(d, n=10000):
    """One-sided meme-level sign flip of the mean difference, with a bootstrap 95% CI over memes."""
    s = rng.choice([-1, 1], (n, len(d)))
    b = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return dict(diff=round(float(d.mean()), 4), p=float((np.sum((s * d).mean(1) >= d.mean() - 1e-12) + 1) / (n + 1)),
                ci=[round(float(x), 4) for x in np.percentile(b, [2.5, 97.5])])


out = {"accuracy": {k: dict(all=round(float(v.mean()), 4), hard_half=round(float(hard(v).mean()), 4)) for k, v in M.items()}, "contrasts": {}}
def contrast(a, b):
    if a in M and b in M:
        out["contrasts"][f"{a} - {b}"] = dict(all=flip(M[a].mean(1) - M[b].mean(1)), hard_half=flip(hard(M[a]) - hard(M[b])))
for k in ("m1", "m5"):
    contrast(f"qwen3-32b neuron {k}", f"qwen3-8b neuron {k}")
    for t in ("qwen3-8b", "qwen3-32b"):
        contrast(f"{t} sae {k}", f"{t} neuron {k}")
        contrast(f"{t} neuron {k}", "tfidf logreg")
        contrast(f"{t} neuron {k}", "embed probe")
c = out["contrasts"].get("qwen3-32b neuron m1 - qwen3-8b neuron m1")
if c:
    out["decision_32b_clearly_better"] = {k: bool(out["contrasts"][f"qwen3-32b neuron {k} - qwen3-8b neuron {k}"]["all"]["diff"] >= .05 and
                                              out["contrasts"][f"qwen3-32b neuron {k} - qwen3-8b neuron {k}"]["all"]["p"] < .05) for k in ("m1", "m5")}
    out["decision_sae_is_a_rival"] = {t: bool(max(M[f"{t} sae {k}"].mean() - M[f"{t} neuron {k}"].mean() for k in ("m1", "m5")) >= -.05) for t in ("qwen3-8b", "qwen3-32b")}
for t in ("qwen3-8b", "qwen3-32b"):                           # post hoc, added after results
    contrast(f"{t} resid_mid probe", f"{t} neuron m5"); contrast("embed proto", f"{t} neuron m5"); contrast("embed proto", f"{t} neuron m1")
json.dump(out, open(R / "summary.json", "w"), indent=1)
for k, v in sorted(out["accuracy"].items(), key=lambda x: -x[1]["all"]):
    print(f"{v['all']:.3f}  hard {v['hard_half']:.3f}  {k}")
for k, v in out["contrasts"].items():
    print(k, v)
print({k: v for k, v in out.items() if k.startswith("decision")})
