"""python steer_summary.py -> results/scale/steer_summary.json. Pre-stated headline (README, part C)."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent.parent / "results/scale"
C = ["funeral", "coin_flip", "secret_identity", "morbid_joke", "revenge_vow"]
out = {}
for f in sorted(R.glob("steer_*_judged.jsonl")):
    rows = [json.loads(x) for x in open(f)]
    g = defaultdict(list)
    for r in rows:
        g[(r["mode"], r["concept"], r["source"] + ":" + r["kind"], r["role"], r["alpha"])].append(r)
    mean = lambda rs, k, ps=None: float(np.mean([r[k] for r in rs if ps is None or r["prompt"] in ps]))
    base_up = g[("up", "none", ":", "none", None)]
    res = {"baseline_up": {k: round(mean(base_up, f"j_{k}"), 3) for k in C} | {"coherent": round(mean(base_up, "j_coherent"), 3)}, "up": [], "down": []}
    for src in ("A:neuron", "B:neuron", "B:sae"):
        for c in C:
            curve = []
            for a in (2, 5, 10, 20):
                t, n = g[("up", c, src, "target", a)], g[("up", c, src, "random", a)]
                curve.append(dict(alpha=a, own=mean(t, f"j_{c}"), own_random=mean(n, f"j_{c}"), coherent=mean(t, "j_coherent"),
                                  coherent_random=mean(n, "j_coherent"), others=float(np.mean([mean(t, f"j_{x}") for x in C if x != c]))))
            lo, hi = range(5), range(5, 10)
            b = mean(base_up, f"j_{c}", lo)
            ok = [(mean(g[("up", c, src, "target", a)], f"j_{c}", lo) - b, a) for a in (2, 5, 10, 20)
                  if mean(g[("up", c, src, "target", a)], "j_coherent", lo) >= .7]
            row = dict(concept=c, source=src, curve=curve)
            if ok:
                a = max(ok)[1]
                t, n = g[("up", c, src, "target", a)], g[("up", c, src, "random", a)]
                row |= dict(alpha=a, own=mean(t, f"j_{c}", hi), random=mean(n, f"j_{c}", hi), baseline=mean(base_up, f"j_{c}", hi),
                            coherent=mean(t, "j_coherent", hi), others=float(np.mean([mean(t, f"j_{x}", hi) for x in C if x != c])))
                row["effect"], row["specificity"] = row["own"] - row["random"], row["own"] - row["others"]
            else:
                row["effect"] = None                                        # never coherent enough on prompts 1 to 5
            res["up"].append(row)
            base = g[("down", c, ":", "none", None)]
            for a in (0, -5):
                t, n = g[("down", c, src, "target", a)], g[("down", c, src, "random", a)]
                res["down"].append(dict(concept=c, source=src, alpha=a, baseline=mean(base, f"j_{c}"), own=mean(t, f"j_{c}"), random=mean(n, f"j_{c}"),
                                        effect=mean(t, f"j_{c}") - mean(n, f"j_{c}"), coherent=mean(t, "j_coherent")))
    res["up_mean_effect"] = {s: round(float(np.mean([r["effect"] or 0 for r in res["up"] if r["source"] == s])), 3) for s in ("A:neuron", "B:neuron", "B:sae")}
    res["down_mean_effect"] = {f"{s} alpha={a}": round(float(np.mean([r["effect"] for r in res["down"] if r["source"] == s and r["alpha"] == a])), 3)
                               for s in ("A:neuron", "B:neuron", "B:sae") for a in (0, -5)}
    out[f.stem[6:-7]] = res
    print("==", f.stem, res["baseline_up"], "\nup", res["up_mean_effect"], "\ndown", res["down_mean_effect"])
    for r in res["up"]:
        print(f"  {r['source']:9s} {r['concept']:16s}", {k: round(r[k], 2) for k in ("alpha", "own", "random", "baseline", "others", "coherent", "effect") if r.get(k) is not None},
              [(x["alpha"], round(x["own"], 2), round(x["own_random"], 2), round(x["coherent"], 2)) for x in r["curve"]])
json.dump(out, open(R / "steer_summary.json", "w"), indent=1, default=float)
