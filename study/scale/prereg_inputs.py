"""python prereg_inputs.py -> results/scale/prereg_inputs.json. Post hoc: DEV pair-accuracy gaps behind preregistration/PROPOSED_CHANGES.md,
by meme group, with bootstrap CIs over memes, and a check that overlap with the selection sentences does not drive the results."""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from items import HERE, STUDY
R = STUDY / "results/scale"
rng = np.random.default_rng(0)
tst, sel = (json.load(open(HERE / f"data/{n}.json")) for n in ("test", "select"))
MEMES = list(tst)
tier = {m["id"]: m["tier"] for m in json.load(open(STUDY.parent / "preregistration/memes_for_tracing.json"))["memes"]}
A = {}
for t in ("qwen3-8b", "qwen3-32b"):
    d = json.load(open(R / f"{t}_features.json"))["B"]
    for k, q in (("neuron", "m1"), ("neuron", "m5"), ("sae", "m1"), ("sae", "m5"), ("resid_mid", "probe"), ("resid_mid", "proto")):
        A[f"{t} {k} {q}"] = np.array([d[k][q][m] for m in MEMES])
for f, ks in (("tfidf", ["logreg"]), ("embed", ["proto", "probe"]), ("judge_qwen3-8b", ["p_yes"])):
    for k in ks:
        A[f"{f} {k}"] = np.array([json.load(open(R / f"baseline_{f}.json"))[k][m] for m in MEMES])
B = rng.integers(0, len(MEMES), (10000, len(MEMES)))


def gap(a, b):
    d = (A[a] - A[b]).mean(1)
    s = d[B].mean(1)
    neu = np.array([tier[m] == "N" for m in MEMES])
    return dict(diff=round(float(d.mean()), 4), ci95=[round(float(x), 4) for x in np.percentile(s, [2.5, 97.5])],
                one_sided_95_lower=round(float(np.percentile(s, 5)), 4), one_sided_95_upper=round(float(np.percentile(s, 95)), 4),
                neutral10=round(float(d[neu].mean()), 4), story24=round(float(d[~neu].mean()), 4))


gaps = {f"{t} {a} - {b}": gap(f"{t} {a}", b if b.split()[0] in ("tfidf", "embed", "judge_qwen3-8b") else f"{t} {b}")
        for t in ("qwen3-8b", "qwen3-32b") for a, b in (("neuron m5", "neuron m1"), ("neuron m5", "tfidf logreg"), ("neuron m1", "tfidf logreg"),
            ("neuron m5", "embed proto"), ("neuron m5", "embed probe"), ("neuron m5", "resid_mid probe"), ("neuron m5", "sae m5"), ("neuron m1", "sae m1"))}
v = TfidfVectorizer(stop_words="english").fit([t for d in sel.values() for k in d for t in d[k]] + [x for d in tst.values() for p in d["pairs"] for x in p.values()])
sim = np.array([[float((v.transform([p["present"]]) @ v.transform(sel[m]["explicit"] + sel[m]["reworded"]).T).toarray().max()) for p in tst[m]["pairs"]] for m in MEMES])
lo = sim <= np.median(sim)
ov = {k: dict(all=round(float(a.mean()), 4), least_similar_half=round(float(a[lo].mean()), 4), most_similar_half=round(float(a[~lo].mean()), 4)) for k, a in A.items()}
ceiling = {k: round(float((a.mean(1) == 1).mean()), 3) for k, a in A.items()}
json.dump(dict(gaps=gaps, select_overlap=ov, share_of_memes_at_ceiling=ceiling), open(R / "prereg_inputs.json", "w"), indent=1)
for k, g in gaps.items(): print(k, g)
print(ceiling)
