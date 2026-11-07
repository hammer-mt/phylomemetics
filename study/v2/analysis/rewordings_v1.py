"""Version 1's rewording test, re-scored under the v2 rule (PLAN.md Section 10, last paragraph; Section 18, change 31).

    cd study && ../.venv/bin/python v2/analysis/rewordings_v1.py job      texts and units for gpu/score_texts.py
    ... gpu/remote.py score qwen3-8b v2/cache/jobs/rewordings_v1.jsonl v2/cache/jobs/rewordings_v1_units.npy rewordings_v1
    cd study && ../.venv/bin/python v2/analysis/rewordings_v1.py score

Version 1 named 31 Qwen3-8B memes and rewrote each one's top sentence so that it shares almost no content word with
the original (results/wording.json). A meme "survived" there if its unit responded more to the rewording than to any
of ten everyday sentences, 11 of 31 did, and a mismatched-rewording null expected about 5. The same units have the
same indices in the full-precision model. Here every sentence is read alone by that model and scored as a share of
the unit's v2 scale (its own sign regime only), where 30% or more is a strong response.

Decided before any of these scores existed:
  strong on the original     the unit's own top sentence, read alone, reaches 30% of scale
  survives (v2 rule)         the rewording reaches 30% of scale and no everyday sentence does
  survives (v1's criterion)  the rewording's share is above every everyday sentence's, whatever its size
  mismatched null            the same two rules with each unit paired with the other 30 memes' rewordings

Writes results/v2/rewordings_v1_qwen3-8b.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

import common as C

TAG = "qwen3-8b"
JOBS = Path("v2/cache/jobs")
V1 = Path("results/wording.json")


def texts_and_units():
    d = json.load(open(V1))
    texts = []
    for k, m in enumerate(d["memes"]):
        texts.append({"id": f"orig{k:02d}", "text": m["original"]})
        texts.append({"id": f"rew{k:02d}", "text": m["reworded"]})
    texts += [{"id": f"neutral{k:02d}", "text": t} for k, t in enumerate(d["neutral"])]
    return d, texts, np.array([m["unit"] for m in d["memes"]], dtype=np.int64)


def job():
    _, texts, units = texts_and_units()
    JOBS.mkdir(parents=True, exist_ok=True)
    with open(JOBS / "rewordings_v1.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    np.save(JOBS / "rewordings_v1_units.npy", units)
    print(len(texts), "texts,", len(units), "units ->", JOBS / "rewordings_v1.jsonl")


def score():
    d, texts, units = texts_and_units()
    z = np.load(C.ACTS / TAG / "rewordings_v1.npz")
    assert [str(i) for i in z["ids"]] == [t["id"] for t in texts] and (z["units"] == units).all()
    scale = C.scale(TAG)[units].astype(float)
    share = np.where(scale > 0, z["peak"].astype(float) / np.maximum(scale, 1e-9), 0.0)        # [text, unit]
    row = {t["id"]: k for k, t in enumerate(texts)}
    n = len(units)
    neutral = np.array([row[f"neutral{k:02d}"] for k in range(len(d["neutral"]))])
    rew = np.array([[share[row[f"rew{j:02d}"], u] for j in range(n)] for u in range(n)])   # [unit, whose rewording]
    orig = np.array([share[row[f"orig{u:02d}"], u] for u in range(n)])
    neu_max = np.array([share[neutral, u].max() for u in range(n)])
    own = np.diag(rew)
    strong = C.THETA
    v2_ok = (own >= strong) & (neu_max < strong)
    v1_ok = own > neu_max
    off = ~np.eye(n, dtype=bool)
    mis_v2 = ((rew >= strong) & (neu_max[:, None] < strong))[off].mean()
    mis_v1 = (rew > neu_max[:, None])[off].mean()
    rows = [{"unit": int(units[u]), "name": C.unit_name(TAG, int(units[u])), "label": d["memes"][u]["label"], "scale": float(scale[u]),
             "share_original": round(float(orig[u]), 3), "share_reworded": round(float(own[u]), 3), "share_everyday_max": round(float(neu_max[u]), 3),
             "survives_v2_rule": bool(v2_ok[u]), "survives_v1_criterion": bool(v1_ok[u]),
             "v1_survived": bool(d["memes"][u]["qwen"]["reworded"] > d["memes"][u]["qwen"]["neutral_max"])} for u in range(n)]
    out = {"model": TAG, "n_units": n, "theta": strong, "units_with_no_scale_in_their_own_regime": int((scale <= 0).sum()),
           "strong_on_the_original_sentence_read_alone": int((orig >= strong).sum()),
           "strong_on_the_rewording": int((own >= strong).sum()),
           "strong_on_an_everyday_sentence": int((neu_max >= strong).sum()),
           "survive_v2_rule": int(v2_ok.sum()), "survive_v2_rule_expected_from_mismatched_rewordings": round(float(mis_v2 * n), 2),
           "survive_v1_criterion": int(v1_ok.sum()), "survive_v1_criterion_expected_from_mismatched_rewordings": round(float(mis_v1 * n), 2),
           "v1_published_survivors": int(sum(r["v1_survived"] for r in rows)),
           "survive_under_both_versions": int(sum(r["v1_survived"] and r["survives_v2_rule"] for r in rows)),
           "median_share": {"original": float(np.median(orig)), "reworded": float(np.median(own)), "everyday_max": float(np.median(neu_max))},
           "units": rows}
    C.save(out, f"rewordings_v1_{TAG}.json")
    print({k: v for k, v in out.items() if k != "units"})


if __name__ == "__main__":
    {"job": job, "score": score}[sys.argv[1]]()
