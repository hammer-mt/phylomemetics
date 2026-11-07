"""Check our activation extraction against activations published in Cammarata's essay.

The essay's "From spelling fragments to Miyazaki" figure embeds, for a few dozen Qwen3-32B neurons, example
contexts with the activation at a marked token. This script

  prepare   downloads the essay page, pulls those examples out, and writes texts.jsonl, units.npy and
            expected.json into <dir> (nothing from the essay is committed to this repository)
  compare   after `score_texts.py qwen3-32b <dir>/texts.jsonl <dir>/units.npy <dir>/scores.npz` has run on the
            GPU, compares our activation at each marked token with the published value

    python published_examples.py prepare <dir>
    python published_examples.py compare <dir> <summary.json>

The published contexts are cut for display (they start with an ellipsis), so deeper neurons, which depend on
the missing context, are expected to differ; neurons in the first layers depend on little more than the token.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

URL = "https://already-clean.goodfire.com/"
WIDTH, LAYERS = 25600, 64          # Qwen3-32B


def prepare(out):
    import requests
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    html = requests.get(URL, timeout=120).text
    m = re.search(r'<script[^>]*id="miy-mini-data"[^>]*>(.*?)</script>', html, re.S)
    data = json.loads(m.group(1))
    assert data["model"] == "Qwen3-32B", data["model"]
    texts, expected, units = [], [], []
    for c in data["circuits"]:
        for nd in c["nodes"]:
            u = nd["layer"] * WIDTH + nd["neuron"]
            if u not in units:
                units.append(u)
            for side in ("positives", "negatives"):
                for ex in nd.get(side) or []:
                    tid = f"{len(texts)}"
                    texts.append({"id": tid, "text": ex["before"].lstrip("…") + ex["token"]})
                    expected.append({"id": tid, "layer": nd["layer"], "neuron": nd["neuron"], "unit": u, "side": side,
                                     "token": ex["token"], "published": ex["activation"], "truncated": ex["before"].startswith("…"),
                                     "position": ex.get("position"), "circuit": c["id"], "title": nd.get("title")})
    with open(out / "texts.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    np.save(out / "units.npy", np.array(units, dtype=np.int64))
    json.dump(expected, open(out / "expected.json", "w"), ensure_ascii=False)
    print(f"{len(texts)} examples for {len(units)} neurons in layers {sorted({e['layer'] for e in expected})}")


def compare(d, summary_path):
    d = Path(d)
    z = np.load(d / "scores.npz")
    expected = json.load(open(d / "expected.json"))
    tokens = json.load(open(d / "scores.tokens.json"))
    col = {int(u): i for i, u in enumerate(z["units"])}
    rows = []
    for r, e in enumerate(expected):
        ours = float(z["last"][r, col[e["unit"]]])
        last_tok = tokens[r][-1]
        rows.append({**e, "ours": ours, "our_last_token": last_tok})
    out = {"n_examples": len(rows), "n_neurons": len(col), "by_layer": {}, "note": "ours = raw activation at the last token of the published (display-truncated) context"}
    strong = [r for r in rows if abs(r["published"]) >= 1.0]
    for name, sel in (("all", rows), ("published magnitude >= 1", strong)):
        if not sel:
            continue
        p, o = np.array([r["published"] for r in sel]), np.array([r["ours"] for r in sel])
        out[name] = {"n": len(sel), "pearson_r": float(np.corrcoef(p, o)[0, 1]), "same_sign": float(np.mean(np.sign(p) == np.sign(o))),
                     "median_abs_relative_error": float(np.median(np.abs(o - p) / np.abs(p).clip(min=1e-6))),
                     "within_10_percent": float(np.mean(np.abs(o - p) <= 0.1 * np.abs(p)))}
    for layer in sorted({r["layer"] for r in strong}):
        sel = [r for r in strong if r["layer"] == layer]
        p, o = np.array([r["published"] for r in sel]), np.array([r["ours"] for r in sel])
        out["by_layer"][str(layer)] = {"n": len(sel), "median_abs_relative_error": float(np.median(np.abs(o - p) / np.abs(p))),
                                       "within_10_percent": float(np.mean(np.abs(o - p) <= 0.1 * np.abs(p))),
                                       "same_sign": float(np.mean(np.sign(p) == np.sign(o)))}
    out["examples"] = [{k: r[k] for k in ("layer", "neuron", "side", "token", "our_last_token", "published", "ours", "truncated")} for r in rows]
    json.dump(out, open(summary_path, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare(sys.argv[2])
    else:
        compare(sys.argv[2], sys.argv[3])
