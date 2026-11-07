"""Lay out, for each unit to be explained, what the explainer agent is allowed to see (PLAN.md, Section 10).

    cd study && ../.venv/bin/python v2/analysis/build_explain_inputs.py <model-tag>

Reads v2/acts/<tag>/top_examples.json (from gpu/top_examples.py with --k 24, for each chosen unit and its opposite
pole), the unit's strongest study sentences (v2/acts/<tag>/sent_*.npy) and results/v2/<tag>/explain_units.json.
The explainer sees the 16 strongest reference examples; those ranked 17 to 24 are held out and written apart.
Context is generous, because a unit read in a 1,000-token document may depend on text far before its peak: the six
strongest examples come with their whole document up to the peak, the other ten with about 1,500 characters, and
each study sentence with up to six sentences that precede it in its work.
Writes one plain-text file per unit to v2/cache/explain/<tag>/, results/v2/<tag>/explain_inputs.json (the same
content, committed, so the explanations can be audited) and results/v2/<tag>/heldout_examples.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

import common as C

MODEL_NAMES = {"qwen3-32b": "Qwen3-32B", "qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama 3.1 8B", "llama3.3-70b": "Llama 3.3 70B"}


def clean(tok):
    return tok.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\\n")


def passage(ex, width_before=420, width_after=110):
    cut = width_before is not None and len(ex["before"]) > width_before
    before = (ex["before"][-width_before:] if cut else ex["before"]).replace("\n", " ")
    after = ex["after"][:width_after].replace("\n", " ")
    return f"{'…' if cut else '[start of document] '}{before}«{ex['token']}»{after}…"


def study_sentences(tag, unit, k=8):
    """The unit's k strongest original-text study sentences: (share, work, sentence, peak token, the sentences before it)."""
    d = C.ACTS / tag
    works = [json.loads(l) for l in open("data/corpus.jsonl")]
    sents = [(w["label"], s, " ".join(w["sentences"][max(j - 6, 0):j])) for w in works for j, s in enumerate(w["sentences"])]
    for prefix in ("sent", "sentx"):
        if not (d / f"{prefix}_units.npy").exists():
            continue
        units = np.load(d / f"{prefix}_units.npy")
        hit = np.flatnonzero(units == unit)
        if len(hit):
            col = np.load(d / f"{prefix}_orig.u8.npy", mmap_mode="r")[:, hit[0]]
            where = np.load(d / f"{prefix}_orig_where.npy", mmap_mode="r")[:, hit[0]]
            toks = json.load(open(d / f"{prefix}_tokens.json"))["orig"]
            top = np.argsort(-np.asarray(col), kind="stable")[:k]
            return [(float(col[i]) / C.STEP, sents[i][0], sents[i][1], clean(toks[i][int(where[i])]) if int(where[i]) < len(toks[i]) else "", sents[i][2])
                    for i in top if col[i] > 0]
    return []


def main(tag):
    chosen = json.load(open(C.RESULTS / tag / "explain_units.json"))["units"]
    tops = json.load(open(C.ACTS / tag / "top_examples.json"))
    scale = C.scale(tag)
    out_dir = Path("v2/cache/explain") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    record, held = [], {}
    for c in chosen:
        u, opp = c["unit"], c["opposite"]
        t = tops[str(u)]
        lines = [f"UNIT {u} ({c['name']}) of {MODEL_NAMES.get(tag, tag)}: the {'positive' if t['pole'] == '+' else 'negative'} pole of neuron {t['neuron']} in layer {t['layer']}.",
                 "Strength is given as a share of the strongest activation this unit has ever shown (1.00 = its maximum).",
                 "In each passage the token on which the activation peaked is marked «like this». The unit reads left to right: it has seen the marked token and everything before it, and nothing after.",
                 "", "A. Strongest passages in a reference corpus of about 48 million tokens (film plots, Wikipedia articles and books in English; and some source code,",
                 "   conversations with an assistant, other languages, news and reviews). The first six are shown with their whole document up to the peak:"]
        ref = []
        for i, ex in enumerate(t["examples"][:16], 1):
            share = ex["activation"] / float(scale[u]) if scale[u] > 0 else 0.0
            lines.append(f"{i:2d}. [{share:.2f}; {ex['source']}: {ex['title'][:60]}] {passage(ex, None if i <= 6 else 1500)}")
            ref.append({"share": round(share, 3), "source": ex["source"], "title": ex["title"], "before": ex["before"] if i <= 6 else ex["before"][-1500:],
                        "token": ex["token"], "after": ex["after"][:110]})
        lines += ["", "B. Strongest sentences in the study's plot summaries of Batman works, their relatives and ordinary films (original texts).",
                  "   Each is shown after the sentences that precede it in its summary (in brackets); the unit fired in the last sentence, on the token named:"]
        st = study_sentences(tag, u)
        for i, (share, work, sent, tok, before) in enumerate(st, 1):
            lines.append(f"{i:2d}. [{share:.2f}; {work}] ({before}) >>> {sent}   (peak on «{tok.strip()}»)")
        if not st:
            lines.append("    (none above zero)")
        lines += ["", "C. For context only: the strongest passages of the OPPOSITE pole of the same neuron (do not explain these):"]
        for i, ex in enumerate(tops.get(str(opp), {}).get("examples", [])[:8], 1):
            lines.append(f"{i:2d}. [{ex['source']}: {ex['title'][:60]}] {passage(ex, 300, 60)}")
        (out_dir / f"unit_{u}.txt").write_text("\n".join(lines) + "\n")
        held[str(u)] = [{"rank": ex.get("rank"), "share": round(ex["activation"] / float(scale[u]), 3) if scale[u] > 0 else 0.0, "source": ex["source"],
                         "title": ex["title"], "before": ex["before"][-420:], "token": ex["token"], "after": ex["after"][:110]} for ex in t["examples"][16:24]]
        record.append({"unit": u, "name": c["name"], "set": c["set"], "reference_examples": ref,
                       "study_sentences": [{"share": round(s, 3), "work": w, "sentence": x, "peak_token": k, "preceding": b} for s, w, x, k, b in st]})
    json.dump({"model": tag, "units": record}, open(C.RESULTS / tag / "explain_inputs.json", "w"), ensure_ascii=False)
    json.dump({"model": tag, "note": "reference examples ranked 17 to 24: never shown to the explainer", "units": held},
              open(C.RESULTS / tag / "heldout_examples.json", "w"), ensure_ascii=False)
    print(tag, len(record), "explainer inputs written to", out_dir)


if __name__ == "__main__":
    main(sys.argv[1])
