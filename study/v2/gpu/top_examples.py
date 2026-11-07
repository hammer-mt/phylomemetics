"""Read back the strongest reference-corpus examples of chosen units, as text with the peak token marked.

    python top_examples.py <model-tag> <reference.jsonl[,supplement.jsonl]> <out-root> <units.npy> <out.json> [--k 24] [--skip 0] [--after 25]

For each unit: its scale, and k of its strongest examples from pass 1 (reference corpus and supplement merged by
finalize_scale.py; after skipping the first --skip), each with the document's source and title, the whole document
up to the peak token, the peak token, and a little of the text after it. Examples from the five reference documents
that are study works under older titles are passed over. No model is loaded, only the tokenizer.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from finalize_scale import OVERLAP
from neurons import MODELS

MAX_TOKENS = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("reference"); ap.add_argument("out_root"); ap.add_argument("units"); ap.add_argument("out")
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--skip", type=int, default=0, help="pass over this many of the strongest examples first (the held-out ones are 17 to 24)")
    ap.add_argument("--before", type=int, default=2048, help="tokens of the document kept before the peak (default: all of them)")
    ap.add_argument("--after", type=int, default=25)
    a = ap.parse_args()
    ref = Path(a.out_root) / a.tag / "ref_all"
    meta = json.load(open(Path(a.out_root) / a.tag / "ref" / "meta.json"))
    sup = Path(a.out_root) / a.tag / "ref_sup" / "meta.json"
    limit = {False: meta["max_tokens"], True: json.load(open(sup))["max_tokens"] if sup.exists() else meta["max_tokens"]}
    L, D = meta["n_layers"], meta["width"]
    U = L * D
    units = np.load(a.units).astype(np.int64)
    tv = {0: np.load(ref / "top_pos_v.npy", mmap_mode="r"), 1: np.load(ref / "top_neg_v.npy", mmap_mode="r")}
    ti = {0: np.load(ref / "top_pos_i.npy", mmap_mode="r"), 1: np.load(ref / "top_neg_i.npy", mmap_mode="r")}
    docs = {}
    for path in a.reference.split(","):
        for line in open(path):
            d = json.loads(line)
            docs[d["i"]] = d
    tok = AutoTokenizer.from_pretrained(MODELS.get(a.tag, a.tag))
    enc_cache = {}

    def enc(di):
        if di not in enc_cache:
            e = tok(docs[di]["text"], add_special_tokens=True, truncation=True, max_length=limit[docs[di]["src"].startswith("sup_")], return_offsets_mapping=True)
            enc_cache[di] = (e["input_ids"], e["offset_mapping"])
        return enc_cache[di]

    out = {}
    for u in units:
        pole, layer, n = int(u // U), int((u % U) // D), int(u % D)
        vals, locs = np.asarray(tv[pole][layer, :, n]), np.asarray(ti[pole][layer, :, n])
        ex, rank = [], 0
        kept = [float(v) for v, loc in zip(vals, locs) if loc >= 0 and v > 0 and int(loc // 2048) not in OVERLAP]
        top = kept[0] if kept else 0.0
        for v, loc in zip(vals, locs):
            if loc < 0 or v <= 0 or int(loc // 2048) in OVERLAP:
                continue
            rank += 1
            if rank <= a.skip:
                continue
            if len(ex) == a.k:
                break
            di, pos = int(loc // 2048), int(loc % 2048)
            ids, offs = enc(di)
            text = docs[di]["text"]
            s, e = offs[pos]
            lo = offs[max(pos - a.before, 0)][0]
            hi = offs[min(pos + a.after, len(ids) - 1)][1]
            ex.append({"rank": rank, "activation": round(float(v), 3), "share_of_reference_max": round(float(v / top), 3), "source": docs[di]["src"],
                       "title": docs[di].get("title", ""), "doc": di, "position": pos, "before": text[lo:s], "token": text[s:e], "after": text[e:hi]})
        out[str(int(u))] = {"unit": int(u), "layer": layer, "neuron": n, "pole": "+" if pole == 0 else "-",
                            "reference_max": round(top, 4), "examples": ex}
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print(f"wrote {len(out)} units to {a.out}")


if __name__ == "__main__":
    main()
