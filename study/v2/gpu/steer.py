"""Steering: hold one unit's activation at a multiple of its scale and see what the model writes.

    python steer.py <model-tag> <out-root> <units.npy> <out.json> [--multiples 1 2 3] [--tokens 40] [--batch 48]
                    [--scale-from ref|study]

For every unit, every opener and every multiple, the model continues the opener (greedy, plain text, no chat
template) with that unit's activation set to multiple x scale on its own pole at every position after the first,
in the prompt and in what it writes. Every batch carries one unsteered row, and each steered continuation is kept
with the unsteered continuation of its own batch (in bfloat16 a greedy continuation can depend on the batch).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neurons import Tap, geometry, load, now

OPENERS = ["The story begins when", "In the next scene,", "The film is about"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("out_root"); ap.add_argument("units"); ap.add_argument("out")
    ap.add_argument("--multiples", type=float, nargs="+", default=[1, 2, 3])
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--scale-from", default="study")
    a = ap.parse_args()
    root = Path(a.out_root) / a.tag
    units = np.load(a.units).astype(np.int64)
    model, tok = load(a.tag)
    L, D = geometry(model)
    U = L * D
    if a.scale_from == "study":
        scale = np.load(root / "study" / "scale.npy")
    else:
        scale = np.concatenate([np.load(root / "ref" / "top_pos_v.npy")[:, 0, :].reshape(-1), np.load(root / "ref" / "top_neg_v.npy")[:, 0, :].reshape(-1)])
    pole, layer, col = units // U, (units % U) // D, units % D
    dev = model.model.embed_tokens.weight.device
    ctx = {}

    def fn(l, g, h):
        hit = ctx["by_layer"].get(l)
        if hit is None:
            return None
        rows, cols, vals = hit
        h = h.clone()
        first = 1 if h.shape[1] > 1 else 0             # the prompt pass: position 0 is left alone; later passes hold one new token
        h[rows.to(h.device), first:, cols.to(h.device)] = vals.to(h.device, h.dtype)[:, None]
        return h

    tap = Tap(model, fn)
    out = {"model": a.tag, "openers": OPENERS, "multiples": a.multiples, "tokens": a.tokens, "baseline": [], "steered": []}
    for pi, prompt in enumerate(OPENERS):
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        normals = set()
        jobs = [(ui, m) for ui in range(len(units)) for m in a.multiples]
        for s in range(0, len(jobs), a.batch):
            chunk = jobs[s:s + a.batch]
            by = {}
            for r, (ui, m) in enumerate(chunk):
                by.setdefault(int(layer[ui]), []).append((r, int(col[ui]), float(m * scale[units[ui]] * (1 if pole[ui] == 0 else -1))))
            ctx["by_layer"] = {l: (torch.tensor([x[0] for x in v]), torch.tensor([x[1] for x in v]), torch.tensor([x[2] for x in v])) for l, v in by.items()}
            with torch.no_grad():                       # the last row of every batch is left unsteered
                gen = model.generate(ids.repeat(len(chunk) + 1, 1), max_new_tokens=a.tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            normal = tok.decode(gen[len(chunk), ids.shape[1]:], skip_special_tokens=True)
            normals.add(normal)
            for r, (ui, m) in enumerate(chunk):
                out["steered"].append({"unit": int(units[ui]), "opener": pi, "multiple": m, "normal": normal,
                                       "text": tok.decode(gen[r, ids.shape[1]:], skip_special_tokens=True)})
        out["baseline"].append(sorted(normals)[0])
        out.setdefault("distinct_unsteered_continuations", []).append(len(normals))
        print(f"{now()} opener {pi + 1}/{len(OPENERS)} done", flush=True)
    tap.remove()
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print(f"{now()} wrote {len(out['steered'])} continuations to {a.out}", flush=True)


if __name__ == "__main__":
    main()
