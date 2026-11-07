"""Pass 1: run the reference corpus through a model and keep, for every unit, its strongest activations.

For each neuron and pole (main sign regime only, see neurons.py) this keeps the K strongest activations seen,
at most one per batch so the examples come from different contexts, with where each occurred
(document index * 2048 + token position). The strongest of them is the unit's reference maximum: the scale
against which "fires strongly" is judged everywhere else, as in Cammarata's essay ("a share of the neuron's
strongest example in the corpus"). The raw maxima over all sign regimes are kept too, to flag units whose
strongest positive responses are the double-negative echo.

    python ref_pass.py <model-tag> <reference.jsonl> <out-root> [--batch-tokens 16384] [--limit-docs N]
    python ref_pass.py <model-tag> <supplement.jsonl> <out-root> --name ref_sup --max-tokens 2048

The second form reads the scale supplement (v2/build_supplement.py) into its own folder; finalize_scale.py merges the two.

Writes <out-root>/<tag>/<name>/: top_pos_v.npy, top_pos_i.npy, top_neg_v.npy, top_neg_i.npy  [layers, K, width]
(negative-pole values are stored as magnitudes), raw_pos_max.npy, raw_neg_min.npy [layers, width], meta.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from neurons import Tap, geometry, layers_of, load, now, provenance, run_hidden, save_json

K = 24
MAX_TOKENS = 1024


def batches_by_length(lens, batch_tokens):
    """Document indices grouped longest first, each batch at most batch_tokens padded tokens."""
    order = np.argsort(-lens, kind="stable")
    out, cur = [], []
    for i in order:
        width = lens[cur[0]] if cur else lens[i]
        if cur and width * (len(cur) + 1) > batch_tokens:
            out.append(cur)
            cur = []
        cur.append(int(i))
    if cur:
        out.append(cur)
    return out


class TopK:
    """Per layer: the K strongest values per neuron, one candidate per batch, with their locations."""

    def __init__(self, model, k):
        self.v, self.i = [], []
        L, D = geometry(model)
        for layer in layers_of(model):
            dev = layer.mlp.gate_proj.weight.device
            self.v.append(torch.zeros(k, D, device=dev))
            self.i.append(torch.full((k, D), -1, dtype=torch.int32, device=dev))

    def update(self, layer, val, loc):
        v, i = self.v[layer], self.i[layer]
        low, slot = v.min(0)
        cols = torch.nonzero(val > low).squeeze(1)
        if len(cols):
            v[slot[cols], cols] = val[cols]
            i[slot[cols], cols] = loc[cols]

    def sorted(self):
        vs, is_ = [], []
        for v, i in zip(self.v, self.i):
            order = v.argsort(0, descending=True)
            vs.append(v.gather(0, order).cpu().numpy())
            is_.append(i.gather(0, order).cpu().numpy())
        return np.stack(vs), np.stack(is_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("reference")
    ap.add_argument("out")
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--checkpoint-every", type=int, default=250, help="batches between checkpoints; a rerun resumes from the last one")
    ap.add_argument("--name", default="ref", help="output folder under <out-root>/<tag>/")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    a = ap.parse_args()

    out = Path(a.out) / a.tag / a.name
    out.mkdir(parents=True, exist_ok=True)
    docs = [json.loads(l) for l in open(a.reference)]
    if a.limit_docs:
        docs = docs[:: max(1, len(docs) // a.limit_docs)][: a.limit_docs]
    model, tok = load(a.tag)
    L, D = geometry(model)
    print(f"{now()} {a.tag}: {L} layers x {D} = {L * D:,} neurons; {len(docs):,} documents", flush=True)

    ids_all = []
    for s in range(0, len(docs), 2000):
        ids_all += tok([d["text"] for d in docs[s:s + 2000]], add_special_tokens=True, truncation=True, max_length=a.max_tokens)["input_ids"]
    lens = np.array([len(x) for x in ids_all])
    batches = batches_by_length(lens, a.batch_tokens)
    doc_index = np.array([d.get("i", j) for j, d in enumerate(docs)])
    print(f"{now()} {int(lens.sum()):,} tokens in {len(batches)} batches", flush=True)

    pos, neg = TopK(model, a.k), TopK(model, a.k)
    devs = [l.mlp.gate_proj.weight.device for l in layers_of(model)]
    raw_pos = [torch.zeros(D, device=d) for d in devs]
    raw_neg = [torch.zeros(D, device=d) for d in devs]
    ctx = {}

    def fn(layer, g, h):
        mask, docs_b = ctx["mask"].to(h.device), ctx["docs"].to(h.device)
        h = h * mask                                         # padding and position 0 contribute nothing
        raw_pos[layer] = torch.maximum(raw_pos[layer], h.amax((0, 1)).float())
        raw_neg[layer] = torch.minimum(raw_neg[layer], h.amin((0, 1)).float())
        main = torch.where(g > 0, h, torch.zeros_like(h))
        for store, sign in ((pos, 1), (neg, -1)):
            v, t = (main.max(1) if sign > 0 else main.min(1))     # [B, D] strongest per document, and where
            bv, bi = (v.max(0) if sign > 0 else v.min(0))         # [D] strongest in the batch, and which document
            tp = t.gather(0, bi[None])[0]
            store.update(layer, sign * bv.float(), (docs_b[bi] * 2048 + tp).to(torch.int32))
        return None

    def checkpoint(done):
        torch.save({"done": done, "fingerprint": fingerprint, "pos_v": [x.cpu() for x in pos.v], "pos_i": [x.cpu() for x in pos.i],
                    "neg_v": [x.cpu() for x in neg.v], "neg_i": [x.cpu() for x in neg.i],
                    "raw_pos": [x.cpu() for x in raw_pos], "raw_neg": [x.cpu() for x in raw_neg]}, out / "checkpoint.tmp")
        (out / "checkpoint.tmp").replace(out / "checkpoint.pt")

    # a checkpoint only fits the batch list it was made with: a resume with other arguments would skip or repeat documents
    fingerprint = [a.batch_tokens, a.k, a.max_tokens, len(docs), len(batches), int(sum(b[0] * (j + 1) for j, b in enumerate(batches)) % (1 << 61))]
    start = 0
    if (out / "checkpoint.pt").exists():
        ck = torch.load(out / "checkpoint.pt")
        assert ck.get("fingerprint", fingerprint) == fingerprint, "the checkpoint was made with other arguments; delete it or rerun with the same ones"
        start = ck["done"]
        for l in range(L):
            pos.v[l], pos.i[l] = ck["pos_v"][l].to(devs[l]), ck["pos_i"][l].to(devs[l])
            neg.v[l], neg.i[l] = ck["neg_v"][l].to(devs[l]), ck["neg_i"][l].to(devs[l])
            raw_pos[l], raw_neg[l] = ck["raw_pos"][l].to(devs[l]), ck["raw_neg"][l].to(devs[l])
        print(f"{now()} resuming after batch {start}", flush=True)

    tap = Tap(model, fn)
    t0, n_tok = time.time(), 0
    for bi, b in enumerate(batches):
        if bi < start:
            continue
        if bi > start and (bi - start) % a.checkpoint_every == 0:
            checkpoint(bi)
        width = int(lens[b].max())
        ids = torch.full((len(b), width), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(b), width), dtype=torch.long)
        for r, j in enumerate(b):
            ids[r, :lens[j]] = torch.tensor(ids_all[j])
            att[r, :lens[j]] = 1
        m = att.clone()
        m[:, 0] = 0
        ctx["mask"] = m[:, :, None].to(model.dtype)
        ctx["docs"] = torch.tensor(doc_index[b], dtype=torch.long)
        run_hidden(model, ids, att)
        n_tok += int(lens[b].sum())
        if bi % 20 == 0 or bi == len(batches) - 1:
            el = time.time() - t0
            print(f"{now()} batch {bi + 1}/{len(batches)}  {n_tok:,} tokens  {n_tok / el:,.0f} tok/s  "
                  f"eta {((lens.sum() - n_tok) / max(n_tok / el, 1)) / 60:.0f} min", flush=True)
    tap.remove()

    for name, store in (("pos", pos), ("neg", neg)):
        v, i = store.sorted()
        np.save(out / f"top_{name}_v.npy", v)
        np.save(out / f"top_{name}_i.npy", i)
    np.save(out / "raw_pos_max.npy", torch.stack([x.cpu() for x in raw_pos]).numpy())
    np.save(out / "raw_neg_min.npy", torch.stack([x.cpu() for x in raw_neg]).numpy())
    save_json({"tag": a.tag, "repo": getattr(model.config, "_name_or_path", a.tag), "n_layers": L, "width": D, "k": a.k,
               "max_tokens": a.max_tokens, "resumed_from_batch": start, "provenance": provenance(model), "n_docs": len(docs), "n_tokens": int(lens.sum()), "doc_index": doc_index.tolist(),
               "doc_tokens": lens.tolist(), "seconds": round(time.time() - t0), "dtype": str(model.dtype),
               "batch_tokens": a.batch_tokens}, out / "meta.json")
    if torch.cuda.is_available():
        print(f"{now()} peak GPU memory {max(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())) / 2**30:.1f} GB", flush=True)
    (out / "checkpoint.pt").unlink(missing_ok=True)
    print(f"{now()} saved to {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
