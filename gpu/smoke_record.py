"""Smoke test, part 1: record gate, up and act = silu(gate) * up for every neuron of one MLP layer over a few hundred
sentences, check the recording, and pick neurons to steer.

    python gpu/smoke_record.py --model Qwen/Qwen3-32B --layer 32 --n 300

Sentences come from the study's corpus (study/data/corpus.jsonl), spread across all 114 works, and are read one
sentence at a time with the first token dropped, as in the study's audits. Only layers 0..layer are loaded, which is all
recording needs. Writes, under $PHYLO_DATA (default ~/phylo-data):

    dense/<model>/smoke_L<layer>.safetensors   gate, up, act [tokens, neurons] bf16; token_ids, sent, pos
    dense/<model>/smoke_L<layer>.json          sentences, residual-norm quantiles, checks, steering candidates
"""
import argparse
import collections
import json
import os
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from common import REGIMES, MLPRecorder, load, pick_device, regime_masks

HERE = os.path.dirname(os.path.abspath(__file__))


def sentences(path, n):
    rows = [json.loads(line) for line in open(path)]
    pool = [(r["title"], s) for r in rows for s in r["sentences"]]
    step = max(1, len(pool) // n)
    return pool[::step][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--corpus", default=os.path.join(HERE, "..", "study", "data", "corpus.jsonl"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = pick_device(a.device)
    data = os.path.expanduser(os.environ.get("PHYLO_DATA", "~/phylo-data"))
    out = a.out or os.path.join(data, "dense", a.model.replace("/", "--"), f"smoke_L{a.layer}")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    t0 = time.time()
    tok, model, layers = load(a.model, dev, n_layers=a.layer + 1)
    print(f"loaded layers 0..{a.layer} of {a.model} on {dev} in {time.time() - t0:.0f}s", flush=True)
    if dev == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"CUDA memory: {(total - free) / 2**30:.1f} GiB used of {total / 2**30:.1f} GiB", flush=True)
    rec = MLPRecorder(layers[a.layer])
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model

    sents = sentences(a.corpus, a.n)
    cols = collections.defaultdict(list)
    worst_recompute = 0.0
    t0 = time.time()
    with torch.inference_mode():
        for b in range(0, len(sents), a.batch):
            texts = [s for _, s in sents[b : b + a.batch]]
            enc = tok(texts, return_tensors="pt", padding=True).to(dev)
            lm(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
            g, u, act, resid = rec.gate, rec.up, rec.act, rec.resid
            recompute = F.silu(g.float()) * u.float()
            scale = act.float().abs().amax().clamp_min(1e-6)
            worst_recompute = max(worst_recompute, ((recompute - act.float()).abs().amax() / scale).item())
            keep = enc.attention_mask.bool().clone()
            keep[:, 0] = False  # the first token is an attention sink with outsized activations
            for i in range(len(texts)):
                k = keep[i]
                cols["gate"].append(g[i][k].cpu())
                cols["up"].append(u[i][k].cpu())
                cols["act"].append(act[i][k].cpu())
                cols["resid_norm"].append(resid[i][k].float().norm(dim=-1).cpu())
                cols["token_ids"].append(enc.input_ids[i][k].cpu())
                cols["sent"].append(torch.full((int(k.sum()),), b + i))
                cols["pos"].append(torch.nonzero(k).flatten().cpu())
    dt = time.time() - t0
    t = {k: torch.cat(v) for k, v in cols.items()}
    n_tok, n_neur = t["act"].shape
    print(f"{len(sents)} sentences, {n_tok:,} tokens x {n_neur:,} neurons in {dt:.1f}s "
          f"({n_tok / dt:,.0f} recorded tok/s, layers 0..{a.layer} only)", flush=True)

    # Check 1: the down_proj input is silu(gate) * up, so gate and up were hooked on the same tokens as act
    print(f"check: act vs silu(gate)*up, worst error relative to the batch max = {worst_recompute:.4f} "
          f"({'ok' if worst_recompute < 0.02 else 'MISMATCH'}; bf16 rounding is ~0.004)")

    # Check 2: a sentence read alone gives the same activations as inside a padded batch
    with torch.inference_mode():
        enc = tok([sents[0][1]], return_tensors="pt").to(dev)
        lm(input_ids=enc.input_ids, use_cache=False)
        alone = rec.act[0, 1:].float().cpu()
    batched = t["act"][t["sent"] == 0].float()
    diff = ((alone - batched).abs().amax() / batched.abs().amax()).item()
    print(f"check: sentence 0 alone vs in its batch, worst relative difference = {diff:.4f} "
          f"({'ok' if diff < 0.05 else 'LARGE: check padding side and attention mask'})")

    # Sign regimes
    masks = regime_masks(t["gate"], t["up"])
    share = {r: m.float().mean().item() for r, m in masks.items()}
    print("share of (token, neuron) pairs by regime: " + ", ".join(f"{r} {v:.1%}" for r, v in share.items()))
    act = t["act"].float()
    open_max = torch.where(masks["open+"], act, torch.zeros(())).amax(0)
    echo_max = torch.where(masks["echo+"], act, torch.zeros(())).amax(0)
    echo_wins = (echo_max > open_max).float().mean().item()
    print(f"neurons whose positive maximum here is an echo firing: {echo_wins:.1%} "
          f"(on a few hundred sentences many neurons never open their gate; neuron-explorer reports ~4% over 2M tokens)")
    q = torch.quantile(t["resid_norm"], torch.tensor([0.1, 0.5, 0.9])).tolist()
    print(f"residual norm leaving layer {a.layer}: p10/p50/p90 = {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}")

    # Top gate-open firings of the strongest neurons, and lexical candidates for steering
    def peak_tokens(n, k=8):
        v = torch.where(masks["open+"][:, n], act[:, n], torch.full((), -1.0))
        idx = v.topk(k).indices
        return [(float(act[i, n]), int(i)) for i in idx if v[i] > 0]

    def show(n):
        print(f"\nL{a.layer} N{n}+ (gate-open max {open_max[n]:.2f}, echo max {echo_max[n]:.2f})")
        for val, i in peak_tokens(n, 5):
            s = int(t["sent"][i])
            ids = t["token_ids"][t["sent"] == s].tolist()
            p = int((t["sent"][:i] == s).sum())
            before, peak, after = tok.decode(ids[max(0, p - 12) : p]), tok.decode(ids[p]), tok.decode(ids[p + 1 : p + 6])
            print(f"  {val:7.2f}  ...{before}[[{peak}]]{after}...   ({sents[s][0]})")

    stop = set("the and that this with from for was were his her their they them she him who which what when where "
               "into onto have has had but not are its it's then than also after before about".split())
    candidates = []
    for n in open_max.topk(min(5000, n_neur)).indices.tolist():
        peaks = peak_tokens(n)
        if len(peaks) < 8:
            continue
        words = collections.Counter(tok.decode(int(t["token_ids"][i])).strip().lower() for _, i in peaks)
        word, count = words.most_common(1)[0]
        if count >= 5 and len(word) >= 3 and word.isalpha() and word not in stop | {c["word"] for c in candidates}:
            candidates.append({"neuron": n, "word": word, "of_top8": count, "open_max": float(open_max[n])})
    candidates = sorted(candidates, key=lambda c: (-c["of_top8"], -c["open_max"]))[:10]
    print("\n== the 5 neurons with the largest gate-open activations")
    for n in open_max.topk(5).indices.tolist():
        show(n)
    print("\n== steering candidates: neurons whose top 8 gate-open firings mostly peak on one word")
    for c in candidates:
        print(f"  N{c['neuron']}: '{c['word']}' in {c['of_top8']}/8, max {c['open_max']:.2f}")
    if candidates:
        show(candidates[0]["neuron"])

    save_file({k: v.contiguous() for k, v in t.items()}, out + ".safetensors",
              metadata={"model": a.model, "layer": str(a.layer)})
    json.dump({"model": a.model, "layer": a.layer, "sentences": sents, "tokens": n_tok, "neurons": n_neur,
               "resid_norm_q10_50_90": q, "regime_share": share, "echo_holds_positive_max": echo_wins,
               "checks": {"recompute_rel_err": worst_recompute, "alone_vs_batch_rel_diff": diff},
               "candidates": candidates}, open(out + ".json", "w"), indent=1)
    print(f"\nwrote {out}.safetensors ({os.path.getsize(out + '.safetensors') / 2**30:.2f} GiB) and {out}.json")


if __name__ == "__main__":
    main()
