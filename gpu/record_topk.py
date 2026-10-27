"""Record every MLP neuron over a large corpus: the top-K firings of each neuron in each sign regime, with pointers back
to the tokens, plus per-regime firing counts. This is the corpus scan of Cammarata's method (each neuron's strongest
examples, read in context), written so it can run for days and resume.

    # throughput benchmark: 2M tokens, all layers, prints the measured rate and the time 100M / 1B / 10B would take
    python gpu/record_topk.py --model Qwen/Qwen3-32B --tokens 2000000 --run bench-q3-32b

    # the real run
    python gpu/record_topk.py --model Qwen/Qwen3-32B --tokens 1000000000 --run q3-32b-1B --k 64

Storing every activation is impossible at this scale (Qwen3-32B has 1.6M neurons; 1B tokens of one regime would be
3 PB), so the scan keeps, per layer, neuron and regime (open+, open-, echo+, echo-, see common.py), the K largest
|act| with their global token positions. Token ids are saved once, so any firing can be read back in context.
Writes under $PHYLO_DATA/acts/<model>/<run>/:

    tokens/tokens-<batch>.npy   uint32 [windows, seq_len]: the windows since the previous checkpoint; in name order
                                they are windows 0, 1, 2, ... of the run
    L<layer>.npz                per regime: top_val float32 [K, I], top_pos int64 [K, I] (window * seq_len + offset),
                                count int64 [I] (tokens in the regime)
    state.json                  windows done, tokens done, settings; --resume continues from here

Position 0 of every window is skipped (attention sink). Windows pack documents separated by EOS, as neuron-explorer's
neuron_stats.py does, from FineWeb-Edu mixed 1:1 with UltraChat rendered through the chat template.
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import torch
from datasets import load_dataset

from common import REGIMES, load, pick_device


def text_stream(tok, seed):
    web = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed)
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).shuffle(seed=seed)
    for w, c in zip(web, chat):
        yield w["text"]
        yield tok.apply_chat_template(c["messages"], tokenize=False, enable_thinking=False)


def windows(tok, seq_len, batch, seed):
    buf, eos = [], tok.eos_token_id
    for text in text_stream(tok, seed):
        buf.extend(tok(text, add_special_tokens=False)["input_ids"] + [eos])
        while len(buf) >= seq_len * batch:
            chunk, buf = buf[: seq_len * batch], buf[seq_len * batch :]
            yield torch.tensor(chunk).view(batch, seq_len)


class TopK:
    """Running top-K of |act| per neuron for each sign regime of one layer, kept on the GPU."""

    def __init__(self, n, k, device):
        self.k = k
        self.val = {r: torch.full((k, n), -1.0, device=device) for r in REGIMES}
        self.pos = {r: torch.full((k, n), -1, device=device, dtype=torch.int64) for r in REGIMES}
        self.count = {r: torch.zeros(n, device=device, dtype=torch.int64) for r in REGIMES}
        self.gate = self.up = None

    def update(self, act, pos):
        g, u, a = (t[:, 1:].flatten(0, 1) for t in (self.gate, self.up, act))  # [T, I], position 0 dropped
        gp, upos = g > 0, u > 0
        mag = a.float().abs()
        for r, m in (("open+", gp & upos), ("open-", gp & ~upos), ("echo+", ~gp & ~upos), ("echo-", ~gp & upos)):
            self.count[r] += m.sum(0)
            v, i = torch.where(m, mag, -1.0).topk(self.k, dim=0)  # [K, I]
            allv = torch.cat([self.val[r], v])
            allp = torch.cat([self.pos[r], pos[i]])
            best = allv.topk(self.k, dim=0).indices
            self.val[r] = allv.gather(0, best)
            self.pos[r] = allp.gather(0, best)

    def save(self, path):
        arrs = {}
        for r in REGIMES:
            key = r.replace("+", "_pos").replace("-", "_neg")
            arrs[f"{key}_top_val"] = self.val[r].cpu().numpy()
            arrs[f"{key}_top_pos"] = self.pos[r].cpu().numpy()
            arrs[f"{key}_count"] = self.count[r].cpu().numpy()
        np.savez(path + ".tmp.npz", **arrs)
        os.replace(path + ".tmp.npz", path)

    def load(self, path):
        z = np.load(path)
        for r in REGIMES:
            key = r.replace("+", "_pos").replace("-", "_neg")
            self.val[r].copy_(torch.from_numpy(z[f"{key}_top_val"]))
            self.pos[r].copy_(torch.from_numpy(z[f"{key}_top_pos"]))
            self.count[r].copy_(torch.from_numpy(z[f"{key}_count"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=500, help="batches between checkpoints")
    ap.add_argument("--run", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    dev = pick_device(a.device)
    data = os.path.expanduser(os.environ.get("PHYLO_DATA", "~/phylo-data"))
    out = os.path.join(data, "acts", a.model.replace("/", "--"), a.run)
    os.makedirs(os.path.join(out, "tokens"), exist_ok=True)

    tok, model, layers = load(a.model, dev)
    wanted = list(range(len(layers))) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    del layers[max(wanted) + 1 :]
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model

    tops = {}
    pos_holder = {}
    for l in wanted:
        mlp = layers[l].mlp
        s = tops[l] = TopK(mlp.down_proj.in_features, a.k, dev)
        mlp.gate_proj.register_forward_hook(lambda m, i, o, s=s: setattr(s, "gate", o))
        mlp.up_proj.register_forward_hook(lambda m, i, o, s=s: setattr(s, "up", o))
        mlp.down_proj.register_forward_pre_hook(lambda m, i, s=s: s.update(i[0], pos_holder["pos"]))

    state_path = os.path.join(out, "state.json")
    start = 0
    if a.resume and os.path.exists(state_path):
        st = json.load(open(state_path))
        assert (st["model"], st["seq_len"], st["batch"], st["seed"], st["k"]) == (a.model, a.seq_len, a.batch, a.seed, a.k)
        start = st["batches_done"]
        for l in wanted:
            tops[l].load(os.path.join(out, f"L{l:02d}.npz"))
        print(f"resuming after {start} batches ({st['tokens_done']:,} tokens); skipping them in the stream", flush=True)
    else:
        json.dump({"model": a.model, "seq_len": a.seq_len, "batch": a.batch, "seed": a.seed, "k": a.k,
                   "layers": wanted, "batches_done": 0, "tokens_done": 0}, open(state_path, "w"))

    n_steps = -(-a.tokens // (a.seq_len * a.batch))
    stream = itertools.islice(windows(tok, a.seq_len, a.batch, a.seed), start, n_steps)
    offs = torch.arange(1, a.seq_len, device=dev)
    shard, t0, done_here = [], time.time(), 0

    def checkpoint(step):
        if shard:
            np.save(os.path.join(out, "tokens", f"tokens-{step:09d}.npy"), np.concatenate(shard).astype(np.uint32))
            shard.clear()
        for l in wanted:
            tops[l].save(os.path.join(out, f"L{l:02d}.npz"))
        json.dump({"model": a.model, "seq_len": a.seq_len, "batch": a.batch, "seed": a.seed, "k": a.k,
                   "layers": wanted, "batches_done": step, "tokens_done": step * a.batch * a.seq_len},
                  open(state_path, "w"))

    step = start
    with torch.inference_mode():
        for step, ids in enumerate(stream, start=start + 1):
            first_window = (step - 1) * a.batch
            win = torch.arange(first_window, first_window + a.batch, device=dev)
            pos_holder["pos"] = (win[:, None] * a.seq_len + offs[None, :]).flatten()
            lm(input_ids=ids.to(dev), use_cache=False)
            shard.append(ids.numpy())
            done_here += a.batch * a.seq_len
            if step % 10 == 0 or step == n_steps:
                rate = done_here / (time.time() - t0)
                eta = (n_steps - step) * a.batch * a.seq_len / rate
                print(f"{step}/{n_steps} batches, {step * a.batch * a.seq_len:,} tokens, {rate:,.0f} tok/s, "
                      f"{eta / 3600:.1f} h left", flush=True)
            if step % a.checkpoint_every == 0:
                checkpoint(step)
    checkpoint(step)

    rate = done_here / (time.time() - t0)
    print(f"\nmeasured {rate:,.0f} tok/s recording {len(wanted)} layers x {tops[wanted[0]].count['open+'].numel():,} "
          f"neurons, top-{a.k} in 4 regimes")
    for n in (1e8, 1e9, 1e10):
        print(f"  {n:.0e} tokens: {n / rate / 3600:,.1f} h = {n / rate / 86400:,.1f} days")


if __name__ == "__main__":
    main()
