"""Run every work in the corpus through a model and record, for every MLP neuron, its strongest
positive and negative activation in each sentence.

A "neuron" here is one channel of the SwiGLU hidden layer (silu(gate) * up), the input to down_proj.
Each neuron has two poles, since its sign is arbitrary: the positive pole (sentence max) and the
negative pole (sentence min) are treated as separate units downstream.

Output (in OUT_DIR/<model-tag>/):
  pos.npy, neg.npy   int8 [n_sentences, n_layers * hidden]   quantized per neuron
  pos_scale.npy, neg_scale.npy  float32 [n_neurons]          multiply to recover activations
  meta.json          model, layers, hidden size, sentence -> work mapping
"""
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

MODELS = {
    "qwen3-8b": "mlx-community/Qwen3-8B-4bit",
    "llama3.1-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "qwen3-1.7b": "mlx-community/Qwen3-1.7B-4bit",
}


class Recorder(nn.Module):
    """Wraps an MLP, reducing its hidden activations to per-sentence max and min."""

    def __init__(self, mlp, sink, idx):
        super().__init__()
        self.mlp = mlp
        self._sink = sink
        self._idx = idx

    def __call__(self, x):
        h = nn.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
        g = h[0][self._sink["gather"]]          # [n_sent, longest sentence, hidden], padded by repeating a token
        self._sink["pos"][self._idx] = g.max(axis=1)
        self._sink["neg"][self._idx] = g.min(axis=1)
        return self.mlp.down_proj(h)


def sentence_spans(hf_tok, sentences):
    """Tokenize the joined text and return token ids plus each sentence's [start, end) token span."""
    text, char_spans, pos = "", [], 0
    for s in sentences:
        if text:
            text += " "
        start = len(text)
        text += s
        char_spans.append((start, len(text)))
    enc = hf_tok(text, return_offsets_mapping=True, add_special_tokens=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    spans = []
    for cs, ce in char_spans:
        toks = [i for i, (a, b) in enumerate(offsets) if b > a and a < ce and b > cs and i > 0]
        spans.append((toks[0], toks[-1] + 1))
    return ids, spans


def main(tag, out_dir):
    repo = MODELS[tag]
    model, tokenizer = load(repo)
    hf_tok = tokenizer._tokenizer
    layers = model.model.layers
    n_layers = len(layers)
    hidden = layers[0].mlp.gate_proj.weight.shape[0]
    sink = {"pos": [None] * n_layers, "neg": [None] * n_layers, "gather": None}
    for i, layer in enumerate(layers):
        layer.mlp = Recorder(layer.mlp, sink, i)

    works = [json.loads(l) for l in open(os.environ.get("CORPUS", "data/corpus.jsonl"))]
    n_sent = sum(len(w["sentences"]) for w in works)
    n_neurons = n_layers * hidden
    print(f"{tag}: {n_layers} layers x {hidden} = {n_neurons:,} neurons; {len(works)} works, {n_sent:,} sentences")
    out = Path(out_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    # fp16 scratch arrays live on disk, not in RAM: an 8B model's full activations don't fit beside it
    pos = np.lib.format.open_memmap(out / "pos_f16.npy", mode="w+", dtype=np.float16, shape=(n_sent, n_neurons))
    neg = np.lib.format.open_memmap(out / "neg_f16.npy", mode="w+", dtype=np.float16, shape=(n_sent, n_neurons))
    mx.set_cache_limit(2 * 1024 ** 3)
    sent_work, row, t0, n_tok = [], 0, time.time(), 0
    for w in works:
        ids, spans = sentence_spans(hf_tok, w["sentences"])
        longest = max(e - s for s, e in spans)
        sink["gather"] = mx.array([[min(s + j, e - 1) for j in range(longest)] for s, e in spans])
        model.model(mx.array([ids]))          # hidden layers only; the vocabulary projection isn't needed
        p = mx.concatenate(sink["pos"], axis=1)
        q = mx.concatenate(sink["neg"], axis=1)
        mx.eval(p, q)
        k = len(spans)
        pos[row:row + k] = np.array(p.astype(mx.float16))
        neg[row:row + k] = np.array(q.astype(mx.float16))
        sent_work += [w["id"]] * k
        del p, q
        mx.clear_cache()
        row += k
        n_tok += len(ids)
        print(f"  {w['label'][:30]:30s} {len(ids):5d} tok  {n_tok / (time.time() - t0):6.0f} tok/s", flush=True)

    del model
    mx.clear_cache()
    pos.flush(); neg.flush()
    for name, arr in (("pos", pos), ("neg", neg)):
        q8 = np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=np.int8, shape=arr.shape)
        scale = np.zeros(n_neurons, dtype=np.float32)
        for c in range(0, n_neurons, 65536):
            block = arr[:, c:c + 65536].astype(np.float32)
            s = np.abs(block).max(axis=0) / 127.0 + 1e-8
            q8[:, c:c + 65536] = np.round(block / s).astype(np.int8)
            scale[c:c + 65536] = s
        q8.flush()
        del q8
        np.save(out / f"{name}_scale.npy", scale)
    del pos, neg
    (out / "pos_f16.npy").unlink()
    (out / "neg_f16.npy").unlink()
    json.dump({"tag": tag, "repo": repo, "n_layers": n_layers, "hidden": hidden,
               "sent_work": sent_work}, open(out / "meta.json", "w"))
    print(f"saved to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
