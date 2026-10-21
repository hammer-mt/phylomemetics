"""Compact neuron extraction for the pre-registration (frozen; prereg-v2).

Runs every work of a texts file (preregistration/data/<set>.jsonl, one {"id", "sentences"} per line) through a cached 4-bit
MLX model as ONE continuous document, exactly as extract.py does (same SwiGLU recording point, same sentence
spans, same per-sentence max and min over each sentence's tokens), and stores, per set and model, in
preregistration/features/<set>/neurons_<model>/:

  agg.npy     float16 [n_works, 2N]   per work and unit, the mean of its 3 strongest sentences (fewer if the work is
                                      shorter). Unit u < N is channel u's positive pole (sentence max); unit u >= N
                                      is channel u - N's negative pole, stored as minus the sentence min, so larger
                                      always means stronger. This is extract.py + experiments.work_scores_masked,
                                      computed on the GPU in float32 instead of from int8 copies.
  sent.npy    float16 [n_sentences, k] per-sentence values for the units in --units (same sign rule)
  units.npy   int64 [k]
  full.npy    float16 [n_sentences, 2N] every unit for every sentence (--full; only for small sets: the T2
                                      examples and background, and DEV for the T5 partner search)
  hidden.npy  float16 [n_sentences, 4, d] mean residual-stream output over the sentence's tokens at 4 layers
                                      (--hidden; quarter, half, three-quarter and last block)
  meta.json   work ids, sentence -> work index, model, units, hidden layers, per-work and total seconds, tokens,
              peak memory, storage bytes
and writes preregistration/features/<set>/cost_neurons_<model>.json.

--prefix adds a fixed neutral sentence before each work whose span is not recorded (used for the T2 example
sentences, which are processed one per work so that no example sees another). A work with a "context_sentences"
field gets those sentences before its own, unrecorded (the m1ctx variant: each example after a fixed DEV paragraph).
Hidden states are captured only with --hidden; bench.py times a neurons-only pass separately so that hidden-state
capture is never charged to the neurons.

usage: HF_HUB_OFFLINE=1 .venv/bin/python preregistration/extract_compact.py <set> <qwen3-8b|llama3.1-8b>
         [--units preregistration/sentence_units_<model>.npy] [--full] [--hidden] [--prefix] [--fresh] [--data DIR] [--out DIR]
Resumable: completed works are recorded in progress.json and skipped on a rerun. progress.json and meta.json store the
texts file's sha256 and number of works; a rerun on a changed file aborts unless --fresh is given, which deletes the old
arrays first (so stale rows can never survive a rebuilt set).
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

PRE = Path(__file__).resolve().parent
MODELS = {"qwen3-8b": ("mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"),          # pinned snapshots (the DEV
          "llama3.1-8b": ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "241a666dad6cb93c8ff213d39a7f34a36bf26db4")}   # study's), as readers.py


MLX_FILES = ["*.json", "*.safetensors", "*.py", "tokenizer.model", "*.tiktoken", "*.txt", "*.jsonl", "*.jinja"]   # the files mlx_lm loads


def _local(repo, rev):
    """The local path of a pinned snapshot (offline after step 0; only the files mlx_lm loads are required)."""
    if rev is None:
        return repo
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=rev, allow_patterns=MLX_FILES)
HIDDEN_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
PREFIX = "This is a sentence from a plot summary."
TOPK = 3


class Recorder(nn.Module):
    """extract.py's recorder: the SwiGLU product silu(gate(x)) * up(x), reduced to per-sentence max and min."""

    def __init__(self, mlp, sink, idx):
        super().__init__()
        self.mlp = mlp
        self._sink = sink
        self._idx = idx

    def __call__(self, x):
        h = nn.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
        g = h[0][self._sink["gather"]]
        self._sink["pos"][self._idx] = g.max(axis=1)
        self._sink["neg"][self._idx] = g.min(axis=1)
        return self.mlp.down_proj(h)


class LayerTap(nn.Module):
    """Wraps a decoder block and keeps the mean of its output over each sentence's tokens."""

    def __init__(self, inner, sink, key):
        super().__init__()
        self.inner = inner
        self._sink = sink
        self._key = key

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            if key == "inner" or "inner" not in self:
                raise
            return getattr(self["inner"], key)

    def __call__(self, x, *a, **k):
        y = self.inner(x, *a, **k)
        g = y[0][self._sink["gather"]]                      # [n_sent, longest, d]
        w = self._sink["weights"]                           # [n_sent, longest, 1]
        self._sink["hidden"][self._key] = (g * w).sum(axis=1) / w.sum(axis=1)
        return y


def sentence_spans(hf_tok, sentences, prefix=None):
    """extract.py's tokenisation: the sentences joined by single spaces, one sequence, special tokens added; each
    sentence's [start, end) token span (the first token of the sequence is never part of a span). With prefix (a
    string), fixed text precedes the sentences and gets no span."""
    text, char_spans = "", []
    if prefix:
        text = prefix
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("model", choices=list(MODELS))
    ap.add_argument("--units", default=None, help=".npy of unit indices to store per sentence")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--hidden", action="store_true")
    ap.add_argument("--prefix", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="delete existing arrays and progress before extracting")
    ap.add_argument("--data", default=str(PRE / "data"))
    ap.add_argument("--out", default=str(PRE / "features"))
    a = ap.parse_args()

    src = Path(a.data) / f"{a.set}.jsonl"
    set_sha = hashlib.sha256(open(src, "rb").read()).hexdigest()
    works = [json.loads(l) for l in open(src) if l.strip()]
    out = Path(a.out) / a.set / f"neurons_{a.model}"
    out.mkdir(parents=True, exist_ok=True)
    prog_path = out / "progress.json"
    if a.fresh:
        for f in list(out.glob("*.npy")) + [prog_path]:
            if f.exists():
                f.unlink()
    if prog_path.exists():
        old = json.load(open(prog_path))
        if old.get("set_sha256") != set_sha or old.get("n_works") != len(works):
            raise SystemExit(f"{a.set} changed since the last run (sha256 or number of works); rerun with --fresh")
    model, tokenizer = load(_local(*MODELS[a.model]))
    hf = tokenizer._tokenizer
    layers = model.model.layers
    n_layers = len(layers)
    hidden = layers[0].mlp.gate_proj.weight.shape[0]
    d_model = layers[0].mlp.down_proj.weight.shape[0]
    N = n_layers * hidden
    sink = {"pos": [None] * n_layers, "neg": [None] * n_layers, "gather": None, "weights": None, "hidden": {}}
    for i, layer in enumerate(layers):
        layer.mlp = Recorder(layer.mlp, sink, i)
    taps = sorted({max(0, int(round(f * n_layers)) - 1) for f in HIDDEN_FRACTIONS}) if a.hidden else []
    for i in taps:
        model.model.layers[i] = LayerTap(model.model.layers[i], sink, i)

    n_sent = sum(len(w["sentences"]) for w in works)
    units = np.load(a.units).astype(np.int64) if a.units else np.zeros(0, np.int64)
    np.save(out / "units.npy", units)
    mode = "r+" if (out / "agg.npy").exists() else "w+"
    agg = np.lib.format.open_memmap(out / "agg.npy", mode=mode, dtype=np.float16, shape=(len(works), 2 * N))
    sent = np.lib.format.open_memmap(out / "sent.npy", mode=mode if (out / "sent.npy").exists() else "w+",
                                     dtype=np.float16, shape=(n_sent, len(units)))
    full = (np.lib.format.open_memmap(out / "full.npy", mode=mode if (out / "full.npy").exists() else "w+",
                                      dtype=np.float16, shape=(n_sent, 2 * N)) if a.full else None)
    hid = (np.lib.format.open_memmap(out / "hidden.npy", mode=mode if (out / "hidden.npy").exists() else "w+",
                                     dtype=np.float16, shape=(n_sent, len(taps), d_model)) if taps else None)
    prog = json.load(open(prog_path)) if prog_path.exists() else {"done": [], "seconds": {}, "tokens": {}, "batches": []}
    prog["set_sha256"], prog["n_works"] = set_sha, len(works)
    done = set(prog["done"])
    units_mx = mx.array(units) if len(units) else None
    mx.set_cache_limit(2 * 1024 ** 3)
    row = 0
    sent_work = []
    t_all = time.time()
    for wi, w in enumerate(works):
        k = len(w["sentences"])
        sent_work += [wi] * k
        if w["id"] in done:
            row += k
            continue
        t0 = time.time()
        ctx = " ".join(w.get("context_sentences") or [])
        pre = " ".join(x for x in ((PREFIX if a.prefix else ""), ctx) if x) or None
        ids, spans = sentence_spans(hf, w["sentences"], pre)
        longest = max(e - s for s, e in spans)
        sink["gather"] = mx.array([[min(s + j, e - 1) for j in range(longest)] for s, e in spans])
        sink["weights"] = mx.array([[[1.0] if s + j < e else [0.0] for j in range(longest)] for s, e in spans])
        model.model(mx.array([ids]))
        V = mx.concatenate([mx.concatenate(sink["pos"], axis=1), -mx.concatenate(sink["neg"], axis=1)], axis=1).astype(mx.float32)
        kk = min(TOPK, k)
        top = mx.sort(V, axis=0)[-kk:].mean(axis=0)
        outs = [top]
        if units_mx is not None:
            outs.append(V[:, units_mx])
        mx.eval(*outs)
        agg[wi] = np.array(top.astype(mx.float16))
        if units_mx is not None:
            sent[row:row + k] = np.array(outs[1].astype(mx.float16))
        if full is not None:
            full[row:row + k] = np.array(V.astype(mx.float16))
        if hid is not None:
            H = mx.stack([sink["hidden"][i] for i in taps], axis=1).astype(mx.float16)
            mx.eval(H)
            hid[row:row + k] = np.array(H)
        del V, top, outs
        mx.clear_cache()
        row += k
        dt = time.time() - t0
        prog["done"].append(w["id"])
        prog["seconds"][w["id"]] = dt
        prog["tokens"][w["id"]] = len(ids)
        if wi % 20 == 0 or wi == len(works) - 1:
            agg.flush()
            sent.flush()
            if full is not None:
                full.flush()
            if hid is not None:
                hid.flush()
            prog.setdefault("batches", []).append({"work_index": wi, "wall": time.time() - t_all, "loadavg": list(os.getloadavg())})
            json.dump(prog, open(prog_path, "w"))
            print(f"  {wi + 1}/{len(works)} {w['id'][:40]:40s} {len(ids):5d} tok {dt:5.2f}s", flush=True)
    agg.flush()
    sent.flush()
    json.dump(prog, open(prog_path, "w"))
    seconds = float(sum(prog["seconds"].values()))
    peak = float(mx.get_peak_memory() / 1e9)
    nbytes = sum(f.stat().st_size for f in out.glob("*.npy"))
    meta = {"set": a.set, "set_sha256": set_sha, "model": a.model, "repo": MODELS[a.model], "n_layers": n_layers, "hidden": hidden, "d_model": d_model,
            "n_units": 2 * N, "work_ids": [w["id"] for w in works], "sent_work": sent_work, "n_sentences": n_sent,
            "units_file": a.units, "full": bool(a.full), "hidden_layers": taps, "prefix": PREFIX if a.prefix else None,
            "topk": TOPK, "seconds": seconds, "wall_seconds_this_run": time.time() - t_all, "tokens": int(sum(prog["tokens"].values())),
            "peak_gb": peak, "bytes": int(nbytes), "loadavg_end": list(os.getloadavg())}
    json.dump(meta, open(out / "meta.json", "w"))
    tag = "neurons+hidden" if taps else "neurons"
    json.dump({"seconds": seconds, "peak_gb": peak, "bytes": int(nbytes), "calls": len(works), "tokens": meta["tokens"],
               "sentences": n_sent, "with_hidden": bool(taps), "loadavg_end": meta["loadavg_end"]},
              open(Path(a.out) / a.set / f"cost_{tag.replace('+', '_')}_{a.model}.json", "w"))
    print(f"done: {len(works)} works, {n_sent} sentences, {seconds:.0f}s, peak {peak:.1f} GB -> {out}")


if __name__ == "__main__":
    main()
