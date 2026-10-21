"""Concept-first mapping of the 31 named memes to Qwen3-8B neurons, the way Cammarata's essay works.

The paper went label-first: it picked units by presence across Batman works and read a label off each unit's eight
strongest sentences, which the audit (cammarata_audit.py) found to be mostly names. This experiment goes the other way:

  1. search    For each meme, blind agents wrote name-free minimal pairs (a sentence with the concept, the same sentence
               without it). Every one of the 884,736 gate-open units is scored on the dev pairs, the best is kept, and it is
               scored again on held-out test pairs. The paper's own unit for the meme is scored on the same pairs.
  2. corpus    The chosen units are read on the paper's corpus (each work as one document): gate-open top sentences on both
               poles, peak tokens, depth, echo share, and the paper's presence rule (mean of the top 3 sentences beats all
               but one of the 69 ordinary films), so the new unit's presence grid can be compared with the paper's unit's.
  3. explain   An agent explains each chosen unit from its corpus firings alone, never seeing the meme label (Agent 1);
               a second agent, seeing only that explanation, writes a blind maximum, five new examples and five near misses
               (Agent 2); the probe pass scores them. "Explanations that could lose."
  4. steer     The unit's channel is boosted during generation from a neutral prompt; a blind judge says which concept, if
               any, the text expresses.

Usage, from study/ with the project venv:
  cammarata_mapping.py scan    WORK PAIRS           standalone passes over every pair sentence; WORK/scan.npy [n_sent, 2N] float16
  cammarata_mapping.py select  WORK PAIRS           per meme, best unit on dev pairs, scored on test; WORK/candidates.json
  cammarata_mapping.py corpus  WORK                 in-context corpus pass for the candidate and paper units; WORK/corpus_units.json
  cammarata_mapping.py probe   WORK PROBES OUT      standalone pass for the chosen units over PROBES (blind max, examples, near misses)
  cammarata_mapping.py steer   WORK OUT             steered generations for the chosen units
  cammarata_mapping.py report  WORK                 writes cammarata-mapping.md and cammarata-mapping.json

A unit is one pole of one SwiGLU channel. All activations here are GATE-OPEN and pole-signed: for the positive pole the
largest h = silu(gate) * up over tokens with gate > 0 and h > 0; for the negative pole the largest -h over tokens with
gate > 0 and h < 0; zero if no token qualifies. The echo (gate < 0) never counts. The first token of a standalone sentence
is excluded, as in wording_test.py.
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
TAG = "qwen3-8b"
HIDDEN, N_LAYERS = 12288, 36
N = HIDDEN * N_LAYERS
TOP_K = 8
TOP_CANDIDATES = 5
MIN_DEV_WINS = 9          # of 10 dev pairs
MIN_MEAN_POS = 0.5        # raw gate-open activation on dev positives; weak activations do not count (Cammarata)


def labels():
    return json.load(open(STUDY / "results" / "labels.json"))


def unit_name(u):
    col = u if u < N else u - N
    return f"L{col // HIDDEN}·{col % HIDDEN}{'+' if u < N else '-'}"


def unit_layer(u):
    return (u % N) // HIDDEN


# ----------------------------------------------------------------------------------------------- model

def load_model():
    from mlx_lm import load
    from extract import MODELS
    import mlx.core as mx
    model, tokenizer = load(MODELS[TAG])
    mx.set_cache_limit(2 * 1024 ** 3)
    return model, tokenizer._tokenizer, mx


class _Base:
    """Factory for MLP wrappers; built inside functions so mlx is imported lazily."""


def make_scan_recorder(mx, nn):
    class ScanRecorder(nn.Module):
        """Per layer, keeps the gate-open pole-signed maximum over tokens (first token excluded) for every channel."""

        def __init__(self, mlp, sink, idx):
            super().__init__()
            self.mlp, self._sink, self._idx = mlp, sink, idx

        def __call__(self, x):
            g = self.mlp.gate_proj(x)
            u = self.mlp.up_proj(x)
            h = nn.silu(g) * u
            gs, hs = g[0, 1:], h[0, 1:]
            open_ = gs > 0
            pos = mx.where(open_ & (hs > 0), hs, 0).max(axis=0)
            neg = mx.where(open_ & (hs < 0), -hs, 0).max(axis=0)
            self._sink[self._idx] = (pos, neg)
            return self.mlp.down_proj(h)
    return ScanRecorder


def make_channel_recorder(mx, nn):
    class ChannelRecorder(nn.Module):
        """Keeps gate, value and product at every token for a list of channels."""

        def __init__(self, mlp, sink, idx, cols):
            super().__init__()
            self.mlp, self._sink, self._idx = mlp, sink, idx
            self._cols = mx.array(cols)

        def __call__(self, x):
            g = self.mlp.gate_proj(x)
            u = self.mlp.up_proj(x)
            h = nn.silu(g) * u
            self._sink[self._idx] = (g[0][:, self._cols], u[0][:, self._cols], h[0][:, self._cols])
            return self.mlp.down_proj(h)
    return ChannelRecorder


def make_steerer(mx, nn):
    class Steerer(nn.Module):
        """Adds a constant to one channel's product before the down projection, at every token."""

        def __init__(self, mlp, col, alpha):
            super().__init__()
            self.mlp, self._col, self._alpha = mlp, col, alpha

        def __call__(self, x):
            h = nn.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
            onehot = mx.zeros((h.shape[-1],), dtype=h.dtype)
            onehot[self._col] = 1.0
            h = h + self._alpha * onehot
            return self.mlp.down_proj(h)
    return Steerer


def gate_open_extremes(g, u, h, s, e):
    """For tokens [s, e) of one channel: (pos act, pos tok, neg act, neg tok) with gate open; 0 and -1 if none."""
    gs, hs = g[s:e], h[s:e]
    open_ = gs > 0
    pos = np.where(open_ & (hs > 0), hs, 0.0)
    neg = np.where(open_ & (hs < 0), -hs, 0.0)
    ip, ineg = int(np.argmax(pos)), int(np.argmax(neg))
    return (float(pos[ip]), s + ip if pos[ip] > 0 else -1, float(neg[ineg]), s + ineg if neg[ineg] > 0 else -1)


# ----------------------------------------------------------------------------------------------- 1. scan + select

def scan(work_dir, pairs_path):
    import mlx.nn as nn
    model, hf_tok, mx = load_model()
    Rec = make_scan_recorder(mx, nn)
    sink = {}
    for i, layer in enumerate(model.model.layers):
        layer.mlp = Rec(layer.mlp, sink, i)
    pairs = json.load(open(pairs_path))
    sents = pairs["sentences"]                       # [{"id","split","pair","side","text"}]
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(work_dir / "scan.npy", mode="w+", dtype=np.float16, shape=(len(sents), 2 * N))
    t0 = time.time()
    for r, s in enumerate(sents):
        ids = hf_tok(s["text"], add_special_tokens=True)["input_ids"]
        model.model(mx.array([ids]))
        pos = mx.concatenate([sink[i][0] for i in range(N_LAYERS)])
        neg = mx.concatenate([sink[i][1] for i in range(N_LAYERS)])
        mx.eval(pos, neg)
        arr[r, :N] = np.array(pos.astype(mx.float16))
        arr[r, N:] = np.array(neg.astype(mx.float16))
        mx.clear_cache()
        if r % 100 == 0:
            print(f"  {r}/{len(sents)}  {time.time() - t0:.0f}s", flush=True)
    arr.flush()
    json.dump({"n_sentences": len(sents), "n_units": 2 * N}, open(work_dir / "scan_meta.json", "w"))
    print(f"scan done: {len(sents)} sentences in {time.time() - t0:.0f}s")


def select(work_dir, pairs_path):
    work_dir = Path(work_dir)
    pairs = json.load(open(pairs_path))
    sents = pairs["sentences"]
    A = np.load(work_dir / "scan.npy", mmap_mode="r")
    labs = labels()
    print("loading scan into memory as float32 by halves...", flush=True)
    sd_all = np.zeros(2 * N, dtype=np.float32)
    for c in range(0, 2 * N, 221184):
        block = np.asarray(A[:, c:c + 221184], dtype=np.float32)
        sd_all[c:c + 221184] = block.std(axis=0)
    sd_all += 1e-3
    by = defaultdict(lambda: defaultdict(dict))
    for r, s in enumerate(sents):
        if s["id"] is not None:
            by[str(s["id"])][s["split"]].setdefault(s["pair"], {})[s["side"]] = r
    out = {"memes": []}
    for uid, lab in labs.items():
        dev = sorted(by[uid]["dev"].items())
        test = sorted(by[uid]["test"].items())
        pr = np.array([p[1]["pos"] for p in dev]); nr = np.array([p[1]["neg"] for p in dev])
        tp = np.array([p[1]["pos"] for p in test]); tn = np.array([p[1]["neg"] for p in test])
        Pd, Nd = np.asarray(A[pr], dtype=np.float32), np.asarray(A[nr], dtype=np.float32)
        Pt, Nt = np.asarray(A[tp], dtype=np.float32), np.asarray(A[tn], dtype=np.float32)
        others = np.array([r for r, s in enumerate(sents) if str(s.get("id")) != uid])
        wins = (Pd > Nd).sum(0)
        mean_pos = Pd.mean(0)
        score = (Pd - Nd).mean(0) / sd_all
        ok = (wins >= MIN_DEV_WINS) & (mean_pos >= MIN_MEAN_POS)
        score_ok = np.where(ok, score, -np.inf)
        cand = np.argsort(-score_ok)[:TOP_CANDIDATES]
        cand = [int(c) for c in cand if np.isfinite(score_ok[c])]

        def describe(u):
            bg = np.asarray(A[others, u], dtype=np.float32)
            return {"unit": int(u), "name": unit_name(u), "layer": unit_layer(u),
                    "dev_wins": int(wins[u]), "dev_mean_pos": round(float(mean_pos[u]), 3), "dev_mean_neg": round(float(Nd[:, u].mean()), 3),
                    "dev_score": round(float(score[u]), 3),
                    "test_wins": int((Pt[:, u] > Nt[:, u]).sum()), "test_n": int(len(tp)),
                    "test_auc": round(float((Pt[:, u][:, None] > Nt[:, u][None, :]).mean() + 0.5 * (Pt[:, u][:, None] == Nt[:, u][None, :]).mean()), 3),
                    "test_mean_pos": round(float(Pt[:, u].mean()), 3), "test_mean_neg": round(float(Nt[:, u].mean()), 3),
                    "other_memes_p99": round(float(np.percentile(bg, 99)), 3), "other_memes_max": round(float(bg.max()), 3),
                    "test_pos_above_other_p99": int((Pt[:, u] > np.percentile(bg, 99)).sum())}

        rec = {"unit_paper": int(uid), "label": lab["label"], "kind": lab["kind"], "n_dev": len(dev), "n_test": len(test),
               "n_units_passing_dev": int(ok.sum()),
               "candidates": [describe(u) for u in cand],
               "paper_unit": describe(int(uid))}
        rec["chosen"] = rec["candidates"][0] if rec["candidates"] else None
        out["memes"].append(rec)
        c = rec["chosen"]
        print(f"{lab['label'][:30]:30s} pass={ok.sum():6d}  chosen {c['name'] if c else '-':12s} dev {c['dev_wins'] if c else 0}/10 "
              f"test {c['test_wins'] if c else 0}/{len(test)}  | paper {unit_name(int(uid))} dev {rec['paper_unit']['dev_wins']}/10 test {rec['paper_unit']['test_wins']}/{len(test)}", flush=True)
    json.dump(out, open(work_dir / "candidates.json", "w"), indent=1, ensure_ascii=False)


# ----------------------------------------------------------------------------------------------- 2. corpus

def corpus(work_dir):
    import mlx.nn as nn
    from extract import sentence_spans
    work_dir = Path(work_dir)
    cands = json.load(open(work_dir / "candidates.json"))
    units = []
    for m in cands["memes"]:
        units += [c["unit"] for c in m["candidates"]] + [m["unit_paper"]]
    units = sorted(set(units))
    cols_by_layer = defaultdict(list)
    for u in units:
        col = u % N
        if col % HIDDEN not in cols_by_layer[col // HIDDEN]:
            cols_by_layer[col // HIDDEN].append(col % HIDDEN)
    model, hf_tok, mx = load_model()
    Rec = make_channel_recorder(mx, nn)
    sink = {}
    for i, layer in enumerate(model.model.layers):
        if i in cols_by_layer:
            layer.mlp = Rec(layer.mlp, sink, i, cols_by_layer[i])
    chans = sorted({u % N for u in units})
    where = {c: (c // HIDDEN, cols_by_layer[c // HIDDEN].index(c % HIDDEN)) for c in chans}
    works = [json.loads(l) for l in open(STUDY / "data" / "corpus.jsonl")]
    n_sent = sum(len(w["sentences"]) for w in works)
    acts = np.zeros((n_sent, len(chans), 2), dtype=np.float32)       # gate-open pos, gate-open neg
    toks = [[None] * len(chans) for _ in range(n_sent)]
    meta, row, t0 = [], 0, time.time()
    for w in works:
        ids, spans = sentence_spans(hf_tok, w["sentences"])
        model.model(mx.array([ids]))
        per_layer = {i: tuple(np.array(a.astype(mx.float32)) for a in arrs) for i, arrs in sink.items()}
        tstr = [hf_tok.decode([t]) for t in ids]
        for (s, e), text in zip(spans, w["sentences"]):
            for j, c in enumerate(chans):
                li, k = where[c]
                g, u, h = (a[:, k] for a in per_layer[li])
                p, ip, q, iq = gate_open_extremes(g, u, h, s, e)
                acts[row, j] = (p, q)
                toks[row][j] = (tstr[ip] if ip >= 0 else None, tstr[iq] if iq >= 0 else None)
            meta.append({"work": w["id"], "label": w["label"], "group": w["group"], "text": text})
            row += 1
        mx.clear_cache()
    print(f"corpus pass: {row} sentences in {time.time() - t0:.0f}s", flush=True)
    sent_work = np.array([m["work"] for m in meta]); group = {w["id"]: w["group"] for w in works}
    bat = [w["id"] for w in works if w["group"] == "batman"]; base = [w["id"] for w in works if w["group"] == "baseline"]

    def unit_summary(u):
        j = chans.index(u % N); pole = 0 if u < N else 1
        own = acts[:, j, pole]; opp = acts[:, j, 1 - pole]
        order = np.argsort(-own); oorder = np.argsort(-opp)
        top = lambda i, a, t: {"work": meta[i]["label"], "group": meta[i]["group"], "act": round(float(a[i]), 3),
                               "peak_token": toks[i][j][t], "text": meta[i]["text"]}
        scores = {}
        for wid in range(len(works)):
            rows = np.where(sent_work == wid)[0]
            k = min(3, len(rows))
            scores[wid] = float(np.sort(own[rows])[-k:].mean())
        base_sorted = sorted(scores[b] for b in base)
        thr = base_sorted[-2]
        present = [wid for wid in range(len(works)) if scores[wid] > thr]
        med = float(np.median(base_sorted))                 # a lenient bar: above the median ordinary film
        above_median = [wid for wid in range(len(works)) if scores[wid] > med]
        return {"unit": u, "name": unit_name(u), "layer": unit_layer(u), "pole": "+" if u < N else "-",
                "corpus_max": round(float(own[order[0]]), 3), "corpus_p99": round(float(np.percentile(own, 99)), 3),
                "corpus_min_top8": round(float(own[order[TOP_K - 1]]), 3),
                "top8": [top(i, own, pole) for i in order[:TOP_K]],
                "peak_tokens_top50": Counter(toks[i][j][pole] for i in order[:50]).most_common(6),
                "negative_pole": {"name": unit_name(u + N if u < N else u - N), "max": round(float(opp[oorder[0]]), 3),
                                  "top8": [top(i, opp, 1 - pole) for i in oorder[:TOP_K]]},
                "work_scores": {works[w]["label"]: round(scores[w], 3) for w in range(len(works))},
                "presence": {"threshold": round(thr, 3), "baseline_median": round(med, 3),
                             "n_batman_above_median": sum(1 for p in above_median if group[p] == "batman"),
                             "n_baseline_above_median": sum(1 for p in above_median if group[p] == "baseline"),
                             "n_batman": sum(1 for p in present if group[p] == "batman"),
                             "n_baseline": sum(1 for p in present if group[p] == "baseline"),
                             "n_cousin": sum(1 for p in present if group[p] == "cousin"),
                             "present_batman": [works[p]["label"] for p in present if group[p] == "batman"],
                             "present_cousin": [works[p]["label"] for p in present if group[p] == "cousin"],
                             "present_baseline": [works[p]["label"] for p in present if group[p] == "baseline"]}}

    out = {str(u): unit_summary(u) for u in units}
    json.dump(out, open(work_dir / "corpus_units.json", "w"), indent=1, ensure_ascii=False)
    print(f"-> {work_dir / 'corpus_units.json'} ({len(units)} units)")


# ----------------------------------------------------------------------------------------------- 3. probes

def probe(work_dir, probes_path, out_name):
    import mlx.nn as nn
    work_dir = Path(work_dir)
    probes = json.load(open(probes_path))                # [{"unit", "kind", "text", ...}]
    units = sorted({p["unit"] for p in probes})
    cols_by_layer = defaultdict(list)
    for u in units:
        col = u % N
        if col % HIDDEN not in cols_by_layer[col // HIDDEN]:
            cols_by_layer[col // HIDDEN].append(col % HIDDEN)
    model, hf_tok, mx = load_model()
    Rec = make_channel_recorder(mx, nn)
    sink = {}
    for i, layer in enumerate(model.model.layers):
        if i in cols_by_layer:
            layer.mlp = Rec(layer.mlp, sink, i, cols_by_layer[i])
    out = []
    for p in probes:
        ids = hf_tok(p["text"], add_special_tokens=True)["input_ids"]
        model.model(mx.array([ids]))
        per_layer = {i: tuple(np.array(a.astype(mx.float32)) for a in arrs) for i, arrs in sink.items()}
        tstr = [hf_tok.decode([t]) for t in ids]
        rec = dict(p)
        rec["units"] = {}
        for u in units:
            col = u % N; li = col // HIDDEN; k = cols_by_layer[li].index(col % HIDDEN)
            g, uu, h = (a[:, k] for a in per_layer[li])
            pp, ip, q, iq = gate_open_extremes(g, uu, h, 1, len(ids))
            rec["units"][str(u)] = {"act": pp if u < N else q, "peak_token": (tstr[ip] if ip >= 0 else None) if u < N else (tstr[iq] if iq >= 0 else None)}
        out.append(rec)
        mx.clear_cache()
    json.dump(out, open(work_dir / out_name, "w"), indent=1, ensure_ascii=False)
    print(f"probe pass: {len(out)} sentences, {len(units)} units -> {work_dir / out_name}")


# ----------------------------------------------------------------------------------------------- 4. steering

STEER_PROMPTS = ["Plot summary: The film opens as", "Plot: In the first act,", "The story begins when"]


def steer(work_dir, out_name, units_path=None):
    import mlx.nn as nn
    from mlx_lm import generate
    work_dir = Path(work_dir)
    spec = json.load(open(units_path or work_dir / "steer_units.json"))   # [{"unit", "alpha"}]
    model, hf_tok, mx = load_model()
    from mlx_lm import load as _load
    _, tokenizer = _load(__import__("extract").MODELS[TAG])
    Steerer = make_steerer(mx, nn)
    layers = model.model.layers
    originals = [l.mlp for l in layers]
    out = []
    for s in spec:
        u = s["unit"]; col = u % N; li = col // HIDDEN; c = col % HIDDEN
        sign = 1.0 if u < N else -1.0
        for alpha in s["alphas"]:
            layers[li].mlp = Steerer(originals[li], c, sign * alpha)
            for prompt in STEER_PROMPTS:
                text = generate(model, tokenizer, prompt=prompt, max_tokens=48, verbose=False)
                out.append({"unit": u, "name": unit_name(u), "alpha": alpha, "prompt": prompt, "text": text})
            layers[li].mlp = originals[li]
            mx.clear_cache()
    for prompt in STEER_PROMPTS:
        out.append({"unit": None, "name": "unsteered", "alpha": 0, "prompt": prompt,
                    "text": generate(model, tokenizer, prompt=prompt, max_tokens=48, verbose=False)})
    json.dump(out, open(work_dir / out_name, "w"), indent=1, ensure_ascii=False)
    print(f"steer: {len(out)} generations -> {work_dir / out_name}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "scan":
        scan(sys.argv[2], sys.argv[3])
    elif cmd == "select":
        select(sys.argv[2], sys.argv[3])
    elif cmd == "corpus":
        corpus(sys.argv[2])
    elif cmd == "probe":
        probe(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "steer":
        steer(sys.argv[2], sys.argv[3], *sys.argv[4:5])
    elif cmd == "report":
        from cammarata_mapping_report import write_report
        write_report(sys.argv[2])
    else:
        raise SystemExit(__doc__)
