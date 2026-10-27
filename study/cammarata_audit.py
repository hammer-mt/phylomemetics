"""Cammarata-style audit of the 31 named Qwen3-8B memes.

Five checks per named meme, after "Neural networks are already clean" (Cammarata, Goodfire):
  1. depth          the neuron's layer; units in the first four layers are flagged as likely token detectors
  2. sign regime    gate and value signs at the peak token of the unit's 50 strongest corpus firings, and the
                    share that are the double-negative "echo" (gate < 0 and value < 0 giving a positive output;
                    gate < 0 and value > 0 for a negative pole). If the echo dominates, the unit is relabelled
                    from its gate-open firings only.
  3. negative pole  the eight strongest sentences of the neuron's opposite pole
  4. blind maximum  sentences written from the label alone (never from the neuron), scored against the unit's
                    corpus maximum; three attempts, the best is kept
  5. near miss      a sentence with the label's words but not its idea

Usage, from study/ with the project venv (../.venv/bin/python):
  cammarata_audit.py corpus WORK            one pass over the corpus, each work as one document, as extract.py does;
                                            writes WORK/corpus.npz and WORK/corpus_meta.json
  cammarata_audit.py probe  WORK PROBES [OUT]   standalone passes over the probe sentences in PROBES (json);
                                            writes WORK/OUT (default probe_acts.json). Run it once on the blind,
                                            near-miss and neutral sentences and once on each unit's eight strongest
                                            corpus sentences (kind "corpus_top8", written to corpus_top8_acts.json),
                                            so the blind maximum can be compared with corpus sentences read the same way
  cammarata_audit.py report WORK            writes cammarata-audit.md and cammarata-audit.json from WORK/*
                                            (and WORK/notes.json, hand-written readings, if present)

A unit is one pole of one SwiGLU channel (extract.py). Activations here are pole-signed: for a negative-pole unit
the reported activation is minus the channel's minimum, so bigger always means stronger. Standalone probe
sentences exclude their first token, as wording_test.py does; corpus sentences are read in the context of their
whole work, as extract.py does.
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
TAG = "qwen3-8b"
EARLY_LAYERS = 4          # layers 0..3 are flagged as likely token or spelling detectors
TOP_N = 50                # firings inspected for the sign regime
TOP_K = 8                 # sentences listed per pole
ECHO_DOMINATES = 0.5      # echo share of the top 50 above which the unit is relabelled from gate-open firings
ECHO_SMALL = 0.25         # echo share at or below which a label counts as clean on check 2
NEAR_MISS_RATIO = 0.5     # a near miss "stays low" below this fraction of the best blind maximum and below the weakest of
                          # the corpus top 8 the label was read from (percentiles mislead for sparse units: p97 can be a
                          # hundredth of the maximum)


def named_units():
    """The 31 named memes, in labels.json order, with their layer, neuron and pole."""
    labels = json.load(open(STUDY / "results" / "labels.json"))
    hidden, n_layers = 12288, 36
    n = hidden * n_layers
    units = []
    for k, v in labels.items():
        u = int(k)
        col = u if u < n else u - n
        units.append({"unit": u, "col": col, "layer": col // hidden, "neuron": col % hidden,
                      "pole": "+" if u < n else "-", "label": v["label"], "kind": v["kind"], "note": v["note"]})
    return units


def uname(u):
    return f"L{u['layer']}·{u['neuron']}{u['pole']}"


# ----------------------------------------------------------------------------------------------- model passes

def _model_setup(units):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from extract import MODELS

    class Recorder(nn.Module):
        """Wraps an MLP; keeps gate, value and product for the audited channels at every token."""

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

    model, tokenizer = load(MODELS[TAG])
    cols_by_layer = defaultdict(list)
    for u in units:
        if u["neuron"] not in cols_by_layer[u["layer"]]:
            cols_by_layer[u["layer"]].append(u["neuron"])
    sink = {}
    for i, layer in enumerate(model.model.layers):
        if i in cols_by_layer:
            layer.mlp = Recorder(layer.mlp, sink, i, cols_by_layer[i])
    where = [(u["layer"], cols_by_layer[u["layer"]].index(u["neuron"])) for u in units]
    mx.set_cache_limit(2 * 1024 ** 3)
    return model, tokenizer._tokenizer, sink, where, mx


def _collect(sink, mx):
    """{layer: (gate, up, h) as float32 numpy [T, k]} for one forward pass."""
    out = {}
    for i, arrs in sink.items():
        out[i] = tuple(np.array(a.astype(mx.float32)) for a in arrs)
    return out


def _extremes(per_layer, where, s, e):
    """For tokens [s, e): per unit, (max h, gate, up, tok, min h, gate, up, tok)."""
    rows = []
    for layer, k in where:
        g, u, h = (a[s:e, k] for a in per_layer[layer])
        ia, ib = int(np.argmax(h)), int(np.argmin(h))
        rows.append((h[ia], g[ia], u[ia], s + ia, h[ib], g[ib], u[ib], s + ib))
    return rows


def corpus_pass(work_dir):
    from extract import sentence_spans
    units = named_units()
    model, hf_tok, sink, where, mx = _model_setup(units)
    works = [json.loads(l) for l in open(STUDY / "data" / "corpus.jsonl")]
    n_sent = sum(len(w["sentences"]) for w in works)
    arr = np.zeros((n_sent, len(units), 8), dtype=np.float32)
    meta_sent, row, t0 = [], 0, time.time()
    for w in works:
        ids, spans = sentence_spans(hf_tok, w["sentences"])
        model.model(mx.array([ids]))
        per_layer = _collect(sink, mx)
        for (s, e), text in zip(spans, w["sentences"]):
            arr[row] = _extremes(per_layer, where, s, e)
            meta_sent.append({"work": w["id"], "label": w["label"], "group": w["group"], "text": text})
            row += 1
        # token strings at the extreme positions
        for j in range(len(units)):
            for col in (3, 7):
                pass
        toks = [hf_tok.decode([t]) for t in ids]
        for r in range(row - len(spans), row):
            meta_sent[r]["tok"] = [toks[int(arr[r, j, 3])] for j in range(len(units))]
            meta_sent[r]["tok_min"] = [toks[int(arr[r, j, 7])] for j in range(len(units))]
        mx.clear_cache()
        print(f"  {w['label'][:30]:30s} {len(ids):5d} tok  {time.time() - t0:5.0f}s", flush=True)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    np.savez(work_dir / "corpus.npz", acts=arr)
    json.dump({"units": units, "sentences": meta_sent}, open(work_dir / "corpus_meta.json", "w"))
    print(f"corpus pass done in {time.time() - t0:.0f}s -> {work_dir}")


def probe_pass(work_dir, probes_path, out_name="probe_acts.json"):
    units = named_units()
    probes = json.load(open(probes_path))
    model, hf_tok, sink, where, mx = _model_setup(units)
    out = []
    for p in probes:
        ids = hf_tok(p["sentence"], add_special_tokens=True)["input_ids"]
        model.model(mx.array([ids]))
        per_layer = _collect(sink, mx)
        rows = _extremes(per_layer, where, 1, len(ids))          # first token excluded, as in wording_test.py
        toks = [hf_tok.decode([t]) for t in ids]
        rec = dict(p)
        rec["units"] = {}
        for u, r in zip(units, rows):
            rec["units"][str(u["unit"])] = {
                "max_h": float(r[0]), "max_gate": float(r[1]), "max_up": float(r[2]), "max_tok": toks[int(r[3])],
                "min_h": float(r[4]), "min_gate": float(r[5]), "min_up": float(r[6]), "min_tok": toks[int(r[7])]}
        out.append(rec)
        mx.clear_cache()
    json.dump(out, open(Path(work_dir) / out_name, "w"), indent=1)
    print(f"probe pass: {len(out)} sentences -> {Path(work_dir) / out_name}")


# ----------------------------------------------------------------------------------------------- analysis

def regime(pole, gate, up):
    """'open' (gate positive), 'echo' (the double negative), or 'other' (a sign pattern that gives the wrong sign)."""
    if pole == "+":
        if gate > 0 and up > 0:
            return "open"
        if gate < 0 and up < 0:
            return "echo"
    else:
        if gate > 0 and up < 0:
            return "open"
        if gate < 0 and up > 0:
            return "echo"
    return "other"


def pole_signed(unit, max_h, min_h):
    return max_h if unit["pole"] == "+" else -min_h


def pct(col, v):
    return float((col < v).mean() * 100)


def build(work_dir):
    work_dir = Path(work_dir)
    acts = np.load(work_dir / "corpus.npz")["acts"]              # [n_sent, 31, 8]
    meta = json.load(open(work_dir / "corpus_meta.json"))
    units, sents = meta["units"], meta["sentences"]
    probes = json.load(open(work_dir / "probe_acts.json")) if (work_dir / "probe_acts.json").exists() else []
    if (work_dir / "corpus_top8_acts.json").exists():
        probes += json.load(open(work_dir / "corpus_top8_acts.json"))
    notes = json.load(open(work_dir / "notes.json")) if (work_dir / "notes.json").exists() else {}
    fams = {f["unit"]: f for f in json.load(open(STUDY / "results" / "qwen3-8b" / "memes.json"))["families"]}

    results = []
    for j, u in enumerate(units):
        uid = str(u["unit"])
        a = acts[:, j, :]
        own = pole_signed(u, a[:, 0], a[:, 4])                      # the unit's own activation per sentence
        opp = pole_signed({"pole": "-" if u["pole"] == "+" else "+"}, a[:, 0], a[:, 4])
        own_g, own_u = (a[:, 1], a[:, 2]) if u["pole"] == "+" else (a[:, 5], a[:, 6])
        own_tok = [s["tok"][j] if u["pole"] == "+" else s["tok_min"][j] for s in sents]
        opp_tok = [s["tok_min"][j] if u["pole"] == "+" else s["tok"][j] for s in sents]
        order = np.argsort(-own)
        top = order[:TOP_N]
        regs = [regime(u["pole"], own_g[i], own_u[i]) for i in top]
        n_echo, n_open = regs.count("echo"), regs.count("open")
        echo_share = n_echo / len(top)
        echo_top8 = regs[:TOP_K].count("echo")
        open_idx = [i for i in order if regime(u["pole"], own_g[i], own_u[i]) == "open"][:TOP_K]
        first_echo_rank = next((k + 1 for k, r in enumerate(regs) if r == "echo"), None)
        peak_counts = Counter(own_tok[i] for i in top).most_common(6)
        peak_by_regime = [[t, n, sum(1 for i in top if own_tok[i] == t and regime(u["pole"], own_g[i], own_u[i]) == "open"),
                           sum(1 for i in top if own_tok[i] == t and regime(u["pole"], own_g[i], own_u[i]) == "echo")]
                          for t, n in peak_counts]
        top8_min = float(own[order[TOP_K - 1]])                   # the weakest of the eight sentences the label was read from
        # the paper scores a work by the mean of its three strongest sentences and calls a meme present when that beats
        # all but one of the 69 ordinary films: how often do those three sentences come from the echo regime?
        work_of = np.array([s["work"] for s in sents])
        group_of = {s["work"]: s["group"] for s in sents}
        top3_echo = {}
        for w in np.unique(work_of):
            rows = np.where(work_of == w)[0]
            best3 = rows[np.argsort(-own[rows])[:3]]
            top3_echo[int(w)] = float(np.mean([regime(u["pole"], own_g[i], own_u[i]) == "echo" for i in best3]))
        echo_work = {g: round(float(np.mean([v for w, v in top3_echo.items() if group_of[w] == g])), 3)
                     for g in ("batman", "baseline", "cousin")}
        echo_work_majority = {g: int(sum(v >= 2 / 3 for w, v in top3_echo.items() if group_of[w] == g))
                              for g in ("batman", "baseline")}
        corpus_max = float(own[order[0]])
        paper_top = [s["text"] for s in fams[u["unit"]]["top_sentences"]]
        agree = sum(sents[i]["text"] in paper_top for i in order[:TOP_K])

        def sent(i, act=None, tok=None):
            d = {"work": sents[i]["label"], "group": sents[i]["group"], "text": sents[i]["text"]}
            if act is not None:
                d["act"] = round(float(act), 2)
            if tok is not None:
                d["peak_token"] = tok
            return d

        own_top8 = [dict(sent(i, own[i], own_tok[i]), regime=regime(u["pole"], own_g[i], own_u[i]),
                         gate=round(float(own_g[i]), 2), value=round(float(own_u[i]), 2)) for i in order[:TOP_K]]
        opp_order = np.argsort(-opp)
        opp_top8 = [sent(i, opp[i], opp_tok[i]) for i in opp_order[:TOP_K]]

        # probes
        blind, near, neutral, standalone = [], None, [], []
        for p in probes:
            if str(p.get("id")) != uid and p["kind"] != "neutral":
                continue
            r = p["units"][uid]
            v = pole_signed(u, r["max_h"], r["min_h"])
            g, up = (r["max_gate"], r["max_up"]) if u["pole"] == "+" else (r["min_gate"], r["min_up"])
            tok = r["max_tok"] if u["pole"] == "+" else r["min_tok"]
            rec = {"kind": p["kind"], "sentence": p["sentence"], "act": round(v, 3), "pct": round(pct(own, v), 1),
                   "ratio_to_max": round(v / corpus_max, 3) if corpus_max else None, "peak_token": tok,
                   "regime": regime(u["pole"], g, up)}
            if p["kind"].startswith("blind"):
                blind.append(rec)
            elif p["kind"] == "near_miss":
                rec["words_used"] = p.get("words_used")
                near = rec
            elif p["kind"] == "neutral":
                neutral.append(rec)
            elif p["kind"] == "corpus_top8":
                rec["work"] = p.get("work")
                standalone.append(rec)
        best = max(blind, key=lambda r: r["act"]) if blind else None
        neutral_max = max((r["act"] for r in neutral), default=None)
        standalone_max = max((r["act"] for r in standalone), default=None)   # the corpus top 8, read on their own
        standalone_min = min((r["act"] for r in standalone), default=None)

        checks = {
            "depth_ok": u["layer"] >= EARLY_LAYERS,
            "echo_small": echo_share <= ECHO_SMALL,
            "blind_beats_max": bool(best and best["act"] >= corpus_max),
            "blind_beats_standalone_max": bool(best and standalone_max is not None and best["act"] >= standalone_max),
            "blind_top1pct": bool(best and best["pct"] >= 99.0),
            "blind_in_top8": bool(best and best["act"] >= top8_min),
            "near_miss_low": bool(near and best and near["act"] < NEAR_MISS_RATIO * best["act"] and near["act"] < top8_min),
        }
        # survives: the blind maximum beats the unit's corpus maximum (in context, as the paper recorded it), the near miss
        # stays low and the echo share is small. Lenient: the blind maximum would at least have made the top 8 the label was
        # read from. The standalone corpus maximum is reported beside these, since the probes are read standalone.
        checks["survives"] = checks["blind_beats_max"] and checks["near_miss_low"] and checks["echo_small"]
        checks["survives_lenient"] = checks["blind_in_top8"] and checks["near_miss_low"] and checks["echo_small"]
        n = notes.get(uid, {})
        results.append({
            "unit": u["unit"], "name": uname(u), "layer": u["layer"], "neuron": u["neuron"], "pole": u["pole"],
            "label": u["label"], "kind": u["kind"], "paper_note": u["note"],
            "n_batman_works": fams[u["unit"]]["n_batman"], "family_size": fams[u["unit"]]["n_members"],
            "depth": {"layer": u["layer"], "early": u["layer"] < EARLY_LAYERS},
            "sign_regime": {"top_n": len(top), "open": n_open, "echo": n_echo, "other": len(top) - n_open - n_echo,
                            "echo_share": round(echo_share, 3), "echo_in_top8": echo_top8,
                            "echo_dominates": echo_share > ECHO_DOMINATES, "first_echo_rank": first_echo_rank,
                            "peak_tokens_top50": peak_by_regime,   # [token, count, gate-open, echo]
                            "work_score_echo_share": echo_work,
                            "works_scored_mostly_by_echo": echo_work_majority,
                            "echo_share_all_firings_above_p99": round(float(np.mean([regime(u["pole"], own_g[i], own_u[i]) == "echo"
                                                                              for i in order[:max(1, len(own) // 100)]])), 3),
                            "gate_open_top8": [sent(i, own[i], own_tok[i]) for i in open_idx],
                            "relabel_from_gate_open": n.get("relabel")},
            "corpus": {"max": round(corpus_max, 3), "max_standalone": round(standalone_max, 3) if standalone_max is not None else None,
                       "min_top8": round(top8_min, 3),
                       "min_top8_standalone": round(standalone_min, 3) if standalone_min is not None else None,
                       "top8_standalone": sorted(standalone, key=lambda r: -r["act"]),
                       "p99": round(float(np.percentile(own, 99)), 3),
                       "p90": round(float(np.percentile(own, 90)), 3), "median": round(float(np.median(own)), 3),
                       "top8_matches_paper": agree, "top8": own_top8},
            "negative_pole": {"name": f"L{u['layer']}·{u['neuron']}{'-' if u['pole'] == '+' else '+'}",
                              "max": round(float(opp[opp_order[0]]), 3), "top8": opp_top8, "reading": n.get("pole")},
            "blind_maximum": {"attempts": blind, "best": best, "neutral_max": neutral_max},
            "near_miss": near,
            "checks": checks,
            "verdict": n.get("verdict"), "new_label": n.get("new_label"),
        })
    return results


def write_report(work_dir):
    res = build(work_dir)
    n = len(res)
    c = lambda k: sum(r["checks"][k] for r in res)
    summary = {
        "n_memes": n, "early_layer": sum(r["depth"]["early"] for r in res),
        "echo_dominates": sum(r["sign_regime"]["echo_dominates"] for r in res),
        "echo_small": c("echo_small"), "blind_beats_max": c("blind_beats_max"), "blind_top1pct": c("blind_top1pct"),
        "blind_in_top8": c("blind_in_top8"),
        "blind_beats_standalone_max": c("blind_beats_standalone_max"),
        "near_miss_low": c("near_miss_low"), "survives": c("survives"), "survives_lenient": c("survives_lenient"),
        "echo_in_any_top8": sum(r["sign_regime"]["echo_in_top8"] > 0 for r in res),
        "work_score_echo_share_mean": {g: round(float(np.mean([r["sign_regime"]["work_score_echo_share"][g] for r in res])), 3)
                                       for g in ("batman", "baseline")},
        "units_where_most_batman_works_scored_by_echo": sum(r["sign_regime"]["works_scored_mostly_by_echo"]["batman"] >= 16 for r in res),
        "units_where_most_films_scored_by_echo": sum(r["sign_regime"]["works_scored_mostly_by_echo"]["baseline"] >= 35 for r in res),
        "relabel": sum(1 for r in res if r.get("new_label") and "keep" not in r["new_label"]),
        "blind_under_20pct_of_max": sum(1 for r in res if r["blind_maximum"]["best"] and r["blind_maximum"]["best"]["ratio_to_max"] < 0.2),
        "near_miss_above_weakest_top8": sum(1 for r in res if r["near_miss"] and r["near_miss"]["act"] > r["corpus"]["min_top8"]),
        "near_miss_beats_best_blind": sum(1 for r in res if r["near_miss"] and r["blind_maximum"]["best"]
                                          and r["near_miss"]["act"] > r["blind_maximum"]["best"]["act"]),
        "thresholds": {"early_layers": EARLY_LAYERS, "echo_small": ECHO_SMALL, "echo_dominates": ECHO_DOMINATES,
                       "near_miss_ratio": NEAR_MISS_RATIO},
    }
    notes = json.load(open(Path(work_dir) / "notes.json")) if (Path(work_dir) / "notes.json").exists() else {}
    out = {"model": TAG, "repo": "mlx-community/Qwen3-8B-4bit", "summary": summary,
           "summary_text": notes.get("_summary"), "memes": res}
    json.dump(out, open(STUDY / "cammarata-audit.json", "w"), indent=1, ensure_ascii=False)
    (STUDY / "cammarata-audit.md").write_text(render_md(out))
    print(json.dumps(summary, indent=1))


def esc(t):
    return t.replace("|", "\\|").replace("\n", " ")


def trim(t, n=170):
    return t if len(t) <= n else t[:n - 1].rstrip() + "…"


def render_md(out):
    s = out["summary"]
    L = []
    L.append("# Cammarata audit of the 31 named Qwen3-8B memes\n")
    L.append("Checks after Nick Cammarata's \"Neural networks are already clean\": depth, SwiGLU sign regime, the negative pole, "
             "a blind maximum written from the label alone, and a near miss. Produced by `cammarata_audit.py`; the data is in "
             "`cammarata-audit.json`. Activations are pole-signed (bigger is stronger) and in the model's raw units. "
             "Corpus sentences were read in the context of their whole work, as in the paper; probe sentences were read on their own, "
             "first token excluded, as in the paper's rewording test. Percentiles are against the unit's responses to the 3,700 corpus sentences.\n")
    L.append("## Summary\n")
    if out.get("summary_text"):
        L.append(out["summary_text"] + "\n")
    L.append("| Count | of 31 |\n| --- | --- |")
    L.append(f"| In the first {s['thresholds']['early_layers']} layers (likely token or spelling detectors) | {s['early_layer']} |")
    L.append(f"| Echo share of the top 50 firings at or below {s['thresholds']['echo_small']:.0%} | {s['echo_small']} |")
    L.append(f"| Echo dominates the top 50 (above {s['thresholds']['echo_dominates']:.0%}), relabelled from gate-open firings | {s['echo_dominates']} |")
    L.append(f"| Best blind maximum beats the unit's corpus maximum (in context, as the paper recorded it) | {s['blind_beats_max']} |")
    L.append(f"| Best blind maximum beats the corpus top 8 read standalone, as the probes are | {s['blind_beats_standalone_max']} |")
    L.append(f"| Best blind maximum would make the corpus top 8 the label was read from | {s['blind_in_top8']} |")
    L.append(f"| Near miss stays low (under half the best blind maximum and under the weakest of the corpus top 8) | {s['near_miss_low']} |")
    L.append(f"| **Survives** (blind beats the corpus maximum, near miss low, echo small) | **{s['survives']}** |")
    L.append(f"| Survives on the lenient bar (blind would make the top 8 instead of beating the maximum) | {s['survives_lenient']} |\n")
    L.append("## What each label should become\n")
    L.append("| # | Paper's label | Unit | Reads as | Suggested label |\n| --- | --- | --- | --- | --- |")
    for i, r in enumerate(out["memes"], 1):
        kind = "keep" if r.get("new_label") and "keep" in r["new_label"] else ("last-layer predictor" if r["layer"] >= 35 else "name / word / token")
        L.append(f"| {i} | {esc(r['label'])} | {r['name']} | {kind} | {esc(r.get('new_label') or '')} |")
    L.append("")
    L.append("## Overview\n")
    L.append("| # | Meme | Unit | Layer | Echo (top 50) | Corpus max (in context) | Corpus top 8 standalone | Best blind | Blind pct | Near miss | Near pct | Verdict |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(out["memes"], 1):
        b, nm = r["blind_maximum"]["best"], r["near_miss"]
        flag = " ⚠" if r["depth"]["early"] else ""
        L.append(f"| {i} | {esc(r['label'])} | {r['name']} | {r['layer']}{flag} | {r['sign_regime']['echo_share']:.0%} | "
                 f"{r['corpus']['max']:.2f} | {r['corpus']['max_standalone']:.2f} | {b['act']:.2f} | {b['pct']:.1f} | "
                 f"{nm['act']:.2f} | {nm['pct']:.1f} | {'survives' if r['checks']['survives'] else ('lenient' if r['checks']['survives_lenient'] else 'fails')} |"
                 if b and nm else f"| {i} | {esc(r['label'])} | {r['name']} | {r['layer']}{flag} | {r['sign_regime']['echo_share']:.0%} | {r['corpus']['max']:.2f} | – | – | – | – | – | no probes |")
    L.append("")
    L.append("## Per-meme tables\n")
    for i, r in enumerate(out["memes"], 1):
        sr, cp, b, nm, ck = r["sign_regime"], r["corpus"], r["blind_maximum"]["best"], r["near_miss"], r["checks"]
        L.append(f"### {i}. {r['label']} — {r['name']}\n")
        L.append(f"*{r['kind']}; present in {r['n_batman_works']} of 32 Batman works; family of {r['family_size']}. Paper's note: {esc(r['paper_note'])}*\n")
        L.append("| Check | Result | Pass |\n| --- | --- | --- |")
        L.append(f"| 1. Depth | Layer {r['layer']} of 36{' — in the first four layers, likely a token or spelling detector' if r['depth']['early'] else ''} | {'no' if r['depth']['early'] else 'yes'} |")
        fe = sr["first_echo_rank"]
        L.append(f"| 2. Sign regime | Top 50 firings: {sr['open']} gate-open, {sr['echo']} echo, {sr['other']} other; echo share {sr['echo_share']:.0%} "
                 f"({sr['echo_in_top8']} of the top 8; first echo at rank {fe if fe else 'none in the top 50'}). "
                 f"Work scores (mean of the top 3 sentences) drawn from echo firings: {sr['work_score_echo_share']['batman']:.0%} in Batman works, "
                 f"{sr['work_score_echo_share']['baseline']:.0%} in ordinary films; {sr['works_scored_mostly_by_echo']['batman']} of 32 Batman works and "
                 f"{sr['works_scored_mostly_by_echo']['baseline']} of 69 films scored mostly by echo. "
                 f"Peak tokens of the top 50 (gate-open/echo): {esc(', '.join(f'{t!r} ×{n} ({o}/{e})' for t, n, o, e in sr['peak_tokens_top50']))} | {'yes' if ck['echo_small'] else 'no'} |")
        L.append(f"| 3. Negative pole | {r['negative_pole']['name']}, max {r['negative_pole']['max']:.2f}. {esc(r['negative_pole']['reading'] or '(reading pending)')} | – |")
        if b:
            L.append(f"| 4. Blind maximum | Best of {len(r['blind_maximum']['attempts'])}: {b['act']:.2f}, against {cp['max']:.2f} for the corpus maximum in context "
                     f"and {cp['max_standalone']:.2f} for the corpus top 8 read standalone (corpus p{b['pct']:.1f}; peak on {esc(repr(b['peak_token']))}, {b['regime']}). "
                     f"The weakest of the corpus top 8 is {cp['min_top8']:.2f} in context, {cp['min_top8_standalone']:.2f} standalone; neutral sentences max {r['blind_maximum']['neutral_max']:.2f}. | "
                     f"{'yes' if ck['blind_beats_max'] else ('would make the top 8' if ck['blind_in_top8'] else 'no')}{'; also beats the standalone top 8' if ck['blind_beats_standalone_max'] else ''} |")
        else:
            L.append("| 4. Blind maximum | not run | – |")
        if nm:
            L.append(f"| 5. Near miss | {nm['act']:.2f} (corpus p{nm['pct']:.1f}; {nm['ratio_to_max']:.0%} of the in-context maximum, "
                     f"{(nm['act'] / b['act'] if b and b['act'] else float('nan')):.0%} of the best blind maximum; peak on {esc(repr(nm['peak_token']))}) | {'yes' if ck['near_miss_low'] else 'no'} |")
        else:
            L.append("| 5. Near miss | not run | – |")
        L.append(f"| **Verdict** | {esc(r['verdict'] or ('survives' if ck['survives'] else 'does not survive'))} | **{'survives' if ck['survives'] else ('lenient' if ck['survives_lenient'] else 'fails')}** |\n")
        if r.get("new_label"):
            L.append(f"Suggested label: **{esc(r['new_label'])}**\n")
        L.append("Strongest corpus sentences (activation, regime, peak token):\n")
        for t in cp["top8"]:
            L.append(f"- {t['act']:.2f} {t['regime']} {t['peak_token']!r} · *{t['work']}*: {trim(t['text'])}")
        if sr["echo_dominates"]:
            L.append("\nEcho dominates; strongest gate-open firings:\n")
            for t in sr["gate_open_top8"]:
                L.append(f"- {t['act']:.2f} {t['peak_token']!r} · *{t['work']}*: {trim(t['text'])}")
            if sr["relabel_from_gate_open"]:
                L.append(f"\nRelabel from gate-open firings: **{sr['relabel_from_gate_open']}**")
        L.append(f"\nNegative pole {r['negative_pole']['name']}, strongest sentences:\n")
        for t in r["negative_pole"]["top8"]:
            L.append(f"- {t['act']:.2f} {t['peak_token']!r} · *{t['work']}*: {trim(t['text'])}")
        if cp.get("top8_standalone"):
            L.append("\nThe corpus top 8 read standalone (activation, in-context activation in the list above):\n")
            for t in cp["top8_standalone"]:
                L.append(f"- {t['act']:.2f} (p{t['pct']:.1f}, {t['regime']}, peak {t['peak_token']!r}) · *{t['work']}*: {trim(t['sentence'])}")
        if r["blind_maximum"]["attempts"]:
            L.append("\nBlind maximum attempts:\n")
            for t in sorted(r["blind_maximum"]["attempts"], key=lambda x: -x["act"]):
                L.append(f"- {t['act']:.2f} (p{t['pct']:.1f}, {t['regime']}, peak {t['peak_token']!r}) {t['kind']}: {t['sentence']}")
        if nm:
            L.append(f"\nNear miss ({nm.get('words_used') or 'label words'}): {nm['sentence']}")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "corpus":
        corpus_pass(sys.argv[2])
    elif cmd == "probe":
        probe_pass(sys.argv[2], sys.argv[3], *sys.argv[4:5])
    elif cmd == "report":
        write_report(sys.argv[2])
    else:
        raise SystemExit(__doc__)
