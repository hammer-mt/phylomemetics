"""Pass 2: run the study's 114 works through a model, original and name-free, and score every unit in every work.

Each work is read as one continuous document (as in v1). For every unit and sentence the strongest activation is
taken, then reduced per work:

  main_top1       strongest sentence, main sign regime    (v2: "does the unit fire strongly anywhere in the work?")
  main_top3       mean of the three strongest sentences, main regime
  raw_top3        mean of the three strongest sentences over all sign regimes  (v1's score, for the v1-rule comparison)
  raw_top1        strongest sentence over all sign regimes  (the v2 rule with the echo regime left in)
  cut_main_top1   main_top1 over the work's opening only (at most 20 sentences and 400 words): the length control

Name-free texts. The neutral codes (X1, X2, ...) are our edit, not the story, so in the main-regime scores of the
name-free texts a code's own tokens cannot carry a unit's peak (codes_main_top1 keeps them in, as a check;
after_main_top1 also masks the token that follows each code, where a model may sum up the unknown entity; the raw
scores are left as v1 computed them). The name-free texts also get main_top1, main_top3 and raw_top3 with
iconography sentences dropped (E1's control).

    python study_pass.py <model-tag> <study-dir> <out-root>

<study-dir> holds corpus.jsonl and corpus_neutral.jsonl. Writes <out-root>/<tag>/study/<text>_<score>.npy, each
float16 [works, 2 * neurons] (positive poles, then negative poles as magnitudes), and meta.json.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from neurons import Tap, encode_document, geometry, load, now, provenance, run_hidden, save_json

ICON = re.compile(r"\b(bats?|batman|capes?|caped|cowl|clowns?|joker|gotham|wayne|crusader|knight|penguin|riddles?|riddler|caves?|signal|asylum|robin)\b", re.I)
CODE = re.compile(r"\bX\d+\b")
CUT_SENTENCES, CUT_WORDS = 20, 400


def cut_length(sentences):
    """How many opening sentences the length control keeps: at most 20, and at most 400 words (always at least one)."""
    words = k = 0
    for s in sentences[:CUT_SENTENCES]:
        n = len(s.split())
        if k and words + n > CUT_WORDS:
            break
        words, k = words + n, k + 1
    return k


def code_tokens(text, offsets):
    """Indices of tokens that overlap a placeholder code in the text."""
    spans = [m.span() for m in CODE.finditer(text)]
    out, j = [], 0
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        while j < len(spans) and spans[j][1] <= a:
            j += 1
        if j < len(spans) and spans[j][0] < b and a < spans[j][1]:
            out.append(i)
    return out


def top13(S, keep=None):
    """Strongest sentence and mean of the three strongest, per unit. S: [sentences, D]."""
    if keep is not None and keep.any():
        S = S[keep]
    top = S.float().topk(min(3, S.shape[0]), dim=0).values
    return top[0], top.mean(0)


def score_names(ver):
    names = ["main_top1", "main_top3", "raw_top3", "raw_top1", "cut_main_top1"]
    if ver == "anon":
        names += ["codes_main_top1", "after_main_top1", "icon_main_top1", "icon_main_top3", "icon_raw_top3"]
    return names


def main(tag, study_dir, out_root):
    out = Path(out_root) / tag / "study"
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load(tag)
    L, D = geometry(model)
    U = L * D
    versions = {"orig": [json.loads(l) for l in open(Path(study_dir) / "corpus.jsonl")],
                "anon": [json.loads(l) for l in open(Path(study_dir) / "corpus_neutral.jsonl")]}
    N = len(versions["orig"])
    cuts = [cut_length(w["sentences"]) for w in versions["orig"]]          # from the original text, used for both versions
    meta = {"tag": tag, "n_layers": L, "width": D, "n_works": N, "cut_sentences": cuts, "provenance": provenance(model),
            "cut_rule": f"opening sentences, at most {CUT_SENTENCES} and at most {CUT_WORDS} words", "versions": {}}
    t0 = time.time()
    for ver, works in versions.items():
        arrs = {n: np.lib.format.open_memmap(out / f"{ver}_{n}.npy", mode="w+", dtype=np.float16, shape=(N, 2 * U)) for n in score_names(ver)}
        sent_work, n_tokens, n_code_tokens = [], [], []
        ctx = {}

        def put(name, w, lo, value):
            arrs[name][w, lo:lo + D] = value.cpu().numpy()

        def fn(layer, g, h):
            g, h = g[0], h[0]
            main = torch.where(g > 0, h, torch.zeros_like(h))
            spans, w, k = ctx["spans"], ctx["w"], ctx["cut"]
            Rp = torch.stack([h[s:e].amax(0) for s, e in spans])
            Rn = torch.stack([-h[s:e].amin(0) for s, e in spans])
            if ver == "anon":                                   # with the codes' own tokens left in, for the check
                Cp = torch.stack([main[s:e].amax(0) for s, e in spans])
                Cn = torch.stack([-main[s:e].amin(0) for s, e in spans])
                if len(ctx["codes"]):
                    main = main.clone()
                    main[ctx["codes"].to(main.device)] = 0
            Sp = torch.stack([main[s:e].amax(0) for s, e in spans])
            Sn = torch.stack([-main[s:e].amin(0) for s, e in spans])
            if ver == "anon":                                   # and with the token after each code masked as well
                if len(ctx["after"]):
                    main = main.clone()
                    main[ctx["after"].to(main.device)] = 0
                Ap = torch.stack([main[s:e].amax(0) for s, e in spans])
                An = torch.stack([-main[s:e].amin(0) for s, e in spans])
            for pole, (S, R) in enumerate(((Sp, Rp), (Sn, Rn))):
                lo = pole * U + layer * D
                t1, t3 = top13(S)
                r1, r3 = top13(R)
                put("main_top1", w, lo, t1); put("main_top3", w, lo, t3); put("raw_top1", w, lo, r1); put("raw_top3", w, lo, r3)
                put("cut_main_top1", w, lo, top13(S[:k])[0])
                if ver == "anon":
                    keep = ctx["keep"].to(S.device)
                    t1, t3 = top13(S, keep)
                    put("icon_main_top1", w, lo, t1); put("icon_main_top3", w, lo, t3); put("icon_raw_top3", w, lo, top13(R, keep)[1])
                    put("codes_main_top1", w, lo, top13(Cp if pole == 0 else Cn)[0])
                    put("after_main_top1", w, lo, top13(Ap if pole == 0 else An)[0])
            return None

        tap = Tap(model, fn)
        for w, work in enumerate(works):
            ids, spans, text, offsets = encode_document(tok, work["sentences"])
            codes = code_tokens(text, offsets) if ver == "anon" else []
            after = sorted({c + 1 for c in codes if c + 1 < len(ids)} - set(codes))
            ctx.update(spans=spans, w=w, cut=cuts[w], codes=torch.tensor(codes, dtype=torch.long), after=torch.tensor(after, dtype=torch.long),
                       keep=torch.tensor([not ICON.search(s) for s in work["sentences"]]))
            run_hidden(model, torch.tensor([ids]))
            sent_work += [w] * len(spans)
            n_tokens.append(len(ids))
            n_code_tokens.append(len(codes))
            if w % 10 == 0 or w == N - 1:
                print(f"{now()} {ver} {w + 1}/{N} {work['label'][:28]:28s} {len(ids):5d} tokens  {(time.time() - t0):.0f}s", flush=True)
        tap.remove()
        for a in arrs.values():
            a.flush()
        meta["versions"][ver] = {"sent_work": sent_work, "n_tokens": n_tokens, "n_code_tokens": n_code_tokens, "scores": score_names(ver)}
        print(f"{now()} {ver}: {len(sent_work)} sentences, {sum(n_tokens):,} tokens, {sum(n_code_tokens):,} of them in codes", flush=True)
    meta["seconds"] = round(time.time() - t0)
    save_json(meta, out / "meta.json")
    print(f"{now()} saved to {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
