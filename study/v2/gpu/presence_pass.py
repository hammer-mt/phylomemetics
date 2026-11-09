"""Second corpora: which units are present in which text, on a scale fixed beforehand.

Reads each text as one continuous document, exactly as study_pass.py reads a study work (same sentences, same
sign regime, position 0 never scored, a code's own tokens never carry a peak), and writes presence as packed bits.
Nothing a text does changes the scale: the scale file is an input (the reference-only scale, scale_ref.npy).

    python presence_pass.py <model-tag> <texts.jsonl> <out-dir> --scale <scale.npy> [--mask-codes [--mask-after]] [--planes all|primary]

<texts.jsonl> holds one record per text: {"id": ..., "sentences": [...]} and optionally (in every record or in none)
    "cut"    how many opening sentences the length control keeps (study_pass.cut_length of the ORIGINAL text)
    "drop"   indices of the sentences left out of the iconography control

Planes (each <out-dir>/<plane>.bits.npy, uint8 [texts, 2 * neurons / 8], units in the usual order: positive poles,
then negative poles, 8 units to a byte, most significant bit first as numpy.packbits writes them):

    top1_20 top1_30 top1_40 top1_50    strongest sentence at least 20, 30, 40, 50% of the unit's scale
    top3_20 top3_30 top3_40 top3_50    mean of the three strongest sentences at least that share
    cut_top1_30                        strongest sentence of the opening at least 30%          (records with "cut")
    icon_top1_30                       the same over the sentences not dropped                  (records with "drop")
    over100                            strongest sentence above the scale itself (v2's own scale would have been
                                       raised by this text for these units)

--planes primary writes only top1_30, cut_top1_30, icon_top1_30 and over100.

Shares are put on the storage grid of v2 first (steps of 0.005, rounded down), so "at least 30%" means the same
here as in common.Setting: floor(score / scale * 200 + 1e-4) >= 60. A unit whose scale is zero is never present.
Also writes <out-dir>/meta.json (provenance, hashes of the inputs, tokens and present units per text). It is written
last and an earlier run's planes and meta.json are removed first, so a folder with a meta.json holds one complete run.

--mask-after (with --mask-codes; the follow-up of v2/second/PLAN.md Section 8): the token directly after each code cannot
carry a peak either, by the rule of study_pass.py's after_main_top1. Every plane of such a run is masked that way. Without
the option nothing is read differently.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from neurons import Tap, encode_document, geometry, load, now, provenance, run_hidden, save_json
from study_pass import code_tokens, top13

STEP = 200
THETAS = [0.2, 0.3, 0.4, 0.5]


def cutoff(theta):
    return int(math.ceil(theta * STEP - 1e-6))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def plane_names(which, has_cut, has_drop):
    names = [f"top1_{round(t * 100)}" for t in THETAS] + [f"top3_{round(t * 100)}" for t in THETAS] if which == "all" else ["top1_30"]
    return names + (["cut_top1_30"] if has_cut else []) + (["icon_top1_30"] if has_drop else []) + ["over100"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("texts"); ap.add_argument("out")
    ap.add_argument("--scale", required=True)
    ap.add_argument("--mask-codes", action="store_true", help="name-free texts: a code's own tokens (X1, X2, ...) cannot carry a peak")
    ap.add_argument("--mask-after", action="store_true", help="with --mask-codes: the token directly after each code cannot carry a peak either")
    ap.add_argument("--planes", default="all", choices=["all", "primary"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    assert a.mask_codes or not a.mask_after, "--mask-after masks the token after each code: it needs --mask-codes"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    texts = [json.loads(l) for l in open(a.texts)][: a.limit]
    N = len(texts)
    has_cut, has_drop = all("cut" in t for t in texts), all("drop" in t for t in texts)
    # checked before the model is loaded: a control is written for every text or for none, and a bad record must not stop
    # the run half way or score other sentences without saying so (a negative index counts from the end, no cut is the whole text)
    assert all(("cut" in t) == has_cut and ("drop" in t) == has_drop for t in texts), 'some records have "cut" or "drop" and some do not'
    for t in texts:
        assert not has_cut or (type(t["cut"]) is int and t["cut"] >= 1), f'text {t["id"]}: cut {t["cut"]!r} is not a whole number of at least 1'
        assert not has_drop or all(type(i) is int and 0 <= i < len(t["sentences"]) for i in t["drop"]), f'text {t["id"]}: a drop index is not one of its {len(t["sentences"])} sentences'
    model, tok = load(a.tag)
    L, D = geometry(model)
    U = L * D
    assert D % 8 == 0
    s = torch.tensor(np.load(a.scale)).reshape(2, L, D)
    safe = torch.where(s > 0, s, torch.full_like(s, float("inf")))
    # every text is tokenised before any text is read, so that a text the tokeniser cannot place stops the run at once
    enc = [encode_document(tok, t["sentences"]) for t in texts]
    names = plane_names(a.planes, has_cut, has_drop)
    for old in [out / "meta.json", *out.glob("*.bits.npy")]:      # a folder holds one run: meta.json is there only once every plane in it
        old.unlink(missing_ok=True)                                # is complete, and no plane of an earlier run is left beside the new ones
    planes = {n: np.lib.format.open_memmap(out / f"{n}.bits.npy", mode="w+", dtype=np.uint8, shape=(N, 2 * U // 8)) for n in names}
    present = {n: np.zeros(N, dtype=np.int64) for n in names}
    n_tokens, n_code_tokens, n_after_tokens = [], [], []
    print(f"{now()} {a.tag}: {N} texts, {sum(len(e[0]) for e in enc):,} tokens, planes {names}, scale {a.scale}", flush=True)
    ctx = {}

    scale_on = {}

    def scale_at(layer, pole, dev):
        if (layer, pole) not in scale_on:
            scale_on[(layer, pole)] = (safe[pole, layer].to(dev), s[pole, layer].to(dev))
        return scale_on[(layer, pole)]

    def fn(layer, g, h):
        g, h = g[0], h[0]
        main = torch.where(g > 0, h, torch.zeros_like(h))
        if len(ctx["codes"]):
            main = main.clone()
            main[ctx["codes"].to(main.device)] = 0
        spans = ctx["spans"]
        Sp = torch.stack([main[b:e].amax(0) for b, e in spans])
        Sn = torch.stack([-main[b:e].amin(0) for b, e in spans])
        for pole, S in enumerate((Sp, Sn)):
            div, raw_scale = scale_at(layer, pole, S.device)

            def steps(score):
                return torch.clamp(torch.floor(score / div * STEP + 1e-4), 0, 255)

            t1, t3 = top13(S)
            t1, t3 = t1.to(torch.float16).float(), t3.to(torch.float16).float()       # the precision study_pass.py stores
            q1 = steps(t1)
            got = {}
            if a.planes == "all":
                q3 = steps(t3)
                for t in THETAS:
                    got[f"top1_{round(t * 100)}"] = q1 >= cutoff(t)
                    got[f"top3_{round(t * 100)}"] = q3 >= cutoff(t)
            else:
                got["top1_30"] = q1 >= cutoff(0.3)
            if has_cut:
                got["cut_top1_30"] = steps(top13(S[:ctx["cut"]])[0].to(torch.float16).float()) >= cutoff(0.3)
            if has_drop:
                got["icon_top1_30"] = steps(top13(S, ctx["keep"].to(S.device))[0].to(torch.float16).float()) >= cutoff(0.3)
            got["over100"] = (t1 > raw_scale) & (raw_scale > 0)
            bits = torch.stack([got[n] for n in names]).cpu().numpy()                 # [planes, D]
            packed = np.packbits(bits, axis=1)
            lo = (pole * U + layer * D) // 8
            for k, n in enumerate(names):
                planes[n][ctx["w"], lo:lo + D // 8] = packed[k]
                present[n][ctx["w"]] += int(bits[k].sum())
        return None

    tap = Tap(model, fn)
    t0 = time.time()
    for w, (t, (ids, spans, text, offsets)) in enumerate(zip(texts, enc)):
        codes = code_tokens(text, offsets) if a.mask_codes else []
        after = sorted({c + 1 for c in codes if c + 1 < len(ids)} - set(codes)) if a.mask_after else []        # study_pass.py's rule
        keep = torch.ones(len(spans), dtype=torch.bool)
        if has_drop:
            keep[torch.tensor(t["drop"], dtype=torch.long)] = False
        ctx.update(w=w, spans=spans, codes=torch.tensor(codes + after, dtype=torch.long), cut=t.get("cut"), keep=keep)
        run_hidden(model, torch.tensor([ids]))
        n_tokens.append(len(ids))
        n_code_tokens.append(len(codes))
        n_after_tokens.append(len(after))
        if w % 50 == 0 or w == N - 1:
            print(f"{now()} {w + 1}/{N}  {len(ids):5d} tokens  {(time.time() - t0):.0f}s", flush=True)
    tap.remove()
    for p in planes.values():
        p.flush()
    save_json({"tag": a.tag, "provenance": provenance(model), "n_layers": L, "width": D, "n_texts": N, "ids": [t["id"] for t in texts],
               "planes": names, "mask_codes": a.mask_codes, "scale_file": str(a.scale), "scale_sha256": sha256(a.scale), "texts_sha256": sha256(a.texts),
               "n_sentences": [len(t["sentences"]) for t in texts], "n_tokens": n_tokens, "n_code_tokens": n_code_tokens,
               "mask_after": a.mask_after, "n_after_tokens": n_after_tokens,
               "present": {n: present[n].tolist() for n in names}, "seconds": round(time.time() - t0)}, out / "meta.json")
    print(f"{now()} saved to {out}", flush=True)


if __name__ == "__main__":
    main()
