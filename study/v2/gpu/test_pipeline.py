"""Checks the GPU pipeline against a slow, obvious re-computation, on a tiny model on CPU.

Run after the passes have been run on the test inputs (see v2/README.md, "Testing the pipeline"):
    python test_pipeline.py <model-tag> <reference.jsonl> <study-dir> <out-root> <selected-units.npy>
Every check prints PASS or FAIL; the script exits non-zero if any fails.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from neurons import Tap, encode_document, geometry, layers_of, lead_in, load
from study_pass import CODE, ICON, code_tokens, cut_length

tag, ref_path, study_dir, out_root, sel_path = sys.argv[1:6]
root = Path(out_root) / tag
model, tok = load(tag)
L, D = geometry(model)
U = L * D
fails = []
VALUE = {}
DEV = model.model.embed_tokens.weight.device
# In bfloat16 a padded batch and a single document give slightly different activations, so on a GPU the checks of the
# batched reference pass are looser; every pass that reads one document at a time must still match exactly.
LOOSE = model.dtype != torch.float32


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name, detail, flush=True)
    if not ok:
        fails.append(name)


def naive(ids):
    """Gate pre-activations and neuron activations of every layer for one unpadded document, from plain hooks."""
    G, H = {}, {}
    hooks = []
    for i, layer in enumerate(layers_of(model)):
        hooks.append(layer.mlp.gate_proj.register_forward_hook(lambda m, a, o, i=i: G.__setitem__(i, o[0].detach().float().cpu())))
        hooks.append(layer.mlp.up_proj.register_forward_hook(lambda m, a, o, i=i: VALUE.__setitem__(i, o[0].detach().float().cpu())))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: H.__setitem__(i, a[0][0].detach().float().cpu())))
    with torch.no_grad():
        out = model.model(input_ids=torch.tensor([ids]).to(DEV)).last_hidden_state.float().cpu()
    for h in hooks:
        h.remove()
    return torch.stack([G[i] for i in range(L)]), torch.stack([H[i] for i in range(L)]), out      # [L, T, D]


def top(S, k):
    return S.topk(min(k, S.shape[0]), dim=0).values.mean(0).reshape(-1).numpy()


def close(got, want, tol=2e-3):
    return np.abs(got - want).max() / (np.abs(want).max() + 1e-9) < tol


def per_sentence(M, spans):
    return torch.stack([M[:, s:e].amax(1) for s, e in spans], 0), torch.stack([-M[:, s:e].amin(1) for s, e in spans], 0)     # [sent, L, D]


# 1. the tap leaves the model's computation unchanged, and the activation is act(gate) * value
orig = [json.loads(l) for l in open(Path(study_dir) / "corpus.jsonl")]
anon = [json.loads(l) for l in open(Path(study_dir) / "corpus_neutral.jsonl")]
ids, spans, _, _ = encode_document(tok, orig[0]["sentences"])
G, H, base = naive(ids)
seen = {}
tap = Tap(model, lambda l, g, h: seen.__setitem__(l, (g[0].float().cpu(), h[0].float().cpu())))
with torch.no_grad():
    tapped = model.model(input_ids=torch.tensor([ids]).to(DEV)).last_hidden_state.float().cpu()
tap.remove()
check("tap leaves the hidden states unchanged", torch.equal(base, tapped))
check("tap sees the same gate and activation as plain hooks", all(torch.allclose(seen[l][0], G[l], atol=1e-5) and torch.allclose(seen[l][1], H[l], atol=1e-5) for l in range(L)))
V = torch.stack([VALUE[i] for i in range(L)])
check("activation is silu(gate) * value", torch.allclose(H, torch.nn.functional.silu(G) * V, atol=2e-2 if LOOSE else 1e-4, rtol=2e-2 if LOOSE else 1e-4))
main = torch.where(G > 0, H, torch.zeros_like(H))
check("echo regime (gate closed) is bounded by 0.2785 * |value|", bool(((H - main).abs() <= (0.29 if LOOSE else 0.2785) * V.abs() + 1e-4).all()))
check("main regime keeps the sign of the value", bool((torch.sign(main) * torch.sign(V) >= 0).all()))
check("position 0 belongs to no sentence", all(s > 0 for s, _ in spans) and spans[0][0] == 1)

# 2. study pass, original text: per-work scores against the obvious computation
st = root / "study"
smeta = json.load(open(st / "meta.json"))
Sp, Sn = per_sentence(main, spans)
Rp, Rn = per_sentence(H, spans)
k = cut_length(orig[0]["sentences"])
want = {"orig_main_top1": np.concatenate([top(Sp, 1), top(Sn, 1)]), "orig_main_top3": np.concatenate([top(Sp, 3), top(Sn, 3)]),
        "orig_raw_top3": np.concatenate([top(Rp, 3), top(Rn, 3)]), "orig_raw_top1": np.concatenate([top(Rp, 1), top(Rn, 1)]),
        "orig_cut_main_top1": np.concatenate([top(Sp[:k], 1), top(Sn[:k], 1)])}
for name, w in want.items():
    check(f"study pass {name} (work 0)", close(np.load(st / f"{name}.npy", mmap_mode="r")[0].astype(np.float32), w))
check("length control keeps an opening of at most 20 sentences and 400 words", smeta["cut_sentences"][0] == k and 1 <= k <= 20
      and (k == 1 or sum(len(s.split()) for s in orig[0]["sentences"][:k]) <= 400))

# 3. study pass, name-free text: placeholder codes cannot carry the peak; iconography sentences dropped
w = next((i for i, x in enumerate(anon) if any(ICON.search(s) for s in x["sentences"])), 0)
ids_a, spans_a, text_a, offs_a = encode_document(tok, anon[w]["sentences"])
codes = code_tokens(text_a, offs_a)
check("placeholder-code tokens are found", len(codes) == smeta["versions"]["anon"]["n_code_tokens"][w] and len(codes) >= len(CODE.findall(text_a))
      and all(CODE.search(text_a[max(offs_a[i][0] - 6, 0):offs_a[i][1] + 6]) for i in codes))
Ga, Ha, _ = naive(ids_a)
main_a = torch.where(Ga > 0, Ha, torch.zeros_like(Ha))
masked = main_a.clone()
masked[:, codes] = 0
Mp, Mn = per_sentence(masked, spans_a)
Cp, Cn = per_sentence(main_a, spans_a)
Rpa, Rna = per_sentence(Ha, spans_a)
keep = torch.tensor([not ICON.search(s) for s in anon[w]["sentences"]])
ka = cut_length(orig[w]["sentences"])
after = sorted({c + 1 for c in codes if c + 1 < len(ids_a)} - set(codes))
masked2 = masked.clone()
masked2[:, after] = 0
Ap_, An_ = per_sentence(masked2, spans_a)
want = {"anon_main_top1": np.concatenate([top(Mp, 1), top(Mn, 1)]), "anon_main_top3": np.concatenate([top(Mp, 3), top(Mn, 3)]),
        "anon_after_main_top1": np.concatenate([top(Ap_, 1), top(An_, 1)]),
        "anon_codes_main_top1": np.concatenate([top(Cp, 1), top(Cn, 1)]), "anon_raw_top3": np.concatenate([top(Rpa, 3), top(Rna, 3)]),
        "anon_cut_main_top1": np.concatenate([top(Mp[:ka], 1), top(Mn[:ka], 1)]),
        "anon_icon_main_top1": np.concatenate([top(Mp[keep], 1), top(Mn[keep], 1)]), "anon_icon_raw_top3": np.concatenate([top(Rpa[keep], 3), top(Rna[keep], 3)])}
for name, x in want.items():
    check(f"study pass {name} (work {w})", close(np.load(st / f"{name}.npy", mmap_mode="r")[w].astype(np.float32), x))
check(f"masking the codes changes something and never raises a score ({int((~keep).sum())} iconography sentences dropped)",
      bool((np.maximum(want["anon_main_top1"], 0) <= np.maximum(want["anon_codes_main_top1"], 0) + 1e-6).all())
      and bool((want["anon_main_top1"] < want["anon_codes_main_top1"] - 1e-6).any()))      # below zero a pole is silent either way

# 4. reference pass: stored maxima against the obvious computation on three documents
meta = json.load(open(root / "ref" / "meta.json"))
docs = {d["i"]: d for d in map(json.loads, open(ref_path))}
tv, ti = np.load(root / "ref" / "top_pos_v.npy"), np.load(root / "ref" / "top_pos_i.npy")
nv, ni = np.load(root / "ref" / "top_neg_v.npy"), np.load(root / "ref" / "top_neg_i.npy")
rawp, rawn = np.load(root / "ref" / "raw_pos_max.npy"), np.load(root / "ref" / "raw_neg_min.npy")
n_le, n_cells, n_off, n_loc = 0, 0, 0, 0
slack = 0.1 if LOOSE else 1e-3
for di in meta["doc_index"][:3]:
    ids_d = tok(docs[di]["text"], add_special_tokens=True, truncation=True, max_length=meta["max_tokens"])["input_ids"]
    Gd, Hd, _ = naive(ids_d)
    md = torch.where(Gd > 0, Hd, torch.zeros_like(Hd))[:, 1:]
    dmax, dpos = md.max(1)
    n_le += int((dmax.numpy() > tv[:, 0, :] * (1 + slack) + 1e-4).sum()) + int((Hd[:, 1:].amax(1).numpy() > rawp * (1 + slack) + 1e-4).sum())
    n_cells += 2 * dmax.numel()
    for vals, locs, sign in ((tv, ti, 1), (nv, ni, -1)):
        hit = (locs // 2048 == di)
        for l, kk, n in list(zip(*np.nonzero(hit)))[:3000]:
            t = locs[l, kk, n] % 2048
            n_loc += 1
            n_off += not abs(sign * float(md[l, t - 1, n]) - vals[l, kk, n]) <= (0.1 if LOOSE else 2e-3) * abs(vals[l, kk, n]) + 1e-4
check(f"reference maxima are at least each document's own maximum ({n_le:,} of {n_cells:,} cells exceed it)", n_le <= (0.02 * n_cells if LOOSE else 0))
check(f"stored top examples point at the right token with the right value ({n_loc:,} checked, {n_off} off)", n_off <= (0.05 * n_loc if LOOSE else 0) and n_loc > 100)
check("top examples are sorted, strongest first", bool((np.diff(tv, axis=1) <= 1e-6).all()) and bool((np.diff(nv, axis=1) <= 1e-6).all()))

# 5. scales and uint8 shares
scale, scale_ref, scale_ref5, scale_raw = (np.load(st / f"{n}.npy") for n in ("scale", "scale_ref", "scale_ref5", "scale_raw"))
merged = root / "ref_all"                    # the reference corpus and, when it was run, the supplement
tv, ti = np.load(merged / "top_pos_v.npy"), np.load(merged / "top_pos_i.npy")
nv, ni = np.load(merged / "top_neg_v.npy"), np.load(merged / "top_neg_i.npy")
if (root / "ref_sup" / "raw_pos_max.npy").exists():
    rawp, rawn = np.maximum(rawp, np.load(root / "ref_sup" / "raw_pos_max.npy")), np.minimum(rawn, np.load(root / "ref_sup" / "raw_neg_min.npy"))
check("merged top examples are sorted, strongest first", bool((np.diff(tv, axis=1) <= 1e-6).all()) and bool((np.diff(nv, axis=1) <= 1e-6).all()))
ref_scale = np.concatenate([tv[:, 0].reshape(-1), nv[:, 0].reshape(-1)])
o1 = np.asarray(np.load(st / "orig_main_top1.npy", mmap_mode="r")).astype(np.float32)
check("scale = max(reference, study original texts)", np.allclose(scale, np.maximum(ref_scale, o1.max(0)), rtol=1e-3) and np.allclose(scale_ref, ref_scale))
check("fifth-example scale is the fifth strongest reference example", np.allclose(scale_ref5, np.concatenate([tv[:, 4].reshape(-1), nv[:, 4].reshape(-1)])))
from finalize_scale import reference_tops  # noqa: E402
dx = meta["doc_index"][0]
_, _, f1, f5 = reference_tops(merged, [dx])
W = -np.sort(-np.where(np.concatenate([ti, ni], 0) // 2048 == dx, 0, np.concatenate([tv, nv], 0)), axis=1)
check(f"leaving a document out of the reference-only scale drops its examples and no others ({int((f1 < ref_scale).sum()):,} units change)",
      np.allclose(f1, W[:, 0].reshape(-1)) and np.allclose(f5, W[:, 4].reshape(-1)) and bool((f1 < ref_scale).any()) and bool((f1 <= ref_scale).all()))
raw_ref = np.concatenate([np.maximum(rawp.reshape(-1), 0), np.maximum(-rawn.reshape(-1), 0)])
r1 = np.asarray(np.load(st / "orig_raw_top1.npy", mmap_mode="r")).astype(np.float32)
check("raw scale = max(reference raw maximum, study original texts)", np.allclose(scale_raw, np.maximum(raw_ref, r1.max(0)), rtol=1e-3))
u8 = np.asarray(np.load(st / "orig_main_top1.u8.npy", mmap_mode="r"))
share = np.clip(o1 / np.where(scale > 0, scale, np.inf), 0, 1.275)
check("uint8 shares are rounded down, in steps of 0.005", bool((u8 / 200 <= share + 1e-5).all()) and bool((share - u8 / 200 < 0.005 + 1e-5).all()))
check("a uint8 share of 60 or more means at least 30% of the scale", bool(((u8 >= 60) == (share >= 0.3 - 1e-6)).all()))

# 6. background pass
bg = root / "background"
bmeta = json.load(open(bg / "meta.json"))
sel = np.load(sel_path).astype(np.int64)
got = {n: np.load(bg / f) for n, f in (("main", "counts.npy"), ("ref", "counts_ref_scale.npy"), ("raw", "counts_raw.npy"))}
want = {n: np.zeros_like(got["main"]) for n in got}
dense, bits, tops = np.load(bg / "selected.u8.npy"), np.load(bg / "present30.bits.npy"), np.load(bg / "top_shares.u8.npy")
n_dense, n_bits, shares = 0, 0, []
inf = lambda s: np.where(s > 0, s, np.inf)
for r, di in enumerate(bmeta["doc_index"]):
    ids_d = encode_document(tok, docs[di]["sentences"])[0]
    Gd, Hd, _ = naive(ids_d)
    md = torch.where(Gd > 0, Hd, torch.zeros_like(Hd))[:, 1:]
    peak = np.concatenate([md.amax(1).reshape(-1).numpy(), -md.amin(1).reshape(-1).numpy()])
    raw = np.concatenate([Hd[:, 1:].amax(1).reshape(-1).numpy(), -Hd[:, 1:].amin(1).reshape(-1).numpy()])
    grid = lambda x: np.clip(np.floor(x * 200 + 1e-4), 0, 255)          # the storage grid: steps of 0.005, rounded down
    f = grid(peak / inf(scale))
    for kk, t in enumerate(bmeta["thresholds"]):
        want["main"][kk] += f >= round(t * 200)
        want["ref"][kk] += grid(peak / inf(scale_ref)) >= round(t * 200)
        want["raw"][kk] += grid(raw / inf(scale_raw)) >= round(t * 200)
    n_dense += int((dense[r] != f[sel]).sum())
    present = np.unpackbits(bits[r])[:2 * U].astype(bool)
    n_bits += int((present != (f >= 60)).sum())
    shares.append(f)
for n in got:
    near = sum(int((np.abs(want[n][kk] - got[n][kk]) > 0).sum()) for kk in range(len(got[n])))
    check(f"background counts ({n} scale) match", near <= 1e-4 * got[n].size, f"{near} of {got[n].size:,} cells differ")
check("background presence bits match", n_bits <= 1e-5 * bits.size * 8, f"{n_bits} bits differ")
check("background shares of the selected units match", n_dense == 0)
w_ = -np.sort(-np.stack(shares), axis=0)
n_top = int((tops[:len(w_)] != w_).sum())
check("background top shares are each unit's highest per-plot shares, sorted", n_top <= 1e-5 * w_.size and not tops[len(w_):].any(), f"{n_top} of {w_.size:,} cells differ")
check("top shares, counts and presence bits agree with one another",
      all(bool(((tops >= round(t * 200)).sum(0) == got["main"][kk]).all()) for kk, t in enumerate(bmeta["thresholds"]))
      and bool((np.unpackbits(bits, axis=1)[:, :2 * U].sum(0) == got["main"][1]).all()))
check("excluded documents are left out", not set(bmeta["excluded_docs"]) & set(bmeta["doc_index"]))

# 7. score_texts
z = np.load(root / "score_test.npz")
texts = [json.loads(l) for l in open(root / "score_test.jsonl")]
ok = True
for r, t in enumerate(texts):
    ids_t = lead_in(tok) + tok(t["text"], add_special_tokens=True)["input_ids"]          # the first word is scored: a lead-in token takes position 0
    Gt, Ht, _ = naive(ids_t)
    mt = torch.where(Gt > 0, Ht, torch.zeros_like(Ht))
    for c, u in enumerate(sel):
        p, l, n = u // U, (u % U) // D, u % D
        v = (mt[l, 1:, n] if p == 0 else -mt[l, 1:, n])
        ok &= abs(float(v.max().clamp(min=0)) - z["peak"][r, c]) <= 2e-3 * abs(z["peak"][r, c]) + 1e-4
        ok &= abs(float(Ht[l, -1, n]) - z["last"][r, c]) <= 2e-3 * abs(z["last"][r, c]) + 1e-4
        if z["peak"][r, c] > 0:
            ok &= int(v.argmax()) + 1 == int(z["where"][r, c])
check("score_texts peak, position and last-token activation match", ok)

# 8. sentence-level pass for chosen units
su = np.load(st / "sent_units.npy")
so, sa = np.load(st / "sent_orig.u8.npy"), np.load(st / "sent_anon.u8.npy")
ok = True
for c, u in enumerate(su):
    p, l, n = u // U, (u % U) // D, u % D
    v0 = (Sp if p == 0 else Sn)[:, l, n].numpy() / inf(scale)[u]
    va = (Mp if p == 0 else Mn)[:, l, n].numpy() / inf(scale)[u]
    row_a = sum(len(x["sentences"]) for x in anon[:w])
    ok &= bool((np.abs(so[:len(v0), c] / 200 - np.clip(v0, 0, 1.275)) < 0.0051).all())
    ok &= bool((np.abs(sa[row_a:row_a + len(va), c] / 200 - np.clip(va, 0, 1.275)) < 0.0051).all())
check("sentence-level shares match, with codes masked in the name-free text", ok)

print("\n%d checks failed" % len(fails) if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
