"""Stage 1: find the neurons that act as Batman memes in one model.

A unit is one pole of one MLP neuron. A work's score on a unit is the mean of its three strongest
sentences. A meme is "present" in a work when the work's score beats all but one of the 70 baseline
films (so an ordinary story trips it about 3% of the time by chance).

Candidate memes are units present in at least MIN_WORKS of the 33 Batman works. Units that co-fire on
the same sentences are merged into one meme family, keeping the strongest unit as its representative.

Writes OUT/<tag>/memes.json with each family's presence pattern and top sentences for labelling.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

MIN_WORKS = 6
TOP_K_SENT = 3
MERGE_CORR = 0.5


def load_units(acts_dir):
    meta = json.load(open(acts_dir / "meta.json"))
    pos = np.load(acts_dir / "pos.npy", mmap_mode="r")
    neg = np.load(acts_dir / "neg.npy", mmap_mode="r")
    ps, ns = np.load(acts_dir / "pos_scale.npy"), np.load(acts_dir / "neg_scale.npy")
    return meta, pos, neg, ps, ns


def work_scores(arr, scale, sent_work, n_works, sign):
    """[n_works, n_neurons] mean of the top-3 sentence activations (sign flips the negative pole)."""
    sent_work = np.asarray(sent_work)
    out = np.zeros((n_works, arr.shape[1]), dtype=np.float32)
    for w in range(n_works):
        rows = np.where(sent_work == w)[0]
        block = sign * arr[rows[0]:rows[-1] + 1].astype(np.float32)
        k = min(TOP_K_SENT, len(rows))
        top = np.partition(block, -k, axis=0)[-k:]
        out[w] = top.mean(axis=0) * scale
    return out


def unit_sentences(arr, scale, unit_cols, sign):
    return sign * arr[:, unit_cols].astype(np.float32) * scale[unit_cols]


def main(tag, acts_root, out_root):
    acts_dir = Path(acts_root) / tag
    works = [json.loads(l) for l in open("data/corpus.jsonl")]
    sentences = [s for w in works for s in w["sentences"]]
    meta, pos, neg, ps, ns = load_units(acts_dir)
    sent_work = meta["sent_work"]
    n_works, n_neurons = len(works), pos.shape[1]
    group = np.array([w["group"] for w in works])
    bat, base = np.where(group == "batman")[0], np.where(group == "baseline")[0]

    print("scoring works...", flush=True)
    S = np.concatenate([work_scores(pos, ps, sent_work, n_works, 1),
                        work_scores(neg, ns, sent_work, n_works, -1)], axis=1)  # [works, 2N]
    base_sorted = np.sort(S[base], axis=0)
    thr = base_sorted[-2]                                  # beat 69 of 70 ordinary films
    spread = base_sorted[-2] - np.median(S[base], axis=0) + 1e-6
    present = S > thr
    n_bat = present[bat].sum(axis=0)
    n_base = present[base].sum(axis=0)

    # how many units clear the bar at each count, vs. what chance would give
    p0 = 2 / (len(base) + 1)
    from scipy.stats import binom
    for k in (4, 6, 8, 12, 20, 27):
        print(f"  units present in >= {k:2d} Batman works: {(n_bat >= k).sum():7d}   "
              f"(chance ~{2 * n_neurons * binom.sf(k - 1, len(bat), p0):.1f})")

    cand = np.where(n_bat >= MIN_WORKS)[0]
    # strength: how far above the bar the Batman works sit, on average, in units of baseline spread
    strength = ((S[bat][:, cand] - thr[cand]) / spread[cand]).clip(min=0).mean(axis=0)
    order = np.argsort(-strength)
    cand, strength = cand[order], strength[order]
    print(f"{len(cand)} candidate units", flush=True)

    # sentence profiles of candidates, for merging units that fire on the same sentences
    is_neg = cand >= n_neurons
    col = np.where(is_neg, cand - n_neurons, cand)
    prof = np.zeros((len(sentences), len(cand)), dtype=np.float32)
    if (~is_neg).any():
        prof[:, ~is_neg] = unit_sentences(pos, ps, col[~is_neg], 1)
    if is_neg.any():
        prof[:, is_neg] = unit_sentences(neg, ns, col[is_neg], -1)
    Z = (prof - prof.mean(0)) / (prof.std(0) + 1e-6)
    D = Z.T @ Z
    D /= -len(sentences)
    D += 1                                   # D = 1 - correlation
    np.clip(D, 0, 2, out=D)
    np.fill_diagonal(D, 0)
    fam = fcluster(linkage(squareform(D, checks=False), "average"), t=1 - MERGE_CORR, criterion="distance")
    del D, Z

    families = []
    for f in np.unique(fam):
        members = np.where(fam == f)[0]
        rep = members[0]  # cand is sorted strongest-first, so the first member is the strongest
        u = cand[rep]
        top = np.argsort(-prof[:, rep])[:8]
        families.append({
            "unit": int(u),
            "layer": int(col[rep] // meta["hidden"]),
            "neuron": int(col[rep] % meta["hidden"]),
            "pole": "-" if is_neg[rep] else "+",
            "n_members": int(len(members)),
            "member_units": [int(cand[m]) for m in members[:50]],
            "n_batman": int(n_bat[u]),
            "n_baseline": int(n_base[u]),
            "strength": float(((S[bat, u] - thr[u]) / spread[u]).clip(min=0).mean()),
            "present": [w["id"] for w in works if present[w["id"], u]],
            "z": [round(float((S[w, u] - np.median(S[base, u])) / spread[u]), 2) for w in range(n_works)],
            "top_sentences": [{"work": works[sent_work[i]]["label"], "act": round(float(prof[i, rep]), 2),
                               "text": sentences[i]} for i in top],
        })
    families.sort(key=lambda f: (-f["n_batman"], -f["strength"]))
    out = Path(out_root) / tag
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"tag": tag, "n_units": int(2 * n_neurons), "threshold_rule": "beats 69 of 70 baseline films",
               "families": families}, open(out / "memes.json", "w"), indent=1)
    np.save(out / "work_scores_cand.npy", S[:, cand])
    np.save(out / "cand_units.npy", cand)
    print(f"{len(families)} meme families -> {out / 'memes.json'}")
    for f in families[:40]:
        print(f"  L{f['layer']:2d} n{f['neuron']:5d}{f['pole']} bat={f['n_batman']:2d} base={f['n_baseline']} "
              f"size={f['n_members']:3d} | {f['top_sentences'][0]['work']}: {f['top_sentences'][0]['text'][:90]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "results")
