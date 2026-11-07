"""Merge candidate units that fire strongly on the same sentences into meme families (v1's rule, on strong activations).

Two units join one family when their sentence profiles over the 3,700 original-text sentences correlate at 0.5 or
more, by average linkage, as in v1. A profile is the unit's share of scale in each sentence, set to zero below the
30% threshold, so only strong firings count. Average linkage cut at 0.5 never joins units from different connected
components of the "correlation at least 0.5" graph, so the clustering is done one component at a time; that gives
the same families as clustering everything at once, without a matrix of all pairs.

    cd study && ../.venv/bin/python v2/analysis/families_v2.py [model-tag ...]

Needs v2/acts/<tag>/sent_orig.u8.npy and sent_units.npy (the sentence-level pass over results/v2/<tag>/sentence_units.npy)
and results/v2/<tag>/pools.npz (canon_v2.py). Writes results/v2/<tag>/families.json.gz.
"""
import gzip
import json
import sys

import numpy as np
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform

import common as C

import experiments as X

MERGE_CORR = 0.5


def correlation_edges(P, thr=MERGE_CORR, block=2000):
    """Pairs (i, j, r) of columns of sparse P [sentences, units] whose Pearson correlation is at least thr."""
    n = P.shape[0]
    mean = np.asarray(P.mean(0)).ravel()
    sd = np.sqrt(np.maximum(np.asarray(P.multiply(P).mean(0)).ravel() - mean ** 2, 1e-12))
    Pt = P.T.tocsr()
    rows, cols, vals = [], [], []
    for s in range(0, P.shape[1], block):
        G = (Pt[s:s + block] @ P).toarray() / n                       # E[xy] for this block against every unit
        R = (G - mean[s:s + block, None] * mean[None, :]) / (sd[s:s + block, None] * sd[None, :])
        i, j = np.nonzero(R >= thr)
        keep = (i + s) < j
        rows.append(i[keep] + s); cols.append(j[keep]); vals.append(R[i[keep], j[keep]])
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def cluster(P):
    """Family label per column of P."""
    U = P.shape[1]
    i, j, _ = correlation_edges(P)
    graph = sparse.coo_matrix((np.ones(len(i), bool), (i, j)), shape=(U, U))
    n_comp, comp = connected_components(graph, directed=False)
    fam = np.zeros(U, dtype=np.int64)
    next_id = 0
    order = np.argsort(comp, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(comp, minlength=n_comp))]
    for c in range(n_comp):
        members = order[bounds[c]:bounds[c + 1]]
        if len(members) == 1:
            fam[members] = next_id
            next_id += 1
            continue
        Z = P[:, members].toarray().astype(np.float32)
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
        Dm = 1 - (Z.T @ Z) / Z.shape[0]
        np.clip(Dm, 0, 2, out=Dm)
        np.fill_diagonal(Dm, 0)
        lab = fcluster(linkage(squareform(Dm, checks=False), "average"), t=1 - MERGE_CORR, criterion="distance")
        fam[members] = next_id + lab - 1
        next_id += lab.max()
    return fam, n_comp


def run_model(tag):
    d = C.ACTS / tag
    units = np.load(d / "sent_units.npy")
    cand = np.load(C.RESULTS / tag / "pools.npz")["candidates"]
    col = {int(u): k for k, u in enumerate(units)}
    idx = np.array([col[int(u)] for u in cand])
    S = np.load(d / "sent_orig.u8.npy", mmap_mode="r")
    thr = int(round(C.THETA * C.STEP))
    A = np.asarray(S[:, idx])
    P = sparse.csc_matrix(np.where(A >= thr, A, 0).astype(np.float32) / C.STEP)
    fam, n_comp = cluster(P)
    So = np.load(d / "orig_main_top1.u8.npy", mmap_mode="r")[:, cand]
    Po = So >= thr
    share = So[X.BAT].astype(np.float32).mean(0)
    nb = Po[X.BAT].sum(0)
    cous = np.concatenate([X.HERO_FILM_COUSINS, X.LIT_COUSINS])
    works = [json.loads(l) for l in open("data/corpus.jsonl")]
    sentences = [(w["label"], s) for w in works for s in w["sentences"]]
    where = np.load(d / "sent_orig_where.npy", mmap_mode="r")
    tokens = json.load(open(d / "sent_tokens.json"))["orig"]
    families = []
    for f in np.unique(fam):
        members = np.flatnonzero(fam == f)
        rep = members[np.lexsort((-share[members], -nb[members]))[0]]          # most Batman works, then strongest on average
        top = np.argsort(-A[:, rep].astype(np.int32), kind="stable")[:8]
        families.append({"unit": int(cand[rep]), "name": C.unit_name(tag, int(cand[rep])), "n_members": int(len(members)),
                         "member_units": [int(cand[m]) for m in members[np.argsort(-nb[members])][:50]],
                         "n_batman": int(nb[rep]), "n_cousins": int(Po[cous][:, rep].sum()), "n_ordinary_films": int(Po[X.FILMS][:, rep].sum()),
                         "tier": "core" if nb[rep] >= 29 else "common" if nb[rep] >= 23 else "optional",
                         "present": [int(w) for w in np.flatnonzero(Po[:, rep])],
                         "top_sentences": [{"work": sentences[i][0], "share": round(float(A[i, rep]) / C.STEP, 3), "text": sentences[i][1],
                                            "peak_token": tokens[i][int(where[i, idx[rep]])] if int(where[i, idx[rep]]) < len(tokens[i]) else ""}
                                           for i in top if A[i, rep] > 0]})
    families.sort(key=lambda r: (-r["n_batman"], -r["n_members"]))
    out = {"model": tag, "n_candidate_units": int(len(cand)), "n_families": len(families), "n_components": int(n_comp),
           "tiers": {t: sum(f["tier"] == t for f in families) for t in ("core", "common", "optional")},
           "core_batman_own": sum(f["tier"] == "core" and f["n_cousins"] <= 3 for f in families),
           "largest_family": max(f["n_members"] for f in families), "families": families}
    (C.RESULTS / tag).mkdir(parents=True, exist_ok=True)
    with gzip.open(C.RESULTS / tag / "families.json.gz", "wt") as fh:
        json.dump(out, fh)
    print(tag, len(cand), "candidate units ->", len(families), "families", out["tiers"], "largest", out["largest_family"])
    return out


if __name__ == "__main__":
    for tag in sys.argv[1:] or [t for t in C.MODELS if (C.ACTS / t / "sent_orig.u8.npy").exists()]:
        run_model(tag)
