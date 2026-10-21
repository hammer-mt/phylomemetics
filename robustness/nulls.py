"""Stronger nulls and controls for the Phase 1 findings, from the robustness review.

F1 chance     units present in >= 6, 23 and 29 of 32 works: Batman against random 32-film pseudo-franchises,
              both scored against the same bar set by the other 37 ordinary films (discover.py's binomial
              chance assumes the works are independent).
F2 agreement  Llama twins of the 200 strongest Qwen memes against three nulls drawn from 30,000 random active
              non-meme Qwen units (matched on sparsity, on sparsity and layer, or on how well a single word
              reproduces the profile); the same twin pairs on the name-free texts; grid agreement and kappa.
F2 trees      the Mantel floor from random non-meme units, partial Mantels after TF-IDF and after summary
              length and meme counts, and the rank each tree gives the documented source comic.
F3 wording    rewording survival against a mismatched-rewording null (each unit scored on the other memes'
              rewordings), and revenge-unit controls (harm without retaliation, vows without revenge).
Labels        blind two-choice label test: judge accuracy with an exact CI, against a word-overlap judge.

Run from the project root (the wording part loads both models with MLX):
  PHYLO_ACTS=... PHYLO_ACTS_ANON=... .venv/bin/python robustness/nulls.py [--skip chance,agreement,trees,wording,labels]
Writes results/robustness/nulls.json (skipped parts keep their earlier values) and results/robustness/label_test.json.
"""
import argparse
import gzip
import json
import os
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import binom, binomtest, wilcoxon
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from discover import MIN_WORKS, load_units, work_scores  # noqa: E402
from phylo import ADAPTATIONS, best_partner, cosine_dist, jaccard, unit_profile  # noqa: E402

ACTS = Path(os.environ.get("PHYLO_ACTS", "activations"))
ANON = Path(os.environ.get("PHYLO_ACTS_ANON", "activations_anon"))
OUT = ROOT / "results/robustness"
MODELS = ["qwen3-8b", "llama3.1-8b"]
SEED = 20261020
NM, N_POOL, N_PERM, N_RAND = 200, 30000, 2000, 20
NULLS = ["sparsity", "sparsity_layer", "lexical"]
CLASSES = ["name", "word", "wiki", "concept", "unclear"]

works = [json.loads(l) for l in open(ROOT / "data/corpus.jsonl")]
sentences = [s for w in works for s in w["sentences"]]
group = np.array([w["group"] for w in works])
n_sent = np.array([len(w["sentences"]) for w in works], dtype=float)
BAT, FILMS, LIN = np.where(group == "batman")[0], np.where(group == "baseline")[0], np.where(group != "baseline")[0]
IU = np.triu_indices(len(LIN), 1)
labels = json.load(open(ROOT / "results/labels.json"))
labels_wf = json.load(open(ROOT / "results/labels_wf.json"))

# The review's reading of the 33 strongest Qwen families (8 top Qwen sentences, 3 of the Llama twin's; the review's
# agreement/top33.json): name/place detector, common-word detector, Wikipedia house style, story concept, unclear.
TOP33_CLASS = {
    27328: "name", 65066: "name", 118766: "name", 577083: "name", 501280: "name", 496238: "name", 45473: "word",
    574705: "name", 481199: "word", 131948: "concept", 74919: "name", 44393: "name", 32791: "name", 17909: "concept",
    523074: "name", 849762: "wiki", 496279: "name", 502805: "name", 465474: "name", 460110: "name", 28675: "word",
    40056: "name", 457566: "name", 68088: "name", 117668: "name", 130069: "name", 43445: "name", 116484: "unclear",
    577158: "name", 119629: "concept", 55673: "name", 426468: "name", 561931: "concept"}
# the same classes from the project's own label kinds
KMAP = {"character": "name", "place": "name", "register": "wiki", "trope": "concept", "genre": "concept",
        "object": "word", "mixed": "unclear"}

# revenge-unit controls written for the review: the wrong itself with no retaliation, and vows about a parent
HARM = [
    "The businessman quietly cheated the widow's husband out of his savings.",
    "The men murdered his parents in an alley and fled with their wallets.",
    "She spent the summer mocking the new girl in front of the whole school.",
    "The pirates attacked the harbor at dawn and set every ship ablaze to steal the cargo.",
    "Hunger was the only thing he could think about in that cell.",
    "The old man loaded his rifle and shot the stray dog.",
    "The detective locked away her innocent brother on forged evidence.",
    "The CEO fired the hacker's entire team to cut costs before the board meeting.",
    "Enemy soldiers killed ten of his men in the ambush.",
    "The family drowned the servant girl in the river to hide their secret.",
    "The bully tormented him every day at school and stole his lunch.",
    "The partner double-crossed him, shot him and left him for dead in the desert.",
]
VOW = [
    "She swore to finish the novel her mother never got to write.",
    "He vowed to rebuild the farm his grandfather had lost in the flood.",
    "They promised to carry on the charity their father had started.",
    "He swore he would one day make his late father proud.",
    "She vowed to become the doctor her mother had always wanted to be.",
    "He swore to keep the family bakery open in his father's memory.",
]


# ------------------------------------------------------------------ shared
def families(tag):
    p = ROOT / "results" / tag / "memes.json"
    return json.load(open(p) if p.exists() else gzip.open(f"{p}.gz"))["families"]


def bar(S, ref):
    """Presence bar: second-highest score among the reference films."""
    return np.partition(S[ref], -2, axis=0)[-2]


def all_scores(root, tag):
    meta, pos, neg, ps, ns = load_units(root / tag)
    n = pos.shape[1]
    S = np.empty((len(works), 2 * n), np.float32)
    S[:, :n] = work_scores(pos, ps, meta["sent_work"], len(works), 1)
    S[:, n:] = work_scores(neg, ns, meta["sent_work"], len(works), -1)
    return S


def profiles(acts, units):
    """Sentence profiles [3700, len(units)] of the given units (read in column order)."""
    _, pos, neg, ps, ns = acts
    units = np.asarray(units)
    o = np.argsort(units, kind="stable")
    out = np.empty((pos.shape[0], len(units)), np.float32)
    out[:, o] = unit_profile(pos, neg, ps, ns, units[o], pos.shape[1])
    return out


def colcorr(A, B):
    A = (A - A.mean(0)) / (A.std(0) + 1e-9)
    B = (B - B.mean(0)) / (B.std(0) + 1e-9)
    return (A * B).mean(0)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if np.isnan(x) else round(x, 3) if abs(x) >= 1 else float(f"{x:.4g}")
    return x


# ------------------------------------------------------------------ F1 chance
def chance(S, n69, rng, reps):
    """Units present in >= k of 32 works: Batman vs 32 random ordinary films, same 37-film bar in each draw."""
    U = S.shape[1]
    ks = (MIN_WORKS, 23, 29)
    counts = []
    for _ in range(reps):
        perm = rng.permutation(FILMS)
        b = bar(S, perm[32:])
        n_p, n_b = (S[perm[:32]] > b).sum(0), (S[BAT] > b).sum(0)
        counts.append([[(n_p >= k).sum(), (n_b >= k).sum()] for k in ks])
    counts = np.array(counts)                                  # [reps, k, (pseudo, batman)]
    out = {"n_units": U, "reps": reps}
    for i, k in enumerate(ks):
        p, b = counts[:, i, 0], counts[:, i, 1]
        out[f"present_in_ge{k}_of_32"] = {
            "batman_bar69": int((n69 >= k).sum()), "binomial_chance_bar69": U * binom.sf(k - 1, 32, 2 / 70),
            "binomial_chance_bar37": U * binom.sf(k - 1, 32, 2 / 38),
            "pseudo_franchise_bar37": {"mean": p.mean(), "min": p.min(), "max": p.max()},
            "batman_bar37": {"mean": b.mean(), "min": b.min(), "max": b.max()},
            "batman_over_pseudo_mean": b.mean() / p.mean() if p.mean() else None,
            "draws_batman_above_pseudo": int((b > p).sum()), "batman_min_above_pseudo_max": bool(b.min() > p.max())}
    return out


# ------------------------------------------------------------------ F2 agreement
def label_class(u):
    ks = [KMAP[d[str(u)]["kind"]] for d in (labels, labels_wf) if str(u) in d]
    if not ks:
        return None
    if len(set(ks)) == 1:
        return ks[0]
    if "name" in ks:                      # labellers disagree and one says name: count it as a name
        return "name"
    return KMAP[labels[str(u)]["kind"]] if str(u) in labels else ks[0]


def kurtosis(X, chunk=2000):
    out = np.empty(X.shape[1], np.float32)
    for c in range(0, X.shape[1], chunk):
        Y = X[:, c:c + chunk]
        Z = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)
        out[c:c + chunk] = (Z ** 4).mean(0) - 3
    return out


def best_word(X, B, chunk=1000):
    """Best correlation of each profile with the presence of one word or bigram, and that term's index."""
    p = np.asarray(B.mean(0)).ravel()
    sb = np.sqrt(p * (1 - p)) + 1e-9
    r, j = np.empty(X.shape[1], np.float32), np.empty(X.shape[1], np.int64)
    for c in range(0, X.shape[1], chunk):
        Y = X[:, c:c + chunk]
        C = np.asarray(B.T @ ((Y - Y.mean(0)) / (Y.std(0) + 1e-9))) / B.shape[0] / sb[:, None]
        j[c:c + chunk], r[c:c + chunk] = C.argmax(0), C.max(0)
    return r, j


def presence(prof, sw):
    S = np.zeros((len(works), prof.shape[1]), np.float32)
    for w in range(len(works)):
        rows = np.where(sw == w)[0]
        k = min(3, len(rows))
        S[w] = np.partition(prof[rows], -k, axis=0)[-k:].mean(0)
    return S > bar(S, FILMS)


def kappa(A, B):
    a, b = A[LIN].ravel(), B[LIN].ravel()
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return {"agree": po, "chance": pe, "kappa": (po - pe) / (1 - pe)}


def retention(qo, qa):
    """Name-free activation on each unit's own top-20 original sentences, as a share of the original elevation."""
    out = []
    for j in range(qo.shape[1]):
        top = np.argsort(-qo[:, j], kind="stable")[:20]
        med = np.median(qo[:, j])
        out.append((qa[top, j].mean() - med) / (qo[top, j].mean() - med + 1e-9))
    return np.array(out)


def agreement(nlin, cand, rng):
    fq = families("qwen3-8b")
    order = np.argsort(-np.array([f["strength"] for f in fq]), kind="stable")[:NM]
    units_m = np.array([fq[k]["unit"] for k in order])
    aq = load_units(ACTS / "qwen3-8b")
    N, hidden, sw = aq[1].shape[1], aq[0]["hidden"], np.asarray(aq[0]["sent_work"])
    active = nlin >= 3
    active[cand] = False
    pool = np.sort(rng.choice(np.where(active)[0], N_POOL, replace=False))
    prof_pool = profiles(aq, pool)
    keep = prof_pool.std(0) > 0
    pool, prof_pool = pool[keep], prof_pool[:, keep]
    prof_m = profiles(aq, units_m)
    print(f"  pool {len(pool)} active non-candidate units", flush=True)

    cv = CountVectorizer(ngram_range=(1, 2), min_df=2, binary=True)
    B = cv.fit_transform(sentences).astype(np.float32).tocsc()
    vocab = cv.get_feature_names_out()
    bw_m, jw_m = best_word(prof_m, B)
    bw_p, _ = best_word(prof_pool, B)
    lk_m, lk_p = (np.log(np.maximum(kurtosis(X) + 3, 1e-3)) for X in (prof_m, prof_pool))
    lay_m, lay_p = (units_m % N) // hidden, (pool % N) // hidden

    def pick(d, n_near):
        return rng.choice(np.argsort(d, kind="stable")[:n_near], 3, replace=False)

    picks = {g: [] for g in NULLS}
    for i in range(NM):
        dk = np.abs(lk_p - lk_m[i])
        picks["sparsity"].append(pick(dk, 50))
        picks["sparsity_layer"].append(pick(np.where(np.abs(lay_p - lay_m[i]) <= 1, dk, np.inf), 25))
        picks["lexical"].append(pick(((bw_p - bw_m[i]) / bw_p.std()) ** 2 + (dk / lk_p.std()) ** 2, 15))
    sel = np.concatenate([np.ravel(picks[g]) for g in NULLS])
    units = np.concatenate([units_m, pool[sel]])
    bw = np.concatenate([bw_m, bw_p[sel]])
    Qo = np.concatenate([prof_m, prof_pool[:, sel]], 1)
    del prof_pool

    def blk(g):
        s = NM + 3 * NM * NULLS.index(g)
        return slice(s, s + 3 * NM)

    al = load_units(ACTS / "llama3.1-8b")
    assert al[0]["sent_work"] == aq[0]["sent_work"]
    print("  twin search, original texts", flush=True)
    r_o, tw = best_partner(Qo, *al[1:])
    Lo = profiles(al, tw)
    aqa, ala = load_units(ANON / "qwen3-8b"), load_units(ANON / "llama3.1-8b")
    Qa, La = profiles(aqa, units), profiles(ala, tw)
    r_af = colcorr(Qa, La)
    print("  twin search, name-free texts", flush=True)
    r_an, _ = best_partner(Qa, *ala[1:])
    ret = retention(Qo[:, :NM], Qa[:, :NM])
    grid = {"orig": (presence(Qo, sw), presence(Lo, sw)), "anon": (presence(Qa, sw), presence(La, sw))}
    twins_phylo = json.load(gzip.open(ROOT / "results/phylo.json.gz"))["models"]["llama3.1-8b"]["partners"]
    r = {"r_orig": r_o, "r_anon_fixed_twin": r_af, "r_anon_fresh_search": r_an}

    cls_read = [TOP33_CLASS.get(int(u)) for u in units_m]
    cls_lab = [label_class(u) for u in units_m]
    cls_lab[:33] = [c or rd for c, rd in zip(cls_lab[:33], cls_read[:33])]
    sets = {"top33": list(range(33))}
    sets.update({f"top33_{c}": [i for i in range(33) if cls_read[i] == c] for c in CLASSES})
    sets["named31"] = [i for i in range(NM) if str(units_m[i]) in labels]
    sets.update({f"labelled_{c}": [i for i in range(NM) if cls_lab[i] == c] for c in CLASSES})
    sets["top200"] = list(range(NM))
    assert len(sets["named31"]) == len(labels) == 31

    def wil(d):            # two-sided paired Wilcoxon
        return float(wilcoxon(d).pvalue) if len(d) >= 5 and np.any(d != 0) else None

    def summarise(idx):
        idx = np.asarray(idx)
        row = {"n": len(idx), "best_word_r": {"memes": np.median(bw[:NM][idx]),
                                              **{g: np.median(np.median(bw[blk(g)].reshape(NM, 3)[idx], 1)) for g in NULLS}},
               "name_free_retention_memes": np.median(ret[idx])}
        for key, v in r.items():
            m = v[:NM][idx]
            d = {"memes": np.median(m), "memes_share_gt_0.7": (m > 0.7).mean()}
            for g in NULLS:
                nv = np.median(v[blk(g)].reshape(NM, 3)[idx], 1)
                diff = m - nv
                d[g] = {"median": np.median(nv), "median_paired_diff": np.median(diff), "wins": int((diff > 0).sum()),
                        "wilcoxon_p": wil(diff)}
            row[key] = d
        for t, (Pq, Pl) in grid.items():
            row[f"grid_{t}"] = {"memes": kappa(Pq[:, idx], Pl[:, idx])}
            for g in NULLS:
                cols = blk(g).start + (idx[:, None] * 3 + np.arange(3)[None]).ravel()
                row[f"grid_{t}"][g] = kappa(Pq[:, cols], Pl[:, cols])
        return row

    per_meme = []
    for i in sorted(set(sets["top33"]) | set(sets["named31"]) | set(sets["labelled_concept"])):
        u = str(units_m[i])
        per_meme.append({"rank": i + 1, "unit": int(u), "label": (labels.get(u) or labels_wf.get(u) or {}).get("label"),
                         "class_reading": cls_read[i], "class_labels": cls_lab[i], "llama_twin": int(tw[i]),
                         **{k: v[i] for k, v in r.items()}, "best_word": str(vocab[jw_m[i]]), "best_word_r": bw_m[i],
                         "name_free_retention": ret[i],
                         "null_r_orig": {g: np.median(r_o[blk(g)].reshape(NM, 3)[i]) for g in NULLS}})
    return {"n_memes": NM, "pool_units": len(pool), "pool_best_word_r_median": np.median(bw_p),
            "twins_match_phylo_json": np.mean([twins_phylo[k]["unit"] == tw[i] for i, k in enumerate(order)]),
            "sets": {k: summarise(v) for k, v in sets.items() if v}, "per_meme": per_meme}


# ------------------------------------------------------------------ F2 trees
def resid(y, controls):
    if not controls:
        return y - y.mean()
    X = np.c_[np.ones(len(y)), np.column_stack([c[IU] for c in controls])]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def mantel(D1, D2, rng, controls=(), n_perm=N_PERM):
    """(Partial) Mantel r of two distance matrices over the lineage works, permuting the second's residuals."""
    a = resid(D1[IU], controls)
    R = np.zeros(D2.shape)
    R[IU] = resid(D2[IU], controls)
    R = R + R.T
    r = np.corrcoef(a, R[IU])[0, 1]
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(len(LIN))
        hits += np.corrcoef(a, R[p][:, p][IU])[0, 1] >= r
    return {"r": r, "p": (hits + 1) / (n_perm + 1)}


def source_ranks(D):
    """Rank of the best documented source among the comics, by distance from each adaptation (phylo.py)."""
    idx = {works[w]["label"]: i for i, w in enumerate(LIN)}
    comics = [i for i, w in enumerate(LIN) if works[w]["medium"] == "comic"]
    out = {}
    for t, srcs in ADAPTATIONS.items():
        order = sorted(comics, key=lambda c: D[idx[t], c])
        out[t] = min(order.index(idx[s]) + 1 for s in srcs)
    return out, len(comics)


def rank_chance(mean_rank, n):
    """Exact chance mean and one-sided p of the mean rank: each best rank is the minimum of k random distinct ranks."""
    dist = {0: 1.0}
    for srcs in ADAPTATIONS.values():
        k = len(srcs)
        pmf = {r: comb(n - r, k - 1) / comb(n, k) for r in range(1, n - k + 2)}
        new = {}
        for s, p in dist.items():
            for r, q in pmf.items():
                new[s + r] = new.get(s + r, 0) + p * q
        dist = new
    m = len(ADAPTATIONS)
    return sum(s * p for s, p in dist.items()) / m, sum(p for s, p in dist.items() if s <= mean_rank * m + 1e-9)


def recovery(D):
    ranks, n = source_ranks(D)
    v = np.array(list(ranks.values()))
    chance_mean, p = rank_chance(v.mean(), n)
    return {"mean_rank": v.mean(), "n_comics": n, "chance_mean_rank": chance_mean, "p_exact": p,
            "top1": int((v == 1).sum()), "top3": int((v <= 3).sum()), "ranks": ranks}


def trees(P_lin, cand, rng):
    D, nfam, nmem, reps = {}, {}, {}, {}
    for tag in MODELS:
        fam = families(tag)
        P = np.zeros((len(works), len(fam)), bool)
        for j, f in enumerate(fam):
            P[f["present"], j] = True
        D[tag], nfam[tag], nmem[tag] = jaccard(P[LIN]), len(fam), P[LIN].sum(1).astype(float)
        reps[tag] = np.array([f["unit"] for f in fam])
    T = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit_transform([" ".join(w["sentences"]) for w in works])
    D["tfidf"] = cosine_dist(T.toarray()[LIN])
    D["lsa"] = cosine_dist(TruncatedSVD(100, random_state=0).fit_transform(T)[LIN])
    Dq, Dl = D["qwen3-8b"], D["llama3.1-8b"]
    absd = lambda v: np.abs(v[:, None] - v[None, :])
    sums = lambda v: v[:, None] + v[None, :]
    ln, nq, nl = n_sent[LIN], nmem["qwen3-8b"], nmem["llama3.1-8b"]
    controls = {"tfidf": [D["tfidf"]], "lsa": [D["lsa"]],
                "length_and_meme_count_differences": [absd(ln), absd(nq), absd(nl)],
                "length_and_meme_count_differences_and_sums": [absd(ln), sums(ln), absd(nq), sums(nq), absd(nl), sums(nl)]}
    out = {"n_works": len(LIN), "n_pairs": len(IU[0]), "n_families": nfam,
           "per_work_meme_count_r": {"qwen_vs_llama": np.corrcoef(nq, nl)[0, 1], "qwen_vs_length": np.corrcoef(nq, ln)[0, 1]},
           "mantel": {"qwen_vs_llama_memes": mantel(Dq, Dl, rng), "qwen_memes_vs_tfidf": mantel(Dq, D["tfidf"], rng),
                      "llama_memes_vs_tfidf": mantel(Dl, D["tfidf"], rng), "tfidf_vs_lsa": mantel(D["tfidf"], D["lsa"], rng)},
           "partial_qwen_vs_llama_memes": {k: mantel(Dq, Dl, rng, cs) for k, cs in controls.items()}}
    print("  memes", {k: round(v["r"], 3) for k, v in out["mantel"].items()},
          {k: round(v["r"], 3) for k, v in out["partial_qwen_vs_llama_memes"].items()}, flush=True)

    rec = {"memes_qwen3-8b": recovery(Dq), "memes_llama3.1-8b": recovery(Dl), "tfidf": recovery(D["tfidf"]), "lsa": recovery(D["lsa"])}
    # random active units (present in >= 3 lineage works), as many as each model has families; two pools:
    # every discover.py candidate removed, or only the family representatives removed (phylo.py's pool)
    floor = {}
    for variant in ("non_candidate", "non_representative"):
        pools = {t: np.setdiff1d(np.where(P_lin[t].sum(0) >= 3)[0], cand[t] if variant == "non_candidate" else reps[t]) for t in MODELS}
        rs, rps, rk = [], [], {t: [] for t in MODELS}
        for _ in range(N_RAND):
            Dr = {t: jaccard(P_lin[t][:, np.sort(rng.choice(pools[t], nfam[t], replace=False))]) for t in MODELS}
            a, b = Dr["qwen3-8b"][IU], Dr["llama3.1-8b"][IU]
            rs.append(np.corrcoef(a, b)[0, 1])
            rps.append(np.corrcoef(resid(a, [D["tfidf"]]), resid(b, [D["tfidf"]]))[0, 1])
            for t in MODELS:
                rk[t].append(np.mean(list(source_ranks(Dr[t])[0].values())))
        floor[variant] = {"pool_sizes": {t: len(p) for t, p in pools.items()}, "draws": N_RAND,
                          "mantel_qwen_vs_llama": {"mean": np.mean(rs), "min": np.min(rs), "max": np.max(rs)},
                          "partial_given_tfidf": {"mean": np.mean(rps), "min": np.min(rps), "max": np.max(rps)}}
        for t in MODELS:
            v = np.array(rk[t])
            rec[f"random_{variant}_{t}"] = {"mean_rank_mean": v.mean(), "min": v.min(), "max": v.max(),
                                            "share_of_draws_better_than_memes": (v < rec[f"memes_{t}"]["mean_rank"]).mean()}
        print("  random", variant, round(np.mean(rs), 3), round(np.mean(rps), 3), flush=True)
    out["random_unit_floor"] = floor
    out["source_recovery"] = rec
    return out


# ------------------------------------------------------------------ F3 wording
def derangements(rng, m, n):
    out = []
    while len(out) < n:
        p = rng.permutation(m)
        if not (p == np.arange(m)).any():
            out.append(p)
    return np.array(out)


def wording(rng, n_null=10000):
    import wording_test as wt
    wt.ACTS = ACTS
    fq = families("qwen3-8b")
    fidx = {f["unit"]: i for i, f in enumerate(fq)}
    twins = json.load(gzip.open(ROOT / "results/phylo.json.gz"))["models"]["llama3.1-8b"]["partners"]
    wj = json.load(open(ROOT / "results/wording.json"))
    units_q = [int(u) for u in labels]
    assert [x["unit"] for x in wj["memes"]] == units_q
    names = [labels[str(u)]["label"] for u in units_q]
    m, nn, nt = len(units_q), len(wt.NEUTRAL), len(wt.TEST)
    batch = [wt.REWORDED[u] for u in units_q] + wt.NEUTRAL + [r for r, _ in wt.TEST] + [c for _, c in wt.TEST] + HARM + VOW
    dera = derangements(rng, m, n_null)
    out, rev_prof = {}, {}
    for tag, key, units in (("qwen3-8b", "qwen", units_q), ("llama3.1-8b", "llama", [twins[fidx[u]]["unit"] for u in units_q])):
        print("  running", tag, flush=True)
        A = wt.run_sentences(tag, batch)
        R, Nmax = A[:m][:, units], A[m:m + nn][:, units].max(0)        # R[i, j] = unit j on meme i's rewording
        prof = wt.corpus_profiles(tag, units)
        pct = lambda j, v: round(float((prof[:, j] < v).mean() * 100), 1)
        Rp = np.array([[pct(j, R[i, j]) for j in range(m)] for i in range(m)])
        Np = np.array([pct(j, Nmax[j]) for j in range(m)])
        res = {"max_abs_pct_diff_vs_wording_json": max(max(abs(Rp[j, j] - wj["memes"][j][key]["reworded"]),
                                                           abs(Np[j] - wj["memes"][j][key]["neutral_max"])) for j in range(m))}
        # percentile: the paper's rule (reworded percentile above the neutral maximum's); raw: activations compared directly
        for crit, M in (("percentile", Rp > Np[None, :]), ("raw", R > Nmax[None, :])):
            own = np.diag(M)
            null = M[dera, np.arange(m)].sum(1)
            res[crit] = {"survivors": int(own.sum()), "survivor_labels": [names[j] for j in np.where(own)[0]],
                         "rate_other_rewording_passes": ((M.sum(0) - own) / (m - 1)).mean(),
                         "mismatched_null_mean": null.mean(), "mismatched_null_p95": np.percentile(null, 95),
                         "p_mismatched_ge_observed": (np.sum(null >= own.sum()) + 1) / (n_null + 1)}
        own_rank = 1 + (R > np.diag(R)[None, :]).sum(0)          # rank of a unit's own rewording among all 31
        res["own_rewording_ranked_first"] = [names[j] for j in np.where(own_rank == 1)[0]]
        res["raw_survivors_ranked_first"] = [names[j] for j in np.where((own_rank == 1) & (np.diag(R) > Nmax))[0]]
        res["own_rewording_rank_median"] = np.median(own_rank)
        ru = wj["revenge"][tag]["chosen"]["unit"]
        o = m + nn
        tr, tc = A[o:o + nt, ru], A[o + nt:o + 2 * nt, ru]
        hm, vw = A[o + 2 * nt:o + 2 * nt + len(HARM), ru], A[o + 2 * nt + len(HARM):, ru]
        res["revenge"] = {"unit": ru, "n_candidates": wj["revenge"][tag]["n_candidates"], "pairs_won": int((tr > tc).sum()),
                          "n_pairs": nt, "auc_revenge_vs_control": wt.auc(tr, tc), "auc_revenge_vs_harm": wt.auc(tr, hm),
                          "auc_revenge_vs_vow": wt.auc(tr, vw), "auc_harm_vs_control": wt.auc(hm, tc)}
        rev_prof[tag] = wt.corpus_profiles(tag, [ru])[:, 0]
        out[tag] = res
        print("  ", tag, {c: (res[c]["survivors"], round(res[c]["mismatched_null_mean"], 2), round(res[c]["p_mismatched_ge_observed"], 3))
                          for c in ("percentile", "raw")}, flush=True)
        del A
    out["n_memes"] = m
    out["null_draws"] = n_null
    out["survive_in_both_percentile"] = sorted(set(out["qwen3-8b"]["percentile"]["survivor_labels"])
                                               & set(out["llama3.1-8b"]["percentile"]["survivor_labels"]))
    out["revenge_units_corpus_r"] = np.corrcoef(rev_prof["qwen3-8b"], rev_prof["llama3.1-8b"])[0, 1]
    return out


# ------------------------------------------------------------------ blind label test
STOP = set("""a an the and or of in on to for with by at from as is are was were be been his her their its
him them he she it they this that these plus mostly also who whom which what out into over under about
than then so not no""".split())


def stems(t):
    return {w[:5] for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP and len(w) > 2}


def import_labels(src):
    """Copy the review's blind test (labelprep/) and the judges' tally (checks.json labels.tally)."""
    lp = src / "labelprep"
    items = json.load(open(lp / "items.json"))
    det = {d["id"]: d for d in json.load(open(lp / "key_details.json"))["items"]}
    tally = json.load(open(src / "checks.json"))["labels"]["tally"]
    key = [{"id": k["id"], "answer": k["answer"], "group": k["group"], "unit": det[k["id"]]["unit"],
            "kind": det[k["id"]]["kind"], "high_rank": det[k["id"]]["high"]["rank"],
            "control_percentile": det[k["id"]]["control"]["percentile"], "steep_tail": det[k["id"]]["steep_tail"]}
           for k in json.load(open(lp / "key.json"))["key"]]
    return {"about": "Blind two-choice test of the Qwen3-8B meme labels. Each item gives a label and two corpus "
                     "sentences: one the unit fires on strongly (activation rank 9 to 30, never one the labeller saw) "
                     "and a control from the same group of works, length-matched, at or below the unit's median. "
                     "Groups: named31 = the 31 labels in results/labels.json; wf120 = 40 drawn at random from the "
                     "eligible labels in results/labels_wf.json. Built by the robustness review (seed 20261020); "
                     "judged blind by the review's workflow, which kept only the per-group tally.",
            "seed": items["seed"], "instructions_for_judge": items["instructions_for_judge"],
            "items": items["items"], "key": key, "judge_tally": tally}


def label_test(src):
    path = OUT / "label_test.json"
    if src:
        lt = import_labels(Path(src))
    elif path.exists():
        lt = json.load(open(path))
    else:
        print("  no label test: pass --import-labels <review folder>")
        return None
    items = {i["id"]: i for i in lt["items"]}
    lex, tie = {}, {}
    for k in lt["key"]:
        it, L = items[k["id"]], stems(items[k["id"]]["label"])
        sa, sb = len(L & stems(it["a"])), len(L & stems(it["b"]))
        tie[k["id"]] = sa == sb
        lex[k["id"]] = 0.5 if sa == sb else float(("a" if sa > sb else "b") == k["answer"])
    res = {}
    for g in ("named31", "wf120", "all"):
        ids = [k["id"] for k in lt["key"] if g == "all" or k["group"] == g]
        t = list(lt["judge_tally"].values()) if g == "all" else [lt["judge_tally"][g]]
        n, c = sum(x["n"] for x in t), sum(x["correct"] for x in t)
        assert n == len(ids)
        ci = binomtest(c, n).proportion_ci()
        acc_lex = float(np.mean([lex[i] for i in ids]))
        res[g] = {"n": n, "correct": c, "accuracy": c / n, "ci95_exact": [ci.low, ci.high],
                  "p_vs_chance": binomtest(c, n, 0.5, alternative="greater").pvalue,
                  "lexical_judge_accuracy": acc_lex, "lexical_judge_ties": sum(tie[i] for i in ids),
                  "p_vs_lexical_judge": binomtest(c, n, acc_lex, alternative="greater").pvalue,
                  "steep_tail_items": sum(k["steep_tail"] for k in lt["key"] if k["id"] in ids)}
    lt["results"] = clean(res)
    lt["lexical_judge"] = "picks the sentence sharing more content-word stems (first 5 letters) with the label; ties score 0.5"
    json.dump(lt, open(path, "w"), indent=1, ensure_ascii=False)
    return lt["results"]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Robustness nulls for F1-F3 and the blind label test.")
    ap.add_argument("--skip", default="", help="comma-separated parts to skip: chance, agreement, trees, wording, labels")
    ap.add_argument("--reps", type=int, default=50, help="pseudo-franchise draws per model (default 50)")
    ap.add_argument("--import-labels", metavar="DIR", help="robustness review folder holding labelprep/ and checks.json")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "nulls.json"
    res = json.load(open(path)) if path.exists() else {}

    def save():
        json.dump(clean(res), open(path, "w"), indent=1, ensure_ascii=False)

    need = MODELS if {"chance", "trees"} - skip else ["qwen3-8b"] if "agreement" not in skip else []
    P_lin, cand = {}, {}
    for i, tag in enumerate(need):
        print("== scores", tag, flush=True)
        S = all_scores(ACTS, tag)
        P = S > bar(S, FILMS)
        n69 = P[BAT].sum(0)
        cand[tag], P_lin[tag] = np.where(n69 >= MIN_WORKS)[0], P[LIN]
        del P
        if "chance" not in skip:
            c = chance(S, n69, np.random.default_rng(SEED + i), a.reps)
            f = ROOT / "results" / tag / "cand_units.npy"
            c["candidates_match_discover"] = bool(np.array_equal(np.sort(np.load(f)), cand[tag])) if f.exists() else None
            res.setdefault("f1_chance", {})[tag] = c
            print("  >=6:", c["present_in_ge6_of_32"]["batman_bar37"]["mean"], "vs pseudo",
                  c["present_in_ge6_of_32"]["pseudo_franchise_bar37"]["mean"], flush=True)
        del S
    if "chance" not in skip:
        res["f1_chance"]["n_families"] = {t: len(families(t)) for t in (*MODELS, "qwen3-1.7b")}
        save()
    if "agreement" not in skip:
        print("== agreement", flush=True)
        res["f2_agreement"] = agreement(P_lin["qwen3-8b"].sum(0), cand["qwen3-8b"], np.random.default_rng(SEED + 10))
        save()
    if "trees" not in skip:
        print("== trees", flush=True)
        res["f2_trees"] = trees(P_lin, cand, np.random.default_rng(SEED + 20))
        save()
    del P_lin
    if "wording" not in skip:
        print("== wording", flush=True)
        res["f3_wording"] = wording(np.random.default_rng(SEED + 30))
        save()
    if "labels" not in skip:
        print("== labels", flush=True)
        lt = label_test(a.import_labels)
        if lt:
            res["label_test"] = {"see": "results/robustness/label_test.json", **lt}
        save()
    print("wrote", path)


if __name__ == "__main__":
    main()
