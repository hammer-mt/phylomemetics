"""Stage 2: build family trees from meme presence, test them, and compare models.

For each model: works x memes presence matrix -> Jaccard distances -> neighbour-joining tree.
Tests:
  era signal      are works from the same era closer than works from different eras? (permutation)
  time signal     does meme distance grow with the gap in years? (Mantel)
  baselines       the same tests on word-overlap (TF-IDF) distances and on random neuron sets
  cross-model     do two models draw the same tree (Mantel between distance matrices), and does
                  each Qwen meme have a matching neuron in Llama (best sentence-level correlation)?
"""
import json
import sys
from pathlib import Path

import numpy as np
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio import Phylo
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer

from discover import load_units, work_scores

RNG = np.random.default_rng(0)
N_PERM = 5000


def jaccard(P):
    P = P.astype(float)
    inter = P @ P.T
    union = P.sum(1)[:, None] + P.sum(1)[None, :] - inter
    D = 1 - inter / np.maximum(union, 1)
    np.fill_diagonal(D, 0)
    return D


def cosine_dist(X):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    D = 1 - X @ X.T
    np.fill_diagonal(D, 0)
    return np.clip(D, 0, 2)


def upper(D):
    i, j = np.triu_indices(len(D), 1)
    return D[i, j]


def mantel(D1, D2, n_perm=N_PERM):
    a, b = upper(D1), upper(D2)
    r = np.corrcoef(a, b)[0, 1]
    n, hits = len(D1), 0
    for _ in range(n_perm):
        p = RNG.permutation(n)
        hits += np.corrcoef(a, upper(D2[p][:, p]))[0, 1] >= r
    return float(r), (hits + 1) / (n_perm + 1)


def era_signal(D, eras, n_perm=N_PERM):
    """Mean between-era distance / mean within-era distance (>1 means eras cluster)."""
    eras = np.asarray(eras)
    iu = np.triu_indices(len(D), 1)

    def ratio(e):
        same = (e[:, None] == e[None, :])[iu]
        d = D[iu]
        return d[~same].mean() / d[same].mean()

    obs = ratio(eras)
    null = np.array([ratio(RNG.permutation(eras)) for _ in range(n_perm)])
    return float(obs), float((np.sum(null >= obs) + 1) / (n_perm + 1))


def nn_same_era(D, eras):
    """Share of works whose nearest neighbour is from the same era."""
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    eras = np.asarray(eras)
    return float((eras[nn] == eras).mean())


# Documented sources: the comic each adaptation says it drew on (target -> sources).
ADAPTATIONS = {
    "Batman Begins": ["Year One"],
    "The Dark Knight": ["The Long Halloween", "The Killing Joke"],
    "Dark Knight Rises": ["Knightfall", "Dark Knight Returns"],
    "Batman v Superman": ["Dark Knight Returns"],
    "Under the Red Hood": ["A Death in the Family"],
    "Arkham Asylum (game)": ["Arkham Asylum (comic)"],
    "Batman '89": ["The Killing Joke"],
    "Joker (2019)": ["The Killing Joke"],
    "The Batman": ["Year One", "The Long Halloween"],
}


def adaptation_test(D, names, works_medium):
    """For each adaptation, rank all comics by distance; report where its documented source lands."""
    idx = {n: i for i, n in enumerate(names)}
    comics = [i for i, n in enumerate(names) if works_medium[i] == "comic"]
    rows = []
    for target, sources in ADAPTATIONS.items():
        t = idx[target]
        order = sorted(comics, key=lambda c: D[t, c])
        ranks = [order.index(idx[s]) + 1 for s in sources]
        rows.append({"target": target, "sources": sources, "rank": min(ranks),
                     "closest": names[order[0]], "n_comics": len(comics)})
    mean_rank = float(np.mean([r["rank"] for r in rows]))
    # chance: the best of k random ranks out of n comics
    sim = [np.mean([min(RNG.choice(len(comics), size=len(r["sources"]), replace=False) + 1) for r in rows])
           for _ in range(N_PERM)]
    p = (np.sum(np.array(sim) <= mean_rank) + 1) / (N_PERM + 1)
    return {"rows": rows, "mean_rank": mean_rank, "chance_mean_rank": float(np.mean(sim)), "p": float(p),
            "top1": sum(r["rank"] == 1 for r in rows), "top3": sum(r["rank"] <= 3 for r in rows)}


def nj_tree(D, names, outgroup=None, ids=None):
    """Neighbour-joining tree as Newick. Leaves are named w<work id> to keep the Newick parseable."""
    m = [[float(D[i, j]) for j in range(i + 1)] for i in range(len(D))]
    leaf = [f"w{i}" for i in ids]
    outgroup = leaf[names.index(outgroup)] if outgroup else None
    tree = DistanceTreeConstructor().nj(DistanceMatrix(leaf, m))
    if outgroup:
        tree.root_with_outgroup(outgroup)
    tree.ladderize()
    s = StringIO()
    Phylo.write(tree, s, "newick")
    return s.getvalue().strip()


def model_presence(tag, results, works):
    fam = json.load(open(Path(results) / tag / "memes.json"))["families"]
    keep = [f for f in fam if f.get("keep", True)]
    P = np.zeros((len(works), len(keep)), dtype=bool)
    for j, f in enumerate(keep):
        P[f["present"], j] = True
    return P, keep


def best_partner(prof_q, B_pos, B_neg, B_ps, B_ns, chunk=16384, shuffle=None):
    """For each column of prof_q [sent, K], best Pearson r over all units (both poles) of model B."""
    Zq = (prof_q - prof_q.mean(0)) / (prof_q.std(0) + 1e-6)
    if shuffle is not None:
        Zq = Zq[shuffle]
    n, K = Zq.shape
    best_r = np.full(K, -1.0, dtype=np.float32)
    best_u = np.zeros(K, dtype=np.int64)
    N = B_pos.shape[1]
    for arr, sc, sign, off in ((B_pos, B_ps, 1, 0), (B_neg, B_ns, -1, N)):
        for c in range(0, N, chunk):
            X = sign * arr[:, c:c + chunk].astype(np.float32) * sc[c:c + chunk]
            Zx = (X - X.mean(0)) / (X.std(0) + 1e-6)
            R = (Zq.T @ Zx) / n                       # [K, chunk]
            j = R.argmax(1)
            r = R[np.arange(K), j]
            better = r > best_r
            best_r[better] = r[better]
            best_u[better] = off + c + j[better]
    return best_r, best_u


def unit_profile(arr_pos, arr_neg, ps, ns, units, n_neurons):
    prof = np.zeros((arr_pos.shape[0], len(units)), dtype=np.float32)
    for k, u in enumerate(units):
        if u < n_neurons:
            prof[:, k] = arr_pos[:, u].astype(np.float32) * ps[u]
        else:
            prof[:, k] = -arr_neg[:, u - n_neurons].astype(np.float32) * ns[u - n_neurons]
    return prof


def main(acts_root, results, primary="qwen3-8b", others=("llama3.1-8b", "qwen3-1.7b")):
    works = [json.loads(l) for l in open("data/corpus.jsonl")]
    sentences = [s for w in works for s in w["sentences"]]
    group = np.array([w["group"] for w in works])
    lineage = np.where(group != "baseline")[0]          # Batman works + cousins
    bat = np.where(group == "batman")[0]
    names = [works[i]["label"] for i in lineage]
    eras_bat = [works[i]["era"] for i in bat]
    years_bat = np.array([works[i]["year"] for i in bat], dtype=float)
    Dyear = np.abs(years_bat[:, None] - years_bat[None, :])
    out = {"works": [{k: works[i][k] for k in ("id", "label", "title", "year", "era", "medium", "group")}
                     for i in lineage], "models": {}}

    # word-overlap baseline: TF-IDF over the plot text
    tfidf = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)
    T = tfidf.fit_transform([" ".join(w["sentences"]) for w in works]).toarray()
    D_tfidf = cosine_dist(T[lineage])
    D_tfidf_bat = cosine_dist(T[bat])
    medium = [works[i]["medium"] for i in lineage]
    out["tfidf"] = {"era_ratio": era_signal(D_tfidf_bat, eras_bat), "nn_same_era": nn_same_era(D_tfidf_bat, eras_bat),
                    "mantel_year": mantel(D_tfidf_bat, Dyear), "adaptation": adaptation_test(D_tfidf, names, medium),
                    "newick": nj_tree(D_tfidf, names, outgroup="Puss in Boots", ids=lineage)}
    print("TF-IDF:", {k: v for k, v in out["tfidf"].items() if k != "newick"})

    dists = {}
    for tag in (primary, *others):
        if not (Path(results) / tag / "memes.json").exists():
            continue
        P, fams = model_presence(tag, results, works)
        D = jaccard(P[lineage])
        Db = jaccard(P[bat])
        dists[tag] = D
        res = {"n_memes": len(fams), "era_ratio": era_signal(Db, eras_bat), "nn_same_era": nn_same_era(Db, eras_bat),
               "mantel_year": mantel(Db, Dyear), "mantel_tfidf": mantel(D, D_tfidf),
               "adaptation": adaptation_test(D, names, medium),
               "newick": nj_tree(D, names, outgroup="Puss in Boots", ids=lineage),
               "D": np.round(D, 3).tolist()}
        # random-neuron control: same number of units, picked at random from all units
        meta, pos, neg, ps, ns = load_units(Path(acts_root) / tag)
        n_neurons = pos.shape[1]
        S = np.concatenate([work_scores(pos, ps, meta["sent_work"], len(works), 1),
                            work_scores(neg, ns, meta["sent_work"], len(works), -1)], axis=1)
        base = np.where(group == "baseline")[0]
        thr = np.sort(S[base], axis=0)[-2]
        Pall = S > thr
        # control pool: neurons just as "alive" in these texts (present in 3+ lineage works) but not picked as memes
        meme_units = np.array(sorted({f["unit"] for f in fams}))
        active = np.setdiff1d(np.where(Pall[lineage].sum(0) >= 3)[0], meme_units)
        if tag == primary:
            active_primary = active
        ratios, nns, adapt = [], [], []
        for _ in range(50):
            pick = RNG.choice(active, size=len(fams), replace=False)
            Dr = jaccard(Pall[bat][:, pick])
            Dr = np.nan_to_num(Dr, nan=1.0)
            iu = np.triu_indices(len(bat), 1)
            e = np.asarray(eras_bat)
            same = (e[:, None] == e[None, :])[iu]
            ratios.append(Dr[iu][~same].mean() / Dr[iu][same].mean())
            nns.append(nn_same_era(Dr, eras_bat))
            adapt.append(adaptation_test(np.nan_to_num(jaccard(Pall[lineage][:, pick]), nan=1.0), names, medium)["mean_rank"])
        res["random_units"] = {"era_ratio_mean": float(np.mean(ratios)), "nn_same_era_mean": float(np.mean(nns)),
                               "adaptation_mean_rank": float(np.mean(adapt))}
        out["models"][tag] = res
        print(tag, {k: v for k, v in res.items() if k not in ("newick", "D")})
        del S, Pall

    # cross-model: does the other model draw the same tree?
    for tag in others:
        if tag in dists:
            out["models"][tag]["mantel_vs_primary"] = mantel(dists[primary], dists[tag])
            print(f"tree agreement {primary} vs {tag}:", out["models"][tag]["mantel_vs_primary"])

    # cross-model: does each primary meme have a matching neuron in the other model?
    _, fams_q = model_presence(primary, results, works)
    mq, qpos, qneg, qps, qns = load_units(Path(acts_root) / primary)
    nq = qpos.shape[1]
    units_q = [f["unit"] for f in fams_q]
    prof_q = unit_profile(qpos, qneg, qps, qns, units_q, nq)
    # null: other neurons that are just as active in this corpus, but weren't picked as Batman memes
    rand_units = RNG.choice(active_primary, size=len(units_q), replace=False)
    prof_rand = unit_profile(qpos, qneg, qps, qns, rand_units, nq)
    for tag in others:
        if not (Path(acts_root) / tag / "meta.json").exists():
            continue
        mb, bpos, bneg, bps, bns = load_units(Path(acts_root) / tag)
        bpos, bneg = np.asarray(bpos), np.asarray(bneg)
        nb = bpos.shape[1]
        r_meme, u_meme = best_partner(prof_q, bpos, bneg, bps, bns)
        r_rand, _ = best_partner(prof_rand, bpos, bneg, bps, bns)
        r_shuf, _ = best_partner(prof_q, bpos, bneg, bps, bns, shuffle=RNG.permutation(len(sentences)))
        # partner's own work-level presence, to compare against the Qwen meme's
        Sb = np.concatenate([work_scores(bpos, bps, mb["sent_work"], len(works), 1),
                             work_scores(bneg, bns, mb["sent_work"], len(works), -1)], axis=1)
        thr_b = np.sort(Sb[np.where(group == "baseline")[0]], axis=0)[-2]
        prof_b = unit_profile(bpos, bneg, bps, bns, u_meme, nb)
        partners = []
        for k, f in enumerate(fams_q):
            u = int(u_meme[k])
            pres_b = np.where(Sb[:, u] > thr_b[u])[0]
            a, b = set(f["present"]), set(pres_b.tolist())
            top = np.argsort(-prof_b[:, k])[:5]
            partners.append({"unit": u, "layer": int((u % nb) // mb["hidden"]), "pole": "-" if u >= nb else "+",
                             "r": round(float(r_meme[k]), 3),
                             "presence_jaccard": round(len(a & b) / max(len(a | b), 1), 3),
                             "present": sorted(b),
                             "top_sentences": [{"work": works[mb["sent_work"][i]]["label"], "text": sentences[i]}
                                               for i in top]})
        out["models"][tag]["partners"] = partners
        out["models"][tag]["partner_r"] = {"memes_median": float(np.median(r_meme)),
                                           "random_units_median": float(np.median(r_rand)),
                                           "shuffled_median": float(np.median(r_shuf)),
                                           "memes": r_meme.round(3).tolist(), "random": r_rand.round(3).tolist(),
                                           "shuffled": r_shuf.round(3).tolist()}
        print(f"{tag} partner r: memes {np.median(r_meme):.3f}  random units {np.median(r_rand):.3f}  "
              f"shuffled {np.median(r_shuf):.3f}")
        del bpos, bneg, Sb

    json.dump(out, open(Path(results) / "phylo.json", "w"))
    print("wrote", Path(results) / "phylo.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
