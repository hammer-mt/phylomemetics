"""Family trees from the v2 meme families, with v1's tests and two controls it lacked (PLAN.md, Section 13).

Distances are Jaccard distances between works' sets of meme families (a family counts as present when its
representative unit is), over the 32 Batman works and 13 relatives; trees are neighbour-joining, rooted on Puss in
Boots, as in v1 (functions imported from phylo.py). Reported per model: clustering by era, and whether the
documented source comic is among an adaptation's nearest comics, against the same number of random units that are
not candidates and against TF-IDF. Between models: the Mantel correlation of the two distance matrices, the same
for random non-candidate units (the floor any two models reach on these texts), and the partial correlation with
TF-IDF distance taken out of both.

What v1 called "tree distance" is the distance between works; no tree is involved in these tests.

    cd study && ../.venv/bin/python v2/analysis/trees_v2.py

Writes results/v2/trees.json.
"""
import gzip
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import common as C

import phylo as PH

works = [json.loads(l) for l in open("data/corpus.jsonl")]
keep = [w["id"] for w in works if w["group"] in ("batman", "cousin")]
names = [works[i]["label"] for i in keep]
N_RANDOM = 20


def family_presence(tag):
    fams = json.load(gzip.open(C.RESULTS / tag / "families.json.gz", "rt"))["families"]
    P = np.zeros((len(works), len(fams)), dtype=bool)
    for j, f in enumerate(fams):
        P[f["present"], j] = True
    return P[keep]


def random_presence(tag, n, rng):
    """N_RANDOM presence matrices of n random units that are present in one of these works and are not candidates."""
    st = C.Setting(tag, icon=False)
    cand = set(np.load(C.RESULTS / tag / "pools.npz")["candidates"].tolist())
    P = st.P["orig"][keep]
    ok = np.flatnonzero(P.any(0) & ~np.isin(st.units, list(cand)))
    return [P[:, rng.choice(ok, min(n, len(ok)), replace=False)] for _ in range(N_RANDOM)]


def partial_mantel(D1, D2, Dc, rng, n_perm=2000):
    """Correlation of D1 and D2 with Dc regressed out of both; p by permuting the works of D1."""
    def resid(y, x):
        A = np.c_[np.ones(len(x)), x]
        return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    c = PH.upper(Dc)
    b = resid(PH.upper(D2), c)
    r = float(np.corrcoef(resid(PH.upper(D1), c), b)[0, 1])
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(len(D1))
        hits += np.corrcoef(resid(PH.upper(D1[p][:, p]), c), b)[0, 1] >= r
    return r, (hits + 1) / (n_perm + 1)


def main():
    rng = np.random.default_rng(C.SEED)
    tags = [t for t in C.MODELS if (C.RESULTS / t / "families.json.gz").exists()]
    out, D, R = {"works": names, "models": {}}, {}, {}
    T = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit_transform([" ".join(works[i]["sentences"]) for i in keep]).toarray()
    D["tfidf"] = PH.cosine_dist(T)
    eras = [works[i]["era"] for i in keep]
    medium = [works[i]["medium"] for i in keep]
    bat = [k for k, i in enumerate(keep) if works[i]["group"] == "batman"]
    for tag in tags:
        P = family_presence(tag)
        D[tag] = PH.jaccard(P)
        R[tag] = [PH.jaccard(Q) for Q in random_presence(tag, P.shape[1], rng)]
        ratio, p = PH.era_signal(D[tag][np.ix_(bat, bat)], [eras[k] for k in bat])
        rand = [PH.adaptation_test(Dr, names, medium) for Dr in R[tag]]
        out["models"][tag] = {"n_families": int(P.shape[1]), "families_per_work_median": float(np.median(P.sum(1))),
                              "newick": PH.nj_tree(D[tag], names, outgroup="Puss in Boots", ids=keep),
                              "era_between_over_within": ratio, "era_p": p,
                              "source_by_distance": PH.adaptation_test(D[tag], names, medium),
                              "source_by_distance_random_units": {"n_draws": N_RANDOM, "mean_rank_mean": float(np.mean([r["mean_rank"] for r in rand])),
                                                                  "mean_rank_min": float(np.min([r["mean_rank"] for r in rand])),
                                                                  "top1_mean": float(np.mean([r["top1"] for r in rand]))}}
    out["tfidf_source_by_distance"] = PH.adaptation_test(D["tfidf"], names, medium)
    out["mantel"] = {}
    for a in range(len(tags)):
        for b in range(a + 1, len(tags)):
            ta, tb = tags[a], tags[b]
            r, p = PH.mantel(D[ta], D[tb])
            floor = [float(np.corrcoef(PH.upper(x), PH.upper(y))[0, 1]) for x, y in zip(R[ta], R[tb])]
            pr, pp = partial_mantel(D[ta], D[tb], D["tfidf"], rng)
            out["mantel"][f"{ta} vs {tb}"] = {"r": r, "p": p, "random_units_r_mean": float(np.mean(floor)), "random_units_r_max": float(np.max(floor)),
                                              "partial_r_given_tfidf": pr, "partial_p": pp}
    for tag in tags:
        r, p = PH.mantel(D[tag], D["tfidf"])
        out["mantel"][f"{tag} vs tfidf"] = {"r": r, "p": p}
    C.save(out, "trees.json")
    for tag in tags:
        s = out["models"][tag]
        print(tag, "source mean rank", round(s["source_by_distance"]["mean_rank"], 2), "random units", round(s["source_by_distance_random_units"]["mean_rank_mean"], 2),
              "chance", round(s["source_by_distance"]["chance_mean_rank"], 2), "| era ratio", round(s["era_between_over_within"], 3))


if __name__ == "__main__":
    main()
