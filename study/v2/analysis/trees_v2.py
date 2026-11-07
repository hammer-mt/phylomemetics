"""Family trees from the v2 meme families, with v1's tests (PLAN.md, Section 11).

Distances are Jaccard distances between works' sets of meme families (a family counts as present when its
representative unit is), over the 32 Batman works and 13 relatives; trees are neighbour-joining, rooted on Puss in
Boots, as in v1 (functions imported from phylo.py). Reported per model: clustering by era, whether the documented
source comic is among the nearest comics, and the Mantel correlation with every other model's distances and with
TF-IDF distances.

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


def family_presence(tag):
    fams = json.load(gzip.open(C.RESULTS / tag / "families.json.gz", "rt"))["families"]
    P = np.zeros((len(works), len(fams)), dtype=bool)
    for j, f in enumerate(fams):
        P[f["present"], j] = True
    return P[keep]


def main():
    tags = [t for t in C.MODELS if (C.RESULTS / t / "families.json.gz").exists()]
    out, D = {"works": names, "models": {}}, {}
    X = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit_transform([" ".join(works[i]["sentences"]) for i in keep]).toarray()
    D["tfidf"] = PH.cosine_dist(X)
    eras = [works[i]["era"] for i in keep]
    medium = [works[i]["medium"] for i in keep]
    bat = [k for k, i in enumerate(keep) if works[i]["group"] == "batman"]
    for tag in tags:
        P = family_presence(tag)
        D[tag] = PH.jaccard(P)
        Db = D[tag][np.ix_(bat, bat)]
        ratio, p = PH.era_signal(Db, [eras[k] for k in bat])
        out["models"][tag] = {"n_families": int(P.shape[1]), "families_per_work_median": float(np.median(P.sum(1))),
                              "newick": PH.nj_tree(D[tag], names, outgroup="Puss in Boots", ids=keep),
                              "era_between_over_within": ratio, "era_p": p,
                              "source_by_distance": PH.adaptation_test(D[tag], names, medium)}
    out["tfidf_source_by_distance"] = PH.adaptation_test(D["tfidf"], names, medium)
    out["mantel"] = {}
    keys = tags + ["tfidf"]
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            r, p = PH.mantel(D[keys[a]], D[keys[b]])
            out["mantel"][f"{keys[a]} vs {keys[b]}"] = {"r": r, "p": p}
    C.save(out, "trees.json")
    for tag in tags:
        s = out["models"][tag]["source_by_distance"]
        print(tag, "source mean rank", round(s["mean_rank"], 2), "chance", round(s["chance_mean_rank"], 2), "| era ratio", round(out["models"][tag]["era_between_over_within"], 3))


if __name__ == "__main__":
    main()
