"""How many placeholder codes the name-free texts hold, by group (a descriptive check added after the explanations
of the discriminator units were read; PLAN.md Section 18, change 29).

    cd study && ../.venv/bin/python v2/analysis/code_counts.py

The name-free texts replace every proper name with a code (X1, X2, ...). If Batman's summaries hold more names than
the hero films', then anything that reacts to the codes or to the gaps they leave separates the two groups without
knowing anything about Batman. Writes results/v2/code_counts.json.
"""
import numpy as np

import common as C

import experiments as X


def auc(a, b):
    return float(np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b]))


def main():
    occ = np.array([sum(len(X.CODE.findall(s)) for s in w["sentences"]) for w in X.neutral], float)
    words = np.array([sum(len(s.split()) for s in w["sentences"]) for w in X.neutral], float)
    distinct = np.array([len({m for s in w["sentences"] for m in X.CODE.findall(s)}) for w in X.neutral], float)
    groups = {"batman_works": X.BAT, "hero_films": X.HERO_FILM_COUSINS, "literary_relatives": X.LIT_COUSINS,
              "ordinary_films": np.setdiff1d(np.arange(len(X.neutral)), np.concatenate([X.BAT, X.HERO_FILM_COUSINS, X.LIT_COUSINS]))}
    out = {"what": "placeholder codes in the name-free texts (whole summaries)", "groups": {}}
    for name, idx in groups.items():
        out["groups"][name] = {"n": int(len(idx)), "code_occurrences_median": float(np.median(occ[idx])), "distinct_codes_median": float(np.median(distinct[idx])),
                               "words_median": float(np.median(words[idx])), "codes_per_100_words_median": round(float(np.median(100 * occ[idx] / words[idx])), 2)}
    b, h = X.BAT, X.HERO_FILM_COUSINS
    out["auc_batman_works_against_hero_films"] = {"code_occurrences": auc(occ[b], occ[h]), "distinct_codes": auc(distinct[b], distinct[h]),
                                                  "codes_per_word": auc(occ[b] / words[b], occ[h] / words[h]), "words": auc(words[b], words[h])}
    C.save(out, "code_counts.json")
    print({k: v for k, v in out["auc_batman_works_against_hero_films"].items()}, {k: v["code_occurrences_median"] for k, v in out["groups"].items()})


if __name__ == "__main__":
    main()
