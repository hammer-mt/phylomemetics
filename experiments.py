"""Phase 2: three pre-registered experiments on neuron-defined memes (design revised after adversarial review).

Units are neuron poles; a unit is "present" in a work when the mean of the work's three strongest sentences
beats all but one of a set of reference ordinary films. Name-free texts (data/corpus_neutral.jsonl) replace
every name with a neutral code (X1, X2...).

E1 Identity. Primary: Qwen3-8B; canon = units present in >= 80% of the other 31 Batman works (original texts);
   the held-out work is scored on its NAME-FREE text by the share of canon units present. Reference films are
   split in half 20 times: one half sets the presence bar, the other half is scored. Primary comparison: Batman
   vs the 8 masked-hero and superhero film cousins. Prediction AUC >= 0.80, bootstrap 95% lower bound >= 0.65,
   above the 95th percentile of a label-shuffle null; holds within 0.05 after dropping sentences that still
   contain iconography words. Falsified if AUC < 0.70 or its CI includes 0.5.
E2 Heredity. Primary: Qwen3-8B name-free; optional units present in 2-16 of 32 Batman works; for adaptation A
   and comic C (published no later than A), rate = share of C's optional units also present in A, z-scored
   against the 11 non-adaptation screen works ("hub" adjustment). Statistic: share of eligible comics that beat
   the documented source (best of its sources), averaged over the 9 adaptations; joint comic-relabelling
   permutation, one-sided p < 0.05. Positive control: sequels recover their prequel (top 2 for >= 5 of 7) on
   name-free texts; if that fails the result is inconclusive.
E3 Selection. Primary: Qwen3-8B name-free; early = 15 Batman works before 2000, later = 17 from 2000 minus the
   8 adaptation-linked works (9 left). Units present in 2 to 80% of early works. Per early work, offspring =
   mean over its units of later_share residualised on early prevalence, then residualised on the work's year,
   plot length and centrality. Spearman with log mean monthly pageviews; permutation within medium (screen vs
   comic). Prediction rho > 0, p < 0.05 one-sided, same sign in Llama. Placebos: reverse time, baseline films.

Writes results/experiments.json.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from discover import load_units

RNG = np.random.default_rng(7)
ACTS = Path(os.environ.get("PHYLO_ACTS", "activations"))
ANON = Path(os.environ.get("PHYLO_ACTS_ANON", "activations_anon"))
MODELS = ["qwen3-8b", "llama3.1-8b"]

works = [json.loads(l) for l in open("data/corpus.jsonl")]
neutral = [json.loads(l) for l in open("data/corpus_neutral.jsonl")]
N = len(works)
group = np.array([w["group"] for w in works])
medium = np.array([w["medium"] for w in works])
year = np.array([w["year"] or 0 for w in works])
label = [w["label"] for w in works]
L = {w["label"]: w["id"] for w in works}
BAT = np.where(group == "batman")[0]
FILMS = np.where(group == "baseline")[0]
HERO_FILM_COUSINS = np.array([L[x] for x in ["The Bat (1926)", "Zorro", "The Shadow", "Superman", "Spider-Man", "Daredevil", "Iron Man", "Green Hornet"]])
LIT_COUSINS = np.array([L[x] for x in ["Puss in Boots", "Count of Monte Cristo", "Sherlock Holmes", "Dracula", "Scarlet Pimpernel"]])
ICON = re.compile(r"\b(bats?|batman|capes?|caped|cowl|clowns?|joker|gotham|wayne|crusader|knight|penguin|riddles?|riddler|caves?|signal|asylum|robin)\b", re.I)
n_sent = np.array([len(w["sentences"]) for w in works])

ADAPTATIONS = {
    "Batman Begins": ["Year One"], "The Dark Knight": ["The Long Halloween", "The Killing Joke"],
    "Dark Knight Rises": ["Knightfall", "Dark Knight Returns"], "Batman v Superman": ["Dark Knight Returns"],
    "Under the Red Hood": ["A Death in the Family"], "Arkham Asylum (game)": ["Arkham Asylum (comic)"],
    "Batman '89": ["The Killing Joke"], "Joker (2019)": ["The Killing Joke"], "The Batman": ["Year One", "The Long Halloween"],
}
STRONG = ["Batman Begins", "Under the Red Hood", "Batman v Superman", "Dark Knight Rises", "The Batman"]
COMICS = np.array([w["id"] for w in works if w["group"] == "batman" and w["medium"] == "comic"])
SCREEN_NONADAPT = np.array([L[x] for x in ["1943 serial", "Batman '66", "Batman Returns", "Batman Forever", "Batman & Robin", "Mask of the Phantasm",
                                            "Return of the Joker", "Arkham City", "Arkham Knight", "Lego Batman", "Batman Ninja"]])
SEQUELS = {"The Dark Knight": "Batman Begins", "Dark Knight Rises": "The Dark Knight", "Batman Returns": "Batman '89",
           "Batman Forever": "Batman Returns", "Batman & Robin": "Batman Forever", "Arkham City": "Arkham Asylum (game)",
           "Arkham Knight": "Arkham City"}
SCREEN = np.array([w["id"] for w in works if w["group"] == "batman" and w["medium"] != "comic"])


# ------------------------------------------------------------------ scores and presence
def work_scores_masked(root, tag, keep=None):
    """[N, 2U] mean of each work's 3 strongest sentences per unit; `keep` = boolean mask over sentences."""
    meta, pos, neg, ps, ns = load_units(root / tag)
    sw = np.asarray(meta["sent_work"])
    U = pos.shape[1]
    S = np.zeros((N, 2 * U), dtype=np.float32)
    for w in range(N):
        rows = np.where((sw == w) & (keep if keep is not None else True))[0]
        if len(rows) == 0:
            rows = np.where(sw == w)[0]
        k = min(3, len(rows))
        for arr, sc, sign, off in ((pos, ps, 1, 0), (neg, ns, -1, U)):
            block = sign * arr[rows].astype(np.float32)
            S[w, off:off + U] = np.partition(block, -k, axis=0)[-k:].mean(0) * sc
    return S


def bar(S, ref):
    """Presence bar per unit: second-highest score among the reference films."""
    return np.partition(S[ref], -2, axis=0)[-2]


def auc_matrix(pos, neg):
    p, q = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return (p > q) + 0.5 * (p == q)


def boot_ci(M, n=2000):
    """Bootstrap 95% CI of mean(M) resampling rows and columns."""
    r, c = M.shape
    vals = [M[RNG.integers(0, r, r)][:, RNG.integers(0, c, c)].mean() for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ------------------------------------------------------------------ E1
def e1(S_o, S_a, S_a_ablate, n_splits=20, n_null=200):
    splits = []
    for _ in range(n_splits):
        perm = RNG.permutation(FILMS)
        splits.append((perm[:34], perm[34:]))

    def run(franchise, neg_fns, S_canon, S_score, n_use=n_splits):
        """Mean AUC matrices (franchise works x negatives) for each negative set, averaged over baseline splits."""
        mats = {k: [] for k in neg_fns}
        for h1, h2 in splits[:n_use]:
            Pc = S_canon > bar(S_canon, h1)
            Ps = Pc if S_score is S_canon else S_score > bar(S_score, h1)
            negs = {k: fn(h2) for k, fn in neg_fns.items()}
            rows = {k: [] for k in neg_fns}
            for b in franchise:
                others = franchise[franchise != b]
                C = np.where(Pc[others].mean(0) >= 0.8)[0]
                for k, ng in negs.items():
                    if len(C) == 0:
                        rows[k].append(np.full(len(ng), 0.5))
                    else:
                        rows[k].append(auc_matrix([Ps[b, C].mean()], Ps[ng][:, C].mean(1))[0])
            for k in neg_fns:
                mats[k].append(np.array(rows[k]))
        return {k: np.mean(v, axis=0) for k, v in mats.items()}

    res = {}
    NEG = {"hero_film_cousins": lambda h2: HERO_FILM_COUSINS, "held_out_films": lambda h2: h2, "literary_cousins": lambda h2: LIT_COUSINS}
    for name, S_canon, S_score in (("A_name_free", S_o, S_a), ("B_canon_from_name_free", S_a, S_a), ("ceiling_original", S_o, S_o),
                                   ("A_iconography_dropped", S_o, S_a_ablate)):
        Ms = run(BAT, NEG, S_canon, S_score)
        out = {k: {"auc": float(M.mean()), "ci95": boot_ci(M)} for k, M in Ms.items()}
        if name == "A_name_free":
            M = Ms["hero_film_cousins"]
            out["per_batman_vs_hero"] = {label[b]: round(float(M[i].mean()), 3) for i, b in enumerate(BAT)}
            out["per_cousin_beaten_by"] = {label[c]: round(float(1 - M[:, j].mean()), 3) for j, c in enumerate(HERO_FILM_COUSINS)}
        res[name] = out
        print("  E1", name, {k: round(v["auc"], 3) for k, v in out.items() if isinstance(v, dict) and "auc" in v}, flush=True)

    # label-shuffle null: random 32 of the 40 (Batman + hero film cousins) play the franchise, 8 the cousins
    pool = np.concatenate([BAT, HERO_FILM_COUSINS])
    null = []
    for i in range(n_null):
        p = RNG.permutation(pool)
        fr, ng = p[:32], p[32:]
        M = run(fr, {"x": lambda h2, ng=ng: ng}, S_o, S_a, n_use=1)["x"]
        null.append(float(M.mean()))
    res["null_label_shuffle"] = {"mean": float(np.mean(null)), "p95": float(np.percentile(null, 95)),
                                 "p_value": float((np.sum(np.array(null) >= res["A_name_free"]["hero_film_cousins"]["auc"]) + 1) / (n_null + 1))}
    # length-only control
    res["length_only_auc_vs_hero"] = float(auc_matrix(n_sent[BAT], n_sent[HERO_FILM_COUSINS]).mean())

    # the name-free core: canon (all 32, original texts, all films as reference) still present in >= 70% of name-free Batman works
    Po = S_o > bar(S_o, FILMS)
    Pa = S_a > bar(S_a, FILMS)
    canon = np.where(Po[BAT].mean(0) >= 0.8)[0]
    core = canon[Pa[BAT][:, canon].mean(0) >= 0.7]
    # the same procedure run on the 13 cousins (are they a "franchise"?) and on random 32-film pseudo-franchises
    cous = np.concatenate([HERO_FILM_COUSINS, LIT_COUSINS])
    cous_canon = np.where(Po[cous].mean(0) >= 0.8)[0]
    cous_core = cous_canon[Pa[cous][:, cous_canon].mean(0) >= 0.7]
    pseudo = []
    for _ in range(20):
        fr = RNG.choice(FILMS, 32, replace=False)
        ref = np.setdiff1d(FILMS, fr)
        Pof, Paf = S_o > bar(S_o, ref), S_a > bar(S_a, ref)
        cn = np.where(Pof[fr].mean(0) >= 0.8)[0]
        pseudo.append(int((Paf[fr][:, cn].mean(0) >= 0.7).sum()))
    res["core"] = {"n_canon_units": int(len(canon)), "n_core_units": int(len(core)), "core_units": core.tolist(),
                   "canon_units": canon.tolist(), "cousins_core_units": int(len(cous_core)),
                   "pseudo_franchise_core_units_mean": float(np.mean(pseudo)), "pseudo_franchise_core_units_max": int(np.max(pseudo)),
                   "core_present_share_in_hero_cousins": float(Pa[HERO_FILM_COUSINS][:, core].mean()) if len(core) else None,
                   "core_present_share_in_films": float(Pa[FILMS][:, core].mean()) if len(core) else None}
    return res


# ------------------------------------------------------------------ E2
def e2(P, band=(2, 16), time_rule=True, use_hub=True, adaptations=None, n_perm=10000):
    adaptations = adaptations or ADAPTATIONS
    nb = P[BAT].sum(0)
    opt = (nb >= band[0]) & (nb <= band[1])

    def rate(a, c):
        oc = opt & P[c]
        return (oc & P[a]).sum() / max(oc.sum(), 1)

    R = {(a, c): rate(a, c) for a in BAT for c in COMICS}
    Z = {}
    for a in BAT:
        for c in COMICS:
            vals = np.array([R[(t, c)] for t in SCREEN_NONADAPT if t != a])
            Z[(a, c)] = (R[(a, c)] - vals.mean()) / (vals.std() + 1e-9) if use_hub else R[(a, c)]

    def z(a, c):
        return Z[(a, c)]

    def stat_for(a, sources, comic_map=None):
        cands = [c for c in COMICS if (not time_rule or year[c] <= year[a])]
        src = [comic_map[s] if comic_map else s for s in sources]
        cands = sorted(set(cands) | set(src))
        sz = {c: z(a, c) for c in cands}
        best = max(sz[s] for s in src)
        others = [c for c in cands if c not in src]
        share_better = np.mean([sz[c] > best for c in others]) if others else 0.0
        rank = 1 + sum(sz[c] > best for c in others)
        return float(share_better), int(rank), len(cands), bool(best > np.median([sz[c] for c in others])) if others else True

    rows = []
    for t, srcs in adaptations.items():
        a = L[t]
        s = [L[x] for x in srcs]
        share, rank, n, beats = stat_for(a, s)
        rows.append({"target": t, "sources": srcs, "share_of_comics_beating_source": round(share, 3), "rank": rank, "n_comics": n, "beats_median": beats})
    obs = float(np.mean([r["share_of_comics_beating_source"] for r in rows]))
    null = []
    for _ in range(n_perm):
        pi = dict(zip(COMICS, RNG.permutation(COMICS)))
        null.append(np.mean([stat_for(L[t], [L[x] for x in srcs], pi)[0] for t, srcs in adaptations.items()]))
    null = np.array(null)
    # placebo targets: the 11 non-adaptation screen works given every documented source set
    placebo = []
    for t in SCREEN_NONADAPT:
        placebo.append(np.mean([stat_for(t, [L[x] for x in srcs])[0] for srcs in adaptations.values()]))
    return {"rows": rows, "mean_share_beating_source": obs, "null_mean": float(null.mean()),
            "p_one_sided": float((np.sum(null <= obs) + 1) / (n_perm + 1)), "beats_median": int(sum(r["beats_median"] for r in rows)),
            "n": len(rows), "placebo_mean": float(np.mean(placebo)), "placebo_min": float(np.min(placebo)),
            "real_below_all_placebos": bool(obs < np.min(placebo))}


def sequel_control(P, band=(2, 16)):
    nb = P[BAT].sum(0)
    opt = (nb >= band[0]) & (nb <= band[1])
    out = []
    for seq, pre in SEQUELS.items():
        s, p = L[seq], L[pre]
        cands = [c for c in SCREEN if year[c] <= year[s] and c != s]
        rate = {c: (opt & P[c] & P[s]).sum() / max((opt & P[c]).sum(), 1) for c in cands}
        order = sorted(cands, key=lambda c: -rate[c])
        out.append({"sequel": seq, "prequel": pre, "rank": order.index(p) + 1, "n": len(cands), "closest": label[order[0]]})
    return {"rows": out, "top2": sum(r["rank"] <= 2 for r in out), "n": len(out)}


# ------------------------------------------------------------------ E3
def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def resid(y, X):
    X = np.c_[np.ones(len(y)), X]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def e3(P, pv, n_perm=10000, drop_adapted=True, reverse=False, placebo_films=False):
    early = BAT[year[BAT] < 2000]
    later = BAT[year[BAT] >= 2000]
    if drop_adapted:
        later = np.array([w for w in later if label[w] not in ADAPTATIONS])
    if reverse:
        early, later = later, early
    pop = np.log10(np.array([pv[str(i)]["views"] / max(pv[str(i)]["months"], 1) + 1 for i in range(N)]))
    Pe = P[early]
    k = Pe.sum(0)
    keep = (k >= 2) & (k <= 0.8 * len(early))
    Pe = Pe[:, keep]
    k = k[keep]
    outcome = P[FILMS][:, keep].mean(0) if placebo_films else P[later][:, keep].mean(0)
    r_unit = resid(outcome, k[:, None].astype(float))
    off = np.array([r_unit[Pe[j]].mean() for j in range(len(early))])
    # work-level covariates: year, plot length, centrality among early works
    J = (Pe[:, None, :] & Pe[None, :, :]).sum(-1) / np.maximum((Pe[:, None, :] | Pe[None, :, :]).sum(-1), 1)
    np.fill_diagonal(J, np.nan)
    central = np.nanmean(J, axis=1)
    cov = np.c_[year[early], n_sent[early], central].astype(float)
    off_r = resid(off, cov)
    p = pop[early]
    rho = spearman(off_r, p)
    strata = [np.where(medium[early] == "comic")[0], np.where(medium[early] != "comic")[0]]
    null = []
    for _ in range(n_perm):
        q = p.copy()
        for s in strata:
            q[s] = RNG.permutation(q[s])
        null.append(spearman(off_r, q))
    null = np.array(null)
    jack = [spearman(np.delete(off_r, i), np.delete(p, i)) for i in range(len(early))]
    return {"n_early": int(len(early)), "n_later": int(len(later)), "n_units": int(keep.sum()), "rho": rho,
            "p_one_sided": float((np.sum(null >= rho) + 1) / (n_perm + 1)), "null_sd": float(null.std()),
            "jackknife_min": float(np.min(jack)), "jackknife_positive": int(sum(j > 0 for j in jack)),
            "works": [{"work": label[w], "year": int(year[w]), "monthly_views": round(10 ** float(pop[w]) - 1),
                       "offspring": round(float(off_r[i]), 5), "n_units": int(Pe[i].sum())} for i, w in enumerate(early)]}


def main():
    pv = json.load(open("data/pageviews.json"))
    early = BAT[year[BAT] < 2000]
    print("early works pageview months:", sorted({pv[str(w)]["months"] for w in early}), "min views:", min(pv[str(w)]["views"] for w in early))
    icon_keep = np.array([not ICON.search(s) for w in neutral for s in w["sentences"]])
    print("iconography sentences dropped from name-free texts:", int((~icon_keep).sum()), "of", len(icon_keep))
    out = {"iconography_sentences_dropped": int((~icon_keep).sum())}
    for tag in MODELS:
        print("==", tag, flush=True)
        S_o = work_scores_masked(ACTS, tag)
        S_a = work_scores_masked(ANON, tag)
        S_ab = work_scores_masked(ANON, tag, keep=icon_keep)
        res = {"E1": e1(S_o, S_a, S_ab)}
        del S_ab
        P_a = S_a > bar(S_a, FILMS)
        P_o = S_o > bar(S_o, FILMS)
        res["E2"] = {"primary": e2(P_a), "original_upper_bound": e2(P_o, n_perm=2000),
                     "raw_rate": e2(P_a, use_hub=False, n_perm=2000), "no_time_rule": e2(P_a, time_rule=False, n_perm=2000),
                     "strong_subset": e2(P_a, adaptations={k: v for k, v in ADAPTATIONS.items() if k in STRONG}, n_perm=2000),
                     "without_killing_joke": e2(P_a, adaptations={k: v for k, v in ADAPTATIONS.items() if "The Killing Joke" not in v}, n_perm=2000),
                     "band_3_16": e2(P_a, band=(3, 16), n_perm=2000), "band_4_16": e2(P_a, band=(4, 16), n_perm=2000), "band_2_8": e2(P_a, band=(2, 8), n_perm=2000),
                     "sequels_name_free": sequel_control(P_a), "sequels_original": sequel_control(P_o)}
        e2p = res["E2"]["primary"]
        print("  E2 primary share beating source", round(e2p["mean_share_beating_source"], 3), "null", round(e2p["null_mean"], 3), "p", e2p["p_one_sided"],
              "| placebo mean", round(e2p["placebo_mean"], 3), "| sequels top2", res["E2"]["sequels_name_free"]["top2"], "/ 7", flush=True)
        res["E3"] = {"primary": e3(P_a, pv), "original_text": e3(P_o, pv, n_perm=2000), "with_adapted": e3(P_a, pv, drop_adapted=False, n_perm=2000),
                     "reverse_time_placebo": e3(P_a, pv, reverse=True, n_perm=2000), "baseline_film_placebo": e3(P_a, pv, placebo_films=True, n_perm=2000)}
        e3p = res["E3"]["primary"]
        print("  E3 primary rho", round(e3p["rho"], 3), "p", e3p["p_one_sided"], "| reverse", round(res["E3"]["reverse_time_placebo"]["rho"], 3),
              "| films placebo", round(res["E3"]["baseline_film_placebo"]["rho"], 3), flush=True)
        out[tag] = res
        del S_o, S_a, P_a, P_o
        json.dump(out, open("results/experiments.json", "w"))
    print("wrote results/experiments.json")


if __name__ == "__main__":
    main()
