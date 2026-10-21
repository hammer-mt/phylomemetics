"""Phase 2: three pre-registered experiments on neuron-defined memes (design revised after adversarial review).

Units are neuron poles; a unit is "present" in a work when the mean of the work's three strongest sentences
beats all but one of a set of reference ordinary films. Name-free texts (data/corpus_neutral.jsonl) replace
every name with a neutral code (X1, X2...).

E1 Identity. Primary: Qwen3-8B; canon = units present in >= 80% of the other 31 Batman works (original texts);
   the held-out work is scored on its NAME-FREE text by the share of canon units present. Reference films are
   split in half 20 times: one half sets the presence bar, the other half is scored. Primary comparison: Batman
   vs the 8 masked-hero and superhero film cousins. Prediction AUC >= 0.80, bootstrap 95% lower bound >= 0.65,
   above the 95th percentile of a label-shuffle null; holds within 0.05 after dropping sentences that still
   contain iconography words. Falsified if AUC < 0.70 or its CI includes 0.5. Controls: sentence count alone, and
   the pre-registered TF-IDF centroid score run through the same leave-one-out. The name-free core's presence in
   ordinary films is measured on the scored half of each split, with the bar set by the other half (2/35 in
   expectation for any unit, because the films set their own bar).
E2 Heredity. Primary: Qwen3-8B name-free; optional units present in 2-16 of 32 Batman works; for adaptation A
   and comic C (published no later than A), rate = share of C's optional units also present in A, z-scored
   against the 11 non-adaptation screen works ("hub" adjustment). Statistic: share of eligible comics that beat
   the documented source (best of its sources), averaged over the 9 adaptations; joint comic-relabelling
   permutation, one-sided p < 0.05. Positive control: sequels recover their prequel (top 2 for >= 5 of 7) on
   name-free texts; if that fails the result is inconclusive. Placebo targets with no eligible comic under the
   time rule (1943 serial, Batman '66) are left out. Chance is also given exactly: every joint relabelling, the
   time-consistent ones, the within-era ones, and the expectation when each source is drawn from eligible comics.
   Pre-registered lexical comparators on the same name-free texts (X-codes stripped): TF-IDF cosine and word
   presence through the same statistic, and the residual test (neuron rate with lexical similarity regressed out
   over the 32 x 12 work-comic pairs). Falsifier: residual p >= 0.05 means the neurons add nothing beyond words.
E3 Selection. Primary: Qwen3-8B name-free; early = 15 Batman works before 2000, later = 17 from 2000 minus the
   8 adaptation-linked works (9 left). Units present in 2 to 80% of early works. Per early work, offspring =
   mean over its units of later_share residualised on early prevalence, then residualised on the work's year,
   plot length and centrality. Spearman with log mean monthly pageviews; permutation within medium (screen vs
   comic). Prediction rho > 0, p < 0.05 one-sided, same sign in Llama. Placebos: reverse time, held-out films
   (presence in the scored half of each film split, bar from the other half; 2/35 in expectation for every unit, so
   its outcome is noise around a constant). Simulated power at n = 15.

Writes results/experiments.json.
"""
import itertools
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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
CODE = re.compile(r"\bX\d+\b")
n_sent = np.array([len(w["sentences"]) for w in works])
ERA = np.where(year <= 1989, 0, np.where(year <= 1999, 1, 2))   # late 80s / 90s / 2000s+

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


def film_splits(n_splits=20):
    """Random halves of the 69 ordinary films: (34 that set the presence bar, 35 that are scored)."""
    splits = []
    for _ in range(n_splits):
        perm = RNG.permutation(FILMS)
        splits.append((perm[:34], perm[34:]))
    return splits


def held_out_film_presence(S, splits):
    """Per unit: share of the scored half of the films where the unit is present, bar from the other half (mean over splits)."""
    tot = np.zeros(S.shape[1])
    for h1, h2 in splits:
        tot += (S[h2] > bar(S, h1)).mean(0)
    return tot / len(splits)


def lexical_texts(src, drop_icon=False):
    """Plain texts with the X-codes stripped; drop_icon removes sentences with iconography words (all kept if none survive)."""
    out = []
    for w in src:
        ss = [s for s in w["sentences"] if not ICON.search(s)] if drop_icon else w["sentences"]
        out.append(" ".join(CODE.sub(" ", s) for s in (ss or w["sentences"])))
    return out


def tfidf():
    """phylo.py's TF-IDF settings."""
    return TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)


def auc_matrix(pos, neg):
    p, q = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return (p > q) + 0.5 * (p == q)


def boot_ci(M, n=2000, rng=None):
    """Bootstrap 95% CI of mean(M) resampling rows and columns."""
    rng = RNG if rng is None else rng
    r, c = M.shape
    vals = [M[rng.integers(0, r, r)][:, rng.integers(0, c, c)].mean() for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ------------------------------------------------------------------ E1
def e1(S_o, S_a, S_a_ablate, splits, n_null=200):
    def run(franchise, neg_fns, S_canon, S_score, n_use=None):
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
    print("  E1 length-only control: sentence count alone, AUC vs hero film cousins", round(res["length_only_auc_vs_hero"], 3), flush=True)

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
    # core presence in ordinary films, measured on the scored half of each split with the bar from the other half
    # (with all films setting the bar, at most 1 of 69 films can pass for any unit, so that share is 1/69 by construction).
    # The held-out share is fixed by construction too: films are exchangeable across splits, so any unit's expected
    # share is 2/35 (held_out_film_chance_share). It checks the bar, not the core.
    Sc = S_a[:, core]
    held = [float((Sc[h2] > bar(Sc, h1)).mean()) for h1, h2 in splits] if len(core) else None
    res["core"] = {"n_canon_units": int(len(canon)), "n_core_units": int(len(core)), "core_units": core.tolist(),
                   "canon_units": canon.tolist(), "cousins_core_units": int(len(cous_core)),
                   "pseudo_franchise_core_units_mean": float(np.mean(pseudo)), "pseudo_franchise_core_units_max": int(np.max(pseudo)),
                   "core_present_share_in_hero_cousins": float(Pa[HERO_FILM_COUSINS][:, core].mean()) if len(core) else None,
                   "core_present_share_in_held_out_films": float(np.mean(held)) if held else None,
                   "core_present_share_in_held_out_films_range": [float(np.min(held)), float(np.max(held))] if held else None,
                   "held_out_film_chance_share": float(np.mean([2 / (len(h1) + 1) for h1, _ in splits]))}
    print("  E1 core", len(core), "units; present in held-out films", round(res["core"]["core_present_share_in_held_out_films"] or 0, 3),
          "(chance", round(res["core"]["held_out_film_chance_share"], 3), "), hero cousins", round(res["core"]["core_present_share_in_hero_cousins"] or 0, 3), flush=True)
    return res


def e1_tfidf(icon_texts):
    """Pre-registered E1 lexical control: the same leave-one-out with a TF-IDF centroid score instead of the canon share.
    The canon is the centroid of the other 31 Batman works (original texts for A, name-free for B); every work is scored
    by the cosine of its name-free text to it. One vocabulary is fitted on the original and name-free texts together."""
    rng = np.random.default_rng(7)
    orig, anon = lexical_texts(works), lexical_texts(neutral)
    vec = tfidf().fit(orig + anon)
    To, Ta, Tab = (vec.transform(t).toarray() for t in (orig, anon, icon_texts))
    NEG = {"hero_film_cousins": HERO_FILM_COUSINS, "ordinary_films": FILMS, "literary_cousins": LIT_COUSINS}
    res = {}
    for name, Tc, Ts in (("A_name_free", To, Ta), ("B_canon_from_name_free", Ta, Ta), ("A_iconography_dropped", To, Tab)):
        rows = {k: [] for k in NEG}
        for b in BAT:
            c = Tc[BAT[BAT != b]].mean(0)
            sc = Ts @ (c / np.linalg.norm(c))          # rows of Ts have unit length, so this is the cosine
            for k, ng in NEG.items():
                rows[k].append(auc_matrix([sc[b]], sc[ng])[0])
        out = {k: {"auc": float(np.mean(v)), "ci95": boot_ci(np.array(v), rng=rng)} for k, v in rows.items()}
        if name == "A_name_free":
            M = np.array(rows["hero_film_cousins"])
            out["per_cousin_beaten_by"] = {label[c]: round(float(1 - M[:, j].mean()), 3) for j, c in enumerate(HERO_FILM_COUSINS)}
        res[name] = out
        print("  E1 TF-IDF centroid", name, {k: round(v["auc"], 3) for k, v in out.items() if "auc" in v}, flush=True)
    res["n_words"] = int(len(vec.vocabulary_))
    return res


# ------------------------------------------------------------------ E2
_MAPS = {}


def source_maps(k):
    """Every injective map of k documented source comics into the 12 comics, [12!/(12-k)!, k] indices into COMICS.
    The joint relabelling permutation only moves the sources, so these maps are its exact distribution."""
    if k not in _MAPS:
        M = np.arange(len(COMICS), dtype=np.int8)[:, None]
        for _ in range(k - 1):
            r, c = np.nonzero(~(M[:, :, None] == np.arange(len(COMICS))).any(1))
            M = np.c_[M[r], c.astype(np.int8)]
        _MAPS[k] = M
    return _MAPS[k]


def unit_rates(P, band=(2, 16)):
    """R[a, j] = share of comic j's optional units (present in band[0]..band[1] Batman works) also present in work a."""
    nb = P[BAT].sum(0)
    opt = (nb >= band[0]) & (nb <= band[1])
    R = np.zeros((N, len(COMICS)))
    for j, c in enumerate(COMICS):
        oc = opt & P[c]
        for a in BAT:
            R[a, j] = (oc & P[a]).sum() / max(oc.sum(), 1)
    return R


def e2(P, band=(2, 16), time_rule=True, use_hub=True, adaptations=None, n_perm=10000, R=None, rng=None):
    """E2 statistic and nulls. R ([N, 12] work-by-comic rate) replaces the unit inheritance rate computed from P
    (lexical comparators and residual tests); rng replaces the shared generator."""
    adaptations = adaptations or ADAPTATIONS
    rng = RNG if rng is None else rng
    R = unit_rates(P, band) if R is None else R
    Z = {}
    for a in BAT:
        for j, c in enumerate(COMICS):
            vals = np.array([R[t, j] for t in SCREEN_NONADAPT if t != a])
            Z[(a, c)] = (R[a, j] - vals.mean()) / (vals.std() + 1e-9) if use_hub else R[a, j]

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
        pi = dict(zip(COMICS, rng.permutation(COMICS)))
        null.append(np.mean([stat_for(L[t], [L[x] for x in srcs], pi)[0] for t, srcs in adaptations.items()]))
    null = np.array(null)
    # placebo targets: the non-adaptation screen works given every documented source set. A target with no eligible
    # comic under the time rule (1943 serial, Batman '66) has nothing to rank its sources against, so it is left out.
    placebo = {}
    for t in SCREEN_NONADAPT:
        if time_rule and not (year[COMICS] <= year[t]).any():
            continue
        placebo[label[t]] = float(np.mean([stat_for(t, [L[x] for x in srcs])[0] for srcs in adaptations.values()]))
    pv = np.array(list(placebo.values()))

    # exact chance: the statistic under every relabelling of the documented sources, under the relabellings in which
    # every mapped source is eligible for (not later than) each adaptation that uses it, and under those that keep
    # each source in its own era; plus the expectation when each adaptation's sources are drawn from its eligible comics
    used = sorted({L[x] for srcs in adaptations.values() for x in srcs})
    col = {s: i for i, s in enumerate(used)}
    M = source_maps(len(used))
    yc, ec = year[COMICS], ERA[COMICS]
    tot = np.zeros(len(M))
    time_ok = np.ones(len(M), bool)
    uniform = []
    for t, srcs in adaptations.items():
        a, s = L[t], [L[x] for x in srcs]
        T = np.full((len(COMICS),) * len(s), np.nan)
        combos = list(itertools.permutations(range(len(COMICS)), len(s)))
        for cmb in combos:
            T[cmb] = stat_for(a, list(COMICS[list(cmb)]))[0]
        tot += T[tuple(M[:, col[x]] for x in s)]
        uniform.append(np.mean([T[cmb] for cmb in combos if not time_rule or all(yc[i] <= year[a] for i in cmb)]))
        for x in s:
            time_ok &= yc[M[:, col[x]]] <= year[a]
    tot /= len(adaptations)
    era_ok = np.all([ec[M[:, col[x]]] == ERA[x] for x in used], axis=0)
    obs_exact = float(np.mean([stat_for(L[t], [L[x] for x in srcs])[0] for t, srcs in adaptations.items()]))

    def exact(keep):
        v = tot[keep]
        return {"n_maps": int(keep.sum()), "mean": float(v.mean()), "p": float(np.mean(v <= obs_exact + 1e-12))}

    chance = {"observed_unrounded": obs_exact, "joint_relabelling_exact": exact(np.ones(len(M), bool)),
              "joint_relabelling_time_consistent": exact(time_ok), "within_era": exact(era_ok),
              "eligible_uniform_expectation": float(np.mean(uniform))}
    return {"rows": rows, "mean_share_beating_source": obs, "null_mean": float(null.mean()),
            "p_one_sided": float((np.sum(null <= obs) + 1) / (n_perm + 1)), "beats_median": int(sum(r["beats_median"] for r in rows)),
            "n": len(rows), "placebo_mean": float(pv.mean()), "placebo_min": float(pv.min()),
            "real_below_all_placebos": bool(obs < pv.min()), "placebo_p5": float(np.percentile(pv, 5)), "n_valid_placebos": len(pv),
            "placebo_excluded_no_eligible_comic": [label[t] for t in SCREEN_NONADAPT if label[t] not in placebo],
            "placebo_by_target": placebo, "chance": chance}


def sequel_control(P, band=(2, 16), sim=None):
    """Prequel rank among the earlier screen works by the unit rate, or by `sim` ([N, N] similarity) when given."""
    if sim is None:
        nb = P[BAT].sum(0)
        opt = (nb >= band[0]) & (nb <= band[1])
    out = []
    for seq, pre in SEQUELS.items():
        s, p = L[seq], L[pre]
        cands = [c for c in SCREEN if year[c] <= year[s] and c != s]
        if sim is None:
            rate = {c: (opt & P[c] & P[s]).sum() / max((opt & P[c]).sum(), 1) for c in cands}
        else:
            rate = {c: sim[s, c] for c in cands}
        order = sorted(cands, key=lambda c: -rate[c])
        out.append({"sequel": seq, "prequel": pre, "rank": order.index(p) + 1, "n": len(cands), "closest": label[order[0]]})
    return {"rows": out, "top2": sum(r["rank"] <= 2 for r in out), "n": len(out)}


def lexical_reps():
    """Name-free texts with X-codes stripped: TF-IDF cosine between works [N, N] and binary word presence [N, V]."""
    A = tfidf().fit_transform(lexical_texts(neutral)).toarray()
    return A @ A.T, A > 0


def residual_test(R, lex):
    """Pre-registered E2 residual test: regress the rate R on the lexical rate(s) over the 32 x 12 (Batman work, comic)
    pairs, a work's pair with itself left out, and rerun the E2 statistic on the residuals."""
    mask = np.zeros(R.shape, bool)
    mask[BAT] = True
    mask[COMICS, np.arange(len(COMICS))] = False
    X = np.column_stack([np.ones(mask.sum())] + [x[mask] for x in lex])
    beta = np.linalg.lstsq(X, R[mask], rcond=None)[0]
    Rr = R - (np.column_stack([np.ones(R.size)] + [x.ravel() for x in lex]) @ beta).reshape(R.shape)
    out = e2(None, R=Rr, rng=np.random.default_rng(7))
    out["r2_of_fit"] = float(1 - np.var(Rr[mask]) / np.var(R[mask]))
    return out


# ------------------------------------------------------------------ E3
def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def resid(y, X):
    X = np.c_[np.ones(len(y)), X]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def e3(P, pv, n_perm=10000, drop_adapted=True, reverse=False, outcome=None):
    """`outcome` (per unit, all units) replaces the later works' presence share (the held-out film placebo)."""
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
    outcome = P[later][:, keep].mean(0) if outcome is None else outcome[keep]
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


def e3_power(rhos=(0.3, 0.5), n_sim=2000, n_perm=2000, alpha=0.05):
    """Simulated power of the E3 test: the 15 early works, a population Spearman rho between offspring and popularity
    (bivariate normal), one-sided permutation within medium as in e3, no covariates."""
    rng = np.random.default_rng(7)
    early = BAT[year[BAT] < 2000]
    n = len(early)
    strata = [np.where(medium[early] == "comic")[0], np.where(medium[early] != "comic")[0]]
    idx = np.tile(np.arange(n), (n_perm, 1))
    for s in strata:
        idx[:, s] = s[np.argsort(rng.random((n_perm, len(s))), axis=1)]
    out = {"n": n, "alpha": alpha, "n_sim": n_sim, "n_perm": n_perm}
    for rho in rhos:
        r = 2 * np.sin(np.pi * rho / 6)            # Pearson r of the normal pair whose Spearman correlation is rho
        x = rng.standard_normal((n_sim, n))
        y = r * x + np.sqrt(1 - r * r) * rng.standard_normal((n_sim, n))
        rx, ry = np.argsort(np.argsort(x, 1), 1), np.argsort(np.argsort(y, 1), 1)
        obs = (rx * ry).sum(1)                      # rank cross-product: monotone in Spearman rho, exact in integers
        p = np.empty(n_sim)
        for i in range(0, n_sim, 100):
            nul = (rx[i:i + 100, None, :] * ry[i:i + 100][:, idx]).sum(-1)
            p[i:i + 100] = ((nul >= obs[i:i + 100, None]).sum(1) + 1) / (n_perm + 1)
        out[f"power_rho_{rho}"] = float(np.mean(p < alpha))
    return out


def main():
    pv = json.load(open("data/pageviews.json"))
    early = BAT[year[BAT] < 2000]
    print("early works pageview months:", sorted({pv[str(w)]["months"] for w in early}), "min views:", min(pv[str(w)]["views"] for w in early))
    icon_keep = np.array([not ICON.search(s) for w in neutral for s in w["sentences"]])
    print("iconography sentences dropped from name-free texts:", int((~icon_keep).sum()), "of", len(icon_keep))
    out = {"iconography_sentences_dropped": int((~icon_keep).sum())}

    # model-free controls: pre-registered lexical comparators on the same name-free texts, and E3 power
    print("== lexical comparators (name-free texts, X-codes stripped)", flush=True)
    sim, Pw = lexical_reps()
    R_tf, R_wd = sim[:, COMICS], unit_rates(Pw)
    lex = {"description": "Name-free texts with X-codes stripped; TF-IDF with phylo.py settings (English stop words, min_df 2, "
                          "sublinear tf). Word presence: a word is present in a work when it occurs there; optional words are "
                          "present in 2-16 of the 32 Batman works, as for units. Same E2 statistic, hub adjustment, time rule "
                          "and 10,000-permutation joint relabelling as the neurons.",
           "n_words": int(Pw.shape[1]),
           "E1_tfidf_centroid": e1_tfidf(lexical_texts(neutral, drop_icon=True)),
           "E2_tfidf_cosine": {"primary": e2(None, R=R_tf, rng=np.random.default_rng(7)), "sequels_name_free": sequel_control(None, sim=sim)},
           "E2_word_presence": {"primary": e2(Pw, rng=np.random.default_rng(7)), "sequels_name_free": sequel_control(Pw),
                                "n_optional_words": int(((Pw[BAT].sum(0) >= 2) & (Pw[BAT].sum(0) <= 16)).sum())}}
    for k in ("E2_tfidf_cosine", "E2_word_presence"):
        r = lex[k]["primary"]
        print("  ", k, "share beating source", round(r["mean_share_beating_source"], 3), "null", round(r["null_mean"], 3), "p", r["p_one_sided"],
              "| exact p", round(r["chance"]["joint_relabelling_exact"]["p"], 5), "| sequels top2", lex[k]["sequels_name_free"]["top2"], "/ 7", flush=True)
    out["lexical_comparators"] = lex
    out["E3_power_simulated"] = e3_power()
    print("  E3 simulated power", out["E3_power_simulated"], flush=True)

    for tag in MODELS:
        print("==", tag, flush=True)
        S_o = work_scores_masked(ACTS, tag)
        S_a = work_scores_masked(ANON, tag)
        S_ab = work_scores_masked(ANON, tag, keep=icon_keep)
        splits = film_splits()
        res = {"E1": e1(S_o, S_a, S_ab, splits)}
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
        ch = e2p["chance"]
        print("  E2 primary share beating source", round(e2p["mean_share_beating_source"], 3), "null", round(e2p["null_mean"], 3), "p", e2p["p_one_sided"],
              "| placebo mean", round(e2p["placebo_mean"], 3), "of", e2p["n_valid_placebos"], "valid, real below all:", e2p["real_below_all_placebos"],
              "| sequels top2", res["E2"]["sequels_name_free"]["top2"], "/ 7", flush=True)
        print("  E2 chance: exact joint", round(ch["joint_relabelling_exact"]["mean"], 3), "p", round(ch["joint_relabelling_exact"]["p"], 5),
              "| time-consistent", round(ch["joint_relabelling_time_consistent"]["mean"], 3), "p", round(ch["joint_relabelling_time_consistent"]["p"], 5),
              "| within era", round(ch["within_era"]["mean"], 3), "p", round(ch["within_era"]["p"], 5),
              "| eligible-uniform", round(ch["eligible_uniform_expectation"], 3), flush=True)
        R_n = unit_rates(P_a)
        res["E2"]["residual_test_beyond_words"] = {
            "neurons_resid_on_tfidf_cosine": residual_test(R_n, [R_tf]), "neurons_resid_on_word_presence": residual_test(R_n, [R_wd]),
            "neurons_resid_on_both": residual_test(R_n, [R_tf, R_wd]),
            "tfidf_cosine_resid_on_neurons": residual_test(R_tf, [R_n]), "word_presence_resid_on_neurons": residual_test(R_wd, [R_n])}
        for k, r in res["E2"]["residual_test_beyond_words"].items():
            print("  E2 residual", k, "share", round(r["mean_share_beating_source"], 3), "p", round(r["p_one_sided"], 4),
                  "exact p", round(r["chance"]["joint_relabelling_exact"]["p"], 4), "R2", round(r["r2_of_fit"], 3), flush=True)
        film_out = held_out_film_presence(S_a, splits)
        res["E3"] = {"primary": e3(P_a, pv), "original_text": e3(P_o, pv, n_perm=2000), "with_adapted": e3(P_a, pv, drop_adapted=False, n_perm=2000),
                     "reverse_time_placebo": e3(P_a, pv, reverse=True, n_perm=2000), "held_out_film_placebo": e3(P_a, pv, outcome=film_out, n_perm=2000)}
        e3p = res["E3"]["primary"]
        print("  E3 primary rho", round(e3p["rho"], 3), "p", e3p["p_one_sided"], "| reverse", round(res["E3"]["reverse_time_placebo"]["rho"], 3),
              "| held-out films placebo", round(res["E3"]["held_out_film_placebo"]["rho"], 3), "p", res["E3"]["held_out_film_placebo"]["p_one_sided"], flush=True)
        out[tag] = res
        del S_o, S_a, P_a, P_o
        json.dump(out, open("results/experiments.json", "w"))
    print("wrote results/experiments.json")


if __name__ == "__main__":
    main()
