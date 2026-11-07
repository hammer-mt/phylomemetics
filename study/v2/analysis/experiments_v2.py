"""The three pre-registered experiments under the v2 presence rule (v2/PLAN.md, Sections 5 and 6).

A unit is present in a work when its strongest sentence reaches 30% of the unit's scale. E1 uses the distinctive
units (present in at most 3% of 2,999 background film plots); E2 and E3 use every unit that can fire. E2, E3, the
sequel control, the word comparators and the residual test are v1's own functions (imported from experiments.py),
fed the new presence matrix. E1 is v1's design without the film splits, which existed only because the films set
v1's bar; its criteria (met, falsified, not met) are those of PLAN.md Section 6.

    cd study && ../.venv/bin/python v2/analysis/experiments_v2.py [model-tag ...]      (words: the model-free part only)

Writes results/v2/experiments_v2.json (all models found in v2/acts/, keeping models already there) and
results/v2/canon_<tag>.npz (the canon units of each fold, for the ablation and the unit selection).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import common as C

import experiments as X          # v1: corpus constants, e2, e3, sequel_control, residual_test, word comparators

BAT, FILMS, HERO, LIT, N = X.BAT, X.FILMS, X.HERO_FILM_COUSINS, X.LIT_COUSINS, X.N
POOL = np.concatenate([BAT, HERO])
NEG = {"hero_film_cousins": HERO, "ordinary_films": FILMS, "literary_cousins": LIT}
label = X.label
CUT_SENTENCES, CUT_WORDS = 20, 400
MIN_CANON = 20                        # a canon share over fewer units or words than this is not scored
BANDS = [(2, 3), (4, 6), (7, 10), (11, 16)]      # frequency bands of the word-presence regressors (number of Batman works)


def cut_length(sentences):
    """As in v2/gpu/study_pass.py: the opening kept by the length control (at most 20 sentences and 400 words)."""
    words = k = 0
    for s in sentences[:CUT_SENTENCES]:
        n = len(s.split())
        if k and words + n > CUT_WORDS:
            break
        words, k = words + n, k + 1
    return k


CUTS = np.array([cut_length(w["sentences"]) for w in X.works])


# ------------------------------------------------------------------ E1: three ways to score a held-out work
class CanonShare:
    """v1's score: the share of the canon (units present in at least 80% of the other franchise works, in Pc) that is
    present in the work (in Ps)."""

    def __init__(self, Pc, Ps):
        self.Pc, self.Ps = Pc, Ps

    def matrix(self, franchise, negatives):
        cnt = self.Pc[franchise].sum(0)
        need = 0.8 * (len(franchise) - 1) - 1e-9
        rows, info = [], {"canon_size": [], "held_out_score": []}
        neg = self.Ps[negatives]
        for b in franchise:
            canon = np.flatnonzero(cnt - self.Pc[b] >= need)
            info["canon_size"].append(int(len(canon)))
            if len(canon) == 0:
                rows.append(np.full(len(negatives), 0.5))
                info["held_out_score"].append(None)
                continue
            own = float(self.Ps[b, canon].mean())
            info["held_out_score"].append(own)
            rows.append(X.auc_matrix([own], neg[:, canon].mean(1))[0])
        return np.array(rows), info

    def reduced(self, pool):
        """For the label-shuffle null: a canon unit needs 25 of the 31 other franchise works, all drawn from the pool."""
        keep = np.flatnonzero(self.Pc[pool].sum(0) >= 25)
        return CanonShare(self.Pc[:, keep], self.Ps[:, keep])


class Centroid:
    """The words' scoring rule: the cosine between the work's vector (rows of Vs) and the centroid of the other
    franchise works (rows of Vc). Built from Gram matrices, so the vectors themselves need not be kept."""

    def __init__(self, Gcc, Gsc, norm_s):
        self.Gcc, self.Gsc, self.norm_s = Gcc, Gsc, np.maximum(norm_s, 1e-12)

    @classmethod
    def from_vectors(cls, Vc, Vs):
        return cls(Vc @ Vc.T, Vs @ Vc.T, np.sqrt((Vs * Vs).sum(1)))

    def matrix(self, franchise, negatives):
        rows, info = [], {"held_out_score": []}
        for b in franchise:
            o = franchise[franchise != b]
            s = self.Gsc[:, o].sum(1) / (max(np.sqrt(self.Gcc[np.ix_(o, o)].sum()), 1e-12) * self.norm_s)
            info["held_out_score"].append(float(s[b]))
            rows.append(X.auc_matrix([s[b]], s[negatives])[0])
        return np.array(rows), info

    def reduced(self, pool):
        return self


class Fixed:
    """A score that needs no canon (text length)."""

    def __init__(self, s):
        self.s = np.asarray(s, float)

    def matrix(self, franchise, negatives):
        return X.auc_matrix(self.s[franchise], self.s[negatives]), {"held_out_score": [float(x) for x in self.s[franchise]]}

    def reduced(self, pool):
        return self


def unit_centroid(Sc, Pc, Ss, Ps, weight, chunk=1 << 16):
    """Centroid scorer on unit shares: a unit's entry is its share of scale where it fires strongly and 0 elsewhere,
    times its weight (the log of the inverse of its base rate)."""
    Gcc, Gsc, ns = np.zeros((N, N)), np.zeros((N, N)), np.zeros(N)
    for c in range(0, Sc.shape[1], chunk):
        w = (weight[c:c + chunk] / C.STEP).astype(np.float32)
        Vc = Sc[:, c:c + chunk] * Pc[:, c:c + chunk] * w
        Vs = Vc if Ss is Sc and Ps is Pc else Ss[:, c:c + chunk] * Ps[:, c:c + chunk] * w
        Gcc += Vc @ Vc.T
        Gsc += Vs @ Vc.T
        ns += (Vs * Vs).sum(1)
    nc = np.sqrt(np.diag(Gcc))                      # each work's vector is put at unit length before the centroid is taken,
    nc = np.where(nc > 0, nc, 1.0)                  # as TF-IDF rows are: otherwise works where more units fire would weigh more
    return Centroid(Gcc / np.outer(nc, nc), Gsc / nc[None, :], np.sqrt(ns))


def shuffle_null(scorer, n, rng):
    """32 of the 40 works (Batman and the hero films) drawn at random play the franchise, the other 8 the relatives."""
    red = scorer.reduced(POOL)
    vals = []
    for _ in range(n):
        p = rng.permutation(POOL)
        vals.append(float(red.matrix(p[:32], p[32:])[0].mean()))
    return np.array(vals)


def verdict(out):
    """PLAN.md Section 6: met (all clauses), falsified (v1's rule: AUC below 0.70 or an interval including 0.5), not met."""
    a = out["A_name_free"]
    auc, (lo, hi) = a["hero_film_cousins"]["auc"], a["hero_film_cousins"]["ci95"]
    c = {"auc_at_least_0.80": auc >= 0.80, "ci_lower_at_least_0.65": lo >= 0.65}
    if "null_label_shuffle" in a:
        c["above_null_p95"] = auc > a["null_label_shuffle"]["p95"]
    if "A_iconography_dropped" in out:
        c["within_0.05_without_iconography"] = out["A_iconography_dropped"]["hero_film_cousins"]["auc"] >= auc - 0.05
    if "openings" in out:
        o = out["openings"]
        c["openings_auc_at_least_0.70"] = o["hero_film_cousins"]["auc"] >= 0.70
        if "null_label_shuffle" in o:
            c["openings_above_own_null_p95"] = o["hero_film_cousins"]["auc"] > o["null_label_shuffle"]["p95"]
    c = {k: bool(v) for k, v in c.items()}
    complete = len(c) == 6
    falsified = bool(auc < 0.70 or lo <= 0.5 <= hi)
    v = "met" if complete and all(c.values()) else "falsified" if falsified else "not met"
    if a.get("canon_units_median", MIN_CANON) < MIN_CANON:
        v = "not evaluable"                          # e.g. a word canon of four names, all removed from the name-free texts
    return {"verdict": v, "clauses": c, "all_six_clauses_evaluated": complete, "falsified_by_v1_rule": falsified}


def e1_block(scorers, rng, n_null=1000, name=""):
    """scorers: A_name_free, B_canon_from_name_free, ceiling_original, A_iconography_dropped, openings (any may be None)."""
    out = {}
    for key, sc in scorers.items():
        if sc is None:
            continue
        r = {}
        for k, ng in NEG.items():
            M, info = sc.matrix(BAT, ng)
            r[k] = {"auc": float(M.mean()), "ci95": X.boot_ci(M, rng=rng)}
            if k == "hero_film_cousins":
                r["per_batman_vs_hero"] = {label[b]: round(float(M[i].mean()), 3) for i, b in enumerate(BAT)}
                r["per_cousin_beaten_by"] = {label[c]: round(float(1 - M[:, j].mean()), 3) for j, c in enumerate(HERO)}
                r["held_out_score"] = {label[b]: (None if s is None else round(s, 5)) for b, s in zip(BAT, info["held_out_score"])}
                if "canon_size" in info:
                    r["canon_units_median"], r["canon_units_min"] = float(np.median(info["canon_size"])), int(min(info["canon_size"]))
        if key in ("A_name_free", "openings") and n_null:
            null = shuffle_null(sc, n_null, rng)
            obs = r["hero_film_cousins"]["auc"]
            r["null_label_shuffle"] = {"n": n_null, "mean": float(null.mean()), "p95": float(np.percentile(null, 95)),
                                       "p_value": float((np.sum(null >= obs) + 1) / (n_null + 1))}
        out[key] = r
    out.update(verdict(out))
    print(f"  E1 {name}:", {k: round(v["hero_film_cousins"]["auc"], 3) for k, v in out.items() if isinstance(v, dict) and "hero_film_cousins" in v},
          "->", out["verdict"], flush=True)
    return out


# ------------------------------------------------------------------ words: the comparators that need no model
_WORDS = {}


def word_measures():
    """Word-based comparators on the same texts (X-codes stripped, as in v1), including the four added in review."""
    if _WORDS:
        return _WORDS
    bg = [d["text"] for d in C.background_docs()]
    anon = X.lexical_texts(X.neutral)
    sim, Pw = X.lexical_reps()                                     # v1: TF-IDF cosine [N, N] and word presence [N, V]
    vocab = X.tfidf().fit(anon)
    names = vocab.get_feature_names_out()
    df_bg = np.asarray(CountVectorizer(stop_words="english", vocabulary=vocab.vocabulary_, binary=True).fit_transform(bg).sum(0)).ravel()
    rare = df_bg <= 0.03 * len(bg) + 1e-9
    A_bg = X.tfidf().fit(bg).transform(anon).toarray()             # word weights from the background plots
    A_bi = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True, ngram_range=(1, 2)).fit_transform(anon).toarray()
    code_sets = [{int(m[1:]) for s in w["sentences"] for m in X.CODE.findall(s)} for w in X.neutral]
    n_codes = np.array([len(c) for c in code_sets], float)
    P_codes = np.zeros((N, max(max(c) for c in code_sets if c)), bool)          # which codes (X1, X2, ...) a work contains
    for w, c in enumerate(code_sets):
        P_codes[w, [k - 1 for k in c]] = True
    sim_codes = -np.abs(n_codes[:, None] - n_codes[None, :])
    _WORDS.update({"P_codes": P_codes, "df_bg": df_bg, "sim": sim, "Pw": Pw, "sim_bg": A_bg @ A_bg.T, "Pw_rare": Pw[:, rare], "sim_bigram": A_bi @ A_bi.T, "sim_codes": sim_codes,
                   "n_codes": n_codes, "n_words": int(Pw.shape[1]), "n_words_rare": int(rare.sum()), "n_background_plots": len(bg),
                   "commonest_words_dropped_by_the_cap": [str(x) for x in names[np.argsort(-df_bg)][:20]]})
    return _WORDS


_RATES = {}


def word_rates():
    """[N, 12] work-by-comic rates for each word measure. The first two are v1's registered pair; with the four
    frequency bands they are the regressors of the decision test (PLAN.md Section 18, change 6)."""
    if not _RATES:
        W = word_measures()
        _RATES.update({"tfidf_cosine": W["sim"][:, X.COMICS], "word_presence": X.unit_rates(W["Pw"])})
        _RATES.update({f"word_presence_in_{lo}_to_{hi}_works": X.unit_rates(W["Pw"], (lo, hi)) for lo, hi in BANDS})
        _RATES.update({"tfidf_background_weights": W["sim_bg"][:, X.COMICS], "word_presence_background_rare": X.unit_rates(W["Pw_rare"]),
                       "tfidf_words_and_pairs": W["sim_bigram"][:, X.COMICS], "code_count_similarity": W["sim_codes"][:, X.COMICS],
                       "code_presence": X.unit_rates(W["P_codes"])})
    return _RATES


def regressors(which):
    """registered: v1's two word measures. decision: those and word presence in four frequency bands. all: every word and code measure."""
    WR = word_rates()
    keys = list(WR)
    return [WR[k] for k in {"registered": keys[:2], "decision": keys[:2 + len(BANDS)], "all": keys}[which]]


def weighted_word_rate(e):
    """A word-overlap rate that leans towards commoner words: v1's word presence with each optional word weighted by
    (1 + number of Batman works it is in) ** e. Words only; e = 0 is v1's word presence."""
    P = word_measures()["Pw"]
    nb = P[BAT].sum(0)
    w = np.where((nb >= 2) & (nb <= 16), (1.0 + nb) ** e, 0.0)
    R = np.zeros((N, len(X.COMICS)))
    for j, c in enumerate(X.COMICS):
        wc = w * P[c]
        R[BAT, j] = (P[BAT] * wc).sum(1) / max(wc.sum(), 1e-12)
    return R


def opening(src, drop_codes=True):
    return [" ".join((X.CODE.sub(" ", s) if drop_codes else s) for s in w["sentences"][:k]) for w, k in zip(src, CUTS)]


def words_block(n_null=1000):
    """E1 and E2 for the word comparators, under the same criteria as the units."""
    rng = np.random.default_rng(7)
    W = word_measures()
    orig, anon, icon = X.lexical_texts(X.works), X.lexical_texts(X.neutral), X.lexical_texts(X.neutral, drop_icon=True)
    o_cut, a_cut = opening(X.works), opening(X.neutral)
    out = {"n_words": W["n_words"], "n_words_in_at_most_3pct_of_background_plots": W["n_words_rare"],
           "commonest_words_dropped_by_the_cap": W["commonest_words_dropped_by_the_cap"], "E1": {}, "E2": {}}

    def centroid_scorers(vec):
        T = {k: vec.transform(t).toarray() for k, t in (("o", orig), ("a", anon), ("i", icon), ("oc", o_cut), ("ac", a_cut))}
        return {"A_name_free": Centroid.from_vectors(T["o"], T["a"]), "B_canon_from_name_free": Centroid.from_vectors(T["a"], T["a"]),
                "ceiling_original": Centroid.from_vectors(T["o"], T["o"]), "A_iconography_dropped": Centroid.from_vectors(T["o"], T["i"]),
                "openings": Centroid.from_vectors(T["oc"], T["ac"])}

    def presence_scorers(vec, keep=None):
        P = {k: np.asarray(vec.transform(t).toarray() > 0) for k, t in (("o", orig), ("a", anon), ("i", icon), ("oc", o_cut), ("ac", a_cut))}
        if keep is not None:
            P = {k: v[:, keep] for k, v in P.items()}
        return {"A_name_free": CanonShare(P["o"], P["a"]), "B_canon_from_name_free": CanonShare(P["a"], P["a"]),
                "ceiling_original": CanonShare(P["o"], P["o"]), "A_iconography_dropped": CanonShare(P["o"], P["i"]),
                "openings": CanonShare(P["oc"], P["ac"])}

    vec = X.tfidf().fit(orig + anon)                                # v1's E1 comparator: one vocabulary over both versions
    bg = [d["text"] for d in C.background_docs()]
    vec_bg = X.tfidf().fit(bg)
    df_bg = np.asarray(CountVectorizer(stop_words="english", vocabulary=vec.vocabulary_, binary=True).fit_transform(bg).sum(0)).ravel()
    n_icon = np.array([sum(not X.ICON.search(s) for s in w["sentences"]) or len(w["sentences"]) for w in X.neutral])
    n_words_all = np.array([sum(len(s.split()) for s in w["sentences"]) for w in X.works])
    out["E1"]["tfidf_centroid"] = e1_block(centroid_scorers(vec), rng, n_null, "TF-IDF centroid")
    out["E1"]["tfidf_centroid_background_weights"] = e1_block(centroid_scorers(vec_bg), rng, n_null, "TF-IDF centroid, background weights")
    out["E1"]["word_presence_canon_share"] = e1_block(presence_scorers(vec), rng, n_null, "word canon share")
    out["E1"]["word_presence_canon_share_background_rare"] = e1_block(presence_scorers(vec, df_bg <= 0.03 * len(bg) + 1e-9), rng, n_null,
                                                                      "word canon share, words in at most 3% of background plots")
    length = {"A_name_free": Fixed(X.n_sent), "A_iconography_dropped": Fixed(n_icon), "openings": Fixed(CUTS)}
    out["E1"]["length_alone_sentences"] = e1_block(length, rng, n_null, "length alone (sentences)")
    n_words_icon = np.array([sum(len(s.split()) for s in w["sentences"] if not X.ICON.search(s)) for w in X.neutral])
    out["E1"]["length_alone_words"] = e1_block({"A_name_free": Fixed(n_words_all), "A_iconography_dropped": Fixed(n_words_icon),
                                                "openings": Fixed([len(t.split()) for t in opening(X.works, False)])}, rng, n_null, "length alone (words)")
    R = word_rates()
    # what words alone give in the residual test: a pass here cannot mean "beyond words"
    cal = {"word_presence_of_words_in_11_to_16_works": R["word_presence_in_11_to_16_works"]}
    cal.update({f"word_presence_weighted_by_frequency_to_the_power_{e}": weighted_word_rate(e) for e in (0.5, 1, 2, 3)})
    out["residual_test_on_words_alone"] = {
        k: {form: dict(pick(r), r2_of_fit=r["r2_of_fit"]) for form, r in (("registered_two_word_measures", residual(Rw, regressors("registered"))),
                                                                          ("decision_with_frequency_bands", residual(Rw, regressors("decision"))))}
        for k, Rw in cal.items()}
    for k, v in out["residual_test_on_words_alone"].items():
        print(f"  words alone, {k}: registered p {v['registered_two_word_measures']['exact_p']:.4f}, with frequency bands p {v['decision_with_frequency_bands']['exact_p']:.4f}", flush=True)
    for k, P in (("tfidf_cosine", None), ("word_presence", W["Pw"]), ("tfidf_background_weights", None),
                 ("word_presence_background_rare", W["Pw_rare"]), ("tfidf_words_and_pairs", None), ("code_count_similarity", None),
                 ("code_presence", W["P_codes"])):
        r = X.e2(P, rng=np.random.default_rng(7)) if P is not None else X.e2(None, R=R[k], rng=np.random.default_rng(7))
        sims = {"tfidf_cosine": "sim", "tfidf_background_weights": "sim_bg", "tfidf_words_and_pairs": "sim_bigram", "code_count_similarity": "sim_codes"}
        out["E2"][k] = {"primary": r, "share_ties_half": share_with_ties(R[k]),
                        "sequels_name_free": X.sequel_control(P) if P is not None else X.sequel_control(None, sim=W[sims[k]])}
        print(f"  words E2 {k}: share {r['mean_share_beating_source']:.3f} exact p {r['chance']['joint_relabelling_exact']['p']:.5f} "
              f"sequels {out['E2'][k]['sequels_name_free']['top2']}/7", flush=True)
    return out


# ------------------------------------------------------------------ E2
def share_with_ties(R, use_hub=True, time_rule=True):
    """v1's E2 statistic with a comic that ties the documented source counted as half."""
    shares = []
    for t, srcs in X.ADAPTATIONS.items():
        a, src = X.L[t], [X.L[x] for x in srcs]
        z = {}
        for j, c in enumerate(X.COMICS):
            vals = np.array([R[s, j] for s in X.SCREEN_NONADAPT if s != a])
            z[c] = (R[a, j] - vals.mean()) / (vals.std() + 1e-9) if use_hub else R[a, j]
        cands = sorted({c for c in X.COMICS if not time_rule or X.year[c] <= X.year[a]} | set(src))
        best = max(z[s] for s in src)
        others = [c for c in cands if c not in src]
        shares.append(float(np.mean([(z[c] > best) + 0.5 * (z[c] == best) for c in others])) if others else 0.0)
    return float(np.mean(shares))


def evaluable(P, band=(2, 16)):
    """PLAN.md Section 6: every comic has at least 50 optional units, and no comic's rate is constant across the 11
    non-adaptation screen works (a constant rate makes the hub adjustment meaningless)."""
    nb = P[BAT].sum(0)
    opt = (nb >= band[0]) & (nb <= band[1])
    n_opt = [int((opt & P[c]).sum()) for c in X.COMICS]
    R = X.unit_rates(P, band)
    const = [bool(np.ptp(R[X.SCREEN_NONADAPT, j]) == 0) for j in range(len(X.COMICS))]
    return {"n_optional_units": int(opt.sum()), "optional_units_per_comic_min": min(n_opt), "optional_units_per_comic_median": float(np.median(n_opt)),
            "comics_with_constant_rate": int(sum(const)), "evaluable": bool(min(n_opt) >= 50 and not any(const))}, R


def pick(r):
    """The numbers quoted from one run of v1's e2()."""
    ch = r["chance"]
    return {"share": ch["observed_unrounded"], "exact_p": ch["joint_relabelling_exact"]["p"], "within_era_p": ch["within_era"]["p"],
            "time_consistent_p": ch["joint_relabelling_time_consistent"]["p"], "chance_mean": ch["joint_relabelling_exact"]["mean"],
            "placebo_mean": r["placebo_mean"], "real_below_all_placebos": r["real_below_all_placebos"]}


def residual(R, lex, adaptations=None, n_perm=200):
    """v1's residual_test (the rate with the word measures regressed out over the 32 x 12 pairs, a work's pair with itself
    left out), with the adaptations and the number of Monte Carlo draws as arguments. The exact p values do not depend on
    the draws."""
    mask = np.zeros(R.shape, bool)
    mask[BAT] = True
    mask[X.COMICS, np.arange(len(X.COMICS))] = False
    A = np.column_stack([np.ones(mask.sum())] + [x[mask] for x in lex])
    beta = np.linalg.lstsq(A, R[mask], rcond=None)[0]
    Rr = R - (np.column_stack([np.ones(R.size)] + [x.ravel() for x in lex]) @ beta).reshape(R.shape)
    out = X.e2(None, R=Rr, adaptations=adaptations, n_perm=n_perm, rng=np.random.default_rng(7))
    out["r2_of_fit"] = float(1 - np.var(Rr[mask]) / np.var(R[mask]))
    return out


def e2_block(Pa, Po, full=True):
    ev, R_n = evaluable(Pa)
    e2p = X.e2(Pa, rng=np.random.default_rng(7))
    res = {"primary": e2p, "summary": pick(e2p), "evaluable": ev, "share_ties_half": share_with_ties(R_n), "sequels_name_free": X.sequel_control(Pa)}
    res["sequel_control_passes"] = bool(res["sequels_name_free"]["top2"] >= 5)
    WR = word_rates()
    two = regressors("registered")
    res["residual_test_beyond_words"] = {
        "decision_words_by_frequency": X.residual_test(R_n, regressors("decision")),
        "registered_both_word_measures": X.residual_test(R_n, two),
        "every_word_and_code_measure": X.residual_test(R_n, regressors("all")),
        "neurons_resid_on_tfidf_cosine": X.residual_test(R_n, two[:1]), "neurons_resid_on_word_presence": X.residual_test(R_n, two[1:]),
        "tfidf_cosine_resid_on_neurons": X.residual_test(WR["tfidf_cosine"], [R_n]),
        "word_presence_resid_on_neurons": X.residual_test(WR["word_presence"], [R_n])}
    res["residual_regressors"] = {"decision_words_by_frequency": list(WR)[:2 + len(BANDS)], "registered_both_word_measures": list(WR)[:2],
                                  "every_word_and_code_measure": list(WR)}
    res["residual_summary"] = {k: dict(pick(r), r2_of_fit=r["r2_of_fit"]) for k, r in res["residual_test_beyond_words"].items()}
    s, d, g = res["summary"], res["residual_summary"]["decision_words_by_frequency"], res["residual_summary"]["registered_both_word_measures"]
    print(f"  E2 share {s['share']:.3f} exact p {s['exact_p']:.5f} within-era p {s['within_era_p']:.4f} | evaluable {ev['evaluable']} "
          f"({ev['n_optional_units']:,} optional units, min per comic {ev['optional_units_per_comic_min']}) | sequels {res['sequels_name_free']['top2']}/7 "
          f"| residual (decision) share {d['share']:.3f} exact p {d['exact_p']:.4f} within-era p {d['within_era_p']:.4f} "
          f"| residual (v1's two measures) exact p {g['exact_p']:.4f}", flush=True)
    if full:
        r7 = lambda: np.random.default_rng(7)      # noqa: E731
        res.update({
            "original_upper_bound": X.e2(Po, n_perm=2000, rng=r7()), "raw_rate": X.e2(Pa, use_hub=False, n_perm=2000, rng=r7()),
            "no_time_rule": X.e2(Pa, time_rule=False, n_perm=2000, rng=r7()),
            "strong_subset": X.e2(Pa, adaptations={k: v for k, v in X.ADAPTATIONS.items() if k in X.STRONG}, n_perm=2000, rng=r7()),
            "without_killing_joke": X.e2(Pa, adaptations={k: v for k, v in X.ADAPTATIONS.items() if "The Killing Joke" not in v}, n_perm=2000, rng=r7()),
            "band_3_16": X.e2(Pa, band=(3, 16), n_perm=2000, rng=r7()), "band_4_16": X.e2(Pa, band=(4, 16), n_perm=2000, rng=r7()),
            "band_2_8": X.e2(Pa, band=(2, 8), n_perm=2000, rng=r7()), "sequels_original": X.sequel_control(Po)})
        res["residual_jackknife_leave_one_adaptation_out"] = {
            t: pick(residual(R_n, regressors("decision"), adaptations={k: v for k, v in X.ADAPTATIONS.items() if k != t})) for t in X.ADAPTATIONS}
    return res


# ------------------------------------------------------------------ E3
def e3_run(P, *a, **k):
    """v1's e3, reseeded, and marked not evaluable when an early work has no unit in the band (v1's code would then
    return the correlation of popularity with the works' order in the list, whatever the data)."""
    assert P.dtype == bool
    X.RNG = np.random.default_rng(7)          # v1's e3 draws from the shared generator: reseeded so no value depends on run order
    r = X.e3(P, *a, **k)
    r["evaluable"] = bool(all(w["n_units"] > 0 and np.isfinite(w["offspring"]) for w in r["works"]) and np.isfinite(r["rho"]))
    if not r["evaluable"]:
        r.update(rho=None, p_one_sided=None, null_sd=None, jackknife_min=None, jackknife_positive=None,
                 works=[dict(w, offspring=None) for w in r["works"]])
    return r


def e3_block(Pa, Po, base, pv, full=True):
    res = {"primary": e3_run(Pa, pv)}
    if full:
        res.update({"original_text": e3_run(Po, pv, n_perm=2000), "with_adapted": e3_run(Pa, pv, drop_adapted=False, n_perm=2000),
                    "reverse_time_placebo": e3_run(Pa, pv, reverse=True, n_perm=2000),
                    "background_base_rate_placebo": e3_run(Pa, pv, outcome=base, n_perm=2000)})
    print("  E3 rho", res["primary"]["rho"], "p", res["primary"]["p_one_sided"], "units", res["primary"]["n_units"], "evaluable", res["primary"]["evaluable"], flush=True)
    return res


# ------------------------------------------------------------------ one model at one setting
def centroid_scorers(st, cols, cut=None, cut_cols=None):
    """The centroid of unit shares over the given columns, each weighted by the log of the inverse of its base rate."""
    def one(stt, c, canon, score):
        w = np.log((stt.n_plots + 1) / (stt.base_count[c] + 1))
        return unit_centroid(stt.S[canon][:, c], stt.P[canon][:, c], stt.S[score][:, c], stt.P[score][:, c], w)
    out = {"A_name_free": one(st, cols, "orig", "anon"), "B_canon_from_name_free": one(st, cols, "anon", "anon"),
           "ceiling_original": one(st, cols, "orig", "orig")}
    if "anon_icon" in st.texts:
        out["A_iconography_dropped"] = one(st, cols, "orig", "anon_icon")
    if cut is not None:
        out["openings"] = one(cut, cut_cols, "orig", "anon")
    return out


def unit_scorers(st, cap=C.CAP, cut=None):
    """E1 scorers on a Setting: the canon share on the distinctive units; the centroid of unit shares on all units
    (the registered secondary) and on the distinctive units (a variant)."""
    d = np.flatnonzero(st.distinctive(cap))
    P = {t: st.P[t][:, d] for t in st.texts}
    share = {"A_name_free": CanonShare(P["orig"], P["anon"]), "B_canon_from_name_free": CanonShare(P["anon"], P["anon"]),
             "ceiling_original": CanonShare(P["orig"], P["orig"])}
    if "anon_icon" in P:
        share["A_iconography_dropped"] = CanonShare(P["orig"], P["anon_icon"])
    dc = np.flatnonzero(cut.distinctive(cap)) if cut is not None else None
    if cut is not None:
        share["openings"] = CanonShare(cut.P["orig"][:, dc], cut.P["anon"][:, dc])
    everything = np.arange(len(st.units))
    cent_all = centroid_scorers(st, everything, cut, np.arange(len(cut.units)) if cut is not None else None)
    cent_capped = centroid_scorers(st, d, cut, dc)
    return share, cent_all, cent_capped, d


def core_block(st, cap, rng):
    """The canon of all 32 works and the name-free core, with v1's comparisons (relatives; random sets of ordinary films)."""
    d = st.distinctive(cap)
    Po, Pa = st.P["orig"], st.P["anon"]
    canon = np.flatnonzero(d & (Po[BAT].mean(0) >= 0.8))
    core = canon[Pa[BAT][:, canon].mean(0) >= 0.7]
    cous = np.concatenate([HERO, LIT])
    cous_canon = np.flatnonzero(d & (Po[cous].mean(0) >= 0.8))
    cous_core = cous_canon[Pa[cous][:, cous_canon].mean(0) >= 0.7]
    pseudo, pseudo_nocap = [], []
    for _ in range(50):
        fr = rng.choice(FILMS, 32, replace=False)
        for store, mask in ((pseudo, d), (pseudo_nocap, np.ones(len(d), bool))):
            cn = np.flatnonzero(mask & (Po[fr].mean(0) >= 0.8))
            store.append(int((Pa[fr][:, cn].mean(0) >= 0.7).sum()))
    canon_nocap = np.flatnonzero(Po[BAT].mean(0) >= 0.8)
    core_nocap = canon_nocap[Pa[BAT][:, canon_nocap].mean(0) >= 0.7]
    f = lambda x: float(x) if len(core) else None      # noqa: E731
    out = {"n_canon_units": int(len(canon)), "n_core_units": int(len(core)), "core_units": st.units[core].tolist(),
           "n_canon_units_without_cap": int(len(canon_nocap)), "n_core_units_without_cap": int(len(core_nocap)),
           "relatives_canon_units": int(len(cous_canon)), "relatives_core_units": int(len(cous_core)),
           "random_film_sets_core_units_mean": float(np.mean(pseudo)), "random_film_sets_core_units_max": int(np.max(pseudo)),
           "random_film_sets_core_units_without_cap_mean": float(np.mean(pseudo_nocap)), "random_film_sets_core_units_without_cap_max": int(np.max(pseudo_nocap)),
           "core_present_share_in_name_free_batman": f(Pa[BAT][:, core].mean()) if len(core) else None,
           "core_present_share_in_hero_films": f(Pa[HERO][:, core].mean()) if len(core) else None,
           "core_present_share_in_ordinary_films": f(Pa[FILMS][:, core].mean()) if len(core) else None,
           "core_mean_base_rate_in_background_plots": f(st.base[core].mean()) if len(core) else None,
           "canon_mean_base_rate_in_background_plots": float(st.base[canon].mean()) if len(canon) else None,
           "bounded_by_the_cap_by_construction": ["random_film_sets_core_units_mean", "random_film_sets_core_units_max", "core_present_share_in_ordinary_films",
                                                  "core_mean_base_rate_in_background_plots", "canon_mean_base_rate_in_background_plots"]}
    print(f"  core: {len(core)} of {len(canon)} canon units (without the cap: {len(core_nocap)} of {len(canon_nocap)}); "
          f"random sets of 32 films: {np.mean(pseudo):.1f} with the cap, {np.mean(pseudo_nocap):.1f} without", flush=True)
    return out, canon, core


def run_setting(st, cut=None, cap_e1=C.CAP, cap_e23=None, full=True, save_canon=None, centroid="all"):
    """All three experiments on one loaded Setting. full=False: the primary statistics only (sensitivity runs), with the
    centroid score on all units or on the units under the E1 cap ("capped")."""
    t0 = time.time()
    rng = np.random.default_rng(7)
    Po, Pa = st.P["orig"], st.P["anon"]
    res = {"setting": dict(st.describe(), base_rate_cap_E1=cap_e1, base_rate_cap_E2_E3=cap_e23),
           "n_units_total": st.n_units_total, "n_units_that_can_fire": st.n_can_fire, "n_units_eligible": st.n_eligible,
           "n_units_present_in_some_work": int(len(st.units)),
           "n_distinctive_units_present_in_some_work": int(st.distinctive(cap_e1).sum()),
           "n_background_plots": st.n_plots,
           "units_present_per_batman_work_original_median": float(np.median(Po[BAT].sum(1))),
           "units_present_per_batman_work_name_free_median": float(np.median(Pa[BAT].sum(1))),
           "units_present_per_hero_film_name_free_median": float(np.median(Pa[HERO].sum(1))),
           "units_present_per_ordinary_film_original_median": float(np.median(Po[FILMS].sum(1)))}
    print(f"== {st.tag} {res['setting']}: {len(st.units):,} of {st.n_units_total:,} units present somewhere; "
          f"{res['n_distinctive_units_present_in_some_work']:,} of them at or under the E1 cap", flush=True)
    share, cent, cent_capped, d = unit_scorers(st, cap_e1, cut)
    if full:
        res["E1"] = {"canon_share": e1_block(share, rng, 1000, "canon share"), "unit_centroid": e1_block(cent, rng, 1000, "centroid of unit shares, all units"),
                     "unit_centroid_distinctive_units": e1_block(cent_capped, rng, 1000, "centroid of unit shares, distinctive units")}
        res["E1"]["core"], canon, core = core_block(st, cap_e1, rng)
        if save_canon:
            cnt = Po[BAT][:, d].sum(0)
            folds = {f"fold_{i:02d}": st.units[d[cnt - Po[b, d] >= 0.8 * 31 - 1e-9]] for i, b in enumerate(BAT)}
            np.savez_compressed(save_canon, canon_all32=st.units[canon], core=st.units[core], batman_works=np.array([label[b] for b in BAT]), **folds)
    else:
        res["E1"] = {"canon_share": e1_block({"A_name_free": share["A_name_free"]}, rng, 200, "canon share"),
                     "unit_centroid": e1_block({"A_name_free": (cent_capped if centroid == "capped" else cent)["A_name_free"]}, rng, 200, "centroid of unit shares")}
    k2 = np.flatnonzero(st.distinctive(cap_e23))
    Pa2, Po2 = Pa[:, k2], Po[:, k2]
    pv = json.load(open("data/pageviews.json"))
    if full:
        res["E2"] = e2_block(Pa2, Po2, True)
        res["E3"] = e3_block(Pa2, Po2, st.base[k2], pv, True)
    else:
        ev, R_n = evaluable(Pa2)
        WR = word_rates()
        res["E2"] = {"summary": pick(X.e2(Pa2, n_perm=200, rng=np.random.default_rng(7))), "evaluable": ev, "share_ties_half": share_with_ties(R_n),
                     "sequels_top2": X.sequel_control(Pa2)["top2"],
                     "residual_decision": pick(residual(R_n, regressors("decision"))),
                     "residual_registered_two_word_measures": pick(residual(R_n, regressors("registered"))),
                     "residual_every_word_and_code_measure": pick(residual(R_n, regressors("all")))}
        if cap_e23 is not None:               # under a cap the words get the same cap: word presence on words that rare in the background
            W = word_measures()
            R_cap = X.unit_rates(W["Pw"][:, W["df_bg"] <= cap_e23 * W["n_background_plots"] + 1e-9])
            res["E2"]["residual_cap_matched_words"] = pick(residual(R_n, regressors("decision") + [R_cap]))
            res["E2"]["word_presence_under_the_same_cap"] = pick(X.e2(None, R=R_cap, n_perm=200, rng=np.random.default_rng(7)))
        e3 = e3_run(Pa2, pv, n_perm=2000)
        res["E3"] = {"rho": e3["rho"], "p_one_sided": e3["p_one_sided"], "n_units": e3["n_units"], "jackknife_min": e3["jackknife_min"], "evaluable": e3["evaluable"]}
    res["seconds"] = round(time.time() - t0)
    return res


def run_model(tag, theta=C.THETA, score="top1", which="study", variant="full", cap_e1=C.CAP, cap_e23=None, full=True, save_canon=False):
    st = C.Setting(tag, theta, score, which, variant)
    cut = C.Setting(tag, theta, "top1", which, "cut") if (full and variant == "full") else None
    return run_setting(st, cut, cap_e1, cap_e23, full, C.RESULTS / f"canon_{tag}.npz" if save_canon else None)


RULE = ("v2: a unit is present in a work when its strongest sentence (main sign regime; placeholder codes masked in the name-free texts) "
        "reaches 30% of the unit's strongest activation over the reference corpus, the supplement and the study's original texts. E1 uses "
        "units present in at most 3% of 2,999 background film plots; E2 and E3 use every unit")
FALLBACKS = [{"theta": 0.2, "score": "top1"}, {"theta": 0.3, "score": "top3"}]       # PLAN.md Section 6, in this order


def main():
    args = sys.argv[1:]
    path = C.RESULTS / "experiments_v2.json"
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    out = json.load(open(path)) if path.exists() else {}
    out["rule"] = RULE
    stamp = hashlib.sha256(Path(__file__).read_bytes() + Path("results/v2/reference_manifest.json").read_bytes()).hexdigest()[:16]
    if out.get("words", {}).get("computed_with") != stamp or "words" in args:      # the word block is recomputed whenever this file changes
        print("== word comparators (no model)", flush=True)
        out["words"] = dict(words_block(), computed_with=stamp)
        C.save(out, "experiments_v2.json")
    tags = [a for a in args if a != "words"] or ([] if "words" in args else [t for t in C.MODELS if (C.ACTS / t / "scale.npy").exists()])
    for tag in tags:
        out[tag] = run_model(tag, save_canon=True)
        if not out[tag]["E2"]["evaluable"]["evaluable"]:
            for fb in FALLBACKS:
                r = run_model(tag, full=False, **fb)
                out[tag].setdefault("E2_fallbacks", []).append({"setting": r["setting"], "E2": r["E2"]})
                if r["E2"]["evaluable"]["evaluable"]:
                    break
        C.save(out, "experiments_v2.json")


if __name__ == "__main__":
    main()
