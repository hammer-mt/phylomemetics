"""The Sherlock Holmes corpus: the tests of v2/second/PLAN.md Section 8, and the proof of this code on Batman.

    cd study
    ../.venv/bin/python v2/second/holmes.py regress <tag>        a spec that restates Batman must give v2's committed numbers
    ../.venv/bin/python v2/second/holmes.py words                everything that needs no model -> results/v2/second/holmes_words.json
                                                                 (ends with an error while a word-only measure passes the residual test)
    ../.venv/bin/python v2/second/holmes.py run <tag> [--planes all|primary]     one model -> results/v2/second/holmes_<tag>.json
    ../.venv/bin/python v2/second/holmes.py synthetic            the whole 'run' path on made-up presence bits (then deleted)
    ../.venv/bin/python v2/second/holmes.py followup <tag>       PLAN Section 8's follow-up to a met E1: E1 again with the token after each
                                                                 code masked too -> results/v2/second/holmes_e1_followup_<tag>.json

'run' reads v2/acts_second/holmes/<tag>/{orig,anon}/<plane>.bits.npy (v2/gpu/presence_pass.py), .../bgdrop/counts_ref_scale.npy
(background_pass.py on the background documents left out) and v2/acts/<tag>/bg_counts_ref_scale.npy. The statistics are in
lib.py; the sets come from data/second/holmes/spec.json. This file holds no list of works.

'run' refuses to start unless results/v2/second/holmes_recognition_<tag>.json exists (PLAN Section 10) and
results/v2/second/bgdrop_check_<tag>.json says "exact": true (or --recounted is given). The recognition file is read in
the form v2/second/probes.py writes:
    {"h1": {"<text id>": {"named": true|false, "logp": <float>, ...}, ...},                          all texts; question H1
     "h2": {"<lineage work id>": {"source_named": true|false, "has_source": true|false, ...}, ...}}  the lineage works; question H2
Only 'synthetic' runs without them: it makes up its own recognition file and marks its output as a dry run.
"""
import argparse
import copy
import json
import math
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "v2/second")
import lib                       # noqa: E402
from lib import C, E, X          # noqa: E402

DATA, OUT, CACHE = Path("data/second/holmes"), Path("results/v2/second"), Path("v2/cache/second")
SEED, DRAWS = 20261108, 100000            # PLAN Section 8: the sampled null of E2
CONTROL_NEED = 7                          # PLAN Section 8: at least 7 of the 9 straight adaptations
THETAS = {"20": 0.2, "30": 0.3, "40": 0.4, "50": 0.5}
GRID = [f"{s}_{t}" for s in ("top1", "top3") for t in THETAS]         # the 8 uncapped settings
peak_gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20      # noqa: E731
rng7 = lambda: np.random.default_rng(7)                                              # noqa: E731  (v2's seed)


# ------------------------------------------------------------------ the two specifications
def standin(path):
    """A stand-in for the name-free texts until data/second/holmes/corpus_neutral.jsonl exists (never written under data/):
    every capitalised word that is not the first of its sentence becomes a code, the same code for the same word in a work."""
    import re
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for w in map(json.loads, open(DATA / "corpus.jsonl")):
            names = {}
            for s in w["sentences"]:
                for m in re.findall(r"(?<=[a-z,;] )[A-Z][\w'’-]+", s):
                    names.setdefault(m, f"X{len(names) + 1}")
            pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + r")\b") if names else None
            sents = [pat.sub(lambda m: names[m.group(1)], s) if pat else s for s in w["sentences"]]
            f.write(json.dumps(dict(w, sentences=sents, n_codes=len(names)), ensure_ascii=False) + "\n")
    return path


def holmes_spec(neutral=None, extra=None):
    sp = json.load(open(DATA / "spec.json"))
    real = DATA / "corpus_neutral.jsonl"
    neutral = Path(neutral) if neutral else real if real.exists() else standin(CACHE / "holmes" / "corpus_neutral_standin.jsonl")
    d = {k: sp[k] for k in ("franchise", "negatives", "lineage", "candidates", "adaptations", "baseline", "early", "strata", "icon_regex", "era",
                            "background_documents_left_out")}
    d.update(later=sp["baseline"],                                       # PLAN Section 8: the later works are the 43 baseline works
             curved_regressors=sp.get("curved_regressors", True),       # review, change 1: for Holmes the residual test regresses out squares and products too
             extra_decision_regressors=extra.split(",") if extra else sp.get("decision_regressors_added", []),      # PLAN Section 8's calibration
             strong=list(sp["adaptations_straight"]), covariates={"screen_works_naming_the_story": sp["n_screen_works_naming_the_story"]})
    S = lib.Spec(d, DATA / "corpus.jsonl", neutral, DATA / "pageviews.json")
    ints = lambda m: {int(a): [int(x) for x in v] for a, v in m.items()}        # noqa: E731
    S.subsets = {"straight_adaptations": ints(sp["adaptations_straight"]), "elements_only_works_tier_C_sources": ints(sp["adaptations_elements_only"])}
    S.straight, S.later_wide, S.name_free = S.subsets["straight_adaptations"], lib.A(sp["later"]), {"file": str(neutral), "stand_in": neutral != real}
    # PLAN Section 12, item 5: the franchise narrowed two ways, and the negatives without the film that is also a Batman-study text
    S.franchise_narrow = {"works_where_holmes_appears": lib.A(sp["franchise_where_holmes_appears"]), "films_only": lib.A(sp["franchise_films_only"])}
    S.shared_with_batman, S.statistic = set(sp["texts_shared_with_the_batman_study"]), "mean_over_sources"
    return S


def batman_spec():
    """The Batman corpus restated as a specification, from v1's own constants (read before install rebinds them)."""
    ids = lambda labels: [int(X.L[t]) for t in labels]                  # noqa: E731
    ad = {int(X.L[t]): ids(v) for t, v in X.ADAPTATIONS.items()}
    early = [int(w) for w in X.BAT if X.year[w] < 2000]
    d = {"franchise": X.BAT.tolist(), "negatives": X.HERO_FILM_COUSINS.tolist(), "lineage": X.BAT.tolist(), "candidates": X.COMICS.tolist(),
         "other_negatives": {"ordinary_films": X.FILMS.tolist(), "literary_cousins": X.LIT_COUSINS.tolist()},
         "adaptations": ad, "baseline": X.SCREEN_NONADAPT.tolist(), "early": early,
         "later": [int(w) for w in X.BAT if X.year[w] >= 2000 and int(w) not in ad],                    # experiments.e3
         "strata": {str(w): "comic" if X.medium[w] == "comic" else "screen" for w in early}, "icon_regex": X.ICON.pattern,
         "source_era": X.ERA.tolist(), "strong": ids(X.STRONG), "sequels": {str(X.L[a]): int(X.L[b]) for a, b in X.SEQUELS.items()},
         "sequel_pool": X.SCREEN.tolist()}
    S = lib.Spec(d, "data/corpus.jsonl", "data/corpus_neutral.jsonl", "data/pageviews.json")
    S.subsets = {"strong_subset": {a: v for a, v in ad.items() if a in ids(X.STRONG)}}
    S.straight, S.later_wide, S.name_free = S.subsets["strong_subset"], None, {"file": "data/corpus_neutral.jsonl", "stand_in": False}
    S.franchise_narrow, S.shared_with_batman, S.statistic = {}, set(), "mean_over_adaptations"
    return S


def with_most_adapted_left_out(S):
    """PLAN Section 8: both statistics without the story with the most adaptations (for Holmes, The Hound of the Baskervilles)."""
    n = {}
    for v in S.adaptations.values():
        for s in v:
            n[s] = n.get(s, 0) + 1
    top = max(sorted(n), key=n.get)
    return dict(S.subsets, **{f"without_the_most_adapted_source_{top}": lib.without_source(S.adaptations, top)})


def settings(S, null, extra=None):
    out = {"band": list(S.band), "word_bands": [list(b) for b in S.word_bands], "e2_null": dict(null), "name_free_texts": S.name_free,
           "decision_regressors_added": S.extra_regressors, "curved_regressors": S.curved,
           "sets": {k: len(getattr(S, k)) for k in ("franchise", "negatives", "lineage", "candidates", "adaptations", "baseline", "early", "later")},
           "background_documents_left_out": len(S.dropped_background),
           "sha256": {Path(p).name: lib.sha256(p) for p in (S.paths["corpus"], S.paths["neutral"], S.paths["pageviews"], DATA / "spec.json", DATA / "works.json")}}
    return dict(out, **(extra or {}))


# ------------------------------------------------------------------ everything that needs no model
def n_distinct_codes(sentences):
    return len({m for s in sentences for m in X.CODE.findall(s)})


def word_tests(S, null, n_null=1000, calibrate=True):
    """E.words_block for any corpus: the word comparators through E1 and E2, the positive control on words, the
    calibration of the residual test on words alone, E3's power and the counts the plan quotes."""
    W, WR = E.word_measures(), E.word_rates()
    era = S.era if len(set(S.era)) > 1 else None
    negsets = dict({"negatives": S.negatives}, **S.other_negatives)
    orig, anon, icon = X.lexical_texts(X.works), X.lexical_texts(X.neutral), X.lexical_texts(X.neutral, drop_icon=True)
    vec = X.tfidf().fit(orig + anon)                                    # v1's E1 comparator: one vocabulary over both versions
    T = {k: vec.transform(t).toarray() for k, t in (("o", orig), ("a", anon), ("i", icon), ("oc", E.opening(X.works)), ("ac", E.opening(X.neutral)))}
    pairs = {"A_name_free": ("o", "a"), "B_canon_from_name_free": ("a", "a"), "ceiling_original": ("o", "o"), "A_iconography_dropped": ("o", "i"), "openings": ("oc", "ac")}
    kept = [[s for s in w["sentences"] if not X.ICON.search(s)] or w["sentences"] for w in X.neutral]
    words = lambda ss: sum(len(s.split()) for s in ss)                 # noqa: E731
    fixed = lambda full, ic, cut: {"A_name_free": E.Fixed(full), "A_iconography_dropped": E.Fixed(ic), "openings": E.Fixed(cut)}      # noqa: E731
    e1s = {"tfidf_centroid": {k: E.Centroid.from_vectors(T[a], T[b]) for k, (a, b) in pairs.items()},
           "word_presence_canon_share": {k: E.CanonShare(T[a] > 0, T[b] > 0) for k, (a, b) in pairs.items()},
           "length_alone_sentences": fixed(X.n_sent, [len(k) for k in kept], E.CUTS),
           "length_alone_words": fixed([words(w["sentences"]) for w in X.works], [words(k) for k in kept], [len(t.split()) for t in E.opening(X.works, False)]),
           "distinct_codes_alone": fixed([n_distinct_codes(w["sentences"]) for w in X.neutral], [n_distinct_codes(k) for k in kept],
                                         [n_distinct_codes(w["sentences"][:c]) for w, c in zip(X.neutral, E.CUTS)])}
    out = {"n_words": W["n_words"], "n_words_in_at_most_3pct_of_background_plots": W["n_words_rare"], "n_background_plots": W["n_background_plots"],
           "regressors": {"decision": list(WR)[:E.N_DECISION], "every_measure": list(WR), "added_by_the_calibration": S.extra_regressors,
                          "curved": S.curved, "columns_beside_the_constant": {k: len(lib.regression_columns(lib.regressors(S, k), lib.fit_cells(S), S.curved))
                                                                              for k in ("decision", "every_measure")}}, "E1": {}, "E2": {}}
    for name, sc in e1s.items():
        out["E1"][name] = lib.e1(sc, S.franchise, negsets, rng7(), n_null, era=era)[0]
        a = out["E1"][name]["A_name_free"]
        print(f"  words E1 {name}: pooled {a['negatives']['auc']:.3f}" + (f" within era {a['within_era']['auc']:.3f}" if era is not None else ""), flush=True)
    subsets = with_most_adapted_left_out(S)
    for k in ("tfidf_cosine", "word_presence", "tfidf_background_weights", "word_presence_background_rare", "tfidf_words_and_pairs", "tfidf_raw_counts",
              "code_count_similarity", "code_presence"):
        out["E2"][k] = dict(lib.e2_report(WR[k], S, subsets, jackknife=k in ("tfidf_cosine", "word_presence"), **null),
                            positive_control=lib.positive_control(WR[k], S, S.straight, CONTROL_NEED))
        r = out["E2"][k]["primary"]
        print(f"  words E2 {k}: over adaptations {r['mean_over_adaptations']['share']:.3f} p {r['mean_over_adaptations']['p']:.5f} | over sources "
              f"{r['mean_over_sources']['share']:.3f} p {r['mean_over_sources']['p']:.5f} | control {out['E2'][k]['positive_control']['in_top_two']}", flush=True)
    if calibrate:
        out["residual_test_on_words_alone"] = lib.calibration(S, **null)
        c = out["residual_test_on_words_alone"]
        print("  words alone through the decision test: lowest p", c["lowest_p_mean_over_adaptations"], "(over adaptations)", c["lowest_p_mean_over_sources"],
              "(over sources) | to add until none passes:", {k: v["regressors_added"] for k, v in c["until_none_passes"].items()}, flush=True)
        if S.curved:                        # review, change 1: curved functions of the regressors must not pass, and a planted effect must
            # verification of change 1: the 'further' functions are word-only too and are not polynomials of the regressors (lib.further_measures)
            for what, measures in (("curved", None), ("further", lib.further_measures(S))):
                out[f"residual_test_on_{what}_functions_of_the_regressors"] = cc = lib.curved_calibration(S, measures=measures, **null)
                for form, v in ((k, v) for k, v in cc.items() if isinstance(v, dict)):
                    print(f"  {what} functions through the decision test, {form} ({v['n_columns']} columns): {v['n_passing_on_either_statistic']} of {v['n_tested']} pass; lowest p",
                          {s_: v[f"lowest_p_{s_}"] and v[f"lowest_p_{s_}"]["p"] for s_ in lib.STATS}, flush=True)
                passing = sorted(cc["with_squares_and_pairwise_products"]["passing"].items(), key=lambda kv: min(kv[1].values()))
                print("  CALIBRATION:", f"no {what} function of the regressors passes" if cc["none_passes_with_the_regression_the_tests_use"] else
                      f"FAILED, {len(passing)} {what} functions pass; the lowest: " + json.dumps(dict(passing[:5])), flush=True)
            out["residual_test_on_a_planted_effect"] = pe = lib.planted_effect(S, **null)
            print(f"  planted effect of {pe['boost_in_standard_deviations']} SD through the decision test: passes in", pe["passes_with_the_boost"], "of", pe["n_draws"],
                  "draws; without the boost", pe["passes_without_it"], flush=True)
    out["E3_power_simulated"] = lib.e3_power(S.strata)
    per_source = {str(s): v["n_adaptations"] for s, v in out["E2"]["tfidf_cosine"]["primary"]["by_source"].items()}
    out["counts"] = {"adaptations_per_source": per_source, "sources_with_an_adaptation": len(per_source),
                     "adaptations_on_the_most_adapted_source": max(per_source.values()), "straight_adaptations": len(S.straight),
                     "subsets": {k: len(v) for k, v in subsets.items()},
                     "franchise_by_era": {e: int((S.era[S.franchise] == e).sum()) for e in sorted(set(S.era))},
                     "negatives_by_era": {e: int((S.era[S.negatives] == e).sum()) for e in sorted(set(S.era))},
                     "sentences": int(S.n_sent.sum()), "works_with_page_views": len(S.pv),
                     "share_of_name_free_sentences_matching_the_iconography_list": {
                         g: float(np.mean([np.mean([bool(X.ICON.search(x)) for x in X.neutral[int(i)]["sentences"]]) for i in ids]))
                         for g, ids in (("franchise", S.franchise), ("negatives", S.negatives))}}
    return out


# ------------------------------------------------------------------ one model at one setting
class FromSetting:
    """common.Setting behind the interface of lib.Bits: the Batman regression reads v2's uint8 shares on v2's own scale."""

    def __init__(self, st, cut):
        self.units = st.units
        at = np.searchsorted(st.units, cut.units)
        assert (st.units[at] == cut.units).all(), "a unit present in an opening is present in the whole text"
        self.P = {("orig", "top1_30"): st.P["orig"], ("anon", "top1_30"): st.P["anon"], ("anon", "icon_top1_30"): st.P["anon_icon"]}
        for k in ("orig", "anon"):
            self.P[(k, "cut_top1_30")] = np.zeros(st.P["orig"].shape, bool)
            self.P[(k, "cut_top1_30")][:, at] = cut.P[k]

    def has(self, k, plane):
        return (k, plane) in self.P

    def count(self, k, plane, rows):
        return self.P[(k, plane)][rows].sum(0)

    def presence(self, k, plane, rows, cols=None, compact=False):
        sub = self.P[(k, plane)][rows] if cols is None else self.P[(k, plane)][rows][:, cols]
        if compact:
            return sub
        out = np.zeros((self.P[(k, plane)].shape[0], sub.shape[1]), bool)
        out[rows] = sub
        return out


def unit_tests(S, src, base, n_plots, null, n_null=1000, keep=None, full=True, plane="top1_30", e1_only=False):
    """E1, E2, the residual test and E3 (PLAN Section 8). src: lib.Bits or FromSetting; base: the base count of each unit
    of src.units; keep: bool over src.units, the units used (None: all). full=False: the four tests without their subsets,
    jackknives and placebos. e1_only: stop after E1 (the follow-up of PLAN Section 8 reads nothing else)."""
    t0, cnt = time.time(), {}
    keep = np.ones(len(src.units), bool) if keep is None else keep
    era = S.era if len(set(S.era)) > 1 else None
    negsets = dict({"negatives": S.negatives}, **S.other_negatives)
    pool, d = np.concatenate([S.franchise] + list(negsets.values())), keep & lib.distinctive(base, n_plots)

    def e1_for(franchise):
        need = math.ceil(0.8 * (len(franchise) - 1) - 1e-9)

        def scorer(c_text, c_plane, s_text, s_plane):
            """The canon share on the distinctive units. Only units that could enter a canon are loaded (present, in the texts
            the canon is chosen on, in 80% of n - 1 works of the franchise and its negatives together), which loses nothing."""
            if (c_text, c_plane) not in cnt:
                cnt[(c_text, c_plane)] = src.count(c_text, c_plane, pool)
            cols = np.flatnonzero(d & (cnt[(c_text, c_plane)] >= need))
            Pc = src.presence(c_text, c_plane, pool, cols)
            return E.CanonShare(Pc, Pc if (c_text, c_plane) == (s_text, s_plane) else src.presence(s_text, s_plane, pool, cols))

        scorers = {"A_name_free": scorer("orig", plane, "anon", plane), "B_canon_from_name_free": scorer("anon", plane, "anon", plane),
                   "ceiling_original": scorer("orig", plane, "orig", plane)}                  # v2's order: its random draws follow it
        if src.has("anon", "icon_top1_30"):
            scorers["A_iconography_dropped"] = scorer("orig", plane, "anon", "icon_top1_30")
        if src.has("orig", "cut_top1_30") and src.has("anon", "cut_top1_30"):
            scorers["openings"] = scorer("orig", "cut_top1_30", "anon", "cut_top1_30")
        return lib.e1(scorers, franchise, negsets, rng7(), n_null, era=era)

    res = {"plane": plane, "n_units_present_in_some_text": int(len(src.units)), "n_units_used": int(keep.sum()), "n_distinctive_units_used": int(d.sum()),
           "n_background_plots": int(n_plots)}
    res["E1"], mats = e1_for(S.franchise)
    if full:                                                            # PLAN Section 12, item 5
        for name, ids in S.franchise_narrow.items():
            res[f"E1_franchise_narrowed_to_{name}"] = dict(e1_for(ids)[0], n_franchise=len(ids))
        if mats.get("A_name_free") is not None and (S.shared_with_batman & set(S.negatives.tolist())):
            keep_neg = np.array([j not in S.shared_with_batman for j in S.negatives.tolist()])
            M = mats["A_name_free"][:, keep_neg]
            res["E1_without_the_negative_shared_with_the_batman_study"] = {
                "left_out": sorted(S.shared_with_batman & set(S.negatives.tolist())), "n_negatives": int(keep_neg.sum()), "auc_pooled": float(M.mean()),
                "auc_within_era": float(lib.within_era(M, S.era[S.franchise], S.era[S.negatives][keep_neg])) if era is not None else None}
    if e1_only:
        return res, mats, None, None
    a = res["E1"]["A_name_free"]
    print(f"  E1 pooled {a['negatives']['auc']:.3f} within era {a.get('within_era', {}).get('auc', float('nan')):.3f} canon {a['canon_units_median']} "
          f"-> {res['E1'].get('verdict_within_era', res['E1']['verdict_pooled'])['verdict']}", flush=True)

    Pl = src.presence("anon", plane, S.lineage, np.flatnonzero(keep), compact=True)              # [lineage works, units used]
    R, ev = lib.full_rates(S, Pl)
    dec, subsets = lib.regressors(S, "decision"), with_most_adapted_left_out(S)
    res["E2"] = dict(lib.e2_report(R, S, subsets if full else None, jackknife=full, **null), evaluable=ev,
                     positive_control=lib.positive_control(R, S, S.straight, CONTROL_NEED))
    res["residual"] = {"decision": lib.residual(R, dec, S, **null), "every_measure": lib.residual(R, lib.regressors(S, "every_measure"), S, **null)}
    if full:
        res["residual"]["registered_two_word_measures"] = lib.residual(R, lib.regressors(S, "registered"), S, **null)
        Ro, evo = lib.full_rates(S, src.presence("orig", plane, S.lineage, np.flatnonzero(keep), compact=True))          # PLAN Section 12, item 8
        res["E2"]["original_texts_ceiling"] = dict(lib.brief(lib.e2(Ro, S, **null)), evaluable=evo)
        del Ro
        res["residual"]["decision_on_subsets"] = {k: lib.brief(lib.residual(R, dec, S, ad, **null)) for k, ad in subsets.items()}
        res["residual"]["decision_leave_one_source_out"] = {str(s): lib.brief(lib.residual(R, dec, S, lib.without_source(S.adaptations, s), **null))
                                                            for s in sorted({s for v in S.adaptations.values() for s in v})}
        loo = lib.leave_one_adaptation_out({"E2": lambda ad: lib.e2(R, S, ad, **null), "decision": lambda ad: lib.residual(R, dec, S, ad, **null)}, S)     # review, change 3
        res["E2"]["leave_one_adaptation_out"], res["residual"]["decision_leave_one_adaptation_out"] = loo["E2"], loo["decision"]
    e, r = res["E2"]["primary"], res["residual"]
    print(f"  E2 over adaptations {e['mean_over_adaptations']['share']:.3f} p {e['mean_over_adaptations']['p']:.5f} | over sources {e['mean_over_sources']['share']:.3f} "
          f"p {e['mean_over_sources']['p']:.5f} | evaluable {ev['evaluable']} | control {res['E2']['positive_control']['in_top_two']}/{res['E2']['positive_control']['n']} "
          f"| residual p: decision {r['decision']['mean_over_adaptations']['p']:.4f} / {r['decision']['mean_over_sources']['p']:.4f}, "
          f"every measure {r['every_measure']['mean_over_adaptations']['p']:.4f} / {r['every_measure']['mean_over_sources']['p']:.4f}", flush=True)

    pos = {int(w): i for i, w in enumerate(S.lineage)}
    rows = lambda ids: Pl[[pos[int(w)] for w in ids]]                   # noqa: E731
    Pe, later = rows(S.early), rows(S.later).sum(0) / len(S.later)
    res["E3"] = {"primary": lib.e3(Pe, later, S, len(S.later), covariates=S.covariates)}
    if full:
        res["E3"]["background_base_rate_placebo"] = lib.e3(Pe, base[keep] / n_plots, S, 0, n_perm=2000, covariates=S.covariates)
        if S.covariates:
            res["E3"]["v2_covariates_only"] = lib.e3(Pe, later, S, len(S.later), n_perm=2000)
        if S.later_wide is not None:
            res["E3"]["later_set_widened"] = lib.e3(Pe, rows(S.later_wide).sum(0) / len(S.later_wide), S, len(S.later_wide), n_perm=2000, covariates=S.covariates)
    print("  E3 rho", res["E3"]["primary"]["rho"], "p", res["E3"]["primary"]["p_one_sided"], "units", res["E3"]["primary"]["n_units"], flush=True)
    k, v1 = S.statistic, res["E1"].get("verdict_within_era", res["E1"]["verdict_pooled"])        # PLAN Section 13: the four tests, one p each
    # review, change 6: where E2 is not evaluable, E2 and the residual test (E2 on residuals) are reported as not evaluable and give no p
    state = "evaluable" if ev["evaluable"] else "not evaluable"
    r["evaluable"] = ev["evaluable"]
    res["confirmatory"] = {"E2_statistic": k, "E1_verdict": v1["verdict"], "E1_openings_auc": v1["openings_auc"], "E1_openings_auc_below_0.70": v1["openings_auc_below_0.70"],
                           "E2": state, "residual_test": state, "E2_p": e[k]["p"] if ev["evaluable"] else None,
                           "residual_p_larger_of_the_two_forms": max(r["decision"][k]["p"], r["every_measure"][k]["p"]) if ev["evaluable"] else None,
                           "E3_p": res["E3"]["primary"]["p_one_sided"]}
    res["seconds"] = round(time.time() - t0, 1)
    return res, mats, R, Pl


# ------------------------------------------------------------------ the Batman regression
def regress(a):
    tag = a.tag
    T = json.load(open("results/v2/experiments_v2.json"))
    t, tw = T[tag], T["words"]
    S = lib.install(batman_spec(), acts="v2/acts", results=CACHE / "regress")
    exact, sampled = dict(enumerate_all=True), dict(n_draws=a.draws, seed=SEED)
    t0 = time.time()
    st = C.Setting(tag)
    src = FromSetting(st, C.Setting(tag, C.THETA, "top1", "study", "cut"))
    res, mats, R, _ = unit_tests(S, src, st.base_count, st.n_plots, exact)
    secs, mem = time.time() - t0, peak_gb()
    A1, c1 = res["E1"], t["E1"]["canon_share"]
    e, rs, e3, ts = res["E2"], res["residual"], res["E3"], t["E2"]["residual_summary"]
    oa = lambda r: r["mean_over_adaptations"]                           # noqa: E731
    rows = [("* E1 AUC (pooled)", c1["A_name_free"]["hero_film_cousins"]["auc"], A1["A_name_free"]["negatives"]["auc"]),
            ("E1 interval, lower", c1["A_name_free"]["hero_film_cousins"]["ci95"][0], A1["A_name_free"]["negatives"]["ci95"][0]),
            ("E1 interval, upper", c1["A_name_free"]["hero_film_cousins"]["ci95"][1], A1["A_name_free"]["negatives"]["ci95"][1]),
            ("E1 label-shuffle null p95", c1["A_name_free"]["null_label_shuffle"]["p95"], A1["A_name_free"]["null_label_shuffle"]["p95"]),
            ("E1 label-shuffle p", c1["A_name_free"]["null_label_shuffle"]["p_value"], A1["A_name_free"]["null_label_shuffle"]["p_value"]),
            ("E1 canon units, median", c1["A_name_free"]["canon_units_median"], A1["A_name_free"]["canon_units_median"]),
            ("E1 against ordinary films", c1["A_name_free"]["ordinary_films"]["auc"], A1["A_name_free"]["ordinary_films"]["auc"]),
            ("E1 canon from name-free", c1["B_canon_from_name_free"]["hero_film_cousins"]["auc"], A1["B_canon_from_name_free"]["negatives"]["auc"]),
            ("E1 original-text ceiling", c1["ceiling_original"]["hero_film_cousins"]["auc"], A1["ceiling_original"]["negatives"]["auc"]),
            ("E1 iconography dropped", c1["A_iconography_dropped"]["hero_film_cousins"]["auc"], A1["A_iconography_dropped"]["negatives"]["auc"]),
            ("E1 openings", c1["openings"]["hero_film_cousins"]["auc"], A1["openings"]["negatives"]["auc"]),
            ("E1 openings null p95", c1["openings"]["null_label_shuffle"]["p95"], A1["openings"]["null_label_shuffle"]["p95"]),
            ("E1 verdict", c1["verdict"], A1["verdict_pooled"]["verdict"]),
            ("* E2 share", t["E2"]["summary"]["share"], oa(e["primary"])["share"]), ("* E2 exact p", t["E2"]["summary"]["exact_p"], oa(e["primary"])["p"]),
            ("E2 mean over all maps", t["E2"]["summary"]["chance_mean"], oa(e["primary"])["null_mean"]),
            ("E2 share, ties as half", t["E2"]["share_ties_half"], oa(e["primary"])["share_ties_half"]),
            ("E2 optional units", t["E2"]["evaluable"]["n_optional_units"], e["evaluable"]["n_optional_units"]),
            ("E2 optional units, least per comic", t["E2"]["evaluable"]["optional_units_per_comic_min"], e["evaluable"]["optional_units_per_candidate_min"]),
            ("E2 strong subset, exact p", t["E2"]["strong_subset"]["chance"]["joint_relabelling_exact"]["p"], oa(e["strong_subset"])["p"]),
            ("E2 original texts, exact p", t["E2"]["original_upper_bound"]["chance"]["joint_relabelling_exact"]["p"], oa(e["original_texts_ceiling"])["p"]),
            ("E2 sequel control, top two (X.sequel_control)", t["E2"]["sequels_name_free"]["top2"], X.sequel_control(st.P["anon"], S.band)["top2"]),
            ("* residual, decision p", ts["decision_words_by_frequency"]["exact_p"], oa(rs["decision"])["p"]),
            ("residual, decision share", ts["decision_words_by_frequency"]["share"], oa(rs["decision"])["share"]),
            ("residual, decision R2", ts["decision_words_by_frequency"]["r2_of_fit"], rs["decision"]["r2_of_fit"]),
            ("residual, every measure p", ts["every_word_and_code_measure"]["exact_p"], oa(rs["every_measure"])["p"]),
            ("residual, v1's two measures p", ts["registered_both_word_measures"]["exact_p"], oa(rs["registered_two_word_measures"])["p"]),
            ("* E3 rho", t["E3"]["primary"]["rho"], e3["primary"]["rho"]), ("* E3 p", t["E3"]["primary"]["p_one_sided"], e3["primary"]["p_one_sided"]),
            ("E3 units", t["E3"]["primary"]["n_units"], e3["primary"]["n_units"]), ("E3 jackknife minimum", t["E3"]["primary"]["jackknife_min"], e3["primary"]["jackknife_min"]),
            ("E3 base-rate placebo rho", t["E3"]["background_base_rate_placebo"]["rho"], e3["background_base_rate_placebo"]["rho"]),
            ("E3 base-rate placebo p", t["E3"]["background_base_rate_placebo"]["p_one_sided"], e3["background_base_rate_placebo"]["p_one_sided"])]
    # the pieces written new, against the originals they replace
    checks = {"lib.rates equals X.unit_rates on the units": bool(np.array_equal(R, X.unit_rates(st.P["anon"]))),
              "with one era the within-era AUC is the pooled AUC": bool(abs(lib.within_era(mats["A_name_free"], S.era[S.franchise], S.era[S.negatives])
                                                                            - res["E1"]["A_name_free"]["negatives"]["auc"]) < 1e-12)}
    # the packed-bit reader: Batman's presence written as presence_pass.py writes it, read back, and the tests run again
    tmp = CACHE / "regress" / tag
    for (k, plane), P in src.P.items():
        (tmp / k).mkdir(parents=True, exist_ok=True)
        m = np.lib.format.open_memmap(tmp / k / f"{plane}.bits.npy", mode="w+", dtype=np.uint8, shape=(S.N, st.n_units_total // 8))
        for w in range(S.N):
            row = np.zeros(st.n_units_total, bool)
            row[st.units] = P[w]
            m[w] = np.packbits(row)
        m.flush()
        json.dump({"n_texts": S.N, "ids": list(range(S.N)), "n_layers": 1, "width": st.n_units_total // 2}, open(tmp / k / "meta.json", "w"))
    bits = lib.Bits({"orig": tmp / "orig", "anon": tmp / "anon"}, list(src.P), C.scale(tag))
    checks["lib.Bits finds the same units as common.Setting"] = bool(np.array_equal(bits.units, st.units))
    again = unit_tests(S, bits, st.base_count, st.n_plots, exact, full=False)[0]
    shutil.rmtree(tmp)
    same = [again["E1"]["A_name_free"]["negatives"] == A1["A_name_free"]["negatives"], oa(again["E2"]["primary"]) == oa(e["primary"]),
            oa(again["residual"]["decision"]) == oa(rs["decision"]), again["E3"]["primary"]["rho"] == e3["primary"]["rho"],
            again["E3"]["primary"]["p_one_sided"] == e3["primary"]["p_one_sided"]]
    checks["the tests through lib.Bits give the same numbers (E1, E2, residual, E3)"] = bool(all(same))
    # the sampled null against the exact one
    smp = [("E2, units", oa(e["primary"])["p"], oa(lib.e2(R, S, **sampled))["p"]),
           ("residual decision, units", oa(rs["decision"])["p"], oa(lib.residual(R, lib.regressors(S, "decision"), S, **sampled))["p"])]
    out = {"tag": tag, "curved_regressors": S.curved, "seconds_setting_and_tests": round(secs, 1), "peak_memory_gb_setting_and_tests": mem}
    if a.words:
        w = word_tests(S, exact)
        WR, cal, c = E.word_rates(), w["residual_test_on_words_alone"], tw["residual_test_on_words_alone"]
        rows += [(f"words E2 {k}, share", tw["E2"][k]["primary"]["chance"]["observed_unrounded"], oa(w["E2"][k]["primary"])["share"]) for k in w["E2"]]
        rows += [(f"words E2 {k}, exact p", tw["E2"][k]["primary"]["chance"]["joint_relabelling_exact"]["p"], oa(w["E2"][k]["primary"])["p"]) for k in w["E2"]]
        rows += [(f"words alone, decision p: {k}", c[k]["decision_test"]["exact_p"], oa(v)["p"]) for k, v in cal["measures"].items()]
        rows += [("words alone, lowest decision p", tw["residual_test_on_words_alone_lowest_decision_p"], cal["lowest_p_mean_over_adaptations_with_the_regressor_itself"])]
        rows += [(f"words E1 {k}", tw["E1"][k]["A_name_free"]["hero_film_cousins"]["auc"], w["E1"][k]["A_name_free"]["negatives"]["auc"])
                 for k in w["E1"] if k in tw["E1"]]
        rows += [(f"words E1 {k}, openings", tw["E1"][k]["openings"]["hero_film_cousins"]["auc"], w["E1"][k]["openings"]["negatives"]["auc"]) for k in w["E1"] if k in tw["E1"]]
        rows += [("words E1 tfidf_centroid, null p95", tw["E1"]["tfidf_centroid"]["A_name_free"]["null_label_shuffle"]["p95"],
                  w["E1"]["tfidf_centroid"]["A_name_free"]["null_label_shuffle"]["p95"])]
        pw = json.load(open("results/experiments.json"))["E3_power_simulated"]
        rows += [(f"E3 power, rho {r}", pw[f"power_rho_{r}"], w["E3_power_simulated"][f"power_rho_{r}"]) for r in (0.3, 0.5)]
        smp += [(f"words {k}", oa(w["E2"][k]["primary"])["p"], oa(lib.e2(WR[k], S, **sampled))["p"]) for k in ("tfidf_cosine", "word_presence")]
        smp += [("words tfidf_cosine, 400,000 draws", oa(w["E2"]["tfidf_cosine"]["primary"])["p"], oa(lib.e2(WR["tfidf_cosine"], S, n_draws=400000, seed=SEED))["p"])]
    print(f"\n{tag}: target (results/v2/experiments_v2.json) against this code; every row must match (floats within 1e-12); * = the six required numbers")
    bad = 0
    for name, want, got in rows:
        ok = want == got or (isinstance(want, float) and abs(want - got) < 1e-12)
        bad += not ok
        print(f"  {name:58s} {want!r:>24} {got!r:>24}  {'same' if want == got else 'within 1e-12' if ok else 'DIFFERS'}")
    print("  sampled null against the exact one (exact, sampled):")
    for name, ex, sm in smp:
        print(f"  {name:58s} {ex!r:>24} {sm!r:>24}")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"  Setting and tests: {secs:.0f} s, peak memory {mem:.2f} GB (whole run {peak_gb():.2f} GB)")
    out.update(rows=[{"what": n, "target": w_, "this_code": g, "same": w_ == g} for n, w_, g in rows], checks=checks,
               sampled_against_exact=[{"what": n, "exact": x, "sampled": s_} for n, x, s_ in smp], peak_memory_gb=peak_gb(),
               passed=bool(bad == 0 and all(checks.values())))
    lib.save(out, OUT / f"batman_regression_{tag}.json")
    print("REGRESSION", "PASSED" if out["passed"] else "FAILED")
    sys.exit(0 if out["passed"] else 1)



# ------------------------------------------------------------------ one model on the Holmes corpus
def grid(S, src, null, keep):
    """PLAN Section 8: E2 and the decision test at the uncapped settings (20 to 50% of scale; strongest sentence or mean of three)."""
    out, dec = {}, lib.regressors(S, "decision")
    for name in GRID:
        if src.has("anon", name):
            R, ev = lib.full_rates(S, src.presence("anon", name, S.lineage, np.flatnonzero(keep), compact=True))
            out[name] = {"evaluable": ev, "E2": lib.brief(lib.e2(R, S, **null)), "residual_decision": lib.brief(lib.residual(R, dec, S, **null))}
    n = {f"settings_where_the_decision_test_passes_{k}": sum(v["evaluable"]["evaluable"] and v["residual_decision"][k]["p"] < 0.05 for v in out.values())
         for k in ("mean_over_sources", "mean_over_adaptations")}
    return dict(out, n_settings=len(out), **n)


def recognition(S, rec, e1, M, R, null, evaluable=True, n_boot=2000):
    """PLAN Section 10's fixed uses: E1's score against H1's log-probability; E1's within-era AUC, with its bootstrap
    interval, for the screen works the model names (H1) and for the rest; E2 and the decision test for the primary
    adaptations whose source it names (H2) and for the rest. Then the computed rule for 'not confined to works the model
    names' (review, change 2), read each time on the side the model does NOT name:
      E1             'holds' if the within-era AUC of the screen works not named is at least 0.70 and its interval
                     excludes 0.5; 'cannot be separated from recognition' if fewer than 10 screen works are not named;
                     otherwise 'confined to named works'. M is E1's own matrix: the works are scored on the full
                     folds' canons and no canon is rebuilt.
      residual test  'holds' if the decision test on the adaptations whose source is not named has p below 0.05 (the
                     corpus's statistic, its own relabellings among the candidates, the same seed); 'cannot be separated
                     from recognition' if there are fewer than 5 such adaptations; otherwise 'confined'. Where E2 is not
                     evaluable, 'not evaluable'.
    E1's rule likewise says 'not evaluable' where E1's own verdict does (its canon is too small: PLAN Section 8 decides that
    first, and Section 11 gives it no reading); found when change 2 was verified. The named side's numbers are descriptive."""
    from scipy.stats import spearmanr
    h1, h2, fr, dec, k = rec["h1"], rec["h2"], S.franchise, lib.regressors(S, "decision"), S.statistic
    named, sc = np.array([bool(h1[str(int(w))]["named"]) for w in fr]), e1["A_name_free"]["held_out_score"]
    ok = [i for i, x in enumerate(sc) if x is not None and h1[str(int(fr[i]))]["logp"] is not None]
    rho = spearmanr([sc[i] for i in ok], [h1[str(int(fr[i]))]["logp"] for i in ok])[0] if len(ok) > 2 else None       # ties get their mean rank
    out = {"E1_score_against_H1_log_probability": {"spearman_rho": None if rho is None or not np.isfinite(rho) else float(rho), "n": len(ok)}, "E1": {}, "E2": {}}
    ef, en = S.era[fr], S.era[S.negatives]
    for side, m in (("named", named), ("not_named", ~named)):
        out["E1"][side] = {"n": int(m.sum()), "too_few_to_read": bool(m.sum() < 10)}
        if m.sum() >= 10:
            out["E1"][side].update(auc_within_era=lib.within_era(M[m], ef[m], en), ci95=lib.boot_within_era(M[m], ef[m], en, n_boot, rng7()), auc_pooled=float(M[m].mean()))
    for side, want in (("source_named", True), ("source_not_named", False)):
        ad = {w: v for w, v in S.adaptations.items() if bool(h2[str(w)]["source_named"]) == want}
        out["E2"][side] = {"n": len(ad), "adaptations": sorted(ad), "too_few_to_read": len(ad) < 5}
        if len(ad) >= 5:
            out["E2"][side].update(E2=lib.brief(lib.e2(R, S, ad, **null)), residual_decision=lib.brief(lib.residual(R, dec, S, ad, **null)))
    a, b = out["E1"]["not_named"], out["E2"]["source_not_named"]
    out["not_confined_to_works_the_model_names"] = {
        "E1": "not evaluable" if e1.get("verdict_within_era", e1["verdict_pooled"])["verdict"] == "not evaluable" else
              "cannot be separated from recognition" if a["too_few_to_read"] else
              "holds" if a["auc_within_era"] >= 0.70 and not a["ci95"][0] <= 0.5 <= a["ci95"][1] else "confined to named works",
        "E1_rule": "screen works not named in H1, on the full folds' canons: within-era AUC at least 0.70 and an interval that excludes 0.5; fewer than 10 such works cannot be read",
        "residual_test": "not evaluable" if not evaluable else "cannot be separated from recognition" if b["too_few_to_read"] else
                         "holds" if b["residual_decision"][k]["p"] < 0.05 else "confined",
        "residual_test_rule": f"primary adaptations whose source is not named in H2: decision test p ({k}) below 0.05; fewer than 5 such adaptations cannot be read",
        "reading": "each says where an effect sits, so each matters only where the test itself passes (PLAN Section 10)"}
    return out


def refuse(why):
    sys.exit("holmes.py run: REFUSED. " + why)


def run(a, S=None, dry_run=None):
    """dry_run: the made-up recognition file of 'synthetic', which alone may skip the two conditions of a real run
    (review, change 5): the model's recognition file is there, and the background plots left out are known to be counted
    as v2 counted them (the check is exact, or the counts were made again over the whole background)."""
    rec_path, check = OUT / f"holmes_recognition_{a.tag}.json", OUT / f"bgdrop_check_{a.tag}.json"
    if dry_run:
        rec_path = Path(dry_run)
    else:
        if not rec_path.exists():
            refuse(f"{rec_path} does not exist. The recognition questions come before any analysis (PLAN Section 10): "
                   f"gpu.py probes {a.tag} holmes_h1, then holmes_h2, then probes.py score {a.tag}.")
        if not a.recounted and not (check.exists() and json.load(open(check)).get("exact") is True):
            refuse(f"{check} does not exist or does not say \"exact\": true, so the base counts cannot be taken from v2's stored counts (PLAN Section 6). "
                   f"Run gpu.py bgdrop {a.tag}; if it is not exact, gpu.py bgfull {a.tag} and then this command with --recounted --stored <the recount>.")
    S = S or lib.install(holmes_spec(a.neutral, a.extra_regressors))
    rec = json.load(open(rec_path))
    lacking = [f"h1 {int(w)}" for w in S.franchise if str(int(w)) not in rec.get("h1", {})] + [f"h2 {w}" for w in S.adaptations if str(w) not in (rec.get("h2") or {})]
    if lacking:
        refuse(f"{rec_path} lacks {len(lacking)} of the replies the rule of PLAN Section 10 needs (H1 for every screen work, H2 for every primary adaptation); "
               f"the first: {', '.join(lacking[:5])}.")
    null, t0, root = dict(n_draws=a.draws, seed=SEED), time.time(), Path(a.acts_root) / a.tag
    stored = Path(a.stored or f"v2/acts/{a.tag}/bg_counts_ref_scale.npy")
    dirs = {"orig": root / "orig", "anon": root / "anon"}
    want = [("orig", "top1_30"), ("anon", "top1_30"), ("anon", "icon_top1_30"), ("orig", "cut_top1_30"), ("anon", "cut_top1_30")]
    want += [("anon", g) for g in GRID] if a.planes == "all" else []
    planes = [(k, p) for k, p in want if (dirs[k] / f"{p}.bits.npy").exists()]
    assert set(want[:2]) <= set(planes), f"no primary planes under {root}"
    src = lib.Bits(dirs, planes, np.load(stored.parent / "scale_ref.npy") if (stored.parent / "scale_ref.npy").exists() else None)
    n_stored = json.load(open(stored.parent / "bg_meta.json"))["n_plots"] if (stored.parent / "bg_meta.json").exists() else 2999
    n_background = len(C.background_docs())        # PLAN Section 6: v2's 2,999 plots without those left out (2,965 for Holmes), counted from the file
    if a.recounted:            # PLAN Section 6: where the 34 plots' stored presence is not reproduced, --stored holds a full recount without them
        # verification of change 5: the recount's own bg_meta.json gives its number of plots, so that v2's counts over 2,999 plots (the default of --stored) stop at the assert below
        base, n_plots = lib.base_counts(stored, None, C.THETA, n_stored)
    else:
        base, n_plots = lib.base_counts(stored, root / "bgdrop" / "counts_ref_scale.npy", C.THETA, n_stored, len(S.dropped_background))
    assert n_plots == n_background, f"the base counts are over {n_plots} plots, the background has {n_background}"
    base, every = base[src.units], np.arange(S.N)
    over = (src.count("orig", "over100", every) > 0) | (src.count("anon", "over100", every) > 0)      # above 100% in any text, either version
    print(f"== {a.tag}: {len(src.units):,} of {src.n2u:,} units present somewhere in {planes}; {int(over.sum()):,} above their scale somewhere; "
          f"{n_plots} background plots", flush=True)
    out = settings(S, null, {"tag": a.tag, "dry_run_on_made_up_presence": bool(dry_run), "recognition_file": str(rec_path),
                             "base_counts": "made up" if dry_run else "recounted over the whole background (--recounted)" if a.recounted else
                                            f"v2's stored counts minus the counts of the plots left out ({check} says exact)",
                             "planes": [f"{k}/{p}" for k, p in planes], "acts": str(root), "stored_counts": str(stored), "n_units_total": src.n2u,
                             "presence_pass": {k: {x: m.get(x) for x in ("scale_sha256", "texts_sha256", "mask_codes", "provenance")} for k, m in src.meta.items()}})
    res, mats, R, _ = unit_tests(S, src, base, n_plots, null)
    out.update(res)
    out["over100"] = {"n_units_above_their_scale_in_some_text": int(over.sum()), "tests_without_them": unit_tests(S, src, base, n_plots, null, keep=~over, full=False)[0]}
    k, ok = S.statistic, res["E2"]["evaluable"]["evaluable"]
    out["recognition"] = recognition(S, rec, res["E1"], mats["A_name_free"], R, null, evaluable=ok)
    rule = out["recognition"]["not_confined_to_works_the_model_names"]
    out["confirmatory"].update(E1_not_confined_to_works_the_model_names=rule["E1"], residual_test_not_confined_to_pairs_whose_source_the_model_names=rule["residual_test"])
    if a.planes == "all":
        out["uncapped_settings"] = grid(S, src, null, np.ones(len(over), bool))
        out["uncapped_settings_without_over100_units"] = grid(S, src, null, ~over)
        n = out["uncapped_settings"][f"settings_where_the_decision_test_passes_{k}"]
        # PLAN Section 8: all four must hold; the fourth is the computed rule (review, change 2). With E2 not evaluable there is nothing to claim either way.
        four = {"decision_p_below_0.05": res["residual"]["decision"][k]["p"] < 0.05, "every_measure_p_below_0.05": res["residual"]["every_measure"][k]["p"] < 0.05,
                "at_least_4_of_8_settings": n >= 4, "not_confined_to_pairs_whose_source_the_model_names": rule["residual_test"] == "holds"}
        # PLAN Section 8 as frozen: on Holmes the residual test is REPORTED and is NOT confirmatory; whatever these say, no claim is made from them
        out["residual_test_reported_not_confirmatory"] = dict(four if ok else dict.fromkeys(four), settings_passing=n, recognition_rule=rule["residual_test"],
                                                              evaluable=ok, all_four_hold=bool(all(four.values())) if ok else None,
                                                              reading="reported only: the paper does not say that units add to words on Holmes (PLAN Section 8)")
    if not ok:      # review, change 6: E2 and the residual test stay "not evaluable" (no confirmatory p); the first evaluable fallback setting is shown for description
        g = next((g for g in ("top1_20", "top3_30") if out.get("uncapped_settings", {}).get(g, {}).get("evaluable", {}).get("evaluable")), None)
        out["E2_fallback"] = {"label": "descriptive only: E2 is not evaluable at the primary setting, so E2 and the residual test are reported as not evaluable "
                                       "and contribute no confirmatory p; this fallback setting decides nothing",
                              "order_tried": ["top1_20", "top3_30"], "setting": g, "results": out["uncapped_settings"][g] if g else None,
                              "why_none": None if g else "neither fallback setting is evaluable" if a.planes == "all" else "the fallback planes were not read (--planes primary)"}
    out["seconds"], out["peak_memory_gb"] = round(time.time() - t0, 1), peak_gb()
    name = f"holmes_{a.tag}.json"
    lib.save(out, a.out or (CACHE / "holmes" / name.replace(".json", "_STANDIN.json") if S.name_free["stand_in"] else OUT / name))
    return out


def words(a):
    S = lib.install(holmes_spec(a.neutral, a.extra_regressors))
    null, t0 = dict(n_draws=a.draws, seed=SEED), time.time()
    out = dict(settings(S, null), **word_tests(S, null))
    c = out["residual_test_on_words_alone"]
    # verification of change 1: the file says in one place whether any word-only measure passes (v2's list, the curved functions, the further functions)
    out["no_word_only_measure_passes_the_residual_test"] = bool(min(c["lowest_p_mean_over_sources"], c["lowest_p_mean_over_adaptations"]) >= 0.05 and all(
        out.get(f"residual_test_on_{what}_functions_of_the_regressors", {}).get("none_passes_with_the_regression_the_tests_use", True) for what in ("curved", "further")))
    out["seconds"], out["peak_memory_gb"] = round(time.time() - t0, 1), peak_gb()
    lib.save(out, a.out or (CACHE / "holmes" / "holmes_words_STANDIN.json" if S.name_free["stand_in"] else OUT / "holmes_words.json"))
    if not out["no_word_only_measure_passes_the_residual_test"]:        # this is why PLAN Section 8 reports the residual test on Holmes and makes no claim from it
        print("holmes.py words: the residual test is not calibrated: word-only measures pass it in the units' place (see the file just written)")


# ------------------------------------------------------------------ the follow-up to a met E1 (PLAN Section 8)
CRITERION = 0.70                          # PLAN Section 8: with the token after each code masked too, the within-era AUC must stay at 0.70 or above


def e1_numbers(r):
    """What the follow-up sets side by side, from one scorer's entry in lib.e1's output or in a committed holmes_<tag>.json."""
    w = r["within_era"]
    return {"within_era_auc": w["auc"], "within_era_ci95": w["ci95"], "by_era": w["by_era"], "within_era_null_label_shuffle": w.get("null_label_shuffle"),
            "pooled_auc": r["negatives"]["auc"], "pooled_ci95": r["negatives"]["ci95"], "pooled_null_label_shuffle": r.get("null_label_shuffle"),
            "canon_units_median": r["canon_units_median"], "canon_units_min": r["canon_units_min"],
            "n_franchise": r["negatives"]["n_franchise"], "n_negatives": r["negatives"]["n_negatives"]}


def masking(S, codes_dir, plane_dir, meta):
    """How much more the second reading masks, from the two meta.json files of presence_pass.py (tokens that cannot carry a
    peak, text by text) and from the two planes (presences at the primary setting, over all units)."""
    was = json.load(open(codes_dir / "meta.json"))
    for k in ("ids", "n_tokens", "n_code_tokens", "texts_sha256", "scale_sha256"):
        assert was[k] == meta[k], f"the two readings of the name-free texts differ in {k}"
    tok, code, aft = (np.array(x) for x in (meta["n_tokens"], meta["n_code_tokens"], meta.get("n_after_tokens", [0] * S.N)))
    A, B = (np.load(d / "top1_30.bits.npy", mmap_mode="r") for d in (codes_dir, plane_dir))
    ones = lambda x: int(np.bitwise_count(x).sum())                    # noqa: E731
    pres = np.array([(ones(A[w]), ones(B[w]), ones(B[w] & ~A[w])) for w in range(S.N)])

    def group(ids):
        t, c, f = (int(x[ids].sum()) for x in (tok, code, aft))
        was_, now, added = (int(x) for x in pres[ids].sum(0))
        return {"texts": len(ids), "tokens": t, "code_tokens": c, "tokens_after_a_code": f,
                "share_of_tokens_masked": {"codes_only": c / t, "after_too": (c + f) / t},
                "median_tokens_masked_per_text": {"codes_only": float(np.median(code[ids])), "after_too": float(np.median((code + aft)[ids]))},
                "presences": {"codes_only": was_, "after_too": now, "share_lost": (was_ - now) / was_, "in_the_second_reading_only": added}}
    return {"what": "tokens that cannot carry a unit's peak in each reading of the name-free texts (codes_only: the tokens of each placeholder code; after_too: "
                    "those and the token directly after each code), and the presences (unit in text, strongest sentence at 30% of the reference-only scale, all "
                    "units) each reading finds. A presence found in the second reading only would mean the two readings differ where the mask changes nothing.",
            "screen_works": group(S.franchise), "detective_films": group(S.negatives), "all_texts_read": group(np.arange(S.N)),
            "per_text": {"id": meta["ids"], "tokens": tok.tolist(), "masked_codes_only": code.tolist(), "masked_after_too": (code + aft).tolist()}}


def followup(a):
    """PLAN Section 8's follow-up to a met E1 (logged in the plan's Section 16 when it is run). E1 on the name-free texts is
    computed as `run` computes it, by the same unit_tests: the canon from the ORIGINAL texts of the other screen works, the
    held-out work and the detective films scored on name-free presence, the pooled and the within-era AUC with the same
    bootstrap, label-shuffle null and seeds. One thing differs: the name-free presence is read from
    <acts-root>/<tag>/after/top1_30.bits.npy (gpu.py after: the token after each code masked too) in place of
    <tag>/anon/top1_30.bits.npy. The fixed criterion: the within-era AUC stays at 0.70 or above. E2, E3 and the other E1
    scorers are not part of the follow-up and are not reported.

    --plane-dir puts another folder in the after folder's place. With <tag>/anon, the committed numbers of `run` must come
    back to the last digit: that is the self-check of this command (it ends with an error if they do not), for any model
    whose bits are here. Its output goes to --out, never under results/."""
    t0, root, stop = time.time(), Path(a.acts_root) / a.tag, lambda why: sys.exit("holmes.py followup: REFUSED. " + why)      # noqa: E731
    done_path, check_path, rec_path = OUT / f"holmes_{a.tag}.json", OUT / f"after_check_{a.tag}.json", OUT / f"holmes_recognition_{a.tag}.json"
    plane_dir, self_check = Path(a.plane_dir or root / "after"), bool(a.plane_dir)
    if not (done_path.exists() and rec_path.exists()):
        stop(f"{done_path} or {rec_path} does not exist: the follow-up is read beside the committed E1 of holmes.py run {a.tag}.")
    done, check = json.load(open(done_path)), json.load(open(check_path)) if check_path.exists() else None
    if self_check:
        if not a.out or OUT.resolve() in Path(a.out).resolve().parents:
            stop("--plane-dir is for the self-check: give --out, somewhere that is not under results/.")
    elif done["confirmatory"]["E1_verdict"] != "met":
        stop(f"PLAN Section 8 fixes this follow-up for a met E1 only, and {done_path} says E1 is '{done['confirmatory']['E1_verdict']}' for {a.tag}.")
    elif not (check and check.get("passed") is True):
        stop(f"{check_path} does not exist or does not say \"passed\": true. gpu.py after {a.tag} writes it before it reads the Holmes texts.")
    if not (plane_dir / "meta.json").exists():
        stop(f"no finished reading under {plane_dir}" + ("." if self_check else f": gpu.py after {a.tag} fetches it."))
    S = lib.install(holmes_spec(a.neutral, a.extra_regressors))
    rec = json.load(open(rec_path))
    if any(str(int(w)) not in rec.get("h1", {}) for w in S.franchise):
        stop(f"{rec_path} lacks H1 for some screen works.")
    # the base counts and the original texts' plane, as the committed run took them (its file says how)
    stored, recounted = Path(done["stored_counts"]), done["base_counts"].startswith("recounted")
    src = lib.Bits({"orig": root / "orig", "anon": plane_dir}, [("orig", "top1_30"), ("anon", "top1_30")],
                   np.load(stored.parent / "scale_ref.npy") if (stored.parent / "scale_ref.npy").exists() else None)
    n_stored = json.load(open(stored.parent / "bg_meta.json"))["n_plots"] if (stored.parent / "bg_meta.json").exists() else 2999
    dropped = None if recounted else root / "bgdrop" / "counts_ref_scale.npy"
    base, n_plots = lib.base_counts(stored, dropped, C.THETA, n_stored, 0 if recounted else len(S.dropped_background))
    assert n_plots == len(C.background_docs()) == done["n_background_plots"], f"the base counts are over {n_plots} plots, the committed run's over {done['n_background_plots']}"
    meta = src.meta["anon"]
    if not self_check and not (meta.get("mask_codes") is True and meta.get("mask_after") is True):
        stop(f"{plane_dir} was not read with --mask-codes --mask-after.")
    print(f"== {a.tag}: {'SELF-CHECK, not the follow-up: ' if self_check else ''}E1 on name-free presence from {plane_dir} (codes masked: {meta.get('mask_codes')}; "
          f"the token after each code too: {meta.get('mask_after', False)})", flush=True)
    res, mats, _, _ = unit_tests(S, src, base[src.units], n_plots, {}, full=False, e1_only=True)
    e1, was = res["E1"], done["E1"]
    as_saved = lambda x: json.loads(json.dumps(x, default=lambda o: o.item()))      # noqa: E731
    # the original texts are the same texts in both runs, so their E1 (canon and score both from them) must come back to the
    # last digit: the same original plane, base counts, sets, seeds and code as the committed run
    if as_saved(e1["ceiling_original"]) != was["ceiling_original"]:
        stop(f"E1 on the original texts does not come back as {done_path} holds it, so the original plane, the base counts, the sets or the code differ from the committed run's.")
    new, old = e1_numbers(as_saved(e1["A_name_free"])), e1_numbers(was["A_name_free"])
    S1 = copy.copy(S)
    S1.adaptations = {}                 # recognition() then has no adaptation to split by H2: only its E1 half computes, by its own code
    rc, rc_was = as_saved(recognition(S1, rec, e1, mats["A_name_free"], None, {})), done["recognition"]
    by_naming = lambda r: {"E1_score_against_H1_log_probability": r["E1_score_against_H1_log_probability"], "E1": r["E1"],      # noqa: E731
                           "not_confined_to_works_the_model_names": r["not_confined_to_works_the_model_names"]["E1"]}
    auc = new["within_era_auc"]
    out = {"what": "The follow-up that v2/second/PLAN.md Section 8 fixes for a met E1: E1 on name-free texts, computed as holmes.py run computes it, with the "
                   "name-free presence read again with the token after each placeholder code masked too (v2's `after` version). Beside it, the committed E1 "
                   f"of {done_path}, where only the codes' own tokens are masked. Nothing else differs between the two.",
           "tag": a.tag, "self_check_with_another_plane": self_check,
           "criterion": {"rule": f"within-era AUC at least {CRITERION:.2f} (PLAN Section 8)", "within_era_auc": auc, "holds": bool(auc >= CRITERION)},
           "E1_name_free": {"token_after_each_code_masked_too": new, "codes_masked_only_as_committed": old, "change_in_within_era_auc": auc - old["within_era_auc"],
                            "change_in_pooled_auc": new["pooled_auc"] - old["pooled_auc"]},
           "E1": {"A_name_free": e1["A_name_free"]},
           "recognition": {"what": rc_was["not_confined_to_works_the_model_names"]["E1_rule"],
                           "token_after_each_code_masked_too": by_naming(rc), "codes_masked_only_as_committed": by_naming(rc_was)},
           "masking": masking(S, root / "anon", plane_dir, meta),
           "E1_on_the_original_texts_comes_back_as_committed": True,
           "pass_check_on_texts_that_are_not_new": None if self_check else dict(check, same_model_libraries_and_gpu_as_this_reading=check.get("provenance") == meta.get("provenance")),
           "presence_pass": {k: {x: m.get(x) for x in ("scale_sha256", "texts_sha256", "mask_codes", "mask_after", "provenance")}
                             for k, m in (("original", src.meta["orig"]), ("name_free_scored", meta), ("name_free_of_the_committed_run", json.load(open(root / "anon" / "meta.json"))))},
           "inputs": {"acts": str(root), "name_free_plane_scored": str(plane_dir / "top1_30.bits.npy"), "stored_counts": str(stored), "base_counts": done["base_counts"],
                      "n_background_plots": int(n_plots), "sets": {k: len(getattr(S, k)) for k in ("franchise", "negatives")},
                      "sha256": dict({Path(f).name: lib.sha256(f) for f in (DATA / "spec.json", DATA / "corpus.jsonl", DATA / "corpus_neutral.jsonl", done_path, rec_path)},
                                     **{str(f): lib.sha256(f) for f in (root / "orig" / "top1_30.bits.npy", root / "anon" / "top1_30.bits.npy", plane_dir / "top1_30.bits.npy",
                                                                        stored, *([] if recounted else [dropped]), *([] if self_check else [check_path]))})},
           "committed_run": {"file": str(done_path), "code": done["code"], "E1_verdict": done["confirmatory"]["E1_verdict"]}}
    out["seconds"], out["peak_memory_gb"] = round(time.time() - t0, 1), peak_gb()
    print(f"  within era {auc:.3f} ({new['within_era_ci95'][0]:.3f} to {new['within_era_ci95'][1]:.3f}), pooled {new['pooled_auc']:.3f}, canon {new['canon_units_median']} | "
          f"committed, codes masked only: within era {old['within_era_auc']:.3f}, pooled {old['pooled_auc']:.3f} | by era "
          + ", ".join(f"{e} {v['auc']:.3f} (was {old['by_era'][e]['auc']:.3f})" for e, v in new["by_era"].items()), flush=True)
    m = out["masking"]["all_texts_read"]
    print(f"  masked tokens: {m['code_tokens']:,} in codes, {m['tokens_after_a_code']:,} after a code, of {m['tokens']:,}; presences {m['presences']['codes_only']:,} -> "
          f"{m['presences']['after_too']:,} ({m['presences']['share_lost']:.1%} lost, {m['presences']['in_the_second_reading_only']} in the second reading only)", flush=True)
    print(f"  screen works the model does not name: within era {rc['E1']['not_named'].get('auc_within_era')} (n {rc['E1']['not_named']['n']}; committed "
          f"{rc_was['E1']['not_named'].get('auc_within_era')}) -> {rc['not_confined_to_works_the_model_names']['E1']}", flush=True)
    if not self_check:
        print(f"  criterion (within-era AUC at least {CRITERION:.2f}):", "HOLDS" if auc >= CRITERION else "DOES NOT HOLD", flush=True)
    else:          # the ordinary name-free plane in the after plane's place: every number of the committed run must come back
        rows = [(k, old[k], new[k]) for k in new] + [("held-out scores of the 81 screen works", was["A_name_free"]["held_out_score"], as_saved(e1["A_name_free"]["held_out_score"])),
                                                     ("E1 by naming (H1): score against log-probability, named, not named, rule", by_naming(rc_was), by_naming(rc))]
        out["self_check"] = {"plane": str(plane_dir), "rows": [{"what": k, "committed": x, "this_code": y, "same": x == y} for k, x, y in rows],
                             "passed": all(x == y for _, x, y in rows)}
        print(f"\n{a.tag}: committed ({done_path}) against this command with {plane_dir} as the plane; every row must be the same")
        for k, x, y in rows:
            short = lambda v: repr(v) if isinstance(v, (int, float)) else json.dumps(v)[:54] + (" ..." if len(json.dumps(v)) > 54 else "")      # noqa: E731
            print(f"  {k:44.44s} {short(x):>60} {short(y):>60}  {'same' if x == y else 'DIFFERS'}")
        print("SELF-CHECK", "PASSED" if out["self_check"]["passed"] else "FAILED")
    lib.save(out, a.out or OUT / f"holmes_e1_followup_{a.tag}.json")
    if self_check and not out["self_check"]["passed"]:
        sys.exit(1)
    return out


# ------------------------------------------------------------------ a dry run on made-up presence
FACTOR = {"top1_20": 1.6, "top1_30": 1.0, "top1_40": 0.6, "top1_50": 0.35, "top3_20": 1.3, "top3_30": 0.8, "top3_40": 0.45, "top3_50": 0.25,
          "cut_top1_30": 0.6, "icon_top1_30": 0.9, "over100": 0.002}
PLANTED = {"top1_20", "top1_30", "top3_20", "top3_30", "cut_top1_30", "icon_top1_30"}


def make_synthetic(S, root, n2u, signal, which, seed=1):
    """Made-up planes for the real Holmes works. Each unit has its own chance of being present (skewed, most units rare; 4%
    have no scale); original texts have a little more than name-free ones. Planted, if `signal`: every work with documented
    sources (any tier) takes 30% of each source's rarer units; the screen works share 300 rare canon units in their
    original texts, and half of them keep these in their name-free texts."""
    rng = np.random.default_rng(seed)
    rate = np.clip(rng.beta(0.25, 6.0, n2u), 0.0005, 0.9).astype(np.float32)
    rate[rng.random(n2u) < 0.04] = 0
    names = list(FACTOR) if which == "all" else ["top1_30", "cut_top1_30", "icon_top1_30", "over100"]
    mm = {}
    for k in ("orig", "anon"):
        (root / k).mkdir(parents=True)
        ns = [n for n in names if k == "anon" or n != "icon_top1_30"]
        mm.update({(k, n): np.lib.format.open_memmap(root / k / f"{n}.bits.npy", mode="w+", dtype=np.uint8, shape=(S.N, n2u // 8)) for n in ns})
        json.dump({"n_texts": S.N, "ids": list(range(S.N)), "n_layers": 1, "width": n2u // 2, "planes": ns, "synthetic": True}, open(root / k / "meta.json", "w"))
    canon = rng.choice(np.flatnonzero((rate > 0) & (rate < 0.01)), 300, replace=False)
    half, franchise, cands = set(rng.permutation(S.franchise)[: len(S.franchise) // 2].tolist()), set(S.franchise.tolist()), set(S.candidates.tolist())
    pairs, kept = {}, {}
    for ad in [S.adaptations] + list(S.subsets.values()):
        for w, v in ad.items():
            pairs.setdefault(w, set()).update(v)
    for w in range(S.N):
        u, po, pa = rng.random(n2u, dtype=np.float32), np.zeros(n2u, bool), np.zeros(n2u, bool)
        if signal:
            for s in pairs.get(w, ()):
                pa |= kept[s]
            if w in franchise:
                c = canon[rng.random(len(canon)) < 0.9]
                po[c] = True
                pa[c] = w in half
        po |= pa
        for (k, n), m in mm.items():
            bits = u < rate * (FACTOR[n] * (1.15 if k == "orig" else 1.0))
            m[w] = np.packbits(bits | (po if k == "orig" else pa) if n in PLANTED else bits)
        if w in cands:
            kept[w] = (u < rate) & (rate < 0.2) & (rng.random(n2u) < 0.3)
    for m in mm.values():
        m.flush()
    f = np.array([FACTOR[f"top1_{t}"] for t in THETAS], np.float32)[:, None]
    drop = rng.binomial(len(S.dropped_background), np.minimum(rate[None] * f, 1.0)).astype(np.int32)
    (root / "bgdrop").mkdir(), (root / "stored").mkdir()
    np.save(root / "bgdrop" / "counts_ref_scale.npy", drop)
    np.save(root / "stored" / "bg_counts_ref_scale.npy", drop + rng.binomial(2999 - len(S.dropped_background), np.minimum(rate[None] * f, 1.0)).astype(np.int32))
    np.save(root / "stored" / "scale_ref.npy", (rate > 0).astype(np.float32))
    # a made-up recognition file in the form probes.py writes: the half of the screen works that keep the canon are "named"; every other lineage work's source is "named"
    h1 = {str(w): {"named": bool(w in half), "logp": float(-rng.exponential(3) - (0 if w in half else 3))} for w in range(S.N)}
    h2 = {str(int(w)): {"source_named": bool(i % 2), "has_source": bool(w in cands or w in pairs)} for i, w in enumerate(S.lineage)}
    json.dump({"h1": h1, "h2": h2}, open(root / "recognition.json", "w"))


def synthetic(a):
    S = lib.install(holmes_spec(a.neutral, a.extra_regressors))
    root = CACHE / "synthetic"
    shutil.rmtree(root, ignore_errors=True)
    t0 = time.time()
    make_synthetic(S, root / "fake", a.units, a.signal, a.planes)
    made, size = time.time() - t0, sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 2 ** 30
    a.tag, a.acts_root, a.stored = "fake", str(root), str(root / "fake" / "stored" / "bg_counts_ref_scale.npy")
    a.out = a.out or CACHE / "holmes" / f"synthetic_{a.units}_{a.planes}_{'signal' if a.signal else 'no_signal'}.json"
    try:
        out = run(a, S, dry_run=root / "fake" / "recognition.json")
    finally:
        shutil.rmtree(root)                                             # the made-up planes are never kept
    e, r, e1 = out["E2"]["primary"], out["residual"], out["E1"]["A_name_free"]
    ps = [e["mean_over_sources"]["p"], e["mean_over_adaptations"]["p"], r["decision"]["mean_over_sources"]["p"], r["every_measure"]["mean_over_sources"]["p"]]
    print(f"synthetic, 2U = {a.units:,}, planes {a.planes}, signal {a.signal}: made in {made:.0f} s ({size:.2f} GB), run {out['seconds']} s, peak memory {out['peak_memory_gb']:.2f} GB")
    print(f"  E2 p (over sources, over adaptations) {ps[0]}, {ps[1]}; residual p (decision, every measure; over sources) {ps[2]}, {ps[3]}; "
          f"E1 within era {e1['within_era']['auc']:.3f} ({out['E1']['verdict_within_era']['verdict']}); E3 rho {out['E3']['primary']['rho']} p {out['E3']['primary']['p_one_sided']}")
    rc, pc = out["recognition"], out["E2"]["positive_control"]          # the paths added after the review, on the made-up recognition file
    rule = rc["not_confined_to_works_the_model_names"]
    print("  recognition rule: E1", rule["E1"], {k_: {x: v.get(x) for x in ("n", "auc_within_era", "ci95")} for k_, v in rc["E1"].items()})
    print("  recognition rule: residual test", rule["residual_test"],
          {k_: {"n": v["n"], "decision_p": v.get("residual_decision", {}).get("mean_over_sources", {}).get("p")} for k_, v in rc["E2"].items()})
    print("  leave one adaptation out, largest p: E2", out["E2"]["leave_one_adaptation_out"]["largest_p"]["mean_over_sources"], "| decision test",
          r["decision_leave_one_adaptation_out"]["largest_p"]["mean_over_sources"])
    print(f"  positive control {pc['in_top_two']} of {pc['n']}; by story", pc["by_story"])
    print("  confirmatory:", out["confirmatory"], "| the four conditions:", out.get("residual_test_reported_not_confirmatory"))
    if "E2_fallback" in out:                # review, change 6: reached with few units (for example --units 1600)
        fb = out["E2_fallback"]
        print("  E2 is not evaluable at the primary setting; fallback (descriptive only):", fb["setting"], fb["why_none"] or
              {k_: fb["results"][k_]["mean_over_sources"]["p"] for k_ in ("E2", "residual_decision")})
    elif a.signal:
        assert max(ps) < 0.01, "the planted heredity was not found"
        print("  the planted heredity was found")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["regress", "words", "run", "synthetic", "followup"])
    ap.add_argument("tag", nargs="?", default=None)
    ap.add_argument("--planes", default="all", choices=["all", "primary"])
    ap.add_argument("--draws", type=int, default=DRAWS, help="relabellings in E2's sampled null")
    ap.add_argument("--neutral", default=None, help="name-free texts (default data/second/holmes/corpus_neutral.jsonl; a labelled stand-in until it exists)")
    ap.add_argument("--acts-root", default="v2/acts_second/holmes", help="folder that holds <tag>/{orig,anon,bgdrop}")
    ap.add_argument("--stored", default=None, help="stored background counts (default v2/acts/<tag>/bg_counts_ref_scale.npy)")
    ap.add_argument("--recounted", action="store_true", help="--stored already counts the background without the documents left out")
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra-regressors", default=None, help="word measures the calibration added to the decision test, comma-separated "
                                                             "(default: decision_regressors_added in spec.json, if there)")
    ap.add_argument("--no-words", dest="words", action="store_false", help="regress: skip the model-free targets")
    ap.add_argument("--units", type=int, default=80000, help="synthetic: 2U")
    ap.add_argument("--no-signal", dest="signal", action="store_false", help="synthetic: plant nothing")
    ap.add_argument("--plane-dir", default=None, help="followup: another folder in the place of <acts-root>/<tag>/after (the self-check: <acts-root>/<tag>/anon)")
    args = ap.parse_args()
    assert args.tag or args.command in ("words", "synthetic"), "a model tag is needed"
    {"regress": regress, "words": words, "run": run, "synthetic": synthetic, "followup": followup}[args.command](args)
