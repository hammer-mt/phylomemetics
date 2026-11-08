"""Second corpora: the statistics of v2, as functions of explicit arguments (PLAN: v2/second/PLAN.md, Sections 6 to 8).

No corpus is held here. A corpus is a Spec (its texts and its sets of works, read from files); install(spec) rebinds the
corpus globals of v1's experiments.py (X), v2's experiments_v2.py (E) and common.py (C) so that their own functions can
be reused; nothing in those files is edited. Run from study/, one corpus per process.

Reused as they are (pure): E.CanonShare, E.Centroid, E.Fixed, E.verdict, E.cut_length, X.auc_matrix, X.boot_ci, X.tfidf,
X.spearman, X.resid. Reused after install: X.lexical_texts, X.lexical_reps, X.unit_rates (words), X.source_maps (the
enumeration), X.sequel_control, E.word_measures, E.word_rates, E.regressors, E.opening. Written new, because rebinding
cannot work or because the plan differs from v2: the packed-bit reader, the base counts, E1's within-era AUC and both
label-shuffle nulls, E2 (vectorised, two summary statistics, sampled or enumerated null), the residual fit with the
sets as arguments, the calibration list with its bands rescaled, E3 and its power simulation.

Added after the plan's review (each place names the finding it answers): a corpus whose specification says
curved_regressors has its residual test regress out the word measures, their squares and their pairwise products, and
the calibration puts curved functions of the regressors and a planted effect through it; leave one adaptation out; the
positive control by story; the openings AUC beside E1's verdict; the commit and state of the code in every output.
Added when those changes were verified: further_measures, word-only functions that are not polynomials of the regressors
(they pass the curved residual test in the units' place, so on Holmes that test is not calibrated).
"""
import itertools
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path[:0] = [".", "v2/analysis"]
import common as C               # noqa: E402
import experiments as X          # noqa: E402  (v1; imports on the Batman corpus, then install() rebinds it)
import experiments_v2 as E       # noqa: E402

_UNIT_RATES, _BACKGROUND, _SAMPLED, _CAL = X.unit_rates, None, {}, {}
A = lambda v: np.asarray(v, dtype=np.int64)          # noqa: E731
STATS = ("mean_over_adaptations", "mean_over_sources")           # E2's two summaries: v2's, and the plan's for Holmes
SPAN = 1e-20        # a measure whose residual variance is below this share of its variance is reproduced by the regression: nothing is left to test


# ------------------------------------------------------------------ the corpus specification
def bands_for(n):
    """PLAN Section 7. Optional band 2 to floor(n/2); word bands end at 3n/32, 6n/32, 10n/32 (rounded, halves up) and
    floor(n/2). n = 32 gives v2's 2-16 and 2-3, 4-6, 7-10, 11-16; n = 141 gives 2-70 and 2-13, 14-26, 27-44, 45-70."""
    ends = [int(math.floor(k * n / 32 + 0.5)) for k in (3, 6, 10)] + [n // 2]
    return (2, n // 2), [(lo + 1, hi) for lo, hi in zip([1] + ends[:-1], ends)]


class Spec:
    """One corpus. `d` holds ids (rows of the corpus file): franchise, negatives, lineage, candidates, adaptations
    {work: [sources]}, baseline, early, later, strata {early work: label}, icon_regex; and optionally era {work: label},
    other_negatives {name: ids}, covariates {name: {early work: value}}, sequels {sequel: prequel}, sequel_pool, strong,
    source_era, background_documents_left_out (beyond v2's 843), extra_decision_regressors, band, word_bands,
    curved_regressors (default false, which is v2's linear residual test; see regression_columns)."""

    def __init__(self, d, corpus, neutral, pageviews=None):
        self.d, self.paths = d, {"corpus": str(corpus), "neutral": str(neutral), "pageviews": str(pageviews)}
        self.works = [json.loads(l) for l in open(corpus)]
        self.neutral = [json.loads(l) for l in open(neutral)]
        self.N = len(self.works)
        assert [w["id"] for w in self.works] == list(range(self.N)) == [w["id"] for w in self.neutral]
        assert all(len(a["sentences"]) == len(b["sentences"]) for a, b in zip(self.works, self.neutral)), "sentence counts differ"
        self.label = [w["label"] for w in self.works]
        assert len(set(self.label)) == self.N, "labels must be unique"
        self.year = np.array([w["year"] or 0 for w in self.works])
        self.n_sent = np.array([len(w["sentences"]) for w in self.works])
        for k in ("franchise", "negatives", "lineage", "candidates", "baseline", "early", "later"):
            setattr(self, k, A(d[k]))
        self.adaptations = {int(a): [int(s) for s in v] for a, v in d["adaptations"].items()}
        self.other_negatives = {k: A(v) for k, v in d.get("other_negatives", {}).items()}
        self.band, self.word_bands = bands_for(len(self.lineage))
        self.band, self.word_bands = tuple(d.get("band", self.band)), [tuple(b) for b in d.get("word_bands", self.word_bands)]
        self.era = np.array([str(d["era"][str(i)]) for i in range(self.N)]) if "era" in d else np.full(self.N, "all")
        self.strata = np.array([str(d["strata"][str(int(i))]) for i in self.early])
        self.covariates = {k: np.array([float(v[str(int(i))]) for i in self.early]) for k, v in d.get("covariates", {}).items()}
        self.dropped_background = [int(x) for x in d.get("background_documents_left_out", [])]       # beyond v2's document 843
        self.extra_regressors = list(d.get("extra_decision_regressors", []))
        self.curved = bool(d.get("curved_regressors", False))
        self.pv = json.load(open(pageviews)) if pageviews and Path(pageviews).exists() else {}
        lin, cand = set(self.lineage.tolist()), self.candidates.tolist()
        assert set(cand) <= lin and set(self.baseline.tolist()) <= lin and set(self.early.tolist()) | set(self.later.tolist()) <= lin
        assert all(a in lin and set(s) <= set(cand) for a, s in self.adaptations.items())
        self.col = {c: j for j, c in enumerate(cand)}                # candidate id -> column of every rate matrix

    def popularity(self, ids):
        """v1's E3 popularity: log10 of mean monthly page views (experiments.e3)."""
        return np.log10(np.array([self.pv[str(int(i))]["views"] / max(self.pv[str(int(i))]["months"], 1) + 1 for i in ids]))


def install(S, acts="v2/acts", results="results/v2/second"):
    """Rebinds every corpus global the three modules read at call time, and checks each was set (with only X rebound,
    E.residual silently used Batman's rows). E.BAT is the lineage, as in E.residual and E.word_rates: E.e1_block, which
    reads it as the franchise, is therefore not used here (e1 below takes the franchise as an argument)."""
    d, lab, none = S.d, S.label, A([])
    labs = lambda ids: [lab[int(i)] for i in ids]                     # noqa: E731
    era_codes = {e: k for k, e in enumerate(sorted(set(S.era)))}
    x = dict(works=S.works, neutral=S.neutral, N=S.N, group=np.array([w.get("group") for w in S.works]),
             medium=np.array([w.get("medium") for w in S.works]), year=S.year, label=lab, L={l: i for i, l in enumerate(lab)},
             BAT=S.lineage, FILMS=S.other_negatives.get("ordinary_films", none), HERO_FILM_COUSINS=S.negatives,
             LIT_COUSINS=S.other_negatives.get("literary_cousins", none), ICON=re.compile(d["icon_regex"], re.I), n_sent=S.n_sent,
             ERA=A(d["source_era"]) if "source_era" in d else A([era_codes[e] for e in S.era]),
             ADAPTATIONS={lab[a]: labs(s) for a, s in S.adaptations.items()}, STRONG=labs(d.get("strong", [])), COMICS=S.candidates,
             SCREEN_NONADAPT=S.baseline, SEQUELS={lab[int(a)]: lab[int(b)] for a, b in d.get("sequels", {}).items()},
             SCREEN=A(d.get("sequel_pool", sorted(set(S.lineage.tolist()) - set(S.candidates.tolist())))))
    e = dict(BAT=S.lineage, FILMS=x["FILMS"], HERO=S.negatives, LIT=x["LIT_COUSINS"], N=S.N, POOL=np.concatenate([S.franchise, S.negatives]),
             NEG=dict({"hero_film_cousins": S.negatives}, **S.other_negatives), label=lab,
             CUTS=np.array([E.cut_length(w["sentences"]) for w in S.works]), BANDS=list(S.word_bands))
    c = dict(ACTS=Path(acts), RESULTS=Path(results))
    for mod, want in ((X, x), (E, e), (C, c)):
        for k, v in want.items():
            assert hasattr(mod, k), f"{mod.__name__}.{k} does not exist"       # a misspelt name would make a global nobody reads
            setattr(mod, k, v)
            assert getattr(mod, k) is v
    X._MAPS.clear(), E._WORDS.clear(), E._RATES.clear(), _CAL.clear(), _SAMPLED.clear()
    # two rebindings beyond the report's list. The default band of X.unit_rates is Batman's 2-16 and E.word_rates calls it
    # without a band; C.background_docs leaves out document 843 only, and the word measures need this corpus's background.
    X.unit_rates = lambda P, band=None: _UNIT_RATES(P, tuple(band or S.band))

    def background_docs():
        global _BACKGROUND
        if _BACKGROUND is None:                                       # the background plots are the first documents of the file
            _BACKGROUND = list(itertools.takewhile(lambda x_: x_["src"] == "bgplot", map(json.loads, open(C.REFERENCE))))
        return [x_ for x_ in _BACKGROUND if x_["i"] not in set(S.dropped_background) | {C.OVERLAP[0]}]
    C.background_docs = background_docs
    return S


# ------------------------------------------------------------------ presence stored as packed bits (v2/gpu/presence_pass.py)
class Bits:
    """The planes of one or more folders written by presence_pass.py, read one text at a time: a plane is
    <dir>/<plane>.bits.npy, uint8 [texts, 2U / 8], most significant bit first. [texts, 2U] is never held as bytes.
    self.units = the units present in at least one text of any plane named in `planes` (and with a positive scale, if a
    scale is given: the pass never marks a zero-scale unit present, so this only guards against a wrong file)."""

    def __init__(self, dirs, planes, scale=None):
        self.dirs = {k: Path(v) for k, v in dirs.items()}
        self.meta = {k: json.load(open(v / "meta.json")) for k, v in self.dirs.items()}
        self.n_texts = self.meta[next(iter(dirs))]["n_texts"]
        self.n2u = 2 * self.meta[next(iter(dirs))]["n_layers"] * self.meta[next(iter(dirs))]["width"]
        anyp = np.zeros(self.n2u // 8, np.uint8)
        for k, plane in planes:
            m = self.open(k, plane)
            assert m.shape == (self.n_texts, self.n2u // 8) and self.meta[k]["ids"] == list(range(self.n_texts)), (k, plane, m.shape)
            for r in range(self.n_texts):
                anyp |= m[r]
        ok = np.unpackbits(anyp).astype(bool)
        self.n_present_anywhere = int(ok.sum())
        if scale is not None:
            assert not (ok & ~(scale > 0)).any(), "a unit with no scale is marked present"
            ok &= scale > 0
        self.units = np.flatnonzero(ok)

    def open(self, k, plane):
        return np.load(self.dirs[k] / f"{plane}.bits.npy", mmap_mode="r")

    def has(self, k, plane):
        return (self.dirs[k] / f"{plane}.bits.npy").exists()

    def rows(self, k, plane, rows):
        """Yields (row, bool over self.units) one text at a time."""
        m = self.open(k, plane)
        for r in rows:
            yield int(r), np.unpackbits(m[int(r)])[self.units].astype(bool)

    def count(self, k, plane, rows):
        """Per unit of self.units: in how many of these texts it is present."""
        n = np.zeros(len(self.units), np.int32)
        for _, b in self.rows(k, plane, rows):
            n += b
        return n

    def presence(self, k, plane, rows, cols=None, compact=False):
        """bool [n_texts, len(cols)] with only `rows` filled (rows are work ids, as the scorers index them), or
        [len(rows), len(cols)] if compact. cols index self.units."""
        cols = np.arange(len(self.units)) if cols is None else cols
        out = np.zeros((len(rows) if compact else self.n_texts, len(cols)), bool)
        for i, (r, b) in enumerate(self.rows(k, plane, rows)):
            out[i if compact else r] = b[cols]
        return out


def base_counts(stored, dropped=None, theta=0.30, n_stored=2999, n_dropped=0):
    """PLAN Section 6: a unit's base count = v2's stored count on the reference-only scale minus its count over the
    background documents left out (background_pass.py's counts for those documents alone). Returns (counts [2U], plots)."""
    row = C.THRESHOLDS.index(theta)
    n = np.load(stored, mmap_mode="r")[row].astype(np.int64)
    if dropped is not None:
        d = np.load(dropped, mmap_mode="r")
        assert d.shape == np.load(stored, mmap_mode="r").shape, "the dropped documents' counts must have the stored counts' shape"
        n = n - d[row]
        assert n.min() >= 0 and d[row].max() <= n_dropped, "the dropped documents' counts do not fit the stored counts"
    return n, n_stored - n_dropped


def distinctive(count, n_plots, cap=C.CAP):
    """common.Setting.distinctive: present in at most `cap` of the background plots (89 of 2,999; 88 of 2,965)."""
    return np.ones(len(count), bool) if cap is None else count <= cap * n_plots + 1e-9


# ------------------------------------------------------------------ E1 (v2's e1_block with the sets as arguments; PLAN Section 8)
def within_era(M, era_f, era_n):
    """Each franchise work against the negatives of its own era only; the mean over the franchise works."""
    return float(np.mean([M[i, era_n == e].mean() for i, e in enumerate(era_f)]))


def boot_within_era(M, era_f, era_n, n, rng):
    """X.boot_ci within era: franchise works and negatives are resampled, with replacement, inside each era."""
    blocks = [(np.flatnonzero(era_f == e), np.flatnonzero(era_n == e)) for e in sorted(set(era_f))]
    vals = [sum(M[rng.choice(r, len(r))][:, rng.choice(c, len(c))].mean(1).sum() for r, c in blocks) / len(era_f) for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def shuffle_null(sc, franchise, negatives, n, rng, era=None):
    """E.shuffle_null and E.CanonShare.reduced with the sizes taken from the data. Works drawn at random from the pool
    play the franchise and the rest the negatives; the value is the AUC by the same held-out procedure. With `era`, labels
    are permuted within era and the value is the within-era AUC. The scorer is first cut to the units that could enter a
    canon at all (present in 80% of n - 1 works of the pool), which loses nothing."""
    pool, nf = np.concatenate([franchise, negatives]), len(franchise)
    if isinstance(sc, E.CanonShare):
        keep = np.flatnonzero(sc.Pc[pool].sum(0) >= math.ceil(0.8 * (nf - 1) - 1e-9))
        sc = E.CanonShare(sc.Pc[:, keep], sc.Ps[:, keep])
    vals = []
    for _ in range(n):
        if era is None:
            p = rng.permutation(pool)
            vals.append(float(sc.matrix(p[:nf], p[nf:])[0].mean()))
        else:
            parts = [(rng.permutation(pool[era[pool] == e]), int((era[franchise] == e).sum())) for e in sorted(set(era[pool]))]
            f, g = np.concatenate([p[:k] for p, k in parts]), np.concatenate([p[k:] for p, k in parts])
            vals.append(within_era(sc.matrix(f, g)[0], era[f], era[g]))
    return np.array(vals)


def _null(null, obs):
    return {"n": len(null), "mean": float(null.mean()), "p95": float(np.percentile(null, 95)), "p_value": float((np.sum(null >= obs) + 1) / (len(null) + 1))}


def e1(scorers, franchise, negsets, rng, n_null=1000, n_boot=2000, era=None, seed=7):
    """scorers: A_name_free, ceiling_original, A_iconography_dropped, openings, ... (E.CanonShare, E.Centroid or E.Fixed).
    negsets: {name: ids}, the first is the primary set. The pooled numbers follow E.e1_block's order of random draws, so
    a Batman spec gives v2's numbers; the within-era numbers (if `era`, labels over all works) use their own generator.
    Returns (results, {scorer: the AUC matrix against the primary negatives})."""
    out, mats, primary, rng_era = {}, {}, next(iter(negsets)), np.random.default_rng(seed)
    for key, sc in scorers.items():
        if sc is None:
            continue
        r = {}
        for k, ng in negsets.items():
            M, info = sc.matrix(franchise, ng)
            r[k] = {"auc": float(M.mean()), "ci95": X.boot_ci(M, n=n_boot, rng=rng), "n_franchise": len(franchise), "n_negatives": len(ng)}
            if k == primary:
                mats[key], r["held_out_score"] = M, info["held_out_score"]
                if "canon_size" in info:
                    r["canon_units_median"], r["canon_units_min"] = float(np.median(info["canon_size"])), int(min(info["canon_size"]))
        nulls = key in ("A_name_free", "openings") and n_null
        if nulls:
            r["null_label_shuffle"] = _null(shuffle_null(sc, franchise, negsets[primary], n_null, rng), r[primary]["auc"])
        if era is not None:
            ef, en = era[franchise], era[negsets[primary]]
            w = {"auc": within_era(mats[key], ef, en), "ci95": boot_within_era(mats[key], ef, en, n_boot, rng_era),
                 "by_era": {e: {"auc": float(mats[key][ef == e][:, en == e].mean()), "n_franchise": int((ef == e).sum()), "n_negatives": int((en == e).sum())}
                            for e in sorted(set(ef))}}
            if nulls:
                w["null_label_shuffle"] = _null(shuffle_null(sc, franchise, negsets[primary], n_null, rng_era, era), w["auc"])
            r["within_era"] = w
        out[key] = r

    def verdict(within):
        """E.verdict (the six clauses, v2's thresholds) on the pooled or on the within-era numbers."""
        view = {}
        for key, r in ((k, v) for k, v in out.items() if primary in v):
            s = r["within_era"] if within else dict(r[primary], **({"null_label_shuffle": r["null_label_shuffle"]} if "null_label_shuffle" in r else {}))
            view[key] = {"hero_film_cousins": {"auc": s["auc"], "ci95": s["ci95"]}}
            view[key].update({k: v for k, v in (("null_label_shuffle", s.get("null_label_shuffle")), ("canon_units_median", r.get("canon_units_median"))) if v is not None})
        # review, change 7: the plan's wording quotes the openings AUC and whether it is below 0.70 beside the verdict; the verdict is E.verdict's, unchanged
        o = view.get("openings", {}).get("hero_film_cousins", {}).get("auc")
        return dict(E.verdict(view), **{"openings_auc": o, "openings_auc_below_0.70": None if o is None else bool(o < 0.70)})

    out["verdict_pooled"] = verdict(False)
    if era is not None:
        out["verdict_within_era"] = verdict(True)
    return out, mats


# ------------------------------------------------------------------ E2
def shared(P, rows=None):
    """[len(rows), len(P)]: how many columns each of `rows` shares with every row of the boolean matrix P. Popcounts over
    packed bits: exact integers, and no large matrix product."""
    pk = np.packbits(P, axis=1)
    pk = np.ascontiguousarray(np.pad(pk, ((0, 0), (0, -pk.shape[1] % 8)))).view(np.uint64)
    return np.array([np.bitwise_count(pk[i] & pk).sum(1) for i in (range(len(P)) if rows is None else rows)], dtype=np.float64)


def rates(Pl, cand_rows, band):
    """X.unit_rates on a compact matrix. Pl: bool [lineage works, units]; cand_rows: each candidate's row of Pl.
    Returns (R [lineage works, candidates], number of optional units, optional units of each candidate). X.unit_rates
    loops candidates x works over every unit; this counts the same integers over the optional units only."""
    nb = np.zeros(Pl.shape[1], np.int32)
    for r in Pl:
        nb += r
    opt = np.flatnonzero((nb >= band[0]) & (nb <= band[1]))
    G = shared(Pl[:, opt], cand_rows)
    n_c = G[np.arange(len(cand_rows)), cand_rows]
    return (G / np.maximum(n_c, 1)[:, None]).T, int(len(opt)), n_c.astype(np.int64)


def full_rates(S, Pl, band=None):
    """rates() for a matrix whose rows are S.lineage, laid out as X.unit_rates lays it out: [all works, candidates]."""
    pos = {int(w): i for i, w in enumerate(S.lineage)}
    Rl, n_opt, n_c = rates(Pl, [pos[int(c)] for c in S.candidates], band or S.band)
    R = np.zeros((S.N, len(S.candidates)))
    R[S.lineage] = Rl
    const = int(sum(np.ptp(R[S.baseline, j]) == 0 for j in range(R.shape[1])))
    # E.evaluable: every candidate has at least 50 optional units and no candidate's rate is constant across the baseline works
    return R, {"n_optional_units": n_opt, "optional_units_per_candidate_min": int(n_c.min()), "optional_units_per_candidate_median": float(np.median(n_c)),
               "candidates_with_constant_rate": const, "band": list(band or S.band), "evaluable": bool(n_c.min() >= 50 and const == 0)}


def shares(z, elig, src):
    """experiments.e2's stat_for, for many source sets at once. z [candidates]: one work's hub-adjusted rates; elig: its
    time-eligible candidates; src [draws, k]: the candidates that play its sources. The candidate set is the eligible
    ones plus the sources; the share is that of the other candidates strictly above the best source (0 if none)."""
    best, zs = z[src].max(1), np.sort(z[elig])
    others = len(zs) - elig[src].sum(1)
    return np.where(others > 0, (len(zs) - np.searchsorted(zs, best, side="right")) / np.maximum(others, 1), 0.0)


def source_maps(true_cols, n_cands, n_draws, seed, enumerate_all=False):
    """[maps, k]: for each map, the candidate column that plays each of the k used sources. Sampled (PLAN Section 8):
    random one-to-one maps, the true labelling first. Enumerated: every map, by v1's own X.source_maps (its exact null)."""
    k = len(true_cols)
    if enumerate_all:
        assert n_cands == len(X.COMICS) and math.perm(n_cands, k) <= 2e7, "too many maps to list"
        return X.source_maps(k), "enumerated"
    key = (tuple(true_cols), n_cands, n_draws, seed)
    if key not in _SAMPLED:
        rng = np.random.default_rng(seed)
        M = np.empty((n_draws, k), np.int16)
        M[0] = true_cols
        for c in range(1, n_draws, 20000):
            M[c:c + 20000] = np.argsort(rng.random((min(20000, n_draws - c), n_cands)), axis=1)[:, :k]
        _SAMPLED.clear()                                              # one set of maps kept: the tests reuse the same sources
        _SAMPLED[key] = M
    return _SAMPLED[key], "sampled"


def e2(R, S, adaptations=None, n_draws=100000, seed=20261108, enumerate_all=False, use_hub=True, time_rule=True):
    """R [all works, candidates]: the rate (units, a word measure, or residuals). Rate to share exactly as experiments.e2:
    hub z-score against the baseline works (the work itself left out, population sd + 1e-9), time rule, share of the
    other eligible candidates above the best documented source. Two summaries: v2's mean over adaptations, and the mean
    over source stories of the mean share of the adaptations naming the story. Null: joint relabelling of the used
    sources (sampled or enumerated); adaptations of one source move together and every share is recomputed under the
    map. p = share of maps whose statistic is <= the observed one + 1e-12."""
    ad = adaptations or S.adaptations
    used = sorted({s for v in ad.values() for s in v})
    ucol, yc = {s: i for i, s in enumerate(used)}, S.year[S.candidates]
    M, kind = source_maps([S.col[s] for s in used], len(yc), n_draws, seed, enumerate_all)
    n_by = {s: sum(s in v for v in ad.values()) for s in used}
    tot, rows = np.zeros((2, len(M))), []
    for a, srcs in ad.items():
        base = S.baseline[S.baseline != a]
        z = np.array([(R[a, j] - R[base, j].mean()) / (R[base, j].std() + 1e-9) for j in range(len(yc))]) if use_hub else R[a].astype(float)
        elig = yc <= S.year[a] if time_rule else np.ones(len(yc), bool)
        sh = shares(z, elig, M[:, [ucol[s] for s in srcs]])
        tot[0] += sh / len(ad)
        tot[1] += sh * sum(1 / n_by[s] for s in srcs) / len(used)
        cols = [S.col[s] for s in srcs]
        best, others = z[cols].max(), elig.copy()
        others[cols] = False
        zo = z[others]
        rows.append({"work": int(a), "sources": srcs, "share": float(shares(z, elig, np.array([cols]))[0]), "rank": int(1 + (zo > best).sum()),
                     "n_candidates": int(others.sum() + len(cols)), "best_source": int(srcs[int(np.argmax(z[cols]))]),
                     "share_ties_half": float(((zo > best).sum() + 0.5 * (zo == best).sum()) / len(zo)) if len(zo) else 0.0})
    out = {"n_adaptations": len(ad), "n_sources": len(used), "null": kind, "n_maps": int(len(M)), "seed": None if enumerate_all else seed, "rows": rows}
    for i, (name, f) in enumerate((("mean_over_adaptations", lambda k: float(np.mean([r[k] for r in rows]))),
                                   ("mean_over_sources", lambda k: float(np.mean([np.mean([r[k] for r in rows if s in r["sources"]]) for s in used]))))):
        obs = f("share")
        out[name] = {"share": obs, "p": float(np.mean(tot[i] <= obs + 1e-12)), "null_mean": float(tot[i].mean()), "share_ties_half": f("share_ties_half")}
    out["by_source"] = {str(s): {"n_adaptations": n_by[s], "mean_share": float(np.mean([r["share"] for r in rows if s in r["sources"]]))} for s in used}
    return out


def brief(r):
    return {k: v for k, v in r.items() if k not in ("rows", "by_source")}


def without_source(ad, s):
    """The adaptations with one source story left out; an adaptation that named only that story drops out."""
    return {a: v for a, v in ((a, [x for x in v if x != s]) for a, v in ad.items()) if v}


def e2_report(R, S, subsets=None, jackknife=True, **kw):
    """E2 on the primary adaptations, on named subsets ({name: adaptations}), and leaving each source story out."""
    out = {"primary": e2(R, S, **kw)}
    out.update({name: e2(R, S, ad, **kw) for name, ad in (subsets or {}).items() if ad})
    if jackknife:
        out["leave_one_source_out"] = {str(s): brief(e2(R, S, without_source(S.adaptations, s), **kw)) for s in sorted({s for v in S.adaptations.values() for s in v})}
    return out


def leave_one_adaptation_out(tests, S):
    """Review, change 3 (one film that names three stories carries 2.5 of the 14 votes of the source-averaged statistic):
    each test of `tests` ({name: function of the adaptations, giving e2's output}) with each adaptation left out in turn,
    and for each summary statistic the largest p and the adaptation whose removal gives it."""
    rows = {name: {} for name in tests}
    for a in S.adaptations:                 # adaptation by adaptation, so that the tests share each set of sampled maps
        rest = {b: v for b, v in S.adaptations.items() if b != a}
        for name, test in tests.items():
            rows[name][str(a)] = brief(test(rest))
    out = {}
    for name, r in rows.items():
        worst = {s: max(r, key=lambda a: r[a][s]["p"]) for s in STATS}
        out[name] = {"largest_p": {s: {"p": r[a][s]["p"], "adaptation_left_out": int(a)} for s, a in worst.items()}, "by_adaptation_left_out": r}
    return out


def positive_control(R, S, adaptations, need, top=2):
    """PLAN Section 8, in place of v2's sequels: each (straight) adaptation's best-placed documented source among its
    eligible candidates by the raw rate, ranked as experiments.sequel_control ranks (falling rate, stable)."""
    rows = []
    for a, srcs in adaptations.items():
        cols = [S.col[s] for s in srcs]
        elig = np.flatnonzero((S.year[S.candidates] <= S.year[a]) | np.isin(np.arange(len(S.candidates)), cols))
        order = elig[np.argsort(-R[a, elig], kind="stable")].tolist()
        rows.append({"work": int(a), "sources": srcs, "rank": min(order.index(c) for c in cols) + 1, "n": len(order), "closest": int(S.candidates[order[0]])})
    n_top = sum(r["rank"] <= top for r in rows)
    by_story = {}                           # review, change 4: the control by story (an adaptation of two stories counts under both)
    for r in rows:
        for s in r["sources"]:
            b = by_story.setdefault(str(s), {"adaptations": 0, "in_top_two": 0})
            b["adaptations"], b["in_top_two"] = b["adaptations"] + 1, b["in_top_two"] + (r["rank"] <= top)
    return {"rows": rows, "in_top_two": n_top, "n": len(rows), "need": need, "passes": bool(n_top >= need), "by_story": by_story}


# ------------------------------------------------------------------ beyond words: the residual test
def fit_cells(S):
    """The cells of the residual fit: lineage work x candidate, a candidate's cell with itself left out."""
    mask = np.zeros((S.N, len(S.candidates)), bool)
    mask[S.lineage] = True
    mask[S.candidates, np.arange(len(S.candidates))] = False
    return mask


def regression_columns(lex, mask, curved):
    """The columns the rate is regressed on, beside a constant. Not curved: the word measures themselves (v2).
    Curved (review, change 1: with a linear regression, squares and products of the regressors passed the test in the
    units' place, because true pairs are outliers in the word measures): each measure, its square and every pairwise
    product, so k measures give k + k + k(k - 1)/2 columns (9 -> 54, 13 -> 104). The measures are first standardised over
    the fitted cells, which changes no fitted value (the columns span the same space) and keeps the least squares well
    conditioned. curved='squares' leaves the products out; the calibration uses it only to show that they are needed."""
    if not curved:
        return list(lex)
    Z = [(x - x[mask].mean()) / (x[mask].std() or 1.0) for x in lex]
    return Z + [z * z for z in Z] + ([a * b for a, b in itertools.combinations(Z, 2)] if curved is True else [])


def residual(R, lex, S, adaptations=None, curved=None, **kw):
    """v1's residual_test (E.residual) with the sets as arguments: least squares of R on a constant and the columns made
    from the word rates `lex` (regression_columns; curved as the corpus specification says, unless given) over the
    lineage x candidate cells, a candidate's cell with itself left out; then E2 on the residuals."""
    curved, mask = S.curved if curved is None else curved, fit_cells(S)
    cols = regression_columns(lex, mask, curved)
    D = np.column_stack([np.ones(mask.sum())] + [x[mask] for x in cols])
    beta = np.linalg.lstsq(D, R[mask], rcond=None)[0]
    Rr = R - (np.column_stack([np.ones(R.size)] + [x.ravel() for x in cols]) @ beta).reshape(R.shape)
    return dict(e2(Rr, S, adaptations, **kw), r2_of_fit=float(1 - np.var(Rr[mask]) / np.var(R[mask])), n_regressors=len(lex), n_cells=int(mask.sum()),
                curved=curved, n_columns=len(cols), residual_variance_share=float(np.var(Rr[mask]) / np.var(R[mask])))


def weighted_word_rate(e, S):
    """E.weighted_word_rate with the band as an argument: word presence with each optional word weighted by
    (1 + number of lineage works it is in) ** e."""
    P = E.word_measures()["Pw"]
    nb = P[S.lineage].sum(0)
    w = np.where((nb >= S.band[0]) & (nb <= S.band[1]), (1.0 + nb) ** e, 0.0)
    R = np.zeros((S.N, len(S.candidates)))
    for j, c in enumerate(S.candidates):
        wc = w * P[c]
        R[S.lineage, j] = (P[S.lineage] * wc).sum(1) / max(wc.sum(), 1e-12)
    return R


def calibration_measures(S):
    """E.words_block's 16 word-only measures that are put through the decision test in the units' place. Its two literal
    bands are rescaled with the word bands: '11 to 16 works' is the top word band and '4 to 16 works' runs from the start
    of the second word band to the top of the optional band."""
    if _CAL:
        return _CAL
    R, W, anon = E.word_rates(), E.word_measures(), X.lexical_texts(X.neutral)
    (lo2, _), (lo4, hi4) = S.word_bands[1], S.word_bands[3]
    cal = {f"word_presence_of_words_in_{lo4}_to_{hi4}_works": R[f"word_presence_in_{lo4}_to_{hi4}_works"]}
    cal.update({f"word_presence_weighted_by_frequency_to_the_power_{e}": weighted_word_rate(e, S) for e in (0.5, 1, 2, 3)})
    cal.update({k: R[k] for k in ("tfidf_background_weights", "word_presence_background_rare", "tfidf_words_and_pairs", "code_count_similarity", "code_presence")})
    cal[f"word_presence_of_words_in_{lo2}_to_{S.band[1]}_works"] = X.unit_rates(W["Pw"], (lo2, S.band[1]))
    for k, vec in (("tfidf_ignoring_words_in_over_half_the_texts", TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True, max_df=0.5)),
                   ("tfidf_raw_counts_words_and_pairs", TfidfVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))),
                   ("tfidf_keeping_stop_words", TfidfVectorizer(min_df=2, sublinear_tf=True)),
                   ("term_counts_without_idf", TfidfVectorizer(stop_words="english", min_df=2, use_idf=False)),
                   ("tfidf_character_4_grams", TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=2, sublinear_tf=True))):
        V = vec.fit_transform(anon).toarray()
        cal[k] = (V @ V.T)[:, S.candidates]
    _CAL.update(cal)
    return _CAL


def regressor_names(S, which, extra=None):
    """The word rates regressed out (E.regressors): 'registered' v1's two; 'decision' PLAN Section 7's seven; 'every_measure'
    all twelve. A measure that the calibration added (S.extra_regressors; PLAN Section 8) joins the last two forms."""
    WR = E.word_rates()
    names = list(WR)[:{"registered": 2, "decision": E.N_DECISION, "every_measure": len(WR)}[which]]
    return names + [k for k in (S.extra_regressors if extra is None else extra) if k not in names and which != "registered"]


def regressors(S, which, extra=None):
    """The rates named by regressor_names, each [all works, candidates]."""
    WR = E.word_rates()
    return [WR[k] if k in WR else calibration_measures(S)[k] for k in regressor_names(S, which, extra)]


def calibration(S, alpha=0.05, **kw):
    """E.words_block's check that words alone cannot pass the decision test, and PLAN Section 8's rule: while a word-only
    measure passes, the one with the lowest p is added to the regressors and the check is repeated. The rule is run twice:
    a measure passes on the plan's statistic (mean over sources), or on either summary statistic. `measures` is the check
    with the regressors the tests use (the specification's added ones included). The rule itself always starts from PLAN
    Section 7's seven, so the path that led to the added regressors stays in the output and can be compared with the
    specification's list. Every fit is the corpus's own (curved if its specification says so; review, change 1)."""
    cal = calibration_measures(S)
    first = next(iter(cal))                 # the top word band is itself a regressor: its residual is rounding error
    stats = STATS
    done = {}

    def test(extra):
        if tuple(extra) not in done:
            dec = regressors(S, "decision", extra)
            done[tuple(extra)] = {k: brief(residual(Rw, dec, S, **kw)) for k, Rw in cal.items() if k not in extra}
        return done[tuple(extra)]

    out = {"measures": test(S.extra_regressors), "regressors_added_before": list(S.extra_regressors), "itself_a_regressor": first,
           "curved_regressors": S.curved, "until_none_passes": {}}
    for s in stats:                         # v2 took the lowest over all 16; the regressor's own p is noise, so it is also given without it
        out[f"lowest_p_{s}"] = min(v[s]["p"] for k, v in out["measures"].items() if k != first)
        out[f"lowest_p_{s}_with_the_regressor_itself"] = min(v[s]["p"] for v in out["measures"].values())
    for rule, use in (("mean_over_sources", stats[1:]), ("either_statistic", stats)):
        extra, rounds, r = [], [], test([])
        while True:
            p = {k: min(v[s]["p"] for s in use) for k, v in r.items() if k != first}
            k = min(p, key=p.get)
            if p[k] >= alpha:
                break
            extra.append(k)
            rounds.append({"added": k, "its_p": p[k], "p": {s: r[k][s]["p"] for s in stats}})
            r = test(extra)
        out["until_none_passes"][rule] = {"added": rounds, "regressors_added": extra, "lowest_p_then": p[k],
                                          "equals_the_regressors_the_tests_use": extra == list(S.extra_regressors)}
    return out


def curved_measures(S):
    """Review, change 1: curved functions of the decision regressors, to go through the decision test in the units' place.
    Every square, pairwise product, cube, square times another regressor and product of three, of the regressors
    standardised over the fitted cells (a function of the raw measures differs from these by terms of lower degree only).
    {name: (kind, [all works, candidates])}; with nine regressors 9 + 36 + 9 + 72 + 84 = 210."""
    names, mask = regressor_names(S, "decision"), fit_cells(S)
    Z = {k: (x - x[mask].mean()) / (x[mask].std() or 1.0) for k, x in zip(names, regressors(S, "decision"))}
    out = {f"{a} squared": ("square", Z[a] ** 2) for a in names}
    out.update({f"{a} x {b}": ("pairwise_product", Z[a] * Z[b]) for a, b in itertools.combinations(names, 2)})
    out.update({f"{a} cubed": ("cube", Z[a] ** 3) for a in names})
    out.update({f"{a} squared x {b}": ("square_times_another", Z[a] ** 2 * Z[b]) for a, b in itertools.permutations(names, 2)})
    out.update({f"{a} x {b} x {c}": ("product_of_three", Z[a] * Z[b] * Z[c]) for a, b, c in itertools.combinations(names, 3)})
    return out


def further_measures(S):
    """Verification of review change 1: curved_measures tries only polynomials of the decision regressors, and none passes.
    These are other word-only functions of the same regressors, for the same test in the units' place. x is a regressor
    (a share or a cosine, never negative) and z the same standardised over the fitted cells. A row is one work's fitted
    cells and a column one candidate's: E2 compares the candidates within a work's row, so a word measure taken relative
    to its row is the nearest rival of a unit rate. {name: (kind, [all works, candidates])}; nine regressors give
    23 x 9 + 72 + 4 x 36 = 423, less any that is constant over the fitted cells."""
    names, mask = regressor_names(S, "decision"), fit_cells(S)
    X = dict(zip(names, regressors(S, "decision")))
    assert all(x[mask].min() >= 0 for x in X.values()), "the square roots and logarithms below need rates that are not negative"
    Z = {k: (x - x[mask].mean()) / (x[mask].std() or 1.0) for k, x in X.items()}
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)           # noqa: E731  (0 is the lowest)
    zscore = lambda v: (v - v.mean()) / (v.std() + 1e-12)                                             # noqa: E731

    def rows(x, f, m=mask):
        """f applied to the fitted cells of each row; every other cell stays 0 (no test reads it)."""
        out = np.zeros(x.shape)
        for i in np.flatnonzero(m.any(1)):
            out[i, m[i]] = f(x[i, m[i]])
        return out
    cols = lambda x, f: rows(x.T, f, mask.T).T                                                        # noqa: E731
    one = [("exp(z)", "exponential", lambda x, z: np.exp(z)), ("exp(z / 2)", "exponential", lambda x, z: np.exp(z / 2)),
           ("exp(2z)", "exponential", lambda x, z: np.exp(2 * z)), ("exp(-z)", "exponential", lambda x, z: np.exp(-z)),
           ("x to the power 1.5", "power of 1.5", lambda x, z: x ** 1.5), ("square root of x", "square root", lambda x, z: np.sqrt(x)),
           ("log(x + 0.01)", "logarithm", lambda x, z: np.log(x + 0.01)), ("z to the power 4", "fourth degree", lambda x, z: z ** 4),
           ("z to the power 5", "fifth power", lambda x, z: z ** 5), ("absolute value of z", "absolute value", lambda x, z: np.abs(z)),
           ("tanh(z)", "squashing", lambda x, z: np.tanh(z)), ("z - 2 where z is above 2, else 0", "threshold", lambda x, z: np.maximum(z - 2, 0)),
           ("1 where z is above 2", "threshold", lambda x, z: (z > 2) * 1.0), ("1 where z is above 3", "threshold", lambda x, z: (z > 3) * 1.0),
           ("rank over all fitted cells", "rank over all cells", lambda x, z: rows(x.reshape(1, -1), rank, mask.reshape(1, -1)).reshape(x.shape)),
           ("rank within its row", "relative to the row", lambda x, z: rows(x, rank)),
           ("1 where it is the largest of its row", "relative to the row", lambda x, z: rows(x, lambda v: (v == v.max()) * 1.0)),
           ("1 where it is among the three largest of its row", "relative to the row", lambda x, z: rows(x, lambda v: (rank(-v) < 3) * 1.0)),
           ("z-score within its row", "relative to the row", lambda x, z: rows(x, zscore)),
           ("share of its row's sum", "relative to the row", lambda x, z: rows(x, lambda v: v / (v.sum() + 1e-12))),
           ("ratio to the largest of its row", "relative to the row", lambda x, z: rows(x, lambda v: v / (v.max() + 1e-12))),
           ("rank within its column", "relative to the column", lambda x, z: cols(x, rank)),
           ("z-score within its column", "relative to the column", lambda x, z: cols(x, zscore))]
    out = {f"{k}: {what}": (kind, f(X[k], Z[k])) for k in names for what, kind, f in one}
    out.update({f"{a} / ({b} + 0.05)": ("ratio of two", X[a] / (X[b] + 0.05)) for a, b in itertools.permutations(names, 2)})
    for a, b in itertools.combinations(names, 2):
        out.update({f"the larger z of {a}, {b}": ("maximum of two", np.maximum(Z[a], Z[b])), f"the smaller z of {a}, {b}": ("minimum of two", np.minimum(Z[a], Z[b])),
                    f"{a} squared x {b} squared": ("fourth degree", Z[a] ** 2 * Z[b] ** 2), f"square root of {a} x {b}": ("square root", np.sqrt(X[a] * X[b]))})
    return {k: v for k, v in out.items() if np.ptp(v[1][mask]) > 0}


def curved_calibration(S, alpha=0.05, measures=None, **kw):
    """Review, change 1: no curved function of the decision regressors may pass the decision test in the units' place
    (p below alpha on either summary statistic). Each of curved_measures (or of `measures`, in the same form) goes
    through the test under three regressions:
    v2's linear one (the reviewer's finding), with squares only (his first suggestion; not enough), and with squares and
    pairwise products (what a curved specification uses; it decides). A measure the regression reproduces to rounding
    error (its squares and products are columns of it) has no residual to test: it is listed and not counted, because the
    p of rounding error means nothing."""
    dec, measures, out = regressors(S, "decision"), measures or curved_measures(S), {}

    def tally(rows):
        live = {k: v for k, v in rows.items() if not v["reproduced_by_the_regression"]}
        t = {"n": len(rows), "n_tested": len(live), "n_reproduced_by_the_regression_not_tested": len(rows) - len(live),
             "n_passing_on_either_statistic": sum(min(v["p"].values()) < alpha for v in live.values())}
        for s in STATS:
            low = min(live, key=lambda k: live[k]["p"][s], default=None)
            t[f"n_passing_{s}"] = sum(v["p"][s] < alpha for v in live.values())
            t[f"lowest_p_{s}"] = None if low is None else {"p": live[low]["p"][s], "measure": low}
        return t

    for form, curved in (("linear_regression", False), ("with_squares_only", "squares"), ("with_squares_and_pairwise_products", True)):
        rows = {}
        for name, (kind, Rw) in measures.items():
            r = residual(Rw, dec, S, curved=curved, **kw)
            rows[name] = {"kind": kind, "residual_variance_share": r["residual_variance_share"], "reproduced_by_the_regression": r["residual_variance_share"] < SPAN,
                          "p": {s: r[s]["p"] for s in STATS}}
        out[form] = dict(tally(rows), n_columns=r["n_columns"], this_is_the_regression_the_tests_use=bool(curved == S.curved),
                         passing={k: v["p"] for k, v in rows.items() if not v["reproduced_by_the_regression"] and min(v["p"].values()) < alpha},
                         by_kind={kind: tally({k: v for k, v in rows.items() if v["kind"] == kind}) for kind in dict.fromkeys(v["kind"] for v in rows.values())},
                         measures=rows)
    out["none_passes_with_the_regression_the_tests_use"] = all(not v["passing"] for v in out.values() if v["this_is_the_regression_the_tests_use"])
    return out


def planted_effect(S, n=40, boost=1.0, alpha=0.05, noise_seed=7, **kw):
    """Review, change 1, the other side: the curved regression must still let a real effect through. A made-up unit rate =
    one decision regressor (standardised; each in turn) + standard normal noise, and in the cells of every primary
    adaptation with its documented sources a boost of `boost` standard deviations of that sum, which no word measure
    knows about. Through the decision test, each draw once with the boost and once without it. (kw goes to E2: its
    relabellings and their seed; the noise has its own seed.)"""
    names, dec, mask = regressor_names(S, "decision"), regressors(S, "decision"), fit_cells(S)
    rng, draws = np.random.default_rng(noise_seed), []
    for i in range(n):
        x = dec[i % len(dec)]
        base = (x - x[mask].mean()) / (x[mask].std() or 1.0) + rng.standard_normal(x.shape)
        row = {"word_measure": names[i % len(dec)]}
        for side, b in (("with_the_boost", boost), ("without_it", 0.0)):
            R = base.copy()
            for a, srcs in S.adaptations.items():
                R[a, [S.col[c] for c in srcs]] += b * base[mask].std()
            r = residual(R, dec, S, **kw)
            row[side] = {s: r[s]["p"] for s in STATS}
        draws.append(row)
    passes = lambda side: {s: sum(d[side][s] < alpha for d in draws) for s in STATS}        # noqa: E731
    return {"n_draws": n, "boost_in_standard_deviations": boost, "noise_seed": noise_seed, "curved_regressors": S.curved, "passes_with_the_boost": passes("with_the_boost"),
            "passes_without_it": passes("without_it"), "draws": draws}


# ------------------------------------------------------------------ E3
def e3(Pe, outcome, S, n_later, n_perm=10000, seed=7, covariates=None):
    """experiments.e3 with the sets as arguments (PLAN Section 8). Pe: bool [early works, units], rows as S.early.
    outcome [units]: the share of the later works that have the unit (or, for the placebo, its background base rate).
    Units present in 2 to 80% of the early works; outcome residualised on the number of early works with the unit; per
    early work the mean over its units, residualised on year, sentences, centrality (mean Jaccard overlap with the other
    early works, here from pairwise counts, not from an [early, early, units] array) and `covariates`; Spearman with log
    mean monthly page views; permutation within strata (in the sorted order of their labels), one-sided."""
    early, covariates = S.early, covariates or {}
    k = Pe.sum(0)
    keep = (k >= 2) & (k <= 0.8 * len(early))
    Pe, k = Pe[:, keep], k[keep]
    r_unit = X.resid(outcome[keep], k[:, None].astype(float))
    off = np.array([r_unit[Pe[j]].mean() for j in range(len(early))])
    G = shared(Pe)
    n = np.diag(G).copy()
    J = G / np.maximum(n[:, None] + n[None, :] - G, 1)
    np.fill_diagonal(J, np.nan)
    cov = np.c_[(S.year[early], S.n_sent[early], np.nanmean(J, axis=1)) + tuple(covariates.values())].astype(float)
    off_r = X.resid(off, cov)
    p = S.popularity(early)
    rho, rng = X.spearman(off_r, p), np.random.default_rng(seed)
    strata = [np.flatnonzero(S.strata == s) for s in sorted(set(S.strata))]
    null = []
    for _ in range(n_perm):
        q = p.copy()
        for s in strata:
            q[s] = rng.permutation(q[s])
        null.append(X.spearman(off_r, q))
    null = np.array(null)
    jack = [X.spearman(np.delete(off_r, i), np.delete(p, i)) for i in range(len(early))]
    ok = bool(n.min() > 0 and np.isfinite(off).all() and np.isfinite(rho))        # E.e3_run: every early work has a unit in the band
    return {"n_early": len(early), "n_later": int(n_later), "n_units": int(keep.sum()), "evaluable": ok, "rho": rho if ok else None,
            "p_one_sided": float((np.sum(null >= rho) + 1) / (n_perm + 1)) if ok else None, "n_perm": n_perm, "null_sd": float(null.std()) if ok else None,
            "jackknife_min": float(np.min(jack)) if ok else None, "jackknife_positive": int(sum(j > 0 for j in jack)) if ok else None, "jackknife": jack if ok else None,
            "covariates": ["year", "sentences", "centrality"] + list(covariates), "strata": {s: int((S.strata == s).sum()) for s in sorted(set(S.strata))},
            "works": [{"work": int(w), "year": int(S.year[w]), "log10_monthly_views": float(p[i]), "offspring": float(off_r[i]), "n_units": int(n[i])} for i, w in enumerate(early)]}


def e3_power(strata_labels, rhos=(0.3, 0.5), n_sim=2000, n_perm=2000, alpha=0.05, seed=7):
    """experiments.e3_power with the strata as an argument: the early works, a population Spearman rho between offspring
    and popularity (bivariate normal), the one-sided permutation within strata, no covariates."""
    rng, lab = np.random.default_rng(seed), np.asarray(strata_labels)
    n, strata = len(lab), [np.flatnonzero(lab == s) for s in sorted(set(lab))]
    idx = np.tile(np.arange(n), (n_perm, 1))
    for s in strata:
        idx[:, s] = s[np.argsort(rng.random((n_perm, len(s))), axis=1)]
    out = {"n": n, "alpha": alpha, "n_sim": n_sim, "n_perm": n_perm, "strata": {str(s): int((lab == s).sum()) for s in sorted(set(lab))}}
    for rho in rhos:
        r = 2 * np.sin(np.pi * rho / 6)            # Pearson r of the normal pair whose Spearman correlation is rho
        x = rng.standard_normal((n_sim, n))
        y = r * x + np.sqrt(1 - r * r) * rng.standard_normal((n_sim, n))
        rx, ry = np.argsort(np.argsort(x, 1), 1), np.argsort(np.argsort(y, 1), 1)
        obs, p = (rx * ry).sum(1), np.empty(n_sim)
        for i in range(0, n_sim, 50):
            nul = (rx[i:i + 50, None, :] * ry[i:i + 50][:, idx]).sum(-1)
            p[i:i + 50] = ((nul >= obs[i:i + 50, None]).sum(1) + 1) / (n_perm + 1)
        out[f"power_rho_{rho}"] = float(np.mean(p < alpha))
    return out


# ------------------------------------------------------------------ output
def sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def provenance():
    """Review, change 5: the commit this code ran from, whether v2/second had uncommitted or untracked files, and the
    hashes of the two files that compute (a commit alone does not say what ran when the tree is not clean)."""
    here = Path(__file__).resolve().parent

    def git(*a):
        try:
            return subprocess.run(("git", "-C", str(here)) + a, capture_output=True, text=True, timeout=60).stdout.strip("\n")
        except (OSError, subprocess.SubprocessError):
            return ""
    changed = git("status", "--porcelain", "--", ".").splitlines()
    commit = git("rev-parse", "HEAD") or None
    return {"git_commit": commit, "v2_second_clean": bool(commit) and not changed, "v2_second_changed_or_untracked": changed,
            "sha256": {f: sha256(here / f) for f in ("lib.py", "holmes.py")}}


def save(obj, path):
    """Every number unrounded. numpy scalars and arrays become plain numbers and lists. Every output records the code
    that made it (provenance, read when this module was loaded)."""
    obj = dict(obj, code=PROVENANCE)

    def plain(o):
        if isinstance(o, (np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=plain, ensure_ascii=False))
    print("wrote", path, flush=True)


PROVENANCE = provenance()
