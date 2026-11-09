"""Remakes decision test: can a property of a candidate film that is neither its words nor its units make the source look special?

    cd study
    OMP_NUM_THREADS=1 ../.venv/bin/python results/v2/second/audit/confounds.py      -> results/v2/second/audit/confounds.json

EXPLORATORY: made after the registered result was seen. Nothing here is part of v2/second/PLAN.md and nothing here changes
a registered outcome. The list below was written before any analysis of it was run; each analysis is run once and every one
is reported, whatever it shows. No file of the plan, the paper, the results or v2/ is changed.

BASELINE (reproduced first, and the script stops if it is not reproduced). The registered test (v2/second/remake_test.py,
through v2/second/remakes.py) on the primary set (220 firm Indian cross-industry pairs, 175 families), frame "set" (the
candidates are the films of the Indian industries other than the remake's own, never later than the remake, never of its
family), the 39 registered word columns, penalty 3, one family left out at a time, every cell in the slope. The unit column
is the driver's: rate = shared / n_optional from v2/acts_second/remakes/<model>/counts/top1_30.npz, hub-corrected over the
220 entries (remake_test.hub_z), then normal scores within the pool. Qwen3-32B must give Z 3.4544, T 25.333, gain 19.609.
Llama 3.1 8B is put through every analysis beside it where that costs no further fit.

Every later analysis changes exactly one thing from the baseline: it adds columns to the word model (C), or it cuts the
pools down (D), or it puts a property in the units' place (P). In C and D the hub correction of every word column and of
the unit rate stays the baseline's (over all 220 entries), so that only the named thing changes.

THE PROPERTIES (of a candidate film c for the entry of remake r; none uses c's words or c's units)
 1 year      gap = year(r) - year(c), from the film table (never negative in a pool)
 2 industry  the candidate's industry label in the film table
 3 reference whether c's page is one of the first 12,000 documents (the film plots, kinds "bgplot" and "plot") of
             v2/cache/reference.jsonl, the corpus that set every unit's scale (the rule of build_holmes.reference_overlap)
 4 length    sentences kept (at most 40), word tokens (\\w+ in the joined text), and the model's own number of optional
             units of the film (n_optional of the counts file)
 5 popularity, leaving the entry's own family out: how many kept pairs of pairs.json (all 482, any set) have c as their
             source, and how many other films are in c's family. A film has one family, so for a candidate (never of the
             entry's family) these are its full counts and for the entry's own source both are 0 by that rule: the columns
             can only say "this film is known to belong to another family"

WHAT IS COMPUTED, FIXED IN ADVANCE

 (a) Is the source unlike the candidates? For every property column: the mean over entries of the source's value
     standardised within its pool (and a t over family means); the same mean weighted by each entry's open mass (one minus
     the held-out word probability of the source); the raw mean for sources and for the candidates (pool means averaged
     over entries).
 (b) Among NON-source candidates, does the unit column go with the property? For every property column: the Pearson
     correlation, within each pool and without the source, between the Qwen3-32B unit column (the normal scores the test
     uses) and the property; its mean over entries and a t over family means; and the same correlation among the 30
     non-source candidates to which the word model gives the highest held-out probability, averaged over entries with
     the open mass as weight (these are the candidates the statistic weighs).
 (P) The property in the units' place, word model unchanged, no hub correction (a film-level property has none), normal
     scores within the pool, ties sharing their rank; signs fixed here as the direction expected to favour a source:
       P1 recent                       minus the year gap
       P2 industry_prior               log share of the other families' sources that are of c's industry, among remakes
                                       of r's industry (C2b's first column), minus the log share of the pool that is of c's
                                       industry (C2b's second column)
       P3 reference_document           1 if c is a reference document
       P4 log_tokens                   log of c's word tokens
       P5 token_match                  minus | log tokens(c) - log tokens(r) |
       P6 log_n_optional               log of c's optional units (Qwen3-32B)
       P7 n_optional_match             minus | log n_optional(c) - log n_optional(r) | (Qwen3-32B)
       P8 not_in_another_family        minus log(1 + other films in c's family), family left out (0 for the source)
 (C) The property added to the word model. Each added column is standardised within the pool exactly as the driver
     standardises its "z" columns (remake_test.Stack.standardise: mean 0 and variance 1 within every pool, a column that is
     constant in a pool becomes 0 there); no hub form; the cube term stays the designers'. The word model is refitted
     leaving one family out, and the units go through slope, sign flips and gain as in the baseline.
       C1  year_gap, year_gap_log (log(1 + gap))                                                        2 columns
       C2a one indicator per industry that has a film in a primary pool (1 if c is of it)              7 columns expected
       C2b industry_log_source_share: log((n_ab + 0.5) / (n_a + 4)), n_ab = primary entries of OTHER families whose
           remake is of r's industry a and whose source is of c's industry b, n_a = their number, 8 = the Indian
           industries other than a; and industry_log_pool_share: log of the share of the entry's pool that is of c's
           industry (so that the model can turn a share of sources into a chance per candidate)         2 columns
       C3  reference_document (1 if c is one); reference_document_x_remake (the same times 1 if r is one) 2 columns
       C4a sentences; log_tokens; token_match; sentence_match (minus | sentences(c) - sentences(r) |)   4 columns
       C4b log_n_optional; n_optional_match (the model's own counts; Qwen3-32B only)                    2 columns
       C5  popularity_source_of (log(1 + kept pairs with c as source)); popularity_relatives (log(1 + other films in
           c's family)); both 0 for a film of the entry's own family                                    2 columns
       C6  all of C1 to C5 together (Qwen3-32B, with its n_optional)                                    21 columns expected
       C6L the same with Llama 3.1 8B's n_optional, for Llama 3.1 8B's units
 (D) Pools cut down to candidates like the source. An entry is dropped when its source is not in the cut pool or when
     fewer than 10 candidates are left beside the source; the word model (39 columns) is refitted on what is left.
       D1a candidates at most 5 years before the remake (gap <= 5)
       D1b candidates at most 10 years before the remake (gap <= 10)
       D1c candidates within 5 years of the SOURCE's year, either side (uses the answer; keeps every entry it can)
       D2  candidates of the source's industry (the scouts' frame of the plan, here with every cell in the slope)
       D3a candidates that are reference documents; entries whose source is one
       D3b candidates that are not reference documents; entries whose source is not one
       D3ab the family sums of D3a and D3b added (a test stratified by reference status; two word models)
       D3c candidates with the source's reference status, all entries, one word model
       D6  candidates of the source's industry, within 5 years of the source's year, and of its reference status
 Reported for every test: T, Z, the sign-flip p as the test computes it (100,000 flips, seed 20261108), the effective open
 pairs, the gain (Qwen3-32B everywhere; Llama 3.1 8B for the baseline, C6L and D6), the same without the family of largest
 sum, the sums over the entries the model does not recognise and over the entries whose source is and is not a reference
 document, and two noise calibrations: the share of the 2,000 registered noise draws (remakes_null_size.json) at or above
 Z, given where the word columns are unchanged and marked "indicative only" where the pools are cut; and the same share
 among 300 noise draws put through THIS analysis's own word model and pools (independent standard normal noise in every
 cell in the units' place, cells under 1e-08 left out as in the registered noise run), with the standard deviation of Z
 over those draws. The stand-in panel is not rebuilt (machine rule), so condition 5 is not re-evaluated for C and D.
 Also from the baseline's contributions, nothing refitted: the sums by remake-is-a-reference-document times
 source-is-a-reference-document.

SECOND ROUND (added after the first round's results were seen; written here before any of it was run; run once)
 The first round left one thing to explain: with the year gap as columns the units' Z stays near 3.3, but with the pools cut
 to films of the same years it falls to 2.0-2.7. Three things are added to say where that comes from:
   G  the gain of each of the eight probes P1 to P8 in the units' place, as the test computes it for the units, so that a
      property has the same three numbers as the units (Z, p, gain).
   S  sums of the baseline's stored contributions, nothing refitted: over the entries that each cut of D keeps (what those
      entries gave while their pools were whole), and by the source's year gap (0-2, 3-5, 6-10, 11 or more years).
   R  for each of D1a, D1b, D1c and D6: the same entries, each pool cut to a random sample of its baseline candidates of
      the size that analysis kept (the source always in); three draws (numpy default_rng(20261109 + draw), draw 1, 2, 3);
      the 39 columns refitted; everything reported as for D. If a pool of that size drawn at random loses as much Z as the
      pool of the same years, the loss is the smaller pool's and not the years'.

Memory: one process, one thread, one test alive at a time. AUDIT_DRY=1 runs every step on every sixth entry with random
numbers in the units' place (to test the code without seeing a result) and writes to AUDIT_OUT.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import gc
import json
import re
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "v2/second")
import remakes as R                # noqa: E402
import remake_test as T            # noqa: E402
import remake_null_size as NS      # noqa: E402

DRY = bool(os.environ.get("AUDIT_DRY"))
OUT = Path(os.environ.get("AUDIT_OUT", "results/v2/second/audit/confounds.json"))
PARTS = os.environ.get("AUDIT_PARTS")          # optional folder: finished blocks are kept there and read back after a crash
TAGS = ("qwen3-32b", "llama3.1-8b")
MAIN = TAGS[0]
MIN_CANDIDATES = 10
NOISE_DRAWS, NOISE_BLOCK, NOISE_SEED = (20, 10, 7_100_000) if DRY else (300, 150, 7_100_000)
TOP = 30
EXPLORATORY = "exploratory: made after the registered result was seen"
T0 = time.time()
peak_gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20      # noqa: E731


def log(*a):
    print(f"{time.time() - T0:7.0f}s {peak_gb():.2f}GB", *a, flush=True)


def part(key, fn):
    """A finished block, kept in PARTS (if set) so that a crash does not run an analysis twice."""
    if PARTS:
        f = Path(PARTS) / f"{key}.json"
        if f.exists():
            log("read back", key)
            return json.load(open(f))
    res = json.loads(json.dumps(fn(), default=R.plain))
    if PARTS:
        Path(PARTS).mkdir(parents=True, exist_ok=True)
        json.dump(res, open(Path(PARTS) / f"{key}.json", "w"))
    return res


# ------------------------------------------------------------------ inputs
W = json.load(open(R.OUT / R.WORDS_FILE))
REG = json.load(open(R.OUT / f"remakes_{MAIN}.json"))
C = R.Corpus()
for k in ("pairs.json", "films.jsonl"):
    assert W["sha256"][k] == C.sha[k] == REG["sha256"][k], k
ENTS = C.entries("primary")
if DRY:
    ENTS = ENTS[::6]
N = len(ENTS)
ROWS = np.array([C.rpos[p["r"]] for p in ENTS])
RR, SS = np.array([p["r"] for p in ENTS]), np.array([p["s"] for p in ENTS])
FAMILIES = [p["family"] for p in ENTS]
FAM = np.array(FAMILIES)
POOLS = [C.pool(p, "primary", "set") for p in ENTS]
ALL = np.arange(N)

_M, _meta = R.word_measures(C, R.CACHE, must_exist=True)
assert _meta["sha256"] == W["sha256"]["word_measures"]
COLS, _added = R.registered_columns(W, R.CACHE)
ME = {k: np.ascontiguousarray(v[ROWS]) for k, v in {**_M, **_added}.items() if k in {m for m, _ in COLS}}      # one row per entry
del _M, _added
gc.collect()

REF = set()
for _l in open("v2/cache/reference.jsonl"):
    _d = json.loads(_l)
    if _d["src"] not in ("bgplot", "plot"):
        break                                                    # the film plots are the first 12,000 documents
    REF.add(_d["id"])
ISREF = np.array([p in REF for p in C.page])
SRC_REF, REM_REF = ISREF[SS], ISREF[RR]


def unit_rate(tag):
    """The driver's rate (read_counts), one row per entry, and the film's number of optional units."""
    root = R.ACTS / tag
    ids = np.asarray(json.load(open(root / "meta.json"))["ids"], dtype=np.int64)
    Z = np.load(root / "counts" / f"{R.PRIMARY_PLANE}.npz")
    assert np.array_equal(ids[Z["rows"]], C.remakes) and Z["shared"].shape == (len(C.remakes), len(ids))
    n_opt = Z["n_optional"]
    rate = np.zeros((N, C.N))
    rate[:, ids] = Z["shared"][ROWS] / np.maximum(n_opt, 1)[None, :]
    nopt = np.ones(C.N)
    nopt[ids] = np.maximum(n_opt, 1)
    if DRY:
        rate = np.random.default_rng(len(tag)).random((N, C.N))
    return rate, nopt


HUB, NOPT, KNOWN = {}, {}, {}
for _tag in TAGS:
    _rate, NOPT[_tag] = unit_rate(_tag)
    HUB[_tag] = T.hub_z(_rate, FAMILIES, C.film_family)          # remake_test.unit_column's hub correction, over all the entries
    _rec = {(q["remake"], q["source"]): q for q in json.load(open(R.OUT / f"remakes_recognition_{_tag}.json"))["pairs"]}
    KNOWN[_tag] = np.array([bool(_rec[(p["remake"], p["source"])]["recognised"]) for p in ENTS])
del _rate
log(f"inputs: {N} entries, {len(set(FAMILIES))} families, {sum(map(len, POOLS))} cells; reference documents among the films read {int(ISREF.sum())} of {C.N}")


# ------------------------------------------------------------------ the test on some entries and part of their pools
class Fixed(T.RemakeTest):
    """remake_test.RemakeTest on some of the entries and part of their pools. The only difference: the hub correction of a
    word column is taken over all the entries of the set (as in the baseline), whichever entries are kept."""

    def __init__(self, cells, measures, film_family, keep, **kw):
        self._keep = keep
        super().__init__(cells, measures, film_family, **kw)

    def word_column(self, M, form, key=None):
        if form == "hz":
            return self.S.column(T.hub_z(ME[key], FAMILIES, C.film_family)[self._keep])
        return self.S.column(M)


def build(keep=ALL, pools=POOLS, extra_cols=(), extra=None):
    full = len(keep) == N
    cells = [{"pool": pool, "source": int(SS[i]), "family": FAMILIES[i]} for i, pool in zip(keep, pools)]
    meas = {k: (v if full else v[keep]) for k, v in ME.items()}
    meas.update({k: (v if full else v[keep]) for k, v in (extra or {}).items()})
    return Fixed(cells, meas, C.film_family, keep, columns=COLS + list(extra_cols))


def summary(test, Ur, keep=None):
    """T, Z and p as remake_test.RemakeTest.slope_test computes them (remakes.Analysis.summary)."""
    S = test.S
    k = np.ones(len(Ur), bool) if keep is None else np.asarray(keep, bool)
    d = S.family_sum(np.where(k, Ur, 0.0))
    ss = float(np.sqrt((d ** 2).sum()))
    return {"entries": int(k.sum()), "families": len(set(S.fam[k].tolist())), "T": float(d.sum()), "Z": float(d.sum() / ss) if ss > 0 else 0.0,
            "p": T.sign_flip_p(d), "families_positive": int((d > 0).sum()), "families_negative": int((d < 0).sum()),
            "effective_pairs_left_open": float(((1 - test.p_source) * k).sum())}, d


def own_noise(test):
    """Z of NOISE_DRAWS columns of independent standard normal noise in the units' place, through this test (cells the word
    model gives under 1e-08 left out of the slope, as in remake_null_size.py). The same seeds for every analysis."""
    S, zs, old = test.S, [], test._cache
    test._cache = R.Folds(test, R.PRUNE)
    try:
        for b in range(0, NOISE_DRAWS, NOISE_BLOCK):
            U = np.column_stack([S.normal_scores(np.random.default_rng(NOISE_SEED + b + j).standard_normal(S.n)) for j in range(NOISE_BLOCK)])
            D = np.column_stack([S.family_sum(c) for c in test.slope(U).T])
            zs.extend((D.sum(0) / np.sqrt((D ** 2).sum(0))).tolist())
            del U
    finally:
        test._cache = old
    return np.array(zs)


def result_block(test, keep, Ur, tag):
    S = test.S
    s, d = summary(test, Ur)
    order = np.argsort(-np.abs(d), kind="stable")
    ss = float((d ** 2).sum())
    brief = lambda x: {k: x[k] for k in ("entries", "families", "T", "Z", "p", "effective_pairs_left_open")}       # noqa: E731
    out = dict(s)
    out["effective_number_of_families"] = float(np.abs(d).sum() ** 2 / ss) if ss > 0 else 0.0
    out["share_of_the_sum_of_squares_in_the_five_largest_families"] = float((d[order[:5]] ** 2).sum() / ss) if ss > 0 else 0.0
    out["largest_family_sums"] = [{"family": S.fam_names[f], "D": float(d[f]), "remakes": [ENTS[i]["remake"] for i, x in zip(keep, S.fam == f) if x]} for f in order[:6]]
    out["without_the_family_of_largest_sum"] = brief(summary(test, Ur, S.fam != order[0])[0])
    out["entries_the_model_does_not_recognise"] = brief(summary(test, Ur, ~KNOWN[tag][keep])[0])
    out["source_is_a_reference_document"] = brief(summary(test, Ur, SRC_REF[keep])[0])
    out["source_is_not_a_reference_document"] = brief(summary(test, Ur, ~SRC_REF[keep])[0])
    out["contributions_by_entry"] = Ur.tolist()
    return out, {S.fam_names[f]: float(d[f]) for f in range(S.n_fam)}


def analyse(key, what, definition, keep=ALL, pools=POOLS, extra_cols=(), extra=None, tags=TAGS, gains=(MAIN,), dropped=None, registered_noise=None):
    def fn():
        t0 = time.time()
        test = build(keep, pools, extra_cols, extra)
        S = test.S
        sizes = S.sizes
        out = {"what": what, "definition": definition, "exploratory": EXPLORATORY, "entries": len(keep), "families": int(S.n_fam), "cells": int(S.n),
               "pool_sizes_with_the_source": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max())},
               "entries_dropped": dropped, "word_columns_with_the_cube": int(test.X.shape[1]), "added_columns": [f"{m}.{f}" for m, f in extra_cols],
               "weights_of_the_added_columns_fitted_on_all_entries": {f"{m}.{f}": float(test.beta_all[len(COLS) + j]) for j, (m, f) in enumerate(extra_cols)},
               "word_model": {"effective_pairs_left_open": float(test.unsolved), "sources_ranked_first": int((test.rank0 == 1).sum()),
                              "held_out_log_likelihood": float(test.ll0.sum()), "seconds_to_fit": round(time.time() - t0, 1)},
               "registered_noise_draws": registered_noise, "units": {}}
        log(f"{key}: fitted, {S.n} cells, {test.X.shape[1]} columns, open {test.unsolved:.2f}")
        U = np.column_stack([S.rank_column(HUB[tag][keep]) for tag in tags])
        Ur = test.slope(U)
        fams = {}
        for j, tag in enumerate(tags):
            blk, fams[tag] = result_block(test, keep, Ur[:, j], tag)
            if tag in gains:
                g = test.gain(U[:, j])
                blk["gain"] = R.gain_brief(g)
            if registered_noise is not None:
                blk["noise_p_registered_draws"] = NS.noise_p(blk["Z"])
            out["units"][tag] = blk
            log(f"{key}: {tag} T {blk['T']:.3f} Z {blk['Z']:.3f} p {blk['p']:.5f} gain {blk.get('gain', {}).get('gain')}")
        zs = own_noise(test)
        out["own_noise"] = {"draws": len(zs), "mean_Z": float(zs.mean()), "sd_Z": float(zs.std()), "share_of_draws_with_Z_above_the_registered_stand_ins_largest_Z_2.228": float((zs > 2.2279788170279393).mean()),
                            "Z_quantiles": {q: float(np.quantile(zs, float(q))) for q in ("0.9", "0.95", "0.975", "0.99")}, "largest_Z": float(zs.max()),
                            "what": "independent standard normal noise in the units' place through this analysis's own word model and pools; cells under 1e-08 left out"}
        for tag in tags:
            blk = out["units"][tag]
            blk["noise_p_own_draws"] = float((1 + (zs >= blk["Z"]).sum()) / (1 + len(zs)))
            blk["Z_in_own_noise_standard_deviations"] = float((blk["Z"] - zs.mean()) / zs.std())
        out["family_sums"] = fams
        out["seconds"], out["peak_memory_gb"] = round(time.time() - t0, 1), peak_gb()
        log(f"{key}: noise sd {zs.std():.3f}; done in {out['seconds']} s")
        del test, U
        gc.collect()
        return out
    return part(key, fn)


# ------------------------------------------------------------------ the properties, one row per entry [entries, films]
def tile(x):
    return np.tile(np.asarray(x, np.float32), (N, 1))


YEAR = C.year.astype(np.float64)
GAP = (YEAR[RR][:, None] - YEAR[None, :]).astype(np.float32)
INDUSTRIES = sorted(set(C.origin[np.concatenate(POOLS)].tolist()))
NSENT = np.array([len(s) for s in C.sentences], dtype=np.float64)
LTOK = np.log(np.array([len(re.findall(r"\w+", t)) for t in C.texts], dtype=np.float64))
RO, SO = C.origin[RR], C.origin[SS]
K_OTHER = len(R.INDIAN) - 1
LSRC, LPOOL = np.zeros((N, C.N), np.float32), np.zeros((N, C.N), np.float32)
for i in range(N):
    other = (FAM != FAM[i]) & (RO == RO[i])
    for b in sorted(R.INDIAN):
        if b != RO[i]:
            LSRC[i, C.origin == b] = np.log(((SO[other] == b).sum() + 0.5) / (other.sum() + 0.5 * K_OTHER))
    cnt = Counter(C.origin[POOLS[i]].tolist())
    LPOOL[i, POOLS[i]] = np.log(np.array([cnt[o] for o in C.origin[POOLS[i]]]) / len(POOLS[i]))
NSRC = np.zeros(C.N)
for p in C.pairs:
    NSRC[p["s"]] += 1
_size = Counter(C.fam[C.fam >= 0].tolist())
RELATIVES = np.array([_size[f] - 1 if f >= 0 else 0 for f in C.fam.tolist()], dtype=np.float64)
OTHER_FAMILY = (C.fam[None, :] != FAM[:, None])


def nopt_columns(tag):
    l = np.log(NOPT[tag])
    return {"log_n_optional": tile(l), "n_optional_match": (-np.abs(l[None, :] - l[RR][:, None])).astype(np.float32)}


PROPS = {
    "year_gap": (GAP, "year(r) - year(c), film table years"),
    "year_gap_log": (np.log1p(np.maximum(GAP, 0)), "log(1 + year gap)"),
    **{f"industry_{k}": (tile(C.origin == k), f"1 if the candidate's industry is {k}") for k in INDUSTRIES},
    "industry_log_source_share": (LSRC, "log((n_ab + 0.5) / (n_a + 4)): n_ab = primary entries of other families whose remake is of r's industry a and whose source is of c's industry b"),
    "industry_log_pool_share": (LPOOL, "log of the share of the entry's pool (source included) that is of c's industry"),
    "reference_document": (tile(ISREF), "1 if c's page is one of the 12,000 film plots of v2's reference corpus"),
    "reference_document_x_remake": ((ISREF[None, :] * REM_REF[:, None]).astype(np.float32), "reference_document times 1 if the remake is a reference document too"),
    "sentences": (tile(NSENT), "sentences kept of c's text (at most 40)"),
    "log_tokens": (tile(LTOK), "log of the number of \\w+ tokens of c's text"),
    "token_match": ((-np.abs(LTOK[None, :] - LTOK[RR][:, None])).astype(np.float32), "minus | log tokens(c) - log tokens(r) |"),
    "sentence_match": ((-np.abs(NSENT[None, :] - NSENT[RR][:, None])).astype(np.float32), "minus | sentences(c) - sentences(r) |"),
    **{k: (v, d) for (k, v), d in zip(nopt_columns(MAIN).items(), (f"log of c's optional units ({MAIN}, top1_30)", f"minus | log n_optional(c) - log n_optional(r) | ({MAIN})"))},
    "popularity_source_of": ((np.log1p(NSRC)[None, :] * OTHER_FAMILY).astype(np.float32), "log(1 + kept pairs of pairs.json with c as source); 0 for a film of the entry's own family"),
    "popularity_relatives": ((np.log1p(RELATIVES)[None, :] * OTHER_FAMILY).astype(np.float32), "log(1 + other films read in c's family); 0 for a film of the entry's own family"),
}
GROUPS = {"C1": ["year_gap", "year_gap_log"], "C2a": [f"industry_{k}" for k in INDUSTRIES], "C2b": ["industry_log_source_share", "industry_log_pool_share"],
          "C3": ["reference_document", "reference_document_x_remake"], "C4a": ["sentences", "log_tokens", "token_match", "sentence_match"],
          "C4b": ["log_n_optional", "n_optional_match"], "C5": ["popularity_source_of", "popularity_relatives"]}
GROUPS["C6"] = [k for g in ("C1", "C2a", "C2b", "C3", "C4a", "C4b", "C5") for k in GROUPS[g]]
PROBES = {"P1 recent": (lambda: -PROPS["year_gap"][0], "minus the year gap"),
          "P2 industry_prior": (lambda: PROPS["industry_log_source_share"][0] - PROPS["industry_log_pool_share"][0], "industry_log_source_share minus industry_log_pool_share"),
          "P3 reference_document": (lambda: PROPS["reference_document"][0], "1 if c is a reference document"),
          "P4 log_tokens": (lambda: PROPS["log_tokens"][0], "log of c's word tokens"),
          "P5 token_match": (lambda: PROPS["token_match"][0], "minus | log tokens(c) - log tokens(r) |"),
          "P6 log_n_optional": (lambda: PROPS["log_n_optional"][0], f"log of c's optional units ({MAIN})"),
          "P7 n_optional_match": (lambda: PROPS["n_optional_match"][0], f"minus | log n_optional(c) - log n_optional(r) | ({MAIN})"),
          "P8 not_in_another_family": (lambda: -PROPS["popularity_relatives"][0], "minus log(1 + other films in c's family), family left out (0 for the source)")}


def family_t(v):
    """Mean over entries, and a t from the family means (families are the clusters)."""
    ok = np.isfinite(v)
    fm = np.array([v[ok & (FAM == f)].mean() for f in sorted(set(FAM[ok].tolist()))])
    return float(v[ok].mean()), float(fm.mean() / (fm.std(ddof=1) / np.sqrt(len(fm)))) if len(fm) > 2 and fm.std(ddof=1) > 0 else None, int(ok.sum())


# ------------------------------------------------------------------ baseline, descriptions, probes
def baseline():
    t0 = time.time()
    test = build()
    S = test.S
    out = {"what": "the registered test reproduced: primary set, frame set, the 39 registered columns, every cell", "entries": N, "families": int(S.n_fam), "cells": int(S.n),
           "word_model": {"effective_pairs_left_open": float(test.unsolved), "sources_ranked_first": int((test.rank0 == 1).sum()), "held_out_log_likelihood": float(test.ll0.sum())},
           "units": {}, "checks": {}}
    if not DRY:
        an = R.Analysis(C, ME, "primary", "set", columns=COLS, by_entry=True)                # the driver's own construction
        out["checks"]["word_columns_equal_the_driver's_in_every_cell"] = bool(np.array_equal(an.test.X, test.X))
        out["checks"]["held_out_source_probabilities_equal_the_driver's"] = bool(np.array_equal(an.test.p_source, test.p_source))
        out["checks"]["unit_column_equals_the_driver's"] = bool(np.array_equal(an.test.unit_column(unit_rate(MAIN)[0], "rate_matrix"), S.rank_column(HUB[MAIN])))
        assert all(out["checks"].values()), out["checks"]
        del an
        gc.collect()
    U = np.column_stack([S.rank_column(HUB[tag]) for tag in TAGS])
    Ur = test.slope(U)
    for j, tag in enumerate(TAGS):
        blk, _ = result_block(test, ALL, Ur[:, j], tag)
        blk["gain"] = R.gain_brief(test.gain(U[:, j]))
        blk["noise_p_registered_draws"] = NS.noise_p(blk["Z"])
        out["units"][tag] = blk
        log(f"baseline: {tag} T {blk['T']:.4f} Z {blk['Z']:.4f} p {blk['p']:.5f} gain {blk['gain']['gain']:.4f}")
    if not DRY:
        reg, got = REG["decision"], out["units"][MAIN]
        out["checks"]["registered"] = {"Z": reg["slope"]["Z"], "T": reg["slope"]["T"], "gain": reg["gain"], "effective_pairs_left_open": reg["slope"]["effective_pairs_left_open"]}
        if not (abs(got["Z"] - reg["slope"]["Z"]) < 5e-5 and abs(got["T"] - reg["slope"]["T"]) < 5e-4 and abs(got["gain"]["gain"] - reg["gain"]) < 5e-4):
            raise SystemExit(f"the registered number is not reproduced: {got['Z']}, {got['T']}, {got['gain']['gain']} against {out['checks']['registered']}")
        out["checks"]["reproduced_to_four_digits"] = True
        tab = REG["analyses"]["primary | set"]["entries_table"]
        mine = [int(1 + (HUB[MAIN][i, POOLS[i]] > HUB[MAIN][i, SS[i]]).sum()) for i in range(N)]
        out["checks"]["unit_rank_of_the_source_equals_the_registered_table"] = bool(mine == [r["unit_rank_of_the_source"] for r in tab])
        out["checks"]["source_is_a_reference_document_equals_the_registered_table"] = bool(SRC_REF.tolist() == [r["source_is_a_document_of_v2_reference_corpus"] for r in tab])
    # the baseline's contributions by reference status of remake and source (nothing refitted)
    by = {}
    for a, la in ((True, "remake is a reference document"), (False, "remake is not")):
        for b, lb in ((True, "source is a reference document"), (False, "source is not")):
            by[f"{la}; {lb}"] = {tag: {k: v for k, v in summary(test, Ur[:, j], (REM_REF == a) & (SRC_REF == b))[0].items()} for j, tag in enumerate(TAGS)}
    out["contributions_by_reference_status_of_remake_and_source"] = by
    # own noise for the baseline (to compare with the registered 2,000 draws: sd of Z 1.239)
    zs = own_noise(test)
    out["own_noise"] = {"draws": len(zs), "mean_Z": float(zs.mean()), "sd_Z": float(zs.std()), "Z_quantiles": {q: float(np.quantile(zs, float(q))) for q in ("0.9", "0.95", "0.975", "0.99")},
                        "largest_Z": float(zs.max())}
    for tag in TAGS:
        out["units"][tag]["noise_p_own_draws"] = float((1 + (zs >= out["units"][tag]["Z"]).sum()) / (1 + len(zs)))
        out["units"][tag]["Z_in_own_noise_standard_deviations"] = float((out["units"][tag]["Z"] - zs.mean()) / zs.std())
    # held-out word probability of every cell
    pcell = np.empty(S.n)
    for f in range(S.n_fam):
        eta = test.X @ test.betas[f]
        e = np.exp(eta - np.repeat(np.maximum.reduceat(eta, S.starts), S.sizes))
        p = e / np.repeat(np.add.reduceat(e, S.starts), S.sizes)
        m = (S.fam == f)[S.group]
        pcell[m] = p[m]
    assert np.allclose(pcell[S.src], test.p_source)
    # (a) and (b)
    u, open_ = U[:, 0], 1 - test.p_source
    desc = {}
    for name, (Mx, definition) in PROPS.items():
        raw = np.asarray(Mx)[S.group, S.cols].astype(np.float64)
        z = S.standardise(raw)
        src_z = z[S.src]
        cand_mean, r_all, r_top = np.empty(N), np.full(N, np.nan), np.full(N, np.nan)
        for i, (st, sz) in enumerate(zip(S.starts, S.sizes)):
            keep = np.ones(sz, bool)
            keep[S.src[i] - st] = False
            x, y, pw = raw[st:st + sz][keep], u[st:st + sz][keep], pcell[st:st + sz][keep]
            cand_mean[i] = x.mean()
            if x.std() > 0:
                r_all[i] = np.corrcoef(x, y)[0, 1]
            top = np.argsort(-pw, kind="stable")[:TOP]
            if x[top].std() > 0 and y[top].std() > 0:
                r_top[i] = np.corrcoef(x[top], y[top])[0, 1]
        a_mean, a_t, _ = family_t(src_z)
        b_mean, b_t, b_n = family_t(r_all)
        okt = np.isfinite(r_top)
        desc[name] = {"definition": definition,
                      "a_source_mean_of_within_pool_standardised_value": a_mean, "a_t_over_family_means": a_t,
                      "a_the_same_mean_weighted_by_open_mass": float((src_z * open_).sum() / open_.sum()),
                      "a_raw_mean_for_sources": float(raw[S.src].mean()), "a_raw_mean_for_candidates_pool_means_averaged_over_entries": float(cand_mean.mean()),
                      "a_raw_mean_for_sources_weighted_by_open_mass": float((raw[S.src] * open_).sum() / open_.sum()),
                      "b_mean_within_pool_correlation_with_the_unit_column_non_sources": b_mean, "b_t_over_family_means": b_t, "b_entries_with_a_correlation": b_n,
                      f"b_the_same_among_the_{TOP}_non_sources_the_words_find_most_plausible_weighted_by_open_mass":
                          float((r_top[okt] * open_[okt]).sum() / open_[okt].sum()) if open_[okt].sum() > 0 else None,
                      "b_entries_with_that_correlation": int(okt.sum())}
    out["properties"] = desc
    # (P) the properties in the units' place
    names = list(PROBES)
    Pm = np.column_stack([S.normal_scores(np.asarray(PROBES[k][0]())[S.group, S.cols]) for k in names])
    Pr = test.slope(Pm)
    zreg = np.asarray(json.load(open(NS.OUT))["by_columns"]["39 columns (registered)"]["Z_sorted"])
    probes = {}
    for j, k in enumerate(names):
        s, _ = summary(test, Pr[:, j])
        probes[k] = {"definition": PROBES[k][1], "form": "normal scores within the pool, no hub correction, ties share their rank", **s,
                     "noise_p_registered_draws": NS.noise_p(s["Z"]), "noise_p_own_draws": float((1 + (zs >= s["Z"]).sum()) / (1 + len(zs))),
                     "share_of_registered_noise_draws_with_absolute_Z_at_or_above": float((1 + (np.abs(zreg) >= abs(s["Z"])).sum()) / (1 + len(zreg)))}
        log(f"probe {k}: T {s['T']:.3f} Z {s['Z']:.3f} p {s['p']:.5f}")
    out["properties_in_the_units_place"] = probes
    out["seconds"], out["peak_memory_gb"] = round(time.time() - t0, 1), peak_gb()
    del test, U, Pm
    gc.collect()
    return out


def cut(ok_fn):
    """Pools cut down: ok_fn(i, pool) -> which films of the pool stay. An entry is dropped when its source does not stay or
    when fewer than MIN_CANDIDATES films stay beside it."""
    keep, pools, dropped = [], [], {"source_not_in_the_cut_pool": 0, f"fewer_than_{MIN_CANDIDATES}_candidates_left": 0}
    for i, pool in enumerate(POOLS):
        ok = np.asarray(ok_fn(i, pool), bool)
        if not ok[pool == SS[i]][0]:
            dropped["source_not_in_the_cut_pool"] += 1
        elif ok.sum() - 1 < MIN_CANDIDATES:
            dropped[f"fewer_than_{MIN_CANDIDATES}_candidates_left"] += 1
        else:
            keep.append(i)
            pools.append(pool[ok])
    return dict(keep=np.array(keep), pools=pools, dropped=dropped)


IND = "indicative only: the word columns are the registered ones but the pools are cut, so the registered noise draws are not this analysis's null"
CUTS = {"D1a year: at most 5 years before the remake": ("D1a", lambda i, pool: GAP[i, pool] <= 5, "pool: candidates with year(r) - year(c) <= 5"),
        "D1b year: at most 10 years before the remake": ("D1b", lambda i, pool: GAP[i, pool] <= 10, "pool: candidates with year(r) - year(c) <= 10"),
        "D1c year: within 5 years of the source's year": ("D1c", lambda i, pool: np.abs(YEAR[pool] - YEAR[SS[i]]) <= 5, "pool: candidates with | year(c) - year(source) | <= 5 (uses the answer)"),
        "D2 industry: the source's industry": ("D2", lambda i, pool: C.origin[pool] == C.origin[SS[i]], "pool: candidates of the source's industry (the plan's scouts' frame; here every cell is in the slope)"),
        "D3a reference documents only": ("D3a", lambda i, pool: ISREF[pool], "pool: candidates that are reference documents; entries whose source is one"),
        "D3b non-reference documents only": ("D3b", lambda i, pool: ~ISREF[pool], "pool: candidates that are not reference documents; entries whose source is not one"),
        "D3c reference status of the source": ("D3c", lambda i, pool: ISREF[pool] == ISREF[SS[i]], "pool: candidates with the source's reference status; all entries, one word model"),
        "D6 industry, year and reference status of the source": ("D6", lambda i, pool: (C.origin[pool] == C.origin[SS[i]]) & (np.abs(YEAR[pool] - YEAR[SS[i]]) <= 5) & (ISREF[pool] == ISREF[SS[i]]),
                                                                 "pool: candidates of the source's industry, with | year(c) - year(source) | <= 5, and of the source's reference status")}
RANDOM_OF, RANDOM_DRAWS, RANDOM_SEED = ("D1a", "D1b", "D1c", "D6"), (1, 2, 3), 20261109


# ------------------------------------------------------------------ second round (added after the first round was seen)
def probe_gains():
    """G: the gain of each property probe in the units' place, as remake_test computes it for the units (baseline word model)."""
    test = build()
    S, out = test.S, {}
    for k in PROBES:
        g = test.gain(S.normal_scores(np.asarray(PROBES[k][0]())[S.group, S.cols]))
        out[k] = R.gain_brief(g)
        log(f"gain of probe {k}: {g['gain']:.3f}")
    del test
    gc.collect()
    return out


def sums_over(U, m, open_):
    """T, Z and p of stored contributions over the entries m (family sums), with their open mass. Nothing is refitted."""
    d = np.array([U[m & (FAM == f)].sum() for f in sorted(set(FAM[m].tolist()))])
    ss = float(np.sqrt((d ** 2).sum()))
    return {"entries": int(m.sum()), "families": len(d), "T": float(d.sum()), "Z": float(d.sum() / ss) if ss > 0 else 0.0, "p": T.sign_flip_p(d),
            "effective_pairs_left_open": float(open_[m].sum())}


def second_round(res):
    out = {"what": "SECOND ROUND: added after the results of the first round were seen, to say where its year results come from. G, S and R were written down "
                   "(in the script's docstring) before any of them was run; each was run once.", "exploratory": EXPLORATORY}
    out["G gains_of_the_properties_in_the_units_place"] = part("probe_gains", probe_gains)
    if not DRY:
        # S: the baseline's stored contributions summed over the entries each cut keeps, and by the age of the source
        open_ = 1 - np.array([r["words_probability_of_the_source"] for r in REG["analyses"]["primary | set"]["entries_table"]])
        gap = YEAR[RR] - YEAR[SS]
        S = {"what": "sums of the baseline's contributions (every pool whole), nothing refitted: over the entries that each cut of D keeps, and by the year gap of the source",
             "over_the_entries_each_cut_keeps": {}, "by_year_gap_of_the_source": {}}
        for label, (key, fn, _) in CUTS.items():
            keep = np.zeros(N, bool)
            keep[cut(fn)["keep"]] = True
            S["over_the_entries_each_cut_keeps"][label] = {tag: sums_over(np.array(res["baseline"]["units"][tag]["contributions_by_entry"]), keep, open_) for tag in TAGS}
        for lo, hi in ((0, 2), (3, 5), (6, 10), (11, 999)):
            m = (gap >= lo) & (gap <= hi)
            S["by_year_gap_of_the_source"][f"{lo} to {hi} years" if hi < 999 else f"{lo} years or more"] = {
                tag: sums_over(np.array(res["baseline"]["units"][tag]["contributions_by_entry"]), m, open_) for tag in TAGS}
        out["S baseline_contributions_over_subsets"] = S
    # R: random pools of the same sizes as the year cuts
    Rr = {}
    for label, (key, fn, _) in CUTS.items():
        if key not in RANDOM_OF:
            continue
        c = cut(fn)
        for draw in RANDOM_DRAWS:
            rng = np.random.default_rng(RANDOM_SEED + draw)
            pools = []
            for i, pool in zip(c["keep"], c["pools"]):
                cand = POOLS[i][POOLS[i] != SS[i]]
                pools.append(np.sort(np.append(rng.choice(cand, size=len(pool) - 1, replace=False), SS[i])))
            Rr[f"R {key} draw {draw}"] = analyse(f"R_{key}_{draw}", f"control for {key}: the same entries, every pool cut to a random sample of its candidates, as many as {key} keeps for the entry",
                                               {"restriction": f"entries of {key}; each pool: the source and a random sample (numpy default_rng({RANDOM_SEED} + {draw}), entries in order, without replacement) "
                                                               f"of the entry's baseline candidates, as many as {key}'s pool holds"},
                                               keep=c["keep"], pools=pools, dropped=c["dropped"], registered_noise=IND)
    out["R random_pools_of_the_same_size"] = Rr
    return out


def main():
    res = {"baseline": part("baseline", baseline)}
    A = {}
    cols_of = lambda g: [(k, "z") for k in GROUPS[g]]            # noqa: E731
    extra_of = lambda g: {k: PROPS[k][0] for k in GROUPS[g]}     # noqa: E731
    defs = lambda g: {k: PROPS[k][1] for k in GROUPS[g]}         # noqa: E731
    std = "each added column standardised within the pool (mean 0, variance 1; constant in a pool: 0), no hub form; word model refitted leaving one family out"
    A["C1 year gap"] = analyse("C1", "the year gap and its logarithm added to the word model", {"columns": defs("C1"), "how": std}, extra_cols=cols_of("C1"), extra=extra_of("C1"))
    A["C2a industry indicators"] = analyse("C2a", "one indicator per candidate industry added to the word model", {"columns": defs("C2a"), "how": std}, extra_cols=cols_of("C2a"), extra=extra_of("C2a"))
    A["C2b industry pair frequencies"] = analyse("C2b", "the remake-industry by candidate-industry frequency of sources, learned leaving the family out, and the pool's make-up, added to the word model",
                                                 {"columns": defs("C2b"), "how": std}, extra_cols=cols_of("C2b"), extra=extra_of("C2b"))
    A["C3 reference document"] = analyse("C3", "whether the candidate is a document of v2's reference corpus added to the word model", {"columns": defs("C3"), "how": std},
                                         extra_cols=cols_of("C3"), extra=extra_of("C3"))
    A["C4a text length"] = analyse("C4a", "sentences, tokens and their match with the remake's added to the word model", {"columns": defs("C4a"), "how": std}, extra_cols=cols_of("C4a"), extra=extra_of("C4a"))
    A["C4b optional units"] = analyse("C4b", "the candidate's number of optional units and its match with the remake's added to the word model (the model's own counts)",
                                      {"columns": defs("C4b"), "how": std}, extra_cols=cols_of("C4b"), extra=extra_of("C4b"), tags=(MAIN,))
    A["C5 popularity"] = analyse("C5", "how often the candidate is a source in the list and how many relatives it has, the entry's family left out, added to the word model",
                                 {"columns": defs("C5"), "how": std, "note": "both columns are 0 for the entry's own source by the leave-family-out rule: they can only mark films known to belong to another family"},
                                 extra_cols=cols_of("C5"), extra=extra_of("C5"))
    A["C6 all properties"] = analyse("C6", "all the columns of C1 to C5 added to the word model together", {"columns": defs("C6"), "how": std}, extra_cols=cols_of("C6"), extra=extra_of("C6"), tags=(MAIN,))
    ex = {**extra_of("C6"), **nopt_columns(TAGS[1])}
    A["C6L all properties (llama3.1-8b's own optional units)"] = analyse("C6L", "C6 with the n_optional columns of Llama 3.1 8B, for its units", {"columns": defs("C6"), "how": std}, extra_cols=cols_of("C6"), extra=ex,
                                                                           tags=(TAGS[1],), gains=(TAGS[1],))
    del ex
    ind = IND
    for label, (key, fn, definition) in CUTS.items():
        c = cut(fn)
        A[label] = analyse(key, "the pools cut down to candidates like the source; the 39 registered columns refitted", {"restriction": definition, "entries_dropped_when": f"the source is not in the cut pool, or fewer than {MIN_CANDIDATES} candidates are left beside it"},
                           keep=c["keep"], pools=c["pools"], dropped=c["dropped"], gains=TAGS if key == "D6" else (MAIN,), registered_noise=ind)
    # D3a and D3b together: a test stratified by reference status
    strat = {}
    for tag in TAGS:
        fa, fb = A["D3a reference documents only"]["family_sums"][tag], A["D3b non-reference documents only"]["family_sums"][tag]
        d = np.array([fa.get(f, 0.0) + fb.get(f, 0.0) for f in sorted(set(fa) | set(fb), key=str)])
        ss = float(np.sqrt((d ** 2).sum()))
        strat[tag] = {"families": len(d), "T": float(d.sum()), "Z": float(d.sum() / ss) if ss > 0 else 0.0, "p": T.sign_flip_p(d), "families_positive": int((d > 0).sum()), "families_negative": int((d < 0).sum()),
                      "noise_p_registered_draws": NS.noise_p(float(d.sum() / ss) if ss > 0 else 0.0)}
    A["D3ab stratified by reference status"] = {"what": "the family sums of D3a and D3b added: every pool holds only films of the source's reference status; two word models",
                                                "exploratory": EXPLORATORY, "entries": A["D3a reference documents only"]["entries"] + A["D3b non-reference documents only"]["entries"],
                                                "effective_pairs_left_open": A["D3a reference documents only"]["word_model"]["effective_pairs_left_open"] + A["D3b non-reference documents only"]["word_model"]["effective_pairs_left_open"],
                                                "registered_noise_draws": ind, "units": strat}
    res["analyses"] = A
    # the table
    b = res["baseline"]
    rows = [{"analysis": "baseline (registered test reproduced)", "entries": b["entries"], "families": b["families"], "columns": 39, "effective_pairs_left_open": b["word_model"]["effective_pairs_left_open"],
             **{f"{tag}: {k}": b["units"][tag].get(k) for tag in TAGS for k in ("T", "Z", "p", "noise_p_registered_draws", "noise_p_own_draws")},
             **{f"{tag}: gain": b["units"][tag]["gain"]["gain"] for tag in TAGS}, "own_noise_sd_of_Z": b["own_noise"]["sd_Z"]}]
    for k, v in b["properties_in_the_units_place"].items():
        rows.append({"analysis": f"{k} in the units' place", "entries": v["entries"], "families": v["families"], "columns": 39, "effective_pairs_left_open": v["effective_pairs_left_open"],
                     "property: T": v["T"], "property: Z": v["Z"], "property: p": v["p"], "property: noise_p_registered_draws": v["noise_p_registered_draws"]})
    for label, v in A.items():
        row = {"analysis": label, "entries": v["entries"], "families": v.get("families"), "columns": v.get("word_columns_with_the_cube", 39),
               "effective_pairs_left_open": v["word_model"]["effective_pairs_left_open"] if "word_model" in v else v["effective_pairs_left_open"],
               "own_noise_sd_of_Z": v.get("own_noise", {}).get("sd_Z")}
        for tag in TAGS:
            if tag in v["units"]:
                u = v["units"][tag]
                row.update({f"{tag}: {k}": u.get(k) for k in ("T", "Z", "p", "noise_p_registered_draws", "noise_p_own_draws")})
                row[f"{tag}: gain"] = u.get("gain", {}).get("gain") if isinstance(u.get("gain"), dict) else None
        rows.append(row)
    # the second round
    res["second_round"] = second_round(res)
    for label, v in res["second_round"]["R random_pools_of_the_same_size"].items():
        u = v["units"]
        rows.append({"analysis": label + " (second round)", "entries": v["entries"], "families": v["families"], "columns": 39, "effective_pairs_left_open": v["word_model"]["effective_pairs_left_open"],
                     "own_noise_sd_of_Z": v["own_noise"]["sd_Z"], **{f"{tag}: {k}": u[tag].get(k) for tag in TAGS for k in ("T", "Z", "p", "noise_p_registered_draws", "noise_p_own_draws")},
                     f"{MAIN}: gain": u[MAIN]["gain"]["gain"]})
    res["table"] = rows
    out = {"what": "Confounds audit of the remakes decision test: properties of a candidate film that are neither its words nor its units (year, industry, membership of v2's reference "
                   "corpus, length, popularity). Made by results/v2/second/audit/confounds.py. Every number unrounded. " + EXPLORATORY + ".",
           "exploratory": EXPLORATORY,
           "how_to_read": {"T": "the test's statistic: the sum over families of the entries' contributions", "Z": "T divided by the root of the sum of the squared family sums",
                           "p": "one-sided sign-flip p of the family sums, as remake_test.sign_flip_p computes it (100,000 flips, seed 20261108); known to be too small on this list",
                           "noise_p_registered_draws": "share of the 2,000 registered pure-noise draws (remakes_null_size.json, 39 columns) with Z at or above this Z, the observed one counted as a draw",
                           "noise_p_own_draws": f"the same share among {NOISE_DRAWS} pure-noise draws put through this analysis's own word model and pools",
                           "gain": "held-out log-likelihood of the true sources with the unit column added to the word model, minus without it",
                           "effective_pairs_left_open": "sum over entries of one minus the held-out word probability of the true source",
                           "condition_5": "not re-evaluated: the stand-in panel was not rebuilt (machine rule); the registered largest stand-in Z, 2.228, belongs to the baseline's word model and pools only"},
           "notes": ["The list of analyses in the script's docstring was written before any of them was run. Before the real run the code was tried once on every sixth entry (37 entries) "
                     "with random numbers in the units' place (AUDIT_DRY=1); that trial showed no unit result, but it did show the descriptions (a) and the property probes (P) "
                     "on that sixth, which use no unit. Nothing was changed after it except a guard against a division by zero.",
                     "The stand-in panel was not rebuilt, the every-film frame and the secondary sets were not analysed (machine rules).",
                     "A second round (G, S, R: see second_round) was added after the first round's results were seen; the first round's blocks were not run again.",
                     "Data: the film table gives 'Vettai (film)', the source of Tadakha (2013), the year 1986. The same plot stands in the table a second time under the page 'Vettai' "
                     "with the year 2012, and the corpus kept the row dated 1986. So that pair's year gap here is 27 years where the film's is 1: D1a and D1b drop the entry, "
                     "D1c and D6 give it a pool of films of 1981 to 1991, and in the registered frame the film is a candidate for remakes of 1986 to 2011 that it could not have been the source of. "
                     "Its contribution to the baseline T (Qwen3-32B): " + ", ".join(f"{u:.3f}" for p, u in zip(ENTS, res["baseline"]["units"][MAIN]["contributions_by_entry"]) if p["source"] == "Vettai (film)") + "."],
           "git_head": R.git_head(), "dry_run_on_random_numbers": DRY,
           "sha256": {**C.sha, "remakes.py": R.sha256("v2/second/remakes.py"), "remake_test.py": R.sha256("v2/second/remake_test.py"), "confounds.py": R.sha256(__file__),
                      R.WORDS_FILE: R.sha256(R.OUT / R.WORDS_FILE), f"remakes_{MAIN}.json": R.sha256(R.OUT / f"remakes_{MAIN}.json"),
                      **{f"{tag}/counts/{R.PRIMARY_PLANE}.npz": R.sha256(R.ACTS / tag / "counts" / f"{R.PRIMARY_PLANE}.npz") for tag in TAGS}},
           "settings": {"models": list(TAGS), "plane": R.PRIMARY_PLANE, "set": "primary", "frame": "set", "penalty": T.LAMBDA, "sign_flips": T.N_FLIPS, "seed": T.SEED,
                        "minimum_candidates_beside_the_source_in_a_cut_pool": MIN_CANDIDATES, "noise_draws_per_analysis": NOISE_DRAWS, "first_noise_seed": NOISE_SEED,
                        "top_candidates_by_word_probability_in_b": TOP, "threads": 1, "industries_with_a_film_in_a_primary_pool": INDUSTRIES},
           "counts": {"films_read": C.N, "films_read_that_are_reference_documents": int(ISREF.sum()), "films_of_the_indian_industries_read": int(C.indian.sum()),
                      "of_them_reference_documents": int(ISREF[C.indian].sum()), "entries": N, "sources_that_are_reference_documents": int(SRC_REF.sum()),
                      "remakes_that_are_reference_documents": int(REM_REF.sum()), "entries_by_remake_and_source_industry": dict(Counter(f"{a} <- {b}" for a, b in zip(RO.tolist(), SO.tolist())).most_common()),
                      "year_gap_of_the_sources": {"median": float(np.median(YEAR[RR] - YEAR[SS])), "share_at_most_5": float(((YEAR[RR] - YEAR[SS]) <= 5).mean()),
                                                  "share_at_most_10": float(((YEAR[RR] - YEAR[SS]) <= 10).mean()), "largest": float((YEAR[RR] - YEAR[SS]).max())}},
           **res, "seconds": round(time.time() - T0, 1), "peak_memory_gb": peak_gb()}
    R.save(out, OUT)


if __name__ == "__main__":
    main()
