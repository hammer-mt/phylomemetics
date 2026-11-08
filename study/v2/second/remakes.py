"""The remakes corpus: the tests of v2/second/PLAN.md Section 9.

    cd study
    ../.venv/bin/python v2/second/remakes.py words           everything that needs no model -> results/v2/second/remakes_words.json
    ../.venv/bin/python v2/second/remakes.py run <tag>       one model -> results/v2/second/remakes_<tag>.json
    ../.venv/bin/python v2/second/remakes.py verify <tag>    the GPU machine's counts recounted here from presence bits -> results/v2/second/remakes_verify_<tag>.json
    ../.venv/bin/python v2/second/remakes.py synthetic       'run' and 'verify' end to end on made-up counts, bits and recognition (then deleted)

The test is v2/second/remake_test.py, the designers' file, unchanged: this file only builds what it asks for and adds up
what it returns. It holds no list of films or pairs: it reads data/second/remakes/pairs.json and
v2/cache/second/remakes/films.jsonl each time it runs and records their SHA-256, and the commit, in every output.

SETS (each analysed on its own). "primary": the firm pairs of the Indian cross-industry block. Secondary:
"firm_other_cross_industry", "firm_same_industry", "firm_all" (every firm pair), "loose_all" (every loose pair).

ENTRIES AND POOLS. Each documented pair of a set is one entry (a remake with two documented sources is two entries of one
family). An entry's pool is its source plus its candidates. A candidate is never later than the remake by the table's year
and never a film of the remake's family (films.jsonl "family"). Which films, by frame:
    "set" (primary for every set; it uses only what defines the set)
        primary                      films of the Indian industries other than the remake's own
        firm_other_cross_industry    films of any industry other than the remake's own
        firm_same_industry           films of the remake's own industry
        firm_all, loose_all          every film read
    "source_industry" (secondary; primary set only)   films of the source's industry
    "all" (secondary; primary set only)               every film read

THE DECISION TEST for one set under one frame is remake_test.RemakeTest on its entries: 31 word columns, the conditional
logit fitted leaving one family out at a time, the unit column as within-pool normal scores of the hub-corrected rate
(hub reference: the other entries of the same set that are outside the film's family), the slope's family sums and their
sign-flip p, the gain of the refitted model, 78 word-only stand-ins in the units' place. Every set and frame goes through
all of it by itself. For the primary set under frame "set" the slope uses every cell. Everywhere else, and in every
simulation, it leaves out the cells to which the word model gives a probability under PRUNE (remake_test's own
cache_prune); 'words' shows what that changes on the primary set (pruning_check) and 'run' shows it for the units.

THE UNIT MEASURE arrives as counts (v2/gpu/copresence.py): v2/acts_second/remakes/<tag>/counts/<plane>.npz with shared
[rows, films] and n_optional [films], the rows being the remakes in the order of data/second/remakes/rows.json and the
columns the films of v2/acts_second/remakes/<tag>/meta.json "ids". rate[r, c] = shared[r, c] / max(n_optional[c], 1). A
model that read only some of the films gets the sets and frames all of whose pools it read; the rest is listed as left out.

'run' refuses without results/v2/second/remakes_words.json made from the same pairs.json, films.jsonl and word-measure
matrices, and without results/v2/second/remakes_recognition_<tag>.json (v2/second/probes.py: {"r1": {page: {"identified":
bool}}, "r2": {remake page: {"source_named": bool}}}). It reads the stand-ins' results from remakes_words.json: they
depend on the words only, and are fixed before any unit is read.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import argparse
import hashlib
import json
import re
import resource
import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

sys.path.insert(0, "v2/second")
import remake_test as T  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
PAIRS, FILMS, ROWS = Path("data/second/remakes/pairs.json"), Path("v2/cache/second/remakes/films.jsonl"), Path("data/second/remakes/rows.json")
CACHE, OUT, ACTS = Path("v2/cache/second/remakes"), Path("results/v2/second"), Path("v2/acts_second/remakes")
WORDS_FILE = "remakes_words.json"
INDIAN = {"Bollywood", "Tamil", "Telugu", "Malayalam", "Kannada", "Bengali", "Marathi", "Punjabi", "Assamese"}       # build_remakes.INDIAN
SETS = {"primary": lambda p: p["wording"] == "firm" and p["block"] == "indian_cross_industry",
        "firm_other_cross_industry": lambda p: p["wording"] == "firm" and p["block"] == "other_cross_industry",
        "firm_same_industry": lambda p: p["wording"] == "firm" and p["block"] == "same_industry",
        "firm_all": lambda p: p["wording"] == "firm",
        "loose_all": lambda p: p["wording"] == "loose"}
SET_FRAME = {"primary": "indian industries other than the remake's own", "firm_other_cross_industry": "any industry other than the remake's own",
             "firm_same_industry": "the remake's own industry", "firm_all": "every film read", "loose_all": "every film read"}
PRIMARY_SET, PRIMARY_FRAME, PRIMARY_PLANE = "primary", "set", "top1_30"
ANALYSES = [(s, "set") for s in SETS] + [(PRIMARY_SET, "source_industry"), (PRIMARY_SET, "all")]      # every (set, frame) that is run
PLANES = [f"{s}_{t}" for s in ("top1", "top3") for t in (20, 30, 40, 50)]
MIN_PLANES = 4                                    # condition 6: the units' p below 0.05 in at least this many of the 8 planes
PRUNE = 1e-8                                      # remake_test's cache_prune: cells the word model gives a probability under this are left out of the slope
#                                                   (the designers simulated with 1e-6; 'words' shows what 1e-8 changes on the primary set: pruning_check)
SEED, DRAWS = 20261108, 10000                     # the heredity null: labellings, the true one counted as one of them
CONTROL_NEED = 0.90                               # positive control: the source strictly above every candidate in at least this share of the copied pairs
SOLVED = 0.99                                     # subset: contributions set to zero where the held-out probability of the source is at least this
LOW_COPY = 0.01                                   # subset: pairs whose share of copied five-word sequences is under this
N_LARGEST = 5                                     # the families of largest |D_f|, each left out in turn
LAMBDAS = (1.0, 10.0)                             # reported beside the registered penalty
HEREDITY_WORDS = {"tfidf_cosine": "cos_tfidf", "bm25": "bm25", "word_presence": "contain_opt"}
NAMES_FREE = "tfidf_words_never_capitalised_inside_a_sentence"      # the extra stand-in a reviewer asked for; not one of the 78
SPLIT_SEEDS, SPLIT_MEASURES = (1, 2, 3), ("contain_opt", "cos_tfidf", "bm25", "lsa150")       # the withheld-vocabulary positive control
BOOSTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]                                      # planted boosts, in pool standard deviations
SIM_FLIPS = 5000                                  # sign flips per simulated draw (the designers' calibrate.py)
WORKERS = 2
OUTCOMES = ("not evaluable", "not calibrated", "no information beyond these word measures was detected", "detected at the primary setting only",
            "recall", "cannot be separated from recall", "claim")
peak_gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20      # noqa: E731
G = {}                                            # what forked workers read


# ------------------------------------------------------------------ output (the conventions of lib.py, without its imports of the Batman corpus)
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def git_head():
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def plain(o):
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save(obj, path, quiet=False):
    """Every number unrounded. numpy scalars and arrays become plain numbers and lists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=plain, ensure_ascii=False))
    if not quiet:
        print("wrote", path, flush=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def settings():
    """Every constant a result depends on, written into every output."""
    return {"lambda": T.LAMBDA, "alpha_one_sided": T.ALPHA, "sign_flips": T.N_FLIPS, "exact_enumeration_up_to_families": T.EXACT_UP_TO, "seed": T.SEED,
            "effective_pairs_needed": T.MIN_UNSOLVED, "standin_share_limit": T.MAX_STANDIN_SHARE, "word_columns": len(T.WORD_COLUMNS) + 1,
            "prune": PRUNE, "cells": {"primary set, frame set": "every cell", "every other set and frame, and every simulation":
                                      f"cells the word model gives a probability under {PRUNE:g} are left out of the slope (remake_test's cache_prune)"},
            "planes": PLANES, "primary_plane": PRIMARY_PLANE, "planes_needed": MIN_PLANES, "heredity_draws": DRAWS, "heredity_seed": SEED,
            "positive_control_share_needed": CONTROL_NEED, "solved_probability": SOLVED, "low_copy_share": LOW_COPY, "largest_families_left_out": N_LARGEST,
            "other_penalties_reported": list(LAMBDAS), "frame_set": SET_FRAME, "analyses": [f"{s} | {f}" for s, f in ANALYSES],
            "scikit_learn": __import__("sklearn").__version__, "numpy": np.__version__, "scipy": __import__("scipy").__version__}


# ------------------------------------------------------------------ the corpus, rebuilt from the two files every time
class Corpus:
    def __init__(self, pairs=PAIRS, films=FILMS, rows=ROWS):
        self.sha = {"pairs.json": sha256(pairs), "films.jsonl": sha256(films)}
        self.pairs = json.load(open(pairs))["pairs"]
        F = [json.loads(l) for l in open(films)]
        assert [f["id"] for f in F] == list(range(len(F))), "a film's id must be its row number"
        self.N = len(F)
        self.page, self.title = [f["page"] for f in F], [f["title"] for f in F]
        self.year, self.origin = np.array([f["year"] for f in F]), np.array([f["origin"] for f in F])
        self.in_pair = np.array([bool(f["in_pair"]) for f in F])
        self.indian = np.isin(self.origin, sorted(INDIAN))
        self.fam = np.array([-1 if f["family"] is None else int(f["family"]) for f in F])                 # -1: a relative of no pair
        self.film_family = np.array([f["family"] if f["family"] is not None else "" for f in F], dtype=object)     # as remake_test wants it
        self.sentences = [f["sentences"] for f in F]
        self.texts = [" ".join(f["sentences"]) for f in F]              # as every word measure reads a film
        row = {p: i for i, p in enumerate(self.page)}
        for k, p in enumerate(self.pairs):
            p["r"], p["s"], p["k"] = row[p["remake"]], row[p["source"]], k
            assert self.fam[p["r"]] == self.fam[p["s"]] == p["family"], "a pair's family is not its films' family"
            assert self.year[p["s"]] <= self.year[p["r"]] and min(len(self.sentences[p["r"]]), len(self.sentences[p["s"]])) >= T.MIN_SENTENCES
            ro, so = self.origin[p["r"]], self.origin[p["s"]]
            assert p["block"] == ("indian_cross_industry" if ro in INDIAN and so in INDIAN and ro != so else "same_industry" if ro == so else "other_cross_industry")
        in_pairs = {i for p in self.pairs for i in (p["r"], p["s"])}
        assert set(np.flatnonzero(self.in_pair).tolist()) == in_pairs, "in_pair does not match the pair list"
        self.remakes = np.array(sorted({p["r"] for p in self.pairs}))                             # the rows of every [remakes, films] matrix here
        self.rpos = {int(r): i for i, r in enumerate(self.remakes)}
        self.pair_films = sorted(in_pairs)
        self.rows_json = json.load(open(rows)) if rows and Path(rows).exists() else None
        self.rows_json_matches = None if self.rows_json is None else bool(self.rows_json["rows"] == self.remakes.tolist() and self.rows_json["pair_films"] == self.pair_films)

    def entries(self, name):
        return [p for p in self.pairs if SETS[name](p)]

    def pool(self, p, name, frame):
        """The pool of one entry, in film-id order: its candidates under the frame, plus its source."""
        r, s = p["r"], p["s"]
        ok = self.year <= self.year[r]                                  # never a film later than the remake
        if frame == "set":
            if name == "primary":
                ok &= self.indian & (self.origin != self.origin[r])
            elif name == "firm_other_cross_industry":
                ok &= self.origin != self.origin[r]
            elif name == "firm_same_industry":
                ok &= self.origin == self.origin[r]
            else:
                assert SET_FRAME[name] == "every film read"
        elif frame == "source_industry":
            ok &= self.origin == self.origin[s]
        else:
            assert frame == "all", frame
        ok &= self.fam != p["family"]                                   # never a film of the remake's family
        ok[s] = True
        return np.flatnonzero(ok)


def counts(C):
    """The counts PLAN Sections 3 and 15 quote."""
    P = C.pairs
    by = lambda f, S=P: {" | ".join(map(str, k)): v for k, v in sorted(Counter(f(p) for p in S).items(), key=str)}       # noqa: E731
    out = {"pairs": len(P), "families_of_the_pairs": len({p["family"] for p in P}), "remakes": len(C.remakes), "sources": len({p["s"] for p in P}),
           "films_in_a_pair": int(C.in_pair.sum()), "films_that_are_relatives_of_a_pair": int((C.fam >= 0).sum()),
           "films_that_are_both_a_remake_and_a_source": len({p["r"] for p in P} & {p["s"] for p in P}),
           "remakes_with_two_or_more_documented_sources": sum(v > 1 for v in Counter(p["r"] for p in P).values()),
           "pairs_by_block_and_wording": by(lambda p: (p["block"], p["wording"])),
           "pairs_by_block_wording_and_copied": by(lambda p: (p["block"], p["wording"], "copied" if p["copied"] else "not copied")),
           "copied_pairs": sum(bool(p["copied"]) for p in P),
           "films_read": C.N, "films_read_by_industry": dict(Counter(C.origin.tolist()).most_common()),
           "films_read_of_the_indian_industries": int(C.indian.sum()),
           "pair_films_by_industry": dict(Counter(C.origin[C.in_pair].tolist()).most_common()),
           "rows_json_matches_this_list": C.rows_json_matches, "sets": {}}
    for name in SETS:
        S = C.entries(name)
        d = {"entries": len(S), "copied": sum(bool(p["copied"]) for p in S), "not_copied": sum(not p["copied"] for p in S),
             "copy_share_under_1pct": sum(p["share_of_five_word_sequences_also_in_source"] < LOW_COPY for p in S),
             "remakes": len({p["r"] for p in S}), "sources": len({p["s"] for p in S}), "families": len({p["family"] for p in S}),
             "families_with_two_or_more_entries": sum(v > 1 for v in Counter(p["family"] for p in S).values()),
             "remakes_with_two_or_more_documented_sources": sum(v > 1 for v in Counter(p["r"] for p in S).values()),
             "remake_industry": by(lambda p: (C.origin[p["r"]],), S), "source_industry": by(lambda p: (C.origin[p["s"]],), S), "pools": {}}
        for s2, frame in ANALYSES:
            if s2 == name:
                n = np.array([len(C.pool(p, name, frame)) for p in S])
                d["pools"][frame] = {"films": SET_FRAME[name] if frame == "set" else {"source_industry": "the source's industry", "all": "every film read"}[frame],
                                     "cells": int(n.sum()), "min": int(n.min()), "median": float(np.median(n)), "mean": float(n.mean()), "max": int(n.max())}
        out["sets"][name] = d
    import build_holmes
    ref = build_holmes.reference_overlap([C.page[i] for i in C.pair_films])
    prim = sorted({i for p in C.entries(PRIMARY_SET) for i in (p["r"], p["s"])})
    tally = lambda pages: dict(Counter(ref[p][1] for p in pages if p in ref))                    # noqa: E731
    out["pair_films_in_v2_reference_corpus"] = {
        "pair_films": len(C.pair_films), "are_reference_documents": len(ref), "by_kind": tally(ref),
        "primary_set_films": len(prim), "primary_set_films_that_are_reference_documents": sum(C.page[i] in ref for i in prim),
        "primary_by_kind": tally([C.page[i] for i in prim]), "documents": {p: ref[p] for p in sorted(ref)}}
    return out


# ------------------------------------------------------------------ the word measures, saved (the LSA column is a randomised SVD)
def measures_key(C):
    return {"films.jsonl": C.sha["films.jsonl"], "remakes": hashlib.sha256(json.dumps(C.remakes.tolist()).encode()).hexdigest()}


def word_measures(C, cache=CACHE, must_exist=False):
    """remake_test.word_measures for every remake against every film read, saved under v2/cache/second/remakes/ with the hash
    of each matrix. They depend on the texts and on which films are remakes, not on the rest of the pair list."""
    f, meta = cache / "word_measures.npz", cache / "word_measures.json"
    if f.exists() and meta.exists():
        m = json.load(open(meta))
        if m["key"] == measures_key(C) and m["file_sha256"] == sha256(f):
            M = dict(np.load(f))
            assert {k: sha_array(v) for k, v in M.items()} == m["sha256"], "the saved word measures are not the ones that were hashed"
            return M, m
    if must_exist:
        raise SystemExit(f"{f} is missing or was not made from the current films.jsonl and pair list: run `remakes.py words` first")
    t0 = time.time()
    M = T.word_measures(C.texts, C.remakes)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **M)
    m = {"key": measures_key(C), "sha256": {k: sha_array(v) for k, v in M.items()}, "file_sha256": sha256(f), "shape": list(next(iter(M.values())).shape),
         "seconds": round(time.time() - t0, 1), "scikit_learn": __import__("sklearn").__version__, "numpy": np.__version__, "scipy": __import__("scipy").__version__}
    json.dump(m, open(meta, "w"), indent=1)
    log(f"word measures: {len(M)} matrices {m['shape']} in {m['seconds']} s -> {f}")
    return M, m


# ------------------------------------------------------------------ the stand-in panel (word-only measures that are not covariates)
def names_free_measure(C):
    """The extra stand-in a reviewer asked for: TF-IDF cosine (cos_tfidf's settings) over the words that are never capitalised
    inside a sentence anywhere in the films read. A word counts as capitalised inside a sentence when, in any stored sentence
    of any film read, it stands anywhere but first and begins with a capital letter. Names of people and places go; so does
    any ordinary word that some film capitalises once. Returns the matrix [remakes, films] and how many words are left."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    token = re.compile(r"(?u)\b\w\w+\b")                                 # scikit-learn's default token
    capital = set()
    for sents in C.sentences:
        for s in sents:
            capital.update(w.lower() for w in token.findall(s)[1:] if w[0].isupper())
    vocab = CountVectorizer(stop_words="english", min_df=2).fit(C.texts).get_feature_names_out()
    kept = [w for w in vocab if w not in capital]
    A = TfidfVectorizer(stop_words="english", sublinear_tf=True, vocabulary=kept).fit_transform(C.texts)
    M = np.asarray((A[C.remakes] @ A.T).todense(), dtype=np.float32)
    return M, {"words_of_cos_tfidf": len(vocab), "words_never_capitalised_inside_a_sentence": len(kept), "words_dropped": len(vocab) - len(kept)}


def standin_panel(C, cache, chunks):
    """remake_test.standin_measures for every remake against every film read, plus the names-free measure: {name: float32
    [remakes, films]}, and what is known of the names-free vocabulary. The designers' function holds every matrix in float64
    until it returns (4.8 GB for 471 remakes and 16,218 films, on top of its working memory), so it is called on the remakes a
    part at a time: a remake's row of a stand-in depends on that remake and on the films read, not on the other remakes. That
    this holds exactly is checked on one remake computed in every call. The matrices are written to a scratch file under the
    cache (2.4 GB), which 'words' deletes when it has finished."""
    import inspect
    f, fj = cache / "standin_panel.scratch.npy", cache / "standin_panel.scratch.json"
    key = dict(measures_key(C), remake_test_code=test_code_sha(), names_free_measure=hashlib.sha256(inspect.getsource(names_free_measure).encode()).hexdigest())
    if f.exists() and fj.exists() and json.load(open(fj))["key"] == key:
        st = json.load(open(fj))
        A = np.load(f, mmap_mode="r")
        log(f"stand-in measures: read from {f}")
        return {k: A[j] for j, k in enumerate(st["names"])}, st["names_free_measure"], st["seconds"], False
    for x in (f, fj):
        x.unlink(missing_ok=True)
    n, t0, A, names, first = len(C.remakes), time.time(), None, None, None
    edges = np.linspace(0, n, chunks + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        P = T.standin_measures(C.texts, np.concatenate([C.remakes[:1], C.remakes[a:b]]))          # the first remake again in every call: the check
        if A is None:
            names, first = list(P), {k: v[0].copy() for k, v in P.items()}
            assert len(names) == 78 and NAMES_FREE not in names
            A = np.lib.format.open_memmap(f, mode="w+", dtype=np.float32, shape=(len(names) + 1, n, C.N))
        assert list(P) == names and all(np.array_equal(P[k][0], first[k]) for k in names), "a stand-in's row depends on which other remakes are computed with it"
        for j, k in enumerate(names):
            A[j, a:b] = P[k][1:]
        del P
        log(f"stand-in measures: remakes {a} to {b} of {n} ({time.time() - t0:.0f} s; peak memory {peak_gb():.1f} GB)")
    A[len(names)], info = names_free_measure(C)
    A.flush()
    names.append(NAMES_FREE)
    seconds = round(time.time() - t0, 1)
    json.dump({"key": key, "names": names, "names_free_measure": info, "seconds": seconds, "parts": chunks}, open(fj, "w"))
    A = np.load(f, mmap_mode="r")
    return {k: A[j] for j, k in enumerate(names)}, info, seconds, True


def drop_panel_scratch(cache):
    for x in (cache / "standin_panel.scratch.npy", cache / "standin_panel.scratch.json"):
        x.unlink(missing_ok=True)


# ------------------------------------------------------------------ one set of entries under one frame
class Folds:
    """remake_test's cache of folds (RemakeTest(..., cache_prune=prune)), each computed when it is asked for and not kept: for
    the larger frames the cache of every family at once does not fit in memory. The numbers are cache_prune's, bit for bit."""

    def __init__(self, test, prune):
        self.test, self.prune = test, prune

    def __getitem__(self, f):
        return self.test._fold(f, self.prune)


class Analysis:
    """remake_test.RemakeTest on the entries of one set under one frame: the 31 word columns, the conditional logit fitted
    leaving one family out at a time. M: the word measures [remakes, films] (with by_entry, already one row per entry).
    extra: further measures, one row per entry, for further columns."""

    def __init__(self, C, M, name, frame, lam=T.LAMBDA, columns=None, extra=None, by_entry=False):
        self.C, self.name, self.frame, self.lam = C, name, frame, lam
        self.entries = C.entries(name)
        self.pools = [C.pool(p, name, frame) for p in self.entries]
        self.rows = np.array([C.rpos[p["r"]] for p in self.entries])            # each entry's row in a [remakes, films] matrix
        self.src = np.array([p["s"] for p in self.entries])
        self.families = [p["family"] for p in self.entries]
        self.copied = np.array([bool(p["copied"]) for p in self.entries])
        self.copy_share = np.array([p["share_of_five_word_sequences_also_in_source"] for p in self.entries])
        self.exact = (name, frame) == (PRIMARY_SET, PRIMARY_FRAME)              # the registered decision uses every cell
        cells = [{"pool": pool, "source": p["s"], "family": p["family"]} for p, pool in zip(self.entries, self.pools)]
        need = {m for m, _ in (columns or T.WORD_COLUMNS)}
        meas = {k: (v if by_entry else np.asarray(v)[self.rows]) for k, v in M.items() if k in need}
        meas.update(extra or {})
        t0 = time.time()
        self.test = T.RemakeTest(cells, meas, C.film_family, lam=lam, columns=columns)
        self.test._hub.clear()                                                  # only the constructor reads it
        self.seconds_fit = round(time.time() - t0, 1)
        self._folds = None

    def label(self):
        return f"{self.name} | {self.frame}"

    def cells_label(self, prune="default"):
        prune = self.prune(prune)
        return "every cell" if prune is None else f"cells the word model gives a probability under {prune:g} are left out"

    def prune(self, prune="default"):
        return (None if self.exact else PRUNE) if prune == "default" else prune

    def per_entry(self, M):
        """A [remakes, films] matrix with one row per entry."""
        return np.asarray(M)[self.rows]

    def hub(self, E):
        """A [entries, films] measure hub-corrected as remake_test does it for the unit column: each cell as a z-score against
        the same film's values with the other entries of this set that are outside the film's family."""
        return T.hub_z(E, self.families, self.C.film_family)

    def cache(self, prune):
        """What RemakeTest.slope reads its folds from."""
        return None if prune is None else self._folds if (self._folds is not None and prune == PRUNE) else Folds(self.test, prune)

    def keep_folds(self):
        """Hold every pruned fold in memory (for the simulations, which test thousands of columns one at a time)."""
        self._folds = [self.test._fold(f, PRUNE) for f in range(self.test.S.n_fam)]
        self.test._cache = self._folds

    def drop_folds(self):
        self._folds = self.test._cache = None

    def contributions(self, U, prune="default"):
        """remake_test.RemakeTest.slope for one stacked unit column [cells] or a block [cells, m]: each entry's contribution U_r."""
        t, old = self.test, self.test._cache
        t._cache = self.cache(self.prune(prune))
        try:
            return t.slope(np.asarray(U))
        finally:
            t._cache = old

    def summary(self, Ur, keep=None):
        """T, Z and p exactly as remake_test.RemakeTest.slope_test computes them, from the entries' contributions; with keep,
        from the contributions of those entries only (nothing refitted). Also the open mass E of the entries counted."""
        S = self.test.S
        k = np.ones(len(Ur), bool) if keep is None else np.asarray(keep, bool)
        d = S.family_sum(np.where(k, Ur, 0.0))
        ss = float(np.sqrt((d ** 2).sum()))
        return {"entries": int(k.sum()), "families": len(set(S.fam[k].tolist())), "T": float(d.sum()), "Z": float(d.sum() / ss) if ss > 0 else 0.0,
                "p": T.sign_flip_p(d), "families_positive": int((d > 0).sum()), "families_negative": int((d < 0).sum()),
                "effective_pairs_left_open": float(((1 - self.test.p_source) * k).sum())}, d


def standins(an, P, form="rate_matrix", prune="default", block_cells=1.2e8, by_entry=False):
    """remake_test.standins_through_test for the measures of P ({name: [remakes, films]}, or with by_entry one row per entry),
    a block of them at a time."""
    names, out = list(P), {}
    block = max(2, int(block_cells // an.test.S.n))
    t, old = an.test, an.test._cache
    t._cache = an.cache(an.prune(prune))
    try:
        for b in range(0, len(names), block):
            out.update(T.standins_through_test(t, {k: (P[k] if by_entry else an.per_entry(P[k])) for k in names[b:b + block]}, form=form))
    finally:
        t._cache = old
    return out


def standin_summary(st):
    """Over the 78 registered stand-ins (the names-free measure is reported beside them and not counted)."""
    st = {k: v for k, v in st.items() if k != NAMES_FREE}
    ps, top, low = np.array([v["p"] for v in st.values()]), max(st, key=lambda k: st[k]["Z"]), min(st, key=lambda k: st[k]["p"])
    return {"n": len(ps), "below_0.05": int((ps < T.ALPHA).sum()), "share_below_0.05": float(np.mean(ps < T.ALPHA)), "below_0.10": int((ps < 0.10).sum()),
            "above_0.90": int((ps > 0.90).sum()), "lowest_p": float(ps.min()), "lowest_p_measure": low, "z_max": float(st[top]["Z"]), "z_max_measure": top,
            "at_most_10pct_below_0.05": bool(np.mean(ps < T.ALPHA) <= T.MAX_STANDIN_SHARE)}


def word_model(an):
    """What the 31 word columns do alone, each entry scored by the weights fitted without its family: for all entries, and for
    those that are not copied (the same fit; the copied entries left out of the tally)."""
    S, ps, r0 = an.test.S, an.test.p_source, an.test.rank0

    def side(keep):
        open_ = (1 - ps) * keep
        return {"entries": int(keep.sum()), "families": len(set(S.fam[keep].tolist())), "sources_ranked_first": int(((r0 == 1) & keep).sum()),
                "ranks_of_the_sources_not_first": sorted(r0[keep & (r0 != 1)].tolist()), "effective_pairs_left_open": float(open_.sum()),
                "evaluable": bool(open_.sum() >= T.MIN_UNSOLVED), "held_out_log_likelihood": float(an.test.ll0[keep].sum()),
                "families_with_held_out_source_probability_under_0.9": len(set(S.fam[keep & (ps < 0.9)].tolist())),
                "entries_with_held_out_source_probability_under": {str(x): int((keep & (ps < x)).sum()) for x in (SOLVED, 0.9, 0.5)},
                "families_with_more_than_a_tenth_of_a_pair_open": int((S.family_sum(open_) > 0.1).sum()),
                "families_with_more_than_half_a_pair_open": int((S.family_sum(open_) > 0.5).sum())}
    return {"set": an.name, "frame": an.frame, "columns": len(an.test.names), "lambda": an.lam, "cells": int(S.n), "pool_min": int(S.sizes.min()),
            "pool_median": float(np.median(S.sizes)), "pool_max": int(S.sizes.max()), "seconds_to_fit": an.seconds_fit,
            "all_entries": side(np.ones(len(ps), bool)), "not_copied": side(~an.copied)}


# ------------------------------------------------------------------ heredity: where a similarity ranks the documented source
def heredity(an, H, keep=None, draws=DRAWS, seed=SEED):
    """H: a similarity [entries, films] (the hub-corrected rate). For each entry counted, the share of its candidates that are
    strictly above its source (ties not counted; a version with ties as half beside). S: the mean over families of the family's
    mean share. Null: in each of draws - 1 relabellings every family draws, uniformly, one film from those that are candidates
    of every one of its entries counted, and that film plays the source of all of them (it is compared with the other films of
    each entry's pool); the true labelling is counted as one more. p: the share of the `draws` labellings whose S is at or below
    the observed S. A family none of whose candidates is shared by all its entries lets each entry draw its own film."""
    idx = np.arange(len(an.entries)) if keep is None else np.flatnonzero(keep)
    if len(idx) == 0:
        return {"entries": 0}
    sorted_h, share, half, rank = {}, {}, {}, {}
    for i in idx:
        srt = np.sort(H[i, an.pools[i]])
        right, left = np.searchsorted(srt, H[i, an.src[i]], side="right"), np.searchsorted(srt, H[i, an.src[i]], side="left")
        n = len(srt) - 1                                                 # the candidates: the pool without the source
        sorted_h[i], share[i], half[i], rank[i] = srt, (len(srt) - right) / n, (len(srt) - right + 0.5 * (right - left - 1)) / n, len(srt) - left
    fams = sorted({an.families[i] for i in idx})
    members = {f: [i for i in idx if an.families[i] == f] for f in fams}
    S = float(np.mean([np.mean([share[i] for i in members[f]]) for f in fams]))
    S_half = float(np.mean([np.mean([half[i] for i in members[f]]) for f in fams]))
    rng = np.random.default_rng(seed)
    u = rng.random((draws - 1, len(fams)))
    null, null_half, alone = np.zeros(draws - 1), np.zeros(draws - 1), 0

    def shares_of(i, films):
        """For entry i, were each of these films its source: the share of the other films of its pool above it, without and with ties."""
        srt = sorted_h[i]
        right, left = np.searchsorted(srt, H[i, films], side="right"), np.searchsorted(srt, H[i, films], side="left")
        return (len(srt) - right) / (len(srt) - 1), (len(srt) - right + 0.5 * (right - left - 1)) / (len(srt) - 1)

    for j, f in enumerate(fams):
        cands = [np.setdiff1d(an.pools[i], [an.src[i]]) for i in members[f]]
        common = cands[0]
        for c in cands[1:]:
            common = np.intersect1d(common, c)
        if len(common):
            v = np.mean([shares_of(i, common) for i in members[f]], axis=0)       # [2, films]: the family's mean share for each film
            pick = np.minimum((u[:, j] * len(common)).astype(np.int64), len(common) - 1)
            null, null_half = null + v[0][pick], null_half + v[1][pick]
        else:
            alone += 1
            for i, c in zip(members[f], cands):
                v = shares_of(i, c)
                pick = np.minimum((rng.random(draws - 1) * len(c)).astype(np.int64), len(c) - 1)
                null, null_half = null + v[0][pick] / len(cands), null_half + v[1][pick] / len(cands)
    null, null_half = null / len(fams), null_half / len(fams)
    ranks = np.array([rank[i] for i in idx])                             # 1 + the number of candidates at or above the source: 1 means strictly above every candidate
    return {"entries": len(idx), "families": len(fams), "S": S, "S_ties_half": S_half,
            "p": float((1 + (null <= S + 1e-12).sum()) / draws), "p_ties_half": float((1 + (null_half <= S_half + 1e-12).sum()) / draws),
            "null_mean": float(null.mean()), "null_min": float(null.min()), "draws": draws, "seed": seed,
            "families_with_no_candidate_shared_by_all_their_entries": alone,
            "source_strictly_above_every_candidate": int((ranks == 1).sum()), "share_source_first": float(np.mean(ranks == 1)),
            "source_in_the_top_10": int((ranks <= 10).sum()), "median_rank_of_the_source": float(np.median(ranks)),
            "mean_share_over_entries": float(np.mean([share[i] for i in idx])),
            "median_candidates": float(np.median([len(an.pools[i]) - 1 for i in idx]))}


def heredity_report(an, E, draws=DRAWS):
    """Heredity by one similarity (E: [entries, films], not yet hub-corrected): on the entries that are not copied (the
    confirmatory form on the primary set), on all entries, and the positive control on the copied entries. The same without
    the hub correction beside it."""
    out = {}
    for label, H in (("hub_corrected", an.hub(E)), ("no_hub_correction", np.asarray(E, dtype=np.float64))):
        d = {"not_copied": heredity(an, H, ~an.copied, draws), "all_entries": heredity(an, H, None, draws)}
        c = heredity(an, H, an.copied, 2)                                # the count only: no null is needed for the control
        if c["entries"]:
            d["positive_control_copied_entries"] = {
                "entries": c["entries"], "source_strictly_above_every_candidate": c["source_strictly_above_every_candidate"], "share": c["share_source_first"],
                "needed": CONTROL_NEED, "passed": bool(c["share_source_first"] >= CONTROL_NEED),
                "reading": "holds" if c["share_source_first"] >= CONTROL_NEED else "failed positive control"}
        out[label] = d
    return out


# ------------------------------------------------------------------ 'words': the calibration on the final list (the designers' calibrate.py, ported)
def primary_panel_file(cache=CACHE):
    return cache / "standin_panel_primary.npz"


def save_primary_panel(C, P, an, cache=CACHE):
    """The stand-ins for the entries of the primary set against the films of the Indian industries (every film a primary pool
    under frame "set" can hold), kept so that 'run' can put them through a model with further columns. Returns the file's hash."""
    cols = np.flatnonzero(C.indian)
    assert all(np.isin(pool, cols).all() for pool in an.pools)
    np.savez_compressed(primary_panel_file(cache), _cols=cols, _entry_remakes=C.remakes[an.rows], **{k: an.per_entry(v)[:, cols] for k, v in P.items()})
    return sha256(primary_panel_file(cache))


def load_primary_panel(C, an, want_sha, cache=CACHE):
    f = primary_panel_file(cache)
    if not f.exists() or sha256(f) != want_sha:
        raise SystemExit(f"{f} is missing or is not the file `remakes.py words` recorded: run `remakes.py words` again")
    Z = np.load(f)
    assert np.array_equal(Z["_entry_remakes"], C.remakes[an.rows])
    cols, out = Z["_cols"], {}
    for k in Z.files:
        if not k.startswith("_"):
            out[k] = np.zeros((len(an.entries), C.N), np.float32)
            out[k][:, cols] = Z[k]
    return out


def compare_standins(a, b):
    """How far two runs of the same stand-ins differ (every cell against pruned cells)."""
    names = list(a)
    dp, dz, dt = (np.array([abs(a[k][x] - b[k][x]) for k in names]) for x in ("p", "Z", "T"))
    return {"n": len(names), "largest_change_in_p": float(dp.max()), "largest_change_in_Z": float(dz.max()), "largest_change_in_T": float(dt.max()),
            "stand_ins_that_change_side_of_0.05": int(sum((a[k]["p"] < T.ALPHA) != (b[k]["p"] < T.ALPHA) for k in names)),
            "largest_Z": [float(max(a[k]["Z"] for k in names if k != NAMES_FREE)), float(max(b[k]["Z"] for k in names if k != NAMES_FREE))]}


def stage_primary_standins(an, P, info):
    """The registered calibration: the 78 stand-ins (and the names-free measure beside them) through the test in the units'
    place on the primary set under frame "set", every cell; the same with pruned cells; the same with no hub correction."""
    t0 = time.time()
    exact = standins(an, P)
    log(f"primary stand-ins, every cell: {time.time() - t0:.0f} s")
    pruned = standins(an, P, prune=PRUNE)
    raw = standins(an, P, form="raw_matrix")
    reg = {k: v for k, v in exact.items() if k != NAMES_FREE}
    # the same stand-ins over subsets of the entries (sums of the same fit's contributions; nothing refitted): where do they pass?
    names = list(reg)
    Ur = an.contributions(np.column_stack([an.test.unit_column(an.per_entry(P[k]), "rate_matrix") for k in names]))
    by_subset = {}
    for label, keep in (("all_entries", None), ("not_copied", ~an.copied), ("copy_share_under_1pct", an.copy_share < LOW_COPY),
                        ("held_out_source_probability_under_0.99", an.test.p_source < SOLVED)):
        rows = {k: an.summary(Ur[:, j], keep)[0] for j, k in enumerate(names)}
        if keep is None:
            assert all(rows[k][x] == reg[k][x] for k in names for x in ("T", "Z", "p")), "the entries' contributions do not add up to remake_test's stand-in results"
        first = next(iter(rows.values()))
        by_subset[label] = dict(standin_summary(rows), entries=first["entries"], families=first["families"], effective_pairs_left_open=first["effective_pairs_left_open"],
                                below_0_05=sorted(k for k in names if rows[k]["p"] < T.ALPHA))
    return {"registered": {"set": an.name, "frame": an.frame, "cells_in_the_slope": an.cells_label(), "form": "hub-corrected rate, normal scores within the pool",
                           "summary": standin_summary(exact), "rows": reg},
            "names_free_measure_not_counted_among_the_78": dict(exact[NAMES_FREE], name=NAMES_FREE, below_0_05=bool(exact[NAMES_FREE]["p"] < T.ALPHA),
                                                                Z_above_every_registered_stand_in=bool(exact[NAMES_FREE]["Z"] > max(v["Z"] for v in reg.values())), **info),
            "pruning_check": dict(compare_standins(exact, pruned), what=f"the same 79 measures with cells under {PRUNE:g} left out of the slope, against every cell",
                                  summary_pruned=standin_summary(pruned)),
            "no_hub_correction": {"cells_in_the_slope": an.cells_label(), "summary": standin_summary(raw), "rows": {k: v for k, v in raw.items() if k != NAMES_FREE},
                                  "names_free_measure": raw[NAMES_FREE]},
            "by_subset_of_the_entries": by_subset, "seconds": round(time.time() - t0, 1)}


def stage_analysis(job):
    """One further (set, frame), or the primary set at another penalty: the word model's held-out result, the stand-in panel
    and heredity by words. Runs in a forked worker; reads the corpus, the word measures and the panel from G."""
    name, frame, lam = job
    C, M, P, t0 = G["C"], G["M"], G["P"], time.time()
    an = Analysis(C, M, name, frame, lam=lam)
    out = {"word_model": word_model(an)}
    st = standins(an, P)
    out["standins"] = {"cells_in_the_slope": an.cells_label(), "summary": standin_summary(st),
                       "rows": {k: v for k, v in st.items() if k != NAMES_FREE}, "names_free_measure": st[NAMES_FREE]}
    if lam == T.LAMBDA:
        out["heredity_by_words"] = {label: heredity_report(an, an.per_entry(M[m])) for label, m in HEREDITY_WORDS.items()}
    out["seconds"], out["peak_memory_gb"] = round(time.time() - t0, 1), peak_gb()
    log(f"{name} | {frame}" + (f" | lambda {lam:g}" if lam != T.LAMBDA else "") + f": {out['seconds']} s, stand-ins below 0.05: {out['standins']['summary']['below_0.05']} of 78, "
        f"E {out['word_model']['all_entries']['effective_pairs_left_open']:.2f}")
    return job, out


def withheld_vocabulary(C, an):
    """The positive control of the test itself (the designers' calibrate.py split()): the covariates are computed on a random
    half of the vocabulary, and word measures on the other half are put in the units' place. That is what information the
    covariates lack looks like, so the test must pass: p below 0.05 for every one of the four measures in every one of the three
    splits. The same withheld-half measures against covariates that see every word are given beside it. Slopes leave out cells
    under PRUNE, as in the designers' run."""
    token = re.compile(r"(?u)\b\w\w+\b")
    vocab = sorted({w.lower() for t in C.texts for w in token.findall(t)})
    uniq = np.array(sorted({p["r"] for p in an.entries}))
    at = {int(r): i for i, r in enumerate(uniq)}
    rows = np.array([at[p["r"]] for p in an.entries])
    out = {"rule": "passed if p is below 0.05 for every measure in every split", "measures": list(SPLIT_MEASURES), "seeds": list(SPLIT_SEEDS),
           "cells_in_the_slope": an.cells_label(PRUNE), "splits": {}}
    for seed in SPLIT_SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(seed)
        in_a = dict(zip(vocab, rng.random(len(vocab)) < 0.5))
        keep = lambda text, a: token.sub(lambda m: m.group(0) if in_a[m.group(0).lower()] == a else "", text)         # noqa: E731
        MA = {k: v[rows] for k, v in T.word_measures([keep(t, True) for t in C.texts], uniq).items()}
        MB = {k: v[rows] for k, v in T.word_measures([keep(t, False) for t in C.texts], uniq).items() if k in SPLIT_MEASURES}
        half = Analysis(C, MA, an.name, an.frame, by_entry=True)
        del MA
        U = np.column_stack([half.test.unit_column(MB[k], "rate_matrix") for k in SPLIT_MEASURES])
        Uh, Uf = half.contributions(U, PRUNE), an.contributions(U, PRUNE)
        wm = word_model(half)["all_entries"]
        row = {"words_in_half_a": int(sum(in_a.values())), "words": len(vocab),
               "word_model_on_half_a": {k: wm[k] for k in ("sources_ranked_first", "effective_pairs_left_open", "held_out_log_likelihood", "evaluable")}, "measures": {}}
        for j, k in enumerate(SPLIT_MEASURES):
            s, _ = half.summary(Uh[:, j])
            s0, _ = an.summary(Uf[:, j])
            g = half.test.gain(U[:, j])
            row["measures"][k] = {"withheld_from_the_covariates": {"T": s["T"], "Z": s["Z"], "p": s["p"], "gain": g["gain"], "gain_sign_flip_p": g["p"],
                                                                   "sources_first_words": g["sources_first_words"], "sources_first_with_the_measure": g["sources_first_with_units"]},
                                  "covariates_see_every_word": {"T": s0["T"], "Z": s0["Z"], "p": s0["p"]}}
            log(f"withheld vocabulary, split {seed}, {k}: Z {s['Z']:.2f}, p {s['p']:.5f}, gain {g['gain']:.1f} | covariates see every word: Z {s0['Z']:.2f}, p {s0['p']:.3f}")
        row["seconds"] = round(time.time() - t0, 1)
        out["splits"][str(seed)] = row
    ps = [m["withheld_from_the_covariates"]["p"] for r in out["splits"].values() for m in r["measures"].values()]
    gs = [m["withheld_from_the_covariates"]["gain"] for r in out["splits"].values() for m in r["measures"].values()]
    p0 = [m["covariates_see_every_word"]["p"] for r in out["splits"].values() for m in r["measures"].values()]
    out.update(n=len(ps), below_0_05=int(sum(p < T.ALPHA for p in ps)), largest_p=float(max(ps)), gains_positive=int(sum(g > 0 for g in gs)), smallest_gain=float(min(gs)),
               passed=bool(all(p < T.ALPHA for p in ps)),
               same_measures_when_the_covariates_see_every_word={"below_0.05": int(sum(p < T.ALPHA for p in p0)), "lowest_p": float(min(p0)), "highest_p": float(max(p0))})
    return out


def synth(test, base, sigma, boost, rng, film_noise=0.0):
    """calibrate.py synth(): base + sigma * noise, standardised within the pool; then `boost` pool standard deviations added to
    the true source; then handed to the test like a unit measure (which keeps only its order within each pool). film_noise adds
    noise that a film carries into every pool it sits in (a hub effect)."""
    S = test.S
    x = base + sigma * rng.standard_normal(S.n)
    if film_noise:
        x = x + film_noise * rng.standard_normal(int(S.cols.max()) + 1)[S.cols]
    x = S.standardise(x)
    x[S.src] += boost
    return test.unit_column(x, "stacked")


def power_job(args):
    name, sigma, film_noise, boost, seed = args
    test = G["test"]
    r = test.slope_test(synth(test, G["bases"][name], sigma, boost, np.random.default_rng(seed), film_noise), n_flips=SIM_FLIPS, seed=seed)
    return name, sigma, film_noise, boost, r["p"], r["Z"]


def smallest_boost(boosts, rates, target=0.8):
    """calibrate.py: the boost at which the detection rate first crosses the target, by straight-line interpolation."""
    for i in range(1, len(boosts)):
        if rates[i] >= target > rates[i - 1]:
            return float(boosts[i - 1] + (target - rates[i - 1]) / (rates[i] - rates[i - 1]) * (boosts[i] - boosts[i - 1]))
    return None if rates[-1] < target else float(boosts[0])


def power(an, Pp, z_max, reps):
    """Planted boosts (calibrate.py power()): a synthetic unit measure = a word-only base + independent noise, standardised
    within the pool, plus a boost for the true source in pool standard deviations; `reps` draws per boost and base. an must hold
    its pruned folds. Pp: the primary panel, one row per entry."""
    test, S = an.test, an.test.S
    zcol = lambda Mx: S.column(an.hub(Mx))                                                                                # noqa: E731
    twin, outside, align, sat, none = ("the word twin of the unit measure (contain_opt, a covariate)", "a word measure outside the covariates (TF-IDF with stop words kept)",
                                       "a word measure outside the covariates (sentence alignment)", "a saturating function of the word consensus, tanh(consensus / 3)",
                                       "nothing (pure noise)")
    bases = {twin: test.X[:, test.names.index("contain_opt.hz")], outside: zcol(Pp["tfidf_stop_words_kept"]), align: zcol(Pp["sentence_alignment"]),
             sat: S.standardise(np.tanh(S.standardise(test.X[:, :28].mean(1)) / 3)), none: np.zeros(S.n)}
    configs = [(twin, 1.0, 0.0), (none, 1.0, 0.0), (twin, 0.5, 0.0), (outside, 1.0, 0.0), (twin, 1.0, 1.0), (sat, 1.0, 0.0), (align, 1.0, 0.0)]
    assert reps <= 1000
    jobs = [(b, s, fn, bo, 7_000_000 + 100_000 * ci + 1_000 * bi + j) for ci, (b, s, fn) in enumerate(configs) for bi, bo in enumerate(BOOSTS) for j in range(reps)]
    G.update(test=test, bases=bases)
    t0 = time.time()
    with get_context("fork").Pool(WORKERS) as pool:
        res = pool.map(power_job, jobs, chunksize=16)
    out = {"what": "the rate at which a planted boost is detected (one-sided p below 0.05; and with the stand-in rule, Z above every stand-in's Z too)",
           "boosts_in_pool_standard_deviations": BOOSTS, "draws_per_boost_and_base": reps, "sign_flips_per_draw": SIM_FLIPS, "cells_in_the_slope": an.cells_label(PRUNE),
           "standin_z_max_on_this_list": z_max, "bases": {}}
    for b, s, fn in configs:
        rows = [(r[3], r[4], r[5]) for r in res if r[0] == b and r[1] == s and r[2] == fn]
        rate = [float(np.mean([p < T.ALPHA for bo2, p, z in rows if bo2 == bo])) for bo in BOOSTS]
        rule = [float(np.mean([(p < T.ALPHA) and (z > z_max) for bo2, p, z in rows if bo2 == bo])) for bo in BOOSTS]
        key = f"{b}; noise {s:g}" + ("; plus film-level noise 1" if fn else "")
        out["bases"][key] = {"rate_p_below_0.05": rate, "rate_p_below_0.05_and_Z_above_the_stand_ins": rule,
                             "smallest_boost_80pct_power_p_only": smallest_boost(BOOSTS, rate), "smallest_boost_80pct_power_with_the_stand_in_rule": smallest_boost(BOOSTS, rule),
                             "rejection_rate_at_boost_0": rate[0], "correlation_with_the_base": 0.0 if b == none else float(1 / np.sqrt(1 + s * s + fn * fn))}
        log(f"power, {key}: " + "  ".join(f"{bo:g}: {r:.2f}" for bo, r in zip(BOOSTS, rate)) + f" | 80%: {out['bases'][key]['smallest_boost_80pct_power_p_only']}, "
            f"with the stand-in rule {out['bases'][key]['smallest_boost_80pct_power_with_the_stand_in_rule']}")
    out["seconds"] = round(time.time() - t0, 1)
    return out


CURVES = {"as it is": lambda x: x, "exp(x/2)": lambda x: np.exp(np.clip(x, -30, 30) / 2), "cube": lambda x: x ** 3,
          "signed square": lambda x: np.sign(x) * x ** 2, "signed root": lambda x: np.sign(x) * np.sqrt(np.abs(x)), "tanh(x/3)": lambda x: np.tanh(x / 3),
          "normal score of the rank": None, "minus log rank": None}
LABELS = "a word score fitted on these pairs' labels (not a legitimate stand-in)"


def per_pool(S, x, f):
    out = np.empty_like(x)
    for st, sz in zip(S.starts, S.sizes):
        out[st:st + sz] = f(x[st:st + sz])
    return out


def copy_base(test, kind, rng):
    """calibrate.py copy_base(): the word-only score a noisy or curved copy starts from."""
    S, X = test.S, test.X
    if kind.startswith("col:"):
        return X[:, int(kind[4:])]
    X = X[:, :30]                                             # the 30 word columns (the consensus cube is left out of the mixes)
    if kind == "mean of the 30 word columns":
        return S.standardise(X.mean(1))
    if kind == "random positive mix of the word columns":
        return S.standardise(X @ rng.exponential(size=X.shape[1]))
    if kind == "random signed mix of the word columns":
        return S.standardise(X @ rng.standard_normal(X.shape[1]))
    if kind == "mean of 5 random word columns":
        return S.standardise(X[:, rng.choice(28, 5, replace=False)].mean(1))
    if kind == "one random word column":
        return X[:, rng.integers(0, 28)]
    if kind == "the word model's own held-out score":
        return G["heldout"]
    if kind == LABELS:
        return S.standardise(test.X @ test.beta_all)
    raise KeyError(kind)


def copy_job(args):
    kind, curve, sigma, seed = args
    test = G["test"]
    S, rng = test.S, np.random.default_rng(seed)
    x = copy_base(test, kind, rng)
    if curve == "normal score of the rank":
        x = per_pool(S, x, lambda v: norm.ppf((rankdata(v) - 0.5) / len(v)))
    elif curve == "minus log rank":
        x = per_pool(S, x, lambda v: -np.log(rankdata(-v)))
    else:
        x = CURVES[curve](x)
    return kind, curve, sigma, test.slope_test(synth(test, S.standardise(x), sigma, 0.0, rng), n_flips=SIM_FLIPS, seed=seed)["p"]


def copies(an, reps):
    """Noisy and curved copies of the covariates in the units' place (calibrate.py copies()): they hold no information the
    covariates lack, so they should be rejected about 5% of the time or less. an must hold its pruned folds."""
    test, S = an.test, an.test.S
    eta = np.empty(S.n)
    for f in range(S.n_fam):                                  # the word model's own held-out score
        rows = np.isin(S.group, np.flatnonzero(S.fam == f))
        eta[rows] = test.X[rows] @ test.betas[f]
    G.update(test=test, heldout=S.standardise(eta))
    kinds = ["mean of the 30 word columns", "random positive mix of the word columns", "random signed mix of the word columns", "mean of 5 random word columns",
             "one random word column", "the word model's own held-out score", LABELS]
    per_column = max(reps // 4, 10)
    jobs, seed = [], 10 ** 6
    for kind in kinds:
        for curve in CURVES:
            for sigma in (0.0, 0.5, 1.0):
                n = (reps // 2 if "random" in kind else 1) if sigma == 0.0 else reps
                for _ in range(n):
                    seed += 1
                    jobs.append((kind, curve, sigma, seed))
    for j in range(30):
        for sigma in (0.5, 1.0, 2.0):
            for _ in range(per_column):
                seed += 1
                jobs.append((f"col:{j}", "as it is", sigma, seed))
    t0 = time.time()
    with get_context("fork").Pool(WORKERS) as pool:
        res = pool.map(copy_job, jobs, chunksize=16)
    out, legit = {"draws_in_all": len(jobs), "draws_per_setting": reps, "sign_flips_per_draw": SIM_FLIPS, "cells_in_the_slope": an.cells_label(PRUNE), "settings": {}}, []
    for kind in kinds:
        for curve in CURVES:
            row = {}
            for sigma in (0.0, 0.5, 1.0):
                ps = np.array([r[3] for r in res if r[0] == kind and r[1] == curve and r[2] == sigma])
                row[f"noise {sigma:g}"] = {"draws": len(ps), "rate_below_0.05": float((ps < T.ALPHA).mean()), "lowest_p": float(ps.min())}
                if kind != LABELS and len(ps) > 1:
                    legit.extend(ps.tolist())
            out["settings"][f"{kind} | {curve}"] = row
    rates = {}
    for j, name in enumerate(test.names[:30]):
        ps = np.array([r[3] for r in res if r[0] == f"col:{j}"])
        rates[name] = float((ps < T.ALPHA).mean())
        legit.extend(ps.tolist())
    worst = max(rates, key=rates.get)
    cells = {k: v2["rate_below_0.05"] for k, v in out["settings"].items() if not k.startswith(LABELS) for s, v2 in v.items() if v2["draws"] > 1}
    worst_cell = max(cells, key=cells.get)
    labels = np.array([r[3] for r in res if r[0] == LABELS and r[2] > 0])
    legit = np.array(legit)
    out["single_word_columns_plus_noise"] = {"rate_by_column": rates, "draws_per_column": 3 * per_column, "overall": float(np.mean(list(rates.values()))),
                                             "worst_column": worst, "worst_column_rate": rates[worst]}
    out["all_copies_that_do_not_use_the_labels"] = {"draws": len(legit), "below_0.05": int((legit < T.ALPHA).sum()), "rate_below_0.05": float((legit < T.ALPHA).mean()),
                                                    "worst_setting_curve": worst_cell, "worst_setting_rate": cells[worst_cell]}
    out["a_word_score_fitted_on_these_pairs_labels_not_legitimate"] = {"draws": len(labels), "rate_below_0.05": float((labels < T.ALPHA).mean())}
    out["seconds"] = round(time.time() - t0, 1)
    log(f"copies: {(legit < T.ALPHA).mean():.3f} of {len(legit)} below 0.05; single columns {out['single_word_columns_plus_noise']['overall']:.3f} (worst {worst} {rates[worst]:.2f}); "
        f"fitted on the labels {out['a_word_score_fitted_on_these_pairs_labels_not_legitimate']['rate_below_0.05']:.3f}")
    return out


def test_code_sha():
    """remake_test.py from its first import on: the designers' code without the docstring above it."""
    text = Path(T.__file__).read_text()
    return hashlib.sha256(text[text.index("\nimport re\n"):].encode()).hexdigest()


def code_sha():
    return hashlib.sha256((sha256(__file__) + test_code_sha()).encode()).hexdigest()


class Parts:
    """The stages of 'words', each written under v2/cache/second/remakes/words_parts/ as it finishes, so that an interrupted
    run resumes. A stage is reused only if it was made from the same pairs.json, films.jsonl, word measures and settings, by
    the same remakes.py and remake_test.py (--any-code, for development, lifts the last condition, and the output then names
    the stages concerned). --fresh starts again from nothing. The output lists which stages were reused."""

    def __init__(self, key, fresh, folder, any_code=False):
        self.key, self.code, self.any_code, self.folder = key, code_sha(), any_code, folder
        self.reused, self.other_code, self.made, self.seconds = [], [], set(), {}
        if fresh:
            shutil.rmtree(folder, ignore_errors=True)
            drop_panel_scratch(folder.parent)
        folder.mkdir(parents=True, exist_ok=True)

    def path(self, name):
        return self.folder / (re.sub(r"[^a-z0-9.]+", "_", name.lower()).strip("_") + ".json")

    def has(self, name):
        if not self.path(name).exists():
            return False
        d = json.load(open(self.path(name)))
        return d["key"] == self.key and (self.any_code or d.get("code") == self.code)

    def put(self, name, result, seconds, made=True):
        save({"key": self.key, "code": self.code, "seconds": round(seconds, 1), "result": result}, self.path(name), quiet=True)
        if made:
            self.made.add(name)

    def read(self, name):
        d = json.load(open(self.path(name)))
        self.seconds[name] = d["seconds"]
        if name not in self.made and name not in self.reused:
            self.reused.append(name)
        if d.get("code") != self.code and name not in self.other_code:
            self.other_code.append(name)
        return d["result"]

    def get(self, name, fn):
        if self.has(name):
            log(f"{name}: kept from an earlier, interrupted run")
        else:
            t0 = time.time()
            self.put(name, fn(), time.time() - t0)
        return self.read(name)                               # read back: what is assembled is what was written


def job_name(job):
    return f"analysis {job[0]} {job[1]}" + (f" lambda {job[2]:g}" if job[2] != T.LAMBDA else "")


def words(a):
    t0 = time.time()
    cache = Path(a.cache)
    C = Corpus(rows=a.rows)
    M, meta = word_measures(C, cache)
    parts = Parts({"pairs.json": C.sha["pairs.json"], "films.jsonl": C.sha["films.jsonl"], "word_measures.npz": meta["file_sha256"],
                   "draws": [a.power_draws, a.copies_draws], "prune": PRUNE}, a.fresh, cache / "words_parts", a.any_code)
    log(f"{len(C.pairs)} pairs, {C.N} films read; word measures {meta['shape']}")
    cnt = parts.get("counts", lambda: counts(C))
    an = Analysis(C, M, PRIMARY_SET, PRIMARY_FRAME)
    wm = word_model(an)
    log(f"primary set, frame set: {wm['all_entries']['entries']} entries, {wm['all_entries']['families']} families, {wm['cells']} cells; sources first "
        f"{wm['all_entries']['sources_ranked_first']}; effective pairs left open {wm['all_entries']['effective_pairs_left_open']:.2f}; fitted in {an.seconds_fit} s")
    jobs = [(s, f, T.LAMBDA) for s in ("firm_all", PRIMARY_SET, "firm_same_industry", "loose_all", "firm_other_cross_industry") for f in ("all", "set", "source_industry")
            if (s, f) in ANALYSES and (s, f) != (PRIMARY_SET, PRIMARY_FRAME)] + [(PRIMARY_SET, PRIMARY_FRAME, lam) for lam in LAMBDAS]       # the largest first
    assert {(s, f) for s, f, _ in jobs} | {(PRIMARY_SET, PRIMARY_FRAME)} == set(ANALYSES)
    todo = [j for j in jobs if not parts.has(job_name(j))]
    if todo or not parts.has("panel") or not parts.has("primary stand-ins") or not primary_panel_file(cache).exists():
        P, info, secs, made = standin_panel(C, cache, a.panel_chunks)
        parts.put("panel", {"stand_ins": len(P) - 1, "names": [k for k in P if k != NAMES_FREE], "standin_panel_primary.npz": save_primary_panel(C, P, an, cache),
                            "names_free_measure": info, "computed_in_parts": a.panel_chunks, "peak_memory_gb": peak_gb()}, secs, made)
        prim = parts.get("primary stand-ins", lambda: stage_primary_standins(an, P, info))
        if todo:
            G.update(C=C, M=M, P=P)
            with get_context("fork").Pool(min(WORKERS, len(todo))) as pool:
                for job, res in pool.imap_unordered(stage_analysis, todo, chunksize=1):
                    parts.put(job_name(job), res, res["seconds"])
            G.clear()
        del P
    panel, prim = parts.read("panel"), parts.read("primary stand-ins")
    others = {j: parts.read(job_name(j)) for j in jobs}
    control = parts.get("withheld vocabulary", lambda: withheld_vocabulary(C, an))
    if not (parts.has("power") and parts.has("copies")):
        an.keep_folds()
        Pp = load_primary_panel(C, an, panel["standin_panel_primary.npz"], cache)
    z_max = prim["registered"]["summary"]["z_max"]
    pw = parts.get("power", lambda: power(an, Pp, z_max, a.power_draws))
    cp = parts.get("copies", lambda: copies(an, a.copies_draws))
    an.drop_folds()
    her = parts.get("heredity by words", lambda: {label: heredity_report(an, an.per_entry(M[m])) for label, m in HEREDITY_WORDS.items()})

    analyses = {an.label(): {"word_model": wm, "standins": prim["registered"], "heredity_by_words": her}}
    for (s, f, lam), res in others.items():
        if lam == T.LAMBDA:
            analyses[f"{s} | {f}"] = {k: res[k] for k in ("word_model", "standins", "heredity_by_words")}
    e, summ = wm["all_entries"], prim["registered"]["summary"]
    twin = next(v for k, v in pw["bases"].items() if k.startswith("the word twin") and k.endswith("noise 1"))
    noise = next(v for k, v in pw["bases"].items() if k.startswith("nothing"))
    out = {
        "what": "The remakes corpus with no model (PLAN.md Sections 9 and 15): counts, what the word model settles, and the calibration of the decision test on the "
                "final list. Made by `remakes.py words`. 'run' reads the stand-ins' results from this file and refuses if the inputs below have changed.",
        "git_head": git_head(),
        "sha256": {**C.sha, "word_measures.npz": meta["file_sha256"], "word_measures": meta["sha256"], "standin_panel_primary.npz": panel["standin_panel_primary.npz"],
                   "remakes.py": sha256(__file__), "remake_test.py": sha256(T.__file__), "remake_test.py_from_its_first_import_on": test_code_sha()},
        "settings": dict(settings(), power_draws_per_boost_and_base=a.power_draws, copies_draws_per_setting=a.copies_draws,
                         withheld_vocabulary={"seeds": list(SPLIT_SEEDS), "measures": list(SPLIT_MEASURES)}, standin_panel={"stand_ins": panel["stand_ins"], "names": panel["names"]}),
        "counts": cnt,
        "analyses": analyses,
        "primary_calibration": {
            "set": PRIMARY_SET, "frame": PRIMARY_FRAME, "word_model": wm, "standins": prim["registered"],
            "names_free_measure_not_counted_among_the_78": prim["names_free_measure_not_counted_among_the_78"],
            "pruning_check": prim["pruning_check"], "no_hub_correction": prim["no_hub_correction"],
            "standins_by_subset_of_the_entries_descriptive": prim["by_subset_of_the_entries"],
            "other_penalties": {f"{lam:g}": {k: res[k] for k in ("word_model", "standins")} for (s, f, lam), res in others.items() if lam != T.LAMBDA},
            "withheld_vocabulary_positive_control": control, "planted_boost_power": pw, "noisy_and_curved_copies": cp},
        "claim_conditions_that_words_decide": {
            "1 evaluable: the word model leaves at least 5 effective pairs open": {"effective_pairs_left_open": e["effective_pairs_left_open"], "holds": bool(e["evaluable"])},
            "2 calibrated: at most 10% of the 78 stand-ins have p below 0.05": {"below_0.05": summ["below_0.05"], "share": summ["share_below_0.05"], "holds": bool(summ["at_most_10pct_below_0.05"])},
            "the positive control of the test (withheld vocabulary) passed": bool(control["passed"]),
            "stand_ins_largest_Z_the_units_must_exceed": {"Z": summ["z_max"], "measure": summ["z_max_measure"]},
            "smallest_boost_detected_with_80pct_power": {"word twin plus noise": twin["smallest_boost_80pct_power_with_the_stand_in_rule"],
                                                         "pure noise": noise["smallest_boost_80pct_power_with_the_stand_in_rule"], "draws_per_boost": a.power_draws,
                                                         "unit": "pool standard deviations of the whole measure, added to the true source"}},
        "run": {"seconds": round(time.time() - t0 + sum(parts.seconds[n] for n in parts.reused), 1), "seconds_by_stage": parts.seconds,
                "seconds_word_measures": meta["seconds"], "stages_kept_from_an_interrupted_run": parts.reused,
                "stages_computed_by_another_version_of_the_code": parts.other_code,
                "peak_memory_gb": peak_gb(), "peak_memory_gb_largest_worker": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20,
                "workers": WORKERS, "threads_per_process": os.environ.get("OMP_NUM_THREADS")}}
    save(out, a.out or OUT / WORDS_FILE)
    if not a.keep_scratch:
        drop_panel_scratch(cache)
    log(f"primary set: E {e['effective_pairs_left_open']:.2f}; stand-ins below 0.05: {summ['below_0.05']} of {summ['n']} (largest Z {summ['z_max']:.2f}, {summ['z_max_measure']}); "
        f"positive control {'passed' if control['passed'] else 'FAILED'}; 80% power at {twin['smallest_boost_80pct_power_with_the_stand_in_rule']} SD (twin), "
        f"{noise['smallest_boost_80pct_power_with_the_stand_in_rule']} SD (noise)")
    return out


# ------------------------------------------------------------------ 'run': one model's unit measure through everything
def read_counts(root, C):
    """The GPU machine's counts as rates. Returns {plane: [remakes, films] float64, 0 for a film the model did not read}, the
    films it read, and what the files say of themselves."""
    meta = json.load(open(root / "meta.json"))
    ids = np.asarray(meta["ids"], dtype=np.int64)
    assert len(np.unique(ids)) == len(ids) and ids.min() >= 0 and ids.max() < C.N, "meta.json ids are not film ids of films.jsonl"
    if C.rows_json is None or C.rows_json["rows"] != C.remakes.tolist():
        raise SystemExit("refused: rows.json does not list the remakes of the current pairs.json (the rows of the counts). It is written by `gpu.py texts`: "
                         "rebuild it from the final list, and count again if the GPU machine used another list.")
    read = np.zeros(C.N, bool)
    read[ids] = True
    assert read[C.remakes].all(), "the model did not read every remake"
    rates, info = {}, {"films_read_by_the_model": int(len(ids)), "films_read_in_all": C.N, "planes": {}}
    for plane in PLANES:
        f = root / "counts" / f"{plane}.npz"
        if not f.exists():
            continue
        Z = np.load(f)
        shared, n_opt = Z["shared"], Z["n_optional"]
        assert shared.shape == (len(C.remakes), len(ids)) and n_opt.shape == (len(ids),), f"{f}: the counts are not [remakes, films read]"
        if "rows" in Z.files:
            assert np.array_equal(ids[Z["rows"]], C.remakes), f"{f}: its rows are not the remakes of rows.json"
        rate = np.zeros((len(C.remakes), C.N))
        rate[:, ids] = shared / np.maximum(n_opt, 1)[None, :]
        rates[plane] = rate
        info["planes"][plane] = {"sha256": sha256(f), **{k: Z[k].tolist() for k in ("band", "n_units_optional", "n_units_present_anywhere", "n_films") if k in Z.files},
                                 "films_with_no_optional_unit": int((n_opt == 0).sum()), "median_optional_units_per_film": float(np.median(n_opt))}
    info["meta"] = {k: meta.get(k) for k in ("tag", "provenance", "n_layers", "width", "n_texts", "scale_sha256", "texts_sha256", "synthetic") if k in meta}
    return rates, read, info


def gain_brief(g):
    return {"gain": g["gain"], "gain_sign_flip_p": g["p"], "unit_weight_mean": g["unit_weight_mean"], "unit_weight_min": g["unit_weight_min"],
            "sources_first_words": g["sources_first_words"], "sources_first_with_the_measure": g["sources_first_with_units"], "log_rank_gain": g["log_rank_gain"]}


CONDITIONS = ("1 evaluable: the word model leaves at least 5 effective pairs open", "2 calibrated: at most 10% of the 78 stand-ins have p below 0.05",
              "3 the p of the slope is below 0.05", "4 the gain of the refitted model is positive", "5 Z is above the Z of every stand-in",
              "6 p is below 0.05 in at least 4 of the 8 planes", "7 not confined to recognised pairs")


def test_measure(an, E, st, form="rate_matrix"):
    """One measure E [entries, films] in the units' place: the slope (remake_test's unit column and contributions), the gain of
    the refitted model, and conditions 1 to 5 against the stand-ins' results st ({"rows": {name: {"Z", "p"}}, "summary"}), which
    'words' computed on the same entries, frame, penalty and form. Returns the report, the unit column, the entries'
    contributions and the family sums."""
    u = an.test.unit_column(E, form)
    Ur = an.contributions(u)
    slope, d = an.summary(Ur)
    g = an.test.gain(u)
    top = max(st["rows"], key=lambda k: st["rows"][k]["Z"])
    z_max, share = st["rows"][top]["Z"], float(np.mean([v["p"] < T.ALPHA for v in st["rows"].values()]))
    cond = dict(zip(CONDITIONS[:5], (bool(an.test.evaluable), bool(share <= T.MAX_STANDIN_SHARE), bool(slope["p"] < T.ALPHA), bool(g["gain"] > 0), bool(slope["Z"] > z_max))))
    rep = {"cells_in_the_slope": an.cells_label(), "form": {"rate_matrix": "hub-corrected, normal scores within the pool", "raw_matrix": "no hub correction, normal scores within the pool"}[form],
           "lambda": an.lam, "slope": slope, "gain": gain_brief(g), "standins": {"n": len(st["rows"]), "z_max": z_max, "z_max_measure": top, "share_below_0.05": share},
           "conditions_1_to_5": cond, "conditions_3_4_5_hold": bool(all(list(cond.values())[2:])), "conditions_1_to_5_hold": bool(all(cond.values()))}
    return rep, u, Ur, d


def recognised_entries(an, rec):
    """An entry is recognised if the model names the remake's source (r2) or identifies both its films (r1)."""
    r1, r2 = rec["r1"], rec["r2"]
    return np.array([bool(r2[p["remake"]]["source_named"]) or (bool(r1[p["remake"]]["identified"]) and bool(r1[p["source"]]["identified"])) for p in an.entries])


def recall_rule(an, Ur, known):
    """Condition 7, from the fit's own contributions with nothing refitted: over the entries the model does not recognise,
    E_n = the sum of (1 - held-out probability of the true source) and p_n = the one-sided sign-flip p of their family sums."""
    s, _ = an.summary(Ur, ~known)
    E_n, p_n = s["effective_pairs_left_open"], s["p"]
    reading = "not confined to recognised pairs" if (E_n >= T.MIN_UNSOLVED and p_n < T.ALPHA) else "cannot be separated from recall" if E_n < T.MIN_UNSOLVED else "recall"
    return {"entries_not_recognised": s["entries"], "entries_recognised": int(known.sum()), "E_n": E_n, "p_n": p_n, "T_n": s["T"], "Z_n": s["Z"],
            "E_n_needed": T.MIN_UNSOLVED, "holds": reading == "not confined to recognised pairs", "reading": reading}


def outcome_of(c, control_passed, recall_reading):
    """The registered wording. c: the seven conditions in order. The first that fails decides."""
    if not c[0] or not control_passed:
        return "not evaluable"
    if not c[1]:
        return "not calibrated"
    if not (c[2] and c[3] and c[4]):
        return "no information beyond these word measures was detected"
    if not c[5]:
        return "detected at the primary setting only"
    if not c[6]:
        assert recall_reading in ("recall", "cannot be separated from recall")
        return recall_reading
    return "claim"


def subsets(an, Ur, d, known):
    """Sums of the fit's contributions over subsets of the entries, nothing refitted, each with its own sign-flip p and open mass."""
    S, ps = an.test.S, an.test.p_source
    out = {"all_entries": an.summary(Ur)[0], "not_copied": an.summary(Ur, ~an.copied)[0], "copied": an.summary(Ur, an.copied)[0],
           "copy_share_under_1pct": an.summary(Ur, an.copy_share < LOW_COPY)[0], "recognised": an.summary(Ur, known)[0], "not_recognised": an.summary(Ur, ~known)[0],
           "contributions_set_to_zero_where_the_held_out_source_probability_is_0.99_or_more": an.summary(Ur, ps < SOLVED)[0],
           "each_of_the_largest_families_left_out": {}}
    for f in np.argsort(-np.abs(d), kind="stable")[:N_LARGEST]:
        out["each_of_the_largest_families_left_out"][str(S.fam_names[f])] = dict(an.summary(Ur, S.fam != f)[0], family_sum_left_out=float(d[f]))
    return out


def family_table(an, Ur, d):
    S, ps = an.test.S, an.test.p_source
    order = np.argsort(-np.abs(d), kind="stable")
    ss = float((d ** 2).sum())
    return {"families": int(S.n_fam), "families_with_a_non_zero_sum": int((d != 0).sum()),
            "effective_number_of_families": float(np.abs(d).sum() ** 2 / ss) if ss > 0 else 0.0,
            "share_of_the_sum_of_squares_in_the_five_largest": float((d[order[:5]] ** 2).sum() / ss) if ss > 0 else 0.0,
            "rows": [{"family": S.fam_names[f], "D": float(d[f]), "entries": int((S.fam == f).sum()), "open_mass": float((1 - ps)[S.fam == f].sum()),
                      "remakes": [p["remake"] for p, x in zip(an.entries, S.fam == f) if x]} for f in order]}


def entry_table(an, Ur, H, known):
    S, t = an.test.S, an.test
    return [{"remake": p["remake"], "source": p["source"], "family": p["family"], "copied": bool(p["copied"]), "copy_share": p["share_of_five_word_sequences_also_in_source"],
             "recognised": bool(known[i]), "pool": int(S.sizes[i]), "words_probability_of_the_source": float(t.p_source[i]), "words_rank_of_the_source": float(t.rank0[i]),
             "unit_rank_of_the_source": int(1 + (H[i, an.pools[i]] > H[i, an.src[i]]).sum()), "contribution": float(Ur[i])} for i, p in enumerate(an.entries)]


def embedding_cosine(C, path):
    E = np.load(path).astype(np.float32)
    assert E.shape[0] == C.N, "the embedding does not have one row per film of films.jsonl"
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return E[C.remakes] @ E.T


def same_word_model(wm, Wa, label):
    """The word model fitted here must be the one 'words' calibrated the stand-ins on."""
    a, b = wm["all_entries"], Wa["word_model"]["all_entries"]
    if a["entries"] != b["entries"] or abs(a["held_out_log_likelihood"] - b["held_out_log_likelihood"]) > 1e-6 or abs(a["effective_pairs_left_open"] - b["effective_pairs_left_open"]) > 1e-6:
        raise SystemExit(f"refused: the word model fitted now for {label} is not the one {WORDS_FILE} describes (held-out log-likelihood "
                         f"{a['held_out_log_likelihood']!r} against {b['held_out_log_likelihood']!r})")


def run_analysis(job):
    """One (set, frame) for one model. Runs in a forked worker; reads its inputs from G."""
    name, frame = job
    C, M, W, rates, rec, cos, cache = (G[k] for k in ("C", "M", "W", "rates", "rec", "cos", "cache"))
    t0 = time.time()
    an = Analysis(C, M, name, frame)
    Wa = W["analyses"][an.label()]
    wm = word_model(an)
    same_word_model(wm, Wa, an.label())
    known = recognised_entries(an, rec)
    E = an.per_entry(rates[PRIMARY_PLANE])
    res, u, Ur, d = test_measure(an, E, Wa["standins"])
    out = {"set": name, "frame": frame, "plane": PRIMARY_PLANE, "entries": len(an.entries), "families": int(an.test.S.n_fam), "cells": int(an.test.S.n),
           "words_only": wm, **res, "recall_rule": recall_rule(an, Ur, known), "subsets": subsets(an, Ur, d, known), "family_sums": family_table(an, Ur, d),
           "heredity": heredity_report(an, E)}
    if (name, frame) == (PRIMARY_SET, PRIMARY_FRAME):
        # condition 6: the eight planes
        planes = [p for p in PLANES if p in rates]
        U = np.column_stack([u if p == PRIMARY_PLANE else an.test.unit_column(an.per_entry(rates[p]), "rate_matrix") for p in planes])
        Up = an.contributions(U)
        per_plane = {p: an.summary(Up[:, j])[0] for j, p in enumerate(planes)}
        assert per_plane[PRIMARY_PLANE]["p"] == res["slope"]["p"] and per_plane[PRIMARY_PLANE]["T"] == res["slope"]["T"]
        n_pass = sum(v["p"] < T.ALPHA for v in per_plane.values())
        out["planes"] = {"available": planes, "missing": [p for p in PLANES if p not in rates], "below_0.05": int(n_pass), "needed": MIN_PLANES, "rows": per_plane}
        cond = dict(res["conditions_1_to_5"])
        cond[CONDITIONS[5]], cond[CONDITIONS[6]] = bool(n_pass >= MIN_PLANES), bool(out["recall_rule"]["holds"])
        control = bool(W["primary_calibration"]["withheld_vocabulary_positive_control"]["passed"])
        out["conditions"] = cond
        out["withheld_vocabulary_positive_control_passed_on_the_final_list"] = control
        out["outcome"] = outcome_of(list(cond.values()), control, out["recall_rule"]["reading"])
        out["entries_table"] = entry_table(an, Ur, an.hub(E), known)
        # reported beside the decision
        cal, sens = W["primary_calibration"], {}
        sens["slope_with_pruned_cells"] = dict(an.summary(an.contributions(u, PRUNE))[0], cells_in_the_slope=an.cells_label(PRUNE))
        sens["no_hub_correction"] = test_measure(an, E, cal["no_hub_correction"], form="raw_matrix")[0]
        for lam in LAMBDAS:
            al = Analysis(C, M, name, frame, lam=lam)
            wl = cal["other_penalties"][f"{lam:g}"]
            same_word_model(word_model(al), wl, f"{an.label()} at penalty {lam:g}")
            sens[f"lambda_{lam:g}"] = dict(test_measure(al, E, wl["standins"])[0], words_only=word_model(al)["all_entries"])
        out["beside_the_decision"] = sens
        if cos is not None:
            Ee = an.per_entry(cos)
            r = test_measure(an, Ee, Wa["standins"])[0]
            emb = {"in_the_units_place": dict(r, passes=r["conditions_3_4_5_hold"], heredity=heredity_report(an, Ee))}
            a33 = Analysis(C, M, name, frame, columns=list(T.WORD_COLUMNS) + [("embedding", "z"), ("embedding", "hz")], extra={"embedding": Ee})
            st = standins(a33, load_primary_panel(C, an, W["sha256"]["standin_panel_primary.npz"], cache), by_entry=True)
            st33 = {"summary": standin_summary(st), "rows": {k: v for k, v in st.items() if k != NAMES_FREE}}
            emb["units_with_the_embedding_among_the_word_columns"] = dict(test_measure(a33, E, st33)[0], columns=len(a33.test.names),
                                                                          words_and_embedding_only=word_model(a33)["all_entries"], standins_summary=st33["summary"])
            out["embedding"] = emb
    out["seconds"], out["peak_memory_gb"] = round(time.time() - t0, 1), peak_gb()
    log(f"{a_label(job)}: slope T {res['slope']['T']:.3f}, Z {res['slope']['Z']:.2f}, p {res['slope']['p']:.5f}; gain {res['gain']['gain']:.3f}; "
        f"stand-ins' largest Z {res['standins']['z_max']:.2f}; {out['seconds']} s")
    return job, out


def a_label(job):
    return f"{job[0]} | {job[1]}"


def run(a):
    t0 = time.time()
    cache, root = Path(a.cache), Path(a.acts_root) / a.tag
    wfile, rfile = Path(a.words or OUT / WORDS_FILE), Path(a.recognition or OUT / f"remakes_recognition_{a.tag}.json")
    if not wfile.exists():
        raise SystemExit(f"refused: {wfile} is not there. Run `remakes.py words` first: the stand-ins and the calibration are fixed before any unit is read.")
    if not rfile.exists():
        raise SystemExit(f"refused: {rfile} is not there. The recognition probes come first (gpu.py probes {a.tag} remakes_r1 remakes_r2; probes.py score {a.tag}).")
    C = Corpus(rows=a.rows)
    W = json.load(open(wfile))
    for k in ("pairs.json", "films.jsonl"):
        if W["sha256"][k] != C.sha[k]:
            raise SystemExit(f"refused: {wfile} was made from another {k} than the one here. Run `remakes.py words` again.")
    M, meta = word_measures(C, cache, must_exist=True)
    if meta["sha256"] != W["sha256"]["word_measures"]:
        raise SystemExit(f"refused: the saved word measures are not the ones {wfile} was made from. Run `remakes.py words` again.")
    rec = json.load(open(rfile))
    pages = {x for p in C.pairs for x in (p["remake"], p["source"])}
    lack1, lack2 = sorted(pages - set(rec.get("r1", {}))), sorted({p["remake"] for p in C.pairs} - set(rec.get("r2", {})))
    if lack1 or lack2:
        raise SystemExit(f"refused: {rfile} does not cover the current pair list ({len(lack1)} films have no R1 answer, {len(lack2)} remakes have no R2 answer).")
    rates, read, info = read_counts(root, C)
    if PRIMARY_PLANE not in rates:
        raise SystemExit(f"refused: no counts for the primary plane {PRIMARY_PLANE} under {root}")
    cos = embedding_cosine(C, a.embedding) if Path(a.embedding).exists() else None
    wanted = [(s, f) for s in ("firm_all", PRIMARY_SET, "firm_same_industry", "loose_all", "firm_other_cross_industry") for f in ("set", "all", "source_industry") if (s, f) in ANALYSES]
    if a.analyses == "primary":
        wanted = [j for j in wanted if j[0] == PRIMARY_SET]
    jobs, left = [], []
    for s, f in wanted:                                     # a model that read only some films gets the analyses all of whose pools it read
        need = np.unique(np.concatenate([C.pool(p, s, f) for p in C.entries(s)]))
        if read[need].all():
            jobs.append((s, f))
        else:
            left.append({"set": s, "frame": f, "why": f"the model did not read {int((~read[need]).sum())} of the {len(need)} films in its pools"})
    if (PRIMARY_SET, PRIMARY_FRAME) not in jobs:
        raise SystemExit("refused: the model did not read every film of the primary pools")
    log(f"{a.tag}: {info['films_read_by_the_model']} films read, planes {list(rates)}; analyses {[a_label(j) for j in jobs]}; left out {[a_label((x['set'], x['frame'])) for x in left]}")
    G.update(C=C, M=M, W=W, rates=rates, rec=rec, cos=cos, cache=cache)
    blocks = {}
    if a.workers > 1 and len(jobs) > 1:
        with get_context("fork").Pool(min(a.workers, len(jobs))) as pool:
            for job, res in pool.imap_unordered(run_analysis, jobs, chunksize=1):
                blocks[a_label(job)] = res
    else:
        for job in jobs:
            blocks[a_label(job)] = run_analysis(job)[1]
    G.clear()
    prim = blocks[a_label((PRIMARY_SET, PRIMARY_FRAME))]
    her, cal = prim["heredity"]["hub_corrected"], W["primary_calibration"]
    control = her.get("positive_control_copied_entries")
    out = {
        "what": "The remakes tests for one model (PLAN.md Section 9). Made by `remakes.py run`. Every number unrounded.",
        "model": a.tag, "git_head": git_head(),
        "sha256": {**C.sha, "word_measures.npz": meta["file_sha256"], WORDS_FILE: sha256(wfile), rfile.name: sha256(rfile), "remakes.py": sha256(__file__),
                   "remake_test.py": sha256(T.__file__), "remake_test.py_from_its_first_import_on": test_code_sha(),
                   "embedding.npy": sha256(a.embedding) if cos is not None else None},
        "settings": settings(), "counts_files": info,
        "analyses_run": list(blocks), "analyses_left_out": left, "analyses_not_asked_for": [a_label(j) for j in ANALYSES if j not in wanted],
        "decision": {"set": PRIMARY_SET, "frame": PRIMARY_FRAME, "plane": PRIMARY_PLANE, "conditions": prim["conditions"], "outcome": prim["outcome"],
                     "slope": prim["slope"], "gain": prim["gain"]["gain"], "standins_largest_Z": prim["standins"]["z_max"], "planes_below_0.05": prim["planes"]["below_0.05"],
                     "planes_available": len(prim["planes"]["available"]), "recall_rule": prim["recall_rule"],
                     "smallest_boost_detected_with_80pct_power_on_this_list": W["claim_conditions_that_words_decide"]["smallest_boost_detected_with_80pct_power"]},
        "heredity": {"what": "confirmatory: the primary entries that are not copied, frame set, hub-corrected rate", "not_copied": her["not_copied"], "all_entries": her["all_entries"],
                     "positive_control_copied_entries": control, "positive_control": None if control is None else control["reading"],
                     "by_words": W["analyses"][a_label((PRIMARY_SET, PRIMARY_FRAME))]["heredity_by_words"]},
        "embedding": prim.get("embedding", "v2/acts_second/remakes/embedding.npy is not there: not run"),
        "analyses": blocks,
        "seconds": round(time.time() - t0, 1), "peak_memory_gb": peak_gb(), "peak_memory_gb_largest_worker": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20,
        "workers": a.workers}
    save(out, a.out or OUT / f"remakes_{a.tag}.json")
    log(f"{a.tag}: outcome: {prim['outcome']}; heredity S {her['not_copied']['S']:.5f}, p {her['not_copied']['p']}; positive control: {out['heredity']['positive_control']}")
    return out


# ------------------------------------------------------------------ 'verify': the GPU machine's counts, recounted here from presence bits
def verify_counts(root, C, chunk=65536):
    """For every remake row and every pair-film column of the primary plane: the optional units the two films share and the
    optional units of the column's film, counted here from the fetched presence bits of the pair films, against the counts file.
    A unit is optional when it is present in at least 2 and at most floor(0.5 * films read by the model) films."""
    meta = json.load(open(root / "meta.json"))
    ids = np.asarray(meta["ids"], dtype=np.int64)
    rj = C.rows_json
    if rj is None or rj["rows"] != C.remakes.tolist() or rj["pair_films"] != C.pair_films:
        raise SystemExit("refused: rows.json does not list the remakes and pair films of the current pairs.json. It is written by `gpu.py texts`.")
    col = {int(f): j for j, f in enumerate(ids)}
    pair_films, rows = rj["pair_films"], rj["rows"]
    bits = np.load(root / f"pair_films_{PRIMARY_PLANE}.bits.npy", mmap_mode="r")
    uc = np.load(root / f"{PRIMARY_PLANE}.unit_counts.npy")
    Z = np.load(root / "counts" / f"{PRIMARY_PLANE}.npz")
    assert bits.shape == (len(pair_films), len(uc) // 8) and len(uc) % 8 == 0, "the bits are not [pair films, units / 8]"
    lo, hi = 2, int(np.floor(0.5 * len(ids)))
    optional = (uc >= lo) & (uc <= hi)
    at = {f: k for k, f in enumerate(pair_films)}
    r_at = np.array([at[r] for r in rows])
    shared, n_opt = np.zeros((len(rows), len(pair_films)), np.int64), np.zeros(len(pair_films), np.int64)
    for b0 in range(0, bits.shape[1], chunk // 8):
        o = optional[b0 * 8:(b0 + chunk // 8) * 8]
        if not o.any():
            continue
        Q = np.unpackbits(np.ascontiguousarray(bits[:, b0:b0 + chunk // 8]), axis=1)[:, o].astype(np.float32)      # sums of 0/1 products stay exact far below 2**24
        n_opt += Q.sum(1).astype(np.int64)
        shared += np.rint(Q[r_at] @ Q.T).astype(np.int64)
    cols = np.array([col[f] for f in pair_films])
    there_shared, there_opt = Z["shared"][:, cols].astype(np.int64), Z["n_optional"][cols].astype(np.int64)
    d_shared, d_opt = int((there_shared != shared).sum()), int((there_opt != n_opt).sum())
    band_there = Z["band"].tolist() if "band" in Z.files else None
    n_units_there = int(Z["n_units_optional"]) if "n_units_optional" in Z.files else None
    exact = d_shared == 0 and d_opt == 0 and band_there in (None, [lo, hi]) and n_units_there in (None, int(optional.sum()))
    return {"what": "The GPU machine's counts for the primary plane, recounted from the presence bits of the films of the pairs (remakes.py verify).",
            "plane": PRIMARY_PLANE, "films_read_by_the_model": int(len(ids)), "band": [lo, hi], "band_in_the_counts_file": band_there,
            "units": int(len(uc)), "optional_units": int(optional.sum()), "optional_units_in_the_counts_file": n_units_there,
            "remake_rows": len(rows), "pair_film_columns": len(pair_films), "cells_compared": int(shared.size),
            "shared_counts_that_differ": d_shared, "n_optional_counts_that_differ": d_opt, "largest_shared_count": int(shared.max()), "exact": bool(exact),
            "sha256": {f"counts/{PRIMARY_PLANE}.npz": sha256(root / "counts" / f"{PRIMARY_PLANE}.npz"), f"pair_films_{PRIMARY_PLANE}.bits.npy": sha256(root / f"pair_films_{PRIMARY_PLANE}.bits.npy"),
                       f"{PRIMARY_PLANE}.unit_counts.npy": sha256(root / f"{PRIMARY_PLANE}.unit_counts.npy"), **C.sha}}


def verify(a):
    C = Corpus(rows=a.rows)
    res = dict(verify_counts(Path(a.acts_root) / a.tag, C), model=a.tag, git_head=git_head())
    save(res, a.out or OUT / f"remakes_verify_{a.tag}.json")
    log(f"{a.tag}: {res['cells_compared']:,} shared counts compared, {res['shared_counts_that_differ']} differ; {res['pair_film_columns']} n_optional counts, "
        f"{res['n_optional_counts_that_differ']} differ; exact: {res['exact']}")
    if not res["exact"]:
        raise SystemExit(1)
    return res


# ------------------------------------------------------------------ 'synthetic': 'verify' and 'run' end to end on made-up units
PLANTED = ("top1_20", "top1_30", "top1_40", "top3_20", "top3_30")       # where the made-up heredity is planted: 5 of the 8 planes
KEEP = {"top1_20": 1.0, "top1_30": 1.0, "top1_40": 0.8, "top1_50": 0.6, "top3_20": 0.95, "top3_30": 0.85, "top3_40": 0.7, "top3_50": 0.5}


def make_synthetic(C, root, n2u, signal, small=False, planes=PLANES, seed=1):
    """Made-up presence of n2u units in every film read, and from it the files a GPU machine returns: the counts of each plane
    (v2/gpu/copresence.py's arithmetic, written again in numpy), the primary plane's unit counts and the bits of the pair
    films, meta.json. No model and no text measure beyond which words a film contains. Three kinds of unit:
      word units    each follows one common word (present where the word is, 20% dropped, a few added): the word twin of a unit;
      noise units   present at random at a rate of their own (16 of them in 70% of films and 16 in none: not optional);
      story units   a dozen or so per family, present at random in 0.3% of films, and with `signal` also in three quarters of
                    the films of their family, in the planes of PLANTED: the planted heredity that no word measure can see.
    The other planes are thinned copies. small: only the Indian industries' films and the pair films are read, as for the
    70B model."""
    from sklearn.feature_extraction.text import CountVectorizer
    assert n2u % 8 == 0
    rng = np.random.default_rng(seed)
    ids = np.flatnonzero(C.indian | C.in_pair) if small else np.arange(C.N)
    F, at = len(ids), {int(f): k for k, f in enumerate(ids)}
    n_word, n_noise = int(0.3 * n2u), int(0.1 * n2u)
    n_story = n2u - n_word - n_noise
    B = CountVectorizer(binary=True, stop_words="english", min_df=2).fit_transform(C.texts).tocsc()
    df = np.asarray(B.sum(0)).ravel()
    ok = np.flatnonzero((df >= 2) & (df <= C.N / 2))
    words = B[:, ok[np.argsort(-df[ok], kind="stable")[:n_word]]].toarray().astype(bool)[ids]
    words = (words & (rng.random(words.shape, dtype=np.float32) < 0.8)) | (rng.random(words.shape, dtype=np.float32) < 0.005)
    rate = rng.uniform(0.002, 0.05, n_noise).astype(np.float32)
    rate[:16], rate[16:32] = 0.7, 0.0                                   # 16 units in most films and 16 in none: outside the optional band
    noise = rng.random((F, n_noise), dtype=np.float32) < rate[None, :]
    fams = np.unique(C.fam[C.fam >= 0])
    unit_family = fams[np.arange(n_story) % len(fams)]
    story_null = rng.random((F, n_story), dtype=np.float32) < 0.003
    story_sig = story_null | ((C.fam[ids][:, None] == unit_family[None, :]) & (rng.random((F, n_story), dtype=np.float32) < 0.75))
    rows, pair = np.array([at[int(r)] for r in C.remakes]), np.array([at[f] for f in C.pair_films])
    (root / "counts").mkdir(parents=True, exist_ok=True)
    for plane in planes:
        Pm = np.concatenate([words, noise, story_sig if (signal and plane in PLANTED) else story_null], axis=1)
        if plane != PRIMARY_PLANE:
            Pm &= np.random.default_rng([seed, PLANES.index(plane)]).random(Pm.shape, dtype=np.float32) < KEEP[plane]
        nb = Pm.sum(0)
        lo, hi = 2, int(np.floor(0.5 * F))
        opt = (nb >= lo) & (nb <= hi)
        Q = Pm[:, opt].astype(np.float32)
        np.savez(root / "counts" / f"{plane}.npz", shared=np.rint(Q[rows] @ Q.T).astype(np.int32), n_optional=Q.sum(1).astype(np.int32), n_present=Pm.sum(1).astype(np.int32),
                 rows=rows, band=np.array([lo, hi]), n_units_optional=int(opt.sum()), n_units_present_anywhere=int((nb > 0).sum()), n_films=F)
        if plane == PRIMARY_PLANE:
            np.save(root / f"{plane}.unit_counts.npy", np.minimum(nb, 65535).astype(np.uint16))
            np.save(root / f"pair_films_{plane}.bits.npy", np.packbits(Pm[pair], axis=1))
        del Pm, Q
    json.dump({"tag": root.name, "ids": ids.tolist(), "n_texts": F, "n_layers": 1, "width": n2u // 2, "planes": list(planes), "synthetic": True}, open(root / "meta.json", "w"))
    r1 = {C.page[f]: {"identified": bool(x)} for f, x in zip(C.pair_films, rng.random(len(C.pair_films)) < 0.35)}
    r2 = {C.page[r]: {"source_named": bool(x)} for r, x in zip(C.remakes, rng.random(len(C.remakes)) < 0.15)}
    json.dump({"model": root.name, "synthetic": True, "r1": r1, "r2": r2}, open(root / "recognition.json", "w"))


def self_checks():
    """The wording rule on every pattern of the seven conditions, and the recall rule's three readings on made-up numbers."""
    import itertools
    seen = Counter()
    for c in itertools.product((True, False), repeat=7):
        for control in (True, False):
            for reading in ("recall", "cannot be separated from recall"):
                o = outcome_of(c, control, reading if not c[6] else "not confined to recognised pairs")
                assert o in OUTCOMES
                assert (o == "claim") == (all(c) and control)
                assert (o == "not evaluable") == (not c[0] or not control)
                if c[0] and control and c[1] and all(c[2:5]) and c[5]:
                    assert o == ("claim" if c[6] else reading)
                seen[o] += 1
    assert set(seen) == set(OUTCOMES)
    return {"patterns": int(sum(seen.values())), "every_wording_reached": True}


def synthetic(a):
    """'verify' and 'run' on made-up counts, bits and recognition, three times: with a planted heredity (every set and frame, the
    made-up embedding too), with none (the primary set), and for a model that read only the Indian industries' films and one
    plane. Everything made up is under v2/cache/second/remakes/synthetic/ and is deleted at the end; the reports stay in
    v2/cache/second/remakes/synthetic_results/."""
    cache = Path(a.cache)
    if not Path(a.words or OUT / WORDS_FILE).exists():
        raise SystemExit("refused: the dry run uses the real stand-ins: run `remakes.py words` first")
    base, keep = cache / "synthetic", cache / "synthetic_results"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    keep.mkdir(parents=True, exist_ok=True)
    C = Corpus(rows=None)
    json.dump({"rows": C.remakes.tolist(), "pair_films": C.pair_films}, open(base / "rows.json", "w"))       # the rows a GPU machine would be given for this list
    C = Corpus(rows=base / "rows.json")
    np.save(base / "embedding.npy", np.random.default_rng(3).standard_normal((C.N, 768)).astype(np.float16))  # a made-up embedding: noise
    scenarios = {"planted": dict(signal=True, small=False, planes=PLANES, analyses="all", embedding=True),
                 "none": dict(signal=False, small=False, planes=PLANES, analyses="primary", embedding=False),
                 "small": dict(signal=True, small=True, planes=[PRIMARY_PLANE], analyses="all", embedding=False)}
    report = {"what": "remakes.py synthetic: 'verify' and 'run' on made-up units (no model).", "units": a.units, "self_checks": self_checks(), "scenarios": {}}
    try:
        for name in a.scenarios.split(","):
            sc, root, t0 = scenarios[name], base / f"synthetic_{name}", time.time()
            make_synthetic(C, root, a.units, sc["signal"], sc["small"], sc["planes"])
            made, size = time.time() - t0, sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 2 ** 30
            v = verify_counts(root, C)
            assert v["exact"], "the recount does not agree with the made-up counts"
            f = root / "counts" / f"{PRIMARY_PLANE}.npz"
            Z = dict(np.load(f))
            col = C.pair_films[len(C.pair_films) // 2] if not sc["small"] else int(np.searchsorted(np.flatnonzero(C.indian | C.in_pair), C.pair_films[len(C.pair_films) // 2]))
            Z["shared"][7, col] += 1                                     # one deliberately wrong count
            np.savez(f, **Z)
            wrong = verify_counts(root, C)
            assert not wrong["exact"] and wrong["shared_counts_that_differ"] == 1, "'verify' did not catch a wrong count"
            Z["shared"][7, col] -= 1
            np.savez(f, **Z)
            args = argparse.Namespace(**{**vars(a), "tag": root.name, "acts_root": str(base), "rows": str(base / "rows.json"), "recognition": str(root / "recognition.json"),
                                         "embedding": str(base / "embedding.npy") if sc["embedding"] else str(base / "no_embedding.npy"), "analyses": sc["analyses"],
                                         "out": str(keep / f"remakes_synthetic_{name}.json")})
            out = run(args)
            d, prim = out["decision"], out["analyses"][a_label((PRIMARY_SET, PRIMARY_FRAME))]
            c = list(d["conditions"].values())
            report["scenarios"][name] = {
                "outcome_had_conditions_1_and_2_held": outcome_of([True, True] + c[2:], True, d["recall_rule"]["reading"]),
                "made_in_seconds": round(made, 1), "made_up_files_gb": size, "verify": {k: v[k] for k in ("cells_compared", "shared_counts_that_differ", "n_optional_counts_that_differ", "exact")},
                "verify_with_one_wrong_count": {k: wrong[k] for k in ("shared_counts_that_differ", "exact")},
                "outcome": d["outcome"], "conditions": d["conditions"], "slope": d["slope"], "gain": d["gain"], "standins_largest_Z": d["standins_largest_Z"],
                "planes_below_0.05": d["planes_below_0.05"], "planes_available": d["planes_available"], "recall_rule": d["recall_rule"],
                "heredity_not_copied": {k: out["heredity"]["not_copied"][k] for k in ("S", "p", "share_source_first")}, "positive_control": out["heredity"]["positive_control"],
                "embedding_passes": out["embedding"]["in_the_units_place"]["passes"] if isinstance(out["embedding"], dict) else None,
                "analyses_run": out["analyses_run"], "analyses_left_out": out["analyses_left_out"],
                "seconds_run": out["seconds"], "seconds_by_analysis": {k: b["seconds"] for k, b in out["analyses"].items()},
                "peak_memory_gb": out["peak_memory_gb"], "peak_memory_gb_largest_worker": out["peak_memory_gb_largest_worker"],
                "seconds_primary_analysis": prim["seconds"], "peak_memory_gb_primary_analysis": prim["peak_memory_gb"]}
            log(f"synthetic {name}: verify exact, one wrong count caught; outcome: {d['outcome']}; run {out['seconds']} s")
            shutil.rmtree(root)
    finally:
        shutil.rmtree(base, ignore_errors=True)                         # nothing made up is kept
    # Conditions 1 and 2 are decided by the words alone, on the real list. Where they fail there, no made-up unit can be claimed, so
    # the dry run checks the wording that conditions 3 to 7 would give had 1 and 2 held, and that the real wording is never "claim" unless they do.
    S, key = report["scenarios"], "outcome_had_conditions_1_and_2_held"
    W = json.load(open(a.words or OUT / WORDS_FILE))["claim_conditions_that_words_decide"]
    words_allow = all(v["holds"] if isinstance(v, dict) and "holds" in v else v for k, v in W.items() if k[0] in "12t")
    report["conditions_1_and_2_and_the_positive_control_hold_on_the_real_list"] = bool(words_allow)
    for name, sc in S.items():
        assert (sc["outcome"] == "claim") == (words_allow and sc[key] == "claim"), "the wording does not follow the conditions"
    if "planted" in S:
        assert S["planted"][key] == "claim", "the planted heredity was not found"
    if "none" in S:
        assert S["none"][key] == "no information beyond these word measures was detected", "something was found with nothing planted"
    if "small" in S:
        assert S["small"][key] == "detected at the primary setting only" and S["small"]["analyses_left_out"], "the small model's run is wrong"
    save(report, keep / "synthetic.json")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["words", "run", "verify", "synthetic"])
    ap.add_argument("tag", nargs="?", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=str(CACHE), help="where the word measures, the primary stand-in panel and the stages of 'words' are kept")
    ap.add_argument("--rows", default=str(ROWS), help="the rows of the counts (default data/second/remakes/rows.json)")
    ap.add_argument("--fresh", action="store_true", help="words: start again from nothing (do not resume an interrupted run)")
    ap.add_argument("--keep-scratch", action="store_true", help="words: do not delete the stand-in matrices (2.4 GB) at the end")
    ap.add_argument("--any-code", action="store_true", help="words, in development: reuse stages computed by an earlier version of the code (the output names them)")
    ap.add_argument("--panel-chunks", type=int, default=3, help="words: the stand-in measures are computed for the remakes in this many parts")
    ap.add_argument("--power-draws", type=int, default=100, help="words: draws per boost and base in the planted-boost power")
    ap.add_argument("--copies-draws", type=int, default=20, help="words: draws per setting for the noisy and curved copies (the designers used 100)")
    ap.add_argument("--acts-root", default=str(ACTS), help="run, verify: the folder that holds <tag>/")
    ap.add_argument("--words", default=None, help=f"run: default results/v2/second/{WORDS_FILE}")
    ap.add_argument("--recognition", default=None, help="run: default results/v2/second/remakes_recognition_<tag>.json")
    ap.add_argument("--embedding", default=str(ACTS / "embedding.npy"))
    ap.add_argument("--analyses", default="all", choices=["all", "primary"], help="run: every set and frame, or the primary set under its three frames")
    ap.add_argument("--workers", type=int, default=WORKERS, help="run: sets and frames analysed at the same time")
    ap.add_argument("--units", type=int, default=8192, help="synthetic: the number of made-up units (2U)")
    ap.add_argument("--scenarios", default="planted,none,small", help="synthetic: which dry runs")
    args = ap.parse_args()
    assert args.tag or args.command in ("words", "synthetic"), "a model tag is needed"
    {"words": words, "run": run, "verify": verify, "synthetic": synthetic}[args.command](args)
