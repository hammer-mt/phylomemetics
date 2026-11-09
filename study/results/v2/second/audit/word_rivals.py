"""Remakes decision test: "it is still words". Word-only measures a hostile referee would say were left out of the word model.

    cd study
    OMP_NUM_THREADS=1 ../.venv/bin/python results/v2/second/audit/word_rivals.py      -> results/v2/second/audit/word_rivals.json
    (about 75 minutes and under 3 GB with its two worker processes; AUDIT_PARTS=<folder> keeps finished blocks)

EXPLORATORY: made after the registered result was seen. Nothing here is part of v2/second/PLAN.md and nothing here changes
a registered outcome. The list below was written before any analysis of it was run; each analysis is run once and every one
is reported, whatever it shows. No file of the plan, the paper, the results or v2/ is changed.

BASELINE (reproduced first; the script stops if it is not reproduced). The registered test (v2/second/remake_test.py, built
as v2/second/remakes.py builds it) on the primary set (220 firm Indian cross-industry pairs, 175 families), frame "set" (the
candidates are the films of the Indian industries other than the remake's own, never later than the remake, never of its
family), the 39 registered word columns, penalty 3, one family left out at a time, every cell in the slope. The unit column
is the driver's: rate = shared / n_optional from v2/acts_second/remakes/<model>/counts/top1_30.npz, hub-corrected over the
220 entries (remake_test.hub_z), then normal scores within the pool. Qwen3-32B must give Z 3.4544, T 25.333, gain 19.609.
Llama 3.1 8B (registered Z 3.7308) is put through every analysis beside it where that costs no further fit.
Two more reproductions check the two routes every later analysis uses:
  route P (a measure in the units' place): the registered stand-in stemmed_words_and_pairs, read from the saved panel, must
          give Z 2.2280; the registered embedding cosine must give Z 0.8509 and gain -0.787;
  route A (a measure added to the word columns): the embedding's two forms added to the 39 columns must give the units
          Z 3.5532, T 25.357, gain 19.921 with 21.307 pairs open (the plan's registered secondary).

Every later analysis changes exactly one thing from the baseline: either one word-only measure stands in the units' place
(P), or columns are added to the word model and the units are tested again (A).

TEXTS AND TOKENS. The texts are the stored sentences of v2/cache/second/remakes/films.jsonl (at most 40 a film), the texts
every registered word measure and the units read. "The films read" are all 16,218 of them: document frequencies, idf,
vocabularies and every standardisation over films come from all of them. A token is scikit-learn's default
(two or more letters or digits). A word TYPE is a lower-cased token. A type is a NAME TYPE when it begins with a capital
letter in more than half of its occurrences that are not the first token of a sentence, over the films read (a type that
only ever stands first is neither a name type nor an ordinary type). A NAME TOKEN is a token that is not first in its
sentence and begins with a capital letter, or is first in its sentence and is of a name type; every other token is an
ORDINARY TOKEN. Stop words are scikit-learn's English list. Stems are remake_test._stem (the panel's suffix stripper).
"TF-IDF cosine" means scikit-learn's TfidfVectorizer with sublinear term frequency, smoothed idf from the films read, terms
in at least 2 films, and the cosine of the two films' vectors. Every measure is computed for every primary remake against
every film of the Indian industries (all a primary pool can hold).

THE RIVALS (each a number for a remake r and a candidate film c; none uses a model's units)
 1 Names
   N1  names_tfidf            TF-IDF cosine over the lower-cased name tokens only (no stop list).
   N1b names_containment      share of c's distinct name-token types (in 2 films or more and in at most half of the films
                              read) that r also has among its name tokens: the unit measure's twin on names.
   N2  no_names_tfidf         TF-IDF cosine over the lower-cased ordinary tokens only, stop words dropped (the complement).
 2 Rare words
   R1  rare_df2to5_tfidf      TF-IDF cosine over the words (stop words dropped) that are in at least 2 and at most 5 films.
   R2  rare_df2to20_tfidf     the same, at least 2 and at most 20 films.
 3 Plot order. Sentence vectors: TF-IDF over all sentences of the films read (stop words dropped, terms in at least 2
   sentences, sublinear, idf over sentences). S[i, j] = cosine of r's sentence i and c's sentence j.
   O1  order_alignment        the best monotone alignment: the largest sum of S over pairs (i1, j1), (i2, j2), ... with
                              i1 < i2 < ... and j1 < j2 < ... (dynamic programme, no gap cost), divided by
                              sqrt(sentences of r * sentences of c).
   O2  order_minus_reversed   O1 minus the same score with c's sentences taken in reverse order: what the alignment owes
                              to the ORDER of the sentences, which no bag-of-words or bag-of-sentences measure sees.
   O3  order_word_positions   over the words (stop words dropped, in at least 2 films and at most 5% of the films read)
                              that both texts have: Pearson correlation rho between the word's first position in r and in
                              c (index among the kept tokens over (kept tokens - 1)); the measure is
                              atanh(rho clipped to +-0.999) * sqrt(k - 3) for k shared words, 0 when k < 4 or a variance is 0.
 4 Translation-robust words
   T1  stems_no_names_tfidf   TF-IDF cosine over the stems of the ordinary tokens, stop words dropped (so: stems, no stop
                              word, no capitalised token, sublinear weights).
   T2  word_classes           16 word classes written down below before anything was run (CLASSES). A token counts for a
                              class when its stem is the stem of a word of the class. Per film and class: log(1 + 1000 *
                              tokens of the class / tokens of the film); each class standardised over the films read; the
                              measure is the cosine of the two films' 16 standardised values.
   T2b word_class_arc         the same in three parts of the film (its sentences cut into three consecutive runs of as
                              equal a number as possible, numpy.array_split): 48 values a film, each standardised over the
                              films read; cosine. A class profile that also follows the order of the plot.
   T3  ppmi_svd_centroid      a count-based word embedding made from these texts alone. Vocabulary: types that are not
                              name types, not stop words, in at least 5 films. Co-occurrence = number of sentences of the
                              films read that hold both words; positive pointwise mutual information; truncated SVD to 200
                              dimensions (randomised, 5 iterations, seed 0), word vector = U * sqrt(singular value), scaled
                              to length 1. A film is the sum of its words' vectors weighted by (1 + log count) * idf; the
                              measure is the cosine of the two film vectors. Two words that are never the same word but
                              keep the same company count as alike.
 5 A supervised word model with more room (the registered model is linear in its columns, with one cube term)
   S1  products               the six word columns with the largest absolute weight in the registered model fitted on
                              all entries (the cube term not counted), and their 15 pairwise products, each product
                              standardised within the pool: 15 added columns. Not one measure, so it has no P analysis.
   S2  gbm_score              a gradient-boosted classifier (scikit-learn HistGradientBoostingClassifier: 200 trees of depth
                              3, learning rate 0.05, at least 20 cells a leaf, L2 1.0, no early stopping, seed 20261109) of
                              "this cell is the source" on the registered 39 columns of every cell, fitted leaving families
                              out: the 175 families are cut at random (numpy default_rng(20261109)) into 10 folds and each
                              fold's cells are scored by the model fitted on the other nine. The held-out score (log odds)
                              is one more column, standardised within the pool (no hub form: it is made of columns that
                              have theirs).
 6 The sentence-embedding comparator's own variants (v2/acts_second/remakes/embedding.npy: one mean sentence vector a film,
   sentence-transformers/all-mpnet-base-v2; each vector first scaled to length 1)
   E1  embedding_minus_top3_pcs    centred on the mean of the films read, the 3 leading principal components (of the films
                                   read) removed, cosine.
   E2  embedding_minus_top10_pcs   the same with 10 components removed.
   E3  embedding_centred_within_industry   each film's vector minus the mean vector of the films read of its industry, cosine.
   (E0 embedding_cosine, the registered comparator, is used for the route checks and in A-ALL+.)
 7 Not new measures: the registered stand-ins nearest to passing
   K   the stand-ins of the registered panel whose p in the units' place is below 0.05 in remakes_words.json (7 of the
       74: the extension rule of the plan stopped while they stood), read from the saved panel
       (v2/cache/second/remakes/standin_panel_primary.npz; the panel is not rebuilt).

WHAT IS COMPUTED, FIXED IN ADVANCE
 (D) Words-only description of each rival (no unit, no test): for how many of the 220 entries its hub-corrected value puts
     the source strictly above every candidate, and the median over entries of 1 + the number of candidates above the source.
 (P) Each of N1, N1b, N2, R1, R2, O1, O2, O3, T1, T2, T2b, T3, S2, E1, E2, E3 in the units' place, word model unchanged:
     hub-corrected and turned into normal scores within the pool exactly as the unit rate is (remake_test.unit_column,
     "rate_matrix"; S2, which is defined on the cells only, is turned into normal scores within the pool without a hub
     correction). T, Z, the sign-flip p as the test computes it (100,000 flips, seed 20261108), the share of the 2,000
     registered noise draws at or above Z (remake_null_size.noise_p), the gain of the model refitted with it, the effective
     open pairs (the baseline's), and the correlation over families of its family sums with the units' family sums.
     Read against: the units' 3.4544, the stand-ins' largest Z 2.2280, the 97.5th percentile of the noise Z 2.3401.
 (A) Columns added to the 39 and the word model refitted leaving one family out; the units (Qwen3-32B and Llama 3.1 8B)
     go through slope and sign flips as in the baseline; gain for Qwen3-32B in every analysis and for Llama 3.1 8B in the
     baseline, A-ALL and A-ALL+. A measure is added in BOTH forms the driver uses (standardised within the pool; hub-
     corrected over the 220 entries, then standardised within the pool), as the driver adds the embedding; S1 and S2 are
     added as columns standardised within the pool only. The cube term stays the designers'.
       A-<rival>   one analysis for each of the 16 measures of P (2 columns each; S2: 1 column)        16 analyses
       A-S1        the 15 products
       A-K         the 7 registered stand-ins below 0.05, both forms (14 columns)
       A-ALL       every rival of groups 1 to 6 at once: the 15 measures in both forms, S2's column and S1's 15 products
                   (46 added columns, 85 with the cube)
       A-ALL+      A-ALL plus the registered embedding cosine in both forms and the 7 stand-ins of K in both forms
                   (62 added columns, 101 with the cube)
     For every A analysis: the word model's effective open pairs, sources ranked first and held-out log-likelihood; for
     the units T, Z, p, the gain, the same without the family of largest sum and over the entries the model does not
     recognise; and two noise calibrations, because the word columns are no longer the registered ones: the share of the
     2,000 registered noise draws at or above Z (marked indicative: those draws went through the 39 columns), and the same
     share among 300 noise draws put through THIS analysis's own word model (independent standard normal noise in every
     cell in the units' place, cells the word model gives under 1e-08 left out, as in the registered noise run), with the
     standard deviation and the 97.5th percentile of Z over those draws. The stand-in panel is not rebuilt (machine rule),
     so condition 5 of the plan is not re-evaluated for the A analyses: the units' Z is read against the registered 2.2280,
     the registered noise 97.5th percentile 2.3401 and the analysis's own noise.

CLASSES (T2, T2b), written before any analysis: see the dictionary CLASSES below.

SECOND ROUND (written at 03:58 UTC on 9 November 2026, after these first-round results had been seen: the baseline, all
of (P), and A-E0, A-N1, A-N1b, A-N2, A-R1; before A-T1, A-ALL and every other A analysis existed. Run once.)
 What the first round had shown by then: four names-free word measures pass the registered test in the units' place
 (N2 Z 3.34, T1 Z 3.48, T2b Z 3.16, T3 Z 2.94), although the plan's own names-free measure (TF-IDF over the words that are
 never capitalised inside a sentence anywhere in the films read; PLAN Section 15) had Z -1.49; and with N2 among the columns
 the units keep Z 3.43. So the registered panel of 74 stand-ins is not the ceiling of what words can do in the units' place,
 and the units' Z under a larger word model needs its own word-only yardstick:
   X0  The saved panel through route P under the registered 39 columns: the 74 registered stand-ins and the plan's names-free
       measure in the units' place. Must reproduce 7 of 74 below 0.05, largest Z 2.2280, names-free Z -1.4939. With it, a
       words-only description of why the plan's names-free measure differs from N2: how many of N2's terms, and what share
       of N2's tokens over the films read, are in the plan's names-free vocabulary; the 40 most frequent N2 terms it drops.
   X2a Conditions 2 and 5 of the plan under the word model of A-T1 (the 39 columns + T1 in both forms; T1 is the rival with
       the largest Z in the units' place): the 74 registered stand-ins in the units' place (share below 0.05, largest Z),
       the units of both models (must give A-T1's Z), and the measures that are not columns of that model (the 14 other
       measures of the first round, the registered embedding cosine, the plan's names-free measure).
   X2b The same under the word model of A-ALL (85 columns): the 74 registered stand-ins, the units of both models (must
       give A-ALL's Z), the registered embedding cosine and the plan's names-free measure.
 No gain is computed in the second round (conditions 2 and 5 use Z and p only). The panel is read from the file `remakes.py
 words` saved (v2/cache/second/remakes/standin_panel_primary.npz), one measure at a time; it is not rebuilt.

Memory: at most two forked worker processes of one thread each, one test alive in each; the two analyses with every rival
run one at a time. AUDIT_DRY=1 runs every step on every sixth entry with random
numbers in the units' place and with every rival's rows shuffled over the entries after (D) (to test the code without
seeing a result) and writes to AUDIT_OUT. AUDIT_PARTS names a folder where finished blocks are kept, so that a crash does
not run an analysis twice. AUDIT_STOP_AFTER_CHECKS=1 stops after the three reproductions.
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
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, "v2/second")
import remakes as R                # noqa: E402
import remake_test as T            # noqa: E402
import remake_null_size as NS      # noqa: E402

CLASSES = {
    "kinship": "father mother son daughter brother sister uncle aunt nephew niece cousin grandfather grandmother grandson granddaughter husband wife widow "
               "widower parents parent family child children baby twin twins stepmother stepfather orphan adopted adopts elder younger sibling siblings relative relatives",
    "romance_marriage": "love loves loved lover lovers romance romantic marry marries married marriage wedding engagement engaged bride groom fiance fiancee "
                        "proposes proposal girlfriend boyfriend affair kiss elope elopes divorce divorced beloved heart",
    "crime": "crime criminal criminals gangster gang don mafia smuggler smuggling smugglers thief thieves steal steals stolen robbery rob robs robbed kidnap "
             "kidnaps kidnapped kidnapping murder murders murdered murderer rape raped rapist blackmail blackmails goon goons henchmen underworld bandit dacoit dacoits culprit killer",
    "law_police": "police inspector officer cop cops constable commissioner arrest arrests arrested jail prison prisoner court judge lawyer advocate trial "
                  "case witness evidence investigate investigates investigation sentenced convicted guilty innocent bail law",
    "violence_death": "kill kills killed killing die dies died death dead fight fights fought beat beats beaten attack attacks attacked shoot shoots shot stab "
                      "stabs stabbed injured wounded revenge avenge vengeance suicide accident blast bomb gun knife",
    "money_business": "money rich poor wealthy wealth poverty loan debt debts business businessman company property estate fortune lakh lakhs crore crores "
                      "rupees bank job salary inheritance heir owner factory mill landlord zamindar millionaire dowry",
    "place_setting": "village villagers town city rural farmer farmers land farm temple forest river palace kingdom house home hostel",
    "education_work": "college school student students teacher professor study studies exam degree doctor engineer nurse medical office boss employee clerk driver servant",
    "illness_body": "ill illness disease cancer blind deaf mute mental mad insane pregnant pregnancy coma memory amnesia hospital operation surgery treatment cure paralysed sick",
    "deception_identity": "disguise disguised pretend pretends pretending lie lies lied secret truth reveal reveals revealed identity mistaken lookalike impostor "
                          "impersonate double fake plan trick tricks cheat cheats cheated misunderstanding",
    "emotion": "angry anger happy sad jealous jealousy hate hates hatred shocked shock fear afraid cry cries tears regret guilt forgive forgives apologise "
               "apologizes proud humiliated insult insults",
    "supernatural_religion": "god goddess ghost spirit soul reincarnation rebirth curse cursed prayer pray priest saint miracle devil demon magic magical astrologer horoscope ritual",
    "politics_power": "minister politician political election party leader government chief king queen prince princess throne ruler",
    "journey": "leave leaves left return returns returned travel travels journey train bus car flight abroad foreign escape escapes run runs flee flees arrive arrives arrived",
    "arts": "singer song sing sings dance dancer music musician actor actress film movie director stage drama painter artist writer poet",
    "friendship_social": "friend friends friendship enemy enemies rival rivals neighbour neighbor colleague classmate roommate stranger guest",
}

DRY = bool(os.environ.get("AUDIT_DRY"))
OUT = Path(os.environ.get("AUDIT_OUT", "results/v2/second/audit/word_rivals.json"))
PARTS = os.environ.get("AUDIT_PARTS")
STOP_AFTER_CHECKS = bool(os.environ.get("AUDIT_STOP_AFTER_CHECKS"))
TAGS = ("qwen3-32b", "llama3.1-8b")
MAIN = TAGS[0]
NOISE_DRAWS, NOISE_BLOCK, NOISE_SEED = (20, 10, 7_300_000) if DRY else (300, 100, 7_300_000)
GBM_SEED, GBM_FOLDS = 20261109, 10
WORKERS = int(os.environ.get("AUDIT_WORKERS", "2"))          # forked worker processes, one thread each; the two analyses with every rival run one at a time
G = {}                                                       # what forked workers read
STANDINS_Z, NOISE_975 = 2.2279788170279393, 2.340116048458559
EXPLORATORY = "exploratory: made after the registered result was seen"
TOKEN = re.compile(r"(?u)\b\w\w+\b")
STOP = ENGLISH_STOP_WORDS
T0 = time.time()
peak_gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20      # noqa: E731


def log(*a):
    print(f"{time.time() - T0:7.0f}s {peak_gb():.2f}GB", *a, flush=True)


def part_file(key):
    return Path(PARTS) / (re.sub(r"[^A-Za-z0-9_+-]", "_", key) + ".json")


def part(key, fn):
    """A finished block, kept in PARTS (if set) so that a crash does not run an analysis twice."""
    if PARTS:
        f = part_file(key)
        if f.exists():
            log("read back", key)
            return json.load(open(f))
    res = json.loads(json.dumps(fn(), default=R.plain))
    if PARTS:
        Path(PARTS).mkdir(parents=True, exist_ok=True)
        json.dump(res, open(f, "w"))
    return res


# ------------------------------------------------------------------ inputs
W = json.load(open(R.OUT / R.WORDS_FILE))
REG = {tag: json.load(open(R.OUT / f"remakes_{tag}.json")) for tag in TAGS}
C = R.Corpus()
for _k in ("pairs.json", "films.jsonl"):
    assert W["sha256"][_k] == C.sha[_k] == REG[MAIN]["sha256"][_k], _k
ENTS = C.entries("primary")
SUB = np.arange(len(ENTS))[::6] if DRY else np.arange(len(ENTS))            # the entries used (all of them, unless a dry run)
ENTS = [ENTS[i] for i in SUB]
N = len(ENTS)
ROWS = np.array([C.rpos[p["r"]] for p in ENTS])                              # each entry's row in a [remakes, films] matrix
RR, SS = np.array([p["r"] for p in ENTS]), np.array([p["s"] for p in ENTS])
FAMILIES = [p["family"] for p in ENTS]
POOLS = [C.pool(p, "primary", "set") for p in ENTS]
IND = np.flatnonzero(C.indian)                                               # every film a primary pool can hold
assert all(np.isin(pool, IND).all() for pool in POOLS)
REM = np.array(sorted(set(RR.tolist())))                                     # the distinct remakes of the entries
REM_OF = np.searchsorted(REM, RR)                                            # each entry's row in a [REM, IND] matrix

GROUP = np.repeat(np.arange(N), [len(pool) for pool in POOLS])                # each cell's entry, the pools one after another (remake_test.Stack)
CELL_FILM = np.concatenate(POOLS)                                            # each cell's film


def small(M):
    """An [entries, films read] matrix kept on the columns of the Indian industries only (all a primary pool can hold)."""
    return np.ascontiguousarray(np.asarray(M)[:, IND], dtype=np.float32)


def full(Ms):
    """Back to [entries, films read] float32, 0 for a film that is in no primary pool (the test never reads those cells)."""
    M = np.zeros((N, C.N), np.float32)
    M[:, IND] = Ms
    return M


def scatter(col):
    """A stacked column [cells] as an [entries, films read] float32 matrix (0 outside the pools)."""
    M = np.zeros((N, C.N), np.float32)
    M[GROUP, CELL_FILM] = col
    return M


def registered_measures():
    """The measures of the 39 registered columns, one row per entry, [entries, films read] float32 (as the driver holds them)."""
    M, meta = R.word_measures(C, R.CACHE, must_exist=True)
    assert meta["sha256"] == W["sha256"]["word_measures"]
    cols, added = R.registered_columns(W, R.CACHE)
    return cols, {k: np.ascontiguousarray(v[ROWS]) for k, v in {**M, **added}.items() if k in {m for m, _ in cols}}


COLS, _me = registered_measures()
ME = {k: small(v) for k, v in _me.items()}                                   # kept small; made full again for each fit
del _me
gc.collect()
SM, CELL = {}, {}                                                            # the rivals: [entries, Indian films] float32, or one value per cell


def unit_rate(tag):
    """The driver's rate (remakes.read_counts), one row per entry."""
    root = R.ACTS / tag
    ids = np.asarray(json.load(open(root / "meta.json"))["ids"], dtype=np.int64)
    Z = np.load(root / "counts" / f"{R.PRIMARY_PLANE}.npz")
    assert np.array_equal(ids[Z["rows"]], C.remakes) and Z["shared"].shape == (len(C.remakes), len(ids))
    rate = np.zeros((N, C.N))
    rate[:, ids] = Z["shared"][ROWS] / np.maximum(Z["n_optional"], 1)[None, :]
    if DRY:
        rate = np.random.default_rng(len(tag)).random((N, C.N))
    return rate


HUB, KNOWN = {}, {}
for _tag in TAGS:
    HUB[_tag] = T.hub_z(unit_rate(_tag), FAMILIES, C.film_family)            # remake_test.unit_column's hub correction, over all the entries
    _rec = {(q["remake"], q["source"]): q for q in json.load(open(R.OUT / f"remakes_recognition_{_tag}.json"))["pairs"]}
    KNOWN[_tag] = np.array([bool(_rec[(p["remake"], p["source"])]["recognised"]) for p in ENTS])
log(f"inputs: {N} entries, {len(set(FAMILIES))} families, {sum(map(len, POOLS))} cells, {len(REM)} remakes, {len(IND)} films of the Indian industries")


# ------------------------------------------------------------------ the test
def measure(m):
    return full(ME[m]) if m in ME else full(SM[m]) if m in SM else scatter(CELL[m])


def build(extra_cols=()):
    """remake_test.RemakeTest on the entries, as remakes.Analysis builds it: the registered columns, then the added ones."""
    cells = [{"pool": pool, "source": int(SS[i]), "family": FAMILIES[i]} for i, pool in enumerate(POOLS)]
    cols = COLS + list(extra_cols)
    meas = {m: measure(m) for m in {m for m, _ in cols}}
    test = T.RemakeTest(cells, meas, C.film_family, columns=cols)
    test._hub.clear()
    del meas
    gc.collect()
    return test


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


def noise_block(zs):
    return {"draws": len(zs), "mean_Z": float(zs.mean()), "sd_Z": float(zs.std()), "Z_quantiles": {q: float(np.quantile(zs, float(q))) for q in ("0.9", "0.95", "0.975", "0.99")},
            "largest_Z": float(zs.max()), "share_of_draws_with_Z_above_2.228": float((zs > STANDINS_Z).mean()),
            "what": "independent standard normal noise in the units' place through this analysis's own word model; cells under 1e-08 left out"}


def units_block(test, Ur, tag):
    S = test.S
    s, d = summary(test, Ur)
    order = np.argsort(-np.abs(d), kind="stable")
    ss = float((d ** 2).sum())
    brief = lambda x: {k: x[k] for k in ("entries", "families", "T", "Z", "p", "effective_pairs_left_open")}       # noqa: E731
    out = dict(s)
    out["effective_number_of_families"] = float(np.abs(d).sum() ** 2 / ss) if ss > 0 else 0.0
    out["largest_family_sums"] = [{"family": S.fam_names[f], "D": float(d[f]), "remakes": [ENTS[i]["remake"] for i in np.flatnonzero(S.fam == f)]} for f in order[:5]]
    out["without_the_family_of_largest_sum"] = brief(summary(test, Ur, S.fam != order[0])[0])
    out["entries_the_model_does_not_recognise"] = brief(summary(test, Ur, ~KNOWN[tag])[0])
    out["contributions_by_entry"] = Ur.tolist()
    return out, d


def expand(Mri):
    """A [REM, IND] matrix as [entries, Indian films] float32."""
    return np.ascontiguousarray(np.asarray(Mri)[REM_OF], dtype=np.float32)


# ------------------------------------------------------------------ the rivals (words only)
def token_tables():
    """One pass over the films read: which types are name types; then each film's name tokens and ordinary tokens (lower-cased)."""
    nmid, ncap = Counter(), Counter()
    for sents in C.sentences:
        for s in sents:
            for w in TOKEN.findall(s)[1:]:
                lw = w.lower()
                nmid[lw] += 1
                if w[0].isupper():
                    ncap[lw] += 1
    name_types = {w for w in nmid if ncap[w] / nmid[w] > 0.5}
    ordinary_types = set(nmid) - name_types
    names, ordinary = [], []
    for sents in C.sentences:
        a, b = [], []
        for s in sents:
            for i, w in enumerate(TOKEN.findall(s)):
                lw = sys.intern(w.lower())
                (a if (w[0].isupper() if i else lw in name_types) else b).append(lw)
        names.append(a)
        ordinary.append(b)
    return name_types, ordinary_types, names, ordinary


def cosine_small(A):
    """Cosine of every remake of REM with every film of IND, from one row per film read (sparse or dense)."""
    A = normalize(A)
    X = A[REM] @ A[IND].T
    return np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float32)


def identity(doc):
    return doc


def monotone(P):
    """P [films, c's sentences (padded with 0), r's sentences]: for every film the largest sum of P over a monotone matching."""
    D = np.zeros((P.shape[0], P.shape[2] + 1), P.dtype)
    for i in range(P.shape[1]):
        D[:, 1:] = np.maximum.accumulate(np.maximum(D[:, 1:], D[:, :-1] + P[:, i, :]), axis=1)
    return D[:, -1]


def order_measures():
    n_s = np.array([len(s) for s in C.sentences])
    starts = np.concatenate([[0], np.cumsum(n_s)])
    flat = [s for ss in C.sentences for s in ss]
    A = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit_transform(flat).tocsr()
    rows = np.concatenate([np.arange(starts[c], starts[c + 1]) for c in IND])
    owner = np.repeat(np.arange(len(IND)), n_s[IND])
    pos = np.concatenate([np.arange(n_s[c]) for c in IND])
    A_ind = A[rows]
    O1, O2 = np.zeros((len(REM), len(IND)), np.float32), np.zeros((len(REM), len(IND)), np.float32)
    for a, r in enumerate(REM):
        Sm = np.asarray((A_ind @ A[starts[r]:starts[r + 1]].T).todense(), dtype=np.float32)      # [sentences of the Indian films, r's sentences]
        P = np.zeros((len(IND), T.MAX_SENTENCES, n_s[r]), np.float32)
        P[owner, pos] = Sm
        fwd, rev = monotone(P), monotone(P[:, :, ::-1])
        norm = np.sqrt(n_s[r] * n_s[IND])
        O1[a], O2[a] = fwd / norm, (fwd - rev) / norm
    return O1, O2, len(flat), A.shape[1]


def word_positions(P_df, cv):
    """O3. P_df: document frequencies of remake_test._presence's vocabulary (stop words dropped, in at least 2 films)."""
    an, vocab = T._base_analyzer(), cv.vocabulary_
    keep = (P_df >= 2) & (P_df <= 0.05 * C.N)
    rows, cols, vals = [], [], []
    for f, text in enumerate(C.texts):
        toks, first = an(text), {}
        for i, w in enumerate(toks):
            j = vocab.get(w)
            if j is not None and keep[j] and j not in first:
                first[j] = i
        den = max(len(toks) - 1, 1)
        for j, i in first.items():
            rows.append(f)
            cols.append(j)
            vals.append(1.0 + i / den)                                        # 1 + position, so that a word that is there is never 0
    FP = sp.csr_matrix((vals, (rows, cols)), shape=(C.N, len(P_df)))
    FPI = FP[IND].tocsc()
    out = np.zeros((len(REM), len(IND)), np.float32)
    for a, r in enumerate(REM):
        row = FP[r]
        x = row.data - 1.0
        sub = np.asarray(FPI[:, row.indices].todense())
        pres = sub > 0
        y = np.where(pres, sub - 1.0, 0.0)
        k = pres.sum(1).astype(np.float64)
        kk = np.maximum(k, 1)
        sx, sy, sxx, syy, sxy = pres @ x, y.sum(1), pres @ (x * x), (y * y).sum(1), y @ x
        cov, vx, vy = sxy - sx * sy / kk, sxx - sx * sx / kk, syy - sy * sy / kk
        ok = (k >= 4) & (vx > 1e-12) & (vy > 1e-12)
        rho = np.where(ok, cov / np.sqrt(np.where(ok, vx * vy, 1.0)), 0.0)
        out[a] = np.where(ok, np.arctanh(np.clip(rho, -0.999, 0.999)) * np.sqrt(np.maximum(k - 3, 0)), 0.0)
    return out, int(keep.sum())


def class_profiles():
    """T2 and T2b: [films, 16] and [films, 48] standardised class rates."""
    names = list(CLASSES)
    stems = {}
    for k, cl in enumerate(names):
        for w in CLASSES[cl].split():
            stems.setdefault(T._stem(w), set()).add(k)
    cache = {}
    parts, ntok_part = np.zeros((C.N, 3, len(names))), np.zeros((C.N, 3))
    for f, sents in enumerate(C.sentences):
        for part_no, idx in enumerate(np.array_split(np.arange(len(sents)), 3)):
            cnt, n = [0] * len(names), 0
            for i in idx:
                for w in TOKEN.findall(sents[i]):
                    lw = w.lower()
                    ks = cache.get(lw)
                    if ks is None:
                        ks = cache[lw] = tuple(stems.get(T._stem(lw), ()))
                    n += 1
                    for k in ks:
                        cnt[k] += 1
            parts[f, part_no], ntok_part[f, part_no] = cnt, n
    whole, ntok = parts.sum(1), ntok_part.sum(1)
    z = lambda x: (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)      # noqa: E731
    A16 = z(np.log1p(1000 * whole / np.maximum(ntok, 1)[:, None]))
    A48 = z(np.log1p(1000 * parts / np.maximum(ntok_part, 1)[:, :, None]).reshape(C.N, -1))
    return A16, A48, {cl: int(whole[:, k].sum()) for k, cl in enumerate(names)}, sum(len(v) > 0 for v in cache.values())


def ppmi_svd(ordinary_types):
    df = Counter()
    for text in C.texts:
        df.update(set(w.lower() for w in TOKEN.findall(text)))
    V = sorted(w for w in ordinary_types if w not in STOP and df[w] >= 5)
    flat = [s for ss in C.sentences for s in ss]
    B = CountVectorizer(vocabulary=V, binary=True).fit_transform(flat).astype(np.float32).tocsr()
    Co = (B.T @ B).tocsr()
    Co.setdiag(0)
    Co.eliminate_zeros()
    Co = Co.tocoo()
    rs = np.asarray(Co.sum(1)).ravel().astype(np.float64)
    pmi = np.log(Co.data.astype(np.float64) * rs.sum() / (rs[Co.row] * rs[Co.col]))
    keep = pmi > 0
    PP = sp.csr_matrix((pmi[keep], (Co.row[keep], Co.col[keep])), shape=Co.shape)
    svd = TruncatedSVD(200, algorithm="randomized", n_iter=5, random_state=0)
    US = svd.fit_transform(PP)
    Wv = normalize(US / np.sqrt(np.maximum(svd.singular_values_, 1e-12))[None, :])
    F = TfidfVectorizer(vocabulary=V, sublinear_tf=True, norm=None).fit_transform(C.texts)
    return cosine_small(np.asarray(F @ Wv)), len(V), int(PP.nnz)


def embedding_variants():
    E = np.load(R.ACTS / "embedding.npy").astype(np.float64)
    assert E.shape[0] == C.N
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    Xc = E - E.mean(0)
    w, V = np.linalg.eigh(Xc.T @ Xc)
    V = V[:, ::-1]                                                            # leading components first
    out = {}
    for k in (3, 10):
        out[f"embedding_minus_top{k}_pcs"] = cosine_small(Xc - (Xc @ V[:, :k]) @ V[:, :k].T)
    Xi = E.copy()
    for o in sorted(set(C.origin.tolist())):
        m = C.origin == o
        Xi[m] -= E[m].mean(0)
    out["embedding_centred_within_industry"] = cosine_small(Xi)
    share = (w[::-1] / w.sum())
    return out, {"share_of_variance_in_the_3_leading_components": float(share[:3].sum()), "share_of_variance_in_the_10_leading_components": float(share[:10].sum())}


def make_rivals():
    """Every rival of groups 1 to 4 and 6 as [REM, IND] float32, with what each was made of."""
    made, info = {}, {}
    name_types, ordinary_types, names, ordinary = token_tables()
    info["tokens"] = {"name_types": len(name_types), "ordinary_types": len(ordinary_types), "name_tokens": int(sum(map(len, names))), "ordinary_tokens": int(sum(map(len, ordinary)))}
    log("tokens", info["tokens"])
    # 1 names
    v = TfidfVectorizer(analyzer=identity, min_df=2, sublinear_tf=True)
    made["names_tfidf"] = cosine_small(v.fit_transform(names))
    info["names_tfidf"] = {"terms": len(v.vocabulary_)}
    cvn = CountVectorizer(analyzer=identity, min_df=2, binary=True)
    Pn = cvn.fit_transform(names).astype(np.float32).tocsr()
    dfn = np.asarray(Pn.sum(0)).ravel()
    made["names_containment"] = T._containment(Pn, REM, (dfn >= 2) & (dfn <= C.N / 2))[:, IND].astype(np.float32)
    info["names_containment"] = {"terms": int(((dfn >= 2) & (dfn <= C.N / 2)).sum())}
    v = TfidfVectorizer(analyzer=lambda d: [w for w in d if w not in STOP], min_df=2, sublinear_tf=True)
    made["no_names_tfidf"] = cosine_small(v.fit_transform(ordinary))
    info["no_names_tfidf"] = {"terms": len(v.vocabulary_)}
    # second round, X0: the plan's names-free vocabulary (remakes.names_free_measure) beside N2's terms; words only
    capital = set()
    for sents in C.sentences:
        for s in sents:
            capital.update(w.lower() for w in TOKEN.findall(s)[1:] if w[0].isupper())
    reg_vocab = CountVectorizer(stop_words="english", min_df=2).fit(C.texts).get_feature_names_out()
    kept = {w for w in reg_vocab if w not in capital}
    cnt = Counter(w for d in ordinary for w in d if w in v.vocabulary_)
    in_plan = [w for w in v.vocabulary_ if w in kept]
    dropped = sorted((w for w in v.vocabulary_ if w not in kept), key=lambda w: -cnt[w])
    info["plan_names_free_vocabulary"] = {
        "what": "the plan's names-free measure keeps a word only if no film read ever capitalises it inside a sentence; N2 drops a token only where it is itself a name token",
        "words_of_the_plans_measure": len(kept), "words_the_plans_measure_drops": int(len(reg_vocab) - len(kept)),
        "terms_of_N2": len(v.vocabulary_), "terms_of_N2_that_the_plans_measure_keeps": len(in_plan),
        "tokens_of_N2_over_the_films_read": int(sum(cnt.values())), "share_of_N2s_tokens_whose_word_the_plans_measure_keeps": float(sum(cnt[w] for w in in_plan) / sum(cnt.values())),
        "the_40_most_frequent_terms_of_N2_that_the_plans_measure_drops": [[w, int(cnt[w])] for w in dropped[:40]]}
    log("plan's names-free vocabulary", {k: x for k, x in info["plan_names_free_vocabulary"].items() if not isinstance(x, (list, str))})
    # 4 T1 (the same tokens as N2, stemmed)
    v = TfidfVectorizer(analyzer=lambda d: [T._stem(w) for w in d if w not in STOP], min_df=2, sublinear_tf=True)
    made["stems_no_names_tfidf"] = cosine_small(v.fit_transform(ordinary))
    info["stems_no_names_tfidf"] = {"terms": len(v.vocabulary_)}
    del names, ordinary, Pn
    gc.collect()
    log("names, no names, stems")
    # 2 rare words
    P, df, cv = T._presence(C.texts)
    vocab = cv.get_feature_names_out()
    for hi in (5, 20):
        m = (df >= 2) & (df <= hi)
        made[f"rare_df2to{hi}_tfidf"] = cosine_small(TfidfVectorizer(stop_words="english", sublinear_tf=True, vocabulary=vocab[m]).fit_transform(C.texts))
        info[f"rare_df2to{hi}_tfidf"] = {"terms": int(m.sum())}
    log("rare words")
    # 3 plot order
    made["order_alignment"], made["order_minus_reversed"], n_sent, n_terms = order_measures()
    info["order_alignment"] = info["order_minus_reversed"] = {"sentences": n_sent, "terms": n_terms}
    made["order_word_positions"], n_terms = word_positions(df, cv)
    info["order_word_positions"] = {"terms": n_terms}
    del P
    gc.collect()
    log("plot order")
    # 4 word classes and the count-based embedding
    A16, A48, counts, n_types = class_profiles()
    made["word_classes"], made["word_class_arc"] = cosine_small(A16), cosine_small(A48)
    info["word_classes"] = info["word_class_arc"] = {"classes": len(CLASSES), "types_in_a_class": n_types, "tokens_by_class_over_the_films_read": counts}
    made["ppmi_svd_centroid"], n_v, nnz = ppmi_svd(ordinary_types)
    info["ppmi_svd_centroid"] = {"words": n_v, "positive_pmi_cells": nnz}
    log("word classes, ppmi")
    # 6 the embedding's variants
    emb, einfo = embedding_variants()
    made.update(emb)
    for k in emb:
        info[k] = einfo
    log("embedding variants")
    return made, info


def self_checks():
    """The dynamic programme against a brute force, and the position correlation against numpy, on made-up numbers."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        n, m = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        S = rng.random((n, m)).astype(np.float32)
        best = np.zeros((n + 1, m + 1))
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                best[i, j] = max(best[i - 1, j], best[i, j - 1], best[i - 1, j - 1] + S[i - 1, j - 1])
        P = np.zeros((1, 8, m), np.float32)
        P[0, :n] = S
        assert abs(monotone(P)[0] - best[n, m]) < 1e-5
    return True


DEFINITIONS = {
    "names_tfidf": "N1: TF-IDF cosine over the lower-cased name tokens (a token capitalised inside a sentence, or first in its sentence and of a name type); no stop list",
    "names_containment": "N1b: share of the candidate's distinct name-token types (in 2 films or more, at most half of the films read) that the remake has among its name tokens",
    "no_names_tfidf": "N2: TF-IDF cosine over the lower-cased ordinary (not name) tokens, stop words dropped",
    "rare_df2to5_tfidf": "R1: TF-IDF cosine over the words (stop words dropped) that are in at least 2 and at most 5 of the films read",
    "rare_df2to20_tfidf": "R2: TF-IDF cosine over the words (stop words dropped) that are in at least 2 and at most 20 of the films read",
    "order_alignment": "O1: best monotone alignment of the two sentence sequences (sum of sentence TF-IDF cosines over an order-keeping matching, no gap cost) / sqrt(sentences of r * sentences of c)",
    "order_minus_reversed": "O2: O1 minus the same score with the candidate's sentences in reverse order",
    "order_word_positions": "O3: atanh(Pearson correlation of the first positions of the shared words in the two texts) * sqrt(shared words - 3); words in 2 films to 5% of the films read; 0 under 4 shared words",
    "stems_no_names_tfidf": "T1: TF-IDF cosine over the stems (remake_test._stem) of the ordinary tokens, stop words dropped",
    "word_classes": "T2: cosine of the two films' 16 word-class rates (log(1 + 1000 * class tokens / tokens), each class standardised over the films read); the classes are CLASSES in the script",
    "word_class_arc": "T2b: the same in three consecutive parts of each film: cosine of 48 standardised values",
    "ppmi_svd_centroid": "T3: cosine of film vectors made of count-based word vectors (sentence co-occurrence in the films read, positive PMI, SVD to 200 dimensions), words weighted by (1 + log count) * idf; no name type, no stop word",
    "gbm_score": "S2: held-out log odds of a gradient-boosted classifier of 'this cell is the source' on the registered 39 columns, fitted leaving families out in 10 folds",
    "embedding_minus_top3_pcs": "E1: cosine of the mean sentence embeddings after centring on the films read and removing the 3 leading principal components",
    "embedding_minus_top10_pcs": "E2: the same with the 10 leading principal components removed",
    "embedding_centred_within_industry": "E3: cosine of the mean sentence embeddings after subtracting the mean vector of the films read of the film's industry",
    "embedding_cosine": "E0: the registered comparator: cosine of the mean sentence embeddings",
}
ORDER = ["names_tfidf", "names_containment", "no_names_tfidf", "rare_df2to5_tfidf", "rare_df2to20_tfidf", "order_alignment", "order_minus_reversed", "order_word_positions",
         "stems_no_names_tfidf", "word_classes", "word_class_arc", "ppmi_svd_centroid", "embedding_minus_top3_pcs", "embedding_minus_top10_pcs", "embedding_centred_within_industry"]
CODE = dict(zip(ORDER, ["N1", "N1b", "N2", "R1", "R2", "O1", "O2", "O3", "T1", "T2", "T2b", "T3", "E1", "E2", "E3"]), gbm_score="S2", embedding_cosine="E0")


def describe(name):
    """(D): where the hub-corrected measure ranks the source in its pool."""
    H = T.hub_z(measure(name), FAMILIES, C.film_family)
    above = np.array([int((H[i, POOLS[i]] > H[i, SS[i]]).sum()) for i in range(N)])
    ties = np.array([int((H[i, POOLS[i]] == H[i, SS[i]]).sum()) - 1 for i in range(N)])
    return {"source_strictly_above_every_candidate": int(((above == 0) & (ties == 0)).sum()), "median_rank_of_the_source": float(np.median(1 + above)),
            "source_in_the_top_10": int((above < 10).sum()), "entries": N}


# ------------------------------------------------------------------ the analyses
def supervised(test):
    """S2's held-out score of every cell and S1's six columns, from the baseline's word columns."""
    S = test.S
    y = np.zeros(S.n, np.int8)
    y[S.src] = 1
    fold_of_fam = np.empty(S.n_fam, np.int64)
    fold_of_fam[np.random.default_rng(GBM_SEED).permutation(S.n_fam)] = np.arange(S.n_fam) % GBM_FOLDS
    cell_fold = fold_of_fam[S.fam[S.group]]
    gbm = np.empty(S.n)
    for k in range(GBM_FOLDS):
        te = cell_fold == k
        g = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=20, l2_regularization=1.0, early_stopping=False, random_state=GBM_SEED)
        g.fit(test.X[~te], y[~te])
        gbm[te] = g.decision_function(test.X[te])
    six = np.argsort(-np.abs(test.beta_all[:len(COLS)]), kind="stable")[:6]
    CELL["gbm_score"] = gbm.astype(np.float32)
    for ia, a in enumerate(six):
        for b in six[ia + 1:]:
            CELL[f"prod:{test.names[a]}*{test.names[b]}"] = (test.X[:, a] * test.X[:, b]).astype(np.float32)
    return gbm, six


def baseline_and_checks():
    """The registered numbers reproduced; route P checked; everything of (P)."""
    t0 = time.time()
    test = build()
    S = test.S
    assert np.array_equal(S.group, GROUP) and np.array_equal(S.cols, CELL_FILM)
    out = {"what": "the registered test reproduced: primary set, frame set, the 39 registered columns, every cell", "exploratory": EXPLORATORY,
           "entries": N, "families": int(S.n_fam), "cells": int(S.n),
           "word_model": {"effective_pairs_left_open": float(test.unsolved), "sources_ranked_first": int((test.rank0 == 1).sum()), "held_out_log_likelihood": float(test.ll0.sum())},
           "units": {}, "checks": {}}
    if not DRY:
        cols, me = registered_measures()
        an = R.Analysis(C, me, "primary", "set", columns=cols, by_entry=True)                 # the driver's own construction, from the measures as the driver holds them
        out["checks"]["word_columns_equal_the_driver's_in_every_cell"] = bool(np.array_equal(an.test.X, test.X))
        out["checks"]["held_out_source_probabilities_equal_the_driver's"] = bool(np.array_equal(an.test.p_source, test.p_source))
        out["checks"]["unit_column_equals_the_driver's"] = bool(np.array_equal(an.test.unit_column(unit_rate(MAIN), "rate_matrix"), S.rank_column(HUB[MAIN])))
        assert all(out["checks"].values()), out["checks"]
        del an, me
        gc.collect()
    gbm, six = supervised(test)
    rank_gbm = S.rank_of_source(gbm)
    mx = np.maximum.reduceat(gbm, S.starts)
    lse = np.log(np.add.reduceat(np.exp(gbm - np.repeat(mx, S.sizes)), S.starts)) + mx
    out["gbm_score"] = {"definition": DEFINITIONS["gbm_score"], "sources_ranked_first_by_the_held_out_score": int((rank_gbm == 1).sum()),
                        "held_out_log_likelihood_of_the_sources_as_a_softmax_of_the_score_within_the_pool": float((gbm[S.src] - lse).sum()),
                        "ranks_of_the_sources_not_first": sorted(rank_gbm[rank_gbm != 1].tolist()),
                        "the_registered_word_model": {"sources_ranked_first": int((test.rank0 == 1).sum()), "held_out_log_likelihood": float(test.ll0.sum())}}
    out["products"] = {"the_six_columns": [test.names[j] for j in six], "their_weights_fitted_on_all_entries": [float(test.beta_all[j]) for j in six],
                       "added_columns": [k for k in CELL if k.startswith("prod:")]}
    log(f"gbm: sources first {out['gbm_score']['sources_ranked_first_by_the_held_out_score']} (word model {int((test.rank0 == 1).sum())}); six columns {out['products']['the_six_columns']}")
    # the units, then the two measures that check route P, then every measure of (P), side by side
    names = ["embedding_cosine", CHECK_STANDIN] + list(ORDER) + ["gbm_score"]
    cols = [S.rank_column(HUB[tag]) for tag in TAGS]
    for k in names:
        cols.append(S.normal_scores(gbm) if k == "gbm_score" else test.unit_column(measure(k), "rate_matrix"))
    U = np.column_stack(cols)
    del cols
    Ur = test.slope(U)
    slopes = [summary(test, Ur[:, j]) for j in range(U.shape[1])]
    if not DRY:                                                                # the reproductions come before anything new is looked at
        a = REG[MAIN]["analyses"]["primary | set"]
        want = {MAIN: REG[MAIN]["decision"]["slope"]["Z"], TAGS[1]: REG[TAGS[1]]["decision"]["slope"]["Z"],
                "embedding_cosine": a["embedding"]["in_the_units_place"]["slope"]["Z"], CHECK_STANDIN: W["analyses"]["primary | set"]["standins"]["rows"][CHECK_STANDIN]["Z"]}
        for j, k in enumerate(list(TAGS) + names[:2]):
            if abs(slopes[j][0]["Z"] - want[k]) > 5e-5:
                raise SystemExit(f"not reproduced: {k} gives Z {slopes[j][0]['Z']} where the registered files give {want[k]}")
    G.update(test=test, U=U)
    try:
        with get_context("fork").Pool(WORKERS) as pool:
            gains = pool.map(gain_job, range(U.shape[1]), chunksize=1)
    finally:
        G.clear()
    fam = {}
    for j, tag in enumerate(TAGS):
        blk, fam[tag] = units_block(test, Ur[:, j], tag)
        blk["gain"] = gains[j]
        blk["noise_p_registered_draws"] = NS.noise_p(blk["Z"])
        out["units"][tag] = blk
        log(f"baseline: {tag} T {blk['T']:.4f} Z {blk['Z']:.4f} p {blk['p']:.5f} gain {blk['gain']['gain']:.4f}")
    if not DRY:
        for tag in TAGS:
            reg, got = REG[tag]["decision"], out["units"][tag]
            out["checks"][f"registered_{tag}"] = {"Z": reg["slope"]["Z"], "T": reg["slope"]["T"], "gain": reg["gain"], "effective_pairs_left_open": reg["slope"]["effective_pairs_left_open"]}
            if not (abs(got["Z"] - reg["slope"]["Z"]) < 5e-5 and abs(got["T"] - reg["slope"]["T"]) < 5e-4 and abs(got["gain"]["gain"] - reg["gain"]) < 5e-4):
                raise SystemExit(f"the registered number is not reproduced for {tag}: {got['Z']}, {got['T']}, {got['gain']['gain']} against {out['checks'][f'registered_{tag}']}")
        out["checks"]["reproduced_to_four_digits"] = True
        j = len(TAGS)
        chk = {"embedding_cosine": (slopes[j][0]["Z"], gains[j]["gain"], a["embedding"]["in_the_units_place"]["slope"]["Z"], a["embedding"]["in_the_units_place"]["gain"]["gain"]),
               CHECK_STANDIN: (slopes[j + 1][0]["Z"], gains[j + 1]["gain"], want[CHECK_STANDIN], None)}
        for k, (z, gain, zreg, gainreg) in chk.items():
            ok = abs(z - zreg) < 5e-5 and (gainreg is None or abs(gain - gainreg) < 5e-4)
            out["checks"][f"route_P_{k}"] = {"Z": z, "registered_Z": zreg, "gain": gain, "registered_gain": gainreg, "reproduced": bool(ok)}
            if not ok:
                raise SystemExit(f"route P is not reproduced for {k}: {out['checks'][f'route_P_{k}']}")
    zs = own_noise(test)
    out["own_noise"] = noise_block(zs)
    for tag in TAGS:
        out["units"][tag]["noise_p_own_draws"] = float((1 + (zs >= out["units"][tag]["Z"]).sum()) / (1 + len(zs)))
    place = {}
    for j, k in enumerate(names, start=len(TAGS)):
        s, d = slopes[j]
        g = gains[j]
        place[k] = {"code": CODE.get(k), "definition": DEFINITIONS.get(k, "the registered stand-in with the largest Z, read from the saved panel: a check of this route"),
                    "form": "normal scores within the pool, no hub correction" if k == "gbm_score" else "hub-corrected, normal scores within the pool (as the unit rate)",
                    **s, "noise_p_registered_draws": NS.noise_p(s["Z"]), "noise_p_own_draws": float((1 + (zs >= s["Z"]).sum()) / (1 + len(zs))), "gain": g,
                    "Z_above_the_stand_ins_largest_2.228": bool(s["Z"] > STANDINS_Z), "Z_above_the_noise_97.5th_percentile_2.340": bool(s["Z"] > NOISE_975),
                    "Z_at_or_above_the_units": bool(s["Z"] >= out["units"][MAIN]["Z"]),
                    "correlation_over_families_of_its_family_sums_with_the_units_family_sums": {tag: float(np.corrcoef(d, fam[tag])[0, 1]) for tag in TAGS}}
        log(f"in the units' place: {k}:" + (" (a dry run: rows shuffled)" if DRY else "") + f" T {s['T']:.3f} Z {s['Z']:.3f} p {s['p']:.5f} gain {g['gain']:.3f}")
    out["in_the_units_place"] = place
    out["seconds"], out["peak_memory_gb"] = round(time.time() - t0, 1), peak_gb()
    del test, U
    gc.collect()
    return out


def gain_job(j):
    """In a forked worker: the gain of the model refitted with column j of the block the parent left in G."""
    return R.gain_brief(G["test"].gain(G["U"][:, j]))


def run_spec(spec):
    """In a forked worker: one analysis of (A)."""
    title, key, what, extra_cols, gains, check = spec
    return title, analyse(key, what, extra_cols, gains=gains, check=check)


def analyse(key, what, extra_cols, gains=(MAIN,), check=None):
    def fn():
        t0 = time.time()
        test = build(extra_cols)
        S = test.S
        out = {"what": what, "exploratory": EXPLORATORY, "entries": N, "families": int(S.n_fam), "cells": int(S.n),
               "word_columns_with_the_cube": int(test.X.shape[1]), "added_columns": [f"{m}.{f}" for m, f in extra_cols],
               "forms": "z: standardised within the pool; hz: hub-corrected over the entries, then standardised within the pool",
               "weights_of_the_added_columns_fitted_on_all_entries": {f"{m}.{f}": float(test.beta_all[len(COLS) + j]) for j, (m, f) in enumerate(extra_cols)},
               "word_model": {"effective_pairs_left_open": float(test.unsolved), "sources_ranked_first": int((test.rank0 == 1).sum()),
                              "held_out_log_likelihood": float(test.ll0.sum()), "seconds_to_fit": round(time.time() - t0, 1)},
               "units": {}}
        log(f"{key}: fitted, {test.X.shape[1]} columns, open {test.unsolved:.2f}, sources first {int((test.rank0 == 1).sum())}")
        U = np.column_stack([S.rank_column(HUB[tag]) for tag in TAGS])
        Ur = test.slope(U)
        for j, tag in enumerate(TAGS):
            blk, _ = units_block(test, Ur[:, j], tag)
            if tag in gains:
                blk["gain"] = R.gain_brief(test.gain(U[:, j]))
            blk["noise_p_registered_draws_indicative"] = NS.noise_p(blk["Z"])
            blk["Z_above_the_stand_ins_largest_2.228"] = bool(blk["Z"] > STANDINS_Z)
            blk["Z_above_the_registered_noise_97.5th_percentile_2.340"] = bool(blk["Z"] > NOISE_975)
            out["units"][tag] = blk
            log(f"{key}: {tag} T {blk['T']:.3f} Z {blk['Z']:.3f} p {blk['p']:.5f} gain {blk.get('gain', {}).get('gain')}")
        zs = own_noise(test)
        out["own_noise"] = noise_block(zs)
        for tag in TAGS:
            blk = out["units"][tag]
            blk["noise_p_own_draws"] = float((1 + (zs >= blk["Z"]).sum()) / (1 + len(zs)))
            blk["Z_above_the_own_noise_97.5th_percentile"] = bool(blk["Z"] > np.quantile(zs, 0.975))
        if check is not None and not DRY:
            got = out["units"][MAIN]
            ok = abs(got["Z"] - check["Z"]) < 5e-5 and abs(got["T"] - check["T"]) < 5e-4 and abs(got["gain"]["gain"] - check["gain"]) < 5e-4 and abs(test.unsolved - check["open"]) < 5e-4
            out["check"] = {"registered": check, "reproduced": bool(ok)}
            if not ok:
                raise SystemExit(f"route A is not reproduced: {got['Z']}, {got['T']}, {got['gain']['gain']}, {test.unsolved} against {check}")
        out["seconds"], out["peak_memory_gb"] = round(time.time() - t0, 1), peak_gb()
        log(f"{key}: own noise sd {zs.std():.3f}, 97.5th {np.quantile(zs, 0.975):.3f}; done in {out['seconds']} s")
        del test, U
        gc.collect()
        return out
    return part(key, fn)


def both(names):
    return [(m, f) for m in names for f in ("z", "hz")]


PERM = np.arange(N)                                                           # a dry run shuffles the rows of every word-only measure with it


def standins_under(spec):
    """Second round, in a forked worker: the registered stand-ins (read from the saved panel, one at a time) and other measures
    in the units' place under the word model with extra_cols added; the units beside them."""
    title, key, what, extra_cols, others, want = spec

    def fn():
        t0 = time.time()
        test = build(extra_cols)
        S = test.S
        Zp = np.load(R.primary_panel_file(R.CACHE))
        assert np.array_equal(Zp["_cols"], IND) and np.array_equal(Zp["_entry_remakes"][SUB], C.remakes[ROWS])
        panel = list(W["word_columns"]["panel"])
        names = panel + list(others)
        cols = [S.rank_column(HUB[tag]) for tag in TAGS]
        for k in names:
            cols.append(test.unit_column(full(Zp[k][SUB][PERM]) if k in Zp.files else measure(k), "rate_matrix"))
        U = np.column_stack(cols)
        del cols
        Ur = test.slope(U)
        rows = {k: summary(test, Ur[:, j])[0] for j, k in enumerate(list(TAGS) + names)}
        st = {k: {x: rows[k][x] for x in ("T", "Z", "p")} for k in panel}
        ps, top = np.array([v["p"] for v in st.values()]), max(st, key=lambda k: st[k]["Z"])
        out = {"what": what, "exploratory": EXPLORATORY, "word_columns_with_the_cube": int(test.X.shape[1]), "added_columns": [f"{m}.{f}" for m, f in extra_cols],
               "word_model": {"effective_pairs_left_open": float(test.unsolved), "sources_ranked_first": int((test.rank0 == 1).sum()), "held_out_log_likelihood": float(test.ll0.sum())},
               "units": {tag: dict({x: rows[tag][x] for x in ("T", "Z", "p")}, noise_p_registered_draws=NS.noise_p(rows[tag]["Z"]),
                               Z_above_every_registered_stand_in=bool(rows[tag]["Z"] > st[top]["Z"])) for tag in TAGS},
               "stand_ins": {"n": len(st), "below_0.05": int((ps < T.ALPHA).sum()), "share_below_0.05": float(np.mean(ps < T.ALPHA)), "below_0.01": int((ps < 0.01).sum()),
                             "at_most_10pct_below_0.05": bool(np.mean(ps < T.ALPHA) <= T.MAX_STANDIN_SHARE), "largest_Z": st[top]["Z"], "largest_Z_measure": top,
                             "the_five_largest": [[k, st[k]["Z"], st[k]["p"]] for k in sorted(st, key=lambda k: -st[k]["Z"])[:5]], "rows": st},
               "other_measures_in_the_units_place": {k: dict({x: rows[k][x] for x in ("T", "Z", "p")}, code=CODE.get(k), noise_p_registered_draws=NS.noise_p(rows[k]["Z"])) for k in others}}
        if want is not None and not DRY:
            got = {"units_Z": {tag: rows[tag]["Z"] for tag in TAGS}, **{k: (rows[k]["Z"] if k in rows else out["stand_ins"][k]) for k in want if k != "units_Z"}}
            ok = all(abs(got["units_Z"][tag] - want["units_Z"][tag]) < 5e-5 for tag in TAGS) and all(abs(got[k] - want[k]) < 5e-5 for k in want if k != "units_Z")
            out["check"] = {"wanted": want, "got": got, "reproduced": bool(ok)}
            if not ok:
                raise SystemExit(f"{key}: not reproduced: {out['check']}")
        out["seconds"], out["peak_memory_gb"] = round(time.time() - t0, 1), peak_gb()
        log(f"{key}: {test.X.shape[1]} columns; stand-ins below 0.05: {out['stand_ins']['below_0.05']} of {len(st)}, largest Z {st[top]['Z']:.3f} ({top}); units " +
            ", ".join(f"{tag} Z {rows[tag]['Z']:.3f}" for tag in TAGS) + "; others " + ", ".join(f"{k} {rows[k]['Z']:.2f}" for k in others))
        del test, U
        gc.collect()
        return out
    return title, part(key, fn)


CHECK_STANDIN = "stemmed_words_and_pairs"


def main():
    assert self_checks()
    t0 = time.time()
    keys = ["baseline", "A-E0"] + [f"A-{CODE[k]}" for k in ORDER] + ["A-S1", "A-S2", "A-K", "A-ALL", "A-ALL+", "X0", "X2a", "X2b"]
    read_back = [k for k in keys if PARTS and part_file(k).exists()]            # the blocks this invocation reads back and does not compute
    made, info = make_rivals()
    for k in ORDER:
        SM[k] = expand(made[k])
    del made
    SM["embedding_cosine"] = small(R.embedding_cosine(C, R.ACTS / "embedding.npy")[ROWS])
    below = sorted(k for k, v in W["analyses"]["primary | set"]["standins"]["rows"].items() if v["p"] < T.ALPHA)
    assert CHECK_STANDIN in below
    for k, v in R.load_primary_panel(C, W["sha256"]["standin_panel_primary.npz"], R.CACHE, below).items():
        SM[k] = small(v[SUB])
    gc.collect()
    description = {k: dict(describe(k), code=CODE[k], definition=DEFINITIONS[k], made_of=info[k]) for k in ORDER}
    description["embedding_cosine"] = dict(describe("embedding_cosine"), code="E0", definition=DEFINITIONS["embedding_cosine"])
    for k in ("cos_tfidf", "bm25"):
        description[f"{k} (a registered word column, for comparison)"] = describe(k)
    for k in below:
        description[f"{k} (a registered stand-in, for comparison)"] = describe(k)
    for k, v in description.items():
        log(f"(D) {k}: source first {v['source_strictly_above_every_candidate']} of {N}, top 10 {v['source_in_the_top_10']}, median rank {v['median_rank_of_the_source']}")
    seconds_rivals = round(time.time() - t0, 1)
    if DRY:                                                                    # a dry run sees no result: every rival's rows are shuffled over the entries
        PERM[:] = np.random.default_rng(1).permutation(N)
        for k in SM:
            SM[k] = SM[k][PERM]
    base = part("baseline", baseline_and_checks)
    if "gbm_score" not in CELL:                                                # the baseline was read back: S2's score and S1's products are made again (no analysis is run again)
        test = build()
        gbm, six = supervised(test)
        assert int((test.S.rank_of_source(gbm) == 1).sum()) == base["gbm_score"]["sources_ranked_first_by_the_held_out_score"]
        assert [test.names[j] for j in six] == base["products"]["the_six_columns"]
        del test
        gc.collect()
    prods = [k for k in CELL if k.startswith("prod:")]
    reg_emb = REG[MAIN]["analyses"]["primary | set"]["embedding"]["units_with_the_embedding_among_the_word_columns"]
    one = (MAIN,)
    check_spec = ("check of route A: the registered embedding added (the plan's registered secondary)", "A-E0",
                  "the registered embedding cosine added to the word columns in both forms: must reproduce the registered secondary", both(["embedding_cosine"]), one,
                  {"Z": reg_emb["slope"]["Z"], "T": reg_emb["slope"]["T"], "gain": reg_emb["gain"]["gain"], "open": reg_emb["slope"]["effective_pairs_left_open"]})
    specs = []
    for k in ORDER[:12]:
        specs.append((f"A-{CODE[k]} {k}", f"A-{CODE[k]}", f"{DEFINITIONS[k]}; added to the word columns in both forms", both([k]), one, None))
    specs.append(("A-S1 products", "A-S1", "the 15 pairwise products of the six word columns of largest absolute weight (" + ", ".join(base["products"]["the_six_columns"]) +
                  "), each standardised within the pool", [(k, "z") for k in prods], one, None))
    specs.append(("A-S2 gbm_score", "A-S2", f"{DEFINITIONS['gbm_score']}; added as one column standardised within the pool", [("gbm_score", "z")], one, None))
    for k in ORDER[12:]:
        specs.append((f"A-{CODE[k]} {k}", f"A-{CODE[k]}", f"{DEFINITIONS[k]}; added to the word columns in both forms", both([k]), one, None))
    specs.append(("A-K the registered stand-ins below 0.05", "A-K", "the stand-ins of the registered panel with p below 0.05 in the units' place (" + ", ".join(below) +
                  "), added to the word columns in both forms", both(below), one, None))
    all_cols = both(ORDER) + [("gbm_score", "z")] + [(k, "z") for k in prods]
    big = [("A-ALL every rival at once", "A-ALL", "every rival of groups 1 to 6 added at once: the 15 measures in both forms, the gradient-boosted score and the 15 products", all_cols, TAGS, None),
           ("A-ALL+ every rival, the registered embedding and the stand-ins below 0.05", "A-ALL+",
            "A-ALL plus the registered embedding cosine in both forms and the registered stand-ins with p below 0.05 in both forms", all_cols + both(["embedding_cosine"]) + both(below), TAGS, None)]
    done = {}
    with get_context("fork").Pool(1) as pool:                                  # the check of route A comes before anything new
        done.update(pool.map(run_spec, [check_spec]))
    if STOP_AFTER_CHECKS:
        log("the three reproductions hold; stopped as asked")
        return
    with get_context("fork").Pool(WORKERS) as pool:
        for title, res in pool.imap_unordered(run_spec, specs, chunksize=1):
            done[title] = res
    for spec in big:                                                           # one at a time: each holds about 100 columns
        with get_context("fork").Pool(1) as pool:
            done.update(pool.map(run_spec, [spec]))
    analyses = {spec[0]: done[spec[0]] for spec in [check_spec] + specs + big}

    # ------------------------------------------------------------------ second round
    wst = W["analyses"]["primary | set"]["standins"]
    nf = W["primary_calibration"]["names_free_measure_not_counted_among_the_stand_ins"]
    t1 = "stemmed_words_and_pairs"
    x_specs = [("X0 the saved panel under the registered 39 columns", "X0", "the 74 registered stand-ins and the plan's names-free measure in the units' place, the 39 registered columns: a reproduction",
                [], [R.NAMES_FREE], {"units_Z": {tag: base["units"][tag]["Z"] for tag in TAGS}, R.NAMES_FREE: nf["Z"], "largest_Z": wst["summary"]["z_max"], t1: wst["rows"][t1]["Z"],
                                     "below_0.05": wst["summary"]["below_0.05"]}),
               ("X2a conditions 2 and 5 under the word model of A-T1", "X2a", "the 74 registered stand-ins and the measures that are not columns in the units' place; word model: the 39 columns + "
                "stems_no_names_tfidf in both forms", both(["stems_no_names_tfidf"]), [k for k in ORDER if k != "stems_no_names_tfidf"] + ["embedding_cosine", R.NAMES_FREE],
                {"units_Z": {tag: analyses["A-T1 stems_no_names_tfidf"]["units"][tag]["Z"] for tag in TAGS}})]
    x_big = ("X2b conditions 2 and 5 under the word model of A-ALL", "X2b", "the 74 registered stand-ins, the registered embedding cosine and the plan's names-free measure in the units' place; "
             "word model: the 85 columns of A-ALL", all_cols, ["embedding_cosine", R.NAMES_FREE], {"units_Z": {tag: analyses["A-ALL every rival at once"]["units"][tag]["Z"] for tag in TAGS}})
    second = {}
    with get_context("fork").Pool(WORKERS) as pool:
        for title, res in pool.imap_unordered(standins_under, x_specs, chunksize=1):
            second[title] = res
    with get_context("fork").Pool(1) as pool:
        second.update(pool.map(standins_under, [x_big]))
    second = {spec[0]: second[spec[0]] for spec in x_specs + [x_big]}
    second["plan_names_free_vocabulary_beside_N2"] = info["plan_names_free_vocabulary"]

    # ------------------------------------------------------------------ the table and the file
    table = [{"analysis": "baseline (registered test reproduced)", "added_columns": 0, "columns": 39, "effective_pairs_left_open": base["word_model"]["effective_pairs_left_open"],
              **{f"{tag}: {x}": base["units"][tag][x] for tag in TAGS for x in ("T", "Z", "p")},
              **{f"{tag}: noise_p_registered_draws": base["units"][tag]["noise_p_registered_draws"] for tag in TAGS},
              **{f"{tag}: noise_p_own_draws": base["units"][tag]["noise_p_own_draws"] for tag in TAGS},
              **{f"{tag}: gain": base["units"][tag]["gain"]["gain"] for tag in TAGS}}]
    for k, a in analyses.items():
        row = {"analysis": k, "added_columns": len(a["added_columns"]), "columns": a["word_columns_with_the_cube"], "effective_pairs_left_open": a["word_model"]["effective_pairs_left_open"],
               "own_noise_sd_Z": a["own_noise"]["sd_Z"], "own_noise_97.5th_percentile_Z": a["own_noise"]["Z_quantiles"]["0.975"]}
        for tag in TAGS:
            u = a["units"][tag]
            row.update({f"{tag}: T": u["T"], f"{tag}: Z": u["Z"], f"{tag}: p": u["p"], f"{tag}: noise_p_registered_draws_indicative": u["noise_p_registered_draws_indicative"],
                        f"{tag}: noise_p_own_draws": u["noise_p_own_draws"], f"{tag}: gain": u.get("gain", {}).get("gain"),
                        f"{tag}: Z_without_the_family_of_largest_sum": u["without_the_family_of_largest_sum"]["Z"], f"{tag}: Z_over_entries_not_recognised": u["entries_the_model_does_not_recognise"]["Z"]})
        table.append(row)
    place = base["in_the_units_place"]
    table_place = [{"measure": k, "code": v["code"], "T": v["T"], "Z": v["Z"], "p": v["p"], "noise_p_registered_draws": v["noise_p_registered_draws"], "gain": v["gain"]["gain"],
                    "effective_pairs_left_open": v["effective_pairs_left_open"], "sources_ranked_first_by_the_measure_alone": description.get(k, description.get(f"{k} (a registered stand-in, for comparison)", {})).get("source_strictly_above_every_candidate")}
                   for k, v in place.items()]
    rivals_only = {k: v for k, v in place.items() if v["code"] not in (None, "E0")}
    closest = max(rivals_only, key=lambda k: rivals_only[k]["Z"])
    singles = {k: a for k, a in analyses.items() if not k.startswith("check")}
    lowest = min(singles, key=lambda k: singles[k]["units"][MAIN]["Z"])
    headline = {
        "units_registered": {tag: {x: base["units"][tag][x] for x in ("T", "Z", "p", "noise_p_registered_draws")} | {"gain": base["units"][tag]["gain"]["gain"]} for tag in TAGS},
        "the_rival_closest_to_the_units_in_the_units_place": {"measure": closest, "Z": rivals_only[closest]["Z"], "p": rivals_only[closest]["p"], "gain": rivals_only[closest]["gain"]["gain"]},
        "rivals_in_the_units_place_with_Z_above_2.228": [k for k, v in rivals_only.items() if v["Z"] > STANDINS_Z],
        "rivals_in_the_units_place_with_Z_at_or_above_the_units_Z": [k for k, v in rivals_only.items() if v["Z"] >= base["units"][MAIN]["Z"]],
        "rivals_in_the_units_place_that_meet_conditions_3_4_5_of_the_plan_against_the_registered_panel": [
            k for k, v in rivals_only.items() if v["p"] < T.ALPHA and v["gain"]["gain"] > 0 and v["Z"] > STANDINS_Z],
        "rivals_in_the_units_place_with_Z_below_minus_2.228": [k for k, v in rivals_only.items() if v["Z"] < -STANDINS_Z],
        "rivals_in_the_units_place_with_p_below_0.05": [k for k, v in rivals_only.items() if v["p"] < T.ALPHA],
        "rivals_in_the_units_place_with_a_positive_gain": [k for k, v in rivals_only.items() if v["gain"]["gain"] > 0],
        "the_added_columns_that_lower_the_units_Z_most": {"analysis": lowest, **{tag: {"Z": singles[lowest]["units"][tag]["Z"], "p": singles[lowest]["units"][tag]["p"]} for tag in TAGS}},
        "analyses_where_the_units_Z_falls_below_2.228": {tag: [k for k, a in singles.items() if a["units"][tag]["Z"] < STANDINS_Z] for tag in TAGS},
        "analyses_where_the_units_Z_falls_below_2.340": {tag: [k for k, a in singles.items() if a["units"][tag]["Z"] < NOISE_975] for tag in TAGS},
        "analyses_where_the_units_Z_is_not_above_the_own_noise_97.5th_percentile": {tag: [k for k, a in singles.items() if not a["units"][tag]["Z_above_the_own_noise_97.5th_percentile"]] for tag in TAGS},
        "second_round_stand_ins_under_larger_word_models": {
            k: {"columns": v["word_columns_with_the_cube"], "stand_ins_below_0.05": v["stand_ins"]["below_0.05"], "stand_ins_share_below_0.05": v["stand_ins"]["share_below_0.05"],
                "stand_ins_largest_Z": v["stand_ins"]["largest_Z"], "stand_ins_largest_Z_measure": v["stand_ins"]["largest_Z_measure"],
                **{f"{tag}: Z": v["units"][tag]["Z"] for tag in TAGS}, **{f"{tag}: Z_above_every_registered_stand_in": v["units"][tag]["Z_above_every_registered_stand_in"] for tag in TAGS},
                "largest_Z_among_the_other_measures_in_the_units_place": max(((x["Z"], m) for m, x in v["other_measures_in_the_units_place"].items()), default=None)}
            for k, v in second.items() if k.startswith("X")},
        "share_of_the_units_registered_gain_that_is_left_with_every_rival_at_once": {
            name: {tag: analyses[key]["units"][tag]["gain"]["gain"] / base["units"][tag]["gain"]["gain"] for tag in TAGS}
            for name, key in (("A-ALL", "A-ALL every rival at once"), ("A-ALL+", "A-ALL+ every rival, the registered embedding and the stand-ins below 0.05"))},
        "units_Z_in_standard_deviations_of_the_analysis_own_noise": {
            "baseline": {tag: (base["units"][tag]["Z"] - base["own_noise"]["mean_Z"]) / base["own_noise"]["sd_Z"] for tag in TAGS},
            **{k: {tag: (a["units"][tag]["Z"] - a["own_noise"]["mean_Z"]) / a["own_noise"]["sd_Z"] for tag in TAGS} for k, a in singles.items()}},
        "with_every_rival_at_once": {name: {"columns": analyses[key]["word_columns_with_the_cube"], "effective_pairs_left_open": analyses[key]["word_model"]["effective_pairs_left_open"],
                                            "own_noise_sd_Z": analyses[key]["own_noise"]["sd_Z"], "own_noise_97.5th_percentile_Z": analyses[key]["own_noise"]["Z_quantiles"]["0.975"],
                                            **{tag: {x: analyses[key]["units"][tag][x] for x in ("T", "Z", "p", "noise_p_own_draws", "noise_p_registered_draws_indicative")} |
                                               {"gain": analyses[key]["units"][tag]["gain"]["gain"]} for tag in TAGS}}
                                     for name, key in (("A-ALL", "A-ALL every rival at once"), ("A-ALL+", "A-ALL+ every rival, the registered embedding and the stand-ins below 0.05"))}}
    blocks = [base] + list(analyses.values()) + [v for k, v in second.items() if k.startswith("X")]
    out = {"what": "Word-rivals audit of the remakes decision test: word-only measures a hostile referee would say were left out of the 39 word columns, each put in the units' place and each "
                   "added to the word columns. Made by results/v2/second/audit/word_rivals.py. Every number unrounded. " + EXPLORATORY + ".",
           "exploratory": EXPLORATORY,
           "how_to_read": {"T": "the statistic of the registered test: the sum over families of the family sums of the entries' contributions",
                           "Z": "T divided by the root of the sum of the squared family sums",
                           "p": "the registered one-sided sign-flip p of the family sums (100,000 flips, seed 20261108); known to be too small on this list (PLAN Section 16, entry 3)",
                           "noise_p_registered_draws": "the share of the 2,000 registered pure-noise draws (remakes_null_size.json, the 39 columns) whose Z is at or above this Z, this one counted; "
                                                       "exact for anything put in the units' place with the word columns unchanged; 'indicative' where columns were added",
                           "noise_p_own_draws": "the same share among 300 noise draws put through the analysis's own word model (cells under 1e-08 left out)",
                           "gain": "the held-out log-likelihood of the true sources with the measure's column added to the word model, minus without it (remake_test.RemakeTest.gain)",
                           "effective_pairs_left_open": "the sum over entries of one minus the held-out word probability of the true source",
                           "thresholds": {"the units' registered Z": base["units"][MAIN]["Z"], "the largest Z of the 74 registered stand-ins": STANDINS_Z,
                                          "the 97.5th percentile of the registered noise Z": NOISE_975},
                           "condition_5": "the stand-in panel is not rebuilt; in the second round the 74 registered stand-ins are read from the saved panel and put in the units' place "
                                          "under the word models of A-T1 and A-ALL, so conditions 2 and 5 of the plan are re-evaluated for those two models and for no other"},
           "notes": ["The list of analyses in the script's docstring, the word classes and every definition were written before any of them was run.",
                     "Before the real run the code was tried on every sixth entry (37 entries) with random numbers in the units' place and every rival's rows shuffled over the entries "
                     "(AUDIT_DRY=1); that trial showed no result. The words-only description (D) of each rival was seen on those 37 entries before the real run.",
                     "The second round (X0, X2a, X2b) was added after part of the first round had been seen; its definition, as written at that moment, is in the script's docstring. "
                     "The first round was not run again: its blocks were read back from the folder AUDIT_PARTS, and X2a and X2b check that they rebuild the same word models "
                     "(they must give the units the Z of A-T1 and of A-ALL).",
                     "After the trial and before the real run, the dynamic programme of O1 and O2, the position correlation of O3 and the name tokens were checked against direct "
                     "computations on words alone; in that check the values of O1, O2 and O3 for the documented sources of six entries were seen. No definition was changed after it.",
                     "With 300 noise draws a noise_p_own_draws cannot be below 1/301 = 0.0033; with the 2,000 registered draws the floor is 0.0005.",
                     "S1's six columns are chosen by their weights in the model fitted on all entries, so the choice (not the weights of the products) has seen every family's label; "
                     "S2's score for a family comes from a model that never saw that family.",
                     "Rivals in the units' place are " + str(len(rivals_only)) + " word-only measures chosen by one reviewer; on this list pure noise has p below 0.05 in 10.9% of draws."],
           "dry_run_on_random_numbers": DRY, "git_head": R.git_head(),
           "sha256": {"pairs.json": C.sha["pairs.json"], "films.jsonl": C.sha["films.jsonl"], "remakes.py": R.sha256("v2/second/remakes.py"), "remake_test.py": R.sha256("v2/second/remake_test.py"),
                      "word_rivals.py": R.sha256(__file__), "remakes_words.json": R.sha256(R.OUT / R.WORDS_FILE), "embedding.npy": R.sha256(R.ACTS / "embedding.npy"),
                      **{f"remakes_{tag}.json": R.sha256(R.OUT / f"remakes_{tag}.json") for tag in TAGS},
                      **{f"{tag}/counts/{R.PRIMARY_PLANE}.npz": R.sha256(R.ACTS / tag / "counts" / f"{R.PRIMARY_PLANE}.npz") for tag in TAGS}},
           "settings": {"models": list(TAGS), "plane": R.PRIMARY_PLANE, "set": "primary", "frame": "set", "penalty": T.LAMBDA, "sign_flips": T.N_FLIPS, "seed": T.SEED,
                        "noise_draws_per_analysis": NOISE_DRAWS, "first_noise_seed": NOISE_SEED, "gbm_seed": GBM_SEED, "gbm_folds": GBM_FOLDS,
                        "worker_processes": WORKERS, "threads_per_process": 1,
                        "films_read": C.N, "films_of_the_indian_industries": int(len(IND)), "word_classes": CLASSES,
                        "scikit_learn": __import__("sklearn").__version__, "numpy": np.__version__, "scipy": __import__("scipy").__version__},
           "definitions": DEFINITIONS, "headline": headline,
           "table_in_the_units_place": table_place, "table_added_to_the_word_columns": table,
           "rivals_described_by_words_alone": description, "baseline": base, "analyses": analyses, "second_round": second,
           "seconds_to_make_the_rivals": seconds_rivals, "seconds_of_the_blocks_summed": round(sum(b["seconds"] for b in blocks), 1),
           "peak_memory_gb_of_the_largest_block": max(b["peak_memory_gb"] for b in blocks),
           "seconds_of_this_invocation": round(time.time() - T0, 1), "peak_memory_gb_of_this_invocation_parent_process": peak_gb(),
           "blocks_read_back_in_this_invocation": read_back}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=R.plain, ensure_ascii=False))
    log("wrote", OUT)


if __name__ == "__main__":
    main()
