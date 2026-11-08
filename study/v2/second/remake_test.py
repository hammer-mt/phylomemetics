"""Do the units say which film is the source beyond what word measures say?  A held-out conditional-logit test.

The test of v2/second/PLAN.md Section 9 for the remakes corpus. One file, numpy + scipy + scikit-learn only. Nothing
here reads a model output: the unit measure is an input. The driver is v2/second/remakes.py. Everything below the
imports is the designers' file as it was calibrated, unchanged (same constants, same functions).

THE TEST IN ONE PARAGRAPH
Each remake r has a pool of candidate films, one of which is its documented source s. A conditional logit over the
pool, P(c is the source of r) proportional to exp(beta . x(r, c)), is fitted on 31 word-only columns x with a ridge
penalty, leaving one source family out at a time. For the left-out family the word model gives every candidate a
probability p(c | r). The statistic is the slope, at zero, of the left-out log-probability of the true sources in
the weight of the unit column, the word weights being free to adjust:

    U_r = e(r, s) - sum_c p(c | r) e(r, c),        e = u - x . A,

where u is the unit column (within each pool, the normal score of the rank of the hub-corrected unit rate) and A is
the ridge regression of u on the word columns among the OTHER families' cells, each cell weighted by p. In words:
the unit similarity of the true source, with what the word columns explain taken out, minus the same quantity
averaged over the candidates the words find plausible. U_r is summed within source families (D_f) and over families
(T). A remake whose source the words already find with certainty has p(s | r) = 1 and U_r = 0: it cannot speak,
whatever the units say. The p value is one-sided and comes from flipping the signs of the family sums D_f. The
held-out log-likelihood gain of the model refitted with the unit column (the starting design) is computed beside it
and must be positive for a pass, and the units must beat every word-only stand-in of a registered panel (decide).

THE FRAME (fixed in PLAN.md Section 9; built by remakes.py, not here)
Films read: every film of v2/cache/second/remakes/films.jsonl. An entry is one documented pair: a remake with two
documented sources is entered twice (two entries of one family, each with its own source in its pool). An entry's pool
is its source plus its candidates; a candidate is never later than the remake and never a film of the remake's family.
Primary frame ("set"): the candidates are the films that the definition of the set allows; for the primary set, the
firm Indian cross-industry pairs, these are the films of the Indian industries other than the remake's own. Secondary
frames, for the primary set only: "source_industry" (films of the source's industry) and "all" (every film read). A
family is a set of films joined by remake statements; it is the unit of clustering. Sets of pairs are analysed one at
a time, each with its own fit, its own hub reference (the other entries of the same set) and its own stand-ins.

USE
    measures = word_measures(texts, remake_rows)                  # dict name -> [n_remakes, n_films]
    test = RemakeTest(cells, measures, film_family)               # fits the words-only model, leave one family out
    result = test.run(unit)                                       # unit: [n_remakes, n_films] containment rates
    panel = standin_measures(texts, remake_rows)                  # word-only measures that are not covariates
    report = decide(test, unit, panel)                            # the registered decision
(test.run(unit, form="hub_z_per_pool") takes instead one array per remake over its pool, already hub-corrected.)

cells: list, one per entry, in the order of remake_rows (a remake entered twice has two rows): {"pool": array of film
rows, "source": film row, "family": hashable}. film_family: array over the films read, the family id of each film that belongs to a study
family (remakes and sources), anything else (None or "") otherwise. texts: one string per film read (its cleaned
sentences, at most 40, joined by one space). word_measures and standin_measures can be replaced by any dict of
[n_remakes, n_films] arrays with the same keys.

CALIBRATION
The test was designed and calibrated on word data only, on a recalled list of 95 pairs with candidates of the
source's industry (preregistration/second_corpus/designs/logit.md). Its calibration on the final list and frame is
`remakes.py words` -> results/v2/second/remakes_words.json: the word model's held-out result, the 78 stand-ins in
the units' place, the withheld-vocabulary positive control, the planted-boost power and the noisy and curved copies
of the covariates, all computed and committed before any unit measure is joined. Run times are in that file.
"""
import re

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm, rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

# ----------------------------------------------------------------------------------------------- constants (frozen)
LAMBDA = 3.0                    # ridge penalty on every column (columns have unit variance within every pool)
ALPHA = 0.05                    # one-sided
N_FLIPS = 100_000               # sign-flip draws when the families are too many to enumerate
EXACT_UP_TO = 20                # enumerate all 2**n sign patterns when at most this many families have a non-zero sum
SEED = 20261108
MIN_UNSOLVED = 5.0              # the test is evaluable only if the words leave at least this many "effective pairs" open
MAX_STANDIN_SHARE = 0.10        # more stand-ins than this below ALPHA: the word covariates do not span words on this list
MAX_SENTENCES, MIN_SENTENCES = 40, 15
BAND_EDGES = (0.002, 0.01, 0.05, 0.5)       # frequency bands of the containment columns, as shares of the films read
BM25_K1, BM25_B = 1.5, 0.75
COSINES = ["cos_tfidf", "cos_rawtf", "cos_tf", "cos_binidf", "bm25", "cos_bigram", "cos_skip3", "cos_char4", "lsa150"]
LSA_DIMS, LSA_SEED, SKIP_WINDOW = 150, 2, 3
CONTAINMENTS = ["contain_opt", "contain_band1", "contain_band2", "contain_band3", "contain_band4"]
# every word measure enters twice: standardised within the pool ("z"), and as its hub-corrected z standardised within
# the pool ("hz"); two length columns enter once
WORD_COLUMNS = [(m, f) for m in COSINES + CONTAINMENTS for f in ("z", "hz")] + [("cand_log_words", "z"), ("len_match", "z")]

ABBREVS = ("Mr.", "Mrs.", "Dr.", "St.", "Jr.", "Sr.", "Ms.", "Lt.", "Sgt.", "vs.", "U.S.", "R.I.P.", "Gen.", "Capt.", "Det.")
SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”’)]))\s+(?=[A-Z\"“‘(])")


# ------------------------------------------------------------------------------------------------------------ texts
def sentences(body):
    """study/fetch_corpus.py sentences(), copied so that this file stands alone."""
    body = re.sub(r"\s+", " ", body).strip()
    merged = []
    for piece in SPLIT.split(body):
        if merged and merged[-1].endswith(ABBREVS):
            merged[-1] += " " + piece
        else:
            merged.append(piece)
    out = [s.strip() for s in merged if len(s.strip()) > 15]
    return out[:MAX_SENTENCES]


def plot_text(plot):
    """The text of a film as every measure here reads it: its first 40 sentences joined by one space."""
    return " ".join(sentences(plot))


# ---------------------------------------------------------------------------------------------- word-only measures
def _dense(x):
    return np.asarray(x.todense() if sp.issparse(x) else x, dtype=np.float64)


def _cosine(vec, texts, R):
    A = vec.fit_transform(texts)
    return _dense(A[R] @ A.T)


def _presence(texts, stop="english", min_df=2):
    cv = CountVectorizer(stop_words=stop, min_df=min_df, binary=True)
    P = cv.fit_transform(texts).astype(np.float32).tocsr()
    return P, np.asarray(P.sum(0)).ravel(), cv


def _containment(P, R, mask=None, weights=None):
    """Share of the candidate's words (in the mask, optionally weighted) that the remake also has."""
    w = np.ones(P.shape[1], np.float32) if weights is None else np.asarray(weights, np.float32)
    if mask is not None:
        w = w * mask
    Pw = P @ sp.diags(w)
    return _dense(P[R] @ Pw.T) / np.maximum(np.asarray(Pw.sum(1)).ravel(), 1e-9)[None, :]


def _bm25(texts, R, k1=BM25_K1, b=BM25_B, stop="english", query_tf=False):
    T = CountVectorizer(stop_words=stop, min_df=2).fit_transform(texts).astype(np.float64).tocsr()
    N = T.shape[0]
    df = np.asarray((T > 0).sum(0)).ravel()
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    length = np.asarray(T.sum(1)).ravel()
    norm = k1 * (1 - b + b * length / length.mean())
    D = T.tocoo()
    Wd = sp.csr_matrix((D.data * (k1 + 1) / (D.data + norm[D.row]), (D.row, D.col)), shape=T.shape)
    Q = (T[R] if query_tf else (T[R] > 0).astype(np.float64)) @ sp.diags(idf)
    return _dense(Q @ Wd.T)


def band_masks(df, n_films):
    """Four bands of document frequency among the films read: [2, 0.2%), [0.2%, 1%), [1%, 5%), [5%, 50%]."""
    e = [2.0] + [x * n_films for x in BAND_EDGES]
    return [(df >= e[b]) & ((df < e[b + 1]) if b < 3 else (df <= e[b + 1])) for b in range(4)]


def word_measures(texts, remake_rows):
    """The fourteen word measures and two length measures of the test, each as [n_remakes, n_films].

    texts: one string per film read. remake_rows: the rows of texts that are remakes. Words are scikit-learn's
    default tokens (lower case, two or more letters or digits) with its English stop list removed, kept if they are in
    at least 2 of the films read. Document frequencies and idf come from the films read.

      cos_tfidf    cosine of TF-IDF vectors, logarithm of counts (the paper's first measure)
      cos_rawtf    the same on raw counts (the paper's seventh)
      cos_tf       cosine of the logarithm of counts with no idf
      cos_binidf   cosine of idf-weighted presence vectors (counts ignored)
      bm25         Okapi BM25 of the candidate for the remake's distinct words, k1 = 1.5, b = 0.75
      cos_bigram   TF-IDF cosine over pairs of adjacent words (stop words removed first)
      cos_skip3    TF-IDF cosine over ordered pairs of words at most 3 apart (stop words removed first)
      cos_char4    TF-IDF cosine over 4-character sequences inside words
      lsa150       cosine in a 150-dimension latent semantic space (truncated SVD, seed 2, of the TF-IDF matrix of
                   words and adjacent pairs): a word-only measure of meaning
      contain_opt  share of the candidate's optional words that the remake has (optional: in at least 2 and at most
                   half of the films read): the word twin of the unit measure
      contain_band1..4   the same within four bands of document frequency (band_masks)
      cand_log_words     log of the candidate's number of optional words (the same for every remake)
      len_match          minus the absolute difference of that log between candidate and remake
    """
    R = np.asarray(remake_rows)
    N = len(texts)
    M = {}
    M["cos_tfidf"] = _cosine(TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True), texts, R)
    M["cos_rawtf"] = _cosine(TfidfVectorizer(stop_words="english", min_df=2), texts, R)
    M["cos_tf"] = _cosine(TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True, use_idf=False), texts, R)
    M["cos_binidf"] = _cosine(TfidfVectorizer(stop_words="english", min_df=2, binary=True), texts, R)
    M["bm25"] = _bm25(texts, R)
    M["cos_bigram"] = _cosine(TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True, ngram_range=(2, 2)), texts, R)
    M["cos_skip3"] = _cosine(TfidfVectorizer(analyzer=_skip_analyzer(SKIP_WINDOW), min_df=2, sublinear_tf=True), texts, R)
    M["cos_char4"] = _cosine(TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=2, sublinear_tf=True), texts, R)
    M["lsa150"] = _lsa(texts, R, LSA_DIMS, LSA_SEED, ngram=(1, 2))
    P, df, _ = _presence(texts)
    opt = (df >= 2) & (df <= N / 2)
    M["contain_opt"] = _containment(P, R, opt)
    for b, m in enumerate(band_masks(df, N)):
        M[f"contain_band{b + 1}"] = _containment(P, R, m)
    n_opt = np.log(np.maximum(np.asarray(P[:, opt].sum(1)).ravel(), 1))
    M["cand_log_words"] = np.tile(n_opt, (len(R), 1))
    M["len_match"] = -np.abs(n_opt[None, :] - n_opt[R][:, None])
    return {k: v.astype(np.float32) for k, v in M.items()}


# ----------------------------------------------------------- stand-ins: word-only measures that are NOT covariates
def _jaccard(P, R, mask=None, kind="jaccard", alpha=0.5):
    Pm = P if mask is None else P @ sp.diags(mask.astype(np.float32))
    inter = _dense(Pm[R] @ Pm.T)
    n = np.asarray(Pm.sum(1)).ravel()
    a, b = n[R][:, None], n[None, :]
    if kind == "jaccard":
        return inter / np.maximum(a + b - inter, 1e-9)
    if kind == "dice":
        return 2 * inter / np.maximum(a + b, 1e-9)
    if kind == "overlap":
        return inter / np.maximum(np.minimum(a, b), 1e-9)
    if kind == "ochiai":
        return inter / np.maximum(np.sqrt(a * b), 1e-9)
    if kind == "tversky":                                    # alpha on the remake's side
        return inter / np.maximum(inter + alpha * (a - inter) + (1 - alpha) * (b - inter), 1e-9)
    if kind == "reverse":                                    # share of the remake's words that the candidate has
        return inter / np.maximum(a, 1e-9)
    raise KeyError(kind)


def _lsa(texts, R, k, seed, ngram=(1, 1)):
    A = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True, ngram_range=ngram).fit_transform(texts)
    Z = normalize(TruncatedSVD(k, random_state=seed).fit_transform(A))
    return Z[R] @ Z.T


def _shingles(texts, R, n):
    """Share of the remake's n-word sequences found in the candidate (the scouts' copy measure when n = 5)."""
    tok = [re.findall(r"\w+", t.lower()) for t in texts]
    vocab, rows, cols = {}, [], []
    for i, w in enumerate(tok):
        for j in range(len(w) - n + 1):
            rows.append(i)
            cols.append(vocab.setdefault(" ".join(w[j:j + n]), len(vocab)))
    S = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(texts), max(len(vocab), 1)))
    S.sum_duplicates()
    S.data[:] = 1
    return _dense(S[R] @ S.T) / np.maximum(np.asarray(S[R].sum(1)).ravel(), 1)[:, None]


_CAP = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b")
_SUFFIXES = ("ingly", "edly", "ations", "ation", "ments", "ment", "ness", "ings", "ing", "ies", "ied", "ers", "est", "ed", "es", "er", "ly", "s")


def _stem(w):
    for s in _SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 3:
            return w[: len(w) - len(s)]
    return w


def _base_analyzer():
    return CountVectorizer(stop_words="english").build_analyzer()


def _stem_analyzer(pairs=False):
    base = _base_analyzer()

    def f(doc):
        w = [_stem(x) for x in base(doc)]
        return w + [a + " " + b for a, b in zip(w, w[1:])] if pairs else w
    return f


def _sentence_pair_analyzer():
    base = _base_analyzer()
    split = re.compile(r"(?<=[.!?])\s+")

    def f(doc):
        out = []
        for s in split.split(doc):
            w = sorted(set(base(s)))[:30]
            out.extend(a + "+" + b for i, a in enumerate(w) for b in w[i + 1:])
        return out
    return f


def _skip_analyzer(window=3):
    base = _base_analyzer()

    def f(doc):
        w = base(doc)
        return [a + ">" + b for i, a in enumerate(w) for b in w[i + 1:i + 1 + window]]
    return f


def _prefix_analyzer(n=5):
    base = _base_analyzer()
    return lambda doc: [w[:n] for w in base(doc)]


def _sentence_scores(sent_lists, R):
    """Three sentence-level measures: the mean over the candidate's sentences of its best TF-IDF cosine with a sentence
    of the remake; the single best sentence pair; the mean of the five best sentence pairs."""
    flat = [s for ss in sent_lists for s in ss]
    n_s = np.array([len(ss) for ss in sent_lists])
    owner = np.repeat(np.arange(len(sent_lists)), n_s)
    starts = np.concatenate([[0], np.cumsum(n_s)])
    A = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit_transform(flat).tocsr()
    align = np.zeros((len(R), len(sent_lists)))
    best1 = np.zeros_like(align)
    best5 = np.zeros_like(align)
    for i, r in enumerate(R):
        Sm = (A @ A[starts[r]:starts[r + 1]].T).tocsr()          # [all sentences, the remake's sentences]
        best = np.asarray(Sm.max(axis=1).todense()).ravel()      # each sentence's best match in the remake
        align[i] = np.bincount(owner, weights=best, minlength=len(sent_lists)) / np.maximum(n_s, 1)
        order = np.lexsort((-best, owner))
        rank = np.arange(len(best)) - starts[:-1][owner[order]]
        b_sorted = best[order]
        best1[i] = np.bincount(owner[order], weights=b_sorted * (rank < 1), minlength=len(sent_lists))
        best5[i] = np.bincount(owner[order], weights=b_sorted * (rank < 5), minlength=len(sent_lists)) / 5
    return align, best1, best5


def _divergences(texts, R, mu=200.0):
    """Three measures on word distributions: 1 - Jensen-Shannon divergence, the Bhattacharyya coefficient, and the
    query likelihood of the remake under the candidate's language model with Dirichlet smoothing."""
    T = CountVectorizer(stop_words="english", min_df=2).fit_transform(texts).astype(np.float64).tocsr()
    length = np.asarray(T.sum(1)).ravel()
    Pn = normalize(T, norm="l1").tocsr()
    coll = np.asarray(T.sum(0)).ravel() / T.sum()
    ent = -np.asarray(Pn.multiply(Pn._with_data(np.log2(Pn.data))).sum(1)).ravel()
    Sq = Pn._with_data(np.sqrt(Pn.data))
    bhat = _dense(Sq[R] @ Sq.T)
    js = np.zeros((len(R), T.shape[0]))
    ql = np.zeros_like(js)
    for i, r in enumerate(R):
        q = Pn[r]
        cols, qd = q.indices, q.data
        sub = Pn[:, cols].toarray()
        m = 0.5 * (sub + qd[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            h_in = -np.where(m > 0, m * np.log2(m), 0).sum(1)
            ent_in = -np.where(sub > 0, sub * np.log2(sub), 0).sum(1)
        h_out = 0.5 * (ent - ent_in) + 0.5 * (1 - sub.sum(1))
        js[i] = 1 - (h_in + h_out - 0.5 * ent - 0.5 * ent[r])
        cnt = T[:, cols].toarray()
        qc = T[r].data
        ql[i] = (qc[None, :] * np.log((cnt + mu * coll[cols][None, :]) / (length[:, None] + mu))).sum(1) / qc.sum()
    return js, bhat, ql


def standin_measures(texts, remake_rows, seed=SEED):
    """The registered panel of word-only measures that are not covariates, each as [n_remakes, n_films]. They are put
    through the test in the units' place on the final data (decide)."""
    R = np.asarray(remake_rows)
    N = len(texts)
    M = {}
    tv = lambda **k: TfidfVectorizer(**{"min_df": 2, "sublinear_tf": True, **k})          # noqa: E731
    P, df, cv = _presence(texts)
    vocab = cv.get_feature_names_out()
    opt = (df >= 2) & (df <= N / 2)
    idf = np.log(N / np.maximum(df, 1))
    rng = np.random.default_rng(seed)
    half_a = rng.random(len(df)) < 0.5
    half_b = rng.random(len(df)) < 0.5
    sents = [sentences(t) for t in texts]
    # single words, other weightings
    M["tfidf_stop_words_kept"] = _cosine(tv(), texts, R)
    M["tfidf_words_in_under_10pct"] = _cosine(tv(stop_words="english", max_df=0.1), texts, R)
    M["tfidf_min5_max30pct"] = _cosine(tv(stop_words="english", min_df=5, max_df=0.3), texts, R)
    M["tfidf_l1_norm"] = _cosine(tv(stop_words="english", norm="l1"), texts, R)
    M["tfidf_top5000_words"] = _cosine(tv(stop_words="english", min_df=3, max_features=5000), texts, R)
    M["tfidf_long_words"] = _cosine(tv(stop_words="english", token_pattern=r"(?u)\b\w{7,}\b"), texts, R)
    M["tfidf_short_words"] = _cosine(tv(stop_words="english", token_pattern=r"(?u)\b\w{2,5}\b"), texts, R)
    M["tfidf_rarest_half"] = _cosine(tv(stop_words="english", vocabulary=vocab[df <= np.median(df)]), texts, R)
    M["raw_counts_stop_kept_no_idf"] = _cosine(TfidfVectorizer(min_df=2, use_idf=False), texts, R)
    M["stop_word_profile"] = _cosine(TfidfVectorizer(vocabulary=sorted(ENGLISH_STOP_WORDS), use_idf=False), texts, R)
    M["bm25_k0.9_b0.4_stop_kept"] = _bm25(texts, R, 0.9, 0.4, stop=None)
    M["bm25_k2_b1"] = _bm25(texts, R, 2.0, 1.0)
    M["bm25_query_counts"] = _bm25(texts, R, 1.2, 0.75, query_tf=True)
    js, bhat, ql = _divergences(texts, R)
    M["jensen_shannon"], M["bhattacharyya"], M["query_likelihood"] = js, bhat, ql
    # presence, other bands, weights and normalisations
    e2 = [5, 0.005 * N, 0.02 * N, 0.2 * N]
    for b in range(3):
        M[f"contain_other_band{b + 1}"] = _containment(P, R, (df >= e2[b]) & (df < e2[b + 1]))
    e3 = np.quantile(df[opt], [0, 1 / 3, 2 / 3, 1])
    for b in range(3):
        M[f"contain_tertile{b + 1}"] = _containment(P, R, opt & (df >= e3[b]) & ((df < e3[b + 1]) if b < 2 else (df <= e3[b + 1])))
    for e in (0.25, 0.5, 1.0, 2.0):
        M[f"contain_weight_df^{e}"] = _containment(P, R, opt, weights=df.astype(np.float64) ** e)
    M["contain_weight_idf^2"] = _containment(P, R, opt, weights=idf ** 2)
    M["contain_random_half_a"] = _containment(P, R, opt & half_a)
    M["contain_random_half_b"] = _containment(P, R, opt & half_b)
    M["contain_a_to_l"] = _containment(P, R, opt & np.array([w[0] <= "l" for w in vocab]))
    Pst, dfst, _ = _presence(texts, stop=None)
    M["contain_stop_words_kept"] = _containment(Pst, R, dfst >= 2)
    T2 = CountVectorizer(stop_words="english", vocabulary=vocab).fit_transform(texts)
    M["contain_repeated_words"] = _containment((T2 >= 2).astype(np.float32).tocsr(), R, opt)
    M["tfidf_random_half_a"] = _cosine(tv(stop_words="english", vocabulary=vocab[half_a], min_df=1), texts, R)
    M["tfidf_random_half_b"] = _cosine(tv(stop_words="english", vocabulary=vocab[half_b], min_df=1), texts, R)
    M["jaccard_optional"] = _jaccard(P, R, opt)
    M["jaccard_rare_1pct"] = _jaccard(P, R, (df >= 2) & (df <= 0.01 * N))
    M["dice"] = _jaccard(P, R, opt, "dice")
    M["overlap_coefficient"] = _jaccard(P, R, opt, "overlap")
    M["ochiai"] = _jaccard(P, R, opt, "ochiai")
    M["tversky_0.8"] = _jaccard(P, R, opt, "tversky", 0.8)
    M["reverse_containment"] = _jaccard(P, R, opt, "reverse")
    M["shared_optional_words_log"] = np.log1p(_dense(P[R] @ (P @ sp.diags(opt.astype(np.float32))).T))
    M["idf_mass"] = _dense(P[R] @ (P @ sp.diags((idf * opt).astype(np.float32))).T)
    M["idf^2_mass"] = _dense(P[R] @ (P @ sp.diags((idf ** 2 * opt).astype(np.float32))).T)
    M["idf^1.5_mass_normalised"] = (lambda X, n: X / np.maximum(np.sqrt(n[R][:, None] * n[None, :]), 1e-9))(
        _dense(P[R] @ (P @ sp.diags((idf ** 1.5 * opt).astype(np.float32))).T), np.asarray((P @ sp.diags((idf ** 1.5 * opt).astype(np.float32))).sum(1)).ravel())
    for b, m in enumerate(band_masks(df, N)):                # logarithm of counts, no idf, within each band
        Tb = CountVectorizer(vocabulary=vocab).fit_transform(texts).astype(np.float64).tocsr()
        Tb.data = 1 + np.log(Tb.data)
        Tb = normalize(Tb @ sp.diags(m.astype(np.float64)))
        M[f"cos_tf_band{b + 1}"] = _dense(Tb[R] @ Tb.T)
    # sequences
    M["char3"] = _cosine(tv(analyzer="char_wb", ngram_range=(3, 3)), texts, R)
    M["char5"] = _cosine(tv(analyzer="char_wb", ngram_range=(5, 5)), texts, R)
    M["char2"] = _cosine(tv(analyzer="char_wb", ngram_range=(2, 2)), texts, R)
    M["char6_raw_counts"] = _cosine(TfidfVectorizer(analyzer="char", ngram_range=(6, 6), min_df=2), texts, R)
    M["char4_5_raw_counts"] = _cosine(TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), min_df=2), texts, R)
    M["words_and_pairs"] = _cosine(tv(stop_words="english", ngram_range=(1, 2)), texts, R)
    M["words_pairs_triples"] = _cosine(tv(stop_words="english", ngram_range=(1, 3)), texts, R)
    M["triples"] = _cosine(tv(stop_words="english", ngram_range=(3, 3)), texts, R)
    M["pairs_stop_words_kept"] = _cosine(tv(ngram_range=(2, 2)), texts, R)
    M["pairs_jaccard"] = _jaccard(CountVectorizer(stop_words="english", min_df=2, ngram_range=(2, 2), binary=True).fit_transform(texts).astype(np.float32).tocsr(), R)
    M["stop_word_pairs"] = _cosine(TfidfVectorizer(min_df=2, use_idf=False,
                                                   analyzer=lambda d, a=CountVectorizer().build_analyzer(): (lambda w: [x + " " + y for x, y in zip(w, w[1:])])([x for x in a(d) if x in ENGLISH_STOP_WORDS])), texts, R)
    M["skip_pairs_window6"] = _cosine(tv(analyzer=_skip_analyzer(6)), texts, R)
    M["copied_5_word_sequences"] = _shingles(texts, R, 5)
    M["copied_3_word_sequences"] = _shingles(texts, R, 3)
    # stems, names, sentences, parts of the text
    M["stemmed_tfidf"] = _cosine(tv(analyzer=_stem_analyzer()), texts, R)
    M["stemmed_words_and_pairs"] = _cosine(tv(analyzer=_stem_analyzer(True)), texts, R)
    M["five_letter_prefixes"] = _cosine(tv(analyzer=_prefix_analyzer(5)), texts, R)
    caps = CountVectorizer(lowercase=False, binary=True, token_pattern=r"\S+", min_df=2).fit_transform([" ".join(set(_CAP.findall(t))) for t in texts])
    M["capitalised_words_jaccard"] = _jaccard(caps.astype(np.float32).tocsr(), R)
    M["words_sharing_a_sentence"] = _cosine(tv(analyzer=_sentence_pair_analyzer()), texts, R)
    M["sentence_alignment"], M["best_sentence_pair"], M["five_best_sentence_pairs"] = _sentence_scores(sents, R)
    M["opening_15_sentences"] = _cosine(tv(stop_words="english"), [" ".join(s[:15]) for s in sents], R)
    M["after_sentence_15"] = _cosine(tv(stop_words="english"), [" ".join(s[15:]) or "none" for s in sents], R)
    # latent semantic analysis: a word-only measure of meaning
    M["lsa_50"] = _lsa(texts, R, 50, 1)
    M["lsa_100"] = _lsa(texts, R, 100, 0)
    M["lsa_300"] = _lsa(texts, R, 300, 0)
    M["lsa_60_words_and_pairs"] = _lsa(texts, R, 60, 3, ngram=(1, 2))
    A = tv(stop_words="english").fit_transform(texts)
    G = np.random.default_rng(seed + 1).standard_normal((A.shape[1], 256)) / 16.0
    Z = normalize(A @ G)
    M["random_projection_256"] = Z[R] @ Z.T
    return {k: np.asarray(v, dtype=np.float32) for k, v in M.items()}


# ------------------------------------------------------------------------------------------------------ transforms
def hub_z(M, remake_family, film_family):
    """Hub correction of a [n_remakes, n_films] measure: each cell as a z-score against the same film's values with
    unrelated remakes (every remake outside the film's family, the cell's own remake left out)."""
    M = np.asarray(M, dtype=np.float64)
    n, m = M.shape
    fams = np.array(list(remake_family), dtype=object)
    film_family = np.asarray(film_family, dtype=object)
    allowed = np.ones((n, m), bool)
    for f in set(fams.tolist()):
        cols = np.flatnonzero(film_family == f)
        if len(cols):
            allowed[np.ix_(np.flatnonzero(fams == f), cols)] = False
    cnt = allowed.sum(0)[None, :] - allowed
    s1 = (M * allowed).sum(0)[None, :] - M * allowed
    s2 = (M * M * allowed).sum(0)[None, :] - M * M * allowed
    mean = s1 / np.maximum(cnt, 1)
    var = np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0)
    return (M - mean) / (np.sqrt(var) + 1e-9)


class Stack:
    """Every (remake, candidate) cell in one table, the pools one after another."""

    def __init__(self, cells):
        self.cells = cells
        self.sizes = np.array([len(c["pool"]) for c in cells])
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.n = int(self.sizes.sum())
        self.group = np.repeat(np.arange(len(cells)), self.sizes)
        self.cols = np.concatenate([np.asarray(c["pool"]) for c in cells])
        self.src = np.array([st + int(np.flatnonzero(np.asarray(c["pool"]) == c["source"])[0]) for st, c in zip(self.starts, cells)])
        fams = [c["family"] for c in cells]
        self.fam_names = sorted(set(fams), key=str)
        index = {f: i for i, f in enumerate(self.fam_names)}
        self.fam = np.array([index[f] for f in fams])
        self.n_fam = len(self.fam_names)

    def standardise(self, x):
        """Mean 0 and variance 1 within every pool (a column that is constant in a pool becomes 0 there)."""
        m = np.add.reduceat(x, self.starts) / self.sizes
        v = np.maximum(np.add.reduceat(x * x, self.starts) / self.sizes - m * m, 0)
        sd = np.sqrt(v)
        sd[sd < 1e-12 * (1 + np.abs(m))] = np.inf
        return (x - np.repeat(m, self.sizes)) / np.repeat(sd, self.sizes)

    def column(self, M):
        """A [n_remakes, n_films] measure as one stacked column, standardised within every pool."""
        return self.standardise(np.asarray(M)[self.group, self.cols].astype(np.float64))

    def normal_scores(self, x):
        """Within every pool, the value replaced by the normal score of its rank: Phi^-1((rank - 0.5) / pool size),
        ties sharing their mean rank. Only the order of the measure inside the pool is kept."""
        x = np.asarray(x, dtype=np.float64)
        out = np.empty(self.n)
        for st, sz in zip(self.starts, self.sizes):
            out[st:st + sz] = norm.ppf((rankdata(x[st:st + sz]) - 0.5) / sz)
        return out

    def rank_column(self, M):
        return self.normal_scores(np.asarray(M)[self.group, self.cols])

    def family_sum(self, x):
        return np.bincount(self.fam, weights=x, minlength=self.n_fam)

    def rank_of_source(self, score):
        """1 + the number of candidates that beat the source, ties counted as half."""
        r = np.empty(len(self.cells))
        for i, (st, sz) in enumerate(zip(self.starts, self.sizes)):
            x, s = score[st:st + sz], score[self.src[i]]
            r[i] = 1 + (x > s).sum() + 0.5 * ((x == s).sum() - 1)
        return r


# ------------------------------------------------------------------------------------------- the conditional logit
def fit_clogit(X, S, w, lam, beta0=None, tol=1e-9, max_iter=200, prune=1e-6):
    """Ridge-penalised conditional logit by damped Newton steps (convex). w: one weight per remake (0 = left out).
    The gradient is exact; the Hessian leaves out cells of weighted probability under `prune`, which changes the path
    to the optimum and not the optimum."""
    k = X.shape[1]
    beta = np.zeros(k) if beta0 is None else np.array(beta0, dtype=float)
    starts, sizes, src, group = S.starts, S.sizes, S.src, S.group
    ng = len(sizes)
    wrep = np.repeat(w, sizes)
    xs = X[src].T @ w

    def parts(b):
        eta = X @ b
        m = np.maximum.reduceat(eta, starts)
        e = np.exp(eta - np.repeat(m, sizes))
        Z = np.add.reduceat(e, starts)
        return -(w * (eta[src] - m - np.log(Z))).sum() + 0.5 * lam * b @ b, e, Z

    f, e, Z = parts(beta)
    for _ in range(max_iter):
        pw = e * np.repeat(w / Z, sizes)
        g = -(xs - X.T @ pw) + lam * beta
        idx = np.flatnonzero(pw > prune)
        Xi, gi, wi = X[idx], group[idx], wrep[idx]
        Xp = Xi * (pw[idx] / wi)[:, None]
        Ex = np.column_stack([np.bincount(gi, weights=Xp[:, j], minlength=ng) for j in range(k)])
        H = (Xp * wi[:, None]).T @ Xi - (Ex * w[:, None]).T @ Ex + (lam + 1e-10) * np.eye(k)
        step = np.linalg.solve(H, g)
        dec = g @ step
        if dec < tol:
            break
        t = 1.0
        while True:
            f2, e2, Z2 = parts(beta - t * step)
            if f2 <= f - 1e-4 * t * dec or t < 1e-8:
                break
            t *= 0.5
        beta, f, e, Z = beta - t * step, f2, e2, Z2
    return beta


def log_prob_of_source(X, S, beta):
    eta = X @ beta
    m = np.maximum.reduceat(eta, S.starts)
    Z = np.add.reduceat(np.exp(eta - np.repeat(m, S.sizes)), S.starts)
    return eta[S.src] - m - np.log(Z), eta


def sign_flip_p(d, n_flips=N_FLIPS, seed=SEED):
    """One-sided p for sum(d) > 0 under random signs of the family sums. Exact when at most EXACT_UP_TO are non-zero."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    obs = d.sum()
    tol = 1e-12 * np.abs(d).sum()
    if len(d) <= EXACT_UP_TO:
        null = np.zeros(1)
        for x in d:
            null = np.concatenate([null + x, null - x])
        return float((null >= obs - tol).mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, n_flips, 10000):
        k = min(10000, n_flips - start)
        signs = rng.integers(0, 2, size=(k, len(d))) * 2.0 - 1.0
        hits += int(((signs @ d) >= obs - tol).sum())
    return float((1 + hits) / (n_flips + 1))


# ---------------------------------------------------------------------------------------------------------- the test
class RemakeTest:
    """The words-only model, fitted leaving one source family out at a time, ready to test unit columns."""

    def __init__(self, cells, measures, film_family, lam=LAMBDA, columns=None, cache_prune=None, consensus_cube=True):
        self.S = S = Stack(cells)
        self.lam = lam
        self.remake_family = [c["family"] for c in cells]
        self.film_family = np.asarray(film_family, dtype=object)
        self.columns = list(WORD_COLUMNS if columns is None else columns)
        self._hub = {}
        self.X = np.column_stack([self.word_column(measures[m], form, key=m) for m, form in self.columns])
        self.names = [f"{m}.{f}" for m, f in self.columns]
        if consensus_cube:
            # the 31st column: the cube of the word consensus (the mean of the word-measure columns, standardised within
            # the pool), so that "far ahead on the words" can count more than in proportion
            which = [i for i, (m, _) in enumerate(self.columns) if m in COSINES + CONTAINMENTS]
            self.X = np.column_stack([self.X, S.standardise(S.standardise(self.X[:, which].mean(1)) ** 3)])
            self.names.append("consensus.cube")
        n = len(cells)
        self.beta_all = fit_clogit(self.X, S, np.ones(n), lam)
        self.betas = np.empty((S.n_fam, self.X.shape[1]))
        self.ll0 = np.empty(n)
        self.rank0 = np.empty(n)
        for f in range(S.n_fam):
            self.betas[f] = fit_clogit(self.X, S, (S.fam != f).astype(float), lam, beta0=self.beta_all)
            ll, eta = log_prob_of_source(self.X, S, self.betas[f])
            test = S.fam == f
            self.ll0[test] = ll[test]
            self.rank0[test] = S.rank_of_source(eta)[test]
        self.p_source = np.exp(self.ll0)
        self.unsolved = float(np.sum(1 - self.p_source))          # "effective pairs" the words leave open
        self.evaluable = self.unsolved >= MIN_UNSOLVED
        self._cache = None
        if cache_prune is not None:                               # for calibration: many columns, one at a time
            self._cache = [self._fold(f, cache_prune) for f in range(S.n_fam)]

    # -- columns
    def word_column(self, M, form, key=None):
        if form == "hz":
            if key is None or key not in self._hub:
                H = hub_z(M, self.remake_family, self.film_family)
                if key is None:
                    return self.S.column(H)
                self._hub[key] = self.S.column(H)
            return self._hub[key]
        return self.S.column(M)

    def unit_column(self, unit, form="rate_matrix"):
        """The unit measure as one stacked column: within every pool, the normal score of its rank (only the order of
        the unit measure inside a pool is used, so any increasing rescaling of it gives the same test).
        rate_matrix (registered): `unit` is the [n_remakes, n_films] matrix of containment rates u between every
            remake and every film read; it is hub-corrected here by hub_z, exactly like the word measures and stand-ins.
        hub_z_per_pool: `unit` is one array per remake over its pool, already hub-corrected by the pipeline (the
            reference films must then be those of hub_z: the remakes outside the candidate's family).
        raw_matrix: the rate matrix with no hub correction (a reported secondary).
        stacked: one value per cell, pools one after another."""
        if form == "rate_matrix":
            return self.S.rank_column(hub_z(unit, self.remake_family, self.film_family))
        if form == "hub_z_per_pool":
            x = np.concatenate([np.asarray(a, dtype=np.float64) for a in unit])
            if len(x) != self.S.n:
                raise ValueError("the unit arrays do not match the pools")
            return self.S.normal_scores(x)
        if form == "raw_matrix":
            return self.S.rank_column(unit)
        if form == "stacked":
            return self.S.normal_scores(unit)
        raise KeyError(form)

    # -- the slope
    def _fold(self, f, prune=0.0):
        S, X, k = self.S, self.X, self.X.shape[1]
        eta = X @ self.betas[f]
        m = np.maximum.reduceat(eta, S.starts)
        e = np.exp(eta - np.repeat(m, S.sizes))
        p = e / np.repeat(np.add.reduceat(e, S.starts), S.sizes)
        idx = np.flatnonzero(p > prune) if prune > 0 else np.arange(S.n)
        g, pi, Xi = S.group[idx], p[idx], X[idx]
        Ex = np.column_stack([np.bincount(g, weights=Xi[:, j] * pi, minlength=len(S.sizes)) for j in range(k)])
        Xc = Xi - Ex[g]
        train = (S.fam != f)[g]
        Q = (Xc * (pi * train)[:, None]).T                        # [k, cells]: the other families' cells, weighted by p
        B = np.linalg.inv(Q @ Xc + self.lam * np.eye(k))
        t = np.flatnonzero(S.fam == f)
        keep = ~train
        return {"idx": idx, "Q": np.ascontiguousarray(Q, dtype=np.float32 if prune > 0 else np.float64), "B": B, "test": t,
                "xs_c": (X[S.src] - Ex)[t], "t_idx": idx[keep], "t_g": g[keep], "t_p": pi[keep]}

    def slope(self, u):
        """Per-remake contributions U_r for one stacked unit column (or a [cells, m] block of columns)."""
        S = self.S
        U2 = u.reshape(S.n, -1)
        out = np.empty((len(S.cells), U2.shape[1]))
        for f in range(S.n_fam):
            fc = self._cache[f] if self._cache is not None else self._fold(f)
            ui = U2[fc["idx"]]
            A = fc["B"] @ (fc["Q"] @ ui.astype(fc["Q"].dtype)).astype(np.float64)                     # [k, m]
            t = fc["test"]
            Eu = np.column_stack([np.bincount(fc["t_g"], weights=U2[fc["t_idx"], j] * fc["t_p"], minlength=len(S.cells))[t]
                                  for j in range(U2.shape[1])])
            out[t] = (U2[S.src[t]] - Eu) - fc["xs_c"] @ A
        return out[:, 0] if u.ndim == 1 else out

    def slope_test(self, u, n_flips=N_FLIPS, seed=SEED):
        U = self.slope(u)
        d = self.S.family_sum(U)
        ss = float(np.sqrt((d ** 2).sum()))
        return {"T": float(d.sum()), "Z": float(d.sum() / ss) if ss > 0 else 0.0, "p": sign_flip_p(d, n_flips, seed),
                "family_sums": d, "per_remake": U, "families_positive": int((d > 0).sum()), "families_negative": int((d < 0).sum())}

    # -- the held-out gain of the refitted model (the starting design)
    def gain(self, u, n_flips=N_FLIPS, seed=SEED):
        S = self.S
        X = np.column_stack([self.X, u])
        n = len(S.cells)
        ll1, rank1, gam = np.empty(n), np.empty(n), np.empty(S.n_fam)
        for f in range(S.n_fam):
            b = fit_clogit(X, S, (S.fam != f).astype(float), self.lam, beta0=np.concatenate([self.betas[f], [0.0]]))
            gam[f] = b[-1]
            ll, eta = log_prob_of_source(X, S, b)
            test = S.fam == f
            ll1[test] = ll[test]
            rank1[test] = S.rank_of_source(eta)[test]
        d = S.family_sum(ll1 - self.ll0)
        return {"gain": float(d.sum()), "p": sign_flip_p(d, n_flips, seed), "family_sums": d, "unit_weight_mean": float(gam.mean()),
                "unit_weight_min": float(gam.min()), "sources_first_words": int((self.rank0 == 1).sum()), "sources_first_with_units": int((rank1 == 1).sum()),
                "log_rank_gain": float(np.sum(np.log(self.rank0) - np.log(rank1))), "rank_with_units": rank1, "log_prob_with_units": ll1}

    def run(self, unit, form="rate_matrix", with_gain=True):
        u = self.unit_column(unit, form)
        out = {"n_remakes": len(self.S.cells), "n_families": self.S.n_fam, "n_cells": self.S.n,
               "words_only": {"held_out_log_likelihood": float(self.ll0.sum()), "sources_ranked_first": int((self.rank0 == 1).sum()),
                              "effective_unsolved_pairs": self.unsolved, "evaluable": bool(self.evaluable)},
               "slope": self.slope_test(u)}
        if with_gain:
            out["gain"] = self.gain(u)
        return out


def standins_through_test(test, panel, with_gain=False, form="rate_matrix"):
    """Every stand-in of the panel in the units' place (hub-corrected and ranked within the pool like the unit column)."""
    names = list(panel)
    U = np.column_stack([test.unit_column(panel[k], form) for k in names])
    D = np.column_stack([test.S.family_sum(c) for c in test.slope(U).T])              # [families, stand-ins]
    out = {}
    for j, name in enumerate(names):
        d = D[:, j]
        ss = float(np.sqrt((d ** 2).sum()))
        out[name] = {"T": float(d.sum()), "Z": float(d.sum() / ss) if ss > 0 else 0.0, "p": sign_flip_p(d)}
        if with_gain:
            g = test.gain(U[:, j])
            out[name].update(gain=g["gain"], gain_p=g["p"])
    return out


def decide(test, unit, panel, form="rate_matrix"):
    """The registered decision. "The units carry information about the source beyond these word measures" is claimed
    only if every line of `conditions` is True."""
    res = test.run(unit, form)
    st = standins_through_test(test, panel)
    z_max = max(v["Z"] for v in st.values())
    share = float(np.mean([v["p"] < ALPHA for v in st.values()]))
    cond = {
        "evaluable: the words leave at least 5 effective pairs open": bool(test.evaluable),
        "slope: one-sided sign-flip p below 0.05": bool(res["slope"]["p"] < ALPHA),
        "gain: the refitted model's held-out log-likelihood gain is positive": bool(res["gain"]["gain"] > 0),
        "stand-ins: the units' Z is above every word-only stand-in's Z": bool(res["slope"]["Z"] > z_max),
        "stand-ins: at most 10% of them are below 0.05": bool(share <= MAX_STANDIN_SHARE),
    }
    res.update(standins=st, standin_z_max=float(z_max), standin_share_below_alpha=share, conditions=cond, claim=bool(all(cond.values())))
    return res
