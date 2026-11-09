"""Reviewer: checks that took no part in choosing the columns, redrawn with other seeds and my own lines (from the designers' calibrate.py),
and pure noise in the units' place. Every cell, remake_test only. Word columns: the registered files."""
import os
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
import sys, json, time
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mine import corpus, T

t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
D = corpus()
cells, Rent, texts, N = D["cells"], D["r"], D["texts"], D["N"]
idof = {f["page"]: f["id"] for f in D["films"]}
allrem = np.array(sorted({idof[p["remake"]] for p in D["pairs"]}))
erow = np.searchsorted(allrem, Rent)
saved, A = np.load("v2/cache/second/remakes/word_measures.npz"), np.load("v2/cache/second/remakes/added_measures.npz")
W = json.load(open("results/v2/second/remakes_words.json"))
added = W["word_columns"]["added_measures"]
cols39 = [tuple(c) for c in W["word_columns"]["columns"]]
assert cols39 == list(T.WORD_COLUMNS) + [(k, f) for k in added for f in ("z", "hz")]
meas = {m: (saved[m] if m in saved.files else A[m])[erow] for m in {m for m, _ in cols39}}
test = T.RemakeTest(cells, meas, D["film_family"], columns=cols39)
S = test.S
ZMAX = W["claim_conditions_that_words_decide"]["stand_ins_largest_Z_the_units_must_exceed"]["Z"]
log(f"39 columns: E {test.unsolved:.4f}, sources first {int((test.rank0 == 1).sum())}, ll0 {test.ll0.sum():.4f} ({time.time() - t0:.0f} s); file says E "
    f"{W['primary_calibration']['word_model']['all_entries']['effective_pairs_left_open']:.4f}, ll0 {W['primary_calibration']['word_model']['all_entries']['held_out_log_likelihood']:.4f}")
OUT = {"E39": test.unsolved, "ll39": float(test.ll0.sum())}
save = lambda: json.dump(OUT, open(os.path.join(HERE, "b2_checks.json"), "w"))


def through(test, U, flips):
    Ur = test.slope(np.ascontiguousarray(U))
    res = []
    for j in range(U.shape[1]):
        d = test.S.family_sum(Ur[:, j])
        ss = float(np.sqrt((d ** 2).sum()))
        res.append((T.sign_flip_p(d, flips), float(d.sum() / ss) if ss > 0 else 0.0))
    return np.array(res)


def noise_block(rng, m):
    """Pure noise in the units' place: independent standard normals, then the normal score of the rank within each pool."""
    out = np.empty((S.n, m))
    for st, sz in zip(S.starts, S.sizes):
        x = rng.standard_normal((sz, m))
        out[st:st + sz] = norm.ppf((np.argsort(np.argsort(x, axis=0), axis=0) + 0.5) / sz)
    return out


def rates(r, label):
    p, z = r[:, 0], r[:, 1]
    d = {"draws": len(p), "below_0.05": int((p < 0.05).sum()), "rate": float((p < 0.05).mean()), "below_0.01": int((p < 0.01).sum()), "sd_Z": float(z.std()), "mean_Z": float(z.mean()),
         "p_below_0.05_and_Z_above_the_stand_ins": int(((p < 0.05) & (z > ZMAX)).sum()), "z_max": float(z.max())}
    log(label, d)
    return d


# ---- (3) noisy and curved copies of the word columns, other seeds
CURVES = {"as it is": lambda x: x, "exp(x/2)": lambda x: np.exp(np.clip(x, -30, 30) / 2), "cube": lambda x: x ** 3, "signed square": lambda x: np.sign(x) * x ** 2,
          "signed root": lambda x: np.sign(x) * np.sqrt(np.abs(x)), "tanh(x/3)": lambda x: np.tanh(x / 3), "normal score of the rank": None, "minus log rank": None}
K = len(test.columns)
X = test.X[:, :K]
nolen = np.array([i for i, (m, _) in enumerate(test.columns) if m not in ("cand_log_words", "len_match")])
eta = np.empty(S.n)
for f in range(S.n_fam):
    rr = np.isin(S.group, np.flatnonzero(S.fam == f))
    eta[rr] = test.X[rr] @ test.betas[f]
heldout = S.standardise(eta)


def per_pool(x, f):
    o = np.empty_like(x)
    for st, sz in zip(S.starts, S.sizes):
        o[st:st + sz] = f(x[st:st + sz])
    return o


def copy(kind, curve, sigma, seed):
    g = np.random.default_rng(seed)
    if kind.startswith("col:"):
        x = test.X[:, int(kind[4:])]
    elif kind == "mean":
        x = S.standardise(X.mean(1))
    elif kind == "positive mix":
        x = S.standardise(X @ g.exponential(size=K))
    elif kind == "signed mix":
        x = S.standardise(X @ g.standard_normal(K))
    elif kind == "mean of 5":
        x = S.standardise(X[:, g.choice(nolen, 5, replace=False)].mean(1))
    elif kind == "one column":
        x = X[:, g.choice(nolen)]
    else:
        x = heldout
    if curve == "normal score of the rank":
        x = per_pool(x, lambda v: norm.ppf((rankdata(v) - 0.5) / len(v)))
    elif curve == "minus log rank":
        x = per_pool(x, lambda v: -np.log(rankdata(-v)))
    else:
        x = CURVES[curve](x)
    y = S.standardise(x) + sigma * g.standard_normal(S.n)
    return test.unit_column(S.standardise(y), "stacked")


jobs, seed = [], 5_000_000
for kind in ("mean", "positive mix", "signed mix", "mean of 5", "one column", "held-out score"):
    for curve in CURVES:
        for sigma in (0.5, 1.0):
            for _ in range(2):
                seed += 1
                jobs.append((kind, curve, sigma, seed))
for j in range(K):
    for sigma in (0.5, 1.0, 2.0):
        for _ in range(2):
            seed += 1
            jobs.append((f"col:{j}", "as it is", sigma, seed))
res = []
for b in range(0, len(jobs), 140):
    U = np.column_stack([copy(*j) for j in jobs[b:b + 140]])
    res.append(through(test, U, 5000))
    log(f"copies: {b + U.shape[1]} of {len(jobs)} ({time.time() - t0:.0f} s)")
    del U
r = np.vstack(res)
kind_of = np.array([j[0] for j in jobs])
single = np.array([k.startswith("col:") for k in kind_of])
OUT["copies_39"] = {"all": rates(r, "copies, 39 columns, all:"), "mixes_and_curves": rates(r[~single], "   mixes and curves:"), "single_columns": rates(r[single], "   single columns plus noise:"),
                    "by_kind": {k: rates(r[kind_of == k], f"   {k}:") for k in ("mean", "positive mix", "signed mix", "mean of 5", "one column", "held-out score")},
                    "single_by_column": {test.names[j]: [int((r[kind_of == f"col:{j}"][:, 0] < 0.05).sum()), int((kind_of == f"col:{j}").sum())] for j in range(K)}}
log("single columns with 3 or more of 12 below 0.05:", {k: v for k, v in OUT["copies_39"]["single_by_column"].items() if v[0] >= 3})
save()

# ---- (1) pure noise in the units' place, 39 columns
rngn = np.random.default_rng(555)
res = []
for _ in range(3):
    res.append(through(test, noise_block(rngn, 140), 20000))
    log(f"noise: {sum(len(x) for x in res)} draws ({time.time() - t0:.0f} s)")
OUT["noise_39"] = rates(np.vstack(res), "pure noise, 39 columns:")
save()
log(f"done ({time.time() - t0:.0f} s)")
