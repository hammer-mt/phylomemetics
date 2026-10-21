"""E2 (heredity) robustness: the specification multiverse, the jackknife and stronger nulls.

Everything is recomputed from the saved int8 activations (PHYLO_ACTS, PHYLO_ACTS_ANON) for Qwen3-8B and Llama 3.1 8B,
on name-free and original texts. The paper's primary specification: work score = mean of the 3 strongest sentences;
present = beats all but one of the 69 ordinary films; optional units = present in 2-16 of the 32 Batman works;
hub-adjusted inheritance rate; only comics published no later than the adaptation are eligible; best of the
documented sources. The statistic is the share of eligible comics whose rate beats the source, averaged over the 9
adaptations (lower = stronger heredity), and every p is the chance of a value at least as low.

e2_multiverse.json: model x text x top-k {1,3,5} x presence bar (beat all but j of 69 films, j in 0,1,2,3,5,7,10)
  x band lo {2,3,4} x hi {8,12,16,20,24} x hub on/off x time rule on/off x source rule {best, mean, first} x
  de-duplication of identical 32-work presence patterns on/off = 30,240 specifications. Each is tested against the
  paper's joint relabelling of comic identities (20,000 random maps), relabelling within era (exact), the source
  sets swapped among the adaptations (exact; all and time-consistent) and random screen targets (5,000 draws).
e2_jackknife.json: primary specification, every model x text cell, exact p: every adaptation, pair and triple
  left out, every comic left out, named subsets, per-adaptation contributions and the rewrite-batch check.
e2_nulls.json: primary specification: the stronger nulls, the era-mates statistic, the placebo targets as coded and
  corrected, the sequel positive control, source rules, alternative source lists and alternative rates.

usage: e2_multiverse.py [--quick]   (--quick: a 768-specification multiverse, written to e2_multiverse_quick.json;
  e2_jackknife.json and e2_nulls.json depend only on the primary specification and are rewritten unchanged)
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from discover import load_units  # noqa: E402

ACTS = {"anon": Path(os.environ.get("PHYLO_ACTS_ANON", ROOT / "activations_anon")),
        "orig": Path(os.environ.get("PHYLO_ACTS", ROOT / "activations"))}
OUT = ROOT / "results/robustness"
MODELS = ["qwen3-8b", "llama3.1-8b"]
TEXTS = ["anon", "orig"]
TOL = 1e-9

works = [json.loads(l) for l in open(ROOT / "data/corpus.jsonl")]
N = len(works)
group = np.array([w["group"] for w in works])
medium = np.array([w["medium"] for w in works])
year = np.array([w["year"] or 0 for w in works])
label = [w["label"] for w in works]
L = {w["label"]: w["id"] for w in works}
BAT = np.where(group == "batman")[0]
FILMS = np.where(group == "baseline")[0]
COMICS = np.array([w["id"] for w in works if w["group"] == "batman" and w["medium"] == "comic"])
CPOS = {c: i for i, c in enumerate(COMICS)}
NC = len(COMICS)
assert np.array_equal(BAT, np.arange(32)) and len(FILMS) == 69 and NC == 12
ADAPTATIONS = {
    "Batman Begins": ["Year One"], "The Dark Knight": ["The Long Halloween", "The Killing Joke"],
    "Dark Knight Rises": ["Knightfall", "Dark Knight Returns"], "Batman v Superman": ["Dark Knight Returns"],
    "Under the Red Hood": ["A Death in the Family"], "Arkham Asylum (game)": ["Arkham Asylum (comic)"],
    "Batman '89": ["The Killing Joke"], "Joker (2019)": ["The Killing Joke"], "The Batman": ["Year One", "The Long Halloween"],
}
ADAPT = list(ADAPTATIONS)
STRONG = ["Batman Begins", "Under the Red Hood", "Batman v Superman", "Dark Knight Rises", "The Batman"]
SCREEN_NONADAPT = np.array([L[x] for x in ["1943 serial", "Batman '66", "Batman Returns", "Batman Forever", "Batman & Robin", "Mask of the Phantasm",
                                            "Return of the Joker", "Arkham City", "Arkham Knight", "Lego Batman", "Batman Ninja"]])
SCREEN = np.array([w["id"] for w in works if w["group"] == "batman" and w["medium"] != "comic"])
SEQUELS = {"The Dark Knight": "Batman Begins", "Dark Knight Rises": "The Dark Knight", "Batman Returns": "Batman '89",
           "Batman Forever": "Batman Returns", "Batman & Robin": "Batman Forever", "Arkham City": "Arkham Asylum (game)",
           "Arkham Knight": "Arkham City"}
ERA = np.array([0 if year[c] <= 1989 else 1 if year[c] <= 1999 else 2 for c in COMICS])   # late 80s, 90s, 2000s+
BATCH = np.arange(N) // 10                                                                # data/anon/batch_XX.jsonl
TARGET_POOL = [L[t] for t in ADAPT] + [w for w in SCREEN_NONADAPT if year[w] >= 1989]
# Source lists read from the adaptations' Wikipedia articles (data/raw, fetched by fetch_corpus.py). 'fuller' adds the
# two extra credits the articles give (Begins <- The Long Halloween, Batman '89 <- Dark Knight Returns); 'wiki_full' adds
# every corpus comic an article names as an influence; 'wiki_strict' keeps named narrative sources only (The Dark
# Knight's article says The Killing Joke did not shape the main narrative; Under the Red Hood's names only 'Under the Hood').
SOURCE_LISTS = {
    "documented": ADAPTATIONS,
    "fuller": {**ADAPTATIONS, "Batman Begins": ["Year One", "The Long Halloween"], "Batman '89": ["The Killing Joke", "Dark Knight Returns"]},
    "wiki_full": {"Batman Begins": ["Year One", "The Long Halloween"], "The Dark Knight": ["The Long Halloween", "The Killing Joke", "Year One", "Dark Knight Returns"],
                  "Dark Knight Rises": ["Knightfall", "Dark Knight Returns"], "Batman v Superman": ["Dark Knight Returns"],
                  "Under the Red Hood": ["A Death in the Family"], "Arkham Asylum (game)": ["Arkham Asylum (comic)", "The Killing Joke"],
                  "Batman '89": ["The Killing Joke", "Dark Knight Returns"], "Joker (2019)": ["The Killing Joke", "Dark Knight Returns"],
                  "The Batman": ["Year One", "The Long Halloween"]},
    "wiki_strict": {"Batman Begins": ["Year One", "The Long Halloween"], "The Dark Knight": ["The Long Halloween", "Year One", "Dark Knight Returns"],
                    "Dark Knight Rises": ["Knightfall", "Dark Knight Returns"], "Batman v Superman": ["Dark Knight Returns"],
                    "Arkham Asylum (game)": ["Arkham Asylum (comic)"], "Batman '89": ["The Killing Joke", "Dark Knight Returns"],
                    "Joker (2019)": ["The Killing Joke", "Dark Knight Returns"], "The Batman": ["Year One", "The Long Halloween"]},
}
FULL_GRID = {"topk": [1, 3, 5], "thr_j": [0, 1, 2, 3, 5, 7, 10], "lo": [2, 3, 4], "hi": [8, 12, 16, 20, 24],
             "dedup": [False, True], "hub": [True, False], "time": [True, False], "rule": ["best", "mean", "first"]}
QUICK_GRID = {"topk": [3], "thr_j": [0, 1, 5, 10], "lo": [2, 4], "hi": [8, 16, 24],
              "dedup": [False], "hub": [True, False], "time": [True, False], "rule": ["best", "first"]}
DIMS = ["model", "text", "topk", "thr_j", "lo", "hi", "hub", "time", "rule", "dedup"]
NULLS = ["p_joint", "p_era", "p_swap", "p_swap_tc", "p_target"]


# ------------------------------------------------------------------ permutations
def injections(pool, m):
    """All ordered m-tuples of distinct elements of `pool` (sorted), in itertools.permutations order: int8 [n, m]."""
    pool = np.asarray(sorted(pool), dtype=np.int8)
    out = pool[:, None]
    for _ in range(m - 1):
        r, c = np.nonzero(~(out[:, :, None] == pool[None, None, :]).any(1))
        out = np.concatenate([out[r], pool[c][:, None]], axis=1)
    return out


def expand(img, srcs):
    """Relabelling maps [n, NC]: column s holds the image of source comic s (other columns are never read)."""
    maps = np.tile(np.arange(NC, dtype=np.int8), (len(img), 1))
    maps[:, list(srcs)] = img
    return maps


_PERMS = {}


def perms(n):
    if n not in _PERMS:
        _PERMS[n] = injections(range(n), n).astype(np.intp)
    return _PERMS[n]


TUPLES = {k: injections(range(NC), k).astype(np.intp) for k in (1, 2, 3, 4)}


# ------------------------------------------------------------------ scores and presence
def batman_scores(model, text, ks, js):
    """Per top-k: Batman-work scores [32, 2U] and, per j, the presence bar (the (j+1)-th highest of the 69 film scores)."""
    meta, pos, neg, ps, ns = load_units(ACTS[text] / model)
    sw = np.asarray(meta["sent_work"])
    U = pos.shape[1]
    kmax, m = max(ks), max(js) + 1
    Sb = {k: np.zeros((len(BAT), 2 * U), np.float32) for k in ks}
    bars = {k: {j: np.zeros(2 * U, np.float32) for j in js} for k in ks}
    for off, arr, scale, sign in ((0, pos, ps, 1), (U, neg, ns, -1)):
        F = {k: np.zeros((len(FILMS), U), np.float32) for k in ks}
        for i, w in enumerate(np.r_[BAT, FILMS]):
            r = np.flatnonzero(sw == w)
            block = sign * np.asarray(arr[r[0]:r[-1] + 1], dtype=np.float32)
            n = len(block)
            kk = min(kmax, n)
            cs = np.cumsum(-np.sort(-np.partition(block, n - kk, axis=0)[n - kk:], axis=0), axis=0)
            for k in ks:
                s = cs[min(k, n) - 1] / min(k, n) * scale
                if i < len(BAT):
                    Sb[k][w, off:off + U] = s
                else:
                    F[k][i - len(BAT)] = s
        for k in ks:
            top = -np.sort(-np.partition(F[k], len(FILMS) - m, axis=0)[len(FILMS) - m:], axis=0)
            for j in js:
                bars[k][j][off:off + U] = top[j]
        del F
    return Sb, bars


# ------------------------------------------------------------------ E2 machinery
def rates(Pb, band, dedup=False):
    """R[a, c] = share of comic c's optional units also present in Batman work a, and the number of optional units."""
    nb = Pb.sum(0)
    X = Pb[:, (nb >= band[0]) & (nb <= band[1])]
    if dedup and X.shape[1]:
        code = np.packbits(X, axis=0, bitorder="little").T.copy().view(np.uint32).ravel()
        X = X[:, np.sort(np.unique(code, return_index=True)[1])]
    Xf = X.astype(np.float32)
    inter = (Xf @ Xf[COMICS].T).astype(np.float64)
    return inter / np.maximum(Xf[COMICS].sum(1).astype(np.float64), 1)[None, :], int(X.shape[1])


def expectation_corrected(Pb, band):
    """Overlap minus the overlap expected from each unit's prevalence in the other 30 Batman works, over comic c's units."""
    nb = Pb.sum(0)
    opt = (nb >= band[0]) & (nb <= band[1])
    X = Pb[:, opt].astype(np.float32)
    C = X[COMICS]
    inter = (X @ C.T).astype(np.float64)
    A = (C @ (nb[opt] - 1).astype(np.float32)).astype(np.float64)
    return (inter - (A[None, :] - inter) / 30) / np.maximum(C.sum(1).astype(np.float64), 1)[None, :]


def hub_z(R, hub=True):
    if not hub:
        return R.copy()
    Z = np.zeros_like(R)
    for a in range(len(BAT)):
        v = R[SCREEN_NONADAPT[SCREEN_NONADAPT != a]]
        Z[a] = (R[a] - v.mean(0)) / (v.std(0) + 1e-9)
    return Z


def elig_mask(a, time=True, drop=()):
    m = (year[COMICS] <= year[a]) if time else np.ones(NC, bool)
    m[list(drop)] = False
    return m


def shares(z, elig, img, rule="best"):
    """Share of comparison comics (eligible, not a source) whose z beats the source, for each row of img (source
    tuples as comic positions, documented order), and the count that beat it. Rules: best (paper), first (first-listed
    source; the other sources stay out of the comparison set), mean (average of the per-source shares), mean_z (beat the
    mean z of the sources)."""
    img = np.asarray(img, dtype=np.intp)
    G = elig[None, :] & (z[None, :] > z[:, None])      # G[s, c]: comic c is eligible and beats s
    beats = G.sum(1)
    den = elig.sum() - elig[img].sum(1)
    k = img.shape[1]
    if rule == "best" or (k == 1 and rule != "mean_z"):
        num = beats[img[np.arange(len(img)), np.argmax(z[img], 1)]]
    elif rule == "first":
        num = beats[img[:, 0]] - G[img[:, :1], img[:, 1:]].sum(1)
    elif rule == "mean":
        num = np.mean([beats[img[:, i]] - G[img[:, i:i + 1], np.delete(img, i, 1)].sum(1) for i in range(k)], 0)
    elif rule == "mean_z":
        ref = z[img].mean(1)
        num = (elig[None, :] & (z[None, :] > ref[:, None])).sum(1) - (elig[img] & (z[img] > ref[:, None])).sum(1)
    else:
        raise ValueError(rule)
    return np.where(den > 0, num / np.maximum(den, 1), 0.0), num


def shares_sets(z, elig, sets, rule):
    out = np.zeros(len(sets))
    for k in {len(s) for s in sets}:
        idx = [i for i, s in enumerate(sets) if len(s) == k]
        out[idx] = shares(z, elig, [sets[i] for i in idx], rule)[0]
    return out


def setup(Z, time=True, rule="best", amap=ADAPTATIONS, drop=()):
    """Per adaptation: sources (comic positions), eligibility, observed share and rank, and the share for every
    ordered source tuple (the lookup table behind the relabelling nulls). `drop`: comic positions left out."""
    rows = []
    for t, srcs in amap.items():
        s = tuple(CPOS[L[x]] for x in srcs if CPOS[L[x]] not in drop)
        if not s:
            continue
        a = L[t]
        el = elig_mask(a, time, drop)
        T = np.full((NC,) * len(s), np.nan)
        T[tuple(TUPLES[len(s)].T)] = shares(Z[a], el, TUPLES[len(s)], rule)[0]
        cnt = shares(Z[a], el, [s], rule)[1][0]
        cand = el.copy()
        cand[list(s)] = True
        rows.append({"target": t, "a": a, "srcs": s, "elig": el, "z": Z[a], "share": float(T[s]), "rank": 1 + int(cnt),
                     "n_compared": int(cand.sum()), "table": T})
    return rows


def obs_stat(rows):
    return float(np.mean([r["share"] for r in rows]))


def src_positions(rows):
    return sorted({s for r in rows for s in r["srcs"]})


def joint_maps(rows, pool=range(NC)):
    """Every injective relabelling of the distinct source comics into `pool`."""
    src = src_positions(rows)
    return expand(injections(pool, len(src)), src)


def era_maps(rows, pool=range(NC)):
    """Every relabelling that permutes comic identities only within era (eras holding no source are left alone)."""
    src = set(src_positions(rows))
    blocks = [b for b in ([c for c in sorted(pool) if ERA[c] == e] for e in range(3)) if src & set(b)]
    maps = []
    for ps in itertools.product(*[itertools.permutations(b) for b in blocks]):
        m = np.arange(NC)
        for b, p in zip(blocks, ps):
            m[b] = p
        maps.append(m)
    return np.array(maps, dtype=np.int8)


def contrib(rows, maps):
    """[n_rows, n_maps] each adaptation's share under each relabelling."""
    return np.stack([r["table"][tuple(maps[:, s] for s in r["srcs"])] for r in rows])


def null_stat(rows, maps):
    tot = np.zeros(len(maps))
    for r in rows:
        tot += r["table"][tuple(maps[:, s] for s in r["srcs"])]
    return tot / len(rows)


def swap_matrix(rows, rule):
    """V[i, j]: adaptation i's share given adaptation j's sources; ok[i, j]: j's sources all predate i."""
    sets = [r["srcs"] for r in rows]
    V = np.array([shares_sets(r["z"], r["elig"], sets, rule) for r in rows])
    ok = np.array([[all(year[COMICS[c]] <= year[r["a"]] for c in s) for s in sets] for r in rows])
    return V, ok


def swap_null(V, ok, movable=None):
    """The statistic with source sets permuted among the adaptations (only among `movable` if given), exact."""
    n = len(V)
    idx = np.arange(n) if movable is None else np.asarray(movable)
    P = np.tile(np.arange(n), (len(perms(len(idx))), 1))
    P[:, idx] = idx[perms(len(idx))]
    return V[np.arange(n)[None, :], P].mean(1), ok[np.arange(n)[None, :], P].all(1)


def target_matrix(Z, rows, time, rule):
    """VT[i, j]: screen work TARGET_POOL[i]'s share given adaptation j's sources."""
    sets = [r["srcs"] for r in rows]
    return np.array([shares_sets(Z[a], elig_mask(a, time), sets, rule) for a in TARGET_POOL])


def eligible_uniform(rows, obs):
    """Each adaptation's sources redrawn uniformly among its own eligible comics, independently: exact distribution."""
    G = 55440                                         # every share is j/m with m <= 11, so share * G is an integer
    dist = np.ones(1)
    for r in rows:
        sh = shares(r["z"], r["elig"], injections(np.flatnonzero(r["elig"]), len(r["srcs"])), "best")[0]
        kv, cnt = np.unique(np.rint(sh * G).astype(int), return_counts=True)
        new = np.zeros(len(dist) + G)
        for k, c in zip(kv, cnt):
            new[k:k + len(dist)] += dist * c / len(sh)
        dist = new
    means = np.arange(len(dist)) / G / len(rows)
    return {"null_mean": float((means * dist).sum()), "p": float(dist[means <= obs + TOL].sum())}


def p_le(null, obs, exact=True):
    k = int(np.sum(null <= obs + TOL))
    return k / len(null) if exact else (k + 1) / (len(null) + 1)


def null_summary(null, obs, exact=True):
    return {"n_maps": int(len(null)), "null_mean": float(null.mean()), "null_sd": float(null.std()), "p": p_le(null, obs, exact)}


def sequel_control(Pb, band=(2, 16), exclude=()):
    """Positive control: rank of each sequel's prequel among the earlier Batman screen works by raw rate."""
    nb = Pb.sum(0)
    X = Pb[:, (nb >= band[0]) & (nb <= band[1])].astype(np.float32)
    inter = (X @ X.T).astype(np.float64)
    cnt = np.maximum(X.sum(1).astype(np.float64), 1)
    rows = []
    for seq, pre in SEQUELS.items():
        s, p = L[seq], L[pre]
        cands = [c for c in SCREEN if year[c] <= year[s] and c != s and c not in exclude]
        order = sorted(cands, key=lambda c: -(inter[c, s] / cnt[c]))
        rows.append({"sequel": seq, "prequel": pre, "rank": order.index(p) + 1, "n_candidates": len(cands), "closest": label[order[0]]})
    return {"top2": sum(r["rank"] <= 2 for r in rows), "n": len(rows), "passes": sum(r["rank"] <= 2 for r in rows) >= 5, "rows": rows}


# ------------------------------------------------------------------ multiverse
def run_spec(Z, time, rule, mc_maps, era, tdraws, flat9):
    rows = setup(Z, time, rule)
    obs = obs_stat(rows)
    nj = null_stat(rows, mc_maps)
    ne = null_stat(rows, era)
    V, ok = swap_matrix(rows, rule)
    sv = V.ravel()[flat9].mean(1)
    tc = ok.ravel()[flat9].all(1)
    VT = target_matrix(Z, rows, time, rule)
    tv = VT[tdraws, np.arange(len(rows))[None, :]].mean(1)
    plac = VT[len(rows):].mean(1)
    return {"share": round(obs, 5), "null_joint_mean": round(float(nj.mean()), 5), "null_joint_sd": round(float(nj.std()), 5),
            "p_joint": p_le(nj, obs, exact=False), "null_era_mean": round(float(ne.mean()), 5), "p_era": p_le(ne, obs),
            "null_swap_mean": round(float(sv.mean()), 5), "p_swap": p_le(sv, obs),
            "null_swap_tc_mean": round(float(sv[tc].mean()), 5), "p_swap_tc": p_le(sv[tc], obs),
            "null_target_mean": round(float(tv.mean()), 5), "p_target": p_le(tv, obs, exact=False),
            "placebo1989_mean": round(float(plac.mean()), 5), "placebo1989_min": round(float(plac.min()), 5),
            "ranks": [r["rank"] for r in rows], "shares": [round(r["share"], 4) for r in rows]}


def multiverse_cell(model, text, Sb, bars, grid, mc_maps, era, tdraws, flat9):
    recs = []
    bands = [(lo, hi) for lo in grid["lo"] for hi in grid["hi"]]
    for k in grid["topk"]:
        for j in grid["thr_j"]:
            Pb = Sb[k] > bars[k][j][None, :]
            nb = Pb.sum(0)
            Pb = Pb[:, (nb >= min(grid["lo"])) & (nb <= max(grid["hi"]))]
            for band in bands:
                seq = sequel_control(Pb, band)["top2"]
                for dedup in grid["dedup"]:
                    R, u = rates(Pb, band, dedup)
                    for hub in grid["hub"]:
                        Z = hub_z(R, hub)
                        for tr in grid["time"]:
                            for rule in grid["rule"]:
                                rec = {"model": model, "text": text, "topk": k, "thr_j": j, "lo": band[0], "hi": band[1], "hub": hub,
                                       "time": tr, "rule": rule, "dedup": dedup, "n_units": u}
                                rec.update(run_spec(Z, tr, rule, mc_maps, era, tdraws, flat9))
                                rec["sequel_top2"] = seq
                                recs.append(rec)
    return recs


def summarise(rs):
    if not rs:
        return {}
    a = {k: np.array([r[k] for r in rs]) for k in ["share", "null_joint_mean", "null_joint_sd", "null_era_mean", "null_swap_mean",
                                                    "null_target_mean", "sequel_top2", "n_units"] + NULLS}
    o = {"n_specs": len(rs), "n_p_joint_lt_05": int((a["p_joint"] < .05).sum()), "frac_p_joint_lt_05": round(float((a["p_joint"] < .05).mean()), 4),
         "n_p_joint_lt_01": int((a["p_joint"] < .01).sum()), "frac_p_joint_lt_01": round(float((a["p_joint"] < .01).mean()), 4),
         "p_joint_percentiles": {str(q): round(float(np.percentile(a["p_joint"], q)), 5) for q in (25, 50, 75, 95, 99, 100)}}
    for k in NULLS[1:]:
        o[k] = {"n_lt_05": int((a[k] < .05).sum()), "frac_lt_05": round(float((a[k] < .05).mean()), 4),
                "median": round(float(np.median(a[k])), 5), "max": round(float(a[k].max()), 5)}
    o["n_all_five_nulls_lt_05"] = int(np.all([a[k] < .05 for k in NULLS], axis=0).sum())
    o["share_percentiles"] = {str(q): round(float(np.percentile(a["share"], q)), 4) for q in (5, 25, 50, 75, 95)}
    gap = a["null_joint_mean"] - a["share"]
    o["null_joint_mean_median"] = round(float(np.median(a["null_joint_mean"])), 4)
    o["gap_median"] = round(float(np.median(gap)), 4)
    o["gap_in_null_sd_median"] = round(float(np.median(gap / a["null_joint_sd"])), 3)
    o["null_era_mean_median"] = round(float(np.median(a["null_era_mean"])), 4)
    o["null_swap_mean_median"] = round(float(np.median(a["null_swap_mean"])), 4)
    o["null_target_mean_median"] = round(float(np.median(a["null_target_mean"])), 4)
    o["era_share_of_gap_median"] = round(float(np.median((a["null_joint_mean"] - a["null_era_mean"]) / np.where(gap > 0, gap, np.nan))), 3)
    o["frac_sequel_control_passes"] = round(float((a["sequel_top2"] >= 5).mean()), 4)
    o["n_units_median"], o["n_units_min"] = int(np.median(a["n_units"])), int(a["n_units"].min())
    return o


def per_adaptation(rs):
    R = np.array([r["ranks"] for r in rs])
    S = np.array([r["shares"] for r in rs])
    return {t: {"frac_rank1": round(float(np.mean(R[:, i] == 1)), 4), "frac_rank_le2": round(float(np.mean(R[:, i] <= 2)), 4),
                "median_rank": float(np.median(R[:, i])), "median_share": round(float(np.median(S[:, i])), 4),
                "frac_share_below_half": round(float(np.mean(S[:, i] < 0.5)), 4)} for i, t in enumerate(ADAPT)}


def multiverse_summary(recs, grid, quick):
    def pick(**kw):
        return [r for r in recs if all(r[k] == v for k, v in kw.items())]
    anon = pick(text="anon")
    fails = [r for r in recs if r["p_joint"] >= .05]
    res = {"quick": quick, "grid": grid, "n_specs": len(recs),
           "nulls": {"p_joint": "paper's null: comic identities relabelled jointly; 20,000 random maps drawn from the 3,991,680 injective maps of the 7 source comics (same draws for every specification); p = (k+1)/(n+1)",
                     "p_era": "comic identities permuted only within era (<=1989, 1990s, 2000+); exact, 1,440 maps",
                     "p_swap": "the 9 documented source sets permuted among the 9 adaptations; exact, 9! maps",
                     "p_swap_tc": "the same, keeping only assignments where every source predates its adaptation",
                     "p_target": "the 9 source sets given to 9 random distinct targets among the 9 adaptations and the 9 non-adaptation screen works from 1989 on; 5,000 draws; p = (k+1)/(n+1)"},
           "overall": summarise(recs), "name_free": summarise(anon), "original": summarise(pick(text="orig")),
           "by_cell": {f"{m}/{t}": summarise(pick(model=m, text=t)) for m in MODELS for t in TEXTS},
           "by_dimension": {d: {str(v): summarise([r for r in recs if r[d] == v]) for v in (grid[d] if d in grid else (MODELS if d == "model" else TEXTS))}
                            for d in DIMS},
           "joint_failures": {"n": len(fails), "max_p": round(max([r["p_joint"] for r in fails], default=0), 5),
                              "n_with_hub_off": sum(not r["hub"] for r in fails),
                              "max_p_with_hub_on": round(max([r["p_joint"] for r in recs if r["hub"]], default=0), 5),
                              "by_dimension": {d: {str(v): sum(r[d] == v for r in fails) for v in sorted({r[d] for r in recs}, key=str)} for d in DIMS}},
           "per_adaptation_all": per_adaptation(recs), "per_adaptation_name_free": per_adaptation(anon),
           "per_adaptation_by_cell": {f"{m}/{t}": per_adaptation(pick(model=m, text=t)) for m in MODELS for t in TEXTS},
           "n_adaptations_rank1_hist": {str(k): int(v) for k, v in zip(*np.unique([sum(x == 1 for x in r["ranks"]) for r in recs], return_counts=True))},
           "n_adaptations_share_below_half_hist": {str(k): int(v) for k, v in zip(*np.unique([sum(s < 0.5 for s in r["shares"]) for r in recs], return_counts=True))},
           "worst_p_joint": [{k: r[k] for k in DIMS + ["n_units", "share", "null_joint_mean", "p_joint", "p_era", "p_swap"]}
                             for r in sorted(recs, key=lambda r: -r["p_joint"])[:10]],
           "worst_p_era": [{k: r[k] for k in DIMS + ["n_units", "share", "null_era_mean", "p_joint", "p_era", "p_swap"]}
                           for r in sorted(recs, key=lambda r: -r["p_era"])[:10]]}
    near = dict(topk=3, hub=True, time=True, rule="best", dedup=False)
    res["near_paper_all_bars_and_bands"] = {f"{m}/{t}": (lambda rs: {"n_specs": len(rs), "n_p_joint_lt_05": sum(r["p_joint"] < .05 for r in rs),
                                                                     "max_p_joint": round(max([r["p_joint"] for r in rs], default=0), 5)})(pick(model=m, text=t, **near))
                                            for m in MODELS for t in TEXTS}
    res["name_free_by_model_and_bar"] = {f"{m}/j={j}": (lambda rs: {"frac_p_joint_lt_05": round(float(np.mean([r["p_joint"] < .05 for r in rs])), 4),
                                                                    "frac_p_era_lt_05": round(float(np.mean([r["p_era"] < .05 for r in rs])), 4),
                                                                    "median_share": round(float(np.median([r["share"] for r in rs])), 4),
                                                                    "median_units": int(np.median([r["n_units"] for r in rs]))})(pick(model=m, text="anon", thr_j=j))
                                         for m in MODELS for j in grid["thr_j"]}
    prim = dict(topk=3, thr_j=1, lo=2, hi=16, hub=True, time=True, rule="best", dedup=False)
    res["paper_cells"] = {}
    for m in MODELS:
        for t in TEXTS:
            for name, change in (("primary", {}), ("mean_of_sources", {"rule": "mean"}), ("first_listed", {"rule": "first"}), ("dedup", {"dedup": True})):
                rs = pick(model=m, text=t, **{**prim, **change})
                if rs:
                    res["paper_cells"][f"{m}/{t}/{name}"] = {k: rs[0][k] for k in ["share", "null_joint_mean", "p_joint", "null_era_mean", "p_era", "p_swap",
                                                                                   "p_swap_tc", "p_target", "placebo1989_mean", "placebo1989_min", "ranks",
                                                                                   "sequel_top2", "n_units"]}
    codes = {"model": MODELS, "text": TEXTS, "rule": ["best", "mean", "first"]}
    cols = DIMS + ["n_units", "share", "null_joint_mean", "p_joint", "p_era", "p_swap", "p_swap_tc", "p_target", "sequel_top2"]

    def enc(r, c):
        v = r[c]
        return codes[c].index(v) if c in codes else int(v) if isinstance(v, bool) else round(v, 6) if isinstance(v, float) else v
    res["specs"] = {"columns": cols, "codes": codes, "booleans": "hub, time, dedup as 1/0", "rows": [[enc(r, c) for c in cols] for r in recs]}
    return res


# ------------------------------------------------------------------ jackknife
def contributions(rows):
    out = []
    for r in rows:
        vals = r["table"][~np.isnan(r["table"])]
        out.append({"target": r["target"], "sources": ADAPTATIONS[r["target"]], "share": round(r["share"], 4), "rank": r["rank"],
                    "n_compared": r["n_compared"], "null_expected_share": round(float(vals.mean()), 4),
                    "marginal_p": round(float(np.mean(vals <= r["share"] + TOL)), 4),
                    "z_by_comic": {label[c]: round(float(r["z"][i]), 3) for i, c in enumerate(COMICS) if r["elig"][i] or i in r["srcs"]}})
    return out


def batch_check(R, Z):
    """Inheritance for non-source screen-comic pairs rewritten in the same batch vs in different batches."""
    src = {(L[t], L[s]) for t, v in ADAPTATIONS.items() for s in v}
    pairs = [(a, c) for a in SCREEN for c in COMICS if (a, c) not in src]
    same = np.array([BATCH[a] == BATCH[c] for a, c in pairs])
    r = np.array([R[a, CPOS[c]] for a, c in pairs])
    z = np.array([Z[a, CPOS[c]] for a, c in pairs])
    return {"n_within_batch": int(same.sum()), "n_across": int((~same).sum()), "rate_within": round(float(r[same].mean()), 4),
            "rate_across": round(float(r[~same].mean()), 4), "z_within": round(float(z[same].mean()), 4), "z_across": round(float(z[~same].mean()), 4)}


def jackknife(Pb):
    R, u = rates(Pb, (2, 16))
    Z = hub_z(R)
    rows = setup(Z)
    n = len(rows)
    obs = np.array([r["share"] for r in rows])
    C = contrib(rows, joint_maps(rows))
    Ctot = C.sum(0)
    Ce = contrib(rows, era_maps(rows))
    V, ok = swap_matrix(rows, "best")

    def test(keep):
        keep = sorted(keep)
        drop = [i for i in range(n) if i not in keep]
        o = float(obs[keep].mean())
        nj = (Ctot - C[drop].sum(0)) / len(keep)
        ne = Ce[keep].mean(0)
        sv = swap_null(V[np.ix_(keep, keep)], ok[np.ix_(keep, keep)])[0]
        return {"n_adaptations": len(keep), "share": round(o, 5), "null_joint_mean": round(float(nj.mean()), 5), "p_joint": p_le(nj, o),
                "null_era_mean": round(float(ne.mean()), 5), "p_era": p_le(ne, o), "null_swap_mean": round(float(sv.mean()), 5), "p_swap": p_le(sv, o)}

    names = [r["target"] for r in rows]
    cell = {"n_optional_units": u, "primary": test(range(n)), "contributions": contributions(rows), "batch_check": batch_check(R, Z)}
    for k in (1, 2, 3):
        cell[f"leave{k}_adaptations_out"] = {" + ".join(names[i] for i in d): test([i for i in range(n) if i not in d])
                                             for d in itertools.combinations(range(n), k)}
    cell["named_subsets"] = {
        "without_killing_joke_trio": test([i for i, r in enumerate(rows) if "The Killing Joke" not in ADAPTATIONS[r["target"]]]),
        "without_rewrite_batch_mates": test([i for i, r in enumerate(rows) if r["target"] not in ("Batman '89", "The Dark Knight")]),
        "strong_attribution_subset": test([i for i, r in enumerate(rows) if r["target"] in STRONG])}
    l1c = {}
    for c in range(NC):
        pool = [p for p in range(NC) if p != c]
        rr = setup(Z, drop=(c,))
        o = obs_stat(rr)
        nj, ne = null_stat(rr, joint_maps(rr, pool)), null_stat(rr, era_maps(rr, pool))
        l1c[label[COMICS[c]]] = {"n_adaptations": len(rr), "share": round(o, 5), "null_joint_mean": round(float(nj.mean()), 5),
                                 "p_joint": p_le(nj, o), "null_era_mean": round(float(ne.mean()), 5), "p_era": p_le(ne, o)}
    cell["leave1_comic_out"] = l1c
    summ = {}
    for key in ("leave1_adaptations_out", "leave2_adaptations_out", "leave3_adaptations_out", "leave1_comic_out"):
        d = cell[key]
        worst = max(d, key=lambda x: d[x]["p_joint"])
        worst_e = max(d, key=lambda x: d[x]["p_era"])
        summ[key] = {"n": len(d), "max_p_joint": d[worst]["p_joint"], "worst": worst, "n_p_joint_ge_05": sum(v["p_joint"] >= .05 for v in d.values()),
                     "max_p_era": d[worst_e]["p_era"], "worst_era": worst_e, "n_p_era_ge_05": sum(v["p_era"] >= .05 for v in d.values())}
        if "p_swap" in next(iter(d.values())):
            summ[key].update({"max_p_swap": max(v["p_swap"] for v in d.values()), "n_p_swap_ge_05": sum(v["p_swap"] >= .05 for v in d.values())})
    cell["summary"] = summ
    return cell


# ------------------------------------------------------------------ stronger nulls and variants
def exact_tests(rows):
    o = obs_stat(rows)
    nj, ne = null_stat(rows, joint_maps(rows)), null_stat(rows, era_maps(rows))
    return {"n_adaptations": len(rows), "share": o, "rows": {r["target"]: round(r["share"], 4) for r in rows},
            "ranks": {r["target"]: r["rank"] for r in rows}, "joint": null_summary(nj, o), "within_era": null_summary(ne, o)}


def era_mates(rows):
    """Share of the source's same-era eligible comics that beat it, against the within-era null."""
    def stat(m=None):
        vals = []
        for r in rows:
            s = r["srcs"] if m is None else tuple(int(m[c]) for c in r["srcs"])
            mates = [c for c in range(NC) if ERA[c] in {ERA[x] for x in s} and r["elig"][c] and c not in s]
            if mates:
                best = max(r["z"][x] for x in s)
                vals.append(np.mean([r["z"][c] > best for c in mates]))
        return float(np.mean(vals))
    o = stat()
    null = np.array([stat(m) for m in era_maps(rows)])
    per = {}
    for r in rows:
        mates = [c for c in range(NC) if ERA[c] in {ERA[x] for x in r["srcs"]} and r["elig"][c] and c not in r["srcs"]]
        best = max(r["z"][x] for x in r["srcs"])
        per[r["target"]] = {"n_era_mates": len(mates), "era_mates_beating_source": [label[COMICS[c]] for c in mates if r["z"][c] > best]}
    return {"share": o, **null_summary(null, o), "per_adaptation": per}


def stronger_nulls(Pb):
    R, u = rates(Pb, (2, 16))
    Z = hub_z(R)
    rows = setup(Z)
    obs = obs_stat(rows)
    maps = joint_maps(rows)
    nj = null_stat(rows, maps)
    tc = np.ones(len(maps), bool)
    for r in rows:
        for s in r["srcs"]:
            tc &= year[COMICS[maps[:, s]]] <= year[r["a"]]
    del maps
    ne = null_stat(rows, era_maps(rows))
    V, ok = swap_matrix(rows, "best")
    sv, stc = swap_null(V, ok)
    live = [i for i, r in enumerate(rows) if medium[r["a"]] == "film"]
    sm = swap_null(V, ok, live)[0]
    VT = target_matrix(Z, rows, True, "best")
    draws = np.argsort(np.random.default_rng(7).random((100000, len(TARGET_POOL))), axis=1)[:, :len(rows)]
    tv = VT[draws, np.arange(len(rows))[None, :]].mean(1)
    res = {"n_optional_units": u, "share": obs, "rows": {r["target"]: {"share": round(r["share"], 4), "rank": r["rank"], "n_compared": r["n_compared"]} for r in rows}}
    res["nulls"] = {
        "joint_relabel": null_summary(nj, obs),
        "joint_relabel_time_consistent": null_summary(nj[tc], obs),
        "within_era": null_summary(ne, obs),
        "source_swap": null_summary(sv, obs),
        "source_swap_time_consistent": null_summary(sv[stc], obs),
        "source_swap_within_medium": {**null_summary(sm, obs), "movable": [rows[i]["target"] for i in live]},
        "eligible_uniform": eligible_uniform(rows, obs),
        "random_screen_targets": {**null_summary(tv, obs, exact=False), "pool": [label[w] for w in TARGET_POOL]},
    }
    nm = res["nulls"]
    res["chance_levels"] = {k: round(v["null_mean"], 4) for k, v in nm.items()}
    res["era_share_of_paper_gap"] = round((nm["joint_relabel"]["null_mean"] - nm["within_era"]["null_mean"]) / (nm["joint_relabel"]["null_mean"] - obs), 4)
    res["era_mates"] = era_mates(rows)

    plac = {}
    for tr in (True, False):
        per = {label[t]: {"mean_share": float(np.mean(shares_sets(Z[t], elig_mask(t, tr), [r["srcs"] for r in rows], "best"))),
                          "n_eligible_comics": int(elig_mask(t, tr).sum())} for t in SCREEN_NONADAPT}
        o = obs if tr else obs_stat(setup(Z, time=False))
        valid = np.array([v["mean_share"] for v in per.values() if v["n_eligible_comics"] > 0])
        allv = np.array([v["mean_share"] for v in per.values()])
        plac["time_rule" if tr else "no_time_rule"] = {
            "real_share": o, "per_target": per, "mean_as_coded": float(allv.mean()), "min_as_coded": float(allv.min()),
            "real_below_all_as_coded": bool(o < allv.min()), "n_valid_targets": int(len(valid)), "mean_valid": float(valid.mean()),
            "min_valid": float(valid.min()), "p5_valid": float(np.percentile(valid, 5)), "real_below_all_valid": bool(o < valid.min())}
    res["placebo"] = plac
    res["sequel_control"] = {"paper_rule": sequel_control(Pb), "without_1943_serial": sequel_control(Pb, exclude=(L["1943 serial"],))}

    rules = {"best_of_sources (primary)": exact_tests(rows), "mean_of_sources": exact_tests(setup(Z, rule="mean")),
             "first_listed": exact_tests(setup(Z, rule="first")),
             "first_listed_only": exact_tests(setup(Z, amap={t: v[:1] for t, v in ADAPTATIONS.items()})),
             "against_mean_source_z": exact_tests(setup(Z, rule="mean_z"))}
    res["source_rules"] = {"definitions": {"mean_of_sources": "average over the documented sources of the share of comparison comics beating that source",
                                           "first_listed": "first-listed source only; the other documented source stays out of the comparison set",
                                           "first_listed_only": "first-listed source only; the other documented source becomes an ordinary comparison comic",
                                           "against_mean_source_z": "share of comparison comics beating the mean z of the sources"}, **rules}
    res["source_lists"] = {name: {"sources": amap, **exact_tests(setup(Z, amap=amap))} for name, amap in SOURCE_LISTS.items()}
    E = expectation_corrected(Pb, (2, 16))
    res["rate_variants"] = {"note": "the expectation-corrected rate is a positive affine transform of the raw rate within each comic, so with "
                                    "the hub adjustment it gives exactly the primary result; it is reported without the hub",
                            "raw_rate_no_hub": exact_tests(setup(hub_z(R, False))),
                            "expectation_corrected_no_hub": exact_tests(setup(hub_z(E, False)))}
    return res


# ------------------------------------------------------------------ main
def dump(obj, path, indent=1):
    json.dump(obj, open(path, "w"), indent=indent, default=lambda x: x.item())


def main():
    quick = "--quick" in sys.argv[1:]
    grid = QUICK_GRID if quick else FULL_GRID
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    src7 = sorted({CPOS[L[s]] for v in ADAPTATIONS.values() for s in v})
    inj = injections(range(NC), len(src7))
    mc_maps = expand(inj[rng.integers(0, len(inj), 20000)], src7)
    tdraws = np.argsort(rng.random((5000, len(TARGET_POOL))), axis=1)[:, :len(ADAPT)]
    del inj
    era = era_maps([{"srcs": tuple(src7)}])
    flat9 = (np.arange(len(ADAPT))[None, :] * len(ADAPT) + perms(len(ADAPT))).astype(np.intp)
    jack, nulls, recs = {}, {}, []
    t0 = time.time()
    for model in MODELS:
        for text in TEXTS:
            key = f"{model}/{text}"
            Sb, bars = batman_scores(model, text, sorted(set(grid["topk"]) | {3}), sorted(set(grid["thr_j"]) | {1}))
            print(key, "scored", round(time.time() - t0), "s", flush=True)
            Pb = Sb[3] > bars[3][1][None, :]
            jack[key] = jackknife(Pb)
            p = jack[key]["primary"]
            print(" jackknife: share", p["share"], "p", round(p["p_joint"], 5), "era", round(p["p_era"], 4), "|",
                  {k: (round(v["max_p_joint"], 4), v["n_p_joint_ge_05"]) for k, v in jack[key]["summary"].items()}, round(time.time() - t0), "s", flush=True)
            nulls[key] = stronger_nulls(Pb)
            print(" nulls:", {k: round(v["p"], 5) for k, v in nulls[key]["nulls"].items()}, "| placebo valid mean",
                  round(nulls[key]["placebo"]["time_rule"]["mean_valid"], 3), "| sequels", nulls[key]["sequel_control"]["paper_rule"]["top2"],
                  round(time.time() - t0), "s", flush=True)
            del Pb
            n0 = len(recs)
            recs += multiverse_cell(model, text, Sb, bars, grid, mc_maps, era, tdraws, flat9)
            ps = np.array([r["p_joint"] for r in recs[n0:]])
            print(" multiverse:", len(ps), "specs, p_joint < .05 in", int((ps < .05).sum()), round(time.time() - t0), "s", flush=True)
            del Sb, bars
    dump({"spec": "primary: top-3, beat all but 1 of 69 films, band 2-16, hub on, time rule on, best source; p exact unless noted", "cells": jack},
         OUT / "e2_jackknife.json")
    dump({"spec": "primary: top-3, beat all but 1 of 69 films, band 2-16, hub on, time rule on, best source; p exact unless noted", "cells": nulls},
         OUT / "e2_nulls.json")
    summ = multiverse_summary(recs, grid, quick)
    path = OUT / ("e2_multiverse_quick.json" if quick else "e2_multiverse.json")
    specs = summ.pop("specs")
    with open(path, "w") as f:
        f.write(json.dumps(summ, indent=1, default=lambda x: x.item())[:-2] + ',\n "specs": ' + json.dumps(specs, separators=(",", ":"), default=lambda x: x.item()) + "\n}\n")
    o = summ["overall"]
    print("multiverse:", o["n_specs"], "specs, p_joint < .05 in", o["n_p_joint_lt_05"], f"({o['frac_p_joint_lt_05']:.1%}); failures with hub off",
          summ["joint_failures"]["n_with_hub_off"], "of", summ["joint_failures"]["n"], "| wrote", path, round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
