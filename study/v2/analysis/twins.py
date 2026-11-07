"""Two models, one meme? (PLAN.md, Section 12.)

For each explained Qwen3-32B unit, its twin in another model is the unit there whose sentence-by-sentence
activations over the 3,700 original-text sentences correlate most strongly (v1's search). The twins are then
scored where they were not chosen: on the tester's texts written for the Qwen3-32B unit, and on the name-free
study sentences, against control units.

    cd study && ../.venv/bin/python v2/analysis/twins.py pool <tag>              a random pool of active units that are not candidates
    ... gpu/study_sentences.py <tag> ... pool.npy pool                           their sentence profiles
    cd study && ../.venv/bin/python v2/analysis/twins.py queries <tag>           explained units and their matched controls -> queries.npz
    ... gpu/twin_search.py <other> ... queries.npz twins.json                    in each other model
    cd study && ../.venv/bin/python v2/analysis/twins.py units <tag> <other>     the twins, to read sentence by sentence and on the tester's texts
    ... gpu/study_sentences.py <other> ... twin_units.npy twin ; gpu/score_texts.py <other> texts.jsonl twin_explained.npy twin_tests.npz
    cd study && ../.venv/bin/python v2/analysis/twins.py score <tag> <other>

Controls are drawn, as in v1, from active units that are not candidates: three per explained unit matched on how
well a single word predicts the unit's sentence profile and on how many sentences it fires strongly in, and three
matched on depth and firing count only.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import common as C
import textcheck as T
from plain_checks import best_word

import experiments as X

N_POOL, PER_UNIT = 6000, 3
THR = int(round(C.THETA * C.STEP))


def jobs_dir():
    d = Path("v2/cache/jobs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def profiles(tag, prefix, text="orig"):
    d = C.ACTS / tag
    return np.load(d / f"{prefix}_units.npy").astype(np.int64), np.load(d / f"{prefix}_{text}.u8.npy", mmap_mode="r")


def explained_units(tag):
    return [c["unit"] for c in json.load(open(C.RESULTS / tag / "explain_units.json"))["units"]]


def step_pool(tag):
    rng = np.random.default_rng(C.SEED + 41)
    st = C.Setting(tag, icon=False)
    taken = set(np.load(C.RESULTS / tag / "sentence_units.npy").tolist())
    active = (st.P["orig"].sum(0) >= 3) & ~np.isin(st.units, list(taken))
    pool = np.sort(rng.choice(st.units[active], min(N_POOL, int(active.sum())), replace=False))
    np.save(jobs_dir() / f"twin_pool_{tag}.npy", pool)
    print(f"{tag}: pool of {len(pool)} active units that are not candidates ({int(active.sum())} to draw from)")


def step_queries(tag):
    rng = np.random.default_rng(C.SEED + 42)
    L, D = C.geometry(tag)
    U = L * D
    su, S = profiles(tag, "sent")
    pu, Pp = profiles(tag, "pool")
    col = {int(u): k for k, u in enumerate(su)}
    units = np.array([u for u in explained_units(tag) if u in col], dtype=np.int64)
    Q = np.stack([np.asarray(S[:, col[int(u)]], dtype=np.float32) for u in units], 1)
    P = np.asarray(Pp, dtype=np.float32)
    keep = P.std(0) > 0
    pu, P = pu[keep], P[:, keep]
    sentences = [s for w in X.works for s in w["sentences"]]
    B = CountVectorizer(ngram_range=(1, 2), min_df=2, binary=True).fit_transform(sentences).astype(np.float32).tocsc()
    bw_q, _ = best_word(Q, B)
    bw_p, _ = best_word(P, B)
    n_q, n_p = np.log1p((Q >= THR).sum(0)), np.log1p((P >= THR).sum(0))          # sentences fired in strongly
    lay_q, lay_p = (units % U) // D, (pu % U) // D
    picks = {"lexical": [], "depth": []}
    for i in range(len(units)):
        d_lex = ((bw_p - bw_q[i]) / bw_p.std()) ** 2 + ((n_p - n_q[i]) / n_p.std()) ** 2
        d_dep = np.where(np.abs(lay_p - lay_q[i]) <= max(1, L // 16), np.abs(n_p - n_q[i]), np.inf)
        picks["lexical"].append(rng.choice(np.argsort(d_lex, kind="stable")[:15], PER_UNIT, replace=False))
        picks["depth"].append(rng.choice(np.argsort(d_dep, kind="stable")[:25], PER_UNIT, replace=False))
    ids, cols = [str(int(u)) for u in units], [Q]
    roles = [{"id": str(int(u)), "role": "explained", "unit": int(u), "best_word_r": float(bw_q[i]), "n_strong": int((Q[:, i] >= THR).sum())} for i, u in enumerate(units)]
    for kind, rows in picks.items():
        for i, sel in enumerate(rows):
            for k, j in enumerate(sel):
                qid = f"{int(units[i])}:{kind}{k}"
                ids.append(qid)
                cols.append(P[:, [j]])
                roles.append({"id": qid, "role": kind, "for": int(units[i]), "unit": int(pu[j]), "best_word_r": float(bw_p[j]), "n_strong": int((P[:, j] >= THR).sum())})
    np.savez_compressed(jobs_dir() / f"twin_queries_{tag}.npz", profiles=np.concatenate(cols, 1), ids=np.array(ids))
    C.save({"model": tag, "n_explained": len(units), "controls_per_unit": PER_UNIT, "pool": int(len(pu)),
            "best_word_r_median": {"explained": float(np.median(bw_q)), "lexical_controls": float(np.median([r["best_word_r"] for r in roles if r["role"] == "lexical"])),
                                   "depth_controls": float(np.median([r["best_word_r"] for r in roles if r["role"] == "depth"]))},
            "strong_sentences_median": {k: float(np.median([r["n_strong"] for r in roles if r["role"] == v])) for k, v in
                                        (("explained", "explained"), ("lexical_controls", "lexical"), ("depth_controls", "depth"))},
            "queries": roles}, f"{tag}/twin_queries.json")


def step_units(tag, other):
    tw = json.load(open(C.ACTS / other / "twins.json"))["queries"]
    roles = {r["id"]: r for r in json.load(open(C.RESULTS / tag / "twin_queries.json"))["queries"]}
    allu = np.array(sorted({q["twins"][0]["unit"] for q in tw}), dtype=np.int64)
    expl = np.array([q["twins"][0]["unit"] for q in tw if roles[q["id"]]["role"] == "explained"], dtype=np.int64)
    np.save(jobs_dir() / f"twin_units_{other}.npy", allu)
    np.save(jobs_dir() / f"twin_explained_{other}.npy", np.unique(expl))
    print(f"{other}: {len(allu)} twin units ({len(np.unique(expl))} twins of explained units)")


def corr(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def step_score(tag, other):
    rng = np.random.default_rng(C.SEED + 43)
    tw = {q["id"]: q["twins"][0] for q in json.load(open(C.ACTS / other / "twins.json"))["queries"]}
    roles = json.load(open(C.RESULTS / tag / "twin_queries.json"))["queries"]
    tests = {r["unit"]: r for r in json.load(open(C.RESULTS / tag / "explanation_tests.json"))["units"]}

    # ---- (b) the name-free study sentences: twin correlation against the controls'
    su, Sa = profiles(tag, "sent", "anon")
    pu, Pa = profiles(tag, "pool", "anon")
    tu, Ta = profiles(other, "twin", "anon")
    cs, cp, ct = ({int(u): k for k, u in enumerate(x)} for x in (su, pu, tu))
    rows = []
    for r in roles:
        mine = Sa[:, cs[r["unit"]]] if r["role"] == "explained" else Pa[:, cp[r["unit"]]]
        rows.append(dict(r, twin=tw[r["id"]]["unit"], r_original=tw[r["id"]]["r"], r_name_free=corr(np.asarray(mine), np.asarray(Ta[:, ct[tw[r["id"]]["unit"]]]))))

    def med(sel, key):
        v = [x[key] for x in rows if sel(x)]
        return float(np.median(v)) if v else None

    def group(units):
        inu = lambda x: (x["unit"] if x["role"] == "explained" else x["for"]) in units      # noqa: E731
        out = {"n": int(sum(x["role"] == "explained" and inu(x) for x in rows))}
        for role in ("explained", "lexical", "depth"):
            out[role] = {"r_original_median": med(lambda x: x["role"] == role and inu(x), "r_original"),
                         "r_name_free_median": med(lambda x: x["role"] == role and inu(x), "r_name_free")}
        # per explained unit: its name-free twin r minus the mean of its three controls'
        for role in ("lexical", "depth"):
            diffs = []
            for x in rows:
                if x["role"] == "explained" and inu(x):
                    c = [y["r_name_free"] for y in rows if y["role"] == role and y["for"] == x["unit"]]
                    diffs.append(x["r_name_free"] - float(np.mean(c)))
            diffs = np.array(diffs)
            boot = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(4000)] if len(diffs) else [np.nan]
            out[f"twin_minus_{role}_controls"] = {"mean": float(diffs.mean()) if len(diffs) else None, "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
                                                  "twins_better": int((diffs > 0).sum())}
        return out

    idea = {u for u, t in tests.items() if t["is_idea"]}
    name = {u for u, t in tests.items() if not t["is_idea"]}
    out = {"model": tag, "other": other, "name_free_sentences": {"all": group(set(tests)), "idea_units": group(idea), "name_units": group(name)}}

    # ---- (a) the tester's texts: does the twin, on its own scale, fire on A and B and not on C and D?
    z = np.load(C.ACTS / other / "twin_tests.npz")
    zc = {int(u): k for k, u in enumerate(z["units"])}
    scale = C.scale(other)[z["units"].astype(np.int64)]
    share = z["peak"] / np.where(scale > 0, scale, np.inf)[None, :]
    row = {str(i): k for k, i in enumerate(z["ids"])}
    owners = [u for u in tests if str(u) in tw and f"{u}:A1" in row and tests[u]["testable_out_of_context"]]
    strong = {g: np.stack([np.stack([share[row[f"{u}:{g}{i + 1}"]] >= C.THETA for i in range(5)]).sum(0) for u in owners]) for g in "ABCD"}     # [owner, twin unit]
    tcol = np.array([zc[tw[str(u)]["unit"]] for u in owners])
    n = len(owners)
    # idea-bound, as in Section 10: at least 3 of 5 keyword-free texts strong and at most one keyword-reusing near miss strong
    ex = {int(u): e for u, e in json.load(open(C.RESULTS / tag / "explanations.json"))["units"].items()}
    texts = {int(u): t for u, t in json.load(open(C.RESULTS / tag / "tests.json"))["units"].items()}
    per = {g: np.stack([np.stack([share[row[f"{u}:{g}{i + 1}"]] >= C.THETA for i in range(5)]) for u in owners]) for g in "BC"}      # [owner, 5, twin unit]
    reuse = np.array([[bool(T.uses(t, T.key_stems(ex[u]["key_words"]))) for t in texts[u]["C"]] for u in owners])
    bound = (per["B"].sum(1) >= 3) & ((per["C"] & reuse[:, :, None]).sum(1) <= 1)                                                    # [owner, twin unit]
    a, c = strong["A"][np.arange(n), tcol], strong["C"][np.arange(n), tcol]
    b = strong["B"][np.arange(n), tcol]

    def stats(ai, ci, bi, sel):
        return {"n": int(sel.sum()), "won": int((ai > ci)[sel].sum()), "tied": int((ai == ci)[sel].sum()), "lost": int((ai < ci)[sel].sum()),
                "strong_rate_A": float(ai[sel].mean() / 5) if sel.any() else None, "strong_rate_B": float(bi[sel].mean() / 5) if sel.any() else None,
                "strong_rate_C": float(ci[sel].mean() / 5) if sel.any() else None,
                "strong_rate_D": float(strong["D"][np.arange(n), tcol][sel].mean() / 5) if sel.any() else None}

    is_idea = np.array([u in idea for u in owners])
    res = {"all": stats(a, c, b, np.ones(n, bool)), "idea_units": stats(a, c, b, is_idea), "name_units": stats(a, c, b, ~is_idea)}
    # mismatch null: each twin scored on another unit's texts
    for key, sel in (("all", np.ones(n, bool)), ("idea_units", is_idea), ("name_units", ~is_idea)):
        idx = np.flatnonzero(sel)
        obs = float((a[idx] > c[idx]).mean()) if len(idx) else None
        null, null_b = [], []
        for _ in range(10000 if len(idx) > 2 else 0):
            p = rng.permutation(len(idx))
            while (p == np.arange(len(idx))).any():
                p = rng.permutation(len(idx))
            null.append(float((strong["A"][idx[p], tcol[idx]] > strong["C"][idx[p], tcol[idx]]).mean()))
            null_b.append(float(bound[idx[p], tcol[idx]].mean()))
        obs_b = float(bound[idx, tcol[idx]].mean()) if len(idx) else None
        res[key]["idea_bound_share"] = obs_b
        res[key]["mismatched_idea_bound_share_mean"] = float(np.mean(null_b)) if null_b else None
        res[key]["p_idea_bound_against_mismatch"] = float((1 + (np.array(null_b) >= obs_b - 1e-12).sum()) / (len(null_b) + 1)) if null_b else None
        res[key]["win_rate"] = obs
        res[key]["mismatched_win_rate_mean"] = float(np.mean(null)) if null else None
        res[key]["p_against_mismatch"] = float((1 + (np.array(null) >= obs - 1e-12).sum()) / (len(null) + 1)) if null else None
    out["testers_texts"] = res
    lex = out["name_free_sentences"]["idea_units"]["twin_minus_lexical_controls"]
    out["twins_beat_both_nulls_for_idea_units"] = bool((res["idea_units"]["p_idea_bound_against_mismatch"] or 1) < 0.05 and lex["ci95"][0] is not None and lex["ci95"][0] > 0)
    out["units"] = [x for x in rows if x["role"] == "explained"]
    C.save(out, f"twins_{tag}_{other}.json")
    print(tag, "->", other, "| name-free twin r, idea units:", out["name_free_sentences"]["idea_units"]["explained"], "| tester texts:", res["idea_units"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"pool": step_pool, "queries": step_queries, "units": step_units, "score": step_score}[cmd](*sys.argv[2:])
