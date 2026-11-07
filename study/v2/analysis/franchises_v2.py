"""Batman among other stories (PLAN.md, Section 13): other franchises, genre sets, and Batman-like films.

Does Batman have more of a canon than other bodies of films? Three comparisons that could fail, all on original
texts with the v2 presence rule:

  other franchises   six film series from the same plot dataset: how many distinctive units are in at least 72% and at
                     least 90% of a franchise's films (Batman's "common" and "core" bars), with base rates taken from the
                     background plots outside the franchise; beside Batman's own counts on random subsets of its works
                     of the same size
  genre sets         50 random sets of 32 background plots of one genre play the franchise, with base rates from the
                     other background plots
  Batman-like films  (exploratory) the background plots ranked by the share of Batman's name-free core they carry

    cd study && ../.venv/bin/python v2/analysis/franchises_v2.py [model-tag ...]

Needs v2/acts/<tag>/bg_present30.bits.npy and fr_present30.bits.npy (pull.py --franchises) and results/v2/canon_<tag>.npz.
Writes results/v2/franchises_<tag>.json.
"""
import json
import sys

import numpy as np

import common as C

import experiments as X

GENRES = ["drama", "comedy", "horror", "thriller", "action"]
RELATIVES = ["batman", "superman", "spider-man", "zorro", "iron man", "daredevil", "green hornet", "the shadow", "scarlet pimpernel", "monte cristo",
             "sherlock holmes", "dracula", "puss in boots"]
BARS = {"at_least_72pct": 0.72, "at_least_90pct": 0.90}


def bit_counts(bits, rows, chunk=48):
    """How many of the given rows each unit is present in (bits: packed [plots, units / 8])."""
    out = np.zeros(bits.shape[1] * 8, np.int32)
    rows = np.asarray(rows)
    for s in range(0, len(rows), chunk):
        out += np.unpackbits(bits[np.sort(rows[s:s + chunk])], axis=1).sum(0, dtype=np.int32)
    return out


def bit_columns(bits, units):
    """Presence [plots, len(units)] of chosen units."""
    units = np.asarray(units, dtype=np.int64)
    return ((bits[:, units // 8] >> (7 - units % 8).astype(np.uint8)) & 1).astype(bool)


def canon_counts(present, n, base_count, n_base, cap=C.CAP):
    """Units at or under the cap that are present in at least 72% and 90% of n films (and in at least 6 of them)."""
    d = base_count <= cap * n_base + 1e-9
    out = {k: int((d & (present >= np.ceil(v * n - 1e-9))).sum()) for k, v in BARS.items()}
    out["at_least_6_films"] = int((d & (present >= 6)).sum())
    out["without_cap"] = {k: int((present >= np.ceil(v * n - 1e-9)).sum()) for k, v in BARS.items()}
    return out


def run_model(tag, n_draws=50):
    rng = np.random.default_rng(C.SEED)
    d = C.ACTS / tag
    bg = np.load(d / "bg_present30.bits.npy", mmap_mode="r")
    bmeta = json.load(open(d / "bg_meta.json"))
    n_bg = bg.shape[0]
    total = bit_counts(bg, np.arange(n_bg))
    assert np.array_equal(total, C.base_count(tag)), "presence bits and background counts disagree"
    titles = [t.lower() for t in bmeta["titles"]]
    out = {"model": tag, "n_background_plots": n_bg}

    # ---- Batman on the same footing: original texts, units present in k of n works, for random subsets of its works
    st = C.Setting(tag, icon=False)
    Po = st.P["orig"][X.BAT]
    base_b = st.base_count
    out["batman"] = {"n_works": 32, **canon_counts(Po.sum(0), 32, base_b, n_bg)}

    # ---- other franchises
    fr = np.load(d / "fr_present30.bits.npy", mmap_mode="r")
    fmeta = json.load(open(d / "fr_meta.json"))
    names = sorted(set(fmeta["franchise"]), key=fmeta["franchise"].index)
    out["franchises"] = {}
    for name in names:
        rows = np.flatnonzero(np.array(fmeta["franchise"]) == name)
        outside = np.array([name.lower() not in t for t in titles])
        base = total - bit_counts(bg, np.flatnonzero(~outside)) if (~outside).any() else total
        res = {"n_films": int(len(rows)), "background_plots_left_out_of_the_base_rate": int((~outside).sum()),
               **canon_counts(bit_counts(fr, rows), len(rows), base, int(outside.sum()))}
        sub = [canon_counts(Po[rng.choice(32, len(rows), replace=False)].sum(0), len(rows), base_b, n_bg) for _ in range(n_draws)]
        res["batman_subsets_of_the_same_size"] = {k: {"mean": float(np.mean([x[k] for x in sub])), "min": int(np.min([x[k] for x in sub]))}
                                                   for k in ("at_least_72pct", "at_least_90pct", "at_least_6_films")}
        out["franchises"][name] = res
        print(f"{tag} {name}: {len(rows)} films, units in 72%+ {res['at_least_72pct']}, 90%+ {res['at_least_90pct']} | "
              f"Batman subsets: {res['batman_subsets_of_the_same_size']['at_least_72pct']['mean']:.0f}, {res['batman_subsets_of_the_same_size']['at_least_90pct']['mean']:.0f}", flush=True)

    # ---- genre sets: 32 plots of one genre play the franchise
    genres = [str(g).strip().lower() for g in bmeta["genres"]]
    out["genre_sets"] = {}
    for g in GENRES:
        pool = np.flatnonzero(np.array(genres) == g)
        if len(pool) < 64:
            out["genre_sets"][g] = {"n_plots_of_this_genre": int(len(pool)), "skipped": "fewer than 64 plots"}
            continue
        draws = []
        for _ in range(n_draws):
            rows = rng.choice(pool, 32, replace=False)
            c = bit_counts(bg, rows)
            x = canon_counts(c, 32, total - c, n_bg - 32)
            x["at_least_23"], x["at_least_29"] = x.pop("at_least_72pct"), x.pop("at_least_90pct")
            draws.append(x)
        out["genre_sets"][g] = {"n_plots_of_this_genre": int(len(pool)), "n_draws": n_draws,
                                **{k: {"mean": float(np.mean([x[k] for x in draws])), "max": int(np.max([x[k] for x in draws]))}
                                   for k in ("at_least_6_films", "at_least_23", "at_least_29")}}
        print(f"{tag} genre {g}: units in 6+ of 32: {out['genre_sets'][g]['at_least_6_films']['mean']:.0f}, 23+: {out['genre_sets'][g]['at_least_23']['mean']:.1f}", flush=True)

    # ---- Batman-like films (exploratory)
    z = np.load(C.RESULTS / f"canon_{tag}.npz")
    core = z["core"].astype(np.int64)
    if len(core):
        share = bit_columns(bg, core).mean(1)
        order = np.argsort(-share, kind="stable")
        rank = np.empty(n_bg, int)
        rank[order] = np.arange(1, n_bg + 1)
        hero = np.array(["superhero" in g for g in genres])
        rel = [i for i, t in enumerate(titles) if any(r in t for r in RELATIVES)]
        out["batman_like_films"] = {"n_core_units": int(len(core)), "share_of_core_median_plot": float(np.median(share)),
                                    "top_30": [{"title": bmeta["titles"][i], "year": bmeta["years"][i], "genre": bmeta["genres"][i], "share_of_core": round(float(share[i]), 3)}
                                               for i in order[:30]],
                                    "superhero_tagged_plots": {"n": int(hero.sum()), "median_rank": float(np.median(rank[hero])) if hero.any() else None,
                                                               "in_top_100": int((rank[hero] <= 100).sum())},
                                    "versions_of_the_relatives": [{"title": bmeta["titles"][i], "year": bmeta["years"][i], "rank": int(rank[i]),
                                                                   "share_of_core": round(float(share[i]), 3)} for i in rel]}
    C.save(out, f"franchises_{tag}.json")


if __name__ == "__main__":
    for tag in sys.argv[1:] or [t for t in C.MODELS if (C.ACTS / t / "fr_present30.bits.npy").exists()]:
        run_model(tag)
