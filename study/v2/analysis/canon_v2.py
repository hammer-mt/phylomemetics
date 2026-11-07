"""Batman's canon under the v2 rule (PLAN.md, Section 13), and the pools the explained units are chosen from (Section 10).

Canon counts: units present in at least 6, 23 and 29 of the 32 Batman works, among the distinctive units and among all
units, beside the same counts for random sets of 32 of the 69 ordinary films. Under the 3% cap a random set of ordinary
films cannot have a canon whatever Batman is like, so the comparison that could fail is the one without the cap.

Pools, fixed here before anyone reads an example:
  candidates           distinctive units present in at least 6 Batman works (original texts); they are merged into families
  name_free_core       canon units (at least 80% of the 32 works, original texts) present in at least 70% of the name-free works
  discriminators       the 40 distinctive units with the largest gap between the share of name-free Batman works and the
                       share of the 8 hero films they are present in
  common_patterns      the 40 units with a base rate above 3% and at most 30% and the largest gap between the share of
                       Batman works (original texts) they are present in and their base rate: chosen without the cap
  inherited, band      20 units drawn from E2's optional units present in both members of at least one documented source
                       and adaptation pair, and 20 from the rest of E2's optional band; units whose name-free presence
                       depends on the placeholder codes are not drawn, nor are weak poles
  weak_poles           every pool unit whose scale is under 25% of the scale of its neuron's other pole (marked, not removed)

    cd study && ../.venv/bin/python v2/analysis/canon_v2.py [model-tag ...]

Writes results/v2/canon_counts_<tag>.json, results/v2/<tag>/pools.npz and results/v2/<tag>/sentence_units.npy (every
unit whose sentence-level activations are fetched next).
"""
import json
import sys

import numpy as np

import common as C

import experiments as X

BAT, FILMS, HERO, N = X.BAT, X.FILMS, X.HERO_FILM_COUSINS, X.N
COUSINS = np.concatenate([X.HERO_FILM_COUSINS, X.LIT_COUSINS])
MIN_WORKS = 6


def counts_at(n_present):
    return {f"at_least_{k}": int((n_present >= k).sum()) for k in (6, 23, 29)}


def reserved_units(tag):
    """Every number in the pre-registration's list of reserved development units (preregistration/dev_units.json). Units
    with these indices are left out of the explained sample, so that v2 never looks at a unit the planted-story test holds back."""
    out = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, int) and not isinstance(x, bool):
            out.add(x)
    walk(json.load(open("../preregistration/dev_units.json")))
    return np.array(sorted(out), dtype=np.int64)


def code_dependent(tag, units, theta=C.THETA):
    """Units at least half of whose name-free presences in the Batman works exist only because a placeholder code's
    own token carried the peak."""
    cut = int(np.ceil(theta * C.STEP - 1e-6))
    with_codes = (np.load(C.ACTS / tag / "anon_codes_main_top1.u8.npy", mmap_mode="r")[BAT][:, units] >= cut).sum(0)
    masked = (np.load(C.ACTS / tag / "anon_main_top1.u8.npy", mmap_mode="r")[BAT][:, units] >= cut).sum(0)
    return (with_codes >= 2) & (with_codes - masked >= 0.5 * with_codes)


def run_model(tag, n_pseudo=50):
    rng = np.random.default_rng(C.SEED)
    st = C.Setting(tag, icon=False)
    d = st.distinctive(C.CAP)
    Po, Pa = st.P["orig"], st.P["anon"]
    nb, nba = Po[BAT].sum(0), Pa[BAT].sum(0)
    res = {"model": tag, "n_units_present_in_some_work": int(len(st.units)), "n_distinctive": int(d.sum()),
           "batman_distinctive_units": counts_at(nb[d]), "batman_all_units": counts_at(nb),
           "batman_name_free_distinctive_units": counts_at(nba[d]), "batman_name_free_all_units": counts_at(nba)}
    for key, mask in (("distinctive_units", d), ("all_units", np.ones(len(d), bool))):
        draws = [counts_at(Po[rng.choice(FILMS, 32, replace=False)][:, mask].sum(0)) for _ in range(n_pseudo)]
        res[f"random_sets_of_32_ordinary_films_{key}"] = {"n_draws": n_pseudo, **{k: {"mean": float(np.mean([p[k] for p in draws])),
                                                                                    "max": int(np.max([p[k] for p in draws]))} for k in draws[0]}}
        res[f"batman_above_every_random_set_at_least_6_{key}"] = bool(all(counts_at(nb[mask])["at_least_6"] > p["at_least_6"] for p in draws))
    cand = np.flatnonzero(d & (nb >= MIN_WORKS))
    tier = np.where(nb[cand] >= 29, "core", np.where(nb[cand] >= 23, "common", "optional"))
    own = Po[COUSINS][:, cand].sum(0) <= 3
    res["candidate_units"] = int(len(cand))
    res["tiers_units"] = {t: {"n": int((tier == t).sum()), "batman_own": int(((tier == t) & own).sum()), "shared_with_relatives": int(((tier == t) & ~own).sum())}
                          for t in ("core", "common", "optional")}
    core = cand[tier == "core"]
    if len(core):
        res["core_share_by_work"] = {X.label[w]: round(float(Po[w, core].mean()), 4) for w in np.r_[BAT, COUSINS]}
        res["core_share_by_work_name_free"] = {X.label[w]: round(float(Pa[w, core].mean()), 4) for w in np.r_[BAT, COUSINS]}
        res["core_share_ordinary_films_mean"] = float(Po[FILMS][:, core].mean())
    L, D = C.geometry(tag)
    U = L * D
    layer = (st.units[cand] % U) // D
    res["candidates_by_layer_quarter"] = {f"q{q + 1}": int(((layer * 4) // L == q).sum()) for q in range(4)}
    res["candidates_negative_pole"] = int((st.units[cand] >= U).sum())

    # ---- pools for the explained sample
    reserved = reserved_units(tag) if tag in ("qwen3-8b", "llama3.1-8b") else np.zeros(0, np.int64)
    free = ~np.isin(st.units, reserved)
    canon = np.flatnonzero(d & (Po[BAT].mean(0) >= 0.8))
    nf_core = canon[Pa[BAT][:, canon].mean(0) >= 0.7]
    nf_core = nf_core[np.lexsort((st.units[nf_core], -nb[nf_core], -nba[nf_core]))]
    gap = Pa[BAT].mean(0) - Pa[HERO].mean(0)
    dd = np.flatnonzero(d & free)
    disc = dd[np.lexsort((st.units[dd], -Pa[BAT][:, dd].mean(0), -gap[dd]))][:40]
    band = (nba >= 2) & (nba <= 16) & free
    pair = np.zeros(len(d), bool)
    for t, srcs in X.ADAPTATIONS.items():
        for s in srcs:
            pair |= Pa[X.L[t]] & Pa[X.L[s]]
    dep = np.zeros(len(d), bool)
    b = np.flatnonzero(band)
    dep[b] = code_dependent(tag, st.units[b])
    weak = C.dominance(tag)[st.units] < 0.25
    in_pair, rest = np.flatnonzero(band & pair & ~dep & ~weak), np.flatnonzero(band & ~pair & ~dep & ~weak)
    mid = np.flatnonzero((st.base > C.CAP + 1e-12) & (st.base <= 0.30 + 1e-12) & free)
    lift = Po[BAT][:, mid].mean(0) - st.base[mid]
    patterns = mid[np.lexsort((st.units[mid], -Po[BAT][:, mid].mean(0), -lift))][:40]
    inherited = np.sort(rng.choice(in_pair, min(20, len(in_pair)), replace=False))
    band_rest = np.sort(rng.choice(rest, min(20, len(rest)), replace=False))
    res["pools"] = {"name_free_core": int(len(nf_core)), "canon_all32": int(len(canon)), "discriminators": int(len(disc)),
                    "e2_band": int(band.sum()), "e2_band_in_a_documented_pair": int((band & pair).sum()),
                    "e2_band_code_dependent_not_drawn": int((band & dep).sum()), "e2_band_weak_poles_not_drawn": int((band & weak & ~dep).sum()),
                    "common_patterns": int(len(patterns)), "common_patterns_gap_min": float(np.sort(lift)[::-1][:40].min()) if len(mid) else None,
                    "common_patterns_base_rate_median": float(np.median(st.base[patterns])) if len(patterns) else None,
                    "units_between_3_and_30_percent": int(len(mid)), "reserved_units_left_out": int((~free).sum()),
                    "discriminator_gap_min": float(gap[disc].min()) if len(disc) else None}
    cand_free = cand[free[cand]]
    (C.RESULTS / tag).mkdir(parents=True, exist_ok=True)
    g = lambda idx: st.units[idx].astype(np.int64)      # noqa: E731
    union = np.unique(np.concatenate([g(cand_free), g(nf_core[free[nf_core]]), g(disc), g(patterns), g(inherited), g(band_rest)]))
    dom = C.dominance(tag)
    np.savez_compressed(C.RESULTS / tag / "pools.npz", candidates=g(cand_free), n_batman=nb[cand_free], name_free_core=g(nf_core[free[nf_core]]),
                        discriminators=g(disc), common_patterns=g(patterns), inherited=g(inherited), band_rest=g(band_rest), canon_all32=g(canon),
                        weak_poles=union[dom[union] < 0.25])
    res["pools"]["weak_poles_among_pool_units"] = int((dom[union] < 0.25).sum())
    res["pools"]["weak_poles_among_canon_units"] = int((dom[g(canon)] < 0.25).sum())
    np.save(C.RESULTS / tag / "sentence_units.npy", union)
    res["units_for_the_sentence_pass"] = int(len(union))
    res["bounded_by_the_cap_by_construction"] = ["random_sets_of_32_ordinary_films_distinctive_units"]
    C.save(res, f"canon_counts_{tag}.json")
    print(tag, "candidates", len(cand), res["tiers_units"], "| pools", res["pools"], "| sentence pass:", len(union), "units")
    return res


if __name__ == "__main__":
    for tag in sys.argv[1:] or [t for t in C.MODELS if (C.ACTS / t / "scale.npy").exists()]:
        run_model(tag)
