"""An audit of the remakes decision test from its committed outputs: statistics and the size of the effect.

Run from study/:   ../.venv/bin/python results/v2/second/audit/statistics.py
Reads only committed outputs (results/v2/second/remakes_*.json, data/second/remakes/pairs.json); refits nothing, loads no
film text and no word-measure cache. Single-threaded, a few hundred megabytes, well under two minutes.
Writes results/v2/second/audit/statistics.json: every number with a one-line description ("what") and where it comes from
("source": key paths in the committed files, or "computed" with the inputs named).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import difflib
import glob
import json
import math
import re
import sys
import time
import unicodedata

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)                                   # results/v2/second
STUDY = os.path.abspath(os.path.join(RES, "..", "..", ".."))
PAIRS = os.path.join(STUDY, "data", "second", "remakes", "pairs.json")
PRIMARY_MODEL = "qwen3-32b"
KEY = "primary | set"
SEED = 20261108                                               # the registered seed of the sign flips
BOOT_SEED = 20261109
BOOT = 20000
STANDIN_Z = None                                              # filled from the words file
OUT = {}
T0 = time.time()


def plain(o):
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return [plain(v) for v in o.tolist()]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return None if (math.isnan(x) or math.isinf(x)) else x
    return o


def put(section, name, value, what, source):
    OUT.setdefault(section, {})[name] = {"value": plain(value), "what": what, "source": source}


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------ the registered pieces
def sign_flip_p(d, n_flips=100000, seed=SEED, exact_up_to=20):
    """remake_test.sign_flip_p, line for line (same seed, same draws), so that the committed p values can be reproduced."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    obs = d.sum()
    tol = 1e-12 * np.abs(d).sum()
    if len(d) <= exact_up_to:
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


def zstat(d):
    d = np.asarray(d, dtype=float)
    ss = math.sqrt(float((d ** 2).sum()))
    return float(d.sum() / ss) if ss > 0 else 0.0


class Fit:
    """One model's primary analysis: the entries' contributions and the family sums, in the test's own family order."""

    def __init__(self, res):
        self.res = res
        self.A = A = res["analyses"][KEY]
        self.E = E = A["entries_table"]
        self.fam_names = sorted({e["family"] for e in E}, key=str)             # remake_test.Stack: sorted by str
        index = {f: i for i, f in enumerate(self.fam_names)}
        self.fam = np.array([index[e["family"]] for e in E])
        self.c = np.array([e["contribution"] for e in E])
        self.p_words = np.array([e["words_probability_of_the_source"] for e in E])
        self.open = 1.0 - self.p_words
        self.rank_words = np.array([e["words_rank_of_the_source"] for e in E])
        self.rank_units = np.array([e["unit_rank_of_the_source"] for e in E])
        self.pool = np.array([e["pool"] for e in E])
        self.copied = np.array([e["copied"] for e in E], bool)
        self.copy_share = np.array([e["copy_share"] for e in E])
        self.rec = np.array([e["recognised"] for e in E], bool)
        self.rec_both = np.array([e["recognised_both_films"] for e in E], bool)
        self.ref = np.array([bool(e["source_is_a_document_of_v2_reference_corpus"]) for e in E], bool)
        self.n_fam = len(self.fam_names)
        self.D = self.family_sum(self.c)
        self.key = [(e["remake"], e["source"]) for e in E]
        rows = {r["family"]: r for r in A["family_sums"]["rows"]}
        self.D_file = np.array([rows[f]["D"] for f in self.fam_names])
        self.open_file = np.array([rows[f]["open_mass"] for f in self.fam_names])
        self.fam_remakes = [rows[f]["remakes"] for f in self.fam_names]

    def family_sum(self, x, keep=None):
        x = np.asarray(x, float)
        if keep is not None:
            x = np.where(keep, x, 0.0)
        return np.bincount(self.fam, weights=x, minlength=self.n_fam)

    def summary(self, keep=None, with_p=True):
        """remakes.py Analysis.summary from the entries' contributions."""
        k = np.ones(len(self.c), bool) if keep is None else np.asarray(keep, bool)
        d = self.family_sum(self.c, k)
        out = {"entries": int(k.sum()), "families": len(set(self.fam[k].tolist())), "T": float(d.sum()), "Z": zstat(d),
               "families_positive": int((d > 0).sum()), "families_negative": int((d < 0).sum()),
               "effective_pairs_left_open": float((self.open * k).sum())}
        if with_p:
            out["p"] = sign_flip_p(d)
        return out, d


def noise_p(z, Zs):
    """Share of the pure-noise draws whose Z is at or above z, the observed value counted as one draw (PLAN 16.3)."""
    return float((1 + (Zs >= z).sum()) / (len(Zs) + 1))


def boot_ci(stat_draws, q=(0.025, 0.975)):
    return [float(np.quantile(stat_draws, q[0])), float(np.quantile(stat_draws, q[1]))]


def bca(theta, draws, jack, q=(0.025, 0.975)):
    """Bias-corrected and accelerated bootstrap interval."""
    draws = np.asarray(draws)
    z0 = stats.norm.ppf(min(max((draws < theta).mean(), 1e-9), 1 - 1e-9))
    jm = jack.mean()
    num, den = ((jm - jack) ** 3).sum(), 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    a = num / den if den > 0 else 0.0
    out = []
    for x in q:
        zq = stats.norm.ppf(x)
        adj = stats.norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))
        out.append(float(np.quantile(draws, adj)))
    return out


def _fold(t, title=False):
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    if title:
        t = re.sub(r"\s*\([^)]*\)\s*$", "", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return " ".join(w for w in t.split() if w not in ("the", "a", "an"))


def names_title(reply, title):
    """A looser reading than the registered title rule, for this audit only: 'exact' when the title's words stand together in
    the reply, 'close' when some run of the reply's words is at least 80% similar to the title (difflib), else None."""
    r, t = _fold(reply), _fold(title, True)
    if len(t.replace(" ", "")) < 4:
        return "exact" if r == t else None
    if re.search(r"\b" + re.escape(t) + r"\b", r):
        return "exact"
    if len(t.replace(" ", "")) < 5:
        return None
    rw, n, best = r.split(), len(t.split()), 0.0
    for i in range(len(rw)):
        for m in (n - 1, n, n + 1):
            seg = " ".join(rw[i:i + m]) if m > 0 else ""
            if seg:
                best = max(best, difflib.SequenceMatcher(None, seg, t).ratio(), difflib.SequenceMatcher(None, seg.replace(" ", ""), t.replace(" ", "")).ratio())
    return "close" if best >= 0.8 else None


def main():
    global STANDIN_Z
    res = load(f"remakes_{PRIMARY_MODEL}.json")
    words = load("remakes_words.json")
    null = load("remakes_null_size.json")
    recog = load(f"remakes_recognition_{PRIMARY_MODEL}.json")
    with open(PAIRS) as f:
        pairs = json.load(f)
    P = {(q["remake"], q["source"]): q for q in pairs["pairs"]}
    F = Fit(res)
    A, E = F.A, F.E
    R = f"remakes_{PRIMARY_MODEL}.json: analyses['{KEY}']"
    NZ = null["by_columns"]["39 columns (registered)"]
    Zs = np.array(NZ["Z_sorted"], float)
    STANDIN_Z = float(words["primary_calibration"]["standins"]["summary"]["z_max"])
    Zobs, Tobs = A["slope"]["Z"], A["slope"]["T"]
    rng = np.random.default_rng(BOOT_SEED)
    put("0_inputs", "files", {"model_file": f"remakes_{PRIMARY_MODEL}.json", "git_head_of_the_model_file": res["git_head"], "noise_draws": int(len(Zs)),
                              "other_model_files_present": sorted(os.path.basename(p) for p in model_files() if PRIMARY_MODEL not in p)},
        "the committed files this audit read", "results/v2/second/")

    # =========================================================================================== 8. arithmetic checks
    S = "8_arithmetic_checks"
    d_all = F.D
    put(S, "T_from_family_sums_minus_committed", float(F.D_file.sum() - Tobs), "sum of family_sums.rows D minus slope.T", R + ".family_sums, .slope")
    put(S, "T_from_entries_minus_committed", float(F.c.sum() - Tobs), "sum of entries_table contributions minus slope.T", R + ".entries_table, .slope")
    put(S, "Z_recomputed_minus_committed", float(zstat(F.D_file) - Zobs), "T / sqrt(sum of D squared) minus slope.Z", R + ".family_sums, .slope")
    put(S, "largest_gap_family_sum_vs_its_entries", float(np.abs(F.D - F.D_file).max()), "largest absolute difference between a family's D and the sum of its entries' contributions", R)
    put(S, "largest_gap_open_mass_vs_its_entries", float(np.abs(F.family_sum(F.open) - F.open_file).max()), "largest absolute difference between a family's open_mass and the sum of (1 - words probability) of its entries", R)
    put(S, "effective_pairs_recomputed_minus_committed", float(F.open.sum() - A["slope"]["effective_pairs_left_open"]), "sum of (1 - words_probability_of_the_source) minus slope.effective_pairs_left_open", R)
    put(S, "sources_first_by_words_recomputed", [int((F.rank_words == 1).sum()), A["gain"]["sources_first_words"]], "entries with words_rank_of_the_source = 1, and gain.sources_first_words", R)
    put(S, "sources_first_by_units_recomputed", [int((F.rank_units == 1).sum()), A["heredity"]["hub_corrected"]["all_entries"]["source_strictly_above_every_candidate"]],
        "entries with unit_rank_of_the_source = 1, and heredity.hub_corrected.all_entries.source_strictly_above_every_candidate", R)
    put(S, "effective_number_of_families_recomputed_minus_committed", float(np.abs(d_all).sum() ** 2 / (d_all ** 2).sum() - A["family_sums"]["effective_number_of_families"]),
        "(sum |D|)^2 / sum D^2 minus the committed value", R + ".family_sums")
    subs = {"all_entries": np.ones(len(E), bool), "not_copied": ~F.copied, "copied": F.copied, "copy_share_under_1pct": F.copy_share < 0.01, "recognised": F.rec,
            "not_recognised": ~F.rec, "recognised_both_films": F.rec_both, "not_recognised_both_films": ~F.rec_both,
            "contributions_set_to_zero_where_the_held_out_source_probability_is_0.99_or_more": F.p_words < 0.99,
            "source_is_a_document_of_v2_reference_corpus": F.ref, "source_is_not_a_document_of_v2_reference_corpus": ~F.ref}
    gaps, p_gaps = {}, {}
    for name, keep in subs.items():
        mine, _ = F.summary(keep)
        theirs = A["subsets"][name]
        gaps[name] = {k: (mine[k] - theirs[k]) for k in ("entries", "families", "T", "families_positive", "families_negative", "effective_pairs_left_open")}
        gaps[name]["Z"] = mine["Z"] - theirs["Z"]
        p_gaps[name] = [mine["p"], theirs["p"]]
    worst = max(abs(v) for name, g in gaps.items() for k, v in g.items() if not (name == "copied" and k == "Z"))
    put(S, "subsets_recomputed_largest_gap", worst, "largest absolute difference, over the eleven committed subsets and their fields (entries, families, T, Z, positive, negative, open pairs), between the committed "
        "value and the value recomputed from entries_table (the Z of the copied subset left aside: see copied_subset_Z)", R + ".subsets, .entries_table")
    put(S, "subset_sign_flip_p_recomputed_and_committed", p_gaps, "each subset's sign-flip p recomputed with the registered seed and draws, beside the committed p", R + ".subsets")
    put(S, "copied_subset_Z", {"T": A["subsets"]["copied"]["T"], "Z_committed": A["subsets"]["copied"]["Z"], "Z_recomputed": F.summary(F.copied, False)[0]["Z"],
                              "largest_absolute_contribution_of_a_copied_entry": float(np.abs(F.c[F.copied]).max())},
        "the copied subset's Z (-1.41 in the file) is a ratio of rounding residues near 1e-13: it is 0/0 and carries no information", R + ".subsets.copied")
    for a, b in (("not_copied", "copied"), ("recognised", "not_recognised"), ("recognised_both_films", "not_recognised_both_films"),
                 ("source_is_a_document_of_v2_reference_corpus", "source_is_not_a_document_of_v2_reference_corpus")):
        sa, sb = A["subsets"][a], A["subsets"][b]
        put(S, f"adds_up__{a}__plus__{b}", {"T_gap": sa["T"] + sb["T"] - Tobs, "entries": sa["entries"] + sb["entries"],
                                           "open_pairs_gap": sa["effective_pairs_left_open"] + sb["effective_pairs_left_open"] - A["slope"]["effective_pairs_left_open"]},
            "committed T, entries and open pairs of the two complementary subsets, added and compared with all entries", R + ".subsets")
    lo = A["subsets"]["each_of_the_largest_families_left_out"]
    put(S, "largest_families_left_out_gap", max(abs(v["T"] + v["family_sum_left_out"] - Tobs) for v in lo.values()), "largest |T without the family + the family's sum - T|, five families", R + ".subsets.each_of_the_largest_families_left_out")
    put(S, "decision_block_equals_the_analysis", bool(res["decision"]["slope"] == A["slope"] and res["decision"]["gain"] == A["gain"]["gain"] and res["decision"]["recall_rule"] == A["recall_rule"]),
        "decision.slope, decision.gain and decision.recall_rule equal the same blocks of the primary analysis", f"remakes_{PRIMARY_MODEL}.json: decision; analyses")
    put(S, "recall_rule_equals_the_not_recognised_subset", bool(A["recall_rule"]["T_n"] == A["subsets"]["not_recognised"]["T"] and A["recall_rule"]["p_n"] == A["subsets"]["not_recognised"]["p"]),
        "recall_rule.T_n and p_n equal subsets.not_recognised", R)
    put(S, "primary_plane_row_equals_the_slope", bool(A["planes"]["rows"]["top1_30"] == A["slope"]), "planes.rows.top1_30 equals slope", R + ".planes")
    put(S, "stand_ins_largest_Z_three_places", [STANDIN_Z, res["decision"]["standins_largest_Z"], null["stand_ins_largest_Z"]],
        "the stand-ins' largest Z in the words file, in the decision block and in the noise file", "remakes_words.json: primary_calibration.standins.summary.z_max; decision.standins_largest_Z; remakes_null_size.json: stand_ins_largest_Z")
    put(S, "noise_file_rates_recomputed", {"share_Z_above_the_stand_ins_recomputed": float((Zs > STANDIN_Z).mean()), "committed": NZ["share_Z_above_the_stand_ins_largest_Z"],
                                          "sd_Z_recomputed": float(Zs.std()), "sd_Z_committed": NZ["sd_Z"]},
        "the noise draws' share of Z above the stand-ins' largest Z and the standard deviation of Z, recomputed from Z_sorted (rounded to 4 places)", "remakes_null_size.json: by_columns['39 columns (registered)']")
    rec_rows = {(q["remake"], q["source"]): q for q in recog["pairs"]}
    put(S, "recognition_flags_equal_the_recognition_file", bool(all(rec_rows[k]["recognised"] == bool(r) and rec_rows[k]["recognised_both_films"] == bool(b) for k, r, b in zip(F.key, F.rec, F.rec_both))),
        "entries_table's two recognition flags equal the 'pairs' rows of the recognition file", f"remakes_recognition_{PRIMARY_MODEL}.json: pairs")
    pair_gaps = [k for k, e in zip(F.key, E) if k not in P or P[k]["family"] != e["family"] or bool(P[k]["copied"]) != e["copied"] or P[k]["block"] != "indian_cross_industry" or P[k]["wording"] != "firm"]
    put(S, "entries_that_disagree_with_pairs_json", pair_gaps, "entries whose family, copied flag, block or wording differ from pairs.json (none expected)", "data/second/remakes/pairs.json: pairs")
    put(S, "counts_of_the_primary_set_against_the_plan", {"entries": len(E), "remakes": len({k[0] for k in F.key}), "sources": len({k[1] for k in F.key}), "families": F.n_fam, "copied": int(F.copied.sum()),
                                                          "copy_share_under_1pct": int((F.copy_share < 0.01).sum()), "source_in_the_reference_corpus": int(F.ref.sum()),
                                                          "plan_section_15_says": [220, 217, 185, 175, 100, 96, 137]},
        "the primary set's counts recomputed from entries_table, beside PLAN.md Section 15 (220 pairs, 217 remakes, 185 sources, 175 families, 100 copied, 96 under 1%, 137 reference sources)", R + ".entries_table; v2/second/PLAN.md Section 15")
    tiny = np.abs(d_all) <= 1e-6
    put(S, "families_counted_as_non_zero_that_are_rounding_residue", {"non_zero": int((d_all != 0).sum()), "of_which_abs_at_most_1e-6": int(((d_all != 0) & tiny).sum()),
                                                                     "of_which_abs_at_most_1e-12": int(((d_all != 0) & (np.abs(d_all) <= 1e-12)).sum()),
                                                                     "positive_among_them": int(((d_all > 0) & tiny).sum()), "negative_among_them": int(((d_all < 0) & tiny).sum())},
        "'47 positive against 57 negative' counts every family whose sum is not exactly 0.0; this many of the 104 are zero to within 1e-6 (the word model is certain of the source and the sum is a rounding residue)", R + ".family_sums.rows")

    # ================================================================================================= 1. influence
    S = "1_influence"
    order = np.argsort(-d_all, kind="stable")
    ladder = []
    for k in range(0, 21):
        x = np.delete(d_all, order[:k])
        z = zstat(x)
        ladder.append({"k": k, "family_removed": None if k == 0 else F.fam_names[order[k - 1]], "its_remakes": None if k == 0 else F.fam_remakes[order[k - 1]],
                       "its_sum": None if k == 0 else float(d_all[order[k - 1]]), "T": float(x.sum()), "Z": z, "sign_flip_p": sign_flip_p(x), "noise_calibrated_p": noise_p(z, Zs)})
    put(S, "largest_families_removed_one_after_another", ladder, "T, Z, sign-flip p (registered seed) and noise-calibrated p after the k families with the largest sums are removed, k = 0 to 20 "
        "(the noise draws are of the whole list, so the last column is a guide after removals)", R + ".family_sums.rows; remakes_null_size.json")
    z_k = [r["Z"] for r in ladder]
    q95 = float(np.quantile(Zs, 0.95))
    put(S, "noise_Z_95th_percentile", q95, "95th percentile of Z over the 2,000 pure-noise draws", "remakes_null_size.json: by_columns['39 columns (registered)'].Z_sorted")
    put(S, "families_to_remove_for_Z_under_the_stand_ins", next(k for k, z in enumerate(z_k) if z < STANDIN_Z), f"smallest k for which Z falls below the stand-ins' largest Z ({STANDIN_Z:.4f})", "computed from the ladder")
    put(S, "families_to_remove_for_Z_under_the_noise_95th_percentile", next(k for k, z in enumerate(z_k) if z < q95), "smallest k for which Z falls below the 95th percentile of the pure-noise Z", "computed from the ladder")
    put(S, "families_to_remove_for_sign_flip_p_at_or_above_0.05", next(r["k"] for r in ladder if r["sign_flip_p"] >= 0.05), "smallest k for which the sign-flip p is 0.05 or more", "computed from the ladder")
    put(S, "families_to_remove_for_T_at_or_below_zero", next(r["k"] for r in ladder if r["T"] <= 0), "smallest k for which the sum is no longer positive", "computed from the ladder")
    loo = np.array([zstat(np.delete(d_all, i)) for i in range(F.n_fam)])
    put(S, "leave_one_family_out_Z_range", [float(loo.min()), float(loo.max())], "smallest and largest Z when any single family is left out", R + ".family_sums.rows")
    wins = []
    for c in (4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.1):
        x = np.clip(d_all, -c, c)
        z = zstat(x)
        wins.append({"family_sums_clipped_at": c, "families_clipped": int((np.abs(d_all) > c).sum()), "T": float(x.sum()), "Z": z, "sign_flip_p": sign_flip_p(x), "noise_calibrated_p": noise_p(z, Zs)})
    put(S, "family_sums_winsorised", wins, "the family sums clipped to plus or minus c: T, Z, sign-flip p, noise-calibrated p", R + ".family_sums.rows")
    wins = []
    for c in (3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25):
        x = F.family_sum(np.clip(F.c, -c, c))
        z = zstat(x)
        wins.append({"contributions_clipped_at": c, "entries_clipped": int((np.abs(F.c) > c).sum()), "T": float(x.sum()), "Z": z, "sign_flip_p": sign_flip_p(x), "noise_calibrated_p": noise_p(z, Zs)})
    put(S, "entry_contributions_winsorised", wins, "the entries' contributions clipped to plus or minus c before the family sums: T, Z, sign-flip p, noise-calibrated p", R + ".entries_table")
    srt = np.sort(d_all)[::-1]
    put(S, "share_of_T_in_the_largest_families", {str(k): float(srt[:k].sum() / Tobs) for k in (1, 2, 3, 5, 10, 15)}, "share of T held by the k largest family sums", R + ".family_sums.rows")
    put(S, "share_of_sum_of_squares_in_the_largest_families", {str(k): float((srt[:k] ** 2).sum() / (d_all ** 2).sum()) for k in (1, 2, 3, 5, 10, 15)}, "share of the sum of squared family sums held by the k largest", R + ".family_sums.rows")
    put(S, "effective_number_of_families", A["family_sums"]["effective_number_of_families"], "(sum |D|)^2 / sum D^2, as committed", R + ".family_sums")
    # the pairs behind the largest contributions, with what pairs.json says of them
    eorder = np.argsort(-np.abs(F.c), kind="stable")
    n_entries_of_remake = {}
    for k in F.key:
        n_entries_of_remake[k[0]] = n_entries_of_remake.get(k[0], 0) + 1
    all_pairs = set(P)
    sources_of = {}
    for (r_, s_) in all_pairs:
        sources_of.setdefault(r_, set()).add(s_)

    def middle_film(k):
        """A film m with (remake, m) and (m, source) both kept pairs of pairs.json: the entry's source is then two steps back."""
        return sorted(m for m in sources_of.get(k[0], ()) if m != k[1] and (m, k[1]) in all_pairs)

    def flags(i):
        e, q = E[i], P[F.key[i]]
        text = " ".join(s["sentence"] for s in q["stated"])
        own = " ".join(s["sentence"] for s in q["stated"] if s["on_page"] == e["remake"])
        years_own = sorted({int(y) for y in re.findall(r"\b(1[89]\d\d|20[0-2]\d)\b", own) if int(y) <= q["remake_year"]})
        out = {"reboot": "reboot" in text.lower(), "third_reader": any(s["readers"] > 2 for s in q["stated"]),
               "stated_only_off_the_remakes_page": all(s["on_page"] != e["remake"] for s in q["stated"]),
               "two_documented_sources": n_entries_of_remake[e["remake"]] > 1, "two_steps_from_the_source_through": middle_film(F.key[i]),
               "table_year_of_the_source_contradicts_the_remakes_page": bool(years_own) and min(abs(y - q["source_year"]) for y in years_own) > 2,
               "years_on_the_remakes_page": years_own, "a_text_of_19_sentences_or_fewer": min(q["remake_sentences"], q["source_sentences"]) <= 19}
        return out

    FL = [flags(i) for i in range(len(E))]
    by_hand = {("Judwaa 2", "Hello Brother (1994 film)"): "Judwaa 2's own article calls it a reboot of Judwaa (1997), itself the remake of Hello Brother: pairs.json holds Judwaa 2 <- Judwaa as a loose same-industry "
                                                          "pair. It is in the primary set as 'firm' only through one sentence on the source's page ('... Judwaa (1997) and its reboot Judwaa 2 (2017)'), on "
                                                          "which the two readers differed. Judwaa is of the same family, so it is not among Judwaa 2's candidates.",
               ("Kondapalli Raja", "Khudgarz"): "Annamalai's article says Kondapalli Raja is the Telugu remake of Annamalai, itself a remake of Khudgarz; Kondapalli Raja's own article says it remakes Khudgarz. "
                                                "Both are entries; Annamalai is of the same family and is not a candidate here.",
               ("Tadakha", "Vettai (film)"): "The film table gives 'Vettai (film)' the year 1986; all three statements say Vettai is a film of 2012. The year gap of this pair is 1, not 27.",
               ("Bhale Dongalu", "Bunty Aur Babli"): "The sentence names two sources: 'A remake of the film Bonnie and Clyde and the Hindi film Bunty Aur Babli (2005)'.",
               ("Khoka 420", "Brindavanam (2010 film)"): "The word model ranks the source 1,676th of 2,975 (below the middle of the pool); the units rank it first, and the model, shown Khoka 420's summary, answers "
                                                         "'Brindavanam' to both recognition questions (counted as recognised)."}
    top = []
    for i in eorder[:20]:
        e, q = E[i], P[F.key[i]]
        top.append({"remake": e["remake"], "source": e["source"], "family": e["family"], "contribution": e["contribution"], "share_of_T": e["contribution"] / Tobs,
                    "words_probability_of_the_source": e["words_probability_of_the_source"], "words_rank": e["words_rank_of_the_source"], "unit_rank": e["unit_rank_of_the_source"], "pool": e["pool"],
                    "recognised": e["recognised"], "source_in_reference_corpus": e["source_is_a_document_of_v2_reference_corpus"],
                    "years": [q["remake_year"], q["source_year"]], "industries": [q["remake_origin"], q["source_origin"]], "sentences": [q["remake_sentences"], q["source_sentences"]],
                    "statements": [{"on_page": s["on_page"], "readers": s["readers"], "sentence": s["sentence"]} for s in q["stated"]],
                    "flags": {k: v for k, v in FL[i].items() if v}, "read_by_hand": by_hand.get(F.key[i])})
    put(S, "the_twenty_largest_contributions", top, "the twenty entries with the largest absolute contributions, with pairs.json's years, industries, sentence counts and stating sentences, the flags computed "
        "from pairs.json and, for five of them, a note from reading the statements by hand", R + ".entries_table; data/second/remakes/pairs.json")

    def flag_block(mask, what_):
        mask = np.asarray(mask, bool)
        s_in, _ = F.summary(mask, False)
        s_out, _ = F.summary(~mask)
        return {"what": what_, "entries": int(mask.sum()), "pairs_with_more_than_1pct_open": [[F.key[i][0], F.key[i][1], float(F.c[i])] for i in np.flatnonzero(mask & (F.open > 0.01))],
                "their_T": s_in["T"], "their_share_of_T": s_in["T"] / Tobs, "their_open_pairs": s_in["effective_pairs_left_open"],
                "without_them": dict(s_out, noise_calibrated_p=noise_p(s_out["Z"], Zs))}
    fb = {"reboot_in_the_statement": flag_block([f["reboot"] for f in FL], "the stating sentence contains 'reboot' (Section 3 lists it among the loose words)"),
          "decided_by_a_third_reader": flag_block([f["third_reader"] for f in FL], "the two readers differed on a statement of the pair"),
          "two_steps_from_the_source": flag_block([bool(f["two_steps_from_the_source_through"]) for f in FL], "pairs.json also holds (remake, m) and (m, source) for some film m: the entry's source is the original two steps back"),
          "table_year_contradicts_the_statement": flag_block([f["table_year_of_the_source_contradicts_the_remakes_page"] for f in FL], "every year on the remake's own stating sentences differs from the table's year of the source by more than 2"),
          "stated_only_off_the_remakes_page": flag_block([f["stated_only_off_the_remakes_page"] for f in FL], "no stating sentence is on the remake's own page"),
          "any_of_reboot_third_reader_two_steps_or_year": flag_block([f["reboot"] or f["third_reader"] or bool(f["two_steps_from_the_source_through"]) or f["table_year_of_the_source_contradicts_the_remakes_page"] for f in FL],
                                                                     "any of the four flags above that bear on what the pair is")}
    put(S, "pairs_whose_statement_is_contestable", fb, "the primary entries by what is unusual in pairs.json about them: how many, which of them the words leave open, their share of T, and the test without them (their "
        "contributions set to zero; nothing refitted)", R + ".entries_table; data/second/remakes/pairs.json")
    short = np.array([min(P[k]["remake_sentences"], P[k]["source_sentences"]) <= 19 for k in F.key])
    s_, _ = F.summary(~short)
    s2, _ = F.summary(short)
    put(S, "by_shortest_text", {"both_texts_20_sentences_or_more": dict(s_, per_open_pair=s_["T"] / s_["effective_pairs_left_open"], noise_calibrated_p=noise_p(s_["Z"], Zs)),
                                "a_text_of_19_sentences_or_fewer": dict(s2, per_open_pair=s2["T"] / s2["effective_pairs_left_open"], noise_calibrated_p=noise_p(s2["Z"], Zs))},
        "the contributions summed over pairs whose shorter text has at least 20 sentences, and over the rest (nothing refitted)", R + ".entries_table; data/second/remakes/pairs.json: remake_sentences, source_sentences")

    # ================================================================================================== 2. the null
    S = "2_null"
    put(S, "registered", {"Z": Zobs, "T": Tobs, "sign_flip_p": A["slope"]["p"], "families": F.n_fam, "effective_pairs_left_open": A["slope"]["effective_pairs_left_open"]}, "the registered statistic and p", R + ".slope")
    put(S, "noise_calibrated_p", {"p": noise_p(Zobs, Zs), "noise_draws_at_or_above_the_observed_Z": int((Zs >= Zobs).sum()), "draws": int(len(Zs)), "largest_noise_Z": float(Zs.max()),
                                  "upper_95pct_bound_on_the_tail_probability_with_no_draw_above": float(1 - 0.05 ** (1 / len(Zs)))},
        "share of the 2,000 pure-noise draws whose Z is at or above the units' Z, the observed value counted as one; no draw reaches it, so 0.0005 is the smallest value this can take", "remakes_null_size.json: Z_sorted")
    put(S, "noise_normal_extrapolations", {"Z_over_the_noise_sd_of_Z": Zobs / NZ["sd_Z"], "p_normal_from_Z": float(stats.norm.sf((Zobs - NZ["mean_Z"]) / NZ["sd_Z"])),
                                           "T_over_the_noise_sd_of_T": Tobs / NZ["sd_T"], "p_normal_from_T": float(stats.norm.sf(Tobs / NZ["sd_T"])),
                                           "noise_sd_T": NZ["sd_T"], "noise_root_mean_sum_of_squares": NZ["root_mean_sum_of_squares_of_family_sums"], "observed_root_sum_of_squares": Tobs / Zobs,
                                           "variance_ratio_T_to_what_sign_flips_assume": (NZ["sd_T"] / NZ["root_mean_sum_of_squares_of_family_sums"]) ** 2},
        "the observed Z and T set against the noise draws' standard deviations (normal tail areas; the noise Z has sd 1.24 where sign flips assume 1)", "remakes_null_size.json: by_columns['39 columns (registered)']")
    # word-only rivals
    gens = words["primary_calibration"]["random_families_of_stand_ins"]["generators"]
    rz = np.concatenate([np.array(v["Z"], float) for v in gens.values()])
    sz = np.array([v["Z"] for v in words["primary_calibration"]["standins"]["rows"].values()], float)
    worst_gen = max(gens, key=lambda k: np.mean(gens[k]["Z"]) + 2 * np.std(gens[k]["Z"], ddof=1) if len(gens[k]["Z"]) >= 40 else -9)
    wz = np.array(gens[worst_gen]["Z"], float)
    put(S, "against_word_only_measures", {
        "stand_ins": {"n": int(len(sz)), "mean_Z": float(sz.mean()), "sd_Z": float(sz.std(ddof=1)), "largest_Z": float(sz.max())},
        "random_word_only_measures": {"n": int(len(rz)), "mean_Z": float(rz.mean()), "sd_Z": float(rz.std(ddof=1)), "largest_Z": float(rz.max()), "above_the_stand_ins_largest_Z": int((rz > STANDIN_Z).sum()),
                                      "share_above_the_stand_ins_largest_Z": float((rz > STANDIN_Z).mean())},
        "word_only_measures_at_or_above_the_units_Z": int((rz >= Zobs).sum() + (sz >= Zobs).sum()), "of": int(len(rz) + len(sz)),
        "empirical_p_against_all_of_them": float(1 / (len(rz) + len(sz) + 1)),
        "margin_of_the_units_Z_over_the_best_word_only_measure": float(Zobs - max(rz.max(), sz.max())),
        "p_normal_against_the_random_measures": float(stats.norm.sf((Zobs - rz.mean()) / rz.std(ddof=1))),
        "strongest_generator": worst_gen, "strongest_generator_n": int(len(wz)), "strongest_generator_mean_Z": float(wz.mean()), "strongest_generator_sd_Z": float(wz.std(ddof=1)),
        "strongest_generator_largest_Z": float(wz.max()), "p_normal_against_the_strongest_generator": float(stats.norm.sf((Zobs - wz.mean()) / wz.std(ddof=1)))},
        "the units' Z set against every word-only measure whose Z is committed: the 74 stand-ins and the 332 random word-only measures (none reaches it; the best has Z 3.12), with normal tail areas "
        "from the mean and sd of all random measures and of the generator with the heaviest upper tail", "remakes_words.json: primary_calibration.standins.rows; .random_families_of_stand_ins.generators[*].Z")
    cp = words["primary_calibration"]["noisy_and_curved_copies"]
    lows = sorted((vv["lowest_p"], f"{k} | {kk}", vv["draws"]) for k, v in cp["settings"].items() for kk, vv in v.items())
    put(S, "against_noisy_and_curved_copies", {"copies_in_these_settings": int(sum(x[2] for x in lows)), "sign_flips_per_copy": cp["sign_flips_per_draw"], "smallest_p_a_copy_can_get": 1 / (cp["sign_flips_per_draw"] + 1),
                                               "five_lowest_p_and_their_settings": [[x[0], x[1]] for x in lows[:5]], "settings_whose_lowest_p_is_at_the_floor": int(sum(x[0] <= 1.0001 / (cp["sign_flips_per_draw"] + 1) for x in lows)),
                                               "rate_below_0.05_all_copies": cp["all_copies_that_do_not_use_the_labels"]["rate_below_0.05"], "copies": cp["all_copies_that_do_not_use_the_labels"]["draws"],
                                               "Z_of_the_copies_committed": False},
        "the noisy and curved copies of the word columns: only rates and each setting's lowest p are committed (no Z), so the units' Z cannot be set against the copies' largest Z; one setting's lowest p is at the floor "
        "of its 5,000 flips (0.0002) and the next lowest is 0.0006", "remakes_words.json: primary_calibration.noisy_and_curved_copies")
    # the bootstrap over families
    n = F.n_fam
    idx = rng.integers(0, n, size=(BOOT, n))
    means = d_all[idx].mean(1)
    jack = np.array([np.delete(d_all, i).mean() for i in range(n)])
    put(S, "bootstrap_mean_family_sum", {"mean": float(d_all.mean()), "percentile_95": boot_ci(means), "bca_95": bca(d_all.mean(), means, jack), "bca_99": bca(d_all.mean(), means, jack, (0.005, 0.995)),
                                         "bca_99.9": bca(d_all.mean(), means, jack, (0.0005, 0.9995)),
                                         "share_of_resamples_at_or_below_zero": float((means <= 0).mean()), "resamples": BOOT, "families": n},
        "bootstrap over the 175 families of the mean family sum (T / 175): percentile and BCa intervals, and the share of resamples with a mean at or below zero", R + ".family_sums.rows")
    om = F.family_sum(F.open)
    ratio = d_all[idx].sum(1) / om[idx].sum(1)
    jack_r = np.array([np.delete(d_all, i).sum() / np.delete(om, i).sum() for i in range(n)])
    put(S, "bootstrap_effect_per_open_pair", {"T_over_open_pairs": float(Tobs / om.sum()), "percentile_95": boot_ci(ratio), "bca_95": bca(Tobs / om.sum(), ratio, jack_r)},
        "T divided by the effective open pairs (normal-score units by which the source's unit rank beats what the word model expects, per pair the words leave open), with a family bootstrap", R + ".entries_table")
    openf = om > 0.1
    idx2 = rng.integers(0, int(openf.sum()), size=(BOOT, int(openf.sum())))
    m2 = d_all[openf][idx2].mean(1)
    put(S, "bootstrap_mean_family_sum_open_families_only", {"families_with_more_than_a_tenth_of_a_pair_open": int(openf.sum()), "mean": float(d_all[openf].mean()), "median": float(np.median(d_all[openf])),
                                                             "percentile_95": boot_ci(m2), "share_of_resamples_at_or_below_zero": float((m2 <= 0).mean()),
                                                             "share_of_T_in_these_families": float(d_all[openf].sum() / Tobs)},
        "the same bootstrap over only the families with more than a tenth of a pair open", R + ".family_sums.rows")
    # sign and signed-rank tests
    def sign_block(mask):
        pos, neg = int((d_all[mask] > 0).sum()), int((d_all[mask] < 0).sum())
        out = {"families": int(mask.sum()), "positive": pos, "negative": neg,
               "sign_test_p_one_sided": float(stats.binomtest(pos, pos + neg, 0.5, alternative="greater").pvalue) if pos + neg else None}
        if pos + neg >= 6:
            w = stats.wilcoxon(d_all[mask & (d_all != 0)], alternative="greater")
            out["wilcoxon_signed_rank_p_one_sided"] = float(w.pvalue)
        return out
    signs = {"every non-zero sum (the file's 47 against 57)": sign_block(d_all != 0)}
    for thr in (1e-6, 1e-3, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
        signs[f"|D| above {thr:g}"] = sign_block(np.abs(d_all) > thr)
    for thr in (0.01, 0.1, 0.25, 0.5, 0.9):
        signs[f"open mass above {thr:g}"] = sign_block(om > thr)
    put(S, "sign_and_signed_rank_tests", signs, "positive and negative family sums, the one-sided sign test and Wilcoxon signed-rank test, by the size of the sum and by how much of a pair the family leaves open", R + ".family_sums.rows")
    nzd = d_all[d_all != 0]
    put(S, "distribution_of_the_family_sums", {
        "families": n, "exactly_zero": int((d_all == 0).sum()),
        "quantiles_of_the_non_zero_sums": {str(q): float(np.quantile(nzd, q)) for q in (0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1)},
        "sum_of_the_positive_sums": float(d_all[d_all > 0].sum()), "sum_of_the_negative_sums": float(d_all[d_all < 0].sum()),
        "largest_positive": float(d_all.max()), "largest_negative": float(d_all.min()),
        "positive_sums_above_0.5": int((d_all > 0.5).sum()), "negative_sums_below_minus_0.5": int((d_all < -0.5).sum()),
        "positive_sums_above_0.1": int((d_all > 0.1).sum()), "negative_sums_below_minus_0.1": int((d_all < -0.1).sum()),
        "median_absolute_negative_sum": float(np.median(-d_all[d_all < 0])), "median_positive_sum": float(np.median(d_all[d_all > 0])),
        "skewness": float(stats.skew(d_all))},
        "the shape of the 175 family sums: a few large positive sums, almost no negative sum of any size", R + ".family_sums.rows")
    big = np.sort(np.abs(d_all))[::-1]
    put(S, "what_the_sign_flip_p_amounts_to", {
        "sum_of_absolute_sums": float(np.abs(d_all).sum()), "T_as_a_share_of_the_largest_possible_T": float(Tobs / np.abs(d_all).sum()),
        "two_to_the_minus_k_for_the_k_sums_above_0.5": float(2.0 ** -int((np.abs(d_all) > 0.5).sum())), "k": int((np.abs(d_all) > 0.5).sum()),
        "exact_p_from_the_20_largest_sums_alone": sign_flip_p(d_all[np.argsort(-np.abs(d_all))[:20]]),
        "exact_p_from_the_16_largest_sums_alone": sign_flip_p(d_all[np.argsort(-np.abs(d_all))[:16]])},
        "T is 91% of the largest value any sign pattern could give, so the registered p is in effect the chance that the 16 or so sizeable sums nearly all come out positive", R + ".family_sums.rows")

    # ============================================================================================ 3. practical size
    S = "3_practical_size"
    wo = A["words_only"]["all_entries"]
    hc = A["heredity"]["hub_corrected"]
    g = A["gain"]
    N = len(E)
    w1, u1 = F.rank_words == 1, F.rank_units == 1
    put(S, "sources_ranked_first", {"entries": N, "by_the_word_model": int(w1.sum()), "by_the_units_alone": int(u1.sum()), "by_the_word_model_with_the_unit_column": g["sources_first_with_the_measure"],
                                    "by_both": int((w1 & u1).sum()), "by_words_not_units": int((w1 & ~u1).sum()), "by_units_not_words": int((~w1 & u1).sum()), "by_neither": int((~w1 & ~u1).sum()),
                                    "not_copied_entries": int((~F.copied).sum()), "not_copied_by_the_word_model": int((w1 & ~F.copied).sum()), "not_copied_by_the_units_alone": int((u1 & ~F.copied).sum())},
        "how often the true source is ranked first among its candidates: by the held-out word model, by the hub-corrected unit rate alone, and by the word model refitted with the unit column",
        R + ".entries_table; .gain.sources_first_with_the_measure; .heredity.hub_corrected")
    wrong = np.flatnonzero(~w1)
    put(S, "pairs_the_word_model_gets_wrong", {"n": int(len(wrong)), "units_rank_the_source_first": int(u1[wrong].sum()), "units_rank_it_above_the_words_rank": int((F.rank_units[wrong] < F.rank_words[wrong]).sum()),
                                               "units_rank_it_below_the_words_rank": int((F.rank_units[wrong] > F.rank_words[wrong]).sum()), "units_rank_in_the_top_10": int((F.rank_units[wrong] <= 10).sum()),
                                               "median_unit_rank": float(np.median(F.rank_units[wrong])), "median_words_rank": float(np.median(F.rank_words[wrong])),
                                               "rows": [{"remake": E[i]["remake"], "source": E[i]["source"], "pool": int(F.pool[i]), "words_rank": float(F.rank_words[i]), "unit_rank": int(F.rank_units[i]),
                                                         "words_probability_of_the_source": float(F.p_words[i]), "contribution": float(F.c[i]), "recognised": bool(F.rec[i])} for i in wrong[np.argsort(F.rank_words[wrong])]]},
        "the 15 pairs whose source the word model does not rank first: where the units alone rank it", R + ".entries_table")
    put(S, "T_by_whether_the_word_model_ranks_the_source_first", {"pairs_the_word_model_gets_wrong": {"n": int(len(wrong)), "T": float(F.c[wrong].sum()), "share_of_T": float(F.c[wrong].sum() / Tobs),
                                                                                                     "contributions_positive": int((F.c[wrong] > 0).sum()), "open_pairs": float(F.open[wrong].sum()),
                                                                                                     "two_to_the_minus_n": float(2.0 ** -len(wrong))},
                                                                  "pairs_the_word_model_gets_right": {"n": int(w1.sum()), "T": float(F.c[w1].sum()), "share_of_T": float(F.c[w1].sum() / Tobs),
                                                                                                     "open_pairs": float(F.open[w1].sum()), "with_more_than_1pct_open": int((w1 & (F.open > 0.01)).sum()),
                                                                                                     "of_those_contribution_positive": int((w1 & (F.open > 0.01) & (F.c > 0)).sum()),
                                                                                                     "Z_of_these_alone": F.summary(w1, False)[0]["Z"], "sign_flip_p_of_these_alone": F.summary(w1)[0]["p"]}},
        "T split by whether the word model ranks the true source first: the 15 pairs it gets wrong hold most of T and every one of their contributions is positive; the registered p is close to the chance of 15 "
        "signs out of 15", R + ".entries_table")
    right = np.flatnonzero(w1)
    put(S, "pairs_the_word_model_gets_right", {"n": int(len(right)), "units_rank_the_source_first": int(u1[right].sum()), "units_rank_in_the_top_10": int((F.rank_units[right] <= 10).sum()),
                                               "units_rank_beyond_10": int((F.rank_units[right] > 10).sum()), "units_rank_beyond_100": int((F.rank_units[right] > 100).sum()),
                                               "not_copied": int((~F.copied[right]).sum()), "not_copied_units_first": int((u1[right] & ~F.copied[right]).sum())},
        "the 205 pairs whose source the word model ranks first: where the units alone rank it", R + ".entries_table")
    nc = ~F.copied
    put(S, "mean_reciprocal_rank", {"words_all": float((1 / F.rank_words).mean()), "units_all": float((1 / F.rank_units).mean()), "words_not_copied": float((1 / F.rank_words[nc]).mean()),
                                    "units_not_copied": float((1 / F.rank_units[nc]).mean()), "words_on_the_15_it_gets_wrong": float((1 / F.rank_words[wrong]).mean()),
                                    "units_on_the_15_words_get_wrong": float((1 / F.rank_units[wrong]).mean()),
                                    "better_of_the_two_ranks_all": float((1 / np.minimum(F.rank_words, F.rank_units)).mean())},
        "mean reciprocal rank of the true source by the word model and by the units alone (the last line takes, pair by pair, the better of the two ranks: an upper bound no rule could reach without knowing the answer)", R + ".entries_table")
    put(S, "the_refitted_model", {"gain_in_held_out_log_likelihood": g["gain"], "gain_sign_flip_p": g["gain_sign_flip_p"], "words_only_held_out_log_likelihood": wo["held_out_log_likelihood"],
                                  "with_the_units": wo["held_out_log_likelihood"] + g["gain"], "geometric_mean_probability_of_the_source_words": math.exp(wo["held_out_log_likelihood"] / N),
                                  "geometric_mean_probability_of_the_source_with_units": math.exp((wo["held_out_log_likelihood"] + g["gain"]) / N),
                                  "share_of_the_words_only_shortfall_recovered": g["gain"] / -wo["held_out_log_likelihood"], "gain_per_open_pair_nats": g["gain"] / A["slope"]["effective_pairs_left_open"],
                                  "sources_first_words": g["sources_first_words"], "sources_first_with_the_measure": g["sources_first_with_the_measure"],
                                  "top_1_answers_gained_net": g["sources_first_with_the_measure"] - g["sources_first_words"], "log_rank_gain": g["log_rank_gain"],
                                  "geometric_mean_rank_ratio_over_all_entries": math.exp(g["log_rank_gain"] / N), "geometric_mean_rank_ratio_if_all_of_it_fell_on_the_15_wrong_pairs": math.exp(g["log_rank_gain"] / len(wrong)),
                                  "unit_weight_mean": g["unit_weight_mean"], "odds_ratio_per_pool_sd_of_the_unit_score": math.exp(g["unit_weight_mean"])},
        "what the committed gain block says about the word model refitted with the unit column: 19.6 nats of held-out log-likelihood, one more source ranked first (205 to 206)", R + ".gain; .words_only.all_entries")
    op = F.p_words < 0.5
    put(S, "pairs_the_words_leave_more_than_half_open", {"n": int(op.sum()), "units_rank_first": int(u1[op].sum()), "units_in_top_10": int((F.rank_units[op] <= 10).sum()), "median_unit_rank": float(np.median(F.rank_units[op])),
                                                        "median_pool": float(np.median(F.pool[op])), "median_unit_rank_as_a_share_of_the_pool": float(np.median(F.rank_units[op] / F.pool[op])),
                                                        "contribution_positive": int((F.c[op] > 0).sum()), "T_share": float(F.c[op].sum() / Tobs)},
        "the 20 entries where the word model gives the true source a probability under 0.5: where the units alone rank the source", R + ".entries_table")
    put(S, "effect_in_the_test_s_own_unit", {"T_per_open_pair": float(Tobs / F.open.sum()), "smallest_planted_boost_with_80pct_power_word_twin": res["decision"]["smallest_boost_detected_with_80pct_power_on_this_list"]["word twin plus noise"],
                                            "smallest_planted_boost_with_80pct_power_pure_noise": res["decision"]["smallest_boost_detected_with_80pct_power_on_this_list"]["pure noise"]},
        "T per open pair (about 1.2 normal-score units) beside the smallest planted boost the test finds with 80% power, in pool standard deviations", R + ".slope; decision.smallest_boost_detected_with_80pct_power_on_this_list")
    emb = A["embedding"]["in_the_units_place"]
    put(S, "units_alone_against_words_alone_heredity", {"units_share_first_not_copied": hc["not_copied"]["share_source_first"], "units_first_not_copied": hc["not_copied"]["source_strictly_above_every_candidate"],
                                                       "words_first_not_copied": A["words_only"]["not_copied"]["sources_ranked_first"], "of": hc["not_copied"]["entries"],
                                                       "embedding_Z_in_the_units_place": emb["slope"]["Z"], "embedding_gain": emb["gain"]["gain"]},
        "on the 120 pairs that are not copied the units alone rank the source first in 76, the word model in 105", R + ".heredity.hub_corrected.not_copied; .words_only.not_copied; .embedding")

    # ============================================================================================== 4. recognition
    S = "4_recognition"
    def per_open(keep):
        s_, d_ = F.summary(keep)
        s_["per_open_pair"] = s_["T"] / s_["effective_pairs_left_open"] if s_["effective_pairs_left_open"] > 0 else None
        s_["noise_calibrated_p_whole_list_null"] = noise_p(s_["Z"], Zs)
        s_["sources_first_by_words"] = int((w1 & keep).sum())
        s_["sources_first_by_units"] = int((u1 & keep).sum())
        nzf = d_[np.abs(d_) > 0.1]
        s_["families_with_abs_sum_above_0.1"] = [int((nzf > 0).sum()), int((nzf < 0).sum())]
        return s_, d_
    splits = {}
    for name, keep in (("recognised", F.rec), ("not_recognised", ~F.rec), ("recognised_both_films", F.rec_both), ("not_recognised_both_films", ~F.rec_both),
                       ("recognised_and_not_copied", F.rec & ~F.copied), ("not_recognised_and_not_copied", ~F.rec & ~F.copied)):
        splits[name] = per_open(keep)[0]
    put(S, "splits", splits, "every summary by recognition (the wide rule used by condition 7, and the narrower one): T, Z, sign-flip p, open pairs, T per open pair, sources first by words and by units; the noise-calibrated p "
        "uses the whole list's noise draws and is only a guide for a subset", R + ".entries_table")
    d_n = F.family_sum(F.c, ~F.rec)
    o_n = np.argsort(-d_n, kind="stable")
    lad = []
    for k in range(0, 13):
        x = np.delete(d_n, o_n[:k])
        lad.append({"k": k, "family_removed": None if k == 0 else F.fam_names[o_n[k - 1]], "its_remakes": None if k == 0 else F.fam_remakes[o_n[k - 1]], "its_sum": None if k == 0 else float(d_n[o_n[k - 1]]),
                    "T_n": float(x.sum()), "Z_n": zstat(x), "p_n": sign_flip_p(x)})
    put(S, "not_recognised_largest_families_removed", lad, "condition 7's statistic after the k largest family sums of the unrecognised entries are removed", R + ".entries_table")
    put(S, "not_recognised_families_to_remove_for_p_at_or_above_0.05", next(r["k"] for r in lad if r["p_n"] >= 0.05), "smallest k for which condition 7's p is 0.05 or more", "computed from the ladder")
    put(S, "not_recognised_concentration", {"families_with_a_sum_above_0.1_in_absolute_value": int((np.abs(d_n) > 0.1).sum()), "effective_number_of_families": float(np.abs(d_n).sum() ** 2 / (d_n ** 2).sum()),
                                           "share_of_T_n_in_the_three_largest": float(np.sort(d_n)[::-1][:3].sum() / d_n.sum()), "share_of_T_n_in_the_five_largest": float(np.sort(d_n)[::-1][:5].sum() / d_n.sum())},
        "how concentrated the unrecognised entries' sum is", R + ".entries_table")
    # the difference between recognised and not, per open pair, family bootstrap
    cr, mr = F.family_sum(F.c, F.rec), F.family_sum(F.open, F.rec)
    cn, mn = F.family_sum(F.c, ~F.rec), F.family_sum(F.open, ~F.rec)
    diff = cr[idx].sum(1) / mr[idx].sum(1) - cn[idx].sum(1) / mn[idx].sum(1)
    put(S, "recognised_minus_not_per_open_pair", {"recognised": float(cr.sum() / mr.sum()), "not_recognised": float(cn.sum() / mn.sum()), "difference": float(cr.sum() / mr.sum() - cn.sum() / mn.sum()),
                                                 "percentile_95": boot_ci(diff), "share_of_resamples_with_difference_at_or_below_zero": float((diff <= 0).mean())},
        "T per open pair among recognised entries minus the same among unrecognised entries, with a family bootstrap", R + ".entries_table")
    # replies of the model for the largest contributions: did the rule miss a title?
    rows = []
    for i in eorder[:30]:
        e = E[i]
        a, b, c2 = recog["r1"].get(e["remake"], {}), recog["r1"].get(e["source"], {}), recog["r2"].get(e["remake"], {})
        rows.append({"remake": e["remake"], "source": e["source"], "contribution": e["contribution"], "recognised": e["recognised"],
                     "R1_reply_for_the_remake": a.get("reply"), "remake_identified": a.get("identified"), "R1_reply_for_the_source": b.get("reply"), "source_identified": b.get("identified"),
                     "R2_reply": c2.get("reply"), "source_named": c2.get("source_named")})
    put(S, "replies_for_the_thirty_largest_contributions", rows, "the model's replies to R1 (both films) and R2 for the thirty entries with the largest absolute contributions, with the rule's verdicts", f"remakes_recognition_{PRIMARY_MODEL}.json: r1, r2")
    # a looser scan of every unrecognised entry: does any of the three replies name any film of the pair's family (the pair's films, the other pairs' films, the relatives)?
    fam_titles = {}
    for q in pairs["pairs"]:
        fam_titles.setdefault(q["family"], set()).update([q["remake"], q["source"]])
    for x in pairs["relatives_that_are_not_in_a_kept_pair"]:
        fam_titles.setdefault(x["family"], set()).add(x["film"])
    loose_hits = []
    for i, e in enumerate(E):
        if e["recognised"]:
            continue
        reps = (("R1 for the remake", recog["r1"][e["remake"]]["reply"]), ("R1 for the source", recog["r1"][e["source"]]["reply"]), ("R2", recog["r2"][e["remake"]]["reply"]))
        found = [[nm, rep, title, how] for title in sorted(fam_titles[e["family"]]) for nm, rep in reps for how in [names_title(rep, title)] if how]
        if found:
            loose_hits.append({"remake": e["remake"], "source": e["source"], "contribution": e["contribution"], "open_mass": float(F.open[i]), "replies_that_name_a_film_of_the_family": found})
    put(S, "looser_title_scan_of_the_unrecognised_entries", {"unrecognised_entries": int((~F.rec).sum()), "entries_with_a_reply_naming_a_film_of_the_family": len(loose_hits), "rows": loose_hits,
                                                            "their_open_pairs": float(sum(h["open_mass"] for h in loose_hits)), "their_T": float(sum(h["contribution"] for h in loose_hits))},
        "every unrecognised primary entry scanned with a looser rule than the registered one (any of the three replies names, exactly or within 80% similarity, any film of the pair's family): how many such "
        "entries there are and how much of the test's weight they carry", f"remakes_recognition_{PRIMARY_MODEL}.json: r1, r2; data/second/remakes/pairs.json: pairs, relatives_that_are_not_in_a_kept_pair")
    # near misses read by hand in those replies (see the report): the reply names another remake of the same source, or most of the remake's title
    near = [("Ilavelpu", "Edhir Paradhathu", "R1 reply for the remake is \"Sharada\": the Hindi remake of the same source is Sharada (1957), named in the source's own stating sentence"),
            ("Aaj Ka Goonda Raj", "Gang Leader", "R1 reply for the remake is 'Gundaraj': two of the title's four words run together, but also the title of another film")]
    near = [x for x in near if (x[0], x[1]) in F.key]
    moved = F.rec.copy()
    for r_, s_, _ in near:
        moved[F.key.index((r_, s_))] = True
    for h in loose_hits:
        moved[F.key.index((h["remake"], h["source"]))] = True
    s_moved, _ = F.summary(~moved)
    put(S, "possible_misses_of_the_title_rule", {"pairs": [[r_, s_, why, float(F.c[F.key.index((r_, s_))])] for r_, s_, why in near],
                                                "condition_7_with_them_counted_as_recognised": {"T_n": s_moved["T"], "Z_n": s_moved["Z"], "p_n": s_moved["p"], "E_n": s_moved["effective_pairs_left_open"], "entries": s_moved["entries"]}},
        "two replies among the thirty that the rule may have missed (read by hand), and condition 7 recomputed with both pairs, and the pairs of the looser scan, moved to the recognised side", "computed; the replies are in replies_for_the_thirty_largest_contributions")
    # how many of the largest unrecognised families would have to be recognised after all for condition 7 to fail
    put(S, "recognised_by_title_summary", {"films_identified": recog["summary"]["films_identified"], "of": recog["summary"]["of"], "remakes_whose_source_is_named": recog["summary"]["remakes_whose_source_is_named"],
                                          "of_remakes": recog["summary"]["of_remakes"], "primary_pairs_recognised": recog["summary"]["primary_pairs_recognised"],
                                          "primary_pairs_recognised_both_films": recog["summary"]["primary_pairs_recognised_both_films"]},
        "the recognition tallies as committed", f"remakes_recognition_{PRIMARY_MODEL}.json: summary")

    # ====================================================================================== 5. the reference corpus
    S = "5_reference_corpus"
    cf, mf = F.family_sum(F.c, F.ref), F.family_sum(F.open, F.ref)
    cx, mx = F.family_sum(F.c, ~F.ref), F.family_sum(F.open, ~F.ref)
    diff = cf[idx].sum(1) / mf[idx].sum(1) - cx[idx].sum(1) / mx[idx].sum(1)
    jack_d = np.array([np.delete(cf, i).sum() / np.delete(mf, i).sum() - np.delete(cx, i).sum() / np.delete(mx, i).sum() for i in range(n)])
    theta = cf.sum() / mf.sum() - cx.sum() / mx.sum()
    sr, sx = per_open(F.ref)[0], per_open(~F.ref)[0]
    put(S, "splits", {"source_in_the_reference_corpus": sr, "source_not_in_the_reference_corpus": sx}, "the contributions summed by whether the source's summary is a document of v2's reference corpus", R + ".entries_table")
    put(S, "per_open_pair_difference", {"in_reference": float(cf.sum() / mf.sum()), "not_in_reference": float(cx.sum() / mx.sum()), "difference": float(theta), "percentile_95": boot_ci(diff),
                                        "bca_95": bca(theta, diff, jack_d), "share_of_resamples_with_difference_at_or_below_zero": float((diff <= 0).mean()),
                                        "bootstrap_95_for_in_reference": boot_ci(cf[idx].sum(1) / mf[idx].sum(1)), "bootstrap_95_for_not_in_reference": boot_ci(cx[idx].sum(1) / mx[idx].sum(1))},
        "T per open pair where the source is a reference document minus where it is not, with a family bootstrap", R + ".entries_table")
    put(S, "shares", {"share_of_entries": float(F.ref.mean()), "share_of_open_pairs": float(F.open[F.ref].sum() / F.open.sum()), "share_of_T": float(F.c[F.ref].sum() / Tobs),
                      "of_the_15_largest_contributions": int(F.ref[eorder[:15]].sum()), "of_the_10_largest_contributions": int(F.ref[eorder[:10]].sum()), "of_the_7_largest_contributions": int(F.ref[eorder[:7]].sum()),
                      "chance_of_7_of_7_at_the_open_pair_share": float((F.open[F.ref].sum() / F.open.sum()) ** 7)},
        "the reference-source pairs' share of the entries, of the open pairs, of T and of the largest contributions", R + ".entries_table")
    sb = words["primary_calibration"]["standins_by_subset_of_the_entries_descriptive"]
    put(S, "against_the_stand_ins_on_the_same_subsets", {
        "in_reference": {"units_Z": sr["Z"], "stand_ins_largest_Z_on_this_subset": sb["source_is_a_document_of_v2_reference_corpus"]["z_max"], "stand_ins_share_below_0.05": sb["source_is_a_document_of_v2_reference_corpus"]["share_below_0.05"],
                         "units_above_every_stand_in": bool(sr["Z"] > sb["source_is_a_document_of_v2_reference_corpus"]["z_max"])},
        "not_in_reference": {"units_Z": sx["Z"], "stand_ins_largest_Z_on_this_subset": sb["source_is_not_a_document_of_v2_reference_corpus"]["z_max"], "stand_ins_share_below_0.05": sb["source_is_not_a_document_of_v2_reference_corpus"]["share_below_0.05"],
                             "units_above_every_stand_in": bool(sx["Z"] > sb["source_is_not_a_document_of_v2_reference_corpus"]["z_max"])}},
        "the units' Z on each subset beside the largest Z of the 74 stand-ins on the same subset (the words file has the stand-ins by subset): where the source is not a reference document the units are below the best stand-in",
        R + ".subsets; remakes_words.json: primary_calibration.standins_by_subset_of_the_entries_descriptive")
    cross = {}
    for a_name, a_keep in (("in_reference", F.ref), ("not_in_reference", ~F.ref)):
        for b_name, b_keep in (("recognised", F.rec), ("not_recognised", ~F.rec)):
            s_, _ = F.summary(a_keep & b_keep)
            cross[f"{a_name} and {b_name}"] = dict(s_, per_open_pair=s_["T"] / s_["effective_pairs_left_open"] if s_["effective_pairs_left_open"] > 0 else None)
    put(S, "reference_by_recognition", cross, "the four cells of reference source by recognition", R + ".entries_table")
    put(S, "units_alone_by_reference", {"not_copied_in_reference": [int((u1 & F.ref & nc).sum()), int((F.ref & nc).sum())], "not_copied_not_in_reference": [int((u1 & ~F.ref & nc).sum()), int((~F.ref & nc).sum())],
                                       "words_not_copied_in_reference": [int((w1 & F.ref & nc).sum()), int((F.ref & nc).sum())], "words_not_copied_not_in_reference": [int((w1 & ~F.ref & nc).sum()), int((~F.ref & nc).sum())]},
        "sources ranked first by the units alone and by the word model, among pairs that are not copied, by whether the source is a reference document [first, of]", R + ".entries_table")

    # =========================================================================================== 6. settings, frames
    S = "6_settings_and_frames"
    pl = {}
    for k, v in A["planes"]["rows"].items():
        pl[k] = {"T": v["T"], "Z": v["Z"], "sign_flip_p": v["p"], "noise_calibrated_p": noise_p(v["Z"], Zs), "Z_above_the_stand_ins": bool(v["Z"] > STANDIN_Z),
                 "families_positive": v["families_positive"], "families_negative": v["families_negative"]}
    put(S, "rule_settings", pl, "the eight rule settings: T, Z, the registered p, the noise-calibrated p (the noise draws do not depend on the setting) and whether Z is above the stand-ins' largest Z", R + ".planes.rows")
    put(S, "rule_settings_counts", {"sign_flip_p_below_0.05": int(sum(v["sign_flip_p"] < 0.05 for v in pl.values())), "noise_calibrated_p_below_0.05": int(sum(v["noise_calibrated_p"] < 0.05 for v in pl.values())),
                                    "Z_above_the_stand_ins": int(sum(v["Z_above_the_stand_ins"] for v in pl.values())), "of": len(pl),
                                    "the_primary_setting_has_the_largest_Z": bool(max(pl, key=lambda k: pl[k]["Z"]) == "top1_30")},
        "how many of the eight settings pass each reading", "computed from rule_settings")
    others = {}
    base = {f: x for f, x in zip(F.fam_names, d_all)}
    for name, X in res["analyses"].items():
        rows_ = {r["family"]: r["D"] for r in X["family_sums"]["rows"]}
        st_ = X["standins"]
        row = {"entries": X["entries"], "families": X["families"], "T": X["slope"]["T"], "Z": X["slope"]["Z"], "sign_flip_p": X["slope"]["p"], "open_pairs": X["slope"]["effective_pairs_left_open"],
               "gain": X["gain"]["gain"], "sources_first_words": X["gain"]["sources_first_words"], "sources_first_with_the_measure": X["gain"]["sources_first_with_the_measure"],
               "stand_ins_largest_Z": st_["z_max"], "stand_ins_share_below_0.05": st_["share_below_0.05"], "units_Z_above_every_stand_in": bool(X["slope"]["Z"] > st_["z_max"]),
               "calibrated_by_the_stand_ins": bool(st_["share_below_0.05"] <= 0.10), "effective_number_of_families": X["family_sums"]["effective_number_of_families"],
               "share_of_sum_of_squares_in_five_largest": X["family_sums"]["share_of_the_sum_of_squares_in_the_five_largest"],
               "not_recognised_Z": X["recall_rule"]["Z_n"], "not_recognised_p": X["recall_rule"]["p_n"]}
        shared = [f for f in rows_ if f in base]
        if name != KEY and len(shared) >= 20:
            a_, b_ = np.array([base[f] for f in shared]), np.array([rows_[f] for f in shared])
            t10a = set(sorted(shared, key=lambda f: -base[f])[:10])
            t10b = set(sorted(shared, key=lambda f: -rows_[f])[:10])
            row["family_sums_shared_with_the_primary_analysis"] = {"families": len(shared), "pearson": float(np.corrcoef(a_, b_)[0, 1]), "spearman": float(stats.spearmanr(a_, b_)[0]), "ten_largest_in_common": len(t10a & t10b)}
        others[name] = row
    put(S, "frames_and_sets", others, "each committed analysis of this model (the two other frames of the primary set, and the secondary sets): the test's numbers, whether the units' Z is above every stand-in, and how its family "
        "sums agree with the primary analysis on the families they share", f"remakes_{PRIMARY_MODEL}.json: analyses[*]")
    bs = A["beside_the_decision"]
    side = {"pruned_cells": {"Z": bs["slope_with_pruned_cells"]["Z"], "T": bs["slope_with_pruned_cells"]["T"]}}
    for k in ("no_hub_correction", "lambda_1", "lambda_10"):
        side[k] = {"Z": bs[k]["slope"]["Z"], "T": bs[k]["slope"]["T"], "p": bs[k]["slope"]["p"], "gain": bs[k]["gain"]["gain"], "stand_ins_largest_Z": bs[k]["standins"]["z_max"],
                   "stand_ins_share_below_0.05": bs[k]["standins"]["share_below_0.05"], "units_above_every_stand_in": bool(bs[k]["slope"]["Z"] > bs[k]["standins"]["z_max"])}
    sc = A["scale_robustness"]
    side["without_units_above_their_scale"] = {"Z": sc["slope"]["Z"], "T": sc["slope"]["T"], "p": sc["slope"]["p"], "gain": sc["gain"]["gain"], "sources_first_with_the_measure": sc["gain"]["sources_first_with_the_measure"]}
    e2 = A["embedding"]["units_with_the_embedding_among_the_word_columns"]
    side["embedding_in_the_units_place"] = {"Z": emb["slope"]["Z"], "p": emb["slope"]["p"], "gain": emb["gain"]["gain"]}
    side["units_with_the_embedding_among_the_columns"] = {"Z": e2["slope"]["Z"], "p": e2["slope"]["p"], "gain": e2["gain"]["gain"], "stand_ins_largest_Z": e2["standins"]["z_max"]}
    put(S, "beside_the_decision", side, "the registered sensitivity settings as committed", R + ".beside_the_decision; .scale_robustness; .embedding")

    # =============================================================================================== 7. across models
    S = "7_across_models"
    fits = {PRIMARY_MODEL: F}
    for path in model_files():
        tag = os.path.basename(path)[len("remakes_"):-len(".json")]
        if tag == PRIMARY_MODEL:
            continue
        try:
            with open(path) as f:
                r2 = json.load(f)
            if KEY in r2.get("analyses", {}) and "entries_table" in r2["analyses"][KEY]:
                fits[tag] = Fit(r2)
        except (ValueError, KeyError) as err:                      # a file still being written
            put(S, f"unreadable__{tag}", str(err), "this model's file could not be read when the audit ran", os.path.basename(path))
    put(S, "models_present", sorted(fits), "the models whose remakes file existed and could be read when this audit ran", "results/v2/second/remakes_<tag>.json")
    per = {}
    for tag, G in fits.items():
        X = G.A
        o_ = np.argsort(-G.D, kind="stable")
        zk = [zstat(np.delete(G.D, o_[:k])) for k in range(0, 21)]
        per[tag] = {"T": X["slope"]["T"], "Z": X["slope"]["Z"], "sign_flip_p": X["slope"]["p"], "noise_calibrated_p": noise_p(X["slope"]["Z"], Zs), "gain": X["gain"]["gain"],
                    "sources_first_words": X["gain"]["sources_first_words"], "sources_first_with_the_measure": X["gain"]["sources_first_with_the_measure"], "sources_first_units_alone": int((G.rank_units == 1).sum()),
                    "families_positive": X["slope"]["families_positive"], "families_negative": X["slope"]["families_negative"], "effective_number_of_families": X["family_sums"]["effective_number_of_families"],
                    "planes_below_0.05": X.get("planes", {}).get("below_0.05"), "planes_available": len(X.get("planes", {}).get("available", [])),
                    "not_recognised": {k: X["recall_rule"][k] for k in ("entries_not_recognised", "E_n", "T_n", "Z_n", "p_n")},
                    "entries_recognised": int(G.rec.sum()), "outcome_conditions_1_to_5": X.get("conditions_1_to_5_hold"),
                    "Z_after_removing_its_k_largest_families": {str(k): zk[k] for k in (1, 3, 5, 10)},
                    "families_to_remove_for_Z_under_the_stand_ins": next((k for k, z in enumerate(zk) if z < STANDIN_Z), None),
                    "ten_largest_families": [[G.fam_names[i], G.fam_remakes[i], float(G.D[i])] for i in o_[:10]],
                    "T_in_reference_source_pairs": float(G.c[G.ref].sum()), "T_not_in_reference": float(G.c[~G.ref].sum()),
                    "per_open_pair_in_reference": float(G.c[G.ref].sum() / G.open[G.ref].sum()), "per_open_pair_not_in_reference": float(G.c[~G.ref].sum() / G.open[~G.ref].sum()),
                    "not_in_reference_Z": G.summary(~G.ref, False)[0]["Z"], "not_in_reference_p": X["subsets"].get("source_is_not_a_document_of_v2_reference_corpus", {}).get("p"),
                    "per_open_pair_recognised": float(G.c[G.rec].sum() / G.open[G.rec].sum()), "per_open_pair_not_recognised": float(G.c[~G.rec].sum() / G.open[~G.rec].sum()),
                    "planes": {k: {"Z": v["Z"], "p": v["p"], "noise_calibrated_p": noise_p(v["Z"], Zs), "Z_above_the_stand_ins": bool(v["Z"] > STANDIN_Z)} for k, v in X.get("planes", {}).get("rows", {}).items()},
                    "units_rank_first_on_the_15_pairs_the_words_get_wrong": int((G.rank_units[G.rank_words > 1] == 1).sum()),
                    "units_rank_below_the_words_rank_on_those_15": int((G.rank_units[G.rank_words > 1] > G.rank_words[G.rank_words > 1]).sum()),
                    "embedding_Z_in_the_units_place": X.get("embedding", {}).get("in_the_units_place", {}).get("slope", {}).get("Z"),
                    "arithmetic": {"T_gap": float(G.c.sum() - X["slope"]["T"]), "Z_gap": float(zstat(G.D) - X["slope"]["Z"]), "family_sums_gap": float(np.abs(G.D - G.D_file).max()),
                                   "not_recognised_T_gap": float(G.c[~G.rec].sum() - X["recall_rule"]["T_n"]),
                                   "sign_flip_p_recomputed": sign_flip_p(G.D), "not_recognised_p_recomputed": sign_flip_p(G.family_sum(G.c, ~G.rec))}}
    put(S, "each_model", per, "the primary analysis of each model present: the test's numbers and its ten largest families", "remakes_<tag>.json: analyses['primary | set']")
    tags = sorted(fits)
    pairsm = {}
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            Fa, Fb = fits[a], fits[b]
            assert Fa.fam_names == Fb.fam_names and Fa.key == Fb.key, "the models' entries differ"
            da, db = Fa.D, Fb.D
            om_ = Fa.family_sum(Fa.open) > 0.1
            ta, tb = set(np.argsort(-da)[:10].tolist()), set(np.argsort(-db)[:10].tolist())
            ta5, tb5 = set(np.argsort(-da)[:5].tolist()), set(np.argsort(-db)[:5].tolist())
            both_big = om_
            pairsm[f"{a} | {b}"] = {
                "pearson_all_175_families": float(np.corrcoef(da, db)[0, 1]), "spearman_all_175_families": float(stats.spearmanr(da, db)[0]),
                "families_with_more_than_a_tenth_of_a_pair_open": int(om_.sum()), "pearson_open_families": float(np.corrcoef(da[om_], db[om_])[0, 1]), "spearman_open_families": float(stats.spearmanr(da[om_], db[om_])[0]),
                "pearson_of_the_entries_contributions": float(np.corrcoef(Fa.c, Fb.c)[0, 1]),
                "ten_largest_families_in_common": len(ta & tb), "five_largest_families_in_common": len(ta5 & tb5),
                "same_sign_among_open_families": int((np.sign(da[both_big]) == np.sign(db[both_big])).sum()),
                f"Z_of_{b}_without_the_ten_largest_families_of_{a}": zstat(np.delete(db, list(ta))), f"Z_of_{a}_without_the_ten_largest_families_of_{b}": zstat(np.delete(da, list(tb))),
                f"share_of_T_of_{b}_in_the_ten_largest_families_of_{a}": float(db[list(ta)].sum() / db.sum()), f"share_of_T_of_{a}_in_the_ten_largest_families_of_{b}": float(da[list(tb)].sum() / da.sum()),
                "pearson_of_unit_ranks_log": float(np.corrcoef(np.log(Fa.rank_units), np.log(Fb.rank_units))[0, 1]),
                "pearson_of_family_sums_per_open_pair_open_families": float(np.corrcoef(da[om_] / Fa.family_sum(Fa.open)[om_], db[om_] / Fa.family_sum(Fa.open)[om_])[0, 1]),
                "pearson_of_family_sum_with_open_mass": {a: float(np.corrcoef(da, Fa.family_sum(Fa.open))[0, 1]), b: float(np.corrcoef(db, Fa.family_sum(Fa.open))[0, 1])},
                "families_whose_sums_differ_in_sign": int((np.sign(da) != np.sign(db)).sum()),
                "entries_recognised_by_both": int((Fa.rec & Fb.rec).sum()), "entries_recognised_by_neither": int((~Fa.rec & ~Fb.rec).sum()),
                "word_model_identical": bool(np.allclose(Fa.p_words, Fb.p_words))}
    put(S, "between_models", pairsm, "for each pair of models: the correlation of their family sums, the overlap of their largest families, and each model's Z without the other's ten largest families",
        "remakes_<tag>.json: analyses['primary | set'].family_sums.rows, .entries_table")
    if len(fits) > 1:
        nobody = np.all([~G.rec for G in fits.values()], axis=0)
        put(S, "entries_no_model_present_recognises", {"entries": int(nobody.sum()), "open_pairs": float(F.open[nobody].sum()),
                                                       "by_model": {tag: {k: v for k, v in G.summary(nobody)[0].items() if k in ("T", "Z", "p", "families_positive", "families_negative")} for tag, G in fits.items()}},
            "the contributions summed over the entries that none of the models present recognises (nothing refitted)", "remakes_<tag>.json: analyses['primary | set'].entries_table")
        mean_d = np.mean([G.D for G in fits.values()], axis=0)
        put(S, "family_sums_averaged_over_models", {"Z": zstat(mean_d), "T": float(mean_d.sum()), "sign_flip_p": sign_flip_p(mean_d), "effective_number_of_families": float(np.abs(mean_d).sum() ** 2 / (mean_d ** 2).sum()),
                                                    "families_to_remove_for_Z_under_the_stand_ins": next((k for k in range(0, 40) if zstat(np.delete(mean_d, np.argsort(-mean_d)[:k])) < STANDIN_Z), None)},
            "the family sums averaged over the models present, through the same Z and sign flips: the models share the families, so this is one sample of families, not several", "computed from the models' family sums")

    # ====================================================================================== 8b. notes for the refits
    S = "9_descriptive_for_the_refit_auditors"
    opn = F.open > 0.01
    eff = F.c[opn] / F.open[opn]
    yg = np.array([P[k]["remake_year"] - P[k]["source_year"] for k in F.key])
    rs = np.array([P[k]["remake_sentences"] for k in F.key])
    ss = np.array([P[k]["source_sentences"] for k in F.key])
    def sp(x):
        r_, p_ = stats.spearmanr(x[opn], F.c[opn])
        return {"spearman_with_the_contribution": float(r_), "p_two_sided": float(p_)}
    put(S, "open_entries_contribution_against_what_pairs_json_knows", {
        "entries_with_more_than_1pct_open": int(opn.sum()), "year_gap": sp(yg), "remake_sentences": sp(rs), "source_sentences": sp(ss), "shorter_text_sentences": sp(np.minimum(rs, ss)),
        "absolute_log_ratio_of_sentences": sp(np.abs(np.log(rs / ss))), "pool_size": sp(F.pool), "open_mass": sp(F.open), "log_words_rank": sp(np.log(F.rank_words)), "log_unit_rank": sp(np.log(F.rank_units))},
        "rank correlations, over the entries the words leave more than 1% open, between the contribution and the year gap, the texts' lengths and the pool size (descriptive: nothing refitted, no adjustment)",
        R + ".entries_table; data/second/remakes/pairs.json")
    by_ind = {}
    for i, k in enumerate(F.key):
        q = P[k]
        for label in (f"source {q['source_origin']}", f"remake {q['remake_origin']}"):
            b = by_ind.setdefault(label, {"entries": 0, "open_pairs": 0.0, "T": 0.0})
            b["entries"] += 1
            b["open_pairs"] += float(F.open[i])
            b["T"] += float(F.c[i])
    for b in by_ind.values():
        b["per_open_pair"] = b["T"] / b["open_pairs"] if b["open_pairs"] > 0.5 else None
    put(S, "by_industry", dict(sorted(by_ind.items(), key=lambda kv: -kv[1]["T"])), "entries, open pairs and T by the source's industry and by the remake's industry (nothing refitted)", R + ".entries_table; pairs.json: remake_origin, source_origin")
    dec = {}
    for i, k in enumerate(F.key):
        lab = "source year before 1990" if P[k]["source_year"] < 1990 else "source year 1990 to 2004" if P[k]["source_year"] < 2005 else "source year 2005 on"
        b = dec.setdefault(lab, {"entries": 0, "open_pairs": 0.0, "T": 0.0})
        b["entries"] += 1
        b["open_pairs"] += float(F.open[i])
        b["T"] += float(F.c[i])
    for b in dec.values():
        b["per_open_pair"] = b["T"] / b["open_pairs"]
    put(S, "by_source_year", dec, "entries, open pairs and T by the table's year of the source", R + ".entries_table; pairs.json: source_year")
    put("0_inputs", "seconds", round(time.time() - T0, 1), "seconds this script took", "computed")

    with open(os.path.join(HERE, "statistics.json"), "w") as f:
        json.dump(plain({"what": "An audit of the remakes decision test (statistics and the size of the effect) from committed outputs only; made by results/v2/second/audit/statistics.py. "
                                 "Each number has 'what' (one line) and 'source' (the committed file and key path it comes from, or the inputs it was computed from).",
                         "sections": OUT}), f, indent=1)
        f.write("\n")
    print(f"wrote {os.path.join(HERE, 'statistics.json')} in {time.time() - T0:.1f} s")


def model_files():
    skip = ("remakes_words.json", "remakes_null_size.json")
    return sorted(p for p in glob.glob(os.path.join(RES, "remakes_*.json"))
                  if os.path.basename(p) not in skip and "recognition" not in os.path.basename(p) and "verify" not in os.path.basename(p))


if __name__ == "__main__":
    sys.exit(main())
