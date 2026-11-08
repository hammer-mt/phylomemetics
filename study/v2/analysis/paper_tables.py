"""The paper's tables, written from the committed results so that no number is copied by hand.

    cd study && ../.venv/bin/python v2/analysis/paper_tables.py

Reads results/v2/*.json and writes results/v2/paper_tables.md. A table is left out while its result file is missing.
"""
import json

import common as C

NAMES = {"qwen3-32b": "Qwen3-32B", "qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama 3.1 8B", "llama3.3-70b": "Llama 3.3 70B"}
ORDER = ["qwen3-32b", "qwen3-8b", "llama3.1-8b", "llama3.3-70b"]


def load(name):
    p = C.RESULTS / name
    return json.load(open(p)) if p.exists() else None


def f(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def pv(p):
    if p is None:
        return "n/a"
    return f"{p:.2g}" if p < 0.001 else (f"{p:.4f}" if p < 0.01 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}")


def n(x):
    return "n/a" if x is None else f"{int(x):,}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    ex = load("experiments_v2.json")
    models = [m for m in ORDER if m in ex]
    parts = []

    # ---- units and presence
    passes = load("passes.json") or {}
    rows = []
    for m in models:
        e, p = ex[m], passes.get(m, {})
        sc = p.get("scale", {})
        rows.append([NAMES[m], n(e["n_units_total"]), n(e["n_units_present_in_some_work"]), n(e["n_distinctive_units_present_in_some_work"]),
                     n(e["units_present_per_batman_work_original_median"]), n(e["units_present_per_batman_work_name_free_median"]),
                     n(sc.get("units_whose_scale_comes_from_the_supplement")), n(sc.get("units_whose_scale_comes_from_the_study_corpus")),
                     n(sc.get("positive_units_whose_strongest_raw_activation_is_an_echo"))])
    parts.append("## Units and presence\n\n" + table(["Model", "Units", "Present in some work", "of those, distinctive", "Per Batman work, original (median)",
                                                      "Per Batman work, name-free (median)", "Scale set by the supplement", "Scale set by the study texts",
                                                      "Positive units whose strongest raw activation is an echo"], rows))

    # ---- E1
    def e1row(label, b):
        a, o = b["A_name_free"], b.get("openings", {})
        return [label, f(a["hero_film_cousins"]["auc"]), f"{f(a['hero_film_cousins']['ci95'][0])} to {f(a['hero_film_cousins']['ci95'][1])}",
                f(a.get("null_label_shuffle", {}).get("p95")), f(o.get("hero_film_cousins", {}).get("auc")), f(o.get("null_label_shuffle", {}).get("p95")),
                f(b.get("A_iconography_dropped", {}).get("hero_film_cousins", {}).get("auc")), f(a.get("ordinary_films", {}).get("auc")),
                f(a.get("literary_cousins", {}).get("auc")), b.get("verdict", "")]
    rows = []
    for m in models:
        rows.append(e1row(f"{NAMES[m]}: canon share", ex[m]["E1"]["canon_share"]))
        rows.append(e1row(f"{NAMES[m]}: centroid of units", ex[m]["E1"]["unit_centroid"]))
        rows.append(e1row(f"{NAMES[m]}: centroid, distinctive units", ex[m]["E1"]["unit_centroid_distinctive_units"]))
    w = ex["words"]["E1"]
    rows.append(e1row("Words: TF-IDF centroid", w["tfidf_centroid"]))
    rows.append(e1row("Words: TF-IDF centroid, background weights", w["tfidf_centroid_background_weights"]))
    rows.append(e1row("Length alone (sentences)", w["length_alone_sentences"]))
    parts.append("## E1, identity: name-free Batman works against the 8 hero films\n\n" +
                 table(["Score", "AUC", "95% interval", "Shuffle p95", "Openings AUC", "Openings shuffle p95", "AUC without iconography", "vs ordinary films",
                        "vs literary relatives", "Verdict"], rows))
    rows = []
    for m in models:
        c = ex[m]["E1"]["core"]
        cs = ex[m]["E1"]["canon_share"]["A_name_free"]
        rows.append([NAMES[m], n(c["n_canon_units"]), n(cs["canon_units_median"]), n(c["n_core_units"]), n(c["n_canon_units_without_cap"]), n(c["n_core_units_without_cap"]),
                     f(c["random_film_sets_core_units_without_cap_mean"], 1)])
    parts.append("## Canon and name-free core\n\n" + table(["Model", "Canon units (all 32 works)", "Canon units per fold (median)", "Name-free core",
                                                            "Canon without the cap", "Name-free core without the cap", "The same for random sets of 32 ordinary films"], rows))

    # ---- E2 and the residual test
    rows = []
    for m in models:
        e2 = ex[m]["E2"]
        s, r = e2["summary"], e2["residual_summary"]
        rows.append([NAMES[m], n(e2["evaluable"]["n_optional_units"]), f(s["share"], 3), pv(s["exact_p"]), pv(s["within_era_p"]), f(s["chance_mean"], 3),
                     f"{e2['sequels_name_free']['top2']} of {e2['sequels_name_free']['n']}", pv(r["decision_words_by_frequency"]["exact_p"]),
                     pv(r["every_word_and_code_measure"]["exact_p"]), pv(r["registered_both_word_measures"]["exact_p"]),
                     f(r["decision_words_by_frequency"]["share"], 3)])
    for label, key in (("Words: TF-IDF cosine", "tfidf_cosine"), ("Words: word presence", "word_presence")):
        e2 = ex["words"]["E2"][key]
        ch = e2["primary"]["chance"]
        rows.append([label, "", f(ch["observed_unrounded"], 3), pv(ch["joint_relabelling_exact"]["p"]), pv(ch["within_era"]["p"]), f(ch["joint_relabelling_exact"]["mean"], 3),
                     f"{e2['sequels_name_free']['top2']} of {e2['sequels_name_free']['n']}", "", "", "", ""])
    parts.append("## E2, heredity, and the residual test\n\n" +
                 table(["Measure", "Optional units", "Share of comics beating the source", "Exact p", "Within-era p", "Chance", "Sequels in top two",
                        "Residual: decision test p", "Residual: every measure p", "Residual: v1's two-measure form p", "Residual share (decision)"], rows))

    # ---- E3
    rows = [[NAMES[m], f(ex[m]["E3"]["primary"]["rho"]), pv(ex[m]["E3"]["primary"]["p_one_sided"]), n(ex[m]["E3"]["primary"]["n_units"]),
             f(ex[m]["E3"]["original_text"]["rho"]), f(ex[m]["E3"]["background_base_rate_placebo"]["rho"])] for m in models]
    parts.append("## E3, selection\n\n" + table(["Model", "rho", "One-sided p", "Units", "rho on original texts", "Placebo (base rate) rho"], rows))

    # ---- contrasts
    con = load("contrasts.json")
    if con:
        rows = [[k.replace(" | ", ": "), f(c["E1_auc"], 3), f(c.get("E1_openings_auc"), 3), f(c["E2_share"], 3), pv(c["E2_exact_p"]), pv(c["residual_exact_p"]), pv(c.get("residual_decision_exact_p")),
                 c["sequels_top2"], f(c["E3_rho"]), pv(c["E3_p"])] for k, c in con["cells"].items()]
        parts.append("## What changed a result: every model under both rules\n\n" +
                     table(["Cell", "E1 AUC", "E1 openings AUC", "E2 share", "E2 exact p", "Residual p (two measures)", "Residual p (decision)", "Sequels", "E3 rho", "E3 p"], rows))
        rows = [[c["contrast"], c["a"].split(" | ")[0], f(c["E1_auc"]["difference"], 3), f"{f(c['E1_auc']['ci95'][0], 3)} to {f(c['E1_auc']['ci95'][1], 3)}",
                 f(c["E2_share"]["difference"], 3), f"{f(c['E2_share']['ci95'][0], 3)} to {f(c['E2_share']['ci95'][1], 3)}"] for c in con["contrasts"]]
        parts.append("### Differences between cells\n\n" + table(["Contrast", "Model", "E1 AUC difference", "95% interval", "E2 share difference", "95% interval"], rows))
        if "counts_over_new_cells" in con:
            rows = [[k, v["cells"], v["below_0.05"], v["expected_if_independent"]] for k, v in con["counts_over_new_cells"].items()]
            parts.append("### Cells below 0.05\n\n" + table(["Test", "Cells", "Below 0.05", "0.05 x cells"], rows))
    cj = load("cells_joint.json")
    if cj:
        rows = [[k, len(cj["cells"]), v["cells_below_0.05"], pv(v["probability_of_at_least_that_many_under_one_null_for_all_cells"])] for k, v in cj["tests"].items()]
        parts.append("### The same counts under one null for all cells\n\n" + table(["Test", "Cells", "Below 0.05", "Probability of at least that many"], rows))

    # ---- sensitivity
    rows = []
    for m in models:
        s = load(f"sensitivity_{m}.json")
        if s:
            rows.append([NAMES[m], f"{s['residual_settings_exact_p_below_0.05']} of {s['n_decision_settings']}", s["n_decision_settings_evaluable"],
                         f"{s['E2_uncapped_settings_exact_p_below_0.05']} of {s['n_decision_settings']}", f"{s['E2_settings_exact_p_below_0.05']} of {s['E2_settings_evaluable']}",
                         f"{f(s['E1_auc_range'][0])} to {f(s['E1_auc_range'][1])}", s["E1_settings_auc_at_least_0.80"],
                         f"{f(s['E3_rho_range'][0])} to {f(s['E3_rho_range'][1])}", s["E3_settings_p_below_0.05"]])
    if rows:
        parts.append("## Sensitivity grid (threshold x score x cap; 56 settings)\n\n" +
                     table(["Model", "Residual decision test below 0.05, uncapped settings", "of which evaluable", "E2 below 0.05, uncapped settings",
                            "E2 below 0.05, evaluable settings", "E1 AUC range", "Settings with E1 AUC of 0.80 or more", "E3 rho range", "E3 settings below 0.05"], rows))

    # ---- canon counts and franchises
    rows = []
    for m in models:
        c = load(f"canon_counts_{m}.json")
        if c:
            d, a, nf = c["batman_distinctive_units"], c["batman_all_units"], c["batman_name_free_distinctive_units"]
            r = c["random_sets_of_32_ordinary_films_all_units"]
            t = c["tiers_units"]
            rows.append([NAMES[m], n(d["at_least_6"]), n(d["at_least_23"]), n(d["at_least_29"]), n(nf["at_least_6"]), n(nf["at_least_23"]), n(nf["at_least_29"]),
                         n(a["at_least_29"]), f(r["at_least_29"]["mean"], 0), f"{t['core']['n']} / {t['common']['n']} / {t['optional']['n']}"])
    if rows:
        parts.append("## Canon counts (distinctive units present in at least k of 32 Batman works)\n\n" +
                     table(["Model", "6+", "23+", "29+", "Name-free 6+", "Name-free 23+", "Name-free 29+", "All units 29+ (no cap)", "Random film sets, all units 29+",
                            "Candidates by tier (core / common / optional)"], rows))
    for m in models:
        fr = load(f"franchises_{m}.json")
        if fr:
            rows = [[k, v["n_films"], v["at_least_72pct"], f(v["batman_subsets_of_the_same_size"]["at_least_72pct"]["mean"], 0), v["at_least_90pct"],
                     f(v["batman_subsets_of_the_same_size"]["at_least_90pct"]["mean"], 0)] for k, v in fr["franchises"].items()]
            rows.append(["Batman (all 32)", 32, fr["batman"]["at_least_72pct"], "", fr["batman"]["at_least_90pct"], ""])
            parts.append(f"## Other film series, {NAMES[m]} (distinctive units in at least 72% and 90% of the films)\n\n" +
                         table(["Series", "Films", "72%+", "Batman subsets, 72%+", "90%+", "Batman subsets, 90%+"], rows))
            g = fr.get("genre_sets", {})
            rows = [[k, v.get("n_plots_of_this_genre"), f(v["at_least_6_films"]["mean"], 0), f(v["at_least_23"]["mean"], 1), v["at_least_23"]["max"], f(v["at_least_29"]["mean"], 1)]
                    for k, v in g.items() if "at_least_23" in v]
            if rows:
                parts.append(f"### Genre sets of 32 plots, {NAMES[m]}\n\n" + table(["Genre", "Plots", "Units in 6+ (mean)", "23+ (mean)", "23+ (max)", "29+ (mean)"], rows))

    # ---- recognition
    rows = []
    for m in models:
        r = load(f"recognition_{m}.json")
        if r:
            lp = r["logp_batman_median"]
            rows.append([NAMES[m], f"{r['batman_works_named_batman']} of 32", f"{r['batman_works_identified']} of 32", f"{r['hero_films_naming_their_own_hero']} of {r['hero_films_with_a_hero_to_name']}",
                         f"{r['hero_films_called_batman']} of 8", f"{r['ordinary_films_identified']} of 69", f(lp["batman"]), f(lp["hero_films"]), f(lp["ordinary_films"]),
                         f(r["auc_logp_batman_batman_vs_hero_films"])])
    if rows:
        parts.append("## Recognition: what the untouched model says about a name-free summary\n\n" +
                     table(["Model", "Batman works it calls Batman", "Batman works it identifies by title", "Hero films whose hero it names", "Hero films it calls Batman",
                            "Ordinary films it identifies by title", "Median log p of Batman: Batman works", "hero films", "ordinary films", "AUC of that log p, Batman vs hero films"], rows))

    # ---- explanation tests
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.0f}%"
    rows, kind_rows = [], []
    for m in models:
        t = load(f"{m}/explanation_tests.json")
        if not t:
            continue
        for label, key in ((NAMES[m], "all"), (f"{NAMES[m]}: idea units", "idea_units"), (f"{NAMES[m]}: name or word units", "name_units")):
            a = t[key]
            if not a.get("n"):
                continue
            sr = a["strong_rate"]
            rows.append([label, a["n_with_untestable"], a["not_testable_out_of_context"], pct(sr["A"]), pct(sr["B"]), pct(sr["C"]), f"{100 * sr['D']:.1f}%",
                         f"{a['won']} / {a['tied']} / {a['lost']}", f"{a['idea_bound']} ({pct(a['idea_bound_share'])})", pct(a["blind_max_reaches_corpus_max"]),
                         f(a["auc_B_vs_C_mean"]["neuron"]), f(a["auc_B_vs_C_mean"]["analyst_keywords"]), f(a["auc_B_vs_C_mean"]["tfidf_to_examples"])])
        mm = t["mismatched_null"]
        rows.append([f"{NAMES[m]}: other units' texts (floor)", "", "", pct(mm["strong_rate_on_other_units_A_texts"]), pct(mm["strong_rate_on_other_units_B_texts"]), "", "",
                     f"win rate {pct(mm['win_rate_on_other_units_texts'])}", f"{100 * mm['idea_bound_rate_on_other_units_texts']:.1f}%", "", "", "", ""])
        for k, a in t["by_kind"].items():
            if a.get("n"):
                kind_rows.append([NAMES[m], k, a["n"], pct(a["strong_rate"]["A"]), pct(a["strong_rate"]["B"]), pct(a["strong_rate"]["C"]),
                                  f"{a['won']} / {a['tied']} / {a['lost']}", a["idea_bound"]])
        c = t["core_representatives_name_place_or_word"]
        kind_rows.append([NAMES[m], "core canon representatives judged a name, place or word", c["n"], "", "", "", f"{c['count']} of {c['n']}",
                          f"95% interval {f(c['ci95'][0])} to {f(c['ci95'][1])}"])
    if rows:
        eb = (load(f"{models[0]}/explanation_tests.json") or load("qwen3-8b/explanation_tests.json"))["essay_benchmark"]
        rows.append(["The essay's gallery (benchmark)", "", "", pct(eb["new_examples"]), "", pct(eb["near_misses"]), f"{100 * eb['distant']:.0f}%",
                     f"{eb['won']} / {eb['tied']} / {eb['lost']}", "", "", "", "", ""])
        parts.append("## Explanation tests\n\n" + table(["Units", "Explained", "Not testable out of context", "New examples strong (A)", "Idea without the key words (B)",
                                                          "Near misses (C)", "Distant (D)", "Won / tied / lost", "Idea-bound", "Blind maximum reaches the corpus maximum",
                                                          "AUC B vs C: unit", "keyword terms", "TF-IDF to examples"], rows))
        parts.append("### By kind (blind kind judge)\n\n" + table(["Model", "Kind", "Units", "A", "B", "C", "Won / tied / lost", "Idea-bound"], kind_rows))
    rows = []
    for m in models:
        t = load(f"{m}/explanation_tests.json")
        if t:
            for k, v in t["neurons_against_words_B_vs_C"].items():
                rows.append([NAMES[m], "all units", k, v["n"], f(v.get("neuron_mean")), f(v.get("words_mean")), v.get("neuron_better"), v.get("words_better"), pv(v.get("p_two_sided"))])
            for k, v in t["neurons_against_words_B_vs_C_idea_units"].items():
                rows.append([NAMES[m], "idea units", k, v["n"], f(v.get("neuron_mean")), f(v.get("words_mean")), v.get("neuron_better"), v.get("words_better"), pv(v.get("p_two_sided"))])
    if rows:
        parts.append("### Units against word measures on the same texts (AUC for B against C)\n\n" +
                     table(["Model", "Units", "Word measure", "n", "Unit mean", "Words mean", "Unit better", "Words better", "Wilcoxon p"], rows))

    # ---- checks that need no judge, coverage, second judge, size test
    rows = []
    for m in models:
        pc = load(f"{m}/plain_checks.json")
        if pc:
            for label, a in [("all", pc["all"])] + [(k, v) for k, v in pc["by_detects"].items()]:
                rows.append([NAMES[m], label, a["n"], pct(a["same_stem_share_of_content_peaks_mean"]), pct(a["share_of_peaks_on_punctuation_or_function_words"]),
                             f(a["best_word_r_median"]), pct(a["held_out_hit_explainer_key_words"]), pct(a["random_window_hit_explainer_key_words"]),
                             pct(a["held_out_hit_analyst_terms"]), pct(a["random_window_hit_analyst_terms"])])
    if rows:
        parts.append("## Checks of the explanations that need no judge\n\n" +
                     table(["Model", "Units", "n", "Peaks sharing the commonest stem", "Peaks on punctuation or function words", "Best single word r (median)",
                            "Held-out examples with an explainer key word", "Random windows", "Held-out examples with an analyst term", "Random windows"], rows))
    rows = []
    for m in models:
        cv = load(f"coverage_{m}.json")
        if cv:
            for label, a in [("all", cv["all"]), ("idea units", cv["idea_units"]), ("name or word units", cv["name_units"])] + list(cv["by_share_band"].items()):
                rows.append([NAMES[m], label, a["units"], a["firing_passages"], pct(a["covered"]), a["silent_passages"], pct(a["wrongly_claimed"])])
    if rows:
        parts.append("## Coverage: does the description predict firings at 30 to 60% of scale?\n\n" +
                     table(["Model", "Units or band", "Units", "Firing passages", "Covered by the description", "Silent passages", "Wrongly claimed"], rows))
    rows = []
    for m in models:
        sj = load(f"second_judge_{m}.json")
        if sj:
            for k in ("kind", "detects", "steer"):
                if sj.get(k):
                    rows.append([NAMES[m], k, sj[k]["n"], pct(sj[k]["agree"]), f(sj[k].get("kappa"))])
    if rows:
        parts.append("## Qwen3-32B as a second judge: agreement with the Claude judge\n\n" + table(["Units of", "Judgment", "Items", "Agreement", "Kappa"], rows))
    sz = load("size_test.json")
    if sz:
        rows = [[k, v["n_big"], v["n_small"], pct(v["win_rate_big"]), pct(v["win_rate_small"]), f(v["difference"], 3), pv(v["p_one_sided"])]
                for k, v in sz.items() if isinstance(v, dict) and "p_one_sided" in v]
        parts.append(f"## Bigger is cleaner? Win rate, {NAMES[sz['big']]} against {NAMES[sz['small']]}\n\n" +
                     table(["Comparison", "Units (big)", "Units (small)", "Win rate (big)", "Win rate (small)", "Difference", "One-sided p"], rows))
        rows = [[k, f(sz[k]["big"], 3), f(sz[k]["small"], 3), f(sz[k]["difference"], 3), f"{f(sz[k]['ci95'][0], 3)} to {f(sz[k]['ci95'][1], 3)}"]
                for k in ("new_example_strong_rate", "near_miss_strong_rate", "idea_bound_share", "one_idea_score")]
        parts.append("### Other differences (big minus small)\n\n" + table(["Measure", "Big", "Small", "Difference", "95% interval"], rows))

    # ---- steering
    sg = load("steering_gallery.json")
    if sg:
        rows = [[f"{k} x scale", v["n"], v["correct"], pct(v["accuracy"]), v["none"]] for k, v in sg["by_multiple"].items()]
        parts.append(f"## Steering calibration on 40 gallery neurons (chance {pct(sg['chance'])}; chosen multiple {sg['chosen_multiple']}, p = {pv(sg['p_above_chance_at_chosen'])})\n\n" +
                     table(["Multiple", "Items", "Correct", "Accuracy", "Answered none"], rows))
    rows = []
    for m in models:
        st_ = load(f"steering_{m}.json")
        if st_:
            for label, a in [("idea units (criterion)", st_["idea_units"]), ("name or word units", st_["name_units"]), ("all units", st_["all_units"]),
                             ("idea units, four distractors of their own kind", st_["idea_units_with_four_distractors_of_their_own_kind"])] + \
                            [(f"depth quarter {q}", a) for q, a in st_["by_depth_quarter"].items()] + [(f"kind: {k}", a) for k, a in st_["by_kind"].items()]:
                rows.append([NAMES[m], label, a["n"], a["correct"], pct(a["accuracy"]), a["none"], pv(a["p_above_chance"])])
            sh = st_["sham_items"]
            rows.append([NAMES[m], "sham items (own title removed): answered none", sh["n"], sh["answered_none"], pct(sh["answered_none"] / max(sh["n"], 1)), "", ""])
            kw = st_["key_words_in_continuations"]["idea_units"]
            rows.append([NAMES[m], "idea units: continuations with a key word (pushed / not pushed / other units pushed)", "", "", f"{pct(kw['own_forced'])} / {pct(kw['unforced'])} / {pct(kw['other_units_forced'])}", "", ""])
    if rows:
        parts.append("## Steering the explained units (chance 20%)\n\n" + table(["Model", "Units", "Items", "Correct", "Accuracy", "Answered none", "Binomial p"], rows))

    # ---- ablation
    rows, follow, begun_rows = [], False, []
    for m in models:
        ab = load(f"ablation_{m}.json")
        if not ab:
            continue
        for label, key in (("summary tokens, name-free (confirmatory)", "story_tokens_name_free"), ("every position", "every_position_name_free"),
                           ("original texts", "story_tokens_original"), ("strong firings only", "strong_firings_only"), ("weak activity only", "weak_activity_only"),
                           ("follow-up test: reply begun for the model", "reply_begun"), ("follow-up, strong firings only", "reply_begun_strong_firings_only"),
                           ("follow-up, weak activity only", "reply_begun_weak_activity_only")):
            a = ab.get(key)
            if not a:
                continue
            r, g, sp = a["matched_random_sets"], a["magnitude_matched_random_sets"], a.get("specificity", {})
            # only the registered version and the follow-up's main run are tests (PLAN.md Section 18, changes 13 and 33): the others give the canon's rank among their 20 sets and no p
            test = key in ("story_tokens_name_free", "reply_begun")
            follow = follow or key.startswith("reply_begun")
            rank = lambda q: f"{q['random_sets_at_least_canon'] + 1} of {q['n_sets'] + 1}"
            rows.append([NAMES[m], label, r["n_sets"], f(r["mean_drop_canon"]), f(r["mean_drop_random_sets_mean"], 3), f(r["mean_drop_random_sets_max"], 3),
                         pv(r["p"]) if test else "", rank(r), f(g["mean_drop_random_sets_max"], 3), pv(g["p"]) if test else "", rank(g),
                         f(sp.get("mean_drop_hero_films_own_hero")), pv(sp.get("p_one_sided")) if test else "",
                         f(r["activation_removed"]["canon"], 0), f(g["activation_removed"]["sets_mean"], 0),
                         f"{a['untouched_reply_says_batman']} -> {a['canon_ablated_reply_still_says_batman']}"])
        if "criterion" in ab:
            c = ab["criterion"]
            n_bad = ab["story_tokens_name_free"]["untouched_replies_that_do_not_start_with_a_name"]
            rows.append([NAMES[m], "criterion", "", "", "", "", pv(c["p_against_sets_matched_on_works"]), "", "", pv(c["p_against_sets_also_matched_on_activity"]), "", "",
                         pv(c["specificity_p"]), "", "", f"untouched replies that do not start with a name: {n_bad} of 32 (limit 4); evaluable {c['evaluable_with_this_question']}; met {c['met']}"])
        if "criterion_reply_begun" in ab:
            c, a = ab["criterion_reply_begun"], ab["reply_begun"]
            rows.append([NAMES[m], f"criterion of the follow-up test (reply begun with '{c['reply_prefix']}')", "", "", "", "", pv(c["p_against_sets_matched_on_works"]), "", "",
                         pv(c["p_against_sets_also_matched_on_activity"]), "", "", f"{pv(c['specificity_p'])} (exact {pv(c['specificity_p_exact'])})", "", "",
                         f"untouched replies that do not begin with a name: {c['untouched_replies_that_do_not_start_with_a_name']} of 32 (limit 4); outcome: {c['outcome']}"])
            off, h, ag = a["replies_with_the_canon_off"], a["specificity"]["hero_films_described"], a["agreement_with_the_registered_run"]
            sub = ag["works_whose_registered_untouched_reply_began_with_a_name"]
            split = a.get("works_whose_untouched_reply_begins_with_a_name")
            lp, lpg = a["registered_form_drop_in_log_probability"]["matched_random_sets"], a["registered_form_drop_in_log_probability"]["magnitude_matched_random_sets"]
            begun_rows.append([NAMES[m], c["reply_prefix"],
                               f"{f(lp['mean_drop_canon'])} against {f(lp['mean_drop_random_sets_mean'], 3)} (largest set {f(lp['mean_drop_random_sets_max'], 3)}); p {pv(lp['p'])} and {pv(lpg['p'])}",
                               f"{f(a['mean_drop_where_the_untouched_model_names_batman'])}, {f(a['mean_drop_where_it_does_not'])}", f"{off['do_not_begin_with_a_name']} of {a['n_batman_works']} ({off['of_which_the_untouched_reply_did']} of them began with a name untouched)",
                               f(off["mean_drop_canon_with_those_works_left_out"]),
                               f"{split['n_works']}: {f(split['matched_random_sets']['mean_drop_canon'])} against {f(split['matched_random_sets']['mean_drop_random_sets_mean'], 3)}"
                               if split else "every reply begins with a name" if c["untouched_replies_that_do_not_start_with_a_name"] == 0 else "none begins with a name",
                               f"{h['untouched_replies_that_begin_with_a_name']} of {a['n_hero_films']}", f"{h['untouched_replies_that_name_their_own_hero']} of {a['n_hero_films']}",
                               f(h["mean_drop_over_the_films_whose_own_hero_the_untouched_model_names"]), f(h["mean_drop_over_all_the_films"]),
                               f"{f(ag['untouched_log_odds']['rho'])}, {f(ag['drop_canon']['rho'])}",
                               f"{f(sub['untouched_log_odds']['rho'])}, {f(sub['drop_canon']['rho'])} (n = {sub['drop_canon']['n']})"])
    if rows:
        note = ("The first version for each model is the registered test. The version with the reply begun for the model is a follow-up test, fixed after the registered "
                "result was seen and before it was run. It is not one of the six confirmatory tests. Its falls are not on the same scale as the registered run's, "
                "because the name's first token is a different one; each run's canon is compared with its own random sets. "
                if follow else "Only the first version for each model is a test. ")
        parts.append("## Ablation: switching off Batman's canon (drop in the log-odds of the hero's name)\n\n" + note +
                     "The others use 20 sets and are reported as the canon's drop beside the sets' and its rank among them.\n\n" +
                     table(["Model", "Version", "Random sets", "Canon, Batman works", "Sets (mean)", "Sets (max)", "p", "Canon's rank", "Activity-matched sets (max)", "p", "Canon's rank",
                            "Canon, hero films (own hero)", "Specificity p", "Activation removed: canon", "activity-matched sets (mean)", "Replies naming Batman: untouched -> canon off"], rows))

    if begun_rows:
        parts.append("## The follow-up ablation described (PLAN.md Section 18, change 33)\n\nDescriptions fixed with the follow-up test. They change no outcome.\n\n" +
                     table(["Model", "Reply begun with", "The same test in log-probability, the form registered before change 16: canon against sets; p against the two controls",
                            "Canon's fall in log-odds where the untouched reply says Batman, and where it does not",
                            "Replies with the canon off that do not begin with a name", "Canon's fall with those works left out",
                            "Works whose untouched reply begins with a name: canon against sets", "Hero films: untouched replies that begin with a name", "that name the film's own hero",
                            "Fall over the films whose hero the model names", "Fall over all seven", "Rank correlation with the registered run: untouched log-odds, canon's fall",
                            "The same over works whose registered reply began with a name"], begun_rows))
    rows = []
    for m in models:
        fc, pr = load(f"{m}/begun_form_check.json"), load(f"recognition_begun_{m}.json")
        for c in (fc or {}).get("checks", []):
            rows.append([NAMES[m], f"check of the prefix on the 69 ordinary films: '{c['prefix']}'", f"{c['replies_that_begin_with_a_name']} of {c['n_films']} replies begin with a name (62 needed)",
                         "passes" if c["passes"] else "fails", "", ""])
        if pr:
            cnt, sb, bn, reg = pr["n"], pr["replies_that_say_batman"], pr["replies_that_begin_with_a_name"], pr["registered_probe_without_the_prefix"]
            rows.append([NAMES[m], f"untouched probe with the reply begun ('{pr['reply_prefix']}')",
                         "replies that begin with a name: " + ", ".join(f"{bn[k]} of {cnt[k]} {k.replace('_', ' ')}" for k in cnt),
                         "replies that say Batman: " + ", ".join(f"{sb[k]} of {cnt[k]} {k.replace('_', ' ')}" for k in cnt),
                         f"{pr['hero_films_naming_their_own_hero']} of {pr['hero_films_with_a_hero_to_name']}", f(pr["auc_logp_batman_batman_vs_hero_films"])])
            rows.append([NAMES[m], "the registered probe, without the prefix", "", "replies that say Batman: " + ", ".join(f"{reg['replies_that_say_batman'][k]} of {cnt[k]} {k.replace('_', ' ')}" for k in cnt),
                         f"{reg['hero_films_naming_their_own_hero']} of {reg['hero_films_with_a_hero_to_name']}", f(reg["auc_logp_batman_batman_vs_hero_films"])])
    if rows:
        parts.append("## The follow-up ablation: the check of the prefix and the untouched probe with the reply begun\n\n" +
                     table(["Model", "What", "Form of the replies", "Result", "Hero films naming their own hero", "AUC of log p(Batman), Batman works against hero films"], rows))

    # ---- twins
    rows = []
    for o in ORDER:
        tw = load(f"twins_{ORDER[0]}_{o}.json")
        if not tw:
            continue
        for label, key in (("all units", "all"), ("idea units", "idea_units"), ("name or word units", "name_units")):
            nf, tt = tw["name_free_sentences"][key], tw["testers_texts"][key]
            rows.append([NAMES[o], label, nf["n"], f(nf["explained"]["r_original_median"]), f(nf["explained"]["r_name_free_median"]), f(nf["lexical"]["r_name_free_median"]),
                         f(nf["depth"]["r_name_free_median"]), f"{f(nf['twin_minus_lexical_controls']['mean'], 3)} ({f(nf['twin_minus_lexical_controls']['ci95'][0], 3)} to {f(nf['twin_minus_lexical_controls']['ci95'][1], 3)})",
                         tt["n"], f"{tt['won']} / {tt['tied']} / {tt['lost']}", pct(tt["strong_rate_A"]), pct(tt["strong_rate_B"]), pct(tt["strong_rate_C"]),
                         pct(tt["idea_bound_share"]), pct(tt["mismatched_idea_bound_share_mean"]), pv(tt["p_idea_bound_against_mismatch"])])
        rows.append([NAMES[o], "twins beat both nulls for idea units", "", "", "", "", "", "", "", "", "", "", "", "", "", str(tw["twins_beat_both_nulls_for_idea_units"])])
    if rows:
        parts.append(f"## Twins of the {NAMES[ORDER[0]]} units in other models\n\n" +
                     table(["Other model", "Units", "n", "Twin r, original sentences (median)", "Twin r, name-free", "Word-matched controls, name-free", "Depth-matched controls, name-free",
                            "Twin minus word-matched controls (95% interval)", "Units with tester texts", "Twin on the tester's texts: won / tied / lost", "A", "B", "C",
                            "Idea-bound", "Mismatched texts", "p against mismatch"], rows))

    # ---- the six confirmatory tests
    cf = load("confirmatory.json")
    if cf:
        rows = [[k, t.get("verdict", ""), pv(t.get("p")), pv(t.get("holm_adjusted_p")), {True: "yes", False: "no", None: "not yet run"}[t["passes"]]] for k, t in cf["tests"].items()]
        parts.append(f"## The six confirmatory tests ({NAMES[cf['model']]})\n\n" + table(["Test", "Verdict", "p", "Holm-adjusted p", "Passes"], rows))

    text = "# Tables for paper v2\n\nWritten by `v2/analysis/paper_tables.py` from the committed results. Do not edit by hand.\n\n" + "\n\n".join(parts) + "\n"
    (C.RESULTS / "paper_tables.md").write_text(text)
    print("wrote", C.RESULTS / "paper_tables.md", f"({len(parts)} tables)")


if __name__ == "__main__":
    main()
