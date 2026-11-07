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
