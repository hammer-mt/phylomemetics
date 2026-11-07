"""Recognition and ablation (PLAN.md, Sections 7 and 11).

    cd study && ../.venv/bin/python v2/analysis/causal.py identify-items <model-tag>     prompts for the blind judge of the replies to question 2
    cd study && ../.venv/bin/python v2/analysis/causal.py identify-collect <model-tag>   -> results/v2/<tag>/identify_judge.json
    cd study && ../.venv/bin/python v2/analysis/causal.py recognition <model-tag>
    cd study && ../.venv/bin/python v2/analysis/causal.py ablation <model-tag>

recognition  reads v2/acts/<tag>/recognise_hero.json and recognise_title.json (gpu/ablate.py on the files written by
             build_probe_jobs.py) and results/v2/<tag>/identify_judge.json (the blind judge's verdict on each reply to
             question 2); writes results/v2/recognition_<tag>.json
ablation     reads v2/acts/<tag>/ablate.json (and the two secondaries if present); writes results/v2/ablation_<tag>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import common as C

import experiments as X
from build_probe_jobs import BATMAN, HEROES

BAT_LABELS = [X.label[b] for b in X.BAT]


def identify_dir(tag):
    d = Path("v2/cache/identify") / tag
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def identify_items(tag, per_prompt=57):
    """The judge sees each reply to question 2 beside the work's true title, in a random order, and nothing else."""
    from explain_agents import ANSWER, wrapped
    rng = np.random.default_rng(C.SEED + 61)
    title = {i["id"]: i for i in json.load(open(C.ACTS / tag / "recognise_title.json"))["items"]}
    works = {w["label"]: w for w in X.works}
    labels = [w["label"] for w in X.works]
    order = rng.permutation(len(labels))
    d = identify_dir(tag)
    template = Path("v2/prompts/identify_judge.md").read_text()
    paths = []
    for b in range(0, len(labels), per_prompt):
        lines = [f"[{labels[i]}] true title: {works[labels[i]]['title']} | reply: {title[labels[i]]['untouched']['reply']!r}" for i in order[b:b + per_prompt]]
        p = d / f"batch_{b // per_prompt}.prompt.txt"
        p.write_text(wrapped(template.replace("{items}", "\n".join(lines)) + ANSWER.format(out=d / f"batch_{b // per_prompt}.out.json", what="item")))
        paths.append(str(p))
    json.dump(paths, open(d / "prompts.json", "w"))
    print(f"{tag}: {len(labels)} replies in {len(paths)} prompt files -> {d}/prompts.json")


def identify_collect(tag):
    out = {}
    for f in sorted(identify_dir(tag).glob("batch_*.out.json")):
        for r in json.load(open(f)):
            out[str(r["id"]).strip("[]")] = bool(r["same_work"])
    missing = [w["label"] for w in X.works if w["label"] not in out]
    C.save({"model": tag, "n": len(out), "missing": missing, "items": out}, f"{tag}/identify_judge.json")
    print(f"{tag}: {len(out)} verdicts, {sum(out.values())} identified; missing {missing}")


def says(reply, names):
    r = reply.lower()
    return any(n.lower().removeprefix("the ") in r for n in names)


def recognition(tag):
    hero = {i["id"]: i for i in json.load(open(C.ACTS / tag / "recognise_hero.json"))["items"]}
    title = {i["id"]: i for i in json.load(open(C.ACTS / tag / "recognise_title.json"))["items"]}
    judged = json.load(open(C.RESULTS / tag / "identify_judge.json"))["items"]
    works = []
    for w in X.works:
        lab = w["label"]
        h = hero[lab]["untouched"]
        own = HEROES.get(lab, [])
        each = dict(zip(BATMAN + own, h["logp_each"]))
        works.append({"work": lab, "group": w["group"], "hero_reply": h["reply"], "title_reply": title[lab]["untouched"]["reply"],
                      "logp_batman": float(np.logaddexp.reduce([each[n] for n in BATMAN])),
                      "logp_own_hero": float(np.logaddexp.reduce([each[n] for n in own])) if own else None,
                      "says_batman": says(h["reply"], BATMAN), "names_own_hero": says(h["reply"], own) if own else None,
                      "identified": bool(judged[lab])})
    by = {r["work"]: r for r in works}
    bat = [by[b] for b in BAT_LABELS]
    hero_films = [by[X.label[w]] for w in X.HERO_FILM_COUSINS]
    films = [by[X.label[w]] for w in X.FILMS]
    out = {"model": tag,
           "batman_works_named_batman": int(sum(r["says_batman"] for r in bat)), "batman_works_identified": int(sum(r["identified"] for r in bat)),
           "hero_films_naming_their_own_hero": int(sum(bool(r["names_own_hero"]) for r in hero_films)),
           "hero_films_with_a_hero_to_name": int(sum(r["names_own_hero"] is not None for r in hero_films)),
           "hero_films_called_batman": int(sum(r["says_batman"] for r in hero_films)), "hero_films_identified": int(sum(r["identified"] for r in hero_films)),
           "ordinary_films_called_batman": int(sum(r["says_batman"] for r in films)), "ordinary_films_identified": int(sum(r["identified"] for r in films)),
           "logp_batman_median": {"batman": float(np.median([r["logp_batman"] for r in bat])), "hero_films": float(np.median([r["logp_batman"] for r in hero_films])),
                                  "ordinary_films": float(np.median([r["logp_batman"] for r in films]))},
           "auc_logp_batman_batman_vs_hero_films": float(X.auc_matrix([r["logp_batman"] for r in bat], [r["logp_batman"] for r in hero_films]).mean())}
    exp = json.load(open(C.RESULTS / "experiments_v2.json")).get(tag)
    if exp:
        a = exp["E1"]["canon_share"]["A_name_free"]
        score, auc_w = a["held_out_score"], a["per_batman_vs_hero"]
        ok = [b for b in BAT_LABELS if score[b] is not None]
        rho = spearmanr([score[b] for b in ok], [by[b]["logp_batman"] for b in ok])
        named = [b for b in BAT_LABELS if by[b]["says_batman"]]
        unnamed = [b for b in BAT_LABELS if not by[b]["says_batman"]]
        out["E1"] = {"rank_correlation_canon_share_with_logp_batman": {"rho": float(rho.statistic), "p_two_sided": float(rho.pvalue), "n": len(ok)},
                     "auc_vs_hero_films_for_works_the_model_names": float(np.mean([auc_w[b] for b in named])) if named else None,
                     "auc_vs_hero_films_for_works_it_does_not_name": float(np.mean([auc_w[b] for b in unnamed])) if unnamed else None,
                     "n_named": len(named), "n_not_named": len(unnamed)}
        both = {t: bool(by[t]["identified"] and any(by[s]["identified"] for s in srcs)) for t, srcs in X.ADAPTATIONS.items()}
        out["E2"] = {"adaptation_and_a_source_both_identified": both, "n_pairs_identified": int(sum(both.values()))}
        for key, rows in (("E2_share", exp["E2"]["primary"]["rows"]),
                          ("residual_share", exp["E2"]["residual_test_beyond_words"]["decision_words_by_frequency"]["rows"])):
            share = {r["target"]: r["share_of_comics_beating_source"] for r in rows}
            out["E2"][key] = {"identified_pairs": float(np.mean([share[t] for t in both if both[t]])) if any(both.values()) else None,
                              "other_pairs": float(np.mean([share[t] for t in both if not both[t]])) if not all(both.values()) else None,
                              "by_adaptation": share}
    out["works"] = works
    C.save(out, f"recognition_{tag}.json")
    print(tag, {k: v for k, v in out.items() if k not in ("works", "E1", "E2")})
    print("  E1:", out.get("E1"), "| E2 pairs identified:", out.get("E2", {}).get("n_pairs_identified"))


def drops(path):
    """Per item: the drop in log-probability of the hero's names for every condition."""
    data = json.load(open(path))
    rows = {}
    for it in data["items"]:
        base = it["untouched"]["logp"]
        rows[it["id"]] = {"untouched": base, "reply": it["untouched"].get("reply"), "drop": {k: base - v["logp"] for k, v in it["conditions"].items()},
                          "removed": {k: v.get("removed") for k, v in it["conditions"].items()},
                          "reply_canon": it["conditions"].get("canon", {}).get("reply")}
    return data, rows


def set_test(rows, ids, prefix):
    """Mean drop over the works for the canon and for every random set of one control; p over sets of units."""
    ks = sorted({k for i in ids for k in rows[i]["drop"] if k.startswith(prefix + ":")}, key=lambda k: int(k.split(":")[1]))
    if not ks:
        return None
    canon = float(np.mean([rows[i]["drop"]["canon"] for i in ids]))
    rand = np.array([[rows[i]["drop"][k] for k in ks] for i in ids])                 # [works, sets]
    means = rand.mean(0)
    per_work_rank = {i: int(1 + (rand[j] >= rows[i]["drop"]["canon"]).sum()) for j, i in enumerate(ids)}
    rem_canon = float(np.mean([rows[i]["removed"]["canon"] or 0 for i in ids]))
    rem = np.array([[rows[i]["removed"][k] or 0 for k in ks] for i in ids]).mean(0)
    return {"n_sets": len(ks), "smallest_possible_p": 1 / (len(ks) + 1), "mean_drop_canon": canon, "mean_drop_random_sets_mean": float(means.mean()), "mean_drop_random_sets_max": float(means.max()),
            "random_sets_at_least_canon": int((means >= canon).sum()), "p": float((1 + (means >= canon).sum()) / (len(ks) + 1)),
            "activation_removed": {"canon": rem_canon, "sets_mean": float(rem.mean()), "sets_min": float(rem.min()), "sets_max": float(rem.max()),
                                   "sets_removing_at_least_as_much": int((rem >= rem_canon).sum())},
            "drop_per_unit_of_activation_removed": {"canon": canon / max(rem_canon, 1e-9), "sets_mean": float((means / np.maximum(rem, 1e-9)).mean()),
                                                    "sets_at_least_canon": int(((means / np.maximum(rem, 1e-9)) >= canon / max(rem_canon, 1e-9)).sum())},
            "works_where_canon_beats_every_set": int(sum(r == 1 for r in per_work_rank.values())),
            "works_where_canon_beats_the_median_set": int(sum(rows[i]["drop"]["canon"] > np.median(rand[j]) for j, i in enumerate(ids))),
            "rank_of_canon_among_sets_by_work": per_work_rank}


def ablation(tag):
    rng = np.random.default_rng(C.SEED)
    out = {"model": tag, "design": json.load(open(C.RESULTS / tag / "ablation_design.json"))}
    for name, fname in (("story_tokens_name_free", "ablate.json"), ("every_position_name_free", "ablate_all_positions.json"), ("story_tokens_original", "ablate_original.json"),
                        ("strong_firings_only", "ablate_strong.json"), ("weak_activity_only", "ablate_weak.json")):
        path = C.ACTS / tag / fname
        if not path.exists():
            continue
        data, rows = drops(path)
        bat = [b for b in BAT_LABELS if b in rows]
        rel = [X.label[w] for w in X.HERO_FILM_COUSINS if X.label[w] in rows]
        res = {"where": data["where"], "n_batman_works": len(bat), "n_hero_films": len(rel),
               "untouched_logp_batman_median": float(np.median([rows[b]["untouched"] for b in bat])),
               "matched_random_sets": set_test(rows, bat, "rand"), "magnitude_matched_random_sets": set_test(rows, bat, "magnitude"),
               "distinctive_random_sets": set_test(rows, bat, "distinctive"),
               "canon_ablated_reply_still_says_batman": int(sum(says(rows[b]["reply_canon"] or "", BATMAN) for b in bat)),
               "untouched_reply_says_batman": int(sum(says(rows[b]["reply"] or "", BATMAN) for b in bat))}
        if rel:
            d_b, d_r = np.array([rows[b]["drop"]["canon"] for b in bat]), np.array([rows[r]["drop"]["canon"] for r in rel])
            obs = d_b.mean() - d_r.mean()
            pooled = np.r_[d_b, d_r]
            null = np.array([(lambda p: p[:len(d_b)].mean() - p[len(d_b):].mean())(rng.permutation(pooled)) for _ in range(10000)])
            res["specificity"] = {"mean_drop_batman_works": float(d_b.mean()), "mean_drop_hero_films_own_hero": float(d_r.mean()),
                                  "difference": float(obs), "p_one_sided": float((1 + (null >= obs).sum()) / 10001),
                                  "hero_films_matched_random_sets": set_test(rows, rel, "rand"),
                                  "drop_by_hero_film": {r: float(rows[r]["drop"]["canon"]) for r in rel}}
        res["by_work"] = {i: {"untouched": rows[i]["untouched"], "drop_canon": rows[i]["drop"]["canon"],
                              "drop_random_mean": float(np.mean([v for k, v in rows[i]["drop"].items() if k.startswith("rand:")] or [np.nan])),
                              "reply": rows[i]["reply"], "reply_canon_off": rows[i]["reply_canon"]} for i in bat + rel}
        out[name] = res
    p = out.get("story_tokens_name_free", {})
    if p:
        m, g, s = p["matched_random_sets"], p["magnitude_matched_random_sets"], p.get("specificity", {})
        out["criterion"] = {"p_against_sets_matched_on_works": m["p"], "p_against_sets_also_matched_on_activity": g["p"], "specificity_p": s.get("p_one_sided"),
                            "met": bool(m["p"] < 0.05 and g["p"] < 0.05 and s.get("p_one_sided", 1) < 0.05)}
        print(tag, "ablation: mean drop canon", round(m["mean_drop_canon"], 3), "random sets", round(m["mean_drop_random_sets_mean"], 3),
              "p", round(m["p"], 4), "| magnitude-matched sets", round(g["mean_drop_random_sets_mean"], 3), "p", round(g["p"], 4),
              "| specificity p", s.get("p_one_sided"), "| met:", out["criterion"]["met"])
    C.save(out, f"ablation_{tag}.json")


if __name__ == "__main__":
    {"recognition": recognition, "ablation": ablation, "identify-items": identify_items, "identify-collect": identify_collect}[sys.argv[1]](sys.argv[2])
