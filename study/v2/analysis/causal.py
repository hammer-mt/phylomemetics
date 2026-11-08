"""Recognition and ablation (PLAN.md, Sections 7 and 11).

    cd study && ../.venv/bin/python v2/analysis/causal.py identify-items <model-tag>     prompts for the blind judge of the replies to question 2
    cd study && ../.venv/bin/python v2/analysis/causal.py identify-collect <model-tag>   -> results/v2/<tag>/identify_judge.json
    cd study && ../.venv/bin/python v2/analysis/causal.py recognition <model-tag>
    cd study && ../.venv/bin/python v2/analysis/causal.py ablation <model-tag>
    cd study && ../.venv/bin/python v2/analysis/causal.py begun-form <model-tag>         the follow-up's check of the prefix (PLAN.md Section 18, change 33)
    cd study && ../.venv/bin/python v2/analysis/causal.py begun-probe <model-tag>        the follow-up's descriptive control

recognition  reads v2/acts/<tag>/recognise_hero.json and recognise_title.json (gpu/ablate.py on the files written by
             build_probe_jobs.py) and results/v2/<tag>/identify_judge.json (the blind judge's verdict on each reply to
             question 2); writes results/v2/recognition_<tag>.json
ablation     reads v2/acts/<tag>/ablate.json (and the secondaries if present), and the follow-up's ablate_begun*.json
             if present; writes results/v2/ablation_<tag>.json
begun-form   reads v2/acts/<tag>/begun_form_<slug>.json (question 1 with the reply begun, the 69 ordinary films);
             writes results/v2/<tag>/begun_form_check.json, which names the prefix the follow-up uses
begun-probe  reads v2/acts/<tag>/recognise_hero_begun.json (the same on all 114 name-free texts);
             writes results/v2/recognition_begun_<tag>.json
"""
import itertools
import json
import math
import re
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


def starts_with_name(reply):
    """A reply that begins with a name, with or without "The" ("Batman", "The Joker"), as against a sentence about the
    story ("The story is about ..."), a placeholder code ("X2 is ...") or markup (PLAN.md Section 18, changes 16 and 28)."""
    words = reply.strip().split()
    if words and words[0].lower() == "the":
        words = words[1:]
    if not words:
        return False
    first = words[0]
    if not (first[0].isalpha() and first[0].isupper()) or (first[0] == "X" and first[1:].rstrip(",.:;").isdigit()):
        return False
    return first.lower().rstrip(",.:;") not in ("story", "summary", "hero", "protagonist", "main", "answer", "plot", "character", "name", "this", "it", "based", "in", "there")


# first words that open a description, a hedge or a refusal and not a name, when a reply has been begun for the model
NOT_A_NAME_AFTER_A_PREFIX = ("a", "an", "i", "he", "she", "they", "that", "likely", "probably", "possibly", "perhaps", "maybe", "unknown", "none", "no", "not", "n/a",
                             "unnamed", "unclear", "unspecified", "sure", "yes", "okay", "ok", "here", "according", "cannot", "sorry",
                             "it", "this", "there", "most", "presumably", "apparently", "could", "might", "uncertain", "unidentified", "undetermined", "unable", "unsure",
                             "insufficient", "given", "since", "because", "although", "while", "if", "we", "you", "my", "these", "those", "nothing", "neither", "either",
                             "someone", "somebody", "something", "certainly", "clearly", "obviously", "definitely", "can", "let", "well", "who", "what", "which", "whoever",
                             "known", "also", "called", "named", "judging", "looking", "considering", "assuming", "please")


def begins_with_name_after_prefix(raw, next_tokens=None):
    """With the reply begun for the model ("Hero:") the score reads what follows one space. The reply, as written after
    the prefix, begins with a name only if it goes on with exactly one space and then a name. A line break, no space,
    two spaces, markup, a quote mark, a code (with or without something attached), a sentence, a description
    (" A masked vigilante"), a hedge (" Likely Batman") or a refusal (" Unknown") does not count, and nor does a
    lower-case " the", which is not a scored path (PLAN.md Section 18, change 33). The rule reads the first word only
    (the second after "The"): a capitalised first word that is on neither list is read as a name, whatever follows.
    starts_with_name is left as it was, so no registered reading changes. next_tokens: the most probable next tokens
    after the prefix, if kept; a lone space token followed by the name reads the same as " Batman" and is not the
    scored path either."""
    if not (bool(raw) and len(raw) > 1 and raw[0] == " " and not raw[1].isspace() and starts_with_name(raw)):
        return False
    if next_tokens and not next_tokens[0][0].strip():
        return False
    words = raw.split()
    if words[0] == "the":
        return False
    first = words[1] if words[0] == "The" and len(words) > 1 else words[0]
    key = re.split("['\u2019]", first.lower().rstrip(",.:;"))[0]          # "It's" and "I'm" are read as "it" and "i"
    return not re.match(r"X\d", first) and key not in NOT_A_NAME_AFTER_A_PREFIX


def prompt_ends_with(items, prefix):
    """Every prompt must end with the empty thinking block and then the prefix: an older script, or a chat template
    called without thinking switched off, would end differently and the score would be read at the wrong place."""
    for it in items:
        tail = "".join(it["prompt_tail_tokens"])
        assert tail.endswith("</think>\n\n" + prefix), f"{it['id']}: the prompt ends {tail[-40:]!r}, not with the empty thinking block and {prefix!r}"


def same_story_side(follow, registered, per_condition):
    """The follow-up output must agree with the registered output it repeats on everything that happens before the
    prefix: where the story's tokens are, the length of the prompt (two tokens more), the conditions, and the activation
    that was switched off (within 5%: rounding on a GPU depends on the length of the prompt, and another model, scale
    or unit file would differ by far more). Stops if they differ.
    per_condition: compare every condition of every item (the main run); otherwise the totals (the strong and weak
    versions, where one activation crossing the 30% line moves a small sum by more than rounding)."""
    reg = {it["id"]: it for it in registered["items"]}
    assert [it["id"] for it in follow["items"]] == [it["id"] for it in registered["items"]], "not the registered items"
    worst, tot_f, tot_r = 0.0, 0.0, 0.0
    for it in follow["items"]:
        r = reg[it["id"]]
        assert it["story_tokens"] == r["story_tokens"], f"{it['id']}: the story's tokens are not where they were in the registered run"
        assert it["n_prompt_tokens"] == r["n_prompt_tokens"] + 2, f"{it['id']}: the prompt is not the registered prompt plus the prefix"
        assert list(it["conditions"]) == list(r["conditions"]), f"{it['id']}: not the registered conditions"
        for k, v in it["conditions"].items():
            a, b = v["removed"], r["conditions"][k]["removed"]
            worst = max(worst, abs(a - b) / max(abs(b), 1.0))
            tot_f, tot_r = tot_f + a, tot_r + b
    total = abs(tot_f - tot_r) / max(abs(tot_r), 1.0)
    assert (worst if per_condition else total) <= 0.05, (f"the activation switched off differs from the registered run's by up to {worst:.2%} in one condition "
                                                           f"and {total:.2%} in all: not the same model, texts, units or positions")
    return {"largest_relative_difference_in_the_activation_switched_off": {"in_one_condition_of_one_text": worst, "in_total": total},
            "checked": "every condition of every text" if per_condition else "the total (one condition can move when an activation crosses the 30% line)"}


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


def log_odds(lp):
    """log(p / (1 - p)) from log p. A drop in log p shrinks with the model's confidence; a drop in log-odds does not."""
    lp = min(lp, -1e-12)
    return lp - np.log(-np.expm1(lp))


def drops(path, odds=True):
    """Per item: the drop in the log-odds (or, as registered, the log-probability) of the hero's names for every condition."""
    data = json.load(open(path))
    f = log_odds if odds else (lambda x: x)
    rows = {}
    for it in data["items"]:
        base = it["untouched"]["logp"]
        canon = it["conditions"].get("canon", {})
        rows[it["id"]] = {"untouched": base, "reply": it["untouched"].get("reply"), "drop": {k: f(base) - f(v["logp"]) for k, v in it["conditions"].items()},
                          "removed": {k: v.get("removed") for k, v in it["conditions"].items()},
                          "reply_canon": canon.get("reply"), "reply_raw": it["untouched"].get("reply_raw"), "reply_canon_raw": canon.get("reply_raw"),
                          "next_tokens": it["untouched"].get("next_tokens"), "next_tokens_canon": canon.get("next_tokens"), "answers": it["answers"]}
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
            "rank_of_canon_among_sets_by_work": per_work_rank,
            "mean_drop_of_each_set": [round(float(x), 4) for x in means], "activation_removed_by_each_set": [round(float(x), 1) for x in rem]}


def exact_specificity_p(d_b, d_r):
    """The share of all ways of calling len(d_r) of the texts hero films that give a difference at least as large as the
    one seen. With the totals fixed, the difference of the two means is at least as large exactly when the hero films'
    sum is no larger."""
    pooled = np.r_[d_b, d_r]
    n, k = len(pooled), len(d_r)
    seen = d_r.sum()
    it = itertools.combinations(range(n), k)
    total = at_most = 0
    while True:
        chunk = np.fromiter(itertools.chain.from_iterable(itertools.islice(it, 400000)), dtype=np.int16)
        if not len(chunk):
            break
        sums = pooled[chunk.reshape(-1, k)].sum(1)
        total += len(sums)
        at_most += int((sums <= seen + 1e-9).sum())
    assert total == math.comb(n, k)
    return {"p_one_sided": at_most / total, "relabellings": total}


REGISTERED = (("story_tokens_name_free", "ablate.json"), ("every_position_name_free", "ablate_all_positions.json"), ("story_tokens_original", "ablate_original.json"),
              ("strong_firings_only", "ablate_strong.json"), ("weak_activity_only", "ablate_weak.json"))
# the follow-up of PLAN.md Section 18, change 33: the same jobs with the reply begun for the model; third, the registered version it repeats
FOLLOW_UP = (("reply_begun", "ablate_begun.json", "story_tokens_name_free"), ("reply_begun_strong_firings_only", "ablate_begun_strong.json", "strong_firings_only"),
             ("reply_begun_weak_activity_only", "ablate_begun_weak.json", "weak_activity_only"))
FORM_CHECK = "qwen3-32b/begun_form_check.json"          # the form check is Qwen3-32B's, and the prefix it selects is used for both models


def prefix_chosen():
    return json.load(open(C.RESULTS / FORM_CHECK))["prefix_chosen"]


def version(path, rng, begun=None):
    """One version of the ablation. begun: the prefix the reply was begun with (the follow-up), which changes how a
    reply is read and adds the descriptions of change 33."""
    data, rows = drops(path)
    _, rows_lp = drops(path, odds=False)
    if begun:           # refuse an output that was not made with the prefix: the old script would ignore it and score the wrong place
        assert data.get("reply_prefix") == begun, f"{path}: made with the prefix {data.get('reply_prefix')!r}, not {begun!r}"
        assert all(a.startswith(" ") and not a[1:2].isspace() for it in data["items"] for a in it["answers"]), f"{path}: an answer does not begin with one space"
        assert all(r["reply_raw"] is not None and r["next_tokens"] for r in rows.values()), f"{path}: the replies were not kept as written"
        prompt_ends_with(data["items"], begun)
        raw = lambda i, canon=False: rows[i]["reply_canon_raw" if canon else "reply_raw"] or ""
        named_at = lambda i, canon=False: begins_with_name_after_prefix(raw(i, canon), rows[i]["next_tokens_canon" if canon else "next_tokens"])
    else:
        assert not data.get("reply_prefix"), f"{path}: a registered version must have no reply prefix"
        raw = lambda i, canon=False: rows[i]["reply_canon" if canon else "reply"] or ""
        named_at = lambda i, canon=False: starts_with_name(raw(i, canon))
    bat = [b for b in BAT_LABELS if b in rows]
    rel = [X.label[w] for w in X.HERO_FILM_COUSINS if X.label[w] in rows]
    res = {"where": data["where"], "n_batman_works": len(bat), "n_hero_films": len(rel),
           "untouched_logp_batman_median": float(np.median([rows[b]["untouched"] for b in bat])),
           "matched_random_sets": set_test(rows, bat, "rand"), "magnitude_matched_random_sets": set_test(rows, bat, "magnitude"),
           "distinctive_random_sets": set_test(rows, bat, "distinctive"),
           "scored_in": "log-odds of the hero's names (PLAN.md Section 18, change 16); the registered log-probability form is given beside it",
           "registered_form_drop_in_log_probability": {"matched_random_sets": set_test(rows_lp, bat, "rand"),
                                                       "magnitude_matched_random_sets": set_test(rows_lp, bat, "magnitude")},
           "mean_drop_where_the_untouched_model_names_batman": float(np.mean([rows[b]["drop"]["canon"] for b in bat if says(rows[b]["reply"] or "", BATMAN)] or [np.nan])),
           "mean_drop_where_it_does_not": float(np.mean([rows[b]["drop"]["canon"] for b in bat if not says(rows[b]["reply"] or "", BATMAN)] or [np.nan])),
           "untouched_replies_that_do_not_start_with_a_name": int(sum(not named_at(b) for b in bat)),
           "canon_ablated_reply_still_says_batman": int(sum(says(rows[b]["reply_canon"] or "", BATMAN) for b in bat)),
           "untouched_reply_says_batman": int(sum(says(rows[b]["reply"] or "", BATMAN) for b in bat))}
    if begun:
        res = dict({"question": data["question"], "reply_prefix": begun,
                    "a_reply_begins_with_a_name_if": "what the model writes after the prefix is exactly one space and then a name (PLAN.md Section 18, change 33)"}, **res)
    if rel:
        d_b, d_r = np.array([rows[b]["drop"]["canon"] for b in bat]), np.array([rows[r]["drop"]["canon"] for r in rel])
        obs = d_b.mean() - d_r.mean()
        pooled = np.r_[d_b, d_r]
        null = np.array([(lambda p: p[:len(d_b)].mean() - p[len(d_b):].mean())(rng.permutation(pooled)) for _ in range(10000)])
        res["specificity"] = {"mean_drop_batman_works": float(d_b.mean()), "mean_drop_hero_films_own_hero": float(d_r.mean()),
                              "difference": float(obs), "p_one_sided": float((1 + (null >= obs).sum()) / 10001),
                              "hero_films_matched_random_sets": set_test(rows, rel, "rand"),
                              "drop_by_hero_film": {r: float(rows[r]["drop"]["canon"]) for r in rel}}
        if begun:
            named = [r for r in rel if says(rows[r]["reply"] or "", HEROES[r])]
            res["specificity"]["hero_films_described"] = {
                "untouched_replies_that_begin_with_a_name": int(sum(named_at(r) for r in rel)),
                "untouched_replies_that_name_their_own_hero": len(named),
                "mean_drop_over_the_films_whose_own_hero_the_untouched_model_names": float(np.mean([rows[r]["drop"]["canon"] for r in named])) if named else None,
                "mean_drop_over_all_the_films": float(d_r.mean()),
                "by_film": {r: {"reply_as_written": raw(r), "begins_with_a_name": bool(named_at(r)), "names_its_own_hero": r in named,
                                "untouched_logp_own_hero": rows[r]["untouched"], "drop_canon": float(rows[r]["drop"]["canon"]),
                                "reply_canon_off_as_written": raw(r, True)} for r in rel}}
    named = [b for b in bat if named_at(b)]
    if 0 < len(named) < len(bat):          # descriptive (PLAN.md Section 18, change 32): the works the evaluability rule does not object to
        res["works_whose_untouched_reply_begins_with_a_name" if begun else "post_hoc_works_whose_untouched_reply_begins_with_a_name"] = {
            "n_works": len(named), "works_left_out": [b for b in bat if b not in named],
            "matched_random_sets": {k: v for k, v in set_test(rows, named, "rand").items() if not isinstance(v, (list, dict))},
            "magnitude_matched_random_sets": {k: v for k, v in set_test(rows, named, "magnitude").items() if not isinstance(v, (list, dict))},
            "mean_drop_canon_in_the_works_left_out": float(np.mean([rows[b]["drop"]["canon"] for b in bat if b not in named]))}
    if begun:           # a reply can change form when the canon is switched off, and a score on the next word counts that as a fall
        kept = [b for b in bat if named_at(b, True)]
        res["replies_with_the_canon_off"] = {
            "do_not_begin_with_a_name": len(bat) - len(kept), "works": [b for b in bat if b not in kept],
            "of_which_the_untouched_reply_did": int(sum(named_at(b) for b in bat if b not in kept)),
            "mean_drop_canon_with_those_works_left_out": float(np.mean([rows[b]["drop"]["canon"] for b in kept])) if kept else None}
    res["by_work"] = {i: {"untouched": rows[i]["untouched"], "drop_canon": rows[i]["drop"]["canon"],
                          "drop_random_mean": float(np.mean([v for k, v in rows[i]["drop"].items() if k.startswith("rand:")] or [np.nan])),
                          "reply": rows[i]["reply"], "reply_canon_off": rows[i]["reply_canon"]} for i in bat + rel}
    if begun:
        for i in bat + rel:
            res["by_work"][i].update(reply_as_written=raw(i), reply_canon_off_as_written=raw(i, True), begins_with_a_name=bool(named_at(i)),
                                     next_tokens=rows[i]["next_tokens"], next_tokens_canon_off=rows[i]["next_tokens_canon"])
        res["prompt_tail_tokens"] = data["items"][0].get("prompt_tail_tokens")
    return res, rows, bat, rel, data


def criterion(p):
    m, g, s = p["matched_random_sets"], p["magnitude_matched_random_sets"], p.get("specificity", {})
    evaluable = p["untouched_replies_that_do_not_start_with_a_name"] <= 4
    return {"p_against_sets_matched_on_works": m["p"], "p_against_sets_also_matched_on_activity": g["p"], "specificity_p": s.get("p_one_sided"),
            "evaluable_with_this_question": bool(evaluable),
            "met": bool(evaluable and m["p"] < 0.05 and g["p"] < 0.05 and s.get("p_one_sided", 1) < 0.05)}


def ablation(tag):
    rng = np.random.default_rng(C.SEED)
    out = {"model": tag, "design": json.load(open(C.RESULTS / tag / "ablation_design.json"))}
    rows_of = {}
    for name, fname in REGISTERED:
        path = C.ACTS / tag / fname
        if not path.exists():
            continue
        out[name], rows_of[name], _, _, _ = version(path, rng)
    p = out.get("story_tokens_name_free", {})
    if p:
        m, g = p["matched_random_sets"], p["magnitude_matched_random_sets"]
        out["criterion"] = criterion(p)
        print(tag, "ablation: mean drop canon", round(m["mean_drop_canon"], 3), "random sets", round(m["mean_drop_random_sets_mean"], 3),
              "p", round(m["p"], 4), "| magnitude-matched sets", round(g["mean_drop_random_sets_mean"], 3), "p", round(g["p"], 4),
              "| specificity p", out["criterion"]["specificity_p"], "| met:", out["criterion"]["met"])
    if any((C.ACTS / tag / fname).exists() for _, fname, _ in FOLLOW_UP):
        # the seeded specificity p depends on how many draws came before it: all five registered versions must be in place, as when change 33 was written
        assert all(name in out for name, _ in REGISTERED), "the follow-up is scored with the five registered output files in place"
        prefix = prefix_chosen()
        assert prefix, "the form check chose no prefix: the follow-up is not run"
        for name, fname, repeats in FOLLOW_UP:
            path = C.ACTS / tag / fname
            if not path.exists():
                continue
            res, rows, bat, rel, data = version(path, rng, begun=prefix)
            res["same_story_side_as_the_registered_run"] = same_story_side(data, json.load(open(C.ACTS / tag / dict(REGISTERED)[repeats])), per_condition=name == "reply_begun")
            reg = rows_of[repeats]          # the two runs' scores are not on the same scale (different first tokens); their ranks can be compared
            ok = [b for b in bat if starts_with_name(reg[b]["reply"] or "")]
            rho = lambda ids, f: (lambda r: {"rho": float(r.statistic), "n": len(ids)})(spearmanr([f(rows[b]) for b in ids], [f(reg[b]) for b in ids]))
            res["agreement_with_the_registered_run"] = {
                "what": "rank correlations over the Batman works between this run and the registered run it repeats",
                "untouched_log_odds": rho(bat, lambda r: log_odds(r["untouched"])), "drop_canon": rho(bat, lambda r: r["drop"]["canon"]),
                "works_whose_registered_untouched_reply_began_with_a_name": {
                    "untouched_log_odds": rho(ok, lambda r: log_odds(r["untouched"])), "drop_canon": rho(ok, lambda r: r["drop"]["canon"])}}
            out[name] = res
        q = out.get("reply_begun")
        if q:
            m, g, s = q["matched_random_sets"], q["magnitude_matched_random_sets"], q["specificity"]
            d = q["specificity"]["drop_by_hero_film"]
            exact = exact_specificity_p(np.array([q["by_work"][b]["drop_canon"] for b in BAT_LABELS if b in q["by_work"] and b not in d]), np.array(list(d.values())))
            s["exact"] = exact
            c = criterion(q)
            failed = [k for k, v in (("the sets matched on works", m["p"]), ("the sets also matched on activity", g["p"]), ("specificity", s["p_one_sided"])) if not v < 0.05]
            out["criterion_reply_begun"] = dict(
                {"what": "a follow-up test, fixed in PLAN.md Section 18, change 33, after the registered result was seen and before it was run; "
                         "not one of the six confirmatory tests", "reply_prefix": prefix}, **c,
                specificity_p_exact=exact["p_one_sided"], untouched_replies_that_do_not_start_with_a_name=q["untouched_replies_that_do_not_start_with_a_name"],
                outcome="not evaluable" if not c["evaluable_with_this_question"] else "met" if c["met"] else "not met", parts_that_failed=failed)
            print(tag, "follow-up (reply begun with", repr(prefix) + "): mean drop canon", round(m["mean_drop_canon"], 3), "random sets", round(m["mean_drop_random_sets_mean"], 3),
                  "p", round(m["p"], 4), "| magnitude-matched p", round(g["p"], 4), "| specificity p", s["p_one_sided"], "exact", round(exact["p_one_sided"], 5),
                  "| replies not beginning with a name", q["untouched_replies_that_do_not_start_with_a_name"], "| outcome:", out["criterion_reply_begun"]["outcome"])
    C.save(out, f"ablation_{tag}.json")


def begun_form(tag):
    """The prefix is checked for form on texts outside the test (PLAN.md Section 18, change 33): question 1 with the
    reply begun, put to the untouched model on the 69 ordinary films' name-free texts. A prefix passes if at least 62
    of the 69 replies begin with a name. "Hero:" is read first and "Answer:" only if it fails."""
    from build_ablation_followup import PREFIXES
    films = [X.label[w] for w in X.FILMS]
    registered = {it["id"]: it for it in json.load(open(C.ACTS / tag / "recognise_hero.json"))["items"]}
    out = {"model": tag, "rule": "a prefix is used if at least 62 of the 69 ordinary films' replies begin with a name: exactly one space after the prefix and then a name "
                                "(begins_with_name_after_prefix in v2/analysis/causal.py)",
           "prefix_chosen": None, "decided": False, "checks": []}
    for slug, prefix in PREFIXES.items():
        path = C.ACTS / tag / f"begun_form_{slug}.json"
        if not path.exists():
            break
        data = json.load(open(path))
        items = {it["id"]: it["untouched"] for it in data["items"]}
        assert data.get("reply_prefix") == prefix and sorted(items) == sorted(films), f"{path}: not the form check for {prefix!r}"
        assert all(a.startswith(" ") for it in data["items"] for a in it["answers"])
        prompt_ends_with(data["items"], prefix)
        assert all(it["n_prompt_tokens"] == registered[it["id"]]["n_prompt_tokens"] + 2 for it in data["items"]), f"{path}: not the registered probe's prompts plus the prefix"
        ok = {f: bool(begins_with_name_after_prefix(items[f]["reply_raw"], items[f]["next_tokens"])) for f in films}
        passes = sum(ok.values()) >= 62
        out["checks"].append({"prefix": prefix, "n_films": len(films), "replies_that_begin_with_a_name": int(sum(ok.values())), "needed": 62, "passes": bool(passes),
                              "prompt_tail_tokens": data["items"][0].get("prompt_tail_tokens"),
                              "replies_as_written": {f: items[f]["reply_raw"] for f in films}, "begins_with_a_name": ok,
                              "next_tokens": {f: items[f].get("next_tokens") for f in films}})
        if passes:
            out["prefix_chosen"], out["decided"] = prefix, True
            break
    else:
        out["decided"] = True               # both prefixes were read and neither passed: the follow-up is not run
    C.save(out, f"{tag}/begun_form_check.json")
    for c in out["checks"]:
        print(tag, "form check", repr(c["prefix"]), c["replies_that_begin_with_a_name"], "of", c["n_films"], "begin with a name; passes:", c["passes"])
    slug = {v: k for k, v in PREFIXES.items()}.get(out["prefix_chosen"], "none" if out["decided"] else "pending")
    print("PREFIX_CHOSEN", slug)


def begun_probe(tag):
    """The follow-up's descriptive control (PLAN.md Section 18, change 33): the untouched probe of Section 7, question 1,
    with the same prefix, on all 114 name-free texts. Does the begun reply say Batman for everything?"""
    prefix = prefix_chosen()
    data = json.load(open(C.ACTS / tag / "recognise_hero_begun.json"))
    assert data.get("reply_prefix") == prefix and all(a.startswith(" ") for it in data["items"] for a in it["answers"])
    prompt_ends_with(data["items"], prefix)
    registered = {it["id"]: it for it in json.load(open(C.ACTS / tag / "recognise_hero.json"))["items"]}
    assert all(it["n_prompt_tokens"] == registered[it["id"]]["n_prompt_tokens"] + 2 for it in data["items"]), "not the registered probe's prompts plus the prefix"
    hero = {i["id"]: i for i in data["items"]}
    works = []
    for w in X.works:
        lab = w["label"]
        h = hero[lab]["untouched"]
        own = HEROES.get(lab, [])
        each = dict(zip(BATMAN + own, h["logp_each"]))
        works.append({"work": lab, "group": w["group"], "reply_as_written": h["reply_raw"], "begins_with_a_name": bool(begins_with_name_after_prefix(h["reply_raw"], h["next_tokens"])),
                      "logp_batman": float(np.logaddexp.reduce([each[n] for n in BATMAN])),
                      "logp_own_hero": float(np.logaddexp.reduce([each[n] for n in own])) if own else None,
                      "says_batman": says(h["reply"], BATMAN), "names_own_hero": says(h["reply"], own) if own else None, "next_tokens": h.get("next_tokens")})
    by = {r["work"]: r for r in works}
    groups = {"batman_works": [by[b] for b in BAT_LABELS], "hero_films": [by[X.label[w]] for w in X.HERO_FILM_COUSINS], "ordinary_films": [by[X.label[w]] for w in X.FILMS]}
    reg = json.load(open(C.RESULTS / f"recognition_{tag}.json"))
    out = {"model": tag, "what": "question 1 put to the untouched model with its reply begun; descriptive, and changes no outcome", "reply_prefix": prefix,
           "n": {k: len(v) for k, v in groups.items()},
           "replies_that_say_batman": {k: int(sum(r["says_batman"] for r in v)) for k, v in groups.items()},
           "replies_that_begin_with_a_name": {k: int(sum(r["begins_with_a_name"] for r in v)) for k, v in groups.items()},
           "hero_films_naming_their_own_hero": int(sum(bool(r["names_own_hero"]) for r in groups["hero_films"])),
           "hero_films_with_a_hero_to_name": int(sum(r["names_own_hero"] is not None for r in groups["hero_films"])),
           "logp_batman_median": {k: float(np.median([r["logp_batman"] for r in v])) for k, v in groups.items()},
           "auc_logp_batman_batman_vs_hero_films": float(X.auc_matrix([r["logp_batman"] for r in groups["batman_works"]], [r["logp_batman"] for r in groups["hero_films"]]).mean()),
           "registered_probe_without_the_prefix": {"replies_that_say_batman": {"batman_works": reg["batman_works_named_batman"], "hero_films": reg["hero_films_called_batman"],
                                                                                "ordinary_films": reg["ordinary_films_called_batman"]},
                                                   "hero_films_naming_their_own_hero": reg["hero_films_naming_their_own_hero"],
                                                   "hero_films_with_a_hero_to_name": reg["hero_films_with_a_hero_to_name"],
                                                   "auc_logp_batman_batman_vs_hero_films": reg["auc_logp_batman_batman_vs_hero_films"]},
           "works": works}
    C.save(out, f"recognition_begun_{tag}.json")
    print(tag, {k: v for k, v in out.items() if k not in ("works", "what")})


if __name__ == "__main__":
    {"recognition": recognition, "ablation": ablation, "identify-items": identify_items, "identify-collect": identify_collect,
     "begun-form": begun_form, "begun-probe": begun_probe}[sys.argv[1]](sys.argv[2])
