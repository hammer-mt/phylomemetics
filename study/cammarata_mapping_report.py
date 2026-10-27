"""Builds cammarata-mapping.md and cammarata-mapping.json from a cammarata_mapping.py work folder.

Inputs in WORK: candidates.json (select), corpus_units.json (corpus), explanations.json (Agent 1, neuron-first),
agent2_acts.json (probe pass over Agent 2's blind maximum, examples and near misses), steer.json and steer_judge.json
(optional), notes.json (optional hand-written summary and per-meme verdicts).
"""
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent
STRONG = 0.5          # an Agent 2 example "fires strongly" at or above this fraction of the unit's corpus maximum


def esc(t):
    return str(t).replace("|", "\\|").replace("\n", " ")


def trim(t, n=160):
    return t if len(t) <= n else t[:n - 1].rstrip() + "…"


def build(work):
    work = Path(work)
    cands = json.load(open(work / "candidates.json"))
    corp = json.load(open(work / "corpus_units.json"))
    expl = json.load(open(work / "explanations.json")) if (work / "explanations.json").exists() else {}
    a2 = json.load(open(work / "agent2_acts.json")) if (work / "agent2_acts.json").exists() else []
    steer = json.load(open(work / "steer_judge.json")) if (work / "steer_judge.json").exists() else []
    notes = json.load(open(work / "notes.json")) if (work / "notes.json").exists() else {}
    paper = {m["unit"]: m for m in json.load(open(STUDY / "cammarata-audit.json"))["memes"]} if (STUDY / "cammarata-audit.json").exists() else {}
    memes = []
    for m in cands["memes"]:
        uid = m["unit_paper"]
        ch = m["chosen"]
        rec = {"label": m["label"], "kind": m["kind"], "paper_unit": {"unit": uid, "name": corp[str(uid)]["name"], **m["paper_unit"]},
               "n_units_passing_dev": m["n_units_passing_dev"], "candidates": m["candidates"]}
        rec["paper_unit"]["corpus"] = corp.get(str(uid))
        if ch is None:
            rec["chosen"] = None
            memes.append(rec); continue
        cu = corp[str(ch["unit"])]
        c = dict(ch)
        c["corpus"] = cu
        c["same_as_paper"] = ch["unit"] == uid
        c["explanation"] = expl.get(str(ch["unit"]), {})
        probes = [p for p in a2 if p["unit"] == ch["unit"]]
        cmax = cu["corpus_max"] or 1e-6
        def rows(kind):
            return [{"text": p["text"], "act": round(p["units"][str(ch["unit"])]["act"], 3),
                     "ratio_to_max": round(p["units"][str(ch["unit"])]["act"] / cmax, 3),
                     "peak_token": p["units"][str(ch["unit"])]["peak_token"]} for p in probes if p["kind"] == kind]
        bm, ex, nm = rows("blind_max"), rows("example"), rows("near_miss")
        c["agent2"] = {"blind_max": max(bm, key=lambda r: r["act"]) if bm else None, "blind_max_attempts": bm,
                       "examples": ex, "near_misses": nm,
                       "examples_strong": sum(r["ratio_to_max"] >= STRONG for r in ex), "near_misses_strong": sum(r["ratio_to_max"] >= STRONG for r in nm)}
        c["steer"] = [s for s in steer if s.get("unit") == ch["unit"]]
        pres_new = set(cu["presence"]["present_batman"]); pres_paper = set(corp[str(uid)]["presence"]["present_batman"])
        c["presence_vs_paper"] = {"new_n_batman": len(pres_new), "paper_n_batman": len(pres_paper),
                                  "new_batman_above_median": cu["presence"].get("n_batman_above_median"),
                                  "paper_batman_above_median": corp[str(uid)]["presence"].get("n_batman_above_median"),
                                  "both": sorted(pres_new & pres_paper), "only_new": sorted(pres_new - pres_paper), "only_paper": sorted(pres_paper - pres_new),
                                  "paper_unit_in_paper_rule": paper.get(uid, {}).get("n_batman_works")}
        checks = {
            "test_pairs_pass": ch["test_wins"] >= ch["test_n"] - 1,
            "specific": ch["test_mean_pos"] > ch["other_memes_p99"],
            "not_early": ch["layer"] >= 4,
            "explanation_matches": c["explanation"].get("judge_match") == "direct",
            "explanation_partial": c["explanation"].get("judge_match") == "partial",
            "blind_beats_max": bool(c["agent2"]["blind_max"] and c["agent2"]["blind_max"]["act"] >= cmax),
            "blind_in_top8": bool(c["agent2"]["blind_max"] and c["agent2"]["blind_max"]["act"] >= cu["corpus_min_top8"]),
            "examples_win": bool(ex) and c["agent2"]["examples_strong"] > c["agent2"]["near_misses_strong"],
            "steer_shows_concept": any(s.get("matches_target") for s in c["steer"]) if c["steer"] else None,
        }
        checks["concept_unit"] = bool(checks["test_pairs_pass"] and checks["explanation_matches"] and checks["examples_win"])
        checks["concept_unit_lenient"] = bool(checks["test_pairs_pass"] and (checks["explanation_matches"] or checks["explanation_partial"]) and checks["examples_win"])
        c["checks"] = checks
        c["verdict"] = notes.get(str(uid), {}).get("verdict")
        rec["chosen"] = c
        memes.append(rec)
    n = len(memes)
    chosen = [m["chosen"] for m in memes if m["chosen"]]
    summary = {
        "n_memes": n, "memes_with_a_dev_candidate": len(chosen),
        "chosen_passes_test_pairs": sum(c["checks"]["test_pairs_pass"] for c in chosen),
        "paper_unit_passes_test_pairs": sum(m["paper_unit"]["test_wins"] >= m["paper_unit"]["test_n"] - 1 for m in memes),
        "chosen_specific": sum(c["checks"]["specific"] for c in chosen),
        "chosen_same_as_paper": sum(c["same_as_paper"] for c in chosen),
        "explanation_matches_label": sum(bool(c["checks"]["explanation_matches"]) for c in chosen),
        "explanation_partial": sum(bool(c["checks"]["explanation_partial"]) for c in chosen),
        "explanation_judge": dict(__import__("collections").Counter(c["explanation"].get("judge_match") for c in chosen)),
        "explainer_conceptual": sum(c["explanation"].get("lexical_or_conceptual") == "conceptual" for c in chosen),
        "blind_beats_max": sum(c["checks"]["blind_beats_max"] for c in chosen),
        "blind_in_top8": sum(c["checks"]["blind_in_top8"] for c in chosen),
        "examples_win": sum(c["checks"]["examples_win"] for c in chosen),
        "steer_shows_concept": sum(bool(c["checks"]["steer_shows_concept"]) for c in chosen),
        "concept_units": sum(c["checks"]["concept_unit"] for c in chosen),
        "concept_units_lenient": sum(c["checks"]["concept_unit_lenient"] for c in chosen),
        "mean_presence_new": round(sum(c["presence_vs_paper"]["new_n_batman"] for c in chosen) / max(1, len(chosen)), 1),
        "mean_presence_paper": round(sum(c["presence_vs_paper"]["paper_n_batman"] for c in chosen) / max(1, len(chosen)), 1),
        "mean_batman_above_median_new": round(sum(c["presence_vs_paper"]["new_batman_above_median"] or 0 for c in chosen) / max(1, len(chosen)), 1),
        "mean_batman_above_median_paper": round(sum(c["presence_vs_paper"]["paper_batman_above_median"] or 0 for c in chosen) / max(1, len(chosen)), 1),
        "mean_test_wins_chosen": round(sum(c["test_wins"] for c in chosen) / max(1, len(chosen)), 2),
        "mean_test_wins_paper": round(sum(m["paper_unit"]["test_wins"] for m in memes) / n, 2),
        "agent2_examples_strong_share": round(sum(c["agent2"]["examples_strong"] for c in chosen) / max(1, sum(len(c["agent2"]["examples"]) for c in chosen)), 3),
        "agent2_near_misses_strong_share": round(sum(c["agent2"]["near_misses_strong"] for c in chosen) / max(1, sum(len(c["agent2"]["near_misses"]) for c in chosen)), 3),
    }
    return {"model": "qwen3-8b", "summary": summary, "summary_text": notes.get("_summary"), "memes": memes}


def render(out):
    s = out["summary"]
    L = ["# Concept-first mapping of the 31 named memes to Qwen3-8B neurons\n",
         "The paper mapped memes to neurons label-first and the audit found the labels mostly lexical. This experiment goes the way "
         "Cammarata's essay works: search every gate-open unit for one that separates name-free minimal pairs of each concept (the paper's "
         "revenge-neuron method, generalised), score it on held-out pairs, then explain the found neuron from its corpus firings alone, let a "
         "second agent write a blind maximum, five examples and five near misses from that explanation, and steer the channel during generation. "
         "Produced by `cammarata_mapping.py` and `cammarata_mapping_report.py`; data in `cammarata-mapping.json`. All activations are gate-open and "
         "pole-signed; probe sentences are read standalone with the first token excluded, corpus sentences in the context of their work.\n",
         "## Summary\n"]
    if out.get("summary_text"):
        L.append(out["summary_text"] + "\n")
    L += ["| Count | of 31 |", "| --- | --- |",
          f"| Memes with at least one unit passing the dev pairs (9 of 10 wins, mean activation ≥ 0.5) | {s['memes_with_a_dev_candidate']} |",
          f"| Chosen unit passes the held-out test pairs (at least 5 of 6) | {s['chosen_passes_test_pairs']} |",
          f"| The paper's unit passes the same test pairs | {s['paper_unit_passes_test_pairs']} |",
          f"| Chosen unit is specific (test positives above its p99 on the other memes' sentences) | {s['chosen_specific']} |",
          f"| Chosen unit is the paper's unit | {s['chosen_same_as_paper']} |",
          f"| Blind explainer rates the unit conceptual (not lexical or mixed) | {s['explainer_conceptual']} |",
          f"| Neuron-first explanation (Agent 1, blind to the label) describes the concept directly / a defining part of it | {s['explanation_matches_label']} / {s['explanation_partial']} |",
          f"| Agent 2's blind maximum beats the unit's corpus maximum / would make its top 8 | {s['blind_beats_max']} / {s['blind_in_top8']} |",
          f"| Agent 2's examples beat its near misses (explanation wins) | {s['examples_win']} |",
          f"| Steering the channel makes the model write the concept | {s['steer_shows_concept']} |",
          f"| **Concept units** (test pairs pass, explanation direct, Agent 2's examples beat its near misses) | **{s['concept_units']}** |",
          f"| Concept units on the lenient bar (explanation direct or partial) | {s['concept_units_lenient']} |",
          f"| Mean test wins of 6: chosen unit / paper's unit | {s['mean_test_wins_chosen']} / {s['mean_test_wins_paper']} |",
          f"| Mean Batman works (of 32) present under the paper's rule: new unit / paper's unit | {s['mean_presence_new']} / {s['mean_presence_paper']} |",
          f"| Mean Batman works scoring above the median ordinary film: new unit / paper's unit | {s['mean_batman_above_median_new']} / {s['mean_batman_above_median_paper']} |",
          f"| Agent 2 examples firing strongly / near misses firing strongly | {s['agent2_examples_strong_share']:.0%} / {s['agent2_near_misses_strong_share']:.0%} |", ""]
    L += ["## Overview\n",
          "| # | Meme | Paper's unit | dev / test | New unit | Layer | dev / test | Other-memes p99 | Explanation (Agent 1) | Match | Blind max / corpus max | Ex / NM strong | Steer | Presence (new / paper) | Concept unit |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i, m in enumerate(out["memes"], 1):
        p, c = m["paper_unit"], m["chosen"]
        if not c:
            L.append(f"| {i} | {esc(m['label'])} | {p['name']} | {p['dev_wins']}/10 · {p['test_wins']}/{p['test_n']} | none | | | | | | | | | | no |"); continue
        a2, ck = c["agent2"], c["checks"]
        bm = f"{a2['blind_max']['act']:.2f} / {c['corpus']['corpus_max']:.2f}" if a2["blind_max"] else "–"
        st = "–" if ck["steer_shows_concept"] is None else ("yes" if ck["steer_shows_concept"] else "no")
        L.append(f"| {i} | {esc(m['label'])} | {p['name']} | {p['dev_wins']}/10 · {p['test_wins']}/{p['test_n']} | {c['name']}{' (same)' if c['same_as_paper'] else ''} | {c['layer']} | "
                 f"{c['dev_wins']}/10 · {c['test_wins']}/{c['test_n']} | {c['other_memes_p99']:.2f} vs {c['test_mean_pos']:.2f} | {esc(trim(c['explanation'].get('explanation', ''), 90))} | "
                 f"{c['explanation'].get('judge_match', '–')} | {bm} | {a2['examples_strong']}/{len(a2['examples'])} · {a2['near_misses_strong']}/{len(a2['near_misses'])} | {st} | "
                 f"{c['presence_vs_paper']['new_n_batman']} / {c['presence_vs_paper']['paper_n_batman']} | {'**yes**' if ck['concept_unit'] else 'no'} |")
    L.append("")
    L.append("## Per-meme detail\n")
    for i, m in enumerate(out["memes"], 1):
        p, c = m["paper_unit"], m["chosen"]
        L.append(f"### {i}. {m['label']}\n")
        pc = p.get("corpus") or {}
        L.append(f"**Paper's unit {p['name']}** (layer {pc.get('layer')}): dev {p['dev_wins']}/10, test {p['test_wins']}/{p['test_n']} (AUC {p['test_auc']}); "
                 f"gate-open corpus max {pc.get('corpus_max')}, present in {pc.get('presence', {}).get('n_batman')} of 32 Batman works under the paper's rule on gate-open activations "
                 f"({m['paper_unit'].get('corpus', {}).get('presence', {}).get('n_baseline')} ordinary films). Top-50 peak tokens: {esc(', '.join(f'{t!r} ×{n}' for t, n in pc.get('peak_tokens_top50', []) if t is not None))}.\n")
        if not c:
            L.append("No unit passed the dev pairs.\n"); continue
        cu, a2, ck = c["corpus"], c["agent2"], c["checks"]
        L.append(f"**New unit {c['name']}** (layer {c['layer']}{', the same unit' if c['same_as_paper'] else ''}): {m['n_units_passing_dev']} units passed the dev bar; this one scored best. "
                 f"Dev {c['dev_wins']}/10 (mean pos {c['dev_mean_pos']}, neg {c['dev_mean_neg']}); test {c['test_wins']}/{c['test_n']} (AUC {c['test_auc']}; mean pos {c['test_mean_pos']}, neg {c['test_mean_neg']}); "
                 f"p99 on the other memes' sentences {c['other_memes_p99']} (max {c['other_memes_max']}). Corpus max {cu['corpus_max']}, p99 {cu['corpus_p99']}; "
                 f"present in {cu['presence']['n_batman']} of 32 Batman works, {cu['presence']['n_cousin']} of 13 relatives, {cu['presence']['n_baseline']} of 69 films.\n")
        L.append(f"- Presence against the paper's unit: both {len(c['presence_vs_paper']['both'])}, only new {len(c['presence_vs_paper']['only_new'])} ({esc(', '.join(c['presence_vs_paper']['only_new'][:8]))}), only paper {len(c['presence_vs_paper']['only_paper'])} ({esc(', '.join(c['presence_vs_paper']['only_paper'][:8]))}). "
                 f"Batman works scoring above the median ordinary film: {c['presence_vs_paper']['new_batman_above_median']} of 32 for the new unit, {c['presence_vs_paper']['paper_batman_above_median']} for the paper's.")
        L.append(f"- Top-50 peak tokens: {esc(', '.join(f'{t!r} ×{n}' for t, n in cu['peak_tokens_top50'] if t is not None))}.")
        L.append(f"- Agent 1 (blind to the label, rates it {c['explanation'].get('lexical_or_conceptual', '–')}, {c['explanation'].get('confidence', '–')} confidence) explains it as: *{esc(c['explanation'].get('explanation', '(none)'))}*")
        L.append(f"- Judge: **{c['explanation'].get('judge_match', '–')}**. {esc(c['explanation'].get('judge_note', ''))} Label the explanation supports: *{esc(c['explanation'].get('judge_better_label', ''))}*")
        if a2["blind_max"]:
            L.append(f"- Agent 2 blind maximum: {a2['blind_max']['act']:.2f} against corpus max {cu['corpus_max']:.2f} (weakest of the top 8: {cu['corpus_min_top8']:.2f}; peak {a2['blind_max']['peak_token']!r}): {esc(a2['blind_max']['text'])}")
            L.append(f"- Agent 2 examples firing strongly (≥ 50% of max): {a2['examples_strong']} of {len(a2['examples'])}; near misses: {a2['near_misses_strong']} of {len(a2['near_misses'])}.")
        if c["steer"]:
            L.append("- Steering: " + "; ".join(f"α={s['alpha']}: {esc(trim(s.get('judge', ''), 80))}{' (concept)' if s.get('matches_target') else ''}" for s in c["steer"]))
        if c.get("verdict"):
            L.append(f"- **Verdict:** {esc(c['verdict'])}")
        L.append("\nStrongest corpus sentences of the new unit (gate-open, peak token):\n")
        for t in cu["top8"]:
            L.append(f"- {t['act']:.2f} {t['peak_token']!r} · *{t['work']}*: {trim(t['text'])}")
        L.append(f"\nIts negative pole {cu['negative_pole']['name']} (max {cu['negative_pole']['max']:.2f}):\n")
        for t in cu["negative_pole"]["top8"][:5]:
            L.append(f"- {t['act']:.2f} {t['peak_token']!r} · *{t['work']}*: {trim(t['text'])}")
        if a2["examples"]:
            L.append("\nAgent 2's examples and near misses (activation, share of corpus max):\n")
            for r in a2["examples"]:
                L.append(f"- example {r['act']:.2f} ({r['ratio_to_max']:.0%}) {r['peak_token']!r}: {trim(r['text'])}")
            for r in a2["near_misses"]:
                L.append(f"- near miss {r['act']:.2f} ({r['ratio_to_max']:.0%}) {r['peak_token']!r}: {trim(r['text'])}")
        if c["steer"]:
            L.append("\nSteered generations:\n")
            for s in c["steer"]:
                L.append(f"- α={s['alpha']} · {esc(s['prompt'])} → {esc(trim(s['text'], 200))}")
        L.append("")
    return "\n".join(L) + "\n"


def write_report(work):
    out = build(work)
    json.dump(out, open(STUDY / "cammarata-mapping.json", "w"), indent=1, ensure_ascii=False)
    (STUDY / "cammarata-mapping.md").write_text(render(out))
    print(json.dumps(out["summary"], indent=1))
