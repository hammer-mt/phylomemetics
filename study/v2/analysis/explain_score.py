"""Score the explanation tests (PLAN.md, Section 10): can the explanations lose?

    cd study && ../.venv/bin/python v2/analysis/explain_score.py texts <model-tag>
    cd study && ../.venv/bin/python v2/analysis/explain_score.py score <model-tag>

texts   writes v2/cache/explain/<tag>/texts.jsonl and units.npy: every tester text (A, B, C, D, M) of every explained
        unit, and for the context check each unit's strongest study sentence and three strongest reference examples,
        to be read alone. gpu/score_texts.py then scores every text on every explained unit.
score   reads v2/acts/<tag>/tests_scores.npz and writes results/v2/<tag>/explanation_tests.json.

A text's response is the unit's strongest activation in it as a share of the unit's scale; 30% or more is strong.
A unit is "not testable out of context" when its strongest study sentence and its three strongest reference examples
(the last 420 characters up to the peak), each read alone, all stay below 30%: such units are counted apart and left
out of the wins, ties and losses and of the idea-bound count.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

import common as C
import textcheck as T

import experiments as X

GROUPS = "ABCD"
IDEA_KINDS = ("story pattern", "genre or setting")


def load(tag):
    d = C.RESULTS / tag
    g = lambda n: {int(u): v for u, v in json.load(open(d / n))["units"].items()}      # noqa: E731
    return json.load(open(d / "explain_units.json"))["units"], g("explanations.json"), g("tests.json"), g("keyword_terms.json"), g("kinds.json")


def study_top(tag, unit):
    """The unit's strongest original-text study sentence: (share in context, text)."""
    d = C.ACTS / tag
    units = np.load(d / "sent_units.npy")
    col = np.load(d / "sent_orig.u8.npy", mmap_mode="r")[:, int(np.flatnonzero(units == unit)[0])]
    i = int(np.argmax(col))
    sents = [s for w in X.works for s in w["sentences"]]
    return float(col[i]) / C.STEP, sents[i]


def step_texts(tag):
    chosen, ex, tests, _, _ = load(tag)
    tops = json.load(open(C.ACTS / tag / "top_examples.json"))
    scale = C.scale(tag)
    out, ctx = [], {}
    for c in chosen:
        u = c["unit"]
        if u not in tests:
            continue
        for g in GROUPS:
            out += [{"id": f"{u}:{g}{i + 1}", "text": t} for i, t in enumerate(tests[u][g])]
        out.append({"id": f"{u}:M", "text": tests[u]["M"]})
        share, sent = study_top(tag, u)
        out.append({"id": f"{u}:ctx_study", "text": sent})
        ctx[str(u)] = {"study": share, "ref": []}
        for j, e in enumerate(tops[str(u)]["examples"][:3]):
            out.append({"id": f"{u}:ctx_ref{j + 1}", "text": (e["before"][-420:] + e["token"]).strip()})
            ctx[str(u)]["ref"].append(e["activation"] / float(scale[u]) if scale[u] > 0 else 0.0)
    d = Path("v2/cache/explain") / tag
    with open(d / "texts.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.save(d / "units.npy", np.array([c["unit"] for c in chosen if c["unit"] in tests], dtype=np.int64))
    json.dump(ctx, open(C.RESULTS / tag / "context_in_document.json", "w"))
    print(f"{tag}: {len(out)} texts for {len(ctx)} units -> {d}/texts.jsonl")


def auc(pos, neg):
    p, q = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    return float(((p > q) + 0.5 * (p == q)).mean())


def boot_share(x, rng, n=4000):
    x = np.asarray(x, float)
    return [float(v) for v in np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)], [2.5, 97.5])] if len(x) else [None, None]


def tfidf_scores(tag, units, tests, tops):
    """Cosine (word weights from the background plots) between each test text and the centroid of the unit's 16
    reference examples, each cut at its peak token as the unit saw it."""
    vec = X.tfidf().fit([d["text"] for d in C.background_docs()])
    out = {}
    for u in units:
        E = vec.transform([(e["before"] + e["token"]) for e in tops[str(u)]["examples"][:16]])
        c = np.asarray(E.mean(0)).ravel()
        c /= max(np.linalg.norm(c), 1e-12)
        out[u] = {g: (vec.transform(tests[u][g]) @ c).tolist() for g in GROUPS}
    return out


def step_score(tag, theta=C.THETA):
    rng = np.random.default_rng(C.SEED)
    chosen, ex, tests, terms, kinds = load(tag)
    z = np.load(C.ACTS / tag / "tests_scores.npz")
    units = [int(u) for u in z["units"]]
    scale = C.scale(tag)[units]
    share = z["peak"] / np.where(scale > 0, scale, np.inf)[None, :]                 # [texts, units]
    share_rest = z["rest"] / np.where(scale > 0, scale, np.inf)[None, :]            # the same with each text's first word left out
    row = {str(i): k for k, i in enumerate(z["ids"])}
    col = {u: k for k, u in enumerate(units)}
    tops = json.load(open(C.ACTS / tag / "top_examples.json"))
    in_doc = json.load(open(C.RESULTS / tag / "context_in_document.json"))
    lex = tfidf_scores(tag, units, tests, tops)
    L, D = C.geometry(tag)
    meta = {c["unit"]: c for c in chosen}
    per_unit, own = [], {}
    for u in units:
        k = col[u]
        s = {g: np.array([share[row[f"{u}:{g}{i + 1}"], k] for i in range(5)]) for g in GROUPS}
        own[u] = s
        strong = {g: (s[g] >= theta) for g in GROUPS}
        rest = {g: np.array([share_rest[row[f"{u}:{g}{i + 1}"], k] for i in range(5)]) >= theta for g in GROUPS}
        key = T.key_stems(ex[u]["key_words"])
        reuse = np.array([bool(T.uses(t, key)) for t in tests[u]["C"]])
        b_clean = np.array([not T.uses(t, key) for t in tests[u]["B"]])
        a_n, c_n = int(strong["A"].sum()), int(strong["C"].sum())
        hits = {g: [T.phrase_hits(t, terms.get(u, [])) for t in tests[u][g]] for g in GROUPS}
        m = float(share[row[f"{u}:M"], k])
        alone = {"study": float(share[row[f"{u}:ctx_study"], k]), "ref": [float(share[row[f"{u}:ctx_ref{j + 1}"], k]) for j in range(len(in_doc[str(u)]["ref"]))]}
        ratios = ([alone["study"] / in_doc[str(u)]["study"]] if in_doc[str(u)]["study"] > 0 else []) + \
                 [a / b for a, b in zip(alone["ref"], in_doc[str(u)]["ref"]) if b > 0]
        kind = kinds.get(u, {"kind": "other", "detects": "name or word"})
        testable = bool(max([alone["study"]] + alone["ref"]) >= theta)
        per_unit.append({
            "unit": u, "name": meta[u]["name"], "set": meta[u]["set"], "layer": meta[u]["layer"], "depth_quarter": int(meta[u]["layer"] * 4 // L) + 1,
            "title": ex[u]["title"], "kind": kind["kind"], "detects": kind["detects"], "testable_out_of_context": testable,
            "alone": {"study_sentence": round(alone["study"], 3), "reference_examples": [round(x, 3) for x in alone["ref"]]},
            "is_idea": bool(kind["kind"] in IDEA_KINDS or kind["detects"] == "idea"), "one_idea": ex[u]["one_idea"], "understood": ex[u]["understood"],
            "shares": {g: [round(float(x), 3) for x in s[g]] for g in GROUPS}, "strong": {g: int(strong[g].sum()) for g in GROUPS},
            "strong_first_word_left_out": {g: int(rest[g].sum()) for g in GROUPS},
            "outcome_first_word_left_out": "win" if rest["A"].sum() > rest["C"].sum() else "tie" if rest["A"].sum() == rest["C"].sum() else "loss",
            "outcome": "win" if a_n > c_n else "tie" if a_n == c_n else "loss",
            "blind_max_share": round(m, 3), "blind_max_reaches_corpus_max": bool(m >= 1.0), "blind_max_strong": bool(m >= theta),
            "B_texts_clean": int(b_clean.sum()), "C_texts_reusing_a_key_word": int(reuse.sum()),
            "strong_among_C_reusing_a_key_word": int((strong["C"] & reuse).sum()),
            "idea_bound": bool(strong["B"].sum() >= 3 and (strong["C"] & reuse).sum() <= 1),
            "analyst_hits": hits, "analyst_hit_any": {g: int(sum(h > 0 for h in hits[g])) for g in GROUPS},
            "auc_B_vs_C": {"neuron": auc(s["B"], s["C"]), "analyst_keywords": auc(hits["B"], hits["C"]), "tfidf_to_examples": auc(lex[u]["B"], lex[u]["C"])},
            "auc_A_vs_C": {"neuron": auc(s["A"], s["C"]), "analyst_keywords": auc(hits["A"], hits["C"]), "tfidf_to_examples": auc(lex[u]["A"], lex[u]["C"])},
            "alone_over_in_context": round(float(np.median(ratios)), 3) if ratios else None})

    # mismatched null: every unit against every other unit's texts
    n = len(units)
    S = {g: np.stack([np.stack([share[row[f"{v}:{g}{i + 1}"]] for i in range(5)]) for v in units]) for g in GROUPS}   # [text-owner, 5, unit]
    strongS = {g: S[g] >= theta for g in GROUPS}
    off = ~np.eye(n, dtype=bool)
    reuse_all = np.stack([[bool(T.uses(t, T.key_stems(ex[v]["key_words"]))) for t in tests[v]["C"]] for v in units])          # [owner, 5]
    b_n = strongS["B"].sum(1)                                                              # [owner, unit]
    c_reuse_n = (strongS["C"] & reuse_all[:, :, None]).sum(1)
    a_n, c_n = strongS["A"].sum(1), strongS["C"].sum(1)
    mism = {"strong_rate_on_other_units_A_texts": float(strongS["A"].mean(1)[off].mean()), "strong_rate_on_other_units_B_texts": float(strongS["B"].mean(1)[off].mean()),
            "idea_bound_rate_on_other_units_texts": float(((b_n >= 3) & (c_reuse_n <= 1))[off].mean()),
            "win_rate_on_other_units_texts": float((a_n > c_n)[off].mean()), "tie_rate_on_other_units_texts": float((a_n == c_n)[off].mean())}

    def summary(rows):
        n_all = len(rows)
        rows = [r for r in rows if r["testable_out_of_context"]]
        if not rows:
            return {"n": 0, "n_with_untestable": n_all, "not_testable_out_of_context": n_all}
        f = lambda key: float(np.mean([r["strong"][key] / 5 for r in rows]))      # noqa: E731
        w = [r["outcome"] for r in rows]
        idea = [r["idea_bound"] for r in rows]
        return {"n": len(rows), "n_with_untestable": n_all, "not_testable_out_of_context": n_all - len(rows),
                "strong_rate": {g: f(g) for g in GROUPS},
                "strong_rate_first_word_left_out": {g: float(np.mean([r["strong_first_word_left_out"][g] / 5 for r in rows])) for g in GROUPS},
                "won_first_word_left_out": [r["outcome_first_word_left_out"] for r in rows].count("win"), "won": w.count("win"), "tied": w.count("tie"), "lost": w.count("loss"),
                "win_share": w.count("win") / len(w), "win_share_ci95": boot_share([x == "win" for x in w], rng),
                "blind_max_reaches_corpus_max": float(np.mean([r["blind_max_reaches_corpus_max"] for r in rows])),
                "blind_max_strong": float(np.mean([r["blind_max_strong"] for r in rows])),
                "blind_max_share_median": float(np.median([r["blind_max_share"] for r in rows])),
                "idea_bound": int(sum(idea)), "idea_bound_share": float(np.mean(idea)),
                "one_idea_score_mean": float(np.mean([r["one_idea"] for r in rows])),
                "alone_over_in_context_median": float(np.median([r["alone_over_in_context"] for r in rows if r["alone_over_in_context"] is not None] or [np.nan])),
                "auc_B_vs_C_mean": {k: float(np.mean([r["auc_B_vs_C"][k] for r in rows])) for k in ("neuron", "analyst_keywords", "tfidf_to_examples")},
                "auc_A_vs_C_mean": {k: float(np.mean([r["auc_A_vs_C"][k] for r in rows])) for k in ("neuron", "analyst_keywords", "tfidf_to_examples")}}

    def versus(rows, key, other):
        a, b = np.array([r[key]["neuron"] for r in rows]), np.array([r[key][other] for r in rows])
        if len(rows) < 6 or np.all(a == b):
            return {"n": len(rows), "p_two_sided": None}
        return {"n": len(rows), "neuron_mean": float(a.mean()), "words_mean": float(b.mean()), "neuron_better": int((a > b).sum()), "words_better": int((b > a).sum()),
                "p_two_sided": float(wilcoxon(a, b, zero_method="wilcox").pvalue)}

    everyone = [dict(r, testable_out_of_context=True) for r in per_unit]
    core = [r for r in per_unit if r["set"] == "core"]
    name_like = [r["kind"] in ("character or name", "place", "word or token") for r in core]
    out = {"model": tag, "theta": theta, "n_units": len(per_unit), "all": summary(per_unit), "all_with_untestable_units_left_in": summary(everyone),
           "by_kind": {k: summary([r for r in per_unit if r["kind"] == k]) for k in sorted({r["kind"] for r in per_unit})},
           "by_detects": {k: summary([r for r in per_unit if r["detects"] == k]) for k in ("name or word", "idea")},
           "idea_units": summary([r for r in per_unit if r["is_idea"]]), "name_units": summary([r for r in per_unit if not r["is_idea"]]),
           "by_set": {k: summary([r for r in per_unit if r["set"] == k]) for k in sorted({r["set"] for r in per_unit})},
           "by_depth_quarter": {str(q): summary([r for r in per_unit if r["depth_quarter"] == q]) for q in (1, 2, 3, 4)},
           "mismatched_null": mism,
           "neurons_against_words_B_vs_C": {k: versus(per_unit, "auc_B_vs_C", k) for k in ("analyst_keywords", "tfidf_to_examples")},
           "neurons_against_words_B_vs_C_idea_units": {k: versus([r for r in per_unit if r["is_idea"]], "auc_B_vs_C", k) for k in ("analyst_keywords", "tfidf_to_examples")},
           "core_representatives_name_place_or_word": {"n": len(core), "count": int(sum(name_like)), "share": float(np.mean(name_like)) if core else None,
                                                       "ci95": boot_share(name_like, rng), "v1_by_hand": "27 of 33"},
           "essay_benchmark": {"new_examples": 0.76, "near_misses": 0.11, "distant": 0.01, "won": 164, "tied": 15, "lost": 4},
           "units": per_unit}
    C.save(out, f"{tag}/explanation_tests.json")
    a = out["all"]
    print(f"{tag}: {a['n']} testable units of {a['n_with_untestable']} | strong A {a['strong_rate']['A']:.2f} B {a['strong_rate']['B']:.2f} C {a['strong_rate']['C']:.2f} D {a['strong_rate']['D']:.2f} "
          f"| won {a['won']} tied {a['tied']} lost {a['lost']} | idea-bound {a['idea_bound']} | mismatched A {mism['strong_rate_on_other_units_A_texts']:.3f}")


if __name__ == "__main__":
    {"texts": step_texts, "score": step_score}[sys.argv[1]](sys.argv[2])
