"""Hand the explanation work to AI agents and collect what they return (PLAN.md, Section 10).

Each agent gets one prompt file and writes one answer file. A prompt file is the committed prompt of its role
(v2/prompts/<role>.md) with the batch filled in, followed by where and how to write the answer. No agent sees
another role's material: the tester, the keyword analyst and the kind judge get titles and explanations only.

    cd study && ../.venv/bin/python v2/analysis/explain_agents.py <step> <model-tag>

    explainer   write the explainers' prompts (needs build_explain_inputs.py's unit files)
    collect-explainer   -> results/v2/<tag>/explanations.json
    others      write the testers', keyword analysts' and kind judges' prompts (needs explanations.json)
    collect-others      -> results/v2/<tag>/tests.json, keyword_terms.json, kinds.json, tests_check.json; and, for
                        tester texts that break the wording rules, a second round of prompts (step "collect-others"
                        again after they are answered)

Prompts and answers live in v2/cache/explain/<tag>/<role>/ (not committed); the collected files are committed.
"""
import json
import sys
from pathlib import Path

import numpy as np

import common as C
import textcheck as T

MODEL_NAMES = {"qwen3-32b": "Qwen3-32B", "qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama 3.1 8B", "llama3.3-70b": "Llama 3.3 70B"}
BATCH = {"explainer": 6, "tester": 6, "keyword_analyst": 20, "kind_judge": 30}
KINDS = ["character or name", "place", "object", "word or token", "story pattern", "genre or setting", "style or format", "grammar", "other"]
ANSWER = ("\n\n---\nHow to answer. Work from this file alone: do not open, search or run anything else. Write your answer as one JSON file at\n{out}\n"
          "It must be a JSON list with one object per {what}, with exactly the fields listed above under \"Return\". "
          "When the file is written, reply with the single word DONE.\n")


def root(tag, role):
    d = Path("v2/cache/explain") / tag / role
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def write_batches(tag, role, items, fill, what, round_=1):
    """items: list of per-unit blocks. Writes prompt files of BATCH[role] items each; returns their paths."""
    d = root(tag, role)
    template = Path(f"v2/prompts/{role}.md").read_text()
    paths = []
    rng = np.random.default_rng(C.SEED + round_)
    order = rng.permutation(len(items))                       # batches mix sets and layers
    n = BATCH[role]
    for k in range(0, len(items), n):
        name = f"r{round_}_batch_{k // n:03d}"
        body = fill(template, [items[i] for i in order[k:k + n]])
        (d / f"{name}.prompt.txt").write_text(body + ANSWER.format(out=d / f"{name}.out.json", what=what))
        paths.append(str(d / f"{name}.prompt.txt"))
    json.dump(paths, open(d / f"r{round_}_prompts.json", "w"))
    print(f"{tag} {role}: {len(items)} items in {len(paths)} prompt files -> {d}/r{round_}_prompts.json")
    return paths


def read_answers(tag, role):
    """Every object in every answer file of the role, later rounds last. Files that do not parse are reported."""
    out, bad = [], []
    for f in sorted(root(tag, role).glob("r*_batch_*.out.json")):
        try:
            data = json.load(open(f))
            out += [dict(x, _file=f.name) for x in (data if isinstance(data, list) else data.get("units") or data.get("items") or [])]
        except Exception as e:      # noqa: BLE001
            bad.append((f.name, repr(e)[:80]))
    missing = [p.name for p in sorted(root(tag, role).glob("r*_batch_*.prompt.txt")) if not Path(str(p).replace(".prompt.txt", ".out.json")).exists()]
    return out, bad, missing


def chosen(tag):
    return json.load(open(C.RESULTS / tag / "explain_units.json"))["units"]


# ------------------------------------------------------------------ explainer
def step_explainer(tag):
    blocks = [(c["unit"], Path(f"v2/cache/explain/{tag}/unit_{c['unit']}.txt").read_text()) for c in chosen(tag)]
    write_batches(tag, "explainer", blocks,
                  lambda t, b: t.replace("{model}", MODEL_NAMES.get(tag, tag)).replace("{units}", "\n\n==========\n\n".join(x[1] for x in b)), "unit")


def step_collect_explainer(tag):
    rows, bad, missing = read_answers(tag, "explainer")
    want = {c["unit"] for c in chosen(tag)}
    out = {}
    for r in rows:
        try:
            u = int(r["unit"])
            kw = [str(k).strip().lower() for k in r["key_words"] if str(k).strip()]
            assert u in want and len(kw) >= 1 and str(r["title"]).strip() and str(r["explanation"]).strip()
            out[u] = {"title": str(r["title"]).strip(), "explanation": str(r["explanation"]).strip(), "key_words": kw,
                      "one_idea": int(r["one_idea"]), "understood": bool(r.get("understood", True))}
        except Exception as e:      # noqa: BLE001
            bad.append((r.get("_file"), repr(e)[:80]))
    json.dump({"model": tag, "n": len(out), "units": {str(u): out[u] for u in sorted(out)}}, open(C.RESULTS / tag / "explanations.json", "w"),
              indent=1, ensure_ascii=False)
    print(f"{tag}: {len(out)} of {len(want)} units explained; unanswered prompts: {missing}; unusable: {bad}; still missing units: {sorted(want - set(out))[:20]}")


# ------------------------------------------------------------------ tester, keyword analyst, kind judge
def explanations(tag):
    return {int(u): v for u, v in json.load(open(C.RESULTS / tag / "explanations.json"))["units"].items()}


def item_text(u, e, with_keys):
    s = f"Unit {u}\nTitle: {e['title']}\nExplanation: {e['explanation']}"
    return s + (f"\nKey words: {', '.join(e['key_words'])}" if with_keys else "")


def step_others(tag):
    ex = explanations(tag)
    units = sorted(ex)
    join = lambda b: "\n\n".join(b)      # noqa: E731
    write_batches(tag, "tester", [item_text(u, ex[u], True) for u in units], lambda t, b: t.replace("{items}", join(b)), "unit")
    write_batches(tag, "keyword_analyst", [item_text(u, ex[u], False) for u in units], lambda t, b: t.replace("{items}", join(b)), "concept")
    write_batches(tag, "kind_judge", [item_text(u, ex[u], False) for u in units], lambda t, b: t.replace("{items}", join(b)), "item")


def check_tests(e, t):
    """The wording rules of PLAN.md Section 10: no B text uses a key-word stem; at least three C texts do."""
    key = T.key_stems(e["key_words"])
    b_bad = {i: sorted(T.uses(x, key)) for i, x in enumerate(t["B"]) if T.uses(x, key)}
    c_reuse = [bool(T.uses(x, key)) for x in t["C"]]
    return {"B_texts_using_a_key_word": b_bad, "C_texts_reusing_a_key_word": c_reuse, "ok": not b_bad and sum(c_reuse) >= 3}


def step_collect_others(tag):
    ex = explanations(tag)
    tests, terms, kinds, problems = {}, {}, {}, []
    rows, bad, missing = read_answers(tag, "tester")
    for r in rows:                                          # later rounds overwrite earlier ones
        try:
            u = int(r["unit"])
            t = {g: [str(x).strip() for x in r[g]] for g in "ABCD"}
            assert u in ex and all(len(t[g]) == 5 and all(t[g]) for g in "ABCD") and str(r["M"]).strip()
            t["M"] = str(r["M"]).strip()
            tests[u] = t
        except Exception as e:      # noqa: BLE001
            problems.append(("tester", r.get("_file"), r.get("unit"), repr(e)[:80]))
    checks = {u: check_tests(ex[u], t) for u, t in tests.items()}
    rows_k, bad_k, missing_k = read_answers(tag, "keyword_analyst")
    for r in rows_k:
        try:
            terms[int(r["unit"])] = [str(x).strip().lower() for x in r["terms"] if str(x).strip()]
        except Exception as e:      # noqa: BLE001
            problems.append(("keyword_analyst", r.get("_file"), r.get("unit"), repr(e)[:80]))
    rows_j, bad_j, missing_j = read_answers(tag, "kind_judge")
    for r in rows_j:
        try:
            assert r["kind"] in KINDS and r["detects"] in ("name or word", "idea")
            kinds[int(r["unit"])] = {"kind": r["kind"], "detects": r["detects"]}
        except Exception as e:      # noqa: BLE001
            problems.append(("kind_judge", r.get("_file"), r.get("unit"), repr(e)[:80]))
    d = C.RESULTS / tag
    dump = lambda o, n: json.dump({"model": tag, "n": len(o), "units": {str(u): o[u] for u in sorted(o)}}, open(d / n, "w"), indent=1, ensure_ascii=False)  # noqa: E731
    dump(tests, "tests.json"); dump(terms, "keyword_terms.json"); dump(kinds, "kinds.json"); dump(checks, "tests_check.json")
    failing = sorted(u for u, c in checks.items() if not c["ok"])
    rounds = len(list(root(tag, "tester").glob("r*_prompts.json")))
    print(f"{tag}: tests for {len(tests)} of {len(ex)} units, terms for {len(terms)}, kinds for {len(kinds)}")
    print(f"  unanswered: tester {missing}, analyst {missing_k}, judge {missing_j}; unreadable files: {bad + bad_k + bad_j}; problems: {problems[:10]}")
    print(f"  wording rules: {len(tests) - len(failing)} units pass, {len(failing)} fail after {rounds} round(s)")
    todo = failing + sorted(set(ex) - set(tests))
    if todo and rounds < 3:                                 # at most three rounds; what still fails is kept and reported
        def note(u):
            c = checks.get(u)
            if not c:
                return ""
            bits = []
            if c["B_texts_using_a_key_word"]:
                bits.append("In an earlier attempt these B texts used a key word or a form of one: " +
                            "; ".join(f"B{i + 1} used {', '.join(w)}" for i, w in c["B_texts_using_a_key_word"].items()) + ". Write all the texts afresh and keep every B text clear of the key words and their forms.")
            if sum(c["C_texts_reusing_a_key_word"]) < 3:
                bits.append("In an earlier attempt fewer than three C texts contained a key word. At least three must.")
            return "\nNote: " + " ".join(bits)
        write_batches(tag, "tester", [item_text(u, ex[u], True) + note(u) for u in todo],
                      lambda t, b: t.replace("{items}", "\n\n".join(b)), "unit", round_=rounds + 1)


STEPS = {"explainer": step_explainer, "collect-explainer": step_collect_explainer, "others": step_others, "collect-others": step_collect_others}

if __name__ == "__main__":
    STEPS[sys.argv[1]](sys.argv[2])
