"""A second model family as judge (PLAN.md, Section 10): Qwen3-32B repeats the kind and steering judgments as
multiple choice, and its agreement with the Claude judge is reported.

    cd study && ../.venv/bin/python v2/analysis/second_judge.py jobs <model-tag>     job files for gpu/ablate.py (run on qwen3-32b)
    cd study && ../.venv/bin/python v2/analysis/second_judge.py score <model-tag>

The untouched model is given each item and the options, numbered; its choice is the option whose number has the
highest probability as the first token of its reply. <model-tag> is the model whose units are being judged; the
judge is always Qwen3-32B. Reads v2/acts/<tag>/judge2_{kind,detects,steer}.json; writes results/v2/second_judge_<tag>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

import common as C
from explain_agents import KINDS
from steering import block

DETECTS = ["name or word", "idea"]
KIND_HELP = {"character or name": "a particular person, character or other proper name", "place": "a particular place, real or fictional",
             "object": "a particular thing or class of physical things", "word or token": "a particular word, spelling, word-piece or symbol, whatever it means",
             "story pattern": "something that happens in a story: an event, a situation, a relationship, a motive, a turn of the plot",
             "genre or setting": "the kind of story or world it is", "style or format": "how the text is written or laid out",
             "grammar": "a grammatical construction or position in a sentence", "other": "none of these"}


def step_jobs(tag):
    out = Path("v2/cache/jobs")
    out.mkdir(parents=True, exist_ok=True)
    ex = {int(u): e for u, e in json.load(open(C.RESULTS / tag / "explanations.json"))["units"].items()}
    desc = lambda e: f"A unit inside a language model has been described like this.\nTitle: {e['title']}\nExplanation: {e['explanation']}"      # noqa: E731
    q_kind = ("What kind of thing does this unit detect?\n" + "\n".join(f"{i + 1}. {k}: {KIND_HELP[k]}" for i, k in enumerate(KINDS)) +
              "\nReply with the number only.")
    q_det = ("Does this unit react to (1) a particular name, word or string, so that text conveying the same thing without that name or word would not fire it, "
             "or (2) an idea that can be put in other words? Reply with the number only.")
    json.dump({"question": q_kind, "reply_tokens": 2, "items": [{"id": str(u), "text": desc(e), "answers": [str(i + 1) for i in range(len(KINDS))]} for u, e in ex.items()]},
              open(out / f"judge2_kind_{tag}.json", "w"), ensure_ascii=False)
    json.dump({"question": q_det, "reply_tokens": 2, "items": [{"id": str(u), "text": desc(e), "answers": ["1", "2"]} for u, e in ex.items()]},
              open(out / f"judge2_detects_{tag}.json", "w"), ensure_ascii=False)
    n = 0
    path = C.RESULTS / tag / "steering_items.json"
    if path.exists():
        items = json.load(open(path))["items"]
        q = ("A language model continued three story openers twice: normally, and while one internal unit was forced to be strongly active. Which candidate best "
             "describes what was pushed into the forced continuations? Reply with the number only, or 0 if none of them fits.")
        json.dump({"question": q, "reply_tokens": 2, "items": [{"id": it["id"], "text": block(it), "answers": ["1", "2", "3", "4", "5", "0"]} for it in items]},
                  open(out / f"judge2_steer_{tag}.json", "w"), ensure_ascii=False)
        n = len(items)
    print(f"{tag}: {len(ex)} kind and detects items, {n} steering items -> {out}/judge2_*_{tag}.json")


def choices(path, options):
    return {it["id"]: options[int(np.argmax(it["untouched"]["logp_each"]))] for it in json.load(open(path))["items"]}


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    po = float(np.mean([x == y for x, y in zip(a, b)]))
    pe = sum((a.count(l) / len(a)) * (b.count(l) / len(b)) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(first, second):
    ids = sorted(set(first) & set(second))
    a, b = [first[i] for i in ids], [second[i] for i in ids]
    return {"n": len(ids), "agree": float(np.mean([x == y for x, y in zip(a, b)])) if ids else None, "kappa": kappa(a, b) if ids else None}


def step_score(tag):
    d = C.ACTS / tag
    kinds = {u: k for u, k in json.load(open(C.RESULTS / tag / "kinds.json"))["units"].items()}
    k2, d2 = choices(d / "judge2_kind.json", KINDS), choices(d / "judge2_detects.json", DETECTS)
    out = {"judged_model": tag, "second_judge": "qwen3-32b",
           "kind": agreement({u: v["kind"] for u, v in kinds.items()}, k2), "detects": agreement({u: v["detects"] for u, v in kinds.items()}, d2),
           "kind_by_second_judge": {k: int(sum(v == k for v in k2.values())) for k in KINDS},
           "second_judge_choices": {u: {"kind": k2.get(u), "detects": d2.get(u)} for u in kinds}}
    if (d / "judge2_steer.json").exists():
        items = {it["id"]: it for it in json.load(open(C.RESULTS / tag / "steering_items.json"))["items"]}
        first = {i: v["choice"] for i, v in json.load(open(C.RESULTS / f"steering_{tag}.json"))["verdicts"].items() if "choice" in v}
        second = {i: ("none" if c == "0" else c) for i, c in choices(d / "judge2_steer.json", ["1", "2", "3", "4", "5", "0"]).items()}
        real = [i for i in second if not items[i]["sham"]]
        out["steering"] = {"agreement_with_first_judge": agreement(first, second),
                           "second_judge_accuracy": float(np.mean([second[i] == items[i]["answer"] for i in real])) if real else None, "n": len(real),
                           "second_judge_none_on_sham": int(sum(second[i] == "none" for i in second if items[i]["sham"])),
                           "n_sham": int(sum(items[i]["sham"] for i in second)),
                           "second_judge_correct": int(sum(second[i] == items[i]["answer"] for i in real)),
                           "second_judge_choices": {i: second[i] for i in sorted(second)}}
    C.save(out, f"second_judge_{tag}.json")
    print(tag, "second judge:", {k: out[k] for k in ("kind", "detects")}, out.get("steering", {}).get("agreement_with_first_judge"))


if __name__ == "__main__":
    {"jobs": step_jobs, "score": step_score}[sys.argv[1]](sys.argv[2])
