"""The steering test of the labels (PLAN.md, Section 11).

    cd study && ../.venv/bin/python v2/analysis/steering.py gallery                 choose the 40 gallery neurons
    ... gpu/steer.py on them at 1, 2, 3 and 5 times the scale -> v2/acts/qwen3-32b/steer_gallery.json
    cd study && ../.venv/bin/python v2/analysis/steering.py gallery-items           the judge's prompts
    cd study && ../.venv/bin/python v2/analysis/steering.py gallery-score           the multiple with the highest accuracy
    ... gpu/steer.py on the explained units at that multiple -> v2/acts/<tag>/steer_units.json
    cd study && ../.venv/bin/python v2/analysis/steering.py items <model-tag>
    cd study && ../.venv/bin/python v2/analysis/steering.py score <model-tag>

Judge prompts and answers live in v2/cache/steer/ (not committed); the items, keys and verdicts are committed
under results/v2/.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import common as C
import textcheck as T
from explain_agents import ANSWER, wrapped

GALLERY = Path("v2/cache/gallery/data_neuron-gallery.json")        # the essay's public gallery (already-clean.goodfire.com)
TAG = "qwen3-32b"
MULTIPLES = [1, 2, 3, 5]
PER_PROMPT = 10


def judge_dir(name):
    d = Path("v2/cache/steer") / name
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def block(item):
    lines = [f"ITEM {item['id']}"]
    for k, (o, b, s) in enumerate(zip(item["openers"], item["normal"], item["forced"]), 1):
        lines += [f"Opener {k}: \"{o}\"", f"  normal: {b!r}", f"  forced: {s!r}"]
    lines += ["Candidates:"] + [f"  {i + 1}. {t}" for i, t in enumerate(item["candidates"])]
    return "\n".join(lines)


def write_prompts(name, items, seed):
    d = judge_dir(name)
    template = Path("v2/prompts/steering_judge.md").read_text()
    order = np.random.default_rng(seed).permutation(len(items))
    paths = []
    for k in range(0, len(items), PER_PROMPT):
        p = d / f"batch_{k // PER_PROMPT:03d}.prompt.txt"
        body = template.replace("{items}", "\n\n".join(block(items[i]) for i in order[k:k + PER_PROMPT]))
        p.write_text(wrapped(body + ANSWER.format(out=d / f"batch_{k // PER_PROMPT:03d}.out.json", what="item")))
        paths.append(str(p))
    json.dump(paths, open(d / "prompts.json", "w"))
    print(f"{name}: {len(items)} items in {len(paths)} prompt files -> {d}/prompts.json")


def read_verdicts(name):
    out = {}
    for f in sorted(judge_dir(name).glob("batch_*.out.json")):
        try:
            for r in json.load(open(f)):
                out[str(r["id"])] = {"choice": str(r["choice"]).strip().lower(), "reason": str(r.get("reason", ""))}
        except Exception as e:      # noqa: BLE001
            print("unreadable", f.name, repr(e)[:80])
    return out


def continuations(path):
    d = json.load(open(path))
    forced, normal = {}, {}
    for r in d["steered"]:
        forced.setdefault((r["unit"], float(r["multiple"])), {})[r["opener"]] = r["text"]
        normal.setdefault((r["unit"], float(r["multiple"])), {})[r["opener"]] = r.get("normal", d["baseline"][r["opener"]])
    return d["openers"], normal, forced


# ------------------------------------------------------------------ the gallery: choosing the multiple
def gallery():
    rng = np.random.default_rng(C.SEED)
    g = [x for x in json.load(open(GALLERY)) if x["k"] == "d" and x.get("m")]
    L, D = C.geometry(TAG)
    chosen = []
    for q in range(4):
        pool = [x for x in g if x["L"] * 4 // L == q]
        for i in sorted(rng.choice(len(pool), 10, replace=False)):
            x = pool[i]
            neg = max(x["m"], key=lambda e: abs(e[1]))[1] < 0
            chosen.append({"id": x["id"], "layer": x["L"], "neuron": x["N"], "pole": "-" if neg else "+", "unit": x["L"] * D + x["N"] + (L * D if neg else 0),
                           "title": x["t"], "depth_quarter": q + 1})
    out = Path("v2/cache/jobs")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "steer_gallery_units.npy", np.array([c["unit"] for c in chosen], dtype=np.int64))
    C.save({"model": TAG, "seed": C.SEED, "n": len(chosen), "neurons": chosen}, "steering_gallery_units.json")


def gallery_items():
    rng = np.random.default_rng(C.SEED + 1)
    neurons = json.load(open(C.RESULTS / "steering_gallery_units.json"))["neurons"]
    openers, base, forced = continuations(C.ACTS / TAG / "steer_gallery.json")
    items = []
    for i, n in enumerate(neurons):
        for m in MULTIPLES:
            others = [j for j in range(len(neurons)) if j != i]
            cand = [i] + [int(j) for j in rng.choice(others, 4, replace=False)]
            order = rng.permutation(5)
            items.append({"id": f"g{i:02d}m{m}", "neuron": n["id"], "multiple": m, "openers": openers,
                          "normal": [base[(n["unit"], float(m))][k] for k in range(len(openers))],
                          "forced": [forced[(n["unit"], float(m))][k] for k in range(len(openers))],
                          "candidates": [neurons[cand[j]]["title"] for j in order], "answer": str(int(np.flatnonzero(order == 0)[0]) + 1)})
    C.save({"items": items}, "steering_gallery_items.json")
    write_prompts("gallery", [{k: v for k, v in it.items() if k != "answer"} for it in items], C.SEED + 2)


def gallery_score():
    items = json.load(open(C.RESULTS / "steering_gallery_items.json"))["items"]
    v = read_verdicts("gallery")
    acc = {}
    for m in MULTIPLES:
        rows = [it for it in items if it["multiple"] == m]
        got = [v.get(it["id"], {}).get("choice") for it in rows]
        acc[str(m)] = {"n": len(rows), "judged": int(sum(x is not None for x in got)), "correct": int(sum(x == it["answer"] for x, it in zip(got, rows))),
                       "none": int(sum(x == "none" for x in got))}
        acc[str(m)]["accuracy"] = acc[str(m)]["correct"] / max(acc[str(m)]["n"], 1)
    best = max(MULTIPLES, key=lambda m: (acc[str(m)]["accuracy"], -m))            # ties go to the smaller multiple
    C.save({"model": TAG, "chance": 0.2, "by_multiple": acc, "chosen_multiple": best,
            "p_above_chance_at_chosen": float(binomtest(acc[str(best)]["correct"], acc[str(best)]["n"], 0.2, alternative="greater").pvalue),
            "verdicts": {it["id"]: dict(v.get(it["id"], {}), answer=it["answer"]) for it in items}}, "steering_gallery.json")
    print("gallery accuracy by multiple:", {m: round(a["accuracy"], 3) for m, a in acc.items()}, "-> multiple", best)


# ------------------------------------------------------------------ the study units
def study_items(tag):
    rng = np.random.default_rng(C.SEED + 3)
    multiple = json.load(open(C.RESULTS / "steering_gallery.json"))["chosen_multiple"]
    tests = json.load(open(C.RESULTS / tag / "explanation_tests.json"))["units"]
    fam = {}
    path = C.RESULTS / tag / "families.json.gz"
    if path.exists():
        import gzip
        for k, f in enumerate(json.load(gzip.open(path, "rt"))["families"]):
            for u in f["member_units"]:
                fam[u] = k
    openers, base, forced = continuations(C.ACTS / tag / "steer_units.json")
    by = {r["unit"]: r for r in tests}
    units = [r["unit"] for r in tests if (r["unit"], float(multiple)) in forced]

    def distractors(u, n):
        same = [v for v in units if v != u and by[v]["kind"] == by[u]["kind"] and fam.get(v, -v - 1) != fam.get(u, -u - 1) and by[v]["title"] != by[u]["title"]]
        pick = [int(x) for x in rng.choice(same, min(n, len(same)), replace=False)] if same else []
        if len(pick) < n:                                      # a kind with too few units is filled from the other kinds
            rest = [v for v in units if v != u and v not in pick and by[v]["title"] != by[u]["title"]]
            pick += [int(x) for x in rng.choice(rest, n - len(pick), replace=False)]
        return pick

    items = []
    sham = set(int(x) for x in rng.choice(units, min(40, len(units)), replace=False))
    for u in units:
        f = [forced[(u, float(multiple))][k] for k in range(len(openers))]
        nrm = [base[(u, float(multiple))][k] for k in range(len(openers))]
        d = distractors(u, 5)
        order = rng.permutation(5)
        cand = [u] + d[:4]
        items.append({"id": f"u{u}", "unit": u, "sham": False, "openers": openers, "normal": nrm, "forced": f,
                      "candidates": [by[cand[j]]["title"] for j in order], "answer": str(int(np.flatnonzero(order == 0)[0]) + 1)})
        if u in sham:
            order = rng.permutation(5)
            items.append({"id": f"s{u}", "unit": u, "sham": True, "openers": openers, "normal": nrm, "forced": f,
                          "candidates": [by[d[j]]["title"] for j in order], "answer": "none"})
    C.save({"model": tag, "multiple": multiple, "items": items}, f"{tag}/steering_items.json")
    write_prompts(tag, [{k: v for k, v in it.items() if k not in ("answer", "unit", "sham")} for it in items], C.SEED + 4)


def study_score(tag):
    d = json.load(open(C.RESULTS / tag / "steering_items.json"))
    tests = {r["unit"]: r for r in json.load(open(C.RESULTS / tag / "explanation_tests.json"))["units"]}
    ex = {int(u): e for u, e in json.load(open(C.RESULTS / tag / "explanations.json"))["units"].items()}
    v = read_verdicts(tag)
    real = [it for it in d["items"] if not it["sham"]]
    sham = [it for it in d["items"] if it["sham"]]

    def acc(rows):
        got = [v.get(it["id"], {}).get("choice") for it in rows]
        k = int(sum(x == it["answer"] for x, it in zip(got, rows)))
        return {"n": len(rows), "judged": int(sum(x is not None for x in got)), "correct": k, "accuracy": k / max(len(rows), 1),
                "none": int(sum(x == "none" for x in got)),
                "p_above_chance": float(binomtest(k, len(rows), 0.2, alternative="greater").pvalue) if rows else None}

    def keyword_rate(rows):
        """Judge-free: the share of continuations that contain one of the explainer's key words."""
        own, other, base = [], [], []
        for it in rows:
            key = T.key_stems(ex[it["unit"]]["key_words"])
            own += [bool(T.uses(t, key)) for t in it["forced"]]
            base += [bool(T.uses(t, key)) for t in it["normal"]]
            for jt in rows:
                if jt["unit"] != it["unit"]:
                    other += [bool(T.uses(t, key)) for t in jt["forced"]]
        return {"own_forced": float(np.mean(own)) if own else None, "unforced": float(np.mean(base)) if base else None,
                "other_units_forced": float(np.mean(other)) if other else None}

    idea = [it for it in real if tests[it["unit"]]["is_idea"]]
    name = [it for it in real if not tests[it["unit"]]["is_idea"]]
    out = {"model": tag, "multiple": d["multiple"], "chance": 0.2, "idea_units": acc(idea), "name_units": acc(name), "all_units": acc(real),
           "by_kind": {k: acc([it for it in real if tests[it["unit"]]["kind"] == k]) for k in sorted({tests[it["unit"]]["kind"] for it in real})},
           "by_depth_quarter": {str(q): acc([it for it in real if tests[it["unit"]]["depth_quarter"] == q]) for q in (1, 2, 3, 4)},
           "idea_units_by_depth_quarter": {str(q): acc([it for it in idea if tests[it["unit"]]["depth_quarter"] == q]) for q in (1, 2, 3, 4)},
           "sham_items": {"n": len(sham), "answered_none": int(sum(v.get(it["id"], {}).get("choice") == "none" for it in sham)),
                          "judged": int(sum(it["id"] in v for it in sham))},
           "key_words_in_continuations": {"idea_units": keyword_rate(idea), "name_units": keyword_rate(name)},
           "criterion_met": bool(idea and acc(idea)["p_above_chance"] < 0.05),
           "verdicts": {it["id"]: dict(v.get(it["id"], {}), answer=it["answer"], unit=it["unit"], sham=it["sham"]) for it in d["items"]}}
    C.save(out, f"steering_{tag}.json")
    print(tag, "steering: idea units", out["idea_units"], "| names", out["name_units"]["accuracy"], "| sham none", out["sham_items"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"gallery": gallery, "gallery-items": gallery_items, "gallery-score": gallery_score}.get(cmd, lambda: {"items": study_items, "score": study_score}[cmd](sys.argv[2]))()
