"""Coverage: does an explanation describe what its unit does at 30 to 60% of scale, where presence is decided?
(PLAN.md, Section 10, "coverage".)

    cd study && ../.venv/bin/python v2/analysis/coverage.py plots <model-tag>       plots to score (texts and units for gpu/score_texts.py)
    ... gpu/score_texts.py -> v2/acts/<tag>/coverage_plots.npz
    cd study && ../.venv/bin/python v2/analysis/coverage.py passages <model-tag>    candidate passages, cut at the peak and at other tokens
    ... gpu/score_texts.py -> v2/acts/<tag>/coverage_passages.npz
    cd study && ../.venv/bin/python v2/analysis/coverage.py items <model-tag>       the judge's prompts
    cd study && ../.venv/bin/python v2/analysis/coverage.py score <model-tag>

For each explained unit, up to six background plots in which it is present at 30 to 60% of its scale are cut at the
token where it peaks ("fires"); from the same plots, as many passages cut at a token where its activation is under
10% of scale ("silent"). A blind judge with only the title and explanation says for each whether the description
predicts a strong firing on the last token.
"""
import json
import sys
from pathlib import Path

import numpy as np

import common as C
from explain_agents import ANSWER
from franchises_v2 import bit_columns

PER_UNIT, TRY_PLOTS, UNITS_PER_PROMPT = 6, 14, 3


def work_dir(tag):
    d = Path("v2/cache/coverage") / tag
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def explained(tag):
    return {int(u): e for u, e in json.load(open(C.RESULTS / tag / "explanations.json"))["units"].items()}


def step_plots(tag):
    rng = np.random.default_rng(C.SEED + 11)
    units = np.array(sorted(explained(tag)), dtype=np.int64)
    bits = np.load(C.ACTS / tag / "bg_present30.bits.npy", mmap_mode="r")
    P = bit_columns(bits, units)                                   # [plots, units]
    chosen = {}
    for j, u in enumerate(units):
        rows = np.flatnonzero(P[:, j])
        chosen[int(u)] = sorted(int(x) for x in rng.choice(rows, min(TRY_PLOTS, len(rows)), replace=False)) if len(rows) else []
    docs = C.background_docs()
    need = sorted({r for v in chosen.values() for r in v})
    d = work_dir(tag)
    with open(d / "plots.jsonl", "w") as f:
        for r in need:
            f.write(json.dumps({"id": f"plot{r}", "text": " ".join(docs[r]["sentences"])}, ensure_ascii=False) + "\n")
    np.save(d / "units.npy", units)
    json.dump(chosen, open(d / "chosen_plots.json", "w"))
    print(f"{tag}: {len(need)} plots to score for {len(units)} units ({sum(bool(v) for v in chosen.values())} present in some plot) -> {d}/plots.jsonl")


def step_passages(tag):
    rng = np.random.default_rng(C.SEED + 12)
    d = work_dir(tag)
    z = np.load(C.ACTS / tag / "coverage_plots.npz")
    units = [int(u) for u in z["units"]]
    scale = C.scale(tag)[units]
    row = {str(i): k for k, i in enumerate(z["ids"])}
    texts = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(d / "plots.jsonl")}
    chosen = json.load(open(d / "chosen_plots.json"))
    out = []
    for j, u in enumerate(units):
        for r in chosen[str(u)]:
            k = row[f"plot{r}"]
            share = float(z["peak"][k, j] / scale[j]) if scale[j] > 0 else 0.0
            if not 0.30 <= share < 0.60:
                continue
            text, end = texts[f"plot{r}"], int(z["where_end"][k, j])
            out.append({"id": f"{u}:{r}:fire", "unit": u, "plot": r, "kind": "fire", "share_in_plot": round(share, 3), "text": text[:end]})
            words = [m for m in range(len(text)) if text[m] == " " and 200 < m < len(text) - 1 and abs(m - end) > 40]
            for c, m in enumerate(rng.choice(words, min(3, len(words)), replace=False) if words else []):
                out.append({"id": f"{u}:{r}:cut{c}", "unit": u, "plot": r, "kind": "cut", "text": text[:int(m)]})
    with open(d / "passages.jsonl", "w") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{tag}: {sum(p['kind'] == 'fire' for p in out)} firing passages and {sum(p['kind'] == 'cut' for p in out)} candidate silent ones -> {d}/passages.jsonl")


def step_items(tag):
    rng = np.random.default_rng(C.SEED + 13)
    d = work_dir(tag)
    ex = explained(tag)
    z = np.load(C.ACTS / tag / "coverage_passages.npz")
    units = [int(u) for u in z["units"]]
    col = {u: k for k, u in enumerate(units)}
    scale = C.scale(tag)
    L, D = C.geometry(tag)
    row = {str(i): k for k, i in enumerate(z["ids"])}
    by_unit = {}
    for p in map(json.loads, open(d / "passages.jsonl")):
        u, k = p["unit"], row[p["id"]]
        sign = 1.0 if u < L * D else -1.0
        p["last_share"] = float(sign * z["last"][k, col[u]] / scale[u]) if scale[u] > 0 else 0.0
        by_unit.setdefault(u, {"fire": [], "cut": []})[p["kind"]].append(p)
    items = []
    for u, g in by_unit.items():
        fires = [p for p in g["fire"] if p["last_share"] >= 0.30][:PER_UNIT]          # the cut text must still end on a strong firing
        plots = {p["plot"] for p in fires}
        silent, seen = [], set()
        for p in g["cut"]:
            if p["plot"] in plots and p["plot"] not in seen and p["last_share"] < 0.10:
                silent.append(p)
                seen.add(p["plot"])
        for p in fires + silent:
            items.append({"id": p["id"], "unit": u, "fires": p["kind"] == "fire", "share": round(p.get("share_in_plot", p["last_share"]), 3),
                          "passage": ("…" if len(p["text"]) > 1200 else "") + p["text"][-1200:]})
    C.save({"model": tag, "n_items": len(items), "units_with_items": len({i["unit"] for i in items}), "items": items}, f"{tag}/coverage_items.json")
    template = Path("v2/prompts/coverage_judge.md").read_text()
    us = sorted({i["unit"] for i in items})
    order = rng.permutation(len(us))
    paths = []
    for b in range(0, len(us), UNITS_PER_PROMPT):
        blocks = []
        for u in [us[i] for i in order[b:b + UNITS_PER_PROMPT]]:
            mine = [i for i in items if i["unit"] == u]
            mine = [mine[i] for i in rng.permutation(len(mine))]
            lines = [f"UNIT {u}", f"Title: {ex[u]['title']}", f"Explanation: {ex[u]['explanation']}", "Passages:"]
            for it in mine:
                text = it["passage"].rstrip()
                cut = text.rfind(" ") + 1
                lines.append(f"[{it['id']}] {text[:cut]}«{text[cut:]}»")
            blocks.append("\n".join(lines))
        name = f"batch_{b // UNITS_PER_PROMPT:03d}"
        (d / f"{name}.prompt.txt").write_text(template.replace("{items}", "\n\n==========\n\n".join(blocks)) + ANSWER.format(out=d / f"{name}.out.json", what="passage"))
        paths.append(str(d / f"{name}.prompt.txt"))
    json.dump(paths, open(d / "prompts.json", "w"))
    print(f"{tag}: {len(items)} passages for {len(us)} units in {len(paths)} prompt files -> {d}/prompts.json")


def step_score(tag):
    d = work_dir(tag)
    items = json.load(open(C.RESULTS / tag / "coverage_items.json"))["items"]
    tests = {r["unit"]: r for r in json.load(open(C.RESULTS / tag / "explanation_tests.json"))["units"]}
    verdict = {}
    for f in sorted(d.glob("batch_*.out.json")):
        try:
            for r in json.load(open(f)):
                verdict[str(r["id"])] = bool(r["fires"])
        except Exception as e:      # noqa: BLE001
            print("unreadable", f.name, repr(e)[:80])

    def rates(rows):
        fire = [verdict[i["id"]] for i in rows if i["fires"] and i["id"] in verdict]
        sil = [verdict[i["id"]] for i in rows if not i["fires"] and i["id"] in verdict]
        return {"firing_passages": len(fire), "covered": float(np.mean(fire)) if fire else None, "silent_passages": len(sil),
                "wrongly_claimed": float(np.mean(sil)) if sil else None, "units": len({i["unit"] for i in rows})}

    kinds = sorted({tests[i["unit"]]["kind"] for i in items if i["unit"] in tests})
    out = {"model": tag, "judged": len(verdict), "of": len(items), "all": rates(items),
           "idea_units": rates([i for i in items if tests.get(i["unit"], {}).get("is_idea")]),
           "name_units": rates([i for i in items if i["unit"] in tests and not tests[i["unit"]]["is_idea"]]),
           "by_kind": {k: rates([i for i in items if tests.get(i["unit"], {}).get("kind") == k]) for k in kinds},
           "by_share_band": {f"{int(lo * 100)}-{int(hi * 100)}%": rates([i for i in items if not i["fires"] or lo <= i["share"] < hi])
                             for lo, hi in ((0.3, 0.4), (0.4, 0.5), (0.5, 0.6))},
           "verdicts": {i["id"]: {"fires": i["fires"], "judged_fires": verdict.get(i["id"])} for i in items}}
    C.save(out, f"coverage_{tag}.json")
    print(tag, "coverage:", out["all"])


if __name__ == "__main__":
    {"plots": step_plots, "passages": step_passages, "items": step_items, "score": step_score}[sys.argv[1]](sys.argv[2])
