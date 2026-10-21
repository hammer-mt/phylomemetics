"""Controlled cost benchmark for the pre-registration (frozen; prereg-v2). DEV texts only, so it can run at F1.

Production timings come from long runs on a shared laptop; this script measures every method's unit costs under fixed
conditions instead: the same 30 DEV reference films (original texts, the first 30 in a fixed sha256 order) stand in
for 30 stories, all 34 memes, one warm-up run (discarded) then 3 timed repeats, nothing else running. Before each run
it records os.getloadavg() and every running python / MLX process (ps); each method runs in its own subprocess under
/usr/bin/time -l (max resident set size), and model load time is recorded apart from work time. Results, and the
per-method cost recipes analysis.run_cost uses (setup, per story, per story and meme, a later meme, storage), go to
preregistration/results/bench.json.

  .venv/bin/python preregistration/bench.py run      # about 1-2 hours
  .venv/bin/python preregistration/bench.py recipes  # rebuild the recipes from the stored runs and the DEV-freeze timers
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PRE = Path(__file__).resolve().parent
ROOT = PRE.parent
STUDY = ROOT / "study"   # the Batman study: the DEV corpus and its committed results
BENCH = PRE / "bench_work"
PY = sys.executable
N_STORIES = 30
REPEATS = 3
SALT = "phylomemetics-bench-v2|"
METHODS = {
    "neurons": ["extract_compact.py", "bench_stories", "{m}", "--units", str(PRE / "sentence_units_{m}.npy"), "--fresh"],
    "neurons+hidden": ["extract_compact.py", "bench_stories", "{m}", "--units", str(PRE / "sentence_units_{m}.npy"), "--hidden", "--fresh"],
    "judge": ["readers.py", "judge", "--set", "bench_stories", "--model", "{m}"],
    "judge-shared": ["readers.py", "judge-shared", "--set", "bench_stories", "--model", "{m}"],
    "judge-1pass": ["readers.py", "judge-1pass", "--set", "bench_stories", "--model", "{m}"],
    "judge-small:qwen3-1.7b": ["readers.py", "judge", "--set", "bench_stories", "--model", "qwen3-1.7b"],
    "summary": ["readers.py", "summarise", "--set", "bench_stories", "--model", "{m}"],
    "summary-judge": ["readers.py", "sjudge", "--set", "bench_stories", "--model", "{m}"],
    "nli": ["readers.py", "nli", "--set", "bench_stories"],
    "emb:qwen3emb-4b": ["readers.py", "embed", "--set", "bench_stories", "--encoder", "qwen3emb-4b"],
    "emb:e5-large": ["readers.py", "embed", "--set", "bench_stories", "--encoder", "e5-large"],
    "emb:e5-base": ["readers.py", "embed", "--set", "bench_stories", "--encoder", "e5-base"],
}


def processes():
    out = subprocess.run(["ps", "-axo", "pid,%cpu,rss,command"], capture_output=True, text=True).stdout.splitlines()
    return [l for l in out if re.search(r"python|mlx", l) and "bench.py" not in l][:20]


def make_set():
    corpus = [json.loads(l) for l in open(STUDY / "data" / "corpus.jsonl")]
    films = sorted([w for w in corpus if w["group"] == "baseline"], key=lambda w: hashlib.sha256((SALT + w["label"]).encode()).hexdigest())[:N_STORIES]
    (BENCH / "data").mkdir(parents=True, exist_ok=True)
    with open(BENCH / "data" / "bench_stories.jsonl", "w") as f:
        for w in films:
            f.write(json.dumps({"id": w["label"], "sentences": w["sentences"]}) + "\n")
    # summaries are needed by sjudge: the 'summary' method writes them before 'summary-judge' runs
    return films


def run_one(name, cmd, model):
    env = dict(os.environ, PREREG_DATA=str(BENCH / "data"), PREREG_FEATURES=str(BENCH / "features"), HF_HUB_OFFLINE="1")
    args = [a.replace("{m}", model) for a in cmd]
    if args[0] == "extract_compact.py":
        args += ["--data", str(BENCH / "data"), "--out", str(BENCH / "features")]
    feat = BENCH / "features" / "bench_stories"
    if name not in ("summary-judge",):
        for f in feat.glob("llm_*_judge*.jsonl") if name.startswith("judge") else []:
            f.unlink()
        if name == "summary":
            for f in feat.glob("llm_*_summary.jsonl"):
                f.unlink()
    for f in feat.glob("llm_*_sjudge.jsonl") if name == "summary-judge" else []:
        f.unlink()
    pre = {"loadavg": list(os.getloadavg()), "processes": processes()}
    t0 = time.time()
    p = subprocess.run(["/usr/bin/time", "-l", PY, str(PRE / args[0])] + args[1:], env=env, capture_output=True, text=True)
    wall = time.time() - t0
    rss = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    costs = {f.name: json.load(open(f)) for f in feat.glob("cost_*.json")} if feat.exists() else {}
    return {"method": name, "model": model, "wall_seconds": wall, "returncode": p.returncode, "max_rss_gb": int(rss.group(1)) / 1e9 if rss else None,
            "pre": pre, "post_loadavg": list(os.getloadavg()), "cost_files": costs, "stderr_tail": p.stderr[-500:]}


def run(model="qwen3-8b"):
    if BENCH.exists():
        shutil.rmtree(BENCH)
    make_set()
    runs = []
    for name, cmd in METHODS.items():
        print("bench", name, flush=True)
        run_one(name, cmd, model)                                   # warm-up, discarded
        runs += [dict(run_one(name, cmd, model), repeat=k) for k in range(REPEATS)]
    out = {"meta": {"machine": os.uname().machine, "date": time.strftime("%Y-%m-%d %H:%M"), "n_stories": N_STORIES, "repeats": REPEATS,
                    "model": model}, "runs": runs}
    out["recipes"] = recipes(out)
    (PRE / "results").mkdir(exist_ok=True)
    json.dump(out, open(PRE / "results" / "bench.json", "w"), indent=1)


def recipes(bench):
    """Per-method recipe (seconds unless noted) from the timed runs and the DEV-freeze timers:
    setup_fixed_s (one-off), setup_per_meme_s (a meme's detector or query), per_story_s (one pass), per_story_per_meme_s,
    new_meme_per_story_s (scoring stored stories for a meme added later), store_all_bytes / store_selected_bytes."""
    runs, n = bench["runs"], bench["meta"]["n_stories"]
    fz = json.load(open(PRE / "dev_freeze.json")) if (PRE / "dev_freeze.json").exists() else {}
    t = fz.get("timers", {})
    n_m = len(json.load(open(PRE / "memes_for_tracing.json"))["memes"])
    model = bench["meta"]["model"]

    def per(name, key="seconds"):
        v = [r["cost_files"] for r in runs if r["method"] == name]
        s = [next((c[key] for c in cf.values() if key in c), None) for cf in v]
        s = [x for x in s if x is not None]
        return float(np.median(s)) if s else None
    rec = {}
    sel = t.get(f"neurons_select_{model}", 0.0) / n_m
    neu = per("neurons")
    # one-off neuron setup: full activations of the background and example sentences (production DEV cost files)
    bgx = sum(json.load(open(f)).get("seconds", 0.0) for f in [PRE / "features" / s_ / f"cost_neurons_hidden_{model}.json"
                                                              for s_ in ("t2_background", "t2_examples")] if f.exists())
    rec["neurons"] = {"setup_fixed_s": bgx, "setup_per_meme_s": sel, "per_story_s": (neu or 0) / n,
                      "per_story_per_meme_s": 0.0, "new_meme_per_story_s": 0.0, "store_all_bytes": 884736 * 2,
                      "store_selected_bytes": 2 * 30 * len(fz.get("sentence_units", {}).get(model, []))}
    nh = per("neurons+hidden")
    rec["neurons+hidden"] = dict(rec["neurons"], per_story_s=(nh or 0) / n)
    for enc in ("qwen3emb-4b", "e5-large", "e5-base"):
        s = per(f"emb:{enc}")
        rec[f"emb:{enc}"] = {"setup_fixed_s": 0.0, "setup_per_meme_s": t.get(f"emb_detectors_{enc}", 0.0) / n_m, "per_story_s": (s or 0) / n,
                             "per_story_per_meme_s": 0.0, "new_meme_per_story_s": 0.0, "store_all_bytes": 30 * 2 * 2560}
    for name in ("judge", "judge-small:qwen3-1.7b"):
        pf, pm = per(name, "prefill_seconds"), per(name, "per_meme_seconds")
        rec[name] = {"setup_fixed_s": 0.0, "per_story_s": (pf or 0) / n, "per_story_per_meme_s": pm or 0.0, "new_meme_per_story_s": (pf or 0) / n + (pm or 0)}
    for name in ("judge-shared", "judge-1pass"):
        sh, s = per(name, "sheet_seconds"), per(name)
        rec[name] = {"setup_fixed_s": sh or 0.0, "per_story_s": (s or 0) / n, "per_story_per_meme_s": 0.0,
                     "new_meme_per_story_s": (s or 0) / n}
    sm, sj, sjm = per("summary"), per("summary-judge", "prefill_seconds"), per("summary-judge", "per_meme_seconds")
    rec["summary-judge"] = {"per_story_s": ((sm or 0) + (sj or 0)) / n, "per_story_per_meme_s": sjm or 0.0, "new_meme_per_story_s": (sjm or 0)}
    nl = per("nli")
    rec["nli"] = {"per_story_s": 0.0, "per_story_per_meme_s": (nl or 0) / n / n_m, "new_meme_per_story_s": (nl or 0) / n / n_m}
    rec["keywords"] = {"setup_per_meme_s": t.get("keywords_and_logreg", 0.0) / n_m, "per_story_s": 0.001}
    rec["lda"] = {"per_story_s": 0.01}
    return rec


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "run"
    if what == "run":
        run()
    elif what == "recipes":
        b = json.load(open(PRE / "results" / "bench.json"))
        b["recipes"] = recipes(b)
        json.dump(b, open(PRE / "results" / "bench.json", "w"), indent=1)
