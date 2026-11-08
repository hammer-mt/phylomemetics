"""The follow-up ablation: the same test with the model's reply begun for it (PLAN.md Section 18, change 33).

    cd study && ../.venv/bin/python v2/analysis/build_ablation_followup.py jobs      every job file, for both prefixes and both models
    cd study && ../.venv/bin/python v2/analysis/build_ablation_followup.py record    -> results/v2/ablation_followup_jobs.json

The registered ablation of Qwen3-32B was not evaluable: five of the 32 untouched replies began with a sentence or a
placeholder code, and the score is the probability of the reply's first word. This writes the same jobs again with
one change. The question is the registered one, word for word, and the model's reply is begun for it ("Hero:"), so
that the next word it writes is the name. Everything else is copied from the jobs of the registered run and is not
rebuilt: the texts, each work's canon units, the same random sets (the same partner file) and the conditions. An
answer is the registered name with the space that follows the colon (" Batman").

Two prefixes are written because the plan checks the first for form on texts outside the test and names one
alternative if it fails: "Hero:" (slug hero) and "Answer:" (slug answer). For each prefix:

  v2/cache/jobs/begun_form_<slug>.json                  question 1 with the prefix, the 69 ordinary films, nothing switched off
  v2/cache/jobs/begun_probe_<slug>.json                 the same for all 114 name-free texts (the descriptive control)
  v2/cache/jobs/ablate_<tag>_begun_<slug>.json          the registered ablation jobs with the prefix
  v2/cache/jobs/ablate_<tag>_begun_<slug>_strong.json   only the canon's strong firings switched off (20 sets)
  v2/cache/jobs/ablate_<tag>_begun_<slug>_weak.json     only its weaker activity (20 sets)

`record` writes the SHA-256 of each of these, of the two partner files and of the two scale files, so that the
committed record identifies what is run (the files themselves are in ignored folders).
"""
import hashlib
import json
import sys
from pathlib import Path

import common as C

import experiments as X
from build_probe_jobs import Q_HERO

PREFIXES = {"hero": "Hero:", "answer": "Answer:"}          # in the order the plan tries them
TAGS = ("qwen3-32b", "qwen3-8b")
JOBS = Path("v2/cache/jobs")


def begun(job, prefix):
    """The same job with the reply begun: the prefix, and every answer led by the space that follows it."""
    assert job["question"] == Q_HERO, "the question must be the registered one, word for word"
    assert "reply_prefix" not in job
    out = dict(job, reply_prefix=prefix, items=[dict(it) for it in job["items"]])
    for it in out["items"]:
        assert "question" not in it and all(a == a.strip() for a in it["answers"])
        it["answers"] = [" " + a for a in it["answers"]]
    return out


def files():
    """Every file of the follow-up, by name."""
    names = []
    for slug in PREFIXES:
        names += [f"begun_form_{slug}.json", f"begun_probe_{slug}.json"]
        names += [f"ablate_{tag}_begun_{slug}{s}.json" for tag in TAGS for s in ("", "_strong", "_weak")]
    return names + [f"ablate_{tag}_partners.npz" for tag in TAGS]


def jobs():
    probe = json.load(open(JOBS / "recognise_hero.json"))                   # the registered probe: all 114 name-free texts, question 1
    films = {X.label[w] for w in X.FILMS}
    assert len(probe["items"]) == 114 and sum(it["id"] in films for it in probe["items"]) == 69
    for slug, prefix in PREFIXES.items():
        p = begun(probe, prefix)
        json.dump(p, open(JOBS / f"begun_probe_{slug}.json", "w"), ensure_ascii=False)
        json.dump(dict(p, items=[it for it in p["items"] if it["id"] in films]), open(JOBS / f"begun_form_{slug}.json", "w"), ensure_ascii=False)
        for tag in TAGS:
            for s in ("", "_strong", "_weak"):
                job = begun(json.load(open(JOBS / f"ablate_{tag}{s}.json")), prefix)
                json.dump(job, open(JOBS / f"ablate_{tag}_begun_{slug}{s}.json", "w"), ensure_ascii=False)
                n_cond = sorted({len(it["conditions"]) for it in job["items"]})
                print(f"ablate_{tag}_begun_{slug}{s}.json", len(job["items"]), "items,", n_cond, "conditions each, where", job.get("where", "story"),
                      "mode", job.get("mode", "all"), "| answers of the first item", job["items"][0]["answers"])


def record():
    sha = {n: hashlib.sha256((JOBS / n).read_bytes()).hexdigest() for n in files()}
    scale = {tag: hashlib.sha256((C.ACTS / tag / "scale.npy").read_bytes()).hexdigest() for tag in TAGS}
    C.save({"what": "SHA-256 of the job files of the follow-up ablation (PLAN.md Section 18, change 33), of the partner files of the registered run, "
                    "which hold the canon units and every random set, and of each model's scale (v2/acts/<tag>/scale.npy), which the two secondaries use. "
                    "Written before any of these jobs was run.",
            "question": Q_HERO, "prefixes_in_the_order_tried": list(PREFIXES.values()), "written_by": "v2/analysis/build_ablation_followup.py", "sha256": sha,
            "scale_sha256": scale}, "ablation_followup_jobs.json")
    print(len(sha), "job files and", len(scale), "scale files recorded")


if __name__ == "__main__":
    {"jobs": jobs, "record": record}[sys.argv[1]]()
