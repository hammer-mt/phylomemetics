# The v2 pipeline

This folder holds the code behind paper v2 (`../paper/v2/`). Paper v1's code, in the folder above, is unchanged and still reproduces v1.

v2 re-runs the Batman study the way Cammarata's "Neural networks are already clean" studies neurons: a larger model at full precision (Qwen3-32B, the model the essay uses), units split by sign regime, "present" meaning a strong activation measured against the unit's own maximum in a large reference corpus, and explanations that are tested and can lose. The corpus and the three pre-registered experiments are v1's. [`PLAN.md`](PLAN.md) is the analysis plan. It was committed before any v2 model read a study text.

## What is where

| Path | What it does | Runs on |
| --- | --- | --- |
| `PLAN.md` | The analysis plan, frozen before the study pass | |
| `prompts/` | The instructions given to the AI agents (explainer, tester, keyword analyst, judges), word for word | |
| `build_reference.py` | Builds the reference corpus (46,000 documents, about 40 million tokens) from pinned public datasets | laptop |
| `build_supplement.py` | Builds the scale supplement: code, chat, other languages, news, reviews (5,434 documents) | laptop |
| `build_franchises.py` | Collects the films of six other franchises from the same plot dataset | laptop |
| `gpu/neurons.py` | Loads a model and reads every MLP neuron, split into main and echo sign regimes | GPU |
| `gpu/ref_pass.py` | Pass 1: every unit's strongest activations over the reference corpus and the supplement (its scale and top examples) | GPU |
| `gpu/study_pass.py` | Pass 2: every unit's score in each of the 114 works, original and name-free | GPU |
| `gpu/finalize_scale.py` | Fixes each unit's scale and stores the scores as shares of it | GPU machine, CPU |
| `gpu/background_pass.py` | Pass 3: each unit's base rate in 2,999 ordinary film plots; also reads the other franchises | GPU |
| `gpu/study_sentences.py`, `gpu/top_examples.py` | Sentence-level activations and top examples of chosen units | GPU |
| `gpu/score_texts.py` | Scores any texts on chosen units (explanation tests, published examples) | GPU |
| `gpu/steer.py`, `gpu/ablate.py` | The two causal tests; `ablate.py` also asks the untouched model the recognition questions | GPU |
| `gpu/twin_search.py` | Finds each unit's twin in another model | GPU |
| `gpu/published_examples.py` | Checks our activations against the ones the essay publishes | GPU and laptop |
| `gpu/test_pipeline.py`, `gpu/gpu_smoke.sh` | Check every pass against a slow re-computation on a tiny model, on a laptop or on the GPU | both |
| `gpu/pod_agent.py`, `gpu/runpod_ctl.py`, `gpu/pull.py`, `gpu/run_study.sh` | Rent a Runpod GPU, run jobs on it, fetch results (checked by hash), shut it down | laptop |
| `analysis/common.py` | The v2 presence rule and loaders | laptop |
| `analysis/experiments_v2.py` | E1, E2 and E3 under the v2 rule (E2 and E3 are v1's own functions), with the word comparators | laptop |
| `analysis/v1_rule.py` | v1's rule on the new models, to separate the effect of the model from the effect of the rule | laptop |
| `analysis/multiverse_v2.py`, `analysis/sensitivity_v2.py` | Robustness of E2, and of E1, the residual test and E3, to the new choices | laptop |
| `analysis/canon_v2.py`, `analysis/families_v2.py`, `analysis/select_units.py` | The canon, meme families, and the rule-based choice of units to explain | laptop |
| `analysis/build_explain_inputs.py`, `analysis/explain_agents.py`, `analysis/explain_score.py`, `analysis/textcheck.py` | What the explainer sees, the agents' prompt and answer files, and the scoring of the explanation tests | laptop |
| `analysis/build_probe_jobs.py`, `analysis/build_ablation.py`, `analysis/causal.py`, `analysis/steering.py` | The recognition probe, the ablation and the steering test: job files and scoring | laptop |
| `analysis/trees_v2.py` | Family trees and their tests | laptop |

Results are written to `../results/v2/`. Large arrays fetched from the GPU machine go to `acts/` and the corpora to `cache/`; neither is committed.

## Setup

The analyses use the project's environment (see the root README). The laptop side also needs `pandas` and `pyarrow` to build the reference corpus, and a CPU build of `torch` with `transformers` for `gpu/test_pipeline.py`. Nothing here needs Apple silicon. The GPU machine uses Runpod's `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` image plus `transformers`, `accelerate` and `hf_transfer`.

`gpu/runpod_ctl.py` reads the Runpod API key from `RUNPOD_API_KEY`.

## Testing the pipeline

```bash
cd study
export HF_HOME=$PWD/v2/cache/hf T=v2/cache/test PY=$PWD/../.venv/bin/python
mkdir -p $T/study && head -4 data/corpus.jsonl > $T/study/corpus.jsonl && head -4 data/corpus_neutral.jsonl > $T/study/corpus_neutral.jsonl
cd v2/gpu
$PY ref_pass.py qwen3-0.6b ../cache/reference.jsonl ../cache/test/out --limit-docs 48 --batch-tokens 4096
$PY study_pass.py qwen3-0.6b ../cache/test/study ../cache/test/out
$PY finalize_scale.py qwen3-0.6b ../cache/test/out
$PY -c "import numpy as np; np.save('../cache/test/sel.npy', np.array([5, 1000, 86023, 136016, 30737]))"
$PY background_pass.py qwen3-0.6b ../cache/reference.jsonl ../cache/test/out --select ../cache/test/sel.npy --limit-docs 6 --exclude-docs 843
$PY study_sentences.py qwen3-0.6b ../cache/test/study ../cache/test/out ../cache/test/sel.npy
printf '%s\n' '{"id": "a", "text": "After his parents are murdered in an alley, the boy swears revenge."}' '{"id": "b", "text": "She bought milk."}' > ../cache/test/out/qwen3-0.6b/score_test.jsonl
$PY score_texts.py qwen3-0.6b ../cache/test/out/qwen3-0.6b/score_test.jsonl ../cache/test/sel.npy ../cache/test/out/qwen3-0.6b/score_test.npz
$PY test_pipeline.py qwen3-0.6b ../cache/reference.jsonl ../cache/test/study ../cache/test/out ../cache/test/sel.npy
```

The last command prints PASS or FAIL for each check and exits non-zero if any fails. On the GPU machine, `bash gpu_smoke.sh` runs the same passes (and the supplement, franchise, steering and ablation scripts) on the same tiny model in bfloat16.
