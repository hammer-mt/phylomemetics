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
| `gpu/pod_agent.py`, `gpu/runpod_ctl.py`, `gpu/pull.py`, `gpu/remote.py`, `gpu/run_study.sh` | Rent a Runpod GPU, run jobs on it, fetch results (checked by hash), shut it down | laptop |
| `check_reference_overlap.py` | Reports which study works are also in the reference corpus | laptop |
| `analysis/common.py` | The v2 presence rule and loaders | laptop |
| `analysis/experiments_v2.py` | E1, E2 and E3 under the v2 rule (E2 and E3 are v1's own functions), with the word comparators | laptop |
| `analysis/v1_rule.py`, `analysis/contrasts.py`, `analysis/cells_joint.py` | v1's rule on the new models; what changed a result (pipeline, rule or model); how surprising a count of cells below 0.05 is | laptop |
| `analysis/multiverse_v2.py`, `analysis/sensitivity_v2.py` | Robustness of E2, and of E1, the residual test and E3, to the new choices | laptop |
| `analysis/canon_v2.py`, `analysis/families_v2.py`, `analysis/select_units.py` | The canon, meme families, and the rule-based choice of units to explain | laptop |
| `analysis/franchises_v2.py`, `analysis/trees_v2.py` | Batman against other film series and genre sets; family trees and their tests | laptop |
| `analysis/build_explain_inputs.py`, `analysis/explain_agents.py`, `analysis/explain_score.py`, `analysis/textcheck.py` | What the explainer sees, the agents' prompt and answer files, and the scoring of the explanation tests | laptop |
| `analysis/plain_checks.py`, `analysis/coverage.py`, `analysis/size_test.py`, `analysis/second_judge.py` | Checks of the explanations that need no AI judge; coverage at 30 to 60% of scale; bigger against smaller model; Qwen3-32B as a second judge | laptop |
| `analysis/build_probe_jobs.py`, `analysis/build_ablation.py`, `analysis/build_ablation_followup.py`, `analysis/causal.py`, `analysis/steering.py` | The recognition probe, the ablation, its follow-up test with the model's reply begun for it, and the steering test: job files and scoring | laptop |
| `analysis/twins.py` | Twins of the explained units in the other models, scored where they were not chosen | laptop |
| `analysis/confirmatory.py`, `analysis/pass_summary.py` | The six confirmatory tests with Holm-adjusted p values; what was run for each model | laptop |
| `analysis/export_units.py`, `analysis/code_counts.py` | One table of the explained units and the works that carry them; how many placeholder codes each name-free text holds | laptop |
| `analysis/paper_tables.py`, `make_paper_assets.sh` | Every table of the paper, written from the committed results to `../results/v2/paper_tables.md`; the summaries, tables and figures in one go | laptop |
| `run_units.sh`, `run_rest.sh`, `run_analysis.sh`, `run_ablation.sh`, `run_ablation_followup.sh` | The order of the steps for one model; the last is the follow-up ablation, which refuses uncommitted or unpushed code | laptop |
| `heavy.sh` | Runs a memory-heavy analysis behind a lock, so that only one runs at a time on a small machine | laptop |
| `figs/` | The figures of paper v2, drawn from the committed results | laptop |
| `second/` | Two further corpora, Sherlock Holmes and film remakes: their own frozen plan, corpus builders, GPU steps, tests, audits and tables. See [`second/README.md`](second/README.md). Results in `../results/v2/second/` | both |

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

## Running it

Everything is run from `study/`. `PY=../.venv/bin/python`, and `RUNPOD_STATE` names the state file of the GPU machine a step should use (`gpu/runpod_ctl.py create` writes it; state files hold access tokens and are not committed).

1. **Corpora** (laptop): `$PY v2/build_reference.py`, `$PY v2/build_supplement.py`, `$PY v2/build_franchises.py`. Upload them and `data/corpus*.jsonl` to the GPU machine with `gpu/runpod_ctl.py put`.
2. **The three passes** (GPU, per model): `ref_pass.py <tag> reference.jsonl out`, the same for `supplement.jsonl` with `--name ref_sup --max-tokens 2048`, then `bash run_study.sh <tag>` (study pass, scale, background base rates, other franchises).
3. **Fetch and analyse** (laptop): `$PY v2/gpu/pull.py <tag> --raw --franchises`, then `bash v2/run_units.sh <tag>` (experiments, canon, sentence pass, meme families, units to explain, their top examples, the explainers' prompts) and `bash v2/run_rest.sh <tag>` (other franchises, v1's rule, sensitivity grid, multiverse).
4. **Explanations**: the prompt files written by `analysis/explain_agents.py` are each answered by a fresh AI agent, which writes one answer file. Then `explain_agents.py collect-explainer`, `others`, `collect-others` (repeated while it sends texts back for rewriting), `explain_score.py texts`, `gpu/remote.py score ...`, `explain_score.py score`. `plain_checks.py`, `coverage.py`, `size_test.py` and `second_judge.py` follow.
5. **Causal tests**: `build_probe_jobs.py` and `gpu/remote.py ask` for the recognition questions, with `causal.py identify-items`, `identify-collect` and `recognition`; `build_ablation.py`, `gpu/remote.py ask` and `causal.py ablation`; for the follow-up ablation (`PLAN.md` Section 18, change 33), `build_ablation_followup.py jobs` and `bash v2/run_ablation_followup.sh <tag>`, Qwen3-32B first, which checks the prefix, runs the jobs and scores them; `steering.py gallery`, `gpu/remote.py steer`, `steering.py gallery-items` and `gallery-score`, then `steering.py items` and `score` for the explained units.
6. **Two models**: `twins.py pool`, `queries`, `gpu/remote.py twins`, `twins.py units` and `score`.
7. **Summaries**: `contrasts.py`, `cells_joint.py`, `trees_v2.py`, `export_units.py`, `code_counts.py`, then `bash v2/make_paper_assets.sh` (`pass_summary.py`, `confirmatory.py`, `paper_tables.py` and the figure scripts in `figs/`).

The analyses of Qwen3-32B and Llama 3.3 70B each hold several gigabytes in memory. On a machine with 16 GB, run them one at a time: `bash v2/heavy.sh $PY v2/analysis/<script>.py <tag>`.

The GPU machines are rented by the hour and shut themselves down when idle or past their deadline. The whole of v2 cost well under the $200 budget; `../results/v2/gpu_cost.json` and the paper give the figure.

## What came out

The paper is `../paper/v2/paper.md`, and `../paper/v2/CHANGES.md` lists what changed from version 1. `../results/v2/confirmatory.json` holds the six confirmatory tests and `../results/v2/paper_tables.md` every table. `PLAN.md` Section 18 logs every change made after the plan was frozen.
