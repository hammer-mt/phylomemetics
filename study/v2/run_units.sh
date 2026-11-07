#!/bin/bash
# From the three passes to the explainers' prompts, for one model. Run from study/ with RUNPOD_STATE pointing at the
# pod that holds the model's outputs, after `v2/gpu/pull.py <tag> --raw --franchises`.
#   bash v2/run_units.sh <model-tag>
# Steps: the experiments and the canon (laptop), the sentence pass over the candidate units (GPU), the meme families
# and the units to explain (laptop), their strongest reference examples (GPU), the explainers' inputs and prompts.
set -euo pipefail
TAG=$1
PY=../.venv/bin/python
export PYTHONPATH=v2/analysis
mkdir -p results/v2/logs
$PY v2/analysis/experiments_v2.py "$TAG" 2>&1 | tee results/v2/logs/experiments_$TAG.log
$PY v2/analysis/canon_v2.py "$TAG" 2>&1 | tee results/v2/logs/canon_$TAG.log
$PY v2/gpu/remote.py sentences "$TAG" results/v2/$TAG/sentence_units.npy 2>&1 | tail -5
$PY v2/analysis/families_v2.py "$TAG" 2>&1 | tee results/v2/logs/families_$TAG.log
$PY v2/analysis/select_units.py "$TAG" 2>&1 | tee results/v2/logs/select_$TAG.log
$PY v2/gpu/remote.py top "$TAG" results/v2/$TAG/explain_units.npy 2>&1 | tail -5
$PY v2/analysis/build_explain_inputs.py "$TAG" 2>&1 | tail -3
$PY v2/analysis/explain_agents.py explainer "$TAG" 2>&1 | tail -3
echo UNITS_DONE "$TAG"
