#!/bin/bash
# The ablation test for one model: job files, the confirmatory run, the four secondaries, scoring. Run from study/
# with RUNPOD_STATE pointing at a pod that has the model's weights and its scale (out/<tag>/study/scale.npy).
#   bash v2/run_ablation.sh <model-tag>
set -euo pipefail
TAG=$1
PY=../.venv/bin/python
export PYTHONPATH=v2/analysis
$PY v2/analysis/build_ablation.py "$TAG" | tail -1 | cut -c1-300
$PY -c "
import sys; sys.path.insert(0, 'v2/gpu')
import runpod_ctl as rc
rc.put('v2/cache/jobs/ablate_${TAG}_partners.npz', 'data/jobs/ablate_${TAG}_partners.npz')"
$PY v2/gpu/remote.py ask "$TAG" v2/cache/jobs/ablate_$TAG.json ablate | tail -3
$PY v2/analysis/causal.py ablation "$TAG"
for s in all_positions original; do $PY v2/gpu/remote.py ask "$TAG" v2/cache/jobs/ablate_${TAG}_$s.json ablate_$s | tail -2; done
for s in strong weak; do $PY v2/gpu/remote.py ask "$TAG" v2/cache/jobs/ablate_${TAG}_$s.json ablate_$s scale | tail -2; done
$PY v2/analysis/causal.py ablation "$TAG"
echo ABLATION_DONE "$TAG"
