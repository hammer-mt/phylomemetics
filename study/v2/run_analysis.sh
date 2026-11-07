#!/bin/bash
# The analyses that need only the three passes, for one model. Run from study/ after `v2/gpu/pull.py <tag> --raw --franchises`.
#   bash v2/run_analysis.sh <model-tag>
set -euo pipefail
TAG=$1
PY=../.venv/bin/python
export PYTHONPATH=v2/analysis
mkdir -p results/v2/logs
$PY v2/analysis/experiments_v2.py "$TAG" 2>&1 | tee results/v2/logs/experiments_$TAG.log
$PY v2/analysis/canon_v2.py "$TAG" 2>&1 | tee results/v2/logs/canon_$TAG.log
$PY v2/analysis/franchises_v2.py "$TAG" 2>&1 | tee results/v2/logs/franchises_$TAG.log
$PY v2/analysis/v1_rule.py "$TAG" 2>&1 | tee results/v2/logs/v1rule_$TAG.log
$PY v2/analysis/sensitivity_v2.py "$TAG" 2>&1 | grep -v "^  E\|^==" | tee results/v2/logs/sensitivity_$TAG.log
$PY v2/analysis/multiverse_v2.py "$TAG" 2>&1 | tee results/v2/logs/multiverse_$TAG.log
