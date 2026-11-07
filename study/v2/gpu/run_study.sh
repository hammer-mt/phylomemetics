#!/bin/bash
# On the GPU machine, after the reference pass and the supplement pass: the study pass, the scale, the background
# base rates and the comparison franchises for one model.
#   bash run_study.sh <model-tag>
set -euo pipefail
TAG=$1
cd /workspace/code
python3 study_pass.py "$TAG" /workspace/data/study /workspace/out 2>&1 | grep --line-buffered -v 'Loading weights\|Fetching'
python3 finalize_scale.py "$TAG" /workspace/out
python3 background_pass.py "$TAG" /workspace/data/reference.jsonl /workspace/out --exclude-docs 843 2>&1 | grep --line-buffered -v 'Loading weights\|Fetching'
python3 background_pass.py "$TAG" /workspace/data/franchises.jsonl /workspace/out --src franchise --name franchises 2>&1 | grep --line-buffered -v 'Loading weights\|Fetching'
ls -la /workspace/out/"$TAG"/study /workspace/out/"$TAG"/background /workspace/out/"$TAG"/franchises
echo STUDY_DONE "$TAG"
