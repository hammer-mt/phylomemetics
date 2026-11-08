#!/bin/bash
# Everything one model does for the two new corpora (v2/second/PLAN.md), in the plan's order. Run from study/ with
# RUNPOD_STATE pointing at a pod on which `pip install transformers==5.22.2 accelerate hf_transfer` has been run.
#
#   bash v2/second/run_pod.sh qwen3-32b                 every plane, every film
#   bash v2/second/run_pod.sh qwen3-8b
#   bash v2/second/run_pod.sh llama3.1-8b
#   bash v2/second/run_pod.sh llama3.3-70b primary small     the primary plane only; of the remakes, the smaller set of films
#
# PARTS chooses what a machine does (default: all of it). A model can be split over two machines to save time:
#   PARTS="check bgdrop probes holmes" bash v2/second/run_pod.sh qwen3-32b
#   PARTS="check remakes"              bash v2/second/run_pod.sh qwen3-32b        (on a second machine)
# Every machine starts with the check on texts that are not new.
#
# Each step keeps what it fetched, so a run that was cut off goes on from where it stopped. gpu.py refuses every step on
# new texts unless the plan, the code and the corpora are committed and pushed and the input files match their hashes.
set -euo pipefail
TAG=$1; shift
PY=../.venv/bin/python
G="$PY v2/second/gpu.py"
PARTS=${PARTS:-"check bgdrop probes holmes remakes"}
has() { [[ " $PARTS " == *" $1 "* ]]; }

# 1. Texts that are not new: the new pass must give v2's stored presence of the Batman works, bit for bit; then the
#    background plots that Holmes's base rates leave out.
if has check; then
  $G check "$TAG" | tail -14
  $PY -c "
import json
c = json.load(open('results/v2/second/pipeline_check_$TAG.json'))
assert c['after_the_freeze'], 'the check did not run from the frozen, pushed commit'
assert c['exact'], 'the new pass does not reproduce the stored presence of the Batman works: nothing else is run on this model'"
fi
if has bgdrop; then
  [ -f results/v2/second/bgdrop_check_$TAG.json ] || $G bgdrop "$TAG" | tail -2
  $PY -c "
import json
assert json.load(open('results/v2/second/bgdrop_check_$TAG.json'))['exact'], 'the left-out background plots are not reproduced exactly: run gpu.py bgfull $TAG and pass --recounted to holmes.py'"
fi

# 2. Recognition. H2 names the franchise, so it is asked after H1 and in a run of its own.
if has probes; then
  $G probes "$TAG" holmes_h1 remakes_r1 remakes_r2
  $G probes "$TAG" holmes_h2
  $PY v2/second/probes.py score "$TAG"
fi

# 3. Presence: Holmes, then the remake films and their co-presence counts.
if has holmes; then $G holmes "$TAG" "$@"; fi
if has remakes; then $G remakes "$TAG" "$@"; fi
echo "$TAG: done ($PARTS)"
