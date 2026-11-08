#!/bin/bash
# The follow-up ablation for one model (PLAN.md Section 18, change 33): the registered jobs again, with the model's
# reply begun for it. Run from study/ with RUNPOD_STATE pointing at a pod on which
# `pip install transformers==5.22.2 accelerate hf_transfer` has been run. The first job downloads the model's weights
# from the Hugging Face Hub to /workspace/hf (65.5 GB for Qwen3-32B, about 16.4 GB for Qwen3-8B). The jobs of
# the registered run (v2/cache/jobs/ablate_<tag>*.json and its partner file) and the model's scale
# (v2/acts/<tag>/scale.npy) must be on this machine: nothing is redrawn. The follow-up's job files are written by
#   ../.venv/bin/python v2/analysis/build_ablation_followup.py jobs
# and must match the hashes committed in results/v2/ablation_followup_jobs.json, as must the scale.
#
#   bash v2/run_ablation_followup.sh qwen3-32b      first: its check of the prefix chooses the prefix for both models
#   bash v2/run_ablation_followup.sh qwen3-8b
set -euo pipefail
TAG=$1
PY=../.venv/bin/python
J=v2/cache/jobs
export PYTHONPATH=v2/analysis:.
unset PHYLO_V2_ACTS          # the scoring must read the folder the outputs are fetched to

# the GPU machine gets the files in this folder: they must be the committed ones, and the log says which commit
DIRTY=$(git status --porcelain -- v2/gpu v2/analysis v2/run_ablation_followup.sh results/v2/ablation_followup_jobs.json v2/PLAN.md experiments.py discover.py data/corpus.jsonl data/corpus_neutral.jsonl)
if [ -n "$DIRTY" ]; then
  echo "uncommitted changes in the follow-up's plan, code, texts or job record: commit them first"; echo "$DIRTY"; exit 1
fi
COMMIT=$(git rev-parse HEAD)
PUSHED=$(git rev-parse '@{u}')
if [ "$COMMIT" != "$PUSHED" ]; then echo "this commit is not pushed: push it first"; exit 1; fi
echo "commit $COMMIT, pushed"

# a job is run once: an output that is already here is read again, not made again, so a run that was cut off goes on from where it stopped
ask() { [ -f "v2/acts/$TAG/$2.json" ] || $PY v2/gpu/remote.py ask "$TAG" "$@" | tail -3; }

$PY - <<EOF
import hashlib, json
rec = json.load(open("results/v2/ablation_followup_jobs.json"))
for name, sha in rec["sha256"].items():
    assert hashlib.sha256(open("$J/" + name, "rb").read()).hexdigest() == sha, name + " is not the file that was recorded"
assert hashlib.sha256(open("v2/acts/$TAG/scale.npy", "rb").read()).hexdigest() == rec["scale_sha256"]["$TAG"], "the scale is not the file that was recorded"
print(len(rec["sha256"]), "job files and the scale match the committed record")
EOF

# the pod must run this commit's ablate.py: an older copy ignores the prefix and would score the wrong place
$PY v2/gpu/remote.py sync -
$PY -c "
import sys; sys.path.insert(0, 'v2/gpu')
import runpod_ctl as rc
rc.put('$J/ablate_${TAG}_partners.npz', 'data/jobs/ablate_${TAG}_partners.npz')
rc.put('v2/acts/${TAG}/scale.npy', 'out/${TAG}/study/scale.npy')"

# 1. The prefix is checked for form on texts outside the test: the 69 ordinary films, nothing switched off.
#    "Hero:" first; "Answer:" only if it fails; if both fail the follow-up is not run.
if [ "$TAG" = qwen3-32b ]; then
  for slug in hero answer; do
    ask $J/begun_form_$slug.json begun_form_$slug
    OUT=$($PY v2/analysis/causal.py begun-form "$TAG"); echo "$OUT" >&2
    CHOSEN=$(echo "$OUT" | tail -1 | cut -d' ' -f2)
    if [ "$CHOSEN" != pending ]; then break; fi
  done
fi
SLUG=$($PY -c "
import json
from build_ablation_followup import PREFIXES
d = json.load(open('results/v2/qwen3-32b/begun_form_check.json'))
assert d['decided'], 'the check of the prefix has not been made'
print({v: k for k, v in PREFIXES.items()}.get(d['prefix_chosen'], 'none'))")
if [ "$SLUG" = none ]; then echo "no prefix passed the check: the follow-up is not run"; exit 0; fi
echo "prefix: $SLUG"

# 2. The test: the registered jobs with the reply begun.
ask $J/ablate_${TAG}_begun_$SLUG.json ablate_begun
$PY v2/analysis/causal.py ablation "$TAG" | tail -2 | cut -c1-500

# 3. The two secondaries (20 sets each) and the descriptive control (all 114 name-free texts, nothing switched off).
for s in strong weak; do ask $J/ablate_${TAG}_begun_${SLUG}_$s.json ablate_begun_$s scale; done
ask $J/begun_probe_$SLUG.json recognise_hero_begun
$PY v2/analysis/causal.py ablation "$TAG" | tail -2 | cut -c1-500
$PY v2/analysis/causal.py begun-probe "$TAG" | tail -1 | cut -c1-700
echo FOLLOWUP_DONE "$TAG"
