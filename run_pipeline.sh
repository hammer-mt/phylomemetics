#!/bin/zsh
# Extracts activations (skipping models already done) and discovers memes per model, then runs the phylogeny.
set -e
cd "$(dirname "$0")"
ACTS=${ACTS:-activations}
mkdir -p logs results
for m in qwen3-8b llama3.1-8b qwen3-1.7b; do
  if [ ! -f $ACTS/$m/meta.json ]; then
    .venv/bin/python extract.py $m $ACTS > logs/extract_$m.log 2>&1 || { echo "FAILED extract $m"; tail -5 logs/extract_$m.log; exit 1; }
    echo "extracted $m: $(tail -1 logs/extract_$m.log)"
  fi
  .venv/bin/python discover.py $m $ACTS results > logs/discover_$m.log 2>&1 || { echo "FAILED discover $m"; tail -5 logs/discover_$m.log; exit 1; }
  echo "discovered $m: $(grep 'meme families' logs/discover_$m.log)"
done
.venv/bin/python phylo.py $ACTS results > logs/phylo.log 2>&1 || { echo "FAILED phylo"; tail -8 logs/phylo.log; exit 1; }
echo "phylo done"
echo PIPELINE_DONE
