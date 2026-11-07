#!/bin/bash
# Rebuild everything the paper shows from the committed results: summaries, tables and figures. Run from study/.
#   bash v2/make_paper_assets.sh
set -uo pipefail
PY=../.venv/bin/python
export PYTHONPATH=v2/analysis
$PY v2/analysis/pass_summary.py | tail -1
$PY v2/analysis/contrasts.py | tail -1
$PY v2/analysis/confirmatory.py | tail -7
$PY v2/analysis/paper_tables.py
for f in contrast canon genome identity heredity explanations ablation; do
  if [ -f v2/figs/$f.py ]; then $PY v2/figs/$f.py | tail -1; fi
done
