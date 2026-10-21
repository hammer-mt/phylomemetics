# Deviations log

Every change after freeze F0 is recorded here (PREREGISTRATION.md section 10), newest last. Each entry gives:

- the date;
- the freeze point it follows;
- what changed and why;
- whether any TEST outcome had been seen;
- its effect on results.

Changes made before F0 (the v1 draft, the AI design review and the pilot) are recorded in PREREGISTRATION.md section 14, not here.

## Freeze points

prereg-v3 runs only the planted-meme stories (T2) with its cost and interpretability (T6); the freezes of the deferred tests (F2-T1b, F2-T1cd, F2-T3 in prereg-v2) are gone with them.

| Freeze | What is committed and pushed | Commit |
|---|---|---|
| F0 | this folder: design, code (with the audit page script), work lists, sources, protocols (after steps P1 and P2) | |
| F1 | `dev_freeze.json`, `sentence_units_<model>.npy` | |
| F1b | `results/bench.json` | |
| F2-T2 | `data/t2_stories.jsonl`, `data/t2_long.jsonl`, `agent/stories/`, `data/manifest.json` | |
| A7 | `data/human_audit_sample.json`, `data/human_audit.jsonl` (before any method, the Claude upper bound A4 included, runs on a TEST story) | |
| F3 | `results/feature_manifest.json` (the sha256 of every feature file, A4's included; `features/` itself is not committed; `analysis.py` re-checks every hash before each test runner) | |

## Log

| Date | After | Change | Reason | TEST outcomes seen? | Effect |
|---|---|---|---|---|---|
| | | | | | |
