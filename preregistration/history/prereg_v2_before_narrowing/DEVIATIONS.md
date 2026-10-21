# Deviations log

Every change after freeze F0 is recorded here (PREREGISTRATION.md section 10), newest last. Each entry gives:

- the date;
- the freeze point it follows;
- what changed and why;
- whether any TEST outcome had been seen;
- its effect on results.

Changes made before F0 (the v1 draft, the AI design review and the pilot) are recorded in PREREGISTRATION.md section 14, not here.

## Freeze points

| Freeze | What is committed and pushed | Commit |
|---|---|---|
| F0 | this folder: design, code, work lists, sources, protocols (after steps P1 and P2) | |
| F1 | `dev_freeze.json`, `sentence_units_<model>.npy` | |
| F1b | `results/bench.json` | |
| F2-T2 | `data/t2_stories.jsonl`, `data/t2_long.jsonl`, `data/t2_stories_zh.jsonl`, `agent/stories/`, `agent/translation/` (T2 part), `data/manifest.json` | |
| F2-T1b | `data/t1d_*`, `data/t1e_fresh_namefree.jsonl`, `data/t4_*`, `data/t1b_*`, `data/t3_pool.jsonl`, `data/t2n_works.jsonl` (texts), `agent/name_removal/`, `agent/drift/` | |
| F2-T1cd | `data/t1c_*`, `data/*_cc.jsonl`, `agent/translation/`, `agent/zhwiki/` | |
| F2-T3 | `data/t2n_works.jsonl` (annotated), `agent/annotation/` | |
| A7 | `data/human_audit.jsonl` | |
| F3 | `results/feature_manifest.json` | |

## Log

| Date | After | Change | Reason | TEST outcomes seen? | Effect |
|---|---|---|---|---|---|
| | | | | | |
