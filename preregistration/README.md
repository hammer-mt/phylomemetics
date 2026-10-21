# Pre-registration: planned experiments, not results

**Status: planned, not run (paused 2026-10-21).** This folder holds the plan, protocols, code and work lists for the next experiments. It contains no results. None of the planned experiments has been run, and no TEST data exists. The only things run so far are design work: the scout downloads of Wikipedia and Wikidata metadata and plot sections, the power simulations, timing checks on made-up text and three DEV works, and `test_analysis.py` on synthetic data.

From the commit that first added this folder to the public repository, its GitHub timestamp precedes any test data. The plan can still change before freeze F0 (each change logged in `PREREGISTRATION.md` section 14). The F0 commit is the one that locks it.

Mike decided on 2026-10-21 to keep the next experiment planned and come back to it.

The finished Batman study, whose corpus and results are this plan's DEV data, is in [`../study/`](../study/). The scripts here read those files from that folder.

## What is planned

The planted-meme stories experiment (`PREREGISTRATION.md`, version prereg-v3). Short stories in genres unrelated to Batman each have memes planted in them, in several versions: explicit, with the meme's usual keywords banned, implicit, with invented code words, and a matched control and decoy. Every method must then say which memes each story contains. The ground truth is known by construction.

Primary claims (each with a stated way to fail):

- **H1.** On reworded stories, Qwen3-8B neurons detect memes better than an evasion-aware keyword analyst and a lexical classifier.
- **H2.** The neurons are within 0.05 AUC of the strongest embeddings (Qwen3-Embedding-4B and multilingual-e5-large).
- **H3.** The 11 concept units found on the Batman texts respond specifically to their own meme in unrelated genres.
- **H6** (decision rule). No rival method is both at least as accurate and no more expensive per story.

Reported secondaries include the code-word condition (the pizzagate case: a group changing its vocabulary to evade keyword moderation) and the overlap analysis (what each method finds that the others miss).

## Where it got to

- Designed and reviewed: v1, then four adversarial AI reviews (v2), then narrowed to planted stories and reviewed again (v3). `test_analysis.py` passes all 151 checks.
- The approved models and packages are installed on Mike's laptop (about 49 GB in the Hugging Face cache); the packages are pinned in the root `requirements.txt`.
- Mike's book is not cited anywhere in this folder outside `history/`.
- The "Why" paragraph in section 14.4 now records Mike's decision without inventing reasons.

## Cost, as reviewed

- **As designed:** about 4,400 agent sessions, about 25 hours of local model time (mostly LLM reading), about 3 days elapsed, and about 3 hours of Mike's time (a 15-minute pilot timing and a 2 to 2.5 hour audit).
- **Trimmed option** (proposed, not yet chosen): defer the nameability label test (about 1,300 sessions) and the long-text condition; run LLM reading as one direct judge plus recursive summarisation on Qwen only; overlap the DEV feature extraction with story writing. About 1.5 days, mostly unattended.

## To resume

Run every command from the project root.

1. Choose the full or trimmed version; record the choice in `PREREGISTRATION.md` section 14 and `DEVIATIONS.md`.
2. Run P1 (example sentences) and P2 (pilot, with Mike timing at most 5 audit stories), then `.venv/bin/python preregistration/test_analysis.py`.
3. F0: commit and push this folder, with the P1 and P2 outputs, before generating any TEST data, and note the commit hash in `DEVIATIONS.md`.
4. Continue with the run plan in `PREREGISTRATION.md` section 3: DEV features, F1, stories (F2-T2), Mike's audit (A7), TEST features (F3), analysis.

## Files

- `PREREGISTRATION.md`: the plan; `analysis.py`, `readers.py`, `extract_compact.py`, `bench.py`, `build_lists.py`, `audit_page.py`: frozen code; `test_analysis.py`: checks on synthetic data.
- `DEVIATIONS.md`: the freeze points and the log of every change after F0 (empty so far).
- Protocols: `planted_story_protocol.md`, `annotation_protocol.md` (and the deferred drift, translation and name-removal protocols).
- Work lists: `memes_for_tracing.json`, `planted_design.json`, `dev_units.json`, and the deferred lists for later phases.
- `sources/`: gzipped copies of the scout caches the work lists are built from, with their licences (`sources/LICENSE.md`).
- `history/`: earlier drafts, power simulations, `prereg_v2_before_narrowing/` (the v2 folder), and `design_workspace/` (the scout's scripts and small metadata files, and the full design and narrowing outputs with every review). The files in `history/` keep their original text, so their paths still say `prereg/` and point at the study files as they were before the repository was reorganised.
- Not committed (`.gitignore`): the scout's raw Wikipedia pages and bulk caches under `history/design_workspace/scout/` (about 126 MB; the scripts in `history/design_workspace/scripts/` fetch them again from Wikipedia, once their hard-coded scratchpad paths are changed, though Wikipedia will have changed since 2026-10-20), and the model features and activations the plan will write.
