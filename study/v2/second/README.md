# Two more corpora

Paper v2 tested its method on 32 Batman works. This folder runs the same method on two corpora built to address what one franchise could not settle: the Sherlock Holmes stories with their screen versions, and film remakes. The plan is [`PLAN.md`](PLAN.md); it was frozen before any model read a new text. The scouting that chose the corpora is in [`preregistration/second_corpus/`](../../../preregistration/second_corpus/).

It was run on 8 and 9 November 2026 and is reported in the paper's version 2 draft (`../../paper/v2/paper.md`, Sections 2.11, 3.11 and 3.12). In short, for Qwen3-32B: on Holmes, heredity is supported, identity is not met (it is met in Llama 3.3 70B, for the works that model names) and selection is not detected; on the remakes, heredity is supported and the decision test meets all seven registered conditions, with the qualifications the paper sets out. The plan's Section 16 logs everything that changed after the freeze.

Run everything from `study/` with `../.venv/bin/python`.

## Files

| File | What it does |
| --- | --- |
| `PLAN.md` | The analysis plan: corpora, tests, how each outcome is read, what was known at the freeze |
| `wiki.py`, `plots.py` | Cached, paced access to Wikipedia; wikitext to plain plot sentences |
| `build_holmes.py` | The Holmes corpus: lists, texts, what each screen work is based on, years, sets (`data/second/holmes/`) |
| `names.py` | Name-free texts by the Batman corpus's procedure (`v2/prompts/name_remover.md`) |
| `build_remakes.py` | The remakes corpus: statements, readers, pairs, film texts (`data/second/remakes/`) |
| `probes.py` | The recognition questions and their scoring |
| `gpu.py`, `run_pod.sh` | What a GPU machine reads and returns; refuses to run unless the plan is frozen, committed and pushed |
| `../gpu/presence_pass.py` | Presence of every unit in every text, as packed bits, on a scale fixed beforehand |
| `../gpu/copresence.py` | Shared optional units between each remake and every film read |
| `../gpu/embed_pass.py` | The sentence-embedding comparator |
| `lib.py`, `holmes.py` | The Holmes tests (E1, E2, the residual test, E3); `holmes.py regress` reproduces v2's Batman numbers |
| `e3_power.py` | Power of the Holmes selection test with its covariates |
| `remake_test.py`, `remakes.py` | The remakes tests: heredity, and whether units add to words |
| `remake_null_size.py` | Pure noise in the units' place through the remakes decision test: how often it is rejected, and the noise-calibrated p |
| `confirmatory.py` | The five confirmatory tests side by side, with Holm's adjustment |
| `corpus_counts.py`, `paper_tables.py` | Every count the paper quotes about the corpora, and every table, written from the committed results (`corpus_counts.json`, `paper_tables.md`) |
| `../figs/holmes.py`, `../figs/remakes.py` | Figures 8 and 9 of the paper |

Data are in `study/data/second/`, results in `study/results/v2/second/`. There, `audit/` holds three exploratory audits of the remakes result that were made after it was seen (statistics, confounds, other word measures), each with its script; `review/` holds a reviewer's redraws of the calibration; `readers/` holds every prompt and answer of the AI readers that labelled the pairs. Presence bits and counts fetched from the GPU machines go to `study/v2/acts_second/` and are not committed.

## Order of work

1. Build the corpora (`build_holmes.py`, `names.py`, `build_remakes.py`). No model reads a text.
2. Words-only checks and calibrations (`holmes.py words`, `e3_power.py`, `remakes.py words`).
3. Freeze: commit and push the plan, the corpora and the code.
4. One machine per model: `bash v2/second/run_pod.sh <tag>`. The first job is a check on texts that are not new.
5. Analyse: `holmes.py run <tag>`, `remakes.py verify <tag>`, `remakes.py run <tag>`.
6. Where Holmes E1 is met, the plan's follow-up (Section 8), on a machine where `gpu.py check <tag>` has first been exact with the present `presence_pass.py`: `gpu.py after <tag>` (`gpu.py after <tag> again` where v2's stored after file of the Batman works is not here) reads the name-free texts again with the token after each code masked too, then `holmes.py followup <tag>`.
7. The confirmatory tests together, then the counts, tables and figures: `confirmatory.py`, `corpus_counts.py`, `paper_tables.py`, `../figs/holmes.py`, `../figs/remakes.py` (or `bash v2/make_paper_assets.sh`, which runs them after the Batman assets).
