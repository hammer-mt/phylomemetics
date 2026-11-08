# Two more corpora

Paper v2 tested its method on 32 Batman works. This folder runs the same method on two corpora built to address what one franchise could not settle: the Sherlock Holmes stories with their screen versions, and film remakes. The plan is [`PLAN.md`](PLAN.md); it was frozen before any model read a new text. The scouting that chose the corpora is in [`preregistration/second_corpus/`](../../../preregistration/second_corpus/).

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

Data are in `study/data/second/`, results in `study/results/v2/second/`. Presence bits and counts fetched from the GPU machines go to `study/v2/acts_second/` and are not committed.

## Order of work

1. Build the corpora (`build_holmes.py`, `names.py`, `build_remakes.py`). No model reads a text.
2. Words-only checks and calibrations (`holmes.py words`, `e3_power.py`, `remakes.py words`).
3. Freeze: commit and push the plan, the corpora and the code.
4. One machine per model: `bash v2/second/run_pod.sh <tag>`. The first job is a check on texts that are not new.
5. Analyse: `holmes.py run <tag>`, `remakes.py verify <tag>`, `remakes.py run <tag>`.
