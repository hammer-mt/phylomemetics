# What the readers were asked, and what they answered

Two kinds of label in the second corpora come from AI readers who saw Wikipedia's wording and nothing else (`study/v2/second/PLAN.md`, Sections 2 and 3). This folder keeps every prompt and every answer, as they were.

- `holmes_sources/`: what each Sherlock Holmes screen work is based on. `r1_*` and `r2_*` are the two readers (each read every article, in a different order); `r3_*` is the third reader of the two works they disagreed on. `*.prompt.txt` is the prompt with the articles (plots left out), `*.out.json` the answer. `build_holmes.py links` turns them into `data/second/holmes/works.json`.
- `remake_statements/`: what each "remake" sentence says. `statements.jsonl` is every sentence found in the opening of a film's article, with the films it links to; `r1_*`, `r2_*` and `r3_*` as above (a third reader for the 9 statements read differently). `build_remakes.py build` turns them into `data/second/remakes/pairs.json`.

The name-free Holmes texts come from a third kind of agent, a rewriter; its rewrites are in `data/second/holmes/anon/` and its prompt is `study/v2/prompts/name_remover.md`.
