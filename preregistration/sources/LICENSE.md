# Sources and licences

These files are the scout caches the work lists are built from (`build_lists.py lists`), copied into the project on 2026-10-21 so the TEST texts can be rebuilt from the public repository after the session scratchpad is gone. Copying them computed no outcome.

- `plays.jsonl.gz`, `films_all.jsonl.gz`, `shakespeare_adaptations.jsonl.gz`, `shakespeare_wikidata.json`: Shakespeare plays and film adaptations (plot sections, metadata, Wikidata P144/P941 links), fetched 2026-10-20.
- `t3_candidates.jsonl.gz`, `t3_pool_eligible.jsonl.gz`, `t3_random_films.jsonl.gz`, `t3_pool_summary.json`: the T3 category pool and random films.
- `relatives_documented_relatives.json`, `relatives_works.jsonl.gz`, `relatives_wikidata.json`, `relatives_wikipedia_evidence.json`: documented Batman relatives and their works.
- `neutral_motif_frame.json`, `neutral_motif_walk.json`: the seeded draw of the 10 neutral motifs (Wikipedia category members fetched 2026-10-21) and the reasons for every rejection.
- `robust_agreement_analysis.json`, `robust_agreement_units_m.npz`, `robust_agreement_lexical_null_units.npz`: the robustness-review files `dev_units.json` is built from (DEV only).
- `scout_manifest_2026-10-20.json`, `*timing*.json`, `devcheck_cost_neurons_qwen3-8b.json`: provenance and the timing measurements on made-up text and three DEV works.

## Wikipedia and Wikidata

Plot sections, leads and evidence sentences are from English Wikipedia, licensed under the [Creative Commons Attribution-ShareAlike 4.0 licence](https://creativecommons.org/licenses/by-sa/4.0/). Attribution: Wikipedia contributors. Each source article is named by its title (`title` field) and can be found at `https://en.wikipedia.org/wiki/<title>`. These files, and every TEST text derived from them, are distributed under CC BY-SA 4.0. Wikidata content is CC0.
