# a4-prereg: audit of preregistration/

All paths are under `preregistration/` unless a longer path is given. Nothing was run for this audit; I read files and parsed JSON only. "DEV" = the finished Batman study; "TEST" = data the plan would create.

## Bottom line

- **Nothing has been run.** The folder is a plan (prereg-v3), paused by Mike on 2026-10-21 before freeze F0. `DEVIATIONS.md` has no commit hashes and an empty log; `memes_for_tracing.json` says `examples_status: not yet written (step P1)`; there is no `agent/`, `data/`, `results/` or `features/` folder. One git commit touches the folder in this clone (268731e, 2026-10-21). It is public but not locked: the plan can still change until F0.
- **What is planned is one experiment**: 816 Claude-written stories (136 plots x 6 versions) with 3 of 34 memes planted in each, and every method asked which memes each story contains. Primary claims H1 (neurons beat evasion-aware keywords and a TF-IDF logistic classifier on reworded stories), H2 (neurons within 0.05 AUC of the best embeddings), H3 (11 Batman-found concept units are specific to their own meme in other genres), H6 (no cheaper rival is as accurate).
- **Everything about lineage, language, culture and cross-model agreement was deferred**: drift (H4), Shakespeare (H5), Chinese translation, name removal, the 2,544-work culture pool (T3), mutation and selection (T4), concept agreement between models (T5), natural annotated summaries (T2-N). Mike chose the narrow scope from three options and gave no reason (`PREREGISTRATION.md` 14.4). The design's own numbers that fit the choice: H4 and H5 had power 0.15 and 0.17 at their smallest effect of interest (`history/power_v2.json`), and v2 cost about 7,300 agent sessions.
- **The design is underpowered at its own smallest effect of interest.** `history/power_v4.json` as quoted in section 6: H1 0.22 to 0.40 at a gap near 0.04 to 0.05 AUC; H2 0.22 when neurons exactly equal the vectors; H3 0.14 at specificity 0.05 (`history/power_v3.json`). The likely verdict for a small true effect is "undetermined" on all three, after about 4,430 sessions and 3 days.
- **Five adversarial AI reviews found 8 to 10 blockers in v1 (the plan says 8; the review JSON holds 10 blocker entries) and 10 majors in v3**, and the fixes are real (twin controls, stronger rivals, plot-level tests, audit gates). What the reviews could not fix: every story, example, ban list, reader and rater is Claude; the only human check is the author's own 2 to 2.5 hour audit; frontier-LLM reading is excluded as a rival because Claude wrote the test.
- **P2 has no live test in the plan.** Llama is a "replication" of the method (same test, p < 0.05), not a test that the two models share units. The one test of that (T5, S20) is deferred, and the DEV evidence already in `sources/robust_agreement_analysis.json` is weak (below).

## Findings

### 1. Status and history

| Version | Date | What | Source |
|---|---|---|---|
| v1 draft | 2026-10-20 | 5 primaries H1 to H5, 24 memes, 120 plots, 5 versions, 21 secondaries S1 to S21, 62 synthetic checks | `history/PREREGISTRATION_draft_v1.md`; `history/design_workspace/design_result.json` (`draft`) |
| v2 | 2026-10-21 | After 4 adversarial reviews (all "needs changes": 8 blockers per `PREREGISTRATION.md` 14.2; my count from the JSON is 3+3+2+2 = 10 blocker entries, several duplicates of the same H4 problem). 6 claims H1 to H6, 34 memes, 136 plots, 6 versions, 107 checks | `history/prereg_v2_before_narrowing/`; `design_result.json` (`reviews`, `final`) |
| v3 | 2026-10-21 | Mike narrows to planted stories. Fifth review: "needs changes", 0 blockers, 10 major, 13 minor. 151 checks (30 of them keep deferred code alive) | `PREREGISTRATION.md` 14.4, 14.5; `history/design_workspace/narrow_result.json` |
| Paused | 2026-10-21 | "Mike decided on 2026-10-21 to keep the next experiment planned and come back to it" | `README.md` |

Done so far (design work only): scout downloads of Wikipedia/Wikidata metadata and plot sections (2026-10-20), power simulations, timing checks on made-up text and three DEV works, `test_analysis.py` on synthetic data, model and package downloads on Mike's laptop (about 49 GB in the Hugging Face cache) (`README.md`).

### 2. What is planned: each hypothesis with its falsifier (prereg-v3)

Models: Qwen3-8B primary, Llama 3.1 8B replication, both 4-bit MLX. A "neuron" is one SwiGLU channel split by sign: 884,736 units in Qwen, 917,504 in Llama. Sentence value = max over tokens; story score = mean of the 3 strongest sentences (`PREREGISTRATION.md` 2.1). "Twin AUC" = each planted story against its own plot's control version (same frame word for word, a different event of equal stakes in the slot).

| | Claim | Statistic | Falsifier |
|---|---|---|---|
| **H1** | On reworded stories (banned + implicit versions) neurons beat `keywords:expanded-qwen3-8b` (Qwen as evasion-aware analyst, 30 to 40 terms, one revision, WordNet) **and** `keywords:logreg` (L2 logistic regression on TF-IDF uni/bigrams) | Mean over memes of twin-AUC difference; intersection-union (largest p of the two rivals); meme-level and plot-level sign flips both required; Holm over H1 to H3; the 10 neutral memes must show the same sign. SESOI 0.05 AUC | Falsified if the test of H0 "difference >= 0.05" rejects at 0.05 against either rival at both levels while the Holm test does not reject. Both reject = "supported, below SESOI". Reading if falsified: "the fuzziness claim does not separate neurons from good keyword practice" |
| **H2** | Neurons not worse than `emb:qwen3emb-4b`, `embwin:qwen3emb-4b` (3-sentence windows) and `emb:e5-large` by more than 0.05 AUC; then superiority as a fixed sequence | Same, intersection-union over 3 rivals, margin 0.05; neutral-subset difference above -0.05 | Falsified if neurons are credibly worse than any one of the three by more than 0.05. Reading: the case for neurons then rests on interpretability, cost and overlap, not accuracy |
| **H3** | The 11 concept units found on DEV Batman text respond specifically to their own meme in unrelated genres | Mean over units of [twin AUC on own meme minus mean twin AUC on every other meme]; null permutes meme labels across units. SESOI 0.05 | Falsified if the falsification test excludes specificity >= 0.05: "they fire on any eventful passage" |
| **H6** | Decision rule, outside Holm: no rival is both at least as accurate (its one-sided 95% lower AUC bound >= the neurons' estimate on shared memes) and no more expensive per story at 34 memes | `analysis.run_cost`, `analysis.h6_rival` | "Dominated" if any rival meets both. Not counted as rivals: either model's neurons and their variants, DEV units, length-residualised neurons, and the Claude-based entries |

Verdict vocabulary (section 5): supported; supported below SESOI; falsified; undetermined; inconclusive (a manipulation check failed); not run (fewer than 100 fully valid plots or fewer than 28 memes kept).

Gates that turn a claim "inconclusive" (section 7): DEV reproduction (compact-extractor E2 within 0.03 of 0.164 Qwen and 0.102 Llama, else stop before TEST); the author's audit (one-sided 90% Clopper-Pearson lower bound: presence >= 0.75 on banned+implicit cells, absence >= 0.80 on control+decoy cells, at most 10% "unsure", at least 4 of 5 blocks); fallback to the cross-family reader (Gemma 2 9B); the plain Qwen analyst's explicit twin AUC >= 0.75.

Secondary analyses with stated predictions:

- **S-coded** (prominent family): on versions where the meme's key idea is renamed with an invented code word, context-using methods (neurons, hidden states, direct judge) beat context-free ones (sentence embeddings, evasion-aware keywords). 6 Qwen tests; the headline is stated only if all 6 pass with q < 0.05 and the coded audit gate passes.
- **Overlap O1/O2** (prominent family): which planted memes each method finds that another misses, at a fixed 10% control false-positive operating point with a twin-consistent detection rule. O1 predicted (neurons find memes the two keyword rivals miss; 4 tests). O2 no prediction against embeddings, hidden states and the judge (20 tests).
- **T2 family (7 tests)**: S8 the direct LLM judge beats the neurons (the plan *predicts* the LLM wins); S9 summary judge below direct judge; S10 neurons lower on implicit than banned; S11 plain keywords rank decoys above controls (construction check); m1 versus m5 ("single neurons are memes" claimed only if m5 minus m1 has an upper bound below 0.03 AUC); neurons versus the same model's hidden states (does a win come from the neuron basis or the dense representation); T2-long (recursive summarisation loses more on 48 long texts).
- **T6 (12 tests)**: blind label test of how nameable units are (Qwen neurons, Llama neurons, LDA topics, embedding clusters; random and best-of arms; S21 neurons beat a word-overlap floor); cost recipes and the cost-accuracy frontier.

Methods compared (section 2): neurons; keywords (plain analyst, evasion-aware analyst, TF-IDF top terms, TF-IDF + WordNet, logistic regression, a Claude ban-list-aligned query); vectors (Qwen3-Embedding-4B, e5-large, e5-base, LaBSE, MiniLM; sentences, windows, documents; prototype or probe); the model's own hidden states at 4 layers; topics (LDA, BERTopic-style); LLM reading (per-meme judge, shared-sheet judge, one-pass judge, Qwen3-1.7B judge, summary judge, DeBERTa NLI); Claude judge as a labelled upper bound; a length floor.

The 34 memes (`memes_for_tracing.json`): Tier A 10 (orphan-vigilante family), Tier B 10 (Batman story memes), Tier C 4 (general), Tier N 10 neutral motifs drawn by seed from Wikipedia motif categories. 18 have a DEV-labelled Qwen unit (`dev_units.json`: 11 concept, 4 name, 3 word).

### 3. What was deferred or cut, and why

Mike's decision of 2026-10-21 deferred all of these together; he gave no reason (14.4). The "why" column is what the folder's own reviews and power figures say about each, not Mike's reasoning.

| Item | What it was | State of the material | What the folder says against it |
|---|---|---|---|
| **Drift (T1b, H4)** | Rewrite 27 Batman + 44 Shakespeare-film texts at 25/50/75/100% of content words replaced plus a style-only control; neurons vs TF-IDF and LSA at levels 50 to 100 over 21 clusters | `drift_protocol.md`, `analysis.run_t1`; deferred banner | v1 statistic rewarded starting badly: an equally robust neuron method passed in 14 to 43% of the reviewers' simulated runs and a chance-level one in 100% (14.2). Fixed in v2, but power only 0.15 at SESOI, 0.44 at 0.10 (`history/power_v2.json`). Confounds named by review 3: Claude rewrites vs Wikipedia originals; the rewriter may leak source words. About 850 sessions (v2 section 11) |
| **Translation (T1c, T2-zh)** | Batman lineage and reference films into Simplified Chinese, comics kept in English (S3, S5); 272 planted stories in Chinese (S12); a natural zh.wikipedia arm | `translation_protocol.md`; deferred | Chinese is not an officially supported Llama 3.1 language; translators may re-introduce canonical names (review 2, 3); X-codes are the only shared tokens, so a neuron win could be cast structure (review 3). About 850 sessions |
| **Name removal** | Agent rewrite only to find names, then a diff replaces them with X-codes; needed by T1d, T1e, T4 and the zh natural arm | `name_removal_protocol.md`; deferred | About 720 sessions. A deterministic masker (for T3) hides 91% of the tokens the agent pass replaced at 97% precision, missing 129 of 13,337 (`name_removal_protocol.md`) |
| **Shakespeare (T1d, H5)** | 40 plays, 177 eligible films, 141 primary, 83 renamed; H5 sample cut to 54 films of 13 plays with lead-sentence or Wikidata P144 evidence | `works_shakespeare.json`, `sources/plays.jsonl.gz`, `sources/shakespeare_adaptations.jsonl.gz` | Ground truth weaker than assumed: 27 of the 83 "loose" films rested on a Wikipedia category only (review 3); Romeo and Juliet supplies 32 of 83; 3 plays have one film. Power 0.17 at SESOI. 21 of 40 plays hit the 40-sentence cap in v1 (review 4) |
| **Culture pool (T3; scout folder `t3/`)** | 2,544 masked works / 62,879 sentences (pool 2,312, contested 115, relatives 57, Batman lineage 39, orphan family 21); built from a 21-root Wikipedia category crawl (3,864 articles fetched, 2,416 eligible; the 1,291 vigilante-only works cut to a 400-work hash subsample), 1,000 random films and 18 relatives outside the pool; do neurons rank documented Batman relatives above the pool (S14 to S17) | `culture_pool.json`, `documented_relatives.json` (36 entities), `sources/t3_*.gz`; deferred | v1 negatives contained Batman stories (about 50 with Batman-family characters), franchises treated as independent works (null-world rejection 0.22 for the work-level test vs 0.045 entity-level), several relatives are iconography-only (review 1, 3). The Count of Monte Cristo link to Batman is undocumented (scout). S16/S17 rest on 3 named characters each |
| **T5 concept agreement** (uses the T3 random films) | Do 27 Qwen concept units have Llama partners that agree on 1,000 held-out random films better than 81 lexically matched null units (S20) | `dev_units.json` (`t5_concepts`), `sources/t3_random_films.jsonl.gz`, `analysis.run_t5`; deferred | v1 held-out corpus was enriched for the concepts (review 1); fixed by using the random films |
| **T4 mutation and selection** | Do units new in The Dark Knight Returns, Year One, The Killing Joke get carried forward more than chance (S18); more selection than words (S19) | `works_batman_extended.json` (32 DEV + 39 TEST works), `analysis.run_t4`; deferred | Circular: the sample adds direct adaptations of the predicted innovators; 42 of 43 new works are 1995 or later (review 1, 3). Forecasting is explicitly not wanted |
| **T2-N natural annotated summaries** | 237 masked natural summaries annotated exhaustively by two Claude annotators + adjudicator (S13) | `annotation_protocol.md` A2; deferred | Claude annotators will recognise masked Batman and Zorro works and label from memory (review 3). About 600 sessions |
| **T1e fresh Batman adaptations** | 7 animated adaptations of DEV comics as a positive control (S7) | in `works_batman_extended.json`; deferred | Wikipedia editors copy plot sentences between adaptation and source (review 3): an 8-gram reuse audit was added |

Cut outright or never adopted (14.3, `design_result.json` `rejected_suggestions`, scout notes): Spanish as a "near" language (scout offered it, v1 dropped it); Hindi (5.21x English tokens for Qwen); Qwen3-Embedding-8B (does not fit 24 GB); listwise LLM lineage reading; drifting the 69 reference films; GPT or Gemini as second reader (no API, stories would leave the machine); annotated public-domain novels for the long-text condition; a cross-fitted "update arm" on TEST; LLM pair questions in both orders; embeddings inside H5; holding T4 to forecasting standards; the 10 prose-novel Shakespeare adaptations the scout noted as a possible second medium (never designed).

### 4. What the reviewers objected to

Round 1, four reviews of v1 (`design_result.json`; counts by severity per review: 3/11/6, 3/15/5, 2/12/6, 2/7/7 blocker/major/minor). The objections that matter most for the research programme:

1. **H4 passed when neurons recovered nothing** (all four reviews).
2. **T2 confounded "this meme" with "something dramatic happens"**: a non-specific detector scored AUC 0.56 to 0.72 for every meme (0.59 in the null world under the draft rule, 0.50 under the twin rule).
3. **H1 was close to true by construction**: 203 of 205 draft Qwen keyword terms were multi-word, ban-list overlap 0.90, only 23 terms could ever fire in a banned story (14.2).
4. **Unequal supervision**: neurons searched about 0.9M units against a contrastive background; rivals got plain prototypes and defaults.
5. **Rivals were not the strongest**: LSA beat TF-IDF on DEV (0.064 vs 0.110) but H4/H5 used TF-IDF only; H2 used e5-base only.
6. **The meme set was chosen with the neurons in view**: 18 of 24 v1 memes came from DEV-labelled units; examples paraphrased Batman scenes. Fixed with 10 seeded neutral motifs and blind example writers.
7. **Unit selection picks word detectors**: review 1 cites the robustness review's finding that the funeral unit correlates r 0.977 with the word "funeral". A null would not separate "no idea-tracking units exist" from "the selection rule missed them".
8. **Selection and scoring contexts differ**: units chosen on one-sentence documents, scored inside 18 to 30 sentence stories (review 4). Answered with the `:m1ctx` variant.
9. **Length confound for max-pooled neurons**: reworded fills are wordier and a max over tokens rises with length (review 2). Answered with length matching and a length floor.
10. **Validation is a Claude monoculture and the manipulation check was circular** (review 2, 3). Answered with Gemma 2 9B and the author's audit.
11. **The cost comparison was circular**: a neuron pass is the same forward pass as the judge's prefill (0.36 s vs 0.36 s), so "cheaper above about 1 meme" is arithmetic, not a finding; storing every unit costs 1.8 MB per story against about 1.5 KB for a mean embedding (review 1, 4).
12. **"Rewording" was synonym paraphrase, which embeddings are trained for**; the real case is opaque coined terms (review 2). This produced the coded condition.
13. **The human-effort claim cannot be tested**: the keyword choice is a 1-second LLM call, and making neuron memes nameable costs labelling calls not charged to neurons (review 4).
14. **Budget undercounted 1.6 to 1.9x; sources lived only in a scratchpad** (review 4).
15. **"Falsified" on any non-significant result** with low power (review 1).

Round 2, one review of v3 (`narrow_result.json`): audit blinding leaked story ids such as `P012-banned`; audit counting double-counted changed answers; the absence gate failed 43% of the time at a true 95% agreement; operating points were unequal under ties, which biased O1 toward neurons (realised control FPR 0.000 for the keyword query vs 0.096 for the rest); the overlap analysis dropped the twin control; H6 counted Llama's neurons as a rival of Qwen's; the BH family had 200 tests in the synthetic run; the coded condition had no check that a code term existed; power for H1/H2 was overstated (single rival with no signal); Mike's time was not fixed. All recorded as fixed in 14.5 with a test each.

Still open after the reviews (stated in the folder): the benchmark "measures memes as Claude writes them"; coded versions are Claude inventing code words inside single stories, "real groups' code words build up across many texts" (section 13); results apply to texts of up to about 40 sentences; whether "do not name the book anywhere" covers `study/conversation/` files is "Mike to decide before F0" (14.5).

### 5. Design-time measurements that exist

Timings (`history/design_workspace/scout/tools/`, copied to `sources/`). All on made-up text, Mike's M-series 24 GB laptop, with another CPU job running:

| Measurement | Qwen3-8B | Llama 3.1 8B | File |
|---|---|---|---|
| Neuron pass, 24-sentence story (about 483 tokens), all units, warm | 0.354 to 0.358 s; 67.8 sentences/s; peak 5.65 GB | 0.356 to 0.363 s; 66.1 sentences/s; peak 5.6 GB | `neuron_timing_<model>.json` |
| Per-story aggregate for all units, fp16 | 1,769,472 bytes | 1,835,008 bytes | same |
| Judge, no cache, per meme | 0.421 s | 0.421 s | `llm_timing_<model>.json` |
| Judge, story cached: prefill + per meme | 0.362 s + 0.122 s (25 memes 3.42 s) | 0.359 s + 0.116 s (3.26 s) | same |
| One call asking 25 memes | 2.448 s; answered "yes" to 19 of 25 in the sample | 3.82 s | same |
| Flat summary; 2-level recursive summary | 2.625 s; 6.71 s | 2.477 s; 6.06 s | same |
| Compact extractor on real DEV text with hidden states | 90 sentences, 2,342 tokens, 2.36 s (about 38 sentences/s), peak 6.6 GB | not in folder | `sources/devcheck_cost_neurons_qwen3-8b.json` |

Encoders (`scout/embed_arch_check.json`, `scout/tools/timing.json`; randomly initialised architectures, so speed only): e5-base 443 sentences/s at 64 tokens on MPS, e5-large 134, LaBSE 432, MiniLM-L6 3,030. Cheap methods on 4,000 synthetic documents (140,000 sentences): TF-IDF 0.48 s, LSA-300 2.22 s, LDA 6.6 to 80 s, HDBSCAN on 140,000 points 46.7 s.

Derived by me from those files, not stated as such in the folder: at 34 memes the cached per-meme judge costs about 0.36 + 34 x 0.12 = 4.5 s per story against 0.36 s for one neuron pass (about 12 to 13x); e5-base would take about 0.05 s for 24 sentences (about 7x cheaper than the neuron pass). The v1 draft summary itself says neurons were "expected to cost about 13 times more than e5 embeddings".

Language checks (`scout/lang_check*.json`), made-up sentences only:

| Language | Tokens vs English, Qwen | Llama |
|---|---|---|
| zh | 0.96 | 1.28 |
| es | 1.53 | 1.53 |
| fr | 1.54 | 1.55 |
| ja | 1.58 | 1.54 |
| ru | 1.63 | 1.49 |
| de | 1.70 | 1.68 |
| th | 1.89 | 1.75 |
| hi | 5.21 | 2.96 |

Every language round-trips and maps onto sentence spans; X-codes tokenise as "X" + digit in every script. Comprehension on one made-up 24-sentence story, six questions, scored by the scout agent itself: both models 6/6 in English; Qwen about 5/6 in Chinese (missed that the mother's death was deliberate); Llama 5/6 in Chinese (got the merchant's escape wrong) and 6/6 in Spanish with Spanish words leaking into English answers. This is one story and one self-scoring agent: it shows the pipeline works, not how well the models read Chinese.

Other measurements: compact extractor matches existing DEV scores at r = 0.99997 to 0.99998 (`design_result.json` draft summary); scout counts for Shakespeare (`scout/shakespeare_summary.json`), the culture pool (`scout/t3/pool_summary.json`; 3,864 articles fetched; 1,000 random films accepted from 2,790 tried out of a population of 169,476) and 36 documented relatives; three power simulations (`history/power_v2.json`, `power_v3.json`, `power_v4.json`); null-world type-I rates in `test_analysis.py` (reported in 14.2: plot dependence gave rejection 0.105 for a non-specific detector, 0.03 after adding the plot-level test).

Cross-model agreement already measured on DEV and committed here (`sources/robust_agreement_analysis.json`, the input to `dev_units.json`). Median correlation between a Qwen unit and its Llama twin:

| Set | n | Original text | Same twin, name-free text | Fresh partner, name-free | Lexically matched null units, name-free | Fresh minus null |
|---|---|---|---|---|---|---|
| top 200 meme families | 200 | 0.636 | 0.142 | 0.426 | 0.401 | 0.010, 107/200 wins, p 0.038 |
| labelled concept units | 27 | 0.621 | 0.343 | 0.492 | 0.375 | 0.077, 18/27 wins, p 0.023 |
| labelled name units | 63 | 0.711 | 0.111 | 0.346 | 0.390 | 0.009, 33/63 wins, p 0.55 |
| labelled word units | 11 | 0.833 | 0.772 | 0.772 | 0.411 | 0.314, 7/11 wins, p 0.10 |

From `dev_units.json` (`t5_concepts`): of the 27 concept units, 14 have a name-free fresh-partner correlation below 0.5 and 5 below 0.3.

H3's own inputs (`dev_units.json`, `t2_mapped`): of the 11 concept units H3 tests, 4 passed the DEV rewording criterion (secret identity, origin wound, funeral, morbid joke), 6 are marked false (wealthy vigilante, orphan ward, vat of chemicals, corrupt police, antidote race, revenge vow) and 1 is unmarked (mantle). For revenge, Qwen's unit failed its DEV test and Llama's was picked from 50 candidates.

### 6. Costs

| | Agent sessions | Local model time | Elapsed | Mike | Disk | Source |
|---|---|---|---|---|---|---|
| v1 draft (all tests) | about 3,400 (review 4: undercounted 1.6 to 1.9x) | about 27 GPU-hours LLM reading | not stated | none planned | about 60 GB | `history/PREREGISTRATION_draft_v1.md` section 11 |
| v2 (all tests) | about 7,300, about 30M tokens | about 35 h | 3 to 4 days | 2 to 3 h | about 90 GB | `history/prereg_v2_before_narrowing/PREREGISTRATION.md` section 11 |
| v3 as designed (planted stories only) | about 4,430, about 12.5M tokens | about 25 h (LLM reading about 13 h) | about 3 days | 15 min pilot + 2 to 2.5 h audit | about 70 GB | `PREREGISTRATION.md` sections 3, 11 |
| v3 trimmed (proposed, not chosen) | about 3,100 (drops the 1,310-session label test) | less (one judge + summaries on Qwen only) | about 1.5 days | same | not stated | `README.md` |

v3 session breakdown: examples 170, pilot 120, planted stories 2,690 (writers 190, reads 1,950, salience 550), ban-list-aligned query 34, Claude upper bound 110, label test 1,310. v2 breakdown of the deferred parts: name removal 720, drift 850, translation 850, T2-N 600, relation typing 80.

### 7. What is needed to resume

From `README.md` and `narrow_result.json` (`run_plan`):

1. Mike chooses full or trimmed; record it in `PREREGISTRATION.md` section 14 and `DEVIATIONS.md`.
2. Mike decides whether the no-book rule covers `study/conversation/export.py`, `conversation.json`, `conversation.js` and the `study/.gitignore` comment (flagged "Mike to decide before F0").
3. Check installs on the laptop (sentence-transformers, umap-learn, nltk + WordNet; models are pinned and cached there). This work cannot be done in a cloud container: the plan assumes Mike's M-series 24 GB machine and MLX.
4. P1: example sentences, about 170 sessions, half a day.
5. P2: pilot on 6 plots, about 120 sessions; Mike times the audit page on at most 5 stories (about 15 min); if his median is above 150 s per story, set `analysis.AUDIT_BLOCKS_SHOWN = 4`.
6. Run `test_analysis.py` (about 5 min), then **F0**: commit and push, write the hash into `DEVIATIONS.md`.
7. DEV features (3 h), F1 (must reproduce E2 within 0.03 of 0.164 / 0.102 or stop), F1b benchmark.
8. Write the 816 stories with reads, ratings, one repair round (about 1 day, 2,690 sessions, 1.5 h of Gemma); F2-T2.
9. Mike's audit, 60 stories, 300 answers; freeze A7.
10. TEST features (about 16 h) + Claude upper bound; F3 manifest.
11. Analysis and label test (about 5 h, 1,310 sessions).

Anything deferred needs a new pre-registration of its own. The scout's raw Wikipedia cache (about 126 MB) is not committed; the scripts have hard-coded scratchpad paths and Wikipedia has changed since 2026-10-20, but gzipped copies of what the work lists need are in `sources/`.

### 8. Every experiment idea in the folder, with status

Planned in prereg-v3, not run:

| # | Idea | Status |
|---|---|---|
| 1 | H1 neurons vs evasion-aware keywords and TF-IDF logistic regression on reworded planted stories | planned, primary |
| 2 | H2 non-inferiority to Qwen3-Embedding-4B (sentences, windows) and e5-large | planned, primary |
| 3 | H3 specificity of 11 Batman-found concept units in other genres | planned, primary |
| 4 | H6 cost-accuracy dominance rule | planned, decision rule |
| 5 | S-coded: invented code words, context-using vs context-free methods | planned, prominent secondary |
| 6 | Overlap O1/O2: which planted memes each of 17 methods finds that others miss | planned, prominent secondary |
| 7 | S8 direct LLM judge beats neurons | planned, secondary (predicts LLM wins) |
| 8 | S9 summary judge below direct judge; T2-long on 48 concatenated texts | planned, secondary |
| 9 | S10 implicit harder than banned | planned, secondary |
| 10 | S11 keywords fooled by decoys | planned, construction check |
| 11 | m1 vs m5: is one neuron enough ("single neurons are memes") | planned, secondary |
| 12 | Neurons vs the same model's hidden states at 4 layers | planned, key secondary contrast |
| 13 | Neuron variants: selected on explicit-only examples, far examples, in-context background, length-residualised | planned, descriptive |
| 14 | T6 blind label test: are neurons more nameable than LDA topics and embedding clusters (S21) | planned; first candidate to cut in the trimmed option |
| 15 | T6 cost recipes, total-cost curves at 100 / 1,000 / 100,000 stories, storage policies | planned |
| 16 | Labelling-effort accounting (calls per usable label) | planned, descriptive |
| 17 | Claude judge upper bound; Claude ban-list-aligned keywords | planned, labelled non-rivals |
| 18 | Small-model judge (Qwen3-1.7B), shared-sheet judge, one-pass judge, DeBERTa NLI | planned, rivals in H6 |
| 19 | Author's blind audit of 60 stories; Gemma 2 9B cross-family reader | planned, manipulation checks |
| 20 | T1a DEV lineage grid at F1 | planned, DEV context only |
| 21 | P1 example writing, P2 pilot | planned, not started |

Deferred (designed and reviewed in v2; code and lists kept; needs its own pre-registration):

| # | Idea | Status |
|---|---|---|
| 22 | T1b graded vocabulary drift, H4, with style-only control, source-leak index, headroom-free retention, S1, S2 | deferred |
| 23 | T1c Chinese translation of the Batman lineage (S3, S5), natural zh.wikipedia arm, machine translation + words baseline | deferred |
| 24 | T1d Shakespeare fresh lineage, H5, S6 "beyond words" residual, recognition probe | deferred |
| 25 | T1e 7 fresh Batman adaptations as positive control (S7), 8-gram reuse audit | deferred |
| 26 | T2-zh planted stories in Chinese (S12) | deferred |
| 27 | T2-N exhaustively annotated natural summaries (S13) | deferred |
| 28 | T3 Batman-likeness across 2,544 works (S14, S15, S15b, directional contrasts vs TF-IDF, LSA, embeddings) | deferred |
| 29 | T3 orphan-vigilante family layer (S16, S17): Zorro, Pimpernel, Iron Man; Spider-Man, John Wick, Puss in Boots | deferred |
| 30 | Relation typing of documented relatives (plot/role vs iconography vs parody vs meta) | deferred |
| 31 | T4 mutation and selection (S18 innovators, S19 selection vs margin-preserving null, Bentley neutral model, batch-effect check) | deferred |
| 32 | T5 concept agreement between Qwen and Llama on held-out random films vs lexically matched nulls (S20) | deferred |
| 33 | X-code baselines: codes-only and code-collapsed sensitivity | deferred with T1 |
| 34 | LLM "likeness to Batman" judge and recognition probes | deferred with T3/T1d |

Replaced, cut or not adopted:

| # | Idea | Status |
|---|---|---|
| 35 | v1 H4 degradation statistic (drift minus level 0) | replaced; shown to pass at chance |
| 36 | v1 H3 "AUC above 0.5" | replaced by specificity; kept as secondary |
| 37 | v1 separate sentence-level T2 and story-level T2b | merged into one benchmark |
| 38 | S4 word methods fail across languages | demoted to construction check |
| 39 | Spanish as a near-language condition | cut in v1 |
| 40 | Hindi, Japanese, Russian, German, French, Thai | token cost measured only |
| 41 | Qwen3-Embedding-8B; bge-m3, mpnet, e5-small as encoders | 8B rejected (memory); others timed, not chosen |
| 42 | Listwise LLM lineage reading; pair questions in both orders | rejected |
| 43 | Drifting the reference films | rejected |
| 44 | Cross-fitted "update arm" re-selecting on TEST | partly adopted as the reworded example set |
| 45 | Annotated public-domain novels as the long-text condition | rejected for concatenated stories |
| 46 | GPT or Gemini as second reader family | rejected |
| 47 | Prose-novel Shakespeare adaptations as a second medium | noted by scout, never designed |
| 48 | Count of Monte Cristo as a Batman relative | unverified; stays in pool, descriptive |
| 49 | T4 held to forecasting standards | not wanted |
| 50 | Trimmed T2 (drop label test and long texts; one judge on Qwen only) | proposed, not chosen |
| 51 | Batching several label-test judge items per session | possible saving, not taken |

Absent from the folder altogether: any SAE or transcoder features as the unit (every "neuron" is a raw MLP channel); any learned mapping between the two models' units beyond a best-correlated twin; any third model; any human rater other than the author; any real-world corpus of evasive language.

## What this means for P1 and P2

**P1 (a meme can be measured by which neurons fire).**

- The plan does not assume P1; it is written as a test P1 could fail. Its own summary of DEV is "words do as well" and "on name-free Wikipedia summaries the neurons have not yet earned their extra cost" (section 1.2). Section 13: "It does not claim the neurons will win."
- The plan's three sharpest P1 tests are H3 (do individual labelled units mean what their label says outside Batman), m1 vs m5 (is one neuron a meme or do you need several) and neurons vs hidden states (is there anything special about the neuron basis). Only H3 is primary, and it starts from weak ground: 6 of its 11 units are marked in `dev_units.json` as not surviving DEV rewording, and its power is 0.14 at the smallest effect of interest.
- The unit is a raw SwiGLU channel, picked by a supervised search over about 0.9M candidates from 10 example sentences. A win shows that a good detector can be found among 0.9M channels, which is a weaker claim than "the model's neurons are the memes". The reviewers said this twice. The neurons-vs-hidden-states contrast is the only part of the design that separates the two.
- Even a clean H1 win has a narrow reading: neurons beat an 8B model's keyword list and a TF-IDF classifier on text Claude wrote to avoid listed words. The plan itself predicts (S8) that an LLM simply reading the story will beat the neurons.
- Conditions under which P1 could hold, per the design: when wording changes but context carries the idea (coded condition), and when many memes are scored per text (cost). Neither has data.

**P2 (the measurement generalises across models).**

- Nothing in the active plan tests P2. Llama "replicates" when the same test gives p < 0.05; that shows the method works twice, not that the two models share features.
- The folder's only direct test (T5, S20) is deferred. The committed DEV numbers it would build on are thin: across the top 200 meme families, a Qwen unit's Llama twin correlates 0.636 on original text and 0.142 on name-free text; a freshly searched partner reaches 0.426 against 0.401 for lexically matched null units. For the 27 concept units the margin over the null is 0.077 (18 of 27 wins). Cross-model agreement in this study is mostly carried by names and words (`sources/robust_agreement_analysis.json`).
- The DEV "beyond words" test "fails for Qwen and passes only for Llama" (section 1.2). That is a result that differs by model, which cuts against P2 as stated.
- H3 in Llama uses each Qwen unit's best-correlated Llama twin chosen on original (name-bearing) text, so a Llama H3 failure would not separate "the concept does not transfer" from "the twin was matched on names".
- No experiment anywhere in the folder learns a map between models (the Jha et al. style of evidence) or adds a third model. Cross-language generality (Chinese) is deferred, and the scout check is one story.

## Experiment seeds

1. **Run the trimmed planted-story experiment as written.**
   - Question: when a meme's usual words are removed, do neurons detect it better than good keyword practice and as well as embeddings?
   - Data: 816 stories from `planted_design.json` and `memes_for_tracing.json` (not yet written).
   - Method: prereg-v3 trimmed option; F0 before any story exists.
   - Rival to beat: `keywords:expanded-qwen3-8b`, `keywords:logreg` (TF-IDF), `emb:qwen3emb-4b`, `embwin:qwen3emb-4b`, `emb:e5-large`, and `judge:qwen3-8b` reading the text.
   - Falsifier: as in section 1.4; H2 falsified if any vector is credibly more than 0.05 AUC better.
   - Cost: about 3,100 sessions, about 1.5 days, 2.5 to 3 h of Mike. Risk: "undetermined" at power 0.2 to 0.4.

2. **Pull T5 forward as the cheapest P2 test.**
   - Question: do Qwen concept units have Llama partners that agree on text neither model was tuned on, beyond what shared vocabulary explains?
   - Data: 1,000 random films already committed (`sources/t3_random_films.jsonl.gz`, 22,225 sentences); 27 concept units and 81 null units in `dev_units.json`.
   - Method: `analysis.run_t5` (S20), with its own short pre-registration; partners searched on DEV only.
   - Rival to beat: the lexically matched null units, and a TF-IDF-predicted activation for each unit (word overlap).
   - Falsifier: mean r(concept) minus mean r(null) not above 0 on the random films (label-swap null, p >= 0.05).
   - Cost: two extraction passes of about 10 minutes each at 38 sentences/s, no agent sessions.

3. **Basis test on DEV before spending on stories.**
   - Question: is anything gained from the neuron basis over the same model's residual stream or an embedding probe?
   - Data: DEV corpus and the P1 example sentences (170 sessions).
   - Method: the F1 step already computes m1, m5, hidden-state prototypes and probes with leave-one-example-out; report them side by side.
   - Rival to beat: `hidden:<model>` probe, `emb:qwen3emb-4b` probe, `keywords:logreg`.
   - Falsifier: m1/m5 leave-one-out percentile no better than the hidden-state probe on held-out reworded examples.
   - Cost: about half a day of agents plus 4 h of compute. DEV only, so it informs the decision to run seed 1 but cannot confirm anything.

4. **Mapped-detector transfer (not in the folder).**
   - Question: does a meme detector found in Qwen still detect the meme after being mapped into Llama?
   - Data: the planted stories from seed 1, once features exist.
   - Method: fit a linear map between the two models' unit activations (or hidden states) on DEV and reference-film sentences; apply Qwen's detector through the map; score twin AUC in Llama.
   - Rival to beat: the keyword list (transfers between models at zero cost), an embedding prototype, and a map fitted on lexically matched null units.
   - Falsifier: mapped twin AUC not above the null-unit map, or below the keyword rival.
   - Cost: compute only, about an hour, after seed 1.

5. **Code words that build up across texts.**
   - Question: can any method track a renamed idea when the meaning of the code word is only learnable across documents?
   - Data: a real corpus of evasive vocabulary, or planted multi-story sets where a code term is introduced in story 1 and used bare in stories 2 to 5.
   - Method: extend the coded condition; score late stories alone.
   - Rival to beat: sentence embeddings, the evasion-aware keyword list after one revision on early stories, and an LLM judge given the early stories as context.
   - Falsifier: neurons no better than the keyword list revised once.
   - Cost: about 300 to 500 sessions if planted.

6. **A fresh lineage with enough clusters.**
   - Question: do neurons recover documented sources better than TF-IDF and LSA outside Batman?
   - Data: `works_shakespeare.json` gives 13 plays for H5; power 0.17 at SESOI, so add lineages until there are at least 40 source clusters.
   - Method: the v2 T1d design with lead-sentence or Wikidata P144 evidence only.
   - Rival to beat: TF-IDF, LSA, Qwen3-Embedding-4B, full-text pairwise LLM reading.
   - Falsifier: rival advantage credibly below 0.05 share, or no method recovers the sources at all (uninformative).
   - Cost: name removal about 3 sessions per work; v2 priced Shakespeare-scale name removal at about 720 sessions.

7. **Drift with the style-only control first.**
   - Question: how much of any neuron-vs-TF-IDF gap under rewriting is caused by Claude re-rendering the text at all?
   - Data: the 20 DEV Batman screen works, style-only rewrite and level 100.
   - Method: `drift_protocol.md` at two levels only.
   - Rival to beat: TF-IDF, LSA, Qwen3-Embedding-4B.
   - Falsifier: the neuron advantage on style-only rewrites is as large as at level 100 (the effect is authorship, not vocabulary).
   - Cost: about 80 rewrites and 80 checks.

## Key sources

Repo files (all confirmed by reading them this session):

| Source | Why it matters | Priority |
|---|---|---|
| `preregistration/PREREGISTRATION.md` | The plan: hypotheses, falsifiers, power, costs, design history 14.1 to 14.5, deferred list in section 15 | A |
| `preregistration/history/design_workspace/design_result.json` | Scout reports, the v1 draft, all four round-1 reviews in full, v2 final | A |
| `preregistration/history/design_workspace/narrow_result.json` | Narrowing changes, open questions, the fifth review, run plan | A |
| `preregistration/sources/robust_agreement_analysis.json` | The only cross-model agreement numbers in the folder (P2) | A |
| `preregistration/dev_units.json` | H3's 11 units with their DEV rewording status; T5 concepts and nulls | A |
| `preregistration/README.md` | Status, costs, resume steps | A |
| `preregistration/history/power_v4.json`, `power_v3.json`, `power_v2.json` | Power at each stage | B |
| `preregistration/history/design_workspace/scout/tools/*.json`, `scout/lang_check*.json` | Timings and language checks | B |
| `preregistration/drift_protocol.md`, `translation_protocol.md`, `name_removal_protocol.md` | Deferred protocols ready to reuse | B |
| `preregistration/history/prereg_v2_before_narrowing/PREREGISTRATION.md` | Full text of the deferred tests T1, T3, T4, T5 | B |

External references for methods the plan names but does not cite (each confirmed on Crossref this session):

| Citation | DOI | Why | Priority |
|---|---|---|---|
| Lakens (2017), "Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses" (preprint record found) | 10.31234/osf.io/97gpc_v1 | Smallest effect size of interest and the supported / falsified / undetermined logic | B |
| Berger and Hsu (1996), "Bioequivalence trials, intersection-union tests and equivalence confidence sets", Statistical Science | 10.1214/ss/1032280304 | The intersection-union tests behind H1 and H2 | C |
| Steen, Yurechko and Klug (2023), "You Can (Not) Say What You Want: Using Algospeak to Contest and Evade Algorithmic Content Moderation on TikTok", Social Media + Society | 10.1177/20563051231194586 | The real phenomenon the coded condition imitates | B |
| Strona, Nappo, Boccacci, Fattorini (2014), "A fast and unbiased procedure to randomize ecological binary matrices with fixed row and column totals", Nature Communications | 10.1038/ncomms5114 | The curveball null in deferred T4 (S19) | C |
| Bentley, Hahn and Shennan (2004), "Random drift and culture change", Proc. R. Soc. B | 10.1098/rspb.2004.2746 | The neutral-copying model in deferred T4 | C |

## Fog

- I did not read `analysis.py` (5,254 lines, 221 functions), `readers.py`, `build_lists.py` or `test_analysis.py` line by line; I read their outlines, constants and the DEFERRED markers. I did not run `test_analysis.py`, so "151 checks pass" is the README's claim.
- Whether the commit that added the folder counts as a public timestamp before test data: yes in substance (no data exists), but F0 has not been declared and P1/P2 have not run, so the plan is unlocked.
- Small inconsistencies I could not resolve: the scout text says 657 Shakespeare articles considered, `scout/shakespeare_summary.json` says 647; section 14.2 says 203 of 205 draft keyword terms were multi-word, review 4 says 204 of 205 (and 183 contain a banned stem); 14.2 says 8 blockers, the review JSON has 10 blocker entries; README says about 3 h of Mike's time, the plan's table says about 2.5 h plus a 15 min pilot; salience ratings appear as 544, 550 and (before the fix) 410.
- `dev_r_namefree_fresh` in `dev_units.json` maps to `r_anon_fresh_m` in the agreement file (`build_lists.py` line 624). I read it as "correlation with a freshly searched Llama partner on name-free text"; the exact search procedure lives in the study's robustness code, which is outside this folder.
- In `sources/robust_agreement_analysis.json` there are two null columns, `null_rk` and `null_rl`. I used `null_rl` and read it as the lexically matched null (the companion file is `robust_agreement_lexical_null_units.npz`); the file does not spell the abbreviation out. In `dev_units.json`, `dev_rewording_survivor: false` may mean "tested and failed" or "did not meet the raw criterion"; `null` (mantle) looks like "not tested".
- The trimmed option's session count (about 3,100) is my subtraction of the 1,310 label-test sessions from 4,430; the README gives only "about 1.5 days".
- Not checked: whether the pinned model revisions still resolve; whether Qwen3-Embedding-4B and Gemma 2 9B are really in Mike's cache; the full v1 draft text beyond its headings, hypotheses, power and budget.
- Why Mike paused, and whether he still wants the code-word and overlap analyses as the headline, is not recorded anywhere in the folder.
