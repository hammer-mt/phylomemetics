# a1-study: audit of the finished Batman study

Files read: `study/paper/paper.md` (PAPER), `study/README.md` (README), `study/results/experiments.json` (EXP), `study/results/robustness/nulls.json` (NULLS), `e2_multiverse.json` (MV), `e2_jackknife.json` (JK), `e2_reimpl.json` (REIMPL), `label_test.json` (LABEL), `e1_tfidf_words.json` (E1W), `study/results/phylo.json.gz` (PHYLO), `study/results/wording.json` (WORD), `study/results/design_reviews.json`, `study/results/factcheck.json`, `images/README.md`, `preregistration/README.md`. Nothing was run. All paths below are relative to ``.

## Bottom line

- The study shows that neuron units are a **countable, addressable, repeatable** way to describe a text. It does **not** show that they measure anything plain word overlap misses. On the primary model (Qwen3-8B) words matched or beat neurons on every pre-registered test (EXP).
- **P1 is untested at its core.** The study assumes neurons are clean; it never checks it causally (no steering, no ablation). Its own evidence says most strong units are word and name detectors: 24 of the 33 strongest are name/place detectors and a single word reproduces their profile at median r = 0.85 (NULLS `f2_agreement.sets.top33`).
- **P2 gets weak support at unit level.** The headline twin correlation (0.82) is matched by non-meme units that are equally word-like (0.86), and the same twin pairs fall to 0.14 once names are removed (NULLS). Over all 14,843 families the median twin r is 0.40 against 0.36 for random active units (PHYLO). Whole-corpus agreement (Mantel 0.98) is mostly a floor that any representation reaches (random units 0.90 to 0.93, TF-IDF vs LSA 0.92; NULLS `f2_trees`).
- **Heredity (E2) is real and very robust, but it is a property of the texts, not of the neurons.** TF-IDF finds the sources as well (11% vs Qwen 16%, Llama 10%; EXP). The one result in the neurons' favour, Llama's residual test (p = 0.010 to 0.033), is a single specification, uncorrected for 6 tests, and its TF-IDF version fails the study's own stricter nulls (p = 0.057 and 0.065; EXP). The paper does not say this.
- **Identity (E1) failed and selection (E3) was uninformative.** E1: AUC 0.52 and 0.54 against hero films, where a TF-IDF centroid gets 0.80 and text length alone gets 0.76 (EXP). E3: rho 0.25 and 0.00 with 27% power at rho = 0.3 (EXP).
- What survives as a neuron-specific signal is small: 27 concept memes with better cross-model twins than word-matched units on original texts only (0.62 vs 0.46, p = 0.001), one Llama revenge unit (10 of 12 minimal pairs), and 2 to 6 units specific to their own rewording (NULLS, WORD). All are post hoc, none pre-registered.

## Findings

### A. Setup (PAPER section 2)
- Corpus: first 40 sentences of English Wikipedia plot sections. 32 Batman works (1943 to 2022), 13 relatives, 69 ordinary films, 3,700 sentences. Name-free version: an LLM rewrite located names, a diff replaced 9,156 spans with codes X1, X2 (PAPER 2.1).
- Units: both poles of every MLP hidden channel. 884,736 in Qwen3-8B, 917,504 in Llama 3.1 8B, 344,064 in Qwen3-1.7B, all 4-bit MLX (PAPER 2.2; NULLS `f1_chance.*.n_units`).
- Presence: mean of a work's top-3 sentences beats all but 1 of 69 reference films (about 3% false-positive rate per unit and work). Candidate memes: present in at least 6 of 32 Batman works. Units co-firing at r >= 0.5 merged into families: 14,843 (Qwen3-8B), 24,732 (Llama), 5,641 (Qwen3-1.7B) (NULLS `f1_chance.n_families`).

### B. Every experiment and analysis, with verdict

| # | Analysis | Design | Result (file) | Verdict |
|---|---|---|---|---|
| 1 | Meme count vs chance | 50 random 32-film pseudo-franchises, bar set by the other 37 films | Units in >= 6 works: Batman 53,985 vs pseudo 29,400 (1.8x Qwen; 2.4x Llama). Units in >= 23 of 32: Batman 1,534 (Llama 1,914), pseudo 0 in all 50 draws (NULLS `f1_chance`) | Supported: a core canon exists. Names are present, so this partly counts name detectors |
| 2 | Canon tiers | Share of 32 works a family appears in | 56 core (24 Batman-only, 32 shared), 194 common, 14,593 optional (PAPER 3.1; from `study/results/canon.json`, not re-traced by me) | Descriptive |
| 3 | Blind label test | 71 labels, two-choice, AI judges, held-out sentences | 56 of 71 = 79% (CI 68 to 88). Word-overlap judge 65%; p = 0.007 approximate. Named 31 only: 77% vs 63%, p = 0.065 (NULLS `label_test`; LABEL) | Mixed. Labels beat chance; for the 31 showcase memes they do not clearly beat word overlap |
| 4 | Entry and spread timelines | First appearance of each meme in the sample | Descriptive; 6 of 15 optional memes "first appear" in 1986 because the sample has a 1967 to 1985 gap (PAPER 3.2) | Descriptive, sample artefact admitted |
| 5 | Twins, top 33 memes (Qwen to Llama) | Best-correlated Llama unit over 3,700 sentences; nulls matched on sparsity, layer, word-likeness | Median r 0.82; sparsity-matched 0.40; word-matched 0.86. Paired: memes ahead by 0.05, 25 of 33, p = 0.007. Names only: p = 0.07. Name-free, same pairs: 0.14 vs 0.33 / 0.23 (random) and 0.56 (word-matched) (NULLS `f2_agreement.sets.top33`) | Mixed. Agreement is real but is mostly shared word detection |
| 6 | Twins, all families | Same search, all 14,843 | Median 0.40 vs 0.36 random active non-meme units vs 0.14 shuffled (PHYLO `models.llama3.1-8b.partner_r`) | Weak |
| 7 | Twins, concept memes | 27 concept-labelled families among top 200 | 0.62 vs 0.46 word-matched, 20 of 27, p = 0.001 on originals. Name-free fixed pairs: 0.34 vs 0.38 (NULLS `labelled_concept`) | Supported narrowly, original texts only |
| 8 | Presence grid agreement | 31 named memes x 45 works, Qwen vs Llama twin | 87% of cells, kappa 0.73; random units kappa 0.35 to 0.45, word-matched 0.57 (NULLS `named31.grid_orig`) | Supported relative to nulls, original texts |
| 9 | Tree agreement | Mantel of Jaccard distances, 45 works | 0.98; random non-meme units 0.90 to 0.93; TF-IDF vs LSA 0.92. Partial given TF-IDF: 0.92 vs random 0.82 to 0.88. Meme distance vs TF-IDF distance: 0.84 and 0.85 (NULLS `f2_trees`) | Mixed. Mostly a floor; small excess over it |
| 10 | Qwen3-1.7B | Same pipeline, original texts only | Mantel with Qwen3-8B 0.983; twin median 0.40 vs 0.355 random vs 0.13 shuffled; tree source recovery 5.78 vs 5.76 chance, p = 0.53 (PHYLO `models.qwen3-1.7b`) | Same picture as Llama; never run name-free |
| 11 | Rewording test | Top sentence of 31 named memes rewritten with about zero shared content words; pass = beats 10 everyday sentences | Qwen 11 of 31 (mismatched-rewording null expects 5.0, p = 0.002). Llama 9 of 31 (null 5.7, p = 0.076). Specific to own rewording: 2 Qwen, 6 Llama. Four pass in both models (NULLS `f3_wording`; WORD) | Mixed. A few units respond to ideas; most do not; Llama not significant |
| 12 | Revenge unit | Chosen from 50 candidates on 10 practice pairs; tested on 12 new minimal pairs with no revenge words | Llama L14.3474: 10 of 12, AUC 0.83 vs control, 0.87 vs harm, 0.79 vs vow. Qwen L27.375: 4 of 12, AUC 0.45 (dev AUC 0.965). Cross-model r 0.39 (WORD `revenge`; NULLS) | Mixed. One unit in one model; Qwen's is a keyword detector |
| 13 | Tree shape, era clustering | Neighbour-joining; between/within-era distance ratio | 1.03 in both big models, p = 0.03 and 0.02. TF-IDF: 1.036, p = 0.0004. Random units: 1.02 (PHYLO `era_ratio`) | Weak; words do it at least as well |
| 14 | First heredity test (trees) | Rank of documented source by meme-tree distance among 12 comics | Mean rank 5.4 (Qwen), 5.3 (Llama) vs 5.8 chance, p = 0.40 and 0.36. TF-IDF 2.9 (p = 0.003), LSA 3.1 (p = 0.006). Random non-meme units beat the memes in 100% of draws (3.2 to 4.3) (NULLS `f2_trees.source_recovery`) | Failed. Meme families were worse than random units and much worse than words |
| 15 | Vigilante lineage | Nearest relatives; 6 orphan-vigilante memes across 13 relatives | Pimpernel and Zorro carry 3 of 56 core memes; Iron Man 35 (PAPER 3.5) | Inconclusive: one summary each, names present, Zorro has 9 sentences |
| 16 | E1 identity (pre-registered) | Held-out work's share of canon units, name-free, vs 8 hero films; prediction AUC >= 0.80 | 0.52 (CI 0.28 to 0.75) Qwen, 0.54 (0.31 to 0.79) Llama; null p95 0.70 / 0.67. vs ordinary films 0.90 / 0.95. With names 0.996 / 0.993. TF-IDF centroid 0.80 (0.64 to 0.93); length alone 0.76. Name-free core: 7 of 407 Qwen units, present in 91% of hero cousins (EXP `*.E1`, `lexical_comparators.E1_tfidf_centroid`) | Failed. Words beat neurons |
| 17 | E2 heredity (pre-registered) | Share of eligible comics beating the documented source, hub-adjusted, 9 adaptations, name-free | 16% Qwen, 10% Llama vs 47% null; exact p 0.0026 and 0.00017. Within-era null 39% / 38%, p 0.026 / 0.0014 (EXP `*.E2.primary.chance`) | Supported |
| 18 | E2 sequel positive control | Prequel in top 2 earlier screen works, bar 5 of 7, name-free | Qwen 4, Llama 5, TF-IDF 6, word presence 6 (EXP `sequels_name_free`) | Qwen fails its own positive control; words pass |
| 19 | E2 placebo | Non-adaptations given the same source sets | As pre-registered (11 targets) the criterion fails; with 9 valid targets it passes (46% / 47%) (EXP; REIMPL `placebo.mean_as_coded` 0.378) | Passes only after a post hoc fix |
| 20 | E2 word comparators | Same statistic with TF-IDF cosine and word presence | TF-IDF 11%, p 0.0003, within-era 0.003. Word presence 17%, p 0.0018 (EXP `lexical_comparators`) | Words match neurons |
| 21 | E2 residual "beyond words" | Regress neuron inheritance on word similarity over 32 x 12 pairs, retest | Qwen: p 0.25 / 0.11 / 0.089. Llama: 0.033 / 0.014 / 0.010 (joint null). Llama under within-era null: 0.065 / 0.019 / 0.017. Reverse: TF-IDF with neurons regressed out stays at p 0.0003 in both. R2 of fit 8 to 21% (EXP `residual_test_beyond_words`) | Failed for the primary model; marginal pass for Llama |
| 22 | E2 robustness | Independent rebuild; 30,240-spec multiverse; jackknife; 4 other nulls | 99.6% of specs p < 0.05; all 132 failures have hub adjustment off. Sequel control passes in 48% of Qwen name-free specs, 82% Llama. Qwen fragile under within-era null with leave-outs (9 of 36 pairs fail) (MV `overall`, `by_cell`; JK) | Supported, for the neuron pattern only |
| 23 | E3 selection (pre-registered) | Spearman of log pageviews with residualised spread, 15 early works | rho 0.25 (p 0.35) Qwen, 0.00 (p 0.58) Llama. Original texts -0.32 and -0.50. Power 27% at rho 0.3, 54% at 0.5 (EXP `*.E3`, `E3_power_simulated`) | Not supported; uninformative |
| 24 | Image side experiment | CLIP RN50x4 and DINOv2 on one poster per work; outside the paper | CLIP: 56 neurons on >= 6 of 29 Batman images (0.6 by chance). DINOv2 twins 0.45 vs 0.36 random (`images/README.md`) | Exploratory, no cheap rival compared |

### C. Every place plain words matched or beat the neurons
1. E2 source finding: TF-IDF 11% vs 16% (Qwen) and 10% (Llama) (EXP).
2. E2 sequel control: words 6 of 7, Qwen 4, Llama 5 (EXP).
3. E2 residual: Qwen adds nothing; TF-IDF keeps p = 0.0003 after neurons are removed, in both models (EXP).
4. E1: TF-IDF centroid 0.80, text length 0.76, neurons 0.52 and 0.54 (EXP).
5. Tree source recovery: TF-IDF 2.9 and LSA 3.1 vs memes 5.4 / 5.3 / 5.8 (NULLS, PHYLO).
6. Twins: word-matched non-meme units 0.86 vs memes 0.82 (NULLS).
7. Label test: word-overlap judge 63% vs 77% on the 31 named memes, p = 0.065 (NULLS).
8. Tree agreement: TF-IDF vs LSA 0.92, random units 0.90 to 0.93 (NULLS).
9. Era clustering: TF-IDF 1.036 (p = 0.0004) vs memes 1.03 (p about 0.03) (PHYLO). Not stated in the paper.
10. Single units: the funeral unit tracks the word "funeral" at r = 0.98; the Lucius Fox unit fires on "The Fox" (PAPER 3.4, Table 1).

### D. Where neurons beat words (all small, mostly post hoc)
- Llama residual test, joint null only (EXP).
- 27 concept memes: twins 0.62 vs 0.46, original texts (NULLS).
- Top-33 paired advantage of 0.05 (p = 0.007); top-200: 0.64 vs 0.57 word-matched, 133 of 200 wins, p = 3e-7 (NULLS `top200.r_orig`; the top-200 figure is not in the paper).
- Partial Mantel after TF-IDF: 0.92 vs random 0.82 to 0.88 (NULLS).
- Qwen rewording survivors 11 vs 5.0 expected (NULLS).
- Llama revenge unit (WORD).

### E. Limitations the paper admits (PAPER 2.5, 3.x, 4.4)
- Wikipedia retellings from the 2020s; related works share wording. One franchise, 32 works, gap 1966 to 1986, short summaries (Zorro 9 sentences).
- E1: 8 negatives; presence bar set by non-superhero films. E2: 9 adaptations, 7 source comics. E3: 15 works, popularity measured 2015 to 2026.
- Small 4-bit models. Labels by an AI reader, never validated by steering. Family count depends on model and merge threshold. Some units track Wikipedia house style.
- E2's band and hub adjustment were designed after the tree test failed. Word comparators and the residual test were pre-registered but run only after review. Placebo set cut from 11 to 9. Qwen3-1.7B replication of E2, Holm correction and several small checks not run.
- Pre-registration order cannot be proven from file times or git history.
- Name-free diff sometimes swallowed a neighbouring word; bias direction unmeasured.
- Judges' item-level answers not kept, so the test against the word-overlap judge is approximate.

### F. Limitations the paper does not admit (my reading)
1. **Llama's "beyond words" pass is fragile.** Under the study's stricter nulls the TF-IDF-residual version gives p = 0.057 (time-consistent) and 0.065 (within-era) (EXP `llama3.1-8b.E2.residual_test_beyond_words.neurons_resid_on_tfidf_cosine.chance`). Table 3 leaves that column blank for the residual rows. Six residual tests (3 versions x 2 models) are uncorrected; Bonferroni needs p < 0.0083 and the best is 0.010.
2. **The multiverse and jackknife never touch the words-vs-neurons question.** `study/robustness/e2_multiverse.py` has no TF-IDF, lexical or residual code (grep). "99.6% of 30,240 specifications" protects a pattern that word counts also show.
3. **The sequel positive control is weak.** Candidate pools are 9, 13, 3, 5, 6, 12, 14 works (EXP `sequels_name_free.rows[].n`). My calculation: random ranking gives 2.1 hits expected, P(>= 4) = 0.10, P(>= 5) = 0.019. Qwen's 4 of 7 is not distinguishable from chance at 0.05.
4. **E2 does not use memes.** It uses about 150,784 raw optional units in Qwen (REIMPL `n_optional_units`), 17% of all units, with no cleanliness or label requirement. It is a high-dimensional similarity score. In the tree test random non-meme units beat the meme families outright. Nothing in the study shows the "clean neuron" is the unit doing the work.
5. **Twin selection is in-sample.** The twin is the best of about 917,000 Llama units on the same 3,700 sentences it is scored on. The nulls share this bias so comparisons are fair, but the absolute 0.82 is optimistic. The name-free result is in effect the only out-of-sample check, and it collapses.
6. **The name-free collapse is reported in its harshest form only.** With a fresh twin search on name-free texts the top-33 memes reach 0.54 (sparsity-matched 0.41, p = 0.027; word-matched 0.69) (NULLS `top33.r_anon_fresh_search`). So twins still exist without names, but word-matched units still do better.
7. **E1's word control may be mostly length.** TF-IDF centroid 0.80 vs length alone 0.76 (EXP). The "crime-city vocabulary" reading rests on a 0.04 gap inside a CI of 0.64 to 0.93.
8. **Training-data contamination.** Qwen and Llama were very likely trained on these Wikipedia pages. Units may respond to recognised text. Not discussed.
9. **Name removal does not remove identity.** The models can infer the franchise from plot context; only 68 iconography sentences were dropped (EXP `iconography_sentences_dropped`).
10. **Only weak rivals were tried.** TF-IDF, binary word presence and LSA. No sentence embeddings, no residual-stream pooling, no SAE or transcoder features, no LLM reading the text. Listed as future work, not as a limitation of the conclusions.
11. **The residual test is a linear partial-out with R2 of 8 to 21%.** A better lexical measure (n-grams, lemmas, BM25, synonyms) could absorb Llama's remaining signal. Residual confounding is not discussed.
12. **One AI model family did the labelling, the name-free rewrites, the design reviews, the blind judging and the fact-check.** No human validation at any step. Stated under "AI use", not treated as a threat to validity.
13. **P2 was tested only as one-to-one neuron matching on in-domain text.** No linear map, no stitching, no translation between models, no held-out text. The paper cites Huh et al. and Jha et al. for convergence but tests none of what they claim.
14. **No causal evidence at all.** Every result is a correlation of activations with texts.

## What this means for P1 and P2

**P1 (a meme can be measured by which neurons fire).**
- Shown: the measurement is well defined and repeatable; it yields a non-trivial franchise core that random film sets lack (NULLS); labels predict held-out activations better than chance (79%, LABEL).
- Not shown: that the measured thing is a meme rather than a word. On this corpus most strong units are lexical, and the unit loses to or ties with word counts on identity, heredity, sequel recovery and tree-based source recovery.
- Cuts against: meme families did worse than random non-meme units at recovering sources from trees (NULLS). Only 2 of 31 Qwen units are specific to their own rewording.
- Conditions where P1 looked best: mid-layer concept units (revenge at Llama L14), and tests built so that words cannot help (minimal pairs, rewording). Conditions where it looked worst: any test on naturally worded Wikipedia text.
- Strength of evidence: weak. P1's central claim (clean, single-job neurons) was assumed from the Goodfire essay, on a 32B model, and never tested on the 8B 4-bit models used here.

**P2 (models converge on the same units, which can be mapped).**
- Shown: strong units have close twins in a model from another company (0.82), and presence grids agree (kappa 0.73).
- Not shown: that this is conceptual convergence. Word-matched non-meme units have equally good twins; the excess is 0.05. Agreement is highest for names (0.86) and common words (0.96), lower for concepts (0.62 to 0.69), and near the random floor for the median family (0.40 vs 0.36).
- Cuts against: the two models' revenge units correlate only 0.39; Qwen's is a keyword detector and Llama's is not. The same-family smaller model (Qwen3-1.7B) agrees no better than the other-company model (0.40 in both; PHYLO), which suggests the agreement level reflects the text, not shared training.
- Three models, two families, all about 8B or smaller, all 4-bit. No test of mapping in the sense of the cited literature.
- Strength of evidence: weak to moderate for "name and word detectors recur"; weak for "concept units recur"; none for "units can be mapped between models".

## Experiment seeds

1. **Residual test under stronger word controls.** Question: does Llama's "beyond words" signal survive better lexical measures and the stricter nulls? Data: existing corpus; activations regenerate in about 30 minutes. Method: rerun the E2 residual with n-gram, lemma, BM25 and sentence-embedding covariates, under all five nulls, inside the multiverse grid. Rival: TF-IDF, BM25, e5 or similar sentence embeddings. Falsified if the residual p exceeds 0.05 in most specifications or under the within-era null. Cost: hours of laptop time, no new data.
2. **Held-out twin test.** Question: do twins found on half the sentences hold on the other half and on unrelated text? Data: the 3,700 sentences split by work, plus a few thousand sentences from non-film Wikipedia. Method: pick twins on the training half, score on held-out; add a ridge map from all Llama units to each Qwen unit. Rival: the best single word, and a TF-IDF regression, predicting the same Qwen unit. Falsified if held-out twin r for concept units is no higher than for word-matched non-meme units. Cost: under a day.
3. **Unit ablation for heredity.** Question: is the clean neuron needed, or does any featurisation work? Data: existing. Method: run E2 with meme families, random non-meme units, mean-pooled residual stream, an open SAE for Llama 3.1 8B if one exists, and a sentence-embedding model. Rival: TF-IDF and sentence embeddings. Falsified (for P1) if random units or pooled embeddings equal the meme units. Cost: one to two days.
4. **Paraphrase ladder.** Question: at what level of rewording does each method lose the source? Data: the 9 adaptations and 12 comics, rewritten at 4 or 5 levels of content-word replacement by an LLM, checked by a second reader. Method: E2 statistic at each level. Rival: TF-IDF, sentence embeddings, an LLM asked to name the source. Falsified if neurons degrade as fast as TF-IDF, or if embeddings or the LLM reader stay ahead at every level. Cost: about 100 rewrites plus extraction; two to three days.
5. **Cross-language heredity.** Question: do neurons find the source when the adaptation summary is in another language? Data: Spanish, French, German or Japanese Wikipedia plots of the same adaptations against English comics. Method: E2 statistic. Rival: multilingual sentence embeddings, machine-translated TF-IDF, LLM reading. Falsified if neurons fall to chance or trail multilingual embeddings. Cost: two days.
6. **Concept-unit minimal pairs at scale.** Question: how many units behave like Llama's revenge unit? Data: 20 to 30 story concepts, 12 or more minimal pairs each, plus decoys that use the concept's words without the concept. Method: pre-register unit choice on dev pairs, score on test pairs, in three models. Rival: keyword list, sentence-embedding similarity to a concept description, zero-shot LLM judge. Falsified if fewer than about a quarter of concepts yield a unit above keyword accuracy, or if the LLM judge beats every unit at lower cost. Cost: three to five days.
7. **Identity with enough negatives.** Question: can name-free Batman be told from other superhero stories? Data: name-free plots of 100 or more superhero films plus the 32 Batman works. Method: cross-validated classifier on unit presence. Rival: TF-IDF logistic regression, text length alone, sentence embeddings. Falsified if the neuron classifier does not beat TF-IDF and length. Cost: three days.
8. **Contamination control.** Question: do units respond to memorised Wikipedia text or to content? Data: fresh summaries of the same works written from scratch, and works released after the models' training cut-offs. Method: compare presence grids and E2 on fresh vs Wikipedia text. Rival: TF-IDF on the same fresh text. Falsified (as a worry) if presence and heredity hold equally on fresh text. Cost: two days.

## Key sources

- `study/results/experiments.json`. All E1 to E3 numbers, word comparators, residual test, stricter nulls. Priority A.
- `study/results/robustness/nulls.json`. Twin nulls by unit class, tree floors, rewording null, source recovery from trees. Priority A.
- `study/paper/paper.md`. The frozen v1 paper, 20 October 2026. Priority A.
- `study/results/robustness/e2_multiverse.json`, `e2_jackknife.json`, `e2_reimpl.json`. E2 robustness; neuron-only. Priority B.
- `study/results/phylo.json.gz`. Per-model tree statistics including Qwen3-1.7B and the TF-IDF tree. Priority B.
- `study/results/wording.json`, `study/results/robustness/label_test.json`. Rewording, revenge and label tests. Priority B.
- `study/results/design_reviews.json`. The three adversarial design reviews; includes an uncommitted probe of E2 on Qwen3-1.7B original texts (9 of 9, p = 0.0008). Priority C.
- Cammarata N. "Neural networks are already clean". Goodfire. https://already-clean.goodfire.com/ (page fetched, HTTP 200, title confirmed). Source of P1. Priority A.
- Gurnee W et al. 2024 "Universal neurons in GPT2 language models". arXiv:2401.12181 (confirmed via arXiv API). Source of P2. Priority A.
- Huh M et al. 2024 "The platonic representation hypothesis". arXiv:2405.07987 (confirmed). Priority B.
- Jha R et al. 2025 "Harnessing the universal geometry of embeddings". arXiv:2505.12540 (confirmed). Priority B.
- da Silva SG, Tehrani JJ. 2016. doi:10.1098/rsos.150645 (confirmed via Crossref). Template for E2. Priority C.
- Movva R et al. 2025 "Sparse autoencoders for hypothesis generation". arXiv:2502.04382 (confirmed). A rival unit. Priority C.

## Fog

- I did not re-trace the canon tier counts (56 / 194 / 14,593) or the 9,156 replaced spans to their committed files; they are quoted from the paper.
- I did not open `study/experiments.py` or `study/robustness/nulls.py`, so I have not checked how the residual regression, the hub adjustment or the word-matched null are coded.
- The sequel-control chance figures (2.1 expected, P(>= 4) = 0.10) are my arithmetic from the pool sizes in EXP, assuming random ranking. No committed script produces them.
- Whether the Llama residual signal is real is the single most important open point and cannot be settled from committed outputs.
- Whether open SAE or transcoder features exist for Qwen3-8B or Llama 3.1 8B at the layers that matter: not checked.
- Whether the Goodfire "already clean" claim has been independently replicated, and on models this small and quantised: not checked here (other agents' topic).
- Training-data overlap between these Wikipedia pages and Qwen3 / Llama 3.1 is assumed likely, not verified.
- The blind label test's judges were AI and item-level answers were discarded, so the 79% cannot be audited.
