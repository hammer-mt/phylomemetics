# Pre-registration: when, and at what cost, do neuron memes beat keywords, vectors and topics?

**Project:** phylomemetics (github.com/hammer-mt/phylomemetics), by Mike Taylor and Claude Opus 5.5.
**Version:** prereg-v1. Drafted 2026-10-20. It is frozen by the commit that adds this folder, once that commit is pushed.
**Status:** no outcome has been computed on any TEST data. The scouting behind this design only read metadata: existence, plot-section length, documented "based on" relationships and, for Shakespeare films, whether the play's character names appear. It also timed the models on made-up text and checked the new extractor against three DEV works.

## 0. How to read this document

- This document fixes the questions, the data, the decisions and the predictions.
- `analysis.py` fixes every computational detail: statistics, settings and seeds. `extract_compact.py` and `readers.py` fix how the model-side features are made, including every LLM prompt.
- The five protocol files fix how agents make the new data:
  - `planted_story_protocol.md`
  - `drift_protocol.md`
  - `translation_protocol.md`
  - `name_removal_protocol.md`
  - `annotation_protocol.md`
- The work lists fix which works are used, with a hash of each Wikipedia text:
  - `works_shakespeare.json`
  - `culture_pool.json`
  - `documented_relatives.json`
  - `works_batman_extended.json`
  - `memes_for_tracing.json`
  - `planted_design.json`
  - `dev_units.json`

If this document and the code disagree about a computational detail, the code wins, and the disagreement is logged as an erratum in `DEVIATIONS.md`. If they disagree about a hypothesis, a prediction or a decision rule, this document wins.

`test_analysis.py` runs every analysis end to end on synthetic data with known answers. All 62 checks pass.

## 1. The question

### 1.1 What Mike wants the method for

The point of neuron memes is that nobody has to choose or count words. A neuron should capture a fuzzy idea and keep tracking it when the words change, the way a fringe group renames things to dodge keyword moderation. Mike wants the method to:

1. say what Batman actually is;
2. trace the shared history between Batmans;
3. see when Batman mutates and which mutations catch on (a measure of cultural selection);
4. find Batman-like things across culture, especially the memes that Batman shares with the other "billionaire vigilante orphans" of Mike Taylor's earlier book.

For the paper, he wants the experiment that shows how much more accurate, or how much cheaper, neurons are than vectors (embeddings) and keywords.

### 1.2 What DEV already showed

The robustness review of the first study found three things:

- **Heredity is real.** Name-free summaries of 9 Batman adaptations point to their documented source comic. That holds in 99.6% of 30,240 analysis variants.
- **Words do just as well.** Plain TF-IDF word overlap recovers the sources as well as the neurons do, and LSA does better.
- **The "beyond words" test splits by model.** The pre-registered test fails for Qwen and passes only for Llama.

On name-free Wikipedia summaries, then, the neurons have not yet earned their extra cost. That points to where they could earn it: wherever the words change but the idea stays the same. Every test below is built around that.

### 1.3 Organising question

> When, and at what cost, do neuron-defined memes beat keywords, vectors and topics?

Every test runs the same methods on the same inputs. The tests either change the words on purpose (drift, translation, banned keywords) or use lineages where the words changed naturally (loose Shakespeare adaptations). Every test reports accuracy and cost together.

### 1.4 The five primary hypotheses

Each hypothesis is tested in Qwen3-8B, the primary model, and replicated in Llama 3.1 8B. All five tests are one-sided, with Holm correction across the five at a familywise α of 0.05.

| | Test | Hypothesis | Statistic | Falsified when |
|---|---|---|---|---|
| **H1** | T2 planted memes | When a meme's usual words are banned (reworded stories), neuron units detect it better than an analyst's keyword search. | Mean over the 24 memes of AUC(neurons, few-shot unit) minus AUC(keywords chosen by Qwen3-8B); sign-flip over memes | Holm-adjusted p ≥ 0.05, or the mean difference ≤ 0 |
| **H2** | T2 planted memes | On reworded stories, neurons are not worse than sentence embeddings by more than 0.05 AUC. If they pass, they are tested for being better. | Mean over memes of AUC(neurons) minus AUC(multilingual-e5-base prototype), plus 0.05; sign-flip | Holm-adjusted p ≥ 0.05 (then neurons may be more than 0.05 AUC worse than embeddings) |
| **H3** | T2 planted memes | The units found and labelled on the DEV Batman corpus, unchanged, fire on their memes when those memes are planted, reworded, in new stories in unrelated genres. | Mean over the 18 DEV-mapped memes of AUC minus 0.5; sign-flip | Holm-adjusted p ≥ 0.05, or the mean ≤ 0.5 |
| **H4** | T1b vocabulary drift | As a text's words are replaced, TF-IDF's recovery of the documented source degrades more than the neurons' does. | Mean over 30 clusters (16 Batman targets and 14 Shakespeare source plays) of degradation(TF-IDF) minus degradation(neurons); sign-flip | Holm-adjusted p ≥ 0.05, or the difference ≤ 0 |
| **H5** | T1d loose Shakespeare adaptations | On a fresh lineage whose adaptations change names, setting and wording, neurons recover the source play better than TF-IDF does. | Play-weighted mean over the 83 loose adaptations of share(neurons) minus share(TF-IDF); sign-flip over the 14 source plays | Holm-adjusted p ≥ 0.05, or the difference ≥ 0 |

**Replication.** A primary hypothesis replicates in Llama when the same test gives p < 0.05 (one-sided, uncorrected) in the same direction.

**How the outcomes read.**

- **H1 and H3 both pass.** Neurons keep tracking memes where keyword search fails. That is the fuzziness claim.
- **H2.** This says whether embeddings do the same job. If H2 fails, the neurons' remaining case is interpretability (T6) and discovery without supervision, not accuracy.
- **H4 and H5.** These say whether the same holds for whole-story lineage, under forced and under natural wording change.
- **Cost (T6).** This says when neurons are cheaper than LLM reading, which is likely the most accurate rival.

A result in which LLM reading beats everything is a finding, and is reported as one.

## 2. Methods compared

Every method gets the same inputs: the same texts and, for named memes, the same label and 5 example sentences. Sentence-level methods are turned into story or work scores the same way the neurons are: **the mean of the 3 strongest sentences** (`analysis.work_from_sentences`). Settings are fixed here. Neurons receive no tuning: their pipeline is the DEV one.

The only choice made on DEV is the hidden-state layer for the same-model vector rival. Nothing is tuned on TEST data. A sensitivity analysis also reports the best of the four encoders on each TEST outcome, chosen with hindsight. It is labelled as favouring the vectors.

### 2.1 Neurons (primary: Qwen3-8B; replication: Llama 3.1 8B)

- **Models.** 4-bit MLX (`mlx-community/Qwen3-8B-4bit`, `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`). Each text is run as one continuous document.
- **Units.** A unit is one SwiGLU channel, split by sign: 884,736 units in Qwen and 917,504 in Llama. A unit's sentence value is the maximum over the sentence's tokens (for a negative pole, minus the minimum).
- **Work score.** The mean of the work's 3 strongest sentences.
- **Presence.** A unit is present in a work when its score beats all but one of the reference films. The reference films are the 69 DEV ordinary films, in the same text condition as the analysis: name-free, masked, original or Chinese.
- **Extraction.** `extract_compact.py` stores per-work aggregates for all units, and per-sentence values only for a fixed unit list. On three DEV works it reproduces extract.py's scores at r = 0.99997 to 0.99998 in both models. Reference films and test works always come from the same extractor.
- **Lineage similarity** (`neurons:<model>`, the E2 rule). Similarity between two works is the share of the candidate's *optional* units that are also present in the row work. Optional units are those present in between 2 works and half of the lineage's works; for Batman's 32 works that is E2's band of 2 to 16.
  - Secondary: `neurons-cos`, the cosine of reference-standardised unit scores.
- **Named memes, few-shot selection** (`neurons:<model>:m1`, primary). Every unit is scored on the meme's 5 example sentences, by the mean of each example's percentile within the unit's distribution over the DEV reference-film sentences (original texts, 2,267 sentences). Examples and background sentences are each run as a separate document after the same fixed prefix sentence, so position and context affect both alike. The background is contrastive: it also holds the other 23 memes' examples, so a unit that only detects how the examples are written (nameless, explicit, in Claude's style) cannot win. Ties are broken by the margin over the 99th percentile. The best unit is the meme's detector.
  - `m5` (secondary) averages the 5 best units after background standardisation.
  - The selection uses only the examples and DEV data, so it is frozen before any TEST text exists.
- **Named memes, DEV-mapped units** (`devunit:<model>`, H3). There are 18 memes with a DEV-labelled Qwen unit (`dev_units.json`). Llama uses each unit's DEV twin, the best-correlated Llama unit on DEV text; for revenge, it uses Llama's DEV-chosen revenge unit.

### 2.2 Keywords

- **`tfidf`** (lineage). TF-IDF cosine with phylo.py's settings: English stop words, min_df 2, sublinear tf. X-codes are stripped.
  - The vocabulary is fitted on the analysis texts plus their reference films.
  - Chinese text uses character uni- and bigrams.
- **`words`** (lineage). The containment of optional words: binary word presence through the same band rule as the neurons.
- **Keyword queries** (named memes). A query is a list of terms, and a sentence's score is the sum over terms:
  - a one-word term counts each occurrence of its stem;
  - a multi-word term counts 1 when all its content-word stems occur in the sentence.

  The stemmer is `analysis.stem`. A story's score is the mean of its 3 best sentences, with total hits as a 0.001 tie-break. There are four queries:
  - `keywords:qwen3-8b` (**primary**): Qwen3-8B, given the label, definition and examples, lists up to 10 search terms (`readers.KEYWORDS`). This models a human analyst.
  - `keywords:llama3.1-8b`: the same prompt, given to Llama.
  - `keywords:tfidf`: the top 10 unigrams and bigrams by TF-IDF in the examples minus the same contrastive background (reference-film sentences plus the other memes' examples), each occurring in at least 2 examples.
  - `keywords:claude`: the same prompt given to Claude. This is a labelled upper bound, since Claude also wrote the ban lists.
- **The ban lists are not queries.** The `keywords` field of `memes_for_tracing.json` holds each meme's usual words. They are the ban list for planted stories and are never given to any method.

### 2.3 Vectors

- **Sentence embeddings** (`emb:<encoder>`), at pinned revisions:
  - `e5-base` (**primary**), intfloat/multilingual-e5-base;
  - `e5-large`, intfloat/multilingual-e5-large;
  - `labse`, sentence-transformers/LaBSE;
  - `minilm`, sentence-transformers/all-MiniLM-L6-v2.

  Every sentence is embedded on its own and L2-normalised; e5 inputs carry the "query: " prefix.
  - Lineage: a work is the mean of its sentence embeddings, and similarity is cosine.
  - Named memes: the prototype is the mean of the example embeddings, and a sentence's score is its cosine to the prototype.
- **`lsa`.** 100-dimensional truncated SVD of the TF-IDF matrix (fewer dimensions if the corpus is small); cosine.
- **`hidden:<model>`** (secondary). The same model's mean residual-stream state over each sentence, at one of four stored layers: Qwen 8, 17, 26, 35; Llama 7, 15, 23, 31. The layer is the one with the best DEV E2 statistic. Vectors are standardised on the reference films; similarity is cosine, and named-meme scoring uses a prototype. This separates the neuron basis from the model's dense representation.

### 2.4 Topics

- **`lda`.** sklearn LDA (batch variational Bayes, default priors, 20 iterations, seed 0) on word counts with English stop words and min_df 2. K is fixed per use:
  - 30 topics, with works as documents, for lineage;
  - 50 topics, with sentences as documents, for T2. A meme's topic is the one most raised in its examples relative to the same contrastive background, and a sentence's score is that topic's probability;
  - 50 topics, with works as documents, for T3;
  - 100 topics, with sentences as documents, for T6.
- **`topics:e5-base`** (BERTopic-style, implemented with sklearn). e5-base sentence embeddings are reduced with UMAP to 5 dimensions (15 neighbours, min_dist 0, cosine), then clustered with HDBSCAN (min_cluster_size 15). If umap-learn is not installed, PCA to 10 dimensions replaces UMAP; that would be a reported deviation.
  - Lineage: a work is its histogram over clusters; cosine.
  - Named memes: a meme's cluster is the one nearest most of its examples, and a sentence's score is minus its distance to that centroid.
  - Clusters are labelled with c-TF-IDF for T6.

### 2.5 LLM reading (the local models in generation mode; Claude only as a labelled upper bound)

The readers are Qwen3-8B with thinking off and Llama 3.1 8B Instruct, through `readers.py`. Every scored answer is one next-token probability: p_yes = P(yes) / (P(yes) + P(no)). Generation is greedy. The prompts are fixed in `readers.py`.

- **`judge:<model>`.** It reads the whole story, then answers one question per meme, giving the label, the definition and the 5 examples. The story prefix is cached, so every further meme costs only its question.
- **`summary-judge:<model>`.** This is recursive summarisation. Each block of 8 sentences is summarised in at most 60 words, and the block summaries are merged into one summary of at most 150 words. The judge then reads the summary instead of the story.
- **`llm:<model>`** (lineage). It reads two works' recursive summaries and gives P(yes) that work B is an adaptation of work A. That number is the pair's similarity.
- **`llm likeness`** (T3). It gives P(yes) that the story, names aside, is like the Batman stories, read from the summary.
- **`judge:claude-upper-bound`** (T2). Claude reads all 5 versions of 61 plots, 305 stories, in which every meme is planted 6 to 10 times (annotation_protocol.md A4). Claude wrote the test texts, so this is an upper bound, not a rival.

## 3. DEV, TEST and the order of work

- **DEV** is the existing Batman corpus (`data/corpus.jsonl`, `data/corpus_neutral.jsonl`, 114 works, 3,700 sentences) and every existing result.
- **TEST** is all new data:
  - drift rewrites;
  - translations;
  - the Shakespeare plays and films;
  - the 43 new Batman works;
  - the culture pool and relatives;
  - the planted stories;
  - the natural-sentence annotations.

The example sentences (5 per meme) and the ban lists in `memes_for_tracing.json` are inputs, written for this pre-registration and frozen with it.

Nobody runs a method on TEST data, or looks at a TEST outcome, before steps 0 to 5 are complete. There are three freeze points, F0, F1 and F2. At each one the named files are committed **and pushed**, and the commit hash is noted in `DEVIATIONS.md`.

**Step 0, freeze F0.** Commit and push this folder.

**Step 1, approvals.** Mike approves:
- installing sentence-transformers 6.1.0 and umap-learn 0.5.12 (with numba and pynndescent);
- downloading the four encoders at the revisions in `readers.ENCODERS`: about 5.3 GB in total.

Nothing else is installed.

**Step 2, DEV features.** All of this runs on DEV only:

```
python prereg/build_lists.py texts dev
python prereg/extract_compact.py dev_namefree <m> --full --hidden      # both models; T5 partners, hidden layer, DEV E2
python prereg/extract_compact.py dev_original <m> --hidden
python prereg/extract_compact.py t2_background <m> --full --hidden --prefix   # reference-film sentences, one per document
python prereg/extract_compact.py t2_examples <m> --full --hidden --prefix
python prereg/extract_compact.py t3_dev_masked <m>
python prereg/readers.py embed --set <dev set> --encoder <e>           # four encoders, the five sets above
python prereg/readers.py keywords --model qwen3-8b | llama3.1-8b | tfidf   (+ Claude per annotation_protocol A5)
python prereg/readers.py zh-keywords --source qwen3-8b | tfidf
python prereg/readers.py summarise --set dev_namefree --model <m>; pairs --rows dev_namefree:screen --cands dev_namefree:comics
```

**Step 3, freeze F1 (DEV freeze).** `python prereg/analysis.py dev-freeze` writes `dev_freeze.json` and `sentence_units_<model>.npy`. It holds:

- the T2 units per meme;
- the keyword queries;
- the embedding prototypes;
- the hidden-state layer;
- the T5 partners;
- the per-sentence unit list;
- the DEV reproduction check;
- every method's DEV E2 result (T1a).

The DEV reproduction check must pass: compact-extractor E2 within 0.03 of the published 0.164 (Qwen) and 0.102 (Llama). If it fails, stop and fix the extractor. Commit and push.

**Step 4, TEST data creation.** Agents run the protocols, in this order:
1. name removal;
2. drift;
3. planted stories, with blind reads;
4. translation of the T1c texts and of the T2 banned and control stories;
5. T2-N annotation.

Then assemble the sets:

```
python prereg/build_lists.py texts wikipedia | names | drift | stories | zh | t2n
```

These checks happen here, before any method runs:
- every Wikipedia text is checked against its frozen hash;
- every quality check in the protocols runs;
- exclusions are recorded.

**Step 5, freeze F2 (TEST data freeze).** Commit and push `prereg/data/` (the texts and `manifest.json` with every set's hash) and `prereg/agent/` (all agent outputs and quality-check records).

**Step 6, TEST features.** These are the commands, for both models `<m>` and the four encoders `<e>`. The sets are named in `build_lists.py`.

```
# neurons
for S in t1b_drift{25,50,75,100} t1b_fresh_drift{25,50,75,100} t1b_shk_drift{25,50,75,100} t1e_fresh_namefree \
         t1c_zh t1c_fresh_zh t1d_plays_namefree t1d_films_namefree t4_batman; do extract_compact.py $S <m> --hidden; done
# (t1d_*_original and t1c_*_mt are used by word methods only: no neuron or embedding features)
for S in t2_stories t2_stories_zh t2n_works t3_pool t2_reference; do extract_compact.py $S <m> --units prereg/sentence_units_<m>.npy --hidden; done
# (t2_reference = the 69 DEV reference films as whole works: DEV data, extracted here because the unit list is fixed at F1)
# embeddings (every set with neuron features above)
readers.py embed --set S --encoder <e>
# LLM reading
readers.py judge --set t2_stories --model <m>;  judge --set t2_stories_zh;  judge --set t2n_works;  judge --set t3_pool --tier A
readers.py summarise --set S --model <m>        # every T1 set, the comics (dev_namefree), t2_stories, t3_pool
readers.py sjudge --set t2_stories --model <m>
readers.py pairs --rows <screen set>:screen --cands dev_namefree:comics    # dev_namefree, t1b_drift*, t1c_zh
readers.py pairs --rows <fresh set>:all --cands dev_namefree:comics        # t1e_fresh_namefree, t1b_fresh_drift*, t1c_fresh_zh
readers.py pairs --rows t1d_films_namefree:primary --cands t1d_plays_namefree
readers.py pairs --rows t1b_shk_drift<L>:all --cands t1d_plays_namefree
readers.py likeness --set t3_pool --model <m>
readers.py mt --set t1c_zh --model qwen3-8b;  mt --set t1c_fresh_zh --model qwen3-8b   # writes t1c_zh_mt, t1c_fresh_zh_mt
```

The Claude upper bound follows annotation_protocol A4. Then `python prereg/analysis.py t6-prepare` builds the label-test items, and agents follow annotation_protocol A3.

**Step 7, analysis.** `python prereg/analysis.py t1 t2 t3 t4 t5 t6 cost primary` writes `prereg/results/*.json`.

**Step 8, report.** Everything listed in section 8 is reported.

## 4. Tests

### T1. Lineage recovery under wording change

**The statistic (all of T1).** This is the E2 statistic, generalised (`analysis.lineage_test`):

- **Inputs.** Each method gives a similarity between every row work (targets and hub works) and every candidate source.
- **Hub adjustment.** Each similarity is turned into a z-score against the same candidate's similarities to the *hub* works. This removes candidates that resemble everything.
- **Score per target.** The share of the eligible, non-source candidates whose z beats the best documented source. Lower is better recovery; chance is about 0.4 to 0.5.
- **Statistic.** The (weighted) mean of that share over targets.
- **Null.** Joint relabelling: one random permutation of the candidates moves every documented source at once. Where the hub rule depends on the labels, it is recomputed for the relabelled sources. There are 20,000 permutations; p is one-sided.

**Head-to-head between two methods** (`analysis.method_contrast`). Take the per-target difference in share, average it within clusters (a source play, or the target itself), then apply a sign-flip test over clusters. The test is exact when there are at most 2^21 sign patterns, and uses 200,000 Monte Carlo flips otherwise.

**Beyond words** (`analysis.residual_sim`). Regress the neuron similarity on TF-IDF cosine and word containment over all row-by-candidate pairs, then rerun the statistic on the residuals.

**Methods.** All of section 2 that apply to whole works:
- neurons and neurons-cos, both models;
- tfidf, words, lsa, lda;
- emb with all four encoders;
- topics:e5-base;
- hidden, both models;
- llm pairs, both models.

#### T1a. DEV reference (not confirmatory)

Every method is run on the DEV name-free E2 (9 adaptations, 12 comics, 11 hub works, time rule). The result is computed at freeze F1, before any TEST step, and reported as DEV. It tells us where each method starts on texts written as Wikipedia summaries.

#### T1b. Vocabulary drift (primary: H4)

**Data.**
- Batman: the 20 DEV screen works and the 7 fresh adaptations (T1e). The 9 DEV adaptations and the 7 fresh ones are the 16 targets; the 12 DEV comics are the candidates. The hub is the 11 DEV screen works that adapt no comic, drifted at the same level. The time rule applies.
- Shakespeare: the 83 loose adaptations, which are both the targets and the hub, drifted at the same level. The 40 plays are the candidates and are never drifted.
- Levels 0, 25, 50, 75 and 100% of content-word types replaced, under `drift_protocol.md`. Each level is rewritten independently from level 0, and the banned sets are nested.

**Procedure.** For every method, compute each target's share at every level.

**Statistic.** For each target t, degradation D(t) = (mean of share at levels 25, 50, 75 and 100) − share at level 0. H4's statistic is the mean over clusters of D_TF-IDF(t) − D_neurons(t). Each of the 16 Batman targets is its own cluster, and each of the 14 Shakespeare source plays is one cluster, averaging its films. That gives 30 clusters of equal weight.

**Null.** Sign-flip over the 30 clusters (Monte Carlo, 200,000), one-sided.

**Prediction.** The statistic is above 0: TF-IDF degrades more.

**Falsifier.** A Holm-adjusted p of 0.05 or more, or a statistic of 0 or less.

**Secondary.**
- The recovery-against-drift curve for every method, with the lineage p at every level.
- The same interaction for Batman alone and for Shakespeare alone.
- The interaction against every other rival.
- Predictions:
  - S1: e5-base degrades less than TF-IDF.
  - S2: LLM pairwise reading degrades least of all methods, reported as an estimate.
  - There is no directional prediction for neurons against e5 or against LLM reading.
- A sensitivity analysis without works flagged by the drift checks.

#### T1c. Cross-language (secondary)

**Data.** The Batman design of T1b at level 0, with the 27 screen works and the 69 reference films translated into Chinese (`translation_protocol.md`). The 12 comics stay in English, and the presence bars are set on the translated reference films. There are 16 targets.

**Predictions.**
- S3: Qwen neurons recover the sources (p < 0.05).
- S4: tfidf, words, lsa and lda do not (p ≥ 0.05; their overlap is near zero by construction).
- S5: e5-base and LaBSE do (p < 0.05).
- There is no directional prediction for neurons against e5, LaBSE or LLM reading; these are two-sided contrasts.
- Llama is expected to be weaker, because Chinese is not one of its supported languages.
- The machine-translation-plus-TF-IDF pipeline is reported with its cost.

#### T1d. A fresh lineage: Shakespeare (primary: H5)

**Data** (`works_shakespeare.json`).
- The candidates are the 40 plays (1,438 sentences after the 40-sentence cap).
- The rows are the 141 primary film adaptations, each with one documented play. They are not in DEV, and their plots name neither Shakespeare nor a non-eponymous play title.
- Names are removed from all texts under `name_removal_protocol.md`.
- The reference films are the 69 DEV films without The Lion King, which is a documented Hamlet adaptation.
- The hub for play c is every primary film that adapts another play.
- There is no time rule: every film is later than every play.
- Weights are play-weighted: each source play counts once.

**H5 sample.** The 83 loose adaptations: primary films whose plots contain fewer than 2 of the play's distinctive principal names, from 14 source plays. These are the adaptations that changed names, setting and wording.

**Statistic.** The play-weighted mean of share(neurons) − share(TF-IDF).

**Null.** Sign-flip over the 14 plays (exact, 16,384 patterns), one-sided.

**Prediction.** The statistic is below 0: neurons recover the source better.

**Falsifier.** A Holm-adjusted p of 0.05 or more, or a statistic of 0 or more.

**Secondary.**
- **S6, beyond words.** The residual test on the loose sample. This was the old pre-registration's falsifier: residual p ≥ 0.05 means the neurons add nothing beyond words. The prediction is p < 0.05.
- The same contrasts against lsa, words, lda, e5-base, topics, hidden states and LLM reading.
- All 141 primary films, and the 58 faithful films (those that keep names or keep them partly), with the difference between the loose and faithful contrasts reported as an estimate.
- **Positive control.** On the original texts, with names, TF-IDF recovers the sources of the films that keep names with a share of 0.10 or less. Otherwise T1d's pipeline is broken and H5 is inconclusive.

#### T1e. Fresh Batman adaptations (secondary)

These are the 7 animated adaptations of DEV comics that are new to the project (TEST): Year One (2011), The Dark Knight Returns (2012), The Killing Joke (2016), Hush (2019), Death in the Family (2020), The Long Halloween (2021) and Knightfall (2026). Each has its source documented in its Wikipedia lead.

**Prediction (S7).** Every method recovers the sources (p < 0.05). These summaries follow their comics closely, so the test is a positive control for the fresh-data pipeline rather than a test of wording change.

### T2. Named memes: planted stories (primary: H1, H2, H3)

**Memes** (`memes_for_tracing.json`). There are 24 memes in three tiers:

- **Tier A (10), the orphan-vigilante family:** secret identity, origin wound, heir to a dead father's fortune, wealthy vigilante, the servant who raised the orphan, orphan ward, foppish facade, a symbol to strike fear, hidden lair, taking up the mantle.
- **Tier B (10), Batman story memes:** coin flip decides a fate, vat of chemicals, funeral, morbid joke, losing the protégé, corrupt police force, fear toxin, mass breakout, the hero refuses to kill, race for an antidote.
- **Tier C (4), general memes:** revenge vow, mistaken twins, death-feigning potion, deal with the devil.

Each meme has a definition, a ban list of its usual words, 5 example sentences in explicit wording (no names, varied settings) and a decoy hint. Eighteen memes have a DEV-labelled Qwen unit.

**Stories** (`planted_design.json`, `planted_story_protocol.md`).
- There are 120 new plots in 12 genres unrelated to Batman, written by 4 independent writer agents.
- Each plot plants 3 memes, and each meme is planted in 15 plots.
- Each plot has 5 versions with an identical frame:
  - **explicit:** the meme in its usual words;
  - **banned:** the meme present, with none of its usual words anywhere in the story;
  - **implicit:** the meme shown only through events;
  - **control:** a neutral event instead;
  - **decoy:** the meme's words without the meme.
- That makes 600 stories of 18 to 30 sentences.
- Blind readers must recognise every planted meme in the explicit and banned versions (both readers) and in the implicit versions (at least one reader), and find no planted meme in the control and decoy versions. Failures are rewritten once, then excluded for that meme.

**Detection AUC** (`analysis.detection_aucs`). For meme m and a condition C (a set of versions):

- positives are the valid stories that plant m in a version in C;
- negatives are every valid story that does not contain m: its own control and decoy versions, and every version of the plots that do not plant it;
- a story that quality control excluded for m (`exclude_memes`) is in neither set.

So for each meme, about 30 reworded positives face about 555 negatives. The **reworded** condition is C = {banned, implicit}.

**H1.**
- Statistic: the mean over memes of AUC(`neurons:qwen3-8b:m1`) − AUC(`keywords:qwen3-8b`), in the reworded condition.
- Null: sign-flip over memes (Monte Carlo, 200,000), one-sided.
- Prediction: above 0.
- Falsifier: a Holm-adjusted p of 0.05 or more, or a mean of 0 or less.
- The banned condition was designed to defeat a meme's usual words. So the informative results are the size of the gap, the neurons' own AUC, and how neurons compare with vectors and LLM reading.

**H2.**
- Statistic: the mean over memes of AUC(neurons) − AUC(`emb:e5-base`) + 0.05, in the reworded condition.
- Null: sign-flip, one-sided.
- Prediction: above 0, meaning neurons are non-inferior within 0.05 AUC.
- If H2 holds, the superiority test (the same statistic without the +0.05) is run at the same Holm level, as a fixed sequence. "Neurons beat embeddings" is claimed only if it passes.
- Falsifier: a Holm-adjusted p of 0.05 or more. Neurons may then be more than 0.05 AUC worse than embeddings.

**H3.**
- Statistic: over the 18 memes with a DEV unit, the mean of AUC(`devunit:qwen3-8b`) − 0.5, in the reworded condition.
- Null: sign-flip, one-sided.
- Prediction: above 0.
- Falsifier: a Holm-adjusted p of 0.05 or more, or a mean of 0.5 or less.

**Secondary** (Benjamini-Hochberg within T2). All of these are computed for every method:

- AUC per condition: explicit, banned, implicit, reworded.
- Hard-negative AUC: against the meme's own control and decoy versions only.
- Paired twin accuracy: whether a planted version beats its own control.
- R-precision.
- F1 at the method's presence bar: the second-highest score over the 69 reference films, or 0.5 for the LLM judges.
- Results by writer and by tier.

The predictions are:
- S8: `judge:qwen3-8b` beats the neurons on reworded stories.
- S9: the summary judge scores below the direct judge.
- S10: every method has a lower AUC on implicit than on banned.
- S11: keyword methods' twin accuracy for decoy against control is above 0.5 (decoys fool keywords).

There are no directional predictions for m5 against m1, hidden states against neurons, LDA or topics against neurons, or Llama against Qwen. The best-encoder-in-hindsight sensitivity analysis is reported.

**T2-zh (secondary).** The banned and control versions of every valid plot are translated into Chinese (240 stories), and the examples stay in English. Keyword queries are translated by Qwen.

- S12: Qwen neurons' banned-condition AUC is above 0.5.

**T2-N, natural plot summaries (secondary).** The 120 TEST works with role batman_lineage, relative or book_family (masked texts) are annotated exhaustively by two blind annotators plus an adjudicator (annotation_protocol A2). Work-level AUC is computed per meme, for memes with at least 3 positive works.

- S13: Qwen neurons' mean AUC is above 0.5.

Agreement (kappa) is reported. Natural sentences are not planted, so this tests real Wikipedia wording.

### T3. Batman-likeness across culture (secondary, with pre-stated predictions)

**Data** (`culture_pool.json`, 2,544 works, 62,879 sentences, masked with `analysis.mask_names` like the DEV reference).

| Part of the pool | Works | Notes |
|---|---|---|
| Category pool | 1,527 | 21 Wikipedia category roots: superhero, vigilante, swashbuckler, Zorro, Pimpernel, Monte Cristo, Iron Man, Puss in Boots, Green Hornet, Phantom, Doc Savage, Spider, Shadow, pulp, Robin Hood, superhero novels. Eligible works are films, TV films, serials, novels and short stories with at least 8 plot sentences, not in DEV |
| of which vigilante-only | 400 | A hash subsample of the 1,291 works found only through the vigilante category (that bucket pulls in revenge and lynching films). Relatives and book-family works in it are always kept |
| Random films | 1,000 | From every 'Category:YYYY films' list, 1915 to 2024, in sha256 order |
| Documented relatives outside the pool | 17 | The Bat play and films, three Sherlock Holmes novels, Dick Tracy, The Man Who Laughs, 'Pimpernel' Smith, Watchmen, Birdman and others |

Each work has a role:
- pool, 2,409 negatives: superhero 743, swashbuckler 191, pulp_crime 475, random 1,000;
- relative, 56 works of 19 documented entities with strength A or B (`documented_relatives.json`);
- book_family, Mike Taylor's earlier book tiers that are not documented relatives;
- batman_lineage, 43 works;
- monte_cristo_unverified, 15 works, excluded from the negatives.

Book tier 1 has 30 works (Zorro 19, Pimpernel 8, Iron Man 3); tier 2 has 22 (Spider-Man 17, John Wick 3, Puss in Boots 2). The tiers come from the book's pp. 21 to 22, quoted only as short phrases.

**Scores** for every vector method (neurons, both models; tfidf; lda K=50; the four encoders):

- **Batman-likeness:** the cosine of a work to the centroid of the 32 DEV Batman works. Both are centred on the 69 masked reference films, and neuron units are standardised on them.
- **Superhero-likeness:** the cosine to the centroid of the pool's superhero-stratum negatives.
- **Batman-specific likeness:** Batman-likeness residualised on superhero-likeness across the pool. This separates Batman's own layer from the generic superhero layer.
- **The orphan-vigilante layer:** the mean within-pool percentile of a work's scores on the 10 Tier A memes, using each method's T2 detectors (neurons, keywords, embeddings, LLM judge).
- **LLM likeness:** P(yes) that the story is like Batman's, from its summary.

**Tests** (`analysis.stratified_perm_test`). The statistic is the entity-weighted mean, over positives, of each positive's AUC against the pool works of its own stratum. Each entity, or each book character, counts once. The null moves the positive labels at random within each stratum, 10,000 times; p is one-sided.

**Predictions.**
- **S14, positive control.** The 43 Batman-lineage works outrank the pool on Batman-likeness (AUC ≥ 0.9) for neurons and TF-IDF. If both fail, T3 is inconclusive.
- **S15.** Documented relatives score above their strata on Batman-specific likeness (Qwen neurons, p < 0.05).
- **S16.** Book tier 1 (Zorro, Pimpernel, Iron Man works) scores above its strata on the orphan-vigilante layer (Qwen neurons, p < 0.05). This is Mike's book claim, tested on works not in DEV.
- **S17.** Book tier 2 likewise.

Method contrasts over entities are two-sided, with no prediction. The 50 most Batman-like works in each method's ranking (Batman-specific likeness) are reported as descriptive results.

### T4. Mutation and selection (secondary; partly DEV)

**Data.** The extended lineage: the 32 DEV Batman works (name-free) and the 43 TEST works (name-free, under name_removal_protocol), 75 in all, in date order. Presence bars come from the 69 DEV name-free reference films. **"First appearance" means first in this sample**, not first in Batman's history. The sample has a gap from 1966 to 1986, and comics after 2017 are missing.

**Definitions.**
- A *mutation* of work w is a unit present in w and in no earlier work.
- A mutation's *success* is the share of later works that carry it.
- A work's influence is the mean success of its mutations.

**Tests.**
- **S18.** The predicted innovators, The Dark Knight Returns (1986), Year One (1987) and The Killing Joke (1988), have higher influence than other works would have in the same chronological slots. The null is 2,000 position-preserving shuffles of which work occupies each slot. Prediction for Qwen neurons: p < 0.05.
- **S19.** The inequality of success (Gini) exceeds a neutral random-copying model. That model is Bentley-style, frequency-dependent copying without selection, fitted to each work's observed unit count and mutation count. Prediction: the observed Gini is above the 95th percentile of 500 simulations.

The keyword comparator runs the same analysis with words as units, where a new rare word counts as a mutation. It is reported with no prediction.

This is exploratory in spirit. The trio's DEV texts are known, and chance presences (about 3% per unit per work) add noise mutations.

### T5. Concept agreement without names (secondary)

**Data.**
- The 27 DEV Qwen concept memes.
- Their 81 lexically matched null units, 3 per concept, from the robustness review.
- Llama partners for all of these, found on DEV name-free sentences at freeze F1.

**Test.** For each unit, the held-out correlation with its DEV-chosen partner is computed on all 600 T2 story texts. The statistic is the mean over concepts of r(concept) − mean r(its nulls). The null swaps the concept label within each matched set, 20,000 times.

**Prediction (S20).** Above 0, with p < 0.05.

### T6. Cost and interpretability

**Cost** (section 9). Wall-clock time, peak memory, storage, and LLM calls and tokens, for every method and every set. From these come the cost per story and per meme, an accuracy-against-cost frontier for T1 and T2, and the break-even number of memes between one neuron pass and the cached LLM judge. There is no hypothesis. Every figure is reported.

**Interpretability** (annotation_protocol A3). This is a blind label test of 40 units of each kind: Qwen neurons, Llama neurons, LDA topics and embedding clusters.
- A labeller names each unit from its 8 top pool sentences.
- Two judges then pick, from two held-out sentences, the one the label describes.

The results are reported as accuracy with exact binomial intervals, as the share of units labelled "unclear", and against a word-overlap floor.

- **S21.** Qwen neuron accuracy exceeds the floor.

There is no directional prediction for neurons against LDA or clusters; those are Fisher tests.

## 5. Multiple comparisons

- **Primary.** H1 to H5 in Qwen3-8B, Holm-Bonferroni at a familywise α of 0.05, one-sided. Within H2, the superiority step is a fixed sequence after non-inferiority, at the same adjusted level. Llama replication is judged separately (section 1.4) and adds no claims.
- **Secondary.** Every test labelled S1 to S21, and every method contrast, gets Benjamini-Hochberg q-values within its family: T1, T2, T3, and T4 to T6 together. Each is called "secondary" wherever it is reported, and none is called confirmed.
- **Exploratory.** Anything not named in this document. It is reported as exploratory, with no p-value framed as a test.

## 6. Power

Power was simulated with the same sign-flip tests (`analysis.signflip`). The spread was set from DEV: the per-adaptation difference in share between neurons and TF-IDF on DEV E2 has an SD of 0.22.

| Hypothesis | Units | Assumed SD | Power at α = 0.05 (and 0.01) |
|---|---|---|---|
| H1, H2 | 24 memes | 0.15 (H2: 0.10) | H1: 0.94 (0.77) for a 0.10 AUC gap. H2: 0.77 (0.51) when the true difference is 0 |
| H3 | 18 memes | 0.15 | 0.80 (0.63) for a mean AUC of 0.60; 0.99 (0.95) for 0.65 |
| H4 | 30 clusters | 0.20 | 0.85 (0.59) for a 0.10 difference in degradation; 0.98 (0.96) for 0.15. With Batman alone (16 targets) it would be 0.61 (0.34), which is why the 83 loose Shakespeare films were added |
| H5 | 14 plays | 0.15 | 0.59 (0.29) for a 0.08 difference; 0.90 (0.63) for 0.12 |

H5 is the least powered test, and a null H5 will be reported as weak evidence. The number of loose adaptations is fixed by Wikipedia, and enlarging it after seeing the data is not allowed.

## 7. Manipulation checks, positive controls and inconclusive results

A test whose check fails is reported as **inconclusive**, not as supported or falsified, and its numbers are still reported.

- **DEV reproduction (all neuron results).** The compact-extractor E2 must be within 0.03 of 0.164 (Qwen) and 0.102 (Llama). If it is not, stop before TEST.
- **T1b (H4).** H4 is inconclusive if either of these holds:
  - at level 100, the mean realised removal of banned-type tokens is below 90%;
  - pooled TF-IDF recovery is not worse at level 100 than at level 0.
- **T1d (H5).** The positive control above.
- **T2 (H1 to H3).** The H1 to H3 conclusions are inconclusive if either of these holds:
  - the Claude upper bound's reworded AUC is below 0.80, meaning the planted memes are not reliably present;
  - the analyst keywords' explicit AUC is below 0.75, meaning the explicit wording did not use the examples' vocabulary.

  Separately, a meme that fails quality control in more than 4 of its 15 plots is dropped for every method.
- **T3.** S14.
- **Missing features.** These are reported as missing (`analysis.run_lineage` records them). They are never imputed.

## 8. What is reported whatever happens

- Every primary statistic in both models: the estimate, a 95% interval, p, the Holm-adjusted p, and the verdict (supported, falsified or inconclusive).
- Every method's result on every test and condition, including T1a on DEV, as tables and as recovery-against-drift curves.
- The cost table and the accuracy-against-cost frontiers.
- Every quality-check outcome: drift checks, translation checks, name-removal audits, story checks Q1 to Q5 with exclusions by meme, version and writer, and inter-annotator kappas.
- The label test.
- The commit hashes of F0, F1 and F2, and every entry in `DEVIATIONS.md`.

A result that goes against the neurons is reported with the same prominence as one that favours them.

## 9. Cost accounting

**What is measured.** Every model-side script writes `cost_<method>.json` per set, recording:

- seconds;
- peak memory;
- bytes stored;
- calls;
- tokens;
- for the cached judge, prefill seconds and seconds per meme.

**What is reported, per method:**

- seconds per work;
- seconds per extra meme (the marginal cost of another meme);
- storage per work;
- one-off setup cost per meme: unit selection, keyword choice (an LLM call), prototype embedding;
- agent effort for data creation, separately, since it is not a method cost.

**Break-even.** The number of memes M* at which one neuron pass per story (which scores every unit at once) costs the same as the cached LLM judge: M* = (neuron seconds per story − judge prefill per story) / judge seconds per meme.

**What timing on made-up text found.** Expected values are given here so that surprises can be seen:

| Method | Cost |
|---|---|
| Neuron pass | about 0.36 s per 24-sentence story (68 sentences per second), 1.8 MB per work in fp16 |
| LLM judge | 0.36 s to read the story, then about 0.2 s per meme with examples in the prompt |
| Recursive summary | about 6.5 s |
| e5-base | about 900 sentences per second, roughly 13 times cheaper than neurons |
| TF-IDF | milliseconds |

So neurons are expected to cost more than embeddings and keywords, and less than LLM reading at any number of memes above about 1. All timings are on the Apple M5 Pro (24 GB) laptop, with no other heavy job running. Conflicting jobs are logged.

## 10. Deviations policy

- Any change after F0 to code, data, a decision rule or a prediction is logged in `DEVIATIONS.md`, with:
  - the date;
  - what changed and why;
  - whether it was made before or after any TEST outcome was seen;
  - its effect.
- Bugs found in frozen code are fixed. Results are then reported from the fixed code, and from the frozen code too wherever the frozen code runs.
- No hypothesis, prediction, sample, threshold or primary method is changed after F2.
- A change between F0 and F2 is allowed only if it is decided without TEST outcomes, logged, and pushed before F2.
- If a step cannot be run as written (for example, an approval is withheld), the closest pre-specified fallback is used, if this document names one (PCA for UMAP). Otherwise the dependent result is reported as not run.

## 11. Budget

| Item | Per model | Notes |
|---|---|---|
| Sentences through the neuron extractor | about 123,000 | T3 63,000; T2 24,000; T1 22,000; DEV 13,400; T4 1,200 |
| Neuron extraction time | about 35 minutes | compact extractor |
| Neuron storage | about 25 GB | 8.8 GB aggregates, 11 GB per-sentence for the small DEV sets, the rest hidden states |
| LLM reading | about 13 hours (about 27 GPU hours for both models with MT) | T2 judge 50 min; T2 summaries 65 min; summary judge 45 min; T3 judge 100 min; T3 summaries 4.6 h; T1 summaries 75 min; lineage pairs (about 21,000 prompts) 1.8 h; the rest under 30 min |
| Embeddings | under 1 hour | all four encoders, all sets |
| Agent calls (Claude) | about 3,400 in total | Name removal 520; drift 930; translation 1,010; planted stories and blind reads 600; T2-N 280; label test 40; upper bound 80 |
| Disk | about 60 GB in total | 207 GB free |

## 12. Files in this folder

| File | What it is |
|---|---|
| `PREREGISTRATION.md` | this document |
| `analysis.py` | frozen analysis: every statistic, the cheap text methods, every test runner, the DEV freeze |
| `extract_compact.py` | compact neuron extraction: per-work aggregates, per-sentence values for a unit list, hidden states |
| `readers.py` | LLM reading (judge, recursive summaries, pairwise lineage, likeness, keyword choice, translation of keywords, MT) and sentence embeddings, with frozen prompts |
| `build_lists.py` | builds the work lists from the scout caches (`lists`) and assembles the TEST texts after F0 (`texts <stage>`) |
| `test_analysis.py` | runs everything on synthetic data with known answers (`--unit` for unit tests only) |
| `memes_for_tracing.json` | the 24 memes: definitions, ban lists, examples, DEV units, decoy hints |
| `planted_design.json` | 120 plots: genre, writer, 3 memes each |
| `works_shakespeare.json` | 40 plays and 177 eligible films, with documented sources, evidence, name flags, primary and loose flags, and text hashes |
| `culture_pool.json` | the 2,544 T3 works, with roles, strata, entities, book tiers and text hashes |
| `documented_relatives.json` | 36 documented Batman relatives, with relation, strength, evidence sentences and works |
| `works_batman_extended.json` | the 32 DEV and 43 TEST Batman works; the 7 fresh adaptations and their documented sources |
| `dev_units.json` | the DEV units for H3, the T5 concepts with their matched nulls, and the label-test candidates |
| `planted_story_protocol.md`, `drift_protocol.md`, `translation_protocol.md`, `name_removal_protocol.md`, `annotation_protocol.md` | the agent protocols, with fixed prompts and quality checks |
| `DEVIATIONS.md` | the deviation log, empty at F0 |

## 13. What this pre-registration does not claim

- It does not claim the neurons will win. DEV says words match them on Wikipedia-style text. The design asks where they win and at what price, and it reports every comparison either way.
- It does not test forecasting, and is not written for a journal.
- "First appearance" in T4 means first in this sample.
- The book family is tested as Mike's claim. It is not treated as ground truth: book tier labels are an input, and documented relatives are the ground truth.
- Claude wrote the planted stories, the drift rewrites, the translations and the ban lists. Any Claude-based method is therefore only an upper bound. The local models are the LLM rivals.
