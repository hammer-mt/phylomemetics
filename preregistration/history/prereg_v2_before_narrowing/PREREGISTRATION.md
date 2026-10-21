# Pre-registration: when, and at what cost, do neuron memes beat keywords, vectors, topics and LLM reading?

**Project:** phylomemetics (github.com/hammer-mt/phylomemetics), by Mike Taylor and Claude Opus 5.5.
**Version:** prereg-v2. Drafted 2026-10-20 (v1); revised 2026-10-21 after four adversarial AI design reviews (section 14). It is frozen by the commit that adds this folder (freeze F0), once that commit is pushed, after steps P1 and P2 below.
**Status:** no outcome has been computed on any TEST data. The scouting behind this design read only metadata (existence, plot-section length, documented "based on" relationships, which character names a plot section mentions) and timed the models on made-up text and three DEV works. The ban-list overlap of the draft keyword queries (section 14) was computed on DEV inputs only.

## 0. How to read this document

- This document fixes the questions, the data, the decisions, the predictions and the verdict rules.
- `analysis.py` fixes every computational detail: statistics, settings and seeds. `extract_compact.py`, `readers.py` and `bench.py` fix how the model-side features and costs are made, including every LLM prompt. `build_lists.py` fixes every work list and data-assembly step.
- The protocol files fix how agents (and the author) make the new data: `planted_story_protocol.md`, `drift_protocol.md`, `translation_protocol.md`, `name_removal_protocol.md`, `annotation_protocol.md`.
- The work lists fix which works are used, with a hash of each Wikipedia text: `works_shakespeare.json`, `culture_pool.json`, `documented_relatives.json`, `works_batman_extended.json`, `memes_for_tracing.json`, `planted_design.json`, `dev_units.json`. `sources/` holds the gzipped scout caches they are built from.

If this document and the code disagree about a computational detail, the code wins and the disagreement is logged as an erratum in `DEVIATIONS.md`. If they disagree about a hypothesis, a prediction or a decision rule, this document wins.

`test_analysis.py` runs every analysis end to end on synthetic data with known answers, and checks the type-I error of every primary statistic in worlds with no effect (200 replicates each). All 107 checks pass (about 4 minutes).

## 1. The question

### 1.1 What Mike wants the method for

The point of neuron memes is that nobody has to choose or count words. A neuron should capture a fuzzy idea and keep tracking it when the words change, the way a fringe group renames things to dodge keyword moderation. Mike wants the method to:

1. say what Batman actually is;
2. trace the shared history between Batmans;
3. see when Batman mutates and which mutations catch on (a measure of cultural selection);
4. find Batman-like things across culture, especially the memes Batman shares with the other "billionaire vigilante orphans" of Mike Taylor's earlier book.

For the paper he wants the experiment that shows how much more accurate, or how much cheaper, neurons are than vectors (embeddings) and keywords, including on new stories written with the right memes put in.

### 1.2 What DEV already showed

The robustness review of the first study found:

- **Heredity is real.** Name-free summaries of 9 Batman adaptations point to their documented source comic, in 99.6% of 30,240 analysis variants.
- **Words do as well.** Plain TF-IDF recovers the sources as well as the neurons (share 0.110 against Qwen 0.164; lower is better), and LSA does better (0.064).
- **"Beyond words" splits by model.** The pre-registered residual test fails for Qwen and passes only for Llama.
- **Chance was overstated.** Time-consistent chance is 42 to 44%, not 47%.

On name-free Wikipedia summaries the neurons have not yet earned their extra cost. This design asks where they could: wherever the words change but the idea stays.

### 1.3 Organising question

> When, and at what cost, do neuron-defined memes beat keywords, vectors, topics and LLM reading?

Every test runs the same methods on the same inputs, with the same supervision and the same tuning budget. The tests either change the words on purpose (vocabulary drift, translation, banned and coded wording), or use lineages where the words changed naturally (renamed Shakespeare adaptations). Every test reports accuracy and cost together.

### 1.4 The primary hypotheses

Each hypothesis is tested in Qwen3-8B (primary) and replicated in Llama 3.1 8B. All tests are one-sided. Each has a smallest effect size of interest (SESOI). Where a hypothesis must hold against several rivals, it is an intersection-union test: its p is the largest of the rival-wise p-values (this needs no further correction).

| | Test | Hypothesis | Statistic | SESOI |
|---|---|---|---|---|
| **H1** | T2 planted memes | On reworded stories (banned and implicit versions), neurons detect a meme better than an **evasion-aware keyword analyst** and better than a **lexical classifier**. | Mean over memes of twin AUC(neurons) minus twin AUC(rival), for `keywords:expanded-qwen3-8b` and `keywords:logreg` (intersection-union) | 0.05 AUC |
| **H2** | T2 planted memes | On reworded stories, neurons are not worse than the strongest vectors by more than 0.05 AUC (non-inferiority; superiority is then tested as a fixed sequence). | Same, against `emb:qwen3emb-4b`, `embwin:qwen3emb-4b` and `emb:e5-large` (intersection-union), with a 0.05 margin | margin 0.05; superiority 0.05 |
| **H3** | T2 planted memes | The 11 concept units found on DEV Batman text respond **specifically** to their own meme in unrelated genres. | Mean over the 11 units of [twin AUC on its own meme minus its mean twin AUC on every other meme]; null: meme labels permuted across the units | 0.05 AUC |
| **H4** | T1b vocabulary drift | Under heavy drift (levels 50 to 100), neurons recover documented sources better than TF-IDF and LSA, and still recover them above chance. | Mean over 21 clusters (8 Batman source-comic clusters, 13 Shakespeare plays) of [mean share at levels 50-100 for the rival minus the same for neurons] (intersection-union over TF-IDF and LSA); required conjunct: the neurons' own pooled lineage test p < 0.05 at levels 75 and 100 | 0.05 share |
| **H5** | T1d renamed Shakespeare adaptations | On a fresh lineage whose adaptations renamed the characters, neurons recover the source play better than TF-IDF and LSA. | Play-weighted mean over the 54 H5 films of share(rival) minus share(neurons), sign-flip over the 13 plays (intersection-union) | 0.05 share |
| **H6** | Cost and accuracy (decision rule) | On reworded planted stories, no method dominates the neurons: none is at least as accurate (its AUC's one-sided 95% lower bound at least the neurons' estimate) and no more expensive per story, at 34 memes. | `analysis.dominates` on the frontier of `analysis.run_cost` | none (a decision, not a test) |

"Twin AUC" compares each planted story with its own plot's control version: the same frame, word for word, with the slot filled by a different event of equal stakes (`analysis.detection_aucs`, scheme `twin`). "Share" is the E2 statistic (section 4, T1): the share of eligible non-source candidates that beat the documented source (lower is better). "Neurons" means each family's configuration selected on DEV (section 2). H1 and H2 must pass both a meme-level and a plot-level sign-flip test (section 5).

**Verdicts** (section 5): *supported* when the Holm-corrected test rejects (and any required condition holds); *falsified* when the one-sided falsification test excludes the SESOI in the predicted direction; *undetermined* otherwise; *inconclusive* when a section-7 check fails. A Llama result *replicates* when the same test gives p < 0.05 uncorrected.

### 1.5 How each outcome reads (fixed in advance)

| | If supported | If falsified |
|---|---|---|
| H1 | Neurons keep tracking a meme when its usual words are removed better than an analyst who expects evasion and better than a lexical classifier. This is the fuzziness claim. | Keywords chosen with evasion in mind, or a lexical classifier, do as well: the fuzziness claim does not separate neurons from good keyword practice. |
| H2 | Neurons are within 0.05 AUC of the strongest sentence, window and same-family embeddings (better, if the superiority step passes). | Embeddings are better by more than 0.05: the case for neurons rests on interpretability (T6) and cost (H6), not accuracy. |
| H3 | Units found on Batman respond to their own meme, not to drama in general, in genres Batman never visits. | They fire on any eventful passage: DEV meme labels do not transfer as meme detectors. |
| H4 | When a text's words are replaced, neurons hold on to its lineage better than word methods and still find it. | Neurons do no better than word methods once the words change. |
| H5 | Neurons find a source play through renamed adaptations better than word methods. | Word methods do as well. If no method recovers the sources, H5 is *uninformative*. |
| H6 | No cheaper method is as accurate. | A cheaper method (embeddings, a shared-sheet LLM judge, a small LLM) is as accurate: the neuron case rests on interpretability. |

A result in which LLM reading beats everything is a finding, and is reported as one. So is a result in which the neurons win only through the model's dense hidden states (the neurons-versus-hidden-states contrast says which).

## 2. Methods compared

**Same inputs.** For a named meme every method gets its label, its definition and the same example sentences: the **main set** (the definition plus 5 explicit and 5 reworded examples) for T2; the **far set** (the definition plus 5 far examples) for T2-N and T3. Every few-shot method contrasts the positives with the same **contrastive background**: the 2,267 DEV reference-film sentences, every other meme's example sentences, and the meme's own 5 hard negatives (its usual words used without the meme). The examples are written before F0 by fresh agent sessions that see only the label and definition (`annotation_protocol.md` A8).

**Same tuning budget.** Every method family has a pre-listed grid of at most 4 configurations, chosen on DEV only and frozen at F1:
- lineage families: the configuration with the lowest DEV E2 statistic (T1a); defaults are reported as a sensitivity analysis;
- T2 detector families: two variants each, chosen by leave-one-example-out on the main set (the held-out positive's percentile among the background).

**Same aggregation.** Sentence-level scores become story or work scores as the neurons' do: the mean of the 3 strongest sentences.

### 2.1 Neurons (primary: Qwen3-8B; replication: Llama 3.1 8B)

- **Models.** 4-bit MLX (`mlx-community/Qwen3-8B-4bit`, `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`). Each text runs as one continuous document.
- **Units.** A unit is one SwiGLU channel split by sign: 884,736 units in Qwen, 917,504 in Llama. A sentence's value is the maximum over its tokens (minus the minimum for a negative pole); a work's score is the mean of its 3 strongest sentences.
- **Presence.** A unit is present in a work when its score beats all but one of the 69 reference films (in the same text condition). T4 uses the strict bar (beats all 69).
- **Lineage grid** (`neurons:<model>@cfg`): `base` (E2's containment of optional units, band 2 to n/2), `idf` (the same, each unit weighted by log(n / its count), so generic units count less), `q` (band 2 to n/4), `cos` (cosine of reference-standardised unit scores).
- **Named memes** (`neurons:<model>`): `m1`, the single unit with the highest mean percentile of the positives within the background (ties: the margin over the 99th percentile), or `m5`, the mean of the 5 best units after background standardisation; LOEO chooses. "Single neurons are memes" is claimed only if m5 minus m1 has a one-sided upper bound below 0.03 AUC.
- **Secondary variants:** `:explicit` (selected on the definition and explicit examples only), `:far` (far set), `:m1ctx` (selected against in-context background: reference-film sentences inside their films, each example after a fixed 3-sentence DEV paragraph), `length_resid` (each unit's sentence values residualised on log word count, slopes from the background).
- **DEV-mapped units** (`devunit:<model>`, H3): 18 memes have a DEV-labelled Qwen unit (`dev_units.json`); 11 of them are concept units (the H3 set), 4 name detectors and 3 word detectors. Llama uses each unit's DEV twin; for revenge, Llama's own DEV-chosen revenge unit (chosen from 50 candidates on DEV; Qwen's failed its DEV test).

### 2.2 Keywords

- **Lineage:** `tfidf@cfg` (`base`: English stop words, min_df 2, sublinear tf; `ng12`: with bigrams; `df1`: min_df 1; `nosub`: raw tf), `words@cfg` (containment of optional words: `base`, `idf`, `q`). X-codes and digits are removed for every lexical method; Chinese uses character uni- and bigrams.
- **Keyword queries** (named memes). A sentence's score is the sum over terms (a one-word term counts its stem's occurrences; a multi-word term counts 1 when all its content stems occur); a story's score is the mean of its 3 best sentences plus 0.001 x total hits. Queries:
  - `keywords:expanded-qwen3-8b` (**H1 rival**): Qwen3-8B as an evasion-aware moderation analyst lists 30 to 40 mostly single-word terms (synonyms, euphemisms, slang, indirect indicators), sees each term's hit counts on the examples and on the DEV background, revises once (`readers.KEYWORDS_EXPANDED`, `KEYWORDS_REVISE`); every one-word term then gets up to 5 WordNet synonyms. `keywords:expanded-llama3.1-8b` likewise.
  - `keywords:qwen3-8b`, `keywords:llama3.1-8b`: the plain analyst (up to 10 terms), secondary.
  - `keywords:tfidf` (top 10 contrastive unigrams and bigrams) and `keywords:tfidf-wordnet` (the same with WordNet synonyms).
  - `keywords:ban-list-aligned`: the plain prompt given to Claude, who also wrote the ban lists; the query most likely to coincide with them.
  - Each query's share of terms that fall on the ban list is published at F1 (`dev_freeze.json`, `t2.ban_list_overlap`).
- **`keywords:logreg`** (**H1 rival**): an L2 logistic regression (C = 1, balanced classes) on TF-IDF word uni- and bigrams of the positives against the contrastive background.
- **The ban lists are never given to any method.**

### 2.3 Vectors

- **Encoders** (pinned revisions, `readers.ENCODERS`): `qwen3emb-4b` (Qwen/Qwen3-Embedding-4B, the strongest encoder that fits this laptop, fixed a priori as primary), `e5-large`, `e5-base` (multilingual), `labse`, `minilm`.
- **Units of text:** sentences (`emb`), 3-sentence windows centred on each sentence (`embwin`, so the encoder sees the slot's context as the neurons do), and whole works (`embdoc`).
- **Named memes:** two variants per encoder, chosen by LOEO: the contrastive prototype (mean positive minus mean background embedding, cosine) and an L2 probe (C = 1, balanced) on the same positives and background.
- **Lineage grid** (`emb:<enc>@cfg`): `mean` (cosine of mean sentence embeddings), `centred` (centred on the reference films), `align` (mean over the candidate's sentences of the best cosine to any row sentence), `presence` (the neuron rule applied to embedding dimensions: top-3 pooling, both poles, the reference-film bar, the band and containment).
- **`lsa@k`**: truncated SVD of TF-IDF, k in {100, 50, 200}.
- **`hidden:<model>@L`**: the same model's mean residual-stream state per sentence at one of four layers (quarter, half, three-quarter, last), reference-standardised; the layer is chosen on DEV E2; named memes by prototype or probe (LOEO). The neurons-versus-hidden contrast says whether a neuron win comes from the neuron basis or from the model's dense representation.

### 2.4 Topics

- **`lda`**: sklearn LDA (batch VB, default priors, 20 iterations, seed 0). Lineage K in {30, 20, 50} on works; T2 K in {50, 100} on sentences (LOEO); T3 K = 50 on works; T6 K = 100 on works.
- **`topics:<enc>`** (BERTopic-style): UMAP to 5 dimensions (15 neighbours, min_dist 0, cosine) then HDBSCAN (min_cluster_size in {15, 10, 30} for lineage, 15 elsewhere). Without umap-learn the pre-registered fallback is PCA to 10 dimensions (a reported deviation).

### 2.5 LLM reading

The readers are Qwen3-8B (thinking off) and Llama 3.1 8B Instruct (`readers.py`); every scored answer is a next-token probability, p_yes, with logit = log P(yes) - log P(no). Claude appears only as a labelled upper bound.

- **`judge:<model>`**: the story is cached once; one full question per meme (label, definition, main examples).
- **`judge-shared:<model>`**: the meme sheet (all 34 memes' labels, definitions and examples) is cached once for the whole set; each story is appended; a short question per meme. The cheapest reasonable LLM judge.
- **`judge-1pass:<model>`**: sheet, story and one answer sheet read in a single forward pass.
- **`judge:qwen3-1.7b`**: the small reader (Qwen3-1.7B, 4-bit MLX, pinned).
- **`summary-judge:<model>`**: recursive summarisation (8-sentence chunks to 60 words, merged to 150 words, in English), then the full-question judge on the summary.
- **`nli:deberta-v3-base`**: zero-shot NLI (MoritzLaurer/deberta-v3-base-zeroshot-v2.0, pinned), hypothesis "This text describes the following: LABEL. DEFINITION", per sentence.
- **Lineage:** `llmsum:<model>` (summary pairs, the recursive-summarisation arm) and `llmfull:<model>` (full-text pairs; the candidate's full text is the cached prefix and each row is appended; on Batman levels 0 and 100 and the T1d primary films). Similarity = logit.
- **T3:** `llm:<model>` likeness reads the full masked text against three fixed DEV Batman exemplar summaries (no name "Batman" in the prompt).
- **Recognition probe:** each LLM names the title or franchise of every name-free or masked text; T1d and T3 are also reported on the unrecognised works only.
- **`judge:claude-upper-bound`** (A4): labelled upper bound, never a check.

### 2.6 Floors

`length` (story: the mean word count of its 3 longest sentences; lineage: minus the absolute difference in log length) and `codes` (lineage: the X-code profile alone: number of distinct codes, codes per sentence, their positional profile). Both appear in every table.

## 3. DEV, TEST and the order of work

- **DEV**: the existing Batman corpus (`data/corpus.jsonl`, `data/corpus_neutral.jsonl`: 32 Batman works, 13 cousins, 69 reference films) and every existing result.
- **TEST**: all new data (drift rewrites, translations, Shakespeare texts, the 39 new Batman-lineage works and the v2 name-free texts of all 71, the culture pool and relatives, the planted stories, the natural-text annotations, the zh.wikipedia texts).

Nobody runs a method on TEST data, or looks at a TEST outcome, before that test's data freeze and the feature freeze F3. At every freeze point the named files are committed **and pushed**, and the commit hash is noted in `DEVIATIONS.md`.

**P0 (done).** The scout caches are in `sources/`; `build_lists.py lists` builds every work list from them.

**P1 (before F0, agents).** (a) Example sentences (A8): `build_lists.py texts examples`, then `build_lists.py memes`. (b) Relation typing (A6): `build_lists.py relations`.

**P2 (before F0, pilot; never TEST).** `build_lists.py texts pilot` writes 6 pilot plots (genres outside the design), 2 DEV reference films for drift at levels 50 and 100, and 3 DEV reference films for translation. The protocols run on them; pass rates and calls per item go to `prereg/agent/pilot/pilot_report.json`. Protocols may change only on pilot evidence, each change logged in section 14 before F0. No method is run on pilot material.

**F0.** `test_analysis.py` passes; commit and push this folder. From here the hypotheses, primary methods, SESOIs, negative sets and verdict rules are locked.

**Step 1, approvals** (Mike): installing sentence-transformers 6.1.0 (LaBSE only), umap-learn 0.5.12 (with numba and pynndescent) and nltk 3.10.3 with its WordNet data; downloading Qwen3-Embedding-4B, multilingual-e5-large, multilingual-e5-base, LaBSE, all-MiniLM-L6-v2, DeBERTa-v3-base-zeroshot-v2.0, Qwen3-1.7B-4bit and Gemma-2-9B-it-4bit (about 20 GB) at the revisions in `readers.py`. Nothing else is installed. A withheld approval triggers the named fallback (PCA for UMAP; no WordNet expansion; Mistral-7B for Gemma) or "not run".

**Step 2, DEV features** (both models `<m>`, every encoder `<e>`):

```
python prereg/build_lists.py texts dev
python prereg/extract_compact.py dev_namefree <m> --full --hidden
python prereg/extract_compact.py dev_original <m> --full --hidden
python prereg/extract_compact.py t2_background <m> --full --hidden --prefix
python prereg/extract_compact.py t2_examples <m> --full --hidden --prefix
python prereg/extract_compact.py t2_examples_ctx <m> --full
python prereg/extract_compact.py t3_dev_masked <m>
python prereg/readers.py embed --set S --encoder <e>            # S = dev_namefree, dev_original, t2_background, t2_examples, t3_dev_masked
python prereg/readers.py keywords --model <m> [--expanded]       # plain and evasion-aware, both models; A5 for Claude
python prereg/readers.py zh-keywords --source qwen3-8b; zh-keywords --source expanded-qwen3-8b
python prereg/readers.py summarise --set dev_namefree --model <m>; summarise --set t3_dev_masked --model <m>
python prereg/readers.py pairs --rows dev_namefree:screen --cands dev_namefree:comics --model <m>   # and pairsfull
```

**F1 (DEV freeze).** `python prereg/analysis.py dev-freeze` writes `dev_freeze.json` and `sentence_units_<model>.npy`: every detector, query, prototype and probe; the lineage selections (T1a); the DEV reproduction check (compact-extractor E2 within 0.03 of 0.164 and 0.102 with E2's tie rule, else stop); the codes-only baseline on DEV; the far-set screen; the in-context check; the T5 partners; the per-sentence unit list; timers for every step. Commit and push. Then `extract_compact.py t2_reference <m> --units ...` and `bench.py run` (DEV texts only); commit `results/bench.json` (F1b).

**Step 4, TEST data, in four staged freezes**, each committed and pushed before that stage's methods run:
- **F2-T2:** planted stories with reads, ratings and repairs (`texts stories`), long texts (`texts long`), Chinese banned and control versions (part of `texts zh`).
- **F2-T1b:** name removal (`texts wikipedia`, `texts names`), drift (`texts drift`).
- **F2-T1cd:** T1c translations (`texts zh`), zh.wikipedia fetch and Chinese name removal (`texts zhwiki`), code-collapsed copies (`texts codes`).
- **F2-T3:** T2-N annotation (`texts t2n`); the culture pool was assembled by `texts wikipedia`.

Every Wikipedia text is checked against its frozen hash, every protocol check runs, and exclusions are recorded before any method touches a stage.

**Step 5, TEST features** (for each stage after its freeze):

```
# neurons (both models)
for S in t1b_drift{25,50,75,100,_style} t1b_fresh_drift{25,50,75,100,_style} t1b_shk_drift{25,50,75,100,_style} \
         t1e_fresh_namefree t1c_zh t1c_fresh_zh t1c_zhwiki t1d_plays_namefree t1d_films_namefree t4_batman \
         t1d_plays_namefree_cc t1d_films_namefree_cc t1c_zh_cc t1c_fresh_zh_cc dev_namefree_cc; do
  extract_compact.py $S <m> --hidden; done                        # t1e_fresh_namefree also --full (reuse sensitivity)
for S in t2_stories t2_stories_zh t2n_works t2_long t3_pool; do extract_compact.py $S <m> --units prereg/sentence_units_<m>.npy --hidden; done
# embeddings: every set above, sentences; t2_stories, t2_stories_zh, t2_long also --unit win3 and --unit doc
# LLM reading (both models unless noted)
readers.py judge | judge-shared | judge-1pass --set t2_stories;  judge --set t2_stories --model qwen3-1.7b
readers.py summarise --set t2_stories; sjudge --set t2_stories;  nli --set t2_stories (and t2_long, t2_stories_zh)
readers.py judge --set t2_stories_zh;  judge --set t2n_works --examples far;  judge --set t2_long;  summarise/sjudge --set t2_long
readers.py judge --set t3_pool --tier A --examples far;  likeness --set t3_pool;  recognise --set t3_pool --model qwen3-8b
readers.py summarise --set S   # every T1 set and the comics
readers.py pairs --rows <row set>:<screen|primary|drift|all> --cands <dev_namefree:comics | t1d_plays_namefree>   # every T1 condition
readers.py pairsfull --rows dev_namefree:screen,t1e_fresh_namefree,t1b_drift100:screen,t1b_fresh_drift100 --cands dev_namefree:comics
readers.py pairsfull --rows t1d_films_namefree:primary --cands t1d_plays_namefree
readers.py recognise --set t1d_films_namefree (both models); mt --set t1c_zh and t1c_fresh_zh (Qwen)
```

The Claude upper bound follows A4. The cross-family check reads run during data creation (protocols).

**F3 (feature freeze).** `python prereg/analysis.py manifest` writes the sha256 of every feature file to `results/feature_manifest.json`; commit and push it. The test runners refuse to run without it, and every run's output hash is appended to `results/run_log.jsonl` (the first run is the confirmatory one).

**Step 6, analysis.** `analysis.py t1 t2 t3 t4 t5`; `t6-prepare`, the label test (A3), `t6`; `cost`; `primary`.

**Minimum viable completion.** A test whose data falls short is reported as "not run" with the reason: T2 needs at least 100 plots with all six versions valid and at least 28 memes not dropped; T1b all four levels for at least 90% of the 71 works; T1c translations of at least 90% of the 96 works; T1d name-free texts of all 40 plays and at least 50 of the 54 H5 films; T3 features for at least 95% of pool works; T6 at least 30 qualifying units per kind and arm.

## 4. Tests

### T1. Lineage recovery under wording change

**The statistic** (`analysis.lineage_test`). Each method gives a similarity between every row work (targets and hub works) and every candidate source. Each similarity is z-scored against the same candidate's similarities to the hub works (the SD floored at the 10th percentile of all column SDs). A target's share is the share of its eligible non-source candidates whose z beats its best documented source, **ties counting one half**. The statistic is the (weighted) mean share. **Null:** joint relabelling of the candidates, **restricted to time-consistent relabellings** (rejection sampling: every relabelled source is published no later than its target), 20,000 permutations; the hub rule is recomputed when it depends on the labels; p is one-sided. The null mean is the honest chance level (42 to 44% on DEV, not 47%).

**Head-to-head** (`analysis.advantage_test`): the per-target advantage share(rival) - share(neurons), averaged within clusters, sign-flip over clusters, with a cluster bootstrap interval and the falsification test.

**Beyond words** (`analysis.residual_sim`): neuron similarity regressed on TF-IDF cosine and word containment, lineage test on the residuals.

**Methods:** every lineage family of section 2 at its DEV-selected configuration; the lexical defaults as sensitivity.

#### T1a. DEV reference (not confirmatory)

Every configuration of every family on the DEV name-free E2 (9 adaptations, 12 comics, 11 hub works, time rule), computed at F1. It fixes each family's configuration and shows where each method starts.

#### T1b. Vocabulary drift (H4)

**Data.** Batman: the 20 DEV screen works and the 7 fresh adaptations (T1e) at levels 0, 25, 50, 75, 100 (`drift_protocol.md`; nested ban sets, every level rewritten from level 0) plus a style-only control; targets: the 9 DEV and 7 fresh adaptations; candidates: the 12 DEV comics; hub: the 11 DEV screen works that adapt no comic, drifted at the same level; time rule. Shakespeare: the 44 drift-sample films (the H5 films, at most 6 per play), drifted at every level; candidates: the 40 plays (never drifted); hub: the drift-sample films adapting other plays.

**Clusters.** Batman targets by their first-listed documented source comic (8 clusters); Shakespeare films by play (13). 21 clusters of equal weight. Sensitivity: Batman targets clustered by connected components of the target-source graph.

**H4 statistic, conjunct and checks** as in section 1.4: `analysis.h4_head_to_head` against TF-IDF and against LSA (intersection-union); the neurons' pooled lineage test at levels 75 and 100 (`analysis.pooled_lineage`, Batman and Shakespeare nulls combined draw by draw with weights 8 and 13) must give p < 0.05.

**Secondary.** The recovery curve of every method against nominal level and against realised content-type survival; the head-to-head against every other method; S1 (embeddings degrade less than TF-IDF: `emb:qwen3emb-4b` against TF-IDF at levels 50-100); S2 (LLM pairs, estimate only); headroom-free retention (`analysis.retention_contrast`: each method's above-chance recovery at levels 50-100 relative to level 0, with each target's own null mean); the draft's degradation statistic (descriptive only); Batman alone and Shakespeare alone; without drift-flagged works; the style-only control; the source-leak index (`drift_protocol.md`); the codes-only baseline.

#### T1c. Cross-language (secondary)

**Data.** The Batman design of T1b at level 0, with the 27 screen works and the 69 reference films agent-translated into Chinese (machine-translated Chinese); the 12 comics in English. **Language-matched bars:** Chinese rows use the translated reference films; the English comics use the English name-free ones; the optional band uses those presences.

**Predictions.** S3: Qwen neurons recover the sources (p < 0.05). S5: Qwen3-Embedding, e5-base and LaBSE do (p < 0.05). Construction check (not a prediction): TF-IDF, words, LSA and LDA do not (character n-grams cannot overlap English). Neurons against embeddings and LLM reading: two-sided. Also: machine translation plus words, with its cost; the natural arm on zh.wikipedia texts; the code-collapsed sensitivity; Llama is expected to be weaker.

#### T1d. A fresh lineage: Shakespeare (H5)

**Data** (`works_shakespeare.json`). Candidates: the 40 plays (commentary sentences removed by a fixed rule, at most 80 sentences). Rows: the 141 primary films (one documented play, not in DEV, no mention of Shakespeare or a non-eponymous title). The **H5 sample**: the 54 primary films that rename the play's characters (fewer than 2 of its distinctive principal names appear; "renamed" measures names, not setting or wording), whose source is documented by a lead sentence or Wikidata P144, and that are not remakes of DEV works: 13 plays, 3 of them with a single film. Names removed (`name_removal_protocol.md`). Reference films: the 69 DEV films without The Lion King. Play-weighted. No time rule.

**H5** as in section 1.4.

**Informativeness.** If no method (including LLM reading and the embeddings) recovers the H5 sources at p < 0.05, H5 is *uninformative*.

**Positive control.** On the original texts with names, TF-IDF recovers the sources of the films that keep names with a share of 0.10 or less; otherwise H5 is inconclusive.

**Secondary.** S6, beyond words, on the H5 sample (prediction: residual p < 0.05); every other method; all renamed films including category-only sources; without single-film plays; unrecognised films only (recognition probe); faithful and all primary films; the code-collapsed sensitivity (every X-code replaced by one token for every method).

#### T1e. Fresh Batman adaptations (secondary)

The 7 animated adaptations of DEV comics new to the project (TEST): Year One (2011), The Dark Knight Returns (2012), The Killing Joke (2016), Hush (2019), Death in the Family (2020), The Long Halloween (2021), Knightfall (2026). **S7:** every method recovers the sources (p < 0.05): a positive control for the fresh-data pipeline. An 8-gram reuse audit lists the sentences each target shares word for word with a comic, and T1e and level-0 Batman are rerun without them.

### T2. Named memes: planted stories (H1, H2, H3)

**Memes** (`memes_for_tracing.json`, 34): Tier A (10, the orphan-vigilante family), Tier B (10, Batman story memes), Tier C (4, general memes) and Tier N (10 **neutral motifs** drawn by seed from Wikipedia's motif and plot-device categories before any unit was examined: closed circle of suspects, poisoned garment, rash promise, locked-out lover, self-fulfilling prophecy, love triangle, late-running collision meeting, literal loophole, thrown shoe, exile; `sources/neutral_motif_walk.json`). Each has a definition, a ban list, a decoy hint and four example sets (A8).

**Stories** (`planted_design.json`, `planted_story_protocol.md`): 136 plots in 17 genres unrelated to Batman, by 4 writer agents; each plants 3 memes; each meme in 12 plots; 6 versions per plot with an identical frame (explicit, banned, implicit, coded, control with an equal-stakes event, decoy); 816 stories of 18 to 30 sentences; every fill within 15% of the explicit fill's length; checks Q1 to Q6 (symmetric exclusions; salience ratings).

**Detection** (`analysis.detection_aucs`). Primary scheme **twin**: each planted story against its own plot's control version (same frame; equal-stakes event; frame and salience matched). Per meme, 24 reworded pairs (banned and implicit) across 12 plots. Secondary schemes: **matched** (negatives: the same-condition versions of plots that do not plant the meme), **all** (the draft's rule), **hard** (own control and decoy), **vs_decoy**, and **decoy against control** (a construction check: keywords should rank decoys above controls). Per protocol (Q4 to Q5 exclusions and dropped memes) and intention-to-treat.

**H1, H2.** As in section 1.4, over all non-dropped memes. Each comparison must pass both the meme-level sign flip and the plot-level sign flip (`analysis.plot_flip_test`, which flips every plot's pairs together across its three memes); falsification likewise needs both. **Required consistency:** the mean difference on the 10 neutral memes has the predicted sign (H1: above 0; H2: above -0.05). Reported beside each: the two-way (meme by plot) bootstrap interval, per-writer estimates, leave-one-writer-out estimates, and the intention-to-treat result (a supported claim whose ITT test fails is labelled "per-protocol only").

**H3.** The specificity test (`analysis.specificity_matrix`, `specificity_test`) on the 11 concept units. Secondary: all 18 DEV units; concept minus name/word units; split by DEV rewording survivors; the draft's AUC-above-0.5 test.

**Secondary predictions** (BH within T2): S8, the direct judge beats neurons; S9, the summary judge scores below the direct judge; S10, every method's AUC is lower on implicit than on banned; S11 (construction check), keyword methods rank decoys above controls; S-coded, context-using methods (neurons, hidden states, the judge) beat sentence embeddings and the evasion-aware keyword query on coded versions; the m1-versus-m5 claim; neurons against hidden states (key contrast); every pairwise contrast; the best encoder chosen in hindsight (labelled as favouring the vectors); the length-residualised neurons; F1 at each method's presence bar; R-precision; per tier; exclusions by meme, version and writer; reader kappa; the human audit; salience balance.

**T2-zh (secondary).** The banned and control versions of every valid plot in Chinese (272 stories), English examples, translated queries. S12: Qwen neurons' banned twin AUC is above 0.5.

**T2-N (secondary).** 237 natural masked plot summaries annotated exhaustively (A2), far-set detectors, work-level AUC per meme (at least 10 positive and 10 negative works, annotator kappa at least 0.4). S13: Qwen neurons' mean AUC is above 0.5.

**T2-long (secondary).** 48 long texts of 12 stories each: per-meme AUC and seconds per text for one neuron pass, the direct judge and recursive summarisation. Prediction: recursive summarisation loses more AUC relative to the direct judge on long texts than on single stories. The accuracy and cost results of T1 and T2 apply to texts of up to about 40 sentences; T2-long is the only long-text evidence.

### T3. Batman-likeness across culture (secondary, with pre-stated predictions)

**Data** (`culture_pool.json`, 2,544 works, 62,879 sentences, texts through `analysis.t3_text`: relation statements removed, names masked, surviving franchise words masked). Category pool (21 Wikipedia category roots, vigilante-only works subsampled to 400), 1,000 random films, and the documented relatives outside the pool.

| Role | Works | Rule |
|---|---|---|
| pool (negatives) | 2,312 | superhero 631, swashbuckler 187, pulp crime 498, random 996 |
| relative | 57 | 27 T3-positive entities (strength A or B; `documented_relatives.json`) |
| book_family | 21 | Mike Taylor's earlier book tiers that are not documented relatives (tier 1: Zorro, Pimpernel, Iron Man; tier 2: Spider-Man, John Wick, Puss in Boots) |
| batman_lineage | 39 | positive control and T4 |
| contested | 115 | excluded from the negatives and the superhero centroid, scored and reported: (a) the pre-mask plot or lead names a Batman-family character (55); (b) a work of a T3-positive entity's franchise that is not a documented relative work (Superman, Robin Hood, Darkman, Watchmen, the Phantom, Dick Tracy, Iron Man and the Avengers, Sherlock Holmes, the Shadow, the Bat, Doc Savage; 57); (c) a hero named in a relative's evidence sentence (Doctor Syn, the Spider, the Green Hornet); and three relabelled lineage titles |

The Count of Monte Cristo works, named by Mike but not documented, stay in the pool and are reported descriptively. **Strata** come from each work's own short description and first lead sentence (genre rule); the category-crawl strata are a sensitivity analysis.

**Scores** (neurons, TF-IDF, LSA, LDA, every encoder): Batman-likeness (cosine to the centroid of the 32 DEV Batman works, centred on the reference films; neuron units reference-standardised); superhero-likeness (cosine to the centroid of the superhero-stratum negatives, **leave-one-out** for those negatives); Batman-specific likeness (Batman-likeness residualised on superhero-likeness, fitted on the negatives); the orphan-vigilante layer (mean within-pool percentile of the Tier A **far-set** detectors); LLM likeness (logit).

**Inference** (`analysis.entity_level_test`). Units are franchises: each positive entity and each title-stem cluster of at least 2 negatives (87 clusters). A unit's score is the mean within-stratum percentile of its works; the positive labels are permuted among all units, so near-duplicate sequels count once on either side. Per-entity scores with intervals, and a sign test, are reported. The draft's work-level test is secondary.

**Relation types** (A6): plot/role, iconography/name, parody, meta, transitive.

**Predictions.**
- **S14, positive control.** The 39 lineage works outrank the pool on Batman-likeness (AUC at least 0.9) for neurons and TF-IDF; if both fail, T3 is inconclusive.
- **S15 (confirmatory).** The franchise-fresh plot/role and parody relatives (entities with no DEV cousin: Doc Savage, Dick Tracy, the Phantom, Darkman, the Watchmen, Bluntman, Rat Pfink, Commander Courage, Blackwolf, Blue Falcon, the Jerry Warren Batwoman and others, as typed) score above the pool on Batman-specific likeness (Qwen neurons, p < 0.05). The DEV-franchise relatives (S15b) are reported separately, beside the known DEV values: on DEV the Qwen canon share put Zorro above 4% of Batman works, the Bat 22%, the Shadow 32%, Superman 71%, Spider-Man 88%, Daredevil 85%, Iron Man 68%, the Green Hornet 19%. Iconography/name, meta and transitive relatives: predicted null.
- **Directional contrasts.** On S15's set, entity-level sign flips of neurons against TF-IDF, LSA and Qwen3-Embedding (one-sided: neurons above).
- **S16, S17.** Book tiers 1 and 2 score above the pool on the orphan-vigilante layer (Qwen neurons). Each rests on 3 named characters; they are claims about those characters, reported with per-entity intervals.

Also reported: the unrecognised-only sensitivity, contested works, the Monte Cristo works, the 50 most Batman-specific pool works per method.

### T4. Mutation and selection (secondary)

**Data.** All 71 Batman works (32 DEV, 39 TEST; no TEST work is earlier than 1995) re-run through the v2 name-removal protocol in one pass, in date order. **"First appearance" means first in this sample.** Presence uses the strict bar (beats all 69 reference films; the usual bar is a sensitivity).

**Definitions.** A mutation of work w is a unit present in w and in no earlier work. Its success is the number of later works carrying it, **leaving out the documented adaptations of w**. A work's influence is the mean success share of its mutations.

**Tests.** S18: The Dark Knight Returns, Year One and The Killing Joke have higher influence than the works in their chronological slots would under 2,000 position-preserving shuffles (the adaptation exclusion travels with the work), with and without the 7 fresh adaptations. S19 (comparative): the neurons' standardised excess of mutation-success inequality (Gini) over a null that keeps every unit's frequency and every work's unit count (curveball randomisation with the time order shuffled) is larger than the words' excess. The Bentley neutral-copying model is secondary. **Batch-effect check:** a classifier separating DEV-origin from TEST-origin works dated 1995-2022 on unit presence; an AUC above 0.7 flags T4.

### T5. Concept agreement without names (secondary)

The 27 DEV Qwen concept memes and their 81 lexically matched null units, with Llama partners searched on DEV name-free sentences at F1. **S20:** on the sentences of the 1,000 random films of the T3 pool (primary held-out corpus) the mean of r(concept) minus the mean r of its nulls is above 0 (null: the concept label swapped within each matched set, 20,000 times; p < 0.05). The T2 stories are secondary; per-concept variance ratios are reported.

### T6. Cost and interpretability

**Cost** (section 9). Every method's recipe, total-cost curves and the frontier; H6.

**Interpretability** (A3): the blind label test in two arms (random; best-of), 40 units per kind and arm (Qwen neurons, Llama neurons, LDA topics, embedding clusters), all present in 1-20% of pool works. **S21:** Qwen neuron accuracy exceeds the word-overlap floor on the same items (exact McNemar). Kinds are compared by Fisher tests without a directional prediction.

**Effort, stated narrowly.** In this design the only human or agent effort the neurons save over keywords is the keyword-choice call. Making neuron memes nameable costs labelling calls; T6 reports calls per usable label and extrapolates to the number of units needed to cover the T3 pool. The calls behind the DEV labels used for H3 are counted from the earlier run's logs at F1.

## 5. Multiple comparisons and verdicts

- **Primary.** H1 to H5 in Qwen3-8B, Holm-Bonferroni at a familywise alpha of 0.05, one-sided. A hypothesis that is inconclusive under section 7 enters Holm with p = 1. H2's superiority step is a fixed sequence after non-inferiority, at H2's Holm level. H6 is a decision rule outside Holm.
- **Verdicts.** Supported: the Holm test rejects and the required condition holds (H1, H2: neutral-subset consistency; H4: the conjunct). Falsified: the falsification test (`analysis.falsification_test`: H0 effect at least the SESOI, one-sided, alpha 0.05; for intersection-union hypotheses, against any rival; for H1 and H2 both the meme-level and plot-level versions) rejects. Undetermined: neither. Inconclusive: a section-7 check failed. H5 can also be uninformative.
- **Llama replication** is judged separately (p < 0.05 uncorrected, same direction, same required conditions) and adds no claims.
- **Secondary.** Every S-test and method contrast gets Benjamini-Hochberg q-values within its family (T1, T2, T3, and T4 to T6 together) and is called secondary wherever it is reported.
- **Exploratory.** Anything not named here.

## 6. Power

Simulated with the same tests (`history/power_sim_v2.py`, `history/power_v2.json`). Holm's first threshold is 0.01 and its last 0.05.

| Hypothesis | Units | Assumption | Power at alpha 0.05 (0.01) |
|---|---|---|---|
| H1, H2 | 34 memes, 136 plots (meme and plot tests) | meme heterogeneity SD 0.12, plot effects | 0.52 (0.25) for a true gap of 0.05 AUC; 0.70 (0.39) for 0.07; 0.85 (0.56) for 0.085 |
| H3 | 11 units | specificity SD 0.15 | 0.24 (0.07) for 0.05; 0.65 (0.34) for 0.10; 0.92 (0.68) for 0.15 |
| H4 | 21 clusters, two rivals (r = 0.7) | per-cluster SD 0.25 (0.20; 0.15) | 0.15 (0.03) for 0.05; 0.44 (0.15) for 0.10; 0.77 (0.45) for 0.15. At SD 0.20: 0.61 (0.29) for 0.10. At SD 0.15: 0.86 (0.60) for 0.10 |
| H5 | 13 plays, two rivals | per-film SD 0.22, between-play SD 0.10, so per-play SD 0.24 for the 3 single-film plays and 0.12 for Romeo and Juliet's 14 | 0.17 (0.03) for 0.05; 0.33 (0.13) for 0.08; 0.69 (0.34) for 0.12; 0.88 (0.57) for 0.15 |

H4 and H5 are underpowered for effects near their SESOI; with the three-way verdict a small true effect gives "undetermined", not "falsified". The numbers of clusters are fixed by the documented lineages and cannot be enlarged after seeing data.

## 7. Manipulation checks, positive controls and inconclusive results

A test whose check fails is reported as **inconclusive** (and enters Holm with p = 1); its numbers are still reported.

- **DEV reproduction (every neuron result).** The compact-extractor E2 (with E2's strict tie rule) within 0.03 of 0.164 (Qwen) and 0.102 (Llama); otherwise stop before TEST.
- **H4.** Mean realised removal of banned-type tokens at level 100 below 90%, or pooled TF-IDF recovery not worse at level 100 than at level 0. If the source-leak index recovers sources, the verdict uses the analysis without the leaking targets and says so.
- **H5.** The positive control; informativeness.
- **H1 to H3.** The human audit's pass rule (A7: at least 85% presence agreement on banned, implicit and coded cells, 95% absence on controls and decoys), or without an audit the cross-family reader's (75% and 90%); and the plain analyst's explicit twin AUC of at least 0.75 (the explicit wording used the usual vocabulary). A meme failing Q4 in more than 3 of its 12 plots is dropped for every method.
- **T3.** S14.
- **T4.** The batch-effect check flags the results.
- **Missing features** are reported as missing, never imputed.

## 8. What is reported whatever happens

- Every primary statistic in both models: estimate, 95% interval, one-sided bound, p, Holm-adjusted p, the falsification test and the verdict.
- Every method on every test and condition (including T1a on DEV), as tables and as recovery curves against nominal and realised drift.
- The cost recipes, the total-cost curves, the storage policies and the frontier.
- Every quality-check outcome: drift checks, translation checks, name-removal audits, story checks Q1 to Q6 with exclusions by meme, version and writer, reader and annotator kappas, cross-family agreement, the human audit, the pilot report.
- The label test.
- The commit hashes of every freeze point and every entry in `DEVIATIONS.md`.

A result against the neurons is reported with the same prominence as one in their favour.

## 9. Cost accounting

**Measured.** Every model-side script writes `cost_<method>.json`: work seconds, model load seconds (separately), MLX peak memory and process max RSS, bytes stored, calls, tokens, and load averages at start and end. Extraction records wall-clock time and load average per batch, so throttling and competing jobs show. `bench.py` measures every method's unit costs on 30 DEV texts after a warm-up, with 3 repeats, nothing else running, under `/usr/bin/time -l`, with the running processes logged; a neurons-only pass is timed separately from hidden-state capture. Every DEV-freeze step is timed.

**Recipes** (`bench.recipes`, `analysis.total_cost`). For each method: one-off setup (neurons: full activations of the background and examples; embeddings: none; LLM judge: none; shared-sheet judge: the sheet prefill), setup per meme (unit selection, probe fit, keyword call), seconds per story (one pass), seconds per story and meme (judge questions, NLI pairs), seconds per story for a meme added later, and storage per story under three retention policies (all units' aggregates, about 1.8 MB per work for neurons; selected units only; none). Translation of the reference films is charged to the methods that need Chinese presence bars.

**Total cost** C(N stories, M memes, Q later memes) for every method; reported at N = 100, 1,000 and 100,000, M = 1, 10 and 34, Q = 0 and 10.

**What is plain in advance.** One neuron pass is the same forward pass as an LLM judge's prefill; the neurons' cost advantage over LLM reading is only the per-meme questions. Embeddings and keywords also score every meme from one pass. With every unit's aggregate stored, a single-unit meme added later costs one unit selection and a lookup; an embedding probe likewise needs only a fit on stored vectors.

**Frontier and H6.** On reworded planted stories: each method's mean twin AUC, its one-sided 95% lower bound, and its cost per story at M = 34 (setup amortised over the T2 stories). A dominates B when A's lower bound is at least B's estimate and A costs no more.

## 10. Deviations policy

- From F0, the hypotheses, primary methods, SESOIs, negative sets and verdict rules are locked. Between F0 and the last F2 only logged bug fixes and feasibility changes are allowed; after any F2 nothing about that test changes.
- Every change after F0 is logged in `DEVIATIONS.md` with the date, the freeze point it follows, what changed and why, whether any TEST outcome had been seen, and its effect.
- Bugs found in frozen code are fixed; results are reported from the fixed code, and from the frozen code too wherever it runs.
- If a step cannot run as written, the named fallback is used, or the result is reported as not run.
- The F3 manifest and the run log make it checkable that `analysis.py` ran once on frozen features.

## 11. Budget

Rebuilt from DEV measurements: the compact extractor ran 90 DEV sentences (2,342 tokens) in 2.36 s with hidden states, about 38 sentences or 1,000 tokens per second. It is about 7 times faster than `extract.py` (3,700 sentences per 12 minutes), which wrote int8 copies of every unit for every sentence to disk; the compact extractor reduces on the GPU and writes only aggregates. e5-base measured 443 sentences per second at 64 tokens.

| Item | Estimate |
|---|---|
| Neuron extraction | about 164,000 sentences per model (DEV 17,000, TEST 147,000), about 1.2 h per model at the measured rate; allow 2 h for throttling |
| Neuron storage | full activations for 11,300 DEV sentences (20 GB per model), aggregates for about 5,600 works (10 GB per model), hidden states and unit lists (about 5 GB per model): about 70 GB |
| Embeddings | 5 encoders, about 230,000 texts for the primary encoder (sentences, windows, documents); about 2 h in all (bench.py measures Qwen3-Embedding) |
| LLM reading | about 15 h for Qwen3-8B (T2 judges 3 h, summaries and summary judge 3 h, T3 judge, likeness and recognition 3.5 h, T1 summaries 1.3 h, summary pairs 2 h, full-text pairs 0.7 h, MT and the rest 1.5 h) and 13 h for Llama; 1 h for Qwen3-1.7B; NLI 40 min |
| Check reader (Gemma 2 9B) | about 6 h (story reads, A2, drift retention, back-translation) |
| Agent calls (Claude) | about 7,300 fresh sessions: examples 170, relation typing 80, pilot 100, name removal 720, drift 850, translation 850, planted stories 2,550 (writers 190, reads 1,950, salience 410), T2-N 600, label test 1,310, upper bound 110, keyword query 34; about 30 million tokens; about 15 to 20 hours at 8 parallel sessions |
| Author | about 2 to 3 hours for the human audit (A7) |
| Disk | about 90 GB including about 20 GB of downloaded models; 208 GB free |

## 12. Files in this folder

| File | What it is |
|---|---|
| `PREREGISTRATION.md` | this document |
| `analysis.py` | frozen analysis: every statistic, the cheap methods, the detectors fitted at F1, every test runner, the cost model, the verdicts |
| `extract_compact.py` | compact neuron extraction (aggregates for every unit, per-sentence values for a unit list, hidden states) |
| `readers.py` | LLM judges, NLI, summaries, lineage reading, likeness, recognition, keyword analysts, translation, embeddings, the cross-family check reader |
| `bench.py` | the controlled cost benchmark |
| `build_lists.py` | the work lists from `sources/` and every data-assembly stage |
| `test_analysis.py` | unit tests, null worlds and the synthetic pipeline |
| `memes_for_tracing.json` | the 34 memes |
| `planted_design.json` | 136 plots: genre, writer, 3 memes |
| `works_shakespeare.json` | 40 plays and 177 films with evidence tiers, renamed, H5 and drift-sample flags, hashes |
| `culture_pool.json` | the 2,544 T3 works with roles (and contested reasons), strata, entities, clusters, book tiers, T2-N flags, hashes |
| `documented_relatives.json` | 36 documented relatives with relation, strength, evidence, relation type, DEV-franchise flag |
| `works_batman_extended.json` | the 32 DEV and 39 TEST Batman works, the relabelled titles and the 7 fresh adaptations |
| `dev_units.json` | DEV units for H3, the T5 concepts and nulls, the T6 unit lists |
| `sources/` | gzipped scout caches (Wikipedia/Wikidata, 2026-10-20), the robustness-review files dev_units needs, the neutral-motif draw; `sources/LICENSE.md` |
| `history/` | the v1 drafts of this document, the memes and the design; the power simulation |
| protocol files | `planted_story_protocol.md`, `drift_protocol.md`, `translation_protocol.md`, `name_removal_protocol.md`, `annotation_protocol.md` |
| `DEVIATIONS.md` | the deviation log and freeze hashes |

## 13. What this pre-registration does not claim

- It does not claim the neurons will win. DEV says words match them on Wikipedia-style text.
- It does not test forecasting, and is not written for a journal.
- "First appearance" in T4 means first in this sample, which has no TEST work before 1995.
- The book family is Mike's claim, tested; it is not ground truth. Documented relatives are the ground truth.
- The planted benchmark measures memes as Claude writes them. Claude wrote the planted stories, drift rewrites, translations, examples and ban lists; Claude-based methods are upper bounds; the local models are the LLM rivals.
- The Chinese T1c texts are machine-translated; the natural arm is small.
- Accuracy and cost results apply to plot summaries of up to about 40 sentences (T2-long aside).

## 14. Design history

### 14.1 v1 (2026-10-20)

The first draft (`history/PREREGISTRATION_draft_v1.md`) had five primary hypotheses: neurons beat Qwen's keyword query on reworded memes (H1), are non-inferior to e5-base (H2), DEV units fire on planted memes (AUC above 0.5, H3), TF-IDF degrades more than neurons under drift (H4), and neurons beat TF-IDF on loose Shakespeare adaptations (H5). Its 62 synthetic checks passed.

### 14.2 AI design review, round 1 (2026-10-21)

Four adversarial reviews (Claude agents given the draft, the code and the robustness verdict) each returned "needs changes", with 8 blockers and about 50 major or minor problems in all. The reviewers ran their own simulations. The main findings, and what changed:

| Finding | Change in v2 |
|---|---|
| **H4 rewarded starting badly.** Degradation (drift minus level 0) is bounded by chance, so a method that starts worse has less room to fall; DEV showed neurons start worse (0.164 vs 0.110). In the reviewers' simulation an equally robust neuron method passed in 14-43% of runs, and a chance-level one in 100%. | H4 is now a head-to-head at levels 50-100 against TF-IDF **and** LSA, with the neurons' own lineage test at 75 and 100 as a required conjunct. The null worlds in `test_analysis.py` show rejection 0.00 for equal robustness with the DEV gap (the draft statistic: 0.19) and 0.055 for total loss (draft: 0.47). Retention relative to each method's own chance is secondary; the draft statistic is descriptive. |
| **T2 confounded meme presence with eventfulness.** Controls were "a neutral event", and negatives mixed other plots' planted versions; a detector of "something dramatic happens" scored AUC 0.56-0.72 for every meme. | Controls are now an event of equal stakes and dramatic weight, checked by blind salience ratings (Q6). The primary comparison is within plot (twin AUC: same frame, slot changed); a non-specific detector scores 0.50 in the null world (the draft's rule: 0.59). H3 became a specificity test with a mismatched-meme permutation null. |
| **H1 was close to true by construction.** 203 of 205 draft Qwen terms were multi-word and the queries overlapped the ban lists by 0.90 on average; only 23 terms could ever fire in a banned story. | H1's rivals are an evasion-aware analyst (30-40 mostly single-word terms, one revision after seeing hit counts, WordNet expansion) and a lexical classifier, both frozen at F1 with their ban-list overlap published. Decoys left the primary negatives. |
| **Frame context could leak** (the frame of a plot planting "heir" carries wealth). | The within-plot twin comparison holds the frame fixed. |
| **Supervision and tuning were unequal**: the neurons searched 0.9M units against a contrastive background with a pipeline refined on DEV, while rivals got plain prototypes and defaults. | Every family gets the same inputs (definition, explicit and reworded examples), the same contrastive background (with hard negatives), two variants chosen by leave-one-example-out, and a lineage grid of at most 4 configurations chosen on DEV E2. Embeddings get contrastive prototypes and probes, 3-sentence windows and documents, centred, alignment and neuron-rule lineage variants. |
| **The rivals were not the strongest.** H4/H5 against TF-IDF only though LSA was better on DEV; H2 against e5-base only. | Intersection-union against TF-IDF and LSA (H4, H5) and against Qwen3-Embedding-4B (sentences and windows) and e5-large (H2). |
| **The meme set and examples were chosen with the neurons in view**; examples paraphrased Batman scenes. | Ten neutral motifs drawn by seed from Wikipedia's motif categories; every example set rewritten before F0 by fresh agents that see only label and definition; a far set screened against DEV Batman sentences for T2-N and T3; a required neutral-subset consistency check. |
| **"Falsified" on any non-significant result**, with low power. | Smallest effects of interest, three-way verdicts, realistic power (section 6). |
| **The lineage null ignored the time rule** (chance 47% instead of 42-44%). | Time-consistent relabelling by rejection sampling; a unit test checks the null mean against exact enumeration. Ties count one half; LLM pairs use the logit; hub SDs are floored. |
| **T3 negatives contained Batman stories and relatives' franchises; works of one franchise were treated as independent; the superhero centroid included the scored works.** | A rule-based contested role (115 works); entity-level inference with franchise clusters (null-world rejection 0.045; the draft's work-level test 0.22); leave-one-out centroids; strata from each work's own genre; relation typing with plot/role relations primary and franchise-fresh entities confirmatory; relation statements stripped and a franchise scan after masking. |
| **The code did not implement several promised analyses and never checked type-I error.** | Implemented: the meme-drop rule, writer and tier results, best-encoder sensitivity, the flagged-works sensitivity, S1 and S9, directional S8, tests for S10-S14 and S21, kappas, intervals; null worlds for H1-H5, a non-specific detector, franchise-clustered T3 and the time-consistent null. The null worlds exposed one more dependence: memes sharing a plot are not independent for a non-specific detector (rejection 0.105), so H1 and H2 now also require a plot-level sign flip (rejection 0.03). |
| **Forking paths stayed open after F0; TEST features were never frozen.** | Locks at F0, staged data freezes, the F3 feature manifest and the run log. |
| **The cost comparison was circular and incomplete.** | A frontier dominance rule (H6); shared-sheet, one-pass and small-model judges and an NLI baseline; recipes with one-off and per-meme setup and storage policies; total-cost curves; `bench.py`; resume-safe extraction. |
| **T4 counted adaptations of the innovators as mutations catching on; the neutral model ignored unit frequencies.** | Success excludes documented adaptations; S19 is comparative against a margin-preserving null; strict presence bar; all 71 works re-run through one name-removal pass; a batch-effect check; four non-DC titles relabelled. |
| **T5's held-out corpus was enriched for the concepts.** | The 1,000 random films are the primary held-out corpus. |
| **Validation was a Claude monoculture.** | A cross-family check reader (Gemma 2 9B) for reads, annotation, drift retention and back-translation; a human audit by the author with pass thresholds. |
| **Length, X-codes, digits, reuse and leakage could carry signal.** | Length matching (Q1b, D1b), a length floor and a length-residualised sensitivity; a codes-only baseline and a code-collapsed sensitivity; digits banned in drift and removed for lexical methods; an 8-gram reuse audit; a source-leak index; a style-only drift control; recognition probes. |
| **Shakespeare ground truth was weaker than assumed; plays were truncated.** | H5 uses lead-sentence or P144 sources and drops remakes of DEV works (54 films, 13 plays); plays keep up to 80 sentences after a commentary filter; the sample is called "renamed", not "loose"; drift effort capped at 6 films per play. |
| **Agent budget undercounted; no pilot; one freeze blocked everything; sources only in the scratchpad.** | Budget recounted (about 7,300 sessions); a pilot before F0; minimum viable completion per test; staged freezes; the scout caches copied into `sources/`. |
| **Coded wording and long texts were missing** (fringe-group renaming; recursive summarisation's real use). | A coded version of every plot; T2-long. |

### 14.3 Suggestions not adopted, and why

- **Listwise LLM lineage reading over all candidates in one context** (review 2): 40 plays of up to 80 sentences exceed what fits in memory and in Qwen3's native context; full-text **pairwise** reading with the candidate cached replaces it, in one order only (candidate first) to keep the cache.
- **Qwen3-Embedding-8B** (review 2): about 16 GB in half precision does not fit beside the system on this 24 GB laptop; the 4B model, the strongest that fits, is fixed a priori instead.
- **Drifting the 69 reference films** (review 3): the presence bars stand for a fixed natural reference corpus, as an analyst would use; the style-only control measures what rewriting itself does.
- **Wikidata P674 (characters) for the contested rule** (review 3): coverage is patchy for these films; the plot and lead text are searched instead.
- **A blind setting-change judgement for "loose" films** (review 3): the sample is renamed "renamed adaptations", which is what the flag measures, rather than adding another agent judgement.
- **A band relative to sources (hi = 2 x median works per source + 1)** and **neurons-cos co-primary with Holm inside H5** (reviews 1 and 2): replaced by the DEV-selected lineage grid, which includes IDF-weighted containment, a quarter band and the cosine rule, so no alpha is split.
- **Embeddings inside H5's intersection-union** (review 2): with 13 plays H5's power is already low; neurons against the primary embedding is a key secondary contrast in T1d and T1b.
- **An "update arm" re-selecting on banned examples from held-out TEST plots** (review 4): partly adopted, before any TEST data exists, through the reworded example set that every method gets; a cross-fitted TEST update arm would double T2 and use TEST labels for selection.
- **Exhaustively annotated public-domain novels for the long-text condition** (review 2): concatenations of planted stories give ground truth by construction at no agent cost.
- **GPT or Gemini as the second reader family** (review 3): no API access is set up, and stories would leave the machine; a local Gemma 2 9B is used.
- **Holding the T4 design to forecasting standards**: not wanted (Mike does not need forecasting).
