# Pre-registration: when, and at what cost, do neuron memes beat keywords, vectors, topics and LLM reading? Phase 2: stories with planted memes

**Project:** phylomemetics (github.com/hammer-mt/phylomemetics), by Mike Taylor and Claude Opus 5.5.
**Version:** prereg-v3. Drafted 2026-10-20 (v1); revised 2026-10-21 after four adversarial AI design reviews (v2, section 14.2); narrowed the same day by Mike to the planted-meme stories experiment (v3, section 14.4) and revised after a fifth AI review of the narrowed design (section 14.5). It is frozen by the commit that adds this folder (freeze F0), once that commit is pushed, after steps P1 and P2 below.
**Status:** no outcome has been computed on any TEST data. The scouting behind this design read only metadata (existence, plot-section length, documented "based on" relationships, which character names a plot section mentions) and timed the models on made-up text and three DEV works. The ban-list overlap of the draft keyword queries (section 14) was computed on DEV inputs only. No planted story has been written yet.

## 0. How to read this document

- This document fixes the questions, the data, the decisions, the predictions and the verdict rules.
- `analysis.py` fixes every computational detail: statistics, settings and seeds. `extract_compact.py`, `readers.py` and `bench.py` fix how the model-side features and costs are made, including every LLM prompt. `build_lists.py` fixes every work list and data-assembly step. `audit_page.py` is the local page for the author's audit.
- Two protocol files fix how agents (and the author) make the new data: `planted_story_protocol.md` and `annotation_protocol.md` (steps A1, A3, A4, A5, A7, A8 and A9).
- Three work lists fix the inputs: `memes_for_tracing.json` (the 34 memes), `planted_design.json` (136 plots) and `dev_units.json` (the DEV units behind H3 and the label test).

**Deferred material.** The tests deferred to a later phase (section 15) keep their code, protocols and work lists in this folder, marked as deferred, so that phase can start from them: `drift_protocol.md`, `translation_protocol.md`, `name_removal_protocol.md`, steps A2 and A6 of `annotation_protocol.md`, `works_shakespeare.json`, `culture_pool.json`, `documented_relatives.json`, `works_batman_extended.json`, the `sources/` caches they are built from, and the functions marked DEFERRED in `analysis.py` and `build_lists.py`. Committing them at F0 records them; it does not pre-register them. Nothing in them is run in this phase, and `analysis.py` refuses to run the deferred tests without an explicit `--deferred` flag.

If this document and the code disagree about a computational detail, the code wins and the disagreement is logged as an erratum in `DEVIATIONS.md`. If they disagree about a hypothesis, a prediction or a decision rule, this document wins.

`test_analysis.py` runs every analysis end to end on synthetic data with known answers, and checks the type-I error of every primary statistic, and of the overlap test, in worlds with no effect (200 replicates each). It also runs the audit page on the synthetic stories, and checks the audit's counting rules, the gates, the tie-breaking of the operating points, a frame-leak world for the overlap analysis, the H6 rival set, the size of every test family and the F3 hash check. All 151 checks pass (about 5 minutes); 30 of them, labelled "(deferred)", keep the deferred code working and are not part of this pre-registration.

## 1. The question

### 1.1 What Mike wants the method for

The point of neuron memes is that nobody has to choose or count words. A neuron should capture a fuzzy idea and keep tracking it when the words change, the way a fringe group renames things to dodge keyword moderation. Mike wants the method to:

1. say what Batman actually is;
2. trace the shared history between Batmans;
3. see when Batman mutates and which mutations catch on (a measure of cultural selection);
4. find Batman-like things across culture, especially the memes Batman shares with the other "billionaire vigilante orphans" (Zorro, the Scarlet Pimpernel and Iron Man).

For the paper he wants the experiment that shows how much more accurate, or how much cheaper, neurons are than vectors (embeddings) and keywords, including on new stories written with the right memes put in.

This phase runs that experiment. It tests the claim underneath all four aims, that a neuron keeps tracking an idea when its words change, on material whose ground truth is known by construction: new stories in genres Batman never visits, each written in versions that plant the same memes with their usual words, without them, without ever naming them, and with invented code words. Aims 2 to 4 need lineages and a culture-wide pool; those tests are deferred (section 15).

### 1.2 What DEV already showed

The robustness review of the first study found:

- **Heredity is real.** Name-free summaries of 9 Batman adaptations point to their documented source comic, in 99.6% of 30,240 analysis variants.
- **Words do as well.** Plain TF-IDF recovers the sources as well as the neurons (share 0.110 against Qwen 0.164; lower is better), and LSA does better (0.064).
- **"Beyond words" splits by model.** The pre-registered residual test fails for Qwen and passes only for Llama.
- **Chance was overstated.** Time-consistent chance is 42 to 44%, not 47%.

On name-free Wikipedia summaries the neurons have not yet earned their extra cost. This design asks whether they earn it where the words change but the idea stays.

### 1.3 Organising question

> When, and at what cost, do neuron-defined memes beat keywords, vectors, topics and LLM reading?

In this phase the question is asked of new stories with planted memes: when a meme's usual words are removed (banned), when it is never named (implicit), and when its key idea is renamed with an invented code word whose meaning comes only from context (coded); and whether the neurons find **different** planted memes from LLM reading, embeddings and keywords, not only more of them. Every method runs on the same stories with the same supervision and the same tuning budget, and every result reports accuracy and cost together.

### 1.4 The primary claims

Each hypothesis is tested in Qwen3-8B (primary) and replicated in Llama 3.1 8B. All tests are one-sided. Each has a smallest effect size of interest (SESOI). Where a hypothesis must hold against several rivals, it is an intersection-union test: its p is the largest of the rival-wise p-values (this needs no further correction). H1 to H3 form the Holm family; H6 is a decision rule.

| | Test | Hypothesis | Statistic | SESOI |
|---|---|---|---|---|
| **H1** | T2 planted memes | On reworded stories (banned and implicit versions), neurons detect a meme better than an **evasion-aware keyword analyst** and better than a **lexical classifier**. | Mean over memes of twin AUC(neurons) minus twin AUC(rival), for `keywords:expanded-qwen3-8b` and `keywords:logreg` (intersection-union) | 0.05 AUC |
| **H2** | T2 planted memes | On reworded stories, neurons are not worse than the strongest vectors by more than 0.05 AUC (non-inferiority; superiority is then tested as a fixed sequence). | Same, against `emb:qwen3emb-4b`, `embwin:qwen3emb-4b` and `emb:e5-large` (intersection-union), with a 0.05 margin | margin 0.05; superiority 0.05 |
| **H3** | T2 planted memes | The 11 concept units found on DEV Batman text respond **specifically** to their own meme in unrelated genres. | Mean over the 11 units of [twin AUC on its own meme minus its mean twin AUC on every other meme]; null: meme labels permuted across the units | 0.05 AUC |
| **H6** | Cost and accuracy (decision rule) | On reworded planted stories, no rival method dominates the neurons: none is at least as accurate (its AUC's one-sided 95% lower bound at least the neurons' estimate, both on the memes they share) and no more expensive per story, at 34 memes. The neurons' own variants (either model's neurons, m1/m5 and the other selection variants, DEV units, length-residualised neurons) and the Claude-based entries are not rivals. | `analysis.run_cost` (`H6`, `analysis.h6_rival`) | none (a decision, not a test) |

**Numbering.** prereg-v2 also had H4 (lineage recovery under vocabulary drift) and H5 (renamed Shakespeare adaptations). Both are deferred with the lineage tests (section 15). The numbers H1, H2, H3 and H6 are kept so that the design history, the reviews and the code stay readable; H4 and H5 are not tested here and enter no correction.

"Twin AUC" compares each planted story with its own plot's control version: the same frame, word for word, with the slot filled by a different event of equal stakes (`analysis.detection_aucs`, scheme `twin`). "Neurons" means each model's detector selected on DEV (section 2). H1 and H2 must pass both a meme-level and a plot-level sign-flip test (section 5).

**Verdicts** (section 5): *supported* when the Holm-corrected test rejects (and any required condition holds); *supported, below SESOI* when it rejects and the falsification test also rejects (a real effect, credibly smaller than the SESOI); *falsified* when only the one-sided falsification test rejects; *undetermined* otherwise; *inconclusive* when a section-7 check fails or cannot be computed; *not run* when the data fall short of the minimum in section 3. A Llama result *replicates* when the same test gives p < 0.05 uncorrected.

### 1.5 How each outcome reads (fixed in advance)

| | If supported | If falsified |
|---|---|---|
| H1 | Neurons keep tracking a meme when its usual words are removed better than an analyst who expects evasion and better than a lexical classifier. This is the fuzziness claim. | Keywords chosen with evasion in mind, or a lexical classifier, do as well: the fuzziness claim does not separate neurons from good keyword practice. |
| H2 | Neurons are within 0.05 AUC of the strongest sentence, window and same-family embeddings (better, if the superiority step passes). | Embeddings are better by more than 0.05: the case for neurons rests on interpretability (T6), cost (H6) and what they find that others miss (the overlap analysis), not accuracy. |
| H3 | Units found on Batman respond to their own meme, not to drama in general, in genres Batman never visits. | They fire on any eventful passage: DEV meme labels do not transfer as meme detectors. |
| H6 | No cheaper method is as accurate. | A cheaper method (embeddings, a shared-sheet LLM judge, a small LLM) is as accurate: the neuron case rests on interpretability. |

A result in which LLM reading beats everything is a finding, and is reported as one. So is a result in which the neurons win only through the model's dense hidden states (the neurons-versus-hidden-states contrast says which), and one in which the neurons score lower than a rival but find planted memes the rival misses (the overlap analysis says which).

### 1.6 Two secondary analyses reported beside the primary claims

Mike asked for these two to be reported prominently. They are secondary, with their predictions fixed here, and form their own Benjamini-Hochberg family of 30 tests in Qwen3-8B (S-coded 6, O1 4, O2 20; section 5), separate from the other secondaries so that easy discoveries elsewhere cannot loosen them. They appear in the same results table as H1 to H3 whatever they show (section 8).

- **S-coded, the code-word condition** (section 4, T2). In the coded version a meme's usual words are replaced by an invented or repurposed code word whose meaning comes only from what happens, the way a group changes its vocabulary to evade keyword moderation. The writer names each code word and a script checks that it is introduced and used only in the coded fill (Q2c). Prediction: methods that read each sentence in the context of the story (the neurons, the same model's hidden states, the direct LLM judge) beat methods that do not (sentence embeddings, the evasion-aware keyword query). The headline "context-using methods beat context-free ones on code words" holds only if all 6 Qwen tests pass in their family and the audit's coded gate passes; otherwise the pairs that pass are listed.
- **Overlap, what each method finds** (section 4, T2). For every pair of methods, which planted memes one detects and the other misses, checked against the design, at an operating point fixed in advance and with a detection rule that keeps the twin control: a planted story counts as detected only when the method flags it and does not flag its own plot's control version. Prediction O1: the neurons find planted memes that the evasion-aware keyword query and the lexical classifier miss. O2: against embeddings, hidden states and the direct judge no direction is predicted; both directions are tested.

## 2. Methods compared

**Same inputs.** For a named meme every method gets its label, its definition and the same example sentences: the **main set** (the definition plus 5 explicit and 5 reworded examples). Two secondary variants use the **explicit set** (definition and explicit examples only) and the **far set** (the definition plus 5 far examples, set away from Batman's usual settings). Every few-shot method contrasts the positives with the same **contrastive background**: the 2,267 DEV reference-film sentences, every other meme's example sentences, and the meme's own 5 hard negatives (its usual words used without the meme). The examples are written before F0 by fresh agent sessions that see only the label and definition (`annotation_protocol.md` A8).

**Same tuning budget.** Every T2 detector family has two variants, chosen by leave-one-example-out (LOEO) on the main set (the held-out positive's percentile among the background), fixed at F1 on DEV inputs only. The hidden-state layer is chosen on the DEV lineage statistic (E2) among four layers, as in v2; the DEV lineage grid (T1a) is computed at F1 for that and for the DEV reproduction check only, and is reported as DEV context.

**Same aggregation.** Sentence-level scores become story scores as the neurons' do: the mean of the 3 strongest sentences.

### 2.1 Neurons (primary: Qwen3-8B; replication: Llama 3.1 8B)

- **Models.** 4-bit MLX (`mlx-community/Qwen3-8B-4bit`, `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`). Each story runs as one continuous document, so every token's activations depend on the story so far.
- **Units.** A unit is one SwiGLU channel split by sign: 884,736 units in Qwen, 917,504 in Llama. A sentence's value is the maximum over its tokens (minus the minimum for a negative pole); a story's score is the mean of its 3 strongest sentences.
- **Presence.** A unit is present in a story when its score beats all but one of the 69 reference films (in the same text condition). Used for F1 at each method's presence bar and for the label test's 1-20% band.
- **Named memes** (`neurons:<model>`): `m1`, the single unit with the highest mean percentile of the positives within the background (ties: the margin over the 99th percentile), or `m5`, the mean of the 5 best units after background standardisation; LOEO chooses. "Single neurons are memes" is claimed only if m5 minus m1 has a one-sided upper bound below 0.03 AUC.
- **Secondary variants:** `:explicit` (selected on the explicit set), `:far` (far set), `:m1ctx` (selected against in-context background: reference-film sentences inside their films, each example after a fixed 3-sentence DEV paragraph), `length_resid` (each unit's sentence values residualised on log word count, slopes from the background).
- **DEV-mapped units** (`devunit:<model>`, H3): 18 memes have a DEV-labelled Qwen unit (`dev_units.json`); 11 of them are concept units (the H3 set), 4 name detectors and 3 word detectors. Llama uses each unit's DEV twin; for revenge, Llama's own DEV-chosen revenge unit (chosen from 50 candidates on DEV; Qwen's failed its DEV test).
- At F1 only, the lineage configuration `neurons:<model>@base` gives the DEV reproduction check (section 7). The other lineage configurations belong to the deferred tests.

### 2.2 Keywords

- **Keyword queries.** A sentence's score is the sum over terms (a one-word term counts its stem's occurrences; a multi-word term counts 1 when all its content stems occur); a story's score is the mean of its 3 best sentences plus 0.001 x total hits. X-codes and digits are removed for every lexical method. Queries:
  - `keywords:expanded-qwen3-8b` (**H1 rival**): Qwen3-8B as an evasion-aware moderation analyst lists 30 to 40 mostly single-word terms (synonyms, euphemisms, slang, indirect indicators), sees each term's hit counts on the examples and on the DEV background, revises once (`readers.KEYWORDS_EXPANDED`, `KEYWORDS_REVISE`); every one-word term then gets up to 5 WordNet synonyms. `keywords:expanded-llama3.1-8b` likewise.
  - `keywords:qwen3-8b`, `keywords:llama3.1-8b`: the plain analyst (up to 10 terms), secondary; `keywords:qwen3-8b` is also the explicit-wording manipulation check (section 7).
  - `keywords:tfidf` (top 10 contrastive unigrams and bigrams) and `keywords:tfidf-wordnet` (the same with WordNet synonyms).
  - `keywords:ban-list-aligned`: the plain prompt given to Claude, who also wrote the ban lists; the query most likely to coincide with them.
  - Each query's share of terms that fall on the ban list is published at F1 (`dev_freeze.json`, `t2.ban_list_overlap`).
- **`keywords:logreg`** (**H1 rival**): an L2 logistic regression (C = 1, balanced classes) on TF-IDF word uni- and bigrams of the positives against the contrastive background.
- **The ban lists are never given to any method.**

### 2.3 Vectors

- **Encoders** (pinned revisions, `readers.ENCODERS`): `qwen3emb-4b` (Qwen/Qwen3-Embedding-4B, the strongest encoder that fits this laptop, fixed a priori as primary), `e5-large`, `e5-base` (multilingual), `labse`, `minilm`.
- **Units of text:** sentences (`emb`), 3-sentence windows centred on each sentence (`embwin`, so the encoder sees the slot's context as the neurons do), and whole stories (`embdoc`).
- **Named memes:** two variants per encoder, chosen by LOEO: the contrastive prototype (mean positive minus mean background embedding, cosine) and an L2 probe (C = 1, balanced) on the same positives and background.
- **`hidden:<model>`**: the same model's mean residual-stream state per sentence at one of four layers (quarter, half, three-quarter, last), reference-standardised; the layer is chosen on DEV E2; named memes by prototype or probe (LOEO). The neurons-versus-hidden contrast says whether a neuron win comes from the neuron basis or from the model's dense representation.

### 2.4 Topics

- **`lda`**: sklearn LDA (batch VB, default priors, 20 iterations, seed 0) on sentences, K in {50, 100} (LOEO); for the label test, K = 100 on stories.
- **`topics:<enc>`** (BERTopic-style): UMAP to 5 dimensions (15 neighbours, min_dist 0, cosine) then HDBSCAN (min_cluster_size 15). Without umap-learn the pre-registered fallback is PCA to 10 dimensions (a reported deviation).

### 2.5 LLM reading

The readers are Qwen3-8B (thinking off) and Llama 3.1 8B Instruct (`readers.py`); every scored answer is a next-token probability, p_yes, with logit = log P(yes) - log P(no). Claude appears only as a labelled upper bound.

- **`judge:<model>`**: the story is cached once; one full question per meme (label, definition, main examples).
- **`judge-shared:<model>`**: the meme sheet (all 34 memes' labels, definitions and examples) is cached once for the whole set; each story is appended; a short question per meme. The cheapest reasonable LLM judge.
- **`judge-1pass:<model>`**: sheet, story and one answer sheet read in a single forward pass.
- **`judge:qwen3-1.7b`**: the small reader (Qwen3-1.7B, 4-bit MLX, pinned).
- **`summary-judge:<model>`**: recursive summarisation (8-sentence chunks to 60 words, merged to 150 words, in English), then the full-question judge on the summary.
- **`nli:deberta-v3-base`**: zero-shot NLI (MoritzLaurer/deberta-v3-base-zeroshot-v2.0, pinned), hypothesis "This text describes the following: LABEL. DEFINITION", per sentence.
- **`judge:claude-upper-bound`** (A4): labelled upper bound, never a check and never a rival in H6; run after the author's audit is frozen.

### 2.6 Floor

`length`: the mean word count of a story's 3 longest sentences. It appears in every table; on twin pairs it should sit at chance (fills are length-matched, check Q1b).

## 3. DEV, TEST and the order of work (the data-generation plan)

- **DEV**: the existing Batman corpus (`study/data/corpus.jsonl`, `study/data/corpus_neutral.jsonl`: 32 Batman works, 13 cousins, 69 reference films) and every existing result.
- **TEST**: the new data of this phase: the 816 planted stories (136 plots in 6 versions) and the 48 long texts built from them.

Nobody runs a method on a TEST story before freeze F2-T2 and the author's audit (A7) are committed, and nobody looks at a TEST outcome before the feature freeze F3. At every freeze point the named files are committed **and pushed**, and the commit hash is noted in `DEVIATIONS.md`. Run Python with `.venv/bin/python` from the project root; models run offline (`HF_HUB_OFFLINE=1`) after step 0. Times are for this laptop (M-series, 24 GB); "sessions" are fresh Claude agent sessions.

| Step | What | Time | Sessions | New disk |
|---|---|---|---|---|
| P0 (done) | scout caches in `sources/`; `build_lists.py lists` and `memes` build every work list | | | |
| 0 | installs and downloads (approved) | about 1 h | | about 20 GB |
| P1 | example sentences (A8) | half a day | about 170 | under 10 MB |
| P2 | pilot of the planted-story protocol; the author times the audit page on at most 5 pilot stories | half a day (Mike: about 15 min) | about 120 | under 10 MB |
| F0 | tests pass; commit and push this folder | minutes | | |
| 2 | DEV features | about 3 h | 34 (A5) | about 42 GB |
| F1, F1b | DEV freeze; reference films; cost benchmark | about 3 h | | about 1 GB |
| 4, F2-T2 | planted stories with reads, ratings (banned, implicit, coded and control fills) and one repair round; long texts | about 1 day | about 2,690 | under 50 MB |
| A7 | the author's audit, through a local page | 2 to 2.5 h (Mike) | | under 1 MB |
| 5, F3 | TEST features, the Claude upper bound (A4); feature manifest | about 16 h | about 110 (A4) | about 6 GB |
| 6 | analysis; label test | about 5 h | about 1,310 | under 1 GB |
| **Total** | | about 3 days elapsed; about 25 h of local model time; about 2.5 h of Mike's time | about 4,430 | about 70 GB |

**P0 (done).** The scout caches are in `sources/`; `build_lists.py lists` builds every work list from them and from the committed DEV results (`study/results/labels.json`, `study/results/wording.json`, the gzipped `study/results/<model>/memes.json.gz`), deterministically, so a clean clone rebuilds them (rebuilt on 2026-10-21 after the renaming in section 14.4: identical except the renamed fields; `dev_units.json` rebuilt from the gzipped copies is byte-identical). Every model is pinned to a snapshot hash in `readers.py` and `extract_compact.py`, the two 8B readers included (the DEV study's snapshots).

**Step 0, installs and downloads** (approved by Mike on 2026-10-21; done first, because the pilot already uses the cross-family reader): sentence-transformers 6.1.0 (LaBSE only), umap-learn 0.5.12 (with numba and pynndescent) and nltk 3.10.3 with its WordNet data; Qwen3-Embedding-4B, multilingual-e5-large, multilingual-e5-base, LaBSE, all-MiniLM-L6-v2, DeBERTa-v3-base-zeroshot-v2.0, Qwen3-1.7B-4bit and Gemma-2-9B-it-4bit (about 20 GB) at the revisions pinned in `readers.py`. Nothing else is installed. A download that fails triggers the named fallback (PCA for UMAP; no WordNet expansion; Mistral-7B for Gemma) or "not run".

**P1 (before F0, agents).** Example sentences (A8): for each of the 34 memes, four fresh writer sessions (explicit, reworded, far, hard negative) see only the label and definition (with the ban list and decoy hint where A8 says) and write 7 sentences each (`preregistration/agent/examples/raw.jsonl`); a blind reader session judges every sentence (`reads.jsonl`); `build_lists.py texts examples` screens them (ban-list rules, forbidden setting words, the TF-IDF far screen) into `examples_final.jsonl`, a set with fewer than 3 survivors going back to a fresh writer once; then `build_lists.py memes`. (Relation typing, A6, belonged to T3 and is deferred.)

**P2 (before F0, pilot; never TEST).** `build_lists.py texts pilot` writes 6 pilot plots in genres outside the design. The planted-story protocol runs on them in full: writer calls (with code terms), checks Q1 to Q6 (Q2c included) with two Claude readers and the cross-family reader (`readers.py check --task xread`), salience ratings of the banned, implicit, coded and control fills, one repair round (about 120 sessions and 10 minutes of Gemma). The audit page (`audit_page.py`, written before F0 and tested by `test_analysis.py` on the synthetic stories; section A7 of `annotation_protocol.md`) is timed by Mike on **at most 5** pilot stories (`build_lists.py texts audit-pilot`, about 15 minutes); the rest of the page is tested by script, not by him. If his median time per story is above 150 seconds (`audit_page.py --pilot-report`), the main audit shows 4 blocks (48 stories) instead of 5 (`analysis.AUDIT_BLOCKS_SHOWN`); the decision is logged in section 14 before F0. Pass rates, calls per item and his times go to `preregistration/agent/pilot/pilot_report.json`. Protocols may change only on pilot evidence, each change logged in section 14 before F0. No method is run on pilot material. (The drift and translation pilots of v2 are deferred.)

**F0.** Before it: the "Why" paragraph of section 14.4 records Mike's decision (done 2026-10-21); the book is not cited in this folder, and the conversation transcript keeps its mentions as a record of what was said; `test_analysis.py` passes. Then commit and push this folder, with the audit page script. From here the hypotheses, primary methods, SESOIs, negative sets, operating points, gates and verdict rules are locked.

**Step 2, DEV features** (both models `<m>`, every encoder `<e>`; about 3 h; full activations about 27 GB, aggregates about 14 GB, hidden states under 1 GB):

```
python preregistration/build_lists.py texts dev
python preregistration/extract_compact.py dev_namefree <m> --hidden                # DEV reproduction check, hidden-state layer
python preregistration/extract_compact.py dev_original <m> --full --hidden         # far-set screen, in-context check, m1ctx
python preregistration/extract_compact.py t2_background <m> --full --hidden --prefix
python preregistration/extract_compact.py t2_examples <m> --full --hidden --prefix
python preregistration/extract_compact.py t2_examples_ctx <m> --full
python preregistration/readers.py embed --set S --encoder <e>            # S = dev_namefree, dev_original, t2_background, t2_examples
python preregistration/readers.py keywords --model <m> [--expanded]       # plain and evasion-aware, both models; A5 for Claude (34 sessions)
```

(v2's DEV runs for the deferred tests, `t3_dev_masked`, the Chinese keyword translations and the DEV summaries and summary pairs, are dropped; T1a's LLM lineage rows are then reported as missing.)

**F1 (DEV freeze, about 1 h).** `python preregistration/analysis.py dev-freeze` writes `dev_freeze.json` and `sentence_units_<model>.npy`: every detector, query, prototype and probe; the hidden-state layer; the DEV reproduction check (compact-extractor E2 within 0.03 of 0.164 and 0.102 with E2's tie rule, else stop); the far-set screen; the in-context check; the per-sentence unit list; timers for every step. Commit and push. **F1b (about 2 h).** `extract_compact.py t2_reference <m> --units preregistration/sentence_units_<m>.npy`, `readers.py embed --set t2_reference --encoder <e>`, and `bench.py run` (DEV texts only, nothing else running); commit `results/bench.json`.

**Step 4, TEST data (freeze F2-T2; about a day; about 2,690 sessions and 1.5 h of Gemma):**
1. 136 writer calls, one per plot, by the 4 style briefs (`planted_story_protocol.md`); each returns its code terms; a script builds the 6 versions into `preregistration/agent/stories/stories.jsonl`.
2. Q1 to Q3 by script (frame, length, words, code terms Q2c, names).
3. Q4 and Q5: every version read by two Claude reader sessions from a pool of fresh sessions (never two versions of one plot) and by the cross-family reader (about 1,630 reads plus Gemma).
4. Q6: blind salience ratings of the banned, implicit, coded and control fills of every plot (544 rater sessions).
5. One repair round for failing plots (the same writer with the failure list), with their reads and ratings (about 400 sessions in all).
6. `build_lists.py texts stories` (valid, exclusions, dropped memes) and `build_lists.py texts long` (48 long texts).
7. Every protocol check runs and exclusions are recorded; commit and push (F2-T2).

**A7, the author's audit (after F2-T2, before step 5; about 2 to 2.5 h of Mike's time, one block or more per sitting).** `build_lists.py texts audit` draws the sample by seed from the frozen stories (`analysis.human_audit_sample`: 60 stories, 10 of each version, one per plot, each asked about its 3 design memes and 2 distractors; 300 cells; 5 balanced blocks of 12 stories, 2 of each version) into `data/human_audit_sample.json`, the page's only input: opaque ids A01 to A60 in presentation order, the numbered sentences and the 5 memes' labels and definitions, and no story id, plot, version, writer or genre (`analysis.audit_sample_file` refuses to write a file that would show one). `python preregistration/audit_page.py` serves the page on 127.0.0.1; Mike opens it in a browser, reads one story at a time and answers "happens", "does not happen" or "unsure" for each of the 5 elements shown; every answer is appended to `data/human_audit.jsonl`, and the last answer of a cell counts. The audit is complete with at least 4 whole blocks. Commit and push both files (freeze A7) before step 5. No method, and not the Claude upper bound, runs on a story until then.

**Step 5, TEST features** (after F2-T2 and A7; both models `<m>` unless noted; neurons about 1 h, embeddings about 1 h, LLM reading about 13 h, NLI about 40 min; and the Claude upper bound, A4, about 110 sessions, into `features/t2_stories/llm_claude_judge.jsonl`):

```
for S in t2_stories t2_long; do extract_compact.py $S <m> --units preregistration/sentence_units_<m>.npy --hidden; done
readers.py embed --set S --encoder <e> [--unit win3 | --unit doc]        # S = t2_stories, t2_long; sentences, windows, documents
readers.py judge | judge-shared | judge-1pass --set t2_stories --model <m>;  judge --set t2_stories --model qwen3-1.7b
readers.py summarise --set t2_stories --model <m>;  sjudge --set t2_stories --model <m>
readers.py judge --set t2_long --model <m>;  summarise --set t2_long --model <m>;  sjudge --set t2_long --model <m>
readers.py nli --set t2_stories;  nli --set t2_long
```

**F3 (feature freeze).** `python preregistration/analysis.py manifest` writes the sha256 of every feature file (the A4 file included; `features/` itself is not committed) to `results/feature_manifest.json`; commit and push it. Before `t2`, `t6-prepare`, `t6` and `cost`, `analysis.py` recomputes the hash of every file in the manifest and refuses to run on any missing, changed or unlisted feature file (`analysis.verify_manifest`); every run's output hash is appended to `results/run_log.jsonl` (the first run is the confirmatory one).

**Step 6, analysis (about 2 h of compute and 3 h of agent time).** `analysis.py t2`; `analysis.py t6-prepare`, then the label test (A3: 32 labeller and 1,280 judge sessions), then `analysis.py t6`; `analysis.py cost`; `analysis.py primary`.

**Minimum viable completion.** A result whose data falls short is reported as "not run" with the reason: T2 needs at least 100 plots with all six versions valid and at least 28 memes not dropped (`analysis.t2_viability`; otherwise H1 to H3, H6 and every T2 secondary are "not run"); T2-long at least 40 long texts; T6 at least 30 qualifying units per kind and arm (a kind and arm below it enters no test); the audit counts as done with at least 4 whole blocks (48 stories) answered (otherwise the cross-family rule of section 7 applies, and the partial audit is reported).

## 4. Tests

### T2. Named memes: planted stories (H1, H2, H3, H6, S-coded, overlap)

**Memes** (`memes_for_tracing.json`, 34): Tier A (10, the orphan-vigilante family: the memes Batman shares with the other "billionaire vigilante orphans", Zorro, the Scarlet Pimpernel and Iron Man), Tier B (10, Batman story memes), Tier C (4, general memes) and Tier N (10 **neutral motifs** drawn by seed from Wikipedia's motif and plot-device categories before any unit was examined: closed circle of suspects, poisoned garment, rash promise, locked-out lover, self-fulfilling prophecy, love triangle, late-running collision meeting, literal loophole, thrown shoe, exile; `sources/neutral_motif_walk.json`). Each has a definition, a ban list, a decoy hint and four example sets (A8).

**Stories** (`planted_design.json`, `planted_story_protocol.md`): 136 plots in 17 genres unrelated to Batman, by 4 writer agents; each plants 3 memes; each meme in 12 plots; 6 versions per plot with an identical frame (explicit, banned, implicit, coded, control with an equal-stakes event, decoy); 816 stories of 18 to 30 sentences; every fill within 15% of the explicit fill's length; checks Q1 to Q6 (symmetric exclusions; salience ratings).

**Detection** (`analysis.detection_aucs`). Primary scheme **twin**: each planted story against its own plot's control version (same frame; equal-stakes event; frame and salience matched). Per meme, 24 reworded pairs (banned and implicit) across 12 plots, and 12 coded pairs. Secondary schemes: **matched** (negatives: the same-condition versions of plots that do not plant the meme), **all** (the draft's rule), **hard** (own control and decoy), **vs_decoy**, and **decoy against control** (a construction check: keywords should rank decoys above controls). Per protocol (Q4 to Q5 exclusions and dropped memes) and intention-to-treat.

**H1, H2.** As in section 1.4, over all non-dropped memes. Each comparison must pass both the meme-level sign flip and the plot-level sign flip (`analysis.plot_flip_test`, which flips every plot's pairs together across its three memes); falsification likewise needs both. **Required consistency:** the mean difference on the 10 neutral memes has the predicted sign (H1: above 0; H2: above -0.05). Reported beside each: the two-way (meme by plot) bootstrap interval, per-writer estimates, leave-one-writer-out estimates, the mean difference per tier (A, B, C, N), and the intention-to-treat result, for H1 and for H2's non-inferiority test alike, with the same meme-level and plot-level tests (the larger p) and the same intersection-union rule (a supported claim whose ITT p exceeds its Holm threshold is labelled "per-protocol only").

**H3.** The specificity test (`analysis.specificity_matrix`, `specificity_test`) on the 11 concept units. Secondary: all 18 DEV units; concept minus name/word units; split by DEV rewording survivors; the draft's AUC-above-0.5 test.

**H6.** Section 9.

**S-coded, the code-word condition** (secondary, prediction stated; `analysis._t2_coded`). In the coded version the fill names the meme's key idea with a consistent invented or repurposed term that a reader can understand only from what happens; none of the meme's usual words appear anywhere in the story (`planted_story_protocol.md`). It models a group changing its vocabulary to evade keyword moderation. A sentence embedding or a keyword list scores the code word without its context; the neurons and the hidden states are read inside the whole story, and the judge reads all of it.
- **Prediction:** each context-using method, `neurons:<model>`, `hidden:<model>` and `judge:<model>`, beats each context-free method, `emb:qwen3emb-4b` (sentences) and `keywords:expanded-qwen3-8b`, on coded versions.
- **Statistic:** mean over non-dropped memes of coded twin AUC(a) minus coded twin AUC(b), up to 12 coded-against-control pairs per meme; one-sided; the meme-level and plot-level sign flips are both required (p = the larger), as for H1. Only memes with at least 6 usable coded pairs enter (`analysis.CODED_MIN_PAIRS`); the others are listed apart with their counts. Six tests in Qwen3-8B, in the prominent Benjamini-Hochberg family (section 5); the Llama tests are a replication. Power: section 6.
- **Condition:** the audit's coded gate (section 7). If it fails, the six tests are inconclusive and enter their family with p = 1.
- **Headline rule:** "context-using methods beat context-free methods on code words" is stated only when all six Qwen tests have q < 0.05 in their family and the coded gate passes; otherwise the pairs that pass are listed, and nothing more is claimed.
- **Sensitivity:** the six tests without the plots whose coded fill is still more than 1 point from the control in salience after repair (Q6).
- **Reported in the main results table:** every method's twin AUC on explicit, banned, implicit, coded and reworded versions and its coded-minus-banned difference (what the code word costs that method); for the neurons, their coded-minus-banned difference against each context-free method's (two-sided, descriptive); the overlap analysis on coded versions; the audit's coded cells; coded salience against the control.

**Overlap: which planted memes each method finds** (secondary; O1 predicted, O2 without prediction; `analysis.overlap_analysis`). The question is whether the neurons find different things from LLM reading, embeddings and keywords, not only whether they score higher.
- **Methods** (fixed, `analysis.OVERLAP_METHODS`, 17): `neurons:qwen3-8b`, `neurons:llama3.1-8b`, `keywords:expanded-qwen3-8b`, `keywords:logreg`, `keywords:tfidf`, `emb:qwen3emb-4b`, `embwin:qwen3emb-4b`, `emb:e5-large`, `hidden:qwen3-8b`, `hidden:llama3.1-8b`, `judge:qwen3-8b`, `judge-shared:qwen3-8b`, `summary-judge:qwen3-8b`, `judge:llama3.1-8b`, `nli:deberta-v3-base`, `lda`, `topics:qwen3emb-4b`. A missing method is reported as missing.
- **Operating point (fixed in advance).** For each method and meme, exactly k = floor(0.10 n) of the n usable control versions (every plot's control version, less Q5 exclusions; `analysis.operating_points`) are flagged. Ties are broken by a seeded uniform key per story and meme, applied in the same way to control and planted stories: stories are ordered by (score, key) and a story is flagged when its pair is above the (k+1)-th largest control pair. Every method, a keyword query whose scores are mostly tied at zero included, thus works at the same realised control false-positive rate, k/n (0.096 with 136 controls), which makes their detections comparable. Sensitivity: 5%.
- **Detection rule (fixed in advance; twin-consistent).** A planted cell counts as detected when the method flags the story **and does not flag its own plot's control version** (same frame, word for word). A method that picks a meme up from the frame of its plots, rather than from the slot, flags the control as well and gets no credit; the threshold alone comes from all 136 plots' controls, of which a meme's own plots are only 12. Reported beside it for every method, meme and condition: the false-alarm rate on the meme's own-plot control versions at the same threshold. Sensitivity: the threshold-only rule (a flagged story counts, whatever its control does).
- **Cells.** Every usable planted (story, meme) cell, by condition: explicit, banned, implicit, coded, and reworded (banned and implicit together; 816 cells, 24 per meme), whose own control version is usable. The ground truth is the design: these cells contain the meme. A pair of methods is compared on the cells where both have a score.
- **Cell level, for every pair of methods and every condition:** the counts and shares of planted cells detected by both, by the first only, by the second only, and by neither; the first-only minus second-only share; 95% intervals from a crossed two-way bootstrap (memes and plots resampled independently, 2,000 draws; a count's interval is its share's interval times the cells). On the decoy versions (the meme's own words used without the meme) the same counts of false alarms.
- **Meme level.** Over memes with at least 6 usable cells in the condition (`analysis.OVERLAP_MIN_CELLS`; the others are listed apart with their counts), a method **finds** a meme when it detects at least 50% of that meme's planted cells, and **misses** it when it detects at most 25%. For each pair: the memes the first finds and the second misses, and the reverse, with their tiers. For pairs that include a model's neurons, in the reworded and coded conditions, each count is tested against a **plot-swap null**: within every plot, the two methods' detections are exchanged with probability 1/2, all of the plot's cells together, 10,000 times; one-sided p = P(null count at least the observed), with the +1 rule.
- **O1 (predicted):** on reworded and on coded versions, the Qwen neurons find planted memes that `keywords:expanded-qwen3-8b` misses, and planted memes that `keywords:logreg` misses (four tests). This follows from H1.
- **O2 (no prediction):** against `emb:qwen3emb-4b`, `embwin:qwen3emb-4b`, `emb:e5-large`, `hidden:qwen3-8b` and `judge:qwen3-8b`, both directions are tested, on reworded and coded versions (twenty tests). No direction is predicted.
- **Conditions.** The tests on reworded versions carry the audit gate of H1 to H3, those on coded versions the coded gate (section 7); a test whose gate fails is inconclusive and enters its family with p = 1.
- **Reading rules (fixed in advance).** "The neurons find planted memes that X misses" when the neurons-to-X test passes (Benjamini-Hochberg q < 0.05 within the prominent family); "X finds planted memes the neurons miss" when the reverse test passes; "they find different memes" when both pass. Cell-level disagreement alone is not evidence of different things, because two noisy detectors of the same thing disagree on some cells; only the meme-level tests support those sentences. The Qwen-against-Llama neuron pair shows how much two neuron detectors disagree with each other. The Llama neurons' tests are reported as a replication, outside every family.

**Other secondary predictions** (the T2 Benjamini-Hochberg family, 7 tests in Qwen3-8B, `analysis.t2_secondary_names`): S8, the direct judge beats the neurons; S9, the summary judge scores below the direct judge; S10, the neurons' AUC is lower on implicit than on banned; S11 (construction check), the plain Qwen analyst ranks decoys above controls; the m1-versus-m5 claim (section 2.1, two-sided); neurons against hidden states (key contrast, two-sided); the T2-long prediction. The Llama versions are a replication.

**Descriptive (no test family):** every pairwise contrast (two-sided, once per pair of methods); S10 for every other method and S11 for every other keyword method; the best encoder chosen in hindsight (labelled as favouring the vectors); the length-residualised neurons; F1 at each method's presence bar; R-precision; every method's twin AUC per tier (A, B, C, N); exclusions by meme, version and writer; reader kappa; the human audit; salience balance.

**T2-long (secondary).** 48 long texts, each 12 valid stories from 12 different plots (2 reworded versions and 10 controls, chosen by seed): per-meme AUC (texts containing the meme against texts that do not) and seconds per text for one neuron pass, the direct judge and recursive summarisation. **Prediction:** recursive summarisation loses more AUC relative to the direct judge on long texts than on single stories: per meme, [AUC on long texts of the summary judge minus the direct judge] minus [the same difference on single reworded stories, negatives every story that does not plant the meme], below 0; sign flip over memes, one-sided (`T2long_prediction_<model>`). The accuracy and cost results of T2 apply to texts of up to about 40 sentences; T2-long is the only long-text evidence.

### T6. Cost and interpretability (on the T2 stories)

**Cost** (section 9). Every method's recipe, total-cost curves and the frontier; H6.

**Interpretability** (A3): the blind label test on the T2 stories, in two arms (random; best-of), 40 units per kind and arm (Qwen neurons, Llama neurons, LDA topics, embedding clusters), all present in 1-20% of the stories; each unit's top sentences come from different plots. A kind and arm with fewer than 30 qualifying units is "not run". **S21:** Qwen neuron accuracy exceeds the word-overlap floor on the same items (exact McNemar), in each arm. Kinds are compared by Fisher tests without a directional prediction, one test per unordered pair of kinds within an arm that includes a neuron kind (5 per arm with four kinds).

**Effort, stated narrowly.** In this design the only human or agent effort the neurons save over keywords is the keyword-choice call. Making neuron memes nameable costs labelling calls; T6 reports calls per usable label and extrapolates to the number needed to label every unit the 34 memes' detectors use. The calls behind the DEV labels used for H3 are counted from the earlier run's logs at F1.

## 5. Multiple comparisons and verdicts

- **Primary.** H1, H2 and H3 in Qwen3-8B, Holm-Bonferroni at a familywise alpha of 0.05, one-sided: the smallest p is compared with 0.05/3, the next with 0.05/2, the largest with 0.05. A hypothesis that is inconclusive under section 7 enters Holm with p = 1. H2's superiority step is a fixed sequence after non-inferiority, at H2's Holm level. H6 is a decision rule outside Holm. The deferred H4 and H5 are not tested and take no share of alpha.
- **Verdicts.** Supported: the Holm test rejects and the required condition holds (H1, H2: neutral-subset consistency). Supported, below SESOI: the Holm test rejects, the condition holds, and the falsification test also rejects (the effect is real and credibly smaller than the SESOI). Falsified: only the falsification test (`analysis.falsification_test`: H0 effect at least the SESOI, one-sided, alpha 0.05; for intersection-union hypotheses, against any rival; for H1 and H2 both the meme-level and plot-level versions) rejects. Undetermined: neither. Inconclusive: a section-7 check failed or could not be computed. Not run: the data fall short of section 3's minimum (`analysis.verdict3`, `analysis.run_primary`).
- **Llama replication** is judged separately (p < 0.05 uncorrected, same direction, same required conditions) and adds no claims.
- **Secondary.** Three Benjamini-Hochberg families, each in Qwen3-8B only, and every result in them is called secondary wherever it is reported (`test_analysis.py` asserts each family's size):
  - **Prominent (30 tests):** S-coded, the 6 pairs of 3 context-using methods (`neurons:qwen3-8b`, `hidden:qwen3-8b`, `judge:qwen3-8b`) against 2 context-free methods (`emb:qwen3emb-4b`, `keywords:expanded-qwen3-8b`); O1, 4 tests (2 rivals by 2 conditions); O2, 20 tests (5 rivals by 2 directions by 2 conditions). An inconclusive test (section 7) enters with p = 1.
  - **T2 (7 tests):** S8, S9, S10 (Qwen neurons), S11 (the plain Qwen analyst), the m1-versus-m5 test, neurons against hidden states, the T2-long prediction.
  - **T6 (12 tests with four kinds per arm):** S21 in each arm (2) and the Fisher tests (5 per arm).
  A family is smaller only when a method is missing or a result is "not run", and the report says which.
- **Replication.** The Llama versions of every secondary test (S-coded, O1, O2 and the T2 family) are reported with uncorrected p, outside every family, and add no claims.
- **Descriptive.** Every pairwise contrast (once per pair), the per-method S10 and S11 tests, the best encoder chosen in hindsight, and the other results listed as descriptive in section 4: estimates and intervals, no family.
- **Exploratory.** Anything not named here.

## 6. Power

Simulated with the actual tests (`history/power_sim_v4.py`, `history/power_v4.json`; 500 replicates per cell, so each figure is within about 0.045): the meme-level and plot-level sign flips (p = the larger) and, for H1 and H2, the intersection-union p over the 2 or 3 rivals. The rivals have signal of their own, story noise shared with the neurons (correlation 0.5), shared meme difficulty (SD 0.08 AUC), method-specific meme heterogeneity (SD 0.06) and plot effects on the planted signal (SD 0.3). Gaps are the realised mean twin-AUC gaps. Holm's three thresholds: 0.05/3 for the smallest p, 0.025, and 0.05. The prominent family's Benjamini-Hochberg thresholds: 0.05 x R/30 with R discoveries (0.05 if all 30, 0.0167 with 10, 0.00167 with one).

| Claim | Test | Assumption | Power at 0.0167 (0.025; 0.05) |
|---|---|---|---|
| H1 | neurons against 2 rivals, intersection-union | both rivals at AUC 0.65 | gap 0.038: 0.22 (0.27; 0.42); 0.060: 0.55 (0.65; 0.78); 0.084: 0.89 (0.93; 0.97); 0.109: 0.99 |
| H1 | same | rivals at 0.60 and 0.68 (the stronger one decides) | gap 0.049: 0.40 (0.46; 0.58); 0.072: 0.76 (0.80; 0.88); 0.098: 0.97 (0.98; 0.99) |
| H2 | non-inferiority at margin 0.05 against 3 rivals, intersection-union | all three rivals at AUC 0.75 | true gap -0.024: 0.03 (0.04; 0.08); 0.001: 0.22 (0.31; 0.44); 0.025: 0.62 (0.71; 0.84); 0.048: 0.93 (0.96; 0.98) |
| H3 | specificity over 11 units | specificity SD 0.15 | 0.05: 0.14 (0.18; 0.30); 0.10: 0.41 (0.50; 0.65); 0.15: 0.81 (0.86; 0.93) (`history/power_v3.json`) |

| Claim | Test | Assumption | Power at BH 0.00167 (0.0167; 0.05) |
|---|---|---|---|
| S-coded (each of 6 tests) | one context-using method against one context-free method, 12 coded pairs per meme | context-free rival at AUC 0.60 | gap 0.045: 0.04 (0.22; 0.38); 0.068: 0.16 (0.50; 0.68); 0.090: 0.36 (0.78; 0.91); 0.115: 0.69 (0.92; 0.98) |
| O1, O2 (each test) | plot-swap meme-level test, 24 reworded cells per meme | two methods detecting 60% of cells, except on k memes (75% against 15%), shared plot effects | k = 2: 0.32 (0.86; 0.92); k = 4: 0.93 (0.98; 1.00); k = 6: 1.00 |

**What the earlier figures were.** prereg-v3 first quoted power from `history/power_sim_v3.py`, which compared the neurons with a single rival that had no signal and independent noise (H1 and H2: 0.34 at 0.0167 for a gap of 0.056; S-coded at alpha 0.05 only). Those are **single-rival figures, upper bounds for the real tests under their own noise model**: the intersection-union tests take the largest p over the rivals, and within one noise model they have less power than a single-rival test when the rivals are equally strong (H1, two rivals at 0.65, gap 0.038: 0.22 against 0.39; H2, true gap 0: 0.22 against 0.48) and the same power when one rival is clearly the stronger. (The new model's shared story noise between methods makes each difference less noisy, so its figures are not directly comparable with the old ones.)

H1 to H3 are underpowered for effects near their SESOI; with the three-way verdict a small true effect gives "undetermined", not "falsified". H2's non-inferiority has little power unless the neurons are at least as accurate as all three vectors. S-coded's single tests are underpowered at the strict end of their family's thresholds unless the gap is about 0.1 AUC; the headline rule (all six tests) needs a large, consistent gap. Narrowing to three primary claims raised Holm's first threshold from 0.01 (five claims) to 0.0167. The number of memes and plots is fixed by the design and cannot be enlarged after seeing data. The overlap test is conservative: with two exchangeable methods it rejected in none of 200 null replicates. It is built to detect memes that one method finds and the other grossly misses, not small differences in detection rate.

## 7. Manipulation checks, positive controls and inconclusive results

A claim whose check fails, or cannot be computed, is reported as **inconclusive** (and enters Holm, or its secondary family, with p = 1); its numbers are still reported (`analysis.t2_manipulation_gates`).

- **DEV reproduction (every neuron result).** The compact-extractor E2 (with E2's strict tie rule) within 0.03 of 0.164 (Qwen) and 0.102 (Llama); otherwise stop before TEST.
- **The author's audit** (A7, `analysis.audit_summary`). Scored on whole blocks only; complete with at least 4 of the 5 blocks (48 stories). Each gate uses the cells the per-protocol analysis uses (Q4 and Q5 exclusions and dropped memes left out), leaves "unsure" answers out of the denominator (they are counted and reported), and passes when the one-sided 90% Clopper-Pearson lower bound of the agreement with the design is at least its threshold and at most 10% of its cells are unsure:
  - **H1 to H3** (and the overlap tests on reworded versions): presence on the banned and implicit design cells, threshold 0.75 (60 cells with 5 blocks: at most 10 disagreements; 48 cells: 7), and absence on the control and decoy design cells, threshold 0.80 (60 cells: at most 7; 48 cells: 5).
  - **S-coded** (and the overlap tests on coded versions): presence on the coded design cells, threshold 0.75 (30 cells: at most 4; 24 cells: 2), and the same absence gate.
  The same gates on every cell, explicit and distractor cells, agreement by version and the author-Claude kappa are reported beside them. A poor coded condition therefore cannot make H1 to H3 inconclusive. With true agreement of 95% on the controls the absence gate fails with probability about 0.01 (60 cells) or 0.03 (48 cells); at 85%, about 0.7.
- **Fallback.** Without a complete audit, the cross-family reader decides: presence at least 75% on the per-protocol banned and implicit cells (H1 to H3) or coded cells (S-coded), and absence at least 90% on every valid control and decoy cell (computed before Q5, which already removes every control cell this reader names). With neither a complete audit nor the cross-family reads, H1 to H3 and S-coded are inconclusive.
- **Explicit wording (H1 to H3 and S-coded).** The plain analyst's explicit twin AUC is at least 0.75 (the explicit wording used the usual vocabulary). If `keywords:qwen3-8b` is missing, so that the check cannot be computed, the claims are inconclusive.
- **Meme drop.** A meme failing Q4 at the explicit or banned level in more than 3 of its 12 plots (plots counted, not versions) is dropped for every method.
- **Missing features** are reported as missing, never imputed.

## 8. What is reported whatever happens

- Every primary statistic in both models: estimate, 95% interval, one-sided bound, p, Holm-adjusted p, the falsification test, the intention-to-treat test and the verdict; the H6 frontier, the rival-by-rival comparisons on shared memes, and the decision.
- One main results table: every method's twin AUC on explicit, banned, implicit, coded and reworded versions, its coded-minus-banned difference, its detection rate at the fixed operating point, and its cost per story; S-coded beside H1 to H3.
- The overlap matrix: for every pair of methods and condition, the counts of planted memes found by both, by one only and by neither, with intervals; the meme lists each way, with tiers, and the memes below the 6-cell minimum; the O1 and O2 tests; every method's realised control false-positive rate and its false-alarm rate on each meme's own-plot controls; the decoy false alarms; the threshold-only and 5% sensitivities.
- Every secondary result of T2 and T6 named in section 4, and per-tier and neutral-meme results for every method.
- The cost recipes, the total-cost curves, the storage policies and the frontier.
- Every quality-check outcome: story checks Q1 to Q6 (Q2c included) with exclusions by meme, version and writer, reader kappas, cross-family agreement, coded salience against the control, the author's audit (each gate with its cells, agreement, lower bound and allowed errors, the same on every cell, agreement by version, distractors, unsure answers, blocks completed), the pilot report with the author's timing.
- The label test.
- The commit hashes of every freeze point and every entry in `DEVIATIONS.md`.

A result against the neurons is reported with the same prominence as one in their favour.

## 9. Cost accounting

**Measured.** Every model-side script writes `cost_<method>.json`: work seconds, model load seconds (separately), MLX peak memory and process max RSS, bytes stored, calls, tokens, and load averages at start and end. Extraction records wall-clock time and load average per batch, so throttling and competing jobs show. `bench.py` measures every method's unit costs on 30 DEV texts after a warm-up, with 3 repeats, nothing else running, under `/usr/bin/time -l`, with the running processes logged; a neurons-only pass is timed separately from hidden-state capture. Every DEV-freeze step is timed. `analysis.run_cost` reads the cost files of the T2 feature sets and the DEV sets they are fitted on.

**Recipes** (`bench.recipes`, `analysis.total_cost`). For each method: one-off setup (neurons: full activations of the background and examples; embeddings: none; LLM judge: none; shared-sheet judge: the sheet prefill), setup per meme (unit selection, probe fit, keyword call), seconds per story (one pass), seconds per story and meme (judge questions, NLI pairs), seconds per story for a meme added later, and storage per story under three retention policies (all units' aggregates, about 1.8 MB per story for neurons; selected units only; none).

**Total cost** C(N stories, M memes, Q later memes) for every method; reported at N = 100, 1,000 and 100,000, M = 1, 10 and 34, Q = 0 and 10.

**What is plain in advance.** One neuron pass is the same forward pass as an LLM judge's prefill; the neurons' cost advantage over LLM reading is only the per-meme questions. Embeddings and keywords also score every meme from one pass. With every unit's aggregate stored, a single-unit meme added later costs one unit selection and a lookup; an embedding probe likewise needs only a fit on stored vectors.

**Frontier and H6.** On reworded planted stories: each method's mean twin AUC, its one-sided 95% lower bound, and its cost per story at M = 34 (setup amortised over the 816 stories). A dominates B when A's lower bound is at least B's estimate and A costs no more (the frontier and every method's dominators are descriptive). For H6 each rival is compared with a model's neurons on the memes the two share: the neurons' mean twin AUC on those memes against the rival's one-sided 95% lower bound on the same memes. The possible dominators are the rival methods only: not the neurons' own variants, of either model (`neurons:<model>` and its m1, m5, explicit, far and m1ctx variants, `devunit:<model>`, `length_resid:<model>`), which share the neurons' cost recipe, and not the Claude-based entries (`judge:claude-upper-bound`, `keywords:ban-list-aligned`; `analysis.h6_rival`). So H6 cannot report "dominated" because Llama's neurons, or the m5 variant, score higher than Qwen's neurons at the same cost.

## 10. Deviations policy

- From F0, the hypotheses, primary methods, SESOIs, negative sets, operating points and verdict rules are locked. Between F0 and F2-T2 only logged bug fixes and feasibility changes are allowed; after F2-T2 nothing about T2 changes.
- Every change after F0 is logged in `DEVIATIONS.md` with the date, the freeze point it follows, what changed and why, whether any TEST outcome had been seen, and its effect.
- Bugs found in frozen code are fixed; results are reported from the fixed code, and from the frozen code too wherever it runs.
- If a step cannot run as written, the named fallback is used, or the result is reported as not run.
- The F3 manifest, the hash check before every test runner and the run log make it checkable that `analysis.py` ran once on frozen features.

## 11. Budget

Rebuilt from DEV measurements: the compact extractor ran 90 DEV sentences (2,342 tokens) in 2.36 s with hidden states, about 38 sentences or 1,000 tokens per second, and reduces on the GPU, writing only aggregates. The cached judge measured 0.36 s to read a 24-sentence story and 0.12 s per meme question; a two-level recursive summary about 6.7 s per story. e5-base measured 443 sentences per second at 64 tokens.

| Item | Estimate |
|---|---|
| Neuron extraction | about 47,000 sentences per model (DEV 14,000, TEST 33,400: 816 stories and 48 long texts), about 25 minutes per model at the measured rate; allow 1 h per model for full-array writes and throttling |
| Neuron storage | full activations for 7,400 DEV sentences (about 27 GB for both models), aggregates for about 4,900 works and stories (about 17 GB), hidden states (about 3 GB), per-sentence unit values: about 48 GB |
| Embeddings | 5 encoders, about 78,000 texts for the primary encoder (sentences, windows, stories); about 1 h in all; under 1 GB |
| LLM reading | about 6 h per 8B model (judge 1.5 h, shared-sheet and one-pass judges 1 h, recursive summaries 1.5 h, summary judge 1 h, T2-long 1 h), about 12 h for both; Qwen3-1.7B 0.5 h; NLI 40 min |
| Check reader (Gemma 2 9B) | about 1.5 h (story reads, pilot and repairs included) |
| Agent sessions (Claude) | about 4,430: examples 170, pilot 120, planted stories 2,690 (writers 190, reads 1,950, salience 550), ban-list-aligned query 34, upper bound 110, label test 1,310; about 12.5 million tokens; about 9 to 12 hours at 8 parallel sessions |
| Author | about 15 minutes timing the page on at most 5 pilot stories (P2); about 2 to 2.5 hours for the audit (A7): 60 stories and 300 questions, or 48 stories and 240 questions if his pilot median is above 150 seconds per story |
| Disk | about 70 GB including about 20 GB of downloaded models; 170 GB free on 2026-10-21 |

## 12. Files in this folder

| File | What it is | Phase |
|---|---|---|
| `PREREGISTRATION.md` | this document | this phase |
| `analysis.py` | frozen analysis: every statistic, the cheap methods, the detectors fitted at F1, the T2 and T6 runners, the overlap analysis, the audit sample, the cost model, the verdicts; the deferred runners (T1, T3, T4, T5, T2-zh, T2-N) are kept and marked DEFERRED | this phase (deferred parts marked) |
| `extract_compact.py` | compact neuron extraction (aggregates for every unit, per-sentence values for a unit list, hidden states) | this phase |
| `readers.py` | LLM judges, NLI, summaries, keyword analysts, embeddings, the cross-family check reader; also the deferred lineage reading, likeness, recognition and translation commands | this phase (deferred commands unused) |
| `bench.py` | the controlled cost benchmark | this phase |
| `build_lists.py` | the work lists from `sources/` and every data-assembly stage (deferred stages marked) | this phase (deferred stages marked) |
| `audit_page.py` | the local page for the author's audit (standard library, 127.0.0.1; reads only the blinded sample; tested by `test_analysis.py`; timed by the author at P2; frozen at F0) | this phase |
| `test_analysis.py` | unit tests, null worlds and the synthetic pipeline; "(deferred)" checks keep the deferred code working | this phase |
| `memes_for_tracing.json` | the 34 memes | this phase |
| `planted_design.json` | 136 plots: genre, writer, 3 memes | this phase |
| `dev_units.json` | DEV units for H3 and the T6 unit lists (and the deferred T5 concepts and nulls) | this phase |
| `planted_story_protocol.md`, `annotation_protocol.md` | how the stories, reads, ratings, examples, upper bounds, label test and audit are made (A2 and A6 deferred) | this phase |
| `drift_protocol.md`, `translation_protocol.md`, `name_removal_protocol.md` | protocols of the deferred tests | deferred |
| `works_shakespeare.json`, `culture_pool.json`, `documented_relatives.json`, `works_batman_extended.json` | work lists of the deferred tests | deferred |
| `sources/` | gzipped scout caches (Wikipedia/Wikidata, 2026-10-20), the robustness-review files `dev_units.json` needs, the neutral-motif draw, timings; `sources/LICENSE.md` | mostly deferred; the neutral-motif draw and timings are used here |
| `history/` | the v1 drafts, the v2, v3 and v4 power simulations (v4: the intersection-union tests, section 6), and a note on citations | record |
| `DEVIATIONS.md` | the deviation log and freeze hashes | this phase |

## 13. What this pre-registration does not claim

- It does not claim the neurons will win. DEV says words match them on Wikipedia-style text.
- It does not test lineage, cross-language transfer, culture-wide likeness, mutation and selection, concept agreement between models, natural annotated summaries or translated stories (section 15). Its results say nothing about them either way.
- It does not test forecasting, and is not written for a journal.
- The planted benchmark measures memes as Claude writes them. Claude wrote the planted stories, examples and ban lists; Claude-based methods are upper bounds; the local models are the LLM rivals. The author's audit and the cross-family reader are the checks on that.
- The coded versions model renamed vocabulary as Claude writers invent it, inside single stories; real groups' code words build up across many texts.
- Detection overlap is measured at one operating point (10% false positives on the controls, 5% as a sensitivity) and with the twin-consistent rule (the threshold-only rule as a sensitivity); at other points, or under other rules, the counts change.
- Accuracy and cost results apply to plot summaries of up to about 40 sentences (T2-long aside).

## 14. Design history

### 14.1 v1 (2026-10-20)

The first draft (`history/PREREGISTRATION_draft_v1.md`) had five primary hypotheses: neurons beat Qwen's keyword query on reworded memes (H1), are non-inferior to e5-base (H2), DEV units fire on planted memes (AUC above 0.5, H3), TF-IDF degrades more than neurons under drift (H4), and neurons beat TF-IDF on loose Shakespeare adaptations (H5). Its 62 synthetic checks passed.

### 14.2 AI design review, round 1 (2026-10-21)

Four adversarial reviews (Claude agents given the draft, the code and the robustness verdict) each returned "needs changes", with 8 blockers and about 50 major or minor problems in all. The reviewers ran their own simulations. The main findings, and what changed in v2:

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

### 14.4 v3: narrowed to the planted-meme stories (2026-10-21, Mike's decision)

**Decision.** On 2026-10-21, before F0 and before any TEST data existed, Mike decided to run only the planted-meme stories experiment (T2: stories with planted memes, neurons against keywords, vectors, topics and LLM reading, with cost), and to defer the lineage tests (T1: drift, cross-language, Shakespeare), the culture scan (T3), mutation and selection (T4), concept agreement (T5), the natural annotated summaries (T2-N) and the translated stories (T2-zh) to a later phase, which is not pre-registered here (section 15). He also approved the model and package downloads of step 0, said he would do the human audit himself (about 2 to 3 hours, T2 cells only, through a simple local page), asked that the source of the orphan-vigilante family no longer be cited (its ideas, such as the "billionaire vigilante orphan" family behind Tier A, are presented without a citation), and asked for two analyses to be kept or added and reported prominently: the code-word condition and the overlap analysis.

**Why.** The decision is Mike's, made on 2026-10-21 when offered three scopes (the full plan, the core tests first, or the planted stories only). He gave no further reason. Planted stories give ground truth by construction, so they need no annotated corpus; that is a property of the design, not a stated reason.

**What changed from v2.**
- Primary claims: H1, H2 and H3 under Holm over three (thresholds 0.0167, 0.025, 0.05), with H6 as the decision rule; H4 and H5 are deferred, keeping their numbers. The statistics, SESOIs, falsifiers and verdict rules of H1, H2, H3 and H6 are unchanged from v2. `analysis.run_primary` evaluates only H1 to H3 and H6; the five-claim path runs only on explicit request and is never used here.
- The code-word condition became a named secondary, S-coded, with a stated prediction and the meme-level and plot-level tests of H1 (v2 had a meme-level test only).
- The overlap analysis was added (`analysis.overlap_analysis`, O1 predicted, O2 without prediction), with its operating point, bootstrap intervals, plot-swap test, a null world and a planted world in `test_analysis.py`, and power in section 6.
- The T2-long prediction is now tested as stated (summarisation's loss relative to the judge, long texts against single stories); v2's code tested the loss on long texts alone, which is kept as descriptive.
- T6 moved from the deferred T3 pool to the T2 stories: the label test draws its units' sentences from the stories (one per plot) with presence against the 69 reference films; cost accounting reads only the T2 feature sets.
- The audit (A7) covers T2 cells only and was restructured to fit 2 to 3 hours: 60 stories stratified by version, one per plot, each asked about its 3 design memes and 2 distractors (300 cells), on a local page, after F2-T2 and before any method runs on a story; "unsure" counts as disagreement; the audit counts as done with at least 48 stories. v2 had 150 cells in 150 separate stories plus T2-N, drift and translation items. (Revised again in section 14.5: blinded file, blocks, gates.)
- Step 0 (downloads) moved before the pilot, which uses the cross-family reader; P1 lost relation typing (A6); P2 lost the drift and translation pilots and gained the audit page; step 2 lost the DEV runs of the deferred tests; step 4 lost every Wikipedia, drift, translation and annotation stage; the freezes are F0, F1, F1b, F2-T2, A7 and F3.
- Power was recomputed for the three-claim family (`history/power_sim_v3.py`); the budget was recounted for T2 (section 11).
- The orphan-vigilante family is presented without a citation everywhere in this folder outside `history/`, which keeps its text with a note (`history/README.md`). The tier A label, and the role, field, relation and strength names in the deferred T3 work lists that referred to the former source, were renamed (`orphan_family`, `family_tier`, `family_entity`, `family_tier1` and `family_tier2`, strength `family`), one citing clause was deleted from an evidence note in `sources/relatives_documented_relatives.json`, and the matching code in `build_lists.py`, `analysis.py` and `test_analysis.py` was changed. The work lists were rebuilt from the sources and differ from v2's only in those names.

### 14.5 AI review of the narrowed design, and what changed (2026-10-21, before F0)

A fifth adversarial review of prereg-v3 returned "needs changes": no blockers, 10 major and 13 minor problems. Every one was fixed or is recorded below; `test_analysis.py` checks each fix (151 checks, all passing).

| Finding | Change |
|---|---|
| **Audit blinding leaked.** Story ids such as `P012-banned` went into the sample file, and the page would have had to open `t2_stories.jsonl`, which holds versions and reads. | The sample file is the page's only input: opaque ids A01 to A60 in presentation order, the numbered sentences and the memes' labels and definitions (`analysis.audit_sample_file`, which refuses to write a file that would reveal a version, plot or story id). The analysis maps the ids back by re-running the deterministic draw and checks the file against it. `audit_page.py` is now written, and `test_analysis.py` checks that neither the file nor the served HTML contains a version name, plot id or story id. |
| **Audit counting.** Every row was counted, so a changed answer counted twice; a story counted as answered with one cell; no test covered changed, unsure or partial answers. | The page appends rows and the last row of a cell counts; a story is answered when all 5 cells are; only whole blocks are scored; tests for each case (`analysis.audit_summary`). |
| **The audit gate did not decide what it claimed.** Coded cells could make H1 to H3 inconclusive; the 95% absence rule on about 48 cells failed 43% of the time at a true 95%; unsure answers counted as errors; excluded cells were scored. | H1 to H3 are gated by banned and implicit presence and control-and-decoy absence; coded presence gates S-coded and the coded overlap tests. Each gate is a one-sided 90% lower bound on the per-protocol cells (presence 0.75, absence 0.80; allowed errors tabulated in section 7 and A7), unsure answers leave the denominator, and more than 10% unsure fails the gate. Every cell is reported beside it. |
| **Operating points were unequal under ties** (realised control FPR 0.000 for the evasion-aware query, 0.046 for TF-IDF, 0.096 for the rest), which favoured the neurons in O1. | Seeded tie-breaking applied in the same way to control and planted stories: exactly floor(0.1 n) controls flagged for every method. The test now requires a realised FPR in [0.08, 0.10] for every method, a tie-heavy one included. |
| **The overlap analysis gave up the twin control**: a method reading a meme off its plots' frame would be credited. | Twin-consistent detection (flagged, and its own plot's control not flagged) is the pre-stated rule; the threshold-only rule is a sensitivity; every method's false-alarm rate on each meme's own-plot controls is reported; a frame-leak world in `test_analysis.py` shows that the leaky method is credited by the old rule (detection 1.00) and not by the new one (0.00). |
| **H6 counted the neurons' own variants as rivals** (Llama's neurons or the m5 variant could "dominate" Qwen's) and compared methods on different meme sets. | Neuron-family entries and Claude-based entries cannot dominate (`analysis.h6_rival`); every comparison is on the memes the neurons and the rival share; a test world where Llama's neurons beat Qwen's. |
| **The T2 BH family was large and inconsistent** (200 tests in the synthetic run, Llama tests inside, duplicated contrasts), so easy discoveries loosened the prominent secondaries. | A prominent family of exactly 30 Qwen tests (S-coded 6, O1 4, O2 20), a T2 family of 7, and a T6 family of 12; Llama tests are replications outside every family; pairwise contrasts (once per pair), per-method S10 and S11 and the hindsight encoder are descriptive; family sizes asserted in `test_analysis.py`. |
| **The coded condition was not well defined**: nothing checked that a code term was introduced, used consistently and only in the coded fill; coded fills had no salience ratings; S-coded had no headline rule. | The writer returns `code_terms`; script check Q2c; coded fills join Q6 (136 more rater sessions) with their own repair and a salience-matched sensitivity; the headline needs all 6 Qwen tests to pass in their family; memes with fewer than 6 usable coded pairs are listed apart. |
| **Section 6 overstated power for H1 and H2** (a single rival with no signal). | `history/power_sim_v4.py` runs the actual intersection-union tests with realistic, correlated rivals and H2 at its margin; section 6 reports those figures and labels the old ones as single-rival upper bounds. S-coded and overlap power are given at the prominent family's thresholds. |
| **Mike's time was not fixed**: trying the page on 36 pilot stories would add about 1.5 h, and stopping early would unbalance the versions. | Mike times at most 5 pilot stories (about 15 min); the page is otherwise tested by script; if his median is above 150 seconds per story the audit shows 4 blocks (48 stories); 60 stories (10 per version) in 5 balanced blocks of 12, so complete means whole blocks. |
| Minor: missing checks did not make claims inconclusive; minimum viable completion rules were not implemented; the meme drop counted versions; A4 ran during the audit; H2 had no ITT; "supported, below SESOI" was undefined; overlap and S-coded had no minimum cells; T6 counted one Fisher test twice; F3 checked only that the manifest existed; `dev_units` read a git-ignored file and the 8B readers were not pinned. | All fixed in code and text, each with a test: inconclusive without a manipulation check or the explicit-wording check; `t2_viability`, the T2-long and T6 minimums; `meme_drop` counts plots; A4 after the A7 freeze, its hash in the F3 manifest; ITT for H1 and H2 with both tests; the new verdict; 6-cell minimums; one Fisher test per unordered pair; `verify_manifest` before every runner; `dev_units` reads the committed `.gz` copies (rebuilt byte-identical); snapshot hashes pinned. |

**Taken differently from the review's suggestion, and why.**
- **The absence gate's threshold.** The review suggested a one-sided 90% lower bound of at least 0.90. At 48 to 60 cells that allows only 1 or 2 disagreements and fails 58 to 70% of the time when the author's true agreement is 95%, worse than the rule it replaces. The pre-stated threshold is 0.80 (at most 7 disagreements in 60 cells, 5 in 48): it fails about 1 to 3% of the time at a true 95% and about 70% at 85%, so it separates clean controls from contaminated ones at these sample sizes.
- **The twin-consistent detection rule.** The review suggested "above the threshold and above its own plot's control score". For a method that reads the frame, planted and control scores are both high and nearly equal, so that rule credits it on about half the cells, enough to "find" (at least 50%) about half the memes by chance. The adopted rule (flagged, and its own control not flagged) gives such a method no credit; its cost is that a true detector loses the about 10% of cells whose control is flagged, equally for every method.
- **The audit strata.** The review's example was 4 blocks of 15 (3/3/3/2/2/2 versions). With the 48-story reduction that cannot be whole blocks; 10 stories of each version in 5 blocks of 12 (2 of each) makes 48 stories exactly 4 whole blocks, and gives 60 rather than 48 absence cells. The banned and implicit cells fall from 72 to 60.
- **The book.** Nothing in `preregistration/` outside `history/` names or cites it (re-checked by the review, including the compressed caches). Outside this folder, the tracked files `study/conversation/export.py`, `conversation.json` and `conversation.js` name the book and `study/.gitignore` (the root `.gitignore` before the repository was reorganised) mentions "the author's book text". They are not part of this pre-registration and were not changed; **Mike to decide before F0** whether "do not cite or name the book anywhere" covers them.
- **The existing paper.** The review noted that whether `study/paper/paper.md` is good enough to share internally before this phase is a separate question that this pre-registration does not answer; it was not reviewed here.


**Pre-freeze fix (2026-10-21).** `test_analysis.py` built its synthetic texts from a Python set in arbitrary order, so the synthetic world, and one deferred check, varied from run to run with hash randomisation. It now sorts the set first. The analysis code is unchanged. The folder also moved from `prereg/` to `preregistration/`, and study paths now point into `study/`.

## 15. Deferred to a later phase (not pre-registered here)

These tests were designed and reviewed in v2 and are deferred by Mike's decision of 2026-10-21 (section 14.4). Their code, protocols and work lists stay in this folder, marked deferred; nothing about them is locked by F0, none of their TEST data is made in this phase, and none of their results is reported as part of it. A later phase that runs them needs a pre-registration of its own, which may change them in the light of this phase.

- **T1, lineage recovery under wording change** (with H4 and H5). Whether the neurons recover documented sources when the words change: T1b rewrites 27 Batman and 44 Shakespeare-film texts at graded levels of vocabulary drift and tests H4, a head-to-head against TF-IDF and LSA at levels 50 to 100 over 21 clusters with the neurons' own lineage test as a conjunct; T1c translates the Batman lineage into Chinese (S3, S5); T1d tests H5, source recovery on 54 renamed Shakespeare adaptations of 13 plays against TF-IDF and LSA; T1e uses 7 fresh Batman adaptations as a positive control (S7). Code: `analysis.run_t1` and the lineage helpers; `drift_protocol.md`, `translation_protocol.md`, `name_removal_protocol.md`; `works_shakespeare.json`, `works_batman_extended.json`. Only T1a, the DEV lineage grid, is computed in this phase, at F1, as DEV context and for the reproduction check and the hidden-state layer.
- **T3, Batman-likeness across culture.** Whether the neurons find Batman's documented relatives among 2,544 masked works (category pool, random films and relatives outside the pool) better than word and vector methods, with franchise-level inference (S14, S15, directional contrasts), and whether the orphan-vigilante family (Zorro, the Scarlet Pimpernel, Iron Man; Spider-Man, John Wick, Puss in Boots) scores high on the Tier A layer (S16, S17). Code: `analysis.run_t3`; `culture_pool.json`, `documented_relatives.json`; relation typing (A6).
- **T4, mutation and selection.** Whether the units new in a Batman work and carried by later works single out The Dark Knight Returns, Year One and The Killing Joke as innovators (S18), and whether the neurons show more cultural selection than words against a margin-preserving null (S19), on all 71 Batman works re-run through one name-removal pass. Code: `analysis.run_t4`.
- **T5, concept agreement without names.** Whether the 27 DEV Qwen concept units have Llama partners that agree on held-out random films better than lexically matched null units (S20). Code: `analysis.run_t5`.
- **T2-N, natural annotated summaries.** Far-set meme detectors on 237 masked natural plot summaries annotated exhaustively by two annotators and an adjudicator (A2; S13). Code: `analysis._t2_natural`; `annotation_protocol.md` A2.
- **T2-zh, translated stories.** The banned and control versions of every valid plot in Chinese (272 stories), with English examples and translated queries (S12). Code: `analysis._t2_zh`; `translation_protocol.md`.
