# a3-method: the method as code (audit)

Scope: read `study/extract.py`, `discover.py`, `phylo.py`, `canon.py`, `experiments.py`, `export.py`, `make_neutral.py`, `run_pipeline.sh`, `wording_test.py` (head), `fetch_corpus.py` (grep), `preregistration/extract_compact.py`, `readers.py`, `bench.py` (head), the committed results, and the full text of Cammarata's essay (fetched 2026-11-08 from https://already-clean.goodfire.com/, 202k characters of text). Nothing was run except small parsing scripts over committed JSON.

## Bottom line

- A "meme" in this code is **one sign of one MLP neuron in one 4-bit model, whose extreme value in a plot summary beats 68 of 69 unrelated film summaries, in at least 6 of 32 Batman summaries**. "Families" barely exist: 95% of them are a single neuron pole (`study/results/qwen3-8b/memes.json.gz`). It is a relative, corpus-dependent detector reading. It is not a validated concept.
- **Nothing in the repo intervenes on a neuron.** There is no steering, ablation or patching anywhere; `Recorder` passes activations through unchanged. Cammarata's case for "clean" neurons rests on exactly those tests (steering, ablation, attribution graphs, blind-maximum and near-miss tests over 10 billion tokens). The study borrows his unit and skips his evidence standard.
- **The selection rule picks word detectors by construction.** "Fires in Batman summaries more than in 69 ordinary films" is most easily met by name and place detectors. The study's own controls say so: random non-meme units recover source comics better than the selected memes on tree distance (mean rank 4.22 and 3.40 against 5.44 and 5.33; TF-IDF 2.89; `study/results/phylo.json.gz`).
- **Several known hazards in the essay are not handled**: the SwiGLU double-negative regime (about 4% of neurons, per the essay), the weak-activation regime (the bar is set by reference films, not by the neuron's own range), small quantised models (8B and 1.7B at 4 bits against the 32B and larger models he audited), and whole-document context (a sentence's value depends on every earlier sentence).
- **Length and pooling confounds are built in**: max over tokens, then mean of the top 3 sentences, on works of 9 to 40 sentences. Sentence count alone gives AUC 0.756 on E1 (`study/results/experiments.json`).
- **The reusable infrastructure is good and mostly unrun**: `preregistration/extract_compact.py` (resumable, pinned snapshots, residual-stream taps, context controls), `readers.py` (LLM judge, embeddings, NLI, keyword analysts, with cost logs), the E2 exact-permutation and residual machinery, and the 30,240-specification multiverse. One neuron pass costs about the same as one uncached LLM yes/no question (`preregistration/sources/*_timing_*.json`).

## Findings

### 1. How a meme is operationalised, step by step

| Step | What the code does | Where |
| --- | --- | --- |
| Models | `mlx-community/Qwen3-8B-4bit` (primary), `Meta-Llama-3.1-8B-Instruct-4bit`, `Qwen3-1.7B-4bit`. Raw text, no chat template, hidden layers only (`model.model(...)`). | `study/extract.py:24-28, 92` |
| Recording site | Every layer's MLP is wrapped. The value recorded is `silu(gate_proj(x)) * up_proj(x)`, the input to `down_proj`. All layers, all channels. Not weighted by the norm of the output weights. | `extract.py:40-45` |
| Unit count | 36 x 12,288 x 2 = 884,736 units (Qwen3-8B); 32 x 14,336 x 2 = 917,504 (Llama); 28 x 6,144 x 2 = 344,064 (Qwen3-1.7B). | `n_units` in each `results/<model>/memes.json.gz` |
| Pole | For each sentence, the max over its tokens (positive pole) and the min over its tokens (negative pole). The two are separate units downstream. 50.3% of Qwen families are negative poles. | `extract.py:43-44`; computed from `memes.json.gz` |
| Context | Each work is ONE sequence: its sentences joined by single spaces. Sentence spans come from character offsets. The first token of the sequence is never in a span (`i > 0`). | `extract.py:48-63` |
| Storage | fp16 scratch memmap, then int8 with one scale per neuron (max abs / 127). | `extract.py:84-119` |
| Work score | Mean of the work's 3 strongest sentence values per unit, computed from the int8 copies. | `discover.py:33-43` (`TOP_K_SENT = 3`) |
| Presence | Work score above the second-highest score among the 69 reference films. Strict `>`. | `discover.py:63-66` |
| Candidate | Present in at least 6 Batman works (`MIN_WORKS = 6`). | `discover.py:77` |
| Family | Average-linkage clustering of candidates on 1 minus Pearson r of their 3,700-sentence profiles, cut at r = 0.5 (`MERGE_CORR`). The strongest member represents the family; the family's presence is the representative's presence only. | `discover.py:84-121` |
| Label | Not produced by any script. `memes.json` stores each family's top 8 sentences; an AI reader wrote labels from them (`results/labels.json`, 31 labels; `results/labels_wf.json`, 120; 125 distinct units). | `discover.py:106, 119`; paper.md line 89 |
| Tree | Binary presence matrix (works x all families), Jaccard distance, neighbour joining (Biopython), rooted on Puss in Boots. | `phylo.py:27-33, 116-127, 199-207` |
| Tests | Era ratio and Mantel with 5,000 permutations; adaptation rank test; TF-IDF baseline (English stop words, min_df 2, sublinear tf); random-unit control (50 draws from units present in 3+ lineage works). | `phylo.py:48-113, 185-233` |
| Cross-model | For each Qwen meme, the best Pearson r over all units of the other model on the same 3,700 sentences. Nulls: random active units, shuffled sentences. | `phylo.py:139-158, 244-286` |
| Canon tiers | Core = 29+ of 32 works, common = 23 to 28, optional = 6 to 22; Batman-specific = present in at most 3 of 13 cousins. | `canon.py:23-35` |
| Name-free text | An LLM rewrite is used only to locate names; a word diff replaces each changed span with a code X1, X2... | `make_neutral.py` |

Facts from the committed outputs that the docstrings do not make obvious:

- **Families are almost all singletons.** Qwen3-8B: 14,843 families from 18,485 candidate units, 94.9% of size 1, largest 167. Llama: 24,732 families, 96.5% singletons. Qwen3-1.7B: 5,641, 94.5%. (Computed from `study/results/*/memes.json.gz`.) "Meme family" in the paper therefore means "single neuron pole" in 19 cases out of 20.
- **Memes sit at every depth.** Qwen3-8B families by layer quartile: 24.6%, 28.5%, 14.9%, 31.9% (same file). A quarter are in the first 9 layers, where token and word detectors are the expected content. No analysis in the repo splits results by layer.
- **`n_baseline` carries no information.** The bar is the second-highest film, so exactly one film is "present" for every unit by construction: 14,549 of 14,843 Qwen families have `n_baseline` = 1. The other 294 have 0, which can only happen when the top two films tie, an artefact of the int8 rounding (same file).
- **Stale strings.** `discover.py`'s docstring and the `threshold_rule` field written into `memes.json` say "beats 69 of 70 baseline films" and "33 Batman works". The corpus has 69 films and 32 Batman works (`study/data/corpus.jsonl`: 32 batman, 13 cousin, 69 baseline, 3,700 sentences). The chance rate used, 2/70, is right.

### 2. What the trees and the first tests show (committed numbers)

All from `study/results/phylo.json.gz`:

- Source recovery by tree distance: selected memes rank the documented source 5.44 of 12 (Qwen) and 5.33 (Llama), chance about 5.8. **Random active non-meme units do better: 4.22 and 3.40. TF-IDF does best: 2.89 (p = 0.004).**
- Era clustering: ratio 1.030 (p = 0.032) for Qwen memes, 1.021 for random units, 1.036 (p = 0.0004) for TF-IDF.
- Meme distance against TF-IDF distance: Mantel r = 0.84 (Qwen), 0.85 (Llama). The neuron tree is largely the word tree.
- Cross-model twins, all families: median best-partner r 0.402 for memes, 0.359 for random active units, 0.136 for shuffled sentences (Llama as partner). The headline 0.82 applies to the 33 strongest only (`study/README.md`), and the README itself says 24 of those 33 are name or place detectors.

From `study/results/experiments.json`:

- E1: canon share against hero-film cousins, AUC 0.515 (CI 0.28 to 0.75); sentence count alone 0.756; TF-IDF centroid 0.801.
- E2: share of comics beating the source 0.164 (Qwen) and 0.102 (Llama), exact p 0.0026 and 0.00017; TF-IDF cosine 0.110, word presence 0.167. Residual on both word measures: p = 0.089 (Qwen), 0.010 (Llama). Sequel control on name-free texts: 4 of 7 (Qwen, fails the bar of 5), 5 of 7 (Llama).

From `study/results/wording.json`: Qwen's "revenge" unit was chosen from 50 candidates with dev AUC 0.965 and scored test AUC 0.451 (4 of 12 minimal pairs). A unit picked on a small dev set did not generalise.

From `study/results/robustness/label_test.json`: blind two-choice label test 56 of 71 (79%); the word-stem overlap judge scores 63% on the 31 named labels and 66% on the 40 others.

### 3. Compute cost and hardware assumptions

- Hardware: Apple M5 Pro MacBook, 24 GB (`study/README.md`, "Reproduce"). MLX 0.32.4 and mlx-lm 0.32.2 (`requirements.txt`). `run_pipeline.sh` has a `#!/bin/zsh` shebang. Extraction needs Apple silicon as written.
- Speed: Qwen3-8B, 24 sentences and 483 tokens, forward pass plus aggregation in 0.354 to 0.358 s, about 68 sentences per second, peak 5.65 GB (`preregistration/sources/neuron_timing_qwen3-8b.json`). Llama: 66 sentences per second, 5.6 GB (`neuron_timing_llama3.1-8b.json`). Three works, 2,342 tokens: 2.36 s, peak 6.6 GB (`devcheck_cost_neurons_qwen3-8b.json`).
- LLM reading on the same model: 0.42 s per meme question uncached, 0.12 s cached (`preregistration/sources/llm_timing_qwen3-8b.json`). So one neuron pass, which yields all 884,736 units, costs about one uncached LLM question. The neuron method's cost advantage only appears when many memes are asked of each story.
- Storage: int8 arrays of 3,700 x 442,368 bytes per pole are 1.64 GB, so 3.3 GB per Qwen3-8B run; all five runs (three models on original text, two on name-free text) come to about 14.6 GB by my arithmetic from the code's array shapes, matching the paper's "about 15 GB ... roughly 30 minutes on a laptop" (`study/paper/paper.md` line 302). The compact extractor stores 1.77 MB per work instead (`aggregate_bytes_fp16` in the timing file).
- Fragile points: `tokenizer._tokenizer` is a private attribute of the mlx-lm wrapper; the model wrappers assume `layer.mlp.gate_proj / up_proj / down_proj`, so any non-SwiGLU or mixture-of-experts model needs new code; `discover.py` builds a dense candidates x candidates matrix (about 3 GB for Llama's 27,287 candidates).
- Porting is cheap in principle: the recorded tensor is exactly the input to `down_proj`, so a forward pre-hook on `down_proj` in PyTorch captures the same site. But MLX 4-bit weights are not the same numbers as any other quantisation, so results would need a reliability check, not an assumption of equality.

### 4. What Cammarata actually claims, against what the code does

Source: Cammarata, "Neural Networks Are Already Clean", Goodfire, 2026, labelled "an informal research essay and working hypothesis", with a technical companion paper promised and not yet out. One passage was revised on 30 October 2026 after the "V-shape neuron" claim was judged "too strong".

| | Cammarata | This study's code |
| --- | --- | --- |
| Status of the claim | "Preliminary audits of Qwen3-32B": roughly 90% of neuron poles have one main job, about 7% two related jobs, 2 to 4% not yet understood. He says the number depends on thresholds and should be read as "a broad pointer". | Taken as the premise for a measuring instrument. |
| Models | Qwen3-32B (dense), Qwen3.6-35B-A3B (mixture of experts), Kimi K3 ("trillions of parameters"). Says small models have broader neurons: "a network short on room ... its neurons just get broader". | Qwen3-8B, Llama 3.1 8B, Qwen3-1.7B, all 4-bit. The paper admits this (paper.md line 254). |
| Unit | The neuron pole, positive and negative sides both important ("think of negative poles as gates in a circuit, not a neuron's opposite"). | Same idea: max and min as separate units. This part is faithful. |
| SwiGLU sign regimes | When gate and value are both negative the product is positive: "for about 4% of neurons, it's the whole top of the list". He separates the four sign combinations. | Records only the product. A positive-pole "meme" can be a double-negative echo. Not checked. |
| Which activations count | Strong ones. A "strong response is 30% or more" of the neuron's strongest example in a 10-billion-token corpus. Weak activations broaden into related territory and "we don't yet have a semantic test for weak activations that we trust". | Presence is relative to 69 films. Nothing ties the bar to the neuron's own range, so presence can sit in the weak regime he says is not understood. |
| Where neurons fire | Often on commas and other punctuation that hold "a running summary" of context; "most humans and agents expect neurons to detect nouns by firing on them, which they mostly don't". | Max over a sentence's tokens catches punctuation, which is fine. But the running-summary motif plus whole-work context means a sentence's value reflects the document so far, not that sentence. Sentence profiles, family merging and twin correlations all inherit this. |
| How a meaning is established | Agents run conjecture and test: steering, ablation, upstream and downstream circuits, blind maximum, and new examples against near misses and far misses (183 neurons: new examples fire strongly 76% of the time, near misses 11%, far concepts about 1%). He says standard top-k labelling mislabels (the "polysemantic comma neuron" that was a Colorado neuron; "capital cities" labels that miss a Santa negative side). | Top 8 sentences from a 3,700-sentence, 114-work corpus, read by one AI and checked by a second; a blind two-choice test at 79% against 65% for word overlap. This is the top-k method he criticises, on a corpus about five orders of magnitude smaller (roughly 100,000 tokens, my estimate from 26 tokens per sentence in `preregistration/sources/devcheck_cost_neurons_qwen3-8b.json`, against his 10 billion). |
| Cross-model | Nothing for language models. One vision remark: "the same neuron shows up in DINOv2 and in our large and medium ConvNeXts". | Best-partner correlation across two 8B models. P2 gets no support from this essay. |
| Culture, memes | Not mentioned. The essay is about legibility of mechanisms for interpretability and safety. | The step from "legible mechanism" to "unit of cultural inheritance" is the study's own. |

The essay supports "many neuron poles in large models have a describable main job when studied with interventions". It does not claim that thresholded neuron activations on a small corpus are a better document feature than words, and the study's results say they are not, on this corpus.

### 5. Design choices that could change the results

Each is a fork the code took once. The E2 multiverse (`study/robustness/e2_multiverse.py`) varies only the top-k (1, 3, 5), the bar, the band, hub adjustment, time rule, source rule and de-duplication.

1. **Recording site.** SwiGLU product. Alternatives never tried: gate alone, value alone, sign-regime split, product times output-weight norm, residual-stream directions, attention outputs, SAE or transcoder latents.
2. **Token pooling.** Max and min over tokens. Extreme values grow with sentence length. Mean, last-token or punctuation-token pooling would behave differently.
3. **Context.** Whole work in one pass, capped at 40 sentences (`fetch_corpus.py:89`), raw text into chat-tuned models. Per-sentence, windowed or prefixed contexts exist only in the unrun `extract_compact.py` (`--prefix`, `context_sentences`). The wording test runs sentences alone and compares them with percentiles from in-document sentences (`wording_test.py` docstring), a mismatch.
4. **Work pooling.** Mean of top 3 of n sentences, with n from 9 to 40. More sentences means more draws and a higher top-3 mean. Sentences per work: Batman 13 / 34 / 40 (min / median / max), cousins 9 / 30 / 40, films 20 / 33 / 40 (`study/data/corpus.jsonl`).
5. **Presence bar.** Rank against 69 films with no superheroes. Change the reference set and the memes change; the bar measures "unusual for an ordinary film", which favours genre vocabulary.
6. **Binary presence, equal weights.** Jaccard over all families gives 14,593 optional singletons the same weight each as the 56 core units.
7. **MIN_WORKS = 6, MERGE_CORR = 0.5, average linkage.** Not varied for discovery or trees.
8. **Quantisation, twice.** 4-bit weights (never compared with 8-bit or fp16) and int8 activations with a max-based scale (one outlier coarsens the whole neuron; ties appear, see the 294 families above).
9. **Model size and family.** 8B and 1.7B; no 32B run, no base (non-instruct) model, no second checkpoint of the same architecture.
10. **Tokenisation.** `i > 0` drops the first token. Llama adds a begin-of-sequence token; I believe Qwen's tokenizer adds none, so for Qwen the first real token of each work is dropped instead [UNVERIFIED: read from code, tokenizer not run]. Qwen and Llama also split names differently, which affects any name-detector twin.
11. **Name-free codes.** X1, X2 are out-of-distribution tokens that may drive their own "placeholder" neurons in every work equally; the paper notes the diff "sometimes swallowed a neighbouring word" and that the loss was not measured by group (paper.md line 254).
12. **Selection and evaluation on the same sentences.** Twins are chosen as the maximum over about 917,000 candidates and scored on the same 3,700 sentences. The shuffled null (median 0.136) shows the size of that max-selection floor; no held-out sentence split is used.
13. **E2 extras.** Frequency band 2 to 16 and hub z-scoring were designed after a simpler test failed (`study/README.md`).
14. **Shared random generators.** `RNG = default_rng(0)` in `phylo.py` and `default_rng(7)` in `experiments.py` are drawn from in sequence, so p-values depend on call order. Minor.

### 6. Missing pieces

- **No causal intervention.** Confirmed by reading every listed file: no code changes an activation. The paper says labels were "never validated by steering, the stronger test" (paper.md line 254).
- **No SAE or transcoder comparison.** Public SAEs exist for one of the study's model families (Llama Scope, arXiv:2410.20526, trained on Llama-3.1-8B). Not used. Whether they transfer to the 4-bit Instruct checkpoint is untested here.
- **No residual-stream or probe rival in the study.** `extract_compact.py --hidden` (mean residual at four depths) and `readers.py embed` (five encoders) exist but have never been run on test data.
- **No LLM-reading rival in the study.** `readers.py` has the judge; unrun.
- **Length and tokenisation confounds** are measured once (length-only AUC 0.756, E1) and controlled once (E3 covariates), not removed from presence itself.
- **Summaries, not primary texts.** One 2020s Wikipedia plot summary per work, at most 40 sentences. No test-retest across independent summaries of the same work, so the reliability of "presence" is unknown.
- **No general-corpus calibration.** A unit's activation is never compared with its distribution on ordinary text, so strong and weak regimes cannot be told apart.
- **No double-negative check, no layer breakdown, no precision check, no larger model.**
- **No second franchise.** Every threshold was set on Batman.

### 7. Infrastructure new experiments can reuse

- `study/extract.py` `Recorder` and `sentence_spans`: the hook is already in the forward path. Multiplying selected channels of `h` before `down_proj` turns it into a steering or ablation hook in a few lines.
- `preregistration/extract_compact.py`: resumable, sha256-guarded, pinned Hugging Face snapshots, GPU-side top-3 aggregation, `--units` (sentence-level subset), `--full`, `--hidden` (residual taps at 25, 50, 75, 100% depth), `--prefix` and `context_sentences` (context control), cost and peak-memory logging.
- `preregistration/readers.py`: next-token yes/no judge with prompt caching in three variants, recursive summaries, adaptation-pair judge, likeness and "recognise the franchise" (a contamination check), naive and evasion-aware keyword analysts with one revision round, Chinese keywords and machine translation, zero-shot NLI (DeBERTa v3), five embedding encoders at sentence, 3-sentence and document level, a cross-family check reader (Gemma 2 9B). Every command writes a cost file.
- `preregistration/bench.py`: fixed-condition cost benchmark for every method. `analysis.py` with `test_analysis.py` (151 checks on synthetic data, per `preregistration/README.md`).
- `study/experiments.py`: exact enumeration of relabellings (`source_maps`), `residual_test`, hub adjustment, split-half presence bar (`film_splits`), `e3_power`, lexical comparators (`tfidf`, `lexical_reps`).
- `study/robustness/`: `e2_multiverse.py` (30,240 specifications), `e2_reimpl.py` (independent rebuild), `nulls.py` (word-matched, sparsity-matched and layer-matched unit nulls; pseudo-franchises).
- `study/phylo.py`: Jaccard, NJ, Mantel, era test, chunked cross-model `best_partner`.
- `study/make_neutral.py`: names to codes by diff against an LLM rewrite.
- Data: `study/data/corpus.jsonl` (114 works, 3,700 sentences), `corpus_neutral.jsonl`, `pageviews.json`, labels for 125 distinct units (31 in `results/labels.json`, 120 in `results/labels_wf.json`, 26 in both), the 71 blind label-test items.

## What this means for P1 and P2

**P1 (a meme can be measured by which neurons fire).** As coded, P1 is tested in a weak form and the evidence is weak.

- The instrument detects something real: Batman summaries share units that random 32-film sets do not (`study/README.md`), and adaptations carry their sources' content on name-free text (E2).
- On every test where a word baseline was run, words matched or beat the neurons: trees (2.89 against 5.44), E1 (0.80 against 0.52), E2 (0.11 and 0.17 against 0.16 and 0.10), sequel control (6 of 7 against 4 of 7, per `study/README.md`). Only Llama's residual test favours neurons (p = 0.010).
- Selected memes do worse than random active units on source recovery. That cuts against the idea that the discovery rule finds the informative units.
- The form of P1 that Cammarata's essay could support (a neuron pole, validated by intervention in a large model, as a legible unit) has not been tested here at all. The conditions he states for cleanliness (large models, strong activations, sign regimes separated, agentic conjecture and test) are each violated or unchecked.
- So the honest reading is: P1 is neither confirmed nor refuted. The current operationalisation is closer to "sparse lexical-plus-context features from a small model" than to "clean neurons".

**P2 (the measurement generalises across models).** The code's evidence is thin and mostly explained by words.

- Across all families the twin correlation is 0.40 against 0.36 for random active units. The gap is small. The strong twins are name detectors, and the README reports that with names removed the same pairs fall to r = 0.14.
- Tree agreement of 0.98 between models sits beside 0.90 to 0.93 for random non-meme units (`study/README.md`), so most of it is the texts, not shared memes.
- The method is one-to-one neuron matching chosen and scored on the same 3,700 sentences. Gurnee et al. (arXiv:2401.12181) found only 1 to 5% of GPT-2 neurons universal even across seeds of the same architecture, using 100 million tokens. One-to-one matching across different architectures is the hardest version of P2; a learned linear map between neuron spaces (the form the Platonic and universal-geometry papers actually support) is not implemented.
- Cammarata's essay offers no language-model universality claim.

## Experiment seeds

1. **Steer the named memes.**
   - Question: does turning a labelled unit up make the model write the labelled idea, as the essay's standard requires?
   - Data: the 31 named units (`study/results/labels.json`), neutral story prompts.
   - Method: add a multiplier to chosen channels in `Recorder`; generate; blind LLM and human judges pick which of two continuations is "on label".
   - Rival to beat: (a) steering a word-matched and a random active unit; (b) a difference-of-means residual direction built from sentences picked by TF-IDF keyword match for the label; (c) an LLM simply told to include the idea (ceiling).
   - Falsified if: on-label rate for meme units is no higher than for random active units, or the TF-IDF-derived direction steers as well.
   - Cost: about a day of code, a few hours of laptop time (my estimate).

2. **Recording-site bake-off.**
   - Question: is the SwiGLU product the right place to read, or do sign-split neurons, output-weighted neurons, residual means or public SAE latents do better?
   - Data: name-free corpus for E2; the planted-meme stories once they exist.
   - Method: extend `extract_compact.py` (gate sign mask, down_proj column norms, `--hidden`); Llama Scope latents for the Llama arm; same E2 statistic and residual test for each.
   - Rival: TF-IDF cosine, word presence, e5 and Qwen3-Embedding sentence embeddings, LLM judge.
   - Falsified if: no site passes the residual-beyond-words test in both models.
   - Cost: 2 to 3 days; extraction minutes per site (my estimate).

3. **Test-retest on independent retellings.**
   - Question: is "meme present in work W" a property of the work or of one summary?
   - Data: two or more independent summaries per work (other wikis, other language editions translated, the "Tell me again!" dataset cited in the paper [UNVERIFIED by me this session]).
   - Method: presence kappa between summaries of the same work, against kappa between different works.
   - Rival: word-presence kappa, embedding cosine, LLM judge agreement on the same pairs.
   - Falsified if: neuron presence is no more stable across retellings than word presence.
   - Cost: 1 to 2 days, mostly collecting texts.

4. **Length and context invariance.**
   - Question: how much of presence is sentence count, sentence length and document position?
   - Data: existing corpus; needs activations regenerated (about 30 minutes, per paper.md).
   - Method: recompute presence with a fixed number of sentences per work, with per-sentence passes (`--prefix`), and with shuffled sentence order; report unit-level kappa against the published presence and rerun E2.
   - Rival: sentence count alone (0.756 on E1), TF-IDF under the same subsampling.
   - Falsified if: kappa under length matching or isolated-sentence context falls below the cross-model kappa (0.73, paper.md line 112), meaning the measurement is less stable under formatting than across models.
   - Cost: 1 day.

5. **Strong-regime presence.**
   - Question: do the findings survive when a unit only counts as present in its own strong regime?
   - Data: a general corpus of a few million tokens to get each unit's range.
   - Method: presence = at least 30% of the unit's own corpus maximum (the essay's criterion), with sign regimes separated; rerun discovery, canon, E2.
   - Rival: TF-IDF, and the current film-relative bar.
   - Falsified if: fewer than about 10% of current families survive, or E2 disappears; either would mean the study's memes live in the weak regime.
   - Cost: 1 to 2 days; a few hours of extraction (my estimate).

6. **Precision and size ladder.**
   - Question: is a meme stable across 4-bit, 8-bit and full precision, and from 1.7B to 8B to a larger model?
   - Data: existing corpus.
   - Method: same pipeline at each precision; unit-level presence kappa (same indices across precisions); for size, the learned-map approach below.
   - Rival: TF-IDF is perfectly stable across "models", so the bar is the cross-model kappa.
   - Falsified if: 4-bit against 8-bit kappa is below 0.9, which would make quantisation a first-order confound.
   - Cost: hours; a 32B model at 4 bits will not fit beside its activations in 24 GB (my estimate).

7. **Learned maps instead of twins (for P2).**
   - Question: does a sparse linear map from Qwen units to Llama units, fitted on half the sentences, predict held-out sentences better than one-to-one twins?
   - Data: existing activations, split by work.
   - Method: ridge or sparse regression per target unit; held-out r; repeat on name-free text.
   - Rival: a map fitted from TF-IDF features to the same target units, and a sentence-embedding map.
   - Falsified if: held-out r on name-free text is no better than the TF-IDF-feature map.
   - Cost: 1 day, no new extraction.

## Key sources

- Cammarata N. 2026. Neural Networks Are Already Clean. Goodfire. https://already-clean.goodfire.com/ . The source of P1; read in full this session. A working hypothesis with stated caveats on thresholds, weak activations, model size and SwiGLU regimes. Priority A.
- Gurnee W. et al. 2024. Universal Neurons in GPT2 Language Models. arXiv:2401.12181. 1 to 5% of neurons universal across seeds (abstract confirmed). The base rate for one-to-one twins. Priority A.
- Bolukbasi T. et al. 2021. An Interpretability Illusion for BERT. arXiv:2104.07143. Top-activating examples from a narrow corpus make neurons look like one clean concept. Directly relevant to labels from 3,700 sentences. Priority A.
- He Z. et al. 2024. Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders. arXiv:2410.20526. A ready SAE comparator for the Llama arm. Priority A.
- Dunefsky J., Chlenski P., Nanda N. 2024. Transcoders Find Interpretable LLM Feature Circuits. arXiv:2406.11944. The transcoder alternative to raw MLP neurons. Priority B.
- Paulo G., Belrose N. 2025. Sparse Autoencoders Trained on the Same Data Learn Different Features. arXiv:2501.16615. Only 30% of latents shared across seeds in a Llama 3 8B SAE; SAE units are not a stable alternative either. Priority B.
- Leask P. et al. 2025. Sparse Autoencoders Do Not Find Canonical Units of Analysis. arXiv:2502.04878. Same point about SAE units. Priority B.
- Heap T. et al. 2025. Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers. arXiv:2501.17727. Auto-interpretability scores can be high without real features; a caution for AI-written labels. Priority B.
- Movva R. et al. 2025. Sparse Autoencoders for Hypothesis Generation. arXiv:2502.04382. Already cited in the paper; SAE features as text-analysis variables with baselines. Priority B.
- Huh M. et al. 2024. The Platonic Representation Hypothesis. arXiv:2405.07987; Jha R. et al. 2025. Harnessing the Universal Geometry of Embeddings. arXiv:2505.12540. Both concern representation geometry and learned maps, not individual neurons. Priority B.
- Yang A. et al. 2025. Qwen3 Technical Report. arXiv:2505.09388. The primary model. Priority C.
- Elhage N. et al. 2022. Toy Models of Superposition. arXiv:2209.10652. The position the essay argues against. Priority C.
- Gemma Scope, Lieberum T. et al. 2024. arXiv:2408.05147. Open SAEs for Gemma 2, the family of the prereg's check reader. Priority C.

## Fog

- I did not run a tokenizer, so the claim that Qwen's first real token is dropped by `i > 0` is an inference from the code. [UNVERIFIED]
- I could not check whether byte-level offsets put the leading space inside a token's span for these tokenizers; if they do, a token at a sentence boundary may be counted in two sentences.
- How labels were actually produced is not in any script. The paper says an AI reader wrote them and a second checked them; the prompts are not in the files I read.
- The essay's numbers (90%, 4%, 76% against 11%) are its own, from "preliminary audits" and "relatively small samples"; the companion paper is not published, so none can be checked.
- Whether Llama Scope SAEs (trained on the base model) transfer to the 4-bit Instruct checkpoint is unknown to me.
- I did not read `preregistration/analysis.py` (5,254 lines), `build_lists.py`, or the robustness scripts beyond their function lists.
- Whether MLX now runs off Apple silicon was not checked; the README says it does not.
- Total token count of the corpus and wall-clock per model are not in committed files; only the paper's "roughly 30 minutes" and the small timing files.
- I did not confirm the "Tell me again!" dataset this session; it is reference 23 in `study/paper/paper.md`.
