# LI-27: Gradient similarity as a source tracer after deep rewording

**Area:** Lineage · **Status:** design registered 10 November 2026, not run · **Priority:** 2.8 of 5

## Question

On stories whose true source is fixed by construction and whose wording is removed, does the similarity of two texts' loss gradients (the direction a model's weights would move if it learned each text) name the source better than word overlap, sentence embeddings, an LLM judge, or the drop in the model's surprise when the candidate is in its context?

## Prediction

On RX-27's 200 held-out deep-transform stories (new setting, names and wording; 20 candidate plots each, chance 5%), gradient similarity in Qwen3-8B names the true source first at least 10 percentage points more often than the best rival, and at least 10 points more often than the same score from three copies with scrambled weights (each matrix's entries shuffled).

## What would refute it

The gradient-minus-best-rival gap has a 95% interval (resampling stories, best rival re-picked each time) that includes zero, or the lead over the scrambled copies' average is under 10 points. In simulation, 200 stories catch a 15-point lead in 85 to 95% of runs and a 10-point lead in 40 to 70%. LI-21's six sealed tales are reported, not tested.

## Data

No new texts. Test: RX-27's stories, not yet built, from 250 little-known film plots in the CMU Movie Summary Corpus (42,306 plots, CC BY-SA): 50 development sources, 200 held out, 20 candidate plots each. Later, descriptive only: LI-21's 768 retellings of Folktexts tales (Zenodo 6575263, CC BY-SA 4.0). Readers: Qwen3-8B and Llama 3.1 8B (4-bit compressed weights), Qwen3-1.7B (full precision).

## Method

A text's gradient: how the weights would move if the model learned it, read through added matrices (adapters) in components chosen from a short menu on development stories; word-embedding layers excluded. Gradients standardised on development texts; score: how closely two align (cosine). Settings committed before held-out stories are scored. Statistic: true source ranked first. Null: source labels permuted within genre.

## Cheap rival it must beat

Word overlap (BM25), longest shared word span, surprise-weighted word overlap, sentence embeddings (e5, story-emb), RX-27's LLM judge, text length, averaged internal state, RX-27's neuron units, the scrambled-weight gradient score, and the in-context surprisal drop. Most dangerous: the surprisal drop, a plagiarism detector run on Qwen3-8B and Llama 3.1 8B (Guo et al. 2026). If it wins, a forward pass suffices.

## Premise it tests

P1, indirectly: the score uses no units. A pass where neuron units fail means descent is in the model but single neurons do not expose it. A fail is graded: beating word rivals and scrambled copies but not the surprisal drop means a forward pass reads descent better; beating neither means gradients record words. P2: Qwen and Llama errors compared.

## Biology analogue

DNA–DNA hybridisation: whole-genome similarity with no named genes.

## Cost

M: about a week on one 24 GB laptop once RX-27's stories exist: two to three days writing gradient code the repository lacks, then about three days of passes (roughly 29,000 gradient, 26,000 forward). Day-one check: if 8B gradients do not fit in memory, Qwen3-1.7B at full precision becomes the registered reader. No paid services.

## Payoff

The first gradient-score accuracy with cheap rivals in the same test and neither text in the model's training data. The nearest earlier test (Hinterleitner et al. 2026) had neither: among five candidates it found a paraphrase's original 99% of the time, and 22% (chance 20%) on newly generated text. Also three internal readouts ranked against a forward pass.

## Main risk

Most likely the surprisal drop, story embeddings or the LLM judge win: BM25 beat gradient attribution at fact tracing (Akyürek et al. 2022). Gradients may mostly record rare words, and plots the reader has memorised give weak gradients. This is the plainest gradient score, so a fail does not close corrected variants. Stories are model-written, and nothing runs before RX-27.

## Reviewer note

Remains: a win is unlikely (the nearest earlier test fell to chance on newly generated text) and a loss by the plainest gradient score closes little. Changed: test moved from LI-21's six tales to RX-27's 200 independent stories; power restated; scrambled-weight control; settings frozen on development sources; earlier test cited.

## Cross-area ranking

Registered on 10 November, after the three cross-area judges had ranked the earlier designs, so it was not ranked by them. Its reviewer's answer to whether it should run first: no: because it tests no unit, moves neither premise directly and will most likely lose to the surprisal drop. Add that rival to RX-27 at step 5 instead (forward passes only), and run this arm after RX-27.

## Overlaps with

An added arm on RX-27's stories and, later, LI-21's trees; it builds no texts. Its surprisal-drop rival should be added to RX-27 and LI-21 whether or not this runs. US-23's re-set drafts could be scored the same way. Unlike LI-21, LI-22 and LI-24, it reads no neuron units. LI-25 and LI-26 were taken, hence LI-27.

## Note

Prompted by the field survey (arXiv:2611.08438). Data check on 10 November 2026: Checked by the reviewer on 10 November 2026. No model was run; the only computation was a power simulation on made-up right and wrong calls. Data and models (fetched again, independently of the drafter): - Zenodo API, record 6575263: "j-hagedorn/trilogy: Updated Ashliman Folktexts dataset", v1.1, 24 May 2022, one zip of 10.8 MB, open access, licence field "other-open". The data paper (doi 10.5334/johd.78) says the dataset is distributed under CC BY-SA 4.0 and holds 1,518 tales. - CMU Movie Summary Corpus page: 42,306 plot summaries; "All data is released under a Creative Commons Attribution-ShareAlike License" (no version given). - Hugging Face API: Qwen/Qwen3-8B and Qwen/Qwen3-1.7B (Apache-2.0); meta-llama/Llama-3.1-8B-Instruct (llama3.1 licence, gated); the three mlx-community 4-bit builds exist. (Cut for length.)

## Sources

- Barez et al. 2026 (survey, section 11; TracIn, cited there as Garima et al. 2020)
- Hinterleitner et al. 2026 (gradient retrieval of paraphrase originals; arXiv 2601.16651)
- Akyürek et al. 2022 (fact tracing; arXiv 2205.11482)
- Guo et al. 2026 (source-conditioned description-length gain; arXiv 2608.03859)
