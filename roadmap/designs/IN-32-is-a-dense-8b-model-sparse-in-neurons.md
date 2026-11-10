# IN-32: Is a dense 8B model sparse in neurons?

**Area:** The instrument · **Status:** design registered 10 November 2026, not run · **Priority:** 2.8 of 5

## Question

In Qwen3-8B and Llama 3.1 8B with four-bit weights, how small a share of each layer's neuron activations can be kept per token before next-word loss (the model's error in predicting the real next word) rises 5%, and at equal small counts do neurons carry more of a layer's effect than principal components (main directions of variation)?

## Prediction

The paper's sparsity was measured on a model that sends each token to 8 of 256 blocks of neurons per layer. It does not carry over to dense models, which use every neuron on every token. In both models, keeping each token's strongest 10% raises loss by 5% or more, and holding the rise under 5% takes a quarter or more. At 128 units in one layer, neurons keep more of the layer's effect than random mixtures of neurons, but not 0.05 more than principal components.

## What would refute it

In either model, on 1 million held-out tokens: a kept share of 15% or less holds the rise under 5% (the paper kept 7.8% of routed activations for a 2.9% rise; under 5% at 10% here means its result carries over outright); or, averaged over four layers at 128 units, neurons beat principal components by 0.05 or more, or fail to beat random mixtures. Power: about 1,500 documents; expected sampling error 0.15 points, unconfirmed.

## Data

Development, for percentiles, means and bases: 2 million tokens from FineWeb's 10-billion-token sample (web text, ODC-By) and 300 plots from the CMU Movie Summary Corpus (42,306 Wikipedia film plots, CC BY-SA). Test: 500,000 tokens from an unused FineWeb file and 500,000 from about 800 other CMU plots, excluding study and pre-registration titles. Nothing is fitted on test text.

## Method

Whole model: per layer and token, keep the strongest share of the study's MLP neuron activations (seven fixed shares, 50% to 5%), or only values beyond each neuron's own percentile band. One layer at a time: keep the top 8 to 512 units per basis; a layer's effect is the damage from removing it. Percentage loss rise, bootstrapped over documents.

## Cheap rival it must beat

Words cannot matter: every arm reads the same text and the outcome is the model's own loss. Controls at 10%: a random share kept; a fixed set of neurons kept; the strongest removed against a random share removed. Bases at equal count: random mixtures of neurons, the layer output's own coordinates, and its principal components, the most dangerous.

## Premise it tests

P1, indirectly. If 10% suffices, the paper's strong-band argument reaches the study's models. If not, the band amendments of 10 November rest on readability alone (IN-23) and the paper's 92% must not be quoted for these models. If principal components match neurons, neurons are not a privileged sparse code here. No outcome shows that a firing names an idea.

## Biology analogue

Minimal genome: delete what the cell survives without, against random deletions.

## Cost

S. Per model: about 20 whole-model arms of 1 million tokens and 70 single-layer arms of 100,000, forward passes only; one to two days for both models at the speed measured in the pre-registration (about 1,000 tokens a second), which must be re-timed. No activations are stored, so 8-bit repeats fit the laptop. Optional and beyond S: the paper's own model as a positive control.

## Payoff

The loss rise at every kept share under the paper's rule; whether each neuron's own top and bottom 3% is enough; damage per value removed, band by band; the share of kept firings made by two negative parts multiplying (each neuron's output is a gate times a value); and whether neurons or principal components carry a layer more sparsely. Yardstick: the loss gap between 4-bit and 8-bit builds.

## Main risk

The share result is largely foreseeable for Llama. Liu et al. 2024 report that removing 46% of Llama 3 8B's MLP hidden state, by a cruder rule, took perplexity from 5.87 to 6.78 (about 8% in loss, our conversion), and that 90% removed model-wide destroyed it. New here: Qwen, four-bit weights, the band profile and the basis test.

## Reviewer note

Remaining: the share result is mostly foreseeable from Liu et al. on Llama 3 8B, and the basis test has no trained dictionary. Changed: the prediction now says the paper's sparsity will not carry over; basis test moved to single layers at small counts; band arm added; GPU day dropped.

## Cross-area ranking

Registered on 10 November, after the three cross-area judges had ranked the earlier designs, so it was not ranked by them. Its reviewer's answer to whether it should run first: no: because Liu et al. 2024 already make the main result foreseeable for Llama 3 8B, and neither outcome changes steps 1 to 5. Run it beside IN-23, which needs the same web-text pass and percentiles.

## Overlaps with

IN-22: its amended top-8 test is this design's single-layer arm; run it once, here, and cite it there. IN-23: this tests the band its amended rule assumes, and can share one web-text pass. IN-29, UN-24: same builds; adds loss under removal at 4-bit and 8-bit. Outside the registry: Liu et al. 2024 already cover Llama 3 8B at full precision.

## Note

Prompted by Goodfire's companion paper. Data check on 10 November 2026: Checked 10 November 2026 by the drafter and again by the reviewer; nothing was run on any model. FineWeb (Hugging Face API and dataset server): HuggingFaceFW/fineweb is public, licence odc-by. Its sample-10BT config has 15 parquet files, 30.6 GB, 14,868,862 documents and a token_count field. That is about 670 tokens a document (by arithmetic), so 500,000 tokens is roughly 750 documents. CMU Movie Summary Corpus: the README says "All data is released under a Creative Commons Attribution-ShareAlike License" (no version given); plots are from the 2 November 2012 Wikipedia dump. The reviewer recounted the 48 MB archive: 42,306 summaries, 13,145,251 words (mean 311, median 187). The 24,134 summaries of 150 words or more hold 11.7 million words (mean 485).

## Sources

- Tigges et al. 2026 (Goodfire, Neurons as an Interpretable Basis), Table 1 and Figure 19
- Liu et al. 2024 (TEAL, arXiv:2408.14690): Tables 1 and 4, which rerun Lee et al. 2024 (CATS), and Table 10
- FineWeb sample-10BT (Hugging Face, ODC-By)
- CMU Movie Summary Corpus (Bamman, O'Connor and Smith 2013)
