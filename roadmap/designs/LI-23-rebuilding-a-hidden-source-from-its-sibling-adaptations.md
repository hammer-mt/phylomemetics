# LI-23: Rebuilding a hidden source from its sibling adaptations

**Area:** Lineage · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

Using neuron memes (story units read from a language model's neurons), can the content of a common source be inferred from three or more of its film adaptations alone, and does the result match the surviving source better than a word-based reconstruction and the genre's typical story?

## Prediction

With names masked, the set of units shared by most sibling films picks the true source first among all other sources in the test pool at least 12 percentage points more often than the same consensus built from words, and at least 10 points more often than one sibling used alone.

## What would refute it

On at least 120 held-out sources, the neuron consensus is not above the word consensus (paired one-sided p at or above 0.05), or it beats a single sibling by under 5 points. 120 sources give about 80% power for a 12-point paired gap when methods disagree on 30% of sources. Not run with fewer.

## Data

Wikidata 'based on' links (CC0): 482 sources have three or more film adaptations with English Wikipedia articles throughout (recounted 2026-11-08). Many are characters or series with no single plot (Scooby-Doo, James Bond) and are excluded; early films often lack plot sections. Plots are CC BY-SA. Target: 40 development and at least 120 test sources with 15-sentence plots, counted before registration.

## Method

Treat the source as lost. Ancestral units are those in most siblings; derived units appear in one. Rank the true source among all other pooled sources, themselves much-adapted works. Test whether ancestral units occur in the real source more often than derived units and than units shared by unrelated same-genre films. Report separately where siblings are remakes of each other.

## Cheap rival it must beat

Word consensus (rarity-weighted words in most siblings), the average of sibling embedding vectors (a general model and story-emb), a local LLM writing the source's plot from the siblings, plot length, random non-meme units, the model's averaged internal state and sparse-autoencoder features. Most dangerous: the LLM, which knows these classics; results are split by fame. The unit-level test stands either way.

## Premise it tests

P1. If neuron units are heritable traits, those shared by siblings should be the source's own, beyond what the genre supplies. A pass supports the unit as a character for reconstruction; a fail says shared units reflect genre convention, not descent. P2 gets a side check: Qwen and Llama should reconstruct overlapping ancestors.

## Biology analogue

Ancestral sequence reconstruction checked against a preserved ancestor.

## Cost

S to M: under 2 days of model time on one laptop, but fetching and name-masking about 1,000 plots is the real work. Extraction under an hour per model; 160 LLM reconstructions.

## Payoff

A tested way to describe a lost or unknown source from its descendants, and a direct check of whether shared units mark common descent or only a shared genre.

## Main risk

Wikipedia editors reuse wording between a source's summary and its adaptations' summaries, so the word consensus wins (passages with copied eight-word runs are dropped); and later films copy earlier films, so the consensus may rebuild the film tradition, not the book.

## Reviewer note

Recount confirmed 482 sources, but the list includes characters and series, so the sample is cut to 120 with a not-run rule and a 12-point claim. Decoys are now other much-adapted sources. Remains: memorised classics, sibling films that copy each other, shared Wikipedia wording.

## Sources

- Wikidata query service counts, 2026-11-08
- da Silva and Tehrani 2016
- Fritz and Purvis 2010
- Claidière, Scott-Phillips and Sperber 2014
