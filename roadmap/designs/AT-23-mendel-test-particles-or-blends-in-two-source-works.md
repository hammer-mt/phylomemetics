# AT-23: Mendel test: particles or blends in two-source works?

**Area:** The atlas · **Status:** design registered 8 November 2026, not run · **Priority:** 2.5 of 5

## Question

When a work has two documented sources that differ on a meme, does the work carry that meme at full parental strength or not at all, as particles would, or at an in-between strength, as blending would?

## Prediction

Of memes passed on from the one parent that has them, at most 15% arrive diluted: 25% to 75% of the way between the two parents' readings, beyond test-retest noise. A blind LLM reader's 0 to 4 ratings of the same motifs give a diluted share within 10 points of that.

## What would refute it

The diluted share for units is 30% or more, or the reader's ratings give a share 20 points higher than the units do, across 150 or more trios (bootstrap over trios; expected standard error about 4 points). If the method cannot tell built blends from built copies at AUC 0.75, the result is void.

## Data

Wikidata items with two 'based on' sources, all with English Wikipedia articles: 1,620 (counted 8 November 2026; CC0), among them 341 films, 124 television series, 96 video games; many others are not stories. Plots from Wikipedia (CC BY-SA 4.0); expect 150 to 250 usable trios. Noise from Tell Me Again: 96,831 summaries of 29,505 stories. 120 constructed crosses, CC BY.

## Method

Frozen AT-21 entries; names masked; equal-length summaries. For each unit present in one parent only, place the child's reading between the parents'. Fit two clusters at the parents (width from two summaries of one story) against a middle component. Nulls: children swapped for unrelated works matched on medium, decade, genre and length; crosses built to copy or fuse.

## Cheap rival it must beat

Same fit, with each work read by its three strongest sentences, on TF-IDF weights of one-parent words, sentence-embedding dimensions, the residual stream, random non-meme units, and an LLM reader's 0 to 4 ratings of motifs. The reader is the danger: if it sees blends the units miss, the particles are a thresholding artefact; if it agrees, neurons were not needed.

## Premise it tests

Assumes P1 for admitted units and tests whether they behave like genes. Particles support treating memes as countable and presence grids as sufficient. Blending supports Sperber's view that ideas are rebuilt, and means thresholded presence throws information away.

## Biology analogue

Mendel's crosses; particulate against blending inheritance

## Cost

M. About 600 plot sections fetched and name-masked by script, 120 constructed crosses, one extraction pass, a spot-check by hand; one to two weeks and under 50 US dollars.

## Payoff

A measured answer to the oldest objection to memetics, that ideas blend and so cannot be counted. It also says whether a set of memes per work is the right data type, or whether readings must be kept as amounts.

## Main risk

A plot summary either mentions an event or does not, so particles may reflect how editors summarise, not how authors combine sources. The constructed crosses test the method's resolution; they cannot test that. In 458 of the 1,620 items one source is based on the other; such chains are analysed apart, and crosses of unrelated stories are few.

## Reviewer note

Changed: the 1,620 count includes non-stories and 458 chains, now stated and split; all bases are pooled alike, because reading the strongest sentences makes units look particulate by construction; the reader's graded rating now decides. Remaining: one generation cannot separate blending from particles with partial dominance (see the two-generation seed).

## Overlaps with

Asks nearly the same question as CE-23, which the judges would run instead.

## Amendment, 10 November 2026

Do not read an in-between score as dilution without a check. The companion paper says a neuron's weaker firings tend to fall on related but different content. Class each child reading by the unit's own bands on general text, and have a blind reader judge the firing passage in every in-between case: same motif, neighbouring motif, or neither.

## Sources

- Wikidata property P144 (based on)
- Hatzel and Biemann 2024, Tell Me Again
- Claidière, Scott-Phillips and Sperber 2014, doi:10.1098/rstb.2013.0368
- Mesoudi and Whiten 2008, doi:10.1098/rstb.2008.0129
- Tigges et al. 2026, Neurons as an Interpretable Basis
