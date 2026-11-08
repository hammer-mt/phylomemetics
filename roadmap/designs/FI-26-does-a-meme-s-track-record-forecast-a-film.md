# FI-26: Does a meme's track record forecast a film?

**Area:** Fitness and forecasting · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

After metadata, can a new film's box office be forecast from how earlier films carrying the same units, and the same pairs of units, performed?

## Prediction

For films released from 2005 to 2012, unit and pair track records built only from earlier films add 0.01 or more to out-of-sample R-squared (share of variance explained) over metadata plus the embedding nearest-neighbour rival, and pair novelty shows an inverted U. We expect metadata to dominate and no gain.

## What would refute it

A gain under 0.01, or an interval that includes zero; or pair terms that add nothing over single units; or a gain confined to films an LLM can name from the masked plot (memorised fame). The 1,784 held-out films detect a correlation of 0.1 after controls with over 95% power; the 607 with a Wikidata budget give about 70%.

## Data

CMU Movie Summary Corpus (CC BY-SA, version not stated): 7,588 films have a plot and box office; with ten or more sentences, 3,881 before 2005 and 1,784 from 2005 to 2012 (recounted 8 November 2026). Wikidata (CC0) gives a budget for 2,133 of these; parse the rest from Wikipedia infoboxes. Names masked; plots cut to their first 10 sentences.

## Method

Outcome: log revenue minus what year, genre, runtime, country, sequel flag, summary length and budget (flagged where missing) predict. A unit's record: mean outcome of earlier films carrying it, shrunk toward zero when few; each pair gets a record and a novelty score (observed against expected co-occurrence). Units chosen on pre-2005 films. Null: outcomes shuffled within year and genre.

## Cheap rival it must beat

Metadata alone; the mean outcome of the 20 most similar earlier films by sentence embedding (most dangerous: a track record that needs no units); the same pipeline on rarity-weighted terms and on the corpus's own genre-tag pairs (363 tags, after Sreenivasan 2013); random units; an LLM guessing from the masked plot, contaminated because it has read which films were hits.

## Premise it tests

P1. A pass says fitness accumulates at the level of units and their combinations, so a work's promise can be read from its parts. A fail says success lives in metadata and whole-work similarity, and that memes are, for forecasting, no better than nearest neighbours.

## Biology analogue

Breeding value from marker records; gene interactions

## Cost

M. About 5,700 plots of 10 sentences (57,000 sentences, under an hour per model); one to two weeks for budget parsing, metadata joins, name masking and the forward-in-time fit.

## Payoff

A fitness table for units and pairs with honest out-of-sample value, and a test of the marketing rule that work 'just past the familiar' does best.

## Main risk

Metadata and budget take nearly all the variance (content features stayed under 0.05 R-squared for tweets). Units may only stand in for budget or genre. Hits get far longer summaries (median 33 sentences in the top revenue third against 12 in the bottom), and the models have read the future.

## Reviewer note

Recounted the corpus. Hits have summaries nearly three times longer, so plots are cut to 10 sentences and length joins the baseline. Wikidata budgets cover only 2,133 films; added a memorised-fame check and a genre-tag rival. Remaining weakness: metadata will dominate, and the models know which films were hits.

## Sources

- CMU Movie Summary Corpus (Bamman, O'Connor, Smith 2013)
- Uzzi et al. 2013
- Sreenivasan 2013
- Martin et al. 2016
