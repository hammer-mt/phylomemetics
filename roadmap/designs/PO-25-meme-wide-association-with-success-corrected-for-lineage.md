# PO-25: Meme-wide association with success, corrected for lineage

**Area:** Population memetics · **Status:** design registered 8 November 2026, not run · **Priority:** 2.0 of 5

## Question

Which units go with a film's box-office rank once genre, era, length, budget and documented lineage are controlled, and do the hits hold in held-out franchises and between instalments of one series?

## Prediction

At least 20 units pass 5% false discovery; at least 60% replicate in sign at p < 0.05 in held-out franchises; their combined score picks the better-earning of two consecutive instalments in at least 55% of within-series pairs, ahead of word and embedding scores. About 450 usable pairs (of 624) give roughly 0.5 power at 55%, 0.85 at 58%.

## What would refute it

Fewer than 5 hits replicate, or within-series accuracy's 95% interval includes 50%, or a word or embedding score does as well (difference interval includes zero). Void if hits survive shuffling outcomes between franchises (inflation factor, observed over expected median test statistic, above 1.1).

## Data

Discovery: CMU Movie Summary Corpus (CC BY-SA): 7,531 plots carry Freebase revenue, 5,693 of them with ten or more sentences (reviewer count, 2026-11-08). Within-family test: Wikidata (CC0; query service, same day) lists 179 film series with two or more films that have box office and an English article (803 films); 3,236 films have both box office and budget.

## Method

Mask names; cut plots to ten sentences. Fix a 5,000-unit panel on development films. Per unit, a mixed model of within-year revenue rank on presence; covariates: genre, country, decade, budget where known, full summary length (a proxy for editor attention) and ten principal components; random effect: franchise. Split discovery and replication by franchise. Then score consecutive instalments within series.

## Cheap rival it must beat

The same pipeline on TF-IDF terms; ridge regression on sentence embeddings; an LLM rating 20 named story features; random non-meme units; length and metadata alone. Most dangerous: metadata (budget, sequel number, genre) overall and embeddings for prediction, since content explains little of success. A null still measures how far lineage inflates naive hits.

## Premise it tests

P1. Replicated, within-series unit hits beyond words would give neuron memes practical content; none, or a tie, leaves the case for neurons resting on readability. Each hit is re-tested on reworded sentences to see whether it is a word or an idea.

## Biology analogue

Genome-wide association with a kinship mixed model; within-family design against confounding.

## Cost

M. About 7,000 plots, under an hour of extraction per model; one to two weeks for joins, models and the within-series test.

## Payoff

A replicated list of success-linked memes with the shrinkage from between-family to within-family effects measured, or a documented null with its cause.

## Main risk

Summaries are written after release, and hits attract longer, more polished ones, so units may detect editor attention, not story content. Budget, stars and marketing dominate revenue; a null is the likely result.

## Reviewer note

The sibling test was unpowered: only 111 sources have two box-office adaptations, so it now compares instalments within 179 series. Added budget and summary-length covariates; revenue count confirmed. Unrepaired: summaries are written after success, so hits may read editor attention.

## Sources

- Klein et al. 2005
- Eliashberg, Hui and Zhang 2007
- Martin et al. 2016
- CMU Movie Summary Corpus (Bamman, O'Connor and Smith 2013)
