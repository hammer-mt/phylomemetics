# FI-23: Selection rerun: do sequels keep what sold?

**Area:** Fitness and forecasting · **Status:** design registered 8 November 2026, not run · **Priority:** 3.5 of 5

## Question

Within a film series, does a later film carry more of the distinctive units of those earlier instalments that earned more at the box office?

## Prediction

On name-free plots, the within-series rank correlation between an earlier film's box office and a later film's inheritance from it is 0.10 or more, after allowing for position in the series and years apart, with the same sign in both models.

## What would refute it

A 95% upper bound below 0.10; or a neuron effect that vanishes once word-overlap inheritance is regressed out. An interval spanning both zero and 0.10 is inconclusive, not a null. With 250 series, 80% power is reached somewhere between 0.10 and 0.15, depending on how alike a series's film pairs are (E3: 27% at 0.30). Stop under 200 series.

## Data

Wikidata (CC0) lists 317 to 539 film series with three or more films on English Wikipedia, depending on how a series is defined (recounted 8 November 2026). Only 137 have Wikidata box office for three films, so most figures must come from Wikipedia infoboxes, adjusted for inflation. Plots from Wikipedia (CC BY-SA 4.0), names masked by the pre-registration's rule-based masker.

## Method

No unit selection; present means beating nearly all of 1,000 reference films. Plots in a series cut to equal length; series capped at 12 films. Inheritance: the share of an earlier instalment's unique units carried by a film at position three or later. Regress on within-series box office rank, position and gap; resample series. Null: box office shuffled within series.

## Cheap rival it must beat

The same statistic from rarity-weighted word counts (TF-IDF or BM25), sentence-embedding similarity, the model's averaged internal state, random non-meme units, and an LLM reading three name-free plots and saying which earlier film the later one borrows from. Most dangerous: TF-IDF, which matched neurons on heredity. If it wins, selection is still measured; only the neuron claim fails.

## Premise it tests

P1. A pass shows success-biased copying can be measured across hundreds of lineages; only a signal beyond word overlap raises belief in P1. A well-powered null says the selection claim does not hold at the level of plot summaries. Agreement between two models is replication, not a test of P2.

## Biology analogue

Selection differential: fitter parents pass on more traits

## Cost

M. Between 1,600 and 3,000 plots (under an hour per model). One to two weeks, mostly building the series table, parsing box office from infoboxes and masking names.

## Payoff

A far better powered answer to the question the study's selection test (E3) asked, with success measured at release, and a reusable table of series, order and box office for other lineage tests. Positive control: the immediate predecessor should rank first.

## Main risk

Box office coverage may cut the sample below 200 series. A sequel continues its predecessor's story whatever it earned, which leaves little room for choice. Sequel summaries recap earlier plots, and editors describe sequels in terms of the famous instalment. Word overlap may tie again.

## Reviewer note

Recount: 317 to 539 series by definition, 137 with Wikidata box office, 40% only three films long, so the claimed 80% power at 0.10 was optimistic. Added equal-length plots, a series cap and an inconclusive zone. Remaining weakness: box office needs infobox parsing, and story continuity limits choice.

## Sources

- study/results/experiments.json (E3: rho 0.25 and 0.00, n = 15, power 0.27)
- Wikidata query service, series counts, 2026-11-08
- Martin et al. 2016
- Salganik, Dodds, Watts 2006
