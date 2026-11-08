# CJ-23: Lindy effect: does a meme's age forecast survival?

**Area:** Field conjectures · **Status:** design registered 8 November 2026, not run · **Priority:** 3.5 of 5

## Question

Among story memes alive in films at a cut-off year, does age predict how well each holds its share over the next 20 years, beyond current frequency and random copying?

## Prediction

Each doubling of a meme's age at the 1992 cut-off raises its 1993 to 2012 retention (later share of films divided by its 1973 to 1992 share) by at least 5%, controlling that earlier share and its trend, and the effect exceeds 95% of 1,000 random-copying simulations.

## What would refute it

The age effect's 95% interval lies below 2% per doubling, or inside the random-copying range, for every unit type: Lindy is frequency in disguise. For neurons: the neuron age effect is no larger than the effect for TF-IDF terms or trope tags (difference interval includes zero). About 300 unit clusters give power 0.94 for a partial correlation of 0.2.

## Data

CMU Movie Summary Corpus: 42,306 Wikipedia film plots, 39,589 with a release year (counted 2026-11-08), CC BY-SA; 7,951 fall in 1973 to 1992 and 19,791 in 1993 to 2012. About 1,300 predate 1930, so ages are capped near 60 years. Series links from Wikidata (CC0). Trope tags from Gala et al. 2020 (licence unconfirmed; commit IDs only). Names masked.

## Method

Units: neuron families chosen on a random third of films from all years, scored on the rest. Age: years since a unit first passed 0.5% of films, capped for units already common in the 1930s. Regression across unit clusters, with bootstrap intervals. Null: random copying (each film draws memes from earlier films by frequency), with real yearly counts.

## Cheap rival it must beat

The same analysis with TF-IDF terms, TV Tropes tags, sentence-embedding clusters, LLM-listed traits and random non-meme units; current frequency alone. Frequency alone is most dangerous: under random copying old memes are simply the common ones. Trope tags may show Lindy with no model, which still settles the conjecture.

## Premise it tests

Assumes both. Lindy found equally in words and tropes is a fact about culture that needs no neurons. A stronger age effect for neuron units than for words, in both Qwen and Llama, would support P1 (ideas outlive their wording) and P2.

## Biology analogue

Taxon age against extinction risk; Van Valen's constant-extinction law is the null.

## Cost

S to M. About 6 hours of forward passes per model at the rate in preregistration/sources/neuron_timing_qwen3-8b.json (optimistic); store presence bits only. Simulations take minutes.

## Payoff

The first measured test of 'old ideas are the safe ones' against the right null. If it holds, age is a free forecasting feature; if not, Lindy is survivorship bias.

## Main risk

Wikipedia remembers the old films that still matter, so a meme that lived only in forgotten early films is dated too young and looks short-lived: survivorship can fake Lindy. Summaries were all written after 2001 in modern words, so wording drift, the neurons' best case, is absent.

## Reviewer note

Units were chosen on pre-1973 films, which excluded every young meme; now chosen on a random third. Thin pre-1930 coverage caps ages, now stated. Swapped an unreachable trope table for Gala et al. Remaining weakness: Wikipedia's memory of old films can fake the effect.

## Sources

- Eliazar 2017 (Lindy's law)
- Bentley, Hahn and Shennan 2004
- CMU Movie Summary Corpus
- Gala et al. 2020 (TV Tropes dataset)
