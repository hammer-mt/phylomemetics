# CJ-26: Remix ledger: new memes or new combinations?

**Area:** Field conjectures · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

Over a century of films, does novelty arrive as new units or as first-time pairings of old units, once the measurement is checked on remixes with known recipes?

## Prediction

Check: on 300 synthetic remixes the ledger ranks the true sources above matched decoys and its inherited shares correlate at least 0.6 with the planted shares. Then, in films after 1980, the share of units that were rare before exceeds the date-shuffled baseline by under 2 percentage points, while the new-pair share exceeds it by at least 5.

## What would refute it

Check correlation below 0.4 for neurons, or not above the best rival by 0.1: the ledger cannot be trusted (300 remixes separate 0.6 from 0.4 by over four standard errors). Conjecture: a post-1980 excess of rare-before units of 5 points or more means real novelty beyond recombination.

## Data

CMU Movie Summary Corpus (39,589 dated plots, 25,208 after 1980, CC BY-SA), names masked. 300 synthetic remixes written by a small open model from two to four known source plots in set proportions, reworded by a second model; released CC BY-SA. Paired summaries of one story (Tell Me Again: 96,831 summaries, 29,505 stories, CC BY-SA 4.0) set the noise floor.

## Method

Units chosen on a random third of films from all years. For each other film, in release order: the share of its units that were rare before (in under 0.5% of earlier films) and the share of its unit pairs never seen together. Baseline: the same ledger with release dates shuffled within genre, which removes a fixed list's saturation.

## Cheap rival it must beat

The same ledger with word n-grams, TV Tropes or keyword pairs (Sreenivasan's method), embedding nearest-neighbour distance, and an LLM asked to list each film's sources. Keyword pairs are most dangerous: human-named and model-free. The check on synthetic remixes says which ledger to believe, so a rival's win still answers the conjecture.

## Premise it tests

P1. A model's units are a closed list, so the ledger registers only novelty the model has a unit for; training after every film in the corpus helps here. Failing the check exposes that limit. Passing it ahead of keyword pairs would support P1 as a way to measure originality.

## Biology analogue

Recombination against new mutation; exon shuffling against de novo genes.

## Cost

M. About 6 hours of forward passes per model for the corpus, a day of local generation for the remixes, a week in all.

## Payoff

'Everything is a remix' turned into three measured shares per work (inherited, new, newly combined) with a known error. An originality score that can be audited.

## Main risk

'New' units may be new things in the world (phones, wars) more than new story ideas, and words catch those equally. Wikipedia covers few early films, so early works look more original than they were. LLM-written remixes may be easier to unpick than human ones.

## Reviewer note

Units chosen on pre-1950 films could never be new, and a fixed list is all seen within a few hundred films, so the hypothesis passed by construction. Units now come from all years and 'new' means rare before. Remix licence corrected to share-alike. Remaining weakness: the closed unit list.

## Overlaps with

Asks nearly the same question as RX-21, which the judges would run instead.

## Sources

- Ferguson, Everything is a Remix (video series)
- Youn et al. 2015
- Uzzi et al. 2013
- Sreenivasan 2013
