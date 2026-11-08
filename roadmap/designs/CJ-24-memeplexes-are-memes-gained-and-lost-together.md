# CJ-24: Memeplexes: are memes gained and lost together?

**Area:** Field conjectures · **Status:** design registered 8 November 2026, not run · **Priority:** 3.5 of 5

## Question

Along documented adaptation links, are memes of the same bundle kept or dropped together more often than their everyday co-occurrence and the link's overall fidelity predict?

## Prediction

For unit pairs in one bundle (bundles defined on unrelated films), the odds of being kept or dropped together across about 2,000 source-to-adaptation links are at least 1.5 times the odds across matched unlinked film pairs, and the excess holds in at least 10 unrelated source lineages.

## What would refute it

The ratio's 95% interval lies below 1.2: bundles travel together no more than their everyday co-occurrence predicts. For neurons: neuron bundles show no higher co-change odds than word-topic bundles or TV Tropes co-occurrence bundles (difference interval includes zero). Power depends on how many bundle pairs each of the 2,000 links carries; count that on development films first.

## Data

Wikidata 'based on' and 'derivative work' links (99,206 and 40,376 statements, CC0; 15,706 'based on' links start from a film) where both works have a Wikipedia plot (CC BY-SA): sample 2,000 links from at least 500 sources, plus 2,000 matched unlinked pairs. Bundles learned on CMU Movie Summary Corpus films outside those lineages. Names masked.

## Method

Units: neuron families, no two tracking one word. Bundle: units that co-occur in development films, each appearing without the others in at least 20% of cases, and peaking in different sentences (so not one scene). Statistic: joint loss and joint gain along links against unlinked pairs matched on genre, decade and likeness. Second null: shuffle keeping row and column totals.

## Cheap rival it must beat

Bundles from a topic model on the same masked text, from TV Tropes tag co-occurrence, from LLM-listed traits, and from random frequency-matched unit sets; summary length. The topic model is most dangerous: a scene's vocabulary travels together. Then bundles are real but are scenes, and neurons add nothing.

## Premise it tests

P1. Memeplex is a core term of memetics that has never been measured. Co-change beyond everyday co-occurrence, repeated across lineages, gives the unit a heredity property a concept detector lacks. If word topics do as well, bundling belongs to text, not neurons.

## Biology analogue

Linkage disequilibrium and supergenes; Pagel's test for correlated trait change.

## Cost

M. About 8,000 plots plus development films: a few hours of forward passes. Name masking, matching unlinked pairs and cleaning the links take most of a week.

## Payoff

The first measured memeplexes: a list of bundles that travel as wholes and the lineages they travelled in. Tells an adapter which elements can be dropped alone and which come as a set.

## Main risk

Co-change may still reflect settings: drop the courtroom and every courtroom unit goes. Wikipedia summaries of adaptations are sometimes written from the source's summary, inflating retention; a plot from another-language Wikipedia for one end (Tell Me Again) is the check.

## Reviewer note

Units that co-occur will also seem to change together between any two films, so the shuffle null was too easy. Added matched unlinked pairs as the comparison and a different-sentence rule against one-scene bundles. Remaining weakness: adaptation summaries may be written from the source's summary.

## Overlaps with

Asks nearly the same question as PO-24, which the judges would run instead.

## Sources

- Blackmore 1999 (The Meme Machine)
- Pagel 1994
- da Silva and Tehrani 2016
- Wikidata (based on P144, derivative work P4969)
