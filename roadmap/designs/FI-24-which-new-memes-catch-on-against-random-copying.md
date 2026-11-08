# FI-24: Which new memes catch on, against random copying?

**Area:** Fitness and forecasting · **Status:** design registered 8 November 2026, not run · **Priority:** 2.5 of 5

## Question

When a unit first appears in a film genre, does its later spread exceed what random copying predicts, and can the spreaders be picked from what was known at its birth?

## Prediction

Units absent from a genre's first 300 films, then present in 5 or more of the next 300 after first appearing, are at least twice as common as under shuffled release order. Features fixed at birth (earlier spread in other genres, number of co-arriving units, host film's box office) predict which spread, with AUC 0.60 or higher (0.50 is chance).

## What would refute it

No excess over shuffled release order (first appearances are then noise or steady background rates, and the design stops). Or AUC under 0.55, or no higher than the 95th percentile in shuffled-order and simulated random-copying worlds. Or new words through the same pipeline score as well. Thousands of first appearances give a standard error near 0.02.

## Data

CMU Movie Summary Corpus (CC BY-SA, version not stated): 42,306 Wikipedia plots with release dates and genres. Five genres with 1,169 to 2,279 dated plots of ten or more sentences (horror, science fiction, crime, fantasy, romantic comedy; recounted 8 November 2026). Names masked using the corpus's own entity tags; plots cut to their first 10 sentences.

## Method

Present: a unit's score beats all 1,000 out-of-genre reference films (0.1% false alarms). Mutation: first presence after a genre's first 300 films; a series counts once. Nulls: release order shuffled within genre (keeps each unit's rate, removes time); simulated random copying with invention (Bentley 2004). Scored on held-out genres. Positive control: ten documented genre innovations, pre-listed; five must register.

## Cheap rival it must beat

New words and rarity-weighted terms through the same pipeline (most dangerous: new ideas often arrive with new words, such as cyborg or hacker, and word counts carry no hindsight); embedding clusters first populated in a given year; an LLM reading the host plot and naming the element most likely to be imitated; copies in the first three years alone.

## Premise it tests

P1. One-sided. If neither units nor words beat random copying, drift is enough to explain meme turnover in film genres. If new words do as well as units, P1 gains nothing. A pass by units alone is weak support, because models trained on later culture may detect the ideas that won more easily.

## Biology analogue

Fate of a new mutation: drift or positive selection

## Cost

M. About 8,000 plots of 10 sentences (80,000 sentences, under an hour per model), storing only units above the strict bar. About two weeks, mostly simulations and name masking.

## Payoff

A dated list of mutations per genre with their fate, a calibrated test of selection against drift, and an early-warning score for new memes. A stronger successor to the Batman-only mutation plan.

## Main risk

The models were trained after all these films, so units exist mainly for ideas that later became common: failed mutations are under-detected. With about 900,000 units, chance alone gives roughly 900 false presences per film. Neutral-model tests are sensitive to time bins and population size (Karjus et al. 2020).

## Reviewer note

Added a shuffled-release-order null and a fixed 300-film window, because steady background rates and unequal exposure could fake 'spread'. Dropped firing strength as a predictor (it leaks hindsight); added host box office. Remaining weakness: models trained on later culture detect winners more easily, so only a fail is clean.

## Sources

- Bentley, Hahn, Shennan 2004
- Feder, Kryazhimskiy, Plotkin 2014
- Karjus et al. 2020
- CMU Movie Summary Corpus (Bamman, O'Connor, Smith 2013)
