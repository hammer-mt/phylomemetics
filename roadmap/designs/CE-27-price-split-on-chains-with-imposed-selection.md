# CE-27: Price split on chains with imposed selection

**Area:** Cultural evolution, re-run · **Status:** design registered 8 November 2026, not run · **Priority:** 3.5 of 5

## Question

When selection of known strength is imposed on retelling chains, does a unit-based Price split report it, and report none when parents are picked at random?

## Prediction

With parents picked at random, no more than 7% of units are flagged as selected at the 5% level. With parents picked by a hidden score, units chosen on development texts as tracking that score are flagged in at least 80% of populations at the strong setting, and their selection term rises with imposed strength (rank correlation at least 0.6).

## What would refute it

More than 10% of units flagged under random parents, or detection under 50% at the strong setting. 3 strengths by 30 populations by 8 texts by 10 generations: 7,200 retellings. Rates over 30 populations have standard error about 0.09, enough to tell 80% from 50%; confirm by simulation.

## Data

Built: 30 starting sets of 8 short tales from Hagedorn and Darányi's Folktexts (Zenodo 6575263; CC BY-SA 4.0 per its README). Each generation, 8 children each retell one parent in about 120 words. Parents are drawn at random, or with odds rising in a hidden judge's score of how badly the story ends. Texts to be released CC BY-SA 4.0.

## Method

The Price equation splits change in a trait's mean into selection (covariance of the trait with number of children) and transmission (mean parent-to-child change). Compute both per unit per generation; flag selection by shuffling parent choices. Units chosen on separate development populations. Reader, writers and judge are different models. Nulls: the random-parent arm; random units matched on firing rate.

## Cheap rival it must beat

The same split on word presence, TF-IDF terms, sentence-embedding components, an LLM reader re-scoring the hidden property, random units and length. Most dangerous: the LLM reader, which nearly is the judge; then word presence, since bad endings have their own words. If they detect as often, units add nothing, but the sample-size curve still stands.

## Premise it tests

P1. A calibration, not proof of memes: it shows whether unit readouts can carry a selection analysis and at what sample size. A fail puts selection claims made with units, such as the study's underpowered E3, out of reach.

## Biology analogue

Artificial selection lines beside drift lines

## Cost

M. 7,200 retellings and as many judge calls, two extraction passes; about three days on one laptop.

## Payoff

A false-alarm rate and a sample-size curve for detecting selection with units, and a worked case where transmission (writers softening endings) can oppose selection.

## Main risk

The hidden property may be carried by a few obvious words, so every rival detects it; or writers' drift may swamp selection at every strength tried.

## Reviewer note

Added by the reviewer. The Price equation is an identity for any trait, so CE24 cannot show the unit readout is right; this gives a known answer first. Weakness: LLM writers under an artificial rule; the selected property may be lexical. Area-wide, novelty was checked on arXiv only.

## Sources

- El Mouden et al. 2014
- Nettle 2020
- MacCallum et al. 2012
- Perez et al. 2024 (arXiv:2407.04503)
