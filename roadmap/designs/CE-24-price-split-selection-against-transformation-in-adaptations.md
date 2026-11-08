# CE-24: Price split: selection against transformation in adaptations

**Area:** Cultural evolution, re-run · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

Along documented book-to-film links, is change in how common a unit is carried by which books get adapted (selection) or by what adaptation does to them (transmission), and are units inherited more faithfully than words?

## Prediction

Memetics predicts selection carries at least half the change: once the change seen under randomly re-paired children (the shift of medium) is subtracted, the selection term exceeds the transmission term for at least 50% of units, and noise-corrected parent-to-child fidelity (regression slope) of units beats frequency-matched words by at least 0.05.

## What would refute it

Transmission exceeds selection for more than 80% of units after that subtraction, or the fidelity gap over words is below 0.02. At least 3,000 book-to-film pairs and 3,000 never-adapted novels, split by source work; with 1,500 held-out pairs a slope difference of 0.05 has standard error about 0.015.

## Data

Wikidata 'based on' links (CC0): 6,110 film-from-written-work links with English Wikipedia articles at both ends, 4,312 distinct sources (query of 2026-11-08). In a sample of 10, 8 had plot sections at both ends; confirm the count before registration. Plots are CC BY-SA 4.0, names masked with the pre-registration's deterministic masker. Plus novels with plots and no recorded adaptation.

## Method

The Price equation splits change in a trait's mean into selection (its covariance with number of adaptations) and transmission (mean parent-to-child change). Both per unit, standardised within medium. Fidelity: parent-child slope, noise-corrected using each film's 2012 (CMU corpus) and current plots. Units chosen on half the sources. Null: children reassigned within decade, genre and length; pairs sharing 8-word runs dropped.

## Cheap rival it must beat

The identical decomposition on word presence, TF-IDF terms, sentence-embedding dimensions and LLM-coded tropes on a 300-pair sample; random units; length. The equation holds for any trait, so only these comparisons are informative. Most dangerous: word presence, because Wikipedia editors reuse phrasing between a book's page and its film's. If words tie, the split still answers Sperber against Dawkins for plots.

## Premise it tests

P1. Higher fidelity than words under a change of medium is the heritability evidence P1 lacks. A dominant transmission term says units are reconstructed, which favours attraction theory over replicator memetics whatever the unit.

## Biology analogue

Price equation; parent-offspring heritability

## Cost

M. Wikidata and Wikipedia pulls, name masking, about 9,000 plots through two models; one to two weeks.

## Payoff

Edmonds's fifth challenge answered with numbers: selection and transmission shares per unit, and a fidelity figure to set beside words.

## Main risk

Plot summaries are written by Wikipedia editors, not the adapters, so the transmission term mixes adaptation with summarising. Wikidata misses adaptations, so 'never adapted' is noisy.

## Reviewer note

Counted the links on Wikidata. Added subtraction of the medium shift, within-medium scaling and a noise correction, because book and film plot pages differ in length and style. Run CE27 first. Weakness: editors, not adapters, wrote both texts, and the selection term will be small and noisy.

## Sources

- El Mouden et al. 2014
- Nettle 2020
- Mesoudi 2021
- Wikidata P144
