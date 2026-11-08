# CE-25: Content, prestige or conformity along citations?

**Area:** Cultural evolution, re-run · **Status:** design registered 8 November 2026, not run · **Priority:** 2.5 of 5

## Question

Along documented citation links, do some ideas pass from sources to citing papers more readily than others, even without their words, and does passing depend on how eminent the carrier is?

## Prediction

Content: a unit's propagation score (how much more often papers citing a carrier show it than papers citing none) is stable between 2015-2019 and 2020-2024, rank correlation at least 0.5. Adoptions sharing none of its trigger words occur at 1.5 times the swapped-reference rate. Prestige: within a paper, adoption odds rise 10% per standard deviation of the carrier's earlier citations.

## What would refute it

Stability under 0.3, or word-free adoptions no more common than under swapped references, or neuron units predicting held-out adoption no better than n-gram units (log-loss gain under 1%). 5,000 citing papers with at least 10 abstract-bearing references each give several hundred thousand paper-by-unit exposures, ample for odds ratios of 1.1; papers split by year.

## Data

5,000 papers from one arXiv category, 2015 to 2024 (arXiv metadata, CC0 1.0), plus every reference that has an abstract, with yearly citation counts, from OpenAlex (CC0). OpenAlex's keyless quota was exhausted when checked, so a free key is needed. How fully it lists references for arXiv preprints is unconfirmed: check before registration.

## Method

500 units chosen on 2015 to 2018, fixed. Propagation score per unit after Kuhn, Perc and Helbing. Logistic model of unit presence in the citing abstract on carrier share, the carrier's citation rank within the list, and unit effects. Control: each reference swapped for its nearest uncited same-year paper by embedding. Conformity reported, not tested: authors choose what they cite.

## Cheap rival it must beat

The same model with n-gram units (the Kuhn, Perc and Helbing meme score), embedding-cluster units, LLM-tagged concepts on 500 papers, random units, abstract length. Most dangerous: n-grams, since a field reuses its terms. If they tie, the estimates stand as cultural-evolution results and neurons are shown unnecessary here.

## Premise it tests

Assumes both. It re-runs a published propagation analysis with the new unit. Neuron units earn credit only if word-free adoptions exceed the swapped-reference rate and add prediction; otherwise P1 gains nothing. Conformity needs assigned exposure, which a citation list is not.

## Biology analogue

Transmissibility of variants; hitch-hiking with fit carriers

## Cost

M. Data assembly dominates; several hundred thousand sentences per model is a few hours of extraction.

## Payoff

A count of ideas that pass along citations without their words, a stability figure for per-idea transmissibility, and a within-paper prestige estimate.

## Main risk

Topic similarity between a paper and what it cites mimics transmission; the swapped-reference control is matched by embedding, so it may remove real transmission too.

## Reviewer note

Rebuilt around the published propagation score. Dropped the content-share test (unit base rates make it a tautology) and the conformity test (authors choose their sources, and a presence cut-off alone bends the curve). Exposure now covers all references with abstracts. Still observational: association only.

## Sources

- Kuhn, Perc & Helbing 2014
- Henrich & Boyd 1998
- Acerbi et al. 2016
- Henrich & Gil-White 2001
