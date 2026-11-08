# FI-28: Do hits leave more descendants than matched flops?

**Area:** Fitness and forecasting · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

After a film's release, do later unrelated films carry more of its distinctive units if it was a hit than if it was a matched flop, compared with films made before it?

## Prediction

Among matched pairs (same broad genre, released within two years, budgets within a factor of 1.5, box office at least three times apart), the hit's units gain more ground from the ten years before release to the ten years after than the flop's in 57% or more of pairs (chance 50%), in both models.

## What would refute it

A share of 52% or less, which with the roughly 390 pairs countable today puts the 95% upper bound below 57%; or an advantage that vanishes once the same before-and-after gain in rarity-weighted word overlap is regressed out. 390 pairs give about 80% power for a true share of 57%.

## Data

CMU Movie Summary Corpus (CC BY-SA, version not stated): 42,306 plots; 5,665 have box office and ten or more sentences. Wikidata budgets (CC0) cover 2,133. Target films from 1960 to 2002 yield about 390 budget-matched pairs, about 1,390 if budget is ignored (rough count, 8 November 2026). Names masked with the corpus's entity tags; plots cut to 10 sentences.

## Method

A film's distinctive units: present in it and in none of 1,000 reference films from other genres. Ground gained: the average share of those units present in same-genre films, over the ten years after release minus the ten years before. Same-series films and remakes excluded. Statistic: hit minus flop within pairs. Null: hit and flop labels swapped within pairs.

## Cheap rival it must beat

The same before-and-after gain from rarity-weighted word overlap (TF-IDF; most dangerous, it matched neurons on heredity), sentence-embedding similarity, the model's averaged internal state, random units, and an LLM shown a later plot and both films and asked which it resembles (subsample). If words win, success-biased copying is still shown; only the neuron claim fails.

## Premise it tests

P1. A pass says cultural selection can be measured across the film record: success raises a work's number of descendants, which E3 could not test. P1 gains only if units show it beyond word overlap. A well-powered null says hits are imitated in ways plot summaries do not record, or not at all.

## Biology analogue

Fitness as number of offspring, measured against ancestry

## Cost

M. About 20,000 plots of 10 sentences (200,000 sentences, about an hour per model at the measured rate; shared with FI-24 and FI-26). One to two weeks, mostly budgets and pair building.

## Payoff

A culture-wide answer to whether a popular work gains offspring, from hundreds of matched comparisons in place of 15 works, plus a list of which units each hit passed on.

## Main risk

Budget matching leaves only about 390 pairs unless more budgets are parsed from infoboxes. Imitators of a hit may be summarised in the hit's own terms, which favours words. Ten sentences may be too little plot. Word overlap may tie again.

## Reviewer note

Added by the reviewer: a culture-wide successor to E3. Comparing before with after cancels 'hits were more mainstream anyway' and any difference in how easily units detect a hit. Remaining weakness: only about 390 budget-matched pairs today, and imitators may be described in the hit's own words.

## Sources

- CMU Movie Summary Corpus (Bamman, O'Connor, Smith 2013)
- study/results/experiments.json (E3: rho 0.25 and 0.00, n = 15, power 0.27)
- Salganik, Dodds, Watts 2006
- Martin et al. 2016
