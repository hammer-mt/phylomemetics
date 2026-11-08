# ME-24: Heylighen's selection criteria on randomised headline tests

**Area:** Old memetics, re-run · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

Do the criteria that Heylighen (1998) says help a meme get noticed and accepted, scored by neuron units chosen in advance, predict which headline wins a randomised test?

## Prediction

On confirmatory tests, at least 4 of 6 criteria (distinctiveness, novelty, simplicity, authority, formality, utility) raise click rate with 95% intervals excluding zero, and together they add at least 2 points of winner accuracy over headline length and punctuation.

## What would refute it

Fewer than 3 of 6 criteria significant in the predicted direction, or under 1 added point, on at least 5,000 confirmatory tests where only the headline differs (standard error about 0.7 points). Separately, units fail if LLM ratings of the same criteria beat them by more than 1 point.

## Data

Upworthy Research Archive (OSF jd64p, CC BY 4.0): 32,487 randomised tests, 150,817 arms, 2013 to 2015. Drop tests the archive's own column flags for unreliable randomisation (about 22%); keep tests where only the headline differs (count not yet checked). Tune on its exploratory part, score once on its confirmatory part. Archived data; nothing is optimised on a live audience.

## Method

Use only criteria Heylighen assigns to assimilation (being noticed, understood, accepted), which a click measures. For each, write 80 contrast pairs (same content, criterion high or low); pick units on 60, check on 20, freeze. No click data chooses units. Within each test, criterion differences predict the winner (conditional logistic regression), controlling length and punctuation. Null: arms shuffled within test.

## Cheap rival it must beat

An LLM rating each headline 1 to 7 on each criterion, readability and word-rarity formulas, an embedding probe trained on the contrast pairs, word n-grams, sparse-autoencoder features (Movva et al. 2025), random units, the mean-pooled residual stream. The LLM rater is most dangerous, because the criteria are defined in words. If it wins, the criteria still get a signed test.

## Premise it tests

P1. If unit-scored criteria predict as well as LLM ratings, units measure abstract properties of a meme and not only words. If only the ratings work, the theory survives without the unit. If neither works, the criteria fail at the attention stage.

## Biology analogue

Selection gradients on named traits

## Cost

M. 150,000 headlines take an hour or two per model; rating six criteria with a local LLM takes about a day.

## Payoff

The first test of Heylighen's assimilation criteria as a signed set on randomised data, and a check of whether units chosen in advance can stand in for a reader's rating.

## Main risk

Simplicity is already known to win on this archive (Shulman et al. 2024), so one criterion is a replication. Heylighen himself hedges on formality. Contrast pairs written by an LLM may differ in style as well as in the criterion. Click-bait habits may swamp the rest.

## Reviewer note

Cut to the six criteria Heylighen assigns to assimilation; expressivity and proselytism belong to later stages. Removed an unsourced count of flagged tests. Simplicity is already published on this archive (Shulman et al. 2024). Remaining: LLM ratings will probably tie the units.

## Sources

- Heylighen 1998
- Matias et al. 2021 (Upworthy Research Archive)
- Shulman, Markowitz and Rogers 2024
- Movva et al. 2025 (arXiv:2502.04382)
