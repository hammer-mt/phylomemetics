# RX-26: Who or what: a Girard and Dawkins split

**Area:** Remix and influence · **Status:** design registered 8 November 2026, not run · **Priority:** 2.5 of 5

## Question

When a researcher is exposed to an idea new to them, is adoption explained more by which idea it is (Dawkins) or by who displayed it (Girard)?

## Prediction

Dawkins: an idea's adoption lift repeats across separate halves of exposers (r of 0.5 or more) and predicts held-out adoption better than the exposer's lift does. Girard: with the idea held fixed, one standard deviation more exposer prestige raises adoption odds by 20%. Our bet: Dawkins on both counts.

## What would refute it

Prestige odds ratio of 1.2 or more with an interval excluding 1, or exposer lift predicting held-out adoption as well as idea lift: a content-only account fails here. Idea split-half r under 0.2: ideas show no stable merit. Needs 5,000 exposers and 2,000 ideas with 20 or more offers each; confirm on development years.

## Data

OpenAlex (CC0; free API key): the RX25 field sample widened to about 60,000 abstracts, 2000 to 2020, with citation links and author identifiers. Prestige is an author's citations received inside the sample before the exposure year. A citation is the exposure record. Early years fix a dictionary of 3,000 repeatable units; later years are scored.

## Method

Event: B's authors cite A in year t; a unit in A, absent from their earlier abstracts and from B, is 'offered'. Outcome: it appears in their abstracts within 3 years. Lift is adoption over the unit's field-year base rate. Logistic model: fixed effect per unit, prestige and prior citer-to-cited similarity. Null: cited authors permuted within field and year.

## Cheap rival it must beat

The same analysis with keywords and noun phrases, embedding clusters and LLM-extracted claims (subsample) as the idea; exposure count alone; random units. Most dangerous: keywords, since in abstracts a new term is the idea. If keywords give the same answer, it stands without neurons. Wording does not confound the who-or-what split itself.

## Premise it tests

Assumes both. Neuron memes measure only the 'what'. If who-effects dominate, a content-only science of memes misses most of the story however good the unit. If the answer changes with the unit type, the unit matters (P1).

## Biology analogue

Strain fitness versus super-spreader hosts.

## Cost

M: one to two weeks. Pulling author histories is the labour; about 2 hours of forward passes per model; the models fit on a laptop with subsampling.

## Payoff

A head-to-head of the mimetic and memetic accounts on one dataset, which the literature search did not find, and a ceiling on what any content measure can explain.

## Main risk

Citing is not random exposure: authors cite those they already resemble, and famous papers draw ceremonial citations that lower adoption, which biases against Girard. A citation does not prove reading. Girard's distinctive claims, rivalry and scapegoating, are not tested; this is prestige bias.

## Reviewer note

The outcome sat in the citing paper itself, so 'adoption' restated why it cited; it is now measured in later papers. The variance ratio reflected unit base rates and group sizes; replaced by split-half and held-out comparisons. Remaining: exposure is self-selected, so identification stays weak.

## Sources

- Henrich and Gil-White 2001
- Salganik, Dodds, Watts 2006
- Youngblood 2019
- Claidière et al. 2014
