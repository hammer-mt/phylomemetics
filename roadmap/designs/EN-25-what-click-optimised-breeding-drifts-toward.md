# EN-25: What click-optimised breeding drifts toward

**Area:** Meme engineering · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

When a breeding loop selects headlines on a click predictor trained on real randomised tests, does it drift toward negativity and unsupported claims, and which brake holds it: a word list, named internal units or an LLM reviewer?

## Prediction

Against a random-selection loop, five generations of click selection raise negative words per headline by at least 50% and unsupported claims by at least 10 points. The word-list brake is evaded by rewording. The LLM reviewer holds judged negativity within 10% of the start while keeping 70% of the predicted click gain; the unit brake does no better.

## What would refute it

No drift beyond random selection (negative words up under 10%, unsupported claims up under 3 points); or the word list holds judged negativity as well as the reviewer; or the unit brake beats the reviewer (a safety use for units). 500 paired tests give about 90% power for a 3-point difference if per-test differences have a spread of 20 points.

## Data

Upworthy Research Archive (Matias et al. 2021): 32,487 randomised headline tests, CC BY 4.0, OSF node jd64p. Drop the 22% of tests flagged for faulty randomisation and arms differing in image. Predictor and unit brake (10 named negativity and exaggeration units) fitted on the exploratory and confirmatory sets; 500 hold-out tests seed the loops. 300 headlines get blind human ratings.

## Method

Each loop starts from a test's losing headline. A local LLM, told only to rewrite, offers 8 variants per generation; the predictor selects. Unseen by the loop, drift is read by a negative-word count, a human-checked judge from another model family, and whether the test's own headlines and excerpt support each new claim. Arms: unbraked; three brakes; random selection.

## Cheap rival it must beat

Brakes: a negative-word list; an LLM reviewer asked whether a variant is more negative or less supported than the original (most dangerous, probably the best brake); a limit on embedding distance. Predictors: embeddings plus a linear model against unit profiles. The real gap between archived winners and losers is the yardstick for how much drift readers reward.

## Premise it tests

Assumes both; tests a use. Because the optimiser rewords its way around word lists, this setting is built for units to beat words. If they still lose to the reviewer, P1 has no safety use here.

## Biology analogue

Serial passage: selecting a pathogen for spread also selects for virulence.

## Cost

M. About 100,000 generated headlines and their forward passes, three days on the laptop, plus judge calls and two hours of human rating.

## Payoff

Measured limits for the programme's ethics rules: which brakes any optimiser must carry, and whether the instrument can enforce them. Nothing generated is shown to an audience.

## Main risk

The click predictor is weak (about 46% at picking the winner among roughly three headlines), so drift may reflect its quirks, not what readers reward. El and Zou 2025 already showed this kind of drift with simulated audiences, so the new part is the comparison of brakes.

## Reviewer note

Drift is now measured against a random-selection loop, because LLM rewriting drifts unaided; the judge differs from the reviewer and is checked by humans; the hypothesis no longer contradicts its falsifier. Remaining weakness: selection is on predicted, not real, clicks.

## Amendment, 10 November 2026

From the field survey (arXiv:2611.08438): the activation monitors it lists are linear probes and alerts on learned features; it names no monitor built on single neurons, and warns that monitors miss inputs their features do not track. Add a fourth brake: a linear probe on the residual stream, fitted on the unit brake's data. If the probe holds and the units do not, the failure belongs to the neuron basis.

## Sources

- Matias et al. 2021, Upworthy Research Archive, doi:10.1038/s41597-021-00934-7
- Robertson et al. 2023, Negativity drives online news consumption
- Ye, Yoganarasimhan, Zheng 2025, LOLA, arXiv:2406.02611
- El and Zou 2025, Moloch's Bargain, arXiv:2510.06105
- Barez, Wu et al. 2026, Understanding AI from the Inside (arXiv:2611.08438)
