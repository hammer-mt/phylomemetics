# FI-22: Forecast forward: do winning headline units spread?

**Area:** Fitness and forecasting · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

Do units that raised clicks in randomised tests before a cut-off date become more common in headlines written afterwards, and do they still pick winners months later?

## Prediction

Using only tests before 1 July 2014: the fitted weights pick 54% or more of winning headlines three to ten months later (chance 50%), and a unit's click effect correlates 0.15 or more (rank correlation) with its later change in share of headlines, after allowing for its starting share and earlier trend.

## What would refute it

Forward accuracy under 52%; or the correlation's 95% interval (resampling tests in weekly blocks, not units) includes zero; or click effects measured after the cut-off predict share changes before it equally well. Exploratory plus confirmatory files should hold about 4,100 clean headline-only tests on each side (scaled from the exploratory count); simulate before registering.

## Data

The Upworthy archive of FI-21 (CC BY 4.0), exploratory and confirmatory files only, so FI-21's holdout stays unopened; tests dated by creation time. Click effects use clean-period same-image headline pairs (24 January to 24 June 2013; 11 January 2014 on). Headline frequencies use every tested headline, including the faulty-randomisation months, which spoil clicks but not what was written.

## Method

Units firing in at least 1% of pre-cut-off headlines, chosen without looking at clicks; settings frozen from FI-21. Click effect from within-test contrasts before the cut-off; if its split-half reliability is under 0.2, call the spread test uninformative. Report word-like units and units no single word reproduces separately. Nulls: test dates shuffled; effects reassigned among units of equal frequency.

## Cheap rival it must beat

The same two forecasts from word sequences (most dangerous: stock phrases probably show the cleanest rise), from sentence-embedding clusters, from headline length, and from 'last quarter's frequency persists'. An LLM judge is excluded as a forecaster because it was trained after 2015. If phrases win, selection on packaging is still shown with causally measured fitness.

## Premise it tests

P1. A pass shows measured fitness followed by spread, with fitness set by randomisation, in one newsroom whose editors saw the results: guided selection, not blind selection. P1 gains only if units that no single word reproduces also respond. A fail says editors did not copy what won, or that fitness is too short-lived to forecast.

## Biology analogue

Response to selection (breeder's equation)

## Cost

S. Reuses FI-21 activations; two to three days of CPU analysis and date tabulation.

## Payoff

A direct test of whether fitter memes spread, with fitness measured by experiment and not inferred from fame, plus the shelf life of a fitness estimate. The wear-out seed falls out of the same tables.

## Main risk

One newsroom is one history. Upworthy changed its style on purpose and Facebook changed its feed in 2014, so frequency shifts may reflect policy, not test results. Click effects for single units, estimated from about 4,100 tests, may be too noisy to correlate with anything.

## Reviewer note

Moved the cut-off from 1 April to 1 July 2014: the original left about 2,800 tests to estimate fitness, now about 4,100 each side. Kept FI-21's holdout unopened; added block resampling, a reliability gate and a reverse-time check. Remaining weakness: one newsroom, whose policy also changed.

## Sources

- Matias et al. 2021, Upworthy Research Archive (with the 2024 correction)
- Robertson et al. 2023
- Ye, Yoganarasimhan, Zheng 2025 (LOLA)
- Parikh et al. 2026 (TEA-Time)
