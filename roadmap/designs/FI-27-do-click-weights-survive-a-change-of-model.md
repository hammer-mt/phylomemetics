# FI-27: Do click weights survive a change of model?

**Area:** Fitness and forecasting · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

Do click weights learned on Qwen3-8B units still pick held-out winning headlines when carried to Llama 3.1 8B through matched units, with no refitting?

## Prediction

Weights carried through matched units keep at least half of the accuracy above chance that Llama's own fitted unit weights reach, in both directions, and beat weights carried through random units by 1.0 point. The same holds for units that no single word reproduces.

## What would refute it

Carried weights keep under a quarter of the fitted lift, or do no better than random-unit mapping; or the transfer disappears once units that a single word reproduces (correlation 0.5 or more) are removed. About 1,900 holdout tests give a standard error of 0.4 to 1.1 points on accuracy; confirm by simulation.

## Data

The FI-21 activations and splits of the Upworthy Research Archive (OSF jd64p, CC BY 4.0). Units are matched on text with no usable click data: the roughly 30,000 packages from the faulty-randomisation months that FI-21 discards.

## Method

Fit click weights on 2,000 Qwen units (confirmatory file). Pair each with the Llama unit whose firing over the matching headlines is most similar. Score holdout pairs in Llama using Qwen's weights on the paired units; repeat in reverse. Statistic: share of the fitted lift retained. Nulls: random Llama units matched on firing rate and layer; pairings shuffled.

## Cheap rival it must beat

Word-count weights, which move between models for nothing (most dangerous: if units carry only what words carry, transfer proves nothing about P2); a straight-line map between the two models' averaged internal states, fitted on the matching headlines; sentence-embedding weights reused unchanged; refitting in Llama on 10% of the data.

## Premise it tests

P2, in its functional form. A pass on units that no word reproduces says a meme's measured fitness belongs to the meme, not to one model. A pass only through word-like units says P2 holds for vocabulary. A fail while the straight-line map transfers says models agree as whole spaces, not unit by unit.

## Biology analogue

Does a gene's measured effect carry to its counterpart in another species?

## Cost

S. No forward passes beyond FI-21; one to two days of CPU.

## Payoff

The first test of whether a fitness estimate, not just a firing pattern, carries between models, and a list of unit pairs that agree on what wins.

## Main risk

Units may give little lift over chance in either model, leaving nothing to carry. Or all transfer runs through word detectors, as in the study, where twin units correlated 0.82 on original text and 0.14 once names were removed.

## Reviewer note

Added by the reviewer, promoted from a seed: the only P2 test in this area, and nearly free after FI-21. Remaining weakness: if unit lift is small or runs through word detectors, transfer shows only that vocabulary is shared, which word weights already give for nothing.

## Sources

- study/results/robustness/nulls.json (twin correlations)
- Matias et al. 2021, Upworthy Research Archive (with the 2024 correction)
- Gurnee et al. 2024
- Huh et al. 2024
