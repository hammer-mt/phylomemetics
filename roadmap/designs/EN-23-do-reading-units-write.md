# EN-23: Do reading units write?

**Area:** Meme engineering · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

Is the unit that best detects a meme in text also the unit that best produces it when pushed?

## Prediction

Mostly not. Across 20 memes, the 10 best detector units and the 10 best writer units share under 20% of their members, and detectors steer successfully less than half as often as writers.

## What would refute it

Detectors steer as well as writers (success rates within 10 points on held-out prompts), or the two top-10 sets overlap by more than 50%. Twenty memes give about 85% power for a 25-point gap if memes differ with a spread of 35 points. Detectors that steer no better than random units are sensors only.

## Data

20 of the 34 meme definitions in `preregistration/memes_for_tracing.json`. For each, 60 LLM-written sentence pairs with and without the meme, its usual words banned, checked by a second model and a 10% human sample: 30 pairs to pick units, 30 held out. The study's own 18 mapped units (`preregistration/dev_units.json`). Qwen3-8B and Llama 3.1 8B open weights.

## Method

Detectors: units ranked by how well activation separates the pairs. Writers: 2,000 shortlisted units, each pushed on tuning prompts and ranked by the rise in probability of meme sentences. Steer the top 10 of each on 20 held-out prompts, keywords banned; a blind judge from the other model, human-checked, scores meme and coherence. Paired permutation test across memes.

## Cheap rival it must beat

Random same-layer units; a difference-of-means direction from the same pairs; a prompt asking for the meme (ceiling); a keyword list on the detection half. Most dangerous: the direction. If it both reads and writes while single units do one or neither, the meme is a direction, not a neuron.

## Premise it tests

P1. If detectors do not write, 'which neurons fire' is a sensor reading and editing needs separately found units. If they do, one unit serves both, the strongest form of P1.

## Biology analogue

Sensory against motor neurons.

## Cost

S to M. Three to five days on the laptop; the writer screen is forward passes only, a few hours.

## Payoff

A rule for choosing edit handles, and a verdict on whether the study's detector units can be reused for engineering at all. It also records whether writers detect, and whether a detector fires when the model is simply asked to write its meme.

## Main risk

Writers are picked by the kind of push they are then tested on, so part of their lead over detectors is built in; the informative numbers are the overlap and detectors against random units. In 4-bit models the dose window may be too narrow for either set.

## Reviewer note

Writers were picked by gradient attribution, which Liu 2026 reports mispredicts control; now a direct push screen on tuning prompts. Added the keyword ban, a human check and the study's own 18 units. Remaining weakness: writers' lead is partly built in by how they are chosen.

## Sources

- Arad, Mueller, Belinkov 2025, arXiv:2505.20063
- Kran Research 2026, Evil Qwen replication (blog.kran.ai/evil-qwen)
- HedwigAI 2026, Qwen personality neurons
- Liu 2026, arXiv:2606.19831
