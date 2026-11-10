# EN-27: Is the unit needed to read the meme?

**Area:** Meme engineering · **Status:** design registered 8 November 2026, not run · **Priority:** 3.5 of 5

## Question

When a meme's detector units are switched off, does the model get worse at saying whether a passage contains that meme, and only that meme?

## Prediction

No. Switching off a meme's 10 best detector units lowers the model's yes-or-no accuracy on that meme by under 3 points, the same as matched random units. Removing the meme's difference-of-means direction (average activation with the meme minus without) lowers it by at least 10 points, and other memes' accuracy by under 3.

## What would refute it

Switching off the 10 units lowers accuracy on their own meme by 5 points or more beyond both matched random units and other memes' questions, with a 95% interval above zero. Twenty memes detect a 5-point drop with about 75% power if memes differ with a spread of 8 points.

## Data

EN-23's material: 20 memes from `preregistration/memes_for_tracing.json`, each with 60 LLM-written passage pairs with and without the meme, its usual words banned, checked by a second model and a 10% human sample; 30 pairs pick units, 30 are held out. The study's own 18 mapped units (`preregistration/dev_units.json`). Qwen3-8B and Llama 3.1 8B open weights.

## Method

The model reads a held-out passage and answers 'does this contain [definition]?' while the units are held at their average value on neutral text. Outcome: the fall in accuracy, and in confidence in the right answer, against the unharmed model. Own meme against other memes, paired permutation test across memes. Null: random units matched on layer and firing rate.

## Cheap rival it must beat

Removing the meme's difference-of-means direction (most dangerous: if it blinds the model while units do not, the meme is a direction). Switching off units that detect the meme's keywords. Matched random units. As a ceiling, deleting the meme's sentence from the passage. A null still tells us detector units are gauges beside the computation, not parts of it.

## Premise it tests

P1. If the model reads the meme just as well without its detector units, they are correlates, and 'the meme is which neurons fire' holds as a measurement only. A specific loss would be the first evidence that the units are part of how the model understands the idea.

## Biology analogue

Lesion study: remove one brain area, test one skill.

## Cost

S. One day on the laptop once EN-23's passages exist: forward passes only, no text generation, no paid judges.

## Payoff

The causal check from the reading side: whether the instrument's units are used by the model or merely move with the meme. It decides whether later work should track units or directions.

## Main risk

Redundancy. Models often have several routes to an answer, so a null may mean back-up routes, not irrelevant units. The 2026 census found that silencing a neuron moves a model about a thousand times less than pushing it. The question may also be too easy or too hard for a 4-bit 8B model.

## Reviewer note

Added by the reviewer: the area steered units but never tested whether detector units are needed for reading. Remaining weakness: a null is ambiguous because of redundancy, and the areas on neurons as units or measurement validity may register a similar lesion test.

## Amendment, 10 November 2026

Add arms of 100 and 500 units ranked by their effect on the answer, since the companion paper traces half of the push toward one name to 155 neurons. Silence units only where they are in their own strong range, and report the paper's damage measure (the change in next-word predictions over the firing token and the next 16) beside accuracy.

## Sources

- Öztürk 2026, LLM Electrodes, doi:10.21203/rs.3.rs-11577020/v1
- Cammarata 2026, Neural networks are already clean
- Arditi et al. 2024, arXiv:2406.11717
- Arad, Mueller, Belinkov 2025, arXiv:2505.20063
- Tigges et al. 2026, Neurons as an Interpretable Basis
