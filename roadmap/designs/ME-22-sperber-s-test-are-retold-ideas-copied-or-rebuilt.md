# ME-22: Sperber's test: are retold ideas copied or rebuilt

**Area:** Old memetics, re-run · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

When a story is retold down a chain, is its new content a copying error passed on to later retellings, or something each chain adds independently because every teller rebuilds the story the same way?

## Prediction

The memetic prediction: of the units a story gains by step 8, those inherited from the chain's own step 4 and absent from sibling chains outnumber those that also arose in sibling chains by at least 2 to 1. A step-8 retelling is matched to its own step-4 ancestor among six siblings at least 50% of the time (chance 17%).

## What would refute it

A copied-to-rebuilt ratio of 1 or less, with a bootstrap interval lying below 2, in both reader models, over 360 test chains from 60 stories (resampled by story; E3 had 15 works) and at least 2,000 gains; or matching under 30%. That is Sperber's outcome: new content is convergent, so shared units cannot be read as descent.

## Data

Built for the test and released with its lineage. 80 new 300-word stories without proper names, written from randomly drawn plot elements (20 for tuning, 60 for scoring). Each starts 6 chains that share only the seed. A chain is 8 retellings of about 150 words by two small open models taking turns, not the reader models: 3,840 texts.

## Method

Readers: Qwen3 8B, Llama 3.1 8B; presence rule set on tuning stories. A gain is a unit absent from the seed and from the story's first retellings, present at step 8, and rare elsewhere (not house style). Copied: also in the chain's own step 4, in no sibling. Rebuilt: in a sibling's step 8. Null: chain labels shuffled within story.

## Cheap rival it must beat

The same count on content words and on LLM-listed plot points; sentence embeddings, text length, random units and the mean-pooled residual stream for the matching test only. The LLM plot-point list is most dangerous. The ratio is a result about transmission whichever tool gives it; units earn a place only if a human check of 200 gains favours them.

## Premise it tests

P1. Also bears on how E2 is read. A ratio above 2 answers Sperber for these tellers. A ratio below 1 means resemblance in retold culture is largely convergence, so a neuron-defined meme would measure a shared attractor and not a lineage.

## Biology analogue

Parallel evolution in replicate lines; homoplasy against shared descent

## Cost

M. About 3,900 short texts from local 4-bit models, one to two days of laptop time, plus two reading passes and one LLM plot-point pass.

## Payoff

A number for how much of what a retold story gains is inherited and how much is rebuilt, plus a list of convergent units to exclude from later lineage tests.

## Main risk

Model tellers are not human minds, and the retelling prompt sets how faithfully they copy, so the ratio is a fact about these tellers. Small models may add little, leaving too few gains. Dropped details are not counted, because selection and attraction predict the same losses.

## Reviewer note

The original ratio depended on how profiles were centred and counted house style as attraction. Replaced with six replicate chains per story and a count of inherited gains against gains that recur in sibling chains. Remaining: model tellers, a prompt that sets fidelity, and no human chains yet.

## Sources

- Sperber 2000
- Claidière, Scott-Phillips and Sperber 2014
- Perez et al. 2024 (arXiv:2407.04503)
- Acerbi and Stubbersfield 2023
