# IN-29: Same neurons at 4-bit, 8-bit and full precision?

**Area:** The instrument · **Status:** design registered 8 November 2026, not run · **Priority:** 4.5 of 5

## Question

Do the same unit indices give the same readings and presence calls when the model is not quantised to 4 bits?

## Prediction

Between the 4-bit and full-precision builds, the median per-unit correlation of sentence readings is at least 0.95, presence kappa (chance-corrected agreement) is at least 0.9, and at least 70% of a unit's 20 strongest sentences are shared.

## What would refute it

Median correlation under 0.9, presence kappa under 0.73 (the study's cross-model kappa, study/results/robustness/nulls.json), or top-20 overlap under 50%. Any of these makes 4-bit quantisation a first-order confound and blocks borrowing published labels or dictionaries made at full precision.

## Data

The study corpus (3,700 Wikipedia plot sentences, CC BY-SA) and 1 million tokens of FineWeb (ODC-By). MLX builds of Qwen3-8B and Llama-3.1-8B-Instruct at 4-bit, 8-bit and bf16 (16-bit floats), all on Hugging Face under the Apache-2.0 and Llama 3.1 licences.

## Method

Units: the study's 125 labelled units, IN21's frozen units and 5,000 random active units, by layer third. Same text through each build; per-unit Pearson correlation of sentence readings, kappa of presence under the frozen rule, overlap of the 20 strongest sentences. Null: a different unit in the same layer, which gives the chance level. 5,000 units make every interval narrow.

## Cheap rival it must beat

TF-IDF, embeddings and an LLM reader do not depend on these weights, so they are perfectly stable here and cannot lose. The bar is therefore the study's cross-model agreement and the chance level from mismatched units. Most dangerous outcome: weak-regime readings shift while strong ones hold, which IN23 would then explain.

## Premise it tests

P1. A pass lets the study's 4-bit units inherit evidence gathered at full precision (public neuron descriptions, Llama Scope dictionaries, the source essay). A fail means every reading so far is specific to one quantised file.

## Biology analogue

Checking a low-coverage sequencing run against a deep one

## Cost

S. Six extraction passes of under an hour each. The bf16 builds are about 16 GB and may not fit beside their activations on a 24 GB laptop; if so, 8-bit is the reference. One day.

## Payoff

A stated precision limit for the instrument, and permission (or not) to reuse public neuron labels and dictionaries.

## Main risk

Likely an uneventful pass: Singh and Sajjad found no drastic neuron-level change from 4-bit quantisation in other models, though they did not track individual units' strongest examples.

## Reviewer note

Promoted from a seed: it is cheap, and IN24 and any reuse of full-precision labels depend on it. All six builds confirmed on Hugging Face. Weakness: covers MLX quantisation only, and prior work points to a pass, so it is informative mainly if it fails.

## Cross-area ranking

In the top twelve of 2 of 3 judges (what decides the premises fastest: rank 8; speed and cost: rank 7). A one-day gate that will probably pass, but if 4-bit readings do not match 8-bit or full precision then every number so far belongs to one quantised file and no published labels or dictionaries can be borrowed.

## Amendment, 10 November 2026

Compare the builds on what the instrument uses: each unit's 97th and 99.9th percentile thresholds at each precision, the overlap of the tokens in that range, and how often the sign of the gate input at a firing token flips. The companion paper reads only the tails of each neuron's range and does not state its own precision. A correlation over all sentences can pass while the tails move.

## Sources

- Singh and Sajjad 2025 (arXiv:2508.16785)
- Cammarata 2026
- study/results/robustness/nulls.json (grid kappa)
- Choi et al. 2024 (Transluce)
- Tigges et al. 2026, Neurons as an Interpretable Basis
