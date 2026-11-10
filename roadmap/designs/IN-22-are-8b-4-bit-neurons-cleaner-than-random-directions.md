# IN-22: Are 8B 4-bit neurons cleaner than random directions?

**Area:** The instrument · **Status:** design registered 8 November 2026, not run · **Priority:** 3.5 of 5

## Question

Do neuron poles in the study's quantised 8B models pass a blind coherence test more often than random directions, and can the test see planted mixtures?

## Prediction

In an odd-one-out test (two sentences from a unit's strongest examples plus one from another unit; chance 33%), accuracy is at least 20 points higher for neuron poles than for random directions in the same layers, and planted two-neuron mixtures score at least 15 points below single neurons.

## What would refute it

Neuron minus random-direction gap under 10 points; or mixtures within 5 points of single neurons (the test cannot see mixing, so no cleanliness claim is possible); or a word-overlap picker matches the blind grader on neuron items (coherent only as word detectors); or a weight-shuffled model scores within 10 points of the trained one.

## Data

FineWeb sample, about 10 million tokens of web text (ODC-By), for activations; no labels needed. Per model: 300 random poles stratified by layer third, 300 random unit-norm directions in the same MLP hidden spaces, 150 sums of two unrelated poles, each scaled to its own maximum. Models: Qwen3-8B and Llama 3.1 8B, 4-bit MLX builds.

## Method

Per unit, 6 triplets from its 20 strongest sentences, peak token marked, odd sentence length-matched. A two-job mixture then has no right answer half the time, so mixing shows. Blind grader: an LLM from another family; a human does 200 items. Accuracy by unit type and layer third, bootstrapped; 300 per arm puts 10 points at 4 standard errors.

## Cheap rival it must beat

A word-overlap picker (choose the sentence least similar by TF-IDF) and an embedding picker on the same items; random directions; the weight-shuffled model. Most dangerous: the word-overlap picker. If it matches the grader, the neurons are coherent only as word detectors, which is still the answer P1 needs.

## Premise it tests

P1. A clear gap on these exact models licenses their neurons as units. No gap, or a test blind to mixtures, removes the source essay's support for the study's models.

## Biology analogue

Mock community of known composition, with positive and negative controls

## Cost

M. One extraction pass of about two hours per model plus one for the shuffled model, about 9,000 grader calls (modest API spend or a local 9B model), 3 hours of human checking; about a week.

## Payoff

The first cleanliness figure for 8B 4-bit models, by layer, with a noise floor and a sensitivity check that the source essay lacked and its outside retest failed.

## Main risk

Coherence is not concept detection: name and token detectors pass. Ten million tokens may not reach many neurons' strong regime, so the 20 strongest sentences are partly weak firings.

## Reviewer note

Replaced the five-way intruder test: with four unit sentences a two-job mixture still has an obvious odd one out, so the mixture control could not fail. Triplets fix that. The word-overlap picker is now a falsifier. Remaining: an LLM grader, and coherence rewards word detectors.

## Cross-area ranking

3 of 3 judges thought the area reviewer rated this too high.

## Amendment, 10 November 2026

Build triplets separately from three bands of each unit's own range (97th to 99th percentile, 99th to 99.9th, above), not only from the 20 strongest sentences. In the companion paper, random directions inside an expert kept one readable main theme at every band; neurons differed by narrowing as firing strengthened. Add its label-free test: keep each token's top 8 neurons, or 8 random directions, per layer and compare how much of the layer's effect on the output survives. From the field survey (arXiv:2611.08438): its protocol fixes principal components beside random directions as baselines, and it lists a coherent label that the model does not use as a classic pitfall. Add 300 principal components of each MLP hidden space as a further unit type. Use two graders from different model families and report their agreement. Report the result as plausibility only.

## Sources

- Cammarata 2026
- IRSI 2026 retest note
- Paulo and Belrose 2025 (arXiv:2507.08473)
- Heap et al. 2025 (arXiv:2501.17727)
- Tigges et al. 2026, Neurons as an Interpretable Basis
- Barez, Wu et al. 2026, Understanding AI from the Inside (arXiv:2611.08438)
