# IN-24: One neuron, a few, a direction or a feature?

**Area:** The instrument · **Status:** design registered 8 November 2026, not run · **Priority:** 4.5 of 5

## Question

How many neurons does a story concept need, and does a random direction, a learned direction or a dictionary feature detect it better?

## Prediction

Averaged over 30 concepts, the best single neuron's held-out AUC (chance 0.5) on concept-without-its-words against words-without-the-concept is within 0.05 of the best 32-neuron set, and neurons beat random directions at equal count by at least 0.05.

## What would refute it

One neuron trails 32 by more than 0.10 on average (the unit is a set); or random directions match neurons at equal count; or a single dictionary feature beats the single neuron by 0.05 on average. Each unit type is scored after subtracting its own shuffled-label result.

## Data

The IN21 pairs and decoys. Llama Scope sparse autoencoders and transcoders (learned dictionaries of rarely active features; a transcoder imitates an MLP layer): 256 dictionaries on Llama-3.1-8B base, 32K and 128K features, Hugging Face, front card Apache-2.0. Gemma Scope transcoders for Gemma-2-2B, CC BY 4.0, as a small-model arm.

## Method

All unit types read at the same four depths of Llama-3.1-8B base at 8-bit, the model the dictionaries were trained on; the study's 4-bit instruct neurons are an extra arm. Rank units on development data and average the top k (1 to 128) without fitted weights. Add a difference-of-means residual direction. Held-out mixed-cell AUC against k, bootstrapped over concepts.

## Cheap rival it must beat

Logistic probe on a sentence embedding, TF-IDF classifier, zero-shot LLM reader, mean-pooled residual stream, all on the same mixed cells. Most dangerous: the difference-of-means direction, which beat dictionary features at detection in AxBench. If it wins, the meme unit should be a direction and neurons remain a cheap readout.

## Premise it tests

P1. Decides whether 'a meme is one neuron' survives, becomes 'a small set', or gives way to directions or features; fixes the unit for later designs.

## Biology analogue

Gene, operon or pathway: which level carries the trait

## Cost

M. Porting dictionary weights to MLX and checking reconstruction is the work; extraction takes minutes. About 1 to 2 weeks on the laptop.

## Payoff

An accuracy-against-size curve per unit type and a decision on which unit the rest of the registry should freeze.

## Main risk

With about 100 development sentences per concept, larger sets are chosen noisily, which flatters single neurons. The dictionaries may reconstruct the 8-bit model poorly; report explained variance first.

## Reviewer note

Per-concept AUC on 20 against 20 items has an error near 0.08, so 'within 0.05' is now a pooled claim. Fitted probes on 100 sentences would overfit; replaced by unweighted top-k averages. Comparison moved to the base model the dictionaries were trained on. Depends on IN21.

## Cross-area ranking

In the top twelve of 1 of 3 judges (what decides the premises fastest: rank 5). Decides whether the unit should be one neuron, a small set or a direction before the other registered designs freeze the wrong one, and its cheap arms need only IN-21's sentences.

## Amendment, 10 November 2026

Add a unit type the companion paper points to: sets of 2 to 8 poles that must all be in their own strong range in the same sentence (an AND rule), beside the planned averages. The paper suggests that ideas with no neuron of their own are carried by neurons that fire together and mean the overlap of their meanings. For each winning dictionary feature, report whether it copies one neuron.

## Sources

- Gurnee et al. 2023
- Arora et al. 2026
- He et al. 2024 (Llama Scope)
- Wu et al. 2025 (AxBench)
- Tigges et al. 2026, Neurons as an Interpretable Basis
