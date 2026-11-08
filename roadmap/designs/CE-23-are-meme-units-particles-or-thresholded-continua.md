# CE-23: Are meme units particles or thresholded continua?

**Area:** Cultural evolution, re-run · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

When a story is passed on blended with a second parent, do unit values in the child behave like particles, as replicator memetics needs, or like a graded quantity that blends, as Henrich and Boyd's continuous model allows?

## Prediction

Memetics predicts that in free blends rewording-robust units fit segregation (child takes one parent's value) better than blending at least 20 percentage points more often than random directions in the residual stream, and at least 20% of units have two-peaked activations against under 5% of random directions. The slot arm must show segregation for at least 80% of slot units.

## What would refute it

The segregation share of units is within 10 points of random directions or of embedding components, or under 20% of units favour segregation on held-out blends. Then discreteness comes from the text and the pooling, not the unit, and no replicator is needed. 150 parent pairs, 8 children per arm; proportions over 500 units have error under 0.03.

## Data

Parents: 300 tales of 300 to 3,000 words from the 1,518 in Hagedorn and Darányi's Folktexts, labelled with ATU tale types (Zenodo 6575263, 10.8 MB; CC BY-SA 4.0 per its README), paired across types. Children written by two open models in three arms: free blend, slot copy, merged summary. Background: 20,000 sentences from the CMU Movie Summary Corpus (CC BY-SA).

## Method

Units chosen for rewording robustness on 50 development pairs, frozen. Per trait, compare held-out likelihood of segregation (mixture of parental values) and blending (weighted parental mean pulled toward the population mean); Hartigan dip test for two peaks. Slot and merged-summary arms calibrate both patterns. Null: children re-paired with wrong parents of similar length. Repeat with sparse-autoencoder features and at 8-bit.

## Cheap rival it must beat

The same comparison on TF-IDF term weights, principal components of sentence embeddings, random residual-stream directions pooled like units, Llama Scope features, random units and length; an LLM reader only codes which parent each event came from. Most dangerous: random directions, because a maximum over tokens can make anything look all-or-nothing. If they match units, Henrich and Boyd's case holds.

## Premise it tests

P1. Segregation and two peaks beyond what random directions show support 'a meme is a neuron firing' as more than a threshold. Blending supports treating neurons as a continuous trait measure and dropping replicator language; the paper's presence rule becomes a convenience.

## Biology analogue

Mendel's crosses against blending inheritance

## Cost

S. 3,600 generated children and two extraction passes; about two days. The two-peak test can be tried first on the study's saved activations if they are still on disk.

## Payoff

Decides which mathematics to use: particulate population genetics or quantitative genetics on continuous traits.

## Main risk

The verdict may describe how LLM writers blend, not how human authors do (see the seed on two-source works), and 4-bit activations may blur real peaks; the slot arm and the 8-bit check bound both.

## Reviewer note

Text is made of discrete events and unit values are maxima over tokens, so any detector looks particulate. Made the claim relative to random directions, added a merged-summary arm as the blending anchor, and replaced binary word presence (which cannot blend) with term weights. The LLM-writer weakness remains.

## Sources

- Henrich & Boyd 2002
- Claidière, Scott-Phillips & Sperber 2014
- Hagedorn & Darányi 2022
- He et al. 2024 (Llama Scope)
