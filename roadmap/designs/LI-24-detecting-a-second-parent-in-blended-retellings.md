# LI-24: Detecting a second parent in blended retellings

**Area:** Lineage · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

When a retelling is known by construction to merge two earlier versions, can neuron memes (story units read from a language model's neurons) show that a second parent exists and name it better than word overlap can, against the simpler view that every text has one parent?

## Prediction

Given one parent, neuron memes name the other first among the remaining earlier texts at least 12 percentage points more often than word overlap does, and their second-parent score separates blended from single-parent texts with AUC (ranking accuracy; 0.5 is chance) of at least 0.70.

## What would refute it

On 120 sealed blended texts, second-parent accuracy is not above the word-overlap version (paired one-sided p at or above 0.05), or the AUC is below 0.60. 120 blended and 276 single-parent texts give about 80% power for a 12-point gap at 30% disagreement and an AUC standard error near 0.03.

## Data

Built on the LI21 traditions (CC BY-SA 4.0): in each tale at the two heaviest rewording levels, 10 extra texts merge two versions at least four steps apart, keeping a planted change from each: 40 development, 120 sealed. Natural check, descriptive only: of 341 Wikidata films with two or more sources (reviewer's query, 2026-11-08), at most 39 blend unrelated stories.

## Method

Each method receives the blended text and one true parent, the same for all methods, and ranks every other earlier text as the second parent after discarding units shared with the given parent. Existence test: the best remaining score, for blended against single-parent texts of the same generation. Settings frozen on development tales. Null: shuffle second-parent labels within a tradition.

## Cheap rival it must beat

The same procedure with rarity-weighted words (TF-IDF), with embeddings after subtracting the given parent's vector, a local LLM asked which other text was used, text length, random non-meme units, the model's averaged internal state and sparse-autoencoder features. Most dangerous: words, because leftover distinctive words point at the second parent. If words tie, sparseness matters, not neurons.

## Premise it tests

P1. Separate units can be split between two parents in a way one blended vector cannot, so this is where a discrete unit should earn its keep. A pass gives a detector of mixed descent validated on known blends and supports networks over trees; a fail leaves the study's failed tree result without a remedy.

## Biology analogue

Hybrid detection and gene-flow tests, validated on known hybrids.

## Cost

S once LI21 exists: about 2 days on one laptop. 160 extra blended retellings with survival checks, extraction in minutes, about 3,000 LLM readings. The natural check adds about 120 plots to fetch and mask.

## Payoff

A tested rule for saying 'this text has a second parent, and it is this one', which the Batman films need, and evidence on whether discrete units handle mixed descent better than a single vector.

## Main risk

Blends built inside one tale may be too easy or too hard for every method and say little about franchises, and the natural set of at most 39 films is too small to confirm anything.

## Reviewer note

Rebuilt. The Wikidata sample cannot work: of 341 two-source films, 131 have one source based on the other, 76 share a series and at most 39 are real blends. Now uses built two-parent texts in LI21, with the films as a descriptive check. Remains: depends on LI21; blends are artificial.

## Overlaps with

Asks nearly the same question as RX-27, which the judges would run instead.

## Amendment, 10 November 2026

From the field survey (arXiv:2611.08438): the learned-feature arm decides whether sparseness or neurons do the work here, so it must be robust. The survey reports that often fewer than half of dictionary features recur across training seeds, and that the feature set changes with dictionary size. Run that arm at two dictionary sizes and report both. A loss by one dictionary is not a loss by learned features.

## Sources

- Roos and Heikkilä 2009
- Wikidata query service counts, 2026-11-08
- Greenhill, Currie and Gray 2009
- Temkin and Eldredge 2007
- Barez, Wu et al. 2026, Understanding AI from the Inside (arXiv:2611.08438)
