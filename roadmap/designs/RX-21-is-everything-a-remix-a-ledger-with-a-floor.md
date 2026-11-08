# RX-21: Is everything a remix? A ledger with a floor

**Area:** Remix and influence · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

Once measurement noise is removed, what share of a film plot's meme units and unit pairs had never appeared in any earlier plot, and is that more than chance once release dates are shuffled?

## Prediction

Remix nearly holds. Under 1% of a late film's repeatable units are first appearances. Yet new memes exist: recurring units born after 2000 number at least twice the count found when release years are shuffled. New pairings of old units exceed their shuffled count by at least 1.5 times.

## What would refute it

Unit birth excess under 1.2 with a bootstrap interval including 1: no detectable invention, or a blind instrument. First-appearance share over 5%: remix fails. Stop early if two-summary agreement (kappa) is under 0.4 or within 0.1 of mismatched films. Check on 1985 to 1999 births that at least 200 born units exist.

## Data

Tell Me Again (Hatzel and Biemann 2024): 96,831 plot summaries of 29,505 stories from five Wikipedia language editions, machine-translated to English, with Wikidata year and genre. Sample 8,000 films with two or more summaries. Wikipedia text is CC BY-SA; the dataset's own licence is unconfirmed. About 13% of summary pairs are direct translations; drop near-duplicates.

## Method

Set each unit's presence bar on 1,500 unscored films. A unit is in a film only if present in two differently worded summaries. Count units and pairs seen in two or more films whose first appearance falls after 2000. Null: release years permuted within genre and length band. Statistic: observed-to-shuffled ratio (birth excess), bootstrapped over films.

## Cheap rival it must beat

The same ledger with word stems and stem pairs, Llama Scope sparse-autoencoder features, random non-meme units and embedding clusters; a reconstruction score (Lu 2025); an LLM listing what is new (500 films); length. Most dangerous: word stems, since new things bring new words. If stems give the same excess, the finding stands without neurons.

## Premise it tests

P1. If presence does not repeat across two wordings better than word stems do, neuron novelty scores are uninterpretable and the unit adds nothing here. Running Qwen and Llama shows whether two models call the same units new, a first P2 check for novelty.

## Biology analogue

De novo genes versus duplication and recombination.

## Cost

M: about a week. Roughly 20,000 summaries, a few hours of forward passes per model; the permuted ledger over unit pairs is the heavy computation.

## Payoff

A version of 'everything is a remix' that can fail, with numbers for inherited, new and newly combined; and the first test-retest reliability figure for meme presence, which every novelty measure depends on.

## Main risk

Summaries in different language editions may share a source, inflating repeatability. A unit seen once cannot be told from noise, so only recurring units count. With 8,000 films, 'new' can still mean 'not sampled'. Birth excess shows time structure, which fashion cycles also produce.

## Reviewer note

The earlier-to-later ratio was 1 by identity (every unit has one first and one last appearance); replaced by a date-shuffle null on recurring units. Tell Me Again now supplies every film. Remaining: dataset licence unconfirmed; the two summaries are not fully independent.

## Sources

- Youn et al. 2015
- Lu et al. 2025
- Hatzel and Biemann 2024 (Tell Me Again)
- Sreenivasan 2013
