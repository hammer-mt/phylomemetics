# RX-23: Atypical meme pairings and clicks, topic held fixed

**Area:** Remix and influence · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

Within one randomised headline test, does the headline whose meme units are paired in ways rarely seen in earlier headlines win more clicks?

## Prediction

Yes, modestly. With test and image held fixed, one standard deviation more rare-pair atypicality raises click odds by at least 1%, after controls for length, words, word pairs and single neuron units. A squared term for an inverted U adds nothing detectable.

## What would refute it

The 95% interval for the rare-pair odds ratio lies wholly below 1.005, or includes 1 while word-pair atypicality passes. Robertson et al. 2023 resolved a 2.3% effect per negative word on these tests, so 1% should be detectable; confirm the standard error on the exploratory set before scoring.

## Data

Upworthy Research Archive (Matias et al. 2021): 32,487 randomised headline tests, 150,817 arms, 2013 to 2015, CC BY 4.0, OSF node jd64p. Fit on the exploratory set (4,873 tests); score once on the confirmatory set (22,743). Drop the faulty-randomisation window (about 22%) and compare only arms sharing an image: roughly 9,000 usable confirmatory tests, to confirm.

## Method

Fix the 2,000 most frequent units on exploratory headlines. Score each unit pair in a headline by how often it co-occurred in earlier headlines against a shuffle keeping unit frequencies and month (Uzzi's z-score). Model clicks per impression with a fixed effect per test, adding feature sets in nested steps. Null: arms permuted within test.

## Cheap rival it must beat

Word-pair atypicality; single neuron units without pairs; TF-IDF difference; embedding distance from earlier headlines; an LLM asked which headline wins and which is more surprising; length and punctuation; random units; sparse-autoencoder features (Movva 2025). Most dangerous: word pairs, because headlines are short. If they win, recombination is tested anyway and neurons are surplus.

## Premise it tests

Assumes both. A pass shows that pairs of neuron units carry combination structure that word pairs and a bag of single units miss. A fail while word pairs pass says recombination matters but this unit does not. Repeat in Llama to see if the pair scores agree.

## Biology analogue

Epistasis: the fitness of gene combinations beyond single genes.

## Cost

S: under 2 days. 150,817 headlines is about 40 minutes of forward passes per model on the pre-registration's timing, which is optimistic; the LLM rival on a subsample is the dearest part.

## Payoff

A topic-controlled test of 'familiar base plus one surprise' (Uzzi; Berlyne's inverted U), and a pair-novelty score that could be used when drafting.

## Main risk

A headline holds too few units for pair statistics to mean much. Novelty is not randomised, only the audience is, so atypical headlines may differ in other ways. Sparse-autoencoder features have already been applied to this archive, so the ground is not new.

## Reviewer note

Arms also vary the image, so compared arms must share it; usable tests fall from 17,700 to about 9,000. Switched from winner-picking accuracy to within-test click odds for power. Added single units as a baseline that pairs must beat. Remaining: the effect is probably tiny.

## Sources

- Matias et al. 2021 (Upworthy Research Archive)
- Uzzi et al. 2013
- Robertson et al. 2023
- Movva et al. 2025
