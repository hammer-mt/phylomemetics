# ME-21: Scientific memes along citations, with shared words controlled

**Area:** Old memetics, re-run · **Status:** design registered 8 November 2026, not run · **Priority:** 3.5 of 5

## Question

Does a paper carry the neuron units of a paper it cites more than those of an uncited paper whose wording is just as close?

## Prediction

On held-out pairs where an uncited same-year decoy is matched to the cited paper on TF-IDF similarity (word overlap weighted by rarity; within 0.02), units pick the cited paper at least 60% of the time (chance 50%), and at least 2 points more often than random units with the same firing rate.

## What would refute it

Unit accuracy under 55%, or within 1 point of random units, on 20,000 held-out pairs from at least 4,000 citing papers (standard error roughly 0.5 points when resampled by citing paper). A second failure after Batman would settle that neurons add nothing to words for descent.

## Data

SNAP cit-HepTh: 27,770 physics papers with abstracts and all 352,807 citation links among them, 1993 to 2003 (checked 2026-11-08; no licence stated, so commit only IDs). Second field: about 30,000 biomedical abstracts with links from OpenAlex (CC0 per its documentation; needs a free key, anonymous access was refused on 2026-11-08).

## Method

Unit: a feed-forward neuron (Llama 3.1 8B, Qwen3 8B), present when it fires strongly. Re-derive Kuhn's propagation score for units and n-grams. Pick the 2,000 best-propagating units on early links. On later links between papers unseen in tuning, sharing no author, count units shared with the cited paper and with the decoy. Repeat without word-detector units. Null: time-preserving link shuffle.

## Cheap rival it must beat

TF-IDF and BM25 (near 50% by design), Kuhn's n-gram score, a general sentence embedding, a citation-trained embedding (SPECTER2) as ceiling, an LLM reading 2,000 pairs, length, random units, the model's averaged internal state, open sparse-autoencoder features if available. The general embedding is most dangerous: popular ones were trained on citation pairs. If it wins, heredity past words stands, without units.

## Premise it tests

P1. A pass is the first evidence on natural text at scale that units follow inherited ideas past the wording. A fail, added to Batman, leaves P1 without practical content for descent. Agreement between the two models bears weakly on P2.

## Biology analogue

Parent-offspring resemblance against relatedness-matched controls

## Cost

M. About 400,000 sentences, a few hours per model on a laptop; most of the two weeks goes on assembling links and matching decoys.

## Payoff

Whether Kuhn's finding survives word matching when it is measured in units, and a reusable heredity benchmark with tens of thousands of documented links, the scale Batman lacked.

## Main risk

A citation is weak evidence that an idea was inherited; most references are background. Matching on TF-IDF leaves shared subtopic; a second decoy set, co-cited with the source by a third paper, tests that. Close decoys exist mainly for loosely related citations, so the test pairs may not be typical.

## Reviewer note

Added a closed, key-free corpus (cit-HepTh) so Kuhn's score can be re-derived; split by paper, dropped shared-author pairs, added random-unit and co-cited-decoy checks. Remaining: a citation is weak evidence of inheritance, and embeddings trained on citation pairs will probably beat units.

## Cross-area ranking

2 of 3 judges thought the area reviewer rated this too high.

## Sources

- Kuhn, Perc and Helbing 2014 (arXiv:1404.3757)
- SNAP cit-HepTh
- OpenAlex
- Edmonds 2002, Three challenges for the survival of memetics
