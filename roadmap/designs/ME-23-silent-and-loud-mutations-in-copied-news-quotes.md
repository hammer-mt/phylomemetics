# ME-23: Silent and loud mutations in copied news quotes

**Area:** Old memetics, re-run · **Status:** design registered 8 November 2026, not run · **Priority:** 2.5 of 5

## Question

When a news quote changes by one word as it is copied, can neuron units tell meaning-keeping changes from meaning-changing ones, and are the meaning-changing versions reused less often?

## Prediction

On 600 hand-labelled natural swaps, the change in pre-chosen units separates silent swaps (meaning kept) from loud ones (meaning changed) with an AUC of at least 0.75 (0.5 is chance), within 0.05 of sentence-embedding similarity. Loud versions are then reused by another site at most 80% as often as silent ones.

## What would refute it

AUC under 0.65, or more than 0.05 below embeddings (standard error about 0.02). For spread: with at least 3,000 swaps, 500 of them loud, a loud-to-silent reuse ratio above 0.95 whose 95% interval excludes 0.8. With fewer swaps the spread test is declared uninformative, not passed.

## Data

MemeTracker phrase clusters (SNAP; one 220 MB file, reachable 2026-11-08; August 2008 to April 2009; no licence stated, so commit only IDs and derived numbers). One-word swaps mined with Lerique and Roth's method (code GPLv3); the number of swaps is unknown until mined. 800 swaps labelled by hand, 150 of them twice. Tuning pairs built from WordNet synonyms and antonyms.

## Method

Pairs share every word but one, so word overlap cannot separate them. Drop tense, number and spelling changes. Choose units on built pairs (synonym swap silent, antonym swap loud) and 200 labelled swaps; score on 600. The best tool labels all swaps; compare reuse by a later, different site within parent quote and week. Null: labels shuffled within parent.

## Cheap rival it must beat

Edit distance (constant by design), WordNet synonymy and word-vector similarity of the two swapped words, sentence-embedding similarity, an LLM judging both versions, random units, the mean-pooled residual stream. The LLM judge is most likely to win. The silent-to-loud split and the spread test then run on its labels and still give memetics a fidelity figure.

## Premise it tests

P1. A pass shows units respond to meaning in natural minimal pairs, where the study had 31 hand-made rewordings. A fail says units are too coarse for short texts. The spread result says whether meaning is protected in copying; Atran and the memeticists both expect that, so it does not separate them.

## Biology analogue

Synonymous against non-synonymous mutation (dN/dS)

## Cost

M. Model time is under a day because quotes are short; mining the swaps and hand-labelling 800 pairs take about three days for one person.

## Payoff

A first copying-fidelity figure for ideas as opposed to strings: what share of copying errors is silent, and whether meaning-changing errors are weeded out.

## Main risk

Most MemeTracker variants are truncations, so clean one-word swaps may number only a few thousand. Which version is the parent is inferred from timing, not documented. A loud change may be the correction of a misquote, which would spread more and hide selection.

## Reviewer note

Replaced PAWS (word-order pairs, the wrong kind) with built synonym and antonym swaps; changed the spread outcome to reuse by another site and said when it is uninformative. Remaining: an LLM judge will win the labelling, swap counts are unknown, and parent direction is inferred.

## Sources

- Leskovec, Backstrom and Kleinberg 2009 (MemeTracker)
- Lerique and Roth 2018
- Simmons, Adamic and Adar 2011
- Atran 2001
