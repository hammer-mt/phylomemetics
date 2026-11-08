# CE-21: Do neuron units track what survives retelling?

**Area:** Cultural evolution, re-run · **Status:** design registered 8 November 2026, not run · **Priority:** 5.0 of 5

## Question

When people retell a story in their own words, do fixed neuron units predict which statements survive, and recover the published content and prestige effects, better than cheap readers?

## Prediction

On 87,421 human-coded cells from 163 people's spoken recall, unit persistence predicts present-or-absent with AUC (ranking accuracy; 0.5 is chance) at least 0.05 above the better of content-word overlap and BM25 word matching, and at least 0.65 where under half a statement's words appear. Unit-coded recall reproduces Berl et al.'s five significant effects: social, survival, negative, biologically counterintuitive, prestige.

## What would refute it

AUC gain over word overlap below 0.03, or AUC under 0.60 on the low-overlap cells, or fewer than 4 of the 5 published effects with the right sign. Sample: 87,421 cells, 538 statements, 163 people, two stories. Bootstrap by person and by statement; the standard error should be near 0.01, to confirm by simulation.

## Data

Berl et al., OSF 72v3f (CC BY 4.0): 326 transcripts of spoken recall, 538 statements coded present or absent, speaker prestige randomised. Acerbi and Stubbersfield, OSF 6v2ps: ChatGPT chains on the same two stories and three older materials, two coders, about 9,000 cells. Breithaupt et al., OSF jr2py: 348 human retellings. The last two state no licence: commit scores only.

## Method

Units chosen for rewording robustness on Breithaupt's human retellings, frozen. A statement's signature is the units firing on its words in the source story; its score, the share still firing in the recall. Cut-offs set on one story, tested on the other. Mixed logistic model on content type and prestige. Nulls: random firing-rate-matched units; recalls shuffled across people; length covariate.

## Cheap rival it must beat

Content-word overlap, BM25, best-match sentence-embedding cosine, an 8B model asked whether each statement is present (20,000-cell sample), recall length, random matched units, mean-pooled residual stream. Most dangerous: the LLM reader, since judging paraphrase is its home ground. If it wins, recall can be coded automatically and neurons are not needed for this work.

## Premise it tests

P1. A pass shows units follow ideas through human rewording against human coders, the regime the paper says it needs. A fail on the cleanest public paraphrase data with human coding leaves P1 little as a measurement claim.

## Biology analogue

One passage, many replicate lines

## Cost

S. No generation needed. Two extraction passes over about 900 short texts and 20,000 short reader calls on one laptop.

## Payoff

An error rate for unit readout under human paraphrase against human coders (their own agreement, kappa 0.74 on a re-coded tenth, sets the ceiling), and a check that unit-coded recall gives the same content and prestige effects as hand coding.

## Main risk

Only two stories, so statements share names and setting. Statements are finer than sentences, so neighbours may share one signature. The LLM reader probably wins outright.

## Reviewer note

Moved the main test to Berl et al.'s human recall (87,421 coded cells), because the ChatGPT chains keep most words: a rough parse put word overlap near AUC 0.85 there, 0.78 on human recall. Dropped the fresh-chain test, which had no independent coder. Weakness: two stories.

## Cross-area ranking

In the top twelve of 3 of 3 judges (what decides the premises fastest: rank 3; speed and cost: rank 4; upside: rank 12). The only cheap test against human coders of human rewording (87,421 coded cells, nothing to generate), so a failure cannot be blamed on machine-written items; judge it against the best embedding and the LLM reader, because the registered pass bar only asks units to beat word overlap.

## Sources

- Berl et al. 2021
- Acerbi & Stubbersfield 2023
- Breithaupt et al. 2024
- Mesoudi, Whiten & Dunbar 2006
