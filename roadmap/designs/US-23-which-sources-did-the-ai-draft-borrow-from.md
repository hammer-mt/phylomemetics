# US-23: Which sources did the AI draft borrow from?

**Area:** Uses · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

After an AI draft rewords and re-sets what it borrowed, can units identify which of twenty candidate sources it drew on, and which sentence it took?

## Prediction

On re-set drafts, neuron units recover the two true sources among 20 candidates with recall at 2 (the share of true sources ranked in the top two; chance 0.10) of at least 0.50 and within 0.05 of the best embedding, and find the borrowed sentence inside the true source within 0.05 of the best embedding's hit rate.

## What would refute it

Recall at 2 on re-set drafts is not above BM25 word matching, or is more than 0.05 below the best embedding (paired bootstrap over 300 drafts), or is below the pairwise LLM reader while costing as much or more.

## Data

Sources: plot sections of English Wikipedia films first released from 2025 (2,403 articles by a Wikidata count on 8 Nov 2026; about a third of film articles had a usable plot in the paused plan's crawl), after Llama 3.1's December 2023 cut-off, names masked, CC BY-SA 4.0. Drafts: 900 stories by two open models from other families than the readers.

## Method

Drafts are randomised to three conditions, 300 each: close paraphrase of two random sources; re-set (new setting and names, the sources' top TF-IDF terms banned, one random sentence per source to carry over); no-source control. Methods rank 20 candidates matched on genre and length, then the true source's sentences. Null: random units. The author blind-audits 60 drafts.

## Cheap rival it must beat

Longest shared word sequence, BM25 (only a floor: the ban handicaps it), sentence and story embeddings, an LLM shown the draft beside each candidate, the model's averaged internal state, text length. Story embeddings and the pairwise LLM reader are most dangerous. If they win on ranking, the remaining case for units is naming which story elements were carried over.

## Premise it tests

Assumes both. A pass supports heredity beyond words where descent is fixed by random assignment, which the Batman test could not offer. A fail leaves source auditing to embeddings and LLM readers.

## Biology analogue

Paternity testing with known parents.

## Cost

M. 900 generated drafts, under 1,000 source plots, 18,000 pairwise LLM reads on one laptop.

## Payoff

A tested way to show what an AI draft took from supplied material when no wording matches, for provenance and originality checks.

## Main risk

Machine-written drafts may borrow in ways people do not; a writer that ignores an assigned source adds label noise the audit has to catch; and a tie with embeddings at higher cost would leave units with no job here.

## Reviewer note

Beating BM25 was true by construction once source terms are banned, so the embedding margin is now the test. Added a randomly assigned borrowed sentence, writers from other model families and a corrected source count. Weakness: the likely result is a tie with story embeddings.

## Sources

- Gupta and Pruthi 2025
- Hatzel and Biemann 2024 (story embeddings)
- Wikidata film count, 8 Nov 2026
