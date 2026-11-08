# RX-22: Undeclared influence, checked against the written record

**Area:** Remix and influence · **Status:** design registered 8 November 2026, not run · **Priority:** 4.4 of 5

## Question

Can a method tuned on declared 'based on' and 'inspired by' links find influence between works that Wikidata does not declare but the written record confirms?

## Prediction

On held-out loose links ('inspired by', 'references work'), neurons rank the true source in the top 5% of matched decoys at least 5 points more often than the best cheap rival. At least 10% of their top 1,000 undeclared pairs are confirmed by the record, against under 2% for random matched pairs.

## What would refute it

Loose-link advantage under 2 points, or its 95% interval includes zero (1,000 pairs give a standard error near 2 points; do not run with under 500). Or the confirmed share of the top 1,000 is no higher than the best rival's. After E2, a second failure would settle that neurons add nothing to words for tracing influence.

## Data

Wikidata links with English Wikipedia articles at both ends, counted 2026-11-08, CC0: based on (P144) 30,685; inspired by (P941) 3,713; references work (P8371) 1,239. Plot sections, CC BY-SA, names masked. Sample 6,000 pairs with plots at both ends, loose links oversampled: 2,000 for development, 4,000 held out. Non-plot article text is the documentary record.

## Method

Score pairs by hub-adjusted share of the source's rare units (the E2 statistic). Test 1: rank held-out true sources among decoys matched on decade, medium and genre, per link type. Test 2: for each method's top 1,000 unlinked pairs with different creators and series, an LLM auditor must quote the article sentence naming the source (200 hand-checked). Null: within-decade permutation.

## Cheap rival it must beat

TF-IDF and BM25 on the same masked plots; e5 sentence embeddings and story-emb, which was built to survive name replacement; an LLM reading both plots (shortlists only); text length; random non-meme units; mean-pooled residual stream. Most dangerous: story-emb and the LLM. If they win, undeclared-influence search still works, just without neurons.

## Premise it tests

P1. A pass is the first evidence that neuron units trace influence where plots are not simply retold. A fail leaves P1 with no practical content for descent. For P2, report how many of the top 1,000 pairs Qwen and Llama share, against the random-unit floor.

## Biology analogue

Detecting horizontal gene transfer, validated on known transfers.

## Cost

M: about two weeks. Roughly 20,000 plots, a few hours of forward passes per model; the LLM reader on shortlists, the auditor on about 5,000 pairs and a 200-pair hand check are the main cost.

## Payoff

A search tool for unacknowledged sources with a known hit rate, tested on thousands of works beyond one franchise; or a clear finding that embeddings already do the job.

## Main risk

Similarity is not influence: top pairs may be genre twins, and critics note resemblance, so only statements that the makers drew on the source count. The record misses unwritten influence. Models may recognise famous plots despite masked names. Outputs are candidates, never accusations of copying.

## Reviewer note

Dropped the 'lowest quarter of word overlap' stratum: selecting on a rival's score makes that rival lose by construction. Strata are now link types. Audit made checkable and enlarged; same-franchise and same-creator pairs excluded. Remaining: the number of loose links with plots at both ends is unconfirmed.

## Cross-area ranking

In the top twelve of 1 of 3 judges (upside: rank 4). It is the closest thing to the tool Mike wants: a search for unacknowledged sources across thousands of works with a hit rate checked against the written record, and it produces a usable list whichever reader wins.

## Sources

- Wikidata query service counts 2026-11-08
- Chang et al. 2024
- Hatzel and Biemann 2024 (story-emb)
- Claidière et al. 2014
