# PO-24: Are linked memes inherited together, beyond chance co-occurrence?

**Area:** Population memetics · **Status:** design registered 8 November 2026, not run · **Priority:** 2.5 of 5

## Question

Do pairs of units that co-occur across a population of films also pass together from a source to its adaptation more often than their separate transmission rates and ordinary co-occurrence predict?

## Prediction

For linked pairs (top 1% of co-occurrence after controls) that peak in different scenes, the odds ratio of joint retention in true source-adaptation pairs is at least 1.3 times that in matched unrelated pairs, after dividing out the same figure for unlinked pairs, and larger than for frequency-matched linked word pairs. Up to 760 held-out sources before length filters.

## What would refute it

Ratio at or below 1.1 with the 95% interval's upper end under 1.3 (bundles are co-occurrence, not co-inheritance), or linked word pairs show an equal or larger ratio (interval of the difference includes zero). Intervals by bootstrap over sources.

## Data

Tell me again! (Hatzel and Biemann 2024; 96,831 name-masked summaries of 29,505 stories; CC BY-SA 4.0). Population panel: 5,000 English film summaries outside any documented pair. Transmission panel: the 1,363 film-source pairs (958 sources) that Wikidata 'based on' links (CC0) join inside it (reviewer's join, 2026-11-08); 20% of sources for development.

## Method

Merge units that co-fire within sentences. On the population panel, correlate presence pairwise after removing genre, decade, length and 20 principal components; keep linked pairs from different layers with different best-matched words whose top sentences in the source lie three or more apart, plus frequency-matched unlinked pairs. Where a held-out source carries both units, compare odds ratios of joint retention.

## Cheap rival it must beat

Word pairs linked by pointwise mutual information; LDA topic pairs; embedding-cluster pairs; an LLM's motif answers paired the same way; random non-meme unit pairs. All under the same different-scene rule. Most dangerous: word pairs, because topic vocabulary travels together. A tie means memeplexes are a text statistic.

## Premise it tests

P1. Co-inheritance beyond word bundles would show units capture inherited structure that vocabulary misses; a tie or a null says the memeplex is co-occurrence. The test only has meaning if single units are heritable (PO21).

## Biology analogue

Linkage disequilibrium, co-segregation and hitch-hiking.

## Cost

M. Shares PO21's dataset and extraction (about 9,000 summaries, an hour per model) plus sentence positions for a 5,000-unit panel; one to two weeks.

## Payoff

A list of meme bundles that travel together down documented lineages, each with a chance baseline: the first empirical test of the memeplex idea.

## Main risk

Leftover sub-genre structure makes any linked pair travel together when an adaptation keeps its source's genre, for words as much as for units, so a tie is the likely result.

## Reviewer note

A probability excess grows mechanically with retention rates: replaced by an odds-ratio contrast against unlinked pairs. Added a different-scene rule (same-sentence pairs co-transmit trivially), principal components and merging. Remaining: residual genre structure; word pairs will probably tie; depends on PO21.

## Sources

- Maynard Smith and Haigh 1974
- da Silva and Tehrani 2016
- Hatzel and Biemann 2024
- Wikidata P144 links (query service, 2026-11-08)
