# PO-27: One film population, two Wikipedias: an Fst check

**Area:** Population memetics · **Status:** design registered 8 November 2026, not run · **Priority:** 3.5 of 5

## Question

When the same films are summarised in two Wikipedias, is the differentiation between the two sets of summaries (which should be zero) small beside the differentiation between national film industries, and smaller for units than for translated words?

## Prediction

Per-unit Fst (the share of a unit's frequency variance that lies between groups) between US and German productions agrees across English and German summaries (rank correlation at least 0.5), and for concept units the median ratio of industry Fst to summariser Fst is at least 2 and above that of translated words.

## What would refute it

Rank correlation below 0.3, or median ratio at or below 1 (the reader measures the summariser, not the film), or translated words reach as high a ratio (bootstrap over films, interval of the difference includes zero). With 1,000 films a side, sampling alone gives Fst near 0.0005.

## Data

Tell me again! (Hatzel and Biemann 2024): 96,831 summaries of 29,505 stories (25,860 films) from five Wikipedias, with English machine translations and name-masked versions; CC BY-SA 4.0. German covers 98% of stories and English 89%. Country of origin from Wikidata (CC0). Needs 1,000 US and 1,000 German-language productions with both summaries of ten or more sentences; count to confirm.

## Method

Read English summaries and machine-translated German ones, names masked, cut to ten sentences. Units, classes and bars fixed on 500 development films. Summariser Fst: the same films, English against German summary. Industry Fst: US against German productions within one Wikipedia. Sampling noise is subtracted; nulls shuffle group labels within decade and genre.

## Cheap rival it must beat

The same two Fst values for machine-translated word presence (translation wording should inflate its summariser Fst; if units do no better, P1 loses an easy case); an LLM's answers to 50 motif questions (most dangerous: probably the cleanest reader); multilingual-e5 dimensions; sparse-autoencoder features; random active units; length.

## Premise it tests

P1. Units that differ between industries but not between summarisers read content and are fit for population comparisons; if summariser Fst matches industry Fst, cross-population claims from summaries are unsafe. P2: per-unit ratios should agree between Qwen and Llama twins (rank correlation at least 0.5).

## Biology analogue

Fst between populations, with a technical-replicate control (the same individuals genotyped twice).

## Cost

S to M. Shares PO21's dataset; about 4,000 summaries, minutes of extraction per model; three days of analysis.

## Payoff

A list of units safe for comparing populations of works, the size of the summariser artefact, and the first Fst-style statistic for memes with a built-in zero point.

## Main risk

German Wikipedia summaries tend to be longer, so some summariser Fst is real difference in what gets told; machine translation adds its own style. Too few German productions may have long English summaries.

## Reviewer note

Added by the reviewer: the area names Fst but had only a low-power seed. The same films in two Wikipedias give a true zero for differentiation. Weakness: summariser Fst mixes wording with real differences in detail; film counts per industry are unconfirmed.

## Sources

- Hatzel and Biemann 2024
- Lewontin and Krakauer 1973
- Ross, Greenhill and Atkinson 2013
- Wikidata query service (2026-11-08)
