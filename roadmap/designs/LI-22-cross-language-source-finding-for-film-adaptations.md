# LI-22: Cross-language source finding for film adaptations

**Area:** Lineage · **Status:** design registered 8 November 2026, not run · **Priority:** 4.4 of 5

## Question

When a film's plot summary is in German, French, Spanish or Chinese and its documented source's summary is in English, so that no words are shared, do neuron memes (story units read from a language model's neurons) find the source better than translating and counting words?

## Prediction

On 400 held-out pairs with names masked, neuron memes read in the film's own language rank the documented source first among 20 matched candidates at least 7 percentage points more often than machine translation followed by word matching, and no more than 5 points below the best embedding model.

## What would refute it

Neuron first-place accuracy is not above translate-then-word-matching (paired one-sided p at or above 0.05), or is more than 5 points below the best embedding model. 400 pairs give about 80% power for a 7-point gap when two methods disagree on 30% of pairs.

## Data

Wikidata 'based on' links (CC0). Films whose source has an English Wikipedia article: 6,690 with a French article, 5,220 German, 4,357 Spanish, 2,559 Chinese (Qwen only). Plots are CC BY-SA; the Tell Me Again dataset holds name-free summaries in five languages. 600 pairs, 15 plot sentences a side, one film per source: 200 development, 400 test. Coverage is counted first.

## Method

Each non-English plot is ranked against 20 English source plots: the true one and 19 decoys matched on medium, decade and genre, excluding same-author and same-series works. Score: share of the source's distinctive units found in the film; thresholds set per language. Settings frozen on development pairs. Null: shuffle source labels within groups. Pairs sharing translated eight-word runs are dropped.

## Cheap rival it must beat

Machine translation then word matching (BM25); neuron memes on the translated plot; multilingual embeddings (multilingual-e5, LaBSE); story-emb, an embedding trained on name-free summaries of one story; a local LLM reading both plots; plot length; random non-meme units; the averaged internal state. Most dangerous: the embeddings, built for this task. Losing to translate-then-match means the units are tied to language.

## Premise it tests

both. P1: a pass shows units carry story content when wording cannot; a fail, after the Batman tie, makes two corpora where neurons add nothing to words for descent. P2: units and settings chosen on English must work in other languages, and Qwen and Llama should rank the same sources.

## Biology analogue

Recognising the same gene across species despite silent spelling changes.

## Cost

M: one to two weeks on one laptop, no paid services. Roughly 1,800 plots; extraction under an hour per model; translating 600 plots and 8,000 LLM pair readings take most of the time, plus name masking in each language where Tell Me Again does not cover a pair.

## Payoff

A verdict on heredity outside Batman with over 40 times as many adaptations, where shared wording is ruled out by construction. It supersedes the deferred translation protocol by using summaries that people wrote in each language.

## Main risk

Multilingual embeddings win easily and neurons trail even translate-then-match; French and Spanish plots may be too short, leaving a German-only test; the models have memorised which famous film came from which book, so results are read on the obscure half; and story-emb was trained on these same Wikipedia summaries.

## Reviewer note

German and French counts confirmed. Added story-emb and neurons-on-translation as rivals, per-language thresholds, a same-author decoy ban, one film per source and a reuse audit. Remains: plot coverage unknown (a tiny sample found long German and short French plots); translation is itself a neural reader; famous pairs are memorised.

## Cross-area ranking

In the top twelve of 1 of 3 judges (upside: rank 6). A family tree across a language border on 400 real adaptations is the strongest possible form of the Batman heredity result, although neurons beating translate-then-match is perhaps a one-in-four chance.

## Overlaps with

Asks nearly the same question as PO-21, which the judges would run instead.

## Sources

- Wikidata query service counts, 2026-11-08
- Hatzel and Biemann 2024 (Tell Me Again; Story Embeddings)
- Chaturvedi et al. 2018
- preregistration/translation_protocol.md
