# RX-24: A writer's imprint across a language border

**Area:** Remix and influence · **Status:** design registered 8 November 2026, not run · **Priority:** 2.5 of 5

## Question

Do an author's distinctive units reappear in documented followers who wrote in another language, where shared words and function-word style cannot carry the signal?

## Prediction

Neurons place the documented influencer in the top 3 of 21 candidate authors for at least 30% of pairs (chance is 14%), at least 5 points more often than the best multilingual embedding and than always guessing the most famous candidate.

## What would refute it

Top-3 rate under 20%, or no higher than multilingual embeddings, translate-then-TF-IDF or the fame-only guess. 300 pairs give a standard error near 2.7 points: ample against chance, but only about 50% power for a 5-point gap over a rival, so a gap between 0 and 5 is undetermined. Do not run with under 250 pairs.

## Data

Wikidata 'influenced by' (P737): 32,322 statements, counted 2026-11-08, CC0. Pairs where both authors died before 1955, share no writing language and have Wikisource pages: 798 pairs, 364 followers. Only 166 such followers have Project Gutenberg author pages, so take texts from Wikisource dumps (public domain or CC BY-SA). Target 300 pairs, 20,000 words of prose per author, original language.

## Method

Choose each influencer's distinctive units on half of that author's text, against same-language authors of the same era. For each follower, rank the documented influencer among 20 decoys matched on language, era and fame (number of Wikipedia language editions) by how many of those units the follower's text carries. Statistic: top-3 rate, clustered by follower. Null: decoys permuted within era.

## Cheap rival it must beat

Multilingual sentence embeddings; machine translation to English, then TF-IDF and Burrows Delta (function-word stylometry); an LLM given both texts and the 21 names; always picking the most famous candidate; length; random units; mean-pooled residual stream. Most dangerous: the LLM, which may simply know that Baudelaire read Poe. Report obscure pairs separately to blunt that.

## Premise it tests

Both. P1: a unit that survives a change of language is not a word detector. P2: Qwen3 and Llama 3.1 should rank the same influencers; report their agreement against the random-unit floor. A fail bounds 'a person's influence' to style and translation.

## Biology analogue

Introgression: tracing one parent's alleles in a hybrid lineage.

## Cost

M: up to two weeks. About 600 authors at 20,000 words is a few hours of forward passes per model; assembling clean public-domain prose in several languages from Wikisource is the labour.

## Payoff

A per-person influence score that works across languages and is checked against documented influence; or a bound showing neurons add nothing to embeddings for it.

## Main risk

Too few pairs survive the prose and length filters. P737 records fame as much as influence, and followers often read the influencer in translation. The models have probably memorised these classics, so units may recognise the texts instead of their content; the LLM reader shares that flaw.

## Reviewer note

Gutenberg yields only 166 qualifying followers, below the design's own floor, so texts now come from Wikisource (364 followers). Added fame-matched decoys and a fame-only rival, because P737 favours famous influencers. Remaining: Wikisource prose coverage per author is unchecked and the expected signal is weak.

## Sources

- Wikidata P737 counts 2026-11-08
- Burrows 2002
- Hughes et al. 2012
- Wegmann et al. 2022
