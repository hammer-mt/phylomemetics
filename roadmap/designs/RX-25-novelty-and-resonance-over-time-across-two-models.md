# RX-25: Novelty and resonance over time, across two models

**Area:** Remix and influence · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

Do novelty, transience and resonance scores built from neuron units predict later citations beyond word-based versions, and do two models give the same scores?

## Prediction

Neuron resonance adds at least 0.01 AUC (ranking accuracy; 0.5 is chance) over TF-IDF, topic-model and embedding resonance together with venue and team size, and adds more with a 10-year forward window than with 3. After removing TF-IDF resonance, Qwen and Llama scores rank-correlate at least 0.1 above scores from random units.

## What would refute it

Added AUC under 0.005 with an interval including zero (15,000 test abstracts; paired standard error about 0.004, to confirm by bootstrap): no use beyond words. Or residual cross-model correlation no higher than for random non-meme units, which agree almost freely: the score carries nothing specific to memes.

## Data

OpenAlex (CC0; the API now needs a free key with a daily budget): about 45,000 abstracts from one field, 2000 to 2025, with reference lists, venue and citations counted to 2025. Abstracts average about 8 sentences. Years 2000 to 2007 fix units and settings; 2008 to 2015 are scored once; later years serve only as the future pool.

## Method

Novelty is an abstract's mean distance from abstracts of the previous 3 years; transience is its distance from the next 3 (or 10); resonance is novelty minus transience (Barron 2018). Predict top-10% citations within field and year with nested logistic models. Null: abstracts shuffled within year. Fields rename ideas, so word scores should fade with the longer window.

## Cheap rival it must beat

TF-IDF backward and forward similarity (Kelly 2021); topic-model resonance (Barron 2018); sentence-embedding resonance; an LLM rating novelty from the abstract (subsample); abstract length; random units; mean-pooled residual stream. Most dangerous: TF-IDF, which already tracks patent quality. If it wins, resonance works and neurons are surplus; the cross-model figure still tests P2.

## Premise it tests

Both. P1 gains only if the neuron score adds prediction where vocabulary has turned over. P2 gains if two models' scores agree beyond what shared words and random units explain; the study's one beyond-words result held in Llama and failed in Qwen.

## Biology analogue

Fate of a new mutation: lost, transient or fixed in descendants.

## Cost

M: about 4 days. API pull of 45,000 abstracts, under two hours of forward passes per model, then regressions.

## Payoff

A model-independent influence score for any dated body of text, or a finding that the TF-IDF ratio is all one needs.

## Main risk

Citations reward fame and venue, so every text measure adds little. Later papers cite what they resemble, so resonance partly restates citation. Resonance in a 3-year window rewards joining a trend as much as starting one.

## Reviewer note

The sample ended in 2015, leaving scored years without later abstracts; the pool now runs to 2025. The cross-model bar was an absolute 0.3, which random units would pass; it is now a margin over random units. Venue and team size added. Remaining: 'TF-IDF suffices' is the likely result.

## Sources

- Barron et al. 2018
- Kelly et al. 2021
- Vicinanza et al. 2022
- OpenAlex
