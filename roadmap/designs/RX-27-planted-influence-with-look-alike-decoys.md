# RX-27: Planted influence with look-alike decoys

**Area:** Remix and influence · **Status:** design registered 8 November 2026, not run · **Priority:** 4.5 of 5

## Question

When a writer is shown a source and told to copy, transform or combine it, can neurons name that source once the wording no longer resembles it?

## Prediction

In the deep-transform condition (new setting, names and wording; content-word overlap with the source no higher than with look-alikes), neurons rank the true source first among 20 candidates for at least 40% of stories (chance is 5%), at least 10 points above the best embedding. For close copies every method is near ceiling.

## What would refute it

Deep-transform accuracy under 15%, or no higher than the best of story-emb, e5 and the LLM judge. 200 held-out stories per condition give a standard error near 3.5 points, enough to see a 10-point paired gap. Stories written with no source must put any candidate first at the 5% chance rate.

## Data

Built. 250 source plots of little-known films from the CMU Movie Summary Corpus (42,306 Wikipedia plots, CC BY-SA). Each gets 19 look-alike decoys of the same genre, decade and length, never shown to the writer. Two LLM writers from families other than the readers produce 1,250 stories: copy, transform, deep transform, combine two sources, or no source.

## Method

Fix units and settings on 50 development sources; score 200 held out. Rank each story's candidates by the hub-adjusted share of the source's rare units (the E2 statistic). Statistic: top-1 accuracy by condition, plus second-source recall for combined stories. A script checks that word overlap falls across conditions. Null: source labels permuted within genre.

## Cheap rival it must beat

TF-IDF and BM25; e5 embeddings and story-emb; an LLM judge reading the story and its candidates; length; random units; mean-pooled residual stream. Most dangerous: story-emb and the LLM judge, both made to ignore surface wording. If they win, influence search works without neurons, and the depth of rewording at which each method fails is still new.

## Premise it tests

P1. Truth is known by construction and wording is removed by design, so a pass is direct evidence that units carry inherited content beyond words. A fail leaves no setting where neurons trace influence better than cheap tools. For P2, compare Qwen and Llama rankings against random units.

## Biology analogue

Experimental crosses with known parentage.

## Cost

S to M: about 3 days. 1,250 generated stories of about 400 words, minutes of forward passes per model; the LLM judge over 1,250 stories with 20 candidates each is the main cost.

## Payoff

The first test where influence is certain and wording cannot explain it; a curve of accuracy against depth of rewording for every method; and a calibration set for RX21 and RX22.

## Main risk

Reader neurons may detect LLM writing habits instead of content; a small human-written arm would settle that and is not costed here. Instructed transformation may be easier to trace than real influence. Overlaps with the paused planted-story plan, which plants memes, not sources.

## Reviewer note

Added by the reviewer, promoted from the planted-influence seed: the area had no design where influence is known by construction. Weakness: writers and readers are both language models, so a human-written arm is needed before a pass is trusted.

## Sources

- Wang et al. 2023
- Li et al. 2026 (FakeWiki)
- Chang et al. 2024
- Hatzel and Biemann 2024 (story-emb)
