# FI-25: Plant one meme, measure real reader preference

**Area:** Fitness and forecasting · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

When two versions of one story premise differ only in which meme is planted, do readers reliably prefer one, and which method forecast those preferences beforehand?

## Prediction

Meme appeal is measurable: scores for the 34 memes from one half of the premises and raters correlate 0.6 or more with scores from the other half, and at least 6 story memes beat the neutral-motif average by 8 points of choice share. Unit-based forecasts correlate 0.3 or more with the scores; we expect an LLM reader to do better.

## What would refute it

Split-half correlation under 0.3: single planted memes have no stable appeal at this dose, and forecasting is moot. Or unit-based forecasts correlate under 0.2 while another forecaster passes 0.4. Thirty-four memes detect a true correlation of 0.5 with about 85% power, but cannot separate two forecasters closer than about 0.4.

## Data

New, by construction: the 34 memes in preregistration/memes_for_tracing.json (24 story memes, 10 neutral motifs). Ten name-free premises, each written in 34 versions that differ in one planted sentence avoiding the meme's usual words (the file's ban list). About 20,000 paired judgments from a consented, paid adult panel. Texts and anonymised judgments released under CC BY 4.0.

## Method

Each judgment: two versions of one premise carrying different memes, equal length, 'which would you rather read?'. Scores from a paired-comparison model allowing for premise and rater. Before launch: a second model family and the author confirm each plant blind; record whether the meme's unit fires; freeze forecasts and code. Ethics: fiction only, AI authorship disclosed, debrief, no optimisation loop.

## Cheap rival it must beat

An LLM reading both versions and predicting the choice share (most dangerous: simulated samples tracked survey experiments at r = 0.85 in Hewitt et al.); human forecasters; sentence-embedding differences; sentiment and negative-word counts; text length. Unit, embedding and word models are trained alike, leaving whole memes out. If the LLM reader wins, the causal dataset still stands.

## Premise it tests

P1. If unit profiles forecast a causal preference shift for memes they were not trained on, units carry fitness information. If the LLM reader forecasts better, neurons are the wrong tool for prediction, though planted memes with measured effects remain the first fitness data at the level of ideas.

## Biology analogue

Knock-in mutation followed by a competition assay

## Cost

L. About 20,000 paid judgments (roughly 2,000 to 4,000 US dollars; a guess, not a quote), 340 planted texts to write and audit, and an ethics review. The one design here that needs more than a laptop.

## Payoff

The first open dataset that randomises the idea and not the packaging: causal appeal scores for 34 memes, how far they hold across premises, and a scoreboard of forecasters.

## Main risk

One sentence may be too small a dose. How well a meme fits a premise may matter more than the meme itself. Units may not fire on plants that avoid the usual words: only four of 31 study units survived rewording in both models. Stated preference is not real reading.

## Reviewer note

Replaced the neutral-filler comparison (any eventful sentence beats filler) with meme-against-meme choices and added a reliability gate. Corrected the meme list: 34 memes, 10 neutral, not 30 and 6. Remaining weakness: 34 memes cannot rank forecasters, and units may not fire on plants written without the usual words.

## Sources

- Hewitt et al. 2024 (published as Ashokkumar et al. 2026)
- Hackenburg et al. 2025
- preregistration/memes_for_tracing.json
- study/results/robustness/nulls.json (rewording test)
