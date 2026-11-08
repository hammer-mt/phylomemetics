# IN-25: Test-retest across independent retellings of one work

**Area:** The instrument · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

Is a unit's reading a stable property of a work, or of one summary of it?

## Prediction

For frozen concept units, the reading (the unit's score for a text) agrees between two independent summaries of the same story at a median intraclass correlation (0 none, 1 perfect) of at least 0.5, and at least 0.10 above word-matched control units.

## What would refute it

Median intraclass correlation under 0.3; or not above both word-matched units and the word-presence rival by 0.05; or no higher than for matched different-story pairs. Any of these means one summary cannot stand for a work.

## Data

Tell Me Again (Hatzel and Biemann 2024): 96,831 summaries of 29,505 stories from five Wikipedia language editions, with machine translations and name-anonymised versions; CC BY-SA 4.0. Sample 600 stories with an English and a translated summary of 20 or more sentences each, dropping pairs the dataset's similarity score marks as translations; plus 600 different-story pairs matched on genre, decade, length.

## Method

Units frozen beforehand: IN21's units, the study's labelled concept units, 2,000 random active and word-matched units. Readings on anonymised text, both summaries cut to equal sentence count. Per-unit intraclass correlation of the continuous reading (presence agreement is secondary, since presence is rare), same-story against different-story pairs, bootstrap over stories. 600 stories gives a standard error near 0.04.

## Cheap rival it must beat

Word-presence and TF-IDF agreement, story-embedding cosine (that model trained on this dataset, so use its test split), an LLM reader asked the label question on each summary, length alone. Most dangerous: the LLM reader. If it is more stable, neuron reliability still caps every validity figure we report.

## Premise it tests

P1. Reliability bounds validity. Low agreement means the Batman grid measured Wikipedia summaries, not works; agreement above word-matched units supports a content reading.

## Biology analogue

Technical replicates: sequencing the same sample twice

## Cost

S. 1,200 summaries per model, minutes of extraction; a day for sampling and the LLM-reader rival.

## Payoff

A reliability coefficient per unit class, and the correction needed to interpret every later correlation.

## Main risk

The second summary is machine-translated, so register differs as well as content; and the dataset paper found 13 of 100 German-English pairs were direct translations, which the similarity filter only partly removes.

## Reviewer note

Dataset confirmed (download live, CC BY-SA 4.0). Its most common summary length is three sentences, so sampling now requires 20 or more. Added the dataset's own translation filter. Main statistic changed from rare binary presence to the continuous reading. Translation style stays mixed with retelling.

## Sources

- Hatzel and Biemann 2024 (Tell Me Again)
- Koo and Li 2016
- Chang 2004
- Hatzel and Biemann 2024 (Story Embeddings)
