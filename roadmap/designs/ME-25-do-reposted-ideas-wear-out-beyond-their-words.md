# ME-25: Do reposted ideas wear out beyond their words

**Area:** Old memetics, re-run · **Status:** design registered 8 November 2026, not run · **Priority:** 3.5 of 5

## Question

When an image is reposted to Reddit under a new title, does repeating the idea of an earlier title cost votes even when no words are shared?

## Prediction

Among reposts sharing no content word with earlier titles of the image, similarity in units to earlier titles lowers vote rank by at least 1.5 percentile points more (most against least similar third) than similarity to later titles, which cannot cause fatigue. The penalty is larger when the earlier post was in the same subreddit.

## What would refute it

An earlier-minus-later gap under 0.5 points, or of the wrong sign, on at least 20,000 reposts that have both earlier and later titles (standard error about 0.6 points, resampled by image). Units also fail if the gap disappears once sentence-embedding similarity is added to the model.

## Data

SNAP Reddit submissions (Lakkaraju et al. 2013): 132,308 posts of 16,736 images, July 2008 to January 2013, one 7.6 MB CSV. Counted on the file on 2026-11-08: 53,515 reposts share no content word with any earlier title, and 46,201 of those also have a later title. No licence stated, so commit only IDs and derived features.

## Method

First re-derive the published word-based title effect; units must reproduce it on reposts that share words. Outcome: vote rank within subreddit and month, adjusted for hour, repost number and days since the last post (a content-free baseline, after Weng). Predictor: similarity to earlier titles, and to later titles as placebo, in units, words and embeddings. Tune on 30% of images.

## Cheap rival it must beat

Word overlap (zero by design), sentence-embedding similarity, an LLM asked whether two titles make the same point, title length, random units, the mean-pooled residual stream. Embeddings are most dangerous. If they win, idea fatigue is still shown beyond words, which is Coscia's claim under control, and the unit is not needed for it.

## Premise it tests

P1. A penalty that units detect where words cannot would show units following a short idea through rewording in wild data. A null or an embedding tie leaves P1 where Batman left it.

## Biology analogue

Negative frequency-dependent selection

## Cost

S. 132,000 short titles, under an hour per model, and a day of analysis.

## Payoff

Whether competition for attention acts on ideas or on strings, and an answer to whether similar memes help or hurt each other with the content held fixed.

## Main risk

Titles are short (median 5 words, counted on the file) and describe an image the model never sees, so unit profiles may be mostly noise. A title close to earlier ones may only be a plain description; the later-title placebo removes that only if plain titles are as common early as late.

## Reviewer note

Made the later-title placebo the test statistic and added a same-subreddit contrast, because closeness to earlier titles may only mark a plain description. Confirmed 53,515 eligible reposts on the file. Remaining: titles have a median of 5 words, so unit profiles may be noise.

## Sources

- Lakkaraju, McAuley and Leskovec 2013
- Coscia 2014
- Weng et al. 2012
