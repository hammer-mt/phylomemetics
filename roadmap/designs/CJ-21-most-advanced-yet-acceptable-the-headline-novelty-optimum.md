# CJ-21: Most advanced yet acceptable: the headline novelty optimum

**Area:** Field conjectures · **Status:** design registered 8 November 2026, not run · **Priority:** 4.5 of 5

## Question

Within a randomised headline test, does the arm that is moderately new relative to recently published headlines win more clicks than very familiar or very new arms?

## Prediction

Click rate is an inverted U in novelty (an arm's distance from the site's previous 90 days of headlines). On confirmatory tests both slopes of a two-lines test are significant at 0.05, the peak falls between the 30th and 80th novelty percentiles, and the peak beats both extremes by at least 3% relative.

## What would refute it

On confirmatory tests no novelty measure shows a sign change at 0.05 (flat or monotonic), or the U is as strong against headlines from six months away (then it is house style, not familiarity). For neurons: adding neuron novelty to embedding and TF-IDF novelty gives no held-out likelihood gain (95% interval includes zero), overall and in the low-word-overlap subset.

## Data

Upworthy Research Archive (Matias et al. 2021): 32,487 tests, 150,817 arms, 2013 to 2015, CC BY 4.0, OSF jd64p. Fit on the exploratory split (4,873); score once on the confirmatory split (22,743). Drop tests flagged for the 2013 randomisation fault and arms differing in image: about 10,000 confirmatory tests should remain. Robertson et al. detected a 2.3% effect in 12,448.

## Method

Units: Qwen3-8B neuron families (neurons that fire together), chosen on exploratory data. Novelty: mean distance from an arm's unit profile to the prior 90 days' headlines. Arms compared within their own test, length controlled; two-lines test (one slope below a break point, one above). Null: novelty against a window six months away. Pre-set subset: arms sharing few stream words.

## Cheap rival it must beat

Novelty from TF-IDF backward similarity, sentence embeddings, the mean-pooled hidden state, sparse-autoencoder features on embeddings (a learned feature dictionary; Movva et al. 2025), an LLM rating 'how unusual for this site' against 20 recent headlines, headline length, random units. Embedding novelty is most dangerous: smooth and cheap. If it wins, the hunch is still tested causally; neurons add nothing.

## Premise it tests

P1. A U found equally with words or embeddings settles the marketing claim and leaves P1 unmoved. A U that survives in the low-word-overlap subset for neuron novelty alone would show units track a familiar frame under new wording. Repeating it in Llama bears on P2.

## Biology analogue

Optimal outbreeding: mates preferred slightly unlike kin.

## Cost

S. About 150,000 short headlines: under an hour of forward passes per model on the laptop. The LLM-rating rival runs on a 3,000-test subsample. No live audience; archived data only.

## Payoff

A causal answer to whether 'just past the familiar' wins, with the size of the effect and where the peak sits. If any measure works, a novelty ruler for planning creative tests.

## Main risk

Arms of one test share a story, so their novelty may differ too little to trace a curve; the U then leans on comparing stories, which were not randomised. Upworthy effects are tiny, and Berger and Packard found a monotonic pattern for lyrics.

## Reviewer note

Replaced the within-month date shuffle, which barely moves a 90-day window, with a misplaced-window null; added headline length, the fault flag, the image rule and an estimated usable sample. Remaining weakness: novelty may vary little within a test. Prior-work check was arXiv only (nothing found); web search was unavailable.

## Sources

- Hekkert, Snelders and van Wieringen 2003
- Matias et al. 2021 (Upworthy Research Archive)
- Simonsohn 2018 (two-lines test)
- Sreenivasan 2013
