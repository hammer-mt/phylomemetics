# FI-21: Do neuron profiles pick winning headlines beyond embeddings?

**Area:** Fitness and forecasting · **Status:** design registered 8 November 2026, not run · **Priority:** 4.4 of 5

## Question

Within a randomised headline test, does the difference between two headlines' neuron-unit profiles predict which got more clicks, beyond length, word counts, sentence embeddings, the model's own averaged internal state and an LLM's guess?

## Prediction

Adding neuron-unit differences to a model that already holds every cheap rival raises held-out pairwise accuracy (which of two headlines with the same image and teaser text won) by at least 1.0 percentage point, in Qwen3-8B and in Llama 3.1 8B. Our honest expectation is a tie: a gain under 0.5 points.

## What would refute it

The gain is under 1.0 point, or its 95% interval (resampling whole tests) includes zero. The holdout should hold about 1,900 usable tests and 15,800 headline pairs (counted on the equal-sized exploratory file), so the gain's standard error is 0.3 to 0.6 points. If simulation shows more than 0.35, register the cross-fitted score over confirmatory plus holdout instead.

## Data

Upworthy Research Archive (OSF jd64p, CC BY 4.0): 32,487 randomised tests, 150,817 headline-and-image packages, 2013 to 2015, in exploratory (4,873 tests), confirmatory (22,743) and holdout files. Drop tests from 25 June 2013 to 10 January 2014 (faulty randomisation, 22%). Keep arm pairs that share image, excerpt and lede, so only the headline differs.

## Method

Units: both signs of every feed-forward (MLP) channel, maximum over tokens. Choose 2,000 units and settings on exploratory, fit on confirmatory, score once on holdout. Statistic: accuracy gain over the rival stack, beside the ceiling click noise allows. Nulls: arm labels swapped within test; random units matched on firing rate and layer. Repeat using only firing that words cannot predict.

## Cheap rival it must beat

Headline length and punctuation; rarity-weighted word and character counts (TF-IDF); sentence embedding plus linear model (46.3% at naming a test's winner in LOLA, chance 33%); sparse-autoencoder features on embeddings (HypotheSAEs) and inside Llama (Llama Scope); a local LLM asked which wins. Most dangerous: a probe on the same model's averaged internal state, which units only re-express.

## Premise it tests

P1. A pass means units carry success-relevant content that words and embeddings miss, and justifies the costlier fitness designs. A fail, which we expect, means neuron profiles re-describe what embeddings already hold: useful for naming causes of success, not for forecasting it. Says nothing about P2.

## Biology analogue

Competition assay; named genes against anonymous markers

## Cost

S. About 150,000 headlines through two 8B models, under an hour each on the laptop; the rest is CPU. Run the LLM-judge rival on a subsample.

## Payoff

A click effect for each named unit, estimated from randomised tests, and a plain answer on whether neuron profiles forecast success better than embeddings. The activations are reused by FI-22, FI-27 and several seeds.

## Main risk

A tie with embeddings is the likeliest result. The labels are noisy: only about 28% of headline pairs differ significantly in click rate, so accuracy tops out far below 100%. Headline memes are packaging (curiosity gap, negativity), not story ideas, so even a win may not carry over to stories.

## Reviewer note

Counted the data: about 1,900 clean headline-only tests in the holdout, so the stated 0.3-point error was optimistic. Added a fallback, a noise ceiling, same-teaser matching and a same-model sparse-feature rival. Prior-art check used arXiv only (web search was unavailable). Remaining weakness: a tie is likely.

## Cross-area ranking

In the top twelve of 1 of 3 judges (speed and cost: rank 8). One 40-minute pass per model over a free archive of randomised headline tests gives a clear win-or-tie against embeddings, and the same activations feed five other designs.

## Sources

- Matias et al. 2021, Upworthy Research Archive (with the 2024 correction)
- Ye, Yoganarasimhan, Zheng 2025 (LOLA)
- Movva et al. 2025 (HypotheSAEs)
- He et al. 2024 (Llama Scope)
