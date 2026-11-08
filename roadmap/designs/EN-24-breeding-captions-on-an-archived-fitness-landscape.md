# EN-24: Breeding captions on an archived fitness landscape

**Area:** Meme engineering · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

When a score-and-select loop may only move among captions that real people already rated, does scoring with model-internal units find funnier captions than scoring with embeddings or an LLM judge?

## Prediction

A tie. After 10 generations the unit-scored loop's final 20 captions sit within 3 percentile points of the embedding-scored loop's in real crowd rating, and both beat random selection by at least 5 points.

## What would refute it

The unit-scored loop ends more than 3 percentile points above or below the embedding-scored loop, with a 95% interval excluding zero across 100 held-out contests; a tie is claimed only if the whole interval sits within 3 points. Or no loop beats random selection by 5 points: caption text alone carries no breedable signal.

## Data

New Yorker caption contest rankings (Zhang et al. 2024; Hugging Face `yguooo/newyorker_caption_ranking`): 365 contests, over 2.2 million captions, over 250 million crowd ratings, plus a text description of each cartoon. CC BY-NC 4.0, research use only: commit only IDs and scores. The earliest 265 contests train the scorers; the latest 100 are held out. Captions with under 25 ratings dropped.

## Method

Unit scorer: a linear model on Qwen3-8B units chosen on training contests. Every scorer sees the cartoon description. Gate: stop if no scorer picks a top-tenth caption over a bottom-half one 55% of the time in training. Runs start from 20 random captions; each generation offers every survivor's 5 nearest real captions by word overlap; the scorer keeps 20.

## Cheap rival it must beat

Scorers built on sentence embeddings (most dangerous: they tie fine-tuned models on headline data), TF-IDF, the mean-pooled residual stream, caption length, an LLM asked which caption is funnier, and random selection. Real ratings used as the scorer give the ceiling. A tie still leaves units able to name what selection favoured.

## Premise it tests

Assumes both; tests P1 as a selection instrument. A win says internal units see appeal that embeddings miss. A tie leaves readable names as their only edge. A loss says breed with embeddings and use units only to describe the result.

## Biology analogue

Directed evolution on a mapped fitness landscape (deep mutational scanning).

## Cost

M. About 10 hours of forward passes for every caption, or 3 on a sample; the LLM-judge arm is the slow part. One week.

## Payoff

The first test of meme breeding in which every candidate has a real human outcome, with no live audience and no persuasion.

## Main risk

Text carries little of the joke. In Zhang et al. 2024, GPT-4 given a cartoon description picked a top-10 caption over one ranked near 1,000 only 65 to 67% of the time. All loops may hover near random and the gate may stop the study.

## Reviewer note

Added a go/no-go gate, the cartoon descriptions as input, a minimum rating count (the contest's bandit gives weak captions few ratings) and a proper rule for claiming a tie. Remaining weakness: nothing is generated, so this tests selection only, where text signal is weak.

## Sources

- Zhang et al. 2024, Humor in AI, arXiv:2406.10522 (caption contest data)
- Ye, Yoganarasimhan, Zheng 2025, LOLA, arXiv:2406.02611
- Bradley et al. 2023, QDAIF, arXiv:2310.13032
- Novikov et al. 2025, AlphaEvolve, arXiv:2506.13131
