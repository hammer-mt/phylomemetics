# UN-24: Precision and size ladder within Qwen3

**Area:** Across models · **Status:** design registered 8 November 2026, not run · **Priority:** 3.5 of 5

## Question

How much does a unit's reading change with weight precision and model size, compared with the change between model families?

## Prediction

Between 4-bit and 8-bit Qwen3-8B, the same unit's sentence responses correlate at a median of 0.95 or more, and work-level presence agrees at kappa 0.9 or more (agreement corrected for chance). Secondary: the share of units with beyond-word Llama twins (UN21) rises with size from Qwen3-0.6B to 14B, rank correlation 0.9 or more.

## What would refute it

Presence kappa between 4-bit and 8-bit is under 0.8 for meme units, close to the 0.73 found between Qwen and Llama for the 31 named memes (`nulls.json`). Secondary: the rank correlation of twin share with size is under 0.9 (five sizes reach 0.9 by chance about 1 time in 24). Sample: 25,925 sentences in 1,114 works.

## Data

The committed study corpus (114 works, 3,700 sentences, original and name-free) and the 1,000 committed random film plots (22,225 sentences); Wikipedia text, CC BY-SA. Models: open Qwen3 weights at 0.6B, 1.7B, 4B, 8B and 14B (Apache 2.0, confirmed on Hugging Face), at 4-bit, 8-bit and 16-bit precision wherever they fit in 24 GB.

## Method

Precision arm: one unit index read at two precisions on the same text, so wording cannot differ; sentence correlation, presence kappa, whether the 56 core and 31 named memes are selected again, and the same for the 8-bit storage of activations. Size arm: UN21's held-out census between each Qwen3 size and Llama. Nulls: rate-matched unit pairs; misaligned sentences.

## Cheap rival it must beat

TF-IDF and text length do not change with precision (kappa 1), so they set the ceiling. The mean-pooled residual stream goes through the same comparison and is most dangerous: if it is stable where single units are not, the unit basis is the fragile part. Random non-meme units show whether meme units are more or less stable than average.

## Premise it tests

Both. Stable readings mean the 4-bit study stands as measured; unstable ones mean every published unit number carries unmeasured noise, and no cross-model agreement can exceed that ceiling. A twin share that grows with size supports convergence with scale; a flat one sides with Koepke et al. against it.

## Biology analogue

Technical replicates before comparing species.

## Cost

S. An hour or two of extraction per model and precision. The 14B model at 4 bits is the largest that fits beside its activations; the 8B model at 16 bits (about 16 GB) may not fit, but 1.7B and 4B at 16 bits do.

## Payoff

A reliability figure for the instrument itself, which the study never measured, and a ceiling against which every cross-model agreement number can be read.

## Main risk

The likely result is dull: readings are stable, which is still worth knowing. Unit indices line up across precisions but not across sizes, so the size arm inherits every weakness of twin matching, has one model per size, and Llama's fixed 8B size may favour the 8B Qwen whatever the trend.

## Reviewer note

Made precision the primary arm and size a stated secondary with its own small-sample test; dropped the claim that this tests whether neurons are clean. All checkpoints confirmed. Remaining: five sizes with one model each cannot separate scale from similarity to Llama's own size.

## Sources

- Cammarata 2026
- Huh et al. 2024
- Koepke et al. 2026
- study/results/robustness/nulls.json
