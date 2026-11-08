# UN-25: Shared units against human judgements of similarity

**Area:** Across models · **Status:** design registered 8 November 2026, not run · **Priority:** 2.5 of 5

## Question

Do the units two models share predict which texts people judge alike better than the units only one model has?

## Prediction

Across sentence pairs rated by people, distance over shared units tracks the human score better than distance over model-specific units, by 0.05 or more in rank correlation after word overlap and length are removed. On story triples, shared units pick the story the annotators chose more often.

## What would refute it

The gap in partial rank correlation is under 0.02, or the shared-unit term adds nothing once word overlap is in the regression (p above 0.05). 8,628 pairs give a standard error near 0.01. The 1,039 story triples can only detect a gap of 5 to 6 points, so they cannot refute alone.

## Data

STS Benchmark (Cer et al. 2017): 8,628 English sentence pairs with human similarity scores; no licence is stated on its dataset card, so confirm it and commit ids only. SemEval-2026 Task 4 (Hatzel et al. 2026): 1,039 story-summary triples with human choices (200 development, 800 test); confirm the licence on the task site before registering.

## Method

Qwen and Llama read each text. Shared units pass UN21's held-out, beyond-word twin test. Model-specific units have no twin and are matched on layer, firing rate, word-predictability and stability across precision (UN24). Statistics: partial rank correlation of unit distance with the human score; triple accuracy; both repeated where word overlap and the human score disagree. Null: random unit sets.

## Cheap rival it must beat

Word overlap and TF-IDF, a general sentence embedding (trained on such data, it will win outright), an LLM asked directly (subsample), length difference, random units, and each model's pooled residual stream, the dangerous one for units. The design still shows whether cross-model agreement marks the part of a reading people share.

## Premise it tests

P2, and the bridge to readers. A pass supports the convergence argument: what models agree on is what people see, so twin strength could filter for real memes. A fail means cross-model agreement and human relevance are separate properties, so universality cannot stand in for human validation.

## Biology analogue

Sequence conserved across species as a sign of function.

## Cost

S. Short texts, two extraction passes, archived human labels, no new raters. Needs the unit lists from UN21 and UN24 first.

## Payoff

The first check of any unit reading in this project against human judgement, at no rater cost, and a test of whether twin strength is a useful filter for memes people would recognise.

## Main risk

Sentence pairs from captions and headlines are far from story memes. The story triples are few, were kept only where two LLMs disagreed, and annotators agreed weakly (alpha 0.33). Units without twins may simply be noisier, which matching can only partly remove.

## Reviewer note

The story triples alone were fatal: 1,039 in all, kept only where LLMs disagree, annotator agreement 0.33, best system 78%. Added 8,628 rated sentence pairs for power and a stability-matched control. Remaining: depends on UN21 and UN24, and sentence similarity is not story similarity.

## Sources

- Cer et al. 2017 (STS Benchmark)
- Hatzel et al. 2026 (SemEval-2026 Task 4)
- Huh et al. 2024
- Marti et al. 2023
