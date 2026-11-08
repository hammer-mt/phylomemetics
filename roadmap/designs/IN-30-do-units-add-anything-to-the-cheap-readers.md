# IN-30: Do units add anything to the cheap readers?

**Area:** The instrument · **Status:** design registered 8 November 2026, not run · **Priority:** 3.5 of 5

## Question

On natural synopses with human-named tropes, do frozen unit readings improve held-out trope detection once word, embedding, LLM-reader and length scores are already in the model?

## Prediction

Adding 16 frozen units per trope to a combined model of the cheap readers raises median held-out AUC (chance 0.5) across about 30 plot-event tropes by at least 0.02, with the interval excluding zero, in both 8B models, and by more than 16 random active units do.

## What would refute it

Gain under 0.01, or no larger than the gain from random or word-matched units, while the cheap readers add 0.05 or more to the units alone. Then units carry nothing about tropes that cheap readers lack.

## Data

TiMoS (Chang et al. 2021): TV Tropes labels for about 5,600 films in training, validation and test files (4,059, 715 and 825); the median trope is tagged on about 375 films. Repository MIT; synopses from IMDb, licence unclear, so commit film names and labels only. Names masked by script; films re-split by franchise.

## Method

Per trope, pick units on the training split in two stages; freeze. On the validation split fit two logistic models: cheap readers only (TF-IDF classifier score, embedding probe score, LLM reader's yes-probability, length, number of tropes tagged as a popularity proxy) and the same plus units. Compare held-out AUC, bootstrap over films; and in reverse. Null: shuffled trope labels.

## Cheap rival it must beat

TF-IDF logistic regression, a sentence-embedding probe, a zero-shot LLM reader, synopsis length, popularity, the mean-pooled residual stream. Most dangerous: the embedding probe, trained on thousands of films. If units add nothing, P1 keeps inspectability and cost only, which is worth knowing before any lineage work.

## Premise it tests

P1. The discriminant rung on natural text at scale. An increment in both models would be the first evidence that neurons carry narrative content cheap readers miss; none would repeat the Batman result on 5,600 films.

## Biology analogue

Does a new biomarker add to the standard panel?

## Cost

M. About two hours of extraction per model; the LLM reader on validation and test films for 30 tropes is the heavy part, about a day; one week in all.

## Payoff

A number for what neurons add beyond cheap readers on human-named memes, per trope, with the reverse comparison.

## Main risk

Crowd labels are noisy and a missing tag is not absence, which shrinks every method's score and may hide a small increment.

## Reviewer note

Added: the area had no stand-alone discriminant test on natural text. Label files checked: 5,599 films, trope prevalence 4 to 33%. Weakness: TV Tropes tags track popularity and miss true cases, and Kantamneni et al. suggest no increment is the likely result.

## Sources

- Chang et al. 2021 (TiMoS)
- Su et al. 2024
- Kantamneni et al. 2025
- Campbell and Fiske 1959
