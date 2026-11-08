# UN-22: Same story, another language, another model

**Area:** Across models · **Status:** design registered 8 November 2026, not run · **Priority:** 5.0 of 5

## Question

When a story is retold in another language and no words are shared, do unit readings still match, within one model and between Qwen and Llama?

## Prediction

On name-replaced summaries, units find the same story across languages (German original against an English summary) at half or more of the rate they reach when both are read in English; the measure is top-1, the right story ranked first among all test stories. Qwen reading German against Llama reading English keeps half or more of the within-model rate.

## What would refute it

The cross-language rate is under a quarter of the same-language rate, or no better than random units matched on firing rate, or the cross-model rate is under a quarter of the within-model rate. At least 1,000 qualifying test stories (count them before registering) make a 5-point paired gap about 3 standard errors.

## Data

Tell Me Again (Hatzel and Biemann 2024): 96,831 Wikipedia plot summaries of 29,505 stories in English, German, French, Italian and Spanish, with supplied machine translations into English, name-replaced versions and an 80/10/10 split (test: 2,951 stories). A 1.7 GB download; text CC BY-SA 4.0, code MIT. Summaries under 8 sentences and pairs flagged as direct translations are dropped.

## Method

Three contrasts per story: German original against its supplied English translation (language only); English summary against the translated German summary (retelling only); English against German original (both). Units: those stable across language on the development split. Cross-model link: a ridge map (a regularised linear fit) from development stories; UN21 twins second. A hard gallery uses same-genre, same-decade stories.

## Cheap rival it must beat

TF-IDF on the supplied English translations (machine translation makes words a rival again), multilingual sentence embeddings, an LLM asked whether two summaries tell one story (500 pairs), length difference, random units, and each model's mean-pooled residual stream. Multilingual embeddings will probably win. The result still shows whether unit agreement exists once shared vocabulary is gone.

## Premise it tests

Both. Agreement that survives translation would be the first cross-model evidence that is not names and words. A collapse means the study's agreement was lexical, and translation-proof memes need directions or embeddings. Replaces the deferred Chinese protocol with human-written text in languages Llama 3.1 lists as supported.

## Biology analogue

Synonymous codons: different sequence, same protein.

## Cost

M. About a week: a data loader, name replacement for the non-English originals, five to ten hours of extraction for two models, no paid API because translations are supplied.

## Payoff

An answer to Mike's fuzziness question at story scale: retelling, translation and name removal are separated in one natural corpus of thousands of stories instead of 32 works. French, Italian and Spanish repeat the German result.

## Main risk

Both models have read these pages in every language and may simply recognise the film, so stories neither model can name are scored separately. Language editions tell different parts of a story. Leftover names, numbers and cognates (words with a shared root) can leak, and translation errors can hide signal.

## Reviewer note

Defined the unit choice, replaced the dependence on UN21 twins with a fitted map, fixed an impossible gallery and added a recognition check. Dataset confirmed. Remaining: the supplied name-replaced texts are probably English only, and language-general middle layers within one model are already reported (Anthropic 2025).

## Cross-area ranking

In the top twelve of 3 of 3 judges (what decides the premises fastest: rank 6; speed and cost: rank 10; upside: rank 3). Translation strips shared wording from thousands of human-written summaries, so one run tests both premises where neither names nor words can carry the signal, and it includes the fitted map between models that the project has never tried.

## Sources

- Hatzel and Biemann 2024 (Tell Me Again, LREC-COLING)
- Wendler et al. 2024
- Anthropic 2025 (On the Biology of a Large Language Model)
- preregistration/translation_protocol.md
