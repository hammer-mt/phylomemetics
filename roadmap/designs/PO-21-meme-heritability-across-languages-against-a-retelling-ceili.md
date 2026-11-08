# PO-21: Meme heritability across languages, against a retelling ceiling

**Area:** Population memetics · **Status:** design registered 8 November 2026, not run · **Priority:** 4.4 of 5

## Question

When a film adapts a documented source and the two plot summaries come from different-language Wikipedias, are neuron units inherited nearly as reliably as they are re-measured in a second summary of the same story, and more reliably than translated words?

## Prediction

On 500 or more held-out sources, among units repeatable across languages (ceiling at least 0.2 above floor on development sources), concept units reach a median heritability index of at least 0.30 (0 = unrelated works, 1 = the same story summarised in another Wikipedia), at least 0.10 above translated words screened the same way.

## What would refute it

Median index at or below 0.10, or a lead over translated words whose 95% bootstrap interval (resampling sources) includes zero. Void, and reported as an unreliable reader, if fewer than 200 concept units pass the repeatability screen. At 500 sources one unit's correlation has a standard error of 0.045.

## Data

Tell me again! (Hatzel and Biemann 2024): 96,831 plot summaries of 29,505 stories from five Wikipedias, with English machine translations, name-masked versions and Wikidata IDs; CC BY-SA 4.0; 1.7 GB. Joined to Wikidata 'based on' links (CC0): 1,363 film-source pairs over 958 sources have both ends in it (reviewer's join, 2026-11-08), before length filters.

## Method

A unit is one sign of one neuron, word-like if one word predicts its presence, else concept; classes and bars fixed on 20% of sources. Non-English summaries are machine-translated. Correlate presence across: one story in two Wikipedias, close translations dropped (ceiling); source and adaptation (transmission); source and genre-matched unrelated film (floor). Index = (transmission - floor)/(ceiling - floor).

## Cheap rival it must beat

Per feature, with the same screen: machine-translated word presence; an LLM answering 50 motif questions per text (most dangerous: repeatable by design); Llama Scope sparse-autoencoder features; random active units; length. Whole profile: multilingual-e5 and the dataset's published retrieval baselines. If the LLM reader wins, we still learn each unit's repeatability, the reader's first error rate.

## Premise it tests

both. P1: a concept-unit lead means units are wording-independent hereditary markers; a tie leaves P1 where words left it; too few repeatable units means the reader is unreliable. P2: heritability rankings of matched units should agree between Qwen and Llama (rank correlation at least 0.5).

## Biology analogue

Parent-offspring regression; repeatability as the ceiling on heritability; synonymous-site baseline.

## Cost

M. About a week: the dataset is ready-made and name-masked; roughly an hour of extraction per model on one laptop, then the Wikidata join and analysis.

## Payoff

A per-unit table of repeatability, silent-change rate and heritability; a list of units fit to be hereditary markers; and a working definition of synonymous change, which dN/dS-style tools need.

## Main risk

Genre-typical content that recurs without descent counts as inherited unless the floor is matched tightly. Machine translation may flatten wording differences. Few units may pass the repeatability screen.

## Reviewer note

The index rewarded unreliable features (small denominator): every class now passes a repeatability screen first. Swapped self-scouted Wikipedias for a ready, name-masked five-language dataset; pair count checked (1,363). Remaining: how tightly the floor is genre-matched decides how much 'heritability' is convergence.

## Cross-area ranking

In the top twelve of 1 of 3 judges (what decides the premises fastest: rank 12). Reruns the one robust Batman result, heredity, across languages on ready-made name-masked data with a same-story ceiling, so a tie with translated words would finally be informative instead of ambiguous.

## Overlaps with

Asks nearly the same question as LI-22, which the judges would run instead.

## Sources

- Hatzel and Biemann 2024
- Visscher et al. 2008
- da Silva and Tehrani 2016
- Nei and Gojobori 1986
