# IN-27: How much reading is length, position and wrapper?

**Area:** The instrument · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

What share of a unit's reading comes from the motif, and what share from text length, position, surrounding text and prompt wrapper?

## Prediction

For frozen units the motif accounts for at least 50% of reading variance, intraclass correlation (0 none, 1 perfect) across filler lengths is at least 0.75, and filler-only texts of 30 sentences fire at no more than 5% against a nominal 3%.

## What would refute it

Motif share under 30%; or false-fire rate on 30-sentence filler at or above 15%; or presence kappa (chance-corrected agreement) across wrappers under 0.6.

## Data

Scripted: for 10 IN21 concepts, 4 test concept sentences and their 4 matched controls, each set in 1, 5 or 30 filler sentences, early or late, under 4 wrappers: 1,920 texts plus 408 filler-only. Filler comes from 34 of the study's 69 ordinary film summaries (Wikipedia, CC BY-SA); the other 35 set the presence bar.

## Method

Units frozen from IN21, whether or not they passed. Wrappers: bare text, chat template, after an instruction, after unrelated text. Per unit, a variance-components model (how much each factor moves the score) with motif = own concept against matched control; report motif share, intraclass correlations and false-fire rates. Null: random and word-matched units through the same design.

## Cheap rival it must beat

TF-IDF and keywords (immune to position and wrapper by construction), whole-text and windowed sentence embeddings, an LLM reader, a length-only predictor. Most dangerous: TF-IDF, perfectly stable here. The test is whether neurons come close; if not, the study's whole-document, 40-sentence readings need redoing.

## Premise it tests

P1. Shows whether presence reflects content or formatting; a fail would explain why sentence count alone scored 0.76 in the study's identity test (study/results/experiments.json).

## Biology analogue

Batch effects: same sample, different lane or reagent lot

## Cost

S. Scripted assembly, about 2,300 short passes per model, one day.

## Payoff

A length-corrected reading rule and a list of formatting choices to fix in every later extraction.

## Main risk

Filler from film summaries contains motifs of its own, inflating the false-fire rate; 34 independent filler films limit how precisely a 5% rate can be told from 3%.

## Reviewer note

The filler films also set the presence bar, which fixes the false-fire rate by construction; the films are now split. Matched controls added so 'motif share' is defined per unit. 50 sentences cut to 30, the most one film summary allows. Needs IN21 first.

## Cross-area ranking

2 of 3 judges thought the area reviewer rated this too low.

## Amendment, 10 November 2026

Make the false-firing rate on filler-only text the headline, with presence set by each unit's own strong range on general text and not by a bar from other films. The companion paper describes random-text windows for measuring false firing and reports no result, so this is the missing number. Add an arm with about 250 tokens of lead-in: the paper found firing changes when earlier context is removed.

## Sources

- Putnick and Bornstein 2016
- Koo and Li 2016
- study/results/experiments.json (E1 length-only AUC)
- Cammarata 2026
- Tigges et al. 2026, Neurons as an Interpretable Basis
