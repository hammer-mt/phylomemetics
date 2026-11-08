# PO-22: ABBA-BABA test for borrowing in crossover works

**Area:** Population memetics · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

Can an excess-sharing statistic on name-free unit presence tell which outside franchise a crossover work borrowed from, more often than plain similarity or the same statistic on words?

## Prediction

Calibration: when donor sentences are spliced into host plots, D rises with the spliced share (rank correlation at least 0.8). Test: on 40 crossovers between separate franchises, D is higher for the true donor than for a wrong donor of the same genre and decade in at least 80%, and more often than word D.

## What would refute it

Tool claim fails at 65% or fewer correct (40 crossovers give about 0.8 power to tell 80% from 60%). Unit claim fails if word D or plain similarity is correct as often (paired interval, clustered by franchise pair, includes zero). Void if calibration fails, or ordinary franchise entries in the crossover slot give mean D outside plus or minus 0.05.

## Data

Code check only: committed matrix (`study/results/qwen3-8b/memes.json.gz`, 114 works, names present). Test: 40 crossovers between two separate franchises, over 15 or more franchise pairs, plus parent works. The cached pool (`preregistration/sources/t3_candidates.jsonl.gz`) has 143 crossover-category plots, but most are same-universe team-ups; about 20 qualify by the reviewer's reading of titles. Fetch the rest from Wikipedia's crossover categories (CC BY-SA 4.0).

## Method

Mask names; keep every unit (the committed matrix keeps only families in six or more Batman works, missing donor material). Equal sentence counts. For crossover X, host work A, donor work B: ABBA = units in X and B, not A; BABA = units in A and B, not X. D = (ABBA - BABA)/(ABBA + BABA), averaged over choices.

## Cheap rival it must beat

The same D on word presence (most dangerous: donor vocabulary survives name masking); plain similarity of crossover to donor with no host correction; embedding similarity gain; an LLM asked to name the donor from masked text (probably near ceiling); random non-meme units. A tie with words still validates D as a tool for unit matrices.

## Premise it tests

both. P1: if unit D only matches word D, borrowing detection needs no neurons; a unit lead on name-free text would be the first population statistic where units see more than words. P2: Qwen and Llama should rank donors alike (rank correlation at least 0.6).

## Biology analogue

Introgression test between non-sister lineages (Green 2010; Durand 2011).

## Cost

S. A day to list crossovers, donors and wrong donors from Wikipedia leads; about 600 works, minutes of extraction per model; splicing needs no new text.

## Payoff

A tested, tree-free statistic for borrowing between franchises with a known smallest detectable share, and a plain answer on whether name-free unit matrices carry donor signal beyond words.

## Main risk

An easy win for every method: crossovers contain the donor's characters and world, so words and the LLM reader will probably tie or lead. It must skip the 39 held-out Batman works (`preregistration/works_batman_extended.json`), several of which are crossovers.

## Reviewer note

The free pilot was biased: the committed matrix drops donor units, so it is now a code check. Added spliced-dose calibration, a plain-similarity rival, a harder wrong donor, separate-franchise crossovers only. Remaining: documented crossovers are easy for words, so the unit claim will probably tie.

## Sources

- Durand et al. 2011
- Green et al. 2010
- Patterson et al. 2012
- preregistration/sources/t3_candidates.jsonl.gz
