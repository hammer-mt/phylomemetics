# PO-26: Open or closed: pan-memomes of retellings and series

**Area:** Population memetics · **Status:** design registered 8 November 2026, not run · **Priority:** 2.5 of 5

## Question

Does the rate at which each added work brings new units recover a documented ordering (retellings of one source below sequel series below matched random film sets), and does it do so better than vocabulary growth?

## Prediction

Name-free, the share of a set's sixth work that is new separates 25 retelling sets from 25 sequel series with AUC at least 0.80 (chance 0.5), and at least 45 of the 50 real sets are more closed than all 50 of their matched random sets. Twenty-five against 25 gives a standard error near 0.06 on the AUC.

## What would refute it

AUC at or below 0.65, or 10 or more of the 50 real sets fall inside their random-set range (no more closed than chance). Units add nothing if word-type growth separates the classes as well (paired bootstrap interval of the AUC difference includes zero).

## Data

Sets of six or more works with English Wikipedia plot sections (CC BY-SA 4.0). Wikidata (CC0; reviewer recount, 2026-11-08) lists 118 sources with six or more film adaptations (22 single stories with twelve or more, such as Dracula) and 119 film series with six or more films. No Shakespeare sets: that list is a paused test. Plus 1,000 random films.

## Method

Mask names; cut to equal length. Keep every unit: the committed matrix drops families found in under six Batman works (`study/discover.py`), hiding rare memes. Per set, average over orderings the share of the sixth work's units absent from the first five; with twelve works, also fit the power-law decline. Compare with 50 random sets matched on decade, genre and length.

## Cheap rival it must beat

Heaps'-law growth of word types and TF-IDF terms; spread of sentence embeddings; distinct motifs an LLM tags on a subsample; random non-meme units. Most dangerous: word types, which will probably give the same ordering. Then the pangenome view is confirmed as a tool but needs no neurons.

## Premise it tests

both. P1: a pass shows the presence matrix carries population structure beyond threshold noise; equal performance from words says canon tiers describe vocabulary. P2: Qwen and Llama should rank the 50 sets alike (rank correlation at least 0.7).

## Biology analogue

Pangenome: core and accessory genes, open or closed (Tettelin 2005).

## Cost

M. About 1,500 plots, minutes of extraction per model; three days to assemble and check the sets; a week in all.

## Payoff

A noise-corrected openness measure with core, shell and cloud tiers for 50 story traditions: a way to say whether a franchise's meme pool is still growing.

## Main risk

The ordering is close to guaranteed for any text representation, so the test may only show that unit matrices behave like vocabulary. Threshold noise (about 3% false presence per unit) may swamp the rare tier.

## Reviewer note

Ten against ten could not tell units from words (an AUC gap of 0.05 is noise): now 25 against 25 sets of six or more, scored at a fixed depth. Dropped the paused Shakespeare list. Remaining: words will give the same ordering; the value is the tool.

## Sources

- Tettelin et al. 2005
- Snel, Bork and Huynen 1999
- study/results/canon.json
- study/results/robustness/nulls.json (f1_chance)
