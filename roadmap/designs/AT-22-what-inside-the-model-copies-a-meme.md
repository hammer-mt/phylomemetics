# AT-22: What inside the model copies a meme?

**Area:** The atlas · **Status:** design registered 8 November 2026, not run · **Priority:** 4.5 of 5

## Question

When a model retells a story in its own words, is a planted meme carried into the retelling by one neuron, by a small set of neurons, or only by a direction spread across thousands of them?

## Prediction

Silencing 64 chosen neurons cuts a motif's survival in retellings by 30 points or more on average across 20 motifs, at least 20 points more than 64 matched random neurons, while the other planted motifs lose under 10 points. Silencing the single best detector neuron cuts under 10 points.

## What would refute it

The 64-neuron cut averages under 15 points, or is within 10 points of 64 random neurons matched on layer and activation (bootstrap over motifs, 95% interval includes zero), or the other motifs fall as far as the target. With 25 test stories per motif and arm (500 per arm), a pooled drop has a standard error near 4 points.

## Data

800 new stories of 200 words (20 motifs, each with 15 development and 25 test stories), each carrying its target motif plus two others and paired with a twin lacking the target; written by an LLM from a different family, released CC BY. Motif definitions as in AT-21. Model: Llama-3.1-8B-Instruct at 4-bit, which reads the story and writes the retelling.

## Method

Arms: untouched; best detector neuron; top 8, 64 and 512 neurons, ranked on development stories by how much setting each to its twin-story value changes the retelling, and silenced the same way; one difference-of-means direction removed. The 64-neuron arm is rerun for reading only and writing only. Keywords are blocked from output. A blind cross-family reader scores survival; 150 hand-checked.

## Cheap rival it must beat

A plain instruction to leave the motif out; the keyword block alone; deleting the motif's sentences from the input; silencing the neurons that best detect the motif's keywords. The instruction will probably remove the motif most cleanly. The design still locates the carrier, which an instruction cannot; the bar to beat is random and keyword neurons.

## Premise it tests

P1. If one neuron or a few dozen are necessary for copying, P1 gains the causal footing it lacks. If only a dense direction works, the atlas's unit should be a direction, and a neuron is at best an index into it. If only deleting the sentences works, memes are not copied as units here.

## Biology analogue

Hershey and Chase, then Watson and Crick; gene knock-out

## Cost

M. Ablation hooks for the 4-bit model, about 7,000 short generations (roughly 30 laptop hours), judge calls under 50 US dollars; two weeks including code.

## Payoff

The size of the thing that copies a meme in one model: one neuron, dozens, hundreds or a direction, and whether it acts at reading or at writing. That decides what an atlas entry should be and whether editing a meme by neuron is realistic. It is a mechanism for machine culture, not for human minds.

## Main risk

A retelling can copy through attention straight from the source words, with no motif-level carrier to silence, so a null may mean 'no carrier' and not 'spread out'; the direction and sentence-deletion arms help tell these apart. Silencing can also damage the model in ways the collateral checks miss.

## Reviewer note

Changed: per-motif pass counts became a pooled drop; added a 512-neuron arm, twin-story silencing and a reading-only against writing-only split. Remaining: in Cammarata's own example, removing 20 labelled neurons left a concept at 76%, so 64 may be too few; damage and removal stay hard to separate. Neighbours: IN24, EN-27.

## Sources

- Arora et al. 2026, arXiv:2601.22594
- Liu, Joshi, Dawson and Piper 2026, arXiv:2610.12989
- Wu et al. 2025, AxBench, arXiv:2501.17148
- Cammarata 2026, Neural networks are already clean
