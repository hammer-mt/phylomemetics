# AT-21: Reference memome: which units pass admission?

**Area:** The atlas · **Status:** design registered 8 November 2026, not run · **Priority:** 4.4 of 5

## Question

For one public checkpoint, how many of 150 story motifs get a neuron entry that passes a fixed admission test on held-out, reworded passages, compared with control units and rival detectors put through the same test?

## Prediction

At least 30 of 150 motifs (20%) gain an admitted single-neuron entry and at least 60 (40%) an admitted set of up to eight neurons; the sets admit at least half as many motifs as a probe on the same model's averaged hidden state. Permuted motif labels admit 3 or fewer; a random-weights copy of the model 5 or fewer.

## What would refute it

Fewer than 15 single-neuron entries admitted; a count inside the permuted-label null's 95% range; a random-weights copy yielding half as many; or neuron sets admitting under half the hidden-state probe's count. With 24 test passages per cell a chance-level unit is admitted under 1% of the time; 150 motifs separate a 20% yield from 10% at z = 4.

## Data

150 motifs: 100 drawn by seed from the 307 Thompson motifs tied to ten or more tales in Hagedorn's trilogy data (CC BY-SA 4.0), 50 from TiMoS's 95 film tropes. Per motif, 192 new 120-word passages: motif present or absent (absent: a twin with another event of equal stakes), usual words present or absent. 28,800 passages, CC BY. Llama-3.1-8B-Instruct, pinned.

## Method

Units are neurons read on one side (917,504). On 96 development passages per motif, pick the best unit, best eight-unit set and threshold; freeze. Admit if, on 96 test passages by a second writer model, AUC (0.5 is chance) for motif-without-words against words-without-motif has a 90% lower bound of 0.70 and other motifs trigger it under 10% of the time.

## Cheap rival it must beat

Keyword lists; a TF-IDF (weighted word count) classifier; e5 and Qwen3-Embedding sentence vectors; a probe (simple classifier) on the model's hidden state; Llama Scope sparse-autoencoder features (learned units); the neuron whose Transluce description best fits the definition; passage length; a zero-shot LLM reader. The LLM reader will probably admit most; the census still gives every method error rates and cost.

## Premise it tests

P1. A yield at the permuted-label or random-weights level means single neurons are not usable meme units in this model. A yield of 20% or more turns P1 into a bounded claim: these named units, with these error rates, and no others. If the hidden-state probe admits twice as many, the atlas's unit should be a direction.

## Biology analogue

Reference genome with evidence codes; mock community

## Cost

M. About 28,800 short passages by API (under 150 US dollars), six hours of laptop extraction on a 4-bit model, a three-hour human audit of 300 passages; about ten days in all.

## Payoff

The first citable list of meme units with measured sensitivity and false-fire rates, from which later experiments draw frozen units. It also sizes the memome: how many of 150 human-named motifs one neuron, a small set, or no unit can read.

## Main risk

Passages written by language models share house styles, so a unit can learn a writer's habit for one cell and pass. A second writer family for test passages and the human audit reduce this; they do not remove it. AT-27 then checks the entries on real tales.

## Reviewer note

Changed: motifs come from sources checked to exist and tied to real tales; motif-absent passages are twins; a hidden-state comparison joined the falsifier. Remaining: every admission passage is machine-written, so AT-27 must follow. IN21 is the natural pilot. The TiMoS download was not re-checked.

## Cross-area ranking

In the top twelve of 1 of 3 judges (upside: rank 9). A public catalogue of meme units with measured hit and false-alarm rates for 150 human-named motifs would be the lasting, citable asset of the programme; run it only after IN-21 shows the yield is not zero.

## Overlaps with

Asks nearly the same question as IN-21, which the judges would run instead.

## Sources

- Cammarata 2026, Neural networks are already clean
- Choi et al. 2024, Scaling automatic neuron description (Transluce)
- Heap et al. 2025, arXiv:2501.17727
- Hagedorn and Darányi 2022, doi:10.5334/johd.78
