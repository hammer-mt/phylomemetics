# EN-26: Knock one motif out of the weights

**Area:** Meme engineering · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

Can a small edit to a model's weights stop it writing one story motif, including in paraphrase, with less side damage than a system prompt or a word ban?

## Prediction

Removing one direction from under 0.1% of the weights halves the motif's rate in free story writing, but a system prompt forbidding the motif leaves the rate at least 10 points lower still, with no more loss of story quality, averaged over 12 motifs. Switching off the motif's 10 best detector units changes nothing.

## What would refute it

Averaged over motifs, the weight edit leaves the motif rate at least 10 points below the prompt's at equal or smaller quality loss, with a 95% interval above zero. Or the edit fails outright: a fall of under 20%, no better than random directions. Twelve motifs give about 75% power if motifs differ with a spread of 12 points.

## Data

Apache 2.0 weights: Qwen3.5-0.8B (the model edited in the MIT post), Qwen3-1.7B and Qwen3-8B (the study's models). 12 motifs from `preregistration/memes_for_tracing.json`, kept only if the unedited model writes them on at least 40% of tuning prompts. Per motif, 300 LLM-written story prompts that invite it without naming it (100 to find the direction, 200 held out) and 100 unrelated prompts.

## Method

Direction: mean activation difference between stories with and without the motif, subtracted from the output weights of two layers, as in the MIT post. On 4-bit Qwen3-8B it is applied at run time. A second arm zeroes the motif's 10 best detector units. A blind, human-checked judge from another model family scores the motif as an idea, not a word.

## Cheap rival it must beat

A system prompt forbidding the motif (most dangerous: prompts and filters matched fine-tuning for forgetting concepts in Thaker et al. 2024). A ban on the motif's keywords, which should leave paraphrases. A fine-tune on motif-free stories. 20 random directions. If the prompt wins, weight edits are a curiosity, but the paraphrase result still shows whether an idea has a location.

## Premise it tests

P1 in its direction form. A clean knockout shows a motif has a compact causal home in the weights. A fail means motifs are spread out or rebuilt elsewhere, so 'the meme is a unit' holds for reading at most.

## Biology analogue

Gene knockout.

## Cost

M. One to two weeks on the laptop, plus judge calls. An agent-found edit, as in the MIT post, would be L.

## Payoff

Meme-free and meme-dosed model variants to use as control writers for planted-story tests, and a first check of weight-space editing on meaning.

## Main risk

Sub-2B models write weak stories and may ignore a 'do not' instruction, so the prompt could lose for a reason that vanishes at 8B; the 8B run decides. The MIT authors found edits remove obstacles and cannot add abilities.

## Reviewer note

The '6 of 10 motifs' rule could not tell a win from a coin flip; replaced by a mean difference across 12 motifs. Added an 8B run, a base-rate gate and a detector-unit knockout arm. Remaining weakness: the prompt is expected to win, as in work on making models forget.

## Sources

- Takehi, Teoh, Zhang, Khattab 2026, Sparks of weight-space programming
- Arditi et al. 2024, arXiv:2406.11717
- Thaker et al. 2024, Guardrail baselines for unlearning, arXiv:2403.03329
- Hase et al. 2023, arXiv:2301.04213
