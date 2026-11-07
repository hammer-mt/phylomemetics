# What changed from version 1 to version 2

Version 1 (20 October 2026) is frozen: `../paper.md`, its PDF, its figures and its Claude Doc are unchanged, and the git tag `paper-v1` marks it. Version 2 re-runs the whole study with a method aligned to the essay that inspired it (Cammarata, "Neural networks are already clean"), and reports what that did to each of version 1's conclusions. The analysis plan, `../../v2/PLAN.md`, was committed before any of the new models read a study text.

## Why

Version 1's own limitations section, its robustness review and use of the results since publication pointed at the same weak spots:

- The models were small (8B parameters) and quantised to 4 bits. The essay studies Qwen3-32B and says larger models have cleaner neurons.
- A meme counted as present in a work whenever the work beat 68 of 69 ordinary films on a neuron, however weak the activation. The essay's point is that strong activations are what carry meaning.
- "Ordinary" was 69 hand-picked films, which also set the bar. Two comparisons turned out to be true by construction and had to be withdrawn.
- Neuron labels were read off eight sentences of the Batman corpus itself, and several were narrower than their neuron.
- Nothing was tested by steering or ablation.
- Twins in a second model were found and scored on the same sentences.
- The order of plan and results rested on a session transcript, not on the repository's history.

## What changed in the method

| | Version 1 | Version 2 |
| --- | --- | --- |
| Models | Qwen3-8B and Llama 3.1 8B, 4-bit weights, on a laptop | Qwen3-32B (primary), Qwen3-8B, Llama 3.1 8B and Llama 3.3 70B, at full precision on rented GPUs |
| Unit | One pole of a neuron, whatever produced the activation | One pole of a neuron, counted only where the gate is open (the essay's main sign regimes) |
| Present | The work beats 68 of 69 reference films | The unit fires strongly: at least 30% of its strongest activation anywhere in a 48-million-token corpus |
| Ordinary stories | 69 films, which also set the bar | 2,999 film plots that are not study works give every unit's base rate |
| Which units | All | Distinctive units (base rate of 3% or less) for the canon and identity; all units for heredity and selection |
| Explanations | Read off 8 Batman sentences, checked by a second reader | Written from the unit's strongest examples in the reference corpus, for units chosen by rule, then tested on new texts where they can lose |
| Causal tests | None | Steering of the explained units; ablation of Batman's canon |
| Recognition | Not measured | Each model is asked which hero and which work a name-free summary is about |
| Words as a rival | Two word measures | The same two, plus the further measures needed so that none of the word-only measures we tried passes the "beyond words" test |
| Length | Checked by one control | Every identity score is also computed on openings of equal length |
| Twins across models | Scored on the sentences they were chosen on | Scored on new texts and on name-free sentences, against word-matched controls |
| Plan | Recorded in a session transcript | Committed to the repository before the study pass; changes logged |
| Extraction check | None | Our activations match the essay's published ones (r = 0.992 on 756 examples) |

The corpus (32 Batman works, 13 relatives, 69 films, and their name-free versions), the three experiments and their statistics are unchanged.
