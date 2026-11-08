# US-22: Real dog whistles: routing unseen coded terms

**Area:** Uses · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

On real coded speech, can units chosen on some coded terms name the group targeted by terms never seen during selection, when the term itself is blanked out?

## Prediction

With the term blanked, neuron units name the right one of four target groups for held-out terms with macro-F1 (a 0-to-1 accuracy score averaged over the groups) at least 0.05 above a TF-IDF classifier on the remaining words and within 0.05 of the embedding probe.

## What would refute it

Term-level macro-F1 is not above the TF-IDF classifier (paired bootstrap over terms, lower 95% bound at or below zero), or is more than 0.05 below the embedding probe, or is no better than a classifier told only the forum or chamber.

## Data

Silent Signals (Kruk et al. 2024): 16,258 real uses of 298 dog-whistle terms (coded phrases with a hidden meaning for an in-group), Reddit and US Congress, CC BY-SA 4.0. An LLM chose them; the authors judged 85% of 400 truly coded. Primary set: the four target groups with 40 or more terms (237 terms, 11,321 posts). Plus 124 human-labelled uses.

## Method

Five folds split by term: every term is tested once and never helps choose its own units. Posts are scored with the term blanked, then visible; a term's posts vote. Statistic: macro-F1 over terms, bootstrapped by term. Null: group labels shuffled across terms within source and decade. Secondary: coded against innocent uses of 13 terms in the human-labelled set.

## Cheap rival it must beat

Keyword list (useless once the term is blanked), TF-IDF classifier on the remaining words, forum or chamber alone, sentence-embedding probe, a local LLM asked which group is targeted, a probe on the model's averaged internal state, random units. The LLM reader is most dangerous, since an LLM chose these examples. A loss still shows whether context alone carries the code.

## Premise it tests

P1. A blanked term stands in for a newly coined word, so a pass means units read the target from context beyond what word counts give. A fail on real data would undercut the paused plan's invented code-word condition before it is run.

## Biology analogue

Camouflage against a predator's search image.

## Cost

S. 16,000 short posts; minutes per model, about an hour for the local LLM rival.

## Payoff

Evidence for or against using units to route newly coined coded language to the right policy category without retraining a filter.

## Main risk

Context may just be the forum: 47 forums, one supplying 44% of Reddit posts. The four groups overlap in meaning, which caps every method. All posts are coded uses, so false alarms on innocent text are tested on 124 items only. The text is offensive: detection only, nothing is generated.

## Reviewer note

Four of the eight groups had 6 to 15 terms, so the test now uses the four with 40 or more, five folds by term, the blanked term as primary and a forum-only rival. Weakness: an LLM chose the examples, innocent posts are scarce, forum identity may explain a win.

## Sources

- Kruk et al. 2024 (Silent Signals)
- Mendelsohn et al. 2023
- Steen, Yurechko and Klug 2023
