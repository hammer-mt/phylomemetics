# ME-26: Do memes survive retelling better in bundles

**Area:** Old memetics, re-run · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

Does a story element survive a chain of retellings more often when its usual partner element (say, a vow of revenge beside an orphaned hero) is in the story than when an unrelated element takes the partner's place?

## Prediction

After four retellings an element survives at least 10 percentage points more often in a whole bundle than in a broken one, the gap is positive in at least 17 of 24 bundles, and a neutral filler element shows less than half that gap.

## What would refute it

A gap under 3 points, or a 95% interval (resampled by bundle) that includes zero, over 1,440 chains from 480 seed stories (standard error about 3.5 points). That would mean elements are lost one by one, and the memeplex adds nothing to its parts in transmission.

## Data

Built for the test. 24 two-element bundles fixed in advance, from the trope pairs that co-occur most in the TV Tropes tables of Gala et al. 2020 (pair list only; the site's licence is non-commercial). Each gets 10 outlines, written whole and broken: 480 name-free 300-word stories, 3 chains of 4 retellings, 5,760 texts. Released with its lineage.

## Method

A memeplex is a set of memes said to help each other spread. A coin flip decides whether the partner is present, so topic cannot explain the gap. Survival at the last step is read by units chosen on separate planted-element stories and checked against 400 human-labelled retellings. Statistic: survival in whole minus broken stories, paired by outline.

## Cheap rival it must beat

A keyword list for each element, an embedding probe, an LLM reader asked whether the element is present, text length, random units, the mean-pooled residual stream. The LLM reader is most dangerous, and the keyword list will win for elements with an obvious word. If either wins, the bundle effect is still the first randomised test of co-adapted memes.

## Premise it tests

P1. The bundle effect itself does not depend on the unit. Units earn a place only if units chosen on planted stories track an element through four rewordings as accurately as the LLM reader, judged against the human labels.

## Biology analogue

Fitness epistasis; co-adapted gene complexes

## Cost

M. About 6,500 short texts from local 4-bit models, two to three days of laptop time, plus half a day labelling 400 retellings.

## Payoff

A causal number for whether memes in a complex protect each other in transmission, and a list of bundles that behave as larger units.

## Main risk

Model tellers are not human minds, and four short retellings may keep everything (chain length is set on tuning stories). Memory research already shows that causally linked events are recalled better (Trabasso and van den Broek 1985), so a pass may be old news under a new name.

## Reviewer note

The citation version could not fail: units that co-fire inside the model, and words in set phrases, stay linked whatever culture does. Rewritten as a randomised test of whether a partner element protects an element in retelling. Remaining: model tellers; memory research partly predicts the result.

## Sources

- Dawkins 1976
- Blackmore 1999
- Trabasso and van den Broek 1985
- Gala et al. 2020 (TV Tropes dataset)
