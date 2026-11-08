# IN-23: Presence by the neuron's own strong regime

**Area:** The instrument · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

Are the study's presence calls made in each unit's strong regime, and do the units' labels hold only there?

## Prediction

At least 30% of presence calls for the 125 labelled units fall below 30% of the unit's own maximum, and blind label-matching accuracy is at least 15 points lower for sentences at 5 to 20% of maximum than for sentences above 50%.

## What would refute it

Under 10% of presence calls fall below 30% of maximum (the film-relative bar already selects the strong regime); or accuracy differs by under 5 points between bins; or a word-overlap judge shows the same drop, so the bins differ in words, not meaning.

## Data

FineWeb sample (ODC-By): 10 million tokens for the bins, 100 million for the maxima of the tested units only. The study corpus of 3,700 Wikipedia plot sentences (CC BY-SA) and its 125 labelled units (study/results/labels.json, labels_wf.json). 500 word-matched and random active units as controls.

## Method

Maximum: largest value over general text and the study corpus; report its growth from 1 to 100 million tokens. Express each presence call as a fraction, flagging firings made by two negative parts multiplying. Blind judges match the label to a binned held-out sentence against a foil under 5% sharing as many of the label's words. Bootstrap over units.

## Cheap rival it must beat

Word-overlap judge per bin, embedding similarity of label to sentence, an LLM reading with no activations, random units through the same bins. Most dangerous: word overlap in the strong bin, because a name unit's strong firings are the name. Foils matched on word overlap are there to expose that.

## Premise it tests

P1. If presence sits in the weak regime where labels fail, the study measured something its source essay does not vouch for, and the presence rule must change.

## Biology analogue

Calling a gene expressed against its own baseline, not another tissue's

## Cost

S to M. One 10-million-token pass per model (about two hours), one 100-million-token pass tracking only the tested units (about a day unattended), about 4,000 judge calls.

## Payoff

A presence rule that does not depend on a reference set of films (a fixed fraction of the unit's own range), or evidence that the film-relative bar was adequate.

## Main risk

Batman-specific units (names, places) may never meet their trigger in general text, so their range comes from the study corpus alone and the test says little about them.

## Reviewer note

Two repairs. Foils that merely contained the label's words made the weak bin harder by construction; foils are now matched on overlap. A 10-million-token maximum understates rare units, so maxima now use 100 million tokens plus the study corpus. Labels still come from Batman text.

## Cross-area ranking

In the top twelve of 1 of 3 judges (what decides the premises fastest: rank 10). Checks whether the study's presence calls sit in the strong range the source essay vouches for, which fixes the presence rule that every document-level design inherits.

## Sources

- Cammarata 2026
- Bolukbasi et al. 2021
- Nanda 2026 (LessWrong comment)
- study/discover.py (presence rule)
