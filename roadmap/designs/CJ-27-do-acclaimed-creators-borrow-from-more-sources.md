# CJ-27: Do acclaimed creators borrow from more sources?

**Area:** Field conjectures · **Status:** design registered 8 November 2026, not run · **Priority:** 2.0 of 5

## Question

Do directors whose films draw on a wider spread of earlier works earn more critical recognition, and within one director's career are the recognised films the broader ones?

## Prediction

Within a director's films, one standard deviation more source breadth (the effective number of distinct genre-and-decade groups among a film's 20 closest earlier sources) raises the odds of an award nomination by at least 10%, with summary length and release year controlled. Between directors, mean breadth predicts nominations in the same direction.

## What would refute it

The within-director effect's 95% interval lies below 3% per standard deviation, or is negative, for every unit type: the hunch fails. For neurons: neuron breadth adds no held-out likelihood over breadth from trope tags and embeddings (interval includes zero). If the declared-source gate fails, the test is not run.

## Data

CMU Movie Summary Corpus (42,306 Wikipedia plots, CC BY-SA), summaries cut to one length band. Directors, nominations and awards from Wikidata (P57, P1411, P166; CC0): 6,950 films with a Freebase ID carry a nomination (counted 2026-11-08). Expect about 3,000 directors with four or more films, where a 10% effect is roughly three standard errors (an estimate); count before registering.

## Method

Sources: earlier films ranked by inheritance of a film's rare units (the study's heredity statistic), own films excluded. Gate: where Wikidata declares a source (P144, P941), the ranking must put it in the top 20 at least 20% of the time (chance: about 0.1%). Breadth: effective number of groups in the top 20. Null: breadth shuffled among each director's films.

## Cheap rival it must beat

Breadth from TF-IDF neighbours, sentence-embedding neighbours, the mean-pooled hidden state, TV Tropes tags, the count of declared Wikidata sources, an LLM asked how many sources a plot draws on, the film's own genre count and summary length. Genre count is most dangerous: breadth may only mean 'a film that mixes genres'.

## Premise it tests

Assumes both. Resemblance is not influence, so this tests a correlate of the hunch. P1 gains only if neuron breadth predicts acclaim after tag and embedding breadth are in the model and the declared-source gate passes.

## Biology analogue

Hybrid vigour; gene flow from many source populations.

## Cost

M. Reuses the corpus pass (about 6 hours per model); a week for the Wikidata join and the declared-source gate.

## Payoff

Evidence for or against advice given to every creative team: read widely, borrow widely. A per-creator breadth profile that can be checked against declared influences.

## Main risk

Awards track fame, budget and prestige genres, and Wikidata records them unevenly. Famous films have longer, richer summaries, which can fake breadth even after trimming. A 10% effect sits near the detection limit.

## Reviewer note

Sources came from the method itself, and famous films have longer summaries, which fakes breadth. Added a numeric declared-source gate, equal-length summaries and a within-director comparison that holds fame fixed. Remaining weakness: resemblance is not borrowing, and Wikidata awards are patchy and fame-driven.

## Sources

- Uzzi et al. 2013
- Barron et al. 2018
- Wikipedia: List of Bob Dylan songs based on earlier tunes
- Wikidata (nominated for P1411, award received P166)
