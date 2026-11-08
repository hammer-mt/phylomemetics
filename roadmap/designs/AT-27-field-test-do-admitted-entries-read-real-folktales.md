# AT-27: Field test: do admitted entries read real folktales?

**Area:** The atlas · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

Does an entry admitted on passages written to order also find its motif in real folktales whose tale type lists that motif, and does passing admission predict how well it does there?

## Prediction

Of entries admitted in AT-21, at least half reach AUC 0.70 (0.5 is chance) on real tales, against at most 20% of entries that failed admission. The mean AUC of admitted entries is at least 0.10 above that of rejected ones.

## What would refute it

Admitted and rejected entries differ by under 0.03 in mean AUC (bootstrap over motifs, 95% interval includes zero), or under a quarter of admitted entries reach 0.70. With about 25 admitted and 75 rejected motifs and a spread of 0.12 in AUC between motifs, a 0.10 gap is detected with about 95% power.

## Data

Hagedorn's trilogy data (Zenodo 6575263, CC BY-SA 4.0): 1,518 English folktale texts of 182 tale types, and a table tying each type to its Thompson motifs. 307 motifs are tied to ten or more of the tales (counted 8 November 2026); AT-21 draws 100 of them. A tale counts as containing a motif if its type lists it.

## Method

Frozen AT-21 entries and thresholds; nothing is tuned on tales. Per motif, AUC for tales whose type lists the motif against tales of other types in the same index division, so topic cannot carry it. Compare admitted with rejected entries. One person reads 200 tale-motif pairs to measure label error. Null: motif labels permuted across types within a division.

## Cheap rival it must beat

Each built from AT-21 passages or the definition alone, never from tales: a keyword list, an e5 prototype (the average embedding of motif-present passages), a hidden-state probe, a zero-shot LLM reader, tale length, random units. The keyword list is the danger: real tales mostly use the usual words, so keywords may match units here although they fail AT-21 by construction.

## Premise it tests

P1. Admission on machine-written text is worth having only if it predicts behaviour on real text. A pass makes AT-21's rule a usable gate and its entries citable for folklore. A fail means the catalogue describes one writer model's passages, not stories.

## Biology analogue

Mock community first, then field samples

## Cost

S. One extraction over 1,518 tales (about 20 minutes), a few thousand reader calls (under 20 US dollars), three hours of reading by one person; under two days once AT-21 exists.

## Payoff

An error rate for each catalogue entry on real, expert-indexed stories, and a check that the admission rule means something outside the laboratory. It also gives the first table of which Thompson motifs a model unit can find in the tales themselves.

## Main risk

Labels sit at the level of the tale type: a given telling may lack a listed motif, and only 37 motifs are shared by two or more of the collection's types, so a unit that recognises 'a Cinderella tale' passes as the slipper motif. Same-division negatives reduce this; they do not remove it.

## Reviewer note

Added by the reviewer, because AT-21 admits entries on machine-written text only. Weakness: labels are per tale type, noisy and mostly tied to one type, and both models have probably read these tales online. Alyami and Finlayson's sentence-level corpus would be better; its public release was not confirmed.

## Sources

- Hagedorn and Darányi 2022, doi:10.5334/johd.78
- Hagedorn, trilogy dataset, Zenodo 6575263
- Alyami and Finlayson 2026, arXiv:2603.19283
- Callahan, McMurdie and Holmes 2017, doi:10.1038/ismej.2017.119
