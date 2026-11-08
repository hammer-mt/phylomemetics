# LI-26: Open-pool search for documented loose influence

**Area:** Lineage · **Status:** design registered 8 November 2026, not run · **Priority:** 2.5 of 5

## Question

Can a search with no candidate list, using neuron memes (story units read from a language model's neurons) and calibrated on credited adaptations, find documented inspirations that were never formal adaptations among thousands of earlier works, better than word search, embeddings or fame alone?

## Prediction

With a shortlist of 30 from roughly 3,000 earlier works (a 1% false-alarm threshold set on development data), neuron memes recover at least 30% of documented inspirations, at least 9 percentage points more than the best of word search, sentence embeddings and a fame-only ranking.

## What would refute it

Recall in that shortlist is not above the best cheap rival (paired one-sided p at or above 0.05), or is under 10%. 150 test pairs give about 80% power for a 9-point gap when methods disagree on 20% of pairs. Not run with fewer than 150 test pairs having adequate plots on both sides.

## Data

Targets: Wikidata 'inspired by' links (CC0) with no 'based on' link for the pair. Reviewer's query, 2026-11-08: 3,713 pairs have English articles on both sides; 528 are dated, plot-bearing works, many of them video games, parodies and television formats; 213 targets are film or television. Pool: the 996 random films in `preregistration/culture_pool.json` plus about 2,000 documented source works, names masked.

## Method

Each target is scored against every earlier work. The threshold is fixed on development 'based on' pairs so that 1% of unlinked earlier works pass: about 30 false alarms per target, a shortlist and not a verdict. Report recall in the top 30 and top 5, parodies apart. One person, blind to method, checks 50 undocumented top hits.

## Cheap rival it must beat

Word search (BM25), sentence embeddings (a general model and story-emb), BM25 shortlisting 20 works for a local LLM to read, a fame-only ranking (how often a work is a documented source), plot length, random non-meme units and the averaged internal state. Most dangerous: fame alone and shortlist-then-read, since inspirations are famous works; results are split by fame.

## Premise it tests

P1. Finding influence nobody credited is the use memetics most needs and the study never tested. A pass shows a detector tuned on declared descent transfers to looser influence at a stated error rate. A fail, or a tie with embeddings or fame, means similarity search already does this and neurons unlock nothing here.

## Biology analogue

Paternity testing, validated on recorded pedigrees.

## Cost

M: one to two weeks on one laptop. About 3,500 plots to fetch and name-mask, extraction about 30 minutes per model, and about 3 hours for one human reader.

## Payoff

A shortlist tool for candidate influences with a measured hit rate and false-alarm rate, and a first list of candidate uncredited influences to check, including among Batman's documented relatives.

## Main risk

Wikidata's 'inspired by' links are sparse, favour famous works and often record a mood or image that no plot summary shows, so recall is low for every method and the comparison is uninformative.

## Reviewer note

Changed: fame-only rival added; decoys are random films and documented sources, since over half the committed culture pool is Batman-like genres; the 1% threshold is restated as a 30-work shortlist; minimum cut to 150 pairs. Remains: 'inspired by' links are declared and sparse, so undeclared influence stays unvalidated.

## Overlaps with

Asks nearly the same question as RX-22, which the judges would run instead.

## Sources

- Wikidata query service counts, 2026-11-08
- preregistration/culture_pool.json
- Chang et al. 2024
- Kelly et al. 2021
