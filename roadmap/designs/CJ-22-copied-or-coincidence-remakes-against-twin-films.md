# CJ-22: Copied or coincidence: remakes against twin films

**Area:** Field conjectures · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

Do incidental 'hitchhiker' details shared by two similar films separate documented descent (remakes) from independent arrival at the same premise (twin films)?

## Prediction

With remake and twin pairs matched on overall likeness (sentence-embedding similarity), genre and length, a hitchhiker-sharing score separates them at AUC 0.70 or more (0.5 is chance) under two-fold cross-fitting (threshold set on one half of the pairs, scored on the other, then swapped), and neuron hitchhikers beat rare-word hitchhikers by at least 0.10 AUC.

## What would refute it

AUC below 0.60 for every hitchhiker definition: similarity cannot show descent in plot summaries. For neurons: AUC not above rare shared words (bootstrap interval of the difference includes zero). About 90 twin against 270 remake pairs gives a standard error near 0.035: enough to tell 0.70 from chance, and to see a neuron advantage only if it is about 0.10.

## Data

Wikipedia 'Twin films' list: 149 pairs (2026-11-08; CC BY-SA); about 110 after screening out shared sources and real events, perhaps 90 with usable plots. Remakes from Wikidata film-to-film 'based on' links (932 with both on English Wikipedia; CC0). Where available, the later film's plot comes from another-language Wikipedia, machine-translated (Tell Me Again, CC BY-SA 4.0). Names masked; dates withheld.

## Method

Hitchhiker: a unit shared by the pair but rare in the genre's other films (rarity set on films outside all pairs). Score: share of the earlier film's rare units found in the later one, discounting units common to many films (the study's heredity statistic). Each twin pair gets up to three likeness-matched remake pairs. Null: mismatched same-genre, same-decade pairs.

## Cheap rival it must beat

Rare shared words and named details, TF-IDF cosine, mean-pooled hidden state, random units, and an LLM asked 'remake or coincidence?', scored only where it cannot name either film, as it may remember them. Sentence-embedding similarity is neutralised by the matching. Rare shared words are most dangerous: remakes keep odd details as words. If they win, descent is readable without neurons.

## Premise it tests

P1. A meme must be copied, not merely present. A pass shows units carry lineage information that similarity lacks. If rare words do as well, P1 gains nothing. If every method fails, Wikipedia plots cannot support the study's influence claims at all.

## Biology analogue

Homology against convergent evolution; neutral markers such as shared pseudogenes.

## Cost

S to M. About 900 plots, minutes of forward passes. Cleaning the twin list, matching pairs and fetching translated plots is the work: three to four days.

## Payoff

A way to say 'this was copied' instead of 'this is similar', the gap in every influence claim so far. Also a reusable labelled set of descent and convergence pairs.

## Main risk

Twin films are often not independent: studios race from leaked scripts, so some 'coincidences' are copies. Remakes resemble their originals far more than twins resemble each other, so matching on likeness may leave too few pairs.

## Reviewer note

Counted the list: 149 rows, about 110 after a keyword screen for shared sources, not 150. Added cross-language plots against copied wording, a named matching measure and a memory guard for the LLM judge. Remaining weakness: about 90 twin pairs can resolve only a 0.10 neuron-over-word gap.

## Cross-area ranking

In the top twelve of 1 of 3 judges (upside: rank 8). Remakes against twin films is the one design that tries to tell 'copied' from 'merely similar', the gap in every influence claim, and it is a few days of work on real films.

## Sources

- Claidière, Scott-Phillips and Sperber 2014
- Wikipedia: Twin films (list)
- Wikidata (based on, P144)
- Hatzel and Biemann 2024 (Tell Me Again)
