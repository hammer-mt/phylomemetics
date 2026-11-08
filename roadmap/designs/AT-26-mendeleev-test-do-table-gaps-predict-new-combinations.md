# AT-26: Mendeleev test: do table gaps predict new combinations?

**Area:** The atlas · **Status:** design registered 8 November 2026, not run · **Priority:** 2.5 of 5

## Question

Do combinations of memes that a table built from older films marks as expected but never yet used turn up in films released after the model was published more often than unused combinations of equally common memes?

## Prediction

The 500 top-ranked gaps are realised in new films at least 2.0 times as often as 500 frequency-matched unused combinations, and this lift exceeds the keyword table's lift by 0.3 or more.

## What would refute it

Lift under 1.3, or within 0.1 of the keyword table's lift (bootstrap over films, 95% interval includes zero). A backtest inside the old pool (fit to 2005, test 2006 to 2012) fixes combination size so that matched combinations are realised 2% to 10% of the time; at 4%, 500 per arm detect a lift of 2.0 with about 80% power.

## Data

Old pool: CMU Movie Summary Corpus, 42,306 Wikipedia film plots to 2012 with release dates (CC BY-SA), plus a sample of about 15,000 Wikipedia plots for 2013 to mid-2024 (CC BY-SA 4.0). Test: plots of films released after Llama 3.1 was published in July 2024; expected over 1,000, count fixed at scouting. Elements: the motifs with an admitted AT-21 entry.

## Method

Each instrument codes the same motifs across the old pool. A factor model with genre and decade terms ranks motif pairs or triples never seen together. A gap is realised only if a blind cross-family reader, given definitions, finds all its motifs in one new film; the reader codes every new film. Null: unused combinations matched on frequency and genre.

## Cheap rival it must beat

The same table built from keyword lists, from e5 embedding prototypes, from an LLM reader's coding of a 3,000-film sample and from random non-meme units; an LLM simply asked to list likely unused combinations; plain frequency. Frequency and the keyword table are the dangers. If keywords match units, the periodic structure belongs to stories, not to neurons.

## Premise it tests

Assumes P1 for the frozen units and asks whether meme space has predictive structure at all. Units ahead supports P1's use for discovery. A pass for every instrument is a result about stories. A fail everywhere says co-occurrence does not predict new combinations.

## Biology analogue

Mendeleev's eka-elements; linkage disequilibrium

## Cost

M. One extraction over about 60,000 plots keeping only frozen units (roughly ten laptop hours), LLM-reader coding of the new films and of a 3,000-film sample for the rival table (under 100 US dollars); about two weeks.

## Payoff

A list of meme combinations predicted before anyone used them, with a measured hit rate. If it works, the atlas becomes a tool for finding open creative ground, which is the practical promise of the whole programme.

## Main risk

The factor structure may be nothing more than genre: gaps are motifs from one genre that had not yet met. Genre terms reduce this but do not remove it. If AT-21 admits few entries, or motifs are rare in films, too few gaps exist or too few are realised in two years of new films.

## Reviewer note

Changed: at the study's 3% presence bar almost no pair is unseen in 57,000 films and chance alone 'realises' over half of pairs, so gaps are now combinations sized by a backtest, and a blind reader, not the unit, decides realisation. Remaining: depends on AT-21's yield; genre may explain everything.

## Sources

- Bamman, O'Connor and Smith 2013, CMU Movie Summary Corpus
- Sreenivasan 2013, doi:10.1038/srep02758
- Li et al. 2024, arXiv:2410.19750
- Uzzi et al. 2013, doi:10.1126/science.1240474
