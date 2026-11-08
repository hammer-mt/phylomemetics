# PO-23: Random-copying null and frequency-increment test on film plots

**Area:** Population memetics · **Status:** design registered 8 November 2026, not run · **Priority:** 3.5 of 5

## Question

Across eighty years of film plots, does any unit's frequency change more than random copying with innovation predicts, and does that verdict differ from the one its own words receive?

## Prediction

At 5% false discovery, in at least two of three time binnings, at least 1% of a 5,000-unit panel reject drift, and for at least 30% of those the unit's word shadow does not reject in the same direction. Before freezing, simulated power must reach 0.8 for a doubling of odds over 40 years.

## What would refute it

Under 0.25% reject (drift suffices at this power), or over 90% of rejecting units are mirrored by their word shadows. Void if the control with release years shuffled within genre, or simulated drift with threshold noise, rejects more than 1% of units.

## Data

CMU Movie Summary Corpus: 42,306 English Wikipedia plot summaries (November 2012 dump) with release date, genre and country from Freebase; 46 MB; CC BY-SA. Median length is 9 sentences; 20,933 have ten or more, leaving 410 to 3,937 films per 5-year bin from 1930 (reviewer count, 2026-11-08). One dump, so the editors' era is constant.

## Method

Mask names; keep plots of ten or more sentences, cut to ten. On 2,000 development films, merge co-firing units and set each unit's bar at its top fifth. Frequencies in 5-year bins, resampled to equal size and a constant genre-by-country mix. Fit drift; apply the frequency-increment test, a t-test on variance-scaled frequency changes. Word shadow: presence predicted from words alone.

## Cheap rival it must beat

The identical test on each unit's word shadow, on frequency-matched word types, LDA topics and embedding clusters; an LLM tagging 50 motifs on 2,000 films; random non-meme units. Most dangerous: words, because most strong units are word detectors. If words flag the same items we still gain the first drift baseline for plots.

## Premise it tests

assumes both. A pass licenses later selection claims and shows units move differently from words; mirrored verdicts mean P1 adds nothing here. P2 gets a functional test: Qwen and Llama twins should receive the same verdict (kappa at least 0.4).

## Biology analogue

Kimura's neutral theory; frequency-increment test on allele time series (Feder 2014).

## Cost

M. One pass over about 21,000 summaries, roughly an hour per model, stored at one bit per unit per film (about 2.5 GB); a week of analysis and power simulation.

## Payoff

A fitted drift baseline for film plots and a short, binning-stable list of units whose rise or fall drift cannot explain: the entry ticket for any selection claim.

## Main risk

Equifinality and survivorship: Wikipedia covers the old films that stayed famous, which mimics selection; binning changes verdicts (Karjus 2020); 16 bins give little power per unit, so a low count may mean weak power, not drift.

## Reviewer note

'First 15 sentences' left most plots shorter (median 9): now exactly ten. The 3% bar gave about 12 present films in early bins: now top fifth. Added equal bin sizes, a multi-word shadow rival, a 1% void threshold. Remaining: Wikipedia survivorship looks like selection.

## Sources

- Bentley, Hahn and Shennan 2004
- Feder et al. 2014
- Karjus et al. 2020
- CMU Movie Summary Corpus (Bamman, O'Connor and Smith 2013)
