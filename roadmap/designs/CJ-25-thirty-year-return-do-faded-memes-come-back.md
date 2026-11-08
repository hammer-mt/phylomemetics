# CJ-25: Thirty-year return: do faded memes come back?

**Area:** Field conjectures · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

After a story meme peaks in films and fades, does its share rebound about 30 years later more than at other lags, and is the rebound carried by directors who were young at the peak?

## Prediction

For memes whose share peaked between 1930 and 1965 and then halved, share 25 to 35 years after the peak is at least 10% higher than the mean of the flanking lags (10 to 20 and 40 to 45 years), in at least three peak decades, and beats 95% of simulations where films copy earlier films' memes by frequency alone.

## What would refute it

The rebound's 95% interval lies below 3%, or inside the simulated range, for every unit type: no 30-year return in film plots. For neurons: the neuron rebound does not exceed the rebound for TF-IDF terms or trope tags (difference interval includes zero). About 200 unit clusters should put a 10% rebound at three standard errors or more (rough estimate).

## Data

CMU Movie Summary Corpus: 39,589 dated Wikipedia film plots, CC BY-SA; 8,373 fall in 1930 to 1965, about 230 a year, so shares use five-year bins. Director birth dates from Wikidata (CC0): 19,200 of a non-random 20,000-film sample had one (2026-11-08). Remake lags from Wikidata 'based on' links (1,499 film-to-film) as a text-free check.

## Method

Units chosen on a random third of films; series built on the rest. Fade: a five-year peak at least 1.5 times the unit's median share, then 10 years below it. Return curve: mean share at each lag. Null: random copying with real yearly counts. Secondary: is the rebound larger in films by directors aged 5 to 25 at the peak?

## Cheap rival it must beat

The same return curve from TF-IDF terms, trope tags and genre labels; the histogram of remake lags, which needs no text, against a null that respects how many films existed each year; random units. TF-IDF is most dangerous: revivals reuse period words. If remake lags alone show the 30-year peak, no unit is needed.

## Premise it tests

Assumes both. Mostly a test of the conjecture. P1 gains only if neuron units show a rebound among name-masked plots where word units show none; P2 only if Qwen and Llama give the same return curve. A known rival view: retellings prefer young sources (Karsdorp and van den Bosch 2016).

## Biology analogue

Seed banks releasing dormant variants; sensitive-period imprinting.

## Cost

S to M. Reuses the corpus pass (about 6 hours per model). Series and simulations take minutes; the Wikidata join for the secondary takes a day.

## Payoff

A number for when a faded meme comes back, which Mike wanted for timing a revival, or evidence that the 30-year cycle is folklore. If the secondary holds, a forecast rule: watch what peaked when today's 45-year-old creators were 15.

## Main risk

One shared cause can fake a cycle: war memes peak in the 1940s and return with 1970s war films. Hence the three-decades rule, but only about four peak decades exist. Wikipedia's coverage of old films favours those later revived or remade.

## Reviewer note

The draft tested a mechanism (directors' youth) before showing any return exists, and period settings confounded it. Rewrote as a direct return-lag test with the director test as a secondary; the mechanism is kept as a seed. Remaining weakness: few independent peak decades, and Wikipedia favours revived films.

## Sources

- Holbrook and Schindler 1989
- Candia et al. 2018 (Nature Human Behaviour)
- Karsdorp and van den Bosch 2016
- CMU Movie Summary Corpus
