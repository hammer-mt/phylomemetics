# IN-26: A multitrait-multimethod matrix with human raters

**Area:** The instrument · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

Do unit readings agree with human judgements of the same named trope more than with other tropes, and how close do they come to human-human agreement?

## Prediction

Against adjudicated human labels, frozen units reach AUC (chance 0.5) of 0.75 or more for the median trope; unit-human kappa (chance-corrected agreement), pooled over tropes, is within 0.15 of human-human kappa; and same-trope agreement exceeds different-trope entries in at least 90% of comparisons.

## What would refute it

Median AUC under 0.65; or same-trope unit-human agreement below different-trope same-method agreement; or units trail the TF-IDF classifier against the human labels.

## Data

TiMoS (Chang et al. 2021): TV Tropes labels for about 5,600 films, repository MIT; synopses from IMDb, licence unclear, so commit film names and labels only. Test: 300 name-masked synopses from held-out franchises, 15 tagged and 15 untagged per trope, 10 plot-event tropes including near neighbours (revenge rampage, disproportionate retribution, karmic death); two paid annotators and an adjudicator.

## Method

Tropes fixed by a written rule before any activations: events visible in a plot, not production trivia. Choose a unit and a 5-unit set per trope on development synopses in two stages; freeze. Fill a 10-trope by 6-method table (neurons in two models, keywords, embeddings, LLM reader, humans); apply Campbell and Fiske's four checks. Nulls: shuffled labels; popularity-matched films.

## Cheap rival it must beat

Keywords, TF-IDF logistic regression, a sentence-embedding probe and a trope-by-trope LLM reader are rows of the same table; length and popularity are covariates. Most dangerous: the LLM reader, though tropes proved hard for GPT-4. If it wins, the table still gives each method's validity against the human ceiling.

## Premise it tests

P1. The first human evidence for or against the instrument. Losing to TF-IDF on human labels would leave P1 with inspectability only.

## Biology analogue

Calibrating the 16S threshold against DNA-DNA hybridisation, the older standard

## Cost

L. About 40 annotator hours for whole synopses (roughly 800 US dollars), a day of compute, two weeks elapsed; needs a short ethics and consent note for paid raters.

## Payoff

Human-human agreement on trope presence, without which no model-human figure can be read, and a validity coefficient for every method.

## Main risk

Humans reading synopses matched the TV Tropes labels at only F1 0.65 in the dataset's own paper, so human agreement may be too low to separate the methods; kappa errors per trope stay near 0.1.

## Reviewer note

Random passages would hold too few positives per trope, and labels are per film, so raters now read whole synopses, enriched by tag. Many TiMoS tropes are production trivia; a selection rule was added. Single neurons will probably fail; the human ceiling is the real payoff.

## Sources

- Campbell and Fiske 1959
- Chang et al. 2021 (TiMoS)
- Su et al. 2024
- Alyami and Finlayson 2026
