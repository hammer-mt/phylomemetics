# US-26: Do empty meme combinations predict what gets made next?

**Area:** Uses · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

Are idea briefs built from meme combinations that no earlier film used realised by later films more often than briefs from random pairs, embedding gaps or simply asking an LLM?

## Prediction

Among gaps as of 2012 with no earlier match, neuron-gap briefs are realised by films of 2013 to 2023 at least 1.5 times as often as random-pair briefs matched on unit frequency, and at least as often as embedding-gap briefs. At most 20% of neuron gaps turn out to have an earlier match.

## What would refute it

The neuron-gap realisation rate is not above the random-pair rate (one-sided two-proportion test, p at or above 0.05, 1,000 briefs each), or is below the embedding-gap rate by a third or more, or over 20% of neuron gaps already existed. If the tuning period's random-pair rate is under 2%, the test is not run.

## Data

CMU Movie Summary Corpus (42,306 plots to 2012, CC BY-SA). Wikipedia film plots from 2013 to 2023, still to be crawled, and from 2025 (2,403 film articles by a Wikidata count on 8 Nov 2026), CC BY-SA 4.0. Names masked; sequels, remakes and adaptations removed using Wikidata links.

## Method

Tune on gaps as of 2000, judged against 2001 to 2012: fix units, ranking rule (two common units that share many partners and never met) and base rates. Test on gaps as of 2012: 1,000 per method, stated by a fixed template; an LLM judge, checked against 100 blind human labels, reads retrieved plots. The LLM-list rival: 2025 films only.

## Cheap rival it must beat

Random never-met pairs matched on unit frequency, pairs ranked by the product of unit frequencies, gaps between sentence-embedding clusters, word-pair gaps from TF-IDF terms, sparse-autoencoder feature pairs, Llama 3.1 asked for unmade film ideas. The frequency-product baseline and the plain LLM list are most dangerous. If they win, meme maps add nothing to idea generation beyond popularity.

## Premise it tests

Assumes both. A pass shows unit co-occurrence captures which combinations are viable, which words and embeddings may miss. A fail does not hurt P1 but removes the creativity use.

## Biology analogue

Vacant niches filled later by adaptive radiation.

## Cost

M. About 55,000 plots through one model, 6,000 briefs, 120,000 short LLM matching calls, one hour of human checking.

## Payoff

A test of gap-finding as an idea source that needs no taste judgement: the later record of what was made is the judge.

## Main risk

Common units that never met may exclude each other, not await a first use; unit labels may be too vague to judge; and being realised later rewards obvious combinations more than good ones.

## Reviewer note

The gap rule was undefined and the 8% and 4% rates had no basis. Added a tuning period fixing rule and base rates, a stop gate, name masking, template briefs and a test on 2013 to 2023 films, since few 2025 plots exist. Weakness: a long chain of LLM steps.

## Sources

- Sreenivasan 2013
- Uzzi et al. 2013
- Krenn et al. 2022
- CMU Movie Summary Corpus
