# US-25: A drift meter that ignores rewording and catches change

**Area:** Uses · **Status:** design registered 8 November 2026, not run · **Priority:** 3.5 of 5

## Question

Can a neuron-based distance between two bodies of text stay flat when the same kind of stories arrive in other words or another language, and rise when the mix of stories changes?

## Prediction

With the alarm set so that fresh stories of the same genre mix from another language edition trigger it 5% of the time, the neuron meter catches a change in genre mix almost as often as the embedding meter: at the dose where that meter catches half of changes on development stories, its catch rate is at most 0.10 lower.

## What would refute it

At that dose the neuron meter's catch rate is more than 0.10 below the sentence-embedding meter's, or below 0.25, or the meter reads higher on same-mix stories from another edition than on a true change of that dose.

## Data

Tell Me Again (Hatzel and Biemann 2024): 96,831 summaries of 29,505 stories from five Wikipedia language editions, shipped with machine translations and name-masked versions; summaries CC BY-SA 4.0 per the dataset's README. Genre labels from Wikidata (CC0).

## Method

Reference: 400 English summaries. Comparison sets: different stories from another edition, in its own language, with the same genre mix (null arm) or 1 to 20% swapped for a genre absent from the reference (change arm). Meter: distance between profiles of how often each unit fires; units are picked on other stories for firing alike across editions. 500 resamples.

## Cheap rival it must beat

Word-frequency divergence on machine translations into English, multilingual sentence-embedding distance, a classifier two-sample test on embeddings, an LLM shown 30 texts from each set, distance between averaged internal states, text length. Embedding distance is most dangerous. If translated word counts match the neurons, drift monitoring needs neither neurons nor a multilingual model.

## Premise it tests

Both. Flat readings across languages need language-independent units (P1); repeating the result in Llama through mapped units tests P2. A fail says unit profiles track wording and edition, not content.

## Biology analogue

Allele-frequency change against sampling noise.

## Cost

S. About 6,000 summaries per model; the resampling is arithmetic.

## Payoff

A calibrated alarm for content pipelines and localised copy, with a known false-alarm rate under rewording, and the units that moved as a first diagnosis.

## Main risk

Genre mix is a coarse stand-in for themes, and language editions differ in length and detail, which every method may read as drift.

## Reviewer note

Reference and comparison now hold different stories, so no method wins by pairing copies; doses lowered and the test dose fixed where the embedding meter is at half power, to avoid a ceiling; translated word counts added as a rival. Weakness: genre is a crude kind of drift.

## Sources

- Hatzel and Biemann 2024 (Tell Me Again)
- Jiang et al. 2025
- Zhong et al. 2024
