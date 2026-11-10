# IN-21: Minimal pairs and decoys for thirty story concepts

**Area:** The instrument · **Status:** design registered 8 November 2026, not run · **Priority:** 5.0 of 5

## Question

For how many story concepts does a neuron chosen on development pairs respond to the concept, and not to its words or to any dramatic event, on held-out pairs?

## Prediction

In Qwen3-8B, the primary model, at least 15 of 30 concepts have a frozen unit that ranks the concept sentence above its matched control in 16 or more of 20 held-out pairs. Pooled over concepts, AUC (chance 0.5) is at least 0.80 on concept-without-its-words against words-without-the-concept. Llama 3.1 8B is reported as a replication.

## What would refute it

Fewer than 8 of 30 concepts reach 16 of 20; or units picked on shuffled development labels pass as often; or a unit wins as often on other concepts' pairs as on its own; or pooled mixed-cell AUC is under 0.70. Separately, a margin under 0.05 over the best embedding detector means the units add inspectability only.

## Data

Built by construction: 30 concepts with near neighbours (revenge, vow, harm). Per concept 60 minimal pairs (the concept told without its usual words; the same frame with another event of equal stakes) and 40 decoys using the words without the concept; 4,800 sentences. Written by an untested model family; 300 hand-checked. CC BY 4.0. 40 pairs development, 20 test.

## Method

Units: neuron poles (one sign of one MLP neuron) in Qwen3-8B, Llama 3.1 8B and Qwen3-1.7B. Shortlist 20 per concept on half the development pairs, pick one on the rest, hash before test items exist. Score held-out pair wins and pooled mixed-cell AUC; bootstrap over concepts. Nulls: shuffled-label selection, other concepts' pairs, word-matched units. Chance of 16 of 20: 0.006.

## Cheap rival it must beat

Keyword list and TF-IDF (fail by construction), sentence-embedding prototype, zero-shot LLM reader from a third model family, mean-pooled residual stream probe, sentence length. Most dangerous: the LLM reader, likely near ceiling. If it wins we still learn what share of concepts have a word-independent unit, which the study measured for one concept only.

## Premise it tests

P1. A pass shows concept units exist beyond one hand-picked case. A fail on most concepts, after Qwen's revenge unit scored 4 of 12 (study/results/wording.json), would mean neurons in 8B models mostly read words.

## Biology analogue

Knock-in with a matched control: change one thing, read one output

## Cost

S to M. About 4,800 short sentences, three extraction passes of minutes each, a few dollars of writing and reading calls; 3 to 5 days, mostly checking items.

## Payoff

A count of validated concept units per model, with the drop from development to test, and a frozen unit list that later designs and the steering area can reuse.

## Main risk

LLM-written pairs carry a style cue, so units or embeddings may win on style. Sentences are read alone, while the study read whole documents, so units that need long context may be missed.

## Reviewer note

Raised development pairs from 20 to 40 with a two-stage pick: with 20, about one of 884,736 units wins every pair by chance. Added a cross-concept null, a primary model and a pooled AUC. It is the cheap sentence-level cousin of the unrun planted-story plan. Style cues remain.

## Cross-area ranking

In the top twelve of 3 of 3 judges (what decides the premises fastest: rank 1; speed and cost: rank 3; upside: rank 2). The cheapest direct test of the claim everything else rests on: for 30 concepts, does a frozen neuron follow the idea when its usual words are gone and stay quiet when only the words are there; a fail is as usable as a pass, and its items and unit list feed at least six later designs.

## Amendment, 10 November 2026

Read each test sentence twice: alone, and after a fixed neutral lead-in of about 250 tokens. Goodfire's companion paper found that about 15% of example firings stop when a short passage is rerun without its document. Count a pair as won only if the concept sentence also reaches the unit's own strong range on general text (its 97th percentile or above). Also report wins at the 99.9th percentile.

## Sources

- Borsboom et al. 2004
- Hewitt and Liang 2019
- study/results/wording.json (revenge unit)
- preregistration/planted_story_protocol.md
- Tigges et al. 2026, Neurons as an Interpretable Basis
