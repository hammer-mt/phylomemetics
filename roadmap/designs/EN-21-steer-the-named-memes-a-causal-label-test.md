# EN-21: Steer the named memes: a causal label test

**Area:** Meme engineering · **Status:** design registered 8 November 2026, not run · **Priority:** 5.0 of 5

## Question

When a labelled meme unit is pushed up while the model writes, do blind judges recover its label, even with the meme's usual words banned?

## Prediction

At one fluent dose per unit, at least 8 of the 31 named Qwen3-8B units pass: judges choose the unit's own label, from 31 plus 'none', on at least half of 20 held-out openings. With the label's words banned from the output, at most 4 still pass, mostly story patterns such as the funeral and the secret identity.

## What would refute it

Fewer than 4 of 31 units pass, or no more pass than at the 99th percentile when labels are shuffled among the 31 units' texts: then the labels are not causal. More than 4 passes under the ban refutes the word-emitter half. Shuffled labels should give about one pass or fewer, so 31 units are enough.

## Data

No new corpus. The 31 unit labels fixed in October 2026 (`study/results/labels.json`, MIT). Ban lists fixed before generation: keywords from `preregistration/memes_for_tracing.json`, each unit's 20 most-promoted output words, and character names. 30 neutral story openings written for this test (10 to set doses, 20 held out). Qwen3-8B at 4-bit and 8-bit; Llama 3.1 8B twins as a replication.

## Method

Steering adds a fixed amount to one unit's activation as the model continues an opening. Each unit keeps the largest of five doses that leaves text at most three times as surprising as normal to the unsteered model. Two LLM judges, checked on 200 blind human ratings, choose a label. Secondary: switch the unit off and ask for the meme.

## Cheap rival it must beat

The unit's 20 most-promoted output words handed to the same judges with no generated text (most dangerous: if it ties, units emit names, confirming labels as words, not ideas). A prompt that asks for the label (ceiling). A direction built from keyword-matched sentences. 62 random units matched on layer and firing rate, showing how often any push draws a label.

## Premise it tests

P1. Passes under the keyword ban are the first causal evidence that a unit carries an idea. Passes only without the ban mean units are word emitters. A general fail means the paper's labels describe what units read, not what they do.

## Biology analogue

Gene overexpression: turn one gene up, look for the trait.

## Cost

S. One to two days on the laptop, a few dollars of judge calls, two hours of human rating.

## Payoff

The causal check the paper named as missing, for every showcase meme, and a shortlist of units safe to use as edit handles.

## Main risk

In small 4-bit models the gap between no effect and gibberish is narrow, so many units may fail for dose reasons and the result reads as untested, not refuted.

## Reviewer note

Replaced the invented 5% null with a shuffled-label null, fixed one dose per unit on tuning openings, and fixed ban lists in advance. Remaining weakness: the file tags only 13 of the 31 labels as tropes; 8 are characters, which can hardly pass the ban.

## Sources

- Kran Research 2026, Evil Qwen replication (blog.kran.ai/evil-qwen)
- Cammarata 2026, Neural networks are already clean
- Liu 2026, arXiv:2606.19831
- Arad, Mueller, Belinkov 2025, arXiv:2505.20063
