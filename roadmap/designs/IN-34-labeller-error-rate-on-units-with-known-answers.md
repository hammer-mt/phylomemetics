# IN-34: Labeller error rate on units with known answers

**Area:** The instrument · **Status:** design registered 10 November 2026, not run · **Priority:** 3.2 of 5

## Question

When the study's labelling step, rebuilt, and the same step with one added question are run on made-up units whose firing rule is known, how often does each call a keyword detector an idea, or claim more than its eight sentences can show?

## Prediction

Asked one added question (one word, an idea, cannot tell, two things, nothing), the reader calls an idea at least 80% of 60 idea twins and at most 20% of 60 keyword twins, calls at most 20% of 60 no-tell units ideas outright, and gets 75% of 90 other units right. The study's own fields, rebuilt, fail the twin ratio.

## What would refute it

Added-question reader: idea calls under twice as common for idea twins as keyword twins, or on under half the idea twins; outright idea calls on 40% or more of no-tell units; or under 60% of the other 90 right. Simulated with whole concepts agreeing, a reader meeting the hypothesis fails each under 3% of runs; a 60%-against-40% reader, over 80%.

## Data

CMU Movie Summary Corpus (Bamman et al. 2013): 42,306 Wikipedia film plots, about 666,000 sentences, CC BY-SA. Per concept (30), two people confirm 28 sentences: 12 telling the concept with its head word (commonest word), 4 with that word but not the concept ('ghost town'), 12 telling it without its usual words. Candidates: string search and a third model family.

## Method

Unit: rule plus eight sentences. Per concept: two keyword twins (six head-word sentences, two off-concept); two idea twins (same six, two word-free); two no-tell (eight head-word, on-concept: 'cannot tell'); one word-free; one blend; one empty. Prompts frozen on five spare concepts before test sentences exist. Script scores calls (study arm: kind 'trope', the paper's story pattern); a blind person matches names; concept bootstrap.

## Cheap rival it must beat

A count rule: one content word in all eight sentences means 'word'. It separates the twins by construction, so the reader adds only where counting is blind: no-tell units, naming, blends, empties. There it faces the nearest of 35 definitions by sentence embedding, a TF-IDF word list and Qwen3-8B. Most dangerous: the count rule; if it wins, compute the call.

## Premise it tests

P1, indirectly: it tests reading, not neurons. The study's claim of story-pattern units rests mostly on labels: 13 of 31 named labels, 22 of 120 in the second reading. A pass says how much weight such a label carries. A fail makes it no evidence against word detection, leaving the rewording test, where 2 of 31 Qwen units were specific.

## Biology analogue

Spike-in standards and decoy sequences for an annotation pipeline

## Cost

M. No model internals. About 1,000 labelling calls (a quarter on a local 8B model), 6,000 plot-reading and 550 rival-judging calls: a few dollars. About 16 hours of people's time: two confirm some 1,500 candidate sentences, one matches 270 labels, a second re-matches 90, one labels 54 units unaided to show the answer is recoverable. About a week.

## Payoff

A bound for each published story-pattern label by its evidence: one word in all eight sentences, or not. Error by the reader's stated confidence. A tested prompt, a free count check to run before any reader, and a reusable answer-key set for new labellers, for repairs such as showing look-alike sentences the unit ignores, and for IN-33's control labels.

## Main risk

The study's prompt was never committed. The rebuilt one must reproduce the committed kind on 60% of 60 held-back study records, and is still a reconstruction. Sentences are clear cases, concepts need a head word with a second use, and rules are all-or-nothing: measured error is a floor. Nothing here shows that eight top sentences represent a neuron.

## Reviewer note

Remaining: the study's prompt is not in the repository, so its arm is a reconstruction; made-up units are clear cases, so error is a floor; counting beats any reader on twins by construction. Changed: no-tell units replace plain keyword units, two prompts, count rule as rival, per-type falsifier with simulated power, human baseline.

## Cross-area ranking

Registered on 10 November, after the three cross-area judges had ranked the earlier designs, so it was not ranked by them. Its reviewer's answer to whether it should run first: no: because it calibrates the label-writing step, which the audit already discounts, and decides neither premise. Run it when labels are next written at scale (IN-33, AT-21). Its free count check on the 120 committed records can be done now.

## Overlaps with

IN-21: concept and usual-word lists. IN-33: scores real labels by prediction and labels its control units; share this frozen prompt and check its score here. IN-22: plants blends for a coherence grader, not a labeller. IN-02, IN-S8: successor with an answer key. IN-S7: relabelling on general text. AT-21: entries can carry the evidence class.

## Note

Prompted by the field survey (arXiv:2611.08438). Data check on 10 November 2026: CMU archive unpacked: 42,306 plots; README: 'Creative Commons Attribution-ShareAlike', no version; 666,239 sentences by a simple splitter; 50 head words counted (fewest 79). Hugging Face: Qwen/Qwen3-8B, Qwen/Qwen3-Embedding-0.6B, Apache-2.0. Repo: 120 records of eight sentences, no prompt; 4 of 22 second-reading story patterns share one word in all eight; the readings agree on kind for 19 of 26.

## Sources

- Barez, Wu et al. 2026, arXiv:2611.08438 (guardrail layers 3 and 4: units built to fool a pipeline, known-answer units, error rates, calibrated confidence)
- Bills et al. 2023, Language models can explain neurons in language models (19 constructed neuron puzzles; the baseline explainer solved 5)
- Schwettmann et al. 2023, FIND: A Function Description Benchmark for Evaluating Interpretability Methods (arXiv:2309.03886)
- study/results/labels.json, labels_wf.json and label_candidates.json
