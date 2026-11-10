# IN-33: Do labels predict firing better than a token table?

**Area:** The instrument · **Status:** design registered 10 November 2026, not run · **Priority:** 3.7 of 5

## Question

Given only the label of a unit (one side of one neuron), can a reader model say which sentences of unseen plot text it fires on better than a table of each token's (word piece's) average activation, the unit's top tokens used as its label, or another unit's label?

## Prediction

At least 20% of scorable units among the study's 125 labelled Qwen3-8B units pass step 1: the label finds their firing sentences with AUC (chance 0.5) at least 0.05 above both token tables and same-kind wrong labels. At least 8 pass step 2, firing on at least 20 of 40 word-free rewrites and under 8 of 40 rewritten controls.

## What would refute it

Fewer than 10% of scorable labels pass step 1, or fewer than 4 units pass step 2. A gap counts only if its 95% interval excludes zero: a label no better than the table then passes about 3% of the time, one truly 0.10 better 69% to 98% (simulated), and a true 20% is missed about once in 60 runs.

## Data

Test, drawn by seed after every fitted part is hashed: 600 film plots from the CMU Movie Summary Corpus (42,306 plots, CC BY-SA), outside the study and pre-registration lists, and 490 Batman television episode summaries from 13 Wikipedia pages (CC BY-SA 4.0); about 22,000 sentences. Development: the study corpus, 300 other CMU plots, 10 million FineWeb tokens (ODC-By).

## Method

The lead unit of each labelled family (the study merged units that fire together into families; 42 of the 125 are a single neuron), frozen at paper v1. Firing: a token beyond its general-text 99.9th percentile; scorable units have 40 firing sentences. A non-Claude reader rates up to 150 firing sentences and three length-matched others each, half from the same documents. Rewrites drop the label's words and the unit's 20 top tokens.

## Cheap rival it must beat

Most dangerous: the token table, which beat written explanations on average in Bills et al. 2023, and a token-pair table; 57 to 59 of the 125 labels name a character or place, depending on which label file is used for the 26 units in both. Also: the unit's 20 top tokens as its label, no label, word overlap, sentence-embedding regression, length. Controls, labelled alike: 125 layer-matched random poles, 125 sums of two.

## Premise it tests

P1. A pass: study units carry nameable content beyond token identity, in the study's own model. A fail shows these labels add nothing to a word table; it counts against the units only if fresh labels and step 2 fail too. If two-pole sums pass as often as single poles, a pass says nothing for the neuron as the unit.

## Biology analogue

Gene Ontology evidence codes: an annotation graded by the test it has passed

## Cost

M. One general-text pass on Qwen3-8B, planned jointly with IN-23; minutes of plot extraction; about 45,000 reader calls of 25 sentences (a few hundred US dollars, estimated); about 15,000 short rewrites by a third model family; a person rating 300 sentences; one to two weeks.

## Payoff

A class for every label: beats a word table or not, fires without its words or not. The share of firings the reader calls on-label, by strength band. The word-level comparison Goodfire's tests lack, for this project's model and labels only. Whether labels rewritten from general-text examples do better. A label-scoring harness that IN-23, AT-21 and EN-21 can reuse.

## Main risk

Most labels may tie the token table, repeating the study's result: in natural text an idea and its words arrive together, so only step 2 separates them. Rewrites may lose the idea; a words-kept rewrite sets each unit's ceiling. A weak reader could fail good labels; an examples-only reader on 30 units and a person's ratings bound that.

## Reviewer note

Remaining: labels, reader and rewrites are all LLM work, checked by one person on 300 sentences, and the units were picked for Batman presence, so a fail says little about neurons in general. Changed: the score now asks only where a unit fires, not how strongly; a firing means the unit's own strong range; each unit passes or fails on its own; the unit's top tokens were added as a rival label; rewrites follow a fixed rule; sentences are sampled for scoring.

## Cross-area ranking

Registered on 10 November, after the three cross-area judges had ranked the earlier designs, so it was not ranked by them. Its reviewer's answer to whether it should run first: no: because these units were picked for Batman presence and nearly half are labelled as characters or places, so the likely fail is discounted; IN-21 asks the same P1 question on frozen concept units. Run it on IN-23's general-text pass.

## Overlaps with

Successor to IN-02 and IN-03: same labels, unseen text, a fitted word rival, answers kept. Nearest registered: IN-23's label test by band; run both on one pass and use this scoring there. IN-S7 is its fresh-label arm. Step 2 resembles IN-21, which starts from concepts, not labels. Two-pole sums follow IN-22.

## Note

Prompted by Goodfire's companion paper. Data check on 10 November 2026: Checked 10 November 2026. CMU archive unpacked: 42,306 plots; README says Creative Commons Attribution-ShareAlike, no version; 10,539 plots of 20 to 60 sentences are outside both lists. Hugging Face: FineWeb, odc-by, 15 files under sample/10BT. Thirteen Wikipedia pages (episode tables for the 1966 Batman series, Batman: The Animated Series, The New Batman Adventures, Batman Beyond, The Batman, Batman: The Brave and the Bold, Beware the Batman, Batman: Caped Crusader and Gotham seasons 1 to 5): 490 summaries, about 2,170 sentences, CC BY-SA 4.0. Label files: 125 units, 115 distinct labels. Bills et al.: token-table passage read.

## Sources

- Tigges et al. 2026, Neurons as an Interpretable Basis (Goodfire)
- Bills et al. 2023, Language models can explain neurons in language models
- Bamman, O'Connor and Smith 2013 (CMU Movie Summary Corpus)
- study/results/labels.json, labels_wf.json, wording.json, robustness/label_test.json
