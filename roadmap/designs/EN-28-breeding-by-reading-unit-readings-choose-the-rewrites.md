# EN-28: Breeding by reading: unit readings choose the rewrites

**Area:** Meme engineering · **Status:** design registered 10 November 2026, not run · **Priority:** 3.0 of 5

## Question

With writer and instruction held fixed, does choosing among candidate rewrites by a frozen reader's unit (neuron) readings move a story toward meme A and away from meme B, as blind human raters judge it, more than choosing by an LLM critic, embeddings, word counts or random units, or than one plain rewrite?

## Prediction

No. With A's usual words banned, the unit loop's human-rated combined score (0 to 1: A up, B down, story kept, fluent) is under 0.05 above the critic loop's, one plain rewrite's and even the random-unit loop's. Its own reading still rises, but rewording by a third model, Phi-4, removes over half of its lead over plain rewrites reworded likewise.

## What would refute it

On 160 held-out stories, each claim falls if the unit loop beats that arm (critic, plain rewrite, random units) by 0.05 or more, 95% interval (16 pairs resampled) above zero. The trigger claim falls if rewording keeps over half the lead. With spreads of 0.2 within and 0.05 between pairs, a true 0.10 passes 99% of the time, 0.05 half.

## Data

EN-22's 208 name-masked, ten-sentence film plots (B present, A absent, human-confirmed; 16 pairs from EN-23's 20 memes) from the CMU Movie Summary Corpus (42,306 Wikipedia summaries, CC BY-SA): 48 tuning, 160 held out. No arm may add A's keywords (pre-registration file) or an A unit's 20 strongest content words in 10 million FineWeb tokens (ODC-By). 25,000 rewrites, released CC BY-SA.

## Method

Reader: 20 Llama 3.1 8B neurons per meme, chosen on EN-23's development sentences and frozen; sentence score: their mean peak, in percentiles of FineWeb sentence peaks. Writer: Llama, told 'more A, less B'; four rounds of five candidates, shown the survivor's scores; highest A minus B survives. Settings fixed on tuning stories. Humans rate four arms blind, Qwen3-8B all.

## Cheap rival it must beat

One plain rewrite. The same loop, its survivor chosen by an LLM critic (Llama rating each sentence in words), sentence embeddings, TF-IDF word counts, a probe on Llama's averaged internal state, or, as the null, random units matched on layer and firing rate; all fitted on the same development sentences, length capped. Most dangerous: the critic, the same-budget prompt arm.

## Premise it tests

P1. Beating the critic: frozen units gauge an idea in a text better than the same model can say; support. Beating only random units: signal, no more than plain reading. Reading up, raters unmoved, lead lost on rewording: triggers; unit scores should not grade optimisers. If Qwen's units for those memes rise too, the models share triggers: caution for P2.

## Biology analogue

Supernormal stimulus: a dummy egg that out-triggers the real one.

## Cost

M. Needs EN-23's sentences and EN-22's stories first; best run as extra arms of EN-22. About 25,000 rewrites by a local 8B writer and 3,000 Phi-4 rewrites and rewordings (four to five laptop days, untimed); about 17 hours of blind rating by two people (unit, critic, random-unit and plain arms side by side; 40 stories rated twice). No paid API.

## Payoff

For units, an LLM critic, embeddings and word counts, the two numbers a grader handed to an optimiser needs: the blind gain its loop buys, and how much of its own rise survives rewording. And whether an open reader can steer a writer without touching weights: a Phi-4 writer repeats the unit and critic loops on 48 held-out stories, Qwen-judged.

## Main risk

Four rounds of rewriting may wear the story down, so every loop falls below one plain rewrite and only loop-against-loop gaps inform. The 8B writer may ignore the scores, leaving selection alone. Gate: stop unless the unit loop's own reading ends above the random-unit loop's in 36 of 48 tuning stories. Tuning-story differences spreading above 0.3: report as underpowered.

## Reviewer note

Remaining weakness: units will probably lose to the critic, so the news is, for each scorer, how far a loop can push its score up without blind raters seeing a gain; 0.05 and the 0.2 spread are conventions. Changed: critic loop is the bar, humans primary, power restated for 16 pairs, tuning gate, rewording baseline, closed-model arm dropped, 'feedback only' claim removed.

## Cross-area ranking

Registered on 10 November, after the three cross-area judges had ranked the earlier designs, so it was not ranked by them. Its reviewer's answer to whether it should run first: no: because it cannot start until EN-23's sentences and EN-22's stories exist, neither a run-first step; units will probably lose to the critic, so it settles neither premise fast. Run it as extra arms if EN-22 runs.

## Overlaps with

EN-22: stories, meme pairs, 20-neuron sets, judge, combined score, plain-rewrite arm. EN-23: contrast sentences, Qwen units. IN-23: own-range thresholds. US-21: its scored-rewrite attack, here given a goal and blind raters. EN-24, EN-25: selection loops with unit scorers. EN-21: ban-list rule. IN-31, the word-ban synthesis design prompted by the same paper, runs this search on single units.

## Note

Prompted by Goodfire's companion paper. Data check on 10 November 2026: Checked 10 November 2026. CMU corpus file opened: 42,306 summaries, about 20,800 with ten or more sentences; README: Creative Commons Attribution-ShareAlike, version unstated. Hugging Face API: Llama-3.1-8B-Instruct (llama3.1 licence, gated), Qwen3-8B (Apache-2.0), Phi-4 (MIT), MLX 4-bit builds of all three, FineWeb (ODC-By). arXiv API: 2501.17148, 2310.13032. Repository: 34 memes; keyword lists include common words. EN-22 and EN-23 are not run.

## Sources

- Tigges et al. 2026, Neurons as an Interpretable Basis (Goodfire)
- Wu et al. 2025, AxBench, arXiv:2501.17148
- Bradley et al. 2023, QDAIF, arXiv:2310.13032
- Bamman, O'Connor, Smith 2013, CMU Movie Summary Corpus
