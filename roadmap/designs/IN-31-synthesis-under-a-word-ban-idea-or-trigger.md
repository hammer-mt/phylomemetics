# IN-31: Synthesis under a word ban: idea or trigger?

**Area:** The instrument · **Status:** design registered 10 November 2026, not run · **Priority:** 3.6 of 5

## Question

When a writing model that is told how strongly a unit (one sign of one neuron) fired on each draft is barred from the unit's usual words, can it learn to write text that blind judges accept as the unit's idea and that fires the unit strongly, and can a second writer then fire the unit without the idea?

## Prediction

At least 18 of 60 concept units pass, against at most 3 of 60 random units matched on layer and firing rate, given the same concepts. A pass: 10 or more of 20 fresh texts are judged on-idea, reach the unit's strong range and clear the token table; under 5 of 20 attacker texts fire it while judged off-idea.

## What would refute it

Fewer than 9 of 60 pass, or the pass share is under 15 points above the random units'. 9 to 17: undecided. Void if the unbanned loop reaches under 9 of 10 study name units, or more than 1 of 10 planted detectors passes. A true 10% is refuted at least 82 times in 100; a true 40% confirmed 90.

## Data

Written by the experiment in English, CC BY 4.0: about 165,000 texts of 50 to 60 tokens. Units: IN-21's 30 concept units each in Qwen3-8B and Llama 3.1 8B. FineWeb sample-10BT (ODC-By): 10 million tokens for word lists and token tables; 20,000 60-token passages fix each strong range: the peak only 1% reach. All read after one 250-token neutral lead-in.

## Method

Frozen: units, ranges, ban lists (keywords, each unit's 50 strongest words), 20 scenes for the fresh texts that the feedback rounds never use, prompts tuned on study units. 14 feedback rounds of 20 drafts, then one fresh text per unseen scene. An attacker edits the idea out, same budget and ban. Units also run unbanned. Two blind LLM judges from other families pick a concept or 'none'. Bootstrap over concepts.

## Cheap rival it must beat

Same game: the model asked 'is this about X?' (likeliest to win); the model's probability that a banned keyword comes next (a tie means units predict words); a direction (mean internal state, concept minus control); an embedding prototype. A token table (each token's average FineWeb firing) voids texts it ranks top 1%. Ten made-up detectors with a known trigger (a hidden word list or a sentence frame) go through the same loop with that screen switched off, and must not pass.

## Premise it tests

P1. A pass share above the random units' adds what IN-21's fixed pairs cannot: the unit follows the idea in wording it was never chosen on, and an optimiser cannot fire it without the idea. Units reached only with their words, or fired by a wording trick, are triggers. One fired by a neighbouring idea is wider than its label.

## Biology analogue

Synonymous recoding: rewrite a gene in other codons and see whether the protein still works.

## Cost

M. About 550 search runs of 300 short texts through two 8B models (one to two days of laptop time); about 10 million tokens written and 35 million read by the writer, 22,000 judge calls (a few hundred dollars at mid-range prices); three hours of human rating. About two weeks, after IN-21 freezes its units and IN-23's FineWeb pass exists.

## Payoff

Every concept unit sorted beside its IN-21 result: idea unit; trigger (reached only with its words, or fired by a wording trick); wider than its label (fired by another idea); or unreachable. Also the controls the paper's synthesis test lacks, on the study's models: the same search on a random unit, a planted word list and with the idea removed.

## Main risk

The count may land between 9 and 17 and decide nothing. A pass needs an LLM attacker to fail, and lenient judges create passes twice, so a human rates 150 accepted writer texts and 150 firing attacker texts judged on-idea; over 15% overturned in either makes the result untested. Many IN-21 units may never reach their strong range.

## Reviewer note

Still rests on an LLM attacker failing, and counts from 9 to 17 decide nothing. Added a token-table screen, unseen settings, planted detectors with known triggers, two missing rivals (the model's own word prediction; the model asked directly), a targeted human audit and fail classes; corrected the power claim.

## Cross-area ranking

Registered on 10 November, after the three cross-area judges had ranked the earlier designs, so it was not ranked by them. Its reviewer's answer to whether it should run first: no: because it cannot start until IN-21 in step 1 has frozen its units, and it is worth two weeks only if some pass there. Run it straight after step 1, as the adversarial check on IN-21's passers.

## Overlaps with

IN-21: concepts, keyword lists, frozen units; this is its adversarial sequel. IN-23: FineWeb pass and own-range thresholds. IN-24: the direction. EN-21: the label-only test in its amendment is round one here, on other units. US-21: its scored-rewrite attack. EN-28: the same feedback loop, run on 20-neuron sets with a goal. IN-30: cheap readers. IN-03: successor.

## Note

Prompted by Goodfire's companion paper. Data check on 10 November 2026: Hugging Face API, 10 Nov 2026: HuggingFaceFW/fineweb (odc-by, config sample-10BT), mlx-community/Qwen3-8B-4bit (apache-2.0), mlx-community/Meta-Llama-3.1-8B-Instruct-4bit (llama3.1), Qwen/Qwen3-Embedding-0.6B (apache-2.0). arXiv API: 2406.17557, 2501.17148. Repo: wording.json and labels.json exist; IN-21's units do not yet. The paper's Figure 13 axis shows 14 rounds; its 120 of 128 used no ban. Goodfire's atlas and tools are not used.

## Sources

- Tigges et al. 2026, Neurons as an Interpretable Basis (Goodfire)
- study/results/wording.json (rewording test, IN-03)
- Wu et al. 2025, AxBench, arXiv:2501.17148
- Penedo et al. 2024, FineWeb, arXiv:2406.17557
