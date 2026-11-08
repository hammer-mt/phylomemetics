# UN-21: Held-out twin census with word-matched nulls

**Area:** Across models · **Status:** design registered 8 November 2026, not run · **Priority:** 4.8 of 5

## Question

What share of Qwen3-8B units have a Llama-3.1-8B twin on unseen text once words, sentence length and position in the plot are accounted for?

## Prediction

At least 2% of sampled Qwen units have a twin (the best-correlated Llama unit, chosen on other films) whose held-out correlation beats a rotation baseline by 0.5. At least half of those keep a partial correlation of 0.3 after the word, length and position predictors are removed, mostly at 40 to 60% depth.

## What would refute it

Fewer than 0.5% of units pass, or fewer than one passer in ten keeps partial correlation 0.3, on original and name-masked text alike. Either refutes neuron-level P2 beyond shared vocabulary. Between 0.5% and 2% is inconclusive. A layer-stratified sample of 20,000 units gives a standard error near 0.1 percentage points.

## Data

The 1,000 random film plots already committed (`preregistration/sources/t3_random_films.jsonl.gz`; 22,225 sentences after the 40-sentence cap; Wikipedia text, CC BY-SA 4.0). A name-masked copy is not committed: make it with an open entity tagger and hand-check 200 sentences for leaks. Split by film: 500 to choose twins, 500 to score. No Batman text.

## Method

Unit: each sign of each MLP channel (a hidden neuron), strongest value per sentence. For each sampled Qwen unit, pick the best-correlated Llama unit on one half, keeping mutual picks; score on the other half. Statistics: correlation above a random rotation of Llama's layer (Gurnee's baseline); partial correlation given word, length and position predictors. Null: sentences misaligned between models.

## Cheap rival it must beat

Each Qwen unit is also predicted from its best single word, a 10-word lasso (a regression that keeps few words), a ridge fit on all words and word pairs, a sentence-embedding ridge, sentence length and position, and Llama's mean-pooled residual stream; an LLM reader only for labelled units. The all-word ridge is most dangerous: if it wins, twins are vocabulary.

## Premise it tests

P2. A pass gives a counted core of shared units to seed a common catalogue. A fail means 'the same neurons' gives way to 'the same directions', and twin strength stops being a mark of a real meme, because it selects word detectors.

## Biology analogue

Orthologue census by reciprocal best hits.

## Cost

S. Four extraction passes of under ten minutes each (two models, two text versions), a few hours of correlation on the laptop, no API spend. Stored whole, the activations take about 80 GB, so process them layer by layer.

## Payoff

The first cross-family neuron census on 8B models, with held-out numbers to replace the in-sample 0.82 and 0.40 (`nulls.json`, `phylo.json.gz`), and a list of beyond-word shared units that every later cross-model design reuses. The study's 125 labelled units and 27 concept units are scored as fixed subgroups.

## Main risk

The pass share moves with the 0.5 threshold, which Gurnee et al. say has no principled value, so the whole curve is reported. A word control fitted on 11,000 sentences can miss rare synonyms and inflate the beyond-word share; known single-word detectors (best-word correlation 0.9 or more) should fail the test, and their pass rate measures that leak.

## Reviewer note

Added sentence length and plot position to the controls (shared position detectors would otherwise pass as beyond-word twins), an all-word control, and a leak check on known word detectors. The name-masked copy does not exist yet. Remaining weakness: rare synonyms can slip past any word control.

## Cross-area ranking

In the top twelve of 2 of 3 judges (what decides the premises fastest: rank 2; speed and cost: rank 1). Settles neuron-level P2 on unseen text with data already committed and no API spend: either a counted list of shared units that survive word, length and position controls, or proof that twins are vocabulary and P2 has to be restated as mapped directions.

## Sources

- Gurnee et al. 2024
- Gröger, Wen and Brbić 2026
- study/results/robustness/nulls.json
- preregistration T5 (S20), which this supersedes
