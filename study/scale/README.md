# Does scale help? Qwen3-32B, steering and SAE features on DEV data

**Exploratory. Not pre-registered, not part of paper v1.** This decides whether the planned experiment in
[`../../preregistration/`](../../preregistration/) should be changed before freeze F0. It makes no TEST data and
writes no planted story. The plan below was committed before any model was run.

## Questions

1. **Precision.** Do the published Qwen3-8B (4-bit, MLX) rewording results hold in the unquantised model?
2. **Scale.** Does Qwen3-32B (the model audited in "Neural networks are already clean") do better?
3. **Basis.** Do sparse-autoencoder (SAE) latents do better than raw neurons in the same model?
4. **Labels.** Does turning a labelled unit up put its concept in the text, and turning it down take it out?

## Part A: the published rewording test, unchanged

31 named memes, one rewording each, the published bar (above all 10 everyday sentences, in corpus percentile) and
the published mismatched-rewording null (10,000 derangements). 8B uses the published units. 32B and the SAEs use
twins, chosen exactly as Llama's were: the unit whose corpus sentence profile correlates best with the 8B unit's.
The revenge test is rerun with each model's own unit (50 candidates, chosen on DEV, tested on 12 pairs).

Known limits: one sentence per meme, and twins inherit the 8B unit's word and name habits. Part B is the main test.

## Part B (main): fresh minimal pairs

For the 34 memes in `memes_for_tracing.json`, two separate blind AI writers made:
- `data/select.json`: 5 explicit and 5 reworded sentences per meme, used only to choose each method's detector.
- `data/test.json`: 8 pairs per meme. "present" shows the meme without its ban-list words; "absent" is the same
  sentence with an equally eventful different event. Neither writer saw the other's file, the labels of any unit, or
  the published test sentences.

Every method gets the same positives (the meme's 10 select sentences) and background (the 69 reference films'
sentences and the other memes' select sentences, each run alone). Score: the share of a meme's 8 pairs in which
"present" outscores "absent" (ties half), averaged over memes; chance 0.5.

Methods: `neuron:m1` (best unit by mean percentile over the two backgrounds, ties by standardised margin) and
`neuron:m5` (mean of the top 5, background-standardised), in 8B and 32B; the same for SAE latents (BatchTopK, 65k,
L0 80, layers at 25/50/75%, pooled); residual prototype and logistic probe (the 50% layer); Qwen3-Embedding-4B
prototype and probe; TF-IDF logistic regression; LLM judge (8B, 32B) as an upper bound. `neuron:m1` in 32B is also
run on random subsets of the 8B's size, to separate a better model from a bigger search. Each detector is also scored
on the other memes' pairs (`*_mismatch`): about 0.5 unless it merely prefers "present" sentences.

**Hard half.** TF-IDF was run first (no model needed) and already wins 87% of pairs, so words remain informative even
without the ban list. Added before any model was run: every method is also scored on each meme's 4 pairs with the
smallest TF-IDF margin. This is selected on TF-IDF's own scores, so TF-IDF's number there is not a fair rival.

Known limits: "absent" sentences are edits, so some read slightly less naturally. Both writers are the same model, and a
few test plots resemble select sentences (highest TF-IDF cosine 0.44), which helps the word and embedding rivals most.

## Decision rule (fixed before results)

- **"32B is clearly better"**: `neuron:m1` pair accuracy in 32B exceeds unquantised 8B by at least 0.05 on average
  over the 34 memes, with one-sided meme-level sign-flip p < 0.05 (10,000 flips). Same rule for m5, reported beside it.
- **"SAE latents should be a rival"**: SAE m1 or m5 is within 0.05 of, or above, the same model's neurons.
- The hard half is a secondary, reported beside the main result whatever it shows.
- Part A and steering inform the reading; they do not decide.
- Anything else is "not clearly better", and the plan's model stays.

## Part C: steering

Five concepts: funeral, coin flip, secret identity, joke (the four that survived in both published models) and
revenge. Units: each model's Part A unit or twin, its Part B `m1` neuron and its Part B SAE `m1` latent. Control for
each: a random unit of the same kind and layer with a similar corpus maximum.

- **Up:** 10 neutral story prompts, 2 samples each, unit set to 2, 5, 10 or 20 times its corpus maximum.
- **Down:** continue the meme's 5 reworded select sentences, 2 samples each, unit set to 0 or to -5 times its maximum.
- Qwen3-32B, unsteered and blind to condition, gives P(yes) for each of the five concepts and for coherence.
- Headline: target minus random-control P(own concept). Strength is the one with the largest own-concept rise at
  coherence at least 0.7, chosen on prompts 1 to 5 and scored on prompts 6 to 10. Specificity: own minus the mean of
  the other four concepts.

## Compute

One rented GPU pod. No private data or credentials go on it: only this repository's public files and public models.
