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

---

## Results (run 26 November 2026; everything above was committed first, except where marked)

Numbers are in [`../results/scale/`](../results/scale/): `summary.json`, `*_features.json`, `baseline_*.json`, `steer_summary.json`.

**Reproduction.** The published 4-bit MLX pipeline, unchanged, on an NVIDIA GPU gives the same survivors by name
(11 Qwen, 9 Llama), the same nulls (5.04 and 5.66) and the same revenge scores (4 and 10 of 12).
`repro_mlx4bit_cuda_f3_wording.json`.

**Decision rule: Qwen3-32B is not clearly better. SAE latents should be a rival.**

| Part B: fresh pairs, chance 0.5 | All | Hard half |
| --- | --- | --- |
| LLM judge, 32B / 8B | 1.000 / 0.984 | 1.000 / 0.974 |
| Residual probe, 8B / 32B | 0.993 / 0.989 | 0.985 / 0.985 |
| Qwen3-Embedding-4B, prototype / probe | 0.982 / 0.967 | 0.971 / 0.949 |
| SAE m5, 32B / 8B | 0.972 / 0.943 | 0.960 / 0.930 |
| Neurons m5, 32B / 8B | 0.960 / 0.932 | 0.956 / 0.908 |
| Neuron m1 (one neuron), 32B / 8B | 0.921 / 0.895 | 0.930 / 0.879 |
| SAE m1 (one latent), 32B / 8B | 0.901 / 0.847 | 0.897 / 0.831 |
| TF-IDF logistic regression | 0.871 | 0.743 (pairs chosen on its scores) |

- **Scale.** 32B minus 8B, m1: +0.026 (95% CI -0.035 to 0.086, p = 0.22). m5: +0.028 (-0.011 to 0.066, p = 0.12). With the
  search cut to the 8B's size, 32B's m1 falls to 0.876, so there is no sign that the larger model's neurons are cleaner.
- **Against words.** One neuron does not clearly beat TF-IDF in 8B (+0.024, p = 0.20) and only just does in 32B
  (+0.050, p = 0.048, CI -0.006 to 0.103). Five do (+0.061, p = 0.003; +0.088, p = 0.0002).
- **Against dense vectors (post hoc).** The 8B's own hidden state beats its five-neuron detector by 0.061 (p = 0.004),
  and the embedding prototype beats it by 0.050 (p = 0.003). In 32B the embedding gap is 0.022 (p = 0.15).
- **Not a style artefact.** Neuron and SAE detectors score 0.50 on other memes' pairs.

**Part A.** Under the published rule 8B unquantised keeps 10 of 31 (null 4.8); 32B twins keep 6 (null 3.6, p = 0.07).
Found while running: the published rule compares percentiles in int8 activations, whose rounding makes ties. Comparing raw
activations gives 12 (null 7.6, p = 0.03) and 14 (null 9.4, p = 0.02). SAE twins keep 6 in each model, but against a null
of only 1.4 (p at most 0.0004): fewer false alarms than neurons. The 8B revenge unit, at 4 of 12 in 4-bit, wins 9 of 12
unquantised; 32B's wins 10. For SAEs the revenge test is undefined: no candidate latent fired on any pair.

**Part C.** Turning units up (effect on P(concept) over random controls, mean of 5 concepts):

| | 8B | 32B |
| --- | --- | --- |
| Published unit or its twin | 0.01 | 0.08 |
| Part B neuron | 0.31 | 0.19 |
| Part B SAE latent | 0.48 | 0.15 |

The published funeral unit sits in layer 1 and makes the model repeat "Funeral Service": a word detector. Units chosen from
examples steer better; secret identity is the clearest (0.89 in 8B, 0.51 in 32B). Setting a unit to 0 does almost
nothing (-0.04 to 0.02). Setting it to -5 times its maximum lowers the concept but often breaks the text, so it is
not evidence about labels. The judge is lenient (it accepts "few knew the identities" as a secret identity) and is the
steered model's own family, each target has one random control, and there are 10 scored texts per cell.

**Limits of all of this.** Single sentences, not stories; 34 memes; AI-written pairs near ceiling for the best methods; one
SAE family. Pair accuracy is not the pre-registration's twin AUC. These memes and ban lists are also the planned
experiment's, so anything changed in the plan because of these results must be logged in its section 14.
