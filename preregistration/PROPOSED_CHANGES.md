# Proposed changes before freeze F0: for Mike to accept or reject

**Status: proposal of 2026-11-27. Nothing here is adopted.** `PREREGISTRATION.md`, the code and the work lists are unchanged.
Each accepted item would be applied, tested and logged in section 14 in one commit before F0. No TEST data exists.

**Evidence.** The exploratory DEV run in [`../study/scale/`](../study/scale/) (`summary.json`, `prereg_inputs.json` in
[`../study/results/scale/`](../study/results/scale/)): 272 AI-written sentence pairs over these 34 memes. They use the plan's
memes, definitions and ban lists, so the choices below are informed by DEV results on the same memes; they are not
independent. The scores are pair accuracies on single sentences, not the plan's twin AUC on stories, so they are a guide to
the direction, not the size, of what T2 will show.

| | Proposal | Recommendation |
| --- | --- | --- |
| 1 | Add SAE latents as a rival | Accept |
| 2 | State what DEV predicts for H1, H2 and H6; keep the rules | Accept |
| 3 | Check for a ceiling in the pilot before writing 816 stories | Accept |
| 4 | Use unquantised models | Reject for now; run a check first |

## 1. Add SAE latents as a rival

**Why.** The run's rule ("within 0.05 of, or above, the same model's neurons") was met in both models. Qwen3-8B: five latents
0.943 against five neurons 0.932; one latent 0.847 against one neuron 0.895. An SAE is the standard alternative basis for
"a neuron is a concept", and the plan has none.

**Change.**
- Section 2.1b, new. `sae:qwen3-8b`: the BatchTopK SAEs of `adamkarvonen/qwen3-8b-saes` (revision
  `460a11d556a0c58ffd65909555f38b33e9d71aea`, trainer 2, 65k latents, L0 80, layers 9, 18 and 27 pooled), tokens with
  residual norm above 10 times the pass median dropped. Same selection, examples, background and `m1`/`m5` LOEO choice as the
  neurons; no DEV-mapped units.
- It enters H6 as a rival (it does not share the neurons' names, so `h6_rival` already admits it) and is reported on
  every version beside the neurons, with its overlap counts.
- It does **not** enter H1 or H2. Adding a rival to an intersection-union test after DEV results would change the claims.
- Section 5: the T2 family grows from 7 to 8 (`sae:qwen3-8b` against `neurons:qwen3-8b`, two-sided);
  `OVERLAP_METHODS` grows from 17 to 18 (descriptive). Section 11: about one more GPU-minute pass; the latents are about 6 GB.
- Qwen only: no Llama 3.1 8B Instruct SAE of the same family and layers was checked. The SAE was trained on the
  unquantised model, so its scores on the 4-bit model are a known mismatch, reported as such.

**Code:** `extract_compact.py`, `analysis.py`, `test_analysis.py` (family sizes), `build_lists.py`.

## 2. State what DEV predicts, and keep the rules

**Why.** A reader should know what we expected. All gaps are pair accuracy, 8B, with bootstrap 95% intervals over memes.

| Claim | DEV evidence | Expectation |
| --- | --- | --- |
| H1 | Five neurons beat `keywords:logreg`'s method (TF-IDF logistic) by 0.061 (0.022 to 0.103); one neuron by 0.024 (-0.029 to 0.072). The evasion-aware analyst was not run. | Plausible for `m5`; open for `m1` |
| H2 | Five neurons are behind the Qwen3-Embedding-4B prototype by 0.050 (0.018 to 0.085) and on the 10 neutral memes by 0.075, beyond the -0.05 consistency bound | Not supported in 8B |
| H6 | Qwen3-Embedding-4B is ahead and half the parameters of the 8B; hidden states are ahead by 0.061 (0.024 to 0.103) | A rival probably dominates |
| "Single neurons are memes" (`m5` minus `m1` upper bound below 0.03) | 0.037, upper bound 0.074 | Not claimed |

**Change.** Add section 1.7 with this table, headed "What DEV predicts". H1, H2, H3, H6, their SESOIs and their verdict rules stay
as written. Section 13 gains one sentence: if H2 and H6 fail, the case for neurons rests on interpretability (T6), and T6
does not test that neurons are more interpretable than SAE latents.

**Not proposed.** Moving H2 to Qwen3-32B. There, five neurons sit 0.022 behind the embedding prototype (one-sided 95% lower
bound -0.048, just inside the margin). But 32B over 8B was +0.028 (-0.011 to 0.066), short of the rule set beforehand, and
choosing the model by which one passes H2 would be selection on the outcome. Reported so that Mike can see it.

## 3. Check for a ceiling before writing 816 stories

**Why.** On DEV pairs the best methods are at the top: 94% of memes are won 8 of 8 by the embedding prototype and by the 8B hidden-state
probe, against 74% for five neurons. At a ceiling no test in section 4 can separate them, and S-coded and H2 have no
room. Plain TF-IDF won 87% of pairs with the ban list removed, so the ban list alone does not make words useless.

**Change.** After P2 and before F0, every method available on DEV scores the 6 pilot plots' banned, implicit and coded versions
against their controls (pilot stories are never TEST). 18 planted memes give 36 reworded and 18 coded pairs, so the result
is a warning light, not a test. If three or more of `emb:qwen3emb-4b`, `hidden:qwen3-8b`, `judge:qwen3-8b` and
`neurons:qwen3-8b` win at least 95% of the reworded pairs, Mike chooses, before F0, one of:
- **(a)** controls become near misses built from each meme's decoy hint (an element that shares the meme's setting and mood
  but not its event), and the pilot is repeated; or
- **(b)** keep the stories and make S-coded the headline, accepting its low power; or
- **(c)** run as designed, knowing the top methods may tie.

Recommendation: (a). It keeps the design and the hypotheses.

## 4. Unquantised models: run a check first

**Why not now.** The run's evidence for this is thin: one revenge unit (4 of 12 pairs in 4-bit, 9 of 12 unquantised) and 11 against 10
survivors under the published rule. The 4-bit model was not run on the 272 pairs. Two corrections to what was said when
this was first suggested: `extract_compact.py` already works in float32 and float16, not int8, so "compare raw activations"
needs no change; and H6's cost is measured on this laptop, where the unquantised model is about four times the size of the
4-bit one, so switching changes the cost claims as well as the accuracy ones.

**Check first (needs approval to spend).** Run `mlx-community/Qwen3-8B-4bit` on the 272 pairs (about one hour of a rented GPU,
about $1 to $3). Switch to unquantised only if `neuron:m5` pair accuracy is at least 0.03 higher unquantised and its
sign-flip p is below 0.05. Otherwise keep 4-bit.

## Section 14.6, ready to paste if any item is accepted

> **14.6 DEV scale run and the changes it prompted (2026-11-26 and 27, before F0).** An exploratory run (`study/scale/`,
> plan and decision rule committed before any model was run; the TF-IDF baseline and the hard half came first, and
> the int8-tie reporting was added during the run) scored Qwen3-8B, Qwen3-32B, SAE latents, embeddings, hidden states and
> LLM reading on 272 AI-written sentence pairs over the 34 memes. Qwen3-32B was not clearly better by the rule set
> beforehand. The writers were the same model, and one selection/test resemblance check is in
> `study/results/scale/prereg_inputs.json`. No planted story, pilot story or TEST item was read or written. Changes
> adopted: [list].

## Still open before F0 (not from this run)

- Choose the full or trimmed version (`README.md`, "To resume").
- Section 14.5 leaves for Mike whether the book instruction covers `study/conversation/` and `study/.gitignore`.
