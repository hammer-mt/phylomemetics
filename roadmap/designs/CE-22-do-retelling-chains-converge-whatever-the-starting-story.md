# CE-22: Do retelling chains converge whatever the starting story?

**Area:** Cultural evolution, re-run · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

Do chains started from different texts drift toward the same forms, and can units separate content copied from the seed from content that appears whatever the seed?

## Prediction

With retellings held to about 150 words, the starting seed's share of unit-profile variation falls at least 30% by generation 8 under an open-ended retelling prompt, more than under a plain rephrase prompt. After subtracting the rephrase arm's drift (house style), a drift direction fitted on 20 seeds predicts drift on 20 held-out seeds with cosine at least 0.3.

## What would refute it

Seed share falls less than 10%, or held-out drift cosine beats random matched units by under 0.05, or sentence-embedding centroids show the same drift consistency (within 0.05). 40 seeds, 3 chains, 8 generations, 2 writers, 2 prompts: 3,840 texts. Inference is by seed: 20 held-out seeds give a standard error near 0.03 if seeds spread 0.15; confirm by simulation.

## Data

Seeds: the 20 initial texts in Perez et al.'s public code (GitHub jeremyperez2/TelephoneGameLLM; no licence file, so commit hashes only) plus 20 tale openings from Hagedorn and Darányi's Folktexts (1,518 tales; Zenodo 6575263; CC BY-SA 4.0 per its README). Chains generated locally. Human check: Breithaupt et al., OSF jr2py, 116 stories with 3 human retellings each; no licence stated.

## Method

Unit profile per text from a reader model that wrote no chain. Statistics: seed share of variance by generation; per-unit rate of persisting where seeded against appearing where not seeded; held-out drift cosine. Nulls: texts re-paired across chains within generation; random firing-rate-matched units. Drift sign checked on human chains within length bands. Repeat with Llama Scope sparse-autoencoder features.

## Cheap rival it must beat

TF-IDF vectors (words weighted by rarity), sentence-embedding centroids, mean-pooled residual stream, text length, random units, and the published property scorers (toxicity, positivity, difficulty). Most dangerous: embedding centroids, which see convergence of style and tone cheaply. If they win, attraction is confirmed in open models and units add only names for what converges.

## Premise it tests

P1, with a side reading on P2. If widely shared units are attractors, similarity without descent is expected and unit trees are exposed to false kinship, which would explain the failed identity test. Two writer models agreeing on the attractor is mild behavioural support for P2.

## Biology analogue

Convergent evolution; homoplasy

## Cost

S. About 11 hours of local generation and one hour of extraction.

## Payoff

A list of attractor units to exclude from lineage work, and a measured split of stability into copying and convergence.

## Main risk

Once house style is subtracted nothing may be left to converge; or writer models differ so much that no common attractor exists.

## Reviewer note

Perez et al. already showed attractors on four scalar properties; the news here is the unit-level list. Fixed text length and switched to seed share of variance, since shrinking texts lower any count's variance; subtracted rephrase drift as house style; inference by seed. Weakness: reader and writers share LLM style.

## Overlaps with

Asks nearly the same question as ME-22, which the judges would run instead.

## Sources

- Perez et al. 2024 (arXiv:2407.04503)
- Acerbi et al. 2021
- Claidière, Scott-Phillips & Sperber 2014
- Breithaupt et al. 2024
