# AT-24: Which kind of meme name reproduces across labs?

**Area:** The atlas · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

Given only a published name, can an independently built second pipeline reproduce the first one's presence calls, and which kind of name is both reproducible and valid: a neuron address, a family cluster, a cross-model twin or a written definition?

## Prediction

Neuron addresses: at least 90 of 100 names reach kappa 0.80 between pipelines (4-bit against 8-bit weights, different runtimes). Family clusters rebuilt from the published rule on a second corpus: under 60. Twin names in a second model, on name-free text: under 30. No fewer addresses than written definitions are both reproducible and valid (AUC 0.70 on reworded passages).

## What would refute it

Fewer than 70 of 100 address names reach kappa 0.80, or rebuilt family names reproduce as often as addresses (a difference under 10 names), or written definitions give 10 or more additional names that are both reproducible and valid. With 2,000 passages per name, kappa is known to about 0.05, and 100 names fix each share to within 10 points.

## Data

2,000 passages: plots of the 1,000 accepted random films in preregistration/sources/t3_random_films.jsonl.gz (Wikipedia, CC BY-SA 4.0) and 1,000 reworded AT-21 test passages with known motifs. Names: the best entry for each of 100 AT-21 motifs; the 31 labelled units in study/results/labels.json are reported separately. Checkpoints: Llama-3.1-8B-Instruct and Qwen3-8B, pinned revisions.

## Method

A name is checkpoint revision, layer, neuron, sign, readout rule and threshold corpus, each hashed. A second pipeline is written from the specification alone, on another runtime and precision. Statistic: Cohen's kappa (agreement beyond chance) per name; validity: AUC on reworded passages. Null: kappa between different units matched on firing rate. Wording cannot confound kappa: both pipelines read identical text.

## Cheap rival it must beat

A keyword pattern (reproduces perfectly by construction, at chance on reworded passages), an embedding prototype with a named encoder, a Llama Scope sparse-autoencoder feature index, and a written definition read by two different LLMs on a 500-passage sample. The written definition is the danger: probably more valid, and if two readers agree at kappa 0.80 it is the better name.

## Premise it tests

Both. A stable address is the one practical advantage of raw neurons over learned features (P1). The twin name is P2 stated as a naming rule: if it fails on name-free text, a meme name is good in one checkpoint only.

## Biology analogue

Exact sequence variants against clustered OTUs; accession numbers

## Cost

M. Two extraction passes per checkpoint on one laptop (8-bit weights of an 8B model fit in 24 GB), a second pipeline written from the specification, two LLM readers on 500 passages (under 30 US dollars); about four days.

## Payoff

A written naming standard with a reproducibility and a validity figure for each kind of name, and the first check that 4-bit weights do not move the units. Two labs could then cite one meme and know how far their readings will agree.

## Main risk

Both pipelines come from one person and one family of coding assistants, so shared habits inflate agreement. This is a rehearsal; only a real second lab tests the standard. Address agreement is close to guaranteed, so most of the information sits in the family, twin and definition rows.

## Reviewer note

Changed: every name now carries a motif, so it is scored for validity as well as reproducibility, and the written definition must be beaten on both; cost raised to M. Remaining: IN29 already tests precision, address agreement is near-certain by construction, and the study showed twins failing on name-free text.

## Sources

- Callahan, McMurdie and Holmes 2017, doi:10.1038/ismej.2017.119
- Paulo and Belrose 2025, arXiv:2501.16615
- Gurnee et al. 2024, arXiv:2401.12181
- study/results/robustness/nulls.json (f2_agreement)
