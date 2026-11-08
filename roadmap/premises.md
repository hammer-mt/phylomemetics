# How strong are the two premises?

Written 8 November 2026. No experiment was run for this review. Numbers from the Batman study name the committed file they come from. The literature was read by AI research agents with web search, and a second agent checked each citation; the full notes and check logs are in [`notes/`](notes/). Work from 2025 and 2026 is recent and thinly reviewed, so treat it as leads to confirm.

The programme rests on two claims.

- **P1, the unit.** A meme can be measured by which neurons fire inside a language model.
- **P2, generality.** The measurement carries over, because different models converge on the same neurons and these can be mapped between models.

## The short answer

P1 holds in a weak form and is untested in the form the programme needs. A neuron reading is cheap, countable, repeatable and citable. Nothing yet shows that it measures an idea and not a word, and nothing shows it does better than counting words.

P2 is true for coarse geometry, partly true for directions with a learned map, and mostly false for individual neurons. The three papers the Batman paper cites for it support only the weakest reading.

The two premises also pull against each other. The neurons that match best across models are the ones a single word reproduces. Filtering memes by "shared across models" selects name and word detectors, which word counts already measure.

## P1: what the source says and what it does not

The source is Nick Cammarata's essay ["Neural networks are already clean"](https://already-clean.goodfire.com/) (Goodfire, published 15 October 2026, revised 30 October). Its claim is that most neuron poles in modern language models have one main job, and that the mess seen in earlier models came from weak tools and small models, not from unrelated concepts being packed together.

What the essay claims, and how firmly:

- For Qwen3-32B: about 90% of neuron poles have one main job, about 7% have two related jobs, and 2 to 4% are not understood. The essay calls these "preliminary audits". The grading was done by the authors' own agents. A technical companion paper is promised and was not out on 8 November 2026.
- The essay's own redo of GPT-2 XL: about 70% of neurons do "roughly one thing", about 5% are pristine. Smaller models have broader neurons.
- A label counts only after steering, circuit tracing, a look at the negative pole, and tests the explanation could lose. Reading the top examples is the method the essay says mislabels neurons.
- Neurons for feelings, social cues and tropes are common, but they often fire on commas and summary positions, not on the words that carry the idea.
- One concept often needs several neurons. The essay's examples use hubs of 4, 8 and 20.

What others found when they checked (all October 2026, none peer reviewed):

- An outside retest by AI research agents ([IRSI](https://irsi.ai/notes/already-clean/)) got 78% when grading the authors' write-ups and 57 to 65% with an independent labeller, against 31% for random directions. Planted two-job mixtures were called one-job 21 to 40% of the time, so the grading cannot reliably see a mixed neuron when one is there.
- Neel Nanda's reply on LessWrong: superposition is still real, weak activations probably hold a lot of mixed content, and models hold more facts than they have neurons.
- The sharpest objection for this project, from the same thread: showing that most neurons encode one concept is not the same as showing that most concepts are encoded by one neuron.

So three claims are being run together.

| Claim | Status | Evidence |
| --- | --- | --- |
| A neuron's strong activations have one describable job | Partly supported; depends on who grades | 90% (authors), 57 to 78% (outside retest), 31% for random directions |
| A concept is carried by one neuron | Not supported | The essay's own hubs of 4 to 20 neurons; Arora et al. 2026 find "small sets of neurons" (arXiv:2601.22594) |
| The neurons list every concept a model has | Rejected by all sides | Models hold more features than neurons |

The Batman study borrowed the unit and skipped the evidence standard.

- It used Qwen3-8B and Llama 3.1 8B at 4-bit. Nobody has audited models this small or this compressed.
- It labelled units from their top eight sentences in a 3,700-sentence corpus. That is the top-examples method.
- A unit counted as present when it beat 68 of 69 reference films. That is a rank within the corpus. It never asks whether the neuron is in its own strong range, which is the only range the essay vouches for.
- Nothing in the repository turns a neuron up or down (`study/extract.py` only records).
- 95% of the "meme families" are a single neuron pole (`study/results/qwen3-8b/memes.json.gz`).

Its results fit the sceptical reading.

- 24 of the 33 strongest units are name or place detectors, and a single word reproduces their profile at a median correlation of 0.85 (`study/results/robustness/nulls.json`).
- Only 2 of 31 Qwen units respond more to their own rewording than to other memes' rewordings (`study/results/wording.json`).
- Random unselected neurons recovered source comics better than the selected memes: mean rank 4.2 and 3.4 against 5.4 and 5.3 (`study/results/phylo.json.gz`).
- With word similarity regressed out, Qwen's neurons no longer point to the sources (p = 0.09 to 0.25). Llama's keep a small signal (p = 0.010 to 0.033), which weakens to 0.057 to 0.065 under the study's stricter nulls and is uncorrected for six tests (`study/results/experiments.json`).

Where "one neuron, one job" is best supported: the hidden neurons of the MLP blocks, at strong activation, in middle layers, in larger models, for word-like and name-like content. Where it is not: the residual stream, attention heads, weak activations, and facts. Whether it holds for story motifs across rewording has not been tested by anyone.

## P2: three levels

| Level | Verdict | Evidence |
| --- | --- | --- |
| Same geometry: models agree on which texts are alike | True, and cheap | Qwen and Llama distance matrices correlate at 0.98, but random neurons give 0.90 to 0.93 and two word methods agree at 0.92 (`nulls.json`). In the literature, local neighbourhoods agree; a single shared global geometry is disputed (Gröger et al. 2026, arXiv:2602.14486; Koepke et al. 2026, arXiv:2604.18572) |
| Same features or directions, with a learned map | Partly true | In one study of 7B models, about half of matched feature pairs pass a correlation of 0.6 (Agarwal 2026, arXiv:2608.05164). An earlier study within model families kept 10 to 30% of pairs after a looser filter (Lan et al. 2024, arXiv:2410.06981). Neither is peer reviewed. A joint dictionary has been trained on this project's exact pair (Jiralerspong and Bricken 2026, arXiv:2602.11729). The instrument is shaky: in the worst reported case, sparse autoencoders share about 30% of features across random seeds (Paulo and Belrose 2025, arXiv:2501.16615) |
| Same individual neurons | Mostly false | 1.2% to 4.2% of GPT-2 neurons are universal across retrainings of the same architecture, and the authors conclude neurons are not the right unit (Gurnee et al. 2024, arXiv:2401.12181). Here the median twin correlation over all 14,843 families is 0.40 against 0.36 for random units (`phylo.json.gz`) |

What the three cited papers show:

- Gurnee et al. is evidence against P2 at the neuron level.
- Huh et al. (the platonic representation hypothesis) is about distances between inputs, not units, and its claim that models converge as they scale has two 2026 rebuttals.
- Jha et al. (vec2vec) translates the output vectors of small sentence encoders. It does not touch the inside of a chat model. It is also fragile: for one pair of encoders from different families, 3 of 15 random seeds converged, and each pair took days of GPU time.

What the study found:

- The 33 strongest memes have twins in Llama at a median correlation of 0.82. Control units that are equally word-like reach 0.86.
- With names removed, the same twin pairs fall to 0.14. A fresh search on the name-free text recovers 0.54, still below word-like controls at 0.69.
- The 27 concept memes beat word-matched controls on the original texts (0.62 against 0.46, p = 0.001) and not on the name-free texts (0.34 against 0.38).
- The two models' revenge units correlate at 0.39. Llama's works without revenge words; Qwen's is a keyword detector.
- Qwen3-1.7B, from the same family, agrees with Qwen3-8B no better than Llama does. The level of agreement seems to come from the text.

All from `study/results/robustness/nulls.json` and `study/results/phylo.json.gz`.

Conditions under which mapping between models works, from the literature: the same texts run through both models, pooling to the sentence, middle layers, models of similar size, frequent and concrete features, and a map between subspaces, not single units. The project has all the inputs for this and has tried none of it.

The constructive lead: inside one model, middle-layer features fire for a concept whatever language it is written in (Anthropic 2025, "On the biology of a large language model"). That is the best evidence that word-free units exist. Translation is also a cleaner way to remove shared wording than name removal.

## A version of the premises the evidence supports

- **P1, restated.** The hidden neurons of a public model are a fixed, citable, cheap set of detectors in which a majority have a describable job at strong activation. In small models most of those jobs are lexical. A meme is more likely a small set of neurons or a direction than one neuron. Neurons are a convenient readout, not yet the natural unit of culture.
- **P2, restated.** Models of similar size share a large part of their feature space at middle depth, and a map fitted on paired text carries a direction from one to the other. They do not share neurons, apart from a small word-like minority.

One real advantage of the raw neuron is worth keeping. A neuron in a public checkpoint is a fixed label that two labs can cite. Sparse-autoencoder features change with the random seed and the dictionary size. Microbiology moved from clustered units to exact sequence variants for the same reason.

## The validation ladder

New measurements have never started with ground truth. Temperature, the gene and the species all began as an operational unit with a decidable test and were corrected by iteration (Chang 2004, *Inventing Temperature*). The missing definition of "meme" is not the obstacle. The obstacle is that the instrument has not passed the tests those fields used.

| Rung | The test | Cheap rival | Where the instrument stands |
| --- | --- | --- | --- |
| 0. Specification | Unit, bar and scoring frozen before test data exist | none | Partly: pre-registration exists, timing cannot be proven from git |
| 1. Reliability | Same motif, different wording, language, length, position, prompt | sentence embeddings | Fails for about two thirds of units on rewording; the rest untested |
| 2. Invariance | Same reading in another model, language or medium | embeddings moved by a linear map | Fails for the top units without names; partial for concept units |
| 3. Convergent validity | Agreement with human annotators, TV Tropes, folklore motif indexes | an LLM reading the text | Untested. No human has checked anything |
| 4. Discriminant validity | Motif present without its words; words present without the motif | TF-IDF, embeddings | Fails in Qwen, weak pass in Llama. This is the rung that decides |
| 5. Predictive validity | Lineage on a fresh franchise; works after the training cut-off | TF-IDF, embeddings, LLM reader | Heredity holds, with no gain over words |
| 6. Causal validity | Minimal pairs; steering; silencing a unit | prompting; random units | One hand-picked revenge unit. Steering never tried |
| 7. Iteration | Revise the unit and the criterion after each round | | Not started |

Pass marks and the cheapest data for each rung are in [`notes/l13-measurement-validity.md`](notes/l13-measurement-validity.md). P2 sits on rung 2: the first test available without ground truth is that two instruments of the same kind agree.

Until a unit beats sentence embeddings and an LLM reader on text where the words are controlled (rung 4), nothing higher on the ladder says anything about neurons.

## What would settle it

1. Steer the named units and see whether the labelled idea appears, against random and word-matched units and against simply prompting.
2. An audit of the 8B 4-bit models themselves, against random directions and planted mixtures.
3. Presence defined by each neuron's own strong range, then the heredity and residual tests rerun.
4. The two-by-two of motif present or absent by key words present or absent, on planted stories with units frozen in advance. It is already designed in `preregistration/`.
5. Twins and a linear map between Qwen and Llama, fitted on one half of a corpus and scored on the other, against word-matched controls. The 1,000 held-out film summaries are already in `preregistration/sources/`.
6. Heredity across languages, where shared wording cannot carry the signal.
7. Human annotators on a few hundred passages.

The registered designs for each are in [`designs/`](designs/) and listed in [`IDEAS.md`](IDEAS.md).
