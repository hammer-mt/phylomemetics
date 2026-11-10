# How strong are the two premises?

Written 8 November 2026. Updated 10 November for two documents published that week: Goodfire's companion paper to the essay behind P1, and a 182-page survey of the field. Both were read, not rerun. No experiment was run for this review. Numbers from the Batman study name the committed file they come from. The literature was read by AI research agents with web search, and a second agent checked each citation; the full notes and check logs are in [`notes/`](notes/). Work from 2025 and 2026 is recent and thinly reviewed, so treat it as leads to confirm.

The programme rests on two claims.

- **P1, the unit.** A meme can be measured by which neurons fire inside a language model.
- **P2, generality.** The measurement carries over, because different models converge on the same neurons and these can be mapped between models.

## The short answer

P1 holds in a weak form and is untested in the form the programme needs. A neuron reading is cheap, countable, repeatable and citable. Nothing yet shows that it measures an idea and not a word, and nothing shows it does better than counting words. Goodfire's companion paper of 10 November supports the weak form for one 35B model: at strong activation about two thirds of neuron poles have one dominant job, and a short description predicts when they fire. It tests neither of those two points.

P2 is true for coarse geometry, partly true for directions with a learned map, and mostly false for individual neurons. The three papers the Batman paper cites for it support only the weakest reading.

The two premises also pull against each other. The neurons that match best across models are the ones a single word reproduces. Filtering memes by "shared across models" selects name and word detectors, which word counts already measure.

## P1: what the sources say and what they do not

There are two sources, both from Goodfire. Nick Cammarata's essay ["Neural networks are already clean"](https://already-clean.goodfire.com/) (published 15 October 2026, revised 30 October) made the claim: most neuron poles in modern language models have one main job, and the mess seen in earlier models came from weak tools and small models, not from unrelated concepts being packed together. Its technical companion, ["Neurons as an Interpretable Basis"](https://www.goodfire.com/research/neurons-as-an-interpretable-basis) (Tigges, Kowal, McGrath, Lewis and Cammarata, 10 November 2026), gives the first measurements. Neither is peer reviewed. The companion came out after the first version of this review, and no outside group has retested it yet.

### The essay

What the essay claims, and how firmly:

- For Qwen3-32B: about 90% of neuron poles have one main job, about 7% have two related jobs, and 2 to 4% are not understood. The essay calls these "preliminary audits". The grading was done by the authors' own agents.
- The essay's own redo of GPT-2 XL: about 70% of neurons do "roughly one thing", about 5% are pristine. Smaller models have broader neurons.
- For a hard neuron, the essay reads the sentences around the firing token and then checks steering, the circuit, the negative pole and tests the explanation could lose. It blames mislabelled neurons on methods that read only the top examples and ignore the negative pole and the wider context. It also says a label read from top activations usually survives its audits.
- Neurons for feelings and social cues are common, but they often fire on commas and summary positions, not on the words that carry the idea.
- One concept often needs several neurons. The essay's examples use hubs of 4, 8 and 20.

### The companion paper (added 10 November)

The paper studies one model, Qwen3.6-35B-A3B. It is a mixture-of-experts model: each layer has 256 sub-networks ("experts") of 512 neurons, and each token is sent to 8 of them. The Batman study did not use it. The paper says its interest is what a neuron computes, and treats whether the texts that fire it look alike as a secondary question. It makes three claims.

**1. Strong firings carry the computation.** At each token and layer the authors keep the 320 strongest of the 4,096 neurons in play and zero the rest, 92.2% of them. Loss rises 2.9%, from 1.886 to 1.940. Keeping a random eighth instead gives 4.214, and removing only the strongest eighth gives 4.587. Scores on reasoning benchmarks fall (13.0 points on SuperGPQA, 20.9 on MBPP+) until the model is fine-tuned on about 100 million tokens. The firings that survive lie almost entirely in the top or bottom 3% of each neuron's own range, and that is the only range the paper goes on to study.

**2. Most poles have one role, or one role that dominates.** A pole's firings are clustered by which upstream neurons drove them (a median of 19 clusters per pole), and a Claude Opus 5.5 judge groups the clusters into themes the paper calls modes. Of 5,468 poles from 2,734 neurons, 47% have one mode, and 69% have one mode that does at least 90% of the damage when the pole is removed. Among the 2,673 poles with several modes, 27% have half or more of the damage outside the main mode. The paper says the counts "are not definitive", since they depend on how finely the judge splits. The sample holds 1,000 firings from each strength band, so it describes the top of the range and not how often each mode occurs in ordinary text.

**3. A short description predicts strong firing.** An agent with tools writes a description of each pole. Three tests follow, all on text the agent did not see.

| Test | Result | Compared with |
| --- | --- | --- |
| Detection: a judge given the description picks which of two passages makes the pole fire (496 poles) | 95% against random text, 89% against text sent to the same expert; 81% for the weakest firings studied | Chance, 50%. No word list, embedding classifier or judge without the description |
| Which mode fired, for poles with several (573 poles, macro-F1, where 1 is perfect) | 0.71 from the text; 0.85 from which downstream neurons respond | Guessing by mode size, 0.29 |
| Simulation: a model predicts the activation token by token from the description, and the prediction replaces the neuron (486 poles) | Restores 73% of what a perfect predictor restores, for firings above the 99.9th percentile only | Removing the neuron. Weaker firings were not evaluated |
| Synthesis: agents write text from the description, with feedback from the real model (128 poles) | 120 of 128 beat the strongest firing in a 10-billion-token corpus, by a median factor of 1.06 | No run with a wrong description or none |

Other results in the paper that bear on a meme reading:

- **Firing size is not readable from the text.** Features of the text explain about 14% of the variance in how strongly a neuron fires, against 59% for the activity of its own inputs. About 15% of example firings stop when the passage is rerun without the document around it.
- **Random directions are readable too.** The paper's judge found one main theme covering a median 0.65 to 0.73 of the firings of a random direction inside an expert, at every strength. For neurons the share rises with strength: 0.32 in the weakest band studied, 0.87 and 0.93 in the two strongest. Neurons are the clearly better basis on sparsity: keeping 8 neurons per token holds as much of a layer's effect as 128 to 256 random directions.
- **Many neurons carry one prediction.** In the paper's audit of one "Colorado" neuron in Qwen3-32B, half of the push toward the word comes from 155 neurons and 80% from 2,299.
- **Counting firings and weighing them give different pictures.** That neuron fires with no place nearby on 61% of its tokens, and those firings carry about 1.2% of its effect on the model's output. On ordinary text 98.5% of its strong firings were on places. On a corpus of code 98% were on a Python library called Tornado. A label depends on the text it was audited on.
- **About a third of all activations are set aside.** Activations where both of a neuron's inputs are negative behave differently and are left to later work.

What the paper does not show, for this programme:

- No comparison with a cheap rival anywhere. The only comparator for detection is chance.
- No test of a unit following an idea when its usual words are absent. Several showcased units follow a word or phrase.
- Nothing on a second model, on dense 8B models, or on 4-bit weights. The authors list generality across models as still to be established.
- Nothing at the level of a document. Every test is one marked token in a window of 58 to 69 tokens around a known strong firing. The false-firing rate on ordinary text is not reported.
- No human grading. Models wrote, grouped and judged everything.

The essay and the paper do not give the same number. The essay's 90% is "one main job" in Qwen3-32B by an unpublished rubric. The paper's nearest figures are 47% and 69% in a different model by a stated one. The two are not directly comparable, and the paper's is the one with a method behind it.

### What others found when they checked the essay

All October 2026, none peer reviewed:

- An outside retest by AI research agents ([IRSI](https://irsi.ai/notes/already-clean/)) got 78% when grading the authors' write-ups and 57 to 65% with an independent labeller, against 31% for random directions. Planted two-job mixtures were called one-job 21 to 40% of the time, so the grading cannot reliably see a mixed neuron when one is there.
- A replication on Qwen3-0.6B and Qwen3-1.7B ([Batutin](https://andriybatutin.substack.com/p/we-tried-to-confirm-neural-networks), a blog post) found that labels held only at peak activation, that none of 50 neurons moved the tokens its label predicted when pushed or silenced, and that the labelling method gave 84% of a randomly initialised network's neurons one job against 52% for the trained one. Qwen3-1.7B is one of this project's models. A second replication on tiny self-trained models mostly supported the essay.
- Neel Nanda's reply on LessWrong: superposition is still real, weak activations probably hold a lot of mixed content, and models hold more facts than they have neurons.
- The sharpest objection for this project, from the same thread: showing that most neurons encode one concept is not the same as showing that most concepts are encoded by one neuron.

The companion paper answers these in part. It runs random directions through its own pipeline, and it shows that weak activations matter little to the model's output. It does not test a randomly initialised network, does not steer at scale, and its own figures (155 neurons for half of one prediction) agree that a concept is not carried by one neuron.

So several claims are being run together.

| Claim | Status | Evidence |
| --- | --- | --- |
| A model's output depends mostly on its strongest neuron firings | Supported in one model | Loss up 2.9% with 92.2% of activations zeroed; a random or inverted choice more than doubles it. Reasoning needs a fine-tune to recover |
| A neuron's strong firings have one describable job | Partly supported; depends on who grades and how | 47% one mode, 69% one dominant mode (the paper); 90% (the essay); 57 to 78% (outside retest of the essay); random directions in an expert score 0.65 to 0.73 on the paper's readability measure |
| A description predicts when a neuron fires strongly | Supported in one model, against chance only | 95% and 89% in a two-way choice; 73% of a ceiling in simulation. No cheap rival was run |
| A concept is carried by one neuron | Not supported | The essay's own hubs of 4 to 20 neurons; 155 neurons for half of one prediction (the paper); Arora et al. 2026 find "small sets of neurons" (arXiv:2601.22594) |
| The neurons list every concept a model has | Rejected by all sides | Models hold more features than neurons |
| A neuron reading measures an idea and not a word | Untested by anyone | Not addressed by the essay or the paper. The Batman study's results point the other way |

### Where the paper differs from the essay

| Point | The essay (15 October) | The companion paper (10 November) | Reading |
| --- | --- | --- | --- |
| Share of poles with one job | "Roughly 90%" have one main job, in "preliminary audits" of Qwen3-32B | 47% have one mode and 69% one dominant mode, of 5,468 poles of Qwen3.6-35B-A3B, on strong firings, grouped by an LLM judge | Not the same model or the same test. The paper gives no figure for Qwen3-32B |
| Removing weak activations | "You can ablate almost all the weak activations of the network for almost no cost" | Loss rises 2.9%. Reasoning, knowledge and coding benchmarks fall by up to 20.9 points until the model is fine-tuned | True for loss. Not true for reasoning unless the model is retrained |
| Dense models | "Dense and Mixture of Experts models seem roughly similar", with no model named | Every statistic is for one mixture-of-experts model | The sentence that matters for this project's dense models is not backed by the paper |
| Smaller models | GPT-2 XL: about 70% do "roughly one thing", about 5% are pristine; "smaller models have broader neurons" | Nothing below 32B. An 8B model appears only as the tool that wrote first-pass labels | How clean an 8B model is still rests on one aside, and the aside says smaller is broader |
| Larger models | "Similar results in models up to trillions of parameters", which "the companion paper will cover" | Not covered. The paper calls itself the first in a series | Still unpublished |
| What counts as strong | About 160 of 100 million tokens reach "the strong half" of one neuron's range | The top or bottom 3% of a neuron's activations | "Strong" is a far wider band in the paper |
| Writing text that fires a neuron | A second agent that sees only the explanation writes one example, which "can" beat anything in 10 billion tokens. No rate | 120 of 128 poles, after several rounds with feedback from the real model; median gain 6% | The paper's version is a search with feedback, not a blind attempt |
| Simulation as a test | "I no longer think it's a great metric for explanations" | The headline test of a description | The paper scores restored damage, not matched activations, and only for the strongest firings |

The announcement thread on X is looser than the paper in two places. It says light fine-tuning "fully recovers" benchmark performance, and that descriptions for 1,500 neurons are released. The paper says the release covers 2,734 neurons. The public [Neuron Atlas](https://www.goodfire.com/research/public_neuron_atlas) shows 3,000 poles and states no licence, and the labels for all 10 million poles that the paper mentions could not be found for download on 10 November. Quote the paper's tables, not the posts.

### The conditions under which P1 holds best

On today's evidence, "a neuron has a describable job" is best supported under these conditions. A meme reading that leaves them is on untested ground.

- **Unit.** One pole (the positive or the negative side) of one hidden neuron of an MLP block, read at the token where it fires. Not the residual stream and not attention heads.
- **Range.** The top or bottom 3% of the neuron's own activations, and best in the top 0.1%. Detection is 95% right at the 99.99th percentile and up, against 86% between the 97th and 99th. Simulation was tested only above the 99.9th.
- **Gate.** Firings with the neuron's gate open. Firings where both of its inputs are negative are about a third of all activations and are not explained.
- **Layers.** The middle of the network. The outside retest of the essay scored highest in layers 16 to 31.
- **Model size.** 32B parameters and up. Nothing is reported at 8B, the essay says smaller models have broader neurons, and the one replication on small Qwen models was negative.
- **Model type and precision.** One mixture-of-experts model, whose precision the paper does not state. Nothing carries to a dense 4-bit model without a check.
- **Content.** Words, names, set phrases and word classes, which is what most of the worked examples are. A few idea-like units are shown, with no numbers. Story motifs across rewording are untested.
- **How it is weighed.** By effect on the model's output, not by count of firings. The paper's Colorado neuron is on-label for about 99% of its effect and for 39% of the tokens where it fires.
- **Corpus.** The kind of text the label was written on. The same neuron reads as places on ordinary text and as a code library on code.
- **Context.** With the whole document in view. About 15% of example firings stop when the model sees only the short passage.
- **Grader.** The authors' own LLM judges. An independent labeller got 57 to 65% for the essay's model, and no human has graded either text.
- **Level.** One marked token in a short passage. No result exists for a reading of a whole document, such as the study's rule of beating 68 of 69 reference films.

The Batman study borrowed the unit and ran none of the checks the essay and the paper rely on.

- It used Qwen3-8B and Llama 3.1 8B at 4-bit. Nobody has audited the 8B 4-bit models. The paper's audit is of a 35B mixture-of-experts model, and its authors say how far the findings carry to other models is still to be established.
- It labelled units from their top eight sentences in a 3,700-sentence corpus. That is the top-examples method. The paper's descriptions come from an agent that sees 200 documents spread across the neuron's range and can run experiments, and they are scored on held-out text.
- A unit counted as present when it beat 68 of 69 reference films. That is a rank within the corpus. It never asks whether the neuron is in its own strong range, which is the only range the essay and the paper vouch for. In the paper that range is the top or bottom 3% of the neuron's own activations.
- It gave each unit one label. The paper finds that about half of poles have more than one role, and that the text alone often cannot say which one fired.
- It scored a work by the mean of its three strongest sentences. The paper finds that the text explains little of how strongly a neuron fires, so firing size is a poor measure of how much of an idea is present.
- Nothing in the repository turns a neuron up or down (`study/extract.py` only records).
- 95% of the "meme families" are a single neuron pole (`study/results/qwen3-8b/memes.json.gz`).

Its results fit the sceptical reading.

- 24 of the 33 strongest units are name or place detectors. A single word reproduces the 33 units' profiles at a median correlation of 0.85, and the 24 name detectors' at 0.94 (`study/results/robustness/nulls.json`).
- Only 2 of 31 Qwen units respond more to their own rewording than to other memes' rewordings (`study/results/robustness/nulls.json`).
- Random unselected neurons recovered source comics better than the selected memes: mean rank 4.2 and 3.4 against 5.4 and 5.3 (`study/results/phylo.json.gz`).
- With word similarity regressed out, Qwen's neurons no longer point to the sources (p = 0.09 to 0.25). Llama's keep a small signal (p = 0.010 to 0.033), which weakens to 0.017 to 0.065 under the study's stricter nulls and is uncorrected for six tests. Only the version with TF-IDF regressed out passes 0.05, at 0.057 and 0.065 (`study/results/experiments.json`).

Where "one neuron, one or a few jobs" is best supported: one pole of a hidden MLP neuron, in the top or bottom 3% of its own activations and most clearly above the 99.9th percentile, in middle layers, in models of 32B parameters and up, for word-like, phrase-like and name-like content. Where it is not: the residual stream, attention heads, weak activations, and facts. Whether it holds for story motifs across rewording has not been tested by anyone, the companion paper included.

## P2: three levels

| Level | Verdict | Evidence |
| --- | --- | --- |
| Same geometry: models agree on which texts are alike | True, and cheap | Qwen and Llama distance matrices correlate at 0.98, but random neurons give 0.90 to 0.93 and two word methods agree at 0.92 (`nulls.json`). In the literature, local neighbourhoods agree; a single shared global geometry is disputed (Gröger et al. 2026, arXiv:2602.14486; Koepke et al. 2026, arXiv:2604.18572) |
| Same features or directions, with a learned map | Partly true | In one study of 7B models, about half of matched feature pairs pass a correlation of 0.6 (Agarwal 2026, arXiv:2608.05164). An earlier study within model families kept 10 to 30% of pairs after a looser filter (Lan et al. 2024, arXiv:2410.06981). Neither is peer reviewed. A joint dictionary has been trained on this project's exact pair (Jiralerspong and Bricken 2026, arXiv:2602.11729). The instrument is shaky: in the worst reported case, sparse autoencoders share about 30% of features across random seeds (Paulo and Belrose 2025, arXiv:2501.16615) |
| Same individual neurons | Mostly false | 1.2% to 4.2% of GPT-2 neurons are universal across retrainings of the same architecture, and the authors conclude neurons are not the right unit (Gurnee et al. 2024, arXiv:2401.12181). Here the median twin correlation over all 14,843 families is 0.40 against 0.36 for random units (`phylo.json.gz`) |

What the documents of November 2026 add:

- **Goodfire's companion paper has nothing on P2.** It studies one model and maps nothing onto another. It adds two cautions. Its strength bands and its hardest control depend on expert routing, which this project's dense models lack, so its thresholds do not carry over unmeasured. And its authors suspect that Qwen3-32B was built by copying layers of Qwen3-14B, since many neurons match 24 layers apart. If models in one family share copied layers, agreement between Qwen3-8B and Qwen3-1.7B may be inheritance and not convergence.
- **Shared neurons thin out as models grow.** Dravid et al. 2026 (arXiv:2606.03990) follow neurons whose firing patterns match across independently trained models, in language models up to 30B parameters. Their number grows with model size but more slowly than the model does, so they are a shrinking share of all neurons. They also become more selective. A cross-model reading built on matched neurons would cover less of a larger model.

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

- **P1, restated.** The hidden neurons of a public model are a fixed, citable, cheap set of detectors. In the one model audited in full, a 35B one, about two thirds of neuron poles have one job or one dominant job in their strongest 3% of firings. Nothing is known for 8B or 4-bit models. In small models most of those jobs are lexical. A meme is more likely a small set of neurons or a direction than one neuron. Where a single pole is used, it has to be read in its own strong range, with the role that fired identified. Neurons are a convenient readout, not yet the natural unit of culture.
- **P2, restated.** Models of similar size share a large part of their feature space at middle depth, and a map fitted on paired text carries a direction from one to the other. They do not share neurons, apart from a small word-like minority.

One real advantage of the raw neuron is worth keeping. A neuron in a public checkpoint is a fixed label that two labs can cite. Sparse-autoencoder features change with the random seed and the dictionary size. A 2026 workshop paper found that across 13 public transcoders, a related kind of learned dictionary, only the features that copy a single neuron had a stable counterpart in the network; the others varied more between training runs (Shabalin and Piotrowski, arXiv:2610.24521). Microbiology moved from clustered units to exact sequence variants for the same reason.

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

Goodfire's four tests of a description (detection, which role fired, simulation and synthesis) are tests of a label, on rungs 1 and 6, scored against chance. They show that a label can be made to predict strong firing. None is a rung 4 test and none includes a cheap rival.

Until a unit beats sentence embeddings and an LLM reader on text where the words are controlled (rung 4), nothing higher on the ladder says anything about neurons.

## What would settle it

1. Steer the named units and see whether the labelled idea appears, against random and word-matched units and against simply prompting (design EN-21).
2. An audit of the 8B 4-bit models themselves, against random directions and planted mixtures (IN-22), a check that 4-bit readings match full precision (IN-29), and the companion paper's sparsity test repeated on them (IN-32).
3. Presence defined by each neuron's own strong range, which the companion paper puts at the top or bottom 3% of its activations, then the heredity and residual tests rerun (IN-23).
4. The two-by-two of motif present or absent by key words present or absent, with units frozen in advance. Design IN-21 does this at sentence level in days. The planted-story plan in `preregistration/` asks the same question at about 4,400 agent sessions.
5. Twins and a linear map between Qwen and Llama, fitted on one half of a corpus and scored on the other, against word-matched controls (UN-21, UN-23). The 1,000 held-out film summaries are already in `preregistration/sources/`.
6. Readings and heredity across languages, where shared wording cannot carry the signal (UN-22, LI-22).
7. Human judgement: existing human coding of retold stories (CE-21), then paid annotators on a few hundred passages (IN-26).
8. The companion paper's label tests run on this project's own units, with the rivals the paper left out: a table of each token's average activation, another unit's label, and no label (IN-33). Then its synthesis loop with the unit's usual words banned (IN-31).

The registered designs are in [`designs/`](designs/) and listed in [`IDEAS.md`](IDEAS.md). [`README.md`](README.md) gives the order in which to run them.
