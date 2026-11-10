# l14-goodfire-companion: what Goodfire's companion paper shows, and what it leaves untested

Source: Tigges, Kowal, McGrath, Lewis and Cammarata, Neurons as an Interpretable Basis (Goodfire, 10 November 2026), https://www.goodfire.com/research/neurons-as-an-interpretable-basis . It is the companion to Cammarata's essay Neural Networks Are Already Clean, the source of P1. Also covered: Goodfire's announcement posts, its public Neuron Atlas and the first reactions.

How this note was made: AI readers read the paper section by section, and a second agent checked each extraction against the text. Numbers that sit only in interactive figures were read from the saved page. Nothing was run: no model, no pipeline, no experiment.

## Bottom line

- **The paper claims less than the essay.** The authors write that the picture is "not that every neuron means one thing". Their claim: in one 35B model, a neuron's strongest firings (activations) have a few roles that a short description predicts. They call the approach "computational/functional rather than semantic".
- **Its headline share is 69%, not 90%.** Of 5,468 neuron poles (a pole is the positive or the negative side of one neuron), 47% have one role and 69% have one role or one dominant role. The count covers each pole's strongest 3% of firings, and one LLM judge (a language model used as grader) drew the roles. The essay said roughly 90%, for another model.
- **Sparsity holds for prediction error and fails on most reasoning benchmarks until the model is retrained.** Zeroing the weakest 92.2% of active neuron activations raises loss (the error in predicting the next word piece) by 2.9%. Of the paper's ten reasoning benchmarks, six fall, by up to 20.9 points, until the weights are trained further. Four rise.
- **Descriptions predict strong firings, and no cheap rival is run.** Detection is scored against chance only. No test sets word counts or sentence embeddings (numeric summaries of a sentence's meaning) against a description, and none uses reworded text or a human grader.
- **Some random directions are readable too.** A random direction is a random mix of units treated as one unit. In the paper's figure data an LLM reader finds one main meaning in about two thirds of the firings of such a direction drawn inside one of the model's sub-networks. For the paper's other two kinds of random direction the share is 0.16 to 0.39.
- **An idea takes many neurons, and one neuron fires mostly off-label by count.** Half of the push toward one place name comes from 155 neurons. In the one neuron audited in depth, a place neuron, 61% of the word pieces where it fires have no place nearby.
- **Nothing bears on P2 or on the study's models.** All main results are on one mixture-of-experts model (each word piece goes to a few of its sub-networks). The authors say: "We must also establish how far these findings generalize across models and behaviors." No outside retest exists yet.

## Findings

### The model and the unit

- Model: Qwen3.6-35B-A3B. Each of its 40 layers holds 256 experts, which are blocks of 512 neurons. A router sends each token (word piece) to 8 experts per layer, and one shared expert is always on. One appendix case study uses one neuron of Qwen3-32B, a dense model (no experts; every neuron runs on every token).
- Unit: one pole of one MLP neuron (a hidden unit in a layer's feed-forward block). Its activation is the product of two input signals, gate and up, times the router's weight for its expert. The authors say the two poles usually mean unrelated things. In name this is the Batman study's unit.
- Range: each neuron is ranked against its own activations over a 10-billion-token corpus. Only the tails are studied: at or above its 97th percentile, or at or below its 3rd. This note calls that the strong band.
- The authors read a neuron by its inputs (earlier neurons that drive it) and its readers (later neurons that use it).
- Sample: 2,734 neurons (5,468 poles), called stratified in one place and random in another. The strata are not stated.

### Sparsity and its controls

- Method (section Router-Weighted Top-K Filtering): at every token and layer, keep the 320 strongest of the 4,096 routed activations (those of the 8 chosen experts) and zero the other 92.2%.
- Result on 8.4M tokens of web text (Table 1): loss rises from 1.886 to 1.940, which is 2.9%. Two controls wreck the model. Keeping a random 64 neurons per expert (87.5% zeroed) gives loss 4.214. Removing only the strongest 64 per expert (12.5% zeroed) gives 4.587.
- Before retraining, six classic benchmarks move by between +0.3 and -2.2 points. Of the ten reasoning benchmarks, six fall: for example SuperGPQA by 13.0 points from a base of 44.8, MMLU-Pro by 8.3 from 78.4, MBPP+ by 20.9 from 71.7. Four rise, HumanEval+ by 15.9.
- Fine-tuning (further training) was needed (appendix Fine-tuning for performance recovery, Table 8). All weights were trained with the zeroing in place on 100M tokens: 91M of imitating the intact model and 9M of reinforcement learning (reward for right answers). Afterwards the sparse model is 0.2 points above the untrained base on SuperGPQA and 2.9 points below it on MMLU-Pro (the 95% interval excludes zero).
- Controls for the fine-tune: the intact model given the same recipe, without the imitation step, gains 5.1 points on SuperGPQA, against 0.2 for the sparse model. The two wrecked controls stay 32.5 and 39.8 points down on it. A new random seed alone moves the base model's score by 2.6 points.
- Authors: the fine-tune mainly corrected the reasoning, teaching the model to commit to an answer, and did not add new information. That neurons are the natural axes of the computation is offered as a hypothesis.
- Reviewer: 92.2% is a share of routed activations, and only 8 of 256 experts run per token. After the zeroing, 320 routed neurons per layer still fire on every token, and the always-on shared expert is left untouched. Nothing here shows that one neuron carries one idea.
- Circuits (appendix Sparse neuron circuits): when only the current token's neurons are removed and earlier tokens are left intact, a median 519 of 163,840 active neurons restore 90% of the model's preference for the token it generated (730 tokens from ten texts). When the earlier tokens are stripped too, the count grows with the length of the text, to about 1% of the neurons active so far.

### The double-negative activations

- A neuron's output is also positive when gate and up are both negative. The authors call this regime Q3. It is about one third of all activations: 35 to 36% of routed cells (a cell is one neuron at one token). Among the 100 cells ranked most causal it is 1 to 8%.
- Authors: Q3 firings tend to be unrelated to the pole's other examples and matter less, except for reasoning. On 24 selected maths problems, zeroing Q3 cut reasoning from 1,550 tokens on average to 176. Real Q3 activations restored long reasoning even when played backwards (13 of 17 problems). Their role is left to future work.
- Reviewer: the problems were selected and few. Q3 firings carry a positive sign, so an instrument that stores only the output reads them as positive-pole firings.

### Subcircuits and modes

- A subcircuit is a group of one pole's strong firings that share the same input neurons. There is a median of 19 per pole, range 1 to 51. On split halves of the documents, 68% of subcircuits were recovered, against 1% when the data were shuffled.
- A mode is a theme that a Claude Opus 5.5 judge invents to group a pole's subcircuits. A second pass looks first for context that could connect modes. Three repeat runs on 219 poles agreed on one mode versus several 96% of the time. No person graded anything. The authors say mode counts are not definitive.
- Exact shares (section Mode Statistics in Detail), out of 5,468 poles: 2,556 (47%) have a single mode. 2,673 have two or more, and in 1,231 of those (46%) the most frequent mode carries at least 90% of the damage from zeroing the pole. Together that is the 69%. Chart labels show 239 poles with no readable mode. Among multi-mode poles, 27% have half or more of their damage outside the main mode.
- Sampling: 5,000 firings per pole, 1,000 from each of five bands from the 97th percentile up. Damage is the change in the model's next-token predictions when the pole is zeroed, measured on 150 to 500 firings per pole. Gate-closed and leftover groups are excluded. The sample is balanced by band, not by real frequency. A footnote reweights and gives a mean of 1.56 and a median of 1.04 causally important modes, but no reweighted 69%.
- Telling modes apart (573 multi-mode poles; macro-F1, an accuracy-like score averaged over modes): inputs score 0.968, which the authors say is not an independent test; readers 0.853; a judge given the mode descriptions and the text snippet 0.707; chance 0.290. The main text calls the reader test independent. The Methods say readers were used when merging clusters.
- Reviewer: few applies to judge-drawn modes, not to 19 subcircuits. By a sum of chart labels, 1,025 poles have five or more modes. Five of the six worked subcircuits fire on a token beside a set phrase, such as haze after corneal.

### The four description tests

The main text says the tested descriptions come from an agent, the Neuron Scientist: a language model that reads 200 documents where the neuron fired, runs experiments, and revises over several rounds to raise a score. The Methods say the detection test used the mode descriptions from the Opus judge's first grouping pass. Scores are reported on documents the describer did not see. The four tests use different samples of poles (496, 573, 486 and 128), and the paper does not say how they overlap.

- **Detection** (496 poles). Claude Opus 5.5 sees the mode descriptions and two 69-token snippets with one token marked. In one the pole fires in its strong band. In the other it stays below its own 90th percentile. Share of pairs right, all bands: 0.946 against random text, 0.885 against text routed to the same expert. In the weakest band (97th to 99th percentile): 0.906 and 0.812. Comparator: chance, 0.50. The Abstract quotes only the easier figure, as 95%.
- **Mode classification.** Reported above: text 0.707 against chance 0.290.
- **Simulation** (486 poles; firings above the 99.9th percentile only). A reader model called Jev, which the paper calls a non-autoregressive decision model and does not otherwise explain, predicts the pole's activation at each token from the description. The predictions replace the real activations. Zeroing the pole is the floor. The ceiling, an oracle that knows the true activation, undoes 96% of the damage. The simulator reaches 73% of that ceiling, weighted by damage: 56% in the lowest band tested, 76% for the top 1,000 firings. A false-firing test on random text is described, with no result.
- **Input synthesis** (128 poles). Writers get only the description. A controller sees the real model's activations after each round and redirects them. For 120 poles (94%) the best text beat the pole's maximum in the corpus, the comparator. The median best text reached 1.06 times that maximum. There is no run with no description or a wrong one.
- Authors: text features explain about 14% of the variance in how strongly a neuron fires, against 59% for its gate inputs. About 15% of example firings stop when the model is rerun on the snippet without its document.
- Reviewer: no test runs a cheap rival. An embedding classifier appears once, inside an ensemble of text methods tried on 40 poles, with no score of its own. Every window is built around a known strong firing, so there is no false-alarm rate on ordinary text. The one reported example of steering (pushing a unit up while the model writes) does not do what a plain reading of its label suggests: a unit that fires when the model writes Wait made the model settle on an answer too quickly when raised. The authors read it as a signal that doubt has already been raised.

### Random-direction baselines

Random directions went through the same clustering and mode detection as neurons, but not through the four description tests. The text claims three differences: sparsity, weight links to specific readers, and narrowing with strength. The numbers are in the hover labels of Figures 19 to 21 (appendix Random Baselines for Neuron Circuits).

- Sparsity. Text: "8 neurons recover as much function as 128-256 random directions". Figure data: keeping each token's top 8 neurons keeps 0.59 of a layer's effect on the output, averaged over four layers. Eight random directions inside one expert keep 0.13, and the figure's own label says about 114 are needed to match 8 neurons. The 128 to 256 range fits only the two baselines drawn in the residual stream (the model's running state between layers). Random directions across experts need about 500.
- Readability. Text: "Random directions remain heterogeneous at all activation ranges." Figure data: the median share of a unit's firings in its main readable mode, from the weakest band (top 5 to 1% of firings) to the strongest (top 0.01%). For 158 neurons: 0.32, 0.61, 0.87, 0.93. For 192 random directions inside an expert: 0.67, 0.66, 0.65, 0.73. For 128 random directions across a layer: 0.16, 0.19, 0.18, 0.18. For 48 in the residual stream: 0.23, 0.24, 0.32, 0.39.
- Reading: the LLM reader finds a main meaning in random directions inside an expert at every strength, so a readable label is no evidence by itself. The other two kinds score well below neurons at strong firing. Neurons differ by being sparser and by narrowing far more with strength.

### The Colorado neuron audit

The appendix A Colorado neuron with a rare Tornado mode examines neuron L55/N11703 of Qwen3-32B, the best-known counterexample to the essay. A critic on X had shown that it also fires on code using Tornado, a Python web framework. Cammarata, the essay's author, did the audit.

- By causal effect the neuron looks clean: about 98.8% of its effect falls on text about Colorado or other places.
- By count it does not. Tokens with no place nearby are 61% of the tokens where it fires and about 1.2% of its effect. Colorado text is 1.4% of those tokens and 56% of its effect.
- The answer depends on the corpus. In 100 million tokens of ordinary text, 98.5% of its 3,812 strong firings were on places. On a code corpus, 98% were on Tornado.
- Authors: "The Tornado association is causally real." Asked which state Tornado was developed in, the model puts 30% on Colorado and 0.7% on California, the right answer. The paper cites the critic for those two figures. Suppressing the neuron gives 1% and 4.7%.
- Over 827 predictions of a Colorado name, half of the push toward the answer comes from 155 neurons and 80% from 2,299.
- The authors suspect Qwen3-32B was initialised by copying layers of Qwen3-14B.
- Reviewer: one chosen neuron, judged by the authors after the fact, with no control neuron. An instrument that only reads activations would mostly read the 61%.

### Transcoder evidence

A transcoder is a learned dictionary of sparse features trained to imitate one MLP layer. The paper sorts every live feature (one that fired at least once in about 10 million positions) of 19 transcoders on eight layers.

- Copies of a single neuron are 4.1% to 51.6% of live features in the 15 dictionaries trained on 234M tokens. They are scarce at the ends of the network: 11.7% at layer 4 and 7.3% at layer 39, against 50.9% at layer 19, all three at the smaller dictionary width. At most 59.6% of a layer's neurons have a copy.
- Most of the rest are one neuron narrowed by context, or sparse sums of 2 to 64 neurons. The paper calls a sum a coalition when its neurons fire together.
- Authors: coalitions mostly mean what their neurons share (60% of two-neuron coalitions). They say the features support a picture in which "concepts without a dedicated neuron are represented through combinations of neurons", and that features spread thinly over many neurons are not ruled out.
- Reviewer: the paper does not say who judged the 60%, on how many features, or what chance would give. Copying neurons is the cheapest way to imitate a layer. Nothing is steered or zeroed.

### What was released, and on what terms

- The paper says it releases pages for the 2,734 neurons and labels for all 10 million poles. It calls those labels low-effort and gives no accuracy figure.
- Found: the public Neuron Atlas, a web app. Its index lists 3,244 neurons (6,488 poles), all but two of them in eight layers, and the app shows a fixed sample of 3,000 poles from 2,202 neurons. The one pole whose files were opened has a firing threshold, modes, examples and an agent-written page. No file of the 10 million labels was found.
- Terms: the atlas shows no licence. Goodfire's website terms allow personal, non-commercial use only and forbid automated copying. The paper names no release of the corpus, the labelling model, Jev or the agent code.
- The atlas has story-level labels, such as a beat boundary or a revelation scene. Several fire on a full stop, a comma or a pivot word.

### The paper's numbers against the essay's

- One-job share: the essay estimated roughly 90% of poles for Qwen3-32B. The paper gives 47% and 69% for another model on another test, and no figure for Qwen3-32B.
- Weak activations: the essay said they can be removed at almost no cost. The paper finds large drops on six of ten reasoning benchmarks until fine-tuning.
- Double negative: the essay said that for about 4% of neurons the top examples echo the other pole. The paper says the regime is about a third of all activations, a different quantity.
- The essay's near-miss test (new examples fired strongly 76% of the time, near misses 11%, on 183 neurons) is not repeated.
- Promised by the essay and absent: results for trillion-parameter models and more on GPT-2 XL.

### The announcement thread

Goodfire posted eight messages on X on 10 November. They say that neurons read best when they fire hardest, and that removing the weakest 92% has modest effects. They say "light fine-tuning fully recovers benchmark performance". They say poles have few roles, which are easy to find, and that descriptions for 1,500 neurons are released. The thread is looser than the paper's tables and methods in three places. Recovery is not full in Table 8: MMLU-Pro stays 2.9 points down. The paper's own main text also claims full restoration. The line about few roles skips the median of 19 subcircuits. And 1,500 matches neither the paper's 2,734 neurons nor the atlas's 3,000 poles.

### First reactions

Checked about 78 minutes after the announcement. There was no outside retest or replication, and one piece of technical pushback. Zhengxuan Wu, a neuron researcher thanked by the lead author, called it a good read but said the evidence about inputs and readers still looks correlational. Tigges replied that both were also collected with attribution methods, and that the paper describes a simpler, cheaper one. Cammarata told the Colorado critic that the team is not sure how the two meanings connect.

## What it does not show

- A neuron reading that beats word counts, sentence embeddings or an LLM reading the text at finding an idea in new text. Every test starts from the neuron and asks whether text predicts it.
- That a unit follows an idea when its usual words are absent. There is no rewording test.
- The same picture in dense 8B models at 4-bit precision (weights stored coarsely), which the Batman study used.
- Carry-over between models. No unit is matched.
- Support for a document-level reading, such as beating 68 of 69 reference films. All tests are per token. Detection, mode classification and simulation score windows of 58 to 69 tokens.
- That cheap labels work. The tested descriptions come from multi-round agents or from the Opus judge's mode pass; the released labels are unscored.
- Human agreement with any label or mode.
- That a pole's activation alone says which meaning fired.
- What a unit does below the 97th percentile or with the gate closed. The authors report a band of weaker activations that cannot be removed and whose function is only sometimes apparent.

## What this means for P1 and P2

**P1.** Three claims need separating. (a) A neuron's strong firings have one or a few describable roles. This is now supported for one 35B model, by LLM grading. (b) An idea lives in one neuron. The authors do not claim this: they say concepts without a dedicated neuron are carried by combinations of neurons, and their own audit needs 155 neurons for half of the push toward one word. (c) A neuron reading measures an idea and not a word, and beats cheap rivals on a whole document. The paper does not test this, and it is the claim the programme needs. The study's finding that word counts matched or beat neurons on its primary model stands unanswered.

The paper also changes how the unit must be read. For the 31% of poles without a dominant mode (by subtraction), knowing that the pole fired does not say which idea fired. A meme has to be a pole plus a mode, read in a stated band with the gate open. The study's presence rule checks none of these.

**P2.** Nothing is stronger. The paper compares no two models. Its bands depend on expert routing, which dense models lack. If Qwen models are initialised by copying layers, agreement inside the Qwen family may be inheritance and not convergence. And if an idea needs sets of neurons, a map between models must carry sets or directions.

## What it changes for the instrument and the designs

Pipeline changes to make before any new run:

- Calibrate each unit on about 10 million tokens of general text and store its own percentiles. These replace the bar set by 69 reference films.
- Record the sign of the gate and count gate-open tokens only.
- Call a unit present when a token reaches its strong band, and rest single-label claims on the 99.9th percentile.
- Score a work by in-band tokens per 1,000 tokens, not by the mean of its top three sentences.
- Keep the firing token and its context, and have a blind judge say whether it matches the label.
- Feed whole documents, or put a neutral lead-in of about 250 tokens before constructed sentences.
- Run random directions through the same pipeline, band by band.
- Add one cheap intervention: zero the unit at its firing token and compare the effect with random units.

Model choice: keep Qwen3-8B and Llama 3.1 8B, the study's pair from two different companies. Qwen3.6-35B-A3B could be added as a P1-only arm. It needs new extraction code and permission to use the atlas.

Rules: sharpen the roadmap's rule 7 to say a unit is read only in its own strong range, set on general text. Add three more. A label is a guess until it predicts held-out firings better than a word-overlap picker. Every presence call says which mode fired. No threshold or label carries to another model, precision or kind of text unmeasured.

Amendments to registered designs:

- IN-21: read each sentence with and without a lead-in; count a win only in the strong band.
- IN-22: test coherence in three bands, and add the paper's top-8 sparsity test.
- IN-23: make the percentile band the primary definition of a firing.
- IN-24: add sets of 2 to 8 poles that must all be in band.
- IN-27: make the false-firing rate on filler text the headline.
- IN-29: compare thresholds, in-band tokens and gate signs across precisions.
- AT-22 and EN-27: add larger arms, since 155 neurons carry half of one name.
- EN-21: score steered text for any consistent effect.

## Experiment seeds


These are first sketches. Four became registered designs after a hostile review: IN-31 (synthesis under a word ban), IN-32 (sparsity of the dense 8B models), IN-33 (labels against a token table) and EN-28 (rewriting by reading). The copied-layers idea is seed UN-S12. The reviewed designs changed several predictions and thresholds, and one prediction was reversed. Where a sketch and a design differ, the page in `../designs/` decides.

1. **Synthesis under a word ban.** Question: with a unit's usual words banned, can an agent with activation feedback write text that blind judges accept as the idea and that fires the unit strongly? Rival: the same search on random units given the same label. Refuted if fewer than 18 of 60 concept units fixed in advance pass, or random units pass within 20 points of that rate.
2. **Is a dense 8B model sparse in neurons?** Question: how small a share of activations can Qwen3-8B and Llama 3.1 8B keep per token before loss rises 5%? Rival: the same share kept among random mixes of the same neurons. Refuted if no kept share of 10% or less holds the rise under 5%, or the random mixes do as well.
3. **Labels against a token table.** Question: does a unit's label predict where it fires on unseen film text better than a table of each token's average activation? Rivals: that table, another unit's label, no label. Refuted if the label wins by at least 0.05 in correlation for fewer than 25 of the 125 labelled units.
4. **Copied layers.** Question: do the study's models hold neurons that copy the same-index neuron a fixed number of layers away? Rival: different-index pairs at the same gap. Refuted if same-index pairs have closely matching input weights (cosine above 0.3) less than twice as often as those.
5. **Rewriting by reading.** Question: can a writer that sees only a reader model's unit activations move a story toward meme A and away from meme B? Rival: a plain prompt. Refuted if the loop wins by less than 0.05 on a 0-to-1 score from blind judges on 160 held-out stories.

## Key sources

| Citation | Link | Why it matters | Priority |
|---|---|---|---|
| Tigges, Kowal, McGrath, Lewis, Cammarata 2026, Neurons as an Interpretable Basis | https://www.goodfire.com/research/neurons-as-an-interpretable-basis | The paper. Read Tables 3 and 8, the Methods and the figure hover data | A |
| Goodfire 2026, Neuron Atlas | https://www.goodfire.com/research/public_neuron_atlas | 3,000 poles of the 35B model; no licence shown | A |
| Cammarata 2026, Neural Networks Are Already Clean | https://already-clean.goodfire.com/ | The essay behind P1; its 90% is not repeated | A |
| Arora, Wu, Steinhardt, Schwettmann 2026, Language Model Circuits Are Sparse in the Neuron Basis | https://arxiv.org/abs/2601.22594 | Foundation of the sparsity claim; supports sets of neurons | A |
| Shabalin, Piotrowski 2026, Comparing Sparsified MLPs and Transcoders | https://arxiv.org/abs/2610.24521 | Source of the fine-tuning recipe and the neuron-copy finding | A |
| Dravid, Bahri, Efros, Gandelsman 2026, Neuron Populations Exhibit Divergent Selectivity with Scale | https://arxiv.org/abs/2606.03990 | Shared neurons are a shrinking share as models grow; bears on P2 | A |
| Korznikov et al. 2026, Sanity Checks for Sparse Autoencoders | https://arxiv.org/abs/2602.14111 | Random sparse-autoencoder features score almost as readable as trained ones | A |
| Voita, Ferrando, Nalmpantis 2024, Neurons in Large Language Models: Dead, N-gram, Positional | https://aclanthology.org/2024.findings-acl.75/ | Many neurons follow tokens and positions | A |
| Brangus 2026, post on X | https://x.com/RatOrthodox/status/2110976835089809678 | The Colorado and Tornado counterexample | B |
| Wu and Tigges 2026, exchange on X | https://x.com/ZhengxuanZenWu/status/2120227051236143514 | The only technical pushback so far | B |
| Goodfire 2026, announcement thread on X | https://x.com/GoodfireAI/status/2120221477245182460 | Looser than the paper's tables; not a source for numbers | C |

## Fog

- The paper was read on the day it came out. It lists no journal or preprint number. Reactions were checked once, through a mirror of X.
- This note was written from a checked section-by-section digest and three assessments of it. Every number about the paper was matched against its text or the saved page. Atlas counts and reactions were not re-verified. Five figures load by script and could not be read.
- The atlas files do not reproduce 2,734 neurons, and the thread's 1,500 could not be confirmed.
- Whether Goodfire's website terms cover the atlas host is unknown. The precision of the model studied is not stated.
- The text reports SuperGPQA at 45.4 after fine-tuning. Table 8 puts the sparse model 0.2 points above 44.8.
- The Introduction says firings above the 99.9th percentile cause 73% of all damage from zeroing a neuron. Nothing else sources that figure. It is not the simulation result.
- Whether the paper's percentiles are taken only over tokens routed to a neuron's expert is an inference. If so, the same percentile is a looser bar in a dense model.
- Open: how often a pole has a second meaning that shows only on another kind of text, such as film synopses.
- The related papers exist; most were read at abstract level only.

## Check log

A second agent checked this note against the paper's saved text and page, the essay, the announcement posts, the saved atlas files and replies, and the roadmap on 10 November 2026. Nothing was run.

Changes made:

- Bottom line and Sparsity. Said reasoning benchmarks fall by up to 20.9 points. Now says six of the paper's ten reasoning benchmarks fall and four rise (HumanEval+ by 15.9). The comparison with the essay now says the same.
- Bottom line. Said chance is the only comparator and that no test uses sentence embeddings. Now says no cheap rival is run and detection is scored against chance only. Simulation and synthesis have other comparators, and one appendix variant puts an embedding classifier inside an ensemble.
- Bottom line. Said random directions are readable, giving only the figure for directions inside an expert. Now adds that the other two kinds of random direction score 0.16 to 0.39.
- Sparsity. Said the intact model was given the same recipe. Now adds that the imitation step was skipped for it, as the table notes say.
- Sparsity. Said the fine-tune added no facts. Now says it mainly corrected reasoning and did not add new information, which is the authors' wording.
- Sparsity. Said 320 neurons per layer still fire. Now says 320 routed neurons, and that the shared expert is left untouched.
- Sparsity. Gave the median of 519 neurons with no condition. Now says it holds when only the current token's neurons are removed, and that the count is about 1% of active neurons when earlier tokens are stripped too.
- Modes. Said a judge read only the text. Now says the judge had the mode descriptions and the snippet.
- Description tests. Said all descriptions come from the Neuron Scientist. Now says the main text says so, and the Methods say detection used the Opus judge's mode descriptions. The line on cheap labels under What it does not show was changed to match.
- Description tests. Said the neurons are the same throughout. Removed as unsupported. Now gives the four sample sizes and says the overlap is not stated.
- Simulation. Said the paper does not explain Jev. Now says the paper calls it a non-autoregressive decision model and says no more.
- Description tests, reviewer. Added that an embedding classifier appears once inside an ensemble, so that "no cheap rival" is exact.
- Description tests, reviewer. Said the steering example cuts against its label. Now says it does not do what a plain reading of the label suggests, and gives the authors' reading.
- Random baselines. Said the 128 to 256 range fits only the weaker baselines. Now says it fits the two residual-stream baselines, and that random directions across experts need about 500.
- Random baselines. Gave readability figures for one kind of random direction. Now gives all three kinds from the same figure. The reading under it was narrowed to directions inside an expert.
- Colorado. Presented the 30% and 0.7% figures as the authors' own. Now says the paper cites the critic for them.
- Colorado. Said the model was built by copying layers. Now says initialised, the paper's word.
- Transcoders. Now says the three layer figures are for the smaller dictionary width.
- Transcoders. Said most of the rest are one neuron narrowed by context or coalitions. Now says sparse sums of 2 to 64 neurons, called coalitions when the neurons fire together. In five of the fifteen dictionaries the first two classes alone are under half of the rest.
- Transcoders. Said the authors conclude that concepts are represented through combinations. Now says the features support a picture in which they are.
- Transcoders, reviewer. Said the 60% has no judge. The paper says the features were judged and does not say by whom. Reworded.
- Atlas. Said the index covers eight layers. A recount found ten: eight layers, plus one hand-added neuron in each of two others. Reworded.
- Atlas. Said each pole has a threshold, modes, examples and a page. Only one pole's files were opened. Now says so.
- Atlas. Said the corpus, the labelling model, Jev and the agent code are not released. Now says the paper names no release of them.
- Thread. Said the thread is looser than the paper on full recovery. The paper's main text makes the same claim. Now says the thread is looser than the paper's tables and methods.
- Reactions. Said there was no outside critique, then reported one. Now says no retest or replication, and one piece of technical pushback.
- Reactions. Said Tigges replied that inputs and readers were traced with a causal method. He said attribution methods, and that the paper describes a simpler one. Reworded.
- What it does not show. Said all tests use windows of 58 to 69 tokens. Now limits that to detection, mode classification and simulation.
- What it does not show. Said a middle band has no clear function. Now says a band of weaker activations whose function is only sometimes apparent.
- P1. Said the authors concede that an idea does not live in one neuron. Now says they do not claim it, and gives what they say.
- P1. Now says word counts matched or beat neurons on the study's primary model, as the roadmap states.
- Model choice. Said Qwen3-8B and Llama 3.1 8B are the only cross-model pair. The study also compared Qwen3-8B with Qwen3-1.7B. Now says the pair from two different companies.
- Key sources. The Korznikov row now says sparse-autoencoder features. The thread row now says looser than the paper's tables.

Confirmed without change: every other number in the Findings, all quotations, the essay figures, the eight posts, the mode counts and the 1,025 and 239 sums from the chart labels, the atlas totals of 3,244 neurons and 6,488 poles, the 3,000-pole sample of 2,202 neurons (recomputed from the app's own sampling code), the website terms, and the roadmap facts (rule 7, the named designs, the 125 labelled units, 68 of 69 reference films).

Could not check:

- The critic's own post as saved does not show the 30% and 0.7% figures. Only the paper's citation links them to him.
- The reactions were read in the saved copies. The posts were not fetched again, and the search for outside retests was not repeated.
- The story-level atlas labels and the search for a file of 10 million labels were checked against the saved atlas notes only.
- The related papers in Key sources were checked against saved reference notes, not against the papers.
- The five figures that load by script (Figures 2, 7, 8, 10 and 17) are still unread.
- Statements that the paper does not state something rest on a search of its text.
- The proposals under What it changes and Experiment seeds are recommendations. Only the roadmap facts they cite were checked.
- How this note was made, and the first two Fog lines, describe the writing process and were taken as given.
