# l15-interpretability-survey: what the field's survey of November 2026 says about units, tools and tests, and what it leaves the programme to prove

Source: Barez, Wu and 25 co-authors, among them Yoshua Bengio, Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety, arXiv:2611.08438, first posted 5 November 2026 (version 2, 9 November), https://arxiv.org/abs/2611.08438 . Mechanistic interpretability studies what a neural network computes by reading its internal numbers and changing them. The survey is about AI safety. This note covers only what bears on the programme.

How this note was made: AI readers read the survey section by section, and a second agent checked each extraction against the text. Nothing was run: no model, no pipeline, no experiment.

## Bottom line

- **Learned features, not neurons, are the survey's default unit, on thin evidence.** It reports that single neurons often respond to several unrelated concepts, and that the field has increasingly shifted to features: structures in a model's activations (its internal numbers), such as a direction, that stand for a concept. The usual tool is a sparse autoencoder (SAE), a small network trained to rewrite activations as a few active entries from a learned dictionary. It gives no rate and no head-to-head number.
- **It concedes two points to the neuron side.** Superposition, the idea that a model packs in more features than it has dimensions, "remains a hypothesis, not an established fact, in large language models". And the model's own arithmetic singles out the hidden neurons of a feed-forward (MLP) block.
- **On models converging, it calls shared algorithms supported and shared representations less supported.** It says nothing about the same neuron recurring in two models.
- **The feature tools are unstable by its own account.** "Often, fewer than 50% of features remain the same across seeds", where a seed is the random starting point of a training run.
- **Any claim about a unit needs a causal test against baselines.** Its framework compares two of its four causal tests, necessity and sufficiency, with random units of matched size. Individual neurons appear in it as a baseline.
- **Machine-written labels are hypotheses, and steering (pushing a model's activations along a direction) is shown for directions, not neurons.** The survey reports that plain prompting often beats methods that change a model's internals.
- **Nothing in it takes text or culture as the thing measured, or traces one work to another.** Training-data attribution links a model's output to that model's own training set, with no accuracy figure.

## Findings

### What kind of source this is

- The survey's object is the model. It counts an analysis as mechanistic only if it works on internals and tests cause by intervention, such as ablation (switching a unit off), patching (swapping in an activation from another run) or steering.
- Reading without intervening is descriptive in its terms. That is where the Batman study sits. The survey says descriptive methods often supply the targets for intervention.
- Most statements used here are reports of other groups' results, usually without numbers. The survey's own contributions are definitions, an evaluation framework, recommendations and open problems.

### Single neurons, polysemanticity, superposition and the privileged basis

- **Polysemanticity** means one neuron answering to several unrelated things. The survey reports it as found by follow-up studies (section Neuron Polysemanticity in Model Representations; one citation, a paper on toy models) and as repeatedly observed and widespread (section The Superposition Hypothesis; three citations). Neither gives a rate or a model size.
- **The survey's own conclusion is hedged.** Single neurons may not be the most effective unit in many cases. The field has increasingly shifted to features: structures in activation space, such as a direction, that stand for a human-readable concept. The survey does not commit to a form.
- **When a neuron stands for a concept.** In the survey's example the neuron fires on the word blue, stays near zero otherwise, and shifts the output towards or away from blue-related content when manipulated.
- **Privileged basis.** In the hidden layer of an MLP block the nonlinear step acts on each neuron separately, so the neuron axes are meaningful coordinates. There a single neuron "can serve as a scalar feature that meaningfully represents a concept". The residual stream, the running vector every layer reads and adds to, has no such basis. The argument is architectural, with no measurement. Whether it holds for the gated variant of the MLP that Llama 3 uses is not stated.
- **Not a promise of clean neurons.** The survey's superposition figure says that when features outnumber dimensions, each neuron axis cuts across several features. So the basis supports reading MLP neurons directly, not treating each as one concept. That step is this note's inference.
- **One sentence for the neuron side.** A boxed description of the MLP block (section Transformer Computation) says its wide hidden space has recently been found to give more interpretable representations natively (Nikankin et al. 2025; Arora et al. 2026). It names no comparator and gives no number.
- **Superposition is still a hypothesis.** The survey says so flatly, as its own caution. It reports Lecomte et al. 2024: mixed neurons can arise even when features do not outnumber neurons.
- **The summary is firmer than the evidence.** The section Core Hypotheses in Mechanistic Interpretability closes by giving superposition as the reason to work with features instead of single neurons, then says each hypothesis holds in some settings and fails in others. A reviewer would say the default unit rests on a hypothesis the survey declines to call established.
- **Features against neurons.** The survey states twice that SAE features are more single-meaning than individual neurons. One place says many features, not all. The other is a table that files the result under small models. Neither gives a number.
- **MLP layers.** For the layers where the study's units live, the survey calls transcoders (dictionaries trained to predict an MLP's output from its input) essential. It does not compare them with raw neurons.

### Universality across models, level by level

- Universality is the claim that different models learn the same things. The survey files it as a hypothesis, not yet rigorously proven, and splits it in two. It cites no source for the split.
- **Same algorithms (weak): supported,** by attention patterns, features and circuits (sets of connected components behind a behaviour) that recur across models, tasks and scales. Four works are cited, with no rate. By their titles, one concerns addition and subtraction and one a training effect called grokking. Two are co-authored by the survey's first author.
- **Same representations (strong): less supported.** It is usually shown by similarity scores between two models' activations, where "recent studies have found the results to be fragile". Representations can diverge with architecture, data or the path training takes.
- **Which holds: not said.** If only the weak form does, even a complete understanding of one model may not carry to others.
- **Individual neurons: not addressed.** The reference list has neither Gurnee et al. 2024 nor Dravid et al. 2026, the roadmap's two sources for that level.
- **Inside one model.** The section Feature Formation, Consolidation, and Drift reports that the components carrying an algorithm can change identity while the algorithm persists (Tigges et al. 2024). A feature found at one saved stage of training has no automatic counterpart at another. Features for surface statistics form first, semantic ones later.

### How reliable sparse autoencoders and related tools are

- **Verdict,** in the survey's voice: SAEs are a standard tool with systematic failure modes that undermine the claims built on them. Its table of conditions says most evidence on SAE variants comes from Llama and Gemma models.
- **Seeds.** The section Feature Instability Across Seeds cites Paulo and Belrose for different seeds giving substantially different dictionaries. The 50% sentence has no citation, and no rule for judging two features the same.
- **Absorption and splitting.** A general feature can stay silent where a specific one fires: a feature for words starting with S may not fire on the word short. In a larger dictionary a feature can split, sometimes into plain word detectors.
- **Identifiability.** It is unknown whether a dictionary recovers real features or one of many decompositions. The survey recommends framing SAE features as "one valid approximate decomposition rather than the true features".
- **Rivals.** When the concept is known in advance and labelled data exists, supervised methods such as probes (small classifiers trained on activations) often beat SAEs at detection and steering. Three citations, no numbers.
- **Random baselines.** The survey asks for them and reports no SAE result from one. A study whose title says interpretability metrics do not tell trained transformers from random ones (Heap et al.) is cited once, in passing. No result compares neurons, features and random units on one test.
- **Replacement models.** A cross-layer transcoder is a set of dictionaries trained together to stand in for all the MLPs. When it fully replaces a model's activations, the top prediction matches the original's on only about 50% of prompts from a standard corpus (one study, an 18-layer model).
- **Proxy fidelity.** The section The Proxy Fidelity Problem warns that "a proxy model may find features and circuits that do not exist in the original model". Not finding a mechanism in the proxy does not show it is absent. This is the survey's argument, with no measured rate.

### The evaluation framework, and what counts as causal verification

The section A Unified Evaluation Framework for Mechanistic Interpretability is the survey's own proposal. It applies to any unit, neurons included.

- **Five dimensions, all required.** Attribution faithfulness (do a method's scores match the measured effect of intervening?), completeness (how much of the behaviour is explained?), stability (across seeds and reworded inputs), causal validity, and human interpretability (for example, do people shown an explanation predict the model better than people shown none?).
- **Four causal tests.** Necessity: ablate the unit and measure the damage. Sufficiency: keep only the unit. Both are compared with random component sets of matched size. Interchange: swap the unit's activation between two inputs and see whether the output follows. Specificity: check that other behaviours are unaffected.
- **Baselines first.** The recommended protocol sets baselines before the method is applied, such as random directions, principal components (the directions along which activations vary most), individual neurons and supervised probes. Failure cases are to be reported.
- **Steering as a test.** A consistent change when pushing along a direction is evidence of influence. The survey says more than once that it is not a full account of the mechanism.
- **Probes** show that information is present, not that the model uses it. The cited remedy is a control task: train the same probe on random labels and report the gap. Unchanged behaviour after erasing a direction does not show the information is unused.
- **Pitfalls** (section Epistemic Risks). A clean human label does not show the model uses the feature. Reports should state what share was examined, and say not found, not absent.
- **Limits.** The survey calls the framework a starting point with uncalibrated thresholds. It says no accepted standard exists.

### Automated labelling and its failure modes

- **The recipe.** In the section Automated Interpretability, one language model reads texts that strongly activate a unit and writes a guess at its function. A second predicts the unit's activations on new text from the guess. The score is the correlation with the real ones. No pass mark is stated.
- **Status.** The pipeline box says a high score confirms the guess. The guardrails that follow call automated output weak evidence until it passes human or causal checks. A reviewer would take the weaker reading.
- **Top examples.** The survey's example of an explanation is a summary of a feature's highly activating samples, which is how the Batman study labelled units. It does not ask whether top samples represent the rest of a unit's firing.
- **Failure modes.** One explainer can spread its errors to every unit it labels (uncited). A verbaliser, a model trained to put an activation into words, may reconstruct the input text and explain that. The survey reports that "such failure modes are severe and prevalent in existing verbalisers" (two 2026 studies, no figures).
- **Guardrails,** proposed and untested: blind human review of a pre-set random share; relabelling with other models; a causal test on a share; tests on units with known answers; an error rate with every finding.

### Steering, representation engineering, erasure and editing

- **Works.** Adding a direction to the residual stream is said to change refusal, helpfulness and style reliably. No effect size is given.
- **Fails.** Steering vectors often degrade on tasks different from the data they were built on (Tan et al. 2024). A vector that stops working raises no error.
- **Prompting** is called often better than methods that change internals (one citation, Wu et al. 2025).
- **Erasure.** Pruning the neurons tied to a concept is called often brittle: the concept is suppressed and can return (one citation, co-authored by the survey's first author).
- **Editing.** Early methods for editing a fact in a model's weights (its trained numbers) caused substantial side effects. Newer ones are said to have largely reduced them, with no figure. The worked example changes the output matrix of one MLP layer, not a single neuron.
- **Reading against writing.** One direction is treated as both the reading and the writing handle, as a hypothesis. No test of whether the best reader is the best writer is reported. The table of safety uses names no single neurons.

### Comparing models: model diffing and crosscoders

- Model diffing means comparing the internals of two models. For different architectures the survey names crosscoders: one sparse dictionary trained on two models at once, with an encoder for each. No success figure is given.
- **False positives.** A boxed warning in the section Model Diffing says such dictionaries can report a feature as exclusive to one model when it is not. The survey concludes such features are hypotheses. Finding a feature in one model and not the other shows neither use nor absence.
- **A cheap rival.** In one cited study a language model read pairs of outputs from two models and listed the differences. It matched the SAE-based baseline on many dimensions. No numbers are given.

### Training-data attribution and tracing influence between texts

- Training-data attribution looks for the training samples that shaped a model's behaviour. The survey says it is not mechanistic by its own criteria. The section Training Data Attribution reports no accuracy figure and no benchmark.
- **Exact-span matching** searches trillions of training tokens (word pieces) for strings in the output. The survey calls its main example more a retrieval tool than an attribution method, and says the matches do not explain composed output such as translation or poetry.
- **Gradient methods** estimate how much one training sample changes a prediction. Influence functions in their original form are said to perform poorly in practice. One later method is called comparable in quality to a costlier method that retrains many models, on its own authors' word.
- **Unsolved, by the survey's list:** compounding approximation error, results that change between training runs, and cost. Whether such methods "can reliably identify the relevant data" is listed as open.
- **Evidence and units.** Controlled manipulation of training data ranks as the strongest evidence for what shaped a model. One 2026 method attributes data at the level of SAE features, with no result reported.
- **For the programme.** Every method links a model's output to that model's own training set, and needs the corpus. None traces one work to another. The family offers a word-level rival (longest shared span), a ground truth (randomly assigned training sets) and an untested adaptation: score a source by how much learning from it would change a model's error on the later text.

### Uses of model internals to measure text, data or culture

- None as such. The words culture and memetic do not occur. Almost every use of internals is aimed at the model. Two passing cases are aimed at the task a model was trained on: what a chess or Go model has learned, and physical equations read off a network built to be readable. None is aimed at text.
- The nearest neighbours are tools: a direction for a character trait, built from a plain-language description and used to score training data; a filter that labels training tokens with SAE features; monitors and probing classifiers that read a property of the input text off activations.

### Open problems the survey shares with the programme

- Whether the units found are real or a product of the method.
- Whether a label is the model's category or the labeller's. The section Interpretation vs. Truth: The Ontology Question says naming a feature reflects possibly subjective human choices.
- No accepted standard of evidence.
- Whether results carry to other models and kinds of input.
- Whether methods that use internals beat methods that do not. The survey says it discusses whether safety tasks can be done just as well without interpretability methods.

### Where it agrees and disagrees with Goodfire's paper

Neither cites the other. Goodfire's paper, Neurons as an Interpretable Basis (10 November 2026), studies one model, Qwen3.6-35B-A3B, which sends each token to a few of many small MLPs called experts.

- **Agree: superposition is unproven.** Goodfire calls it a proposal and does not test it.
- **Agree: the MLP neuron basis is special.** The survey argues from architecture. Goodfire measures: keeping each token's top 8 neurons keeps 0.59 of a layer's effect on the output, against 0.13 for 8 random directions inside the same expert (averaged over four layers).
- **Agree: a dictionary is a proxy.** In Goodfire's own transcoders, copies of a single neuron are 4.1% to 51.6% of live features across the 15 dictionaries trained on 234M tokens.
- **Disagree: whether a transformation is needed.** The section Feature Extraction and Analysis says its methods share the premise that raw representations are too tangled to read. Goodfire reads raw neurons.
- **Talk past each other: how mixed neurons are.** The survey says often, with no number. Goodfire reports that 69% of 5,468 neuron poles have one role or one dominant role. A pole is one side, positive or negative, of a neuron. The count covers only each pole's strongest 3% of firings, as grouped by a language-model judge.
- **Disagree: what verifies a label.** The survey asks for interventions against random units, and for human raters. Goodfire tests prediction: a judge given the description picks the firing snippet 0.946 of the time against random text, where chance is 0.50 (496 poles). It reports no human rater.
- **Cannot referee.** Goodfire restricts its claim to strong firings. The survey never asks whether a neuron's meaning depends on how strongly it fires.

## What it does not show

- A rate of mixed neurons, or raw neurons compared with learned features on one test.
- Any result on neurons in models of the study's size, anything on 4-bit weights, or anything on Qwen beyond a citation of its technical report.
- Whether the same neuron or content unit recurs in two models, or a validated map between them.
- That any unit follows an idea when its usual words are absent.
- A reading of a text, a corpus or culture as the thing studied, by any internal unit.
- A tracer of influence from one work to another.
- Effect sizes for steering or ablating single neurons.

## What this means for P1 and P2

**P1.** The survey neither overturns the neuron as a readout nor endorses it as a unit of meaning. The standard reason for abandoning neurons is weaker than usually stated: superposition is unproven in large models, and MLP neurons are a privileged basis. The alternative is weaker too: dictionary features change with the seed and may not exist in the model. That backs the raw neuron's one stated advantage, a fixed address two labs can cite. And the bar is the field's, not only that of the Goodfire essay behind P1: a unit stands for a concept only after a causal test against random units. The Batman study's labels, read from eight top sentences, were put through none of the survey's causal or human checks. The claim the programme needs, that a neuron reading follows an idea and beats cheap rivals, is outside the survey's scope. One table cell in the roadmap's premises review stated superposition as a fact. It was softened on 10 November.

**P2.** The survey's split cuts against the form of P2 the programme needs. The supported level is shared algorithms, not shared content units. Shared representations is the level it calls less supported. On single neurons it is silent, so the roadmap's base rates stand. Two cautions are new: units change identity along one model's training, and a shared dictionary can invent differences between models. The roadmap's table of levels gained a row for algorithms on 10 November, marked as not carrying a unit-level reading.

## What it changes for the instrument and the designs

Rules:

- Rule 1 (cheap rivals): add a linear probe on the same activations, principal components and random directions. Report each chosen unit's score on unseen data minus its score with shuffled labels.
- Rule 7 (do not assume clean neurons): a unit is causally checked only when silencing it, pushing it and swapping its activation between matched passages each beat random units matched on layer and firing rate.
- Rule 8 (a human in the loop): a machine label is a hypothesis until it beats a judge given another unit's label, passes a blind human check on a pre-set sample, and carries an error rate from units with known answers.
- New rule: every result states what share of units, works and activation range was examined.

Gaps in the validation ladder: no rung asks whether reading strength predicts the effect of pushing a unit, or what share of a result the named units carry. Before this note, no design had an interchange test.

Amendments proposed to registered designs:

- EN-21 (steer the named memes): add a difference-of-means direction per label (the average activation with the label minus the average without) as a positive control. The survey's working handles are directions, so a fail on neurons alone is ambiguous.
- EN-22 (rewriting against prompting): expect prompting to win. Test on a second source of stories.
- EN-23 (do reading units write?): report the rank correlation between detection score and steering effect over all screened units.
- EN-26 (knock out a motif): add a relearning test, since the stated failure is recovery later.
- EN-27 (is the unit needed?): add the interchange arm.
- IN-24 (neuron, direction or feature?): run the dictionary arm at two sizes. One dictionary is one draw.
- LI-21 and LI-24 (lineage): add longest shared word span as a rival.
- UN-23 (carrying a reading between models): treat units a joint dictionary calls exclusive as hypotheses.

## Experiment seeds


These are first sketches. Three became registered designs after a hostile review: IN-34 (the labeller's error rate on units with known answers), IN-35 (coverage of the heredity result by named units) and LI-27 (gradient similarity as a descent tracer). The others are seeds in `../IDEAS.md`. Where a sketch and a design differ, the page in `../designs/` decides.

1. **Gradient similarity as a descent tracer.** Question: on texts with a known parent, does the similarity of two texts' gradients (the change each would make to a model's weights if learned) name the parent better than text-only methods? Rival: the drop in the model's surprise when the candidate is placed in its context. Refuted if it ranks the true parent first less than 5 percentage points more often than the best rival.
2. **Labeller error rate on known answers.** Question: run on made-up units whose function is fixed by construction (a keyword rule, text length), how often does the labelling step name the function? Rival: the top words of the unit's best examples. Refuted if fewer than 70% of 200 units get a label a blind human accepts.
3. **Influence absorbed in training.** Question: when a small model is trained further on a random subset of obscure tales, can any reader name those tales from its stories? Rival: a language model given the stories and the candidates. Refuted if neuron units score an AUC under 0.60 (a detection score where 0.5 is chance), or within 0.03 of the best text-only rival.
4. **A meme reader built from its definition.** Question: does a direction built from a motif's written definition detect the motif on words-controlled text better than a frozen neuron? Rival: a language model reading with the same definition. Refuted if AUC is under 0.70.

## Key sources

| Citation | Link | Why it matters | Priority |
|---|---|---|---|
| Barez, Wu and 25 others 2026, Understanding AI from the Inside | https://arxiv.org/abs/2611.08438 | The survey; many boxes are in the HTML version only | A |
| Paulo, Belrose 2025, Sparse Autoencoders Trained on the Same Data Learn Different Features | https://arxiv.org/abs/2501.16615 | Source for seed instability | A |
| Arora et al. 2026, Language Model Circuits Are Sparse in the Neuron Basis | https://arxiv.org/abs/2601.22594 | A neuron-level work cited for the MLP hidden layer being natively more interpretable, and for circuits found at the level of neurons | A |
| Wu et al. 2025, AxBench | https://arxiv.org/abs/2501.17148 | Prompting beats internal handles; probes beat SAEs | A |
| Jiralerspong, Bricken 2026, Cross-Architecture Model Diffing with Crosscoders | https://arxiv.org/abs/2602.11729 | By its abstract, covers Qwen3-8B and Llama 3.1 8B models | A |
| Gröger, Wen, Brbić 2026, Revisiting the Platonic Representation Hypothesis | https://arxiv.org/abs/2602.14486 | One of two works the survey cites for similarity-score results being fragile | A |
| Lecomte et al. 2024, What Causes Polysemanticity? | https://arxiv.org/abs/2312.03096 | Mixed neurons without superposition | B |
| Minder, Dumas et al. 2025, Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning | https://arxiv.org/abs/2504.02922 | Source for falsely exclusive features | B |
| Kempf et al. 2026, Simple LLM Baselines are Competitive for Model Diffing | https://arxiv.org/abs/2602.10371 | A text-only rival matched the SAE method on many dimensions | B |
| Li et al. 2025, Do Activation Verbalization Methods Convey Privileged Information? | https://arxiv.org/abs/2509.13316 | Worded readings can be the input restated | B |
| Chen et al. 2025, Persona Vectors | https://arxiv.org/abs/2507.21509 | A direction from a description, used to score data | B |
| Kowal et al. 2026, Concept Influence | https://arxiv.org/abs/2602.14869 | Attribution at the level of SAE features | B |

Years and author order follow the arXiv postings. The survey cites later or earlier versions of four of these: Paulo and Belrose as 2026, Li et al. as 2026, Jiralerspong and Bricken as 2025, and the crosscoder paper as Dumas et al. 2025.

## Fog

- This note was written from a checked section-by-section digest and two assessments made from it. The eight quotations were matched again against the text. The key sources with arXiv links were confirmed by title and authors only.
- The plain-text copy drops the survey's boxes. A rough recount on 10 November found 95 in the HTML version, five of them a legend in an appendix, and could match none of them in the plain text. Those used here were read from the HTML version. Others could have been missed.
- The survey colour-codes citations by peer-review status. That coding was lost, so this note makes no claim about the review status of any cited work.
- The 50% figure for seeds has no citation. The roadmap quotes 30% from the cited paper for one setting.
- The announcing post says 182 pages. The saved PDF of version 2 has 183. The survey's code notebooks (https://github.com/TSGLAB/Interpretability_Survey_Safety) were not opened.
- The first authors of Concept Influence (Kowal) and of the study of circuits across training (Tigges) share surnames with authors of Goodfire's paper. Whether they are the same people was not checked.
- Open: whether the privileged-basis argument holds for gated MLPs such as the one in Llama 3, and how often an MLP neuron is mixed at 8B.

## Check log

A second agent checked this note against the survey's text on 10 November 2026. Claims about this repository were checked against the roadmap's premises review and README, and the figures from Goodfire's paper against a saved copy of that paper. About 245 factual statements were checked and 33 edits made. The eight quotations are exact.

Changes:

- Bottom line. Said the field "has moved to features: directions". Now says it has increasingly shifted to features, which are structures such as a direction. The survey does not fix a form.
- Bottom line and P2. Said the survey "doubts" shared representations. Now says it calls them less supported, which is the survey's wording.
- Bottom line. Said the framework compares "the main causal tests" with random units. Now says two of its four, necessity and sufficiency.
- Bottom line. Said nothing in the survey measures text or culture. Now says nothing takes text or culture as the thing measured, since its probes and data filters do read properties of text.
- What kind of source. Gave ablation, patching and steering as the full list of interventions. Now "such as", as in the survey.
- Features against neurons. Added that one of the two statements says many features, not all.
- Universality. Said the split into two levels "is its own". Now says the survey cites no source for it.
- Universality. Said "internal patterns" recur across models. Now "attention patterns", across models, tasks and scales, as in the survey.
- Rivals. Said supervised probes often beat SAEs. Now supervised methods such as probes, when labelled data exists.
- Random baselines. Said the survey reports no result from a random baseline. Now "no SAE result", because its section on probes reports one result on a randomly initialised network.
- Random baselines. Described Heap et al. as the study that ran SAEs on randomly initialised models. The survey does not describe the study. Now described by its title.
- Replacement models. Said a cross-layer transcoder is one dictionary. Now a set of dictionaries trained together.
- Evaluation framework. Human interpretability was defined by one test. Now given as an example. The survey lists four tasks.
- Baselines first. Said the protocol fixes four baselines and that failures must be reported. Now says the protocol is recommended, the four are examples, and failure cases are to be reported.
- Editing. Said the worked example edits one whole MLP layer. Now says it changes that layer's output matrix.
- Model diffing. A heading and a table cell said a cheap rival "tied" the SAE method. Now says it matched it on many dimensions, which is the survey's wording.
- Exact-span matching. Said the survey calls it a retrieval tool that cannot explain composed output. Now says the survey calls its main example more a retrieval tool than an attribution method, and that the matches do not explain composed output.
- Gradient methods. Said one method is comparable to expensive retraining. Now comparable in quality to a costlier method that retrains many models.
- Uses. Said every use of internals is aimed at the model. Now almost every use, with two passing cases aimed at a board game and at physical equations. None is aimed at text.
- Open problems. Said the survey judges methods partly on whether internals beat other methods. Now says it discusses whether safety tasks can be done just as well without interpretability methods.
- What it does not show. Said the survey has nothing on models of the study's size or on Qwen. It names public SAE weights for Llama 3.1 8B and cites the Qwen3 technical report. Now limited to results on neurons at that size, 4-bit weights, and Qwen beyond that citation.
- What it does not show. "A reading of a text, a corpus or culture" now adds "as the thing studied".
- P1. Said the Batman study's labels pass none of the survey's checks. Now says they were put through none of its causal or human checks, which is what the premises review supports.
- P1 and P2. Said a premises table cell states superposition as a fact, and that the three-level table should gain a row. Both changes were already in the premises review on 10 November. Now in the past tense.
- Validation ladder. Said no design has an interchange test. EN-27 gained one on 10 November. Now "before this note".
- Designs. "Amendments to registered designs" now reads "Amendments proposed to registered designs".
- Key sources. Arora et al. was called the one neuron-level work cited in favour of neurons. Nikankin et al. is cited in the same sentence. Now described by what it is cited for.
- Key sources. The Gröger row said "doubt" and now uses the survey's "fragile". The crosscoder paper's title is given in full. A line under the table says where the survey's years and author order differ from the arXiv postings.
- Fog. Said about 83 of 94 boxes are missing from the plain-text copy. A recount found 95 boxes and could match none of them in the plain text.

Could not check:

- How the note was made: the line under the source and the first Fog bullet.
- Whether the split of universality into two levels is older than the survey.
- The papers behind the arXiv links. The links were compared with saved arXiv records for title, authors, date and abstract only.
- The survey's colour coding of citations by review status.
- Whether Kowal and Tigges are the same people as Goodfire's authors.
- The rules, proposed amendments and experiment seeds. They are proposals, with no source to check. Seven of the nine designs named carried a matching amendment in their design files on 10 November. Two did not: LI-24's amendment there is about dictionary size, not a shared-span rival, and UN-23 had none from the survey.
