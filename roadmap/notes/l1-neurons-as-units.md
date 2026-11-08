# l1-neurons-as-units: how true is P1 (a meme = whatever makes one neuron fire)?

Scope: Cammarata's "Neural Networks Are Already Clean" (read in full from the plain-text version, https://already-clean.goodfire.com/essay.md, fetched 2026-11-08), the responses to it, and the superposition / polysemanticity / SAE literature. No experiments run. Study numbers are quoted from the committed `study/paper/paper.md` (v1, frozen), which draws on `study/results/experiments.json` and `study/results/robustness/`.

## Bottom line

- **P1 as the paper states it is stronger than its source.** Cammarata claims "most neuron poles have one main job" (neuron -> job legibility). He does not claim one concept lives in one neuron, and he does not claim neurons enumerate a model's concepts. A meme unit needs the last two. His own examples use hubs of 4, 8 and 20 neurons for one concept.
- **The headline 90% is self-graded, preliminary and unreplicated.** It is for Qwen3-32B, from Goodfire's own agents, with "one job" allowing "related offshoots", and the technical paper is still unpublished (as of 2026-11-08). The only outside retest (IRSI, AI-agent-written, not peer reviewed) got 78% reading Goodfire's own write-ups and 57 to 65% with an independent dashboard labeler, against 31% for random directions. So: neurons are a better-than-random basis, and the true "one main job" share is somewhere between about 55% and 90% depending on who grades and how.
- **The study did not use the audited model or the audited method.** It used Qwen3-8B and Llama 3.1 8B at 4-bit; the essay says smaller models have broader neurons (GPT-2 XL redo: about 70% "roughly one thing", about 5% pristine). The study labels units from their eight top sentences within a 3,700-sentence corpus; the essay says top-k labelling is the method that mislabels (comma neurons, "capital cities" for a Santa neuron), and that most neurons do not fire on nouns. No audit of Qwen3-8B exists.
- **Where "one neuron = one concept" is best supported:** MLP hidden neurons (a privileged basis), strong activations only, middle layers, larger models, and lexical or name-like content. It is not supported for the residual stream, attention heads, weak activations, or facts and long-tail knowledge. Narrative tropes have not been audited by anyone.
- **The study's own results fit the sceptical reading.** Its strongest units are mostly name and word detectors, its concept units lose their cross-model advantage without names, and blind judges beat a word-overlap judge only modestly (`study/paper/paper.md`, abstract and section 3.9). That is what you would expect if the neuron-defined units in an 8B model are mostly lexical.
- **Superposition is not dead.** Cammarata redefines it narrowly (arbitrary packing of unrelated things) and concedes the standard definition (more features than dimensions, especially in the residual stream). Nanda, StefanHex and others hold that facts and weak activations are still in superposition. The dispute is live; it will be settled by weak-activation ablations on language models, planted-mixture controls and neuron-versus-random-direction baselines, not by more galleries.

## Findings

### 1. What the essay is

- Nick Cammarata, Goodfire, published 15 October 2026 (page metadata `citation_publication_date` 2026/10/15), revised 30 October 2026 (footnote 3). About 27,000 words. Page description: "A working hypothesis about neurons, generalization, and the ecology of circuits." A technical companion paper (with Curt Tigges) is promised and not yet out. https://already-clean.goodfire.com/
- Its four-step argument, verbatim in outline: (1) networks use one learning principle (generalization), not two; (2) with modern models and tools, direct study goes much better than expected; (3) understanding compounds as the map fills in; (4) "Highly speculative: this might be most of what we need".

### 2. Exact claims and the evidence offered

Vision (the well-developed half):
- InceptionV1 last layer: about 5% interpretable (his recollection; others about 0%) with 2020 tools; about 50% on the same weights with better tools; about 70% when the same architecture is retrained with modern regularisation; about 99 to 100% in a ConvNeXt of the same parameter count and data. He trained versions across a 300x size range and reports them "broadly legible at every scale"; small models have coarser neurons, not mixed ones.
- The canonical polysemantic neuron (InceptionV1 4e:55, cats and cars) is reread as responding to a shared V / bilateral pointed shape. Footnote 3 walks this back: the earlier wording "it's a V-shape neuron" "was too strong"; it does detect the objects; the claim is only that the mixture is not arbitrary (credited to Liv Gorton's pushback).
- The Zoom In "car into dogs" superposition circuit is reread as generalized pose neurons plus feature-visualization "reward hacking". "To my knowledge, no one has found a superposition circuit in a real network since my original one, which was an error."

Language models (the half P1 rests on; all "preliminary"):
- Qwen3-32B: "roughly 90% of neuron poles have one main job, often with small related offshoots; another ~7% have two related jobs; and only ~2-4% are not yet understood". Footnote 1: the share "varies depending on how you set the threshold for counting weak activations, depth, density of sub-circuits, and what you view as 'one thing'"; "closer to 97% non-arbitrary mixtures" on small samples of strong activations; "I don't think coming to a single score is actually that important".
- Basis: 1,000 randomly chosen "neuron papers" (agent-written investigations; essay: "Here are 1000 neuron papers ... chosen randomly"; IRSI counts the released gallery as 500 neurons, 1,000 poles). Said to be similar in mixture-of-experts models ("Dense and Mixture of Experts models seem roughly similar"; that sentence names no model, the MoE model used elsewhere in the essay is Qwen3.6-35B-A3B) and in Kimi K3 ("We found similar results in models up to trillions of parameters, like Kimi K3"; and of the Kimi graph: "We haven't begun full audits of this scale yet").
- GPT-2 XL redo (500 neuron papers, "almost no iteration"): "about 70% look like roughly one thing"; "pristine ones are still low, about 5%"; the rest "do a few broad things, up to about six, mostly two". Against under 2% in the 2023 OpenAI project. "Smaller models have broader neurons, with more residual generalization than modern models."
- Falsifiable test ("Explanations That Could Lose"): 200 random gallery neurons; 183 with positive-side explanations; texts written from the explanation alone fired the neuron strongly (at least 30% of corpus max) 76% of the time, near misses 11%; explanation won 164, tied 15, lost 4; "most distant concept" texts fired about 1% of the time and never for 179 of 183.
- "Blind maximum": a second agent, seeing only the explanation, writes a sentence that can beat the best example in 10 billion tokens.
- SwiGLU double-negative regime: for "about 4% of neurons" the top-activation list is an echo of the negative pole.
- Attribution graphs in the neuron basis reproduce an Anthropic cross-layer-transcoder case (preeclampsia, the essay's spelling; originally Claude 3.5 Haiku, redone in Qwen3-32B) with a four-neuron hub; a Santa example needs "eight neurons with a Santa negative side"; removing "the twenty neurons actually labeled Christmas or Santa" only drops Santa to 76%.
- Weak activations: "you can ablate almost all the weak activations of the network for almost no cost". The quantitative support is vision (Inception final layer 70.08% -> 69.47% keeping the strongest quarter). For language: "We don't yet have a semantic test for weak activations that we trust"; "Broader category-based screening gives mixed results, so these spot checks cannot establish what weak activations mean in general."

### 3. What it says about superposition

- It uses a narrow definition on purpose: "neurons learning unrelated things together because they rarely interfere" (arbitrary packing as a second learning principle). Its replacement is "recursive generalization": a neuron grows related offshoots ("iy" -> "diy" -> "do it yourself").
- It concedes the standard Toy Models definition: "more features than dimensions ... even a fully monosemantic network would satisfy, because its neurons compose"; on the residual stream, "A stream can hold far more than it has dimensions ... this alone isn't packing"; "Ruling that out entirely may not be possible."
- On Toy Models of Superposition: "we found no problems with its experiments"; the toy "prices the cost of sharing a neuron, interference, but not the benefit, reuse"; and "there it does pack". So the disagreement is about which regime real networks are in.
- It concedes the opposite-error risk ("an explanation of the gaps: assuming every neuron must have a sensible story").
- Attention: "We've made less progress on attention than on neurons".

### 4. Things in the essay that cut against using neurons the way the study does

- "Most humans and agents expect neurons to detect nouns by firing on them, which they mostly don't." The neuron is "better understood as a verb than a noun"; its job is "to draw a distinction downstream neurons can use".
- Neurons for feelings and social cues are "some of the most common", but they fire on commas and summary positions, "a pile of seemingly unrelated commas", and are identified by steering and circuits, not by reading top sentences.
- "standard top-k automated interpretability" mislabels: a Colorado neuron came out as a "polysemantic comma neuron"; a Santa neuron as "capital cities" because the negative pole was ignored.
- Short context breaks things: "two paragraphs felt like enough. But this accidentally made all the long-context neurons look polysemantic." The study keeps the first 40 sentences of each summary (`study/paper/paper.md` section 2.1).
- Strong activations are extremely rare: "100 million tokens through one Qwen3-32B neuron. Almost all sit near zero; about 160 reach the strong half." The study's corpus is 3,700 sentences, and a unit is "present" when its score beats 68 of 69 reference films (`study/paper/paper.md` sections 2.1 and 2.3; `study/extract.py` records per-sentence max and min only). That is a within-corpus rank, with no reference to the neuron's own activation range. My inference: for most of the 884,736 poles the study is thresholding the weak regime, which is where the essay says meaning is haziest and where critics say superposition lives.
- The study says its unit follows [10] in splitting poles and that names were "not validated by steering" (`study/paper/paper.md` section 2.3). The essay's own standard for a label is steering plus circuits plus negative pole plus tests that can lose.
- The model gap is acknowledged in the paper's limitations ("smaller than the 32B model audited in [10]", `study/paper/paper.md` section 4.4).

### 5. Responses and critiques (all dated October 2026)

(The LessWrong thread is "Linkpost: Neural Networks Are Already Clean", posted by Yair Halberstadt on 15 October 2026, score 220, 51 comments when checked through the LessWrong API on 2026-11-08. All comment quotes below were matched against the API text.)

- **Neel Nanda** (Google DeepMind), LessWrong comment, score 111, cross-posted from X: "interesting work, but I don't think it's a significant update to the big picture"; "I still think superposition is real and this work doesn't really affect my cruxes"; "language models have far more facts and memorized things in them than they have neurons"; "I expect that weak neuron activations contain lots of semantic superposition-y info"; "the post defines monosemantic too loosely, I think I'd call a fair few of those polysemantic"; "It's also plausible that the most important concepts are likely to correspond to a neuron (or a handful)". He supports neuron-based attribution graphs for pragmatic work and credits Transluce. https://www.lesswrong.com/posts/x4kvabkF2Jk8GLZp8/linkpost-neural-networks-are-already-clean/comment/7wcSqBNzQF2o9SHyY
- **StefanHex** (LessWrong handle; real name not confirmed this session), LessWrong: not convinced by "neurons is all we need"; expects computation in superposition (Hänni et al.); "maximum-activating examples tell a plausible but incomplete story"; the low-dimension squeeze experiments are not evidence, because superposition "only really works in high dimensions". Likes recursive generalization as a motif. Same thread, comment bySRD2nBT2GLfPmJS (LessWrong username Stefan42, score 11, 18 Oct 2026): https://www.lesswrong.com/posts/x4kvabkF2Jk8GLZp8/linkpost-neural-networks-are-already-clean/comment/bySRD2nBT2GLfPmJS
- **"Manic Pixie"**, LessWrong (the sharpest point for P1): "Showing that most neurons in a model encode one primary concept is not the same as showing that most concepts a model knows are represented by one neuron." (comment TWv5r7a9MW2opm5sP, score 15, 16 Oct 2026, same thread)
- **johnswentworth**, LessWrong (comment TZAHrGgy3B8EiwQLg, score 22, 16 Oct 2026, same thread): after an hour on the ConvNeXt microscope, "the semantics of each neuron are not quickly obvious to me"; wants a per-neuron artifact with explanation and evidence; "AFAICT that artifact doesn't exist yet".
- **Alex Gibson**, LessWrong (score 72, comment yTi9dKZawGmqmbZo2, 15 Oct 2026): the surrounding discourse overclaims; compares with the 2023 SAE optimism (he quotes the Towards Monosemanticity post).
- **Dean Valentine** (LessWrong username lc, comment GnTbLpcx6xiyw4ZFa), quoting a comment by Adam Scherlis on the LessWrong post "SAE feature geometry is outside the superposition hypothesis" (post by Jake Mendel, 24 June 2024). Valentine labels the quote "Adam Scherlis, 2024", but the quoted text is in Scherlis's comment dated 6 January 2026 (comment 2BwacbvQkLiqz9Ljn), not in his June 2024 comment on the same post: "cosine similarities between feature vectors have rich semantic content ... Features aren't just crammed together arbitrarily; they're grouped with similar features." So "related things share directions" is not new evidence against superposition. https://www.lesswrong.com/posts/MFBTjb2qf3ziWmzz6/sae-feature-geometry-is-outside-the-superposition-hypothesis
- **MIT Technology Review**, 19 Oct 2026 (by Will Douglas Heaven): Chris Potts (Stanford) calls it little more than a hunch (the reporter's paraphrase, not Potts's own words; his direct quotes are "nothing about this blog post makes the problems any easier" and "even a large number here is not a substitute for rigorous evaluation"); Jacob Hilton (ARC), from his X posts as quoted in the article: "For a full understanding, we need to go beyond neurons"; Oren Etzioni: the view revisits the 1986 picture ("a recapitulation acknowledging a 40-year-old theory"). The article also quotes Nanda's X post. No quote from Chris Olah. https://www.technologyreview.com/2026/10/19/1147149/we-were-looking-at-it-wrong-the-explosive-claim-thats-upended-the-ai-world/
- **IRSI retest** (Institute for Recursive Self Improvement; run 17 Oct, page updated 18 Oct 2026; written by AI research agents, not peer reviewed; treat as one data point). https://irsi.ai/notes/already-clean/
  - One-job share, three LLM classifiers reading Goodfire's own write-ups (500 neurons, 960 poles): 76.1% raw, 78.3% layer-reweighted (95% CI 75.5 to 81.1). Loosening definitions post hoc: 80.7% or 83.4% with one definition relaxed, 87.4% with two, and 91.6% (89.1 to 93.9) only with all three relaxed at once, two of which depart from the essay's own counting. (Corrected: an earlier draft of this line said "84 to 92%".)
  - Independent dashboard labeler on random neurons: 57.3% (N=48), 64.6% (N=192). Random directions: 31.3%. The gap missed the pre-registered 30-point line at N=48 and passed post hoc at N=192.
  - By layer: about 77% (layers 0 to 15), 87% (16 to 31), 73% (32 to 63). In layers 0 to 15 neurons did not beat random directions (69% vs 67%; post hoc, and only 12 random directions were tested in those layers. Corrected: the 12 is the count of random directions, not of neurons).
  - Planted two-job mixtures were called one-job 39.6% of the time as built (20.8% rebuilt). So the grader cannot reliably see polysemanticity when it is there.
  - A "sparsest rotation" test that neurons pass also scores 0.998 on planted mixtures and 0.997 on random GELU features. It cannot discriminate.
  - Held: texts written from Goodfire's explanations fire the neuron strongly 83.8% of the time versus 14.6% for near misses.
  - Did not hold: the double-negative ablation claim; the weak-activation claim narrowly (PopQA accuracy 22.4% -> 20.2% removing the weakest 75%); the scale and width claims on their tests.
  - Blinding was imperfect (a classifier tells neuron dashboards from random-direction ones at AUC about 0.85; the Claude labelers had read the pass lines).
- Not found: any public response from Anthropic, Chris Olah or Transluce. Joshua Batson (Anthropic) is quoted in the essay as a draft reader. Leo Gao and Roko replied on X (mentioned by Alex Gibson as "quite good", in his reply comment afnLHgeiaqx9SDErc, not in his score-72 comment); I could not retrieve them [UNVERIFIED: the X replies themselves were not seen].
- An IRSI-derived secondary write-up exists at remio.ai; I used it only to find the IRSI page. [UNVERIFIED: not re-checked by the citation checker; no URL was recorded and the web search budget was used up.]

### 6. The literature it is set against

Superposition as theory and its first evidence:
- Elhage et al. 2022, Toy Models of Superposition (arXiv:2209.10652): sparse independent features are stored in fewer dimensions than features in toy ReLU models. Source of the "privileged basis" idea: an elementwise nonlinearity makes the MLP hidden layer's neuron basis special; the residual stream has none. This is why P1 can only ever apply to MLP neurons.
- Elhage et al. 2022, Softmax Linear Units (SoLU) (https://transformer-circuits.pub/2022/solu/index.html): the essay quotes it (the sentence is in the SoLU paper itself, which continues: "Our SoLU results seem like moderate evidence for preferring the superposition hypothesis over alternatives"): superposition "is often treated as the default hypothesis simply because it seems intuitively more compelling than other explanations, but there is little evidence."
- Scherlis et al. 2022, Polysemanticity and Capacity (arXiv:2210.01892); Hänni et al. 2024, Mathematical Models of Computation in Superposition (arXiv:2408.05451): theory that networks can compute more functions than they have neurons. Theory, not measurement in real LLMs.
- Gurnee et al. 2023, Finding Neurons in a Haystack (arXiv:2305.01610), abstract: "early layers make use of sparse combinations of neurons to represent many features in superposition, that middle layers have seemingly dedicated neurons to represent higher-level contextual features, and that increasing scale causes representational sparsity to increase on average". This is the best pre-2026 statement of the conditions: layer and scale both matter. The essay disputes its one "social security" superposition case.

How many neurons are interpretable, by measurement:
- Bills et al. 2023 (OpenAI, https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html): GPT-4 explains GPT-2 XL neurons, scored by simulation. The essay summarises it as "about 1.7% of neurons interpretable by our most generous measure".
- Bricken et al. 2023, Towards Monosemanticity (https://transformer-circuits.pub/2023/monosemantic-features/index.html): one-layer model; per the essay, the median neuron scored 0 of 14 on its rubric. The SAE paradigm starts here.
- Templeton et al. 2024, Scaling Monosemanticity (https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html): SAEs on Claude 3 Sonnet's residual stream. (I confirmed the page exists; I did not re-read its neuron-versus-feature comparison this session.)
- Gurnee et al. 2024, Universal Neurons in GPT2 (arXiv:2401.12181), abstract: across five seeds "1-5% of neurons are universal". These "usually have clear interpretations". The clean, stable neurons are a small minority in GPT-2.
- Choi et al. 2024, Transluce, Scaling Automatic Neuron Description (Dami Choi, Vincent Huang, Kevin Meng, Daniel D. Johnson, Jacob Steinhardt, Sarah Schwettmann; published 23 October 2024; https://transluce.org/neuron-descriptions): descriptions for all 458,752 MLP neurons of Llama-3.1-8B-Instruct, both polarities. Mean simulation correlation 0.61 for automated descriptions versus 0.55 for human ones on 100 neurons (0.38 vs 0.34 with a prompted simulator). Scores fall from layer 1 to about layer 11, then rise. No "fraction monosemantic" is reported; the page raises concept superposition as a possible cause of misfiring neurons. This is one of the study's two models.
- Arora et al. 2026, Language Model Circuits Are Sparse in the Neuron Basis (arXiv:2601.22594, ICML 2026): "MLP neurons are as sparse a feature basis as SAEs" for circuits in Llama 3.1 8B; "a circuit of ~10^2 MLP neurons is enough to control" subject-verb agreement; "small sets of neurons encode specific latent reasoning steps". Sets, not single neurons. (Checked: arXiv comment reads "ICML Spotlight, camera-ready"; the authors are at Transluce and Stanford; the sparsity and circuit benchmarks use the Llama 3.1 8B base model and the multi-hop case study uses Llama 3.1 8B Instruct, with Gemma 2 results in an appendix.)
- Birardi 2026 (Giuseppe Birardi, LessWrong, "Do we need sparsity afterall?", posted 6 January 2026, MATS 10.0 application, about 20 hours; https://www.lesswrong.com/posts/Hk8tCLZKdAPHdoSdb/do-we-need-sparsity-afterall): in Gemma-2-2B raw MLP neurons fire above threshold on 10 to 25% of tokens (his TL;DR: "Most MLP neurons in Gemma-2-2B are dense and polysemantic"). Weak evidence, but it is the only public look at a small model near the study's size class.
- Liu et al. 2026, NeuronScope (arXiv:2601.03671): a multi-agent framework "for explaining polysemantic neurons". The field was still treating LLM neurons as polysemantic in January 2026.
- Bolukbasi et al. 2021, An Interpretability Illusion for BERT (arXiv:2104.07143): a neuron's top-activating examples look like one clean concept within one dataset and a different one in another. Directly relevant: the study names units from eight sentences in a Batman-heavy corpus.

Why the "alternative unit" (SAE latents) is not canonical either:
- Chanin et al. 2024, A is for Absorption (arXiv:2409.14507): feature splitting and absorption.
- Leask et al. 2025, Sparse Autoencoders Do Not Find Canonical Units of Analysis (arXiv:2502.04878).
- Paulo & Belrose 2025, Sparse Autoencoders Trained on the Same Data Learn Different Features (arXiv:2501.16615).
- Kantamneni et al. 2025, Are Sparse Autoencoders Useful? (arXiv:2502.16681); GDM team 2025, "Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research (GDM Mech Interp Team Progress Update #2)" (Smith, Rajamanoharan, Conmy, McDougall, Lieberum, Kramár, Shah, Nanda; LessWrong, 26 March 2025; https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks); Karvonen et al. 2025, SAEBench (arXiv:2503.09532).
- Transcoders: Dunefsky et al. 2024 (arXiv:2406.11944); Paulo et al. 2025, Transcoders Beat Sparse Autoencoders for Interpretability (arXiv:2501.18823); Anthropic circuit-tracing methods 2025 (https://transformer-circuits.pub/2025/attribution-graphs/methods.html), whose largest replacement model, per the essay, matches the original's next token "only half the time".
- Net: SAE units depend on dictionary size, seed and training choices. That removes "use SAE features instead" as an easy fix. It does not make neurons canonical; it means no unit currently is.
- Sharkey et al. 2025, Open Problems in Mechanistic Interpretability (arXiv:2501.16496): field-level statement that the right unit is unresolved.

### 7. Verdict: where is "one neuron = one concept" true?

Three claims are being run together. Separate them:

| Claim | Status | Best evidence |
|---|---|---|
| (a) Legibility: a neuron's strong activations have one describable main job | Partly supported, grader-dependent | 90% (Goodfire, self-graded, Qwen3-32B); 78% (IRSI on Goodfire's write-ups); 57 to 65% (IRSI independent labeler) vs 31% random directions; about 70% loose / 5% pristine in GPT-2 XL |
| (b) Uniqueness: a concept is carried by one neuron | Not supported | Arora et al.: "small sets of neurons"; essay's own hubs of 4, 8, 20 neurons; Nanda: "a neuron (or a handful)" at best |
| (c) Completeness: the neurons list the concepts the model has | Rejected by all sides, including Cammarata | "more features than dimensions" via composition; facts > neurons (Nanda, StefanHex) |

By model, layer and concept type:

| Where | One neuron ~ one job? | Notes |
|---|---|---|
| Vision, modern convnet late layers (ConvNeXt), shapes, poses, classes | Best case. 99% claimed; IRSI finds coherence above random rotations on its visual metric (Cohen's d 2.1 to 2.3, about 1.6 seed-corrected) but its WordNet metric cleared d of 1 in only 2 of 4 model sizes; IRSI's own verdict is "Partial" | One author, own models, own grading. Fails on original InceptionV1. IRSI did not see coherence rise with size |
| LLM MLP neurons, Qwen3-32B, middle layers, strong activations | Most likely place in language. About 87% in layers 16 to 31 (IRSI) | Early layers: no advantage over random directions in IRSI's small subset; Gurnee 2023 says early layers are in superposition. Late layers about 73%; the essay says the last layer is largely used to suppress single tokens |
| LLM MLP neurons, 8B class (the study's models) | Unknown. No audit exists | Essay predicts broader neurons; Transluce gives mean simulation correlation about 0.6 for Llama 3.1 8B, no fraction; 4-bit quantisation is untested by anyone |
| GPT-2 class | Minority. 1 to 5% universal across seeds; about 5% pristine | Gurnee 2024; essay's own redo |
| Residual stream | No | No privileged basis; conceded by the essay |
| Attention heads | Not claimed | Essay: "less progress" |
| Weak activations of any neuron | Contested; the crux | Vision ablations support "doesn't matter"; language evidence is spot checks; IRSI's PopQA test narrowly failed |
| Concept types: tokens, spelling fragments, names, output "say X / don't say X" | Clearest cases | Also the least interesting for memetics, and what word overlap already captures |
| Concept types: feelings, social context, "tropes" | Claimed common, but fire on punctuation and need steering to identify | Not what a top-sentence reader will find |
| Concept types: facts, entities, long-tail knowledge | Disputed | The standing argument for superposition |
| Narrative motifs across paraphrase | Untested by anyone | The study's rewording test is the only data: "Only a few units ... clearly survive rewording" (`study/paper/paper.md` abstract) |

How it is measured, and why the numbers differ by two orders of magnitude (1.7% to 90%):
1. Simulation score (Bills 2023; Transluce): predict per-token activations from the description. Punishes position effects and weak-activation detail; gives low numbers.
2. Rubric by human or LLM reading top examples (Towards Monosemanticity; IRSI dashboard labeler): middling numbers; blind to planted mixtures 20 to 40% of the time.
3. Agentic "neuron paper" with steering, circuits, both poles, long context, then a judgment of "one main job" (Goodfire): highest numbers; least independent; "one job" has no fixed boundary.
4. Prediction tests that can lose (generate positives, near misses, distant concepts): the most defensible. Note what it shows: explanations predict strong firing. It does not test whether the neuron has other jobs, and Cammarata says so ("A neuron in superposition that's just X or Y would do fantastically on near misses").
5. Cross-seed or cross-model universality (Gurnee 2024; the study's twins): a different property, and the one P2 needs.

### 8. What evidence would settle it

- Publication of the Goodfire technical paper with per-neuron data, the grading rubric and thresholds, by-layer and by-model-size breakdown, and inter-rater agreement.
- An independent, blinded audit with planted-mixture positive controls and random-direction negative controls, at 1.7B, 8B, 32B and one MoE model, same rubric. The key curve is "one-job share minus random-direction share" by layer and by model size.
- Weak-activation ablation on language models with knowledge-heavy evals split by fact rarity (the direct test of Nanda's crux). If removing weak activations selectively destroys rare facts, superposition holds where it matters.
- Concept -> neuron counts: for a fixed list of several hundred concepts, how many neurons are needed to detect and to steer each, and how much of the model's behaviour on that concept is left when those are ablated.
- One convincing "superposition circuit" in a real LLM (unrelated features packed into shared neurons and read out downstream). Cammarata says "I would update strongly on even one convincing example".
- Dataset-shift test of labels (Bolukbasi): do the top examples of a neuron mean the same thing on fiction, news, code and chat?
- For this project specifically: the above on the study's actual models and units, plus a paraphrase-invariance test.

## What this means for P1 and P2

**P1.** The paper's sentence "define a meme as whatever makes a single neuron ... fire" rests on a claim that is (i) preliminary and self-graded, (ii) about a model four times larger than the one used, (iii) about strong activations found over billions of tokens with steering and circuits, not about within-corpus rank on 3,700 sentences, and (iv) about legibility, not about concepts mapping to single neurons. The fair statement is: "MLP neurons are a privileged, cheap, better-than-random basis in which a majority of units have a describable main job at high activation; which jobs those are is dominated by lexical and positional functions in small models." That supports neurons as a convenient instrument. It does not support neurons as the natural unit of culture. The study's findings (units mostly track names and words; word overlap matches them; Qwen residual test p = 0.09 to 0.25, `study/paper/paper.md` Table 2) are what this weaker statement predicts. The study's family-merging step (correlation at least 0.5) is a sensible patch for claim (b), and should be described as such: the unit in practice is a co-firing family, not a neuron.

One specific mismatch is worth fixing before any new run: the study's presence rule is relative to 69 reference films, so it never asks whether the neuron is in its own strong regime. Cammarata's whole case is about the strong regime. A unit that is "present" at 5% of its global maximum is, on his account, an analogy ("Titanic-ish"), and on Nanda's account possibly a different feature.

**P2.** This topic bears on P2 through two results. Gurnee et al. 2024: only 1 to 5% of GPT-2 neurons recur across seeds of the same architecture. The study: twins are strong mainly for name and word units, and "without names the agreement collapses" (`study/paper/paper.md` abstract). If only the lexical minority of neurons is universal, then cross-model generality at the neuron level reduces to cross-model agreement on words, which needs no neurons. Cammarata's essay shows one pose neuron recurring across DINOv2 and ConvNeXts, which is an anecdote. The essay's prior-work list cites "Neuron Populations Exhibit Divergent Selectivity with Scale" (arXiv:2606.03990, Dravid et al.), whose title points the other way; I did not read it [UNVERIFIED content]. Checker's update, from the arXiv abstract only (Dravid, Bahri, Efros, Gandelsman, 2 June 2026): "Rosetta Neurons" (units whose activation patterns are similar across independently trained models) were counted in language models up to 30B parameters and vision models up to 5B; their population "follows a sublinear power law in model size, growing in absolute number but occupying a shrinking fraction of the total neuron count", and they "become more selective and increasingly monosemantic with scale, separating from a growing non-Rosetta population that remains less selective". So the paper exists and says what its title suggests: the cross-model-shared share of neurons falls as models grow. That cuts against P2 at the neuron level for large models, while the shared minority gets cleaner. The body of the paper is still unread.

## Experiment seeds

1. **Strong-regime audit of the study's own units.**
   Question: are the named meme units firing in their strong regime on Batman text, or at a few percent of their global maximum?
   Data: the committed units and labels (`study/results/labels.json`, `study/results/qwen3-8b/`); a generic corpus of 100M+ tokens (e.g. FineWeb) to get each unit's global activation distribution; for Llama, Transluce's public exemplars.
   Method: for each named unit, express its Batman-corpus activations as a fraction of its global max and as a global percentile; relabel from global top examples; compare with the in-corpus label.
   Rival to beat: the single best-matching word or stem (the study's word-overlap judge). If the global-top-example label is just a word, the unit is lexical.
   Falsifies P1-as-used if: most "present" calls sit below about 30% of global max and global labels disagree with in-corpus labels for more than a third of units.
   Cost: one or two days of local inference at 8B; no new method.

2. **Planted-mixture and random-direction controls for the labelling pipeline.**
   Question: can the study's label-and-blind-test pipeline tell a real neuron from a random direction or from a synthetic two-job mixture?
   Data: same activations; random unit-norm directions in MLP hidden space; sums of two unrelated labelled units.
   Method: run the identical naming and blind sentence-choice test on real units, random directions and planted mixtures, blind to type.
   Rival to beat: word-overlap judge on the same items.
   Falsifies: if blind-test accuracy on random directions is within about 10 points of real units, or mixtures pass as single memes more than 30% of the time, the labels carry no evidence of one-neuron-one-concept.
   Cost: hours of compute plus LLM judging; under a day.

3. **Paraphrase-invariance census.**
   Question: what fraction of units respond to an idea when no content word is shared?
   Data: 200 to 500 short scene descriptions of fixed motifs (secret identity, origin in loss, mentor's death, mask, revenge vow), each written in several ways with disjoint vocabulary, plus matched non-motif controls. Generated by an LLM and checked for stem overlap.
   Method: per-unit AUC for motif versus control across paraphrase sets with zero shared stems; count units above a pre-set AUC; repeat at 1.7B, 8B, 32B and by layer.
   Rival to beat: sentence-embedding cosine to a motif prototype, TF-IDF (which should be at chance by construction), and an LLM asked "does this passage contain motif X?".
   Falsifies P1 for narrative content if: fewer than about 1% of candidate units pass, or the sentence-embedding and LLM-reader rivals match the best unit on every motif. Then the neuron adds inspectability only.
   Cost: a few days; the corpus is the main work.

4. **Concept -> neuron count (the converse claim).**
   Question: how many neurons does one motif need?
   Data: the motif sets from seed 3.
   Method: k-sparse probes over MLP neurons (as in Gurnee et al. 2023) for k = 1, 2, 4, 8, 32, 128; report accuracy against k, by layer and model size. Optional: RelP-style attribution as in Arora et al. to list the causal set.
   Rival to beat: a logistic probe on a sentence embedding, and k-sparse probes on random directions.
   Falsifies: if k = 1 accuracy is far below k = 32 for most motifs, "a meme is one neuron" is wrong and the unit should be a small set; if random directions do as well as neurons at equal k, the neuron basis is not special for these concepts.
   Cost: one to two days.

5. **Model-size and quantisation check of legibility.**
   Question: does the one-job share really rise from 1.7B to 8B to 32B, and does 4-bit quantisation change which units fire?
   Data: 200 random units per model; full-precision and 4-bit activations on the same text.
   Method: (i) rank correlation of unit activations between 4-bit and full precision, and overlap of top-20 examples; (ii) the same blinded rubric on each model size, with random-direction baseline.
   Rival to beat: random directions (for legibility); for the memetics use, word-stem detectors.
   Falsifies: if 4-bit top examples overlap with full precision at under about 70%, the study's units are partly quantisation artefacts; if the neuron-minus-random gap does not grow with size, the "bigger is cleaner" premise fails (IRSI already found no rise in its vision test).
   Cost: needs a 32B run; days on rented GPU. Out of scope for a laptop.

6. **Weak-regime superposition test on the study's units.**
   Question: do a unit's weak activations carry the same meaning as its strong ones?
   Data: per unit, sentences binned by activation as a fraction of global max (above 50%, 20 to 50%, 5 to 20%).
   Method: blind judges match the label to sentences from each bin against controls; plot accuracy by bin.
   Rival to beat: word-overlap judge per bin.
   Falsifies: if accuracy in the 5 to 20% bin falls to the word-overlap rate or to chance, then presence calls made in that bin are not the labelled meme.
   Cost: one day.

## Key sources

| Citation | URL / DOI | Why it matters | Priority |
|---|---|---|---|
| Cammarata 2026, Neural Networks Are Already Clean (Goodfire; published 15 Oct 2026, revised 30 Oct 2026) | https://already-clean.goodfire.com/ (text: https://already-clean.goodfire.com/essay.md ; PDF: https://already-clean.goodfire.com/already-clean.pdf) | The source of P1. Read the footnotes and the appendix, not the summary | A |
| Nanda 2026, LessWrong comment on the linkpost | https://www.lesswrong.com/posts/x4kvabkF2Jk8GLZp8/linkpost-neural-networks-are-already-clean/comment/7wcSqBNzQF2o9SHyY | The main expert rebuttal: facts, weak activations, loose definition | A |
| IRSI 2026, retest note (AI-agent-written, not peer reviewed) | https://irsi.ai/notes/already-clean/ | Only outside numbers: 78%, 57 to 65% vs 31% random directions; planted-mixture blind spot | A |
| Arora, Wu, Steinhardt, Schwettmann 2026, Language Model Circuits Are Sparse in the Neuron Basis | arXiv:2601.22594, https://arxiv.org/abs/2601.22594 (ICML 2026; arXiv comment: "ICML Spotlight, camera-ready") | Peer-reviewed support for the neuron basis in Llama 3.1 8B; shows sets, not single neurons | A |
| Choi, Huang, Meng, Johnson, Steinhardt, Schwettmann 2024, Scaling Automatic Neuron Description (Transluce, 23 Oct 2024) | https://transluce.org/neuron-descriptions | Descriptions and scores for every neuron of one of the study's models | A |
| Gurnee, Nanda, Pauly, Harvey, Troitskii, Bertsimas 2023, Finding Neurons in a Haystack: Case Studies with Sparse Probing | arXiv:2305.01610, https://arxiv.org/abs/2305.01610 | Layer and scale conditions; k-sparse probing method for seed 4 | A |
| Gurnee et al. 2024, Universal Neurons in GPT2 Language Models | arXiv:2401.12181, https://arxiv.org/abs/2401.12181 | 1 to 5% universal: the base rate for P2 at neuron level | A |
| Elhage et al. 2022, Toy Models of Superposition | arXiv:2209.10652 | Standard definition; privileged basis | B |
| Bills et al. 2023, Language models can explain neurons in language models | https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html | The 2023 baseline the essay overturns; simulation scoring | B |
| Bricken et al. 2023, Towards Monosemanticity | https://transformer-circuits.pub/2023/monosemantic-features/index.html | Origin of the SAE turn; the "four unrelated things" neuron | B |
| Templeton et al. 2024, Scaling Monosemanticity | https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html | SAE features at scale; feature splitting | B |
| Bolukbasi et al. 2021, An Interpretability Illusion for BERT | arXiv:2104.07143 | Top examples depend on the corpus; applies to in-corpus labelling | A |
| StefanHex 2026, LessWrong comment | https://www.lesswrong.com/posts/x4kvabkF2Jk8GLZp8/linkpost-neural-networks-are-already-clean/comment/bySRD2nBT2GLfPmJS | Computation-in-superposition objection; max-activation caveat | B |
| Hänni et al. 2024, Mathematical Models of Computation in Superposition | arXiv:2408.05451 | Theory behind "more computations than neurons" | C |
| Scherlis et al. 2022, Polysemanticity and Capacity in Neural Networks | arXiv:2210.01892 | Capacity account of polysemanticity | C |
| Chanin et al. 2024, A is for Absorption | arXiv:2409.14507 | SAE units are not clean either | B |
| Leask et al. 2025, SAEs Do Not Find Canonical Units of Analysis | arXiv:2502.04878 | No unit is canonical | B |
| Paulo & Belrose 2025, SAEs Trained on the Same Data Learn Different Features | arXiv:2501.16615 | Seed-dependence of SAE units | C |
| Kantamneni et al. 2025, Are Sparse Autoencoders Useful? | arXiv:2502.16681 | SAEs lose to simple baselines on probing; the same "beat the cheap rival" lesson | B |
| GDM mech interp team 2025 (Smith, Rajamanoharan, Conmy, McDougall, Lieberum, Kramár, Shah, Nanda), Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research (26 Mar 2025) | https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks | Why the field was already moving off SAEs | C |
| Dunefsky et al. 2024, Transcoders Find Interpretable LLM Feature Circuits | arXiv:2406.11944 | Transcoder alternative | C |
| Paulo et al. 2025, Transcoders Beat Sparse Autoencoders for Interpretability | arXiv:2501.18823 | Same | C |
| Liu, Miao, Zhao, Liu, Du 2026, NeuronScope: A Multi-Agent Framework for Explaining Polysemantic Neurons in Language Models | arXiv:2601.03671, https://arxiv.org/abs/2601.03671 | Field still assumed polysemantic neurons in Jan 2026 | C |
| Sharkey et al. 2025, Open Problems in Mechanistic Interpretability | arXiv:2501.16496 | Field consensus that the unit question is open | B |
| Heaven 2026, "'We were looking at it wrong': The explosive claim that's upended the AI world", MIT Technology Review, 19 Oct 2026 | https://www.technologyreview.com/2026/10/19/1147149/we-were-looking-at-it-wrong-the-explosive-claim-thats-upended-the-ai-world/ | Outside reactions (Potts, Hilton, Etzioni) | C |
| Birardi 2026, Do we need sparsity afterall? (LessWrong, 6 Jan 2026) | https://www.lesswrong.com/posts/Hk8tCLZKdAPHdoSdb/do-we-need-sparsity-afterall (mirror: https://www.greaterwrong.com/posts/Hk8tCLZKdAPHdoSdb/do-we-need-sparsity-afterall) | Small-model neurons are dense; weak but on point | C |
| Mendel 2024, LessWrong post "SAE feature geometry is outside the superposition hypothesis" (24 June 2024) and Adam Scherlis's comment on it (the quoted comment is dated 6 Jan 2026, id 2BwacbvQkLiqz9Ljn) | https://www.lesswrong.com/posts/MFBTjb2qf3ziWmzz6/sae-feature-geometry-is-outside-the-superposition-hypothesis | "Related things are near each other" predates the essay | C |

## Fog

- The Goodfire technical paper is not out. Every language-model number in the essay is "preliminary". The rubric for "one main job", the activation threshold, and inter-rater agreement are unpublished.
- No audit of any kind exists for Qwen3-8B or Qwen3-1.7B, and none for 4-bit quantised models [UNVERIFIED: a claim of absence; the checker could not search for one, so read it as "none found"]. Whether the study's units are legible in Cammarata's sense is unknown.
- IRSI is a single group, its note is written by AI agents and not peer reviewed, and I read it through a summarising fetch, not line by line. Its numbers need a direct check before being quoted in a paper. (Checker's update, 2026-11-08: every IRSI number quoted in these notes was matched against the page text; two were corrected, see the check log. That confirms what the page says. It does not confirm that IRSI's results are right.)
- I found no public response from Anthropic, Chris Olah or Transluce. Leo Gao's and Roko's X replies were not retrievable. X is poorly indexed, so absence here is weak evidence.
- I did not re-read Scaling Monosemanticity or Towards Monosemanticity this session; their content is reported as the essay describes it or from the confirmed page titles.
- "Neuron Populations Exhibit Divergent Selectivity with Scale" (arXiv:2606.03990) and "Constructing Interpretable Features from Compositional Neuron Groups" (arXiv:2506.10920) are listed in the essay's prior work and look directly relevant; I did not open them [UNVERIFIED]. Checker's update: both exist on arXiv with these titles, and I read the abstracts only. 2606.03990 (Dravid, Bahri, Efros, Gandelsman, June 2026): cross-model-shared "Rosetta Neurons" grow in number but shrink as a fraction of all neurons as models scale, and become more monosemantic. 2506.10920 (Shafran, Geiger, Geva; arXiv comment says ACL 2026 main conference): decomposes MLP activations with semi-nonnegative matrix factorisation into features that are "sparse linear combinations of co-activated neurons", and reports that these beat SAEs on causal steering in Llama 3.1, Gemma 2 and GPT-2. That is direct support for "the unit is a small neuron group, not one neuron". The bodies of both papers are unread.
- My claim that the study mostly thresholds weak-regime activations is an inference from the presence rule and the essay's "about 160 in 100 million tokens" figure. Batman-specific neurons may well be in their strong regime on Batman text. Seed 1 checks it.
- Nobody has measured the concept -> neuron direction at scale. The fraction of human-nameable narrative concepts that have a dedicated neuron is unknown in any model.
- Whether "recursive generalization" and "superposition of related features" make different predictions for a downstream user (as opposed to for an interpretability researcher) is unclear. For memetics, a neuron that fires on "iy", "diy" and "do it yourself" is a bad unit under either story.

## Check log

Citation and claim check, 2026-11-08. Methods: arXiv API (`id_list` query for all 18 arXiv ids), direct fetch of the Goodfire essay (`essay.md` and the HTML metadata), the IRSI note, the MIT Technology Review article, the Transluce page, the SoLU, Towards Monosemanticity and OpenAI neuron-explainer pages, the arXiv HTML of Arora et al., the LessWrong GraphQL API (posts and comments), and `grep` on `study/paper/paper.md`. WebSearch was not available (the session's search budget was used up), and greaterwrong.com reset every connection. About 170 items were checked: 27 key sources and roughly 145 quotes, numbers and attributions.

Corrections (text changed in place, nothing deleted):

1. Section 5, IRSI: "Reaches 84 to 92% only when several definitions are loosened together" was wrong. The IRSI page gives 80.7% and 83.4% with one definition relaxed, 87.4% with two, 91.6% (89.1 to 93.9) with all three. Rewritten.
2. Section 5, IRSI: "(69% vs 67%, only 12 neurons)" was wrong. The 12 is the number of random directions tested in layers 0 to 15, and the comparison is post hoc. Rewritten.
3. Section 7 table, vision row: "d about 2" was incomplete. IRSI's visual metric gives d of 2.1 to 2.3 (about 1.6 seed-corrected), its WordNet metric cleared d of 1 in only 2 of 4 sizes, and IRSI's verdict is "Partial". Rewritten.
4. Section 5, Dean Valentine: "a 2024 comment by Adam Scherlis" was wrong. Valentine labels it 2024, but the quoted text is in Scherlis's comment of 6 January 2026 (id 2BwacbvQkLiqz9Ljn) on Jake Mendel's June 2024 post. His June 2024 comment on that post does not contain it. Fixed in section 5 and in the Key sources row.
5. Section 5, MIT Technology Review: "little more than a hunch" was shown as a Potts quote. It is the reporter's paraphrase. Quote marks removed, Potts's two direct quotes added, and Hilton's line marked as coming from his X posts as quoted in the article.
6. Section 4: the essay's wording is "to draw a distinction downstream neurons can use", not "draws a distinction ...". Fixed.
7. Section 2: "pre-eclampsia" changed to the essay's spelling "preeclampsia", with the note that the original case was Claude 3.5 Haiku.
8. Section 2: the sentence tying "We haven't begun full audits of this scale yet" to both the MoE Qwen and Kimi K3 was imprecise. The essay's sentences are "Dense and Mixture of Experts models seem roughly similar" (no model named) and "We found similar results in models up to trillions of parameters, like Kimi K3"; the "full audits" quote is about the Kimi graph only. Rewritten with the exact quotes.
9. "What this means for P2": the `[UNVERIFIED content]` flag on Dravid et al. (arXiv:2606.03990) is now resolved from the abstract. The paper exists and reports that cross-model-shared neurons shrink as a fraction of all neurons as models grow. Added, with the caveat that only the abstract was read.
10. Fog: the `[UNVERIFIED]` flag on arXiv:2606.03990 and arXiv:2506.10920 is resolved to "exist, abstracts read, bodies unread", with one line on each.

Additions (authors, dates, URLs):

11. Section 5: a header note naming the LessWrong linkpost (Yair Halberstadt, 15 Oct 2026, score 220, 51 comments).
12. Section 5: comment ids, scores and dates for StefanHex (full URL added), Manic Pixie, johnswentworth, Alex Gibson and Dean Valentine. The Leo Gao and Roko mention is in Alex Gibson's reply comment afnLHgeiaqx9SDErc, not his score-72 comment.
13. Section 6: full title of the SoLU paper and confirmation that the quoted sentence is in it; authors and date for the Transluce report; models and venue note for Arora et al.; author, date and canonical lesswrong.com URL for Birardi; authors, date and full title for the GDM post.
14. Key sources: essay dates and the essay.md and PDF URLs; arXiv URLs and full titles or authors for Arora, Gurnee 2023, Gurnee 2024 and NeuronScope; authors for the Transluce report and the GDM post; full URL for the StefanHex comment; author and headline for the MIT Technology Review article; lesswrong.com URL for Birardi (the greaterwrong link is kept as a mirror); author and dates for the SAE feature geometry row.
15. Fog: a note that all IRSI numbers quoted here now match the page text.

Marked `[UNVERIFIED]`:

16. The remio.ai secondary write-up (no URL recorded, could not be searched).
17. The X replies by Leo Gao and Roko (only Alex Gibson's mention of them was seen).
18. "No audit of any kind exists for Qwen3-8B ..." (a claim of absence).

Not marked, but not independently confirmed either:

- "Not found: any public response from Anthropic, Chris Olah or Transluce" is a claim of absence. The MIT Technology Review article and the LessWrong thread contain none, which is all that was checked.
- StefanHex's real name (the notes already say it was not confirmed).
- "The technical paper is still unpublished": the essay text fetched today still says "forthcoming technical paper", and the 19 October article says results will be published "soon". No search for the paper was possible.
- One-line descriptions of Scherlis 2022, Hänni 2024, Chanin 2024, Leask 2025, Paulo and Belrose 2025, Kantamneni 2025, Karvonen 2025, Dunefsky 2024, Paulo et al. 2025 and Sharkey 2025 were checked against title, authors and year only, not against the paper text. All ten exist with the stated titles and first authors.
- Scaling Monosemanticity (Templeton et al. 2024): the URL returns 200; its content was not re-read.

Confirmed with no change needed:

- All 18 arXiv ids resolve to the stated titles, first authors and years. Venue notes from arXiv comments: Leask et al. ICLR 2025; Chanin et al. NeurIPS 2025 (oral); Karvonen et al. ICML 2025; Dunefsky et al. NeurIPS 2024; Hänni et al. ICML 2024 MI workshop.
- Essay: publication date 2026/10/15 in the page metadata, "Revised October 30, 2026" in footnote 3, 27,050 words in essay.md (the MIT Technology Review article says 26,000), and every other quoted phrase and number in sections 1 to 4 and 7 (90%, ~7%, ~2-4%, 97%, 1000 neuron papers, 500 GPT-2 XL papers, 70%, 5%, 200, 183, 76%, 11%, 164, 15, 4, about 1%, 179 of 183, about 4%, 70.08% and 69.47%, 300x, 10 billion tokens, 160 in 100 million, four-neuron hub, eight neurons, twenty neurons and 76%, "0 on its rubric out of a possible 14", "about 1.7%", "only half the time").
- IRSI: 76.1%, 78.3% (75.5 to 81.1), 500 neurons and 960 scored poles, 57.3% (48 neurons), 64.6% (192), 31.3%, the missed 30-point line, 77%, 87% and 73% by layer, 39.6% and 20.8% planted mixtures, 0.998 and 0.997, 83.8% and 14.6%, PopQA 22.4% to 20.2%, AUC about 0.85, run 17 Oct and updated 18 Oct 2026, "Written by IRSI's AI research agents; not peer reviewed", and the failed double-negative, size and width tests.
- Nanda's comment: all six quotes, score 111, the Transluce credit.
- Gurnee 2023 and Gurnee 2024 abstract quotes; Arora et al. abstract quotes; Bolukbasi et al. summary.
- Transluce: 458,752 neurons, both polarities, 0.61 vs 0.55 on 100 neurons, 0.38 vs 0.34 with the non-finetuned simulator, scores falling from layer 1 to layer 11, the concept-superposition remark.
- Birardi: 10 to 25% of tokens, MATS application, about 20 hours.
- Study numbers against `study/paper/paper.md`: 884,736 units, 3,700 sentences, first 40 sentences, 4-bit, eight sentences per name, correlation of at least 0.5, "following [10]", "not validated by steering", "beats all but one of the reference films" (section 2.3; "68 of the 69" is in the Figure caption in section 3.1), p = 0.09 to 0.25 (Table 2), "without names the agreement collapses" and "clearly survive rewording" (abstract), "smaller than the 32B model audited in [10]" (section 4.4), 79% vs 65% for blind judges against the word-overlap judge (section 3.9). The paths `study/results/labels.json`, `study/results/qwen3-8b/`, `study/results/experiments.json`, `study/results/robustness/` and `study/extract.py` all exist.
