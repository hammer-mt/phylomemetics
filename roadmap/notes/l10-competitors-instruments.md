# l10 — Competitors and instruments: who else uses model internals to measure culture or text, and is the neuron-meme idea new?

Sweep date 2026-11-08. Web search plus arXiv API and Crossref. No experiments run. Every source below was confirmed to exist in this session unless marked `[UNVERIFIED]`.

## Bottom line

- **"Model internals as a measuring instrument for text" is already a mainstream research line, and it is not Mike's.** Movva et al. (ICML 2025), Jiang et al. with Neel Nanda (ICML 2026 poster) and Peng et al. (ICML 2026 position paper) all use sparse internal units as interpretable columns for corpus analysis. Every one of them uses **sparse-autoencoder (SAE) features, not raw neurons**, and the two papers that looked at culture say why: single neurons mix abstract cultural content with surface word cues (Zou, Jiao and Hu, Findings of ACL 2026).
- **Verdict on "meme = neuron": new as a named definition, not new as a method, and its one distinctive choice is the contested one.** I found no paper before or after October 2026 that defines a meme as what makes a neuron fire; the only hit is Mike's own repository. But the parts all exist: culture neurons (Namazifard and Galke 2025; Yamamoto et al. 2025), cultural SAE features (Zou et al. 2026; Khanuja et al. 2026), narrative neurons and features (Liu, Joshi, Dawson and Piper, arXiv October 2026). What is original is choosing the raw neuron over the SAE feature, and that choice rests on one un-replicated essay.
- **Verdict on "phylomemetics with neurons": open ground, with two scoped exceptions.** Nobody was found using model-internal units as heritable traits in a descent test on cultural works. The nearest are Tanaka 2025 (phylogenetic networks of Japanese waka from a neural language model's vectors, checked against a documented allusion record) and Goodfire's Evo 2 result (the bacterial tree of life recovered from a genomics model's activations). Both use dense geometry, not units, and neither tests heredity of named traits. Claim "first to use language-model units as heritable traits in a pre-registered heredity test", not "first neural phylogeny of literature".
- **Competitors have done five things this project has not:** used SAE features; shown cause by ablation and steering; scored against human-labelled ground truth (TV Tropes trope sets, SemEval-2026 narrative similarity, motif-indexed folktales); recovered planted hypotheses in synthetic data; and benchmarked on cost and accuracy against an LLM reading the text and against dense embeddings.
- **The project has done three things the competitors have not:** descent over time; pre-registration with a 30,240-specification multiverse; and a word-overlap residual test. That residual test is harsher than anything in the competitor papers, and none of them would obviously pass it.
- **Since 15 October 2026 the responses to "Neural networks are already clean" are blog posts and one GitHub replication, no paper.** The early replications are on small Qwen3 models and cut against the essay (labels hold only at peak activation; a random-initialised network was labelled "one job" more often than the trained one). Goodfire's technical companion was still pending in the sources found. None of the responses uses neurons to study culture.

## Findings

### A. The instrument line: sparse model units as interpretable features of text

1. **Movva, Peng, Garg, Kleinberg, Pierson, "Sparse Autoencoders for Hypothesis Generation" (ICML 2025, PMLR 267:44997–45023; arXiv:2502.04382).** Train an SAE on *text embeddings*, pick the features that predict an outcome, have an LLM name each. Reported: better recovery of planted reference hypotheses on synthetic data (at least +0.06 F1), about twice as many significant findings on real data (headline engagement, congressional speech), at one to two orders of magnitude less compute than LLM-based methods. https://proceedings.mlr.press/v267/movva25a.html. This is the paper's reference [25] (`study/paper/paper.md` line 296). It is the only competitor in this section that the paper cites.
2. **Jiang, Sun, Dunlap, Smith, Nanda, "Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit" (arXiv:2512.10092, December 2025; listed as an ICML 2026 poster).** SAEs on LLM hidden states give "SAE embeddings" whose columns are named concepts. Four tasks: dataset diffing, correlation discovery, controllable clustering, property-based retrieval. Claims: bigger differences found than by LLM comparison at 2 to 8 times lower cost; beats dense embeddings on property retrieval. **This is the closest methodological competitor**: it is exactly "run text through a model, read off which interpretable units fire, compare corpora". It has no time axis and no descent. https://arxiv.org/abs/2512.10092
3. **Peng, Movva, Kleinberg, Pierson, Garg, "Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts" (arXiv:2506.23845; ICML 2026 position paper as "Position: Use Sparse Autoencoders to Discover Unknowns").** Argues SAEs lose to simple baselines when the concept is specified in advance (detection, steering) and earn their keep only when finding concepts nobody specified. https://arxiv.org/abs/2506.23845. **This predicts the Batman result.** Finding a documented source comic is a known-target detection task, and there the cheap baseline tied. The same paper says where the win should be: discovery and naming.
4. **Zhong, Wang, Klein, Steinhardt, "Explaining Datasets in Words: Statistical Models with Natural Language Parameters" (NeurIPS 2024; arXiv:2409.08466)** and **Zhong et al., "Goal Driven Discovery of Distributional Differences via Language Descriptions" (arXiv:2302.14233).** Clustering, time-series and classification models whose parameters are natural-language predicates, scored by an LLM reading each text. No internals: this is the "LLM simply reads the text" rival, formalised, with a time-series variant that tracks how a corpus changes. https://arxiv.org/abs/2409.08466
5. **Lam, Teoh, Landay, Heer, "Concept Induction: Analyzing Unstructured Text with High-Level Concepts Using LLooM" (arXiv:2404.12259).** LLM-induced concepts with explicit inclusion criteria. Same family as 4. https://arxiv.org/abs/2404.12259
6. **Tamkin, McCain, Handa et al., "Clio: Privacy-Preserving Insights into Real-World AI Use" (arXiv:2412.13678).** One million Claude conversations summarised, embedded and clustered by a model. Not internals: an LLM reader plus dense embeddings at scale. It is a competitor as the rival, not as the instrument. https://arxiv.org/abs/2412.13678
7. **Choi, Huang, Meng, Johnson, Steinhardt, Schwettmann, "Scaling Automatic Neuron Description" (Transluce, 23 October 2024).** Descriptions for all 458,752 MLP neurons of Llama-3.1-8B-Instruct at about $0.046 per neuron. The one large *neuron-level* instrument, and the paper already uses it (reference [24]). A 2026 paper that used it cautions the descriptions are artefacts, not ground truth (arXiv:2606.15507, seen in search result only).

### B. Model internals applied to culture, literature and narrative

8. **Zou, Jiao, Hu, "Deciphering Cultural Representations in Large Language Models via Sparse Autoencoders" (Findings of ACL 2026, pp. 5656–5677; doi:10.18653/v1/2026.findings-acl.278).** Says earlier "cultural neuron" work is hard to read because single neurons are polysemous, "conflating abstract cultural knowledge with surface-level lexical cues". Uses SAE features instead; reports features stable across paraphrase and task format, with ablation and steering as causal evidence. https://aclanthology.org/2026.findings-acl.278/. **This is the Batman study's own finding (strong units are mostly names and words) stated as the motivation for abandoning neurons.**
9. **Namazifard and Galke, "Isolating Culture Neurons in Multilingual Large Language Models" (arXiv:2508.02241)** and **Yamamoto, Kumon, Bollegala, Yanaka, "Neuron-Level Analysis of Cultural Understanding in Large Language Models" (arXiv:2510.08284).** Both find small neuron sets (under 1% in the second) whose suppression hurts cultural benchmarks (up to 30%) but not general ones. They disagree on where the neurons are (upper layers against shallow-to-middle layers). "Culture" here means national culture, not memes. https://arxiv.org/abs/2508.02241, https://arxiv.org/abs/2510.08284
10. **Khanuja et al., "Steering LLMs for Culturally Localized Generation" (arXiv:2603.23301).** Aggregates SAE features into "Cultural Embeddings" for analysis and steering. https://arxiv.org/abs/2603.23301
11. **Liu, Joshi, Dawson, Piper, "Language Models Have Interpretable Narrative Mechanisms" (arXiv:2610.12989, 9 October 2026).** In GPT-2-Small, Gemma-2-2B and **Llama-3.1-8B**, finds attention heads, MLP neurons and SAE features that correlate with narrativity; ablating them raises loss more on high-narrativity text and lowers the narrativity of generations. https://arxiv.org/abs/2610.12989. **Andrew Piper is a leading cultural-analytics researcher: the humanities side has arrived at neurons at the same moment, one week before the Cammarata essay, on one of the study's two models, and with causal tests the study lacks.** It studies one property (narrativity), not a catalogue of tropes, and no lineage.
12. **Mahran and Simbeck, "GPT and Prejudice: A Sparse Approach to Understanding Learned Representations in Large Language Models" (arXiv:2510.01252).** Trains a GPT on 37 nineteenth-century novels and reads SAE features for eleven social and moral categories. Uses the model as a probe of a corpus. https://arxiv.org/abs/2510.01252
13. **Presa and Oliveira, "Compositional Literary Primitives in Instruction-Tuned LLMs: Cross-Architectural SAE Features for Self, Style, and Affect" (arXiv:2605.18808).** SAE features in Llama 3.1 8B-Instruct and Gemma 2 9B-IT; reports the same four feature classes in both but an asymmetry (Llama names an emotion, Gemma evokes it). Single unreviewed preprint. https://arxiv.org/abs/2605.18808
14. **Matthews et al., NAACL 2025 (aclanthology 2025.naacl-long.559): k-sparse autoencoders to track the meaning of French "indigène(s)" over 125 years.** Sparse units as a diachronic instrument for one word. Title and details from search summary only; treat specifics as `[UNVERIFIED]`.

### C. The scientific-instrument line outside text (Goodfire, biology)

15. **Simon and Zou, "InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders" (arXiv:2412.12101; bioRxiv 2024.11.14.623630).** SAE features on ESM-2 align with up to 143 known biological concepts; single neurons align with far fewer (up to 46 neurons per layer across 15 concepts). Publication in Nature Methods is `[UNVERIFIED]`. https://arxiv.org/abs/2412.12101. A direct neurons-against-features comparison with annotated ground truth, and neurons lose. Caveat: ESM-2 is an older, smaller architecture, which is exactly what Cammarata says explains messy neurons.
16. **Goodfire and Arc Institute, Evo 2.** BatchTopK SAEs on layer 26 of a DNA model; features for coding sequences, protein secondary structure and so on (https://goodfire.ai/blog/interpreting-evo-2). **"Finding the Tree of Life in Evo 2" (Goodfire, 27 August 2025; https://www.goodfire.ai/papers/phylogeny-manifold):** evolutionary distance between 2,400+ bacterial species shows up as distance on a manifold in the model's activations, without the tree being supplied in training. An independent reproduction by ARENA 6.0 participants confirmed the qualitative result and found weaker correlations, and noted the species sample was under-specified (LessWrong/GreaterWrong, "Extracting performant algorithms using mechanistic..." series; seen in search summary). **This is the cleanest existing analogue of "phylogeny from model internals"**, in biology, where the true tree is known.
17. **Goodfire, "On optimism for interpretability"** (https://www.goodfire.ai/blog/on-optimism-for-interpretability) frames interpretability as a new science producing "natural ontologies". I could not confirm the phrase "interpretability as a way of knowing" on any Goodfire page: `[UNVERIFIED]` as a title. The Alzheimer's biomarker work with Prima Mente is a company-authored claim (https://www.goodfire.ai/research/interpretability-for-alzheimers-detection).
18. **Model diffing.** Anthropic's sparse crosscoders (Lindsey et al. 2024); Minder, Dumas et al., "Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning" (arXiv:2504.02922), which showed the original shared-versus-exclusive test produced artefacts; Anthropic's 2025 update noting that features exclusive to one model tend to be more polysemantic and dense (https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html). **Jiralerspong and Bricken, "Cross-Architecture Model Diffing with Crosscoders" (arXiv:2602.11729, February 2026)** is the first cross-architecture use, and its models include **Qwen3-8B and Llama-3.1-8B-Instruct, the study's two models** (which model was diffed against which was not checked beyond the abstract). It finds model-exclusive features: Chinese Communist Party alignment in Qwen3-8B, American exceptionalism in Llama-3.1-8B-Instruct. https://arxiv.org/abs/2602.11729

### D. The cheap rivals: computational narrative similarity, tropes, motifs

19. **Chaturvedi, Srivastava, Roth 2018 (doi:10.18653/v1/N18-2106)** and **Hatzel and Biemann, "Tell me again!" (LREC-COLING 2024)** are already cited (`study/paper/paper.md` lines 293–294).
20. **Hatzel and Biemann, "Story Embeddings — Narrative-Focused Representations of Fictional Stories" (EMNLP 2024, pp. 5931–5943; doi:10.18653/v1/2024.emnlp-main.339).** A Mistral-7B/E5 adapter trained contrastively on Tell-Me-Again so that retellings of one story land together; built to survive name replacement. **A stronger rival than TF-IDF and one the study has not faced.** A released model exists (uhhlt/story-emb).
21. **Hatzel, Artemova, Stiemer, Gius, Biemann, "SemEval-2026 Task 4: Narrative Story Similarity and Narrative Representation Learning" (arXiv:2604.21782).** More than 1,000 story-summary triples with human judgements of which story is closer to an anchor in theme, course of action and outcome; 71 submissions from 46 teams; LLM ensembles top the classification track. **A ready, human-labelled benchmark for exactly the quantity the study's distance matrix claims to measure.** https://arxiv.org/abs/2604.21782
22. **TV Tropes datasets.** Gala, Khursheed, Lerner, O'Connor, Iyyer, "Analyzing Gender Bias within Narrative Tropes" (NLP+CSS 2020; doi:10.18653/v1/2020.nlpcss-1.23): about 30,000 tropes, 1.9 million examples, linked to IMDb and Goodreads; code at github.com/dhruvilgala/tvtropes. Chang et al., "Situation and Behavior Understanding by Trope Detection on Films" (WWW 2021; arXiv:2101.07632): TiMoS, 5,623 synopses labelled with 95 tropes. Su et al., "Unveiling Narrative Reasoning Limits of Large Language Models with Trope in Movie Synopses" (Findings of EMNLP 2024; arXiv:2409.14324): trope detection stays hard for GPT-4; trope-wise querying adds 11.8 F1; chain-of-thought hallucinates tropes. García-Ortega, García-Sánchez, Merelo, "Tropes in films: an initial analysis" (arXiv:2006.05380) and a PLOS ONE 2021 follow-up finding 42 trope clusters. **Nobody was found aligning neurons or SAE features to TV Tropes labels** (arXiv API query for neurons + tropes + "language model": zero hits).
23. **Motif and tale-type classification.** Arčon, Robnik-Šikonja, Tratnik, "Large language models for folktale type automation based on motifs: Cinderella case study" (arXiv:2510.18561): an LLM answers yes/no per motif per tale, and the presence matrix is clustered. **This is the LLM-reader version of the study's meme-by-work grid, applied to the classic phylogenetics material.** Alyami and Finlayson, "Automated Motif Indexing on the Arabian Nights" (arXiv:2603.19283): 2,670 motif expressions of 200 motifs across 58,450 sentences, hand-annotated against El-Shamy's index. Eklund et al. 2023 (SVMs on 1,518 texts, 182 ATU types) and Yarlott and Finlayson 2016 from search summary only. No SAE or neuron work on folktales found (arXiv API: zero hits).
24. **Tanaka, "Mean-reverting self-excitation drives evolution: phylogenetic analysis of a literary genre, waka, with a neural language model" (Humanities and Social Sciences Communications 12, 19 March 2025; doi:10.1057/s41599-025-04714-1; Crossref-confirmed).** Builds phylogenetic networks from a neural language model's vector representation of poems from the eighth to sixteenth century; parent–child links agree significantly with the scholarly record of honkadori (allusive borrowing). **The nearest "phylomemetics" competitor, and it succeeded at what the study's trees failed at: recovering documented ancestry.** It uses dense vectors, has no named units, and poems are 31 syllables.
25. **Adjacent: PhyloLM (Yax, Oudeyer, Palminteri, arXiv:2404.04671)**, phylogenetic trees of 156 LLMs from their outputs. **Acerbi and Stubbersfield, PNAS 120(44) e2313790120, 2023**, transmission chains run through GPT-3. **Brinkmann et al., "Machine culture", Nature Human Behaviour 7, 1855–1868, 2023.** These treat the model as a cultural subject, not as an instrument.

### E. Responses to "Neural networks are already clean" (published 15 October 2026)

26. **Batutin, "We Tried to Confirm 'Neural Networks Are Already Clean.' The Controls Said No." (Substack, 19 October 2026).** Qwen3-0.6B and Qwen3-1.7B. Simulate-and-score correlation 0.26 at the extreme top of activation, 0.09 in the top 1%, about 0 below; 0 of 299 neurons clear the legibility bar; **a random-initialised network had 84% of neurons labelled "one job" against 52% for the trained one**; 0 of 50 neurons moved the tokens their label predicted when ablated or amplified. Verdict: the neuron basis is privileged and peaks are legible, but "already clean" overstates it and the labelling method has no noise floor. Small models, single seeds, a blog. https://andriybatutin.substack.com/p/we-tried-to-confirm-neural-networks
27. **LessWrong, "Why clean neurons are not enough"** (https://www.lesswrong.com/posts/rfnjfHCdKJTGgLP8L/why-clean-neurons-are-not-enough). Could not fetch the page (HTTP 401, mirror reset); from the search summary: a neuron explanation matters only if it predicts downstream behaviour, and the authors propose a fixed-budget comparison.
28. **AMX07/already-clean-replication (GitHub)**: small self-trained models; weak-against-strong activation ablation and the SwiGLU sign split. https://github.com/AMX07/already-clean-replication
29. **Neel Nanda's reaction**, from press summaries only: superposition is still real; an audit that looks only at the strongest activations overstates cleanliness if information hides in low-level firing. `[UNVERIFIED]` at primary source.
30. **No arXiv paper citing the essay was found** (arXiv API query for "already clean" + neurons + Cammarata: zero hits). Sources say Goodfire's technical paper was "maybe a few weeks" away. **The only work found that uses the essay to study culture is Mike's own repository** (github.com/hammer-mt/phylomemetics and phylomemetics-images, both returned by search).

### F. Where the study stands against this field (numbers from committed files)

- Word overlap finds source comics as well as neurons: TF-IDF cosine 11% of eligible comics beat the source, word presence 17%, against 16% (Qwen) and 10% (Llama) for neurons (`study/paper/paper.md` lines 9 and 187; underlying values in `study/results/experiments.json` under `lexical_comparators` and each model's `E2`).
- Residual test beyond words: Qwen p = 0.09 to 0.25, Llama p = 0.010 to 0.033 (`study/paper/paper.md` line 9 and Table 3; `study/results/experiments.json`, `E2.residual_test_beyond_words`).
- Trees do not recover documented sources: mean rank 5.4 (Qwen) and 5.3 (Llama) of 12 against 5.8 by chance and 2.9 for TF-IDF; every draw of random non-meme units beat the memes (`study/paper/paper.md` line 154).
- Identity: canon share AUC 0.52 to 0.54 against masked-hero films, TF-IDF centroid 0.80 (`study/paper/paper.md` lines 9 and 171).
- The paper's own future-work list already contains a head-to-head, planted memes and a fresh lineage (`study/paper/paper.md` lines 262–266). The competitors have already run the first two in their own settings.

## What this means for P1 and P2

**P1 (a meme can be measured by which neurons fire).**
- Split it in two. **P1a, "sparse interpretable units inside a model are usable as columns for describing texts"**: moderately well supported, by peer-reviewed work at ICML 2025 and 2026 with cost and accuracy benchmarks. **P1b, "the raw MLP neuron is that unit"**: weak and contested. Every competitor chose SAE features. Zou et al. 2026 and InterPLM give the reason with data. The early replications of the essay on small Qwen3 models find labels valid only at peak activation and a labelling method that calls random networks clean. The study's own cross-model result (twins mostly detect names and words) sides with the competitors.
- Condition under which P1b could still hold: large modern models (the essay's audit is on Qwen3-32B; the replications and the study are on 0.6B to 8B), and only at high activation. The study counts a unit as present only above a bar set by ordinary films (`study/paper/paper.md` line 61), so it works nearer the regime where labels are most defensible, but it used 8B models and has not checked any label against behaviour.
- Peng et al.'s position reframes the win condition. Sparse units beat baselines at **discovering and naming** concepts nobody specified, and lose or tie at **detecting** a specified relation. The Batman heredity test was a detection task, and it tied. The study's best asset by this reading is the catalogue of 14,843 families, the named memes and the name-free superhero template (`study/paper/paper.md` lines 7, 9 and 240), not the heredity p-values.
- No competitor has shown that internal units beat word overlap on a residual test. Movva and Jiang compare with LLM readers and dense embeddings, not with a regress-out-the-words control. So the study's negative result is not an outlier against positive competitor results; the competitors have not run that test.

**P2 (the measurement generalises because models converge).**
- Cross-architecture crosscoders on the study's own two models find **model-exclusive features that are culturally and politically loaded** (Jiralerspong and Bricken 2026). Convergence is partial, and the non-shared part is not random: it is where cultures differ. A memetics instrument has to report which model it used.
- Anthropic's diffing update says exclusive features tend to be denser and more polysemantic, and Minder et al. showed the first shared-versus-exclusive method produced artefacts. Unit mapping across models is an active methods problem, not a solved step.
- Evo 2's tree-of-life manifold is evidence that a model can encode a real phylogeny without being told it, but the independent reproduction was quantitatively weaker, and it is a geometry result, not a unit result.
- The competitors' default answer to "which model?" is to train the SAE on a text-embedding model (Movva) or one reader model (Jiang) and not worry about generality. Only this project makes cross-model agreement a test, which is a strength.

## Experiment seeds

Each names its cheap rival. Costs are rough guesses, not quotes.

1. **Unit bake-off on the existing Batman texts: neurons against SAE features against words.**
   - Question: is the tie with word overlap a property of the corpus or of the unit?
   - Data: the committed name-free texts and documented source pairs; public SAEs for Llama-3.1-8B (Llama Scope, arXiv:2410.20526, seen in search results; whether they fit the exact checkpoint the study used was not checked).
   - Method: rerun E2 and its residual test unchanged with SAE features in place of neuron poles.
   - Rival to beat: TF-IDF cosine and word presence (the existing comparators), plus story embeddings (Hatzel and Biemann).
   - Falsifier: SAE features also fail the residual test in the same way, which puts the blame on Wikipedia-summary wording and not on neurons; or SAE features pass where neurons fail, which falsifies P1b.
   - Cost: low. One GPU-day; no new data.

2. **Trope ground truth: do units detect human-labelled tropes?**
   - Question: do neuron or feature units recover TV Tropes labels better than words?
   - Data: TiMoS (5,623 synopses, 95 tropes); Gala et al. for scale.
   - Method: per trope, best single unit and best k-unit probe; report F1 against trope frequency.
   - Rival to beat: TF-IDF logistic regression at matched k, sentence-embedding probe, and LLM trope-wise querying (Su et al. 2024).
   - Falsifier: no trope where a small unit set beats TF-IDF at matched k; or unit–trope matches vanish when trope-name words are masked.
   - Cost: low to medium. Extraction over about 5,600 synopses on one 8B model.

3. **Narrative similarity against human judgement.**
   - Question: does meme-set distance track what people call "the same story"?
   - Data: SemEval-2026 Task 4 triples (over 1,000, human-labelled).
   - Method: Jaccard distance on unit sets picks the closer story; accuracy against annotators.
   - Rival to beat: TF-IDF, a generic sentence embedding, story-emb, and an LLM judge (the task's top systems).
   - Falsifier: unit distance at or below TF-IDF accuracy.
   - Cost: low. Short texts, public labels.

4. **Cross-language heredity, where word overlap is zero by construction.**
   - Question: do units carry a story across languages when no words are shared?
   - Data: Tell-Me-Again (summaries of the same work from several Wikipedia language editions).
   - Method: match each summary to its same-work counterpart in another language by unit overlap, using multilingual models (Qwen3).
   - Rival to beat: machine-translate then TF-IDF; multilingual sentence embeddings; an LLM reading both texts.
   - Falsifier: units do not beat translate-then-TF-IDF. If the cheap pipeline wins here too, the "beyond words" claim is dead for summaries.
   - Cost: medium. Translation calls plus extraction over a few thousand pairs.

5. **Planted-meme recovery on simulated trees (Movva-style synthetic truth).**
   - Question: on a lineage with known topology and known trait transmission, which method recovers the tree and the traits?
   - Data: LLM-written stories evolved down a simulated tree with planted tropes, reworded at each step, plus decoys that use a trope's words without the trope (already sketched in `study/paper/paper.md` line 263).
   - Method: recover topology (Robinson–Foulds distance) and trait presence (F1).
   - Rival to beat: TF-IDF, story-emb, and an LLM-coded yes/no motif matrix in the manner of Arčon et al.
   - Falsifier: units do not beat the LLM-coded matrix on trait F1, or do not beat TF-IDF on topology once rewording is heavy.
   - Cost: medium. A few thousand generated stories.

6. **Shared against exclusive features as the cross-model unit.**
   - Question: do memes live in the features two models share, or in the ones only one has?
   - Data: the Batman corpus; a cross-architecture crosscoder for Qwen3-8B and Llama-3.1-8B (Jiralerspong and Bricken; availability of weights not checked).
   - Method: replace correlation twins with crosscoder shared features; repeat the with-names and name-free agreement tests.
   - Rival to beat: the study's word-matched null (units as well reproduced by a single word).
   - Falsifier: shared-feature agreement collapses without names as the twins did.
   - Cost: medium to high if a crosscoder must be trained; low if released weights exist.

7. **Lineage with an independent ancestry record: honkadori.**
   - Question: can unit inheritance recover documented allusion where a dense-vector method already has?
   - Data: the waka database and honkadori record used by Tanaka 2025.
   - Method: E2-style source ranking with units.
   - Rival to beat: character n-gram overlap, Tanaka's vectors, and Ogawa et al.'s fine-tuned waka embeddings (ALP 2025, from search summary).
   - Falsifier: units rank documented sources no better than n-gram overlap.
   - Cost: medium. Needs data access and a model competent in classical Japanese; that competence is itself unverified.

## Key sources

| Citation | URL / DOI | Why it matters | Priority |
|---|---|---|---|
| Jiang, Sun, Dunlap, Smith, Nanda 2025, Interpretable Embeddings with Sparse Autoencoders | https://arxiv.org/abs/2512.10092 | Closest method competitor: internal units as corpus-analysis columns, with cost benchmarks | A |
| Peng, Movva, Kleinberg, Pierson, Garg 2025/2026, Use SAEs to Discover Unknown Concepts | https://arxiv.org/abs/2506.23845 | Predicts the Batman tie; says where units should win | A |
| Zou, Jiao, Hu 2026, Deciphering Cultural Representations via SAEs | doi:10.18653/v1/2026.findings-acl.278 | States that neurons conflate culture with lexical cues; uses causal tests | A |
| Liu, Joshi, Dawson, Piper 2026, Language Models Have Interpretable Narrative Mechanisms | https://arxiv.org/abs/2610.12989 | Cultural analytics reaches neurons and features for narrative, on Llama-3.1-8B, with ablation | A |
| Tanaka 2025, phylogenetic analysis of waka with a neural language model | doi:10.1057/s41599-025-04714-1 | Nearest phylomemetics competitor; recovered documented ancestry | A |
| Jiralerspong and Bricken 2026, Cross-Architecture Model Diffing with Crosscoders | https://arxiv.org/abs/2602.11729 | Covers both of the study's models; finds culturally loaded exclusive features | A |
| Batutin 2026, We Tried to Confirm "Neural Networks Are Already Clean" | https://andriybatutin.substack.com/p/we-tried-to-confirm-neural-networks | First replication attempt with controls; random-init result | A |
| Movva et al. 2025, Sparse Autoencoders for Hypothesis Generation | https://proceedings.mlr.press/v267/movva25a.html | Synthetic-truth recovery design; already cited | B |
| Hatzel et al. 2026, SemEval-2026 Task 4 | https://arxiv.org/abs/2604.21782 | Human-labelled narrative similarity benchmark | B |
| Hatzel and Biemann 2024, Story Embeddings | doi:10.18653/v1/2024.emnlp-main.339 | Name-robust rival stronger than TF-IDF | B |
| Chang et al. 2021, TiMoS; Su et al. 2024 | https://arxiv.org/abs/2101.07632 ; https://arxiv.org/abs/2409.14324 | Trope ground truth and the LLM-reader baseline on it | B |
| Gala et al. 2020, TV Tropes dataset | doi:10.18653/v1/2020.nlpcss-1.23 | 30K tropes, 1.9M examples | B |
| Arčon, Robnik-Šikonja, Tratnik 2025, LLM motif detection, Cinderella | https://arxiv.org/abs/2510.18561 | LLM-coded motif matrix: the reader rival on folktale material | B |
| Goodfire 2025, Finding the Tree of Life in Evo 2 | https://www.goodfire.ai/papers/phylogeny-manifold | Phylogeny recovered from activations where the tree is known | B |
| Simon and Zou 2024, InterPLM | https://arxiv.org/abs/2412.12101 | Neurons against SAE features with annotated ground truth | B |
| Namazifard and Galke 2025; Yamamoto et al. 2025 | https://arxiv.org/abs/2508.02241 ; https://arxiv.org/abs/2510.08284 | Prior "culture neuron" work | C |
| Zhong et al. 2024, natural-language-parameter models | https://arxiv.org/abs/2409.08466 | Formal version of the LLM-reader rival, with a time-series form | C |
| Tamkin et al. 2024, Clio | https://arxiv.org/abs/2412.13678 | LLM reader plus clustering at a million documents | C |
| Alyami and Finlayson 2026, Motif Indexing on the Arabian Nights | https://arxiv.org/abs/2603.19283 | Sentence-level motif ground truth | C |
| Minder et al. 2025; Anthropic crosscoder update 2025 | https://arxiv.org/abs/2504.02922 ; https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html | Artefacts in shared-versus-exclusive feature tests | C |
| Yax, Oudeyer, Palminteri 2024, PhyloLM | https://arxiv.org/abs/2404.04671 | Phylogenetics applied to models themselves | C |

## Fog

- I could not read the LessWrong post "Why clean neurons are not enough" directly; my account of it rests on a search summary.
- Neel Nanda's objection is from press summaries, not his own text.
- Goodfire's technical companion to the essay: not found. If it has appeared since late October it would change section E.
- "Interpretability as a way of knowing": no Goodfire page with that title was confirmed.
- InterPLM in Nature Methods and Evo 2 in Nature: stated by Goodfire's site and a secondary news item; journal pages not opened.
- Novelty verdicts rest on absence of hits in about 30 web and arXiv queries over roughly 15 minutes. That is evidence of absence at blog-and-arXiv depth only. Not searched: Google Scholar citations of the Movva and Jiang papers, the Journal of Cultural Analytics, CHR and NLP4DH 2026 proceedings (one NLP4DH 2026 trope paper, aclanthology 2026.nlp4dh-1.13, surfaced and was not opened).
- Whether the Jiralerspong and Bricken crosscoder weights are public was not checked.
- Whether any competitor's result would survive a regress-out-the-words residual test is unknown; none reports one.
- The Batutin replication is on 0.6B and 1.7B models. Whether its negative controls transfer to 8B or 32B is open, and it is the single most useful thing to know for P1b.
- Matthews et al. NAACL 2025 and Eklund et al. 2023 details come from search summaries only.
