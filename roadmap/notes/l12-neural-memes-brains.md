# Memes as neural patterns, in brains and in models (l12-neural-memes-brains)

Session date 2026-11-08. Web research and file reading only; nothing was run. Sources marked `[UNVERIFIED]` were not confirmed in this session.

## Bottom line

- **The theory wants brain states; the study measures something else.** Aunger (2002) and McNamara (2011) put the meme inside a nervous system (the "neuromeme", the "i-meme") and treat texts as its external form ("e-meme"). A unit firing in a language model is a third thing: the internal state of one artificial reader exposed to an e-meme. Nobody has shown it tracks an i-meme. P1 is an instrument claim, and the bridge to human minds is an untested assumption.
- **Brain-to-model alignment is real but coarse, and it is under attack.** Schrimpf 2021, Goldstein 2022 and Caucheteux & King 2022 show linear maps from model layers to voxels or electrodes. Feghhi 2024 and Hadidi 2025 show most of that fit is reproduced by word rate, sentence position and static word embeddings (over 80% of GPT2-XL's explained variance in each of three datasets: over 85%, over 80% and nearly 100%; Hadidi et al. 2026, Nat Commun 17:5769, checked against the paper text). None of it is at the grain of one unit to one concept.
- **Tuckute 2024 is the strongest causal result, and it is not about concepts.** Model-chosen sentences drove and suppressed the human language network in new participants, but the drivers were surprisal and well-formedness, not which idea a sentence carried.
- **Human checks on model units are thin and mostly discouraging for raw neurons.** Top-example coherence is what gets rated, by a few experts or by another LLM. Huang 2023 found the best auto-explanations have high error rates and little causal effect; Heap 2025 found untrained random transformers yield latents that score about as "interpretable" as trained ones; Bolukbasi 2021 showed the concept you read off top examples changes with the corpus. Cammarata's 90% figure is an expert audit of a 32B model on small samples, not a blind human agreement study.
- **The Batman study has no human validation at all.** Labels were written by an AI, judged by AIs (`study/paper/paper.md` lines 75, 254, 308). The blind test beat a word-overlap judge overall (79% against 65%) but not for the 31 named memes (77% against 63%, p = 0.065; `study/results/robustness/label_test.json`).
- **Where the stand-in breaks:** names (the strongest units are token detectors, unlike invariant concept cells), concepts the model has not learned (new memes have no unit by construction), and culture (models read like WEIRD English speakers; human concepts vary even within one culture, Marti 2023). The first experiment to run is a human intruder and presence-rating study with a word-overlap rival built in.

## Findings

### 1. Memes as brain states: what the theory actually claims

- **Aunger, *The Electric Meme* (2002, Free Press).** Proposes "neuromemetics": a meme is a self-replicating electrical state at a node in the brain, where a node may be one neuron in a state or a set of linked neurons. Memes must be replicators first; artefacts and signals are not memes but the means by which one brain's state instigates a similar state in another. Reviews were mixed: Kirkus is sceptical ("skeptical readers ... will probably remain unconvinced"), the Wilson Quarterly review by Jay Kirk is favourable ("captivating if sometimes challenging"). That they noted the lack of empirical work is `[UNVERIFIED: neither linked review says so in terms]`. Kirkus confirms the publisher and date (Free Press, July 2002, ISBN 0-7432-0150-7), the "node" definition, "replicators first" and the "instigator signals" account. Sources: Kirkus review https://www.kirkusreviews.com/book-reviews/robert-aunger/the-electric-meme/print ; Wilson Quarterly https://www.wilsonquarterly.com/quarterly/_/electric-meme-new-theory-how-we-think-and-communicate/ . I read reviews, not the book.
- **McNamara, "Can we measure memes?" (2011), Frontiers in Evolutionary Neuroscience 3:1, doi:10.3389/fnevo.2011.00001.** A Perspective article, not an experiment. Splits memes into i-memes (held as a neural substrate) and e-memes (held in the environment). Argues neuroimaging can quantify changes in functional connectivity profiles as people perceive, learn, memorise and imitate ("replicate") memes, and that an inexact definition does not stop controlled experiments, as genes were measured through phenotypes before anyone found them on chromosomes. Blunt reading: 15 years on, I found no programme of imaging studies that took this up. The proposal was never executed as a programme. Correction from the check: the article itself points to one earlier fMRI study by its author as a proof of concept, McNamara, Buccino, Menz, Gläscher, Wolbers, Baumgärtner, Binkofski (2008), "Neural dynamics of learning sound-action associations", PLoS ONE 3:e3845, doi:10.1371/journal.pone.0003845, which it describes as "specifically designed as an exercise in creating artificial memes" (learning novel sound-action pairings, with a PPI connectivity analysis). So one small study exists; a follow-on programme was not found `[UNVERIFIED: absence rests on a short search]`. Published 25 May 2011; the gene-and-phenotype argument is in the full text.
- **Dennett.** In *From Bacteria to Bach and Back* (2017) words are the first and best memes, and a meme is any informational way of behaving passed between people. A critical paper in *Avant* (Grabarczyk, "Can memes explain the birth of comprehension?", AVANT 10(3), 2019, doi:10.26913/avant.2019.03.29) argues he never gives identity conditions for a meme, so transmission stays vague `[UNVERIFIED: author, title and DOI confirmed on Crossref; the argument was not checked against the paper]`. Nature review: Dan Jones, "Cognitive science: Dennett rides again", Nature 542:30 (2017), doi:10.1038/542030a, https://www.nature.com/articles/542030a . This matters for the study: if words are the paradigm memes, then "word overlap matches neurons" is close to what Dennett would predict, and is not an embarrassment for memetics, only for the claim that neurons add something.
- **Hofstadter.** "Metamagical Themas", Scientific American, January 1983, subtitle "Virus-like sentences and self-replicating structures" (https://www.scientificamerican.com/article/metamagical-themas-1983-01/ ; reprinted in the 1985 book, https://en.wikipedia.org/wiki/Metamagical_Themas ). Secondary sources say it dissects a meme into bait and hook and co-memes and stresses competition for mind-space. I confirmed only the title and date, not the content. Hofstadter's own view of concepts (active, analogy-making, blurry symbols) is an argument against single-unit concepts `[UNVERIFIED: from memory, not checked this session]`.
- **Marti, Wu, Piantadosi, Kidd, "Latent diversity in human concepts" (2023), Open Mind 7:79-92, doi:10.1162/opmi_a_00072.** From similarity and feature judgements they estimate at least ten to thirty quantifiably different variants of meaning for even common nouns, and people wrongly believe others share their semantics. The paper cites this as its reference [16] (`study/paper/paper.md` line 238). Caveat: a 2023 CogSci paper by Duan and Lupyan asks whether such divergence is principled or noise `[UNVERIFIED beyond the search summary]`.

### 2. Concept cells: the nearest thing to a "meme neuron" in a brain

- **Quian Quiroga, Reddy, Kreiman, Koch, Fried (2005), Nature 435:1102-1107, doi:10.1038/nature03687.** Single neurons in the human medial temporal lobe respond selectively and invariantly to a specific person or place across different photos, drawings and the written name (the "Jennifer Aniston" and "Halle Berry" cells).
- **Quian Quiroga (2012), "Concept cells: the building blocks of declarative memory functions", Nat Rev Neurosci 13:587-597, doi:10.1038/nrn3251.** Frames them as memory units that encode the meaning of a stimulus (abstract: a "sparse, explicit and abstract representation" that is "crucial for memory functions"), and may stop encoding a concept that becomes irrelevant `[UNVERIFIED: not in the abstract]`. A 2026 update: Arruda & Quian Quiroga, "Concept cells and the neural bases of human memory", Acta Physiologica 242(6):e70234, doi:10.1111/apha.70234.
- **Two defining criteria: selectivity and invariance.** A Nature paper, Bausch, Niediek, Reber, Mackay, Boström, Elger, Mormann, "Distinct neuronal populations in the human brain combine content and context", Nature 650:690-700, doi:10.1038/s41586-025-09910-2 (published online 7 January 2026, so 2026 and not 2025; Mormann lab; 3,109 neurons from 16 neurosurgical patients; copy at https://sonar.unidistance.ch/documents/335982/files/Distinct-neuronal-populations-in-the-human-brain-combine-content-and-context.pdf ) reports concept cells stay largely context-invariant while separate populations carry context.
- **Coding is sparse, not one-cell-one-concept.** Each concept is carried by many cells and each cell by several related concepts `[UNVERIFIED: standard reading of Quiroga's work, the exact estimates were not checked this session]`.
- **Jamali et al. (2024), Nature 631:610-616, doi:10.1038/s41586-024-07643-2.** Single cells in left prefrontal cortex respond selectively to word meanings during story listening, depend on sentence context, and as ensembles decode broad semantic categories. So single-unit semantic selectivity exists in humans during language, at the level of broad categories (not "the origin wound").
- **The explicit bridge to models already exists and came from P1's author.** Goh, Cammarata et al., "Multimodal neurons in artificial neural networks", Distill 2021, doi:10.23915/distill.00030, compared CLIP units directly with the Halle Berry cell: one unit fires for a spider photo, the word "spider" and Spider-Man in costume or drawing. This is a model, a vision-language one, passing the invariance test. It was an analogy by inspection; no brain data were collected. Check: authors are Goh, Cammarata, Voss, Carter, Petrov, Schubert, Radford, Olah, Distill 6(3), March 2021; the page cites Quian Quiroga 2005 and names the "Spider-Man" neuron; the exact "Halle Berry" wording could not be text-checked because the page body is script-rendered.
- **I found no study comparing human concept-cell recordings with language-model units.** One conference abstract compares temporal-pole concept cells with "embeddings"; I could not tell whether they came from a language model `[UNVERIFIED: no reference given, not located in the check]`.

### 3. Language models and human brains: alignment, and what drives it

- **Schrimpf et al. (2021), PNAS 118(45), "The neural architecture of language: Integrative modeling converges on predictive processing"** (article e2105646118, doi:10.1073/pnas.2105646118; open copy https://hdl.handle.net/1721.1/138214). Best transformers predicted nearly 100% of explainable variance in responses to sentences; brain score tracked next-word prediction accuracy.
- **Goldstein et al. (2022), Nat Neurosci 25:369-380, "Shared computational principles for language processing in humans and deep language models"** (doi:10.1038/s41593-022-01026-4; the lab PDF link first given, https://hassonlab.princeton.edu/sites/g/files/toruqf3591/files/documents/Goldstein_et_al_NN_2022.pdf, returned 404 at the check). ECoG, nine participants, 30-minute podcast: pre-onset prediction, post-onset surprise, contextual embeddings.
- **Goldstein et al. (2024), Nat Commun 15:2768, doi:10.1038/s41467-024-46631-y.** "Brain embeddings" in inferior frontal gyrus (three participants) share geometry with contextual embeddings; a left-out word's brain pattern can be predicted zero-shot from its relations to other words. Geometry-level evidence, n = 3.
- **Caucheteux & King (2022), Commun Biol 5:134, doi:10.1038/s42003-022-03036-1.** 102 subjects, fMRI and MEG, 400 sentences: similarity to the brain tracks a model's ability to predict words from context; convergence is "partial". **Caucheteux, Gramfort & King (2023), Nat Hum Behav 7:430-441, doi:10.1038/s41562-022-01516-2:** 304 people, longer-range and multi-level predictions improve the fit.
- **Tuckute et al. (2024), Nat Hum Behav 8:544-561, doi:10.1038/s41562-023-01783-7.** An encoding model built on GPT2-XL from fMRI responses to 1,000 sentences picked new sentences predicted to drive or suppress the language network; they did, in participants not used to fit the model. Surprisal and well-formedness were the main drivers. This is closed-loop, out-of-sample control of a human brain region by a model, which is the right standard of proof. But the controlled quantity is overall network response strength, not a concept.
- **Which model units predict brains.** AlKhamissi, Tuckute, Bosselut, Schrimpf, "The LLM Language Network", arXiv:2411.02280 (NAACL 2025): applying the neuroscience localiser (sentences against non-words) to 18 LLMs finds language-selective units; ablating 1% of them causes large language deficits where random units do not; these units align better with human language-network recordings than random units. So some units matter more for brain alignment, but selected by a contrast (language vs not), not by concept.
- **2026: feature-level encoding models.** Lepori, Kay & Tuckute, "Interpreting brain responses to language with sparse features from language models", arXiv:2606.06857 (June 2026): SAE features predict voxel responses about as well as dense hidden states, with about 18 Matryoshka SAE features per voxel (confirmed in the paper text: "on average, 18 Matryoshka SAE features were selected per voxel"; the language model is gemma-2-2b), 7T fMRI, eight participants, 200 sentences. Guo, Wu & Yiu, "Sparse autoencoders map brain-LLM alignment onto cortical semantic topography", arXiv:2605.23035 (May 2026, accepted at CoNLL 2026): GPT-2 XL and Llama-3.1-8B, semantic features alone recover about 94% of peak encoding performance (confirmed in the abstract; r = 0.285 against 0.304), across English, Chinese and French. A NeurIPS 2026 oral (Alicia Zeng, "Learning Sparse Semantic-Cortical Atoms for Multisubject Naturalistic fMRI Encoding", https://neurips.cc/virtual/2026/oral/151713) keeps 88% of dense performance with 50 atoms (confirmed on the NeurIPS page; story-listening fMRI). I read abstracts and summaries only. These are the closest existing work to "a model feature stands for a concept in a brain", and note Llama-3.1-8B is one of the study's own models. They use SAE features, not raw neurons, and voxels, each averaging hundreds of thousands of neurons.
- **The critics.** Antonello & Huth (2024; first online February 2023), Neurobiology of Language 5(1):64-79, doi:10.1162/nol_a_00087: the fit does not favour predictive coding; models fit brains because they capture many linguistic features at once. Feghhi, Hadidi, Song, Blank, Kao, arXiv:2406.01538 (2024): with shuffled train-test splits a trivial temporal-autocorrelation feature beats LLMs; untrained models' scores are explained by sentence length and position; trained models' scores largely by length, position and static word embeddings. Hadidi, Feghhi, Song et al. (2025), bioRxiv doi:10.1101/2025.03.09.642245 (a search result lists a Nature Communications version, doi:10.1038/s41467-026-72253-7, which I did not open): positional signals and word rate fully account for untrained models, and with GloVe account for over 80% of GPT2-XL's explained variance (summary figure). A 2026 cross-lingual study (arXiv:2605.21049) finds alignment robust across Mandarin, English and French but uninformative about shared computation.
- **This is the same result as the Batman study, in another field.** In both, a headline correspondence shrinks to a small residual once word identity and position are controlled. The brain-score critics' static-embedding baseline is the neuroscience twin of the study's word-overlap baseline.

### 4. Human behavioural checks on model units

- **Raw neurons, as rated by people.** Cammarata's essay (saved copy a saved copy of the essay, lines 63 and 69) reports that Anthropic's 2023 "Towards Monosemanticity" gave the median neuron 0 of a possible 14 on its human rubric, and that OpenAI's 2023 work found about 1.7% of neurons interpretable by its most generous measure. The same essay (lines 70, 76) then claims roughly 90% of neuron poles in Qwen3-32B have one main job, "in preliminary audits", on "relatively small samples". It gives no blind-rater protocol or inter-rater agreement that I could find. It also says both human and automated interpreters are biased towards reading neurons as groups of nouns (line 212) and that neurons are better seen as control flow than as nouns (line 910). So P1's source rests on expert judgement by the people making the claim.
- **Bills et al. (2023), "Language models can explain neurons in language models"**, OpenAI (https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html): GPT-4 explanations for about 300K GPT-2 XL neurons, scored by simulation.
- **Huang, Geiger, D'Oosterlinck, Wu, Potts (2023), "Rigorously assessing natural language explanations of neurons"**, arXiv:2309.10312 (BlackboxNLP 2023): even the most confident of those explanations have high error rates and little to no causal efficacy. They also question whether single neurons and natural-language labels are the right level.
- **Bolukbasi et al. (2021), "An interpretability illusion for BERT"**, arXiv:2104.07143: a unit can appear to encode one simple concept on one corpus and a different one on another; the cause is embedding geometry plus narrow corpora. Their advice: test on several datasets. The Batman labels were read off one franchise corpus, the exact condition the paper warns about.
- **Heap et al. (2025), "Sparse autoencoders can interpret randomly initialized transformers"**, arXiv:2501.17727: SAE latents from untrained transformers score about as interpretable as those from trained ones under an open auto-interp pipeline. Their hypothesis: the structure is in the input text. Top-example coherence can be produced by token statistics alone. This is the word-overlap problem stated as an interpretability result.
- **Automated scores against humans.** Paulo, Mallen, Juang, Belrose, "Automatically interpreting millions of features in LLMs", arXiv:2410.13928 (ICML 2025): fuzzing correlates best with human judgements (rho = 0.69), simulation 0.60, detection 0.59 (figures from a summary). Paulo & Belrose, "Evaluating SAE interpretability without explanations", arXiv:2507.08473 (2025): an intruder-detection test adapted from topic modelling; humans averaged 64% intruder accuracy over 105 latents, and Claude Sonnet 3.5 correlated 0.84 with humans on 56 latents (secondary summary; not checked against the paper). Humans at 64% on a five-way task is far from "people agree the examples share a concept".
- **Crowd ratings.** Marks et al., "Sparse feature circuits", arXiv:2403.19647: crowdworkers rated features 0 to 100; SAE features scored above neurons; the authors note annotator choice influenced the result.
- **Vision psychophysics, the best-controlled human evidence.** Zimmermann, Klein, Brendel, "Scale alone does not improve mechanistic interpretability in vision models", arXiv:2307.05471 (NeurIPS 2023): over 130,000 human responses on 767 units in nine models; none was easier for people to interpret than GoogLeNet. Earlier work in the same line by Borowski et al. (2021) `[UNVERIFIED this session]`. No language-model study of this scale and rigour turned up.
- **What these tests measure.** They ask whether people (or an LLM) can find a common thread in a unit's strongest examples. They do not test the weak and middle activations where most firing happens, the causal role, or whether a reader of the text would report the concept unprompted. And any set of sentences sharing a word passes.
- **The study's own check** (`study/results/robustness/label_test.json`; `study/paper/paper.md` line 226). Blind AI judges: 56 of 71 (accuracy 0.7887), word-overlap judge 0.6479, p = 0.0074. Named 31: 24 of 31 (0.7742) against 0.629, p = 0.065. My arithmetic on that file: the word-overlap judge was tied on 48 of 71 items (scored 0.5 each), so on the 23 items where it could decide it was right 22 times. When the label's words appear in a sentence, word matching is nearly perfect; the AI judges' margin comes from items where no label word appears. No human judged anything.

### 5. Names, culture and unlearned concepts

- **Names.** The study's 33 strongest Qwen memes have Llama twins at median r = 0.82, but 24 of 33 are name or place detectors, a single word reproduces their profile at median r = 0.85, and on name-free text the twins fall to a median of 0.14, no better than random units (`study/paper/paper.md` line 126; underlying file `study/results/robustness/nulls.json`, key `f2_agreement`, which I did not open). A concept cell fires to the name too, but also to the face and voice. The study's name units fail Quiroga's invariance criterion as tested.
- **Unlearned concepts.** Ferrando, Obeso, Rajamanoharan, Nanda, "Do I know this entity?", arXiv:2411.14257 (ICLR 2025): models carry features that signal whether an entity is known or unknown, and steering them flips refusal and hallucination. A model has an explicit "I have no representation of this" state. A meme newer than the training data, or too rare to be learned, cannot have its own unit. For a science of cultural change, where novelty is the object, that is a structural gap. The reverse problem also holds: the model reads 1939 Batman already knowing 2022 Batman.
- **Culture.** Thompson, Roberts & Lupyan (2020), Nat Hum Behav 4:1029-1038, doi:10.1038/s41562-020-0924-8: across 41 languages word meanings align better between culturally and historically closer languages; number and kinship align well, animals, food and emotions poorly. Atari, Xue, Park, Blasi, Henrich, "Which humans?" (2023 preprint, https://www.hks.harvard.edu/centers/cid/publications/which-humans): LLM responses resemble WEIRD populations and similarity falls with cultural distance (r = -0.70). A model unit is one reader with a particular cultural position.
- **Behavioural concept alignment.** Du et al. (2025), Nat Mach Intell, doi:10.1038/s42256-025-01049-z (arXiv:2407.01067): 4.7 million odd-one-out triplets from LLMs over 1,854 objects give low-dimensional embeddings similar but not identical to human ones. This is population-level similarity structure for concrete objects, the easy case.

### 6. Verdict: how far a firing neuron stands in for a concept in human minds

| Claim | Status |
|---|---|
| Model populations and human language cortex share linear structure | Supported, repeatedly; much of it explained by word-level and positional features |
| A model can pick text that changes human brain responses out of sample | Supported once, strongly (Tuckute 2024), for response magnitude, not concepts |
| Sparse model features map onto cortical semantic topography | Emerging (2026 preprints), voxel-level, SAE features not neurons |
| People agree a neuron's top examples share a concept | Weakly supported; raw neurons historically scored poorly; scores are confounded by token overlap and corpus |
| One model neuron = one human concept | No direct evidence. Human concepts vary between people; brain concept coding is sparse and distributed; no unit-to-unit brain comparison exists |
| The study's neuron memes match what human readers perceive | Untested. Zero human data in the study |

A firing unit is defensible as a reproducible measuring device for text. It is not yet defensible as a stand-in for a concept in human minds. The strongest honest claim today: "a fixed model is one consistent reader, and its units are that reader's features".

## What this means for P1 and P2

**P1 (a meme can be measured by which neurons fire).**
- The memetics literature that P1 inherits (Aunger, McNamara) defines the unit in a brain. P1 quietly swaps the brain for a model. That swap needs its own evidence and has none in the study.
- The evidence that model units are human-readable concepts is expert audit (Cammarata 2026, 32B model, small samples) against a body of controlled work that is more negative for raw neurons (Huang 2023; Zimmermann 2023; the 2023 rubric results Cammarata himself cites). The study used 8B and 1.7B 4-bit models, smaller than the audited one (`study/paper/paper.md` line 254).
- Heap 2025 and Bolukbasi 2021 predict the study's main embarrassment: top-example coherence can come from token statistics, so word overlap will match neurons. The study's blind test for its 31 named memes did not beat word overlap (p = 0.065, `study/results/robustness/label_test.json`).
- Conditions under which P1 looks most likely to hold: concepts that are frequent in training data, expressible without a fixed name, read in the model's dominant language and culture, in a model large enough for clean units, and validated by steering plus human judgement. Conditions where it fails: named entities (token detectors), novel or rare memes, culture-specific readings.
- P1 can be rescued in a weaker form: the model is a standardised reader. Then validity is an empirical question about agreement with human readers, to be measured against the human-to-human ceiling (Marti 2023 says that ceiling is lower than people assume).

**P2 (models converge, so the measure generalises).**
- Brain-alignment work supports convergence at the level of populations and geometry, the same level as the Platonic-representation claim. The critics show that level of agreement can be carried by word identity, word rate and position. Shared geometry is cheap; shared units are not shown.
- The study agrees: cross-model twins are strong for names and words and collapse without names (`study/paper/paper.md` line 126). The 2026 cross-lingual brain results point the same way: robust alignment that says little about shared computation.
- Human concept diversity limits what "universal" can mean. If people hold 10 to 30 variants of a common noun, model convergence may reflect a shared lexical-statistical core in shared training text, not convergence on the concept.
- A positive sign for P2: Guo et al. 2026 report semantic SAE features aligning with cortex across three languages and two model families, including Llama-3.1-8B. If it holds up it is convergence between models and brains on semantic features, though at SAE-feature and voxel level.

## Experiment seeds

1. **Human intruder test on the study's units.**
   - Question: can naive people tell which sentence does not belong among a unit's strong examples, better than they can for word-defined sets?
   - Data: the study's corpus; 100 units (31 named memes plus a random sample), original and name-free text.
   - Method: Paulo & Belrose's design: four activating sentences from different works plus one intruder; five or more raters per item; fans and non-fans as separate groups; report inter-rater agreement.
   - Rival to beat: sets of the same size built from one shared word stem (word overlap), from a TF-IDF cluster, and from a sentence-embedding cluster. Also an LLM doing the same task.
   - Falsified if: neuron sets are no easier than word-stem sets, or the advantage vanishes on name-free text.
   - Cost: about 2,500 judgements; a few hundred dollars on a panel; two days.

2. **Does the unit predict what human readers say is present?**
   - Question: when people read a passage, does the neuron's activation predict whether they say "the secret identity" (and each named meme) is there?
   - Data: 300 to 500 passages sampled across the activation range, including the middle; 31 named memes.
   - Method: yes/no and 1 to 5 ratings from several readers each; score each predictor by AUC against the majority vote; report human-to-human agreement as the ceiling.
   - Rival to beat: keyword match on the label, TF-IDF similarity to the label note, sentence-embedding similarity, and an LLM simply reading the passage and answering the same question. The last is the hard one.
   - Falsified if: neuron AUC is at or below the keyword or LLM-reader AUC, or far below the human ceiling.
   - Cost: under 1,000 dollars; one week.

3. **Invariance test for name units (the concept-cell criterion).**
   - Question: does a character unit fire on the character without the name?
   - Data: for each of the roughly 24 name units, hand-written sets: aliases (the Dark Knight, the Caped Crusader), name-free descriptions, the name in other languages, the same token in an unrelated sense ("fox", "joker" as a playing card, "robin" the bird).
   - Method: reading-only pass; compute selectivity and invariance indices as Quiroga does. For a vision-language model, add images (ties to `images/`).
   - Rival to beat: a token-match dictionary; embedding similarity to the character's Wikipedia lead.
   - Falsified if: units fire on the homonym and not on the description. The Lucius Fox unit firing on Zorro's "Fox" (`study/paper/paper.md` line 122) already suggests this outcome.
   - Cost: a day of writing stimuli, minutes of compute.

4. **Novel-meme test.**
   - Question: can a frozen model's units register a meme invented after training?
   - Data: 20 invented tropes and characters, each written into 10 short story summaries by people or an LLM, with matched controls; plus real memes dated after the model's cutoff.
   - Method: look for any unit, or small set, that separates carriers from controls; compare with humans who learn the trope from three examples.
   - Rival to beat: a centroid of sentence embeddings from three examples; an LLM given three examples in context; word overlap.
   - Falsified (for P1 as a general measure) if: no stable unit emerges while the rivals and humans succeed. Then neuron memetics is limited to memes the model already knows.
   - Cost: two to three days.

5. **Retelling chains: is a neuron-defined meme a replicator in people?**
   - Question: do units' contents survive human retelling, which is what makes something a meme under Aunger and McNamara?
   - Data: 20 name-free summaries; chains of four retellers each (Bartlett-style serial reproduction).
   - Method: run the retellings through the model; survival of a unit = still above threshold at generation four; predict survival from first-generation activation.
   - Rival to beat: word frequency, concreteness and surprisal of the sentences; TF-IDF salience; an LLM asked "what will people remember from this?".
   - Falsified if: activation predicts survival no better than those.
   - Cost: 1,000 to 2,000 dollars; two to three weeks. This is the only seed that touches the i-meme.

6. **Brain check on public fMRI, no new scanning.**
   - Question: do concept-labelled units explain cortical responses to stories beyond word-level features?
   - Data: open narrative-listening fMRI sets `[dataset names UNVERIFIED this session]`; Llama-3.1-8B, as in Guo et al. 2026.
   - Method: voxelwise encoding with contiguous train-test splits, following Feghhi and Hadidi; variance partitioning.
   - Rival to beat: word rate plus position plus static word embeddings (their confound set); dense hidden states; SAE features.
   - Falsified if: units add no unique variance over the confound set.
   - Cost: compute only, one to two weeks, needs someone who knows fMRI encoding pitfalls.

7. **Cross-cultural readers.**
   - Question: do units agree with readers from different cultures equally well?
   - Data: seed 2's passages translated; reader panels in two or three countries; Qwen (Alibaba) against Llama (Meta).
   - Method: model-to-human agreement per culture; test whether agreement falls with cultural distance as in Atari et al.
   - Rival to beat: multilingual sentence embeddings; back-translation word alignment (Thompson's method).
   - Falsified (for universality) if: agreement drops sharply outside the model's dominant culture, or the two models split along it.
   - Cost: 2,000 dollars or more; a month. Lower priority until seeds 1 and 2 are in.

## Key sources

| Citation | URL / DOI | Why it matters | Priority |
|---|---|---|---|
| Tuckute et al. 2024, Nat Hum Behav 8:544-561 | doi:10.1038/s41562-023-01783-7 | Closed-loop, out-of-sample control of the human language network by a model; the standard of proof to copy | A |
| Hadidi, Feghhi, Song et al. 2025, "Illusions of alignment..." | doi:10.1101/2025.03.09.642245 | Word rate, position and static embeddings explain most brain-score; the neuroscience twin of the word-overlap result | A |
| Feghhi et al. 2024, "What are LLMs mapping to in the brain?" | arXiv:2406.01538 | Same critique, first version; shuffled splits inflate scores | A |
| Marti, Wu, Piantadosi, Kidd 2023, Open Mind 7:79-92 | doi:10.1162/opmi_a_00072 | Human concepts vary between people; sets the ceiling for any model-to-human match | A |
| Huang et al. 2023, "Rigorously assessing natural language explanations of neurons" | arXiv:2309.10312 | Best neuron explanations have high error and little causal efficacy | A |
| Heap et al. 2025, "SAEs can interpret randomly initialized transformers" | arXiv:2501.17727 | "Interpretable" top examples arise from text statistics alone | A |
| Paulo & Belrose 2025, "Evaluating SAE interpretability without explanations" | arXiv:2507.08473 | Intruder test with human baseline; the design for seed 1 | A |
| McNamara 2011, "Can we measure memes?" | doi:10.3389/fnevo.2011.00001 | i-memes and e-memes; the measurement programme memetics proposed and never ran | A |
| Lepori, Kay & Tuckute 2026 | arXiv:2606.06857 | Sparse model features predicting voxels; the state of the art for a feature-to-brain bridge | A |
| Guo et al. 2026, "SAEs map brain-LLM alignment onto cortical semantic topography" | arXiv:2605.23035 | Semantic features, three languages, includes Llama-3.1-8B | A |
| Quian Quiroga et al. 2005, Nature 435:1102-1107 | doi:10.1038/nature03687 | Concept cells; selectivity and invariance as criteria | B |
| Quian Quiroga 2012, Nat Rev Neurosci 13:587-597 | doi:10.1038/nrn3251 | Review; concept cells as memory units | B |
| Goh, Cammarata et al. 2021, "Multimodal neurons in artificial neural networks" | doi:10.23915/distill.00030 | The model analogue of the Halle Berry cell, by P1's author | B |
| Schrimpf et al. 2021, PNAS 118(45) | https://hdl.handle.net/1721.1/138214 | The headline brain-score result | B |
| Goldstein et al. 2022, Nat Neurosci 25:369-380 | https://hassonlab.princeton.edu/sites/g/files/toruqf3591/files/documents/Goldstein_et_al_NN_2022.pdf | ECoG evidence for shared prediction and contextual embedding | B |
| Goldstein et al. 2024, Nat Commun 15:2768 | doi:10.1038/s41467-024-46631-y | Shared geometry, zero-shot, n = 3 | B |
| Caucheteux & King 2022, Commun Biol | doi:10.1038/s42003-022-03036-1 | "Partial" convergence, 102 subjects | B |
| Caucheteux, Gramfort & King 2023, Nat Hum Behav 7:430-441 | doi:10.1038/s41562-022-01516-2 | Predictive hierarchy, 304 subjects | C |
| Antonello & Huth 2024, Neurobiology of Language 5:64-79 | doi:10.1162/nol_a_00087 | Fit reflects general feature discovery, not prediction | B |
| AlKhamissi et al. 2024/2025, "The LLM language network" | arXiv:2411.02280 | Which units carry brain alignment; localiser method | B |
| Jamali et al. 2024, Nature 631:610-616 | doi:10.1038/s41586-024-07643-2 | Human single cells selective for word meaning in context | B |
| Bolukbasi et al. 2021, "An interpretability illusion for BERT" | arXiv:2104.07143 | Concepts read off top examples depend on the corpus | B |
| Zimmermann, Klein, Brendel 2023 | arXiv:2307.05471 | 130k human judgements; scale does not make units more human-readable | B |
| Paulo et al. 2024, "Automatically interpreting millions of features" | arXiv:2410.13928 | Auto-interp scores against human judgement | B |
| Bills et al. 2023, "Language models can explain neurons" | https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html | The auto-interp method everything else critiques | B |
| Marks et al. 2024, "Sparse feature circuits" | arXiv:2403.19647 | Crowd ratings: SAE features above neurons | C |
| Ferrando et al. 2024, "Do I know this entity?" | arXiv:2411.14257 | Models flag unknown entities; limits on unlearned memes | B |
| Thompson, Roberts & Lupyan 2020, Nat Hum Behav 4:1029-1038 | doi:10.1038/s41562-020-0924-8 | Word meanings align less across distant cultures | B |
| Atari et al. 2023, "Which humans?" | https://www.hks.harvard.edu/centers/cid/publications/which-humans | LLMs read like WEIRD respondents | B |
| Du et al. 2025, Nat Mach Intell | doi:10.1038/s42256-025-01049-z | Human-like object concept dimensions from model behaviour | C |
| Aunger 2002, *The Electric Meme* | https://www.kirkusreviews.com/book-reviews/robert-aunger/the-electric-meme/print | Memes as neural states; the theory P1 inherits | B |
| Dennett 2017, *From Bacteria to Bach and Back* | https://www.nature.com/articles/542030a (review) | Words as the paradigm memes | C |
| Hofstadter 1983, Metamagical Themas (Jan) | https://www.scientificamerican.com/article/metamagical-themas-1983-01/ | Early anatomy of a meme | C |
| Cammarata 2026, "Neural networks are already clean" | https://already-clean.goodfire.com/ | P1's source; expert audit, small samples | A |

## Fog

- I read abstracts, search summaries and reviews, not full texts. Figures marked "from a summary" (the 80%, 94%, 18 features, 64%, 0.84, rho values) need checking against the papers before they are quoted anywhere.
- The 2026 preprints (arXiv:2606.06857, 2605.23035, 2605.21049, the NeurIPS oral) are days to months old and unreplicated.
- Whether Hadidi et al. is published in Nature Communications, and its full author list, is unconfirmed.
- I found no study that puts human concept-cell recordings next to language-model units, and no language-model equivalent of Zimmermann's large psychophysics study. Either they do not exist or I missed them in the time available.
- I did not find any imaging study that executed McNamara's 2011 proposal. Absence is from a short search.
- Aunger, Dennett and Hofstadter were covered from reviews and catalogue records. Their finer positions (Aunger's replication mechanism, Hofstadter on concepts as analogy) are not checked.
- Cammarata's essay: I searched the saved text for rater protocols and found only the audit estimates. A fuller read may show more method than I credit.
- Does steering a unit change what human readers perceive in generated text? That would be the Tuckute-style test for concepts. I found no such study.
- What is the human-to-human agreement on whether a story "contains" a trope? Without that number no model-to-human figure can be interpreted.
- Open public fMRI narrative datasets suitable for seed 6 were not confirmed by name this session.
