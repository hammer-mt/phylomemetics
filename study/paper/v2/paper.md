# Can Creativity Be Reverse-Engineered? Language-model neurons as units of cultural inheritance in 80 years of Batman

Mike Taylor and Claude (Anthropic) · Version 2, draft of 7 November 2026

*This is version 2 of the paper. Version 1 (20 October 2026) is unchanged in the repository. `CHANGES.md`, beside this file, lists what changed and why.*

## Abstract

[DRAFT IN PROGRESS: the abstract is written last, from the six confirmatory tests.]

## 1. Introduction

Cultural evolution borrowed the gene's logic but never found its gene. Dawkins proposed the meme as the unit of cultural inheritance in 1976 [1], and the idea spread faster than any method for testing it. Two decades later the field's own stock-take asked whether memetics was a science at all [2]. The recurring answer was that no one could say where one meme ends and the next begins, so no one could count memes, track their copying, or measure their fitness [3]. Cultural evolution research moved on without the unit, studying traits that could be coded by hand [4].

This study began with a recent claim from interpretability research. In "Neural networks are already clean", Cammarata argued that most neurons in modern language models such as Qwen3 already have one main job, often as specific as a feeling or a narrow trope, and that the apparent mess of earlier models came from fixable training and tooling problems rather than from superposition [10]. If that is right, a neuron is a ready-made, inspectable and testable unit, and memetics has the definition it lacked: a meme is whatever makes a given neuron fire. The rest of this introduction sets out why such a unit is needed, what it has to beat, and what the first version of this paper found.

The obstacles are specific. Concepts like "Batman" lack objectively measurable properties, and people hold different versions of them, so there is no agreed place to look for a "Batman meme" and no way to check that two people carry the same one. Members of a meme family such as the Scarlet Pimpernel, Zorro, Batman and Iron Man differ in more ways than they agree, so rule-based coding struggles to group them. Without a rigorous definition, memes cannot be categorised at scale, used for testable predictions, or measured without slow and error-prone human labelling. In that sense memetics is still pre-Mendel: no one has traced the inheritance of thousands of memes.

Hand-coded traits have carried that work a long way. Tehrani reconstructed the family tree of Little Red Riding Hood from coded plot features [5], and da Silva and Tehrani coded which international tale types each of 50 Indo-European populations tells, then used comparative phylogenetic methods to trace some tales to the Bronze Age [6]. The trait list in those studies comes from folklorists' indexes. It is reliable and expert, but it is fixed in advance, coarse, and costly to extend to a new body of stories.

Other work traces inheritance from the words themselves, with no hand-coded list. Barbrook and colleagues built a family tree of the manuscripts of *The Canterbury Tales* from their copying variants [21]. Chavalarias and Cointet rebuilt the branching lineages of scientific fields from the terms their papers share, and called the result phylomemetic [19]; this project uses the same word for a different method. Kuhn, Perc and Helbing found scientific memes, words and phrases that are inherited along citation links, with no linguistic knowledge [20]. In stories, film remakes can be matched from their Wikipedia plot summaries [22], and summaries of the same story become much harder to match once their names are replaced [23]. Word-based units like these are automatic too. So the question for any new unit is what it finds that words miss.

Language models offer a candidate unit that is neither fixed in advance nor hand-coded. Interpretability research has found single neurons that respond to one concept across many surface forms, including a Spider-Man neuron in CLIP that fires for photographs, drawings and the written name [7], much like the concept cells recorded in the human brain [18]. Neurons with similar jobs recur across separately trained models [8], and different models' representations appear to converge as they scale [9]. Text embeddings from different models can even be translated into one another without any paired examples, which suggests a shared geometry of meaning [27]. Automatic descriptions now exist for every MLP neuron of Llama 3.1 8B Instruct [24], one of the models used here, and features learned from text embeddings have been used to find and name the properties of texts that predict an outcome [25]. Cammarata's claim that most such neurons are already clean [10] is what would make them usable as a list of traits.

We test that proposal on one long, well-documented lineage: Batman, from the 1943 film serial to The Batman (2022), alongside literary and superhero relatives from Puss in Boots (1697) to The Green Hornet (2011). Batman is a convenient model organism. Its lineage is long and branching, many works state their sources, and fans and critics have argued for decades about what is and is not canon.

We ask three questions, matching the three things a science of memes needs. Identity: is there a set of neuron-defined memes that a story must carry to be Batman, and does it survive when every name is removed? Heredity: do adaptations carry the optional memes of their documented source comics more than those of other comics? Selection: do memes carried by popular early works spread further into later ones? Alongside these we ask whether the units respond to ideas or only to particular words, whether they find anything that plain word overlap does not, and whether the model uses them.

**What version 1 found.** The first version of this paper (20 October 2026) ran the study on Qwen3-8B and Llama 3.1 8B, with weights quantised to 4 bits, on a laptop. Adaptations carried their source comic's units more than other comics', so heredity was supported, but plain word overlap found the sources as well. Without names the canon could not tell Batman from other masked heroes. Most of the strongest units detected names and words, and selection was not detected.

That version's limitations section listed what its method left open. The models were smaller than the 32B model the essay studies, and quantised. A meme counted as present in a work whenever the work beat 68 of 69 ordinary films, so a weak activation counted as much as a strong one, where the essay's point is that the strong ones are what matter. The labels were read off the Batman corpus itself. Nothing was tested by steering or ablation.

**What this version changes.** This version re-runs the study the way the essay studies neurons. The main model is Qwen3-32B at full precision. Only a unit's own sign regime counts. A unit is present when it fires strongly: at least 30% of the strongest activation it reaches anywhere in a broad reference corpus. Explanations are written from the strongest examples in that corpus and then tested on new texts, where they can lose. Steering tests the labels, and ablation tests whether the model uses Batman's canon to know it is reading Batman. The corpus, the three experiments and their word-based comparators are those of version 1. Section 3.1 reports what the change of method did to each of version 1's conclusions.

## 2. Material and methods

The analysis plan for this version is `study/v2/PLAN.md` in the repository. It was committed and pushed before any of the new models read a study text. Nineteen changes were made after that, each logged in the plan with its reason before the result it bears on existed. Section 2.10 lists the ones that matter for reading the results.

### 2.1 Corpus

The corpus is unchanged from version 1. We used the plot sections of English Wikipedia articles, keeping the first 40 sentences of each. These retellings were written by people. The lineage comprises 32 Batman works from 1943 to 2022: 13 live-action films and serials, 12 comics, 4 animated films and 3 video games. The comparison set has 13 relatives. Five are literary ancestors: Puss in Boots, The Count of Monte Cristo, A Study in Scarlet, Dracula and The Scarlet Pimpernel. Eight are masked-hero or superhero films: The Bat (1926), The Mark of Zorro (1940), The Shadow, Superman, Spider-Man, Daredevil, Iron Man and The Green Hornet. There are also 69 popular films with no superheroes. In version 1 those films set the bar for every meme. Here they are 69 more ordinary stories. The corpus has 3,700 sentences.

The name-free version of every text is also unchanged: each proper name is replaced by a neutral code (X1, X2 and so on, consistent within a work). Two things about these texts are handled differently.

- **Codes.** The codes are our edit and not part of the story, and their numbers differ by work. In version 1 the bar for a name-free text came from name-free films, which have codes too. Here the bar comes from ordinary text without codes, so a unit that reacts to a code would look rare and specific. A code's own tokens are therefore not allowed to carry a unit's peak. Scores with the codes left in are kept as a check.
- **Length.** A unit's score is a peak over sentences, so a longer text has more chances, and Batman's summaries are longer than the hero films' (sentence count alone separates them with an AUC of 0.76). Every score is therefore also computed on each work's opening: its first sentences, at most 20 and at most 400 words, which leaves Batman works and hero films the same length (AUC 0.45).

### 2.2 Models and units

The main model is Qwen3-32B [11], the model the essay studies. Qwen3-8B is the same family at a quarter of the size, and is the model version 1 used. Llama 3.1 8B Instruct [12], from a different company, was version 1's replication model, and Llama 3.3 70B Instruct is that company's model at large scale. All four ran in PyTorch at the precision of their published weights (16-bit floating point) on rented GPUs. Version 1 ran the two 8B models with weights quantised to 4 bits.

A neuron is one channel of a feed-forward block: the product of a gate and a value, which feeds the block's output projection. Its positive and negative extremes are separate units, as in version 1 and in [10]. That gives 3,276,800 units in Qwen3-32B, 884,736 in Qwen3-8B, 917,504 in Llama 3.1 8B and 4,587,520 in Llama 3.3 70B. New in this version, an activation counts only where the gate is open. Where the gate is closed the product is a small echo of the opposite sign, which the essay shows to be a different signal from the unit's own. Each text was read as one continuous document, and the first position of a document is ignored.

To check the extraction, we compared our activations with the ones the essay publishes. For 63 Qwen3-32B neurons in layers 0 to 12 the essay gives 756 example contexts with the activation at a marked token. Ours correlate with the published values at r = 0.992.

### 2.3 Scale, presence and base rate

**Scale.** The essay calls a response strong when it reaches 30% of the neuron's strongest example in a large corpus. We measured each unit's strongest activation over a reference corpus of 46,000 documents (about 41 million tokens): 12,000 film plots, 24,000 English Wikipedia articles and 10,000 passages from books. English prose alone would understate the maximum of a unit whose real job is code, chat or another language, so a supplement of 5,434 documents (about 7 million tokens) adds source code, conversations with an assistant, Wikipedia in ten other languages, news and reviews. A unit's scale is its strongest activation in the reference corpus, the supplement or the study's own original texts.

**Presence.** A unit is present in a work when its strongest activation in any sentence of the work is at least 30% of its scale. Version 1 asked instead whether the work beat 68 of 69 reference films, so any activation counted, however weak.

**Base rate.** Three thousand of the film plots, cut to 40 sentences like the study's works, play no other part. A unit's base rate is the share of them in which it is present. One turned out to be a study film under an older title and was left out, which leaves 2,999. A unit is distinctive when its base rate is at most 3%, the rate at which an ordinary film crossed version 1's bar.

**Which units enter which analysis.** The canon, the identity experiment and most of the units chosen for explanation use distinctive units, because there the question is what marks Batman off from ordinary stories. The heredity and selection experiments use every unit, as version 1 did. There the question is which units two works share, and the word-based comparators drop no word either.

### 2.4 Memes, canon and other franchises

Candidate memes are distinctive units present in at least 6 of the 32 Batman works. Candidates that fire strongly on the same sentences (correlation of at least 0.5, average linkage) were merged into meme families, as in version 1. Canon tiers follow the number of Batman works a meme appears in: core (29 or more), common (23 to 28) and optional (6 to 22).

Version 1 compared Batman's canon with random sets of the 69 films. Under a 3% cap a random set of ordinary films cannot have a canon, whatever Batman is like. So this version adds comparisons that could fail: random sets of 32 background plots of one genre, and six other film series from the same plot dataset in which one character's name is in the title of at least 15 films (Godzilla, Frankenstein, Dracula, Doraemon, Tarzan and Kamen Rider).

### 2.5 Three pre-registered experiments

The designs, statistics and nulls are version 1's. E2, E3, the sequel control and the residual test are computed by version 1's own code on the new presence data. These are the same 32 works, analysed a second time by a method fixed in advance, with version 1's outcome known. Agreement between the two versions is therefore not independent confirmation.

**E1, identity.** For each Batman work, canon units are distinctive units present in at least 80% of the other 31 works (original texts). The held-out work is scored by the share of canon units present in its name-free text, and compared with the 8 masked-hero and superhero films. The prediction is met if the AUC is at least 0.80, its bootstrap 95% lower bound is at least 0.65, it is above the 95th percentile of a label-shuffle null, it holds within 0.05 after dropping sentences with iconography words, and, on the openings, it is at least 0.70 and above its own label-shuffle null. The last clause is new: without it a pass could be a matter of length. It is falsified, as version 1 registered, if the AUC is below 0.70 or its interval includes 0.5.

Version 1 put its failure on E1 down to the scoring rule, because a word-based centroid score reached 0.80 where the canon share reached 0.52. So the units are also given the words' rule: the cosine between a work's vector of strong unit activations, each weighted by the unit's rarity in the background plots, and the centroid of the other 31 works. The same criteria are applied to that score, to the TF-IDF centroid and to length alone.

**E2, heredity.** Nine adaptations have a documented source comic among the 12 comics. Optional units are those present in 2 to 16 of the 32 Batman works (name-free texts). For adaptation A and comic C, the inheritance rate is the share of C's optional units also present in A, z-scored against the same rate in the 11 Batman screen works that adapt no comic. The statistic is the share of eligible comics that beat the documented source, averaged over the adaptations. It is tested exactly, over all 3,991,680 ways of relabelling the comics. The prediction is an exact p below 0.05. The positive control is that at least 5 of 7 sequels rank their prequel among the top two earlier screen works. The word comparators (TF-IDF cosine and word presence on the same name-free texts) go through the same test.

**Beyond words: the residual test.** Whether the neurons add anything beyond shared words is decided by one test: the neuron inheritance rate with word measures regressed out, tested the same way. Version 1 regressed out two word measures. Before any Qwen3-32B result existed, two reviewers showed with words alone that this form can pass with nothing beyond words in it: a third word measure that leans towards commoner words, or counts repeats, passes it in the neurons' place (p = 0.002 and 0.0002). The decision test therefore regresses out seven word measures: version 1's two, word presence in four bands of frequency, and TF-IDF on raw counts. Sixteen further word-only measures were then put through it in the neurons' place. None passes (lowest p = 0.12), where eight of the sixteen pass version 1's form. "The neurons add something beyond shared words" is claimed only if the decision test passes at the primary setting and in at least half of eight sensitivity settings, and a version with every word and code measure regressed out passes too. Version 1's two-measure form is reported beside it.

**E3, selection.** Unchanged: across the 15 Batman works before 2000, the rank correlation between log pageviews and the spread of a work's units into the 9 later works that adapt no comic. With 15 works the test has about 27% power against a true correlation of 0.3, so this version makes no claim about selection whatever the result.

**Recognition.** A 32B model has read these Wikipedia summaries. Given a name-free one it may simply work out which story it is. So each model was asked, for each name-free text, which famous fictional hero the story is about and which work it summarises, and the experiments are read against its answers.

### 2.6 Explaining units, and letting the explanations lose

Version 1 named its memes from eight sentences of the Batman corpus itself. Here the units to explain were chosen by rule, before anyone read an example, in eight sets:

- *core*: the representative of every core-canon family, up to 60;
- *common* and *optional*: 30 families drawn at random from each of those tiers;
- *name-free core*: canon units still present in at least 70% of the name-free Batman works, up to 40;
- *discriminators*: the 40 distinctive units that best separate name-free Batman works from the hero films;
- *common story patterns*: the 40 units with a base rate between 3% and 30% that are most over-represented in Batman works, which the 3% cap would otherwise hide;
- *inherited* and *not inherited*: 20 units drawn at random from E2's optional units that an adaptation shares with its source, and 20 from the rest.

The same rule was applied to Qwen3-8B. Each role below was played by AI agents (Claude) that saw nothing but their own prompt. The prompts are in `study/v2/prompts/`.

- An *explainer* read each unit's 16 strongest examples from the reference corpus and supplement, the first six with their whole document up to the peak, its 8 strongest study sentences, and the strongest examples of its opposite pole. It wrote a title, a short explanation and at least 8 key words. It ran no experiments.
- A *tester* saw only the title, explanation and key words. Before anything was measured it wrote five new texts that should make the unit fire (A), five that should make it fire while using none of the key words (B), five near misses that should not fire, at least three of which reuse a key word in another sense (C), five on a distant subject (D), and one meant to fire it as hard as possible.
- A *keyword analyst* saw only the title and explanation and wrote the search terms of someone who expects the wording to be evaded.
- A *kind judge* saw only the title and explanation. It said what kind of thing the unit detects, and whether it reacts to a particular name or word or to an idea that can be put in other words.

A text's response is the unit's strongest activation in it as a share of its scale, and 30% or more is strong, as in the essay. An explanation *wins* when more of its A texts than of its C texts are strong. A unit is *idea-bound* when at least 3 of its 5 keyword-free texts are strong and at most one keyword-reusing near miss is. Every text was scored on every explained unit, which gives the rate at which a unit fires strongly on texts written for other units. Two word-based measures were scored on the same texts: the analyst's search terms, and TF-IDF similarity to the unit's examples.

Test texts are short and read alone, while a unit's scale comes from documents of about 1,000 tokens. A unit whose own strongest study sentence and strongest reference examples all stay below 30% when read alone is counted apart, as *not testable out of context*. A coverage check asked a blind judge whether each description predicts the unit's firings at 30 to 60% of scale, where presence is decided, on passages from the background plots.

This is label validation. It is not the planted-story test that the pre-registration reserves for the question of whether neurons follow an idea when the words change (Section 4.5).

### 2.7 Steering and ablation

Neither test was in version 1. Both were specified in the plan before being run.

**Steering.** The model continued three neutral story openers for 40 tokens with one unit's activation held at a multiple of its scale, and once without. A blind judge who saw both sets of continuations picked which of five unit titles had been pushed in, or "none of these". The multiple was chosen beforehand on 40 neurons from the essay's public gallery, not on study units. The criterion is accuracy on the units whose kind is an idea, against 20% chance.

**Ablation.** Each model was asked which hero each name-free Batman summary is about. The outcome is the log-odds that its reply starts with Batman's name, with nothing touched and with that work's canon units switched off on the tokens of the summary only. The canon's effect is compared with two controls of 200 random sets each. In the first, every canon unit is replaced by a unit of the same layer and pole that is present in about as many name-free Batman works. In the second, the replacement is also matched on how strongly it fires in these texts, so that the sets remove about as much activation as the canon does. The criterion needs p below 0.05 against both controls, and a larger effect on the Batman works than on seven hero films scored on their own hero's name.

### 2.8 Two models

For each explained Qwen3-32B unit, its twin in another model is the unit there whose activations across the 3,700 original-text sentences correlate most strongly with it. Version 1 scored twins on those same sentences. Here they are scored where they were not chosen: on the tester's texts written for the Qwen3-32B unit, and on the name-free sentences, against control units matched on how well a single word predicts them.

### 2.9 What is confirmatory, and robustness

Six tests are confirmatory, all for Qwen3-32B under the new rule: E1, E2, the residual test, E3, the ablation and steering on idea units. Each is read at 0.05, with Holm-adjusted p values beside them. The other three models are replications. Everything else is description or a sensitivity check.

The sensitivity checks were fixed in advance. They vary the threshold (20, 30, 40 and 50% of scale), the score (strongest sentence or the mean of the three strongest), the base-rate cap (1% to 30%, or none), the scale (without the study texts; the fifth-strongest example in place of the strongest), text length (openings), the handling of the codes, the echo regime, weak poles and units whose scale comes from the supplement. To separate the effect of the new rule from the effect of the new models, every primary statistic was also computed under version 1's rule on the new activations.

### 2.10 Changes after the plan was frozen

The plan lists nineteen. The ones that change how a result is read are these. The residual test regresses out seven word measures, not two (Section 2.5). Its sensitivity rule uses the eight settings with no base-rate cap, because a cap applied to units and not to words can make the test pass or fail for reasons unrelated to the question. The ablation has a second control matched on activation removed, and is scored in log-odds, because a drop in log-probability shrinks as the model grows confident (Section 2.7). Genre sets take their base rates from plots of other genres. A whole-work E1 score is always quoted with its opening-only score and with length alone. All were made before the result they bear on existed, and none after a result was seen.

## 3. Results

[DRAFT IN PROGRESS: results are being written as the runs finish.]

## 4. Discussion

[DRAFT IN PROGRESS]

## References

1. Dawkins R. 1976 *The selfish gene*. Oxford, UK: Oxford University Press.
2. Aunger R (ed.). 2000 *Darwinizing culture: the status of memetics as a science*. Oxford, UK: Oxford University Press.
3. Edmonds B. 2005 The revealed poverty of the gene-meme analogy: why memetics per se has failed to produce substantive results. *J. Memetics* **9**.
4. Mesoudi A, Whiten A, Laland KN. 2006 Towards a unified science of cultural evolution. *Behav. Brain Sci.* **29**, 329–383.
5. Tehrani JJ. 2013 The phylogeny of Little Red Riding Hood. *PLoS ONE* **8**, e78871.
6. da Silva SG, Tehrani JJ. 2016 Comparative phylogenetic analyses uncover the ancient roots of Indo-European folktales. *R. Soc. Open Sci.* **3**, 150645. [doi:10.1098/rsos.150645](https://doi.org/10.1098/rsos.150645)
7. Goh G, Cammarata N, Voss C, Carter S, Petrov M, Schubert L, Radford A, Olah C. 2021 Multimodal neurons in artificial neural networks. *Distill* **6**, e30.
8. Gurnee W, Horsley T, Guo ZC, Kheirkhah TR, Sun Q, Hathaway W, Nanda N, Bertsimas D. 2024 Universal neurons in GPT2 language models. *Trans. Mach. Learn. Res.*
9. Huh M, Cheung B, Wang T, Isola P. 2024 The platonic representation hypothesis. In *Proc. 41st Int. Conf. Machine Learning*.
10. Cammarata N. Neural networks are already clean. Goodfire. [already-clean.goodfire.com](https://already-clean.goodfire.com/) (accessed 7 November 2026).
11. Qwen Team. 2025 Qwen3 technical report. arXiv:2505.09388.
12. Grattafiori A *et al.* 2024 The Llama 3 herd of models. arXiv:2407.21783.
13. Saitou N, Nei M. 1987 The neighbor-joining method: a new method for reconstructing phylogenetic trees. *Mol. Biol. Evol.* **4**, 406–425.
14. Mantel N. 1967 The detection of disease clustering and a generalized regression approach. *Cancer Res.* **27**, 209–220.
15. Cohen J. 1960 A coefficient of agreement for nominal scales. *Educ. Psychol. Meas.* **20**, 37–46.
16. Marti L, Wu S, Piantadosi ST, Kidd C. 2023 Latent diversity in human concepts. *Open Mind* **7**, 79–92.
17. Heylighen F, Chielens K. 2009 Evolution of culture, memetics. In *Encyclopedia of complexity and systems science* (ed. RA Meyers). New York, NY: Springer.
18. Quian Quiroga R, Reddy L, Kreiman G, Koch C, Fried I. 2005 Invariant visual representation by single neurons in the human brain. *Nature* **435**, 1102–1107.
19. Chavalarias D, Cointet J-P. 2013 Phylomemetic patterns in science evolution: the rise and fall of scientific fields. *PLoS ONE* **8**, e54847. [doi:10.1371/journal.pone.0054847](https://doi.org/10.1371/journal.pone.0054847)
20. Kuhn T, Perc M, Helbing D. 2014 Inheritance patterns in citation networks reveal scientific memes. *Phys. Rev. X* **4**, 041036. [doi:10.1103/PhysRevX.4.041036](https://doi.org/10.1103/PhysRevX.4.041036)
21. Barbrook AC, Howe CJ, Blake N, Robinson P. 1998 The phylogeny of The Canterbury Tales. *Nature* **394**, 839. [doi:10.1038/29667](https://doi.org/10.1038/29667)
22. Chaturvedi S, Srivastava S, Roth D. 2018 Where have I heard this story before? Identifying narrative similarity in movie remakes. In *Proc. 2018 Conf. North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, vol. 2, pp. 673–678. [doi:10.18653/v1/N18-2106](https://doi.org/10.18653/v1/N18-2106)
23. Hatzel HO, Biemann C. 2024 Tell me again! A large-scale dataset of multiple summaries for the same story. In *Proc. 2024 Joint Int. Conf. Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)*, pp. 15732–15741. [aclanthology.org/2024.lrec-main.1366](https://aclanthology.org/2024.lrec-main.1366)
24. Choi D, Huang V, Meng K, Johnson DD, Steinhardt J, Schwettmann S. 2024 Scaling automatic neuron description. Transluce. [transluce.org/neuron-descriptions](https://transluce.org/neuron-descriptions) (accessed 20 October 2026).
25. Movva R, Peng K, Garg N, Kleinberg J, Pierson E. 2025 Sparse autoencoders for hypothesis generation. In *Proc. 42nd Int. Conf. Machine Learning*. arXiv:2502.04382.
26. Coogan P. 2006 *Superhero: the secret origin of a genre*. Austin, TX: MonkeyBrain Books.
27. Jha R, Zhang C, Shmatikov V, Morris JX. 2025 Harnessing the universal geometry of embeddings. arXiv:2505.12540. [doi:10.48550/arXiv.2505.12540](https://doi.org/10.48550/arXiv.2505.12540)

## Data, code and AI use

**Data and code.** Everything is in the study folder, `study/`, of [github.com/hammer-mt/phylomemetics](https://github.com/hammer-mt/phylomemetics). The code for this version is in `study/v2/`, its results in `study/results/v2/`, and the analysis plan in `study/v2/PLAN.md`. Version 1's code and results are unchanged beside them, and version 1 of the paper is in `study/paper/`. The tables in this paper are written from the committed results by `study/v2/analysis/paper_tables.py`, and the figures by the scripts in `study/v2/figs/`. Every explanation, every test text and every judge's verdict is committed, with the prompts the agents were given. The reference corpus and supplement are built from pinned public datasets by `study/v2/build_reference.py` and `study/v2/build_supplement.py`, and every document is listed in the results. Raw activations are not stored; the passes can be re-run with the scripts on a rented GPU. The code is under the MIT licence. Plot summaries and texts derived from them come from English Wikipedia and remain under CC BY-SA 4.0.

**Pre-registration.** The three experiments were specified, with falsifying results, before version 1 ran them. For this version the analysis plan was committed and pushed to the repository (commit `1f32fd1`, 02:55 UTC on 7 November 2026) before any of its models read a study text, so the repository's history records the order. The plan was reviewed beforehand by independent AI reviewers asked to find what was wrong with it. Section 18 of the plan logs the changes made after the freeze, each with its reason and with what was known at the time. The planted-story experiments that version 1 promised are pre-registered separately in `preregistration/`. None has run, and this version does not replace them.

**Authors' contributions.** Mike Taylor conceived the study and its theoretical framing, chose the Batman case and directed the work. For this version he asked for the method to be aligned with Cammarata's essay and for the weaknesses of version 1 to be fixed, and provided the GPU budget. Claude wrote the plan and the code for this version, ran the models and the analyses, and drafted the paper.

**AI use.** Claude is an AI model made by Anthropic, and is credited as an author for the contributions above. AI agents of the same model family also played every role in the explanation tests (explainer, tester, keyword analyst, kind judge, coverage judge and steering judge), each seeing only its own prompt, and carried out the adversarial reviews of the plan and the fact-check of this paper. Qwen3-32B repeated two of the judgments as a second model family. No human has audited the explanations or the judges' verdicts. The items and answer keys are committed so that an audit can be done.

**Competing interests.** None declared.
