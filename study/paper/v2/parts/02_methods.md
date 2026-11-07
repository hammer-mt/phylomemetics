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
