# Can Creativity Be Reverse-Engineered? Language-model neurons as units of cultural inheritance in 80 years of Batman

Mike Taylor · 20 October 2026

## Abstract

Memetics stalled because no one could say where one meme ends and the next begins. We test a proposal from *Marketing Memetics*: define a meme as whatever makes a single neuron inside an artificial neural network fire. In Qwen3-8B we recorded 884,736 neuron poles across Wikipedia plot summaries of 32 Batman works (1943–2022), 13 literary and superhero relatives, and 69 ordinary films. We found 14,843 meme families present in Batman far beyond chance. Fifty-six form a core canon present in at least 29 of the 32 works.

The strongest memes are model-independent. Llama 3.1 8B, trained by a different company, has twins that fire on the same sentences (median r = 0.82), and the two models' family trees of the lineage correlate at 0.98. Some memes respond to ideas rather than words.

We ran three pre-registered experiments on name-free texts. Heredity was supported in both models: adaptations carry their documented source comic's optional memes more than other comics' (p = 0.004 and p = 0.0002). Identity was not. Without names, Batman's canon separates Batman from ordinary stories (AUC 0.90 to 0.95) but not from masked-hero films (0.52 to 0.54). Its name-free core is the superhero template (a secret identity, an origin in loss, a costume, a mission by night), and Spider-Man, Daredevil and Superman carry Batman's canon more strongly than most Batman stories do. Selection was not detected. In images, CLIP contains a visual Batman canon and DINOv2 largely does not.

Neuron-defined memes make cultural heredity measurable. They show a franchise to be a shared memeplex topped with named memes. Selection remains to be tested at scale.

## 1. Introduction

Cultural evolution borrowed the gene's logic but never found its gene. Dawkins proposed the meme as the unit of cultural inheritance in 1976 [1], and the idea spread faster than any method for testing it. Two decades later the field's own stock-take asked whether memetics was a science at all [2]. The recurring answer was that no one could say where one meme ends and the next begins, so no one could count memes, track their copying, or measure their fitness [3]. Cultural evolution research moved on without the unit, studying traits that could be coded by hand [4].

Taylor's account of the failure, written from inside marketing practice, is specific [18]. Concepts like "Batman" lack objectively measurable properties. People hold different versions of them. "We can't pinpoint a 'Batman meme' in one's brain or identify the same sequence in others' brains" [18, p. 21]. Members of a meme family such as the Scarlet Pimpernel, Zorro, Batman and Iron Man differ in more ways than they agree, so rule-based programs cannot group them. Without a rigorous definition, memes cannot be categorised at scale, used for testable predictions, or measured without slow and error-prone human labelling [18, pp. 21–23]. His summary is that "the chief problem is that it's hard to define where a meme begins and ends" [18, p. 185], and that memetics is still pre-Mendel: no one has traced the inheritance of thousands of memes [18, pp. 194–195].

Hand-coded traits have carried that work a long way. Tehrani reconstructed the family tree of Little Red Riding Hood from coded plot features [5], and da Silva and Tehrani coded which international tale types each of 50 Indo-European populations tells, then used comparative phylogenetic methods to trace some tales to the Bronze Age [6]. The trait list in those studies comes from folklorists' indexes. It is reliable and expert, but it is fixed in advance, coarse, and costly to extend to a new body of stories.

Language models offer a candidate unit that is neither fixed in advance nor hand-coded. Interpretability research has found single neurons that respond to one concept across many surface forms, including a Spider-Man neuron in CLIP that fires for photographs, drawings and the written name [7], much like the concept cells recorded in the human brain [21]. Neurons with similar jobs recur across separately trained models [8], and different models' representations appear to converge as they scale [9]. Most recently, Cammarata argued that most neurons in modern models such as Qwen3 already have one main job, often as specific as a feeling or a narrow trope, and that the apparent mess of earlier models came from fixable training and tooling problems rather than from superposition [10]. If that is right, a neuron is a ready-made, inspectable and testable definition of a meme: the meme is whatever makes that neuron fire. Taylor proposed exactly this route in 2023, calling multimodal neurons that respond to a cluster of concepts around one theme "a meme", and arguing that probing which neurons fire for specific memes would let us reverse-engineer the process [18, p. 194]. This paper puts that proposal into practice, on the same worked example his book uses: Batman as "a basket of associated memes" [18, pp. 16–18].

We test that proposal on one long, well-documented lineage: Batman, from the 1943 film serial to The Batman (2022), alongside literary and superhero relatives from Puss in Boots (1697) to The Green Hornet (2011). Batman is a convenient model organism. Its lineage is long and branching, many works state their sources, and fans and critics have argued for decades about what is and is not canon.

We ask three questions, matching the three things a science of memes needs. Identity: is there a set of neuron-defined memes that a story must carry to be Batman, and does it survive when every name is removed? Heredity: do adaptations carry the optional memes of their documented source comics more than those of other comics? Selection: do memes carried by popular early works spread further into later ones? Alongside these we ask whether the units are model-independent, by comparing two language models from different companies and two vision models, and whether they respond to ideas rather than to particular words.

## 2. Material and methods

### 2.1 Corpus

We used the plot sections of English Wikipedia articles, keeping the first 40 sentences of each. These are retellings written by people, not by a model. The lineage comprises 32 Batman works from 1943 to 2022: 13 live-action films and serials, 12 comics, 4 animated films and 3 video games. The comparison set has 13 relatives ("cousins"). Five are literary ancestors: Puss in Boots, The Count of Monte Cristo, A Study in Scarlet, Dracula and The Scarlet Pimpernel. Eight are masked-hero or superhero films: The Bat (1926), The Mark of Zorro (1940), The Shadow, Superman, Spider-Man, Daredevil, Iron Man and The Green Hornet. A background pool of 69 popular films with no superheroes sets the bar against which every meme is measured. Three works were dropped because their plot sections had fewer than 8 sentences (the 1949 serial Batman and Robin, the Moon Knight miniseries and Pulp Fiction). The corpus has 3,700 sentences.

A name-free version of every text was built in two steps. A language model (Claude) rewrote each summary with proper names replaced, and a second pass checked the rewrite. The rewrite was used only to locate the names: a word-level diff against the original marked each changed span, and each span was replaced with a neutral code (X1, X2, and so on, consistent within a work). In all, 9,156 spans were replaced. This removes the role words the rewriter chose, such as "the vigilante" (used 366 times in the Batman rewrites, 16 times in the cousins and never in the ordinary films), which would otherwise leak identity.

### 2.2 Models and units

The main model is Qwen3-8B [11]. Llama 3.1 8B Instruct [12], trained by a different company on different data, is the independent second opinion, and Qwen3-1.7B is a smaller model from the same family. All ran locally at 4-bit precision with MLX. A unit is one channel of a feed-forward block's hidden layer (the SwiGLU product that feeds the down-projection). Because a channel's sign is arbitrary, its positive and negative extremes are treated as separate units, following [10]. That gives 884,736 units in Qwen3-8B, 917,504 in Llama 3.1 8B and 344,064 in Qwen3-1.7B. For each unit we recorded its strongest response in every sentence, with each text processed as one continuous document.

### 2.3 Presence, memes and canon

A work's score on a unit is the mean of its three strongest sentences. A unit is present in a work when that score beats all but one of the reference films, so an ordinary story crosses the bar about 3% of the time by chance. Candidate memes are units present in at least 6 of the 32 Batman works (18,485 in Qwen3-8B, against about 230 expected by chance). Units that fire on the same sentences (correlation of at least 0.5, average linkage) were merged into 14,843 meme families. Families were named by reading the eight sentences each fires on most strongly, and every name was checked by a second, sceptical reader. The names were not validated by steering.

Canon tiers follow the share of Batman works a meme appears in: core (at least 90%, 29 or more works), common (70 to 89%) and optional (19 to 69%). A meme is Batman-specific if it appears in at most 3 of the 13 cousins, and shared with the genre otherwise.

### 2.4 Trees, overlap between models and rewording

Works were compared by the Jaccard distance between their sets of memes, and trees were built by neighbour-joining [13], rooted on the oldest work (Puss in Boots). Agreement between two models' trees is the Mantel correlation of their distance matrices [14]. For each Qwen meme, its twin in another model is the unit whose responses across the 3,700 sentences correlate most strongly with it. There are two baselines: the same search for active Qwen units that were not selected as memes, and the same search with sentences shuffled. Agreement on the meme grid is the share of meme-by-work cells marked the same way, with Cohen's kappa [15].

To test whether units respond to ideas rather than words, the top sentence of each of 31 named memes was rewritten to share almost no content words with the original (none in 28 of 31; one common word in the other three). A unit still fires if it responds more strongly to the rewording than to any of ten everyday sentences. Separately, each model's revenge unit was chosen from the 41 corpus sentences that use revenge words, among 50 candidates, on 10 practice pairs that use no such words. It was then scored on 12 new minimal pairs, in which a revenge sentence is matched with the same sentence made harmless.

### 2.5 Three pre-registered experiments

The designs below were revised after an adversarial review by three independent AI reviewers, before any experimental result was computed. Each names one primary test, with Qwen3-8B as the primary model and Llama 3.1 8B as the replication.

**E1, identity.** For each Batman work, canon units are those present in at least 80% of the other 31 works (original texts). The held-out work is scored by the share of canon units present in its name-free text. The reference films are split in half 20 times: one half sets the presence bar, the other half is scored. Primary comparison: Batman against the 8 masked-hero and superhero film cousins, scored with the same fold's canon. Prediction: AUC of at least 0.80, a bootstrap 95% lower bound of at least 0.65, and an AUC above the 95th percentile of a label-shuffle null, in which 32 of the 40 works are drawn at random to play the franchise. Controls: the original-text ceiling; canon chosen on name-free texts; dropping every name-free sentence that still contains an iconography word (bat, cape, clown, cave, signal, asylum and so on); text length alone; held-out ordinary films; and literary cousins.

**E2, heredity.** Nine adaptations have a documented source comic in the corpus, and 12 comics are available. Optional units are those present in 2 to 16 of the 32 Batman works. For adaptation A and comic C, the inheritance rate is the share of C's optional units also present in A. It is z-scored against the same rate in the 11 Batman screen works that adapt no comic, which corrects for comics that resemble everything. Only comics published no later than A are eligible. The statistic is the share of eligible comics that beat the documented source (the best of its sources), averaged over the adaptations, and it is tested against a permutation that relabels comic identities jointly (10,000 draws). Positive control: on name-free texts, each of 7 sequels should rank its prequel among the top 2 earlier screen works for at least 5 of 7. If it fails, the comic test is inconclusive rather than falsified. Controls: placebo targets, the original-text upper bound, raw rates, no time rule, a strong-attribution subset, dropping the three Killing Joke adaptations, and three alternative frequency bands.

**E3, selection.** Popularity is the mean monthly English Wikipedia pageviews from July 2015 to September 2026. The early set is the 15 Batman works before 2000. The later set is the 17 from 2000 onward, minus the 8 that adapt a comic, which removes the circular route by which fame follows adaptation. For units present in 2 to 80% of early works, a unit's later share is residualised on its early prevalence. Each early work's offspring score is the mean residual over its units, residualised in turn on the work's year, plot length and centrality. The primary test is the Spearman correlation of offspring with log popularity, with popularity permuted within medium (screen or comic, 10,000 draws), predicted positive at one-sided p < 0.05 and with the same sign in Llama. Placebos: the same test run backward in time, and with the reference films as the outcome.

### 2.6 Images

The lead image of each work's Wikipedia article (poster, cover or box art; retrieved for 111 of the 114 works, including 29 of the 32 Batman works) was passed to two vision models: OpenAI CLIP RN50x4 [16], whose multimodal neurons [7] we read from the last two convolutional stages, and Meta DINOv2-base [17], read from each MLP neuron in all 12 layers. Each image was seen as 10 views (the full image, its four corner crops and their mirror images). A unit's score is the mean of its strongest response in each view. Presence and canon follow the text rules. Twins across the two vision models were matched over all views. Copyrighted images were analysed but are not reproduced.

## 3. Results

### 3.1 Memes and the canon

In Qwen3-8B, 18,485 units were present in at least 6 of the 32 Batman works, against about 230 expected by chance. Merging units that fire on the same sentences gave 14,843 meme families. Most strong families are character and place detectors (Alfred, Robin, the Joker, Gotham). Some are story patterns: the origin wound, the secret identity, taking up a hero's mantle, a funeral, a coin flip deciding a hostage's fate. A minority track Wikipedia's house style rather than story content, such as cast lists and publication history. These were labelled as such and kept, because every analysis uses all families.

Fifty-six memes appear in at least 29 of the 32 works (the core canon): 24 are Batman's own and 32 are shared with the superhero cousins. A further 194 appear in 23 to 28 works, and 14,593 are optional. Most works from 1966 onward carry 90 to 100% of the core. The two lowest are the Joker-centred Joker (2019), at 55%, and The Killing Joke, at 77%, matching the 1943 serial. Among the cousins, the share of Batman's core carried rises with date. Ancestors written before Batman's creation in 1939 carry at most 21% (The Bat, 1926). The later superheroes carry far more: Spider-Man 79%, Iron Man 62%, Superman 61% and Daredevil 55%.

### 3.2 Two models carve the same memes

The 33 strongest Qwen memes have Llama twins that fire on the same sentences, with a median correlation of 0.82, and 70% exceed 0.7. Across all 14,843 families the median falls to 0.40. That is barely above the 0.36 for active Qwen units that were not selected as memes, and well above the 0.14 obtained with shuffled sentences. Agreement therefore concentrates in the strongest memes. Families encoded by many redundant Qwen units match better (median r of 0.77 for families of 30 or more units). For the 31 named memes, Llama's twins mark the same works present or absent in 87% of grid cells, against 52% expected by chance (Cohen's kappa 0.73). Across all families the figure is 76% (kappa 0.34). The two models' distance matrices over the 45 Batman works and cousins correlate at 0.98, and still at 0.98 after controlling for summary length and the number of memes per work. Qwen3-1.7B gives the same picture.

### 3.3 Ideas or words

When the top sentence of each named meme was rewritten to share almost no content words with the original (none in 28 of 31), 11 of 31 Qwen memes and 9 of 31 Llama twins still fired more strongly than on any of ten everyday sentences. Four survived in both models: a funeral, Two-Face's coin flip, the secret identity and the joke. In Qwen, every character meme except Harvey Dent failed once the name was gone. Llama's revenge unit (L14·3474) chose the revenge sentence over its harmless twin in 10 of 12 new pairs written without revenge words, and fired harder on "he swore to make the men who killed his father pay for what they had done" than on the same sentence with "revenge". Qwen's best revenge unit (L27·375) responded to both "revenge" and "vengeance" but was at chance without them (4 of 12).

### 3.4 Trees and the vigilante lineage

All three models' trees place the three Arkham games on their own branch, place Batman v Superman beside Superman, and make The Killing Joke and Joker (2019) sister taxa. The 2019 film borrows the comic's failed-comedian origin. Qwen's tree has a camp-leaning clade (Batman '66, Batman & Robin, The Lego Batman Movie, Batman Ninja and Return of the Joker, with Batman R.I.P. and White Knight). Llama's has a grounded crime clade (Year One and The Long Halloween with the three Nolan films, Earth One and The Batman). Clustering by decade-style era labels was weak (between-era over within-era distance of 1.03 in both models, p of about 0.03): the branches follow tone more than date. Taylor's "billionaire vigilante orphan" lineage [18, pp. 21–22] is partly recovered. Zorro's nearest relative is the Scarlet Pimpernel in both models. After Spider-Man, Iron Man's nearest works are recent Batman films (Batman Begins and The Dark Knight Rises in Llama; Batman v Superman and Batman Begins in Qwen). Zorro itself sits far from Batman, but its summary has only 9 sentences.

**Table 1.** The three pre-registered experiments. Primary model Qwen3-8B; replication Llama 3.1 8B. All tests on name-free texts.

| Experiment | Prediction | Qwen3-8B | Llama 3.1 8B | Verdict |
| --- | --- | --- | --- | --- |
| E1 Identity | AUC ≥ 0.80 vs 8 hero film cousins | 0.52 (0.28–0.75) | 0.54 (0.31–0.79) | Not supported |
| E2 Heredity | Fewer comics beat the source than chance, p < 0.05 | 16% vs 47%, p = 0.004 | 10% vs 47%, p = 0.0002 | Supported |
| E3 Selection | Spearman ρ > 0, p < 0.05 | ρ = 0.25, p = 0.35 | ρ = 0.00, p = 0.58 | Not supported |

### 3.5 E1: identity

The pre-registered prediction failed. On name-free texts, the held-out work's share of canon units separated Batman from held-out ordinary films (AUC 0.90 in Qwen, 0.95 in Llama) and from the literary ancestors (0.89 and 0.95). Against the 8 masked-hero and superhero films it was at chance: 0.52 (95% CI 0.28 to 0.75) in Qwen and 0.54 (0.31 to 0.79) in Llama, inside the label-shuffle null (95th percentile 0.70 and 0.67). With names the same test scores 0.996. Dropping iconography sentences (68 of 3,700) changed nothing (0.52 and 0.53), and neither did choosing the canon on name-free texts (0.45 and 0.60). The film cousins split in two. Spider-Man, Daredevil and Superman carry Batman's canon more strongly than most name-free Batman stories: a Batman story out-scores them only 12%, 15% and 29% of the time in Qwen (20%, 19% and 40% in Llama), and Iron Man only 32% (50% in Llama). The older masked heroes carry much less: Batman stories out-score Zorro 96% of the time, The Bat 78% and The Green Hornet 81% (Llama 100%, 98% and 72%). Only 7 of 407 Qwen canon units (Llama: 9 of 539) remain present in at least 70% of name-free Batman works. They respond to a secret identity kept by disguise, an origin in loss leading to a costume, fighting crime in costume by night, and taking up a mantle. They are present in 91% of the hero-film cousins and 1% of ordinary films. The same procedure finds no name-free core for the cousins as a group or for random 32-film pseudo-franchises.

![Figure 1](fig1_identity.png)

**Figure 1.** Without names, Batman's canon separates Batman stories from ordinary films and literary ancestors, but not from masked-hero and superhero films.

### 3.6 E2: heredity

The prediction held in both models. On name-free texts, the share of eligible comics whose hub-adjusted inheritance rate beats the documented source was 16% in Qwen and 10% in Llama, against 47% under joint relabelling (p = 0.004 and p = 0.0002). The source beat the median non-source comic for 9 of 9 adaptations in both models. Placebo targets (non-adaptations given the same source sets) averaged 38% and 39%. Every pre-registered variant held: with names (12% and 8%), raw rates, no time rule, the strong-attribution subset, dropping the three Killing Joke adaptations (p = 0.023 and 0.0015), and three alternative frequency bands. In the sequel positive control, without names Qwen recovered the prequel among the top two for 4 of 7 sequels and Llama for 5 (bar: 5). With names, the counts were 7 and 6.

![Figure 2](fig2_heredity.png)

**Figure 2.** For most adaptations the documented source is the closest or second-closest eligible comic by inherited memes, in both models.

### 3.7 E3: selection

The prediction was not supported. Across the 15 pre-2000 works, the rank correlation between log monthly pageviews and the residualised spread of a work's memes into the 9 later non-adaptations was 0.25 in Qwen (p = 0.35) and 0.00 in Llama (p = 0.58). The baseline-film placebo outcome gave similar values (0.29 and 0.26), and the reverse-time placebo was near zero (−0.08 and −0.17). With 15 works the test can detect only large effects, so this is a weak null.

### 3.8 Images

In CLIP RN50x4, 56 units were present in at least 6 of the 29 available Batman images (chance 0.6), 10 in at least 15, and 4 in at least 20. The strongest fires on 28 of the 29 and on one ordinary-film poster. Its top images all show Batman's pointed cowl. Other strong units respond to a hero insignia (bat or S), the BATMAN wordmark, a large bat silhouette, and drawn comic covers, which is medium rather than meme. DINOv2, trained without text, had more candidate units (416 at 6 or more, chance 5.4) but only 2 present in 12 or more Batman images. CLIP's visual memes matched DINOv2 units at a median correlation of 0.45, against 0.36 for random active units and 0.13 for shuffled views. On generated test images with no Batman in them, the cowl unit responded only moderately to gothic night scenes (73rd to 75th percentile of its corpus responses) and hardly to a beach (4th). A masked-or-hooded-figure unit fired strongly on a cloaked figure beside a gargoyle (96th percentile).

## 4. Discussion

Defining a meme as whatever makes a neuron fire gives memetics a unit it can count. Each meme has an address, can be queried on any text, and comes with its strongest examples attached. The strong ones reappear in a model trained by a different company on different data. With that unit, one of the three classic questions became testable and came out positive: adaptations inherit their source comic's memes, even when every name is removed, in both models.

### 4.1 What the unit answers in Taylor's account

Taylor gives four reasons memetics could not become a science [18, pp. 21–23, 185–186]. Memes lack measurable properties. They have no physical locus. They differ between people. And their family resemblances defeat rule-based programs. The neuron definition answers the first two directly. A meme now has a location in an artificial brain, and a work either crosses its threshold or does not.

The third becomes an experiment. Two models play two readers: they agree closely on the strong memes (correlation 0.82, 87% agreement on the meme grid) and diverge on the weak ones. That divergence echoes the finding that people's concepts of even common words vary widely from person to person [19].

The fourth is where the result is most interesting. Taylor's "billionaire vigilante orphan" family (the Scarlet Pimpernel, Zorro, Batman and Iron Man) shares more differences than similarities, yet people know it belongs together [18, pp. 21–22]. E1 recovers part of that family. With names removed, Iron Man carries Batman's canon about as strongly as a typical Batman story, and so do Spider-Man, Daredevil and Superman. The shared core is the superhero template: a secret identity, an origin in loss, a costume, a mission by night. The older end of the family carries less: on the full canon, Batman stories out-score Zorro, the 1926 film The Bat and The Green Hornet most of the time, and the Scarlet Pimpernel groups with the literary ancestors, though Zorro's summary is only 9 sentences long. What is uniquely Batman sits in the named memes, such as Gotham, Alfred, the Joker and Robin. In Taylor's terms, Batman is a basket of memes [18, pp. 16–18]. Its lower layer is a memeplex he shares with other masked heroes, and its upper layer is a set of characters and places that only he carries.

### 4.2 Heredity, measured

E2 is, to our knowledge, the first test of vertical transmission between specific cultural works that uses units the analyst did not hand-code. It mirrors the logic of da Silva and Tehrani [6], with model neurons in place of a folklorist's tale-type index. The effect survived every check the reviewers asked for: comics that resemble everything, comics published after the film, placebo films, names, frequency bands and attribution strength. Proper names carried little of it: with names restored, the share of comics beating the source fell only from 16% to 12%. The sequel positive control was borderline in Qwen (4 of 7 against a bar of 5) and passed in Llama (5 of 7). The method keeps some power without names, but less than with them.

### 4.3 Selection is still out of reach

E3 found no sign that memes from popular early works spread further. This is the weakest test of the three. It has only 15 early works, it measures popularity in 2015 to 2026 rather than at release, and the baseline-film placebo produced a similar coefficient. A credible test of selection needs a larger lineage (every superhero film, or every adaptation of a public-domain tale) and success measured when each work appeared: box office, sales or critical reception. Taylor's own hypotheses give sharper targets. The Lindy effect predicts that a meme's age forecasts its survival [18, p. 95]. Faded memes should return after about 30 years, once enough time has passed "for theft to become nostalgia" [18, p. 27]. Memeplexes should be gained and lost as bundles [18, p. 190; 20]. And works just past the familiar should do best [18, pp. 43, 101].

### 4.4 Limitations

The texts are 2020s Wikipedia retellings, so the method sees what happens in a story, not how it was told at the time. The corpus is one franchise of 32 works, with a gap between 1966 and 1986, and some summaries are short (9 sentences for Zorro). The models are small and quantised (8B and 1.7B parameters at 4 bits), not the 32B model audited in [10]. Neuron labels were written by an AI reader and checked by a second, but never validated by steering, the stronger test [10]. Many weak memes are not shared across models, and some neurons track Wikipedia's house style rather than story content. The name-free texts depend on an AI rewriter to find names, and the diff that replaced them sometimes swallowed a neighbouring word. That bias falls hardest on Batman texts, which have the most names, so it makes E1 harder for Batman rather than easier. The image experiment is exploratory: one image per work, three Batman covers missing, and posters, covers and box art differ in kind.

### 4.5 Next

Three extensions would test the definition hardest. The first is to run it on da Silva and Tehrani's folktale corpus and compare neuron-defined memes with the expert tale-type codes. The second is to validate the strongest memes by steering, in a model as large as Qwen3-32B. The third is to measure selection on a lineage large enough to detect it. If those hold, the answer to the question in our title is a qualified yes. Creativity is not yet reverse-engineered, but its parts can now be counted, and their inheritance traced.

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
10. Cammarata N. Neural networks are already clean. Goodfire. [already-clean.goodfire.com](https://already-clean.goodfire.com/) (accessed 20 October 2026).
11. Qwen Team. 2025 Qwen3 technical report. arXiv:2505.09388.
12. Grattafiori A *et al.* 2024 The Llama 3 herd of models. arXiv:2407.21783.
13. Saitou N, Nei M. 1987 The neighbor-joining method: a new method for reconstructing phylogenetic trees. *Mol. Biol. Evol.* **4**, 406–425.
14. Mantel N. 1967 The detection of disease clustering and a generalized regression approach. *Cancer Res.* **27**, 209–220.
15. Cohen J. 1960 A coefficient of agreement for nominal scales. *Educ. Psychol. Meas.* **20**, 37–46.
16. Radford A *et al.* 2021 Learning transferable visual models from natural language supervision. In *Proc. 38th Int. Conf. Machine Learning*.
17. Oquab M *et al.* 2024 DINOv2: learning robust visual features without supervision. *Trans. Mach. Learn. Res.*
18. Taylor M. 2023 *Marketing memetics: reverse-engineering creativity to drive brand performance*. Page numbers refer to the r1 PDF.
19. Marti L, Wu S, Piantadosi ST, Kidd C. 2023 Latent diversity in human concepts. *Open Mind* **7**, 79–92.
20. Heylighen F, Chielens K. 2009 Evolution of culture, memetics. In *Encyclopedia of complexity and systems science* (ed. RA Meyers). New York, NY: Springer.
21. Quian Quiroga R, Reddy L, Kreiman G, Koch C, Fried I. 2005 Invariant visual representation by single neurons in the human brain. *Nature* **435**, 1102–1107.

## Data, code and AI use

**Data and code.** All code, the corpus, meme families, labels, experiment outputs and an interactive companion page are at [github.com/hammer-mt/phylomemetics](https://github.com/hammer-mt/phylomemetics). The code is under the MIT licence. Plot summaries and texts derived from them come from English Wikipedia and remain under CC BY-SA 4.0. Copyrighted posters and covers were analysed but are not redistributed; scripts re-download them from Wikipedia. Raw neuron activations (about 15 GB) are not stored but can be regenerated with the scripts in roughly 30 minutes on a laptop.

**Pre-registration.** The three experiments were specified, with falsifying results, before any experimental result was computed. The designs were then revised once after adversarial review; the reviews and the revised designs are in the repository.

**AI use.** The analysis code, neuron labels, name-free rewrites, adversarial design reviews and a first draft of this paper were produced with Claude (Anthropic), working at the author's direction. The original illustrations and test images on the companion page were generated with Gemini and depict no characters; the page also reproduces public-domain images from Wikimedia Commons.

**Competing interests.** None declared.

**Funding.** None.
