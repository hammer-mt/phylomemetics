### 3.2 Presence, the canon, and other film series

**Presence.** Of Qwen3-32B's 3,276,800 units, 1,812,516 fire strongly in at least one of the 114 works, and 885,060 of those are distinctive, meaning present in at most 3% of ordinary film plots. A typical Batman summary has 140,954 units present with its names and 102,256 without (medians). A third of all units (1,035,162) reach their strongest activation in the supplement of code, chat, other languages, news and reviews, which is what the supplement was added for. Only 4,518 take their scale from the study's own texts. For 82,189 positive poles, 5.0% of neurons, the strongest raw activation is an echo of the other sign regime. The essay puts that figure at about 4%, and under version 1's rule those echoes counted as the unit's own.

**The canon.** 7,520 distinctive units are present in at least 6 of the 32 Batman works. Merging those that fire strongly on the same sentences gives 4,996 meme families: 74 in the core (29 or more works), 135 common and 4,787 optional. Counted as units, 334 are in at least 23 works and 120 in at least 29 (Figure 2A). The two 8B models have fewer (44 and 59 units in at least 29 works) and Llama 3.3 70B about as many (114).

Remove the names and most of the canon goes with them. On the name-free texts 48 distinctive units are in at least 23 works and 12 in at least 29. E1's canon, the distinctive units present in at least 80% of the works, has 204 units, and exactly one of them is still present in 70% of the name-free works. Qwen3-8B has none, Llama 3.1 8B one and Llama 3.3 70B ten. Version 1 reported a name-free core of 7 units in Qwen3-8B (9 in Llama 3.1 8B) and read it as "close to the standard definition of a superhero" [26]. Under a rule that counts only strong activations, even that has all but gone. Section 3.7 says what the one surviving unit is.

**Is that more of a canon than other bodies of films have?** Three comparisons could have failed.

- *Random sets of 32 ordinary films.* With the 3% cap a random set cannot have a canon, by construction. Without the cap, Batman has 5,765 units in at least 29 works against a mean of 4,033 for 50 random sets, and the largest of the 50 has 4,492. So Batman's summaries share more units than random films do, but about 70% of what 32 Batman summaries share is what any 32 plot summaries share.
- *Genre sets.* When 32 plots of one genre stand in for the franchise, with base rates taken from the other genres, no drama, comedy, thriller or action set has a single distinctive unit in 23 of its 32 plots, and horror sets average 0.28. Batman has 334. A genre is not a canon by this measure.
- *Other film series.* Batman has more distinctive units in at least 90% of its works than four of the six series have in theirs, about as many as Kamen Rider, and a third as many as Doraemon (Table 2, Figure 2B). At the 72% bar Kamen Rider is ahead of Batman too.

| Series | Films | Units in 90% or more of the films | Batman works, random subsets of the same size (mean) |
| --- | --- | --- | --- |
| Godzilla | 28 | 89 | 102 |
| Frankenstein | 24 | 39 | 116 |
| Dracula | 23 | 38 | 119 |
| Doraemon | 17 | 327 | 108 |
| Tarzan | 16 | 50 | 114 |
| Kamen Rider | 15 | 119 | 128 |

**Table 2.** Batman against six other film series from the same plot dataset, Qwen3-32B, original texts: distinctive units present in at least 90% of a series' films.

So Batman does have a canon in this sense, and so do other series. What the count measures is how uniform a series' summaries are, in names and stock phrases as much as in story. A series with a fixed cast and a formula, as Doraemon has, scores highest. In all four models Doraemon is far ahead of the other series and Kamen Rider comes second.

![Figure 2](fig_canon.png)

**Figure 2.** The canon under the new rule, Qwen3-32B. (A) Units present in at least 6, 23 and 29 of the 32 Batman works: distinctive units on the original texts and on the name-free texts, and all units with no base-rate cap beside the mean for random sets of 32 ordinary films. (B) Distinctive units present in at least 90% of each film series' films, against Batman works in random subsets of the same size.

![Figure 3](fig_genome.png)

**Figure 3.** Which works carry which units. Rows are units chosen by rule and explained (Section 3.7), with the explainer's title. Columns are the 32 Batman works and then the 13 relatives, each in date order. The upper half of a cell is filled where the unit is present in the text as written, the lower half where it is present with every name removed. Titles in black were judged, blind, to describe an idea; titles in grey a particular name or word.

### 3.3 E1: identity

**Without names, Batman's canon does not tell Batman from other masked heroes.** For Qwen3-32B the canon share has an AUC of 0.36 against the eight hero films, with a 95% interval of 0.17 to 0.56 (Figure 4). The prediction needed 0.80, so by the criteria registered for version 1 it is falsified, as it is in Qwen3-8B (0.26) and Llama 3.1 8B (0.39). Llama 3.3 70B is different, and is taken up below. The score is at or below chance, which means the hero films carry more of Batman's canon than the name-free Batman works do. Superman, Spider-Man, Daredevil and Iron Man each carry more of it than at least 79% of the Batman works. The result is not a matter of length: on openings of equal length the AUC is 0.37.

The reason is visible in the units. With names present the canon share separates Batman from everything at an AUC of 1.00. Nearly all of those units respond to Batman's names and places, and the name-free texts delete them. What is left fires on superheroes in general, and the summaries of superhero films use that vocabulary more than Batman's own do. Against ordinary films and the literary relatives the name-free canon share still separates Batman (AUC 0.96 and 0.98), as in version 1. It tells a superhero story from other stories. It does not tell Batman from Superman.

**It was not the scoring rule.** Version 1 put this failure down to its scoring rule, because a word-based centroid score reached 0.80. Given the same rule, the units do no better: the centroid of unit activations has an AUC of 0.62, and on the openings 0.43. Its whole-summary value is no better than text length alone, which reaches 0.76 on whole summaries and 0.45 on openings. The word-based centroid stays ahead at 0.80, and it is the only score that also holds on the openings (0.73, above its own shuffled-label null). It still does not meet the full prediction, because the lower end of its interval is 0.64 against a required 0.65. A canon chosen on the name-free texts themselves, a control from version 1, reaches 0.60.

**The model itself does better than its canon.** Asked which hero a name-free summary is about, Qwen3-32B answers Batman for 21 of the 32 Batman works, and also for 4 of the 8 hero films (Table 3). The log-probability it gives to Batman's name separates Batman works from hero films with an AUC of 0.78, about as well as word counts and far better than the canon share. The two are related: across the 32 works the canon share and that log-probability correlate at rho = 0.63. So the canon units do track the model's recognition of Batman. But the recognition draws on more than the units that count as present.

**The largest model is the exception, and it is the one that knows the plots.** Llama 3.3 70B's canon share reaches 0.77 (95% interval 0.60 to 0.91), and 0.65 on openings. That misses the bar, the interval and the openings criterion, so the prediction is not met. It is not falsified either, and ten of this model's 215 canon units survive in the name-free works. This is also the model that says a name-free summary is about Batman for 29 of the 32 works and names the exact film for all 69 ordinary plots (Table 3). For the 29 works it names, the canon share separates them from the hero films at 0.81. For the three it does not name, the figure is 0.33. So in this model the canon units fire on a name-free text when the model has worked out whose story it is. That is recognition of summaries it has read. It is not a name-free essence, and the registered test cannot tell the two apart.

![Figure 4](fig_identity.png)

**Figure 4.** Telling name-free Batman summaries from the eight masked-hero and superhero films, by each model's units, by word counts and by length alone. Large marks are whole summaries with 95% bootstrap intervals; small marks use only the opening of each summary, which is the same length in both groups. The pre-registered bar is 0.80.

### 3.4 E2: heredity, and whether neurons add anything to words

**Adaptations carry their source comic's units.** In Qwen3-32B, on name-free texts, 15% of eligible comics beat the documented source, against 46% expected by chance (exact p = 0.0016; p = 0.0083 when comics are relabelled only within their era). The documented source is the closest eligible comic for four of the nine adaptations and second for two more (Figure 5). The real value is below every placebo target, and 5 of 7 sequels rank their prequel in the top two, so the positive control passes. Heredity is supported, as it was in version 1. Qwen3-8B gives 8% (p = 0.00018), Llama 3.1 8B 14% (p = 0.00047) and Llama 3.3 70B 6% (p = 0.00011). Llama 3.1 8B fails the sequel control (4 of 7), so its result is supported with a failed positive control.

**Word counts find the sources at least as well.** TF-IDF cosine on the same name-free texts gives 11% (p = 0.00031) and ranks the source first for six of the nine adaptations. Word presence gives 17% (p = 0.0018). Qwen3-32B does not pull ahead of words: its 15% is behind TF-IDF's 11%. Llama 3.3 70B has the best figure of any measure at 6%, and its residual test fails like the others.

**The neurons add nothing that these word measures do not have.** With the seven word measures regressed out of the neuron inheritance rate, 49% of comics beat the source, which is chance (decision test, exact p = 0.69). With every word and code measure regressed out, p = 0.86. In version 1's two-measure form p = 0.20. The other three models agree (decision test p = 0.62, 0.57 and 0.20). The reverse is not true: with the neurons regressed out of TF-IDF cosine, the words still find the sources (11%, p = 0.0004). On this corpus the words carry the trail and the neurons carry part of it.

Version 1 reported that this test passed for Llama 3.1 8B (p = 0.010 to 0.033) and read it as a hint that Llama's neurons add a little beyond words. That reading does not survive. The form of the test used there passes for word-only measures in the neurons' place (Section 2.5). Put through the decision test, no model passes under either rule (Section 3.10).

![Figure 5](fig_heredity.png)

**Figure 5.** Heredity on name-free texts. (A) For each adaptation, the rank of its documented source among the eligible comics, by each model's units and by TF-IDF word counts. (B) The share of eligible comics that beat the documented source, averaged over the nine adaptations, for units, for the two word measures, and for units after the word measures are regressed out.

### 3.5 E3: selection

Selection was not detected, as in version 1. In Qwen3-32B the correlation between an early work's popularity and the spread of its units into later works is rho = 0.35 (one-sided p = 0.37). Qwen3-8B gives 0.35 (p = 0.37), Llama 3.1 8B 0.17 (p = 0.56) and Llama 3.3 70B −0.17 (p = 0.83). With 15 early works the test would detect a true correlation of 0.3 about a quarter of the time, so this says little either way.

### 3.6 Does the model simply recognise the plots?

These models have read Wikipedia, and it shows (Table 3). Given a name-free summary, Qwen3-32B names the exact work for 52 of the 69 ordinary films and 5 of the 8 hero films. It does so for only 4 of the 32 Batman works: it knows it is reading Batman, but not which Batman. Llama 3.3 70B names the exact work for every one of the 69 ordinary films and for 15 of the Batman works.

| | Qwen3-32B | Qwen3-8B | Llama 3.1 8B | Llama 3.3 70B |
| --- | --- | --- | --- | --- |
| Batman works it says are about Batman | 21 of 32 | 20 of 32 | 16 of 32 | 29 of 32 |
| Batman works it identifies by title | 4 of 32 | 2 of 32 | 2 of 32 | 15 of 32 |
| Hero films it says are about Batman | 4 of 8 | 4 of 8 | 1 of 8 | 1 of 8 |
| Hero films whose own hero it names | 4 of 7 | 3 of 7 | 5 of 7 | 7 of 7 |
| Ordinary films it identifies by title | 52 of 69 | 36 of 69 | 54 of 69 | 69 of 69 |
| AUC of its log-probability of "Batman", Batman works against hero films | 0.78 | 0.57 | 0.76 | 0.90 |

**Table 3.** What each untouched model says about a name-free summary.

This matters for how the experiments are read. Had E1 passed, the paper would have had to say that the model recognises Batman's plots, which is a different thing from a name-free essence. In the three models up to 32B it failed, so the question does not arise. In Llama 3.3 70B it nearly passes, and Section 3.3 reads it that way. For E2 the worry runs the other way: could memorised plots carry the source trail? Qwen3-32B identifies both the adaptation and one of its sources for only one of the nine pairs (The Dark Knight Rises), and that pair is the one where the units place the source worst. Llama 3.3 70B identifies both for five of the nine, and its units place the source no better for those five than for the other four (7% and 6% of comics beat the source). So recall of specific works does not explain the heredity result.
