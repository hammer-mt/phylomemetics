# What changed from version 1 to version 2

Version 1 (20 October 2026) is frozen: `../paper.md`, its PDF, its figures and its Claude Doc are unchanged, and the git tag `paper-v1` marks it. Version 2 re-runs the whole study with a method aligned to the essay that inspired it (Cammarata, "Neural networks are already clean"), and reports what that did to each of version 1's conclusions. The analysis plan, `../../v2/PLAN.md`, was committed before any of the new models read a study text.

## Why

Version 1's own limitations section, its robustness review and use of the results since publication pointed at the same weak spots:

- The models were small (8B parameters) and quantised to 4 bits. The essay studies Qwen3-32B and says larger models have cleaner neurons.
- A meme counted as present in a work whenever the work beat 68 of 69 ordinary films on a neuron, however weak the activation. The essay's point is that strong activations are what carry meaning.
- "Ordinary" was 69 hand-picked films, which also set the bar. Two comparisons turned out to be true by construction and had to be withdrawn.
- Neuron labels were read off eight sentences of the Batman corpus itself, and several were narrower than their neuron.
- Nothing was tested by steering or ablation.
- Twins in a second model were found and scored on the same sentences.
- The order of plan and results rested on a session transcript, not on the repository's history.

## What changed in the method

| | Version 1 | Version 2 |
| --- | --- | --- |
| Models | Qwen3-8B and Llama 3.1 8B, 4-bit weights, on a laptop | Qwen3-32B (primary), Qwen3-8B, Llama 3.1 8B and Llama 3.3 70B, at full precision on rented GPUs |
| Unit | One pole of a neuron, whatever produced the activation | One pole of a neuron, counted only where the gate is open (the essay's main sign regimes) |
| Present | The work beats 68 of 69 reference films | The unit fires strongly: at least 30% of its strongest activation anywhere in a 48-million-token corpus |
| Ordinary stories | 69 films, which also set the bar | 2,999 film plots that are not study works give every unit's base rate |
| Which units | All | Distinctive units (base rate of 3% or less) for the canon and identity; all units for heredity and selection |
| Explanations | Read off 8 Batman sentences, checked by a second reader | Written from the unit's strongest examples in the reference corpus, for units chosen by rule, then tested on new texts where they can lose |
| Causal tests | None | Steering of the explained units; ablation of Batman's canon |
| Recognition | Not measured | Each model is asked which hero and which work a name-free summary is about |
| Words as a rival | Two word measures | The same two, plus the further measures needed so that none of the word-only measures we tried passes the "beyond words" test |
| Length | Checked by one control | Every identity score is also computed on openings of equal length |
| Twins across models | Scored on the sentences they were chosen on | Scored on new texts and on name-free sentences, against word-matched controls |
| Plan | Recorded in a session transcript | Committed to the repository before the study pass; changes logged |
| Extraction check | None | Our activations match the essay's published ones (r = 0.992 on 756 examples) |

The corpus (32 Batman works, 13 relatives, 69 films, and their name-free versions), the three experiments and their statistics are unchanged.

## What changed in the results

Numbers are for the primary model of each version (Qwen3-8B at 4 bits in version 1, Qwen3-32B at full precision in version 2) unless a model is named.

| Conclusion in version 1 | Version 2 | Verdict |
| --- | --- | --- |
| Heredity is supported: 16% of eligible comics beat the documented source (exact p = 0.0026) | 15% (exact p = 0.0016), and supported in each of the three other models (in Llama 3.1 8B with a failed positive control: 4 of 7 sequels) | Stands |
| Plain word overlap finds the sources as well as the neurons do | TF-IDF 11%, word presence 17%, units 15% | Stands |
| The test of whether neurons add anything beyond words passes for Llama 3.1 8B (p = 0.010 to 0.033) | That form of the test can be passed by word measures alone. Under a test that no word-only measure we tried passes, no model passes under either rule (lowest p = 0.14 over seven cells) | Withdrawn |
| Without names, canon share does not separate Batman from masked-hero films (AUC 0.52 to 0.54) | AUC 0.36, with an interval that excludes the bar; 0.26 and 0.39 in the two 8B models. Llama 3.3 70B reaches 0.77, short of the bar, and it recognises nearly every summary it is shown | Stands. Falsified by the registered criteria in three models, not met in the fourth |
| That null "comes from the scoring rule", because a word centroid reaches 0.80 | Given the same rule, the units reach 0.62, and 0.43 on openings of equal length; text length alone reaches 0.76 | Not supported |
| A name-free core of 7 units in Qwen3-8B (9 in Llama 3.1 8B) is "close to the standard definition of a superhero" | None in Qwen3-8B. One unit in Qwen3-32B, a superhero-comics genre unit that is also present in 5 of the 8 hero films (one and ten units in the two Llama models) | Does not survive the 30% rule |
| Heredity holds in 99.6% of 30,240 specifications | Lower under the new rule, because the grid now includes tight base-rate caps where few units are left. With no cap it holds in 7 of 8 settings | Qualified |
| Most of the strongest units detect names and words (24 name or place detectors and 3 common-word detectors among the 33 strongest, by hand) | 38 of the 60 explained core families are names, places or words, by a blind judge. The interval (52% to 75%) lies below version 1's 82% | Stands, less strongly |
| Selection was not detected | rho = 0.35, p = 0.37 | Stands |
| The strongest units have twins in Llama 3.1 8B (median r = 0.82), mostly for names; without names the agreement collapses | Twins respond to the same descriptions well above chance but much more weakly than the original unit, and on name-free sentences match no better than word-matched controls | Stands, with a new test |
| A few named memes survive rewording: 11 of 31 respond more to a rewording than to any everyday sentence, where about 5 would by mismatch | By the same criterion on the new activations, 8 of 31 (2.3 by mismatch). Under the rule that counts only strong responses, 1 of 31 (Two-Face's coin) | Qualified |
| The trees do not recover documented sources | Mean rank 4.4 against 5.8 by chance (p = 0.12); TF-IDF 2.4. In Llama 3.3 70B they do (2.2) | Stands up to 32B |

New in version 2:

- The explanation tests, a secondary check. In short texts written to fit the descriptions of 241 units chosen by rule, 81% of new examples and 3% of near misses fire the unit strongly, and 46% of units still fire when the description is put without its key words. In ordinary plots near the presence threshold a description accounts for about half of a unit's firings.
- Recognition. Qwen3-32B says a name-free Batman summary is about Batman for 21 of the 32 works, and names the exact work for 52 of 69 ordinary films.
- Steering. Not met on the study's units (17 of 126 idea units, against 20% by chance), although the same mechanism identifies 23 of 40 of the essay's gallery neurons, under an easier test.
- Ablation. Switching off Batman's canon lowers Qwen3-32B's log-odds of naming Batman by 2.39, against 0.02 for matched random sets, and does not lower other heroes' names on average. The test is not evaluable under a rule set beforehand (5 of 32 replies do not begin with a name, limit 4), so it is not counted as a pass. A post hoc look at the other 27 works gives the same picture (2.62 against 0.02). In Qwen3-8B the test could be evaluated and was not specific to Batman. A follow-up test, written after those results and fixed before it was run, applies the same criterion with the model's reply begun for it ("Answer:"), so that the next word is the name. It is met in Qwen3-32B (1.81 against 0.03; one reply of 32 does not begin with a name) and not met in Qwen3-8B (0.89 against 0.04, again not specific to Batman). It reads the same texts and units, it is not one of the six tests, and the registered test stays not evaluable.
- Other film series. Batman has more recurring distinctive units than four of six other series and fewer than Doraemon.
- A limit of the name-free texts. They hold more placeholder codes for Batman than for the hero films (median 105.5 against 70.5), and units that react to the gaps are the ones that best separate the two.

## What was withdrawn or corrected

- The reading that Llama's neurons add something beyond shared words.
- The reading that E1's failure was caused by the scoring rule.
- The name-free "superhero core" as a finding about Batman.
- The comparison of Batman's canon with random sets of ordinary films under a cap, which cannot fail. It is replaced by comparisons with genre sets and other film series.

## Draft of 9 November 2026: two more corpora

The draft of 8 November reported Batman alone. This draft adds two corpora, the Sherlock Holmes stories with their screen versions and film remakes, analysed under a second plan (`../../v2/second/PLAN.md`) that was committed before any model read them as study works. No result of the Batman sections changed.

**What was added**

- Section 2.11: the two corpora, the tests, and how the remakes test was calibrated with no model.
- Section 3.11 (Sherlock Holmes) and Section 3.12 (film remakes), with Tables 10 to 13 and Figures 8 and 9.
- The abstract, the introduction, Sections 3.1 and 4.1 to 4.5 and the closing notes now say what the two corpora add and what qualifies it.

**What the two corpora show**

| Question | Batman (above) | Two more corpora | Verdict |
| --- | --- | --- | --- |
| Heredity: do adaptations carry their source's units? | Supported (p = 0.0016) | Holmes (Qwen3-32B): 3.3% of stories rank above the documented story, against 44% by chance (p = 0.00001). Remakes (Qwen3-32B, the 120 pairs that are not copied): 1.6% of candidates rank above the source, against 50% (p = 0.0001, the smallest the null can give) | Supported on all three corpora; words find the sources too (at least as well on Batman and the remakes, about as well on Holmes) |
| Identity: without names, does the canon find the franchise? | Falsified (AUC 0.36) | Holmes, against detective films of the same era: 0.77 in Qwen3-32B, short of the bar of 0.80; 0.90 in Llama 3.3 70B, which meets all six clauses of the test. In each model the score follows the works the model can name | Not established; recognition is not ruled out |
| Selection | Not detected (27% power) | Holmes: rho = 0.14, p = 0.30 (about 32% power) | Not detected; the test cannot settle it |
| Do units add to words? | No (p = 0.69, nine adaptations) | Remakes: where a model built from 18 word measures is unsure of the source, the units of Qwen3-32B lean towards it (Z = 3.45; noise-calibrated p = 0.0005), under all seven conditions set in advance | Supported on the remakes, with qualifications |

**What qualifies the remakes result** (Section 3.12): the test's p values are too small, by a factor of two at 0.05 and by more below it, and its calibration condition was met at the margin; the result rests on about 21 open pairs and a few families; the test controlled for words only, and the year of release, put in the units' place, also scores above every word-only stand-in; a word measure built afterwards without names scores as high in the units' place, and with such measures added the units keep about half of their gain; it is larger for sources whose summaries were in the reference corpus; the two 8B models agree on the same pairs, and Llama 3.3 70B shows it only for films it recognises; adding the units puts the documented source first for one more pair (206 of the 220 against 205). Three audits made after the result was seen are reported as exploratory.

**Smaller changes**: Section 2.10 gives the plan's log as thirty-five entries; the GPU cost is updated; the date of the draft.
