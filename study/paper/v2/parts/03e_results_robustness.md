### 3.10 Robustness

All of the checks below were fixed in the plan. Unless stated they are for Qwen3-32B.

**The threshold, the score and the cap.** Across 56 settings (a threshold of 20, 30, 40 or 50% of scale; the strongest sentence or the mean of the three strongest; a base-rate cap from 1% to 30%, or none), the name-free canon share never reaches the bar: its AUC against the hero films runs from 0.25 to 0.67, and in no setting is it above its shuffled-label null. Heredity is below 0.05 in 7 of the 8 settings with no cap and in 42 of the 45 settings where it can be evaluated. The residual decision test passes in none of the 8 uncapped settings. Selection is below 0.05 in no setting. The two smaller models give the same picture (Table 6).

| | Qwen3-32B | Qwen3-8B | Llama 3.1 8B |
| --- | --- | --- | --- |
| E1: canon share AUC, range over 56 settings | 0.25 to 0.67 | 0.26 to 0.79 | 0.27 to 0.85 |
| E2: uncapped settings below 0.05 | 7 of 8 | 7 of 8 | 8 of 8 |
| E2: evaluable settings below 0.05 | 42 of 45 | 36 of 40 | 40 of 41 |
| Residual decision test: uncapped settings below 0.05 | 0 of 8 | 0 of 8 | 0 of 8 |
| E3: settings below 0.05 | 0 of 56 | 0 of 51 | 0 of 52 |

**Table 6.** The sensitivity grid: threshold by score by base-rate cap. E1 values are whole-summary AUCs, which are not length-controlled.

Llama 3.1 8B's canon share does reach 0.80 in three settings (0.83, 0.85 and 0.83), all with no base-rate cap. With no cap the canon is hundreds to thousands of common units, and a score built from them rises with the length of a text, which on its own gives 0.76. Those settings have no openings version, so length cannot be ruled out for them. At the primary setting, where the openings exist, Llama's canon share is 0.39 on whole summaries and 0.40 on openings.

**The cap and heredity.** Heredity weakens as the cap tightens. In Qwen3-8B's E2 multiverse of 12,528 specifications, 81% of the uncapped name-free specifications are below 0.05, against 39% at a 5% cap and 10% at a 1% cap. Llama 3.1 8B goes from 99% to 80% and 29%. Words behave the same way: word presence restricted to words in at most 3% of ordinary plots loses the trail (p = 0.06). The source trail is carried by ordinary story vocabulary and the units that follow it, not by rare units. Version 1 reported that heredity held in 99.6% of its specifications. Under the new rule the comparable figure is lower, because the grid now includes the mean-of-three score at high thresholds and tight caps, where few units are left.

**Other versions of the rule, at the primary setting.** None of these changes a conclusion (Table 7).

| Version | E1 canon share AUC | E2 share | E2 exact p | Residual p | E3 rho |
| --- | --- | --- | --- | --- | --- |
| Primary | 0.36 | 0.149 | 0.0016 | 0.69 | 0.35 |
| Scale without the study texts | 0.35 | 0.149 | 0.0016 | 0.69 | 0.31 |
| Scale from the fifth-strongest example | 0.37 | 0.072 | 0.00005 | 0.73 | 0.34 |
| Openings only | 0.37 | 0.120 | 0.0003 | 0.40 | 0.40 |
| Codes left in | 0.43 | 0.124 | 0.0007 | 0.87 | 0.06 |
| Token after each code masked too | 0.34 | 0.109 | 0.0002 | 0.60 | 0.37 |
| Echo regime left in | 0.43 | 0.219 | 0.017 | 0.87 | 0.13 |
| Weak poles left out (under 25% of the other pole) | 0.36 | 0.124 | 0.0005 | 0.65 | 0.33 |
| Units scaled by the supplement left out | 0.37 | 0.123 | 0.0009 | 0.69 | 0.35 |

**Table 7.** Qwen3-32B under other versions of the presence rule. The residual p is the decision test's.

Two rows deserve a note. Leaving the echo regime in, as version 1 did, weakens heredity (22% against 15%) and costs the sequel control a pass (4 of 7). The echo is noise for this purpose, as the essay's account of it would predict. And taking each unit's scale from its fifth-strongest example, which one freak token cannot set, makes heredity stronger (7%).

**What changed a result: the pipeline, the rule or the model.** Every primary statistic was also computed under version 1's rule on the new activations (Table 8, Figure 1).

| Model | Version | E1 AUC | E2 share | E2 exact p | Residual p, version 1's form | Residual p, decision test | E3 rho |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | Version 1 as published (4-bit) | 0.52 | 16% | 0.0026 | 0.089 | | 0.25 |
| Qwen3-8B | Full precision, version 1's rule | 0.51 | 14% | 0.0003 | 0.015 | 0.22 | 0.29 |
| Qwen3-8B | Full precision, new rule | 0.26 | 8% | 0.0002 | 0.26 | 0.62 | 0.35 |
| Llama 3.1 8B | Version 1 as published (4-bit) | 0.54 | 10% | 0.0002 | 0.010 | | 0.00 |
| Llama 3.1 8B | Full precision, version 1's rule | 0.53 | 11% | 0.0003 | 0.035 | 0.14 | −0.04 |
| Llama 3.1 8B | Full precision, new rule | 0.39 | 14% | 0.0005 | 0.20 | 0.57 | 0.17 |
| Qwen3-32B | Full precision, version 1's rule | 0.52 | 14% | 0.0007 | 0.023 | 0.26 | 0.25 |
| Qwen3-32B | Full precision, new rule | 0.36 | 15% | 0.0016 | 0.20 | 0.69 | 0.35 |

**Table 8.** The same statistics by model, precision and rule. E1 is the name-free canon share against the hero films. The decision test did not exist in version 1.

Three things follow.

- *Full precision changed little.* Under version 1's rule the full-precision 8B models give nearly the numbers version 1 published, and Qwen3-32B gives the same again (E1 0.52, E2 14%).
- *The fall in E1 comes from the rule.* In all three models the canon share drops from about 0.52, which is chance, to between 0.26 and 0.39 when only strong activations count. Below chance means the hero films carry more of the canon than the name-free Batman works do (Section 3.3).
- *Version 1's residual test passes under version 1's rule in all three models,* Qwen3-8B included (p = 0.015, 0.035 and 0.023), and in none under the new rule. The decision test passes in none of the six. This is what a test that leans on common words would do: weak activations follow common words, and the form of the test version 1 used can be passed by a word measure that does the same (Section 2.5).

![Figure 1](fig_contrast.png)

**Figure 1.** What changed between versions. E1, E2 and the residual test for each model: as published in version 1, at full precision under version 1's rule, and under the new rule.

**Counting cells honestly.** The six model-by-rule cells analyse the same 32 works, so six results below 0.05 are not six independent findings. E2 is below 0.05 in all six. [CELLS-JOINT] The residual decision test is below 0.05 in none.

**Trees.** As in version 1, a tree built from the distances between works' meme sets does not recover documented sources. Ranking the 12 comics by distance to each adaptation puts the source at a mean rank of 4.4 in Qwen3-32B, 4.7 in Qwen3-8B and 4.6 in Llama 3.1 8B, against 5.8 by chance (p = 0.12, 0.17 and 0.14). TF-IDF distance on the same texts puts it at 2.4 (p = 0.0004). The models' distance matrices agree closely with one another (Mantel r = 0.98 to 0.99, and 0.94 to 0.96 with TF-IDF distance held fixed), well above matrices built from random units (0.64 to 0.79 on average). So the models agree about which works are alike, and what they agree on is not the line of descent. Works of the same era sit slightly closer together than works of different eras (ratio 1.03, p = 0.04 in Qwen3-32B).
