# Tables for paper v2

Written by `v2/analysis/paper_tables.py` from the committed results. Do not edit by hand.

## Units and presence

| Model | Units | Present in some work | of those, distinctive | Per Batman work, original (median) | Per Batman work, name-free (median) | Scale set by the supplement | Scale set by the study texts | Positive units whose strongest raw activation is an echo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-32B | 3,276,800 | 1,812,516 | 885,060 | 140,954 | 102,256 | n/a | n/a | n/a |
| Qwen3-8B | 884,736 | 455,422 | 238,530 | 37,127 | 23,442 | 304,263 | 1,161 | 26,262 |
| Llama 3.1 8B | 917,504 | 577,567 | 284,096 | 46,660 | 35,504 | 249,304 | 1,003 | 6,240 |

## E1, identity: name-free Batman works against the 8 hero films

| Score | AUC | 95% interval | Shuffle p95 | Openings AUC | Openings shuffle p95 | AUC without iconography | vs ordinary films | vs literary relatives | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-32B: canon share | 0.36 | 0.17 to 0.56 | 0.68 | 0.37 | 0.70 | 0.38 | 0.96 | 0.98 | falsified |
| Qwen3-32B: centroid of units | 0.62 | 0.38 to 0.84 | 0.68 | 0.43 | 0.70 | 0.61 | 0.91 | 0.93 | falsified |
| Qwen3-32B: centroid, distinctive units | 0.48 | 0.25 to 0.70 | 0.69 | 0.39 | 0.69 | 0.47 | 0.94 | 0.96 | falsified |
| Qwen3-8B: canon share | 0.26 | 0.12 to 0.43 | 0.68 | 0.29 | 0.70 | 0.28 | 0.88 | 0.80 | falsified |
| Qwen3-8B: centroid of units | 0.68 | 0.46 to 0.88 | 0.69 | 0.48 | 0.70 | 0.64 | 0.91 | 0.91 | falsified |
| Qwen3-8B: centroid, distinctive units | 0.43 | 0.21 to 0.65 | 0.68 | 0.40 | 0.69 | 0.45 | 0.93 | 0.95 | falsified |
| Llama 3.1 8B: canon share | 0.39 | 0.17 to 0.63 | 0.68 | 0.40 | 0.70 | 0.39 | 0.93 | 0.88 | falsified |
| Llama 3.1 8B: centroid of units | 0.69 | 0.48 to 0.86 | 0.68 | 0.46 | 0.70 | 0.65 | 0.94 | 0.94 | falsified |
| Llama 3.1 8B: centroid, distinctive units | 0.45 | 0.20 to 0.69 | 0.68 | 0.43 | 0.69 | 0.45 | 0.96 | 0.97 | falsified |
| Words: TF-IDF centroid | 0.80 | 0.64 to 0.93 | 0.69 | 0.73 | 0.69 | 0.76 | 0.97 | 0.96 | not met |
| Words: TF-IDF centroid, background weights | 0.83 | 0.68 to 0.95 | 0.69 | 0.77 | 0.69 | 0.77 | 0.98 | 0.97 | not met |
| Length alone (sentences) | 0.76 | 0.58 to 0.91 | 0.68 | 0.45 | 0.68 | 0.72 | 0.55 | 0.49 | not met |

## Canon and name-free core

| Model | Canon units (all 32 works) | Canon units per fold (median) | Name-free core | Canon without the cap | Name-free core without the cap | The same for random sets of 32 ordinary films |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-32B | 204 | 212 | 1 | 10,889 | 5,749 | 4690.0 |
| Qwen3-8B | 75 | 78 | 0 | 5,101 | 2,763 | 2705.7 |
| Llama 3.1 8B | 101 | 104 | 1 | 7,752 | 5,485 | 4633.5 |

## E2, heredity, and the residual test

| Measure | Optional units | Share of comics beating the source | Exact p | Within-era p | Chance | Sequels in top two | Residual: decision test p | Residual: every measure p | Residual: v1's two-measure form p | Residual share (decision) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-32B | 500,944 | 0.149 | 0.0016 | 0.0083 | 0.464 | 5 of 7 | 0.69 | 0.86 | 0.20 | 0.485 |
| Qwen3-8B | 105,147 | 0.077 | 0.00018 | 0.0028 | 0.483 | 6 of 7 | 0.62 | 0.66 | 0.26 | 0.473 |
| Llama 3.1 8B | 147,294 | 0.136 | 0.00047 | 0.0063 | 0.473 | 4 of 7 | 0.57 | 0.51 | 0.20 | 0.464 |
| Words: TF-IDF cosine |  | 0.110 | 0.00031 | 0.0028 | 0.467 | 6 of 7 |  |  |  |  |
| Words: word presence |  | 0.167 | 0.0018 | 0.0042 | 0.470 | 6 of 7 |  |  |  |  |

## E3, selection

| Model | rho | One-sided p | Units | rho on original texts | Placebo (base rate) rho |
| --- | --- | --- | --- | --- | --- |
| Qwen3-32B | 0.35 | 0.37 | 303,007 | -0.10 | -0.47 |
| Qwen3-8B | 0.35 | 0.37 | 64,959 | -0.02 | -0.57 |
| Llama 3.1 8B | 0.17 | 0.56 | 90,873 | -0.27 | -0.41 |

## What changed a result: every model under both rules

| Cell | E1 AUC | E1 openings AUC | E2 share | E2 exact p | Residual p (two measures) | Residual p (decision) | Sequels | E3 rho | E3 p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-8b: v1 published | 0.515 | n/a | 0.164 | 0.0026 | 0.089 | n/a | 4 | 0.25 | 0.35 |
| qwen3-8b: full precision, v1 rule | 0.507 | n/a | 0.139 | 0.00034 | 0.015 | 0.22 | 5 | 0.29 | 0.31 |
| qwen3-8b: full precision, v2 rule | 0.260 | 0.291 | 0.077 | 0.00018 | 0.26 | 0.62 | 6 | 0.35 | 0.37 |
| llama3.1-8b: v1 published | 0.544 | n/a | 0.102 | 0.00017 | 0.010 | n/a | 5 | 0.00 | 0.58 |
| llama3.1-8b: full precision, v1 rule | 0.529 | n/a | 0.114 | 0.00034 | 0.035 | 0.14 | 5 | -0.04 | 0.63 |
| llama3.1-8b: full precision, v2 rule | 0.387 | 0.396 | 0.136 | 0.00047 | 0.20 | 0.57 | 4 | 0.17 | 0.56 |

### Differences between cells

| Contrast | Model | E1 AUC difference | 95% interval | E2 share difference | 95% interval |
| --- | --- | --- | --- | --- | --- |
| pipeline (4-bit and 8-bit storage against full precision), same model and rule | qwen3-8b | -0.008 | -0.071 to 0.048 | -0.025 | -0.116 to 0.052 |
| rule, same model | qwen3-8b | -0.248 | -0.375 to -0.114 | -0.062 | -0.154 to 0.026 |
| pipeline (4-bit and 8-bit storage against full precision), same model and rule | llama3.1-8b | -0.015 | -0.055 to 0.017 | 0.012 | -0.025 to 0.050 |
| rule, same model | llama3.1-8b | -0.142 | -0.256 to -0.043 | 0.022 | -0.028 to 0.081 |

### Cells below 0.05

| Test | Cells | Below 0.05 | 0.05 x cells |
| --- | --- | --- | --- |
| E2_exact_p | 4 | 4 | 0.2 |
| residual_decision_test | 4 | 0 | 0.2 |
| residual_registered_two_measure_form_which_can_pass_on_words_alone | 4 | 2 | 0.2 |
| E3_p | 4 | 0 | 0.2 |
| E1_openings_against_their_own_label_shuffle_v2_rule_only | 2 | 0 | 0.1 |

### The same counts under one null for all cells

| Test | Cells | Below 0.05 | Probability of at least that many |
| --- | --- | --- | --- |
| E2 | 4 | 4 | 0.014 |
| residual_decision_test | 4 | 0 | 1.00 |
| residual_registered_two_measure_form | 4 | 2 | 0.055 |

## Sensitivity grid (threshold x score x cap; 56 settings)

| Model | Residual decision test below 0.05, uncapped settings | of which evaluable | E2 below 0.05, uncapped settings | E2 below 0.05, evaluable settings | E1 AUC range | Settings with E1 AUC of 0.80 or more | E3 rho range | E3 settings below 0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-32B | 0 of 8 | 8 | 7 of 8 | 42 of 45 | 0.25 to 0.67 | 0 | -0.72 to 0.35 | 0 |
| Qwen3-8B | 0 of 8 | 7 | 7 of 8 | 36 of 40 | 0.26 to 0.79 | 0 | -0.63 to 0.46 | 0 |
| Llama 3.1 8B | 0 of 8 | 8 | 8 of 8 | 40 of 41 | 0.27 to 0.85 | 3 | -0.54 to 0.59 | 0 |

## Canon counts (distinctive units present in at least k of 32 Batman works)

| Model | 6+ | 23+ | 29+ | Name-free 6+ | Name-free 23+ | Name-free 29+ | All units 29+ (no cap) | Random film sets, all units 29+ | Candidates by tier (core / common / optional) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-32B | 7,520 | 334 | 120 | 1,494 | 48 | 12 | 5,765 | 4033 | 120 / 214 / 7186 |
| Qwen3-8B | 1,998 | 113 | 44 | 272 | 6 | 4 | 3,056 | 2626 | 44 / 69 / 1885 |
| Llama 3.1 8B | 3,279 | 192 | 59 | 561 | 10 | 5 | 5,012 | 3740 | 59 / 133 / 3087 |

## Other film series, Qwen3-32B (distinctive units in at least 72% and 90% of the films)

| Series | Films | 72%+ | Batman subsets, 72%+ | 90%+ | Batman subsets, 90%+ |
| --- | --- | --- | --- | --- | --- |
| Godzilla | 28 | 236 | 291 | 89 | 102 |
| Frankenstein | 24 | 74 | 302 | 39 | 116 |
| Dracula | 23 | 101 | 319 | 38 | 119 |
| Doraemon | 17 | 778 | 305 | 327 | 108 |
| Tarzan | 16 | 113 | 334 | 50 | 114 |
| Kamen Rider | 15 | 542 | 390 | 119 | 128 |
| Batman (all 32) | 32 | 279 |  | 120 |  |

### Genre sets of 32 plots, Qwen3-32B

| Genre | Plots | Units in 6+ (mean) | 23+ (mean) | 23+ (max) | 29+ (mean) |
| --- | --- | --- | --- | --- | --- |
| drama | 433 | 95 | 0.0 | 0 | 0.0 |
| comedy | 321 | 192 | 0.0 | 0 | 0.0 |
| horror | 164 | 568 | 0.3 | 1 | 0.0 |
| thriller | 114 | 101 | 0.0 | 0 | 0.0 |
| action | 99 | 223 | 0.0 | 0 | 0.0 |

## Other film series, Qwen3-8B (distinctive units in at least 72% and 90% of the films)

| Series | Films | 72%+ | Batman subsets, 72%+ | 90%+ | Batman subsets, 90%+ |
| --- | --- | --- | --- | --- | --- |
| Godzilla | 28 | 65 | 99 | 25 | 39 |
| Frankenstein | 24 | 46 | 104 | 26 | 44 |
| Dracula | 23 | 36 | 107 | 18 | 46 |
| Doraemon | 17 | 247 | 103 | 128 | 40 |
| Tarzan | 16 | 38 | 115 | 21 | 42 |
| Kamen Rider | 15 | 197 | 129 | 28 | 47 |
| Batman (all 32) | 32 | 94 |  | 44 |  |

### Genre sets of 32 plots, Qwen3-8B

| Genre | Plots | Units in 6+ (mean) | 23+ (mean) | 23+ (max) | 29+ (mean) |
| --- | --- | --- | --- | --- | --- |
| drama | 433 | 29 | 0.0 | 0 | 0.0 |
| comedy | 321 | 44 | 0.0 | 0 | 0.0 |
| horror | 164 | 138 | 0.0 | 0 | 0.0 |
| thriller | 114 | 25 | 0.0 | 0 | 0.0 |
| action | 99 | 58 | 0.0 | 0 | 0.0 |

## Other film series, Llama 3.1 8B (distinctive units in at least 72% and 90% of the films)

| Series | Films | 72%+ | Batman subsets, 72%+ | 90%+ | Batman subsets, 90%+ |
| --- | --- | --- | --- | --- | --- |
| Godzilla | 28 | 128 | 155 | 42 | 54 |
| Frankenstein | 24 | 40 | 164 | 20 | 60 |
| Dracula | 23 | 57 | 174 | 22 | 62 |
| Doraemon | 17 | 342 | 166 | 159 | 56 |
| Tarzan | 16 | 77 | 187 | 34 | 58 |
| Kamen Rider | 15 | 300 | 222 | 61 | 67 |
| Batman (all 32) | 32 | 142 |  | 59 |  |

### Genre sets of 32 plots, Llama 3.1 8B

| Genre | Plots | Units in 6+ (mean) | 23+ (mean) | 23+ (max) | 29+ (mean) |
| --- | --- | --- | --- | --- | --- |
| drama | 433 | 48 | 0.0 | 0 | 0.0 |
| comedy | 321 | 92 | 0.0 | 0 | 0.0 |
| horror | 164 | 253 | 0.0 | 0 | 0.0 |
| thriller | 114 | 38 | 0.0 | 0 | 0.0 |
| action | 99 | 121 | 0.0 | 0 | 0.0 |

## Recognition: what the untouched model says about a name-free summary

| Model | Batman works it calls Batman | Batman works it identifies by title | Hero films whose hero it names | Hero films it calls Batman | Ordinary films it identifies by title | Median log p of Batman: Batman works | hero films | ordinary films | AUC of that log p, Batman vs hero films |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-32B | 21 of 32 | 4 of 32 | 4 of 7 | 4 of 8 | 52 of 69 | -0.57 | -5.61 | -13.24 | 0.78 |
| Qwen3-8B | 20 of 32 | 2 of 32 | 3 of 7 | 4 of 8 | 36 of 69 | -0.73 | -3.70 | -11.36 | 0.57 |
| Llama 3.1 8B | 16 of 32 | 2 of 32 | 5 of 7 | 1 of 8 | 54 of 69 | -1.27 | -5.23 | -8.53 | 0.76 |

## Explanation tests

| Units | Explained | Not testable out of context | New examples strong (A) | Idea without the key words (B) | Near misses (C) | Distant (D) | Won / tied / lost | Idea-bound | Blind maximum reaches the corpus maximum | AUC B vs C: unit | keyword terms | TF-IDF to examples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-32B | 241 | 0 | 81% | 45% | 3% | 0.1% | 232 / 9 / 0 | 110 (46%) | 14% | 0.87 | 0.51 | 0.45 |
| Qwen3-32B: idea units | 126 | 0 | 74% | 51% | 2% | 0.2% | 118 / 8 / 0 | 72 (57%) | 9% | 0.92 | 0.52 | 0.46 |
| Qwen3-32B: name or word units | 115 | 0 | 88% | 38% | 5% | 0.0% | 114 / 1 / 0 | 38 (33%) | 19% | 0.82 | 0.51 | 0.43 |
| Qwen3-32B: other units' texts (floor) |  |  | 6% | 5% |  |  | win rate 17% | 2.4% |  |  |  |  |
| Qwen3-8B | 207 | 0 | 85% | 40% | 4% | 0.7% | 201 / 6 / 0 | 73 (35%) | 20% | 0.87 | 0.51 | 0.44 |
| Qwen3-8B: idea units | 124 | 0 | 83% | 44% | 2% | 0.8% | 121 / 3 / 0 | 52 (42%) | 18% | 0.93 | 0.50 | 0.42 |
| Qwen3-8B: name or word units | 83 | 0 | 87% | 33% | 6% | 0.5% | 80 / 3 / 0 | 21 (25%) | 23% | 0.79 | 0.53 | 0.46 |
| Qwen3-8B: other units' texts (floor) |  |  | 4% | 3% |  |  | win rate 13% | 1.3% |  |  |  |  |
| The essay's gallery (benchmark) |  |  | 76% |  | 11% | 1% | 164 / 15 / 4 |  |  |  |  |  |

### By kind (blind kind judge)

| Model | Kind | Units | A | B | C | Won / tied / lost | Idea-bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-32B | character or name | 50 | 93% | 62% | 6% | 50 / 0 / 0 | 32 |
| Qwen3-32B | genre or setting | 23 | 99% | 92% | 4% | 23 / 0 / 0 | 23 |
| Qwen3-32B | grammar | 53 | 68% | 34% | 1% | 48 / 5 / 0 | 19 |
| Qwen3-32B | object | 5 | 88% | 72% | 0% | 5 / 0 / 0 | 4 |
| Qwen3-32B | other | 24 | 59% | 31% | 2% | 21 / 3 / 0 | 8 |
| Qwen3-32B | place | 7 | 83% | 40% | 9% | 7 / 0 / 0 | 2 |
| Qwen3-32B | story pattern | 12 | 77% | 55% | 0% | 12 / 0 / 0 | 7 |
| Qwen3-32B | style or format | 11 | 53% | 38% | 0% | 10 / 1 / 0 | 4 |
| Qwen3-32B | word or token | 56 | 89% | 24% | 5% | 56 / 0 / 0 | 11 |
| Qwen3-32B | core canon representatives judged a name, place or word | 60 |  |  |  | 38 of 60 | 95% interval 0.52 to 0.75 |
| Qwen3-8B | character or name | 39 | 93% | 45% | 5% | 39 / 0 / 0 | 15 |
| Qwen3-8B | genre or setting | 17 | 98% | 64% | 0% | 17 / 0 / 0 | 12 |
| Qwen3-8B | grammar | 55 | 68% | 28% | 1% | 52 / 3 / 0 | 12 |
| Qwen3-8B | object | 5 | 92% | 68% | 12% | 5 / 0 / 0 | 3 |
| Qwen3-8B | other | 22 | 83% | 52% | 1% | 21 / 1 / 0 | 12 |
| Qwen3-8B | place | 10 | 98% | 56% | 8% | 10 / 0 / 0 | 5 |
| Qwen3-8B | story pattern | 8 | 98% | 50% | 2% | 8 / 0 / 0 | 4 |
| Qwen3-8B | style or format | 5 | 92% | 32% | 4% | 5 / 0 / 0 | 1 |
| Qwen3-8B | word or token | 46 | 87% | 27% | 7% | 44 / 2 / 0 | 9 |
| Qwen3-8B | core canon representatives judged a name, place or word | 28 |  |  |  | 16 of 28 | 95% interval 0.39 to 0.75 |

### Units against word measures on the same texts (AUC for B against C)

| Model | Units | Word measure | n | Unit mean | Words mean | Unit better | Words better | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-32B | all units | analyst_keywords | 241 | 0.87 | 0.51 | 216 | 23 | 1.6e-36 |
| Qwen3-32B | all units | tfidf_to_examples | 241 | 0.87 | 0.45 | 225 | 13 | 1.7e-38 |
| Qwen3-32B | idea units | analyst_keywords | 126 | 0.92 | 0.52 | 119 | 6 | 1.6e-21 |
| Qwen3-32B | idea units | tfidf_to_examples | 126 | 0.92 | 0.46 | 120 | 5 | 1.5e-21 |
| Qwen3-8B | all units | analyst_keywords | 207 | 0.87 | 0.51 | 188 | 14 | 8.6e-31 |
| Qwen3-8B | all units | tfidf_to_examples | 207 | 0.87 | 0.44 | 187 | 19 | 4.4e-31 |
| Qwen3-8B | idea units | analyst_keywords | 124 | 0.93 | 0.50 | 118 | 3 | 1.8e-21 |
| Qwen3-8B | idea units | tfidf_to_examples | 124 | 0.93 | 0.42 | 118 | 6 | 1.1e-21 |

## Checks of the explanations that need no judge

| Model | Units | n | Peaks sharing the commonest stem | Peaks on punctuation or function words | Best single word r (median) | Held-out examples with an explainer key word | Random windows | Held-out examples with an analyst term | Random windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-32B | all | 241 | 37% | 46% | 0.36 | 44% | 12% | 40% | 10% |
| Qwen3-32B | name or word | 115 | 42% | 37% | 0.40 | 46% | 9% | 40% | 7% |
| Qwen3-32B | idea | 126 | 32% | 55% | 0.33 | 43% | 15% | 41% | 13% |
| Qwen3-8B | all | 207 | 44% | 49% | 0.33 | 47% | 11% | 46% | 10% |
| Qwen3-8B | name or word | 83 | 50% | 40% | 0.38 | 51% | 10% | 46% | 7% |
| Qwen3-8B | idea | 124 | 40% | 56% | 0.29 | 45% | 13% | 46% | 12% |

## Coverage: does the description predict firings at 30 to 60% of scale?

| Model | Units or band | Units | Firing passages | Covered by the description | Silent passages | Wrongly claimed |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-32B | all | 239 | 1419 | 48% | 1418 | 3% |
| Qwen3-32B | idea units | 125 | 742 | 52% | 742 | 4% |
| Qwen3-32B | name or word units | 114 | 677 | 44% | 676 | 2% |
| Qwen3-32B | 30-40% | 239 | 1004 | 43% | 1418 | 3% |
| Qwen3-32B | 40-50% | 239 | 305 | 59% | 1418 | 3% |
| Qwen3-32B | 50-60% | 239 | 110 | 65% | 1418 | 3% |

## Qwen3-32B as a second judge: agreement with the Claude judge

| Units of | Judgment | Items | Agreement | Kappa |
| --- | --- | --- | --- | --- |
| Qwen3-32B | kind | 241 | 55% | 0.45 |
| Qwen3-32B | detects | 241 | 80% | 0.61 |
| Qwen3-8B | kind | 207 | 58% | 0.49 |
| Qwen3-8B | detects | 207 | 80% | 0.61 |

## Bigger is cleaner? Win rate, Qwen3-32B against Qwen3-8B

| Comparison | Units (big) | Units (small) | Win rate (big) | Win rate (small) | Difference | One-sided p |
| --- | --- | --- | --- | --- | --- | --- |
| testable_units | 241 | 207 | 96% | 97% | -0.008 | 0.82 |
| all_units | 241 | 207 | 96% | 97% | -0.008 | 0.81 |
| testable_units_within_selection_set | 241 | 207 | 96% | 97% | -0.008 | 0.86 |
| testable_units_within_set_and_kind | 241 | 207 | 96% | 97% | -0.008 | 0.86 |

### Other differences (big minus small)

| Measure | Big | Small | Difference | 95% interval |
| --- | --- | --- | --- | --- |
| new_example_strong_rate | 0.806 | 0.847 | -0.042 | -0.091 to 0.008 |
| near_miss_strong_rate | 0.035 | 0.038 | -0.003 | -0.023 to 0.017 |
| idea_bound_share | 0.456 | 0.353 | 0.104 | 0.013 to 0.191 |
| one_idea_score | 6.000 | 6.406 | -0.406 | -0.713 to -0.100 |

## Steering calibration on 40 gallery neurons (chance 20%; chosen multiple 5, p = 2e-07)

| Multiple | Items | Correct | Accuracy | Answered none |
| --- | --- | --- | --- | --- |
| 1 x scale | 40 | 12 | 30% | 23 |
| 2 x scale | 40 | 13 | 32% | 20 |
| 3 x scale | 40 | 18 | 45% | 14 |
| 5 x scale | 40 | 23 | 57% | 6 |

## Steering the explained units (chance 20%)

| Model | Units | Items | Correct | Accuracy | Answered none | Binomial p |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-32B | idea units (criterion) | 126 | 17 | 13% | 63 | 0.98 |
| Qwen3-32B | name or word units | 115 | 12 | 10% | 84 | 1.00 |
| Qwen3-32B | all units | 241 | 29 | 12% | 147 | 1.00 |
| Qwen3-32B | idea units, four distractors of their own kind | 126 | 17 | 13% | 63 | 0.98 |
| Qwen3-32B | depth quarter 1 | 67 | 6 | 9% | 51 | 1.00 |
| Qwen3-32B | depth quarter 2 | 34 | 5 | 15% | 24 | 0.84 |
| Qwen3-32B | depth quarter 3 | 63 | 11 | 17% | 26 | 0.74 |
| Qwen3-32B | depth quarter 4 | 77 | 7 | 9% | 46 | 1.00 |
| Qwen3-32B | kind: character or name | 50 | 5 | 10% | 36 | 0.98 |
| Qwen3-32B | kind: genre or setting | 23 | 7 | 30% | 7 | 0.16 |
| Qwen3-32B | kind: grammar | 53 | 5 | 9% | 30 | 0.99 |
| Qwen3-32B | kind: object | 5 | 1 | 20% | 2 | 0.67 |
| Qwen3-32B | kind: other | 24 | 1 | 4% | 15 | 1.00 |
| Qwen3-32B | kind: place | 7 | 1 | 14% | 4 | 0.79 |
| Qwen3-32B | kind: story pattern | 12 | 1 | 8% | 6 | 0.93 |
| Qwen3-32B | kind: style or format | 11 | 1 | 9% | 5 | 0.91 |
| Qwen3-32B | kind: word or token | 56 | 7 | 12% | 42 | 0.95 |
| Qwen3-32B | sham items (own title removed): answered none | 40 | 21 | 52% |  |  |
| Qwen3-32B | idea units: continuations with a key word (pushed / not pushed / other units pushed) |  |  | 24% / 25% / 22% |  |  |

## Ablation: switching off Batman's canon (drop in the log-odds of the hero's name)

Only the first version for each model is a test. The others use 20 sets and are reported as the canon's drop beside the sets' and its rank among them.

| Model | Version | Random sets | Canon, Batman works | Sets (mean) | Sets (max) | p | Canon's rank | Activity-matched sets (max) | p | Canon's rank | Canon, hero films (own hero) | Specificity p | Activation removed: canon | activity-matched sets (mean) | Replies naming Batman: untouched -> canon off |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-32B | summary tokens, name-free (confirmatory) | 200 | 2.39 | 0.020 | 0.115 | 0.0050 | 1 of 201 | 0.111 | 0.0050 | 1 of 201 | -0.07 | 0.0017 | 41201 | 24749 | 20 -> 13 |
| Qwen3-32B | criterion |  |  |  |  | 0.0050 |  |  | 0.0050 |  |  | 0.0017 |  |  | untouched replies that do not start with a name: 5 of 32 (limit 4); evaluable False; met False |
| Qwen3-8B | summary tokens, name-free (confirmatory) | 200 | 1.33 | 0.033 | 0.113 | 0.0050 | 1 of 201 | 0.122 | 0.0050 | 1 of 201 | 0.80 | 0.28 | 8366 | 4933 | 20 -> 16 |
| Qwen3-8B | every position | 20 | 5.10 | 0.041 | 0.494 |  | 1 of 21 | 0.107 |  | 1 of 21 | 2.06 |  | 9075 | 5078 | 20 -> 2 |
| Qwen3-8B | original texts | 20 | 1.28 | 0.061 | 0.113 |  | 1 of 21 | 0.127 |  | 1 of 21 | 0.83 |  | 30057 | 6338 | 26 -> 28 |
| Qwen3-8B | strong firings only | 20 | 0.34 | 0.015 | 0.044 |  | 1 of 21 | 0.052 |  | 1 of 21 | 0.15 |  | 111 | 93 | 20 -> 18 |
| Qwen3-8B | weak activity only | 20 | 1.05 | 0.033 | 0.093 |  | 1 of 21 | 0.081 |  | 1 of 21 | 0.32 |  | 8718 | 4735 | 20 -> 16 |
| Qwen3-8B | criterion |  |  |  |  | 0.0050 |  |  | 0.0050 |  |  | 0.28 |  |  | untouched replies that do not start with a name: 0 of 32 (limit 4); evaluable True; met False |

## Twins of the Qwen3-32B units in other models

| Other model | Units | n | Twin r, original sentences (median) | Twin r, name-free | Word-matched controls, name-free | Depth-matched controls, name-free | Twin minus word-matched controls (95% interval) | Units with tester texts | Twin on the tester's texts: won / tied / lost | A | B | C | Idea-bound | Mismatched texts | p against mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | all units | 241 | 0.61 | 0.28 | 0.38 | 0.30 | -0.094 (-0.130 to -0.059) | 241 | 146 / 90 / 5 | 33% | 22% | 3% | 19% | 5% | 0.0001 |
| Qwen3-8B | idea units | 126 | 0.60 | 0.31 | 0.35 | 0.29 | -0.024 (-0.065 to 0.015) | 126 | 79 / 46 / 1 | 34% | 24% | 2% | 21% | 6% | 0.0001 |
| Qwen3-8B | name or word units | 115 | 0.63 | 0.22 | 0.44 | 0.31 | -0.171 (-0.224 to -0.119) | 115 | 67 / 44 / 4 | 32% | 20% | 5% | 16% | 5% | 0.0001 |
| Qwen3-8B | twins beat both nulls for idea units |  |  |  |  |  |  |  |  |  |  |  |  |  | False |
