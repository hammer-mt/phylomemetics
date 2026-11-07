# Tables for paper v2

Written by `v2/analysis/paper_tables.py` from the committed results. Do not edit by hand.

## Units and presence

| Model | Units | Present in some work | of those, distinctive | Per Batman work, original (median) | Per Batman work, name-free (median) | Scale set by the supplement | Scale set by the study texts | Positive units whose strongest raw activation is an echo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 884,736 | 455,422 | 238,530 | 37,127 | 23,442 | 304,263 | 1,161 | 26,262 |
| Llama 3.1 8B | 917,504 | 577,567 | 284,096 | 46,660 | 35,504 | 249,304 | 1,003 | 6,240 |

## E1, identity: name-free Batman works against the 8 hero films

| Score | AUC | 95% interval | Shuffle p95 | Openings AUC | Openings shuffle p95 | AUC without iconography | vs ordinary films | vs literary relatives | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
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
| Qwen3-8B | 75 | 78 | 0 | 5,101 | 2,763 | 2705.7 |
| Llama 3.1 8B | 101 | 104 | 1 | 7,752 | 5,485 | 4633.5 |

## E2, heredity, and the residual test

| Measure | Optional units | Share of comics beating the source | Exact p | Within-era p | Chance | Sequels in top two | Residual: decision test p | Residual: every measure p | Residual: v1's two-measure form p | Residual share (decision) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 105,147 | 0.077 | 0.00018 | 0.0028 | 0.483 | 6 of 7 | 0.62 | 0.66 | 0.26 | 0.473 |
| Llama 3.1 8B | 147,294 | 0.136 | 0.00047 | 0.0063 | 0.473 | 4 of 7 | 0.57 | 0.51 | 0.20 | 0.464 |
| Words: TF-IDF cosine |  | 0.110 | 0.00031 | 0.0028 | 0.467 | 6 of 7 |  |  |  |  |
| Words: word presence |  | 0.167 | 0.0018 | 0.0042 | 0.470 | 6 of 7 |  |  |  |  |

## E3, selection

| Model | rho | One-sided p | Units | rho on original texts | Placebo (base rate) rho |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 0.35 | 0.37 | 64,959 | -0.02 | -0.57 |
| Llama 3.1 8B | 0.17 | 0.56 | 90,873 | -0.27 | -0.41 |

## What changed a result: every model under both rules

| Cell | E1 AUC | E1 openings AUC | E2 share | E2 exact p | Residual p (two measures) | Residual p (decision) | Sequels | E3 rho | E3 p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-8b | v1 published | 0.515 | n/a | 0.164 | 0.0026 | 0.089 | n/a | 4 | 0.25 | 0.35 |
| qwen3-8b | full precision, v1 rule | 0.507 | n/a | 0.139 | 0.00034 | 0.015 | n/a | 5 | 0.29 | 0.31 |
| qwen3-8b | full precision, v2 rule | 0.260 | 0.291 | 0.077 | 0.00018 | 0.26 | 0.62 | 6 | 0.35 | 0.37 |
| llama3.1-8b | v1 published | 0.544 | n/a | 0.102 | 0.00017 | 0.010 | n/a | 5 | 0.00 | 0.58 |
| llama3.1-8b | full precision, v2 rule | 0.387 | 0.396 | 0.136 | 0.00047 | 0.20 | 0.57 | 4 | 0.17 | 0.56 |

### Differences between cells

| Contrast | Model | E1 AUC difference | 95% interval | E2 share difference | 95% interval |
| --- | --- | --- | --- | --- | --- |
| pipeline (4-bit and 8-bit storage against full precision), same model and rule | qwen3-8b | -0.008 | -0.035 to 0.018 | -0.025 | -0.116 to 0.051 |
| rule, same model | qwen3-8b | -0.248 | -0.325 to -0.167 | -0.062 | -0.149 to 0.026 |

## Sensitivity grid (threshold x score x cap; 56 settings)

| Model | Residual decision test below 0.05, uncapped settings | of which evaluable | E2 below 0.05, uncapped settings | E2 below 0.05, evaluable settings | E1 AUC range | Settings with E1 AUC of 0.80 or more | E3 rho range | E3 settings below 0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 0 of 8 | 7 | 7 of 8 | 36 of 40 | 0.26 to 0.79 | 0 | -0.63 to 0.46 | 0 |

## Canon counts (distinctive units present in at least k of 32 Batman works)

| Model | 6+ | 23+ | 29+ | Name-free 6+ | Name-free 23+ | Name-free 29+ | All units 29+ (no cap) | Random film sets, all units 29+ | Candidates by tier (core / common / optional) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 1,998 | 113 | 44 | 272 | 6 | 4 | 3,056 | 2626 | 44 / 69 / 1885 |
| Llama 3.1 8B | 3,279 | 192 | 59 | 561 | 10 | 5 | 5,012 | 3740 | 59 / 133 / 3087 |

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
| Qwen3-8B | 20 of 32 | 2 of 32 | 3 of 7 | 4 of 8 | 36 of 69 | -0.73 | -3.70 | -11.36 | 0.57 |
| Llama 3.1 8B | 16 of 32 | 2 of 32 | 5 of 7 | 1 of 8 | 54 of 69 | -1.27 | -5.23 | -8.53 | 0.76 |
