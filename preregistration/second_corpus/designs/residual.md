# Design report: residual

One of three designs for the remakes test of whether units add to word overlap (study/v2/second/PLAN.md, Section 9). Written by an AI agent working apart from the other two, on word data only, before any model had read a remake text. It was calibrated on lists the scouts had recalled, with the narrower candidate frame; the plan re-runs the calibration on the final list. Paths under /tmp are the agent's working folder and are not in the repository.

### name

Word-conditioned relabelling test of the unit residual (remakes residual test)

### specification

QUESTION AND NULL. Given the word measures, do the unit scores still say which film of the pool is the source? H0: within each remake's pool, conditional on the word measures of every cell, the documented source is independent of the unit scores.

1. CELLS. For every remake r of the final pair list, the cells are (r, c) for every c of the fixed pool C(r); the source s(r) is one of them. Texts are the original texts, first 40 sentences. A family is a connected set of films linked by documented remake statements. The family is the cluster everywhere: fits leave a whole family out, relabelling draws are shared within a family, deviations are averaged within a family.

2. UNIT SCORE. u(r,c) = the hub-corrected z of the containment rate, as supplied (primary). The raw containment rate and whatever other unit settings the plan lists are sensitivity settings.

3. WORD MEASURES. The 33 covariates of the next field, computed over the films read (every pool film, remake and source).

4. STANDARDISING WITHIN A POOL. Each word measure and the unit score become z = (x - m)/s within each pool, with m and s the mean and SD of the pool's values after leaving out its k highest values, k = max(2, ceil(1% of the pool size)). If s = 0 the SD of all values is used; if that is 0, z = 0. Word z values are clipped to [-5, 10].

5. WORD-ONLY SOURCE MODEL. Conditional logit over each pool: p_r(c) proportional to exp(beta . z_r(c)), 33 coefficients, no intercept. Penalised maximum likelihood (L-BFGS), each remake weighted 1/(remakes in its family), ridge penalty (lambda/2)|beta|^2. lambda is the value of {1, 3, 10, 30} with the highest leave-one-family-out log-likelihood. The probabilities used, p^_r(c), always come from the fit that left r's family out.

6. RESIDUAL. y_r(c) = z of the unit score. One pooled ridge regression (penalty 1.0, intercept unpenalised) of y (clipped to [-5, 10] for the fit only) on 67 columns: 1, z_k and max(z_k - 2, 0) for the 33 measures, over all cells of the other families (leave one family out). e_r(c) = y_r(c) - fitted value. q_r(c) = normal score of e within the pool: Phi^-1((rank - 0.5)/n), ties sharing the mean rank.

7. STATISTIC. D_r = q_r(s(r)) - sum_c p^_r(c) q_r(c). D_f = mean of D_r over the remakes of family f. T = mean of D_f over families.

8. NULL AND P (one-sided: the source's residual is higher than the word model expects).
(a) Relabelling: 20,000 draws, seed 7. In each draw every family draws one uniform number U; each of its remakes has its source redrawn as the candidate at which the running total of p^_r, candidates taken in order of falling p^_r (ties in pool order), first reaches U. T* is T with the redrawn sources. p_relabel = (1 + number of T* >= T)/20,001; z_relabel = (T - mean T*)/SD T*.
(b) Spread: z_spread = sum_f D_f / sqrt(sum_f D_f^2); p_spread = 1 - Phi(z_spread).
p = max(p_relabel, p_spread).

9. PLACEBOS (words only; run and recorded before the unit score is loaded). 95 word-only measures that are not covariates: 30 + 19 named (wordlib.placebos_batch1, placebos_batch2), 6 curved functions of covariates (calibrate.curved), 40 random (wordlib.random_placebos, 20 random halves of the vocabulary and 20 log-normal word weightings, seed 11). Each goes through steps 4 to 8 in the unit score's place twice: as it is, and hub-corrected (each cell standardised by the candidate's mean and SD over the remakes of other families, related cells left out, SD floor = 0.5 x the median SD over candidates). That is 190 runs. The bar B is the largest z_relabel among them.

10. EVALUABLE if at least 10 families are open; a family is open when the mean of p^_r(s(r)) over its remakes is below 0.9.

11. POSITIVE CONTROL. The 33 covariates are recomputed between each whole remake and the FIRST half (first ceil(n/2) sentences) of every candidate; the unit score, unchanged, goes through steps 4 to 8 against them.

12. REPORTED WHATEVER THE OUTCOME. T, both z and both p; the placebo table; the estimated boost = sum_f D_f / sum_f mean_r(1 - p^_r(s(r))) with +/- 1.96 x sqrt(sum_f D_f^2) over the same denominator; for each remake the word probability and word rank of its source and the source's residual score; z_spread without the most favourable family; the same split by whether the model names the films and by the share of copied 5-word sequences.

ORDER OF WORK. Steps 3 to 5, 9 and 10 use words only and are computed, recorded and committed on the final pair list before any unit array is opened.

CONSTANTS (all in remake_residual_test.py): trim 1% and at least 2; clip [-5, 10]; bend at 2; regression ridge 1.0; word-model ridge grid {1, 3, 10, 30}; 20,000 draws; seed 7; alpha 0.05; open below 0.9; at least 10 open families; hub floor 0.5; placebo pass share at most 10%.

### word_covariates

33 measures between a remake r and a candidate c. Tokens: lower case, two or more word characters (scikit-learn default). Stop words: scikit-learn's English list. df = number of films read that contain the term; N = films read. "Optional words" = non-stop words in at least 2 films and at most half of them. "TF-IDF cosine" = scikit-learn TfidfVectorizer fitted on the films read (smoothed idf, L2 norm, terms in at least 2 films), cosine between the two films.

The paper's decision set, carried over:
1 tfidf_cosine: TF-IDF cosine, stop words removed, log-scaled counts (the paper's setting).
2 word_presence: share of c's optional words that are also in r.
3-7 word_presence_band1..5: the same within five df bands, df/N in [2 films, 0.1%), [0.1%, 0.33%), [0.33%, 1.6%), [1.6%, 6.4%), [6.4%, 50%].
8 tfidf_raw_counts: as 1 on raw counts.

Added before any calibration run:
9 shared_word_count: number of optional words in both.
10 candidate_log_words: ln of c's number of optional words.
11 tfidf_char4: TF-IDF cosine over character 4-grams inside word boundaries, log counts.
12 tfidf_words_and_pairs: as 1 over words and adjacent word pairs.
13 tfidf_stop_words_kept: as 1 with stop words kept.
14 counts_no_idf: cosine of raw counts, no idf.
15 bm25: Okapi BM25, k1 = 1.5, b = 0.75, idf = ln(1 + (N - df + 0.5)/(df + 0.5)); query = r's distinct non-stop words, document = c.
16 tfidf_crude_stems: as 1 after stripping one ending (ing, edly, ed, es, ly, s) from words with at least 3 characters before it.
17-18 lsa_100, lsa_300: cosine in the first 100 / 300 dimensions of a truncated SVD (randomised, 7 iterations, seed 0) of matrix 1.
19 tfidf_no_capitals: as 1 after deleting capitalised words that do not start a sentence.

Added because they or close relatives passed in the units' place against the 19 above:
20 words_to_triples_tfidf_stop_kept: TF-IDF cosine over word 1-, 2- and 3-grams, stop words kept, log counts.
21 bm25_reversed: BM25 with c's words as the query and r as the document.
22 jaccard_rare_words_under_1pct: Jaccard over words with df/N below 1%.
23 jaccard_all_words: Jaccard over optional words.
24-25 char3_tfidf, char5_tfidf: as 11 with 3- and 5-grams.
26 tf_sublinear_no_idf: cosine of log counts, no idf.
27 tfidf_binary_tf: TF-IDF cosine on term presence.
28 lsa_50: as 17 with 50 dimensions.
29 shared_5_word_sequences: number of distinct 5-word sequences (stop words kept, in at least 2 films) in both.

Added because within-sentence word pairs passed against the 29, and Dice against the 32:
30 sentence_pair_presence: share of c's within-sentence word pairs (unordered pairs of distinct non-stop words in one sentence, pairs found in at least 2 films) that are also in r.
31 sentence_alignment_forward: for each sentence of r its best TF-IDF cosine with any sentence of c (word weights fitted on whole films), averaged over r's sentences.
32 sentence_alignment_reverse: the same with r and c exchanged.
33 dice_all_words: Dice coefficient over optional words.

Exact code: wordlib.covariates(). Hub-corrected copies of the covariates were tried and dropped: they did not improve the word model (left-out log-likelihood -52.7 against -50.2) and hub-corrected stand-ins do not pass without them.

### statistic_and_null

Statistic: the documented source's residual score minus what the word model expects, averaged over families. The residual is the paper's: the unit score with the word measures regressed out, as a normal score within the pool. The expectation is taken over the pool with the word-only probabilities of being the source.

Null: the paper relabelled sources uniformly. Here each remake's source is redrawn in proportion to a word-only probability (conditional logit on the 33 measures, fitted without that remake's family). Under H0 the documented source is one such draw, so T is compared with 20,000 redrawn T*. A remake whose source the words identify is redrawn as itself and adds nothing; a remake the words leave open is compared with the candidates the words cannot tell from its source. Nothing is extrapolated and no pair is dropped: each pair's weight is 1 minus the word probability of its source.

Two p values, the larger is used. p_relabel is exact if the word probabilities are right, but too small if they are too sharp. p_spread (sum of family deviations over the root of their sum of squares, normal reference) takes its variance from the data, and cannot be passed by one or two families alone. Clustering: one uniform per family in every draw, deviations averaged within family, all fits leave a family out.

Why this form: T is the score statistic for adding the unit residual to the word model, so it is the locally most powerful statistic for a constant boost; and its mean is zero if either the word probabilities or the residual fit is right.

### decision_rule

"The units carry information about which film is the source beyond these word measures" is claimed (Qwen3-32B, primary unit setting) only if all six hold:
1. Evaluable: at least 10 open families.
2. Calibrated on the final list: at most 19 of the 190 word-only placebo runs (10%) have p below 0.05.
3. p = max(p_relabel, p_spread) below 0.05, one-sided.
4. The units' z_relabel is above the bar B, the largest z_relabel of the 190 placebo runs (1.81 on the recalled list).
5. p below 0.05 in at least half of the plan's sensitivity settings of the unit score (the bar applies at the primary setting only).
6. Positive control: against covariates that saw only the first half of every candidate, the same unit score has p below 0.05.

If 1 or 2 fails: "not evaluable" or "not calibrated", no statement either way. If 6 fails: "not interpretable", whether 3 passed or not. Otherwise, if 3, 4 or 5 fails: "not supported", reported with the estimated boost, its 95% interval and the smallest boost the calibration could detect (0.65 to 0.70 pool SD when words explain 40% of the unit score). A result with 3 passed and 4 failed is worded "no more than a word-only measure achieves". No constant, covariate, placebo or rule is changed after a unit array is opened.

### calibration_false_positives

Data: 98 recalled pairs (the firm and Indian lists together, both plots at 15+ sentences, non-American source, Kushi left out), 88 families, pools of 40 to 1,249 films (median 376), 42,542 cells, 6,230 films read. No model involved.

Word model with the 33 covariates: source first for 90 of 98, in the top five for 93; expected number first 86.2, so calibrated in aggregate; left-out log-likelihood -50.2 (seven measures -81.7, nineteen -54.9). 22 remakes in 21 families are open (source probability below 0.9); 15 below 0.5, 4 below 0.01. Sum of (1 - probability) = 14.0.

Word-only measures in the units' place, each as it is and hub-corrected, final test:
- Batch 1, 30 named measures seen while the covariates were chosen: 0 of 60 runs pass.
- Batch 2, 19 named measures written after the covariates were fixed and run once: 2 of 38 pass, both hub-corrected word-sequence measures (word triples only p = 0.035; stemmed words and pairs with stop words p = 0.045).
- Six curved functions of covariates (square, root, exp, cube, log, product): 0 of 12.
- 20 random vocabulary halves and 20 random word weightings: 1 of 80 (p = 0.035).
- Total 3 of 190 (1.6%); named 2 of 110. Largest z 1.81. Named p values: 7% below 0.10, 15% below 0.20.
By kind, p (as it is / hub-corrected): character 6-grams 0.65/0.67, 7-grams 0.73/0.57, 2-5-grams 0.40/0.09; word pairs only 0.47/0.45; 4-word sequences 0.67/0.53; BM25 k1 0.9 0.76/0.77, k1 2 0.085/0.68, on stems 0.14/0.19; raw-count TF-IDF with stop words 0.84/0.62; presence in other bands 0.24 to 0.89; rare-band Jaccard 0.80/0.78; LSA 30, 200, 250 dimensions 0.19 to 0.88; Hellinger 0.70/0.55; capitalised words 0.99/0.92.
How often: on 500 random three-quarter subsets of the families the mean pass rate over the 190 runs is 1.6%; the three that pass on the full list pass in 32 to 42% of subsets, every other run in 16% or less.

Noisier copies: a random mixture of the 33 covariates plus noise (words explaining 20 to 95%) passes 2 to 3% of 200 draws; one word measure plus noise 0 to 2.5%; pure noise 4%.

What made the test strict enough (same test, fewer covariates): against the paper's seven, 94 of 158 named runs pass (59%, z up to 5.9, 7 of 12 curved); against nineteen, 19 of 138 (14%); against 33, 2 of 110. The covariates grew 19 to 29 to 32 to 33 as listed in the covariate field.

The paper's own form for comparison (one linear fit on non-source cells, uniform relabelling): 62 of 110 named runs pass with seven measures, and still 42 of 110 with all 33 (8 of 12 curved).

Sensitivity: with the word probabilities flattened (square root) 19 of 110 named runs pass and z reaches 5.6, so the test depends on the word model not being too flat; rule 2 and the bar are there to catch that on the final list. With them sharpened (squared) the relabelling p alone passes 8 runs, the max rule 3. With the ridge fixed at 1, 3, 10, 30 the named passes are 5, 2, 0, 0.

Not independent: the 190 runs share texts and pairs, so "3 of 190" is a count, not a binomial sample.

### calibration_power

Planted signal: synthetic unit score = a word measure standardised within the pool + independent normal noise + a boost on the true source's cell, the boost in SDs of the score across the pool. 200 draws per point, 98 pairs, three different word measures as the base.

Smallest boost detected with 80% power at p below 0.05:
- words explain 40% of the score (Batman's unit rates were 35 to 49% explained): 0.60 to 0.65 SD; 0.66 to 0.70 SD when the score must also beat the bar (z above 1.81).
- words explain 80%: 0.41 to 0.45 SD; 0.43 to 0.47 with the bar.
Curve for the first base at 40%: boost 0.25 passes 14%, 0.4 37%, 0.5 62%, 0.6 76%, 0.7 84%, 0.8 96%, 1.0 99%. For scale, the true source sits a median 7.9 pool SDs above its pool in TF-IDF.
No boost: 0 to 2.5% pass.

What the residual step buys: without it the 80% point moves from 0.65 to about 0.73 (words explain 40%) and from 0.45 to about 0.70 (80%).

Why the figure is not smaller: only the pairs the words leave open carry weight (14.0 pair-equivalents of 98). The scout's hoped-for 0.25 SD assumed all pairs count; they do not. On a null the 95% interval of the estimated boost is about +/- 0.6 SD.

Real extra information of the word kind: with covariates that saw only the first half of every candidate (source first for 82 of 98, 40 open families), four measures that saw only the second half pass in 8 of 8 runs, z 4.8 to 6.0, p at most 0.0003; four whole-text measures pass, z 3.0 to 6.5. Reversed, half-text measures against whole-text covariates: 0 of 8 pass (p 0.09 to 0.91).

End to end (results/example.json): a synthetic score with a 1 SD boost gives z 3.40, p 0.0017, estimated boost 0.96 [0.32, 1.61], beats the bar, claim supported; the same with no boost gives p 0.65, not supported.

### positive_control

Pasted pairs cannot be this test's positive control. On the 45 pasted pairs (39 families) every single word measure tried ranks the source first for 45 of 45, and the word model gives each source a probability of at least 0.997. No family is open, the test reports "not evaluable", and a planted boost of 2 SD passes 0 of 50 times there. Adding the 45 to the 98 changes nothing (weight 14.02 against 14.01; a 2 SD boost passes 50 of 50). They stay the positive control of the plain heredity ranking: the unit score must rank the source first there.

The residual test's positive control is the half-text control (rule 6): word covariates recomputed on the first half of every candidate's sentences, the unit score left as it is. It needs no extra GPU work. Word measures pass it with z 3.0 to 6.5. If the unit score fails it, the units did not show even what half a summary's words hold, and a null of the main test says nothing.

### weaknesses

1. Few pairs carry the result. Words settle about three quarters of the pairs; the weight is 14 pair-equivalents in 21 families. The test says nothing about the pairs words solve. A pass means the units help where words hesitate, not that they carry the story in general.
2. The open pairs are odd ones: loose remakes, thin summaries, and probably table errors. Four remakes have a source probability under 0.01 (word ranks 34, 118, 162 among them). A row holding the wrong film's plot can be found by nothing and only adds noise. Check those rows against the articles before any unit is seen.
3. "Beyond words" means beyond these 33 measures. Each new kind of word measure passed against a smaller set (59% against seven, 14% against nineteen, sentence pairs against 29). The two fresh-batch passes were both word-sequence measures: reworded pasted text is where the covariates are weakest. No embedding could be run offline; one should join the placebos, and if it passes, a unit pass means "beyond counts" only.
4. The test trusts a word model fitted on about 100 pairs. Too flat, and word-only measures pass (19 of 110). The placebo table guards by example, not by proof.
5. A pass does not say what kind of information. Any reading of story similarity whose errors are independent of the words' passes (20% in the self-test for a fourth reading no better than the other three). Synonyms or paraphrase would do it; plot structure is not shown. Nor does a pass show units find sources better than words.
6. Recognition is untouched: a unit may fire for the source because the model knows the film or the remake link.
7. Hidden relatives in the pools count against units and words alike (conservative).
8. The calibration is on the recalled list, with a crude rule for bracketed names and without American-source pairs. The final list may leave more or fewer pairs open; detectable boost scales roughly with 1/sqrt(open weight). The word-only steps must be rerun on it.
9. Power is against a constant boost unrelated to words. If the units' extra information sits in the pairs words solve, the test cannot see it. Normal scores cap large boosts.
10. p_spread uses a normal reference with about 21 informative families; the placebo runs are correlated.
11. Not done: a boosted-tree fit in place of the linear one (too slow on the shared machine, abandoned after one unfinished run).

Result patterns that would mislead:
- Units rank sources very well: that is heredity, which words pass too, not this test.
- p_relabel small and p_spread not, or z_spread collapsing without the most favourable family: one or two remakes.
- A pass barely above the bar while several placebos pass.
- A pass confined to pairs the model names, or to pairs with copied sequences.
- A null read as "units add nothing": it rules out boosts above about 0.65 SD in the open pairs only.

### what_i_changed_from_the_starting_point

The starting point (regress the unit score on word measures, test the source's residual by uniform relabelling) fails here even with 33 covariates: 42 of 110 word-only runs pass it, 8 of 12 curved functions among them. Four changes:
1. The null. Sources are relabelled in proportion to a word-only probability of being the source, not uniformly. This is what removes the extrapolation: a source that words identify has no weight, and one they leave open is compared only with candidates as word-plausible as itself.
2. The residual is kept but refitted: pooled over the other families' cells, on values standardised within each pool without its top 1%, with a bend at z = 2, and scored as a within-pool normal score. It is no longer what makes the test valid (without it 0 of 190 placebo runs pass too); it buys power (80% point 0.65 against 0.73 SD, and 0.45 against 0.70 when the score is mostly words).
3. The covariates went from 7 to 33. With seven, 59% of word-only runs pass this same test.
4. Two guards added to the decision: the larger of two p values, and the unit score must beat every one of 190 word-only placebo runs computed on the final list beforehand. Plus a positive control that fits this test (half of every candidate hidden from the words), since pasted pairs carry no weight in it.

### run_time

Word measures for 6,230 films and 146 remakes: about 6 minutes on one core (covariates 99 s, placebos 185 s, half-text control 61 s). Word model with the ridge grid and leave-one-family-out fits: 18.5 s. One test: 0.07 s. The 190 placebo runs with hub correction: about 1 minute. Whole calibration: about 35 minutes on a shared, loaded machine (false positives 11 min, power 12 min, sensitivity 3 min, the rest 4 min). Memory under 1.5 GB. The cached measures take 367 MB and can be deleted and rebuilt.

Run from /home/user/phylomemetics/study with ../.venv/bin/python, in this order: prep_data.py, build_measures.py, calibrate.py (sections can be named as arguments). remake_residual_test.py run alone does a 6-second self-test on made-up data. Nothing under the repository was written.

### code_path

/tmp/claude-0/-home-user-phylomemetics/a789a30b-498f-5fe1-bc57-f1de263eb0f6/scratchpad/second/design/residual/remake_residual_test.py

### calibration_script_path

/tmp/claude-0/-home-user-phylomemetics/a789a30b-498f-5fe1-bc57-f1de263eb0f6/scratchpad/second/design/residual/calibrate.py (word measures: /tmp/claude-0/-home-user-phylomemetics/a789a30b-498f-5fe1-bc57-f1de263eb0f6/scratchpad/second/design/residual/wordlib.py; data preparation: /tmp/claude-0/-home-user-phylomemetics/a789a30b-498f-5fe1-bc57-f1de263eb0f6/scratchpad/second/design/residual/prep_data.py and /tmp/claude-0/-home-user-phylomemetics/a789a30b-498f-5fe1-bc57-f1de263eb0f6/scratchpad/second/design/residual/build_measures.py; outputs: /tmp/claude-0/-home-user-phylomemetics/a789a30b-498f-5fe1-bc57-f1de263eb0f6/scratchpad/second/design/residual/results/)