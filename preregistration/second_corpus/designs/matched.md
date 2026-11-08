# Design report: matched

One of three designs for the remakes test of whether units add to word overlap (study/v2/second/PLAN.md, Section 9). Written by an AI agent working apart from the other two, on word data only, before any model had read a remake text. It was calibrated on lists the scouts had recalled, with the narrower candidate frame; the plan re-runs the calibration on the final list. Paths under /tmp are the agent's working folder and are not in the repository.

### name

Matched hard-decoy test: the true source against the decoys that a fitted word model still prefers to it

### specification

PRE-REGISTRATION TEXT (every constant fixed before any unit measure exists)

Question. Among the candidate sources that the word measures like at least as much as the true source, do the units still prefer the true source?

Cells (the frame, unchanged). For each remake r of the primary list: its pool C(r), its true source s(r) in C(r), its family f(r). For every cell (r, c), c in C(r): the unit containment rate u_raw(r, c) and its hub-corrected z, u_z(r, c).

Step 1, word covariates. 14 word measures m(r, c), each computed from the two texts the model reads, each in two forms (as it is, and hub-corrected like the unit measure): 28 covariates x_1..x_28. Exact definitions are under "Word covariates".

Step 2, word score. A conditional logit is fitted on the true sources of the primary list: within pool C(r) the source is candidate c with probability exp(b.x(r,c)) / sum over C(r) of exp(b.x(r,c')). Each covariate is first standardised to mean 0 and sd 1 over all cells of all pools. b maximises the log likelihood minus 0.5 * 1.0 * |b|^2 (ridge 1.0; damped Newton to convergence). The word score is eta(r, c) = b.x(r, c). It is fitted knowing the true sources, on purpose: the word model is given every advantage.

Step 3, hard decoys. D(r) = every c in C(r), c not s(r), with eta(r, c) >= eta(r, s(r)). A remake is informative if D(r) is not empty. All other remakes drop out: words already put their source first, so nothing can be learned from them.

Step 4, statistic. q_r = share of D(r) whose unit measure is above the source's (a tie counts one half). T = mean over informative families of the family's mean q_r. T is the share of word-preferred decoys that also beat the source in the units; 0.5 or more if the units add nothing.

Step 5, null and p. Under the null the source's place among its hard decoys is uniform: the number of decoys beating it is floor(U * (|D(r)| + 1)), U uniform on (0,1), ONE U per family (remakes of one source move together). T* is computed the same way over 200,000 draws (numpy default_rng(20261108), blocks of 20,000). p = (1 + number of T* <= T) / 200,001, one-sided.

Step 6, decision rule. See "Decision rule": primary test on u_z, the same on u_raw, at least 8 informative families, and a positive control.

Gate before any unit is seen (words only, 5 minutes). On the final pair list, count the informative families and put the 101 word-only measures of the calibration through the test in the units' place. If fewer than 8 families are informative, or 3 or more of the 101 pass the decision rule, the test is not used as registered; it is strengthened or the pools or list are enlarged, and the change is written down, before the units are computed.

Reported beside the decision: the informative remakes by name with |D(r)| and q_r; T and p for both forms; the largest p when one informative family is left out; the positive control; the share of 5-word sequences each informative remake copies from its source; the same test for TF-IDF cosine and BM25 in the units' place; and, if a sentence embedding is available, the embedding in the units' place.

Constants. Text: first 40 sentences, pool floor 15 sentences (frame). Vocabulary and document frequencies: every eligible film of the table (16,683 in the calibration; one row per page, one per identical plot). Frequency bands 2-15, 16-127, 128-1,023, 1,024 and over. BM25 k1 = 1.2, b = 0.75. LSA 300 dimensions, random_state 0. Ridge 1.0. Draws 200,000, seed 20261108. Alpha 0.05, one-sided. Minimum informative families 8. Positive-control covariates: word presence in the 1,024-and-over band, both forms. Software: Python 3.12, numpy 2.5, scipy 1.18, scikit-learn 1.9.1 (its tokeniser, English stop list and TruncatedSVD).

On the recalled list (102 usable pairs of the firm-105 and indian-78 lists, 92 families, pools of 27 to 9,866 films, median 359) the word score puts the source first for 92 remakes; 10 remakes in 10 families are informative, with 153, 83, 22, 14, 5, 4, 3, 2, 2 and 1 hard decoys; the test passes there if T <= 0.31.

### word_covariates

Texts: exactly the text the model reads (first 40 sentences joined by a space). Words: scikit-learn's default tokens (lower-cased runs of 2 or more word characters). Unless said otherwise its English stop list is removed and a word, pair or n-gram counts only if it occurs in at least 2 eligible films. N = number of eligible films of the table, df = number of films a word is in, idf = ln((1 + N) / (1 + df)) + 1.

The paper's seven:
1. tfidf: cosine of TF-IDF vectors, tf = 1 + ln(count).
2. tfidf_raw: the same on raw counts.
3. pres: share of the candidate's distinct words that are also in the remake (after stop-word removal no word is in over half the films, so every word is "optional").
4-7. pres_2_15, pres_16_127, pres_128_1023, pres_1024_up: the same share within a band of df.
Five stronger single measures (each ranks the source first more often than TF-IDF does: 81% for 8 and 9 against 68%):
8. tfidf_binary: cosine of presence x idf.
9. bm25: sum over the remake's distinct words w of idf_w * tf_cw * 2.2 / (tf_cw + 1.2 * (0.25 + 0.75 * len_c / mean len)).
10. tfidf_words_and_pairs: TF-IDF (1 + ln) cosine over single words and pairs of neighbouring words.
11. tfidf_char4: TF-IDF (1 + ln) cosine over 4-character pieces inside words (char_wb).
12. lsa_300: cosine on the first 300 singular vectors of measure 1's TF-IDF matrix over the eligible films (TruncatedSVD, random_state 0). Topics from word counts alone.
Two for reused phrasing:
13. tfidf_pairs_stop_words_kept: TF-IDF (1 + ln) cosine over pairs of neighbouring words, stop words kept.
14. shared_3_word_sequences: number of distinct 3-word sequences (stop words kept) in both texts.

Two forms of each, 28 covariates:
- raw: m(r, c).
- hub-corrected: z(r, c) = (m(r, c) - mean) / max(sd, floor), mean and sd of m(x, c) over the other remakes x of the primary list, leaving out any x of c's family. floor = the median positive sd of that measure over the eligible films. The floor is needed for words only: a measure built on few words can be constant for a film, and the paper's sd + 1e-9 then sends z to 1e9 (this broke my first composite).

Both forms are needed. With hub-corrected covariates alone, raw-form word measures pass in the units' place (overlap counts, BM25 and LSA at p = 0.0002 against the paper's seven in z form), because a hub correction by itself reorders the top candidates.

The code that computes them is matched_test.word_measure_matrices() and cut_to_pools(); it reproduces the calibration's cached matrices exactly (largest difference 0.0 on all 14).

### statistic_and_null

Statistic. For each informative remake, q_r = (number of hard decoys with unit measure above the source's + half the ties) / |D(r)|. T = mean over informative families of the family's mean q_r. It is the paper's "share of candidates that beat the documented source", taken over the word-preferred decoys only. Each family has equal weight and a bounded contribution, so no single remake with many decoys can carry the result alone.

Null. H0: within a pool the unit measure is some increasing function of the word score plus noise that is exchangeable across candidates (this includes "a noisier copy of the words" and "pure noise"). Under H0 each hard decoy beats the source with probability at least 1/2, and the number that do is, draw by draw, at least what an exchangeable ranking gives. The reference distribution is therefore the exchangeable one: the source's place among D(r) and itself is uniform, b_r = floor(U * (|D(r)| + 1)). One U per family: remakes that share a source get the same draw, which is the most conservative way to treat their dependence. p = (1 + #{T* <= T}) / 200,001 over 200,000 draws, one-sided. The test is exact for pure noise (5.15% of 2,000 simulations pass at 0.05) and conservative for anything word-like: the 101 word-only measures give a median T of 0.61, not 0.50.

Why this answers regression to the mean. A decoy that tops ONE word measure is partly a noise pick, and any second measure prefers the true source to it. Measured: with hard decoys chosen by TF-IDF alone, 41 of 101 word-only measures pass in the units' place, and a noisy copy of Jaccard with no extra signal passes in 97 to 100% of simulations. A decoy that stays above the true source in a score fitted on 28 covariates with the answers known is not a noise pick of one measure: 0 of 101 pass. What remains is stated under weaknesses: a second, genuinely independent reading of the story would still pass, and should.

### decision_rule

"The units carry information about which film is the source beyond the word measures" is claimed only if all four hold:
1. Primary test: unit measure = hub-corrected z, 28 covariates, ridge 1.0: p < 0.05 (one-sided).
2. The same test on the raw unit rate, with the same hard decoys: p < 0.05.
3. At least 8 families have a hard decoy. Otherwise: "not evaluable", not a null.
4. Positive control: against a word score built from common words alone (presence of words found in 1,024 or more films, both forms), the hub-corrected unit measure has p < 0.05 with at least 8 informative families. If it fails, the unit measure cannot beat even a weak word model, and the test is reported as not evaluable whatever 1 shows.

If 1 to 4 hold but the largest p with one informative family left out is 0.05 or more, the claim is worded as resting on a single remake, which is named.
If 3 and 4 hold and 1 or 2 fails: "nothing detected in the units beyond these word measures", with the detectable size quoted beside it (a boost of about 1.3 pool sd present in most hard pairs; see power).
Nothing is claimed from T < 0.5 with p >= 0.05, and T near 0.6 is what a word measure itself gives, not evidence that units are worse than words.
Code: matched_test.decision(remakes, word_measures, unit_z, unit_raw).

### calibration_false_positives

Run on the real texts, recalled list, 102 pairs. "Passes" = the decision rule's clauses 1 and 2 (both forms below 0.05). Each word-only measure was put in the units' place in both forms. None is among the 28 covariates.

Frozen test:
- Development set, 66 measures (seen while I designed the test): 0 pass. Lowest p 0.078 (raw-count character 5-grams, z form); lowest raw-form p 0.35.
- Fresh set, 35 measures made after the design was frozen and run once: 0 pass. Lowest p 0.24 (z) and 0.14 (raw).
- By kind, lowest p of either form: character n-grams (4 measures) 0.078; word pairs, triples and sequences (7) 0.39; BM25 variants (3) 0.52; TF-IDF without idf (2) 0.76; word presence in other bands (9) 0.39; a random half of the vocabulary (15 measures, 9 splits) 0.14; random re-weightings of the vocabulary (15) 0.43; Jaccard of rare words 0.57; TF-IDF with stop words kept 0.83; LSA and NMF topics (5) 0.44; names (3) 0.62; order of the plot (words tagged by third) 0.70; best-matching sentences 0.78; crude stems 0.43; pseudo-units made from words and put through the unit rule (word classes; topic poles present where a sentence reaches 30% or 50% of the pole's maximum) 0.24.
- Families resampled 200 times, word score refitted each time: mean pass rate 0.36%; 77 of 101 never pass; the worst two are TF-IDF on a random half of the vocabulary (10% and 6%); every other measure 3.5% or less.
- Pure noise in the units' place: 5.15% of 2,000 (nominal). The 14 covariates themselves: lowest p 0.22. Hub z without the sd floor: 0 of 101.

Easier tests, for contrast (what I changed away from):
- Hard decoys chosen by TF-IDF alone (the starting point taken literally; 33 informative remakes): 26 of 66 and 15 of 35 pass (41%).
- Decoys that beat the source on every one of the paper's seven measures in both forms: 1 of 146 forms below 0.05, but only 5 informative remakes.
- Word score fitted with the remake's family left out: 14 to 15 informative remakes, 5 to 8 of about 130 forms below 0.05 (4 to 6%).
- Likelihood-ratio test in a conditional logit with the same 28 covariates: 2 of 66 and 3 of 35 pass both forms (5 of 35 in raw form). With a cross-fitted word score as its only covariate: 57 to 63 of 130 to 153 forms (over 40%).

A half of the vocabulary held out FROM the covariates (covariates see one random half, the "unit" is TF-IDF on the other): passes in 6 of 6 splits (T 0.06 to 0.17, p = 0.00005). Against the full covariates the same six measures give p 0.42 to 0.97. So new word evidence is detected when the covariates lack it, and is not when they hold all the words.

### calibration_power

Planted signal. Synthetic unit = rho x (a word measure that is not a covariate, standardised per remake) + sqrt(1 - rho^2) x independent noise + a boost for the true source, built over the whole remake-by-film matrix and hub-corrected like the units. Boost in sd of the measure across a pool. 200 simulations per cell, full decision rule (both forms, positive control), 102 pairs.

Base Jaccard, share passing:
| boost | rho 0.5 | rho 0.8 | rho 0.95 |
| 0 | 0% | 0% | 0% |
| 0.5 | 9.5% | 0.5% | 0% |
| 1.0 | 45.5% | 38% | 21.5% |
| 1.5 | 88% | 94.5% | 99% |
| 2.0 | 99% | 100% | 100% |

Smallest boost detected with 80% power: 1.38, 1.33 and 1.33 pool sd (Jaccard, rho 0.5, 0.8, 0.95); 1.25, 1.10 and 0.86 with term counts without idf as the base. For scale, a true source sits a median of 5.3 to 6.9 pool sd above its pool on a single word measure, so the detectable extra signal is about a fifth of the word signal.

That figure assumes the boost is in every pair, including the 10 hard ones. If only a random half of the pairs carry it (rho 0.8), power is 21.5% at 2 sd, 34% at 3 sd and 35.5% at 4 sd: it does not rise further.

A noisier copy of the word covariates does not pass. Copies of each of the 14 covariates at rho 0.5, 0.8 and 0.95 with no boost (100 simulations each): 0% in 35 of 42 cells, 1 to 2% in four cells at rho 0.5, and 5% for the rarest-word band at every rho (among hard decoys it behaves like noise, so it sits at the nominal rate). Copies of the fitted word score: 3.5% at rho 0.3, 1% at 0.5, 0% at 0.8 and 0.95. Mean T for these copies is 0.51 to 0.79.

Two changes outside the frame would roughly double the informative remakes; I report them and did not adopt them:
- Pools of every eligible film of the table (median 11,281 instead of 359): 23 informative remakes, 0 of 101 word measures pass, power at a 1.0 sd boost 97% instead of 38%.
- Adding the 22 usable loosely documented pairs: 21 informative remakes (11 of them loose), 0 of 101 pass, power at 1.0 sd 95%.

### positive_control

Pasted-summary pairs (44 usable, 38 families), scored by the word model fitted on the main list: the source is first by the word score in all 44, so no remake has a hard decoy and the test is not evaluable on them. They cannot serve as a positive control for this test; they can only check the unit measure itself (does it rank the pasted source first). Adding them to the fit leaves the 10 informative remakes unchanged.

Registered positive control instead: the same test against a deliberately weak word score (common words only). On the recalled list it has 65 informative remakes in 63 families; 108 of 114 word measures pass it in the units' place, and the six that fail are measures with almost no trail (the same band, the two rarest-word bands at p 0.052 and 0.056, 5-word sequences, length similarity). Every synthetic unit in the power simulations passed it (100%). It shows only that the unit measure can beat a weak word model; it is a floor, not a validation.

### weaknesses

1. About 10 remakes carry the whole test. Words put the source first for 92 of 102, and the other 92 say nothing. Resampling families gives 6 to 10 informative families (quartiles) and fewer than 8 in 42% of resamples, so "not evaluable" on the final list is a real possibility. It is known from words alone, before any model runs.

2. Those 10 are not typical pairs. They are the loosest remakes with independently written summaries; 3 are Hollywood to Bollywood with pools near 9,000, and one (Samrat) is also on the loose list. A table row holding the wrong plot would land here and count against the units.

3. The test is conservative, so a null is weak. A word-like measure gives T about 0.6 and a pass needs 0.31. A contribution under about 1.3 pool sd, or one present in only half the hard pairs (power 36% at most), is missed.

4. A pass means "beyond these 14 word measures", not beyond words. A held-out half of the vocabulary passes when the covariates do not see it. Word-derived information of a kind I did not cover (a synonym dictionary, embeddings trained elsewhere) could pass. LSA is the only control for topic. A sentence embedding should go through the same test beside the units.

5. Names stay in the text. If the model recognises the film or knows what it remakes, a pass shows memory, not story. The test cannot tell these apart, and 10 remakes cannot be split by recognition.

6. Reworded pasting. Word pairs and 3-word sequences catch reused phrasing; they do not catch a pasted summary reworded sentence by sentence.

7. The word score is fitted on the list itself. Any change to the list, the covariates or the ridge changes which remakes are informative (28 covariates: 10; five strong measures: 12; family left out: 14 to 15). All must be frozen with the list.

8. The validity argument covers unit measures that rise with the word score. A measure that falls with it among top candidates could pass with nothing in it. I checked this only empirically, on 101 word measures in two forms.

9. Two remakes hold 236 of the 289 hard decoys. Families are weighted equally, but the drop-one-family p must be read.

10. All models are tested on the same 10 remakes; their results are not independent.

11. Calibrated on a recalled list and on my rebuild of the frame. My pools are smaller than the frame says (median 359, not several hundred to 2,500), and I stripped bracketed names by my own rule. The gate must be re-run on the final list.

Patterns that would mislead:
- p >= 0.05 read as "units add nothing".
- T between 0.31 and 0.5 read as a trend.
- T about 0.6 read as "units are worse than words".
- A pass in one form of the unit measure only, or one that disappears when a single family is dropped.
- A pass read as meaning when the model names the films.
- "Not evaluable" reported as a null.
- Passes in several models counted as separate confirmations.

### what_i_changed_from_the_starting_point

I kept the idea (nonparametric, rank-based, hard negatives only) and changed how hard negatives are chosen and what the null is.

1. Hard negatives by one word measure fail, as predicted. With decoys that TF-IDF likes at least as much as the source, 41 of 101 word-only measures pass in the units' place and a noisy copy of another word measure passes 97 to 100% of the time.
2. Requiring a decoy to beat the source on every word measure removes that, but leaves 5 informative remakes of 102: for 95% of remakes at least one of seven word measures ranks the source first.
3. Final rule: decoys that a conditional logit on 28 word covariates, fitted with the true sources known, still scores at least as high as the source. 10 informative remakes, 0 of 101 word measures pass.
4. I fit in-sample, not with the family left out. Leaving the family out gives 14 to 15 informative remakes but lets 4 to 6% of word-measure forms through; it leaves the word score slightly worse for that pool, and other word measures fill the gap.
5. Covariates strengthened from the paper's seven to 14 measures in two forms: added binary TF-IDF, BM25, words-and-pairs, character 4-grams, LSA-300, word pairs with stop words, shared 3-word sequences, and the hub-corrected form of each. A floor on the sd of the hub correction for word measures.
6. A parametric test was tried and dropped: a conditional-logit likelihood ratio lets 3 of 35 fresh word measures through with the same covariates, and over 40% with a cross-fitted score.
7. Decision rule: both forms of the unit measure, at least 8 informative families, and a positive control against a weak word model, because the pasted-summary pairs have no hard decoy and cannot test this test.

The finding that matters most for the plan: on this corpus and frame a calibrated "beyond words" test has about 10 informative remakes, close to Batman's 9. Larger pools or the loose pairs would give 21 to 23.

### code_path

/tmp/claude-0/-home-user-phylomemetics/a789a30b-498f-5fe1-bc57-f1de263eb0f6/scratchpad/second/design/matched/matched_test.py

### calibration_script_path

/tmp/claude-0/-home-user-phylomemetics/a789a30b-498f-5fe1-bc57-f1de263eb0f6/scratchpad/second/design/matched/calibrate.py

### run_time

The test: 2 seconds for the whole decision rule (word-score fit, both forms of the unit measure, positive control, 200,000 null draws, drop-one-family) on 102 remakes and 84,051 cells. `python matched_test.py` runs a self-test on made-up numbers in under a second.

The 14 word measures over 16,683 films: about 3 minutes (6 when the machine was busy with other jobs), under 2 GB.

Calibration (/tmp/claude-0/-home-user-phylomemetics/a789a30b-498f-5fe1-bc57-f1de263eb0f6/scratchpad/second/design/matched/calibrate.py, sections run one at a time on 4 shared cores): frozen 5 s, false positives 5 min, bootstrap 10 min, power 10 to 20 min, noisier copies 10 min, controls 30 s, larger pools 70 s, loose pairs 25 s, check 3 to 6 min. Results: /tmp/claude-0/-home-user-phylomemetics/a789a30b-498f-5fe1-bc57-f1de263eb0f6/scratchpad/second/design/matched/calibration_results.json, with the logs (log_*.txt) beside it.

Data for the calibration take about 11 minutes and 1.3 GB to build, in this order, from /home/user/phylomemetics/study with ../.venv/bin/python: build_corpus.py (10 s), build_measures.py (4 min), build_fresh.py (6 min), all in the same folder. To free disk I deleted the 101 non-covariate matrices afterwards (the disk was at 90%); the 14 covariate matrices remain, so the "frozen" section runs as is and the other sections need the two builders run again first.

Nothing under /home/user/phylomemetics was written. The explore*.py and core.py files in the folder are my working scripts, kept for the numbers quoted from them; they are not part of the deliverable.