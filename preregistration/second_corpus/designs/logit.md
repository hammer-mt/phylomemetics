# Design report: logit

One of three designs for the remakes test of whether units add to word overlap (study/v2/second/PLAN.md, Section 9). Written by an AI agent working apart from the other two, on word data only, before any model had read a remake text. It was calibrated on lists the scouts had recalled, with the narrower candidate frame; the plan re-runs the calibration on the final list. Paths under /tmp are the agent's working folder and are not in the repository.

### name

Held-out conditional-logit slope test against word-plausible candidates, with a registered panel of word-only stand-ins

### specification

PRE-REGISTRATION TEXT (every constant fixed; nothing is chosen after the units are seen)

1. Cells. As the frame fixes them: for each remake r, the pool C(r) is every film of the source's industry with at least 15 sentences, released in or before the remake's year, minus the remake's own family; the documented source s(r) is one member. A cell is a pair (r, c) with c in C(r). A family is a connected component of the documented remake links (remakes of one source, and chains). The family is the unit of clustering everywhere below. "Films read" F = the union of all pools, remakes and sources. Texts are the original plot texts, first 40 sentences (study/fetch_corpus.py sentences()), joined by one space.

2. Word measures. 14 word measures and 2 length measures (exact definitions under "word covariates") are computed between every remake and every film read (a matrix n_remakes x |F| per measure), from the texts alone.

3. Word columns (31). For each of the 14 word measures m, two columns over the cells:
   - "z": m standardised within the pool (mean 0, SD 1 over C(r));
   - "hz": the hub-corrected z of m, then standardised within the pool. Hub correction of cell (r, c): (m(r,c) - mean) / SD, the mean and SD taken over m(r', c) for every remake r' that is not in c's family and is not r.
   Plus the two length measures in "z" form (30 columns so far). Column 31, the consensus cube: the mean of the 28 word-measure columns, standardised within the pool, cubed, standardised within the pool again.

4. Unit column (1). The containment rate u(r, c) for every remake and every film read is hub-corrected by the same rule as in 3, and within each pool is replaced by the normal score of its rank, Phi^-1((rank - 0.5) / |C(r)|), ties sharing their mean rank. Only the order of the unit measure inside a pool is used.

5. Word model. Conditional logit over each pool, P(c is the source of r) proportional to exp(beta . x(r, c)), no intercept, ridge penalty (lambda/2)|beta|^2 with lambda = 3 on all 31 coefficients, fitted by Newton steps to convergence (Newton decrement below 1e-9). It is fitted once per family f on all remakes outside f, giving beta_f and probabilities p_f(c | r) for every cell.

6. Statistic. For each family f:
   A_f = (sum over remakes outside f and their cells of p_f xt xt' + lambda I)^-1 (sum of p_f xt u), where xt = x minus its p_f-weighted mean over the pool. (The ridge regression of the unit column on the word columns among the other families' cells, each cell weighted by the word model's probability.)
   For each remake r in f: U_r = [u(r,s) - sum_c p_f(c|r) u(r,c)] - [x(r,s) - sum_c p_f(c|r) x(r,c)] . A_f.
   D_f = sum of U_r over the remakes of family f. T = sum_f D_f. Z = T / sqrt(sum_f D_f^2).
   T is the slope, at zero, of the held-out log-probability of the true sources in the weight of the unit column, the word weights free to adjust.

7. p value. One-sided, by flipping the signs of the family sums: p = (1 + number of draws with sum_f e_f D_f >= T) / (1 + 100,000), e_f = +1 or -1 with equal chance, seed 20261108. If at most 20 families have a non-zero sum, all sign patterns are enumerated.

8. Held-out gain (the starting design, kept as a condition). The model is refitted with 32 columns (31 word columns and the unit column), same penalty, leaving one family out; G = sum over remakes of [log P_32(s | r) - log P_31(s | r)], each remake scored with the weights fitted without its family.

9. Stand-ins. 78 word-only measures that are not among the covariates (list in the code, function standin_measures) are put through steps 4, 6 and 7 in the units' place on the final data.

10. Evaluable. E = sum_r (1 - P_31(s | r)), held out. The test is evaluable only if E >= 5.

11. Decision: see "decision rule".

12. Constants. lambda = 3; alpha = 0.05 one-sided; 100,000 sign flips, seed 20261108, exact enumeration up to 20 families; E >= 5; stand-in share limit 10%; frequency bands at 0.2%, 1%, 5% and 50% of the films read; BM25 k1 = 1.5, b = 0.75; skip window 3; LSA 150 dimensions, seed 2; texts of 40 sentences, floor 15; scikit-learn 1.9.1 as installed in the repository.

13. Order of work. (i) The final pair list, families and pools are frozen. (ii) The word measures, the word model, E, the 78 stand-in results and the planted-boost power on the final list (calibrate.py) are computed and committed, with a hash of the word-measure matrices, before any unit measure is joined. (iii) Then the unit column is read. The decision is taken on Qwen3-32B; the other models go through the same test and are reported, and a count of models that pass is not called that many confirmations.

14. Reported beside the decision, not part of it: lambda = 1 and 10; the unit rate with no hub correction; the table of family sums D_f; T, Z and p with each of the five largest |D_f| left out in turn; the result split by the degree of copied text and by whether the model names the films.

Input the GPU side must supply: the containment rate between every remake and every film read (not only its own pool), so that the hub correction is done by the same code for units, words and stand-ins. If the pipeline's own z is used instead, its reference films must be the remakes outside the candidate's family.

### word_covariates

Common rules. Words are scikit-learn's default tokens (lower case, two or more letters or digits) with its English stop list removed, kept if they occur in at least 2 of the films read. Document frequency (df) and idf are taken from the films read (N films). "Optional" words: df between 2 and N/2, the word twin of the optional units.

Fourteen word measures, each computed for every (remake, film read):
1. cos_tfidf: cosine of TF-IDF vectors with the logarithm of counts (TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)). The paper's first measure.
2. cos_rawtf: the same on raw counts. The paper's seventh.
3. cos_tf: cosine of the logarithm of counts with no idf (use_idf=False).
4. cos_binidf: cosine of idf-weighted presence vectors, counts ignored (binary=True).
5. bm25: Okapi BM25 of the candidate for the remake's distinct words, k1 = 1.5, b = 0.75, idf = ln(1 + (N - df + 0.5)/(df + 0.5)).
6. cos_bigram: TF-IDF cosine over pairs of adjacent words (ngram_range=(2,2), stop words removed first).
7. cos_skip3: TF-IDF cosine over ordered pairs of words at most 3 apart.
8. cos_char4: TF-IDF cosine over 4-character sequences inside words (analyzer="char_wb").
9. lsa150: cosine in a 150-dimension latent semantic space (TruncatedSVD, random_state=2, of the TF-IDF matrix of words and adjacent pairs). A word-only measure of meaning.
10. contain_opt: the share of the candidate's optional words that the remake also has. The paper's word presence.
11 to 14. contain_band1..4: the same within four df bands: [2, 0.2% of N), [0.2%, 1%), [1%, 5%), [5%, 50%]. The paper's four bands, redrawn for a corpus of thousands.

Two length measures: cand_log_words (log of the candidate's number of optional words) and len_match (minus the absolute difference of that log between candidate and remake).

Columns of the model (31): each of the 14 measures twice, standardised within the pool ("z") and as its hub-corrected z standardised within the pool ("hz"); the two length measures once; and the consensus cube (the mean of the 28 measure columns, standardised within the pool, cubed, standardised again).

What this set does on the calibration list (95 pairs, held out by family): TF-IDF cosine alone ranks the source first for 71 of 95; the paper's seven measures for 79; the 31 columns for 86. The nine sources it misses sit at ranks 2, 2, 3, 5, 8, 40, 63, 131 and 229. It gives the true source a probability above 0.99 in 57 pairs and below 0.5 in 11. Held-out log-likelihood: -108.9 (TF-IDF alone), -86.4 (the paper's seven), -52.0 (31 columns).

Why each addition was made (all on word data): measures 3 to 5 because a cosine without idf, BM25 and idf-weighted presence passed in the units' place against the paper's seven; 6 to 8 because pair and character measures did; 9 because a latent semantic measure reached p = 0.017; the hub-corrected form because the unit measure has one; the cube because convex functions of word overlap passed 83 to 100% of the time without it.

### statistic_and_null

Statistic. T = sum over families of D_f, where D_f sums U_r over the family's remakes and
U_r = e(r, s) - sum_c p(c | r) e(r, c), with e = u - x . A.
p(c | r) is the word model's probability for candidate c, fitted without r's family. A is the ridge regression of the unit column on the 31 word columns among the other families' cells, each weighted by p. In plain words: the unit rank of the true source, with what the word columns explain taken out, minus the same quantity averaged over the candidates the words find plausible. Equivalently U_r = sum over wrong candidates of p(c | r) [e(r,s) - e(r,c)]: the true source against each wrong candidate, weighted by how much the words like that wrong candidate. This is the test against "each remake's nearest candidates by words" that the judge asked for, with the weights supplied by the fitted word model and no extrapolation from the mass of dissimilar films.

T is exactly the derivative at zero of the held-out log-likelihood in the weight of the unit column. A positive T says: giving the units a small positive weight raises the held-out probability of the true sources.

What cannot speak. A remake whose source the words find with certainty has p(s | r) = 1 and U_r = 0, whatever the units say. On the calibration list 57 of 95 pairs are in that state, and the open mass, E = sum (1 - p(s | r)), is 12.4 "effective pairs".

Null. Given the word columns, the unit column carries no information about which candidate is the source: P(source = c | words, units) = P(source = c | words). Under it each family sum has mean zero. The p value further takes the family sums as independent and symmetric about zero, and is the share of sign patterns whose total is at least T (one-sided; 100,000 random patterns with a fixed seed, or all of them when at most 20 families are non-zero). Clustering: remakes that share a source are summed into one D_f and their family is left out of the fit together, so several remakes of one source count once.

Checks that this null behaves (word data only, 95 pairs): pure noise in the units' place is rejected 4.5% of the time; a covariate plus noise 1%; with noise that a film carries into every pool it sits in, 2.5%.

Companion statistic. G, the held-out log-likelihood gain of the model refitted with the unit column (the starting design). It is a condition (G > 0), not the test, because on its own it is too easy and two-sided: put through its own sign-flip test, word stand-ins with gains under one log unit pass (2 of 78), with a negative fitted weight.

### decision_rule

"The units carry information about which film is the source beyond these word measures" is claimed for Qwen3-32B only if all five hold:
1. Evaluable: the words leave at least 5 effective pairs open (E >= 5).
2. Slope: one-sided sign-flip p below 0.05.
3. Gain: the refitted model's held-out log-likelihood gain G is positive.
4. The units' Z is above the Z of every one of the 78 word-only stand-ins, computed on the same final data.
5. At most 10% of the stand-ins are below 0.05 on the final data.

Wording of the other outcomes, fixed now:
- Condition 1 fails: "not evaluable: words alone settle the sources". No claim either way.
- Condition 5 fails: "the word covariates do not span word measures on this list". No claim either way.
- 1 and 5 hold and any of 2 to 4 fails: "no information beyond these word measures was detected", quoted together with the smallest planted boost the test detects with 80% power on the final list (from calibrate.py re-run on that list before the units are joined).

A pass is worded "beyond these word measures", never "beyond words". The claim is not made from the gain alone, from the rank of the source alone, from a sensitivity setting, or from a count of models.

### calibration_false_positives

All on word data: 95 recalled pairs in 85 families, 3,924 films read, 39,760 cells.

How easy the test was before strengthening (stand-ins below 0.05, of 78, in the final test form):
- TF-IDF cosine as the only covariate: 65.
- The paper's seven measures: 54.
- The paper's seven in both forms plus lengths (16 columns): 23.
- 30 word columns without the cube: 4 (lowest p 0.006), and convex functions of word overlap (cube, square, exponential of the mean word column, plus noise) passed 83 to 100% of the time.
- Final design (31 columns, unit as rank): 0 of 78. Lowest p 0.069 (TF-IDF on 3-character sequences). 1 below 0.10, 15 above 0.90, so the test leans conservative for word-like measures.

The kinds asked for, final design, p of each:
- Character n-grams: 3-grams 0.069, 5-grams 0.50, 2-grams 0.88, 6-grams on raw counts 0.47, 4 to 5-grams on raw counts 0.48.
- Word pairs: words and pairs 0.91, pairs with stop words kept 0.60, pair Jaccard 0.29, triples 0.89, pairs within 6 words 0.48.
- BM25 with other settings: 0.96 (k1 2, b 1), 0.99 (k1 0.9, b 0.4, stop words kept), 0.59 (with query counts).
- Counts without idf, stop words kept: 0.77. Log counts without idf in each band: 0.58, 0.56, 0.62, 0.25.
- Word presence in other bands: 0.94, 0.38, 0.87; in tertiles: 0.53, 0.80, 0.72; weighted towards common words (df to the power 0.25, 0.5, 1, 2): 0.77, 0.75, 0.82, 0.74.
- A random half of the vocabulary (covariates see every word): presence 0.70 and 0.43, TF-IDF 0.47 and 0.69.
- Jaccard of rare words 0.88; Jaccard of optional words 0.88; Dice 0.88.
- TF-IDF with stop words kept 0.98.
- Latent semantic analysis: 0.86 (50 dimensions), 0.60 (100), 0.92 (300), 0.74 (60, with pairs).
- Copied sequences: 5 words 0.19, 3 words 0.25. Stemmed TF-IDF 0.50. Sentence alignment 0.81. Capitalised words 0.91.

Random families, drawn after the covariates and never used to choose them: 6 of 332 below 0.05 (1.8%), one below 0.01 (0.007). By generator: presence with random word weights 4 of 60; with a random tilt to common or rare words 0 of 60; TF-IDF on a random half of the words 0 of 60, of the word pairs 2 of 60, of the 4-character sequences 0 of 40, of each film's sentences 0 of 40; LSA of random size and seed 0 of 12.

Noisier and curved copies of the covariates: 1.7% of 13,450 draws below 0.05. These are the mean of the word columns, random positive and signed mixes, means of 5 columns, single columns, and the word model's own held-out score, each as it is and through eight curves (exponential, cube, signed square, signed root, tanh, normal score of rank, minus log rank), with noise 0, 0.5 and 1 SD. The worst cells are 7 to 8%. Single columns plus noise: 3.3% overall; one column stands out, the hub-corrected presence in the 1 to 5% band, at 11 of 75.

Other settings: penalty 0.3, 1, 3, 10, 30 give 0, 0, 0, 1 and 6 of 78 stand-ins below 0.05, so a much heavier penalty would make the test too easy; the unit rate with no hub correction gives 0 of 78.

A warning the calibration produced: a word score whose weights were fitted on these same pairs' labels passes up to 11% of the time. The unit measure must not be tuned on which film is the source.

Honest limit of (a): the fixed panel of 78 was partly used to choose the covariates (measures that passed were moved into the covariates and replaced). The random families and the copies were not.

### calibration_power

Planted signal on the 95 pairs: synthetic unit = a word-only base + independent noise, standardised within the pool, plus a boost for the true source in pool standard deviations of the whole measure; 200 draws per boost.

Smallest boost detected with 80% power (one-sided 0.05; the stand-in condition changes nothing, because the stand-ins' largest Z is 1.51):
- base = the word twin of the unit measure (a covariate), correlation 0.89 with it: 0.67 SD
- the same, correlation 0.71: 0.82 SD
- the same, plus noise each film carries into every pool: 0.89 SD
- base = a saturating function of the word consensus: 0.93 SD
- pure noise: 0.96 SD
- base = a word measure outside the covariates: 1.36 SD (sentence alignment), 1.47 to 1.55 SD (TF-IDF with stop words kept)
So: about 0.7 to 1.0 SD when the word-related part of the units is covered by the covariates or absent, about 1.5 SD when it resembles a word measure the covariates lack. For scale, the true source's TF-IDF cosine sits 7.0 pool SDs above its pool at the median, and the word twin's hub z 4.2.

A noisier copy does not pass: at boost 0 the rejection rate is 0 to 4.5% in every one of the eight settings above.

With the refit (24 draws per boost, twin plus noise): all three conditions (slope, gain > 0, above the stand-ins) hold in 0% of draws at boost 0, 42% at 0.5, 100% at 1.0, 1.5 and 2.0. Mean held-out gain 5.6 log units at 1.0 and 10.3 at 1.5. The gain condition never cost a pass.

A real independent witness. The covariates were given a random half of the vocabulary and the "units" were word measures on the withheld half (three splits, four measures): slope Z 4.0 to 5.3, p = 0.00001 in 12 of 12; held-out gain 20 to 68 log units; sources ranked first rise from 66, 79, 70 to as many as 79, 80, 82. The same withheld-half measures, when the covariates see every word: p 0.20 to 0.93. This is the only check here of what genuinely new information does, and it is the reading of "a random half of the vocabulary held out from the covariates" under which a pass is the correct answer.

What the power rests on. The boost was planted in every true source, but only the open pairs can show it: 11 families hold more than half a pair of open mass each, 20 more than a tenth. For a typical measure the statistic rests on about 4 families (median effective number 3.9, range 1.4 to 8.2) and the five largest carry 89% of its variance. The six hardest pairs are Baazigar to Samrat (rank 229), Karz to Enakkul Oruvan (131), Kandireega to Main Tera Hero (63), Suryavamsam to Sooryavansham (40), Manasantha Nuvve to Jeena Sirf Merre Liye (8), Nuvvu Naaku Nachav to Vaseegara (5).

Not run: a simulation at other numbers of pairs. Power follows the open mass E more than the count of pairs, so the power figures should be recomputed on the final list (one command, 4 minutes).

### positive_control

The pasted summaries cannot be a positive control for this test, and the calibration shows why. On the 44 usable pasted pairs (38 families, pools of 59 to 1,242) the word model ranks the source first in 44 of 44 with a held-out probability of 1 (log-likelihood -0.02). The open mass is 0.02 effective pairs, so the test is "not evaluable" by rule 1. Planted boosts are never detected there: 0 of 200 at every size up to 3 SD. No stand-in passes (0 of 78). Whatever the units do on pasted text, this test says nothing, which is the right answer: when words already settle the source there is nothing left to add.

What the pasted pairs are still good for: the heredity statistic (units should find these sources trivially) and a check on the pipeline.

The positive control for the test itself is the withheld-vocabulary check under "power": covariates on half the words, the other half in the units' place, p = 0.00001 in 12 of 12. It should be re-run on the final list before the units are joined.

Related, on the main list: the copy measures in the units' place do not pass (share of the remake's 5-word sequences found in the candidate p = 0.19; 3-word sequences p = 0.25). So leftover verbatim reuse alone does not pass on this list. Reworded copying that keeps sentence order is not a word measure and is not covered.

### weaknesses

1. It rests on about a dozen pairs. Words settle 57 of 95 sources outright; 11 families hold most of the open mass and about 4 carry a typical statistic. Some of those hard pairs are loose remakes or wrong table rows (Baazigar to Samrat and Karz to Enakkul Oruvan are the scouts' own doubtful cases). If the final list cleans them out, E falls and so does power; if E drops under 5 the test is not evaluable.

2. "Beyond these word measures", not "beyond words". Any measure with information the 14 measures lack passes, whether or not a model is involved: the withheld half of the vocabulary passes at p = 0.00001. The stand-in panel guards only against the 78 measures in it. Moving measures into the covariates each time one passed is how the set was built, and a 15th family could exist.

3. Tuned on the list it was calibrated on, and that list is recalled, not final. The covariates, the cube and the rank form were all chosen on these 95 pairs. Condition 5 and the re-run of calibrate.py on the final list are the protection; they are not proof.

4. Conservative for word-like units. Word stand-ins get mostly negative slopes (15 of 78 above p = 0.90), so a unit measure that behaves like a word measure outside the covariates needs a boost near 1.5 SD, not 0.8. A real but modest contribution can be missed.

5. The null assumes family sums that are independent and symmetric. Pools share candidates and the hub correction uses the other remakes, so families are not strictly independent. One form of shared noise was simulated (2.5% rejection); others were not.

6. One column is not fully covered: a noisy copy of hub-corrected presence in the 1 to 5% band passed 11 of 75 times.

7. The power figures are for one planted form (the same boost in every true source, Gaussian noise). A real effect confined to a few pairs, or absent in exactly the hard ones, would behave differently.

8. It says nothing about recognition or reworded copying. A model that knows the remake link, or that follows sentence order in a summary pasted and then reworded, would pass, and the pass would not be story inheritance.

9. Reproducibility of two measures. LSA uses a randomised SVD and depends on the library version; the word-measure matrices must be saved and hashed before the units are joined.

10. The calibration pools (27 to 1,242, median 350) are smaller than the frame's "several hundred to about 2,500", and 5 pairs with American sources and 4 the table dates backwards were left out by rule. Larger pools mean more hard negatives and probably more open mass.

Result patterns that would mislead a reader:
- A null read as "units equal words". It only excludes boosts of roughly 0.7 to 1.5 SD in a dozen pairs.
- A pass whose family sums are dominated by two or three families, or that disappears when one family is left out (both are reported for this reason).
- A pass on pairs the model names, or on pairs with leftover copied text, read as inheritance.
- The gain statistic quoted alone with its own p: word stand-ins pass that way with gains under one log unit.
- "Sources ranked first 86 to 88" quoted as the effect: there is room for nine at most.
- Several models passing counted as independent confirmations: they share the same dozen pairs.
- The heredity statistic at ceiling for units and words alike, read as agreement between them.

Housekeeping: nothing under the repository was written by me, no git command was run, and Python ran with bytecode writing off. Other processes were writing under study/v2/cache and changed requirements.txt while I worked.

### code_path

/tmp/claude-0/-home-user-phylomemetics/a789a30b-498f-5fe1-bc57-f1de263eb0f6/scratchpad/second/design/logit/remake_logit_test.py

### calibration_script_path

/tmp/claude-0/-home-user-phylomemetics/a789a30b-498f-5fe1-bc57-f1de263eb0f6/scratchpad/second/design/logit/calibrate.py

### run_time

Measured on 95 remakes, 3,924 films, 39,760 cells, with about one core free on a busy 4-core machine.

The test (remake_logit_test.py):
- word_measures: 20 s
- standin_measures (78 stand-ins): about 2 min
- RemakeTest (the word model, 85 leave-one-family-out fits): 2 s
- decide (unit column, the refit, 78 stand-ins, exact sums, 100,000 sign flips each): 13 to 17 s
End to end, about 3 minutes. Memory stays under 2 GB. Time grows roughly in proportion to the number of cells; at 100 remakes with pools of 2,500 expect about 6 times longer for the fits.

The calibration (calibrate.py all): about 25 minutes. By stage: corpus 6 s, measures 4 min, words 1 min, standins 15 s, random 2 min, copies 4 min, power 4 min, gain 6 min, split 3 min, copied 20 s, demo 15 s. Results are in the results folder beside it (JSON and logs): /tmp/claude-0/-home-user-phylomemetics/a789a30b-498f-5fe1-bc57-f1de263eb0f6/scratchpad/second/design/logit/results

Usage: measures = word_measures(texts, remake_rows); test = RemakeTest(cells, measures, film_family); panel = standin_measures(texts, remake_rows); report = decide(test, unit, panel). `unit` is the [n_remakes, n_films] matrix of containment rates, or with form="hub_z_per_pool" one hub-corrected array per remake over its pool. Earlier versions and exploration scripts are in the old folder beside the two files and are not needed.

### what_i_changed_from_the_starting_point

Kept: a conditional logit over each remake's pool, word measures as covariates, weights fitted on other families, a ridge penalty for the near-separation, clustering by source family.

Changed, each because a check on word data showed the starting form was wrong or weak:

1. The statistic is the slope of the held-out log-likelihood in the unit weight, not the difference between two fitted models. The difference, tested by sign flips over families, is passed by tiny consistent gains and by negative unit weights: with the paper's seven measures it passed 4 of 28 stand-ins, and in the final design it still passes 2 of 78 with gains under one log unit. A Wilcoxon test on the gains was worse (10 of 28). The change in the source's rank has no room (nine sources are not first). The slope is one-sided, needs no refit (so it could be calibrated on 14,000 draws), and is a direct comparison of the true source with the candidates words prefer. The fitted gain is kept as a condition (it must be positive).

2. The word covariates grew from the paper's 7 measures to 14, each in two forms, plus two length columns: 30 columns. With the seven, 54 of 78 word-only stand-ins passed in the units' place.

3. A 31st column, the cube of the word consensus. Without it, a measure that is only a convex function of word overlap passed 83 to 100% of the time. This is the judge's "curved function of word overlap", confirmed.

4. The unit measure enters as its rank within the pool (normal scores), not its value. The test is then unchanged by any increasing rescaling of the unit measure. Stand-ins below 0.05 fell from 3 to 0 of 78 and the power loss was under 0.1 SD of boost.

5. The penalty is fixed at 3, not tuned. Stand-ins below 0.05 are 0, 0, 0, 1, 6 of 78 at penalties 0.3, 1, 3, 10, 30.

6. Two rules added to the decision: the units must beat every one of 78 registered word-only stand-ins on the final data, and the test must be evaluable (at least 5 effective pairs left open by words).

Tried and dropped: word columns as ranks only (easy pairs then stop looking easy and every pair speaks, which lets any measure that agrees with words add apparent information); a local rank test among the source's neighbours in word score (uses only the nine unsolved pairs); a null drawn from the fitted word model (too dependent on that model being right in the tail).