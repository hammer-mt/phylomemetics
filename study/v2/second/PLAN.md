# Two more corpora: analysis plan

**Status: frozen before any study model read a new text as a study work.** This file, the two corpora, the code and the words-only results it cites were committed and pushed on 8 November 2026, before the first of those readings; the commit and its time are the first entry of Section 16. Section 15 lists what had been run and what was known by then. Changes made later are logged in Section 16, each with its reason.

Paper v2 (`study/v2/PLAN.md`, `study/paper/v2/paper.md`) re-analysed 32 Batman works with a new method. Its own limits section says what one franchise of 32 works cannot settle. This plan runs the same method on two corpora chosen to address those limits: the Sherlock Holmes stories and their screen versions, and film remakes. The scouting that led to them is in `preregistration/second_corpus/`.

The method of v2 is not changed: the units, the sign regime, the 30% rule, the reference corpus, the background plots, the 3% cap, the statistics of E1, E2 and E3. Where a new corpus forces a difference, this plan says so and why. The differences are collected in Section 6 (the scale) and in the "differs from v2" notes of Sections 8 and 9. The largest is the test of whether units add to words: v2's residual test turned out, before any model was run, not to be trustworthy on either new corpus, and the remakes get a different test (Section 9).

## 1. Why two more corpora

| Limit named in paper v2 | Holmes | Remakes |
| --- | --- | --- |
| One franchise, 32 works | A second franchise: 81 screen works and the 60 stories | Not a franchise |
| 9 adaptations on 7 sources; the test of neurons against words could not detect a small effect | 22 adaptations on 14 stories, 8 of them of one story. Still too few: the question is not tested on Holmes (Section 8) | The corpus built for this question (Sections 3 and 9) |
| 8 hero-film negatives | 229 detective films chosen by rule | No identity test |
| 15 early works for selection, 27% power | 60 stories; still low power once adaptations are controlled for (Section 8) | None |
| Name-free texts depend on an AI rewriter, and the codes leave a mark | The same procedure, so the same limit | Original texts are used: a remake changes the names by itself |
| Wikipedia tells a source and its adaptation in the same words | Not addressed | Not removed; the test is recast for it (Section 9) |
| The models have read the summaries | Expected to be at least as true as for Batman; measured (Section 10) | Measured (Section 10) |
| A second analysis of the same works | New texts, plan frozen first | New texts, plan frozen first |

What neither corpus addresses: presence is decided at 30% of scale, where descriptions are weakest; the scale is smaller than the essay's; the explanations are one AI reading; steering failed. None of the explanation, steering, twin or ablation tests of v2 is repeated here (Section 14).

## 2. The Sherlock Holmes corpus

Built by `v2/second/build_holmes.py` from live English Wikipedia on 8 November 2026. Every text is the plot section of the work's own article (the section names `study/fetch_corpus.py` accepts), converted from wikitext by `v2/second/plots.py`, split by v1's sentence splitter, first 40 sentences kept, at least 8 sentences required. Page id, revision id and revision time are stored with each text. The lists were made by rule:

- **Stories (60).** The Conan Doyle canon: 4 novels and 56 short stories.
- **Screen works (81).** Every page with a usable plot in Category:Sherlock Holmes films and its sub-categories, Category:Sherlock (TV series) episodes, and Category:Video games based on Sherlock Holmes and its sub-categories. By the kind of infobox: 46 films, 9 television films or programmes, 13 episodes, 13 video games.
- **Ancestors (5).** Poe's three Dupin tales, The Moonstone, Monsieur Lecoq. They are in the corpus and are read by the models, but no test of this plan uses them and nothing is reported on them.
- **Detective films (229).** Every American or British film in the local film table (the Kaggle "Wikipedia Movie Plots" table v2's reference corpus was drawn from) whose genre says mystery, detective or whodunit, whose plot there names neither Holmes nor Watson, whose page is in no Sherlock Holmes category, and whose live page has a usable plot and a film infobox. Four pages whose first infobox is a book's or a television programme's are kept in the file as `not_used`.

`data/second/holmes/corpus.jsonl` holds the 379 texts (9,837 sentences) in that order; ids are row numbers and never change. **Year, medium and era** come from each page's first infobox by rule: the earliest year in its date fields; the kind of infobox; era = up to 1949, 1950 to 1999, from 2000.

**What each screen work is based on** (`data/second/holmes/works.json`). Two readers (AI agents) read each screen work's article (its first 1,400 words, with the plot, cast, gameplay, reception, release, awards, music and end sections left out; six of the 81 articles were cut short by the limit) and wrote down every Conan Doyle story the article says it is based on, with the sentence that says so and a tier fixed by a written rule: A, the article states it is an adaptation of the story; B, it states a loose or partial adaptation; C, it states only that elements, characters or a title are taken. The readers saw no plot and no measure of word overlap. They agreed on all but two works; a third reader read those, and the majority decided (a tie goes to the weaker tier). The rule and the prompt are in `build_holmes.py` (`TIER_RULE`).

- **Primary adaptations: 22**, the screen works with at least one story at tier A or B, on 14 stories. Eight are versions of The Hound of the Baskervilles.
- Straight adaptations (a tier A story): 9. Elements only (tier C and nothing stronger): 16. No stated Conan Doyle source: 43.
- By the readers' own label, two of the 81 pages are about a whole series, not one work; they stay in. The readers also said whether Sherlock Holmes appears as a character: in 74 of the 81 (a split between the two readers counts as no).

No pair is added, dropped or moved on any other ground. In particular nobody's judgement of the plots decides a tier.

**Name-free texts** (`data/second/holmes/corpus_neutral.jsonl`), by the Batman corpus's procedure: an AI rewriter replaces every name of a person, place, organisation or title; the rewrite is used only to find the names; a word-level comparison then puts a code (X1, X2, ..., the same code for the same name within a work) where the rewriter changed something (`v2/second/names.py`, which reproduces `data/corpus_neutral.jsonl` from the Batman rewrites exactly; the prompt is `v2/prompts/name_remover.md`). A rewrite is refused and redone if its sentence count differs, if more than two names are left, or if it changed more than 12% of the lower-case words.

**Sets** (`data/second/holmes/spec.json`, made by rule from the two files above; the analysis code reads the sets from it and holds none of its own):

| Set | Works |
| --- | --- |
| franchise (E1) | the 81 screen works |
| negatives (E1) | the 229 detective films |
| lineage (E2) | the 60 stories and the 81 screen works |
| candidate sources (E2) | the 60 stories |
| adaptations (E2) | the 22 primary adaptations, each with its tier A and B stories |
| baseline (E2's hub adjustment, and the later works of E3) | the 43 screen works with no stated Conan Doyle source. In `spec.json` this is the key `baseline`; the key `later` there holds the wider set of 59 that E3 reports beside |
| early works (E3) | the 60 stories |

**Page views** (`data/second/holmes/pageviews.json`): total English Wikipedia views of each story's page, July 2015 to September 2026, the window of `data/pageviews.json`.

## 3. The remakes corpus

Built by `v2/second/build_remakes.py`. The films are those of the film table v2's reference corpus was drawn from (the pinned copy of the Kaggle "Wikipedia Movie Plots" table: 34,886 rows, 34,070 Wikipedia pages, plots as they stood in 2017-18). Which film remakes which is taken from what the films' own articles say today, and from nothing else.

**Statements.** For every page of the table the opening section of the live English Wikipedia article was fetched (8 November 2026; revision ids kept). A statement is a sentence of that opening that contains "remake", "remade" or "reboot" and links to at least one other page of the table.

**What each statement says.** Two readers (AI agents) saw, for each statement, the page's title, the sentence and the linked films, and nothing else: no plot, no measure of overlap. For each linked film they recorded whether the sentence says the page's film is a remake of it, that it is a remake of the page's film, or neither; and whether the wording is firm ("is a remake of", "was remade in Hindi as") or loose ("loosely", "unofficial", "inspired by", "has been described as", "reboot"). The rule is `REL_RULE` in the script. Where the two differ a third reader reads the statement and the majority decides; with no majority the link is dropped.

**Pairs** (`data/second/remakes/pairs.json`, each with its stating sentences). A pair (remake, source) is kept when:

- at least one statement says so, and none says the reverse;
- the table's year for the remake is not earlier than the source's;
- both plots have at least 15 sentences after cleaning;
- the two pages do not hold the same text.

*One row per page.* The table lists some films once for every industry they were released or dubbed in (780 pages have several rows, often with a later year). The row used is the one whose year is nearest to the earliest year in the date fields of the live article's first infobox (Section 2's rule); a tie, or an article with no such year, goes to the first row. Its year, industry and plot are the film's.

*One film, one text.* Two pages whose cleaned texts are identical are one film: the page that comes first in the table stands for both, and a pair naming the other is moved to it.

A pair is *firm* if at least one of its statements is firm by the readers' majority, else *loose*. Each kept pair is placed in a block by the table's industry labels: **Indian cross-industry** (both films from different Indian industries: Bollywood, Tamil, Telugu, Malayalam, Kannada, Bengali, Marathi, Punjabi, Assamese), other cross-industry, or same industry. The pairs that fail a condition are listed in the same file with the reason.

**Families.** A family is a set of films joined by any of: a remake statement that links two films, whatever the readers made of it (so two remakes of one original are relatives, and so are the films of a pair dropped for a short plot); two table pages that lead to the same live article; a film that holds at least 10% of the five-word sequences of a film of a kept pair (a copy of it under another title). A film is never a candidate source for a remake of its own family, and every test treats a family as one cluster. `pairs.json` lists the relatives that are in no kept pair.

**Copied summaries.** Some remake articles have a plot pasted from the source's article. A pair is flagged *copied* when at least 10% of the five-word sequences of the remake's text also occur in the source's text. That is 100 of the 220 pairs of the primary set. *Differs from the scouting notes*, which asked for the copied pairs to be moved out: they stay in the pair list, the heredity test is run on the pairs that are not copied with the copied ones as its positive control, and the decision test keeps them because they carry almost none of its weight (Section 9). Every result is given both ways.

**The primary set** is the firm Indian cross-industry pairs. The scouting judges asked for a homogeneous primary set, and these are the films a model is least likely to know by heart. The other two blocks and the loose pairs are secondary sets, reported separately and never pooled into the primary.

**Texts.** The table's plot for each film, cleaned by rule (`clean()` in the script) for every film alike: reference marks removed; any bracketed run of one to four capitalised words removed, which is how the table gives cast names ("Saravanan (Rajinikanth), a psychiatrist" becomes "Saravanan, a psychiatrist"); v1's sentence splitter; sentences that contain the word "remake", "remade" or "reboot" dropped; the first 40 sentences kept. Names are left in: a remake in another language changes the names by itself, and the word measures of Section 7 see the same names the units see. There is no name-free version.

**Films read.** Every film of the table with at least 15 cleaned sentences whose industry is the industry of the source of some kept pair, and every film of a kept pair; a text found under several pages is kept once. The texts are rebuilt from the pinned table by the script (`v2/cache/second/remakes/films.jsonl`); the committed index `data/second/remakes/films_index.json` gives each film's page, year, industry, sentence count and text hash.

**Candidates** are defined in Section 9. **Rows of the counts** (`data/second/remakes/rows.json`): the remakes of the kept pairs; and the films of the kept pairs, whose presence bits are fetched so that the counts can be recounted away from the GPU machine.

**In the reference corpus.** The table is the one v2's reference corpus was drawn from, so about a third of the films read are among its 12,000 film plots, and about 9% among the 3,000 background plots, in the same 2017-18 wording. The count for the films of the pairs is in Section 15. Nothing is done about it: the scale is the reference-only scale for every film alike, and the remake tests use no base rate.

## 4. What the models had already read

"No model has read these texts" needs two qualifications, and the plan states them here.

**The reference corpus holds older summaries of some of these films.** v2's reference corpus was drawn from the same film table, and was screened only against Batman's titles. Of the 310 films of the Holmes corpus (screen works and detective films), 112 are among its 12,000 film plots, in the table's 2017-18 wording: 34 among the 3,000 background plots and 78 among the other 9,000. All four models read those documents in v2's reference pass, and for the background plots the units present at 30% are stored. Nobody has analysed any of them as a study work. `spec.json` lists them (`in_reference_corpus`). The same count for the remake films is in Section 15.

**The scouting used words only.** The scouts computed word overlap between candidate pairs (TF-IDF ranks and cosines; `preregistration/second_corpus/`). The author of this plan has seen those figures. No rule in Sections 2 and 3 uses them, and no pair's membership depends on them.

**Two texts of the Holmes corpus are, sentence for sentence, texts of the Batman study**: A Study in Scarlet (story 0), which was one of v2's literary relatives, and The Bat (1926) (detective film 147), one of its eight hero-film negatives. All four models read them as study works in v2, and the pipeline check read them again. No choice in this plan used their units. E2 is also given without story 0 (the leave-one-story-out rows) and E1 without film 147 (Section 12). Of the films read for the remakes, 73 are, in the table's older wording, works of the Batman study (its ordinary films, mostly); seven of them are films of secondary pairs and none is in a primary pair.

No unit activation has been computed on any other text of `data/second/`, or on any remake film as a study work. The GPU scripts were tested on a 0.6B test model with Batman texts and made-up sentences only, and before the freeze two of the study models re-read the Batman works and the 34 older summaries just mentioned, as a check of the pipeline (Section 15).

## 5. Models and budget

The four models of v2 in the same roles. Qwen3-32B decides every test. Qwen3-8B, Llama 3.1 8B and Llama 3.3 70B are replications, reported whatever they show. "The same sign in Llama" means Llama 3.1 8B. The loader takes each model's current files from the Hub; what shows that the weights are v2's is the pipeline check of Section 6, which must reproduce v2's stored presence bit for bit.

Llama 3.3 70B is run at the primary setting only (strongest sentence at 30%; no other rule setting), as in v2. Of the remakes it reads the films of the Indian industries and every film of a pair (about a quarter of the films read), so for it the primary set can be run in its own frame and in the scouts' frame, not in the every-film frame, and the secondary sets cannot. It is run only if the money spent on this phase before it starts plus 25 dollars stays under the cap. The rule uses cost only.

Budget: 72.70 of the 200 dollars were spent on v2. This phase is capped at 60 dollars, the 2 dollars of the pipeline check before the freeze included. The models are started in this order: Qwen3-32B, Llama 3.1 8B, Qwen3-8B, Llama 3.3 70B. A model is started only if the money spent on this phase so far plus its allowance stays within the cap; the allowances are 15 dollars for Qwen3-32B, 6 for each 8B model and 25 for Llama 3.3 70B. If Llama 3.1 8B is not run, E3 is read on Qwen3-32B alone and the paper says so.

## 6. Units, scale, presence and base rate

Unchanged from v2 (`v2/PLAN.md`, Sections 4 and 5): a unit is one pole of one MLP neuron, main sign regime only; each text is read as one continuous document; position 0 is never scored; in name-free texts a code's own tokens cannot carry a unit's peak; a unit is **present** in a text when its strongest sentence reaches 30% of its scale, on v2's storage grid (steps of 0.005, rounded down); a unit is **distinctive** when it is present in at most 3% of the background plots.

**The scale: one difference from v2, fixed here.** In v2 a unit's scale was its strongest activation in the reference corpus, the supplement or the study's own original texts. Here the scale is the **reference-only scale** of v2 (`scale_ref.npy`: reference corpus and supplement, without v2's five study works that were found in the reference corpus; one of v2's registered sensitivity versions). Its SHA-256 for each model is in `results/v2/second/gpu_inputs.json`. No text of either new corpus can change it. Reasons:

1. It is fixed before the freeze, and it is the same for both corpora.
2. The remakes corpus has several thousand candidate films. v2's definition would let all of them raise the scale, which would need every text read twice.
3. In v2 the choice changed almost nothing (Qwen3-32B: E1 0.354 against 0.361; E2 0.149 under both; residual p 0.692 against 0.691), because the study texts set the scale of 4,518 of 3,276,800 units.

What this does not do: it does not remove the 112 Holmes-corpus films (and the remake films of Section 3) that are reference documents in their older wording; their activations are part of the reference maximum and cannot be taken out, because the reference pass's top lists were not kept. For a unit whose strongest activation anywhere is in one of the new texts, the stored share is above 100% and the unit is present there, as it would be under v2's definition; what differs is that other texts are measured against the lower, reference-only maximum. The pass records those units (plane `over100`), and every Holmes test is also reported with them left out (Section 12).

**Base rates.** The background is v2's 2,999 film plots, minus the 34 that are older summaries of Holmes-corpus films: 2,965 plots. A unit's base count is v2's stored count on the reference-only scale (`bg_counts_ref_scale.npy`) minus its count over those 34, which are read again by v2's own `background_pass.py` with the same four scale files. That run also reproduces the 34 plots' stored presence at v2's main scale; if it does not reproduce it exactly for a model, the base rates of that model are recounted over all 2,965 plots by a full run instead (`gpu.py bgfull`). Distinctive means at most 3% of 2,965, that is 88 plots or fewer. Base rates enter E1 and the canon only. The remake tests use no base rate.

**What is stored.** `v2/gpu/presence_pass.py` reads each text once and writes presence as packed bits, one plane per rule setting: strongest sentence at 20, 30, 40, 50% of scale; mean of the three strongest sentences at the same four; the opening only (at most 20 sentences and 400 words, v2's length control) at 30%; iconography sentences dropped at 30%; and `over100`. Its first job on each machine is a check on texts that are not new: it must reproduce, bit for bit, v2's stored presence of the Batman works for that model at v2's main scale: original and name-free texts, strongest sentence at 20, 30, 40 and 50%, the opening and the iconography-dropped text at 30%. If it does not, nothing else is run on that model until the cause is found and logged.

## 7. Word measures and other rivals

As in v2, every unit-based statistic is put beside the same statistic computed from words, and in the "beyond words" tests the part of the unit measure that the word measures predict is taken out first ("regressed out"). The word measures are v2's (`experiments_v2.py`: `word_measures`, `word_rates`), computed on the texts the units see (name-free with the codes replaced by a space for Holmes; the original cleaned texts for the remakes):

- TF-IDF cosine (English stop words dropped, words in at least 2 texts, sublinear term frequency);
- word presence: the inheritance rate of Sections 8 and 9 computed on words in place of units (a word is present in a text or not);
- word presence in four frequency bands;
- TF-IDF on raw counts.

These seven are v2's decision regressors; on each new corpus a calibration made before the freeze extends them (Sections 8 and 9). The every-measure form adds: TF-IDF with word weights from the background plots; word presence restricted to words in at most 3% of the background plots; TF-IDF over words and word pairs; and, for Holmes, the two code measures (similarity of the number of distinct codes; which codes a text contains).

The frequency bands follow the size of the lineage. Batman's optional band was 2 to 16 of 32 works with word bands 2-3, 4-6, 7-10, 11-16. For a lineage of n works the optional band is 2 to floor(n/2), and the word bands end at round(3n/32), round(6n/32), round(10n/32) and floor(n/2). For the Holmes lineage of 141: optional band 2 to 70; word bands 2-13, 14-26, 27-44, 45-70.

## 8. The tests on Holmes

All on name-free texts unless said otherwise, for Qwen3-32B at the primary setting (strongest sentence, 30% of the reference-only scale). The statistics are v2's; the code is `v2/second/holmes.py`, which must first reproduce v2's committed Batman numbers from a specification that restates the Batman corpus (Section 15).

**E1, identity.** As v2's Section 6. Each of the 81 screen works is held out in turn (a *fold*). Its canon units are the distinctive units present, in the original texts, in at least 80% of the other 80 (64 or more). The held-out work is scored by the share of that canon present in its name-free text; each detective film is scored on the same fold's canon. The AUC is the chance that a screen work scores above a detective film: 0.5 is a coin toss, 1 is perfect.

*Differs from v2:* the negatives are 229 films chosen by rule and they are older than the franchise (148 of 229 are from before 1950, against 24 of 81 screen works), so the AUC is taken **within era**: each screen work is compared with the detective films of its own era only, and the AUC is the mean over the 81 works. Its 95% interval comes from resampling screen works and detective films, independently within each era (a bootstrap, 2,000 draws). The label-shuffle null asks what AUC arises when the labels "screen work" and "detective film" are dealt at random: labels are permuted within each era, each era keeps its number of screen works, and every canon is rebuilt (1,000 draws). All six clauses below use the within-era numbers. The pooled AUC of v2 and its verdict are reported beside them.

- *Met* (all six, v2's thresholds): (1) AUC at least 0.80; (2) the interval's lower bound at least 0.65; (3) above the 95th percentile of the label-shuffle null; (4) within 0.05 of that AUC after dropping iconography sentences; (5) on the openings an AUC of at least 0.70; (6) which is above the 95th percentile of its own label-shuffle null.
- *Falsified:* AUC below 0.70, or an interval that includes 0.5.
- *Not met* otherwise. *Not evaluable* if the median canon has fewer than 20 units; this is decided first.

A pass would not yet be "identity holds". The count of distinct codes alone already separates the franchise from the detective films (0.65 within era, words only), and v2's checks of the codes are not part of this run. So if E1 is met, one follow-up is fixed now and logged in Section 16 when it is run: the name-free texts are read again with the token after each code masked too (v2's `after` version), and the within-era AUC must stay at 0.70 or above. Until then a pass is worded "met; not yet separated from the placeholder codes".

The iconography list for Holmes, fixed here (sentences of the name-free text matching it are dropped; the regular expression is in `spec.json`): sherlock, holmes, watson, moriarty, lestrade, mycroft, adler, baker, deerstalker, pipe(s), violin(s), magnifying, hound(s), deduce and its forms, detective(s), sleuth(s), consulting, landlady, cocaine, disguise and its forms, scotland yard, inspector(s), doctor, hansom, moor, reichenbach.

Reported beside it through the same procedure, as in v2: the original-text ceiling; sentence count alone; word count alone; the number of distinct codes alone; the TF-IDF centroid; the word canon share. Wherever the whole-work AUC is quoted, the openings AUC and the length-alone AUC are quoted with it. v2's secondary unit-centroid score is not run (the pass stores presence, not shares), so a failure here is a failure of the canon share: the paper does not say that the units cannot find the franchise.

The 60 stories take no part in E1: every negative is a film.

**E2, heredity.** As v2's Section 6. Optional units are present in 2 to 70 of the 141 lineage works. The rate R[a, c] is the share of story c's optional units also present in work a. Hub adjustment: the z-score of R[a, c] against the same rate in the 43 baseline screen works (the work itself left out). Time rule: a story is eligible for an adaptation if its year is not later than the adaptation's; documented sources are always eligible. For each adaptation, the share of eligible stories, other than its documented sources, whose z-score is above the best of its documented sources: 0 means the documented source always comes first, and about 0.44 is what a random relabelling gives.

*Differs from v2, two points.*

1. *The statistic gives each story one vote.* Eight of the 22 adaptations are versions of The Hound of the Baskervilles. The statistic is the mean, over the 14 source stories, of the mean share of the adaptations that name the story (an adaptation that names two stories counts under both; one 1980 film names three and so carries 2.5 of the 14 votes, which is why Section 12 also leaves each adaptation out in turn). v2's statistic, the plain mean over adaptations, is reported beside it, and so are both without The Hound (a story left out is removed from every adaptation's list; an adaptation left with no story drops out, which leaves 14 adaptations on 13 stories; each subset has its own 100,000 relabellings).
2. *The null is sampled.* v2 listed every relabelling of 7 sources among 12 comics. Here there are about 10^24. The null is 100,000 random one-to-one relabellings of the 14 source stories among the 60 (seed 20261108, the true labelling counted as one of them); adaptations of one story move together. p is the share of relabellings whose statistic is at or below the observed one. The smallest p it can give is 0.00001.

Prediction: p below 0.05. Evaluable only if every story has at least 50 optional units and no story's rate is constant across the baseline works; fallbacks as in v2 (threshold 20%, then mean of three at 30%), labelled as fallbacks.

Positive control, in place of v2's sequels (the corpus has four stated sequel pairs, too few): at least 7 of the 9 straight adaptations rank one of their tier A stories in the top two of the stories eligible by the time rule, by the raw rate (ties go to the story listed first). If it fails, a pass is reported as "supported, with a failed positive control". It is reported by story, because five of the nine are versions of The Hound.

If E2 is not evaluable at the primary setting, E2 and the residual test are both reported as not evaluable; the fallback setting is described and enters no claim and no adjustment.

Also reported: the share with ties counted as half; the straight adaptations alone; the 16 elements-only works as a secondary set (their tier C stories as sources); each jackknife that leaves one source story out.

**Beyond words on Holmes: the residual test is reported and is not confirmatory.** v2's residual test regresses the unit rate on word measures over all (work, story) cells and reruns E2 on what is left. On Holmes it is computed and its result is given, but **whatever it shows, the paper does not say that units add to words on Holmes.** The reasons were found before the freeze, with words alone:

- Words rank the documented source first for 16 of the 22 adaptations, so the regression has to extrapolate to the true pairs. That is the weakness Section 9 gives for not using this test on the remakes, and it holds here too.
- A straight-line regression lets curved functions of its own regressors through, because a documented source is an outlier in every word measure. With v2's form, 30 of 210 squares, products and cubes of the word measures passed in the units' place (8 squares, 21 pairwise products, 1 cube; p as low as 0.00002). With the squares and pairwise products regressed out as well, none of the 165 others of that kind passes (lowest p 0.054). But 82 of 422 further word-only functions of the same nine measures still pass: a measure divided by its largest value for the same work (8 of the 9 measures; TF-IDF cosine at p 0.00001), a power of 1.5 (6 of 9), its rank among the work's stories (4 of 9). The regression also over-corrects: most of the functions tried give p above 0.975. The family of such functions has no end, so no finite check makes a pass, or a null, here mean what it should.
- 14 sources are too few for the test built for the remakes.

The remakes decision test of Section 9 is the only test of that question in this plan.

*The form that is reported.* The unit rate R is regressed, over the 141 x 60 cells with a story's cell with itself left out, on a constant, nine word measures, their squares and their pairwise products (54 columns); the E2 statistic and its sampled null are computed on what is left. The nine are v2's seven and two that a calibration added before the freeze by v2's own rule (its change 6): every word-only measure of v2's list went through the test in the units' place, and while any passed on either statistic the one with the lowest p was added (word presence of words in 14 to 70 lineage works, p 0.010 and 0.006; then TF-IDF with background-plot weights, p 0.026 on v2's statistic; then none, lowest p 0.36). The every-measure form does the same with every word and code measure of Section 7 (thirteen, 104 columns). A planted effect of one standard deviation that no word measure knows about is found in 39 of 40 draws, and in none of 40 without it. Reported with it: the 8 rule settings, the subsets of E2, and the result for the adaptations whose source the model does and does not name. The numbers of this paragraph are in `results/v2/second/holmes_words.json`.

**E3, selection.** As v2's Section 6, with the sets of Section 2. Early works: the 60 stories. Later works: the 43 baseline screen works. Units present in 2 to 48 of the stories (at most 80%). For each unit, the share of later works that have it, with the number of stories that have it regressed out; for each story, the mean of that over its units, then regressed on the story's year, its number of sentences, its centrality (mean Jaccard overlap of its units with the other stories') and, new here, the number of screen works whose article names the story as a source at any tier. Statistic: Spearman correlation with log10 of mean monthly page views. Null: 10,000 permutations of page views within novels and within short stories; one-sided p.

Prediction: rho above 0, p below 0.05, and the same sign in Llama 3.1 8B.

*Power, computed before the freeze with no model* (2,000 simulated data sets with the real page views and strata; the four novels are four of the six most-viewed stories): against a true correlation of 0.3 the test has 68% power with no covariate, 56% with v2's three covariates, and 32% with the fourth covariate added, because the number of screen works that name a story is itself correlated with its page views (0.58). v2's own simulation, which ignores covariates, gives 75%. So with the covariate the scouting judges asked for, E3 again has too little power to settle anything: neither a pass nor a null is read as a finding about selection (Section 11). The version with v2's three covariates is reported beside it.

Also reported: the later set widened to the 59 screen works that are not primary adaptations; v2's placebo with the units' background base rate as the outcome (v2's reversed-time placebo is not run: nearly every story precedes every screen work; the three earliest screen works, of 1914, 1916 and 1922, precede up to 14 of the stories and are kept); the leave-one-out jackknife.

## 9. The tests on the remakes

On the original cleaned texts, for Qwen3-32B at the primary setting (strongest sentence, 30% of the reference-only scale). The registered tests are `v2/second/remake_test.py` and `v2/second/remakes.py` at the freeze commit (their SHA-256 is in Section 15); this section describes them. A difference between this section and the code that is found later is logged in Section 16 and the result is given both ways.

**The unit measure** is v2's inheritance rate, counted over thousands of films. A unit is *optional* when it is present in at least 2 of the films a model read and in at most half of them (v2's band, 2 to half of the lineage). For a remake r and any film c, rate(r, c) is the share of c's optional units that are also present in r. The counts are made on the GPU machine from the stored presence (`v2/gpu/copresence.py`), for every remake against every film read, and recounted on this machine for the films of the pairs (`remakes.py verify`); any difference stops the analysis. **Hub adjustment**, as in v2: each rate is turned into a z-score against the same film's rates with the other entries of the set being analysed that are not the film's relatives (the other remakes play the part of v2's non-adapting works).

**Entries and candidates.** Each documented pair is one entry (a remake with two documented sources is two entries). An entry's candidates are never later than the remake by the table's year and never of the remake's family. Which films are candidates follows from what defines the set, and from nothing about the answer:

| Set | Candidates |
| --- | --- |
| Indian cross-industry (primary) | films of the Indian industries other than the remake's own |
| other cross-industry | films of any industry other than the remake's own |
| same industry | films of the remake's own industry |
| all firm pairs; loose pairs | every film read |

*Differs from the scouting notes*, which proposed the films of the source's industry: that uses part of the answer. Two other frames are run for the primary set as registered secondaries: the scouts' (films of the source's industry) and every film read. The second leaves more pairs open, but a reviewer showed that half of its word-similar wrong candidates are films of the remake's own industry, which cannot be the source of a cross-industry pair; that is why it is not primary.

**Heredity.** For each entry, the share of its candidates whose adjusted rate is above its source's. The statistic is the mean over families of the family's mean share. Null: 10,000 relabellings (seed 20261108, the true labelling counted as one); in each, one film is drawn at random for each family from the films that are candidates of every entry of the family, and plays the source of all of them; p is the share of relabellings at or below the observed statistic. **The confirmatory statistic is computed on the 120 primary pairs that are not copied**; the figure for all 220 is given beside it. Prediction: p below 0.05. This is expected to hold, and the word measures pass it too; it is reported for the units and for TF-IDF, BM25 and word presence side by side, with how often each ranks the source first. **Positive control:** in the 100 copied primary pairs the units must rank the source strictly above every candidate in at least 90%. If they do not, every remakes result carries the words "with a failed positive control"; no verdict changes.

**Do units add to words? The decision test.** v2's residual test cannot be used here, and three statisticians (AI agents, working apart, on word data only) were asked to design and calibrate a replacement (`preregistration/second_corpus/designs/`). All three found the same two things. First, v2's seven word measures are far too weak as a control on this corpus: with them, more than half of a panel of other word-only measures "passed" in the units' place. Second, only the pairs whose source the words leave in doubt can speak. The adopted test is the one with the best calibration and power. In outline:

1. *A word model.* It gives every candidate a probability of being the source, from words alone. Fourteen word measures between each remake and each candidate (TF-IDF in four weightings, BM25, word pairs, word pairs up to three apart, character sequences, a latent semantic space built from word counts, word presence and its four frequency bands), each in two forms (standardised within the entry's candidates; hub-adjusted, then standardised), two length measures and one curved term (the cube of the word measures' average, so that "far ahead on the words" can count for more than in proportion): the designers' 31 columns. Four more word measures were added before the freeze by the rule given below, in both forms, which makes 39. The model is a conditional logit: within each entry's candidates, the probability that c is the source is proportional to the exponential of a weighted sum of c's columns. The weights are fitted with a ridge penalty of 3 (a pull towards zero that keeps them stable), **leaving one family out at a time**, so every entry is scored by a model that never saw its family (*held out*).
2. *The statistic.* The unit measure enters only as its rank among the entry's candidates. For each entry: the unit score of the true source, minus the average unit score of the candidates, each weighted by the probability the word model gives it; both after taking out of the unit score what the word columns explain (among the other families). These are summed within families, then over families (T); Z is T divided by the root of the sum of the squared family sums. An entry whose source the words already find with near certainty contributes almost nothing, whatever the units say. The pairs that can speak are the *open* ones: the number of effective open pairs is the sum over entries of one minus the held-out probability of the true source.
3. *p value.* One-sided. If the units carried nothing beyond the words, each family's sum would be as likely negative as positive; so the signs of the family sums are flipped at random (100,000 draws, seed 20261108; every sign pattern is listed when at most 20 family sums are non-zero) and p is the share of draws whose total is at least the observed one.
4. *The gain.* The model is refitted with the unit column added; the held-out probability it gives the true sources (summed as logarithms) must go up.
5. *Stand-ins.* Word-only measures that are not among the columns go through steps 2 and 3 in the units' place on the same data: the designers' 78, less the four that became columns, so 74.

"The units carry information about which film is the source beyond these word measures" is claimed only if all seven conditions hold for the primary set:

1. evaluable: the word model leaves at least 5 effective pairs open, and the test's own positive control passed on the final list (below);
2. calibrated on the final list: at most 10% of the stand-ins have p below 0.05;
3. the units' p is below 0.05;
4. the gain is positive;
5. the units' Z is above the Z of every stand-in;
6. the units' p is below 0.05 in at least 4 of the 8 rule settings (threshold 20, 30, 40, 50% by strongest sentence or mean of three);
7. the pass is not recall. Read from the primary fit's per-entry contributions, nothing refitted: over the entries Qwen3-32B does not recognise (Section 10), the open pairs must number at least 5 and the sign-flip p of their family sums must be below 0.05.

The wording of each outcome, fixed now:

- condition 1 or 2 fails: "not evaluable" or "not calibrated"; no claim either way;
- condition 3, 4 or 5 fails: "no information beyond these word measures was detected", with the smallest planted effect the test detects with 80% power on the final list beside it (Section 15);
- only condition 6 fails: "detected at the primary setting only"; no claim;
- only condition 7 fails: with fewer than 5 open pairs among the unrecognised entries, "cannot be separated from recall", no claim; otherwise "the advantage sits in pairs the model recognises: recall";
- conditions 6 and 7 both fail: both sentences are given; no claim;
- all hold: the claim, worded "beyond these word measures", never "beyond words". If the sum over the entries with a copy share under 1% is not above zero, it is worded "in pairs whose summaries share copied wording".

**An embedding in the units' place.** One named sentence-embedding model (`sentence-transformers/all-mpnet-base-v2`; a film is the mean of its sentence vectors; `v2/gpu/embed_pass.py`) goes through the same test as the units. It is not a study model and its result decides nothing. It "passes" when conditions 3 to 5 hold for it. If the units pass and the embedding passes too, the paper says "beyond word counts; a general-purpose embedding does as well". If the units fail and the embedding passes: "a general-purpose sentence embedding carries information beyond these word measures; the units do not". Registered secondary: the unit test again with the embedding's two forms added to the word columns; if the units pass only without it, the claim reads "not beyond a general-purpose embedding".

**Reported beside the decision, whatever it shows.** Sums of the primary fit's contributions over subsets, nothing refitted, each with its own p and open pairs: not copied; copy share under 1%; recognised and not recognised; each of the five families with the largest sums left out in turn; contributions set to zero where the word model gives the true source a probability of 0.99 or more. Also: penalties of 1 and 10; the unit rate without the hub adjustment; the two secondary frames; the secondary sets (each through conditions 1 to 5 and heredity); the other models; heredity and the decision test with the units left out that are above their scale in any film read (a second count made on the GPU machine: the remakes' counterpart of Section 12, item 2); and the contributions summed separately for entries whose source is, and is not, a document of v2's reference corpus. For Llama 3.3 70B a unit is optional among the films it read.

**The word columns were extended before the freeze, by rule.** The designers calibrated their 31 columns on about 95 pairs the scouts had recalled, where no stand-in passed. The final list is larger and harder (220 primary pairs, many with copied or partly copied wording): with the 31 columns, 13 of the 78 stand-ins had p below 0.05 in the units' place (the rule allows 10%), in the primary frame and with words alone. As it stood the test could only have answered "not calibrated". So the designers' own procedure, which is also v2's rule for its residual test, was applied before any unit existed: while more than 10% of the stand-ins that are not yet columns have p below 0.05, or any has p below 0.01, the one with the lowest p joins the word columns in both forms, the word model is refitted and the rest are run again (primary set, primary frame, at most 20 additions). It stopped after four: word pairs with stop words kept, a latent semantic space of 50 dimensions, the share of a candidate's repeated words that the remake has, and a latent semantic space of 300 dimensions. With the 39 columns, 7 of the 74 remaining stand-ins have p below 0.05 (9.5%), none is below 0.01, and the largest Z is 2.23. The word model then ranks the source first for 205 of the 220 pairs and leaves 21.1 effective pairs open (23.8 before the additions). The 39 columns are the registered word model for every set, frame, penalty and model, and the 74 remaining stand-ins are the registered panel. The stand-ins took part in choosing the columns, so they are not an independent check; the checks that took no part are listed next.

**What was calibrated on the final list with no model** (numbers in Section 15). Done before the freeze: how many pairs the words leave open; the extension of the columns; the panel; and the test's own positive control (the word model is given a random half of the vocabulary, and word measures on the other half are put in the units' place: they passed in each of three splits, Z 5.2 to 5.7, p 0.00001).

Still being computed when the plan was frozen, with words alone, and logged in Section 16 before any unit analysis is run: the checks that took no part in choosing the columns (about 330 random word-only measures of the designers' random families; noisy and curved copies of the word columns) and the planted-effect power (the smallest effect the test detects with 80% power). Their consequence is fixed now: if more than 10% of the random measures, or of the copies, have p below 0.05 in the units' place, the test is "not calibrated" (condition 2 fails) and no claim is made either way. Nothing else in the plan depends on them.

The driver `remakes.py` computes what this section defines from `remake_test.py`, the 39 columns and the 74 stand-ins, all fixed at the freeze. It was still being finished and reviewed when the plan was frozen, because the GPU machines' work does not depend on it; it is committed, with its words-only output `remakes_words.json`, before any unit analysis is run, and anything in it that departs from this section is logged in Section 16.

## 10. Does the model recognise the works?

As v2's Section 7: recognition is measured on the untouched model and every result is split by it. The questions are fixed here, word for word. All replies are greedy; the reply length is 12 tokens for question H1 and 24 for the others.

**Holmes, on the name-free texts.**

- H1, all 379 texts, v2's question 1 unchanged: "Every name in this plot summary has been replaced by a code such as X1. Which famous fictional hero is the story about? Reply with only the name the hero is best known by, not a code." Kept: the reply, and the log of the probability that it starts with "Sherlock" or with "Holmes" (the two are added; neither is the beginning of the other). A text is *named* when the reply contains holmes, sherlock, watson, moriarty or baker street (case ignored).
- H2, the 141 lineage works: "Every name in this plot summary has been replaced by a code such as X1. It is a summary of a Sherlock Holmes story, or of a film, television episode or game about Sherlock Holmes. Which story by Arthur Conan Doyle does it tell or adapt? Reply with the story's title only." A work's source is *named* when the reply contains the title of the story itself (for a story) or of one of its documented stories (for an adaptation), by the rule below. H2 is asked after H1 and in a separate run, since it gives the franchise away.

Fixed uses: the rank correlation, over the 81 screen works, between E1's score and H1's log-probability; E1's AUC and E2's statistic for the works the model names and for those it does not. Reported with H1, as in paper v2: how many of the 229 detective films the model names as Holmes, and the within-era AUC of H1's log-probability between screen works and detective films, quoted beside E1. Reported with H2: for the 43 screen works with no stated source, which stories the model gives (its guesses where there is nothing to recognise).

**Is a pass confined to what the model recognises? The rule**, on Qwen3-32B's own replies. The *not-named side* is: for E1, the screen works not named in H1; for E2, the primary adaptations whose source is not named in H2; for the remakes, the primary pairs that are not recognised (below).

- E1: the within-era AUC of the not-named screen works, on the full folds' canons, must be at least 0.70 with an interval that excludes 0.5. With fewer than 10 such works the result is reported as it fell with the words "cannot be separated from recognition", and "identity holds" is not claimed. Where E1 is not evaluable, the rule is not evaluable either.
- E2 and the reported residual test are given for the not-named adaptations alone (their own 100,000 relabellings; not read with fewer than 5 such adaptations), for description.
- Remakes: Section 9.

If the side can be read and fails, the paper says the model recognises the plots, and does not say that identity is supported.

**Remakes, on the texts as the models read them (names in).**

- R1, every film of a kept pair: "Which film is this a plot summary of? Reply with the film's title only." A film is *identified* when the reply contains its title.
- R2, every remake of a kept pair: "This is the plot summary of a film that is a remake of an earlier film. Which earlier film does it remake? Reply with the earlier film's title only." A remake's source is *named* when the reply contains the title of one of its documented sources.

**"Contains the title"** is decided by rule, with no judge, for Holmes and the remakes alike (`v2/second/probes.py`). Reply and title are put in lower case and stripped of accents, punctuation and the words the, a, an wherever they stand; a bracketed suffix and a leading "The Adventure of" are dropped from the title. The title is contained when its words stand together as whole words in the reply, allowing for another spelling: no letter may differ in a title under 5 letters, 1 in a title of 5 to 9, 2 in 10 to 19, 3 in longer ones. A title of fewer than four letters must be the whole reply. A film known under another title will still be missed, so the counts are close to lower bounds.

A pair is *recognised* when the model names its source in R2 or identifies either of its films in R1. *Differs from v2*, which used a blind judge for titles: the rule can miss a title, and a missed title would put a recognised pair on the unrecognised side, so the wide definition (either film) is the one used. The count with both films identified is reported beside it. Every remakes result of Section 9 is given for the recognised pairs and for the rest; the rule that decides whether a pass is recall is in Section 9.

The probes are asked of each model on the untouched texts, and their tallies are committed before any unit analysis of that model is run (the analysis scripts refuse to run without them). No design choice waits for them.

## 11. How each outcome will be read

One statistic per question, all for Qwen3-32B at the primary setting. The wording for each way a test can fall is fixed here.

| Question | Batman (paper v2) | The statistic here | If it holds | If it does not |
| --- | --- | --- | --- | --- |
| Holmes E1. Without names, does a canon of units tell the franchise from its nearest genre? | Falsified: AUC 0.36 against 8 hero films | Within-era AUC against the 229 detective films, six clauses | Met: "met; not yet separated from the placeholder codes", and the follow-up of Section 8 is run. Only if that holds too, and the pass is not confined to works the model names (Section 10): "identity holds on a second franchise where it failed on the first". The length-alone and code-count AUCs are quoted beside it | Falsified, with the openings AUC also below 0.70: "the canon is made of names on a second franchise too". Falsified on whole works only: "not met; the screen works' texts are shorter" (sentence count alone gives 0.39 here, so length works against the franchise). Not met: "identity is not established; without names the canon reaches x (openings y)". Not evaluable: no reading |
| Holmes E2. Do adaptations share optional units with their documented source? | p 0.0016; words pass too | Source-averaged share, sampled p | (expected to hold) "Heredity by units holds on a second franchise"; the word measures' own result is given beside it, not tested against it | p of 0.05 or more: heredity by units is not supported on Holmes, and the paper says so |
| Holmes E3. Do more popular stories leave more of their units in later works? | No claim (27% power) | Spearman rho, permutation p within novels and short stories | "E3's registered prediction is met on Holmes" (rho, p, Holm p). The paper does not say that selection is detected: the test finds a true correlation of 0.3 one time in three, and page views are measured after the adaptations | No claim, as in v2 |
| Remakes: do remakes share optional units with their documented source? | (no counterpart) | Family-averaged share of candidates above the source, sampled p (Section 9) | (expected to hold) "The units find a remake's source among its candidates"; the word measures' own result is given beside it | p of 0.05 or more: heredity by units is not supported on the remakes, and the paper says so. A failed positive control is stated in the same sentence as any remakes result and changes no verdict |
| Remakes: do units add to words? | No on Batman: p 0.69, with 9 adaptations | The decision test and its conditions (Section 9), firm Indian cross-industry pairs | "The units carry information about the source beyond these word measures", with the size of the effect, how many families carry it, and what the embedding did; only if not confined to recognised pairs | Not evaluable or not calibrated: no claim. Otherwise: "no information beyond these word measures was detected", with the smallest effect the test could have detected. This is the paper's main answer to the question |

The paper's answer to whether units add to words is the remakes decision test, whichever way Holmes falls.

A result that cannot be evaluated is an outcome of its own and is never read as a pass or as a failure. Replications are reported whatever they show. A result that holds only in a replication model or only at a sensitivity setting is labelled so wherever it appears.

## 12. Robustness fixed in advance

Each item will be reported, or logged in Section 16 as not run with the reason.

1. **Rule settings.** Holmes E2 and its reported residual test, and the remakes decision test, at each of the 8 uncapped settings (threshold 20, 30, 40, 50% by strongest sentence or mean of three), for Qwen3-32B, Qwen3-8B and Llama 3.1 8B. For the remakes these are the settings of condition 6. Llama 3.3 70B has the primary setting only.
2. **Scale.** The four Holmes tests with the `over100` units left out: every unit that is above its scale in any of the 379 texts, in either version (the units for which the reference-only scale and v2's own definition differ). Subsets, jackknives and placebos are not rerun.
3. **Length.** E1 on the openings (part of its criterion). E2 on the openings is not run.
4. **Codes.** The number of distinct codes alone through E1; the two code measures in the reported residual test. v2's "codes left in" version is not run; its "token after the code masked" version is run only as the follow-up of Section 8.
5. **Franchise definition.** E1 with the franchise narrowed to the 74 screen works in which, by the readers of Section 2, Sherlock Holmes appears as a character; with the franchise narrowed to its 46 films (from 2000 on, 26 of the 35 screen works are episodes or games, set against 20 detective films); and without detective film 147 (Section 4).
6. **E2 subsets** (and the same for the reported residual test). v2's statistic (mean over adaptations); without The Hound of the Baskervilles; straight adaptations only; the elements-only set; leave one source story out; leave one adaptation out.
7. **E3.** The later set widened to 59 works; the base-rate placebo; leave one story out.
8. **Original texts.** E2 on the original texts (names in), as the ceiling.

The remakes' list of what is reported beside the decision is in Section 9.

Not run, and why: v2's base-rate caps other than 3% (they were descriptive in v2 and need no new information here); E1 and E3 at the other thresholds and with the mean of three; the unit-centroid score; the fifth-example scale and the weak-pole and echo-regime versions (properties of the units, examined on Batman); the E2 multiverse of v1, its placebo targets and its stronger nulls (their code is written for 12 comics).

## 13. What is confirmatory

One family per corpus, all for Qwen3-32B at the primary setting.

- **Holmes, three tests:** E1 (met or not); E2 (sampled p); E3 (p). The residual test on Holmes is not in the family (Section 8).
- **Remakes, two tests:** heredity (sampled p); the decision test (the p of its statistic; the claim also needs the other conditions of Section 9).

Each test is read at 0.05. Holm-adjusted p values are given beside them within each corpus, and over all the new tests of both corpora together (`v2/second/confirmatory.py`); E1 has no p, as in v2, so for Holmes the adjustment is over E2 and E3, for the remakes over heredity and the decision test, and together over the four. Holm's adjustment multiplies the smallest p by the number of tests, the next by one fewer, and so on. E3 has about a third of the power it would need (Section 8), so its p will usually be large and E2 is in effect read at half its level. A result below 0.05 whose Holm p within its corpus is 0.05 or more is called nominal wherever it appears. A test that is not evaluable or not calibrated contributes no p; a fallback setting never enters a family.

Everything else is secondary or descriptive and is labelled so: the replication models, the sensitivity settings, the subsets, the splits by recognition, the word comparators' own results.

## 14. What this plan is not

- It does not repeat v2's explanation tests, steering, twins or ablation on the new corpora. Those are about the units, not about the corpus. The follow-up ablation's form (the reply begun with "Answer:") has still been run only once, on Batman.
- It does not add a second name-free version with ordinary names swapped in, which the scouting notes asked for. Assigning fitting names needs a second AI pass that knows whether a code stands for a person, a place or a ship, and it would be a new, untested text procedure. Nothing stands in for it on Holmes beyond the code-count comparator and the follow-up fixed in Section 8; the remakes use original texts.
- Other conditions of the scouting notes that are not met: the detective films are not matched by decade (three eras are used, because several decades have fewer than five works on one side); there is no title-only control; candidates are not limited to films within five years of the remake; no sentence is cut from a Holmes text for naming its source (the name-free version turns titles and authors into codes).
- It does not check the film table's plots against the films, beyond the one-row rule, the year rule and the same-text rule of Section 3. A row that holds the wrong film's plot counts against words and units alike.
- It does not use a human reading of any text or pair. Every label comes from a rule applied to Wikipedia's own wording by AI readers who could not see the plots.
- It does not replace the planted-story pre-registration (`preregistration/`), which is still not run, and makes no claim about keywords, embeddings as a product, LLM reading, or cost.
- It writes only under `study/v2/second/`, `study/data/second/`, `study/results/v2/second/`, `study/v2/gpu/` (four new files: `presence_pass.py`, its test, `copresence.py`, `embed_pass.py`), `preregistration/second_corpus/` and, for the report, the v2 draft of the paper and its Claude Doc. `study/v2/PLAN.md` gains one entry in its Section 18 that points here. Paper v1, the rest of v2's plan, v2's committed results and the companion page stay as they are.
- No number from the design reports, the scouting or an interim list goes into the paper: every number comes from a committed file under `study/results/v2/second/`.

## 15. What had been run and what was known when this plan was frozen

**No study model had read any text of `data/second/` or any film text of the remakes corpus as a study work, with the two exceptions of Section 4** (two Holmes-corpus texts that are also texts of the Batman study). What had been run:

**The corpora.** Built on 8 November 2026 as Sections 2 and 3 describe. Holmes: 379 texts; the readers of the screen works' articles agreed on all but two works; 22 primary adaptations on 14 stories. Remakes: 34,070 pages scanned (172 have no article today); 1,693 statements that link to another film of the table, read by two readers who differed on 9 of them; 1,130 stated pairs; 482 kept (643 dropped for a plot under 15 sentences, 5 for the table's years); of the kept, 220 firm Indian cross-industry (the primary set, 100 of them copied), 50 firm other cross-industry, 145 firm same-industry, 67 loose; 105 copied in all; 400 families, with 69 relatives that are in no kept pair; 16,218 films read. For 126 of the 780 pages with several rows a later row is used. Interim pair lists (before the last pages were scanned, and before the reviewers' corrections to the builder) were used for the one design choice of Section 9 that looked at data, the candidate frame; that look used words only.

**Words only, on Holmes** (`results/v2/second/holmes_words.json`, `holmes_e3_power.json`; no model):

- E1 by words, within-era AUC on name-free texts: TF-IDF centroid 0.62 (interval 0.53 to 0.70; pooled 0.65; iconography dropped 0.52), falsified by the criteria; word canon share not evaluable (3 words in the canon); sentence count alone 0.39; word count alone 0.47; number of distinct codes alone 0.65 (0.57 to 0.73), above its shuffle null's 95th percentile of 0.58. On the original texts the TF-IDF centroid reaches 0.98. So on name-free Holmes texts the word comparators do not pass E1 either, and the count of codes separates the franchise better than the words do.
- E2 by words: every word measure has p = 0.00001 (the smallest the null can give); shares over sources 0.058 (TF-IDF) to 0.095; the positive control holds for TF-IDF (9 of 9). The elements-only set is not found by TF-IDF (p 0.45).
- The residual test's calibration, and why it is not confirmatory: Section 8.
- E3 power: Section 8.

**Words only, on the remakes** (`results/v2/second/remakes_words_parts/`; no model). Primary set: 220 pairs (217 remakes, 185 sources, 175 families; 100 copied, 120 not; 96 with under 1% of five-word sequences shared). In the primary frame a remake has between 90 and 3,374 candidates (median 1,545); in the scouts' frame 27 to 1,241 (median 382); with every film read, 3,200 to 16,217 (median 11,940).

- The designers' 31 columns: source first for 204 of 220; 23.8 effective pairs open; 13 of 78 stand-ins below 0.05. After the four additions of Section 9 (39 columns): source first for 205; 21.1 effective pairs open; 7 of 74 stand-ins below 0.05 (9.5%), none below 0.01; largest Z 2.23 (stemmed words and word pairs, p 0.011). So conditions 1 and 2 hold on the final list, with words alone, and the units' Z must be above 2.23.
- A names-free word measure a reviewer asked for (TF-IDF over words never capitalised inside a sentence), not one of the 74: Z -1.5, p 0.93.
- The test's own positive control: passed in all three splits (Z 5.2 to 5.7, p 0.00001; the half-vocabulary word model leaves 43 to 47 pairs open).
- The same 39 columns in the other frames and sets (stand-ins below 0.05, of 74): every film read 2; the scouts' frame 12; same industry 31; other cross-industry 22; loose pairs 36. So the scouts' frame and the three secondary sets are "not calibrated" by words alone: they are run and reported, and make no claim either way. The every-film frame is calibrated and leaves 28.3 pairs open.
- Of the 887 films of the pairs, 479 are documents of v2's reference corpus (159 of them background plots); of the 401 films of the primary set, 228; the source is a reference document in 137 of the 220 primary pairs.
- Still running at the freeze: the random families, the copies and the power (Section 9).

**The design of the remakes test.** Three statistician agents, each with word data only and lists the scouts had recalled (about 100 pairs), proposed and calibrated a test (`preregistration/second_corpus/designs/`). The plan's author chose among them before any unit existed.

**The pipeline, on texts that are not new.**

- `v2/gpu/presence_pass.py` was tested on a 0.6B test model on CPU with Batman texts and made-up sentences: bit for bit against a slow re-computation and against v2's `study_pass.py` with `finalize_scale.to_shares` (`v2/gpu/test_presence_pass.py`), and an independent reviewer showed the test catches deliberate errors.
- On GPU machines (8 November, 18:01 to 18:44 and 20:00 to 20:38 UTC; RTX PRO 6000; about 7 dollars) Qwen3-32B, Qwen3-8B and Llama 3.1 8B read the 114 works of the Batman study with the new pass at v2's main scale. The presence it wrote equals v2's stored presence in every bit, on every plane that can be compared: strongest sentence at 20, 30, 40 and 50%, the opening and the iconography-dropped text at 30%, for original and name-free texts (`results/v2/second/pipeline_check_<tag>.json`). The same three models re-read the 34 background plots that Holmes's base rates leave out, with v2's `background_pass.py`: their presence equals v2's stored presence in every bit (`bgdrop_check_<tag>.json`). These are texts v2 had already read. The machines were then ended. The check is repeated on every machine after the freeze; Llama 3.3 70B has it only then.
- `v2/gpu/copresence.py` was checked against a brute-force count on made-up bits, with and without the units above their scale.
- Every new text was passed through each model's tokeniser (no model): all can be placed; the longest is 1,770 tokens.
- The analysis code for Holmes (`v2/second/lib.py`, `holmes.py`), given a specification that restates the Batman corpus, reproduces v2's committed numbers (`results/v2/second/batman_regression_<tag>.json`: 77 rows for Qwen3-8B and Qwen3-32B, all equal, one in the sixteenth digit), and was run end to end on made-up presence with and without a planted signal. An independent reviewer re-derived two of its statistics by brute force.

**Reviews.** The GPU pass and the Holmes analysis code were each reviewed by an independent AI reviewer told to find what was wrong, and the fixes to the Holmes code were verified by another; the review of the remakes driver was still to come at the freeze (Section 9). This plan was reviewed by three more (a statistician, an auditor of the pre-registration and a sceptical reader). Their findings changed it in many places: the Holmes residual test became descriptive, the candidates of the remakes follow from what defines the set, families include every stated relative and copy, the recognition rule is computed, and the two texts shared with the Batman study are disclosed.

**What the author knew.** Paper v2's results on Batman. The scouts' word-overlap figures for candidate pairs. The words-only numbers above. Nothing about any unit on any new text.

**Hashes** of every file a model will read, of each model's scale file and stored background counts, and of the test code (`remake_test.py`, `lib.py`, `holmes.py`, `probes.py`, the GPU scripts) are in `results/v2/second/gpu_inputs.json`, committed with this plan. `v2/second/gpu.py` refuses to run a step on a file that does not match, or unless the plan, the corpora, the code (the driver `remakes.py` apart, Section 9) and the words-only results are committed and pushed.

## 16. Changes after the freeze

Every change below was made before the result it bears on existed, unless it says otherwise.
