# a2-robustness: audit of the robustness work in the Batman study

Scope: `study/robustness/*.py`, `study/results/robustness/*.json`, `study/results/{design_reviews,factcheck,wording,labels,labels_wf,name_free_rewrite_notes}.json`, and the two commits `7e8c4c5` ("Fix the analysis after the robustness review") and `6db7807` ("Revise the paper, page and README to match the corrected results"). Nothing was run; numbers were read from committed files with small parsing scripts. Paths are relative to `study/` (at the time of the two commits the same files sat at the repo root).

## Bottom line

- The review did not change the primary E1/E2/E3 numbers. It changed what they mean. Almost every headline of the first draft was either killed or cut down to "word counts do this too" (diff of `paper/paper.md` in `6db7807`).
- The only result that survived every attack is the pattern "adaptations sit closer to their documented source comic than to other comics, even without names". It holds in 30,108 of 30,240 specifications (`results/robustness/e2_multiverse.json`). But TF-IDF finds the sources just as well (11% against 16% Qwen and 10% Llama; `results/experiments.json`, `lexical_comparators`), so this is a fact about the texts, not about neurons.
- The single piece of evidence that neurons see more than words on E2 is Llama's residual test (exact p 0.033, 0.014, 0.010). It is one specification, 9 adaptations, uncorrected for 6 tests, never put through the multiverse or jackknife, and under the stricter within-era null its TF-IDF version is p = 0.065 (`results/experiments.json`, `llama3.1-8b.E2.residual_test_beyond_words`). The primary model, Qwen, fails it outright (p 0.09 to 0.25).
- "Two models carve the same memes" is dead as written. The strong twins are name and word detectors; non-meme units matched for word-likeness twin just as well (0.86 against 0.82); with names removed the same twin pairs fall to r = 0.14, below the word-matched null of 0.56 (`results/robustness/nulls.json`, `f2_agreement.sets.top33`).
- Three bugs or artifacts were of the same kind: a comparison whose answer was fixed by how it was built (two placebo targets scored as perfect because they had nothing to be compared with; two "1% of ordinary films" style comparisons where the films set their own bar; a binomial "230 by chance" that assumed independent works). The first fact-check pass (396 claims checked, 24 flagged; `results/factcheck.json`) caught none of them.
- Still open: no human anywhere in the loop (rewriter, labeller, reviewers, judges and fact-checker are all AI), no steering, several pre-registered checks never run, pre-registration timing cannot be proven from git, and length effects were never controlled.

## Findings

### 1. What the two commits did

`7e8c4c5` (20 Oct 2026, 21:52): 14 files, 21,299 insertions. `experiments.py` +305 lines. Commit message claims "Primary numbers are unchanged" and that is true: I diffed `results/experiments.json` before and after. E2 share, null mean and Monte Carlo p are identical in both models; E1 AUCs identical; E3 rho identical. What changed:
- E2 placebo: `placebo_mean` 0.378 to 0.462 (Qwen) and 0.385 to 0.471 (Llama); `placebo_min` 0.0 to 0.31/0.34; `real_below_all_placebos` False to True. Cause: the 1943 serial and Batman '66 predate every comic, had zero eligible comics, and scored 0 = "perfect match".
- E1: `core_present_share_in_films` (0.012) deleted, replaced by `core_present_share_in_held_out_films` (0.052/0.055) with `held_out_film_chance_share` 0.057. Both versions are fixed by construction.
- E3: `baseline_film_placebo` (rho 0.29/0.26) deleted, replaced by `held_out_film_placebo` (rho -0.09/0.38), also noise around a constant by construction.
- Added: `lexical_comparators` (E1 TF-IDF centroid, E2 TF-IDF cosine, E2 word presence), `E2.residual_test_beyond_words`, `E2.*.chance` (exact, time-consistent, within-era, eligible-uniform), `E3_power_simulated`.
- New scripts: `robustness/nulls.py` (631 lines), `robustness/e2_multiverse.py` (712), `robustness/e2_reimpl.py` (167). `robustness/e1_words.py` came in `6db7807`.

`6db7807` (22:30, 38 minutes later): `paper/paper.md` 158 changed lines, README 78, companion page 139 (git show --stat). Rewrote abstract, results, discussion, limitations; added Sections 2.6 and 3.8, Table 2, "Deviations from the plan", and prior-work citations.

### 2. Claims killed or corrected (first draft to v1), each with the committed evidence

| # | First-draft claim | Fate | Evidence file |
|---|---|---|---|
| 1 | "14,843 meme families present in Batman far beyond chance"; 18,485 candidate units "against about 230 expected by chance" | Killed. Binomial chance assumed independent works. Against random 32-film pseudo-franchises Batman has only 1.8x (Qwen) and 2.4x (Llama) as many units in 6+ works. | `results/robustness/nulls.json` `f1_chance` (53,985 vs 29,400 mean; 71,934 vs 29,372) |
| 2 | (implicit) the core canon is special | Survives, strengthened. No pseudo-franchise has a single unit in 23+ of 32 films in 50 draws; Batman has 1,534 (Qwen) and 1,914 (Llama). | same, `present_in_ge23_of_32` |
| 3 | "The strongest memes are model-independent" (median twin r = 0.82) | Killed as a claim about memes. 24 of 33 top units are name/place detectors, 3 common-word, 4 concepts. One word reproduces their profile at r = 0.85. Word-matched non-meme units twin at 0.86. Meme advantage over word-matched null: median +0.05, 25 of 33 wins, p = 0.007; for the 24 names alone p = 0.07. | `nulls.json` `f2_agreement.sets.top33`, `top33_name` |
| 4 | Twins persist | Killed on name-free text: same twin pairs fall to 0.14, vs 0.33 sparsity-matched, 0.56 word-matched (memes lose, p = 0.012). Re-searching for a fresh twin on name-free text gives 0.54 vs 0.69 for word-matched. | same, `r_anon_fixed_twin`, `r_anon_fresh_search` |
| 5 | "87% agreement on the meme grid against 52% by chance" | Corrected. Rare random units agree on 89 to 92% of cells because most cells are empty. Kappa is the fair figure: 0.73 memes vs 0.35 to 0.57 nulls. | same, `named31.grid_orig` |
| 6 | "The two models' family trees correlate at 0.98 ... still 0.98 after controlling for length" | Gutted. Trees from random non-meme units correlate at 0.90 to 0.93; TF-IDF vs LSA 0.92. The "after controlling for length" figure had no script. Now: 0.976 with differences, 0.895 with sums added; partial given TF-IDF 0.92 vs random-unit floor 0.82 to 0.88. | `nulls.json` `f2_trees` |
| 7 | (not disclosed in first draft) | New admission: the trees do not recover documented sources. Mean rank 5.4 (Qwen), 5.3 (Llama) vs 5.8 chance, p = 0.40/0.36. TF-IDF 2.9 (p = 0.003). Every draw of random non-meme units beats the memes. E2's band and hub adjustment were designed after this failure. | `nulls.json` `f2_trees.source_recovery` |
| 8 | "Some memes respond to ideas rather than words": 11 of 31 (Qwen), 9 of 31 (Llama) survive rewording | Cut down. Another meme's rewording passes the bar 16%/18% of the time, so 5.0/5.7 survive by chance (p = 0.002 Qwen, 0.076 Llama). Only 2 Qwen units (funeral, Two-Face's coin) and 6 Llama units prefer their own rewording to all 30 others. Median rank of own rewording: 11 and 12 of 31. | `nulls.json` `f3_wording`; raw in `results/wording.json` |
| 9 | Llama's revenge unit shows a concept | Kept with a disclosure: it was picked from 50 candidates. Passes harm (AUC 0.87) and vow (0.79) controls. Qwen's best unit is at chance (4 of 12, AUC 0.45; fires 93.8 pct on "finish the race his father never got to run"). The two models' revenge units correlate 0.39 across the corpus. | `nulls.json` `f3_wording.*.revenge`, `revenge_units_corpus_r`; `wording.json` `revenge` |
| 10 | "Spider-Man, Daredevil and Superman carry Batman's canon more strongly than most Batman stories" | Demoted to noisy single-summary point estimates caused by a presence bar set by films with no superheroes. Under TF-IDF no hero film out-scores more than half the Batman works. | `experiments.json` `lexical_comparators.E1_tfidf_centroid.A_name_free.per_cousin_beaten_by` |
| 11 | Name-free core "present in 91% of hero cousins and 1% of ordinary films" | The 1% was true by construction (films set their own bar; at most 1 of 69 passes). Removed. | diff of `experiments.py` in `7e8c4c5` (comment in `e1`) |
| 12 | E1 null means Batman = superhero template | Reframed: "the null comes from the scoring rule". TF-IDF centroid separates Batman from hero films at AUC 0.80 (CI 0.64 to 0.93). | `experiments.json` `lexical_comparators.E1_tfidf_centroid` |
| 13 | E2 "placebo targets averaged 38% and 39%" | Bug. Corrected to 46%/47% over 9 valid targets. As pre-registered (all 11) the placebo criterion fails (5th percentile 0). Reported as a deviation. | `results/robustness/e2_reimpl.json` `placebo`; `e2_nulls.json` `placebo.time_rule` (`mean_as_coded` vs `mean_valid`) |
| 14 | E2 p = 0.004 and 0.0002 | Replaced by exact p over all 3,991,680 relabellings: 0.0026 and 0.00017. Chance of 47% shown to be generous: 42% time-consistent, 44% eligible-uniform, 39%/38% within-era (p = 0.026 Qwen, 0.0014 Llama). | `e2_reimpl.json` `exact_null`; `experiments.json` `E2.primary.chance` |
| 15 | "The effect survived every check the reviewers asked for" | False at the time. The pre-registered word comparators and residual test had not been run. Run late: TF-IDF 11% (exact p 0.0003), word presence 17% (0.0018), both pass the sequel control 6 of 7. | `experiments.json` `lexical_comparators` |
| 16 | "Neuron-defined memes make cultural heredity measurable" (as a neuron result) | Replaced by "On this corpus they trace it about as well as word counts do." Residual test: Qwen fails (0.25, 0.11, 0.089), Llama passes (0.033, 0.014, 0.010). TF-IDF keeps its signal with neurons regressed out (p = 0.0003 in both). | `experiments.json` `E2.residual_test_beyond_words` |
| 17 | "to our knowledge, the first test of vertical transmission ... with units the analyst did not hand-code" | Killed. Prior work added: Barbrook 1998, Chavalarias and Cointet 2013, Kuhn et al. 2014, Chaturvedi et al. 2018, Hatzel and Biemann 2024. | `paper/paper.md` diff |
| 18 | E3 "baseline-film placebo produced a similar coefficient" | Removed as fixed by construction. Added: power 27% at rho 0.3, 54% at rho 0.5; original-text rho negative (-0.32, -0.50). | `experiments.json` `E3_power_simulated`, `E3.original_text` |
| 19 | Designs revised "before any experimental result was computed" | Softened to "before the experiments were run". One design reviewer had already probed original-text E2 on Qwen3-1.7B (9 of 9, p = 0.0008). README adds that git history cannot prove the order. | `results/design_reviews.json` E2 flaw 2; `README.md` diff |
| 20 | Rewriter damage "makes E1 harder for Batman rather than easier" | Withdrawn: "we make no claim about the direction of any bias". | `paper/paper.md` diff, limitations |
| 21 | Labels "checked by a second, sceptical reader" | Supplemented by a blind two-choice test: 56 of 71 (79%, CI 68 to 88%). Word-overlap judge 65%. For the 31 named memes: 24 of 31 vs 63%, p = 0.065, "not clearly better than their words". | `results/robustness/label_test.json`, `nulls.json` `label_test` |
| 22 | "the share of Batman's core carried rises with date" among cousins | Corrected: The Shadow (1994) 25%, Green Hornet (2011) 18%. | `paper/paper.md` diff |

### 3. Null models and controls actually used

Phase 1 (descriptive findings), all in `robustness/nulls.py`, outputs in `results/robustness/nulls.json`:
- Pseudo-franchise null: 50 draws; 32 ordinary films play the franchise, the other 37 set the presence bar, the same bar is applied to Batman. Replaces the binomial.
- Twin correlation: (a) sentence shuffle (0.14); (b) 30,000 random active non-meme Qwen units matched on sparsity; (c) matched on sparsity and layer; (d) matched on "best single word r" (lexical null). Paired Wilcoxon per meme. Then the same fixed twin pairs on name-free texts, and a fresh twin search on name-free texts.
- Grid agreement: Cohen's kappa against the same three unit nulls.
- Trees: Mantel permutation (p floor 1/2001); Mantel floor from 20 draws of random non-candidate and non-representative units; partial Mantel given TF-IDF distance, LSA distance, length and meme-count differences, and differences plus sums.
- Source recovery by tree distance: exact p; TF-IDF and LSA comparators; random-unit draws.
- Rewording: original bar = fire above the max of 10 everyday sentences. New null = each unit scored on another meme's rewording, 10,000 derangements. Plus "own rewording ranked first of 31".
- Revenge: 12 minimal pairs, harm-without-retaliation and vow-without-revenge controls.
- Labels: blind two-alternative forced choice; exact binomial CI; word-stem-overlap judge as rival.

E1 (`experiments.py`): leave-one-out canon; 20 split-half film bars (negatives never set the bar); bootstrap CI over rows and columns; label-shuffle null (32 of 40 lineage works play the franchise, 200 draws; 95th percentile 0.70/0.67); canon from name-free texts; iconography sentences dropped (68 of 3,700); sentence count alone (AUC 0.756, `E1.length_only_auc_vs_hero`); TF-IDF centroid; core size vs pseudo-franchise and vs cousins.

E2 (`experiments.py`, `robustness/e2_reimpl.py`, `robustness/e2_multiverse.py`):
- Joint relabelling of comic identities: 10,000 Monte Carlo, then exact over 12!/5! = 3,991,680 injective maps.
- Time-consistent relabelling (483,840 maps), within-era relabelling (1,440 maps; eras up to 1989, 1990s, 2000+), eligible-uniform expectation.
- Source-set swap among the 9 adaptations (9! = 362,880), time-consistent swap (241,920), swap within medium (5,040), random screen targets (5,000 draws).
- Placebo targets: 9 valid non-adaptation screen works.
- Sequel positive control (prequel in top 2 for 5 of 7).
- Multiverse: 2 models x 2 texts x top-k {1,3,5} x 7 presence bars x 3 band floors x 5 band ceilings x hub on/off x time rule on/off x 3 source rules x dedup on/off = 30,240, each against 5 nulls.
- Jackknife: leave out every adaptation (9), pair (36), triple (84), comic (12); named subsets.
- Alternative source lists (`fuller`, `wiki_full`, `wiki_strict`), source rules, expectation-corrected rate, rewrite-batch check (within-batch rate 0.069 vs across 0.074 Qwen: no batch effect; `e2_jackknife.json` `batch_check`).
- Word comparators through the identical statistic; residual test (OLS over 32 x 12 pairs, statistic rerun on residuals, both directions).
- Independent re-implementation: all 9 per-adaptation shares match in 4 cells (`e2_reimpl.json` `vs_experiments_json.rows_match`).

E3: permutation within medium (10,000); residualisation on year, length, centrality; reverse-time placebo; simulated power.

### 4. What the reviewers attacked

First round, before the experiments (`results/design_reviews.json`; three AI reviewers, one per experiment, each "sound with fixes"):
- E1, 9 flaws, 1 fatal. Fatal: ordinary films used three times (set the bar, select the canon, serve as negatives), so the AUC was inflated and its null was not 0.5. Major: unmatched fold canon; no pipeline null; genre confound and the cousin test as the real test; residual iconography in rewrites; minor: dead name units in the denominator, length (top-3 mean is a max-type statistic that grows with sentence count), forking paths, no lexical baseline.
- E2, 12 flaws. "7 of 9" bar had P = 0.26 under the null; original-text result already known from a probe; rewriter placeholder vocabulary and batch-mates; shared Wikipedia wording ("TF-IDF already put the source in the top 3 for 6 of 9"); hub comics; no placebo targets; no power check without names; band chosen after a null; time direction; non-independent sources (7 distinct); thresholds; weak attributions.
- E3, 11 flaws, 1 fatal. Fatal: popularity proxies later adaptation, so E3 would restate E2 through fame. Also medium confound, wrong split counts (15/17, not 16/16), about 15 real observations, era, length, name recurrence, prototypicality, forking paths, pageview hygiene, no placebo outcome.

The first-round reviewers asked for the lexical baselines (E1 flaw 9, E2 flaw 4). They were pre-registered and then not run. One reviewer note explains the pressure: the TF-IDF control was to be "kept off the main display, since the user asked to drop the word-counting material".

Fact-check pass on the first draft (`results/factcheck.json`): 396 claims checked across 6 parts, 24 flagged, 23 confirmed. All were surface errors: wrong counts (the "319" figure for "the vigilante" has no script; true counts 276/366), a wrong clade, wrong page citations, rounding, "no words in common" when 3 of 31 rewordings share a word, an overclaim about the family resemblance. It missed every substantive problem.

Second round, after the first write-up (the "robustness review"; its working notes are not in the repo, only the scripts and outputs it left): it re-ran the headline numbers and found the placebo bug, the two by-construction comparisons, the unrun pre-registered controls, the binomial chance figure, the word-likeness of the twins, the lenient rewording bar, the high Mantel floor, a number with no script, and the era structure in the E2 null. README: "Numbers from the review's working notes that no committed script produces are not cited."

Rewriter self-audit (`results/name_free_rewrite_notes.json`): 12 batches, 114 works, 86 logged fixes. Typical: the rewriter had added information not in the source ("sidekick", "famous", "neighboring", "an act named after the district attorney", "now named after the hero's home planet").

Label re-read (`results/labels.json` vs `results/labels_wf.json`): of the 26 named memes that appear in both files, 5 were renamed by the second reader (for example "Superhero powers (peaks on Superman)" became "Superman lore": "no top sentence is about powers"). Of 120 re-read labels, 24 are low confidence and 19 are kind "mixed" (one neuron splits "exactly four and four" between Arkham Asylum and Lucius Fox).

### 5. Weaknesses still open (my reading; none of these is fixed in v1)

A. The "beyond words" evidence is one fragile test.
- Llama residual p: 0.033 / 0.014 / 0.010 under the joint null. Under time-consistent: 0.057 / 0.021 / 0.017. Under within-era: 0.065 / 0.019 / 0.017 (`results/experiments.json`, `llama3.1-8b.E2.residual_test_beyond_words.*.chance`). The paper's Table 2 leaves the within-era column blank for the residual rows.
- Six neuron residual tests (2 models x 3 covariate sets), no multiplicity correction. The pre-registered falsifier named TF-IDF or word presence; the best p (0.010) is the "both" version.
- The residual test was never run through the multiverse or the jackknife. Those cover only the neuron-alone statistic, which the word comparators also pass. So "99.6% of 30,240 specifications" says nothing about P1.
- OLS on a noisy covariate under-corrects. R2 of the fit is 0.08 to 0.21, so word similarity as measured explains little of neuron inheritance; some of what is left can be wording that TF-IDF measures badly (synonyms, inflections, phrases). No stronger text rival was tried: no BM25, no n-grams, no sentence embedding, no LLM reader.
- n = 9 adaptations over 7 distinct sources.

B. The primary model fails its own positive control and its beyond-words test. Qwen name-free sequel control: 4 of 7 (bar 5). It passes in only 48% of Qwen name-free specifications, 82% of Llama's (`e2_multiverse.json` `by_cell.*.frac_sequel_control_passes`). TF-IDF and word presence pass 6 of 7.

C. Length is an uncontrolled confound in at least three places.
- Sequel control: on name-free text the closest earlier screen work is the 13-sentence 1943 serial for 3 of 7 sequels in both models (`e2_nulls.json` `sequel_control.paper_rule.rows`). Drop the serial and Qwen passes 5 of 7 (`without_1943_serial`). The inheritance rate rewards short candidates.
- E1: sentence count alone gives AUC 0.756 against the hero films. The TF-IDF centroid gives 0.80 with CI 0.64 to 0.93. So "word counts separate Batman from the hero films" is not shown to exceed "Batman summaries are longer".
- The pre-registered truncation checks (13 and 23 sentences) were never run.

D. The E1 TF-IDF "crime city" reading is generous. The top 30 separating words (`results/robustness/e1_tfidf_words.json`) include asylum, commissioner, face, district, game, characters. Those look like leftover iconography (Arkham Asylum, Commissioner Gordon, Two-Face, district attorney) and video-game or Wikipedia register, not only a crime-city theme. E1 has 8 negatives.

E. Label test is weaker than "79% vs 65%" suggests.
- Item-level judge answers were not kept, so no paired test.
- The word-overlap judge ties on 48 of 71 items and is right on 22 of the 23 it can decide (0.6479 x 71 = 46 = 24 from ties + 22). So where a label shares a stem with a sentence, words settle it; elsewhere the rival is a coin. A fairer rival (embedding similarity of label to sentence, or an LLM that never sees activations) was not tried.
- Controls were sentences at or below the unit's median activation. That tests recall on mid-rank activating sentences against easy negatives. It does not test specificity against hard negatives (same words, different idea).
- No steering or ablation. All judges are AI.

F. Cross-model agreement (P2) has no held-out test. Twins are chosen as the best-correlated of about 900k Llama units on the same 3,700 sentences that then report r. The nulls go through the same search, so relative comparisons are fair, but the absolute 0.82 is a selected maximum. The one surviving concept result (27 concept units, 0.62 vs 0.46 word-matched, p = 0.001) disappears on name-free text (0.34 vs 0.38; `f2_agreement.sets.labelled_concept`). Class labels for the 33 are "the review's reading", one AI reader.

G. Family counts are not stable objects. 14,843 (Qwen3-8B), 24,732 (Llama 3.1 8B), 5,641 (Qwen3-1.7B) from the same procedure (`nulls.json` `f1_chance.n_families`).

H. Rewording test design. n = 31, one rewording each, written by an AI that knew the label; bar = 10 everyday sentences. Even the best survivor is near-lexical: the paper notes the funeral unit tracks the word "funeral" at r = 0.98 and its rewording uses mourners, grave, coffin.

I. Qwen is fragile under the within-era null. 9 of 36 pairs, 37 of 84 triples and 2 of 12 single-comic leave-outs give p >= 0.05; largest 0.13 without The Dark Knight Returns; without the Killing Joke trio p = 0.083; strong-attribution subset p = 0.071 (`e2_jackknife.json` `summary`, `named_subsets`). Era alone accounts for 27% (Qwen) and 23% (Llama) of the gap between the observed share and the paper's 47% chance level (`e2_nulls.json` `era_share_of_paper_gap`).

J. Forking paths remain. E2's band and hub adjustment were chosen after the tree test failed. All 132 multiverse failures have the hub adjustment off (111 of them Qwen). The multiverse varies analysis choices only. It does not vary the corpus, the 40-sentence cap, the rewrite, the reference film pool or the adaptation list, and its 30,240 cells share the same 9 adaptations, so they are not independent evidence.

K. Provenance. "The files cannot prove that the designs came before the first results, because the git history starts after both were written" (README). One design reviewer had probed the data.

L. Pre-registered checks never run (paper, "Deviations from the plan"): Qwen3-1.7B replication of E2; band check on the original meme families; fixed-length truncation; Holm correction for E3; placeholder-unit exclusion. I also found no substitution-count AUC and no medium-matched E1 in `experiments.py`.

M. Rewrite quality unmeasured. 86 logged fixes show the rewriter injects content. The diff that inserted X-codes "sometimes swallowed a neighbouring word" and no committed code measures that by group.

N. E3 is untested, not null. n = 15, power 27% at rho 0.3. Popularity is 2015 to 2026 pageviews. Its replacement placebo is uninformative by construction.

O. One franchise, one text source (2020s Wikipedia plot summaries, which share editors and phrasing), two 8B models at 4 bits. Every reviewer, labeller, judge, rewriter and fact-checker was an AI, mostly the same model family, so their blind spots are correlated.

### 6. Methodological lessons every future experiment must respect

1. Run the cheap rival first, through the identical statistic, null and controls, and put it in the headline table. E2's rival was pre-registered and skipped; when run it matched the neurons.
2. State the claim as "beyond the rival" and test that directly (residual or paired comparison). A neuron result that a word count also produces is a result about the texts.
3. Check every comparison for "true by construction" before reading it. Ask: could this number have come out differently? A reference set must never set its own bar and then be scored against it. A placebo with nothing to compare against must be dropped, not scored.
4. Use an empirical null built from the same pipeline (pseudo-franchises, random units, mismatched pairs), not an analytic one that assumes independence. Works in a franchise are not independent; units are not independent.
5. Match the null to the unit's surface properties. For any neuron claim, compare with non-selected units matched on sparsity, layer and word-likeness. If the word-matched null does as well, the finding is lexical.
6. Report the floor for every similarity statistic. Mantel 0.98 meant nothing once random units gave 0.90 to 0.93. Report kappa, not raw agreement, on sparse grids.
7. Make the permutation respect the structure that is not the hypothesis: time order, era, medium, batch. Report the strictest defensible null next to the pre-registered one.
8. Compute exact p when the permutation space is enumerable, and always carry the stricter nulls into every secondary test (including the residual test).
9. Run a positive control with the names removed and treat failure as "no power", for the neuron measure and for the rival alike.
10. Control length explicitly. Any max-type presence score grows with sentence count. Fix text length or subsample, and report the AUC of length alone.
11. Disclose selection. A unit picked from 50 candidates, a twin picked from 900k, a band picked after a failed test: each needs held-out data or a selection-aware null.
12. Pre-register in a way that can be proven (timestamped commit or external registry before any activation is extracted), and list every planned check with a "ran / did not run" column in the paper.
13. Put the claim that matters through the multiverse and the jackknife, not only the easy claim. Vary data choices (corpus, rewrite, reference pool), not only analysis knobs.
14. Every reported number needs a committed script and output. A number from a reviewer's notes does not count.
15. Keep item-level records of every judgement so paired tests are possible. Test labels against hard negatives, and validate at least a sample causally (steering or ablation) or with human judges.
16. Do not rely on a fact-check for validity. Checking that the paper matches the code does not check that the code is right. Use an independent re-implementation and an adversarial reviewer who tries to break the result, ideally not the same model family, plus at least one human.
17. Report minimum detectable effect before the test. With 9 or 15 units of analysis, most nulls are "inconclusive".
18. Use a lineage where shared wording cannot carry the signal (paraphrase, translation, loose adaptation), or the rival wins by default.

## What this means for P1 and P2

P1 (a meme can be measured by which neurons fire).
- For: units exist, are addressable and nameable; the Batman core is real against a fair null (0 vs 1,534 units in 23+ of 32 works); blind judges match labels to unseen sentences 79% of the time; a handful of units survive rewording (2 Qwen, 6 Llama rank their own rewording first of 31); Llama's revenge unit passes two hard controls; Llama's E2 residual passes under the joint null.
- Against: on the one hereditary test, plain TF-IDF equals or beats the neurons, and passes a positive control Qwen's neurons fail. The primary model adds nothing detectable beyond words. For the 31 hand-named memes, labels are not clearly better than word overlap (p = 0.065). Most strong units are name detectors; 19 of 120 re-read units are "mixed". Family counts swing 5,641 to 24,732 across models.
- Net: P1 is supported in the weak sense (neurons give countable, inspectable units that track content). The strong sense (they measure ideas that words miss) rests on a few units and one uncorrected test in the replication model. Conditions under which it has held so far: original text with names present; strongest units; Llama rather than Qwen.

P2 (the measurement generalises because models converge on the same units).
- For: top-33 twins correlate 0.82; concept units beat a word-matched null on original text (0.62 vs 0.46, p = 0.001); four rewording survivors are shared by both models.
- Against: the agreement is carried by names and words. Word-matched non-meme units twin as well (0.86). On name-free text the fixed twins fall to 0.14, below random units, and concept twins lose their edge (0.34 vs 0.38). Across all 14,843 families the median twin r is 0.40 against 0.36 for non-meme units (paper, from the first analysis). Revenge units correlate 0.39 across models, and only one of the two works. Tree agreement of 0.98 is mostly a property of any two text representations.
- Net: at the single-neuron level, in 8B models, on this corpus, the study's own controls cut against P2. What converges is lexical detection. No mapping between models (as in the universal-geometry line) was tested; only nearest-neuron matching.

## Experiment seeds

1. **Residual-test multiverse with stronger text rivals.**
   Question: does Llama's "beyond words" signal on E2 survive the same multiverse, jackknife and within-era null as the headline, and a stronger lexical covariate set?
   Data: existing activations and name-free texts; no new extraction.
   Method: for each of the 30,240 specifications, regress neuron inheritance on TF-IDF, word presence, BM25, character n-grams and a sentence-embedding cosine; rerun the statistic on residuals under all five nulls; Holm across models.
   Rival to beat: TF-IDF, BM25, n-gram overlap, sentence embeddings, and an LLM asked to rank the comics by reading the texts.
   Falsifier: residual p < 0.05 under the within-era null in fewer than half of specifications, or the signal vanishes once sentence embeddings are in the covariate set.
   Cost: CPU hours; one day of work.

2. **Pipeline false-positive calibration on pseudo-lineages.**
   Question: how often does the full E2 pipeline (band, hub, time rule, best-of-sources) declare heredity where none exists?
   Data: the 69 ordinary films plus a larger pool of unrelated film summaries; fabricate "adaptation of" links matched on era, genre and length.
   Method: run the unchanged pipeline on 1,000 pseudo-lineages for neurons and for TF-IDF.
   Rival: TF-IDF through the same pipeline.
   Falsifier: more than 5% of pseudo-lineages reach p < 0.05, which would mean the published p values are miscalibrated.
   Cost: hours of extraction on new summaries, then CPU.

3. **Length-controlled rerun.**
   Question: do E1, E2 and the sequel control survive when every work is cut to the same number of sentences?
   Data: existing per-sentence activations.
   Method: subsample each work to 13 sentences, 20 repeats; rerun E1 (neurons and TF-IDF), E2, sequel control; report AUC of length alone each time.
   Rival: sentence count alone; TF-IDF on the same subsamples.
   Falsifier: E1 TF-IDF falls to the length-only AUC; or the sequel control and E2 lose significance.
   Cost: CPU minutes to hours.

4. **Held-out twin test by unit class (P2).**
   Question: do Llama twins found on half the sentences still track their Qwen unit on the other half, on paraphrases and on a fresh corpus, once word-likeness is matched?
   Data: existing activations split by work; plus a new non-Batman corpus and paraphrased sentences.
   Method: choose twins on split A, score r on split B; compare with sparsity-, layer- and word-matched non-meme units; classes assigned by two independent readers with agreement reported.
   Rival: a bag-of-words regression that predicts the Qwen unit's profile from words (if it predicts Llama's twin as well as Qwen's unit does, the twin is lexical); sentence-embedding linear probe.
   Falsifier: concept units' held-out twin r is no higher than the word-matched null on paraphrased or name-free text.
   Cost: one extraction run on a new corpus; a day.

5. **Label specificity with hard negatives and kept records.**
   Question: do the labels say what the unit detects, or only which words it likes?
   Data: the 151 labelled units; for each, sentences that (a) express the idea without its words, (b) use its words without the idea, (c) neither.
   Method: predict activation rank from the label alone (simulation scoring) and run 2AFC with decoys; store every item-level answer; include human judges on a sample; steer the top 20 units.
   Rival: stem overlap; embedding similarity of label to sentence; an LLM reading the sentence for the label with no neuron involved.
   Falsifier: on (a) vs (b) items the unit sides with the words more often than with the idea; or judges do not beat the embedding rival on a paired McNemar test.
   Cost: a few hundred written sentences, one extraction run, modest human time.

6. **Independent red-team replication.**
   Question: does a reviewer from a different model family, and one human statistician, find defects the same-family review missed?
   Data: the frozen v1 repo.
   Method: give each the paper and code with a bounty per confirmed defect; compare the defect lists with `factcheck.json` and the robustness review.
   Rival: the existing AI fact-check (396 claims, 24 flags) as the baseline detector.
   Falsifier: no new substantive defect found, which would support the current process.
   Cost: a few days of reviewer time.

## Key sources

Internal (all in `study/`):
- `results/robustness/nulls.json` (and `robustness/nulls.py`). Stronger nulls for the canon, twins, trees, rewording, labels. Priority A.
- `results/experiments.json` (`lexical_comparators`, `E2.residual_test_beyond_words`, `E2.primary.chance`, `E3_power_simulated`). The words-versus-neurons evidence. Priority A.
- `results/robustness/e2_multiverse.json`, `e2_jackknife.json`, `e2_nulls.json` (and `robustness/e2_multiverse.py`). Specification curve, leave-outs, alternative nulls. Priority A.
- `results/design_reviews.json`. The 32 flaws three reviewers raised before the experiments. Priority A.
- `results/robustness/e2_reimpl.json`. Independent rebuild and the placebo bug. Priority B.
- `results/robustness/label_test.json`. 71 blind items and tallies. Priority B.
- `results/factcheck.json`. What a surface fact-check catches and misses. Priority B.
- `results/wording.json`, `results/labels.json`, `results/labels_wf.json`, `results/name_free_rewrite_notes.json`, `results/robustness/e1_tfidf_words.json`. Priority C.
- Commits `7e8c4c5` and `6db7807` (git show). Priority A for the before/after.

Literature (each confirmed to exist this session via Crossref, arXiv API or the publisher page):
- Steegen, Tuerlinckx, Gelman, Vanpaemel 2016, "Increasing Transparency Through a Multiverse Analysis", Perspectives on Psychological Science. doi:10.1177/1745691616658637. The method behind the 30,240-specification check. Priority B.
- Simonsohn, Simmons, Nelson 2020, "Specification curve analysis", Nature Human Behaviour. doi:10.1038/s41562-020-0912-z. How to do inference across specifications that share data. Priority B.
- Simmons, Nelson, Simonsohn 2011, "False-Positive Psychology", Psychological Science. doi:10.1177/0956797611417632. Forking paths; relevant to E2's post-hoc band and hub choice. Priority C.
- Guillot, Rousset 2013, "Dismantling the Mantel tests", Methods in Ecology and Evolution. doi:10.1111/2041-210x.12018. Why Mantel and partial Mantel p values are unreliable with autocorrelated data; bears on the tree-agreement claim. Priority B.
- Bolukbasi et al. 2021, "An Interpretability Illusion for BERT", arXiv:2104.07143. Neurons that look like one concept on one corpus and another elsewhere; the risk in naming units from top sentences of a single corpus. Priority A.
- Huang, Geiger, D'Oosterlinck, Wu et al. 2023, "Rigorously Assessing Natural Language Explanations of Neurons", arXiv:2309.10312. Tests neuron labels for precision as well as recall; the template for seed 5. Priority A.
- Gurnee et al. 2024, "Universal Neurons in GPT2 Language Models", arXiv:2401.12181. P2 source. My recollection that only a small share of neurons are universal was not rechecked this session [UNVERIFIED number]. Priority A.
- Huh, Cheung, Wang, Isola 2024, "The Platonic Representation Hypothesis", arXiv:2405.07987. P2 source; about representation geometry, not single neurons. Priority B.
- Chaturvedi, Srivastava, Roth 2018, "Where Have I Heard This Story Before? Identifying Narrative Similarity in Movie Remakes", NAACL. doi:10.18653/v1/N18-2106. Prior art for matching remakes from plot summaries. Priority B.
- Hatzel and Biemann 2024, "Tell me again! A large-scale dataset of multiple summaries for the same story", LREC-COLING. https://aclanthology.org/2024.lrec-main.1366 (page returns 200; title as cited in the paper). Name-replaced summaries are much harder to match; a ready-made benchmark where word overlap is weakened. Priority A.

## Fog

- The robustness review's own working notes, prompts and reviewer identities are not in the repo. Only its scripts and outputs are. I cannot say how many reviewers there were or what they tried that left no script.
- I could not check whether the "review's reading" of the 33 top units as name/word/concept (hard-coded as `TOP33_CLASS` in `robustness/nulls.py`) was made blind to the result.
- Whether the Llama residual signal survives a multiverse is unknown. Nobody ran it.
- The funeral unit's r = 0.98 with the word "funeral" is stated in the paper; I did not trace it to a specific JSON key (likely `f2_agreement.per_meme[*].best_word_r`).
- Whether pre-registration preceded the first results cannot be established from the repo.
- How much the 4-bit quantisation and the 8B size limit unit quality relative to the 32B model in the "already clean" source is untested here.
- I did not read `e2_multiverse.py` and `nulls.py` line by line for bugs. The independent rebuild covers only the E2 primary statistic, so the Phase 1 nulls in `nulls.py` have had no independent re-implementation.
- The first-draft median twin r of 0.40 vs 0.36 over all families is quoted from the paper; I did not find it in `results/robustness/`.
