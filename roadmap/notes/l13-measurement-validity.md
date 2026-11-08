# l13 - Measurement validity: what would make "neurons that fire" a valid measurement of a meme?

Scope: how new measurements earned trust (temperature, psychological constructs, genes, species, microbial units), then a validation ladder for the neuron-meme instrument. No experiments were run. Study numbers name their committed file and were read from it this session.

How far to trust the literature claims: every source cited was confirmed to exist this session (Crossref, the arXiv API or web search). I did not re-read the papers. What each one says is from my memory of it unless the text says it was checked online. Specific figures and stories I am relying on memory for are marked "[recalled]"; anything I could not confirm at all is marked "[UNVERIFIED]".

Checker pass, 2026-11-08: every DOI and arXiv ID below was re-resolved (Crossref, arXiv abstract pages) and the study numbers were re-read from the committed files. Corrections are marked "[checked]" or "[corrected]" in place and listed in the "Check log" at the end.

## Bottom line

- **No measurement in history started with ground truth, so the lack of an agreed definition of "meme" is not the obstacle.** Temperature, genes, species and microbial taxa all began with an operational unit plus a decidable test, and were corrected by iteration (Chang 2004; Gerstein et al. 2007; de Queiroz 2007; Stackebrandt and Goebel 1994). The obstacle is that the neuron instrument has not yet passed the tests those fields used.
- **Scored against a standard validation ladder, the Batman instrument currently fails or is untested on most rungs.** Paraphrase reliability: 11 of 31 named units survive rewording in Qwen against about 5 by chance (`study/results/robustness/nulls.json`, `f3_wording`). Discriminant validity against words: fails in Qwen, weak pass in Llama (`study/results/experiments.json`, `E2.residual_test_beyond_words`). Human convergent validity and steering: never tested.
- **Premise P2 is the first validity test, not an optional extra.** Chang shows that the first test available without ground truth is comparability: instruments of the same kind must agree with each other. Regnault rejected the mercury thermometer on that test [recalled]. The study's strongest units fail it once names are removed: twin correlation falls from 0.82 to 0.14 (`nulls.json`, `f2_agreement.sets.top33`).
- **The discriminant rung is the kill rung.** In the study, TF-IDF word overlap still finds the sources after the neurons are partialled out (p = 0.0003 to 0.003 in both models; [corrected: this range is the TF-IDF comparator only; the word-presence comparator gives p = 0.0014 to 0.0035 in Qwen and 0.028 to 0.059 in Llama]), while Qwen's neurons find nothing after words are partialled out (p = 0.09 to 0.37) (`experiments.json`, `E2.residual_test_beyond_words`). Campbell and Fiske (1959) call this shared method variance. Until a unit beats sentence embeddings and an LLM reader on text where the words are controlled, nothing higher on the ladder matters for P1.
- **A raw neuron in a public checkpoint has one real advantage as a unit: it is a fixed, citable label.** Sparse-autoencoder features change with the random seed and the dictionary size (Paulo and Belrose 2025; Leask et al. 2025). This is the same advantage that made exact sequence variants replace clustered OTUs in microbiology (Callahan et al. 2017). The study's "meme families" give that advantage away, because they are corpus-dependent clusters.
- **Cheapest decisive experiment: a 2x2 of motif present or absent by key words present or absent, on planted stories with frozen units.** It tests paraphrase reliability, discriminant validity and input-side causal validity at once. It is already designed in `preregistration/` (test T2) and has not been run.

## Findings

### A. How other measurements earned trust

**Temperature (Chang 2004, *Inventing Temperature*, doi:10.1093/0195171276.001.0001).**
- The "problem of nomic measurement": to check that a thermometer reads temperature correctly you need to know how its fluid expands with temperature, which needs a thermometer you already trust. The circle has no foundation to stand on. Chang's answer is "epistemic iteration": start from an imperfect standard, respect it, use it to build a better one, and let the better one correct the first (chapter 5; confirmed from the OUP chapter page https://academic.oup.com/book/5530/chapter/148476793). [corrected: the "problem of nomic measurement" is in chapter 2, "Spirit, Air, and Quicksilver" (doi:10.1093/0195171276.003.0002); "epistemic iteration" is in chapter 5, "Measurement, Justification, and Scientific Progress" (doi:10.1093/0195171276.003.0005). Both attributions and the "respects existing traditions while at the same time correcting them" gloss are confirmed by Tal's SEP entry, https://plato.stanford.edu/entries/measurement-science/, which cites "Chang 2004: Ch. 2" and "Ch. 5"; chapter titles confirmed on Crossref. The OUP chapter URL returned HTTP 403 to the checker, so that this URL is chapter 5 is [UNVERIFIED].]
- The stages were: bodily sensation, then thermoscopes that only order things, then numerical thermometers, then an abstract theoretical scale [recalled]. Ordinal agreement came long before any claim about amounts.
- Fixed points (freezing, boiling) were the calibration anchors, and they had to be defended, because boiling point shifted with the vessel and the water (chapter title "The Defense of Fixity", confirmed in the table of contents). [corrected: this is not a chapter title. Chapter 1 is "Keeping the Fixed Points Fixed" (Crossref, doi:10.1093/0195171276.003.0001, pp. 8-56). "The Defense of Fixity" is, as far as the checker recalls, a section heading inside that chapter; the heading itself is [UNVERIFIED].]
- Regnault's comparability test: before asking whether a thermometer type is right, ask whether instances of that type agree with each other. Mercury-in-glass thermometers made of different glass disagreed, so the type was rejected; air thermometers agreed [recalled; the account is in chapter 2, whose title is confirmed, but the Regnault details could not be re-checked online: UNVERIFIED this session]. The test needs no theory and no ground truth.
- Translation: human reader judgement is the "sensation" stage for memes. Planted-motif stories and motif-free controls are the fixed points. Different models, seeds and quantisations are "instances of the same thermometer type". Activation size should not be read as "amount of meme" until present-or-absent ordering is settled.

**Construct validity (Cronbach and Meehl 1955, doi:10.1037/h0040957).**
- Written for exactly this case: no criterion measure defines the construct. Validity is then argued through a "nomological network": the theory says how the construct relates to other observables, and the test is valid to the degree those predicted relations hold. A failed prediction can indict the test, the theory or the experiment, and the authors say you must decide which.
- For memes the network is: fires regardless of wording and language; adaptations carry their source's units; units mutate and vary in how well they spread. The study tested the second link (E2) and it held, but a rival construct, "shared vocabulary", predicts the same result. The network as tested does not separate the two.

**Multitrait-multimethod matrix (Campbell and Fiske 1959, doi:10.1037/h0046016).**
- Measure several traits by several methods. Four requirements: (1) same trait by different methods correlates highly; (2) that correlation exceeds different-trait, different-method correlations; (3) it also exceeds different-trait, same-method correlations; (4) the pattern of trait relations is the same across methods. Failure of (3) means the method, not the trait, drives the scores.
- The study already contains a failure of this kind. Family trees from the two models correlate 0.978, but trees built from random non-meme units correlate 0.90 to 0.93, and neuron trees correlate 0.84 to 0.85 with the TF-IDF tree (`study/results/robustness/nulls.json`, `f2_trees.mantel` and `f2_trees.random_unit_floor`). Almost all the agreement is common to any units read off the same texts.

**Validity as a causal claim (Borsboom, Mellenbergh and van Heerden 2004, doi:10.1037/0033-295X.111.4.1061).**
- A test is valid for an attribute if the attribute exists and variation in it causes variation in the test outcome. This is stricter than correlation with other measures and maps directly onto text interventions: put the motif in, take it out, hold the words as fixed as possible, and see whether the unit moves.
- Messick (1995, doi:10.1037/0003-066X.50.9.741) and Kane (2013, doi:10.1111/jedm.12000): validity belongs to an interpretation or use of a score, not to the instrument. "This unit detects motif M in a passage", "shared units mean shared descent" and "the unit is the meme" are three different claims needing three arguments.
- Freiesleben (2026, arXiv:2603.15121) argues the nomological account fits language-model research better than the causal one, and that interpretability evidence for the causal account is still fragmentary.

**Reliability and invariance.**
- Reliability bounds validity: the correlation between a measure and a criterion cannot exceed the square root of the product of their reliabilities (classical test theory). So human annotator agreement is the ceiling. For folklore motifs that ceiling is modest: about kappa 0.72 overall, 0.85 for simply expressed motifs and 0.53 for complexly expressed ones in Alyami and Finlayson (2026, arXiv:2603.19283) [corrected and checked against Table 4 of the arXiv HTML full text: Cohen's kappa on 5,548 sentences and 60 motifs; the 0.87 first written here is the cell for motifs that are simple in both concept and expression, not all simply expressed motifs], and Fleiss' kappa above 0.55 in GOLEM (Yarlott et al. 2024, https://aclanthology.org/2024.lrec-main.689/; PDF https://aclanthology.org/2024.lrec-main.689.pdf) [checked: the abstract says "Fleiss' kappa of >0.55"; the "to 0.7 across teams" first written here is UNVERIFIED].
- Conventional benchmarks: intraclass correlation below 0.5 poor, 0.5 to 0.75 moderate, 0.75 to 0.9 good, above 0.9 excellent (Koo and Li 2016, doi:10.1016/j.jcm.2016.02.012) [checked: the abstract gives exactly these bands, applied to the 95% confidence interval of the estimate]; kappa 0.61 to 0.80 "substantial" (Landis and Koch 1977, doi:10.2307/2529310) [recalled; paper confirmed on Crossref, Biometrics 33(1):159; the band is consistent with Alyami and Finlayson calling 0.72 "substantial" and 0.53 "moderate" with a citation to it].
- Measurement invariance (Meredith 1993, doi:10.1007/BF02294825; Vandenberg and Lance 2000, doi:10.1177/109442810031002; Putnick and Bornstein 2016, doi:10.1016/j.dr.2016.06.004): configural (the same structure exists in each group), metric (same slopes), scalar (same intercepts, so the same threshold means the same thing), strict. Without scalar invariance, level comparisons between groups are not interpretable. Common pass mark: fit worsens by no more than 0.01 in CFI at each step (Cheung and Rensvold 2002, doi:10.1207/S15328007SEM0902_5).
- This bites the study directly. Its presence bar is set against 69 English Wikipedia film summaries (`study/paper/paper.md`, lines 37 and 47) [corrected: the 69-film pool is stated on line 37, the presence rule on line 47]. Any change of register (older prose, translation, a script instead of a summary) can move baseline activations, so "the 1943 serial lacks meme X" could be a threshold artefact.
- Questionable measurement practices (Flake and Fried 2020, doi:10.1177/2515245920952393): undisclosed flexibility in how a measure is built and scored. The cure is the one the pre-registration already uses: freeze units, thresholds and scoring before test data exist.

**Genes.**
- The gene was an operational unit for a century before anyone agreed what it was. Mendel's factor was defined by a transmission test (ratios in crosses); Morgan's gene by map position; then successively a protein blueprint, a transcribed code, an open reading frame found by a sequence pattern, and an annotated database entity (history in Gerstein et al. 2007, doi:10.1101/gr.6339607, and Portin and Wilkins 2017, doi:10.1534/genetics.116.196956) [stage list recalled].
- Benzer (1955, doi:10.1073/pnas.41.6.344, and his 1957 chapter "The elementary units of heredity" [UNVERIFIED this session]) split "the gene" into three units defined by three tests: the cistron (unit of function, by the cis-trans complementation test), the recon (unit of recombination) and the muton (unit of mutation). They were different sizes. [checked and corrected: the 1957 chapter exists: Benzer, S. (1957), "The elementary units of heredity", in McElroy and Glass (eds.), *The Chemical Basis of Heredity*, Johns Hopkins Press, pp. 70-93 (cited by Wikipedia, https://en.wikipedia.org/wiki/Cistron, which says "cistron" was coined there). The three terms come from the 1957 chapter; the 1955 paper ("Fine structure of a genetic region in bacteriophage") is the mapping work behind them. That recon and muton are in the same chapter, and the size claim, are recalled.]
- After ENCODE, Gerstein et al. proposed "a union of genomic sequences encoding a coherent set of potentially overlapping functional products" [checked: this is the wording in the abstract]. Griffiths and Stotz (2006, doi:10.1007/s11017-006-9020-y) describe several gene concepts in use side by side.
- Lesson: expect "the meme" to split the same way. The unit of detection (a neuron), the unit of transmission (what is copied between works), the unit of selection and the unit of function (what steering changes) need not coincide, and each needs its own test.

**Species.**
- Biology has more than twenty species concepts [checked: Mayden (1997) recorded about 24 and the philosopher John Wilkins counted 26, per https://en.wikipedia.org/wiki/Species; neither primary source was opened]. De Queiroz (2007, doi:10.1080/10635150701701083) separates the concept (a separately evolving lineage) from the operational criteria (interbreeding, diagnosability, monophyly), which become lines of evidence that are expected to disagree in the grey zone. Hey (2001, doi:10.1016/S0169-5347(01)02145-0) is the sceptical view that the disagreement is partly in our heads.
- Lesson: "a neuron fires" should be one operational criterion for a meme among several (keyword present, human codes it, LLM reader says yes). The scientific object is the agreement and disagreement between criteria, not the crowning of one.

**Microbial units: OTUs, 16S, barcodes.**
- "Operational taxonomic unit" was coined to be deliberately agnostic about what the unit really is (Sokal and Sneath 1963, *Principles of Numerical Taxonomy*; existence confirmed through its 1964 review in Science, doi:10.1126/science.144.3619.712 [checked: the review is by G. G. Simpson, Science 144(3619):712-713]).
- The 97% 16S threshold was calibrated against the previously accepted standard, 70% DNA-DNA reassociation, and only as a one-way rule: below 97% means different species; above it proves nothing (Stackebrandt and Goebel 1994, doi:10.1099/00207713-44-4-846). This is Chang's "respect the earlier standard" in practice. [checked: the paper exists, Int. J. Syst. Bacteriol. 44(4):846-849, and its title names DNA-DNA reassociation and 16S analysis; Edgar's abstract confirms the 97% threshold "was proposed in 1994". The 70% figure and the one-way reading are recalled; no abstract could be retrieved.]
- The threshold was arbitrary and later revised upward (Edgar 2018, doi:10.1093/bioinformatics/bty113) [checked: the abstract gives optimal thresholds of about 99% for full-length sequences and about 100% for the V4 region]. Clustered OTUs depend on the dataset, so they cannot be compared across studies; exact sequence variants replaced them because they are consistent labels (Callahan, McMurdie and Holmes 2017, doi:10.1038/ismej.2017.119) [checked: "consistent labels" is the abstract's phrase]. Short 16S regions cannot resolve species at all (Johnson et al. 2019, doi:10.1038/s41467-019-13036-1) [corrected: the abstract's claim is weaker: short-read sequencing of variable regions "cannot achieve the taxonomic resolution afforded by sequencing the entire (~1500 bp) gene"]. Clustering method choices change results (Schloss and Westcott 2011, doi:10.1128/AEM.02810-10).
- DNA barcoding works only where a "barcode gap" exists: variation within a species is smaller than the distance between species (Hebert et al. 2003, doi:10.1098/rspb.2002.2218). [corrected: Hebert et al. 2003 proposed COI barcoding and its abstract does not use the term. The "barcoding gap", and the finding that it often fails (substantial overlap, error of about 17% in incompletely sampled cowries), is Meyer and Paulay 2005, "DNA Barcoding: Error Rates Based on Comprehensive Sampling", PLoS Biology 3(12):e422, doi:10.1371/journal.pbio.0030422.]
- Microbiologists validate pipelines on mock communities of known composition [general practice; no single source checked].
- Lessons: (1) calibrate against the older standard; (2) publish the threshold as provisional; (3) prefer exact, reusable labels to corpus-dependent clusters; (4) state the resolution limit; (5) build a mock community. The study's family counts already show the OTU problem: 14,843, 24,732 or 5,641 families depending on model and thresholds (`nulls.json`, `f1_chance.n_families`) [corrected: in that file the three counts are three models, Qwen3-8B, Llama 3.1 8B and Qwen3-1.7B, under one rule; it does not vary the thresholds].

### B. What interpretability research has learned about its own measurements (2021 to 2026)

- **Sanity checks fail more often than expected.** Saliency maps looked the same on trained and randomised networks (Adebayo et al. 2018, arXiv:1810.03292). Probes can score well on meaningless control labels (Hewitt and Liang 2019, arXiv:1909.03368). Automated interpretability scores do not separate trained from randomly initialised transformers (Heap et al., arXiv:2501.17727, retitled in 2026) [checked: Heap, Lawson, Farnik and Aitchison; current title "Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers", first posted January 2025, revised 27 January 2026; the abstract limits the finding to "many settings"].
- **Labels from top-activating examples mislead.** The same unit looks like a different clean concept on a different corpus (Bolukbasi et al. 2021, arXiv:2104.07143). Natural-language neuron explanations fail when tested as causal claims (Huang et al. 2023, arXiv:2309.10312).
- **Learned features are not stable units.** Sparse autoencoders trained on the same data with different seeds learn different features (Paulo and Belrose 2025, arXiv:2501.16615), and features split or merge as the dictionary grows (Leask et al. 2025, arXiv:2502.04878). Weight regularisation helps (Jedryszek and Crook 2026, arXiv:2603.04198). General features go silent where a specific one takes over ("absorption"; Chanin et al. 2024, arXiv:2409.14507), which is a systematic source of false negatives.
- **Steering is contested.** AxBench found prompting beat every steering method and a simple difference of means beat sparse-autoencoder features at detection (Wu et al. 2025, arXiv:2501.17148). Replies show feature selection closes part of the gap (Arad et al. 2025, arXiv:2505.20063; Jørgensen and Hansen 2026, arXiv:2605.31183).
- **Features often add nothing over cheap baselines on detection** (Kantamneni et al. 2025, arXiv:2502.16681). This is the same shape as the Batman result.
- **Measurement theory is arriving in AI evaluation** (Jacobs and Wallach, arXiv:1912.05511; Wallach et al. 2025, arXiv:2502.00561; Bean et al. 2025, arXiv:2511.04703), but I found no paper that applies a full validation ladder to neurons or features as measurements of cultural content. SAEBench (Karvonen et al. 2025, arXiv:2503.09532) and concept-annotation evaluation (Klotz et al. 2026, arXiv:2606.24716) are the nearest. [checked: Klotz et al. evaluate sparse autoencoders on vision models (CLIP and DINOv2), not language models.]
- **The source of P1 is itself at the lowest rung.** Cammarata's "roughly 90% of neuron poles have one main job" comes from "preliminary audits of Qwen3-32B" (local copy a saved copy of the essay, line 70). It is a rater judgement. I saw no blind protocol, inter-rater agreement or random-direction control in the essay; a companion paper is promised. Its steering examples are illustrations. A LessWrong reply accepts the claim for argument's sake and disputes its usefulness (https://www.lesswrong.com/posts/rfnjfHCdKJTGgLP8L/why-clean-neurons-are-not-enough). [UNVERIFIED: on 2026-11-08 this URL returned HTTP 401 with LessWrong's "you don't have access to this page ... removed by the author" message, and the site's API refused the post ID as not allowed. The post may have been withdrawn or set to draft; do not cite it without re-checking in a browser.]

### C. Where the Batman instrument stands today

| Rung | Evidence in the repo | Verdict |
| --- | --- | --- |
| Reliability: paraphrase | 11 of 31 named units survive rewording in Qwen, 9 in Llama, 4 in both; a unit passes on another meme's rewording 16 to 18% of the time, so about 5 to 6 by chance; p = 0.002 and 0.076 (`study/results/robustness/nulls.json`, `f3_wording`) | Fails for about two thirds of units |
| Reliability: translation, length, tokenisation, prompt, seed | Not tested. Length correlates with meme count at 0.57 (`nulls.json`, `f2_trees.per_work_meme_count_r`) | Untested |
| Invariance across models | Top 33 units: twin r = 0.82, but word-matched non-meme units reach 0.86; name-free 0.14. Concept units (27): 0.62 against 0.46 for word-matched units on original text (p = 0.001), 0.34 against 0.38 name-free (`nulls.json`, `f2_agreement.sets`) | Fails for the top units; partial for concept units on original text only |
| Invariance across languages | Not tested (`preregistration/translation_protocol.md` is deferred) | Untested |
| Invariance across modalities | CLIP and DINOv2 twins 0.45 against 0.36 for random active neurons; one image per work; exploratory (`images/README.md`) | Weak |
| Convergent: human, TV Tropes, motif indexes | None. The blind label test used AI judges: 56 of 71 correct against 65% for a word-overlap judge, p = 0.007 (`nulls.json`, `label_test.all`) | Untested with humans |
| Discriminant against words | Neurons after removing words: Qwen p = 0.09 to 0.25 (0.13 to 0.37 under stricter nulls), Llama p = 0.011 to 0.036 (0.017 to 0.065 under stricter nulls). Words after removing neurons: p = 0.0003 to 0.003 (TF-IDF, both models) (`study/results/experiments.json`, `<model>.E2.residual_test_beyond_words`) | Fails in Qwen; weak pass in Llama |
| Discriminant against embeddings | Not tested | Untested |
| Predictive | Lineage (E2) holds but words do as well; selection (E3) null with about 27% power (`study/README.md`) | No increment shown |
| Causal: text intervention | One hand-picked "revenge" unit per model, chosen from 50: wins 10 of 12 minimal pairs in Llama, 4 of 12 in Qwen (`nulls.json`, `f3_wording.<model>.revenge`) | Anecdotal |
| Causal: steering | Not tested; the paper names it as future work (`study/paper/paper.md`, line 268) | Untested |

Two further threats run across every rung.
- **Multiplicity.** Qwen3-8B has 884,736 units and Llama 917,504 (`nulls.json`, `f1_chance`). For any label there will be some unit that fits a small sample. Every rung must use units frozen on development data and scored on held-out data, with the shrinkage reported.
- **Leakage.** Both models have read Wikipedia on Batman. A "prediction" about works inside the training data is recall, not prediction.

### D. The validation ladder

Definitions. A "unit" is one neuron (or one frozen family) in a named public checkpoint. Its "reading" is the study's score for a passage. "Fires" means the reading crosses the frozen presence bar. Every pass mark below is my proposal; the conventional ones are cited above. Each rung names the cheap rival, because a unit that passes a rung no better than the rival has earned nothing.

**Rung 0. Specification.** Test: unit, reading, presence bar, family rule and scoring are frozen and hashed before test data exist; units are chosen on development data only. Pass: commit hash precedes test data; every later number reports the drop from development to test. Cheapest data: none; the pre-registration's freeze points already do this.

**Rung 1. Reliability.** The motif stays the same and a nuisance factor changes. A frozen model gives the same activation every time for the same text, so test-retest is trivially perfect; all the real reliability questions are about the stimulus and the model instance.

| Facet | Test | Pass mark | Cheapest data |
| --- | --- | --- | --- |
| Paraphrase | Same motif instance reworded with almost no shared content words; also "lures" that share the words but lack the motif | Presence agreement kappa at least 0.61 between original and rewording; own rewording separated from other motifs' rewordings at AUC at least 0.85; at least 70% of units pass, against the 16 to 18% false-pass rate. Rival: sentence-embedding similarity (words fail by construction) | The 31 rewordings already in `study/results/wording.json`; the reworded and banned-word versions in `preregistration/planted_story_protocol.md` |
| Translation | Same passage machine-translated into 2 or 3 languages and back | Kappa at least 0.61 between languages; AUC drop no more than 0.05. Rival: a multilingual sentence encoder | Machine-translate the planted stories (`preregistration/translation_protocol.md` plans 272 Chinese stories) |
| Length and position | The same motif sentence set in 1, 5 and 50 sentences of neutral filler, early and late | Intraclass correlation at least 0.75 across lengths; false-fire rate on filler-only text stays at the nominal 3%. Rival: a length-only predictor and TF-IDF | Concatenate existing planted stories (the pre-registration's 48 long texts) |
| Tokenisation | Casing, spacing, hyphenation, British and American spelling, swapped proper names | Kappa at least 0.8 (this should be easy); not worse than keywords by more than 0.05 | Scripted edits of existing passages; no new writing |
| Prompt | Bare text, with a chat template, after an instruction, after unrelated preceding text | Intraclass correlation at least 0.75; the rank order of passages is preserved (Spearman at least 0.8) | Re-run the same passages with 4 wrappers |
| Seed and build | The same architecture retrained from other seeds; 4-bit against full precision; for families, membership rebuilt on a disjoint corpus | Full precision against 4-bit: r at least 0.95. Across seeds: a twin unit with r at least 0.7 for at least 70% of validated units. Family membership across two corpora: adjusted Rand index at least 0.6 | The five GPT-2 seeds used by Gurnee et al. (arXiv:2401.12181); other multi-seed model families exist but I did not confirm them this session |

**Rung 2. Invariance.** The question changes from "is the reading stable" to "does the same reading mean the same thing in each group".

| Level | Test | Pass mark | Cheapest data |
| --- | --- | --- | --- |
| Configural | For each validated unit, find its best twin in each other model, language or modality | Twin r at least 0.7 on name-free, word-controlled text for at least 70% of units, and higher than for word-matched control units (paired test p below 0.01, median gap at least 0.15) | The two models' activations on the name-free corpus already exist; add 3 open models |
| Metric | Dose-response: 0, 1, 2, 3 instances of the motif, or graded explicitness | Monotone in every group; slopes within 25% of each other | Planted stories at graded doses |
| Scalar | Differential functioning: given the human label, does firing depend on group (language, era, register, length band)? Logistic regression with group and interaction terms | Group odds ratios inside 0.67 to 1.5 for at least 75% of units; AUC gap between groups at most 0.05 | Translated and period-styled planted stories (`preregistration/drift_protocol.md`) |

Rival for the whole rung: a sentence-embedding detector moved between models by a linear map (the claim of Jha et al. 2025, arXiv:2505.12540) or between languages by a multilingual encoder. [corrected: Jha et al. do not claim a linear map. Their vec2vec translator is unsupervised and non-linear: multilayer perceptrons trained with adversarial and cycle-consistency losses through a shared latent space (checked in the arXiv HTML full text). A linear map is a cheaper rival the experiment can still use, but it is not their result.] Embeddings are built to be invariant, so they are the hard rival here.

**Rung 3. Convergent validity.**

| Criterion | Test | Pass mark | Cheapest data |
| --- | --- | --- | --- |
| Human annotation | Two annotators and an adjudicator mark motif presence on name-masked passages; first report human-human kappa, then unit-human agreement | Unit-human kappa within 0.10 of human-human kappa; AUC at least 0.80 against adjudicated labels | 200 passages by 10 motifs, about 20 annotator hours. The pre-registration's audit (A7) has one human, the author, so it gives no human-human figure |
| TV Tropes | Work-by-trope table as a noisy criterion. A missing tag is not evidence of absence, so score precision among the top-ranked works and hand-check a sample of untagged ones | AUC at least 0.75 on tagged against hand-checked untagged works | The trope-labelled film synopsis sets of Chang et al. (2021, arXiv:2101.07632) and Su et al. (2024, arXiv:2409.14324) |
| Folklore motif indexes | Expert-annotated motifs in tales (Thompson, ATU, El-Shamy) | AUC at least 0.75; report by simple and complex motifs separately, since human kappa differs (0.87 against 0.53) | GOLEM (Yarlott et al. 2024); the Arabian Nights set (Alyami and Finlayson 2026); tale-type data behind da Silva and Tehrani (2016, doi:10.1098/rsos.150645) is at population level, not text level |
| Full matrix | Campbell and Fiske's layout: at least 8 motifs, including near neighbours such as revenge, vow and harm, by 6 methods (neurons in model A, neurons in model B, keywords, embeddings, LLM reader, humans) | Same-motif unit-human agreement at least 0.6 and above the different-motif entries in its row and column in at least 90% of comparisons; above different-motif, same-method entries | The same 200 annotated passages |

Rival: a zero-shot LLM reader. It is likely to be the strongest method against human labels, and the honest framing is that humans are the yardstick and the LLM reader is the method to beat on accuracy per unit of cost.

**Rung 4. Discriminant validity.** This is the kill rung.

| Test | Pass mark | Cheapest data |
| --- | --- | --- |
| 2x2 of motif present or absent by key words present or absent | On the two mixed cells (motif without its words, words without the motif) unit AUC at least 0.80, and at least 0.05 above the best sentence-embedding detector with the interval excluding zero. Keywords score at or below 0.5 there by construction | Planted stories with banned-word and decoy versions; already designed |
| Increment beyond cheap measures | After partialling word overlap, TF-IDF, embedding similarity and length, the unit still predicts the criterion: p below 0.01 in both models and a gain of at least 0.05 AUC. Also run it the other way; if the cheap measure survives partialling and the unit does not, the unit fails | Existing E2 data for lineage; annotated passages for detection |
| Control units | Random units, word-matched units, and units from a randomly initialised copy of the model, pushed through the same pipeline | Control units pass every other rung at no more than their chance rate; the validated units' margin over word-matched controls is reported for each rung | One more extraction pass |
| Control labels | Assign motif labels to passages at random and repeat unit selection (Hewitt and Liang's selectivity) | Real-label accuracy exceeds control-label accuracy by at least 0.2 | Free |

**Rung 5. Predictive validity.**

| Test | Pass mark | Cheapest data |
| --- | --- | --- |
| Lineage on a fresh franchise | Source ranked in the top 20% and the increment over words and embeddings significant at p below 0.01, in both models | `preregistration/works_shakespeare.json` |
| Works after the training cut-off | The same test on films and series released after each model's cut-off, with documented sources | Wikipedia; one day of scouting |
| Retention in transmission chains | Units present at step 1 of a retelling chain predict what survives to step 5 better than word frequency does | Chains generated by LLM retelling; human chains if a public set can be found |
| Human similarity | Unit-profile similarity between works predicts human similarity ratings | 300 pairwise ratings |

Rival: TF-IDF, embeddings, LLM reader. Pass mark throughout: out-of-sample, and better than the best rival by a margin whose interval excludes zero.

**Rung 6. Causal validity.** Two directions, which answer different questions.

| Direction | Test | Pass mark | Cheapest data |
| --- | --- | --- | --- |
| Text to unit | Minimal pairs: insert or delete the motif with the smallest edit; also insert a different motif | The pre-selected unit wins at least 80% of pairs; effect size at least 0.8 standard deviations; no response to off-target insertions beyond chance. No choosing the unit after the fact | 12 to 20 pairs per motif; LLM-written, human-checked |
| Unit to text (steering) | Raise the unit while the model continues a neutral story. Blind judges say which motif, if any, appeared. Controls: no steering; a random unit at the same strength; a word-matched unit; simply prompting with the label; a difference-of-means direction | Target motif appears in at least 60% of steered continuations and at most 10% of controls; off-target motifs rise by at most 0.1; judges pick the steered unit out of 5 candidates at least 50% of the time (chance 20%); fluency at least 80% of baseline | No dataset; about 3,000 short generations |
| Necessity | Silence the unit and ask the model whether the passage contains the motif | Accuracy falls more than for random units of equal activation, with the interval excluding zero | Same passages as rung 3 |

Rival: prompting. AxBench found prompting steers better than any internal method, so the pass mark is specificity against matched control units, not beating prompting; the prompting result must still be reported. Steering shows the unit's effect on output, which can be narrower or broader than what makes it fire, so a steering pass does not replace the text-to-unit test.

**Rung 7. Iteration.** After each round revise the unit, the bar and, where justified, the human criterion. The instrument may overrule human annotators only where human agreement is low, two instruments from different model families agree with each other, and the disagreement makes sense on inspection. That is how thermometers came to overrule the hand.

**Order of work.** Rungs 0, 1 (paraphrase), 4 (2x2) and 6 (text to unit) share one dataset: planted stories. Run that first. If units fail rung 4 against embeddings and the LLM reader, stop: the distinctive claim of P1 is not supported for these models. If they pass, do rung 2 across models next, because it is the test of P2 and needs no new annotation.

## What this means for P1 and P2

**P1 (a meme can be measured by which neurons fire).**
- Status: a reasonable operational definition, not yet a validated measurement. History says that is a respectable starting point; the gene, the species and the OTU all began there.
- Evidence for: a minority of units behave like wording-independent detectors (4 survive rewording in both models: a funeral, Two-Face's coin, the secret identity, the joke; `nulls.json`, `f3_wording.survive_in_both_percentile`); blind judges match labels to firing above a word-overlap judge (79% against 65%; `nulls.json`, `label_test.all`); Llama's units carry some lineage information beyond words.
- Evidence against: most named units fail the rewording test; Qwen's units add nothing to words while words add to the units; 24 of the 33 strongest units are name or place detectors by hand reading (`study/README.md`). No human criterion and no steering exist.
- Conditions under which P1 could hold: for a subset of units, selected on development data, that pass the 2x2 test. P1 as a claim about all neurons, or about the units the discovery rule ranks highest, is contradicted by the study.
- The source claim (about 90% of neuron poles have one main job) is a single-lab rater audit on a different, larger model. It has not been through any rung above face validity.

**P2 (the measurement generalises because models converge).**
- In measurement terms P2 is the comparability test, the first test Chang's history offers when there is no ground truth. It has to be passed on the measurand, with the confounds removed.
- The study's evidence is against P2 for the strongest units (0.82 falls to 0.14 without names, and word-matched non-meme units match as well) and weakly for it on concept units on original text only (0.62 against 0.46).
- Whole-tree agreement (0.98) is not evidence for P2: random units give 0.90 to 0.93.
- Convergence of whole representation spaces (Huh et al. 2024, arXiv:2405.07987; Jha et al. 2025) is a claim about geometry, and it favours the rival: if spaces align, a sentence-embedding detector transfers between models at least as well as a neuron does. Gurnee et al. (2024) report that only a small minority of neurons are universal across GPT-2 seeds (1 to 5% [recalled from the abstract]). Neither result shows that individual neurons map one-to-one.
- Invariance has levels. Even if matching units exist (configural), comparing how much of a meme two eras or two languages carry needs scalar invariance, which nobody has tested.

## Experiment seeds

1. **The 2x2: motif by words.**
   - Question: does a frozen unit fire on the motif when its usual words are absent, and stay silent when the words are present without the motif?
   - Data: 30 motifs by 4 cells by 10 passages (1,200 short passages), LLM-written under the planted-story protocol, human spot-audit of 100.
   - Method: units chosen on development passages and frozen; AUC on the two mixed cells; both models.
   - Rival: sentence-embedding nearest-example detector and a zero-shot LLM reader (keywords and TF-IDF are reported but fail by construction).
   - Falsified if: unit AUC on the mixed cells is below 0.7, or not at least 0.05 above the embedding detector.
   - Cost: about 1 to 1.5 days, mostly unattended, on the study laptop; under 50 dollars of LLM calls. This is the pre-registered T2.
2. **Regnault comparability across a model zoo.**
   - Question: do units that pass seed 1 have twins in other models once names and shared words are controlled?
   - Data: the seed 1 passages run through 5 open models plus one multi-seed family.
   - Method: best-twin correlation for each validated unit, against word-matched control units.
   - Rival: word-matched non-meme units; a linear map between the models' sentence embeddings.
   - Falsified if: twin r is not above the control units' (as happened at 0.14), or the embedding map transfers detection better.
   - Cost: 1 to 2 days of extraction.
3. **A multitrait-multimethod matrix for memes.**
   - Question: do neuron readings agree with humans on the same motif more than they agree with themselves across different motifs?
   - Data: 200 name-masked plot summaries from existing trope and motif corpora, 10 motifs including near neighbours, two annotators and an adjudicator.
   - Method: fill the full matrix for 6 methods; apply Campbell and Fiske's four checks.
   - Rival: keywords, TF-IDF, embeddings and an LLM reader are rows of the same matrix.
   - Falsified if: the unit-human same-motif agreement is below different-motif same-method entries, or more than 0.10 below the LLM reader's agreement with humans.
   - Cost: about 20 annotator hours (roughly 400 dollars) and a day of compute.
4. **Differential functioning across language and era.**
   - Question: given the same human label, does a unit fire at the same rate in English, Chinese and Spanish text, and in 1940s and 2020s prose?
   - Data: 300 planted stories, translated and period-styled.
   - Method: logistic regression of firing on label, group and their interaction.
   - Rival: a multilingual sentence encoder and translated keyword lists.
   - Falsified if: group odds ratios fall outside 0.67 to 1.5 for more than a quarter of units, or the encoder shows less bias.
   - Cost: 1 day.
5. **Steering confusion matrix.**
   - Question: does raising a unit put its motif, and only its motif, into the model's continuation?
   - Data: none bought; 20 units by 20 neutral prompts by 5 conditions.
   - Method: blind 5-way identification of the steered unit by a cross-family LLM judge, with 100 items checked by a human.
   - Rival: prompting with the label; a difference-of-means direction; random and word-matched units at equal strength.
   - Falsified if: identification is at or below 30% (chance 20%), off-target motifs rise as much as the target, or only name units steer.
   - Cost: 1 to 2 days.
6. **Mock community.**
   - Question: does the reading rise with the number of planted instances, independent of text length?
   - Data: long texts assembled from planted stories at known doses (0, 1, 2, 5).
   - Method: dose-response slope and false-fire rate on zero-dose texts of each length.
   - Rival: TF-IDF counts; an LLM reader with recursive summarising; length alone.
   - Falsified if: Spearman correlation with dose is below 0.5, or length alone does as well.
   - Cost: half a day; reuses seed 1 stories.
7. **Random-network and embedding-layer controls.**
   - Question: does the whole pipeline (discover, label, reword, canon, tree) produce "memes" from a model that has learned nothing?
   - Data: the study corpus.
   - Method: run the pipeline on a randomly initialised copy and on the token-embedding layer alone.
   - Rival: these controls are the rival, alongside word overlap.
   - Falsified if: the controls match the real model's label-test, canon or tree figures.
   - Cost: 1 day.
8. **Blind replication of the "one main job" audit.**
   - Question: is the 90% figure reproducible by independent raters, and is the neuron basis special?
   - Data: 200 random neuron poles from Qwen3-8B and 200 random directions in the same layers.
   - Method: two blind raters plus an LLM rater; kappa; rate of "one main job" for neurons against random directions.
   - Rival: random directions; an LLM reading the top examples with no neuron hypothesis.
   - Falsified if: kappa is below 0.4, or random directions are rated clean at a similar rate.
   - Cost: about 10 rater hours.
9. **Lineage after the training cut-off.**
   - Question: do units identify the documented source of works the model cannot have read about?
   - Data: name-masked summaries of adaptations released after each model's cut-off.
   - Method: the E2 statistic with the residual test in both directions.
   - Rival: TF-IDF, embeddings, LLM reader.
   - Falsified if: the increment beyond words is again absent in either model.
   - Cost: 1 day of scouting plus extraction.

## Key sources

| Citation | URL or DOI | Why it matters | Priority |
| --- | --- | --- | --- |
| Chang 2004, *Inventing Temperature* | doi:10.1093/0195171276.001.0001 | How a measurement is built with no ground truth; comparability as the first test | A |
| Cronbach and Meehl 1955 | doi:10.1037/h0040957 | Construct validity and the nomological network | A |
| Campbell and Fiske 1959 | doi:10.1037/h0046016 | The multitrait-multimethod matrix; the formal home of "words do as well" | A |
| Borsboom, Mellenbergh and van Heerden 2004 | doi:10.1037/0033-295X.111.4.1061 | Validity as causation; grounds the text-intervention test | A |
| Callahan, McMurdie and Holmes 2017 | doi:10.1038/ismej.2017.119 | Why fixed, reusable labels beat corpus-dependent clusters | A |
| Wu et al. 2025, AxBench | arXiv:2501.17148 | Steering and detection lose to simple baselines | A |
| Heap et al. 2025/2026 | arXiv:2501.17727 | Interpretability scores do not separate trained from random networks | A |
| Paulo and Belrose 2025 | arXiv:2501.16615 | Learned features depend on the seed | A |
| Gerstein et al. 2007 | doi:10.1101/gr.6339607 | A century of working gene definitions | B |
| de Queiroz 2007 | doi:10.1080/10635150701701083 | Concept against operational criteria | B |
| Stackebrandt and Goebel 1994; Edgar 2018 | doi:10.1099/00207713-44-4-846; doi:10.1093/bioinformatics/bty113 | A threshold calibrated on the older standard, later revised | B |
| Hebert et al. 2003 | doi:10.1098/rspb.2002.2218 | The barcode gap condition | B |
| Meredith 1993; Putnick and Bornstein 2016; Cheung and Rensvold 2002 | doi:10.1007/BF02294825; doi:10.1016/j.dr.2016.06.004; doi:10.1207/S15328007SEM0902_5 | Levels of invariance and pass marks | B |
| Leask et al. 2025; Chanin et al. 2024 | arXiv:2502.04878; arXiv:2409.14507 | Features are not canonical; absorption causes false negatives | B |
| Hewitt and Liang 2019; Adebayo et al. 2018 | arXiv:1909.03368; arXiv:1810.03292 | Control tasks and randomisation checks | B |
| Bolukbasi et al. 2021; Huang et al. 2023 | arXiv:2104.07143; arXiv:2309.10312 | Labels from top examples mislead | B |
| Yarlott et al. 2024, GOLEM; Alyami and Finlayson 2026 | https://aclanthology.org/2024.lrec-main.689.pdf; arXiv:2603.19283 | Human-annotated motif data with agreement figures | B |
| Chang et al. 2021; Su et al. 2024 | arXiv:2101.07632; arXiv:2409.14324 | Trope-labelled film synopses | B |
| Freiesleben 2026 | arXiv:2603.15121 | Nomological against causal validity for language models | B |
| Messick 1995; Kane 2013 | doi:10.1037/0003-066X.50.9.741; doi:10.1111/jedm.12000 | Validity belongs to the interpretation | C |
| Flake and Fried 2020 | doi:10.1177/2515245920952393 | Questionable measurement practices | C |
| Wallach et al. 2025; Bean et al. 2025; Jacobs and Wallach | arXiv:2502.00561; arXiv:2511.04703; arXiv:1912.05511 | Measurement theory applied to AI evaluation | C |
| Kantamneni et al. 2025; Arad et al. 2025; Jørgensen and Hansen 2026 | arXiv:2502.16681; arXiv:2505.20063; arXiv:2605.31183 | Features against baselines; the steering dispute | C |
| Portin and Wilkins 2017; Griffiths and Stotz 2006; Benzer 1955 | doi:10.1534/genetics.116.196956; doi:10.1007/s11017-006-9020-y; doi:10.1073/pnas.41.6.344 | Gene concept history | C |
| Johnson et al. 2019; Schloss and Westcott 2011 | doi:10.1038/s41467-019-13036-1; doi:10.1128/AEM.02810-10 | Resolution limits and method dependence of 16S units | C |
| Tal, "Measurement in Science" (SEP); Tal 2013 | https://plato.stanford.edu/entries/measurement-science/; doi:10.1111/phc3.12089 | Overview of the philosophy of measurement | C |
| Karvonen et al. 2025, SAEBench; Klotz et al. 2026 | arXiv:2503.09532; arXiv:2606.24716 | Multi-metric evaluation of features | C |
| Koo and Li 2016; Landis and Koch 1977 | doi:10.1016/j.jcm.2016.02.012; doi:10.2307/2529310 | Conventional reliability benchmarks | C |

## Fog

- I did not re-read Chang's book this session. The Regnault comparability story, the stage sequence and the "respect the earlier standard" principle are from memory; the book, its chapter titles and the iteration thesis were confirmed online.
- The motif-annotation kappa figures (0.72, 0.87, 0.53; above 0.55 to 0.7) come from search summaries, not from the PDFs. Check before quoting.
- Benzer's 1957 chapter, the count of species concepts, the Gurnee 1 to 5% figure, the Koo and Li and Landis and Koch bands, and the mock-community practice were not confirmed this session.
- I found no study of how stable individual neuron activations are under paraphrase, translation or prompt wrapping. The reliability rung may be genuinely new work.
- I found no public human retelling-chain corpus and did not confirm any multi-seed model family other than Gurnee's GPT-2 seeds.
- TV Tropes: I did not check the size, licence or label quality of the trope corpora, nor whether Batman works are in them.
- Every pass mark in the ladder is a proposal. The right marks depend on human-human agreement on the same items, which nobody has measured for this project.
- Whether Cammarata's companion paper reports rater agreement or a random-direction control is unknown; the essay alone does not.
- Open design question: should the validated unit be a single neuron (fixed label, noisy) or a frozen family (more reliable, corpus-dependent)? Psychometrics would treat a family as a scale and ask for internal consistency on a second corpus. Nothing in the repo tests that.
- The seed rung may not be testable for Qwen or Llama at all, since no second-seed checkpoints of those models are public as far as I know.
