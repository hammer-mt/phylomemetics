# l11-datasets: datasets for the next experiments

Scout date 2026-11-08. No experiments were run. "Checked" below means I fetched the page or API myself today; "search" means I only saw a search summary of the source and did not open it. Anything else is marked `[UNVERIFIED]`.

Ground-truth types: (a) known descent or documented influence, (b) measured success, (c) expert or crowd labels of the memes or motifs.

## Bottom line

- **Most "descent" datasets cannot help the neuron claim at all.** In manuscript copies, bill text reuse, MemeTracker phrases, newspaper reprints and chain letters, the ground truth is itself shared wording. Word overlap wins those by construction. They are good only for checking that the tree-building code recovers a known tree.
- **Three kinds of data can separate neurons from words**: same meme in different words (folktale types, tropes, translations, adaptations across media), same topic with different outcomes (headline A/B tests, caption contests), and memes that humans have already named (TV Tropes, ATU, SCP tags). The top of the ranking is drawn from these.
- **Best single dataset: the Upworthy Research Archive.** 32,487 randomised headline tests, CC BY 4.0, openly downloadable, with an exploratory/confirmatory split built for pre-registration. It fits on a laptop. Two warnings: the archive's own maintainers reported in June 2024 that randomisation probably failed in part of it (a cache fault from 25 June 2013 to 10 January 2014, about 22% of tests; https://upworthy.natematias.com/2024-06-upworthy-archive-update.html), and sparse-autoencoder features have already been applied to headline click data (Movva et al., ICML 2025), so this is not virgin ground.
- **Best "named memes" data: TV Tropes labels joined to Wikipedia plot summaries.** 30K tropes over 40K works exist as a research dataset. The licence is non-commercial share-alike, which clashes with this repo's MIT and CC BY-SA files, so commit only IDs and labels, not trope text.
- **Several famous sources are unusable or risky**: WhoSampled (closed, and audio), paired tweets (the data page returns 404 and it is tweet IDs only), Archive of Our Own (has all three kinds of ground truth, but its community treats scraping as theft), Meta Ad Library (no outcome measure), RecipeNLG (licence unknown).
- **All of the top 15 can run on one laptop, six of them whole** (Upworthy, ATU folktales, the stemmatology sets, the CMU film corpus, the Berezkin matrix, and the SCP wiki after a crawl). The rest run as a sample. Three cannot run in full: the OpenAlex snapshot (about 745 GB compressed as JSON Lines in the September 2026 release, per OpenAlex's own documentation; the 330 GB first written here was wrong or out of date; use the API for a subset) and the raw Viral Texts and MemeTracker corpora (use the cluster files).

## Findings

### Why the ranking is not by fame

The Batman study found that plain word overlap found documented sources as well as neurons did: on name-free texts, 16% (Qwen) and 10% (Llama) of eligible comics beat the documented source, against TF-IDF 11% and word presence 17%; the pre-registered residual test failed for Qwen and passed for Llama (`study/README.md`, row "E2 Heredity"). So a dataset is only worth running if word overlap could plausibly lose on it. I scored each candidate on four things: strength of ground truth, whether surface wording can carry the answer, access and licence, and laptop size.

### Datasets checked, by ground-truth type

**(b) Success with topic held fixed**

| Dataset | Size | Language | Access | Licence | Notes |
| --- | --- | --- | --- | --- | --- |
| Upworthy Research Archive | 32,487 tests, 150,817 arms, 538,272,878 assignments, 24 Jan 2013 to 30 Apr 2015 (checked: abstract of the Scientific Data paper via Crossref). Confirmatory set 22,743 tests and exploratory set 4,873 tests (checked: upworthy.natematias.com); 105,551 packages in the confirmatory set (checked: arXiv 2603.07018 calls it the "105,551-arm confirmatory subset") | English | OSF node `jd64p`, public (checked via OSF API, last modified 2024-03-13) | CC BY 4.0 (checked via OSF API; Crossref lists the paper as CC BY 4.0 too) | Randomised, so wording is the only cause. The fault was reported by the archive's maintainers, not first by the 2026 paper: their June 2024 update (https://upworthy.natematias.com/2024-06-upworthy-archive-update.html, checked) says a Cloudflare cache glitch probably served one arm at a time from 2013-06-25 to 2014-01-10, about 22% of tests, and that no published finding was materially affected. arXiv 2603.07018 (checked, full text) follows that update and excludes the window. Movva et al. dropped it as well: 28K rows removed, leaving 122,565 headlines and 14,128 tests with two or more headlines (checked: appendix of arXiv 2502.04382) |
| New Yorker caption contest ranking | 365 contests (numbers 530 to 895), over 2.2M captions, over 250M ratings (checked: full text of arXiv 2406.10522, Zhang et al., published at NeurIPS 2024) | English | Hugging Face `yguooo/newyorker_caption_ranking` (checked: named in the paper and present on the Hugging Face API) | CC BY-NC 4.0 (checked via Hugging Face API) | Same cartoon, many captions, dense ratings. The cartoon is an image, so the text-only signal is partial |
| Paired tweets (Tan, Lee, Pang 2014) | 1.77M topic- and author-controlled pairs, filtered to 11,404 (checked: full text of arXiv 1405.1438) | English | **Broken.** `chenhaot.com/pages/wording-for-propagation.html` returned 404 today (checked; 404 again on the checker's retry). Tweet IDs only, so texts need paid X API | Not stated | Do not plan on it |
| r/Jokes | 573,335 jokes with upvote scores, 11 years `[UNVERIFIED]` (count and span come from a search summary; the repository README states neither. The paper itself is confirmed: Weller and Seppi, "The r/Jokes Dataset: a Large Scale Humor Collection", LREC 2020, https://aclanthology.org/2020.lrec-1.753/; data current to 1 Jan 2020 per the README) | English | GitHub `orionw/rJokesData` (checked: README fetched) | Reddit terms; must honour deletions (checked: README) | No topic control. Upvotes depend on timing |
| Kickstarter (Web Robots) | Monthly crawl of all projects since March 2016, JSON and CSV; latest dump 2026-10-12 (checked on webrobots.io) | Mostly English | Direct download | Not stated `[UNVERIFIED]` | Blurb plus funded or not. Confounded by category, goal and creator |
| CMU Movie Summary Corpus | 42,306 plot summaries from the 2 Nov 2012 Wikipedia dump; Freebase metadata (4 Nov 2012 dump) on 81,741 films including box office (checked: cs.cmu.edu/~ark/personas and its README.txt). By Bamman, O'Connor and Smith, ACL 2013 | English | Direct download | CC BY-SA, no version stated (checked: project page) | Box office is driven by budget and stars far more than plot. Useful as the success column in a joined film table |
| Meta Ad Library | Political and issue ads; all EU and UK ads for one year (search, vendor guides; `[UNVERIFIED]`, no primary Meta page was opened by the scout or the checker) | Many | API with identity verification; no creative images (search) | Meta terms | Spend and impression ranges only. No effectiveness outcome. Weak |

**(c) Named memes and motifs**

| Dataset | Size | Language | Access | Licence | Notes |
| --- | --- | --- | --- | --- | --- |
| TV Tropes (Gala et al. 2020) | 30K tropes, 1.9M examples, 40K works (checked: full text of arXiv 2011.00092; exact counts 29,457 tropes, 1,919,844 examples, 40,435 works) | English | GitHub `dhruvilgala/tvtropes`. The data is a Google Drive file of about 650 MB linked from the README, not a GitHub release (checked: README) | Site is CC BY-NC-SA since July 2012 (search: Wikipedia; `[UNVERIFIED]` for the date and the exact licence name: the checker confirmed from the Wikipedia article only that the site changed its licence to allow non-commercial distribution only). Dataset licence not confirmed | Crowd labels, not expert. Uneven coverage: popular works have many more tropes. The release includes film and TV tables already matched to IMDb and a literature table matched to Goodreads (checked: README) |
| Tropes in films (García-Ortega et al. 2020) | Film to trope table from TropeScraper, April 2020 (checked: arXiv 2006.05380 abstract and full text) | English | GitHub `raiben/made_recommender` (search; `[UNVERIFIED]`: the repository page returned 403 and its README 404 to the checker, and the arXiv paper does not name it) | Paper says CC BY-SA 4.0 (search; `[UNVERIFIED]`: no licence statement for the dataset appears in the full text of arXiv 2006.05380v3; it says only that TropeScraper is free software) | Already joined to IMDb ratings (`[UNVERIFIED]` for this paper: the text of arXiv 2006.05380v3 never mentions IMDb. The IMDb-rating join probably belongs to the same authors' companion "StarTroper" paper, which this one cites as reference 5 and which the checker did not open) |
| Hugging Face TV Tropes dumps | Up to 651,522 wiki pages (search: `RyokoExtra/TvTroper`; exact count `[UNVERIFIED]`, the Hugging Face API gives only the size class 100K to 1M) | English | Hugging Face | Labelled Apache-2.0 (checked via Hugging Face API and dataset card), which does not match the site licence. Do not trust | |
| ATU-labelled folktales (Hagedorn and Darányi 2022) | Ashliman Folktexts with ATU type per tale. Zenodo record 6575263, one 10.8 MB zip, dated 2022-05-24 (checked via Zenodo API) | English translations | Zenodo, GitHub `j-hagedorn/trilogy` | "other-open" on Zenodo (checked). The data paper says CC BY-SA 4.0 (checked: Hagedorn and Darányi, "Bearing a Bag-of-Tales", Journal of Open Humanities Data 8, 2022, https://doi.org/10.5334/johd.78) | The authors say they found no other open ATU-annotated English set (checked in the data paper: they "could not identify suitable candidates on GitHub, Kaggle or Google"). The same paper notes that the Multilingual Folk Tale Database has put a large amount of ATU data online; the scout did not look at it. The set covers the 214 Folktexts pages with clear ATU annotations out of 370 |
| Irish and Gaelic folktales (Meaney et al. 2024) | 4,692 tales, of which about 2,543 carry an ATU number: all 2,091 Irish tales and only 452 of the 2,601 Scottish Gaelic ones (corrected from "4,692 tales with ATU numbers"; checked: full text of arXiv 2411.05593, Table 1. The prose says 451 Gaelic tales, the table 452) | Irish, Scottish Gaelic | Not confirmed | Not confirmed | Low-resource languages: a hard test for P2 |
| Tehrani 2013, Little Red Riding Hood | 58 tales by 72 plot characters (checked on the PLOS ONE article page, doi 10.1371/journal.pone.0078871: 58 variants of ATU 333/123 in English translation from 33 populations, 72 plot variables) | Coded matrix | Paper supplement (not opened): character list in File S1, matrix in File S2, tale sources in Table S1 (named on the article page) | CC BY (checked: article page; Crossref lists CC BY 4.0) | Matrix only. No tale texts, but Table S1 lists where each English translation was published |
| da Silva and Tehrani 2016 | 275 tale types by 50 Indo-European populations (the study's model paper; doi 10.1098/rsos.150645 checked via Crossref; the 275 and 50 checked in the full text, PMC4736946) | Coded matrix | Supplement (not opened) | CC BY 4.0 (checked: licence statement in the full text; was `[UNVERIFIED]`) | Matrix only |
| Berezkin catalogue | 2,564 motifs `[UNVERIFIED]` over 958 groups (958 groups checked in the NBER w25430 PDF, which says "thousands of motifs across 958 world societies"; the figure 2,564 does not appear in that PDF and comes from a search summary); about 50,000 text abstracts (search; `[UNVERIFIED]`) | Russian abstracts; English motif names since 2014 (`[UNVERIFIED]`, not rechecked) | ruthenia.ru/folklore/berezkin (not opened) | Not confirmed | Labels are per tradition, not per text. Used in Michalopoulos and Xue, QJE 2021, 136(4), 1993–2046 (doi 10.1093/qje/qjab003, checked via Crossref; working paper doi 10.3386/w25430; replication package not confirmed) |
| SCP wiki | Articles dated, rated and tagged on the live wiki. Hugging Face `recursal/SCP-RECURSAL` covers 2008 to 2022, about 36M tokens, but ratings were stripped (checked: dataset card says about 36.23M tokens of tales and series 1 to 8, with author credits and ratings removed in cleaning) | English; many language branches | No ready set with ratings, tags and dates together. Needs own crawl | CC BY-SA 3.0 (checked: dataset card and Hugging Face API) | Gives (b) and (c) on the same dated items. Licence is compatible with the repo's Wikipedia data |
| Know Your Meme | 18.3K entries as of May 2018 (checked: full text of arXiv 1805.12512, Zannettou et al., "On the Origins of Memes by Means of Fringe Web Communities"; they crawled 15.6K of them) | English | Scrape only; no official dump | Not confirmed | Mostly image memes. Wrong unit for text neurons |

**(a) Descent or documented influence**

| Dataset | Size | Language | Access | Licence | Notes |
| --- | --- | --- | --- | --- | --- |
| Wikidata links | Counted today on the Wikidata query service: `P144` based on 99,206 (99,207 on the checker's recount later the same day; the other four counts matched exactly); `P4969` derivative work 40,376; `P1877` after a work by 16,339; `P941` inspired by 9,297; `P8371` references work 2,727 | Language-neutral IDs | SPARQL or dumps | CC0 | Texts must come from Wikipedia plots (CC BY-SA). The pre-registration already uses P144 and P941 for Shakespeare and Batman relatives (`preregistration/sources/LICENSE.md`) |
| Artificial stemmatology (CASC) | Three hand-copied traditions with known true trees: Heinrichi (67 copies `[UNVERIFIED]`: the data page gives no count, only that the organisers and 17 volunteers copied it by hand and that some copies are held back; old Finnish, checked), Parzival (English translation by A. T. Hatto, checked), Notre Besoin (French, checked). The page says the correct stemma and a scoring script are on the "Causality workbench" (not opened) | Finnish, English, French | cs.helsinki.fi/u/ttonteri/casc (checked). Also in the R package `stemmatology` (search) | **Restricted.** The page says the data is "provided only for use related to the challenge" and further use needs permission (checked) | Truth is word-level copying error. Word overlap will win |
| Open Stemmata | About 50 stemmata in 2021, mostly Old French (search; `[UNVERIFIED]`: the README shows only live count badges, which the checker could not read) | Old French, Middle High German and others | GitHub `OpenStemmata/database` (site checked; repo blocked in this session; the checker fetched its README through raw.githubusercontent.com). Founded 20 Nov 2020 by Jean-Baptiste Camps and Gustavo Riva (checked: project site) | CC BY-SA 4.0 (checked: repo README) | Tree graphs from scholarship. Each record is a metadata file plus a GraphViz DOT file of the stemma (checked: README). Witness texts are generally not included, so little use for text methods |
| Bible translations (eBible corpus) | 1,009 translations in 833 languages across 75 language families (checked: arXiv 2304.09919 abstract) | 833 languages | GitHub `BibleNLP/ebible` (checked: README), Hugging Face | Varies per translation; many public domain or CC (checked: the README says each text is public domain, under a permissive CC licence, or used with the rights holder's permission, with one licence file per translation under `metadata/licenses`) | Verse alignment gives exact "same content, different words" pairs. The revision lineage of English versions is documented in general histories `[UNVERIFIED this session]` |
| Viral Texts | Reprint clusters from nineteenth-century US newspapers (Chronicling America) | English | GitHub org `ViralTexts`; repos with "clusters" in the name exist (search; contents not confirmed). The project site, led by Ryan Cordell and David Smith, also points to a cluster database at https://clusters.viraltexts.org (checked: viraltexts.org; the database itself not opened) | Not confirmed | Dated; reprint count is a success measure. Clusters are found by n-gram alignment, so truth is lexical |
| MemeTracker | 96,608,034 documents, 210,999,824 phrases, 418,237,269 links, Aug 2008 to Apr 2009 (checked on SNAP, and confirmed again by the checker; SNAP labels the 210,999,824 "memes"). Phrase-cluster file about 220 MB (search; not rechecked) | English | snap.stanford.edu/memetracker | Not stated | Phrase variants cluster by wording. Truth is lexical |
| OpenAlex (arXiv and other citations) | Full snapshot about 745 GB compressed as JSON Lines (works alone about 615 GB) or about 770 GB as Parquet, about 626 million records, as of the September 2026 release (checked: docs.openalex.org/download-all-data/openalex-snapshot). The scout's first figure, "about 330 GB compressed (search, secondary source)", is out of date by more than a factor of two | Mostly English | Public S3 bucket; free API | CC0 (checked: `LICENSE.txt` in the bucket is CC0 1.0) | Citation is documented influence and citation count is success, both on dated items. A citation is not proof an idea was inherited |
| WhoSampled | 1,250,246 songs in a set given to researchers under agreement (checked: full text of arXiv 2410.15996). A 2026 paper pulled metadata and audio for about 170k tracks, about 20% of the database, and calls the annotations incomplete (checked: full text of arXiv 2609.15465, a two-page late-breaking abstract) | Audio | No public API (search). Academic licence reportedly exists (search, secondary). The 2026 paper releases its mined annotations as "WhoSampled130k" (about 92k sampling annotations; https://mtg.github.io/whosampled-130k-dataset/, named in the paper, not opened) and calls it the largest publicly available sample-identification dataset | Closed at source; a mined subset is now public, so "closed" overstates it | Audio, not text. Out of scope for language-model neurons |
| Constitutions (Rockmore et al.) | 591 constitutions, 1789 to 2008 (search; arXiv 1711.06899 checked; doi 10.1002/asi.23971) | Many, in English translation | Text source not confirmed | Not confirmed | Descent is inferred by the paper, not documented. Long legal texts |
| Bill text reuse (Wilkerson, Smith, Stramp 2015) | Replication data on Harvard Dataverse, doi 10.7910/DVN/27584, described as aggregate data (search) | English | Dataverse (not opened) | Not confirmed | Truth is text reuse. Useless as a neuron test |
| Chain letters | 33 letters, 1980 to 1995 (search: Bennett, Li, Ma, Scientific American 2003). VanArsdale's archive at silcom.com/~barnowl returned 502 through the proxy today | English | Could not check | Not confirmed | A phylogenetics blog found the letters "not particularly tree-like" (search) |
| Archive of Our Own | 12.6M works were scraped and posted to Hugging Face in April 2025, then treated as theft by the community (search: Fanlore). | Many | Rate-limited; terms forbid scraping for commercial generative AI (search, secondary) | Authors keep copyright | Has "inspired by" links, kudos and tags: all three ground truths. Too risky for a public repo without author consent |

**Not labelled, not checked, or wrong shape**

- Project Gutenberg: the site forbids robots (checked: gutenberg.org/policy/robot_access.html); mirrors and offline catalogues are the allowed route. It is a text source, not a labelled dataset. Useful only as the text behind Wikidata links between public-domain works.
- RecipeNLG: 2,231,142 recipes; licence "unknown", access by a permission form from Poznań University of Technology (search). No descent or success labels.
- Hymnary.org: over a million hymn and tune records; bulk export unclear (search).
- Reddit copypasta: no dedicated dataset found. The nearest study, Adamic et al. 2014 (arXiv 1402.6792, checked), used Facebook data that is not public.
- Proverbs, software forks, book blurbs with sales, Billboard lyrics, TED talks, patents: not checked in the time box. `[UNVERIFIED]`

### Ranked top 15

Laptop column: YES means the whole set; SAMPLE means a subset is enough and fits; NO means the full set does not fit.

| # | Dataset | Truth | Experiment it unlocks | Laptop |
| --- | --- | --- | --- | --- |
| 1 | Upworthy Research Archive | b | Within-test winner prediction: can neuron units pick the headline that won when topic is fixed and word overlap between arms is uninformative? | YES |
| 2 | TV Tropes labels joined to Wikipedia plots | c | Do neuron units line up with memes people already named? Predict trope presence from name-free plot text | SAMPLE |
| 3 | ATU-labelled folktales (Hagedorn and Darányi) | c, a | Tale-type retrieval: find the same tale type in a different wording, culture or translation | YES |
| 4 | Wikidata `P144`, `P941`, `P4969` plus Wikipedia plots | a | Descent detection across media at scale, beyond one franchise: linked pairs against matched unlinked pairs | SAMPLE |
| 5 | Verse-aligned Bible translations (eBible) | a | Cross-language and cross-model agreement: does the same unit fire on the same verse in another language or another model? The cleanest P2 test here | SAMPLE |
| 6 | New Yorker caption contest ranking | b | Replicate #1 in a second genre: rank captions within one cartoon | SAMPLE |
| 7 | Artificial stemmatology (CASC) | a | Calibration only: does `study/phylo.py`-style tree building recover a tree that is known for certain? Ask permission first | YES |
| 8 | SCP wiki | b, c | Tags as named memes and rating as success on the same dated items; meme frequency over time against a neutral-copying model | YES after a crawl |
| 9 | OpenAlex (arXiv abstracts and citations) | a, b | Does a citing abstract share units with what it cites beyond shared vocabulary, and do units predict citation count within a field and year? | SAMPLE via API; NO for the snapshot |
| 10 | CMU Movie Summary Corpus | b | The success column for a joined film table with #2 and #4: which tropes or units go with box office after year and genre | YES |
| 11 | Kickstarter (Web Robots) | b | Predict funded or not from the blurb within category and goal band | SAMPLE (one monthly dump) |
| 12 | Viral Texts reprint clusters | a, b | Predict how often a passage was reprinted from its content, with date and paper controlled | SAMPLE; NO for raw pages |
| 13 | r/Jokes | b | Predict joke score; weak without topic control. Second choice after #6 | SAMPLE |
| 14 | Berezkin catalogue | c | Motif presence by tradition; needs the Russian abstracts and a multilingual model | YES for the matrix |
| 15 | MemeTracker phrase clusters | a, b | Mutation and selection of short quotes against a neutral model; lexical truth, so calibration only | YES for the cluster file; NO for raw |

Dropped from the ranking: WhoSampled (closed, audio), paired tweets (dead link), Archive of Our Own (consent), bill text reuse and Open Stemmata (lexical truth or no texts), constitutions (descent not documented), chain letters (33 items, site unreachable), Know Your Meme (images), Meta Ad Library (no outcome), recipes and hymns (no labels).

## What this means for P1 and P2

**P1 (a meme is a neuron).** No dataset here is evidence for P1. What the scout shows is which data could test it fairly, and that most of the obvious data cannot.

- Copy-descent data (stemmata, reprints, phrase clusters, bill reuse, chain letters) define descent as shared wording. If neurons match word overlap there, that says nothing; if they lose, that is expected. Using them as a headline test would repeat the Batman result (`study/README.md`, E2) with a foregone conclusion.
- The fair tests are where the meme survives a change of words. ATU types, tropes, translations and cross-media adaptations all have that property. If neurons do not beat TF-IDF and sentence embeddings on those, P1 has no practical content left.
- Success data is a different claim from P1: "the units that fire predict what spreads". The nearest published work used sparse-autoencoder features on text embeddings, not raw neurons, on headline click data, and reported 45 of 60 generated hypotheses significant against at most 24 for baselines (Movva et al., ICML 2025; search summary of arXiv 2502.04382, ID checked). So learned features, not raw neurons, are the incumbent there. P1 as stated by Cammarata (raw neurons are already clean) has to beat that as well as word overlap.
- Crowd labels are not ground truth for "the real memes". TV Tropes coverage tracks popularity, so a unit that tracks "famous film" would look like it detects tropes. Any trope test needs a popularity control.

**P2 (units generalise across models).** The study itself warns against assuming this: family counts differ by model (14,843 for Qwen3-8B, 24,732 for Llama 3.1 8B, 5,641 for Qwen3-1.7B) and only Llama's neurons added anything beyond words (`study/README.md`, rows "The core canon is real" and "E2 Heredity"). Datasets that can test P2 directly are the aligned ones: verse-aligned Bibles (#5) and translated folktales (#3). They give thousands of item pairs known to carry the same content, so one can ask whether unit A in model X and its mapped unit in model Y fire on the same items, in a language neither mapping was fitted on. Nothing else on the list tests P2 as cleanly. Multilingual sentence embeddings are built for exactly this task and will be a hard rival.

## Experiment seeds

Cost basis: the pre-registration's timing file gives 67.8 sentences per second for Qwen3-8B 4-bit, aggregate only, measured on 24 made-up sentences (`preregistration/sources/neuron_timing_qwen3-8b.json`). Treat every time below as optimistic and re-time on real text first.

1. **Headline winner test (Upworthy).**
   - Question: with topic fixed by randomisation, do neuron units predict which headline arm won?
   - Data: Upworthy Research Archive. Fit on the exploratory set, test once on the confirmatory set. Drop the period with the reported randomisation fault.
   - Method: for each pair of arms in a test, predict the higher click rate from the difference in features. Compare feature sets at equal model class (regularised logistic regression).
   - Rival to beat: word and character n-gram differences; headline length and punctuation counts; sentence-embedding difference; an LLM asked "which headline gets more clicks?"; and sparse-autoencoder features on embeddings as in Movva et al.
   - Falsifier: neurons are not better than n-grams plus length on confirmatory pairwise accuracy, or the LLM judge beats neurons at lower cost.
   - Cost: 150,817 headlines is about 40 minutes of forward passes at the timing above. LLM judging of pairs is the expensive rival. One to two days in all.

2. **Named memes test (tropes on film plots).**
   - Question: is there a unit, or a small set, that detects a given trope in name-free plot text better than words do?
   - Data: Gala et al. or García-Ortega trope tables, joined by IMDb or Wikidata ID to Wikipedia plot sections. Start with about 2,000 films and the 200 most frequent tropes. Commit IDs and labels only.
   - Method: per trope, area under the curve for presence, with films split by franchise so no franchise is in both train and test. Control for number of tropes per film as a popularity proxy.
   - Rival to beat: TF-IDF logistic regression; sentence embeddings with logistic regression; an LLM reading the plot and answering yes or no per trope.
   - Falsifier: neurons trail embeddings by more than 0.05 AUC on the median trope, or the neuron advantage vanishes once popularity is controlled.
   - Cost: 2,000 plots, assuming about 35 sentences each, is about 20 minutes of forward passes. LLM reading at 200 tropes per film is the heavy part; subsample.

3. **Tale-type retrieval (ATU).**
   - Question: given one telling of a tale type, can the method find other tellings of the same type among distractors?
   - Data: Hagedorn and Darányi ATU-labelled Folktexts (Zenodo 6575263). Keep types with at least 4 tellings. Run with and without proper names.
   - Method: nearest-neighbour retrieval, mean reciprocal rank per type.
   - Rival to beat: TF-IDF cosine; word presence; multilingual-e5 or Qwen3-Embedding cosine; LLM pairwise "same tale type?".
   - Falsifier: neurons do not beat TF-IDF on name-free texts, or beat it only on types that share rare words.
   - Cost: the whole set is one 10.8 MB zip. Under an hour of model time.

4. **Same verse, other language, other model (P2).**
   - Question: do units mapped between Qwen and Llama fire on the same verses, and does a unit fire on the same verse in a language it was not selected on?
   - Data: about 8,000 New Testament verses in 6 public-domain translations across 4 languages from the eBible corpus. Check each translation's licence file.
   - Method: select units on English, test verse-level agreement on the other languages and on the other model's mapped units against a shuffled-verse null.
   - Rival to beat: multilingual sentence embeddings (retrieval of the aligned verse); for same-language pairs, word overlap.
   - Falsifier: cross-model unit agreement is no better than agreement between random unit pairs matched on firing rate, or embeddings retrieve aligned verses far better than units at lower cost.
   - Cost: 48,000 verses, about 15 minutes per model.

5. **Known-tree calibration (CASC).**
   - Question: does the study's tree-building recover a tree known for certain, and how does it compare with plain edit distance?
   - Data: Parzival and Notre Besoin, and Heinrichi if a model handles old Finnish. Get permission, since the page restricts use to the challenge.
   - Method: build trees from neuron units and from word variants; score against the true stemma with the challenge's own measure.
   - Rival to beat: word-variant distance with neighbour joining, which is the standard and should win.
   - Falsifier: this cannot support P1. It fails the pipeline if neuron trees are much worse than word trees, which would mean the Batman phylogeny has no demonstrated accuracy.
   - Cost: minutes. Three tiny sets.

6. **Adaptation links at scale (Wikidata).**
   - Question: does the Batman heredity result hold across thousands of unrelated source and adaptation pairs?
   - Data: a sample of `P144` pairs where both ends have a Wikipedia plot section, with names removed, and matched unlinked pairs of the same medium, genre and decade.
   - Method: rank the true source among decoys, as in E2. Pre-register the residual test that removes word similarity first.
   - Rival to beat: TF-IDF and word presence on the same name-free texts, as in `study/README.md` E2; sentence embeddings; an LLM pairwise judge.
   - Falsifier: the residual test fails again for the main model. Two failures across different corpora would settle that neurons add nothing to words for descent.
   - Cost: 2,000 pairs is about 4,000 plots; under an hour of model time, plus the name-removal step, whose cost I did not check.

## Key sources

| Citation | URL or DOI | Why it matters | Priority |
| --- | --- | --- | --- |
| Matias, Munger, Le Quere, Ebersole. The Upworthy Research Archive, a time series of 32,487 experiments in U.S. media. Scientific Data, 2021 | https://doi.org/10.1038/s41597-021-00934-7 ; data https://osf.io/jd64p/ | Randomised success data with a confirmatory split, CC BY 4.0 | A |
| Movva, Peng, Garg, Kleinberg, Pierson. Sparse Autoencoders for Hypothesis Generation. ICML 2025 | https://arxiv.org/abs/2502.04382 | Prior art: interpretable features predicting headline clicks, with baselines | A |
| Gala, Khursheed, Lerner, O'Connor, Iyyer. Analyzing Gender Bias within Narrative Tropes. 2020 | https://arxiv.org/abs/2011.00092 | The 30K-trope, 40K-work dataset | A |
| Hagedorn and Darányi. Ashliman Folktexts with ATU types. 2022 | https://doi.org/10.5281/zenodo.6575263 | The only open ATU-labelled English text set found | A |
| Wikidata query service counts, 2026-11-08 | https://query.wikidata.org/ | Size of the documented-influence graph, CC0 | A |
| Akerman et al. The eBible Corpus. 2023 | https://arxiv.org/abs/2304.09919 | Verse-aligned translations for the P2 test | A |
| Computer-Assisted Stemmatology Challenge data page | https://www.cs.helsinki.fi/u/ttonteri/casc/data.html | Known true trees; note the use restriction | B |
| Zhang, Jain, Guo, Chen et al. Humor in AI. 2024 | https://arxiv.org/abs/2406.10522 | 2.2M rated captions, topic-controlled | B |
| García-Ortega, García-Sánchez, Merelo-Guervós. Tropes in films. 2020 | https://arxiv.org/abs/2006.05380 | Film to trope table under CC BY-SA | B |
| CMU Movie Summary Corpus, project page (the paper's author list was not confirmed this session) | https://www.cs.cmu.edu/~ark/personas/ | Plots with box office | B |
| Web Robots Kickstarter datasets | https://webrobots.io/kickstarter-datasets/ | Monthly dumps, current to 2026-10-12 | B |
| da Silva and Tehrani. Comparative phylogenetic analyses uncover the ancient roots of Indo-European folktales. 2016 | https://doi.org/10.1098/rsos.150645 | The study's model; matrix data only | B |
| Tehrani. The Phylogeny of Little Red Riding Hood. 2013 | https://doi.org/10.1371/journal.pone.0078871 | 58 tales by 72 characters | B |
| Meaney, Alex, Lamb. LLMs and folktales in low-resource languages. 2024 | https://arxiv.org/abs/2411.05593 | 4,692 ATU-numbered Irish and Gaelic tales | B |
| Arčon, Robnik-Šikonja, Tratnik. LLMs for folktale type automation, Cinderella. 2025 | https://arxiv.org/abs/2510.18561 | LLM reading as a rival on ATU motifs | B |
| Leskovec, Backstrom, Kleinberg. MemeTracker data | https://snap.stanford.edu/data/memetracker9.html | Phrase clusters, lexical truth | C |
| Cordell and Smith. Viral Texts | https://viraltexts.org/ | Dated reprint clusters | C |
| Tan, Lee, Pang. The effect of wording on message propagation. ACL 2014 | https://arxiv.org/abs/1405.1438 | The paired design to copy; data link dead | C |
| Rockmore, Fang, Foti, Ginsburg, Krakauer. The Cultural Evolution of National Constitutions. 2017 | https://arxiv.org/abs/1711.06899 | 591 dated constitutions, inferred descent | C |
| Wilkerson, Smith, Stramp. Tracing the Flow of Policy Ideas in Legislatures. AJPS 2015 | https://doi.org/10.1111/ajps.12175 | Text reuse as descent; lexical truth | C |
| Michalopoulos and Xue. Folklore. QJE 2021 | https://doi.org/10.1093/qje/qjab003 | Berezkin catalogue in use | C |
| Figueiredo, Panoutsos, Andrade. Surprising Patterns in Musical Influence Networks. 2024 | https://arxiv.org/abs/2410.15996 | WhoSampled only by agreement | C |
| Araz, Lizarraga, Serra, Bogdanov. Building a Dataset for Music Sample Identification. 2026 | https://arxiv.org/abs/2609.15465 | WhoSampled annotations incomplete | C |
| Open Stemmata | https://openstemmata.github.io/ | Real stemmata, graphs only | C |
| OTW. AI and Data Scraping on the Archive | https://www.transformativeworks.org/ai-and-data-scraping-on-the-archive/ | Why Archive of Our Own is off limits | C |
| Parikh et al. TEA-Time. 2026 | https://arxiv.org/abs/2603.07018 | Reports the Upworthy randomisation fault | B |

## Fog

- I did not open any dataset file. Sizes, fields and licences from search summaries need a second look before anything is planned around them. GitHub API calls to other repositories were blocked in this session, so repository licences (Gala et al. tropes, r/Jokes, eBible, Viral Texts clusters, Open Stemmata) are unconfirmed.
- The Upworthy randomisation fault is from a search summary of one 2026 paper. The archive's own documentation should be read for the exact affected window and share.
- TV Tropes licence: is committing a film-by-trope label table (no trope text) to a public MIT and CC BY-SA repository acceptable under CC BY-NC-SA? I think labels are facts, but this is a judgement, not a checked legal position.
- The CASC use restriction may no longer be enforced, and the same three sets ship in a CRAN package whose licence I did not read.
- SCP wiki: I found no dataset with ratings, tags and dates together, and did not confirm an API. Article count is unknown to me.
- Not checked at all: proverbs, software forks, patents, book blurbs with sales, Billboard lyrics, TED talks, Goodreads, TMDB, Google's ads transparency data, current Reddit dump availability.
- The Paper Chain Letter Archive site was unreachable (502 through the proxy), so its size, dating and terms are unknown.
- Where Tehrani's and da Silva and Tehrani's supplementary matrices sit, and whether tale texts exist for the 58 Little Red Riding Hood variants, is not confirmed.
- Whether the English Bible revision lineage is clean enough to serve as a known tree was not checked; revisers consulted several earlier versions, so it is likely a network, not a tree.
- Web Robots' Kickstarter licence and whether the dumps include failed projects in full were not checked.
