# Cultural phylogenetics and stemmatics: methods and ground-truth benchmarks for neuron-meme trees

Key: l6-cultural-phylogenetics. Written 2026-11-08. Web research and file reading only; nothing was run.

## Bottom line

- **Ground truth for cultural descent is rare and small.** The only text datasets with a truly known tree are the artificial manuscript traditions made by volunteer scribes: Parzival (English, 21 witnesses made, 16 in the public challenge release), Notre Besoin (French, 13) and Heinrichi (Old Finnish, the largest: 37 witnesses released, 65 nodes in the published true graph). Everything else in the brief (folktales, chain letters, Canterbury Tales, projectile points, music) has an inferred tree, not a known one. Bennett, Li and Ma's 33 chain letters are often cited as a benchmark; they are not one, because their true history is unknown.
- **Those benchmarks are the wrong place to show neurons beating words.** Scribes change spelling and single words. Surface strings are the signal, and the cheap rivals already sit in the published results table: on Notre Besoin, n-gram clustering scored 66.4%, compression 70.6%, parsimony 74.5% and neighbour joining 76.2%; on Parzival every method landed between 72.6% and 83.7% (CASC results page, https://www.cs.helsinki.fi/u/ttonteri/casc/results.html; the 83.7% is SplitsTree ParsimonySplits, an organiser-added row). A neuron method can at best tie there. Use them as a floor check, not as the headline.
- **The most convincing test is an "artificial oral tradition".** Port the Roos and Heikkilä design from copying to retelling: a pre-drawn stemma of 60 or more nodes with missing witnesses, a few two-parent (contaminated) nodes and some translation edges, where every step is a retelling from memory so that wording is destroyed and content is inherited. Reconstruct blind, score with their triplet "average sign distance" plus parent-rank, and compare neurons against TF-IDF, compression distance, sentence embeddings and an LLM reading the texts. The pre-registered claim should be a dose-response: the neuron advantage grows as parent-child word overlap falls, and is zero for verbatim copies.
- **The study's own tree result currently cuts against the premise.** By tree distance the documented source comic ranks 5.44 of 12 for Qwen memes against 5.78 by chance (p = 0.40), while TF-IDF ranks it 2.89 (p = 0.004) (`study/results/robustness/nulls.json` → `f2_trees.source_recovery`; `study/results/phylo.json.gz` → `tfidf.adaptation`). The meme distance matrix correlates 0.84 with the TF-IDF matrix (`nulls.json` → `f2_trees.mantel.qwen_memes_vs_tfidf`).
- **The standard signal measures do not diagnose what people use them for.** Nunn et al. (2010) showed by simulation that a low retention index is produced more by fast change than by borrowing. Greenhill, Currie and Gray (2009) showed that tree topologies survive realistic borrowing. So a high RI or low delta score for a neuron tree would not prove descent, and a low one would not prove blending. Only a known tree settles it.
- **No one has published a language-model-feature or embedding reconstruction scored on a known stemma.** I found none in 2025 or 2026 searches. Hoenen wrote in 2022: "In stemmatology, there is to date no known DL approach at all" (abstract of arXiv 2206.05603, https://arxiv.org/abs/2206.05603, read directly; that paper then proposes one, a sequence-to-sequence model for placing a witness on a stemma). This is open ground.

## Findings

### 1. How tree-like is culture: the measures

| Measure | What it is | Reading | Source |
| --- | --- | --- | --- |
| Retention index (RI) | Share of apparent shared-derived similarity kept on the most parsimonious tree | 0 to 1; higher means less homoplasy | Farris 1989, doi:10.1111/j.1096-0031.1989.tb00573.x |
| Delta score | Per-quartet departure from the four-point condition, averaged per taxon | 0 = perfectly tree-like, 1 = not at all | Holland et al. 2002, doi:10.1093/oxfordjournals.molbev.a004030 |
| Q-residual | Scale-dependent sibling of delta | lower = more tree-like | Used in Gray, Bryant and Greenhill 2010, doi:10.1098/rstb.2010.0162 |
| NeighborNet | Split network that draws conflicting signal as boxes rather than forcing a tree | Boxiness shows reticulation | Bryant and Moulton 2004, doi:10.1093/molbev/msh018 |
| Pagel's lambda | Scaling of the tree's internal branches that best fits a trait | 0 = no phylogenetic signal, 1 = Brownian motion on the tree | Pagel 1999, doi:10.1038/44766 |
| D statistic | Sum of sister-clade differences for a binary trait, scaled by two nulls | 1 = random, 0 = Brownian, below 0 = strongly clumped | Fritz and Purvis 2010, doi:10.1111/j.1523-1739.2010.01455.x |

All six need either a character matrix or a distance matrix, and lambda and D need a reference tree that is trusted on other grounds. The Batman study has distance matrices and a neighbour-joining tree but has not reported a delta score, a Q-residual or a NeighborNet for them (I found none in `study/paper/paper.md` or `study/results/`).

### 2. Branching versus blending

- **Collard, Shennan and Tehrani 2006** (doi:10.1016/j.evolhumbehav.2005.07.003) compared retention indices of cultural and biological datasets and concluded that branching is not swamped by blending, and that the question must be settled case by case. The repository record confirms the RI is their tool. The averages are confirmed at second hand: Tehrani 2013 (full text read) reports them as "cultural (n = 21, mean = 0.59) and biological (n = 21, mean = 0.61) datasets analysed by Collard et al." The Collard paper itself was not read.
- **Nunn, Arnold, Matthews and Borgerhoff Mulder 2010** (doi:10.1098/rstb.2010.0009; PubMed abstract read): "Higher rates of evolutionary change have a stronger depressive impact on measures of homology than higher rates of horizontal transmission; thus, low consistency or retention indices are not necessarily indicative of 'ethnogenesis'." They also report that horizontal transmission inflates the type I error rate of independent contrasts when traits travel as a pair.
- **Greenhill, Currie and Gray 2009** (doi:10.1098/rspb.2008.1944; abstract read): simulated borrowing on a language model; Bayesian tree topologies are "robust to realistic levels of borrowing", dates less so (they are underestimated).
- **Gray, Bryant and Greenhill 2010** (doi:10.1098/rstb.2010.0162; abstract read): tree-likeness is a continuous dimension to be measured per domain with delta plots and Q-residuals; they compare lexical evolution across regions. I could not retrieve the numeric delta values (PDF fetch failed). One value at second hand, from Tehrani 2013: Gray et al.'s splits graph of 12 Indo-European languages, built on data that included loan words and the creole Sranan, had an average delta score of 0.23 and a Q-residual of 0.03.
- **Temkin and Eldredge 2007** (spelled Tëmkin in the journal record; "Phylogenetics and material cultural evolution", Current Anthropology 48(1):146-154, doi:10.1086/510463; citation confirmed, text not read): the classic case that material culture (cornets, Baltic psaltery) is strongly reticulate and that tree methods can mislead there. Dataset sizes not checked.
- **O'Brien, Darwent and Lyman 2001** (doi:10.1006/jasc.2001.0681; citation confirmed): cladistics on Paleoindian projectile points. No true tree exists. Sizes not checked.
- **Tehrani and Collard 2002** (doi:10.1016/s0278-4165(02)00002-8): Turkmen textiles, the founding RI-based branching study.

What this adds up to: the field's verdict is "often tree-like enough, varies by domain, and the summary statistics cannot separate borrowing from fast change". Franchise fiction, where each film draws on many comics, is close to the Temkin and Eldredge end. A tree is a poor model for Batman; a directed graph with several parents per work is the honest target, and E2's parent-ranking design already treats it that way.

### 3. Folktales

- **Tehrani 2013**, Little Red Riding Hood (doi:10.1371/journal.pone.0078871; page read): 58 variants of ATU 333/123 from 33 populations, 72 plot characters. Parsimony RI 0.72 (0.76 without the East Asian tales), average delta 0.30, Q-residual 0.03. He cites 0.60 as the RI level of branching-dominated datasets, and attributes that threshold to the Nunn et al. simulations ("datasets that return RIs of 0.60 and above are likely to have been mainly generated by branching phylogenesis"). That reading of Nunn et al. sits awkwardly beside their own abstract, which says low RI is not a reliable sign of borrowing; the two are compatible only if a high RI is informative and a low one is not. The East Asian tales are read as hybrids of ATU 333 and 123: a hypothesised blend, not a documented one. Matrix is supplementary File S2, article under CC BY.
- **da Silva and Tehrani 2016** (doi:10.1098/rsos.150645; full text read via Europe PMC): presence or absence of 275 ATU tales of magic in 50 Indo-European-speaking populations. D ranged from -2.06 to 3.9; 100 tales showed more phylogenetic clumping than chance at alpha 0.05; an autologistic model with spatial and phylogenetic neighbour graphs left 76 where vertical transmission outweighed horizontal. CC BY 4.0; data in electronic supplementary table S1. The reference tree is the language tree, so the test asks "does the tale follow language descent", not "is this the true tale tree".
- **Ross, Greenhill and Atkinson 2013**, "Population structure and cultural geography of a folktale in Europe" (Proc. R. Soc. B 280(1756):20123065, doi:10.1098/rspb.2012.3065; abstract read): 700 variants of one tale (the kind and unkind girls; the abstract gives the count but does not name the tale) in 31 European ethnolinguistic populations, analysed with population-genetic tools; ethnolinguistic boundaries matter more for the tale than for genes. No tree is claimed.
- **Bortolini et al. 2017** (doi:10.1073/pnas.1614395114; abstract read): 596 ATU tale types in Eurasia against genomic distances; language-biased diffusion fits best; 15 tales look demically transmitted. d'Huy and colleagues objected to the Eurocentric ATU index; reply at doi:10.1073/pnas.1715139114.
- **Indexes.** ATU (Uther 2004) and Thompson's Motif-Index are book-form expert classifications; they give type and motif labels, not tale texts, and are under copyright. **Berezkin and Duvakin's catalogue** (https://ruthenia.ru/folklore/berezkin/eng.html) had 2,450 motifs across 945 traditions as of April 2018 and abstracts of more than 55,000 texts `[UNVERIFIED]` (figures from search summaries of Berezkin's own articles; the check opened the live page, which loads and is the English motif list itself, but it states none of these totals; licence not found). Thuillard, Le Quellec, d'Huy and Berezkin 2018 (doi:10.3176/tr.2018.4.05) is the large-scale analysis; its abstract gives a confirmed snapshot of the same catalogue: "2264 motifs from ca. 40.000 myths recorded among 934 peoples". Michalopoulos and Xue 2021 (doi:10.1093/qje/qjab003) reuse the catalogue in economics.
- **Karsdorp and van den Bosch 2016** (doi:10.1098/rsos.160071; abstract read): a network of Dutch literary retellings of Red Riding Hood plus a paper chain-letter corpus, with links to each story's "pre-text". The links are inferred by the authors' method, so this is a model of descent, not ground truth. Sizes not checked.
- **2025.** Arčon, Robnik-Šikonja and Tratnik (arXiv 2510.18561, 21 Oct 2025): large language models detect motifs across a large collection of Cinderella variants and cluster them. This is the "LLM simply reading the text" rival already being applied to motif coding.

Blunt reading: every folktale dataset here is hand-coded presence or absence of types, motifs or plot features. None includes aligned full texts with known descent, so none can test whether a text-derived unit recovers ancestry. Their use for this project is as a model of how to test a unit for heritability (the D statistic and the autologistic control), not as benchmarks.

### 4. Manuscripts and stemmatics

- **Barbrook, Howe, Blake and Robinson 1998** (doi:10.1038/29667): phylogenetics of The Canterbury Tales. Already cited in the paper. No true tree.
- **Howe and Windram 2011**, "Phylomemetics: evolutionary analysis beyond the gene" (doi:10.1371/journal.pbio.1001069): the review that uses the project's own word, and describes artificial traditions copied along a predetermined history and analysed blind.
- **Windram, Shaw, Robinson and Howe 2008** (doi:10.1093/llc/fqn023): Dante's Monarchia as a test against an expert stemma.
- **Spencer, Davidson, Barbrook and Howe 2004** (doi:10.1016/j.jtbi.2003.11.022; abstract read): artificial manuscripts "with a known true phylogeny, produced by modern 'scribes'"; "we were able to reconstruct most of the true phylogeny", even though surviving ancestors and multifurcations are far commoner than in biology. This is the Parzival set.
- **Baret, Macé and Robinson 2006**, "Testing methods on an artificially created textual tradition", in *The Evolution of Texts* (Linguistica Computazionale 24-25, pp. 255-281 according to the Helsinki stemmatology wiki reference list, https://wiki.helsinki.fi/xwiki/bin/view/stemmatology/Tradition%2C%20artificial/, read in the check; the notes first said 255-283, which was not confirmed; no DOI exists in Crossref for the chapter). This is the Notre Besoin set.
- **Roos and Heikkilä 2009** (doi:10.1093/llc/fqp002): three artificial benchmarks, many methods; conclusions favour maximum parsimony in PAUP* and their compression-based method RHM.
- **Computer-Assisted Stemmatology Challenge** (https://www.cs.helsinki.fi/u/ttonteri/casc/, pages read). Scoring is the "average sign distance": for each triplet of documents, does the estimate agree with the truth about which of two is closer to the third? An empty graph scores a little over 50%. Results table as published:

| Method | Heinrichi | Parzival | Notre Besoin |
| --- | --- | --- | --- |
| RHM06 (compression-based) | 76.0% | 79.9% | 76.9% |
| PAUP parsimony | 74.4% | 77.8% | 74.5% |
| PAUP neighbour joining | 64.4% | 81.5% | 76.2% |
| n-gram clustering (Team Demokritos) | 64.4% | 79.3% | 66.4% |
| CompLearn (compression distance) | 52.7% | 81.5% | 70.6% |
| SplitsTree NeighborNet | 59.1% | 77.8% | 70.2% |
| Hierarchical clustering | not given | 72.6% | 60.2% |
| PAUP least squares (row added in the check) | 64.2% | 81.5% | 70.2% |
| SplitsTree SplitDecomposition (row added in the check) | 53.1% | 74.5% | 73.1% |
| SplitsTree ParsimonySplits (row added in the check) | 56.8% | 83.7% | 71.6% |

  The table above was checked cell by cell against the results page; the first seven rows are correct, and the last three rows were on the page but missing here. Only the first two rows were competition entries; the rest were added by the organisers. A separate page (https://www.cs.helsinki.fi/u/ttonteri/casc/stemma.html) gives a later neighbour-joining baseline for Heinrichi of 67.887% (skeleton) and 58.467% (orientation) with the `tripled.py` script, against 64.4% in this table, so scores from the script and from the 2007 table are not directly comparable.

  Two lessons. First, nothing is close to 100%: even with collated word-level variants the best method gets about three quarters of triplets right (best per set: 76.0%, 83.7%, 76.9%). Second, on the two small traditions cheap string methods are within a few points of the best; only on the larger, lossier Heinrichi do they fall behind. (Check: this holds for Parzival, where CompLearn at 81.5% is 2.2 points below the best. It is overstated for Notre Besoin, where CompLearn at 70.6% is 6.3 points and n-gram clustering at 66.4% is 10.5 points below RHM06 at 76.9%.)
- **Camps, Randon-Furling and Godreau 2025** (arXiv 2505.19246) and **Camps and Randon-Furling 2022** (arXiv 2210.16577): birth-death models of how stemmata get their shapes and how much is lost (the term "birth-death" is `[UNVERIFIED]`: the 2022 abstract says "agent-based models" and the 2025 abstract says "stochastic modelling, computer simulations, and data analysis"; both IDs, titles and authors confirmed on arxiv.org). The 2025 abstract reports that for medieval French chivalric literature "up to 60% of texts and 99% of manuscripts were lost". Useful as a generator of realistic tree shapes for simulated traditions.
- **Open Stemmata** (https://github.com/OpenStemmata/database): about 50 published stemmata in 2021 `[UNVERIFIED]`, mostly Old French, listed as CC BY-SA 4.0 (the licence is confirmed from the repository README, read in the check; the README names Old French and Middle High German as the most represented languages and shows only live count badges, so the 2021 figure could not be checked). These are scholars' hypotheses, not ground truth.
- **Hoenen 2026** (arXiv 2604.11724): OCR systems for Old Church Slavonic and a purely visual stemma reconstruction. Hoenen's 2018 LREC paper ("Multi Modal Distance - An Approach to Stemma Generation With Weighting", ACL Anthology L18-1332, https://aclanthology.org/L18-1332/) calls the three artificial sets "an extremely limited testbed" `[UNVERIFIED]`: the check read the PDF text and found neither "limited" nor "testbed" in it. What the paper does say is "From a machine learning perspective, these data sets are quite small", and it gives the sizes as "PRZ has 21 manuscripts and the alignment has 855 lines, NB features 13 manuscripts of 1035 lines and HR 64 manuscripts of 1208 lines".
- **Marmerola, Oikawa, Dias, Goldenstein and Rocha 2016**, "On the reconstruction of text phylogeny trees" (doi:10.1371/journal.pone.0167822; abstract read in the check: it tests on "a set of artificial near-duplicate documents with known phylogeny" and on Wikipedia revision histories, with a qualitative stemmatology case): the multimedia-forensics line that builds text (and, in sister papers, image) phylogeny trees from synthetic near-duplicates with known truth. Relevant to `images/` as well.

### 5. Chain letters, music and other descent-with-records

- **Bennett, Li and Ma 2003**, "Chain letters and evolutionary histories" (Sci. Am. 288(6):76-81, doi:10.1038/scientificamerican0603-76): 33 English paper chain letters collected 1980 to 1995 (count and dates confirmed from the blog post below, which adds that 8 other letters were excluded), tree built from compression distance. The true tree is unknown. A 2019 post on The Genealogical World of Phylogenetic Networks (https://phylonetworks.blogspot.com/2019/08/phylogenetics-of-chain-letters.html) judges the data "not particularly tree-like". The method became the normalized compression distance (Cilibrasi and Vitányi 2005, doi:10.1109/tit.2005.844059), which is a mandatory cheap rival.
- **Liben-Nowell and Kleinberg 2008** (doi:10.1073/pnas.0708471105; abstract read): 637 copies of an Iraq-war petition with nearly 20,000 signatories `[UNVERIFIED]` (neither figure is in the abstract, and the check could not fetch the full text; the abstract confirms only the "narrow but very deep tree-like pattern, continuing for several hundred steps"); the forwarding tree is reconstructed from the signature lists and is narrow and several hundred steps deep. This is as close to a true diffusion tree as exists, but the descent is recorded in the signatures while the petition text hardly changes, so it cannot test a content unit. Public data release not confirmed.
- **Youngblood 2019** (doi:10.1371/journal.pone.0211860 and doi:10.1098/rsos.191149): music sampling traditions; who-sampled-whom records are documented borrowing events. **Youngblood, Baraghith and Savage 2020** (arXiv 2011.02460): electronic-music phylogeny by dynamic community detection over 1,498,483 collaborative relationships, built to allow transmission between lineages. Audio, not text; data-source licences not checked.
- **Savage et al. 2022** (doi:10.1016/j.cub.2022.01.039): sequence alignment of folk-song melodies in tune families. Size, from the abstract read in the check: 10,062 Japanese and English melodies, of which 328 pairs are "highly related".
- **Transmission chains.** Mesoudi, Whiten and Dunbar 2006 (doi:10.1348/000712605x85871); Stubbersfield, Tehrani and Flynn 2014 (doi:10.1111/bjop.12073; online 2014, in print 2015 as Br. J. Psychol. 106(2):288-307, so it is often cited as 2015); Moussaïd, Brighton and Gaissmaier 2015 (doi:10.1073/pnas.1421883112); Breithaupt et al. 2018 (doi:10.3389/fpsyg.2018.02210). Each chain is a known line of descent by retelling, and several chains from one seed form a known star-shaped tree. Sizes and data availability not checked.
- **LLM chains.** Acerbi and Stubbersfield 2023 (doi:10.1073/pnas.2313790120): five preregistered transmission-chain experiments with an LLM (ChatGPT-3, per the abstract) show human-like content biases. Perez et al. 2024 (arXiv 2407.04503): LLM telephone games and cultural attractors. These make known-descent text cheap to generate.
- **PhyloLM**, Yax, Oudeyer and Palminteri 2024 (arXiv 2404.04671): phylogenetic trees of language models from output similarity, checked against known model families (111 open-source and 45 closed models, per the abstract). A methodological cousin on the P2 side.

### 6. Every dataset with a known true tree or documented borrowing

"Known" means recorded when the copies were made. "Documented" means a historical record of the link. "Inferred" means the tree is a research result and cannot serve as truth.

| Dataset | Truth status | Size | Language | Licence or access | Source |
| --- | --- | --- | --- | --- | --- |
| Parzival (Spencer et al. 2004) | Known tree | 21 witnesses, 855 aligned lines (Hoenen 2018, PDF read in the check). The public CASC release holds 16 witnesses (`parzival.nex`: ntax = 16, nchar = 855) | English (Hatto's translation of Wolfram) | "Provided only for use related to the challenge. Further use ... requires an explicit permission" (CASC data page); base text is a copyrighted translation | https://www.cs.helsinki.fi/u/ttonteri/casc/data.html |
| Notre Besoin (Baret et al. 2006) | Known tree | 13 witnesses, 1,035 aligned lines (`notrebesoin.nex`: ntax = 13, nchar = 1035; Hoenen 2018 agrees; the notes first said "about 1,030") | French (Dagerman, tr. Bouquet) | Same CASC restriction; base text copyrighted | same |
| Heinrichi (Roos and Heikkilä 2009) | Known graph; some witnesses withheld; more than one parent allowed; three outliers dropped from the truth | 37 witnesses released (`heinrichi.nex`: ntax = 37, nchar = 1208, confirmed in the check). The published true graph `correct.txt` has 65 nodes: 34 released witnesses plus 31 numbered withheld or hidden nodes, with the outliers Da, I and J left out. Hoenen 2018 gives 64 manuscripts of 1,208 lines for the full tradition (PDF read). The figure of 67 copies made is still `[UNVERIFIED]`; copies were made by 17 volunteers (CASC data page) | Old Finnish (Death-Psalm of Bishop Henry) | Released with the true graph and scoring script, no licence stated; folk text is public domain | https://www.cs.helsinki.fi/u/ttonteri/casc/stemma.html |
| Legend of St Henry | No known tree (real tradition) | 52 versions, about 900 words (`legend.nex`: ntax = 52, nchar = 1185) | Latin | Same CASC restriction | CASC data page |
| Largest artificial traditions | Known | "slightly over a hundred copies" as of May 2015 | not stated | not stated | Helsinki stemmatology wiki, "Tradition, artificial", https://wiki.helsinki.fi/xwiki/bin/view/stemmatology/Tradition%2C%20artificial/ (quote confirmed in the check) |
| Text phylogeny trees (Marmerola et al. 2016) | Known (synthetic) | not checked | not checked | article CC BY (PLOS) | doi:10.1371/journal.pone.0167822 |
| Iraq petition chain letter (Liben-Nowell and Kleinberg 2008) | Near-known forwarding tree from signatures | 637 copies, about 20,000 signatories `[UNVERIFIED]` | English | release not confirmed | doi:10.1073/pnas.0708471105 |
| Human transmission chains (Mesoudi 2006; Stubbersfield 2014; Moussaïd 2015; Breithaupt 2018) | Known linear descent | not checked | English, German | per paper; not checked | DOIs above |
| LLM transmission chains (Acerbi and Stubbersfield 2023; Perez et al. 2024) | Known by construction | not checked | English | not checked | DOIs above |
| Simulated trees with set borrowing rates (Greenhill et al. 2009; Nunn et al. 2010) | Known, but traits not texts | any | none | code availability not checked | DOIs above |
| Music sampling (Youngblood 2019) | Documented borrowing (who sampled whom) | not checked | audio | source site terms not checked | doi:10.1371/journal.pone.0211860 |
| Batman adaptations, this repo | Documented sources for 9 adaptations among 12 comics | 32 works plus 13 relatives | English plot summaries | see `DATA_LICENSE.md` | `study/results/robustness/nulls.json`, `study/results/phylo.json.gz` |
| Shakespeare plays and films; documented relatives, this repo | Documented adaptation links (planned, not run) | see files | English | see `DATA_LICENSE.md` | `preregistration/works_shakespeare.json`, `preregistration/documented_relatives.json` |
| Open Stemmata | Expert hypothesis, not truth | about 50 stemmata (2021) `[UNVERIFIED]` | mostly Old French | CC BY-SA 4.0 (repository README, read in the check) | https://github.com/OpenStemmata/database |
| Bennett, Li and Ma chain letters | Inferred only | 33 letters | English | texts in the article and course PDF; licence not found | doi:10.1038/scientificamerican0603-76 |
| Little Red Riding Hood (Tehrani 2013) | Inferred; hybrid origin hypothesised | 58 variants by 72 characters | coded traits | CC BY (File S2) | doi:10.1371/journal.pone.0078871 |
| Indo-European tales (da Silva and Tehrani 2016) | Inferred against the language tree | 275 tales by 50 populations | coded presence | CC BY 4.0 (table S1) | doi:10.1098/rsos.150645 |
| Kind and unkind girls (Ross et al. 2013) | None claimed | 700 variants, 31 populations | coded traits | not checked | Proc. R. Soc. B 280:20123065, doi:10.1098/rspb.2012.3065 |
| ATU types against genomes (Bortolini et al. 2017) | None claimed | 596 tale types | coded presence | SI appendix; not checked | doi:10.1073/pnas.1614395114 |
| Berezkin and Duvakin catalogue | None; areal distribution | 2,450 motifs by 945 traditions (2018) `[UNVERIFIED]`; confirmed snapshot in Thuillard et al. 2018: 2,264 motifs, about 40,000 myths, 934 peoples | Russian and English abstracts | licence not found | https://ruthenia.ru/folklore/berezkin/eng.html |
| Canterbury Tales; Monarchia | Inferred or expert stemma | not checked | Middle English; Latin | not checked | doi:10.1038/29667; doi:10.1093/llc/fqn023 |
| Cornets and psaltery; Paleoindian points; Turkmen textiles | Historical records for cornets; otherwise inferred | not checked | artefacts | not checked | doi:10.1086/510463; doi:10.1006/jasc.2001.0681 |

Count of text datasets with a genuinely known tree and downloadable texts: three, in three languages, none with an open licence, all of them verbatim copying.

### 7. What the Batman study already shows about trees

- Tree distance does not recover documented sources: mean rank 5.444 of 12 for Qwen memes, chance 5.778, exact p = 0.4016 (`study/results/robustness/nulls.json` → `f2_trees.source_recovery.memes_qwen3-8b`). TF-IDF on the same texts: mean rank 2.889, p = 0.0044 (`study/results/phylo.json.gz` → `tfidf.adaptation`). Check: both files were parsed and every number in this section matches. The two files give different p values for the same TF-IDF rank because one is a permutation estimate and the other exact: `nulls.json` → `f2_trees.source_recovery.tfidf` has `p_exact` = 0.002869, and `phylo.json.gz` → `models.qwen3-8b.adaptation` has p = 0.4053 for the memes against 0.4016 exact. Quote one file per comparison.
- Two models' trees agree at Mantel r = 0.9783, but trees from random non-meme units already agree at 0.90 to 0.93, and the Qwen meme matrix correlates 0.8408 with TF-IDF (`nulls.json` → `f2_trees.mantel`, `f2_trees.random_unit_floor`).
- The paper says most internal branches are much shorter than the terminal ones and that works group by tone more than by date (`study/paper/paper.md`, Figure 4 caption and section on the tree). That is the description of a star-like, weakly tree-like matrix.
- E2 succeeded only after this whole-profile test failed and a frequency band and hub adjustment were added, and word overlap matched it (`study/README.md`, E2 row and "E2's design followed a failed first test").

## What this means for P1 and P2

**P1 (the unit).** Cultural phylogenetics supplies an outside criterion for a good unit of inheritance: on a tree known from other evidence, the unit should show phylogenetic signal (D near or below 0, lambda near 1), low homoplasy, and should let a blind method recover the tree. Nothing in the literature or the study shows that neuron units meet this criterion better than words. The one tree-based test in the study failed for neurons and passed for TF-IDF. The stemmatology record adds a sobering prior: when descent is by copying, the choice of unit barely matters, and compression or n-grams come close to hand-collated variants. P1 can only earn support from descent that scrambles wording, and no benchmark of that kind exists yet. Evidence for P1 from this literature: none for, weak against.

**P2 (generality).** The phylogenetic version of P2 is: two models' units give the same tree, and that tree is the true one, beyond what shared vocabulary gives. The study's cross-model tree agreement is mostly a vocabulary effect (random-unit floor 0.90 to 0.93 against 0.98). A known-tree benchmark turns this into a sharper test: is each model's tree closer to the truth than the lexical tree is, and do the two models err in the same places? The three artificial traditions happen to be in English, French and Old Finnish, which gives a small built-in cross-language check, though only on copying. Evidence for P2 from this literature: none either way; PhyloLM shows the tools transfer to models themselves.

Conditions under which the premises could plausibly hold: transmission by retelling, translation or change of medium; long texts; many witnesses; units restricted to those that survive rewording (the study found 11 of 31 named memes do so in Qwen, `study/README.md`).

## Experiment seeds

1. **Artificial oral tradition (the headline test).**
   - Question: when descent destroys wording, do neuron memes recover the true tree better than surface and embedding methods?
   - Data: a new tradition built to the Roos and Heikkilä recipe. One public-domain seed story of 600 to 1,000 words; a pre-drawn stemma of 60 to 100 nodes drawn from a birth-death shape (Camps and Randon-Furling); 20 to 30% of nodes withheld; 5 to 8 two-parent nodes; 10 or so translation edges. Each node is a retelling from memory after one reading. Two arms: human volunteers, and LLM writers from a different model family than the reader. Release under CC BY with the true graph sealed until submission.
   - Method: per method, a distance matrix, neighbour joining and a parent-ranking; score with the CASC triplet average sign distance, parent mean rank, and correct identification of the second parent at contaminated nodes. Bootstrap over nodes.
   - Rival to beat: TF-IDF cosine, word-presence Jaccard, normalized compression distance, two sentence-embedding models, and an LLM asked to rank candidate parents.
   - Falsified if: neurons do not beat the best rival on triplet score with a bootstrap interval excluding zero, or if the neuron advantage does not grow as parent-child word overlap falls.
   - Rough cost: 80 to 100 volunteer retellings at 15 minutes each (about 25 person-hours) or a day of agent sessions for the LLM arm; a few GPU hours for extraction.

2. **Scribe-copy floor check on Parzival, Notre Besoin and Heinrichi.**
   - Question: on verbatim copying, do neuron memes at least tie string methods, and do they hold up in French and Old Finnish?
   - Data: the three CASC sets (permission needed from the organisers).
   - Method: same scoring script (`tripled.py`), published numbers as the reference.
   - Rival to beat: the published n-gram, CompLearn, neighbour-joining and parsimony scores in the table above.
   - Falsified if: neurons fall more than 5 points below n-gram clustering on Parzival. A tie is the expected, uninteresting result; a loss says the unit is too coarse to see copying at all.
   - Rough cost: a few hours; texts are tiny.

3. **Per-unit heritability on a known tree.**
   - Question: taken one unit at a time, are neuron memes more heritable than words?
   - Data: the tradition from seed 1 (true tree), then the Batman and Shakespeare documented links.
   - Method: code each unit as present or absent per witness; compute Fritz and Purvis's D for every neuron meme and every word; compare the D distributions with units matched on frequency, as da Silva and Tehrani did for tales. Add their autologistic control with "same writer" in place of spatial neighbour.
   - Rival to beat: single words and word bigrams as characters; embedding dimensions binarised at the median.
   - Falsified if: the median D of neuron memes is not lower than that of frequency-matched words.
   - Rough cost: analysis only once activations exist; under a day.

4. **Tree-likeness audit of the existing Batman matrices (no new model runs).**
   - Question: is the Batman meme matrix tree-like at all, and is it more or less so than the TF-IDF matrix?
   - Data: the 45 by 45 meme distance matrices already in `study/results/phylo.json.gz` (`models.<model>.D`). That file stores only a Newick tree for TF-IDF, so the TF-IDF matrix has to be rebuilt from the committed texts (word counting, no model).
   - Method: delta scores and Q-residuals per work, NeighborNet in SplitsTree; compare against a matrix simulated on a star tree and on the documented-source graph.
   - Rival to beat: the TF-IDF and LSA matrices on the same works.
   - Falsified if: the meme matrix has a delta score no lower than TF-IDF's, or indistinguishable from the star-tree simulation. Given Nunn et al., a good score proves little; a bad score ends the tree framing for franchises.
   - Rough cost: an afternoon of arithmetic on committed outputs.

5. **Borrowing detection at documented two-parent works.**
   - Question: can a method name the second source of a work known to draw on two?
   - Data: contaminated nodes in seed 1 and in Notre Besoin if its truth file marks them; documented multi-source Batman films.
   - Method: after removing the best single parent, rank remaining candidates on residual similarity.
   - Rival to beat: the same procedure with TF-IDF residuals and with an LLM asked directly.
   - Falsified if: neurons name the second parent no more often than TF-IDF.
   - Rough cost: analysis only.

6. **Cross-model tree agreement against truth.**
   - Question: do Qwen and Llama trees agree with each other because both are right, or because both follow vocabulary?
   - Data: seed 1.
   - Method: triplet score of each model's tree against truth; partial Mantel between the two models controlling for the TF-IDF matrix and for the true path-length matrix.
   - Rival to beat: the random-unit floor (0.90 to 0.93 in `nulls.json`) and TF-IDF.
   - Falsified if: agreement between models, after partialling out truth, is as high as before, or neither model beats the random-unit trees on triplet score.
   - Rough cost: analysis only.

## Key sources

| Citation | URL or DOI | Why it matters | Priority |
| --- | --- | --- | --- |
| Roos and Heikkilä 2009, Evaluating methods for computer-assisted stemmatology using artificial benchmark data sets | doi:10.1093/llc/fqp002; https://www.cs.helsinki.fi/u/ttonteri/casc/ | The known-truth benchmark, the scoring rule and published baseline scores | A |
| Spencer, Davidson, Barbrook and Howe 2004, Phylogenetics of artificial manuscripts | doi:10.1016/j.jtbi.2003.11.022 | First known-tree test of phylogenetics on text; the design to copy | A |
| da Silva and Tehrani 2016 | doi:10.1098/rsos.150645 | The paper's own model; D statistic plus autologistic control per trait | A |
| Nunn et al. 2010, Simulating trait evolution for cross-cultural comparison | doi:10.1098/rstb.2010.0009 | RI and CI do not diagnose borrowing | A |
| Greenhill, Currie and Gray 2009 | doi:10.1098/rspb.2008.1944 | Trees are robust to realistic borrowing; simulation template | A |
| Tehrani 2013, The phylogeny of Little Red Riding Hood | doi:10.1371/journal.pone.0078871 | Worked example reporting RI, delta and Q-residual for a story | B |
| Gray, Bryant and Greenhill 2010, On the shape and fabric of human history | doi:10.1098/rstb.2010.0162 | Tree-likeness as a continuum; delta and Q-residual in culture | B |
| Collard, Shennan and Tehrani 2006 | doi:10.1016/j.evolhumbehav.2005.07.003 | Branching versus blending via RI | B |
| Howe and Windram 2011, Phylomemetics | doi:10.1371/journal.pbio.1001069 | Review that shares the project's name; describes blind artificial traditions | B |
| Fritz and Purvis 2010; Pagel 1999; Farris 1989; Holland et al. 2002; Bryant and Moulton 2004 | DOIs in section 1 | Definitions of D, lambda, RI, delta, NeighborNet | B |
| Bennett, Li and Ma 2003; Cilibrasi and Vitányi 2005 | doi:10.1038/scientificamerican0603-76; doi:10.1109/tit.2005.844059 | Compression distance, the cheapest rival; not a ground-truth dataset | B |
| Baret, Macé and Robinson 2006, Testing methods on an artificially created textual tradition | Linguistica Computazionale 24-25:255-281 (pages per the Helsinki wiki; no DOI; https://wiki.helsinki.fi/xwiki/bin/view/stemmatology/Tradition%2C%20artificial/) | Notre Besoin, the artificial tradition with French text | B |
| Camps, Randon-Furling and Godreau 2025, On the transmission of texts: written cultures as complex systems | arXiv 2505.19246, https://arxiv.org/abs/2505.19246 | Realistic stemma shapes and loss rates for simulation | B |
| Acerbi and Stubbersfield 2023; Perez et al. 2024 | doi:10.1073/pnas.2313790120; arXiv 2407.04503, https://arxiv.org/abs/2407.04503 | LLM transmission chains: cheap known-descent text | B |
| Liben-Nowell and Kleinberg 2008 | doi:10.1073/pnas.0708471105 | A real, deep diffusion tree; text does not mutate | C |
| Bortolini et al. 2017; Ross, Greenhill and Atkinson 2013 | doi:10.1073/pnas.1614395114; Proc. R. Soc. B 280:20123065, doi:10.1098/rspb.2012.3065 | Folktale diffusion without trees | C |
| Temkin (Tëmkin) and Eldredge 2007; O'Brien, Darwent and Lyman 2001 | doi:10.1086/510463; doi:10.1006/jasc.2001.0681 | Reticulation in material culture; cladistics on artefacts | C |
| Youngblood 2019; Youngblood, Baraghith and Savage 2020 | doi:10.1371/journal.pone.0211860; arXiv 2011.02460, https://arxiv.org/abs/2011.02460 | Documented borrowing in music; a phylogeny method that allows it | C |
| Berezkin and Duvakin catalogue; Thuillard et al. 2018 | https://ruthenia.ru/folklore/berezkin/eng.html; doi:10.3176/tr.2018.4.05 | Largest motif distribution database | C |
| Arčon, Robnik-Šikonja and Tratnik 2025, Large language models for folktale type automation based on motifs: Cinderella case study | arXiv 2510.18561, https://arxiv.org/abs/2510.18561 | LLM reading as motif coder: the rival in action | C |
| Yax, Oudeyer and Palminteri 2024, PhyloLM | arXiv 2404.04671, https://arxiv.org/abs/2404.04671 | Phylogenetics of models with known families | C |
| Hoenen 2022, Can the language of the collation be translated into the language of the stemma? (row added in the check; cited in the Bottom line) | arXiv 2206.05603, https://arxiv.org/abs/2206.05603 | States that no deep-learning approach to stemmatology existed in 2022 | C |
| Hoenen 2018, Multi Modal Distance (row added in the check; cited in sections 4 and 6) | https://aclanthology.org/L18-1332/ | Sizes of the three artificial traditions | C |
| Marmerola et al. 2016 | doi:10.1371/journal.pone.0167822 | Synthetic text phylogeny trees with known truth | C |

## Fog

- Heinrichi's size: Hoenen's 2018 paper (via a search summary) says 64 witnesses; I recall 67 made and 37 released. Not resolved. Parzival's released count may also be smaller than 21. I did not download any of the benchmark archives; the CASC data page links them under `download/`, and whether those links still work was not tested. (Check, 2026-11-08: mostly resolved. The `.nex` files under `download/` still serve; their headers give 37 witnesses for Heinrichi, 16 for Parzival, 13 for Notre Besoin and 52 for the Legend. `correct.txt` lists 65 nodes for Heinrichi. Only the "67 made" figure is still unconfirmed. The true graph image links `correct_heinrichi.png` and `.dot` on the results page return 403.)
- Licences: the CASC page restricts the data to challenge use and asks for permission. No licence was found for the later Heinrichi release, the Berezkin catalogue, the chain letters, or the music data.
- Notre Besoin's contamination: I believe its true stemma includes two-parent witnesses, but did not confirm it.
- Numeric delta scores for language families in Gray, Bryant and Greenhill 2010 were not retrieved. Collard et al.'s mean RIs are from memory. (Check: the two means, 0.59 and 0.61 with n = 21 each, are now confirmed through Tehrani 2013's citation of them.)
- Dataset sizes not checked: Canterbury Tales, Paleoindian points, cornets, folk-song melodies, Karsdorp's retelling corpus, the transmission-chain studies, Marmerola et al.
- I found no 2025 or 2026 paper that scores embeddings, sparse-autoencoder features or neurons on an artificial tradition. Absence in a 15-minute search is weak evidence; the Studia Stemmatologica workshops and the SHERBET project (a 2026 slide deck asks for a "golden standard") `[UNVERIFIED]` should be checked. (Check: no URL was given for the SHERBET deck and the check had no web search left to find it.)
- Whether human retelling chains with branching and public texts already exist at useful scale (Breithaupt's lab is the most likely source) was not checked. If they do, seed 1 gets much cheaper.
- An LLM-written tradition read by LLM neurons risks circularity: the writer's habits may be exactly what the reader's neurons detect. The human arm is what makes seed 1 convincing.
- The critique of folktale phylogenies by Lajoye, d'Huy and Le Quellec (a comment on Tehrani 2013) is known to me but I could not confirm its reference in this session `[UNVERIFIED]`.
