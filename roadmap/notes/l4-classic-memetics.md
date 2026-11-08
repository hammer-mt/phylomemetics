# l4-classic-memetics: classical and internet memetics as replication targets

Verification note. Citations below were confirmed in this session through Crossref, the arXiv API, the Journal of Memetics archive (cfpm.org), SNAP, Zenodo or Wikipedia, unless marked `[UNVERIFIED]`. The shared web-search budget ran out part-way through, so several details (effect sizes, dataset hosting) are from memory and are marked `[UNVERIFIED]`. Study numbers come from `study/paper/paper.md` (abstract), a committed file. A second-pass citation check on 2026-11-08 changed several entries; every change is listed under "Check log" at the end.

## Bottom line

- **The field's own verdict on why it failed is not "no unit", it is "no added value".** Edmonds (2005), closing the Journal of Memetics, wrote that memetics "has not provided any extra explanatory or predictive power beyond that available without the gene-meme analogy". That is the same test as "beat word overlap". On the Batman corpus the neuron unit has so far failed it for Qwen and passed it narrowly for Llama (`study/paper/paper.md`: p = 0.09 to 0.25 against p = 0.010 to 0.033).
- **A learned-representation unit was already tried in 1997.** Pocklington and Best, and Best, defined memes as latent semantic (SVD) term clusters in Usenet posts. The unit worked and changed nothing. Neurons are a better instrument for the same move, so the burden is a result that words cannot give.
- **Internet memetics succeeded with crude units** (quoted strings, hashtags, n-grams, image hashes) because copy events were logged at scale. Where copying is verbatim, edit distance is close to a perfect instrument and neurons have nothing to add.
- **The critics describe the only regime where neurons can win.** Sperber and Atran say transmission is reconstruction, not copying: the wording changes and the idea is rebuilt. Word overlap fails by construction there. The best targets are therefore paraphrase, translation, adaptation and citation, not copy-paste memes.
- **Best single target: Kuhn, Perc and Helbing (2014).** Millions of documented inheritance links (citations), a published word-level heredity statistic to beat, and open data today.
- **Expect small selection effects.** Weng et al. (2012) reproduce meme popularity with no content differences at all. A selection test on 15 works was never going to detect anything (corrected from "32 works": the lineage has 32 works, but the selection test ran "across the 15 pre-2000 works"; `study/paper/paper.md`, selection results).

## Findings

### A. Classical memetics: what each author claims that can be tested

| Author | Testable claim (my reading) | Status | What the neuron method could do |
| --- | --- | --- | --- |
| Dawkins 1976, *The Selfish Gene* (confirmed via Wikipedia "Memetics") | Memes are replicators; success depends on fidelity, fecundity and longevity; memes travel in co-adapted complexes `[UNVERIFIED wording: the book and the coinage are confirmed, but the three criteria and "co-adapted complexes" are not in the Wikipedia "Memetics" or "Meme" text fetched in the check; from memory of chapter 11]` | Never operationalised by Dawkins | Fidelity = heredity of a unit along a documented link; fecundity = number of descendants carrying it; complexes = units co-inherited more than chance (linkage). The Batman study measured the first only |
| Dennett (*Darwin's Dangerous Idea*, 1995, which Wikipedia "Memetics" cites for the quote; *From Bacteria to Bach and Back*, 2017) | Memes are "the smallest elements that replicate themselves with reliability and fecundity" | A claim about grain size, never measured | Plot heredity against grain: single neuron, merged family, sentence, document. Rival curve: unigram, bigram, phrase. A knee would be the first empirical unit-size result |
| Blackmore 1999, *The Meme Machine* (Wikipedia) | Imitation is the copying mechanism; she distinguishes "copying the instructions" from "copying the product" (confirmed: Wikipedia "Meme", citing *The Meme Machine*); that copying instructions is the higher-fidelity mode is `[UNVERIFIED detail, from memory]` | Contested by Sperber and Atran | LLM chains: paraphrase the text (product) against regenerate from an outline (instructions); count units retained |
| Hull 1988 *Science as a Process* (doi:10.7208/chicago/9780226360492.001.0001); Hull 2000 "Taking memetics seriously: Memetics will be what we make it" (doi:10.1093/acprof:oso/9780192632449.003.0003) | Science is a selection process on conceptual lineages; "Stop talking about Memetics and start doing it" (quoted by Edmonds 2002, as said at the Cambridge meme conference) | Realised by Kuhn et al. 2014 | See target 1 |
| Lynch 1996 *Thought Contagion*; Lynch 1998, "Units, Events and Dynamics in Memetic Evolution", J. Memetics 2 (http://cfpm.org/jom-emit/1998/vol2/lynch_a.html) | Beliefs spread by distinct modes (parental, proselytic, preservational and others) `[UNVERIFIED list, from memory]`; content that instructs its own spread spreads further | Confirmed in one place: Adamic et al. found text sequences that "confer a replicative advantage" and move between memes | Look for a "pass this on" unit and test whether it predicts copy number in chain letters or copypasta. Rival: a keyword list |
| Brodie 1996 *Virus of the Mind* (Wikipedia) | Memes that press evolved "buttons" (danger, food, sex) spread `[UNVERIFIED detail, from memory]` | Supported in spirit by Heath et al. 2001 (disgust) and Stubbersfield et al. 2015 (social and survival information) | Affect and threat units as predictors of retention. Rival: emotion lexicon counts, or an LLM rating the text |
| Aunger 2000 (ed.) *Darwinizing Culture* (doi:10.1093/acprof:oso/9780192632449.001.0001); Aunger 2002 *The Electric Meme* (confirmed via reviews, e.g. doi:10.1017/s0012217300006338) | The meme is a state inside a brain (the "internalist" school; Wikipedia "Memetics"); artefacts are only signals | Untestable in 2002 | The neuron-defined meme is Aunger's programme run on an artificial brain. P2 is the analogue of "the same meme in two brains" |
| Distin 2005 *The Selfish Meme* (doi:10.1017/cbo9780511614286; Crossref lists 2004) | Memes are representational content, carried by representational systems (languages, notations), and the content survives a change of system `[UNVERIFIED detail, from memory]` | Untested | Invariance of a unit under translation and under change of medium. The repo already has `preregistration/translation_protocol.md` |
| Gabora 1997, "The Origin and Evolution of Culture and Creativity", J. Memetics 1 (http://cfpm.org/jom-emit/1997/vol1/gabora_l.html); Gabora 2004 "Ideas are not replicators but minds are" (doi:10.1023/b:biph.0000013234.87103.76) | Ideas are not transmitted intact; the evolving thing is the whole worldview; what is received depends on the receiver | An anti-unit position | Chains with varied receivers: how much variance in retained units is explained by the source, and how much by the receiver? |
| Heylighen 1998 "What makes a meme successful? Selection criteria for cultural evolution", Proc. 15th Int. Congress on Cybernetics (author's copy, fetched in the check: https://pespmc1.vub.ac.be/Papers/MemeticsNamur.html; the Cogprints mirror https://web-archive.southampton.ac.uk/cogprints.org/1132/index.html now answers automated requests with a bot challenge); Chielens and Heylighen 2005 "Operationalization of meme selection criteria" (https://corpora.tika.apache.org/base/docs/govdocs1/907/907198.pdf) `[UNVERIFIED in the check: the host is unreachable from this session and the title is not in Crossref]` | A meme scoring higher on utility, novelty, simplicity, coherence, authority, proselytism (and others) becomes more numerous. Heylighen calls this falsifiable | Only a pilot survey on virus hoaxes `[UNVERIFIED: the paper could not be opened in the check]`; results not seen | Score each criterion from units and predict spread. Rival: an LLM asked to rate each criterion directly, which is cheap and probably strong |

The internalist against externalist split (Lynch, Brodie and Aunger against Gatherer and Benzon; Wikipedia "Memetics") maps directly onto this project. The externalists said: internal states are unobservable, so measure the artefact. Neurons make an internal state observable, but the Batman result so far favours the externalists: counting words in the artefact did as well.

### B. The Journal of Memetics and why it closed

- Ran electronically 1997 to 2005, first hosted at the Centre for Policy Modelling, Manchester Metropolitan University (Wikipedia "Memetics"; archive at http://cfpm.org/jom-emit/).
- Edmonds 2002, "Three Challenges for the Survival of Memetics" (http://cfpm.org/jom-emit/2002/vol6/edmonds_b_letter.html), read in full this session. The challenges: (1) a conclusive case study; (2) a theory of when memetic models are appropriate; (3) a simulation of the emergence of a memetic process.
- Challenge 1 has an explicit checklist, which is the best available specification for a neuron-based study:
  1. a replicator mechanism that is "physical and not in the mind";
  2. faithful replication with low error, "many times over";
  3. unbroken lineages long enough for adaptation;
  4. success "demonstrably correlated to identifiable comparable advantages";
  5. dynamics "numerically consistent" with population genetics, for example the Price equation (Edmonds' words: "Price's covariance and selection theorem").
- Edmonds 2005, "The revealed poverty of the gene-meme analogy – why memetics per se has failed to produce substantive results" (http://cfpm.org/jom-emit/2005/vol9/edmonds_b.html), read in full. The journal "has ceased due to a lack of quality submissions". The challenges "were not met". The narrow gene-meme analogy failed; the broad evolutionary approach continued "under other names, and in other journals".
- Edmonds' own evidence is thin: one plot of yearly paper counts (a Google Scholar series and an ISI series) with sixth-degree polynomial trend lines fitted. Commenters call the plot a statistical mistake (secondary sources, LessWrong thread "Memes?", https://www.lesswrong.com/posts/m8umj7J4QHrwF3SgM/memes: one commenter says the article "is largely based on a statistical mistake", a second agrees the plot "is indeed very weak"). The argument does not depend on the plot.
- Chvaja 2020, "Why Did Memetics Fail? Comparative Case Study", Perspectives on Science 28:542–570 (doi:10.1162/posc_a_00350): the commitment to gene-like units biased memeticists "to work on the ontology of meme units as opposed to hypotheses testing, which was adopted by the gene-culture scholars"; he traces this to the gene-centred argument of Williams and Dawkins and the a priori idea "that there is no evolution without discrete units of selection" (corrected in the check from the Crossref abstract; the earlier wording, "less attractive to empirical researchers", came from a search summary and is not what the abstract says; full paper not read).
- **Blunt reading for Mike.** The paper's opening claim, that memetics stalled because nobody could bound a meme, is one account of three. The others are "no added predictive value" (Edmonds) and "fixation on a gene-like unit while the rival school got on without one" (Chvaja; Boyd and Richerson). A better unit answers only the first.

### C. The critiques

| Critic | Objection | Testable form | Does the Batman study bear on it? |
| --- | --- | --- | --- |
| Sperber 2000, "An objection to the memetic approach to culture" (doi:10.1093/acprof:oso/9780192632449.003.0008); Claidière, Scott-Phillips and Sperber 2014, "How Darwinian is cultural evolution?" (doi:10.1098/rstb.2013.0368) | Items are re-produced by inference, not copied. Stability comes from attractors, not fidelity. Resemblance plus causation is not inheritance: the copy must get its properties from the source `[UNVERIFIED wording, from memory]` | Seed chains with different stories. Units that appear in chains that never had them are attractors; units that persist only where seeded are inherited | Yes, and it cuts both ways. Heredity (source comic beats other comics) is source-specific, which is what Sperber demands. The identity null is an attractor reading: the name-free Batman core "is close to the standard definition of a superhero" (`study/paper/paper.md`) |
| Atran 2001, "The trouble with memes" (doi:10.1007/s12110-001-1003-0) | Minds infer rather than imitate; surface mutation is so high that there is no replicator; stability comes from evolved cognitive modules | Low surface fidelity with high retention of module-fitting content | Not tested |
| Bloch 2000, "A well-disposed social anthropologist's problems with memes" (doi:10.1093/acprof:oso/9780192632449.003.0010) | Culture does not come in discrete bits; unit boundaries are imposed by the observer `[UNVERIFIED detail, from memory]` | Do independent observers carve the same units? | This is P2. The study's answer: yes for name and word detectors, much less so for concepts |
| Kuper 2000, "If memes are the answer, what is the question?" (doi:10.1093/acprof:oso/9780192632449.003.0009) | Memetics explains nothing that history and anthropology do not already | The added-value test | Same as Edmonds |
| Boyd and Richerson 2000, "Memes: universal acid or a better mousetrap?" (doi:10.1093/acprof:oso/9780192632449.003.0007); Henrich and Boyd 2002 (doi:10.1163/156853702320281836) | Cumulative cultural evolution needs neither replicators nor discrete units | Models, not data | Undercuts the premise that a unit is required at all |

### D. Empirical internet memetics

| Study | Unit | Testable claim | Data public? | Replicate or extend with neurons |
| --- | --- | --- | --- | --- |
| Pocklington and Best 1997, "Cultural Evolution and Units of Selection in Replicating Text", J. Theor. Biol. 188:79–87 (doi:10.1006/jtbi.1997.0460); Best 1997, "Models for Interacting Populations of Memes: Competition and Niche Behavior", J. Memetics 1 (http://cfpm.org/jom-emit/1997/vol1/best_ml.html) | Latent semantic term clusters in Usenet posts (Best's page: Latent Semantic Indexing by SVD on a "NetNews" corpus) | Such clusters satisfy the conditions for selection; populations of them compete | Usenet archives exist; their corpus `[UNVERIFIED]` | The direct ancestor of this project. Rerun with neurons and with LSI side by side |
| Leskovec, Backstrom and Kleinberg 2009 (doi:10.1145/1557019.1557077) | Clusters of quoted-phrase variants | The news cycle has regular rise and fall; blogs lag mainstream media by hours `[UNVERIFIED number, from memory: 2.5 h; still unconfirmed after the check, because the paper PDF could not be fetched and Crossref holds no abstract]` | **Yes.** SNAP, https://snap.stanford.edu/data/memetracker9.html: 96,608,034 documents, 210,999,824 "memes" (SNAP's own label; the page also says it tracks more than 17 million different phrases, so this figure counts phrase occurrences) and 418,237,269 links, Aug 2008 to Apr 2009. Quotes and links only, no article text | Cluster variants by neuron signature, not string distance. Weak target on its own: no ground truth for paraphrased variants |
| Simmons, Adamic and Adar 2011, "Memes Online: Extracted, Subtracted, Injected, and Recollected" (doi:10.1609/icwsm.v5i1.14120) | Quote variants in the same data `[UNVERIFIED that the data are the MemeTracker set: the abstract does not name the dataset]` | Changes to quoted text are frequent, and their likelihood depends "on the authority of the copied source and the type of site that is copying" (abstract; corrected in the check from "source type and quote properties") | Yes, by inheritance from SNAP `[UNVERIFIED, same caveat]` | Split mutations into meaning-preserving and meaning-changing by unit change; test which spread |
| Lerique and Roth 2018, "The semantic drift of quotations in blogspace: a case study in short-term cultural evolution", Cognitive Science 42:188–219, online 2017 (doi:10.1111/cogs.12494) | Word substitutions inside quotes | Substitutions are biased towards easier words, consistent with attraction (confirmed from the abstract: words "harder to recall in lists have a higher tendency to be substituted, and words easier to recall are produced instead") | Yes, same SNAP data `[UNVERIFIED: the abstract does not name the dataset; the Crossref reference list does cite Leskovec et al. 2009 and Simmons et al. 2011]` | Thousands of natural rewording pairs: does the unit survive when the word changes? See seed 2 |
| Adamic, Lento, Adar and Ng 2014/2016 (arXiv:1402.6792; doi:10.1145/2835776.2835827) | Copy-pasted Facebook status text | From the abstract: variant populations follow a Yule process set by the mutation rate; edit distance grows with distance in the cascade; some sequences confer a replicative advantage and transfer "laterally"; subpopulations prefer particular variants | **No.** Facebook-internal; I found no public release | Not replicable on the original. The claims can be tested on public copy-paste text. Verbatim regime: words will win |
| Weng, Flammini, Vespignani and Menczer 2012 (doi:10.1038/srep00335) | Hashtag | Limited attention plus network structure reproduces the heavy tails of meme popularity and lifetime, with no intrinsic differences between memes | `[UNVERIFIED]`; the Indiana data page I tried returned 404 | Use as the null. Any selection claim must beat a content-free model |
| Weng, Menczer and Ahn 2013 (doi:10.1038/srep02522) | Hashtag | Early spread across communities predicts virality | `[UNVERIFIED]` (was hosted at Indiana) | Baseline for "does content add to structure?" |
| Coscia 2013 (arXiv:1304.1712; doi:10.1609/icwsm.v7i1.14385); Coscia 2014 "Average is boring" (doi:10.1038/srep06477) | Image-macro templates on Quickmeme | Memes compete and collaborate `[UNVERIFIED: not in the 2013 abstract]`; memes similar to many others are less likely to succeed (2014 abstract: "memes with higher similarity to other memes have a significant disadvantage in their potential popularity") | `[UNVERIFIED]`; not found on the author's data pages I fetched | Recompute similarity with neuron signatures; test distinctiveness against success |
| Shifman 2013 (doi:10.1111/jcc4.12013); Shifman 2014 *Memes in Digital Culture* (doi:10.7551/mitpress/9429.001.0001; Crossref lists 2013) | Groups of items sharing content, form and stance, made with awareness of each other | The three dimensions vary and are inherited separately | Conceptual, no dataset | Do neuron families sort into content, form and stance? |
| Segev, Nissenbaum, Stolero and Shifman 2015 (doi:10.1111/jcc4.12120) | Meme families from Know Your Meme | Families cohere around a "quiddity"; cohesiveness relates to how concrete it is (from the title) | KYM pages are public; their coded sample `[UNVERIFIED]` | Predict family membership from units |
| Bauckhage 2011 (doi:10.1609/icwsm.v5i1.14097); Bauckhage, Kersting and Hadiji 2013 (doi:10.1609/icwsm.v7i1.14392) | Search-interest time series of famous memes | Epidemic and fad models fit the rise and fall | Google Trends, re-queryable | Poor target: content-free time series |
| Kuhn, Perc and Helbing 2014 (arXiv:1404.3757; doi:10.1103/physrevx.4.041036) | n-grams in titles and abstracts | A meme score (frequency times propagation along citations) picks out real scientific concepts; validated on about 50 million records from Web of Science, PubMed Central and APS, with human annotators and randomised networks (abstract) | Partly: PubMed Central open; APS on request; Web of Science proprietary. Open substitutes exist now (OpenAlex, arXiv) `[UNVERIFIED that these suffice]` | **Best target.** See seed 1 |
| Chavalarias and Cointet 2013 (doi:10.1371/journal.pone.0054847); Chavalarias, Lobbé and Delanoë 2021 "Draw me Science" (doi:10.1007/s11192-021-04186-5) | Co-word clusters linked through time | Fields emerge, branch, merge and decline in recurring patterns | Source corpus `[UNVERIFIED]`; method reimplementable | Build the same lattice from neuron families and compare branches |
| Zannettou et al. 2018 (arXiv:1805.12512; doi:10.1145/3278532.3278550) | Perceptual-hash image clusters labelled with KYM | 160M images from 2.6B posts; fringe communities are disproportionate sources (abstract) | **Yes.** https://zenodo.org/record/3699670 (doi:10.5281/zenodo.3699670; open access) | Image neurons against perceptual hash and CLIP embedding. Ties to `images/` |
| Ling et al. 2021 (arXiv:2101.06535; doi:10.1145/3449155) | Image memes on /pol/ | A codebook of visual features separates viral from non-viral (abstract: 100 annotated memes; classifier AUC 0.866) | `[UNVERIFIED]` | Do vision neurons recover the codebook? |
| Tommasini, Ilievski and Wijesiriwardene 2023, IMKG (doi:10.1007/978-3-031-33455-9_21) | KYM entries as a knowledge graph | Resource, no claim | Public by design `[UNVERIFIED URL]` | A source of documented parent and child relations |

### E. Adjacent studies that are better targets than most of the list above

They control for the confounds that sank content-based prediction elsewhere.

- **Tan, Lee and Pang 2014** (arXiv:1405.1438; doi:10.3115/v1/p14-1017). Pairs of tweets by the same author with the same URL and different wording: which gets more retweets? Author and topic are held fixed. Data released by the authors; hosting not confirmed this session `[UNVERIFIED]`. Human and classifier accuracies are in the paper: average human accuracy 61.3%; the authors' classifier 66.5% in cross-validation and 65.6% held-out (confirmed in the check from the arXiv PDF; the earlier "about 61% and 66%" was from memory).
- **Danescu-Niculescu-Mizil, Cheng, Kleinberg and Lee 2012**, "You had me at hello: how phrasing affects memorability", ACL 2012 (arXiv:1203.6360; https://aclanthology.org/P12-1094/). Memorable movie quotes matched to non-memorable lines by the same speaker in the same scene. Memorable quotes use less common words on a common syntactic scaffold (abstract). Data released by the authors; hosting not confirmed `[UNVERIFIED]`.
- **Lakkaraju, McAuley and Leskovec 2013** (doi:10.1609/icwsm.v7i1.14408). The same image resubmitted to Reddit with different titles. Public: https://snap.stanford.edu/data/web-Reddit.html, 132,308 submissions of 16,736 images, July 2008 to Jan 2013.
- **Heath, Bell and Sternberg 2001** (doi:10.1037/0022-3514.81.6.1028): urban legends are selected for emotion, disgust in particular. **Stubbersfield, Tehrani and Flynn 2015** (doi:10.1111/bjop.12073): social and survival information is retained in chains. **Acerbi and Stubbersfield 2023** (doi:10.1073/pnas.2313790120): an LLM shows the same content biases in transmission chains (abstract: "ChatGPT-3", five preregistered experiments, biases "analogous to humans").
- **Bennett, Li and Ma 2003** (doi:10.1038/scientificamerican0603-76): a phylogeny of chain-letter variants. **Liben-Nowell and Kleinberg 2008** (doi:10.1073/pnas.0708471105): chain-letter trees are deep and narrow.
- Berger and Milkman 2012 (doi:10.1509/jmr.10.0353): arousal predicts sharing of news articles.
- Cheng et al. 2014, "Can cascades be predicted?" (doi:10.1145/2566486.2567997) and Gleeson et al. 2014 (doi:10.1103/physrevlett.112.048701): spread structure and neutral competition explain most of the variance in popularity `[UNVERIFIED detail, from memory]`.

### F. 2024 to 2026 work

- Perez et al. 2024, "When LLMs play the telephone game" (arXiv:2407.04503): LLM chains drift to attractors.
- Brinkmann et al. 2023, "Machine culture" (doi:10.1038/s41562-023-01742-2).
- Milička and Bednářová 2026, "Sydney telling fables on AI and humans: a corpus tracing memetic transfer of persona between LLMs" (arXiv:2602.22481).
- Guo, Wu and Yiu 2026, "Model collapse as cultural evolution" (arXiv:2605.23054).
- Acerbi 2026, "Associations between content and engagement vary across social media communities" (doi:10.31235/osf.io/xwktf_v1; title only seen). If the title holds, selection on content depends on the community. Added in the check, from the Crossref abstract: 1,078,044 items from 587 communities in five corpora (Reddit, Stack Exchange, X, YouTube); models with community-varying slopes fit better in all five, the slopes vary in size and often in direction, and the average slopes stay small. It is a preprint.
- Benzon 2025, "Tracking the 'Xanadu' meme" (doi:10.2139/ssrn.5346920; title only seen).
- In the one search I could run, I found **no** 2025 or 2026 paper that uses neurons or sparse-autoencoder features as the unit of inheritance on real lineage data. One search is weak evidence of absence.

### G. Ranked: the ten best replication targets

Ranked on public data, documented inheritance links, a published word-level baseline, room for a unit to beat words, and cost.

1. **Kuhn, Perc and Helbing 2014**: scientific memes along citations.
2. **Lerique and Roth 2018, with Simmons et al. 2011**, on the public MemeTracker quotes `[UNVERIFIED that both papers used this dataset; see table D]`: does the idea survive when the word is swapped?
3. **Tan, Lee and Pang 2014**: wording and propagation with author and topic held fixed.
4. **Transmission chains: Acerbi and Stubbersfield 2023, Heath et al. 2001, Stubbersfield et al. 2015**, used to test Sperber's attraction against inheritance, and to test the Heylighen, Lynch and Brodie selection criteria.
5. **Know Your Meme genealogies** (Segev et al. 2015; IMKG; Zannettou et al. 2018): documented parent and child memes.
6. **Danescu-Niculescu-Mizil et al. 2012**: memorable against non-memorable quotes.
7. **Adamic et al. 2014/2016**: Yule process, mutation with distance, lateral transfer. Original data closed; test the claims on public copy-paste text.
8. **Coscia 2014**: similarity kills success.
9. **Chavalarias and Cointet 2013**: rebuild a phylomemy from neuron families.
10. **Lakkaraju et al. 2013**: Reddit resubmissions, as a second selection test.

Not replication targets, but to adopt: **Weng et al. 2012** as the null model, and **Edmonds 2002, Challenge 1**, as the checklist.

Poor targets: Bauckhage (content-free time series), Leskovec et al. 2009 on its own (verbatim strings), Shifman (conceptual).

## What this means for P1 and P2

**P1 (the unit).** Nothing in this literature tests P1 directly. It does three things.

- It sets the bar. Edmonds' "added value" and Kuper's "what is the question?" both ask what the unit predicts that could not be predicted without it. On the Batman corpus the answer is: little for Qwen, a small amount for Llama (`study/paper/paper.md`).
- It supplies a discouraging precedent. A statistically learned unit (Pocklington and Best 1997) did not rescue memetics.
- It tells us where P1 can matter. Where copies are verbatim, crude units already work. P1 earns its keep only if neurons track an idea through rewording, translation or adaptation. The study found only "a few units, such as a funeral, a coin flip and Llama's revenge unit" that clearly survive rewording (`study/paper/paper.md`). That is the weak point, and targets 1, 2 and 4 are aimed at it.

One conceptual point: under Edmonds' checklist the replicator must be "physical and not in the mind". The neuron is then the measuring instrument, and the text is the replicator. That framing is easier to defend than "the neuron is the meme".

**P2 (generality).** Bloch's and Kuper's objection, that unit boundaries are imposed by the observer, is the classical form of the question P2 answers. Agreement between independently trained models is the only operational reply on offer. The study's evidence for it is mostly lexical: twins have median r = 0.82, but "most of them detect names and words" and "without names the agreement collapses" (`study/paper/paper.md`). So P2 as evidenced shows that models agree on words, which nobody doubted. No source in this topic adds evidence for P2. A fairer test would compare model-to-model agreement with model-to-human agreement on hand-coded motifs.

**Conditions.** P1 and P2 are most likely to hold, and to matter, when (a) transmission is reconstructive, (b) inheritance links are documented independently of the text, and (c) there are thousands of links. Batman met (b) only.

## Experiment seeds

1. **Scientific memes without the words.**
   - Question: is a neuron unit inherited along citation links beyond what shared n-grams explain?
   - Data: open abstracts with citation links (PubMed Central open-access subset, or arXiv with OpenAlex citations).
   - Method: compute Kuhn's propagation ratio (unit present given a cited paper has it, against present given no cited paper has it) for neuron families and for n-grams; then test units on citing pairs after removing shared content words.
   - Rival to beat: Kuhn's n-gram meme score; TF-IDF; sentence embeddings.
   - Falsified if: neuron units give no gain over n-grams in predicting held-out citation links once shared vocabulary is controlled.
   - Cost: low to moderate; about 100,000 abstracts through an 8B model, a few GPU-days at most.

2. **Does the idea survive the swap?**
   - Question: when one word in a quote is replaced in transmission, do the concept units stay on?
   - Data: MemeTracker quote variants (SNAP), using Lerique and Roth's substitution pairs.
   - Method: compare unit activations before and after substitution; separate meaning-preserving from meaning-changing swaps; test which variants out-spread their parent.
   - Rival: sentence-embedding cosine; WordNet synonymy; an LLM judging "same meaning?".
   - Falsified if: unit conservation predicts neither human or LLM meaning judgements nor spread better than embedding cosine.
   - Cost: low; short strings.

3. **Attractor or inheritance (Sperber's test).**
   - Question: in transmission chains, which units persist only where seeded, and which appear regardless of seed?
   - Data: Stubbersfield's and Acerbi's chain materials; new LLM chains with several seeds.
   - Method: for each unit, estimate seeded persistence against unseeded appearance.
   - Rival: the same two rates computed for content words and for embedding clusters.
   - Falsified if: the unit-level split into inherited and attracted is no cleaner than the word-level split.
   - Cost: low.

4. **Wording as fitness, with everything else fixed.**
   - Question: can units pick the more-retweeted of two tweets by the same author about the same link?
   - Data: Tan, Lee and Pang pairs.
   - Method: paired classifier on unit differences, on the authors' split.
   - Rival: their published features and bag-of-words; a zero-shot LLM; the human accuracy they report.
   - Falsified if: units do not beat bag-of-words.
   - Cost: very low.

5. **Know Your Meme family tree.**
   - Question: do units recover documented parent and child relations between memes?
   - Data: KYM entries and relations (IMKG); image clusters from Zannettou et al.
   - Method: the Batman heredity test at scale: does a child carry its documented parent's optional units more than other memes' units?
   - Rival: word overlap of entry text; perceptual hash and CLIP embedding for images.
   - Falsified if: word overlap or image hash finds parents as well.
   - Cost: low for text; moderate for images.

6. **Edmonds' Challenge 1, items 4 and 5.**
   - Question: do unit frequencies across generations satisfy a Price-equation decomposition with a non-zero selection term?
   - Data: any corpus from seeds 1, 2 or 5 with generations and copy counts.
   - Method: Price decomposition per unit; compare the selection term with a neutral simulation in the style of Weng et al.
   - Rival: the same decomposition on n-grams.
   - Falsified if: selection terms are indistinguishable from the neutral model.
   - Cost: low; analysis only.

7. **How big is a meme (Dennett's grain).**
   - Question: is there a grain at which heredity per unit peaks?
   - Data: seeds 1 or 5.
   - Method: heredity statistic at single neuron, merged family, sentence and document grain.
   - Rival: the same curve for unigram, bigram, phrase and topic.
   - Falsified if: the curve is flat or matches the n-gram curve.
   - Cost: low.

## Key sources

| Citation | URL or DOI | Why it matters | Priority |
| --- | --- | --- | --- |
| Edmonds 2005, "The revealed poverty of the gene-meme analogy", J. Memetics 9 | http://cfpm.org/jom-emit/2005/vol9/edmonds_b.html | The field's own post-mortem: no added value | A |
| Edmonds 2002, "Three challenges for the survival of memetics", J. Memetics 6 | http://cfpm.org/jom-emit/2002/vol6/edmonds_b_letter.html | A five-item checklist for a convincing case study | A |
| Kuhn, Perc and Helbing 2014, Phys. Rev. X 4:041036 | doi:10.1103/physrevx.4.041036; arXiv:1404.3757 | Top target: heredity statistic on citation links with a word baseline | A |
| Pocklington and Best 1997, J. Theor. Biol. 188:79–87 | doi:10.1006/jtbi.1997.0460 | A learned latent unit as meme, 29 years ago | A |
| Sperber 2000, in *Darwinizing Culture* | doi:10.1093/acprof:oso/9780192632449.003.0008 | Attraction against inheritance; defines where neurons could beat words | A |
| Lerique and Roth 2018, Cognitive Science 42:188–219 | doi:10.1111/cogs.12494 | Natural rewording pairs on public data | A |
| Leskovec, Backstrom and Kleinberg 2009, KDD; SNAP data | doi:10.1145/1557019.1557077; https://snap.stanford.edu/data/memetracker9.html | Largest public meme-variant corpus `[UNVERIFIED superlative; the size itself is confirmed on the SNAP page]` | A |
| Tan, Lee and Pang 2014, ACL | doi:10.3115/v1/p14-1017; arXiv:1405.1438 | Selection with author and topic fixed | A |
| Adamic, Lento, Adar and Ng 2016, WSDM | doi:10.1145/2835776.2835827; arXiv:1402.6792 | Cleanest evidence of mutation and selection in text; data closed | B |
| Weng et al. 2012, Sci. Rep. | doi:10.1038/srep00335 | The content-free null | B |
| Acerbi and Stubbersfield 2023, PNAS | doi:10.1073/pnas.2313790120 | LLM chains reproduce human content biases | B |
| Atran 2001, Human Nature | doi:10.1007/s12110-001-1003-0 | Inference, not imitation | B |
| Aunger (ed.) 2000, *Darwinizing Culture* | doi:10.1093/acprof:oso/9780192632449.001.0001 | Holds the Sperber, Bloch, Kuper, Boyd and Richerson, and Hull chapters | B |
| Heylighen 1998; Chielens and Heylighen 2005 `[UNVERIFIED in the check]` | https://pespmc1.vub.ac.be/Papers/MemeticsNamur.html (Heylighen 1998, fetched in the check); https://web-archive.southampton.ac.uk/cogprints.org/1132/index.html (bot challenge to automated requests) | A ready list of selection hypotheses | B |
| Chavalarias and Cointet 2013, PLoS ONE | doi:10.1371/journal.pone.0054847 | Uses "phylomemetic patterns" in its title; word-based rival. Corrected in the check: it is not the origin of the word. Howe and Windram 2011, "Phylomemetics—Evolutionary Analysis beyond the Gene", PLoS Biology (doi:10.1371/journal.pbio.1001069), is two years earlier; who coined it first is `[UNVERIFIED]` | B |
| Coscia 2014, Sci. Rep. | doi:10.1038/srep06477 | Distinctiveness and success | B |
| Zannettou et al. 2018, IMC; Zenodo data | doi:10.1145/3278532.3278550; arXiv:1805.12512; https://zenodo.org/record/3699670 (doi:10.5281/zenodo.3699670) | Public image-meme clusters with KYM labels | B |
| Segev et al. 2015, JCMC | doi:10.1111/jcc4.12120 | Meme families from KYM | C |
| Chvaja 2020, Perspectives on Science | doi:10.1162/posc_a_00350 | A comparative genealogy of the failure, set against gene-culture coevolution (the earlier label "a historian's account" was not checked) | C |
| Danescu-Niculescu-Mizil et al. 2012, ACL | arXiv:1203.6360; https://aclanthology.org/P12-1094/ | Memorability with matched controls | C |
| Lakkaraju, McAuley and Leskovec 2013, ICWSM | doi:10.1609/icwsm.v7i1.14408; https://snap.stanford.edu/data/web-Reddit.html | Public resubmission data | C |

## Fog

- The shared web-search budget ran out. I could not confirm hosting for the Tan et al., Cornell movie-quote, Coscia, Weng, Ling and IMKG datasets. Check each before planning around it.
- Everything marked `[UNVERIFIED]` is from memory: the 2.5-hour blog lag, the Tan accuracies, Lynch's modes, Brodie's buttons, Blackmore's product and instruction distinction, Distin's and Bloch's detailed positions, the details of Simmons et al. and of Lerique and Roth.
- I did not read the full texts of Sperber, Bloch, Kuper, Atran, Chvaja, Chielens and Heylighen, or any of the books. Their positions are from abstracts, secondary summaries and memory.
- Whether open scholarly data (OpenAlex, arXiv) have enough abstract coverage and clean enough citation links to rerun Kuhn et al. is unchecked.
- Is there any public analogue of the Facebook copy-paste data with cascade structure? I do not know of one.
- Know Your Meme's terms of use for scraping, and how complete its parent and child links are, are unchecked.
- Whether anyone has used neuron or sparse-autoencoder features as cultural units on diffusion data in 2025 or 2026: one search found nothing. It needs a proper search.
- Open design question: in verbatim-copy corpora, is there any outcome at all on which a neuron unit could beat edit distance? If not, half the classic internet datasets are useful only as calibration.
- Added by the second-pass check (2026-11-08). Now confirmed, so no longer fog: the Tan accuracies (61.3% human; 66.5% and 65.6% classifier), the claims of Simmons et al. and of Lerique and Roth (from their abstracts), Blackmore's instructions and product distinction (but not which mode has higher fidelity), and Chvaja's argument (from the abstract). Still open after the check: the 2.5-hour blog lag; Chielens and Heylighen 2005 (existence, venue and the virus-hoax pilot); Dawkins' exact three criteria; whether Simmons et al. and Lerique and Roth worked on the public MemeTracker data, which target 2 and seed 2 depend on; dataset hosting for Tan et al., the Cornell movie quotes, Coscia, Weng, Ling and IMKG.
- The check ran without web search (the shared budget was already used up) and four hosts could not be fetched: corpora.tika.apache.org (outside the session's allowed network), cs.cornell.edu and cs.stanford.edu (TLS errors), and the Cogprints mirror (bot challenge). github.com returned 403 for the guessed IMKG repository address.

## Check log

Second-pass check, 2026-11-08. Methods: Crossref API for all 46 DOIs, arXiv abstract pages for all 10 arXiv IDs, direct fetches of the Journal of Memetics pages, SNAP, Zenodo, LessWrong, Wikipedia ("Memetics", "Meme"), nature.com, the ACL Anthology, Heylighen's own site, the Tan et al. arXiv PDF, and `study/paper/paper.md` for study numbers. No web search was available.

Corrections to facts

1. Bottom line, last bullet: "A selection test on 32 works" changed to 15 works. `study/paper/paper.md` (selection results) says the test ran "across the 15 pre-2000 works"; 32 is the size of the whole lineage.
2. Key sources, Chavalarias and Cointet 2013: "Origin of the word 'phylomemetic'" was wrong. Howe and Windram 2011, "Phylomemetics—Evolutionary Analysis beyond the Gene" (doi:10.1371/journal.pbio.1001069, confirmed in Crossref) is earlier. Added, with first coinage marked `[UNVERIFIED]`.
3. Table D, Simmons, Adamic and Adar 2011: the claim "depends on source type and quote properties" replaced by the abstract's wording (authority of the copied source and type of copying site). Title added.
4. Section B, Chvaja 2020: "less attractive to empirical researchers" replaced by what the Crossref abstract says (memeticists worked on the ontology of units, not on hypothesis testing; traced to Williams and Dawkins). Volume and pages added. Key sources label "a historian's account" reworded, since the author's discipline was not checked.
5. Table D, MemeTracker: "210,999,824 phrase mentions" relabelled. SNAP calls this figure "Number of memes"; the 418,237,269 links figure added.
6. Section E, Tan, Lee and Pang 2014: "about 61% and 66%" replaced by the paper's figures (61.3% average human; 66.5% cross-validation and 65.6% held-out for the classifier). `[UNVERIFIED]` removed for the numbers; data hosting stays `[UNVERIFIED]`.
7. Section B: the journal was "first hosted" at the Centre for Policy Modelling (Wikipedia's wording), not simply "hosted".
8. Section B: the sixth-degree polynomial is fitted to two series (Google Scholar and ISI), and the LessWrong criticism is two commenters; wording and quotes added.

`[UNVERIFIED]` marks removed or narrowed

9. Table D, Lerique and Roth 2018: claim confirmed from the abstract (harder-to-recall words are substituted, easier-to-recall words produced). Full title, volume, pages and online year added.
10. Table A, Blackmore: the "copying the instructions" against "copying the product" distinction is confirmed (Wikipedia "Meme", citing *The Meme Machine*). Only the fidelity ranking stays `[UNVERIFIED]`.
11. Section F, Acerbi 2026: abstract now read; sample sizes and the main result added. The original "title only seen" sentence is kept.

`[UNVERIFIED]` marks added

12. Table A, Dawkins: the three criteria and "co-adapted complexes" are not in the Wikipedia text the notes cite.
13. Table A and Key sources, Chielens and Heylighen 2005: the URL could not be fetched and Crossref has no such title; the "pilot survey on virus hoaxes" claim is marked too.
14. Table D, Simmons et al. and Lerique and Roth: that they used the public MemeTracker data is not in either abstract. Lerique and Roth's reference list does cite Leskovec et al. 2009. Same mark added to target 2 in section G.
15. Table D, Coscia 2013: "memes compete and collaborate" is not in the abstract.
16. Table D, Leskovec et al. 2009: the 2.5-hour lag stays `[UNVERIFIED]`, with the reason.
17. Key sources, Leskovec et al. 2009: "largest public meme-variant corpus" marked as an unverified superlative.

URLs, DOIs and titles added

18. Heylighen 1998: working URL https://pespmc1.vub.ac.be/Papers/MemeticsNamur.html, full title and venue. The page confirms the criteria and the "falsifiable hypothesis" sentence. The Cogprints URL is kept with a note.
19. Zannettou et al. 2018: Zenodo DOI 10.5281/zenodo.3699670 (open access) and arXiv ID in Key sources.
20. Danescu-Niculescu-Mizil et al. 2012: title and https://aclanthology.org/P12-1094/.
21. Lakkaraju et al. 2013: DOI in Key sources; dataset timespan.
22. arXiv IDs added to the Key sources rows for Kuhn et al., Tan et al. and Adamic et al.; page ranges for Pocklington and Best and for Lerique and Roth.
23. Titles added for Lynch 1998, Gabora 1997, Best 1997, Pocklington and Best 1997, Hull 2000 and Edmonds 2005 (subtitle).
24. Small precision notes: Dennett's quote is sourced to *Darwin's Dangerous Idea* (1995); Hull's remark was made at the Cambridge meme conference; Edmonds' item 5 names "Price's covariance and selection theorem"; Crossref year differences noted for Distin (2004) and Shifman (2013); Ling et al. detail (100 memes, AUC 0.866); Acerbi and Stubbersfield detail (ChatGPT-3, five preregistered experiments).

Confirmed with no change

- All 46 DOIs resolve in Crossref to the stated authors and titles. Year differences are online against print dates only (the *Darwinizing Culture* chapters show 2001 online for the 2000 book; the ICWSM papers show 2021 because AAAI registered them late; Stubbersfield et al. shows 2014 online for the 2015 volume).
- All 10 arXiv IDs match, including the 2026 ones (2602.22481, 2605.23054).
- Every Edmonds 2002 and 2005 quotation is verbatim on the cfpm.org pages.
- Kuhn et al. ("close to 50 million publication records"), Adamic et al. (Yule process, edit distance, replicative advantage, "laterally", subpopulations), Zannettou et al. (160M images, 2.6B posts), Weng et al. 2012 ("without the need to assume different intrinsic values among ideas"), Coscia 2014 and Danescu-Niculescu-Mizil et al. match their abstracts.
- SNAP Reddit figures (132,308 submissions, 16,736 images) and MemeTracker document count match.
- Study numbers and quotations (p = 0.09 to 0.25; p = 0.010 to 0.033; median r = 0.82; the superhero, funeral, names-and-words and agreement-collapses phrases) match `study/paper/paper.md`.
- Wikipedia "Memetics" supports the journal dates, Dennett's quote, and the internalist and externalist line-up (Lynch and Brodie, with Aunger's 2002 book as "an advanced statement of the internalist school"; Gatherer and Benzon).
