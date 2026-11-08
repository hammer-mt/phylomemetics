# l8-girard-remix-influence: influence, originality and remix

Nothing was run for these notes: repo files were read, and sources were checked by web search, the arXiv API, Crossref DOI lookups and publisher pages. Details marked `[mem]` are from memory of a paper whose existence was confirmed this session but whose text was not re-read. `[UNVERIFIED]` means the source itself could not be confirmed.

## Bottom line
- The novelty-and-influence literature already has a working recipe, and it runs on cheap units (journal pairs, patent codes, keywords, LDA topics, function words, TF-IDF): **novelty = distance from the past, influence = similarity of the future to you** (Kelly et al. 2021; Barron et al. 2018; Park et al. 2020; Elgammal and Saleh 2015). Neuron memes are a new unit for that recipe. They are not a new method, and they have to beat those units to be worth their cost.
- The Batman study has not shown that they do. On declared sources, TF-IDF found the source as well as the neurons (11% of comics beat the source, against 16% for Qwen and 10% for Llama; `study/results/experiments.json`), and on raw tree distance the neurons were at chance while TF-IDF was not (`study/paper/paper.md` section 3.5). There is no evidence yet that neuron memes detect **undeclared** influence at all.
- Similarity is not influence. Almost every study here measures resemblance and calls it influence. The only designs that identify influence are the ones where it is known by construction (planted sources, declared remix links, fine-tuning on a known exemplar). Training-data attribution learned this the hard way, and BM25 still beats gradient influence at finding the passage that contains the fact (Chang et al. 2024).
- Girard against Dawkins has never been tested head to head. It becomes testable as **who you copy** (model-based bias) against **what gets copied** (content-based bias). Neuron memes can only measure the "what". Girard needs exposure data: who saw whom.
- "Everything is a remix" cannot fail as stated. It becomes a test once written as a ledger: what share of a work's units appeared earlier, what share is new units, and what share is new pairings of old units. The ledger is only as good as the reference pool and the noise floor; with about 114 works, "new" means nothing.
- Most headline findings in this area are contested or fragile: Uzzi's atypical combinations (field confound), Berlyne's inverted U (mixed record), the six story shapes, Hughes's "declining influence" (style similarity is not influence), and the decline in disruptiveness (dataset artefact claim). Treat them as designs to borrow and not as settled facts.

## Findings

### A. Girard against Dawkins
- Dawkins: the meme is the replicator; selection acts on the content, and carriers are vehicles (cited in `study/paper/paper.md` ref [1]). Prediction: a meme's spread depends on its own properties (fidelity, fecundity, longevity) and should be roughly stable across who displays it.
- Girard: people copy each other's **desires**, through a model; shared desire for the same object produces rivalry, rivals grow alike ("doubles"), and the crisis is discharged on a scapegoat. Overview: Internet Encyclopedia of Philosophy, https://iep.utm.edu/girard/. Prediction: adoption depends on **who** displays the thing, not on what it is; rivals converge; conflict follows convergence.
- Girard reportedly dismissed Dawkins as having no awareness of mimetic rivalry, crisis or scapegoating (reported in commentary: Burgis, https://read.lukeburgis.com/p/mimetic-desire-102-mimes-memes, which prints the line "Dawkins has no awareness of mimetic rivalry, mimetic crisis, scapegoating and other figures uncovered by mimetic theory" as a Girard quotation and gives no source for it; original passage not located, `[UNVERIFIED]` at source). A sceptical reading of Girard: "Against Girardian mimesis", https://www.greaterwrong.com/posts/tzuArKci3JpN7Zye4/against-girardian-mimesis (canonical copy: https://www.lesswrong.com/posts/tzuArKci3JpN7Zye4/against-girardian-mimesis). I found **no peer-reviewed empirical paper** that pits the two against each other.
- Lab evidence for the mimetic-desire mechanism: Lebreton, Kawa, Forgeot d'Arc, Daunizeau, Pessiglione 2012, "Your goal is mine: unraveling mimetic desires in the human brain", J. Neurosci. 32(21):7146-7157, doi:10.1523/JNEUROSCI.4821-11.2012. Objects were rated as more desirable after being seen as the goal of another agent's action; mirror-system activity modulated the valuation system, and the strength of that coupling predicted individual susceptibility (all three points are in the abstract). Weak points (my reading): single lab, low-stakes objects (sweets, clothes, tools, toys `[mem]`: the object list is not in the abstract), ratings as the outcome, an fMRI connectivity claim, and nothing on rivalry or scapegoating, which is the distinctive part of Girard. The authors frame it as goal contagion (Aarts, Gollwitzer, Hassin 2004, doi:10.1037/0022-3514.87.1.23), which predates and does not need Girard.
- Related value-conformity work, same caveats: Klucharev et al. 2009 (doi:10.1016/j.neuron.2008.11.027); Campbell-Meiklejohn et al. 2010 (doi:10.1016/j.cub.2010.04.055); Zaki, Schirmer, Mitchell 2011 (doi:10.1177/0956797611411057).
- Field-scale evidence that "who and how many" matters beyond content: Salganik, Dodds, Watts 2006 (doi:10.1126/science.1121066), where showing download counts made song success more unequal and less predictable. Cultural evolution already has the vocabulary: prestige bias (Henrich and Gil-White 2001, doi:10.1016/S1090-5138(00)00071-4), conformity, and the neutral-copying null (Bentley, Hahn, Shennan 2004, doi:10.1098/rspb.2004.2746). Youngblood 2019 fits transmission biases to music sampling and finds population-level patterns most consistent with conformity bias (abstract; doi:10.1098/rsos.191149).
- A third position cuts against both: Claidière, Scott-Phillips, Sperber 2014 (doi:10.1098/rstb.2013.0368) argue culture is mostly reconstruction toward attractors, not copying. If so, shared units between two works need no line of descent at all. This is the biggest threat to any inheritance reading, and the study's own identity result (the name-free "Batman core" is the generic superhero template, `study/paper/paper.md` section 4.1) looks like an attractor.

### B. "Everything is a Remix"
- Ferguson's series (2010 to 2012; complete version 2023; transcripts at https://everythingisaremix.info/articles/category/Transcripts) claims creativity is copy, transform, combine. It is an essay with examples. It names no case that would count against it.
- Scholarly pushback is mostly legal: Joo, "Remix without romance" (via https://www.copyhype.com/2012/04/remix-without-romance-what-free-culture-gets-wrong/), argues free-culture claims lack empirical support.
- Empirical remix work uses platforms with **declared** parent links, which is the ground truth neuron memes need:
  - Flath, Friesike, Wirth, Thiesse 2017, "Copy, transform, combine", Thingiverse 3D models over six years (doi:10.1057/s41265-017-0043-9). Explorative; it borrows Ferguson's title.
  - Hill and Monroy-Hernández 2013, "The remixing dilemma", Scratch (doi:10.1177/0002764212469359): projects that get remixed most yield less original remixes `[mem]`.
  - Youngblood 2019, WhoSampled sampling networks (doi:10.1371/journal.pone.0211860).
- Combination dominates in patents: Youn, Strumsky, Bettencourt, Lobo 2015 (doi:10.1098/rsif.2015.0272), where new technology codes dried up while new combinations of codes kept pace with patenting `[mem]`. Fleming 2001 (doi:10.1287/mnsc.47.1.117.10671): unfamiliar combinations lower average success and raise variance `[mem]`.
- Machine version of the remix ledger: Lu et al. 2025, "AI as Humanity's Salieri" (ICLR 2025, arXiv:2410.04265). Their Creativity Index scores how much of a text can be rebuilt from existing web snippets; professional human authors score about 66% higher than LLMs (abstract figure). It is a verbatim and near-verbatim measure, so it is the lexical rival for any "inherited against new" neuron ledger. McCoy et al. 2023 (RAVEN, doi:10.1162/tacl_a_00567) found model text less novel than human text locally and as novel at larger scale, with occasional copied passages over 1,000 words.

### C. Novelty and influence, measured
| Study | Unit | Novelty | Influence | Status |
|---|---|---|---|---|
| Uzzi et al. 2013, doi:10.1126/science.1240474 | pairs of cited journals | z-score of pair frequency against a randomised citation network | citations | Contested: Boyack and Klavans 2014 reproduced it on Scopus but found field and journal effects (Nature, Science, PNAS made 9.4% of atypical pairs); Fontana et al. 2020 (doi:10.1016/j.respol.2020.104063) and Bradley et al. (arXiv:1909.08738) attack the normalisation |
| Kelly et al. 2021, doi:10.1257/aeri.20190499 | TF-IDF of patent text | low similarity to earlier patents | high similarity to later patents; importance is the ratio | Correlates with citations and adds to them (abstract). A pure word-count method |
| Barron et al. 2018, doi:10.1073/pnas.1717729115 | LDA topic mixtures of speeches | KL from preceding speeches | resonance = novelty minus transience `[mem]` | 40,000 speeches, 1,000 speakers; gives **per-person** scores (left innovated, right preserved) |
| Hughes et al. 2012, doi:10.1073/pnas.1115407109 | function-word frequencies | none | similarity to earlier authors | "Influence is declining" is really "style similarity to the past is declining"; critics said so at the time |
| Elgammal and Saleh 2015, arXiv:1506.00711 | visual features of 62K paintings | dissimilar to earlier work | similar to later work, as network centrality | Validation is a "time machine" check; press critics noted bias toward the representational features used |
| Park, Nam, Park 2020, doi:10.1140/epjds/s13688-019-0214-8 | chord transitions in piano works | generation probability given past works | same, forward | Information-theoretic; recovers period styles |
| Sreenivasan 2013, doi:10.1038/srep02758 | IMDb keywords and keyword pairs | rarity given earlier films | none | Revenue against novelty "resembles the Wundt-Berlyne curve" |
| Gerow et al. 2018, doi:10.1073/pnas.1719792115 | topic model over time | none | how far a paper shifts later language `[mem]` | Text influence beyond citations |
| Vicinanza et al. 2022, doi:10.1093/pnasnexus/pgac275 | BERT perplexity then and later `[mem]` | improbable now, probable later | prescience | Prescient ideas come from the periphery (title) |
| Kuhn, Perc, Helbing 2014 (paper.md ref [20]) | n-grams along citation links | none | propagation along declared links | Inheritance needs the declared link |
- Also relevant: Arts, Hou, Gomez 2021 on new words and word pairs in patents (doi:10.1016/j.respol.2020.104144); Wang, Veugelers, Stephan 2017 on delayed recognition of novel papers (doi:10.1016/j.respol.2017.06.006); Funk and Owen-Smith 2017 CD index (doi:10.1287/mnsc.2015.2366) and Park, Leahey, Funk 2023 (doi:10.1038/s41586-022-05543-x), challenged by Holst et al. 2024 as a dataset artefact (arXiv:2402.14583); Underwood et al. 2022 on cohort succession in literary change (doi:10.15195/v9.a8); text reuse and quote mutation (Leskovec et al. 2009, doi:10.1145/1557019.1557077; Smith, Cordell, Dillon 2013, doi:10.1109/BigData.2013.6691675; Lerique and Roth 2018, doi:10.1111/cogs.12494).
- Pattern across the table: every one of these works without a model's neurons. Several are more interpretable than neurons (a keyword pair needs no label).

### D. How much novelty wins
- Berlyne's inverted U (liking peaks at moderate arousal potential) and Loewy's MAYA are the theory. Record is mixed: Chmiel and Schubert 2017 review music studies and find most compatible with an inverted U (doi:10.1177/0305735617697507) `[mem]`; Martindale, Moore, Borkum 1990 report "anomalous findings" for Berlyne's theory (title) and find typicality the stronger predictor `[mem]` (doi:10.2307/1423259). Hekkert, Snelders, van Wieringen 2003 test MAYA: typicality and novelty are negatively related and both raise preference `[mem]` (doi:10.1348/000712603762842147).
- Field data: Askin and Mauskapf 2017, Billboard songs do best when somewhat but not very different from contemporaries (doi:10.1177/0003122417728662) `[mem]`; Berger and Packard 2018, songs whose lyric topics differ more from their genre are more popular, monotonic in their data, weaker in pop and dance (doi:10.1177/0956797618759465); Uzzi's "conventional base plus an atypical intrusion".
- Blunt reading: "optimal differentiation" is a family of results with different shapes (inverted U, monotonic, mixed), each tied to one feature set and one success measure. An inverted U is easy to find after the fact. The study's selection test (E3) had 27% power at a true correlation of 0.3 (`study/results/experiments.json`, `E3_power_simulated`), so it cannot speak to any of this.

### E. Story shape
- Reagan et al. 2016: six emotional arcs from 1,327 Gutenberg stories by sentiment and SVD (doi:10.1140/epjds/s13688-016-0093-1). Toubia, Berger, Eliashberg 2021: speed, volume and circuitousness of the embedding path predict success (doi:10.1073/pnas.2011695118) `[mem]`. Boyd, Blackburn, Pennebaker 2020: staging, plot progression and cognitive tension from function words (doi:10.1126/sciadv.aba2196).
- The six-shapes claim is widely suspected of being a smoothing and decomposition artefact; I could not re-confirm the critiques this session (search budget ran out), so that stays `[UNVERIFIED]`.
- These measure the **order** of content. The study's memes are a bag of units per work and ignore order, so two stories with the same units in a different sequence look identical.

### F. Stylometry and authorship
- Mosteller and Wallace 1963 on the Federalist papers (doi:10.1080/01621459.1963.10500849); Burrows 2002 Delta (doi:10.1093/llc/17.3.267); Stamatatos 2009 survey (doi:10.1002/asi.21001). Function-word counts identify authors well. This is the cheap rival for any claim about a **person's** imprint.
- Neural: Rivera-Soto et al. 2021, universal authorship representations (https://aclanthology.org/2021.emnlp-main.70/); Wegmann, Schraagen, Nguyen 2022 show authorship-trained style vectors leak topic unless content is controlled (arXiv:2204.04907); Huang, Chen, Shu 2024 test zero-shot LLM authorship verification (arXiv:2403.08213).
- Interpretability work on style, 2025 to 2026: SAE features separate machine from human text (Kuznetsov et al., arXiv:2503.03601); authorship signal consolidates in early to mid layers of encoders (Kulumba et al., arXiv:2605.19908); SAE "literary primitive" features in Llama 3.1 8B and Gemma 2 9B, validated by steering (Presa and de Oliveira, arXiv:2605.18808; two authors, not peer reviewed as far as I could tell).
- Lesson for the project: style and content are separate channels of influence. Content-word and name removal (what the study did) strips the content channel and leaves style, and stylometry reads style better and cheaper than neurons are likely to.

### G. Training-data attribution, the machine analogue
- Influence functions ask a counterfactual: how would the output change if this training item were added or removed (Koh and Liang 2017, arXiv:1703.04730; scaled to LLMs by Grosse et al. 2023, arXiv:2308.03296, who use TF-IDF filtering to shortlist candidates). Alternatives: datamodels (arXiv:2202.00622), TRAK (arXiv:2303.14186).
- They are fragile and do not answer the question people think they do (Basu, Pope, Feizi 2020, arXiv:2006.14651; Bae et al. 2022, arXiv:2209.05364).
- **Influence is not similarity.** Chang et al. 2024 (arXiv:2410.17413): at 8B parameters and 160B tokens, their gradient method best finds examples that move the prediction, "but classical, model-agnostic retrieval methods such as BM25 still perform better at finding passages which explicitly contain relevant facts". Same shape as the Batman result.
- Ground truth by construction is the accepted evaluation: Wang et al. 2023 fine-tune a model toward an exemplar so the outputs are influenced by it by design (arXiv:2306.09345). Li, Banburski-Fahey, Lanier and colleagues 2026 build FakeWiki, 3,537 fabricated articles with paraphrases and "hard anti-documents" that stay on topic but drop the key facts, explicitly to weaken lexical shortcuts (arXiv:2605.05687). That is the planted-story idea in `preregistration/`, already published for provenance.
- Closest thing to "neuron memes for influence": Kowal, Paulo, Jaburi and colleagues 2026, "Concept Influence" (arXiv:2602.14869). They attribute behaviour to semantic directions (probes or SAE features) because per-example influence is biased "toward syntactic rather than semantic similarity", and simple probe-based versions do as well at over ten times less cost.
- Copying against style in image models: Somepalli et al. 2022 (arXiv:2212.03860); Carlini et al. 2023 (arXiv:2301.13188).

### H. What the Batman study already says on the four targets
- **Declared influence** (E2): supported as a pattern, and not specific to neurons. Share of eligible comics beating the documented source: Qwen 0.164, Llama 0.102, TF-IDF cosine 0.110, word presence 0.167 (`study/results/experiments.json`: `qwen3-8b/E2/primary`, `llama3.1-8b/E2/primary`, `lexical_comparators`). Residual test after removing word similarity: fails in Qwen (p = 0.09 to 0.25), passes in Llama (p = 0.010 to 0.033) (`study/paper/paper.md` Table 2; per-pair rows in `experiments.json` under `E2/residual_test_beyond_words`).
- **Undeclared influence**: not tested. The nearest thing, the meme tree, ranks the true source 5.4 of 12 against 5.8 by chance and 2.9 for TF-IDF, and random non-meme units beat the memes (`study/paper/paper.md` section 3.5; I did not trace these to a results file). The Killing Joke and Joker (2019) pairing is one anecdote.
- **Inherited against new**, **recombination**, **a person's influence**: not measured. The paper only proposes a definition of mutation (a unit no earlier work had) in section 4.3.
- **Cross-model**: distance matrices agree at Mantel 0.98, but random non-meme units give 0.90 to 0.93 and TF-IDF against LSA gives 0.92 (`study/paper/paper.md` section 3.3). Agreement on distances is nearly free.

## What this means for P1 and P2
**P1 (neurons as the unit).**
- For influence and novelty, the unit must do one of two things that words cannot: survive a change of surface (paraphrase, translation, medium), or be nameable in a way that a keyword is not. The literature gives no case where a neuron-level unit has been shown to beat lexical or embedding units at tracing influence between human works. The study's residual test is the only direct evidence, and it splits by model.
- Supportive but indirect: Concept Influence (2026) shows feature-level attribution fixes a known lexical bias inside models. That is about a model's own training data, not about culture.
- New risk that P1 has to face here: novelty measures count **first appearances**, so they are dominated by false positives. A unit that crosses the presence bar by noise is scored as an invention. The study reports no test-retest reliability across two independent summaries of the same story. Until that exists, any "share of this work that is new" figure is uninterpretable.
- A bag of present units ignores order and combination structure, which is where story-shape work and recombination work say the signal is.
- Sperber's attractor argument says shared units do not imply descent. Neurons trained on all of culture are, if anything, attractor detectors: they fire on what stories of a kind tend to contain.

**P2 (generality across models).**
- A science of influence needs the ledger to be model-independent: two models should call the same units inherited and the same units new. The study shows agreement at the level of work-to-work distance, where random units and TF-IDF agree almost as well, and weak agreement at the level of single units (median twin correlation 0.40 between Qwen3-8B and Qwen3-1.7B; the two revenge units correlate 0.39; `study/paper/paper.md` sections 3.3 and 3.4).
- The one result that matters for influence, the beyond-words residual, is model-dependent (Qwen fails, Llama passes). That cuts against P2 for this use.
- No source I found tests whether novelty or resonance rankings are stable across models. That is an open and cheap question.

## Experiment seeds

**S1. Planted influence with a paraphrase gradient.**
- Question: can neuron memes detect that work B drew on work A when B's wording no longer resembles A's?
- Data: built. LLM writers produce stories in unrelated genres. Each writer is either shown a source story or not, and told to copy closely, transform (new setting, names, wording), combine two sources, or write freely. Truth is known by construction, as in Wang et al. 2023 and FakeWiki. Include "anti-sources": same genre and topic, not shown to the writer.
- Method: hub-adjusted inheritance rate of the source's rare units (the E2 statistic), ranked against anti-sources; accuracy by condition.
- Rival to beat: TF-IDF cosine, word presence, e5 or Qwen3-Embedding cosine, and an LLM judge reading both texts.
- Falsified if: in the transform condition the neurons are not above the best embedding or the LLM judge, or their advantage vanishes once word similarity is regressed out.
- Cost: low to moderate. A few hundred generated stories; one extraction pass per model. It can ride on the paused planted-story plan in `preregistration/`.

**S2. Off-the-shelf narrative-similarity benchmark.**
- Question: on human-judged story similarity that deliberately ignores names and surface detail, do neuron-meme sets rank the right story?
- Data: SemEval-2026 Task 4 triples (anchor, two candidates; Wikipedia synopses; arXiv:2604.21782) and the Tell Me Again multi-summary dataset (paper.md ref [23]).
- Method: Jaccard or weighted overlap of meme sets; pick the closer candidate; report accuracy.
- Rival to beat: published baselines (e5 at 66.75% on the embedding track beat the purpose-built story-emb at 63.25%), plus TF-IDF and an LLM judge.
- Falsified if: neurons score below e5, or no better than TF-IDF.
- Cost: lowest of all. No new data; one extraction pass. Also gives the **test-retest noise floor**: two summaries of one story should share nearly all units.

**S3. The remix ledger, validated on synthetic remixes.**
- Question: can the share of a work that is inherited, new, or newly combined be measured, and does the measured share track the true share?
- Data: (a) synthetic: stories assembled from k known sources plus m planted new elements, with wording rewritten; (b) real: a large dated pool (thousands of plot summaries), since "new" is relative to the pool.
- Method: for each work, classify each unit as seen in an earlier work or first appearance; classify each unit pair as seen together before or not (Uzzi-style z-score against a shuffled null); estimate source mixture by non-negative regression on candidate sources. Subtract the noise floor from S2.
- Rival to beat: the same ledger with word n-grams (Kelly-style TF-IDF backward similarity), keyword pairs (Sreenivasan), embedding nearest-neighbour distance, and a DJ-Search-style reconstruction score.
- Falsified if: the neuron ledger's correlation with planted proportions is no higher than the rivals', or its "new" share on duplicate summaries is as large as on unrelated works.
- Cost: moderate. Synthetic part is cheap; the real pool needs a few thousand extractions.

**S4. A person's influence as resonance.**
- Question: do the units a person introduces persist in other people's later work more than matched units introduced by others, and does this predict known influence?
- Data: a dated corpus with author labels and an outside measure of influence: scientific abstracts with citations, or parliamentary speeches (Barron's design), or Gutenberg authors (Hughes's corpus).
- Method: per-author novelty (distance from prior pool), transience and resonance on neuron-meme profiles; test whether resonance predicts later citations or adoption.
- Rival to beat: the same three quantities from LDA topics (Barron), TF-IDF forward-to-backward ratio (Kelly), sentence embeddings, and function-word stylometry (Burrows Delta) for the style channel.
- Falsified if: neuron resonance adds no predictive power once the TF-IDF and topic versions are in the model.
- Cost: moderate; mostly extraction over tens of thousands of short texts.

**S5. Model against content: a Girard and Dawkins variance split.**
- Question: is the adoption of a unit explained more by which unit it is or by who displayed it?
- Data: a community with timestamps and a visible exposure graph and declared parent links: Scratch or Thingiverse remix trees, fan fiction with "inspired by" links, or a forum with reply threads.
- Method: for each exposure, model whether the exposed person's next work carries the unit; compare variance explained by unit fixed effects (content) and by exposer fixed effects and prestige (model). Girard's extra prediction: pairs in documented conflict converge in their units over time.
- Rival to beat: identical analysis with hashtags, keywords or embedding clusters as the unit.
- Falsified if (for the neurons): the split is the same with keyword units. Falsified (for Girard): exposer effects add nothing beyond unit effects and exposure counts.
- Cost: high. Data access and exposure reconstruction are the hard part.

**S6. Imprint of a known author on a writer, by construction.**
- Question: when a writer is steeped in author X, do neuron memes pick out X against decoy authors?
- Data: LLM writers conditioned (in context or by light fine-tuning) on one of several public-domain authors, then asked for stories on unrelated subjects.
- Method: directed inheritance of X's distinctive units; identify the conditioning author.
- Rival to beat: function-word Delta, a style-embedding model, and an LLM judge.
- Falsified if: stylometry identifies the author as well or better. I expect this outcome; the test is worth running because it would bound what "influence of a person" neuron memes can claim.
- Cost: low.

## Key sources
| Citation | URL or DOI | Why it matters | Priority |
|---|---|---|---|
| Kelly, Papanikolaou, Seru, Taddy 2021, Measuring technological innovation over the long run | doi:10.1257/aeri.20190499 | The cleanest backward-and-forward similarity design; pure TF-IDF; the rival to copy and beat | A |
| Barron, Huang, Spang, DeDeo 2018 | doi:10.1073/pnas.1717729115 | Novelty, transience, resonance per person; template for S4 | A |
| Chang et al. 2024, Scalable influence and fact tracing | arXiv:2410.17413 | BM25 beats gradient influence at finding the explicit source: influence is not similarity | A |
| Kowal et al. 2026, Concept Influence | arXiv:2602.14869 | Attribution to SAE features and probes; closest published analogue to neuron memes for influence | A |
| Hatzel et al. 2026, SemEval-2026 Task 4 | arXiv:2604.21782 | Ready-made, human-judged, surface-independent story-similarity benchmark with baselines | A |
| Wang, Efros, Zhu, Zhang 2023, Evaluating data attribution for text-to-image models | arXiv:2306.09345 | Influence known by construction; the evaluation logic for S1 and S6 | A |
| Li et al. 2026 (with Lanier), DataDignity | arXiv:2605.05687 | FakeWiki with paraphrases and anti-documents built to defeat lexical shortcuts | A |
| Lebreton et al. 2012 | doi:10.1523/JNEUROSCI.4821-11.2012 | The lab test of mimetic desire; narrow, single lab | B |
| Uzzi, Mukherjee, Stringer, Jones 2013 | doi:10.1126/science.1240474 | Atypical-combination z-score; method for the recombination ledger; contested | B |
| Fontana, Iori, Montobbio, Sinatra 2020 | doi:10.1016/j.respol.2020.104063 | Critique of novelty indicators | B |
| Lu et al. 2025, AI as Humanity's Salieri | arXiv:2410.04265 | Reconstruction-from-prior-text score; lexical rival for inherited against new | B |
| Youn, Strumsky, Bettencourt, Lobo 2015 | doi:10.1098/rsif.2015.0272 | Combination against new units in patents | B |
| Flath, Friesike, Wirth, Thiesse 2017 | doi:10.1057/s41265-017-0043-9 | Declared remix links on Thingiverse | B |
| Hill and Monroy-Hernández 2013 | doi:10.1177/0002764212469359 | Declared remix links on Scratch; originality of remixes | B |
| Sreenivasan 2013 | doi:10.1038/srep02758 | Keyword and keyword-pair novelty in film; inverted U with revenue | B |
| Elgammal and Saleh 2015 | arXiv:1506.00711 | Originality and influence as network inference; validation problem stated openly | B |
| Hughes, Foti, Krakauer, Rockmore 2012 | doi:10.1073/pnas.1115407109 | Style similarity sold as influence; the cautionary case | B |
| Claidière, Scott-Phillips, Sperber 2014 | doi:10.1098/rstb.2013.0368 | Attraction against copying; shared traits without descent | B |
| Salganik, Dodds, Watts 2006 | doi:10.1126/science.1121066 | Social influence beyond content, experimentally | B |
| Wegmann, Schraagen, Nguyen 2022 | arXiv:2204.04907 | Style vectors leak topic; content control is needed | B |
| Burrows 2002, Delta | doi:10.1093/llc/17.3.267 | Cheap stylometric rival for person-level claims | B |
| Askin and Mauskapf 2017 | doi:10.1177/0003122417728662 | Optimal differentiation in music | C |
| Berger and Packard 2018 | doi:10.1177/0956797618759465 | Atypical lyrics more popular; monotonic, not inverted U | C |
| Toubia, Berger, Eliashberg 2021 | doi:10.1073/pnas.2011695118 | Shape of the semantic path | C |
| Reagan et al. 2016 | doi:10.1140/epjds/s13688-016-0093-1 | Six arcs; contested | C |
| Hekkert, Snelders, van Wieringen 2003 | doi:10.1348/000712603762842147 | MAYA tested | C |
| Chmiel and Schubert 2017; Martindale et al. 1990 | doi:10.1177/0305735617697507; doi:10.2307/1423259 | The mixed record on the inverted U | C |
| Grosse et al. 2023; Koh and Liang 2017; Bae et al. 2022; Basu et al. 2020 | arXiv:2308.03296; 1703.04730; 2209.05364; 2006.14651 | Influence functions and their limits | C |
| Park, Nam, Park 2020; Gerow et al. 2018; Vicinanza et al. 2022 | doi:10.1140/epjds/s13688-019-0214-8; doi:10.1073/pnas.1719792115; doi:10.1093/pnasnexus/pgac275 | Other novelty and influence formalisms | C |
| Presa and de Oliveira 2026; Kuznetsov et al. 2025; Kulumba et al. 2026 | arXiv:2605.18808; 2503.03601; 2605.19908 | Style and authorship features inside models | C |

## Fog
- The shared web-search budget ran out part-way through. Not checked as a result: critiques of the six emotional arcs; any replication of Lebreton et al. 2012; 2025 to 2026 work on LLM-embedding novelty measures for science or film; whether anyone has already traced influence between human works with SAE features.
- No peer-reviewed head-to-head of Girard and Dawkins was found. Girard's remark about Dawkins comes through commentary; the original passage was not located.
- Details marked `[mem]` (Barron's exact definitions, Youn's and Fleming's findings, Askin and Mauskapf's data, Toubia's three measures, Hill and Monroy-Hernández's result) should be re-read before they are quoted anywhere.
- Not confirmed this session: Boyack and Klavans 2014 as a formal publication (seen only through a search summary of slides and a conference paper); Alcácer and Gittelman 2006 on examiner-added patent citations (a possible "undeclared influence" ground truth; Crossref lookup failed) `[UNVERIFIED]`; Berlyne 1971 and Loewy 1951 as primary texts (known through Martindale 1990, Chmiel and Schubert 2017 and Hekkert et al. 2003); the Hatzel and Biemann story-embedding paper behind the story-emb baseline.
- The tree-distance source ranks (5.4, 5.8, 2.9) are taken from `study/paper/paper.md` section 3.5; I did not trace them to a file in `study/results/`.
- Open design problems: how to set a reference pool large enough that "new" means something; how to separate descent from convergence on an attractor without declared links; whether a unit's first crossing of the presence bar is reliable enough to count as a mutation; whether the SemEval-2026 Task 4 licence and its LLM-generated triples suit a pre-registered test.
- Whether the study has a reliability figure for the same story summarised twice was not checked (`study/results/wording.json` exists and was not read).
