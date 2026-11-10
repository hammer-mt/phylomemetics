# Roadmap: the state of play and what to test next

Written 8 November 2026 and updated on 10 November. This folder is a map of the research programme: an audit of everything tried so far, a review of how strong the two core premises are, and a registry of experiment designs to run next.

No experiment, model or pipeline was run to make it. The paper (v1) is unchanged.

The update of 10 November adds two documents published that week: Goodfire's technical companion to the essay behind P1, and a 182-page survey of the field. What they change is in point 7 of the short version, in section 6, and in [`premises.md`](premises.md).

## The programme in one paragraph

The bet is that a meme can be measured by which neurons fire inside a language model (premise P1), and that the measurement carries over because models converge on the same neurons and can be mapped onto one another (premise P2). If both hold, memetics gets what genetics got from sequencing: a cheap reader for the unit of inheritance, and with it family trees of ideas, tests of selection, forecasts and editing.

## The short version

1. **The instrument works as a counter.** Neuron readings are cheap, repeatable and citable, and they find real structure: Batman has a core that random sets of films lack, and adaptations carry their sources' content with every name removed.
2. **It has not yet shown that it reads ideas and not words.** On the primary model, plain word counts matched or beat neurons on every pre-registered test that had a word comparator: identity, heredity and the residual test. Most strong units are name and word detectors.
3. **P1 is supported in a weak form only.** Neurons are a better-than-random basis in which most units have a describable job at strong activation. The evidence is an essay and, since 10 November, its technical companion paper, each about a model four times larger than the ones used here. The paper's own figure is that 69% of neuron poles have one job or one dominant job, as grouped by an LLM judge. It tests no cheap rival and no rewording, and the project has never steered or switched off a neuron. The field's survey of November 2026 still treats learned features, not neurons, as the default unit, while granting that superposition is unproven in large models and that MLP neurons are a privileged basis.
4. **P2 is true for geometry, partly true for mapped directions, mostly false for single neurons.** One to five percent of neurons recur across retrained models. In this project, agreement between models collapses once names are removed. A 2026 study finds that neurons shared between models become a smaller share as models grow. The survey of November 2026 sorts the evidence along a different axis and reaches a compatible verdict: models sharing algorithms is the supported kind of convergence, and models sharing representations is "less supported".
5. **The next step is one rung, not twelve.** Until a unit beats sentence embeddings and an LLM reader on text where the words are controlled, the grander uses (forecasting, editing, trees of influence) rest on a reader that may only be reading words.
6. **Part of the ground is open.** Using a model's internal units to describe texts is already an active line of work, done with sparse-autoencoder features, human labels and steering. Nobody has published a descent test on cultural works using model units, or an edit of a story motif by steering. The cheap rivals are strong, so each claim has to be earned against them.

## What is in this folder

| File | What it holds |
| --- | --- |
| [`premises.md`](premises.md) | How strong P1 and P2 are, the validation ladder, and what would settle them |
| [`IDEAS.md`](IDEAS.md) | Every idea by research area: run, planned earlier, newly registered, or still only an idea |
| [`designs/`](designs/) | One file per registered design: question, prediction, what would refute it, data, method, the cheap rival to beat |
| [`biology-map.md`](biology-map.md) | 40 tools from biology and their memetic versions; which landmark of biology's history memetics is at |
| [`ESSAYS.md`](ESSAYS.md) | The 210 measurable claims in Mike's 2022 essays, each restated as a hypothesis and linked to the design or seed that tests it; and the 40 sources those essays cite that are most worth re-testing |
| [`sources.md`](sources.md) | The reading list, ranked |
| [`notes/`](notes/) | The research notes behind all of this, with citation check logs |
| [`map/index.html`](map/index.html) | The visual map: a chart of every idea, with the fog marked. Open it in a browser |
| `ideas.json`, `premises.json`, `biology.json`, `sources.json`, `summary.json`, `essays.json` | The data. Edit these, then run `python3 roadmap/build.py` to regenerate the files above |
| `ideas.csv` | The ideas as a spreadsheet |

## 1. What has been tried

Sixteen analyses have been run, all on the Batman corpus. Each is an entry in [`IDEAS.md`](IDEAS.md).

| Result | Verdict |
| --- | --- |
| A core canon exists beyond chance (IN-01) | Held |
| Heredity: adaptations carry their source comics, names removed (LI-02) | Held, and word counts do as well. Qwen's neurons failed the sequel check that word counts passed |
| Identity: telling Batman from other masked heroes without names (AT-01) | Did not hold; word counts did it |
| Family trees recover the source comic (LI-01) | Did not hold; word counts and random neurons did better |
| Selection: memes of popular works spread further (FI-01) | No signal; the test had 27% power |
| Neurons add something beyond shared words (LI-03) | Mixed: no for Qwen, marginal for Llama |
| Twins across two companies' models (UN-01, UN-02, UN-03) | Mixed: mostly shared name and word detectors |
| Units respond to ideas, not words (IN-03, IN-04) | Mixed: a handful do, most do not |
| Blind test of the labels (IN-02) | Mixed: 79% right, a word-matching judge gets 65% |
| Vision neurons on posters (UN-04) | Exploratory, no rival compared |
| The vigilante lineage: the Pimpernel, Zorro, Batman, Iron Man (LI-04) | Could not tell: one summary each, names present |
| What is left of Batman without names (AT-02) | Descriptive: the textbook definition of a superhero |
| When memes enter and recur (PO-01) | Descriptive: the sample has no works from 1967 to 1985 |

Three things the audit found that the paper does not say:

- The one pre-registered result in the neurons' favour, Llama's "beyond words" test (p = 0.010 to 0.033 across its three versions), is weaker under the study's own stricter nulls: p = 0.017 to 0.065. The version with TF-IDF regressed out rises to 0.057 and 0.065; the other two stay at 0.017 to 0.021. None is corrected for the six tests (`study/results/experiments.json`).
- The 30,240-variant robustness check protects the heredity pattern. It never tested the words-against-neurons question.
- The heredity test does not use the named memes. It uses about 150,000 raw units, and 95% of "meme families" are a single neuron (`study/results/robustness/e2_reimpl.json`, `study/results/qwen3-8b/memes.json.gz`).

Full audits: [`notes/a1-study.md`](notes/a1-study.md), [`notes/a2-robustness.md`](notes/a2-robustness.md), [`notes/a3-method.md`](notes/a3-method.md).

## 2. What was planned and not run

The pre-registration in [`../preregistration/`](../preregistration/) holds one planned experiment, the planted-meme stories, and a set of deferred ones: vocabulary drift, translation, Shakespeare, a 2,544-work culture pool, mutation and selection, and concept agreement between models. They appear in [`IDEAS.md`](IDEAS.md) as "planned earlier".

Two points from the audit ([`notes/a4-prereg.md`](notes/a4-prereg.md)):

- The planted-stories design has low power at its own smallest effect of interest (0.14 to 0.40). Claude writes and rates every story; the only outside checks are one Gemma 2 9B reader and Mike's blind audit of 60 stories.
- The deferred test of concept agreement between models needs only extraction on 1,000 film summaries that are already in `preregistration/sources/`. It is the cheapest unrun test of P2.

## 3. Rules every next experiment must follow

These come from what went wrong or nearly went wrong in the Batman study.

1. Put the cheap rivals in the same test: word overlap, a sentence-embedding model, an LLM reading the text, text length, random neurons. Where the concept is fixed in advance, add a linear probe on the same model's activations.
2. Make it impossible for words to carry the signal: paraphrase, translate, change medium, or use minimal pairs.
3. Take ground truth from outside the method: documented descent, randomised outcomes, human labels or construction.
4. Choose units on development data and score them on held-out data. There are about 900,000 units; some unit will fit any small sample.
5. State the refuting result with a number, and the sample size that gives enough power.
6. Use nulls that respect date, genre and length. Three errors in the study were comparisons whose answer was fixed by how they were built.
7. Do not assume the neurons are clean. Check it, or steer. Read a unit only in its own strong range. The threshold comes from the unit's own activations on general text, never from a comparison set of works.
8. Put a human in the loop somewhere. So far every labeller, rewriter, judge, reviewer and fact-checker was an AI.
9. A label is a guess until it predicts. From the label alone, a blind judge must pick held-out firing passages over same-topic distractors, and must beat a word-overlap picker and the same judge given another unit's label. The labelling method needs an error rate, measured on units whose true function is known.
10. Say which role fired. A neuron pole can have more than one. Keep the firing token and its context for every presence call, and count the call only if a blind check puts it in the labelled role.
11. Nothing carries over unmeasured. A threshold, a label or a published description holds for the model, the precision and the kind of text it was measured on. Re-measure before borrowing one.
12. State the coverage of every result: which units, works and range of activation were examined, and how much of the effect the named units leave unexplained. Report fragile findings beside robust ones.

Rules 9 to 12, and the additions to rules 1 and 7, were added on 10 November from Goodfire's companion paper and the survey of the field.

## 4. What to run first

Three judges ranked all 85 new designs and the 14 older plans across areas, each through a different lens: what decides the premises fastest, what is quickest and cheapest, and where the upside is. All three put the same small tests first. Nine designs made at least two of the three shortlists. A tenth, IN-24, is marked to run early as well, because it settles which unit every later design should freeze.

| Step | Designs | What it settles | Cost |
| --- | --- | --- | --- |
| 1 | IN-29, UN-21, IN-21 | Are the 4-bit readings real? Do any neurons recur between Qwen and Llama on unseen text once words are controlled? Does any neuron follow an idea without its words, for 30 story concepts? | About a week, laptop only |
| 2 | EN-21, CE-21 | Does pushing a named unit make the model write that idea? Do units track what human readers keep when they retell a story, judged by human coders? | Three or four days |
| 3 | IN-24, UN-23 | Is a meme one neuron, a few, or a direction? Does a fitted map carry a reading from one model to the other? | One to two weeks |
| 4 | IN-25, UN-22 | Is a reading a property of the story or of one summary? Does it survive a change of language? | About a week, one download |
| 5 | RX-27 | The first use test: name a known source after deep rewording, against look-alike decoys | About three days |

Two rules came from all three judges.

- Write down beforehand which results would move the unit from single neurons to small sets or directions. If fewer than 8 of 30 concepts pass IN-21 and under 0.5% of units pass UN-21, the single-neuron version of both premises is refuted before anything expensive is built.
- Start no lineage, forecasting or product design until a frozen unit has beaten sentence embeddings and an LLM reader on text where the words are controlled.

Several older plans are now superseded by cheaper designs that ask the same question: the planted-meme stories (IN-05, about 4,400 agent sessions) by IN-21, the Chinese translation (UN-05) by UN-22, the drift ladder (LI-05) by RX-27 and LI-21, and concept agreement on held-out films (UN-06) by UN-21. Each entry in [`IDEAS.md`](IDEAS.md) says what it overlaps with.

## 5. Where the treasure might be

These are the designs that would matter most if the first steps go well. Each is built so that it pays off whichever reader wins.

| What you want | Design | Why it is promising |
| --- | --- | --- |
| A family tree of influence, including influences nobody declared | RX-22, LI-22, CJ-22 | A search for unacknowledged sources checked against the written record; source finding across a language border; and remakes against "twin films" to tell copying from coincidence |
| Spotting a promising new idea | US-26, FI-22 | Meme combinations no earlier film used, judged by which films were later made; and whether units that won randomised headline tests then spread |
| More of one meme, less of another | EN-23, EN-22 | Whether the units that detect a meme are the ones that write it, and whether any internal handle beats simply asking a model to rewrite |
| A catalogue two labs can cite | AT-21, AT-24 | A reference list of units with measured hit and false-alarm rates for 150 human-named motifs |
| A periodic table of memes | AT-28 | Whether a grid of motifs predicts the unit for an empty cell before any text for it exists. Two days, under 30 dollars |
| Mendel's experiment | CE-23, AT-23 | Whether memes are inherited as particles or blend |
| The accepted science, re-run | CE-21, ME-22, PO-23 | What survives retelling; whether retold ideas are copied or rebuilt; the random-copying null that any selection claim has to beat |

The judges' shared view of the likely end state: a neuron's lasting job may be to name what was inherited, not to out-predict embeddings. A 2026 position paper on sparse model features argues the same split: they tie or lose at detecting a relation someone has specified, and win at discovering and naming ones nobody specified ([`notes/l10-competitors-instruments.md`](notes/l10-competitors-instruments.md)).

## 6. Short answers to the questions asked directly

**What Goodfire's companion paper changes (10 November).** ["Neurons as an Interpretable Basis"](https://www.goodfire.com/research/neurons-as-an-interpretable-basis) is the measured version of the essay behind P1, on one 35B model. It finds that the model's output depends mostly on its strongest neuron firings, that 47% of neuron poles have one role and 69% one dominant role in that strong range, and that a short description predicts when a neuron fires strongly (95% right against random text, where chance is 50%). It scores no word-count or embedding rival, runs no measured test of an idea with its usual words removed, repeats no statistic on a second model, and has nothing at the level of a document. For this programme it changes how a unit should be read, not whether the reading means anything: in the unit's own strong range, one role at a time, with a label that has been made to predict. Details: [`premises.md`](premises.md) and [`notes/l14-goodfire-companion.md`](notes/l14-goodfire-companion.md).

**What the survey of the field changes (5 November).** ["Understanding AI from the Inside"](https://arxiv.org/abs/2611.08438), 182 pages by Barez, Wu and 25 co-authors, among them Yoshua Bengio, is the field's own account of itself. Its default unit is the learned feature, not the neuron, though it grants that superposition is "a hypothesis, not an established fact" in large models and that MLP neurons are a privileged basis. On agreement between models it is compatible with this roadmap: shared algorithms are supported, shared representations "less supported". Its standard for a claim about a unit is causal, with the baselines fixed first. It has no test that a reading follows an idea and not a word, so the rung that decides this programme is the programme's own. Its chapter on tracing a model's output back to its training data is the nearest existing method to tracing undeclared influence. By the survey's own account those methods are thinly validated, and each works only inside one model's training set. Design LI-27 adapts one to pairs of texts. Details: [`notes/l15-interpretability-survey.md`](notes/l15-interpretability-survey.md).

**The MIT work on editing a model by handing over its weights.** Most likely ["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/) by Takehi, Teoh, Zhang and Khattab (MIT CSAIL), a blog post of 2 November 2026. A frontier model read and edited the weights of small networks. Its one language-model edit removed a looping bug in a 0.8B Qwen. It is not peer reviewed, and it did not edit content, style or anything meme-like. The match to the work Mike described is an inference.

**People pushing single neurons on X.** The "Evil Qwen" posts of October 2026 pushed two neurons in a 4B Qwen. A pre-registered replication found a villain voice in 82 to 86% of answers, coherent text in 5 to 36% depending on the judge, and no effect from the second neuron. One census found 130 of 294,912 neurons with a clean, consistent steering effect. The neuron that reads a concept is often not the one that writes it.

**Can an idea be mutated on purpose today?** Broad traits such as tone and persona can be steered with a direction on open 8B models, inside a narrow dose window. On the one large benchmark, plain prompting beats every activation method. Nobody has published "more of motif A, less of motif B" on an existing story. Designs EN-21 to EN-23 test it. Goodfire's companion paper adds a route that touches no weights: agents wrote text that fired 120 of 128 neuron poles harder than anything in a 10-billion-token corpus, guided only by reading the activation back. Whether such text carries more of the idea, or only more of a trigger word, is untested (designs IN-31 and EN-28). Details for these three answers: [`notes/l3-steering-editing.md`](notes/l3-steering-editing.md).

**Dawkins against Girard.** Never tested head to head. It becomes testable as what gets copied (the content) against who gets copied (the person imitated). Neuron memes measure only the "what". Girard needs exposure data: who saw whom. Design RX-26 and seeds RX-S3, RX-S10 and RX-S11.

**Everything is a remix.** As stated it cannot fail, and no proof exists. It becomes a test as a ledger: the share of a work's units seen in earlier works, the share that is new, and the share that is new pairings of old units, measured against the noise between two summaries of the same story. Designs RX-21 and CJ-26. Details for these two answers: [`notes/l8-girard-remix-influence.md`](notes/l8-girard-remix-influence.md).

**Which moment of biology's history is this?** Memetics already has its Linnaeus (folktale indexes, TV Tropes), its Darwin (Dawkins; Boyd and Richerson) and its Kimura (the random-copying model). It lacks a Mendel, meaning clean units in controlled crosses, and a Sanger, meaning a cheap reader with a known error rate. Neuron reading is a candidate Sanger. There is no Watson and Crick for human culture: a model neuron is an instrument, not the substrate in a head. Goodfire's two texts are closest to Beadle and Tatum's "one gene, one enzyme": a clean rule, stated in October and already revised to "one or a few roles" in November. A Mendeleev test is design AT-28. The full table is in [`biology-map.md`](biology-map.md).

**The claims in Mike's 2022 essays.** All 122 essays were read in full. [`ESSAYS.md`](ESSAYS.md) lists 210 measurable claims, each with its essay, a hypothesis, the cheap rival and the design or seed that tests it. For 57 of them a neuron reading could add something; the other 153 could be tested with any unit, word counts included. 36 claims are already tested by a registered or earlier design, 42 by a seed (37 seeds were added for this), and 132 have no entry because they do not need the method. Seven became full designs under Field conjectures: the Lindy effect, the thirty-year return, memeplexes, most advanced yet acceptable, the remix ledger, borrowing from many sources, and copied or coincidence. The same page lists the 40 sources the essays cite that are most worth re-testing.

**Fourteen sources to read first.**

1. [Cammarata 2026, "Neural networks are already clean"](https://already-clean.goodfire.com/). The source of P1. Read the footnotes.
2. [Tigges et al. 2026, "Neurons as an Interpretable Basis"](https://www.goodfire.com/research/neurons-as-an-interpretable-basis). The essay's technical companion: one 35B model, no cheap rival. Read the methods and the appendix on random directions.
3. [Barez, Wu et al. 2026, "Understanding AI from the Inside"](https://arxiv.org/abs/2611.08438). The field's survey. Sections 4.2, 5, 6.5, 7.5 and 14 bear on this programme.
4. [Nanda's reply](https://www.lesswrong.com/posts/x4kvabkF2Jk8GLZp8/linkpost-neural-networks-are-already-clean/comment/7wcSqBNzQF2o9SHyY). The main expert objection.
5. [IRSI's retest](https://irsi.ai/notes/already-clean/) and [Batutin's replication](https://andriybatutin.substack.com/p/we-tried-to-confirm-neural-networks). The only outside numbers, both about the essay. Neither is peer reviewed.
6. [Gurnee et al. 2024, universal neurons](https://arxiv.org/abs/2401.12181). The base rate for P2: 1 to 5%.
7. [Jiralerspong and Bricken 2026](https://arxiv.org/abs/2602.11729). A joint dictionary on this project's exact pair of models.
8. [Movva et al. 2025](https://arxiv.org/abs/2502.04382) and [Peng et al.](https://arxiv.org/abs/2506.23845). The nearest competing line: sparse features as columns for describing texts.
9. [Chang 2004, *Inventing Temperature*](https://doi.org/10.1093/0195171276.001.0001). How a measurement earns trust with no ground truth.
10. [Edmonds 2005](http://cfpm.org/jom-emit/2005/vol9/edmonds_b.html). Why memetics closed: no added value, the same test as beating word counts.
11. [Henrich and Boyd 2002](https://doi.org/10.1163/156853702320281836). Why cultural evolution says it needs no discrete unit.
12. [Hatzel and Biemann 2024, Tell Me Again](https://aclanthology.org/2024.lrec-main.1366). The data behind three of the run-first designs.
13. [Berl et al. 2021](https://doi.org/10.1017/ehs.2021.37). Human retellings coded by people: the data behind CE-21.
14. [Matias et al. 2021, the Upworthy Research Archive](https://doi.org/10.1038/s41597-021-00934-7). 32,487 randomised headline tests.

## 7. Where each area stands

| Area | In biology | Where it stands | Best next design |
| --- | --- | --- | --- |
| The instrument | Sequencing | Four small tests: the canon count held, the other three were mixed. No human check, no steering. Goodfire's paper gives the range to read and the label tests to copy | IN-21 |
| Across models | One genetic code | Twins were mostly shared word detectors. No learned map tried | UN-21, UN-22 |
| The atlas | Genome project | Batman's name-free core is the textbook superhero. No catalogue yet for this project's models. Goodfire's public atlas describes 3,000 neuron poles of a different model | AT-21, after IN-21 |
| Old memetics, re-run | Classical genetics | Untouched. Public data and published results to re-derive | ME-22 |
| Lineage | Phylogenetics | Heredity held, word counts matched it, trees failed | LI-22 |
| Remix and influence | Recombination | Nothing run. Similarity is not influence | RX-27 |
| Population memetics | Population genetics | Descriptive timelines only | PO-21 |
| Cultural evolution, re-run | Evolutionary ecology | Untouched. Human retelling data is public | CE-21 |
| Fitness and forecasting | Fitness landscapes | One underpowered test that could not have found a signal | FI-21 |
| Meme engineering | Gene editing | Nothing run. Prompting is the rival to beat | EN-21 |
| Field conjectures | Natural history | 210 claims from Mike's 2022 essays, none measured ([`ESSAYS.md`](ESSAYS.md)) | CJ-21, CJ-22 |
| Uses | Biotechnology | Nothing run in the open | US-23, US-26 |

## 8. How the registry works

A design in [`designs/`](designs/) is registered, not pre-registered. It fixes the question, the prediction, the refuting result and the rival before any data exists, and it is dated by this commit. To run one, expand it into a full pre-registration first.

Each design was drafted by one AI agent from the notes and then attacked and revised by a second. Priorities run from 1 to 5: the value of the answer times the chance it can be done. The area reviewer set a first value and the three cross-area judges adjusted it: 4.5 or more means at least two judges put the design in their top twelve, with IN-24 as the one exception noted above. They are a first sort. Mike decides.

## 9. How this was made, and how far to trust it

Twenty research agents each wrote one notes file: seven auditing this repository and Mike's own material, thirteen reviewing the outside literature with web search. A second agent checked every citation in each literature file. Twelve more agents drafted the designs, twelve reviewed them, and three ranked them across areas. Two of the audit files draw on private material and are not in this folder. A second pass on 9 November read all ten chapters of Mike's 2022 essays in full: six agents read the chapters, one mapped the claims to the database and one reviewed the essays' own sources.

The audits quote committed outputs and can be checked against them. The literature notes are leads: recent work is thinly reviewed, and anything not confirmed is marked `[UNVERIFIED]`. Confirm a source before citing it in a paper.
