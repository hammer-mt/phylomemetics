# Roadmap: the state of play and what to test next

Written 8 November 2026. This folder is a map of the research programme: an audit of everything tried so far, a review of how strong the two core premises are, and a registry of experiment designs to run next.

No experiment, model or pipeline was run to make it. The paper (v1) is unchanged.

## The programme in one paragraph

The bet is that a meme can be measured by which neurons fire inside a language model (premise P1), and that the measurement carries over because models converge on the same neurons and can be mapped onto one another (premise P2). If both hold, memetics gets what genetics got from sequencing: a cheap reader for the unit of inheritance, and with it family trees of ideas, tests of selection, forecasts and editing.

## The short version

1. **The instrument works as a counter.** Neuron readings are cheap, repeatable and citable, and they find real structure: Batman has a core that random sets of films lack, and adaptations carry their sources' content with every name removed.
2. **It has not yet shown that it reads ideas and not words.** On the primary model, plain word counts matched or beat neurons on every pre-registered test. Most strong units are name and word detectors.
3. **P1 is supported in a weak form only.** Neurons are a better-than-random basis in which most units have a describable job. The claim comes from a preliminary essay about a model four times larger, and the project has never run the test that essay treats as decisive, steering.
4. **P2 is true for geometry, partly true for mapped directions, mostly false for single neurons.** One to five percent of neurons recur across retrained models. In this project, agreement between models collapses once names are removed.
5. **The next step is one rung, not twelve.** Until a unit beats sentence embeddings and an LLM reader on text where the words are controlled, the grander uses (forecasting, editing, trees of influence) rest on a reader that may only be reading words.
6. **The ground is open.** Nobody has published a neuron-defined meme, a descent test on cultural works using model units, or an edit of a story motif by steering. The cheap rivals are strong, so each claim has to be earned against them.

## What is in this folder

| File | What it holds |
| --- | --- |
| [`premises.md`](premises.md) | How strong P1 and P2 are, the validation ladder, and what would settle them |
| [`IDEAS.md`](IDEAS.md) | Every idea by research area: run, planned earlier, newly registered, or still only an idea |
| [`designs/`](designs/) | One file per registered design: question, prediction, what would refute it, data, method, the cheap rival to beat |
| [`biology-map.md`](biology-map.md) | 37 tools from biology and their memetic versions; which landmark of biology's history memetics is at |
| [`sources.md`](sources.md) | The reading list, ranked |
| [`notes/`](notes/) | The research notes behind all of this, with citation check logs |
| [`map/index.html`](map/index.html) | The visual map: a chart of every idea, with the fog marked. Open it in a browser |
| `ideas.json`, `premises.json`, `biology.json`, `sources.json`, `summary.json` | The data. Edit these, then run `python3 roadmap/build.py` to regenerate the files above |
| `ideas.csv` | The ideas as a spreadsheet |

## 1. What has been tried

Sixteen analyses have been run, all on the Batman corpus. Each is an entry in [`IDEAS.md`](IDEAS.md).

| Result | Verdict |
| --- | --- |
| A core canon exists beyond chance (IN-01) | Held |
| Heredity: adaptations carry their source comics, names removed (LI-02) | Held, and word counts do as well |
| Identity: telling Batman from other masked heroes without names (AT-01) | Did not hold; word counts did it |
| Family trees recover the source comic (LI-01) | Did not hold; word counts and random neurons did better |
| Selection: memes of popular works spread further (FI-01) | No signal; the test had 27% power |
| Neurons add something beyond shared words (LI-03) | Mixed: no for Qwen, marginal for Llama |
| Twins across two companies' models (UN-01, UN-02, UN-03) | Mixed: mostly shared name and word detectors |
| Units respond to ideas, not words (IN-03, IN-04) | Mixed: a handful do, most do not |
| Blind test of the labels (IN-02) | Mixed: 79% right, a word-matching judge gets 65% |
| Vision neurons on posters (UN-04) | Exploratory, no rival compared |

Three things the audit found that the paper does not say:

- The one result in the neurons' favour, Llama's "beyond words" test, weakens to p = 0.057 to 0.065 under the study's own stricter nulls and is uncorrected for six tests (`study/results/experiments.json`).
- The 30,240-variant robustness check protects the heredity pattern. It never tested the words-against-neurons question.
- The heredity test does not use the named memes. It uses about 150,000 raw units, and 95% of "meme families" are a single neuron (`study/results/robustness/e2_reimpl.json`, `study/results/qwen3-8b/memes.json.gz`).

Full audits: [`notes/a1-study.md`](notes/a1-study.md), [`notes/a2-robustness.md`](notes/a2-robustness.md), [`notes/a3-method.md`](notes/a3-method.md).

## 2. What was planned and not run

The pre-registration in [`../preregistration/`](../preregistration/) holds one planned experiment, the planted-meme stories, and a set of deferred ones: vocabulary drift, translation, Shakespeare, a 2,544-work culture pool, mutation and selection, and concept agreement between models. They appear in [`IDEAS.md`](IDEAS.md) as "planned earlier".

Two points from the audit ([`notes/a4-prereg.md`](notes/a4-prereg.md)):

- The planted-stories design has low power at its own smallest effect of interest (0.14 to 0.40), and every story, reader and rater in it is one AI model family.
- The deferred test of concept agreement between models needs only extraction on 1,000 film summaries that are already in `preregistration/sources/`. It is the cheapest unrun test of P2.

## 3. Rules every next experiment must follow

These come from what went wrong or nearly went wrong in the Batman study.

1. Put the cheap rivals in the same test: word overlap, a sentence-embedding model, an LLM reading the text, text length, random neurons.
2. Make it impossible for words to carry the signal: paraphrase, translate, change medium, or use minimal pairs.
3. Take ground truth from outside the method: documented descent, randomised outcomes, human labels or construction.
4. Choose units on development data and score them on held-out data. There are about 900,000 units; some unit will fit any small sample.
5. State the refuting result with a number, and the sample size that gives enough power.
6. Use nulls that respect date, genre and length. Three errors in the study were comparisons whose answer was fixed by how they were built.
7. Do not assume the neurons are clean. Check it, or steer.
8. Put a human in the loop somewhere. So far every labeller, rewriter, judge, reviewer and fact-checker was an AI.

## 4. What to run first

Three judges ranked all 85 new designs and the 14 older plans across areas, each through a different lens: what decides the premises fastest, what is quickest and cheapest, and where the upside is. All three put the same small tests first. Nine designs made at least two of the three shortlists.

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

## 6. Where each area stands

| Area | In biology | Where it stands | Best next design |
| --- | --- | --- | --- |
| The instrument | Sequencing | Four small tests, all mixed. No human check, no steering | IN-21 |
| Across models | One genetic code | Twins were mostly shared word detectors. No learned map tried | UN-21, UN-22 |
| The atlas | Genome project | Batman's name-free core is the textbook superhero. No catalogue yet | AT-21, after IN-21 |
| Old memetics, re-run | Classical genetics | Untouched. Public data and published results to re-derive | ME-22 |
| Lineage | Phylogenetics | Heredity held, word counts matched it, trees failed | LI-22 |
| Remix and influence | Recombination | Nothing run. Similarity is not influence | RX-27 |
| Population memetics | Population genetics | Descriptive timelines only | PO-21 |
| Cultural evolution, re-run | Evolutionary ecology | Untouched. Human retelling data is public | CE-21 |
| Fitness and forecasting | Fitness landscapes | One underpowered null | FI-21 |
| Meme engineering | Gene editing | Nothing run. Prompting is the rival to beat | EN-21 |
| Field conjectures | Natural history | About 35 hunches, none measured | CJ-21, CJ-22 |
| Uses | Biotechnology | Nothing run in the open | US-23, US-26 |

## 7. How the registry works

A design in [`designs/`](designs/) is registered, not pre-registered. It fixes the question, the prediction, the refuting result and the rival before any data exists, and it is dated by this commit. To run one, expand it into a full pre-registration first.

Each design was drafted by one AI agent from the notes and then attacked and revised by a second. Priorities run from 1 to 5: the value of the answer times the chance it can be done. The area reviewer set a first value and the three cross-area judges adjusted it: 4.5 or more means at least two judges put the design in their top twelve. They are a first sort. Mike decides.

## 8. How this was made, and how far to trust it

Twenty research agents each wrote one notes file: seven auditing this repository and Mike's own material, thirteen reviewing the outside literature with web search. A second agent checked every citation in each literature file. Twelve more agents drafted the designs, twelve reviewed them, and three ranked them across areas. Two of the audit files draw on private material and are not in this folder.

The audits quote committed outputs and can be checked against them. The literature notes are leads: recent work is thinly reviewed, and anything not confirmed is marked `[UNVERIFIED]`. Confirm a source before citing it in a paper.
