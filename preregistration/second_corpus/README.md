# A second corpus: scouting notes

**Status: the scouting is done and two corpora were built from it (8 November 2026).** Mike asked whether the Batman study should be tried on a different corpus, perhaps a different franchise. Four scouts each proposed one, and two judges compared them. Mike then chose both of the judges' candidates: the Sherlock Holmes corpus and the remakes. They were rebuilt by rule from Wikipedia's own wording, with a plan frozen before any model read them: `study/v2/second/PLAN.md` (corpora in `study/data/second/`, results in `study/results/v2/second/`).

This folder keeps the scouting as it was. Its candidate lists were recalled from memory and are superseded by the lists built from the articles; its word-overlap figures are sizing notes, and none of them goes into the paper. [`designs/`](designs/) holds the three statisticians' reports on how to test whether units add to words on the remakes; the plan adopts one of them and says how it departs from all three. It is not part of the planted-story plan in the folder above.

## Why a second corpus

The Batman corpus left four things open (paper v2, Sections 4.2 to 4.5):

1. Only 9 adaptations with a documented source, so the test of whether neurons add anything to words had little power.
2. Wikipedia describes a film and its source comic in much the same words, so word counts already find most sources.
3. Names had to be replaced by placeholder codes, and the codes leave a mark of their own.
4. The models have read these summaries. The largest names the exact film for every ordinary plot.

## The four options

| | Sherlock Holmes | Cross-language remakes | Indian cross-industry remakes | Transposed retellings |
| --- | --- | --- | --- | --- |
| What it is | 60 Conan Doyle stories and the screen works that draw on them | A film and its remake in another language or industry | The same corpus under stricter rules | Renamed, relocated retellings of 48 classics (West Side Story from Romeo and Juliet) |
| Documented pairs | 33 adaptations; about 20 firmly documented, on 13 stories | 105 (plus 24 loose, 54 with copied text) | 78 (plus 17 famous East Asian pairs) | 80, about half with a flat credit |
| Candidate sources | 61 | 92, plus 2,977 background plots | 70, or every film of the industry | 72 |
| Identity test | Yes: 56 screen works against about 160 detective films (Batman had 8) | Not in the paper's sense | Not in the paper's sense | For five to eight small lineages |
| Selection test | Yes: 60 early works (about 75% power, against 27%) | No | No | No |
| Do names differ by themselves? | No: Holmes and Watson stay, so the codes return | Yes | Yes (31 of 78 pairs share no name) | Mostly |
| Do words already find the source? | First for 16 of 32; misses mostly the weakly documented pairs | First for 82 to 85% | First for 62 of 78 | As often as on Batman (12% of wrong sources beat the true one) |
| Will the models recognise them? | Almost certainly | Indian films probably less; not measured | The same, with a famous block set apart | Worst: the links themselves are common knowledge |
| Where the texts come from | Live Wikipedia (checked 8 November 2026) | All plots are in the local film table | All plots are in the local film table | 74 of 80 films local; 59 source texts still to fetch |

The word figures are the scouts' own quick TF-IDF measures on Wikipedia plot text, made to size the problem. They are not results of the study's method and no neuron was involved.

## What the two judges said

They agree on the facts and disagree on which comes first.

- **The judge standing in for the project's aims chose Sherlock Holmes.** It is the only franchise among the four and the only option where identity, heredity and selection all run as designed. It is a better-powered replication, and it repeats three of Batman's limits: the codes return, the stories are famous, and words find half the sources.
- **The statistician chose the cross-language remakes**, built by the stricter rules of the Indian proposal. It is the only option with enough pairs (about 100 on about 90 sources) for a null on "do neurons add to words" to mean something. It cannot test identity or selection.
- **Both would run the other as a second arm.** Running the models is cheap in either case. The work is in building the lists and freezing a plan.

## Conditions both judges attached

For the remakes:

- Rebuild the pair list from the lead of each remake's Wikipedia article, not from memory. The statistician read 40 leads drawn at random and 39 named the claimed source. One listed pair is wrong: Kushi (2001) was planned as a bilingual with the Tamil film and is not called a remake.
- Move pairs whose summary was copied from the source's article into a positive-control set (10% or more shared five-word sequences), by a rule fixed beforehand.
- Recast the residual test. True pairs are far more word-similar than wrong ones (median TF-IDF cosine 0.20 against 0.05), so a regression fitted on wrong pairs would extrapolate. Test against each remake's nearest candidates by words, among several thousand films of the same industry and period, and put word-only measures through the same test in the units' place.
- Ask each model which film a text is, and which film it remakes, before anything else. Report every result separately for pairs it does and does not name.
- Strip actors' names in brackets. Check that each row of the film table holds the right film's plot: some do not.
- Never choose or drop a pair by its word-overlap rank. The lists here carry such columns for sizing only.

For Holmes:

- Count only the firmly documented tiers (straight adaptations and credited ones) as primary: about 20 adaptations on 13 stories, six of them The Hound of the Baskervilles. Derive the tiers from the articles' wording by a written rule, blind to the word ranks.
- Match the detective-film negatives by decade, and fetch them from live Wikipedia as the Holmes texts are.
- Add a second name-free version that swaps names for other ordinary names, not codes.

## What is in this folder

- `scouting_2026-11-08.json`: the four proposals and the two verdicts in full, as the scouts and judges returned them.
- `candidates/holmes_*.json`: the Holmes adaptations with their sources, tiers and the documenting sentence from each article; the 60 story articles checked; pageviews of the stories.
- `candidates/remakes_firm_105.csv`, `remakes_loose_24.csv`, `remakes_copied_54.csv`: the cross-language list, by the slugs of the local film table (`study/v2/cache/wiki_movie_plots.csv`, not committed).
- `candidates/remakes_indian_78.csv`: the stricter Indian list. `remakes_checked_live_40.json`: the 40 pairs whose leads the statistician read.
- `candidates/retellings_pilot_rows.csv`: the word-overlap pilot on 59 retellings.

Most pair lists were written from the scouts' memory and then checked only in part. Treat them as candidates to verify, not as a work list.

One thing the scouts found that changes the practicalities: live Wikipedia does answer from the cloud machine, slowly and with rate limits. The earlier note that it was unreachable was wrong.
