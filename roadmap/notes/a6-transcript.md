# a6-transcript: the Mike/Claude working session, and the images/ side experiment

Sources read: `study/conversation/conversation.json` (374 events, 2026-10-20 15:20 UTC to 2026-10-21 18:30 UTC; 47 Mike messages, 162 Claude replies, 558 one-line tool actions; raw tool output is not in the export), `images/README.md`, `images/vision_analyze.py`, `images/results/vision.json`, `vision_final.json`, `vision_labels.json`, plus spot checks of `preregistration/README.md`, `preregistration/PREREGISTRATION.md` section 15, `study/README.md` and `study/paper/paper.md`. "#N" below is the event index in `conversation.json`. Numbers quoted from the transcript are as stated in the session; several early ones were later corrected, so re-check any of them against `study/results/` before reuse.

## Bottom line

- Mike's goal from his first message (#0) is a measurable memetics: he thinks memetics failed because nobody could say where a meme starts and stops, and he proposes to let a model's neurons draw the boundary. He stated P2 as a test ("see whether the same memes light up in other models") but also gave himself a fallback in the same message: it "might also just be useful in general if you just define a meme as whatever qwen says a meme is." So P2 is a hope, not a requirement, in his own framing.
- The session's arc is enthusiasm, then a sobering robustness review (#206, #210): the numbers were honest, but plain word overlap found the source comics as well as the neurons did, the cross-model agreement was mostly shared name and word detectors, and "ideas, not words" rested on a handful of neurons. The claim that would justify the method (neurons follow a meme when the wording changes) was never tested at scale.
- Mike's instruction to "get rid of the word counting stuff" (#63) is the session's most costly scope decision: the pre-registered "beyond words" control went unrun until the review, and when run it failed for Qwen and passed only for Llama (#210).
- About 16 follow-up ideas were floated in under 40 minutes (00:38 to 01:15 UTC on 21 Oct). One was dropped by Mike (forecasting), one was designed and paused on cost (planted-meme stories: about 4,400 agent sessions and 3 days), the wording-change, culture-wide, mutation and selection tests were designed and deferred, and at least three (Die Hard offspring test, real algospeak test, mimetics carrier test) never reached the written plan at all.
- What Mike says he actually values (#227 to #244, #316) is descriptive: what Batman is made of (canon), how the Batmans are related, when Batman mutates, what Batman-like things exist in the culture, and spotting a hit's offspring. He explicitly let forecasting go.
- The image experiment is weak evidence for anything: one poster per work, no cheap rival, a medium confound, the BATMAN wordmark counted as a "meme", a cross-model twin score of 0.45 against 0.36 for random neurons (`images/results/vision.json`), and no link at all between the text memes and the image units. It is not a cross-modal test of P1 or P2.

## Findings

### 1. Mike's stated goals and beliefs (in his words, sparingly)

- **Why memetics is not a science** (#0): "you basically can't define what is a meme ... and therefore it can't be a science". Without a unit you cannot test whether a meme replicated or will replicate.
- **The proposal** (#0): a meme is "what neurons fire"; with that, do "comparative phylogenetics or phylo-memetics" and plot a meme's evolution, "or at least ... predict ... whether something is going to go viral".
- **P2 as a testable side question** (#0): seeing the same memes in other models "would tell us whether this hypothesis was correct that you could use model boundary". He was struck by the first cross-model numbers (#50: "that's pretty crazy as well if that's true").
- **Fuzziness is the point** (#47): word overlap is "less useful ... because it's not semantically related"; a vengeance neuron should fire for revenge; "that's what I was excited about, is like the fuzziness of this". Repeated at #217: "isnt the point that you dont have to count the words?"
- **A misbelief Claude corrected** (#217, #221): Mike assumed word counting meant someone chose the words. It did not (all words, rarer ones weighted more). The paper's line that neurons were units "the analyst didn't hand-code" was dropped for that reason.
- **Visual and concrete first** (#0): "Make it a visual experiment because I think that will be easier for people to understand", Batman as the case. At #316 he asked why the canon, genome, family tree, spread, across-culture and neuron-example figures were missing from the paper; they were then added.
- **Credibility over publication** (#191): "I dont care about the journals, I just want to know if the finding is robust and good and credible"; (#196) "make it as strong as possible".
- **What it is for, after the review**: identifying "what batman actually is, and the shared history between batmans" (#227); "seeing when batman is mutating" (#229); "a way of measuring cultural selection" (#231); finding "what batman like things exist in the culture" and the "billionaire vigilante orphans" family (#233); spotting when "a cultural pregenitor that was very popular has a new offspring" (#244).
- **Method must earn its keep** (#236): the paper needs "the experiment that shows how much more accurate or cheaper this is vs vectors, keywords"; (#239) test with new stories "where we put the right memes in"; rivals to include "recursive summarization with luna".
- **Accuracy is definition-dependent** (#242): "llm reading might be the most accurate I guess but it depends on how you define accurate".
- **Memetics plus mimetics** (his 2023 Reddit post, pasted at #222): memetics is what is transmitted, mimetics is who transmits; both matter. The post's anecdote: keyword tracking of an extremist group falsely shows it fading because the group changes its words; follower-graph tracking alone gives false positives; the working method seeds a cluster with keywords, adds the follower graph, then prunes sparse layers.
- **Housekeeping decisions**: open source under MIT with code, data and paper (#118); publish the whole conversation (#246); credit Claude Opus 5.5 as co-author (#183); do not cite his self-published book (#272); his shared links are public and may be linked (#361).

### 2. Ledger of ideas, hypotheses and experiments

Status key: DONE, PAUSED (designed and reviewed, not run), DEFERRED (designed, in `preregistration/PREREGISTRATION.md` section 15), DROPPED, UNPLANNED (floated, never written into the plan), UNCLEAR.

| # | Idea (who, where) | Status | Reason / outcome |
| --- | --- | --- | --- |
| 1 | Pilot: neurons as memes on 32 Batman works, 13 relatives, 69 ordinary films; three models (Mike #0, Claude #35) | DONE | Qwen3-8B gave 14,843 candidate memes (#15; `study/README.md` line 20) |
| 2 | Cross-model twins, Qwen vs Llama (Mike #0) | DONE | Top 33 memes: median r 0.82, chance 0.14. All 14,843: 0.40 vs 0.36 for non-meme neurons (#62). Later: agreement "collapses once names are removed" (#210) |
| 3 | Steering check on the 31 named neurons, Goodfire's stronger test (Claude #44) | UNPLANNED, never run | Offered, not taken up. `study/paper/paper.md` line 254 says labels were "never validated by steering" |
| 4 | Rewording test on 31 memes and revenge/vengeance test (Mike #47) | DONE, small | 11 survive in Qwen, 9 in Llama, 4 in both; Llama revenge neuron 10 of 12, Qwen 4 of 12; one rewording per meme written by Claude itself (#62) |
| 5 | "Phase 2": reword or translate the whole corpus, keep surviving memes, rerun trees (Claude #35, #62) | Partly DONE (name-stripping), rest DEFERRED | Became name-free texts for E1/E2; full drift and translation deferred (T1b, T1c) |
| 6 | E1 identity, E2 heredity, E3 selection, pre-registered (Claude #64) | DONE | E1 first reported backwards (#182), then found to be a scoring artefact: word counts separate name-free Batman from other superheroes at AUC 0.80 (#210). E2 holds in 99.6% of 30,240 variants (#210; `study/README.md` line 50) but word overlap ties. E3 had "almost no statistical power" (15 early works) |
| 7 | Drop the word-counting comparison (Mike #63) | DONE, then REVERSED | Reversed after the robustness review; word and LSA baselines restored (#216, #260) |
| 8 | Canon tiers: must-have vs optional (Mike #63) | DONE | 56 must-have, 194 usually-there, about 14,600 optional (#326). Random 32-film sets have no such core (#228) |
| 9 | Spread of memes over time (Mike #63) | DONE, descriptive | "First appearance" means first in the 32-work sample; gap 1967 to 1985 (#326) |
| 10 | Image experiment with vision models (Mike #63, #69) | DONE, exploratory | Cut from the paper (#194 to #198), own page, own repo (#216), later folded back into `images/` (git commits e6de34e, 6813b31) |
| 11 | Paper in the style of the folktale study (Mike #75) | DONE | v1 frozen 21 Oct 2026 |
| 12 | Referee panel on publishability (Claude #189) | DROPPED mid-run | Mike does not care about journals (#191); replaced by a re-run-everything robustness workflow (#192) |
| 13 | Robustness review: baselines, multiverse, blind label test, code audit (Claude #192) | DONE | Blind label test 79% (56 of 71); placebo bug fixed; chance was 39 to 44%, not the 47% first reported (#210, #260) |
| 14 | Cross-language test: Chinese film summaries vs English comics (Claude #210) | DEFERRED (T1c, T2-zh) | Word overlap falls to chance by design, so it is the cleanest fuzziness test |
| 15 | Shakespeare loose adaptations (Lion King/Hamlet, West Side Story) (Claude #210) | DEFERRED (T1d) | 54 renamed adaptations of 13 plays in the plan |
| 16 | Vocabulary-drift curve, 0 to 100% of content words replaced (Claude #223, from Mike's Reddit post) | DEFERRED (T1b) | |
| 17 | Mimetics: track carriers (Miller, Moore, Nolan) alongside memes (Claude #223) | UNPLANNED | "Needs a larger lineage"; the word "mimetic" does not appear in `PREREGISTRATION.md` |
| 18 | Forecasting: thousands of films 1950 to 2025, hold out recent years, beat a persistence baseline; rules from Mike's earlier notes (Lindy, roughly 30-year returns, memeplexes, "just past the familiar") (Claude #226) | DROPPED by Mike (#227) | "ok if not forecast thats fine"; 32 works far too few |
| 19 | Finish labelling the canon: 13 unlabelled core memes, 9 mixed, 3 Wikipedia-style; blind test on the core (Claude #228) | UNCLEAR | Promised; I found no later confirmation in the transcript |
| 20 | Mutation analysis: mutations per work, which stick, innovators (Dark Knight Returns, Year One, Killing Joke), distance from canon over time (Mike #229, Claude #230) | DEFERRED (T4) | Needs all 71 Batman works; paper mentions it only as future design (line 250) |
| 21 | Neutral-model selection test, random copying as for baby names (Claude #232) | DEFERRED (T4, S19) | 32 works can detect only strong selection |
| 22 | Culture-wide "Batman detector" and the orphan-vigilante family test (Mike #233, Claude #235) | DEFERRED (T3) | Floated at about 1,500 works; plan has 2,544 masked works. Preliminary with-names table: Puss in Boots carries none of six Batman memes (#235) |
| 23 | Head-to-head vs keywords, vectors, topic models, with cost (Mike #236) | PAUSED | Core of the planted-stories plan (H1, H2, H6) |
| 24 | Planted-meme stories with controls, decoys, three subtlety levels (Mike #239, Claude #241) | PAUSED by Mike (#308) | Estimate grew from "about a day" to about 3 days and 4,400 agent sessions (#304). "lets just have the test planned for now" |
| 25 | LLM reading and recursive summarisation as rivals (Mike #239) | PAUSED (in the plan) | "Luna" was never identified; Claude asked twice (#241, #243) and got no answer |
| 26 | Real algospeak test ("unalive", "seggs") (Claude #243) | UNPLANNED | Zero hits for "algospeak" or "unalive" in `PREREGISTRATION.md` |
| 27 | Synthetic community with evolving and invented code words (Claude #243) | PAUSED, reduced | Survives as the planted-stories "code-word condition", a reported secondary |
| 28 | Overlap analysis: what each method finds that the others miss, detection vs discovery (Claude #243) | PAUSED (secondary) | |
| 29 | "Die Hard on a ___" hidden-offspring test (Claude #245) | UNPLANNED | Promised for "the next design round"; zero hits in `PREREGISTRATION.md` after the plan was narrowed |
| 30 | Blind "how nameable are the units" test and long-text condition (#304) | Proposed for deferral | About 1,300 agent sessions |
| 31 | Concept agreement without names, Llama partners vs lexically matched nulls (T5) | DEFERRED | The proper P2 test; not run |
| 32 | Natural annotated summaries (T2-N), 237 texts, two annotators | DEFERRED | |
| 33 | Mike's 2 to 3 hour human audit of generated stories (#283, #284) | Agreed, NOT DONE | Test paused before any story was written |
| 34 | Enron-email memetics-plus-mimetics graph analysis (Mike's 2023 Reddit post, #222) | Never started | Mentioned only as his older intention |

### 3. Dead ends and errors the session itself surfaced

- **E1 stated backwards** (#182): Claude misread an output number; a 30-agent fact-check over about 200 claims caught it, plus 22 smaller errors.
- **E1 null was an artefact of scoring** (#210): plain word counts separate name-free Batman from other superheroes (AUC 0.80), mostly through crime-city words (asylum, police, commissioner).
- **Pre-registered control not run** (#210): the "beyond words" test existed on paper only until the review.
- **Two comparisons were true by construction** and were replaced (#216). A placebo bug and overstated chance rates were fixed (#260).
- **Cross-model chart used weak baselines**; replaced by matched ones (random neurons, single-word-tracking neurons) (#269).
- **Era-matched null**: some Qwen leave-one-out runs slip above p = 0.05; Llama holds (#260).
- **False provenance claim**: the paper first said "phylomemetic" was borrowed from Chavalarias and Cointet; corrected to "used earlier for a different method" (#269).
- **Label contamination**: "franchise roll-call" tracks how Wikipedia writes about games; the Lucius Fox neuron fires on Zorro's nickname "The Fox"; the "Alfred" neuron fires on the Kingpin and Superman's father (#44, #228). Character neurons sit in layers 0 to 1, "close to plain name detectors" (#44).
- **Mike's slide family "half holds up"**: Puss in Boots shares essentially nothing with Batman (#35, #235).
- **Time estimates were unreliable**: planted stories went from one day to three (#304); Mike's reaction, "why would it take 3 days" (#306), preceded the pause.
- **Operational friction**: Wikipedia rate limiting; disk 98% full (#35); a context reset mid-session (#168); Reddit blocked in both browsers (#221); Zorro's source text swapped twice for length (#5, #9).

### 4. Scope decisions

- Batman as the case, Wikipedia plot summaries as the text (#0, #35).
- Images out of the paper, then out of the repo, then back in as `images/` (#194, #208, git log).
- Paper reorganised around what Batman is, how Batmans are related, and mutation, with selection demoted to "a brief honest null" (#228).
- Of three run options, Mike chose "Planted stories only" (#284), then paused even that (#308).
- Repo layout: `study/`, `preregistration/`, images separate (#329).
- Book not cited anywhere outside the transcript (#272); transcript kept as said (#307).
- The source post by Cammarata moved to the paper's opening (#308). The Jha et al. paper was added in the last minutes of the session after Mike shared an X post (#366 to #371).

### 5. Cost and hardware

- **Machine**: Apple M5 Pro MacBook, 24 GB memory; MLX, 4-bit models (`study/README.md` line 104). Models: Qwen3-8B (36 layers, 12,288 MLP neurons per layer, #44), Llama 3.1 8B, Qwen3-1.7B.
- **The P1 source figure does not transfer automatically**: the "about 90% of neurons have one main job" claim was for Qwen3-32B; the study used an 8B model compressed to 4-bit and "didn't audit it" (#44).
- **Extraction is cheap**: rebuilding all neuron recordings takes about 12 minutes (#35); the three experiments on both models about half an hour (#98). Storage is the pinch: 7.4 GB of recordings early (#35), about 15 GB later (#182), on a disk that was 98% full.
- **Scale-up estimates** (Claude's, unverified): about 1,500 works is "a few hours per model on the laptop" plus about an hour of agent work for name removal (#235); several thousand films is "a few hours of extraction" (#226).
- **The expensive part is not neurons**: the planted-stories plan needs about 25 hours of local model time, mostly LLM reading meme by meme, about 4,400 agent sessions at no more than 14 at once, about 3 days; trimmed version about 1.5 days (#307; `preregistration/README.md`). The full v2 plan was 3 to 4 days and about 7,300 sessions (#283). About 49 GB of models were downloaded for it (#294).
- **Claude's cost argument** (#237, #241): neurons lose to keywords and embeddings on compute; their possible edge is that one forward pass scores all roughly 15,000 memes, whereas an LLM judge needs a call per meme. This is an expectation, not a result.

### 6. Links Mike shared

| Link | What it is | Event |
| --- | --- | --- |
| https://already-clean.goodfire.com/ | Cammarata, "Neural Networks Are Already Clean" (page fetched this session; title confirmed) | #0 |
| https://royalsocietypublishing.org/rsos/article/3/1/150645/36447/Comparative-phylogenetic-analyses-uncover-the | da Silva and Tehrani 2016, the folktale phylogenetics paper (DOI 10.1098/rsos.150645, confirmed via Crossref) | #0 |
| https://docs.google.com/presentation/d/1GvP9ZuJV3lHkoj_1-7wd6gWojQO3K3x6WxjX8Zqw25A/ | Mike's talk deck on reverse-engineering creativity; source of the "Billionaire Vigilante Orphans" slide | #0 |
| Notion collection of his essays | Mike's earlier notes. The URL slug contains the book title, so it is not reproduced here; see the event | #90 |
| https://ambiguous-celery-039.notion.site/e4da53351c3442f18274588479d67b19?v=95abac9fd89e4b7da7aa5ba09711cfe0&p=f9e74c31af8849b3b7146fa372ed5800&pm=c | His essay on why memetics did not become a science (chapter title "Memetics as a Science", #370). Claude could not open it in the browser and read the same text from the attached PDF | #90 |
| https://www.reddit.com/r/memetics/comments/11hesgu/memetics_mimetics/ | His r/memetics post, "Memetics + Mimetics" (text pasted at #222) | #217 |
| https://x.com/emollick/status/1925744569636856072 | An X post by Ethan Mollick; the paper behind it is Jha et al. 2025 (arXiv 2505.12540, confirmed). I did not fetch the post itself | #366 |

### 7. Audit of `images/`

**What was tried** (`images/README.md`, `images/vision_analyze.py`)
- Two vision models: CLIP RN50x4 (3,840 channels of its last two conv stages) and DINOv2-base (36,864 MLP neurons, all 12 layers).
- 111 lead images from Wikipedia articles: 29 Batman, 13 relatives, 69 ordinary films. One image per work; 10 views each (full, four corners, mirrors).
- Presence rule copied from the text study: a unit is present when it beats all but one of the 69 film posters. Candidate "visual memes" are units present in at least 6 of 29 Batman images. Chance is a binomial with p = 2/70 per image, independent across images.
- Twins: for each CLIP candidate, the best-correlated DINOv2 unit over 1,110 views; baselines are random active CLIP units and shuffled views.
- Seven generated probe images (bats, beach, cape, kitchen, mansion, meadow, skyline), scored as a percentile of each unit's 111 corpus responses.
- Labels written by an AI reader from contact sheets of each unit's top 8 images, with a second pass. No blind label test, no steering.

**What was found** (all from `images/results/vision.json` unless noted)
- CLIP: 56 units at 6 or more Batman images (chance 0.56), 23 at 9 or more, 17 at 12 or more, 10 at 15 or more, 4 at 20 or more. The top unit (2929) is present in 28 of 29 Batman images and 1 relative.
- DINOv2: 416 units at 6 or more (chance 5.4), 32 at 9 or more, 2 at 12 or more, 0 at 15 or more.
- Cross-model twins: median r 0.451 for CLIP candidates, 0.361 for random active units, 0.128 shuffled. The top CLIP unit's twin has r 0.536 and a presence Jaccard of 0.438.
- Labels (`images/results/vision_labels.json`, 24 CLIP and 12 DINOv2): of the 24 CLIP labels, 9 are "mixed", 4 are medium or style (drawn comic cover, printed cover), 1 is typography (the BATMAN wordmark); only 12 are flagged Batman-specific and 8 are high confidence. Of the 12 DINOv2 labels, 3 are Batman-specific.
- Probes: the cowl unit scores 73.0 on bats, mansion and skyline, 74.8 on cape, and 3.6, 0.9, 1.8 on beach, kitchen, meadow. The masked-or-hooded unit (2887) scores 95.5 on the cape probe.

**Blunt reading**
- The visual "memes" are costume, logo, wordmark and medium. Nothing here touches story memes (origin wound, secret identity), which is what the text study is about.
- The wordmark is the name in pixel form. CLIP is known to read text in images (Goh et al. 2021), and there is no image equivalent of the name-free control. This is the same trap the text study fell into with character names.
- Medium confound: the Batman set mixes comic covers, game box art and posters, but the reference set is 69 film posters. A "drawn comic cover" unit passes the threshold simply because the baseline has no comics.
- No cheap rival at all: no CLIP-embedding nearest-neighbour or linear probe, no colour histogram, no OCR, no caption word overlap. By the brief's own standard the design cannot say whether single units add anything.
- The probe result is weaker than the README's wording ("responds only moderately"). 73.0 is 81 of 111, and the cowl unit is present in exactly 30 corpus images (28 Batman, 1 relative, 1 film). So each of three gothic scenes with no Batman in it outscores all 81 corpus images where the unit counts as absent, including the film that sets the threshold. By the experiment's own presence rule the cowl "meme" is therefore present in those three probes, though below every real Batman image. The unit is at least partly a dark-gothic detector, or generated images shift its response. (My inference from the percentiles; not checked against raw activations, which are not committed.)
- "Concept neurons seem to need language" is a hunch. The CLIP versus DINOv2 contrast changes training objective, architecture (conv channels vs ViT MLP neurons), unit count (3,840 vs 36,864) and layer coverage (last two stages vs all layers) at once.
- The twin test barely clears its baseline (0.45 vs 0.36). For P2 in vision this is close to a null.
- The text memes and the image units were never compared on the same 111 works. No cross-modal claim can be made from this folder.

**What a proper cross-modal test of P1 and P2 would need**
1. Images that depict the meme, not marketing art: comic panels, film stills or generated scenes, several per work, with human annotation of which memes each shows.
2. Ground truth by construction: generated scenes with a meme planted or absent, plus decoys (a bat with no Batman, a waiter who is not the guardian butler), mirroring the paused planted-stories design.
3. A name-free control for pixels: logos and lettering masked or inpainted; a typography-only decoy (the word on a blank card).
4. A medium-matched reference set (comic covers against comic covers, posters against posters).
5. For P1 across modalities, one model with shared weights for both inputs (a vision-language model), so the same neuron can be read on text tokens and image tokens. Prior work reports such units exist in some models (Goh et al. 2021; Schwettmann et al. 2023; Pan et al. 2023), which makes the test feasible, not settled.
6. For P2, twins between two independently trained models, scored against matched nulls (random active units, units tracking colour or medium), on held-out images. Rosetta Neurons (Dravid et al. 2023) is the method precedent.
7. Rivals in every comparison: CLIP zero-shot similarity to a one-line meme description; a captioner followed by TF-IDF; a vision-language model simply asked "does this image show X?"; a colour histogram.
8. A blind label test and a causal check (ablation or feature visualisation), as the text study's labels had the first but not the second.
9. Pre-registration, since the text study's history shows how much the unregistered choices mattered.

## What this means for P1 and P2

**P1 (a meme can be measured by which neurons fire).**
- For: single neurons and neuron families give nameable units; a blind judge matched labels to unseen sentences 79% of the time (56 of 71, #210); Batman has a core of units present in nearly all 32 works where random 32-film sets have none (#228); a few neurons (funeral, coin flip, secret identity, joke, Llama's revenge neuron) keep firing under rewording (#62).
- Against, on the study's own evidence: on every test where a cheap rival was run, word overlap tied or won (source-comic recovery, name-free identity). Many "memes" are name or word detectors. Labels are hypotheses and were never steered. The cleanliness figure P1 rests on was measured on a 32B model, not the 4-bit 8B model used.
- Net: P1 holds as "a countable, nameable unit exists". It does not yet hold as "a better unit than words or embeddings". Mike's own reason for wanting neurons, fuzziness under wording change, is exactly the part with the thinnest evidence (11 of 31 hand-reworded memes in one model, one rewording each).
- Condition under which P1 could matter: when vocabulary changes but the idea stays (his pizzagate example). Batman summaries on Wikipedia are, as Claude put it, "a group that never changed its vocabulary" (#223), so the study could not show the advantage even if it exists.

**P2 (models converge, so the measurement generalises).**
- For: top 33 Qwen memes have Llama twins at median r 0.82 against chance 0.14; grid agreement 87% against 52% (#62).
- Against: across all 14,843 candidates the median is 0.40 against 0.36 for neurons never selected as memes (#62), a tiny margin; the agreement "collapses once names are removed" (#210); in vision, CLIP and DINOv2 twins score 0.45 against 0.36 for random units (`images/results/vision.json`). The Jha et al. paper was added in the session's last minutes from a social-media post (#366 to #371) and concerns whole-embedding geometry, not neuron identity.
- Net: at the single-neuron level the study's evidence for P2 is weak and mostly lexical. The proper test (concept units, no names, lexically matched nulls: T5) is designed and unrun.
- Mike's fallback (#0) makes P2 optional for a working science: fix one reference model and define memes by it. The cost is that every result is then tied to one model and one quantisation.

## Experiment seeds

1. **Steer the named neurons.** Question: does turning a labelled neuron up make the model write its meme? Data: the 31 named and 56 core Qwen3-8B units. Method: activation steering at several strengths on neutral prompts; blind judge picks the label from a list. Rival to beat: the existing read-the-top-sentences label (79% blind match) and steering a matched-activity random neuron. Falsifier: judged match no better than the random-neuron control, or steering only inserts the name or keyword. Cost: hours on the laptop; no agents beyond judging.
2. **Run the trimmed planted-stories test.** Question: when keywords are banned, do neurons detect planted memes better than keyword and lexical methods and within 0.05 AUC of embeddings (H1, H2, H6)? Data: generated stories per `preregistration/planted_story_protocol.md`. Method: as pre-registered (v3). Rivals: evasion-aware keyword analyst, lexical classifier, Qwen3-Embedding-4B, multilingual-e5-large, direct LLM judge, recursive summarisation. Falsifier: H1 or H2 fails, or a rival is at least as accurate and no dearer. Cost: about 1.5 days trimmed, 3 days full; about 3 hours of Mike's audit time.
3. **Die Hard's children.** Question: can neurons find a hit's renamed offspring? Data: name-free Wikipedia plots of a few hundred action films, with documented "Die Hard on a ..." descendants as positives. Method: rank films by share of Die Hard's distinctive units. Rivals: TF-IDF cosine, sentence embeddings, an LLM asked directly. Falsifier: neuron recall at k no better than TF-IDF on name-free text. Cost: under an hour of extraction; needs a defensible positives list first.
4. **Real algospeak.** Question: does a death or sex neuron fire on "unalive" or "seggs" sentences? Data: a few hundred matched sentence pairs (standard word, algospeak, neutral control). Method: per-neuron activation contrast. Rivals: seed keyword list, sentence embeddings, LLM reading. Falsifier: no activation gap over controls, or embeddings do as well at far lower cost. Cost: minutes of compute; the labour is building clean pairs.
5. **Text-image meme agreement on the existing 111 works.** Question: do works that share text memes also share image units, beyond medium and franchise? Data: the committed corpus and `images/data/works.jsonl`. Method: Mantel correlation between text-meme and image-unit work distances, partialling out medium and group. Rivals: TF-IDF distance vs CLIP whole-embedding distance. Falsifier: neuron-level agreement no higher than the embedding-level one, or it vanishes once medium is partialled out. Cost: re-extraction only (about 12 minutes text, similar for vision). Expect a null; it is cheap enough to be worth knowing.
6. **Planted images in one vision-language model.** Question: does the same neuron fire for a meme stated in text and shown in an image? Data: generated scenes with planted memes, decoys, masked lettering. Method: read MLP units on text tokens and image tokens in one open VLM; test with matched nulls. Rivals: CLIP zero-shot similarity, captioner plus TF-IDF, the VLM asked directly. Falsifier: no unit beats the matched null on both modalities, or the VLM's direct answer dominates on accuracy and cost. Cost: a VLM that fits in 24 GB; a few hundred generated images.
7. **Image experiment redone with controls.** Question: do Batman units survive when lettering is masked and the reference set is medium-matched? Data: the same images plus comic-cover and box-art baselines. Method: repeat `vision_analyze.py` with masks and matched baselines. Rivals: colour histogram, CLIP-embedding linear probe, OCR. Falsifier: counts fall to chance, or the histogram classifies Batman as well. Cost: an afternoon.
8. **Concept twins without names (T5).** Question: do Qwen concept units have Llama partners that agree on held-out, name-free random films better than lexically matched null units? Data and code already in `preregistration/` (`analysis.run_t5`). Rival: units that track a single word. Falsifier: no margin over the lexical null. Cost: small; needs its own pre-registration.
9. **Carrier test (mimetics).** Question: after accounting for shared memes, does sharing a writer or director predict which mutations a later work inherits? Data: the extended 71-work Batman list with credits. Method: regress inheritance on carrier links. Rival: credits metadata alone, and word overlap. Falsifier: carrier adds nothing over meme overlap, or word overlap predicts inheritance as well as neurons. Cost: low compute; 71 works is thin, so treat as a pilot.

## Key sources

| Citation | URL / DOI | Why it matters | Priority |
| --- | --- | --- | --- |
| `study/conversation/conversation.json` (this repo) | committed file | The only record of why each scope decision was taken and which ideas fell away | A |
| `preregistration/PREREGISTRATION.md` section 15 and `preregistration/README.md` | committed files | Authoritative list of what is paused or deferred, with costs | A |
| `images/results/vision.json` | committed file | Every image-experiment number | B |
| Cammarata, "Neural Networks Are Already Clean" (Goodfire, 2026) | https://already-clean.goodfire.com/ | Source of P1; its cleanliness figure is for a larger model than the study used | A |
| da Silva and Tehrani (2016), Comparative phylogenetic analyses uncover the ancient roots of Indo-European folktales, R. Soc. Open Sci. | https://doi.org/10.1098/rsos.150645 | The template Mike asked the paper to follow | B |
| Jha, Zhang, Shmatikov and Morris (2025), Harnessing the Universal Geometry of Embeddings | https://arxiv.org/abs/2505.12540 | Added at the end of the session as P2 support; it is about embedding spaces, not neurons | B |
| Gurnee et al. (2024), Universal Neurons in GPT2 Language Models | https://arxiv.org/abs/2401.12181 | P2's neuron-level source; existence confirmed, content not re-read here | B |
| Huh, Cheung, Wang and Isola (2024), The Platonic Representation Hypothesis | https://arxiv.org/abs/2405.07987 | P2's representation-level source; existence confirmed | B |
| Goh et al. (2021), Multimodal Neurons in Artificial Neural Networks, Distill | https://doi.org/10.23915/distill.00030 | CLIP units respond to a concept as photo, drawing and written word; the reason the wordmark confound matters | A (for cross-modal) |
| Schwettmann et al. (2023), Multimodal Neurons in Pretrained Text-Only Transformers | https://arxiv.org/abs/2308.01544 | Precedent for reading the same unit across modalities | B |
| Pan et al. (2023), Finding and Editing Multi-Modal Neurons in Pre-Trained Transformers | https://arxiv.org/abs/2311.07470 | Method precedent for seed 6 | C |
| Dravid et al. (2023), Rosetta Neurons: Mining the Common Units in a Model Zoo | https://arxiv.org/abs/2306.09346 | Method precedent for cross-model unit matching in vision | B |
| Bentley, Hahn and Shennan (2004), Random drift and culture change, Proc. R. Soc. B | https://doi.org/10.1098/rspb.2004.2746 | The neutral-model approach Claude proposed for the selection test (#232) | B |
| Chavalarias and Cointet (2013), Phylomemetic Patterns in Science Evolution, PLoS ONE | https://doi.org/10.1371/journal.pone.0054847 | Prior use of "phylomemetic" for a different method; the paper first mis-stated this | C |
| Mike Taylor, "Memetics + Mimetics", r/memetics (2023) | https://www.reddit.com/r/memetics/comments/11hesgu/memetics_mimetics/ | The vocabulary-evasion case that motivates the whole "beyond words" programme; text is in the transcript (#222), page itself not fetched | A |

Existence of each paper above was confirmed this session through the arXiv API or Crossref. I did not re-read their contents, so the one-line descriptions of Goh, Schwettmann, Pan and Dravid rest on titles and my prior knowledge.

## Fog

- **What is "Luna"?** Mike named it twice as a recursive-summarisation rival (#239, #242). Claude asked twice and got no answer. It may be a tool Mike uses or a speech-to-text slip.
- **Was the canon labelling finished?** Claude promised to label the 13 unlabelled core memes and blind-test the core (#228). I found no confirmation.
- **Qwen3-1.7B.** It was run, but the transcript barely discusses it. A same-family size comparison bears on both P1 (cleanliness vs size) and P2.
- **The transcript omits raw tool output**, workflow reports and Claude's reasoning. Robustness details live in `study/results/robustness/`, which I did not read.
- **Scale-up time estimates** (#226, #235) are Claude's guesses and were wrong by a factor of three for the planted-stories plan.
- **Mike's deck and Notion essays** were not opened here; their contents are known only through Claude's summaries in the transcript.
- **The X post** was not fetched; only the paper behind it was confirmed.
- **My reading of the probe percentiles** (gothic no-Batman probes clearing the cowl unit's presence threshold) is arithmetic on committed numbers, not a check of activations.
- **Whether a 32B model fits** on the 24 GB laptop at 4-bit is unknown; it matters because the cleanliness claim was made at that size.
- **The image repo** `hammer-mt/phylomemetics-images` may still exist publicly alongside the folded-in `images/`; not checked.
