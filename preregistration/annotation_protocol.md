# Annotation protocol, v3

Agents and the author provide these judgements:

- A1: the blind recognisability reads that admit planted stories into the benchmark;
- A2: the exhaustive ground truth for natural plot summaries (T2-N): **deferred**, not part of this pre-registration;
- A3: the blind label test of how nameable each method's units are (T6), on the T2 stories;
- A4 and A5: two labelled Claude upper bounds (a judge and a ban-list-aligned keyword query);
- A6: the typing of documented Batman relations (T3): **deferred**, not part of this pre-registration;
- A7: the author's human audit of planted story-by-meme cells, through a local page;
- A8: the example sentences for every meme (before F0);
- A9: the salience ratings of planted-story slots (Q6).

v3 (2026-10-21) narrows the study to the planted stories (`PREREGISTRATION.md` sections 14.4 and 15): A2 and A6 are kept as written in v2 for a later phase and are not run; A3 moves from the T3 pool to the T2 stories; A7 covers T2 cells only.

Ground rules for every agent step:

- Every call is a fresh session. It sees only the material this protocol gives it, never a method's output, a design file or another annotator's answer, except where a step says otherwise (adjudication).
- Every output file records the agent model and the date.
- Items are shuffled with `numpy.random.default_rng(20261020)` before they are batched.
- Agreement is reported for every step with two or more annotators, whatever it turns out to be.
- Two Claude sessions agreeing measures one model's consistency, not independent agreement. Every gating judgement therefore also gets a cross-family reading (Gemma 2 9B, `readers.py check`) or a human audit (A7), and that agreement is reported.

## A1. Blind recognisability reads (T2 planted stories)

Defined in `planted_story_protocol.md`, checks Q4 and Q5: two Claude readers and one cross-family reader per version. Cohen's kappa between the two Claude readers, and Claude-versus-cross-family agreement, are reported per version type (explicit, banned, implicit, coded, control, decoy) over all story-by-meme cells.

## A2. Exhaustive annotation of natural plot summaries (T2-N): deferred

> **Deferred (prereg-v3).** T2-N is not part of this pre-registration (`PREREGISTRATION.md` section 15). This step is kept as written in v2 for a later phase and is not run.

**Works.** The 237 works of `culture_pool.json` with `t2n: true` (set `t2n_works`, texts through `analysis.t3_text`): the 39 Batman-lineage works, the 57 relatives, the 21 orphan-family works, and 40 pool works from each of the random, superhero and pulp-crime strata drawn in sha256 order (so that natural positives and negatives also occur outside Batman). Each work is annotated in full, so the ground truth does not depend on any method's ranking.

**Annotators.** Two independent Claude annotators per work. Each sees the masked work as numbered sentences and the 34 meme labels and definitions, but no keywords, no examples and no method output.

```
Below is a plot summary (names are replaced by codes such as X1) and a list of 34 story elements. Judge
ONLY from the text in front of you, not from anything you may know about the work. For each numbered
sentence, list the ids of the elements that the sentence shows happening, however it is described; an
element that spans several sentences is marked on each of them. For every element you mark, quote the
words that show it. Most sentences contain none. Return JSON: {"sentences": [[ids], ...],
"evidence": {"<id>": "<quoted words>", ...}} with one list per sentence, in order.
```

**Adjudication.** A third fresh session sees the work and both annotations for the work-by-meme cells where the two disagree about whether the meme appears anywhere. It decides each disputed cell, marks the sentences and quotes the evidence. A meme is **present** when both annotators mark it, or the adjudicator does. The cross-family reader also gives work-level presence (`check --task xannotate`); its agreement is reported.

**Inclusion.** A meme enters the T2-N mean only with at least 10 positive and 10 negative works and annotator kappa of at least 0.4. Kappa is reported at work-by-meme and sentence-by-meme level, overall and per meme. Outputs: `preregistration/agent/annotation/t2n_raw.jsonl` and `t2n_adjudicated.jsonl`; `build_lists.py texts t2n` adds them to `t2n_works`. T2-N uses every method's **far** detectors.

## A3. Blind label test (T6 interpretability)

**Units.** Two arms, each with one selection rule for every kind, scored on the T2 planted stories (all six versions of every valid plot; v2 used the deferred T3 pool):

| Kind | Random arm | Best-of arm |
|---|---|---|
| `neurons:qwen3-8b`, `neurons:llama3.1-8b` | the 400 random units per model in `dev_units.json` (`label_test_random`), in order | the 200 DEV meme families ranked by Batman enrichment (`label_test_best_of`), in order |
| `lda` (K = 100, fitted on the stories and the 69 reference films as **works**) | topics in seeded random order | topics by UMass coherence |
| `topics:qwen3emb-4b` (HDBSCAN clusters of the stories' sentence embeddings) | clusters in seeded random order | clusters by mean membership probability |

In both arms a unit qualifies only if it is present in between 1% and 20% of the stories (neurons and LDA topics: beating all but one of the 69 reference films, `t2_reference`; clusters: at least one sentence of the story in the cluster), and the first 40 that qualify are used.

**Top sentences.** For each unit, its 16 highest-scoring story sentences, at most one per plot (the six versions of a plot share their frame), duplicates removed, ranked by the unit's own score among its members: a neuron's activation, an LDA topic's sentence-level probability under the work-level model, a cluster's membership probability among its members. Ranks 1 to 8 go to the labeller; 9 to 16 are held out.

**Labeller.** One fresh session per batch of 10 units of one kind and arm, never told the kind:

```
Each group below lists 8 sentences that one feature of a text-analysis system responds to most. For each
group, write a label of at most 8 words naming what the sentences have in common, and one sentence
describing it. If they share nothing you can name, write "unclear". Return JSON: [{"group": g, "label":
"...", "description": "..."}].
```

**Judges.** Each labelled unit gets two items. Each pairs one of the unit's held-out sentences with a held-out sentence of a different unit of the same kind and arm. Two fresh judge sessions answer every item, seeing the label, the description and the two sentences in random order:

```
A feature of a text-analysis system was described as: LABEL: DESCRIPTION. Which of these two sentences is
the feature more likely to respond to? Answer A or B.
A: ...
B: ...
```

A judgement is correct when it picks the unit's own sentence; a unit labelled `unclear` scores all its judgements as incorrect. As a floor, a script answers every item by word overlap (TF-IDF cosine of label plus description with each sentence). `analysis.run_t6` reports, per kind and arm: accuracy with an exact binomial 95% interval, the share of `unclear` labels, the kappa between the two judges, an exact McNemar test against the floor on the same items, Fisher tests of each neuron kind against the other kinds, and labeller calls per usable label. Answers go to `preregistration/results/label_test_answers.json` as `{"kind", "arm", "unit", "item", "judge", "correct"}`.

## A4. Claude upper-bound judge (T2)

A judge reading all 6 versions of a subset of plots, chosen in design order while any meme it plants has been covered fewer than 6 times. A fresh session reads up to 4 stories, each from a different plot, and never two versions of one plot.

```
Here is a plot summary. For each of the story elements below, answer yes if it happens in the story,
however it is described, and no otherwise. (When a session reads several stories, it answers for each
story separately and returns one JSON object per story.) Each element has a definition and examples from
other stories. Return JSON: {"<id>": "yes" | "no", ...} for all 34.
<story>...</story>
{34 blocks: id, LABEL. DEFINITION. Examples: the main example set}
```

**When.** After the A7 freeze (the author's answers are committed and pushed), with the other TEST features of step 5 and before F3, so that no method output exists while the author audits. About 110 sessions.

Answers go to `features/t2_stories/llm_claude_judge.jsonl` (`p_yes` 1 or 0), scored like any method on the subset read, and reported as `judge:claude-upper-bound`. Like every feature file it is not committed (`features/` is not in git); its sha256 is recorded in the F3 feature manifest. Claude wrote and admitted the stories, so this is an upper bound. It is **not** a manipulation check (that role is A7's and the cross-family reader's), and it never counts as a rival in H6.

## A5. Ban-list-aligned keywords (T2)

One fresh session per meme receives `readers.KEYWORDS` (the plain analyst prompt) with the meme's label, definition and main examples. Parsed with `readers.parse_list` and saved as `features/t2_examples/keywords_ban-list-aligned.json`. Claude also wrote the ban lists, so this query is the one most likely to coincide with them; it is reported as `keywords:ban-list-aligned`, never as an upper bound.

## A6. Relation typing (T3): deferred

> **Deferred (prereg-v3).** T3 is not part of this pre-registration (`PREREGISTRATION.md` section 15), so relation typing is not run at P1. This step is kept as written in v2 for a later phase.

Two fresh coder sessions per documented relative in `documented_relatives.json` (36 entities) see the entity's name, relation and evidence sentences, and nothing about any method:

```
Below is a documented relation between Batman and another work or character, with the sentences that
document it. Classify the relation as exactly one of:
plot_role: the two share plot or role-level content (a hero with a secret identity, an orphan avenger,
  a masked crime-fighter's methods, a villain's story);
iconography_name: only a look, a costume, a device or a name was borrowed;
parody: one is a parody or pastiche of the other;
meta: a reference, casting or homage outside the story;
transitive: related to Batman only through another relative.
Return JSON: {"relation_type": "...", "reason": "<one sentence>"}
```

A third session adjudicates disagreements. Kappa is reported. Output: `preregistration/agent/relations/raw.jsonl` and `relation_types_final.jsonl`; `build_lists.py relations` merges them. S15's primary set is the plot_role and parody relations; iconography_name, meta and transitive relations are reported separately with a predicted null.

## A7. Human audit (the author, through a local page)

Mike, the author, checks a sample of planted story-by-meme cells himself. Nothing outside T2 is audited. It takes about 2 to 2.5 hours and can be split across sittings, one block at a time.

**When.** After freeze F2-T2 (the stories, reads and ratings are committed) and before any method is run on a TEST story (step 5), so the author has seen no method output. The sample and his answers are committed and pushed (freeze A7) before step 5 starts. The Claude upper bound (A4) runs only after the A7 freeze.

**Sample** (`analysis.human_audit_sample`, seed 20261020, drawn from the frozen `t2_stories.jsonl`). 60 valid stories, 10 of each version (explicit, banned, implicit, coded, control, decoy), never two versions of one plot. Each story is asked about 5 memes: its 3 design memes and 2 memes it does not plant (distractors, drawn by the same seed), in shuffled order. That is 300 cells: 30 explicit, 30 banned, 30 implicit, 30 coded, 30 control and 30 decoy design cells, and 120 distractor cells. The stories are split into **5 balanced blocks of 12**, each with 2 stories of every version, shuffled within the block; the page shows block 1, then block 2, and so on. Each story gets an opaque id, A01 to A60 in presentation order.

**What the page reads.** `build_lists.py texts audit` writes `data/human_audit_sample.json` (`analysis.audit_sample_file`): for each opaque id, its block, the story as numbered sentences, and the 5 memes' labels and definitions. It contains no story id, plot, version, writer or genre, and the script refuses to write it if any would show (`analysis.audit_leaks`). The page never opens `t2_stories.jsonl`; the analysis maps the opaque ids back by re-running the deterministic sample (`analysis._human_audit`, which also checks that the file matches). `test_analysis.py` checks, before F0, that no version name, plot id or story id appears in the sample file or in the HTML the page serves.

**The page** (`audit_page.py`, standard library only, served on `127.0.0.1`; nothing leaves the machine). It shows one story at a time, as plain numbered sentences, followed by the 5 memes as label and definition (no keywords, no examples), each with three buttons: **happens**, **does not happen**, **unsure**. Instructions at the top: "Judge only from the story. Does this element happen in it, however it is described?" A progress line shows stories and blocks done. Every click **appends** one row to `preregistration/data/human_audit.jsonl`: `{"kind": "t2_cell", "audit_id", "meme", "present": true | false | null, "seconds", "time"}` (`null` = unsure; `seconds` since the story was shown). An answer can be changed by clicking again; a new row is appended and the **last row of a cell counts**. `test_analysis.py` runs the page on the synthetic world and checks this format, last-answer-wins and the refusal of unknown ids.

**What counts.** A story is answered when all 5 of its cells have an answer (unsure is an answer). A block is complete when all 12 of its stories are answered. Only complete blocks are scored, so the versions stay balanced. The audit is **complete** with at least 4 complete blocks (48 stories).

**Pilot timing (P2, before F0).** The author times the page on at most 5 pilot stories (`build_lists.py texts audit-pilot`, about 15 minutes); the rest of the page is tested by script. If his median time per story is above 150 seconds (`audit_page.py --pilot-report`), the main audit shows only the first 4 blocks (48 stories; `analysis.AUDIT_BLOCKS_SHOWN = 4`), fixed in section 14 before F0.

**Gates.** Each gate is computed on the cells the per-protocol analysis uses (Q4 and Q5 exclusions and dropped memes left out), with "unsure" answers left out of the denominator and reported. A gate passes when the one-sided 90% Clopper-Pearson lower bound of the agreement with the design is at least its threshold and at most 10% of its cells are unsure:

| Gate | Cells (all 5 blocks) | Threshold | Errors allowed (60 cells; 48 cells) | Decides |
|---|---|---|---|---|
| presence, reworded | banned and implicit design cells (60) | 0.75 | 10; 7 | H1 to H3 (with the absence gate) |
| absence | control and decoy design cells (60) | 0.80 | 7; 5 | H1 to H3 and S-coded |
| presence, coded | coded design cells (30) | 0.75 | 4 of 30; 2 of 24 | S-coded and the overlap tests on coded versions (with the absence gate) |

`analysis.audit_max_errors` gives the allowed count for any number of answered cells. If the author's true agreement is 95% on the controls, the absence gate fails with probability about 0.01 at 60 cells and 0.03 at 48 (at 85%, about 0.7). Reported beside the gates: the same gates on every cell, agreement on explicit cells, absence on the distractor cells, agreement per version, the number of unsure answers, and the kappa between the author and the two Claude readers.

**Fallback.** If the audit is not complete, H1 to H3 and S-coded are reported with the note "no complete human audit", and the cross-family reader's agreement decides instead: presence at least 75% on the per-protocol banned and implicit cells (H1 to H3) and on the per-protocol coded cells (S-coded), absence at least 90% on every valid control and decoy cell (before Q5, which removes every control cell this reader names). If neither check is available, H1 to H3 and S-coded are inconclusive.

## A8. Example sentences (step P1, before F0)

For each of the 34 memes, four fresh sessions, one per set, see only the meme's label and definition (and, for the reworded and hard-negative sets, its ban list; for the hard-negative set, its decoy hint). None sees `dev_units.json`, any DEV sentence or any other set. Each writes 7 sentences (5 are kept after screening):

- **explicit:** "Write 7 one-sentence plot-summary sentences, each from a different invented story and setting, in which this element plainly happens and is described in the words people usually use for it. No names of real people, famous characters or franchises."
- **reworded:** the same, "...in which this element plainly happens, without using any of these words or words derived from them: {BAN LIST}."
- **far:** the same as explicit, "...set far from the usual settings: no crime-ridden city, no colonial California, no revolutionary France, no billionaire or inventor, no costumes, masks, capes or vigilantes."
- **hard_negative:** "Write 7 one-sentence plot-summary sentences, each using at least two of these words: {BAN LIST}, in a sense that does NOT make the element happen (ideas: {DECOY HINT})."

A blind reader session then judges every sentence: "Does this sentence show the element happening? yes or no." `build_lists.py texts examples` keeps, in the order written, the first 5 sentences of each set that pass: explicit with at least 1 ban-list term and judged present; reworded with none and judged present; far with none, no forbidden setting word (`analysis.FAR_FORBIDDEN`) or franchise name, judged present, and passing the TF-IDF screen against DEV Batman and cousin sentences (`analysis.far_screen_tfidf`); hard negatives with at least 2 terms and judged absent. A set with fewer than 3 survivors goes back to a fresh writer once. At F1 the far sets are screened again with e5-base embeddings (`analysis.far_screen`), dropping failures but keeping at least 3. The meme's definition is used by every method as one more positive sentence.

## A9. Salience ratings (Q6)

Defined in `planted_story_protocol.md`, check Q6.
