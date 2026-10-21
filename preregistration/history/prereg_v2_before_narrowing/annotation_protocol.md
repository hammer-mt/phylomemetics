# Annotation protocol, v2

Agents and the author provide these judgements:

- A1: the blind recognisability reads that admit planted stories into the benchmark;
- A2: the exhaustive ground truth for natural plot summaries (T2-N);
- A3: the blind label test of how nameable each method's units are (T6);
- A4 and A5: two labelled Claude upper bounds (a judge and a ban-list-aligned keyword query);
- A6: the typing of documented Batman relations (before F0);
- A7: the author's human audit;
- A8: the example sentences for every meme (before F0);
- A9: the salience ratings of planted-story slots (Q6).

Ground rules for every agent step:

- Every call is a fresh session. It sees only the material this protocol gives it, never a method's output, a design file or another annotator's answer, except where a step says otherwise (adjudication).
- Every output file records the agent model and the date.
- Items are shuffled with `numpy.random.default_rng(20261020)` before they are batched.
- Agreement is reported for every step with two or more annotators, whatever it turns out to be.
- Two Claude sessions agreeing measures one model's consistency, not independent agreement. Every gating judgement therefore also gets a cross-family reading (Gemma 2 9B, `readers.py check`) or a human audit (A7), and that agreement is reported.

## A1. Blind recognisability reads (T2 planted stories)

Defined in `planted_story_protocol.md`, checks Q4 and Q5: two Claude readers and one cross-family reader per version. Cohen's kappa between the two Claude readers, and Claude-versus-cross-family agreement, are reported per version type (explicit, banned, implicit, coded, control, decoy) over all story-by-meme cells.

## A2. Exhaustive annotation of natural plot summaries (T2-N)

**Works.** The 237 works of `culture_pool.json` with `t2n: true` (set `t2n_works`, texts through `analysis.t3_text`): the 39 Batman-lineage works, the 57 relatives, the 21 book-family works, and 40 pool works from each of the random, superhero and pulp-crime strata drawn in sha256 order (so that natural positives and negatives also occur outside Batman). Each work is annotated in full, so the ground truth does not depend on any method's ranking.

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

**Inclusion.** A meme enters the T2-N mean only with at least 10 positive and 10 negative works and annotator kappa of at least 0.4. Kappa is reported at work-by-meme and sentence-by-meme level, overall and per meme. Outputs: `prereg/agent/annotation/t2n_raw.jsonl` and `t2n_adjudicated.jsonl`; `build_lists.py texts t2n` adds them to `t2n_works`. T2-N uses every method's **far** detectors.

## A3. Blind label test (T6 interpretability)

**Units.** Two arms, each with one selection rule for every kind, scored on the T3 pool (masked texts):

| Kind | Random arm | Best-of arm |
|---|---|---|
| `neurons:qwen3-8b`, `neurons:llama3.1-8b` | the 400 random units per model in `dev_units.json` (`label_test_random`), in order | the 200 DEV meme families ranked by Batman enrichment (`label_test_best_of`), in order |
| `lda` (K = 100, fitted on the pool **works**) | topics in seeded random order | topics by UMass coherence |
| `topics:qwen3emb-4b` (HDBSCAN clusters of the pool's sentence embeddings) | clusters in seeded random order | clusters by mean membership probability |

In both arms a unit qualifies only if it is present in between 1% and 20% of pool works (neurons and LDA topics: beating all but one masked reference film; clusters: at least one sentence of the work in the cluster), and the first 40 that qualify are used.

**Top sentences.** For each unit, its 16 highest-scoring pool sentences, at most one per work, duplicates removed, ranked by the unit's own score among its members: a neuron's activation, an LDA topic's sentence-level probability under the work-level model, a cluster's membership probability among its members. Ranks 1 to 8 go to the labeller; 9 to 16 are held out.

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

A judgement is correct when it picks the unit's own sentence; a unit labelled `unclear` scores all its judgements as incorrect. As a floor, a script answers every item by word overlap (TF-IDF cosine of label plus description with each sentence). `analysis.run_t6` reports, per kind and arm: accuracy with an exact binomial 95% interval, the share of `unclear` labels, the kappa between the two judges, an exact McNemar test against the floor on the same items, Fisher tests of each neuron kind against the other kinds, and labeller calls per usable label. Answers go to `prereg/results/label_test_answers.json` as `{"kind", "arm", "unit", "item", "judge", "correct"}`.

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

Answers go to `features/t2_stories/llm_claude_judge.jsonl` (`p_yes` 1 or 0), scored like any method on the subset read, and reported as `judge:claude-upper-bound`. Claude wrote and admitted the stories, so this is an upper bound. It is **not** a manipulation check (that role is A7's and the cross-family reader's).

## A5. Ban-list-aligned keywords (T2)

One fresh session per meme receives `readers.KEYWORDS` (the plain analyst prompt) with the meme's label, definition and main examples. Parsed with `readers.parse_list` and saved as `features/t2_examples/keywords_ban-list-aligned.json`. Claude also wrote the ban lists, so this query is the one most likely to coincide with them; it is reported as `keywords:ban-list-aligned`, never as an upper bound.

## A6. Relation typing (T3; step P1, before F0)

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

A third session adjudicates disagreements. Kappa is reported. Output: `prereg/agent/relations/raw.jsonl` and `relation_types_final.jsonl`; `build_lists.py relations` merges them. S15's primary set is the plot_role and parody relations; iconography_name, meta and transitive relations are reported separately with a predicted null.

## A7. Human audit (the author)

Before freeze F3, the author labels, blind to version and to every method:

- 150 planted story-by-meme cells, stratified: 30 explicit, 30 banned, 30 implicit, 20 coded, 20 control, 20 decoy (each: "does this element happen in this story?");
- 10 T2-N works (presence of every meme marked by either annotator);
- 20 drift pairs (levels 50 to 100) and 10 translation pairs (same, partly or different).

The sample is drawn by seed 20261020 and presented by a script that hides the version. **Pass rule (H1 to H3):** presence agreement with the design of at least 85% on banned, implicit and coded cells, and absence agreement of at least 95% on control and decoy cells. If the author cannot do the audit, H1 to H3 are reported with the note "no human audit", and the pass rule is applied to the cross-family reader instead, with thresholds of 75% and 90%. Results go to `prereg/data/human_audit.jsonl` as `{"kind", "story", "meme", "present"}` (and the other kinds).

## A8. Example sentences (step P1, before F0)

For each of the 34 memes, four fresh sessions, one per set, see only the meme's label and definition (and, for the reworded and hard-negative sets, its ban list; for the hard-negative set, its decoy hint). None sees `dev_units.json`, any DEV sentence or any other set. Each writes 7 sentences (5 are kept after screening):

- **explicit:** "Write 7 one-sentence plot-summary sentences, each from a different invented story and setting, in which this element plainly happens and is described in the words people usually use for it. No names of real people, famous characters or franchises."
- **reworded:** the same, "...in which this element plainly happens, without using any of these words or words derived from them: {BAN LIST}."
- **far:** the same as explicit, "...set far from the usual settings: no crime-ridden city, no colonial California, no revolutionary France, no billionaire or inventor, no costumes, masks, capes or vigilantes."
- **hard_negative:** "Write 7 one-sentence plot-summary sentences, each using at least two of these words: {BAN LIST}, in a sense that does NOT make the element happen (ideas: {DECOY HINT})."

A blind reader session then judges every sentence: "Does this sentence show the element happening? yes or no." `build_lists.py texts examples` keeps, in the order written, the first 5 sentences of each set that pass: explicit with at least 1 ban-list term and judged present; reworded with none and judged present; far with none, no forbidden setting word (`analysis.FAR_FORBIDDEN`) or franchise name, judged present, and passing the TF-IDF screen against DEV Batman and cousin sentences (`analysis.far_screen_tfidf`); hard negatives with at least 2 terms and judged absent. A set with fewer than 3 survivors goes back to a fresh writer once. At F1 the far sets are screened again with e5-base embeddings (`analysis.far_screen`), dropping failures but keeping at least 3. The meme's definition is used by every method as one more positive sentence.

## A9. Salience ratings (Q6)

Defined in `planted_story_protocol.md`, check Q6.
