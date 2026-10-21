# Planted-story protocol (T2), v2

This protocol makes the planted-meme benchmark: 136 new plot summaries, each written in 6 versions, with a known set of memes planted in each. It is fixed before any story is written, after the pilot (step P2, `PREREGISTRATION.md` section 3). The writers, readers, raters and checkers are Claude agents, except one cross-family check reader (Gemma 2 9B, run locally through `readers.py check`). Each Claude call is a fresh session that sees only what this protocol gives it. No method (neurons, keywords, embeddings, topics, NLI or LLM readers) touches a story until every check below is finished and the story files and their hashes are committed (freeze F2-T2).

The benchmark measures memes **as Claude writes them**. That is a limitation, and it is stated wherever T2 results are reported.

## Inputs

- `planted_design.json` gives each plot (P001 to P136) its genre, its writer (W1 to W4) and the 3 memes it plants. Each of the 34 memes is planted in exactly 12 plots, never twice in one genre, in at most 4 plots by one writer. No two memes share more than 2 plots, and memes listed under `do_not_combine_with` never share a plot.
- `memes_for_tracing.json` gives each meme's label, definition, `keywords` (the ban list) and `decoy_hint`.
- The full list of 34 labels and definitions is given to writers, so they know what to avoid, and to readers.

## The six versions

Each plot has one fixed frame: the sentences that are the same in every version. The frame has three slots, one per planted meme. Each slot is 1 to 3 sentences long, has the same number of sentences in every version, and every fill of a slot has within 15% of the explicit fill's word count (or within 4 words for short fills). The versions differ only in what fills the slots:

| Version | What each slot contains | Listed keywords |
|---|---|---|
| explicit | the meme, plainly present, in the words people usually use for it | at least 1 of the meme's keywords in the slot |
| banned | the meme, plainly present; a reader could name it in a phrase | none, anywhere in the story |
| implicit | the meme, shown only through concrete events and actions; never named or summarised | none, anywhere in the story |
| coded | the meme, with its key concept named by a consistent invented or repurposed term that the fill introduces and that is understood only from context (the way a group renames something to dodge moderation) | none, anywhere in the story |
| control | **an event of equal stakes and dramatic weight**, of the same length, that fits the plot and is none of the 34 elements | none, anywhere in the story |
| decoy | the meme's keywords used in a sense that does NOT make the meme happen; none of the 34 elements | at least 2 of the meme's keywords in the slot |

"Keywords" means the 3 planted memes' listed terms with their derivations, as matched by `analysis.ban_hits`: a multi-word term matches as a phrase; a one-word term matches any word whose stem equals its stem, shares a prefix of at least 5 letters with a stem of 5 or more letters (avenge/avenger, funeral/funerary, orphan/orphanage), or is in a frozen family of a short stem (kill/killer, bribe/bribery). The frame never contains them in any version.

## Writer instructions (one call per plot)

The four writers are separate agent sessions, each with a fixed style brief:
- W1: plain and literal, short sentences.
- W2: rich and descriptive, longer sentences.
- W3: brisk, event after event, few adjectives.
- W4: formal, in the register of an encyclopedia summary.

Every writer call receives the prompt below, with the fields filled in from the design and the meme files.

```
You are writing test material for a research study. Write ONE plot summary in the style of a Wikipedia
"Plot" section: present tense, third person, {STYLE_BRIEF}. Setting and genre: {GENRE}. The story must be
original. Use invented names only: no real people, no famous fictional characters, no names or places from
any existing franchise, and no superheroes or masked crime-fighters in a city.

The story must contain exactly three story elements, each in its own slot:
{for each of the 3 memes: LABEL: DEFINITION}

Write it as a fixed FRAME plus six ways of filling each slot.
1. FRAME: 18 to 30 sentences in total, counting the slot sentences. Mark each slot with a placeholder line
   <<SLOT:{meme_id}>> standing for 1 to 3 sentences. Spread the slots through the story. The frame must not
   contain any element from this list of 34: {ALL 34 LABELS AND DEFINITIONS}. It must not contain any of
   these words, their inflections or words derived from them: {KEYWORDS OF THE 3 MEMES}. The rest of the
   story must make sense whichever way the slots are filled. Nothing in the frame may depend on an element
   having happened, and the frame must not hint at an element (no setting detail that only matters for it).
2. For each slot, write six fills. All six have the same number of sentences, and each has within 15% of
   the explicit fill's number of words.
   - explicit: the element plainly happens, described in its usual words. Use at least one of: {KEYWORDS}.
   - banned: the element plainly happens, and a reader could name it in a phrase, but use none of these
     words, their inflections or derived words anywhere: {KEYWORDS OF ALL 3 MEMES}. Use other words.
   - implicit: the element happens only through concrete events and actions. Never name it, summarise it
     or use an abstract noun for it. None of the listed words.
   - coded: the element happens, and the fill gives its key idea a consistent invented or repurposed name
     (a made-up word, or an ordinary word used in a private sense) that a reader can only understand from
     what happens. None of the listed words.
   - control: a DIFFERENT event of EQUAL stakes and dramatic weight (as surprising, as consequential, as
     emotional) that fits the plot and is none of the 34 elements. None of the listed words.
   - decoy: use at least two of these words: {KEYWORDS OF THIS MEME}, in a sense that does NOT make the
     element happen (for ideas: {DECOY_HINT}). Contain none of the 34 elements.
   No fill may contain any element from the list of 34 other than its own (the control and decoy fills
   contain none).
Return JSON only: {"plot": "{PLOT_ID}", "frame": [sentences and placeholder lines],
"fills": {"{meme_id}": {"explicit": [...], "banned": [...], "implicit": [...], "coded": [...],
"control": [...], "decoy": [...]}}}
```

A script builds the six versions by putting each version's fills into the frame. It records each slot's sentence indices (`slots`) and writes one line per version to `prereg/agent/stories/stories.jsonl`, as `{"plot", "version", "sentences", "slots", "writer"}`.

## Quality checks

The checks run in order. A version that fails a check is never repaired by hand.

- **Q1 frame (script, `analysis.story_checks`).** The story has 18 to 30 sentences, one slot per planted meme of 1 to 3 sentences, slots disjoint. All six versions have the same length, slot positions and frame sentences.
- **Q1b length (script).** Every fill of a slot has within 15% (or 4 words) of the explicit fill's word count.
- **Q2 words (script, `analysis.ban_hits`).** The frame contains none of the 3 memes' keywords or their derivations. The banned, implicit, coded and control versions contain none anywhere. Each explicit slot has at least 1 of its meme's keywords; each decoy slot at least 2. The share of each banned, implicit and coded fill's content words that also occur in its meme's definition is recorded (definition residue, reported).
- **Q3 names (script).** No sentence contains a franchise name from `analysis.FRANCHISE`.
- **Q4 recognisability (blind readers).** Each version is read by two Claude reader sessions drawn at random from a pool of fresh sessions, and by the cross-family check reader. A reader never sees two versions of the same plot, the design, the keywords or the examples. For each planted meme:
  - explicit, banned and coded versions pass when both Claude readers name it;
  - implicit versions pass when at least one Claude reader names it.
- **Q5 contamination (symmetric).** A version counts as a negative for a meme (any unplanted meme, or a planted meme in its control or decoy version) only when **no** reader, Claude or cross-family, names that meme. If any reader names it, the version is excluded for that meme.
- **Q6 salience (blind raters).** For every plot, a fresh rater session sees the slot fills of the control version and of the banned and implicit versions, each in its frame, one version per session, and rates each slot's event on stakes and dramatic weight from 1 (trivial) to 5 (life-changing). The check passes when, per plot, the planted fills' mean rating is within 1.0 point of the control fills' mean, and over all plots the mean difference (planted minus control) lies within ±0.3. A control fill whose plot fails is rewritten once by the same writer with the ratings shown.

The reader prompt is:

```
Read this plot summary. Below is a list of 34 story elements. List every element that happens in the
story, however it is described, and only those. Answer with the element ids, one per line, or "none".
<story>{STORY}</story>
{34 lines: id: LABEL. DEFINITION}
```

The rater prompt is:

```
Here is a plot summary. The sentences marked [1], [2] and [3] each describe one event. For each marked
event, rate how high its stakes and how great its dramatic weight are, from 1 (trivial, changes nothing)
to 5 (life-changing or life-threatening). Rate the event as written; do not guess what it stands for.
Return JSON: {"1": n, "2": n, "3": n}
<story>{STORY WITH MARKED SLOTS}</story>
```

Reader answers go to `prereg/agent/stories/blind_reads.jsonl` as `{"plot", "version", "reader", "family", "memes_found": [ids]}` (family `claude` or the check reader's tag); ratings to `prereg/agent/stories/salience.jsonl` as `{"plot", "version", "slot", "rating"}`.

**Repair.** A plot that fails Q1 to Q3, has any slot failing Q4, or fails Q6, is rewritten once: the same writer gets a new call with the failure list appended, and the new plot is checked again from Q1. After that:

- A version still failing Q1 to Q3 is marked `valid: false` and dropped from every T2 analysis.
- A slot still failing Q4, or a version excluded under Q5, is listed in that version's `exclude_memes`. The version is left out of both the positive and the negative set for those memes, and kept for every other meme.
- A meme is dropped from the per-protocol T2 analysis, for every method, if more than 3 of its 12 plots fail Q4 at the explicit or banned level after repair (`dropped_memes`). Dropped memes are reported with the reason.
- The **intention-to-treat** analysis keeps every valid story (Q1 to Q3) as designed and ignores Q4 to Q6. It is reported beside every primary T2 result.

**Assembly.** `build_lists.py texts stories` assembles `prereg/data/t2_stories.jsonl`, with `valid`, `checks`, `exclude_memes`, `dropped_memes`, the readers' answers and the salience ratings. Exclusion counts by version, meme and writer are reported in every case. The Cohen's kappa between the two Claude readers is reported per version, and the agreement between Claude and the cross-family reader is reported per version.

## Long texts (T2-long)

After assembly, `build_lists.py texts long` builds 48 long texts, each the concatenation of 12 valid stories from 12 different plots (2 banned or implicit versions and 10 control versions, chosen by seed). The memes present are those of the two planted stories. No agent work is involved.

## Chinese versions (secondary)

The banned and control versions of every valid plot are translated into Chinese under `translation_protocol.md`. Their `valid` and `exclude_memes` fields carry over.

## Human audit

A stratified random sample of story-by-meme cells is labelled blind by the author (`annotation_protocol.md` A7). Its pass rule is a condition for the H1 to H3 verdicts.

## What the readers' answers are not used for

The blind reads and ratings decide which stories enter the benchmark. They are never scored as a method. The Claude upper bound in T2 (`annotation_protocol.md` A4) uses separate sessions and is labelled as an upper bound, not a check.
