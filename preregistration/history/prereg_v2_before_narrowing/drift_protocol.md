# Vocabulary-drift protocol (T1b), v2

This protocol rewrites lineage texts so that their words change while their plot stays the same. It produces graded levels of drift, so that every method gets a recovery-against-drift curve, and it feeds H4 (neurons against TF-IDF and LSA at levels 50 to 100).

## Works

- The 20 DEV Batman screen works: their name-free texts from `data/corpus_neutral.jsonl` (level 0).
- The 7 fresh Batman adaptations in `works_batman_extended.json`, after name removal (`name_removal_protocol.md`).
- The 44 Shakespeare films with `drift_sample: true` in `works_shakespeare.json`: the H5 sample (renamed primary films documented by a lead sentence or Wikidata P144, not remakes of DEV works), at most 6 per play in sha256 order, after name removal.

The 12 Batman comics and the 40 plays are the sources and are never rewritten. The reference films are not rewritten either: an analyst works with a fixed natural reference corpus, and the presence bars stay those of natural text. The style-only control below measures what rewriting itself does.

## Levels

Four rewritten levels, 25, 50, 75 and 100, plus a style-only control. Level 0 is the unmodified name-free text.

## Step 1: which words to replace (script)

For each work and level, `analysis.drift_banned_types(work_id, sentences, level)` lists the content-word types to replace:

- the stems of the work's tokens that have at least 3 letters and are not English stop words or X-codes, **and every number** (so issue numbers such as #484 cannot carry the lineage);
- ordered by `sha256(work_id|type)`;
- the first `round(level% × number of types)`.

The sets are nested (25 ⊂ 50 ⊂ 75 ⊂ 100). At level 100 every content type and number of the original is banned.

## Step 2: rewrite (agent)

Each level is rewritten **independently from the level-0 text**, never from another level, by a fresh session that sees only this prompt:

```
Rewrite this plot summary, replacing only the words listed below.
Rules:
1. Keep the same number of sentences, in the same order; sentence k of your rewrite must tell what
   sentence k of the original tells: the same events, relationships and motives, nothing added, nothing
   left out. Keep each sentence about as long as the original.
2. Codes such as X1, X2 stand for names. Keep every code exactly as it is, in the same sentence, and do not
   add codes, names, titles or capitalised words.
3. Replace each listed word, in every form (plural, tense, -ing, -ed, and words derived from it such as
   killer from kill), with a synonym, a euphemism or a paraphrase. Write numbers in other ways or leave
   them out if the sentence still says the same. Change other words only where grammar then requires it;
   keep every other word.
   WORDS TO REPLACE: {comma-separated banned stems, each shown with one example of its original form}
4. Do not use any well-known nickname, epithet or catchphrase of a famous character or story.
5. Write plain English in the style of a plot summary. Do not explain, comment or add notes.
Return JSON only: {"id": "{WORK_ID}", "level": {LEVEL}, "sentences": [...]}
<numbered original sentences>
```

**Style-only control.** One more rewrite per work, by a fresh session, with the same prompt and `WORDS TO REPLACE: (none)`, so the rewriter re-renders the text keeping its wording wherever possible. It measures what an agent rewrite does to each method without any banned words (level `style`).

## Step 3: script checks

- **D1 alignment.** Same number of sentences. Every X-code of an original sentence appears in the matching rewritten sentence, and no code is added.
- **D1b length.** Each rewritten sentence has within 20% of the original's word count, or within 3 words.
- **D1c capitals and epithets.** No capitalised token that is not sentence-initial and not an X-code, and no phrase from the frozen epithet list: *caped crusader, dark knight, the bat, world's greatest detective, clown prince, boy wonder, star-crossed, the shrew, the moor, the dane, the bard, the scottish play, merchant of venice, prince of denmark, thane, the tempest* (case-insensitive).
- **D2 words.** No banned type appears, with the derivational rule (`analysis.types_banned`). `analysis.realised_replacement` records the share of the original's banned-type tokens that are gone and the share of all its content types that survive.

## Step 4: plot retention (blind agents)

A fresh checker session sees the original and rewritten sentences side by side, numbered, without the level or the list. For each pair it answers `same`, `partly` or `different` for the events, relationships and motives, and flags added information. The cross-family check reader answers the same question on the same pairs (`readers.py check --task xretain`); its agreement with the Claude checker is reported and does not gate.

- **D3 retention.** At least 90% of pairs `same`, none `different` (Claude checker).
- **D4 additions.** No added information flagged.

```
For each numbered pair, does the REWRITE tell the same events, relationships and motives as the
ORIGINAL? Answer same, partly or different, and add "+info" if the rewrite adds information the original
does not contain. Return JSON: [{"k": 1, "verdict": "same", "added": false}, ...]
```

## Repair and flags

A rewrite failing any of D1 to D4 goes back once to a fresh rewriting session with the failure list appended. If it still fails, it is **kept and flagged** in `drift_flags` (dropping it would make some levels easier than others). H4 is also reported without any work flagged at any level.

## Outputs

Rewrites go to `prereg/agent/drift/level{L}.jsonl` (L = 25, 50, 75, 100, style) as `{"id", "level", "sentences", "flags"}`. `build_lists.py texts drift` assembles `t1b_drift{L}`, `t1b_fresh_drift{L}` and `t1b_shk_drift{L}` (and `_style`).

## Manipulation checks and confounds

- **Removal.** H4 is inconclusive if, at level 100, the mean realised removal of banned-type tokens is below 90%, or if TF-IDF recovers the sources no worse at level 100 than at level 0 on the pooled targets.
- **Spacing.** Mean content-type survival must fall by at least 15 points between adjacent levels; otherwise the levels are reported as merged (the curve is drawn against realised survival in every case).
- **Source leak.** `analysis.introduced_types` lists the words each rewrite introduced. A TF-IDF lineage test on those words alone (pooled over both arms) at levels 50, 75 and 100 measures whether the rewriter brought in words of the documented source. If it recovers sources (p < 0.05), H4 is labelled confounded and its verdict uses the analysis without the leaking targets (those whose documented source ranks first on introduced words).
- **Style.** The neurons-minus-rival advantage on the style-only rewrites is reported beside the level-0 advantage.
