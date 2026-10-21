# Name-removal protocol (T1d, T1e, T4, T1c natural arm), v2

This protocol applies the DEV procedure that built `data/corpus_neutral.jsonl` to new texts. An agent rewrite is used **only to find the names**. A word-level diff against the original then replaces each changed span with a neutral code (X1, X2 and so on, consistent within a work), so any role word the rewriter chose ("the vigilante", "the prince") is discarded and cannot leak identity. The code is `analysis.neutralise`, copied from `make_neutral.py` (`analysis.neutralise_cjk`, a character-level diff, for Chinese).

## Texts

- The 40 Shakespeare plays (`works_shakespeare.json`, plays: commentary sentences dropped by the fixed rule, at most 80 sentences).
- The 177 eligible Shakespeare films (all processed; the 141 primary films enter the analyses).
- The 39 TEST Batman-lineage works (`works_batman_extended.json`, test), which include the 7 fresh adaptations.
- **The 32 DEV Batman works again, from their original texts** (`t4_dev_original`), so that all 71 works of the T4 lineage go through the same v2 procedure in one pass (`batman_dev_v2.jsonl`). The DEV comparisons (T1a, T1b level 0) keep the published DEV name-free texts.
- The zh.wikipedia plot sections of the T1c natural arm (`translation_protocol.md`, N2).

The original texts are the frozen Wikipedia plot sections; `build_lists.py texts wikipedia` writes them and checks them against their hashes.

## Step 1: rewrite (agent)

Each call goes to a fresh session that sees one work:

```
Rewrite this plot summary with every proper name removed. Replace each name of a person, character,
place, organisation, ship, brand, title of a work or named object with a short plain description of the
role it plays in the sentence (for example "the prince", "the city", "the newspaper"). Change NOTHING else:
keep every other word, the word order and the punctuation exactly as they are wherever you can, keep the
same number of sentences, and do not add, explain or drop anything. The same name should get the same
description throughout. Titles that are also ordinary words (a duke, a king) stay as they are; a title
fused to a name ("King Lear", "Lady Macbeth") is a name.
Return JSON only: {"id": "{WORK_ID}", "rewrite": [sentences]}
<numbered sentences>
```

## Step 2: codes (script)

`analysis.neutralise(original, rewrite)` replaces every span the rewrite changed with an X-code; the same original words get the same code within a work; possessives and punctuation are kept; spans the rewrite deleted are deleted. The number of spans replaced is recorded.

## Step 3: audit (agent and script)

- **N1 alignment (script).** Equal sentence counts.
- **N2 leftovers (script).** Every capitalised token that is not sentence-initial and not an X-code is listed for the auditor, as is every hit of `analysis.FRANCHISE`.
- **N3 audit (agent).** A fresh session reads the X-coded text with the N2 list and names any remaining proper name or identifying title, and any non-name content the diff swallowed:

```
This plot summary should contain no proper names; names were replaced by codes such as X1. List every
remaining proper name, title of a work or other identifying name, with its sentence number, and every
place where ordinary (non-name) words seem to be missing. Return JSON: {"names": [[k, "text"]],
"missing": [[k, "note"]]}; empty lists if none.
```

For any flagged sentence the rewrite is redone once with the flag shown, and Step 2 is rerun. Remaining flags are reported, and the swallowed-content rate is reported by group (plays, renamed films, faithful films, TEST Batman works, DEV Batman works v2).

## Outputs

Rewrites go to `prereg/agent/name_removal/{plays,films,batman_new,batman_dev_v2}.jsonl` as `{"id", "rewrite"}`. `build_lists.py texts names` writes `t1d_plays_namefree`, `t1d_films_namefree`, `t1e_fresh_namefree` and `t4_batman` (all 71 Batman works in the v2 pass, each with `origin` dev or test).

## Where the deterministic masker is used instead

T3 (2,544 pool works plus the DEV reference) and T2-N use `analysis.t3_text` on every text, the DEV reference included: relation statements ("a parody of X", "based on", "inspired by", "homage to", "pastiche") are removed first, then `analysis.mask_names` (a capitalisation-based masker), then any surviving franchise word (`analysis.FRANCHISE`) is masked as X0 and counted. An agent pass over 2,500 works would be expensive, and the masker treats every text identically. On DEV the masker hides 91% of the tokens the agent procedure replaced, at 97% precision; 129 of 13,337 agent-replaced tokens are capitalised words it misses, which the franchise scan catches when they are franchise names.
