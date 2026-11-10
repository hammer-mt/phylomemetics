# IN-35: Coverage: can named memes alone carry heredity?

**Area:** The instrument · **Status:** design registered 10 November 2026, not run · **Priority:** 2.8 of 5

## Question

On Batman plots with names replaced by codes, can the study's heredity result (adaptations sharing units with their documented source comic) be reproduced from the labelled units alone, where a unit is one sign of one MLP neuron, and do those units find sources better than equally many arbitrary units or words?

## Prediction

In Qwen3-8B the 125 labelled units alone pass the paper's own heredity test (exact relabelling p under 0.05; all 150,784 units leave 16% of rival comics outranking the true source, chance 47%), and score better than 95% of 1,000 random unit sets matched on size, layer and the number of Batman works each unit is present in.

## What would refute it

The labelled set fails its own test, or more than 5% of matched random sets beat it. With nine adaptations a set passes only near 28%, three fifths of the way from 47% to 16% (null spread 11 points); smaller shares cannot be told from none. 1,000 draws fix a percentile within 1.4 points. It describes these nine adaptations only.

## Data

Committed in study/results: experiments.json and the robustness outputs (full-set score, nulls, word comparators), each model's memes.json.gz (families), labels.json and labels_wf.json (125 labelled Qwen families with 1,727 member units; the committed file lists 1,263 of them, at most 50 a family, so the other 464 must be rebuilt in the re-run). Name-free corpus: 114 Wikipedia plots, 3,700 sentences, names replaced by codes, CC BY-SA 4.0. Per-unit name-free readings are not committed: re-run the frozen extraction once (Apple silicon, MLX).

## Method

Sets, hashed before the re-run: labelled units, their family members, catalogue families, random sets of 31 to 50,000. The re-run must reproduce the nine committed scores, or stop. The frozen statistic counts only units present in 2 to 16 of 32 Batman works and treats a tie as the source not beaten. For small sets, where ties are common, also report the score with a tie counted as half. Report each set's gap below its own shuffled-source chance, also within era.

## Cheap rival it must beat

1,000 more random unit sets matched also on how well one word predicts the unit; random and top TF-IDF (weighted word count) sets of 31 to 1,309 words; the readings' first principal axes; reference lines for a sentence embedding, plot length and an 8B model naming the source. Most dangerous: equally many words, since 1,309 words already match 150,784 units.

## Premise it tests

P1. Pass: named units can stand in for the raw reading; if equally many words do as well, it shows listing, not neurons. Fail: descent belongs to any large set of readings, as to word counts; the heredity claim is about unit profiles, not named memes. Ties in over a quarter of comparisons: unscorable, too small for lineage.

## Biology analogue

Missing heritability: named variants against the whole genome

## Cost

S. One re-run of the frozen name-free extraction per model (the paper gives roughly 30 minutes for all its runs; needs Apple silicon and MLX), then about a night of CPU for 12,000 set scores with exact nulls. Llama 3.1 8B repeats the size curve only: it has no labelled units. Later, if scorable: units admitted by IN-21 or AT-21.

## Payoff

A coverage line to print beside the heredity claim: the labelled set's retained gap and its rank among arbitrary sets; how many arbitrary units, and how many words, a typical set needs to pass; the adaptations each set misses; a split by label kind (57 to 59 of 125 labels are characters or places, depending on which label file is used for the 26 units in both). First worked example of the roadmap's coverage rule.

## Main risk

Probably a fail: 73 of 125 labelled units sit in over 16 Batman works on named text, outside the band the score counts. Qwen failed the study's sequel check (4 of 7; bar 5), so the effect being divided is shaky. Retained gaps do not add: disjoint sets can each keep most. One franchise, where words find the source too.

## Reviewer note

Still an audit of one result on nine adaptations in one franchise, where word counts match neurons; a fail is expected and the admitted tier will likely be unscorable. Changed: the 50% bar became the paper's own test plus random-set rank; word sets joined; a planned second half on human-coded statements was dropped.

## Cross-area ranking

Registered on 10 November, after the three cross-area judges had ranked the earlier designs, so it was not ranked by them. Its reviewer's answer to whether it should run first: no: because it audits one result on nine adaptations that word counts already match, and its decisive tier needs IN-21 first; its labelled tier can ride along when step 1's IN-29 re-reads the study corpus.

## Overlaps with

LI-01: random units beat meme families in trees on named text; this repeats that on name-free text with the pre-registered score. LI-02, LI-03: same score and corpus. IN-21, AT-21: supply the later tier. IN-30 measures increment, not coverage. AT-27 and CE-21 already measure what frozen units miss in stories coded by people, so a planned second half of this design, on the human-coded statements of Berl et al. 2021, was dropped.

## Note

Prompted by the field survey (arXiv:2611.08438). Data check on 10 November 2026: Repository: 125 labelled families with 1,727 member units; on named text 73 are in over 16 of 32 Batman works and 52 in the band; 57 to 59 are character or place labels; null spread 0.114; 1,309 words score 17%; no activations committed. Survey HTML: coverage defined by intervention. Crossref: Manolio DOI. OSF 72v3f (538 statements, CC BY 4.0) checked, then dropped.

## Sources

- Barez, Wu et al. 2026, arXiv:2611.08438 (section 7.5 completeness, defined there by intervention; section 13.7 on stating coverage)
- Batman study outputs, study/results (paper v1)
- Manolio et al. 2009, doi:10.1038/nature08494
