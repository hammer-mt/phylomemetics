# LI-21: Retelling traditions with a known tree and dosed rewording

**Area:** Lineage · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

When a story is passed down a tree drawn in advance, with rewording fixed at one of four levels from verbatim copy to retelling through another language, do neuron memes (story units read from which neurons fire) recover each text's true parent better than word and embedding methods?

## Prediction

On the six sealed tales, at the two heaviest rewording levels, neuron memes name the true parent first at least 10 percentage points more often than the best cheap rival, and that lead is larger than at the lighter levels. On verbatim copies every method scores above 95% and the difference is within 3 points.

## What would refute it

On the 276 sealed parent-child steps at the two heaviest levels, the neuron-minus-best-rival gap has a 95% interval, from resampling tales and then steps within them, that includes zero, or the gap does not widen from copy to heaviest level. That sample gives about 80% power for a 10 to 12-point gap at 30% disagreement; confirm by simulation.

## Data

Built, then released under CC BY-SA 4.0 like its seed corpus: 8 little-known tales from Folktexts (Zenodo 6575263, pre-1930 editions), each grown at four rewording levels (copy, paraphrase, retold from notes, retold through another language) into a 24-text tree: 768 texts. Two tales for development, six sealed. Two open writer models unrelated to the readers, drawn at random per step.

## Method

Each step plants plot changes, never swapped names or objects; a second reader confirms they survive. Each method ranks earlier texts as candidate parents, settings frozen on development tales. Statistic: share of true parents ranked first; also a triplet score (which two of three texts are closer). Null: relabel within tradition and generation. A 24-text human-retold tradition checks writer habits.

## Cheap rival it must beat

Word-overlap scores (TF-IDF, BM25), compression distance, two sentence-embedding models (vectors that summarise a passage), a local LLM ranking candidate parents, text length, random non-meme units, the model's averaged internal state, and learned sparse-autoencoder features (Llama Scope). Most dangerous: sentence embeddings, trained on paraphrase. If they win, the benchmark still shows where each method breaks.

## Premise it tests

both. P1: a neuron advantage that grows as wording is destroyed would be the first evidence that neurons carry descent that words miss; a tie or loss at heavy rewording leaves P1 without a use in lineage work. P2: it shows whether Qwen and Llama make the same parent errors beyond what a word-based ranking predicts.

## Biology analogue

Lab-evolved lineages with a recorded tree; simulated sequence evolution.

## Cost

M: about a week on one laptop. About 740 local retellings with survival checks over two or three nights, extraction in minutes, a few hours of LLM parent ranking, and about 6 volunteer hours for the human-retold tradition.

## Payoff

The first public known-tree benchmark where wording does not carry descent, and a curve showing the rewording level at which each method stops finding parents.

## Main risk

A child differs from its parent by only a few plot changes, so at heavy rewording every method may sit near the floor; or writer-model habits are read as descent; or sentence embeddings win at every level and neurons add nothing.

## Reviewer note

Changed: eight tales with smaller trees so tales can be resampled (was two sealed tales); planted changes are plot events, not nouns that survive paraphrase; random writer per step; licence CC BY-SA. Remains: model-written text; six sealed tales is few; prior work not searched (search budget spent).

## Amendment, 10 November 2026

From the field survey (arXiv:2611.08438): here descent passes through the writer's context window. The survey's attribution methods concern influence absorbed in training; it gives no accuracy figure for them, and says exact-span matching struggles with composed text. Add the longest shared word span as a named rival, and state that a pass shows copying from a visible source only. Influence that arrives through training needs a test with randomly assigned training sets. From the review of LI-27 (10 November): add a rival that needs forward passes only, the drop in a model's surprise at the story when the candidate source is placed in front of it. The reviewer expects it to be among the strongest rivals.

## Sources

- Roos and Heikkilä 2009
- Spencer et al. 2004
- Acerbi and Stubbersfield 2023
- Hagedorn and Darányi 2022 (Folktexts, Zenodo 6575263)
- Barez, Wu et al. 2026, Understanding AI from the Inside (arXiv:2611.08438)
- Guo et al. 2026, arXiv:2608.03859
