# LI-25: Shakespeare as a second lineage with original-text parents

**Area:** Lineage · **Status:** design registered 8 November 2026, not run · **Priority:** 4.0 of 5

## Question

Does source finding repeat in a second well-documented lineage when the parent is the 400-year-old play text and the child a modern film summary, and do neuron memes (story units read from a language model's neurons) beat words and embeddings there, having tied with words on Batman's Wikipedia summaries?

## Prediction

Across 80 documented films, at most 8 per play, neuron memes rank the true play first among 37 at least 15 percentage points more often than the best of word overlap and sentence embeddings when the parent is the original play text. With the play's Wikipedia synopsis as parent, neurons and words are expected to tie.

## What would refute it

With original-text parents, the neuron-minus-best-rival gap has a 95% interval, from resampling plays, that includes zero. 80 films from 20 plays, 8 of them with one film, give about 80% power only for a gap near 18 to 20 points; a smaller true gap will read as undetermined, and that limit is accepted in advance.

## Data

Committed list `preregistration/works_shakespeare.json`: 40 plays, 177 eligible films, 107 primary films documented by a lead sentence or Wikidata link across 20 plays; 80 after capping each play at 8. Parents: the public-domain Moby text (37 plays, shakespeare.mit.edu), Wikipedia synopses (CC BY-SA), and the Lambs' 1807 prose retellings (20 plays, covering 72 of the 80 films; digital copy to be located).

## Method

With character names masked, each film's summary is scored against every play using the frozen Batman heredity score. Three parent forms are compared (original verse in equal-length samples, 1807 prose retelling, Wikipedia synopsis) and two film tiers (same-title, renamed). Null: shuffle play labels within comedies, histories and tragedies. A 'name the play' question to the model flags films it recognises.

## Cheap rival it must beat

Rarity-weighted word overlap (TF-IDF, BM25), a general sentence-embedding model and the story-trained story-emb, a local LLM reading film and play, text length, random non-meme units, the model's averaged internal state and sparse-autoencoder features. Most dangerous: the LLM reader, which knows Shakespeare by heart; it is reported apart. The claim is decided against the best of BM25 and the embeddings.

## Premise it tests

P1. It asks whether the Batman tie between neurons and words came from shared Wikipedia wording. A neuron lead with original-text or 1807 parents but not with synopses would support units as carriers of content across medium and four centuries of language. A second tie means neurons add nothing to words for descent.

## Biology analogue

A second model organism, as the mouse followed the fruit fly.

## Cost

M: about a week on one laptop. The 37 play texts extract in under an hour per model; masking names in plays, retellings and film summaries is the main labour.

## Payoff

Heredity tested in a second lineage with 37 candidate parents in their primary text, showing whether the Batman result was about stories or about Wikipedia summaries. It supersedes the deferred Shakespeare protocol.

## Main risk

Only 80 usable films from 20 plays, so only a large gap can be confirmed; small models may read verse so differently from prose summaries that every method sits near chance; and every model knows these plays by heart.

## Reviewer note

Counts confirmed in the committed list. Changed: the claim is decided against the best of words and embeddings, not BM25 alone; the Lambs' 1807 prose retellings added as a third parent form; an untestable 5-point tie claim dropped; power corrected. Remains: 80 films, memorised plays.

## Sources

- preregistration/works_shakespeare.json
- MIT Complete Works of Shakespeare (Moby text), shakespeare.mit.edu
- Lamb and Lamb 1807, Tales from Shakespeare
- Hatzel and Biemann 2024 (Story Embeddings)
