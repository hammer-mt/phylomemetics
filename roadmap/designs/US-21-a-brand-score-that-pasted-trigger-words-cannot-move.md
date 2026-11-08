# US-21: A brand score that pasted trigger words cannot move

**Area:** Uses · **Status:** design registered 8 November 2026, not run · **Priority:** 4.5 of 5

## Question

Can a neuron-based brand score keep ranking genuinely on-brand text in new wording above off-brand text after an optimiser has tried to raise the off-brand score by pasting and rewriting?

## Prediction

After both attacks, a score that counts a unit only when it fires in three or more separate sentences still ranks reworded genuine episodes above attacked off-brand texts: mean AUC (the chance a genuine text outscores an attacked one; 0.50 is a coin flip) of at least 0.75 over 8 series, and within 0.05 of the embedding centroid.

## What would refute it

After the attacks, the best neuron score chosen on development data has mean AUC below 0.65, or trails the attacked sentence-embedding centroid by more than 0.05, or three pasted sentences alone lift an off-brand text by more than one standard deviation of genuine scores.

## Data

Plot sections of English Wikipedia episode articles for 8 scripted series, each with about 180 to 800 episode articles by a Wikidata count on 8 Nov 2026 (three are animated sitcoms), capped at 300 per series, names masked, CC BY-SA 4.0. Off-brand: episodes of the other seven series, plus the 1,000 random film plots in preregistration/sources.

## Method

Choose units and pooling rule per series on 60% of episodes. Reword held-out genuine and off-brand texts alike under a banned-word list (third model family). Attack each scorer twice: paste in 1, 3 or 5 of its top sentences; then 20 scored rewrites per text, at most a fifth of sentences changed, plot kept (blind check). Null: random units.

## Cheap rival it must beat

TF-IDF centroid (weighted word counts), sentence-embedding centroid, an LLM asked to rate fit, a probe on the model's averaged internal state, sparse-autoencoder features, text length; all attacked the same way. The LLM reader is most dangerous: it should see through pasted clauses. If it wins, we still learn which pooling rule makes an inspectable score hard to game.

## Premise it tests

P1. A pass shows units can respond to ideas spread through a text and not to pasted wording. A fail says the usable signal is lexical, as the Batman tie with word overlap suggested, and neuron scores should not be handed to an optimising agent as graders.

## Biology analogue

Batesian mimicry: harmless species wearing warning colours.

## Cost

M. About 3,400 short texts through two 8B models; the rewriting attack (15 texts per series, 5 scorers) is about 12,000 rewrites by a local model.

## Payoff

A grader for brand or style fit that an optimising agent cannot satisfy by copying phrases, or a clear statement that no neuron score of that kind exists yet.

## Main risk

Masked, reworded episodes of sibling series may be impossible to tell apart, so every method sits near chance; or summaries are so formulaic that all sit near the ceiling. Either way the result turns on how good the rewording is.

## Reviewer note

Both arms now pass through the same rewriter; sibling series added as hard negatives; a scored rewriting attack with an edit cap is the main test, because a three-sentence rule beats one pasted sentence by construction. Weakness: episode summaries stand in for brand copy; plot sections not yet counted.

## Sources

- Wikipedia episode articles via Wikidata (count 8 Nov 2026)
- He et al. 2024 (Llama Scope)
- Kantamneni et al. 2025
- Bradley et al. 2023 (QDAIF)
