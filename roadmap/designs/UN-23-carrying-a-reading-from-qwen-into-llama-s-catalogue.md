# UN-23: Carrying a reading from Qwen into Llama's catalogue

**Area:** Across models · **Status:** design registered 8 November 2026, not run · **Priority:** 4.6 of 5

## Question

Can a tale read by Qwen be matched against a library of tales read by Llama, using a map fitted only on unrelated film text?

## Prediction

Carried by a ridge map (a regularised linear fit) fitted on film sentences, Qwen's readings of name-masked tales find same-type tales in Llama's library with 80% or more of the above-chance precision Llama reaches when it reads the queries itself. Single twins keep under 50%.

## What would refute it

The ridge-carried reading keeps under 50% of Llama's own above-chance precision, or does no better than a map fitted on misaligned sentences (sign-flip test across 81 tale types; about 80% power for a 0.03 gap if type differences vary by 0.1). Then a catalogue cannot be shared by mapping.

## Data

Annotated Folktales in the `trilogy` repository (Hagedorn and Darányi, Zenodo 6575263; the repository states CC BY-SA 4.0): 1,518 English tales, each given one of 182 Aarne-Thompson-Uther tale types by the folklorist D. L. Ashliman. Without 20 tales marked non-commercial, 81 types have 8 or more tellings (1,066 tales): the test set. Types with 4 to 7 tellings are for development.

## Method

Each name-masked tale is read as in the study (top sentences per unit) and as a mid-depth residual-stream vector (the state passed between layers). Qwen's reading is carried into Llama's coordinates by single twins, a 10-unit lasso, full-layer ridge, a rotation, or anchor similarities with no fitting. Statistic: average precision of same-type retrieval. Null: maps fitted on misaligned sentences.

## Cheap rival it must beat

TF-IDF similarity (needs no map), a sentence-embedding model, an LLM asked whether two tales share a type (subsample), tale length, random units carried the same way, and Llama's pooled residual stream reading the queries itself. TF-IDF is most dangerous; if it wins on precision, retention still settles whether maps work. Same-type pairs with low word overlap are scored separately.

## Premise it tests

P2. Retention by ridge but not by twins supports P2 restated as shared directions and makes one catalogue practical. Low retention by every map means one catalogue per model, which is Mike's fallback of a single reference model.

## Biology analogue

Lifting gene annotations from one genome to another.

## Cost

S. Under an hour of extraction for both models (tales capped at 40 sentences); maps and retrieval in a day or two on the laptop; LLM reading on a subsample.

## Payoff

How much of a reading survives a change of model, for each kind of map, and whether a meme travels as one neuron, a few neurons or a direction. Readings of texts that cannot be redistributed could then be shared and used with another model.

## Main risk

Tale types share key nouns (hare, slipper), so TF-IDF may sit near the ceiling and leave the maps little room to differ. A film-fitted map may fail on nineteenth-century folktale prose because the text differs, not the models; a map fitted on development tales measures that. Both models have probably read these tales.

## Reviewer note

Rebuilt as retrieval. Picking a detector from 900,000 units on four tellings would overfit, as the study's Qwen revenge unit did; now no unit is chosen on tale labels. Counts confirmed from the file. Remaining: labels come from one folklorist, and tale types may be too lexical.

## Cross-area ranking

In the top twelve of 2 of 3 judges (what decides the premises fastest: rank 9; speed and cost: rank 6). Tests P2 in the only form the literature supports, a map fitted on paired text instead of one-to-one twins, in a day or two, and says whether one catalogue can be shared or each model needs its own.

## Sources

- Hagedorn and Darányi 2022 (Zenodo 6575263)
- Chen et al. 2025 (model stitching)
- Moschella et al. 2023
- Oozeer et al. 2025
