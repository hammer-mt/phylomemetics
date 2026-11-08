# AT-28: Eka-meme: does a table predict an unseen unit?

**Area:** The atlas · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

When motifs are laid out in a grid of who acts by what they do, do an empty cell's row and column predict the direction, and the neuron, that reads that cell's motif before any passage for it has been seen?

## Prediction

Directions predicted from a cell's row and column reach a mean AUC of 0.70 (0.5 is chance) on held-out cells and do at least as well as the same arithmetic on e5 sentence embeddings. The neuron nearest the predicted direction beats a matched random neuron in at least 24 of 36 cells.

## What would refute it

Mean AUC of predicted directions under 0.60, or 0.05 below the embedding arithmetic (bootstrap over rows and columns, 95% interval excludes zero), or the predicted neuron beats a random one in fewer than 22 of 36 cells. 24 of 36 has a sign-test p of 0.03; 36 cells put the mean AUC within about 0.03.

## Data

A grid of 36 motifs: six family roles (father, mother, stepmother, brother, sister, wife) by six acts (rescues, abandons, murders, sells, betrays, curses). Definitions follow Thompson's index in Hagedorn's trilogy data (CC BY-SA 4.0), where a crude keyword match finds 29 cells. Per cell, 96 new passages avoiding the motif's usual words, with twins; 3,456 in all, CC BY.

## Method

Model: Llama-3.1-8B-Instruct. A cell's direction: mean hidden state of motif passages minus twins. Hold out each cell in turn; predict its direction as row mean plus column mean minus grand mean. Score AUC for the held-out cell's test passages against those of its row and column neighbours. The predicted neuron is the one whose output weights lie nearest that direction.

## Cheap rival it must beat

The same row-plus-column arithmetic on e5 embeddings and on TF-IDF vectors; the neuron whose Transluce description best fits the cell's definition; a random direction; passage length; an LLM reader given the definition, which will win and sets the ceiling. Embedding arithmetic is the danger: analogies are what embeddings are known for. The neuron must also beat the description lookup.

## Premise it tests

P1. A table that predicts where an unseen meme's neuron sits would give the atlas real periodic structure and a way to find units without examples. If only dense directions compose, the structure belongs to the model's geometry, not its neurons; if embeddings do as well, it belongs to language.

## Biology analogue

Mendeleev's eka-silicon: properties predicted from the gap

## Cost

S. About 3,500 short passages by API (under 30 US dollars), one extraction pass of under an hour, arithmetic on a laptop; under two days.

## Payoff

A direct test of whether meme space has rows and columns: a property of a missing entry is predicted, then checked. If it works, the catalogue can be extended to motifs nobody has written passages for, including the cells the folktale index itself leaves empty.

## Main risk

Row and column words can leak: stepmother passages share vocabulary even when the usual words are avoided, so the predicted direction may be a role detector plus an act detector. Scoring against row and column neighbours limits this. Family roles and violent acts are also an easy grid; a pass may not carry to subtler motifs.

## Reviewer note

Promoted from a seed and made concrete by the reviewer, to take the Mendeleev slot from AT-26, which US26 already covers. Weakness: embedding arithmetic will probably tie the direction test, so the informative part is whether a single neuron sits where the table says; the grid is an easy one.

## Sources

- Li et al. 2024, arXiv:2410.19750
- Hagedorn, trilogy dataset, Zenodo 6575263
- Choi et al. 2024, Scaling automatic neuron description (Transluce)
- Gurnee et al. 2023, arXiv:2305.01610
