# AT-25: Linnaean key: named units against the folktale index

**Area:** The atlas · **Status:** design registered 8 November 2026, not run · **Priority:** 3.5 of 5

## Question

Can a short identification key made of named meme units place folktales of unseen tale types into the right division of the Aarne-Thompson-Uther index better than a key of the same length made of words?

## Prediction

A key of 15 units puts tales of held-out types in the right division 35% of the time or better (the largest division holds 11% of tales), at least 8 points above a 15-word key and within 15 points of nearest-neighbour search on e5 sentence embeddings.

## What would refute it

The unit key is under 3 points above the word key (bootstrap over tale types, 95% interval includes zero), or falls inside the range of keys trained on permuted division labels. Five-fold testing over 159 types (1,359 tales) gives a standard error near 3 points on the paired difference.

## Data

Hagedorn's trilogy data: 1,518 English folktale texts of 182 tale types; median 543 words (Zenodo 6575263, CC BY-SA 4.0; counted 8 November 2026). Divisions are set from the type number, because the file's division column is blank for 20 types and its chapter column is wrong for 338 tales. 20 divisions hold three or more types: 1,359 tales, 159 types.

## Method

Five folds by tale type; the other types build each key. A key is 15 yes/no characters ('is unit X present?') chosen greedily; a tale joins the division whose build tales share most answers. Unit names are Transluce's public descriptions, checked blind on tales. Null: division labels permuted across types within length bands. Repeated with character and animal nouns masked.

## Cheap rival it must beat

A 15-word key; a key of 15 questions written and answered by an LLM; a key on Llama Scope sparse-autoencoder features; e5 nearest neighbours, TF-IDF and a hidden-state probe as unconstrained references; tale length; random non-meme units. The LLM key should win on accuracy at many times the cost. Units earn a place only by beating the word key.

## Premise it tests

P1. If named units make a better key than words of equal length, the unit carves stories at joints that experts recognise. If the word key ties, the atlas's names add nothing a word list lacks.

## Biology analogue

Linnaeus's identification keys; diagnostic characters

## Cost

S. One extraction over 1,518 tales (about 20 minutes per model), an hour of local LLM answering for the rival key, key fitting on a laptop; under two days.

## Payoff

A readable taxonomy: the 15 units that sort folktales, each with a name, checked against a century of expert classification. It shows whether the atlas can be used the way a field guide is.

## Main risk

The search picks from 900,000 units but about 20,000 words, so the unit key overfits more easily; held-out types guard against it. Divisions are partly defined by who is in the tale (animals, clergy), which favours words. The tales sit on the open web with their type numbers, so both models have probably read them.

## Reviewer note

Changed: divisions are recoded from type numbers, since the file leaves them blank for 20 types and mislabels the chapter of 338 tales; one underpowered 60-type split became five folds; names come from public descriptions. Remaining: divisions are topical, so a tie with words is the likely result.

## Sources

- Hagedorn, trilogy dataset, Zenodo 6575263
- Arčon, Robnik-Šikonja and Tratnik 2025, arXiv:2510.18561
- Eklund, Hagedorn and Darányi 2023, doi:10.1515/fabula-2023-0005
- Peng et al. 2025, arXiv:2506.23845
