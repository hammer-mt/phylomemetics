# IN-28: Do units read known works differently from unseen ones?

**Area:** The instrument · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

Does a unit's reading of a plot change when the model already knows the work, compared with a work released after the model?

## Prediction

Agreement between a unit's reading of a Wikipedia plot and of its paraphrase, and of the plot with character names swapped for new ones, is the same for films the model recognises and films released after it: difference in kappa (chance-corrected agreement) under 0.05, with the 90% interval inside plus or minus 0.10.

## What would refute it

Kappa difference of 0.10 or more between recognised and unseen films on either alteration. A null result counts only if the positive control works: on 100 famous public-domain passages the model's token probability must be clearly higher for the verbatim text than for its paraphrase.

## Data

English Wikipedia plot sections (CC BY-SA 4.0): 300 films released after June 2025 (after Qwen3, April 2025, and Llama 3.1, 2024), original stories only, no sequels or adaptations; 300 films from 2005 to 2020 matched on genre and length. One paraphrase and one name-swapped version each; 60 checked by hand. 100 famous public-domain passages.

## Method

Units: frozen IN21 units, labelled concept units, 2,000 random active units; name detectors excluded. Split older films by whether the model names the film from its masked plot (the recognition check in preregistration/readers.py). Compare verbatim-paraphrase and verbatim-name-swap agreement across recognised, unrecognised and post-release films (difference in differences), bootstrap over films.

## Cheap rival it must beat

TF-IDF and word presence (cannot recall anything; they show the loss from paraphrase alone), sentence embeddings, an LLM reader that may recognise the film. Most dangerous: paraphrase damage, equal in all groups by design. If no effect appears, the Batman readings stand as content readings on this point.

## Premise it tests

P1. A recognition effect would mean readings on famous works are partly recall, weakening every Wikipedia-based result; no effect removes a standing objection.

## Biology analogue

Reference bias: reads matching the reference genome map better

## Cost

S to M. 1,800 texts per model, minutes of extraction, 1,200 rewriting calls and a recognition pass; 3 days.

## Payoff

Whether future tests must use works released after the models, or Wikipedia text is safe.

## Main risk

Recent films differ from older ones in summary style and length, which matching cannot fully remove; the number of eligible post-release original films was not confirmed.

## Reviewer note

Release date is a weak stand-in for memorisation: an 8B model has not memorised most older plots. Added a per-film recognition split, a name-swap arm (recall keyed by names is the real Batman threat), a positive control and an equivalence margin. Sequels and adaptations are now excluded.

## Sources

- Ferrando et al. 2024
- Stoehr et al. 2024 (arXiv:2403.19851)
- Yang et al. 2025 (Qwen3 Technical Report)
- preregistration/readers.py (franchise recognition check)
