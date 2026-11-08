# US-24: Spotting a campaign's claims: accuracy per unit of compute

**Area:** Uses · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

For spotting the named claims of an organised campaign in public text, is any neuron-based reader on the cost and accuracy frontier against an LLM that simply reads each paragraph?

## Prediction

With 10 labelled examples per claim, neuron units reach macro-F1 (a 0-to-1 accuracy score averaged over claims) within 0.05 of a local LLM reader given the written claim definitions, at under one fifth of its measured time per paragraph for 17 claims, and no cheaper rival matches them, in English and on German translations.

## What would refute it

Any rival that costs no more per paragraph has a 95% lower bound on macro-F1 at or above the neurons' estimate (neurons are dominated), or neuron macro-F1 is more than 0.05 below the LLM reader.

## Data

CARDS (Coan et al. 2021): paragraphs from climate-contrarian blogs and think tanks, human-coded with one of 17 sub-claims or none; 23,436 train, 2,605 validation, 2,904 cleaned test (files opened 8 Nov 2026; article CC BY 4.0, data licence unstated). Labelled paragraphs carry no outlet or date; the dated 4.6-million-row file has classifier labels and no text.

## Method

Choose units and fit rivals on 10, 100 or all training examples per claim; test on the cleaned set, in English and after open machine translation into German. Report macro-F1 and measured seconds and storage per 1,000 paragraphs at 1, 5 and 17 claims, including a one-call LLM read and claims added later from stored activations. Null: random units.

## Cheap rival it must beat

Word-count logistic regression (half of the paper's own ensemble), embedding probe, the paper's released classifier, a local LLM asked once for all claims and once per claim, a probe on averaged internal states. The embedding probe is most dangerous: cheaper to run and store, and probably as accurate. If it dominates, the cost argument for neurons is dead for tracking.

## Premise it tests

P1. Mostly a usefulness test: a pass shows units carry a named claim across wordings and a language change cheaply. A fail moves the case for neurons from cost to interpretability alone.

## Biology analogue

Genomic surveillance of a spreading variant.

## Cost

M. 29,000 labelled paragraphs through two 8B models plus LLM readers; hours of laptop time.

## Payoff

A measured answer to when, if ever, reading neurons is cheaper than asking an LLM when many claims are monitored over a large public corpus.

## Main risk

Train and test share outlets, so word features may win easily in English and the German arm then carries the test. Spread across outlets and years cannot be checked with these files.

## Reviewer note

The files hold 17 sub-claims, not 27, and no outlet or date on labelled paragraphs, so held-out outlets and yearly shares were replaced by few-shot and translated tests; a one-call LLM read was added to the costs. Weakness: embedding probes will probably dominate, and spread over time is untested.

## Sources

- Coan et al. 2021 (CARDS)
- preregistration/sources/neuron_timing_qwen3-8b.json
- Jiang et al. 2025
