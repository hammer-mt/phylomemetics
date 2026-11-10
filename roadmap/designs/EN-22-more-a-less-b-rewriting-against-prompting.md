# EN-22: More A, less B: rewriting against prompting

**Area:** Meme engineering · **Status:** design registered 8 November 2026, not run · **Priority:** 3.5 of 5

## Question

Can any internal handle (one neuron, a direction or a sparse-autoencoder feature) move an existing story toward meme A and away from meme B better than simply asking a model to rewrite it?

## Prediction

No. The prompt-only rewrite gets the highest combined score, and the best internal handle reaches under 80% of it. Among handles, the difference-of-means direction (average activation on texts with the meme minus texts without) does best and the single neuron worst.

## What would refute it

The handle arm that scored best on the tuning stories, alone or added to the prompt, beats prompt-only by 0.05 or more on the 0-to-1 combined score, with a 95% interval above zero, on 160 held-out stories. That gives about 85% power if per-story differences have a spread of 0.2, less if meme pairs differ a lot.

## Data

208 film plots from the CMU Movie Summary Corpus (42,306 Wikipedia summaries, CC BY-SA), skipping the pre-registration's reserved films, cut to 10 sentences, names masked. For each of 16 seeded meme pairs from `preregistration/memes_for_tracing.json`: 13 plots containing B and not A, confirmed by an LLM reader and a human (3 tuning, 10 held out). Llama Scope sparse autoencoders (Apache 2.0).

## Method

Six arms on Llama 3.1 8B: prompt only; one neuron; 20 neurons; direction; sparse-autoencoder feature (a unit from a separately trained dictionary of activations); prompt plus direction. Handles use EN-23's contrast sentences. A Qwen3-8B judge and blind humans score A up, B down, story kept and fluency, combined so that failing one sinks the score. Repeated with A's keywords banned.

## Cheap rival it must beat

Prompt-only rewriting, the most dangerous: it beat every activation method on AxBench. Also keyword insertion (paste A's typical words in), a random direction of equal size, and text length. If prompting wins, neurons stay meters and editing tools should be built on prompts scored by the instrument.

## Premise it tests

P1. A handle that beats prompting shows units are causal levers on meaning, strong support. If prompting wins, P1 survives only as a way to read memes, and meme engineering means prompt and measure.

## Biology analogue

Gene editing against selective breeding.

## Cost

M. About a week on the laptop, with PyTorch for the feature arm or a small rented GPU; judge calls; four hours of human rating.

## Payoff

A straight answer on whether to build a meme editor on activations or on prompts, and a benchmark for narrow story motifs, which AxBench does not cover.

## Main risk

A prompt win is the expected result after AxBench, so the news value is modest. Judges may reward stated keywords, so arms look alike. The public sparse autoencoder was trained on the base model; if it rebuilds under 90% of the 4-bit Instruct model's activation variance, that arm is dropped.

## Reviewer note

The 1,000 random films are the paused pre-registration's held-out set, so the data moved to the CMU corpus. Base stories must now contain B, or 'less B' is empty. One pre-chosen handle contrast replaces five. Remaining weakness: a prompt win is already expected.

## Amendment, 10 November 2026

From the field survey (arXiv:2611.08438): its sources agree that prompting often beats handles inside the model, so the prediction stands. It adds that steering vectors often degrade on related but different text, and fail without any error. Score every handle at three doses and on 40 extra held-out stories from a second source such as folktales, and report B-down, story-kept and fluency separately. A win on film plots alone is not general.

## Sources

- Wu et al. 2025, AxBench, arXiv:2501.17148
- Chen et al. 2025, Persona vectors, arXiv:2507.21509
- He et al. 2024, Llama Scope, arXiv:2410.20526
- Bamman, O'Connor, Smith 2013, CMU Movie Summary Corpus
- Barez, Wu et al. 2026, Understanding AI from the Inside (arXiv:2611.08438)
