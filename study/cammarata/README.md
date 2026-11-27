# An independent test of "neural networks are already clean", on its own terms

**Plan of 2026-11-27, committed before any test sentence was run through a model. Exploratory; not part of paper v1.**
Sources: Cammarata, [Neural networks are already clean](https://already-clean.goodfire.com/) (15 October 2026), and Tigges et al.,
[Neurons as an interpretable basis](https://www.goodfire.com/research/neurons-as-an-interpretable-basis) (10 November 2026).

The point is a fair test that can come out either way. What can be shown is that the published explanations predict
strong firing, in a lab that did not write them, and that the Batman memes behave like those neurons. It cannot prove the
whole thesis (superposition is absent), and the "90% have one main job" figure has no published protocol to replicate. The
companion paper, on another model (Qwen3.6-35B-A3B), puts 47% of poles at one mode and 69% at one dominant mode.

## Their rules, followed here

Model Qwen3-32B, unquantised. A firing counts only when it is strong (at least 30% of the neuron's strongest corpus
example) and the SwiGLU gate is open. Gate-closed firings with a negative value ("double negative" echoes) are separated,
not mixed in. Poles are separate. Sentences are written from the explanation alone, before any measurement.

## Part 1. Replicate the published test (the method itself)

60 dense neurons drawn at random (seed 20261127) from the 500 Qwen3-32B neurons in the essay's public gallery
(`data/gallery_sample.json`; the explanations are Goodfire's and are not copied here). Fresh agents see only the
explanation and say which pole it describes, then write, per neuron:
5 texts that should fire it, 5 near misses, 5 texts on the most distant concept they can invent, and 1 blind maximum.
A neuron is scored when the writer, from the explanation alone, finds that it describes the positive side.

**Published (200 random gallery neurons, 183 scored):** should-fire texts fire strongly 76% of the time, near misses 11%,
distant texts 1%; the explanation wins 164, ties 15, loses 4. **Replicates** if should-fire is at least 60% and exceeds near misses
by at least 40 points, with distant at most 5%. **Fails** if should-fire minus near miss is below 20 points. Between is partial.

**Added control (theirs has none):** every neuron is also scored on the other neurons' should-fire texts. Strength is judged
against a reference maximum from our own corpus sample, not theirs: the top gate-open response over a fixed public
text sample, recorded with its size. A smaller sample gives lower maxima and so inflates "strong" rates, so the published thresholds
are also reported against the peak value stated in the gallery's own cue list, where there is one.

## Part 2. Are Batman's memes such neurons? (the project's claim)

The same protocol on: (a) the 32 Batman-core meme poles found in Qwen3-32B as in `discover.py` (present in at least 29 of 32
works) with the gate-open rule; (b) 32 random poles of the same layers and activity; (c) 32 **random directions**: random unit
vectors in the same MLP layers' activation space, explained and tested the same way. (c) is the test of whether the neuron basis is
special. Explanations are written by agents from the 12 strongest gate-open passages (with context), and a second, blind agent
writes the texts. Agents are not told which group a unit is in.

**Supported** if (a) beats (c) on should-fire minus near-miss rate by at least 15 points with a unit-level permutation p < 0.05, and
(a) is not more than 10 points behind (b). Also reported: the share of each published Batman meme's strongest corpus firings that
are gate-closed echoes.

## Part 3. The published Batman units, rechecked

For the 31 named units of the paper: what share of their 12 strongest corpus sentences have the gate open, and whether the
name-detector units (24 of 33 by the paper's own count) still look like names when only gate-open firings are kept.

## Compute (not yet approved)

Two short sessions on one rented GPU: about 1 hour to collect top passages and activations, then the agents write, then about 1 hour to score.
