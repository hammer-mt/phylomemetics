### 3.8 Steering and ablation

**Steering works on the essay's neurons and not on ours.** The steering multiple was chosen on 40 neurons from the essay's public gallery. With a neuron held at 5 times its scale, a blind judge picked its title out of five for 23 of the 40 (against 20% by chance, p = 2 × 10⁻⁷). At 1, 2 and 3 times the scale the judge was right for 12, 13 and 18. So the mechanism does what the essay says it does, on the neurons the essay chose to show.

On the study's units it fails. The judge picked the right title for 17 of the 126 idea units (13%, p = 0.98 against chance), and for 12 of the 115 name or word units (10%). The criterion is not met. Pushing a unit did change what the model wrote: in only 3 of the 241 items were all three continuations the same as the unpushed ones. But the change did not point to the unit. The judge answered "none of these" for 147 of the 241 items, and a unit's own key words turn up in 24% of the continuations when it is pushed and in 25% when it is not. Genre units did best, at 7 of 23 (30%, p = 0.16).

We can think of three reasons and cannot tell them apart. The gallery neurons were picked by the essay's authors as clear cases, and ours were picked by rule from Batman's canon, where most are name pieces, slots before a name and grammar, none of which has much to say in a neutral story opening. The judge's task was also harder here, because the four wrong titles were of the same kind as the right one. And our mechanism is our own guess (Section 4.4). Whatever the reason, the descriptions in Section 3.7 are supported by what the units respond to, and not by what they do when pushed.

**Ablation: Batman's canon carries the model's recognition of Batman, but the test does not pass by its own rules.** Qwen3-32B is asked which hero a name-free summary is about, with that work's canon units switched off while it reads the summary: a median of 212 units out of 3.3 million. The log-odds that its reply begins with Batman's name fall by 2.39 on average (Table 5, Figure 7). Random sets of as many units, matched on layer, pole and how many name-free Batman works they are present in, lower them by 0.02 on average, and the largest of 200 such sets by 0.12. Sets also matched on how much activation they remove do no more (largest 0.11). The canon's effect is larger than every set's in 24 of the 32 works. With the canon off, the model's replies name Batman for 13 works where they named him for 20. And the effect is specific on average: the same units switched off in the seven hero films do not lower the log-odds of those films' own heroes (mean −0.07; single films range from a rise of 2.62 to a fall of 1.06; p = 0.0017 for the difference).

| | Qwen3-32B | Qwen3-8B |
| --- | --- | --- |
| Canon units switched off per work (median) | 212 | 78 |
| Fall in the log-odds of Batman's name, mean over the 32 works | 2.39 | 1.33 |
| 200 random sets matched on works: mean, largest | 0.02, 0.12 | 0.03, 0.11 |
| p against those sets | 0.005 | 0.005 |
| 200 sets also matched on activation removed: largest | 0.11 | 0.12 |
| p against those sets | 0.005 | 0.005 |
| Hero films, log-odds of their own hero's name: mean fall | −0.07 | 0.80 |
| p that the fall is larger for Batman | 0.0017 | 0.28 |
| Replies that name Batman: untouched, then canon off | 20, then 13 | 20, then 16 |
| Untouched replies that do not begin with a name (limit 4) | 5 | 0 |
| Criterion | not evaluable | not met |

**Table 5.** The ablation. With 200 random sets the smallest possible p is 0.005.

By the rules set beforehand, neither model passes. The plan said the test would not count if more than 4 of the 32 untouched replies began with something other than a name, because the score is the probability of the reply's first word. In Qwen3-32B five do (three begin "The story is about", two begin with a placeholder code), so its ablation is not evaluable with this question, and we do not count it as a pass. Qwen3-8B answers with a bare name every time, and its canon's effect is as clear (1.33 against 0.03), but it is not specific to Batman: the hero films' own names fall by 0.80 (p = 0.28).

So the numbers say that a few hundred units carry much of what lets Qwen3-32B recognise Batman in a name-free summary, and the registered test does not confirm it. A question that forces a one-word answer would settle it, and that is a new test.

**Where the effect comes from.** Four further versions were run with 20 random sets each. They are descriptions, not tests, and the canon's effect was the largest of the 21 in every one.

- *Most of the effect is in weak activity.* Switching off only the canon's strong firings, at 30% of scale or more, lowers the log-odds by 0.42. Switching off only its weaker activity lowers them by 1.75. In Qwen3-8B the figures are 0.34 and 1.05. On name-free text the canon units seldom fire strongly, which is why E1 fails. Their weak activity still adds up to most of what the model uses.
- *With the names present the canon is not needed.* On the original texts the fall is 0.28, and the model names Batman for 30 of the 32 works before and after. The names reach the answer by other routes as well.
- *Switching the canon off on the question as well as the summary* gives a larger and less specific effect: a fall of 5.34, replies naming Batman down from 20 to 2, and the hero films' own names down by 1.50.

![Figure 7](fig_ablation.png)

**Figure 7.** Switching off Batman's canon while a model reads a name-free summary. For each model, the mean fall in the log-odds that the reply begins with Batman's name, for the canon and for each of the random sets of matched units, and the fall in the hero films' own hero names.

### 3.9 The same units in other models

Version 1 found that its strongest memes had twins in Llama 3.1 8B, the units there that correlate best with them, at a median correlation of 0.82. It also found that the agreement was mostly about names and collapsed without them. Those figures were measured on the sentences the twins were chosen on. Here each of the 241 explained Qwen3-32B units has a twin in each other model, chosen on the original sentences and then scored where it was not chosen.

**A twin responds to the same description, more weakly.** On the texts a tester wrote for the Qwen3-32B unit, its Qwen3-8B twin fires strongly on 33% of the new examples and 3% of the near misses, where the unit itself fires on 81% and 3% (Table 6). The twin's description wins for 61% of the units, against 24% when units are paired with texts written for another unit (p = 0.0001), and 19% of twins are idea-bound against 5%. The two Llama models, from another company, give much the same at 8B and at 70B: 28% and 27% of new examples, 3% of near misses, and wins for 52% and 49% of units against 20% and 19%.

**On name-free sentences a twin is no closer than a unit that shares its best word.** Twins correlate with their Qwen3-32B unit at a median of 0.61 to 0.62 on the original sentences in all three models. On the name-free sentences that falls to between 0.26 and 0.28. Control units, matched to the twin on how well a single word predicts them, reach 0.38 in all three. For the idea units alone the twins and the controls are level (a difference of −0.02, 95% interval −0.07 to 0.02, in Qwen3-8B). The plan asked whether twins beat both controls for idea units, and in no model do they.

| | Qwen3-8B | Llama 3.1 8B | Llama 3.3 70B |
| --- | --- | --- | --- |
| Twin's correlation with the unit, original sentences (median) | 0.61 | 0.62 | 0.62 |
| The same on name-free sentences | 0.28 | 0.26 | 0.27 |
| Word-matched control units, name-free sentences | 0.38 | 0.38 | 0.38 |
| Twin fires strongly on the unit's new examples (A) | 33% | 28% | 27% |
| On its keyword-free texts (B) | 22% | 20% | 19% |
| On its near misses (C) | 3% | 3% | 3% |
| Twin's description wins, against mismatched pairs | 61%, 24% | 52%, 20% | 49%, 19% |
| Twins that are idea-bound, against mismatched pairs | 19%, 5% | 18%, 4% | 17%, 3% |

**Table 6.** Twins of the 241 explained Qwen3-32B units in three other models. The Qwen3-32B units themselves fire on 81% of their new examples, 45% of their keyword-free texts and 3% of their near misses.

So the same descriptions pick out related units across models and across companies, which is some support for treating a unit's idea as more than an accident of one network. But a twin is a much weaker detector of that idea than the unit it was matched to, and in name-free text the match is no better than shared vocabulary would give, as version 1 found.
