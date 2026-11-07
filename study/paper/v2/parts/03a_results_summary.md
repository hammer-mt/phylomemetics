## 3. Results

### 3.1 The six confirmatory tests, and what became of version 1's conclusions

One of the six confirmatory tests passes (Table 1).

| Test | What was predicted | Qwen3-32B | Outcome |
| --- | --- | --- | --- |
| E1, identity | Without names, the share of Batman's canon a work carries separates Batman from other hero films (AUC at least 0.80) | AUC 0.36 (95% interval 0.17 to 0.56); 0.37 on openings | Falsified |
| E2, heredity | Adaptations carry their documented source comic's units more than other comics' | 15% of comics beat the source, against 46% by chance; exact p = 0.0016 | Supported |
| Residual test | The units still find the source once word measures are regressed out | p = 0.69; 0.86 with every measure regressed out | Not supported |
| E3, selection | Units of popular early works spread further | rho = 0.35, p = 0.37 | Not detected |
| Ablation | Switching off the canon lowers the model's recognition of Batman more than matched random units, and more than it lowers other heroes' | Fall of 2.39 in log-odds against 0.02; p = 0.005, 0.005 and 0.0017 | Not evaluable under the rule set beforehand |
| Steering | A blind judge can tell which idea unit was pushed (chance 20%) | 17 of 126 (13%), p = 0.98 | Not met |

**Table 1.** The six confirmatory tests, all on Qwen3-32B under the new rule. With Holm's adjustment over the five tests that have a p value, E2's is 0.008 and the ablation's 0.020.

Read against version 1, the conclusions are these.

1. **Heredity stands.** It is supported in Qwen3-32B and in each of the other three models (Sections 3.4 and 3.10).
2. **Words still find the sources as well as neurons do.** The larger model, the full-precision weights and the stricter rule do not change that, and no model adds anything to the words (Section 3.4).
3. **The hint that Llama's neurons add something beyond words is withdrawn.** The test behind it can be passed by words alone. Under a test that cannot, no model passes under either rule (Sections 2.5 and 3.10).
4. **Identity fails, and more plainly.** The canon share falls from 0.52 to 0.36. Version 1 blamed the scoring rule. Given the words' scoring rule, the units reach 0.62, which is no better than the length of the text (Section 3.3).
5. **The name-free core has gone.** Version 1 found twelve name-free units that read like the definition of a superhero. Under a rule that counts only strong activations, one unit is left (Section 3.2).
6. **The canon is mostly names, now by a blind judgment.** 38 of 60 core families detect a name, place or word (Section 3.7).
7. **Selection is still not detected,** in a test that could not have detected much (Section 3.5).

Three things are new.

8. **The units can be described, and the descriptions can be tested.** Descriptions written from a unit's strongest examples predict its firing on new text at about the rate the essay reports, and 46% of units still fire when their idea is put in other words (Section 3.7).
9. **The model's recognition of Batman runs through the canon units,** in numbers that the registered test does not let us count as a pass (Section 3.8).
10. **Pushing a unit does not produce its description,** although the same procedure works on the essay's own neurons (Section 3.8).

[CONTRAST-SENTENCE]
