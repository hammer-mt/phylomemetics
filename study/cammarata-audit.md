# Cammarata audit of the 31 named Qwen3-8B memes

Checks after Nick Cammarata's "Neural networks are already clean": depth, SwiGLU sign regime, the negative pole, a blind maximum written from the label alone, and a near miss. Produced by `cammarata_audit.py`; the data is in `cammarata-audit.json`. Activations are pole-signed (bigger is stronger) and in the model's raw units. Corpus sentences were read in the context of their whole work, as in the paper; probe sentences were read on their own, first token excluded, as in the paper's rewording test. Percentiles are against the unit's responses to the 3,700 corpus sentences.

## Summary

**Headline.** One of the 31 labels survives all three checks: Two-Face's coin, a layer-26 unit that a name-free coin-flip sentence drives past its corpus maximum while a coin in a vending machine leaves it at 15% of that. Bats survives on the lenient bar as a clean word-sense unit (the animal, not the cricket bat). The other 29 labels should change: 26 to the name, word or sub-word token the unit fires on, 2 to a last-layer next-token description, and the secret identity to a note that it is a lexical cluster (identity, masked, vigilante) that falls just short of its top 8. The paper read each label off a unit's eight strongest sentences, and those sentences are where the names are: 49 of the 50 strongest firings of "the billionaire vigilante" peak on the token Gordon, all 50 of "franchise roll-call" on -ham in Arkham, 43 of "the Batcave and bat gear" on the Bat- prefix, every gate-open firing of "Harvey Dent" on Harvey. Of the paper's four memes that survived rewording in both models (a funeral, Two-Face's coin, the secret identity, the joke), this audit keeps the coin, keeps the secret identity with a lexical note, and relabels the funeral and the joke as the words funeral and joke/Joker.

**Depth.** 8 units sit in layers 0 to 3 and all 8 behave as token detectors (chemicals, funeral, Selina, Riddler, antidote, joke, bomb, mutating). Depth is not sufficient: the layer-10 and layer-11 units are name detectors too, and the two layer-35 units predict the next token.

**Sign regime.** The echo never reaches a top 8 (the first echo firing sits at rank 11 to 45), so relabelling from gate-open firings gives the same eight sentences, and for the 7 echo-dominated units (lair 78%, antidote 74%, fear toxin 74%, Jason Todd 70%, funeral 68%, mutating 66%, Ra's al Ghul 56%) the gate-open label is the keyword alone. The echo matters below the top 8 in two ways. It is where the apparent generalisation lives: the funeral unit's 13 wedding firings, the antidote unit's 24 against, 6 post- and 3 non- firings, the mutating unit's transforms and genetically, and the lair unit's -face and race are all gate-closed; a reader of the top 50 would call these concept units, and the gate says otherwise. And the paper's presence rule scores a work by its three strongest sentences: on average 49% of the Batman work scores and 83% of the ordinary-film scores that set the presence threshold come from echo firings; for 18 of the 31 units at least half of the Batman works are scored mostly by echo, and for 26 units most of the 69 films are. A presence call then means "the keyword occurs" for a Batman work and "echo noise" for the bar it beats.

**Negative poles.** Mostly sub-word noise at a tenth of the positive pole's strength. Five draw a readable distinction: the secret-identity channel runs from a masked identity to a nickname declared in quotes; the Superman channel's other pole is apparent death and return (seemingly killed, presumed dead, resurrection); the rogues'-gallery and origin-wound channels oppose Batman names to other franchises' names; the bats channel opposes bats to other animals.

**Blind maximum.** 5 units' best blind sentence beats the in-context corpus maximum, and 4 of those are word detectors that their own keyword drives (joke, lair, chemicals, antidote); the coin is the fifth. Read standalone, as the probes are, the corpus top 8 usually score higher than in context, and only the joke and chemicals units are beaten that way. 9 blind sentences would have made their unit's top 8. 18 units never get a name-free sentence past 20% of their maximum, including every character label but the coin's.

**Near miss.** 12 stay low. For 10 units the near miss scores above the weakest sentence the label was read from, and for 15 it beats the best blind sentence outright: a box-office bomb (17.3 against 13.9 for any real bomb), a funeral home turned bakery, Harvey the postman's dented gate, two brothers named Jason and Todd, the star Algol's name al-ghul, a fox in the allotments, a badger's lair, a mutating flu strain, a comedy that is an antidote to loud sequels.

**Caveats.** Probe sentences are read on their own and corpus sentences in the context of their whole work, as the paper's rewording test did; for keyword units the standalone reading is as high or higher, so the comparison does not favour the corpus, but for the context-dependent units (the orphan ward, the corrupt police force, Selina Kyle) it lowers the probes, and those units fail for other reasons too. Three blind attempts per label, written by agents that saw the label and its kind only, is a small search: a miss shows the label does not predict the unit, not that no sentence could. One near miss per label was written by a fourth agent from the label alone; the Selina Kyle one began with the name, which the first-token exclusion drops, and was reworded and re-run. Everything is one forward pass per sentence through 4-bit Qwen3-8B with MLX; Llama 3.1 8B was not audited. The thresholds (first four layers, echo share 25%, near miss under half the blind maximum and under the weakest of the top 8) are stated in the script. Moving the echo threshold anywhere between 25% and 100% changes no count; moving the near-miss ratio to 1.0 changes none either while the near miss must also stay below the weakest of the top 8. Only dropping that second condition changes the result, letting the joke and chemicals units through with near misses at 95% and 73% of their blind maxima, which is what the half-of-the-blind-maximum rule is there to exclude.

| Count | of 31 |
| --- | --- |
| In the first 4 layers (likely token or spelling detectors) | 8 |
| Echo share of the top 50 firings at or below 25% | 18 |
| Echo dominates the top 50 (above 50%), relabelled from gate-open firings | 7 |
| Best blind maximum beats the unit's corpus maximum (in context, as the paper recorded it) | 5 |
| Best blind maximum beats the corpus top 8 read standalone, as the probes are | 2 |
| Best blind maximum would make the corpus top 8 the label was read from | 9 |
| Near miss stays low (under half the best blind maximum and under the weakest of the corpus top 8) | 12 |
| **Survives** (blind beats the corpus maximum, near miss low, echo small) | **1** |
| Survives on the lenient bar (blind would make the top 8 instead of beating the maximum) | 2 |

## What each label should become

| # | Paper's label | Unit | Reads as | Suggested label |
| --- | --- | --- | --- | --- |
| 1 | The origin wound | L10·9068+ | name / word / token | Batman-universe character names in a Batman summary (Bat-, Wayne, Dent, Gordon, Drake, Todd) |
| 2 | The secret identity | L13·8608+ | keep | The secret identity, as a lexical cluster (identity, masked, vigilante); keep with that note |
| 3 | Heir to a dead father's fortune | L10·5634+ | name / word / token | The surname Wayne (and Stark) |
| 4 | The billionaire vigilante | L4·1072+ | name / word / token | The name Gordon |
| 5 | The Batcave and bat gear | L11·9843+ | name / word / token | The Bat- prefix in compounds (Batcave, Batsuit, Batmobile, the Bat) |
| 6 | Bats | L8·7130+ | keep | Bats, as the word bat(s) in its animal sense; keep |
| 7 | Gotham City | L6·10402+ | name / word / token | The token City after Gotham |
| 8 | Commissioner Gordon | L11·12069- | name / word / token | Batman-world names (Gordon, Batman, Bruce), weak |
| 9 | Losing the protégé | L11·7340- | name / word / token | Batman-world names (Batman, Wayne, Todd, Gordon), weak |
| 10 | The joke | L2·6468+ | name / word / token | The tokens Joker and joke |
| 11 | The rogues' gallery | L10·5942+ | name / word / token | Batman villain names (Poison Ivy, Scarecrow, Penguin, Harley Quinn, Bane) |
| 12 | Superhero powers (peaks on Superman) | L10·7189+ | name / word / token | Superman names and kryptonite (Kal-El, Clark, kryptonite) |
| 13 | Franchise roll-call (Wikipedia style) | L5·3626+ | name / word / token | The token -ham in Arkham |
| 14 | The butler who raised him | L6·4705- | name / word / token | The token -worth in Pennyworth |
| 15 | Taking in an orphan as a ward | L35·6441- | last-layer predictor | Layer-35 next-token unit around Dick Grayson sentences (predicts Dick / Robin) |
| 16 | The villain's lair | L4·7177- | name / word / token | The word lair |
| 17 | A bomb | L3·1967- | name / word / token | The word bomb, any sense |
| 18 | A funeral | L1·5621+ | name / word / token | The word funeral |
| 19 | Fell into a vat of chemicals | L0·6344- | name / word / token | The tokens chemical(s) and -ylum (Asylum) |
| 20 | Two-Face's coin | L26·3329- | keep | A coin flip deciding a fate (Two-Face's coin); keep |
| 21 | Harvey Dent | L5·6648+ | name / word / token | The first name Harvey |
| 22 | Selina Kyle | L1·1051+ | name / word / token | The token Sel- (Selina) in a Batman context |
| 23 | The Riddler | L1·10818- | name / word / token | The token -ler in Riddler (and riddl-) |
| 24 | Jason Todd | L4·4504- | name / word / token | The name Todd |
| 25 | Ra's al Ghul | L4·9760- | name / word / token | The token -ul in Ghul |
| 26 | Lucius Fox | L5·10113- | name / word / token | The surnames Quinn and Fox (and Zorro) |
| 27 | A corrupt police force | L35·4074- | last-layer predictor | Layer-35 unit on Gotham before (City) Police Department |
| 28 | Prison and the inmates released | L4·8535+ | name / word / token | The tokens Penny(worth) and Penit(entiary) |
| 29 | The antidote | L1·10779- | name / word / token | The token antid- / anti- |
| 30 | A mutating super-formula | L3·10668+ | name / word / token | The stem mutat- / Mut- |
| 31 | Fear toxin | L5·5136+ | name / word / token | The token comp(ound), weak |

## Overview

| # | Meme | Unit | Layer | Echo (top 50) | Corpus max (in context) | Corpus top 8 standalone | Best blind | Blind pct | Near miss | Near pct | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | The origin wound | L10·9068+ | 10 | 0% | 18.75 | 17.00 | 3.03 | 75.8 | 0.13 | 22.2 | fails |
| 2 | The secret identity | L13·8608+ | 13 | 0% | 3.27 | 4.09 | 2.19 | 99.6 | 0.10 | 5.8 | fails |
| 3 | Heir to a dead father's fortune | L10·5634+ | 10 | 0% | 1.45 | 1.40 | 0.27 | 90.4 | 0.12 | 34.3 | fails |
| 4 | The billionaire vigilante | L4·1072+ | 4 | 0% | 18.62 | 18.88 | 0.07 | 87.8 | 0.92 | 97.6 | fails |
| 5 | The Batcave and bat gear | L11·9843+ | 11 | 0% | 5.66 | 5.62 | 2.05 | 97.8 | 0.18 | 59.7 | fails |
| 6 | Bats | L8·7130+ | 8 | 0% | 14.19 | 13.50 | 12.88 | 99.9 | 2.20 | 97.0 | lenient |
| 7 | Gotham City | L6·10402+ | 6 | 0% | 3.03 | 3.03 | 0.37 | 95.5 | 2.14 | 99.7 | fails |
| 8 | Commissioner Gordon | L11·12069- | 11 | 2% | 1.16 | 1.07 | 0.18 | 58.4 | 0.10 | 17.0 | fails |
| 9 | Losing the protégé | L11·7340- | 11 | 0% | 2.55 | 2.53 | 0.20 | 69.6 | 0.27 | 90.2 | fails |
| 10 | The joke | L2·6468+ | 2 ⚠ | 6% | 10.31 | 12.38 | 12.69 | 100.0 | 12.00 | 100.0 | fails |
| 11 | The rogues' gallery | L10·5942+ | 10 | 0% | 6.47 | 6.75 | 0.40 | 85.4 | 0.16 | 39.6 | fails |
| 12 | Superhero powers (peaks on Superman) | L10·7189+ | 10 | 0% | 7.03 | 7.16 | 5.34 | 99.6 | 0.08 | 24.3 | fails |
| 13 | Franchise roll-call (Wikipedia style) | L5·3626+ | 5 | 0% | 13.00 | 13.00 | 0.02 | 31.7 | 0.03 | 39.7 | fails |
| 14 | The butler who raised him | L6·4705- | 6 | 2% | 6.69 | 7.25 | 0.19 | 87.9 | 0.03 | 10.2 | fails |
| 15 | Taking in an orphan as a ward | L35·6441- | 35 | 0% | 70.50 | 109.50 | 0.32 | 67.6 | 0.05 | 32.7 | fails |
| 16 | The villain's lair | L4·7177- | 4 | 78% | 8.12 | 9.81 | 9.12 | 100.0 | 9.75 | 100.0 | fails |
| 17 | A bomb | L3·1967- | 3 ⚠ | 30% | 13.88 | 14.38 | 13.75 | 100.0 | 17.25 | 100.0 | fails |
| 18 | A funeral | L1·5621+ | 1 ⚠ | 68% | 13.88 | 14.12 | 12.31 | 100.0 | 12.88 | 100.0 | fails |
| 19 | Fell into a vat of chemicals | L0·6344- | 0 ⚠ | 0% | 5.06 | 5.69 | 5.81 | 100.0 | 4.22 | 99.9 | fails |
| 20 | Two-Face's coin | L26·3329- | 26 | 6% | 24.88 | 42.50 | 32.25 | 100.0 | 4.84 | 99.5 | survives |
| 21 | Harvey Dent | L5·6648+ | 5 | 50% | 10.38 | 10.12 | 0.24 | 97.8 | 13.00 | 100.0 | fails |
| 22 | Selina Kyle | L1·1051+ | 1 ⚠ | 38% | 2.55 | 5.22 | 0.02 | 86.4 | 1.45 | 99.4 | fails |
| 23 | The Riddler | L1·10818- | 1 ⚠ | 42% | 11.25 | 12.50 | 4.22 | 99.3 | 4.44 | 99.3 | fails |
| 24 | Jason Todd | L4·4504- | 4 | 70% | 20.25 | 20.88 | 0.08 | 85.1 | 12.50 | 99.8 | fails |
| 25 | Ra's al Ghul | L4·9760- | 4 | 56% | 9.38 | 9.94 | 0.10 | 93.4 | 13.44 | 100.0 | fails |
| 26 | Lucius Fox | L5·10113- | 5 | 0% | 13.81 | 15.50 | 0.12 | 65.6 | 9.06 | 99.8 | fails |
| 27 | A corrupt police force | L35·4074- | 35 | 28% | 40.00 | 33.50 | 2.22 | 94.9 | 0.70 | 23.8 | fails |
| 28 | Prison and the inmates released | L4·8535+ | 4 | 36% | 8.19 | 7.91 | 0.17 | 98.1 | 0.08 | 91.7 | fails |
| 29 | The antidote | L1·10779- | 1 ⚠ | 74% | 7.91 | 8.44 | 8.38 | 100.0 | 9.50 | 100.0 | fails |
| 30 | A mutating super-formula | L3·10668+ | 3 ⚠ | 66% | 10.56 | 11.06 | 9.75 | 99.9 | 11.12 | 100.0 | fails |
| 31 | Fear toxin | L5·5136+ | 5 | 74% | 3.73 | 5.34 | 0.21 | 98.2 | 0.04 | 27.9 | fails |

## Per-meme tables

### 1. The origin wound — L10·9068+

*trope; present in 32 of 32 Batman works; family of 129. Paper's note: A childhood loss that turns someone into a costumed hero: parents' funeral, fear of bats, a costume to frighten the killers.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 10 of 36 | yes |
| 2. Sign regime | Top 50 firings: 50 gate-open, 0 echo, 0 other; echo share 0% (0 of the top 8; first echo at rank none in the top 50). Work scores (mean of the top 3 sentences) drawn from echo firings: 0% in Batman works, 31% in ordinary films; 0 of 32 Batman works and 21 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Bat' ×9 (9/0), ' Dent' ×8 (8/0), ' Wayne' ×7 (7/0), ' Gordon' ×4 (4/0), ' Drake' ×3 (3/0), ' Todd' ×3 (3/0) | yes |
| 3. Negative pole | L10·9068-, max 6.56. The opposite pole peaks on the acronym S.H.I.E.L.D. and on Marvel names (Aunt May, Stane), so the channel runs from Batman-universe surnames to other franchises' names, not from the origin wound to anything. | – |
| 4. Blind maximum | Best of 3: 3.03, against 18.75 for the corpus maximum in context and 17.00 for the corpus top 8 read standalone (corpus p75.8; peak on 'ante', open). The weakest of the corpus top 8 is 15.25 in context, 6.69 standalone; neutral sentences max 0.15. | no |
| 5. Near miss | 0.13 (corpus p22.2; 1% of the in-context maximum, 4% of the best blind maximum; peak on ' before') | yes |
| **Verdict** | Fails. Every one of the top 50 is gate-open and peaks on a franchise name read inside a Batman summary (Bat-, Dent, Wayne, Gordon, Drake, Todd, Conroy); the three top sentences happen to be origin scenes because origin scenes name Bruce Wayne. The best name-free sentence about the trauma reaches 16% of the maximum, at the token -ante of vigilante. | **fails** |

Suggested label: **Batman-universe character names in a Batman summary (Bat-, Wayne, Dent, Gordon, Drake, Todd)**

Strongest corpus sentences (activation, regime, peak token):

- 18.75 open ' bat' · *Earth One*: Bruce dons a bat-themed costume as a disguise to frighten his parents' alleged killers.
- 16.88 open ' Wayne' · *Batman v Superman*: In a flashback to his childhood, Bruce Wayne flees his parents' funeral.
- 16.75 open ' Wayne' · *Batman Begins*: In Gotham City, young Bruce Wayne falls down a well and develops a fear of bats.
- 16.00 open ' Drake' · *Knightfall*: He also assigns Robin (Tim Drake) to train Jean-Paul Valley in detective work to aid them as an ally, hoping to guide Valley's brainwashing away from making him a villai…
- 15.81 open ' Dent' · *Dark Knight Rises*: Meanwhile, eight years after the death of Gotham City district attorney Harvey Dent and the arrest of the Joker, organized crime has been eradicated in Gotham by the Den…
- 15.62 open ' Wayne' · *Mask of the Phantasm*: A young Bruce Wayne, after years of training abroad, vows to fight crime in Gotham City in memory of his murdered parents.
- 15.25 open ' Todd' · *Dark Knight Returns*: Over a decade after the United Nations outlawed superheroes and the Joker murdered Batman's protégé, Robin (Jason Todd), a 55-year-old Bruce Wayne battles alcoholism in…
- 15.25 open 'roy' · *Arkham City*: Returning characters from Arkham Asylum include Batman (Kevin Conroy), the Joker (Mark Hamill)—in what Hamill stated would be his final time voicing the character; (he s…

Negative pole L10·9068-, strongest sentences:

- 6.56 '.L' · *Iron Man*: Potts meets with Agent Phil Coulson of S.H.I.E.L.D., an intelligence agency, to inform him of Stane's activities.
- 5.38 '.L' · *Iron Man*: In a post-credits scene, S.H.I.E.L.D. director Nick Fury visits Stark at his home, telling him that he has become part of a "bigger universe", and that he wants to discu…
- 4.34 ' May' · *Spider-Man*: Peter's aunt May invites Mary Jane, Harry, and Norman for Thanksgiving dinner.
- 4.19 '.L' · *Iron Man*: Potts and several S.H.I.E.L.D. agents attempt to arrest Stane, but he dons his suit and overpowers them.
- 3.86 ' Black' · *Under the Red Hood*: He announces a takeover of their drug trade, taking 40 percent of their profit while offering them protection from Black Mask and Batman, but threatens to kill them if a…
- 3.66 'human' · *Spider-Man*: The chemical gives him superhuman strength but causes him to go insane and kill one of his scientists.
- 3.34 'are' · *Daredevil*: By night, he fights crime as the costumed vigilante "Daredevil".
- 3.17 'ears' · *Under the Red Hood*: Black Mask sends the Fearsome Hand of Four to lure Red Hood.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 17.00 (p100.0, open, peak ' Wayne') · *Batman Begins*: In Gotham City, young Bruce Wayne falls down a well and develops a fear of bats.
- 16.88 (p99.9, open, peak ' Dent') · *Dark Knight Rises*: Meanwhile, eight years after the death of Gotham City district attorney Harvey Dent and the arrest of the Joker, organized crime has been eradicated in Gotham by the Den…
- 16.62 (p99.9, open, peak ' Wayne') · *Batman v Superman*: In a flashback to his childhood, Bruce Wayne flees his parents' funeral.
- 15.69 (p99.9, open, peak ' Wayne') · *Mask of the Phantasm*: A young Bruce Wayne, after years of training abroad, vows to fight crime in Gotham City in memory of his murdered parents.
- 15.19 (p99.8, open, peak ' Todd') · *Dark Knight Returns*: Over a decade after the United Nations outlawed superheroes and the Joker murdered Batman's protégé, Robin (Jason Todd), a 55-year-old Bruce Wayne battles alcoholism in…
- 14.94 (p99.7, open, peak ' Todd') · *A Death in the Family*: While eavesdropping on a child pornography ring and awaiting police backup in Gotham City, Jason Todd (Robin) ignores Batman's orders and attacks the criminals.
- 14.12 (p99.3, open, peak ' Drake') · *Knightfall*: He also assigns Robin (Tim Drake) to train Jean-Paul Valley in detective work to aid them as an ally, hoping to guide Valley's brainwashing away from making him a villai…
- 6.69 (p87.0, open, peak '-themed') · *Earth One*: Bruce dons a bat-themed costume as a disguise to frighten his parents' alleged killers.

Blind maximum attempts:

- 3.03 (p75.8, open, peak 'ante') blind_C: As a child he watched his parents gunned down in an alley, and that single traumatic wound, the origin of all his pain, forged him into the vigilante he became.
- 0.20 (p43.4, echo, peak ' on') blind_A: As a boy he knelt in the wet alley beside his parents' bodies, the pearls scattered on the pavement, and the loss carved him into something else.
- 0.12 (p16.6, open, peak ' the') blind_B: As a boy he watches a mugger shoot his parents dead in an alley, and the trauma of that night defines the rest of his life.

Near miss (origin, wound): The clock's origin was a Swiss workshop, and the shopkeeper wound it every morning before opening.

### 2. The secret identity — L13·8608+

*trope; present in 30 of 32 Batman works; family of 5. Paper's note: A masked vigilante, unmasking, a double life. Also fires for Zorro, the Scarlet Pimpernel and Monte Cristo.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 13 of 36 | yes |
| 2. Sign regime | Top 50 firings: 50 gate-open, 0 echo, 0 other; echo share 0% (0 of the top 8; first echo at rank none in the top 50). Work scores (mean of the top 3 sentences) drawn from echo firings: 3% in Batman works, 80% in ordinary films; 1 of 32 Batman works and 57 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' identity' ×15 (15/0), 'ante' ×5 (5/0), ' masked' ×4 (4/0), ' Spider' ×3 (3/0), ' spider' ×3 (3/0), '4' ×2 (2/0) | yes |
| 3. Negative pole | L13·8608-, max 13.25. The opposite pole fires on the opening quote of a nickname (Sebastian "Seb", Donald "Donny Boy", James "Jim"): an alias declared in the open, against an identity concealed behind a mask. | – |
| 4. Blind maximum | Best of 3: 2.19, against 3.27 for the corpus maximum in context and 4.09 for the corpus top 8 read standalone (corpus p99.6; peak on ' masked', open). The weakest of the corpus top 8 is 2.47 in context, 1.20 standalone; neutral sentences max 0.18. | no |
| 5. Near miss | 0.10 (corpus p5.8; 3% of the in-context maximum, 4% of the best blind maximum; peak on ' was') | yes |
| **Verdict** | Marginal. A lexical cluster for identity, masked, mask, vigilante and Spider-Man (all gate-open, no echo). The blind sentences reach the corpus top 1% on the word masked and the near miss with secret and identity but no mask stays silent (0.10), so the unit does carry the sense; but the best blind sentence (2.19) falls just short of the weakest of the top 8 (2.47) and of the maximum (3.27). | **fails** |

Suggested label: **The secret identity, as a lexical cluster (identity, masked, vigilante); keep with that note**

Strongest corpus sentences (activation, regime, peak token):

- 3.27 open 'masked' · *The Batman*: The Riddler gives himself up and is unmasked as forensic accountant Edward Nashton; he is incarcerated in Arkham State Hospital, where he tells Batman he took inspiratio…
- 2.88 open '4' · *Knightfall*: The next storyline, in Detective Comics #654-656 (December 1992 - February 1993), involves a young military student usurping power in Gotham's underworld and assaulting…
- 2.84 open ' masked' · *Batman v Superman*: Bruce is now a billionaire who has operated in Gotham City as the masked vigilante Batman for twenty years.
- 2.67 open ' identity' · *Batman v Superman*: Clark Kent, Superman's civilian identity, seeks to condemn Batman's brutal methods in Daily Planet articles.
- 2.58 open 'mask' · *Hush*: In the ensuing confrontation, Elliot is shot by the reborn Harvey Dent and plummets into the water, with Batman never having a chance to unmask him (it is only assumed i…
- 2.53 open ' identity' · *Batman '89*: Conflicted with his love for her, Wayne visits her apartment intending to reveal his secret identity, only for the Joker to interrupt the meeting.
- 2.52 open ' masked' · *Zorro*: Don Diego quickly adopts the guise of El Zorro ("The Fox"), a masked outlaw dressed entirely in black, who becomes the defender of the common people and a champion for j…
- 2.47 open ' Spider' · *Spider-Man*: Norman notices Peter's injury and deduces that he is Spider-Man.

Negative pole L13·8608-, strongest sentences:

- 13.25 ' "' · *La La Land*: While stuck in Los Angeles traffic ("Another Day of Sun"), jazz pianist Sebastian "Seb" Wilder has a moment of road rage directed at aspiring actress Amelia "Mia" Dolan.
- 12.81 ' "' · *Rocky*: Rocky builds a romantic relationship with Adrianna "Adrian" Pennino, a shy woman who is working part-time at the J&M Tropical Fish pet store.
- 11.94 ' "' · *The Long Halloween*: Falcone responds by hiring the Sullivan family, a gang of Irish hitmen led by Donald "Donny Boy" Sullivan, to destroy Dent's home with a bomb, but he and his wife Gilda…
- 11.38 ' "' · *Iron Man*: Tony Stark, who has inherited the defense contractor Stark Industries from his late father Howard Stark, tours in war-torn Afghanistan with his best friend and military…
- 11.00 ' "' · *Hush*: Batman attempts to kill the Joker but is then dissuaded by former police commissioner James "Jim" Gordon, who intervenes by superficially wounding Batman with a gunshot.
- 10.75 ' "' · *Arkham Asylum (comic)*: He blocked out the memory, only to have it return after an inmate, Martin "Mad Dog" Hawkins, raped and murdered Arkham's wife and daughter.
- 10.62 ' "' · *Back to the Future*: That night, Marty meets his eccentric scientist friend, Emmett "Doc" Brown, in the Twin Pines mall parking lot.
- 10.50 ' "' · *The Godfather*: Vito had earlier refused to offer protection to drug baron Virgil "The Turk" Sollozzo for dealing narcotics in the city.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 4.09 (p100.0, open, peak ' identity') · *Batman v Superman*: Clark Kent, Superman's civilian identity, seeks to condemn Batman's brutal methods in Daily Planet articles.
- 3.16 (p100.0, open, peak '4') · *Knightfall*: The next storyline, in Detective Comics #654-656 (December 1992 - February 1993), involves a young military student usurping power in Gotham's underworld and assaulting…
- 2.80 (p99.9, open, peak 'ante') · *Batman v Superman*: Bruce is now a billionaire who has operated in Gotham City as the masked vigilante Batman for twenty years.
- 2.61 (p99.9, open, peak 'mask') · *Hush*: In the ensuing confrontation, Elliot is shot by the reborn Harvey Dent and plummets into the water, with Batman never having a chance to unmask him (it is only assumed i…
- 1.65 (p99.3, open, peak 'masked') · *The Batman*: The Riddler gives himself up and is unmasked as forensic accountant Edward Nashton; he is incarcerated in Arkham State Hospital, where he tells Batman he took inspiratio…
- 1.47 (p99.0, open, peak ' identity') · *Batman '89*: Conflicted with his love for her, Wayne visits her apartment intending to reveal his secret identity, only for the Joker to interrupt the meeting.
- 1.33 (p98.6, open, peak ' masked') · *Zorro*: Don Diego quickly adopts the guise of El Zorro ("The Fox"), a masked outlaw dressed entirely in black, who becomes the defender of the common people and a champion for j…
- 1.20 (p98.3, open, peak ' Spider') · *Spider-Man*: Norman notices Peter's injury and deduces that he is Spider-Man.

Blind maximum attempts:

- 2.19 (p99.6, open, peak ' masked') blind_C: By day a mild-mannered civilian known under his real name, by night a masked hero whose true identity remains a closely guarded secret from everyone he loves.
- 2.00 (p99.5, open, peak 'ante') blind_B: By day he poses as an idle playboy, concealing from everyone, including his closest friends, that he is the masked vigilante who prowls the city at night.
- 0.17 (p34.8, echo, peak ' smiled') blind_A: By day he smiled for the cameras as a harmless playboy, but under the tailored jacket he hid the bruises of the masked figure the city feared.

Near miss (secret, identity): The committee kept the winning recipe secret, though the identity of the bakery was printed on every box.

### 3. Heir to a dead father's fortune — L10·5634+

*trope; present in 28 of 32 Batman works; family of 1. Paper's note: Inherited wealth and a late father's company. Peaks on Tony Stark inheriting Stark Industries.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 10 of 36 | yes |
| 2. Sign regime | Top 50 firings: 50 gate-open, 0 echo, 0 other; echo share 0% (0 of the top 8; first echo at rank none in the top 50). Work scores (mean of the top 3 sentences) drawn from echo firings: 22% in Batman works, 93% in ordinary films; 8 of 32 Batman works and 67 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Wayne' ×31 (31/0), 'son' ×3 (3/0), ' Penguin' ×3 (3/0), '.' ×2 (2/0), ' Todd' ×2 (2/0), ' Planet' ×2 (2/0) | yes |
| 3. Negative pole | L10·5634-, max 2.45. The opposite pole peaks weakly (2.5) on words of change (transformation, improved, evident); the positive pole is the surname Wayne, 31 of the top 50. | – |
| 4. Blind maximum | Best of 3: 0.27, against 1.45 for the corpus maximum in context and 1.40 for the corpus top 8 read standalone (corpus p90.4; peak on ' money', echo). The weakest of the corpus top 8 is 1.19 in context, 0.95 standalone; neutral sentences max 0.25. | no |
| 5. Near miss | 0.12 (corpus p34.3; 8% of the in-context maximum, 45% of the best blind maximum; peak on ' that') | yes |
| **Verdict** | Fails. A weak unit (maximum 1.45) that fires on the token Wayne, with Stark once; the inheritance reading came from the single Iron Man sentence in the top 8. A name-free heir inheriting his late father's estate scores 0.27, no more than the neutral sentences. | **fails** |

Suggested label: **The surname Wayne (and Stark)**

Strongest corpus sentences (activation, regime, peak token):

- 1.45 open ' Wayne' · *Lego Batman*: The following day, Bruce Wayne attends a gala celebrating both the retirement of Commissioner Gordon and the ascension of his daughter, Barbara Gordon, to replace him.
- 1.37 open ' Stark' · *Iron Man*: Tony Stark, who has inherited the defense contractor Stark Industries from his late father Howard Stark, tours in war-torn Afghanistan with his best friend and military…
- 1.34 open ' Wayne' · *Dark Knight Returns*: Over a decade after the United Nations outlawed superheroes and the Joker murdered Batman's protégé, Robin (Jason Todd), a 55-year-old Bruce Wayne battles alcoholism in…
- 1.30 open ' Wayne' · *Knightfall*: Within the regular series, the buildup to "Knightfall" begins with a six-issue run in Batman #484-489 (September 1992 - February 1993), in which Batman (at the onset of…
- 1.23 open ' Wayne' · *Batman Forever*: Edward Nygma, an eccentric and egotistical researcher at Wayne Enterprises, approaches his employer, Bruce Wayne, to present an invention that can beam television signal…
- 1.22 open ' Wayne' · *Knightfall*: As Bruce Wayne, he contacts holistic therapist Shondra Kinsolving for treatment.
- 1.20 open ' Wayne' · *The Batman*: Reclusive billionaire Bruce Wayne, who has operated for two years as the vigilante Batman, investigates the murder alongside Lieutenant Jim Gordon, much to the disdain o…
- 1.19 open ' Wayne' · *Batman v Superman*: In a flashback to his childhood, Bruce Wayne flees his parents' funeral.

Negative pole L10·5634-, strongest sentences:

- 2.45 ' transformation' · *Groundhog Day*: Impressed by his apparent overnight transformation, Rita successfully bids for him at a charity bachelor auction.
- 2.03 ' improved' · *Daredevil*: Urich prepares to publish an article revealing Daredevil's identity, but decides against it at the last minute, finally accepting that Daredevil's efforts have improved…
- 1.71 ' evident' · *Sherlock Holmes*: The first pill produces no evident effect, but the second kills the terrier.
- 1.70 ' bring' · *Batman Returns*: Batman's efforts to quell the violence bring him into conflict with Catwoman, while in their civilian lives Selina and Bruce begin a romance.
- 1.65 ' to' · *Arkham Asylum (game)*: Afterward, Joker injects Ivy with Titan, enhancing her powers, and she begins to ravage Arkham Island with giant mutant plants.
- 1.41 ' starts' · *The Shining*: A month passes, and Danny starts having frightening visions, including those of the murdered Grady sisters.
- 1.40 ',' · *The Devil Wears Prada*: Noticing Andy's increased commitment to the job, Miranda delegates more important tasks to her.
- 1.40 ',' · *Groundhog Day*: Contrary to his prediction, the blizzard strikes the area, preventing all travel, and although he desperately searches for a way to leave, he is forced to spend the nigh…

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 1.40 (p100.0, open, peak ' Wayne') · *Knightfall*: Within the regular series, the buildup to "Knightfall" begins with a six-issue run in Batman #484-489 (September 1992 - February 1993), in which Batman (at the onset of…
- 1.38 (p100.0, open, peak ' Stark') · *Iron Man*: Tony Stark, who has inherited the defense contractor Stark Industries from his late father Howard Stark, tours in war-torn Afghanistan with his best friend and military…
- 1.38 (p100.0, open, peak ' Wayne') · *Knightfall*: As Bruce Wayne, he contacts holistic therapist Shondra Kinsolving for treatment.
- 1.37 (p99.9, open, peak ' Wayne') · *Dark Knight Returns*: Over a decade after the United Nations outlawed superheroes and the Joker murdered Batman's protégé, Robin (Jason Todd), a 55-year-old Bruce Wayne battles alcoholism in…
- 1.33 (p99.9, open, peak ' Wayne') · *Lego Batman*: The following day, Bruce Wayne attends a gala celebrating both the retirement of Commissioner Gordon and the ascension of his daughter, Barbara Gordon, to replace him.
- 1.22 (p99.8, open, peak ' Wayne') · *Batman Forever*: Edward Nygma, an eccentric and egotistical researcher at Wayne Enterprises, approaches his employer, Bruce Wayne, to present an invention that can beam television signal…
- 1.20 (p99.8, open, peak ' Wayne') · *Batman v Superman*: In a flashback to his childhood, Bruce Wayne flees his parents' funeral.
- 0.95 (p99.4, open, peak ' Wayne') · *The Batman*: Reclusive billionaire Bruce Wayne, who has operated for two years as the vigilante Batman, investigates the murder alongside Lieutenant Jim Gordon, much to the disdain o…

Blind maximum attempts:

- 0.27 (p90.4, echo, peak ' money') blind_A: When the will was read, the grieving son inherited his late father's entire estate, the manor, the company, and more money than he could ever spend.
- 0.13 (p35.7, echo, peak ' the') blind_B: After his father's death, the young man inherits the family's vast industrial fortune, the mansion, and a controlling stake in the corporation that bears his name.
- 0.12 (p30.8, echo, peak '.') blind_C: After his father's death the young heir inherited the entire family fortune, the mansion, the company and a vast estate bequeathed in the late patriarch's will.

Near miss (heir, dead, father, fortune): The dead language course covered the word heir, and her father joked that fortune favoured the diligent student.

### 4. The billionaire vigilante — L4·1072+

*trope; present in 23 of 32 Batman works; family of 7. Paper's note: A reclusive rich man who fights crime outside the law.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 4 of 36 | yes |
| 2. Sign regime | Top 50 firings: 50 gate-open, 0 echo, 0 other; echo share 0% (0 of the top 8; first echo at rank none in the top 50). Work scores (mean of the top 3 sentences) drawn from echo firings: 43% in Batman works, 93% in ordinary films; 14 of 32 Batman works and 65 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Gordon' ×49 (49/0), 'ideon' ×1 (1/0) | yes |
| 3. Negative pole | L4·1072-, max 0.55. The opposite pole is sub-word noise at a thirtieth of the strength (-son in Grayson, cords, min); the channel is one-sided. | – |
| 4. Blind maximum | Best of 3: 0.07, against 18.62 for the corpus maximum in context and 18.88 for the corpus top 8 read standalone (corpus p87.8; peak on ' dispens', echo). The weakest of the corpus top 8 is 17.62 in context, 16.25 standalone; neutral sentences max 0.04. | no |
| 5. Near miss | 0.92 (corpus p97.6; 5% of the in-context maximum, 1397% of the best blind maximum; peak on 'ardin') | no |
| **Verdict** | Fails, and the label is simply wrong. 49 of the top 50 peak on the token Gordon; the label was read off Bruce Wayne and Batman appearing in the same sentences. The best name-free billionaire-vigilante sentence scores 0.07, 0.4% of the maximum. This, not L11·12069-, is the paper's real Gordon detector. | **fails** |

Suggested label: **The name Gordon**

Strongest corpus sentences (activation, regime, peak token):

- 18.62 open ' Gordon' · *The Batman*: Reclusive billionaire Bruce Wayne, who has operated for two years as the vigilante Batman, investigates the murder alongside Lieutenant Jim Gordon, much to the disdain o…
- 18.50 open ' Gordon' · *The Dark Knight*: The vigilante Batman, district attorney Harvey Dent, and police lieutenant Jim Gordon ally to eliminate Gotham's organized crime.
- 18.00 open ' Gordon' · *Lego Batman*: The following day, Bruce Wayne attends a gala celebrating both the retirement of Commissioner Gordon and the ascension of his daughter, Barbara Gordon, to replace him.
- 17.88 open ' Gordon' · *Dark Knight Rises*: Police commissioner James Gordon has kept Dent's killing spree a secret and allowed the blame for his crimes to fall on Batman.
- 17.75 open ' Gordon' · *The Long Halloween*: Batman meets with district attorney Harvey Dent and police captain Jim Gordon.
- 17.62 open ' Gordon' · *Return of the Joker*: They buried the Joker beneath the asylum, and only Barbara's father and predecessor, James Gordon, knew of the incident.
- 17.62 open ' Gordon' · *The Killing Joke*: The Joker shoots and paralyzes Barbara Gordon and kidnaps her father, Commissioner James "Jim" Gordon.
- 17.62 open ' Gordon' · *Arkham Knight*: He is supported by his allies, Robin (Matthew Mercer), Nightwing (Scott Porter), Catwoman (Grey DeLisle), Barbara Gordon (Ashley Greene)—who assists Batman covertly as t…

Negative pole L4·1072-, strongest sentences:

- 0.55 'en' · *Arkham Knight*: Other villains include the pyromaniac Firefly (Crispin Freeman), the beastly Man-Bat (Loren Lester), the veteran assassin Deathstroke (Mark Rolston), the serial killer P…
- 0.48 'son' · *Lego Batman*: While smitten by Barbara, Bruce absent-mindedly agrees to adopt enthusiastic orphan Dick Grayson.
- 0.48 ' cords' · *Gravity*: They try to grab onto the ISS, but only make tenuous connection to the Soyuz's parachute cords, which are not strong enough for them both.
- 0.48 'son' · *Hush*: Dick Grayson returns to Gotham City for Elliot's funeral.
- 0.47 'min' · *Dracula*: Dracula communicates with Seward's patient, Renfield, an insane man who eats vermin to absorb their life force.
- 0.47 'son' · *Dark Knight Returns*: Having long abandoned his mantle, he remains deeply estranged from Jason's predecessor, Dick Grayson.
- 0.47 'one' · *The Godfather*: Vito is head of the Corleone mafia family and runs it with his eldest son Santino a.k.a.
- 0.46 'son' · *Batman Forever*: Dick Grayson, the youngest member of the Flying Graysons family of acrobats, prevents the bomb from killing anyone by throwing it into a river with the help of his famil…

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 18.88 (p100.0, open, peak ' Gordon') · *Dark Knight Rises*: Police commissioner James Gordon has kept Dent's killing spree a secret and allowed the blame for his crimes to fall on Batman.
- 18.75 (p100.0, open, peak ' Gordon') · *Return of the Joker*: They buried the Joker beneath the asylum, and only Barbara's father and predecessor, James Gordon, knew of the incident.
- 18.50 (p99.9, open, peak ' Gordon') · *The Dark Knight*: The vigilante Batman, district attorney Harvey Dent, and police lieutenant Jim Gordon ally to eliminate Gotham's organized crime.
- 18.38 (p99.9, open, peak ' Gordon') · *The Batman*: Reclusive billionaire Bruce Wayne, who has operated for two years as the vigilante Batman, investigates the murder alongside Lieutenant Jim Gordon, much to the disdain o…
- 18.12 (p99.9, open, peak ' Gordon') · *Lego Batman*: The following day, Bruce Wayne attends a gala celebrating both the retirement of Commissioner Gordon and the ascension of his daughter, Barbara Gordon, to replace him.
- 17.75 (p99.9, open, peak ' Gordon') · *Arkham Knight*: He is supported by his allies, Robin (Matthew Mercer), Nightwing (Scott Porter), Catwoman (Grey DeLisle), Barbara Gordon (Ashley Greene)—who assists Batman covertly as t…
- 16.75 (p99.7, open, peak ' Gordon') · *The Long Halloween*: Batman meets with district attorney Harvey Dent and police captain Jim Gordon.
- 16.25 (p99.6, open, peak ' Gordon') · *The Killing Joke*: The Joker shoots and paralyzes Barbara Gordon and kidnaps her father, Commissioner James "Jim" Gordon.

Blind maximum attempts:

- 0.07 (p87.8, echo, peak ' dispens') blind_C: The billionaire playboy spends his vast fortune on gadgets and armor, then prowls the city at night as a masked vigilante dispensing justice outside the law.
- 0.04 (p71.3, echo, peak ' man') blind_A: The richest man in the city spent his fortune on armor and gadgets, then prowled the rooftops at night beating criminals the law could not touch.
- 0.03 (p57.5, echo, peak ' billionaire') blind_B: A billionaire industrialist spends his inherited wealth on armor, vehicles and gadgets, and uses them to fight crime at night as a masked vigilante outside the law.

Near miss (billionaire, vigilante): The billionaire's yacht, named Vigilante after a racehorse, won the regatta off Sardinia by two minutes.

### 5. The Batcave and bat gear — L11·9843+

*object; present in 29 of 32 Batman works; family of 41. Paper's note: Hidden headquarters, the Bat-Signal, bat-themed vehicles and devices.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 11 of 36 | yes |
| 2. Sign regime | Top 50 firings: 50 gate-open, 0 echo, 0 other; echo share 0% (0 of the top 8; first echo at rank none in the top 50). Work scores (mean of the top 3 sentences) drawn from echo firings: 0% in Batman works, 47% in ordinary films; 0 of 32 Batman works and 33 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Bat' ×43 (43/0), 'ats' ×3 (3/0), 'Bat' ×3 (3/0), ' bat' ×1 (1/0) | yes |
| 3. Negative pole | L11·9843-, max 1.03. The opposite pole is weak (1.0) and generic (City, company, studio): no distinction to draw. | – |
| 4. Blind maximum | Best of 3: 2.05, against 5.66 for the corpus maximum in context and 5.62 for the corpus top 8 read standalone (corpus p97.8; peak on ' bat', open). The weakest of the corpus top 8 is 5.06 in context, 1.84 standalone; neutral sentences max 0.28. | no |
| 5. Near miss | 0.18 (corpus p59.7; 3% of the in-context maximum, 9% of the best blind maximum; peak on 'l') | yes |
| **Verdict** | Fails as a concept, right as a word. 43 of the top 50 peak on the Bat- prefix of Batcave, Batsuit, Batmobile and the Bat. The best name-free description of the cave and its gear reaches 36% of the maximum, at bat-shaped; the near miss with cave, bats and gear but no Bat- prefix is silent (0.18). | **fails** |

Suggested label: **The Bat- prefix in compounds (Batcave, Batsuit, Batmobile, the Bat)**

Strongest corpus sentences (activation, regime, peak token):

- 5.66 open ' Bat' · *Arkham Knight*: Oracle has set up her communications headquarters in the Gotham clock tower, which also houses a makeshift Batcave.
- 5.62 open ' Bat' · *Dark Knight Rises*: With no way to stop the detonation, Batman reveals his identity to Gordon, and uses his aerial craft, the Bat, to haul the bomb over the bay, where it safely explodes.
- 5.50 open ' Bat' · *Year One*: Batman uses a signal device to attract a swarm of bats from the Batcave, concealing his escape.
- 5.44 open ' Bat' · *Batman R.I.P.*: Back in present-day Gotham, Le Bossu tells his henchmen that Batman has not been seen in months and that they are free to commit crimes unmolested, when suddenly a Bat-s…
- 5.16 open ' Bat' · *Lego Batman*: Batman initially refuses, so Alfred allows Dick to enter the Batcave.
- 5.12 open 'ats' · *Return of the Joker*: Subduing the Jokerz with Ace's help, Terry confronts Tim, who subdues Terry by disabling his Batsuit and then transforms into the Joker as Bruce watches in horror from t…
- 5.12 open ' Bat' · *White Knight*: Napier, Batman and the GTO organize an assault on the Neo Joker's freeze ray, piloting a fleet of Batmobiles taken from Wayne Manor.
- 5.06 open ' Bat' · *Batman R.I.P.*: He lets Jezebel so deep into his life that he even introduces her to the Batcave.

Negative pole L11·9843-, strongest sentences:

- 1.03 ' City' · *Arkham City*: Arkham City features a large ensemble cast of characters from the history of Batman comics.
- 0.98 ' his' · *The Social Network*: Returning to his dorm, Zuckerberg writes an insulting post about her on his LiveJournal blog.
- 0.89 'xs' · *Groundhog Day*: Alongside his new producer Rita Hanson and cameraman Larry, Phil travels to Punxsutawney for his annual coverage of the Groundhog Day festivities.
- 0.76 ' company' · *The Social Network*: Saverin deposits $18,000 into a new account as Zuckerberg relocates the company to Palo Alto on Parker's advice; Saverin remains in New York to work on business developm…
- 0.76 ' New' · *Daredevil*: Ben Urich, a New York Post reporter who chronicles Daredevil's exploits, attracts attention for a series of articles on "The Kingpin", a shadowy underworld figure who al…
- 0.60 ' studio' · *Singin' in the Rain*: A clause in Lina's contract allows her to sue the studio for negative press, so R.
- 0.59 ' Plan' · *The Martian*: In 2035, the crew of the Ares III mission to Mars is exploring Acidalia Planitia on Martian solar day (sol) 18 of their 31-sol expedition.
- 0.56 ' developers' · *Arkham Asylum (game)*: The Mad Hatter was almost included in the game, but the developers removed him.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 5.62 (p99.9, open, peak ' Bat') · *Batman R.I.P.*: Back in present-day Gotham, Le Bossu tells his henchmen that Batman has not been seen in months and that they are free to commit crimes unmolested, when suddenly a Bat-s…
- 5.03 (p99.8, open, peak ' Bat') · *Arkham Knight*: Oracle has set up her communications headquarters in the Gotham clock tower, which also houses a makeshift Batcave.
- 4.56 (p99.4, open, peak ' Bat') · *Dark Knight Rises*: With no way to stop the detonation, Batman reveals his identity to Gordon, and uses his aerial craft, the Bat, to haul the bomb over the bay, where it safely explodes.
- 4.50 (p99.4, open, peak ' Bat') · *Return of the Joker*: Subduing the Jokerz with Ace's help, Terry confronts Tim, who subdues Terry by disabling his Batsuit and then transforms into the Joker as Bruce watches in horror from t…
- 4.09 (p99.0, open, peak ' Bat') · *White Knight*: Napier, Batman and the GTO organize an assault on the Neo Joker's freeze ray, piloting a fleet of Batmobiles taken from Wayne Manor.
- 3.70 (p98.6, open, peak ' Bat') · *Year One*: Batman uses a signal device to attract a swarm of bats from the Batcave, concealing his escape.
- 3.56 (p98.5, open, peak ' Bat') · *Lego Batman*: Batman initially refuses, so Alfred allows Dick to enter the Batcave.
- 1.84 (p97.6, open, peak ' Bat') · *Batman R.I.P.*: He lets Jezebel so deep into his life that he even introduces her to the Batcave.

Blind maximum attempts:

- 2.05 (p97.8, open, peak ' bat') blind_C: Beneath the mansion lies his secret cave headquarters, stocked with the armored car, the grappling hook, the utility belt, the cowl and the bat-shaped throwing blades.
- 0.40 (p89.2, open, peak ' bat') blind_B: Beneath the mansion lies a vast cavern headquarters where he stores his armored car, his grappling hooks, his bat-shaped throwing blades and his cowled suit.
- 0.16 (p54.1, open, peak 'ene') blind_A: Beneath the manor, in a dripping cavern lit by banks of monitors, hung the cowl, the utility belt, the grapnel gun, and the armored black car.

Near miss (bat, cave, gear): The spelunkers packed climbing gear and headlamps before descending into the limestone cave, where thousands of bats roosted.

### 6. Bats — L8·7130+

*object; present in 24 of 32 Batman works; family of 7. Paper's note: Literal bats: a swarm in a cave, performers dressed as bats, the costumed killer in the 1926 film The Bat.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 8 of 36 | yes |
| 2. Sign regime | Top 50 firings: 50 gate-open, 0 echo, 0 other; echo share 0% (0 of the top 8; first echo at rank none in the top 50). Work scores (mean of the top 3 sentences) drawn from echo firings: 1% in Batman works, 42% in ordinary films; 0 of 32 Batman works and 31 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Bat' ×24 (24/0), ' bat' ×13 (13/0), ' bats' ×6 (6/0), 'Bat' ×3 (3/0), '-S' ×2 (2/0), 'ats' ×1 (1/0) | yes |
| 3. Negative pole | L8·7130-, max 1.12. The opposite pole peaks on other named animals (clownfish, tiger shark, Tyrannosaurus): the channel runs from bats to other creatures. | – |
| 4. Blind maximum | Best of 3: 12.88, against 14.19 for the corpus maximum in context and 13.50 for the corpus top 8 read standalone (corpus p99.9; peak on ' bats', open). The weakest of the corpus top 8 is 11.19 in context, 11.12 standalone; neutral sentences max 0.27. | would make the top 8 |
| 5. Near miss | 2.20 (corpus p97.0; 16% of the in-context maximum, 17% of the best blind maximum; peak on ' bat') | yes |
| **Verdict** | Survives on the lenient bar and is the cleanest unit of the 31: the word bat in its animal sense (and people dressed as bats). The best blind sentence reaches 91% of the in-context maximum and 95% of the standalone one, on the word bats; cricket bats score a sixth of that. Half the top 50 are capitalised Bat (the Bat, Bat Clan, Bat-Signal), so the label should say it is a word, not a trope. | **lenient** |

Suggested label: **Bats, as the word bat(s) in its animal sense; keep**

Strongest corpus sentences (activation, regime, peak token):

- 14.19 open ' bats' · *Batman v Superman*: He falls into a cave, where a circling vortex of bats elevates him back to the surface.
- 13.12 open ' bats' · *Batman Begins*: One evening, Bruce visits the opera with his parents; the performers masquerading as bats upset him, and he asks to leave.
- 12.38 open ' bat' · *Arkham Asylum (comic)*: After believing to have seen the creature himself (a bat), Arkham killed her to end her suffering.
- 12.00 open ' bat' · *Batman Ninja*: Batman learns the Bat Clan helped Nightwing, Red Hood, Red Robin and Robin upon their arrival and the clan has followed a prophecy of a foreign bat ninja restoring order…
- 11.81 open ' bats' · *Batman Begins*: In Gotham City, young Bruce Wayne falls down a well and develops a fear of bats.
- 11.62 open ' bat' · *The Bat (1926)*: The Bat, a figure dressed as a bat, murders Gideon and steals the emeralds.
- 11.38 open ' bat' · *Year One*: A bat crashes through a window and settles on the bust, inspiring Bruce to save Gotham as Batman.
- 11.19 open ' bats' · *Year One*: Batman uses a signal device to attract a swarm of bats from the Batcave, concealing his escape.

Negative pole L8·7130-, strongest sentences:

- 1.12 'fish' · *Finding Nemo*: Marlin and Coral, a clownfish couple who are waiting for their eggs to hatch, live on the edge of the Great Barrier Reef.
- 0.96 ' tiger' · *Jaws*: When local fishermen catch a tiger shark, Vaughn declares the beaches safe.
- 0.94 'aurus' · *Jurassic Park*: The Tyrannosaurus devours Gennaro and injures Malcolm while Grant, Lex, and Tim escape.
- 0.92 'aurus' · *Jurassic Park*: The survivors are cornered by the two raptors in the visitor center, but the Tyrannosaurus appears and fights the raptors, allowing the group to flee.
- 0.88 'aurus' · *Jurassic Park*: Sattler helps the game warden Robert Muldoon search for survivors; they find Malcolm just before the Tyrannosaurus returns and chases them away.
- 0.70 'less' · *Up*: Along the way, they encounter a giant flightless bird that Russell names Kevin (though the bird is later revealed to be female) and Dug, a Golden Retriever whose collar…
- 0.70 'etic' · *E.T. the Extra-Terrestrial*: When the children ask about his origins, E.T. displays telekinetic abilities by levitating several balls to represent his planetary system and later revives a dead chrys…
- 0.66 ' ninja' · *Batman Ninja*: As Batman is surrounded by the Joker's minions, he is whisked away by ninja led by Eian of the Bat Clan of Hida.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 13.50 (p100.0, open, peak ' bat') · *Year One*: A bat crashes through a window and settles on the bust, inspiring Bruce to save Gotham as Batman.
- 13.25 (p100.0, open, peak ' bats') · *Batman v Superman*: He falls into a cave, where a circling vortex of bats elevates him back to the surface.
- 12.31 (p99.9, open, peak ' bats') · *Batman Begins*: One evening, Bruce visits the opera with his parents; the performers masquerading as bats upset him, and he asks to leave.
- 12.31 (p99.9, open, peak ' bat') · *The Bat (1926)*: The Bat, a figure dressed as a bat, murders Gideon and steals the emeralds.
- 12.25 (p99.9, open, peak ' bat') · *Batman Ninja*: Batman learns the Bat Clan helped Nightwing, Red Hood, Red Robin and Robin upon their arrival and the clan has followed a prophecy of a foreign bat ninja restoring order…
- 11.81 (p99.9, open, peak ' bats') · *Batman Begins*: In Gotham City, young Bruce Wayne falls down a well and develops a fear of bats.
- 11.38 (p99.8, open, peak ' bat') · *Arkham Asylum (comic)*: After believing to have seen the creature himself (a bat), Arkham killed her to end her suffering.
- 11.12 (p99.8, open, peak ' bats') · *Year One*: Batman uses a signal device to attract a swarm of bats from the Batcave, concealing his escape.

Blind maximum attempts:

- 12.88 (p99.9, open, peak ' bats') blind_C: A colony of bats, nocturnal winged mammals that roost upside down in dark caves, burst screeching out of the cavern in a flapping swarm.
- 11.25 (p99.8, open, peak ' bats') blind_B: A swarm of bats bursts from the cave and engulfs him, their leathery wings and shrieks filling the darkness as the animals fly out into the night.
- 8.19 (p99.3, open, peak ' cave') blind_A: A thousand bats burst shrieking from the cave mouth at dusk, leathery wings beating the air, a black cloud of winged mammals swallowing the sky.

Near miss (bats, bat): The sporting goods shop sold cricket bats and leather balls, and the owner oiled each bat by hand.

### 7. Gotham City — L6·10402+

*place; present in 30 of 32 Batman works; family of 49. Paper's note: The city itself as a setting: its mayor, its mob bosses, its murders.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 6 of 36 | yes |
| 2. Sign regime | Top 50 firings: 50 gate-open, 0 echo, 0 other; echo share 0% (0 of the top 8; first echo at rank none in the top 50). Work scores (mean of the top 3 sentences) drawn from echo firings: 15% in Batman works, 79% in ordinary films; 4 of 32 Batman works and 56 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' City' ×44 (44/0), ' Gotham' ×4 (4/0), 'm' ×1 (1/0), 'av' ×1 (1/0) | yes |
| 3. Negative pole | L6·10402-, max 1.41. The opposite pole peaks on pieces of foreign names (shikigami, Fitzwilliam, Zihuatanejo): sub-word noise. | – |
| 4. Blind maximum | Best of 3: 0.37, against 3.03 for the corpus maximum in context and 3.03 for the corpus top 8 read standalone (corpus p95.5; peak on ' garg', echo). The weakest of the corpus top 8 is 2.22 in context, 2.22 standalone; neutral sentences max 0.06. | no |
| 5. Near miss | 2.14 (corpus p99.7; 71% of the in-context maximum, 577% of the best blind maximum; peak on ' Gotham') | no |
| **Verdict** | Fails. 44 of the top 50 peak on City immediately after Gotham. The Nottinghamshire village of Gotham scores 70% of the maximum; the best name-free description of a crime-ridden metropolis scores 12%. | **fails** |

Suggested label: **The token City after Gotham**

Strongest corpus sentences (activation, regime, peak token):

- 3.03 open ' City' · *The Batman*: On Halloween, Gotham City mayor Don Mitchell Jr. is murdered by a masked serial killer, the Riddler.
- 2.98 open ' City' · *The Long Halloween*: At a wedding in June, Gotham City mob boss Carmine Falcone tries to pressure Bruce Wayne to help launder his money, but Bruce refuses.
- 2.41 open ' City' · *Batman Begins*: In Gotham City, young Bruce Wayne falls down a well and develops a fear of bats.
- 2.41 open ' City' · *Batman Returns*: In Gotham City, two wealthy socialites, horrified by the birth of their malformed and feral son Oswald, abandon him in the sewers, where a colony of penguins takes him i…
- 2.41 open ' City' · *Batman Forever*: In Gotham City, Batman defuses a hostage situation orchestrated by the criminal Two-Face, formerly district attorney Harvey Dent, who escapes.
- 2.34 open ' City' · *Year One*: Billionaire Bruce Wayne returns home to Gotham City after twelve years abroad, training for a one-man war against crime.
- 2.33 open ' City' · *A Death in the Family*: While eavesdropping on a child pornography ring and awaiting police backup in Gotham City, Jason Todd (Robin) ignores Batman's orders and attacks the criminals.
- 2.22 open ' City' · *Earth One*: Bruce Wayne is the eight-year-old son of Dr. Thomas Wayne, a mayoral candidate for Gotham City, and Martha Wayne (née Arkham).

Negative pole L6·10402-, strongest sentences:

- 1.41 'ami' · *Spirited Away*: Chihiro sees paper shikigami spirits attacking a dragon and recognizes the dragon as a metamorphosed Haku.
- 1.39 'iam' · *Pride & Prejudice*: Unaware that Jane is Elizabeth's sister, Fitzwilliam mentions that Darcy recently untangled Bingley from an imprudent match with an "unsuitable" family.
- 1.39 'ami' · *Spirited Away*: A shikigami that stowed away on her back shapeshifts into Yubaba's twin sister Zeniba, who turns Yubaba's son, Boh, into a mouse and creates a false copy of him.
- 1.20 'GER' · *Arkham City*: This facility is subsequently placed in the care of psychiatrist Hugo Strange—who is secretly manipulating Sharp—and monitored by a rogue private military firm, TYGER Se…
- 1.14 'asc' · *Babe*: There, he is taken in by Border Collie Fly, her irascible husband Rex and their puppies and befriends a duck named Ferdinand, who wakes people by crowing like a rooster…
- 1.08 'ops' · *Jurassic Park*: Most of the dinosaurs fail to appear, and Sattler stays behind to study a sick Triceratops.
- 1.05 'uat' · *The Shawshank Redemption*: He tells a skeptical Red that he dreams of living in Zihuatanejo, a town on the Mexican Pacific coast, where he can start anew.
- 1.01 'dem' · *Ghostbusters*: Supernatural activity rapidly increases across the city and the Ghostbusters become famous; they hire a fourth member, Winston Zeddemore, to cope with the growing demand.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 3.03 (p100.0, open, peak ' City') · *The Batman*: On Halloween, Gotham City mayor Don Mitchell Jr. is murdered by a masked serial killer, the Riddler.
- 2.98 (p99.9, open, peak ' City') · *The Long Halloween*: At a wedding in June, Gotham City mob boss Carmine Falcone tries to pressure Bruce Wayne to help launder his money, but Bruce refuses.
- 2.44 (p99.9, open, peak ' City') · *Batman Returns*: In Gotham City, two wealthy socialites, horrified by the birth of their malformed and feral son Oswald, abandon him in the sewers, where a colony of penguins takes him i…
- 2.41 (p99.9, open, peak ' City') · *Batman Begins*: In Gotham City, young Bruce Wayne falls down a well and develops a fear of bats.
- 2.39 (p99.9, open, peak ' City') · *Batman Forever*: In Gotham City, Batman defuses a hostage situation orchestrated by the criminal Two-Face, formerly district attorney Harvey Dent, who escapes.
- 2.33 (p99.8, open, peak ' City') · *A Death in the Family*: While eavesdropping on a child pornography ring and awaiting police backup in Gotham City, Jason Todd (Robin) ignores Batman's orders and attacks the criminals.
- 2.33 (p99.8, open, peak ' City') · *Year One*: Billionaire Bruce Wayne returns home to Gotham City after twelve years abroad, training for a one-man war against crime.
- 2.22 (p99.8, open, peak ' City') · *Earth One*: Bruce Wayne is the eight-year-old son of Dr. Thomas Wayne, a mayoral candidate for Gotham City, and Martha Wayne (née Arkham).

Blind maximum attempts:

- 0.37 (p95.5, echo, peak ' garg') blind_A: The sprawling, rain-slicked metropolis of gargoyles and smokestacks festered with mob bosses, crooked cops, and madmen, a crime-ridden city that seemed to breed its own monsters.
- 0.17 (p82.0, echo, peak 'ante') blind_C: The sprawling, crime-ridden metropolis of brooding skyscrapers, rain-soaked alleys, corrupt officials and costumed criminals is the dark city the caped vigilante swore to protect.
- 0.08 (p53.7, echo, peak 'ante') blind_B: The story is set in a sprawling, rain-soaked, crime-ridden metropolis of towering skyscrapers, corrupt officials and organized crime families, where a masked vigilante patrols the rooftops.

Near miss (Gotham, city): The village of Gotham in Nottinghamshire, miles from any city, is known for a medieval tale about its wise fools.

### 8. Commissioner Gordon — L11·12069-

*character; present in 27 of 32 Batman works; family of 1. Paper's note: Jim Gordon, the honest cop and ally.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 11 of 36 | yes |
| 2. Sign regime | Top 50 firings: 49 gate-open, 1 echo, 0 other; echo share 2% (0 of the top 8; first echo at rank 43). Work scores (mean of the top 3 sentences) drawn from echo firings: 14% in Batman works, 95% in ordinary films; 3 of 32 Batman works and 66 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Gordon' ×14 (14/0), ' Batman' ×12 (12/0), ' Bruce' ×8 (8/0), 'ina' ×4 (4/0), ' Jason' ×1 (1/0), '3' ×1 (1/0) | yes |
| 3. Negative pole | L11·12069+, max 0.47. The opposite pole is noise below 0.5. | – |
| 4. Blind maximum | Best of 3: 0.18, against 1.16 for the corpus maximum in context and 1.07 for the corpus top 8 read standalone (corpus p58.4; peak on ' summon', echo). The weakest of the corpus top 8 is 0.85 in context, 0.41 standalone; neutral sentences max 0.25. | no |
| 5. Near miss | 0.10 (corpus p17.0; 8% of the in-context maximum, 55% of the best blind maximum; peak on ' for') | no |
| **Verdict** | Fails. A weak unit (maximum 1.16) that peaks on Gordon, Batman and Bruce: a general Batman-name detector. A name-free honest commissioner lighting the rooftop signal scores 0.18, below the neutral sentences (0.25). The strong Gordon unit is L4·1072+, labelled the billionaire vigilante. | **fails** |

Suggested label: **Batman-world names (Gordon, Batman, Bruce), weak**

Strongest corpus sentences (activation, regime, peak token):

- 1.16 open ' Gordon' · *Dark Knight Returns*: Batman informs close-to-retirement Commissioner James "Jim" Gordon that Dent may be planning a larger scheme.
- 1.10 open ' Gordon' · *Hush*: Batman attempts to kill the Joker but is then dissuaded by former police commissioner James "Jim" Gordon, who intervenes by superficially wounding Batman with a gunshot.
- 1.07 open ' Batman' · *1943 serial*: The Batman / Bruce Wayne and his ward, Robin / Dick Grayson, secret government agents following the Japanese attack on Pearl Harbor, become aware of a Japanese sabotage…
- 1.04 open ' Gordon' · *Year One*: James "Jim" Gordon moves to Gotham with his pregnant wife, Barbara, after a transfer from Chicago.
- 0.96 open ' Gordon' · *The Killing Joke*: The Joker shoots and paralyzes Barbara Gordon and kidnaps her father, Commissioner James "Jim" Gordon.
- 0.94 open 'ina' · *Hush*: Later, in Gotham City, Bruce Wayne, Selina Kyle, Leslie Thompkins, and Dr. Elliot are attending the opera Pagliacci, when Harley Quinn tries to rob everyone in the theat…
- 0.87 open ' Gordon' · *The Batman*: Reclusive billionaire Bruce Wayne, who has operated for two years as the vigilante Batman, investigates the murder alongside Lieutenant Jim Gordon, much to the disdain o…
- 0.86 open ' Gordon' · *The Long Halloween*: Batman meets with district attorney Harvey Dent and police captain Jim Gordon.

Negative pole L11·12069+, strongest sentences:

- 0.47 ' Pepper' · *Iron Man*: Personal assistant Pepper Potts places the original reactor inside a small glass showcase.
- 0.44 ',' · *Count of Monte Cristo*: Dantès cuts through the sack and swims to a nearby island, where, claiming to be a shipwrecked sailor, he is rescued by Genoese smugglers.
- 0.41 ' a' · *Saving Private Ryan*: Miller leads a surviving group up the cliffs to neutralize the coastal defenses firing onto the beach, ensuring the success of the landings.
- 0.41 ' street' · *Notting Hill*: He takes her to his flat across the street so she can change.
- 0.40 'ont' · *Singin' in the Rain*: In 1927, silent film stars Don Lockwood and Lina Lamont attend the premiere of their latest film, The Royal Rascal, produced by Hollywood studio Monumental Pictures.
- 0.39 ')' · *Little Miss Sunshine*: Due to financial constraints, they go on an 800-mile (1,300 km) road trip in their yellow Volkswagen van.
- 0.39 ' an' · *Dark Knight Rises*: Bane, a former member of the League of Shadows, leads an attack on a CIA plane over Uzbekistan to abduct nuclear physicist Dr. Leonid Pavel.
- 0.38 'aba' · *Spirited Away*: Kamaji instead asks a worker named Lin to bring Chihiro to Kamaji's master Yubaba, the witch who runs the bathhouse and who transformed Chihiro's parents.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 1.07 (p99.9, open, peak ' Batman') · *1943 serial*: The Batman / Bruce Wayne and his ward, Robin / Dick Grayson, secret government agents following the Japanese attack on Pearl Harbor, become aware of a Japanese sabotage…
- 0.96 (p99.9, open, peak ' Gordon') · *Hush*: Batman attempts to kill the Joker but is then dissuaded by former police commissioner James "Jim" Gordon, who intervenes by superficially wounding Batman with a gunshot.
- 0.95 (p99.9, open, peak ' Gordon') · *The Batman*: Reclusive billionaire Bruce Wayne, who has operated for two years as the vigilante Batman, investigates the murder alongside Lieutenant Jim Gordon, much to the disdain o…
- 0.94 (p99.8, open, peak ' Gordon') · *The Long Halloween*: Batman meets with district attorney Harvey Dent and police captain Jim Gordon.
- 0.84 (p99.8, open, peak ' Gordon') · *Dark Knight Returns*: Batman informs close-to-retirement Commissioner James "Jim" Gordon that Dent may be planning a larger scheme.
- 0.83 (p99.8, open, peak ' Bruce') · *Hush*: Later, in Gotham City, Bruce Wayne, Selina Kyle, Leslie Thompkins, and Dr. Elliot are attending the opera Pagliacci, when Harley Quinn tries to rob everyone in the theat…
- 0.80 (p99.7, open, peak ' Gordon') · *The Killing Joke*: The Joker shoots and paralyzes Barbara Gordon and kidnaps her father, Commissioner James "Jim" Gordon.
- 0.41 (p97.6, open, peak ' Gordon') · *Year One*: James "Jim" Gordon moves to Gotham with his pregnant wife, Barbara, after a transfer from Chicago.

Blind maximum attempts:

- 0.18 (p58.4, echo, peak ' summon') blind_C: The honest, moustached police commissioner, a decent cop in a corrupt department, climbs to the rooftop at night to light the signal lamp summoning the caped vigilante.
- 0.07 (p8.2, echo, peak ' summoned') blind_A: The grey-moustached police commissioner, the last honest cop in the department, climbed to the precinct roof and switched on the great searchlight that summoned his masked ally.
- 0.06 (p4.9, echo, peak ' summon') blind_B: The honest, moustached police commissioner, one of the few uncorrupted officers in the department, lights the rooftop signal lamp to summon the masked vigilante.

Near miss (Commissioner, Gordon): The European Commissioner for agriculture met Gordon, a dairy farmer from Ayrshire, to discuss milk quotas.

### 9. Losing the protégé — L11·7340-

*trope; present in 27 of 32 Batman works; family of 1. Paper's note: Robin in danger, tortured or killed, and Batman's grief over it.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 11 of 36 | yes |
| 2. Sign regime | Top 50 firings: 50 gate-open, 0 echo, 0 other; echo share 0% (0 of the top 8; first echo at rank none in the top 50). Work scores (mean of the top 3 sentences) drawn from echo firings: 12% in Batman works, 98% in ordinary films; 3 of 32 Batman works and 68 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Batman' ×10 (10/0), ' Wayne' ×8 (8/0), ' Todd' ×7 (7/0), ' Gordon' ×7 (7/0), ' parents' ×6 (6/0), ' Barbara' ×3 (3/0) | yes |
| 3. Negative pole | L11·7340+, max 0.66. The opposite pole peaks on Ivy (Poison Ivy) in villain lists. | – |
| 4. Blind maximum | Best of 3: 0.20, against 2.55 for the corpus maximum in context and 2.53 for the corpus top 8 read standalone (corpus p69.6; peak on ',', echo). The weakest of the corpus top 8 is 1.55 in context, 1.60 standalone; neutral sentences max 0.19. | no |
| 5. Near miss | 0.27 (corpus p90.2; 11% of the in-context maximum, 136% of the best blind maximum; peak on ' kept') | no |
| **Verdict** | Fails. Peaks on Batman, Wayne, Todd, Gordon and parents: Batman-world names, with Jason Todd's death supplying the top five. A name-free mentor losing his sidekick scores 0.20, no more than the neutral sentences. | **fails** |

Suggested label: **Batman-world names (Batman, Wayne, Todd, Gordon), weak**

Strongest corpus sentences (activation, regime, peak token):

- 2.55 open ' Todd' · *A Death in the Family*: While eavesdropping on a child pornography ring and awaiting police backup in Gotham City, Jason Todd (Robin) ignores Batman's orders and attacks the criminals.
- 2.30 open ' Todd' · *Under the Red Hood*: He realizes his mistake when he is unable to control Joker, who captures Jason Todd, the second Robin and Batman's partner.
- 2.30 open ' Todd' · *Dark Knight Returns*: Over a decade after the United Nations outlawed superheroes and the Joker murdered Batman's protégé, Robin (Jason Todd), a 55-year-old Bruce Wayne battles alcoholism in…
- 2.02 open ' Todd' · *Knightfall*: In addition, since the events from A Death in the Family, Batman struggles with the trauma of the loss of his protégé Jason Todd.
- 1.98 open ' Todd' · *White Knight*: When the Joker's obsession with Batman culminated in Jason Todd's torture, Quinzel left him to try and help save Todd's life.
- 1.69 open ' parents' · *Batman v Superman*: In a flashback to his childhood, Bruce Wayne flees his parents' funeral.
- 1.63 open ' Gordon' · *Return of the Joker*: After fending off the attack, Terry demands information from Bruce and Police Commissioner Barbara Gordon, the former Batgirl, but neither gives him the answers he seeks.
- 1.55 open ' Gordon' · *Arkham Knight*: He is supported by his allies, Robin (Matthew Mercer), Nightwing (Scott Porter), Catwoman (Grey DeLisle), Barbara Gordon (Ashley Greene)—who assists Batman covertly as t…

Negative pole L11·7340+, strongest sentences:

- 0.66 ' Ivy' · *Arkham City*: Returning villains include the Riddler (Wally Wingert), Victor Zsasz (Danny Jacobs), Bane (Fred Tatasciore), and Poison Ivy (Tasia Valenza).
- 0.65 ' Ivy' · *Batman Ninja*: Having captured a power converter from Harley, Catwoman attempts to bargain with Grodd in bringing her back to Gotham; however, they need to obtain other power converter…
- 0.62 'ney' · *The Martian*: Watney awakens after the storm, having narrowly survived his injuries.
- 0.61 ' Ivy' · *Hush*: Batman recovers and discovers that Poison Ivy used Catwoman to steal the ransom before taking the money from her.
- 0.59 ' Ivy' · *Batman & Robin*: Woodrue attempts to kill her by overturning a shelf of various toxins; instead, Isley is mutated by the toxins into Poison Ivy, who murders Woodrue with a venom-laden ki…
- 0.59 ' Ivy' · *Arkham Knight*: Throughout the city, Batman is faced with several supervillains: he must overcome Scarecrow's (John Noble) plot to threaten Gotham City, dismantle the Penguin's (Nolan N…
- 0.58 'ush' · *Arkham City*: The villain Hush (also voiced by Conroy), the mysterious Azrael (Khary Payton), and reporter Vicki Vale (also voiced by DeLisle) also appear.
- 0.57 'in' · *White Knight*: When the Joker's obsession with Batman culminated in Jason Todd's torture, Quinzel left him to try and help save Todd's life.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 2.53 (p100.0, open, peak ' Todd') · *A Death in the Family*: While eavesdropping on a child pornography ring and awaiting police backup in Gotham City, Jason Todd (Robin) ignores Batman's orders and attacks the criminals.
- 2.53 (p100.0, open, peak ' Todd') · *Under the Red Hood*: He realizes his mistake when he is unable to control Joker, who captures Jason Todd, the second Robin and Batman's partner.
- 2.50 (p100.0, open, peak ' Todd') · *White Knight*: When the Joker's obsession with Batman culminated in Jason Todd's torture, Quinzel left him to try and help save Todd's life.
- 2.30 (p99.9, open, peak ' Todd') · *Dark Knight Returns*: Over a decade after the United Nations outlawed superheroes and the Joker murdered Batman's protégé, Robin (Jason Todd), a 55-year-old Bruce Wayne battles alcoholism in…
- 2.28 (p99.9, open, peak ' Todd') · *Knightfall*: In addition, since the events from A Death in the Family, Batman struggles with the trauma of the loss of his protégé Jason Todd.
- 1.74 (p99.9, open, peak ' Gordon') · *Arkham Knight*: He is supported by his allies, Robin (Matthew Mercer), Nightwing (Scott Porter), Catwoman (Grey DeLisle), Barbara Gordon (Ashley Greene)—who assists Batman covertly as t…
- 1.67 (p99.8, open, peak ' parents') · *Batman v Superman*: In a flashback to his childhood, Bruce Wayne flees his parents' funeral.
- 1.60 (p99.8, open, peak ' Gordon') · *Return of the Joker*: After fending off the attack, Terry demands information from Bruce and Police Commissioner Barbara Gordon, the former Batgirl, but neither gives him the answers he seeks.

Blind maximum attempts:

- 0.20 (p69.6, echo, peak ',') blind_A: He arrived too late, and cradled his young sidekick's broken body in the smoking rubble, the boy he had trained now dead because of him.
- 0.18 (p61.6, echo, peak ' loses') blind_C: The mentor loses his young protégé when the sidekick he trained is beaten and killed, leaving the hero grieving his apprentice and blaming himself for the loss.
- 0.15 (p44.0, echo, peak ' boy') blind_B: The hero's young sidekick is beaten and killed by his archenemy, and the mentor, blaming himself, hangs the boy's costume in a glass case.

Near miss (losing, protégé): The chess master's protégé kept losing on time in blitz games, so she drilled him on clock management every afternoon.

### 10. The joke — L2·6468+

*trope; present in 26 of 32 Batman works; family of 1. Paper's note: Telling jokes, morbid jokes, the world as a black joke. Fires on the Killing Joke and Joker (2019) most.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 2 of 36 — in the first four layers, likely a token or spelling detector | no |
| 2. Sign regime | Top 50 firings: 47 gate-open, 3 echo, 0 other; echo share 6% (0 of the top 8; first echo at rank 44). Work scores (mean of the top 3 sentences) drawn from echo firings: 41% in Batman works, 99% in ordinary films; 14 of 32 Batman works and 69 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Joker' ×36 (36/0), ' joke' ×5 (5/0), 'oker' ×4 (4/0), ' jokes' ×2 (2/0), 'ler' ×2 (0/2), 'iddle' ×1 (0/1) | yes |
| 3. Negative pole | L2·6468-, max 0.42. The opposite pole is noise (Jones, -okes in Goodfellas, choked). | – |
| 4. Blind maximum | Best of 3: 12.69, against 10.31 for the corpus maximum in context and 12.38 for the corpus top 8 read standalone (corpus p100.0; peak on ' joke', open). The weakest of the corpus top 8 is 2.78 in context, 4.34 standalone; neutral sentences max 0.07. | yes; also beats the standalone top 8 |
| 5. Near miss | 12.00 (corpus p100.0; 116% of the in-context maximum, 95% of the best blind maximum; peak on ' joke') | no |
| **Verdict** | A word detector, layer 2. The top 8 peak on joke(s) and 36 of the top 50 on the name Joker. The best blind sentence beats the corpus maximum, but so does a restaurant whose prices are a joke (12.0 against 12.7). | **fails** |

Suggested label: **The tokens Joker and joke**

Strongest corpus sentences (activation, regime, peak token):

- 10.31 open ' joke' · *The Killing Joke*: He says that the situation reminds him of a joke about two inmates in an asylum who try to escape.
- 10.00 open ' joke' · *The Killing Joke*: Batman dodges a series of booby traps while the Joker tries to persuade his nemesis that the world is a "black, awful joke" that is not worth fighting for, and it only t…
- 10.00 open ' jokes' · *Joker (2019)*: As the show broadcasts live, Arthur tells morbid jokes, confesses to the subway murders, rants about societal neglect of the downtrodden, and berates Murray for ridiculi…
- 9.75 open ' joke' · *Joker (2019)*: Sometime later at Arkham, Arthur laughs to himself about a joke during a session with his new therapist.
- 9.06 open ' joke' · *The Killing Joke*: Batman chuckles at the joke's punchline, and as the two old foes share a laugh, he grabs the Joker as police arrive.
- 8.25 open ' jokes' · *Joker (2019)*: Sophie attends Arthur's stand-up routine, which fails due to his disorder and his jokes falling flat.
- 7.94 open ' joke' · *Joker (2019)*: After a final joke, Arthur shoots Murray in the head, killing him.
- 2.78 open ' Joker' · *Batman Ninja*: The Bat-Family saves Catwoman and Grodd before the Joker merges all of the castles into the super robot Lord Joker.

Negative pole L2·6468-, strongest sentences:

- 0.42 ' Jones' · *Raiders of the Lost Ark*: In 1936, American archaeologist Indiana Jones recovers a Golden Idol from a booby-trapped Peruvian temple.
- 0.42 'okes' · *Goodfellas*: Feeling betrayed by Henry's drug dealing and realizing that he is a liability, Paulie gives him $3,200 out of pity and ends their association - which revokes Henry's pro…
- 0.39 'okes' · *Some Like It Hot*: Meanwhile, Jerry, as Daphne, tries to get out of his engagement, listing reasons why "she" and Osgood cannot marry—she can't wear his mother's wedding gown because she's…
- 0.38 ' choked' · *Mad Max: Fury Road*: After entering enemy territory and fending off a rival gang, Furiosa drives into a dust choked supercell and loses all of her pursuers except Nux, who attempts to sacrif…
- 0.35 'ky' · *Forrest Gump*: Among their tenants is Elvis Presley, who incorporates Forrest's jerky dance movements into his performances.
- 0.35 ' Jake' · *Chinatown*: As Gittes is led away by his associates, one tells him, "Forget it, Jake.
- 0.32 'ke' · *Some Like It Hot*: Broke, terrified, and desperate to leave Chicago, Joe and Jerry disguise themselves as women named Josephine and Daphne so they can join Sweet Sue and her Society Syncop…
- 0.31 'ly' · *Jaws*: At nightfall, Quint and Hooper drunkenly exchange stories about their assorted body scars.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 12.38 (p100.0, open, peak ' joke') · *Joker (2019)*: Sometime later at Arkham, Arthur laughs to himself about a joke during a session with his new therapist.
- 11.62 (p100.0, open, peak ' jokes') · *Joker (2019)*: As the show broadcasts live, Arthur tells morbid jokes, confesses to the subway murders, rants about societal neglect of the downtrodden, and berates Murray for ridiculi…
- 11.50 (p100.0, open, peak ' joke') · *The Killing Joke*: He says that the situation reminds him of a joke about two inmates in an asylum who try to escape.
- 10.94 (p100.0, open, peak ' joke') · *The Killing Joke*: Batman dodges a series of booby traps while the Joker tries to persuade his nemesis that the world is a "black, awful joke" that is not worth fighting for, and it only t…
- 10.81 (p100.0, open, peak ' joke') · *The Killing Joke*: Batman chuckles at the joke's punchline, and as the two old foes share a laugh, he grabs the Joker as police arrive.
- 9.56 (p99.9, open, peak ' joke') · *Joker (2019)*: After a final joke, Arthur shoots Murray in the head, killing him.
- 9.31 (p99.9, open, peak ' jokes') · *Joker (2019)*: Sophie attends Arthur's stand-up routine, which fails due to his disorder and his jokes falling flat.
- 4.34 (p99.8, open, peak ' Joker') · *Batman Ninja*: The Bat-Family saves Catwoman and Grodd before the Joker merges all of the castles into the super robot Lord Joker.

Blind maximum attempts:

- 12.69 (p100.0, open, peak ' joke') blind_A: The grinning killer told his joke with perfect timing, setup, pause, punchline, then howled with laughter at a gag only he found funny.
- 11.94 (p100.0, open, peak ' joke') blind_C: He delivered the joke with a grin, the setup, the punchline and the gag landing so well that everyone laughed uproariously at the comedian's hilarious jest.
- 9.56 (p99.9, open, peak ' joke') blind_B: The clown-faced villain tells a long joke to his captive, breaks into hysterical laughter at the punchline, and insists that life itself is one big joke.

Near miss (joke): Critics called the restaurant's prices a joke, but the dining room was fully booked for months.

### 11. The rogues' gallery — L10·5942+

*trope; present in 31 of 32 Batman works; family of 12. Paper's note: A roll-call of costumed villains encountered one after another.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 10 of 36 | yes |
| 2. Sign regime | Top 50 firings: 50 gate-open, 0 echo, 0 other; echo share 0% (0 of the top 8; first echo at rank none in the top 50). Work scores (mean of the top 3 sentences) drawn from echo firings: 0% in Batman works, 70% in ordinary films; 0 of 32 Batman works and 52 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Ivy' ×12 (12/0), 'crow' ×11 (11/0), ' Penguin' ×11 (11/0), ' Harley' ×7 (7/0), 'ane' ×3 (3/0), 'dd' ×1 (1/0) | yes |
| 3. Negative pole | L10·5942-, max 11.69. The opposite pole peaks on other franchises' names (Oliver Queen, Stark, Osborn, the Ten Rings, Jor-El, Uncle Ben): Batman's villains against other heroes' casts. | – |
| 4. Blind maximum | Best of 3: 0.40, against 6.47 for the corpus maximum in context and 6.75 for the corpus top 8 read standalone (corpus p85.4; peak on ' assembled', open). The weakest of the corpus top 8 is 5.53 in context, 3.44 standalone; neutral sentences max 0.24. | no |
| 5. Near miss | 0.16 (corpus p39.6; 2% of the in-context maximum, 41% of the best blind maximum; peak on ' landscapes') | yes |
| **Verdict** | Fails. Peaks on the villain names Poison Ivy, Scarecrow, Penguin, Harley and Bane (all gate-open). A name-free roster of costumed villains scores 6% of the maximum. The unit does generalise across Batman villain names, which is more than a single-name detector, but it is names. | **fails** |

Suggested label: **Batman villain names (Poison Ivy, Scarecrow, Penguin, Harley Quinn, Bane)**

Strongest corpus sentences (activation, regime, peak token):

- 6.47 open 'crow' · *Arkham Asylum (comic)*: Through the asylum, Batman encounters Clayface, Doctor Destiny, Scarecrow, Mad Hatter, Maxie Zeus, and Killer Croc.
- 6.34 open ' Ivy' · *Batman Ninja*: Having captured a power converter from Harley, Catwoman attempts to bargain with Grodd in bringing her back to Gotham; however, they need to obtain other power converter…
- 6.12 open ' Ivy' · *Arkham City*: Returning villains include the Riddler (Wally Wingert), Victor Zsasz (Danny Jacobs), Bane (Fred Tatasciore), and Poison Ivy (Tasia Valenza).
- 6.09 open 'crow' · *Arkham Knight*: Throughout the city, Batman is faced with several supervillains: he must overcome Scarecrow's (John Noble) plot to threaten Gotham City, dismantle the Penguin's (Nolan N…
- 5.94 open ' Ivy' · *Hush*: Batman recovers and discovers that Poison Ivy used Catwoman to steal the ransom before taking the money from her.
- 5.72 open 'crow' · *Knightfall*: The rescue of Mayor Krol from the teaming of the Joker and Scarecrow pushes Batman to his mental and physical limits: a dose of Scarecrow's fear gas makes him relive the…
- 5.59 open 'crow' · *Hush*: He then encounters the cause of the Huntress' attack, the Scarecrow, but overpowers him in a graveyard.
- 5.53 open ' Ivy' · *Arkham Asylum (game)*: In the asylum, Batman is faced with several supervillains; he must defend himself from an enraged Bane (Fred Tatasciore), subdue indiscriminate serial killer Victor Zsas…

Negative pole L10·5942-, strongest sentences:

- 11.69 ' Queen' · *Dark Knight Returns*: However, an aging Oliver Queen manages to shoot Superman with a kryptonite-tipped arrow to weaken him, which allows Batman to beat him into submission.
- 6.72 ' Stark' · *Iron Man*: Tony Stark, who has inherited the defense contractor Stark Industries from his late father Howard Stark, tours in war-torn Afghanistan with his best friend and military…
- 6.56 'born' · *Spider-Man*: On a high school field trip, Peter Parker visits a Columbia University genetics laboratory with his best friend, Harry Osborn, and his crush, Mary Jane Watson.
- 6.06 ' Rings' · *Iron Man*: After the demonstration, his convoy is ambushed by a terrorist group, the Ten Rings, and Stark is gravely wounded by a missile used by the attackers—one of his company's…
- 5.84 'El' · *Superman*: On the planet Krypton, Jor-El, a member of the Kryptonian high council, sentences criminals General Zod, Ursa, and Non to the Phantom Zone.
- 5.84 ' Man' · *Iron Man*: The next day, at a press conference, Stark publicly admits to being the superhero the press has dubbed "Iron Man".
- 5.25 ' Rings' · *Iron Man*: He is captured and imprisoned in a cave by the Ten Rings.
- 5.03 ' Ben' · *Spider-Man*: Soon after, Peter's Uncle Ben is killed by the same thief.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 6.75 (p100.0, open, peak 'crow') · *Arkham Asylum (comic)*: Through the asylum, Batman encounters Clayface, Doctor Destiny, Scarecrow, Mad Hatter, Maxie Zeus, and Killer Croc.
- 6.50 (p100.0, open, peak ' Ivy') · *Batman Ninja*: Having captured a power converter from Harley, Catwoman attempts to bargain with Grodd in bringing her back to Gotham; however, they need to obtain other power converter…
- 6.19 (p99.9, open, peak 'crow') · *Arkham Knight*: Throughout the city, Batman is faced with several supervillains: he must overcome Scarecrow's (John Noble) plot to threaten Gotham City, dismantle the Penguin's (Nolan N…
- 5.75 (p99.9, open, peak 'crow') · *Knightfall*: The rescue of Mayor Krol from the teaming of the Joker and Scarecrow pushes Batman to his mental and physical limits: a dose of Scarecrow's fear gas makes him relive the…
- 5.50 (p99.7, open, peak ' Ivy') · *Hush*: Batman recovers and discovers that Poison Ivy used Catwoman to steal the ransom before taking the money from her.
- 5.47 (p99.7, open, peak ' Ivy') · *Arkham City*: Returning villains include the Riddler (Wally Wingert), Victor Zsasz (Danny Jacobs), Bane (Fred Tatasciore), and Poison Ivy (Tasia Valenza).
- 5.38 (p99.7, open, peak ' Ivy') · *Arkham Asylum (game)*: In the asylum, Batman is faced with several supervillains; he must defend himself from an enraged Bane (Fred Tatasciore), subdue indiscriminate serial killer Victor Zsas…
- 3.44 (p99.1, open, peak 'crow') · *Hush*: He then encounters the cause of the Huntress' attack, the Scarecrow, but overpowers him in a graveyard.

Blind maximum attempts:

- 0.40 (p85.4, open, peak ' assembled') blind_C: The hero's rogues' gallery, a whole roster of recurring costumed villains, archenemies, crime bosses and madmen, assembled together to plot his downfall.
- 0.26 (p71.6, echo, peak ' a') blind_A: One by one his old enemies returned, the clown, the puzzle-obsessed killer, the umbrella-wielding mobster, the scarred prosecutor, the cat burglar, a whole gallery of costumed villains.
- 0.23 (p63.8, echo, peak '-') blind_B: The vigilante's recurring enemies, a gallery of costumed criminals including a clown, a riddle-obsessed killer, a scarred attorney and a plant-controlling botanist, join forces against him.

Near miss (gallery, rogue): The gallery on the second floor displayed landscapes, and the docent explained that rogue waves inspired the largest canvas.

### 12. Superhero powers (peaks on Superman) — L10·7189+

*trope; present in 31 of 32 Batman works; family of 5. Paper's note: Superpowered heroes and kryptonite. Strongest on Superman's origin.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 10 of 36 | yes |
| 2. Sign regime | Top 50 firings: 50 gate-open, 0 echo, 0 other; echo share 0% (0 of the top 8; first echo at rank none in the top 50). Work scores (mean of the top 3 sentences) drawn from echo firings: 0% in Batman works, 46% in ordinary films; 0 of 32 Batman works and 31 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): 'rypton' ×11 (11/0), ' Bat' ×10 (10/0), ' Clark' ×5 (5/0), ' Superman' ×3 (3/0), 'ats' ×3 (3/0), 'El' ×2 (2/0) | yes |
| 3. Negative pole | L10·7189-, max 9.31. The opposite pole fires on seemingly killed, presumed dead, faked his death and resurrection: apparent death and return, a cleaner concept than the positive pole. | – |
| 4. Blind maximum | Best of 3: 5.34, against 7.03 for the corpus maximum in context and 7.16 for the corpus top 8 read standalone (corpus p99.6; peak on ' yellow', open). The weakest of the corpus top 8 is 5.78 in context, 1.39 standalone; neutral sentences max 0.35. | no |
| 5. Near miss | 0.08 (corpus p24.3; 1% of the in-context maximum, 2% of the best blind maximum; peak on ' elderly') | yes |
| **Verdict** | Fails narrowly. Peaks on kryptonite, Clark, Kal-El and Superman, plus Bat-: a Superman-name detector. Yet the packed name-free sentence reaches 76% of the maximum on yellow (sun), so some Superman lore is read without the name; the near miss is silent. | **fails** |

Suggested label: **Superman names and kryptonite (Kal-El, Clark, kryptonite)**

Strongest corpus sentences (activation, regime, peak token):

- 7.03 open 'El' · *Superman*: Before the planet's destruction, Jor-El and his wife Lara send their infant son Kal-El to Earth.
- 6.91 open 'rypton' · *Dark Knight Returns*: However, an aging Oliver Queen manages to shoot Superman with a kryptonite-tipped arrow to weaken him, which allows Batman to beat him into submission.
- 6.78 open 'rypton' · *Batman v Superman*: Wearing an armored exosuit and wielding a kryptonite spear, he lights the Bat-Signal and awaits a battle with Superman.
- 6.66 open ' Clark' · *Superman*: They adopt him, naming him Clark.
- 6.59 open 'rypton' · *Hush*: Using various weapons, including flashbang grenades, hypersonics, electroshock, and a kryptonite ring to defend himself, Batman stalls Superman while Catwoman lets Lois…
- 6.28 open ' Clark' · *Batman v Superman*: Clark Kent, Superman's civilian identity, seeks to condemn Batman's brutal methods in Daily Planet articles.
- 6.16 open 'rypton' · *Batman v Superman*: LexCorp mogul Lex Luthor attempts to persuade US Senator June Finch to allow him to import kryptonite discovered in the aftermath of Zod's terraforming attempt, so that…
- 5.78 open ' Clark' · *Dark Knight Returns*: Clark Kent talks with Wayne and is then deployed by Washington to the fictional Latin American country of Corto Maltese, where he fights Soviet combat forces in a confli…

Negative pole L10·7189-, strongest sentences:

- 9.31 ' seemingly' · *Raiders of the Lost Ark*: Nazi soldiers and mercenaries attack Jones, and Marion is seemingly killed, leaving him despondent.
- 9.12 ' seemingly' · *Daredevil*: Moments later, when Bullseye finds Elektra, Matt is helpless as the assassin seemingly kills her.
- 8.88 ' seemingly' · *Die Hard*: Karl, enraged by Tony's death, attacks McClane and is seemingly killed.
- 7.91 ' death' · *The Long Halloween*: Holiday is revealed to be Alberto Falcone, son of Carmine Falcone, who had faked his death.
- 7.28 ' dead' · *Dark Knight Rises*: In the aftermath, Batman is presumed dead and honored as a hero.
- 7.22 ' resurrection' · *Under the Red Hood*: Ra's explains that he felt responsible for Jason's death and, as a peace offering, swapped Jason's body for a fake and revived him in the Lazarus Pit, but following his…
- 6.84 'ush' · *Hush*: He comes to be referred to as Hush.
- 6.72 ' apparently' · *Hush*: In the ensuing struggle, Dr. Elliot is apparently shot dead by the Joker.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 7.16 (p100.0, open, peak 'rypton') · *Batman v Superman*: LexCorp mogul Lex Luthor attempts to persuade US Senator June Finch to allow him to import kryptonite discovered in the aftermath of Zod's terraforming attempt, so that…
- 6.97 (p100.0, open, peak 'El') · *Superman*: Before the planet's destruction, Jor-El and his wife Lara send their infant son Kal-El to Earth.
- 6.19 (p99.8, open, peak 'rypton') · *Dark Knight Returns*: However, an aging Oliver Queen manages to shoot Superman with a kryptonite-tipped arrow to weaken him, which allows Batman to beat him into submission.
- 5.25 (p99.6, open, peak 'rypton') · *Batman v Superman*: Wearing an armored exosuit and wielding a kryptonite spear, he lights the Bat-Signal and awaits a battle with Superman.
- 5.03 (p99.6, open, peak ' Clark') · *Superman*: They adopt him, naming him Clark.
- 3.56 (p98.8, open, peak ' Superman') · *Batman v Superman*: Clark Kent, Superman's civilian identity, seeks to condemn Batman's brutal methods in Daily Planet articles.
- 3.48 (p98.6, open, peak 'rypton') · *Hush*: Using various weapons, including flashbang grenades, hypersonics, electroshock, and a kryptonite ring to defend himself, Batman stalls Superman while Catwoman lets Lois…
- 1.39 (p92.5, open, peak ' Kent') · *Dark Knight Returns*: Clark Kent talks with Wayne and is then deployed by Washington to the fictional Latin American country of Corto Maltese, where he fights Soviet combat forces in a confli…

Blind maximum attempts:

- 5.34 (p99.6, open, peak ' yellow') blind_C: The caped alien superhero possesses superhuman powers: flight, super-strength, invulnerability, heat vision, x-ray vision, super-speed and freezing breath, drawn from the yellow sun.
- 1.74 (p93.9, open, peak ' vision') blind_B: The caped alien hero flies at supersonic speed, lifts a collapsing bridge with his bare hands, melts steel with his heat vision, and shrugs off bullets.
- 0.52 (p84.4, open, peak ' alien') blind_A: The caped alien lifted the collapsing bridge over his head, bullets flattening against his chest, then flew faster than sound with heat burning from his eyes.

Near miss (powers, super): The lawyer drafted powers of attorney for the elderly couple, and the apartment super signed as a witness.

### 13. Franchise roll-call (Wikipedia style) — L5·3626+

*register; present in 23 of 32 Batman works; family of 167. Paper's note: Sentences listing a franchise's cast and characters. A trace of how the summaries are written, not of the stories.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 5 of 36 | yes |
| 2. Sign regime | Top 50 firings: 50 gate-open, 0 echo, 0 other; echo share 0% (0 of the top 8; first echo at rank none in the top 50). Work scores (mean of the top 3 sentences) drawn from echo firings: 52% in Batman works, 91% in ordinary films; 19 of 32 Batman works and 65 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): 'ham' ×50 (50/0) | yes |
| 3. Negative pole | L5·3626-, max 4.16. The opposite pole peaks on the Orca (Jaws), Kapoor, Ozarks and Azrael: sub-word noise. | – |
| 4. Blind maximum | Best of 3: 0.02, against 13.00 for the corpus maximum in context and 13.00 for the corpus top 8 read standalone (corpus p31.7; peak on 'ler', echo). The weakest of the corpus top 8 is 10.44 in context, 10.75 standalone; neutral sentences max 0.15. | no |
| 5. Near miss | 0.03 (corpus p39.7; 0% of the in-context maximum, 129% of the best blind maximum; peak on ' uniforms') | no |
| **Verdict** | Fails. All 50 of the top 50 peak on -ham in Arkham. Name-free sentences in Wikipedia roll-call style score 0.02. | **fails** |

Suggested label: **The token -ham in Arkham**

Strongest corpus sentences (activation, regime, peak token):

- 13.00 open 'ham' · *Arkham City*: Arkham City features a large ensemble cast of characters from the history of Batman comics.
- 13.00 open 'ham' · *Arkham Knight*: Arkham Knight features a large ensemble cast of characters from the history of Batman comics.
- 13.00 open 'ham' · *Arkham Asylum (game)*: Arkham Asylum is set in the fictional Arkham Asylum, a facility on Arkham Island off the coast of Gotham City that houses criminally insane supervillains.
- 11.12 open 'ham' · *Batman Ninja*: While battling Gorilla Grodd at Arkham Asylum, Batman is caught in Grodd's time displacement machine, dubbed the "Quake Engine", and sent to feudal Japan.
- 10.88 open 'ham' · *Earth One*: Bruce Wayne is the eight-year-old son of Dr. Thomas Wayne, a mayoral candidate for Gotham City, and Martha Wayne (née Arkham).
- 10.75 open 'ham' · *Arkham City*: Returning characters from Arkham Asylum include Batman (Kevin Conroy), the Joker (Mark Hamill)—in what Hamill stated would be his final time voicing the character; (he s…
- 10.50 open 'ham' · *Arkham Knight*: Batman investigates the facility and discovers a large militia being dispatched to occupy Gotham, led by the mysterious "Arkham Knight".
- 10.44 open 'ham' · *Arkham Asylum (comic)*: On April 1, Commissioner Gordon informs Batman that the patients of Arkham Asylum, led by the Joker, have taken over the facility, threatening to murder the staff unless…

Negative pole L5·3626-, strongest sentences:

- 4.16 'ca' · *Jaws*: Despite initial tension between Quint and Hooper, and Brody's fear of the ocean, the three head out to sea on Quint's boat, the Orca, to hunt for the shark.
- 2.78 'ca' · *Jaws*: To Brody's relief, Quint speeds the Orca toward shore to draw the shark into shallower waters, but the damaged engine fails.
- 2.62 ' Kapoor' · *The Martian*: Mars missions director Vincent Kapoor realizes this strategy, and quickly visits Jet Propulsion Laboratory (JPL) director Bruce Ng to use their replica of the probe.
- 2.16 'arks' · *Gone Girl*: Amy hides in a campground in the Ozarks.
- 1.49 'rael' · *Arkham City*: The villain Hush (also voiced by Conroy), the mysterious Azrael (Khary Payton), and reporter Vicki Vale (also voiced by DeLisle) also appear.
- 1.38 'rael' · *Arkham Knight*: Batman's loyal butler Alfred Pennyworth (Martin Jarvis) and Wayne Enterprises colleague Lucius Fox (Dave Fennoy) provide Batman with tactical support, and the holy warri…
- 1.19 'rael' · *Knightfall*: Jean-Paul Valley (introduced in Batman: Sword of Azrael #1-4 (October 1992-January 1993) by Dennis O'Neil and Joe Quesada), a graduate student at Gotham University who d…
- 1.18 'lo' · *Jurassic Park*: Upon arrival, the group is shocked to see living Brachiosaurus and Parasaurolophus.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 13.00 (p99.9, open, peak 'ham') · *Arkham Asylum (game)*: Arkham Asylum is set in the fictional Arkham Asylum, a facility on Arkham Island off the coast of Gotham City that houses criminally insane supervillains.
- 12.94 (p99.9, open, peak 'ham') · *Arkham City*: Arkham City features a large ensemble cast of characters from the history of Batman comics.
- 12.94 (p99.9, open, peak 'ham') · *Arkham Knight*: Arkham Knight features a large ensemble cast of characters from the history of Batman comics.
- 11.81 (p99.9, open, peak 'ham') · *Arkham Knight*: Batman investigates the facility and discovers a large militia being dispatched to occupy Gotham, led by the mysterious "Arkham Knight".
- 11.12 (p99.9, open, peak 'ham') · *Batman Ninja*: While battling Gorilla Grodd at Arkham Asylum, Batman is caught in Grodd's time displacement machine, dubbed the "Quake Engine", and sent to feudal Japan.
- 10.88 (p99.9, open, peak 'ham') · *Earth One*: Bruce Wayne is the eight-year-old son of Dr. Thomas Wayne, a mayoral candidate for Gotham City, and Martha Wayne (née Arkham).
- 10.88 (p99.9, open, peak 'ham') · *Arkham City*: Returning characters from Arkham Asylum include Batman (Kevin Conroy), the Joker (Mark Hamill)—in what Hamill stated would be his final time voicing the character; (he s…
- 10.75 (p99.8, open, peak 'ham') · *Dark Knight Returns*: Batman's return is reported on the news, prompting the Joker to awaken from catatonia at Arkham Asylum.

Blind maximum attempts:

- 0.02 (p31.7, echo, peak 'ler') blind_B: The film is the fourth installment in the franchise, with the principal cast reprising their roles as the vigilante, his butler, the police commissioner and the sidekick.
- 0.02 (p27.9, echo, peak 'ler') blind_C: The sequel, the third installment in the long-running film series, features the returning cast reprising their roles as the hero, the butler, the commissioner and the love interest.
- 0.01 (p11.6, echo, peak ' reboot') blind_A: The film was the fourth installment in the live-action series, following three earlier entries, and was itself followed by a sequel, a reboot, and an animated spin-off.

Near miss (franchise, roll call): The fast-food franchise held a morning roll call of staff before opening, checking uniforms and name badges.

### 14. The butler who raised him — L6·4705-

*character; present in 18 of 32 Batman works; family of 47. Paper's note: Alfred Pennyworth as the family butler and surrogate parent.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 6 of 36 | yes |
| 2. Sign regime | Top 50 firings: 49 gate-open, 1 echo, 0 other; echo share 2% (0 of the top 8; first echo at rank 44). Work scores (mean of the top 3 sentences) drawn from echo firings: 52% in Batman works, 81% in ordinary films; 19 of 32 Batman works and 62 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): 'worth' ×21 (21/0), 'oby' ×4 (4/0), 'o' ×2 (2/0), 'iti' ×2 (2/0), 'ista' ×1 (1/0), 'ne' ×1 (1/0) | yes |
| 3. Negative pole | L6·4705+, max 1.47. The opposite pole is noise (Vaughn, Stone, Laszlo). | – |
| 4. Blind maximum | Best of 3: 0.19, against 6.69 for the corpus maximum in context and 7.25 for the corpus top 8 read standalone (corpus p87.9; peak on 'aged', echo). The weakest of the corpus top 8 is 6.34 in context, 6.50 standalone; neutral sentences max 0.20. | no |
| 5. Near miss | 0.03 (corpus p10.2; 0% of the in-context maximum, 14% of the best blind maximum; peak on ' morning') | yes |
| **Verdict** | Fails. Every top-8 peak and 21 of the top 50 are -worth in Pennyworth. A name-free elderly butler who raised the orphaned boy scores 0.19, below the neutral sentences. | **fails** |

Suggested label: **The token -worth in Pennyworth**

Strongest corpus sentences (activation, regime, peak token):

- 6.69 open 'worth' · *Joker (2019)*: He goes to Wayne Manor, where he meets Thomas's young son Bruce, but flees after a scuffle with family butler Alfred Pennyworth.
- 6.62 open 'worth' · *Batman Begins*: Bruce is raised by the family butler, Alfred Pennyworth.
- 6.56 open 'worth' · *Earth One*: After receiving death threats, Thomas contacts his friend from his military service, Alfred Pennyworth, to become the head of security at Wayne Manor.
- 6.44 open 'worth' · *The Batman*: Bruce's butler and caretaker, Alfred Pennyworth, is hospitalized after opening a letter bomb addressed to Bruce.
- 6.44 open 'worth' · *Hush*: His butler, Alfred Pennyworth, follows his instructions to summon Bruce's childhood friend Thomas Elliot, who is now a renowned brain surgeon.
- 6.41 open 'worth' · *Batman & Robin*: Meanwhile, Alfred Pennyworth's niece, Barbara Wilson, makes a surprise visit and is invited by Bruce to stay at Wayne Manor until she goes back to school.
- 6.34 open 'worth' · *A Death in the Family*: Traumatized, Batman takes Jason and Sheila's remains to Gotham and holds a burial with Alfred Pennyworth, Commissioner James Gordon, and Barbara Gordon.
- 6.34 open 'worth' · *Batman '89*: Vicki is taken to the Batcave by Wayne's butler, Alfred Pennyworth, who had been coaxing the relationship between the pair.

Negative pole L6·4705+, strongest sentences:

- 1.47 'n' · *Jaws*: Mayor Larry Vaughn persuades him to reconsider, fearing the town's summer economy will suffer.
- 1.21 ' Stone' · *Gravity*: Dr. Ryan Stone is aboard on her first space mission, to perform hardware upgrades on the telescope.
- 1.21 'lo' · *Casablanca*: She is accompanied by her husband, Victor Laszlo, a renowned fugitive Czechoslovak Resistance leader.
- 1.12 ' Malone' · *Forrest Gump*: He also witnesses Governor George Wallace's Stand in the Schoolhouse Door, during which Forrest returns a dropped book to Vivian Malone Jones.
- 1.04 'ster' · *The Exorcist*: In northern Iraq, priest Lankester Merrin takes part in an archaeological dig in the ancient ruins of Hatra.
- 0.93 ' Wales' · *Scarlet Pimpernel*: Despite being the talk of London society, only his followers and possibly the Prince of Wales know the Pimpernel's true identity.
- 0.92 'man' · *Ghostbusters*: After Columbia University parapsychology professors Peter Venkman, Ray Stantz, and Egon Spengler experience their first encounter with a ghost at the New York Public Lib…
- 0.90 'ara' · *Gone Girl*: Tommy O'Hara claims that she framed him for rape and ruined his life after they broke up.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 7.25 (p100.0, open, peak 'worth') · *Batman Begins*: Bruce is raised by the family butler, Alfred Pennyworth.
- 6.91 (p100.0, open, peak 'worth') · *The Batman*: Bruce's butler and caretaker, Alfred Pennyworth, is hospitalized after opening a letter bomb addressed to Bruce.
- 6.88 (p100.0, open, peak 'worth') · *Batman & Robin*: Meanwhile, Alfred Pennyworth's niece, Barbara Wilson, makes a surprise visit and is invited by Bruce to stay at Wayne Manor until she goes back to school.
- 6.72 (p100.0, open, peak 'worth') · *Hush*: His butler, Alfred Pennyworth, follows his instructions to summon Bruce's childhood friend Thomas Elliot, who is now a renowned brain surgeon.
- 6.69 (p100.0, open, peak 'worth') · *Earth One*: After receiving death threats, Thomas contacts his friend from his military service, Alfred Pennyworth, to become the head of security at Wayne Manor.
- 6.66 (p100.0, open, peak 'worth') · *Joker (2019)*: He goes to Wayne Manor, where he meets Thomas's young son Bruce, but flees after a scuffle with family butler Alfred Pennyworth.
- 6.56 (p99.9, open, peak 'worth') · *A Death in the Family*: Traumatized, Batman takes Jason and Sheila's remains to Gotham and holds a burial with Alfred Pennyworth, Commissioner James Gordon, and Barbara Gordon.
- 6.50 (p99.9, open, peak 'worth') · *The Dark Knight*: Bruce struggles to understand the Joker's motives, to which his butler Alfred Pennyworth says that "some men just want to watch the world burn."

Blind maximum attempts:

- 0.19 (p87.9, echo, peak 'aged') blind_A: The elderly butler who had bandaged the orphaned boy's knees now stitched the grown man's wounds in the kitchen, scolding him gently as a father would.
- 0.10 (p58.4, echo, peak ' as') blind_B: The family's elderly butler, who raised the orphaned boy after his parents' murder, continues to serve as his confidant, surgeon and surrogate father.
- 0.03 (p12.8, echo, peak ' his') blind_C: The loyal elderly butler, who raised the orphaned boy as a surrogate father after his parents died, still serves the manor, tends his wounds and offers dry advice.

Near miss (butler, raised): The butler raised the blinds at seven each morning and set the newspaper beside the breakfast tray.

### 15. Taking in an orphan as a ward — L35·6441-

*trope; present in 9 of 32 Batman works; family of 1. Paper's note: A wealthy guardian adopts an orphan (Dick Grayson, the Gaslight orphans) who becomes a partner.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 35 of 36 | yes |
| 2. Sign regime | Top 50 firings: 50 gate-open, 0 echo, 0 other; echo share 0% (0 of the top 8; first echo at rank none in the top 50). Work scores (mean of the top 3 sentences) drawn from echo firings: 67% in Batman works, 95% in ordinary films; 23 of 32 Batman works and 67 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ',' ×9 (9/0), ' and' ×7 (7/0), '.' ×7 (7/0), ' with' ×2 (2/0), ' save' ×2 (2/0), 'ed' ×1 (1/0) | yes |
| 3. Negative pole | L35·6441+, max 75.50. The opposite pole peaks on a capital D token about to start a name (Dug, Deadshot, Diana): the same channel seen from the other side, a last-layer letter predictor. | – |
| 4. Blind maximum | Best of 3: 0.32, against 70.50 for the corpus maximum in context and 109.50 for the corpus top 8 read standalone (corpus p67.6; peak on ' ac', echo). The weakest of the corpus top 8 is 46.50 in context, -0.03 standalone; neutral sentences max 0.24. | no |
| 5. Near miss | 0.05 (corpus p32.7; 0% of the in-context maximum, 15% of the best blind maximum; peak on ' laundry') | yes |
| **Verdict** | Fails. A layer-35 (last layer) unit whose peaks are function tokens (comma, and, full stop, accepts, except) in Batman Forever sentences about Dick Grayson: a next-token unit, probably for Dick or Robin, not a story pattern. Name-free orphan-ward sentences score 0.3 against 70 in context; one of the top-8 sentences read standalone scores below zero. | **fails** |

Suggested label: **Layer-35 next-token unit around Dick Grayson sentences (predicts Dick / Robin)**

Strongest corpus sentences (activation, regime, peak token):

- 70.50 open ',' · *Gotham by Gaslight*: Batman and Selina are saved by Alfred and the orphans, who are now Bruce Wayne's wards.
- 60.25 open 'ed' · *Batman Forever*: Bruce invites the now-orphaned Dick to live at Wayne Manor as his ward, where he later discovers that Bruce is Batman.
- 60.00 open 'ens' · *Lego Batman*: Appearing as Batman before Dick, Bruce states that he is additionally adopting him and christens him as Robin.
- 56.50 open ' accepts' · *Batman Forever*: Bruce finally accepts Dick as his comrade, Robin.
- 54.00 open ' except' · *Batman Forever*: Dick Grayson, the youngest member of the Flying Graysons family of acrobats, prevents the bomb from killing anyone by throwing it into a river with the help of his famil…
- 53.25 open ',' · *Batman Forever*: Seeking to avenge the death of his family, Dick demands to join Batman in crime-fighting, hoping to kill Two-Face, but Bruce declines in order to help Dick move on inste…
- 48.25 open ' with' · *Batman Forever*: Batman visits Chase, who explains that she has fallen in love with Bruce, and Bruce reveals his secret identity to her.
- 46.50 open ' and' · *Batman & Robin*: Alfred is cured, and Bruce and Dick agree to let Barbara join them in fighting crime.

Negative pole L35·6441+, strongest sentences:

- 75.50 ' D' · *Up*: Along the way, they encounter a giant flightless bird that Russell names Kevin (though the bird is later revealed to be female) and Dug, a Golden Retriever whose collar…
- 48.25 'D' · *Arkham Asylum (game)*: In the asylum, Batman is faced with several supervillains; he must defend himself from an enraged Bane (Fred Tatasciore), subdue indiscriminate serial killer Victor Zsas…
- 32.50 'D' · *Arkham Knight*: Other villains include the pyromaniac Firefly (Crispin Freeman), the beastly Man-Bat (Loren Lester), the veteran assassin Deathstroke (Mark Rolston), the serial killer P…
- 30.88 ' d' · *Get Out*: At Missy's prompting, his consciousness falls into a dark void she dubs the "sunken place", where his body becomes temporarily paralyzed.
- 30.62 ' D' · *Big*: He soon attracts the attention of Susan Lawrence, a fellow executive, and a romance begins to develop, much to the dismay of her former boyfriend Davenport.
- 23.12 ' d' · *Up*: Russell receives his badge for assisting the elderly, and Carl presents Russell with a bottle cap Ellie gave Carl when they first met as kids years ago, which he dubs "T…
- 17.38 ' D' · *Babe*: Dismayed when the Hoggetts buy an alarm clock, Ferdinand persuades Babe to help him get rid of it.
- 14.88 ' und' · *Gravity*: She enters Tiangong's Shenzhou capsule just as the station enters the upper atmosphere, and undocks.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 109.50 (p100.0, open, peak ' that') · *Batman Forever*: Bruce invites the now-orphaned Dick to live at Wayne Manor as his ward, where he later discovers that Bruce is Batman.
- 55.75 (p99.9, open, peak ' except') · *Batman Forever*: Dick Grayson, the youngest member of the Flying Graysons family of acrobats, prevents the bomb from killing anyone by throwing it into a river with the help of his famil…
- 49.75 (p99.8, open, peak 'ens') · *Lego Batman*: Appearing as Batman before Dick, Bruce states that he is additionally adopting him and christens him as Robin.
- 29.75 (p99.6, open, peak '.') · *Batman Forever*: Seeking to avenge the death of his family, Dick demands to join Batman in crime-fighting, hoping to kill Two-Face, but Bruce declines in order to help Dick move on inste…
- 10.81 (p99.0, open, peak '.') · *Batman Forever*: Bruce finally accepts Dick as his comrade, Robin.
- 2.31 (p96.9, open, peak '.') · *Batman & Robin*: Alfred is cured, and Bruce and Dick agree to let Barbara join them in fighting crime.
- 0.02 (p25.9, echo, peak ' Sel') · *Gotham by Gaslight*: Batman and Selina are saved by Alfred and the orphans, who are now Bruce Wayne's wards.
- -0.03 (p4.5, other, peak ' fallen') · *Batman Forever*: Batman visits Chase, who explains that she has fallen in love with Bruce, and Bruce reveals his secret identity to her.

Blind maximum attempts:

- 0.32 (p67.6, echo, peak ' ac') blind_B: After the boy's acrobat parents are killed, the wealthy bachelor takes the orphan into his home as his legal ward and raises him as his own.
- 0.11 (p45.2, echo, peak 'ape') blind_A: After the circus boy's parents fell from the trapeze, the wealthy bachelor took the orphan into his mansion as his legal ward and raised him.
- 0.00 (p18.6, echo, peak 'ive') blind_C: After the boy's parents were murdered, the wealthy bachelor took the orphan into his home as his legal ward, becoming guardian and adoptive father to the child.

Near miss (ward, took in, orphan): The hospital ward took in extra patients during the storm, and an orphan sock turned up in the laundry.

### 16. The villain's lair — L4·7177-

*place; present in 8 of 32 Batman works; family of 6. Paper's note: Raiding or escaping a criminal's hideout or laboratory.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 4 of 36 | yes |
| 2. Sign regime | Top 50 firings: 11 gate-open, 39 echo, 0 other; echo share 78% (0 of the top 8; first echo at rank 11). Work scores (mean of the top 3 sentences) drawn from echo firings: 91% in Batman works, 97% in ordinary films; 30 of 32 Batman works and 69 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): 'c' ×16 (0/16), 'ir' ×9 (9/0), 'face' ×9 (0/9), ' races' ×5 (0/5), ' race' ×3 (0/3), 'q' ×2 (0/2) | no |
| 3. Negative pole | L4·7177+, max 17.12. The opposite pole fires on the string lac- (laced, lacing, lacrosse): nothing to do with lairs. | – |
| 4. Blind maximum | Best of 3: 9.12, against 8.12 for the corpus maximum in context and 9.81 for the corpus top 8 read standalone (corpus p100.0; peak on 'ir', open). The weakest of the corpus top 8 is 7.28 in context, 7.66 standalone; neutral sentences max 0.04. | yes |
| 5. Near miss | 9.75 (corpus p100.0; 120% of the in-context maximum, 107% of the best blind maximum; peak on 'ir') | no |
| **Verdict** | A word detector, layer 4. The gate-open firings are all -ir in lair; the badger's lair scores higher than any blind sentence (9.75 against 9.12). Below the lair sentences, 78% of the top 50 are echo firings on -c, -face and race, so a reader going past the top 8 would see Two-Face and Clayface and call it villains. | **fails** |

Suggested label: **The word lair**

Strongest corpus sentences (activation, regime, peak token):

- 8.12 open 'ir' · *Batman & Robin*: Batman, Robin, and the police arrive at Freeze's lair in response to his escape, discovering Nora preserved in a cryogenic chamber and that Freeze has developed a cure f…
- 7.91 open 'ir' · *Return of the Joker*: He and Ace then rescue the unconscious Tim as the lair explodes.
- 7.72 open 'ir' · *1943 serial*: Batman and Robin infiltrate the gang's lair (also Dr. Daka's laboratory), hidden inside a still-open business, a Fun House ride.
- 7.66 open 'ir' · *Batman Forever*: As Bruce recovers, he and his butler, Alfred Pennyworth, deduce that Nygma is the Riddler through clues he left in his riddles, while Chase is held prisoner in the Riddl…
- 7.44 open 'ir' · *Superman*: With Otis and his girlfriend Eve Teschmacher, Lex retrieves it and traps Superman in his lair, revealing his plan to sink the western U.S., making his desert land prime…
- 7.41 open 'ir' · *Batman Forever*: At the Riddler's lair, Robin defeats Two-Face but chooses to spare him, which allows Two-Face to capture Robin at gunpoint.
- 7.34 open 'ir' · *Arkham Asylum (game)*: He learns from Ivy that the spores required to create an antidote are found exclusively in Killer Croc's lair in a sewer.
- 7.28 open 'ir' · *Arkham City*: Batman deduces that the therapeutic properties of Ra's al Ghul's blood can complete the cure and tracks one of his assassins to his underground lair, leading Batman into…

Echo dominates; strongest gate-open firings:

- 8.12 'ir' · *Batman & Robin*: Batman, Robin, and the police arrive at Freeze's lair in response to his escape, discovering Nora preserved in a cryogenic chamber and that Freeze has developed a cure f…
- 7.91 'ir' · *Return of the Joker*: He and Ace then rescue the unconscious Tim as the lair explodes.
- 7.72 'ir' · *1943 serial*: Batman and Robin infiltrate the gang's lair (also Dr. Daka's laboratory), hidden inside a still-open business, a Fun House ride.
- 7.66 'ir' · *Batman Forever*: As Bruce recovers, he and his butler, Alfred Pennyworth, deduce that Nygma is the Riddler through clues he left in his riddles, while Chase is held prisoner in the Riddl…
- 7.44 'ir' · *Superman*: With Otis and his girlfriend Eve Teschmacher, Lex retrieves it and traps Superman in his lair, revealing his plan to sink the western U.S., making his desert land prime…
- 7.41 'ir' · *Batman Forever*: At the Riddler's lair, Robin defeats Two-Face but chooses to spare him, which allows Two-Face to capture Robin at gunpoint.
- 7.34 'ir' · *Arkham Asylum (game)*: He learns from Ivy that the spores required to create an antidote are found exclusively in Killer Croc's lair in a sewer.
- 7.28 'ir' · *Arkham City*: Batman deduces that the therapeutic properties of Ra's al Ghul's blood can complete the cure and tracks one of his assassins to his underground lair, leading Batman into…

Relabel from gate-open firings: **The word lair (gate-open firings only; the echo adds -face, race)**

Negative pole L4·7177+, strongest sentences:

- 17.12 'acing' · *Batman '89*: The Joker begins terrorizing Gotham by lacing various hygiene products with "Smylex" – a deadly chemical that causes victims to die laughing.
- 13.12 'aced' · *White Knight*: He then drugs Batman's rogues gallery with drinks laced with Clayface's essence which he can use to control them by proxy.
- 9.94 ' lac' · *Mean Girls*: A fully recovered Regina joins the school's lacrosse team to channel her anger positively, Gretchen joins the "Cool Asians" clique, and Karen becomes the school's weathe…
- 8.88 'aced' · *Batman v Superman*: Superman arrives in Gotham and tries reasoning with Batman, who attacks him using kryptonite-laced gas.
- 2.42 ' lashes' · *Frozen*: Hans proposes to Anna, but Elsa objects to the marriage and lashes out, accidentally revealing her powers to the terrified court.
- 2.39 'clas' · *Die Hard*: McClane unclasps the watch, and Gruber falls to his death.
- 1.38 'aden' · *Batman & Robin*: Woodrue attempts to kill her by overturning a shelf of various toxins; instead, Isley is mutated by the toxins into Poison Ivy, who murders Woodrue with a venom-laden ki…
- 0.79 'AX' · *The Silence of the Lambs*: Crawford cross-references Lecter's profile with names from Johns Hopkins and the Known Offenders database, identifying Jame Gumb as the prime suspect; the name is confir…

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 9.81 (p100.0, open, peak 'ir') · *Return of the Joker*: He and Ace then rescue the unconscious Tim as the lair explodes.
- 9.50 (p100.0, open, peak 'ir') · *Batman & Robin*: Batman, Robin, and the police arrive at Freeze's lair in response to his escape, discovering Nora preserved in a cryogenic chamber and that Freeze has developed a cure f…
- 8.88 (p100.0, open, peak 'ir') · *Arkham Asylum (game)*: He learns from Ivy that the spores required to create an antidote are found exclusively in Killer Croc's lair in a sewer.
- 8.12 (p100.0, open, peak 'ir') · *1943 serial*: Batman and Robin infiltrate the gang's lair (also Dr. Daka's laboratory), hidden inside a still-open business, a Fun House ride.
- 7.88 (p99.9, open, peak 'ir') · *Arkham City*: Batman deduces that the therapeutic properties of Ra's al Ghul's blood can complete the cure and tracks one of his assassins to his underground lair, leading Batman into…
- 7.72 (p99.9, open, peak 'ir') · *Batman Forever*: As Bruce recovers, he and his butler, Alfred Pennyworth, deduce that Nygma is the Riddler through clues he left in his riddles, while Chase is held prisoner in the Riddl…
- 7.72 (p99.9, open, peak 'ir') · *Superman*: With Otis and his girlfriend Eve Teschmacher, Lex retrieves it and traps Superman in his lair, revealing his plan to sink the western U.S., making his desert land prime…
- 7.66 (p99.9, open, peak 'ir') · *Batman Forever*: At the Riddler's lair, Robin defeats Two-Face but chooses to spare him, which allows Two-Face to capture Robin at gunpoint.

Blind maximum attempts:

- 9.12 (p100.0, open, peak 'ir') blind_C: The villain's secret lair, a hidden underground hideout beneath an abandoned warehouse, serves as the headquarters where the mastermind and his henchmen plot their schemes.
- 8.38 (p100.0, open, peak 'ir') blind_B: The villain operates from a hidden lair in an abandoned amusement park, where henchmen guard the entrance and a captive is chained to a wall.
- 8.31 (p100.0, open, peak 'ir') blind_A: Deep inside the abandoned amusement park, the madman's secret lair glowed with flickering neon, its walls papered with maps and its doors rigged with traps.

Near miss (lair, villain): The badger's lair under the oak was mapped by naturalists, while the village pantomime cast its villain for December.

### 17. A bomb — L3·1967-

*object; present in 14 of 32 Batman works; family of 10. Paper's note: Time bombs, car bombs, threats to detonate.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 3 of 36 — in the first four layers, likely a token or spelling detector | no |
| 2. Sign regime | Top 50 firings: 35 gate-open, 15 echo, 0 other; echo share 30% (0 of the top 8; first echo at rank 36). Work scores (mean of the top 3 sentences) drawn from echo firings: 72% in Batman works, 99% in ordinary films; 23 of 32 Batman works and 69 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' bomb' ×26 (26/0), ' bombs' ×7 (7/0), ' up' ×7 (0/7), ' off' ×3 (0/3), ' down' ×3 (0/3), ' bombing' ×2 (2/0) | no |
| 3. Negative pole | L3·1967+, max 0.60. The opposite pole is weak (0.6); explodes appears at a twentieth of the strength. | – |
| 4. Blind maximum | Best of 3: 13.75, against 13.88 for the corpus maximum in context and 14.38 for the corpus top 8 read standalone (corpus p100.0; peak on ' bomb', open). The weakest of the corpus top 8 is 12.69 in context, 12.88 standalone; neutral sentences max 0.07. | would make the top 8 |
| 5. Near miss | 17.25 (corpus p100.0; 124% of the in-context maximum, 125% of the best blind maximum; peak on ' bomb') | no |
| **Verdict** | A word detector, layer 3. Gate-open firings are bomb, bombs, bombing; the echo (30%) supplies blew up, set off, down. A box-office bomb scores 17.25, above every real bomb in the corpus (13.88). | **fails** |

Suggested label: **The word bomb, any sense**

Strongest corpus sentences (activation, regime, peak token):

- 13.88 open ' bombing' · *The Godfather*: Meanwhile, Michael survives a hit in Sicily, which instead kills Apollonia via car bombing.
- 13.56 open ' bomb' · *Batman Forever*: Two-Face hijacks the event and threatens to detonate a bomb unless Batman reveals his identity.
- 13.38 open ' bomb' · *The Batman*: The Riddler abducts Colson, straps a timed collar bomb to his neck, and sends him to interrupt Mitchell's funeral.
- 13.38 open ' bomb' · *Under the Red Hood*: They chase Red Hood to a train station, where he escapes after detonating a bomb, which injures Nightwing.
- 13.31 open ' bomb' · *A Death in the Family*: The Joker beats Jason with a crowbar and restrains him and Sheila in the warehouse with a time bomb.
- 13.31 open ' bomb' · *The Batman*: Bruce's butler and caretaker, Alfred Pennyworth, is hospitalized after opening a letter bomb addressed to Bruce.
- 13.06 open ' bomb' · *Dark Knight Returns*: Soon after, Dent announces his intention to hold Gotham ransom with a bomb.
- 12.69 open ' bomb' · *Arkham Asylum (game)*: A bomb kills Young, and Joker steals the completed batches of Titan.

Negative pole L3·1967+, strongest sentences:

- 0.61 'ap' · *Count of Monte Cristo*: Shortly before Napoleon's escape from his exile on Elba in 1815, Edmond Dantès, the young first mate of the merchant ship Pharaon, honours his dying captain's last wish:…
- 0.59 ' explodes' · *Under the Red Hood*: Joker eventually leaves and locks Jason inside the warehouse with a bomb, which explodes and kills Jason before Batman arrives.
- 0.59 ' explode' · *Star Wars*: Guided by the voice of Obi-Wan's spirit, Luke uses the Force to aim his torpedoes into the exhaust port, causing the Death Star to explode moments before it can fire on…
- 0.58 'ap' · *Some Like It Hot*: The hotel hosts a conference for the Friends of Italian Opera Society, a front for a national Mafia meeting, presided over by Little Bonaparte—who already has a grudge a…
- 0.54 'ap' · *Some Like It Hot*: In male disguises, Joe and Jerry evade Spats' men by hiding under a table at the syndicate banquet, where Little Bonaparte has Spats and his gang gunned down.
- 0.50 ' brib' · *The Shawshank Redemption*: In 1963, Norton begins exploiting prison labor for public works, profiting by undercutting skilled labor costs and receiving bribes.
- 0.50 ' explode' · *The Dark Knight*: Having deduced Batman's feelings for Rachel, the Joker reveals she and Dent are being held separately in buildings rigged to explode.
- 0.48 ' explodes' · *Dark Knight Rises*: With no way to stop the detonation, Batman reveals his identity to Gordon, and uses his aerial craft, the Bat, to haul the bomb over the bay, where it safely explodes.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 14.38 (p100.0, open, peak ' bomb') · *A Death in the Family*: The Joker beats Jason with a crowbar and restrains him and Sheila in the warehouse with a time bomb.
- 14.19 (p100.0, open, peak ' bomb') · *The Batman*: Bruce's butler and caretaker, Alfred Pennyworth, is hospitalized after opening a letter bomb addressed to Bruce.
- 13.88 (p100.0, open, peak ' bombing') · *The Godfather*: Meanwhile, Michael survives a hit in Sicily, which instead kills Apollonia via car bombing.
- 13.88 (p100.0, open, peak ' bomb') · *The Batman*: The Riddler abducts Colson, straps a timed collar bomb to his neck, and sends him to interrupt Mitchell's funeral.
- 13.81 (p100.0, open, peak ' bomb') · *Under the Red Hood*: They chase Red Hood to a train station, where he escapes after detonating a bomb, which injures Nightwing.
- 13.56 (p99.9, open, peak ' bomb') · *Batman Forever*: Two-Face hijacks the event and threatens to detonate a bomb unless Batman reveals his identity.
- 13.12 (p99.8, open, peak ' bomb') · *Under the Red Hood*: Jason sets off a time bomb and Batman subdues Joker before attempting to save Jason.
- 12.88 (p99.8, open, peak ' bomb') · *Dark Knight Returns*: Soon after, Dent announces his intention to hold Gotham ransom with a bomb.

Blind maximum attempts:

- 13.75 (p100.0, open, peak ' bomb') blind_A: The timer on the bomb ticked down through red digits, wires taped to blocks of plastic explosive, thirty seconds before the whole building blew apart.
- 12.88 (p99.8, open, peak ' bomb') blind_B: The villain plants a bomb with a ticking timer beneath the crowded stadium, and the hero has only minutes to defuse it before it detonates.
- 12.19 (p99.7, open, peak ' bomb') blind_C: The terrorist planted a bomb, a ticking explosive device wired with a detonator and timer, set to blow up and explode in the crowded stadium.

Near miss (bomb): The musical was a box-office bomb despite its budget, and the producers blamed a poorly timed release date.

### 18. A funeral — L1·5621+

*trope; present in 9 of 32 Batman works; family of 1. Paper's note: Funerals and staged deaths: Bruce fleeing his parents' funeral, Batman's own faked funeral.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 1 of 36 — in the first four layers, likely a token or spelling detector | no |
| 2. Sign regime | Top 50 firings: 16 gate-open, 34 echo, 0 other; echo share 68% (0 of the top 8; first echo at rank 17). Work scores (mean of the top 3 sentences) drawn from echo firings: 88% in Batman works, 100% in ordinary films; 30 of 32 Batman works and 69 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' wedding' ×13 (0/13), ' funeral' ×9 (9/0), 'eral' ×2 (2/0), 'ement' ×2 (0/2), 'ane' ×2 (0/2), 'ennial' ×2 (0/2) | no |
| 3. Negative pole | L1·5621-, max 0.47. The opposite pole is noise: fragments that resemble the word (-eral in emeralds, -nel in Pimpernel). | – |
| 4. Blind maximum | Best of 3: 12.31, against 13.88 for the corpus maximum in context and 14.12 for the corpus top 8 read standalone (corpus p100.0; peak on ' funeral', open). The weakest of the corpus top 8 is 11.06 in context, 11.12 standalone; neutral sentences max 0.02. | would make the top 8 |
| 5. Near miss | 12.88 (corpus p100.0; 93% of the in-context maximum, 105% of the best blind maximum; peak on ' funeral') | no |
| **Verdict** | A word detector, layer 1, as the paper's own r = 0.98 with the word already suggested. All nine gate-open peaks in the top 50 are funeral; the 13 wedding firings are echo, so the apparent generalisation to ceremonies is the gate-closed artefact. A funeral home turned bakery scores as high as any funeral; mourners, coffin and grave without the word reach 20%. | **fails** |

Suggested label: **The word funeral**

Strongest corpus sentences (activation, regime, peak token):

- 13.88 open ' funeral' · *Batman v Superman*: In a flashback to his childhood, Bruce Wayne flees his parents' funeral.
- 12.12 open ' funeral' · *Dark Knight Returns*: After Wayne's funeral, it is revealed that his death was staged using an antiarrhythmic agent that suspended his vital life signs.
- 12.06 open ' funeral' · *The Batman*: The Riddler abducts Colson, straps a timed collar bomb to his neck, and sends him to interrupt Mitchell's funeral.
- 11.75 open ' funeral' · *Gotham by Gaslight*: At Sister Leslie's funeral, Bruce is met by psychiatrist Hugo Strange, claiming he knows the killer and wants to meet with Batman.
- 11.56 open ' funeral' · *Hush*: Dick Grayson returns to Gotham City for Elliot's funeral.
- 11.19 open ' funeral' · *Batman v Superman*: A memorial is held for Superman in Metropolis, while Bruce and Diana attend a funeral for Clark, also declared dead.
- 11.12 open ' funeral' · *Spider-Man*: At Norman's funeral, Harry vows revenge on Spider-Man, who he mistakenly believes killed his father.
- 11.06 open ' funeral' · *Dark Knight Returns*: Clark attends the funeral and winks at the disguised Carrie after hearing Wayne's heartbeat before she starts digging him out.

Echo dominates; strongest gate-open firings:

- 13.88 ' funeral' · *Batman v Superman*: In a flashback to his childhood, Bruce Wayne flees his parents' funeral.
- 12.12 ' funeral' · *Dark Knight Returns*: After Wayne's funeral, it is revealed that his death was staged using an antiarrhythmic agent that suspended his vital life signs.
- 12.06 ' funeral' · *The Batman*: The Riddler abducts Colson, straps a timed collar bomb to his neck, and sends him to interrupt Mitchell's funeral.
- 11.75 ' funeral' · *Gotham by Gaslight*: At Sister Leslie's funeral, Bruce is met by psychiatrist Hugo Strange, claiming he knows the killer and wants to meet with Batman.
- 11.56 ' funeral' · *Hush*: Dick Grayson returns to Gotham City for Elliot's funeral.
- 11.19 ' funeral' · *Batman v Superman*: A memorial is held for Superman in Metropolis, while Bruce and Diana attend a funeral for Clark, also declared dead.
- 11.12 ' funeral' · *Spider-Man*: At Norman's funeral, Harry vows revenge on Spider-Man, who he mistakenly believes killed his father.
- 11.06 ' funeral' · *Dark Knight Returns*: Clark attends the funeral and winks at the disguised Carrie after hearing Wayne's heartbeat before she starts digging him out.

Relabel from gate-open firings: **The word funeral (gate-open only; the echo adds wedding)**

Negative pole L1·5621-, strongest sentences:

- 0.47 'ies' · *Rear Window*: "Jeff" Jefferies, recuperating from assignment-related injuries, in a cast from his hip to his foot, is wheelchair-bound in his Greenwich Village apartment.
- 0.46 'eral' · *The Bat (1926)*: Gideon Bell, owner of the Favre Emeralds, receives a letter stating that the emeralds will be stolen at midnight by "the Bat", and that police will not be able to stop t…
- 0.46 'eral' · *The Bat (1926)*: The Bat, a figure dressed as a bat, murders Gideon and steals the emeralds.
- 0.43 'nel' · *Scarlet Pimpernel*: Meanwhile, the "League of the Scarlet Pimpernel", a secret society of twenty English aristocrats, "one to command, and nineteen to obey", is engaged in rescuing their Fr…
- 0.41 'ates' · *Alien*: Parker incinerates Ash's remains.
- 0.39 'nel' · *Scarlet Pimpernel*: Their leader, the mysterious Scarlet Pimpernel, takes his nom de guerre from the small, red wayside flower he draws on his messages.
- 0.37 'nel' · *Scarlet Pimpernel*: At a ball attended by the Blakeneys, Percy's verse about the "elusive Pimpernel" becomes an instant success.
- 0.36 'ets' · *Pride & Prejudice*: Jane recovers and, soon after, Mr Bennet's cousin Mr Collins, a pompous clergyman, visits the Bennets; as the closest male relative, Collins will inherit Longbourn as it…

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 14.12 (p100.0, open, peak ' funeral') · *Dark Knight Returns*: Clark attends the funeral and winks at the disguised Carrie after hearing Wayne's heartbeat before she starts digging him out.
- 13.88 (p100.0, open, peak ' funeral') · *Batman v Superman*: In a flashback to his childhood, Bruce Wayne flees his parents' funeral.
- 13.69 (p100.0, open, peak ' funeral') · *Gotham by Gaslight*: At Sister Leslie's funeral, Bruce is met by psychiatrist Hugo Strange, claiming he knows the killer and wants to meet with Batman.
- 12.69 (p100.0, open, peak ' funeral') · *Dark Knight Returns*: After Wayne's funeral, it is revealed that his death was staged using an antiarrhythmic agent that suspended his vital life signs.
- 12.56 (p100.0, open, peak ' funeral') · *Spider-Man*: At Norman's funeral, Harry vows revenge on Spider-Man, who he mistakenly believes killed his father.
- 12.12 (p99.9, open, peak ' funeral') · *The Batman*: The Riddler abducts Colson, straps a timed collar bomb to his neck, and sends him to interrupt Mitchell's funeral.
- 11.88 (p99.9, open, peak ' funeral') · *Batman v Superman*: A memorial is held for Superman in Metropolis, while Bruce and Diana attend a funeral for Clark, also declared dead.
- 11.12 (p99.8, open, peak ' funeral') · *Hush*: Dick Grayson returns to Gotham City for Elliot's funeral.

Blind maximum attempts:

- 12.31 (p100.0, open, peak ' funeral') blind_B: At the funeral, mourners in black gather at the graveside in the rain as the coffin is lowered and the priest delivers the eulogy.
- 12.31 (p100.0, open, peak ' funeral') blind_C: At the funeral the grieving mourners in black gathered around the coffin in the cemetery, listened to the eulogy, and wept as the casket was lowered into the grave.
- 2.83 (p99.7, open, peak ' coffin') blind_A: Rain fell on the black umbrellas as the mourners stood by the open grave, and the coffin was lowered slowly into the wet earth.

Near miss (funeral): The funeral home on Elm Street was converted into a bakery in 1998 and now sells sourdough.

### 19. Fell into a vat of chemicals — L0·6344-

*trope; present in 8 of 32 Batman works; family of 4. Paper's note: The Joker's chemical-plant origin: bleached skin, a fall into a vat.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 0 of 36 — in the first four layers, likely a token or spelling detector | no |
| 2. Sign regime | Top 50 firings: 50 gate-open, 0 echo, 0 other; echo share 0% (0 of the top 8; first echo at rank none in the top 50). Work scores (mean of the top 3 sentences) drawn from echo firings: 0% in Batman works, 4% in ordinary films; 0 of 32 Batman works and 1 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): 'ylum' ×20 (20/0), ' chemical' ×6 (6/0), 's' ×5 (5/0), ' emotional' ×4 (4/0), ' chemicals' ×2 (2/0), ' pharmaceutical' ×2 (2/0) | yes |
| 3. Negative pole | L0·6344+, max 0.16. The opposite pole is noise below 0.2. | – |
| 4. Blind maximum | Best of 3: 5.81, against 5.06 for the corpus maximum in context and 5.69 for the corpus top 8 read standalone (corpus p100.0; peak on ' chemicals', open). The weakest of the corpus top 8 is 3.70 in context, 4.00 standalone; neutral sentences max 0.22. | yes; also beats the standalone top 8 |
| 5. Near miss | 4.22 (corpus p99.9; 83% of the in-context maximum, 73% of the best blind maximum; peak on ' chemicals') | no |
| **Verdict** | A token detector, layer 0. The blind maximum beats the corpus maximum, but a brewery vat scrubbed with mild chemicals scores 73% of it; 20 of the top 50 peak on -ylum in Asylum and others on emotional and pharmaceutical, a layer-0 cluster of unrelated tokens. The fall into the vat is not read. | **fails** |

Suggested label: **The tokens chemical(s) and -ylum (Asylum)**

Strongest corpus sentences (activation, regime, peak token):

- 5.06 open ' chemicals' · *The Killing Joke*: He discovers to his shock that the chemicals have permanently bleached his skin chalk-white, stained his lips red and dyed his hair green.
- 5.03 open ' chemicals' · *Batman '89*: During a scuffle with Batman, Napier topples off a catwalk and falls into a vat of chemicals.
- 4.25 open ' chemical' · *The Killing Joke*: An unnamed man quits his job at a chemical company to become a stand-up comedian, only to fail miserably.
- 3.88 open ' chemical' · *The Killing Joke*: Desperate to support his pregnant wife, he agrees to guide two criminals through the chemical plant where he previously worked so that they can rob the playing card comp…
- 3.83 open 's' · *Batman '89*: Meanwhile, mob boss Carl Grissom sends his sociopathic second-in-command Jack Napier to break into Axis Chemicals and retrieve incriminating evidence.
- 3.80 open ' chemical' · *Spider-Man*: The chemical gives him superhuman strength but causes him to go insane and kill one of his scientists.
- 3.77 open 's' · *Batman '89*: Corrupt Gotham City police lieutenant Max Eckhardt arranges the hit on Napier by conducting an unauthorized raid on Axis Chemicals.
- 3.70 open 's' · *Arkham Knight*: After Batman escapes Ace Chemicals, he alerts Gordon to his daughter's kidnapping, revealing that she aids his vigilantism.

Negative pole L0·6344+, strongest sentences:

- 0.16 'tridges' · *Puss in Boots*: Once outfitted with boots, Puss sets out to catch game, such as rabbits and partridges, by using cunning traps.
- 0.14 ' Superman' · *Batman v Superman*: Bruce tells Diana that he regrets failing Superman and asks for her help in forming a team of metahumans to protect Earth in Superman's absence.
- 0.13 'use' · *Some Like It Hot*: Back at the hotel, Jerry announces that he has accepted Osgood's proposal of marriage, anticipating a divorce and cash settlement when the ruse is revealed.
- 0.13 ' Superman' · *Batman v Superman*: Believing Superman was complicit in the attack, Batman breaks into LexCorp and steals the kryptonite to weaponize it.
- 0.13 ' Wonderland' · *Arkham Asylum (comic)*: He turns to the stack of tarot cards and knocks them over, reciting a passage from Alice's Adventures in Wonderland: "Who cares for you?
- 0.13 ' Superman' · *Batman v Superman*: Clark Kent, Superman's civilian identity, seeks to condemn Batman's brutal methods in Daily Planet articles.
- 0.13 'use' · *The Princess Bride*: To maintain the ruse, Humperdinck has his enforcers organize a "brute squad" to clear out the nearby thieves' forest.
- 0.13 'ants' · *Joker (2019)*: As the show broadcasts live, Arthur tells morbid jokes, confesses to the subway murders, rants about societal neglect of the downtrodden, and berates Murray for ridiculi…

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 5.69 (p100.0, open, peak ' chemicals') · *The Killing Joke*: He discovers to his shock that the chemicals have permanently bleached his skin chalk-white, stained his lips red and dyed his hair green.
- 5.66 (p100.0, open, peak ' chemicals') · *Batman '89*: During a scuffle with Batman, Napier topples off a catwalk and falls into a vat of chemicals.
- 5.00 (p99.9, open, peak ' chemical') · *Spider-Man*: The chemical gives him superhuman strength but causes him to go insane and kill one of his scientists.
- 4.69 (p99.9, open, peak 's') · *Arkham Knight*: After Batman escapes Ace Chemicals, he alerts Gordon to his daughter's kidnapping, revealing that she aids his vigilantism.
- 4.25 (p99.9, open, peak ' chemical') · *The Killing Joke*: An unnamed man quits his job at a chemical company to become a stand-up comedian, only to fail miserably.
- 4.03 (p99.9, open, peak ' chemical') · *The Killing Joke*: Desperate to support his pregnant wife, he agrees to guide two criminals through the chemical plant where he previously worked so that they can rob the playing card comp…
- 4.00 (p99.9, open, peak 's') · *Batman '89*: Meanwhile, mob boss Carl Grissom sends his sociopathic second-in-command Jack Napier to break into Axis Chemicals and retrieve incriminating evidence.
- 4.00 (p99.9, open, peak 's') · *Batman '89*: Corrupt Gotham City police lieutenant Max Eckhardt arranges the hit on Napier by conducting an unauthorized raid on Axis Chemicals.

Blind maximum attempts:

- 5.81 (p100.0, open, peak ' chemicals') blind_C: During the fight at the chemical plant the criminal slipped over the railing and fell into a vat of toxic chemicals, which bleached his skin and disfigured him.
- 5.75 (p100.0, open, peak ' chemicals') blind_B: During the struggle at the chemical plant, the gangster slips from the catwalk and falls into a vat of toxic chemicals, which bleach his skin and disfigure his face.
- 5.38 (p100.0, open, peak ' chemicals') blind_A: He slipped from the catwalk and plunged into the bubbling vat of green chemicals below, surfacing with bleached white skin and hair turned emerald.

Near miss (vat, chemicals, fell): The brewery's oak vat was scrubbed with mild chemicals each autumn, and the price of barley fell that year.

### 20. Two-Face's coin — L26·3329-

*object; present in 9 of 32 Batman works; family of 4. Paper's note: Deciding fate with the flip of a scarred coin.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 26 of 36 | yes |
| 2. Sign regime | Top 50 firings: 47 gate-open, 3 echo, 0 other; echo share 6% (0 of the top 8; first echo at rank 45). Work scores (mean of the top 3 sentences) drawn from echo firings: 69% in Batman works, 84% in ordinary films; 24 of 32 Batman works and 61 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): 'ace' ×10 (10/0), ' of' ×4 (4/0), ' coin' ×3 (3/0), ',' ×2 (2/0), ' "' ×2 (2/0), ' ' ×2 (2/0) | yes |
| 3. Negative pole | L26·3329+, max 4.78. The opposite pole peaks once on -sided (the two-sided coin) and is otherwise noise: the same coin seen from the other side. | – |
| 4. Blind maximum | Best of 3: 32.25, against 24.88 for the corpus maximum in context and 42.50 for the corpus top 8 read standalone (corpus p100.0; peak on ' or', open). The weakest of the corpus top 8 is 14.94 in context, 0.33 standalone; neutral sentences max 0.44. | yes |
| 5. Near miss | 4.84 (corpus p99.5; 20% of the in-context maximum, 15% of the best blind maximum; peak on ' both') | yes |
| **Verdict** | Survives: the one label of 31 that does. A name-free coin-flip sentence (32.3) beats the in-context corpus maximum (24.9); the coin-and-two-faced-clock near miss scores 15% of it; echo share 6%. Peaks fall on the phrasing of the flip (or, one, side, of, flips) rather than on the word coin, and 10 of the top 50 are the name Two-Face. Read standalone the corpus top 8 reach 42.5, so even here the blind sentence does not beat every corpus sentence read the same way. | **survives** |

Suggested label: **A coin flip deciding a fate (Two-Face's coin); keep**

Strongest corpus sentences (activation, regime, peak token):

- 24.88 open ' coin' · *Arkham Asylum (comic)*: Two-Face declares that they will kill Batman if the coin lands scratched side up and let him go if the unscarred side lands.
- 21.00 open ' it' · *Arkham Asylum (comic)*: Two-Face looks at the coin and it is revealed that it actually landed scratched side up, implying that he made his own choice.
- 20.00 open ' of' · *Batman Forever*: The trauma of the incident caused Dent to develop a split personality, make decisions based on the flip of a coin, and swear vengeance against Batman for failing to prev…
- 19.38 open ' flips' · *Batman Forever*: Two-Face corners them and flips his coin to decide their fate, but loses his balance after Batman throws multiple identical coins in the air, and falls to his death.
- 18.00 open ' one' · *Batman Forever*: Flashbacks reveal that Dent, once one of Batman's biggest supporters, was horribly disfigured with acid on one side of his face by mobster Sal Maroni.
- 17.75 open ' of' · *Arkham Asylum (comic)*: At the asylum, Batman discovers that Two-Face's mental condition has deteriorated as a result of Adams' therapy; she replaced Two-Face's trademark coin with a six-sided…
- 15.56 open ' declares' · *Arkham Asylum (comic)*: He flips the coin and declares Batman free.
- 14.94 open ' coin' · *The Dark Knight*: He flips his coin to decide their fates, but Batman tackles him to save Gordon's son, and Dent falls to his death.

Negative pole L26·3329+, strongest sentences:

- 4.78 '-sided' · *Arkham Asylum (comic)*: At the asylum, Batman discovers that Two-Face's mental condition has deteriorated as a result of Adams' therapy; she replaced Two-Face's trademark coin with a six-sided…
- 1.70 ' stone' · *The Shawshank Redemption*: An irate Norton throws a stone at a poster of Raquel Welch hanging on the cell wall, revealing a tunnel that Andy had dug with his rock hammer over 19 years.
- 1.66 'September' · *Knightfall*: Within the regular series, the buildup to "Knightfall" begins with a six-issue run in Batman #484-489 (September 1992 - February 1993), in which Batman (at the onset of…
- 1.62 ' enter' · *2001: A Space Odyssey*: Concerned about HAL's behaviour, Bowman and Poole enter an EVA pod so they can talk in private without HAL overhearing.
- 1.48 ' Columbia' · *Spider-Man*: On a high school field trip, Peter Parker visits a Columbia University genetics laboratory with his best friend, Harry Osborn, and his crush, Mary Jane Watson.
- 1.36 ',' · *Casablanca*: As they walk away into the fog, Rick says, "Louis, I think this is the beginning of a beautiful friendship."
- 1.30 ' a' · *Batman & Robin*: During a confrontation at the natural history museum, Freeze steals a diamond and flees, freezing Robin to distract Batman.
- 1.20 ' the' · *Star Wars*: Leia is taken prisoner, but the droids R2-D2 and C-3PO escape with the plans, crash-landing on the nearby desert planet of Tatooine.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 42.50 (p100.0, open, peak ' lands') · *Arkham Asylum (comic)*: Two-Face declares that they will kill Batman if the coin lands scratched side up and let him go if the unscarred side lands.
- 29.50 (p100.0, open, peak ' landed') · *Arkham Asylum (comic)*: Two-Face looks at the coin and it is revealed that it actually landed scratched side up, implying that he made his own choice.
- 24.25 (p100.0, open, peak ' coin') · *Arkham Asylum (comic)*: He flips the coin and declares Batman free.
- 22.25 (p100.0, open, peak ' coin') · *The Dark Knight*: He flips his coin to decide their fates, but Batman tackles him to save Gordon's son, and Dent falls to his death.
- 21.38 (p100.0, open, peak ' a') · *Batman Forever*: The trauma of the incident caused Dent to develop a split personality, make decisions based on the flip of a coin, and swear vengeance against Batman for failing to prev…
- 16.50 (p99.8, open, peak ' coin') · *Arkham Asylum (comic)*: At the asylum, Batman discovers that Two-Face's mental condition has deteriorated as a result of Adams' therapy; she replaced Two-Face's trademark coin with a six-sided…
- 14.50 (p99.8, open, peak ' to') · *Batman Forever*: Two-Face corners them and flips his coin to decide their fate, but loses his balance after Batman throws multiple identical coins in the air, and falls to his death.
- 0.33 (p44.7, echo, peak ' Mar') · *Batman Forever*: Flashbacks reveal that Dent, once one of Batman's biggest supporters, was horribly disfigured with acid on one side of his face by mobster Sal Maroni.

Blind maximum attempts:

- 32.25 (p100.0, open, peak ' or') blind_C: The scarred villain flips his two-headed silver coin, one side scratched and defaced, and lets the coin toss, heads or tails, decide whether his victim lives or dies.
- 22.75 (p100.0, open, peak ' one') blind_B: The scarred villain flips his two-headed silver coin, one side scratched and defaced, to decide whether his victim lives or dies.
- 22.12 (p100.0, open, peak ' side') blind_A: The scarred man flipped his silver dollar, one side clean and one side gouged, and let the coin decide whether the hostage lived or died.

Near miss (coin, two-faced): The vending machine took one coin, and the two-faced clock above the station concourse showed both platforms' times.

### 21. Harvey Dent — L5·6648+

*character; present in 11 of 32 Batman works; family of 15. Paper's note: The district attorney before and after he becomes Two-Face.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 5 of 36 | yes |
| 2. Sign regime | Top 50 firings: 25 gate-open, 25 echo, 0 other; echo share 50% (0 of the top 8; first echo at rank 21). Work scores (mean of the top 3 sentences) drawn from echo firings: 86% in Batman works, 94% in ordinary films; 30 of 32 Batman works and 67 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Harvey' ×11 (11/0), 'ap' ×4 (4/0), 'gon' ×3 (0/3), ' silent' ×3 (0/3), 'March' ×2 (2/0), 'arte' ×2 (2/0) | no |
| 3. Negative pole | L5·6648-, max 0.42. The opposite pole is noise (Notting, Humperdinck). | – |
| 4. Blind maximum | Best of 3: 0.24, against 10.38 for the corpus maximum in context and 10.12 for the corpus top 8 read standalone (corpus p97.8; peak on ' knight', echo). The weakest of the corpus top 8 is 7.56 in context, 7.25 standalone; neutral sentences max 0.13. | no |
| 5. Near miss | 13.00 (corpus p100.0; 125% of the in-context maximum, 5485% of the best blind maximum; peak on 'vey') | no |
| **Verdict** | Fails. Every gate-open peak is the first name Harvey; Harvey the retired postman's dented gate scores 13.0, above the strongest Harvey Dent sentence (10.4), and a name-free acid-scarred district attorney scores 0.24. Half the top 50 are echo on junk tokens. | **fails** |

Suggested label: **The first name Harvey**

Strongest corpus sentences (activation, regime, peak token):

- 10.38 open ' Harvey' · *Hush*: In the ensuing confrontation, Elliot is shot by the reborn Harvey Dent and plummets into the water, with Batman never having a chance to unmask him (it is only assumed i…
- 10.19 open ' Harvey' · *Dark Knight Returns*: Batman foils a group of criminals working for Harvey Dent.
- 8.81 open ' Harvey' · *The Long Halloween*: Batman meets with district attorney Harvey Dent and police captain Jim Gordon.
- 8.75 open ' Harvey' · *Earth One*: As a teenager, Bruce befriends his classmate, Jessica Dent, and develops a rivalry with her twin brother, Harvey.
- 8.69 open ' Harvey' · *The Dark Knight*: The vigilante Batman, district attorney Harvey Dent, and police lieutenant Jim Gordon ally to eliminate Gotham's organized crime.
- 8.56 open ' Harvey' · *Gotham by Gaslight*: Selina Kyle, a famed local actress, berates Gotham Police Commissioner James Gordon and Chief of Police Harvey Bullock for their failure to stop the Ripper murders.
- 7.94 open ' Harvey' · *Batman Forever*: In Gotham City, Batman defuses a hostage situation orchestrated by the criminal Two-Face, formerly district attorney Harvey Dent, who escapes.
- 7.56 open ' Harvey' · *Dark Knight Rises*: Meanwhile, eight years after the death of Gotham City district attorney Harvey Dent and the arrest of the Joker, organized crime has been eradicated in Gotham by the Den…

Negative pole L5·6648-, strongest sentences:

- 0.42 'ting' · *Notting Hill*: William Thacker owns a travel book store in Notting Hill, London.
- 0.37 ' Legion' · *Vertigo*: Scottie follows Madeleine to the grave of Carlotta Valdes, who died in 1857, at the Mission San Francisco de Asís, and to the Legion of Honor art museum, where she gazes…
- 0.37 'ck' · *The Princess Bride*: He absconds with Buttercup ahead of Humperdinck's party.
- 0.36 'ck' · *The Princess Bride*: The couple then flees into the dangerous Fire Swamp to avoid Humperdinck's party.
- 0.36 'ck' · *The Princess Bride*: A masked man in black pursues them across the sea, as do Humperdinck and his soldiers.
- 0.35 'mond' · *Count of Monte Cristo*: Shortly before Napoleon's escape from his exile on Elba in 1815, Edmond Dantès, the young first mate of the merchant ship Pharaon, honours his dying captain's last wish:…
- 0.35 ' notes' · *Up*: After Muntz shows them the bird skeleton, Russell notes its resemblance to Kevin.
- 0.35 'ck' · *The Princess Bride*: Five years later, Buttercup, still grief-stricken, is forcibly betrothed to Florin's arrogant Prince Humperdinck.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 10.12 (p99.9, open, peak ' Harvey') · *Hush*: In the ensuing confrontation, Elliot is shot by the reborn Harvey Dent and plummets into the water, with Batman never having a chance to unmask him (it is only assumed i…
- 9.12 (p99.9, open, peak ' Harvey') · *Dark Knight Returns*: Batman foils a group of criminals working for Harvey Dent.
- 7.97 (p99.8, open, peak ' Harvey') · *Batman Forever*: In Gotham City, Batman defuses a hostage situation orchestrated by the criminal Two-Face, formerly district attorney Harvey Dent, who escapes.
- 7.94 (p99.8, open, peak ' Harvey') · *The Dark Knight*: The vigilante Batman, district attorney Harvey Dent, and police lieutenant Jim Gordon ally to eliminate Gotham's organized crime.
- 7.94 (p99.8, open, peak ' Harvey') · *Dark Knight Rises*: Meanwhile, eight years after the death of Gotham City district attorney Harvey Dent and the arrest of the Joker, organized crime has been eradicated in Gotham by the Den…
- 7.81 (p99.8, open, peak ' Harvey') · *The Long Halloween*: Batman meets with district attorney Harvey Dent and police captain Jim Gordon.
- 7.53 (p99.8, open, peak ' Harvey') · *Earth One*: As a teenager, Bruce befriends his classmate, Jessica Dent, and develops a rivalry with her twin brother, Harvey.
- 7.25 (p99.8, open, peak ' Harvey') · *Gotham by Gaslight*: Selina Kyle, a famed local actress, berates Gotham Police Commissioner James Gordon and Chief of Police Harvey Bullock for their failure to stop the Ripper murders.

Blind maximum attempts:

- 0.24 (p97.8, echo, peak ' knight') blind_B: The crusading district attorney, hailed as the city's incorruptible white knight, has half his face burned away by acid and turns against the justice system he once served.
- 0.22 (p96.8, echo, peak 'enge') blind_A: The handsome, crusading district attorney, once the city's white knight, had half his face burned away with acid and became a scarred, vengeful killer.
- 0.20 (p95.0, echo, peak ' knight') blind_C: The crusading district attorney, the city's handsome white knight prosecutor, has half his face scarred by acid and becomes a split-personality villain obsessed with duality and chance.

Near miss (Harvey, dent): Harvey, a retired postman, noticed a small dent in the garden gate after the delivery van reversed into it.

### 22. Selina Kyle — L1·1051+

*character; present in 7 of 32 Batman works; family of 40. Paper's note: The cat burglar and Bruce's romantic foil.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 1 of 36 — in the first four layers, likely a token or spelling detector | no |
| 2. Sign regime | Top 50 firings: 31 gate-open, 19 echo, 0 other; echo share 38% (0 of the top 8; first echo at rank 32). Work scores (mean of the top 3 sentences) drawn from echo firings: 84% in Batman works, 99% in ordinary films; 28 of 32 Batman works and 68 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Sel' ×29 (29/0), 'ansom' ×8 (0/8), 'z' ×4 (0/4), 'el' ×2 (2/0), 'asz' ×2 (0/2), 'se' ×2 (0/2) | no |
| 3. Negative pole | L1·1051-, max 0.41. The opposite pole is noise (-el fragments). | – |
| 4. Blind maximum | Best of 3: 0.02, against 2.55 for the corpus maximum in context and 5.22 for the corpus top 8 read standalone (corpus p86.4; peak on 'ation', echo). The weakest of the corpus top 8 is 1.88 in context, 0.01 standalone; neutral sentences max 0.04. | no |
| 5. Near miss | 1.45 (corpus p99.4; 57% of the in-context maximum, 6568% of the best blind maximum; peak on ' Sel') | no |
| **Verdict** | Fails. 29 of the top 50 peak on Sel- in Selina (the rest are echo on ransom and Zsasz); a name-free cat burglar scores 0.02. Selina the choir soprano scores 1.45, 28% of the standalone maximum, so even the name needs the Batman context to fire fully. (The first near miss began with the name, which the first-token exclusion drops; it was reworded and re-run.) | **fails** |

Suggested label: **The token Sel- (Selina) in a Batman context**

Strongest corpus sentences (activation, regime, peak token):

- 2.55 open ' Sel' · *Dark Knight Rises*: Cat burglar Selina Kyle steals Bruce's fingerprints from Wayne Manor for Daggett, but he double-crosses her, and she alerts the police.
- 2.33 open ' Sel' · *The Batman*: Annika disappears; while searching for answers, Batman and Selina discover that Savage and Mitchell were on Falcone's payroll, as is district attorney Gil Colson.
- 2.28 open ' Sel' · *Gotham by Gaslight*: At the Monarch Theatre with his friend, District Attorney Harvey Dent, Bruce meets Selina and the two bond at the exclusive Dionysus Club.
- 2.05 open ' Sel' · *The Long Halloween*: It is not until Saint Patrick's Day that Selina Kyle figures out what has happened to him and, as Catwoman, frees him from Poison Ivy's clutches.
- 2.00 open ' Sel' · *Dark Knight Rises*: Batman encounters Selina, and both narrowly escape Bane's henchmen.
- 1.99 open ' Sel' · *Hush*: Later, in Gotham City, Bruce Wayne, Selina Kyle, Leslie Thompkins, and Dr. Elliot are attending the opera Pagliacci, when Harley Quinn tries to rob everyone in the theat…
- 1.91 open ' Sel' · *Gotham by Gaslight*: Selina Kyle, a famed local actress, berates Gotham Police Commissioner James Gordon and Chief of Police Harvey Bullock for their failure to stop the Ripper murders.
- 1.88 open ' Sel' · *Dark Knight Rises*: Selina agrees to take Batman to Bane, but instead leads him into a trap under Wayne Tower.

Negative pole L1·1051-, strongest sentences:

- 0.41 'gel' · *Sherlock Holmes*: He attempted to chase Drebber with a cudgel but claimed to have lost sight of him.
- 0.26 'el' · *Scarlet Pimpernel*: At a ball attended by the Blakeneys, Percy's verse about the "elusive Pimpernel" becomes an instant success.
- 0.23 'el' · *Alien*: Dallas, Kane, and Lambert discover that the transmission comes from a derelict alien vessel.
- 0.22 'el' · *Dark Knight Returns*: Gordon's successor as commissioner, Captain Ellen Yindel, declares Batman a wanted criminal.
- 0.22 'old' · *Knightfall*: Bane, introduced in Batman: Vengeance of Bane #1 (January 1993) by Chuck Dixon and Graham Nolan, an orphan born and raised in a Central American island prison, self-taug…
- 0.21 'El' · *Superman*: Before the planet's destruction, Jor-El and his wife Lara send their infant son Kal-El to Earth.
- 0.20 'ist' · *Ghostbusters*: Following a paranormal encounter in her apartment, cellist Dana Barrett visits the Ghostbusters.
- 0.20 'El' · *Superman*: Kal-El's spaceship touches down near Smallville, Kansas.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 5.22 (p100.0, open, peak ' Sel') · *Dark Knight Rises*: Cat burglar Selina Kyle steals Bruce's fingerprints from Wayne Manor for Daggett, but he double-crosses her, and she alerts the police.
- 3.11 (p100.0, open, peak ' Sel') · *The Batman*: Annika disappears; while searching for answers, Batman and Selina discover that Savage and Mitchell were on Falcone's payroll, as is district attorney Gil Colson.
- 2.36 (p100.0, open, peak ' Sel') · *Gotham by Gaslight*: At the Monarch Theatre with his friend, District Attorney Harvey Dent, Bruce meets Selina and the two bond at the exclusive Dionysus Club.
- 2.34 (p100.0, open, peak ' Sel') · *The Long Halloween*: It is not until Saint Patrick's Day that Selina Kyle figures out what has happened to him and, as Catwoman, frees him from Poison Ivy's clutches.
- 2.34 (p100.0, open, peak ' Sel') · *Dark Knight Rises*: Batman encounters Selina, and both narrowly escape Bane's henchmen.
- 2.17 (p99.9, open, peak ' Sel') · *Hush*: Later, in Gotham City, Bruce Wayne, Selina Kyle, Leslie Thompkins, and Dr. Elliot are attending the opera Pagliacci, when Harley Quinn tries to rob everyone in the theat…
- 0.01 (p55.6, echo, peak ' Police') · *Gotham by Gaslight*: Selina Kyle, a famed local actress, berates Gotham Police Commissioner James Gordon and Chief of Police Harvey Bullock for their failure to stop the Ripper murders.
- 0.01 (p52.8, echo, peak ' instead') · *Dark Knight Rises*: Selina agrees to take Batman to Bane, but instead leads him into a trap under Wayne Tower.

Blind maximum attempts:

- 0.02 (p86.4, echo, peak 'ation') blind_A: The slinky cat burglar in the black leather catsuit cracked her whip, stole the diamond necklace, and purred a flirtation at the vigilante chasing her.
- 0.02 (p85.7, echo, peak 'eline') blind_C: The sleek cat burglar in a skintight catsuit with a whip and claws, a feline antiheroine and jewel thief, flirts with the caped vigilante while stealing diamonds.
- 0.01 (p67.3, echo, peak ' from') blind_B: A cat burglar in a skintight black catsuit and cat-eared cowl, armed with a whip, steals jewels from the wealthy and carries on a flirtatious romance with the vigilante.

Near miss (Selina, Kyle): The choir's soprano, Selina, and the organist, Kyle, rehearsed the Christmas programme every Thursday evening.

### 23. The Riddler — L1·10818-

*character; present in 10 of 32 Batman works; family of 51. Paper's note: Edward Nygma and his riddles and puzzles.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 1 of 36 — in the first four layers, likely a token or spelling detector | no |
| 2. Sign regime | Top 50 firings: 29 gate-open, 21 echo, 0 other; echo share 42% (0 of the top 8; first echo at rank 28). Work scores (mean of the top 3 sentences) drawn from echo firings: 83% in Batman works, 99% in ordinary films; 28 of 32 Batman works and 68 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): 'ler' ×26 (26/0), 'ok' ×5 (0/5), 'xs' ×3 (0/3), ' off' ×2 (0/2), 'iddles' ×1 (1/0), ' ways' ×1 (1/0) | no |
| 3. Negative pole | L1·10818+, max 8.56. The opposite pole fires on word endings (-room in ballroom, -way in Runway): sub-word noise. | – |
| 4. Blind maximum | Best of 3: 4.22, against 11.25 for the corpus maximum in context and 12.50 for the corpus top 8 read standalone (corpus p99.3; peak on 'iddle', open). The weakest of the corpus top 8 is 10.12 in context, 10.56 standalone; neutral sentences max 0.20. | no |
| 5. Near miss | 4.44 (corpus p99.3; 39% of the in-context maximum, 105% of the best blind maximum; peak on 'iddle') | no |
| **Verdict** | Fails. All 26 gate-open peaks are -ler in Riddler. A name-free riddle-leaving villain reaches 37% of the maximum, and so does a barn riddled with holes (4.44 against 4.22). Layer 1. | **fails** |

Suggested label: **The token -ler in Riddler (and riddl-)**

Strongest corpus sentences (activation, regime, peak token):

- 11.25 open 'ler' · *The Batman*: On Halloween, Gotham City mayor Don Mitchell Jr. is murdered by a masked serial killer, the Riddler.
- 10.50 open 'ler' · *The Batman*: Reclusive billionaire Bruce Wayne, who has operated for two years as the vigilante Batman, investigates the murder alongside Lieutenant Jim Gordon, much to the disdain o…
- 10.31 open 'ler' · *Batman Forever*: At the Riddler's lair, Robin defeats Two-Face but chooses to spare him, which allows Two-Face to capture Robin at gunpoint.
- 10.31 open 'ler' · *The Batman*: The following night, the Riddler kills the police commissioner, Pete Savage, and leaves another message for Batman.
- 10.25 open 'ler' · *Arkham City*: Returning villains include the Riddler (Wally Wingert), Victor Zsasz (Danny Jacobs), Bane (Fred Tatasciore), and Poison Ivy (Tasia Valenza).
- 10.19 open 'ler' · *The Long Halloween*: The Riddler becomes the first target to be spared by Holiday on April Fool's Day, which Batman comes to suspect as being a message from Holiday to Falcone.
- 10.12 open 'ler' · *Batman '66*: Batman and Robin head back to Commissioner Gordon's office, where they deduce that the tip was a set-up by the United Underworld, a gathering of four of the most powerfu…
- 10.12 open 'ler' · *The Batman*: The Riddler abducts Colson, straps a timed collar bomb to his neck, and sends him to interrupt Mitchell's funeral.

Negative pole L1·10818+, strongest sentences:

- 8.56 'room' · *The Shining*: Jack defiantly sulks in the ballroom, where ghostly bartender Lloyd entices him back to drinking.
- 8.25 'room' · *The Sound of Music*: During a grand party at the villa, where guests in formal attire waltz in the ballroom, Maria and the children look on from the garden terrace.
- 8.00 'room' · *The Shining*: Returning to the ballroom, Jack finds it filled with ghostly figures, including waiter Delbert Grady, who urges Jack to "correct" his wife and child.
- 7.62 'room' · *The Bat (1926)*: The candles in the ballroom go out when lit, and a shape appears to float towards Anderson and Lizzie, but it turns out to be the Japanese butler Billy carrying a lamp.
- 7.28 'room' · *The Bat (1926)*: Dr. Wells searches for the hidden room by knocking on walls, which causes the others to investigate the sound, leading them to a ballroom which is supposedly haunted.
- 6.19 'way' · *The Devil Wears Prada*: Andy spends the night with an attractive writer, Christian Thompson, who tells her that Jacqueline Follet (Miranda's counterpart as editor-in-chief at French Runway) is…
- 5.97 'way' · *The Devil Wears Prada*: Despite her lack of knowledge of the fashion industry, she is hired as a junior personal assistant to Miranda Priestly, the cruel editor-in-chief of Runway magazine in N…
- 5.97 'way' · *The Devil Wears Prada*: Andy resolves to tolerate Miranda's abusive treatment until she can use her connections from Runway to find a journalism-focused job.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 12.50 (p100.0, open, peak 'ler') · *Batman Forever*: At the Riddler's lair, Robin defeats Two-Face but chooses to spare him, which allows Two-Face to capture Robin at gunpoint.
- 11.81 (p100.0, open, peak 'ler') · *The Long Halloween*: The Riddler becomes the first target to be spared by Holiday on April Fool's Day, which Batman comes to suspect as being a message from Holiday to Falcone.
- 11.81 (p100.0, open, peak 'ler') · *The Batman*: The Riddler abducts Colson, straps a timed collar bomb to his neck, and sends him to interrupt Mitchell's funeral.
- 11.25 (p100.0, open, peak 'ler') · *The Batman*: On Halloween, Gotham City mayor Don Mitchell Jr. is murdered by a masked serial killer, the Riddler.
- 11.19 (p100.0, open, peak 'ler') · *Arkham City*: Returning villains include the Riddler (Wally Wingert), Victor Zsasz (Danny Jacobs), Bane (Fred Tatasciore), and Poison Ivy (Tasia Valenza).
- 11.06 (p100.0, open, peak 'ler') · *The Batman*: When Batman arrives, the Riddler calls him via a phone duct-taped to Colson and threatens to detonate the bomb if Colson cannot answer three riddles.
- 11.00 (p100.0, open, peak 'ler') · *The Batman*: The following night, the Riddler kills the police commissioner, Pete Savage, and leaves another message for Batman.
- 10.56 (p100.0, open, peak 'ler') · *The Batman*: Reclusive billionaire Bruce Wayne, who has operated for two years as the vigilante Batman, investigates the murder alongside Lieutenant Jim Gordon, much to the disdain o…

Blind maximum attempts:

- 4.22 (p99.3, open, peak 'iddle') blind_A: The gaunt man in the green bowler hat and question-mark suit left a taunting riddle at every crime scene, daring the detective to solve it.
- 2.78 (p99.3, open, peak 'iddles') blind_C: The green-suited villain in a bowler hat covered in question marks taunts the detective with riddles, puzzles, cryptic clues and brain-teasers left at every crime scene.
- 2.45 (p99.3, open, peak 'iddles') blind_B: A criminal mastermind in a green suit covered with question marks leaves taunting riddles and puzzles at each crime scene, daring the detective to solve them.

Near miss (riddled, riddle): The old barn roof was riddled with holes, and the farmer used a wire riddle to sift the seed potatoes.

### 24. Jason Todd — L4·4504-

*character; present in 8 of 32 Batman works; family of 18. Paper's note: The second Robin, his death at the Joker's hands and his return.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 4 of 36 | yes |
| 2. Sign regime | Top 50 firings: 15 gate-open, 35 echo, 0 other; echo share 70% (0 of the top 8; first echo at rank 14). Work scores (mean of the top 3 sentences) drawn from echo firings: 88% in Batman works, 96% in ordinary films; 30 of 32 Batman works and 68 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Todd' ×8 (8/0), ' fle' ×8 (0/8), ' Elliot' ×4 (0/4), 'roy' ×2 (2/0), ' trolls' ×2 (2/0), 'ri' ×2 (0/2) | no |
| 3. Negative pole | L4·4504+, max 0.41. The opposite pole is noise (tornado, Toto, topples: to- onsets). | – |
| 4. Blind maximum | Best of 3: 0.08, against 20.25 for the corpus maximum in context and 20.88 for the corpus top 8 read standalone (corpus p85.1; peak on ' crow', echo). The weakest of the corpus top 8 is 12.62 in context, 16.62 standalone; neutral sentences max 0.10. | no |
| 5. Near miss | 12.50 (corpus p99.8; 62% of the in-context maximum, 14881% of the best blind maximum; peak on ' Todd') | no |
| **Verdict** | Fails. Fires on the token Todd: two brothers named Jason and Todd from Leeds score 12.5, a name-free second sidekick murdered with a crowbar 0.08. Echo fills 70% of the top 50 below the Todd sentences (flee, Elliot). | **fails** |

Suggested label: **The name Todd**

Strongest corpus sentences (activation, regime, peak token):

- 20.25 open ' Todd' · *A Death in the Family*: While eavesdropping on a child pornography ring and awaiting police backup in Gotham City, Jason Todd (Robin) ignores Batman's orders and attacks the criminals.
- 19.88 open ' Todd' · *White Knight*: When the Joker's obsession with Batman culminated in Jason Todd's torture, Quinzel left him to try and help save Todd's life.
- 19.50 open ' Todd' · *Knightfall*: In addition, since the events from A Death in the Family, Batman struggles with the trauma of the loss of his protégé Jason Todd.
- 19.50 open ' Todd' · *Dark Knight Returns*: Over a decade after the United Nations outlawed superheroes and the Joker murdered Batman's protégé, Robin (Jason Todd), a 55-year-old Bruce Wayne battles alcoholism in…
- 19.12 open ' Todd' · *Hush*: He discovers that the current Robin, Tim Drake, has been captured by the former second Robin, Jason Todd, who had previously perished.
- 19.00 open ' Todd' · *Under the Red Hood*: He realizes his mistake when he is unable to control Joker, who captures Jason Todd, the second Robin and Batman's partner.
- 16.62 open ' Todd' · *Knightfall*: The rescue of Mayor Krol from the teaming of the Joker and Scarecrow pushes Batman to his mental and physical limits: a dose of Scarecrow's fear gas makes him relive the…
- 12.62 open ' Todd' · *A Death in the Family*: Her first initial is "S", not "C" as in Catherine Todd, the woman he knew as his mother.

Echo dominates; strongest gate-open firings:

- 20.25 ' Todd' · *A Death in the Family*: While eavesdropping on a child pornography ring and awaiting police backup in Gotham City, Jason Todd (Robin) ignores Batman's orders and attacks the criminals.
- 19.88 ' Todd' · *White Knight*: When the Joker's obsession with Batman culminated in Jason Todd's torture, Quinzel left him to try and help save Todd's life.
- 19.50 ' Todd' · *Knightfall*: In addition, since the events from A Death in the Family, Batman struggles with the trauma of the loss of his protégé Jason Todd.
- 19.50 ' Todd' · *Dark Knight Returns*: Over a decade after the United Nations outlawed superheroes and the Joker murdered Batman's protégé, Robin (Jason Todd), a 55-year-old Bruce Wayne battles alcoholism in…
- 19.12 ' Todd' · *Hush*: He discovers that the current Robin, Tim Drake, has been captured by the former second Robin, Jason Todd, who had previously perished.
- 19.00 ' Todd' · *Under the Red Hood*: He realizes his mistake when he is unable to control Joker, who captures Jason Todd, the second Robin and Batman's partner.
- 16.62 ' Todd' · *Knightfall*: The rescue of Mayor Krol from the teaming of the Joker and Scarecrow pushes Batman to his mental and physical limits: a dose of Scarecrow's fear gas makes him relive the…
- 12.62 ' Todd' · *A Death in the Family*: Her first initial is "S", not "C" as in Catherine Todd, the woman he knew as his mother.

Negative pole L4·4504+, strongest sentences:

- 0.41 ' tornado' · *The Wizard of Oz*: She returns just as a tornado approaches the farm.
- 0.41 ' Toy' · *Big*: With Billy's help, Josh learns that it will take at least six weeks to find the Zoltar machine again, so Josh rents a room in a flophouse in New York City and gets a job…
- 0.41 'enuous' · *Gravity*: They try to grab onto the ISS, but only make tenuous connection to the Soyuz's parachute cords, which are not strong enough for them both.
- 0.39 'oto' · *The Wizard of Oz*: Her dog Toto is seized by neighbor Almira Gulch, who has obtained a sheriff's order after being bitten by him.
- 0.38 ' topp' · *Batman '89*: During a scuffle with Batman, Napier topples off a catwalk and falls into a vat of chemicals.
- 0.37 'obb' · *Pride & Prejudice*: Bingley and Jane are immediately taken with each other, while Elizabeth instantly dislikes the snobbish Darcy and overhears his dismissive remarks about her.
- 0.37 'att' · *The Godfather*: After the peace meeting, Vito realizes that Barzini is the mastermind behind Sonny's death, not Tattaglia.
- 0.36 ' bubb' · *Mean Girls*: She struggles to make friends and is patronized by the teachers, but befriends outcasts Janis Ian and Damian Leigh, who explain the school's various cliques to her and w…

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 20.88 (p100.0, open, peak ' Todd') · *White Knight*: When the Joker's obsession with Batman culminated in Jason Todd's torture, Quinzel left him to try and help save Todd's life.
- 20.25 (p100.0, open, peak ' Todd') · *A Death in the Family*: While eavesdropping on a child pornography ring and awaiting police backup in Gotham City, Jason Todd (Robin) ignores Batman's orders and attacks the criminals.
- 19.75 (p99.9, open, peak ' Todd') · *Knightfall*: In addition, since the events from A Death in the Family, Batman struggles with the trauma of the loss of his protégé Jason Todd.
- 19.50 (p99.9, open, peak ' Todd') · *Dark Knight Returns*: Over a decade after the United Nations outlawed superheroes and the Joker murdered Batman's protégé, Robin (Jason Todd), a 55-year-old Bruce Wayne battles alcoholism in…
- 19.38 (p99.9, open, peak ' Todd') · *Hush*: He discovers that the current Robin, Tim Drake, has been captured by the former second Robin, Jason Todd, who had previously perished.
- 19.38 (p99.9, open, peak ' Todd') · *Knightfall*: The rescue of Mayor Krol from the teaming of the Joker and Scarecrow pushes Batman to his mental and physical limits: a dose of Scarecrow's fear gas makes him relive the…
- 18.88 (p99.8, open, peak ' Todd') · *Under the Red Hood*: He realizes his mistake when he is unable to control Joker, who captures Jason Todd, the second Robin and Batman's partner.
- 16.62 (p99.8, open, peak ' Todd') · *A Death in the Family*: Her first initial is "S", not "C" as in Catherine Todd, the woman he knew as his mother.

Blind maximum attempts:

- 0.08 (p85.1, echo, peak ' crow') blind_C: The second boy sidekick, a hot-headed street orphan caught stealing the hero's car tires, was beaten with a crowbar and killed by the clown, later returning as a gun-toting vigilante.
- 0.08 (p83.6, echo, peak ' crow') blind_A: The hot-headed street kid who stole the hero's tires became the second sidekick, was beaten to death with a crowbar, and later returned as a gun-toting antihero.
- 0.06 (p74.3, echo, peak ' crow') blind_B: The second sidekick, a reckless street orphan caught stealing the hero's tires, is murdered with a crowbar and later returns from the dead as a gun-wielding vigilante.

Near miss (Jason, Todd): Jason and Todd, two brothers from Leeds, opened a bicycle repair shop on the high street in 2004.

### 25. Ra's al Ghul — L4·9760-

*character; present in 8 of 32 Batman works; family of 96. Paper's note: The immortal leader of the League of Assassins and his mission to destroy Gotham.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 4 of 36 | yes |
| 2. Sign regime | Top 50 firings: 22 gate-open, 28 echo, 0 other; echo share 56% (0 of the top 8; first echo at rank 21). Work scores (mean of the top 3 sentences) drawn from echo firings: 82% in Batman works, 99% in ordinary films; 27 of 32 Batman works and 68 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): 'ul' ×17 (17/0), ' longer' ×6 (0/6), ' freezing' ×4 (0/4), 'ar' ×2 (2/0), ' long' ×2 (0/2), 'ug' ×1 (1/0) | no |
| 3. Negative pole | L4·9760+, max 0.48. The opposite pole is noise (Gh- in Ghoul, Vaughn). | – |
| 4. Blind maximum | Best of 3: 0.10, against 9.38 for the corpus maximum in context and 9.94 for the corpus top 8 read standalone (corpus p93.4; peak on ' decad', echo). The weakest of the corpus top 8 is 7.56 in context, 8.00 standalone; neutral sentences max 0.07. | no |
| 5. Near miss | 13.44 (corpus p100.0; 143% of the in-context maximum, 13175% of the best blind maximum; peak on 'ul') | no |
| **Verdict** | Fails. Fires on -ul in Ghul: the star Algol's Arabic name al-ghul scores 13.4, above the strongest Ra's al Ghul sentence (9.4); a name-free immortal assassin leader scores 0.10. 56% of the top 50 are echo (longer, freezing). | **fails** |

Suggested label: **The token -ul in Ghul**

Strongest corpus sentences (activation, regime, peak token):

- 9.38 open 'ul' · *Dark Knight Rises*: Bane, aware of Batman's identity, gloats that he intends to fulfill Ra's al Ghul's mission to destroy Gotham City before brutally crippling Batman in combat.
- 8.56 open 'ul' · *Arkham Asylum (game)*: The body of Ra's al Ghul is in the asylum's morgue and the Ventriloquist's dummy, Scarface, appears several times throughout the story.
- 8.19 open 'ul' · *Hush*: After foiling an armored car robbery by the Riddler, Nightwing and Batman discover evidence that Ra's al Ghul is also involved in what Batman has come to think of as a g…
- 8.12 open 'ul' · *Under the Red Hood*: Ra's al Ghul hires the Joker as a distraction so that Ra's could destroy Europe's financial districts.
- 8.06 open 'ul' · *Arkham Knight*: Arkham Knight also features appearances and voice cameos by various characters drawn from the history of Batman comics and the previous Arkham games, including pharmaceu…
- 7.84 open 'ul' · *Dark Knight Rises*: He takes Bruce to an ancient underground prison in the Middle East, where Bruce learns that Ra's al Ghul's wife was sent there while pregnant with his child.
- 7.59 open 'ul' · *Batman Begins*: In a Bhutan prison, Bruce is approached by Henri Ducard, who recruits him into the League of Shadows, led by Ra's al Ghul.
- 7.56 open 'ul' · *Arkham City*: Wasserman), the League of Assassins's leader, Ra's al Ghul (Dee Bradley Baker), the mind-controlling Mad Hatter (Peter MacNicol), and the assassin Deadshot (Chris Cox),…

Echo dominates; strongest gate-open firings:

- 9.38 'ul' · *Dark Knight Rises*: Bane, aware of Batman's identity, gloats that he intends to fulfill Ra's al Ghul's mission to destroy Gotham City before brutally crippling Batman in combat.
- 8.56 'ul' · *Arkham Asylum (game)*: The body of Ra's al Ghul is in the asylum's morgue and the Ventriloquist's dummy, Scarface, appears several times throughout the story.
- 8.19 'ul' · *Hush*: After foiling an armored car robbery by the Riddler, Nightwing and Batman discover evidence that Ra's al Ghul is also involved in what Batman has come to think of as a g…
- 8.12 'ul' · *Under the Red Hood*: Ra's al Ghul hires the Joker as a distraction so that Ra's could destroy Europe's financial districts.
- 8.06 'ul' · *Arkham Knight*: Arkham Knight also features appearances and voice cameos by various characters drawn from the history of Batman comics and the previous Arkham games, including pharmaceu…
- 7.84 'ul' · *Dark Knight Rises*: He takes Bruce to an ancient underground prison in the Middle East, where Bruce learns that Ra's al Ghul's wife was sent there while pregnant with his child.
- 7.59 'ul' · *Batman Begins*: In a Bhutan prison, Bruce is approached by Henri Ducard, who recruits him into the League of Shadows, led by Ra's al Ghul.
- 7.56 'ul' · *Arkham City*: Wasserman), the League of Assassins's leader, Ra's al Ghul (Dee Bradley Baker), the mind-controlling Mad Hatter (Peter MacNicol), and the assassin Deadshot (Chris Cox),…

Negative pole L4·9760+, strongest sentences:

- 0.48 ' Gh' · *Return of the Joker*: A new faction of the Jokerz gang—consisting of Bonk, Woof, Ghoul, Chucko, and the Dee-Dee twins—attempts to steal high-tech electronic equipment but is intercepted by Ba…
- 0.47 ' Vaugh' · *Jaws*: Mayor Larry Vaughn persuades him to reconsider, fearing the town's summer economy will suffer.
- 0.39 'ary' · *Arkham Knight*: Batman's loyal butler Alfred Pennyworth (Martin Jarvis) and Wayne Enterprises colleague Lucius Fox (Dave Fennoy) provide Batman with tactical support, and the holy warri…
- 0.39 'ary' · *Arkham City*: The villain Hush (also voiced by Conroy), the mysterious Azrael (Khary Payton), and reporter Vicki Vale (also voiced by DeLisle) also appear.
- 0.37 ' Vaugh' · *Jaws*: When local fishermen catch a tiger shark, Vaughn declares the beaches safe.
- 0.34 'en' · *The Princess Bride*: Humperdinck then secretly orders his advisor Count Rugen, the six-fingered man, to take Westley to his torture chamber, the Pit of Despair.
- 0.34 'ma' · *Batman Forever*: Criminal psychologist Chase Meridian diagnoses Nygma as psychotic.
- 0.33 'uer' · *Scarlet Pimpernel*: In 1792, an early stage of the French Revolution, Marguerite St. Just, a beautiful French actress, is married to wealthy English fop Sir Percy Blakeney, baronet.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 9.94 (p100.0, open, peak 'ul') · *Dark Knight Rises*: He takes Bruce to an ancient underground prison in the Middle East, where Bruce learns that Ra's al Ghul's wife was sent there while pregnant with his child.
- 9.31 (p100.0, open, peak 'ul') · *Dark Knight Rises*: Bane, aware of Batman's identity, gloats that he intends to fulfill Ra's al Ghul's mission to destroy Gotham City before brutally crippling Batman in combat.
- 9.31 (p100.0, open, peak 'ul') · *Arkham Asylum (game)*: The body of Ra's al Ghul is in the asylum's morgue and the Ventriloquist's dummy, Scarface, appears several times throughout the story.
- 8.81 (p100.0, open, peak 'ul') · *Under the Red Hood*: After discovering that Jason's corpse is fake, Batman confronts Ra's al Ghul and demands to know the truth.
- 8.69 (p100.0, open, peak 'ul') · *Hush*: After foiling an armored car robbery by the Riddler, Nightwing and Batman discover evidence that Ra's al Ghul is also involved in what Batman has come to think of as a g…
- 8.25 (p99.9, open, peak 'ul') · *Arkham Knight*: Arkham Knight also features appearances and voice cameos by various characters drawn from the history of Batman comics and the previous Arkham games, including pharmaceu…
- 8.06 (p99.9, open, peak 'ul') · *Under the Red Hood*: Ra's al Ghul hires the Joker as a distraction so that Ra's could destroy Europe's financial districts.
- 8.00 (p99.9, open, peak 'ul') · *Batman Begins*: In a Bhutan prison, Bruce is approached by Henri Ducard, who recruits him into the League of Shadows, led by Ra's al Ghul.

Blind maximum attempts:

- 0.10 (p93.4, echo, peak ' decad') blind_B: The immortal leader of a secret order of assassins, kept alive for centuries by bathing in rejuvenating pits, trains the hero and plots to destroy the decadent city.
- 0.07 (p85.6, echo, peak ' decad') blind_C: The centuries-old immortal leader of a secret league of assassins, kept alive by regenerating pits, trains the hero, whose lover is his daughter, and plots to purge the decadent city.
- 0.04 (p48.6, echo, peak ' plotting') blind_A: The centuries-old master of a league of assassins rose dripping from the glowing green pit that restored his youth, plotting again to cleanse the world.

Near miss (Ra's, al-ghul): The lecture covered Ra's temple at Heliopolis and the star Algol, whose Arabic name al-ghul means the ghoul.

### 26. Lucius Fox — L5·10113-

*character; present in 14 of 32 Batman works; family of 4. Paper's note: The Wayne Enterprises engineer. Also fires on Zorro's nickname "The Fox" and on the Zorro movie young Bruce sees before his parents die.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 5 of 36 | yes |
| 2. Sign regime | Top 50 firings: 50 gate-open, 0 echo, 0 other; echo share 0% (0 of the top 8; first echo at rank none in the top 50). Work scores (mean of the top 3 sentences) drawn from echo firings: 52% in Batman works, 80% in ordinary films; 19 of 32 Batman works and 61 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Quinn' ×18 (18/0), ' Fox' ×12 (12/0), 'orro' ×3 (3/0), ' fo' ×3 (3/0), 'iki' ×3 (3/0), 'let' ×1 (1/0) | yes |
| 3. Negative pole | L5·10113+, max 1.38. The opposite pole is noise (Fernet, bat-ropes). | – |
| 4. Blind maximum | Best of 3: 0.12, against 13.81 for the corpus maximum in context and 15.50 for the corpus top 8 read standalone (corpus p65.6; peak on ' brilliant', echo). The weakest of the corpus top 8 is 8.56 in context, 9.81 standalone; neutral sentences max 0.04. | no |
| 5. Near miss | 9.06 (corpus p99.8; 66% of the in-context maximum, 7812% of the best blind maximum; peak on ' fox') | no |
| **Verdict** | Fails. Fires on the surnames Quinn (18 of the top 50) and Fox (12), including Zorro, The Fox, and the Zorro movie; a fox in the allotments scores 9.1, a name-free applied-sciences executive 0.12. | **fails** |

Suggested label: **The surnames Quinn and Fox (and Zorro)**

Strongest corpus sentences (activation, regime, peak token):

- 13.81 open ' Fox' · *Batman Begins*: Company archivist Lucius Fox, a friend of Bruce's father, allows him access to prototype defense technologies, including a protective bodysuit and the Tumbler, an armore…
- 13.50 open ' Fox' · *The Dark Knight*: With the help of Wayne Enterprises CEO Lucius Fox, Batman finds Lau in Hong Kong and returns him to the custody of the Gotham police.
- 13.31 open ' Fox' · *Arkham Knight*: Batman's loyal butler Alfred Pennyworth (Martin Jarvis) and Wayne Enterprises colleague Lucius Fox (Dave Fennoy) provide Batman with tactical support, and the holy warri…
- 13.25 open ' Fox' · *Earth One*: Bruce then seeks the help of Lucius Fox, a 22-year-old intern at Wayne Medical, who agrees to repair his equipment in exchange for funding for a project to develop artif…
- 12.94 open ' Fox' · *Knightfall*: Within the regular series, the buildup to "Knightfall" begins with a six-issue run in Batman #484-489 (September 1992 - February 1993), in which Batman (at the onset of…
- 12.31 open ' Fox' · *Dark Knight Rises*: Gordon finds the Bat-Signal repaired, while Lucius Fox discovers that Bruce had fixed the Bat's malfunctioning autopilot, implying that he escaped the Bat before the det…
- 11.81 open ' Fox' · *Zorro*: Don Diego quickly adopts the guise of El Zorro ("The Fox"), a masked outlaw dressed entirely in black, who becomes the defender of the common people and a champion for j…
- 8.56 open 'orro' · *Batman R.I.P.*: In the epilogue, young Bruce Wayne is walking home with his parents after seeing a Zorro movie.

Negative pole L5·10113+, strongest sentences:

- 1.38 'ernet' · *Dark Knight Rises*: In Florence, drinking a Fernet Branca, Alfred discovers that Bruce is alive and in a relationship with Selina, and they acknowledge each other with a smile.
- 0.91 'rop' · *Batman '66*: The duo quietly leaves United World Headquarters by climbing out of the window and descending on their bat-ropes.
- 0.65 'ico' · *The Silence of the Lambs*: Clarice Starling, a young FBI trainee at Quantico, is recruited by Unit chief Jack Crawford to interview Dr. Hannibal Lecter, a brilliant and cannibalistic psychiatrist…
- 0.58 ' Box' · *Batman Forever*: They commit a series of robberies to finance Nygma's new company and mass-produce his brainwave device dubbed the Box, which steals information from minds and transfers…
- 0.57 'UM' · *The Shining*: Back in their private apartment, Danny begins to chant, from a murmur, "red rum" more and more loudly ("red RUM") and writes it out in lipstick on the bathroom door.
- 0.52 'icl' · *The Social Network*: After narrowly losing in the 2004 Henley Royal Regatta for Harvard against the Hollandia Roeiclub in United Kingdom, the Winklevoss twins discover that, through Parker,…
- 0.50 'ot' · *Moonlight*: The next day, Chiron admits to Juan and Teresa that he hates his mother and asks what a "faggot" is.
- 0.50 ' tum' · *The Princess Bride*: While tumbling down, he shouts, "As you wish!"

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 15.50 (p100.0, open, peak ' Fox') · *Batman Begins*: Company archivist Lucius Fox, a friend of Bruce's father, allows him access to prototype defense technologies, including a protective bodysuit and the Tumbler, an armore…
- 14.94 (p100.0, open, peak ' Fox') · *Arkham Knight*: Batman's loyal butler Alfred Pennyworth (Martin Jarvis) and Wayne Enterprises colleague Lucius Fox (Dave Fennoy) provide Batman with tactical support, and the holy warri…
- 14.88 (p100.0, open, peak ' Fox') · *Earth One*: Bruce then seeks the help of Lucius Fox, a 22-year-old intern at Wayne Medical, who agrees to repair his equipment in exchange for funding for a project to develop artif…
- 14.25 (p100.0, open, peak ' Fox') · *The Dark Knight*: With the help of Wayne Enterprises CEO Lucius Fox, Batman finds Lau in Hong Kong and returns him to the custody of the Gotham police.
- 14.25 (p100.0, open, peak ' Fox') · *Dark Knight Rises*: Gordon finds the Bat-Signal repaired, while Lucius Fox discovers that Bruce had fixed the Bat's malfunctioning autopilot, implying that he escaped the Bat before the det…
- 13.50 (p99.9, open, peak ' Fox') · *Knightfall*: Within the regular series, the buildup to "Knightfall" begins with a six-issue run in Batman #484-489 (September 1992 - February 1993), in which Batman (at the onset of…
- 12.38 (p99.9, open, peak ' Fox') · *Zorro*: Don Diego quickly adopts the guise of El Zorro ("The Fox"), a masked outlaw dressed entirely in black, who becomes the defender of the common people and a champion for j…
- 9.81 (p99.8, open, peak 'orro') · *Batman R.I.P.*: In the epilogue, young Bruce Wayne is walking home with his parents after seeing a Zorro movie.

Blind maximum attempts:

- 0.12 (p65.6, echo, peak ' brilliant') blind_C: The company's brilliant chief executive and head of applied sciences quietly supplies the billionaire with experimental armor, gadgets and vehicles while discreetly keeping his secret.
- 0.11 (p65.0, echo, peak ' vigil') blind_B: The company's quietly brilliant head of applied sciences supplies the vigilante with experimental armor, vehicles and gadgets, while discreetly running the corporation on his behalf.
- 0.09 (p51.4, echo, peak '-sp') blind_A: The soft-spoken executive who ran the company's applied sciences division slid the prototype armor across the table and quietly asked no questions about its purpose.

Near miss (fox, Lucius): A fox slipped through the fence of the Lucius Street allotments and dug up the newly planted onions.

### 27. A corrupt police force — L35·4074-

*trope; present in 12 of 32 Batman works; family of 1. Paper's note: Dirty cops and an honest officer purging corruption.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 35 of 36 | yes |
| 2. Sign regime | Top 50 firings: 36 gate-open, 14 echo, 0 other; echo share 28% (0 of the top 8; first echo at rank 32). Work scores (mean of the top 3 sentences) drawn from echo firings: 64% in Batman works, 99% in ordinary films; 21 of 32 Batman works and 68 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Gotham' ×23 (23/0), ' the' ×7 (7/0), ' Ki' ×6 (0/6), ' Ar' ×2 (0/2), ' K' ×2 (0/2), 'N' ×1 (1/0) | no |
| 3. Negative pole | L35·4074+, max 42.50. The opposite pole peaks on the first piece of a foreign surname (Da-song, Zazu, Schraeder, Noirtier): a last-layer name-onset predictor. | – |
| 4. Blind maximum | Best of 3: 2.22, against 40.00 for the corpus maximum in context and 33.50 for the corpus top 8 read standalone (corpus p94.9; peak on 'oked', open). The weakest of the corpus top 8 is 11.88 in context, 2.02 standalone; neutral sentences max 1.10. | no |
| 5. Near miss | 0.70 (corpus p23.8; 2% of the in-context maximum, 32% of the best blind maximum; peak on ' the') | yes |
| **Verdict** | Fails. A layer-35 unit: every top-8 peak is Gotham immediately before City Police Department or police, so it predicts the police after Gotham. The packed name-free corrupt-police sentence scores 2.2 against 40; the neutral sentences reach 1.1. | **fails** |

Suggested label: **Layer-35 unit on Gotham before (City) Police Department**

Strongest corpus sentences (activation, regime, peak token):

- 40.00 open ' Gotham' · *Year One*: Jim focuses on purging corruption from the Gotham City Police Department after witnessing his partner, Arnold Flass, abuse his power.
- 30.88 open ' Gotham' · *The Batman*: Reclusive billionaire Bruce Wayne, who has operated for two years as the vigilante Batman, investigates the murder alongside Lieutenant Jim Gordon, much to the disdain o…
- 29.75 open ' Gotham' · *The Dark Knight*: With the help of Wayne Enterprises CEO Lucius Fox, Batman finds Lau in Hong Kong and returns him to the custody of the Gotham police.
- 24.38 open ' Gotham' · *Arkham Knight*: Only criminals remain in the city, leaving Commissioner Gordon and the Gotham City Police Department outnumbered.
- 23.00 open ' Gotham' · *Earth One*: Meanwhile, Detective James Gordon of the Gotham City Police Department meets his new partner, Harvey Bullock, an experienced LAPD detective who has come to Gotham to sol…
- 16.88 open ' Gotham' · *Dark Knight Returns*: Batman and Robin (Kelley) track him to a county fair while evading a Gotham police pursuit.
- 15.38 open ' Gotham' · *Batman '89*: Corrupt Gotham City police lieutenant Max Eckhardt arranges the hit on Napier by conducting an unauthorized raid on Axis Chemicals.
- 11.88 open ' Gotham' · *Dark Knight Rises*: Bane's henchmen capture Gordon in the sewers but he escapes and is found by steadfast Gotham City police officer John Blake, an orphan who has deduced Bruce's secret ide…

Negative pole L35·4074+, strongest sentences:

- 42.50 ' Da' · *Parasite*: Ki-woo recommends "Jessica," actually Ki-jung, as an art therapist for the Parks' young son, Da-song, who has been traumatised from having a seizure after seeing a "ghos…
- 31.88 ' Z' · *Ghostbusters*: Supernatural activity rapidly increases across the city and the Ghostbusters become famous; they hire a fourth member, Winston Zeddemore, to cope with the growing demand.
- 27.25 ' Da' · *Parasite*: Leaving to study abroad, he suggests that Ki-woo lie about his qualifications to take over Min-hyuk's job as an English tutor for Da-hye, the daughter of the wealthy but…
- 23.00 ' Z' · *The Lion King*: Mufasa is alerted by his majordomo, the hornbill Zazu, and rescues the cubs.
- 20.00 ' Sch' · *The Sound of Music*: When the Captain returns to the villa with Baroness Elsa Schraeder, a wealthy socialite, and their mutual friend Max Detweiler, they are greeted by Maria and the childre…
- 15.44 ' No' · *Count of Monte Cristo*: Villefort is Noirtier's son, and knows that his political career would be ruined if it were known that his father is a Bonapartist.
- 14.94 ' S' · *Arkham City*: After the Joker attempts to assassinate Catwoman, Batman tracks him to his hideout in the Sionis Steelmill, believing Joker may know the truth behind Protocol 10.
- 14.69 ' Dum' · *Joker (2019)*: Arthur pursues a relationship with his neighbor, single mother Sophie Dumond, and invites her to see his routine at a comedy club.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 33.50 (p100.0, open, peak ' Gotham') · *The Dark Knight*: With the help of Wayne Enterprises CEO Lucius Fox, Batman finds Lau in Hong Kong and returns him to the custody of the Gotham police.
- 33.25 (p100.0, open, peak ' Gotham') · *The Batman*: Reclusive billionaire Bruce Wayne, who has operated for two years as the vigilante Batman, investigates the murder alongside Lieutenant Jim Gordon, much to the disdain o…
- 18.75 (p99.9, open, peak ' Gotham') · *Year One*: Jim focuses on purging corruption from the Gotham City Police Department after witnessing his partner, Arnold Flass, abuse his power.
- 18.00 (p99.9, open, peak ' Gotham') · *Dark Knight Returns*: Batman and Robin (Kelley) track him to a county fair while evading a Gotham police pursuit.
- 13.81 (p99.8, open, peak ' Gotham') · *Arkham Knight*: Only criminals remain in the city, leaving Commissioner Gordon and the Gotham City Police Department outnumbered.
- 12.00 (p99.8, open, peak ' Gotham') · *Earth One*: Meanwhile, Detective James Gordon of the Gotham City Police Department meets his new partner, Harvey Bullock, an experienced LAPD detective who has come to Gotham to sol…
- 8.12 (p99.7, open, peak ' Gotham') · *Dark Knight Rises*: Bane's henchmen capture Gordon in the sewers but he escapes and is found by steadfast Gotham City police officer John Blake, an orphan who has deduced Bruce's secret ide…
- 2.02 (p91.7, open, peak ' Gotham') · *Batman '89*: Corrupt Gotham City police lieutenant Max Eckhardt arranges the hit on Napier by conducting an unauthorized raid on Axis Chemicals.

Blind maximum attempts:

- 2.22 (p94.9, open, peak 'oked') blind_C: The city's corrupt police force is rotten to the core, with crooked cops on the mob's payroll, bribed detectives, dirty officers and a bought-off chief.
- 0.59 (p15.4, echo, peak ' brown') blind_A: Half the precinct was on the mob's payroll, detectives pocketing bribes in brown envelopes while evidence vanished and witnesses turned up dead.
- 0.59 (p15.1, echo, peak '.') blind_B: The city's police department is riddled with corruption, with officers on the mob's payroll, detectives taking bribes and the chief burying evidence for crime bosses.

Near miss (police, corrupt, force): The police station's hard drive was corrupt after the power cut, and the force had to reorder its parking permits.

### 28. Prison and the inmates released — L4·8535+

*place; present in 20 of 32 Batman works; family of 3. Paper's note: Blackgate Penitentiary, prison breaks, freeing the inmates.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 4 of 36 | yes |
| 2. Sign regime | Top 50 firings: 32 gate-open, 18 echo, 0 other; echo share 36% (0 of the top 8; first echo at rank 29). Work scores (mean of the top 3 sentences) drawn from echo firings: 72% in Batman works, 98% in ordinary films; 27 of 32 Batman works and 69 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Penny' ×22 (22/0), 'it' ×4 (4/0), ' sentenced' ×2 (0/2), 'otts' ×2 (0/2), ' spear' ×2 (0/2), ' Pen' ×1 (1/0) | no |
| 3. Negative pole | L4·8535-, max 0.44. The opposite pole is noise (Palantine, Pennino). | – |
| 4. Blind maximum | Best of 3: 0.17, against 8.19 for the corpus maximum in context and 7.91 for the corpus top 8 read standalone (corpus p98.1; peak on ' of', echo). The weakest of the corpus top 8 is 2.03 in context, 1.90 standalone; neutral sentences max 0.17. | no |
| 5. Near miss | 0.08 (corpus p91.7; 1% of the in-context maximum, 50% of the best blind maximum; peak on ' plaque') | no |
| **Verdict** | Fails. A sub-word detector for Pen-: 22 of the top 50 peak on Penny(worth), the top 4 on -it- in Penitentiary, others on Penelope and Pennino. A name-free prison break with inmates pouring out scores 0.17. | **fails** |

Suggested label: **The tokens Penny(worth) and Penit(entiary)**

Strongest corpus sentences (activation, regime, peak token):

- 8.19 open 'it' · *Dark Knight Rises*: Effectively taking over Gotham, he exposes Dent's crimes to the world, and releases the prisoners of Blackgate Penitentiary.
- 8.12 open 'it' · *Gotham by Gaslight*: Sentenced to Blackgate Penitentiary following the trial, Bruce is met by Selina, who urges him to reveal he is Batman to clear his name and help save the girls from Jack.
- 7.78 open 'it' · *Arkham City*: Declaring both the asylum and Blackgate Penitentiary no longer suitable to contain the city's detainees, Sharp's administration orders both facilities closed, and he pur…
- 6.47 open 'it' · *Arkham Asylum (game)*: At the penitentiary, Quinn releases Poison Ivy from her cell before being imprisoned by Batman.
- 4.50 open ' Pen' · *Arkham Asylum (game)*: Joker then directs Batman to the captured Bane, who has been experimented on by asylum doctor Penelope Young.
- 2.31 open 'ino' · *Rocky*: Rocky builds a romantic relationship with Adrianna "Adrian" Pennino, a shy woman who is working part-time at the J&M Tropical Fish pet store.
- 2.03 open ' Penny' · *Batman Forever*: As Bruce recovers, he and his butler, Alfred Pennyworth, deduce that Nygma is the Riddler through clues he left in his riddles, while Chase is held prisoner in the Riddl…
- 2.03 open ' Penny' · *Joker (2019)*: He goes to Wayne Manor, where he meets Thomas's young son Bruce, but flees after a scuffle with family butler Alfred Pennyworth.

Negative pole L4·8535-, strongest sentences:

- 0.44 'antine' · *Taxi Driver*: Travis becomes infatuated with Betsy, a campaign worker for Senator Charles Palantine, who is running for president.
- 0.41 'pe' · *Rear Window*: During a heat wave, Jeff watches his neighbors through open windows, including a newlywed couple; a professional dancer coined "Miss Torso"; a songwriter with writer's b…
- 0.40 'is' · *Raiders of the Lost Ark*: After returning to the United States, Jones is briefed by two Army Intelligence agents that Nazi German forces have discovered and are excavating at Tanis, Egypt, and on…
- 0.38 ' Penn' · *Rocky*: Rocky builds a romantic relationship with Adrianna "Adrian" Pennino, a shy woman who is working part-time at the J&M Tropical Fish pet store.
- 0.36 'antine' · *Taxi Driver*: He begins attending Palantine's rallies to scope out his security.
- 0.35 'af' · *Legally Blonde*: Elle learns of Warner's engagement to his previous girlfriend, Vivian Kensington, and befriends local manicurist Paulette Bonafonté.
- 0.32 ' Renaissance' · *The Grand Budapest Hotel*: Kovacs announces that she has bequeathed a famous Renaissance painting, Boy with Apple, to Gustave.
- 0.32 'nel' · *Scarlet Pimpernel*: Meanwhile, the "League of the Scarlet Pimpernel", a secret society of twenty English aristocrats, "one to command, and nineteen to obey", is engaged in rescuing their Fr…

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 7.91 (p99.9, open, peak 'it') · *Dark Knight Rises*: Effectively taking over Gotham, he exposes Dent's crimes to the world, and releases the prisoners of Blackgate Penitentiary.
- 7.66 (p99.9, open, peak 'it') · *Gotham by Gaslight*: Sentenced to Blackgate Penitentiary following the trial, Bruce is met by Selina, who urges him to reveal he is Batman to clear his name and help save the girls from Jack.
- 7.19 (p99.9, open, peak 'it') · *Arkham City*: Declaring both the asylum and Blackgate Penitentiary no longer suitable to contain the city's detainees, Sharp's administration orders both facilities closed, and he pur…
- 5.88 (p99.9, open, peak 'it') · *Arkham Asylum (game)*: At the penitentiary, Quinn releases Poison Ivy from her cell before being imprisoned by Batman.
- 3.75 (p99.9, open, peak ' Pen') · *Arkham Asylum (game)*: Joker then directs Batman to the captured Bane, who has been experimented on by asylum doctor Penelope Young.
- 2.75 (p99.9, open, peak 'ino') · *Rocky*: Rocky builds a romantic relationship with Adrianna "Adrian" Pennino, a shy woman who is working part-time at the J&M Tropical Fish pet store.
- 2.08 (p99.8, open, peak ' Penny') · *Batman Forever*: As Bruce recovers, he and his butler, Alfred Pennyworth, deduce that Nygma is the Riddler through clues he left in his riddles, while Chase is held prisoner in the Riddl…
- 1.90 (p99.7, open, peak ' Penny') · *Joker (2019)*: He goes to Wayne Manor, where he meets Thomas's young son Bruce, but flees after a scuffle with family butler Alfred Pennyworth.

Blind maximum attempts:

- 0.17 (p98.1, echo, peak ' of') blind_A: When the walls of the prison were blown open, hundreds of inmates in orange jumpsuits poured through the gates and scattered howling into the streets.
- 0.10 (p94.6, echo, peak ' of') blind_B: The villain blows open the walls of the city prison and the asylum, releasing thousands of convicts and inmates who pour into the streets.
- 0.04 (p67.6, echo, peak ' blew') blind_C: The villain blew open the prison walls and released every inmate, freeing thousands of convicts, prisoners and criminals from their cells to run loose through the city.

Near miss (prison, inmates, released): The prison museum's guide listed famous inmates on a plaque, and an audio tour was released for visitors in 2015.

### 29. The antidote — L1·10779-

*trope; present in 8 of 32 Batman works; family of 5. Paper's note: A poison or toxin and the race to make an antidote.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 1 of 36 — in the first four layers, likely a token or spelling detector | no |
| 2. Sign regime | Top 50 firings: 13 gate-open, 37 echo, 0 other; echo share 74% (0 of the top 8; first echo at rank 13). Work scores (mean of the top 3 sentences) drawn from echo firings: 88% in Batman works, 99% in ordinary films; 30 of 32 Batman works and 69 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' against' ×24 (0/24), ' antid' ×8 (8/0), ' post' ×6 (0/6), ' non' ×3 (0/3), ' anti' ×2 (2/0), ' antim' ×1 (1/0) | no |
| 3. Negative pole | L1·10779+, max 0.56. The opposite pole is noise (-ino, -is in Otis). | – |
| 4. Blind maximum | Best of 3: 8.38, against 7.91 for the corpus maximum in context and 8.44 for the corpus top 8 read standalone (corpus p100.0; peak on ' antid', open). The weakest of the corpus top 8 is 6.84 in context, 6.72 standalone; neutral sentences max 0.08. | yes |
| 5. Near miss | 9.50 (corpus p100.0; 120% of the in-context maximum, 113% of the best blind maximum; peak on ' antid') | no |
| **Verdict** | A token detector, layer 1. Gate-open firings are antid-, anti- and antim- (antimatter scores 7.3); the 24 against, 6 post and 3 non firings are all echo, so the apparent against/anti- sense is the gate-closed artefact. A comedy called an antidote to loud sequels scores 9.5, above the corpus maximum 7.9. | **fails** |

Suggested label: **The token antid- / anti-**

Strongest corpus sentences (activation, regime, peak token):

- 7.91 open ' antid' · *Batman Begins*: He awakens on his birthday, having been saved by Alfred and Fox, who developed an antidote for the toxin.
- 7.78 open ' antid' · *Return of the Joker*: After Terry saves Bruce's life with an antidote, Barbara finally explains the Joker's disappearance: After Nightwing (Dick Grayson) had left Gotham City for Blüdhaven, t…
- 7.59 open ' antid' · *Arkham Asylum (game)*: He learns from Ivy that the spores required to create an antidote are found exclusively in Killer Croc's lair in a sewer.
- 7.31 open ' antim' · *Batman '66*: This plan fails when the henchmen unexpectedly disappear into antimatter once struck: the Penguin mistakenly rehydrates them with toxic heavy water used to recharge the…
- 7.28 open ' antid' · *Arkham City*: With Ra's al Ghul's blood, Freeze is able to develop an antidote, but Harley Quinn steals it before Batman can use it.
- 7.28 open ' antid' · *Batman Begins*: Evading the police in the Tumbler, Batman saves Rachel and administers Fox's antidote.
- 6.94 open ' antid' · *Arkham Asylum (game)*: However, he can only synthesize one dose of the antidote before Ivy's plants breach the cave and destroy his equipment.
- 6.84 open ' antid' · *Arkham Asylum (game)*: Batman refuses to transform, uses the antidote on himself, and defeats the Titan-affected Joker and his henchmen.

Echo dominates; strongest gate-open firings:

- 7.91 ' antid' · *Batman Begins*: He awakens on his birthday, having been saved by Alfred and Fox, who developed an antidote for the toxin.
- 7.78 ' antid' · *Return of the Joker*: After Terry saves Bruce's life with an antidote, Barbara finally explains the Joker's disappearance: After Nightwing (Dick Grayson) had left Gotham City for Blüdhaven, t…
- 7.59 ' antid' · *Arkham Asylum (game)*: He learns from Ivy that the spores required to create an antidote are found exclusively in Killer Croc's lair in a sewer.
- 7.31 ' antim' · *Batman '66*: This plan fails when the henchmen unexpectedly disappear into antimatter once struck: the Penguin mistakenly rehydrates them with toxic heavy water used to recharge the…
- 7.28 ' antid' · *Arkham City*: With Ra's al Ghul's blood, Freeze is able to develop an antidote, but Harley Quinn steals it before Batman can use it.
- 7.28 ' antid' · *Batman Begins*: Evading the police in the Tumbler, Batman saves Rachel and administers Fox's antidote.
- 6.94 ' antid' · *Arkham Asylum (game)*: However, he can only synthesize one dose of the antidote before Ivy's plants breach the cave and destroy his equipment.
- 6.84 ' antid' · *Arkham Asylum (game)*: Batman refuses to transform, uses the antidote on himself, and defeats the Titan-affected Joker and his henchmen.

Relabel from gate-open firings: **The token antid-/anti- (gate-open only; the echo adds against, post-, non-)**

Negative pole L1·10779+, strongest sentences:

- 0.56 'ino' · *The Godfather*: Vito is head of the Corleone mafia family and runs it with his eldest son Santino a.k.a.
- 0.44 'is' · *Superman*: With Otis and his girlfriend Eve Teschmacher, Lex retrieves it and traps Superman in his lair, revealing his plan to sink the western U.S., making his desert land prime…
- 0.39 'is' · *Superman*: Criminal mastermind Lex Luthor discovers a joint U.S. Army/U.S. Navy missile test and plots to target the San Andreas Fault with reprogrammed missiles, though one is mis…
- 0.38 ' tir' · *Batman Returns*: Before regaining control, Batman records Oswald's insulting tirade against Gotham's citizens and later plays it during Oswald's mayoral rally, destroying his public imag…
- 0.37 'enti' · *Arkham Asylum (game)*: At the penitentiary, Quinn releases Poison Ivy from her cell before being imprisoned by Batman.
- 0.37 'is' · *Superman*: After saving the West Coast, he imprisons Luthor and Otis, then soars into the sunrise.
- 0.34 'enti' · *Dark Knight Rises*: Effectively taking over Gotham, he exposes Dent's crimes to the world, and releases the prisoners of Blackgate Penitentiary.
- 0.34 ' anim' · *Back to the Future*: Early attempts to get his parents acquainted fail, Marty gains the animosity of Biff, and Lorraine's infatuation with Marty deepens.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 8.44 (p100.0, open, peak ' antid') · *Batman Begins*: Evading the police in the Tumbler, Batman saves Rachel and administers Fox's antidote.
- 8.38 (p100.0, open, peak ' antid') · *Return of the Joker*: After Terry saves Bruce's life with an antidote, Barbara finally explains the Joker's disappearance: After Nightwing (Dick Grayson) had left Gotham City for Blüdhaven, t…
- 8.00 (p100.0, open, peak ' antim') · *Batman '66*: This plan fails when the henchmen unexpectedly disappear into antimatter once struck: the Penguin mistakenly rehydrates them with toxic heavy water used to recharge the…
- 8.00 (p100.0, open, peak ' antid') · *Arkham Asylum (game)*: Batman refuses to transform, uses the antidote on himself, and defeats the Titan-affected Joker and his henchmen.
- 7.84 (p100.0, open, peak ' antid') · *Arkham Asylum (game)*: He learns from Ivy that the spores required to create an antidote are found exclusively in Killer Croc's lair in a sewer.
- 7.69 (p99.9, open, peak ' antid') · *Batman Begins*: He awakens on his birthday, having been saved by Alfred and Fox, who developed an antidote for the toxin.
- 7.47 (p99.9, open, peak ' antid') · *Arkham Asylum (game)*: However, he can only synthesize one dose of the antidote before Ivy's plants breach the cave and destroy his equipment.
- 6.72 (p99.8, open, peak ' antid') · *Arkham City*: With Ra's al Ghul's blood, Freeze is able to develop an antidote, but Harley Quinn steals it before Batman can use it.

Blind maximum attempts:

- 8.38 (p100.0, open, peak ' antid') blind_C: The scientist synthesized an antidote, a cure and counter-agent to the poison, and injected the serum into the victim to neutralize the toxin and save him.
- 8.00 (p100.0, open, peak ' antid') blind_B: The hero synthesizes an antidote to the villain's poison, injects the dying hostage with the cure, and arranges for mass-produced doses to be distributed across the city.
- 7.12 (p99.8, open, peak ' antid') blind_A: With minutes left before the poison stopped her heart, he snapped the vial, filled the syringe with the antidote, and drove it into her arm.

Near miss (antidote): Critics called the gentle comedy an antidote to a summer of loud sequels, and audiences agreed at the box office.

### 30. A mutating super-formula — L3·10668+

*object; present in 7 of 32 Batman works; family of 2. Paper's note: The Titan steroid and other formulas that mutate their users. Mostly the Arkham games.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 3 of 36 — in the first four layers, likely a token or spelling detector | no |
| 2. Sign regime | Top 50 firings: 17 gate-open, 33 echo, 0 other; echo share 66% (0 of the top 8; first echo at rank 18). Work scores (mean of the top 3 sentences) drawn from echo firings: 88% in Batman works, 99% in ordinary films; 28 of 32 Batman works and 68 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): ' Mut' ×6 (6/0), ' transforms' ×6 (0/6), ' mutant' ×4 (4/0), ' mutating' ×3 (3/0), ' genetically' ×3 (0/3), ' transformed' ×3 (0/3) | no |
| 3. Negative pole | L3·10668-, max 0.40. The opposite pole is noise (-unt in Muntz, Mur-). | – |
| 4. Blind maximum | Best of 3: 9.75, against 10.56 for the corpus maximum in context and 11.06 for the corpus top 8 read standalone (corpus p99.9; peak on 'agen', open). The weakest of the corpus top 8 is 5.75 in context, 6.06 standalone; neutral sentences max 0.03. | would make the top 8 |
| 5. Near miss | 11.12 (corpus p100.0; 105% of the in-context maximum, 114% of the best blind maximum; peak on ' mutating') | no |
| **Verdict** | A stem detector, layer 3. Gate-open firings are Mut-, mutant, mutating; transforms, genetically and transformed are echo. A mutating flu strain scores 11.1, above the corpus maximum 10.6; the vivid blind sentence, which only has mutated, scores 4.6. | **fails** |

Suggested label: **The stem mutat- / Mut-**

Strongest corpus sentences (activation, regime, peak token):

- 10.56 open ' mutating' · *Arkham City*: There, Batman learns that the unstable properties of the Titan formula are mutating in Joker's blood, gradually killing him.
- 10.25 open ' mutating' · *Arkham Asylum (game)*: Batman attempts to resist the change, and an upset Joker takes an overdose of Titan, mutating into a monster.
- 10.06 open ' mutating' · *Hush*: At this time, Killer Croc's increasing savageness due to his condition mutating allows him to escape from Arkham Asylum, with Batman tracking him.
- 9.12 open 'agen' · *Arkham City*: Meanwhile, the Joker is dying from a disease caused by his previous overdose of the Titan formula, an unstable steroid mutagen that turns men into powerful and savage mo…
- 8.25 open ' mutate' · *Batman & Robin*: After witnessing Woodrue use it to mutate serial killer Antonio Diego into Bane, she threatens to expose Woodrue's experiments.
- 6.94 open ' Mut' · *Dark Knight Returns*: Batman's absence has left Gotham City overrun by crime, much of it driven by a gang called "the Mutants".
- 6.03 open ' mutated' · *Batman & Robin*: Woodrue attempts to kill her by overturning a shelf of various toxins; instead, Isley is mutated by the toxins into Poison Ivy, who murders Woodrue with a venom-laden ki…
- 5.75 open ' mutated' · *Arkham Knight*: Arkham Knight also features appearances and voice cameos by various characters drawn from the history of Batman comics and the previous Arkham games, including pharmaceu…

Echo dominates; strongest gate-open firings:

- 10.56 ' mutating' · *Arkham City*: There, Batman learns that the unstable properties of the Titan formula are mutating in Joker's blood, gradually killing him.
- 10.25 ' mutating' · *Arkham Asylum (game)*: Batman attempts to resist the change, and an upset Joker takes an overdose of Titan, mutating into a monster.
- 10.06 ' mutating' · *Hush*: At this time, Killer Croc's increasing savageness due to his condition mutating allows him to escape from Arkham Asylum, with Batman tracking him.
- 9.12 'agen' · *Arkham City*: Meanwhile, the Joker is dying from a disease caused by his previous overdose of the Titan formula, an unstable steroid mutagen that turns men into powerful and savage mo…
- 8.25 ' mutate' · *Batman & Robin*: After witnessing Woodrue use it to mutate serial killer Antonio Diego into Bane, she threatens to expose Woodrue's experiments.
- 6.94 ' Mut' · *Dark Knight Returns*: Batman's absence has left Gotham City overrun by crime, much of it driven by a gang called "the Mutants".
- 6.03 ' mutated' · *Batman & Robin*: Woodrue attempts to kill her by overturning a shelf of various toxins; instead, Isley is mutated by the toxins into Poison Ivy, who murders Woodrue with a venom-laden ki…
- 5.75 ' mutated' · *Arkham Knight*: Arkham Knight also features appearances and voice cameos by various characters drawn from the history of Batman comics and the previous Arkham games, including pharmaceu…

Relabel from gate-open firings: **The stem mutat- (gate-open only; the echo adds transforms, genetically)**

Negative pole L3·10668-, strongest sentences:

- 0.40 'unt' · *Up*: Eight-year-old Carl Fredricksen idolizes famed explorer Charles Muntz, who is discredited when a giant bird skeleton he brought back from Paradise Falls in Venezuela is…
- 0.34 'unt' · *Up*: Muntz travels back to the falls, vowing not to return until he brings back a live specimen.
- 0.32 'unt' · *Up*: The dogs' master is revealed to be an elderly Charles Muntz, who invites Carl and Russell aboard his airship and discusses his quest to capture the bird.
- 0.30 ' Mur' · *Interstellar*: Following a dust storm, Cooper and Murph discover that dust patterns in Murph's room, which she had attributed to a ghost, result from a gravitational anomaly.
- 0.30 'unt' · *Up*: Carl befriends Ellie, another young Muntz fan, who keeps a scrapbook of her adventures and dreams of moving her "clubhouse" (an abandoned house) to Paradise Falls and ha…
- 0.29 ' mug' · *Batman '89*: Both attend a fundraiser hosted by billionaire Bruce Wayne, who is secretly Batman, having chosen this path after witnessing a mugger murder his parents when he was a ch…
- 0.28 'Mur' · *Interstellar*: Joseph Cooper, a widowed former NASA test pilot, works as a farmer and raises his children, Murphy "Murph" and Tom, with his father-in-law Donald.
- 0.28 ' mur' · *The Shining*: Back in their private apartment, Danny begins to chant, from a murmur, "red rum" more and more loudly ("red RUM") and writes it out in lipstick on the bathroom door.

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 11.06 (p100.0, open, peak ' mutating') · *Arkham City*: There, Batman learns that the unstable properties of the Titan formula are mutating in Joker's blood, gradually killing him.
- 10.69 (p100.0, open, peak ' mutating') · *Hush*: At this time, Killer Croc's increasing savageness due to his condition mutating allows him to escape from Arkham Asylum, with Batman tracking him.
- 9.62 (p99.9, open, peak ' mutating') · *Arkham Asylum (game)*: Batman attempts to resist the change, and an upset Joker takes an overdose of Titan, mutating into a monster.
- 9.44 (p99.9, open, peak 'agen') · *Arkham City*: Meanwhile, the Joker is dying from a disease caused by his previous overdose of the Titan formula, an unstable steroid mutagen that turns men into powerful and savage mo…
- 8.31 (p99.9, open, peak ' mutate') · *Batman & Robin*: After witnessing Woodrue use it to mutate serial killer Antonio Diego into Bane, she threatens to expose Woodrue's experiments.
- 7.16 (p99.9, open, peak ' mutated') · *Batman & Robin*: Woodrue attempts to kill her by overturning a shelf of various toxins; instead, Isley is mutated by the toxins into Poison Ivy, who murders Woodrue with a venom-laden ki…
- 6.84 (p99.8, open, peak ' Mut') · *Dark Knight Returns*: Batman's absence has left Gotham City overrun by crime, much of it driven by a gang called "the Mutants".
- 6.06 (p99.8, open, peak ' mutated') · *Arkham Knight*: Arkham Knight also features appearances and voice cameos by various characters drawn from the history of Batman comics and the previous Arkham games, including pharmaceu…

Blind maximum attempts:

- 9.75 (p99.9, open, peak 'agen') blind_B: The scientist injects himself with an experimental mutagenic formula that transforms his body, swelling his muscles to monstrous size and turning him into a towering superhuman brute.
- 9.75 (p99.9, open, peak 'agen') blind_C: The experimental super-serum, a mutagenic steroid formula pumped through tubes into his veins, mutated the henchman into a hulking, monstrous super-strong brute.
- 4.59 (p99.7, open, peak ' mutated') blind_A: The green serum hissed into his veins and his muscles swelled grotesquely, bones cracking and lengthening as the formula mutated him into a monstrous brute.

Near miss (super, formula, mutating): The supermarket's super-saver infant formula sold out quickly, while a mutating flu strain kept pharmacists busy all winter.

### 31. Fear toxin — L5·5136+

*object; present in 6 of 32 Batman works; family of 2. Paper's note: Hallucinogens that weaponize fear: the blue flower compound, Scarecrow's gas.*

| Check | Result | Pass |
| --- | --- | --- |
| 1. Depth | Layer 5 of 36 | yes |
| 2. Sign regime | Top 50 firings: 13 gate-open, 37 echo, 0 other; echo share 74% (0 of the top 8; first echo at rank 12). Work scores (mean of the top 3 sentences) drawn from echo firings: 85% in Batman works, 98% in ordinary films; 29 of 32 Batman works and 68 of 69 films scored mostly by echo. Peak tokens of the top 50 (gate-open/echo): 'zz' ×6 (0/6), ' compound' ×3 (3/0), ' comp' ×3 (3/0), 'ce' ×3 (0/3), 'nel' ×3 (0/3), 'p' ×2 (2/0) | no |
| 3. Negative pole | L5·5136-, max 0.39. The opposite pole is noise (-undy in Grundy). | – |
| 4. Blind maximum | Best of 3: 0.21, against 3.73 for the corpus maximum in context and 5.34 for the corpus top 8 read standalone (corpus p98.2; peak on 'ap', echo). The weakest of the corpus top 8 is 1.30 in context, 0.68 standalone; neutral sentences max 0.15. | no |
| 5. Near miss | 0.04 (corpus p27.9; 1% of the in-context maximum, 18% of the best blind maximum; peak on ' covered') | yes |
| **Verdict** | Fails. A weak unit (maximum 3.7) whose gate-open peaks are compound and comp- (compound, complicit, complicity, Thompkins); the three Batman Begins compound sentences gave the label. A name-free fear-toxin sentence scores 0.21 against a neutral maximum of 0.15. 74% of the top 50 are echo on junk tokens. | **fails** |

Suggested label: **The token comp(ound), weak**

Strongest corpus sentences (activation, regime, peak token):

- 3.73 open ' compound' · *Batman Begins*: Ducard trains Bruce to conquer his fear using a hallucinogenic compound derived from the mountain's blue flowers.
- 3.00 open ' compound' · *Batman Begins*: Falcone threatens to inform the police about a secret compound in the drugs unless Crane's mysterious employer cuts a deal with him.
- 2.41 open ' compound' · *Batman Begins*: At Bruce's birthday party, Ducard reveals himself as the true Ra's al Ghul, who employed Crane to weaponize the League's blue flower compound.
- 2.16 open ' comp' · *Return of the Joker*: Following the Jokerz' arrests, Barbara hides Tim's complicity to protect him while the Joker is declared dead.
- 1.82 open ' comp' · *Earth One*: Gordon learns that Bullock is indirectly responsible for the kidnapping, since he checked out the Wayne murder files under Gordon's name, and Gordon was threatened by Co…
- 1.61 open ' comp' · *Batman v Superman*: Believing Superman was complicit in the attack, Batman breaks into LexCorp and steals the kryptonite to weaponize it.
- 1.45 open 'p' · *Return of the Joker*: Over the next year, Tim recovered with help from Bruce's family friend Leslie Thompkins, but was forced to retire from superheroics by the latter, and severed ties with…
- 1.30 open 'p' · *Hush*: Later, in Gotham City, Bruce Wayne, Selina Kyle, Leslie Thompkins, and Dr. Elliot are attending the opera Pagliacci, when Harley Quinn tries to rob everyone in the theat…

Echo dominates; strongest gate-open firings:

- 3.73 ' compound' · *Batman Begins*: Ducard trains Bruce to conquer his fear using a hallucinogenic compound derived from the mountain's blue flowers.
- 3.00 ' compound' · *Batman Begins*: Falcone threatens to inform the police about a secret compound in the drugs unless Crane's mysterious employer cuts a deal with him.
- 2.41 ' compound' · *Batman Begins*: At Bruce's birthday party, Ducard reveals himself as the true Ra's al Ghul, who employed Crane to weaponize the League's blue flower compound.
- 2.16 ' comp' · *Return of the Joker*: Following the Jokerz' arrests, Barbara hides Tim's complicity to protect him while the Joker is declared dead.
- 1.82 ' comp' · *Earth One*: Gordon learns that Bullock is indirectly responsible for the kidnapping, since he checked out the Wayne murder files under Gordon's name, and Gordon was threatened by Co…
- 1.61 ' comp' · *Batman v Superman*: Believing Superman was complicit in the attack, Batman breaks into LexCorp and steals the kryptonite to weaponize it.
- 1.45 'p' · *Return of the Joker*: Over the next year, Tim recovered with help from Bruce's family friend Leslie Thompkins, but was forced to retire from superheroics by the latter, and severed ties with…
- 1.30 'p' · *Hush*: Later, in Gotham City, Bruce Wayne, Selina Kyle, Leslie Thompkins, and Dr. Elliot are attending the opera Pagliacci, when Harley Quinn tries to rob everyone in the theat…

Relabel from gate-open firings: **The token comp- (gate-open only)**

Negative pole L5·5136-, strongest sentences:

- 0.39 'y' · *Arkham City*: Tracking Penguin to his fortified museum, Batman defeats his forces, his imprisoned monster Solomon Grundy, and ultimately the Penguin himself, before liberating Mr. Fre…
- 0.38 'undy' · *Mrs. Doubtfire*: One day, the station's CEO, Jonathan Lundy, sees Daniel playing with props on the set of a recently cancelled children's show.
- 0.38 'y' · *The Long Halloween*: Dent is rushed to a hospital but escapes into the sewers, befriending Solomon Grundy when he encounters him.
- 0.37 'ka' · *The Grand Budapest Hotel*: In a cemetery in the former nation of Zubrowka, a young woman visits the shrine of a renowned writer, known simply as "Author", carrying a copy of his bestselling novel…
- 0.36 'y' · *Arkham City*: Other characters include the zombie Solomon Grundy (also voiced by Tatasciore), the shapeshifter Clayface (Rick D.
- 0.34 ' connects' · *The Notebook*: Noah cannot bring himself to sell the house since it connects him to Allie.
- 0.34 'way' · *Spider-Man*: That night, Norman captures Mary Jane and a Roosevelt Island Tramway car full of children.
- 0.33 ' body' · *Rear Window*: Now convinced his theory is true, Jeff looks at slides taken two weeks earlier and notices that Thorwald has replanted flowers in his garden, possibly to bury a body par…

The corpus top 8 read standalone (activation, in-context activation in the list above):

- 5.34 (p100.0, open, peak ' compound') · *Batman Begins*: Falcone threatens to inform the police about a secret compound in the drugs unless Crane's mysterious employer cuts a deal with him.
- 4.47 (p100.0, open, peak ' compound') · *Batman Begins*: At Bruce's birthday party, Ducard reveals himself as the true Ra's al Ghul, who employed Crane to weaponize the League's blue flower compound.
- 4.34 (p100.0, open, peak ' compound') · *Batman Begins*: Ducard trains Bruce to conquer his fear using a hallucinogenic compound derived from the mountain's blue flowers.
- 1.79 (p99.9, open, peak ' comp') · *Return of the Joker*: Following the Jokerz' arrests, Barbara hides Tim's complicity to protect him while the Joker is declared dead.
- 1.56 (p99.8, open, peak 'p') · *Return of the Joker*: Over the next year, Tim recovered with help from Bruce's family friend Leslie Thompkins, but was forced to retire from superheroics by the latter, and severed ties with…
- 1.41 (p99.8, open, peak ' comp') · *Earth One*: Gordon learns that Bullock is indirectly responsible for the kidnapping, since he checked out the Wayne murder files under Gordon's name, and Gordon was threatened by Co…
- 1.30 (p99.8, open, peak 'p') · *Hush*: Later, in Gotham City, Bruce Wayne, Selina Kyle, Leslie Thompkins, and Dr. Elliot are attending the opera Pagliacci, when Harley Quinn tries to rob everyone in the theat…
- 0.68 (p99.7, open, peak ' comp') · *Batman v Superman*: Believing Superman was complicit in the attack, Batman breaks into LexCorp and steals the kryptonite to weaponize it.

Blind maximum attempts:

- 0.21 (p98.2, echo, peak 'ap') blind_A: The burlap-masked doctor sprayed his fear toxin into her face, and within seconds she was shrieking at hallucinated nightmares crawling out of the walls.
- 0.04 (p27.6, echo, peak 'ucing') blind_C: The masked psychiatrist sprays his victims with a hallucinogenic fear toxin, a terror-inducing gas that makes them see their worst nightmares and scream in panic.
- 0.03 (p15.5, echo, peak 'rollable') blind_B: The masked psychiatrist sprays his victims with a hallucinogenic fear toxin that makes them see their worst nightmares and scream in uncontrollable terror.

Near miss (toxin, fear): The biology lecture covered the toxin produced by pufferfish, and the professor's fear of flying kept him from the Tokyo conference.

