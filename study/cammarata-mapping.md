# Concept-first mapping of the 31 named memes to Qwen3-8B neurons

The paper mapped memes to neurons label-first and the audit found the labels mostly lexical. This experiment goes the way Cammarata's essay works: search every gate-open unit for one that separates name-free minimal pairs of each concept (the paper's revenge-neuron method, generalised), score it on held-out pairs, then explain the found neuron from its corpus firings alone, let a second agent write a blind maximum, five examples and five near misses from that explanation, and steer the channel during generation. Produced by `cammarata_mapping.py` and `cammarata_mapping_report.py`; data in `cammarata-mapping.json`. All activations are gate-open and pole-signed; probe sentences are read standalone with the first token excluded, corpus sentences in the context of their work.

## Summary

**Headline.** Searching concept-first, the way Cammarata works, maps the 31 memes to different and better units than the paper's: 28 of the 31 units found by name-free minimal pairs pass six held-out pairs, against 4 of the paper's 31 units (mean wins 5.6 against 1.9 of 6), and none of the new units is the paper's. Blind explainers who never saw the labels call 23 of the 31 conceptual and 0 lexical, and an independent judge finds that 8 blind explanations describe the labelled concept outright and 17 describe a defining part of it (facial disfigurement for Harvey Dent, immortality for Ra's al Ghul, guilt for failing to save someone for losing the protégé). On Cammarata's full bar, held-out pairs plus a direct blind explanation plus Agent 2's examples beating its near misses, 6 memes have a concept unit: the secret identity, the heir, the ward, the villain's lair, the bomb and the coin; 18 do if a partial explanation is allowed. The audit of the paper's own labels had found 1.

**What the found units are.** They sit in layers 8 to 34 (median 15), none in the first eight. Agent 2's examples, written from the explanation alone, fire strongly 54% of the time and its near misses 5% (Cammarata reports 76% and 11% on Qwen3-32B); the explanation wins for 24 of 31. Eight blind maxima beat the unit's corpus maximum and 24 would have made its top 8. Steering the channel during generation makes the model write the concept for 4 units: wills and heirs in six of six generations for the inheritance unit, bombs in six of six, a lair in two, rebirth once for the immortality unit; the three unsteered generations were judged concept-free. Characters do not have single units in this model; their defining features do.

**What this means for the paper.** The paper's presence rule asks a unit to beat 68 of the 69 ordinary films on a work's three strongest sentences. Names do that; concepts do not, because inheritance, bombs and hideouts occur in ordinary films too. Under that rule the new units are present in a mean of 3.2 Batman works against 17.8 for the paper's units, while above the median ordinary film they are present in 22.7 against 21.8. The paper's core canon is therefore a canon of Batman-exclusive triggers, which at the single-channel level are names and Bat- compounds; the story patterns the labels name live in general-purpose units that a Batman-exclusivity rule cannot select. The one exception is the villain's lair, present in 21 works for the concept unit against 6 for the paper's word unit. A canon built from concept units would need a rule that scores works against the unit's own typical firing rather than against the films, and such a canon would not be Batman-specific, which is what the paper's E1 found on name-free texts.

**Caveats.** Six held-out pairs per meme is a short test, and with 884,736 candidates the chosen unit is one of thousands that clear the dev bar (2,000 to 13,000 per meme, near what chance gives for 9 of 10 wins), so the ranking by effect size is doing the work and 21 of 31 chosen units are specific (test positives above the unit's 99th percentile on the other memes' sentences). The pairs, explanations, Agent 2 sentences and judgements were all written by Claude agents, blind to each other but of one style. Agent 1 explained units from the paper's 3,700-sentence corpus rather than from billions of tokens, so the explanations are corpus-bound (the prison unit is read entirely from one film). Steering used two strengths and three prompts. One model, Qwen3-8B at 4 bits; nothing here was run on Llama.

| Count | of 31 |
| --- | --- |
| Memes with at least one unit passing the dev pairs (9 of 10 wins, mean activation ≥ 0.5) | 31 |
| Chosen unit passes the held-out test pairs (at least 5 of 6) | 28 |
| The paper's unit passes the same test pairs | 4 |
| Chosen unit is specific (test positives above its p99 on the other memes' sentences) | 21 |
| Chosen unit is the paper's unit | 0 |
| Blind explainer rates the unit conceptual (not lexical or mixed) | 23 |
| Neuron-first explanation (Agent 1, blind to the label) describes the concept directly / a defining part of it | 8 / 17 |
| Agent 2's blind maximum beats the unit's corpus maximum / would make its top 8 | 8 / 24 |
| Agent 2's examples beat its near misses (explanation wins) | 24 |
| Steering the channel makes the model write the concept | 4 |
| **Concept units** (test pairs pass, explanation direct, Agent 2's examples beat its near misses) | **6** |
| Concept units on the lenient bar (explanation direct or partial) | 18 |
| Mean test wins of 6: chosen unit / paper's unit | 5.58 / 1.94 |
| Mean Batman works (of 32) present under the paper's rule: new unit / paper's unit | 3.2 / 17.8 |
| Mean Batman works scoring above the median ordinary film: new unit / paper's unit | 22.7 / 21.8 |
| Agent 2 examples firing strongly / near misses firing strongly | 54% / 5% |

## Overview

| # | Meme | Paper's unit | dev / test | New unit | Layer | dev / test | Other-memes p99 | Explanation (Agent 1) | Match | Blind max / corpus max | Ex / NM strong | Steer | Presence (new / paper) | Concept unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | The origin wound | L10·9068+ | 10/10 · 6/6 | L26·1351- | 26 | 10/10 · 6/6 | 2.77 vs 1.66 | This unit fires in investigations of a death at the token just before the deduced cause o… | adjacent | 5.19 / 9.31 | 3/5 · 0/5 | no | 0 / 32 | no |
| 2 | The secret identity | L13·8608+ | 10/10 · 4/6 | L13·6897+ | 13 | 9/10 · 5/6 | 1.23 vs 1.06 | This unit fires on the word that completes a secret-identity equation, i.e. the realisati… | direct | 3.72 / 5.50 | 2/5 · 0/5 | no | 9 / 30 | **yes** |
| 3 | Heir to a dead father's fortune | L10·5634+ | 3/10 · 1/6 | L12·3817+ | 12 | 10/10 · 6/6 | 4.85 vs 9.75 | This unit fires on inheritance and bequest: 'bequeathed' (peaking on 'que'), 'bequest', '… | direct | 12.56 / 18.75 | 5/5 · 1/5 | yes | 0 / 29 | **yes** |
| 4 | The billionaire vigilante | L4·1072+ | 0/10 · 0/6 | L27·6120- | 27 | 9/10 · 6/6 | 9.13 vs 5.01 | This unit fires on 'war' when it names a declared personal campaign ('war on crime', 'war… | partial | 30.75 / 38.50 | 4/5 · 0/5 | no | 5 / 23 | no |
| 5 | The Batcave and bat gear | L11·9843+ | 7/10 · 4/6 | L12·8048- | 12 | 9/10 · 5/6 | 1.08 vs 1.60 | This unit fires on lines, tethers and grappling devices: 'grapple' (x2), 'grappling hook'… | partial | 5.75 / 6.94 | 3/5 · 0/5 | no | 1 / 30 | no |
| 6 | Bats | L8·7130+ | 9/10 · 5/6 | L10·2879+ | 10 | 9/10 · 6/6 | 0.78 vs 3.04 | This unit fires on small flying insects and swarms: 'fly', 'mosquitoes', 'moth', 'flying'… | adjacent | 17.38 / 7.50 | 4/5 · 0/5 | no | 0 / 24 | no |
| 7 | Gotham City | L6·10402+ | 0/10 · 1/6 | L11·10296+ | 11 | 9/10 · 6/6 | 1.10 vs 0.47 | This unit fires on the token that completes a New York place name or institution — upstat… | adjacent | 6.53 / 7.25 | 2/5 · 0/5 | no | 0 / 30 | no |
| 8 | Commissioner Gordon | L11·12069- | 3/10 · 1/6 | L8·9586+ | 8 | 9/10 · 6/6 | 1.01 vs 1.06 | This unit fires on the title of the person who heads an organisation, usually right befor… | partial | 2.86 / 3.06 | 2/5 · 0/5 | no | 0 / 30 | no |
| 9 | Losing the protégé | L11·7340- | 3/10 · 0/6 | L14·6104- | 14 | 10/10 · 6/6 | 2.84 vs 4.46 | This unit fires on guilt and self-blame for failing to save or protect someone: 'feels pa… | partial | 9.25 / 9.75 | 3/5 · 0/5 | no | 4 / 31 | no |
| 10 | The joke | L2·6468+ | 3/10 · 1/6 | L30·2954+ | 30 | 10/10 · 6/6 | 2.77 vs 6.26 | This unit is the weak side (max 10) of a smile/laughter channel: it fires on realised lau… | partial | 8.56 / 10.12 | 3/5 · 3/5 | no | 4 / 23 | no |
| 11 | The rogues' gallery | L10·5942+ | 5/10 · 4/6 | L34·5108+ | 34 | 10/10 · 6/6 | 14.00 vs 9.02 | This late-layer unit fires at the position where a villain's name or epithet is about to… | partial | 10.31 / 55.25 | 0/5 · 0/5 | no | 4 / 31 | no |
| 12 | Superhero powers (peaks on Superman) | L10·7189+ | 8/10 · 6/6 | L26·9345+ | 26 | 9/10 · 4/6 | 1.86 vs 1.75 | This unit fires in contexts of weight, heaviness and the strength needed to lift or light… | partial | 10.75 / 8.25 | 3/5 · 2/5 | no | 0 / 31 | no |
| 13 | Franchise roll-call (Wikipedia style) | L5·3626+ | 0/10 · 0/6 | L26·11032- | 26 | 10/10 · 6/6 | 3.59 vs 11.33 | This unit fires inside sentences about the Batman franchise, peaking on fragments of Batm… | direct | 13.19 / 22.75 | 0/5 · 1/5 | no | 11 / 23 | no |
| 14 | The butler who raised him | L6·4705- | 1/10 · 1/6 | L12·978- | 12 | 10/10 · 5/6 | 2.57 vs 2.50 | This unit fires on the modifier or noun that defines a close kinship or intimate relation… | partial | 4.25 / 4.84 | 4/5 · 0/5 | no | 2 / 19 | no |
| 15 | Taking in an orphan as a ward | L35·6441- | 0/10 · 0/6 | L29·4544+ | 29 | 10/10 · 5/6 | 4.51 vs 4.54 | This unit fires on a child or young creature who is being placed under someone's care, ab… | direct | 3.33 / 14.38 | 1/5 · 0/5 | no | 0 / 10 | **yes** |
| 16 | The villain's lair | L4·7177- | 4/10 · 2/6 | L10·605- | 10 | 10/10 · 6/6 | 7.08 vs 9.80 | This unit fires on the token that completes the name of a character's secret headquarters… | direct | 19.88 / 15.50 | 5/5 · 0/5 | yes | 21 / 6 | **yes** |
| 17 | A bomb | L3·1967- | 2/10 · 2/6 | L10·4927+ | 10 | 10/10 · 6/6 | 10.15 vs 13.45 | This unit fires on explosives and their use — bomb, bombs, C-4, explosives, detonate, blo… | direct | 19.50 / 24.62 | 5/5 · 0/5 | yes | 7 / 14 | **yes** |
| 18 | A funeral | L1·5621+ | 5/10 · 3/6 | L34·6351+ | 34 | 10/10 · 5/6 | 4.68 vs 10.24 | This late-layer unit fires at the position just before a spirit/soul-type word is produce… | adjacent | 26.12 / 32.25 | 4/5 · 0/5 | no | 0 / 9 | no |
| 19 | Fell into a vat of chemicals | L0·6344- | 7/10 · 4/6 | L15·7058- | 15 | 10/10 · 6/6 | 0.69 vs 0.96 | This unit fires very weakly on the proper nouns of fairy-tale and fantasy worlds, with ha… | none | 0.82 / 1.68 | 0/5 · 0/5 | no | 2 / 8 | no |
| 20 | Two-Face's coin | L26·3329- | 10/10 · 5/6 | L12·4733- | 12 | 10/10 · 6/6 | 1.08 vs 5.52 | This unit fires on the word that picks out one of the two sides of a two-sided object, ab… | direct | 6.91 / 8.25 | 5/5 · 0/5 | no | 3 / 9 | **yes** |
| 21 | Harvey Dent | L5·6648+ | 0/10 · 0/6 | L17·7382- | 17 | 9/10 · 6/6 | 2.97 vs 3.63 | This unit fires on physical disfigurement and deformity, especially of the face: 'disfigu… | partial | 6.62 / 7.00 | 2/5 · 0/5 | no | 9 / 9 | no |
| 22 | Selina Kyle | L1·1051+ | 0/10 · 0/6 | L32·5618+ | 32 | 9/10 · 4/6 | 1.80 vs 1.73 | This unit fires on the first sub-word fragment of a word the tokenizer has split in two —… | none | 2.28 / 16.75 | 0/5 · 0/5 | no | 0 / 7 | no |
| 23 | The Riddler | L1·10818- | 4/10 · 2/6 | L12·11755- | 12 | 9/10 · 6/6 | 0.66 vs 0.95 | This unit fires on riddles, quiz or test questions and the act of answering them correctl… | partial | 3.62 / 3.52 | 3/5 · 0/5 | no | 1 / 10 | no |
| 24 | Jason Todd | L4·4504- | 0/10 · 0/6 | L11·823- | 11 | 10/10 · 6/6 | 2.19 vs 1.15 | This unit fires on the word that names a dead or mutilated body or completes the act of k… | partial | 10.19 / 7.47 | 3/5 · 0/5 | no | 0 / 8 | no |
| 25 | Ra's al Ghul | L4·9760- | 0/10 · 0/6 | L8·8851- | 8 | 9/10 · 6/6 | 1.57 vs 6.07 | This unit fires on words meaning immortality, permanence, or not dying — immortality, Imm… | partial | 11.94 / 9.19 | 5/5 · 0/5 | yes | 2 / 8 | no |
| 26 | Lucius Fox | L5·10113- | 0/10 · 0/6 | L9·3350+ | 9 | 9/10 · 4/6 | 0.71 vs 0.55 | This unit fires on words that describe advanced technology or a scientific breakthrough —… | partial | 7.84 / 3.05 | 4/5 · 0/5 | no | 4 / 14 | no |
| 27 | A corrupt police force | L35·4074- | 5/10 · 1/6 | L15·5647+ | 15 | 9/10 · 5/6 | 1.54 vs 1.86 | This unit fires mid-construction on the token where a corrupt or collusive deal is being… | partial | 2.53 / 4.56 | 0/5 · 0/5 | no | 2 / 13 | no |
| 28 | Prison and the inmates released | L4·8535+ | 2/10 · 2/6 | L31·9620- | 31 | 10/10 · 6/6 | 17.23 vs 41.42 | This unit fires at very high strength on function words (to, by, after, at, from, the) ru… | partial | 106.50 / 134.00 | 2/5 · 0/5 | no | 0 / 20 | no |
| 29 | The antidote | L1·10779- | 3/10 · 2/6 | L27·3551- | 27 | 10/10 · 6/6 | 1.76 vs 5.20 | This unit fires in sentences about a poison, toxin or disease at the point where a cure o… | direct | 9.75 / 16.50 | 0/5 · 0/5 | no | 8 / 7 | no |
| 30 | A mutating super-formula | L3·10668+ | 3/10 · 2/6 | L33·5542- | 33 | 10/10 · 6/6 | 6.82 vs 8.93 | This unit fires when a character's behaviour or nature is being described as changed or a… | partial | 44.50 / 42.25 | 2/5 · 1/5 | no | 0 / 6 | no |
| 31 | Fear toxin | L5·5136+ | 1/10 · 1/6 | L22·4641- | 22 | 9/10 · 5/6 | 2.75 vs 7.79 | This unit fires on the word that introduces an imagined, delusional or illusory perceptio… | partial | 13.94 / 15.00 | 4/5 · 0/5 | no | 1 / 8 | no |

## Per-meme detail

### 1. The origin wound

**Paper's unit L10·9068+** (layer 10): dev 10/10, test 6/6 (AUC 0.861); gate-open corpus max 18.75, present in 32 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Dent' ×8, ' Bat' ×8, ' Wayne' ×7, ' Gordon' ×4, ' Drake' ×3, ' Todd' ×3.

**New unit L26·1351-** (layer 26): 5587 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 3.552, neg 0.366); test 6/6 (AUC 0.889; mean pos 1.664, neg 0.205); p99 on the other memes' sentences 2.77 (max 5.656). Corpus max 9.312, p99 3.328; present in 0 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 0, only new 0 (), only paper 32 (1943 serial, A Death in the Family, Arkham Asylum (comic), Arkham Asylum (game), Arkham City, Arkham Knight, Batman & Robin, Batman '66). Batman works scoring above the median ordinary film: 19 of 32 for the new unit, 32 for the paper's.
- Top-50 peak tokens: "'s" ×6, ' was' ×4, ' the' ×4, ' a' ×3, "'" ×2, ' fatally' ×2.
- Agent 1 (blind to the label, rates it conceptual, high confidence) explains it as: *This unit fires in investigations of a death at the token just before the deduced cause or verdict is stated: 'the mayor's theory that [it was a boating accident]', 'died by [suicide]', 'deduces that Drebber was [poisoned]', 'concludes it was a [shark attack]', 'infers that the dead trooper was [ticketing]', 'could have been [perpetrated by]', 'presumed to have been a random [homeless man]', 'body shows no [wounds]'. It anticipates a forensic or detective conclusion about how someone died. The opposite pole fires on adverbs of immediacy and closeness: 'immediately', 'shortly', 'closely', 'close to death'.*
- Judge: **adjacent**. It captures a detective's deduced cause of death, which shares the theme of a death but is a forensic conclusion rather than the formative loss that defines the origin wound. Label the explanation supports: *Anticipating a deduced cause of death*
- Agent 2 blind maximum: 5.19 against corpus max 9.31 (weakest of the top 8: 5.25; peak ' that'): Examining the body, the detective deduces that the victim was poisoned and concludes it was murder, rejecting the coroner's theory that he died by suicide.
- Agent 2 examples firing strongly (≥ 50% of max): 3 of 5; near misses: 0 of 5.
- Steering: α=9.31: none; α=9.31: none; α=9.31: none; α=18.62: none; α=18.62: none; α=18.62: none
- **Verdict:** Not the origin wound. The search found a forensic unit (the deduced cause of a death) that the murdered-parents pairs trigger; name-free, held-out 6 of 6, but the blind reader sees a detective's verdict, not a formative loss.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 9.31 ' that' · *Jaws*: The coroner, apparently under pressure, now concurs with the mayor's theory that it was a boating accident.
- 7.81 ' by' · *Knives Out*: Police detectives Lieutenant Elliot and Trooper Wagner believe Harlan died by suicide, but private detective Benoit Blanc is anonymously hired to investigate H…
- 7.31 ' was' · *Sherlock Holmes*: Dismissing the message as a ploy to mislead the police, Holmes deduces that Drebber was poisoned and supplies a description of the murderer.
- 6.91 ' a' · *Jaws*: After the coroner concludes it was a shark attack, Amity police chief Martin Brody plans to close the beaches.
- 6.44 ' was' · *Fargo*: She correctly infers that the dead trooper was ticketing a car with dealer plates.
- 5.50 ' been' · *The Batman*: Alfred has suspicions that Falcone had Thomas and Martha killed to silence them, but has no proof of it, stating that the murder could have been perpetrated by…
- 5.34 ' random' · *Parasite*: Geun-sae is presumed to have been a random homeless man, and the motives for the murders remain unsolved.
- 5.25 ' no' · *Sherlock Holmes*: The room is splashed in blood, and the word "RACHE" (German for "revenge") is written in it on one wall; however, Drebber's body shows no wounds.

Its negative pole L26·1351+ (max 9.12):

- 9.12 ' immediately' · *Knightfall*: Despite the advice of everyone in his life, including Dr. Kinsolving, Bruce refuses to rest and continues to pursue his self-imposed duty despite his worsening…
- 5.25 ' closely' · *The Grand Budapest Hotel*: Zero rescues Gustave by pushing Jopling off the cliff, and the two men flee, closely trailed by Zubrowkan police.
- 5.19 ' immediately' · *Taxi Driver*: Betsy agrees to go on a second date with him, during which he takes her to an adult movie theater, which she leaves immediately.
- 5.00 ' and' · *Rear Window*: Lacking proof to accuse Thorwald of murder, Lisa allows herself to be arrested for breaking and entering to be taken away safely.
- 4.03 ' immediate' · *Sherlock Holmes*: Drebber, a drunk, attempted to kiss Mrs. Charpentier's daughter, Alice, which caused their immediate eviction.

Agent 2's examples and near misses (activation, share of corpus max):

- example 7.47 (80%) ' a': The sheriff concludes the swimmer's death was a shark attack, though the mayor insists it was a boating accident.
- example 3.28 (35%) ' was': The body shows no wounds, so the pathologist infers that the heiress was smothered in her sleep.
- example 2.30 (25%) ' was': The consulting detective deduces that the dead man was poisoned by a pill forced on him by a cab driver.
- example 7.69 (83%) ' by': Police rule that the billionaire died by suicide, but his butler suspects the fall was staged.
- example 6.44 (69%) ' was': From the ticket book in his hand, the rookie infers that the dead trooper was writing a citation when he was shot.
- near miss 0.27 (3%) ' races': Batman deduces that the Joker's next target is the mayor's parade and races to the rooftop.
- near miss 2.59 (28%) ' the': Harvey Dent dies in the fall from the half-built tower and Gordon covers up the truth immediately.
- near miss 0.08 (1%) ' son': Lucius Fox's theory that the sonar could map the whole city is proved right during the final showdown.
- near miss 0.04 (0%) ' noon': The board meeting concludes shortly after noon and Bruce Wayne leaves closely followed by the press.
- near miss 0.15 (2%) ' the': Rachel's body lies in the warehouse rubble while Bruce sits in silence at Wayne Manor.

Steered generations:

- α=9.31 · Plot summary: The film opens as → a young woman, named Lina, is walking through the streets of a city. She is carrying a small bag of groceries. She is followed by a man, who seems to be a stalker. She is trying to avoid him,
- α=9.31 · Plot: In the first act, → the story introduces the protagonist, a young woman named Clara who is struggling to find her place in the world. She is working as a barista in a small town, and feels trapped in her mundane life. O…
- α=9.31 · The story begins when → the narrator, a young man, is walking through the forest. He is alone, and he feels a bit uneasy. He hears a sound, like a footstep, and turns around. But there is no one there. He continues walking
- α=18.62 · Plot summary: The film opens as → a young woman, named Lina, is walking through a forest. She is followed by a man, who seems to be a hunter. She is trying to escape, but he is gaining on her. She runs into a clearing, where
- α=18.62 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elara, who lives in a small village. She is known for her kindness and her ability to communicate with animals. One day, she discovers a myst…
- α=18.62 · The story begins when → the narrator, a young man, is walking through the forest. He is alone, and he is thinking about his life. He is thinking about his past, his family, and his future. He is also thinking about the fore…

### 2. The secret identity

**Paper's unit L13·8608+** (layer 13): dev 10/10, test 4/6 (AUC 0.75); gate-open corpus max 3.266, present in 30 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' identity' ×15, 'ante' ×5, ' masked' ×4, ' Spider' ×3, ' spider' ×3, '4' ×2.

**New unit L13·6897+** (layer 13): 2333 units passed the dev bar; this one scored best. Dev 9/10 (mean pos 1.729, neg 0.115); test 5/6 (AUC 0.819; mean pos 1.057, neg 0.188); p99 on the other memes' sentences 1.232 (max 2.484). Corpus max 5.5, p99 1.938; present in 9 of 32 Batman works, 5 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 9, only new 0 (), only paper 21 (1943 serial, A Death in the Family, Arkham Asylum (comic), Arkham Asylum (game), Arkham City, Arkham Knight, Batman & Robin, Batman '89). Batman works scoring above the median ordinary film: 31 of 32 for the new unit, 32 for the paper's.
- Top-50 peak tokens: ' identity' ×9, ' Batman' ×8, ',' ×7, ' him' ×3, ' is' ×3, ' same' ×2.
- Agent 1 (blind to the label, rates it conceptual, high confidence) explains it as: *This unit fires on the word that completes a secret-identity equation, i.e. the realisation, revelation or question that two personas are the same person: 'reveal he is Batman', 'the same man', 'the same person', 'deduces that he is Spider-Man', 'Wayne is Batman's alter-ego', 'the Goblin's identity', 'Dent may be Holiday'. 'identity' and 'Batman' dominate its top tokens; the opposite pole is negligible.*
- Judge: **direct**. It describes the moment two personas are revealed or deduced to be the same person, with 'identity' and 'alter-ego' dominant, which is the secret identity itself. Label the explanation supports: *Revealing two personas are the same person*
- Agent 2 blind maximum: 3.72 against corpus max 5.50 (weakest of the top 8: 4.12; peak ' same'): The reporter finally deduces that the billionaire and the caped vigilante are the same person, revealing the masked hero's secret identity.
- Agent 2 examples firing strongly (≥ 50% of max): 2 of 5; near misses: 0 of 5.
- Steering: α=5.5: weak Heir to a dead father's fortune (letter is from his father, who died); α=5.5: none; α=5.5: none; α=11.0: weak Franchise roll-call (Wikipedia style) (Jack (Leonardo DiCaprio)); α=11.0: none; α=11.0: none
- **Verdict:** Concept unit. Fires where two personas are found to be the same person; the blind explanation is the secret identity itself, Agent 2's examples fire and no near miss does. Present in 9 Batman works under the paper's rule against 30 for the word unit, because unmaskings happen in Zorro and Spider-Man too.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 5.50 ' Batman' · *Gotham by Gaslight*: Sentenced to Blackgate Penitentiary following the trial, Bruce is met by Selina, who urges him to reveal he is Batman to clear his name and help save the girls…
- 5.28 ' man' · *Gotham by Gaslight*: Citizens of Gotham believe the Batman and Jack to be the same man.
- 4.75 ' Batman' · *Batman '66*: Catwoman, disguised as Soviet journalist Kitka, helps the group kidnap Bruce Wayne and pretends to be kidnapped with him, as part of a plot to lure Batman and…
- 4.75 ' same' · *Some Like It Hot*: Realizing that Josephine and Junior are the same person, Sugar follows them.
- 4.41 '-Man' · *Spider-Man*: Norman notices Peter's injury and deduces that he is Spider-Man.
- 4.38 ' identity' · *Spider-Man*: Peter is still unaware of the Goblin's identity, but realizes that the Goblin is targeting his loved ones.
- 4.19 ',' · *The Long Halloween*: Gordon deduces Dent may be Holiday, but Batman refuses to believe it until he can talk to Dent himself.
- 4.12 'devil' · *Daredevil*: As the police arrive to arrest Fisk, he threatens to reveal Daredevil's identity to the world, but Matt points out that no one will ever believe that Daredevil…

Its negative pole L13·6897- (max 0.93):

- 0.93 ' reactor' · *Iron Man*: Stark and Yinsen secretly build a small, powerful electric generator called an arc reactor to power Stark's electromagnet and construct a prototype armored sui…
- 0.69 ' coron' · *Jaws*: After the coroner concludes it was a shark attack, Amity police chief Martin Brody plans to close the beaches.
- 0.65 'y' · *Dracula*: A note by Jonathan Harker seven years later states that the Harkers have a son, named Quincey after their friend.
- 0.65 ' Lucy' · *Dracula*: Lucy Westenra's letter to her best friend, Harker's fiancée Mina Murray, describes her marriage proposals from Dr John Seward, Quincey Morris, and Arthur Holmw…
- 0.62 ' Hill' · *Goodfellas*: In 1955, teenager Henry Hill becomes enamored by the criminal life and Mafia presence in East New York, a working class Italian American neighborhood in Brookl…

Agent 2's examples and near misses (activation, share of corpus max):

- example 2.39 (44%) ' same': The detective realises the mild-mannered photographer and the web-slinging hero are the same man.
- example 1.67 (30%) ' the': At the climax the clown unmasks the vigilante and reveals to the city that he is the billionaire.
- example 4.25 (77%) 'enger': Her suspicion grows until she confronts him with proof that the masked avenger is her own husband.
- example 1.27 (23%) '.': The boy deduces that his quiet neighbour must be the costumed crime-fighter from the news reports.
- example 3.27 (59%) ' same': The prosecutor suspects that the serial killer and the district attorney may be one and the same.
- near miss 0.02 (0%) ' drives': Batman drives the Batmobile through Gotham, chasing the Joker's henchmen across the bridge at midnight.
- near miss 0.42 (8%) ' and': The thief steals her passport and credit cards, and she spends months fighting the identity theft.
- near miss 0.00 (0%) None: The two brothers arrive at the same restaurant at the same time and order the same meal.
- near miss 0.42 (8%) ' that': The sequel reveals that the villain's lair is hidden beneath the old amusement park on the pier.
- near miss 0.88 (16%) ' of': Harvey Dent gives a speech about the identity of Gotham as a city of hope rather than fear.

Steered generations:

- α=5.5 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is searching for something. He finds a small, red box. He opens it, and finds a letter inside. The letter is from his father, who died in a
- α=5.5 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is struggling to find her place in the world. She is an aspiring artist, but her family expects her to pursue a more stable career…
- α=5.5 · The story begins when → the narrator, a young man, is walking through the forest. He is in search of a certain tree, which he has heard is the oldest tree in the world. He finds the tree, but then he is told by an old man
- α=11.0 · Plot summary: The film opens as → a young man, Jack (Leonardo DiCaprio), is walking through a forest. He is followed by a boy, who is clearly his son, named Christopher. Jack is in a wheelchair, and Christopher is carrying a backpack…
- α=11.0 · Plot: In the first act, → the story introduces the protagonist, a young woman named Emily who is struggling to find her place in the world. She is an aspiring writer who has recently moved to a small town to escape the hustle…
- α=11.0 · The story begins when → the narrator, a young man, is walking through the forest. He is thinking about his past, and he remembers the time when he was a child. He was playing in the forest with his dog, and he was very happ…

### 3. Heir to a dead father's fortune

**Paper's unit L10·5634+** (layer 10): dev 3/10, test 1/6 (AUC 0.583); gate-open corpus max 1.453, present in 29 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Wayne' ×31, 'son' ×3, ' Penguin' ×3, '.' ×2, ' Todd' ×2, ' Planet' ×2.

**New unit L12·3817+** (layer 12): 4561 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 8.637, neg 0.927); test 6/6 (AUC 1.0; mean pos 9.75, neg 1.585); p99 on the other memes' sentences 4.851 (max 10.688). Corpus max 18.75, p99 3.469; present in 0 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 0, only new 0 (), only paper 29 (1943 serial, A Death in the Family, Arkham Asylum (game), Arkham City, Arkham Knight, Batman & Robin, Batman '66, Batman '89). Batman works scoring above the median ordinary film: 22 of 32 for the new unit, 32 for the paper's.
- Top-50 peak tokens: 'lan' ×6, ' inheritance' ×5, ' wealthy' ×5, ' will' ×4, ' his' ×4, 'que' ×3.
- Agent 1 (blind to the label, rates it conceptual, high confidence) explains it as: *This unit fires on inheritance and bequest: 'bequeathed' (peaking on 'que'), 'bequest', 'inherit', 'estate is left to', 'leaves his possessions to his sons', and a will becoming 'effective', with 'inheritance', 'wealthy', 'will' and Harlan (whose will is read) among its top tokens; it tracks property passing to heirs on someone's death. The opposite pole is negligible.*
- Judge: **direct**. It describes inheritance and bequest, property passing to heirs on death, which is the defining idea of the labelled trope. Label the explanation supports: *Inheritance and bequest on death*
- Agent 2 blind maximum: 12.56 against corpus max 18.75 (weakest of the top 8: 10.31; peak 'que'): When the wealthy patriarch dies, his will is read and his entire estate is bequeathed to his sons, who inherit the fortune.
- Agent 2 examples firing strongly (≥ 50% of max): 5 of 5; near misses: 1 of 5.
- Steering: α=18.75: weak Heir to a dead father's fortune (she has a small inheritance) (concept); α=18.75: strong Heir to a dead father's fortune (a will that leaves her entire estate) (concept); α=18.75: strong Heir to a dead father's fortune (the sole heir of a large estate) (concept); α=37.5: strong Heir to a dead father's fortune (leaves his entire estate to his nephew) (concept); α=37.5: strong Heir to a dead father's fortune (the son, who is named as the heir) (concept); α=37.5: strong Heir to a dead father's fortune (leaves all her money to her nephew) (concept)
- **Verdict:** Concept unit, and the clearest case. Inheritance and bequest; the blind maximum reaches 67% of the corpus maximum, all five examples fire, and steering the channel makes the model write about wills and heirs in six of six generations.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 18.75 'que' · *Knives Out*: At the reading of Harlan's will, Marta is bequeathed his entire fortune and property, stunning the Thrombeys.
- 16.62 'que' · *The Grand Budapest Hotel*: Kovacs announces that she has bequeathed a famous Renaissance painting, Boy with Apple, to Gustave.
- 15.12 'que' · *Mamma Mia!*: Bill learns that Donna received the money for her villa from his great-aunt Sofia whereas he had understood the money was bequeathed "to family"; as a result,…
- 12.62 ' be' · *Pride & Prejudice*: Angry and heartbroken, Darcy delivers Elizabeth a letter describing Wickham's true character: Wickham squandered the bequest Darcy's father left him, then atte…
- 12.06 ' to' · *Dark Knight Rises*: Since Bruce is also presumed dead, Wayne Manor becomes an orphanage, and Bruce's estate is left to Alfred.
- 11.31 ' inherit' · *Pride & Prejudice*: Jane recovers and, soon after, Mr Bennet's cousin Mr Collins, a pompous clergyman, visits the Bennets; as the closest male relative, Collins will inherit Longb…
- 10.44 ' effective' · *The Grand Budapest Hotel*: Serge tells them that Madame D. had a missing second will which would only become effective if she was murdered.
- 10.31 ' to' · *Puss in Boots*: The story begins with a poor miller who, upon his death, leaves his possessions to his three sons.

Its negative pole L12·3817- (max 2.12):

- 2.12 'ory' · *Finding Nemo*: He meets Dory, a blue tang with short-term memory loss who offers to help him locate his son.
- 2.11 ' Blanc' · *Knives Out*: Police detectives Lieutenant Elliot and Trooper Wagner believe Harlan died by suicide, but private detective Benoit Blanc is anonymously hired to investigate H…
- 2.00 'iosa' · *Mad Max: Fury Road*: Meanwhile, Joe sends his lieutenant Imperator Furiosa in the armoured "War Rig" to trade produce and water for petrol and ammunition with two of his allies, th…
- 1.96 'q' · *Raiders of the Lost Ark*: Rival archaeologist René Belloq corners him and steals the idol; Jones escapes in a waiting float plane.
- 1.90 'e' · *Arkham City*: Stana Katic lends her voice as Talia al Ghul, and Maurice LaMarche voices both Mr. Freeze and Calendar Man.

Agent 2's examples and near misses (activation, share of corpus max):

- example 11.69 (62%) ' death': After the old millionaire's death, his will leaves the mansion and all his possessions to his estranged daughter.
- example 17.12 (91%) ' from': The orphan learns he has inherited a vast estate from a grandfather he never met.
- example 17.88 (95%) 'que': A dying widow bequeaths her fortune to the gardener, and her furious children contest the will.
- example 9.81 (52%) ' will': When the family patriarch is found dead, the reading of his will reveals the inheritance goes to his nurse.
- example 14.06 (75%) ' the': The heirs gather at the lawyer's office to hear who receives the estate now that their uncle has died.
- near miss 3.83 (20%) ' wealthy': Harlan Thrombey will host a lavish party at his estate on Saturday, inviting his wealthy neighbors from across the county.
- near miss 0.10 (1%) ' leaves': The detective leaves his possessions in the car and walks to the housing estate where the body was found.
- near miss 10.50 (56%) ' nothing': She will inherit nothing but trouble if she keeps dating the wealthy playboy, her mother warns.
- near miss 1.09 (6%) ' wealthy': The wealthy banker buys a second yacht and sails it to Monaco for the summer season.
- near miss 4.00 (21%) ' will': The soldier's will to survive carries him across the desert after his unit is left behind.

Steered generations:

- α=18.75 · Plot summary: The film opens as → the narrator, a young woman, is in her late 20s, and she has a small inheritance. She is living in a small town in the state of New York. She has a sister, who is living in New York
- α=18.75 · Plot: In the first act, → the story is set in the year 2000. The protagonist, a young woman named Clara, is found to have a will that leaves her entire estate to her estranged cousin, Henry. Clara is shocked by this news and
- α=18.75 · The story begins when → the narrator, a young woman named Clara, is found to be the sole heir of a large estate. She is told by the lawyer that she is the only one who will inherit the property, and she must sign a document…
- α=37.5 · Plot summary: The film opens as → a will, which leaves his entire estate to his nephew, who is a distant relative. The nephew is shocked by the inheritance and challenges the will in court. The story is about a man who dies and leave…
- α=37.5 · Plot: In the first act, → the estate is left to the son, who is named as the heir. The son is a young man who has a wife and child. The son is not happy with the inheritance, so he marries his cousin, who is a wealthy
- α=37.5 · The story begins when → the narrator, who is a woman, dies and leaves all her money to her nephew, who is a distant relative. The nephew is shocked because he was not told about the inheritance, and he feels betrayed. The f…

### 4. The billionaire vigilante

**Paper's unit L4·1072+** (layer 4): dev 0/10, test 0/6 (AUC 0.5); gate-open corpus max 18.625, present in 23 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Gordon' ×49, 'ideon' ×1.

**New unit L27·6120-** (layer 27): 3748 units passed the dev bar; this one scored best. Dev 9/10 (mean pos 11.035, neg 1.278); test 6/6 (AUC 0.944; mean pos 5.013, neg 0.786); p99 on the other memes' sentences 9.127 (max 46.0). Corpus max 38.5, p99 8.064; present in 5 of 32 Batman works, 2 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 5, only new 0 (), only paper 18 (A Death in the Family, Arkham Asylum (comic), Arkham Asylum (game), Arkham Knight, Batman '66, Batman '89, Batman Begins, Batman R.I.P.). Batman works scoring above the median ordinary film: 31 of 32 for the new unit, 24 for the paper's.
- Top-50 peak tokens: ' war' ×7, ' declares' ×4, ' his' ×4, ' a' ×4, ' declare' ×3, ' secret' ×3.
- Agent 1 (blind to the label, rates it mixed, medium confidence) explains it as: *This unit fires on 'war' when it names a declared personal campaign ('war on crime', 'war against crime'), on 'declare(s)' when someone formally proclaims hostility or a status, and on 'de' in 'nom de guerre' (anticipating guerre = war); it responds to the idea of declared, campaign-like hostility rather than to literal battles. The opposite pole is weak and has no clear theme.*
- Judge: **partial**. It captures a declared personal war on crime, which is what the vigilante does, but nothing of wealth, the costume or vigilantism itself. Label the explanation supports: *A declared war or campaign*
- Agent 2 blind maximum: 30.75 against corpus max 38.50 (weakest of the top 8: 15.56; peak ' war'): The vigilante declares war on crime, proclaiming a one-man war against the city's underworld under his chosen nom de guerre.
- Agent 2 examples firing strongly (≥ 50% of max): 4 of 5; near misses: 0 of 5.
- Steering: α=38.5: none; α=38.5: none; α=38.5: none; α=77.0: none; α=77.0: none; α=77.0: none
- **Verdict:** Partial. A declared war or crusade (war on crime, declares) rather than wealth plus vigilantism; held-out 6 of 6, four of five examples fire, no near miss does. Closer to the meme than the paper's Gordon detector by any measure.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 38.50 ' war' · *Dark Knight Returns*: Some time afterward, Bruce Wayne, now looking forward to life, leads Robin, Queen, and the rest of his followers into the caverns beyond the Batcave to continu…
- 34.00 ' war' · *Year One*: He sits before his father's bust, requesting guidance in his war against crime.
- 20.88 ' war' · *Year One*: Billionaire Bruce Wayne returns home to Gotham City after twelve years abroad, training for a one-man war against crime.
- 19.50 ' declares' · *Dark Knight Returns*: Gordon's successor as commissioner, Captain Ellen Yindel, declares Batman a wanted criminal.
- 18.25 ' declare' · *Heat*: Both men express mutual respect, but declare they will kill the other if necessary.
- 16.88 ' C' · *Mean Girls*: After violating the Plastics' dress code rules by wearing sweatpants on Monday due to her weight gain, Regina is ousted from the group, as Cady replaces her as…
- 15.56 ' de' · *Scarlet Pimpernel*: Their leader, the mysterious Scarlet Pimpernel, takes his nom de guerre from the small, red wayside flower he draws on his messages.
- 15.56 'nof' · *Green Hornet*: Chudnofsky, Scanlon, and their men chase Britt and Kato throughout L.A., ending in a fight at the Daily Sentinel office.

Its negative pole L27·6120+ (max 4.75):

- 4.75 ' D' · *The Social Network*: Marylin Delpy, a junior lawyer for Zuckerberg, tells Zuckerberg they will settle with Saverin because the sordid details of Facebook's founding and Zuckerberg'…
- 3.78 ',' · *Casablanca*: At the last moment, Rick makes Ilsa board the plane to Lisbon with Laszlo, telling her that she would regret it if she stayed, "Maybe not today, maybe not tomo…
- 2.70 ' of' · *Puss in Boots*: He presents these animals as gifts to the king, claiming they are from his master, who he names the "Marquis of Carabas".
- 2.64 ' for' · *The Shadow*: The Tulku, a mysterious holy man who exhibits psychic abilities, abducts Cranston and offers him a chance to become a force for good.
- 2.56 ' makes' · *The Shining*: Wendy finds Jack's manuscript written with nothing but countless repetitions of the proverb "All work and no play makes Jack a dull boy".

Agent 2's examples and near misses (activation, share of corpus max):

- example 27.62 (72%) ' declares': After his parents are murdered, the orphan declares war on crime and vows to clean up the city.
- example 31.38 (82%) ' war': The district attorney declares a war against corruption, publicly naming every crooked official in the city.
- example 16.62 (43%) ' war': The mayor declares a state of emergency and announces his own war on the mob families.
- example 41.25 (107%) ' personal': A reformed thief adopts a nom de guerre and wages a personal war against the cartel that betrayed him.
- example 21.75 (56%) ' war': The widow declares open war on the syndicate, hunting its leaders one by one across the city.
- near miss 5.78 (15%) ' war': During the war, the soldiers fight across the muddy fields of France, dodging shells in the trenches.
- near miss 0.00 (0%) ' the': At customs, the traveler must declare the bottle of wine he bought in Bordeaux.
- near miss 5.66 (15%) ' the': The clone troopers fight across distant planets in the Clone Wars, battling droid armies in the mud.
- near miss 0.00 (0%) None: The chef's pâté de foie gras wins the Cordon Bleu prize at the Paris competition.
- near miss 9.00 (23%) ' the': The veteran returns home from the war in Vietnam, haunted by the battles he fought in the jungle.

Steered generations:

- α=38.5 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is carrying a backpack and seems to be in a hurry. He is followed by a man in a black coat. Jack is trying to avoid being caught. The man in
- α=38.5 · Plot: In the first act, → the story introduces the main characters and the setting. The protagonist, a young woman named Emily, is struggling with her career and personal life. She works as a freelance writer, but her latest…
- α=38.5 · The story begins when → the narrator, a young man, is caught in a rainstorm in the mountains. He takes shelter in a cave, where he discovers a hidden room. In this room, he finds a journal belonging to a man named Victor, w…
- α=77.0 · Plot summary: The film opens as → the war in Europe is ending. The war in the Pacific is still going on. In the war's aftermath, the United States is in a state of peace. The story is told through the eyes of a young man, Robert, who
- α=77.0 · Plot: In the first act, → the story introduces the main characters, the war on drugs, and the conflict between the two sides. The second act escalates the war, with the two sides engaging in a war of attrition. The third act…
- α=77.0 · The story begins when → the war is declared. The narrator is a young man who is at war. He is a soldier in the war. He is in the war because he is a part of the army. He is in the war because he is a part

### 5. The Batcave and bat gear

**Paper's unit L11·9843+** (layer 11): dev 7/10, test 4/6 (AUC 0.694); gate-open corpus max 5.656, present in 30 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Bat' ×43, 'ats' ×3, 'Bat' ×3, ' bat' ×1.

**New unit L12·8048-** (layer 12): 3725 units passed the dev bar; this one scored best. Dev 9/10 (mean pos 1.924, neg 0.133); test 5/6 (AUC 0.875; mean pos 1.596, neg 0.297); p99 on the other memes' sentences 1.082 (max 2.938). Corpus max 6.938, p99 1.235; present in 1 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 1, only new 0 (), only paper 29 (1943 serial, A Death in the Family, Arkham Asylum (comic), Arkham Asylum (game), Arkham City, Arkham Knight, Batman & Robin, Batman '66). Batman works scoring above the median ordinary film: 29 of 32 for the new unit, 32 for the paper's.
- Top-50 peak tokens: ' line' ×3, ' Anna' ×3, 'apple' ×2, ' gun' ×2, ' dream' ×2, ' Matrix' ×2.
- Agent 1 (blind to the label, rates it conceptual, medium confidence) explains it as: *This unit fires on lines, tethers and grappling devices: 'grapple' (x2), 'grappling hook', 'grappling hook gun', 'grapple gun', 'tangled parachute tethers', a harpoon 'line', and 'Ariadne' (the thread-giver), i.e. ropes and cables that attach, pull or snag. The opposite pole is negligible.*
- Judge: **partial**. It captures grappling hooks, lines and tethers, one piece of bat gear, but nothing of the Batcave or the wider kit. Label the explanation supports: *Grappling hooks, tethers and lines*
- Agent 2 blind maximum: 5.75 against corpus max 6.94 (weakest of the top 8: 3.59; peak ' gun'): The vigilante fires his grapple gun, and the grappling hook's steel line snags the ledge as the tangled tether pulls him upward.
- Agent 2 examples firing strongly (≥ 50% of max): 3 of 5; near misses: 0 of 5.
- Steering: α=6.94: none; α=6.94: none; α=6.94: none; α=13.88: none; α=13.88: none; α=13.88: none
- **Verdict:** Partial: grappling hooks, lines and tethers, one item of the kit. Held-out 5 of 6. The gear has a unit; the cave does not, in this search.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 6.94 'apple' · *Hush*: As Batman swings through Gotham City in pursuit of her, his grapple is cut and he falls to the ground, fracturing his skull.
- 6.84 ' grappling' · *Batman '89*: Batman uses a grappling hook to attach Joker's leg to a crumbling gargoyle that eventually falls off the roof.
- 6.59 ' gun' · *Alien*: She dons a spacesuit, loads a grappling hook gun, and straps herself to a chair before discharging gases to drive the alien from its hiding place.
- 6.03 'apple' · *Earth One*: Fox, after seeing Batman uses the grapple gun he repaired for Bruce on the news, deduces that the young billionaire is the vigilante and begins building a bat-…
- 5.97 'ad' · *Inception*: Cobb accepts the offer and recruits a forger named Eames, a chemist named Yusuf, and an architecture student named Ariadne.
- 4.59 ' tangled' · *Gravity*: As she maneuvers it away from the ISS, the tangled parachute tethers snag, preventing the spacecraft from leaving.
- 3.97 ' line' · *Jaws*: Quint, estimating it is 25 feet (7.6 m) long and weighs 3 tonnes (3.0 long tons; 3.3 short tons), harpoons it with a line attached to a flotation barrel, but t…
- 3.59 ' gun' · *Earth One*: He attacks Weaver, but the grapple gun he developed malfunctions and Weaver escapes.

Its negative pole L12·8048+ (max 1.05):

- 1.05 'ickle' · *Taxi Driver*: In New York City, former Marine and Vietnam War veteran Travis Bickle takes a job as a night-shift taxi driver to cope with his chronic insomnia and loneliness…
- 0.77 ' cameo' · *Arkham Asylum (game)*: The shape-shifting Clayface appears in cameo, taking on the guise of other characters as he tries to trick the player into releasing him.
- 0.60 'ace' · *Batman Forever*: Two-Face hijacks the event and threatens to detonate a bomb unless Batman reveals his identity.
- 0.57 ' rock' · *Parasite*: As the family has financial struggles and low-income jobs, family friend Min-hyuk gives them a scholar's rock meant to promise wealth.
- 0.56 'y' · *The Social Network*: Marylin Delpy, a junior lawyer for Zuckerberg, tells Zuckerberg they will settle with Saverin because the sordid details of Facebook's founding and Zuckerberg'…

Agent 2's examples and near misses (activation, share of corpus max):

- example 3.59 (52%) ' hook': The thief fires a grappling hook across the alley and slides down the cable to the rooftop below.
- example 4.03 (58%) 'angle': Her parachute tethers tangle in the tree, leaving her dangling by the cords above the river.
- example 1.84 (27%) ' rope': The whaler's harpoon line goes taut and the rope drags the boat behind the wounded beast.
- example 14.31 (206%) 'ool': Before entering the labyrinth, the hero is handed a ball of thread to unspool behind him.
- example 0.48 (7%) ' rope': The climbers rope themselves together, and when one slips, the line snaps tight and holds him.
- near miss 0.01 (0%) '-': The actor forgets his line in the final scene and ad-libs a joke that saves the show.
- near miss 0.00 (0%) None: The boxer lands a vicious left hook, and the champion grapples with doubt in his corner.
- near miss 2.38 (34%) 'riad': Ariadne designs the dream architecture for the heist, folding the city streets over on themselves.
- near miss 0.20 (3%) ' the': The whaler stands in line at the harbour office, waiting to collect his wages for the season.
- near miss 0.11 (2%) ' news': The cable news anchor reads a thread of angry messages on air while the studio audience laughs.

Steered generations:

- α=6.94 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is carrying a backpack and seems to be searching for something. He finds a small clearing and stops. He looks around, then takes out a small o…
- α=6.94 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is a talented but struggling artist. She lives in a small town where everyone knows each other, and she's constantly trying to mak…
- α=6.94 · The story begins when → the narrator, a young man, is walking in the forest. He is followed by a stranger. The stranger is very tall, with a long beard, and dressed in a dark cloak. He is very pale, and his eyes are very
- α=13.88 · Plot summary: The film opens as → a young man, named Jack, is walking through the forest. He is carrying a large reel of fishing line. He is trying to find his way through the forest, but he is lost. He is trying to find his way back…
- α=13.88 · Plot: In the first act, → the story introduces the world of the game, the characters, and the initial conflict. The second act develops the plot, introduces obstacles, and builds tension. The third act resolves the conflict,…
- α=13.88 · The story begins when → the narrator, a young man, is introduced to the narrator's friend, who is a young man. The two men are in a small town, and the narrator is in love with the friend's sister. The friend is very jealou…

### 6. Bats

**Paper's unit L8·7130+** (layer 8): dev 9/10, test 5/6 (AUC 0.75); gate-open corpus max 14.188, present in 24 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Bat' ×24, ' bat' ×13, ' bats' ×6, 'Bat' ×3, '-S' ×2, 'ats' ×1.

**New unit L10·2879+** (layer 10): 2597 units passed the dev bar; this one scored best. Dev 9/10 (mean pos 3.133, neg 0.386); test 6/6 (AUC 0.944; mean pos 3.035, neg 0.488); p99 on the other memes' sentences 0.781 (max 3.688). Corpus max 7.5, p99 1.023; present in 0 of 32 Batman works, 1 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 0, only new 0 (), only paper 24 (1943 serial, A Death in the Family, Arkham Asylum (comic), Arkham Asylum (game), Arkham Knight, Batman & Robin, Batman '66, Batman '89). Batman works scoring above the median ordinary film: 13 of 32 for the new unit, 32 for the paper's.
- Top-50 peak tokens: ' moth' ×3, ' of' ×3, ' fly' ×2, ' Fly' ×2, ' flying' ×2, 'asz' ×2.
- Agent 1 (blind to the label, rates it mixed, high confidence) explains it as: *This unit fires on small flying insects and swarms: 'fly', 'mosquitoes', 'moth', 'flying', and 'of' in 'a swarm of [bats]'. It is partly lexical, since it also fires on 'Fly' when that is a Border Collie's name and on 'damselfish'. The opposite pole is negligible (max 1.3).*
- Judge: **adjacent**. It captures small flying insects and swarms, with 'a swarm of [bats]' as one case, but describes insects and flight rather than bats. Label the explanation supports: *Small flying insects and swarms*
- Agent 2 blind maximum: 17.38 against corpus max 7.50 (weakest of the top 8: 1.91; peak ' mosquitoes'): A swarm of mosquitoes and moths rises from the marsh, every fly and gnat flying in a buzzing cloud of insects around the lantern.
- Agent 2 examples firing strongly (≥ 50% of max): 4 of 5; near misses: 0 of 5.
- Steering: α=7.5: none; α=7.5: weak The antidote (alchemy can be used to heal the world); α=7.5: none; α=15.0: none; α=15.0: weak Franchise roll-call (Wikipedia style) (Scout Finch, her brother Jem, and t…; α=15.0: none
- **Verdict:** Adjacent. Flying swarms and small insects (a swarm of bats among them); the blind maximum beats the corpus maximum but the concept is swarming flight, not the animal. The paper's own bats unit is the better match for this label.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 7.50 ' fly' · *Daredevil*: Elsewhere, a heavily bandaged Bullseye awakens in a heavily guarded hospital room and kills a fly with a syringe needle.
- 5.75 ' mosquitoes' · *Jurassic Park*: At the park's visitor center, the group learns that the cloning was accomplished by extracting dinosaur DNA from prehistoric mosquitoes preserved in amber.
- 5.56 ' Fly' · *Babe*: There, he is taken in by Border Collie Fly, her irascible husband Rex and their puppies and befriends a duck named Ferdinand, who wakes people by crowing like…
- 5.00 ' moth' · *The Bat (1926)*: A circular light shines on the wall, with the shadow of a bat in its center, but after investigating, the group finds that the shape was caused by a miller mot…
- 4.06 ' of' · *Year One*: Batman uses a signal device to attract a swarm of bats from the Batcave, concealing his escape.
- 2.42 'self' · *Finding Nemo*: He meets the "Tank Gang", led by Gill, a scarred Moorish idol and consisting of Peach the Sea Star, Deb the striped damselfish, Gurgle the royal gramma, Bloat…
- 2.27 ' of' · *Batman Returns*: Oswald pushes the Ice Princess to her death with a swarm of bats, framing Batman.
- 1.91 'aps' · *Chinatown*: Frustrated, he repeatedly slaps Evelyn until she breaks down and reveals that Katherine is both her sister and daughter; the girl's father is Cross, who impreg…

Its negative pole L10·2879- (max 1.27):

- 1.27 ' feet' · *The Wizard of Oz*: Before she can seize her deceased sister's ruby slippers, Glinda magically transports them onto Dorothy's feet and tells her to keep them on.
- 0.62 ' that' · *Lost in Translation*: In the days that follow, Bob and Charlotte spend more time together, and their friendship strengthens.
- 0.46 ' Sachs' · *The Devil Wears Prada*: Aspiring journalist Andrea "Andy" Sachs has recently graduated from Northwestern University.
- 0.45 ' next' · *Forrest Gump*: After a night of sexual intercourse, Jenny leaves the next morning.
- 0.40 ' the' · *Lost in Translation*: On Bob's taxi ride to the airport, he sees Charlotte on a crowded street, stops the car, and walks to her.

Agent 2's examples and near misses (activation, share of corpus max):

- example 15.19 (202%) ' mosquitoes': Trapped in the swamp cabin, the hikers are driven mad by a swarm of mosquitoes that attacks every night.
- example 11.00 (147%) 'fly': A scientist accidentally merges with a housefly in his teleportation pod and slowly becomes a monstrous insect.
- example 0.82 (11%) ' sw': The caped vigilante summons a swarm of bats that pours out of the cave and swallows the police helicopters.
- example 8.19 (109%) ' insects': A giant moth hatches on a remote island and goes flying toward the capital, trailed by a cloud of smaller insects.
- example 8.25 (110%) 'ats': The farmer's crops are destroyed by a plague of locusts and gnats that darkens the sky for three days.
- near miss 0.08 (1%) ' the': Batman flies the Batwing over Gotham harbour and cuts the Joker's balloons loose from the parade.
- near miss 2.41 (32%) ' of': A swarm of reporters surrounds Bruce Wayne outside the courthouse, shouting questions about Harvey Dent.
- near miss 0.11 (1%) 'ar': Lucius Fox finds a bug in the Batcomputer's sonar software and patches it before the broadcast.
- near miss 0.00 (0%) None: The Penguin's army of emperor penguins marches through the sewers carrying rockets on their backs.
- near miss 0.02 (0%) ' kick': Robin lands a flying kick on the henchman and the two tumble down the museum stairs.

Steered generations:

- α=7.5 · Plot summary: The film opens as → a young girl, Lina, is playing with her brother, who is a bit of a bully. She is trying to get him to play with her, but he refuses. She then sees her mother, who is a nurse, and
- α=7.5 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elara, who is a skilled alchemist. She lives in a world where magic and science coexist, and she is determined to prove that alchemy can be u…
- α=7.5 · The story begins when → the young girl, who is the protagonist, is playing with her dog in the garden. She is very happy and carefree. But then she sees a butterfly, which is very beautiful. She wants to catch it, but the b…
- α=15.0 · Plot summary: The film opens as → a young man, named Jimmy, is in the process of mating with a female fruit fly. After mating, Jimmy is caught in a spider's web, where he is eaten by the spider. Jimmy's body is then shown as a m
- α=15.0 · Plot: In the first act, → the story introduces the setting of the play, which is the town of Maycomb, Alabama, during the Great Depression. The main characters are Scout Finch, her brother Jem, and their father Atticus Finch.…
- α=15.0 · The story begins when → the narrator is walking through the woods. He is very tired and decides to rest under a tree. He is then joined by a fly, which he tries to swat. The fly is persistent, and the narrator is frustrated…

### 7. Gotham City

**Paper's unit L6·10402+** (layer 6): dev 0/10, test 1/6 (AUC 0.583); gate-open corpus max 3.031, present in 30 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' City' ×44, ' Gotham' ×4, 'm' ×1, 'av' ×1.

**New unit L11·10296+** (layer 11): 6085 units passed the dev bar; this one scored best. Dev 9/10 (mean pos 1.251, neg 0.122); test 6/6 (AUC 1.0; mean pos 0.475, neg 0.086); p99 on the other memes' sentences 1.097 (max 2.141). Corpus max 7.25, p99 1.298; present in 0 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 0, only new 0 (), only paper 30 (1943 serial, A Death in the Family, Arkham Asylum (game), Arkham City, Arkham Knight, Batman & Robin, Batman '66, Batman '89). Batman works scoring above the median ordinary film: 24 of 32 for the new unit, 31 for the paper's.
- Top-50 peak tokens: ' York' ×10, 'oled' ×4, 'state' ×3, ' Crane' ×3, ' Columbia' ×2, ' Jacobs' ×2.
- Agent 1 (blind to the label, rates it conceptual, high confidence) explains it as: *This unit fires on the token that completes a New York place name or institution — upstate (New York), New York Public Library, Columbia, Roosevelt Island, the Gambino crime family, 'York' itself — and it fires on 'upstate' even when New York is not named in the sentence, so it is tracking New York-ness from context. The opposite pole is weak and lands on other American locales and their people (Punxsutawney, Miami, Philadelphia's southpaw and Goldmill), so the unit picks out New York as opposed to elsewhere in the United States.*
- Judge: **adjacent**. It captures New York place names and institutions, the real city Gotham is modelled on, but nothing of the fictional, crime-ridden city itself. Label the explanation supports: *New York place names and institutions*
- Agent 2 blind maximum: 6.53 against corpus max 7.25 (weakest of the top 8: 2.97; peak 'state'): Fleeing the crime family that runs the borough, the detective drives upstate from the city, leaving the public library and the island behind.
- Agent 2 examples firing strongly (≥ 50% of max): 2 of 5; near misses: 0 of 5.
- Steering: α=7.25: weak Franchise roll-call (Wikipedia style) (Alice (Mia Wasikowska)); α=7.25: none; α=7.25: none; α=14.5: weak Franchise roll-call (Wikipedia style) (Lila (Molly Ringwald)); α=14.5: none; α=14.5: none
- **Verdict:** Adjacent, and telling: the search for a name-free Gotham landed on New York place names (upstate, Columbia, the Gambino family), the city Gotham is modelled on. Not the fictional city.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 7.25 'state' · *Get Out*: Six months later, Black photographer Chris Washington travels to upstate New York for a weekend getaway to meet the family of his white girlfriend, Rose Armita…
- 7.00 'state' · *Goodfellas*: Realizing that the unsanctioned murder of a made man will invite retribution, Jimmy, Henry and Tommy bury the body in Upstate New York, stabbing and fatally sh…
- 4.78 ' Public' · *Ghostbusters*: After Columbia University parapsychology professors Peter Venkman, Ray Stantz, and Egon Spengler experience their first encounter with a ghost at the New York…
- 3.97 'state' · *Rear Window*: A skeptical Doyle investigates, finds nothing suspicious, and posits that Thorwald sent his wife on a vacation upstate.
- 3.69 'bie' · *Forrest Gump*: At the anti-war March on the Pentagon rally, Forrest meets Abbie Hoffman and briefly reunites with Jenny, who has been living a hippie life.
- 3.22 'ino' · *Goodfellas*: On June 11, 1970, Billy Batts, a made man in the Gambino crime family who was recently released from prison, incessantly insults Tommy at a nightclub owned by…
- 3.14 ' Columbia' · *The Social Network*: As TheFacebook grows in popularity, Zuckerberg expands the network to Yale, Columbia, and Stanford.
- 2.97 ' Island' · *Spider-Man*: That night, Norman captures Mary Jane and a Roosevelt Island Tramway car full of children.

Its negative pole L11·10296- (max 1.62):

- 1.62 'aw' · *Groundhog Day*: Alongside his new producer Rita Hanson and cameraman Larry, Phil travels to Punxsutawney for his annual coverage of the Groundhog Day festivities.
- 1.45 ' dealer' · *Moonlight*: In Liberty City, Miami at the height of the crack epidemic, Afro-Cuban drug dealer Juan finds Chiron, a withdrawn child who goes by the nickname "Little", hidi…
- 1.33 'way' · *Spider-Man*: That night, Norman captures Mary Jane and a Roosevelt Island Tramway car full of children.
- 1.24 'ie' · *Goodfellas*: He begins working for local caporegime Paulie Cicero and his associates Jimmy Conway, an Irish American truck hijacker and gangster, and Tommy DeVito, a fellow…
- 1.09 ' Ice' · *Batman Returns*: During Gotham's Christmas-tree lighting, Oswald and Catwoman kidnap Gotham's beauty queen, the Ice Princess, and lure Batman to a rooftop above the ceremony.

Agent 2's examples and near misses (activation, share of corpus max):

- example 10.94 (151%) 'state': A struggling actress leaves her apartment in the city and takes a train upstate to her mother's farmhouse.
- example 1.41 (19%) ' family': Two mobsters from the outer boroughs rise through the ranks of the city's most powerful crime family.
- example 0.64 (9%) ' subway': A ghost hunter sets up shop in the city's grand public library before a spirit chases him into the subway.
- example 7.88 (109%) 'state': A professor at the city's famous uptown university uncovers a conspiracy reaching from the island to the state capital upstate.
- example 2.97 (41%) 'state': A cab driver works the five boroughs by night and sends his savings to his family upstate.
- near miss 0.00 (0%) None: A weatherman is trapped reliving the same day in Punxsutawney, Pennsylvania, until he learns to love his neighbors.
- near miss 0.12 (2%) ' New': A detective in New Orleans chases a voodoo cult through the French Quarter during Mardi Gras.
- near miss 0.19 (3%) ' heavyweight': A southpaw boxer from Philadelphia gets one shot at the heavyweight title and trains on the museum steps.
- near miss 0.11 (2%) ' in': A Yorkshire shepherd enters his dog in the county trials and goes upstairs to tell his wife the news.
- near miss 0.02 (0%) ' the': A retired cop in Miami joins forces with a drug agent to break up a cartel operating out of the Keys.

Steered generations:

- α=7.25 · Plot summary: The film opens as → a young woman, named Alice (Mia Wasikowska), is reading a book titled "Alice's Adventures in Wonderland." She is in a room with a window, and she is reading the book aloud. She is joined by a
- α=7.25 · Plot: In the first act, → the story introduces the main characters and the setting. The protagonist, a young woman named Elara, lives in a small village surrounded by dense forests. She is curious and adventurous, often explo…
- α=7.25 · The story begins when → the narrator, a young man, is walking through the woods. He is in a hurry to get to a town called Salem. He is in a hurry because he has to meet a friend there. He is also in a hurry because he
- α=14.5 · Plot summary: The film opens as → a young woman, Lila (Molly Ringwald) is walking through a mall. She is in a hurry, but she stops to look at a window display. She is drawn to a display of a doll. She is about to
- α=14.5 · Plot: In the first act, → the story introduces the world of the play, the characters, and the central conflict. In this case, the first act introduces the world of the play, the characters, and the central conflict of the pla…
- α=14.5 · The story begins when → the narrator, a young man, is walking through the woods. He is thinking about the beauty of the forest and the animals that live there. He is also thinking about the people who live in the city and h…

### 8. Commissioner Gordon

**Paper's unit L11·12069-** (layer 11): dev 3/10, test 1/6 (AUC 0.347); gate-open corpus max 1.164, present in 30 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Gordon' ×14, ' Batman' ×12, ' Bruce' ×9, 'ina' ×4, ' Jason' ×1, '3' ×1.

**New unit L8·9586+** (layer 8): 5129 units passed the dev bar; this one scored best. Dev 9/10 (mean pos 1.115, neg 0.227); test 6/6 (AUC 0.833; mean pos 1.059, neg 0.321); p99 on the other memes' sentences 1.015 (max 2.484). Corpus max 3.062, p99 0.738; present in 0 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 0, only new 0 (), only paper 30 (1943 serial, A Death in the Family, Arkham Asylum (game), Arkham City, Arkham Knight, Batman & Robin, Batman '66, Batman '89). Batman works scoring above the median ordinary film: 17 of 32 for the new unit, 31 for the paper's.
- Top-50 peak tokens: ' CEO' ×6, 'D' ×4, ' chief' ×3, ' super' ×3, ' chef' ×3, ' director' ×3.
- Agent 1 (blind to the label, rates it conceptual, high confidence) explains it as: *This unit fires on the title of the person who heads an organisation, usually right before that person's name: 'CEO', 'police chief', 'head chef', 'supercargo', 'director'. The opposite pole fires on administrative and bureaucratic processes: 'registration tags', 'registers', 'signs ... up', 'enroll', 'prison labor', 'military service', 'taxed'.*
- Judge: **partial**. It captures the title of an organisation's head before a name, including 'police chief', which is Gordon's office, but nothing of the police ally or the signal. Label the explanation supports: *Title of an organisation's head before a name*
- Agent 2 blind maximum: 2.86 against corpus max 3.06 (weakest of the top 8: 1.47; peak ' chief'): The police chief meets the company's new CEO, the hospital director and the hotel's head chef to plan security for the mayor's fundraiser.
- Agent 2 examples firing strongly (≥ 50% of max): 2 of 5; near misses: 0 of 5.
- Steering: α=3.06: none; α=3.06: none; α=3.06: none; α=6.12: none; α=6.12: none; α=6.12: none
- **Verdict:** Partial. The title of an organisation's head before a name (police chief, CEO, director): Gordon's office, not Gordon. Weak unit (maximum 3.1).

Strongest corpus sentences of the new unit (gate-open, peak token):

- 3.06 ' CEO' · *Mrs. Doubtfire*: One day, the station's CEO, Jonathan Lundy, sees Daniel playing with props on the set of a recently cancelled children's show.
- 2.28 ' chief' · *Jaws*: After the coroner concludes it was a shark attack, Amity police chief Martin Brody plans to close the beaches.
- 2.22 ' chief' · *Fargo*: Brainerd police chief Marge Gunderson, who is seven months pregnant, begins investigating.
- 2.08 ' CEO' · *Singin' in the Rain*: Don arrives at the afterparty, where Monumental Pictures' CEO R.
- 1.91 ' super' · *Count of Monte Cristo*: When the Pharaon arrives in the port of Marseille, the ship's owner, Morrel, decides to promote Dantès to captain, provoking the distaste and jealousy of the s…
- 1.91 ' chef' · *The Shining*: Before leaving for the seasonal break, the Overlook's head chef, Dick Hallorann, informs Danny of a telepathic ability the two share, which Hallorann's grandmo…
- 1.67 ' CEO' · *Dark Knight Rises*: Bane kills Daggett to pursue his own goals while Bruce and Wayne Enterprises's new CEO, Miranda Tate, become lovers.
- 1.47 ' CEO' · *Iron Man*: Meanwhile, the Ten Rings gather the pieces of Stark's prototype suit and meet with Stane, who has been trafficking arms to the Ten Rings and has staged a coup…

Its negative pole L8·9586- (max 3.30):

- 3.30 ' registration' · *Fargo*: A state trooper stops them near Brainerd for not displaying temporary registration tags on the new Oldsmobile.
- 3.23 ' up' · *Babe*: When Esme leaves on a trip, Hoggett signs Babe up for sheepdog trials.
- 3.06 ' registers' · *Psycho*: Norman Bates, the proprietor, whose house overlooks the motel, registers Marion (who uses an alias) and invites her to dinner with him in the motel's office.
- 2.98 ' labor' · *The Shawshank Redemption*: In 1963, Norton begins exploiting prison labor for public works, profiting by undercutting skilled labor costs and receiving bribes.
- 2.83 ' enroll' · *Goodfellas*: Henry concludes that Jimmy plans to have him killed and decides to finally become an informant and enroll with his family in the witness protection program.

Agent 2's examples and near misses (activation, share of corpus max):

- example 0.35 (11%) ' commissioner': The police commissioner orders a city-wide manhunt after the clown escapes from the asylum.
- example 3.97 (130%) ' CEO': The company's CEO is found dead in his office, and the board names his ambitious young rival to replace him.
- example 1.38 (45%) ' chef': A talented line cook clashes with the restaurant's tyrannical head chef on the night of the critic's visit.
- example 2.50 (82%) ' chief': The asylum's director secretly experiments on his patients while the police chief looks the other way.
- example 1.33 (43%) ' super': The ship's captain and the supercargo argue over the route while the crew mutinies below deck.
- near miss 0.14 (5%) ' chief': Gordon's chief concern is the Joker, and he keeps his head down as the bullets fly past the armoured car.
- near miss 0.00 (0%) None: Bruce Wayne registers the new Tumbler, enrolls Alfred's nephew in military service and pays the city's taxes.
- near miss 0.38 (12%) ' upstairs': Alfred directs the delivery men to the east wing and heads upstairs to draw Bruce's bath.
- near miss 0.30 (10%) ' sergeant': A young police officer named Blake patrols the Narrows and reports to the sergeant at the precinct.
- near miss 0.00 (0%) None: Alfred cooks a chef's salad for Bruce and leaves it on the tray beside the Batcomputer.

Steered generations:

- α=3.06 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is carrying a backpack and seems to be in a hurry. He is followed by a man in a black coat. Jack is trying to avoid being caught. The man in
- α=3.06 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is struggling to find her place in the world. She is an aspiring artist, but her family expects her to pursue a more stable career…
- α=3.06 · The story begins when → the narrator, a young man, is walking through the forest. He is alone, and he feels a bit uneasy. He hears a sound, and then sees a figure in the distance. The figure is a woman, and she is dressed
- α=6.12 · Plot summary: The film opens as → a young girl, named Lina, is playing with her dog. She is told by her mother that the dog is going to be taken away. Lina is devastated. She then goes to the park, where she meets a boy named
- α=6.12 · Plot: In the first act, → the story introduces the protagonist, a young woman named Clara who is struggling to find her place in the world. She is an aspiring artist, but her family expects her to pursue a more stable career.…
- α=6.12 · The story begins when → the narrator, a young man, is walking through the forest. He is in search of a certain tree, which he has heard about. He finds the tree, but it is not what he expected. He is told by the tree that

### 9. Losing the protégé

**Paper's unit L11·7340-** (layer 11): dev 3/10, test 0/6 (AUC 0.486); gate-open corpus max 2.547, present in 31 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Batman' ×10, ' Wayne' ×8, ' Todd' ×7, ' Gordon' ×7, ' parents' ×6, ' Barbara' ×3.

**New unit L14·6104-** (layer 14): 2835 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 3.149, neg 0.477); test 6/6 (AUC 1.0; mean pos 4.457, neg 0.309); p99 on the other memes' sentences 2.844 (max 5.969). Corpus max 9.75, p99 3.61; present in 4 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 4, only new 0 (), only paper 27 (1943 serial, Arkham Asylum (comic), Arkham Asylum (game), Arkham City, Arkham Knight, Batman & Robin, Batman '66, Batman '89). Batman works scoring above the median ordinary film: 19 of 32 for the new unit, 32 for the paper's.
- Top-50 peak tokens: ' save' ×4, ' responsible' ×2, ' himself' ×2, ' failure' ×2, ' guilty' ×2, ' as' ×2.
- Agent 1 (blind to the label, rates it conceptual, high confidence) explains it as: *This unit fires on guilt and self-blame for failing to save or protect someone: 'feels partially responsible', 'failed to save [spring lambs]', 'blames himself', 'his greatest failure', 'unable to save', 'was his fault', 'feeling guilty'. It responds to the concept across many wordings (responsible, fault, guilty, failure, blames himself, failed to save). The opposite pole is weak (max 2.4) and diffuse, leaning carefree ('hakuna matata', a road trip).*
- Judge: **partial**. It captures guilt and self-blame for failing to save someone, the consequence of losing a protege, but no mentor or protege figures in it. Label the explanation supports: *Guilt for failing to save someone*
- Agent 2 blind maximum: 9.25 against corpus max 9.75 (weakest of the top 8: 6.09; peak ' save'): Haunted by the night he failed to save his parents, the orphaned heir blames himself, feeling guilty and responsible for a death that was never his fault.
- Agent 2 examples firing strongly (≥ 50% of max): 3 of 5; near misses: 0 of 5.
- Steering: α=9.75: none; α=9.75: weak The antidote (a plague strikes the village); α=9.75: none; α=19.5: none; α=19.5: weak The origin wound (haunted by the memory of the accident); α=19.5: none
- **Verdict:** Partial. Guilt and self-blame for failing to save someone: the emotional core of losing a protégé, without the mentor. Held-out 6 of 6; three of five examples fire, no near miss.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 9.75 ' responsible' · *Get Out*: While entranced, he reveals that his mother died in a hit-and-run when he was 11 years old; he feels partially responsible, as he delayed calling for help.
- 9.25 ' save' · *The Silence of the Lambs*: Lecter demands deeper revelations and offers a clue—"we covet what we see every day"—before Starling recounts a traumatic childhood incident: after her father'…
- 8.12 ' himself' · *A Death in the Family*: Batman blames himself for Jason's death and resolves to carry on alone, rejecting Alfred's suggestion to involve Dick Grayson, the first Robin.
- 7.25 ' himself' · *Batman v Superman*: Superman blames himself for not detecting it and self-imposes exile.
- 7.25 ' failure' · *Knightfall*: The rescue of Mayor Krol from the teaming of the Joker and Scarecrow pushes Batman to his mental and physical limits: a dose of Scarecrow's fear gas makes him…
- 6.59 ' save' · *Mad Max: Fury Road*: After seeing a vision of the child he was unable to save, Max catches up with the group and convinces them to return to the Citadel since they do not know what…
- 6.28 ' fault' · *The Lion King*: As Simba mourns his father, Scar deceives him into believing that Mufasa's death was his fault and tells him to leave the kingdom and never return.
- 6.09 ' guilty' · *The Lion King*: Still feeling guilty over Mufasa's death, Simba refuses and leaves angrily.

Its negative pole L14·6104+ (max 2.42):

- 2.42 ' naked' · *Year One*: In revenge, Jim tracks Flass down, beats him, and leaves him naked and handcuffed in the snow.
- 2.03 ' trip' · *Little Miss Sunshine*: Due to financial constraints, they go on an 800-mile (1,300 km) road trip in their yellow Volkswagen van.
- 1.68 'ata' · *The Lion King*: Simba grows up with his two new friends in their oasis, living a carefree life under their motto "hakuna matata" ("no worries" in Swahili).
- 1.64 ' Hood' · *Shrek*: The next day, Robin Hood and his band of Merry Men harass the three, but Fiona easily defeats them in physical combat.
- 1.63 ' "' · *Knives Out*: As Ransom is taken into custody and the rest of the family is gathered outside in defeat, Marta watches from the balcony of what is now her mansion, sipping fr…

Agent 2's examples and near misses (activation, share of corpus max):

- example 7.44 (76%) ' himself': The lifeguard blames himself for the boy's drowning and quits the beach, convinced the death was his fault.
- example 8.88 (91%) ' guilt': Unable to save his partner in the warehouse fire, the firefighter is consumed by guilt and turns to drink.
- example 1.82 (19%) ' responsible': The sheepdog feels responsible for the lambs lost to the wolf and refuses to leave the barn.
- example 3.44 (35%) ' failure': The caped vigilante regards the district attorney's fall as his greatest failure and withdraws from the city for years.
- example 7.59 (78%) ',': Feeling guilty that she could not protect her younger brother from their father, the nurse devotes herself to abused children.
- near miss 0.15 (2%) ' the': Lucius Fox is responsible for the applied sciences division and signs off on every new gadget.
- near miss 0.10 (1%) ' the': The jury finds the Joker guilty on all counts and the judge sentences him to Arkham.
- near miss 0.16 (2%) ' saves': An electrical fault in the Batmobile's engine stalls the chase and Batman saves time by taking the Batpod.
- near miss 1.13 (12%) ' for': The Joker blames Batman for the chaos in Gotham and tells the television cameras it is all his fault.
- near miss 0.37 (4%) ' son': A power failure at Wayne Tower wipes the servers before Lucius can save the sonar data.

Steered generations:

- α=9.75 · Plot summary: The film opens as → a young girl, Lina, is playing with her brother, who is in a wheelchair. She is told by her mother that they are going to the park. They arrive, and Lina is told by her mother to stay with the
- α=9.75 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elara, who is a skilled but inexperienced healer. She lives in a small village where the people rely on her for their medical needs. One day,…
- α=9.75 · The story begins when → the narrator, a young man, is told by his father that he must take the train to the city. The father gives him a bag of money and a letter of introduction. The narrator is told to go to the city and…
- α=19.5 · Plot summary: The film opens as → a young girl, Lila, is playing with her dog. She is told by her mother that the dog is not going to make it. She is then told by her mother that the dog is not going to make it. She is
- α=19.5 · Plot: In the first act, → the protagonist, a young woman named Emily, is in a car accident. She is the sole survivor, but she is not sure if she is responsible for the accident. In the second act, she is haunted by the memory…
- α=19.5 · The story begins when → the protagonist, a young man named Nick, is in a bar. He is sitting next to a man named George. George is drunk and starts to argue with Nick. Nick is trying to convince George that he is not going t…

### 10. The joke

**Paper's unit L2·6468+** (layer 2): dev 3/10, test 1/6 (AUC 0.5); gate-open corpus max 10.312, present in 23 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Joker' ×39, ' joke' ×5, 'oker' ×4, ' jokes' ×2.

**New unit L30·2954+** (layer 30): 4094 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 8.536, neg 0.02); test 6/6 (AUC 0.917; mean pos 6.257, neg 0.647); p99 on the other memes' sentences 2.773 (max 10.5). Corpus max 10.125, p99 1.969; present in 4 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 4, only new 0 (), only paper 19 (A Death in the Family, Arkham Asylum (comic), Arkham Asylum (game), Arkham City, Arkham Knight, Batman '66, Batman Ninja, Dark Knight Returns). Batman works scoring above the median ordinary film: 20 of 32 for the new unit, 23 for the paper's.
- Top-50 peak tokens: ' Joker' ×6, ' and' ×5, ' laughing' ×4, ' laughs' ×2, ' a' ×2, ' by' ×2.
- Agent 1 (blind to the label, rates it mixed, medium confidence) explains it as: *This unit is the weak side (max 10) of a smile/laughter channel: it fires on realised laughter and on the Joker, peaking on 'laughs', 'laughing', 'Joker', an 'audience', and tokens in Arthur Fleck scenes. The far stronger opposite pole (max 79.5) fires on the 'Sm' onset of smile/Smiling/Smylex and on the token right before a smile or laugh is named ('driving off with a [smile]', 'watches and [laughs]', 'rictus grin'), so the channel distinguishes an anticipated or forming smile from the completed laugh word and the Joker.*
- Judge: **partial**. It captures laughter, smiling and the Joker, which are the teller and the effect of the joke, but never the joke itself as an idea or punchline. Label the explanation supports: *Laughter, smiles and the Joker*
- Agent 2 blind maximum: 8.56 against corpus max 10.12 (weakest of the top 8: 6.97; peak ' laughs'): On stage at the comedy club the failed clown laughs and keeps laughing, and the audience laughs with him until the whole room is laughing.
- Agent 2 examples firing strongly (≥ 50% of max): 3 of 5; near misses: 3 of 5.
- Steering: α=10.12: none; α=10.12: none; α=10.12: none; α=20.25: none; α=20.25: none; α=20.25: none
- **Verdict:** Partial and lexical-leaning. The weak side of a smile-and-laughter channel that also fires on the Joker; three of five near misses fire, so the explanation does not win.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 10.12 ' laughs' · *The Exorcist*: He assures her that she will not and re-enters the room, finding Merrin dead from a heart attack while Regan watches and laughs.
- 9.44 ' laughs' · *Joker (2019)*: Sometime later at Arkham, Arthur laughs to himself about a joke during a session with his new therapist.
- 8.62 "'s" · *Joker (2019)*: Later, while performing at a children's hospital, Arthur's gun accidentally falls out of his pocket, resulting in him getting fired.
- 7.38 '3' · *The Grand Budapest Hotel*: In 1932, Zero flees his war-ravaged country to work as a lobby boy for Monsieur Gustave H., the philandering concierge at the Grand Budapest Hotel.
- 7.25 ' Joker' · *Under the Red Hood*: Though having forgiven Batman for not saving him, Jason is upset and angry that Joker is still alive after killing him.
- 7.19 ' laughing' · *Joker (2019)*: He suffers from a neurological disorder that causes him to have random, uncontrollable laughing fits, for which he takes medication that he obtains through the…
- 7.09 ' audience' · *Singin' in the Rain*: Lina and Kathy both flee in embarrassment, but Don has the audience stop Kathy and proudly announces that she is "the real star of the picture" ("You Are My Lu…
- 6.97 ' he' · *Batman '89*: Joker eventually pulls Batman and Vicki over the cathedral's roof, leaving them hanging while he calls for a helicopter piloted by his goons, who throw down a…

Its negative pole L30·2954- (max 79.50):

- 79.50 ' sm' · *Joker (2019)*: Arthur stands atop the police car, dances to the cheers of the crowd, and smears the blood on his face into a smile.
- 56.75 'Sm' · *Batman '89*: The Joker begins terrorizing Gotham by lacing various hygiene products with "Smylex" – a deadly chemical that causes victims to die laughing.
- 41.00 ' Sm' · *Notting Hill*: Smiling, she answers "Indefinitely".
- 29.50 ' ever' · *Batman '89*: Joker asks Wayne, "Have you ever danced with the devil in the pale moonlight?", which Wayne recognizes as the phrase used by the mugger who murdered his parent…
- 29.25 ' a' · *Taxi Driver*: Travis drops her off at her home but declines to take her money, driving off with a smile.

Agent 2's examples and near misses (activation, share of corpus max):

- example 12.19 (120%) ' laughs': The clown laughs maniacally as the hostages watch, his laughing echoing through the abandoned theatre.
- example 2.38 (24%) ' audience': A struggling comedian bombs at the open mic, laughing nervously while the audience sits in silence.
- example 10.31 (102%) ' laughs': The talk show host introduces the clown, and the studio audience laughs as he dances down the stairs.
- example 2.14 (21%) ' laughs': Afflicted with a condition that makes him laugh at the wrong moments, the party clown laughs on the bus and frightens a mother.
- example 17.88 (176%) ' laughs': The green-haired villain laughs hysterically at his own joke while the crowd of rioters in clown masks laughs with him.
- near miss 9.69 (96%) ' smile': Batman drives off with a smile as Alfred watches, smiling, from the manor steps.
- near miss 0.00 (0%) None: The Smylex gas leaves its victims with a rictus grin frozen on their faces.
- near miss 5.94 (59%) ' it': Bruce Wayne tells a joke at the gala but nobody at the table finds it funny.
- near miss 0.00 (0%) None: The Riddler deals a hand of poker and the joker card turns up wild in Harvey's hand.
- near miss 5.53 (55%) ' and': Selina Kyle smiles slyly and slips the pearls into her pocket before the guards notice.

Steered generations:

- α=10.12 · Plot summary: The film opens as → a young girl, Lina, is playing with her toys. She is told by her mother that they are moving to a new house. Lina is upset, but her mother reassures her that the new house will be better. L
- α=10.12 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is struggling to find her place in the world. She is an aspiring artist, but her family expects her to pursue a more stable career…
- α=10.12 · The story begins when → the narrator, a young man, is caught in a rainstorm. He is in a small town, and he is trying to find shelter. He goes into a small hotel, but the hotel is closed. He is very tired, so
- α=20.25 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is carrying a backpack and seems to be in a hurry. He encounters a woman, named Sarah, who is also walking through the forest. They talk brief…
- α=20.25 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elara, who lives in a small village. She is curious and adventurous, often exploring the nearby forest. One day, she discovers a hidden cave…
- α=20.25 · The story begins when → the narrator, a young man, is walking through the forest. He is in search of a certain tree, which he has heard is the source of the magical powers of the forest. He finds the tree, but it is not wha…

### 11. The rogues' gallery

**Paper's unit L10·5942+** (layer 10): dev 5/10, test 4/6 (AUC 0.778); gate-open corpus max 6.469, present in 31 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Ivy' ×12, 'crow' ×11, ' Penguin' ×11, ' Harley' ×7, 'ane' ×3, 'dd' ×1.

**New unit L34·5108+** (layer 34): 8008 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 12.869, neg 0.595); test 6/6 (AUC 1.0; mean pos 9.021, neg 0.426); p99 on the other memes' sentences 14.0 (max 20.5). Corpus max 55.25, p99 15.375; present in 4 of 32 Batman works, 1 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 4, only new 0 (), only paper 27 (1943 serial, A Death in the Family, Arkham Asylum (comic), Arkham Asylum (game), Arkham Knight, Batman & Robin, Batman '89, Batman Begins). Batman works scoring above the median ordinary film: 31 of 32 for the new unit, 32 for the paper's.
- Top-50 peak tokens: ' the' ×14, ' W' ×6, ' sh' ×4, ' superv' ×2, 'N' ×2, ' Sc' ×2.
- Agent 1 (blind to the label, rates it conceptual, medium confidence) explains it as: *This late-layer unit fires at the position where a villain's name or epithet is about to be produced: on 'the' before 'Wicked Witch of the East' and before 'Joker, the Penguin, the Riddler', on 'W' beginning 'Wicked Witch', and on 'superv' beginning 'supervillain'; it anticipates a villain designation as the next token, as opposed to positions completing an ordinary proper name such as a voice actor's surname.*
- Judge: **partial**. It captures the anticipation of a villain's name or epithet, including the roll-call 'the Joker, the Penguin, the Riddler', but describes naming a villain rather than the ensemble of recurring villains. Label the explanation supports: *Anticipating a villain's name or epithet*
- Agent 2 blind maximum: 10.31 against corpus max 55.25 (weakest of the top 8: 35.00; peak ' the'): Every supervillain in the city, the clown, the umbrella-wielding gangster, the puzzle-obsessed trickster and the wicked witch, unites against the caped vigilante.
- Agent 2 examples firing strongly (≥ 50% of max): 0 of 5; near misses: 0 of 5.
- Steering: α=55.25: none; α=55.25: none; α=55.25: none; α=110.5: none; α=110.5: none; α=110.5: none
- **Verdict:** Partial. A layer-34 unit that fires just before a villain's name or epithet is produced (the Joker, the Penguin, the Riddler); it is a predictor of villain naming, not a concept of the ensemble, and Agent 2 could not drive it (0 of 5).

Strongest corpus sentences of the new unit (gate-open, peak token):

- 55.25 ' the' · *The Wizard of Oz*: Dorothy is greeted by a good witch named Glinda, who comes down in a bubble and informs her that the house landed on the Wicked Witch of the East, killing her.
- 50.75 ' W' · *The Wizard of Oz*: The Munchkins celebrate the witch's death, until her sister, the Wicked Witch of the West, shows up.
- 47.50 ' W' · *The Wizard of Oz*: During their quest, Dorothy and Toto are captured by flying monkeys and taken to the Wicked Witch's castle.
- 43.00 ' W' · *The Wizard of Oz*: The Wizard appears as a giant ghostly head and tells them he will grant their wishes if they bring him the Wicked Witch's broom.
- 40.25 ' superv' · *Knightfall*: The two characters were quickly added to the cast in the monthly Batman titles, with Azrael being a superhero-in-training who fights alongside Batman, while Ba…
- 39.75 ' W' · *The Wizard of Oz*: The Winkies gratefully give Dorothy the Witch's broom, and the group returns to the Wizard, but he tells them to come back the next day.
- 37.25 ' the' · *Batman '66*: Batman and Robin head back to Commissioner Gordon's office, where they deduce that the tip was a set-up by the United Underworld, a gathering of four of the mo…
- 35.00 ' W' · *The Wizard of Oz*: They reach the Emerald City, despite the efforts of the Wicked Witch.

Its negative pole L34·5108- (max 50.75):

- 50.75 'as' · *Puss in Boots*: The marriage is celebrated with a grand feast, and the miller's son, now truly the "Marquis of Carabas", lives happily ever after with the princess, thanks to…
- 50.25 ' character' · *Arkham City*: Returning characters from Arkham Asylum include Batman (Kevin Conroy), the Joker (Mark Hamill)—in what Hamill stated would be his final time voicing the charac…
- 49.50 'um' · *Arkham City*: Black Mask (also voiced by North), Killer Croc (Steve Blum), and Freeze's wife, Nora Fries, make cameo appearances in the game.
- 48.00 'erman' · *Arkham City*: Wasserman), the League of Assassins's leader, Ra's al Ghul (Dee Bradley Baker), the mind-controlling Mad Hatter (Peter MacNicol), and the assassin Deadshot (Ch…
- 47.25 '.C' · *The Exorcist*: In Georgetown, Washington, D.C., actress Chris MacNeil is starring in a film directed by her friend Burke Dennings.

Agent 2's examples and near misses (activation, share of corpus max):

- example 10.25 (19%) ' the': The farm girl's house lands on the wicked witch, and her sister vows revenge on the newcomer.
- example 17.75 (32%) ' the': Freed from the asylum, the clown, the cat burglar and the scarecrow-masked doctor join forces against the city's protector.
- example 5.47 (10%) ' lord': The young wizard learns that the dark lord who murdered his parents has returned to power.
- example 6.72 (12%) 'ill': A supervillain with a freeze ray holds the city hostage until the masked hero intervenes.
- example 5.69 (10%) ' the': The rogues' gallery assembles: the hat-wearing hypnotist, the crocodile man and the plant-controlling seductress plot the hero's downfall.
- near miss 0.35 (1%) ',': Mark Hamill, Kevin Conroy and Efrem Zimbalist Jr. recorded their lines in a studio in Burbank.
- near miss 20.12 (36%) ' super': The superhero and his superpowered sidekick visit the supermarket to buy supplies for the orphanage.
- near miss 7.28 (13%) ' the': The wicked sense of humour of the young heroine wins over the Winkie farmers of the West.
- near miss 0.00 (0%) None: The jokers at the poker table, the penguins at the zoo and the riddles in the crossword fill the detective's quiet Sunday.
- near miss 2.08 (4%) 'ets': Glinda the Good Witch of the North greets Dorothy with a kiss and points her toward the Emerald City.

Steered generations:

- α=55.25 · Plot summary: The film opens as → a young girl, Lina, is playing with her toys. She is told by her mother that they are moving to a new house. Lina is upset, but her mother reassures her that the new house will be better. L
- α=55.25 · Plot: In the first act, → the story introduces the protagonist, a young woman named Clara who is struggling to find her place in the world. She is an aspiring writer, but her work is not being recognized. She lives in a small…
- α=55.25 · The story begins when → the young girl, who is the protagonist, is playing in the woods. She finds a small, red box. She opens it and finds a letter inside. The letter is from a girl named Eliza. Eliza says that she is
- α=110.5 · Plot summary: The film opens as → a young girl, Lina, is playing with her toys. She is told by her mother that they are moving to a new house. Lina is upset, but her mother reassures her that it's for the best. Lina
- α=110.5 · Plot: In the first act, → the story introduces the protagonist, a young woman named Clara who is struggling to find her place in the world. She is an aspiring writer, but her work is not being recognized. She lives in a small…
- α=110.5 · The story begins when → the protagonist, a young man named Jack, is walking through the forest. He is lost and has no idea where he is. He sees a glowing light in the distance and follows it. As he approaches, he finds hims…

### 12. Superhero powers (peaks on Superman)

**Paper's unit L10·7189+** (layer 10): dev 8/10, test 6/6 (AUC 0.861); gate-open corpus max 7.031, present in 31 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: 'rypton' ×11, ' Bat' ×10, ' Clark' ×5, ' Superman' ×3, 'ats' ×3, 'El' ×2.

**New unit L26·9345+** (layer 26): 2925 units passed the dev bar; this one scored best. Dev 9/10 (mean pos 3.386, neg 0.115); test 4/6 (AUC 0.708; mean pos 1.755, neg 0.814); p99 on the other memes' sentences 1.859 (max 4.844). Corpus max 8.25, p99 1.313; present in 0 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 0, only new 0 (), only paper 31 (1943 serial, A Death in the Family, Arkham Asylum (comic), Arkham Asylum (game), Arkham City, Arkham Knight, Batman & Robin, Batman '66). Batman works scoring above the median ordinary film: 18 of 32 for the new unit, 32 for the paper's.
- Top-50 peak tokens: ' a' ×7, ' to' ×3, ' ' ×3, ' the' ×2, ' his' ×2, ' and' ×2.
- Agent 1 (blind to the label, rates it conceptual, medium confidence) explains it as: *This unit fires in contexts of weight, heaviness and the strength needed to lift or lighten a load (an infant lifting a truck, weighing 3 tonnes, lightening a ship or house so it can fly, a 'guess the weight' contest, a drug that gives strength, clinging to walls), typically peaking on the function word just after the weight/strength word, as opposed to things falling or being pulled by gravity.*
- Judge: **partial**. It captures weight and the strength to lift it (an infant lifting a truck, clinging to walls), one superpower, but is framed around heaviness and loads rather than powers. Label the explanation supports: *Weight and the strength to lift it*
- Agent 2 blind maximum: 10.75 against corpus max 8.25 (weakest of the top 8: 3.22; peak ' three'): The infant lifts the three-tonne truck, whose enormous weight of three tonnes he bears with ease, and lightens the ship so it can fly.
- Agent 2 examples firing strongly (≥ 50% of max): 3 of 5; near misses: 2 of 5.
- Steering: α=8.25: none; α=8.25: none; α=8.25: none; α=16.5: none; α=16.5: none; α=16.5: none
- **Verdict:** Partial. Weight and the strength to lift it, one superpower; held-out only 4 of 6 and two near misses fire. The paper's Superman-name unit passes the pairs better (6 of 6), because the pairs describe Superman.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 8.25 '3' · *Jaws*: Quint, estimating it is 25 feet (7.6 m) long and weighs 3 tonnes (3.0 long tons; 3.3 short tons), harpoons it with a line attached to a flotation barrel, but t…
- 7.00 ' lifts' · *Superman*: He is found by Jonathan and Martha Kent, who are astonished when the infant lifts their truck.
- 6.59 ' the' · *Interstellar*: In the process, Cooper and TARS detach in their craft to lighten the Endurance so that Brand and CASE may reach Edmunds' planet.
- 5.38 ' to' · *Up*: Carl lightens his house by emptying it completely, enabling it to fly again.
- 5.28 ' an' · *Babe*: Following his usage in a "guess the weight" contest at an agricultural show, orphaned piglet Babe is brought home to the farm of the contest winner, Arthur Hog…
- 4.34 ' his' · *Arkham Asylum (game)*: There, Batman learns that Joker returned to the asylum to gain access to Young, who has been developing Titan—a more powerful version of the Venom drug that gi…
- 3.22 ' cling' · *Spider-Man*: The next day, Peter develops spider-like abilities, including enhanced strength, senses, agility, and speed; organic webbing in his wrists; and the ability to…
- 3.22 ' the' · *The Martian*: Watney quickly improvises by piercing his pressure suit, and propels himself with the escaping air, reaches Lewis, and manages to hold on.

Its negative pole L26·9345- (max 3.67):

- 3.67 ' gravity' · *The Martian*: Astrodynamicist Rich Purnell devises an alternative plan: send the Taiyang Shen launcher to resupply the Hermes, which will then use Earth's gravity to "slings…
- 2.19 ' of' · *Finding Nemo*: Then, they can roll their bags out of the window and into the harbour.
- 1.97 ' with' · *Inception*: However, Yusuf performs his kick too soon by driving off a bridge, forcing Arthur and Eames to improvise a new set of kicks synchronized with them hitting the…
- 1.64 ' to' · *Interstellar*: With insufficient fuel, Cooper and Brand resort to a slingshot around Gargantua, which costs them 51 years due to time dilation.
- 1.48 ' "' · *Little Miss Sunshine*: The hitherto-unseen dance routine Edwin had taught Olive is revealed to be a striptease performed to the Rocasound remix of Rick James' "Super Freak".

Agent 2's examples and near misses (activation, share of corpus max):

- example 10.62 (129%) ' weighing': A serum gives the boy such strength that he hoists a bus weighing twelve tonnes over his head.
- example 10.56 (128%) ' the': The carnival barker runs a guess-the-weight contest, and the strongman lifts the ox with one arm to prove it.
- example 3.61 (44%) ' loose': To make the house fly, the old man must lighten it, so he cuts loose every heavy piece of furniture.
- example 15.75 (191%) ' a': The drug makes her strong enough to carry the boulder, which weighs more than a car, up the hill.
- example 2.14 (26%) 's': Clinging to the wall by sheer strength, the thief hauls the heavy safe upward on his back.
- near miss 9.38 (114%) ' carries': The lift carries Bruce Wayne to the top floor of Wayne Tower while heavy rain falls on Gotham.
- near miss 5.31 (64%) ' pulls': The piano falls from the window and gravity pulls it down onto the street below.
- near miss 1.29 (16%) ' pulled': Lightning strikes the ship and the mast drops into the sea, pulled under by the current.
- near miss 0.00 (0%) None: She waits by the window all night, a heavy silence between them as the snow drifts down.
- near miss 0.02 (0%) ' signal': The signal's strength fades as the plane descends, falling through clouds toward the runway.

Steered generations:

- α=8.25 · Plot summary: The film opens as → a young girl, Lina, is playing with her toys. She is told by her mother that they are moving to a new house. Lina is upset, but her mother reassures her that it's for the best. Lina
- α=8.25 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is struggling to find her place in the world. She is an aspiring artist, but her family expects her to pursue a more stable career…
- α=8.25 · The story begins when → the narrator, a young man, is walking through the forest. He is alone, and he feels a bit uneasy. He hears a sound, like a footstep, and turns around. But there is no one there. He continues walking
- α=16.5 · Plot summary: The film opens as → a young girl, Lina, is playing with her toys. She is told by her mother that they are moving to a new house. Lina is upset, but her mother comforts her by saying that they will be moving to a new
- α=16.5 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is struggling to find her place in the world. She is an aspiring artist, but her family expects her to pursue a more stable career…
- α=16.5 · The story begins when → the narrator, a young man, is caught in a rainstorm in the mountains. He takes shelter in a cave, where he discovers a skeleton. The skeleton is wearing a ring, which the narrator takes. He later fin…

### 13. Franchise roll-call (Wikipedia style)

**Paper's unit L5·3626+** (layer 5): dev 0/10, test 0/6 (AUC 0.5); gate-open corpus max 13.0, present in 23 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: 'ham' ×50.

**New unit L26·11032-** (layer 26): 13369 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 18.231, neg 0.656); test 6/6 (AUC 1.0; mean pos 11.333, neg 2.535); p99 on the other memes' sentences 3.594 (max 7.656). Corpus max 22.75, p99 6.219; present in 11 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 9, only new 2 (1943 serial, Batman '66), only paper 14 (A Death in the Family, Batman & Robin, Batman Begins, Batman Forever, Dark Knight Returns, Earth One, Gotham by Gaslight, Hush). Batman works scoring above the median ordinary film: 30 of 32 for the new unit, 26 for the paper's.
- Top-50 peak tokens: 'c' ×4, ' in' ×3, 'ze' ×2, ' the' ×2, '1' ×2, ' Lego' ×2.
- Agent 1 (blind to the label, rates it conceptual, low confidence) explains it as: *This unit fires inside sentences about the Batman franchise, peaking on fragments of Batman-specific lore names and production references (Dr. Daka, Batcave, Commodore Schmidlapp, Jezebel Jet, 'series of film and television', 'in the monthly Batman titles', Lego), as opposed to the voice-actor names and generic superhero/supervillain role words that appear in the same cast-list sentences.*
- Judge: **direct**. It describes cast-list sentences about the Batman franchise peaking on lore names and production references, which is the Wikipedia-style roll-call register itself. Label the explanation supports: *Batman franchise lore and production references*
- Agent 2 blind maximum: 13.19 against corpus max 22.75 (weakest of the top 8: 12.94; peak ' television'): In the monthly comic titles and a long-running series of film and television adaptations, the caped vigilante operates from a cave beneath his manor, aided by his butler.
- Agent 2 examples firing strongly (≥ 50% of max): 0 of 5; near misses: 1 of 5.
- Steering: α=22.75: none; α=22.75: none; α=22.75: none; α=45.5: none; α=45.5: none; α=45.5: none
- **Verdict:** Direct by the judge, but a Batman-lore unit (Batcave, Dr Daka, monthly Batman titles) rather than the register; Agent 2's examples do not fire (0 of 5). Low-confidence explanation.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 22.75 ' D' · *1943 serial*: Once again, Dr. Daka's sabotage efforts are stopped by Batman and Robin.
- 21.75 ' film' · *Arkham Asylum (game)*: Three voice actors who worked on the DC Animated Universe series of film and television reprised their roles for the game.
- 18.88 'c' · *Batman R.I.P.*: As he passes out, Hurt and minions of his diabolical club, Black Glove, infiltrate the Batcave.
- 17.75 'l' · *Batman '66*: Commodore Schmidlapp accidentally breaks the powdered Council members' vials and sneezes on them, scattering the dust.
- 15.94 ' in' · *Arkham City*: Returning characters from Arkham Asylum include Batman (Kevin Conroy), the Joker (Mark Hamill)—in what Hamill stated would be his final time voicing the charac…
- 15.62 'c' · *Return of the Joker*: Subduing the Jokerz with Ace's help, Terry confronts Tim, who subdues Terry by disabling his Batsuit and then transforms into the Joker as Bruce watches in hor…
- 14.75 'ze' · *Batman R.I.P.*: It is revealed that Jezebel Jet has been a part of Black Glove all along.
- 12.94 ' in' · *Knightfall*: The two characters were quickly added to the cast in the monthly Batman titles, with Azrael being a superhero-in-training who fights alongside Batman, while Ba…

Its negative pole L26·11032+ (max 6.06):

- 6.06 ' final' · *Arkham City*: Returning characters from Arkham Asylum include Batman (Kevin Conroy), the Joker (Mark Hamill)—in what Hamill stated would be his final time voicing the charac…
- 5.69 ' superv' · *Knightfall*: The two characters were quickly added to the cast in the monthly Batman titles, with Azrael being a superhero-in-training who fights alongside Batman, while Ba…
- 5.66 ' Spider' · *Spider-Man*: Upon graduating, Peter begins using his abilities to fight crime, donning a costume and adopting the alias Spider-Man.
- 4.91 ' De' · *Arkham Knight*: Arkham Knight also features appearances and voice cameos by various characters drawn from the history of Batman comics and the previous Arkham games, including…
- 4.66 ' superv' · *Arkham Asylum (game)*: Arkham Asylum is set in the fictional Arkham Asylum, a facility on Arkham Island off the coast of Gotham City that houses criminally insane supervillains.

Agent 2's examples and near misses (activation, share of corpus max):

- example 0.09 (0%) ' where': The caped vigilante returns to his cave beneath the manor, where the butler has repaired the armoured car.
- example 1.71 (8%) 'y': The brick-toy animated film follows the caped vigilante as he teams up with his young ward against the clown.
- example 0.89 (4%) "'s": In the serial's final chapter, the vigilante storms the spy's hidden laboratory beneath the amusement arcade.
- example 2.33 (10%) ' titles': The monthly comic titles introduce a new love interest for the billionaire, a model with a mysterious past.
- example 6.53 (29%) ' television': Across a series of film and television adaptations, the vigilante's cave headquarters grows ever more elaborate.
- near miss 2.36 (10%) ' to': Kevin Conroy, Mark Hamill and Tara Strong lend their voices to the hero, the villain and the sidekick.
- near miss 1.15 (5%) ' side': The superhero fights the supervillain atop the skyscraper while the sidekick rescues the hostages below.
- near miss 1.85 (8%) ' where': Peter Parker swings through Manhattan to the Daily Bugle, where J. Jonah Jameson demands pictures of Spider-Man.
- near miss 3.20 (14%) ' in': The actor who voices the villain also narrates a nature documentary series about bats in Mexican caves.
- near miss 17.62 (78%) ' television': Clark Kent returns to the Fortress of Solitude after a series of film and television appearances made him a household name.

Steered generations:

- α=22.75 · Plot summary: The film opens as → a young girl, named Lina, is playing with her toys. She is told by her mother that they are moving to a new house. Lina is upset, but her mother comforts her by saying that the new house is bigger and
- α=22.75 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is struggling to find her place in the world. She is living in a small town, working a mundane job, and feeling unfulfilled. One d…
- α=22.75 · The story begins when → the young girl, who is the protagonist, is playing in the woods. She finds a small, red box. She opens it and finds a letter inside. The letter is from a girl named Eliza who is writing to her friend.
- α=45.5 · Plot summary: The film opens as → a young girl named Lila is playing with her toys. She is told by her mother that she is going to be a big sister. She is excited, but also a bit nervous. She is told that her new sibling will be a
- α=45.5 · Plot: In the first act, → the story introduces the protagonist, a young woman named Clara who lives in a small coastal town. She is an aspiring writer, but her work is not well received by the townspeople. One day, she discov…
- α=45.5 · The story begins when → the young girl, who is the protagonist, is playing in the woods. She finds a small, red box. She opens it and finds a letter inside. The letter is from a girl named Elise who is in the same grade as

### 14. The butler who raised him

**Paper's unit L6·4705-** (layer 6): dev 1/10, test 1/6 (AUC 0.583); gate-open corpus max 6.688, present in 19 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: 'worth' ×21, 'oby' ×4, 'o' ×2, 'iti' ×2, 'ut' ×2, 'ista' ×1.

**New unit L12·978-** (layer 12): 3596 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 3.113, neg 0.55); test 5/6 (AUC 0.889; mean pos 2.5, neg 0.715); p99 on the other memes' sentences 2.566 (max 4.875). Corpus max 4.844, p99 1.57; present in 2 of 32 Batman works, 3 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 1, only new 1 (Batman v Superman), only paper 18 (A Death in the Family, Arkham City, Arkham Knight, Batman & Robin, Batman '89, Batman Begins, Batman Forever, Batman Ninja). Batman works scoring above the median ordinary film: 23 of 32 for the new unit, 27 for the paper's.
- Top-50 peak tokens: ' close' ×3, ' brother' ×3, ' develops' ×2, ' best' ×2, ' bond' ×2, ' biological' ×2.
- Agent 1 (blind to the label, rates it conceptual, medium confidence) explains it as: *This unit fires on the modifier or noun that defines a close kinship or intimate relationship — adoptive (mother), adopted (son), best (friend), biological, brother, close, bond — typically on the word just before the relationship noun, or on 'bond' itself when two characters share a deep connection. The opposite pole fires on casual or first-time social contact (friend request, hang out, met, meets, meeting, makes the acquaintance of, wave to each other), so the unit draws deep family-like bonds as opposed to acquaintanceship.*
- Judge: **partial**. It captures deep adoptive or family-like bonds as opposed to acquaintance, which is the surrogate-parent relationship, but nothing of a butler, service or raising a child. Label the explanation supports: *Deep family-like bonds versus acquaintance*
- Agent 2 blind maximum: 4.25 against corpus max 4.84 (weakest of the top 8: 2.39; peak 'ive'): Raised by his adoptive mother, the orphan forms a close bond with his biological brother and his best friend.
- Agent 2 examples firing strongly (≥ 50% of max): 4 of 5; near misses: 0 of 5.
- Steering: α=4.84: none; α=4.84: none; α=4.84: none; α=9.69: none; α=9.69: weak The antidote (her grandmother, who is the town's healer); α=9.69: none
- **Verdict:** Partial. Deep family-like bonds (adoptive, best friend, brother) as opposed to acquaintance; the surrogate-parent relation without the butler. Four of five examples fire.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 4.84 'ive' · *Batman v Superman*: Luthor lures Superman out by kidnapping Martha Kent, Clark's adoptive mother, and Lois.
- 3.94 ' adopted' · *The Godfather*: Sonny and adopted son Tom Hagen, a lawyer who is the family's consigliere.
- 3.78 ' develops' · *The Shadow*: Ultimately, Cranston becomes the Tulku's student and develops psychic abilities of his own, including how to hypnotize others and bend their perceptions so tha…
- 3.72 ' best' · *Iron Man*: Tony Stark, who has inherited the defense contractor Stark Industries from his late father Howard Stark, tours in war-torn Afghanistan with his best friend and…
- 3.16 ' bond' · *Arkham Knight*: Nine months after the death of the Joker during the events of Arkham City, Batman is struggling to cope with the absence of his archenemy and the uncomfortable…
- 2.67 ' bond' · *E.T. the Extra-Terrestrial*: As Elliott and the creature bond, they share thoughts and emotions.
- 2.55 '1' · *Saving Private Ryan*: The United States Department of War receives communication that three of four Ryan brothers have been killed in action; the last, James Francis Ryan of the 101…
- 2.39 ' close' · *Count of Monte Cristo*: Knowing himself to be close to death from catalepsy, Faria tells Dantès the location of a vast treasure hidden by his former employer on the island of Monte Cr…

Its negative pole L12·978+ (max 2.28):

- 2.28 ' request' · *The Social Network*: Alone, Zuckerberg sends Albright a Facebook friend request and repeatedly refreshes the page.
- 2.09 ' hang' · *Mean Girls*: Cady begins dating Aaron and continues to hang out with Janis and Damian, the former of whom starts dating Kevin.
- 2.09 ' met' · *Mrs. Doubtfire*: At their next custody hearing, Daniel points out that he has met the judge's requirements ahead of schedule, then explains that his actions were done out of lo…
- 1.88 ' wave' · *Home Alone*: Marley notices Kevin, and the pair wave to each other knowingly.
- 1.77 ' flattering' · *Pride & Prejudice*: Darcy's manner has softened considerably, impressing the Gardiners, and Georgiana shares her brother's flattering reports about Elizabeth.

Agent 2's examples and near misses (activation, share of corpus max):

- example 2.06 (43%) ' adopted': A grieving widow tracks down the biological father of her adopted son before his illness worsens.
- example 2.75 (57%) ' bond': Two childhood best friends open a restaurant together, their bond tested when one falls for the other's sister.
- example 3.94 (81%) 'ive': The soldier returns home to care for his younger brother and the adoptive parents who raised them both.
- example 3.19 (66%) ' deep': A lonely robot and a stray dog form a deep bond as they cross the ruined desert together.
- example 2.64 (55%) ' friend': The detective's closest friend turns out to be the twin brother of the man he is hunting.
- near miss 1.02 (21%) ' she': The new intern sends a friend request to a stranger she met once at a conference in Denver.
- near miss 0.00 (0%) None: The city council adopted the zoning plan and the mayor closed the meeting before the press could ask.
- near miss 0.02 (0%) ' director': The film wins best picture, and the director makes the acquaintance of a producer at the after-party.
- near miss 0.08 (2%) ' biological': A bail bondsman tracks a fugitive who skipped out on a biological weapons trial in Baltimore.
- near miss 0.13 (3%) ' then': Two strangers wave to each other across the platform, then hang out at a diner until the trains resume.

Steered generations:

- α=4.84 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is carrying a backpack and seems to be in a hurry. He is followed by a man in a black coat. Jack is trying to avoid being caught. The man in
- α=4.84 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is struggling to find her place in the world. She is living in a small town, and her only friend is her neighbor, Clara. One day,…
- α=4.84 · The story begins when → the young girl, who is the protagonist, is playing with her dog. She is very close to her dog. She is playing with her dog in the park. She is very happy. She is playing with her dog, and then she
- α=9.69 · Plot summary: The film opens as → a young man, who is a student of the art of the sword, is told by his master that he must find his own way. He is then given a letter to deliver to a certain person. He sets out on his journey,
- α=9.69 · Plot: In the first act, → the story introduces the protagonist, a young woman named Clara, who is struggling to find her place in the world. She is living in a small town with her grandmother, who is the town's healer. Clara…
- α=9.69 · The story begins when → the young girl, Lina, is playing with her friend, Mina. They are playing with a doll, and Lina is pretending to be a doctor. She is checking Mina's arm for a cut. Mina is pretending

### 15. Taking in an orphan as a ward

**Paper's unit L35·6441-** (layer 35): dev 0/10, test 0/6 (AUC 0.5); gate-open corpus max 70.5, present in 10 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ',' ×9, ' and' ×7, '.' ×7, ' with' ×2, ' save' ×2, 'ed' ×1.

**New unit L29·4544+** (layer 29): 3299 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 8.15, neg 1.195); test 5/6 (AUC 0.833; mean pos 4.542, neg 1.37); p99 on the other memes' sentences 4.515 (max 10.75). Corpus max 14.375, p99 4.22; present in 0 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 0, only new 0 (), only paper 10 (1943 serial, A Death in the Family, Batman & Robin, Batman Forever, Dark Knight Returns, Gotham by Gaslight, Hush, Lego Batman). Batman works scoring above the median ordinary film: 21 of 32 for the new unit, 16 for the paper's.
- Top-50 peak tokens: ' a' ×5, ' taken' ×3, ' his' ×3, "'s" ×3, ' to' ×3, ' Batman' ×3.
- Agent 1 (blind to the label, rates it conceptual, medium confidence) explains it as: *This unit fires on a child or young creature who is being placed under someone's care, abandoned, adopted or taken in: 'abandon him [in the sewers, where penguins take him in]', 'sends his son off', 'borne his child ... kept the child', 'taken in by Border Collie Fly', 'orphanage', 'joint custody ... the children', 'signs Babe up'. The peak is on the word for the child or on the verb of taking in. The opposite pole fires on imprisonment and sentences: 'locked in', 'two consecutive life sentences', 'life imprisonment', 'witness protection program'.*
- Judge: **direct**. It describes a child being abandoned, adopted or taken into someone's care, with 'orphanage' and the verb of taking in, which is the labelled trope itself. Label the explanation supports: *A child abandoned, adopted or taken in*
- Agent 2 blind maximum: 3.33 against corpus max 14.38 (weakest of the top 8: 7.88; peak ' taken'): Ashamed of their deformed baby, the wealthy couple abandon the infant in the river, where he is taken in and raised by the creatures of the sewer.
- Agent 2 examples firing strongly (≥ 50% of max): 1 of 5; near misses: 0 of 5.
- Steering: α=14.38: none; α=14.38: none; α=14.38: none; α=28.75: none; α=28.75: none; α=28.75: none
- **Verdict:** Concept unit. A child abandoned, adopted or taken into someone's care; the blind explanation is the trope itself. Only one of five examples fires strongly (the unit's maximum, 14.4, is set by Batman Forever's Dick Grayson sentences), so it passes on the explanation and the near misses.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 14.38 ' him' · *Batman Returns*: In Gotham City, two wealthy socialites, horrified by the birth of their malformed and feral son Oswald, abandon him in the sewers, where a colony of penguins t…
- 11.88 ' son' · *Forrest Gump*: Forrest sends his son off on his first day of school as the feather from the movie's opening floats away in the wind.
- 9.88 ' child' · *Vertigo*: A local historian explains that Carlotta Valdes committed suicide: she had been the mistress of a wealthy married man and borne his child, and the otherwise ch…
- 9.06 ' taken' · *Babe*: There, he is taken in by Border Collie Fly, her irascible husband Rex and their puppies and befriends a duck named Ferdinand, who wakes people by crowing like…
- 8.38 ' a' · *Goodfellas*: Realizing that the unsanctioned murder of a made man will invite retribution, Jimmy, Henry and Tommy bury the body in Upstate New York, stabbing and fatally sh…
- 8.06 ' orphan' · *Dark Knight Rises*: Since Bruce is also presumed dead, Wayne Manor becomes an orphanage, and Bruce's estate is left to Alfred.
- 7.97 ' children' · *Mrs. Doubtfire*: She arranges joint custody, allowing Daniel to take the children daily after school.
- 7.88 ' Babe' · *Babe*: When Esme leaves on a trip, Hoggett signs Babe up for sheepdog trials.

Its negative pole L29·4544- (max 8.38):

- 8.38 ' Ram' · *Saving Private Ryan*: Remorseful for ignoring their behavior, Miller shouts for anyone who knows Ryan; one soldier tells him that Ryan was reassigned to defend a vital bridge in the…
- 7.28 ' a' · *The Shawshank Redemption*: He is befriended by Ellis Boyd "Red" Redding, a contraband smuggler serving a life sentence, who procures for him a rock hammer and a large poster of Rita Hayw…
- 6.41 'au' · *Scarlet Pimpernel*: But Marguerite is being blackmailed by Citizen Chauvelin, the wily new French envoy to England: Chauvelin's agents have stolen a letter proving that Armand is…
- 5.94 ' locked' · *Titanic*: Jack is arrested and locked in the master-at-arms' office, and Cal pockets the necklace.
- 5.78 ' consecutive' · *The Shawshank Redemption*: In 1947, Portland, Maine, banker Andy Dufresne arrives at Shawshank State Prison to serve two consecutive life sentences for murdering his wife and her lover.

Agent 2's examples and near misses (activation, share of corpus max):

- example 11.56 (80%) ' taken': After his parents are murdered, the orphaned boy is taken in by the family butler, who raises him as his own.
- example 1.52 (10%) ' a': A runt piglet won at a fair is adopted by a sheepdog on the farm and raised alongside her puppies.
- example 2.45 (17%) ' the': The divorcing couple fight for custody of the children, and the father disguises himself as a nanny to see them.
- example 6.03 (42%) ' off': A scientist sends his infant son off in a rocket as the planet dies, and farmers find the child in a field.
- example 7.06 (49%) 'ed': The circus acrobat's orphaned son is placed in the city orphanage until the millionaire agrees to become his guardian.
- near miss 0.19 (1%) ' sentenced': The Joker is locked in Arkham Asylum and sentenced to two consecutive life terms for the ferry bombings.
- near miss 0.00 (0%) None: Batman adopts a new strategy and abandons the Batmobile in the tunnel before the Joker can trace it.
- near miss 0.60 (4%) ' in': Gordon takes in the view from the rooftop and switches on the signal over the city.
- near miss 0.21 (2%) ' lam': The Scarecrow is taken into police custody after Batman leaves him tied to a lamppost in the Narrows.
- near miss 1.25 (9%) ' his': Young Bruce Wayne watches the opera with his parents and asks to leave early because of the bat costumes.

Steered generations:

- α=14.38 · Plot summary: The film opens as → a young girl, Lina, is playing with her brother, Pablo, in the streets of Barcelona. They are playing a game of "chess" with sticks, and Lina is trying to win. Pablo is trying to get her
- α=14.38 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is struggling to find her place in the world. She is an aspiring artist, but her family expects her to pursue a more stable career…
- α=14.38 · The story begins when → the narrator, a young man, is walking through the forest. He is alone, and he feels a bit uneasy. He hears a sound, like a footstep, and turns around. But there is no one there. He continues walking
- α=28.75 · Plot summary: The film opens as → a young girl, Lina, is playing with her brother, Pablo, in the streets of Barcelona. They are playing a game of "chess" with sticks, and Lina is trying to win. Pablo is trying to get her
- α=28.75 · Plot: In the first act, → the story introduces the protagonist, a young woman named Emily, who is struggling to find her place in the world. She is an aspiring writer, but her work is not being recognized. She moves to a smal…
- α=28.75 · The story begins when → the narrator, a young man, is walking in the forest. He is alone, and he is thinking about his life. He is thinking about his past, his family, and his future. He is also thinking about the forest it…

### 16. The villain's lair

**Paper's unit L4·7177-** (layer 4): dev 4/10, test 2/6 (AUC 0.583); gate-open corpus max 8.125, present in 6 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: 'ir' ×9, ' Las' ×3, ' Lazar' ×3, ' La' ×3, ' Lars' ×2.

**New unit L10·605-** (layer 10): 5746 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 10.931, neg 3.065); test 6/6 (AUC 0.944; mean pos 9.802, neg 3.283); p99 on the other memes' sentences 7.077 (max 14.25). Corpus max 15.5, p99 9.002; present in 21 of 32 Batman works, 1 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 6, only new 15 (A Death in the Family, Arkham Knight, Batman '66, Batman '89, Batman Ninja, Batman R.I.P., Batman Returns, Dark Knight Returns), only paper 0 (). Batman works scoring above the median ordinary film: 32 of 32 for the new unit, 13 for the paper's.
- Top-50 peak tokens: 'ave' ×21, 'out' ×6, ' la' ×6, ' base' ×3, 'ench' ×3, 'ill' ×2.
- Agent 1 (blind to the label, rates it conceptual, high confidence) explains it as: *This unit fires on the token that completes the name of a character's secret headquarters or hideout — Batc\|ave, Bat's Cave, hide\|out, la\|ir, base — when a story describes someone setting up, keeping, or being taken to such a hidden base; the firing sits on the word that names the place rather than on the character or the action. The opposite pole is weak and scattered (released, burns, deadly, flash, shoot), roughly weapons and violent actions, so the unit is about a place of concealment as opposed to the fighting that happens elsewhere.*
- Judge: **direct**. It describes the naming of a secret headquarters or hideout (lair, hideout, base, Batcave), the defining idea of a lair, though it does not distinguish a villain's from a hero's. Label the explanation supports: *A character's secret hideout or base*
- Agent 2 blind maximum: 19.88 against corpus max 15.50 (weakest of the top 8: 12.88; peak ' base'): After dark the caped vigilante retreats beneath the manor to his secret lair, the hidden cave that serves as his base.
- Agent 2 examples firing strongly (≥ 50% of max): 5 of 5; near misses: 0 of 5.
- Steering: α=15.5: weak The Batcave and bat gear (reaches a secret underground base); α=15.5: none; α=15.5: none; α=31.0: strong The villain's lair (the main base of the villain is a mobile) (concept); α=31.0: weak The Batcave and bat gear (the group of heroes is in a secret base); α=31.0: strong The villain's lair (The villain has a lair) (concept)
- **Verdict:** Concept unit, and the one that is more present in Batman than the paper's unit (21 works against 6). Fires on the token that names a hideout, lair or base; blind maximum 128% of the corpus maximum; all five examples fire; steering produces 'the villain has a lair'.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 15.50 ' base' · *Mask of the Phantasm*: Heartbroken, Bruce commits to a new life as a vigilante crime fighter, converting a cavern under Wayne Manor into a base and assuming the mantle of Batman for…
- 15.00 'ave' · *Batman Ninja*: Batman discovers that Alfred Pennyworth is also in the past and has built a Batcave outside Edo.
- 14.19 'ave' · *Arkham Knight*: Oracle has set up her communications headquarters in the Gotham clock tower, which also houses a makeshift Batcave.
- 13.88 ' Cave' · *1943 serial*: In his secret Bat's Cave, the Batman interrogates one of Daka's henchmen, who reveals the radium was to have been taken to The House of the Open Door, located…
- 13.75 'out' · *Batman Returns*: Thirty-three years later, during the Christmas season, wealthy industrialist Max Shreck is abducted by the Red Triangle Gang—former circus performers implicate…
- 13.56 'ave' · *Arkham Asylum (game)*: Afterward, he goes to a secret Batcave installation he had hidden on the island, where he restocks his gadgets.
- 13.44 'ave' · *Batman '89*: Batman rescues Vicki, takes her to the Batcave, and provides her with all of his research on Smylex, which will allow Gotham's residents to escape the toxin.
- 12.88 ' la' · *1943 serial*: Batman and Robin infiltrate the gang's lair (also Dr. Daka's laboratory), hidden inside a still-open business, a Fun House ride.

Its negative pole L10·605+ (max 4.12):

- 4.12 ' released' · *Arkham Asylum (comic)*: On April Fools' Day (the date Arkham's family was murdered), Cavendish released the patients and lured Batman to the asylum, believing him to be the bat Arkham…
- 3.78 ' burns' · *Iron Man*: The armored Stark battles his way out of the cave to find the dying Yinsen, then burns the Ten Rings' weapons and flies away, crashing in the desert and destro…
- 3.58 'sh' · *Knightfall*: Within the regular series, the buildup to "Knightfall" begins with a six-issue run in Batman #484-489 (September 1992 - February 1993), in which Batman (at the…
- 3.55 ' deadly' · *The Long Halloween*: On New Year's Eve, Batman stops the Joker from using deadly laughing gas to kill everyone in Gotham Square.
- 3.52 ' flash' · *Hush*: Using various weapons, including flashbang grenades, hypersonics, electroshock, and a kryptonite ring to defend himself, Batman stalls Superman while Catwoman…

Agent 2's examples and near misses (activation, share of corpus max):

- example 10.62 (68%) 'out': The thief and his crew set up a hideout in an abandoned warehouse on the edge of the city.
- example 15.00 (97%) ' la': The clown is taken blindfolded to the mob boss's secret lair beneath the old slaughterhouse.
- example 14.62 (94%) ' base': Wounded, the masked archer drags himself back to the cave that serves as his hidden base.
- example 10.25 (66%) ' headquarters': The young spy discovers that the arms dealer keeps a secret headquarters inside a hollowed-out volcano.
- example 11.06 (71%) 'out': Hunted by the police, the gang retreats to its hideout in the sewers beneath the opera house.
- near miss 0.75 (5%) ' cave': A colony of bats pours out of the limestone cave at dusk while the geologists watch from the ridge.
- near miss 0.43 (3%) ' letter': The rookie hides the stolen letter under his pillow and lies awake, afraid his roommate will find it.
- near miss 0.10 (1%) ' home': The shortstop steals second base in the ninth inning and the home crowd rises to its feet.
- near miss 3.17 (20%) ' Joker': Batman and the Joker shoot it out on the rooftop as the building burns and the deadly flash fades.
- near miss 2.28 (15%) ' and': The company moves its headquarters to a glass tower downtown and invites the press to the opening.

Steered generations:

- α=15.5 · Plot summary: The film opens as → a young man, named Jack, is trying to escape from a group of armed men. He is pursued through the streets of a city, dodging bullets and explosions. Eventually, he reaches a secret underground base,…
- α=15.5 · Plot: In the first act, → the protagonist is introduced. The protagonist is a young man who lives in a small town. He is a bit of a loner, but he has a loyal friend. The town is a bit run-down, but it's not too bad
- α=15.5 · The story begins when → the young girl, who is the daughter of the king of the island of the same name, is kidnapped by the villain. The girl's name is Princess Lina, and the villain is Captain Hook. The king's name is King…
- α=31.0 · Plot summary: The film opens as → a mobile, the main base of the villain is a mobile, and the heroes have a base. In the 1960s, there were many secret bases, but in the 1980s, there were more
- α=31.0 · Plot: In the first act, → the group of heroes is in a secret base. The base has a circular track that goes around the base. The track is 100 meters in diameter. The base has a circular track that goes around the base. The tra…
- α=31.0 · The story begins when → the hero, a young woman, is kidnapped by a villain. The villain has a fortress with a secret room. The hero must escape. The villain has a robot. The hero must outsmart the villain. The villain has a…

### 17. A bomb

**Paper's unit L3·1967-** (layer 3): dev 2/10, test 2/6 (AUC 0.444); gate-open corpus max 13.875, present in 14 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' bomb' ×26, ' bombs' ×7, 'mobile' ×5, ' bombing' ×2, 'k' ×2.

**New unit L10·4927+** (layer 10): 2511 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 14.228, neg 1.724); test 6/6 (AUC 1.0; mean pos 13.448, neg 0.408); p99 on the other memes' sentences 10.155 (max 17.5). Corpus max 24.625, p99 9.0; present in 7 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 6, only new 1 (The Dark Knight), only paper 8 (Arkham Asylum (game), Arkham Knight, Batman v Superman, Dark Knight Returns, Lego Batman, Mask of the Phantasm, The Long Halloween, Year One). Batman works scoring above the median ordinary film: 28 of 32 for the new unit, 19 for the paper's.
- Top-50 peak tokens: ' bomb' ×19, ' deton' ×6, ' explosives' ×4, ' blow' ×4, ' bombs' ×3, '4' ×2.
- Agent 1 (blind to the label, rates it conceptual, high confidence) explains it as: *This unit fires on explosives and their use — bomb, bombs, C-4, explosives, detonate, blow (up) — on the noun or verb that names the explosive device or its detonation, whether it is thrown, planted, or threatened. The opposite pole is weak and covers other kinds of weapons: flamethrower, directed-energy weapon, machine gun, gun, gas, injection, so the unit draws explosives as opposed to guns, flame, energy, and chemical weapons.*
- Judge: **direct**. It describes explosives and their detonation (bomb, C-4, detonate, blow up) as opposed to other weapons, which is the labelled object itself. Label the explanation supports: *Explosives and detonation*
- Agent 2 blind maximum: 19.50 against corpus max 24.62 (weakest of the top 8: 14.06; peak ' blow'): The terrorist plants C-4 explosives beneath the stadium and threatens to detonate the bomb and blow up the city unless he is paid.
- Agent 2 examples firing strongly (≥ 50% of max): 5 of 5; near misses: 0 of 5.
- Steering: α=24.62: strong A bomb (a bomb is placed in the building) (concept); α=24.62: strong A bomb (the bomb was defused by the police) (concept); α=24.62: weak A bomb (the explosion will be 100% efficient) (concept); α=49.25: weak A bomb (The explosion is 1000 pounds) (concept); α=49.25: strong A bomb (the bomb was a 1950s model. The explosion) (concept); α=49.25: weak A bomb (possible to blow up the whole thing) (concept)
- **Verdict:** Concept unit. Explosives and detonation as opposed to other weapons (bomb, C-4, detonate, blow up); all five examples fire, no near miss, and steering produces bombs and explosions in six of six generations. Replaces the paper's layer-3 word unit that a box-office bomb fooled.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 24.62 '4' · *Die Hard*: Meanwhile, McClane kills more terrorists and recovers their bag of C-4 and detonators.
- 19.12 '4' · *Die Hard*: McClane throws some C-4 down an elevator shaft, causing an explosion that kills some of the terrorists and ends the counterattack.
- 17.62 ' bomb' · *A Death in the Family*: Jason throws himself on the bomb to protect Sheila as the warehouse explodes.
- 16.12 ' bomb' · *Under the Red Hood*: They chase Red Hood to a train station, where he escapes after detonating a bomb, which injures Nightwing.
- 15.19 ' bomb' · *The Long Halloween*: Falcone responds by hiring the Sullivan family, a gang of Irish hitmen led by Donald "Donny Boy" Sullivan, to destroy Dent's home with a bomb, but he and his w…
- 14.56 ' bomb' · *Batman Forever*: Two-Face hijacks the event and threatens to detonate a bomb unless Batman reveals his identity.
- 14.19 ' deton' · *The Dark Knight*: The explosives detonate, killing Rachel and severely burning Dent's face on one side.
- 14.06 ' bomb' · *Under the Red Hood*: Joker eventually leaves and locks Jason inside the warehouse with a bomb, which explodes and kills Jason before Batman arrives.

Its negative pole L10·4927- (max 1.78):

- 1.78 'row' · *Alien*: He then enters the ducts with a flamethrower but is ambushed and killed by the alien.
- 1.68 '-energy' · *Return of the Joker*: When a directed-energy weapon strikes Pryce's yacht, Terry rescues him before turning him over to the police.
- 1.54 ' burns' · *Iron Man*: The armored Stark battles his way out of the cave to find the dying Yinsen, then burns the Ten Rings' weapons and flies away, crashing in the desert and destro…
- 1.48 ' gun' · *Back to the Future*: After Doc inputs a destination time of November 5, 1955 (the day he conceived his time travel invention), the terrorists arrive unexpectedly and gun him down.
- 1.45 ' inject' · *Knightfall*: In Batman #489-490 (February–March 1993), Bane interferes with encounters pitting villains Killer Croc and the Riddler against Batman, and, to test Batman's li…

Agent 2's examples and near misses (activation, share of corpus max):

- example 14.00 (57%) ' blows': A bomb squad officer races to defuse a device wired to a city bus before it blows up.
- example 18.25 (74%) ' deton': The clown rigs two ferries with explosives and dares the passengers to detonate each other first.
- example 25.75 (105%) ' blow': A demolitions expert plants charges of C-4 throughout the dam, planning to blow it sky-high at dawn.
- example 12.31 (50%) ' off': A rogue agent threatens to set off a nuclear bomb in the capital unless the president resigns.
- example 18.12 (74%) ' deton': The resistance fighters throw grenades into the tunnel and detonate the bridge as the convoy crosses.
- near miss 1.28 (5%) ' a': The gunman sprays the lobby with a machine gun while his partner torches the vault with a flamethrower.
- near miss 3.00 (12%) ' blows': The comedy bombs at the box office, and the studio blows its entire marketing budget on a sequel anyway.
- near miss 6.59 (27%) ' bomb': The chemistry between the two leads is explosive, and the bombshell revelation in the finale stuns audiences.
- near miss 0.00 (0%) None: The nurse administers a lethal injection while the guards flood the cell block with nerve gas.
- near miss 0.04 (0%) ' blows': She blows a kiss from the train window as he photobombs the wedding picture on the platform.

Steered generations:

- α=24.62 · Plot summary: The film opens as → a bomb is placed in the building. The bomb is set to explode in 10 minutes. The bomb is set to explode in 10 minutes. The bomb is set to explode in 10 minutes. The bomb is set
- α=24.62 · Plot: In the first act, → the bomb was set to explode in 10 minutes. But the bomb was defused by the police. But the bomb was defused by the police. But the bomb was defused by the police. But the bomb was defused
- α=24.62 · The story begins when → the sun is about to set. The story is very powerful, and the explosion will be 100% efficient. The story is 100 meters from the ground. The ground is 10 meters wide. The explosion
- α=49.25 · Plot summary: The film opens as → the sun goes down. The explosion is 1000 pounds. The sun goes down. The explosion is 1000 pounds. The sun goes down. The explosion is 1000 pounds. The sun goes
- α=49.25 · Plot: In the first act, → the bomb was a 1950s model. The explosion was massive. The blast was so powerful it was enough to make the entire building weigh 10 tons. If it weighs 10 tons, then the explosion is
- α=49.25 · The story begins when → the sun is about to set. But the sun is a bit of a trick, and it's possible to blow up the whole thing. But the sun is a bit of a trick, and it's possible to blow up the whole thing

### 18. A funeral

**Paper's unit L1·5621+** (layer 1): dev 5/10, test 3/6 (AUC 0.722); gate-open corpus max 13.875, present in 9 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' funeral' ×9, 'eral' ×2, 'is' ×2, 'rier' ×2, 'url' ×1.

**New unit L34·6351+** (layer 34): 4430 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 10.197, neg 0.255); test 5/6 (AUC 0.972; mean pos 10.238, neg 0.359); p99 on the other memes' sentences 4.685 (max 8.312). Corpus max 32.25, p99 5.971; present in 0 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 0, only new 0 (), only paper 9 (Arkham Knight, Batman Returns, Batman v Superman, Dark Knight Returns, Dark Knight Rises, Gotham by Gaslight, Hush, The Batman). Batman works scoring above the median ordinary film: 12 of 32 for the new unit, 9 for the paper's.
- Top-50 peak tokens: ' that' ×4, ' the' ×4, "'s" ×3, ' sh' ×3, ' of' ×3, ' a' ×3.
- Agent 1 (blind to the label, rates it conceptual, medium confidence) explains it as: *This late-layer unit fires at the position just before a spirit/soul-type word is produced: on 'Mufasa's' before 'spirit', on 'sh' beginning 'shaman', on 'pir' in 'free-spirited', before 'Soul' in 'Heart and Soul', before 'an entity', after 'superstitions regarding'; it anticipates a spiritual or supernatural-essence noun as the next token, as opposed to positions anticipating concrete material things or surnames.*
- Judge: **adjacent**. It captures the anticipation of a spirit or soul noun, which shares only the afterlife theme with a funeral and describes no ceremony, mourning or burial. Label the explanation supports: *Anticipating a spirit or soul noun*
- Agent 2 blind maximum: 26.12 against corpus max 32.25 (weakest of the top 8: 11.94; peak ' restless'): Haunted by his dead father's spirit, the young prince consults the tribal shaman about the restless soul and the ghostly entity that haunts the kingdom.
- Agent 2 examples firing strongly (≥ 50% of max): 4 of 5; near misses: 0 of 5.
- Steering: α=32.25: none; α=32.25: none; α=32.25: weak Heir to a dead father's fortune (the king is dead. The king was the father); α=64.5: none; α=64.5: none; α=64.5: none
- **Verdict:** Adjacent. A layer-34 unit anticipating a spirit or soul noun; the funeral pairs drive it (6 of 6 held-out at 5 of 6) through mourning and the afterlife, not the ceremony. Four of five examples fire, so the explanation is right about the unit and the unit is wrong for the label.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 32.25 "'s" · *The Lion King*: He encounters Rafiki, who tells Simba that Mufasa's spirit lives on in him.
- 26.75 ' sh' · *The Lion King*: King Mufasa and Queen Sarabi's newborn son, Simba, is presented to the gathered animals by Rafiki, the mandrill who is the kingdom's shaman and advisor.
- 22.88 ' regarding' · *The Exorcist*: Unaware of traditional superstitions regarding the board, she plays with it alone.
- 19.75 'pir' · *The Sound of Music*: In 1938 in Salzburg, at the end of the Federal State of Austria regime, Maria is a free-spirited young Austrian woman studying to become a nun at Nonnberg Abbe…
- 17.25 ' of' · *La La Land*: While stuck in Los Angeles traffic ("Another Day of Sun"), jazz pianist Sebastian "Seb" Wilder has a moment of road rage directed at aspiring actress Amelia "M…
- 15.38 ' with' · *The Exorcist*: She tells her mother she has made repeated contact with an entity she calls "Captain Howdy."
- 14.19 ',' · *Puss in Boots*: The youngest son laments his poor inheritance, but the cat, who turns out to be extraordinarily clever and resourceful, assures his master that he can bring hi…
- 11.94 ' and' · *Big*: They use the store's Walking Piano to play duets ("Heart and Soul" and "Chopsticks"), and MacMillan invites Josh to a massive marketing campaign pitch meeting…

Its negative pole L34·6351- (max 23.25):

- 23.25 ' Silver' · *Arkham Knight*: Arkham Knight also features appearances and voice cameos by various characters drawn from the history of Batman comics and the previous Arkham games, including…
- 21.25 ' into' · *Spirited Away*: Spirits begin to appear, and Chihiro discovers that her parents have been transformed into pigs and that she cannot cross the now-flooded river.
- 16.88 ' with' · *Dark Knight Returns*: However, an aging Oliver Queen manages to shoot Superman with a kryptonite-tipped arrow to weaken him, which allows Batman to beat him into submission.
- 16.88 ' "' · *Ghostbusters*: Peter meets with Dana and informs her that Zuul was a demigod worshipped as a servant to "Gozer the Gozerian", a shapeshifting god of destruction.
- 16.62 ' to' · *Dracula*: Dracula crumbles to dust, freeing Mina from her vampiric curse.

Agent 2's examples and near misses (activation, share of corpus max):

- example 30.38 (94%) "'s": The grieving widow visits a shaman who claims he can summon her husband's spirit from beyond the grave.
- example 18.12 (56%) ' wandering': A free-spirited drifter arrives in a mountain town rife with superstitions regarding ghosts and wandering souls.
- example 31.88 (99%) ' restless': The priest warns the family that an entity has taken hold of the house and its restless spirit must be exorcised.
- example 23.75 (74%) "'s": On a desert pilgrimage, the exile is visited by his dead mentor's ghost, who tells him to reclaim his soul.
- example 6.22 (19%) ' whose': The medium channels the spirits of the drowned sailors, whose souls still haunt the lighthouse.
- near miss 0.00 (0%) None: The bartender pours the finest spirits from behind the bar: whisky, gin and a bottle of aged rum.
- near miss 0.00 (0%) None: Mufasa's paw print is pressed into the mud where Simba stands, learning the geography of the Pride Lands.
- near miss 2.33 (7%) ',': The band plays soul music all night at the Detroit club, and the crowd dances until dawn.
- near miss 0.00 (0%) None: The lawyer explains that the shell company is a separate legal entity registered in Delaware.
- near miss 0.00 (0%) None: The cardiologist examines the patient's heart and lungs before scheduling the surgery for Monday.

Steered generations:

- α=32.25 · Plot summary: The film opens as → a young girl, Lina, is playing with her toys. She is told by her mother that they are moving to a new house. Lina is upset, but her mother reassures her that the new house will be better. L
- α=32.25 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is struggling to find her place in the world. She is an aspiring artist, but her family expects her to pursue a more stable career…
- α=32.25 · The story begins when → the narrator, a young man, is caught in a storm in the village of Elsinore. He is told by a peasant that the king is dead. The king was the father of the prince, who is now the ruler of Denmark
- α=64.5 · Plot summary: The film opens as → a young girl, Lina, is playing with her toys. She is told by her mother that they are moving to a new house. Lina is upset, but her mother reassures her that the new house will be better. L
- α=64.5 · Plot: In the first act, → the story introduces the main characters and the setting. The protagonist, a young woman named Clara, is struggling to find her place in the world. She is an aspiring writer, but her work is not bein…
- α=64.5 · The story begins when → the narrator, a young man, is walking through the forest. He is alone, and he feels a bit uneasy. He hears a sound, and then sees a man in a dark cloak. The man is very pale, and he has

### 19. Fell into a vat of chemicals

**Paper's unit L0·6344-** (layer 0): dev 7/10, test 4/6 (AUC 0.722); gate-open corpus max 5.062, present in 8 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: 'ylum' ×20, ' chemical' ×6, 's' ×5, ' emotional' ×4, ' chemicals' ×2, ' pharmaceutical' ×2.

**New unit L15·7058-** (layer 15): 3164 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 2.429, neg 0.627); test 6/6 (AUC 0.903; mean pos 0.963, neg 0.229); p99 on the other memes' sentences 0.689 (max 2.203). Corpus max 1.68, p99 0.523; present in 2 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 1, only new 1 (Batman Returns), only paper 7 (A Death in the Family, Arkham Asylum (game), Arkham Knight, Batman '89, Batman R.I.P., The Killing Joke, White Knight). Batman works scoring above the median ordinary film: 18 of 32 for the new unit, 29 for the paper's.
- Top-50 peak tokens: ' the' ×3, ' a' ×3, ' of' ×2, ' Witch' ×2, ' fire' ×2, 'ini' ×1.
- Agent 1 (blind to the label, rates it mixed, low confidence) explains it as: *This unit fires very weakly on the proper nouns of fairy-tale and fantasy worlds, with half its strongest firings coming from The Wizard of Oz — Wicked Witch, Wizard, Emerald City, (return to) Kansas — plus Vizzini, the Lazarus Pit, and confit byaldi, so it seems to respond to named fantastical or storybook entities. The opposite pole, slightly stronger, lands in realistic dialogue-driven social scenes (an assistant's reference, orders her to leave, a diplomatic incident, a reporter's question, a letter of gratitude), so the weak distinction is storybook fantasy as opposed to everyday social drama. The signal is too faint to be sure of.*
- Judge: **none**. It describes faint firing on storybook and fantasy proper nouns, mostly from The Wizard of Oz, with the Lazarus Pit as one stray, and nothing of a fall into chemicals. Label the explanation supports: *Storybook fantasy names (faint signal)*
- Agent 2 blind maximum: 0.82 against corpus max 1.68 (weakest of the top 8: 1.00; peak 'erald'): A farm girl and her little dog follow the yellow brick road to the emerald city, where the wizard must save them from the wicked witch.
- Agent 2 examples firing strongly (≥ 50% of max): 0 of 5; near misses: 0 of 5.
- Steering: α=1.68: none; α=1.68: none; α=1.68: none; α=3.36: none; α=3.36: none; α=3.36: none
- **Verdict:** Miss. The best separator is a faint unit (maximum 1.7) on storybook names; held-out 6 of 6 only because its activations are near zero on both sides. No concept unit for the vat found.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 1.68 ' of' · *Batman Returns*: Max attempts to kill his timid secretary, Selina Kyle, by pushing her out of a window after she discovers his plan to build a power plant that would secretly d…
- 1.25 ' Witch' · *The Wizard of Oz*: During their quest, Dorothy and Toto are captured by flying monkeys and taken to the Wicked Witch's castle.
- 1.22 'ini' · *The Princess Bride*: Before the wedding, she is kidnapped by crafty Sicilian criminal Vizzini and his accomplices: gentle giant Fezzik, and expert Spanish swordsman Inigo Montoya,…
- 1.22 ' Wizard' · *The Wizard of Oz*: They join Dorothy, hoping that the Wizard can help them as well.
- 1.20 ' Pit' · *Under the Red Hood*: Ra's explains that he felt responsible for Jason's death and, as a peace offering, swapped Jason's body for a fake and revived him in the Lazarus Pit, but foll…
- 1.05 ' to' · *The Wizard of Oz*: Glinda then reappears and tells Dorothy she always had the power to return to Kansas using the ruby slippers, but had to find that out for herself.
- 1.04 ' Emerald' · *The Wizard of Oz*: They reach the Emerald City, despite the efforts of the Wicked Witch.
- 1.00 'it' · *Ratatouille*: Remy prepares confit byaldi, a variation of ratatouille, which evokes in Ego a sudden and involuntary recollection of his childhood and of his mother's cooking.

Its negative pole L15·7058+ (max 2.20):

- 2.20 ' a' · *The Devil Wears Prada*: The editor recounts that when he called Runway for a reference, Miranda told him that Andy was the biggest disappointment she had ever had as an assistant and…
- 2.16 ' leave' · *Pride & Prejudice*: Deeply insulted, Elizabeth orders her to leave.
- 1.41 ' a' · *A Death in the Family*: The Joker is Iran's representative to the UN and will be giving a speech on the floor of the General Assembly, and any confrontation between Batman and him cou…
- 1.25 ' a' · *Notting Hill*: She says she would, then requests a reporter to repeat his question, "How long are you intending to stay here in Britain?"
- 1.07 ' letter' · *Vertigo*: The next day, Madeleine stops to deliver a letter of gratitude to Scottie, and they spend the day together.

Agent 2's examples and near misses (activation, share of corpus max):

- example 0.10 (6%) 'erald': A scarecrow, a tin man and a cowardly lion help a lost girl reach the wizard's emerald palace.
- example 0.03 (2%) ' dread': A farmhand, a giant and a swordsman rescue a princess from a scheming prince and a dread pirate.
- example 0.24 (14%) ' witch': The wicked witch sends her flying monkeys to capture the girl, whose ruby slippers can carry her home.
- example 0.59 (35%) ' pit': An immortal assassin bathes in a mystical pit that restores life, then plots to destroy the caped vigilante's city.
- example 0.06 (3%) ' by': A wooden puppet who longs to be a real boy is swallowed by a whale alongside his old carpenter father.
- near miss 0.01 (1%) ' to': An assistant asks her boss for a reference, and he orders her to leave the office before lunch.
- near miss 0.02 (1%) ' elevator': A wheat farmer in Kansas sells his harvest at the grain elevator and argues with the bank about his loan.
- near miss 0.07 (4%) ' investigation': The senator calls the investigation a witch hunt, and a reporter's question causes a diplomatic incident.
- near miss 0.05 (3%) ' gratitude': The accountant is a wizard with spreadsheets, and his boss writes him a letter of gratitude.
- near miss 0.14 (9%) ' shop': A jeweler appraises an emerald ring in his Seattle shop while his daughter negotiates the lease with the landlord.

Steered generations:

- α=1.68 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is carrying a backpack and seems to be searching for something. He finds a small, red box. He opens it and finds a letter inside. The letter s…
- α=1.68 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is struggling to find her place in the world. She is an aspiring artist, but her family expects her to pursue a more stable career…
- α=1.68 · The story begins when → the narrator, a young man, is caught in a storm in the village of Elsinore. He is told by a peasant that the king is in the castle, but he is not sure if the king is the same as the king
- α=3.36 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is carrying a backpack and seems to be in a hurry. He is followed by a man in a black coat. Jack is trying to avoid being caught. The man in
- α=3.36 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is struggling to find her place in the world. She is living in a small town, working as a barista, and feeling trapped in her mund…
- α=3.36 · The story begins when → the narrator, a young man, is caught in a storm in the village of Elsinore. He is told by a peasant that the king is in the castle, but he is not sure if the king is the same as the king

### 20. Two-Face's coin

**Paper's unit L26·3329-** (layer 26): dev 10/10, test 5/6 (AUC 0.944); gate-open corpus max 24.875, present in 9 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: 'ace' ×10, ' of' ×4, ' coin' ×3, ' ' ×3, ',' ×2, ' "' ×2.

**New unit L12·4733-** (layer 12): 1996 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 5.277, neg 0.603); test 6/6 (AUC 1.0; mean pos 5.516, neg 0.361); p99 on the other memes' sentences 1.076 (max 6.25). Corpus max 8.25, p99 0.614; present in 3 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 3, only new 0 (), only paper 6 (Arkham Asylum (game), Arkham City, Arkham Knight, Batman '89, Dark Knight Returns, The Long Halloween). Batman works scoring above the median ordinary film: 15 of 32 for the new unit, 23 for the paper's.
- Top-50 peak tokens: 'ace' ×8, 'face' ×4, ' side' ×2, ' flips' ×2, ' on' ×2, ' playing' ×2.
- Agent 1 (blind to the label, rates it mixed, medium confidence) explains it as: *This unit fires on the word that picks out one of the two sides of a two-sided object, above all Two-Face's coin: 'scratched side up', 'the unscarred side', 'one side ... the other', 'two-headed', 'flips the coin', 'the flip of a coin'; among the top 50 the dominant tokens are 'ace'/'face' from 'Two-Face'. The opposite pole is very weak (max 1.7) and scattered across lunar place names (Tycho, Clavius).*
- Judge: **direct**. It describes the two sides of a two-sided object, above all Two-Face's coin and its flip, which is the labelled object itself. Label the explanation supports: *The two sides of a flipped coin*
- Agent 2 blind maximum: 6.91 against corpus max 8.25 (weakest of the top 8: 1.12; peak ' side'): The scarred district attorney flips his two-headed coin, one side scratched and the other clean, and lets it land scratched side up to decide who lives.
- Agent 2 examples firing strongly (≥ 50% of max): 5 of 5; near misses: 0 of 5.
- Steering: α=8.25: none; α=8.25: none; α=8.25: none; α=16.5: none; α=16.5: none; α=16.5: none
- **Verdict:** Concept unit. The two sides of a two-sided object and its flip; all five examples fire, no near miss; present in 3 Batman works against 9 for the paper's unit, which also fires on the name Two-Face. The paper's unit remains the stronger detector of this particular meme, since its blind maximum beat its corpus maximum in the audit.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 8.25 ' side' · *Arkham Asylum (comic)*: Two-Face declares that they will kill Batman if the coin lands scratched side up and let him go if the unscarred side lands.
- 7.56 ' other' · *Raiders of the Lost Ark*: An imam deciphers the medallion for Jones, revealing that one side bears a warning against disturbing the Ark and the other—which the Nazis do not possess—cont…
- 5.41 ' side' · *Arkham Asylum (comic)*: Two-Face looks at the coin and it is revealed that it actually landed scratched side up, implying that he made his own choice.
- 1.98 ' flips' · *Arkham Asylum (comic)*: He flips the coin and declares Batman free.
- 1.84 '-headed' · *The Dark Knight*: Dent defers his decision-making to his now half-scarred, two-headed coin, killing the corrupt officers and the mafia involved in Rachel's death.
- 1.57 ' on' · *The Exorcist*: He has a linguist analyze the language on the tape, who concludes that it is perfect English being spoken backward.
- 1.30 ' a' · *Batman Forever*: The trauma of the incident caused Dent to develop a split personality, make decisions based on the flip of a coin, and swear vengeance against Batman for faili…
- 1.12 ' open' · *Knightfall*: Instead of a crossover with a definite ending, the publishers treated the scenario as though it were the new status quo, leaving it open-ended.

Its negative pole L12·4733+ (max 1.68):

- 1.68 'cho' · *2001: A Space Odyssey*: His mission is to investigate a recently found artefact, a monolith buried four million years earlier near the lunar crater Tycho.
- 1.05 'us' · *2001: A Space Odyssey*: Millions of years later, Dr Heywood Floyd, Chairman of the United States National Council of Astronautics, travels to Clavius Base, an American lunar outpost.
- 1.03 ' of' · *Batman Forever*: Flashbacks reveal that Dent, once one of Batman's biggest supporters, was horribly disfigured with acid on one side of his face by mobster Sal Maroni.
- 0.98 ' on' · *Rear Window*: His mid-floor rear window looks onto a courtyard with small garden plots, surrounded on four sides by apartments in adjoining buildings.
- 0.94 'us' · *2001: A Space Odyssey*: During a stopover at Space Station Five, he meets Russian scientists who are concerned that Clavius seems to be unresponsive.

Agent 2's examples and near misses (activation, share of corpus max):

- example 9.81 (119%) ' side': The disfigured prosecutor flips a coin to decide the hostage's fate, and it lands on the scarred side.
- example 12.31 (149%) ' side': The gangster carries a two-headed silver dollar, one side burnt and the other clean, and leaves every choice to the flip of a coin.
- example 4.44 (54%) ' side': At the ranch the sheriff calls heads, but the drifter's coin comes up tails, the unscarred side glinting in the dust.
- example 8.12 (98%) ' side': The hitman tosses his lucky coin before every job and only shoots if the scratched side lands up.
- example 10.50 (127%) ' side': The twin brothers settle the inheritance with a coin flip, the good side for one and the scarred side for the other.
- near miss 0.00 (0%) None: Harvey Dent stands at Rachel's side during the press conference and promises Gotham a brighter future.
- near miss 2.33 (28%) ' over': The Joker flips the truck end over end on Lower Fifth Street while Batman swerves on the Batpod.
- near miss 0.05 (1%) ' the': The Penguin dumps a sack of gold coins on the table and counts them twice before the deal.
- near miss 0.00 (0%) None: Catwoman slashes the Penguin's face with her claws and escapes across the rooftops of Gotham.
- near miss 0.29 (4%) ' turns': The ace of spades turns up in the Riddler's deck and Harvey Dent wins the hand.

Steered generations:

- α=8.25 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is carrying a backpack and a small bag. He is wearing a red shirt. He is walking towards a cliff. He looks around, then looks at the camera. He
- α=8.25 · Plot: In the first act, → the story introduces the main characters and the setting. The protagonist, a young woman named Emily, is struggling to find her place in the world. She is an aspiring writer who has been rejected mul…
- α=8.25 · The story begins when → the narrator, a young man, is caught in a storm in the mountains. He takes shelter in a cabin. He finds a journal in the cabin, which is written by a man named Victor. Victor's journal describes his…
- α=16.5 · Plot summary: The film opens as → a young man, named Jack, is walking through the woods. He is carrying a backpack with a sleeping bag inside. He is walking in the woods, and he is not wearing a shirt. He is walking in the woods, and…
- α=16.5 · Plot: In the first act, → the story introduces the main characters and the setting. The protagonist, a young woman named Emily, is struggling to find her place in the world. She has just moved to a new city for a fresh start,…
- α=16.5 · The story begins when → the narrator is 12 years old. He is a student at a small school in a small town. He is very good at math. He is also very good at sports. He is very popular with his classmates. He is the

### 21. Harvey Dent

**Paper's unit L5·6648+** (layer 5): dev 0/10, test 0/6 (AUC 0.5); gate-open corpus max 10.375, present in 9 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Harvey' ×11, 'arker' ×6, 'ap' ×4, ' observes' ×3, 'arte' ×2, 'March' ×2.

**New unit L17·7382-** (layer 17): 3128 units passed the dev bar; this one scored best. Dev 9/10 (mean pos 4.233, neg 0.575); test 6/6 (AUC 0.944; mean pos 3.628, neg 0.917); p99 on the other memes' sentences 2.969 (max 6.188). Corpus max 7.0, p99 1.805; present in 9 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 5, only new 4 (Batman '89, Batman Returns, Mask of the Phantasm, The Killing Joke), only paper 4 (Dark Knight Rises, Earth One, Gotham by Gaslight, Year One). Batman works scoring above the median ordinary film: 24 of 32 for the new unit, 15 for the paper's.
- Top-50 peak tokens: 'figure' ×3, 'fig' ×3, ' face' ×3, 'red' ×3, 'ous' ×2, 'ly' ×2.
- Agent 1 (blind to the label, rates it conceptual, high confidence) explains it as: *This unit fires on physical disfigurement and deformity, especially of the face: 'disfigurements', 'disfiguring half of Dent's face', 'disfigurement', 'malformed', 'burning Dent's face', 'hideous appearance', and 'plastic surgery' to repair it. The opposite pole is weak and loosely about trance-like or psychic mental states.*
- Judge: **partial**. It captures facial disfigurement, including Dent's burned face, the character's defining feature, but nothing of the prosecutor, the coin or the split personality. Label the explanation supports: *Facial disfigurement and deformity*
- Agent 2 blind maximum: 6.62 against corpus max 7.00 (weakest of the top 8: 4.56; peak ' plastic'): The acid burns away half of the prosecutor's face, leaving a malformed, hideous disfigurement that no plastic surgery can ever repair.
- Agent 2 examples firing strongly (≥ 50% of max): 2 of 5; near misses: 0 of 5.
- Steering: α=7.0: none; α=7.0: none; α=7.0: none; α=14.0: none; α=14.0: weak The antidote (a skilled but unorthodox healer); α=14.0: none
- **Verdict:** Partial. Facial disfigurement, the character's defining feature; held-out 6 of 6, present in 9 Batman works, the same count as the paper's name unit but in different works. Two of five examples fire.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 7.00 'figure' · *Batman '89*: Although presumed dead, Napier survives with various disfigurements including chalk white skin and emerald-green hair and nails.
- 6.66 'fig' · *The Long Halloween*: During the trial, he throws a vial of acid—secretly given to him earlier by Vernon—at Dent, disfiguring half of Dent's face.
- 5.50 ' malformed' · *Batman Returns*: In Gotham City, two wealthy socialites, horrified by the birth of their malformed and feral son Oswald, abandon him in the sewers, where a colony of penguins t…
- 5.41 'figure' · *Mask of the Phantasm*: Bruce later examines an old photograph of the mobsters and recognizes that Valestra's enforcer was the Joker before his disfigurement.
- 5.38 ' surgery' · *Dark Knight Returns*: Previously known as Two-Face, Dent underwent extensive therapy and plastic surgery to reenter society before disappearing.
- 5.31 ' face' · *The Dark Knight*: The explosives detonate, killing Rachel and severely burning Dent's face on one side.
- 4.75 'figure' · *The Killing Joke*: His disfigurement, compounded by the loss of his family, drives him completely insane and marks the birth of the Joker.
- 4.56 'ous' · *Batman '89*: Driven insane by his hideous appearance, Napier, now calling himself "the Joker", kills Grissom, massacres Grissom's associates, and takes over his operations.

Its negative pole L17·7382+ (max 1.98):

- 1.98 ' Lucy' · *Dracula*: Lucy Westenra's letter to her best friend, Harker's fiancée Mina Murray, describes her marriage proposals from Dr John Seward, Quincey Morris, and Arthur Holmw…
- 1.98 'walk' · *Dracula*: Lucy begins to sleepwalk, unaware that the Count is influencing her mind.
- 1.80 ' cane' · *Daredevil*: Afterward, Urich deduces that Matt is Daredevil after realizing that he disguises his baton as a white cane.
- 1.80 'an' · *The Exorcist*: The demon says it will remain in Regan until she is dead.
- 1.60 ' shock' · *Jaws*: The shark enters a nearby lagoon, killing a boater and nearly killing Brody's son, Michael, who is hospitalized with shock.

Agent 2's examples and near misses (activation, share of corpus max):

- example 4.50 (64%) ' deform': A fire in the opera house leaves the composer's face horribly scarred, and he hides the deformity behind a mask.
- example 3.48 (50%) ',': Born with a malformed skull, the boy is paraded as a freak until a surgeon takes pity on him.
- example 4.97 (71%) ' plastic': After the chemical spill disfigures him, the gangster undergoes botched plastic surgery that twists his mouth into a permanent grin.
- example 2.50 (36%) ' face': The soldier returns from the front with half his face burned away and is shunned by his hometown.
- example 2.97 (42%) ' facial': The heiress is left with a hideous facial scar after the car crash, and her fiancé cannot bear to look at her.
- near miss 0.25 (4%) ' press': Harvey Dent faces the press outside the courthouse, half the reporters shouting questions at once.
- near miss 0.20 (3%) ' risky': The heart surgeon performs a risky operation while the hospital burns around her.
- near miss 0.16 (2%) ' a': Under hypnosis, the witness slips into a trance and describes the psychic visions that haunt her.
- near miss 0.77 (11%) ' a': The appearance of a mysterious stranger at the party sets the guests whispering about his hidden fortune.
- near miss 0.95 (14%) ' plastic': The lawyer's plastic briefcase melts in the burning car, destroying the evidence against the mob.

Steered generations:

- α=7.0 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is trying to find his way out. He is very confused. He is trying to find his way to the village. He is very tired. He is trying to find
- α=7.0 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elara, who lives in a small coastal village. She is an aspiring painter, but her family expects her to take over the family business, a fish…
- α=7.0 · The story begins when → the narrator, a young man, is caught in a storm in the village of Elsinore. He is told by a peasant that he is in the kingdom of Denmark. He is then taken to the castle of the king, who is
- α=14.0 · Plot summary: The film opens as → a young man, who is in the process of being transformed into a vampire. He is being transformed by a man who is a vampire hunter. The man is trying to kill the young man, but the young man is trying…
- α=14.0 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elara, who is a skilled but unorthodox healer. She lives in a small village where the people are wary of her because of her unconventional me…
- α=14.0 · The story begins when → the narrator, a young man, is caught in a sudden storm. He takes shelter in a small, remote inn. The innkeeper, who is also the only other person there, is initially reluctant to let him in. But the…

### 22. Selina Kyle

**Paper's unit L1·1051+** (layer 1): dev 0/10, test 0/6 (AUC 0.5); gate-open corpus max 2.547, present in 7 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Sel' ×29, 'el' ×2, 'els' ×1.

**New unit L32·5618+** (layer 32): 5033 units passed the dev bar; this one scored best. Dev 9/10 (mean pos 3.763, neg 0.181); test 4/6 (AUC 0.694; mean pos 1.732, neg 0.319); p99 on the other memes' sentences 1.802 (max 4.531). Corpus max 16.75, p99 5.126; present in 0 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 0, only new 0 (), only paper 7 (Batman Returns, Dark Knight Rises, Gotham by Gaslight, Hush, The Batman, The Long Halloween, Year One). Batman works scoring above the median ordinary film: 19 of 32 for the new unit, 7 for the paper's.
- Top-50 peak tokens: ' Rem' ×2, ' The' ×2, ' ab' ×2, ' Ren' ×2, ' and' ×2, ' Gro' ×2.
- Agent 1 (blind to the label, rates it mixed, medium confidence) explains it as: *This unit fires on the first sub-word fragment of a word the tokenizer has split in two — em\|bezzled, sw\|indled, Rem\|inded, pil\|oting, Mur\|dock, ab\|ducts, re\|venge, Gro.., Ren.. — and on 'The' opening a multi-word title, i.e. on a token that is incomplete and must be continued on the next token. The much stronger opposite pole fires on complete tokens at word and clause boundaries (by, dollars, has, that, a comma, an opening quote), so the distinction is 'word still in progress' as opposed to 'word finished, boundary reached'. The firings share no meaning; they share a tokenization situation.*
- Judge: **none**. It describes a tokenisation situation, the first fragment of a split word versus a finished word boundary, with no meaning shared across firings and nothing of the character. Label the explanation supports: *Incomplete word-piece awaiting continuation*
- Agent 2 blind maximum: 2.28 against corpus max 16.75 (weakest of the top 8: 9.88; peak ' sw'): The embezzler, reminded of his swindled accomplices, abducts the pilot's daughter, seeking revenge aboard a hydrofoil bound for the archipelago.
- Agent 2 examples firing strongly (≥ 50% of max): 0 of 5; near misses: 0 of 5.
- Steering: α=16.75: none; α=16.75: none; α=16.75: weak Heir to a dead father's fortune (the king is dead. The king was the father); α=33.5: none; α=33.5: none; α=33.5: weak Heir to a dead father's fortune (the king is dead. The king was the father)
- **Verdict:** Miss. The chosen unit is a tokenisation artefact (the first fragment of a split word); held-out 4 of 6. No name-free unit for the character found.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 16.75 ' em' · *Mask of the Phantasm*: Andrea admits to Bruce that her father worked for Valestra and was forced to take her into hiding in Europe because he was unable to repay money that he had em…
- 16.25 ' sw' · *Back to the Future*: Doc unveils a time machine built from a modified DeLorean, powered by plutonium he swindled from Libyan terrorists.
- 15.12 ' Rem' · *Get Out*: Reminded of his own mother's death, Chris guiltily takes her into the car, but she reawakens and attacks him, causing the car to crash and kill her.
- 11.06 ' pil' · *White Knight*: Napier, Batman and the GTO organize an assault on the Neo Joker's freeze ray, piloting a fleet of Batmobiles taken from Wayne Manor.
- 10.69 ' The' · *Singin' in the Rain*: In 1927, silent film stars Don Lockwood and Lina Lamont attend the premiere of their latest film, The Royal Rascal, produced by Hollywood studio Monumental Pic…
- 10.19 're' · *Sherlock Holmes*: The room is splashed in blood, and the word "RACHE" (German for "revenge") is written in it on one wall; however, Drebber's body shows no wounds.
- 10.12 ' Mur' · *Daredevil*: In 1982, young Matt Murdock is accidentally blinded by radioactive waste.
- 9.88 ' ab' · *The Silence of the Lambs*: When Buffalo Bill abducts Catherine Martin, daughter of U.S. Senator Ruth Martin, Crawford authorizes Starling to offer Lecter a fraudulent deal: a prison tran…

Its negative pole L32·5618- (max 34.00):

- 34.00 ' by' · *Spirited Away*: Chihiro soon forgets her real name, and Haku explains that Yubaba controls people by taking their names; if Chihiro completely forgets her name like Haku did,…
- 15.81 ' dollars' · *Gone Girl*: Nick discovers thousands of dollars' worth of items purchased with his credit card, unauthorized and hidden in his sister Margo's woodshed.
- 15.56 ' Sid' · *Toy Story*: Sid Phillips, Andy's sadistic next-door neighbor, grabs Woody and Buzz with the claw and takes them to his house, where they encounter his mutant toys, made fr…
- 15.56 ' has' · *Saving Private Ryan*: Miller muses that people often guessed his career before he became a soldier, while his men could not, implying that war has changed him, and worries whether h…
- 15.25 ',' · *Joker (2019)*: Budget cuts shut down the social services program, leaving Arthur without his medication.

Agent 2's examples and near misses (activation, share of corpus max):

- example 2.34 (14%) ' em': A disgraced accountant embezzles millions, swindles his partners and absconds to a remote archipelago.
- example 0.00 (0%) None: The kidnapper abducts a cartographer's niece, demanding a ransom of uncut sapphires and unmarked bullion.
- example 3.45 (21%) 'il': Piloting a dilapidated seaplane, the smuggler outmanoeuvres the coastguard and vanishes into the fjords.
- example 0.00 (0%) None: Reminded of her murdered brother, the vigilante swears revenge on the embezzling tycoon who orchestrated it.
- example 0.00 (0%) None: An ornithologist stumbles upon a counterfeiting syndicate operating from a decommissioned lighthouse.
- near miss 0.00 (0%) None: He has that, by a thousand dollars, and she said, "No, not by a lot."
- near miss 0.00 (0%) None: A man has a dog, and the dog has a ball, and that is all there is to it.
- near miss 0.00 (0%) None: She got the job, so he paid her ten dollars a day, which was more than she had.
- near miss 1.26 (8%) ' em': His ab workout is done by six, and the em dash in his note was a joke.
- near miss 7.84 (47%) ' Gro': Sam, Ren and Gro meet at the bar, and by ten all three have gone home.

Steered generations:

- α=16.75 · Plot summary: The film opens as → a young girl, Lina, is playing with her toys. She is told by her mother that they are moving to a new house. Lina is upset, but her mother comforts her by saying that the new house is bigger and bett…
- α=16.75 · Plot: In the first act, → the story introduces the protagonist, a young woman named Clara who is struggling to find her place in the world. She is living in a small town where everyone knows each other, and she feels like an…
- α=16.75 · The story begins when → the narrator, a young man, is caught in a storm in the village of Elsinore. He is told by a peasant that the king is dead. The king was the father of the prince, who is now the ruler of Denmark
- α=33.5 · Plot summary: The film opens as → a young girl, Lina, is playing with her toys. She is told by her mother that they are moving to a new house. Lina is upset, but her mother reassures her that the new house will be better. L
- α=33.5 · Plot: In the first act, → the story introduces the protagonist, a young woman named Clara who is struggling to find her place in the world. She is an aspiring writer, but her work is not being recognized. She lives in a small…
- α=33.5 · The story begins when → the narrator, a young man, is caught in a storm in the village of Elsinore. He is told by a peasant that the king is dead. The king was the father of the prince, who is now the ruler of Denmark

### 23. The Riddler

**Paper's unit L1·10818-** (layer 1): dev 4/10, test 2/6 (AUC 0.611); gate-open corpus max 11.25, present in 10 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: 'ler' ×27, ' Rock' ×3, 'iddles' ×1, ' ways' ×1, 'lers' ×1.

**New unit L12·11755-** (layer 12): 4286 units passed the dev bar; this one scored best. Dev 9/10 (mean pos 2.146, neg 0.065); test 6/6 (AUC 0.972; mean pos 0.955, neg 0.073); p99 on the other memes' sentences 0.656 (max 1.961). Corpus max 3.516, p99 0.377; present in 1 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 1, only new 0 (), only paper 9 (Arkham Asylum (game), Arkham City, Arkham Knight, Batman '66, Earth One, Hush, Knightfall, The Batman). Batman works scoring above the median ordinary film: 21 of 32 for the new unit, 10 for the paper's.
- Top-50 peak tokens: 'idd' ×5, ' the' ×3, ',' ×3, ' answers' ×2, 'iddle' ×2, ' to' ×2.
- Agent 1 (blind to the label, rates it mixed, medium confidence) explains it as: *This unit fires on riddles, quiz or test questions and the act of answering them correctly: 'correctly answers the tiebreaker question', 'answer three riddles', 'answers correctly', 'the Riddler reveals his final riddle', 'Yubaba tests Chihiro, asking her to [identify]', 'scores ... 179 on the LSAT'. The 'riddle' stem dominates the top 50, but the strongest firing is on 'answers'. The opposite pole fires on theft: 'rob', 'thieves', 'thief', 'sack', 'hide and seek'.*
- Judge: **partial**. It captures riddles and test questions being answered correctly, the Riddler's defining device, but is broader (LSAT, Yubaba's test) and about answering rather than the villain who poses them. Label the explanation supports: *Riddles and test questions answered correctly*
- Agent 2 blind maximum: 3.62 against corpus max 3.52 (weakest of the top 8: 0.93; peak ' answers'): To free the hostages the detective must answer three riddles, and when he correctly answers the final riddle the question-obsessed villain screams in fury.
- Agent 2 examples firing strongly (≥ 50% of max): 3 of 5; near misses: 0 of 5.
- Steering: α=3.52: none; α=3.52: none; α=3.52: none; α=7.03: none; α=7.03: none; α=7.03: none
- **Verdict:** Partial. Riddles and test questions answered correctly: the villain's device, broader than the villain. Three of five examples fire, no near miss.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 3.52 ' answers' · *Mean Girls*: At the state finals, she correctly answers the tiebreaker question, winning the championship.
- 1.73 'iddles' · *The Batman*: When Batman arrives, the Riddler calls him via a phone duct-taped to Colson and threatens to detonate the bomb if Colson cannot answer three riddles.
- 1.58 ' correctly' · *Spirited Away*: After she answers correctly that none of the pigs are her parents, her contract disappears and she is given back her real name.
- 1.55 'iddle' · *Batman Forever*: The Riddler reveals his final riddle: Chase and Robin, representing the two sides of Batman's personality, are trapped in tubes above a deadly drop, and he onl…
- 1.34 'iddle' · *Batman Forever*: Batman distracts the Riddler with a riddle himself, before destroying the Riddler's brainwave receiver with a Batarang, damaging the Riddler's mind and enablin…
- 1.27 ' to' · *Spirited Away*: When they arrive at the bathhouse, Yubaba tests Chihiro, asking her to identify her parents among a group of pigs.
- 1.03 'son' · *Sherlock Holmes*: Inspectors Gregson and Lestrade are already on the scene.
- 0.93 '7' · *Legally Blonde*: After months of studying, Elle scores a near-perfect 179 on the LSAT.

Its negative pole L12·11755+ (max 1.79):

- 1.79 ' rob' · *The Dark Knight*: A gang of masked criminals rob a mafia-owned bank in Gotham City, betraying and killing each other until the sole survivor, the Joker, reveals himself as the m…
- 1.34 ' hide' · *Arkham Asylum (comic)*: The Joker forces Batman into a game of hide and seek, giving him one hour to escape Arkham before his adversaries are sent to hunt him down.
- 1.16 ' sack' · *Puss in Boots*: The youngest son laments his poor inheritance, but the cat, who turns out to be extraordinarily clever and resourceful, assures his master that he can bring hi…
- 1.05 ' thieves' · *Inception*: Dom Cobb and Arthur are professional thieves who perform corporate espionage using experimental dream-sharing technology to infiltrate their targets' subconsci…
- 0.94 ' of' · *The Bat (1926)*: A circular light shines on the wall, with the shadow of a bat in its center, but after investigating, the group finds that the shape was caused by a miller mot…

Agent 2's examples and near misses (activation, share of corpus max):

- example 6.25 (178%) ' answers': A slum kid correctly answers every question on the television quiz show and is accused of cheating.
- example 3.58 (102%) ' correctly': The sphinx poses a riddle at the bridge, and only the youngest brother answers it correctly and crosses.
- example 1.75 (50%) 'iddle': The green-suited villain leaves a riddle at each crime scene, daring the caped vigilante to solve them before midnight.
- example 2.00 (57%) ' asking': The witch tests the girl by asking her to identify her parents among a herd of identical pigs.
- example 1.55 (44%) ' the': A sorority girl scores a near-perfect mark on the law school admission test and wins a place against the odds.
- near miss 1.20 (34%) ' the': Alfred answers the door at Wayne Manor and shows the commissioner into the study.
- near miss 0.07 (2%) ' Joker': The Joker's getaway van is riddled with bullets as it tears away from the Gotham National Bank.
- near miss 0.15 (4%) ' a': Lucius Fox takes the new Tumbler for a test drive through the tunnels under Wayne Tower.
- near miss 0.00 (0%) None: The thieves rob the Gotham museum and hide the stolen jewels in the Penguin's sewer lair.
- near miss 0.61 (17%) ' the': Rachel questions Harvey's loyalty after he answers the press with a lie about the Batman's identity.

Steered generations:

- α=3.52 · Plot summary: The film opens as → a young girl, Lina, is playing with her toys. She is told by her mother that she is going to be a big sister. She is excited. She is then shown playing with her toys, pretending to be a mother.
- α=3.52 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elara, who lives in a small village. She is curious and adventurous, often exploring the nearby forest. One day, she discovers a hidden cave…
- α=3.52 · The story begins when → the narrator, a young man, is caught in a storm in the mountains. He takes shelter in a cave. He finds a strange object in the cave, which turns out to be a box containing a small, living creature. T…
- α=7.03 · Plot summary: The film opens as → a young girl, named Lina, is playing with her toys. She is told by her mother that she is going to be a mother soon. Lina is confused, but her mother explains that she is going to have a baby.
- α=7.03 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elara, who lives in a small village. She is curious and adventurous, often exploring the nearby forest. One day, she discovers an ancient, gl…
- α=7.03 · The story begins when → the narrator, a young man, is caught in a storm in the mountains. He finds shelter in a cave. There he meets a strange old man. The old man is very strange. He has a long beard, and he is dressed

### 24. Jason Todd

**Paper's unit L4·4504-** (layer 4): dev 0/10, test 0/6 (AUC 0.5); gate-open corpus max 20.25, present in 8 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Todd' ×8, 'roy' ×2, ' trolls' ×2, 'ht' ×2, 'dd' ×2.

**New unit L11·823-** (layer 11): 3599 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 2.008, neg 0.265); test 6/6 (AUC 0.917; mean pos 1.152, neg 0.267); p99 on the other memes' sentences 2.186 (max 4.688). Corpus max 7.469, p99 3.25; present in 0 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 0, only new 0 (), only paper 8 (A Death in the Family, Arkham City, Arkham Knight, Dark Knight Returns, Hush, Knightfall, Under the Red Hood, White Knight). Batman works scoring above the median ordinary film: 25 of 32 for the new unit, 9 for the paper's.
- Top-50 peak tokens: ' killed' ×7, ' dead' ×6, ' body' ×3, ' corpse' ×3, ' heart' ×3, ' to' ×3.
- Agent 1 (blind to the label, rates it conceptual, high confidence) explains it as: *This unit fires on the word that names a dead or mutilated body or completes the act of killing — corpse, body, severed head, skinned face, slit throat, removes their skin, taking his life, killed, dead — when a plot describes a violent death or what is done to the victim's remains. The opposite pole is weak and points at the aftermath and handling of death (feeding a body into a woodchipper, an inquest ruling, searching dead soldiers' dog tags, boxes of earth) rather than the killing or the corpse itself.*
- Judge: **partial**. It captures violent death and what is done to the corpse, which is the event that defines Jason Todd, but it is generic to any killing and says nothing of a sidekick or protege. Label the explanation supports: *Violent death and the mutilated corpse*
- Agent 2 blind maximum: 10.19 against corpus max 7.47 (weakest of the top 8: 4.91; peak ' skins'): The killer slits the guard's throat, severs his head, skins his face and leaves the mutilated corpse sprawled across the floor.
- Agent 2 examples firing strongly (≥ 50% of max): 3 of 5; near misses: 0 of 5.
- Steering: α=7.47: none; α=7.47: none; α=7.47: none; α=14.94: none; α=14.94: none; α=14.94: none
- **Verdict:** Partial. Violent death and the mutilated corpse: the event that defines the character, generic to any killing. Held-out 6 of 6.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 7.47 'inned' · *The Silence of the Lambs*: That night, he uses a piece of Chilton's pen to unlock his restraints, murders two guards, and escapes in an ambulance disguised in a dead guard's uniform and…
- 5.81 ' head' · *The Godfather*: Hagen departs and the next morning, Woltz wakes to find the severed head of his prized $600,000 racehorse Khartoum in his bed with him.
- 5.56 ' life' · *Dark Knight Returns*: Batman paralyzes Joker but stops short of taking his life.
- 5.22 ' slit' · *Knives Out*: The next morning, Harlan's housekeeper Fran discovers him dead with a slit throat.
- 5.12 ' body' · *Spider-Man*: Peter takes Norman's body to the Osborn house, where Harry confronts him, but Peter escapes.
- 5.03 ' corpse' · *Die Hard*: Realizing Powell is about to leave, having found nothing amiss, McClane drops a terrorist's corpse onto his car.
- 4.91 ' skin' · *The Silence of the Lambs*: Officially conducting a psychological survey, Starling is actually meant to draw out insights that could help capture "Buffalo Bill", a serial killer who murde…
- 4.91 'its' · *Gone Girl*: She seduces Desi, slits his throat during sex, and returns home, clearing Nick of suspicion.

Its negative pole L11·823+ (max 4.12):

- 4.12 'chip' · *Fargo*: She sees the Ciera, then discovers Gaear feeding Carl's body into a woodchipper.
- 2.47 ' residual' · *The Shining*: He tells Danny that the hotel also "shines" due to its residual and unpleasant history and warns him to avoid room 237.
- 2.00 ' a' · *Vertigo*: An inquest into Madeleine's death declares it a suicide, though the coroner rebukes Scottie for not doing more to save her.
- 1.95 ' out' · *The Bat (1926)*: The candles in the ballroom go out when lit, and a shape appears to float towards Anderson and Lizzie, but it turns out to be the Japanese butler Billy carryin…
- 1.91 ' of' · *Saving Private Ryan*: Wade admonishes Reiben, Mellish, and Jackson for callously searching through a pile of deceased soldiers' dog tags in front of passing troops, hoping to find R…

Agent 2's examples and near misses (activation, share of corpus max):

- example 8.44 (113%) ' slit': The detectives find the missing banker's corpse in the river, his throat slit and his hands removed.
- example 4.16 (56%) ' body': The clown murders the mayor on live television, leaving the body slumped over the desk for the cameras.
- example 6.56 (88%) ' skins': The hunter is killed by the creature, which skins its victims and wears their faces as masks.
- example 1.71 (23%) ' head': A farmer discovers a severed head in his cornfield and the sheriff realizes a serial killer has returned.
- example 3.64 (49%) ' body': The assassin takes the senator's life with a single shot, and his dead body is found at dawn.
- near miss 0.04 (1%) ' late': The governing body of the league rules that the striker's late goal stands, and the fans celebrate.
- near miss 1.10 (15%) ' dead': The road is a dead end, so the driver kills the engine and the headlights and waits for morning.
- near miss 2.48 (33%) ' the': The head of the department skins the budget to the bone and the junior staff lose their desks.
- near miss 0.54 (7%) ' fallen': The widow sorts through the dog tags of fallen soldiers while the inquest rules the fire accidental.
- near miss 0.00 (0%) None: The comedian kills at the open mic, and the crowd's laughter nearly brings the house down.

Steered generations:

- α=7.47 · Plot summary: The film opens as → a young girl, Lina, is playing with her dog in the woods. She is followed by a man, who is not a friend. He is a hunter. He kills the dog, then takes Lina. She is taken to
- α=7.47 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is struggling to find her place in the world. She is living in a small town, and feels trapped by the monotony of her life. One da…
- α=7.47 · The story begins when → the narrator, a young man, is walking through the forest. He is alone, and he feels a sense of foreboding. He hears a sound, and then sees a man in the distance. The man is dressed in a strange
- α=14.94 · Plot summary: The film opens as → a young girl, named Lila, is found dead in the woods. She is the daughter of a local fisherman, who is also found dead. The two are found in a small village, which is called El Pueblo. The
- α=14.94 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elara, who lives in a small village. She is a skilled hunter and tracker, known for her ability to find the truth in the wilderness. One day,…
- α=14.94 · The story begins when → the narrator, a young man, is walking through the forest. He is followed by a dog. The dog is not a normal dog. It is a very old dog. The dog is not a normal dog. It is a very old

### 25. Ra's al Ghul

**Paper's unit L4·9760-** (layer 4): dev 0/10, test 0/6 (AUC 0.5); gate-open corpus max 9.375, present in 8 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: 'ul' ×17, 'ar' ×2, 'ag' ×2, 'ug' ×1, 'ig' ×1.

**New unit L8·8851-** (layer 8): 4394 units passed the dev bar; this one scored best. Dev 9/10 (mean pos 5.705, neg 0.535); test 6/6 (AUC 1.0; mean pos 6.068, neg 0.202); p99 on the other memes' sentences 1.575 (max 4.969). Corpus max 9.188, p99 1.016; present in 2 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 2, only new 0 (), only paper 6 (Arkham Knight, Batman Begins, Batman R.I.P., Dark Knight Rises, Hush, Under the Red Hood). Batman works scoring above the median ordinary film: 18 of 32 for the new unit, 9 for the paper's.
- Top-50 peak tokens: ' death' ×5, ' on' ×3, 'ally' ×3, 'urance' ×3, 'ible' ×2, ' die' ×2.
- Agent 1 (blind to the label, rates it mixed, medium confidence) explains it as: *This unit fires on words meaning immortality, permanence, or not dying — immortality, Immortan, indestructible, eternal, everlasting, 'will never die', endurance, death — the token that asserts something lasts forever or cannot be killed. It also fires more weakly on the bare prefix 'imm' even when the word turns out to be immaturity or immersing, so part of the response is a lexical anticipation of 'immortal'. The opposite pole fires on 'life' in the sense of a person's ordinary life, living things, and life support, so the distinction is deathlessness as opposed to mortal life.*
- Judge: **partial**. It captures immortality and not dying, the defining feature of the character, but nothing of the assassin league, the pits or the mentor role. Label the explanation supports: *Immortality and deathlessness*
- Agent 2 blind maximum: 11.94 against corpus max 9.19 (weakest of the top 8: 3.09; peak ' die'): Drinking from the sacred spring, the ancient warlord becomes immortal, an indestructible and everlasting being who will never die for all eternity.
- Agent 2 examples firing strongly (≥ 50% of max): 5 of 5; near misses: 0 of 5.
- Steering: α=9.19: weak A mutating super-formula (more powerful, but he also becomes more dangerou…; α=9.19: none; α=9.19: none; α=18.38: weak Ra's al Ghul (He is born again and again) (concept); α=18.38: weak Superhero powers (peaks on Superman) (gods are immortal, but they are not…; α=18.38: none
- **Verdict:** Partial. Immortality and deathlessness; all five examples fire, no near miss, and steering once produced 'he is born again and again'. The defining feature has a unit; the character does not.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 9.19 'ort' · *Mad Max: Fury Road*: Max Rockatansky is captured by hydraulic despot cult leader Immortan Joe's War Boys and taken to his fortress called the Citadel.
- 9.06 'ortality' · *Arkham City*: Before Joker can take advantage of the situation, Talia arrives and offers him immortality in exchange for sparing Batman's life.
- 6.84 'ible' · *Arkham Asylum (game)*: Young learned that Joker had been secretly funding her research to create an army of indestructible and savage superhuman monsters; her refusal to hand over th…
- 6.03 ' eternal' · *Frozen*: She builds an ice palace and decides to live a hermit's life, unaware that her magic has plunged Arendelle into an eternal winter.
- 5.75 ' die' · *Batman R.I.P.*: The story begins with Batman and Robin perched onto a gothic building lit by lightning, where the former defiantly proclaims, "Batman and Robin will never die!…
- 5.09 ' everlasting' · *The Killing Joke*: He offers to help the Joker rehabilitate in order to end their everlasting war, which Batman fears may one day result in a fight to the death.
- 3.44 ' imm' · *Home Alone*: Their youngest son, Kevin, is the subject of ridicule by his older siblings because of his immaturity.
- 3.09 ' imm' · *Batman Begins*: After confronting Falcone, who says real power comes from being feared, Bruce spends the next seven years traveling the world, training in combat, and immersin…

Its negative pole L8·8851+ (max 3.41):

- 3.41 ' life' · *Lego Batman*: During his latest mission to stop the Joker and his cronies from destroying the city, Batman succeeds, but additionally hurts Joker's feelings when he deems hi…
- 3.22 ' life' · *Knightfall*: Despite the advice of everyone in his life, including Dr. Kinsolving, Bruce refuses to rest and continues to pursue his self-imposed duty despite his worsening…
- 2.53 ' things' · *The Lion King*: Mufasa explains to Simba the responsibilities of kingship and the "circle of life," which connects all living things.
- 2.11 ' lives' · *The Martian*: Five years later, as the Ares V is launching, those involved in Watney's rescue are seen in their current lives watching the launch footage.
- 2.02 ' in' · *The Matrix*: Even so, they remain outmatched by the Agents—sentient programs that protect the system—and death in the Matrix means death in the real world.

Agent 2's examples and near misses (activation, share of corpus max):

- example 10.38 (113%) 'ortality': A thousand-year-old vampire, weary of his immortality, searches the city for someone who can finally kill him.
- example 10.50 (114%) 'less': The scientist's serum makes the soldier indestructible, and the army sends its deathless weapon into the war.
- example 14.31 (156%) 'ortality': The explorers seek the fountain whose waters grant eternal youth and immortality to anyone who drinks.
- example 14.81 (161%) ' immortal': The desert tyrant calls himself immortal and promises his war boys they will ride everlasting into the afterlife.
- example 8.81 (96%) 'ying': The curse means the pirate crew cannot die, doomed to sail forever as undying skeletons beneath the moon.
- near miss 0.91 (10%) ' alive': She devotes her life to nursing her father, who is kept alive on life support after the crash.
- near miss 3.41 (37%) ' mortal': A mortal wound takes the captain within the hour, and his short life ends on the muddy battlefield.
- near miss 0.31 (3%) ' the': The young recruit dies in his first battle, and his mother never recovers from the loss.
- near miss 0.00 (0%) None: The lifeguard saves a drowning child and returns to his ordinary life teaching biology at the high school.
- near miss 0.68 (7%) ' dies': The old gardener lives out his last summer tending the living things in the greenhouse before he dies.

Steered generations:

- α=9.19 · Plot summary: The film opens as → a young man, named Kip, is walking through the forest. He finds a strange, glowing stone. He takes it home, and it begins to affect him. He becomes more powerful, but he also becomes more dangerous.…
- α=9.19 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elara, who lives in a small village. One day, she discovers an ancient map that leads to a hidden treasure. As she follows the map, she encou…
- α=9.19 · The story begins when → the young girl, Lina, is playing with her friend, Kiki, in the forest. They find a magical tree with glowing leaves. As they touch the leaves, they are transported to a world where they can talk to a…
- α=18.38 · Plot summary: The film opens as → a young man, Kip, is born in the year 1000. He is a proud and powerful being, but he is not immortal. He is born again and again, but each time he is born, he is
- α=18.38 · Plot: In the first act, → the story introduces the world of the gods, the Olympians, and the mortal world. The gods are immortal, but they are not all-powerful. They are subject to the whims of fate, but they are not free. Th…
- α=18.38 · The story begins when → the sun sets on the city of New York. The sun sets on the city of New York. The sun sets on the city of New York. The sun sets on the city of New York. The sun sets on the city of New

### 26. Lucius Fox

**Paper's unit L5·10113-** (layer 5): dev 0/10, test 0/6 (AUC 0.333); gate-open corpus max 13.812, present in 14 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Quinn' ×18, ' Fox' ×12, 'orro' ×3, ' fo' ×3, 'iki' ×3, 'let' ×1.

**New unit L9·3350+** (layer 9): 5342 units passed the dev bar; this one scored best. Dev 9/10 (mean pos 1.629, neg 0.123); test 4/6 (AUC 0.653; mean pos 0.553, neg 0.169); p99 on the other memes' sentences 0.713 (max 1.07). Corpus max 3.047, p99 0.633; present in 4 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 3, only new 1 (Batman Forever), only paper 11 (Arkham Asylum (game), Arkham City, Batman Begins, Batman Ninja, Batman R.I.P., Dark Knight Rises, Earth One, Hush). Batman works scoring above the median ordinary film: 23 of 32 for the new unit, 23 for the paper's.
- Top-50 peak tokens: ' by' ×6, ' new' ×6, ' technology' ×4, ' of' ×3, ' advanced' ×3, '-tech' ×2.
- Agent 1 (blind to the label, rates it conceptual, medium confidence) explains it as: *This unit fires on words that describe advanced technology or a scientific breakthrough — technology, technological, high-tech, biotechnology, breakthrough, advanced, new, genetic engineering — on the token that labels something as cutting-edge science (it even fires on 'cutting' in an unrelated sense, apparently anticipating 'cutting-edge'). The opposite pole, weaker, is specifically time-travel and futuristic science-fiction devices (time machine, time travel, the future, reprogrammed missiles, experimental dream-sharing), so the unit prefers generic present-day advanced tech over speculative sci-fi machinery.*
- Judge: **partial**. It captures cutting-edge technology and scientific breakthroughs, Lucius Fox's domain, but nothing of a person who supplies or runs it. Label the explanation supports: *Cutting-edge technology and breakthroughs*
- Agent 2 blind maximum: 7.84 against corpus max 3.05 (weakest of the top 8: 1.73; peak '-tech'): A young researcher unveils a breakthrough in advanced biotechnology, cutting-edge genetic engineering and high-tech new technology that stuns the scientific world.
- Agent 2 examples firing strongly (≥ 50% of max): 4 of 5; near misses: 0 of 5.
- Steering: α=3.05: none; α=3.05: none; α=3.05: none; α=6.09: none; α=6.09: none; α=6.09: none
- **Verdict:** Partial. Cutting-edge technology and breakthroughs, his domain; held-out only 4 of 6, four of five examples fire. The blind maximum is 2.6 times the corpus maximum, so the paper's corpus barely exercises this unit.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 3.05 ' technology' · *Batman Forever*: Bruce rejects the device as he is concerned that the technology could manipulate minds.
- 3.00 ' breakthrough' · *The Martian*: With this breakthrough, Henderson is finally allowed to inform Watney's crewmates.
- 2.47 ' technological' · *Dark Knight Returns*: Superman tries to reason with Batman, but Batman uses his technological inventions to fight him on equal ground.
- 2.42 ' of' · *Jurassic Park*: Malcolm warns of the implications of genetic engineering while Grant and Sattler express uncertainty over the ability of humans and dinosaurs to coexist.
- 2.31 '-tech' · *Ghostbusters*: They develop high-tech nuclear-powered equipment to capture and contain ghosts, although business is initially slow.
- 1.94 'otechnology' · *Return of the Joker*: This biotechnology allows him to survive death by taking over Tim's body permanently.
- 1.83 ' advanced' · *Under the Red Hood*: Batman stops an attempted theft of a shipment that belonged to Black Mask, which contains the advanced android Amazo.
- 1.73 ' cutting' · *Jurassic Park*: To access the embryo storage room, Nedry deactivates the park's security system, cutting power to the tour vehicles.

Its negative pole L9·3350- (max 1.50):

- 1.50 'program' · *Superman*: Criminal mastermind Lex Luthor discovers a joint U.S. Army/U.S. Navy missile test and plots to target the San Andreas Fault with reprogrammed missiles, though…
- 1.38 ' time' · *Back to the Future*: After Doc inputs a destination time of November 5, 1955 (the day he conceived his time travel invention), the terrorists arrive unexpectedly and gun him down.
- 1.35 ' time' · *Back to the Future*: Doc unveils a time machine built from a modified DeLorean, powered by plutonium he swindled from Libyan terrorists.
- 1.19 ' time' · *Back to the Future*: Doc explains the only source available in 1955 capable of generating the 1.21 gigawatts of power required for time travel is a lightning bolt.
- 1.14 ',' · *Batman Forever*: Edward Nygma, an eccentric and egotistical researcher at Wayne Enterprises, approaches his employer, Bruce Wayne, to present an invention that can beam televis…

Agent 2's examples and near misses (activation, share of corpus max):

- example 3.56 (117%) ' technology': A billionaire inventor builds a high-tech suit of armor using the most advanced technology his company can produce.
- example 1.13 (37%) ' in': A geneticist's breakthrough in genetic engineering lets a theme park resurrect creatures extinct for millions of years.
- example 5.28 (173%) '-edge': The start-up's new biotechnology promises to cure blindness, but its cutting-edge implant has a hidden flaw.
- example 3.41 (112%) ' technological': A team of engineers races to perfect an advanced technological breakthrough before a rival corporation steals it.
- example 2.17 (71%) ' state': The caped vigilante's armorer equips him with state-of-the-art technology and a high-tech surveillance network.
- near miss 0.02 (1%) ' a': A teenager accidentally travels thirty years into the past in a time machine built from a sports car.
- near miss 0.46 (15%) ' breakthrough': In therapy the widow finally has a breakthrough and admits she blames herself for the accident.
- near miss 0.93 (30%) ' advanced': A woman of advanced age moves into a new house and befriends the boy next door.
- near miss 0.17 (6%) ' using': A thief enters people's dreams using an experimental dream-sharing device to steal corporate secrets.
- near miss 0.25 (8%) ' civil': A civil engineer in a small town fights to save the old bridge from being torn down.

Steered generations:

- α=3.05 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is carrying a backpack and seems to be searching for something. He finds a small, wooden box. He opens it and finds a letter inside. The lette…
- α=3.05 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is struggling to find her place in the world. She is an aspiring artist, but her family expects her to pursue a more stable career…
- α=3.05 · The story begins when → the narrator, a young man, is walking through the forest. He is alone, and he feels a bit uneasy. He hears a sound, and then sees a man in a dark cloak. The man is very pale, with a long
- α=6.09 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is carrying a backpack and seems to be in a hurry. He is followed by a dog. As he walks, he is talking to himself, saying things like "I
- α=6.09 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elara, who is a skilled alchemist. She lives in a world where magic and science coexist, but there are tensions between the two. Elara is det…
- α=6.09 · The story begins when → the narrator, a young man, is walking through the forest. He is alone, and the forest is dark and cold. He feels a bit scared, but he continues walking. Suddenly, he sees a light in the distance. He…

### 27. A corrupt police force

**Paper's unit L35·4074-** (layer 35): dev 5/10, test 1/6 (AUC 0.583); gate-open corpus max 40.0, present in 13 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Gotham' ×27, ' the' ×13, ' police' ×2, ' corrupt' ×2, 'N' ×1, 'd' ×1.

**New unit L15·5647+** (layer 15): 2956 units passed the dev bar; this one scored best. Dev 9/10 (mean pos 2.205, neg 0.304); test 5/6 (AUC 0.889; mean pos 1.862, neg 0.363); p99 on the other memes' sentences 1.543 (max 2.344). Corpus max 4.562, p99 2.188; present in 2 of 32 Batman works, 1 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 0, only new 2 (A Death in the Family, Batman R.I.P.), only paper 13 (Arkham Knight, Batman '89, Batman Begins, Batman Returns, Dark Knight Returns, Dark Knight Rises, Earth One, Gotham by Gaslight). Batman works scoring above the median ordinary film: 29 of 32 for the new unit, 29 for the paper's.
- Top-50 peak tokens: ',' ×4, ' him' ×3, ' with' ×3, ' to' ×2, ' Joker' ×2, ' receiving' ×1.
- Agent 1 (blind to the label, rates it conceptual, medium confidence) explains it as: *This unit fires mid-construction on the token where a corrupt or collusive deal is being specified — receiving (bribes), lets criminals do (their business), (hunting) rights in exchange for a murder, presenting the Ark to Hitler, they draft a denunciation, weapons delivered to the Ten Rings, blackmails her into giving him, an ally of Chudnofsky — when a character is complicit with or trading favors with villains. The opposite pole, which is the stronger side of this channel, fires on acting against wrongdoing or for others' good (crime-fighting, combat, protect, help patients, destroy the criminal element, loves him), so the unit encodes complicity with evil as opposed to opposing or protecting against it.*
- Judge: **partial**. It captures complicity and corrupt deals with villains, the defining idea of corruption, but no example involves police or any institution. Label the explanation supports: *Complicity and corrupt deals with villains*
- Agent 2 blind maximum: 2.53 against corpus max 4.56 (weakest of the top 8: 3.11; peak ' operate'): The corrupt police chief receives bribes from the cartel, lets its killers operate freely, and trades hunting rights in exchange for a murder.
- Agent 2 examples firing strongly (≥ 50% of max): 0 of 5; near misses: 0 of 5.
- Steering: α=4.56: none; α=4.56: none; α=4.56: none; α=9.12: none; α=9.12: none; α=9.12: none
- **Verdict:** Partial. Complicity and corrupt deals with villains; held-out 5 of 6 but Agent 2's examples do not fire (0 of 5), so the explanation loses. Weak unit (maximum 4.6).

Strongest corpus sentences of the new unit (gate-open, peak token):

- 4.56 ' receiving' · *The Shawshank Redemption*: In 1963, Norton begins exploiting prison labor for public works, profiting by undercutting skilled labor costs and receiving bribes.
- 4.12 ' do' · *Earth One*: Gordon, angry, but also terrified about what Bullock did, explains to him that he was once as fearless and resistant to Gotham's corruption as Bullock, but bec…
- 3.94 ' rights' · *The Lion King*: Scar visits the hyenas and convinces them to help him kill Mufasa and Simba in exchange for hunting rights in the Pride Lands.
- 3.50 ' Hitler' · *Raiders of the Lost Ark*: The vessel travels to an island in the Aegean Sea, where Belloq intends to test the power of the Ark before presenting it to Hitler.
- 3.45 ' they' · *Count of Monte Cristo*: He and Danglars meet to plan Dantès' downfall: they draft an anonymous letter denouncing Dantès as a Bonapartist.
- 3.36 ' to' · *Iron Man*: At a charity event, reporter Christine Everhart informs Stark that his company's weapons were recently delivered to the Ten Rings and are being used to attack…
- 3.34 ' him' · *A Death in the Family*: He blackmails her into giving him the medical supplies her agency has stockpiled in a warehouse.
- 3.11 ' Ch' · *Green Hornet*: Britt discovers that Scanlon is corrupt and an ally of Chudnofsky.

Its negative pole L15·5647- (max 5.38):

- 5.38 ' patients' · *Arkham Asylum (game)*: There, Batman learns that Joker returned to the asylum to gain access to Young, who has been developing Titan—a more powerful version of the Venom drug that gi…
- 5.03 ' element' · *Arkham City*: During a fight between the two, Strange activates Protocol 10, which is revealed to be a scheme to wipe out the entire population of Arkham City and destroy th…
- 4.97 ' combat' · *Knightfall*: Jean-Paul is shown to be a different, but not dangerous, Batman until an encounter with the Scarecrow, which results in Jean-Paul being infected by Scarecrow's…
- 4.69 'ighting' · *Batman Forever*: Seeking to avenge the death of his family, Dick demands to join Batman in crime-fighting, hoping to kill Two-Face, but Bruce declines in order to help Dick mov…
- 4.19 ' protect' · *Earth One*: Gordon, angry, but also terrified about what Bullock did, explains to him that he was once as fearless and resistant to Gotham's corruption as Bullock, but bec…

Agent 2's examples and near misses (activation, share of corpus max):

- example 1.70 (37%) ' the': A crooked senator takes bribes from the mob and lets the crime boss run gambling out of the harbor.
- example 2.00 (44%) ' manufacturer': An arms manufacturer secretly delivers weapons to the terrorists in exchange for a share of their profits.
- example 1.43 (31%) ' in': The archaeologist agrees to hand the relic to the dictator's regime in return for his brother's release.
- example 0.97 (21%) '.': The blackmailer forces the judge into giving him the names of witnesses, which he sells to the cartel.
- example 1.23 (27%) ' a': A city official becomes an ally of the gangster, drafting a denunciation of the reporter investigating them both.
- near miss 0.10 (2%) ' after': The caped vigilante fights crime night after night, protecting the city from the criminal element.
- near miss 0.10 (2%) ' medical': A young doctor refuses the bribe and reports the hospital administrator to the medical board.
- near miss 0.24 (5%) ' worth': A farmer trades his old tractor in exchange for a fishing boat and a season's worth of bait.
- near miss 0.48 (10%) ' stores': A rookie signs a record deal and the label delivers his first album to stores across the country.
- near miss 0.02 (0%) ' families': The nurse works double shifts to help patients and becomes an ally of the families fighting the hospital closure.

Steered generations:

- α=4.56 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is carrying a backpack and a small bag. He is followed by a dog. He is wearing a red shirt. He is walking toward a cabin. He enters the cabin
- α=4.56 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is struggling to find her place in the world. She is working as a barista at a small café, and feels trapped in her mundane life.…
- α=4.56 · The story begins when → the narrator, a young man, is walking through the forest. He is alone, and he feels a sense of unease. He hears a sound, and then sees a figure in the distance. The figure is described as being tall…
- α=9.12 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is carrying a backpack. He is followed by a woman, named Sarah. She is wearing a red jacket. They are both wearing black clothing. Jack is wea…
- α=9.12 · Plot: In the first act, → the story introduces the protagonist, a young woman named Emily, who is struggling to find a job. She is determined to succeed and is willing to do anything to achieve her goals. The story then intro…
- α=9.12 · The story begins when → the narrator, a young man, is walking through the forest. He is alone, and he feels a sense of foreboding. He hears a sound, and then sees a figure approaching. He is afraid, but he tries to be

### 28. Prison and the inmates released

**Paper's unit L4·8535+** (layer 4): dev 2/10, test 2/6 (AUC 0.667); gate-open corpus max 8.188, present in 20 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Penny' ×26, 'it' ×4, ' Pe' ×2, ' Pen' ×1, 'ino' ×1.

**New unit L31·9620-** (layer 31): 4618 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 43.675, neg 1.934); test 6/6 (AUC 1.0; mean pos 41.417, neg 3.339); p99 on the other memes' sentences 17.233 (max 42.75). Corpus max 134.0, p99 36.002; present in 0 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 0, only new 0 (), only paper 20 (A Death in the Family, Arkham Asylum (game), Arkham City, Arkham Knight, Batman & Robin, Batman '89, Batman Begins, Batman Forever). Batman works scoring above the median ordinary film: 29 of 32 for the new unit, 21 for the paper's.
- Top-50 peak tokens: ' to' ×8, ' after' ×3, ' at' ×3, ' from' ×3, ' the' ×3, ' by' ×2.
- Agent 1 (blind to the label, rates it conceptual, medium confidence) explains it as: *This unit fires at very high strength on function words (to, by, after, at, from, the) running through sentences about life inside a prison — arriving to serve a sentence, being befriended by another inmate, a fellow inmate, guards, parole after 40 years — so it is tracking a narrative context rather than any content word. Every one of the strongest firings comes from the same Shawshank Redemption summary, so the data cannot separate 'prison-life context' from 'this particular story'. The opposite pole fires on the legal process that puts people in prison — judge, testifying, implicate, death sentence, ten-year prison sentences, sentencing without trial — so the distinction is life inside the walls as opposed to the courtroom and sentencing.*
- Judge: **partial**. It captures life inside a prison, including parole, but admits it cannot separate the concept from one Shawshank summary and barely touches the release or breakout half of the label. Label the explanation supports: *Life inside a prison (one story's context)*
- Agent 2 blind maximum: 106.50 against corpus max 134.00 (weakest of the top 8: 85.00; peak ' fellow'): After arriving at the prison to serve a life sentence, the banker is befriended by a fellow inmate, protected from the guards, and paroled after forty years.
- Agent 2 examples firing strongly (≥ 50% of max): 2 of 5; near misses: 0 of 5.
- Steering: α=134.0: none; α=134.0: weak The origin wound (haunted by a personal tragedy that has left); α=134.0: none; α=268.0: weak The butler who raised him (a young serving man); α=268.0: none; α=268.0: strong The butler who raised him (the serving man, who is in the service)
- **Verdict:** Partial, confounded. Life inside a prison, read entirely from the Shawshank summary; the strongest unit in the set (maximum 134), held-out 6 of 6, present in 0 Batman works under the paper's rule because that one film sets the bar.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 134.00 ' to' · *The Shawshank Redemption*: In 1947, Portland, Maine, banker Andy Dufresne arrives at Shawshank State Prison to serve two consecutive life sentences for murdering his wife and her lover.
- 128.00 ' by' · *The Shawshank Redemption*: He is befriended by Ellis Boyd "Red" Redding, a contraband smuggler serving a life sentence, who procures for him a rock hammer and a large poster of Rita Hayw…
- 106.00 'friend' · *The Shawshank Redemption*: In 1965, Andy and Red befriend Tommy Williams, a young inmate convicted of burglary.
- 96.50 'ars' · *The Shawshank Redemption*: In 1949, Andy overhears the captain of the guards, Byron Hadley, complaining about being taxed on an inheritance and offers to help him shelter the money legal…
- 93.50 ' to' · *The Shawshank Redemption*: After the Sisters beat Andy to near-death, Hadley cripples Bogs, who is subsequently transferred to a minimum-security hospital; Andy is not attacked again.
- 88.50 ' fellow' · *The Shawshank Redemption*: Red worries that Andy is suicidal, especially after learning that he asked a fellow inmate for a rope.
- 87.50 ' by' · *The Shawshank Redemption*: Assigned to work in the prison laundry, Andy is frequently raped by "the Sisters" gang, led by Bogs Diamond.
- 85.00 ' after' · *The Shawshank Redemption*: The following year, Red is paroled after serving 40 years, but struggles to adapt to life outside prison and fears that he never will.

Its negative pole L31·9620+ (max 21.88):

- 21.88 ' ' · *Goodfellas*: On June 11, 1970, Billy Batts, a made man in the Gambino crime family who was recently released from prison, incessantly insults Tommy at a nightclub owned by…
- 19.00 ' imp' · *Year One*: Flass supplies Dent with the evidence and testimony needed to implicate Loeb, who resigns in disgrace.
- 15.88 ' test' · *Year One*: Batman uses the opportunity to terrorize Skeevers into testifying against Flass.
- 15.00 ' death' · *Gone Girl*: Her original plan was to drown herself after Nick's arrest and have her body found to ensure his death sentence.
- 14.06 ' judge' · *Mrs. Doubtfire*: Disturbed by Daniel's behavior, the judge grants Miranda full custody of the children, only granting Daniel supervised visitation rights every Saturday, and su…

Agent 2's examples and near misses (activation, share of corpus max):

- example 73.00 (55%) ' by': Inside the penitentiary, the new inmate is befriended by the prison librarian and learns to survive the guards.
- example 62.75 (47%) ' in': A young thief serving five years in a brutal prison is taken under the wing of an older lifer.
- example 77.00 (57%) ' fellow': Years into his sentence, the inmate runs the prison library and smuggles letters for his fellow convicts.
- example 59.25 (44%) ' after': Denied parole after decades behind bars, the old convict is finally released and struggles to adjust to freedom.
- example 30.12 (22%) ' w': In the prison yard, the inmates trade cigarettes and the warden's guards beat anyone who steps out of line.
- near miss 19.75 (15%) ' to': The judge sentences the banker to two life terms after his former partner testifies against him in court.
- near miss 1.09 (1%) ' employee': After arriving at the hotel to serve as a bellhop, the boy is befriended by a fellow employee and the chef.
- near miss 0.12 (0%) ' the': The prosecutor implicates the mayor, and the jury returns a death sentence without a full trial.
- near miss 3.59 (3%) ' turns': The guards at the museum are befriended by a fellow art lover who turns out to be a thief.
- near miss 3.53 (3%) ' defendant': The lawyer argues for a ten-year sentence while the defendant's wife waits outside the courtroom.

Steered generations:

- α=134.0 · Plot summary: The film opens as → a young man, named Nick, is serving in the military in the Middle East. He is in a vehicle with his fellow soldiers, and they are attacked by a group of insurgents. Nick is shot in the leg, and he is…
- α=134.0 · Plot: In the first act, → the story introduces the protagonist, a young woman named Emily, who is serving as a nurse in a small town. She is known for her compassion and dedication to her patients. However, she is also haunte…
- α=134.0 · The story begins when → the narrator, a young man, is serving in the army. He is assigned to a group of soldiers who are serving in a remote area. One day, he is told that he is to be transferred to a new unit. He is
- α=268.0 · Plot summary: The film opens as → a young serving man, serving in the British army, is serving in the British army in the early 1900s. He is serving in the British army in the early 1900s. He is serving in
- α=268.0 · Plot: In the first act, → the story serves to introduce the characters, the setting, and the overall tone of the film. The first act of "In Serving  H  In  H  In  H  In  H 
- α=268.0 · The story begins when → the serving man, who is in the service of the Count, is serving the Count's son, who is in the service of the Count. The serving man is serving the Count's son, who is in the service of the Count.

### 29. The antidote

**Paper's unit L1·10779-** (layer 1): dev 3/10, test 2/6 (AUC 0.528); gate-open corpus max 7.906, present in 7 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' antid' ×8, ' anti' ×2, ' antim' ×1, ' antiqu' ×1, 'icip' ×1.

**New unit L27·3551-** (layer 27): 6729 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 7.279, neg 0.302); test 6/6 (AUC 0.917; mean pos 5.199, neg 0.986); p99 on the other memes' sentences 1.765 (max 6.438). Corpus max 16.5, p99 3.813; present in 8 of 32 Batman works, 2 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 5, only new 3 (Arkham Knight, Batman '89, White Knight), only paper 2 (Batman v Superman, Dark Knight Returns). Batman works scoring above the median ordinary film: 30 of 32 for the new unit, 7 for the paper's.
- Top-50 peak tokens: ' antid' ×5, ' can' ×3, ' has' ×3, ' created' ×2, "'s" ×2, ' develops' ×2.
- Agent 1 (blind to the label, rates it conceptual, medium confidence) explains it as: *This unit fires in sentences about a poison, toxin or disease at the point where a cure or antidote is being produced or named: on the verb just before 'a cure' ('has created', 'has been developing'), on the onset of 'antidote' and 'inoculated', and on 'to' in 'allow Gotham's residents to [escape the toxin]'. It anticipates the remedy for the poisoning rather than the poisoning itself (the 'hospitals have been poisoned' firing is the one exception, and 'Rachel now respects' is a stray).*
- Judge: **direct**. It describes the point in a poisoning where a cure or antidote is produced or named, which is the antidote trope itself. Label the explanation supports: *The cure for a poisoning*
- Agent 2 blind maximum: 9.75 against corpus max 16.50 (weakest of the top 8: 7.34; peak ' developing'): After the fear toxin poisons the city's water supply, the caped vigilante's butler has been developing a cure and an antidote so residents can be inoculated against the poison.
- Agent 2 examples firing strongly (≥ 50% of max): 0 of 5; near misses: 0 of 5.
- Steering: α=16.5: none; α=16.5: none; α=16.5: none; α=33.0: none; α=33.0: none; α=33.0: none
- **Verdict:** Direct by the judge: the point where a cure for a poison is produced. But Agent 2's examples do not fire (0 of 5) and the blind maximum reaches 59% of the maximum, so the explanation loses; the unit's peak position (the verb before 'a cure') is hard to hit from a description.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 16.50 ' to' · *Batman '89*: Batman rescues Vicki, takes her to the Batcave, and provides her with all of his research on Smylex, which will allow Gotham's residents to escape the toxin.
- 13.50 ' developing' · *Arkham City*: Desperate to save himself and innocent citizens, Batman seeks out Mr. Freeze, who has been developing a cure but has since been kidnapped by the Penguin.
- 13.25 ' created' · *Arkham City*: Freeze tells Batman that he has created a cure, but it is rendered useless via instability.
- 8.69 ' have' · *Arkham City*: Joker also reveals that Gotham hospitals have been poisoned with his infected blood.
- 7.94 ' now' · *Batman Begins*: Rachel now respects Bruce but refuses to be with him, promising they can be together when Gotham no longer needs Batman.
- 7.66 ' antid' · *Batman Begins*: He awakens on his birthday, having been saved by Alfred and Fox, who developed an antidote for the toxin.
- 7.44 ' inoc' · *Batman Begins*: As the gas strikes the Narrows, Rachel, the inoculated Gordon, and Batman struggle to survive.
- 7.34 'ola' · *The Bat (1926)*: Van Gorder asks the gardener about his knowledge on alopecia, urticaria, and rubeola, and he answers as if the terms referred to plants rather than medical con…

Its negative pole L27·3551+ (max 3.06):

- 3.06 ' his' · *Get Out*: Six months later, Black photographer Chris Washington travels to upstate New York for a weekend getaway to meet the family of his white girlfriend, Rose Armita…
- 1.48 ' early' · *Batman & Robin*: Batman, Robin, and the police arrive at Freeze's lair in response to his escape, discovering Nora preserved in a cryogenic chamber and that Freeze has develope…
- 1.03 ' a' · *Jaws*: In the New England beach town of Amity Island, a young woman goes for a late-night ocean swim.
- 0.73 ' B' · *Knightfall*: Bane, introduced in Batman: Vengeance of Bane #1 (January 1993) by Chuck Dixon and Graham Nolan, an orphan born and raised in a Central American island prison,…
- 0.72 ' maybe' · *Casablanca*: At the last moment, Rick makes Ilsa board the plane to Lisbon with Laszlo, telling her that she would regret it if she stayed, "Maybe not today, maybe not tomo…

Agent 2's examples and near misses (activation, share of corpus max):

- example 5.53 (34%) ' created': With the nerve gas spreading through the subway, the scientist has created a cure and races to inoculate the survivors.
- example 3.77 (23%) ' antid': The doctor synthesises an antidote to the snake venom just in time to save the poisoned child.
- example 5.38 (33%) ' a': After the plague sweeps the village, the healer has been developing a cure from the rare mountain flower.
- example 7.19 (44%) ' to': The hero steals the formula for the antidote, allowing the city's residents to escape the toxin before midnight.
- example 5.28 (32%) ' inoc': Infected by the laughing gas, the detective must find the chemist who has produced a cure and get everyone inoculated.
- near miss 1.34 (8%) ' residents': The Scarecrow releases his fear toxin across the Narrows and the poisoned residents riot through the night.
- near miss 0.00 (0%) None: The butcher cures the ham in the smokehouse while Alfred develops the photographs in the darkroom.
- near miss 2.36 (14%) ' developing': Bruce Wayne has been developing a new business plan to allow Gotham's residents to escape poverty.
- near miss 1.42 (9%) ' antid': Rachel finds Harvey's cynicism the perfect antidote to Bruce's brooding, and the two bond over dinner.
- near miss 3.47 (21%) ' Ivy': Poison Ivy seduces Robin with her toxic kiss and the Penguin disappears into the sewers.

Steered generations:

- α=16.5 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is carrying a backpack and seems to be searching for something. He finds a small, red box. He opens it and finds a letter inside. The letter s…
- α=16.5 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elise, who is struggling to find her place in the world. She is an aspiring artist, but her family expects her to pursue a more stable career…
- α=16.5 · The story begins when → the narrator, a young man, is walking through the forest. He is in search of a certain tree, which he has heard is the most beautiful tree in the forest. He is told by an old woman that the tree is n…
- α=33.0 · Plot summary: The film opens as → a young woman, named Alice, is walking through a forest. She is followed by a man in a red coat. She tries to escape, but he catches up to her. She is then taken to a strange house, where she is
- α=33.0 · Plot: In the first act, → the story introduces the main characters and the setting. The protagonist, a young woman named Elise, is struggling to find her place in the world. She is an aspiring artist, but her family expects h…
- α=33.0 · The story begins when → the narrator, a young man, is caught in a rainstorm. He is in a small town, and he is trying to find shelter. He goes into a small, run-down hotel called the "Wooly" and meets the

### 30. A mutating super-formula

**Paper's unit L3·10668+** (layer 3): dev 3/10, test 2/6 (AUC 0.667); gate-open corpus max 10.562, present in 6 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: ' Mut' ×6, ' mutant' ×4, ' mutating' ×3, ' mutated' ×2, 'agen' ×1.

**New unit L33·5542-** (layer 33): 7570 units passed the dev bar; this one scored best. Dev 10/10 (mean pos 16.959, neg 2.305); test 6/6 (AUC 1.0; mean pos 8.932, neg 1.276); p99 on the other memes' sentences 6.82 (max 20.25). Corpus max 42.25, p99 11.189; present in 0 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 0, only new 0 (), only paper 6 (Arkham Asylum (game), Arkham City, Arkham Knight, Batman & Robin, Dark Knight Returns, Hush). Batman works scoring above the median ordinary film: 18 of 32 for the new unit, 6 for the paper's.
- Top-50 peak tokens: ' him' ×3, ' drug' ×3, ' behaving' ×2, ' become' ×2, ' notices' ×2, ' are' ×2.
- Agent 1 (blind to the label, rates it conceptual, high confidence) explains it as: *This unit fires when a character's behaviour or nature is being described as changed or abnormal, peaking on the verb just before the manner word that says so: 'behaving [strangely]', 'behaving [so out of character]', 'behaves [unnaturally]', 'reacts [angrily]', 'causes him [to go insane]', 'become [a new type of supersoldier]', 'has grown [into an adult overnight]', 'medical tests find [no physical cause]'. It predicts an upcoming description of abnormal behaviour or transformation; the opposite pole instead fires on the 'ab-'/'Ab' word-onset itself ('abnormally violent', 'Abruptly', 'Abraham') plus some altered-state words (hypnosis, hideous, alcohol).*
- Judge: **partial**. It captures anticipated abnormal behaviour or transformation (supersoldier, grown overnight), the effect of a formula, but no chemical or serum is described. Label the explanation supports: *Anticipating abnormal behaviour or transformation*
- Agent 2 blind maximum: 44.50 against corpus max 42.25 (weakest of the top 8: 21.38; peak ' behaving'): After the experiment the boy's parents notice him behaving strangely and acting so out of character that medical tests find no physical cause for the change.
- Agent 2 examples firing strongly (≥ 50% of max): 2 of 5; near misses: 1 of 5.
- Steering: α=42.25: none; α=42.25: weak Superhero powers (peaks on Superman) (her unique abilities, which are cons…; α=42.25: none; α=84.5: none; α=84.5: none; α=84.5: none
- **Verdict:** Partial. Anticipated abnormal behaviour or transformation; the blind maximum beats the corpus maximum, but only two of five examples fire and one near miss does.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 42.25 ' behaving' · *Vertigo*: Gavin Elster, an acquaintance from college, asks Scottie to follow his wife, Madeleine, claiming that she has been behaving strangely.
- 33.75 ' behaving' · *Hush*: Batman tells him of his suspicions that some mastermind is behind all his enemies behaving so out of character.
- 28.88 ' find' · *The Exorcist*: Regan's personality becomes violent, and medical tests find no physical cause.
- 26.75 ' behaves' · *Get Out*: Like Walter and Georgina, Logan behaves unnaturally, and is married to a white woman thirty years his senior.
- 26.12 ' him' · *Spider-Man*: The chemical gives him superhuman strength but causes him to go insane and kill one of his scientists.
- 25.00 ' become' · *Knightfall*: Bane, introduced in Batman: Vengeance of Bane #1 (January 1993) by Chuck Dixon and Graham Nolan, an orphan born and raised in a Central American island prison,…
- 23.50 ' reacts' · *The Shining*: He blames Danny for inflicting the bruises on himself and reacts angrily when Wendy suggests leaving the hotel.
- 21.38 ' grown' · *Big*: The next morning, Josh discovers that he has grown into an adult overnight.

Its negative pole L33·5542+ (max 42.75):

- 42.75 ' ab' · *Year One*: With sleep deprivation causing Jim to be paranoid and abnormally violent, he opens fire on Viti's men.
- 15.31 'roy' · *Arkham Knight*: Throughout the city, Batman is faced with several supervillains: he must overcome Scarecrow's (John Noble) plot to threaten Gotham City, dismantle the Penguin'…
- 13.69 ' Abr' · *Batman R.I.P.*: Abruptly, Bruce finds that Honor is no longer sitting next to him and learns from a local dealer that Honor had blown two hundred dollars on heroin and overdos…
- 12.94 ' hyp' · *Dracula*: Using hypnosis, Van Helsing exploits Mina's faint psychic connection to Dracula to track his movements and they pursue, guided by Mina.
- 12.19 'Keith' · *Arkham Knight*: Arkham Knight also features appearances and voice cameos by various characters drawn from the history of Batman comics and the previous Arkham games, including…

Agent 2's examples and near misses (activation, share of corpus max):

- example 26.38 (62%) ' behaving': Back from the expedition, the professor starts behaving strangely, forgetting names and snapping at his students.
- example 4.72 (11%) ' become': The serum causes the frail recruit to become a towering supersoldier within minutes of the injection.
- example 45.25 (107%) ' behaving': His wife notices the astronaut behaving so out of character that she begins to suspect he is not her husband.
- example 9.88 (23%) ' reacts': Exposure to the meteorite causes the farm boy to go insane, and he reacts violently when his mother approaches.
- example 20.50 (48%) ' physical': The twelve-year-old wakes to find she has grown into an adult overnight, and her doctors find no physical cause.
- near miss 26.62 (63%) ' behaves': Alfred behaves impeccably at the charity gala and keeps the guests supplied with champagne.
- near miss 1.85 (4%) ' becomes': Harvey Dent becomes district attorney and vows to clean up Gotham's streets within the year.
- near miss 0.11 (0%) ' the': Abruptly, Abraham abandons the abbey and drives the abandoned road to Gotham in an absolute downpour.
- near miss 5.00 (12%) ' grown': The ivy has grown over the walls of Wayne Manor during the years the house stood empty.
- near miss 9.19 (22%) ' find': Medical tests find a hairline fracture in Bruce's wrist after the fall from the fire escape.

Steered generations:

- α=42.25 · Plot summary: The film opens as → a young man, named Jack, is walking through a forest. He is carrying a backpack and seems to be searching for something. He finds a small, wooden box. He opens it and finds a letter inside. The lette…
- α=42.25 · Plot: In the first act, → the story introduces the protagonist, a young woman named Elara, who lives in a small village. She is an outcast due to her unique abilities, which are considered dangerous by the villagers. One day,…
- α=42.25 · The story begins when → the narrator, a young man, is walking through the forest. He is alone, and he feels a sense of unease. He hears a sound, and then sees a figure in the distance. The figure is tall and thin, with
- α=84.5 · Plot summary: The film opens as → a young aboriginal man, Kombi, is abducting a white aborigine, Tjukabai, from the aboriginal camp. He abducts him to abduct him to the aboriginal camp of the
- α=84.5 · Plot: In the first act, → the abductors take the girl to the forest. She is abducting by a man who is abducting her. She is abducting by a man who is abducting her. She is abducting by a
- α=84.5 · The story begins when → the abominable abominable abominable abominable abominable abominable abominable abominable abominable abominable abominable abominable abominable abominable abominable abomin

### 31. Fear toxin

**Paper's unit L5·5136+** (layer 5): dev 1/10, test 1/6 (AUC 0.583); gate-open corpus max 3.734, present in 8 of 32 Batman works under the paper's rule on gate-open activations (1 ordinary films). Top-50 peak tokens: 'D' ×5, ' compound' ×3, ' comp' ×3, 'p' ×2, 'ui' ×2.

**New unit L22·4641-** (layer 22): 8123 units passed the dev bar; this one scored best. Dev 9/10 (mean pos 9.42, neg 0.577); test 5/6 (AUC 0.806; mean pos 7.792, neg 2.864); p99 on the other memes' sentences 2.754 (max 21.5). Corpus max 15.0, p99 4.719; present in 1 of 32 Batman works, 0 of 13 relatives, 1 of 69 films.

- Presence against the paper's unit: both 0, only new 1 (Arkham Asylum (comic)), only paper 8 (Batman Begins, Batman v Superman, Dark Knight Rises, Earth One, Hush, Return of the Joker, The Killing Joke, White Knight). Batman works scoring above the median ordinary film: 15 of 32 for the new unit, 12 for the paper's.
- Top-50 peak tokens: ' to' ×3, ' that' ×3, ' she' ×3, 'us' ×3, ' of' ×2, ',' ×2.
- Agent 1 (blind to the label, rates it conceptual, high confidence) explains it as: *This unit fires on the word that introduces an imagined, delusional or illusory perception, i.e. content a character believes or sees that is not real: 'imagining [that he sees her]', 'delusions of [being tormented]', 'pretend suitors', 'an imaginary intimate relationship,', 'believing [that Batman will become]', 'unseen forces', 'appears to [float ... but turns out to be]', 'a now-delusional Nygma exclaims that'. The opposite pole fires instead on doubt about reality and memory: déjà vu, 'not sure whether this actually happened', 'sense of reality', awakening from a dream, memory loss.*
- Judge: **partial**. It captures delusional or imagined perception, which is the toxin's hallucinatory effect, but neither fear nor a chemical agent appears. Label the explanation supports: *Imagined or delusional perceptions*
- Agent 2 blind maximum: 13.94 against corpus max 15.00 (weakest of the top 8: 8.56; peak ' imagining'): The lonely clerk begins imagining that his neighbour loves him, nursing an imaginary romance and delusions of being watched by unseen forces that are not real.
- Agent 2 examples firing strongly (≥ 50% of max): 4 of 5; near misses: 0 of 5.
- Steering: α=15.0: none; α=15.0: none; α=15.0: none; α=30.0: weak A mutating super-formula (turns into a giant, monstrous, green creature); α=30.0: weak The origin wound (haunted by the ghost of her late mother); α=30.0: weak The origin wound (with his mother, who is dying)
- **Verdict:** Partial. Imagined or delusional perception, the toxin's effect without the toxin; four of five examples fire, no near miss, held-out 5 of 6.

Strongest corpus sentences of the new unit (gate-open, peak token):

- 15.00 ' imagining' · *Vertigo*: Following his release, he frequents the places that Madeleine visited, often imagining that he sees her.
- 11.31 ' to' · *The Bat (1926)*: The candles in the ballroom go out when lit, and a shape appears to float towards Anderson and Lizzie, but it turns out to be the Japanese butler Billy carryin…
- 9.88 ' of' · *Arkham Asylum (comic)*: In flashbacks, it is shown that Arkham's mentally ill mother, Elizabeth, suffered delusions of being tormented by a supernatural entity.
- 9.69 ' pretend' · *Rear Window*: During a heat wave, Jeff watches his neighbors through open windows, including a newlywed couple; a professional dancer coined "Miss Torso"; a songwriter with…
- 9.50 ',' · *Joker (2019)*: She developed an imaginary intimate relationship with Thomas, which included Arthur being their biological son.
- 9.25 ' believing' · *Arkham Knight*: Batman and Robin capture Harley and the infected, but one of the patients kills the others and himself, believing that Batman will become the perfect "Joker".
- 8.62 ' unseen' · *The Shining*: Danny gets lured to room 237 by unseen forces, and Wendy later finds him with signs of physical trauma, for which she blames Jack.
- 8.56 ' that' · *Batman Forever*: Committed to Arkham Asylum, a now-delusional Nygma exclaims that he is Batman.

Its negative pole L22·4641+ (max 4.09):

- 4.09 ' experiences' · *Groundhog Day*: He experiences the previous day's events repeating exactly and believes he is experiencing déjà vu.
- 3.58 ' what' · *Big*: His ideas become valuable assets to MacMillan Toys; however, he begins to forget what it is like to be a child, and his tight schedule rarely allows him to spe…
- 3.20 ' as' · *The Killing Joke*: He later admits that he is not sure whether this actually happened, as he tends to remember his past differently at different times.
- 2.97 ' sense' · *Batman R.I.P.*: This statement devastates Bruce and further confuses his sense of reality.
- 2.86 ' believing' · *The Matrix*: Neo awakens at home, believing the encounter was a nightmare until Trinity and her companions remove the device and take him to Morpheus.

Agent 2's examples and near misses (activation, share of corpus max):

- example 11.56 (77%) ' imagining': The widower keeps imagining that he sees his dead wife in the crowd, though no one else can see her.
- example 10.19 (68%) ' being': Suffering delusions of being hunted by the government, the mathematician invents a handler who was never there.
- example 10.62 (71%) ' imaginary': The lonely girl carries on an imaginary friendship with a talk show host, believing that he will one day call her.
- example 1.18 (8%) ' it': The ghost appears to float through the hallway, but it turns out to be a projection rigged by the caretaker.
- example 7.59 (51%) ' him': Now delusional, the inventor exclaims that the caped vigilante is speaking to him through the television.
- near miss 0.33 (2%) ' the': Bruce wakes from the dream and is not sure whether the fall into the well actually happened.
- near miss 1.18 (8%) ' it': Wayne Enterprises engineers imagine a new design for the Tumbler and build it in the applied sciences lab.
- near miss 0.17 (1%) ' the': The Joker appears at the mob meeting and makes a pencil disappear in front of the bosses.
- near miss 0.34 (2%) ' and': Selina has a sudden sense of déjà vu in the ballroom and cannot remember how she got there.
- near miss 0.15 (1%) ' unseen': The previously unseen footage of the Batmobile chase is released as a bonus feature on the disc.

Steered generations:

- α=15.0 · Plot summary: The film opens as → a young girl, named Lina, is playing with her dog in the park. She is chased by a man in a black coat, who is trying to catch her. She runs into a tree, and the man catches her. She
- α=15.0 · Plot: In the first act, → the audience is introduced to the world of the play. The setting is a small town where the protagonist, a young woman named Clara, lives. She is in love with a man named Ethan, who is a local artist.…
- α=15.0 · The story begins when → the narrator, a young man, is walking in the forest. He is alone, and he feels the presence of a stranger. He hears a voice calling him. He turns around, but there is no one. He continues walking, but
- α=30.0 · Plot summary: The film opens as → a young girl, named Lina, is playing with her dog in the woods. She sees a man in the distance, and then the man turns into a giant, monstrous, green creature. She screams, and then the screen fades…
- α=30.0 · Plot: In the first act, → the audience sees the protagonist, a young woman named Clara, who is haunted by the ghost of her late mother. She is constantly hearing her mother's voice, but no one else can hear it. She is in a me…
- α=30.0 · The story begins when → the narrator, a young man, is in a room with his mother, who is dying. He sees a vision of his father, who is not present. The narrator is transported to a place where he sees his father as a child,

