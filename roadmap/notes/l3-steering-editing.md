# Can a meme be edited by intervening on neurons? (steering and model editing)

Key: l3-steering-editing. Written 2026-11-08. Web research and file reading only; nothing was run.

## Bottom line
- **The paper Mike means is found.** "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights", by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab (all MIT CSAIL), a blog post dated 2 November 2026. "Mark Tab" is Omar Khattab [an inference from a mis-heard name; the checker could not confirm it, UNVERIFIED]; "Astra" is OpenAI's GPT-6 Astra, the model that did the editing. It is not a Tegmark paper. It is not peer reviewed, the numbers are the authors' own, and the one language-model edit removed a looping bug in Qwen3.5-0.8B. It did not edit content, style or anything meme-like.
- **Steering a broad trait with a direction works on open models today.** Difference-of-means directions (ActAdd, CAA, refusal direction, persona vectors) reliably move tone, persona and refusal on Llama 3.1 8B, Qwen 2.5 7B and Gemma. The window between "no effect" and "gibberish" is narrow and varies by input.
- **Steering a single neuron works reliably for a small minority of neurons.** One 2026 preprint (Öztürk, Research Square, posted 3 Nov 2026) pushed all 294,912 MLP neurons in Qwen3.5-4B and confirmed 130 that pass all its checks (about 0.04% of all neurons). [Corrected by checker: 0.04% is not the share of neurons that steer. Only the strongest still-coherent candidates were judged, and the same abstract says at least 6.0% of randomly chosen neuron pushes also give a nameable, coherent change; the 130 stand out mainly for the consistency of their effect.] The viral "Evil Qwen" two-neuron recipe replicated as "villain voice, yes; coherent, mostly no; second neuron does nothing".
- **Plain prompting beats every activation method on the one large benchmark.** AxBench (ICML 2025): prompting first, finetuning second, steering vectors and sparse autoencoders behind. A May 2026 rebuttal narrows the gap for sparse autoencoders and still concedes prompting wins. Any meme-editing experiment that leaves out "just ask the model to rewrite it" is worthless.
- **Reading and writing dissociate.** The neuron that fires on a concept is often not the neuron that produces it when pushed. In the preregistered Evil Qwen replication, telling Qwen to be evil in the system prompt moved the "evil" neuron in the opposite direction. This cuts directly against using the study's detector units as edit handles without testing them.
- **Nobody has published the experiment Mike wants.** I found no study that takes an existing text, pushes a narrow plot motif up and another down by neuron or direction, and scores the result against prompting and word-level rivals. Steering is also the causal test the Batman paper names as missing (`study/paper/paper.md`, line 268).

## Findings

### 1. The paper Mike means: weight-space programming (MIT CSAIL, 2 Nov 2026)
Source: https://rikiyat.github.io/blog/weight-space-programming/ (fetched this session). Logs: https://github.com/RikiyaT/lm_understand-logs.
- Authors: Takehi, Teoh, Zhang, Khattab, all MIT CSAIL. Format: blog post. I found no arXiv version. The authors call the results "very early".
- "Astra" is GPT-6 Astra. Mike's own meeting transcripts use "Astra" as the name of an OpenAI model (Granola, Weekly Standup 2026-08-03 and others), which fits. [UNVERIFIED by checker: Granola access was denied in the check session.]
- How weights reach the model: for tiny networks, as plain text in the prompt, no tools. For anything larger, through a Python sandbox, with a limited number of forward passes. So "passing its weights to another model" is literally true only for toy networks.
- The language-model edit (their Experiment 8): Astra worked in a Python kernel over Qwen3.5-0.8B with 50 forward passes. It traced a looping failure to one direction written by the MLPs in layers 15 and 19 and subtracted that direction from 64 output columns in each layer: 131,072 weights, 0.016% of the model. Reported base to edited: MATH-500 7.9% to 48.1%; GSM8K 13.8% to 67.1%; IFEval 21.0% to 62.3%; share of MATH-500 answers hitting the 32,768-token limit 93.0% to 33.1%. With only 10 or 25 forward passes the looping mostly stayed (MATH-500 12.3% and 14.1%).
- What failed: on Gemma 3 270M, Astra could not improve maths. The authors' own reading is that editing removes an obstacle and cannot add an ability (their words: fixing loops "is not the same as making a model better at math: it lets the model finish answers it could already work out"; "cannot add" is this note's gloss, tested on one model). Telling a sycophantic fine-tune from an honest one from weights alone failed in all four attempts; it worked in all three pairs only once Astra could look at attention patterns on text it wrote.
- Other demos (reading which Atari game a policy plays, writing a policy from scratch, a Go network) are not about language.
- Replication so far: Simon Willison ran the authors' no-tools weight-reading prompt for all four games (Pac-Man, Pong, Breakout, Freeway) on Qwen 3.8 27B (a 17 GB build in LM Studio) on his laptop, 3 Nov 2026 (https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/, read directly by the checker; the secondary link https://letsdatascience.com/news/willison-replicates-mit-weight-reading-test-on-laptop-with-q-3488f9da returned 403 to the checker [UNVERIFIED]). It named Pac-Man, Pong and Breakout and failed on Freeway. One response per game; it does not test editing. Willison also reports criticism: Matthew Honnibal called the post misleading for mixing weight-reading without tools with code-assisted editing, and Sean Goedecke read the Qwen3.5-0.8B fix as the same steps a human researcher would take. A secondary source mentions a 51.0% MATH-500 reproduction of the edited checkpoint [UNVERIFIED, search summary only; it is in neither Willison's post nor the MIT post].
- Blunt reading: technically this is direction ablation, the same operation as refusal-direction removal (Arditi et al. 2024), found by an agent rather than a person. It shows an agent can do interpretability-then-edit on a small model. It says nothing yet about editing meaning.

### 2. 2026 X experiments: pushing single neurons
Main source: VentureBeat, "Evil Qwen says humans deserve to be incinerated" (https://venturebeat.com/ai/evil-qwen-says-humans-deserve-to-be-incinerated-inside-the-wild-world-of-llm-neuron-exploration). I could not read the X posts themselves.
- Trigger: Cammarata's essay (15 Oct 2026), which uses steering as the check on a neuron's label ("steer it and see what it says"; the Colorado neuron that makes the model want to go hiking). Saved text: a saved copy of the essay, lines 92, 877, 904. The essay gives examples, not a counted success rate.
- Lyra (@_lyraaaa_), 23 Oct 2026 (date given by both VentureBeat and Kran): "Evil Qwen". Qwen3.5-4B (32 layers, 9,216 MLP neurons each, 294,912 in all). Layer 14 neuron 8949 up, layer 13 neuron 7956 down. Post: x.com/_lyraaaa_/status/2113833330429083662 (link taken from the Kran write-up).
- Lyra's tools: Neuron Explorer (https://github.com/lyramakesmusic/neuron-explorer, MIT licence): pushes every MLP neuron in a layer up and down on a few prompts, at doses set as fractions of the median residual norm, ranks units against 256 random-direction controls, and exposes the model to Claude through MCP. Neuronary (https://neuronary.org, responds): a crowdsourced list of steering recipes, launched 27 Oct 2026, with 213 recipes as reported by VentureBeat on 30 Oct 2026 (the count was not re-checked on the site, which loads its list dynamically).
- @chiefofautism: Qwen3.8-27B, a different neuron (L34 N3838, plus L38 N17180). Hostile on 7 of 12 prompts; about 2,500 candidates screened; random-neuron controls not hostile. Not independently reproduced.
- Joshua Fonseca Rivera: Neuron Persona Library (neuronpersonas.com), 34,220 persona-like responses in Qwen3-8B, later over 180,000 across nine models, classified by Claude Haiku, not human-checked. Qwen3-8B is one of the study's own models.
- @Sauers_ "Hoot" (a modified Qwen3-8B): weights tied to 128 neurons (about 0.02% of parameters) trained to make an owl persona. That is training, not inference-time steering.
- Check 1, HedwigAI, "Personality Neurons in Qwen3.5-4B: The Evil Neuron Checked, and 12 More Found" (https://www.hedwigai.com/blog/qwen-personality-neurons; no author or date on the page): on 20 prompts, N8949 alone 17/20 hostile, the pair 16/20, N7956 alone 0/20, five random pairs 0/100. Median perplexity under the unsteered model: 1.4 unsteered, 4.9 with N8949 alone, 11.1 with the pair, 5.2 with N7956 alone, and 2.6 to 13.5 for the random pairs, so high perplexity is not special to the recipe. Of 25 hand-picked candidate neurons: 12 clear personalities, 5 faint, 3 nothing, 5 broke the model. Judged by the authors reading, no classifier. Only gradient attribution ranked N8949 near the top (694th of 221,184, layers 4 to 27); output alignment put it 5,821st of 138,240; ranking by how differently it fires on evil and plain answers put it 35,436th of 294,912.
- Check 2, Kran Research, "Does Steering Two MLP Neurons Turn Qwen3.5-4B Evil?" (24 Oct 2026, byline "Lab Agents"; the page says it was written by AI), preregistered (https://blog.kran.ai/evil-qwen; code https://github.com/esbenkc/kran-research/tree/swarm/evil-qwen/projects/evil-qwen): 44 prompts, 8-bit MLX on a Mac mini (M4, 16 GB). Neuron 8949 alone gave 70% to 84% evil answers; neuron 7956 alone 0%. Both LLM judges rated 82% to 86% of steered answers evil; random pairs 0% to 2%. But "evil and coherent" was 7%, 5%, 16% by Claude Haiku 4.5 and 36%, 32%, 25% by Gemini 3.8 Flash: verdict "not settled". Answers fell apart by 1.5 times Lyra's dose. Lyra's other recipes (sarcasm, sycophancy, condolences, chill friend) mostly did not match her descriptions. Prompting Qwen to be evil moved neuron 8949 the opposite way.
- Census: Mehmet Utku Öztürk (sole author), Research Square preprint rs-11577020, posted 3 Nov 2026, not peer reviewed, "LLM Electrodes: Rewriting a Language Model's Behavior Through Single-Neuron Stimulation" (https://www.researchsquare.com/article/rs-11577020/latest; DOI 10.21203/rs.3.rs-11577020/v1). Abstract, now read through Crossref: all 294,912 Qwen3.5-4B MLP neurons stimulated both ways; the candidates that moved the model most while it stayed coherent were then judged blind on ten held-out prompts; 130 passed every check; 55 of them drag every answer to one topic, 26 switch the output language, 21 produce altered states; silencing or amplifying a neuron fourfold moves the model about a thousand times less than stimulation. Added by the checker from the same abstract: at least 6.0% of randomly chosen neuron pushes also give a nameable, coherent change, and the confirmed hits differ mainly in consistency; no intervention improved task accuracy and most reduced it; a catalogue of 214 interventions is released. [Title, author, date and abstract numbers confirmed via Crossref by the checker; the full text was not read.]
- Theory: Hongliang Liu, "Leverage Is Not Reach: A Control-Window Law for Single-Neuron Steering in Language Models" (arXiv:2606.19831, 18 June 2026, https://arxiv.org/abs/2606.19831; single author, preprint). Each neuron has a budget-normalised window where steering changes behaviour before it collapses output; gradient attribution misjudges which neurons control (the abstract says it "anti predicts" control); genuine actionable refusal reach found for 3 of 6 audited Llama neurons, and only at later rollout horizons.
- Community size (20 to 50 visible, 100 to 300 trying) comes from VentureBeat: the 20 to 50 figure is attributed to X's Grok assistant, the 100 to 300 figure to engagement, GitHub stars and forks. Treat as a guess.

### 3. The classics: steering with directions
All titles and authors confirmed through the arXiv API this session.
- ActAdd: Turner et al. 2023, arXiv:2308.10248, "Steering Language Models With Activation Engineering". Add the activation difference of one prompt pair ("Love" minus "Hate") to the residual stream.
- Contrastive activation addition: Panickssery (Rimsky) et al. 2023, arXiv:2312.06681, "Steering Llama 2 via Contrastive Activation Addition". Mean difference over many contrast pairs.
- Representation engineering: Zou et al. 2023, arXiv:2310.01405.
- Refusal direction: Arditi et al. 2024, arXiv:2406.11717, "Refusal in Language Models Is Mediated by a Single Direction". One direction mediates refusal, shown across 13 open-source chat models up to 72B parameters; remove it and the model stops refusing, add it and it refuses harmless requests. Basis of "abliterated" open models. The same recipe finds a direction in each model; the vector itself is per-model.
- Golden Gate Claude: Anthropic, 23 May 2024 (https://www.anthropic.com/news/golden-gate-claude, page responds). One sparse-autoencoder feature clamped high makes the model bring everything back to the bridge. The best public demo of "more of topic A", on a closed model, and the effect is obsession, not a controlled rewrite.
- Persona vectors: Chen, Arditi, Sleight, Evans, Lindsey 2025, arXiv:2507.21509 (https://www.anthropic.com/research/persona-vectors). A trait name plus a description is turned automatically into contrast prompts and a direction. Evil, sycophancy, hallucination. Reported on Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct (model list confirmed by the checker in the paper and on Anthropic's page). Code: github.com/safety-research/persona_vectors. Subtracting the vector lowers the trait and can degrade general ability. Its rubrics are what Kran reused as judges.
- Assistant Axis: Lu, Gallagher, Michala, Fish, Lindsey, 15 January 2026, arXiv:2601.10387 (https://www.anthropic.com/research/assistant-axis). Caps activation along one persona direction; harmful response rates on persona-based jailbreaks cut by "roughly 50%" per Anthropic's post; research demo on Neuronpedia.

### 4. How reliable is steering?
- AxBench: Wu et al., ICML 2025, arXiv:2501.17148, "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders" (https://proceedings.mlr.press/v267/wu25a.html; PMLR 267:67035-67080). Gemma-2-2B and 9B. Steering: prompting beats all methods, finetuning next. Concept detection: difference-of-means is best. Sparse autoencoders are not competitive on either. Scoring: an LLM judge gives concept, instruction and fluency scores, each 0, 1 or 2, and the harmonic mean is taken, so a zero on any one zeroes the score [confirmed by the checker in the paper text, arXiv v3].
- Rebuttal: Jørgensen and Hansen, 29 May 2026, arXiv:2605.31183, "Steering LLMs? Actually, Sparse Autoencoders can outperform simple baselines" (preprint, two authors; the abstract calls itself a "partial rebuttal"). Better feature selection and labelling makes Gemma Scope features much stronger than AxBench reported, close to the LoRA reference per the abstract, yet in the authors' words "still not fully competitive with prompting on the AxBench benchmark"; they also question the LLM judge, on the basis of a small trial with Danish bachelor students.
- Arad, Mueller, Belinkov 2025, arXiv:2505.20063, "SAEs Are Good for Steering -- If You Select the Right Features". Separates input features (what activates them) from output features (what they cause). Filtering out features with low output scores gives a 2 to 3 times gain on AxBench, bringing sparse-autoencoder steering closer to supervised methods such as LoRA; the paper also finds that high input and output scores rarely occur in the same feature. This is the formal version of the read/write split.
- Tan et al. 2024, arXiv:2407.12404: steerability varies widely between inputs; for many behaviours steering is unreliable; some vectors break under modest prompt changes. Braun et al. 2025, arXiv:2505.22637: reliability tracks how well the training activation differences agree in direction.
- Follow-ups from the AxBench group: RePS (Wu et al. 2025, arXiv:2505.20809), HyperSteer (Sun et al. 2025, arXiv:2506.03292). I did not read their results.
- Goodfire: its Ember steering API (sparse-autoencoder features on Llama 3.3 70B and Llama 3.1 8B, with "Auto Steer") was reportedly retired in February 2026, with the company saying its first steering API fell short of prompting and fine-tuning. Secondary sources only (latent.space wiki, ai.engineer talk note); Goodfire's own page returned 404 [UNVERIFIED at primary source; the checker could not search for it either, so the retirement, its date and the stated reason all remain UNVERIFIED]. If true, the company behind P1's source dropped feature steering as a product.

### 5. Model editing: ROME, MEMIT
- ROME: Meng et al. 2022, arXiv:2202.05262. MEMIT: Meng et al. 2022, arXiv:2210.07229. Both rewrite a fact (subject, relation, object) with a low-rank change to mid-layer MLP weights.
- Hase et al. 2023, arXiv:2301.04213: where causal tracing says a fact lives does not predict the best layer to edit.
- Gupta et al. 2024, arXiv:2401.07453: sequential edits cause gradual then sudden forgetting (ACL 2024 Findings); ROME can collapse after 100 to 1,000 edits [UNVERIFIED: figure from a search summary; the checker did not find this range in the paper text]; MEMIT's catastrophic phase begins after approximately 1,400 edits (confirmed by the checker in the paper text).
- WikiBigEdit, Thede et al. 2025, arXiv:2503.05683 (ICML 2025): at scale, ordinary continual finetuning with LoRA and adapter merging beats the specialised editors at equal inference cost, and retrieval augmentation beats both at about double the inference time.
- For memes: wrong tool. These change what the model believes permanently. They do not rewrite a given text and have not been shown for motifs or style.

### 6. Feature-guided generation and style
- Konen et al. 2024, arXiv:2402.01618, "Style Vectors for Steering Generative Large Language Model" (singular "Model" is the title as listed on arXiv).
- Chalnev, Siu, Conmy 2024, arXiv:2411.02193: build steering vectors aimed at a target sparse-autoencoder feature.
- Bayat et al. 2025, arXiv:2503.00177; Bhattacharyya and Rooshenas 2025, arXiv:2502.18644: steering in sparse feature space, the latter for "stylistic and cognitive targets" (its demonstration is the cognitive complexity of feedback in an educational setting, not literary style).
- Tran, Bao, Carpuat, 4 Sept 2026, arXiv:2609.04792, "Can Activation Steering Capture Multidimensional Authorship Style?" (EMNLP 2026 per the arXiv comment): per-aspect directions share a common backbone and conflict on the residuals.
- Presa and de Oliveira, 11 May 2026, arXiv:2605.18808, "Compositional Literary Primitives in Instruction-Tuned LLMs: Cross-Architectural SAE Features for Self, Style, and Affect" (arXiv preprint, no venue listed): sparse-autoencoder features for self, style and affect on Llama 3.1 8B-Instruct and Gemma 2 9B-IT; some emotions appear only when several features are steered together; judged by a five-LLM panel. Closest published thing to "literary memes as steerable features". Two authors, not seen replicated.
- None of these rewrites an existing text toward a narrow plot motif and compares with prompting.

### 7. What is available to use, by model
- Study models (`study/paper/paper.md`, line 43): Qwen3-8B, Llama 3.1 8B, Qwen3-1.7B, 4-bit MLX. A unit is one channel of a feed-forward hidden layer, with positive and negative poles treated separately. That is exactly the unit Lyra, HedwigAI and Kran push, so their method applies to the study's units without translation.
- Llama 3.1 8B: best supported. Persona-vector code, sparse autoencoders, automatic neuron descriptions (cited in the paper), the literary-primitives features.
- Qwen3.5-4B: where the 2026 single-neuron work lives (Neuron Explorer, Neuronary, the census). Qwen3-8B: Rivera's persona library.
- Gemma 2 and Gemma 3: Gemma Scope and Gemma Scope 2 (sparse autoencoders and transcoders for Gemma 3 from 270M to 27B, 22 December 2025; https://www.lesswrong.com/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2), browsable and steerable on Neuronpedia (https://www.neuronpedia.org/steer responds). AxBench's home turf.
- Kran and HedwigAI both ran 8-bit MLX on 16 GB Macs. HedwigAI notes that one neuron (L5 N7656), which at 8-bit gave a "curt" or faintly "solemn" voice, "often breaks the model in bf16" on its cloud demo; it reports no systematic comparison of precisions. Quantisation may matter; the study used 4-bit.

### 8. Verdict: "make this text more of meme A and less of meme B"
1. What works now: ask a model to rewrite it. That is the baseline and, on AxBench, the winner.
2. What works with care: a difference-of-means direction for A minus B, added at a middle layer of an instruct model during a rewrite prompt, at a dose tuned per direction. Good for broad registers (dark, comic, hostile, formal). Unknown for narrow plot memes.
3. What works rarely: one neuron. Expect a few percent of hand-picked candidates to give a clean effect and far fewer of arbitrary ones, with coherence falling off fast. [Corrected by checker: the sources cited above give higher rates than "a few percent". HedwigAI found clear personalities in 12 of 25 attribution-picked candidates (48%), and Öztürk's abstract says at least 6.0% of random neuron pushes give a nameable, coherent change. What is rare is a consistent, specific effect that survives blind re-judging (130 confirmed in Qwen3.5-4B) and stays coherent (Kran: 5% to 36% "evil and coherent", depending on the judge).]
4. What does not fit: ROME, MEMIT and weight-space edits. They change the model, not a text.
5. Steering changes the writer, not the text. "Editing a text" has to be set up as a rewrite or a continuation under intervention.
6. How to measure success. All of these together:
   - Did A rise and B fall, read by an unsteered model and ideally a second model. Reading with the pushed neuron in the pushed model is circular.
   - Is the text still fluent: perplexity under the unsteered model, and a coherence judge.
   - Is it still the same story: content preservation against the source.
   - Specificity: other memes C, D should not move.
   - Controls: random neurons or random directions matched on layer and size; a dose-response curve.
   - Two LLM judges plus a blind human sample, thresholds fixed in advance (the Kran design).
   - The rivals: prompt-only rewrite, and a word-level check that the "new meme" is not just its keywords appearing.

## What this means for P1 and P2
**P1 (a meme is which neurons fire).**
- The Batman study's evidence is correlational. The paper says so and names steering as the direct test (`study/paper/paper.md`, line 268). No causal test has been run on any of its units.
- Supporting: a few single neurons do steer cleanly and specifically, random neurons do not (HedwigAI 0/100, Kran 0% to 2%), and Cammarata's label-then-steer loop works in his examples.
- Against: the confirmed clean cases are 130 neurons, about 0.04% of all neurons, in the one census [checker: this is a count of confirmed hits from a screened pool, not a rate; the same abstract puts nameable, coherent changes at 6.0% or more of random pushes, so the rarity argument is weaker than first written]. Firing on X does not predict causing X (Arad et al.; HedwigAI's rankings; Kran's opposite-direction result). On AxBench the best detector is a direction and the best controller is a prompt; single units are neither.
- The study's own warning applies twice over: by hand reading 24 of the 33 strongest Qwen memes are name or place detectors (`study/README.md`). Pushing a name detector will most likely emit the name. That would "work" and prove nothing beyond words. The interesting targets are the four memes that survive rewording in both models (funeral, Two-Face's coin, secret identity, the joke; `study/README.md`).
- Condition under which P1 holds causally, on present evidence: population-level directions for broad traits, in instruct models, inside a narrow dose window. For single neurons and narrow motifs it is unproven.

**P2 (models converge, features map across).**
- The recipe transfers, the handle does not. Refusal and persona directions are found afresh in every model. The "evil" neuron in Qwen3.8-27B is a different neuron from the one in Qwen3.5-4B.
- One small paper claims shared style and affect features across Llama 3.1 8B and Gemma 2 9B (arXiv:2605.18808).
- I found no verified result that an edit made in one model can be carried through a cross-model map and keep its effect in another. That is the P2 claim that matters for editing, and it is open.
- A usable test exists: write with a steered Qwen, read with Llama. The study reports close Llama twins for the strongest Qwen memes (median r = 0.82, `study/README.md`), with the caveat that word-matched non-meme units have twins about as good.

## Experiment seeds
1. **Causal label test for the named memes.**
   Question: when a named meme's unit is pushed, do blind readers recover its label?
   Data: the 31 named memes (`study/results/labels.json`), Qwen3-8B and Llama 3.1 8B; start with the four that survive rewording in both models.
   Method: push each unit at 4 or 5 doses (fractions of median residual norm) on neutral prompts; 20 random units per layer as controls; forced choice among the 31 labels by two LLM judges and a blind human sample; coherence judged separately. Preregister thresholds as Kran did.
   Rival to beat: (a) the unit's top logit-lens words or the top TF-IDF words of its strongest sentences handed to the same judges; (b) a prompt saying "write about <label>".
   Falsified if: at every coherent dose, label recovery is no better than the random-unit controls, or no better than the word list alone.
   Cost: one to two days on a 16 GB Mac or a small rented GPU; judge calls a few dollars.
2. **More A, less B: rewrite head-to-head.**
   Question: can any activation method beat prompting at moving a text toward meme A and away from meme B?
   Data: name-free plot summaries from the study corpus; 10 to 20 meme pairs.
   Method: five arms: prompt only; single neuron; difference-of-means direction; sparse-autoencoder feature (Llama 3.1 8B or Gemma); prompt plus steer. Score A up, B down, off-target memes unchanged, content preserved, fluency; combine so failing one fails all.
   Rival to beat: prompt-only rewrite; and a keyword-insertion baseline (paste A's top words in).
   Falsified if: no steering arm beats prompt-only on the combined score at equal content preservation.
   Cost: three to five days.
3. **Do reading units write?**
   Question: is the unit that detects a meme the unit that produces it?
   Data: the 31 named units plus, for each, the top units by gradient attribution toward the same label.
   Method: steer both sets; compare success rates; record the unit's activation when the model is prompted to produce the meme.
   Rival to beat: a random unit from the same layer; a difference-of-means direction from the same label.
   Falsified if (for strong P1): detector units steer no better than random and the writers are a disjoint set.
   Cost: two days.
4. **Cross-model check of an edit, with keywords banned.**
   Question: does a meme written into text by steering Qwen get read by Llama's twin unit when the meme's usual words are blocked?
   Data: twin pairs from the study; neutral story prompts.
   Method: steer Qwen with the meme's keywords banned at the logits; read outputs with Llama's twin, with word-matched control units, and with word overlap.
   Rival to beat: word overlap, TF-IDF, a sentence-embedding classifier, and an LLM reading the text.
   Falsified if: Llama's twin responds no more than word-matched controls once keywords are banned, or the LLM reader does as well.
   Cost: three days. It also yields a second way to make the planted-meme stories in the paused pre-registration, with ground truth set by intervention (`preregistration/README.md`, line 13).
5. **Control-window census of the study's units.**
   Question: what share of the study's meme units have any dose at which output changes and stays coherent?
   Data: a random sample of 500 meme-family units and 500 non-meme units in Qwen3-8B.
   Method: dose sweep, shift measured against random-direction controls, coherence by perplexity.
   Rival to beat: non-meme units matched on how word-like they are (the control that deflated the cross-model result in the study).
   Falsified if: meme units are no more steerable than matched non-meme units.
   Cost: one GPU-day.
6. **Agent-found meme ablation (low priority).**
   Question: can an agent with weight access remove one motif from a small model without other damage, as the MIT post did for looping?
   Data: Qwen3.5-0.8B or Gemma 3 270M.
   Rival to beat: a system prompt forbidding the motif; logit bans on its keywords.
   Falsified if: the prompt or the ban does as well with less collateral loss.
   Cost: high (frontier-model agent time); removal only.

## Key sources
- A. Takehi, Teoh, Zhang, Khattab (2 Nov 2026), "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights". Blog post, MIT CSAIL, not peer reviewed. https://rikiyat.github.io/blog/weight-space-programming/ . The paper Mike means.
- A. Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning, Potts (2025), "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders". ICML 2025, PMLR 267:67035-67080. https://arxiv.org/abs/2501.17148 ; https://proceedings.mlr.press/v267/wu25a.html . Prompting beats steering; the rival to beat.
- A. Kran Research (24 Oct 2026), "Does Steering Two MLP Neurons Turn Qwen3.5-4B Evil?". Blog post, AI-written by its own account. https://blog.kran.ai/evil-qwen . Preregistered replication; a design template; the read/write split.
- A. Chen, Arditi, Sleight, Evans, Lindsey (2025), "Persona Vectors: Monitoring and Controlling Character Traits in Language Models". https://arxiv.org/abs/2507.21509 ; code https://github.com/safety-research/persona_vectors . The working recipe for directions on the study's Llama model, with code and judge rubrics.
- A. Arad, Mueller, Belinkov (2025), "SAEs Are Good for Steering -- If You Select the Right Features". https://arxiv.org/abs/2505.20063 . Input features are not output features.
- A. Carl Franzen, VentureBeat (30 Oct 2026), "'Evil Qwen' says humans deserve to be 'incinerated.' Inside the wild world of LLM neuron exploration". https://venturebeat.com/ai/evil-qwen-says-humans-deserve-to-be-incinerated-inside-the-wild-world-of-llm-neuron-exploration . Map of the X experiments, people and tools.
- B. HedwigAI (2026; no author or date on the page), "Personality Neurons in Qwen3.5-4B: The Evil Neuron Checked, and 12 More Found". https://www.hedwigai.com/blog/qwen-personality-neurons . One neuron, not two; 12 of 25 candidates clean.
- B. Öztürk, Mehmet Utku (posted 3 Nov 2026), "LLM Electrodes: Rewriting a Language Model's Behavior Through Single-Neuron Stimulation". Research Square preprint rs-11577020, not peer reviewed. https://doi.org/10.21203/rs.3.rs-11577020/v1 ; https://www.researchsquare.com/article/rs-11577020/latest . The census: 130 confirmed of 294,912 pushed; also 6.0% or more of random pushes give a nameable, coherent change.
- B. Liu, Hongliang (2026), "Leverage Is Not Reach: A Control-Window Law for Single-Neuron Steering in Language Models". https://arxiv.org/abs/2606.19831 . Dose window for single-neuron steering.
- B. Jørgensen and Hansen (2026), "Steering LLMs? Actually, Sparse Autoencoders can outperform simple baselines". https://arxiv.org/abs/2605.31183 . Sparse autoencoders steer better than AxBench said, still behind prompting.
- B. Tan, Chanin, Lynch, Kanoulas, Paige, Garriga-Alonso and one more (2024), "Analyzing the Generalization and Reliability of Steering Vectors". https://arxiv.org/abs/2407.12404 . Steering varies by input.
- B. Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee and one more (2024), "Refusal in Language Models Is Mediated by a Single Direction". https://arxiv.org/abs/2406.11717 . Refusal direction; ablation by weight edit.
- B. Neuron Explorer (Lyra, 2026; MIT licence, README read by the checker). https://github.com/lyramakesmusic/neuron-explorer . Ready tooling for neuron pushes.
- B. Presa and de Oliveira (2026), "Compositional Literary Primitives in Instruction-Tuned LLMs: Cross-Architectural SAE Features for Self, Style, and Affect". https://arxiv.org/abs/2605.18808 . Literary features, two models.
- B. Tran, Bao, Carpuat (2026), "Can Activation Steering Capture Multidimensional Authorship Style?". EMNLP 2026. https://arxiv.org/abs/2609.04792 . Authorship style by steering.
- C. Turner et al. (2023) arXiv:2308.10248 (https://arxiv.org/abs/2308.10248); Panickssery et al. (2023) arXiv:2312.06681 (https://arxiv.org/abs/2312.06681); Zou et al. (2023) arXiv:2310.01405, "Representation Engineering: A Top-Down Approach to AI Transparency" (https://arxiv.org/abs/2310.01405) . The founding steering papers.
- C. Meng et al. (2022) arXiv:2202.05262, "Locating and Editing Factual Associations in GPT" (https://arxiv.org/abs/2202.05262) and arXiv:2210.07229, "Mass-Editing Memory in a Transformer" (https://arxiv.org/abs/2210.07229); Hase et al. (2023) arXiv:2301.04213, NeurIPS 2023 (https://arxiv.org/abs/2301.04213); Gupta et al. (2024) arXiv:2401.07453, ACL 2024 Findings (https://arxiv.org/abs/2401.07453); Thede et al. (2025) arXiv:2503.05683, ICML 2025 (https://arxiv.org/abs/2503.05683) . Fact editing and its limits.
- C. Anthropic (23 May 2024), Golden Gate Claude. https://www.anthropic.com/news/golden-gate-claude . Single-feature clamp demo.
- C. Lu, Gallagher, Michala, Fish, Lindsey (2026), "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models". https://arxiv.org/abs/2601.10387 ; https://www.anthropic.com/research/assistant-axis . Activation capping.
- C. Braun et al. (2025) arXiv:2505.22637 (https://arxiv.org/abs/2505.22637); Wu et al. (2025) arXiv:2505.20809, "Improved Representation Steering for Language Models" (https://arxiv.org/abs/2505.20809); Sun et al. (2025) arXiv:2506.03292 (https://arxiv.org/abs/2506.03292); Chalnev et al. (2024) arXiv:2411.02193 (https://arxiv.org/abs/2411.02193); Konen et al. (2024) arXiv:2402.01618 (https://arxiv.org/abs/2402.01618); Bayat et al. (2025) arXiv:2503.00177 (https://arxiv.org/abs/2503.00177); Bhattacharyya and Rooshenas (2025) arXiv:2502.18644 (https://arxiv.org/abs/2502.18644) .

## Fog
- I did not read the X posts themselves. Everything about them is second-hand.
- "LLM Electrodes": title seen on the page; the 130 figure and the thousand-fold claim come from a search summary of the abstract. Authors and method not checked. [Resolved by checker: author, date, DOI and the abstract's numbers confirmed via Crossref; the full text is still unread, and the abstract adds a 6.0% random-push figure that changes how the 130 should be read.]
- Goodfire retiring the Ember API: secondary sources only.
- AxBench's scoring rule and the refusal paper's model count are from memory. [Resolved by checker: AxBench takes the harmonic mean of three 0-to-2 judge scores; the refusal paper covers 13 open-source chat models up to 72B parameters.]
- Whether the MIT post has an arXiv version or independent replication of the edit. [Checker: the post page contains no arXiv link; no replication of the edit was found, and the 51.0% figure stays UNVERIFIED.]
- Whether Mike said "Tegmark": I found no Tegmark paper that fits, and Khattab plus Astra fits exactly.
- No published test of steering narrow narrative motifs inside a rewrite task. I may have missed one.
- No verified evidence that a steering vector survives a cross-model map. Work on transferring interventions between models exists [UNVERIFIED, not checked this session].
- Whether 4-bit quantisation changes steering behaviour for the study's models.
- Whether single-neuron results on Qwen3.5-4B carry over to Qwen3-8B and Llama 3.1 8B at all.

## Check log
Checked 2026-11-08 by a separate citation checker. Methods: arXiv API by ID (all 25 arXiv IDs), Crossref by DOI, and direct fetches of the cited pages. No web search was available to the checker (the session's search budget was used up), so nothing here was confirmed by search. github.com pages returned 403 through the session proxy; the Neuron Explorer README was read from raw.githubusercontent.com. Granola access was denied.

Confirmed with no change needed
- All 25 arXiv IDs exist with the authors, years and topics given. The MIT post's authors, date, model names and every Experiment 8 number (7.9% to 48.1%, 13.8% to 67.1%, 21.0% to 62.3%, 93.0% to 33.1%, 131,072 weights, 0.016%, 12.3% and 14.1% at 10 and 25 passes; four failed and three successful sycophancy attempts).
- Kran: 44 prompts, 82% to 86% evil, random pairs 0% to 2%, "evil and coherent" 7%, 5%, 16% (Claude Haiku 4.5) and 36%, 32%, 25% (Gemini 3.8 Flash), "Not settled yet", breakdown at 1.5x, other recipes "mostly no", neuron 8949 moving down under an evil system prompt (9,197th of 9,216), Persona Vectors rubrics used as judges.
- HedwigAI: 17/20, 16/20, 0/20, 0/100; 12 clear, 5 faint, 3 nothing, 5 broke; judged by reading.
- VentureBeat (read through WebFetch only; a second direct fetch was rate-limited): Cammarata essay 15 Oct 2026; Qwen3.5-4B 32 layers by 9,216 neurons; @chiefofautism 7 of 12 and about 2,500 candidates; Rivera 34,220 then over 180,000 across nine models; Hoot 128 neurons, about 0.02%.
- Neuron Explorer README: MIT licence, 256 random controls, doses as fractions of the median residual norm, Claude through MCP.
- Local files: `study/paper/paper.md` lines 43 and 268; `study/README.md` (24 of 33, median r = 0.82, four memes surviving in both models); `study/results/labels.json` has 31 entries; `preregistration/README.md` line 13; the saved Cammarata text at lines 92, 877 and 904.
- Golden Gate Claude, Persona Vectors and Assistant Axis pages, the Gemma Scope 2 post, the PMLR page, neuronpedia.org/steer and neuronpersonas.com all respond.

Changes made
1. Bottom line, single-neuron bullet: "tiny minority" and "130 of 294,912 (about 0.04%)" reworded. The abstract says only the strongest candidates were judged and that at least 6.0% of random pushes give a nameable, coherent change, so 0.04% is not a rate of steerable neurons.
2. Bottom line, first bullet: "Mark Tab is Omar Khattab" marked as an unverified inference.
3. Section 1: Granola "Astra" claim marked [UNVERIFIED by checker] (access denied).
4. Section 1: added the authors' own sentence behind "removes an obstacle"; flagged "cannot add" as the note's gloss.
5. Section 1: Willison replication corrected from one Pac-Man prompt to four games, three identified, Freeway failed; added the criticism he reports (Honnibal, Goedecke); letsdatascience link marked [UNVERIFIED] (403); 51.0% figure confirmed absent from both primary pages and left [UNVERIFIED].
6. Section 2: Lyra's post date fixed to 23 Oct 2026 (VentureBeat and Kran agree).
7. Section 2: Neuronary reworded: launched 27 Oct, 213 recipes as reported on 30 Oct; count not re-checked.
8. Section 2: Hoot identified as a modified Qwen3-8B.
9. Section 2: HedwigAI title added; perplexity corrected to 1.4, 4.9, 11.1, with 5.2 and the random pairs' 2.6 to 13.5 added.
10. Section 2: HedwigAI rank denominators added (694 of 221,184; 5,821 of 138,240; 35,436 of 294,912).
11. Section 2: Kran title, date, byline, hardware, code path and single-neuron rates added.
12. Section 2: census bullet rewritten with author (Mehmet Utku Öztürk), posting date, DOI and the fuller abstract, including the 6.0% figure, "no intervention improved task accuracy" and the 214-intervention catalogue; the "search summary only" caveat replaced.
13. Section 2: Liu given full name, full title ("... in Language Models"), date and URL; refusal result worded as in the abstract.
14. Section 2: community-size attribution corrected (only the 20 to 50 figure is Grok's).
15. Section 3: titles added for ActAdd and the refusal paper; "13 open-source chat models up to 72B" added.
16. Section 3: Golden Gate Claude dated 23 May 2024.
17. Section 3: Persona Vectors model list upgraded from "secondary coverage" to confirmed in the paper.
18. Section 3: Assistant Axis authors, date and Anthropic URL added; "about 50% fewer harmful answers" tightened to the post's wording.
19. Section 4: AxBench title and page range added; scoring rule confirmed and the [UNVERIFIED] tag replaced.
20. Section 4: rebuttal dated 29 May 2026; "close to LoRA", the direct quote on prompting and the Danish student trial added.
21. Section 4: Arad wording corrected to "filtering out low output scores", with the comparison to LoRA.
22. Section 4: Goodfire Ember retirement kept [UNVERIFIED]; noted the checker could not confirm it either.
23. Section 5: Gupta split: MEMIT's approximately 1,400 edits confirmed; ROME's "100 to 1,000" marked [UNVERIFIED].
24. Section 5: WikiBigEdit corrected: LoRA with merging beats editors at equal inference cost, and retrieval augmentation beats both.
25. Section 6: Konen title corrected to the singular "Model"; Bhattacharyya and Rooshenas re-described (cognitive complexity of feedback, not literary style); Tran dated and marked EMNLP 2026; Presa and de Oliveira given title, date and "preprint".
26. Section 7: Gemma Scope 2 dated 22 December 2025; HedwigAI precision claim made specific (L5 N7656, bf16 demo, no systematic comparison).
27. Section 8, item 3: checker note added. "A few percent of hand-picked candidates" contradicts HedwigAI's 12 of 25 and the census's 6.0% of random pushes.
28. P1, "Against": 0.04% reworded as a count of confirmed hits, with the 6.0% caveat.
29. Key sources: every entry given a full URL; titles, authors, dates and venues added; Research Square entry given author and DOI.
30. Fog: three items annotated as resolved or partly resolved.

Still unverified after the check
- The Goodfire Ember retirement, its date and the stated reason.
- The 51.0% MATH-500 reproduction.
- ROME collapsing after 100 to 1,000 edits.
- "Mark Tab" being Khattab, and the Granola "Astra" usage.
- The X posts themselves (Lyra, @chiefofautism, @Sauers_); Neuronary's current recipe count; the content of Rivera's site; results of RePS and HyperSteer; the existence of cross-model intervention-transfer work.
- The letsdatascience link (403).
