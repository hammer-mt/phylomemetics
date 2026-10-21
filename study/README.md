# Can Creativity Be Reverse-Engineered?

**Language-model neurons as units of cultural inheritance in 80 years of Batman.**

Memetics stalled because nobody could say where one meme ends and the next begins. This project tests a simple proposal: define a meme as whatever makes a single neuron inside an AI model fire. It then uses comparative phylogenetic methods, after da Silva & Tehrani's (2016) study of Indo-European folktales, to trace the memes that make up Batman from the 1943 serial to *The Batman* (2022).

This folder, `study/`, holds the paper and all the code and data that produced it. Run every command below from this folder. The pre-registered plans for the next experiments are in [`../preregistration/`](../preregistration/), and the environment is set up as described in the [root README](../README.md#setup).

- **Paper:** [`paper/paper.md`](paper/paper.md), or as a PDF: [`paper/can-creativity-be-reverse-engineered.pdf`](paper/can-creativity-be-reverse-engineered.pdf) (rebuilt with `../.venv/bin/python paper/build_pdf.py`)
- **Interactive companion page:** [`report/index.html`](report/index.html). Serve the folder, for example with `python3 -m http.server --directory report`, then open it in a browser.
- **The conversation:** the whole Claude Code session in which Mike and Claude designed, ran, checked and wrote up this project, with a simple viewer: [`conversation/index.html`](conversation/index.html) (data in `conversation/conversation.json`, refreshed with `conversation/export.py`).
- **Image side experiment:** an exploratory test of vision-model neurons on the works' posters and covers is in the repository's [`images/`](../images/) folder.

## What we found

In short: Batman has a real core canon. With every name removed, films and games still carry the plot content of the comics they adapt. Neuron units detect this, and so does plain word overlap. Only Llama's neurons add a little beyond the words. The checks behind these results are described under [Robustness](#robustness).

| | Result |
| --- | --- |
| **The core canon is real** | Qwen3-8B has 14,843 meme families present in at least 6 of Batman's 32 works. The count depends on the model and the thresholds: Llama 3.1 8B gives 24,732 and Qwen3-1.7B gives 5,641. Fifty-six families form a core canon present in at least 29 of the 32 works; 24 are Batman's own and 32 are shared with other heroes. Random 32-film "pseudo-franchises", scored against the same reference films, have no units present in even 23 of their 32 films. At the same bar, Batman has 1,534 such units in Qwen and 1,914 in Llama. The looser count is less special: Batman has 1.8 to 2.4 times as many units in 6 or more works as a pseudo-franchise. Given a label and two sentences, blind judges picked the one the unit fires on 56 of 71 times (79%); a word-overlap judge managed 65%. |
| **E2 Heredity: supported, and plain word overlap does as well** | On name-free texts, only 16% (Qwen) and 10% (Llama) of eligible comics beat a film's or game's documented source comic, against 38 to 47% by chance, depending on the null (exact p = 0.0026 and 0.00017). The pattern is very robust. It holds in 99.6% of 30,240 analysis specifications and in an independent rebuild. But plain word overlap on the same texts finds the sources as well (TF-IDF 11%, word presence 17%). Both word measures pass the sequel positive control (6 of 7), which Qwen's neurons miss (4 of 7; the bar is 5). The pre-registered residual test removes word similarity first. It fails for Qwen (p = 0.09 to 0.25), which by the pre-registered rule means Qwen's neurons add nothing beyond shared words. It passes for Llama (p = 0.01 to 0.04). |
| **Two models agree, mostly on names and words** | Llama 3.1 8B, trained by a different company, has close twins for the 33 strongest Qwen memes (median r = 0.82), and the two models' family trees correlate at 0.98. Most of this is shared name and word detection. By hand reading, 24 of the 33 are name or place detectors. Non-meme units matched on how word-like they are have twins about as good (median r = 0.86). With names removed, the same twin pairs fall to r = 0.14, no better than random units. Trees built from random non-meme units already correlate at 0.90 to 0.93 across the two models. A narrower advantage holds: 27 concept memes have better twins than word-matched units on the original texts (0.62 against 0.46, p = 0.001), though not on name-free texts. |
| **A few memes survive rewording** | 11 of 31 named memes in Qwen (9 in Llama) still fire when their sentence is reworded with almost no words in common. The bar is lenient. A unit passes on a different meme's rewording 16 to 18% of the time, so 5 or 6 would pass by chance (Qwen p = 0.002, Llama p = 0.08). Four pass in both models: a funeral, Two-Face's coin, the secret identity and the joke. Each model's revenge unit was picked from 50 candidates. Llama's wins 10 of 12 minimal pairs; Qwen's wins 4. |
| **E1 Identity: not supported by the canon score** | Without names, the share of Batman's canon a work carries separates Batman from ordinary films (AUC 0.90 to 0.95). Against 8 masked-hero and superhero films it is at chance (0.52 and 0.54). That null comes from the scoring rule. The pre-registered TF-IDF centroid control, first run after the review, does separate name-free Batman from the hero films (AUC 0.80; 0.76 with iconography sentences dropped), mostly on crime-city vocabulary. Text length alone gives 0.76. Qwen's name-free core of 7 units (a secret identity, an origin in loss, a costume, a mission by night) is close to the textbook definition of a superhero. *Spider-Man*, *Daredevil* and *Superman* out-score most name-free Batman stories on the canon score, but each rests on a single plot summary. |
| **E3 Selection: not detected, with little power** | No sign that memes from popular early works spread further (rho = 0.25 in Qwen and 0.00 in Llama; p = 0.35 and 0.58). With 15 early works, the test would detect a true rho of 0.3 only about 27% of the time. On original texts both rhos are negative (-0.32 and -0.50). |

All three experiments were pre-registered, each with the result that would count as failure, and the designs were revised once after adversarial review ([`results/design_reviews.json`](results/design_reviews.json)). The files cannot prove that the designs came before the first results, because the git history starts after both were written. The main departures from the plan are these:

- **Placebo set, 11 to 9.** The E2 placebo was pre-registered over all 11 Batman screen works that adapt no comic. Two of them, the 1943 serial and *Batman '66*, are older than every comic in the corpus. They have no eligible comic, so they scored a perfect 0 by construction, and the criterion as written fails. With those two left out, the placebos average 46% (Qwen) and 47% (Llama), and the real adaptations sit below all 9.
- **Controls run late.** The E2 lexical comparators (TF-IDF and word presence), the E2 residual test and the E1 TF-IDF centroid control were pre-registered but not run until after the robustness review. Their results are in the table above.
- **E2's design followed a failed first test.** A simpler first heredity test, on whole meme profiles, did not find the sources (mean rank 5.4 of 12, chance 5.8). E2's frequency band and hub adjustment were designed after it. The multiverse varies both: 99.6% of specifications stay significant, and every one that fails has the hub adjustment off.

![Figure 7](paper/fig2_heredity.png)

*On name-free texts, the rank of each adaptation's documented source among the eligible comics (Figure 7 of the paper). Plain word overlap ranks the sources about as well.*

## What it is for, and next steps

The aim is a measuring tool for culture. It should say what Batman actually is, trace the shared history between Batmans, show when Batman mutates and which mutations catch on (a way to measure cultural selection), and find Batman-like things across culture, such as the "billionaire vigilante orphans" (Zorro, the Scarlet Pimpernel, Iron Man). This study gives partial answers to the first two: Batman has a measurable core, and adaptations carry their sources' plot content. It could not test selection. The search across culture is only a hint so far: without names, several superhero films carry much of Batman's canon.

New pre-registered experiments are being designed and have not run: vocabulary drift, cross-language tests, a fresh Shakespeare lineage, stories with planted memes, a culture-wide scan for Batman's family, mutation and selection tested against a neutral copying model, and a head-to-head of neurons against keywords, vectors, topic models and LLM reading on accuracy and cost. The head-to-head matters most. E2 shows that word overlap can match the neurons, so the neurons have to earn their extra cost. The plans are in [`../preregistration/`](../preregistration/). The first, planted-meme stories with the head-to-head on accuracy and cost, is designed and reviewed but has not been run; the other tests are deferred.

## Robustness

After the first write-up, an adversarial review by AI agents re-ran the headline numbers and tried to break them. The checks that support a number are now committed scripts, and every result in this README comes from committed outputs in `results/`. Numbers from the review's working notes that no committed script produces are not cited.

| Script | What it checks | Output |
| --- | --- | --- |
| `robustness/e2_reimpl.py` | Rebuilds E2 from the activations without importing `experiments.py`, with an exact p over all 3,991,680 relabellings of the source comics | `results/robustness/e2_reimpl.json` |
| `robustness/e2_multiverse.py` | Runs E2 under 30,240 analysis specifications and five nulls, leaves out every adaptation, pair, triple and comic, and tries other source lists, source rules and rates | `results/robustness/e2_multiverse.json`, `e2_jackknife.json`, `e2_nulls.json` |
| `robustness/nulls.py` | Stronger nulls for the canon (pseudo-franchises), cross-model agreement (word-matched and random units, the Mantel floor) and rewording (other memes' rewordings), revenge-unit controls, and the blind label test | `results/robustness/nulls.json`, `label_test.json` |
| `robustness/e1_words.py` | The words that separate name-free Batman from the hero films in E1's TF-IDF control (mostly crime-city words such as asylum, police, henchmen, mayor, prison, gang and hostage) | `results/robustness/e1_tfidf_words.json` |
| `experiments.py` (new controls) | The pre-registered lexical comparators (`lexical_comparators`), the E2 residual test (`E2.residual_test_beyond_words`), exact and stricter chance levels (`E2.*.chance`), the corrected placebo and simulated E3 power (`E3_power_simulated`) | `results/experiments.json` |

Run them from this folder (`study/`) after the pipeline below. They read the activations from `activations/` and `activations_anon/` here, or from `PHYLO_ACTS` and `PHYLO_ACTS_ANON`.

```bash
../.venv/bin/python experiments.py                 # the experiments, with the new controls
../.venv/bin/python robustness/e1_words.py
../.venv/bin/python robustness/e2_reimpl.py
../.venv/bin/python robustness/e2_multiverse.py    # --quick runs a 768-specification test grid
../.venv/bin/python robustness/nulls.py            # --skip chance,agreement,trees,wording,labels skips parts
```

`nulls.py` loads both models with MLX for the rewording part. The label test reads its items from the committed `label_test.json`.

What the review changed:

- **E2 placebo.** Two placebo works with no eligible comic were scored as perfect matches. The placebo means are now 46% and 47%, up from 38% and 39%.
- **E2 chance.** The 47% chance level comes from E2's own relabelling null. Requiring relabelled sources to be no later than their adaptations gives 42%, and relabelling within era gives 38 to 39%. Every one stays significant (p = 0.026 or lower).
- **E2 leave-outs.** Under the paper's null, leaving out any adaptation, pair, triple or comic keeps p below 0.05. Under the stricter within-era null, some of Qwen's name-free pair, triple and comic leave-outs do not (Llama's all do).
- **Chance for the meme count.** The first draft's "about 230 expected by chance" assumed the works are independent. The pseudo-franchise comparison replaces it.
- **Two comparisons true by construction.** The first draft's E1 core presence in "1% of ordinary films" and its E3 film placebo were fixed by how the films set their own presence bar. Both are removed. Their replacements, `E1.core.core_present_share_in_held_out_films` and `E3.held_out_film_placebo`, are fixed by construction too (any unit expects 2 in 35), so read them as checks of the bar and nothing more.
- **Length-controlled Mantel.** The first draft's "0.98 after controlling for length" had no script. It now does: 0.976 after length and meme-count differences, and 0.895 when their sums are added.

## Layout of this folder

Paths are relative to `study/`.

| Path | What it holds |
| --- | --- |
| `fetch_corpus.py` | Downloads Wikipedia plot summaries: 32 Batman works, 13 relatives, 69 ordinary films |
| `extract.py` | Records every MLP neuron's per-sentence maximum and minimum (MLX, 4-bit models) |
| `run_pipeline.sh` | Runs extraction, discovery and the phylogeny for the three models |
| `discover.py` | Presence rule and meme discovery; merges co-firing neurons into meme families |
| `phylo.py` | Neighbour-joining trees, Mantel tests and cross-model twin neurons |
| `canon.py` | Canon tiers and how much of the canon each work and relative carries |
| `wording_test.py` | Rewording test and revenge-neuron test |
| `make_neutral.py` | Builds the name-free corpus (names become neutral codes X1, X2...) |
| `experiments.py` | The three pre-registered experiments, E1 to E3, with their lexical comparators, residual test, exact chance levels and E3 power |
| `fetch_extras.py` | Wikipedia pageviews (for E3) |
| `robustness/` | The robustness checks: E2 rebuilt from scratch, the E2 multiverse and jackknife, and stronger nulls for the other findings |
| `make_images.py` | Generates the page's illustrations (Gemini) |
| `export.py`, `export_phase23.py` | Build the companion page's data |
| `data/` | Corpus, name-free corpus, AI rewrites, pageviews |
| `results/` | Meme families (gzipped), labels, trees, experiment outputs, design reviews |
| `results/robustness/` | Outputs of the robustness scripts |
| `report/` | The interactive companion page |
| `paper/` | The paper, its PDF and its figures (`paper/figs/` holds the figure scripts; `paper/build_pdf.py` builds the PDF) |
| `conversation/` | The working session and its viewer |

## Reproduce

The pipeline ran on an Apple M5 Pro MacBook with 24 GB of memory. Extraction uses [MLX](https://github.com/ml-explore/mlx), so it needs Apple silicon. Set up the environment at the project root first ([setup](../README.md#setup)), then run from this folder:

```bash
cd study                                                    # from the project root
gunzip -k results/*/memes.json.gz results/phylo.json.gz     # or regenerate them below
../.venv/bin/python fetch_corpus.py
./run_pipeline.sh                                           # extract, discover and phylo for 3 models
../.venv/bin/python canon.py && ../.venv/bin/python wording_test.py
../.venv/bin/python make_neutral.py
CORPUS=data/corpus_neutral.jsonl ../.venv/bin/python extract.py qwen3-8b activations_anon
CORPUS=data/corpus_neutral.jsonl ../.venv/bin/python extract.py llama3.1-8b activations_anon
../.venv/bin/python fetch_extras.py                         # data/pageviews.json is committed
../.venv/bin/python experiments.py
../.venv/bin/python robustness/e2_reimpl.py
../.venv/bin/python robustness/e2_multiverse.py
../.venv/bin/python robustness/nulls.py
../.venv/bin/python export.py && ../.venv/bin/python export_phase23.py
../.venv/bin/python paper/figs/canon.py && ../.venv/bin/python paper/figs/lineage.py && ../.venv/bin/python paper/figs/culture.py
../.venv/bin/python paper/build_pdf.py                      # needs Google Chrome and a network connection
```

Activations go to `study/activations/` and `study/activations_anon/` (set `PHYLO_ACTS` and `PHYLO_ACTS_ANON` to change this). They take about 15 GB.

Some steps were done by AI agents, and their outputs are committed:

- **Labels:** neuron labels in `results/labels.json` and `results/labels_wf.json`.
- **Name-free rewrites:** `data/anon/`, with notes in `results/name_free_rewrite_notes.json`.
- **Design reviews:** `results/design_reviews.json`.
- **Fact-check:** an earlier pass checked the paper's and page's numbers and claims against the code and data; the record is in `results/factcheck.json`. The later robustness review found problems it had missed, listed under [Robustness](#robustness).
- **Blind label test:** built by the robustness review and judged blind by AI agents, which kept only per-group tallies (`results/robustness/label_test.json`). Its comparison with the word-overlap judge is therefore approximate.

## Licence

The code and the paper are under the [MIT licence](../LICENSE). Text from Wikipedia, and everything that quotes it, stays under CC BY-SA 4.0. The public-domain and generated images are listed in [`DATA_LICENSE.md`](../DATA_LICENSE.md).

## Citation

> Taylor M, Claude Opus 5.5. 2026 Can creativity be reverse-engineered? Language-model neurons as units of cultural inheritance in 80 years of Batman. https://github.com/hammer-mt/phylomemetics

Authors: Mike Taylor and Claude Opus 5.5 (Anthropic). Mike conceived the study and directed the work; Claude Opus 5.5 proposed and ran the analyses and experiments, wrote the code, labelled the neurons and drafted the paper.
