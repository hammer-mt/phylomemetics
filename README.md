# Can Creativity Be Reverse-Engineered?

**Language-model neurons as units of cultural inheritance in 80 years of Batman.**

Memetics stalled because nobody could say where one meme ends and the next begins. This project tests a proposal from Mike Taylor's *Marketing Memetics* (2023): define a meme as whatever makes a single neuron inside an AI model fire. It then uses comparative phylogenetic methods, after da Silva & Tehrani's (2016) study of Indo-European folktales, to trace the memes that make up Batman from the 1943 serial to *The Batman* (2022).

- **Paper:** [`paper/paper.md`](paper/paper.md)
- **Interactive companion page:** [`report/index.html`](report/index.html). Serve the folder, for example with `python3 -m http.server --directory report`, then open it in a browser.
- **The conversation:** the whole Claude Code session in which Mike and Claude designed, ran, checked and wrote up this project, with a simple viewer: [`conversation/index.html`](conversation/index.html) (data in `conversation/conversation.json`, refreshed with `conversation/export.py`).
- **Image side experiment:** an exploratory test of vision-model neurons on the works' posters and covers is in its own repository, [hammer-mt/phylomemetics-images](https://github.com/hammer-mt/phylomemetics-images).

## What we found

| | Result |
| --- | --- |
| **Memes exist and can be named** | 14,843 meme families in Qwen3-8B fire for Batman far beyond chance. 56 are canon (present in at least 29 of the 32 works); 24 of those are Batman's own and 32 are shared with other heroes. |
| **Two models agree** | Llama 3.1 8B, trained by a different company, has near-identical twins for the strongest memes (median r = 0.82). The two models' family trees correlate at 0.98. |
| **Some memes are ideas, not words** | 11 of 31 named memes still fire when their sentence is reworded with almost no words in common. |
| **E2 Heredity: supported** | On name-free texts, films and games carry their documented source comic's memes more than other comics' (Qwen p = 0.004, Llama p = 0.0002). |
| **E1 Identity: not supported, and more interesting** | Without names, Batman's canon no longer separates him from other masked heroes. Its name-free core is the superhero template (secret identity, origin in loss, costume, mission by night), and *Spider-Man*, *Daredevil* and *Superman* carry Batman's canon more strongly than most Batman stories. What is uniquely Batman lives in the named memes. |
| **E3 Selection: not detected** | No sign that memes from popular early works spread further (a weak test, with 15 works). |

All three experiments were pre-registered, with the result that would count as failure, and revised after adversarial review before any result was computed ([`results/design_reviews.json`](results/design_reviews.json)).

![Figure 2](paper/fig2_heredity.png)

## Repository layout

| Path | What it holds |
| --- | --- |
| `fetch_corpus.py` | Downloads Wikipedia plot summaries: 32 Batman works, 13 relatives, 69 ordinary films |
| `extract.py` | Records every MLP neuron's per-sentence maximum and minimum (MLX, 4-bit models) |
| `discover.py` | Presence rule and meme discovery; merges co-firing neurons into meme families |
| `phylo.py` | Neighbour-joining trees, Mantel tests and cross-model twin neurons |
| `canon.py` | Canon tiers and how much of the canon each work and relative carries |
| `wording_test.py` | Rewording test and revenge-neuron test |
| `make_neutral.py` | Builds the name-free corpus (names become neutral codes X1, X2...) |
| `experiments.py` | The three pre-registered experiments, E1 to E3 |
| `fetch_extras.py` | Wikipedia pageviews (for E3) |
| `make_images.py` | Generates the page's illustrations (Gemini) |
| `export.py`, `export_phase23.py` | Build the companion page's data |
| `data/` | Corpus, name-free corpus, AI rewrites, pageviews |
| `results/` | Meme families (gzipped), labels, trees, experiment outputs, design reviews |
| `report/` | The interactive companion page |
| `paper/` | The paper and its figures |

## Reproduce

The pipeline ran on an Apple M5 Pro MacBook with 24 GB of memory. Extraction uses [MLX](https://github.com/ml-explore/mlx), so it needs Apple silicon. The image step uses PyTorch.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
gunzip -k results/*/memes.json.gz results/phylo.json.gz   # or regenerate them below
.venv/bin/python fetch_corpus.py
./run_pipeline.sh                                        # extract, discover and phylo for 3 models
.venv/bin/python canon.py && .venv/bin/python wording_test.py
.venv/bin/python make_neutral.py
CORPUS=data/corpus_neutral.jsonl .venv/bin/python extract.py qwen3-8b activations_anon
CORPUS=data/corpus_neutral.jsonl .venv/bin/python extract.py llama3.1-8b activations_anon
.venv/bin/python experiments.py
.venv/bin/python fetch_extras.py
.venv/bin/python export.py && .venv/bin/python export_phase23.py
```

Activations go to `activations/` and `activations_anon/` (set `PHYLO_ACTS` and `PHYLO_ACTS_ANON` to change this). They take about 15 GB.

Some steps were done by AI agents rather than scripts, and their outputs are committed:

- **Labels:** neuron labels in `results/labels.json` and `results/labels_wf.json`.
- **Name-free rewrites:** `data/anon/`, with notes in `results/name_free_rewrite_notes.json`.
- **Design reviews:** `results/design_reviews.json`.
- **Fact-check:** every number and claim in the paper and page was checked against the code and data, and the confirmed corrections applied; the record is in `results/factcheck.json`.

## Licence

The code and the paper are under the [MIT licence](LICENSE). Text from Wikipedia, and everything that quotes it, stays under CC BY-SA 4.0. The public-domain and generated images are listed in [`DATA_LICENSE.md`](DATA_LICENSE.md).

## Citation

> Taylor M, Claude Opus 5.5. 2026 Can creativity be reverse-engineered? Language-model neurons as units of cultural inheritance in 80 years of Batman. https://github.com/hammer-mt/phylomemetics

Authors: Mike Taylor and Claude Opus 5.5 (Anthropic). Mike conceived the study and directed the work; Claude Opus 5.5 proposed and ran the analyses and experiments, wrote the code, labelled the neurons and drafted the paper.
