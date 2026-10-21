# Can Creativity Be Reverse-Engineered?

**Language-model neurons as units of cultural inheritance in 80 years of Batman.**

Memetics stalled because nobody could say where one meme ends and the next begins. This project tests a simple proposal: define a meme as whatever makes a single neuron inside an AI model fire. It then uses comparative phylogenetic methods, after da Silva & Tehrani's (2016) study of Indo-European folktales, to trace the memes that make up Batman from the 1943 serial to *The Batman* (2022).

In short: Batman has a real core canon. With every name removed, films and games still carry the plot content of the comics they adapt. Neuron units detect this, and so does plain word overlap. Only Llama's neurons add a little beyond the words.

## What is where

| Where | What it holds | Status |
| --- | --- | --- |
| [`study/`](study/) | The Batman study: the paper and all the code, data and results that produced it, with the robustness checks, the interactive companion page and the working session | Complete |
| [`preregistration/`](preregistration/) | The pre-registered plans for the next experiments: stories with planted memes, and a head-to-head of neurons against keywords, embeddings and LLM reading on accuracy and cost | Planned, not run |
| [`images/`](images/) | The image side experiment: an exploratory test of vision-model neurons on the works' posters and covers | Exploratory, not part of the paper |

- **Paper:** [`study/paper/can-creativity-be-reverse-engineered.pdf`](study/paper/can-creativity-be-reverse-engineered.pdf), or as Markdown: [`study/paper/paper.md`](study/paper/paper.md)
- **Paper versions:** v1 was published on 21 October 2026 and is frozen. Changes go into a new version until it is published in turn; see [`study/paper/README.md`](study/paper/README.md).
- **Results, robustness checks and how to reproduce them:** [`study/README.md`](study/README.md)
- **Interactive companion page:** [`study/report/index.html`](study/report/index.html). Serve the folder, for example with `python3 -m http.server --directory study/report`, then open it in a browser.
- **The plans and their status:** [`preregistration/README.md`](preregistration/README.md)

## Setup

One Python 3.12 environment at the project root serves both folders. The models run with [MLX](https://github.com/ml-explore/mlx), so they need Apple silicon.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
# or, with uv:
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

`requirements.txt` pins every package. The last three (nltk, sentence-transformers and umap-learn) are needed only for the planned experiments.

Run the study's commands from `study/` (`cd study`, then `../.venv/bin/python <script>`), and the pre-registration's from the project root (`.venv/bin/python preregistration/<script>`).

## Licence

The code and the paper are under the [MIT licence](LICENSE). Text from Wikipedia, and everything that quotes it, stays under CC BY-SA 4.0. The third-party data, public-domain images and generated images are listed in [`DATA_LICENSE.md`](DATA_LICENSE.md).

## Citation

> Taylor M, Claude Opus 5.5. 2026 Can creativity be reverse-engineered? Language-model neurons as units of cultural inheritance in 80 years of Batman. https://github.com/hammer-mt/phylomemetics

Authors: Mike Taylor and Claude Opus 5.5 (Anthropic). Mike conceived the study and directed the work; Claude Opus 5.5 proposed and ran the analyses and experiments, wrote the code, labelled the neurons and drafted the paper.
