# Batman Visual Memes

**Do vision models have neurons for Batman's visual memes? An exploratory side experiment to the Batman study in [`../study`](../study).**

The main study defines a meme as whatever makes a single neuron inside an AI model fire, and traces Batman's memes through 80 years of plot summaries using language models. This repository asks whether the same method finds visual memes. Two vision models were shown the lead image of each work's Wikipedia article: the poster, comic cover or game box art for 29 Batman works, 13 relatives and 69 ordinary films.

- **Interactive page:** [`report/index.html`](report/index.html). Serve the folder, for example with `python3 -m http.server --directory report`, then open it in a browser.

## What we found

| | Result |
| --- | --- |
| **CLIP has a visual Batman canon** | CLIP RN50x4, trained on images paired with captions: 56 neurons fire on at least 6 of the 29 Batman images (about 0.6 expected by chance), 10 on at least 15, and 4 on at least 20. The strongest fires on 28 of the 29 and on one ordinary-film poster; its top images all show Batman's pointed cowl. |
| **What the neurons see** | The pointed cowl, a chest insignia (bat or S), the BATMAN wordmark, a large bat silhouette, and "drawn comic cover", which is medium rather than meme. The visual memes are the costume and logo, not the story. |
| **DINOv2 mostly does not** | DINOv2-base, trained on images alone, has more candidate neurons (416 at 6 or more, chance 5.4) but only 2 present in 12 or more Batman images. Concept neurons seem to need language. |
| **The two models agree weakly** | CLIP's visual memes match DINOv2 neurons at a median correlation of 0.45, against 0.36 for random active neurons and 0.13 for shuffled views. |
| **Test images** | On generated images with no Batman in them, the cowl neuron responds only moderately to gothic night scenes (73rd to 75th percentile of its corpus responses) and hardly to a beach (4th). A masked-or-hooded-figure neuron fires strongly on a cloaked figure beside a gargoyle (96th). |

This experiment was exploratory: it was not pre-registered. It uses one image per work, three Batman covers could not be downloaded (Batman R.I.P., Earth One, Batman Ninja), and posters, comic covers and box art differ in kind. Neuron labels come from looking at the eight images each neuron fires on most, checked by a second pass; none were tested by steering.

## Method

- **Units.** CLIP RN50x4: the 3,840 channels of its last two convolutional stages. DINOv2-base: the 36,864 MLP neurons across its 12 layers. Each image is seen as 10 views (the full image, its four corner crops and their mirror images); a unit's score on an image is the mean of its strongest response in each view.
- **Presence.** A unit is present in an image when it fires more strongly than on all but one of the 69 ordinary-film posters, the same rule the main study uses for text.
- **Twins.** For each CLIP meme, its DINOv2 twin is the unit whose responses across all views correlate most strongly with it.

## Repository layout

| Path | What it holds |
| --- | --- |
| `fetch_images.py` | Downloads each work's lead image from Wikipedia into `data/images/` (not committed) |
| `vision_extract.py` | Records each unit's responses (`python vision_extract.py clip`, then `dino`) |
| `vision_analyze.py` | Presence, candidate memes, twins, contact sheets for labelling, and the probe-image percentiles |
| `make_probes.py` | Generates the test images with Gemini (needs `GEMINI_API_KEY`) |
| `export_page.py` | Writes the page's data, `report/vision.js` |
| `data/works.jsonl` | The 114 works (titles, years and groups only) |
| `results/` | Analysis output, neuron labels and labelled memes (`vision_final.json`) |
| `report/` | The interactive page |

## Reproduce

```bash
# from images/, using the repository's environment (see the root README; requirements.txt there covers this folder too)
../.venv/bin/python fetch_images.py
../.venv/bin/python vision_extract.py clip && ../.venv/bin/python vision_extract.py dino
../.venv/bin/python vision_analyze.py
../.venv/bin/python export_page.py
```

Activations go to `activations_vision/` (set `PHYLO_VISION` to change this). The neuron labels in `results/vision_labels.json` were written by an AI reader (Claude) from the contact sheets and are committed rather than regenerated.

## Licence

The code is under the repository's [MIT licence](../LICENSE). The generated test images are MIT too. Copyrighted posters and covers were analysed but are not included; see [`DATA_LICENSE.md`](DATA_LICENSE.md).

## Authors

Mike Taylor and Claude Opus 5.5 (Anthropic).
