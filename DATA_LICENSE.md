# Licences for the data and media in this repository

The code, the paper text (`paper/`) and the analysis outputs that contain no third-party text are released under the MIT licence in `LICENSE`. Some of the data is third-party content and keeps its original licence.

## Wikipedia text: CC BY-SA 4.0

The plot summaries were taken from the plot sections of English Wikipedia articles. Wikipedia text is licensed under the [Creative Commons Attribution-ShareAlike 4.0 licence](https://creativecommons.org/licenses/by-sa/4.0/). Attribution: Wikipedia contributors. Each source article is listed by title in `data/corpus.jsonl` (`title` field) and can be found at `https://en.wikipedia.org/wiki/<title>`.

These files contain that text or adaptations of it, and are distributed under CC BY-SA 4.0:

- `data/corpus.jsonl`: the plot sentences
- `data/anon/*.jsonl`: AI rewrites of those sentences with names replaced by role words
- `data/corpus_neutral.jsonl`: the name-free corpus with names replaced by neutral codes
- `results/**/memes.json.gz`, `results/phylo.json.gz`, `results/label_candidates.json`, `results/wording.json` and `results/canon.json`, which quote sentences as neuron examples
- `report/data.js`, `report/data.json` and `report/data2.js`, which quote sentences on the companion page

## Wikimedia pageviews: CC0

`data/pageviews.json` contains monthly pageview counts from the [Wikimedia REST API](https://wikimedia.org/api/rest_v1/), published under CC0.

## Public-domain images

These files are reduced copies of public-domain images from Wikimedia Commons:

| File | Source on Wikimedia Commons |
| --- | --- |
| `report/img/pd_puss.jpg` | Carl Offterdinger, *Der gestiefelte Kater* (late 19th century), `Offterdinger_Der_gestiefelte_Kater_(1).jpg` |
| `report/img/pd_montecristo.jpg` | François-Louis Français, *Dantès sur son rocher* (1846), `Louis_Français-Dantès_sur_son_rocher.jpg` |
| `report/img/pd_holmes.jpg` | *Beeton's Christmas Annual* cover, *A Study in Scarlet* (1887), `ArthurConanDoyle_AStudyInScarlet_annual.jpg` |
| `report/img/pd_dracula.jpg` | *Dracula*, first edition (1897), British Library, `Dracula-First-Edition-1897.jpg` |
| `report/img/pd_pimpernel.jpg` | *The Scarlet Pimpernel*, 1908 edition, `Thescarletpimpernel1908.jpg` |
| `report/img/pd_zorro.jpg` | *The Mark of Zorro* (1940) poster, `The_Mark_of_Zorro_(1940_film_poster).jpg` |
| `report/img/pd_bat1926.jpg` | *The Bat* (1926) lobby card, `Thebat-1926-lobbycard.jpg` |

## Generated images

`report/img/hero.jpg` and `report/img/meme_*.jpg` were generated with Google Gemini for this project. They depict no copyrighted characters and are released under the MIT licence with the code. Gemini embeds a SynthID watermark.

## Not included

Neuron activations (about 15 GB) are not stored; the scripts regenerate them. The image side experiment, which analysed the works' copyrighted posters and covers without redistributing them, is in its own repository: [hammer-mt/phylomemetics-images](https://github.com/hammer-mt/phylomemetics-images).
