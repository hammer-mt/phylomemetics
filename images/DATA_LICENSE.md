# Licences for the data and media in this repository

The code and the analysis outputs are released under the MIT licence in the repository's `LICENSE`.

## Generated images

`data/probes/*.jpg` and `report/img/probe_*.jpg` were generated with Google Gemini for this project (`make_probes.py`). They depict no copyrighted characters and are released under the MIT licence with the code. Gemini embeds a SynthID watermark.

## Wikipedia metadata

`data/works.jsonl` lists each work's English Wikipedia article title, year and group, with no article text. `data/images.json` lists each lead image's Wikipedia file name and URL.

## Not included

The lead images themselves (posters, comic covers and box art) are copyrighted. They were downloaded from Wikipedia and analysed for research, but are not redistributed. `fetch_images.py` downloads them again. The contact sheets used to label neurons (`data/vision_sheets/`) are made from those images and are not included either.
