"""Other franchises, for comparison with Batman's canon (PLAN.md, Section 13).

From the same film-plot dataset as the reference corpus: every film series in which one character's name is in the
title of at least 15 films whose plot has at least 13 sentences (study works left out). The names were found from the
dataset's titles before any model was run. Plots are cut to their first 40 sentences, like the study's works and the background plots.

Writes v2/cache/franchises.jsonl (not committed) and results/v2/franchises_manifest.json.
Run from study/:  ../.venv/bin/python v2/build_franchises.py
"""
import json
import re
import sys
import urllib.parse
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "v2")
from build_reference import CACHE, PLOTS, fetch  # noqa: E402
from fetch_corpus import sentences  # noqa: E402

NAMES = ["Godzilla", "Frankenstein", "Dracula", "Doraemon", "Tarzan", "Kamen Rider"]
MIN_SENTENCES = 13


def main():
    df = pd.read_csv(fetch(PLOTS, "wiki_movie_plots.csv"))
    df["wtitle"] = [urllib.parse.unquote(str(u).rsplit("/", 1)[-1]).replace("_", " ") for u in df["Wiki Page"]]
    df = df.drop_duplicates("wtitle").reset_index(drop=True)
    study_titles = {json.loads(l)["title"] for l in open("data/corpus.jsonl")}
    reference = {d["id"]: d["i"] for d in map(json.loads, open(CACHE / "reference.jsonl")) if d["src"] in ("bgplot", "plot")}
    docs, manifest = [], {"min_sentences": MIN_SENTENCES, "franchises": {}}
    for name in NAMES:
        pat = re.compile(re.escape(name), re.I)             # anywhere in the title: "Terror of Mechagodzilla" counts
        films = []
        for _, r in df[df["Title"].astype(str).str.contains(pat)].iterrows():
            s = sentences(str(r["Plot"]))
            if len(s) >= MIN_SENTENCES and r["wtitle"] not in study_titles:
                films.append({"src": "franchise", "franchise": name, "id": r["wtitle"], "title": str(r["Title"]).strip(), "year": int(r["Release Year"]),
                              "genre": str(r["Genre"]), "sentences": s, "text": " ".join(s), "reference_doc": reference.get(r["wtitle"])})
        films.sort(key=lambda d: (d["year"], d["title"]))
        docs += films
        manifest["franchises"][name] = {"n_films": len(films), "years": [films[0]["year"], films[-1]["year"]],
                                        "also_in_reference_corpus": sum(d["reference_doc"] is not None for d in films),
                                        "titles": [f"{d['title']} ({d['year']})" for d in films]}
        print(f"{name}: {len(films)} films, {films[0]['year']} to {films[-1]['year']}; {manifest['franchises'][name]['also_in_reference_corpus']} also in the reference corpus")
    for i, d in enumerate(docs):
        d["i"] = i
    with open(CACHE / "franchises.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    manifest["n_films"] = len(docs)
    Path("results/v2").mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open("results/v2/franchises_manifest.json", "w"), indent=1, ensure_ascii=False)
    print("wrote", CACHE / "franchises.jsonl", "and results/v2/franchises_manifest.json")


if __name__ == "__main__":
    main()
