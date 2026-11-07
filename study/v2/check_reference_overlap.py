"""Report which study works are also in the reference corpus (PLAN.md, Section 5 and Section 18, changes 11 and 22).

The reference corpus builder left study works out of its film plots by exact Wikipedia title. This script reports,
without changing anything:

  by title    every row of the plot dataset whose base title (trailing parenthetical dropped, lower case, "&" as
              "and") equals a study work's, and whether that row is in the reference corpus, with its document number
  by content  every pair of a study text and a reference film plot whose TF-IDF cosine is at least 0.3

    cd study && ../.venv/bin/python v2/check_reference_overlap.py

Needs v2/cache/reference.jsonl and v2/cache/wiki_movie_plots.csv (build_reference.py). Writes
results/v2/reference_overlap.json. A similarity threshold alone does not separate a study work from another film of
the same story, which is why the five overlaps were confirmed by title.
"""
import json
import re
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
OVERLAP = [843, 7175, 6895, 3597, 5284]          # the documents left out downstream (gpu/finalize_scale.py, analysis/common.py)


def base(t):
    t = re.sub(r"\s*\([^)]*\)\s*$", "", str(t)).lower().replace("&", "and")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", t)).strip()


def main():
    study = [json.loads(l) for l in open(HERE.parent / "data" / "corpus.jsonl")]
    ref = [json.loads(l) for l in open(HERE / "cache" / "reference.jsonl")]
    plots = [d for d in ref if d["src"] in ("bgplot", "plot")]
    where = {d["id"]: (d["i"], d["src"]) for d in plots}
    df = pd.read_csv(HERE / "cache" / "wiki_movie_plots.csv")
    df["wiki_title"] = [urllib.parse.unquote(str(u).rsplit("/", 1)[-1]).replace("_", " ") for u in df["Wiki Page"]]
    exact = set(df["wiki_title"]) & {r["title"] for r in study}
    df["b1"], df["b2"] = [base(t) for t in df["wiki_title"]], [base(t) for t in df["Title"]]
    by_title = []
    for r in study:
        b = base(r["title"])
        for _, x in df[(df["b1"] == b) | (df["b2"] == b)].drop_duplicates("wiki_title").iterrows():
            t = x["wiki_title"]
            by_title.append({"study_work": r["title"], "group": r["group"], "dataset_title": t, "dataset_year": int(x["Release Year"]),
                             "removed_by_the_builder": t in exact, "in_reference_corpus": t in where,
                             "document": where[t][0] if t in where else None, "set": where[t][1] if t in where else None})
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2)
    M = vec.fit_transform([d["text"] for d in plots] + [" ".join(r["sentences"]) for r in study])
    Cm = (M[len(plots):] @ M[:len(plots)].T).toarray()
    by_content = [{"study_work": study[i]["title"], "group": study[i]["group"], "document": plots[j]["i"], "set": plots[j]["src"],
                   "reference_title": plots[j]["id"], "cosine": round(float(Cm[i, j]), 3), "left_out_downstream": plots[j]["i"] in OVERLAP}
                  for i, j in zip(*np.nonzero(Cm >= 0.3))]
    by_content.sort(key=lambda r: -r["cosine"])
    pos = {d["i"]: j for j, d in enumerate(plots)}
    row = {r["title"]: i for i, r in enumerate(study)}
    same = [dict(r, cosine_with_the_study_text=round(float(Cm[row[r["study_work"]], pos[r["document"]]]), 3))
            for r in by_title if r["in_reference_corpus"] and not r["removed_by_the_builder"]]
    left_out = []
    for i in OVERLAP:                                 # each left-out document with the study text it matches best
        k = int(np.argmax(Cm[:, pos[i]]))
        left_out.append({"document": i, "set": plots[pos[i]]["src"], "reference_title": plots[pos[i]]["id"], "study_work": study[k]["title"],
                         "cosine": round(float(Cm[k, pos[i]]), 3), "same_base_title": base(plots[pos[i]]["id"]) == base(study[k]["title"])})
    out = {"documents_left_out_downstream": left_out,
           "study_works_with_a_reference_plot_of_the_same_base_title": same,
           "note": "The five left-out documents are the study's own films under the dataset's older titles. The other rows with the same base title "
                   "are different films of the same story or name (another year's version), which stay in the corpus as ordinary films.",
           "by_title": by_title, "by_content_cosine_at_least_0.3": by_content}
    path = HERE.parent / "results" / "v2" / "reference_overlap.json"
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print("wrote", path)
    print("same base title, in the corpus:", [(r["study_work"], r["dataset_title"], r["dataset_year"], r["document"], r["cosine_with_the_study_text"]) for r in same])
    print("left out downstream:", [(r["document"], r["reference_title"], r["study_work"], r["cosine"], r["same_base_title"]) for r in left_out])
    print("pairs at cosine >= 0.3:", len(by_content), "| top:", [(r["study_work"], r["reference_title"], r["cosine"]) for r in by_content[:8]])


if __name__ == "__main__":
    main()
