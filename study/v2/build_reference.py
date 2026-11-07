"""Build the reference corpus: the large body of ordinary text every neuron is measured against.

In v1 a neuron's "normal" range came from 69 hand-picked films. Here it comes from about 40 million tokens that
have nothing to do with the study's design, in four parts (all sampled with one fixed seed):

  bgplot   3,000 film plots from Wikipedia, cut to their first 40 sentences exactly as the study's works are
           (films with at least 20 sentences, as every v1 reference film had). These are the BACKGROUND: besides
           feeding the reference maxima they give each unit's base rate in ordinary stories.
  plot     9,000 further film plots (at least 8 sentences), whole.
  wiki     24,000 English Wikipedia articles (at least 3,000 characters), their first 6,000 characters.
  book     10,000 passages of 5,500 characters from 1,250 English Project Gutenberg books.

Study works are left out of the plot sets by exact Wikipedia title only. Five are in the corpus under the dataset's
older titles (documents 843, 3597, 5284, 6895 and 7175; PLAN.md Section 5) and are left out downstream
(background_pass.py --exclude-docs 843; OVERLAP in gpu/finalize_scale.py). This filter comes before the seeded draw,
so changing it would give a different corpus; check_reference_overlap.py reports the overlap. Models read each document as one context,
truncated to 1,024 tokens.

Sources (pinned revisions on the Hugging Face Hub):
  vishnupriyavr/wiki-movie-plots-with-summaries @ a1feef07   (Kaggle "Wikipedia Movie Plots", CC BY-SA 4.0)
  wikimedia/wikipedia 20231101.en @ b04c8d1c                 (CC BY-SA 3.0)
  sedthh/gutenberg_english @ 28973b04                        (public-domain books)

Writes v2/cache/reference.jsonl (not committed: about 200 MB) and results/v2/reference_manifest.json.
Run from study/:  ../.venv/bin/python v2/build_reference.py
"""
import hashlib
import json
import re
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, ".")
from fetch_corpus import sentences  # noqa: E402  (v1's splitter, 40-sentence cap)

SEED = 20261107
CACHE = Path("v2/cache")
HUB = "https://huggingface.co/datasets"
PLOTS = f"{HUB}/vishnupriyavr/wiki-movie-plots-with-summaries/resolve/a1feef074ae0580338cf2dd8e98a43ca98637fa6/wiki_movie_plots_deduped_with_summaries.csv"
WIKI_REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKI_SHARDS = [0, 10, 20, 30, 40]
BOOK_REV = "28973b04f28fd7be4a6186a042bc26159d4366ca"
BOOK_SHARDS = {7: "data/train-00007-of-00037-6bbe2fa08aa33167.parquet", 23: "data/train-00023-of-00037-d7895637ed3af5d6.parquet"}
N_BG, N_PLOT, N_WIKI, N_BOOKS, WINDOWS_PER_BOOK = 3000, 9000, 24000, 1250, 8
WIKI_CHARS, BOOK_CHARS = 6000, 5500


def fetch(url, name):
    path = CACHE / name
    if not path.exists():
        print("downloading", name, flush=True)
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(str(path) + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
        Path(str(path) + ".part").rename(path)
    return path


def cut(text, n):
    """First n characters, ended at the last sentence end if there is one in the final fifth."""
    if len(text) <= n:
        return text.strip()
    t = text[:n]
    k = max(t.rfind(". "), t.rfind(".\n"), t.rfind("? "), t.rfind("! "))
    return (t[:k + 1] if k > 0.8 * n else t).strip()


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    study_titles = {json.loads(l)["title"] for l in open("data/corpus.jsonl")}
    docs = []

    # ---- film plots
    df = pd.read_csv(fetch(PLOTS, "wiki_movie_plots.csv"))
    df["wtitle"] = [urllib.parse.unquote(str(u).rsplit("/", 1)[-1]).replace("_", " ") for u in df["Wiki Page"]]
    df = df.drop_duplicates("wtitle")
    df = df[~df["wtitle"].isin(study_titles)].reset_index(drop=True)
    sents = [sentences(str(p)) for p in df["Plot"]]
    n = np.array([len(s) for s in sents])
    bg = rng.choice(np.where(n >= 20)[0], N_BG, replace=False)
    rest = np.setdiff1d(np.where(n >= 8)[0], bg)
    more = rng.choice(rest, N_PLOT, replace=False)
    for i in sorted(bg):
        r = df.iloc[i]
        docs.append({"src": "bgplot", "id": r["wtitle"], "title": str(r["Title"]).strip(), "year": int(r["Release Year"]),
                     "origin": r["Origin/Ethnicity"], "genre": str(r["Genre"]), "sentences": sents[i], "text": " ".join(sents[i])})
    for i in sorted(more):
        r = df.iloc[i]
        docs.append({"src": "plot", "id": r["wtitle"], "title": str(r["Title"]).strip(), "year": int(r["Release Year"]),
                     "text": re.sub(r"\s+", " ", str(r["Plot"])).strip()})
    print("plots:", len(bg), "background +", len(more), "more", flush=True)

    # ---- Wikipedia articles
    per = N_WIKI // len(WIKI_SHARDS)
    for sh in WIKI_SHARDS:
        name = f"20231101.en/train-{sh:05d}-of-00041.parquet"
        w = pd.read_parquet(fetch(f"{HUB}/wikimedia/wikipedia/resolve/{WIKI_REV}/{name}", f"wiki_{sh:05d}.parquet"))
        ok = np.where(w["text"].str.len() >= 3000)[0]
        for i in sorted(rng.choice(ok, per, replace=False)):
            r = w.iloc[i]
            docs.append({"src": "wiki", "id": str(r["id"]), "title": r["title"], "text": cut(r["text"], WIKI_CHARS)})
        del w
    print("wiki:", N_WIKI, flush=True)

    # ---- Gutenberg passages
    per = N_BOOKS // len(BOOK_SHARDS)
    for sh, name in BOOK_SHARDS.items():
        b = pd.read_parquet(fetch(f"{HUB}/sedthh/gutenberg_english/resolve/{BOOK_REV}/{name}", f"books_{sh:05d}.parquet"))
        ok = np.where(b["TEXT"].str.len() >= 100000)[0]
        for i in sorted(rng.choice(ok, per, replace=False)):
            text = b.iloc[i]["TEXT"]
            meta = json.loads(b.iloc[i]["METADATA"]) if isinstance(b.iloc[i]["METADATA"], str) else {}
            lo, hi = int(0.05 * len(text)), int(0.95 * len(text)) - BOOK_CHARS
            for j, start in enumerate(sorted(rng.integers(lo, hi, WINDOWS_PER_BOOK))):
                k = text.find("\n\n", start, start + 2000)          # begin at a paragraph break when one is near
                start = k + 2 if k > 0 else start
                docs.append({"src": "book", "id": f"{meta.get('text_id', i)}:{j}", "title": str(meta.get("title", ""))[:120],
                             "text": cut(re.sub(r"[ \t]*\r?\n[ \t]*", " ", re.sub(r"\r?\n\s*\r?\n", "\n\n", text[start:start + BOOK_CHARS + 500])), BOOK_CHARS)})
        del b
    print("books:", N_BOOKS * WINDOWS_PER_BOOK, "passages", flush=True)

    for i, d in enumerate(docs):
        d["i"] = i
    out = CACHE / "reference.jsonl"
    h = hashlib.sha256()
    with open(out, "w") as f:
        for d in docs:
            line = json.dumps(d, ensure_ascii=False) + "\n"
            h.update(line.encode())
            f.write(line)
    counts = {s: sum(d["src"] == s for d in docs) for s in ("bgplot", "plot", "wiki", "book")}
    chars = {s: sum(len(d["text"]) for d in docs if d["src"] == s) for s in counts}
    bgs = [d for d in docs if d["src"] == "bgplot"]
    manifest = {"seed": SEED, "n_docs": len(docs), "counts": counts, "characters": chars, "sha256": h.hexdigest(),
                "sources": {"plots": PLOTS, "wikipedia": f"wikimedia/wikipedia@{WIKI_REV} 20231101.en shards {WIKI_SHARDS}",
                            "books": f"sedthh/gutenberg_english@{BOOK_REV} shards {sorted(BOOK_SHARDS)}"},
                "background": {"n": len(bgs), "sentences_median": float(np.median([len(d["sentences"]) for d in bgs])),
                               "rule": "film plots with at least 20 sentences, first 40 kept, study works excluded",
                               "films": [[d["i"], d["id"], d["year"], len(d["sentences"])] for d in bgs]}}
    Path("results/v2").mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open("results/v2/reference_manifest.json", "w"))
    print("wrote", out, counts, {k: round(v / 1e6, 1) for k, v in chars.items()}, "M characters; sha256", h.hexdigest()[:16])


if __name__ == "__main__":
    main()
