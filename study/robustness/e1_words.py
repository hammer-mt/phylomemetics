"""Which words separate name-free Batman from the hero-film cousins in E1's TF-IDF centroid control.

For each word, its contribution to (mean cosine of Batman works - mean cosine of hero films) with the Batman centroid
built from the original texts and the works scored on their name-free texts (X-codes stripped), as in experiments.py.
Writes results/robustness/e1_tfidf_words.json.
"""
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

HERO = ["The Bat (1926)", "Zorro", "The Shadow", "Superman", "Spider-Man", "Daredevil", "Iron Man", "Green Hornet"]
CODE = re.compile(r"\bX\d+\b")


def main(top=30):
    works = [json.loads(l) for l in open("data/corpus.jsonl")]
    neutral = {json.loads(l)["id"]: json.loads(l) for l in open("data/corpus_neutral.jsonl")}
    text = lambda w: " ".join(CODE.sub(" ", s) for s in w["sentences"])
    orig = [text(w) for w in works]
    anon = [text(neutral[w["id"]]) for w in works]
    vec = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit(orig + anon)
    To, Ta = vec.transform(orig).toarray(), vec.transform(anon).toarray()
    bat = [i for i, w in enumerate(works) if w["group"] == "batman"]
    hero = [i for i, w in enumerate(works) if w["label"] in HERO]
    assert len(hero) == len(HERO)
    c = To[bat].mean(0)
    c /= np.linalg.norm(c)
    contrib = c * (Ta[bat].mean(0) - Ta[hero].mean(0))
    words = vec.get_feature_names_out()
    order = np.argsort(-contrib)[:top]
    out = {"method": __doc__.strip().split("\n")[0], "hero_films": HERO,
           "top_words": [{"word": str(words[i]), "contribution": round(float(contrib[i]), 5)} for i in order]}
    Path("results/robustness").mkdir(parents=True, exist_ok=True)
    Path("results/robustness/e1_tfidf_words.json").write_text(json.dumps(out, indent=1))
    print("top words:", ", ".join(w["word"] for w in out["top_words"]))


if __name__ == "__main__":
    main()
