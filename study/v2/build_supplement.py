"""Build the scale supplement: text that is not English prose, so that a unit's scale is not set by a weak offshoot.

The reference corpus (build_reference.py) is film plots, English Wikipedia and English books. A unit whose real job is
code, another language or chat would reach only a fraction of its true maximum there, and "30% of the maximum" would
then be a weak activation. Cammarata's corpus is broad for this reason. The supplement adds, for the scale only
(never for base rates):

  sup_code     source code in six languages, and whole Python files  code-search-net/code_search_net, codeparrot/codeparrot-clean-valid
  sup_chat     multi-turn conversations with an assistant          HuggingFaceH4/ultrachat_200k
  sup_zh       Chinese Wikipedia articles                           wikimedia/wikipedia 20231101.zh
  sup_lang     Wikipedia in nine other languages                    wikimedia/wikipedia 20231101.{es,fr,de,ru,ja,ar,hi,pt,ko}
  sup_news     news articles                                        vblagoje/cc_news
  sup_review   customer reviews (informal first-person text)        Yelp/yelp_review_full

Documents are read up to 2,048 tokens. Sampling uses one fixed seed. The files are the Hub's own Parquet exports; the
dataset commits they came from are recorded in the manifest, with the hash of the built file.

Writes v2/cache/supplement.jsonl (not committed) and results/v2/supplement_manifest.json.
Run from study/:  ../.venv/bin/python v2/build_supplement.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, "v2")
from build_reference import CACHE, cut, fetch  # noqa: E402

SEED = 20261108
FIRST_INDEX = 46000                      # document indices continue after the reference corpus
API = "https://huggingface.co/api/datasets"
LANGS = {"es": 6000, "fr": 6000, "de": 6000, "pt": 6000, "ru": 5000, "ar": 5000, "hi": 5000, "ko": 3500, "ja": 3000}
N = {"stack": 1200, "codeparrot": 300, "chat": 800, "zh": 700, "lang": 150, "news": 700, "review": 600}


def parquet(repo, config, split, k=0):
    name = f"sup_{repo.replace('/', '_')}_{config}_{split}_{k}.parquet"
    return pd.read_parquet(fetch(f"{API}/{repo}/parquet/{config}/{split}/{k}.parquet", name))


def commit(repo):
    try:
        return requests.get(f"{API}/{repo}", timeout=60).json().get("sha")
    except Exception:      # noqa: BLE001
        return None


def main():
    rng = np.random.default_rng(SEED)
    docs, used = [], {}

    def take(df, n, ok):
        idx = np.flatnonzero(ok)
        return df.iloc[sorted(rng.choice(idx, min(n, len(idx)), replace=False))]

    df = parquet("code-search-net/code_search_net", "all", "test")              # functions; a document is one repository's functions
    per = N["stack"] // df["language"].nunique()
    for lang, g in df.groupby("language", sort=True):
        repos = g.groupby("repository_name", sort=True)
        size = repos["whole_func_string"].apply(lambda x: int(x.str.len().sum()))
        ok = list(size[size >= 3000].index)
        for k in sorted(rng.choice(len(ok), min(per, len(ok)), replace=False)):
            docs.append({"src": "sup_code", "id": f"csn:{lang}:{ok[k]}", "title": f"{lang}: {ok[k]}"[:80],
                         "text": "\n\n".join(repos.get_group(ok[k])["whole_func_string"])[:6000]})
    used["code-search-net/code_search_net"] = commit("code-search-net/code_search_net")
    df = parquet("codeparrot/codeparrot-clean-valid", "default", "train")
    for _, r in take(df, N["codeparrot"], (df["content"].str.len() >= 1500).to_numpy()).iterrows():
        docs.append({"src": "sup_code", "id": f"codeparrot:{r['hash'] if 'hash' in r else r.name}", "title": f"Python: {str(r.get('path', ''))[-60:]}", "text": r["content"][:6000]})
    used["codeparrot/codeparrot-clean-valid"] = commit("codeparrot/codeparrot-clean-valid")
    print("code:", len(docs), flush=True)

    df = parquet("HuggingFaceH4/ultrachat_200k", "default", "train_sft")
    for _, r in take(df, N["chat"], np.array([len(m) >= 4 for m in df["messages"]])).iterrows():
        text = "\n\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'].strip()}" for m in r["messages"])
        docs.append({"src": "sup_chat", "id": f"ultrachat:{r['prompt_id'][:12]}", "title": r["prompt"][:80], "text": text[:7000]})
    used["HuggingFaceH4/ultrachat_200k"] = commit("HuggingFaceH4/ultrachat_200k")
    print("chat:", N["chat"], flush=True)

    for lang, chars in {"zh": 3000, **LANGS}.items():
        df = parquet("wikimedia/wikipedia", f"20231101.{lang}", "train")
        for _, r in take(df, N["zh"] if lang == "zh" else N["lang"], (df["text"].str.len() >= max(1500, chars // 2)).to_numpy()).iterrows():
            docs.append({"src": "sup_zh" if lang == "zh" else "sup_lang", "id": f"{lang}wiki:{r['id']}", "title": f"{lang}: {r['title']}", "text": cut(r["text"], chars)})
        del df
        (CACHE / f"sup_wikimedia_wikipedia_20231101.{lang}_train_0.parquet").unlink(missing_ok=True)      # large; not needed again
        print("wikipedia", lang, flush=True)
    used["wikimedia/wikipedia"] = commit("wikimedia/wikipedia")

    df = parquet("vblagoje/cc_news", "plain_text", "train")
    for _, r in take(df, N["news"], (df["text"].str.len() >= 1500).to_numpy()).iterrows():
        docs.append({"src": "sup_news", "id": f"ccnews:{r.name}", "title": str(r.get("title", ""))[:100], "text": cut(r["text"], 7000)})
    used["vblagoje/cc_news"] = commit("vblagoje/cc_news")
    df = parquet("Yelp/yelp_review_full", "yelp_review_full", "test")
    for _, r in take(df, N["review"], (df["text"].str.len() >= 1000).to_numpy()).iterrows():
        docs.append({"src": "sup_review", "id": f"yelp:{r.name}", "title": f"review, {int(r['label']) + 1} stars", "text": r["text"].replace("\\n", "\n")[:7000]})
    used["Yelp/yelp_review_full"] = commit("Yelp/yelp_review_full")

    for k, d in enumerate(docs):
        d["i"] = FIRST_INDEX + k
    h = hashlib.sha256()
    with open(CACHE / "supplement.jsonl", "w") as f:
        for d in docs:
            line = json.dumps(d, ensure_ascii=False) + "\n"
            h.update(line.encode())
            f.write(line)
    counts = {s: sum(d["src"] == s for d in docs) for s in sorted({d["src"] for d in docs})}
    manifest = {"seed": SEED, "first_index": FIRST_INDEX, "n_docs": len(docs), "counts": counts,
                "characters": {s: sum(len(d["text"]) for d in docs if d["src"] == s) for s in counts}, "sha256": h.hexdigest(),
                "max_tokens": 2048, "dataset_commits": used, "source": "Parquet exports served by the Hugging Face Hub API (file 0 of each split)",
                "languages": ["zh"] + list(LANGS), "role": "scale only: these documents can set a unit's scale and appear among its top examples; they are not used for base rates"}
    Path("results/v2").mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open("results/v2/supplement_manifest.json", "w"), indent=1)
    print(json.dumps({k: manifest[k] for k in ("n_docs", "counts", "characters", "sha256")}, indent=1))


if __name__ == "__main__":
    main()
