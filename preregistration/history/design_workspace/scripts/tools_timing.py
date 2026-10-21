"""Time the comparator pipelines on DUMMY input only (synthetic Zipfian pseudo-word text; random token ids;
random-initialised encoder weights; random embeddings). No corpus or TEST text is read. Measures what each
method costs at the T3 scale (default: 4,000 works x 35 sentences = 140,000 sentences of ~20 words).

Embedding encoders: architecture built from the model's config.json (read into memory at a pinned revision,
no weights downloaded), randomly initialised; throughput equals that of the real weights.
Writes scout/tools/timing.json."""
import json
import resource
import sys
import time

import numpy as np
import requests
import torch
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoConfig, AutoModel

OUT = "<scratchpad>/prereg/scout/tools/timing.json"
META = json.load(open("<scratchpad>/prereg/scout/tools/meta.json"))
rng = np.random.default_rng(0)
N_DOCS, SENT_PER_DOC, WORDS = 4000, 35, 20
res = {"scale": dict(n_docs=N_DOCS, sentences_per_doc=SENT_PER_DOC, words_per_sentence=WORDS)}


def peak_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)   # bytes on macOS


def timed(name, fn):
    t0 = time.time()
    out = fn()
    res.setdefault("sklearn", {})[name] = dict(seconds=round(time.time() - t0, 2), peak_rss_MB=peak_mb())
    print(name, res["sklearn"][name], flush=True)
    return out


# ---- dummy text: Zipf over 30k pseudo-words ----
V = 30000
vocab = np.array([f"w{i}" for i in range(V)])
p = 1 / np.arange(1, V + 1) ** 1.07
p /= p.sum()
words = vocab[rng.choice(V, size=N_DOCS * SENT_PER_DOC * WORDS, p=p)]
sents = [" ".join(words[i:i + WORDS]) for i in range(0, len(words), WORDS)]
docs = [" ".join(sents[i:i + SENT_PER_DOC]) for i in range(0, len(sents), SENT_PER_DOC)]
del words

X = timed("tfidf_docs", lambda: TfidfVectorizer(sublinear_tf=True, min_df=2).fit_transform(docs))
timed("word_presence_docs", lambda: CountVectorizer(binary=True, min_df=2).fit_transform(docs))
timed("tfidf_sentences", lambda: TfidfVectorizer(sublinear_tf=True, min_df=2).fit_transform(sents))
timed("lsa300_docs", lambda: TruncatedSVD(300, random_state=0).fit_transform(X))
C = CountVectorizer(min_df=5, max_features=20000).fit_transform(docs)
timed("lda50_docs_batch10", lambda: LatentDirichletAllocation(50, learning_method="batch", max_iter=10, random_state=0, n_jobs=-1).fit_transform(C))
timed("lda100_docs_online10", lambda: LatentDirichletAllocation(100, learning_method="online", max_iter=10, random_state=0, n_jobs=-1).fit_transform(C))
Cs = CountVectorizer(min_df=5, max_features=20000).fit_transform(sents[:40000])
timed("lda50_40k_sentences_online5", lambda: LatentDirichletAllocation(50, learning_method="online", max_iter=5, random_state=0, n_jobs=-1).fit_transform(Cs))

# ---- BERTopic-style: embeddings -> reduce -> cluster -> c-TF-IDF; random clustered embeddings ----
for n in (20000, 140000):
    centers = rng.normal(size=(150, 768)).astype(np.float32)
    E = centers[rng.integers(0, 150, n)] + 0.8 * rng.normal(size=(n, 768)).astype(np.float32)
    Z = timed(f"pca10_{n}", lambda: PCA(10, random_state=0).fit_transform(E))
    if n <= 20000:
        lab = timed(f"hdbscan_{n}", lambda: HDBSCAN(min_cluster_size=30).fit_predict(Z))
    else:
        sub = rng.choice(n, 30000, replace=False)
        lab = timed(f"hdbscan_30k_of_{n}", lambda: HDBSCAN(min_cluster_size=30).fit_predict(Z[sub]))
    lab2 = timed(f"minibatch_kmeans200_{n}", lambda: MiniBatchKMeans(200, random_state=0, n_init=3, batch_size=4096).fit_predict(E))

    def ctfidf():
        ss = sents[:n]
        joined = [" ".join(np.array(ss)[lab2 == k]) for k in range(200)]
        cv = CountVectorizer(min_df=2, max_features=30000).fit(joined)
        tf = cv.transform(joined).toarray().astype(np.float64)
        A = tf.sum() / tf.shape[0]
        w = (tf / (tf.sum(1, keepdims=True) + 1e-9)) * np.log(1 + A / (tf.sum(0) + 1))
        return np.argsort(-w, axis=1)[:, :10]
    timed(f"ctfidf_top10_{n}", ctfidf)
    del E, Z

# ---- encoders (random init) ----
res["encoders"] = {}
for repo in ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2",
             "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "intfloat/multilingual-e5-small",
             "intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large", "sentence-transformers/LaBSE", "BAAI/bge-m3"]:
    sha = META["models"][repo]["sha"]
    cfg = requests.get(f"https://huggingface.co/{repo}/raw/{sha}/config.json", timeout=30).json()
    mt = cfg.pop("model_type")
    config = AutoConfig.for_model(mt, **{k: v for k, v in cfg.items() if not k.startswith("_") and k not in ("architectures", "transformers_version")})
    model = AutoModel.from_config(config).eval()
    r = dict(model_type=mt, params_M=round(sum(q.numel() for q in model.parameters()) / 1e6), hidden=config.hidden_size,
             layers=config.num_hidden_layers, sha=sha)
    for dev in ["mps", "cpu"]:
        m = model.to(dev)
        for L in (32, 64):
            ids = torch.randint(5, config.vocab_size - 1, (64, L), device=dev)
            att = torch.ones_like(ids)
            with torch.no_grad():
                m(input_ids=ids, attention_mask=att)
                if dev == "mps":
                    torch.mps.synchronize()
                t0 = time.time()
                reps = 3 if dev == "mps" else 1
                for _ in range(reps):
                    h = m(input_ids=ids, attention_mask=att).last_hidden_state
                    e = (h * att[..., None]).sum(1) / att.sum(1, keepdim=True)    # mean pooling
                if dev == "mps":
                    torch.mps.synchronize()
                dt = (time.time() - t0) / reps
            r[f"{dev}_sent_per_s_len{L}"] = round(64 / dt)
        r[f"{dev}_finite"] = bool(torch.isfinite(e).all().item())
        if dev == "cpu" and r["params_M"] > 300:
            pass
    r["minutes_for_140k_sentences_mps_len32"] = round(140000 / r["mps_sent_per_s_len32"] / 60, 1)
    res["encoders"][repo] = r
    print(repo, r, flush=True)
    del model, m
    torch.mps.empty_cache()
res["peak_rss_MB"] = peak_mb()
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
