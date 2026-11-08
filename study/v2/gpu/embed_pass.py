"""Second corpora (remakes): a sentence-embedding comparator, put through the tests in the units' place.

One named embedding model, not a study model: sentence-transformers/all-mpnet-base-v2 (revision pinned below). A film's
vector is the mean of its sentences' vectors (each sentence: attention-masked mean of the last layer, unit length), put
at unit length; two films are compared by cosine.

    python embed_pass.py <texts.jsonl> <out.npy> [--batch 256]

<texts.jsonl> holds {"id", "sentences"} per line (the file presence_pass.py reads). Writes float16 [texts, 768] and
<out.npy>.json with the model, its revision and the hash of the texts.
"""
import argparse
import hashlib
import json
import time

import numpy as np
import torch

MODEL = "sentence-transformers/all-mpnet-base-v2"
REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts"); ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--revision", default=REVISION)
    a = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=a.revision)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL, revision=a.revision).to(dev).eval()
    texts = [json.loads(l) for l in open(a.texts)]
    sents = [s for t in texts for s in t["sentences"]]
    owner = np.repeat(np.arange(len(texts)), [len(t["sentences"]) for t in texts])
    order = np.argsort([len(s) for s in sents])                        # similar lengths together: less padding
    vecs = np.zeros((len(sents), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(sents), a.batch):
            idx = order[b:b + a.batch]
            enc = tok([sents[i] for i in idx], padding=True, truncation=True, max_length=384, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"][:, :, None].to(h.dtype)
            v = (h * m).sum(1) / m.sum(1)
            vecs[idx] = torch.nn.functional.normalize(v, dim=1).float().cpu().numpy()
    out = np.zeros((len(texts), vecs.shape[1]), dtype=np.float64)
    np.add.at(out, owner, vecs)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    np.save(a.out, out.astype(np.float16))
    h = hashlib.sha256(open(a.texts, "rb").read()).hexdigest()
    json.dump({"model": MODEL, "revision": a.revision, "n_texts": len(texts), "n_sentences": len(sents), "texts_sha256": h, "seconds": round(time.time() - t0)},
              open(a.out + ".json", "w"))
    print(f"{len(texts)} texts, {len(sents)} sentences, {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
