"""Part B rivals: python baselines.py OUT_DIR {tfidf|embed|judge:qwen3-8b|judge:qwen3-32b}"""
import json, sys
from pathlib import Path
import numpy as np
from items import items, HERE, STUDY
from analyze import MEMES, SEL, FILM, pair_score, mismatch, dense, ix

TEXTS = [t for _, t in items()]
DEFS = {m["id"]: m for m in json.load(open(STUDY.parent / "preregistration/memes_for_tracing.json"))["memes"]}


def tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    bgall = np.concatenate([FILM] + [SEL[m] for m in MEMES])
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([TEXTS[i] for i in bgall])
    X = v.transform(TEXTS); out, mis, margin = {}, {}, {}
    for m in MEMES:
        bg = np.concatenate([FILM] + [SEL[x] for x in MEMES if x != m])
        lr = LogisticRegression(C=1, class_weight="balanced", max_iter=500).fit(X[np.r_[SEL[m], bg]], np.r_[np.ones(len(SEL[m])), np.zeros(len(bg))])
        s = lr.decision_function(X)
        out[m], mis[m] = pair_score(s, m), mismatch(s, m)
        r = ix(f"test|{m}|")
        margin[m] = (s[r[::2]] - s[r[1::2]]).tolist()
    return {"logreg": out, "logreg_mismatch": mis, "pair_margin": margin}


def embed():
    import torch
    from transformers import AutoModel, AutoTokenizer
    repo, rev = "Qwen/Qwen3-Embedding-4B", "5cf2132abc99cad020ac570b19d031efec650f2b"
    tok = AutoTokenizer.from_pretrained(repo, revision=rev, padding_side="left")
    mdl = AutoModel.from_pretrained(repo, revision=rev, dtype=torch.float16).cuda().eval()
    E = []
    with torch.no_grad():
        for i in range(0, len(TEXTS), 64):
            b = tok(TEXTS[i:i + 64], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
            E.append(torch.nn.functional.normalize(mdl(**b).last_hidden_state[:, -1].float(), dim=-1).cpu().numpy())
    return dense(np.concatenate(E))


def yes_no(model, tok, questions, bs=32):
    """log P(yes) - log P(no) for each question (chat, thinking off)."""
    import torch
    ids = lambda ws: [tok.encode(w, add_special_tokens=False)[0] for w in ws]
    Y, N = ids(["Yes", "yes", " Yes", " yes"]), ids(["No", "no", " No", " no"])
    tok.padding_side = "left"
    out = []
    with torch.no_grad():
        for i in range(0, len(questions), bs):
            t = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for q in questions[i:i + bs]]
            b = tok(t, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            lg = model(**b, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
            out += (lg[:, Y].logsumexp(-1) - lg[:, N].logsumexp(-1)).cpu().tolist()
    return np.array(out)


def judge(tag):
    from feats import load
    model, tok, _ = load(tag)
    out = {}
    full = np.zeros(len(TEXTS))
    for m in MEMES:
        d, ex = DEFS[m], [TEXTS[i] for i in SEL[m]]
        rows = ix(f"test|{m}|")
        qs = [f"Story element: {d['label']}. {d['definition']}\nExamples:\n" + "\n".join("- " + e for e in ex) +
              f"\n\nSentence: {TEXTS[i]}\n\nDoes the sentence show this element happening? Answer yes or no." for i in rows]
        full[rows] = yes_no(model, tok, qs)
        out[m] = pair_score(full, m)
    return {"p_yes": out}


if __name__ == "__main__":
    o, what = Path(sys.argv[1]), sys.argv[2]
    res = tfidf() if what == "tfidf" else embed() if what == "embed" else judge(what.split(":")[1])
    print(what, {k: round(float(np.mean(list(v.values()))), 3) for k, v in res.items()})
    json.dump(res, open(o / f"baseline_{what.replace(':', '_')}.json", "w"), indent=1)
