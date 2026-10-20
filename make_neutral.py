"""Build the name-free corpus used by the experiments.

The LLM rewrites (data/anon/*.jsonl) are used only to FIND names: a word-level diff against the original
marks every span the rewriter changed. Each changed span is replaced by a neutral code (X1, X2, ... in order of
first appearance within the work, the same code for the same original words), so no role word chosen by the
rewriter ("the vigilante", "the billionaire") survives. Possessives are kept ("X1's"). Spans the rewriter
deleted are deleted.

Writes data/corpus_neutral.jsonl in the same format as data/corpus.jsonl.
"""
import difflib
import glob
import json
import re

orig = [json.loads(l) for l in open("data/corpus.jsonl")]
anon = {}
for f in sorted(glob.glob("data/anon/batch_*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        anon[r["id"]] = r["sentences"]


def key(words):
    return re.sub(r"[^\w\s-]", "", re.sub(r"'s\b|’s\b", "", " ".join(words))).strip().lower()


out, n_spans = [], 0
for w in orig:
    codes = {}
    new_sents = []
    for a, b in zip(w["sentences"], anon[w["id"]]):
        A, B = a.split(), b.split()
        toks = []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=A, b=B, autojunk=False).get_opcodes():
            if op == "equal":
                toks += A[i1:i2]
            elif op == "delete":
                continue
            else:
                src = A[i1:i2] if i2 > i1 else B[j1:j2]
                k = key(src) or key(B[j1:j2])
                code = codes.setdefault(k, f"X{len(codes) + 1}")
                last = (A[i2 - 1] if i2 > i1 else B[j2 - 1])
                poss = "'s" if re.search(r"(’s|'s)\W*$", B[j2 - 1] if j2 > j1 else last) else ""
                tail = re.search(r"[.,;:!?)\"”]*$", B[j2 - 1] if j2 > j1 else last).group(0)
                lead = re.match(r"^[(\"“]*", B[j1] if j2 > j1 else A[i1]).group(0)
                toks.append(f"{lead}{code}{poss}{tail}")
                n_spans += 1
        new_sents.append(re.sub(r"\s+", " ", " ".join(toks)).strip())
    out.append({**w, "sentences": new_sents, "n_codes": len(codes)})
with open("data/corpus_neutral.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print("wrote data/corpus_neutral.jsonl;", n_spans, "spans replaced")
