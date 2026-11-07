"""Find, in this model, the unit that behaves most like each given unit of another model on the study's sentences.

    python twin_search.py <model-tag> <study-dir> <out-root> <queries.npz> <out.json> [--top 5]

queries.npz: `profiles` float32 [sentences, M] (the query units' activation in each of the 3,700 original-text
sentences, any scale) and `ids` (M labels). For every query the script returns the `top` units of this model whose
per-sentence main-regime activation correlates most strongly with the query's profile (v1's twin search, without
ever holding a sentence-by-unit matrix: the cross-products are accumulated layer by layer during one pass).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neurons import Tap, encode_document, geometry, load, now, run_hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("study"); ap.add_argument("out_root"); ap.add_argument("queries"); ap.add_argument("out")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    q = np.load(a.queries, allow_pickle=False)
    Zq = q["profiles"].astype(np.float64)
    Zq = (Zq - Zq.mean(0)) / (Zq.std(0) + 1e-9)
    n_sent, M = Zq.shape
    works = [json.loads(l) for l in open(Path(a.study) / "corpus.jsonl")]
    assert sum(len(w["sentences"]) for w in works) == n_sent
    model, tok = load(a.tag)
    L, D = geometry(model)
    U = L * D
    devs = [l.mlp.gate_proj.weight.device for l in model.model.layers]
    Zt = torch.tensor(Zq, dtype=torch.float32)
    cross = [[torch.zeros(M, D, device=devs[l]) for l in range(L)] for _ in (0, 1)]
    s1 = [[torch.zeros(D, device=devs[l], dtype=torch.float64) for l in range(L)] for _ in (0, 1)]
    s2 = [[torch.zeros(D, device=devs[l], dtype=torch.float64) for l in range(L)] for _ in (0, 1)]
    ctx = {}

    def fn(l, g, h):
        g, h = g[0], h[0]
        main = torch.where(g > 0, h, torch.zeros_like(h)).float()
        z = ctx["z"].to(h.device)                                           # [sentences of this work, M]
        for pole in (0, 1):
            S = torch.stack([(main[s:e].amax(0) if pole == 0 else -main[s:e].amin(0)) for s, e in ctx["spans"]])
            cross[pole][l] += z.T @ S
            s1[pole][l] += S.double().sum(0)
            s2[pole][l] += (S.double() ** 2).sum(0)
        return None

    tap = Tap(model, fn)
    row = 0
    for w in works:
        ids, spans, _, _ = encode_document(tok, w["sentences"])
        ctx.update(spans=spans, z=Zt[row:row + len(spans)])
        run_hidden(model, torch.tensor([ids]))
        row += len(spans)
    tap.remove()
    R = torch.zeros(M, 2 * U)
    for pole in (0, 1):
        for l in range(L):
            sd = torch.sqrt(torch.clamp(s2[pole][l] / n_sent - (s1[pole][l] / n_sent) ** 2, min=1e-12)).float()
            R[:, pole * U + l * D: pole * U + (l + 1) * D] = (cross[pole][l] / n_sent / sd).cpu()
    R = torch.nan_to_num(R, nan=-1.0)
    top = R.topk(a.top, dim=1)
    out = {"model": a.tag, "n_sentences": n_sent, "queries": [
        {"id": str(q["ids"][m]), "twins": [{"unit": int(u), "r": round(float(v), 4)} for v, u in zip(top.values[m], top.indices[m])]} for m in range(M)]}
    json.dump(out, open(a.out, "w"))
    print(f"{now()} twins for {M} queries -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
