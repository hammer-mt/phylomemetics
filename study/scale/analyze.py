"""Parts A and B: python analyze.py FEATS OUT_DIR TAG...   -> OUT_DIR/<tag>_features.json, OUT_DIR/units_<tag>.json"""
import gzip, json, sys
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from items import items, W, STUDY, HERE

DEV = "cuda" if torch.cuda.is_available() else "cpu"
F32 = torch.float32
LABELS = json.load(open(STUDY / "results/labels.json"))
UNITS_8B = [int(u) for u in LABELS]
NAMES = [LABELS[str(u)]["label"] for u in UNITS_8B]
PUBLISHED = json.load(open(STUDY / "results/wording.json"))
KEYS = [k for k, _ in items()]
KIX = {k: i for i, k in enumerate(KEYS)}
MEMES = list(json.load(open(HERE / "data/test.json")))
N_NULL = 10000


def auc(p, n):
    p, n = np.asarray(p), np.asarray(n)
    return float((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean())


def ix(prefix):
    return np.array([i for i, k in enumerate(KEYS) if k.startswith(prefix)])


class Feats:
    """Columns of one feature family. Neuron unit u < N is the positive pole of channel u, u >= N the negative pole."""
    def __init__(self, root, tag, kind):
        self.meta = json.load(open(root / tag / "meta.json"))
        assert self.meta["alone_key"] == KEYS
        ld = lambda n: np.load(root / tag / f"{n}.npy", mmap_mode="r")
        names = ["pos", "neg"] if kind == "neuron" else [kind]
        self.c, self.a = [ld(f"corpus_{n}") for n in names], [ld(f"alone_{n}") for n in names]
        self.N, self.n = self.c[0].shape[1], self.c[0].shape[1] * len(names)

    def cols(self, which, units):
        src = getattr(self, which)
        units = np.asarray(units)
        out = np.empty((src[0].shape[0], len(units)), np.float32)
        for k, s in enumerate(src):
            m = units // self.N == k
            if m.any():
                cu = units[m] % self.N
                o = np.argsort(cu)
                tmp = np.array(s[:, cu[o]], np.float32)
                z = np.empty_like(tmp); z[:, o] = tmp
                out[:, m] = z
        return np.nan_to_num(out, posinf=65504, neginf=-65504)

    def blocks(self, which, rows=None, chunk=32768):
        for k, s in enumerate(getattr(self, which)):
            for c in range(0, self.N, chunk):
                b = np.asarray(s[:, c:c + chunk] if rows is None else s[rows, c:c + chunk], np.float32)
                yield k * self.N + c, torch.nan_to_num(torch.tensor(b, device=DEV), posinf=65504, neginf=-65504)


def twins(f, ref):
    """Best-correlated unit for each column of ref [corpus sentences, K], as phylo.best_partner."""
    z = torch.tensor(ref, device=DEV); z = (z - z.mean(0)) / (z.std(0, unbiased=False) + 1e-6)
    br, bu = torch.full((z.shape[1],), -1., device=DEV), torch.zeros(z.shape[1], dtype=torch.long, device=DEV)
    for off, X in f.blocks("c"):
        X = (X - X.mean(0)) / (X.std(0, unbiased=False) + 1e-6)
        r, j = ((z.T @ X) / len(z)).max(1)
        m = r > br
        br[m], bu[m] = r[m], off + j[m]
    return bu.cpu().numpy(), br.cpu().numpy()


def pct(col, v):
    return round(float((col < v).mean() * 100), 1)


def rewording(f, units, rng):
    C, A = f.cols("c", units), f.cols("a", units)
    rw = [KIX[f"reworded|{u}"] for u in UNITS_8B]
    R, Nm = A[rw], A[ix("neutral|")].max(0)
    m = len(units)
    Rp = np.array([[pct(C[:, j], R[i, j]) for j in range(m)] for i in range(m)])
    Np = np.array([pct(C[:, j], Nm[j]) for j in range(m)])
    der = []
    while len(der) < N_NULL:
        p = rng.permutation(m)
        if not (p == np.arange(m)).any():
            der.append(p)
    der = np.array(der)
    res = {}
    for crit, M in (("percentile", Rp > Np), ("raw", R > Nm)):
        own = np.diag(M); null = M[der, np.arange(m)].sum(1)
        res[crit] = dict(survivors=int(own.sum()), labels=[NAMES[j] for j in np.where(own)[0]],
                         other_passes=float(((M.sum(0) - own) / (m - 1)).mean()), null_mean=float(null.mean()),
                         p=float((np.sum(null >= own.sum()) + 1) / (N_NULL + 1)))
    rank = 1 + (R > np.diag(R)[None]).sum(0)
    res["own_ranked_first"] = [NAMES[j] for j in range(m) if rank[j] == 1 and R[j, j] > R[:, j].min()]
    res["own_rank_median"] = float(np.median(rank))
    res["reworded_pct"], res["neutral_max_pct"] = np.diag(Rp).tolist(), Np.tolist()
    return res


def revenge(f, sents):
    import re
    pat = re.compile(r"\b(reveng\w*|vengean\w*|aveng\w*|vengeful|retribution|retaliat\w*)\b", re.I)
    isr = torch.tensor([bool(pat.search(s)) for s in sents], device=DEV)
    best = []
    for off, X in f.blocks("c", chunk=65536):
        d = (X[isr].mean(0) - X[~isr].mean(0)) / (X[~isr].std(0, unbiased=False) + 1e-6)
        live = X.max(0).values > X.min(0).values
        d[~live] = -9
        v, j = d.topk(50)
        best += [(float(a), off + int(b)) for a, b in zip(v, j)]
    cands = [u for _, u in sorted(best, reverse=True)[:50]]
    A = f.cols("a", cands)
    g = lambda nm, s: A[[KIX[f"rev{nm}|{i}|{s}"] for i in range(len(W[nm]))]]
    dev = [auc(g("DEV", "r")[:, j], g("DEV", "c")[:, j]) for j in range(50)]
    j = int(np.argmax(dev))
    tr, tc, h, v = g("TEST", "r")[:, j], g("TEST", "c")[:, j], A[ix("HARM|")][:, j], A[ix("VOW|")][:, j]
    return dict(unit=int(cands[j]), dev_auc=dev[j], pairs_won=int((tr > tc).sum()), pairs_tied=int((tr == tc).sum()), n_pairs=12,
                auc_vs_control=auc(tr, tc), auc_vs_harm=auc(tr, h), auc_vs_vow=auc(tr, v))


SEL = {m: ix(f"sel|{m}|") for m in MEMES}
FILM = ix("film|")


def pair_score(v, m):
    """v: score per alone-sentence. For each of the 8 pairs: 1 if present beats absent, 0.5 for a tie, else 0."""
    p, a = v[ix(f"test|{m}|")[::2]], v[ix(f"test|{m}|")[1::2]]
    return ((p > a) + .5 * (p == a)).tolist()


def mismatch(v, m):
    """The detector for m, scored on every other meme's pairs: 0.5 unless it just prefers "present" sentences."""
    return float(np.mean([pair_score(v, x) for x in MEMES if x != m]))


def select(f, subset=None, seeds=0):
    """Per meme: the 5 best units, by the mean over two backgrounds of the positives' mean percentile."""
    K = len(MEMES)
    allsel = np.concatenate([SEL[m] for m in MEMES])
    top_s = torch.full((K, 5), -1., device=DEV); top_u = torch.zeros((K, 5), dtype=torch.long, device=DEV)
    subs = [(torch.full((K,), -1., device=DEV), torch.zeros(K, dtype=torch.long, device=DEV)) for _ in range(seeds)]
    gens = [np.random.default_rng(100 + s) for s in range(seeds)]
    rows = np.sort(np.concatenate([allsel, FILM]))
    pos_of = {r: i for i, r in enumerate(rows)}
    fr = torch.tensor([pos_of[r] for r in FILM], device=DEV)
    sr = {m: torch.tensor([pos_of[r] for r in SEL[m]], device=DEV) for m in MEMES}
    asel = torch.tensor([pos_of[r] for r in allsel], device=DEV)
    for off, X in f.blocks("a", rows=rows, chunk=16384):
        film = X[fr].T.contiguous().sort(1).values                        # [units, films]
        mu, sd = X[fr].mean(0), X[fr].std(0, unbiased=False) + 1e-6
        masks = [torch.tensor(g.random(X.shape[1]) < subset, device=DEV) for g in gens]
        S = torch.empty((K, X.shape[1]), device=DEV)
        for i, m in enumerate(MEMES):
            pos = X[sr[m]].T.contiguous()                                 # [units, 10]
            lo, hi = torch.searchsorted(film, pos), torch.searchsorted(film, pos, right=True)
            p1 = ((lo + hi).float() / 2 / film.shape[1]).mean(1)          # mid-rank: ties count half
            oth = X[asel[~torch.isin(asel, sr[m])]].T.contiguous().sort(1).values
            lo, hi = torch.searchsorted(oth, pos), torch.searchsorted(oth, pos, right=True)
            p2 = ((lo + hi).float() / 2 / oth.shape[1]).mean(1)
            margin = ((pos.mean(1) - mu) / sd).clamp(-50, 50)
            S[i] = (p1 + p2) / 2 + 1e-5 * margin
        v, j = torch.cat([top_s, S], 1).topk(5, dim=1)
        cand = torch.cat([top_u, torch.arange(X.shape[1], device=DEV).expand(K, -1) + off], 1)
        top_s, top_u = v, cand.gather(1, j)
        for (bs, bu), mk in zip(subs, masks):
            v, j = (S - (~mk).float() * 9).max(1)
            m = v > bs
            bs[m], bu[m] = v[m], off + j[m]
    return top_u.cpu().numpy(), top_s.cpu().numpy(), [b.cpu().numpy() for _, b in subs]


def detectors(f, subset=None, seeds=0):
    U, S, subs = select(f, subset, seeds)
    out = {k: {} for k in ("m1", "m5", "m1_mismatch", "m5_mismatch")}
    for i, m in enumerate(MEMES):
        A = f.cols("a", U[i])
        bg = np.concatenate([FILM] + [SEL[x] for x in MEMES if x != m])
        Z = (A - A[bg].mean(0)) / (A[bg].std(0) + 1e-6)
        out["m1"][m], out["m5"][m] = pair_score(Z[:, 0], m), pair_score(Z.mean(1), m)
        out["m1_mismatch"][m], out["m5_mismatch"][m] = mismatch(Z[:, 0], m), mismatch(Z.mean(1), m)
    for s, u in enumerate(subs):
        A = f.cols("a", u)
        out[f"m1_subset{s}"] = {m: pair_score(A[:, i], m) for i, m in enumerate(MEMES)}
    out["units"] = {m: U[i].tolist() for i, m in enumerate(MEMES)}
    out["train_score"] = {m: float(S[i, 0]) for i, m in enumerate(MEMES)}
    return out


def dense(X):
    """Prototype and logistic probe on a dense representation [alone sentences, dims]."""
    X = np.nan_to_num(np.asarray(X, np.float32))
    bgall = np.concatenate([FILM] + [SEL[x] for x in MEMES])
    X = (X - X[bgall].mean(0)) / (X[bgall].std(0) + 1e-6)
    out = {"proto": {}, "probe": {}, "proto_mismatch": {}, "probe_mismatch": {}}
    for m in MEMES:
        bg = np.concatenate([FILM] + [SEL[x] for x in MEMES if x != m])
        w = X[SEL[m]].mean(0) - X[bg].mean(0)
        s = X @ w / (np.linalg.norm(X, axis=1) * np.linalg.norm(w) + 1e-9)
        out["proto"][m], out["proto_mismatch"][m] = pair_score(s, m), mismatch(s, m)
        y = np.r_[np.ones(len(SEL[m])), np.zeros(len(bg))]
        lr = LogisticRegression(C=1, class_weight="balanced", max_iter=300).fit(X[np.r_[SEL[m], bg]], y)
        s = lr.decision_function(X)
        out["probe"][m], out["probe_mismatch"][m] = pair_score(s, m), mismatch(s, m)
    return out


def main(root, outdir, tags):
    root, outdir = Path(root), Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    works = [json.loads(x) for x in open(STUDY / "data/corpus.jsonl")]
    sents = [s for w in works for s in w["sentences"]]
    base = Feats(root, "qwen3-8b" if (root / "qwen3-8b").exists() else tags[0], "neuron")
    uref = UNITS_8B if base.meta["tag"] == "qwen3-8b" else [u % base.n for u in UNITS_8B]
    ref = base.cols("c", uref)
    for tag in tags:
        rng = np.random.default_rng(50)
        res = {"tag": tag, "memes_part_a": NAMES, "A": {}, "B": {}, "revenge": {}}
        for kind in ("neuron", "sae"):
            f = Feats(root, tag, kind)
            if kind == "neuron" and tag == base.meta["tag"]:
                u, r = np.array(uref), np.ones(len(uref))
            else:
                u, r = twins(f, ref)
            res["A"][kind] = rewording(f, u, rng) | dict(units=u.tolist(), twin_r=r.tolist(), twin_r_median=float(np.median(r)))
            res["revenge"][kind] = revenge(f, sents)
            big = kind == "neuron" and f.n > base.n
            res["B"][kind] = detectors(f, base.n / f.n, 5) if big else detectors(f)
            b = res["B"][kind]
            print(tag, kind, "A", res["A"][kind]["percentile"]["survivors"], round(res["A"][kind]["percentile"]["null_mean"], 2),
                  "| B m1", np.mean(list(b["m1"].values())), "m5", np.mean(list(b["m5"].values())), "| revenge", res["revenge"][kind]["pairs_won"], flush=True)
        R = np.load(root / tag / "alone_resid.npy", mmap_mode="r")
        d = R.shape[1] // 3
        res["B"]["resid_mid"] = dense(R[:, d:2 * d])
        print(tag, "resid", {k: np.mean(list(v.values())) for k, v in res["B"]["resid_mid"].items()}, flush=True)
        json.dump(res, open(outdir / f"{tag}_features.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
