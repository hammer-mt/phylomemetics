"""E2 (heredity) re-implemented from scratch, with the exact permutation p.

Written without importing experiments.py or robustness/e2_multiverse.py, as a check on both. For each model and text
set it recomputes every work's score on every unit (mean of its 3 strongest sentences), presence (beats all but one
of the 69 ordinary films), the optional units (present in 2-16 of the 32 Batman works), each comic's inheritance rate
in each screen work (share of the comic's optional units also present in the work), the hub adjustment (z against
the 11 screen works that are not adaptations), and the statistic: the share of eligible comics (published no later
than the adaptation) whose z beats the best documented source, averaged over the 9 adaptations. Lower is stronger.

The p-value is exact: the statistic is recomputed under every injective relabelling of the 7 distinct source comics
into the 12 comics (12!/5! = 3,991,680 maps, the observed one included), and p is the share of maps scoring at least
as low as observed. The paper's placebo is recomputed as coded and with the two placebo targets that have no
eligible comic (the 1943 serial and Batman '66, which score 0 by construction) left out.

Writes results/robustness/e2_reimpl.json.
"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from discover import load_units  # noqa: E402

ACTS = {"anon": Path(os.environ.get("PHYLO_ACTS_ANON", ROOT / "activations_anon")),
        "orig": Path(os.environ.get("PHYLO_ACTS", ROOT / "activations"))}
MODELS = ["qwen3-8b", "llama3.1-8b"]
OUT = ROOT / "results/robustness/e2_reimpl.json"

works = [json.loads(l) for l in open(ROOT / "data/corpus.jsonl")]
ID = {w["label"]: w["id"] for w in works}
YEAR = {w["id"]: w["year"] or 0 for w in works}
batman = [w["id"] for w in works if w["group"] == "batman"]
films = [w["id"] for w in works if w["group"] == "baseline"]
comics = [i for i in batman if works[i]["medium"] == "comic"]
screen = [i for i in batman if works[i]["medium"] != "comic"]
SOURCES = {"Batman Begins": ["Year One"], "The Dark Knight": ["The Long Halloween", "The Killing Joke"],
           "Dark Knight Rises": ["Knightfall", "Dark Knight Returns"], "Batman v Superman": ["Dark Knight Returns"],
           "Under the Red Hood": ["A Death in the Family"], "Arkham Asylum (game)": ["Arkham Asylum (comic)"],
           "Batman '89": ["The Killing Joke"], "Joker (2019)": ["The Killing Joke"], "The Batman": ["Year One", "The Long Halloween"]}
adaptations = {ID[t]: [ID[s] for s in v] for t, v in SOURCES.items()}
plain_screen = [i for i in screen if i not in adaptations]
assert batman == list(range(32)) and len(films) == 69 and len(comics) == 12 and len(plain_screen) == 11


def presence(model, text):
    """Bool [32, 2U]: a Batman work's score beats the second-highest of the 69 film scores (both poles)."""
    meta, pos, neg, ps, ns = load_units(ACTS[text] / model)
    sw = np.asarray(meta["sent_work"])
    U = pos.shape[1]
    span = {}
    for w in batman + films:
        r = np.flatnonzero(sw == w)
        assert r[-1] - r[0] + 1 == len(r) >= 3
        span[w] = (r[0], r[-1] + 1)
    P = np.zeros((len(batman), 2 * U), bool)
    for half, (arr, scale, sign) in enumerate(((pos, ps, 1), (neg, ns, -1))):
        def score(w):
            block = np.sort(sign * np.asarray(arr[span[w][0]:span[w][1]], dtype=np.float32), axis=0)
            return block[-3:].mean(0) * scale
        bar = np.sort(np.stack([score(w) for w in films]), axis=0)[-2]
        for w in batman:
            P[w, half * U:(half + 1) * U] = score(w) > bar
    return P


def inheritance(P, lo=2, hi=16):
    nb = P.sum(0)
    opt = (nb >= lo) & (nb <= hi)
    rate = {}
    for c in comics:
        oc = opt & P[c]
        n = max(int(oc.sum()), 1)
        for a in screen:
            rate[a, c] = int((oc & P[a]).sum()) / n
    return rate, int(opt.sum())


def hub(rate):
    z = {}
    for a in screen:
        ref = [t for t in plain_screen if t != a]
        for c in comics:
            v = np.array([rate[t, c] for t in ref])
            z[a, c] = (rate[a, c] - v.mean()) / (v.std() + 1e-9)
    return z


def share(z, a, srcs):
    """(share of rivals beating the best source, rank of the best source, comics compared incl. sources)."""
    rivals = [c for c in comics if YEAR[c] <= YEAR[a] and c not in srcs]
    best = max(z[a, s] for s in srcs)
    beat = int(sum(z[a, c] > best for c in rivals))
    return (beat / len(rivals) if rivals else 0.0), 1 + beat, len(rivals) + len(srcs)


def exact_null(z, amap):
    """The statistic under every injective relabelling of the distinct source comics into the 12 comics."""
    src = sorted({s for v in amap.values() for s in v})
    nc = len(comics)
    maps = np.fromiter(itertools.chain.from_iterable(itertools.permutations(range(nc), len(src))), dtype=np.int8)
    maps = maps.reshape(-1, len(src))
    total = np.zeros(len(maps))
    for a, srcs in amap.items():
        table = np.full((nc,) * len(srcs), np.nan)
        for tup in itertools.permutations(range(nc), len(srcs)):
            table[tup] = share(z, a, [comics[j] for j in tup])[0]
        total += table[tuple(maps[:, src.index(s)] for s in srcs)]
    return total / len(amap)


def placebo(z):
    out = {}
    for t in plain_screen:
        n_elig = sum(YEAR[c] <= YEAR[t] for c in comics)
        out[works[t]["label"]] = {"mean_share": float(np.mean([share(z, t, s)[0] for s in adaptations.values()])), "n_eligible_comics": n_elig}
    return out


def main():
    paper = json.load(open(ROOT / "results/experiments.json"))
    out = {"spec": "top-3 sentence mean; present = beats all but 1 of 69 films; optional = present in 2-16 of 32 Batman works; "
                   "hub-adjusted rate; time rule on; best of the documented sources; p = exact share of the 3,991,680 relabellings "
                   "with a mean share <= observed"}
    for model in MODELS:
        out[model] = {}
        for text in ("anon", "orig"):
            P = presence(model, text)
            rate, n_opt = inheritance(P)
            z = hub(rate)
            rows = []
            for a, srcs in adaptations.items():
                s, rank, n = share(z, a, srcs)
                rows.append({"adaptation": works[a]["label"], "sources": [works[c]["label"] for c in srcs], "share": s, "rank": rank, "n_compared": n})
            obs = float(np.mean([r["share"] for r in rows]))
            null = exact_null(z, adaptations)
            res = {"n_optional_units": n_opt, "mean_share": obs, "rows": rows,
                   "exact_null": {"n_maps": int(len(null)), "mean": float(null.mean()), "sd": float(null.std()),
                                  "p": float(np.mean(null <= obs + 1e-9))}}
            ref = paper[model]["E2"]["primary" if text == "anon" else "original_upper_bound"]
            mine = [(r["adaptation"], round(r["share"], 3), r["rank"], r["n_compared"]) for r in rows]
            theirs = [(r["target"], r["share_of_comics_beating_source"], r["rank"], r["n_comics"]) for r in ref["rows"]]
            res["vs_experiments_json"] = {"rows_match": mine == theirs, "stored_mean_share": ref["mean_share_beating_source"],
                                          "stored_mean_from_3dp_shares": round(float(np.mean([m[1] for m in mine])), 6),
                                          "stored_p_monte_carlo": ref["p_one_sided"], "stored_null_mean": ref["null_mean"]}
            if text == "anon":
                pl = placebo(z)
                valid = [v["mean_share"] for v in pl.values() if v["n_eligible_comics"] > 0]
                res["placebo"] = {"targets": pl, "mean_as_coded": float(np.mean([v["mean_share"] for v in pl.values()])),
                                  "stored_mean": ref["placebo_mean"], "mean_valid_targets": float(np.mean(valid)),
                                  "min_valid_targets": float(np.min(valid)), "n_valid_targets": len(valid),
                                  "real_below_all_valid_targets": bool(obs < np.min(valid))}
            out[model][text] = res
            print(model, text, "units", n_opt, "share", round(obs, 4), "exact p", res["exact_null"]["p"], "null mean",
                  round(res["exact_null"]["mean"], 4), "| rows match experiments.json:", res["vs_experiments_json"]["rows_match"], flush=True)
            del P
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
