"""Bigger is cleaner? (PLAN.md, Section 10, "Size and precision".)

The essay's claim, on our units: explanations should win more often in the larger model. The win rate (win against
tie or loss) of the Qwen3-32B units is compared with that of the Qwen3-8B units, among units testable out of
context, permuting the model label within kind (one-sided, 10,000 draws). The same with the untestable units left in
is given beside it, with the difference in near-miss firing and the one-idea scores.

    cd study && ../.venv/bin/python v2/analysis/size_test.py [big-tag small-tag]

Writes results/v2/size_test.json.
"""
import json
import sys

import numpy as np

import common as C


def load(tag):
    return json.load(open(C.RESULTS / tag / "explanation_tests.json"))["units"]


def test(big, small, rng, n=10000):
    kinds = sorted({r["kind"] for r in big} | {r["kind"] for r in small})
    win = lambda rs: np.array([r["outcome"] == "win" for r in rs], float)      # noqa: E731
    wb, ws = win(big), win(small)
    obs = wb.mean() - ws.mean()
    kb, ks = np.array([r["kind"] for r in big]), np.array([r["kind"] for r in small])
    null = np.zeros(n)
    for k in kinds:
        pool = np.r_[wb[kb == k], ws[ks == k]]
        nb = int((kb == k).sum())
        if nb == 0 or nb == len(pool):
            null += (wb[kb == k].sum() / len(wb)) - (ws[ks == k].sum() / len(ws))          # a kind found in one model only cannot be permuted
            continue
        idx = np.argsort(rng.random((n, len(pool))), axis=1)
        perm = pool[idx]
        null += perm[:, :nb].sum(1) / len(wb) - perm[:, nb:].sum(1) / len(ws)
    return {"n_big": len(big), "n_small": len(small), "win_rate_big": float(wb.mean()), "win_rate_small": float(ws.mean()), "difference": float(obs),
            "p_one_sided": float((1 + (null >= obs - 1e-12).sum()) / (n + 1)),
            "by_kind": {k: {"n_big": int((kb == k).sum()), "n_small": int((ks == k).sum()),
                            "win_rate_big": float(wb[kb == k].mean()) if (kb == k).any() else None,
                            "win_rate_small": float(ws[ks == k].mean()) if (ks == k).any() else None} for k in kinds}}


def boot_diff(a, b, rng, n=4000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n)]
    return {"big": float(a.mean()), "small": float(b.mean()), "difference": float(a.mean() - b.mean()), "ci95": [float(x) for x in np.percentile(d, [2.5, 97.5])]}


def main(big_tag="qwen3-32b", small_tag="qwen3-8b"):
    rng = np.random.default_rng(C.SEED + 31)
    big, small = load(big_tag), load(small_tag)
    tb, ts = [r for r in big if r["testable_out_of_context"]], [r for r in small if r["testable_out_of_context"]]
    out = {"big": big_tag, "small": small_tag, "testable_units": test(tb, ts, rng), "all_units": test(big, small, rng),
           "not_testable_out_of_context": {"big": len(big) - len(tb), "small": len(small) - len(ts)},
           "near_miss_strong_rate": boot_diff([r["strong"]["C"] / 5 for r in tb], [r["strong"]["C"] / 5 for r in ts], rng),
           "new_example_strong_rate": boot_diff([r["strong"]["A"] / 5 for r in tb], [r["strong"]["A"] / 5 for r in ts], rng),
           "one_idea_score": boot_diff([r["one_idea"] for r in big], [r["one_idea"] for r in small], rng),
           "idea_bound_share": boot_diff([r["idea_bound"] for r in tb], [r["idea_bound"] for r in ts], rng)}
    out["criterion_met"] = bool(out["testable_units"]["p_one_sided"] < 0.05)
    C.save(out, "size_test.json")
    t = out["testable_units"]
    print(f"size test: win rate {t['win_rate_big']:.3f} ({big_tag}) vs {t['win_rate_small']:.3f} ({small_tag}), p {t['p_one_sided']:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
