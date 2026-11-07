"""Checks of the explanations that need no AI judge (PLAN.md, Section 10).

For each explained unit:
  same stem       the share of its 16 strongest reference examples whose peak token has the unit's commonest stem,
                  counted over the peaks that fall on content words; and the share of peaks on punctuation or function
                  words (where a unit may be summing up a clause, which is not the same as detecting a word)
  best word       the correlation of its profile over the 3,700 study sentences with the presence of its best single
                  word or word pair (v1's measure)
  held out        how often the explainer's key words, and the keyword analyst's terms, occur in the 40 words up to
                  the peak of the reference examples ranked 17 to 24 (never shown to the explainer), against random
                  40-word windows from documents of the same kind

All three are tabulated against the kind judge's verdict (a name or word, or an idea).

    cd study && ../.venv/bin/python v2/analysis/plain_checks.py <model-tag>

Writes results/v2/<tag>/plain_checks.json.
"""
import json
import sys
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import common as C
import textcheck as T

import experiments as X

WINDOW = 40


def best_word(profiles, B, chunk=1000):
    """v1's measure (robustness/nulls.py): the best correlation of each profile with the presence of one word or bigram."""
    p = np.asarray(B.mean(0)).ravel()
    sb = np.sqrt(p * (1 - p)) + 1e-9
    r, j = np.empty(profiles.shape[1], np.float32), np.empty(profiles.shape[1], np.int64)
    for c in range(0, profiles.shape[1], chunk):
        Y = profiles[:, c:c + chunk]
        Cm = np.asarray(B.T @ ((Y - Y.mean(0)) / (Y.std(0) + 1e-9))) / B.shape[0] / sb[:, None]
        j[c:c + chunk], r[c:c + chunk] = Cm.argmax(0), Cm.max(0)
    return r, j


def content(token):
    w = token.strip().lower()
    return bool(T.WORD.fullmatch(w)) and w not in T.STOP and len(w) > 1


def window(text):
    return " ".join(text.split()[-WINDOW:])


def main(tag):
    rng = np.random.default_rng(C.SEED + 21)
    d = C.RESULTS / tag
    g = lambda n: {int(u): v for u, v in json.load(open(d / n))["units"].items()}      # noqa: E731
    ex, terms, kinds = g("explanations.json"), g("keyword_terms.json"), g("kinds.json")
    inputs = {r["unit"]: r for r in json.load(open(d / "explain_inputs.json"))["units"]}
    held = {int(u): v for u, v in json.load(open(d / "heldout_examples.json"))["units"].items()}
    units = sorted(u for u in ex if u in inputs)

    # best single word on the study sentences
    su = np.load(C.ACTS / tag / "sent_units.npy")
    col = {int(u): k for k, u in enumerate(su)}
    S = np.load(C.ACTS / tag / "sent_orig.u8.npy", mmap_mode="r")
    prof = np.stack([np.asarray(S[:, col[u]], dtype=np.float32) for u in units], 1)
    sentences = [s for w in X.works for s in w["sentences"]]
    cv = CountVectorizer(ngram_range=(1, 2), min_df=2, binary=True)
    B = cv.fit_transform(sentences).astype(np.float32).tocsc()
    vocab = cv.get_feature_names_out()
    bw, jw = best_word(prof, B)

    # random windows from documents of the same kind, for the held-out comparison
    by_src = {}
    for path in (C.REFERENCE, C.REFERENCE.with_name("supplement.jsonl")):
        if path.exists():
            for doc in map(json.loads, open(path)):
                by_src.setdefault(doc["src"], []).append(doc["text"])

    def random_window(src):
        words = by_src[src][rng.integers(len(by_src[src]))].split()
        end = int(rng.integers(min(WINDOW, len(words)), len(words) + 1))
        return " ".join(words[max(end - WINDOW, 0):end])

    rows = []
    for k, u in enumerate(units):
        peaks = [e["token"] for e in inputs[u]["reference_examples"]]
        cw = [T.stem(p.strip()) for p in peaks if content(p)]
        top = Counter(cw).most_common(1)
        key, alt = T.key_stems(ex[u]["key_words"]), terms.get(u, [])
        hw = [window(e["before"] + e["token"]) for e in held.get(u, [])]
        rw = [random_window(e["source"]) for e in held.get(u, []) for _ in range(5)]
        rows.append({"unit": u, "kind": kinds.get(u, {}).get("kind"), "detects": kinds.get(u, {}).get("detects"),
                     "peaks_on_content_words": len(cw), "peaks_on_punctuation_or_function_words": len(peaks) - len(cw),
                     "same_stem_share_of_content_peaks": (top[0][1] / len(cw)) if cw else None, "commonest_stem": top[0][0] if cw else None,
                     "best_word": str(vocab[jw[k]]), "best_word_r": float(bw[k]),
                     "held_out_examples": len(hw),
                     "held_out_hit_explainer_key_words": float(np.mean([bool(T.uses(t, key)) for t in hw])) if hw else None,
                     "held_out_hit_analyst_terms": float(np.mean([T.phrase_hits(t, alt) > 0 for t in hw])) if hw else None,
                     "random_window_hit_explainer_key_words": float(np.mean([bool(T.uses(t, key)) for t in rw])) if rw else None,
                     "random_window_hit_analyst_terms": float(np.mean([T.phrase_hits(t, alt) > 0 for t in rw])) if rw else None})

    def summary(rs):
        f = lambda key: (float(np.mean([r[key] for r in rs if r[key] is not None])) if any(r[key] is not None for r in rs) else None)      # noqa: E731
        return {"n": len(rs), "same_stem_share_of_content_peaks_mean": f("same_stem_share_of_content_peaks"),
                "share_of_peaks_on_punctuation_or_function_words": float(np.mean([r["peaks_on_punctuation_or_function_words"] / 16 for r in rs])) if rs else None,
                "units_with_all_content_peaks_on_one_stem": int(sum(r["same_stem_share_of_content_peaks"] == 1 and r["peaks_on_content_words"] >= 8 for r in rs)),
                "best_word_r_median": float(np.median([r["best_word_r"] for r in rs])) if rs else None,
                "held_out_hit_explainer_key_words": f("held_out_hit_explainer_key_words"), "random_window_hit_explainer_key_words": f("random_window_hit_explainer_key_words"),
                "held_out_hit_analyst_terms": f("held_out_hit_analyst_terms"), "random_window_hit_analyst_terms": f("random_window_hit_analyst_terms")}

    out = {"model": tag, "all": summary(rows), "by_detects": {k: summary([r for r in rows if r["detects"] == k]) for k in ("name or word", "idea")},
           "by_kind": {k: summary([r for r in rows if r["kind"] == k]) for k in sorted({r["kind"] for r in rows if r["kind"]})}, "units": rows}
    C.save(out, f"{tag}/plain_checks.json")
    print(tag, "plain checks:", {k: v for k, v in out["by_detects"].items()})


if __name__ == "__main__":
    main(sys.argv[1])
