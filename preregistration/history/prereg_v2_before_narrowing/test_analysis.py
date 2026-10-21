"""Tests for the frozen analysis (prereg/analysis.py, prereg-v2) on SYNTHETIC data with known answers.

No DEV or TEST text and no model is used: the test builds a small fake world in a temporary folder (pseudo-word texts,
random neuron aggregates with planted inheritance, random embeddings, fake LLM outputs) in the exact file formats
that extract_compact.py and readers.py write, then runs every runner end to end. Real metadata files from prereg/
(memes_for_tracing.json, planted_design.json, works_shakespeare.json, culture_pool.json, dev_units.json) supply the
structure only. It has three parts:
  unit tests       every helper on hand-made inputs with exact answers
  null worlds      type-I error of every primary statistic (and of the draft statistics they replace) over 200+
                   replicates of a world with no effect: H1/H2 (no method difference), H3 (a detector with no meme
                   specificity), H4 (equally robust methods, neurons starting worse; and neurons at chance), H5, a
                   non-specific T2 detector against salience-matched controls, a franchise-clustered T3 world, and the
                   time-consistent lineage null
  pipeline         the whole analysis on a synthetic world with planted effects

  .venv/bin/python prereg/test_analysis.py          # everything (about 5-10 minutes)
  .venv/bin/python prereg/test_analysis.py --unit   # unit tests and null worlds only
"""
import itertools
import json
import os
import shutil
import string
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TMP = Path(tempfile.mkdtemp(prefix="prereg_test_", dir=os.environ.get("PREREG_TEST_TMP")))
os.environ["PREREG_DATA"] = str(TMP / "data")
os.environ["PREREG_FEATURES"] = str(TMP / "features")
os.environ["PREREG_RESULTS"] = str(TMP / "results")
sys.path.insert(0, str(HERE))
import analysis as A  # noqa: E402

RNG = np.random.default_rng(1)
U = 240                     # synthetic units per model (both poles)
MODELS = A.MODELS
ENCS = ["qwen3emb-4b", "e5-large", "e5-base"]
FAILS = []
ALPHA_MC = 0.05 + 2 * np.sqrt(0.05 * 0.95 / 200)     # alpha within Monte Carlo error for 200 replicates


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


# =============================================================================================== unit tests
def unit_tests():
    print("unit tests")
    check(A.stem("flipped") == A.stem("flips") == A.stem("flip") == A.stem("flipping"), "stemmer: flip family")
    check(A.stem("buried") == A.stem("bury"), "stemmer: bury/buried")
    check(A.keyword_hits("He flipped two coins and a coin.", ["coin"]) == 2, "keyword hits count occurrences")
    check(A.keyword_hits("The robbers killed both his parents.", ["parents killed"]) == 1, "multi-word query = all content words")
    check(A.ban_hits("The avenger arrives.", ["avenge"]) == 1 and A.ban_hits("A funerary urn.", ["funeral"]) == 1,
          "ban list: shared 5-letter prefix catches derivations")
    check(A.ban_hits("The killer waits.", ["kill"]) == 1 and A.ban_hits("He paid a bribery fee.", ["bribe"]) == 1,
          "ban list: frozen families for short stems")
    check(A.ban_hits("The hill is green.", ["kill"]) == 0, "ban list: no false prefix hit on short stems")
    check(A.phrase_hits("He raised the flag.", ["raised him"]) == 0 and A.phrase_hits("The butler raised him.", ["raised him"]) == 1,
          "phrase hits are exact")
    check(A.keyword_hits("他抛硬币决定生死。", ["硬币"]) == 1, "Chinese terms match substrings")
    check(abs(A.auc([1, 2, 3], [0, 0.5]) - 1.0) < 1e-12 and abs(A.auc([1], [1]) - 0.5) < 1e-12, "AUC with ties")
    r = A.signflip(np.ones(10), alternative="greater")
    check(r["exact"] and abs(r["p"] - 1 / 1024) < 1e-12, "exact sign flip p = 1/2^k")
    h = A.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    check(h["a"]["reject"] and not h["b"]["reject"] and abs(h["a"]["p_holm"] - 0.03) < 1e-12, "Holm")
    q = A.bh({"a": 0.01, "b": 0.04, "c": 0.03})
    check(abs(q["a"] - 0.03) < 1e-12 and abs(q["b"] - 0.04) < 1e-12, "Benjamini-Hochberg")
    check(abs(A.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) - 1) < 1e-12 and abs(A.cohen_kappa([1, 0, 1, 0], [1, 1, 0, 0])) < 1e-12, "Cohen's kappa")
    mc = A.mcnemar_exact([1] * 10 + [0] * 2, [0] * 10 + [0] * 2)
    check(mc["a_only"] == 10 and mc["p"] < 0.01, "exact McNemar")
    check(A.verdict3(True, False) == "supported" and A.verdict3(False, True) == "falsified" and A.verdict3(False, False) == "undetermined"
          and A.verdict3(True, True, inconclusive=True) == "inconclusive", "three-way verdicts")
    check(A.iut([0.01, 0.04, 0.2]) == 0.2, "intersection-union p")
    masked, n = A.mask_names(["Bruce Wayne meets Alfred at the manor.", "Later, Bruce's butler Alfred warns him.", "After that he leaves."])
    check(masked[0].startswith("X1 meets X2") and "X1's" in masked[1] and masked[2] == "After that he leaves.", "name masker")
    s, info = A.t3_text(["Bluntman (a parody of Batman and Robin) fights.", "It is inspired by a comic.", "Then batman leaves."])
    check(len(s) == 2 and "parody" not in s[0] and "X0" in s[1] and info["franchise_hits"].get("batman") == 1, "T3 text rule: relation statements and franchise scan")
    neu, n = A.neutralise(["Bruce meets Alfred in Gotham."], ["The heir meets the butler in the city."])
    check(neu[0] == "X1 meets X2 in X3." and n == 3, "neutralise (make_neutral rule)")
    nz, n = A.neutralise_cjk(["布鲁斯在哥谭见到阿尔弗雷德。"], ["那位继承人在城市见到管家。"])
    check(n >= 2 and "X1" in nz[0], "neutralise for Chinese")
    check(A.collapse_codes(["X1 meets X12."]) == ["X0 meets X0."], "code collapse")
    t25 = set(A.drift_banned_types("w", ["Alpha bravo charlie delta echo foxtrot golf hotel 484."], 25))
    t75 = set(A.drift_banned_types("w", ["Alpha bravo charlie delta echo foxtrot golf hotel 484."], 75))
    check(t25 < t75 and len(A.drift_banned_types("w", ["Alpha bravo 484."], 100)) == 3, "drift ban sets are nested and include numbers")
    rr = A.realised_replacement(["The knight travels north."], ["The rider travelled north."], {"knight", "travel"})
    check(rr["banned_token_share_removed"] == 0.5 and rr["banned_types_left"] == ["travel"], "realised replacement with the derivational rule")
    dc = A.drift_checks(["X1 rides north at dawn."], ["X1 travels toward the Dark Knight's tower."], {"rid"})
    check(not dc["valid"] and any("D1c" in f for f in dc["failures"]), "drift checks catch new capitals and epithets")
    check(A.drift_checks(["X1 rides north at dawn."], ["X1 travels north at sunrise."], {"rid", "dawn"})["valid"], "drift checks pass a clean rewrite")
    check(A.recognised("This is The Mask of Zorro", "The Mask of Zorro") and not A.recognised("unknown", "The Mask of Zorro")
          and not A.recognised("a film about a lawyer", "The Mask of Zorro"), "recognition probe rule")
    check(A.ban_list_overlap(["secret identity", "anonymity"], ["secret identity", "mask"]) == 0.5, "ban-list overlap of a query")
    zc = A.zh_checks(["X1 meets the man."], ["X1见到了蝙蝠侠。"])
    check(not zc["valid"] and any("L3" in f for f in zc["failures"]), "Chinese franchise rendering check")
    # story checks: Q1b length and Q2 for the coded version
    memes = {"a": {"keywords": ["coin"]}, "b": {"keywords": ["funeral"]}, "c": {"keywords": ["vat"]}}
    plot = {"plot": "P1", "memes": ["a", "b", "c"]}
    frame = [f"Frame sentence number {k} happens." for k in range(18)]
    def story(version, fills):
        S = list(frame)
        for i, f in zip((2, 8, 14), fills):
            S[i] = f
        return {"plot": "P1", "version": version, "sentences": S, "slots": {"a": [2], "b": [8], "c": [14]}}
    expl = story("explicit", ["He tosses a coin to decide.", "They hold a funeral at dawn.", "She falls into a vat today."])
    coded = story("coded", ["He lets the silver disc choose his path.", "They plant the old man at dawn.", "She takes the green bath today."])
    longf = story("banned", ["He lets the small spinning silver disc on the old table choose which of the two roads to take tonight.",
                             "They plant the old man at dawn.", "She takes the green bath today."])
    check(A.story_checks(coded, plot, memes, [expl])["valid"], "story checks pass a matched coded version")
    check(any("Q1b" in f for f in A.story_checks(longf, plot, memes, [expl])["failures"]), "story checks: Q1b catches a fill 15% longer")
    bad = story("coded", ["He tosses the funerary disc.", "They plant the old man at dawn.", "She takes the green bath today."])
    check(any("Q2" in f for f in A.story_checks(bad, plot, memes, [expl])["failures"]), "story checks: Q2 bans derivations in coded versions")
    ex = A.q4_q5_exclusions("control", ["a", "b"], ["a", "b", "z"], [set(), {"z"}], [set()])
    check(ex == {"z"}, "Q5 is symmetric: one reader naming an unplanted meme excludes the cell")
    ex = A.q4_q5_exclusions("banned", ["a", "b"], ["a", "b"], [{"a", "b"}, {"a"}], [])
    check(ex == {"b"}, "Q4: banned versions need both Claude readers")
    # lineage test: ties, time-consistent null (exact enumeration), planted recovery
    T, C = 4, 6
    Sim = np.zeros((8, C))
    hub = A.hub_groups_fixed(8, list(range(T, 8)), C)
    r = A.lineage_test(Sim, list(range(T)), [[0], [1], [2], [3]], np.ones((T, C), bool), hub, n_perm=50)
    check(abs(r["stat"] - 0.5) < 1e-12, "lineage share counts ties as one half (constant similarity gives 0.5)")
    rng = np.random.default_rng(3)
    Sim = rng.normal(0, 1, (8, C))
    srcs = [[0], [1], [3], [4]]
    elig = np.array([[c <= k + 2 for c in range(C)] for k in range(T)])
    r = A.lineage_test(Sim, list(range(T)), srcs, elig, hub, n_perm=20000, rng=np.random.default_rng(4))
    vals = []
    Z = A._zscores(Sim, list(range(T)), *hub[:2])
    for pi in itertools.permutations(range(C)):
        if all(elig[t, pi[s]] for t, ss in enumerate(srcs) for s in ss):
            vals.append(A._shares_batch(Z, A._src_mask(srcs, T, C, pi), elig).mean())
    check(abs(r["null_mean"] - np.mean(vals)) < 0.01, f"lineage null mean = exact time-consistent mean ({r['null_mean']:.3f} vs {np.mean(vals):.3f})")
    T, C = 9, 12
    Sim = RNG.normal(0, 1, (20, C))
    src = [[t % C] for t in range(T)]
    for t in range(T):
        Sim[t, src[t][0]] += 5
    hub = A.hub_groups_fixed(20, list(range(T, 20)), C)
    r = A.lineage_test(Sim, list(range(T)), src, np.ones((T, C), bool), hub, n_perm=500)
    check(r["stat"] < 0.05 and r["p"] < 0.01, f"lineage test finds planted sources (share {r['stat']:.3f})")
    films = 30
    srcs = [[k % 6] for k in range(films)]
    S2 = RNG.normal(0, 1, (films, 8))
    for k in range(films):
        S2[k, srcs[k][0]] += 2
    r = A.lineage_test(S2, list(range(films)), srcs, np.ones((films, 8), bool), A.hub_groups_other_targets(films, list(range(films)), srcs, 8), n_perm=300)
    check(r["p"] < 0.01, "lineage test with label-dependent hub")
    # neuron detectors: one pass with leave-one-example-out
    bg = RNG.normal(0, 1, (500, 300)).astype(np.float32)
    exa = RNG.normal(0, 1, (12, 300)).astype(np.float32)
    exa[:6, 123] += 6
    exa[6:, 45] += 6
    B = np.vstack([bg, exa])
    det = A.neuron_detectors(B, {("m", "main"): list(range(500, 506)), ("n", "main"): list(range(506, 512))},
                             {"m": list(range(500, 506)), "n": list(range(506, 512))}, chunk=128)
    crit = A.neuron_loeo_criteria(B, {("m", "main"): list(range(500, 506))}, {"m": list(range(500, 506))}, {("m", "main"): det[("m", "main")]})
    check(det[("m", "main")]["m1"] == 123 and det[("n", "main")]["m1"] == 45 and crit[("m", "main")]["m1"] > 0.95,
          "neuron detectors find the planted units, with LOEO criteria")
    best, sc, mg = A.select_units(exa[:6], bg, m=3, chunk=128)
    check(best[0] == 123, "few-shot unit selection")
    P = RNG.normal(0, 1, (10, 16)) + 2
    Bk = RNG.normal(0, 1, (300, 16))
    check(A.loeo_linear("probe", P, Bk) > 0.9 and A.loeo_linear("cproto", P, Bk) > 0.9, "linear detectors with LOEO")
    # T4 helpers
    Pm = np.zeros((5, 4), bool)
    Pm[0, 0] = Pm[1, 0] = Pm[2, 1] = Pm[3, 1] = Pm[4, 1] = True
    nm, ms, later, fa = A.mutation_stats(Pm)
    check(list(fa) == [0, 2, -1, -1] and later[1] == 2, "mutation statistics")
    nm, ms, later, fa = A.mutation_stats(Pm, skip={2: {3}})
    check(later[1] == 1, "mutation success leaves out documented adaptations")
    Q = RNG.random((20, 30)) < 0.3
    Rq = A.curveball(Q, 200, np.random.default_rng(0))
    check((Rq.sum(0) == Q.sum(0)).all() and (Rq.sum(1) == Q.sum(1)).all() and (Rq != Q).any(), "curveball keeps row and column totals")
    r = A.t5_test(np.array([0.9, 0.8, 0.85]), np.array([0.1] * 9), [[0, 1, 2], [3, 4, 5], [6, 7, 8]], n_perm=500)
    check(r["p"] < 0.05, "T5 matched-null test")


# =============================================================================================== null worlds
def null_worlds(R=200):
    print(f"null worlds ({R} replicates each; alpha within Monte Carlo error <= {ALPHA_MC:.3f})")
    rng = np.random.default_rng(7)
    # H1 / H2: no method difference, 34 memes
    rej = np.mean([A.signflip(rng.normal(0, 0.15, 34), alternative="greater", n_mc=2000, rng=rng)["p"] < 0.05 for _ in range(R)])
    check(rej <= ALPHA_MC, f"H1/H2 null (no difference over 34 memes): rejection {rej:.3f}")
    # H4: equally robust (same proportional loss of above-chance recovery), neurons start worse (DEV gap); neurons at chance
    C = 0.45
    cl = np.repeat(np.arange(21), 2)
    for label, s_tf, s_nr, keep in (("equal robustness, DEV-like gap", 0.11, 0.164, {0: 1, 25: .8, 50: .6, 75: .4, 100: .2}),
                                    ("equal robustness, neurons at 0.30", 0.11, 0.30, {0: 1, 25: .8, 50: .6, 75: .4, 100: .2}),
                                    ("total loss", 0.11, 0.164, {0: 1, 25: 0, 50: 0, 75: 0, 100: 0})):
        new, old = [], []
        for _ in range(R):
            tf = {L: np.clip(C - keep[L] * (C - s_tf) + rng.normal(0, .12, 42), 0, 1) for L in keep}
            nr = {L: np.clip(C - keep[L] * (C - s_nr) + rng.normal(0, .12, 42), 0, 1) for L in keep}
            d = np.mean([tf[L] - nr[L] for L in A.H4_LEVELS], 0)
            new.append(A.signflip(d, clusters=cl, alternative="greater")["p"] < 0.05)
            old.append(A.drift_interaction(tf, nr, cl)["p"] < 0.05)
        check(np.mean(new) <= ALPHA_MC, f"H4 null ({label}): head-to-head rejection {np.mean(new):.3f} (draft degradation statistic {np.mean(old):.3f})")
    # neurons flat at chance cannot pass H4's conjunct (their own lineage test at levels 75 and 100)
    src = [[t % 12] for t in range(9)]
    hub = A.hub_groups_fixed(20, list(range(9, 20)), 12)
    ps = [A.lineage_test(rng.normal(0, 1, (20, 12)), list(range(9)), src, np.ones((9, 12), bool), hub, n_perm=200, rng=rng)["p"] for _ in range(R)]
    check(np.mean(np.array(ps) < 0.05) <= ALPHA_MC, f"lineage null with no signal: rejection {np.mean(np.array(ps) < 0.05):.3f} (H4 conjunct cannot pass at chance)")
    # time-consistent lineage null with a Batman-like time rule
    cyear = np.array([1986, 1987, 1988, 1988, 1989, 1989, 1993, 1996, 2002, 2008, 2012, 2017])
    years = [2005, 2008, 2012, 2016, 2010, 2009, 1989, 2019, 2022]
    srcs = [[1], [7, 2], [6, 0], [0], [3], [5], [2], [2], [1, 7]]
    el = np.array([[c <= y for c in cyear] for y in years])
    ps = [A.lineage_test(rng.normal(0, 1, (20, 12)), list(range(9)), srcs, el, hub, n_perm=200, rng=rng)["p"] for _ in range(R)]
    check(np.mean(np.array(ps) < 0.05) <= ALPHA_MC, f"time-consistent lineage null: rejection {np.mean(np.array(ps) < 0.05):.3f}")
    # H5: advantage over 13 plays, no difference
    rej = np.mean([A.signflip(rng.normal(0, 0.22, 13), alternative="greater")["p"] < 0.05 for _ in range(R)])
    check(rej <= ALPHA_MC, f"H5 null (13 plays): rejection {rej:.3f}")
    # T2: a detector with NO meme specificity, salience-matched controls (the v2 protocol), on the real design
    design = [{"plot": p["plot"], "version": v, "memes": p["memes"]} for p in json.load(open(HERE / "planted_design.json"))["plots"]
              for v in A.VERSIONS]
    memes = sorted({m for d in design for m in d["memes"]})
    n = len(design)
    twin, alln, spec_rej, h1_rej = [], [], [], []
    cache = {}
    for rep in range(R):
        g = rng.normal(0, 1, n) + np.array([1.0 if d["version"] != "explicit" else 1.0 for d in design])   # every slot equally eventful
        g_draft = rng.normal(0, 1, n) + np.array([1.0 if d["version"] in A.PLANTED else 0.0 for d in design])  # draft 'neutral event' controls
        M = np.repeat(g[:, None], len(memes), 1)
        Md = np.repeat(g_draft[:, None], len(memes), 1)
        a = A.detection_aucs(M, design, memes, A.REWORDED)
        twin.append(np.mean(list(a.values())))
        alln.append(np.mean(list(A.detection_aucs(Md, design, memes, A.REWORDED, scheme="all").values())))
        Mk = rng.normal(0, 1, (n, len(memes)))
        kw = A.detection_aucs(Mk, design, memes, A.REWORDED)
        p_meme = A.paired_meme_test(a, kw, memes, "greater")["p"]
        p_plot = A.plot_flip_test(M, Mk, design, memes, A.REWORDED, n_mc=2000)["p"]
        h1_rej.append(max(p_meme, p_plot) < 0.05)
        det = memes[:11]
        Gd = np.repeat((rng.normal(0, 1, n) + np.array([1.0 if d["version"] in A.PLANTED else 0.0 for d in design]))[:, None], 11, 1)
        Am = A.specificity_matrix(Gd, det, design, memes, A.REWORDED, pairs_cache=cache)
        spec_rej.append(A.specificity_test(Am, det, memes, n_perm=400, rng=rng)["p"] < 0.05)
    check(abs(np.mean(twin) - 0.5) < 0.02, f"non-specific detector against salience-matched twins: AUC {np.mean(twin):.3f} (draft all-negatives rule {np.mean(alln):.3f})")
    check(np.mean(h1_rej) <= ALPHA_MC, f"H1 null with a non-specific 'neuron' detector: rejection {np.mean(h1_rej):.3f}")
    check(np.mean(spec_rej) <= ALPHA_MC, f"H3 specificity null (generic 'drama' units, draft-style controls): rejection {np.mean(spec_rej):.3f}")
    # T3: franchise-clustered world with no effect (positive entities and negative franchises share random effects)
    ent_rej, work_rej = [], []
    for rep in range(R):
        score, stratum, ent_of, cl_of, pos, negs = [], [], {}, {}, [], {"s": []}
        k = 0
        for e in range(5):                        # 5 positive entities of 6 near-duplicate works each
            mu = rng.normal(0, 1)
            for _ in range(6):
                score.append(mu + rng.normal(0, 0.2)); stratum.append("s"); ent_of[k] = f"E{e}"; pos.append(k); k += 1
        for c in range(40):                       # 40 negative franchises of 2-8 works and 120 singletons
            mu = rng.normal(0, 1)
            for _ in range(int(rng.integers(2, 9))):
                score.append(mu + rng.normal(0, 0.2)); stratum.append("s"); cl_of[k] = f"C{c}"; negs["s"].append(k); k += 1
        for _ in range(120):
            score.append(rng.normal(0, 1.02)); stratum.append("s"); cl_of[k] = f"single{k}"; negs["s"].append(k); k += 1
        score = np.array(score)
        ent_rej.append(A.entity_level_test(score, pos, ent_of, negs, stratum, cl_of, n_perm=300, rng=rng)["p"] < 0.05)
        work_rej.append(A.stratified_perm_test(score, pos, A.entity_weights(pos, ent_of), negs, stratum, n_perm=300, rng=rng)["p"] < 0.05)
    check(np.mean(ent_rej) <= ALPHA_MC, f"T3 franchise-clustered null: entity-level rejection {np.mean(ent_rej):.3f} (draft work-level test {np.mean(work_rej):.3f})")


# =============================================================================================== synthetic world
LETTERS = np.array(list(string.ascii_lowercase))


def pseudo_words(n, rng):
    out = set()
    while len(out) < n:
        w = "".join(rng.choice(LETTERS, rng.integers(5, 9)))
        if w not in A.STOP:
            out.add("z" + w)
    return sorted(out)


FILLER = pseudo_words(400, RNG)
GENRE = pseudo_words(60, RNG)
CJK = [chr(c) for c in range(0x4E00, 0x4E00 + 3000)]
GENERIC_UNIT = 235          # fires in every slot of every version (equal-stakes controls): no meme specificity


def sentence(words):
    return " ".join(words).capitalize() + "."


def zh_sentence(rng, k=14):
    return "".join(rng.choice(CJK, k)) + "。"


def write_set(name, works):
    d = TMP / "data"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{name}.jsonl", "w") as f:
        for w in works:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")


def feat(name):
    p = TMP / "features" / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_neurons(name, works, agg, sent=None, units=None, full=None, hidden=None):
    for m in MODELS:
        d = feat(name) / f"neurons_{m}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "agg.npy", np.asarray(agg[m] if isinstance(agg, dict) else agg, np.float16))
        if sent is not None:
            np.save(d / "sent.npy", np.asarray(sent[m] if isinstance(sent, dict) else sent, np.float16))
            np.save(d / "units.npy", np.asarray(units, np.int64))
        if full is not None:
            np.save(d / "full.npy", np.asarray(full[m] if isinstance(full, dict) else full, np.float16))
        if hidden is not None:
            np.save(d / "hidden.npy", np.asarray(hidden, np.float16))
        sw = [i for i, w in enumerate(works) for _ in w["sentences"]]
        json.dump({"work_ids": [w["id"] for w in works], "sent_work": sw, "n_units": U, "hidden_layers": [1, 2, 3, 4],
                   "seconds": 0.01 * len(sw), "peak_gb": 5.6}, open(d / "meta.json", "w"))
        json.dump({"seconds": 0.015 * len(sw), "peak_gb": 5.6, "bytes": 10, "calls": len(works), "tokens": 30 * len(sw)},
                  open(feat(name) / f"cost_neurons_{m}.json", "w"))


LATENT = RNG.normal(0, 1, (U, 32))


def write_embeddings(name, works, unitsets, noise=0.6, signal=1.0):
    n = sum(len(w["sentences"]) for w in works)
    for enc in ENCS:
        rows = []
        for w, us in zip(works, unitsets):
            base = LATENT[list(us)].mean(0) if len(us) else np.zeros(32)
            for _ in w["sentences"]:
                s = signal * base + noise * RNG.normal(0, 1, 32)
                rows.append(s / (np.linalg.norm(s) + 1e-9))
        E = np.array(rows) if rows else np.zeros((0, 32))
        assert len(E) == n
        np.save(feat(name) / f"emb_{enc}.npy", E.astype(np.float16))
        json.dump({"seconds": 0.001 * n}, open(feat(name) / f"emb_{enc}.json", "w"))


def hidden_for(works, unitsets):
    rows = []
    for w, us in zip(works, unitsets):
        base = LATENT[list(us), :16].mean(0) if len(us) else np.zeros(16)
        for _ in w["sentences"]:
            rows.append(np.stack([base + RNG.normal(0, s, 16) for s in (2.0, 0.5, 0.8, 1.5)]))
    return np.array(rows)


def agg_for(unitsets, strength=3.0, noise=1.0):
    A_ = RNG.normal(0, noise, (len(unitsets), U))
    for i, us in enumerate(unitsets):
        A_[i, list(us)] += strength
    return {m: A_ + RNG.normal(0, 0.1, A_.shape) for m in MODELS}


def texts_for(wordsets, n_sent=16, own=2, rng=RNG):
    out = []
    for ws in wordsets:
        ws = list(ws)
        out.append([sentence(list(rng.choice(ws, own)) + list(rng.choice(GENRE, 4)) + list(rng.choice(FILLER, 5))) if ws
                    else sentence(list(rng.choice(FILLER, 10))) for _ in range(n_sent)])
    return out


# ---------------------------------------------------------------- Batman DEV-shaped world
COMICS = [("Dark Knight Returns", 1986), ("Year One", 1987), ("The Killing Joke", 1988), ("A Death in the Family", 1988),
          ("Gotham by Gaslight", 1989), ("Arkham Asylum (comic)", 1989), ("Knightfall", 1993), ("The Long Halloween", 1996),
          ("Hush", 2002), ("Batman R.I.P.", 2008), ("Earth One", 2012), ("White Knight", 2017)]
SCREEN = [("Batman Begins", 2005), ("The Dark Knight", 2008), ("Dark Knight Rises", 2012), ("Batman v Superman", 2016),
          ("Under the Red Hood", 2010), ("Arkham Asylum (game)", 2009), ("Batman '89", 1989), ("Joker (2019)", 2019),
          ("The Batman", 2022), ("1943 serial", 1943), ("Batman '66", 1966), ("Batman Returns", 1992), ("Batman Forever", 1995),
          ("Batman & Robin", 1997), ("Mask of the Phantasm", 1993), ("Return of the Joker", 2000), ("Arkham City", 2011),
          ("Arkham Knight", 2015), ("Lego Batman", 2017), ("Batman Ninja", 2018)]


def batman_world():
    rng = RNG
    cu = {c: set(rng.choice(U, 14, replace=False).tolist()) for c, _ in COMICS}
    cw = {c: set(pseudo_words(30, rng)) for c, _ in COMICS}

    def inherit(srcs, keep_w=0.12):
        us, ws = set(), set()
        for s in srcs:
            us |= set(rng.choice(sorted(cu[s]), 8, replace=False).tolist())
            ws |= set(rng.choice(sorted(cw[s]), int(30 * keep_w), replace=False).tolist())
        us |= set(rng.choice(U, 4, replace=False).tolist())
        ws |= set(pseudo_words(30, rng))
        return us, ws
    rows, units, words = [], [], []
    for c, y in COMICS:
        rows.append({"id": c, "group": "batman", "medium": "comic", "year": y})
        units.append(cu[c])
        words.append(cw[c])
    for s, y in SCREEN:
        rows.append({"id": s, "group": "batman", "medium": "film", "year": y})
        if s in A.BAT_ADAPTATIONS:
            us, ws = inherit(A.BAT_ADAPTATIONS[s])
        else:
            us, ws = set(rng.choice(U, 12, replace=False).tolist()), set(pseudo_words(25, rng))
        units.append(us)
        words.append(ws)
    for k in range(69):
        rows.append({"id": "The Lion King" if k == 0 else f"Film {k}", "group": "baseline", "medium": "film", "year": None})
        units.append(set(rng.choice(U, 6, replace=False).tolist()))
        words.append(set(pseudo_words(20, rng)))
    frows, funits, fwords = [], [], []
    for t, (y, srcs) in A.FRESH_ADAPTATIONS.items():
        us, ws = inherit(srcs, keep_w=0.15)
        frows.append({"id": t, "group": "batman", "medium": "film", "year": y})
        funits.append(us)
        fwords.append(ws)
    return rows, units, words, frows, funits, fwords


def drift_words(ws, level, rng):
    ws = sorted(ws)
    k = int(round(len(ws) * level / 100))
    gone = set(rng.choice(ws, k, replace=False).tolist()) if k else set()
    return (set(ws) - gone) | set(pseudo_words(k, rng))


def coded_texts(texts):
    """Give every work a few X-codes, as the name-free texts have."""
    return [[s[:-1] + f" X{1 + (k % 3)}." for k, s in enumerate(t)] for t in texts]


def write_pairs(name, rows, uss, cand_units, task="pairs"):
    with open(feat(name) / f"llm_qwen3-8b_{task}.jsonl", "w") as f:
        for r, us in zip(rows, uss):
            for c, cu in cand_units.items():
                ov = len(us & cu) / 14
                p = float(1 / (1 + np.exp(-(8 * ov - 2 + RNG.normal(0, 0.5)))))
                f.write(json.dumps({"a": r["id"], "b": c, "p_yes": p, "logit": float(np.log(p / (1 - p)))}) + "\n")


def build_batman():
    rows, units, words, frows, funits, fwords = batman_world()
    texts = coded_texts(texts_for(words))
    dev = [dict(r, sentences=t) for r, t in zip(rows, texts)]
    write_set("dev_namefree", dev)
    write_set("dev_original", dev)
    agg = agg_for(units)
    fagg = agg_for(funits)
    n_sent = sum(len(w["sentences"]) for w in dev)
    full = {m: np.repeat(agg[m], [len(w["sentences"]) for w in dev], axis=0) + RNG.normal(0, 0.5, (n_sent, U)) for m in MODELS}
    lat = RNG.normal(0, 1, (n_sent, 5))
    for i in range(5):
        full["qwen3-8b"][:, 200 + i] = lat[:, i] + RNG.normal(0, 0.3, n_sent)
        full["llama3.1-8b"][:, 210 + i] = lat[:, i] + RNG.normal(0, 0.3, n_sent)
        for j in range(3):
            full["qwen3-8b"][:, 170 + 3 * i + j] = RNG.normal(0, 1, n_sent)
    write_neurons("dev_namefree", dev, agg, full=full, hidden=hidden_for(dev, units))
    write_neurons("dev_original", dev, agg, full=full, hidden=hidden_for(dev, units))
    write_embeddings("dev_namefree", dev, units)
    write_embeddings("dev_original", dev, units)
    fresh = [dict(r, sentences=t) for r, t in zip(frows, coded_texts(texts_for(fwords)))]
    write_set("t1e_fresh_namefree", fresh)
    write_neurons("t1e_fresh_namefree", fresh, fagg, hidden=hidden_for(fresh, funits))
    write_embeddings("t1e_fresh_namefree", fresh, funits)
    screen_idx = [i for i, r in enumerate(rows) if r["group"] == "batman" and r["medium"] != "comic"]
    comic_units = {c: units[i] for i, (c, _) in enumerate(COMICS)}
    write_pairs("dev_namefree", [rows[i] for i in screen_idx], [units[i] for i in screen_idx], comic_units)
    write_pairs("dev_namefree", [rows[i] for i in screen_idx], [units[i] for i in screen_idx], comic_units, "pairsfull")
    write_pairs("t1e_fresh_namefree", frows, funits, comic_units)
    for lv in A.DRIFT_LEVELS[1:] + ["style"]:
        L = 5 if lv == "style" else lv
        suf = "_style" if lv == "style" else str(lv)
        dr = [dict(rows[i], sentences=t) for i, t in zip(screen_idx, coded_texts(texts_for([drift_words(words[i], L, RNG) for i in screen_idx])))]
        write_set(f"t1b_drift{suf}", dr)
        su = [units[i] for i in screen_idx]
        write_neurons(f"t1b_drift{suf}", dr, {m: agg[m][screen_idx] + RNG.normal(0, 0.3, (len(screen_idx), U)) for m in MODELS}, hidden=hidden_for(dr, su))
        write_embeddings(f"t1b_drift{suf}", dr, su)
        fr = [dict(r, sentences=t) for r, t in zip(frows, coded_texts(texts_for([drift_words(w, L, RNG) for w in fwords])))]
        write_set(f"t1b_fresh_drift{suf}", fr)
        write_neurons(f"t1b_fresh_drift{suf}", fr, {m: fagg[m] + RNG.normal(0, 0.3, fagg[m].shape) for m in MODELS}, hidden=hidden_for(fr, funits))
        write_embeddings(f"t1b_fresh_drift{suf}", fr, funits)
    zi = screen_idx + [i for i, r in enumerate(rows) if r["group"] == "baseline"]
    zh = [dict(rows[i], sentences=[zh_sentence(RNG) + f" X{1 + k % 3}" for k in range(12)]) for i in zi]
    write_set("t1c_zh", zh)
    zu = [units[i] for i in zi]
    write_neurons("t1c_zh", zh, agg_for(zu, strength=2.5), hidden=hidden_for(zh, zu))
    write_embeddings("t1c_zh", zh, zu)
    fz = [dict(r, sentences=[zh_sentence(RNG) for _ in range(12)]) for r in frows]
    write_set("t1c_fresh_zh", fz)
    write_neurons("t1c_fresh_zh", fz, agg_for(funits, strength=2.5), hidden=hidden_for(fz, funits))
    write_embeddings("t1c_fresh_zh", fz, funits)
    return dev


# ---------------------------------------------------------------- Shakespeare-shaped world
def build_shakespeare():
    sh = json.load(open(HERE / "works_shakespeare.json"))
    plays = [p["id"] for p in sh["plays"]]
    pu = {p: set(RNG.choice(U, 14, replace=False).tolist()) for p in plays}
    pw = {p: set(pseudo_words(30, RNG)) for p in plays}
    names = {p: [f"zname{i}{j}" for j in range(3)] for i, p in enumerate(plays)}
    prow = [{"id": p} for p in plays]
    play_texts = coded_texts(texts_for([pw[p] for p in plays]))
    write_set("t1d_plays_namefree", [dict(r, sentences=t) for r, t in zip(prow, play_texts)])
    write_set("t1d_plays_original", [dict(r, sentences=[s[:-1] + " " + " ".join(names[r["id"]]) + "." for s in t]) for r, t in zip(prow, play_texts)])
    films, fu, fw = [], [], []
    keys = ("primary", "renamed", "h5", "drift_sample", "evidence_tier", "sources", "names_flag", "remake_of_dev")
    for f in sh["films"]:
        src = f["sources"][0]
        loose = f["names_flag"] == "changes_names"
        us = set(RNG.choice(sorted(pu[src]), 8, replace=False).tolist()) | set(RNG.choice(U, 4, replace=False).tolist())
        ws = set(RNG.choice(sorted(pw[src]), 1 if loose else 12, replace=False).tolist()) | set(pseudo_words(30, RNG))
        films.append({"id": f["id"], **{k: f[k] for k in keys}})
        fu.append(us)
        fw.append(ws)
    ft = coded_texts(texts_for(fw))
    write_set("t1d_films_namefree", [dict(r, sentences=t) for r, t in zip(films, ft)])
    write_set("t1d_films_original", [dict(r, sentences=[s[:-1] + (" " + " ".join(names[r["sources"][0]]) if r["names_flag"] == "keeps_names" else "") + "." for s in t])
                                     for r, t in zip(films, ft)])
    fagg = None
    for name, us in (("t1d_plays_namefree", [pu[p] for p in plays]), ("t1d_films_namefree", fu)):
        works = A.read_jsonl(TMP / "data" / f"{name}.jsonl")
        g = agg_for(us)
        if name == "t1d_films_namefree":
            fagg = g
        write_neurons(name, works, g, hidden=hidden_for(works, us))
        write_embeddings(name, works, us)
    with open(feat("t1d_films_namefree") / "llm_qwen3-8b_recognise.jsonl", "w") as f:
        for k, r in enumerate(films):
            f.write(json.dumps({"work": r["id"], "answer": r["id"] if k % 10 == 0 else "unknown"}) + "\n")
    li = [i for i, f in enumerate(films) if f["drift_sample"]]
    for lv in A.DRIFT_LEVELS[1:] + ["style"]:
        L = 5 if lv == "style" else lv
        suf = "_style" if lv == "style" else str(lv)
        dr = [dict(films[i], sentences=t) for i, t in zip(li, coded_texts(texts_for([drift_words(fw[i], L, RNG) for i in li])))]
        write_set(f"t1b_shk_drift{suf}", dr)
        su = [fu[i] for i in li]
        write_neurons(f"t1b_shk_drift{suf}", dr, {m: fagg[m][li] + RNG.normal(0, 0.3, (len(li), U)) for m in MODELS}, hidden=hidden_for(dr, su))
        write_embeddings(f"t1b_shk_drift{suf}", dr, su)
    for name in ("t1d_plays_original", "t1d_films_original"):
        works = A.read_jsonl(TMP / "data" / f"{name}.jsonl")
        write_neurons(name, works, agg_for([set()] * len(works)))


# ---------------------------------------------------------------- planted stories
def build_t2():
    memes = json.load(open(HERE / "memes_for_tracing.json"))["memes"]
    ids = [m["id"] for m in memes]
    conc = {m: i for i, m in enumerate(ids)}
    kwu = {m: 34 + i for i, m in enumerate(ids)}
    proto = {m: RNG.normal(0, 1, 32) for m in ids}
    kdir = {m: RNG.normal(0, 1, 32) for m in ids}
    syn = {m: pseudo_words(6, RNG) for m in ids}          # 'reworded' vocabulary that never appears in stories
    ex_works, ex_full, ex_emb = [], [], []
    for m in memes:
        mid = m["id"]
        rows = [("definition", 0)] + [(st, k) for st in ("explicit", "reworded", "far", "hard_negative") for k in range(5)]
        for st, k in rows:
            if st in ("definition", "explicit"):
                words = list(RNG.choice(FILLER, 6)) + m["keywords"][:2]
            elif st == "hard_negative":
                words = list(RNG.choice(FILLER, 6)) + m["keywords"][:2]
            else:
                words = list(RNG.choice(FILLER, 6)) + list(RNG.choice(syn[mid], 2))
            ex_works.append({"id": f"{mid}#{st}{k}", "meme": mid, "set": st, "sentences": [sentence(words)]})
            v = RNG.normal(0, 1, U)
            if st != "hard_negative":
                v[conc[mid]] += 6 if st != "far" else 5
            if st in ("definition", "explicit", "hard_negative"):
                v[kwu[mid]] += 4.5
            v[GENERIC_UNIT] += 3
            ex_full.append(v)
            e = (proto[mid] if st != "hard_negative" else 0) + (0.6 * kdir[mid] if st in ("definition", "explicit", "hard_negative") else 0) \
                + 0.4 * RNG.normal(0, 1, 32)
            ex_emb.append(e / np.linalg.norm(e))
    write_set("t2_examples", ex_works)
    write_set("t2_examples_ctx", ex_works)
    ex_full = np.array(ex_full)
    hid = np.repeat(RNG.normal(0, 1, (len(ex_works), 1, 16)), 4, 1) + np.array([proto[w["meme"]][:16] if w["set"] != "hard_negative" else np.zeros(16)
                                                                              for w in ex_works])[:, None, :]
    write_neurons("t2_examples", ex_works, ex_full, full=ex_full, hidden=hid)
    write_neurons("t2_examples_ctx", ex_works, ex_full, full=ex_full + RNG.normal(0, 0.2, ex_full.shape))
    for enc in ENCS:
        np.save(feat("t2_examples") / f"emb_{enc}.npy", np.array(ex_emb, np.float16))
    films = [{"id": f"Film {k}", "group": "baseline", "sentences": [sentence(list(RNG.choice(FILLER, 10))) for _ in range(10)]} for k in range(69)]
    bg = [{"id": f"{w['id']}#{j}", "film": w["id"], "group": "baseline", "sentences": [s]} for w in films for j, s in enumerate(w["sentences"])]
    write_set("t2_background", bg)
    write_set("t2_reference", films)
    nb = 690
    bfull = RNG.normal(0, 1, (nb, U))
    write_neurons("t2_background", bg, bfull, full=bfull, hidden=RNG.normal(0, 1, (nb, 4, 16)))
    write_neurons("t2_reference", films, RNG.normal(0, 1, (69, U)), full=bfull + RNG.normal(0, 0.2, bfull.shape))
    for enc in ENCS:
        E = RNG.normal(0, 1, (nb, 32))
        E = (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float16)
        np.save(feat("t2_background") / f"emb_{enc}.npy", E)
        np.save(feat("t2_reference") / f"emb_{enc}.npy", E)
    kd = feat("t2_examples")
    json.dump({m["id"]: m["keywords"][:5] for m in memes}, open(kd / "keywords_qwen3-8b.json", "w"))
    json.dump({m["id"]: m["keywords"][:3] for m in memes}, open(kd / "keywords_llama3.1-8b.json", "w"))
    json.dump({m["id"]: m["keywords"] + ["zzunrelated"] for m in memes}, open(kd / "keywords_expanded-qwen3-8b.json", "w"))
    json.dump({m["id"]: m["keywords"] for m in memes}, open(kd / "keywords_ban-list-aligned.json", "w"))
    json.dump({m["id"]: ["硬币"] for m in memes}, open(kd / "keywords_qwen3-8b_zh.json", "w"), ensure_ascii=False)
    return memes, conc, kwu, proto, kdir


def build_stories(memes, conc, kwu, proto, kdir, ubm):
    design = json.load(open(HERE / "planted_design.json"))["plots"]
    ids = [m["id"] for m in memes]
    kw = {m["id"]: m["keywords"] for m in memes}
    works, full, emb, judge = [], [], [], []
    for p in design:
        for v in A.VERSIONS:
            sents, acts, es = [], [], []
            slots = {3: p["memes"][0], 7: p["memes"][1], 10: p["memes"][2]}
            for k in range(12):
                a = RNG.normal(0, 1, U)
                e = 0.8 * RNG.normal(0, 1, 32)
                words = list(RNG.choice(FILLER, 9))
                m = slots.get(k)
                if m is not None:
                    a[GENERIC_UNIT] += 3                      # every slot is equally eventful (equal-stakes control)
                    if v == "explicit":
                        words += kw[m][:2]
                        a[conc[m]] += 4
                        a[kwu[m]] += 4
                        e += proto[m] + 0.6 * kdir[m]
                    elif v == "banned":
                        a[conc[m]] += 4
                        e += 0.8 * proto[m]
                    elif v == "implicit":
                        a[conc[m]] += 2.5
                        e += 0.5 * proto[m]
                    elif v == "coded":
                        a[conc[m]] += 3
                        e += 0.3 * proto[m]
                    elif v == "decoy":
                        words += kw[m][:2]
                        a[kwu[m]] += 4
                        e += 0.6 * kdir[m] + 0.2 * proto[m]
                sents.append(sentence(words))
                acts.append(a)
                es.append(e / np.linalg.norm(e))
            planted = set(p["memes"]) if v in A.PLANTED else set()
            works.append({"id": f"{p['plot']}-{v}", "plot": p["plot"], "version": v, "memes": p["memes"], "valid": True, "writer": p["writer"],
                          "genre": p["genre"], "exclude_memes": [], "dropped_memes": [], "sentences": sents,
                          "reads_claude": [sorted(planted), sorted(planted)], "reads_other": [sorted(planted)]})
            full.append(np.array(acts))
            emb.append(np.array(es))
            for m in ids:
                base = {"explicit": 0.95, "banned": 0.9, "implicit": 0.7, "coded": 0.75}.get(v, 0.05) if m in planted else 0.05
                pj = float(np.clip(base + RNG.normal(0, 0.05), 0.001, 0.999))
                judge.append({"work": works[-1]["id"], "meme": m, "p_yes": pj, "logit": float(np.log(pj / (1 - pj)))})
    write_set("t2_stories", works)
    F = np.concatenate(full)
    lat = RNG.normal(0, 1, (len(F), 5))
    FL = F.copy()
    for i in range(5):
        F[:, 200 + i] = lat[:, i] + RNG.normal(0, 0.4, len(F))
        FL[:, 210 + i] = lat[:, i] + RNG.normal(0, 0.4, len(F))
    sent = {"qwen3-8b": F[:, ubm["qwen3-8b"]], "llama3.1-8b": FL[:, ubm["llama3.1-8b"]]}
    write_neurons("t2_stories", works, np.array([A.topk_mean(f) for f in full]), sent=sent, units=ubm["qwen3-8b"],
                  hidden=np.repeat(RNG.normal(0, 1, (len(F), 1, 16)), 4, 1))
    E = np.concatenate(emb)
    for enc in ENCS:
        np.save(feat("t2_stories") / f"emb_{enc}.npy", E.astype(np.float16))
        W = np.concatenate([np.array([(x[max(0, k - 1):k + 2]).mean(0) for k in range(len(x))]) for x in emb])
        np.save(feat("t2_stories") / f"emb_{enc}_win3.npy", (W / np.linalg.norm(W, axis=1, keepdims=True)).astype(np.float16))
        D = np.array([x.mean(0) for x in emb])
        np.save(feat("t2_stories") / f"emb_{enc}_doc.npy", (D / np.linalg.norm(D, axis=1, keepdims=True)).astype(np.float16))
    np.save(feat("t2_stories") / "nli_deberta-v3-base.npy", np.clip(RNG.random((len(F), len(ids))) * 0.3, 0, 1).astype(np.float16))
    for task in ("judge", "judge-shared"):
        with open(feat("t2_stories") / f"llm_qwen3-8b_{task}.jsonl", "w") as f:
            for j in judge:
                f.write(json.dumps(j) + "\n")
    with open(feat("t2_stories") / "llm_qwen3-8b_sjudge.jsonl", "w") as f:
        for j in judge:
            f.write(json.dumps(dict(j, p_yes=float(np.clip(j["p_yes"] - 0.1, 0, 1)))) + "\n")
    with open(TMP / "data" / "human_audit.jsonl", "w") as f:
        for w in works[:120]:
            for m in w["memes"]:
                f.write(json.dumps({"kind": "t2_cell", "story": w["id"], "meme": m, "present": w["version"] in A.PLANTED}) + "\n")
    zw = [dict(w, sentences=[zh_sentence(RNG) for _ in w["sentences"]]) for w in works if w["version"] in ("banned", "control")]
    keep = [i for i, w in enumerate(works) if w["version"] in ("banned", "control")]
    write_set("t2_stories_zh", zw)
    Fz = np.concatenate([full[i] for i in keep])
    write_neurons("t2_stories_zh", zw, np.array([A.topk_mean(full[i]) for i in keep]), sent={m: Fz[:, ubm[m]] for m in MODELS}, units=ubm["qwen3-8b"])
    Ez = np.concatenate([emb[i] for i in keep])
    for enc in ENCS:
        np.save(feat("t2_stories_zh") / f"emb_{enc}.npy", Ez.astype(np.float16))
    # T2-long: 12 long texts of 12 stories
    lw, lf = [], []
    for k in range(12):
        pick = RNG.choice(len(design), 12, replace=False)
        vs = ["banned", "implicit"] + ["control"] * 10
        idxs = [A.VERSIONS.index(v) + 6 * int(p) for v, p in zip(vs, pick)]
        lw.append({"id": f"L{k}", "sentences": [s for i in idxs for s in works[i]["sentences"]],
                   "memes_present": sorted({m for i in idxs[:2] for m in works[i]["memes"]}), "exclude_memes": []})
        lf.append(np.concatenate([full[i] for i in idxs]))
    write_set("t2_long", lw)
    FLg = np.concatenate(lf)
    write_neurons("t2_long", lw, np.array([A.topk_mean(f) for f in lf]), sent={m: FLg[:, ubm[m]] for m in MODELS}, units=ubm["qwen3-8b"])
    return works


# ---------------------------------------------------------------- culture pool
def build_t3(memes, conc, ubm):
    pool = json.load(open(HERE / "culture_pool.json"))
    keep = [w for w in pool["works"] if w["role"] != "pool"]
    rest = [w for w in pool["works"] if w["role"] == "pool"]
    clustered = [w for w in rest if (w.get("neg_cluster_size") or 0) >= 2][:120]
    others = [w for w in rest if w not in clustered]
    keep += clustered + [others[i] for i in RNG.choice(len(others), 300, replace=False)]
    json.dump(dict(pool, works=keep), open(TMP / "culture_pool_test.json", "w"))
    bat_units = set(range(100, 120))
    tierA = [m["id"] for m in memes if m["tier"] == "A"]
    works, us, sents_full = [], [], []
    for w in keep:
        u = set(RNG.choice(U, 6, replace=False).tolist())
        if w["role"] == "batman_lineage":
            u |= bat_units
        elif w["role"] == "relative":
            u |= set(RNG.choice(sorted(bat_units), 8, replace=False).tolist())
        rows = []
        for k in range(10):
            a = RNG.normal(0, 1, U)
            if w.get("book_tier") == 1 and k < 3:
                for m in tierA[:5]:
                    a[conc[m]] += 3
            rows.append(a)
        works.append({"id": w["title"], "role": w["role"], "stratum": w["stratum"], "sentences": [sentence(list(RNG.choice(FILLER, 10))) for _ in range(10)]})
        us.append(u)
        sents_full.append(np.array(rows))
    write_set("t3_pool", works)
    F = np.concatenate(sents_full)
    lat = RNG.normal(0, 1, (len(F), 5))
    FL = F.copy()
    for i in range(5):
        F[:, 200 + i] = lat[:, i] + RNG.normal(0, 0.4, len(F))
        FL[:, 210 + i] = lat[:, i] + RNG.normal(0, 0.4, len(F))
    write_neurons("t3_pool", works, agg_for(us)["qwen3-8b"], sent={"qwen3-8b": F[:, ubm["qwen3-8b"]], "llama3.1-8b": FL[:, ubm["llama3.1-8b"]]},
                  units=ubm["qwen3-8b"])
    write_embeddings("t3_pool", works, us)
    dev = A.read_jsonl(TMP / "data" / "dev_namefree.jsonl")
    write_set("t3_dev_masked", dev)
    du = [bat_units if w["group"] == "batman" else set(RNG.choice(U, 6, replace=False).tolist()) for w in dev]
    write_neurons("t3_dev_masked", dev, agg_for(du))
    write_embeddings("t3_dev_masked", dev, du)
    with open(feat("t3_pool") / "llm_qwen3-8b_likeness.jsonl", "w") as f:
        for w, u in zip(works, us):
            p = float(np.clip(len(u & bat_units) / 20, 0.01, 0.99))
            f.write(json.dumps({"work": w["id"], "p_yes": p}) + "\n")
    # T2-N works: 60 natural works with annotations
    nw, nf = [], []
    ids = [m["id"] for m in memes]
    for k in range(140):
        present = list(RNG.choice(ids[:6], 2, replace=False)) if k % 2 == 0 else []
        rows = []
        for j in range(10):
            a = RNG.normal(0, 1, U)
            if j < len(present):
                a[conc[present[j]]] += 4
            rows.append(a)
        nf.append(np.array(rows))
        nw.append({"id": f"natural {k}", "memes_present": present, "annotators": [present, present],
                   "sentences": [sentence(list(RNG.choice(FILLER, 10))) for _ in range(10)]})
    write_set("t2n_works", nw)
    FN = np.concatenate(nf)
    write_neurons("t2n_works", nw, RNG.normal(0, 1, (len(nw), U)), sent={m: FN[:, ubm[m]] for m in MODELS}, units=ubm["qwen3-8b"])


def build_t4():
    rows = []
    trio = {"Dark Knight Returns": 1986, "Year One": 1987, "The Killing Joke": 1988}
    for i in range(72):
        rows.append({"id": f"Bat work {i}", "year": 1943 + i, "origin": "dev" if i % 2 else "test"})
    for n, y in trio.items():
        rows.append({"id": n, "year": y, "origin": "dev"})
    rows.sort(key=lambda r: (r["year"], r["id"]))
    for r in rows:
        r["sentences"] = [sentence(list(RNG.choice(FILLER, 10))) for _ in range(8)]
    P = RNG.random((len(rows), U)) < 0.05
    for i, r in enumerate(rows):
        if r["id"] in trio:
            new = RNG.choice(U, 10, replace=False)
            P[i, new] = True
            P[i + 1:, new] |= RNG.random((len(rows) - i - 1, 10)) < 0.6
    agg = np.where(P, 8.0, 0.0) + RNG.normal(0, 0.3, P.shape)
    write_set("t4_batman", rows)
    write_neurons("t4_batman", rows, {m: agg for m in MODELS})


def build_bench():
    rec = {"neurons": {"setup_fixed_s": 600, "setup_per_meme_s": 2.0, "per_story_s": 0.36, "per_story_per_meme_s": 0.0,
                       "new_meme_per_story_s": 0.0, "store_all_bytes": 1.8e6, "store_selected_bytes": 2e3},
           "emb:qwen3emb-4b": {"setup_fixed_s": 60, "setup_per_meme_s": 0.5, "per_story_s": 0.05, "store_all_bytes": 2e4},
           "emb:e5-large": {"setup_fixed_s": 30, "setup_per_meme_s": 0.5, "per_story_s": 0.03, "store_all_bytes": 1e4},
           "judge": {"setup_fixed_s": 0, "per_story_s": 0.36, "per_story_per_meme_s": 0.2},
           "keywords": {"setup_per_meme_s": 1.0, "per_story_s": 0.001}}
    A.write_json({"recipes": rec, "meta": {"machine": "synthetic"}}, A.RES / "bench.json")


def pipeline():
    print("synthetic pipeline in", TMP)
    t0 = time.time()
    build_batman()
    build_shakespeare()
    memes, conc, kwu, proto, kdir = build_t2()
    ids = [m["id"] for m in memes]
    real = json.load(open(HERE / "dev_units.json"))
    mapped = {}
    for m, v in real["t2_mapped"].items():
        u = conc[m] if v.get("dev_class") == "concept" else GENERIC_UNIT
        mapped[m] = {"qwen3-8b": u, "llama3.1-8b": u, "dev_class": v.get("dev_class"), "dev_rewording_survivor": v.get("dev_rewording_survivor")}
    dev_units = {"t2_mapped": mapped, "t5_concepts": [{"qwen_unit": 200 + i, "null_units": [170 + 3 * i + j for j in range(3)]} for i in range(5)],
                 "label_test_best_of": {m: list(range(120, 170)) for m in MODELS}, "label_test_random": {m: list(range(70, 100)) for m in MODELS}}
    json.dump(dev_units, open(TMP / "dev_units.json", "w"))
    fz = A.dev_freeze(out_path=TMP / "dev_freeze.json", dev_units_file=TMP / "dev_units.json", unit_list_dir=TMP, n_perm=200)
    sel = {mid: fz["t2"]["neurons"]["qwen3-8b"]["main"][mid]["m1"]["units"][0] for mid in ids}
    check(sum(sel[m] == conc[m] for m in ids) >= 32, f"DEV freeze selects the concept unit for most memes ({sum(sel[m] == conc[m] for m in ids)}/34)")
    check(all(fz["t2"]["neurons"]["qwen3-8b"]["far"][m]["m1"]["units"][0] == conc[m] for m in ids[:5]), "far-set detectors are fitted")
    check(set(fz["t2"]["loeo"]["neurons:qwen3-8b"]) == {"m1", "m5"} and fz["t2"]["emb_selected"]["qwen3emb-4b"] in ("cproto", "probe"),
          "leave-one-example-out selection for neurons and embeddings")
    check(fz["t2"]["lda_K"] in (50, 100) and "tfidf-wordnet" in fz["t2"]["keywords"] and "logreg" in fz["t2"], "LDA K, WordNet query and lexical probe frozen")
    check("ban_list_overlap" in fz["t2"] and fz["t2"]["ban_list_overlap"]["ban-list-aligned"][ids[0]] == 1.0, "ban-list overlap published per query")
    check(all(f in fz["lineage_selected"] for f in ("tfidf", "lsa", "neurons:qwen3-8b", "words")), f"lineage grid selected on DEV ({fz['lineage_selected']})")
    check("t5" in fz and all(b == 210 + i for i, (a, b) in enumerate(fz["t5"]["concept_pairs"])), "DEV freeze finds the T5 partners")
    check("in_context_check" in fz["t2"] and "neurons_ctx" in fz["t2"] and fz["t2"]["neurons_ctx"].get("qwen3-8b"), "in-context check and m1ctx variant")
    check(all(k in fz["timers"] for k in ("neurons_select_qwen3-8b", "t1a_grid", "keywords_and_logreg")), "every DEV-freeze step is timed")
    units = sorted(set(fz["sentence_units"]["qwen3-8b"]) | set(fz["sentence_units"]["llama3.1-8b"]) | {GENERIC_UNIT} | set(range(200, 215)))
    ubm = {m: units for m in MODELS}
    build_stories(memes, conc, kwu, proto, kdir, ubm)
    build_t3(memes, conc, ubm)
    build_t4()
    build_bench()
    fz = json.load(open(TMP / "dev_freeze.json"))
    print(f"  world built in {time.time() - t0:.0f}s")

    t1 = A.run_t1(fz, methods=A.lineage_methods(encoders=ENCS, llm=True, hidden=True, full_reader=False), n_perm=200)
    lv0 = t1["t1b_drift"]["0"]
    nm = t1["neuron_method"]["qwen3-8b"]
    check(lv0[nm]["p"] < 0.05 and lv0[t1["rival_methods"]["tfidf"]]["p"] < 0.05, "T1b level 0: neurons and TF-IDF recover sources")
    h4 = t1["H4"]["qwen3-8b"]
    check(h4["by_rival"]["tfidf"]["n"] == 8 + 13, f"H4 pools 8 Batman source clusters and 13 Shakespeare plays ({h4['by_rival']['tfidf']['n']})")
    check(h4["stat"] > 0 and h4["p_iut"] < 0.05 and h4["conjunct_ok"],
          f"H4: neurons beat TF-IDF and LSA under synthetic drift (min d {h4['stat']:.3f}, IUT p {h4['p_iut']:.4f}, conjunct {h4['conjunct_ok']})")
    check("retention" in h4 and "degradation_descriptive" in h4 and "style_control_advantage" in h4 and "components_clustering" in h4,
          "H4 secondaries: retention, descriptive degradation, style control, components clustering")
    check("t1b_source_leak" in t1 and all("pooled_p" in v for v in t1["t1b_source_leak"].values()), "source-leak index computed")
    check(t1["t1c_zh"][nm]["p"] < 0.05 and t1["t1c_zh"][t1["rival_methods"]["tfidf"]]["p"] > 0.05, "T1c: neurons cross languages, TF-IDF does not")
    ls_q, ls_l = A.resolve("llmsum:qwen3-8b"), A.resolve("llmsum:llama3.1-8b")
    check(ls_q in lv0 and "stat" in lv0[ls_q], "LLM summary-pair lineage method runs (logit similarity)")
    check(lv0.get(ls_l, {}).get("error"), "missing LLM ratings are reported, not imputed")
    h5 = t1["H5"]["qwen3-8b"]
    check(h5["stat"] > 0 and h5["p_iut"] < 0.05, f"H5: neurons beat TF-IDF and LSA on renamed synthetic adaptations (d {h5['stat']:.3f})")
    check(set(t1["t1d"]) >= {"h5", "renamed_all_evidence", "h5_without_single_film_plays", "h5_unrecognised", "faithful"}, "T1d sensitivity subsets run")
    check(t1["t1d_positive_control"][A.resolve("tfidf")]["stat"] < 0.05 and t1["t1d_informative_methods"], "T1d positive control and informativeness")
    check(f"{nm}|resid_on_words" in t1["t1d"]["h5"], "residual (beyond words) test runs")
    check(len(t1["t1e_fresh"]["targets"]) == 7 and "reuse_audit" in t1, "T1e has the 7 fresh adaptations and the reuse audit")
    A.write_json(t1, A.RES / "t1.json")

    t2 = A.run_t2(fz)
    bym = t2["by_method"]
    h1 = t2["H1"]["qwen3-8b"]
    check(h1["stat"] > 0.2 and h1["p_iut"] < 0.01 and h1["neutral_consistent"],
          f"H1: neurons beat the evasion-aware analyst and the lexical probe on reworded memes (min dAUC {h1['stat']:.2f})")
    check(bym["keywords:qwen3-8b"]["mean"]["explicit"] > 0.7, "keywords work on explicit wording (manipulation check)")
    check("p_iut" in t2["H2"]["qwen3-8b"] and "superiority_p_iut" in t2["H2"]["qwen3-8b"], "H2 non-inferiority and superiority (intersection-union)")
    h3 = t2["H3"]["qwen3-8b"]
    check(len(h3["concept_units"]) == 11 and h3["stat"] > 0.2 and h3["p"] < 0.01, f"H3: the 11 concept units are meme-specific (specificity {h3['stat']:.2f})")
    check(h3["concept_minus_name_word"] > 0.1, "H3 secondary: concept units more specific than the name/word units")
    check(abs(bym["length"]["mean"]["reworded"] - 0.5) < 0.1, "length floor is at chance on twin pairs")
    check(bym["judge:qwen3-8b"]["mean"]["reworded"] > 0.9 and "judge-shared:qwen3-8b" in bym and "nli:deberta-v3-base" in bym, "LLM judges and NLI are read")
    check("embwin:qwen3emb-4b" in bym and "embdoc:qwen3emb-4b" in bym and "keywords:logreg" in bym, "window, document and lexical-probe rivals run")
    check("lda" in bym and "topics:qwen3emb-4b" in bym, "topic rivals run")
    check(abs(bym["keywords:qwen3-8b"]["mean"]["decoy_vs_control"] - 1.0) < 0.05, "S11 construction check: keywords rank decoys above controls")
    sec = t2["secondary"]
    check(any(k.startswith("S8_") for k in sec) and any(k.startswith("S9_") for k in sec) and any(k.startswith("Scoded_") for k in sec)
          and "m5_minus_m1_qwen3-8b" in sec, "T2 secondaries S8, S9, coded and m1-vs-m5 computed")
    check(t2["human_audit"]["pass"] and t2["reader_kappa"]["banned"]["claude_claude"] is not None
          and t2["reader_kappa"]["banned"]["claude_cross_family"] is not None and t2["cross_family_check"]["pass"], "human audit, reader kappas and cross-family check")
    check(t2["zh"]["S12"]["p"] < 0.05 and t2["natural"]["S13"]["p"] < 0.05 and t2["long"]["by_method"], "Chinese, natural-text and long-text secondaries run")
    check(bym["neurons:qwen3-8b"]["by_writer_reworded"] and "itt_reworded" in bym["neurons:qwen3-8b"], "per-writer and intention-to-treat results")
    A.write_json(t2, A.RES / "t2.json")

    t3 = A.run_t3(fz, pool_file=TMP / "culture_pool_test.json", n_perm=300)
    bq = t3["by_method"]["neurons:qwen3-8b"]
    check(bq["S14_positive_control_lineage_auc"] > 0.95, "T3 positive control (Batman lineage on top)")
    check(bq["relatives_all|batman"]["stat"] > 0.7 and bq["relatives_all|batman"]["p"] < 0.05, "T3 relatives rank high (entity-level test)")
    lay = t3["by_method"]["layer|neurons:qwen3-8b"]["S16_book_tier1|layer"]
    check(lay["stat"] > 0.7 and lay["n_entities"] == 3, "T3 orphan-vigilante layer finds tier-1 book family (3 entities)")
    check("directional_contrasts" in t3 and "category_strata_sensitivity" in t3 and bq.get("contested_mean_percentile") is not None,
          "T3 directional contrasts, category-strata sensitivity, contested works reported")
    check("llm:qwen3-8b" in t3["by_method"], "T3 LLM likeness read")
    t4 = A.run_t4(fz, n_perm=100, n_sims=30)
    check(t4["neurons:qwen3-8b"]["all"]["innovators"]["p"] < 0.05, f"T4 innovators detected (p {t4['neurons:qwen3-8b']['all']['innovators']['p']:.3f})")
    check("S19_neurons:qwen3-8b_vs_words" in t4 and "batch_effect_neurons:qwen3-8b" in t4, "T4 comparative selection test and batch-effect check")
    t5 = A.run_t5(fz)
    check(t5["random_films_primary"]["test"]["p"] < 0.05, "T5 concept twins beat matched nulls on held-out random films")
    items = A.t6_prepare(n_units=10, dev_units_file=TMP / "dev_units.json")
    arms = {it["arm"] for it in items["items"]}
    check(len(items["items"]) > 0 and arms == {"random", "best_of"} and any(k.startswith("lda|") for k in items["units"]), f"T6 items built in two arms ({len(items['items'])} items)")
    A.write_json({k: {u: {"label": "a label", "description": "a description"} for u in us} for k, us in items["units"].items()}, A.RES / "label_test_labels.json")
    lab = [{"kind": it["kind"], "arm": it["arm"], "unit": it["unit"], "item": it["item"], "judge": j,
            "correct": bool(RNG.random() < (0.8 if it["kind"].startswith("neurons") else 0.6))} for it in items["items"] for j in range(2)]
    A.write_json(lab, A.RES / "label_test_answers.json")
    t6 = A.run_t6()
    k0 = next(k for k in t6["kinds"] if k.startswith("neurons:qwen3-8b"))
    check(t6["kinds"][k0]["vs_floor_mcnemar"] is not None and t6["kinds"][k0]["judge_kappa"] is not None, "T6 label test: McNemar vs floor and judge kappa")
    cost = A.run_cost()
    check("frontier" in cost and cost["H6"]["qwen3-8b"]["dominated_by"] is not None and cost["curves"]["neurons"], "cost: recipes, curves, frontier and H6")
    A.write_json(cost, A.RES / "cost.json")
    prim = A.run_primary(t1, t2, cost=cost)
    q = prim["models"]["qwen3-8b"]
    check(set(q["holm"]) == {"H1", "H2", "H3", "H4", "H5"} and "H2_superiority_supported" in q and q["H6"]["verdict"], f"primary verdicts {q['verdict']}")
    check(q["verdict"]["H4"] == "inconclusive" and q["p_entered"]["H4"] == 1.0,
          "H4 is inconclusive (and enters Holm with p = 1) when the drift did not remove the banned words")
    check(q["verdict"]["H1"].startswith("supported") and q["verdict"]["H3"].startswith("supported"), "H1 and H3 supported in the planted world")
    A.write_json(prim, A.RES / "primary.json")
    man = A.feature_manifest()
    check(len(man) > 50, f"F3 feature manifest ({len(man)} files)")
    print(f"  pipeline finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    try:
        unit_tests()
        null_worlds()
        if "--unit" not in sys.argv:
            pipeline()
    finally:
        if "--keep" not in sys.argv:
            shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(FAILS)} failure(s)" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)
