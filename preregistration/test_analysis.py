"""Tests for the frozen analysis (preregistration/analysis.py, prereg-v3) on SYNTHETIC data with known answers.

No DEV or TEST text and no model is used: the test builds a small fake world in a temporary folder (pseudo-word texts,
random neuron aggregates with planted inheritance, random embeddings, fake LLM outputs) in the exact file formats
that extract_compact.py and readers.py write, then runs every runner end to end. Real metadata files from preregistration/
(memes_for_tracing.json, planted_design.json, works_shakespeare.json, culture_pool.json, dev_units.json) supply the
structure only. It has three parts:
  unit tests       every helper on hand-made inputs with exact answers, including the audit's counting rules and gates,
                   the operating points' tie-breaking, a frame-leak world for the overlap analysis, the code-term
                   check (Q2c), the meme-drop rule, the manipulation gates, 'not run' and the new verdicts
  null worlds      type-I error of every primary statistic (and of the draft statistics they replace) over 200+
                   replicates of a world with no effect: H1/H2 (no method difference), H3 (a detector with no meme
                   specificity), a non-specific T2 detector against salience-matched controls, and the overlap
                   analysis's meme-level test with two exchangeable methods
  pipeline         the whole analysis on a synthetic world with planted effects: the DEV freeze, T2 (H1-H3, S-coded,
                   the overlap analysis, T2-long, the size of every test family), the audit page served on 127.0.0.1
                   (blinding, row format), T6 on the T2 stories, cost (H6 and its rival set), 'primary' (Holm over
                   H1-H3) and the F3 hash check

Checks labelled "(deferred)" exercise the code of the tests deferred to a later phase (T1 with H4 and H5, T2-zh, T2-N,
T3, T4, T5; PREREGISTRATION.md section 15). That code is kept, and kept working, but it is not part of this
pre-registration.

  .venv/bin/python preregistration/test_analysis.py          # everything (about 5-10 minutes)
  .venv/bin/python preregistration/test_analysis.py --unit   # unit tests and null worlds only
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
    check(A.verdict3(True, True) == "supported, below SESOI" and A.verdict3(True, True, note="per-protocol only") == "supported, below SESOI (per-protocol only)",
          "a significant effect credibly below the SESOI: 'supported, below SESOI'")
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
    coded["code_terms"] = {"a": "silver disc", "b": "plant", "c": "green bath"}
    longf = story("banned", ["He lets the small spinning silver disc on the old table choose which of the two roads to take tonight.",
                             "They plant the old man at dawn.", "She takes the green bath today."])
    check(A.story_checks(coded, plot, memes, [expl])["valid"], "story checks pass a matched coded version")
    check(any("Q1b" in f for f in A.story_checks(longf, plot, memes, [expl])["failures"]), "story checks: Q1b catches a fill 15% longer")
    bad = story("coded", ["He tosses the funerary disc.", "They plant the old man at dawn.", "She takes the green bath today."])
    bad["code_terms"] = coded["code_terms"]
    check(any(f.startswith("Q2 ") for f in A.story_checks(bad, plot, memes, [expl])["failures"]), "story checks: Q2 bans derivations in coded versions")
    nocode = dict(coded, code_terms=None)
    q2c = lambda st, sib: [f for f in A.story_checks(st, plot, memes, sib)["failures"] if f.startswith("Q2c")]
    dawn = dict(coded, code_terms={"a": "silver disc", "b": "dawn", "c": "green bath"})        # 'dawn' is also in the explicit version
    funer = dict(coded, code_terms={"a": "silver disc", "b": "funeral rite", "c": "green bath"})   # on meme b's ban list
    twice = story("coded", ["He lets the silver disc choose his path, and the disc spins.", "They plant the old man at dawn.", "She takes the green bath today."])
    twice["slots"] = {"a": [2, 3], "b": [8], "c": [14]}
    twice["code_terms"] = coded["code_terms"]
    check(q2c(nocode, [expl]) and q2c(dawn, [expl]) and q2c(funer, [expl]) and any("fewer than 2" in f for f in q2c(twice, []))
          and not q2c(coded, [expl]) and q2c(dict(expl, sentences=[s.replace("coin", "silver disc") for s in expl["sentences"]]), [coded]),
          "Q2c: a code term is required, used in its slot (twice in a 2-sentence slot), absent from other versions and from every ban list")
    recs = [("P1", "explicit", {"a"}), ("P1", "banned", {"a"}), ("P2", "banned", {"a"}), ("P3", "explicit", {"a"}), ("P4", "implicit", {"a"}),
            ("P5", "banned", {"b"}), ("P6", "banned", {"b"}), ("P7", "banned", {"b"}), ("P8", "explicit", {"b"})]
    dr, cnt = A.meme_drop(recs)
    check(dr == ["b"] and cnt == {"a": 3, "b": 4}, "meme drop counts failing plots, not versions (one plot failing twice counts once)")
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
    # T2 overlap analysis on the real design: operating points and exact counts in a planted world
    plots = json.load(open(HERE / "planted_design.json"))["plots"]
    design = [{"plot": p["plot"], "version": v, "memes": p["memes"]} for p in plots for v in A.VERSIONS]
    memes = sorted({m for d in design for m in d["memes"]})
    n = len(design)
    sc = np.full((n, len(memes)), np.nan)
    ctrl = [i for i, d in enumerate(design) if d["version"] == "control"]
    sc[ctrl] = np.arange(len(ctrl))[:, None]
    op = A.operating_points(sc, design, memes)
    k_ = int(np.floor(0.1 * len(ctrl)))
    check(np.allclose(op["realised"], k_ / len(ctrl)) and op["thr"][0] == len(ctrl) - 14, f"operating point flags floor(10%) of the controls (FPR {op['realised'][0]:.3f})")
    ties = np.where(np.random.default_rng(9).random((n, len(memes))) < 0.95, 0.0, 1.0)      # a keyword query with no hit on 95% of stories
    opt = A.operating_points(ties, design, memes, key="ties")
    check(np.allclose(opt["realised"], k_ / len(ctrl)), f"tie-heavy method: seeded tie-breaking gives the same realised FPR ({np.nanmin(opt['realised']):.3f}-{np.nanmax(opt['realised']):.3f})")
    rng = np.random.default_rng(5)
    def planted_scores(found):
        s = rng.normal(0, 0.1, (n, len(memes)))
        for i, d in enumerate(design):
            for j, m in enumerate(memes):
                if d["version"] != "control":
                    s[i, j] = 1.0 if (m in found and m in d["memes"] and d["version"] in A.PLANTED) else -1.0
                elif m in d["memes"]:
                    s[i, j] = -0.5                    # the meme's own-plot controls are clean, so the twin rule keeps exact counts
        return s
    sa, sb = planted_scores(set(memes[0:10])), planted_scores(set(memes[5:15]))
    ov = A.overlap_analysis({"neurons:qwen3-8b": sa, "judge:qwen3-8b": sb}, design, memes, n_boot=200, n_perm=500)
    r = ov["pairs"]["neurons:qwen3-8b|judge:qwen3-8b"]["reworded"]
    check(r["counts"] == {"both": 120, "a_only": 120, "b_only": 120, "neither": 34 * 24 - 360} and r["n_cells"] == 34 * 24,
          f"overlap counts in a planted world ({r['counts']})")
    check(r["memes_a_finds_b_misses"] == memes[0:5] and r["memes_b_finds_a_misses"] == memes[10:15]
          and r["p_a_finds_b_misses"] < 0.01 and r["p_b_finds_a_misses"] < 0.01, "overlap: meme-level lists and plot-swap tests find both directions")
    check(r["ci95"]["a_only"][0] < r["shares"]["a_only"] < r["ci95"]["a_only"][1] and abs(r["a_only_minus_b_only"]) < 1e-12,
          "overlap: bootstrap intervals bracket the shares")
    check(ov["pairs"]["neurons:qwen3-8b|judge:qwen3-8b"]["coded"]["counts"]["a_only"] == 60 and "decoy_false_alarms" in ov
          and ov["decoy_false_alarms"]["neurons:qwen3-8b|judge:qwen3-8b"]["counts"]["both"] == 0, "overlap by condition and on decoys")
    sec = A.overlap_secondaries({"pairs": {"neurons:qwen3-8b|keywords:logreg": {"reworded": dict(r)}}}, models=["qwen3-8b"])
    check(list(sec) == ["O1_neurons:qwen3-8b_finds_what_keywords:logreg_misses|reworded"] and sec[list(sec)[0]]["predicted"], "O1 test built from the overlap pairs")
    # frame-leak world: method L scores a meme high on EVERY version of the plots that plant it (it reads the frame, blind
    # to the slot); method T detects the slot. The threshold-only rule credits L; the twin rule does not.
    leak = rng.normal(0, 1, (n, len(memes)))
    true_ = rng.normal(0, 1, (n, len(memes)))
    for i, d in enumerate(design):
        for j, m in enumerate(memes):
            if m in d["memes"]:
                leak[i, j] = 4 + rng.normal(0, 0.1)
                if d["version"] in A.PLANTED:
                    true_[i, j] += 4
    fl = A.overlap_analysis({"neurons:qwen3-8b": leak, "judge:qwen3-8b": true_}, design, memes, n_boot=50, n_perm=200, conditions=("reworded",))
    flt = A.overlap_analysis({"neurons:qwen3-8b": leak, "judge:qwen3-8b": true_}, design, memes, n_boot=50, n_perm=0, conditions=("reworded",),
                             tested=(), rule="threshold")
    rl = fl["pairs"]["neurons:qwen3-8b|judge:qwen3-8b"]["reworded"]
    check(fl["detection_rate"]["neurons:qwen3-8b"]["reworded"] < 0.05 and not rl["memes_a_finds_b_misses"] and rl["p_a_finds_b_misses"] > 0.5
          and len(rl["memes_b_finds_a_misses"]) == 34 and flt["detection_rate"]["neurons:qwen3-8b"]["reworded"] > 0.95
          and fl["own_control_false_alarm"]["neurons:qwen3-8b"]["reworded"]["mean"] > 0.9
          and fl["own_control_false_alarm"]["judge:qwen3-8b"]["reworded"]["mean"] < 0.3,
          f"frame-leak world: the twin rule does not credit a method that reads the frame (detection {fl['detection_rate']['neurons:qwen3-8b']['reworded']:.2f}; "
          f"threshold-only rule {flt['detection_rate']['neurons:qwen3-8b']['reworded']:.2f}; own-control false alarms "
          f"{fl['own_control_false_alarm']['neurons:qwen3-8b']['reworded']['mean']:.2f})")
    # minimum usable cells: a meme with only 4 usable reworded cells is listed apart, not in the meme lists
    d4 = [dict(d) for d in design]
    kept = 0
    for d in d4:
        if memes[0] in d["memes"] and d["version"] in A.REWORDED:
            kept += 1
            if kept > 4:
                d["exclude_memes"] = [memes[0]]
    o4 = A.overlap_analysis({"neurons:qwen3-8b": sa, "judge:qwen3-8b": sb}, d4, memes, n_boot=50, n_perm=200, conditions=("reworded",))
    r4 = o4["pairs"]["neurons:qwen3-8b|judge:qwen3-8b"]["reworded"]
    check(memes[0] not in r4["memes_a_finds_b_misses"] and r4["memes_below_min_cells"].get(memes[0]) == 4 and memes[1] in r4["memes_a_finds_b_misses"],
          "overlap: memes with fewer than 6 usable cells are reported apart, not listed or tested")
    d5 = [dict(d) for d in design]
    kept = 0
    for d in d5:
        if memes[0] in d["memes"] and d["version"] == "coded":
            kept += 1
            if kept > 4:
                d["exclude_memes"] = [memes[0]]
    sc5 = {"neurons:qwen3-8b": sa + planted_scores(set(memes)), "emb:qwen3emb-4b": sb}
    bym5 = {}
    for k, v in sc5.items():
        r5 = {c: A.detection_aucs(v, d5, memes, A.CONDITIONS[c]) for c in ("coded", "banned", "explicit", "implicit", "reworded")}
        r5["mean"] = {c: float(np.nanmean(list(x.values()))) for c, x in r5.items()}
        bym5[k] = r5
    c5 = A._t2_coded(sc5, bym5, d5, memes, {m: j for j, m in enumerate(memes)}, None)
    t5_ = c5["tests"].get("neurons:qwen3-8b|emb:qwen3emb-4b", {})
    check(c5["memes_below_min_pairs"] == [memes[0]] and c5["n_coded_pairs"][memes[0]] == 4 and t5_.get("n") == 33 and "plot_level" in t5_,
          "S-coded: memes with fewer than 6 usable coded pairs are listed apart and leave both tests")
    # A7 audit sample: stratified by version, one version per plot, 3 design memes and 2 distractors per story, 5 balanced blocks
    works = [{"id": f"{d['plot']}-{d['version']}", "valid": True, **d} for d in design]
    smp = A.human_audit_sample(works, memes)
    from collections import Counter
    check(Counter(s["version"] for s in smp) == Counter(A.AUDIT_STRATA) and len({s["plot"] for s in smp}) == len(smp) == 60
          and all(len(s["memes_asked"]) == 5 and set(next(w for w in works if w["id"] == s["story"])["memes"]) <= set(s["memes_asked"]) for s in smp)
          and smp == A.human_audit_sample(works, memes), "A7 audit sample: 60 stories by version, one per plot, 5 questions each, reproducible")
    check(all(Counter(s["version"] for s in smp if s["block"] == b) == Counter({v: 2 for v in A.VERSIONS}) for b in range(1, 6))
          and [s["audit_id"] for s in smp] == [f"A{k:02d}" for k in range(1, 61)] and [s["block"] for s in smp] == sorted(s["block"] for s in smp),
          "A7 audit sample: 5 blocks of 12 (2 of each version), opaque ids A01-A60 in presentation order")
    byw = {w["id"]: w for w in works}
    truth = lambda it, m: m in byw[it["story"]]["memes"] and byw[it["story"]]["version"] in A.PLANTED
    rows = [{"kind": "t2_cell", "audit_id": it["audit_id"], "meme": m, "present": truth(it, m)} for it in smp for m in it["memes_asked"]]
    first = smp[0]
    changed = [{"kind": "t2_cell", "audit_id": first["audit_id"], "meme": m, "present": not truth(first, m)} for m in first["memes_asked"]] + rows
    au = A.audit_summary(smp, changed, byw)
    check(au["n_cells"] == 300 and au["n_stories"] == 60 and au["complete"] and au["pass_reworded"] and au["pass_coded"]
          and au["gates"]["presence_reworded"]["n_answered"] == 60 and au["gates"]["absence_controls"]["n_agree"] == 60,
          "audit: a changed answer counts once, and only the last answer counts")
    ctrl_cells = [(it["audit_id"], m) for it in smp for m in it["memes_asked"] if byw[it["story"]]["version"] in ("control", "decoy") and m in byw[it["story"]]["memes"]]
    uns = rows + [{"kind": "t2_cell", "audit_id": a, "meme": m, "present": None} for a, m in ctrl_cells[:5]]
    au2 = A.audit_summary(smp, uns, byw)
    uns8 = rows + [{"kind": "t2_cell", "audit_id": a, "meme": m, "present": None} for a, m in ctrl_cells[:7]]
    au3 = A.audit_summary(smp, uns8, byw)
    check(au2["n_unsure"] == 5 and au2["gates"]["absence_controls"]["n_answered"] == 55 and au2["gates"]["absence_controls"]["agreement"] == 1.0
          and au2["pass_reworded"] and not au3["gates"]["absence_controls"]["pass"] and not au3["pass_reworded"],
          "audit: 'unsure' answers leave the denominator and are counted; more than 10% unsure fails the gate")
    part = [r for r in rows if not (r["audit_id"] == "A01" and r["meme"] in first["memes_asked"][:2])]
    au4 = A.audit_summary(smp, part, byw)
    part2 = [r for r in rows if int(r["audit_id"][1:]) <= 36 or r["audit_id"] == "A40"]
    au5 = A.audit_summary(smp, part2, byw)
    check(au4["n_stories"] == 59 and au4["blocks_complete"] == [2, 3, 4, 5] and au4["complete"] and au4["n_stories_scored"] == 48
          and au5["n_stories"] == 37 and au5["blocks_complete"] == [1, 2, 3] and not au5["complete"] and not au5["pass_reworded"],
          "audit: a story counts only with all 5 answers, only whole blocks are scored, complete = at least 4 blocks")
    wrong = [dict(r, present=not r["present"]) if (r["audit_id"], r["meme"]) in set(ctrl_cells[:k]) else r for k in (6,) for r in rows]
    au6 = A.audit_summary(smp, wrong, byw)
    check(A.audit_max_errors(60, A.AUDIT_ABSENCE_LB) == 7 and A.audit_max_errors(48, A.AUDIT_ABSENCE_LB) == 5 and A.audit_max_errors(60, A.AUDIT_PRESENCE_LB) == 10
          and au6["gates"]["absence_controls"]["pass"] and not A.audit_summary(smp, [dict(r, present=not r["present"]) if (r["audit_id"], r["meme"]) in set(ctrl_cells[:8]) else r
                                                                                for r in rows], byw)["gates"]["absence_controls"]["pass"],
          "audit gates: one-sided 90% lower bound (absence >= 0.80: at most 7 errors in 60, 5 in 48; presence >= 0.75: 10 in 60)")
    ex = [dict(w) for w in works]
    for w in ex:
        if w["version"] == "coded":
            w["exclude_memes"] = list(w["memes"])
    au7 = A.audit_summary(smp, [dict(r, present=False) if byw[next(i for i in smp if i["audit_id"] == r["audit_id"])["story"]]["version"] == "coded" else r
                                for r in rows], {w["id"]: w for w in ex})
    au8 = A.audit_summary(smp, [dict(r, present=False) if byw[next(i for i in smp if i["audit_id"] == r["audit_id"])["story"]]["version"] == "coded" else r
                                for r in rows], byw)
    check(au8["pass_reworded"] and not au8["pass_coded"] and au7["gates"]["presence_coded"]["n_cells"] == 0
          and au7["all_cells"]["presence_coded"]["n_cells"] == 30,
          "audit: coded cells gate S-coded only; gates use the per-protocol cells, every cell is reported beside them")
    g = A.t2_manipulation_gates(None, None, 0.9)
    g2 = A.t2_manipulation_gates(au, None, None)
    g3 = A.t2_manipulation_gates(au5, {"pass_reworded": True, "pass_coded": False}, 0.9)
    check(not g["reworded_ok"] and not g["coded_ok"] and not g2["reworded_ok"] and g3["source"] == "cross-family reader" and g3["reworded_ok"]
          and not g3["coded_ok"] and A.t2_manipulation_gates(au, None, 0.9)["reworded_ok"],
          "manipulation gates: none available or no explicit-wording check -> fail; an incomplete audit falls back to the cross-family reader")
    hh = {"p_iut": 0.001, "neutral_consistent": True, "falsified": False, "itt_p_iut": 0.001}
    t2f = {"by_method": {"keywords:qwen3-8b": {"mean": {"explicit": 0.9}}}, "human_audit": au, "cross_family_check": None,
           "H1": {m: dict(hh) for m in MODELS}, "H2": {m: dict(hh, superiority_p_iut=0.5) for m in MODELS}, "H3": {m: {"p": 0.001, "falsified": False} for m in MODELS}}
    vq = lambda t: A.run_primary(t, freeze={}, cost={})["models"]["qwen3-8b"]["verdict"]
    nochk = dict(t2f, human_audit=None)
    nokw = dict(t2f, by_method={})
    below = dict(t2f, H1={m: dict(hh, falsified=True) for m in MODELS})
    nr = dict(t2f, not_run="only 90 plots have all six versions valid (at least 100 are needed)")
    check(set(vq(t2f).values()) == {"supported"} and set(vq(nochk).values()) == {"inconclusive"} and set(vq(nokw).values()) == {"inconclusive"}
          and vq(below)["H1"] == "supported, below SESOI" and all(v.startswith("not run") for v in vq(nr).values()),
          "primary: inconclusive with no manipulation check or no explicit-wording check; 'supported, below SESOI'; 'not run' below minimum completion")
    vw = [dict(w, valid=not (w["version"] == "decoy" and int(w["plot"][1:]) <= 40)) for w in works]
    v1, v2 = A.t2_viability(works, memes), A.t2_viability(vw, memes)
    v3 = A.t2_viability([dict(w, dropped_memes=memes[:7]) for w in works], memes)
    check(v1["ok"] and not v2["ok"] and v2["plots_all_six_valid"] == 96 and not v3["ok"] and v3["memes_not_dropped"] == 27,
          "minimum viable completion of T2: at least 100 plots with all six versions valid and 28 memes not dropped")
    # deferred tests' helpers (T4, T5): code kept for the later phase
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
        check(np.mean(new) <= ALPHA_MC, f"(deferred) H4 null ({label}): head-to-head rejection {np.mean(new):.3f} (draft degradation statistic {np.mean(old):.3f})")
    # neurons flat at chance cannot pass H4's conjunct (their own lineage test at levels 75 and 100)
    src = [[t % 12] for t in range(9)]
    hub = A.hub_groups_fixed(20, list(range(9, 20)), 12)
    ps = [A.lineage_test(rng.normal(0, 1, (20, 12)), list(range(9)), src, np.ones((9, 12), bool), hub, n_perm=200, rng=rng)["p"] for _ in range(R)]
    check(np.mean(np.array(ps) < 0.05) <= ALPHA_MC, f"(deferred) lineage null with no signal: rejection {np.mean(np.array(ps) < 0.05):.3f} (H4 conjunct cannot pass at chance)")
    # time-consistent lineage null with a Batman-like time rule
    cyear = np.array([1986, 1987, 1988, 1988, 1989, 1989, 1993, 1996, 2002, 2008, 2012, 2017])
    years = [2005, 2008, 2012, 2016, 2010, 2009, 1989, 2019, 2022]
    srcs = [[1], [7, 2], [6, 0], [0], [3], [5], [2], [2], [1, 7]]
    el = np.array([[c <= y for c in cyear] for y in years])
    ps = [A.lineage_test(rng.normal(0, 1, (20, 12)), list(range(9)), srcs, el, hub, n_perm=200, rng=rng)["p"] for _ in range(R)]
    check(np.mean(np.array(ps) < 0.05) <= ALPHA_MC, f"(deferred) time-consistent lineage null: rejection {np.mean(np.array(ps) < 0.05):.3f}")
    # H5: advantage over 13 plays, no difference
    rej = np.mean([A.signflip(rng.normal(0, 0.22, 13), alternative="greater")["p"] < 0.05 for _ in range(R)])
    check(rej <= ALPHA_MC, f"(deferred) H5 null (13 plays): rejection {rej:.3f}")
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
    # overlap null: two exchangeable methods (the same meme, plot and cell signal, independent noise), meme-level test
    cells = A._overlap_cells(design, memes, A.REWORDED)
    eng = A._overlap_engine(cells, len(memes), 10, 400, "nullworld")
    ok = np.ones(len(cells["rows"]), bool)
    ov_rej = []
    for rep in range(R):
        z = rng.normal(0, 1.0, len(memes))[cells["meme"]] + rng.normal(0, 0.5, cells["n_plots"])[cells["plot"]] + rng.normal(0, 0.5, len(ok))
        da, db = z + rng.normal(0, 0.7, len(ok)) > 0, z + rng.normal(0, 0.7, len(ok)) > 0
        r = A.overlap_pair(da, db, ok, cells, eng, memes)
        ov_rej.append(r["p_a_finds_b_misses"] < 0.05)
    check(np.mean(ov_rej) <= ALPHA_MC, f"overlap null (exchangeable methods): meme-level 'finds what the other misses' rejection {np.mean(ov_rej):.3f}")
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
    check(np.mean(ent_rej) <= ALPHA_MC, f"(deferred) T3 franchise-clustered null: entity-level rejection {np.mean(ent_rej):.3f} (draft work-level test {np.mean(work_rej):.3f})")


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
        ws = sorted(ws)  # sorted: set order varies with hash randomisation
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
    works, full, emb, judge, hid = [], [], [], [], []
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
                hv = RNG.normal(0, 1, 16)               # the model's hidden state reads the slot in context (S-coded's context-using rival)
                if m is not None and v in A.PLANTED:
                    hv = hv + {"explicit": 1.5, "banned": 1.5, "implicit": 1.0, "coded": 1.3}[v] * proto[m][:16]
                hid.append(hv)
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
                  hidden=np.repeat(np.array(hid)[:, None, :], 4, 1))
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
    byid = {w["id"]: w for w in works}
    A.write_json(A.audit_sample_file(works, memes), TMP / "data" / "human_audit_sample.json")   # the page's only input
    with open(TMP / "data" / "human_audit.jsonl", "w") as f:      # the A7 sample, answered as the design says, as the page appends rows
        for k, s in enumerate(A.audit_items(works, ids)):
            w = byid[s["story"]]
            for j, m in enumerate(s["memes_asked"]):
                truth = m in w["memes"] and w["version"] in A.PLANTED
                if k == 0 and j == 0:                 # a changed answer: the first row is wrong, the last one counts
                    f.write(json.dumps({"kind": "t2_cell", "audit_id": s["audit_id"], "meme": m, "present": not truth, "seconds": 20.0}) + "\n")
                ans = None if (k == 1 and j == 0) else truth  # one 'unsure' answer
                f.write(json.dumps({"kind": "t2_cell", "audit_id": s["audit_id"], "meme": m, "present": ans, "seconds": 30.0 + j}) + "\n")
    zw = [dict(w, sentences=[zh_sentence(RNG) for _ in w["sentences"]]) for w in works if w["version"] in ("banned", "control")]
    keep = [i for i, w in enumerate(works) if w["version"] in ("banned", "control")]
    write_set("t2_stories_zh", zw)
    Fz = np.concatenate([full[i] for i in keep])
    write_neurons("t2_stories_zh", zw, np.array([A.topk_mean(full[i]) for i in keep]), sent={m: Fz[:, ubm[m]] for m in MODELS}, units=ubm["qwen3-8b"])
    Ez = np.concatenate([emb[i] for i in keep])
    for enc in ENCS:
        np.save(feat("t2_stories_zh") / f"emb_{enc}.npy", Ez.astype(np.float16))
    # T2-long: 48 long texts of 12 stories
    lw, lf = [], []
    for k in range(48):
        pick = RNG.choice(len(design), 12, replace=False)
        vs = ["banned", "implicit"] + ["control"] * 10
        idxs = [A.VERSIONS.index(v) + 6 * int(p) for v, p in zip(vs, pick)]
        lw.append({"id": f"L{k}", "sentences": [s for i in idxs for s in works[i]["sentences"]],
                   "memes_present": sorted({m for i in idxs[:2] for m in works[i]["memes"]}), "exclude_memes": []})
        lf.append(np.concatenate([full[i] for i in idxs]))
    write_set("t2_long", lw)
    FLg = np.concatenate(lf)
    write_neurons("t2_long", lw, np.array([A.topk_mean(f) for f in lf]), sent={m: FLg[:, ubm[m]] for m in MODELS}, units=ubm["qwen3-8b"])
    with open(feat("t2_long") / "llm_qwen3-8b_judge.jsonl", "w") as fj, open(feat("t2_long") / "llm_qwen3-8b_sjudge.jsonl", "w") as fs:
        for w in lw:                                  # recursive summaries lose planted memes in long texts
            for m in ids:
                pj = float(np.clip((0.9 if m in w["memes_present"] else 0.1) + RNG.normal(0, 0.05), 0.001, 0.999))
                lost = m in w["memes_present"] and RNG.random() < 0.6      # the summary drops the meme: it scores like an absent one
                ps = float(np.clip(0.1 + RNG.normal(0, 0.05), 0.001, 0.999)) if lost else pj
                fj.write(json.dumps({"work": w["id"], "meme": m, "p_yes": pj}) + "\n")
                fs.write(json.dumps({"work": w["id"], "meme": m, "p_yes": ps}) + "\n")
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
            if w.get("family_tier") == 1 and k < 3:
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
    check(lv0[nm]["p"] < 0.05 and lv0[t1["rival_methods"]["tfidf"]]["p"] < 0.05, "(deferred) T1b level 0: neurons and TF-IDF recover sources")
    h4 = t1["H4"]["qwen3-8b"]
    check(h4["by_rival"]["tfidf"]["n"] == 8 + 13, f"(deferred) H4 pools 8 Batman source clusters and 13 Shakespeare plays ({h4['by_rival']['tfidf']['n']})")
    check(h4["stat"] > 0 and h4["p_iut"] < 0.05 and h4["conjunct_ok"],
          f"(deferred) H4: neurons beat TF-IDF and LSA under synthetic drift (min d {h4['stat']:.3f}, IUT p {h4['p_iut']:.4f}, conjunct {h4['conjunct_ok']})")
    check("retention" in h4 and "degradation_descriptive" in h4 and "style_control_advantage" in h4 and "components_clustering" in h4,
          "(deferred) H4 secondaries: retention, descriptive degradation, style control, components clustering")
    check("t1b_source_leak" in t1 and all("pooled_p" in v for v in t1["t1b_source_leak"].values()), "(deferred) source-leak index computed")
    check(t1["t1c_zh"][nm]["p"] < 0.05 and t1["t1c_zh"][t1["rival_methods"]["tfidf"]]["p"] > 0.05, "(deferred) T1c: neurons cross languages, TF-IDF does not")
    ls_q, ls_l = A.resolve("llmsum:qwen3-8b"), A.resolve("llmsum:llama3.1-8b")
    check(ls_q in lv0 and "stat" in lv0[ls_q], "(deferred) LLM summary-pair lineage method runs (logit similarity)")
    check(lv0.get(ls_l, {}).get("error"), "(deferred) missing LLM ratings are reported, not imputed")
    h5 = t1["H5"]["qwen3-8b"]
    check(h5["stat"] > 0 and h5["p_iut"] < 0.05, f"(deferred) H5: neurons beat TF-IDF and LSA on renamed synthetic adaptations (d {h5['stat']:.3f})")
    check(set(t1["t1d"]) >= {"h5", "renamed_all_evidence", "h5_without_single_film_plays", "h5_unrecognised", "faithful"}, "(deferred) T1d sensitivity subsets run")
    check(t1["t1d_positive_control"][A.resolve("tfidf")]["stat"] < 0.05 and t1["t1d_informative_methods"], "(deferred) T1d positive control and informativeness")
    check(f"{nm}|resid_on_words" in t1["t1d"]["h5"], "(deferred) residual (beyond words) test runs")
    check(len(t1["t1e_fresh"]["targets"]) == 7 and "reuse_audit" in t1, "(deferred) T1e has the 7 fresh adaptations and the reuse audit")
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
    sec, prom, rep, desc = t2["secondary"], t2["secondary_prominent"], t2["replication"], t2["descriptive"]
    check(set(sec) == set(A.t2_secondary_names("qwen3-8b")) and len(sec) == A.T2_FAMILY_SIZE == 7 and len(t2["secondary_bh"]) == 7,
          f"T2 secondary family: exactly 7 Qwen tests (S8, S9, S10, S11, m1-vs-m5, neurons vs hidden, T2-long) ({len(sec)})")
    check(len(prom) == A.PROMINENT_FAMILY_SIZE == 30 and len(t2["secondary_prominent_bh"]) == 30
          and sum(k.startswith("Scoded_") for k in prom) == 6 and sum(k.startswith("O1_") for k in prom) == 4 and sum(k.startswith("O2_") for k in prom) == 20
          and all("qwen3-8b" in k for k in prom) and not any("llama" in k.split("_finds_what_")[0] for k in prom if k.startswith("O")),
          f"prominent secondary family: exactly 30 Qwen tests (S-coded 6, O1 4, O2 20) ({len(prom)})")
    check(any(k.startswith("Scoded_neurons:llama3.1-8b") for k in rep) and any(k.startswith("O1_neurons:llama3.1-8b") for k in rep)
          and any(k.startswith("S8_") or k.startswith("S10_neurons:llama") for k in rep) and not any(k in sec or k in prom for k in rep),
          "Llama tests are replications outside every family")
    pair_keys = [k for k in desc if k.count("|") == 2 and k.endswith("|reworded")]
    check(len(pair_keys) == len({frozenset(k.split("|")[:2]) for k in pair_keys}) and "neurons:qwen3-8b|neurons:llama3.1-8b|reworded" in desc
          and "neurons:llama3.1-8b|neurons:qwen3-8b|reworded" not in desc and any(k.startswith("best_encoder_hindsight") for k in desc)
          and any(k.startswith("S10_emb") for k in desc), "pairwise contrasts (once per pair), per-method S10 and the hindsight encoder are descriptive")
    ha = t2["human_audit"]
    check(ha["pass_reworded"] and ha["pass_coded"] and ha["complete"] and ha["n_stories"] == 60 and ha["n_cells"] == 300 and ha["n_unsure"] == 1
          and ha["all_cells"]["absence_distractors"]["agreement"] == 1.0 and ha["n_rows"] == 301
          and t2["reader_kappa"]["banned"]["claude_claude"] is not None and t2["reader_kappa"]["banned"]["claude_cross_family"] is not None
          and t2["cross_family_check"]["pass_reworded"] and t2["manipulation_gates"]["source"] == "human audit" and t2["manipulation_gates"]["reworded_ok"],
          "human audit through opaque ids (60 stories, 300 cells, a changed and an unsure answer), reader kappas and cross-family check")
    check(bym["neurons:qwen3-8b"]["by_writer_reworded"] and "itt_reworded" in bym["neurons:qwen3-8b"] and "itt_p_iut" in t2["H2"]["qwen3-8b"]
          and "plot_level" in h1["by_rival"]["keywords:logreg"]["itt"] and h1["itt_p_iut"] < 0.05,
          "per-writer results; intention-to-treat for H1 and H2 with the meme-level and plot-level tests")
    check(set(h1["by_rival"]["keywords:logreg"]["by_tier_mean_d"]) == {"A", "B", "C", "N"}, "H1 per-tier differences reported")
    cd = t2["coded"]["tests"]
    check(all(cd[f"{a}|{b}"]["p"] < 0.05 and "plot_level" in cd[f"{a}|{b}"] for a in ("neurons:qwen3-8b", "judge:qwen3-8b") for b in A.CODED_CONTEXT_FREE)
          and "coded_minus_banned" in t2["coded"]["table"]["neurons:qwen3-8b"],
          f"S-coded: context-using methods beat sentence embeddings and the evasion-aware query on coded versions (neurons vs emb d {cd['neurons:qwen3-8b|emb:qwen3emb-4b']['stat']:.2f})")
    check(t2["coded"]["headline"]["all_six_pass"] and t2["coded"]["headline"]["n_tests"] == 6 and not t2["coded"]["memes_below_min_pairs"],
          "S-coded headline: all six Qwen tests pass in the prominent family")
    ov = t2["overlap"]
    k_ = len(ov["methods"])
    fr = [v for v in ov["realised_fpr_range"].values() if v is not None]
    check(len(ov["pairs"]) == k_ * (k_ - 1) // 2 and all("ci95" in v["reworded"] for v in ov["pairs"].values()) and ov["rule"] == "twin"
          and all(0.08 <= lo <= hi <= 0.10 for lo, hi in fr) and len(fr) == k_,
          f"overlap: every pair of {k_} methods with bootstrap intervals; realised control FPR in [0.08, 0.10] for every method "
          f"(keywords:expanded-qwen3-8b {ov['realised_fpr_controls']['keywords:expanded-qwen3-8b']:.3f}, keywords:tfidf {ov['realised_fpr_controls']['keywords:tfidf']:.3f})")
    check(all(ov["own_control_false_alarm"][k]["reworded"]["mean"] is not None for k in ov["methods"])
          and t2["overlap_threshold_rule_sensitivity"]["rule"] == "threshold",
          "overlap: own-plot control false alarms for every method; the threshold-only rule kept as a sensitivity")
    o1 = prom["O1_neurons:qwen3-8b_finds_what_keywords:expanded-qwen3-8b_misses|reworded"]
    check(o1["p"] < 0.05 and o1["count"] >= 20, f"O1: neurons find planted memes the evasion-aware query misses ({o1['count']} memes)")
    check(any(k.startswith("O2_") for k in prom) and "overlap_fpr05_sensitivity" in t2 and "O1_neurons:qwen3-8b_finds_what_keywords:expanded-qwen3-8b_misses|reworded" in t2["secondary_prominent_bh"],
          "O2 tests, the FPR 0.05 sensitivity and BH over the prominent family")
    lp = t2["long"].get("T2long_prediction_qwen3-8b")
    check(lp is not None and lp["p"] < 0.05, "T2-long prediction: summarisation loses more on long texts than on single stories")
    keep_min = A.T2LONG_MIN_TEXTS
    A.T2LONG_MIN_TEXTS = 49
    short = A._t2_long(memes, fz, bym)
    A.T2LONG_MIN_TEXTS = keep_min
    check(short.get("not_run", "").startswith("only 48 long texts"), "T2-long is 'not run' below its minimum number of texts")
    check("zh" not in t2 and "natural" not in t2 and set(t2["deferred_not_run"]) == {"zh", "natural"}, "T2-zh and T2-N are not run (deferred)")
    A.write_json(t2, A.RES / "t2.json")
    audit_page_test()

    # ---------------------------------------------------------------- DEFERRED tests: code kept for the later phase
    zh, nat = A._t2_zh(memes, fz), A._t2_natural(memes, fz)
    check(zh["S12"]["p"] < 0.05 and nat["S13"]["p"] < 0.05, "(deferred) T2-zh and T2-N secondaries still run")
    t3 = A.run_t3(fz, pool_file=TMP / "culture_pool_test.json", n_perm=300)
    bq = t3["by_method"]["neurons:qwen3-8b"]
    check(bq["S14_positive_control_lineage_auc"] > 0.95, "(deferred) T3 positive control (Batman lineage on top)")
    check(bq["relatives_all|batman"]["stat"] > 0.7 and bq["relatives_all|batman"]["p"] < 0.05, "(deferred) T3 relatives rank high (entity-level test)")
    lay = t3["by_method"]["layer|neurons:qwen3-8b"]["S16_family_tier1|layer"]
    check(lay["stat"] > 0.7 and lay["n_entities"] == 3, "(deferred) T3 orphan-vigilante layer finds the tier-1 family (3 entities)")
    check("directional_contrasts" in t3 and "category_strata_sensitivity" in t3 and bq.get("contested_mean_percentile") is not None,
          "(deferred) T3 directional contrasts, category-strata sensitivity, contested works reported")
    check("llm:qwen3-8b" in t3["by_method"], "(deferred) T3 LLM likeness read")
    t4 = A.run_t4(fz, n_perm=100, n_sims=30)
    check(t4["neurons:qwen3-8b"]["all"]["innovators"]["p"] < 0.05, f"(deferred) T4 innovators detected (p {t4['neurons:qwen3-8b']['all']['innovators']['p']:.3f})")
    check("S19_neurons:qwen3-8b_vs_words" in t4 and "batch_effect_neurons:qwen3-8b" in t4, "(deferred) T4 comparative selection test and batch-effect check")
    t5 = A.run_t5(fz)
    check(t5["random_films_primary"]["test"]["p"] < 0.05, "(deferred) T5 concept twins beat matched nulls on held-out random films")
    refused = []
    for cmd in A.DEFERRED_COMMANDS:
        try:
            A.main([cmd])
        except SystemExit as e:
            refused.append("deferred" in str(e))
    check(refused == [True] * len(A.DEFERRED_COMMANDS), "the CLI refuses the deferred tests t1, t3, t4, t5 without --deferred")
    # ---------------------------------------------------------------- T6 on the T2 stories, cost, primary
    items = A.t6_prepare(n_units=40, dev_units_file=TMP / "dev_units.json")
    arms = {it["arm"] for it in items["items"]}
    kinds = {it["kind"] for it in items["items"]}
    check(len(items["items"]) > 0 and arms == {"random", "best_of"} and any(k.startswith("lda|") for k in items["units"])
          and "neurons:qwen3-8b" in kinds, f"T6 items built on the T2 stories in two arms ({len(items['items'])} items, kinds {sorted(kinds)})")
    plot_of = {s: w["plot"] for w in A.read_jsonl(TMP / "data" / "t2_stories.jsonl") for s in w["sentences"]}
    lab_sents = [s for us in items["units"].values() for u in us.values() for s in u["label_sentences"]]
    check(all(s in plot_of for s in lab_sents), "T6 label sentences come from the T2 stories")
    A.write_json({k: {u: {"label": "a label", "description": "a description"} for u in us} for k, us in items["units"].items()}, A.RES / "label_test_labels.json")
    lab = [{"kind": it["kind"], "arm": it["arm"], "unit": it["unit"], "item": it["item"], "judge": j,
            "correct": bool(RNG.random() < (0.8 if it["kind"].startswith("neurons") else 0.6))} for it in items["items"] for j in range(2)]
    A.write_json(lab, A.RES / "label_test_answers.json")
    t6 = A.run_t6(min_units=5)
    k0 = next(k for k in t6["kinds"] if k.startswith("neurons:qwen3-8b"))
    n_kinds = {a: sum(k.endswith("|" + a) for k in t6["kinds"]) for a in ("random", "best_of")}
    n_fisher = sum(1 for k in t6["secondary_bh"] if not k.startswith("S21|"))
    n_neur = {a: sum(k.startswith("neurons") and k.endswith("|" + a) for k in t6["kinds"]) for a in n_kinds}
    want = sum(n_neur[a] * (n_kinds[a] - n_neur[a]) + n_neur[a] * (n_neur[a] - 1) // 2 for a in n_kinds)
    check(t6["kinds"][k0]["vs_floor_mcnemar"] is not None and t6["kinds"][k0]["judge_kappa"] is not None
          and any(k.startswith("S21|") for k in t6["secondary_bh"]) and n_fisher == want,
          f"T6 label test: McNemar vs floor (S21), judge kappa, one Fisher test per unordered pair ({n_fisher} Fisher tests, {len(t6['secondary_bh'])} in the T6 family)")
    t6d = A.run_t6()
    nu = {k: len({a["unit"] for a in lab if f"{a['kind']}|{a['arm']}" == k}) for k in {f"{a['kind']}|{a['arm']}" for a in lab}}
    check(all((k in t6d["not_run"]) == (v < A.T6_MIN_UNITS) for k, v in nu.items()) and all(k not in t6d["kinds"] for k in t6d["not_run"]),
          f"T6: a kind and arm with fewer than 30 qualifying units is 'not run' and enters no test ({sorted(t6d['not_run'])})")
    cost = A.run_cost()
    check("frontier" in cost and cost["H6"]["qwen3-8b"]["dominated_by"] is not None and cost["curves"]["neurons"], "cost: recipes, curves, frontier and H6")
    A.write_json(cost, A.RES / "cost.json")
    h6_test(ids)
    prim = A.run_primary(t2, cost=cost)
    q = prim["models"]["qwen3-8b"]
    check(set(q["holm"]) == {"H1", "H2", "H3"} and abs(min(v["threshold"] for v in q["holm"].values()) - 0.05 / 3) < 1e-12
          and "H2_superiority_supported" in q and q["H6"]["verdict"] and prim["deferred_not_evaluated"] == ["H4", "H5"],
          f"primary: Holm over H1-H3 only, H6 decided, H4 and H5 not evaluated ({q['verdict']})")
    check(q["verdict"]["H1"].startswith("supported") and q["verdict"]["H3"].startswith("supported"), "H1 and H3 supported in the planted world")
    A.write_json(prim, A.RES / "primary.json")
    old = A.run_primary(t2, cost=cost, t1=t1, claims=("H1", "H2", "H3", "H4", "H5"))["models"]["qwen3-8b"]
    check(set(old["holm"]) == {"H1", "H2", "H3", "H4", "H5"} and old["verdict"]["H4"] == "inconclusive" and old["p_entered"]["H4"] == 1.0,
          "(deferred) the prereg-v2 five-claim path still runs: H4 inconclusive when the drift did not remove the banned words")
    man = A.feature_manifest()
    check(len(man) > 50, f"F3 feature manifest ({len(man)} files)")
    clean = A.verify_manifest()
    victim = A.FEAT / "t2_stories" / "llm_qwen3-8b_judge.jsonl"
    keep = victim.read_bytes()
    victim.write_bytes(keep + b"\n")
    bad = A.verify_manifest()
    try:
        A.main(["t2"])
        refused = False
    except SystemExit as e:
        refused = "manifest" in str(e)
    victim.write_bytes(keep)
    check(clean == [] and bad == ["changed t2_stories/llm_qwen3-8b_judge.jsonl"] and refused and A.verify_manifest() == [],
          "F3: every feature hash is re-checked before t2 or t6, and the runner refuses on a mismatch")
    print(f"  pipeline finished in {time.time() - t0:.0f}s")


def audit_page_test():
    """The audit page (audit_page.py) on the synthetic sample: no version name, plot id or story id in the file it reads
    or the HTML it serves; answers are appended in the documented row format; bad answers are refused."""
    import threading
    import urllib.request
    import urllib.error
    sys.path.insert(0, str(HERE))
    import audit_page as P
    works = A.read_jsonl(TMP / "data" / "t2_stories.jsonl")
    sample_path = TMP / "data" / "human_audit_sample.json"
    sample = json.load(open(sample_path))
    out = TMP / "page_answers.jsonl"
    srv = P.make_server(sample_path, out, port=0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        pages = [urllib.request.urlopen(f"{base}/?i={i}").read().decode() for i in (0, 1, 59)]
        it = sample["items"][0]
        post = lambda body: urllib.request.urlopen(urllib.request.Request(f"{base}/answer", data=json.dumps(body).encode(),
                                                                          headers={"Content-Type": "application/json"}))
        for k, m in enumerate(it["memes"]):
            post({"audit_id": it["audit_id"], "meme": m["id"], "answer": ("yes", "no", "unsure", "yes", "no")[k], "seconds": 10 + k})
        post({"audit_id": it["audit_id"], "meme": it["memes"][0]["id"], "answer": "no", "seconds": 40})
        try:
            post({"audit_id": "A99", "meme": "x", "answer": "yes"})
            refused = False
        except urllib.error.HTTPError as e:
            refused = e.code == 400
        after = urllib.request.urlopen(f"{base}/?i=0").read().decode()
    finally:
        srv.shutdown()
    rows = A.read_jsonl(out)
    leaks = A.audit_leaks(sample, works, strict=True) + [x for h in pages for x in A.audit_leaks(h, works, strict=True)]
    check(not leaks and all(it_["sentences"][0] and "<li>" in pages[0] for it_ in sample["items"][:1]) and set(sample["items"][0]) == {"audit_id", "block", "sentences", "memes"},
          f"audit page: no version name, plot id or story id in the sample file or the served HTML ({leaks[:3]})")
    check(refused and len(rows) == 6 and all(set(r) == {"kind", "audit_id", "meme", "present", "seconds", "time"} and r["kind"] == "t2_cell" for r in rows)
          and [r["present"] for r in rows] == [True, False, None, True, False, False] and P.latest(rows)[(it["audit_id"], it["memes"][0]["id"])] is False
          and "1 of 60 stories done" in after and P.pilot_report(out)["median_seconds"] == 40.0,
          "audit page: answers appended as {kind, audit_id, meme, present, seconds, time}, the last one counts, bad answers refused, progress shown")


def h6_test(ids):
    """H6: the neurons' own variants (the other model's neurons, m5, DEV units) and the Claude upper bound never
    dominate; each rival is compared on the memes it shares with the neurons."""
    rng = np.random.default_rng(11)
    per = lambda mu, sd=0.02, memes=ids: {m: float(mu + rng.normal(0, sd)) for m in memes}
    half = ids[:20]
    bym = {"neurons:qwen3-8b": {"reworded": {**per(0.95, memes=half), **per(0.5, memes=ids[20:])}},
           "neurons:llama3.1-8b": {"reworded": per(0.99)}, "neurons:qwen3-8b:m5": {"reworded": per(0.99)}, "devunit:qwen3-8b": {"reworded": per(0.99)},
           "judge:claude-upper-bound": {"reworded": per(0.99)}, "emb:qwen3emb-4b": {"reworded": per(0.6)},
           "emb:e5-large": {"reworded": {**per(0.9, memes=half), **{m: float("nan") for m in ids[20:]}}}}
    c1 = A.run_cost(t2={"n_stories": 816, "by_method": bym})
    old = [b["method"] for b in c1["frontier"] if b["method"] != "neurons:qwen3-8b" and A.dominates(b, next(f for f in c1["frontier"] if f["method"] == "neurons:qwen3-8b"))]
    bym2 = dict(bym, **{"emb:e5-large": {"reworded": per(0.99)}})
    c2 = A.run_cost(t2={"n_stories": 816, "by_method": bym2})
    check(c1["H6"]["qwen3-8b"]["dominated_by"] == [] and "neurons:llama3.1-8b" in old and "emb:e5-large" in old
          and c1["H6"]["qwen3-8b"]["comparisons"]["emb:e5-large"]["n_common_memes"] == 20 and c2["H6"]["qwen3-8b"]["dominated_by"] == ["emb:e5-large"],
          f"H6: Llama's neurons, the m5 variant, DEV units and the Claude upper bound never dominate; rivals compared on shared memes (old rule: {sorted(old)})")


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
