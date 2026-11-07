"""Shared loaders and the v2 presence rule (PLAN.md, Sections 4 and 5). Run everything from study/ (`cd study`).

Per model, v2/acts/<tag>/ (downloaded from the GPU machine by v2/gpu/pull.py, not committed) holds

  scale.npy, scale_ref.npy, scale_ref5.npy, scale_raw.npy   [2U] float32   each unit's scale: reference corpus, supplement
                                 and study texts; without the study texts; fifth per-batch reference maximum; raw product
  scale_source.npy               [2U] uint8   what set the scale: 0 reference corpus, 1 supplement, 2 study texts
  <text>_main_<score>.u8.npy     [114, 2U]   each work's score as a share of the unit's scale, in steps of 0.005, rounded
                                             down. text: orig, anon, anon_icon; score: top1 (strongest sentence), top3
  <text>_main_top1.ref.u8.npy, .ref5.u8.npy   the same against the two alternative scales
  {orig,anon}_cut_main_top1      the same on each work's opening (at most 20 sentences and 400 words)
  anon_codes_main_top1           name-free text with the placeholder codes allowed to carry the peak
  anon_after_main_top1           name-free text with the codes and the token after each code masked
  {orig,anon}_raw_top1           the raw product (all sign regimes, codes left in) as a share of scale_raw
  bg_counts*.npy                 [4, 2U] int32   background plots in which the unit reaches 20/30/40/50% of its scale
                                 (main scale; reference-only; fifth-example; raw product on its own scale)
  bg_top_shares.u8.npy           [160, 2U]   the unit's 160 highest per-plot shares of the main scale
  bg_present30.bits.npy          [plots, 2U / 8]   which units are present in which background plot (packed bits)
  bg_meta.json, study_meta.json, meta.json, scale_summary.json, pulled.json

Units: positive poles first (layer * width + neuron), then negative poles.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

STEP = 200                              # uint8 units per 1.0 of scale
THRESHOLDS = [0.2, 0.3, 0.4, 0.5]       # shares of scale at which background counts were taken
THETA, CAP = 0.30, 0.03                 # primary: fires strongly at 30% of scale; distinctive: at most 3% of ordinary plots
CAPS = [None, 0.30, 0.20, 0.10, 0.05, 0.03, 0.01]
ACTS = Path(os.environ.get("PHYLO_V2_ACTS", "v2/acts"))
RESULTS = Path("results/v2")
REFERENCE = Path("v2/cache/reference.jsonl")
MODELS = ["qwen3-32b", "qwen3-8b", "llama3.1-8b", "llama3.3-70b"]
SEED = 20261107
OVERLAP = [843, 7175, 6895, 3597, 5284]      # reference documents that are study works under older titles (PLAN.md, Section 5)
SCALE_FILES = {"study": "scale.npy", "ref": "scale_ref.npy", "ref5": "scale_ref5.npy", "raw": "scale_raw.npy"}
COUNT_FILES = {"study": "bg_counts.npy", "ref": "bg_counts_ref_scale.npy", "ref5": "bg_counts_ref5_scale.npy", "raw": "bg_counts_raw.npy"}

sys.path.insert(0, ".")


def geometry(tag):
    m = json.load(open(ACTS / tag / "meta.json"))
    return m["n_layers"], m["width"]


def unit_name(tag, u):
    """L<layer>·N<neuron><pole>, e.g. L10·N9068+."""
    L, D = geometry(tag)
    U = L * D
    return f"L{(u % U) // D}·N{u % D}{'+' if u < U else '-'}"


def n_plots(tag):
    m = json.load(open(ACTS / tag / "bg_meta.json"))
    assert m["excluded_docs"] == [843], "the background pass must leave out document 843 (PLAN.md Section 5)"
    return m["n_plots"]


def scale(tag, which="study"):
    return np.load(ACTS / tag / SCALE_FILES[which])


def dominance(tag):
    """Each pole's scale as a share of the larger of its neuron's two scales (1 for the stronger pole)."""
    s = scale(tag)
    U = len(s) // 2
    top = np.maximum(s[:U], s[U:])
    top = np.concatenate([top, top])
    return s / np.where(top > 0, top, 1)


def scale_source(tag):
    """Where each unit's scale comes from: 0 the reference corpus, 1 the supplement, 2 the study's texts."""
    return np.load(ACTS / tag / "scale_source.npy")


def base_count(tag, theta=THETA, which="study"):
    """Background plots in which each unit is present at `theta` of the chosen scale (strongest sentence)."""
    return np.load(ACTS / tag / COUNT_FILES[which])[THRESHOLDS.index(theta)].astype(np.int64)


def base_rate(tag, theta=THETA, which="study"):
    return base_count(tag, theta, which) / n_plots(tag)


def score_file(text, score="top1", which="study"):
    if which == "raw":
        return f"{text}_raw_top1.u8.npy"
    return f"{text}_main_{score}{'' if which == 'study' else '.' + which}.u8.npy"


class Setting:
    """One version of the presence rule for one model, held in memory for the units that are present anywhere.

    theta    share of the scale that counts as firing strongly (0.2, 0.3, 0.4 or 0.5)
    score    "top1" the strongest sentence, "top3" the mean of the three strongest
    which    the scale: "study" (reference corpus, supplement and study texts), "ref" (no study texts), "ref5" (the fifth
             strongest of the per-batch reference maxima)
    variant  "full" the whole text with the placeholder codes masked (the main analysis); "cut" the openings;
             "codes" the name-free text with the codes left in; "after" with the token after each code masked too;
             "raw" the raw product with the echo regime left in (codes left in, its own scale)
    min_dominance    leave out poles whose scale is under this share of the neuron's stronger pole
    drop_supplement  leave out units whose scale comes from the supplement

    Gives, for the units present in at least one of the 114 works in any loaded text (no other unit can enter any
    analysis): units (their indices), base (their base rate in the background plots), and for each text in
    self.texts a boolean presence matrix P[text] [114, n] and the stored shares S[text] [114, n] (uint8).
    """

    def __init__(self, tag, theta=THETA, score="top1", which="study", variant="full", icon=True, min_dominance=None, drop_supplement=False):
        if variant == "raw":
            which = "raw"
        assert score == "top1" or (variant == "full" and which == "study"), "the mean-of-three score exists for the full texts on the main scale only"
        assert which in ("study", "raw") or variant == "full", "the alternative scales exist for the full texts only"
        self.tag, self.theta, self.score, self.which, self.variant = tag, theta, score, which, variant
        self.min_dominance, self.drop_supplement = min_dominance, drop_supplement
        names = {"orig": "orig_cut" if variant == "cut" else "orig",
                 "anon": {"full": "anon", "cut": "anon_cut", "codes": "anon_codes", "after": "anon_after", "raw": "anon"}[variant]}
        if icon and variant == "full":
            names["anon_icon"] = "anon_icon"
        cut = int(np.ceil(theta * STEP - 1e-6))
        ok = scale(tag, which) > 0
        self.n_units_total, self.n_can_fire = int(ok.size), int(ok.sum())
        if min_dominance is not None:
            ok &= dominance(tag) >= min_dominance
        if drop_supplement:
            ok &= scale_source(tag) != 1
        full, anyp = {}, np.zeros(ok.shape, bool)
        for text, name in names.items():
            full[text] = np.load(ACTS / tag / score_file(name, score, which))
            anyp |= (full[text] >= cut).any(0)
        self.units = np.flatnonzero(anyp & ok)
        self.S = {t: np.ascontiguousarray(a[:, self.units]) for t, a in full.items()}
        del full
        self.P = {t: a >= cut for t, a in self.S.items()}
        self.texts = list(names)
        self.n_eligible = int(ok.sum())
        self.base_count = base_count(tag, theta, which)[self.units]
        self.n_plots = n_plots(tag)
        self.base = self.base_count / self.n_plots

    def distinctive(self, cap=CAP):
        """Columns (into self.units) whose base rate is at most `cap`; cap None keeps every unit."""
        return np.ones(len(self.units), bool) if cap is None else self.base <= cap + 1e-12

    def describe(self):
        d = {"theta": self.theta, "score": self.score, "scale": self.which, "variant": self.variant}
        if self.min_dominance is not None:
            d["poles_at_least_this_share_of_the_stronger_pole"] = self.min_dominance
        if self.drop_supplement:
            d["units_scaled_by_the_supplement"] = "left out"
        return d


def background_docs():
    """The background plots (reference documents with src bgplot), in the order the background pass read them."""
    return [d for d in map(json.loads, open(REFERENCE)) if d["src"] == "bgplot" and d["i"] != OVERLAP[0]]


def save(obj, name):
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote", path)
