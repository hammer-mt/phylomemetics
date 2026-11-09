"""Figure for the second corpora (v2/second/PLAN.md): Sherlock Holmes, experiment E1 ("identity") and recognition.

fig_holmes.png, two panels of horizontal dot plots. Every mark is a within-era AUC: the chance that the name-free
summary of a Holmes screen work scores above that of a detective film of its own era (0.5 is chance).
  A  Identity without names. One row per model: the share of the model's Holmes canon units present in a summary.
     Then three scores that use no model: the TF-IDF centroid, the number of placeholder codes alone and the number
     of sentences alone. The large mark is the whole summary with its 95% interval; the small mark under it uses the
     opening only. Both AUCs are printed at the right. Under Llama 3.3 70B, a small open triangle gives the follow-up
     reading (the token after each placeholder code masked too). Lines at chance and at the pre-registered bar.
  B  Is it recognition? For each model, the same AUC for the screen works the model names as Holmes from the
     name-free summary and for those it does not, with intervals and the number of works in each group. Lines at
     chance and at the level the plan's rule (Section 10) asks of the works that are not named.

Reads results/v2/second/ (or $RESULTS_DIR): holmes_<tag>.json for whichever of the four models are there,
holmes_words.json, and holmes_e1_followup_<tag>.json where it exists. A model without a file, or whose file cannot
be read yet, is skipped. Every number drawn is read from those files at run time and printed with the key it came
from. The title is a sentence about the results only when the results support it (see `title`); otherwise a title
that makes no claim is used. Writes fig_holmes.png to paper/v2/ (or $PAPER_DIR).

    cd study && ../.venv/bin/python v2/figs/holmes.py
"""
import json
import os
import re
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
os.environ.setdefault("PAPER_DIR", str(STUDY / "paper" / "v2"))
sys.path.insert(0, str(STUDY / "paper" / "figs"))
import style  # noqa: E402

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TITLE = "Without names, only the largest model clears the bar, mostly on works it names"
TITLE_NO_CLAIM = "Without names: telling Sherlock Holmes screen works from detective films"   # when the results do not say that
# {works} and {films} are filled in from the results
SUBTITLE = ("How well a score tells the {works} name-free Holmes screen works from the {films} detective films of "
            "the same era (AUC).\nSmall marks in A use only the opening of each summary. "
            "B splits the screen works by whether the model names Holmes.")
X_LABEL = "AUC: 0.5 is chance, 1 is perfect"
PANEL_A = "Identity without names: all {works} screen works"
PANEL_B = "Is it recognition? Named works and others"
GROUP_UNITS = "With a model's units"
GROUP_WORDS = "Without a model"
FOLLOW_UP = "token after each\ncode masked too"
NOT_EVALUABLE = "too few units\nto judge"  # under a model whose registered verdict is "not evaluable" (its canon is too small)
# a caption for the paper, printed with the numbers of this run
CAPTION = ("**Figure N.** How well a score tells the name-free summaries of the {works} Sherlock Holmes screen works from "
           "the {films} detective films, each screen work compared only with films of its own era (AUC: 0.5 is chance, "
           "1 is perfect). A: the share of each model's Holmes canon units that is present in a summary ({aucs}), then "
           "three scores that use no model: word counts, the number of placeholder codes and the number of sentences. "
           "Large marks are whole summaries with 95% bootstrap intervals and small marks use only the opening of each "
           "summary{follow}. The pre-registered bar is an AUC of {bar}{not_judged}. B: the same score for the screen "
           "works each model names as Sherlock Holmes when asked whose story the name-free summary is, and for those it "
           "does not name, with the number of works in each group{level}.")
CAPTION_FOLLOW = "; the triangle is {name} read again with the token after each placeholder code masked too ({auc})"
CAPTION_NOT_JUDGED = "; {name} has too few canon units for the plan to judge it (a median of {n})"
CAPTION_LEVEL = "; {level} is the level the plan asks of the works that are not named ({name}: {rest})"

RESULTS = Path(os.environ.get("RESULTS_DIR") or STUDY / "results" / "v2" / "second")
GREY = style.SHARED
MODELS = [  # tag in the results, display name, marker face, marker edge, size in billions of parameters
    ("qwen3-32b", "Qwen3-32B", style.NIGHT, style.NIGHT, 32),   # the primary model is the solid dark one
    ("qwen3-8b", "Qwen3-8B", "white", style.NIGHT, 8),
    ("llama3.1-8b", "Llama 3.1 8B", GREY, GREY, 8),
    ("llama3.3-70b", "Llama 3.3 70B", "white", GREY, 70),
]
WORD_SCORES = [("tfidf_centroid", "Word counts\n(TF-IDF centroid)"),
               ("distinct_codes_alone", "Number of\nplaceholder codes"),
               ("length_alone_sentences", "Number of\nsentences")]
CHANCE = 0.5  # an AUC of one half is chance by definition; it is not a result
FS = 7        # labels; nothing is below 6.5 pt


def load(name):
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:  # a file that is still being written is treated as not there yet
        print(f"  {name}: cannot be read yet, skipped")
        return None


def e1(block, version, where):
    """One E1 score in one version (whole summaries or openings): the within-era AUC and its 95% interval."""
    w = block[version]["within_era"]
    print(f"  {where}.{version}.within_era: auc {w['auc']:.4f}, ci95 {w['ci95'][0]:.4f} to {w['ci95'][1]:.4f}")
    return {"auc": w["auc"], "ci95": w["ci95"]}


# ---- what is drawn ----
runs = [(m, load(f"holmes_{m[0]}.json")) for m in MODELS]
runs = [(m, d) for m, d in runs if d and "E1" in d]
print("models in the folder:", ", ".join(m[0] for m, _ in runs) or "none")
if not runs:
    print("no holmes_<tag>.json in", RESULTS, ": nothing to draw yet")
    sys.exit(0)
words = (load("holmes_words.json") or {}).get("E1", {})

rows_a = []  # (group, label, marker, face, edge, whole, opening, follow-up or None, note under the label or None)
for (tag, name, face, edge, _), d in runs:
    src = f"holmes_{tag}.json: E1"
    follow = None
    f = load(f"holmes_e1_followup_{tag}.json")
    if f:
        g = f["E1_name_free"]["token_after_each_code_masked_too"]
        follow = {"auc": g["within_era_auc"], "ci95": g["within_era_ci95"]}
        print(f"  holmes_e1_followup_{tag}.json: E1_name_free.token_after_each_code_masked_too: within_era_auc "
              f"{follow['auc']:.4f}, within_era_ci95 {follow['ci95'][0]:.4f} to {follow['ci95'][1]:.4f}")
    verdict = d["E1"]["verdict_within_era"]["verdict"]
    print(f"  holmes_{tag}.json: E1.verdict_within_era.verdict = {verdict!r}, E1.A_name_free.canon_units_median = "
          f"{d['E1']['A_name_free'].get('canon_units_median')}")
    rows_a.append(("units", name, "o", face, edge, e1(d["E1"], "A_name_free", src), e1(d["E1"], "openings", src),
                   follow, NOT_EVALUABLE if verdict == "not evaluable" else None))
for key, label in WORD_SCORES:  # comparators, not models: a neutral grey diamond
    if key in words:
        src = f"holmes_words.json: E1.{key}"
        rows_a.append(("words", label, "D", style.INK_3, style.INK_3, e1(words[key], "A_name_free", src),
                       e1(words[key], "openings", src), None, None))

rows_b = []  # (label, face, edge, {named: ..., not_named: ...})
for (tag, name, face, edge, _), d in runs:
    rec = (d.get("recognition") or {}).get("E1")
    if not rec:
        continue
    for side in ("named", "not_named"):
        s = rec[side]
        auc, ci = s.get("auc_within_era"), s.get("ci95") or [float("nan")] * 2
        print(f"  holmes_{tag}.json: recognition.E1.{side}: n {s['n']}, auc_within_era "
              f"{float('nan') if auc is None else auc:.4f}, ci95 {ci[0]:.4f} to {ci[1]:.4f}")
    rows_b.append((name, face, edge, rec))

verdicts = [d["E1"]["verdict_within_era"] for _, d in runs]
# the pre-registered bar is spelled in the name of its clause in the results ("auc_at_least_0.80")
BAR = float(next(k for v in verdicts for k in v["clauses"] if k.startswith("auc_at_least_")).rsplit("_", 1)[1])
# the level asked of the works a model does not name is spelled in the rule the results quote (PLAN Section 10)
rule = next((d["recognition"]["not_confined_to_works_the_model_names"]["E1_rule"] for _, d in runs
             if "recognition" in d), "")
found = re.search(r"AUC at least (\d\.\d+)", rule)
LEVEL = float(found.group(1)) if found else None
sizes = {(c["n_franchise"], c["n_negatives"]) for _, d in runs for c in [d["E1"]["A_name_free"]["negatives"]]}
assert len(sizes) == 1, sizes
N_WORKS, N_FILMS = sizes.pop()
print(f"  bar {BAR} (verdict_within_era.clauses), level for the works not named {LEVEL} (recognition...E1_rule), "
      f"{N_WORKS} screen works, {N_FILMS} detective films (E1.A_name_free.negatives)")

# the title is a sentence about the data: say so loudly if a re-run no longer supports it
passing = [m[0] for m, d in runs if d["E1"]["A_name_free"]["within_era"]["auc"] >= BAR]
largest = max(runs, key=lambda r: r[0][4])
split = (largest[1].get("recognition") or {}).get("E1") or {}
named, rest = (split.get(side, {}).get("auc_within_era") for side in ("named", "not_named"))
supported = (len(runs) == len(MODELS) and passing == [largest[0][0]] and None not in (named, rest, LEVEL)
             and rest < LEVEL <= named)
title = TITLE if supported else TITLE_NO_CLAIM
if not supported:
    print("NOTE: these results do not say what the usual title says (models at or above the bar:",
          ", ".join(passing) or "none", f"; {len(runs)} of {len(MODELS)} models here); a title with no claim is used")
print("title:", title)


def shown(x):
    """Two decimals as in the figure; a value that would print as the bar or the level although it is under it gets one more digit."""
    return f"{x:.3f}" if any(v is not None and x < v and f"{x:.2f}" == f"{v:.2f}" for v in (BAR, LEVEL)) else f"{x:.2f}"


follows = [(row[1], row[7]) for row in rows_a if row[7]]
small = [(m[1], d["E1"]["A_name_free"].get("canon_units_median")) for m, d in runs
         if d["E1"]["verdict_within_era"]["verdict"] == "not evaluable"]
print("caption:", CAPTION.format(
    works=N_WORKS, films=N_FILMS, aucs=", ".join(f"{row[1]} {shown(row[5]['auc'])}" for row in rows_a if row[0] == "units"),
    follow="".join(CAPTION_FOLLOW.format(name=name, auc=shown(f["auc"])) for name, f in follows), bar=f"{BAR:.2f}",
    not_judged="".join(CAPTION_NOT_JUDGED.format(name=name, n=f"{n:g}") for name, n in small),
    level=CAPTION_LEVEL.format(level=f"{LEVEL:.2f}", name=largest[0][1], rest=shown(rest))
    if LEVEL is not None and rest is not None else ""))

# ---- layout, in inches; y runs downwards from the top of the two plots ----
W = style.WIDTH_IN - 0.16                       # the tight crop adds 0.08 in on each side
A_LABEL, A_PLOT, A_RIGHT = 1.08, 1.62, 0.86     # panel A: row labels; plot; the two columns of values
B_LABEL, B_PLOT, B_RIGHT = 0.84, 1.60, 0.38     # panel B: the groups and their sizes; plot; the column of values
SPACE = W - (A_LABEL + A_PLOT + A_RIGHT + B_LABEL + B_PLOT + B_RIGHT)   # between the panels
BOTTOM, HEAD = 0.70, 0.48                       # tick labels, axis label and legend; panel headings and the lines' labels
PITCH, DROP, PAD = 0.38, 0.13, 0.17             # row pitch; drop of the small mark; ends
SUB, GAP = 0.29, 0.58                           # drop of the follow-up's row; from the last unit row to the words
X0, X1 = 0.25, 1.0

y, ys = PAD, []
for i, row in enumerate(rows_a):
    if i and row[0] != rows_a[i - 1][0]:
        y += GAP - PITCH
    ys.append(y)
    y += PITCH + (SUB if row[7] else 0)
ax_h = ys[-1] + DROP + PAD + (SUB if rows_a[-1][7] else 0)
subtitle = SUBTITLE.format(works=N_WORKS, films=N_FILMS)
H = (ax_h + BOTTOM + HEAD + 0.14 * (subtitle.count("\n") + 1) + 0.20) / 0.94  # style.title sets the subtitle 6% down
fig = plt.figure(figsize=(W, H))
style.title(fig, title, subtitle)
print("subtitle:", subtitle.replace("\n", " / "))
a_left = A_LABEL
b_left = A_LABEL + A_PLOT + A_RIGHT + SPACE + B_LABEL
axa = fig.add_axes([a_left / W, BOTTOM / H, A_PLOT / W, ax_h / H])
axb = fig.add_axes([b_left / W, BOTTOM / H, B_PLOT / W, ax_h / H])
head = dict(va="baseline", fontsize=FS, color=style.INK_2, linespacing=1.15)  # one baseline for the header row
Y_HEAD = -0.07


def frame(ax, lines):
    """The shared look of a panel: no spines, light grid, and the labelled reference lines."""
    ax.set_xlim(X0, X1)
    ax.set_ylim(ax_h, 0)
    ax.spines[:].set_visible(False)
    ax.set_yticks([])
    ticks = [0.25, 0.5, 0.75, 1]
    ax.set_xticks(ticks, [f"{t:.2f}" for t in ticks])
    ax.tick_params(axis="x", length=0, pad=4, labelsize=FS)
    ax.set_xlabel(X_LABEL, labelpad=3, fontsize=FS)
    for t in ticks:
        ax.axvline(t, color=style.RULE, lw=0.8, zorder=0.5, clip_on=False)
    for x, text, dashes, ha in lines:
        ax.axvline(x, color=style.INK_3, lw=0.8, ls=(0, dashes), zorder=1)
        ax.text(x, Y_HEAD, text, ha=ha, **head)


def inch(ax_plot_w, inches):
    """Inches from a plot's left edge, as an x on its axis (for the text beside the plot)."""
    return X0 + inches / ax_plot_w * (X1 - X0)


def panel_heading(ax, plot_w, label_w, letter, text):
    ax.text(inch(plot_w, -label_w + 0.01 * W), -HEAD + 0.08, letter, fontsize=9, fontweight="bold", va="center")
    ax.text(inch(plot_w, -label_w + 0.01 * W + 0.18), -HEAD + 0.08, text, fontsize=7.6, fontweight="bold", va="center")


# ---- A: identity without names ----
frame(axa, ((CHANCE, "chance", (4, 2), "center"), (BAR, "pre-registered\nbar", (1, 2), "center")))
panel_heading(axa, A_PLOT, A_LABEL, "A", PANEL_A.format(works=N_WORKS))
ax_ = lambda inches: inch(A_PLOT, inches)  # noqa: E731
X_WHOLE, X_OPENING = ax_(A_PLOT + 0.36), ax_(A_PLOT + A_RIGHT)  # right edges of the two columns of values
axa.text(X_WHOLE, Y_HEAD, "whole", ha="right", **head)
axa.text(X_OPENING, Y_HEAD, "opening", ha="right", **head)
group_label = dict(ha="left", va="center", fontsize=7.5, color=style.INK)
for i, (group, label, marker, face, edge, whole, opening, follow, remark) in enumerate(rows_a):
    y = ys[i]
    if i == 0:
        axa.text(ax_(-A_LABEL + 0.01 * W), Y_HEAD, GROUP_UNITS, **{**group_label, "va": "baseline"})
    elif group != rows_a[i - 1][0]:
        axa.text(ax_(-A_LABEL + 0.01 * W), y - 0.26, GROUP_WORDS, **group_label)
    size = 6.4 if marker == "D" else 7.5
    axa.text(ax_(-0.13), y + (DROP / 2 if "\n" in label else 0), label, ha="right", va="center", fontsize=FS,
             color=style.INK_2, linespacing=1.15)
    if remark:  # the plan does not judge this model: say so under its name
        axa.text(ax_(-0.13), y + 0.075, remark, ha="right", va="top", fontsize=6.5, color=style.INK_3, linespacing=1.1)
    axa.plot(whole["ci95"], [y, y], color=style.INK_2, lw=0.9, zorder=3, solid_capstyle="butt")
    for cell, yy, ms, mew, x_text, fs, ink in ((whole, y, size, 1.3, X_WHOLE, 7.5, style.INK),
                                               (opening, y + DROP, 0.6 * size, 1.0, X_OPENING, FS, style.INK_3)):
        axa.plot(cell["auc"], yy, marker=marker, ms=ms, mfc=face, mec=edge, mew=mew, zorder=4, clip_on=False)
        # two decimals, as in the paper's tables
        axa.text(x_text, y, f"{cell['auc']:.2f}", ha="right", va="center", fontsize=fs, color=ink)
    if follow:  # the follow-up reading of the same model, on its own short row
        yf = y + SUB
        axa.plot(follow["ci95"], [yf, yf], color=style.INK_3, lw=0.7, zorder=3, solid_capstyle="butt")
        axa.plot(follow["auc"], yf, marker="^", ms=5.2, mfc="white", mec=edge, mew=1.0, zorder=4)
        axa.text(ax_(-0.13), yf, FOLLOW_UP, ha="right", va="center", fontsize=6.5, color=style.INK_3,
                 linespacing=1.1)
        axa.text(X_WHOLE, yf, f"{follow['auc']:.2f}", ha="right", va="center", fontsize=FS, color=style.INK_3)

# ---- B: is it recognition? ----
frame(axb, ((CHANCE, "chance", (4, 2), "center"),)
      + (((LEVEL, f"{LEVEL:.2f}, asked of the\nworks not named", (1, 2), "left"),) if LEVEL else ()))
panel_heading(axb, B_PLOT, B_LABEL, "B", PANEL_B)
bx_ = lambda inches: inch(B_PLOT, inches)  # noqa: E731
block = ax_h / max(len(rows_b), 1)  # one block per model: its name, then the works it names and the rest
for k, (name, face, edge, rec) in enumerate(rows_b):
    top = k * block + (block - 0.80) / 2
    axb.text(bx_(-B_LABEL + 0.01 * W), top + 0.15, name, **group_label)
    for side, word, yy in (("named", "named", top + 0.40), ("not_named", "not named", top + 0.65)):
        s = rec[side]
        axb.text(bx_(-0.13), yy, f"{s['n']} {word}", ha="right", va="center", fontsize=FS, color=style.INK_2)
        if s.get("auc_within_era") is None:  # the plan does not read a group of fewer than 10 works
            continue
        if s.get("ci95"):
            axb.plot(s["ci95"], [yy, yy], color=style.INK_2, lw=0.9, zorder=3, solid_capstyle="butt")
        axb.plot(s["auc_within_era"], yy, marker="o", ms=7.5, mfc=face, mec=edge, mew=1.3, zorder=4)
        axb.text(bx_(B_PLOT + B_RIGHT), yy, f"{s['auc_within_era']:.2f}", ha="right", va="center", fontsize=7.5,
                 color=style.INK)

# ---- legend: what the marks mean (models and groups are named beside their marks) ----
key = dict(color=style.INK_2, mfc=style.INK_2, mec=style.INK_2)
handles = [Line2D([], [], lw=0.9, marker="o", ms=7, **key, label="Whole summary, with 95% interval"),
           Line2D([], [], ls="none", marker="o", ms=4.2, **key, label="Opening only")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 0), ncol=2, fontsize=FS, handlelength=2.0,
           handletextpad=0.6, columnspacing=1.8, borderpad=0.01 * W * 72 / FS, borderaxespad=0)
box = fig.get_tightbbox()
if box.x0 < -0.004 or box.x1 > W + 0.004:  # the PNG must stay at the text width
    print(f"WARNING: something is drawn outside the canvas: from {box.x0:.3f} to {box.x1:.3f} in of {W:.2f}")
style.save(fig, "fig_holmes.png")
