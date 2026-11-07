"""Figure for paper v2, experiment E1 ("identity"): telling Batman from other hero films without names.

A horizontal dot plot of the AUC with which each score separates the name-free Batman summaries from the
masked-hero and superhero films (0.5 is chance). Rows, top to bottom: the canon share and the centroid of
units, one line per model, then two word-based comparators (the TF-IDF centroid and text length alone).
The large mark is the AUC on whole summaries, with its 95% interval as a line. The small mark under it is
the AUC on the openings only, which are the same length in both groups. A grey tick on each line marks the
95th percentile of that score's label-shuffle null. Both AUCs are printed at the right.

Reads results/v2/experiments_v2.json (or $RESULTS_DIR/experiments_v2.json) and draws whichever of the four
models are in it. Writes fig_identity.png to paper/v2/ (or $PAPER_DIR).

    cd study && ../.venv/bin/python v2/figs/identity.py
"""
import json
import os
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
os.environ.setdefault("PAPER_DIR", str(STUDY / "paper" / "v2"))
sys.path.insert(0, str(STUDY / "paper" / "figs"))
import style  # noqa: E402

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

TITLE = "Telling Batman from other hero films once the names are removed"
SUBTITLE = "AUC of the name-free Batman summaries against masked-hero and superhero films, for four scores"

RESULTS = Path(os.environ.get("RESULTS_DIR") or STUDY / "results" / "v2")
E1 = {tag: block["E1"] for tag, block in json.loads((RESULTS / "experiments_v2.json").read_text()).items()
      if isinstance(block, dict) and "E1" in block}

GREY = style.SHARED
MODELS = [  # tag in the results, display name, marker face, marker edge; the primary model is the solid one
    ("qwen3-32b", "Qwen3-32B", style.NIGHT, style.NIGHT),
    ("qwen3-8b", "Qwen3-8B", "white", style.NIGHT),
    ("llama3.1-8b", "Llama 3.1 8B", GREY, GREY),
    ("llama3.3-70b", "Llama 3.3 70B", "white", GREY),
]
UNIT_SCORES = [("canon_share", "Canon share\n(units)"), ("unit_centroid", "Centroid of units")]
WORD_SCORES = [("tfidf_centroid", "Word counts\n(TF-IDF centroid)"),
               ("length_alone_sentences", "Length alone\n(number of sentences)")]
WHOLE, OPENING, AGAINST = "A_name_free", "openings", "hero_film_cousins"
CHANCE = 0.5  # an AUC of one half is chance by definition; it is not a result
FS = 7        # labels; nothing is below 6.5 pt


def read(block, version):
    """One score in one version: AUC against the hero films, its 95% interval, and the shuffle null's p95."""
    v = block.get(version) or {}
    if AGAINST not in v:
        return None
    return {**v[AGAINST], "p95": (v.get("null_label_shuffle") or {}).get("p95")}


# ---- rows: (score label, [(row name, marker, face, edge, that score's block in the results), ...]) ----
groups = []
for key, label in UNIT_SCORES:
    rows = [(name, "o", face, edge, E1[tag][key]) for tag, name, face, edge in MODELS if key in E1.get(tag, {})]
    if rows:
        groups.append((label, rows))
for key, label in WORD_SCORES:  # comparators, not models: a neutral grey diamond
    if key in E1.get("words", {}):
        groups.append((label, [("", "D", style.INK_3, style.INK_3, E1["words"][key])]))
blocks = [row[4] for _, rows in groups for row in rows]
# the pre-registered bar is spelled in the name of its clause in the results ("auc_at_least_0.80")
BAR = float(next(k for b in blocks for k in b.get("clauses", {}) if k.startswith("auc_at_least_")).rsplit("_", 1)[1])

# ---- layout, in inches ----
W = style.WIDTH_IN - 0.16             # the tight crop adds 0.08 in on each side
LEFT, RIGHT, BOTTOM = 2.2, 0.9, 0.6   # label columns; value columns; tick labels and legend
PITCH, GAP, DROP, PAD = 0.30, 0.20, 0.115, 0.17  # row pitch; extra between groups; drop of the small mark; ends
n_rows = sum(len(rows) for _, rows in groups)
ax_h = 2 * PAD + (n_rows - 1) * PITCH + (len(groups) - 1) * GAP + DROP
H = (ax_h + BOTTOM + 0.14 * (SUBTITLE.count("\n") + 1) + 0.36) / 0.94  # style.title sets the subtitle 6% down
plot_w = W - LEFT - RIGHT
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([LEFT / W, BOTTOM / H, plot_w / W, ax_h / H])
style.title(fig, TITLE, SUBTITLE)

ax.set_xlim(0, 1)
ax.set_ylim(ax_h, 0)  # y runs downwards, in inches from the top of the plot
ax.spines[:].set_visible(False)
ax.set_yticks([])
ticks = [0, 0.25, 0.5, 0.75, 1]
ax.set_xticks(ticks, [f"{t:.2f}" for t in ticks])
ax.tick_params(axis="x", length=0, pad=4, labelsize=FS)
for t in ticks:
    ax.axvline(t, color=style.RULE, lw=0.8, zorder=0.5, clip_on=False)
head = dict(va="bottom", fontsize=FS, color=style.INK_2)
for x, text, dashes in ((CHANCE, "chance", (4, 2)), (BAR, "pre-registered bar", (1, 2))):
    ax.axvline(x, color=style.INK_3, lw=0.8, ls=(0, dashes), zorder=1)
    ax.text(x, -0.05, text, ha="center", **head)
X_WHOLE, X_OPENING = 1 + 0.38 / plot_w, 1 + RIGHT / plot_w  # right edges of the two columns of values
ax.text(X_WHOLE, -0.05, "whole", ha="right", **head)
ax.text(X_OPENING, -0.05, "opening", ha="right", **head)

# ---- marks ----
in_fig = blended_transform_factory(fig.dpi_scale_trans, ax.transData)  # x in inches from the left edge, y on rows
y = PAD
for label, rows in groups:
    y_mid = y + (len(rows) - 1) * PITCH / 2
    ax.text(0.01 * W, y_mid + DROP / 2, label, transform=in_fig, ha="left", va="center", fontsize=8,
            color=style.INK, linespacing=1.25)
    for name, marker, face, edge, block in rows:
        whole, opening = read(block, WHOLE), read(block, OPENING)
        size = 6.4 if marker == "D" else 7.5
        ax.text(LEFT - 0.13, y, name, transform=in_fig, ha="right", va="center", fontsize=FS, color=style.INK_2)
        if whole and whole.get("ci95"):
            ax.plot(whole["ci95"], [y, y], color=style.INK_2, lw=0.9, zorder=3, solid_capstyle="butt", clip_on=False)
        for cell, yy, ms, x_text, ink, half in ((whole, y, size, X_WHOLE, style.INK, 0.06),
                                                (opening, y + DROP, 0.6 * size, X_OPENING, style.INK_3, 0.04)):
            if cell is None:
                continue
            if cell["p95"] is not None:  # label-shuffle null, 95th percentile
                ax.plot([cell["p95"]] * 2, [yy - half, yy + half], color=GREY, lw=1.0, zorder=2)
            ax.plot(cell["auc"], yy, marker=marker, ms=ms, mfc=face, mec=edge, mew=0.17 * ms, zorder=4, clip_on=False)
            ax.text(x_text, y, f"{cell['auc']:.2f}", ha="right", va="center", fontsize=7.5 if cell is whole else FS,
                    color=ink)
        y += PITCH
    y += GAP

# ---- legend: what the marks mean (the models are named on their rows) ----
key = dict(color=style.INK_2, mfc=style.INK_2, mec=style.INK_2)
handles = [
    Line2D([], [], lw=0.9, marker="o", ms=7, **key, label="Whole summary, with 95% interval"),
    Line2D([], [], ls="none", marker="o", ms=4.2, **key, label="Opening of the summary only"),
    Line2D([], [], ls="none", marker="|", ms=9, mew=1.0, color=GREY, label="95th percentile with labels shuffled"),
]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=3, fontsize=FS, handlelength=2.0,
           handletextpad=0.6, columnspacing=1.8, borderpad=0.2, borderaxespad=0.0)
style.save(fig, "fig_identity.png")
