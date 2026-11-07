"""Figure for paper v2: the size of Batman's canon, and the same count for other film series.

fig_canon.png
  A  For each of three tiers (units present in at least 6, 23 and 29 of the 32 Batman works), four counts on a
     log axis: distinctive units on the original texts, distinctive units on the name-free texts, all units with
     no base-rate cap (original texts), and the mean of the same uncapped count when 32 random ordinary films
     play the franchise. Distinctive units are those under the base-rate cap in ordinary film plots.
  B  One row per film series: the number of distinctive units present in at least 90% of its films (open circle),
     joined to the mean of the same count over random subsets of Batman works of the same size (solid circle).
     Rows are sorted by the series' count.

Filled marks are Batman, open marks are other films. One model is drawn: Qwen3-32B when its files are there,
otherwise the first model in MODELS that has both files. The subtitle names it.

Reads results/v2/canon_counts_<model>.json and results/v2/franchises_<model>.json (another folder:
RESULTS_DIR=...). Every number drawn is read from those files at run time. Writes fig_canon.png to paper/v2/
(or $PAPER_DIR).

    cd study && ../.venv/bin/python v2/figs/canon.py
"""
import json
import os
import sys
from math import ceil, log10
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
os.environ.setdefault("PAPER_DIR", str(STUDY / "paper" / "v2"))
sys.path.insert(0, str(STUDY / "paper" / "figs"))
import style  # noqa: E402

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator, NullLocator  # noqa: E402

TITLE = "How many units recur across Batman works and across other film series"
SUBTITLE = ("Units of {model}, counted by the number of works in a set where they are present.\n"
            "Distinctive units are those that are rare in ordinary film plots.")
PANEL_A = "Batman's canon, with and without names"
PANEL_B = "Batman against other film series"
X_LABEL_A = "Number of units (log scale)"
X_LABEL_B = "Distinctive units in at least {share}% of the films"

MODELS = [("qwen3-32b", "Qwen3-32B"), ("qwen3-8b", "Qwen3-8B"), ("llama3.1-8b", "Llama 3.1 8B"),
          ("llama3.3-70b", "Llama 3.3 70B")]  # in order of preference; the first one present is drawn
SERIES_TIER = "at_least_90pct"  # panel B's tier, as the franchise files spell it
GREY = style.SHARED
# panel A, top to bottom within a tier: key in the canon counts file, legend label, mark
ROWS_A = [
    ("batman_distinctive_units", "Distinctive units, original texts",
     dict(marker="o", ms=6.2, mfc=style.NIGHT, mec=style.NIGHT, mew=0.8)),
    ("batman_name_free_distinctive_units", "Distinctive units, texts with the names removed",
     dict(marker="o", ms=6.2, mfc=style.NIGHT_SOFT, mec=style.NIGHT, mew=0.8)),
    ("batman_all_units", "All units, original texts", dict(marker="o", ms=6.2, mfc=GREY, mec=GREY, mew=0.8)),
    ("random_sets_of_{n}_ordinary_films_all_units", "All units, random sets of {n} ordinary films (mean)",
     dict(marker="s", ms=5.2, mfc="white", mec=style.INK_3, mew=1.0)),
]
SERIES = dict(marker="o", ms=5.6, mfc="white", mec=style.INK_2, mew=1.2)       # panel B: the series' own films
BATMAN = dict(marker="o", ms=6.2, mfc=style.NIGHT, mec=style.NIGHT, mew=0.8)   # panel B: Batman subsets

RESULTS = Path(os.environ.get("RESULTS_DIR") or STUDY / "results" / "v2")
present = [(tag, name) for tag, name in MODELS
           if all((RESULTS / f"{kind}_{tag}.json").exists() for kind in ("canon_counts", "franchises"))]
print("models with both files:", ", ".join(tag for tag, _ in present) or "none")
tag, model = present[0]
c = json.loads((RESULTS / f"canon_counts_{tag}.json").read_text())
f = json.loads((RESULTS / f"franchises_{tag}.json").read_text())
assert c["model"] == f["model"] == tag, (c["model"], f["model"], tag)
print("drawn:", tag)


def number(x):
    """Whole numbers with a thousands comma; means are rounded as in v2/analysis/paper_tables.py."""
    return f"{x:,.0f}"


# ---- what is drawn ----
n_works = f["batman"]["n_works"]
tiers = sorted(c[ROWS_A[0][0]], key=lambda k: int(k.rsplit("_", 1)[1]))  # at_least_6, at_least_23, at_least_29
a_rows = []                                                              # per tier: (label, [(value, mark), ...])
for i, tier in enumerate(tiers):
    k = tier.rsplit("_", 1)[1]
    cells = [c[key.format(n=n_works)][tier] for key, _, _ in ROWS_A]
    values = [cell["mean"] if isinstance(cell, dict) else cell for cell in cells]
    a_rows.append((f"At least {k}" if i else f"In at least {k}\nof the {n_works} works",
                   list(zip(values, (kw for _, _, kw in ROWS_A)))))
    print(f"  A, {tier}:", ", ".join(number(v) for v in values))
share = SERIES_TIER.split("_")[-1].removesuffix("pct")
b_rows = sorted(((f"{name} ({s['n_films']} films)", s[SERIES_TIER],
                  s["batman_subsets_of_the_same_size"][SERIES_TIER]["mean"]) for name, s in f["franchises"].items()),
                key=lambda r: -r[1])
for label, own, batman in b_rows:
    print(f"  B, {label}: {number(own)} against {number(batman)} for Batman subsets")

# ---- geometry, in inches; y is measured down from the top ----
W = style.WIDTH_IN - 0.16                 # saving adds 0.08 in on each side, so the PNG is exactly the text width
FS = 7
xa, xa0, xa1 = 0.05, 1.02, 2.72           # panel A: labels start, axis from, axis to (the values run past it)
xb, xb0, xb1 = 3.42, 4.76, 6.28           # panel B: the same
PITCH, GAP, PAD = 0.165, 0.13, 0.10       # panel A: row pitch, extra between tiers, ends
y0 = 1.02                                 # top of both axes
hp = 2 * PAD + (len(tiers) * len(ROWS_A) - 1) * PITCH + (len(tiers) - 1) * GAP
y_leg = y0 + hp + 0.50                    # top of the legends
H = y_leg + len(ROWS_A) * 0.155 + 0.04

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])           # the whole canvas, in inches, for the labels around the two plots
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
style.title(fig, TITLE)
fig.text(0.01, 1 - (0.015 * H + 0.20) / H, SUBTITLE.format(model=model), ha="left", va="top", fontsize=7.5,
         color=style.INK_2, linespacing=1.3)
for x, letter, text in ((xa, "A", PANEL_A), (xb, "B", PANEL_B)):
    ax.text(x, y0 - 0.20, letter, fontsize=9, fontweight="bold", va="center")
    ax.text(x + 0.18, y0 - 0.20, text, fontsize=7.6, fontweight="bold", va="center")


def plot_area(x_from, x_to, x_label, **where):
    """An axis over the given span, with y in inches from its top and only the x axis shown."""
    a = fig.add_axes([x_from / W, 1 - (y0 + hp) / H, (x_to - x_from) / W, hp / H], facecolor="none")
    a.set_ylim(hp, 0)
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.tick_params(axis="x", length=0, pad=4)
    a.grid(axis="x", color=style.RULE, linewidth=0.6, zorder=0)
    ax.text(where.pop("x", (x_from + x_to) / 2), y0 + hp + 0.31, x_label, va="center", fontsize=FS,
            color=style.INK_2, **{"ha": "center", **where})
    return a


def value(a, x, y, text, side=1, color=style.INK):
    """A number beside its mark, on a white ground so that no grid line runs through it."""
    a.annotate(text, (x, y), xytext=(6.5 * side, 0), textcoords="offset points", ha="left" if side > 0 else "right",
               va="center", fontsize=FS, color=color, zorder=2, bbox=dict(fc="white", ec="none", pad=0.6))


# ---- A: the three tiers. The axis is linear from 0 to 1 and logarithmic above, so a count of zero has a place ----
top = 10 ** ceil(log10(max(max(v for v, _ in rows) for _, rows in a_rows)))
decades = [10 ** k for k in range(round(log10(top)) + 1)]
pa = plot_area(xa0, xa1, X_LABEL_A)
pa.set_xscale("symlog", linthresh=1, linscale=0.4)
pa.set_xlim(0, top)
pa.set_xticks([0] + decades, ["0"] + [f"{t:,}" if i % 2 == 0 else "" for i, t in enumerate(decades)])
pa.xaxis.set_minor_locator(NullLocator())
y = PAD
for label, rows in a_rows:
    if y > PAD:  # a thin rule between tiers
        ax.plot([xa, xa1 + 0.50], [y0 + y - (PITCH + GAP) / 2] * 2, color=style.RULE, linewidth=0.5, zorder=0)
    ax.text(xa, y0 + y + (len(rows) - 1) * PITCH / 2, label, va="center", fontsize=7.5, linespacing=1.25)
    for i, (v, kw) in enumerate(rows):
        pa.plot(v, y, ls="", clip_on=False, zorder=3, **kw)
        value(pa, v, y, number(v), color=style.INK if i < 2 else style.INK_3)
        y += PITCH
    y += GAP

# ---- B: each series against Batman subsets of its size ----
pb = plot_area(xb0, xb1, X_LABEL_B.format(share=share), x=W - 0.04, ha="right")  # wider than its axis
widest = max(max(own, batman) for _, own, batman in b_rows)
pb.set_xlim(-0.12 * widest, 1.08 * widest)   # room on the left for a number beside a mark near zero
pb.xaxis.set_major_locator(MaxNLocator(4, integer=True))
for j, (label, own, batman) in enumerate(b_rows):
    y = hp * (j + 0.5) / len(b_rows)
    ax.text(xb, y0 + y, label, va="center", fontsize=7.5)
    pb.plot([own, batman], [y, y], color=GREY, linewidth=1.0, zorder=2.5, solid_capstyle="butt")
    pb.plot(own, y, ls="", clip_on=False, zorder=3, **SERIES)
    pb.plot(batman, y, ls="", clip_on=False, zorder=4, **BATMAN)
    value(pb, own, y, number(own), side=1 if own > batman else -1)        # each number on the outer side
    value(pb, batman, y, number(batman), side=-1 if own > batman else 1)

# ---- legends, one under each panel, in the order of the rows ----
legend = dict(loc="upper left", ncol=1, fontsize=FS, labelcolor=style.INK_2, handlelength=1.2, handletextpad=0.5,
              labelspacing=0.45, borderpad=0, borderaxespad=0)
fig.legend(handles=[Line2D([], [], ls="", label=text.format(n=n_works), **kw) for _, text, kw in ROWS_A],
           bbox_to_anchor=(xa / W, 1 - y_leg / H), **legend)
fig.legend(handles=[Line2D([], [], ls="", label="Films of the series", **SERIES),
                    Line2D([], [], ls="", label="Batman works, random subsets of the same size (mean)", **BATMAN)],
           bbox_to_anchor=(xb / W, 1 - y_leg / H), **legend)
style.save(fig, "fig_canon.png")
