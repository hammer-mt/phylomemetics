"""Figure for paper v2: what changed the results between version 1 and version 2.

fig_contrast.png
  Three small panels that share the same three columns: version 1 as published (4-bit weights, version 1's rule),
  full precision with version 1's rule, and full precision with version 2's rule. One line per model joins the
  columns it has: a model with one column is a single mark, and a line passes a column the model lacks without
  a mark. Values are printed at the last column and, for the highest and lowest model, at the first.
    1  Identity: the AUC of the canon share against the hero films, with its 95% interval (chance and the bar).
    2  Heredity: the share of eligible comics that beat the documented source, with its exact p (chance).
    3  Beyond words: the exact p of the residual test in version 1's two-measure form, on a log axis (p = 0.05).

Reads results/v2/contrasts.json for every mark, and results/v2/experiments_v2.json for the two reference lines
that are results of the study (the pre-registered AUC bar and the chance mean of the heredity share). Another
folder: RESULTS_DIR=... Models that are not in the files yet are skipped, so the script is re-run when they arrive.

    cd study && ../.venv/bin/python v2/figs/contrast.py
"""
import json
import os
import sys
from math import ceil, floor, log10
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
os.environ.setdefault("PAPER_DIR", str(STUDY / "paper" / "v2"))
sys.path.insert(0, str(STUDY / "paper" / "figs"))
import style  # noqa: E402

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

TITLE = "The same three tests in version 1 and in version 2, model by model"
SUBTITLE = ("Each model's results as published in version 1 (4-bit weights), then at full precision under each version's rule.\n"
            "Dashed lines are chance, dotted lines the pre-registered bars. Vertical lines in the first panel are 95% intervals.")
PANELS = ["Identity: AUC against hero films", "Heredity: comics that beat the source", "Beyond words: residual p"]
# cell name in the results; first and second line of its label ("full precision" is written once, over both rules)
CELLS = [("v1 published", "v1 as", "published"), ("full precision, v1 rule", "", "v1 rule"),
         ("full precision, v2 rule", "", "v2 rule")]
GROUP = "full precision"

# tag in the results, display name, mark, line. The primary model is the solid dark dot; the others are quieter.
MODELS = [
    ("qwen3-32b", "Qwen3-32B", dict(marker="o", ms=5.6, mfc=style.NIGHT, mec=style.NIGHT, mew=1.0), dict(color=style.NIGHT, lw=1.5)),
    ("qwen3-8b", "Qwen3-8B", dict(marker="o", ms=5.0, mfc="white", mec=style.NIGHT, mew=1.2), dict(color=style.NIGHT, lw=0.9)),
    ("llama3.1-8b", "Llama 3.1 8B", dict(marker="o", ms=5.4, mfc=style.SHARED, mec=style.SHARED, mew=1.0), dict(color=style.SHARED, lw=1.2)),
    ("llama3.3-70b", "Llama 3.3 70B", dict(marker="o", ms=5.0, mfc="white", mec=style.SHARED, mew=1.2), dict(color=style.SHARED, lw=0.9)),
]
CHANCE_AUC = 0.5  # an AUC of one half is chance by definition; it is not a result
ALPHA = 0.05      # the conventional level the residual test is held to; it is not a result
DASH, DOT = (0, (4, 2)), (0, (1, 2))
FS, FS_S = 7, 6.5  # nothing is below 6.5 pt

RESULTS = Path(os.environ.get("RESULTS_DIR") or STUDY / "results" / "v2")
cells = json.loads((RESULTS / "contrasts.json").read_text())["cells"]
exp = {k: b for k, b in json.loads((RESULTS / "experiments_v2.json").read_text()).items() if isinstance(b, dict)}
models = [m for m in MODELS if any(f"{m[0]} | {c[0]}" in cells for c in CELLS)]
print("models in the file:", ", ".join(m[0] for m in models) or "none")
# the pre-registered bar is spelled in the name of its clause ("auc_at_least_0.80"), as in v2/figs/identity.py
clauses = [k for b in exp.values() for s in b.get("E1", {}).values() if isinstance(s, dict) for k in s.get("clauses", {})]
BAR = float(next(k for k in clauses if k.startswith("auc_at_least_")).rsplit("_", 1)[1])
chances = [b["E2"]["summary"]["chance_mean"] for b in exp.values() if "summary" in b.get("E2", {})]
CHANCE_E2 = sum(chances) / len(chances)  # the same line as in v2/figs/heredity.py
print(f"bar {BAR}; heredity chance {CHANCE_E2:.4f}, the mean of {len(chances)} chance means ({min(chances):.4f} to {max(chances):.4f})")


def pv(p):
    """A p-value, formatted as in v2/analysis/paper_tables.py so that figure and tables agree."""
    return f"{p:.2g}" if p < 0.001 else (f"{p:.4f}" if p < 0.01 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}")


def spread(ys, fixed, gap):
    """Move labels apart (ys in points) until neighbours are at least gap apart; fixed ones stay where they are."""
    ys, order = list(ys), sorted(range(len(ys)), key=lambda i: ys[i])
    for _ in range(200):
        for a, b in zip(order, order[1:]):
            d = gap - (ys[b] - ys[a])
            if d > 0:
                ys[a] -= 0 if fixed[a] else d if fixed[b] else d / 2
                ys[b] += 0 if fixed[b] else d if fixed[a] else d / 2
    return ys


values = lambda key: [c[key] for c in cells.values() if c.get(key) is not None]  # noqa: E731
panels = [  # statistic, y limits, log axis, reference lines, the two parts of a printed value
    dict(key="E1_auc", ylim=(0, 1), ticks=[0, 0.25, 0.5, 0.75, 1], refs=[(CHANCE_AUC, "chance", DASH), (BAR, f"bar {BAR:.2f}", DOT)],
         text=lambda c: (f"{c['E1_auc']:.2f}", "")),
    dict(key="E2_share", ylim=(0, max(0.55, ceil(max(values("E2_share")) * 10) / 10 + 0.05)), refs=[(CHANCE_E2, "chance", DASH)],
         text=lambda c: (f"{c['E2_share']:.2f}", "p = " + pv(c["E2_exact_p"]))),
    dict(key="residual_exact_p", ylim=(10 ** floor(log10(min(values("residual_exact_p")) / 2)), 1), log=True,
         refs=[(ALPHA, f"p = {ALPHA:g}", DOT)], text=lambda c: (pv(c["residual_exact_p"]), "")),
]

# ---- geometry, in inches ----
W = style.WIDTH_IN - 0.16                 # saving adds 0.08 in on each side, so the PNG is exactly the text width
H, TOP, AX_H = 3.68, 1.07, 1.95           # canvas height; top of the panels; their height
YT, PAD_L, PAD_R = 0.37, 0.17, [0.50, 1.05, 0.50]  # y tick labels; room left of the first column; room for the printed values
U = (W - 0.02 - 3 * (YT + PAD_L) - sum(PAD_R)) / 6  # one step between columns, the same in every panel
DODGE = 0.05 / U                          # models sit side by side in a column, so intervals and equal values stay apart
fig = plt.figure(figsize=(W, H))
fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, fc="none", ec="none"))  # keeps the tight crop at the full canvas
style.title(fig, TITLE, SUBTITLE)
to_pt = 72 / fig.dpi

x0 = 0
for head, p, pad_r in zip(PANELS, panels, PAD_R):
    w = PAD_L + 2 * U + pad_r
    ax = fig.add_axes([(x0 + YT) / W, 1 - (TOP + AX_H) / H, w / W, AX_H / H])
    name, _, what = head.partition(": ")
    fig.text(x0 / W + 0.01, 1 - (TOP - 0.36) / H, name, fontsize=8, fontweight="bold", va="center")
    fig.text(x0 / W + 0.01, 1 - (TOP - 0.21) / H, what, fontsize=FS, color=style.INK_2, va="center")
    x0 += YT + w
    ax.set_xlim(-PAD_L / U, 2 + pad_r / U)
    if p.get("log"):
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
        ax.minorticks_off()
    elif "ticks" in p:
        ax.set_yticks(p["ticks"], [f"{t:g}" for t in p["ticks"]])
    ax.set_ylim(*p["ylim"])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0, pad=3)
    ax.grid(axis="y", color=style.RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(CELLS)), [f"{a}\n{b}" for _, a, b in CELLS], linespacing=1.3)
    ax.set_xticks([1.5], [GROUP + "\n"], minor=True, linespacing=1.3)  # one label over the two full-precision columns
    ax.tick_params(which="minor", length=0, pad=3)
    y_pt = lambda v: ax.transData.transform((0, v))[1] * to_pt  # noqa: E731
    stack = []                            # labels in the room at the right: (x, y, main text, small text, fixed)
    for y, label, dashes in p["refs"]:
        ax.axhline(y, color=style.INK_3, lw=0.8, ls=dashes, zorder=1)
        stack.append((None, y, label, "", True))
    firsts, half = [], (len(models) - 1) / 2 * DODGE
    for i, (tag, _, mark, line) in enumerate(models):
        pts = [(j, cells[f"{tag} | {c[0]}"]) for j, c in enumerate(CELLS)
               if cells.get(f"{tag} | {c[0]}", {}).get(p["key"]) is not None]
        if not pts:
            continue
        xs, ys = [j + i * DODGE - half for j, _ in pts], [c[p["key"]] for _, c in pts]
        for x, (_, c) in zip(xs, pts):
            if p["key"] == "E1_auc" and c.get("E1_ci95"):
                ax.plot([x, x], c["E1_ci95"], color=line["color"], lw=0.7, alpha=0.55, zorder=2, solid_capstyle="butt")
        ax.plot(xs, ys, zorder=3 + (i == 0), clip_on=False, **mark, **line)
        stack.append((pts[-1][0] + half, ys[-1], *p["text"](pts[-1][1]), False))  # one column of values, right of all marks
        if len(pts) > 1 and pts[0][0] == 0:
            firsts.append((ys[0], xs[0], p["text"](pts[0][1])[0]))
    # values at the last column: beside the mark, moved apart where they would touch each other or a line's label
    placed = spread([y_pt(y) + (4.5 if fixed else 0) for _, y, _, _, fixed in stack], [s[4] for s in stack], 8.6)
    for (x, y, main, small, fixed), y_new in zip(stack, placed):
        kw = dict(textcoords="offset points", va="center", zorder=5)
        if fixed:
            ax.annotate(main, (1, y), xycoords=("axes fraction", "data"), xytext=(0, y_new - y_pt(y)), ha="right",
                        fontsize=FS_S, color=style.INK_3, **kw)
            continue
        ax.annotate(main, (x, y), xytext=(6.5, y_new - y_pt(y)), fontsize=FS, color=style.INK, **kw)
        ax.annotate(small, (x, y), xytext=(26, y_new - y_pt(y)), fontsize=FS_S, color=style.INK_3, **kw)
    # values at the first column, where there is room: above the highest mark and below the lowest
    firsts.sort()
    for (y, x, main), up in zip(firsts[-1:] + firsts[:1] if len(firsts) > 1 else firsts, (1, -1)):
        ax.annotate(main, (x, y), xytext=(0, 7.5 * up), textcoords="offset points", ha="center", va="center", fontsize=FS,
                    color=style.INK, zorder=5, bbox=dict(fc="white", ec="none", pad=0.4))

handles = [Line2D([], [], label=name, **mark, **line) for _, name, mark, line in models]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 0), ncol=len(handles), fontsize=FS, labelcolor=style.INK_2,
           handlelength=2.2, handletextpad=0.5, columnspacing=1.8, borderaxespad=0.3)
style.save(fig, "fig_contrast.png")
