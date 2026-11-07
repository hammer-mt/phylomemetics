"""Figure for paper v2: which works carry each of the units that were chosen by rule and explained.

fig_genome.png
  A presence grid. Each row is one unit, with the title its explainer gave it; each column is a work: the Batman
  works by date, a thin gap, then the relatives by date. A cell is split in two. The upper half is filled where the
  unit is present in the text as written, the lower half where it is present in the same text with every name
  removed. Rows come in up to three sets, each under a small header: the core canon, the units still present without
  names, and the common story patterns. A set is sorted by the number of Batman works that carry the unit (as
  written, then without names) and cut to its first 22 rows. Beside each title are those two counts. A title is dark
  when the unit detects an idea and grey when it detects a name or a word, as judged blind from its description.
  The strip over the columns is shaded by decade: the works' "era" values are not the keys of style.ERA. A year is
  printed over every fourth Batman work, and each relative is named under its column.

One model is drawn: the first in MODELS that has a units table. The subtitle names it. With two sets the figure
is under 8 inches tall; with three full sets of 22 rows it is about 9 inches, at the smallest row pitch the labels
allow (lower MAX_ROWS if that is too tall).

Reads results/v2/<model>/units_table.json (another folder: RESULTS_DIR=...). Every number drawn is read from that
file at run time. Writes fig_genome.png to paper/v2/ (or $PAPER_DIR).

    cd study && ../.venv/bin/python v2/figs/genome.py
"""
import json
import logging
import os
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
os.environ.setdefault("PAPER_DIR", str(STUDY / "paper" / "v2"))
sys.path.insert(0, str(STUDY / "paper" / "figs"))
import style  # noqa: E402

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_hex  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

TITLE = "Most of the core canon disappears when Batman's names are removed"
SUBTITLE = ("Units of {model} chosen by rule and explained (rows) in {nb} Batman works and {nr} relatives "
            "(columns, oldest first).\n"
            "Each cell shows one work twice: the text as written (upper half) and with every name removed "
            "(lower half).")
SETS = [("core", "Core canon"), ("name_free_core", "Still present without names"),
        ("common_patterns", "Common story patterns")]  # the sets drawn, in this order, and their headers
KEY_UPPER = "Upper half: present in the text as written"
KEY_LOWER = "Lower half: present with every name removed"
KEY_TITLE = "Colour of a title, as judged blind from the description of the unit:"
KEY_COUNTS = "Numbers: Batman works that carry the unit, as written · with every name removed"
KEY_STRIP = "Strip over the columns: decade of the work, from the {first}s"

MODELS = [("qwen3-32b", "Qwen3-32B"), ("qwen3-8b", "Qwen3-8B"), ("llama3.1-8b", "Llama 3.1 8B"),
          ("llama3.3-70b", "Llama 3.3 70B")]  # in order of preference; the first one present is drawn
MAX_ROWS, MAX_CHARS = 22, 46              # rows kept per set; characters kept of a title
DETECTS = {"idea": style.INK}             # title colour by what the unit detects; anything else is grey
# a title can hold a character the house font lacks (a CJK token): let installed fonts that have it stand in
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # the stand-ins have one weight; that is fine
plt.rcParams["font.family"] = ["sans-serif"] + sorted(
    {f.name for f in font_manager.fontManager.ttflist} & {"WenQuanYi Zen Hei", "IPAPGothic", "Noto Sans CJK JP"})

RESULTS = Path(os.environ.get("RESULTS_DIR") or STUDY / "results" / "v2")
present = [(tag, name) for tag, name in MODELS if (RESULTS / tag / "units_table.json").exists()]
print("models with a units table:", ", ".join(tag for tag, _ in present) or "none")
if not present:
    sys.exit(f"no model has <model>/units_table.json in {RESULTS}")
tag, model = present[0]
d = json.loads((RESULTS / tag / "units_table.json").read_text())
assert d["model"] == tag, (d["model"], tag)
works, nb = d["works"], d["n_batman_works"]
nr = len(works) - nb
assert all((w["group"] == "batman") == (i < nb) for i, w in enumerate(works))


def cut(title):
    return title if len(title) <= MAX_CHARS else title[:MAX_CHARS - 1].rstrip() + "…"


# ---- what is drawn: the sets that have units, each sorted and cut ----
groups = []
for key, head in SETS:
    units = sorted((u for u in d["units"] if u["set"] == key),
                   key=lambda u: (-u["batman_works_original"], -u["batman_works_name_free"]))
    for u in units:  # the counts printed beside a title are the cells drawn in its row
        assert sum(u["present_original"][:nb]) == u["batman_works_original"], u["name"]
        assert sum(u["present_name_free"][:nb]) == u["batman_works_name_free"], u["name"]
    shown = units[:MAX_ROWS]
    print(f"  {key}: {len(units)} units, {len(shown)} drawn")
    if shown:
        groups.append((f"{head} ({f'first {len(shown)} of ' if len(shown) < len(units) else ''}{len(units)})", shown))
rows = [u for _, shown in groups for u in shown]
decades = sorted({w["year"] // 10 * 10 for w in works})   # a neutral ramp, light to dark, one step per decade
ramp = LinearSegmentedColormap.from_list("decade", [style.RULE, style.INK])
shade = {dec: to_hex(ramp(k / max(len(decades) - 1, 1))) for k, dec in enumerate(decades)}

# ---- geometry, in inches; y is measured down from the top ----
W = style.WIDTH_IN - 0.16                 # saving adds 0.08 in on each side, so the PNG is exactly the text width
FS, FS_KEY = 6.6, 6.8                     # row labels (also the smallest text); legend
fig = plt.figure(figsize=(W, 1))          # resized below, once the labels are measured


def width(text, fs, **kw):
    t = fig.text(0, 0, text, fontsize=fs, **kw)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


x_text, GAP, HEAD, y_grid = 0.05, 0.8, 0.17, 1.04    # left edge; gap before the relatives, in columns; header; grid top
MAX_H = 8.5 - 0.16                        # rows get closer together, down to the size of their labels, to stay under it
num_w = width("0" * len(str(nb)), FS)
x_title = x_text + max(width(cut(u["title"]), FS) for u in rows)     # titles end here
x_n1 = x_title + 0.09 + num_w                                        # the first count ends here
x_n2 = x_n1 + 0.11 + num_w                                           # the second count ends here
x0, x1 = x_n2 + 0.07, W - 0.16                                       # the grid
COL = (x1 - x0) / (len(works) + GAP)
KEY_TOP, KEY_PITCH = 0.36, 0.18           # legend: its first line under the grid, and its line pitch
names_h = max(width(w["label"], 6.5) for w in works[nb:])            # the relatives' names hang under their columns
bottom = max(0.10 + names_h, KEY_TOP + 3 * KEY_PITCH + 0.06) + 0.05  # the names, and the legend beside them
ROW = min(0.118, max(0.094, (MAX_H - y_grid - HEAD * len(groups) - bottom) / len(rows)))
H = y_grid + HEAD * len(groups) + ROW * len(rows) + bottom
fig.set_size_inches(W, H)
ax = fig.add_axes([0, 0, 1, 1])           # the whole canvas, in inches
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
style.title(fig, TITLE)
fig.text(0.01, 1 - 0.30 / H, SUBTITLE.format(model=model, nb=nb, nr=nr), ha="left", va="top", fontsize=7.5,
         color=style.INK_2, linespacing=1.3)


def X(i):
    """Left edge of column i; the relatives start after the gap."""
    return x0 + (i + (GAP if i >= nb else 0)) * COL


def cell(x, y, upper, lower, px=0.007, py=0.011):
    """One work in one row: the upper half for the text as written, the lower half for the name-free text."""
    w, h = COL - 2 * px, ROW - 2 * py
    ax.add_patch(Rectangle((x + px, y + py), w, h, fc=style.OFF, lw=0))
    if upper:
        ax.add_patch(Rectangle((x + px, y + py), w, h / 2, fc=style.NIGHT, lw=0))
    if lower:
        ax.add_patch(Rectangle((x + px, y + py + h / 2), w, h / 2, fc=style.SIGNAL, lw=0))


# ---- over the columns: the two blocks, a year over every fourth Batman work, the decade strip ----
y_strip = y_grid - 0.07
for a, b, text in ((0, nb, f"{nb} Batman works"), (nb, len(works), f"{nr} relatives")):
    ax.plot([X(a) + 0.01, X(b - 1) + COL - 0.01], [y_grid - 0.25] * 2, color=style.INK_3, lw=0.6)
    ax.text((X(a) + X(b - 1) + COL) / 2, y_grid - 0.28, text, ha="center", va="bottom", fontsize=FS_KEY,
            color=style.INK_2)
for i, w in enumerate(works):
    ax.add_patch(Rectangle((X(i) + 0.007, y_strip), COL - 0.014, 0.045, fc=shade[w["year"] // 10 * 10], lw=0))
    if i < nb and i % 4 == 0:
        ax.plot([X(i) + COL / 2] * 2, [y_strip - 0.035, y_strip - 0.012], color=style.INK_3, lw=0.5)
        ax.text(X(i) + COL / 2, y_strip - 0.045, w["year"], ha="center", va="bottom", fontsize=6.5, color=style.INK_3)

# ---- the grid ----
y = y_grid
for head, shown in groups:
    ax.text(x_text, y + HEAD - 0.04, head, va="bottom", fontsize=7, fontweight="bold")
    ax.plot([x_text, x1], [y + HEAD - 0.02] * 2, color=style.RULE, lw=0.5)
    y += HEAD
    for u in shown:
        label = dict(va="center", ha="right", fontsize=FS)
        ax.text(x_title, y + ROW / 2, cut(u["title"]), color=DETECTS.get(u["detects"], style.INK_3), **label)
        ax.text(x_n1, y + ROW / 2, u["batman_works_original"], color=style.INK_2, **label)
        ax.text((x_n1 + x_n2 - num_w) / 2, y + ROW / 2, "·", color=style.INK_3, va="center", ha="center", fontsize=FS)
        ax.text(x_n2, y + ROW / 2, u["batman_works_name_free"], color=style.INK_2, **label)
        for i, (upper, lower) in enumerate(zip(u["present_original"], u["present_name_free"], strict=True)):
            cell(X(i), y, upper, lower)
        y += ROW

# ---- under the relatives: their names. The names need more room than a column, so they fan out a little ----
step = max(COL, 0.094)
for j, w in enumerate(works[nb:]):
    xc = X(nb + j) + COL / 2
    xl = (X(nb) + X(len(works) - 1) + COL) / 2 + (j - (nr - 1) / 2) * step
    ax.plot([xc, xl], [y + 0.012, y + 0.085], color=style.SHARED, lw=0.5)
    ax.text(xl, y + 0.10, w["label"], rotation=90, rotation_mode="anchor", ha="right", va="center", fontsize=6.5,
            color=style.INK_2)


# ---- legend, bottom left: four lines ----
def put(x, yk, text, color=style.INK_2):
    ax.text(x, yk, text, va="center", fontsize=FS_KEY, color=color)
    return x + width(text, FS_KEY)


yk = [y + KEY_TOP + KEY_PITCH * k for k in range(4)]
x = x_text
for upper, lower, text in ((1, 0, KEY_UPPER), (0, 1, KEY_LOWER)):
    cell(x, yk[0] - ROW / 2, upper, lower)
    x = put(x + COL + 0.04, yk[0], text) + 0.2
x = put(x_text, yk[1], KEY_TITLE)
x = put(x + 0.08, yk[1], "idea", style.INK)
put(x + 0.14, yk[1], "name or word", style.INK_3)
put(x_text, yk[2], KEY_COUNTS)
x = put(x_text, yk[3], KEY_STRIP.format(first=decades[0])) + 0.06
for dec in decades:
    ax.add_patch(Rectangle((x, yk[3] - 0.03), 0.075, 0.06, fc=shade[dec], lw=0))
    x += 0.085
put(x + 0.04, yk[3], f"to the {decades[-1]}s")
style.save(fig, "fig_genome.png")
