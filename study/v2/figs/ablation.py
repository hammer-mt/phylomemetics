"""Figure for paper v2: the ablation test (switching off the units of Batman's canon).

Each model reads a plot summary in which every name is replaced by a code, and is asked which hero the story is
about. The outcome is the log-odds that its reply starts with the hero's name; a drop is how far they fall when a
set of units is switched off while the model reads the summary.

fig_ablation.png
  A  One row per model and test, on one axis: the mean drop over the Batman works with the canon switched off (the
     model's mark), the same canon's mean drop on the other hero films, each scored on its own hero (hollow
     diamond), and under them the mean drop of every random set of matched units, one thin tick per set, in two
     strips (light: matched on layer, pole and number of texts; dark: also matched on how strongly the units fire).
     The three p values of the criterion are printed at the right. Under a model's registered test comes its
     follow-up test, with the reply begun for the model (PLAN.md Section 18, change 33), when the results hold it.
  B  The first model in MODELS that has a file, work by work: the drop with the canon switched off, the Batman
     works on the left and the hero films on the right, each block sorted from the largest drop to the smallest. A
     grey dash is the work's mean drop over the random sets. The hero films are named under the axis; of the
     Batman works only the three largest and the three smallest drops are named.

Reads results/v2/ablation_<model>.json for every model that has one (another folder: RESULTS_DIR=...) and skips the
others. Every number drawn is read from those files at run time. Writes fig_ablation.png to paper/v2/ (or $PAPER_DIR).

    cd study && ../.venv/bin/python v2/figs/ablation.py
"""
import json
import os
import sys
from math import floor, log10
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
os.environ.setdefault("PAPER_DIR", str(STUDY / "paper" / "v2"))
sys.path.insert(0, str(STUDY / "paper" / "figs"))
import style  # noqa: E402

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

TITLE = "Switching off Batman's canon lowers the model's recognition of Batman"
SUBTITLE = ("Each model is asked which hero the story is about. Shown is how far the log-odds that its reply starts\n"
            "with the hero's name fall when a set of units is switched off (0 = no change). By the rules set beforehand\n"
            "neither test passes: five Qwen3-32B replies do not begin with a name, and Qwen3-8B's effect is not specific.")
PANEL_A = "Mean drop over the works: the canon against random sets of matched units"
PANEL_B = "{model}, registered test, work by work: drop in the log-odds of the hero's name with the canon switched off"
X_LABEL = "Mean drop in the log-odds of the hero's name"
CAPTION_BATMAN = "{n} Batman works, from the largest drop to the smallest"
CAPTION_HEROES = "{n} other hero films,\neach scored on its own hero"

REGISTERED = "\nregistered test"
FOLLOW_UP = "\nfollow-up: reply\nbegun with \u201c{prefix}\u201d"

GREY = style.SHARED
MODELS = [  # tag in the file name, display name, marker face, marker edge; the primary model is the solid one
    ("qwen3-32b", "Qwen3-32B", style.NIGHT, style.NIGHT),
    ("qwen3-8b", "Qwen3-8B", "white", style.NIGHT),
    ("llama3.1-8b", "Llama 3.1 8B", GREY, GREY),
    ("llama3.3-70b", "Llama 3.3 70B", "white", GREY),
]
HERO = dict(marker="D", mfc="white", mec=style.INK_2)  # the canon switched off on the other hero films
# the two controls: key in the results, colour, legend text, key of its p in the criterion, column head of that p
CONTROLS = [
    ("matched_random_sets", GREY, "{n} random sets matched on layer, pole and number of texts",
     "p_against_sets_matched_on_works", "p against\nrandom sets"),
    ("magnitude_matched_random_sets", style.INK_3, "{n} random sets also matched on how strongly they fire",
     "p_against_sets_also_matched_on_activity", "p against sets\nalso matched\non firing"),
]
P_COLUMNS = [(key, head) for _, _, _, key, head in CONTROLS] + [("specificity_p", "p against\nhero films")]

RESULTS = Path(os.environ.get("RESULTS_DIR") or STUDY / "results" / "v2")
names = {tag: name for tag, name, _, _ in MODELS}
runs = []  # per model that has a file: display name, marker colours, the name-free result, the criterion
for tag, _, face, edge in MODELS:
    if (RESULTS / f"ablation_{tag}.json").exists():
        d = json.loads((RESULTS / f"ablation_{tag}.json").read_text())
        assert d["model"] == tag, (d["model"], tag)
        runs.append((names[d["model"]] + REGISTERED, dict(mfc=face, mec=edge), d["story_tokens_name_free"], d["criterion"]))
        if "reply_begun" in d:      # the follow-up test: the same jobs with the model's reply begun for it
            runs.append((names[d["model"]] + FOLLOW_UP.format(prefix=d["criterion_reply_begun"]["reply_prefix"]), dict(mfc=face, mec=edge),
                         d["reply_begun"], d["criterion_reply_begun"]))
if not runs:
    sys.exit(f"no ablation_<model>.json in {RESULTS}")
print("models with a file:", ", ".join(name for name, *_ in runs))


def num(x):
    """Two decimals, as in v2/analysis/paper_tables.py, with a true minus sign."""
    return f"{x:.2f}".replace("-", "−")


def fmt_p(p):
    """Two significant figures, never in scientific notation (as in heredity.py)."""
    return "n/a" if p is None else f"{p:.{max(2, 1 - floor(log10(p)))}f}" if p > 0 else "0"


def value(a, x, y, text, side=1):
    """A number beside its mark, with a white outline so that no line runs through it."""
    a.annotate(text, (x, y), xytext=(7 * side, 0), textcoords="offset points", ha="left" if side > 0 else "right",
               va="center", fontsize=7, color=style.INK, zorder=5, path_effects=[withStroke(linewidth=2.5, foreground="white")])


# ---- geometry, in inches; y is measured down from the top ----
W = style.WIDTH_IN - 0.16             # saving adds 0.08 in on each side, so the PNG is exactly the text width
FS, FS_S = 7.5, 6.8
x_text, xa0, xa1 = 0.05, 1.05, 4.0    # labels start here; panel A's axis from and to
p_right = [4.88, 5.74, W - 0.04]      # right edges of the three columns of p values
ROW, STRIP = 0.52, 0.03               # panel A: pitch of the rows; half the height of a strip
ya = 1.42                             # top of panel A's axis
ha = len(runs) * ROW
yb_head = ya + ha + 0.68
yb, hb = yb_head + 0.40, 2.3          # top and height of panel B's axis
xb0, xb1 = 0.40, W - 0.04
PITCH_H, GAP, PAD = 0.20, 0.36, 0.40  # panel B: pitch of the hero films; gap between blocks; room for names, top and bottom
y_leg = yb + hb + 0.66                # top of the legend
H = y_leg + 3 * 0.155 + 0.04

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])       # the whole canvas, in inches, for the text around the two plots
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
style.title(fig, TITLE)
fig.text(0.01, 1 - (0.015 * H + 0.20) / H, SUBTITLE, ha="left", va="top", fontsize=7.5, color=style.INK_2, linespacing=1.3)
for y, letter, text in ((0.86, "A", PANEL_A), (yb_head, "B", PANEL_B.format(model=runs[0][0].split("\n")[0]))):
    ax.text(x_text, y, letter, fontsize=9, fontweight="bold", va="center")
    ax.text(x_text + 0.18, y, text, fontsize=7.6, fontweight="bold", va="center")

# ---- A: the canon against the random sets, model by model ----
pa = fig.add_axes([xa0 / W, 1 - (ya + ha) / H, (xa1 - xa0) / W, ha / H], facecolor="none")
pa.set_ylim(ha, 0)
pa.set_yticks([])
pa.spines["left"].set_visible(False)
pa.tick_params(axis="x", length=0, pad=4)
pa.grid(axis="x", color=style.RULE, linewidth=0.6, zorder=0)
pa.axvline(0, color=GREY, linewidth=0.8, zorder=1)
pa.xaxis.set_major_locator(MaxNLocator(6))
ax.text((xa0 + xa1) / 2, ya + ha + 0.31, X_LABEL, ha="center", va="center", fontsize=7, color=style.INK_2)
for x, (_, head) in zip(p_right, P_COLUMNS):
    ax.text(x, ya - 0.06, head, ha="right", va="bottom", fontsize=FS_S, color=style.INK_2, linespacing=1.2)
drawn = [0.0]
for r, (name, colours, s, criterion) in enumerate(runs):
    y = r * ROW + 0.13                # the lane of the marks; the two strips run under it
    if r:
        ax.plot([x_text, W - 0.04], [ya + r * ROW] * 2, color=style.RULE, linewidth=0.5, zorder=0)
    ax.text(x_text, ya + y + 0.06, name, va="center", fontsize=7.2, linespacing=1.15)
    canon = s[CONTROLS[0][0]]["mean_drop_canon"]
    hero = (s.get("specificity") or {}).get("mean_drop_hero_films_own_hero")
    for k, (key, colour, *_) in enumerate(CONTROLS):
        sets = s[key]["mean_drop_of_each_set"]
        pa.vlines(sets, y + 0.125 + 0.075 * k - STRIP, y + 0.125 + 0.075 * k + STRIP, color=colour, linewidth=0.5, alpha=0.7)
        drawn += sets
    if hero is not None:              # joined to the canon's mark, each number on the outer side
        pa.plot([hero, canon], [y, y], color=GREY, linewidth=1.0, zorder=2.5, solid_capstyle="butt")
        pa.plot(hero, y, ls="", ms=5.6, mew=1.1, zorder=3, **HERO)
        value(pa, hero, y, num(hero), side=-1 if hero < canon else 1)
    pa.plot(canon, y, ls="", marker="o", ms=7, mew=1.2, zorder=4, **colours)
    value(pa, canon, y, num(canon), side=-1 if hero is not None and canon <= hero else 1)
    drawn += [canon] + [hero] * (hero is not None)
    ps = [criterion.get(key) for key, _ in P_COLUMNS]
    for x, p in zip(p_right, ps):
        ax.text(x, ya + y, fmt_p(p), ha="right", va="center", fontsize=7, color=style.INK)
    print(f"  A, {name}: canon {num(canon)}, hero films {hero if hero is None else num(hero)}, p", ", ".join(map(fmt_p, ps)))
room = 0.17 * (max(drawn) - min(drawn))   # for a number beside the outermost mark
pa.set_xlim(min(drawn) - room, max(drawn) + room)

# ---- B: the first model, work by work ----
name, colours, s, _ = runs[0]
works = s["by_work"]
heroes = sorted((s.get("specificity") or {}).get("drop_by_hero_film", {}), key=lambda w: -works[w]["drop_canon"])
batman = sorted((w for w in works if w not in heroes), key=lambda w: -works[w]["drop_canon"])
pitch = (xb1 - xb0 - GAP - PITCH_H * len(heroes)) / len(batman)
at = {w: (i + 0.5) * pitch for i, w in enumerate(batman)}                 # x in inches from the axis' left edge
at.update({w: xb1 - xb0 - (len(heroes) - j - 0.5) * PITCH_H for j, w in enumerate(heroes)})
ys = [0.0] + [cell[k] for cell in works.values() for k in ("drop_canon", "drop_random_mean")]
per_inch = (max(ys) - min(ys)) / (hb - 2 * PAD)                           # log-odds per inch of height
bx = fig.add_axes([xb0 / W, 1 - (yb + hb) / H, (xb1 - xb0) / W, hb / H], facecolor="none")
bx.set_xlim(0, xb1 - xb0)
bx.set_ylim(min(ys) - PAD * per_inch, max(ys) + PAD * per_inch)
bx.spines[:].set_visible(False)
bx.tick_params(length=0, pad=3)
bx.yaxis.set_major_locator(MaxNLocator(7, integer=True))
bx.grid(axis="y", color=style.RULE, linewidth=0.6, zorder=0)
bx.axhline(0, color=GREY, linewidth=0.8, zorder=1)
bx.set_xticks([at[w] for w in heroes], heroes, rotation=40, ha="right", rotation_mode="anchor", fontsize=FS_S, color=style.INK_2)
for w, cell in works.items():
    mark = dict(ms=4.6, mew=1.0, **HERO) if w in heroes else dict(marker="o", ms=5.0, mew=1.0, **colours)
    bx.plot([at[w] - 0.045, at[w] + 0.045], [cell["drop_random_mean"]] * 2, color=style.INK_3, linewidth=1.1,
            solid_capstyle="butt", zorder=2)
    bx.plot(at[w], cell["drop_canon"], ls="", zorder=3, **mark)
    print(f"  B, {w}: {num(cell['drop_canon'])} (random sets {num(cell['drop_random_mean'])})")


def stair(block, side):
    """Name three neighbouring dots of a sorted block, the one nearest its middle first. side=1 writes above and to
    the right of each dot, side=-1 below and to the left: the side where the sorted block has no other dot. A name
    that would touch the one before it is moved one line further out."""
    level = None
    for w in block:
        near = side * works[w]["drop_canon"] + 0.05 * per_inch
        level = near if level is None else max(near, level + 0.105 * per_inch)
        bx.text(at[w] - side * 0.035, side * level, w, ha="left" if side > 0 else "right", va="bottom" if side > 0 else "top",
                fontsize=FS_S, color=style.INK_2, zorder=4, path_effects=[withStroke(linewidth=2, foreground="white")])


stair(batman[2::-1], 1)    # the three largest drops
stair(batman[-3:], -1)     # the three smallest
caption = dict(va="bottom", fontsize=FS_S, color=style.INK_2, linespacing=1.2)
ax.text(xb0 + len(batman) * pitch / 2, yb - 0.04, CAPTION_BATMAN.format(n=len(batman)), ha="center", **caption)
if heroes:
    ax.text(xb1, yb - 0.04, CAPTION_HEROES.format(n=len(heroes)), ha="right", **caption)

# ---- legend ----
n_sets = [s[key]["n_sets"] for key, *_ in CONTROLS]
handles = [Line2D([], [], ls="", marker="o", ms=6, mfc=style.INK_2, mec=style.INK_2, label="Canon switched off, Batman works"),
           Line2D([], [], ls="", ms=5, mew=1.1, label="Canon switched off, other hero films", **HERO),
           Line2D([], [], color=style.INK_3, linewidth=1.1, label="B: mean over the random sets, for each work")]
handles += [Line2D([], [], ls="", marker="|", ms=7, mew=1.0, color=colour, label="A: " + text.format(n=n))
            for (_, colour, text, *_), n in zip(CONTROLS, n_sets)]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x_text / W, 1 - y_leg / H), ncol=2, fontsize=7,
           labelcolor=style.INK_2, handlelength=1.2, handletextpad=0.5, labelspacing=0.45, columnspacing=2.0,
           borderpad=0, borderaxespad=0)
style.save(fig, "fig_ablation.png")
