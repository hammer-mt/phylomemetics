"""Figure for paper v2: the explanation tests (can the explanations lose?).

For each unit chosen by rule, an AI explainer wrote a description from the unit's strongest examples. A tester who
saw only the description then wrote, before anything was measured, new texts that should make the unit fire (A),
texts that should make it fire while using none of the description's key words (B), near misses that should not
(C) and texts on a distant subject (D). A text's response is strong when the unit reaches the threshold share of
its scale ("theta" in the results).

fig_explanations.png
  A  The share of each kind of text that is strong, one mark per model. The grey diamonds are what the essay
     reports for its own gallery of Qwen3-32B neurons (it has no figure for B). The grey dash under a model's
     mark is the floor: how often its units are strong on the A and B texts written for other units.
  B  How many explanations win (more of the A texts than of the C texts are strong), tie and lose, as shares of a
     bar with the counts beside it: every tested unit of each model, then its idea units and its name or word
     units, and last the essay's gallery. Units not testable out of context get a segment and a column if any.
  C  The mean AUC for telling the B texts from the C texts by the unit itself, by the keyword analyst's search
     terms and by TF-IDF similarity to the unit's examples.

Reads results/v2/<model>/explanation_tests.json for every model that has one (another folder: RESULTS_DIR=...) and
skips the others. Every number drawn is read from those files at run time. Writes fig_explanations.png to
paper/v2/ (or $PAPER_DIR).

    cd study && ../.venv/bin/python v2/figs/explanations.py
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
from matplotlib.patches import Rectangle  # noqa: E402

TITLE = "Testing the descriptions of the units on newly written texts"
SUBTITLE = ("An AI explainer described each unit from its strongest examples. A tester who saw only the description\n"
            "then wrote the test texts, before anything was measured.")
PANEL_A = "Share of test texts that make the unit fire strongly"
STRONG = "(the unit reaches at least {theta:.0%} of its scale)"
PANEL_B = ["Explanations that win, tie and lose", "A win: more new examples than near misses are strong"]
PANEL_C = ["Telling the idea without the words", "from the words without the idea"]
X_LABEL_C = "Mean AUC over the units: texts with the idea but\nnone of the key words, against the near misses"
ESSAY_BAR, ESSAY_KEY = "Essay's gallery", "The essay's own gallery of Qwen3-32B neurons"
FLOOR_KEY = "The same units on the texts written for other units"

GREY = style.SHARED
MODELS = [  # tag of the results folder, display name, marker face, marker edge; the primary model is the solid one
    ("qwen3-32b", "Qwen3-32B", style.NIGHT, style.NIGHT),
    ("qwen3-8b", "Qwen3-8B", "white", style.NIGHT),
    ("llama3.1-8b", "Llama 3.1 8B", GREY, GREY),
    ("llama3.3-70b", "Llama 3.3 70B", "white", GREY),
]
# the kinds of test text: key in the results, label, key of the essay's figure for it (it reports none for B)
TYPES = [("A", "New examples", "new_examples"), ("B", "The idea without\nthe key words", None),
         ("C", "Near misses", "near_misses"), ("D", "Distant subject", "distant")]
SHOULD = [("Texts that should make the unit fire", "AB"), ("Texts that should not", "CD")]
FLOOR = "strong_rate_on_other_units_{}_texts"   # in mismatched_null: there is one for A and one for B
OUTCOMES = [("won", "won", style.NIGHT), ("tied", "tied", style.NIGHT_SOFT), ("lost", "lost", GREY)]
UNTESTABLE = ("not_testable_out_of_context", "not\ntestable", style.OFF)
SUBSETS = [("all", None), ("idea_units", "idea units"), ("name_units", "name or word units")]
DIAMOND = dict(marker="D", mec=style.INK_3, mew=1.0)  # the essay and the word measures are not models: diamonds
MEASURES = [("neuron", "The unit itself", None), ("analyst_keywords", "Keyword search terms", {"mfc": style.INK_3}),
            ("tfidf_to_examples", "TF-IDF to its examples", {"mfc": "white"})]
CHANCE = 0.5  # an AUC of one half is chance by definition; it is not a result

RESULTS = Path(os.environ.get("RESULTS_DIR") or STUDY / "results" / "v2")
data = {tag: json.loads(path.read_text()) for tag, *_ in MODELS
        if (path := RESULTS / tag / "explanation_tests.json").exists()}
models = [m for m in MODELS if m[0] in data]
print("models with explanation tests:", ", ".join(data) or "none")
if not models:
    sys.exit(f"no <model>/explanation_tests.json under {RESULTS}")
essay, theta = data[models[0][0]]["essay_benchmark"], data[models[0][0]]["theta"]
assert all(d["essay_benchmark"] == essay and d["theta"] == theta for d in data.values()), "the models' files disagree"
columns = OUTCOMES + [UNTESTABLE] * any(d[s].get(UNTESTABLE[0]) for d in data.values() for s, _ in SUBSETS)

# ---- geometry, in inches; y is measured down from the top ----
W = style.WIDTH_IN - 0.16             # saving adds 0.08 in on each side, so the PNG is exactly the text width
n = len(models)
FS, FS_S, X_TEXT = 7, 6.8, 0.05
AX0, YA, HA = 0.42, 1.34, 1.7         # panel A's plot: left, top, height
WA = W - 0.03 - AX0
cat, slot = WA / len(TYPES), min(0.30, 1.30 / (n + 1))  # width of a kind of text; of one model's lane in it
PITCH, GAP = 0.165, 0.11              # rows of panels B and C; extra between models
BLOCK = len(SUBSETS) * PITCH + GAP
YH = YA + HA + 0.66                   # heads of panels B and C
Y0 = YH + 0.66                        # centre of their first row
edge = lambda k: Y0 + k * BLOCK - (PITCH + GAP) / 2  # noqa: E731  the line above model k's rows
col_x = [3.52 - 0.5 * (len(columns) > len(OUTCOMES)) - 0.31 * (len(OUTCOMES) - 1 - i) for i in range(len(OUTCOMES))]
col_x += [3.52] * (len(columns) > len(OUTCOMES))  # right edges of the columns of counts
BX0, BX1 = 1.03, col_x[0] - 0.30      # the bars of panel B
CX0, CX1, CHEAD = 4.94, 6.17, 3.72    # panel C's axis, and where its head and labels start
H = Y0 + n * BLOCK + 0.86

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])       # the whole canvas, in inches
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
style.title(fig, TITLE)               # the subtitle sits a fixed distance under it, as in the other tall figures
fig.text(0.01, 1 - 0.30 / H, SUBTITLE, ha="left", va="top", fontsize=7.5, color=style.INK_2)


def head(x, y, letter, text, second=None, **kw):
    """A panel's letter and bold head; a second line under it is bold too unless kw says otherwise."""
    ax.text(x, y, letter, fontsize=9, fontweight="bold", va="center")
    if second:
        ax.text(x + 0.18, y + 0.15, second, va="center", **{"fontsize": 7.6, "fontweight": "bold", **kw})
    return ax.text(x + 0.18, y, text, fontsize=7.6, fontweight="bold", va="center")


def inset(x0, y0, x1, y1):
    """Axes between two corners given in inches from the top left of the figure, without a frame."""
    new = fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H], facecolor="none")
    new.spines[:].set_visible(False)
    new.tick_params(length=0, pad=4)
    return new


# ---- A: the share of each kind of text that is strong ----
t = head(X_TEXT, YA - 0.54, "A", PANEL_A)
ax.annotate(STRONG.format(theta=theta), (1, 0.5), xycoords=t, xytext=(5, 0), textcoords="offset points",
            va="center", fontsize=FS, color=style.INK_2)
a = inset(AX0, YA, AX0 + WA, YA + HA)
a.set_xlim(0, WA)                     # in inches, so that the lanes of the models are a fixed width
a.set_ylim(0, 1)
ticks = [0, 0.25, 0.5, 0.75, 1]
a.set_yticks(ticks, [f"{v:.0%}" for v in ticks])
a.grid(axis="y", color=style.RULE, linewidth=0.8)
a.set_axisbelow(True)
centre = {key: (k + 0.5) * cat for k, (key, _, _) in enumerate(TYPES)}
a.set_xticks(list(centre.values()), [label for _, label, _ in TYPES], fontsize=7.5, color=style.INK, linespacing=1.25)
a.tick_params(axis="x", pad=7)
for text, keys in SHOULD:             # a bracket over the kinds of text that share an intention
    x0, x1 = AX0 + centre[keys[0]] - cat / 2 + 0.12, AX0 + centre[keys[-1]] + cat / 2 - 0.12
    ax.plot([x0, x1], [YA - 0.2] * 2, color=style.INK_3, linewidth=0.6)
    ax.text((x0 + x1) / 2, YA - 0.29, text, ha="center", va="center", fontsize=FS, color=style.INK_2)
above = dict(xytext=(0, 5.5), textcoords="offset points", ha="center", va="bottom", fontsize=FS)
for key, _, bench in TYPES:
    for j, (tag, _, face, rim) in enumerate(models):
        x, v = centre[key] + (j - n / 2) * slot, data[tag]["all"]["strong_rate"][key]
        a.plot(x, v, marker="o", ms=6.5, mfc=face, mec=rim, mew=1.3, zorder=4, clip_on=False)
        a.annotate(f"{v:.0%}", (x, v), **above)
        floor = data[tag]["mismatched_null"].get(FLOOR.format(key))
        if floor is not None:         # a short dash with its value at its right, centred under the mark
            a.plot([x - 0.12, x - 0.03], [floor] * 2, color=GREY, linewidth=1.6, solid_capstyle="butt", zorder=3)
            a.annotate(f"{floor:.0%}", (x - 0.03, floor), xytext=(1.5, 0), textcoords="offset points", va="center",
                       fontsize=6.5, color=style.INK_3)
    if bench:                         # the essay's lane is the last one
        a.plot(centre[key] + n / 2 * slot, essay[bench], ms=5.6, mfc=style.INK_3, zorder=4, clip_on=False, **DIAMOND)
        a.annotate(f"{essay[bench]:.0%}", (centre[key] + n / 2 * slot, essay[bench]), color=style.INK_3, **above)

# ---- B: wins, ties and losses; C: the idea against the words ----
head(X_TEXT, YH, "B", PANEL_B[0], PANEL_B[1], fontsize=FS, fontweight="normal", color=style.INK_2)
head(CHEAD, YH, "C", *PANEL_C)
for (_, label, colour), x in zip(columns, col_x):  # each column's head is underlined in its segment's colour
    ax.text(x, Y0 - 0.25, label, ha="right", va="bottom", fontsize=FS_S, color=style.INK_2, linespacing=1.1)
    ax.plot([x - 0.25, x], [Y0 - 0.2] * 2, color=colour, linewidth=2.5, solid_capstyle="butt")
c = inset(CX0, edge(0), CX1, edge(n))
c.set_xlim(0, 1)
c.set_ylim(edge(n), edge(0))
c.set_xticks(ticks, [f"{v:g}" if v in (0, CHANCE, 1) else "" for v in ticks])
c.set_yticks([])
c.grid(axis="x", color=style.RULE, linewidth=0.6)
c.set_axisbelow(True)
c.axvline(CHANCE, color=style.INK_3, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
c.annotate("chance", (CHANCE, edge(0)), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
           fontsize=FS, color=style.INK_3)
ax.text(W - 0.03, edge(n) + 0.21, X_LABEL_C, ha="right", va="top", fontsize=FS, color=style.INK_2, linespacing=1.25)


def bar(y, counts, height, fs, ink):
    """One bar of wins, ties and losses as shares of its length, with the counts in the columns at its right."""
    values = [counts.get(key, 0) for key, _, _ in columns]
    left = BX0
    for (_, _, colour), value, x in zip(columns, values, col_x):
        if value:
            width = (BX1 - BX0) * value / sum(values)
            ax.add_patch(Rectangle((left, y - height / 2), width, height, fc=colour, ec="white", linewidth=0.6))
            left += width
        if sum(values):
            ax.text(x, y, f"{value}", ha="right", va="center", fontsize=fs, color=ink)


for k, (tag, name, face, rim) in enumerate(models):
    y = Y0 + k * BLOCK
    ax.plot([X_TEXT, W - 0.03], [edge(k)] * 2, color=style.RULE, linewidth=0.5, zorder=0)
    for i, (subset, label) in enumerate(SUBSETS):
        main = label is None          # the model's own row; its subsets get thinner bars and smaller labels
        ax.text(X_TEXT, y + i * PITCH, name if main else label, va="center", fontsize=7.5 if main else FS_S,
                color=style.INK if main else style.INK_3)
        bar(y + i * PITCH, data[tag][subset], 0.125 if main else 0.07, FS if main else FS_S,
            style.INK if main else style.INK_3)
    for i, (key, label, look) in enumerate(MEASURES):
        v = data[tag]["all"]["auc_B_vs_C_mean"][key]
        ax.text(CX0 - 0.08, y + i * PITCH, label, ha="right", va="center", fontsize=FS_S, color=style.INK_3)
        c.plot(v, y + i * PITCH, zorder=3, clip_on=False,
               **({**DIAMOND, "ms": 4.6, **look} if look else dict(marker="o", ms=5.5, mfc=face, mec=rim, mew=1.2)))
        c.annotate(f"{v:.2f}", (v, y + i * PITCH), xytext=(7, 0), textcoords="offset points", va="center",
                   fontsize=FS, zorder=2, bbox=dict(fc="white", ec="none", pad=0.6))
ax.plot([X_TEXT, W - 0.03], [edge(n)] * 2, color=style.RULE, linewidth=0.5, zorder=0)
ax.text(X_TEXT, Y0 + n * BLOCK, ESSAY_BAR, va="center", fontsize=7.5)
bar(Y0 + n * BLOCK, essay, 0.125, FS, style.INK)

# ---- legend: the models, then the two grey marks of panel A ----
kw = dict(loc="lower left", fontsize=FS, labelcolor=style.INK_2, handlelength=1.2, handletextpad=0.4,
          columnspacing=1.5, borderaxespad=0.3)
fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=6.5, mfc=face, mec=rim, mew=1.3, label=name)
                    for _, name, face, rim in models], bbox_to_anchor=(0, 0.2 / H), ncol=n, **kw)
fig.legend(handles=[Line2D([], [], ls="", ms=5.6, mfc=style.INK_3, label=ESSAY_KEY, **DIAMOND),
                    Line2D([], [], color=GREY, linewidth=1.6, label=FLOOR_KEY)], bbox_to_anchor=(0, 0), ncol=2, **kw)
style.save(fig, "fig_explanations.png")
