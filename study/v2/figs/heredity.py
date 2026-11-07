"""Figure for paper v2: heredity (experiment E2) and the residual test.

fig_heredity.png
  A  One row per screen adaptation that has a documented source comic. The grey slots are the comics eligible to
     be its source, ranked by how many of the comic's optional units the adaptation carries (1st = closest). Each
     model's mark sits in the slot of the documented source. A grey diamond gives the same rank from TF-IDF word
     counts. Marks that share a slot sit side by side.
  B  The share of eligible comics that beat the documented source, averaged over the adaptations (0 = the source
     is always closest), with its exact p: the units of each model, two word-count measures, and the units after
     everything word counts explain is regressed out (the decision test). The dashed line is chance.

Reads results/v2/experiments_v2.json (another folder: RESULTS_DIR=...). Every number drawn is read from that file
at run time. Models that are not in the file yet are skipped, so the script is simply re-run when they arrive.

    cd study && ../.venv/bin/python v2/figs/heredity.py
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
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

TITLE = "Where the documented source comic of each screen adaptation ranks"
SUBTITLE = "Eligible comics are ranked by how many of their optional units the adaptation carries, or by word counts alone"
PANEL_A = "Rank of the documented source among the eligible comics (the grey slots; 1st = closest)"
PANEL_B = "Share of eligible comics that beat the documented source, averaged over the adaptations"
X_LABEL = "Share of eligible comics that beat the source (0 = the source is always closest; lower is better)"
ROWS_B = ["Units", "Word counts: TF-IDF", "Word counts: word presence", "Units, after word counts\nare regressed out"]

# tag in the result file, display name, mark. The primary model is the solid dark dot; the others are quieter.
# The marks are about 5 pt across, so that four of them fit side by side in one slot of panel A without touching.
MODELS = [
    ("qwen3-32b", "Qwen3-32B", dict(marker="o", ms=5.3, mfc=style.NIGHT, mec="white", mew=0.3)),
    ("qwen3-8b", "Qwen3-8B", dict(marker="o", ms=3.9, mfc="white", mec=style.NIGHT, mew=1.0)),
    ("llama3.1-8b", "Llama 3.1 8B", dict(marker="o", ms=4.9, mfc=style.SHARED, mec="white", mew=0.3)),
    ("llama3.3-70b", "Llama 3.3 70B", dict(marker="o", ms=3.9, mfc="white", mec=style.SHARED, mew=1.0)),
]
TFIDF = dict(marker="D", ms=4.0, mfc=style.INK_3, mec="white", mew=0.5)     # word counts are not a model: diamonds
PRESENCE = dict(marker="D", ms=3.2, mfc="white", mec=style.INK_3, mew=0.9)
# titles spelled as in version 1's figure
NAMES = {"Dark Knight Rises": "The Dark Knight Rises", "Dark Knight Returns": "The Dark Knight Returns",
         "Batman '89": "Batman (1989)", "The Batman": "The Batman (2022)"}
X_MAX = 0.6          # right end of panel B's axis, extended if a share is larger

RESULTS = Path(os.environ.get("RESULTS_DIR", STUDY / "results" / "v2"))
d = json.loads((RESULTS / "experiments_v2.json").read_text())
models = [m for m in MODELS if "E2" in d.get(m[0], {})]
words = d["words"]["E2"]
print("models in the file:", ", ".join(m[0] for m in models) or "none")


def summary(block, what):
    """Share, exact p and chance mean of one E2 result, and a line saying where they were read."""
    if "summary" in block:
        s, how = block["summary"], "its summary"
    elif "chance" in block["primary"]:  # the fields experiments_v2.py copies into a model's summary
        ch = block["primary"]["chance"]
        s, how = {"share": ch["observed_unrounded"], "exact_p": ch["joint_relabelling_exact"]["p"],
                  "chance_mean": ch["joint_relabelling_exact"]["mean"]}, "primary.chance (it has no summary)"
    else:
        rows = block["primary"]["rows"]
        s, how = {"share": sum(r["share_of_comics_beating_source"] for r in rows) / len(rows)}, "the mean of its rows"
    print(f"  {what}: share {s['share']:.4f}, exact p {s.get('exact_p')}, read from {how}")
    return s


def fmt_p(p):
    """Two significant figures, never in scientific notation."""
    return "p = " + (f"{p:.{max(2, 1 - floor(log10(p)))}f}" if p > 0 else "0")


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# ---- what is drawn ----
base = words["tfidf_cosine"]["primary"]["rows"]      # row order, sources and slot counts
ranks = [(kw, {r["target"]: r for r in d[tag]["E2"]["primary"]["rows"]}) for tag, _, kw in models]
groups = [
    [(kw, summary(d[tag]["E2"], f"units, {tag}")) for tag, _, kw in models],
    [(TFIDF, summary(words["tfidf_cosine"], "words, tfidf_cosine"))],
    [(PRESENCE, summary(words["word_presence"], "words, word_presence"))],
    [(kw, summary({"summary": d[tag]["E2"]["residual_summary"]["decision_words_by_frequency"]},
                  f"units after word counts are regressed out (the decision test), {tag}")) for tag, _, kw in models],
]
stats = [s for g in groups for _, s in g]
chances = [s["chance_mean"] for s in stats if "chance_mean" in s]
chance = sum(chances) / len(chances)
print(f"chance line at {chance:.4f}: mean of {len(chances)} chance means, from {min(chances):.4f} to {max(chances):.4f}")

# ---- geometry, in inches; y is measured down from the top ----
W = style.WIDTH_IN - 0.16            # saving adds 0.08 in on each side, so the PNG is exactly the text width
FS, FS_S = 7.5, 6.8
row_h, sub_h, gap = 0.31, 0.17, 0.11
x_text, sx0 = 0.05, 2.16             # labels start here; slots and panel B's axis start at sx0
n_slots = max(r["n_comics"] for r in base)
pitch = (W - 0.03 - sx0) / n_slots
cell_w, cell_h = pitch - 0.045, 0.215
slot_x = lambda rank: sx0 + (rank - 1) * pitch + cell_w / 2  # noqa: E731
ya = 1.09                            # centre of panel A's first row
yb_head = ya + (len(base) - 0.5) * row_h + 0.30
yb = yb_head + 0.36                  # top of panel B's axis
hb = sum(len(g) * sub_h + gap for g in groups if g) - gap + 0.08
H = yb + hb + 0.78

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])      # the whole canvas, in inches
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
style.title(fig, TITLE)              # the subtitle sits a fixed distance under it, as in the other tall figures
fig.text(0.01, 1 - 0.30 / H, SUBTITLE, ha="left", va="top", fontsize=7.5, color=style.INK_2)

# ---- A: the rank of the documented source, adaptation by adaptation ----
ax.text(x_text, ya - 0.43, "A", fontsize=9, fontweight="bold", va="center")
ax.text(x_text + 0.18, ya - 0.43, PANEL_A, fontsize=7.6, fontweight="bold", va="center")
for n in (1, n_slots):
    ax.text(slot_x(n), ya - 0.20, ordinal(n), ha="center", va="center", fontsize=FS_S, color=style.INK_3)
for k, r in enumerate(base):
    y = ya + k * row_h
    ax.text(x_text, y - 0.055, NAMES.get(r["target"], r["target"]), va="center", fontsize=8)
    ax.text(x_text, y + 0.082, "from " + " / ".join(NAMES.get(s, s) for s in r["sources"]), va="center",
            fontsize=FS_S, color=style.INK_3)
    for j in range(r["n_comics"]):
        ax.add_patch(FancyBboxPatch((sx0 + j * pitch, y - cell_h / 2), cell_w, cell_h, fc=style.OFF, ec="none",
                                    boxstyle="round,pad=0,rounding_size=0.03"))
    in_slot = {}                     # models that share a slot sit side by side, in the order of the legend
    for kw, by_target in ranks:
        assert by_target[r["target"]]["n_comics"] == r["n_comics"], r["target"]
        in_slot.setdefault(by_target[r["target"]]["rank"], []).append(kw)
    for rank, kws in in_slot.items():
        step = min(0.085, (cell_w - 0.085) / max(len(kws) - 1, 1))
        for i, kw in enumerate(kws):
            ax.plot(slot_x(rank) + (i - (len(kws) - 1) / 2) * step, y - 0.045, ls="", zorder=3, **kw)
    ax.plot(slot_x(r["rank"]), y + 0.055, ls="", zorder=3, **TFIDF)   # word counts: the slot's lower lane

# ---- B: the share of eligible comics that beat the source ----
ax.text(x_text, yb_head, "B", fontsize=9, fontweight="bold", va="center")
ax.text(x_text + 0.18, yb_head, PANEL_B, fontsize=7.6, fontweight="bold", va="center")
x_max = max(X_MAX, ceil(max(s["share"] for s in stats) * 10) / 10)
bx = fig.add_axes([sx0 / W, 1 - (yb + hb) / H, 3.4 / W, hb / H], facecolor="none")
bx.set_xlim(0, x_max)
bx.set_ylim(hb, 0)
bx.xaxis.set_major_locator(MultipleLocator(0.1))
bx.xaxis.set_major_formatter("{x:.1f}")
bx.set_yticks([])
bx.spines["left"].set_visible(False)
bx.tick_params(axis="x", length=0, pad=4)
bx.grid(axis="x", color=style.RULE, linewidth=0.6, zorder=0)
bx.set_xlabel(X_LABEL, labelpad=4, fontsize=7)
bx.axvline(chance, color=style.INK_3, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
bx.annotate("chance", (chance, 0), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
            fontsize=7, color=style.INK_3)
y = 0.04
for label, items in zip(ROWS_B, groups):
    if not items:
        continue
    if y > 0.04:
        ax.plot([x_text, W - 0.03], [yb + y - gap / 2] * 2, color=style.RULE, linewidth=0.5, zorder=0)
    ax.text(x_text, yb + y + len(items) * sub_h / 2, label, va="center", fontsize=FS, linespacing=1.25)
    for kw, s in items:
        yc = y + sub_h / 2
        bx.plot(s["share"], yc, ls="", clip_on=False, zorder=3, **kw)
        note = dict(xy=(s["share"], yc), textcoords="offset points", va="center", zorder=2,
                    bbox=dict(fc="white", ec="none", pad=0.6))
        bx.annotate(f"{s['share']:.2f}", xytext=(7, 0), fontsize=7, color=style.INK, **note)
        if "exact_p" in s:
            bx.annotate(fmt_p(s["exact_p"]), xytext=(26, 0), fontsize=FS_S, color=style.INK_3, **note)
        y += sub_h
    y += gap

# ---- legend ----
handles = [Line2D([], [], ls="", label=name, **kw) for _, name, kw in models]
handles.append(Line2D([], [], ls="", label=ROWS_B[1], **TFIDF))
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=len(handles), fontsize=7,
           labelcolor=style.INK_2, handletextpad=0.2, columnspacing=1.5, borderaxespad=0.3)
style.save(fig, "fig_heredity.png")
