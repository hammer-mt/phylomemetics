"""Figure: Batman's relatives across culture.

A  Share of Batman's 56 core memes that each of the 13 relatives carries (original texts, names present).
B  Which of six orphan-vigilante memes each relative carries, with Batman's own count out of 32 works.

Data: results/canon.json (per_cousin, per_work, memes[].present), results/labels.json, data/corpus.jsonl.

    .venv/bin/python paper/figs/culture.py
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

BATMAN_YEAR = 1939
canon = json.loads((ROOT / "results/canon.json").read_text())
labels = json.loads((ROOT / "results/labels.json").read_text())
corpus = {r["id"]: r for r in map(json.loads, (ROOT / "data/corpus.jsonl").read_text().splitlines())}

# ---- relatives, in the order of the version summarised ----
cousins = sorted(canon["per_cousin"], key=lambda c: c["year"])
assert len(cousins) == 13
# row names follow the text that was read (A Study in Scarlet, The Mark of Zorro), not the corpus short label
NAME = {32: "Puss in Boots", 33: "Count of Monte Cristo", 34: "A Study in Scarlet", 35: "Dracula",
        36: "The Scarlet Pimpernel", 38: "The Bat", 37: "The Mark of Zorro", 40: "Superman", 39: "The Shadow",
        41: "Spider-Man", 42: "Daredevil", 43: "Iron Man", 44: "The Green Hornet"}
FAMILY = {36, 37, 43, "batman"}  # the "billionaire vigilante orphan" family
for c in cousins:
    assert corpus[c["id"]]["group"] == "cousin" and corpus[c["id"]]["year"] == c["year"]

# rows: relatives before 1939, Batman's own works as a reference row, then relatives after
before = [c for c in cousins if c["year"] < BATMAN_YEAR]
after = [c for c in cousins if c["year"] >= BATMAN_YEAR]
rows = [("cousin", c) for c in before] + [("batman", None)] + [("cousin", c) for c in after]
N = len(rows)
YB = len(before)  # Batman's row index

bat_works = canon["per_work"]
assert len(bat_works) == 32
bat_shares = [w["core_share"] for w in bat_works]
bat_median = statistics.median(bat_shares)
n_core = canon["n_core"]

# ---- the six orphan-vigilante memes, ordered by how many Batman works carry them ----
UNITS = {131948: "Origin\nwound", 168352: "Secret\nidentity", 128514: "Heir to a\ndead father's\nfortune",
         50224: "Billionaire\nvigilante", 520801: "Butler\nwho raised\nhim", 878889: "Taking in\nan orphan\nas a ward"}
memes = {m["unit"]: m for m in canon["memes"]}
cols = []
for u, short in UNITS.items():
    m = memes[u]
    pres = set(m["present"])
    cols.append({"unit": u, "short": short, "label": labels[str(u)]["label"], "pres": pres,
                 "n_bat": sum(w["id"] in pres for w in bat_works)})
    assert cols[-1]["n_bat"] == m["n_batman"]
cols.sort(key=lambda c: -c["n_bat"])

# ---- layout (inches) ----
W, H = style.WIDTH_IN - 0.16, 4.55  # tight bbox adds 0.08 in each side, so the PNG is exactly text width
ROW = 0.205
top_axes = H - 1.13
ax_h = N * ROW
left_a, w_a = 1.5, 1.4
left_b = left_a + w_a + 0.2
w_b = W - left_b - 0.02
fig = plt.figure(figsize=(W, H))
ax_a = fig.add_axes([left_a / W, (top_axes - ax_h) / H, w_a / W, ax_h / H])
ax_b = fig.add_axes([left_b / W, (top_axes - ax_h) / H, w_b / W, ax_h / H])

style.title(fig, "Batman's relatives across culture",
            sub="Qwen3-8B, one Wikipedia plot summary per relative, read with names present. "
                "Rows run by the year of the version summarised.")

BAND = "#eef2f8"
BEFORE_C, AFTER_C = style.SHARED, style.NIGHT
FS = 7  # body labels; nothing below 6.5 pt


def row_color(c):
    return BEFORE_C if c["year"] < BATMAN_YEAR else AFTER_C


def row_name(kind, c):
    if kind == "batman":
        return "Batman, 32 works (1943–2022)"
    return f"{NAME[c['id']]} ({c['year']})"


# ---- panel (a): share of the core canon ----
for ax in (ax_a, ax_b):
    ax.set_ylim(N - 0.5, -0.5)
    ax.axhspan(YB - 0.5, YB + 0.5, color=BAND, zorder=0, lw=0)
ax_a.set_xlim(0, 1.0)
for v in (0, 0.25, 0.5, 0.75, 1.0):
    ax_a.axvline(v, color=style.RULE, lw=0.6, zorder=0.5)
ax_a.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
ax_a.tick_params(axis="x", length=0, pad=3, labelsize=FS)
ax_a.spines[["left", "bottom", "top", "right"]].set_visible(False)
ax_a.set_yticks(range(N), [row_name(k, c) for k, c in rows])
ax_a.tick_params(axis="y", length=0, pad=10, labelsize=FS, labelcolor=style.INK)
for i, (kind, c) in enumerate(rows):
    fam = (c["id"] if c else "batman") in FAMILY
    lab = ax_a.get_yticklabels()[i]
    if fam:
        lab.set_fontweight("bold")
        ax_a.plot([-0.035], [i], marker="D", ms=3.6, color=style.SIGNAL, mec="white", mew=0.5,
                  transform=ax_a.get_yaxis_transform(), clip_on=False, zorder=5)
    if kind == "batman":
        ax_a.plot([min(bat_shares), max(bat_shares)], [i, i], color=style.NIGHT_SOFT, lw=1.6, zorder=2,
                  solid_capstyle="round")
        ax_a.scatter(bat_shares, [i] * len(bat_shares), s=13, color=style.NIGHT, alpha=0.5, lw=0, zorder=3)
        ax_a.text(min(bat_shares) - 0.03, i, f"median {bat_median:.0%}", ha="right", va="center",
                  fontsize=FS, color=style.INK_2)
        continue
    v = c["core_share"]
    ax_a.barh(i, max(v, 0.006), height=0.56, color=row_color(c), lw=0, zorder=2)
    ax_a.text(v + 0.025, i, f"{v:.0%}", ha="left", va="center", fontsize=FS, color=style.INK_2)

# ---- panel (b): the six memes; a filled cell takes its row's colour, so colour always means date ----
ax_b.set_xlim(-0.5, len(cols) - 0.5)
ax_b.axis("off")
CW, CH = 0.58, 0.64
for i, (kind, c) in enumerate(rows):
    for j, col in enumerate(cols):
        if kind == "batman":
            ax_b.text(j, i, f"{col['n_bat']}/32", ha="center", va="center", fontsize=7.5, fontweight="bold",
                      color=style.INK)
            continue
        on = c["id"] in col["pres"]
        ax_b.add_patch(FancyBboxPatch((j - CW / 2, i - CH / 2), CW, CH,
                                      boxstyle="round,pad=0,rounding_size=0.07", mutation_aspect=1,
                                      fc=row_color(c) if on else style.OFF, lw=0, zorder=2))
for j, col in enumerate(cols):
    ax_b.text(j, -0.85, col["short"], ha="center", va="bottom", fontsize=6.6, color=style.INK,
              linespacing=1.05, clip_on=False)

# ---- panel titles ----
y_pt = (top_axes + 0.56) / H
fig.text(0.01, y_pt, f"A  Share of Batman's {n_core} core memes each carries", ha="left",
         va="bottom", fontsize=8, fontweight="bold", color=style.INK)
fig.text((left_b + 0.08) / W, y_pt, "B  Six orphan-vigilante memes, present or absent", ha="left", va="bottom",
         fontsize=8, fontweight="bold", color=style.INK)

# ---- legends, below the plots ----
y1 = (top_axes - ax_h - 0.36) / H
y2 = y1 - 0.17 / H


RENDERER = fig.canvas.get_renderer()


def swatch(x_in, y, color, text, kind="rect"):
    """Draw one legend key at x_in (inches) and return the x (inches) where the next key may start."""
    xf = x_in / W
    if kind == "rect":
        fig.add_artist(Rectangle((xf, y - 0.04 / H), 0.15 / W, 0.085 / H, transform=fig.transFigure,
                                 fc=color, lw=0))
        tx = x_in + 0.2
    elif kind == "diamond":
        fig.add_artist(plt.Line2D([xf + 0.04 / W], [y], marker="D", ms=3.6, color=color, mec="white",
                                  mew=0.5, transform=fig.transFigure))
        tx = x_in + 0.12
    elif kind == "strip":
        fig.add_artist(plt.Line2D([xf, xf + 0.16 / W], [y, y], color=style.NIGHT_SOFT, lw=1.6,
                                  solid_capstyle="round", transform=fig.transFigure))
        for dx in (0.02, 0.07, 0.13):
            fig.add_artist(plt.Line2D([xf + dx / W], [y], marker="o", ms=3, color=style.NIGHT, alpha=0.6,
                                      lw=0, transform=fig.transFigure))
        tx = x_in + 0.21
    t = fig.text(tx / W, y, text, ha="left", va="center", fontsize=FS, color=style.INK_2)
    return tx + t.get_window_extent(RENDERER).width / fig.dpi + 0.28


x = swatch(0.06, y1, BEFORE_C, f"Before Batman ({BATMAN_YEAR})")
x = swatch(x, y1, AFTER_C, f"{BATMAN_YEAR} or later")
x = swatch(x, y1, None, "One dot per Batman work", kind="strip")
x = swatch(x, y1, style.SIGNAL, "Bold: the \u201cbillionaire vigilante orphan\u201d family", kind="diamond")
assert x - 0.28 <= W, f"legend line 1 runs to {x - 0.28:.2f} in, wider than the figure"
x = swatch(0.06, y2, style.OFF, "Pale cell: absent. Filled cell: present, meaning the summary scores above all "
                                "but one of the 69 ordinary films.")
assert x - 0.28 <= W, f"legend line 2 runs to {x - 0.28:.2f} in"

style.save(fig, "fig_culture.png")

# ---- numbers for the caption ----
print(f"Batman works core share: min {min(bat_shares):.3f}, median {bat_median:.3f}, max {max(bat_shares):.3f}")
for kind, c in rows:
    if kind == "batman":
        print("  Batman", [(col["label"], col["n_bat"]) for col in cols])
        continue
    have = [col["label"] for col in cols if c["id"] in col["pres"]]
    print(f"  {c['year']} {NAME[c['id']]:26s} core {c['core_share']:.3f} ({round(c['core_share'] * n_core)} of {n_core}) "
          f"specific {c['core_specific_share']:.3f}  six: {have}")
