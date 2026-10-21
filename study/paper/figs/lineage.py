"""Figures: the family tree, and how memes enter and spread.

fig_tree.png    The Qwen3-8B neighbour-joining tree of the 32 Batman works and 13 relatives, rooted on Puss in Boots,
                drawn as a rectangular cladogram (branching order only), leaves coloured by era group.
fig_spread.png  One row per named meme: a dot for each Batman work where it is present, placed by year, with the
                first appearance in this sample highlighted; relatives where it is present as light open circles.

Data: results/phylo.json.gz (newick, works), report/data.json (named memes, presence, era groups),
report/data2.js (the companion page's swim rows), results/canon.json (cross-check of presence).

    ../.venv/bin/python paper/figs/lineage.py
"""
import gzip
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from Bio import Phylo  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

FS = 7.0      # body labels (pt); nothing below 6.5
FS_S = 6.7    # small annotations
# bbox_inches="tight" adds 0.08 in each side, so a canvas this wide saves at exactly the text width
CANVAS_W = style.WIDTH_IN - 0.16

phylo = json.loads(gzip.open(ROOT / "results/phylo.json.gz").read())
data = json.loads((ROOT / "report/data.json").read_text())
s = (ROOT / "report/data2.js").read_text()
pm2 = json.loads(s[s.index("{"):s.rstrip().rstrip(";").rindex("}") + 1])
canon = json.loads((ROOT / "results/canon.json").read_text())

works = {w["id"]: w for w in data["works"]}
assert len(works) == 45 and sum(w["group"] == "batman" for w in works.values()) == 32
assert data["trees"]["qwen3-8b"] == phylo["models"]["qwen3-8b"]["newick"]

# Display names: year added once; relatives named after the text that was read.
SHORT = {7: "Arkham Asylum comic", 20: "Arkham Asylum game", 30: "Joker", 38: "The Bat",
         32: "Puss in Boots", 33: "Count of Monte Cristo", 34: "A Study in Scarlet", 35: "Dracula",
         36: "Scarlet Pimpernel", 37: "The Mark of Zorro", 39: "The Shadow", 40: "Superman", 41: "Spider-Man",
         42: "Daredevil", 43: "Iron Man", 44: "The Green Hornet"}


def name(i, year=True):
    n = SHORT.get(i, works[i]["label"])
    return f"{n} ({works[i]['year']})" if year else n


# =====================================================================================
# Figure: the family tree
# =====================================================================================
def tree_figure():
    tree = Phylo.read(io.StringIO(data["trees"]["qwen3-8b"]), "newick")
    root = tree.root
    # rooted on Puss in Boots: it is the root's own leaf child
    assert any(c.is_terminal() and c.name == "w32" for c in root.clades)

    leaves = root.get_terminals()  # newick order, top to bottom
    ids = [int(l.name[1:]) for l in leaves]
    assert sorted(ids) == list(range(45))
    row = {i: k for k, i in enumerate(ids)}

    # cladogram: x = steps above the leaves (leaves aligned on the right), y = mean of children
    def height(c):
        c.h = 0 if c.is_terminal() else 1 + max(height(k) for k in c.clades)
        return c.h

    H = height(root)

    def place(c):
        if c.is_terminal():
            c.y = row[int(c.name[1:])]
        else:
            for k in c.clades:
                place(k)
            c.y = (c.clades[0].y + c.clades[-1].y) / 2

    place(root)

    def clade_of(members):
        """The smallest clade holding exactly these leaves (so a bracket never claims a clade the tree lacks)."""
        ca = tree.common_ancestor([f"w{i}" for i in members])
        got = sorted(int(t.name[1:]) for t in ca.get_terminals())
        assert got == sorted(members), (members, got)
        return [row[i] for i in members]

    n = len(ids)
    row_h = 0.136                     # inches per leaf (tight enough that tree and caption share a page)
    top_in, bot_in = 0.62, 0.62       # title block, legend block
    fig_h = top_in + n * row_h + bot_in
    fig = plt.figure(figsize=(CANVAS_W, fig_h))
    # axes spans the full width in inches; data coords: x in inches, y in rows
    ax = fig.add_axes([0, bot_in / fig_h, 1, n * row_h / fig_h])
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(n - 0.5, -0.5)
    ax.axis("off")

    x0, x1 = 0.10, 2.50               # tree spans these inches
    X = lambda d: x1 - (d / H) * (x1 - x0)  # noqa: E731
    lw = 0.8

    def draw(c):
        for k in c.clades:
            ax.plot([X(c.h), X(k.h)], [k.y, k.y], color=style.INK_3, lw=lw, solid_capstyle="butt")
            draw(k)
        if c.clades:
            ax.plot([X(c.h), X(c.h)], [c.clades[0].y, c.clades[-1].y], color=style.INK_3, lw=lw,
                    solid_capstyle="projecting")

    draw(root)
    ax.plot([X(H) - 0.08, X(H)], [root.y, root.y], color=style.INK_3, lw=lw)  # root stub

    dot_x, lab_x = x1 + 0.09, x1 + 0.17
    for i in ids:
        w = works[i]
        y = row[i]
        ax.plot([x1, dot_x - 0.035], [y, y], color=style.RULE, lw=0.6)
        ax.plot(dot_x, y, "o", ms=5.2, mfc=style.ERA[w["group_label"]], mec="white", mew=0.7, zorder=3)
        cousin = w["group"] == "cousin"
        ax.text(lab_x, y, name(i), ha="left", va="center", fontsize=FS,
                color=style.INK_2 if cousin else style.INK, fontstyle="italic" if cousin else "normal")

    # ---- the clades the paper describes, as brackets on the right ----
    bx, tx = 4.22, 4.32
    notes = [
        ([35, 34, 36, 33], "Four literary ancestors group together"),
        ([30, 4], "The Killing Joke and Joker (2019): the film\nborrows the comic's failed-comedian origin"),
        ([40, 26], "Batman v Superman beside Superman"),
        ([43, 42, 41], "The modern superhero films"),
        ([31, 18, 6, 3], "Grounded crime and detective stories"),
        ([8, 24, 23, 17], "Batman Begins and The Dark Knight Rises,\napart from The Dark Knight. Llama's tree\n"
                          "joins all three Nolan films in one\ngrounded crime clade"),
        ([19, 29, 1, 15, 27, 28, 14], "Camp-leaning clade, which also takes in\nBatman R.I.P. and White Knight"),
        ([20, 25, 22], "The three Arkham games on their own branch"),
    ]
    for members, text in notes:
        rows = clade_of(members)
        a, b = min(rows) - 0.32, max(rows) + 0.32
        ax.plot([bx - 0.05, bx, bx, bx - 0.05], [a, a, b, b], color=style.INK_3, lw=0.7)
        ax.text(tx, (a + b) / 2, text, ha="left", va="center", fontsize=FS_S, color=style.INK_2, linespacing=1.25)

    style.title(fig, "The family tree follows tone more than date")
    fig.text(0.01, 1 - 0.30 / fig_h, "Qwen3-8B neighbour-joining tree of 32 Batman works and 13 relatives (italic), from "
             "shared memes. Rooted on\nPuss in Boots. Only the branching order is drawn: branch lengths are not to scale.",
             ha="left", va="top", fontsize=7.5, color=style.INK_2, linespacing=1.35)

    # era legend, three columns under the tree
    groups = list(style.ERA)
    present = {works[i]["group_label"] for i in ids}
    assert set(groups) == present
    handles = [Line2D([], [], ls="", marker="o", ms=5.2, mfc=style.ERA[g], mec="white", mew=0.7,
                      label=g.replace("Cousins & ancestors", "Relatives (cousins and ancestors)")) for g in groups]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.0), ncol=3, fontsize=FS,
               handletextpad=0.2, columnspacing=1.4, borderaxespad=0.2, labelspacing=0.5)
    return style.save(fig, "fig_tree.png")


# =====================================================================================
# Figure: how memes enter and spread
# =====================================================================================
HEAD_NAME = {33: "Monte Cristo", 34: "A Study in Scarlet", 35: "Dracula", 36: "Scarlet Pimpernel", 38: "The Bat",
             37: "Zorro", 39: "The Shadow", 40: "Superman", 41: "Spider-Man", 42: "Daredevil", 43: "Iron Man",
             44: "Green Hornet", 32: "Puss in Boots"}


def spread_figure():
    memes = {m["label"]: m for m in data["memes"]}
    cmemes = {m["unit"]: m for m in canon["memes"]}

    # comparison rows: three core canon memes (in >= 29 of 32 works); then the page's swim rows
    core_rows = ["The origin wound", "The secret identity", "Gotham City"]
    swim = [r["label"] for r in pm2["swim"]]
    for must in ["Fell into a vat of chemicals", "Ra's al Ghul"]:
        assert must in swim

    def entry(label):
        m = memes[label]
        bat = sorted((i for i in m["present"] if i < 32), key=lambda i: (works[i]["year"], i))
        cou = [i for i in m["present"] if i >= 32]
        # the page's swim rows and canon.json agree with data.json
        sw = next((r for r in pm2["swim"] if r["label"] == label), None)
        if sw:
            assert sorted(sw["batman"]) == sorted(bat) and sorted(sw["cousins"]) == sorted(cou), label
        assert sorted(i for i in cmemes[m["unit"]]["present"] if i < 45) == sorted(m["present"]), label
        # every work from the first year counts as the first appearance: the sample has years, not dates
        firsts = [i for i in bat if works[i]["year"] == works[bat[0]]["year"]]
        return {"label": label, "unit": m["unit"], "bat": bat, "cou": cou, "first": bat[0], "firsts": firsts,
                "n": len(bat)}

    top = [entry(l) for l in core_rows]
    for e in top:
        assert e["n"] >= 29 and cmemes[e["unit"]]["tier"] == "core" and works[e["first"]]["year"] == 1943
    opt = sorted((entry(l) for l in swim), key=lambda e: (works[e["first"]]["year"], -e["n"], e["label"]))
    for e in opt:
        assert e["n"] <= 16  # optional band
    ties = {e["label"]: [works[i]["label"] for i in e["firsts"]] for e in top + opt if len(e["firsts"]) > 1}
    assert ties == {"Fell into a vat of chemicals": ["The Killing Joke", "A Death in the Family"]}, ties

    layout = [("head", "For comparison: three core canon memes (in 29 or more of 32 works)")]
    layout += [("row", e) for e in top]
    layout += [("head", "Optional memes, in order of first appearance in this sample")]
    layout += [("row", e) for e in opt]

    rel_ids = sorted({i for kind, e in layout if kind == "row" for i in e["cou"]}, key=lambda i: works[i]["year"])
    early = [i for i in rel_ids if works[i]["year"] < 1939]
    late = [i for i in rel_ids if works[i]["year"] >= 1939]

    # ---- geometry, in inches (y measured down from the top) ----
    W = CANVAS_W
    row_h, head_h = 0.174, 0.24
    title_in = 0.62
    names_in = 1.02          # rotated relative names
    foot_in = 0.56           # year axis, rug, legend
    heights = [head_h if kind == "head" else row_h for kind, _ in layout]
    r0 = title_in + names_in
    tops = [r0 + sum(heights[:k]) for k in range(len(layout))]
    rows_bot = r0 + sum(heights)
    fig_h = rows_bot + foot_in
    fig = plt.figure(figsize=(W, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(fig_h, 0)
    ax.axis("off")

    lab_r = 1.44                         # row labels end here
    pre0, pre1 = 1.52, 2.06              # relatives before 1939: one slot each, in date order
    m0, m1 = 2.20, 6.12                  # the timeline
    y_lo, y_hi = 1938.5, 2023.5
    XY = lambda yr: m0 + (yr - y_lo) / (y_hi - y_lo) * (m1 - m0)  # noqa: E731
    pre_x = {i: pre0 + (k + 0.5) * (pre1 - pre0) / len(early) for k, i in enumerate(early)}
    cnt_x = W - 0.02

    # same-year Batman works sit side by side
    by_year = {}
    for i in range(32):
        by_year.setdefault(works[i]["year"], []).append(i)
    off = {i: (k - (len(ii) - 1) / 2) * 0.55 for ii in by_year.values() for k, i in enumerate(sorted(ii))}
    bx = lambda i: XY(works[i]["year"] + off[i])  # noqa: E731

    # ---- background: the 1967 to 1985 gap, gridlines, the "before 1939" panel ----
    gap_a, gap_b = XY(1966.6), XY(1985.4)
    ax.add_patch(Rectangle((gap_a, r0), gap_b - gap_a, rows_bot - r0, fc="#f1f2f6", ec="none", zorder=0))
    ax.text((gap_a + gap_b) / 2, tops[0] + head_h / 2 + 0.01, "No Batman works\n1967 to 1985",
            ha="center", va="center", fontsize=FS_S, color=style.INK_3, linespacing=1.15)
    for yr in [1940, 1960, 1980, 2000, 2020]:
        ax.plot([XY(yr)] * 2, [r0, rows_bot], color=style.RULE, lw=0.5, zorder=0)
    ax.add_patch(Rectangle((pre0, r0), pre1 - pre0, rows_bot - r0, fc="#f6f7f9", ec="none", zorder=0))

    # ---- rows ----
    for k, (kind, e) in enumerate(layout):
        y = tops[k] + heights[k] / 2
        if kind == "head":
            if k > 0:
                ax.plot([0.04, cnt_x], [tops[k]] * 2, color=style.RULE, lw=0.6)
            ax.text(0.04, y + 0.01, e, ha="left", va="center", fontsize=FS, fontweight="bold", color=style.INK)
            continue
        ax.text(lab_r, y, e["label"], ha="right", va="center", fontsize=FS, color=style.INK)
        ax.text(cnt_x, y, str(e["n"]), ha="right", va="center", fontsize=FS, color=style.INK_2)
        yb, yc = y - 0.02, y + 0.048    # Batman lane; relatives lane just below
        xs = [bx(i) for i in e["bat"]]
        ax.plot([xs[0], xs[-1]], [yb, yb], color=style.NIGHT_SOFT, lw=1.1, zorder=1, solid_capstyle="round")
        for i in e["bat"]:
            if i not in e["firsts"]:
                ax.plot(bx(i), yb, "o", ms=3.7, mfc=style.NIGHT, mec="white", mew=0.45, zorder=3)
        for f in e["firsts"]:
            ax.plot(bx(f), yb, "o", ms=5.6, mfc=style.SIGNAL, mec=style.INK, mew=0.6, zorder=4)
        f, fl = e["firsts"][0], e["firsts"][-1]
        first_names = ", ".join(name(i, year=False) for i in e["firsts"])
        if works[f]["year"] < 1960:   # no room on the left: label to the right, masking the line under it
            ax.text(bx(fl) + 0.065, yb, first_names, ha="left", va="center", fontsize=FS_S, color=style.INK_2,
                    zorder=2, bbox=dict(fc="white", ec="none", pad=0.4))
        else:
            ax.text(bx(f) - 0.065, yb, first_names, ha="right", va="center", fontsize=FS_S,
                    color=style.INK_2, zorder=2)
        for i in e["cou"]:
            x = pre_x[i] if i in pre_x else XY(works[i]["year"])
            ax.plot(x, yc, "o", ms=3.4, mfc="white", mec=style.SHARED, mew=0.9, zorder=2)

    # ---- column heads: relatives, rotated, fanned apart where their years crowd ----
    hy = r0 - 0.075
    ax.text(cnt_x, hy, "Batman\nworks", ha="right", va="bottom", fontsize=FS_S, color=style.INK_2, linespacing=1.15)
    targets = [(pre_x[i], i) for i in early] + [(XY(works[i]["year"]), i) for i in late]
    gapmin = 0.115
    pos = [t[0] for t in targets]
    for _ in range(500):
        moved = False
        for a in range(len(pos) - 1):
            d = pos[a + 1] - pos[a]
            if d < gapmin - 1e-9:
                push = (gapmin - d) / 2
                pos[a] -= push
                pos[a + 1] += push
                moved = True
        if not moved:
            break
    for (tx, i), px in zip(targets, pos):
        ax.plot([px, px, tx, tx], [hy + 0.005, hy + 0.03, r0 - 0.035, r0 - 0.005], color=style.SHARED, lw=0.5)
        ax.text(px, hy, f"{HEAD_NAME[i]} {works[i]['year']}", rotation=90, ha="center", va="bottom",
                fontsize=FS_S, color=style.INK_2)
    ax.text(pre0 - 0.04, r0 - 0.62, "Relatives\nbefore 1939", ha="right", va="center", fontsize=FS_S,
            color=style.INK_3, linespacing=1.15)
    ax.text(XY(1952), r0 - 0.62, "Relatives\nafter 1939", ha="center", va="center", fontsize=FS_S,
            color=style.INK_3, linespacing=1.15)

    # ---- year axis, and a rug of the 32 Batman works ----
    ay = rows_bot + 0.035
    for i in range(32):
        ax.plot([bx(i)] * 2, [ay, ay + 0.06], color=style.INK_3, lw=0.6)
    for yr in [1940, 1960, 1980, 2000, 2020]:
        ax.text(XY(yr), ay + 0.10, str(yr), ha="center", va="top", fontsize=FS, color=style.INK_3)
    for x in (pre1 + 0.045, pre1 + 0.085):   # break between the two time spans
        ax.plot([x - 0.02, x + 0.02], [ay + 0.06, ay], color=style.INK_3, lw=0.6)

    # ---- legend ----
    ly = fig_h - 0.12
    items = [
        ("o", dict(ms=5.6, mfc=style.SIGNAL, mec=style.INK, mew=0.6), "First appearance in this sample"),
        ("o", dict(ms=3.7, mfc=style.NIGHT, mec="white", mew=0.45), "Later Batman work"),
        ("o", dict(ms=3.4, mfc="white", mec=style.SHARED, mew=0.9), "Relative (one summary each)"),
        ("|", dict(ms=5, mec=style.INK_3, mew=0.6), "Each Batman work in the sample"),
    ]
    lx = 0.10
    for marker, kw, text in items:
        ax.plot(lx, ly, marker, **kw)
        t = ax.text(lx + 0.08, ly, text, ha="left", va="center", fontsize=FS, color=style.INK_2)
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transData.inverted())
        lx = bb.x1 + 0.24

    style.title(fig, "Core memes are there from the first work; most optional ones enter later and recur")
    fig.text(0.01, 1 - 0.31 / fig_h, "Qwen3-8B, 18 of the 31 named memes. A dot for each Batman work where the meme is "
             "present, placed by year.\nLabels were written by an AI reader and are imperfect.",
             ha="left", va="top", fontsize=7.5, color=style.INK_2, linespacing=1.35)
    return style.save(fig, "fig_spread.png")


if __name__ == "__main__":
    tree_figure()
    spread_figure()
