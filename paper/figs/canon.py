"""Figures: Batman's canon and the meme genome, plus the table of neuron examples.

fig_canon.png       How many of the 32 Batman works each Qwen3-8B meme family appears in, with the three tiers,
                    the pseudo-franchise null, and the named must-have memes (Batman's own against shared).
fig_genome.png      The 31 named memes (rows, by kind) across the 32 Batman works in date order and the 13 relatives,
                    filled where the Qwen meme is present, with a dot where its Llama 3.1 8B twin is present.
neuron_examples.md  Eight named memes and two of the sentences their neuron fires on strongly, trimmed.

Data (read only): report/data.json, results/canon.json, results/labels.json, results/labels_wf.json,
results/robustness/nulls.json.

    .venv/bin/python paper/figs/canon.py
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import transforms  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

D = json.loads((ROOT / "report/data.json").read_text())
canon = json.loads((ROOT / "results/canon.json").read_text())
nulls = json.loads((ROOT / "results/robustness/nulls.json").read_text())
lab_old = {int(k): v for k, v in json.loads((ROOT / "results/labels.json").read_text()).items()}
lab_wf = {int(k): v for k, v in json.loads((ROOT / "results/labels_wf.json").read_text()).items()}

bat = sorted([w for w in D["works"] if w["group"] == "batman"], key=lambda w: (w["year"], w["label"]))
cous = sorted([w for w in D["works"] if w["group"] == "cousin"], key=lambda w: w["year"])
assert len(bat) == 32 and len(cous) == 13

# ordinal ramp for the tiers, one hue light -> dark (checked with the dataviz validator, --ordinal: all pass)
TIER_COL = {"optional": "#9fb4d6", "common": "#5b7db3", "core": style.NIGHT}
TIER_NAME = {"core": "Must have", "common": "Usually there", "optional": "Optional"}
TIER_RANGE = {"core": (29, 32), "common": (23, 28), "optional": (6, 22)}


def itext(fig, x, y, s, **kw):
    """Text placed in inches from the figure's bottom-left corner."""
    return fig.text(x, y, s, transform=fig.dpi_scale_trans, **kw)


def iline(fig, xs, ys, **kw):
    fig.add_artist(Line2D(xs, ys, transform=fig.dpi_scale_trans, **kw))


def width_in(fig, t):
    return t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi


def round_half_up(x):
    return int(math.floor(x + 0.5))


# =====================================================================================================
# Figure: Batman's canon
# =====================================================================================================
def label_for(u):
    """Same rule as the companion page: the hand-checked labels first, then the workflow labels."""
    if u in lab_old:
        return {**lab_old[u], "confidence": "high"}
    return lab_wf.get(u)


def fig_canon():
    rows = canon["memes"]
    assert len(rows) == 14843
    counts = {n: 0 for n in range(6, 33)}
    for r in rows:
        counts[r["n_batman"]] += 1
    tiers = canon["tiers"]
    tot = {t: tiers[f"{t}|specific"] + tiers[f"{t}|shared"] for t in ("core", "common", "optional")}
    assert tot == {"core": 56, "common": 194, "optional": 14593}, tot
    assert tiers["core|specific"] == 24 and tiers["core|shared"] == 32

    # named core memes, as on the companion page: a confident label that is not 'mixed' or Wikipedia style
    named = []
    for r in rows:
        if r["tier"] != "core":
            continue
        lab = label_for(r["unit"])
        if not lab or lab.get("kind") in ("mixed", "register") or lab.get("confidence") == "low":
            continue
        named.append({**r, "label": lab["label"], "kind": lab["kind"]})
    n_none = n_low = n_reg = 0
    for r in rows:
        if r["tier"] == "core":
            lab = label_for(r["unit"])
            if not lab:
                n_none += 1
            elif lab.get("kind") == "register":
                n_reg += 1
            elif lab.get("kind") == "mixed" or lab.get("confidence") == "low":
                n_low += 1
    assert n_none + n_low + n_reg + len(named) == 56
    n_own = sum(r["specific"] for r in named)
    n_shared = len(named) - n_own
    assert (n_own, n_shared) == (13, 16), (n_own, n_shared)

    W, H = style.WIDTH_IN, 6.0
    fig = plt.figure(figsize=(W, H))
    style.title(fig, "Batman's canon: 56 meme families are in nearly every Batman work",
                "Qwen3-8B meme families by how many of the 32 Batman works they appear in (original texts, names present)")
    y_panel = 5.24  # panel titles A and B

    # ---------------- A: the distribution ----------------
    ax_l, ax_b, ax_w, ax_h = 0.62, 3.12, 3.55, 1.55
    axA = fig.add_axes([ax_l / W, ax_b / H, ax_w / W, ax_h / H])
    xs = list(range(6, 33))

    def tier_of(n):
        return "core" if n >= 29 else "common" if n >= 23 else "optional"

    axA.bar(xs, [counts[n] - 1 for n in xs], width=0.72, bottom=1, color=[TIER_COL[tier_of(n)] for n in xs],
            linewidth=0, zorder=3)
    axA.set_yscale("log")
    axA.set_ylim(1, 2e4)
    axA.set_xlim(5.3, 32.7)
    axA.set_yticks([1, 10, 100, 1000, 10000])
    axA.set_yticklabels(["1", "10", "100", "1,000", "10,000"])
    axA.minorticks_off()
    axA.set_xticks([6, 10, 15, 20, 23, 29, 32])
    axA.set_xlabel("Number of the 32 Batman works the meme family appears in", labelpad=3)
    axA.set_ylabel("Meme families (log scale)", labelpad=3)
    axA.grid(axis="y", color=style.RULE, linewidth=0.6, zorder=0)
    axA.spines["left"].set_visible(False)
    axA.tick_params(axis="y", length=0)
    for n in (29, 30, 31, 32):  # selective labels: the four core bars
        axA.text(n, counts[n] * 1.18, f"{counts[n]:,}", ha="center", va="bottom", fontsize=6.6, color=style.INK_2)
    # tier brackets above the plot: name, then the number of families
    blend = transforms.blended_transform_factory(axA.transData, axA.transAxes)
    for t in ("optional", "common", "core"):
        a, b = TIER_RANGE[t]
        axA.plot([a - 0.36, b + 0.36], [1.04, 1.04], transform=blend, clip_on=False, color=TIER_COL[t],
                 linewidth=2.2, solid_capstyle="butt")
        mid = (a + b) / 2
        axA.text(mid, 1.075, f"{tot[t]:,}" + (" families" if t == "optional" else ""), transform=blend,
                 ha="center", va="bottom", fontsize=7.2, color=style.INK)
        axA.text(mid, 1.19, TIER_NAME[t], transform=blend, ha="center", va="bottom", fontsize=7.2,
                 fontweight="bold", color=style.INK)
    itext(fig, 0.06, y_panel, "A", fontsize=9, fontweight="bold", va="bottom")
    itext(fig, 0.24, y_panel, "Most meme families are optional. A small core is in almost every work",
          fontsize=7.6, fontweight="bold", va="bottom", color=style.INK)

    # ---------------- B: pseudo-franchise null, as numbers ----------------
    q = nulls["f1_chance"]["qwen3-8b"]
    assert q["reps"] == 50
    rowsB = [("at least 6", q["present_in_ge6_of_32"]), ("at least 23", q["present_in_ge23_of_32"]),
             ("at least 29", q["present_in_ge29_of_32"])]
    for _, v in rowsB[1:]:
        assert v["pseudo_franchise_bar37"]["max"] == 0 and v["draws_batman_above_pseudo"] == 50
    xB = 4.56
    x0, c1, c2 = xB + 0.18, xB + 1.40, W - 0.11
    itext(fig, xB, y_panel, "B", fontsize=9, fontweight="bold", va="bottom")
    itext(fig, x0, y_panel, "Random sets of 32 films have no core", fontsize=7.6, fontweight="bold", va="bottom",
          color=style.INK)
    itext(fig, x0, y_panel - 0.24, "Single neurons present in", fontsize=6.8, color=style.INK_2, va="bottom")
    yh = y_panel - 0.56
    itext(fig, x0, yh, "works of 32", fontsize=6.8, color=style.INK_3, va="bottom")
    itext(fig, c1, yh, "Batman", fontsize=6.8, color=style.INK_3, va="bottom", ha="right")
    itext(fig, c2, yh, "Random\nsets", fontsize=6.8, color=style.INK_3, va="bottom", ha="right", linespacing=1.1)
    iline(fig, [x0, c2], [yh - 0.05, yh - 0.05], color=style.INK_3, linewidth=0.6)
    for k, (lab, v) in enumerate(rowsB):
        y = yh - 0.30 - k * 0.27
        itext(fig, x0, y, lab, fontsize=7.2, color=style.INK, va="bottom")
        itext(fig, c1, y, f"{round_half_up(v['batman_bar37']['mean']):,}", fontsize=7.6, fontweight="bold",
              color=style.NIGHT, va="bottom", ha="right")
        rnd = v["pseudo_franchise_bar37"]["mean"]
        itext(fig, c2, y, f"{round_half_up(rnd):,}", fontsize=7.6, fontweight="bold" if rnd == 0 else "normal",
              color=style.INK, va="bottom", ha="right")
        iline(fig, [x0, c2], [y - 0.07, y - 0.07], color=style.RULE, linewidth=0.6)
    itext(fig, x0, yh - 0.30 - 2 * 0.27 - 0.17,
          "Means of 50 draws in which 32 ordinary films\nplay the franchise. Units are single neurons\n"
          "before merging; the presence bar is set by\nthe other 37 films. No draw had a unit in\n"
          "23 or more of its 32 films.",
          fontsize=6.6, color=style.INK_3, va="top", linespacing=1.3)

    # ---------------- C: the named core, own vs shared, story patterns vs names ----------------
    yC = 2.42
    itext(fig, 0.06, yC, "C", fontsize=9, fontweight="bold", va="bottom")
    itext(fig, 0.24, yC, "The named must-have memes. Batman's own are mostly names and places. "
          "The story patterns are shared", fontsize=7.6, fontweight="bold", va="bottom", color=style.INK)
    itext(fig, 0.24, yC - 0.19,
          f"{len(named)} of the 56 have a confident label from an AI reader. The other {56 - len(named)} are unnamed, "
          "unclear, or track Wikipedia's house style.", fontsize=6.8, color=style.INK_2, va="bottom")
    print(f"core labels: {len(named)} named, {n_none} unlabelled, {n_low} mixed or low confidence, {n_reg} register")
    STORY = ("trope", "genre")
    cols = []
    for own in (True, False):
        for story in (True, False):
            items, seen = [], {}
            for r in sorted(named, key=lambda r: (-r["n_batman"], r["label"])):
                if r["specific"] != own or (r["kind"] in STORY) != story:
                    continue
                if r["label"] in seen:  # Gotham City has two Batman-own units
                    seen[r["label"]]["dup"] += 1
                    continue
                seen[r["label"]] = {"label": r["label"], "dup": 1}
                items.append(seen[r["label"]])
            cols.append({"own": own, "story": story, "items": items,
                         "head": "Story patterns" if story else "Characters, places, things"})
    fs, line_h, mk = 6.6, 0.142, 0.075
    # measure each column, then spread the spare width as gaps
    for c in cols:
        ws = []
        for it in c["items"]:
            s = it["label"] + (f" (×{it['dup']})" if it["dup"] > 1 else "")
            it["text"] = s
            t = itext(fig, 0, 0, s, fontsize=fs)
            ws.append(width_in(fig, t) + mk + 0.05)
            t.remove()
        t = itext(fig, 0, 0, c["head"], fontsize=6.8, fontweight="bold")
        c["w"] = max(ws + [width_in(fig, t)])
        t.remove()
    left, right = 0.06, W - 0.11
    spare = right - left - sum(c["w"] for c in cols)
    gap_in = 0.1  # inside the own and shared groups; the rest separates the two groups
    gap_mid = spare - 2 * gap_in
    assert gap_mid > 0.2, spare
    x = left
    for k, c in enumerate(cols):
        c["x"] = x
        x += c["w"] + (gap_mid if k == 1 else gap_in)
    y_head = yC - 0.47
    for own, name, n_named, n_all, sub in ((True, "Batman's own", n_own, 24, "in at most 3 of the 13 relatives"),
                                           (False, "Shared with other heroes", n_shared, 32, "in 4 or more of the 13 relatives")):
        cs = [c for c in cols if c["own"] == own]
        xa, xb = cs[0]["x"], cs[1]["x"] + cs[1]["w"]
        th = itext(fig, xa, y_head, f"{name}: {n_named} named of {n_all}", fontsize=7.2, fontweight="bold",
                   va="bottom", color=style.INK)
        itext(fig, xa + width_in(fig, th) + 0.08, y_head, sub, fontsize=6.6, color=style.INK_3, va="bottom")
        iline(fig, [xa, xb], [y_head - 0.05, y_head - 0.05], color=style.INK_3, linewidth=0.6)
    y_sub = y_head - 0.24
    for c in cols:
        itext(fig, c["x"], y_sub, c["head"], fontsize=6.8, fontweight="bold", color=style.INK_2, va="bottom")
        for k, it in enumerate(c["items"]):
            yy = y_sub - 0.13 - k * line_h
            fig.add_artist(FancyBboxPatch((c["x"], yy - mk / 2), mk, mk, boxstyle="round,pad=0,rounding_size=0.015",
                                          transform=fig.dpi_scale_trans,
                                          facecolor=style.NIGHT if c["own"] else style.SHARED, linewidth=0))
            itext(fig, c["x"] + mk + 0.05, yy, it["text"], fontsize=fs, color=style.INK, va="center")
    lowest = y_sub - 0.13 - (max(len(c["items"]) for c in cols) - 1) * line_h
    assert lowest > 0.08, lowest
    return style.save(fig, "fig_canon.png")


# =====================================================================================================
# Figure: the meme genome
# =====================================================================================================
KIND_ORDER = [("trope", "Story patterns"), ("object", "Objects"), ("place", "Places"), ("character", "Characters"),
              ("register", "Wikipedia style")]


def short_label(w):
    """Column label: year, then the work's short label with a repeated year removed."""
    lab = w["label"]
    for tok in (f" ({w['year']})", f" '{str(w['year'])[2:]}", f"{w['year']} "):
        lab = lab.replace(tok, "")
    return f"{w['year']}  {lab}"


def fig_genome():
    memes = D["memes"]
    assert len(memes) == 31
    bat_ids = [w["id"] for w in bat]
    for m in memes:
        m["_nb"] = sum(i in set(m["present"]) for i in bat_ids)
        assert m["_nb"] == m["n_batman"], (m["label"], m["_nb"], m["n_batman"])

    # columns: Batman works by date, a narrow gap for 1967-85 (no works), a gap, then the relatives by date
    xpos, x, gap_x = {}, 0.0, None
    for w in bat:
        if w["year"] > 1966 and gap_x is None:
            gap_x = x
            x += 0.75
        xpos[w["id"]] = x
        x += 1
    rel_start = x + 0.8
    x = rel_start
    for w in cous:
        xpos[w["id"]] = x
        x += 1
    ncol = x

    # rows: kind groups with a header each; memes by Batman prevalence inside a group
    layout = []
    for kind, name in KIND_ORDER:
        group = sorted([m for m in memes if m["kind"] == kind], key=lambda m: (-m["_nb"], m["label"]))
        assert group, kind
        layout.append(("head", f"{name} ({len(group)})" if len(group) > 1 else name))
        layout.extend(("meme", m) for m in group)
    assert sum(t == "meme" for t, _ in layout) == 31

    W = style.WIDTH_IN
    label_w = 2.0
    col_in = (W - label_w - 0.12) / ncol
    row_in, head_in = 0.114, 0.16
    grid_h = sum(head_in if t == "head" else row_in for t, _ in layout)
    top_band, collab_h, bottom_band = 0.60, 1.42, 0.74
    H = top_band + collab_h + grid_h + bottom_band
    fig = plt.figure(figsize=(W, H))
    style.title(fig, "The meme genome: 31 named memes across 80 years of Batman and 13 relatives",
                "Filled cell: the Qwen3-8B meme is present in the work. Dot: its twin in Llama 3.1 8B is present.")
    gy_top = H - top_band - collab_h  # top of the grid, inches from the bottom

    def X(col):
        return label_w + col * col_in

    # column labels and era strip
    strip_y = gy_top + 0.05
    tallest = 0
    for w in bat + cous:
        fig.add_artist(Rectangle((X(xpos[w["id"]]) + 0.008, strip_y), col_in - 0.016, 0.045,
                                 transform=fig.dpi_scale_trans, facecolor=style.ERA[w["group_label"]], linewidth=0))
        t = itext(fig, X(xpos[w["id"]] + 0.5), strip_y + 0.085, short_label(w), rotation=90, rotation_mode="anchor",
                  ha="left", va="center", fontsize=6.5, color=style.INK if w["group"] == "batman" else style.INK_2,
                  style="normal" if w["group"] == "batman" else "italic")
        tallest = max(tallest, t.get_window_extent(renderer=fig.canvas.get_renderer()).height / fig.dpi)
    assert tallest < collab_h - 0.12, tallest
    itext(fig, X(gap_x + 0.375), strip_y + 0.085, "1967–85: no works", rotation=90, rotation_mode="anchor",
          ha="left", va="center", fontsize=6.5, color=style.INK_3, style="italic")
    itext(fig, label_w - 0.05, gy_top + 0.03, "Batman\nworks", ha="right", va="bottom", fontsize=6.5,
          color=style.INK_3, linespacing=1.05)

    # era legend in the empty corner above the row labels
    ly = gy_top + collab_h - 0.06
    itext(fig, 0.06, ly, "Era of each column (strip)", fontsize=6.8, fontweight="bold", color=style.INK_2, va="center")
    for k, (era, col) in enumerate(style.ERA.items()):
        yy = ly - 0.112 * (k + 1)
        fig.add_artist(Rectangle((0.06, yy - 0.03), 0.09, 0.06, transform=fig.dpi_scale_trans, facecolor=col, linewidth=0))
        itext(fig, 0.2, yy, "Relatives and ancestors" if era.startswith("Cousins") else era, fontsize=6.6,
              color=style.INK_2, va="center")
    assert ly - 0.112 * len(style.ERA) - 0.05 > gy_top + 0.25  # clear of the 'Batman works' header

    # grid
    y = gy_top
    agree = total = 0
    for t, payload in layout:
        if t == "head":
            y -= head_in
            itext(fig, 0.06, y + 0.035, payload, fontsize=7.0, fontweight="bold", color=style.INK, va="bottom")
            iline(fig, [0.06, X(ncol)], [y + 0.02, y + 0.02], color=style.RULE, linewidth=0.5)
            continue
        m = payload
        y -= row_in
        q, ll = set(m["present"]), set(m["llama"]["present"])
        itext(fig, label_w - 0.28, y + row_in / 2, m["label"], ha="right", va="center", fontsize=6.7, color=style.INK)
        itext(fig, label_w - 0.05, y + row_in / 2, str(m["_nb"]), ha="right", va="center", fontsize=6.7,
              color=style.INK_2)
        for w in bat + cous:
            i = w["id"]
            on, lon = i in q, i in ll
            agree += on == lon
            total += 1
            x0 = X(xpos[i])
            fig.add_artist(FancyBboxPatch((x0 + 0.009, y + 0.009), col_in - 0.018, row_in - 0.018,
                                          boxstyle="round,pad=0,rounding_size=0.012", transform=fig.dpi_scale_trans,
                                          facecolor=style.NIGHT if on else style.OFF, linewidth=0))
            if lon:
                fig.add_artist(Line2D([x0 + col_in / 2], [y + row_in / 2], transform=fig.dpi_scale_trans, marker="o",
                                      markersize=2.3, markeredgewidth=0, linestyle="none",
                                      color="white" if on else style.NIGHT))
    kappa = D["stats"]["agreement"]["llama3.1-8b"]["named"]["kappa"]
    assert round(agree / total, 3) == D["stats"]["agreement"]["llama3.1-8b"]["named"]["agree"], agree / total
    grid_bottom = y

    # group captions under the columns
    yb = grid_bottom - 0.05
    for a, b, s in ((X(0), X(xpos[bat[-1]["id"]] + 1), "32 Batman works, oldest first"),
                    (X(rel_start), X(ncol), "13 relatives, oldest first")):
        iline(fig, [a + 0.01, b - 0.01], [yb, yb], color=style.INK_3, linewidth=0.6)
        itext(fig, (a + b) / 2, yb - 0.04, s, ha="center", va="top", fontsize=6.8, color=style.INK_2)

    # key, then the agreement figure on its own line
    ky = grid_bottom - 0.36
    kx = 0.06
    cw, chh = col_in - 0.018, row_in - 0.018
    for kind, s in (("q", "Qwen3-8B meme present"), ("both", "Llama twin present too"),
                    ("l", "Llama twin present, Qwen meme absent"), ("off", "Neither")):
        fig.add_artist(FancyBboxPatch((kx, ky - chh / 2), cw, chh, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.dpi_scale_trans,
                                      facecolor=style.OFF if kind in ("l", "off") else style.NIGHT, linewidth=0))
        if kind in ("both", "l"):
            fig.add_artist(Line2D([kx + cw / 2], [ky], transform=fig.dpi_scale_trans, marker="o", markersize=2.3,
                                  markeredgewidth=0, linestyle="none", color="white" if kind == "both" else style.NIGHT))
        t = itext(fig, kx + cw + 0.05, ky, s, fontsize=6.8, color=style.INK_2, va="center")
        kx += cw + 0.05 + width_in(fig, t) + 0.2
    assert kx < W, kx
    itext(fig, 0.06, ky - 0.17, f"The two models mark {agree / total:.0%} of the {total:,} cells the same way "
          f"(Cohen's kappa {kappa:.2f}).", fontsize=6.6, color=style.INK_3, va="center")
    itext(fig, 0.06, ky - 0.31, "Present means the neuron fired more strongly for the work than for 68 of the 69 "
          "ordinary films.", fontsize=6.6, color=style.INK_3, va="center")
    return style.save(fig, "fig_genome.png")


# =====================================================================================================
# Table: what single neurons fire on
# =====================================================================================================
KIND_WORD = {"trope": "story pattern", "object": "object", "place": "place", "character": "character",
             "register": "Wikipedia style"}

# (meme, [(work, sentence prefix, first word, number of words), ...]). Every sentence comes from data.json:
# the meme's top sentences, or its strongest sentence in that work (best_sentence). Text is only trimmed.
EXAMPLES = [
    ("The origin wound", [("Earth One", "Bruce dons", 0, 16), ("Batman Begins", "In Gotham City, young", 0, 16)]),
    ("The secret identity", [("The Batman", "The Riddler gives", 0, 13), ("Spider-Man", "Norman notices", 0, 16)]),
    ("The joke", [("The Killing Joke", "He says that", 0, 16), ("Joker (2019)", "As the show", 0, 14)]),
    ("A funeral", [("Batman v Superman", "In a flashback", 0, 16), ("Dark Knight Returns", "After Wayne's funeral", 0, 11)]),
    ("Two-Face's coin", [("Arkham Asylum (comic)", "Two-Face declares", 0, 14),
                         ("Batman Forever", "Two-Face corners", 0, 11)]),
    ("Commissioner Gordon", [("Year One", "James", 0, 16), ("Dark Knight Returns", "Batman informs", 0, 16)]),
    ("Harvey Dent", [("The Long Halloween", "Batman meets", 0, 16), ("Gotham by Gaslight", "Selina Kyle, a famed", 6, 12)]),
    ("Lucius Fox", [("Batman Begins", "Company archivist", 0, 16), ("Zorro", "Don Diego", 0, 11)]),
]


def sentences_for(m, work_label):
    out = [t["text"] for t in m["top"] if t["work"] == work_label]
    wid = next(w["id"] for w in D["works"] if w["label"] == work_label)
    bs = m["best_sentence"].get(str(wid))
    if bs and bs not in out:
        out.append(bs)
    return out


def trim(s, start, n):
    words = s.split()
    txt = " ".join(words[start:start + n]).rstrip(",;:")
    assert txt in s, (txt, s)
    if start > 0:
        txt = "…" + txt
    if start + n < len(words):
        txt += "…"
    return txt


def neuron_table():
    by_label = {m["label"]: m for m in D["memes"]}
    lines = [
        "**Table 1.** What single neurons fire on. Eight of the 31 named memes with their kind, the neuron that "
        "defines each (L10·9068+ means layer 10, neuron 9068, positive pole), the number of the 32 Batman works it is "
        "present in, and two sentences from different works that it fires on strongly. Each sentence is one of the "
        "meme's eight strongest or its strongest in that work, trimmed. Spider-Man and Zorro are relatives, not Batman "
        "works. The labels were written by an AI reader from such sentences and are imperfect.",
        "",
        "| Meme | Neuron | Batman works | Fires on |",
        "| --- | --- | --- | --- |",
    ]
    for label, picks in EXAMPLES:
        m = by_label[label]
        cells, used = [], set()
        for work, prefix, start, n in picks:
            cand = [s for s in sentences_for(m, work) if s.startswith(prefix)]
            assert len(cand) == 1, (label, work, cand)
            used.add(work)
            cells.append(f"*{work}:* “{trim(cand[0], start, n)}”")
        assert len(used) == 2
        lines.append(f"| {label}<br>*{KIND_WORD[m['kind']]}* | {m['addr']} | {m['n_batman']} of 32 | "
                     + "<br>".join(cells) + " |")
    lines += [
        "",
        "Two of these labels are narrower than their neurons. The Lucius Fox neuron also fires on Zorro's nickname, "
        "“The Fox”, and on “a Zorro movie” in Batman R.I.P., so it tracks the word as much as the character. The "
        "Harvey Dent neuron's strongest sentence in Gotham by Gaslight names a different Harvey, police chief Harvey "
        "Bullock, so it partly detects the first name. Even a clean unit has stray hits. The origin wound's "
        "fourth-strongest sentence (Knightfall) is about training a new hero, not an origin.",
        "",
    ]
    # the claims in that note, checked against the data
    assert "Zorro" in by_label["Lucius Fox"]["best_sentence"]["37"] and 37 in by_label["Lucius Fox"]["present"]
    assert "Zorro movie" in " ".join(sentences_for(by_label["Lucius Fox"], "Batman R.I.P."))
    assert "Harvey Bullock" in by_label["Harvey Dent"]["best_sentence"]["6"]
    assert "Dent" not in by_label["Harvey Dent"]["best_sentence"]["6"]
    assert by_label["The origin wound"]["top"][3]["work"] == "Knightfall"
    path = Path(__file__).resolve().parent / "neuron_examples.md"
    path.write_text("\n".join(lines))
    print("wrote", path)
    return path


if __name__ == "__main__":
    fig_canon()
    fig_genome()
    neuron_table()
