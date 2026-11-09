"""Figure for the second corpora (v2/second/PLAN.md, Section 9): film remakes, the decision test.

The decision test asks whether a measure points to a remake's documented source beyond what the word model already
says. Its score is Z: 0 means that the measure adds nothing to the word model.

fig_remakes.png, two panels.
  A  The units against everything that has no access to them. One row per model: the Z of its units (primary set,
     primary frame, primary setting), with a line down to the axis; the primary model is the solid dark mark. Under
     them, what went through the same test in the units' place: the sentence-embedding comparator (grey diamond),
     the word-only stand-ins and the random word-only measures (one tick each) and the draws of pure noise
     (histogram). The dashed line is the stand-ins' largest Z, which the units had to exceed (the plan's condition
     5). Each row's Z, or the largest Z in the row, is printed at the right.
  B  Where the primary model's score comes from: the sum of the test's contributions within each family of remakes
     (films joined by remake statements), one thin bar per family with a non-zero sum, largest first. Dark bars are
     families in which the model recognises no pair, grey bars those in which it recognises at least one (it names
     the source, or identifies either film). The largest are named by the remake that contributes most.

Reads results/v2/second/ (or $RESULTS_DIR): remakes_<tag>.json for whichever of the four models are there,
remakes_words.json and remakes_null_size.json. A model without a file, or whose file cannot be read yet, is
skipped, so the script is simply re-run when the others arrive. Every number drawn is read from those files at run
time and printed with the key it came from. The title is one of three sentences, chosen by which models' units are
above the stand-ins' largest Z (`headline`); it is printed, with the subtitle and a caption filled in from the same
numbers. The title says nothing about how the score is spread over the films: panel B shows that, with the share
of the total that the largest families add. Writes fig_remakes.png to paper/v2/ (or $PAPER_DIR).

    cd study && ../.venv/bin/python v2/figs/remakes.py
"""
import json
import os
import re
import sys
from math import ceil, floor
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
os.environ.setdefault("PAPER_DIR", str(STUDY / "paper" / "v2"))
sys.path.insert(0, str(STUDY / "paper" / "figs"))
import style  # noqa: E402

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

# the title is chosen from the results (see `headline`); {k} and {n} count models
TITLE_ALL = "On remakes, the units clear the bar set by words alone"
TITLE_SOME = "On remakes, the units of {k} of the {n} models clear the bar set by words alone"
TITLE_NOT = "On remakes, the primary model's units do not clear the bar set by words alone"
# {pairs} and {primary} are filled in from the results
SUBTITLE = ("Decision test on {pairs} remake pairs: does a measure point to the documented source beyond what "
            "the word model says?\nA: the score of everything put through the test. "
            "B: what each family of remakes adds to {primary}'s score.")
PANEL_A = "The test's score: units against words and noise"
PANEL_B = "Where {primary}'s score comes from"
GROUP_UNITS = "The units of"
GROUP_PLACE = "In the units' place"
BAR_LABEL = "the bar: the largest\nstand-in, {z}"
X_LABEL_A = "Score on the decision test (Z)\n0 = adds nothing to the word model"
ROW_EMBEDDING = "a sentence\nembedding"
ROW_STANDINS = "{n} word-only\nstand-ins"
ROW_RANDOM = "{n} random\nword measures"
ROW_NOISE = "{n:,} draws\nof pure noise"
UP_TO = "up to {z}"
Y_LABEL_B = "What the family adds to the test's total"
X_LABEL_B = "{shown} of {families} families of remakes, largest\nfirst; the rest add exactly nothing"
KEY = "Recognised by the model?"
KEY_NOT = "no: {n} families"
KEY_REC = "yes: {n} families"
LARGEST = "the {k} largest add {share} of the total;\n"   # {share}: their sums over the test's total, as a percentage
COUNTS = "{positive} families add more than 0.1,\n{negative} take away more than 0.1"   # sums under 0.1 in size are mostly rounding residue
COUNTS_SMALL = ";\nmost of the rest are too small to see"
# a caption for the paper, printed with the numbers of this run ({others}: the replication models)
CAPTION = ("**Figure N.** The decision test on the {pairs} documented remake pairs asks whether a measure points to a "
           "remake's source beyond what a model of the words alone already says; a score (Z) of 0 means that it adds "
           "nothing. A: the units of {primary} score {z}{others}; the {n_standins} word-only measures put through "
           "the test in their place (the stand-ins) reach at most {bar}, which is the bar the units had to clear, "
           "{n_random} more made at random at most {random_max}, and {n_noise:,} draws of pure noise at most "
           "{noise_max}{embedding}. B: what each family of remakes (films linked by remake statements) adds to "
           "{primary}'s score: {shown} of the {families} families have a sum that is not exactly zero, {positive} add more "
           "than 0.1 and {negative} take away more than 0.1, and the {k} largest add {share} of the total{left_out}; dark bars are the {n_not} families in which "
           "the model recognises no pair, grey bars the {n_rec} in which it recognises at least one, and the largest "
           "are named by the remake that adds most.")
CAPTION_OTHERS = " ({list})"
CAPTION_EMBEDDING = "; a general-purpose sentence embedding scores {z}"
CAPTION_LEFT_OUT = " (with any one of them left out, the score stays between {lo} and {hi})"

RESULTS = Path(os.environ.get("RESULTS_DIR") or STUDY / "results" / "v2" / "second")
GREY = style.SHARED
MODELS = [  # tag in the file name, display name, marker face, marker edge; the primary model is the solid dark one
    ("qwen3-32b", "Qwen3-32B", style.NIGHT, style.NIGHT),
    ("qwen3-8b", "Qwen3-8B", "white", style.NIGHT),
    ("llama3.1-8b", "Llama 3.1 8B", GREY, GREY),
    ("llama3.3-70b", "Llama 3.3 70B", "white", GREY),
]
PRIMARY, PRIMARY_NAME = MODELS[0][:2]
ANALYSIS = "primary | set"                # the primary set in its primary frame (PLAN Section 9)
NOISE = "39 columns (registered)"         # the registered word model, in remakes_null_size.json
NAMED = 4                                 # families named in panel B
FS = 7                                    # labels; nothing is below 6.5 pt


def load(name):
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:  # a file that is still being written is treated as not there yet
        print(f"  {name}: cannot be read yet, skipped")
        return None


def num(z):
    """Two decimals, with a true minus sign as on the axis."""
    return f"{z:.2f}".replace("-", "\u2212")


def units_z(d, tag):
    """Z of a model's units in the decision test, for the primary set in its primary frame; None if it has none."""
    a = (d.get("analyses") or {}).get(ANALYSIS) or {}
    dec = d.get("decision") or {}
    if "Z" in (a.get("slope") or {}):
        where, block = f'analyses["{ANALYSIS}"]', a
    elif f"{dec.get('set')} | {dec.get('frame')}" == ANALYSIS and "Z" in (dec.get("slope") or {}):
        where, block = "decision", dec
    else:
        print(f"  remakes_{tag}.json: no decision test for {ANALYSIS!r} yet, skipped")
        return None
    print(f"  remakes_{tag}.json: {where}.slope.Z = {block['slope']['Z']:.4f} (plane {block.get('plane')}, "
          f"p {block['slope'].get('p')})")
    return block["slope"]["Z"]


def headline(z, bar):
    """The title, from what the results show: which models' units are above the stand-ins' largest Z."""
    clear = [tag for tag in z if z[tag] > bar]
    if PRIMARY not in clear:
        return TITLE_NOT
    if len(clear) < len(z):
        return TITLE_SOME.format(k=len(clear), n=len(z))
    return TITLE_ALL


# ---- what is drawn ----
files = {m[0]: load(f"remakes_{m[0]}.json") for m in MODELS}
print("models in the folder:", ", ".join(tag for tag, d in files.items() if d) or "none")
primary = files[PRIMARY]
if not (primary and "Z" in ((primary.get("analyses") or {}).get(ANALYSIS) or {}).get("slope", {})):
    print(f"no remakes_{PRIMARY}.json with {ANALYSIS!r} in", RESULTS, ": nothing to draw yet")
    sys.exit(0)
runs = [(m, z) for m in MODELS if files[m[0]] for z in [units_z(files[m[0]], m[0])] if z is not None]
z_units = {m[0]: z for m, z in runs}

words, null = load("remakes_words.json"), load("remakes_null_size.json")
if not (words and null):
    sys.exit(f"remakes_words.json and remakes_null_size.json are needed in {RESULTS}")
cal = words["primary_calibration"]
standins = sorted(row["Z"] for row in cal["standins"]["rows"].values())
BAR = cal["standins"]["summary"]["z_max"]
src = "remakes_words.json: primary_calibration"
print(f"  {src}.standins.rows.*.Z: {len(standins)} stand-ins, from {standins[0]:.4f} to {standins[-1]:.4f}; "
      f".standins.summary.z_max = {BAR:.4f} ({cal['standins']['summary']['z_max_measure']})")
assert len(standins) == cal["standins"]["summary"]["n"] and standins[-1] == BAR
generators = cal["random_families_of_stand_ins"]["generators"]
random_z = sorted(z for g in generators.values() for z in g["Z"])
print(f"  {src}.random_families_of_stand_ins.generators.*.Z: {len(random_z)} random word-only measures from "
      f"{len(generators)} generators, from {random_z[0]:.4f} to {random_z[-1]:.4f}")
assert len(random_z) == cal["random_families_of_stand_ins"]["all"]["draws"]
noise_z = sorted(null["by_columns"][NOISE]["Z_sorted"])
print(f'  remakes_null_size.json: by_columns["{NOISE}"].Z_sorted: {len(noise_z)} draws of pure noise, from '
      f"{noise_z[0]:.4f} to {noise_z[-1]:.4f}")
assert len(noise_z) == null["draws"]
# the same bar is stored with the noise draws and with every model's decision
bars = {"remakes_null_size.json: stand_ins_largest_Z": null["stand_ins_largest_Z"]}
for tag in z_units:
    stored = (files[tag].get("decision") or {}).get("standins_largest_Z")
    if stored is not None:
        bars[f"remakes_{tag}.json: decision.standins_largest_Z"] = stored
for where, value in bars.items():
    if abs(value - BAR) > 1e-9:
        print(f"WARNING: {where} is {value}, not the {BAR} drawn")

emb = ((primary.get("embedding") or {}).get("in_the_units_place") or {}).get("slope", {}).get("Z")
if emb is not None:
    print(f"  remakes_{PRIMARY}.json: embedding.in_the_units_place.slope.Z = {emb:.4f}")

a = primary["analyses"][ANALYSIS]
sums = a["family_sums"]
by_family = {}
for e in a["entries_table"]:
    by_family.setdefault(e["family"], []).append(e)
families = sorted((r for r in sums["rows"] if r["D"] != 0), key=lambda r: -r["D"])
recognised = {r["family"]: any(e["recognised"] for e in by_family[r["family"]]) for r in families}
positive, negative = sum(r["D"] > 0 for r in families), sum(r["D"] < 0 for r in families)
big_positive, big_negative = sum(r["D"] > 0.1 for r in families), sum(r["D"] < -0.1 for r in families)   # what the figure states
n_rec = sum(recognised.values())
src = f'remakes_{PRIMARY}.json: analyses["{ANALYSIS}"]'
print(f"  {src}.family_sums.rows.*.D: {len(families)} of {sums['families']} families with a non-zero sum, "
      f"{positive} above zero and {negative} below, from {families[-1]['D']:.4f} to {families[0]['D']:.4f}")
print(f"  {src}.entries_table.*.recognised: at least one recognised pair in {n_rec} of those families, "
      f"none in {len(families) - n_rec}")
print(f"  {src}.family_sums.share_of_the_sum_of_squares_in_the_five_largest = "
      f"{sums['share_of_the_sum_of_squares_in_the_five_largest']:.4f}")
assert len(families) == sums["families_with_a_non_zero_sum"]
assert (positive, negative) == (a["slope"]["families_positive"], a["slope"]["families_negative"])
assert sums["families"] == a["slope"]["families"]


def short_name(row):
    """The family's remake that contributes most, without the bracketed suffix of its page title."""
    entry = max(by_family[row["family"]], key=lambda e: abs(e["contribution"]))
    assert entry["remake"] in row["remakes"]
    return re.sub(r"\s*\([^)]*\)$", "", entry["remake"])


named = [(i, r, short_name(r)) for i, r in enumerate(families[:NAMED], start=1)]
for i, r, name in named:
    print(f"  {src}.family_sums.rows (family {r['family']}): D = {r['D']:.4f}, named {name!r} from "
          f"remakes {r['remakes']}, {'recognised' if recognised[r['family']] else 'not recognised'}")

# how much of the test's total the largest families add (the file's own subsets leave each of them out in turn)
left_out = a["subsets"]["each_of_the_largest_families_left_out"]
K = len(left_out)
assert {str(r["family"]) for r in families[:K]} == set(left_out), "the families left out in turn are the largest"
total = a["slope"]["T"]
assert abs(sum(r["D"] for r in families) - total) < 1e-9
share_largest = sum(r["D"] for r in families[:K]) / total
z_left_out = sorted(v["Z"] for v in left_out.values())
print(f"  {src}.family_sums.rows.*.D and .slope.T: the {K} largest families add {share_largest:.4f} of the total "
      f"{total:.4f}")
print(f"  {src}.subsets.each_of_the_largest_families_left_out.*.Z: from {z_left_out[0]:.4f} to {z_left_out[-1]:.4f}")
title = headline(z_units, BAR)
subtitle = SUBTITLE.format(pairs=a["slope"]["entries"], primary=PRIMARY_NAME)
print(f"  {src}.slope.entries = {a['slope']['entries']}")
others = {"the stand-ins": standins[-1], "the random word-only measures": random_z[-1], "the noise draws": noise_z[-1]}
for tag, z in z_units.items():
    print(f"  {tag}: Z {z:.4f} is " + ", ".join(f"{'above' if z > top else 'not above'} every one of {what} "
          f"(largest {top:.4f})" for what, top in others.items())
          + f"; noise draws at or above it: {sum(z0 >= z for z0 in noise_z)} of {len(noise_z)}")
print("title:", title)
print("subtitle:", subtitle.replace("\n", " / "))
print("caption:", CAPTION.format(
    pairs=a["slope"]["entries"], primary=PRIMARY_NAME, z=num(z_units[PRIMARY]),
    others=CAPTION_OTHERS.format(list=", ".join(f"{m[1]} {num(z)}" for m, z in runs[1:])) if runs[1:] else "",
    n_standins=len(standins), bar=num(BAR), n_random=len(random_z), random_max=num(random_z[-1]),
    n_noise=len(noise_z), noise_max=num(noise_z[-1]),
    embedding=CAPTION_EMBEDDING.format(z=num(emb)) if emb is not None else "", shown=len(families),
    families=sums["families"], positive=big_positive, negative=big_negative, k=K, share=f"{share_largest:.0%}",
    left_out=CAPTION_LEFT_OUT.format(lo=num(z_left_out[0]), hi=num(z_left_out[-1])),
    n_not=len(families) - n_rec, n_rec=n_rec))

# ---- layout, in inches; in panel A, y runs downwards from the top of the plot ----
W = style.WIDTH_IN - 0.16                       # the tight crop adds 0.08 in on each side
A_LABEL, A_PLOT, A_RIGHT = 1.05, 2.20, 0.58     # panel A: row labels; plot; the column of values
SPACE, B_LABEL = 0.18, 0.38                     # between the panels; panel B's axis label and tick labels
B_PLOT = W - 0.03 - (A_LABEL + A_PLOT + A_RIGHT + SPACE + B_LABEL)
BOTTOM, HEAD = 0.50, 0.50                       # tick and axis labels; panel headings and the bar's label
PAD, PITCH, GROUP, ROW, HIST = 0.16, 0.26, 0.36, 0.31, 0.75  # ends; model rows; to the group label; rows; histogram
TICK = 0.055                                    # half the height of a tick

ys = [PAD + k * PITCH for k in range(len(runs))]
y_group = ys[-1] + GROUP
y = y_group - 0.04
rows_y = {}
for key in (["embedding"] if emb is not None else []) + ["standins", "random"]:
    y += ROW
    rows_y[key] = y
hist_top = y + 0.24
ax_h = hist_top + HIST
H = (ax_h + BOTTOM + HEAD + 0.14 * (subtitle.count("\n") + 1) + 0.20) / 0.94  # style.title sets the subtitle 6% down
fig = plt.figure(figsize=(W, H))
fig.add_axes([0, 0.5, 1, 1e-3], zorder=-1).axis("off")  # an empty line across the canvas: the crop keeps its width
style.title(fig, title, subtitle)
axa = fig.add_axes([A_LABEL / W, BOTTOM / H, A_PLOT / W, ax_h / H])
axb = fig.add_axes([(A_LABEL + A_PLOT + A_RIGHT + SPACE + B_LABEL) / W, BOTTOM / H, B_PLOT / W, ax_h / H])
head = dict(va="baseline", fontsize=FS, color=style.INK_2, linespacing=1.15)
Y_HEAD = -0.07

# ---- A: the units against everything that has no access to them ----
drawn = noise_z + standins + random_z + list(z_units.values()) + ([emb] if emb is not None else [])
X0, X1 = min(drawn) - 0.4, max(drawn) + 0.4
axa.set_xlim(X0, X1)
axa.set_ylim(ax_h, 0)
axa.spines[:].set_visible(False)
axa.set_yticks([])
step = 1 if X1 - X0 <= 9 else 2
ticks = [t for t in range(ceil(X0 / step) * step, floor(X1) + 1, step)]
axa.set_xticks(ticks)
axa.tick_params(axis="x", length=0, pad=4, labelsize=FS)
for t in ticks:
    axa.plot([t, t], [0, ax_h], color=style.RULE, lw=0.8, zorder=0.5)
axa.set_xlabel(X_LABEL_A, labelpad=4, fontsize=FS, linespacing=1.3)


def xa(inches):
    """Inches from the plot's left edge, as an x on panel A's axis (for the text beside the plot)."""
    return X0 + inches / A_PLOT * (X1 - X0)


X_TEXT, X_NAME, X_VALUE = xa(-A_LABEL + 0.01 * W), xa(-0.13), xa(A_PLOT + A_RIGHT)
row_label = dict(ha="right", va="center", fontsize=FS, color=style.INK_2, linespacing=1.15)
group_label = dict(ha="left", fontsize=7.5, color=style.INK)
value = dict(ha="right", va="center")
axa.text(X_TEXT, -HEAD + 0.08, "A", fontsize=9, fontweight="bold", va="center")
axa.text(xa(-A_LABEL + 0.01 * W + 0.18), -HEAD + 0.08, PANEL_A, fontsize=7.6, fontweight="bold", va="center")
axa.text(X_TEXT, Y_HEAD, GROUP_UNITS, va="baseline", **group_label)
axa.text(X_VALUE, Y_HEAD, "Z", ha="right", **head)

# the bar: the stand-ins' largest Z
axa.plot([BAR, BAR], [0, ax_h], color=style.INK_3, lw=0.8, ls=(0, (4, 2)), zorder=1)
axa.text(BAR, Y_HEAD, BAR_LABEL.format(z=num(BAR)), ha="center", **head)

# the units of each model: a mark in its row and a line down to the axis
for ((tag, name, face, edge), z), y in zip(runs, ys):
    first = tag == PRIMARY
    axa.plot([z, z], [y, ax_h], color=edge, lw=1.4 if first else 0.8, zorder=3 if first else 2.5,
             solid_capstyle="butt")
    axa.plot(z, y, marker="o", ms=7.5, mfc=face, mec=edge, mew=1.3, zorder=4, clip_on=False)
    axa.text(X_NAME, y, name, **{**row_label, "color": style.INK if first else style.INK_2})
    axa.text(X_VALUE, y, num(z), fontsize=7.5, color=style.INK, fontweight="bold" if first else "normal", **value)

# what went through the test in the units' place
axa.text(X_TEXT, y_group, GROUP_PLACE, va="center", **group_label)
if emb is not None:  # a comparator, not a study model: a neutral grey diamond
    y = rows_y["embedding"]
    axa.plot([emb, emb], [y, ax_h], color=style.INK_3, lw=0.6, zorder=2, solid_capstyle="butt")
    axa.plot(emb, y, marker="D", ms=6.0, mfc=style.INK_3, mec=style.INK_3, mew=1.0, zorder=4)
    axa.text(X_NAME, y, ROW_EMBEDDING, **row_label)
    axa.text(X_VALUE, y, num(emb), fontsize=FS, color=style.INK_3, **value)
for key, zs, label, colour, lw in (("standins", standins, ROW_STANDINS, style.INK_3, 0.7),
                                   ("random", random_z, ROW_RANDOM, GREY, 0.5)):
    y = rows_y[key]
    axa.vlines(zs, y - TICK, y + TICK, color=colour, lw=lw, zorder=2)
    axa.text(X_NAME, y, label.format(n=len(zs)), **row_label)
    axa.text(X_VALUE, y, UP_TO.format(z=num(zs[-1])), fontsize=FS, color=style.INK_3, **value)

BIN = 0.25
edges = [k * BIN for k in range(floor(noise_z[0] / BIN), ceil(noise_z[-1] / BIN) + 1)]
counts = [sum(lo <= z < hi for z in noise_z) for lo, hi in zip(edges, edges[1:])]
assert sum(counts) == len(noise_z)
tops = [ax_h - c / max(counts) * (HIST - 0.03) for c in counts]
axa.stairs(tops, edges, baseline=ax_h, fill=True, fc=style.OFF, ec="none", zorder=1.5)
axa.stairs(tops, edges, baseline=ax_h, fill=False, ec=GREY, lw=0.6, zorder=1.6)
axa.text(X_NAME, hist_top + HIST / 2, ROW_NOISE.format(n=len(noise_z)), **row_label)
axa.text(X_VALUE, hist_top + HIST / 2, UP_TO.format(z=num(noise_z[-1])), fontsize=FS, color=style.INK_3, **value)

# ---- B: where the primary model's score comes from, family by family ----
n = len(families)
tone = {False: style.NIGHT, True: GREY}        # no pair recognised; at least one
D = [r["D"] for r in families]
axb.set_xlim(0, n + 1)
axb.set_ylim(min(D) - 0.3, max(D) + 0.3)
axb.spines[:].set_visible(False)
axb.set_xticks([])
axb.yaxis.set_major_locator(MultipleLocator(1))
axb.tick_params(axis="y", length=0, pad=3, labelsize=FS)
axb.grid(axis="y", color=style.RULE, linewidth=0.6, zorder=0)
axb.set_axisbelow(True)
axb.axhline(0, color=style.INK_3, lw=0.6, zorder=1)
axb.vlines(range(1, n + 1), 0, D, color=[tone[recognised[r["family"]]] for r in families],
           lw=0.62 * B_PLOT * 72 / (n + 1), zorder=2)
axb.set_ylabel(Y_LABEL_B, labelpad=3, fontsize=FS)
axb.set_xlabel(X_LABEL_B.format(shown=n, families=sums["families"]), labelpad=6, fontsize=FS, linespacing=1.3,
               x=0.5 - B_LABEL / B_PLOT / 2)  # centred under the panel, axis labels included
axb.text(-B_LABEL / B_PLOT + 0.01, 1 + (HEAD - 0.08) / ax_h, "B", fontsize=9, fontweight="bold", va="center",
         transform=axb.transAxes)
axb.text(-B_LABEL / B_PLOT + 0.01 + 0.18 / B_PLOT, 1 + (HEAD - 0.08) / ax_h, PANEL_B.format(primary=PRIMARY_NAME),
         fontsize=7.6, fontweight="bold", va="center", transform=axb.transAxes)
note = dict(fontsize=6.8, color=style.INK_2, linespacing=1.25, zorder=1.5,
            bbox=dict(fc="white", ec="none", pad=0.6))  # the grid lines stop at the text
for i, r, name in named:  # the largest families, named beside the top of their bars
    axb.annotate(name, (i, r["D"]), xytext=(4, -1), textcoords="offset points", va="bottom", **note)


def hung(dx, dy):
    """x in fractions of panel B's width, y on its axis, moved by (dx, dy) points: text hung under a grid line."""
    return offset_copy(axb.get_yaxis_transform(), fig=fig, x=dx, y=dy, units="points")


# what the two tones mean: three lines between the two highest grid lines, right of the names
KEY_X, KEY_Y, LINE = 0.74 / B_PLOT, floor(max(D)), 9.2
axb.text(KEY_X, KEY_Y, KEY, ha="left", va="center", transform=hung(0, -8), **note)
for k, (flag, text) in enumerate(((False, KEY_NOT.format(n=n - n_rec)), (True, KEY_REC.format(n=n_rec))), start=1):
    axb.plot(KEY_X, KEY_Y, marker="s", ms=4.6, mfc=tone[flag], mec="none", transform=hung(2.5, -8 - k * LINE))
    axb.text(KEY_X, KEY_Y, text, ha="left", va="center", transform=hung(9, -8 - k * LINE), **note)
# how many families are on each side of zero, over the run of bars that cannot be seen
small = sum(abs(d) < 0.01 * max(map(abs, D)) for d in D)  # bars under a hundredth of the largest
axb.text(1.0, 1, LARGEST.format(k=K, share=f"{share_largest:.0%}") + COUNTS.format(positive=big_positive, negative=big_negative)
         + (COUNTS_SMALL if small > n / 2 else ""), ha="right", va="top", transform=hung(0, -4), **note)
print(f"  panel B: {small} of the {n} bars are under a hundredth of the largest")

box = fig.get_tightbbox()
if box.x0 < -0.004 or box.x1 > W + 0.004:  # the PNG must stay at the text width
    print(f"WARNING: something is drawn outside the canvas: from {box.x0:.3f} to {box.x1:.3f} in of {W:.2f}")
style.save(fig, "fig_remakes.png")
