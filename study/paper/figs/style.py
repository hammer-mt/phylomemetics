"""Shared look for the paper's figures: palette, type and saving. Every figure script imports this."""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# figures are written next to the paper.md they belong to: paper/ (v1, published and frozen) by default,
# or a draft version folder, e.g. PAPER_DIR=paper/v2 python paper/figs/canon.py
OUT = Path(os.environ["PAPER_DIR"]).resolve() if os.environ.get("PAPER_DIR") else Path(__file__).resolve().parent.parent

INK = "#141821"
INK_2 = "#444b59"
INK_3 = "#5c6372"
RULE = "#dfe2e8"
NIGHT = "#1f3a64"      # Batman's own memes, the primary model
NIGHT_SOFT = "#c9d5e8"
SIGNAL = "#f2b705"     # one highlight at most per figure
SHARED = "#a3a9b5"     # memes shared with other heroes; cousins
OFF = "#eceef2"        # absent cells
# era groups, in order of appearance (same hues as the companion page)
ERA = {"Early (1943–66)": "#2a78d6", "Grim & gritty comics (1986–96)": "#e34948", "Burton & Schumacher films": "#6250d6",
       "Animated films": "#e87ba4", "Modern comics (2002–17)": "#eb6834", "Nolan trilogy": "#1baf7a",
       "Arkham games": "#eda100", "Recent films (2016–22)": "#008300", "Cousins & ancestors": "#a3a9b5"}

WIDTH_IN = 6.72      # full text width of the A4 PDF (the existing figures are 1344 px at 200 dpi)
DPI = 200


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.labelsize": 7.5,
        "axes.edgecolor": RULE, "axes.labelcolor": INK_2, "axes.linewidth": 0.6,
        "xtick.color": INK_3, "ytick.color": INK_3, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "legend.fontsize": 7,
    })


def title(fig, text, sub=None):
    """Left-aligned figure title and optional subtitle, in the style of the existing figures."""
    fig.text(0.01, 0.985, text, ha="left", va="top", fontsize=10, fontweight="bold", color=INK)
    if sub:
        fig.text(0.01, 0.94, sub, ha="left", va="top", fontsize=7.5, color=INK_2)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", path)
    return path
