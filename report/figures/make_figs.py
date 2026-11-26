"""Figures for report/main.tex. Run from the repo root:

    uv run --with matplotlib --with numpy report/figures/make_figs.py

Reads only files already in the repo; computes nothing new:
  fig_guards.pdf  deck/assets/rb_before_after.json (bin/bluff verdicts, 132 T'=3 classes)
  fig_tilt.pdf    redblack/rb-track-out/gbook/tvs4_sigma0.5.jsonl (tilted_vs_served.py, sigma 0.5,
                  20 tilts per class; values already in exploitability = nash_conv / 2).
                  No longer used by main.tex (the endgame book is not served); kept for the archive.
  fig_bound.pdf   the BOUND_ROWS table below, copied by hand from board/rb-track/bound-ledger.md
                  (upper bounds and probe lower bounds) and board/rb-track/progress.md (probe,
                  2026-11-21 03:50). Sum units (the papers' eps_G, cap 2).
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def fig_guards():
    d = json.loads((ROOT / "deck/assets/rb_before_after.json").read_text())
    rows = sorted(d["rows"], key=lambda r: (r["old"], r["new"]), reverse=True)
    names = [("old", "Old guards\n(served to Nov 21)"),
             ("new", "Fixed guards"),
             ("book", "Endgame book\n(certified, not served)")]
    floor = 1e-7
    m = np.array([[max(r[k], floor) for r in rows] for k, _ in names])
    cmap = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#bcd5f3", "#5d9be3", "#2a78d6", "#123d73"])
    fig, ax = plt.subplots(figsize=(6.5, 1.05))
    im = ax.imshow(m, aspect="auto", cmap=cmap, norm=LogNorm(vmin=floor, vmax=0.3), interpolation="nearest")
    for y in (0.5, 1.5):
        ax.axhline(y, color="white", lw=2)
    ax.set_yticks(range(3), [n for _, n in names])
    ax.set_xticks([])
    ax.set_xlabel("132 three-card endgames, sorted by exploitability under the old guards", labelpad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    i = next(j for j, r in enumerate(rows) if r["class"] == "d0-3.2.1-1.0.1")
    ax.annotate("d0-3.2.1-1.0.1: 0.129 old, 0.0045 fixed", xy=(i, -0.5), xytext=(i + 6, -1.15),
                fontsize=6.5, color=INK2, annotation_clip=False,
                arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6})
    med = [statistics.median(r[k] for r in d["rows"]) for k, _ in names]
    for y, v in enumerate(med):
        ax.text(133.5, y, f"median {v:.3g}" if v > 1e-3 else f"median {v:.1e}",
                va="center", fontsize=6.5, color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.14, ticks=[1e-7, 1e-5, 1e-3, 1e-1])
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_visible(False)
    cb.set_label("exploitability", fontsize=6.5, color=INK2)
    fig.savefig(OUT / "fig_guards.pdf")
    fig.savefig(OUT / "fig_guards.png", dpi=220)


def fig_tilt():
    rows = [json.loads(line) for line in (ROOT / "redblack/rb-track-out/gbook/tvs4_sigma0.5.jsonl").open()]
    rows = [r for r in rows if r["layer"] == 3]
    assert len(rows) == 132
    floor = 1e-8
    series = [("book", "Tight LP book", BLUE), ("cfr", "CFR+ book (served)", ORANGE)]
    panels = [("prior", "At the class prior\n(where both are certified)"),
              ("tilted_median", "At tilted entry beliefs\n(median of 20 tilts per endgame)")]
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 1.9), sharey=True)
    for ax, (pk, title) in zip(axes, panels):
        for xi, (sk, label, col) in enumerate(series):
            ys = np.array([max(r[sk][pk], floor) for r in rows])
            xs = xi + rng.uniform(-0.28, 0.28, len(ys))
            ax.scatter(xs, ys, s=7, color=col, alpha=0.55, linewidths=0)
            med = statistics.median(r[sk][pk] for r in rows)
            ax.plot([xi - 0.36, xi + 0.36], [med, med], color=INK, lw=1.4, solid_capstyle="round")
            txt = f"{med:.1e}" if med < 1e-3 else f"{med:.3f}"
            ax.text(xi + 0.4, med, txt, va="center", fontsize=6.5, color=INK)
        ax.set_yscale("log")
        ax.set_ylim(3e-8, 0.6)
        ax.set_xlim(-0.6, 1.9)
        ax.set_xticks([0, 1], [s[1] for s in series])
        ax.set_title(title, fontsize=7.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("exploitability (log)")
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(OUT / "fig_tilt.pdf")
    fig.savefig(OUT / "fig_tilt.png", dpi=220)


# One row per bot the site has served, oldest first:
#   (when, what the bot played, certified bound or None, probe take or None, probe SE,
#    True when the probe reading is interim: it then gets an asterisk, explained in the caption)
# Sources: bound-ledger.md "Upper bounds" (0.6992, 0.30206458009, 0.2522332185, 0.2197367419),
# "Probe lower bounds" (0.232, 0.2632, 0.2407) and the r4y deploy record (0.189, SE 0.045, at 900
# of 2,000 pairs); progress.md probe 2026-11-21 03:50 (a_old 1.1880 SE 0.0201, c_noprune_book
# 1.1227 SE 0.0206, 3,000 pairs each).
# REFRESH: the LAST row is what prod serves (store r4y, since 2026-11-23 12:55 ET). If a later
# store replaces it, drop "(served now)" from that row and append the new one (take None while it
# has no probe reading); when the r4y probe finishes its 2,000 pairs, replace 0.2097 / 0.0356 (1,450 pairs), set
# its flag to False and drop the "starred reading" sentence from the caption in main.tex.
BOUND_ROWS = [
    ("to Nov 21", "per-round solver, pruning guards", None, 1.1880, 0.0201, False),
    ("Nov 21", "guards fixed, endgame book", None, 1.1227, 0.0206, False),
    ("Nov 22", "told-type policy over the book", 0.6992, 0.232, 0.030, False),
    ("Nov 23", "told-type policy everywhere", 0.3021, 0.2632, 0.0192, False),
    ("Nov 23", "hand hidden for one round", 0.2522, 0.2407, 0.0249, False),
    ("Nov 23", "hidden for two rounds", 0.2197, 0.2097, 0.0356, False),
    ("Nov 23", "hidden at every round end", 0.1953, 0.2560, 0.0429, False),
    ("Nov 23", "keeps drawing (served now)", 0.1525, 0.0880, 0.0502, True),
]


def fig_bound():
    rows = BOUND_ROWS
    n = len(rows)
    xmax = 1.3
    fig, ax = plt.subplots(figsize=(6.5, 1.6))
    fig.subplots_adjust(left=0.345, right=0.8, top=0.86, bottom=0.2)
    half = 0.3
    for i, (when, what, bound, take, se, interim) in enumerate(rows):
        y = n - 1 - i
        if bound is not None:
            ax.fill_between([bound, xmax], y - half, y + half, color="#e3edfa", lw=0, zorder=1)
            ax.plot([bound, bound], [y - half, y + half], color=BLUE, lw=2, solid_capstyle="butt", zorder=3)
        if take is not None:
            ax.plot([take - 1.96 * se, take + 1.96 * se], [y, y], color=ORANGE, lw=1.1, zorder=4,
                    solid_capstyle="butt")
            ax.scatter([take], [y], s=17, color=ORANGE, edgecolors="white", linewidths=0.6, zorder=5)
        ax.text(-0.755, y, when, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=7, color=INK2, clip_on=False)
        ax.text(-0.595, y, what, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=7, color=INK, clip_on=False)
        probe = "not yet run" if take is None else f"{take:.3f} ({se:.3f}){'*' if interim else ''}"
        ax.text(1.03, y, "none" if bound is None else f"{bound:.4f}", ha="left", va="center",
                transform=ax.get_yaxis_transform(), fontsize=7, color=MUTED if bound is None else INK,
                clip_on=False)
        ax.text(1.16, y, probe, ha="left", va="center", transform=ax.get_yaxis_transform(),
                fontsize=7, color=MUTED if take is None else INK, clip_on=False)
    top = n - 1 + 0.95
    ax.plot([1.035, 1.035], [top - 0.25, top + 0.25], color=BLUE, lw=2, solid_capstyle="butt",
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.text(1.05, top, "ceiling", ha="left", va="center", transform=ax.get_yaxis_transform(),
            fontsize=7, color=INK2, clip_on=False)
    ax.scatter([1.175], [top], s=17, color=ORANGE, edgecolors="white", linewidths=0.6,
               transform=ax.get_yaxis_transform(), clip_on=False, zorder=5)
    ax.text(1.195, top, "exploiter (SE)", ha="left", va="center",
            transform=ax.get_yaxis_transform(), fontsize=7, color=INK2, clip_on=False)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
    ax.tick_params(axis="x", length=2, labelsize=6.5)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("what a best-responding opponent gains, summed over both seats (0 to 2)", labelpad=2,
                  fontsize=7)
    fig.savefig(OUT / "fig_bound.pdf", bbox_inches=None)
    fig.savefig(OUT / "fig_bound.png", dpi=220, bbox_inches=None)


if __name__ == "__main__":
    fig_guards()
    fig_tilt()
    fig_bound()
