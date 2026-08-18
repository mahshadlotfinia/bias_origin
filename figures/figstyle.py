"""
figures/figstyle.py
Created on August 13, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import numpy as np
import matplotlib.pyplot as plt

TABLES = "/PATH/Repositories_target_files/bias_origin/final_tables"
FIGS = "/PATH/manuscripts/bias_origin/figs"

RACE = "#2166AC"
AGE = "#B2182B"
ACHIEVABLE = "#92C5DE"
DECODABILITY = "#4393C3"
DECODABILITY_NL = "#A6BDDB"
PERFORMANCE = "#4D4D4D"
REF_GREY = "#999999"
FAINT = "#C8C8C8"
NEUTRAL = "#777777"
HAIRLINE = "#EDEDED"
TEXT = "#222222"

RCPARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 18.0,
    "axes.labelsize": 17.5,
    "axes.titlesize": 19.5,
    "xtick.labelsize": 14.5,
    "ytick.labelsize": 14.5,
    "legend.fontsize": 15.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.15,
    "xtick.major.width": 1.05,
    "ytick.major.width": 1.05,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def set_rcparams(scale=1.0):
    d = dict(RCPARAMS)
    for k in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
              "ytick.labelsize", "legend.fontsize"):
        d[k] = round(d[k] * scale, 2)
    d["_scale"] = scale
    plt.rcParams.update({k: v for k, v in d.items() if not k.startswith("_")})
    global SCALE
    SCALE = scale


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


SCALE = 1.0


def fs(size):
    return round(size * SCALE, 2)


def place_labels(fig, labels, gap=0.019, lift=0.006):
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter, title in labels:
        boxes = [ax.get_window_extent(r)]
        if ax.yaxis.label.get_text():
            boxes.append(ax.yaxis.label.get_window_extent(r))
        lo, hi = sorted(ax.get_ylim())
        for tick, loc in zip(ax.yaxis.get_ticklabels(), ax.yaxis.get_ticklocs()):
            if tick.get_text().strip() and lo <= loc <= hi:
                boxes.append(tick.get_window_extent(r))
        x0 = min(b.x0 for b in boxes)
        y1 = max(b.y1 for b in boxes)
        fx, fy = inv.transform((x0, y1))
        fig.text(fx, fy + lift, letter, fontsize=25 * SCALE, fontweight="bold",
                 ha="left", va="bottom", color=TEXT)
        fig.text(fx + gap, fy + lift, title, fontsize=19.5 * SCALE,
                 ha="left", va="bottom", color=TEXT)


def annotate_free_corner(ax, text, fontsize=14.5, pad=0.035):
    fontsize = fs(fontsize)
    pts = []
    for ln in ax.lines:
        d = ln.get_xydata()
        if len(d):
            d = np.asarray(d, dtype=float)
            dense = [d]
            for a, b in zip(d[:-1], d[1:]):
                dense.append(a + np.linspace(0, 1, 12)[:, None] * (b - a))
            pts.append(ax.transLimits.transform(np.vstack(dense)))
    for co in ax.collections:
        off = co.get_offsets()
        if off is not None and len(off):
            pts.append(ax.transLimits.transform(np.asarray(off)))
    corners = {("left", "top"): (pad, 1 - pad), ("right", "top"): (1 - pad, 1 - pad),
               ("left", "bottom"): (pad, pad), ("right", "bottom"): (1 - pad, pad)}
    if pts:
        P = np.vstack(pts)
        P = P[np.isfinite(P).all(axis=1)]
        best, best_d = None, -1.0
        for (ha, va), (x, y) in corners.items():
            d = np.hypot(P[:, 0] - x, P[:, 1] - y).min() if len(P) else 1.0
            if d > best_d:
                best, best_d = (ha, va, x, y), d
        ha, va, x, y = best
    else:
        ha, va, x, y = "left", "top", pad, 1 - pad
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize, color=TEXT, ha=ha, va=va)
