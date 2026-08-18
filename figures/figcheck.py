"""
figures/figcheck.py
Created on August 13, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import numpy as np
from matplotlib.collections import QuadMesh
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Annotation, Text


def _visible_texts(fig, renderer):
    out = []
    for ax in fig.axes:
        on = ax.axison
        for t in ([ax.title, ax.xaxis.label, ax.yaxis.label] if on else []):
            if t.get_text().strip() and t.get_visible():
                out.append((ax, t))
        if on:
            for axis, lo, hi in ((ax.xaxis, *sorted(ax.get_xlim())), (ax.yaxis, *sorted(ax.get_ylim()))):
                if not axis.get_visible():
                    continue
                for tick, loc in zip(axis.get_ticklabels(), axis.get_ticklocs()):
                    if tick.get_text().strip() and lo <= loc <= hi:
                        out.append((ax, tick))
        for t in ax.texts:
            if t.get_text().strip() and t.get_visible():
                out.append((ax, t))
        leg = ax.get_legend()
        if leg is not None:
            out += [(ax, t) for t in leg.get_texts() if t.get_text().strip()]
    out += [(None, t) for t in fig.texts if t.get_text().strip() and t.get_visible()]
    return [(ax, t, (Text.get_window_extent(t, renderer) if isinstance(t, Annotation)
                     else t.get_window_extent(renderer))) for ax, t in out]


def _patch_boxes(fig, renderer):
    out = []
    for ax in fig.axes:
        if getattr(ax, "_figcheck_skip", False):
            continue
        for p in ax.patches:
            if not isinstance(p, FancyBboxPatch):
                continue
            if not p.get_visible() or getattr(p, "_figcheck_skip", False):
                continue
            fc = p.get_facecolor()
            if len(fc) == 4 and fc[3] < 0.02:
                continue
            try:
                out.append((ax, p, p.get_window_extent(renderer)))
            except Exception:
                pass
    return out


def _overlap(a, b):
    return not (a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0)


def check(fig, pad=0.5):
    renderer = fig.canvas.get_renderer()
    texts = _visible_texts(fig, renderer)
    bad = []

    for i in range(len(texts)):
        ax_i, t_i, b_i = texts[i]
        for j in range(i + 1, len(texts)):
            ax_j, t_j, b_j = texts[j]
            if ax_i is ax_j and t_i.get_rotation() and t_j.get_rotation():
                continue
            if _overlap(b_i.expanded(1 + pad / 100, 1 + pad / 100), b_j):
                bad.append(f"text on text: {t_i.get_text()[:32]!r} / {t_j.get_text()[:32]!r}")

    fb = fig.bbox
    for _, t, b in texts:
        if b.x0 < fb.x0 - 1 or b.x1 > fb.x1 + 1 or b.y0 < fb.y0 - 1 or b.y1 > fb.y1 + 1:
            bad.append(f"text outside the canvas: {t.get_text()[:40]!r}")

    panel_pts = {}
    for ax in fig.axes:
        if not ax.axison or getattr(ax, "_figcheck_skip", False):
            continue
        pts = []
        for ln in ax.lines:
            if ln.get_transform() is not ax.transData:
                continue
            d = ln.get_xydata()
            if len(d):
                pts.append(ax.transData.transform(d))
        for co in ax.collections:
            if isinstance(co, QuadMesh):
                continue
            off = co.get_offsets()
            if off is not None and len(off):
                pts.append(ax.transData.transform(np.asarray(off)))
        if pts:
            P = np.vstack(pts)
            box = ax.get_window_extent(renderer)
            inside = ((P[:, 0] >= box.x0) & (P[:, 0] <= box.x1)
                      & (P[:, 1] >= box.y0) & (P[:, 1] <= box.y1))
            if inside.any():
                panel_pts[ax] = P[inside]

    mpad = 8.0
    for ax_t, t, b in texts:
        for ax_d, P in panel_pts.items():
            inside = ((P[:, 0] > b.x0 - mpad) & (P[:, 0] < b.x1 + mpad)
                      & (P[:, 1] > b.y0 - mpad) & (P[:, 1] < b.y1 + mpad))
            if inside.sum():
                where = "its own panel" if ax_d is ax_t else "another panel"
                bad.append(f"text on data ({where}): {t.get_text()[:32]!r} covers {int(inside.sum())} points")

    boxes = [pb for _, _, pb in _patch_boxes(fig, renderer) if pb.width > 0 and pb.height > 0]
    for ax_t, t, tb in texts:
        touched = [pb for pb in boxes
                   if not (tb.x1 <= pb.x0 or tb.x0 >= pb.x1 or tb.y1 <= pb.y0 or tb.y0 >= pb.y1)]
        if not touched:
            continue
        contains = [pb for pb in touched
                    if tb.x0 >= pb.x0 - 1 and tb.x1 <= pb.x1 + 1
                    and tb.y0 >= pb.y0 - 1 and tb.y1 <= pb.y1 + 1]
        if contains:
            smallest_container = min(pb.width * pb.height for pb in contains)
            crossed = [pb for pb in touched
                       if pb not in contains
                       and pb.width * pb.height < 0.92 * smallest_container]
            if crossed:
                bad.append("text overruns a smaller box: %r" % t.get_text()[:38])
                continue
        if contains:
            continue
        bad.append(f"text across a box edge: {t.get_text()[:32]!r}")

    for ax in fig.axes:
        if not ax.axison:
            continue
        lo, hi = sorted(ax.get_ylim())
        span = hi - lo
        for ln in ax.lines:
            if ln.get_transform() is not ax.transData:
                continue
            y = np.asarray(ln.get_ydata(), dtype=float)
            y = y[np.isfinite(y)]
            if len(y) and (y.min() < lo - span or y.max() > hi + span):
                bad.append(f"line far outside its own limits: {ln.get_label()}")
    return bad


def report(fig, name):
    bad = check(fig)
    print(f"[figcheck] {name}: {'clean' if not bad else str(len(bad)) + ' issue(s)'}")
    for b in dict.fromkeys(bad):
        print("   -", b)
    return bad
