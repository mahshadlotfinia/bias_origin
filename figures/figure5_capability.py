"""
figures/figure5_capability.py
Created on August 13, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figures import figcheck, figstyle


class Figure5Capability:
    FONT_SCALE = 1.060
    OUT_PDF = os.path.join(figstyle.FIGS, "figure5_capability.pdf")

    ATTRS = {"race_grp": "Race", "age_grp": "Age", "sex_grp": "Sex"}
    COLORS = {"race_grp": figstyle.RACE, "age_grp": figstyle.AGE, "sex_grp": figstyle.PERFORMANCE}
    CONTRASTS = [("image_text_minus_ssl", "image-text vs\nself-supervised"),
                 ("image_text_minus_supervised", "image-text vs\nlabel supervision")]
    LEVELS = ["unfreeze_depth0", "unfreeze_depth1", "unfreeze_depth2", "unfreeze_depth3"]
    LEVEL_LABELS = ["linear\nhead", "last\nblock", "LoRA", "full\nfinetune"]
    OBJECTIVES = {"ssl": "self-supervised", "supervised": "label supervision", "image_text": "image-text"}

    def __init__(self, out_pdf=None):
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        figstyle.set_rcparams(Figure5Capability.FONT_SCALE)
        self._load_data()
        self.fig = None
        self._labels = []

    def _load_data(self):
        perf = pd.read_csv(os.path.join(figstyle.TABLES, "results_performance.csv"), low_memory=False)
        stat = pd.read_csv(os.path.join(figstyle.TABLES, "results_statistics.csv"), low_memory=False)

        self.contr = stat[(stat.experiment == "e1") & (stat.data_composition.astype(str) == "natural")
                          & (stat.estimate_name.isin(["auroc_worst_difference", "es_auc_difference"]))]

        e1 = perf[(perf.experiment == "e1") & (perf.data_composition == "natural")
                  & (perf.head_type == "linear")]
        self.matrix = e1.pivot_table(index=["encoder", "backbone", "finding", "attribute", "seed"],
                                     columns="metric_name", values="value_raw", aggfunc="first").reset_index()
        self.matrix_sd = e1[e1.metric_name.isin(["es_auc", "auroc_gap"])].set_index(
            ["encoder", "backbone", "finding", "attribute", "seed", "metric_name"])

        e6 = perf[perf.experiment == "e6"]
        self.unfreeze = e6.pivot_table(index=["encoder", "finding", "attribute", "data_composition"],
                                       columns="metric_name", values="value_raw", aggfunc="first").reset_index()

        n = len(self.contr[(self.contr.estimate_name == "auroc_worst_difference")
                           & (self.contr.encoder_objective == "image_text_minus_ssl")
                           & (self.contr.attribute == "race_grp")])
        if n != 52:
            raise ValueError(f"expected 52 comparisons per contrast, found {n}")

    def _gains(self, estimate, contrast, attr):
        s = self.contr[(self.contr.estimate_name == estimate)
                       & (self.contr.encoder_objective == contrast)
                       & (self.contr.attribute == attr)]
        return s.estimate.values, int(s.significant_fdr05.sum()), len(s)

    def _level(self, attr, metric):
        s = self.unfreeze[self.unfreeze.attribute == attr]
        return [s[s.data_composition == lv][metric].values for lv in self.LEVELS]

    def _panel_a(self, ax):
        figstyle.clean_axes(ax)
        rng = np.random.default_rng(0)
        for pos, a in enumerate(self.ATTRS):
            v, sig, n = self._gains("auroc_worst_difference", "image_text_minus_ssl", a)
            col = self.COLORS[a]
            parts = ax.violinplot([v], positions=[pos], widths=0.8, showextrema=False, vert=False)
            body = parts["bodies"][0]
            verts = body.get_paths()[0].vertices
            verts[:, 1] = np.clip(verts[:, 1], pos, None)
            body.set_facecolor(col); body.set_alpha(0.28); body.set_edgecolor(col); body.set_linewidth(1.2)
            ax.scatter(v, pos - 0.24 + rng.uniform(-0.08, 0.08, len(v)), s=20, facecolor=col,
                       alpha=0.55, edgecolor="none", zorder=3)
            ax.plot([np.median(v)] * 2, [pos - 0.06, pos + 0.34], color=col, lw=2.8, zorder=4)
            ax.text(verts[:, 0].max() + 0.004, pos, f"{np.median(v):.3f}", ha="left", va="center",
                    fontsize=figstyle.fs(12.5), color=col, zorder=5)
        ax.axvline(0, color=figstyle.REF_GREY, ls="--", lw=1.2, zorder=2)
        ax.set_yticks(range(len(self.ATTRS)))
        ax.set_yticklabels([self.ATTRS[a] for a in self.ATTRS], fontsize=figstyle.fs(13.5))
        ax.invert_yaxis()
        ax.set_ylim(len(self.ATTRS) - 0.25, -0.8)
        ax.set_xlabel("Gain in worst-group AUROC,\nimage-text over self-supervised")
        self._labels.append((ax, "a", "The gain from a stronger objective"))

    def _panel_b(self, ax):
        figstyle.clean_axes(ax)
        x = np.arange(len(self.ATTRS))
        for i, (contrast, clab) in enumerate(self.CONTRASTS):
            for j, a in enumerate(self.ATTRS):
                v, sig, n = self._gains("auroc_worst_difference", contrast, a)
                off = (i - 0.5) * 0.40
                ax.bar(x[j] + off, np.median(v), width=0.36, color=self.COLORS[a],
                       alpha=0.85 if i == 0 else 0.40, edgecolor=self.COLORS[a], linewidth=1.3, zorder=2)
                ax.plot([x[j] + off] * 2, [np.quantile(v, .25), np.quantile(v, .75)],
                        color=self.COLORS[a], lw=2.2, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([self.ATTRS[a] for a in self.ATTRS], fontsize=figstyle.fs(13.5))
        ax.set_xlim(-0.6, len(x) - 0.4)
        ax.set_ylim(0, 0.105)
        ax.set_ylabel("Median gain")
        handles = [Line2D([0], [0], lw=9, color=figstyle.PERFORMANCE, alpha=0.85, label="over self-supervised"),
                   Line2D([0], [0], lw=9, color=figstyle.PERFORMANCE, alpha=0.40, label="over label supervision")]
        ax.legend(handles=handles, frameon=False, loc="upper center", fontsize=figstyle.fs(12.0),
                  handletextpad=0.5, borderpad=0.2, bbox_to_anchor=(0.5, 1.02))
        self._labels.append((ax, "b", "Against either baseline"))

    def _panel_c(self, ax):
        figstyle.clean_axes(ax)
        rows = []
        for est, lab in [("auroc_worst_difference", "worst-group"), ("es_auc_difference", "equity-scaled")]:
            for a in self.ATTRS:
                _, sig, n = self._gains(est, "image_text_minus_ssl", a)
                rows.append((f"{self.ATTRS[a]}, {lab}", a, sig, n))
        y = np.arange(len(rows))
        for i, (lab, a, sig, n) in enumerate(rows):
            ax.barh(i, sig, height=0.68, color=self.COLORS[a], alpha=0.85, edgecolor="white",
                    linewidth=0.8, zorder=3)
            ax.barh(i, n - sig, left=sig, height=0.68, color=figstyle.FAINT, alpha=0.7,
                    edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(sig - 1.2, i, f"{sig}", ha="right", va="center", fontsize=figstyle.fs(12.5), color="white", zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=figstyle.fs(12.5))
        ax.invert_yaxis()
        ax.set_xlim(0, 52)
        ax.set_xticks([0, 13, 26, 39, 52])
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_xlabel("Comparisons of 52")
        self._labels.append((ax, "c", "Comparisons reaching significance"))

    def _panel_d(self, ax):
        figstyle.clean_axes(ax)
        grid = np.linspace(0.55, 0.92, 220)
        for a, sign in (("race_grp", 1), ("age_grp", 1)):
            col = self.COLORS[a]
            for i, v in enumerate(self._level(a, "es_auc")):
                v = v[np.isfinite(v)]
                if len(v) < 3:
                    continue
                kde = np.exp(-0.5 * ((grid[:, None] - v[None, :]) / 0.022) ** 2).sum(axis=1)
                kde = kde / kde.max() * 0.78
                base = (len(self.LEVELS) - 1 - i) * 1.0 + (0.0 if a == "race_grp" else 0.0)
                ax.fill_between(grid, base, base + kde, color=col, alpha=0.22 if a == "race_grp" else 0.16,
                                linewidth=0, zorder=2 + i)
                ax.plot(grid, base + kde, color=col, lw=1.6, zorder=2 + i)
                ax.plot([np.median(v)] * 2, [base, base + 0.32], color=col, lw=2.4, zorder=9)
        ax.set_yticks(np.arange(len(self.LEVELS))[::-1])
        ax.set_yticklabels(self.LEVEL_LABELS, fontsize=figstyle.fs(13.0))
        ax.set_ylim(-0.25, len(self.LEVELS) - 0.1)
        ax.set_xlim(0.55, 0.92)
        ax.set_xlabel("Equity-scaled AUROC")
        self._labels.append((ax, "d", "Unfreezing raises capability"))

    def _panel_e(self, ax):
        figstyle.clean_axes(ax)
        rng = np.random.default_rng(1)
        for a, off in (("race_grp", -0.18), ("age_grp", 0.18)):
            col = self.COLORS[a]
            for i, v in enumerate(self._level(a, "auroc_gap")):
                v = v[np.isfinite(v)]
                jitter = rng.uniform(-0.10, 0.10, len(v))
                ax.scatter(np.full(len(v), i + off) + jitter, v, s=26, facecolor=col, alpha=0.45,
                           edgecolor="none", zorder=3)
                ax.plot([i + off - 0.16, i + off + 0.16], [np.median(v)] * 2, color=col, lw=2.6, zorder=4)
        ax.set_xticks(range(len(self.LEVELS)))
        ax.set_xticklabels(self.LEVEL_LABELS, fontsize=figstyle.fs(12.5))
        ax.set_xlim(-0.6, len(self.LEVELS) - 0.4)
        ax.set_ylabel("Subgroup difference")
        self._labels.append((ax, "e", "The difference across levels"))

    def _panel_f(self, ax):
        figstyle.clean_axes(ax)
        med = [np.median(v[np.isfinite(v)]) for v in self._level("race_grp", "auroc_overall")]
        start = med[0]
        ax.bar(0, start, width=0.62, color=figstyle.PERFORMANCE, alpha=0.35,
               edgecolor=figstyle.PERFORMANCE, linewidth=1.4, zorder=3)
        ax.text(0, start + 0.002, f"{start:.3f}", ha="center", va="bottom", fontsize=figstyle.fs(12.5),
                color=figstyle.TEXT)
        run = start
        for i in range(1, len(med)):
            step = med[i] - med[i - 1]
            ax.bar(i, step, bottom=run, width=0.62, color=figstyle.RACE, alpha=0.55,
                   edgecolor=figstyle.RACE, linewidth=1.4, zorder=3)
            ax.plot([i - 0.31, i - 0.69], [run, run], color=figstyle.FAINT, lw=1.2, zorder=2)
            ax.text(i, run + step + 0.002, f"+{step:.3f}", ha="center", va="bottom", fontsize=figstyle.fs(12.5),
                    color=figstyle.RACE)
            run += step
        ax.bar(len(med), run, width=0.62, color=figstyle.PERFORMANCE, alpha=0.35,
               edgecolor=figstyle.PERFORMANCE, linewidth=1.4, zorder=3)
        ax.text(len(med), run + 0.002, f"{run:.3f}", ha="center", va="bottom", fontsize=figstyle.fs(12.5),
                color=figstyle.TEXT)
        ax.set_xticks(range(len(med) + 1))
        ax.set_xticklabels(["linear\nhead", "last\nblock", "LoRA", "full\nfinetune", "reached"],
                           fontsize=figstyle.fs(12.0))
        ax.set_xlim(-0.6, len(med) + 0.5)
        ax.set_ylim(0.74, 0.855)
        ax.set_ylabel("Overall disease AUROC")
        self._labels.append((ax, "f", "The gain between levels"))

    def _panel_g(self, axes):
        objs = ["ssl", "image_text"]
        cols = {"ssl": figstyle.ACHIEVABLE, "image_text": figstyle.RACE}
        rows = [("es_auc", "Equity-scaled AUROC"), ("auroc_gap", "Race difference")]
        for ax, (metric, mlab) in zip(axes, rows):
            figstyle.clean_axes(ax)
            lo_all, hi_all = [], []
            for y, o in enumerate(objs):
                r = self.matrix_sd.loc[(o, "vit_b", "cardiomegaly", "race_grp", 0.0, metric)]
                v, sd = float(r.value_raw), float(r.std_raw)
                lo, hi = float(r.ci_low_raw), float(r.ci_high_raw)
                lo_all.append(lo); hi_all.append(hi)
                ax.plot([lo, hi], [y, y], color=cols[o], lw=2.2, solid_capstyle="round", zorder=3)
                ax.plot([v - sd, v + sd], [y, y], color=cols[o], lw=6.0, solid_capstyle="butt",
                        alpha=0.55, zorder=4)
                ax.scatter([v], [y], s=130, facecolor=cols[o], edgecolor="white", linewidth=1.0, zorder=5)
                ax.text(v, y - 0.30, f"{v:.3f} $\\pm$ {sd:.3f}", ha="center", va="bottom",
                        fontsize=figstyle.fs(12.5), color=cols[o], zorder=6)
            span = max(hi_all) - min(lo_all)
            ax.set_xlim(min(lo_all) - span * 0.28, max(hi_all) + span * 0.28)
            ax.set_yticks(range(len(objs)))
            ax.set_yticklabels([self.OBJECTIVES[o] for o in objs], fontsize=figstyle.fs(13.0))
            ax.invert_yaxis()
            ax.set_ylim(len(objs) - 0.35, -0.75)
            ax.set_xlabel(mlab)
        self._labels.append((axes[0], "g", "One worked comparison, cardiomegaly at ViT-B"))

    def _panel_h(self, ax):
        figstyle.clean_axes(ax)
        m = self.matrix[self.matrix.seed == 0.0]
        key = ["backbone", "finding", "attribute"]
        a = m[m.encoder == "ssl"].set_index(key)
        b = m[m.encoder == "image_text"].set_index(key)
        idx = a.index.intersection(b.index)
        for attr in self.ATTRS:
            sel = [i for i in idx if i[2] == attr]
            ax.scatter(a.loc[sel].auroc_worst, b.loc[sel].auroc_worst, s=60,
                       facecolor=self.COLORS[attr], alpha=0.7, edgecolor="white", linewidth=0.7,
                       zorder=3, label=self.ATTRS[attr])
        lim = (0.52, 0.95)
        ax.plot(lim, lim, ls="--", lw=1.2, color=figstyle.REF_GREY, zorder=1)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel("Self-supervised")
        ax.set_ylabel("Image-text")
        figstyle.annotate_free_corner(ax, "worst-group AUROC", fontsize=13.5)
        self._labels.append((ax, "h", "Every comparison"))

    def _legend(self, ax):
        ax.axis("off")
        handles = [Line2D([0], [0], marker="o", lw=0, ms=14, mfc=figstyle.RACE, mec="white", label="Race"),
                   Line2D([0], [0], marker="o", lw=0, ms=14, mfc=figstyle.AGE, mec="white", label="Age"),
                   Line2D([0], [0], marker="o", lw=0, ms=14, mfc=figstyle.PERFORMANCE, mec="white", label="Sex"),
                   Patch(facecolor=figstyle.RACE, alpha=0.55, edgecolor=figstyle.RACE,
                         label="Gain from the level before"),
                   Patch(facecolor=figstyle.PERFORMANCE, alpha=0.35, edgecolor=figstyle.PERFORMANCE,
                         label="Overall AUROC at that level")]
        ax.legend(handles=handles, ncol=5, frameon=False, loc="center", bbox_to_anchor=(0.5, 0.5),
                  columnspacing=2.0, handletextpad=0.6)

    def build(self):
        self._labels = []
        self.fig = plt.figure(figsize=(18.5, 20.0), facecolor="white")
        gsl = self.fig.add_gridspec(1, 1, left=0.045, right=0.985, top=0.998, bottom=0.966)
        gs1 = self.fig.add_gridspec(1, 3, left=0.130, right=0.985, top=0.930, bottom=0.700,
                                    width_ratios=[1.10, 0.90, 1.05], wspace=0.75)
        gs2 = self.fig.add_gridspec(1, 3, left=0.115, right=0.990, top=0.618, bottom=0.382,
                                    width_ratios=[1.05, 1.00, 1.00], wspace=0.62)
        gs3 = self.fig.add_gridspec(1, 2, left=0.180, right=0.985, top=0.300, bottom=0.060,
                                    width_ratios=[1.25, 0.95], wspace=0.60)

        self._legend(self.fig.add_subplot(gsl[0, 0]))
        self._panel_a(self.fig.add_subplot(gs1[0, 0]))
        self._panel_b(self.fig.add_subplot(gs1[0, 1]))
        self._panel_c(self.fig.add_subplot(gs1[0, 2]))
        self._panel_d(self.fig.add_subplot(gs2[0, 0]))
        self._panel_e(self.fig.add_subplot(gs2[0, 1]))
        self._panel_f(self.fig.add_subplot(gs2[0, 2]))
        gsg = gs3[0, 0].subgridspec(2, 1, hspace=0.95)
        self._panel_g([self.fig.add_subplot(gsg[0, 0]), self.fig.add_subplot(gsg[1, 0])])
        self._panel_h(self.fig.add_subplot(gs3[0, 1]))

        self.fig.canvas.draw()
        figstyle.place_labels(self.fig, self._labels)
        return self

    def save(self):
        if self.fig is None:
            raise RuntimeError("call build() before save()")
        os.makedirs(self.out_pdf.parent, exist_ok=True)
        self.fig.canvas.draw()
        figcheck.report(self.fig, self.out_pdf.name)
        self.fig.savefig(self.out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
        plt.close(self.fig)
        print(f"Saved: {self.out_pdf}")
        return str(self.out_pdf)
