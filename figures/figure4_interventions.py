"""
figures/figure4_interventions.py
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


class Figure4Interventions:
    FONT_SCALE = 1.064
    OUT_PDF = os.path.join(figstyle.FIGS, "figure4_interventions.pdf")

    ATTRS = {"race_grp": "Race", "age_grp": "Age"}
    COLORS = {"race_grp": figstyle.RACE, "age_grp": figstyle.AGE}
    METHODS = {"resample": "group-balanced resampling", "reweigh": "reweighing",
               "group_dro": "group DRO", "adversarial": "adversarial removal",
               "reduction": "exponentiated gradient", "eo_shift": "operating point shift",
               "platt_recal": "Platt recalibration", "leace": "LEACE", "inlp": "INLP"}
    FAMILIES = {"resample": "preprocessing", "reweigh": "preprocessing",
                "group_dro": "in-processing", "adversarial": "in-processing",
                "reduction": "in-processing", "eo_shift": "postprocessing",
                "platt_recal": "postprocessing", "leace": "erasure", "inlp": "erasure"}
    FAMILY_ORDER = ["preprocessing", "in-processing", "postprocessing", "erasure"]
    MEASURES = [("auroc_gap", "AUROC\ndifference"), ("tpr_gap", "sensitivity\ndifference"),
                ("fpr_gap", "FPR\ndifference"), ("ece_gap", "calibration\ndifference")]
    UNFREEZE = ["unfreeze_depth0", "unfreeze_depth1", "unfreeze_depth2", "unfreeze_depth3"]
    UNFREEZE_LABELS = ["linear\nhead", "last\nblock", "LoRA", "full\nfinetune"]
    CONTRASTS = [("image_text_minus_ssl", "image-text vs\nself-supervised"),
                 ("image_text_minus_supervised", "image-text vs\nlabel supervision"),
                 ("ssl_minus_supervised", "self-supervised vs\nlabel supervision")]
    COMPOSITION = [("e1_data_composition", "race-balanced\npretraining data"),
                   ("e1_scrubbed", "demographic terms\nremoved"),
                   ("e1_amplified", "demographic sentence\nprepended")]

    def __init__(self, out_pdf=None):
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        figstyle.set_rcparams(Figure4Interventions.FONT_SCALE)
        self._load_data()
        self.fig = None
        self._labels = []

    def _load_data(self):
        perf = pd.read_csv(os.path.join(figstyle.TABLES, "results_performance.csv"), low_memory=False)
        stat = pd.read_csv(os.path.join(figstyle.TABLES, "results_statistics.csv"), low_memory=False)

        e2 = perf[perf.experiment == "e2"]
        keep = list(self.METHODS) + ["none"]
        w = e2[e2.mitigation.isin(keep)].pivot_table(
            index=["encoder", "finding", "attribute", "mitigation"], columns="metric_name",
            values="value_raw", aggfunc="first").reset_index()
        self.base = w[w.mitigation == "none"].set_index(["encoder", "finding", "attribute"])
        self.mit = {m: w[w.mitigation == m].set_index(["encoder", "finding", "attribute"])
                    for m in self.METHODS}
        ceil = e2[e2.mitigation.astype(str).str.startswith("ceiling:")]
        self.ceiling = ceil[ceil.metric_name == "ceiling_gap"].groupby(
            ["encoder", "finding", "attribute"]).value_raw.first()

        s1 = stat[(stat.experiment == "e1") & (stat.estimate_name == "auroc_gap_difference")]
        self.contrasts = s1[s1.data_composition.astype(str) == "natural"]
        self.compositions = s1

        e6 = perf[(perf.experiment == "e6") & (perf.metric_name == "auroc_gap")]
        self.unfreeze = e6.pivot_table(index=["encoder", "finding", "attribute"],
                                       columns="data_composition", values="value_raw", aggfunc="first")

        e1 = perf[(perf.experiment == "e1") & (perf.metric_name == "auroc_gap")
                  & (perf.data_composition == "natural") & (perf.head_type == "linear")]
        seeds = e1.pivot_table(index=["encoder", "backbone", "finding", "attribute"],
                               columns="seed", values="value_raw", aggfunc="first").dropna()
        self.seed_spread = (seeds.max(axis=1) - seeds.min(axis=1))

        for a in self.ATTRS:
            n = len(self.base[self.base.index.get_level_values("attribute") == a])
            if n != 130:
                raise ValueError(f"expected 130 combinations for {a}, found {n}")

    def _delta(self, method, attr, metric):
        idx = [k for k in self.base.index if k[2] == attr]
        return (self.mit[method].loc[idx][metric] - self.base.loc[idx][metric])

    def _cost(self, method, attr):
        idx = [k for k in self.base.index if k[2] == attr]
        return (self.base.loc[idx].auroc_overall - self.mit[method].loc[idx].auroc_overall)

    def _reduction(self, attr):
        idx = [k for k in self.base.index if k[2] == attr]
        return (self.base.loc[idx].auroc_gap - self.ceiling.loc[idx])

    def _panel_a(self, ax):
        figstyle.clean_axes(ax)
        order = sorted(self.METHODS, key=lambda m: self._delta(m, "race_grp", "auroc_gap").median())
        y = np.arange(len(order))
        for a, off in (("race_grp", -0.19), ("age_grp", 0.19)):
            v = [self._delta(m, a, "auroc_gap").median() for m in order]
            ax.barh(y + off, v, height=0.36, color=self.COLORS[a], alpha=0.85, edgecolor="white",
                    linewidth=0.8, zorder=3)
        seed = self.seed_spread[self.seed_spread.index.get_level_values("attribute") == "race_grp"].median()
        ax.axvspan(-seed, 0, color=figstyle.FAINT, alpha=0.45, zorder=1)
        ax.text(-seed / 2, -1.28, "seed-to-seed\nspread", ha="center", va="top", fontsize=figstyle.fs(13.0),
                color=figstyle.TEXT)
        ax.axvline(0, color=figstyle.TEXT, lw=1.2, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([self.METHODS[m] for m in order], fontsize=figstyle.fs(13.5))
        ax.invert_yaxis()
        ax.set_xlim(-0.0155, 0.004)
        ax.set_ylim(len(order) - 0.35, -1.35)
        ax.set_xlabel("Change in the difference")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        self._labels.append((ax, "a", "The reduction each method gives"))

    def _panel_b(self, ax):
        figstyle.clean_axes(ax)
        for m in self.METHODS:
            gain = -self._delta(m, "race_grp", "auroc_gap").median()
            cost = self._cost(m, "race_grp").median()
            rate = abs(self._delta(m, "race_grp", "fpr_gap").median())
            ax.scatter([cost], [gain], s=90 + rate * 3200, facecolor=figstyle.RACE, alpha=0.30,
                       edgecolor=figstyle.RACE, linewidth=1.6, zorder=3)
            if cost > 0.02:
                place = {"group_dro": "above", "adversarial": "below", "leace": "above", "inlp": "right"}
                where = place.get(m, "right")
                if where == "above":
                    ax.text(cost, gain + 0.0016 + rate * 0.012, self.METHODS[m], fontsize=figstyle.fs(12.5),
                            color=figstyle.TEXT, ha="center", va="bottom", zorder=4)
                elif where == "below":
                    ax.text(cost, gain - 0.0016 - rate * 0.012, self.METHODS[m], fontsize=figstyle.fs(12.5),
                            color=figstyle.TEXT, ha="center", va="top", zorder=4)
                else:
                    ax.text(cost + 0.005 + rate * 0.022, gain, self.METHODS[m], fontsize=figstyle.fs(12.5),
                            color=figstyle.TEXT, ha="left", va="center", zorder=4)
        ax.axhline(0, color=figstyle.REF_GREY, ls="--", lw=1.2, zorder=1)
        ax.axvline(0, color=figstyle.REF_GREY, ls="--", lw=1.2, zorder=1)

        ax.set_xlim(-0.018, 0.122)
        ax.set_xticks([0, 0.04, 0.08, 0.12])
        ax.set_ylim(-0.0025, 0.0195)
        ax.set_xlabel("Disease AUROC paid")
        ax.set_ylabel("Race difference bought")
        figstyle.annotate_free_corner(ax, "five methods at a cost\nbelow 0.01", fontsize=12.5)
        self._labels.append((ax, "b", "The trade each method makes"))

    def _panel_c(self, ax):
        figstyle.clean_axes(ax)
        for a in self.ATTRS:
            v = np.sort(self._reduction(a).values)
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=self.COLORS[a], lw=2.4,
                    zorder=3, label=self.ATTRS[a])
            ax.plot([np.median(v)], [0.5], marker="o", ms=11, mfc=self.COLORS[a], mec="white",
                    mew=0.9, zorder=4)
            ax.text(0.058, 0.20 if a == "race_grp" else 0.10, f"{self.ATTRS[a]} median {np.median(v):.3f}",
                    fontsize=figstyle.fs(13.0), color=self.COLORS[a], ha="right", va="center")
        ax.axhline(0.5, color=figstyle.REF_GREY, ls="--", lw=1.2, zorder=1)
        ax.set_xlim(-0.002, 0.06)
        ax.set_xticks([0, 0.02, 0.04, 0.06])
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.set_xlabel("Reduction reached")
        ax.set_ylabel("Share of combinations")
        self._labels.append((ax, "c", "The reduction reached"))

    def _panel_d(self, ax):
        figstyle.clean_axes(ax)
        rows = []
        for key, lab in self.CONTRASTS:
            for a in self.ATTRS:
                s = self.contrasts[(self.contrasts.encoder_objective == key)
                                   & (self.contrasts.attribute == a)]
                rows.append((lab, a, s.estimate.median(), s.estimate.quantile(.25),
                             s.estimate.quantile(.75), int(s.significant_fdr05.sum()), len(s)))
        for key, lab in self.COMPOSITION:
            for a in self.ATTRS:
                s = self.compositions[(self.compositions.fdr_family == f"{key}::{a}")]
                rows.append((lab, a, s.estimate.median(), s.estimate.quantile(.25),
                             s.estimate.quantile(.75), int(s.significant_fdr05.sum()), len(s)))
        labels = list(dict.fromkeys(r[0] for r in rows))
        for lab, a, med, q1, q3, sig, n in rows:
            i = labels.index(lab)
            off = -0.19 if a == "race_grp" else 0.19
            ax.plot([q1, q3], [i + off] * 2, color=self.COLORS[a], lw=2.4, solid_capstyle="round", zorder=3)
            ax.scatter([med], [i + off], s=110, facecolor=self.COLORS[a] if sig else "white",
                       edgecolor=self.COLORS[a], linewidth=1.7, zorder=4)
            ax.text(0.0265, i + off, f"{sig}/{n}", fontsize=figstyle.fs(12.5), color=self.COLORS[a],
                    ha="right", va="center")
        ax.axvline(0, color=figstyle.REF_GREY, ls="--", lw=1.2, zorder=1)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=figstyle.fs(13.0))
        ax.invert_yaxis()
        ax.set_xlim(-0.040, 0.027)
        ax.set_ylim(len(labels) - 0.4, -0.6)
        ax.set_xlabel("Paired change in the difference")
        self._labels.append((ax, "d", "Changing how the encoder is pretrained"))

    def _panel_e(self, ax):
        figstyle.clean_axes(ax)
        x = np.arange(len(self.UNFREEZE))
        for a in self.ATTRS:
            s = self.unfreeze[self.unfreeze.index.get_level_values("attribute") == a]
            for _, row in s.iterrows():
                ax.plot(x, row[self.UNFREEZE].values, color=self.COLORS[a], lw=1.0, alpha=0.18, zorder=2)
            med = [s[c].median() for c in self.UNFREEZE]
            ax.plot(x, med, color=self.COLORS[a], lw=3.0, marker="o", ms=11, mec="white", mew=0.9,
                    zorder=4, label=self.ATTRS[a])
            ax.text(x[-1] + 0.38, med[-1], f"{med[-1]:.3f}", fontsize=figstyle.fs(13.5), color=self.COLORS[a],
                    ha="left", va="center")

        ax.set_xticks(x)
        ax.set_xticklabels(self.UNFREEZE_LABELS, fontsize=figstyle.fs(12.5), rotation=22, ha="right")
        ax.set_xlim(-1.25, len(x) + 0.35)
        ax.set_ylabel("Subgroup difference")
        ax.legend(frameon=False, loc="upper right", fontsize=figstyle.fs(14.0), handletextpad=0.6, borderpad=0.2)
        self._labels.append((ax, "e", "Unfreezing the backbone"))

    def _panel_f(self, ax):
        figstyle.clean_axes(ax)
        data, colors, labels = [], [], []
        for a in self.ATTRS:
            data.append(self.seed_spread[self.seed_spread.index.get_level_values("attribute") == a].values)
            colors.append(self.COLORS[a]); labels.append(f"{self.ATTRS[a]} seeds")
            data.append(self._reduction(a).values)
            colors.append(self.COLORS[a]); labels.append(f"{self.ATTRS[a]} mitigation")
        parts = ax.violinplot(data, positions=np.arange(len(data)), widths=0.78, showextrema=False, vert=False)
        for body, col in zip(parts["bodies"], colors):
            body.set_facecolor(col); body.set_alpha(0.30); body.set_edgecolor(col); body.set_linewidth(1.4)
        for i, (v, col, body) in enumerate(zip(data, colors, parts["bodies"])):
            ax.plot([np.median(v)] * 2, [i - 0.28, i + 0.28], color=col, lw=2.8, zorder=4)
            right = body.get_paths()[0].vertices[:, 0].max()
            ax.text(right + 0.0035, i, f"{np.median(v):.3f}", ha="left", va="center",
                    fontsize=figstyle.fs(12.5), color=col, zorder=5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=figstyle.fs(12.5))
        ax.invert_yaxis()
        ax.set_ylim(len(labels) - 0.3, -0.7)
        xs = np.concatenate([b.get_paths()[0].vertices[:, 0] for b in parts["bodies"]])
        ax.set_xlim(min(0.0, xs.min()), xs.max() * 1.32)
        ax.set_xlabel("Change in the difference")
        self._labels.append((ax, "f", "Retraining vs mitigating"))

    def _panel_g(self, ax, cbar_ax):
        figstyle.clean_axes(ax)
        order = [m for f in self.FAMILY_ORDER for m in self.METHODS if self.FAMILIES[m] == f]
        vals = np.array([[self._delta(m, "race_grp", met).median() for m, _ in [(m, None)]][0]
                         for met, _ in self.MEASURES for m in order]).reshape(len(self.MEASURES), len(order))
        scale = float(np.abs(vals).max())
        im = ax.imshow(vals, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
        rgba = im.cmap(im.norm(vals))
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                r, g, b = rgba[i, j, :3]
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, f"{vals[i, j]:+.3f}", ha="center", va="center", fontsize=figstyle.fs(12.0),
                        color="white" if lum < 0.45 else figstyle.TEXT)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([self.METHODS[m] for m in order], rotation=32, ha="right", fontsize=figstyle.fs(12.5))
        ax.set_yticks(range(len(self.MEASURES)))
        ax.set_yticklabels([lab for _, lab in self.MEASURES], fontsize=figstyle.fs(13.0))
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        cb = self.fig.colorbar(im, cax=cbar_ax)
        cb.set_label("Paired change", fontsize=figstyle.fs(14.0))
        cb.ax.tick_params(labelsize=figstyle.fs(12.5))
        cb.outline.set_visible(False)
        self._labels.append((ax, "g", "Every method on all four measures"))

    def _panel_h(self, ax):
        figstyle.clean_axes(ax)
        data, cols = [], []
        for f in self.FAMILY_ORDER:
            ms = [m for m in self.METHODS if self.FAMILIES[m] == f]
            data.append(np.concatenate([self._delta(m, "race_grp", "auroc_gap").values for m in ms]))
        bp = ax.boxplot(data, positions=np.arange(len(data)), widths=0.5, vert=False,
                        patch_artist=True, medianprops=dict(color=figstyle.TEXT, lw=2.0),
                        whiskerprops=dict(color=figstyle.TEXT, lw=1.2),
                        capprops=dict(color=figstyle.TEXT, lw=1.2),
                        flierprops=dict(marker="o", ms=3.5, mfc=figstyle.FAINT, mec="none"))
        for patch in bp["boxes"]:
            patch.set(facecolor="white", edgecolor=figstyle.RACE, linewidth=1.7)
        ax.axvline(0, color=figstyle.REF_GREY, ls="--", lw=1.2, zorder=1)
        ax.set_yticks(range(len(self.FAMILY_ORDER)))
        ax.set_yticklabels(self.FAMILY_ORDER, fontsize=figstyle.fs(13.5))
        ax.invert_yaxis()
        ax.set_xlim(-0.155, 0.038)
        ax.set_xlabel("Change in the race AUROC difference")
        self._labels.append((ax, "h", "By family"))

    def _legend(self, ax):
        ax.axis("off")
        handles = [Line2D([0], [0], marker="o", lw=0, ms=14, mfc=figstyle.RACE, mec="white", label="Race"),
                   Line2D([0], [0], marker="o", lw=0, ms=14, mfc=figstyle.AGE, mec="white", label="Age"),
                   Line2D([0], [0], marker="o", lw=0, ms=14, mfc="white", mec=figstyle.NEUTRAL, mew=1.7,
                          label="Not significant at an FDR of 0.05"),
                   Patch(facecolor=figstyle.FAINT, edgecolor="none", label="Seed-to-seed spread")]
        ax.legend(handles=handles, ncol=4, frameon=False, loc="center", bbox_to_anchor=(0.5, 0.5),
                  columnspacing=2.0, handletextpad=0.6)

    def build(self):
        self._labels = []
        self.fig = plt.figure(figsize=(18.5, 19.62), facecolor="white")
        gsl = self.fig.add_gridspec(1, 1, left=0.045, right=0.985, top=0.998, bottom=0.966)
        gs1 = self.fig.add_gridspec(1, 3, left=0.190, right=0.975, top=0.930, bottom=0.715,
                                    width_ratios=[1.05, 1.00, 1.00], wspace=0.74)
        gs2 = self.fig.add_gridspec(1, 3, left=0.165, right=0.985, top=0.630, bottom=0.385,
                                    width_ratios=[1.15, 0.92, 1.05], wspace=1.02)
        gs3 = self.fig.add_gridspec(1, 2, left=0.130, right=0.880, top=0.290, bottom=0.092,
                                    width_ratios=[1.45, 0.85], wspace=0.55)
        cax = self.fig.add_axes([0.906, 0.135, 0.011, 0.130])
        cax._figcheck_skip = True

        self._legend(self.fig.add_subplot(gsl[0, 0]))
        self._panel_a(self.fig.add_subplot(gs1[0, 0]))
        self._panel_b(self.fig.add_subplot(gs1[0, 1]))
        self._panel_c(self.fig.add_subplot(gs1[0, 2]))
        self._panel_d(self.fig.add_subplot(gs2[0, 0]))
        self._panel_e(self.fig.add_subplot(gs2[0, 1]))
        self._panel_f(self.fig.add_subplot(gs2[0, 2]))
        self._panel_g(self.fig.add_subplot(gs3[0, 0]), cax)
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
