"""
figures/figure2_reference.py
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

from figures import figcheck, figstyle


class Figure2Reference:
    FONT_SCALE = 1.082
    TABLES = figstyle.TABLES
    OUT_PDF = os.path.join(figstyle.FIGS, "figure2_reference.pdf")

    ATTRS = {"race_grp": "Race", "age_grp": "Age"}
    COLORS = {"race_grp": figstyle.RACE, "age_grp": figstyle.AGE}
    REF_GREY = figstyle.REF_GREY
    TEXT = figstyle.TEXT
    MIN_EVALUABLE_PATIENTS = 20

    FINDING_LABELS = {
        "atelectasis": "atelectasis", "cardiomegaly": "cardiomegaly", "consolidation": "consolidation",
        "edema": "edema", "enlarged_cardiomediastinum": "enlarged\ncardiomediastinum",
        "fracture": "fracture", "lung_lesion": "lung lesion", "lung_opacity": "lung opacity",
        "pleural_effusion": "pleural effusion", "pleural_other": "pleural other",
        "pneumonia": "pneumonia", "pneumothorax": "pneumothorax", "support_devices": "support devices",
    }

    def __init__(self, out_pdf=None):
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        self._set_rcparams()
        self._load_data()
        self.fig = None

    @staticmethod
    def _set_rcparams():
        figstyle.set_rcparams(Figure2Reference.FONT_SCALE)

    def _load_data(self):
        perf = pd.read_csv(os.path.join(self.TABLES, "results_performance.csv"), low_memory=False)
        stat = pd.read_csv(os.path.join(self.TABLES, "results_statistics.csv"), low_memory=False)
        counts = pd.read_csv(os.path.join(self.TABLES, "results_subgroup_counts.csv"), low_memory=False)

        e8 = perf[(perf.experiment == "e8") & (perf.modality == "cxr")
                  & (perf.data_composition == "source_e2") & (perf.mitigation == "none")]
        w = e8.pivot_table(index=["encoder", "finding", "attribute"],
                           columns="metric_name", values="value_raw").reset_index()
        tests = stat[(stat.experiment == "e8") & (stat.data_composition == "source_e2")
                     & (stat.test_name == "fair_model_simulation")]
        w = w.merge(tests[["encoder", "finding", "attribute", "p_fdr", "significant_fdr05"]],
                    on=["encoder", "finding", "attribute"], how="left")
        w["share"] = 100.0 * w.gap_null / w.gap_observed
        w["exceeds"] = w.significant_fdr05.astype(bool)

        c = counts[(counts.modality == "cxr") & (counts.split == "test")]
        c = c.groupby(["attribute", "finding", "subgroup"], as_index=False).agg(
            n_pos=("n_positive", "sum"), n_pat=("n_patients", "sum"))
        c = c[c.n_pat >= self.MIN_EVALUABLE_PATIENTS]
        smallest = c.groupby(["attribute", "finding"], as_index=False).agg(
            smallest_positive=("n_pos", "min"), n_subgroups=("subgroup", "count"))
        smallest = smallest[smallest.finding.isin(w.finding.unique())
                            & smallest.attribute.isin(self.ATTRS)]
        self.d = w.merge(smallest, on=["attribute", "finding"], how="left")
        self.smallest = smallest

        rho = stat[(stat.experiment == "e8") & (stat.test_name == "spearman_permutation")
                   & (stat.n_units == 130)]
        self.rho = {r.attribute: (r.estimate, r.p_fdr) for r in rho.itertuples()}

        for a in self.ATTRS:
            n = len(self.d[self.d.attribute == a])
            if n != 130:
                raise ValueError(f"expected 130 combinations for {a}, found {n}")

    def _sub(self, attr):
        return self.d[self.d.attribute == attr]

    @staticmethod
    def _clean(ax):
        figstyle.clean_axes(ax)

    def _panel_label(self, ax, letter, title, x=None, y=None, gap=None):
        self._labels.append((ax, letter, title))

    def _place_labels(self):
        figstyle.place_labels(self.fig, self._labels)

    def _scatter(self, ax, s, attr):
        col = self.COLORS[attr]
        hi = s.exceeds
        ax.scatter(s.gap_null[~hi], s.gap_observed[~hi], s=95, marker="o", facecolor="white",
                   edgecolor=col, linewidth=1.5, zorder=3)
        ax.scatter(s.gap_null[hi], s.gap_observed[hi], s=95, marker="o", facecolor=col,
                   edgecolor="white", linewidth=0.8, alpha=0.9, zorder=4)

    def _panel_ab(self, ax, attr, letter):
        self._clean(ax)
        s = self._sub(attr)
        lim = 0.23
        ax.plot([0, lim], [0, lim], ls="--", lw=1.2, color=self.REF_GREY, zorder=1)
        self._scatter(ax, s, attr)
        r, p = self.rho[attr]
        lines = [f"median {s.gap_observed.median():.3f} vs {s.gap_null.median():.3f}",
                 f"{int(s.exceeds.sum())} of {len(s)} exceed",
                 f"Spearman $\\rho$ = {r:.3f}"]
        ax.text(0.965, 0.035, "\n".join(lines), transform=ax.transAxes, fontsize=figstyle.fs(15.0),
                ha="right", va="bottom", color=self.TEXT, linespacing=1.45)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ticks = [0, 0.05, 0.10, 0.15, 0.20]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_yticklabels([""] + [f"{v:.2f}" for v in ticks[1:]])
        ax.set_xlabel("Fair-model reference")
        ax.set_ylabel("Observed difference")
        self._panel_label(ax, letter, self.ATTRS[attr], x=-0.155, gap=0.062)

    def _panel_c(self, ax):
        self._clean(ax)
        data = [self._sub(a).share.values for a in self.ATTRS]
        bp = ax.boxplot(data, positions=[0, 1], widths=0.42, patch_artist=True,
                        medianprops=dict(color=self.TEXT, lw=2.0),
                        whiskerprops=dict(color=self.TEXT, lw=1.2),
                        capprops=dict(color=self.TEXT, lw=1.2),
                        flierprops=dict(marker="", ls="none"), zorder=3)
        for patch, a in zip(bp["boxes"], self.ATTRS):
            patch.set(facecolor="white", edgecolor=self.COLORS[a], linewidth=1.8)
        rng = np.random.default_rng(0)
        for i, a in enumerate(self.ATTRS):
            s = self._sub(a)
            x = i + rng.uniform(-0.16, 0.16, len(s))
            hi = s.exceeds.values
            ax.scatter(x[~hi], s.share.values[~hi], s=34, facecolor="white",
                       edgecolor=self.COLORS[a], linewidth=1.1, alpha=0.85, zorder=2)
            ax.scatter(x[hi], s.share.values[hi], s=34, facecolor=self.COLORS[a],
                       edgecolor="none", alpha=0.55, zorder=2)
            ax.text(i + 0.30, s.share.median(), f"{s.share.median():.0f}%", ha="left",
                    va="center", fontsize=figstyle.fs(16.5), color=self.COLORS[a])
        ax.axhline(100, ls="--", lw=1.2, color=self.REF_GREY, zorder=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([self.ATTRS[a] for a in self.ATTRS])
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(0, 145)
        ax.set_ylabel("Reference as a share of\nthe observed difference (%)")
        self._panel_label(ax, "c", "Share accounted for", x=-0.30, gap=0.105)

    def _panel_d(self, ax):
        self._clean(ax)
        for a in self.ATTRS:
            s = self._sub(a)
            hi = s.exceeds.values
            ax.scatter(s.smallest_positive.values[~hi], s.gap_null.values[~hi], s=80, facecolor="white",
                       edgecolor=self.COLORS[a], linewidth=1.4, zorder=3)
            ax.scatter(s.smallest_positive.values[hi], s.gap_null.values[hi], s=80,
                       facecolor=self.COLORS[a], edgecolor="white", linewidth=0.7, alpha=0.9, zorder=3)
        for finding, dx, dy, ha in [("pleural_other", 2.1, 0.009, "left"),
                                    ("support_devices", 1.77, 0.035, "center")]:
            row = self._sub("race_grp")
            row = row[row.finding == finding]
            x0, y0 = row.smallest_positive.median(), row.gap_null.median()
            ax.annotate(self.FINDING_LABELS[finding].replace("\n", " "), xy=(x0, y0),
                        xytext=(x0 * dx, y0 + dy), fontsize=figstyle.fs(14.5), color=self.TEXT,
                        ha=ha, va="center" if dy > 0 else "top",
                        arrowprops=dict(arrowstyle="-", color=figstyle.FAINT, lw=1.0,
                                        shrinkA=2, shrinkB=4))
        ax.set_xscale("log")
        ax.set_xlabel("Positive cases in the smallest\nevaluable subgroup")
        ax.set_ylabel("Fair-model reference")
        ax.set_ylim(0, 0.098)
        self._panel_label(ax, "d", "Size dependence", x=-0.26, gap=0.092)

    def _order(self):
        s = self.smallest[self.smallest.attribute == "race_grp"].set_index("finding")
        return list(s.smallest_positive.sort_values().index)

    def _panel_e(self, ax):
        figstyle.clean_axes(ax)
        order = self._order()
        y = np.arange(len(order))
        counts = {}
        for a in self.ATTRS:
            counts[a] = self._sub(a).groupby("finding").exceeds.sum().loc[order].values.astype(float)
        ax.barh(y, -counts["race_grp"], height=0.66, color=self.COLORS["race_grp"], alpha=0.85,
                edgecolor="white", linewidth=0.8, zorder=3)
        ax.barh(y, counts["age_grp"], height=0.66, color=self.COLORS["age_grp"], alpha=0.85,
                edgecolor="white", linewidth=0.8, zorder=3)
        for i in y:
            for a, sign in (("race_grp", -1), ("age_grp", 1)):
                v = counts[a][i]
                if v:
                    ax.text(sign * (v + 0.35), i, f"{int(v)}", ha="right" if sign < 0 else "left",
                            va="center", fontsize=figstyle.fs(12.5), color=self.COLORS[a])
        ax.axvline(0, color=figstyle.TEXT, lw=1.2, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([self.FINDING_LABELS[f].replace("\n", " ") for f in order], fontsize=figstyle.fs(13.5))
        ax.invert_yaxis()
        ax.set_xlim(-13.2, 13.2)
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_xticklabels(["10", "5", "0", "5", "10"])
        ax.set_ylim(len(order) + 0.15, -0.7)
        ax.set_xlabel("Encoders exceeding the reference (of 10)")
        for a, x, ha in (("race_grp", 0.02, "left"), ("age_grp", 0.98, "right")):
            n = int(counts[a].sum())
            ax.text(x, 0.012, f"{self.ATTRS[a]}, {n} of 130", transform=ax.transAxes, ha=ha,
                    va="bottom", fontsize=figstyle.fs(14.5), color=self.COLORS[a])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        self._labels.append((ax, "e", "Exceedance by finding"))

    def _panel_f(self, ax):
        self._clean(ax)
        s = self._sub("race_grp").groupby("finding").agg(
            obs=("gap_observed", "median"), ref=("gap_null", "median")).sort_values("obs")
        order = list(s.index)
        y = np.arange(len(order))
        col = self.COLORS["race_grp"]
        for i, f in enumerate(order):
            ax.plot([s.ref[f], s.obs[f]], [i, i], color=figstyle.FAINT, lw=1.6, zorder=1)
        ax.scatter(s.ref.values, y, s=95, marker="D", facecolor="white", edgecolor=col,
                   linewidth=1.6, zorder=3, label="reference")
        ax.scatter(s.obs.values, y, s=105, marker="o", facecolor=col, edgecolor="white",
                   linewidth=0.8, zorder=3, label="observed")
        ax.set_yticks(y)
        ax.set_yticklabels([self.FINDING_LABELS[f].replace("\n", " ") for f in order], fontsize=figstyle.fs(14.0))
        ax.invert_yaxis()
        ax.set_xlim(0, 0.155)
        ax.set_ylim(len(order) - 0.35, -0.65)
        ax.set_xlabel("Race difference")
        self._panel_label(ax, "f", "Per-finding medians, race", x=-0.52, gap=0.155)

    def _panel_g(self, ax):
        self._clean(ax)
        rng = np.random.default_rng(1)
        for i, a in enumerate(self.ATTRS):
            v = self.smallest[self.smallest.attribute == a].smallest_positive.values
            x = i + rng.uniform(-0.13, 0.13, len(v))
            ax.scatter(x, v, s=85, facecolor=self.COLORS[a], edgecolor="white", linewidth=0.8,
                       alpha=0.9, zorder=3)
            med = np.median(v)
            ax.plot([i - 0.30, i + 0.30], [med, med], color=self.TEXT, lw=2.2, zorder=4)
            ax.text(i + 0.36, med, f"{med:,.0f}", fontsize=figstyle.fs(15.5), color=self.TEXT,
                    ha="left", va="center")
        ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{self.ATTRS[a]}\n({int(self.smallest[self.smallest.attribute == a].n_subgroups.iloc[0])} subgroups)"
                            for a in self.ATTRS])
        ax.set_xlim(-0.6, 1.75)
        ax.set_ylabel("Positive cases in the smallest\nevaluable subgroup")
        self._panel_label(ax, "g", "Subgroup sizes", x=-0.30, gap=0.105)

    def _legend(self, ax):
        ax.axis("off")
        handles = [
            Line2D([0], [0], marker="o", lw=0, markersize=15, mfc=self.COLORS["race_grp"],
                   mec="white", mew=0.8, label="Race"),
            Line2D([0], [0], marker="o", lw=0, markersize=15, mfc=self.COLORS["age_grp"],
                   mec="white", mew=0.8, label="Age"),
            Line2D([0], [0], marker="o", lw=0, markersize=15, mfc=figstyle.NEUTRAL, mec="white",
                   mew=0.8, label="Exceeds its reference"),
            Line2D([0], [0], marker="o", lw=0, markersize=15, mfc="white", mec=figstyle.NEUTRAL,
                   mew=1.6, label="Does not exceed"),
            Line2D([0], [0], ls="--", lw=1.4, color=self.REF_GREY, label="Identity or 100% line"),
        ]
        ax.legend(handles, [h.get_label() for h in handles], ncol=5, frameon=False,
                  loc="center", bbox_to_anchor=(0.5, 0.5), columnspacing=1.8, handletextpad=0.5)

    def build(self):
        self._labels = []
        self.fig = plt.figure(figsize=(18.5, 20.5), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.045, right=0.985, top=0.995, bottom=0.955)
        gs_r1 = self.fig.add_gridspec(1, 2, left=0.075, right=0.985, top=0.915, bottom=0.665, wspace=0.24)
        gs_r2 = self.fig.add_gridspec(1, 3, left=0.075, right=0.978, top=0.590, bottom=0.345,
                                      width_ratios=[0.85, 1.05, 1.10], wspace=0.42)
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.205, right=0.985, top=0.272, bottom=0.045,
                                      width_ratios=[1.30, 0.80], wspace=0.42)

        self._legend(self.fig.add_subplot(gs_leg[0, 0]))
        self._panel_ab(self.fig.add_subplot(gs_r1[0, 0]), "race_grp", "a")
        self._panel_ab(self.fig.add_subplot(gs_r1[0, 1]), "age_grp", "b")
        self._panel_c(self.fig.add_subplot(gs_r2[0, 0]))
        self._panel_d(self.fig.add_subplot(gs_r2[0, 1]))
        self._panel_e(self.fig.add_subplot(gs_r2[0, 2]))
        self._panel_f(self.fig.add_subplot(gs_r3[0, 0]))
        self._panel_g(self.fig.add_subplot(gs_r3[0, 1]))
        self.fig.canvas.draw()
        self._place_labels()
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
