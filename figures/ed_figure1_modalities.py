"""
figures/ed_figure1_modalities.py
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


class EDFigure1Modalities:
    FONT_SCALE = 1.000
    OUT_PDF = os.path.join(figstyle.FIGS, "ed_figure1_modalities.pdf")

    MODALITIES = {"derm": "Dermatology", "fundus": "Funduscopy"}
    MOD_COLORS = {"derm": figstyle.RACE, "fundus": figstyle.AGE}
    ATTR_LABELS = {"age_grp": "age", "sex_grp": "sex", "fst_grp": "skin type",
                   "race_grp": "race", "ethnicity_grp": "ethnicity"}
    ATTR_ORDER = ["age_grp", "race_grp", "ethnicity_grp", "fst_grp", "sex_grp"]
    MIN_EVALUABLE_PATIENTS = 20

    def __init__(self, out_pdf=None):
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        figstyle.set_rcparams(EDFigure1Modalities.FONT_SCALE)
        self._load_data()
        self.fig = None
        self._labels = []

    def _load_data(self):
        perf = pd.read_csv(os.path.join(figstyle.TABLES, "results_performance.csv"), low_memory=False)
        stat = pd.read_csv(os.path.join(figstyle.TABLES, "results_statistics.csv"), low_memory=False)
        counts = pd.read_csv(os.path.join(figstyle.TABLES, "results_subgroup_counts.csv"), low_memory=False)

        e8 = perf[(perf.experiment == "e8") & (perf.modality.isin(self.MODALITIES))]
        w = e8.pivot_table(index=["modality", "encoder", "finding", "attribute"], columns="metric_name",
                           values="value_raw", aggfunc="first").reset_index()
        tests = stat[(stat.experiment == "e8") & (stat.modality.isin(self.MODALITIES))
                     & (stat.test_name == "fair_model_simulation")]
        self.units = w.merge(tests[["modality", "encoder", "finding", "attribute", "significant_fdr05"]],
                             on=["modality", "encoder", "finding", "attribute"], how="left")
        self.units["exceeds"] = self.units.significant_fdr05.astype(bool)

        e4 = perf[perf.experiment == "e4"]
        none = e4[e4.mitigation == "none"].pivot_table(
            index=["modality", "encoder", "finding", "attribute"], columns="metric_name",
            values="value_raw", aggfunc="first")
        ceil = e4[e4.mitigation.astype(str).str.startswith("ceiling:")]
        ceil = ceil[ceil.metric_name == "ceiling_gap"].groupby(
            ["modality", "encoder", "finding", "attribute"]).value_raw.first()
        self.mitigated = none.join(ceil.rename("achievable"), how="left").reset_index()

        c = counts[(counts.modality.isin(self.MODALITIES)) & (counts.split == "test")]
        c = c.groupby(["modality", "attribute", "finding", "subgroup"], as_index=False).agg(
            n_pos=("n_positive", "sum"), n_pat=("n_patients", "sum"))
        c = c[c.n_pat >= self.MIN_EVALUABLE_PATIENTS]
        self.smallest = c.groupby(["modality", "attribute", "finding"]).n_pos.min().groupby(
            level=[0, 1]).median()

        if len(self.units) != 63:
            raise ValueError(f"expected 63 combinations over the two modalities, found {len(self.units)}")

    def _groups(self):
        out = []
        for mod in self.MODALITIES:
            for attr in self.ATTR_ORDER:
                s = self.units[(self.units.modality == mod) & (self.units.attribute == attr)]
                if len(s):
                    out.append((mod, attr, s))
        return out

    def _panel_a(self, ax):
        figstyle.clean_axes(ax)
        groups = self._groups()
        for i, (mod, attr, s) in enumerate(groups):
            col = self.MOD_COLORS[mod]
            ref, obs = s.gap_null.median(), s.gap_observed.median()
            ax.annotate("", xy=(obs, i), xytext=(ref, i),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, shrinkA=3, shrinkB=3))
            ax.scatter([ref], [i], s=95, marker="D", facecolor="white", edgecolor=col,
                       linewidth=1.7, zorder=4)
            ax.scatter([obs], [i], s=110, facecolor=col, edgecolor="white", linewidth=0.9, zorder=5)
            n = int(s.exceeds.sum())
            ax.text(0.104, i, f"{n} of {len(s)}", ha="right", va="center", fontsize=figstyle.fs(12.5), color=col)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([f"{self.MODALITIES[m]}, {self.ATTR_LABELS[a]}" for m, a, _ in groups],
                           fontsize=figstyle.fs(13.0))
        ax.invert_yaxis()
        ax.set_xlim(0, 0.108)
        ax.set_ylim(len(groups) - 0.4, -0.6)
        ax.set_xlabel("Subgroup difference")
        handles = [Line2D([0], [0], marker="D", lw=0, ms=11, mfc="white", mec=figstyle.PERFORMANCE,
                          mew=1.7, label="fair-model reference"),
                   Line2D([0], [0], marker="o", lw=0, ms=12, mfc=figstyle.PERFORMANCE, mec="white",
                          label="observed")]
        ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=figstyle.fs(12.5),
                  handletextpad=0.4, borderpad=0.2)
        self._labels.append((ax, "a", "Reference and observed"))

    def _panel_b(self, ax):
        figstyle.clean_axes(ax)
        mods = list(self.MODALITIES)
        for i, attr in enumerate(self.ATTR_ORDER):
            for j, mod in enumerate(mods):
                s = self.units[(self.units.modality == mod) & (self.units.attribute == attr)]
                if not len(s):
                    continue
                obs = s.gap_observed.median()
                n = int(s.exceeds.sum())
                col = self.MOD_COLORS[mod]
                ax.scatter([j], [i], s=60 + obs * 9000, facecolor=col, alpha=0.30 if n == 0 else 0.75,
                           edgecolor=col, linewidth=1.6, zorder=3)
                ax.text(j + 0.32, i, f"{n}/9", ha="left", va="center", fontsize=figstyle.fs(12.5), color=col)
        ax.set_xticks(range(len(mods)))
        ax.set_xticklabels([self.MODALITIES[m] for m in mods], fontsize=figstyle.fs(13.5))
        ax.set_yticks(range(len(self.ATTR_ORDER)))
        ax.set_yticklabels([self.ATTR_LABELS[a] for a in self.ATTR_ORDER], fontsize=figstyle.fs(13.5))
        ax.invert_yaxis()
        ax.set_xlim(-0.55, len(mods) - 0.05)
        ax.set_ylim(len(self.ATTR_ORDER) - 0.4, -0.6)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        self._labels.append((ax, "b", "Difference and exceedance"))

    def _panel_c(self, ax):
        figstyle.clean_axes(ax)
        u = self.units
        xlim = u.gap_null.max() * 1.10
        ylim = u.gap_observed.max() * 1.07
        ax.plot([0, xlim], [0, xlim], ls="--", lw=1.2, color=figstyle.REF_GREY, zorder=1)
        for mod in self.MODALITIES:
            s = u[u.modality == mod]
            col = self.MOD_COLORS[mod]
            hi = s.exceeds.values
            ax.scatter(s.gap_null[~hi], s.gap_observed[~hi], s=70, facecolor="white", edgecolor=col,
                       linewidth=1.5, zorder=3)
            ax.scatter(s.gap_null[hi], s.gap_observed[hi], s=70, facecolor=col, edgecolor="white",
                       linewidth=0.8, alpha=0.9, zorder=4)
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        ax.set_xticks([0.02, 0.04, 0.06])
        ax.set_yticks([0, 0.03, 0.06, 0.09, 0.12])
        ax.set_xlabel("Fair-model reference")
        ax.set_ylabel("Observed difference")
        handles = [Line2D([0], [0], lw=7, color=self.MOD_COLORS["derm"], label="Dermatology"),
                   Line2D([0], [0], lw=7, color=self.MOD_COLORS["fundus"], label="Funduscopy"),
                   Line2D([0], [0], marker="o", lw=0, ms=11, mfc=figstyle.PERFORMANCE, mec="white",
                          label="exceeds its reference"),
                   Line2D([0], [0], marker="o", lw=0, ms=11, mfc="white", mec=figstyle.PERFORMANCE,
                          mew=1.5, label="does not exceed")]
        ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=figstyle.fs(12.0),
                  handletextpad=0.5, borderpad=0.2)
        self._labels.append((ax, "c", "Every combination against its reference"))

    def _panel_d(self, ax):
        figstyle.clean_axes(ax)
        groups = self._groups()
        for i, (mod, attr, s) in enumerate(groups):
            n = int(s.exceeds.sum())
            col = self.MOD_COLORS[mod]
            ax.barh(i, n, height=0.62, color=col, alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)
            ax.barh(i, 9 - n, left=n, height=0.62, color=figstyle.FAINT, alpha=0.65,
                    edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([self.ATTR_LABELS[a] for _, a, _ in groups], fontsize=figstyle.fs(13.0))
        ax.invert_yaxis()
        ax.set_xlim(0, 9)
        ax.set_xticks([0, 3, 6, 9])
        ax.set_ylim(len(groups) - 0.4, -0.6)
        ax.set_xlabel("Combinations exceeding, of 9")
        self._labels.append((ax, "d", "Exceedance count"))

    def _panel_e(self, ax):
        figstyle.clean_axes(ax)
        groups = self._groups()
        for i, (mod, attr, _) in enumerate(groups):
            s = self.mitigated[(self.mitigated.modality == mod) & (self.mitigated.attribute == attr)]
            col = self.MOD_COLORS[mod]
            obs, ach = s.auroc_gap.median(), s.achievable.median()
            ax.plot([ach, obs], [i, i], color=figstyle.FAINT, lw=1.8, zorder=1)
            ax.scatter([obs], [i], s=110, facecolor=col, edgecolor="white", linewidth=0.9, zorder=4)
            ax.scatter([ach], [i], s=95, marker="s", facecolor="white", edgecolor=col,
                       linewidth=1.7, zorder=4)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([self.ATTR_LABELS[a] for _, a, _ in groups], fontsize=figstyle.fs(13.0))
        ax.invert_yaxis()
        ax.set_xlim(0, 0.108)
        ax.set_ylim(len(groups) - 0.4, -0.6)
        ax.set_xlabel("Subgroup difference")
        handles = [Line2D([0], [0], marker="o", lw=0, ms=12, mfc=figstyle.PERFORMANCE, mec="white",
                          label="unmitigated"),
                   Line2D([0], [0], marker="s", lw=0, ms=11, mfc="white", mec=figstyle.PERFORMANCE,
                          mew=1.7, label="best of the nine")]
        ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=figstyle.fs(12.5),
                  handletextpad=0.4, borderpad=0.2)
        self._labels.append((ax, "e", "The best the nine methods reach"))

    def _panel_f(self, ax):
        figstyle.clean_axes(ax)
        place = {("derm", "age_grp"): (1.35, -0.004, "left", "top"),
                 ("derm", "fst_grp"): (0.95, 0.006, "center", "bottom"),
                 ("derm", "sex_grp"): (1.35, 0.004, "left", "bottom"),
                 ("fundus", "age_grp"): (1.35, 0.004, "left", "bottom"),
                 ("fundus", "race_grp"): (1.40, -0.005, "left", "top"),
                 ("fundus", "ethnicity_grp"): (1.05, -0.006, "center", "top"),
                 ("fundus", "sex_grp"): (1.35, -0.004, "left", "top")}
        for mod, attr, s in self._groups():
            col = self.MOD_COLORS[mod]
            x = float(self.smallest.loc[(mod, attr)])
            y = s.gap_null.median()
            fx, dy, ha, va = place[(mod, attr)]
            ax.scatter([x], [y], s=130, facecolor=col, edgecolor="white", linewidth=0.9, zorder=3)
            ax.annotate(self.ATTR_LABELS[attr], xy=(x, y), xytext=(x * fx, y + dy),
                        fontsize=figstyle.fs(12.5), color=figstyle.TEXT, ha=ha, va=va,
                        arrowprops=dict(arrowstyle="-", color=figstyle.FAINT, lw=1.0,
                                        shrinkA=2, shrinkB=5))
        ax.set_xscale("log")
        ax.set_xlim(35, 3000)
        ax.set_ylim(0, 0.062)
        ax.set_xlabel("Positive cases in the smallest\nevaluable subgroup")
        ax.set_ylabel("Fair-model reference")
        self._labels.append((ax, "f", "The reference tracks cohort size here too"))

    def build(self):
        self._labels = []
        self.fig = plt.figure(figsize=(18.5, 18.30), facecolor="white")
        gs1 = self.fig.add_gridspec(1, 2, left=0.215, right=0.985, top=0.950, bottom=0.690,
                                    width_ratios=[1.25, 0.80], wspace=0.55)
        gs2 = self.fig.add_gridspec(1, 2, left=0.160, right=0.985, top=0.630, bottom=0.375,
                                    width_ratios=[1.05, 0.90], wspace=0.55)
        gs3 = self.fig.add_gridspec(1, 2, left=0.160, right=0.985, top=0.305, bottom=0.055,
                                    width_ratios=[1.05, 0.95], wspace=0.55)

        self._panel_a(self.fig.add_subplot(gs1[0, 0]))
        self._panel_b(self.fig.add_subplot(gs1[0, 1]))
        self._panel_c(self.fig.add_subplot(gs2[0, 0]))
        self._panel_d(self.fig.add_subplot(gs2[0, 1]))
        self._panel_e(self.fig.add_subplot(gs3[0, 0]))
        self._panel_f(self.fig.add_subplot(gs3[0, 1]))

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
