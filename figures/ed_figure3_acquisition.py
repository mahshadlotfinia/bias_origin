"""
figures/ed_figure3_acquisition.py
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


class EDFigure3Acquisition:
    FONT_SCALE = 1.093
    OUT_PDF = os.path.join(figstyle.FIGS, "ed_figure3_acquisition.pdf")

    COND = ["stratum:all", "stratum:view=AP", "stratum:view=PA"]
    COND_LABELS = {"stratum:all": "Pooled", "stratum:view=AP": "AP only", "stratum:view=PA": "PA only"}
    DEMOG = ["race_grp", "age_grp"]
    ATTRS = ["view", "site", "race_grp", "age_grp"]
    ATTR_LABELS = {"view": "Acquisition view", "site": "Data source",
                   "race_grp": "Race", "age_grp": "Age"}
    ATTR_COLORS = {"view": figstyle.DECODABILITY, "site": figstyle.ACHIEVABLE,
                   "race_grp": figstyle.RACE, "age_grp": figstyle.AGE}
    MEASURES = [("auroc_gap", "AUROC"), ("tpr_gap", "Sensitivity"),
                ("fpr_gap", "FPR"), ("ece_gap", "Calibration error")]

    def __init__(self, out_pdf=None):
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        figstyle.set_rcparams(EDFigure3Acquisition.FONT_SCALE)
        self._load_data()
        self.fig = None
        self._labels = []

    def _load_data(self):
        perf = pd.read_csv(os.path.join(figstyle.TABLES, "results_performance.csv"), low_memory=False)
        stat = pd.read_csv(os.path.join(figstyle.TABLES, "results_statistics.csv"), low_memory=False)
        e = perf[perf.experiment == "e10"]
        self.units = e.pivot_table(index=["encoder", "finding", "attribute", "data_composition"],
                                   columns="metric_name", values="value_raw", aggfunc="first").reset_index()
        s = stat[stat.experiment == "e10"]
        self.paired = {(r.attribute, r.data_composition): (r.p_fdr, int(r.n_units))
                       for r in s[s.finding.str.startswith("pooled_vs")].itertuples()}
        self.vs_ref = {(r.attribute, r.data_composition): (r.p_fdr, int(r.n_units))
                       for r in s[s.finding.str.startswith("observed_vs")].itertuples()}

    def _sub(self, attr, cond="stratum:all"):
        return self.units[(self.units.attribute == attr) & (self.units.data_composition == cond)]

    def _wide(self, attr, metric="auroc_gap"):
        s = self.units[self.units.attribute == attr]
        return s.pivot_table(index=["encoder", "finding"], columns="data_composition",
                             values=metric, aggfunc="first")[self.COND]

    @staticmethod
    def _fmt_p(p):
        if p < 0.001:
            e = int(np.floor(np.log10(p)))
            return "$P=%.1f\\times10^{%d}$" % (p / 10 ** e, e)
        return "$P=%.3f$" % p

    def _panel_a(self, ax):
        figstyle.clean_axes(ax)
        y, ticks, labs = 0, [], []
        for attr in self.DEMOG:
            for cond in self.COND:
                s = self._sub(attr, cond)
                obs, ref = s.auroc_gap.median(), s.gap_fair_reference.median()
                col = self.ATTR_COLORS[attr]
                ax.barh(y, obs, height=0.62, color=col, alpha=0.75 if cond == "stratum:all" else 0.42,
                        edgecolor=col, linewidth=1.4, zorder=2)
                ax.plot([ref], [y], marker="D", ms=11, mfc="white", mec=figstyle.REF_GREY,
                        mew=2.0, zorder=4)
                ax.text(obs + 0.004, y, "%.3f" % obs, va="center", ha="left",
                        fontsize=figstyle.fs(13.0), color=figstyle.TEXT)
                ticks.append(y)
                labs.append("%s, %s" % (self.ATTR_LABELS[attr], self.COND_LABELS[cond]))
                y -= 1
            y -= 0.6
        ax.set_yticks(ticks)
        ax.set_yticklabels(labs, fontsize=figstyle.fs(13.5))
        ax.set_ylim(y + 0.7, 0.7)
        ax.set_xlim(0, 0.125)
        ax.set_xlabel("Subgroup performance difference")
        self._labels.append((ax, "a", "Holding the acquisition view fixed"))

    def _panel_b(self, ax):
        figstyle.clean_axes(ax)
        ax.axhline(0, color=figstyle.REF_GREY, ls="--", lw=1.3, zorder=1)
        notes = []
        for attr in self.DEMOG:
            w = self._wide(attr)
            col = self.ATTR_COLORS[attr]
            for cond, mk in (("stratum:view=AP", "o"), ("stratum:view=PA", "^")):
                d = w[cond].values - w["stratum:all"].values
                ax.scatter(w["stratum:all"].values, d, s=52, marker=mk, facecolor=col,
                           edgecolor="white", linewidth=0.7, alpha=0.75, zorder=3)
                pv, _ = self.paired[(attr, cond)]
                notes.append((col, "%s %s: %s" % (self.ATTR_LABELS[attr],
                                                  self.COND_LABELS[cond].split()[0], self._fmt_p(pv))))
        for k, (col, s) in enumerate(notes):
            ax.text(0.985, 0.995 - 0.062 * k, s, transform=ax.transAxes, ha="right", va="top",
                    fontsize=figstyle.fs(12.5), color=col)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.62 * (hi - lo))
        ax.set_xlim(0.01, 0.175)
        ax.set_xticks([0.05, 0.10, 0.15])
        ax.set_xlabel("Pooled difference")
        ax.set_ylabel("Change inside one view")
        handles = [Line2D([0], [0], marker="o", lw=0, ms=11, mfc=figstyle.PERFORMANCE, mec="white",
                          label="AP"),
                   Line2D([0], [0], marker="^", lw=0, ms=11, mfc=figstyle.PERFORMANCE, mec="white",
                          label="PA")]
        ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=figstyle.fs(13.0),
                  borderpad=0.2, handletextpad=0.5)
        self._labels.append((ax, "b", "The paired change"))

    def _panel_c(self, ax, cax):
        cols, labs = [], []
        for attr in self.DEMOG:
            for cond in ("stratum:view=AP", "stratum:view=PA"):
                w = self._wide(attr)
                d = (w[cond] - w["stratum:all"]).groupby("finding").median()
                cols.append(d); labs.append("%s\n%s" % (self.ATTR_LABELS[attr], self.COND_LABELS[cond]))
        M = pd.concat(cols, axis=1)
        M = M.loc[M.mean(axis=1).sort_values().index]
        lim = float(np.nanmax(np.abs(M.values)))
        im = ax.imshow(M.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, fontsize=figstyle.fs(12.0))
        ax.set_yticks(range(len(M)))
        ax.set_yticklabels([i.replace("_", " ") for i in M.index], fontsize=figstyle.fs(12.0))
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        cb = self.fig.colorbar(im, cax=cax)
        cb.set_label("Change from pooled", fontsize=figstyle.fs(13.0))
        cb.ax.tick_params(labelsize=figstyle.fs(12.0))
        cb.outline.set_visible(False)
        cax._figcheck_skip = True
        self._labels.append((ax, "c", "The change per finding"))

    def _panel_d(self, ax):
        figstyle.clean_axes(ax)
        rows = [(a, self._sub(a).auroc_gap.median(), self._sub(a).gap_fair_reference.median())
                for a in self.ATTRS]
        rows.sort(key=lambda r: r[1])
        for i, (a, obs, ref) in enumerate(rows):
            col = self.ATTR_COLORS[a]
            ax.plot([ref, obs], [i, i], color=col, lw=3.4, zorder=2, solid_capstyle="round")
            ax.plot([ref], [i], marker="D", ms=12, mfc="white", mec=figstyle.REF_GREY, mew=2.0, zorder=4)
            ax.plot([obs], [i], marker="o", ms=15, mfc=col, mec="white", mew=1.2, zorder=4)
            p, n = self.vs_ref[(a, "stratum:all")]
            ax.text(obs + 0.005, i, "%.3f (%s)" % (obs, self._fmt_p(p)), va="center", ha="left",
                    fontsize=figstyle.fs(12.5), color=figstyle.TEXT)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([self.ATTR_LABELS[a] for a, _, _ in rows], fontsize=figstyle.fs(13.5))
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlim(0, 0.185)
        ax.set_xlabel("Subgroup performance difference")
        self._labels.append((ax, "d", "Technical variables read as attributes"))

    def _panel_e(self, ax):
        figstyle.clean_axes(ax)
        for a in self.ATTRS:
            v = np.sort(self._sub(a).auroc_gap.values)
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                    color=self.ATTR_COLORS[a], lw=2.6, zorder=3)

        ax.axhline(0.5, color=figstyle.REF_GREY, ls="--", lw=1.2, zorder=1)
        ax.set_xlim(0, 0.30)
        ax.set_ylim(0, 1.10)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Subgroup performance difference")
        ax.set_ylabel("Share of the 26 combinations")
        self._labels.append((ax, "e", "The full distribution per attribute"))

    def _panel_f(self, ax):
        figstyle.clean_axes(ax)
        for j, (met, lab) in enumerate(self.MEASURES):
            vals = [(a, self._sub(a)[met].median()) for a in self.ATTRS]
            lo, hi = min(v for _, v in vals), max(v for _, v in vals)
            ax.plot([lo, hi], [j, j], color=figstyle.FAINT, lw=2.0, zorder=1)
            for a, v in vals:
                ax.plot([v], [j], marker="o", ms=15, mfc=self.ATTR_COLORS[a], mec="white",
                        mew=1.2, zorder=3)
        ax.set_yticks(range(len(self.MEASURES)))
        ax.set_yticklabels([lab for _, lab in self.MEASURES], fontsize=figstyle.fs(13.5))
        ax.set_ylim(-0.6, len(self.MEASURES) - 0.4)
        ax.set_xlim(0, 0.36)
        ax.set_xlabel("Subgroup performance difference")
        self._labels.append((ax, "f", "Every measure gives the same order"))

    def _panel_g(self, ax):
        figstyle.clean_axes(ax)
        x = np.arange(len(self.COND))
        for attr in self.DEMOG:
            col = self.ATTR_COLORS[attr]
            ref = [self._sub(attr, c).gap_fair_reference.median() for c in self.COND]
            ax.plot(x, ref, "-o", color=col, lw=2.8, ms=13, mfc=col, mec="white", mew=1.2, zorder=3)
            for xi, v in zip(x, ref):
                ax.text(xi, v + 0.0028, "%.3f" % v, ha="center", va="bottom",
                        fontsize=figstyle.fs(12.5), color=col)
            ax.text(x[-1] + 0.10, ref[-1], self.ATTR_LABELS[attr], color=col,
                    fontsize=figstyle.fs(13.5), va="center", ha="left")
        ax.set_xticks(x)
        ax.set_xticklabels([self.COND_LABELS[c] for c in self.COND], fontsize=figstyle.fs(13.5))
        ax.set_xlim(-0.28, len(self.COND) + 0.42)
        ax.set_ylim(0, 0.075)
        ax.set_ylabel("Fair-model reference")
        self._labels.append((ax, "g", "The reference inside a view"))

    def _legend(self, ax):
        ax.axis("off")
        handles = [Line2D([0], [0], marker="o", lw=0, ms=14, mfc=self.ATTR_COLORS[a], mec="white",
                          label=self.ATTR_LABELS[a]) for a in self.ATTRS]
        handles.append(Line2D([0], [0], marker="D", lw=0, ms=12, mfc="white", mec=figstyle.REF_GREY,
                              mew=2.0, label="Fair-model reference"))
        ax.legend(handles=handles, ncol=5, frameon=False, loc="center", bbox_to_anchor=(0.5, 0.5),
                  columnspacing=2.0, handletextpad=0.6)

    def build(self):
        self._labels = []
        self.fig = plt.figure(figsize=(18.5, 20.50), facecolor="white")
        gsl = self.fig.add_gridspec(1, 1, left=0.045, right=0.985, top=0.998, bottom=0.968)
        gs1 = self.fig.add_gridspec(1, 3, left=0.108, right=0.900, top=0.932, bottom=0.690,
                                    width_ratios=[1.00, 1.06, 0.94], wspace=0.72)
        gs2 = self.fig.add_gridspec(1, 2, left=0.115, right=0.985, top=0.615, bottom=0.360,
                                    width_ratios=[1.05, 1.00], wspace=0.30)
        gs3 = self.fig.add_gridspec(1, 2, left=0.115, right=0.985, top=0.278, bottom=0.058,
                                    width_ratios=[1.05, 1.00], wspace=0.30)

        self._legend(self.fig.add_subplot(gsl[0, 0]))
        self._panel_a(self.fig.add_subplot(gs1[0, 0]))
        self._panel_b(self.fig.add_subplot(gs1[0, 1]))
        cax = self.fig.add_axes([0.914, 0.703, 0.011, 0.216])
        self._panel_c(self.fig.add_subplot(gs1[0, 2]), cax)
        self._panel_d(self.fig.add_subplot(gs2[0, 0]))
        self._panel_e(self.fig.add_subplot(gs2[0, 1]))
        self._panel_f(self.fig.add_subplot(gs3[0, 0]))
        self._panel_g(self.fig.add_subplot(gs3[0, 1]))
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
