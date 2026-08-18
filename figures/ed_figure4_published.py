"""
figures/ed_figure4_published.py
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

AUDIT = ("/PATH/Repositories_target_files/bias_origin/"
         "results_e11_published/generalization_audit.csv")


class EDFigure4Published:
    FONT_SCALE = 1.088
    OUT_PDF = os.path.join(figstyle.FIGS, "ed_figure4_published.pdf")

    MODS = ["chest radiograph", "dermatology", "computational pathology",
            "funduscopy", "mammography", "abdominal CT"]
    MOD_LABELS = {"chest radiograph": "Chest radiograph", "dermatology": "Dermatology",
                  "computational pathology": "Pathology", "funduscopy": "Funduscopy",
                  "mammography": "Mammography", "abdominal CT": "Abdominal CT"}
    MOD_COLORS = {"chest radiograph": figstyle.PERFORMANCE, "dermatology": figstyle.RACE,
                  "computational pathology": figstyle.AGE, "funduscopy": figstyle.DECODABILITY,
                  "mammography": figstyle.ACHIEVABLE, "abdominal CT": figstyle.DECODABILITY_NL}
    MOD_MARKERS = {"chest radiograph": "o", "dermatology": "s", "computational pathology": "^",
                   "funduscopy": "D", "mammography": "v", "abdominal CT": "P"}
    STATS = [("rate", "Rate difference"), ("auroc", "AUROC difference")]
    DATASETS = {"mimic_cxr": "MIMIC-CXR", "chexpert": "CheXpert", "ddi": "DDI",
                "ddi_common": "DDI common", "mgb_breast": "MGB breast", "mgb_lung": "MGB lung",
                "tcga_gbmlgg": "TCGA glioma", "burlina_dr": "Burlina DR", "emory": "Emory",
                "pdac": "PDAC"}
    PAPERS = {"seyyed_kalantari_2021": "Seyyed-Kalantari 2021", "glocker_2023": "Glocker 2023",
              "lotter_2024": "Lotter 2024", "zong_2023_medfair": "MEDFAIR",
              "daneshjou_2022": "Daneshjou 2022", "vaidya_2024": "Vaidya 2024",
              "burlina_2021": "Burlina 2021", "yala_2021": "Yala 2021",
              "tayebi_arasteh_2024": "Tayebi Arasteh 2024"}

    def __init__(self, out_pdf=None):
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        figstyle.set_rcparams(EDFigure4Published.FONT_SCALE)
        self._load_data()
        self.fig = None
        self._labels = []

    def _load_data(self):
        a = pd.read_csv(AUDIT)
        a = a[a.auditable == True].copy()
        a["stat"] = np.where(a.metric == "auroc", "auroc", "rate")
        a["exceeds"] = a.significant_fdr05.astype(bool)
        self.a = a

    def _panel_a(self, ax):
        figstyle.clean_axes(ax)
        lim = self.a.reported_difference.max() * 1.06
        ax.plot([0, lim], [0, lim], ls="--", lw=1.3, color=figstyle.REF_GREY, zorder=1)
        for m in self.MODS:
            s = self.a[self.a.modality == m]
            col, mk = self.MOD_COLORS[m], self.MOD_MARKERS[m]
            for hi, kw in ((True, dict(facecolor=col, edgecolor="white", linewidth=0.8)),
                           (False, dict(facecolor="white", edgecolor=col, linewidth=1.6))):
                q = s[s.exceeds == hi]
                ax.scatter(q.reference, q.reported_difference, s=78, marker=mk, zorder=3 + int(hi),
                           alpha=0.9, **kw)
        ax.set_xlim(0, self.a.reference.max() * 1.10)
        ax.set_ylim(0, lim)
        ax.set_xticks([0.05, 0.10, 0.15])
        ax.set_xlabel("Fair-model reference")
        ax.set_ylabel("Difference the study reported")
        self._labels.append((ax, "a", "Claims against their references"))

    def _panel_b(self, ax):
        figstyle.clean_axes(ax)
        rng = np.random.RandomState(0)
        pos = 0
        ticks, labs = [], []
        for key, lab in self.STATS:
            for m in self.MODS:
                s = self.a[(self.a.stat == key) & (self.a.modality == m)]
                if not len(s):
                    continue
                v = s.reference_share_pct.dropna().values
                col = self.MOD_COLORS[m]
                if len(v) >= 3 and np.ptp(v) > 0:
                    parts = ax.violinplot([v], positions=[pos], vert=False, widths=0.86,
                                          showextrema=False, showmedians=False)
                    for b in parts["bodies"]:
                        b.set_facecolor(col); b.set_alpha(0.30); b.set_edgecolor(col); b.set_linewidth(1.2)
                ax.scatter(v, np.full(len(v), pos) + rng.uniform(-0.13, 0.13, len(v)), s=30,
                           facecolor=col, edgecolor="none", alpha=0.75, zorder=3)
                ax.plot([np.median(v)] * 2, [pos - 0.30, pos + 0.30], color=figstyle.TEXT, lw=2.8, zorder=4)
                ticks.append(pos)
                labs.append("%s\n%d of %d exceed" % (self.MOD_LABELS[m], int(s.exceeds.sum()), len(s)))
                pos -= 1
            pos -= 0.6
        ax.axvline(100, color=figstyle.REF_GREY, ls="--", lw=1.3, zorder=1)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labs, fontsize=figstyle.fs(12.0))
        ax.set_ylim(pos + 0.7, 0.7)
        ax.set_xlim(-4, 165)
        ax.set_xlabel("Reference as a share of the reported value (%)")
        ax.text(0.985, 0.985, "Rate differences", transform=ax.transAxes, ha="right", va="top",
                fontsize=figstyle.fs(13.5), color=figstyle.TEXT)
        ax.text(0.985, 0.40, "AUROC differences", transform=ax.transAxes, ha="right", va="top",
                fontsize=figstyle.fs(13.5), color=figstyle.TEXT)
        self._labels.append((ax, "b", "The reference as a share"))

    def _panel_c(self, ax):
        figstyle.clean_axes(ax)
        rows = []
        for p, lab in self.PAPERS.items():
            s = self.a[self.a.paper == p]
            rows.append((lab, int(s.exceeds.sum()), len(s), self.MOD_COLORS[s.modality.iloc[0]]))
        rows.sort(key=lambda r: r[2])
        for i, (lab, k, n, col) in enumerate(rows):
            ax.barh(i, n, height=0.66, color=figstyle.FAINT, edgecolor="none", zorder=2)
            ax.barh(i, k, height=0.66, color=col, edgecolor="white", linewidth=1.0, zorder=3)
            ax.text(n + 0.5, i, "%d of %d" % (k, n), va="center", ha="left",
                    fontsize=figstyle.fs(12.5), color=figstyle.TEXT)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=figstyle.fs(12.5))
        ax.set_ylim(-0.62, len(rows) - 0.38)
        ax.set_xlim(0, 30)
        ax.set_xlabel("Claims")
        self._labels.append((ax, "c", "By study"))

    def _panel_d(self, ax):
        figstyle.clean_axes(ax)
        for m in self.MODS:
            s = self.a[self.a.modality == m]
            col, mk = self.MOD_COLORS[m], self.MOD_MARKERS[m]
            for hi, kw in ((True, dict(facecolor=col, edgecolor="white", linewidth=0.8)),
                           (False, dict(facecolor="white", edgecolor=col, linewidth=1.6))):
                q = s[s.exceeds == hi]
                ax.scatter(q.smallest_denominator, q.reference, s=78, marker=mk, alpha=0.9,
                           zorder=3 + int(hi), **kw)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cases in the smallest subgroup")
        ax.set_ylabel("Fair-model reference")
        self._labels.append((ax, "d", "The reference follows subgroup size"))

    def _panel_e(self, ax):
        figstyle.clean_axes(ax)
        r = self.a[self.a.stat == "rate"]
        xs, ys, cs, ns = [], [], [], []
        for m in self.MODS:
            s = r[r.modality == m]
            if not len(s):
                continue
            xs.append(s.smallest_denominator.median())
            ys.append(s.reference_share_pct.median())
            cs.append(self.MOD_COLORS[m]); ns.append((self.MOD_LABELS[m], len(s)))
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], color=figstyle.FAINT, lw=2.4, zorder=1)
        order = np.argsort(xs)
        for k, idx in enumerate(order):
            x, y, c, (lab, n) = xs[idx], ys[idx], cs[idx], ns[idx]
            ax.plot([x], [y], marker="o", ms=19, mfc=c, mec="white", mew=1.4, zorder=3)
            up = k % 2 == 0
            ax.text(x, y + (6.5 if up else -7.5), "%s\n%d%% at %d cases" % (lab, round(y), round(x)),
                    ha="center", va="bottom" if up else "top",
                    fontsize=figstyle.fs(12.0), color=c)
        ax.set_xscale("log")
        ax.set_xlim(20, 4200)
        ax.set_ylim(0, 108)
        ax.set_xlabel("Median smallest subgroup (cases)")
        ax.set_ylabel("Median reference share (%)")
        self._labels.append((ax, "e", "Across the modalities"))

    def _panel_f(self, ax):
        figstyle.clean_axes(ax)
        s = self.a.loc[self.a.groupby("dataset").reported_difference.idxmax()]
        s = s.sort_values("reported_difference")
        for i, r in enumerate(s.itertuples()):
            col = self.MOD_COLORS[r.modality]
            ax.plot([r.reference_ci_low, r.reference_ci_high], [i, i], color=figstyle.REF_GREY,
                    lw=2.4, solid_capstyle="round", zorder=2)
            ax.plot([r.reference], [i], marker="D", ms=11, mfc="white", mec=figstyle.REF_GREY,
                    mew=2.0, zorder=4)
            ax.plot([r.reported_difference], [i], marker="o", ms=14, zorder=4,
                    mfc=col if r.exceeds else "white", mec="white" if r.exceeds else col,
                    mew=1.2 if r.exceeds else 1.8)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(["%s, %s\n%s, $n=%d$" % (self.DATASETS.get(r.dataset, r.dataset),
                                                    str(r.metric).upper(),
                                                    self.PAPERS[r.paper].split()[0],
                                                    int(r.smallest_denominator))
                            for r in s.itertuples()], fontsize=figstyle.fs(11.5))
        ax.set_ylim(-0.62, len(s) - 0.38)
        ax.set_xlim(0, 0.56)
        ax.set_xlabel("Difference")
        self._labels.append((ax, "f", "The largest claim in each cohort"))

    def _panel_g(self, ax):
        figstyle.clean_axes(ax)
        order = ["reported", "derived_from_reported", "reconstructed_from_pool_shares"]
        labs = {"reported": "Printed in the article", "derived_from_reported": "Derived by arithmetic",
                "reconstructed_from_pool_shares": "Reconstructed from composition"}
        left = np.zeros(len(order))
        base = np.arange(len(order))
        for m in self.MODS:
            v = np.array([len(self.a[(self.a.counts_source == o) & (self.a.modality == m)]) for o in order])
            ax.barh(base, v, left=left, height=0.6, color=self.MOD_COLORS[m], alpha=0.85,
                    edgecolor="white", linewidth=1.0, zorder=3)
            left = left + v
        for i, tot in enumerate(left):
            ax.text(tot + 0.6, i, "%d" % tot, va="center", ha="left",
                    fontsize=figstyle.fs(12.5), color=figstyle.TEXT)
        ax.set_yticks(base)
        ax.set_yticklabels([labs[o] for o in order], fontsize=figstyle.fs(12.5))
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlim(0, 86)
        ax.set_xlabel("Claims")
        self._labels.append((ax, "g", "The provenance of the counts"))

    def _legend(self, ax):
        ax.axis("off")
        handles = [Line2D([0], [0], marker=self.MOD_MARKERS[m], lw=0, ms=13,
                          mfc=self.MOD_COLORS[m], mec="white", label=self.MOD_LABELS[m])
                   for m in self.MODS]
        handles += [Line2D([0], [0], marker="o", lw=0, ms=13, mfc="white", mec=figstyle.PERFORMANCE,
                           mew=1.8, label="Does not exceed its reference"),
                    Line2D([0], [0], marker="D", lw=0, ms=12, mfc="white", mec=figstyle.REF_GREY,
                           mew=2.0, label="Fair-model reference")]
        ax.legend(handles=handles, ncol=5, frameon=False, loc="center", bbox_to_anchor=(0.5, 0.5),
                  columnspacing=2.0, handletextpad=0.6)

    def build(self):
        self._labels = []
        self.fig = plt.figure(figsize=(18.5, 20.20), facecolor="white")
        gsl = self.fig.add_gridspec(1, 1, left=0.045, right=0.985, top=0.998, bottom=0.968)
        gs1 = self.fig.add_gridspec(1, 3, left=0.070, right=0.985, top=0.930, bottom=0.690,
                                    width_ratios=[1.00, 1.22, 0.92], wspace=0.62)
        gs2 = self.fig.add_gridspec(1, 2, left=0.070, right=0.985, top=0.610, bottom=0.370,
                                    width_ratios=[1.00, 1.06], wspace=0.28)
        gs3 = self.fig.add_gridspec(1, 2, left=0.150, right=0.985, top=0.288, bottom=0.055,
                                    width_ratios=[1.20, 0.88], wspace=0.42)

        self._legend(self.fig.add_subplot(gsl[0, 0]))
        self._panel_a(self.fig.add_subplot(gs1[0, 0]))
        self._panel_b(self.fig.add_subplot(gs1[0, 1]))
        self._panel_c(self.fig.add_subplot(gs1[0, 2]))
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
