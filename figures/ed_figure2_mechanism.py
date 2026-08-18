"""
figures/ed_figure2_mechanism.py
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


class EDFigure2Mechanism:
    FONT_SCALE = 1.083
    OUT_PDF = os.path.join(figstyle.FIGS, "ed_figure2_mechanism.pdf")

    RHO_ORDER = ["rho0.00", "rho0.10", "rho0.25", "rho0.50", "rho0.75", "rho1.00"]
    GEOM = ["decodability", "geometric_overlap", "leace_collateral"]
    GEOM_LABELS = {"decodability": "Linear\ndecodability",
                   "geometric_overlap": "Geometric\noverlap",
                   "leace_collateral": "Erasure\ncost"}
    R2_ROWS = [("cross_fit_r2_entanglement", "Entanglement"),
               ("cross_fit_r2_decodability_only", "Linear\ndecodability"),
               ("cross_fit_r2_decodability_nonlinear_only", "Nonlinear\ndecodability"),
               ("logo_r2_entanglement", "Entanglement,\nencoder held out")]
    MOD_LABELS = {"cxr": "Chest radiograph", "derm": "Dermatology", "fundus": "Funduscopy"}
    MOD_COLORS = {"cxr": figstyle.PERFORMANCE, "derm": figstyle.RACE, "fundus": figstyle.AGE}

    def __init__(self, out_pdf=None):
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        figstyle.set_rcparams(EDFigure2Mechanism.FONT_SCALE)
        self._load_data()
        self.fig = None
        self._labels = []

    def _load_data(self):
        perf = pd.read_csv(os.path.join(figstyle.TABLES, "results_performance.csv"), low_memory=False)
        stat = pd.read_csv(os.path.join(figstyle.TABLES, "results_statistics.csv"), low_memory=False)

        syn = perf[perf.modality == "synthetic"]
        self.syn = syn.pivot_table(index="data_composition", columns="metric_name",
                                   values="value_raw", aggfunc="first").reindex(self.RHO_ORDER)
        self.rho = np.array([float(r.replace("rho", "")) for r in self.RHO_ORDER])
        s7 = stat[stat.experiment == "e7"]
        self.syn_rho = {r.finding: (r.estimate, r.p_fdr, int(r.n_units)) for r in s7.itertuples()}

        e3 = perf[(perf.experiment == "e3") & (perf.modality == "cxr")]
        self.units = e3.pivot_table(index=["encoder", "finding"], columns="metric_name",
                                    values="value_raw", aggfunc="first").reset_index()

        s3 = stat[(stat.experiment == "e3") & (stat.estimate_name.str.contains("r2", na=False))]
        self.r2 = {r.estimate_name: (r.estimate, r.p_fdr, bool(r.significant_fdr05), int(r.n_units))
                   for r in s3.itertuples()}
        s5 = stat[stat.estimate_name.str.startswith("transfer_r2_cxr_to")]
        self.transfer = {r.estimate_name.split("_to_")[1]: (r.estimate, r.p_fdr, int(r.n_units))
                         for r in s5.itertuples()}

        e4 = perf[(perf.experiment == "e4") & (perf.metric_name == "ceiling_gap")]
        self.target = {"cxr": self.units.ceiling_gap_target.values,
                       "derm": e4[(e4.modality == "derm") & (e4.attribute == "fst_grp")].value_raw.values,
                       "fundus": e4[(e4.modality == "fundus") & (e4.attribute == "race_grp")].value_raw.values}

    @staticmethod
    def _fmt_p(p):
        return "$P<0.001$" if p < 0.001 else ("$P>0.999$" if p > 0.999 else "$P=%.3f$" % p)

    def _panel_a(self, ax):
        figstyle.clean_axes(ax)
        gap, ceil = self.syn["gap"].values, self.syn["ceiling"].values
        ax.fill_between(self.rho, ceil, gap, color=figstyle.FAINT, alpha=0.75, zorder=1,
                        label="Removable by mitigation")
        ax.fill_between(self.rho, 0, ceil, color=figstyle.ACHIEVABLE, alpha=0.55, zorder=1,
                        label="Achievable difference")
        ax.plot(self.rho, gap, "-o", color=figstyle.RACE, lw=2.4, ms=10, zorder=3,
                label="Observed difference")
        ax.plot(self.rho, ceil, "-D", color=figstyle.DECODABILITY, lw=2.0, ms=8, zorder=3)
        rho_c, p_c, n_c = self.syn_rho["rho_vs_ceiling"]
        ax.text(0.03, 0.93, "Collinearity vs achievable difference\n$\\rho_s=%.3f$, %s, $n=%d$"
                % (rho_c, self._fmt_p(p_c), n_c), transform=ax.transAxes, fontsize=figstyle.fs(13.5),
                va="top", ha="left", color=figstyle.TEXT)
        ax.set_xlabel("Collinearity")
        ax.set_ylabel("Difference")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(0, max(gap) * 1.30)
        ax.legend(frameon=False, loc="upper left", fontsize=figstyle.fs(13.0), bbox_to_anchor=(0.0, 0.72),
                  handletextpad=0.6, borderpad=0.2, labelspacing=0.35)
        self._labels.append((ax, "a", "The synthetic model across collinearity"))

    def _panel_b(self, ax):
        figstyle.clean_axes(ax)
        ov, cl = self.syn["geometric_overlap"].values, self.syn["leace_collateral"].values
        ax.plot(self.rho, ov, "-s", color=figstyle.DECODABILITY, lw=2.2, ms=9)
        ax.plot(self.rho, cl, "-^", color=figstyle.ACHIEVABLE, lw=2.2, ms=9)
        ax.set_xlabel("Collinearity")
        ax.set_ylabel("Overlap and erasure cost")
        ax.set_xlim(-0.03, 1.62)
        ax.set_ylim(-0.012, 0.24)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.text(1.06, ov[-1], "Geometric\noverlap", color=figstyle.DECODABILITY, fontsize=figstyle.fs(13.5),
                va="center", ha="left")
        ax.text(1.06, cl[-1], "Erasure cost\n(AUROC)", color=figstyle.ACHIEVABLE, fontsize=figstyle.fs(13.5),
                va="center", ha="left")

        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        dec = self.syn["decodability"].values
        ax2.plot(self.rho, dec, "--o", color=figstyle.AGE, lw=2.2, ms=9)
        ax2.set_ylabel("Linear decodability")
        ax2.set_ylim(0.560, 0.638)

        ax2.text(1.06, dec[-1], "Linear\ndecodability", color=figstyle.AGE, fontsize=figstyle.fs(13.5),
                 va="center", ha="left")
        rho_d, p_d, n_d = self.syn_rho["rho_vs_decodability"]
        ax.text(0.28, 0.99, "$\\rho_s=%.3f$, %s, $n=%d$" % (rho_d, self._fmt_p(p_d), n_d),
                transform=ax.transAxes, fontsize=figstyle.fs(13.0), va="top", ha="left",
                color=figstyle.TEXT)
        self._labels.append((ax, "b", "Geometry and decodability move apart"))

    def _panel_c(self, ax):
        figstyle.clean_axes(ax)
        gap, ceil = self.syn["gap"].values, self.syn["ceiling"].values
        share = 100.0 * ceil / gap
        x = np.arange(len(self.rho))
        ax.bar(x, share, width=0.66, color=figstyle.ACHIEVABLE, edgecolor="white", linewidth=1.0,
               label="Achievable, so not removable")
        ax.bar(x, 100.0 - share, width=0.66, bottom=share, color=figstyle.FAINT, edgecolor="white",
               linewidth=1.0, label="Removable by mitigation")
        for xi, s in zip(x, share):
            ax.text(xi, 102.0, "%d%%" % round(s), ha="center", va="bottom",
                    fontsize=figstyle.fs(13.0), color=figstyle.DECODABILITY)
        ax.set_xticks(x)
        ax.set_xticklabels(["%.2f" % r for r in self.rho])
        ax.set_xlabel("Collinearity")
        ax.set_ylabel("Share of the difference (%)")
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.legend(frameon=False, loc="upper center", fontsize=figstyle.fs(13.0), ncol=1,
                  bbox_to_anchor=(0.5, 1.04), handletextpad=0.6, borderpad=0.2, labelspacing=0.3)
        self._labels.append((ax, "c", "The removable share"))

    def _panel_d(self, ax_top, ax_bot):
        u = self.units
        for ax, key, lab in ((ax_top, "encoder", "By encoder"), (ax_bot, "finding", "By finding")):
            figstyle.clean_axes(ax)
            order = u.groupby(key).ceiling_gap_target.median().sort_values().index.tolist()
            rng = np.random.RandomState(0)
            for i, g in enumerate(order):
                v = u[u[key] == g].ceiling_gap_target.values
                ax.scatter(v, np.full(len(v), i) + rng.uniform(-0.16, 0.16, len(v)), s=26,
                           facecolor=figstyle.DECODABILITY if key == "encoder" else figstyle.AGE,
                           edgecolor="none", alpha=0.55, zorder=2)
                ax.plot([np.median(v)] * 2, [i - 0.34, i + 0.34], color=figstyle.TEXT, lw=2.6, zorder=3)
            ax.set_yticks([])
            ax.set_ylim(-0.7, len(order) - 0.3)
            ax.set_xlim(0, u.ceiling_gap_target.max() * 1.05)
            ax.set_ylabel("%s\n(%d groups)" % (lab, len(order)), fontsize=figstyle.fs(14.0))
        ax_top.set_xticklabels([])
        ax_top.set_xlabel("")
        ax_bot.set_xlabel("Achievable race difference")
        self._labels.append((ax_top, "d", "The target moves with the finding"))

    def _panel_e(self, ax):
        figstyle.clean_axes(ax)
        u = self.units
        prof = u.groupby("encoder")[self.GEOM].median()
        z = (prof - prof.mean()) / prof.std()
        tgt = u.groupby("encoder").ceiling_gap_target.median()
        lo, hi = tgt.idxmin(), tgt.idxmax()
        x = np.arange(len(self.GEOM))
        for enc in z.index:
            hot = enc in (lo, hi)
            ax.plot(x, z.loc[enc, self.GEOM].values, "-o",
                    color=figstyle.DECODABILITY if enc == hi else (figstyle.AGE if enc == lo else figstyle.FAINT),
                    lw=2.8 if hot else 1.6, ms=9 if hot else 6, zorder=3 if hot else 2,
                    alpha=1.0 if hot else 0.9)
        ax.axhline(0, color=figstyle.REF_GREY, ls="--", lw=1.1, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([self.GEOM_LABELS[g] for g in self.GEOM])
        ax.set_ylabel("Standardized value")
        ax.set_xlim(-0.28, len(self.GEOM) - 0.30)
        for enc, lab, col in ((hi, "Largest\ntarget", figstyle.DECODABILITY),
                              (lo, "Smallest\ntarget", figstyle.AGE)):
            ax.text(len(self.GEOM) - 0.92, z.loc[enc, self.GEOM[-1]], lab, color=col,
                    fontsize=figstyle.fs(13.0), va="center", ha="left")
        ax.text(-0.22, z[self.GEOM[0]].max() + 0.30, "The other 8 encoders", color=figstyle.TEXT,
                fontsize=figstyle.fs(13.0), va="bottom", ha="left")
        self._labels.append((ax, "e", "Geometry by encoder"))

    def _panel_f(self, ax):
        figstyle.clean_axes(ax)
        vals, labs, sig, ps, ns = [], [], [], [], []
        for key, lab in self.R2_ROWS:
            est, p, s, n = self.r2[key]
            vals.append(est); labs.append(lab); sig.append(s); ps.append(p); ns.append(n)
        x = np.array([0.0, 1.0, 2.0, 3.35])
        for xi, v, s in zip(x, vals, sig):
            ax.bar(xi, v, width=0.6, color=figstyle.DECODABILITY if s else "white",
                   edgecolor=figstyle.DECODABILITY, linewidth=1.8, zorder=3)
        ax.axhline(0, color=figstyle.REF_GREY, ls="--", lw=1.3, zorder=2)
        for xi, v, p in zip(x, vals, ps):
            off = 0.0035 if v >= 0 else -0.0035
            ax.text(xi, v + off, "%.3f\n%s" % (v, self._fmt_p(p)), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=figstyle.fs(13.0), color=figstyle.TEXT)
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=figstyle.fs(12.5), rotation=18, ha="right",
                           rotation_mode="anchor")
        ax.set_xlim(-0.95, 4.05)
        ax.set_ylim(-0.040, 0.046)
        ax.set_yticks([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04])
        ax.set_ylabel("Cross-fitted out-of-sample $R^2$")
        self._labels.append((ax, "f", "Predicting the target from the geometry"))

    def _panel_g(self, ax):
        figstyle.clean_axes(ax)
        keys = ["derm", "fundus"]
        y = np.arange(len(keys))[::-1]
        for yi, k in zip(y, keys):
            est, p, n = self.transfer[k]
            c = self.MOD_COLORS[k]
            ax.plot([0, est], [yi, yi], color=c, lw=3.0, zorder=2, solid_capstyle="butt")
            ax.plot([est], [yi], "o", ms=15, mfc="white", mec=c, mew=2.4, zorder=3)
            ax.text(est - 0.13, yi, "%.2f" % est, ha="right", va="center", fontsize=figstyle.fs(14.0),
                    color=figstyle.TEXT)
            ax.text(-0.10, yi + 0.30, "%s ($n=%d$, %s)" % (self.MOD_LABELS[k], n, self._fmt_p(p)),
                    ha="right", va="bottom", fontsize=figstyle.fs(13.5), color=figstyle.TEXT)
        ax.axvline(0, color=figstyle.REF_GREY, ls="--", lw=1.3, zorder=1)
        ax.set_yticks([])
        ax.set_ylim(-0.6, len(keys) - 0.25)
        ax.set_xlim(-3.95, 0.65)
        ax.set_xlabel("Out-of-sample $R^2$ of the transferred predictor")
        ax.text(0.06, 0.02, "0 is the mean of the target modality", transform=ax.transAxes,
                fontsize=figstyle.fs(13.0), ha="left", va="bottom", color=figstyle.TEXT)
        self._labels.append((ax, "g", "The predictor in another modality"))

    def _panel_h(self, ax):
        figstyle.clean_axes(ax)
        rng = np.random.RandomState(0)
        for i, k in enumerate(["cxr", "derm", "fundus"]):
            v = self.target[k]
            c = self.MOD_COLORS[k]
            ax.scatter(v, np.full(len(v), i) + rng.uniform(-0.20, 0.20, len(v)), s=44,
                       facecolor=c, edgecolor="none", alpha=0.5, zorder=2)
            m = float(np.median(v))
            ax.plot([m, m], [i - 0.30, i + 0.30], color=figstyle.TEXT, lw=3.0, zorder=3)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["%s\n($n=%d$, median %.3f)" % (self.MOD_LABELS[k], len(self.target[k]),
                                                           float(np.median(self.target[k])))
                            for k in ["cxr", "derm", "fundus"]], fontsize=figstyle.fs(13.0))
        ax.set_ylim(-0.60, 2.60)
        ax.set_xlim(0, max(self.target["cxr"].max(), 0.11) * 1.05)
        ax.set_xlabel("Achievable difference")
        self._labels.append((ax, "h", "The target in each modality"))

    def build(self):
        self._labels = []
        self.fig = plt.figure(figsize=(18.5, 17.90), facecolor="white")
        gs1 = self.fig.add_gridspec(1, 3, left=0.062, right=0.955, top=0.963, bottom=0.710,
                                    width_ratios=[1.20, 1.12, 0.92], wspace=0.50)
        gs2 = self.fig.add_gridspec(1, 3, left=0.062, right=0.985, top=0.640, bottom=0.300,
                                    width_ratios=[0.98, 0.92, 1.20], wspace=0.30)
        gs3 = self.fig.add_gridspec(1, 2, left=0.062, right=0.985, top=0.232, bottom=0.057,
                                    width_ratios=[0.86, 1.24], wspace=0.24)

        self._panel_a(self.fig.add_subplot(gs1[0, 0]))
        self._panel_b(self.fig.add_subplot(gs1[0, 1]))
        self._panel_c(self.fig.add_subplot(gs1[0, 2]))
        sub = gs2[0, 0].subgridspec(2, 1, hspace=0.14)
        self._panel_d(self.fig.add_subplot(sub[0, 0]), self.fig.add_subplot(sub[1, 0]))
        self._panel_e(self.fig.add_subplot(gs2[0, 1]))
        self._panel_f(self.fig.add_subplot(gs2[0, 2]))
        self._panel_g(self.fig.add_subplot(gs3[0, 0]))
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
