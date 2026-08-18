"""
figures/figure3_decodability.py
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


class Figure3Decodability:
    INJECTION_ENCODER = "rad_dino"
    FONT_SCALE = 1.094
    OUT_PDF = os.path.join(figstyle.FIGS, "figure3_decodability.pdf")

    DEC = [("inject:decodable_only:s0.00", "0"), ("inject:decodable_only:s0.25", "0.25"),
           ("inject:decodable_only:s0.50", "0.50"), ("inject:decodable_only:s1.00", "1.00")]
    ENT = [("inject:entangled:k0", "0"), ("inject:entangled:k1", "1"), ("inject:entangled:k4", "4"),
           ("inject:entangled:k16", "16"), ("inject:entangled:k64", "64"), ("inject:entangled:k256", "256")]
    OPERATORS = {"decodable_only": ("Decodability injection", figstyle.DECODABILITY, "o"),
                 "entangled": ("Entanglement injection", figstyle.RACE, "D")}
    QUANTITY_ROWS = [("decodability", "linear\ndecodability"),
                     ("decodability_nonlinear", "nonlinear\ndecodability"),
                     ("auroc_gap", "race\ndifference"),
                     ("ceiling_gap", "achievable\ndifference"),
                     ("auroc_overall", "disease\nAUROC")]
    RHO_ROWS = [("strength_vs_decodability", "decodability"),
                ("strength_vs_decodability_nonlinear", "nonlinear dec."),
                ("strength_vs_auroc_gap", "difference"),
                ("strength_vs_ceiling_gap", "achievable"),
                ("strength_vs_geometric_overlap", "overlap")]
    PROPERTY_LABELS = {"vs_ceiling::decodability": "linear decodability",
                       "vs_ceiling::leace_collateral": "erasure cost",
                       "vs_ceiling::geometric_overlap": "geometric overlap"}
    R2_SETS = [("cross_fit_r2_entanglement", "erasure cost, overlap"),
               ("cross_fit_r2_decodability_only", "linear dec."),
               ("cross_fit_r2_decodability_nonlinear_only", "nonlinear dec.")]

    def __init__(self, out_pdf=None):
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        figstyle.set_rcparams(Figure3Decodability.FONT_SCALE)
        self._load_data()
        self.fig = None
        self._labels = []

    def _load_data(self):
        perf = pd.read_csv(os.path.join(figstyle.TABLES, "results_performance.csv"), low_memory=False)
        stat = pd.read_csv(os.path.join(figstyle.TABLES, "results_statistics.csv"), low_memory=False)

        e2 = perf[(perf.experiment == "e2") & (perf.attribute == "race_grp")]
        base = e2[e2.mitigation.isin(["none", "leace", "inlp"])].pivot_table(
            index=["encoder", "finding", "mitigation"], columns="metric_name",
            values="value_raw", aggfunc="first").reset_index()
        ceil = e2[e2.mitigation.astype(str).str.startswith("ceiling:")]
        ceil = ceil[ceil.metric_name == "ceiling_gap"].set_index(["encoder", "finding"]).value_raw
        self.panel = base[base.mitigation == "none"].set_index(["encoder", "finding"]).join(
            ceil.rename("ceiling_gap")).reset_index()
        self.erasure = base

        e9 = perf[(perf.experiment == "e9") & (perf.encoder == self.INJECTION_ENCODER)
                  & (perf.data_composition.astype(str).str.startswith("inject:"))]
        unmit = e9[e9.mitigation == "none"].pivot_table(
            index=["data_composition", "finding"], columns="metric_name", values="value_raw", aggfunc="first")
        cg = e9[e9.mitigation.astype(str).str.startswith("ceiling:")]
        cg = cg[cg.metric_name == "ceiling_gap"].groupby(["data_composition", "finding"]).value_raw.first()
        unmit["ceiling_gap"] = cg
        self.inj = unmit

        s3 = stat[stat.experiment == "e3"]
        self.rho_geo = s3[s3.finding.astype(str).str.startswith("vs_ceiling::")].set_index("finding")
        self.r2 = s3[s3.estimate_name.astype(str).str.startswith("cross_fit_r2")].set_index("estimate_name")
        s9 = stat[(stat.experiment == "e9") & (stat.encoder == self.INJECTION_ENCODER)
                  & (stat.test_name == "spearman_permutation")
                  & (stat.mitigation.isin(list(self.OPERATORS)))]
        self.rho_inj = s9.set_index(["mitigation", "finding"])

        n_enc = e9.encoder.nunique()
        if n_enc != 1:
            raise ValueError(f"[fig3] expected one injection encoder, found {n_enc}")
        if len(self.panel) != 130:
            raise ValueError(f"expected 130 combinations, found {len(self.panel)}")

    def _mean_curve(self, keys, metric):
        rows = [k for k, _ in keys]
        return self.inj[metric].unstack(0)[rows].mean(axis=0).values

    def _per_finding(self, keys, metric):
        rows = [k for k, _ in keys]
        return self.inj[metric].unstack(0)[rows]

    def _panel_a(self, ax):
        figstyle.clean_axes(ax)
        g = self.panel.groupby("encoder").agg(dec=("decodability", "median"), gap=("auroc_gap", "median"),
                                              ceil=("ceiling_gap", "median")).sort_values("dec")
        lo, hi = g.gap.min(), g.gap.max()
        ax.axhspan(lo, hi, color=figstyle.RACE, alpha=0.10, zorder=0)
        ax.scatter(g.dec, g.gap, s=140, marker="o", color=figstyle.RACE, edgecolor="white",
                   linewidth=0.8, label="observed difference", zorder=3)
        ax.scatter(g.dec, g.ceil, s=125, marker="s", color=figstyle.ACHIEVABLE, edgecolor="white",
                   linewidth=0.8, label="achievable difference", zorder=3)
        ax.annotate("untrained ViT-S", xy=(g.dec.iloc[0], g.gap.iloc[0]),
                    xytext=(g.dec.iloc[0] + 0.015, g.gap.iloc[0] - 0.013), fontsize=figstyle.fs(14.5),
                    color=figstyle.TEXT, ha="left", va="top",
                    arrowprops=dict(arrowstyle="-", color=figstyle.FAINT, lw=1.0, shrinkA=2, shrinkB=4))
        ax.text(0.897, hi + 0.0042, f"span {hi - lo:.3f}", fontsize=figstyle.fs(14.5), color=figstyle.RACE,
                ha="right", va="bottom")
        ax.set_xlim(0.63, 0.90)
        ax.set_ylim(0.030, 0.105)
        ax.set_xlabel("Linear race decodability")
        ax.set_ylabel("Race difference")
        ax.legend(frameon=False, loc="upper left", fontsize=figstyle.fs(14.5), handletextpad=0.5, borderpad=0.2)
        self._labels.append((ax, "a", "Decodability across encoders"))

    def _panel_b(self, ax, ax2):
        figstyle.clean_axes(ax)
        keys = list(self.PROPERTY_LABELS)
        for i, k in enumerate(keys):
            rho = float(self.rho_geo.loc[k, "estimate"])
            p = float(self.rho_geo.loc[k, "p_fdr"])
            ax.plot([0, rho], [i, i], color=figstyle.FAINT, lw=2.2, zorder=1)
            ax.scatter([rho], [i], s=150, facecolor="white" if p >= 0.05 else figstyle.DECODABILITY,
                       edgecolor=figstyle.DECODABILITY, linewidth=1.8, zorder=3)
        ax.axvline(0, color=figstyle.REF_GREY, ls="--", lw=1.2, zorder=0)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([self.PROPERTY_LABELS[k] for k in keys], fontsize=figstyle.fs(15.0))
        ax.invert_yaxis()
        ax.set_xlim(-0.32, 0.32)
        ax.set_ylim(len(keys) - 0.4, -0.6)
        ax.set_xlabel("Spearman $\\rho$ with the\nachievable difference")

        figstyle.clean_axes(ax2)
        vals = [float(self.r2.loc[key, "estimate"]) for key, _ in self.R2_SETS]
        for i, v in enumerate(vals):
            ax2.barh(i, v, height=0.62, color=figstyle.DECODABILITY_NL, edgecolor="white", linewidth=0.8)
            ax2.text(v - 0.0012, i, f"{v:.3f}", fontsize=figstyle.fs(13.0), color=figstyle.TEXT,
                     ha="right", va="center")
        ax2.axvline(0, color=figstyle.REF_GREY, ls="--", lw=1.2, zorder=0)
        ax2.set_yticks(range(len(vals)))
        ax2.set_yticklabels([lab for _, lab in self.R2_SETS], fontsize=figstyle.fs(14.0))
        ax2.invert_yaxis()
        ax2.set_xlim(-0.030, 0.004)
        ax2.set_xticks([-0.02, -0.01, 0.0])
        ax2.set_xlabel("Cross-fitted out-of-sample $R^2$")
        self._labels.append((ax, "b", "Predictors of the difference"))

    def _panel_c(self, ax):
        figstyle.clean_axes(ax)
        rows = [("decodability", "linear\ndecodability", figstyle.DECODABILITY),
                ("decodability_nonlinear", "nonlinear\ndecodability", figstyle.DECODABILITY_NL),
                ("auroc_gap", "race\ndifference", figstyle.RACE),
                ("auroc_overall", "disease\nAUROC", figstyle.PERFORMANCE)]
        med = self.erasure.groupby("mitigation").median(numeric_only=True)
        paired = {}
        for m in ("leace", "inlp"):
            a = self.erasure[self.erasure.mitigation == "none"].set_index(["encoder", "finding"])
            b = self.erasure[self.erasure.mitigation == m].set_index(["encoder", "finding"])
            paired[m] = ((b.auroc_gap - a.auroc_gap).median(), (a.auroc_overall - b.auroc_overall).median())
        for i, (metric, _, col) in enumerate(rows):
            base = med.loc["none", metric]
            for m, marker in (("leace", "o"), ("inlp", "D")):
                ax.plot([base, med.loc[m, metric]], [i, i], color=figstyle.FAINT, lw=1.8, zorder=1)
                ax.scatter([med.loc[m, metric]], [i], s=115, marker=marker, facecolor="white",
                           edgecolor=col, linewidth=1.8, zorder=3)
            ax.scatter([base], [i], s=125, facecolor=col, edgecolor="white", linewidth=0.8, zorder=4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([lab for _, lab, _ in rows], fontsize=figstyle.fs(14.5))
        ax.invert_yaxis()
        ax.set_xlim(0, 1.72)
        ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_xlabel("Value")
        txt = ("paired change,\ndifference / AUROC\n"
               f"LEACE {paired['leace'][0]:+.3f} / {paired['leace'][1]:.3f}\n"
               f"INLP {paired['inlp'][0]:+.3f} / {paired['inlp'][1]:.3f}")
        ax.text(0.995, 0.52, txt, transform=ax.transAxes, fontsize=figstyle.fs(12.5), color=figstyle.TEXT,
                ha="right", va="center", linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=figstyle.FAINT, lw=1.0))
        handles = [Line2D([0], [0], marker="o", lw=0, ms=12, mfc=figstyle.PERFORMANCE, mec="white", label="unmitigated"),
                   Line2D([0], [0], marker="o", lw=0, ms=12, mfc="white", mec=figstyle.PERFORMANCE, mew=1.8, label="LEACE"),
                   Line2D([0], [0], marker="D", lw=0, ms=11, mfc="white", mec=figstyle.PERFORMANCE, mew=1.8, label="INLP")]
        ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=figstyle.fs(14.0),
                  handletextpad=0.4, borderpad=0.2)
        self._labels.append((ax, "c", "Erasure"))

    def _panel_d(self, ax):
        figstyle.clean_axes(ax)
        for mode, keys in (("decodable_only", self.DEC), ("entangled", self.ENT)):
            label, col, marker = self.OPERATORS[mode]
            x = self._mean_curve(keys, "decodability")
            y = self._mean_curve(keys, "auroc_gap")
            ax.plot(x, y, color=col, lw=2.0, zorder=2, alpha=0.85)
            ax.scatter(x, y, s=np.linspace(70, 230, len(x)), marker=marker, facecolor=col,
                       edgecolor="white", linewidth=0.9, zorder=3)
            for k in range(len(x) - 1):
                ax.annotate("", xy=(x[k + 1], y[k + 1]), xytext=(x[k], y[k]),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, alpha=0.9,
                                            shrinkA=7, shrinkB=9))
            ax.text(x[-1], y[-1] + (0.004 if mode == "entangled" else -0.005),
                    ("$k$ = 256" if mode == "entangled" else "$s$ = 1"), fontsize=figstyle.fs(14.5), color=col,
                    ha="center", va="bottom" if mode == "entangled" else "top")
        x0, y0 = self._mean_curve(self.DEC, "decodability")[0], self._mean_curve(self.DEC, "auroc_gap")[0]
        ax.scatter([x0], [y0], s=200, marker="*", color=figstyle.TEXT, zorder=4)
        ax.text(x0, y0 - 0.0035, "no injection", fontsize=figstyle.fs(14.5), color=figstyle.TEXT,
                ha="center", va="top")
        xs = np.concatenate([self._mean_curve(self.DEC, "decodability"), self._mean_curve(self.ENT, "decodability")])
        ys = np.concatenate([self._mean_curve(self.DEC, "auroc_gap"), self._mean_curve(self.ENT, "auroc_gap")])
        xpad = (xs.max() - xs.min()) * 0.16
        ax.set_xlim(xs.min() - xpad, xs.max() + xpad * 0.6)
        ax.set_ylim(ys.min() - 0.013, ys.max() + 0.020)
        ax.set_xlabel("Linear race decodability")
        ax.set_ylabel("Race difference")
        self._labels.append((ax, "d", "The path each operator takes"))

    def _heat(self, ax, mode, keys, cbar_ax=None):
        figstyle.clean_axes(ax)
        rows = [m for m, _ in self.QUANTITY_ROWS]
        vals = np.vstack([self._mean_curve(keys, m) for m in rows])
        delta = vals - vals[:, [0]]
        scale = float(np.abs(delta).max())
        im = ax.imshow(delta, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
        rgba = im.cmap(im.norm(delta))
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                r, g, b = rgba[i, j, :3]
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=figstyle.fs(12.5),
                        color="white" if lum < 0.45 else figstyle.TEXT)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([lab for _, lab in keys])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([lab for _, lab in self.QUANTITY_ROWS], fontsize=figstyle.fs(13.5))
        ax.set_xlabel("Injected strength $s$" if mode == "decodable_only" else "Deflated dimensions $k$")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        if cbar_ax is not None:
            cb = self.fig.colorbar(im, cax=cbar_ax)
            cb.set_label("Change from the uninjected value", fontsize=figstyle.fs(14.0))
            cb.ax.tick_params(labelsize=figstyle.fs(12.5))
            cb.outline.set_visible(False)
        return im

    def _panel_g(self, ax):
        figstyle.clean_axes(ax)
        base = self._per_finding(self.DEC, "auroc_gap")[self.DEC[0][0]]
        bars = [("no\ninjection", base, figstyle.PERFORMANCE),
                ("decodability", self._per_finding(self.DEC, "auroc_gap")[self.DEC[-1][0]],
                 figstyle.DECODABILITY),
                ("entanglement", self._per_finding(self.ENT, "auroc_gap")[self.ENT[-1][0]],
                 figstyle.RACE)]
        for i, (lab, v, col) in enumerate(bars):
            ax.bar(i, v.mean(), width=0.62, color=col, alpha=0.30, edgecolor=col, linewidth=1.5, zorder=2)
            ax.scatter(np.full(len(v), i), v.values, s=60, facecolor="white", edgecolor=col,
                       linewidth=1.4, zorder=4)
            ax.plot([i - 0.32, i + 0.32], [v.mean()] * 2, color=col, lw=2.8, zorder=5)
            ax.text(i, max(v.max(), v.mean()) + 0.006, f"{v.mean():.3f}", ha="center", va="bottom",
                    fontsize=figstyle.fs(13.5), color=col, zorder=6)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([lab for lab, _, _ in bars], fontsize=figstyle.fs(13.0),
                           rotation=18, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.65, len(bars) - 0.35)
        ax.set_ylabel("Race difference")
        ax.set_ylim(0, 0.17)
        self._labels.append((ax, "g", "The end of each grid"))

    def _panel_h(self, ax):
        figstyle.clean_axes(ax)
        y = np.arange(len(self.RHO_ROWS))
        for mode, off in (("decodable_only", -0.17), ("entangled", 0.17)):
            _, col, marker = self.OPERATORS[mode]
            for i, (key, _) in enumerate(self.RHO_ROWS):
                r = float(self.rho_inj.loc[(mode, key), "estimate"])
                p = float(self.rho_inj.loc[(mode, key), "p_fdr"])
                ax.scatter([r], [i + off], s=120, marker=marker,
                           facecolor=col if p < 0.05 else "white", edgecolor=col, linewidth=1.7, zorder=3)
        for i in y:
            ax.plot([-1, 1], [i, i], color=figstyle.HAIRLINE, lw=1.0, zorder=1)
        ax.axvline(0, color=figstyle.REF_GREY, ls="--", lw=1.2, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([lab for _, lab in self.RHO_ROWS], fontsize=figstyle.fs(14.0))
        ax.invert_yaxis()
        ax.set_xlim(-1.1, 1.1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_ylim(len(self.RHO_ROWS) - 0.4, -0.6)
        ax.set_xlabel("Spearman $\\rho$ with strength")
        self._labels.append((ax, "h", "Response to strength"))

    def _legend(self, ax):
        ax.axis("off")
        handles = [Line2D([0], [0], marker="o", lw=0, ms=14, mfc=figstyle.DECODABILITY, mec="white",
                          label="Decodability injection"),
                   Line2D([0], [0], marker="D", lw=0, ms=13, mfc=figstyle.RACE, mec="white",
                          label="Entanglement injection"),
                   Line2D([0], [0], marker="o", lw=0, ms=14, mfc="white", mec=figstyle.NEUTRAL, mew=1.7,
                          label="Not significant at an FDR of 0.05"),
                   Patch(facecolor=figstyle.RACE, alpha=0.10, edgecolor="none",
                         label="Span of the observed difference")]
        ax.legend(handles=handles, ncol=4, frameon=False, loc="center", bbox_to_anchor=(0.5, 0.5),
                  columnspacing=2.0, handletextpad=0.6)

    def build(self):
        self._labels = []
        self.fig = plt.figure(figsize=(18.5, 20.0), facecolor="white")
        gsl = self.fig.add_gridspec(1, 1, left=0.045, right=0.985, top=0.995, bottom=0.962)
        gs1 = self.fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.925, bottom=0.700,
                                    width_ratios=[1.15, 0.92, 1.05], wspace=0.50)
        gs2 = self.fig.add_gridspec(1, 2, left=0.075, right=0.905, top=0.615, bottom=0.375,
                                    width_ratios=[1.05, 1.20], wspace=0.38)
        gs3 = self.fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.290, bottom=0.055,
                                    width_ratios=[1.08, 0.96, 1.16], wspace=0.46)
        cax = self.fig.add_axes([0.930, 0.400, 0.011, 0.180])

        self._legend(self.fig.add_subplot(gsl[0, 0]))
        self._panel_a(self.fig.add_subplot(gs1[0, 0]))
        gsb = gs1[0, 1].subgridspec(2, 1, hspace=0.85, height_ratios=[1.0, 0.78])
        self._panel_b(self.fig.add_subplot(gsb[0, 0]), self.fig.add_subplot(gsb[1, 0]))
        self._panel_c(self.fig.add_subplot(gs1[0, 2]))
        self._panel_d(self.fig.add_subplot(gs2[0, 0]))
        ax_e = self.fig.add_subplot(gs2[0, 1])
        self._heat(ax_e, "decodable_only", self.DEC, cbar_ax=cax)
        self._labels.append((ax_e, "e", "Decodability injection"))
        ax_f = self.fig.add_subplot(gs3[0, 0])
        self._heat(ax_f, "entangled", self.ENT)
        self._labels.append((ax_f, "f", "Entanglement injection"))
        self._panel_g(self.fig.add_subplot(gs3[0, 1]))
        self._panel_h(self.fig.add_subplot(gs3[0, 2]))

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
