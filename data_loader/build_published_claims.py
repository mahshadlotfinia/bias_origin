"""
data_loader/build_published_claims.py
Created on August 12, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List

import pandas as pd

from config.serde import read_config


COLUMNS = ["claim_id", "paper", "modality", "dataset", "split", "task", "attribute", "condition",
           "subgroup", "metric", "reported_value", "reported_difference_direct", "denominator",
           "n_pos", "n_neg", "overall_value", "counts_source", "src", "note"]


def _claim(rows: List[Dict], cid, paper, modality, dataset, split, task, attribute, condition,
           metric, subgroups, counts_source, src, note, overall=None, difference=None):
    for name, value, n_pos, n_neg in subgroups:
        den = n_pos if metric == "tpr" else (n_neg if metric == "fpr" else n_pos + n_neg)
        rows.append({"claim_id": cid, "paper": paper, "modality": modality,
                     "dataset": dataset, "split": split,
                     "task": task, "attribute": attribute, "condition": condition,
                     "subgroup": name, "metric": metric, "reported_value": value,
                     "reported_difference_direct": difference,
                     "denominator": den, "n_pos": n_pos, "n_neg": n_neg,
                     "overall_value": overall, "counts_source": counts_source,
                     "src": src, "note": note})


def _not_auditable(rows, paper, reason):
    rows.append({"claim_id": f"{paper}__not_auditable", "paper": paper, "modality": "",
                 "dataset": "",
                 "split": "", "task": "", "attribute": "", "condition": "",
                 "subgroup": "", "metric": "auroc", "reported_value": float("nan"),
                 "reported_difference_direct": float("nan"),
                 "denominator": float("nan"), "n_pos": float("nan"), "n_neg": float("nan"),
                 "overall_value": float("nan"), "counts_source": "none",
                 "src": "whole article", "note": reason})


def build_claims() -> pd.DataFrame:
    r: List[Dict] = []

    _claim(r, "SK1", "seyyed_kalantari_2021", "chest radiograph", "mimic_cxr", "test", "no_finding",
           "race_sex", "baseline", "fpr",
           [("White Male", 1153 / 6688, 0, 6688),
            ("White Female", 823 / 4930, 0, 4930),
            ("Black Male", 295 / 1152, 0, 1152),
            ("Black Female", 512 / 1772, 0, 1772)],
           "reported", "Supplementary Tables S5 and S3",
           "Rates recomputed from the printed false-positive and Not-healthy counts, "
           "not read off a figure. Denominators cross-checked against Table S3.")

    gl_src = "Tables 1, 2 and 3; overall AUC from the Results"
    gl_note = ("Negatives derived as scans minus label positives, both printed in Table 1. "
               "AUC values are printed to two decimals, so the reported difference is "
               "quantized at 0.01.")
    gl = {
        ("chexpert", "no_finding"): ((2509, 27335), (495, 5155), (312, 2434), 0.87),
        ("chexpert", "pleural_effusion"): ((12152, 17692), (2461, 3189), (897, 1849), 0.87),
        ("mimic_cxr", "no_finding"): ((12142, 30256), (659, 1423), (3891, 6891), 0.85),
        ("mimic_cxr", "pleural_effusion"): ((11446, 30952), (578, 1504), (1819, 8963), 0.90),
    }
    gl_values = {
        ("chexpert", "no_finding"): {"auroc": (0.87, 0.88, 0.88), "tpr": (0.79, 0.80, 0.84),
                                     "fpr": (0.20, 0.20, 0.23)},
        ("chexpert", "pleural_effusion"): {"auroc": (0.86, 0.88, 0.86), "tpr": (0.77, 0.78, 0.71),
                                           "fpr": (0.21, 0.19, 0.16)},
        ("mimic_cxr", "no_finding"): {"auroc": (0.85, 0.86, 0.85), "tpr": (0.75, 0.74, 0.80),
                                      "fpr": (0.19, 0.17, 0.25)},
        ("mimic_cxr", "pleural_effusion"): {"auroc": (0.89, 0.90, 0.91), "tpr": (0.84, 0.84, 0.79),
                                            "fpr": (0.22, 0.20, 0.15)},
    }
    n = 0
    for (ds, task), ((wp, wn), (ap, an), (bp, bn), overall_auc) in gl.items():
        for metric, (vw, va, vb) in gl_values[(ds, task)].items():
            n += 1
            _claim(r, f"GL{n}", "glocker_2023", "chest radiograph", ds, "test", task, "race", "original", metric,
                   [("White", vw, wp, wn), ("Asian", va, ap, an), ("Black", vb, bp, bn)],
                   "reported", gl_src, gl_note,
                   overall=overall_auc if metric == "auroc" else None)

    lo_counts = {"Asian": (1303, 689), "Black": (5922, 4413), "White": (25840, 12442)}
    lo_note = ("Denominators derived as the Fig. 4 per-race image counts times the "
               "Supplementary Table 4 per-race No-Findings share. Images whose 'no finding' "
               "label is missing are absorbed into the Findings-Present count, so the "
               "positive denominator is a slight over-estimate and the reference a slight "
               "under-estimate.")
    lo_sens = {
        ("mxr", "baseline"): (0.805, 0.758, 0.835),
        ("mxr", "data_augmentation"): (0.803, 0.743, 0.840),
        ("mxr", "per_view_threshold"): (0.823, 0.791, 0.833),
        ("cxp", "baseline"): (0.778, 0.741, 0.816),
        ("cxp", "data_augmentation"): (0.774, 0.732, 0.827),
        ("cxp", "per_view_threshold"): (0.789, 0.762, 0.816),
    }
    lo_spec = {
        ("mxr", "baseline"): (0.805, 0.835, 0.743),
        ("mxr", "data_augmentation"): (0.796, 0.839, 0.740),
        ("mxr", "per_view_threshold"): (0.736, 0.756, 0.677),
        ("cxp", "baseline"): (0.788, 0.800, 0.691),
        ("cxp", "data_augmentation"): (0.806, 0.803, 0.699),
        ("cxp", "per_view_threshold"): (0.746, 0.742, 0.641),
    }
    n = 0
    for (train, strategy), sens in lo_sens.items():
        n += 1
        _claim(r, f"LO{n}", "lotter_2024", "chest radiograph", "mimic_cxr", "test", "findings_present", "race",
               f"trained_{train}__{strategy}", "tpr",
               [(g, v, lo_counts[g][0], lo_counts[g][1]) for g, v in zip(("Asian", "Black", "White"), sens)],
               "derived_from_reported", "Supplementary Table 3, Fig. 4 caption, Supplementary Table 4",
               lo_note)
    for (train, strategy), spec in lo_spec.items():
        n += 1
        _claim(r, f"LO{n}", "lotter_2024", "chest radiograph", "mimic_cxr", "test", "findings_present", "race",
               f"trained_{train}__{strategy}", "fpr",
               [(g, 1.0 - v, lo_counts[g][0], lo_counts[g][1]) for g, v in zip(("Asian", "Black", "White"), spec)],
               "derived_from_reported", "Supplementary Table 3, Fig. 4 caption, Supplementary Table 4",
               lo_note + " FPR is one minus the printed specificity.")

    me_note = ("Counts reconstructed as the Table A2 pool share times the stated 10% test "
               "fraction, and positives from the printed unhealthy share; MEDFAIR prints no "
               "per-subgroup test count. The paper does not say which subgroup is the "
               "worst-case, so the value-to-subgroup assignment is arbitrary, and neither "
               "the reported difference nor the reference depends on it. Only two-subgroup "
               "attributes are auditable, since a five-band age attribute is not determined "
               "by its minimum and its difference alone.")
    me = {
        ("chexpert", "race"): (("White", 1254, 11309), ("non-White", 979, 8737), 0.8784, 0.0016, 0.8792),
        ("chexpert", "sex"): (("Male", 1306, 11915), ("Female", 926, 8131), 0.8807, 0.0002, 0.8809),
        ("mimic_cxr", "race"): (("White", 7818, 14647), ("non-White", 7021, 7609), 0.8552, 0.0085, 0.8626),
        ("mimic_cxr", "sex"): (("Male", 7229, 12120), ("Female", 7610, 10133), 0.8562, 0.0141, 0.8645),
    }
    n = 0
    for (ds, attr), ((g1, p1, n1), (g2, p2, n2), lo, diff, overall) in me.items():
        n += 1
        _claim(r, f"ME{n}", "zong_2023_medfair", "chest radiograph", ds, "test", "no_finding", attr, "erm_pareto",
               "auroc", [(g1, lo + diff, p1, n1), (g2, lo, p2, n2)],
               "reconstructed_from_pool_shares", "Tables A2 and A9", me_note, overall=overall)


    da_src = "Supplementary Tables 1 and 2"
    da_note = ("Positives and negatives printed per Fitzpatrick band in Supplementary Table 1. "
               "FST I-II and FST V-VI are matched on diagnosis, age, sex and date, and the paper "
               "states its primary comparison is between them, so the unmatched FST III-IV band "
               "is not part of the difference. FPR is one minus the printed specificity.")
    da_counts = {"ddi": ((49, 159), (48, 159)), "ddi_common": ((42, 148), (16, 140))}
    da = {
        "modelderm": {"ddi": (0.64, 0.55, 0.65, 0.41, 0.12, 0.36, 0.75, 0.89, 0.83),
                      "ddi_common": (0.68, 0.70, 0.74, 0.45, 0.25, 0.47, 0.77, 0.88, 0.83)},
        "deepderm": {"ddi": (0.61, 0.50, 0.56, 0.69, 0.23, 0.53, 0.38, 0.68, 0.53),
                     "ddi_common": (0.64, 0.55, 0.64, 0.71, 0.31, 0.64, 0.39, 0.68, 0.52)},
        "ham10000": {"ddi": (0.72, 0.57, 0.67, 0.02, 0.06, 0.06, 0.99, 0.99, 0.99),
                     "ddi_common": (0.75, 0.62, 0.71, 0.02, 0.06, 0.07, 0.99, 0.99, 0.99)},
        "dermatologist_ensemble": {"ddi": (0.75, 0.62, 0.72, 0.84, 0.40, 0.71, 0.60, 0.79, 0.67),
                                   "ddi_common": (0.80, 0.81, 0.82, 0.93, 0.62, 0.88, 0.61, 0.79, 0.67)},
    }
    n = 0
    for labeler, per_ds in da.items():
        for ds, (a1, a5, aA, s1, s5, sA, p1, p5, pA) in per_ds.items():
            (m1, b1), (m5, b5) = da_counts[ds]
            for metric, v1, v5, ov in (("auroc", a1, a5, aA),
                                       ("tpr", s1, s5, sA),
                                       ("fpr", 1.0 - p1, 1.0 - p5, 1.0 - pA)):
                n += 1
                _claim(r, f"DA{n}", "daneshjou_2022", "dermatology", ds, "test", "malignancy",
                       "skin_tone", labeler, metric,
                       [("FST I-II", v1, m1, b1), ("FST V-VI", v5, m5, b5)],
                       "reported", da_src, da_note, overall=ov)


    va_src = "Supplementary Data Tables 2 to 4 and 5 to 7"
    va_note = ("Recall per tumor class, race-stratified, on the study's own independent test "
               "cohort with no resampling, at the Youden J threshold taken from the matching "
               "validation fold. Counts are slides; for the two MGB cohorts one slide is one case. "
               "Only the UNI encoder with ABMIL and no data-processing intervention is audited, "
               "which is the paper's flagship configuration. The reported AUROC differences are "
               "not auditable, since the article prints them as differences and gives no "
               "per-subgroup AUROC anywhere in its text or supplement.")
    va = {
        ("breast_subtyping", "mgb_breast"): {
            "IDC": ((0.850, 759), (0.907, 69), (0.895, 109), 0.862),
            "ILC": ((0.955, 145), (0.877, 73), (0.780, 55), 0.898)},
        ("lung_subtyping", "mgb_lung"): {
            "LUAD": ((0.969, 1359), (0.982, 115), (0.903, 100), 0.966),
            "LUSC": ((0.890, 271), (0.817, 26), (0.855, 28), 0.882)},
        ("idh1_mutation", "tcga_gbmlgg"): {
            "IDH1 wild-type": ((0.834, 608), (0.944, 9), (0.865, 57), 0.835),
            "IDH1 mutant": ((0.829, 375), (0.898, 21), (0.603, 15), 0.821)},
    }
    n = 0
    for (task, ds), per_class in va.items():
        for cls, ((vw, nw), (va_, na), (vb, nb), ov) in per_class.items():
            n += 1
            _claim(r, f"VA{n}", "vaidya_2024", "computational pathology", ds, "test", task,
                   "race", f"UNI ({cls})", "tpr",
                   [("White", vw, nw, 0), ("Asian", va_, na, 0), ("Black", vb, nb, 0)],
                   "reported", va_src, va_note, overall=ov)


    bu_src = "Tables 1 and 3"
    bu_note = ("Denominators printed directly in Table 1: 100 referable and 100 healthy per "
               "skin group in the test set. FPR is one minus the printed specificity. The "
               "overall value is the pooled rate over the two groups, which is exact here "
               "because the two denominators are equal.")
    bu_counts = {"Lighter skin": (100, 100), "Darker skin": (100, 100)}
    bu = {
        "baseline": ((0.850, 0.350), (0.610, 0.860)),
        "debiased_retina_appearance": ((0.740, 0.560), (0.830, 0.860)),
        "debiased_dr_status": ((0.780, 0.580), (0.660, 0.850)),
    }
    n = 0
    for cond, ((sl, sd), (pl, pd_)) in bu.items():
        n += 1
        _claim(r, f"BU{n}", "burlina_2021", "funduscopy", "burlina_dr", "test",
               "referable_dr", "skin_tone", cond, "tpr",
               [(g, v, bu_counts[g][0], bu_counts[g][1])
                for g, v in (("Lighter skin", sl), ("Darker skin", sd))],
               "reported", bu_src, bu_note, overall=(sl + sd) / 2.0)
        n += 1
        _claim(r, f"BU{n}", "burlina_2021", "funduscopy", "burlina_dr", "test",
               "referable_dr", "skin_tone", cond, "fpr",
               [(g, 1.0 - v, bu_counts[g][0], bu_counts[g][1])
                for g, v in (("Lighter skin", pl), ("Darker skin", pd_))],
               "reported", bu_src, bu_note, overall=(2.0 - pl - pd_) / 2.0)

    ya_src = "Tables 3 and 4"
    ya_note = ("Positives printed in Table 4 as the examinations followed by a future cancer; "
               "negatives are that count subtracted from the printed examination count. FPR is "
               "one minus the printed specificity. The overall value is the study's own Emory "
               "row in Table 3, which covers the whole cohort and not only the two race groups.")
    ya_counts = {"African American": (301, 4311 - 301), "White": (306, 3728 - 306)}
    ya = {
        "mirai_at_tc_specificity": ((0.339, 0.400), (0.837, 0.855), 0.367, 0.849),
        "mirai_at_tc_sensitivity": ((0.229, 0.206), (0.907, 0.919), 0.220, 0.915),
    }
    n = 0
    for cond, ((ta, tw), (sa, sw), ov_t, ov_s) in ya.items():
        n += 1
        _claim(r, f"YA{n}", "yala_2021", "mammography", "emory", "test", "cancer_within_5y",
               "race", cond, "tpr",
               [(g, v, ya_counts[g][0], ya_counts[g][1])
                for g, v in (("African American", ta), ("White", tw))],
               "reported", ya_src, ya_note, overall=ov_t)
        n += 1
        _claim(r, f"YA{n}", "yala_2021", "mammography", "emory", "test", "cancer_within_5y",
               "race", cond, "fpr",
               [(g, 1.0 - v, ya_counts[g][0], ya_counts[g][1])
                for g, v in (("African American", sa), ("White", sw))],
               "reported", ya_src, ya_note, overall=1.0 - ov_s)

    ta_src = "Table 1 and Table 1 (continued), PDAC block"
    ta_note = ("Tumour and control counts per subgroup printed directly in Table 1, so both "
               "denominators are given. Each row is one privacy budget, meaning one trained "
               "model, and the non-private model is the row at epsilon = infinity. The "
               "UKA-CXR chest radiograph block of the same table is not auditable, since its "
               "per-subgroup AUROC is averaged over eight diagnoses.")
    ta_counts = {"Male": (95, 102), "Female": (77, 50),
                 "Youngest 25%": (23, 63), "Second 25%": (48, 37),
                 "Third 25%": (54, 25), "Oldest 25%": (48, 27)}
    ta = {
        "0.29": (0.8684, 0.8811, 0.8547, 0.8792, 0.8587, 0.8444, 0.8915),
        "0.54": (0.9260, 0.9362, 0.9100, 0.9377, 0.9197, 0.9005, 0.9563),
        "1.06": (0.9558, 0.9670, 0.9352, 0.9657, 0.9484, 0.9383, 0.9843),
        "2.04": (0.9749, 0.9850, 0.9536, 0.9798, 0.9690, 0.9706, 0.9936),
        "4.71": (0.9831, 0.9919, 0.9638, 0.9848, 0.9784, 0.9830, 0.9997),
        "5.0": (0.9833, 0.9920, 0.9641, 0.9848, 0.9786, 0.9837, 1.0000),
        "6.0": (0.9839, 0.9922, 0.9655, 0.9857, 0.9784, 0.9835, 1.0000),
        "7.0": (0.9841, 0.9922, 0.9660, 0.9862, 0.9788, 0.9825, 1.0000),
        "8.0": (0.9928, 0.9977, 0.9813, 0.9959, 0.9923, 0.9837, 1.0000),
        "inf": (0.9970, 0.9997, 0.9901, 0.9998, 0.9994, 0.9847, 1.0000),
    }
    n = 0
    for eps, (tot, male, female, y1, y2, y3, y4) in ta.items():
        cond = "non_private" if eps == "inf" else f"dp_eps_{eps}"
        n += 1
        _claim(r, f"TA{n}", "tayebi_arasteh_2024", "abdominal CT", "pdac", "test", "pdac",
               "sex", cond, "auroc",
               [(g, v, ta_counts[g][0], ta_counts[g][1])
                for g, v in (("Male", male), ("Female", female))],
               "reported", ta_src, ta_note, overall=tot)
        n += 1
        _claim(r, f"TA{n}", "tayebi_arasteh_2024", "abdominal CT", "pdac", "test", "pdac",
               "age", cond, "auroc",
               [(g, v, ta_counts[g][0], ta_counts[g][1])
                for g, v in (("Youngest 25%", y1), ("Second 25%", y2),
                             ("Third 25%", y3), ("Oldest 25%", y4))],
               "reported", ta_src, ta_note, overall=tot)

    _not_auditable(r, "brown_2023",
                   "Reports no per-subgroup performance and no per-subgroup count. Its source "
                   "data give only an aggregate condition AUC, an equalised-odds value and a "
                   "race AUC per replicate, and every fairness result is plotted.")
    _not_auditable(r, "groh_2021",
                   "Reports accuracy by Fitzpatrick skin type and no AUROC, true positive rate "
                   "or false positive rate anywhere.")
    _not_auditable(r, "kinyanjui_2020",
                   "Reports accuracy across skin tones and no AUROC, true positive rate or "
                   "false positive rate anywhere.")
    _not_auditable(r, "howard_2021",
                   "Its AUROCs measure site and feature prediction, so the article reports no "
                   "performance difference of a diagnostic model between demographic subgroups.")
    _not_auditable(r, "huang_2025",
                   "Group-wise AUROC and true positive rate disparity appear only inside "
                   "figures, and race appears only as a cohort-level percentage, so no "
                   "per-subgroup denominator exists without inventing one.")
    _not_auditable(r, "zhang_2022",
                   "Every result is plotted, with per-group AUROC only in figure axis text, and "
                   "no per-subgroup count is printed in either the conference or the arXiv version.")
    _not_auditable(r, "luo_2024_fairclip",
                   "Prints group-wise AUC and the demographic composition but never the glaucoma "
                   "label distribution, so the positive and negative counts an AUROC reference "
                   "needs cannot be obtained from the article.")
    _not_auditable(r, "tayebi_arasteh_2023_domain_transfer",
                   "Reports the average of AUC differences from a single-institutional baseline, "
                   "averaged over five one-versus-all classifications, so no per-subgroup AUROC "
                   "value is printed anywhere in the article or its appendix.")
    _not_auditable(r, "kaess_2025",
                   "Reports accuracy, F1 and the Matthews correlation coefficient for "
                   "performance, and statistical parity difference and disparate impact for "
                   "fairness, none of which is a statistic the reference is derived for.")

    _not_auditable(r, "seyyed_kalantari_2021_chexclusion",
                   "Reports the difference between a subgroup's true-positive rate and the "
                   "median over subgroups, on a sub-sampled test set of 22,274 images whose "
                   "per-subgroup positive counts are printed nowhere, and releases no code.")
    _not_auditable(r, "larrazabal_2020",
                   "Reports every result as a box plot. No AUROC value and no between-subgroup "
                   "difference appears numerically anywhere in the article, which has no tables "
                   "and no supplementary material, and the released repository provides the data "
                   "splits but reports its additional results as further box plots.")
    _not_auditable(r, "knolle_2026",
                   "Reports no per-subgroup performance value. Every subgroup comparison is a "
                   "count of records in a membership-inference risk tail against the count "
                   "expected from the subgroup's share, tested by chi-squared.")
    _not_auditable(r, "yang_2024",
                   "Reports no per-subgroup false-positive or false-negative rate, and no "
                   "per-subgroup positive or negative count for any split. Every fairness "
                   "result is a difference between two named subgroups, plotted without values.")

    return pd.DataFrame(r, columns=COLUMNS)


def main_build_published_claims(global_config_path: str) -> str:
    cfg = read_config(global_config_path)
    out = cfg["BiasOrigin"]["published"]["claims_csv"]
    df = build_claims()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)
    n_aud = df[df["reported_value"].notna()]["claim_id"].nunique()
    print(f"[claims] {df['claim_id'].nunique()} claims from {df['paper'].nunique()} papers "
          f"({n_aud} auditable) -> {out}", flush=True)
    return out
