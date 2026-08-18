"""
data_loader/build_cxr_paired_reports.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import reroot_mimic_report_path

_REPORT_COLS = ["case_id", "dataset", "image_key", "split",
                "report_path", "report_text", "report_source"]
_PLUS_SECTIONS = ("section_findings", "section_impression")
_PLUS_FALLBACK = "report"

_AGE_PATTERNS = [
    re.compile(r"\b\d{1,3}\s*[- ]?\s*(?:year|yr|yrs)s?[- ]?old\b", re.I),
    re.compile(r"\b\d{1,3}\s*[- ]?\s*y/?o\b", re.I),
    re.compile(r"\bage[d]?\s*\d{1,3}\b", re.I),
]


def _collect_mimic(pool: pd.DataFrame, image_root: str) -> pd.DataFrame:
    rows = pool[pool["dataset"] == "mimic"].copy()
    if rows.empty:
        return pd.DataFrame(columns=_REPORT_COLS)
    has = (rows["report_rel_path"].notna() &
           (rows["report_rel_path"].astype(str).str.strip().str.lower() != "nan") &
           (rows["report_rel_path"].astype(str).str.strip() != ""))
    rows = rows[has].copy()
    if rows.empty:
        return pd.DataFrame(columns=_REPORT_COLS)
    abs_paths = rows["report_rel_path"].apply(lambda r: os.path.join(image_root, str(r)))
    exists = abs_paths.apply(os.path.exists)
    n_missing = int((~exists).sum())
    if n_missing:
        print(f"[paired_reports] MIMIC: {n_missing} report files not found; excluded.")
    rows, abs_paths = rows[exists], abs_paths[exists]
    out = pd.DataFrame({
        "case_id": rows["case_id"].values, "dataset": "mimic",
        "image_key": rows["image_key"].values, "split": rows["split"].values,
        "report_path": abs_paths.values, "report_text": np.nan,
        "report_source": "mimic_report_file",
    })
    print(f"[paired_reports] MIMIC: {len(out)} rows.")
    return out.reset_index(drop=True)


def _clean_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip()
    return s.mask(s.str.lower().isin(("nan", "none")), "")


def _collect_chexpert(pool: pd.DataFrame, scfg: dict) -> pd.DataFrame:
    plus_csv = scfg.get("chexpert_plus_csv")
    rows = pool[pool["dataset"] == "chexpert"].copy()
    if rows.empty or not plus_csv or not os.path.exists(plus_csv):
        if rows.empty:
            return pd.DataFrame(columns=_REPORT_COLS)
        print("[paired_reports] CheXpert Plus CSV missing; no CheXpert reports.")
        return pd.DataFrame(columns=_REPORT_COLS)
    join_col = scfg.get("report_join_col", "jpg_rel_path")
    usecols = [join_col, *(_PLUS_SECTIONS), _PLUS_FALLBACK]
    plus = read_csv_defensively(plus_csv, usecols=lambda c: c in set(usecols))
    plus = plus.drop_duplicates(subset=[join_col])
    merged = rows[["case_id", "image_key", "split"]].merge(
        plus.rename(columns={join_col: "image_key"}), on="image_key", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=_REPORT_COLS)

    idx = merged.index
    empty = pd.Series("", index=idx)
    findings = _clean_series(merged[_PLUS_SECTIONS[0]]) if _PLUS_SECTIONS[0] in merged.columns else empty
    impression = _clean_series(merged[_PLUS_SECTIONS[1]]) if _PLUS_SECTIONS[1] in merged.columns else empty
    has_f, has_i = findings.ne(""), impression.ne("")

    combined = pd.Series(np.nan, index=idx, dtype=object)
    combined[has_f & has_i] = findings[has_f & has_i] + "\n" + impression[has_f & has_i]
    combined[has_f & ~has_i] = findings[has_f & ~has_i]
    combined[~has_f & has_i] = impression[~has_f & has_i]

    source = pd.Series(np.nan, index=idx, dtype=object)
    source[combined.notna()] = "+".join(_PLUS_SECTIONS)

    fallback = _clean_series(merged[_PLUS_FALLBACK]) if _PLUS_FALLBACK in merged.columns else empty
    need_fallback = combined.isna()
    fb_has = need_fallback & fallback.ne("")
    text = combined.copy()
    text[fb_has] = fallback[fb_has]
    source[fb_has] = _PLUS_FALLBACK
    source[need_fallback & ~fb_has] = "none"

    merged["report_text"], merged["report_source"] = text, source
    merged = merged[merged["report_text"].notna()].copy()
    out = pd.DataFrame({
        "case_id": merged["case_id"].values, "dataset": "chexpert",
        "image_key": merged["image_key"].values, "split": merged["split"].values,
        "report_path": np.nan, "report_text": merged["report_text"].values,
        "report_source": merged["report_source"].values,
    })
    print(f"[paired_reports] CheXpert: {len(out)} rows.")
    return out.reset_index(drop=True)


def _load_scrub_terms(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        print(f"[paired_reports] scrub-terms file not found ({path}); using built-in minimal list.")
        return ["male", "female", "man", "woman", "gentleman", "lady",
                "white", "black", "caucasian", "african american", "asian",
                "hispanic", "latino", "latina"]
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def _build_scrubber(terms: List[str]):
    term_res = [re.compile(rf"\b{re.escape(t)}\b", re.I) for t in terms]

    def scrub(text: Optional[str]) -> Optional[str]:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return text
        s = str(text)
        for rgx in _AGE_PATTERNS:
            s = rgx.sub(" ", s)
        for rgx in term_res:
            s = rgx.sub(" ", s)
        return re.sub(r"\s+", " ", s).strip()
    return scrub


def _materialize_text(row: pd.Series, mimic_root: str) -> Optional[str]:
    if isinstance(row.get("report_text"), str) and row["report_text"].strip():
        return row["report_text"]
    p = reroot_mimic_report_path(row.get("report_path"), mimic_root)
    if p and os.path.exists(p):
        try:
            with open(p, "r", errors="ignore") as f:
                return f.read()
        except OSError:
            return None
    return None


def _source_counts(df: pd.DataFrame) -> pd.Series:
    return df["dataset"].astype(str).value_counts()


def _compare_sources(src_in: pd.Series, src_out: pd.Series):
    lost, thin = [], []
    for src, n_in in src_in.items():
        n_out = int(src_out.get(src, 0))
        if n_out == 0:
            lost.append(f"{src}: 0 of {n_in}")
        elif n_out < n_in:
            thin.append(f"{src}: {n_out} of {n_in}")
    return lost, thin


def _assert_no_source_lost(paired: pd.DataFrame, out: pd.DataFrame,
                           arm: str, mimic_root: str) -> None:
    lost, thin = _compare_sources(_source_counts(paired), _source_counts(out))
    if thin:
        print(f"[paired_reports] WARNING {arm}: partial materialization - "
              + "; ".join(thin))
    if lost:
        raise RuntimeError(
            f"[paired_reports] {arm}: no text could be materialized for "
            + "; ".join(lost)
            + f". The report files are read through this machine's mimic "
            f"image_root '{mimic_root}'; check that the MIMIC report tree is "
            f"there, and that the paired manifest's report_path column reroots "
            f"onto it.")


_REBUILD_HINT = {
    "scrubbed": "main_build_cxr_scrubbed_reports",
    "amplified": "main_build_cxr_amplified_reports",
}


def assert_derived_manifest_complete(derived_csv: str, paired_csv: str,
                                     arm: str) -> None:
    src_in = _source_counts(read_csv_defensively(
        paired_csv, usecols=lambda c: c == "dataset"))
    src_out = _source_counts(read_csv_defensively(
        derived_csv, usecols=lambda c: c == "dataset"))
    lost, thin = _compare_sources(src_in, src_out)
    if thin:
        print(f"[paired_reports] WARNING {arm} manifest is short of its source: "
              + "; ".join(thin))
    if lost:
        raise RuntimeError(
            f"[paired_reports] the {arm} reports manifest {derived_csv} is "
            f"missing whole sources present in {os.path.basename(paired_csv)} ("
            + "; ".join(lost)
            + f"). It was built where those reports could not be read, so it is "
            f"truncated, not merely small. Rebuild it on a machine that holds "
            f"the MIMIC report tree by running "
            f"{_REBUILD_HINT.get(arm, 'the matching report builder')}, then copy "
            f"the manifest across; its text is inline, so it is valid on every "
            f"machine once built.")


def main_build_cxr_paired_reports(global_config_path: str) -> str:
    cfg     = read_config(global_config_path)["BiasOrigin"]
    cxr     = cfg["cxr"]
    pool_csv = cxr["pool_manifest_csv"]
    out_csv  = cxr["paired_reports_csv"]
    if not os.path.exists(pool_csv):
        raise FileNotFoundError(f"[paired_reports] pool manifest not found: {pool_csv}. Run build_cxr_pool first.")
    pool = read_csv_defensively(pool_csv, usecols=lambda c: c in {
        "case_id", "dataset", "image_key", "split", "report_rel_path"})
    parts = []
    srcs = cxr.get("reports", {}).get("sources", ["mimic", "chexpert_plus"])
    if "mimic" in srcs:
        d = _collect_mimic(pool, cxr["sites"]["mimic"]["image_root"])
        if not d.empty:
            parts.append(d)
    if "chexpert_plus" in srcs:
        d = _collect_chexpert(pool, cxr["sites"]["chexpert"])
        if not d.empty:
            parts.append(d)
    if not parts:
        raise RuntimeError("[paired_reports] no report rows produced.")
    paired = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    paired.to_csv(out_csv, index=False)
    print(f"[paired_reports] -> {out_csv}  ({len(paired)} rows; "
          f"text inline={paired['report_text'].notna().sum()}, "
          f"path={paired['report_path'].notna().sum()})")
    return out_csv


def main_build_cxr_scrubbed_reports(global_config_path: str) -> str:
    cfg      = read_config(global_config_path)["BiasOrigin"]
    cxr      = cfg["cxr"]
    paired_csv = cxr["paired_reports_csv"]
    out_csv  = cxr["reports"]["scrubbed_reports_csv"]
    if not os.path.exists(paired_csv):
        main_build_cxr_paired_reports(global_config_path)
    mimic_root = cxr["sites"]["mimic"]["image_root"]
    paired = read_csv_defensively(paired_csv)
    scrub = _build_scrubber(_load_scrub_terms(cxr["reports"].get("scrub_terms_file")))

    raw_text = paired.apply(lambda r: _materialize_text(r, mimic_root), axis=1)
    scrubbed = raw_text.apply(scrub)
    out = pd.DataFrame({
        "case_id": paired["case_id"].values, "dataset": paired["dataset"].values,
        "image_key": paired["image_key"].values, "split": paired["split"].values,
        "report_path": np.nan, "report_text": scrubbed.values,
        "report_source": "scrubbed_inline",
    })
    out = out[out["report_text"].notna() & (out["report_text"].astype(str).str.strip() != "")].copy()
    _assert_no_source_lost(paired, out, "scrubbed", mimic_root)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[paired_reports] scrubbed -> {out_csv}  ({len(out)} rows materialized + scrubbed; "
          f"by source: {dict(out['dataset'].astype(str).value_counts())})")
    return out_csv


_SEX_PHRASE = {"F": "female", "M": "male"}
_AGE_PHRASE = {"0_40": "aged under 40 years", "40_60": "aged 40 to 60 years",
               "60_80": "aged 60 to 80 years", "80_plus": "aged over 80 years"}


def _demographic_sentence(row: pd.Series, attrs: List[str], template: str) -> str:
    parts = []
    for a in attrs:
        v = row.get(a)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        v = str(v).strip()
        if not v or v.lower() == "nan":
            continue
        if a == "race_grp":
            parts.append(f"{v.replace('_or_', ' or ').replace('_', ' ')} race")
        elif a == "sex_grp":
            parts.append(_SEX_PHRASE.get(v, v))
        elif a == "age_grp":
            parts.append(_AGE_PHRASE.get(v, f"age group {v}"))
        else:
            parts.append(f"{a.replace('_grp', '')} {v}")
    if not parts:
        return ""
    return template.format(fields=", ".join(parts))


def main_build_cxr_amplified_reports(global_config_path: str) -> str:
    cfg      = read_config(global_config_path)["BiasOrigin"]
    cxr      = cfg["cxr"]
    rcfg     = cxr["reports"]
    paired_csv = cxr["paired_reports_csv"]
    pool_csv = cxr["pool_manifest_csv"]
    out_csv  = rcfg["amplified_reports_csv"]
    if not os.path.exists(paired_csv):
        main_build_cxr_paired_reports(global_config_path)
    if not os.path.exists(pool_csv):
        raise FileNotFoundError(
            f"[paired_reports] pool manifest not found: {pool_csv}. The amplified "
            f"arm reads the harmonized demographic columns from it.")
    attrs = list(rcfg.get("amplify_attributes", ["race_grp", "sex_grp", "age_grp"]))
    template = rcfg.get("amplify_template", "Patient demographics: {fields}.")

    paired = read_csv_defensively(paired_csv)
    pool = read_csv_defensively(pool_csv, usecols=lambda c: c in ({"case_id"} | set(attrs)))
    missing = [a for a in attrs if a not in pool.columns]
    if missing:
        raise KeyError(f"[paired_reports] pool manifest has no {missing} column(s); "
                       f"cannot build the amplified arm.")
    mimic_root = cxr["sites"]["mimic"]["image_root"]
    merged = paired.merge(pool.drop_duplicates(subset=["case_id"]), on="case_id", how="left")

    raw_text = merged.apply(lambda r: _materialize_text(r, mimic_root), axis=1)
    sentences = merged.apply(lambda r: _demographic_sentence(r, attrs, template), axis=1)
    amplified = [
        (re.sub(r"\s+", " ", f"{s} {t}").strip() if isinstance(t, str) and s
         else (t if isinstance(t, str) else np.nan))
        for s, t in zip(sentences.values, raw_text.values)
    ]
    out = pd.DataFrame({
        "case_id": merged["case_id"].values, "dataset": merged["dataset"].values,
        "image_key": merged["image_key"].values, "split": merged["split"].values,
        "report_path": np.nan, "report_text": amplified,
        "report_source": "amplified_inline",
        "_tagged": (sentences.astype(str).str.strip() != "").values,
    })
    out = out[out["report_text"].notna() & (out["report_text"].astype(str).str.strip() != "")].copy()
    n_tagged = int(out["_tagged"].sum())
    out = out.drop(columns=["_tagged"])
    _assert_no_source_lost(paired, out, "amplified", mimic_root)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[paired_reports] amplified -> {out_csv}  ({len(out)} rows materialized; "
          f"{n_tagged} carry a demographic sentence; "
          f"by source: {dict(out['dataset'].astype(str).value_counts())})")
    return out_csv
