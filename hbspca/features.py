from __future__ import annotations
from typing import List
import pandas as pd
import streamlit as st
import hbsir

PARENT_OTHER_SUFFIX = " — Other"
UNCLASS_OTHER       = "Other — Unclassified"

@st.cache_data(show_spinner=False)
def add_multi_class_labels(df: pd.DataFrame, max_level: int) -> pd.DataFrame:
    out = df.copy()
    for lvl in range(1, max_level + 1):
        try:
            out = hbsir.add_classification(out, target="Commodity_Code",
                                           levels=[lvl], aspects=["label"])
        except Exception:
            continue
    return out

@st.cache_data(show_spinner=True)
def add_class_labels(df: pd.DataFrame, level: int) -> pd.DataFrame:
    levels = list(range(1, level + 1))
    try:
        out = hbsir.add_classification(df, target="Commodity_Code",
                                       levels=levels, aspects=["label"])
    except Exception as e:
        raise RuntimeError(f"Classification labels up to level {level} are not available.") from e
    return out


def _norm_label(x: object) -> str:
    s = str(x).strip()
    if s.lower() in {"", "none", "nan", "null"}:
        return UNCLASS_OTHER
    return s

def _is_other_like(s: str) -> bool:
    s_low = s.lower()
    return (
        s == UNCLASS_OTHER
        or " — other" in s_low             # already an “— Other”
        or s_low.startswith("other_")      # taxonomy leaf like other_breads
        or s_low.startswith("other ")      # just in case
    )

def _collapse_other_chain(s: str) -> str:
    # collapse “ — Other — Other — Other” -> “ — Other”
    while " — Other — Other" in s:
        s = s.replace(" — Other — Other", " — Other")
    return s

def complete_path_to_level(df: pd.DataFrame, level: int) -> pd.DataFrame:
    """
    Ensure label_1..label_level exist for every row.
    Rules:
    - normalize None/NaN/"" -> UNCLASS_OTHER
    - never produce “ — Other — Other…”
    - if ancestor is already other-like (UNCLASS_OTHER or real other_* leaf),
      then just repeat that ancestor for deeper levels (no extra “ — Other”)
    - otherwise: add ONE local “ — Other” at the first missing level, and
      copy that value down to the remaining deeper levels.
    """
    x = df.copy()

    # Ensure columns exist and normalize
    for j in range(1, level + 1):
        col = f"label_{j}"
        if col not in x.columns:
            x[col] = UNCLASS_OTHER
        x[col] = x[col].map(_norm_label).astype(object)

    # Walk levels left→right and fill missing segments
    for j in range(2, level + 1):
        parent = f"label_{j-1}"
        col    = f"label_{j}"
        # row-wise: decide fill per rules
        missing = (x[col].astype(str).str.strip() == "") | x[col].isna() | (x[col] == UNCLASS_OTHER)

        if missing.any():
            parent_vals = x.loc[missing, parent].astype(str).map(_norm_label)
            # case A: parent already other-like  -> repeat parent
            mask_other_parent = parent_vals.apply(_is_other_like)
            if mask_other_parent.any():
                x.loc[missing[missing].index[mask_other_parent], col] = parent_vals[mask_other_parent]
            # case B: parent not other-like -> exactly one local “ — Other”
            mask_not_other = ~mask_other_parent
            if mask_not_other.any():
                x.loc[missing[missing].index[mask_not_other], col] = (
                    parent_vals[mask_not_other] + PARENT_OTHER_SUFFIX
                )

        # collapse accidental chains produced upstream
        x[col] = x[col].astype(str).map(_collapse_other_chain)

    # Copy forward: if any deeper level still slipped to empty/UNCLASS_OTHER while parent is informative,
    # repeat the last non-empty value (but keep collapsed)
    labels = [f"label_{j}" for j in range(1, level + 1)]
    x[labels] = x[labels].applymap(_norm_label)
    for j in range(2, level + 1):
        prev, col = f"label_{j-1}", f"label_{j}"
        need_copy = (x[col] == UNCLASS_OTHER) & (x[prev] != UNCLASS_OTHER)
        if need_copy.any():
            x.loc[need_copy, col] = x.loc[need_copy, prev].astype(str).map(_collapse_other_chain)

    # Final cleanup
    for j in range(1, level + 1):
        col = f"label_{j}"
        x[col] = x[col].astype(str).map(_collapse_other_chain)

    return x


@st.cache_data(show_spinner=True)
def build_household_matrix_by_level(df: pd.DataFrame, years: list, level: int, exp_measure: str):
    df = df[df["Year"].isin(years)].copy()
    df = add_class_labels(df, level=level)
    df = complete_path_to_level(df, level=level)  # <-- important
    label_col = f"label_{level}"

    # 3) aggregate and pivot
    agg = (
        df.groupby(["Year", "ID", label_col], observed=False, dropna=False)[exp_measure]
          .sum(min_count=1).reset_index()
    )
    wide = agg.pivot_table(index=["Year", "ID"], columns=label_col, values=exp_measure,
                           aggfunc="sum", fill_value=0, observed=False)
    wide.columns = wide.columns.astype(str)
    wide.sort_index(axis=1, inplace=True)

    meta_cols = [c for c in ["Province_code","Province_farsi_name",
                             "County_code","County_farsi_name",
                             "Urban_Rural_name","Urban_Rural"] if c in df.columns]
    meta = (df.drop_duplicates(subset=["Year","ID"]).set_index(["Year","ID"])[meta_cols]
              .reindex(wide.index))
    return wide, meta, label_col
    
def feature_meta_for_year(df_year: pd.DataFrame, level: int, feature_names: List[str]) -> pd.DataFrame:
    """
    Create a mapping table with columns: feature (=label_level), label_1..label_level.
    Uses completed paths so every feature has full lineage.
    """
    dfc = add_class_labels(df_year, level=level)
    dfc = complete_path_to_level(dfc, level=level)  # <-- important
    path_cols = [f"label_{i}" for i in range(1, level + 1)]
    uniq = dfc[path_cols].drop_duplicates().rename(columns={f"label_{level}": "feature"})

    meta = pd.DataFrame({"feature": [str(f) for f in feature_names]}).merge(uniq, on="feature", how="left")

    # Guarantee presence and string dtype
    for i in range(1, level):
        col = f"label_{i}"
        if col not in meta.columns:
            meta[col] = "(unknown)"
        meta[col] = meta[col].fillna("(unknown)").astype(str)
    return meta
