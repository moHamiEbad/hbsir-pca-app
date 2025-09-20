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

def group_unlabeled_to_parent_other(df: pd.DataFrame, target_level: int) -> pd.DataFrame:
    tcol = f"label_{target_level}"
    if tcol not in df.columns:
        raise ValueError(f"Expected '{tcol}' column on the frame.")

    x = df.copy()
    x[tcol] = x[tcol].astype(object)
    missing = x[tcol].isna() | (x[tcol].astype(str).str.strip() == "")

    if not missing.any():
        return x

    if target_level == 1:
        x.loc[missing, tcol] = UNCLASS_OTHER
        return x

    parent_cols = [f"label_{j}" for j in range(target_level - 1, 0, -1) if f"label_{j}" in x.columns]
    if not parent_cols:
        x.loc[missing, tcol] = UNCLASS_OTHER
        return x

    parent_series = x[parent_cols].bfill(axis=1).iloc[:, 0]
    have_parent = parent_series.notna() & (parent_series.astype(str).str.strip() != "")
    replace_vals = pd.Series(UNCLASS_OTHER, index=x.index, dtype=object)
    replace_vals.loc[have_parent] = parent_series.loc[have_parent].astype(str) + PARENT_OTHER_SUFFIX
    x.loc[missing, tcol] = replace_vals.loc[missing].values
    return x

@st.cache_data(show_spinner=True)
def build_household_matrix_by_level(df: pd.DataFrame, years: list, level: int, exp_measure: str):
    df = df[df["Year"].isin(years)].copy()
    df = add_class_labels(df, level)
    label_col = f"label_{level}"
    df = group_unlabeled_to_parent_other(df, target_level=level)

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
    dfc = add_multi_class_labels(df_year, level)
    path_cols = [f"label_{i}" for i in range(1, level + 1) if f"label_{i}" in dfc.columns]
    if f"label_{level}" in path_cols:
        uniq_paths = dfc[path_cols].dropna(subset=[f"label_{level}"]).drop_duplicates()
        uniq_paths = uniq_paths.rename(columns={f"label_{level}": "feature"})
    else:
        uniq_paths = pd.DataFrame(columns=["feature"])

    meta = pd.DataFrame({"feature": [str(f) for f in feature_names]}).merge(uniq_paths, on="feature", how="left")
    for i in range(1, level):
        col = f"label_{i}"
        if col in meta.columns:
            meta[col] = meta[col].fillna("(unknown)")
    return meta
