# hbsir.setup(years=years,
#     method="create_from_raw",
#     download_source="mirror",
#     replace=False)

# data_io.py
import os
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import hbsir


# -----------------------------
# Setup & Loading
# -----------------------------
@st.cache_data(show_spinner=False)
def setup_hbsir(years: List[int],
                method: str = "download_cleaned",
                download_source: str = "mirror") -> bool:
    # hbsir.setup(
    #     years=years,
    #     method=method,
    #     download_source=download_source,
    #     replace=False,
    # )
    hbsir.setup(years=years,
    method="create_from_raw",
    download_source="mirror",
    replace=False)
    return True


@st.cache_data(show_spinner=True)
def load_households(years: List[int]) -> pd.DataFrame:
    """Household_information, normalized; add only HBSIR-provided attributes."""
    hh = hbsir.load_table("household_information", years=years, form="normalized")
    hh = hbsir.add_attribute(hh, "Urban_Rural")
    hh = hbsir.add_attribute(hh, "Province", aspects=["code", "farsi_name"])
    hh = hbsir.add_attribute(hh, "County",   aspects=["code", "farsi_name"])
    return hh


@st.cache_data(show_spinner=True)
def load_expenditures(years: List[int]) -> pd.DataFrame:
    """Expenditures (normalized). If aggregate 'Expenditures' table is missing, concat category tables."""
    try:
        exp = hbsir.load_table("Expenditures", years=years, form="normalized")
    except Exception:
        cats = [
            "food","tobacco","cloth","home","furniture","medical",
            "transportation","communication","entertainment","education",
            "hotel","miscellaneous","durable","investment",
        ]
        frames = []
        for c in cats:
            try:
                frames.append(hbsir.load_table(c, years=years, form="normalized").assign(Table_Name=c))
            except Exception:
                pass
        if not frames:
            raise RuntimeError("No expenditure tables available for the selected years.")
        exp = pd.concat(frames, ignore_index=True)

    # Ensure a Weight column exists on expenditure rows
    if "Weight" not in exp.columns:
        try:
            exp = hbsir.add_weight(exp)
        except Exception:
            pass
    return exp


@st.cache_data(show_spinner=True)
def merge_exp_hh(exp: pd.DataFrame, hh: pd.DataFrame) -> pd.DataFrame:
    """Left merge; keep all columns; do not rename or drop."""
    return exp.merge(hh, on=["Year", "ID"], how="left", validate="m:1")


# -----------------------------
# Filters (Area & Settlement)
# -----------------------------
def urban_rural_col(df: pd.DataFrame) -> Optional[str]:
    if "Urban_Rural_name" in df.columns: return "Urban_Rural_name"
    if "Urban_Rural" in df.columns:      return "Urban_Rural"
    return None


@st.cache_data(show_spinner=False)
def filter_area_settlement(df: pd.DataFrame,
                           years: List[int],
                           level_choice: str,
                           provinces: List[str],
                           counties: List[str],
                           settlement: str,
                           province_col: str = "Province_farsi_name",
                           county_col: str   = "County_farsi_name") -> pd.DataFrame:
    """Apply only the filters allowed before PCA (years, area, settlement)."""
    out = df[df["Year"].isin(years)].copy()

    urc = urban_rural_col(out)
    if settlement != "Both" and urc in out.columns:
        out = out[out[urc].str.lower() == settlement.lower()]

    if level_choice == "Province" and provinces:
        out = out[out[province_col].isin(provinces)]
    if level_choice in ("County", "Province") and counties:
        # If province-level chosen but counties provided, treat as sub-filter
        out = out[out[county_col].isin(counties)]

    return out


# -----------------------------
# Classification labels (1..k)
# -----------------------------
@st.cache_data(show_spinner=False)
def add_multi_class_labels(df: pd.DataFrame, max_level: int) -> pd.DataFrame:
    """
    Attach labels sequentially from level=1..max_level.
    Skip silently if a level is unavailable in local metadata.
    """
    out = df.copy()
    for lvl in range(1, max_level + 1):
        try:
            out = hbsir.add_classification(out, target="Commodity_Code",
                                           levels=[lvl], aspects=["label"])
        except Exception:
            continue
    return out


# -----------------------------
# Helpers
# -----------------------------
PARENT_OTHER_SUFFIX = " — Other"
UNCLASS_OTHER       = "Other — Unclassified"


@st.cache_data(show_spinner=True)
def add_class_labels(df: pd.DataFrame, level: int) -> pd.DataFrame:
    """
    Attach label_1 ... label_{level} for Commodity_Code using HBSIR metadata.
    Only 'label' aspect is requested.
    """
    # Ask for all levels up to 'level' in one call (faster than many small calls)
    levels = list(range(1, level + 1))
    try:
        out = hbsir.add_classification(
            df, target="Commodity_Code",
            levels=levels,
            aspects=["label"],
        )
    except Exception as e:
        raise RuntimeError(
            f"Classification labels up to level {level} are not available in your installed metadata."
        ) from e
    return out


def group_unlabeled_to_parent_other(df: pd.DataFrame, target_level: int) -> pd.DataFrame:
    """
    For rows missing label_{target_level}, fill with:
      - '{nearest_parent_label} — Other' if some parent (label_{<L}) exists
      - 'Other — Unclassified'          otherwise
    Vectorized implementation (no per-row Python loops).
    """
    tcol = f"label_{target_level}"
    if tcol not in df.columns:
        raise ValueError(f"Expected '{tcol}' column on the frame.")

    x = df.copy()
    # Work with object dtype to avoid pandas casting surprises
    x[tcol] = x[tcol].astype(object)
    missing = x[tcol].isna() | (x[tcol].astype(str).str.strip() == "")

    if not missing.any():
        return x

    if target_level == 1:
        # No parents exist at L1
        x.loc[missing, tcol] = UNCLASS_OTHER
        return x

    # Find the highest parent that is non-null in the row.
    parent_cols = [f"label_{j}" for j in range(target_level - 1, 0, -1) if f"label_{j}" in x.columns]
    if not parent_cols:
        # Fallback: nothing to look at
        x.loc[missing, tcol] = UNCLASS_OTHER
        return x

    # Because parent_cols are in descending order (L-1, L-2, ..., 1),
    # bfill horizontally will push the first non-null to the leftmost column.
    parent_series = x[parent_cols].bfill(axis=1).iloc[:, 0]

    # Compose replacement values
    have_parent = parent_series.notna() & (parent_series.astype(str).str.strip() != "")
    replace_vals = pd.Series(UNCLASS_OTHER, index=x.index, dtype=object)
    replace_vals.loc[have_parent] = parent_series.loc[have_parent].astype(str) + PARENT_OTHER_SUFFIX

    # Fill only where target label is missing
    x.loc[missing, tcol] = replace_vals.loc[missing].values
    return x




# -----------------------------
# Feature matrix at chosen commodity level
# -----------------------------
@st.cache_data(show_spinner=True)
def build_household_matrix_by_level(
    df: pd.DataFrame,
    years: list,
    level: int,
    exp_measure: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Build the household-wide matrix at the chosen classification level.

    Steps:
      1) filter by years
      2) attach label_1...label_{level}
      3) fill missing label_{level} to '{nearest_parent} — Other' (or 'Other — Unclassified')
      4) group Year, ID, label_{level} and pivot to wide
      5) return (wide, meta, label_col_name)
    """
    # 1) Year filter
    df = df[df["Year"].isin(years)].copy()

    # 2) Labels for 1..level
    df = add_class_labels(df, level)
    label_col = f"label_{level}"

    # 3) Group unlabeled into parent '— Other' (or 'Other — Unclassified')
    df = group_unlabeled_to_parent_other(df, target_level=level)

    # 4) Aggregate and pivot
    # Sum expenditure to (Year, ID, label_{level})
    agg = (
        df.groupby(["Year", "ID", label_col], observed=False, dropna=False)[exp_measure]
          .sum(min_count=1)
          .reset_index()
    )

    # Wide: households × features (chosen level)
    wide = agg.pivot_table(
        index=["Year", "ID"],
        columns=label_col,
        values=exp_measure,
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    # Ensure string column names
    wide.columns = wide.columns.astype(str)
    wide.sort_index(axis=1, inplace=True)

    # 5) Meta (Province/County/Settlement if present)
    meta_cols = [c for c in [
        "Province_code", "Province_farsi_name",
        "County_code",   "County_farsi_name",
        "Urban_Rural_name", "Urban_Rural"
    ] if c in df.columns]
    meta = (
        df.drop_duplicates(subset=["Year", "ID"])
          .set_index(["Year", "ID"])[meta_cols]
          .reindex(wide.index)
    )

    return wide, meta, label_col



# -----------------------------
# Weights
# -----------------------------
def guess_weight_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("Weight", "Weight_x", "Weight_y", "weight"):
        if c in df.columns: return c
    return None
