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
# Feature matrix at chosen commodity level
# -----------------------------
@st.cache_data(show_spinner=True)
def build_household_matrix_by_level(df: pd.DataFrame,
                                    years: List[int],
                                    level: int,
                                    exp_measure: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Aggregate household expenditures *by chosen commodity level* (1..5).
    - Columns = categories at that level (use label if present, otherwise 'Unlabeled_L{level}').
    - No row/column is dropped other than the area/settlement/year filters applied upstream.
    Returns: (wide_features, household_meta, label_col_name)
    """
    df = df[df["Year"].isin(years)].copy()
    df = add_multi_class_labels(df, level)
    label_col = f"label_{level}"

    # Build category key = label if available else 'Unlabeled_L{level}'
    # (We keep unlabeled as a single bucket; you can fan-out by code if you prefer.)
    cat_key = df[label_col].copy()
    if label_col not in df.columns:
        # If the chosen level is completely unavailable, fallback to a single bucket
        cat_key = pd.Series([f"Unlabeled_L{level}"] * len(df), index=df.index)
    else:
        cat_key = cat_key.fillna(f"Unlabeled_L{level}")

    tall = (
        df.assign(__cat__=cat_key)
          .groupby(["Year", "ID", "__cat__"], observed=False, dropna=False)[exp_measure]
          .sum(min_count=1)
          .reset_index()
    )

    # Pivot to wide household × categories
    wide = (tall
            .pivot_table(index=["Year", "ID"], columns="__cat__", values=exp_measure,
                         aggfunc="sum", fill_value=0, observed=False)
            .sort_index(axis=1))
    wide.columns = wide.columns.astype(str)

    # Household meta (Province/County/Settlement)
    keep_meta = ["Province_code","Province_farsi_name","County_code","County_farsi_name"]
    urc = urban_rural_col(df)
    if urc: keep_meta.append(urc)

    base = (df.drop_duplicates(subset=["Year", "ID"])
              .set_index(["Year", "ID"]))
    meta = base[[c for c in keep_meta if c in base.columns]].reindex(wide.index)

    return wide, meta, label_col


# -----------------------------
# Weights
# -----------------------------
def guess_weight_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("Weight", "Weight_x", "Weight_y", "weight"):
        if c in df.columns: return c
    return None
