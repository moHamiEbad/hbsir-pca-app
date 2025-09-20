from __future__ import annotations
from typing import List, Optional
import pandas as pd
import streamlit as st
import hbsir

from .config import PROVINCE_COL, COUNTY_COL, URBAN_RURAL_NAME, URBAN_RURAL_FALLBACK

@st.cache_data(show_spinner=False)
def setup_hbsir(years: List[int],
                method: str = "create_from_raw",
                download_source: str = "mirror") -> bool:
    """
    Prepare HBSIR local store (cached).
    """
    hbsir.setup(years=years, method=method, download_source=download_source, replace=False)
    return True

@st.cache_data(show_spinner=True)
def load_households(years: List[int]) -> pd.DataFrame:
    hh = hbsir.load_table("household_information", years=years, form="normalized")
    hh = hbsir.add_attribute(hh, "Urban_Rural")
    hh = hbsir.add_attribute(hh, "Province", aspects=["code", "farsi_name"])
    hh = hbsir.add_attribute(hh, "County",   aspects=["code", "farsi_name"])
    return hh

@st.cache_data(show_spinner=True)
def load_expenditures(years: List[int]) -> pd.DataFrame:
    try:
        exp = hbsir.load_table("Expenditures", years=years, form="normalized")
    except Exception:
        cats = ["food","tobacco","cloth","home","furniture","medical",
                "transportation","communication","entertainment","education",
                "hotel","miscellaneous","durable","investment"]
        frames = []
        for c in cats:
            try:
                frames.append(hbsir.load_table(c, years=years, form="normalized").assign(Table_Name=c))
            except Exception:
                pass
        if not frames:
            raise RuntimeError("No expenditure tables available for the selected years.")
        exp = pd.concat(frames, ignore_index=True)

    if "Weight" not in exp.columns:
        try:
            exp = hbsir.add_weight(exp)
        except Exception:
            pass
    return exp

@st.cache_data(show_spinner=True)
def merge_exp_hh(exp: pd.DataFrame, hh: pd.DataFrame) -> pd.DataFrame:
    return exp.merge(hh, on=["Year", "ID"], how="left", validate="m:1")

def _urban_rural_col(df: pd.DataFrame) -> Optional[str]:
    if URBAN_RURAL_NAME in df.columns: return URBAN_RURAL_NAME
    if URBAN_RURAL_FALLBACK in df.columns: return URBAN_RURAL_FALLBACK
    return None

@st.cache_data(show_spinner=False)
def filter_area_settlement(df: pd.DataFrame,
                           years: List[int],
                           level_choice: str,
                           provinces: List[str],
                           counties: List[str],
                           settlement: str,
                           province_col: str = PROVINCE_COL,
                           county_col: str   = COUNTY_COL) -> pd.DataFrame:
    out = df[df["Year"].isin(years)].copy()

    urc = _urban_rural_col(out)
    if settlement != "Both" and urc in out.columns:
        out = out[out[urc].str.lower() == settlement.lower()]

    if level_choice == "Province" and provinces:
        out = out[out[province_col].isin(provinces)]
    if level_choice in ("County", "Province") and counties:
        out = out[out[county_col].isin(counties)]
    return out

def guess_weight_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("Weight", "Weight_x", "Weight_y", "weight"):
        if c in df.columns:
            return c
    return None
