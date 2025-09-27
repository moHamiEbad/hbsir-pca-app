from __future__ import annotations
from typing import List, Tuple
import os
import streamlit as st
from hbspca.id_filter import parse_ids_from_file, parse_ids_from_text
import pandas as pd

from .types import SidebarState
from .config import (YEARS_MIN, YEARS_MAX, DEFAULT_YEARS, NORM_MODES, EXP_MEASURES,
                     PROVINCE_COL, COUNTY_COL)

def sidebar(years_full: List[int], hh_for_menus) -> SidebarState:
    st.sidebar.header("Data source")
    use_existing_opt = st.sidebar.radio("Do you already have PCA result files?",
                                        ["No — I will run PCA now", "Yes — I have saved files"])
    use_existing = use_existing_opt == "Yes — I have saved files"
    load_dir = None
    if use_existing:
        st.sidebar.info("Provide a folder containing per-year Excel files (`pca_year_YYYY.xlsx`).")
        load_dir = st.sidebar.text_input("Existing results directory", value="./pca_outputs/pca_output", placeholder="/path/to/output")
        if load_dir and os.path.isdir(load_dir):
            st.success("Directory looks valid. Select years and jump to Plotting, or re-run PCA.")
        else:
            st.warning("Enter a valid directory path; otherwise run PCA.")

    st.sidebar.header("1. Years")
    year_mode = st.sidebar.radio("Year selection mode", ["Interval", "List"], horizontal=True)
    if year_mode == "Interval":
        c1, c2 = st.sidebar.columns(2)
        min_year = c1.number_input("Start year", YEARS_MIN, YEARS_MAX+3, DEFAULT_YEARS[0], 1)
        max_year = c2.number_input("End year",   YEARS_MIN, YEARS_MAX+3, max(DEFAULT_YEARS[-1], min_year), 1)
        years = list(range(int(min_year), int(max_year) + 1))
    else:
        years = sorted(list(set(st.sidebar.multiselect("Years", list(range(YEARS_MIN, YEARS_MAX+1)),
                                                      default=DEFAULT_YEARS))))

    st.sidebar.header("2. Area")
    area_kind = st.sidebar.radio("Area for PCA", ["Whole country", "Subareas"])

    province_col = PROVINCE_COL
    county_col = COUNTY_COL

    selected_provinces = []
    selected_counties  = []
    if area_kind == "Subareas":
        if hh_for_menus is None or len(hh_for_menus) == 0:
            st.sidebar.info("Select years, then click **Load area menus for selected years** in the main pane.")
            selected_provinces = []
            selected_counties = []
        else:
            provinces = sorted(hh_for_menus[PROVINCE_COL].dropna().unique().tolist())
            selected_provinces = st.sidebar.multiselect("Select provinces", provinces, default=[])
            counties = []
            if selected_provinces:
                county_df = hh_for_menus[hh_for_menus[PROVINCE_COL].isin(selected_provinces)]
                counties = sorted(county_df[COUNTY_COL].dropna().unique().tolist())
            selected_counties = st.sidebar.multiselect("Select counties (optional)", counties, default=[])

    st.sidebar.header("Settlement")
    settlement = st.sidebar.radio("Settlement type", ["Both", "Urban", "Rural"], horizontal=True)

    st.sidebar.header("3. Commodity level")
    level = st.sidebar.selectbox("Commodity classification level", options=[1,2,3,4,5], index=1,
                                 format_func=lambda x: f"Level {x}")

    st.sidebar.header("4–5. PCA options")
    weighted = st.sidebar.checkbox("Use household weights", value=True)
    norm_mode = st.sidebar.selectbox("Preprocessing", NORM_MODES, index=3)
    exp_measure = st.sidebar.radio("Expenditure measure", EXP_MEASURES, horizontal=True)

    st.sidebar.header("6. Save")
    save_dir = st.sidebar.text_input("Output directory", value="./pca_outputs/pca_output", placeholder="./pca_outputs/pca_output")
    save_as_excel = st.sidebar.checkbox("Save as Excel (.xlsx with sheets)", value=True)

    # ---- 7. Optional household ID filter (collapsible) ----
    with st.sidebar.expander("7. Household ID Filter (optional)", expanded=False):
        ignore_year_for_ids = st.checkbox(
            "Ignore Year column in ID file",
            value=True
        )

        ids_file = st.file_uploader(
            "CSV/XLSX with 'ID' (and optional 'Year'/'cluster')",
            type=["csv", "xlsx"],
        )
        cluster_col = st.text_input("Cluster column name (optional)", value="")
        keep_clusters_txt = st.text_input("Keep clusters (comma-separated)", value="")

        id_filter_df = None
        keep_clusters = [v.strip() for v in keep_clusters_txt.split(",") if v.strip()] if keep_clusters_txt else None

        if ids_file is not None:
            try:
                id_filter_df = parse_ids_from_file(
                    ids_file,
                    cluster_col=cluster_col if cluster_col else None,
                    keep_clusters=keep_clusters,
                    ignore_year=ignore_year_for_ids,
                )
                st.success(f"Found {len(id_filter_df)} IDs.")
            except Exception as e:
                st.error(f"Error reading ID file: {e}")

        # Optional text input
        ids_text = st.text_area("Or paste IDs here (space/comma/newline separated):")
        if ids_text.strip():
            ids_text_df = parse_ids_from_text(ids_text)
            if id_filter_df is None:
                id_filter_df = ids_text_df
            else:
                id_filter_df = pd.concat([id_filter_df, ids_text_df]).drop_duplicates().reset_index(drop=True)

    return SidebarState(
        use_existing=use_existing,
        load_dir=load_dir,
        years=years,
        area_kind=area_kind,
        selected_provinces=selected_provinces,
        selected_counties=selected_counties,
        settlement=settlement,
        level=int(level),
        weighted=weighted,
        norm_mode=norm_mode,
        exp_measure=exp_measure,
        save_dir=save_dir,
        save_as_excel=save_as_excel,
        id_filter_df=id_filter_df,
        ignore_year_for_ids=ignore_year_for_ids,
    )
