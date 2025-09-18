# app.py
import os
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from data_io import (
    setup_hbsir, load_households, load_expenditures, merge_exp_hh,
    filter_area_settlement, build_household_matrix_by_level,
    guess_weight_col
)

from pca_utils import (
    normalize_features, weighted_standardize, weighted_pca,
    align_signs_to_reference, variance_table,
    biplot_loadings, save_year_results, ensure_dir,
    plot_cumulative_variance, make_scores_scatter, make_biplot, full_loadings_df
)

# ----------------------------------
# Page
# ----------------------------------
st.set_page_config(page_title="HBSIR PCA — Household level", layout="wide")
st.title("HBSIR PCA — Household level workflow")
st.caption("PCA runs on households; area/settlement only filter. Per-year results are saved, then plotted with flexible aggregation.")

# ----------------------------------
# Session store
# ----------------------------------
if "pca_results" not in st.session_state:
    st.session_state["pca_results"] = {}              # type: Dict[int, Dict[str, Any]]
if "ref_components" not in st.session_state:
    st.session_state["ref_components"] = None
if "years_run" not in st.session_state:
    st.session_state["years_run"] = []                # type: List[int]


# ----------------------------------
# Step 0 — Use existing saved files?
# ----------------------------------
st.sidebar.header("Data source")
use_existing = st.sidebar.radio(
    "Do you already have PCA result files?",
    ["No — I will run PCA now", "Yes — I have saved files"],
    horizontal=False,
)

load_dir = None
if use_existing == "Yes — I have saved files":
    st.sidebar.info("Provide the directory containing per-year Excel files (`pca_year_YYYY.xlsx`) or CSV trios.")
    load_dir = st.sidebar.text_input("Existing results directory", value="", placeholder="/path/to/output")
    if load_dir and os.path.isdir(load_dir):
        st.success("Directory looks valid. You can jump to the **Plotting** section below after selecting years.")
    else:
        st.warning("Enter a valid directory path; otherwise you can run PCA now.")

# ----------------------------------
# 1) Years
# ----------------------------------
st.sidebar.header("1. Years")
year_mode = st.sidebar.radio("Year selection mode", ["Interval", "List"], horizontal=True)
if year_mode == "Interval":
    c1, c2 = st.sidebar.columns(2)
    min_year = c1.number_input("Start year", 1370, 1403, 1398, 1)
    max_year = c2.number_input("End year",   1370, 1403, max(1398, min_year), 1)
    years: List[int] = list(range(int(min_year), int(max_year)+1))
else:
    years = sorted(list(set(st.sidebar.multiselect("Years", list(range(1363, 1501)), default=[1398,1399,1400]))))

# HBSIR setup
setup_hbsir(years)

# ----------------------------------
# 2) Area selection (Whole country or Subareas)
# ----------------------------------
st.sidebar.header("2. Area")
area_kind = st.sidebar.radio("Area for PCA", ["Whole country", "Subareas"], horizontal=False)

hh_for_menus = load_households(years)
province_col = "Province_farsi_name"
county_col   = "County_farsi_name"

selected_provinces: List[str] = []
selected_counties:  List[str] = []

if area_kind == "Subareas":
    provinces = sorted(hh_for_menus[province_col].dropna().unique().tolist())
    selected_provinces = st.sidebar.multiselect("Select provinces", provinces, default=[])
    counties = []
    if selected_provinces:
        county_df = hh_for_menus[hh_for_menus[province_col].isin(selected_provinces)]
        counties = sorted(county_df[county_col].dropna().unique().tolist())
    selected_counties = st.sidebar.multiselect("Select counties (optional)", counties, default=[])

# 2.1) Settlement
st.sidebar.header("Settlement")
settlement = st.sidebar.radio("Settlement type", ["Both", "Urban", "Rural"], horizontal=True)

# ----------------------------------
# 3) Commodity level (no feature picking)
# ----------------------------------
st.sidebar.header("3. Commodity level")
level = st.sidebar.selectbox("Commodity classification level", options=[1,2,3,4,5], index=1,
                             format_func=lambda x: f"Level {x}")

# ----------------------------------
# 5) Weights & PCA method
# ----------------------------------
st.sidebar.header("4–5. PCA options")
weighted = st.sidebar.checkbox("Use household weights", value=True)
norm_mode = st.sidebar.selectbox(
    "Preprocessing",
    [
        "Raw levels",
        "Budget shares (center-only)",
        "Budget shares (z-score columns)",
        "CLR (compositional)",
        "Log(1+x)",
        "Budget shares + Log"
    ],
    index=2  # start on the safer z-score variant
)

exp_measure = st.sidebar.radio("Expenditure measure", ["Net_Expenditure","Gross_Expenditure"], horizontal=True)

# ----------------------------------
# 6) Save location
# ----------------------------------
st.sidebar.header("6. Save")
save_dir = st.sidebar.text_input("Output directory", value="./pca_output", placeholder="./pca_output")
save_as_excel = st.sidebar.checkbox("Save as Excel (.xlsx with sheets)", value=True)

# ----------------------------------
# 7) Run PCA
# ----------------------------------
run_btn = st.sidebar.button("Run PCA", type="primary")

# Load raw data early (cached)
hh = load_households(years)
exp = load_expenditures(years)
exp_hh_all = merge_exp_hh(exp, hh)

# Filter by area/settlement only (no data deletion otherwise)
level_choice_for_filter = "Province" if area_kind == "Subareas" else "Country"
exp_hh_view = filter_area_settlement(
    df=exp_hh_all, years=years,
    level_choice=level_choice_for_filter,
    provinces=selected_provinces, counties=selected_counties,
    settlement=settlement, province_col=province_col, county_col=county_col
)

if run_btn:
    st.session_state.pca_results.clear()
    st.session_state.ref_components = None
    st.session_state.years_run = list(years)

    # quick sanity
    if not save_dir:
        st.warning("No output directory specified. Files will still be offered as downloads after each year.")
    ensure_dir(save_dir) if save_dir else None

    with st.spinner("Running per-year PCA at household level…"):
        # Build feature matrix ONCE per year (by chosen commodity level)
        weight_col = guess_weight_col(exp_hh_view)

        col1, col2 = st.columns([1,1])
        for idx, yr in enumerate(sorted(years)):
            # Household matrix for this year
            wide_y, meta_y, _ = build_household_matrix_by_level(
                exp_hh_view, [yr], level, exp_measure
            )
            if wide_y.empty:
                st.warning(f"Year {yr}: no data after filters. Skipped.")
                continue

            # weights aligned to households of this year
            w = None
            if weighted and weight_col is not None:
                hh_w = (exp_hh_view.drop_duplicates(subset=["Year","ID"])
                        [["Year","ID",weight_col]]
                        .set_index(["Year","ID"]))
                w = hh_w.reindex(wide_y.index)[weight_col].fillna(0).values.astype(float)
                w = np.where(w <= 0, 1e-12, w)

            # preprocess
            X = normalize_features(wide_y, norm_mode)
            # PREPROCESSING
            if norm_mode == "Raw levels":
                # z-score raw aggregated expenditures
                Z, _, _ = weighted_standardize(wide_y.values, w if weighted else None)

            elif norm_mode == "Budget shares (center-only)":
                S = wide_y.div(wide_y.sum(axis=1).where(wide_y.sum(axis=1) != 0, 1), axis=0)
                # center columns only (no scaling) – original behavior
                if weighted and w is not None and np.sum(w) > 0:
                    ww = w / np.sum(w)
                    mu = np.average(S.values, axis=0, weights=ww)
                else:
                    mu = np.nanmean(S.values, axis=0)
                Z = S.values - mu

            elif norm_mode == "Budget shares (z-score columns)":
                S = wide_y.div(wide_y.sum(axis=1).where(wide_y.sum(axis=1) != 0, 1), axis=0)
                # now z-score columns (weighted) so no single category dominates
                Z, _, _ = weighted_standardize(S.values, w if weighted else None)

            elif norm_mode == "CLR (compositional)":
                # shares → add epsilon → log → row-mean center → z-score columns
                S = wide_y.div(wide_y.sum(axis=1).where(wide_y.sum(axis=1) != 0, 1), axis=0)
                eps = 1e-6
                S = S.clip(lower=eps)
                logS = np.log(S)
                clr = logS.sub(logS.mean(axis=1), axis=0)  # row-wise mean-log center
                Z, _, _ = weighted_standardize(clr.values, w if weighted else None)

            elif norm_mode == "Log(1+x)":
                L = np.log1p(wide_y)
                Z, _, _ = weighted_standardize(L.values, w if weighted else None)

            elif norm_mode == "Budget shares + Log":
                # shares then log of shares (not log1p of shares)
                S = wide_y.div(wide_y.sum(axis=1).where(wide_y.sum(axis=1) != 0, 1), axis=0)
                eps = 1e-6
                logS = np.log(S.clip(lower=eps))
                Z, _, _ = weighted_standardize(logS.values, w if weighted else None)

            else:
                # fallback
                Z, _, _ = weighted_standardize(wide_y.values, w if weighted else None)


            k = Z.shape[1]
            comps, lam, scores, _ = weighted_pca(Z, w if weighted else None, n_components=k)

            # Sign align to first year
            if st.session_state.ref_components is None:
                st.session_state.ref_components = comps.copy()
            else:
                comps, scores = align_signs_to_reference(st.session_state.ref_components, comps, scores)

            # Dataframes
            scores_df = pd.DataFrame(scores, index=wide_y.index,
                                     columns=[f"PC{i}" for i in range(1, scores.shape[1]+1)]).reset_index()
            scores_with_meta = (scores_df.merge(meta_y.reset_index(), on=["Year","ID"], how="left")
                                          .sort_values(["Year","ID"]))
            var_df = variance_table(lam)

            # ALL loadings for saving/plotting later (ALL PCs)
            feature_names = wide_y.columns.astype(str).tolist()
            load_df_all = full_loadings_df(
                components=comps,
                explained_var=lam,
                feature_names=feature_names,
                scale=1.0,         # keep canonical values in files
            )

            # Save to disk
            saved = {}
            try:
                if save_dir:
                    saved = save_year_results(save_dir, yr, scores_with_meta, load_df_all, var_df, as_excel=save_as_excel)
            except Exception as e:
                st.error(f"Year {yr}: failed to save files: {e}")

            # Keep in memory (cache for plotting)
            st.session_state.pca_results[yr] = dict(
                X_cols = wide_y.columns.tolist(),
                comps  = comps, lam = lam,
                scores_df = scores_df,
                scores_with_meta = scores_with_meta,
                load_df_all = load_df_all,      # ← store ALL loadings here
                var_df = var_df,
                saved = saved
            )

            # Show cumulative variance figure
            fig_var = plot_cumulative_variance(var_df, title=f"Year {yr} — Cumulative variance")
            (col1 if idx % 2 == 0 else col2).plotly_chart(fig_var, use_container_width=True)

        st.success("PCA finished. See the **Plotting** section below.")


# ----------------------------------
# 9–15) Plotting section (appears after run or when loading existing)
# ----------------------------------
st.markdown("---")
st.header("Plotting")

# If user wants to load existing files from disk and skip PCA run
if use_existing == "Yes — I have saved files" and load_dir and os.path.isdir(load_dir) and not st.session_state.pca_results:
    # Try to read per-year Excel
    found_years = []
    for y in years:
        xlsx = os.path.join(load_dir, f"pca_year_{y}.xlsx")
        scores_df = load_df = var_df = None
        if os.path.isfile(xlsx):
            try:
                scores_with_meta = pd.read_excel(xlsx, sheet_name="scores")
                var_df = pd.read_excel(xlsx, sheet_name="variance")
                load_df_all = pd.read_excel(xlsx, sheet_name="loadings")
                # recover minimal scores_df (Year, ID, PCs)
                pc_cols = [c for c in scores_with_meta.columns if c.startswith("PC")]
                scores_df = scores_with_meta[["Year","ID"] + pc_cols].copy()
                st.session_state.pca_results[y] = dict(
                    X_cols=[], comps=None, lam=var_df["Explained variance"].values,
                    scores_df=scores_df, scores_with_meta=scores_with_meta,
                    load_df_all=load_df_all, var_df=var_df, saved={"excel": xlsx}
                )
                found_years.append(y)
            except Exception:
                pass
    if found_years:
        st.success(f"Loaded saved results for years: {found_years}")
        st.session_state.years_run = found_years
    else:
        st.warning("No readable per-year files found in the directory.")

if not st.session_state.pca_results:
    st.info("Run PCA above or load existing files to enable plotting.")
    st.stop()

years_ready = sorted(st.session_state.pca_results.keys())

# PC selection for plots (2D / 3D — we’ll use 2D as spec focuses on biplots + scores)
st.subheader("PC selection")
max_pc_global = max(len([c for c in v["scores_df"].columns if c.startswith("PC")])
                    for v in st.session_state.pca_results.values())

plot_dim = st.radio("Plot dimensionality", ["2D", "3D"], horizontal=True)

pc1 = st.number_input("PC for X-axis", min_value=1, max_value=max_pc_global, value=1, step=1)
pc2 = st.number_input("PC for Y-axis", min_value=1, max_value=max_pc_global, value=2, step=1)

if plot_dim == "3D":
    pc3 = st.number_input("PC for Z-axis", min_value=1, max_value=max_pc_global, value=3, step=1)
    pcs = (int(pc1), int(pc2), int(pc3))
else:
    pcs = (int(pc1), int(pc2))

# Choose plot type
plot_kind = st.radio("Plot type", ["Scores", "Loadings", "Biplot"], horizontal=True)
loading_scale = 1.0
if plot_kind == "Biplot":
    loading_scale = st.slider("Loadings scale", 0.2, 100.0, 1.5, 0.1)

# For Scores: point granularity and filters
point_level = None
color_by = None
filter_provs: List[str] = []
filter_counties: List[str] = []
if plot_kind in ("Scores", "Biplot"):
    st.subheader("Scores settings")
    point_level = st.radio(
        "Each point represents a…", ["Household", "County", "Province"], horizontal=True
    )
    if point_level == "Household":
        show_subset = st.checkbox("Filter households by Province/County", value=False)
        if show_subset:
            # build options from all results
            meta_any = next(iter(st.session_state.pca_results.values()))["scores_with_meta"]
            provs = sorted(meta_any["Province_farsi_name"].dropna().unique().tolist()) if "Province_farsi_name" in meta_any.columns else []
            filter_provs = st.multiselect("Provinces", provs, default=provs)
            cnts = []
            if filter_provs:
                cnts = sorted(meta_any[meta_any["Province_farsi_name"].isin(filter_provs)]
                              ["County_farsi_name"].dropna().unique().tolist())
            filter_counties = st.multiselect("Counties (optional)", cnts, default=cnts)
        color_by = st.selectbox("Color households by", ["None","Settlement","Province","County"], index=1)
        if color_by == "None": color_by = None

        # Transparency helps reveal density & colors underneath
        point_opacity = st.slider("Point opacity", 0.05, 1.0, 0.30, 0.05,
                                help="Lower = more transparent; overlapping points look darker.")
        randomize_order = False
        if point_level == "Household" and color_by is not None:
            randomize_order = st.checkbox("Randomize category draw order", value=True,
                                        help="Prevents a single color from always drawing on top.")

    else:
        # Province/County level
        meta_any = next(iter(st.session_state.pca_results.values()))["scores_with_meta"]
        if point_level == "Province":
            provs = sorted(meta_any["Province_farsi_name"].dropna().unique().tolist()) if "Province_farsi_name" in meta_any.columns else []
            choose = st.checkbox("Show only a subset of provinces", value=False)
            filter_provs = st.multiselect("Provinces", provs, default=provs if not choose else [])
        else:
            cnts = sorted(meta_any["County_farsi_name"].dropna().unique().tolist()) if "County_farsi_name" in meta_any.columns else []
            choose = st.checkbox("Show only a subset of counties", value=False)
            filter_counties = st.multiselect("Counties", cnts, default=cnts if not choose else [])

# For Loadings filtering (by feature names)
loading_filter = None
if plot_kind in ("Loadings", "Biplot"):
    st.subheader("Loadings filter")
    # union of all feature names across years
    features_all = sorted(set().union(*[
    set(v["load_df_all"]["feature"].astype(str).tolist())
    for v in st.session_state.pca_results.values()
    ]))
    loading_filter = st.multiselect("Keep only these features (optional)", features_all, default=features_all)

# Year stepper
st.subheader("Year")
if "plot_year_idx" not in st.session_state:
    st.session_state.plot_year_idx = 0
c_prev, c_year, c_next = st.columns([1,3,1])
with c_prev:
    if st.button("◀ Prev"):
        st.session_state.plot_year_idx = max(0, st.session_state.plot_year_idx - 1)
with c_next:
    if st.button("Next ▶"):
        st.session_state.plot_year_idx = min(len(years_ready)-1, st.session_state.plot_year_idx + 1)
cur_year = years_ready[st.session_state.plot_year_idx]
c_year.markdown(f"**Showing year:** {cur_year}")

# Build the plot for the current year
res = st.session_state.pca_results[cur_year]
scores_with_meta = res["scores_with_meta"].copy()
pc_cols = [f"PC{i}" for i in range(1, len([c for c in res['scores_df'].columns if c.startswith('PC')])+1)]

# Scores dataframe to plot
scores_plot = scores_with_meta[["Year","ID"] + pc_cols].copy()
if "Province_farsi_name" in scores_with_meta.columns:
    scores_plot["Province"] = scores_with_meta["Province_farsi_name"]
if "County_farsi_name" in scores_with_meta.columns:
    scores_plot["County"]   = scores_with_meta["County_farsi_name"]
urc = "Urban_Rural_name" if "Urban_Rural_name" in scores_with_meta.columns else ("Urban_Rural" if "Urban_Rural" in scores_with_meta.columns else None)
if urc: scores_plot["Settlement"] = scores_with_meta[urc]

# Aggregate if needed
if plot_kind in ("Scores", "Biplot"):
    if point_level == "Household":
        if filter_provs:
            scores_plot = scores_plot[scores_plot.get("Province").isin(filter_provs)]
        if filter_counties:
            scores_plot = scores_plot[scores_plot.get("County").isin(filter_counties)]
        # optional color column is selected above
    elif point_level == "Province":
        if filter_provs:
            scores_plot = scores_plot[scores_plot.get("Province").isin(filter_provs)]
        keep = ["Year","Province"] + pc_cols
        scores_plot = (scores_plot[keep]
                       .groupby(["Year","Province"], as_index=False)[pc_cols].mean())
        color_by = None  # coloring by Province creates one color per point; skip
    else:  # County
        if filter_counties:
            scores_plot = scores_plot[scores_plot.get("County").isin(filter_counties)]
        keep = ["Year","County"] + pc_cols
        scores_plot = (scores_plot[keep]
                       .groupby(["Year","County"], as_index=False)[pc_cols].mean())
        color_by = None


# ---- Give every point a human-readable name for hover/text ----
# Household → ID; Province/County → their label
if point_level == "Household":
    scores_plot["point_name"] = scores_plot["ID"].astype(str)
elif point_level == "Province":
    # after the groupby above, 'Province' exists
    scores_plot["point_name"] = scores_plot["Province"].fillna("(unnamed)").astype(str)
else:  # "County"
    scores_plot["point_name"] = scores_plot["County"].fillna("(unnamed)").astype(str)



# Loadings to plot (with optional feature filter and scale)
load_df_plot = res["load_df_all"].copy()
if plot_kind in ("Loadings","Biplot") and loading_filter:
    load_df_plot = load_df_plot[load_df_plot["feature"].astype(str).isin(loading_filter)].copy()
if plot_kind == "Biplot" and loading_scale != 1.0:
    # rescale selected PCs' loadings
    for c in load_df_plot.columns:
        if c.startswith("PC"):
            load_df_plot[c] = load_df_plot[c] * loading_scale

# Finally draw the selected plot
if plot_kind == "Scores":
    if plot_dim == "2D":
        fig = make_scores_scatter(
            scores_plot, pcs=(pcs[0], pcs[1]),
            color_by=color_by,
            title=f"Scores — Year {cur_year} ({point_level})",
            opacity=point_opacity,
            randomize_trace_order=randomize_order,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # 3D scores
        xlab, ylab, zlab = f"PC{pcs[0]}", f"PC{pcs[1]}", f"PC{pcs[2]}"
        missing = [c for c in (xlab, ylab, zlab) if c not in scores_plot.columns]
        if missing:
            st.warning(f"Selected PCs not available for {cur_year}: {missing}")
        else:
            fig = go.Figure()
            df3 = scores_plot.copy()
            if color_by and color_by in df3.columns:
                col = color_by
                # keep a visible '(missing)' bucket and don't drop NaNs
                df3[col] = df3[col].where(df3[col].notna(), "(missing)")
                groups = list(df3.groupby(col, dropna=False))
                if randomize_order and len(groups) > 1:
                    idx = np.random.permutation(len(groups))
                    groups = [groups[i] for i in idx]
                for key, grp in groups:
                    fig.add_trace(go.Scatter3d(
                        x=grp[xlab], y=grp[ylab], z=grp[zlab],
                        mode="markers",
                        name=str(key),
                        marker=dict(size=3, opacity=point_opacity),
                        text=grp.get("point_name"),
                        hovertemplate="<b>%{text}</b><br>"
                                      f"{xlab}=%{{x:.3f}}<br>"
                                      f"{ylab}=%{{y:.3f}}<br>"
                                      f"{zlab}=%{{z:.3f}}"
                                      "<extra>%{fullData.name}</extra>"
                    ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=df3[xlab], y=df3[ylab], z=df3[zlab],
                    mode="markers",
                    marker=dict(size=3, opacity=point_opacity),
                    text=df3.get("point_name"),
                    hovertemplate="<b>%{text}</b><br>"
                                  f"{xlab}=%{{x:.3f}}<br>"
                                  f"{ylab}=%{{y:.3f}}<br>"
                                  f"{zlab}=%{{z:.3f}}"
                                  "<extra></extra>"
                ))
            fig.update_layout(
                title=f"Scores — Year {cur_year} ({point_level})",
                scene=dict(xaxis_title=xlab, yaxis_title=ylab, zaxis_title=zlab),
                template="plotly_white",
                height=700
            )
            st.plotly_chart(fig, use_container_width=True)

elif plot_kind == "Loadings":
    st.dataframe(load_df_plot)

else:  # Biplot
    if plot_dim == "2D":
        plot_df = scores_plot.rename(columns={f"PC{pcs[0]}": f"PC{pcs[0]}", f"PC{pcs[1]}": f"PC{pcs[1]}"})
        fig = make_biplot(
            plot_df, load_df_plot, pcs=(pcs[0], pcs[1]),
            title=f"Biplot — Year {cur_year} ({point_level})",
            opacity=point_opacity,
            randomize_trace_order=randomize_order,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # 3D biplot (scores + loadings arrows)
        xlab, ylab, zlab = f"PC{pcs[0]}", f"PC{pcs[1]}", f"PC{pcs[2]}"
        missing_s = [c for c in (xlab, ylab, zlab) if c not in scores_plot.columns]
        missing_l = [c for c in (xlab, ylab, zlab) if c not in load_df_plot.columns]
        if missing_s or missing_l:
            st.warning(f"Selected PCs not available for {cur_year}. "
                       f"Scores missing: {missing_s}; Loadings missing: {missing_l}")
        else:
            fig = go.Figure()

            # scores (NaN-safe grouping + transparency)
            df3 = scores_plot.copy()
            if color_by and color_by in df3.columns:
                col = color_by
                df3[col] = df3[col].where(df3[col].notna(), "(missing)")
                groups = list(df3.groupby(col, dropna=False))
                if randomize_order and len(groups) > 1:
                    idx = np.random.permutation(len(groups))
                    groups = [groups[i] for i in idx]
                for key, grp in groups:
                    fig.add_trace(go.Scatter3d(
                        x=grp[xlab], y=grp[ylab], z=grp[zlab],
                        mode="markers",
                        name=str(key),
                        marker=dict(size=3, opacity=point_opacity),
                        text=grp.get("point_name"),
                        hovertemplate="<b>%{text}</b><br>"
                                      f"{xlab}=%{{x:.3f}}<br>"
                                      f"{ylab}=%{{y:.3f}}<br>"
                                      f"{zlab}=%{{z:.3f}}"
                                      "<extra>%{fullData.name}</extra>"
                    ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=df3[xlab], y=df3[ylab], z=df3[zlab],
                    mode="markers",
                    name="Scores",
                    marker=dict(size=3, opacity=point_opacity),
                    text=df3.get("point_name"),
                    hovertemplate="<b>%{text}</b><br>"
                                  f"{xlab}=%{{x:.3f}}<br>"
                                  f"{ylab}=%{{y:.3f}}<br>"
                                  f"{zlab}=%{{z:.3f}}"
                                  "<extra></extra>"
                ))

            # loadings arrows
            for _, r in load_df_plot.iterrows():
                fig.add_trace(go.Scatter3d(
                    x=[0, r.get(xlab, 0.0)],
                    y=[0, r.get(ylab, 0.0)],
                    z=[0, r.get(zlab, 0.0)],
                    mode="lines+markers+text",
                    text=[None, r["feature"]],
                    textposition="top center",
                    marker=dict(size=[0, 2]),
                    line=dict(width=1),
                    showlegend=False,
                    hovertext=r["feature"]
                ))

            fig.update_layout(
                title=f"Biplot — Year {cur_year} ({point_level})",
                scene=dict(xaxis_title=xlab, yaxis_title=ylab, zaxis_title=zlab),
                template="plotly_white",
                height=750
            )
            st.plotly_chart(fig, use_container_width=True)
