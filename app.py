import os
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils import notify_done
from hbspca.ui import sidebar
from hbspca.io import setup_hbsir, load_households, load_expenditures, merge_exp_hh, filter_area_settlement, guess_weight_col
from hbspca.features import build_household_matrix_by_level, feature_meta_for_year
from hbspca.pca import weighted_standardize, weighted_pca, align_signs_to_reference, variance_table, full_loadings_df
from hbspca.plotting import plot_cumulative_variance, make_scores_scatter, make_biplot, make_scores_scatter3d, make_biplot3d, make_loadings_plot
from hbspca.saving import ensure_dir, save_year_results
from hbspca.types import YearResult
from hbspca.similarity import render_feature_similarity_explorer
from hbspca.importance import render_feature_importance
from hbspca.id_filter import apply_household_id_filter

st.set_page_config(page_title="HBSIR PCA — Household level", layout="wide")
st.title("HBSIR PCA — Household level workflow")
st.caption("PCA runs on households; only area/settlement filter applies. Per-year results are saved, then plotted with flexible aggregation.")

# --- Session state ---
if "pca_results" not in st.session_state:
    st.session_state["pca_results"] = {}
if "ref_components" not in st.session_state:
    st.session_state["ref_components"] = None
if "years_run" not in st.session_state:
    st.session_state["years_run"] = []
if "hh_menus" not in st.session_state:
    st.session_state["hh_menus"] = None  # province/county menus, loaded on-demand

# --- Sidebar (no downloads here) ---
years_full = list(range(1379, 1403))  # just for the year selector choices
hh_for_menus = st.session_state["hh_menus"]  # None on first render
state = sidebar(years_full, hh_for_menus)    # must accept None


# --- If user has saved files, let them load them now (no downloads, no PCA) ---
if state.use_existing and state.load_dir and os.path.isdir(state.load_dir):
# --- If user has saved files, let them load them now (no downloads, no PCA) ---
    if state.use_existing and state.load_dir and os.path.isdir(state.load_dir):
        if st.button("Load saved results"):
            from hbspca.saving import load_saved_results
            loaded = load_saved_results(state.load_dir)
            if not loaded:
                st.warning("No saved files found in that folder (expected: pca_year_YYYY.xlsx).")
            else:
                # Put into session the same way the PCA path does
                st.session_state.pca_results = {yr: obj.__dict__ for yr, obj in loaded.items()}
                st.session_state.ref_components = None
                st.session_state.years_run = sorted(loaded.keys())
                st.session_state.plot_year_idx = 0   # ✅ reset index here

                st.success(f"Loaded results for years: {', '.join(map(str, st.session_state.years_run))}")
                # Make sure plotting sees the new data immediately
                notify_done(sound=True, toast=True, desktop_note=True)
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()




# --- Optional: load area menus ONLY if user asks and ONLY for selected years ---
if state.area_kind == "Subareas" and st.session_state["hh_menus"] is None:
    if st.button("Load area menus for selected years"):
        with st.spinner("Preparing area menus…"):
            # Only prepare the chosen years
            setup_hbsir(state.years)
            st.session_state["hh_menus"] = load_households(state.years)
            notify_done(sound=True, toast=True, desktop_note=True)
        st.rerun()

# --- Run PCA (this is the first time we touch the data engines) ---
run_btn = st.sidebar.button("Run PCA", type="primary")
if run_btn:
    # Reset results
    st.session_state.pca_results.clear()
    st.session_state.ref_components = None
    st.session_state.years_run = list(state.years)

    if not state.save_dir:
        st.warning("No output directory specified. Files will still be offered as downloads after each year.")

    if state.save_dir:
        ensure_dir(state.save_dir)

    with st.spinner("Running per-year PCA at household level…"):
        # Prepare ONLY selected years (download/cache if needed)
        setup_hbsir(state.years)

        # Load ONLY selected years
        hh  = load_households(state.years)
        exp = load_expenditures(state.years)
        exp_hh_all = merge_exp_hh(exp, hh)

        # Apply filters (province/county/settlement) to restrict data BEFORE PCA
        level_choice_for_filter = "Province" if state.area_kind == "Subareas" else "Country"
        exp_hh_view = filter_area_settlement(
            df=exp_hh_all, years=state.years,
            level_choice=level_choice_for_filter,
            provinces=state.selected_provinces, counties=state.selected_counties,
            settlement=state.settlement
        )

        if getattr(state, "id_filter_df", None) is not None and not state.id_filter_df.empty:
            # Apply optional ID filter (after area/settlement filter, before PCA)
            if state.id_filter_df is not None and not state.id_filter_df.empty:
                exp_hh_view = apply_household_id_filter(
                    df=exp_hh_view,
                    ids_df=state.id_filter_df,
                    ignore_year=state.ignore_year_for_ids,  # True => match on ID only
                )
                if exp_hh_view.empty:
                    st.error("پس از اعمال فیلتر ID، دیتایی باقی نماند.")
                    st.stop()

        weight_col = guess_weight_col(exp_hh_view)

        for idx, yr in enumerate(sorted(state.years)):
            # Build per-year household × features matrix
            try:
                wide_y, meta_y, _ = build_household_matrix_by_level(exp_hh_view, [yr], state.level, state.exp_measure)
            except MemoryError:
                st.error(f"Year {yr}: ran out of memory while building the matrix. "
                        "Try a lower classification level, fewer areas, or a different normalization.")
                continue
            if wide_y.empty:
                st.warning(f"Year {yr}: no data after filters. Skipped.")
                continue

            # --- weights ---
            w = None
            if state.weighted and weight_col is not None:
                hh_w = (exp_hh_view.drop_duplicates(subset=["Year", "ID"])[["Year", "ID", weight_col]]
                        .set_index(["Year", "ID"]))

                import numpy as np
                w = hh_w.reindex(wide_y.index)[weight_col].fillna(0).values.astype(float)
                w = np.where(w <= 0, 1e-12, w)

            # --- preprocessing -> Z ---
            norm_mode = state.norm_mode
            import numpy as np
            if norm_mode == "Raw levels":
                Z, _, _ = weighted_standardize(wide_y.values, w if state.weighted else None)

            elif norm_mode == "Budget shares (center-only)":
                row_sums = wide_y.sum(axis=1).where(wide_y.sum(axis=1) != 0, 1)
                S = wide_y.div(row_sums, axis=0)
                if state.weighted and w is not None and np.sum(w) > 0:
                    ww = w / np.sum(w)
                    mu = np.average(S.values, axis=0, weights=ww)
                else:
                    mu = np.nanmean(S.values, axis=0)
                Z = S.values - mu

            elif norm_mode == "Budget shares (z-score columns)":
                row_sums = wide_y.sum(axis=1).where(wide_y.sum(axis=1) != 0, 1)
                S = wide_y.div(row_sums, axis=0)
                Z, _, _ = weighted_standardize(S.values, w if state.weighted else None)

            elif norm_mode == "CLR (compositional)":
                row_sums = wide_y.sum(axis=1).where(wide_y.sum(axis=1) != 0, 1)
                S = wide_y.div(row_sums, axis=0).clip(lower=1e-6)
                logS = np.log(S)
                clr = logS.sub(logS.mean(axis=1), axis=0)
                Z, _, _ = weighted_standardize(clr.values, w if state.weighted else None)

            elif norm_mode == "Log(1+x)":
                L = np.log1p(wide_y)
                Z, _, _ = weighted_standardize(L.values, w if state.weighted else None)

            elif norm_mode == "Budget shares + Log":
                row_sums = wide_y.sum(axis=1).where(wide_y.sum(axis=1) != 0, 1)
                S = wide_y.div(row_sums, axis=0).clip(lower=1e-6)
                logS = np.log(S)
                Z, _, _ = weighted_standardize(logS.values, w if state.weighted else None)

            else:
                Z, _, _ = weighted_standardize(wide_y.values, w if state.weighted else None)

            # --- PCA ---
            k = Z.shape[1]
            comps, lam, scores, _ = weighted_pca(Z, w if state.weighted else None, n_components=k)
            if st.session_state.ref_components is None:
                st.session_state.ref_components = comps.copy()
                st.session_state.ref_features = wide_y.columns.tolist()
            else:
                comps, scores = align_signs_to_reference(
                    st.session_state.ref_components, comps, scores,
                    st.session_state.ref_features, wide_y.columns.tolist()
                )

            # --- Outputs & saving ---
            import pandas as pd
            scores_df = pd.DataFrame(scores, index=wide_y.index,
                                     columns=[f"PC{i}" for i in range(1, scores.shape[1] + 1)]).reset_index()
            scores_with_meta = (scores_df.merge(meta_y.reset_index(), on=["Year", "ID"], how="left")
                                          .sort_values(["Year", "ID"]))
            var_df = variance_table(lam)

            feature_names = wide_y.columns.astype(str).tolist()
            load_df_all = full_loadings_df(components=comps, explained_var=lam,
                                           feature_names=feature_names, scale=1.0)

            # Use the UNFILTERED pool for taxonomy coverage so all parent labels exist
            df_year_meta_source = exp_hh_all[exp_hh_all["Year"] == yr]
            feature_meta = feature_meta_for_year(
                df_year_meta_source, state.level, feature_names
            ).drop_duplicates("feature")

            parent_cols = [c for c in feature_meta.columns if c.startswith("label_")]
            load_df_all = load_df_all.merge(feature_meta[["feature"] + parent_cols], on="feature", how="left")

            saved = {}
            try:
                if state.save_dir:
                    saved = save_year_results(state.save_dir, yr, scores_with_meta, load_df_all, var_df,
                                              as_excel=state.save_as_excel)
            except Exception as e:
                st.error(f"Year {yr}: failed to save files: {e}")

            from hbspca.types import YearResult
            st.session_state.pca_results[yr] = YearResult(
                X_cols=feature_names, comps=comps, lam=lam,
                scores_df=scores_df, scores_with_meta=scores_with_meta,
                load_df_all=load_df_all, var_df=var_df, level=state.level,
                feature_meta=feature_meta, saved=saved
            ).__dict__

        st.success("PCA finished. See the **Plotting** section below.")
        st.session_state.plot_year_idx = 0 
        notify_done(sound=True, toast=True, desktop_note=True)


# Plotting
st.markdown("---")
st.header("Plotting")
if not st.session_state.pca_results:
    st.info("Run PCA above or load existing files to enable plotting.")
    st.stop()

years_ready = sorted(st.session_state.pca_results.keys())

# 👇 Clamp plot_year_idx to a valid range, then define cur_year once
idx = st.session_state.get("plot_year_idx", 0)
if not years_ready:
    st.info("No results to plot. Run PCA or load saved results.")
    st.stop()
idx = max(0, min(idx, len(years_ready) - 1))
st.session_state.plot_year_idx = idx
cur_year = years_ready[idx]


# Master toggle: one set of controls shared across years
apply_all = st.checkbox(
    "Apply selections to all years",
    value=st.session_state.get("apply_all_years", True),
    key="apply_all_years",
)

def k(name: str) -> str:
    """Stable state keys: global when apply_all=True, per-year otherwise."""
    return f"plot_all__{name}" if apply_all else f"plot_y{cur_year}__{name}"

# Plot type
_plot_kind_default = st.session_state.get(k("plot_kind"), "Scores")
plot_kind = st.radio(
    "Plot type", ["Scores", "Loadings", "Biplot"], horizontal=True,
    index=["Scores","Loadings","Biplot"].index(_plot_kind_default),
    key=k("plot_kind"),
)

# Loadings scale (only used for Biplot)
if plot_kind == "Biplot":
    loading_scale_default = st.session_state.get(k("loading_scale"), 1.5)
    loading_scale = st.slider("Loadings scale", 0.2, 100.0, loading_scale_default, 0.1, key=k("loading_scale"))
else:
    loading_scale = 1.0

point_level = None
color_by = None
filter_provs = []
filter_counties = []
point_opacity = 0.30
randomize_order = False

if plot_kind in ("Scores", "Biplot"):
    st.subheader("Scores settings")
    _plevel_default = st.session_state.get(k("point_level"), "Household")
    point_level = st.radio("Each point represents a…", ["Household", "County", "Province"],
                           horizontal=True, index=["Household","County","Province"].index(_plevel_default),
                           key=k("point_level"))

    sample_any = next(iter(st.session_state.pca_results.values()))["scores_with_meta"]

    if point_level == "Household":
        show_subset = st.checkbox("Filter households by Province/County",
                                  value=st.session_state.get(k("hh_subset"), False), key=k("hh_subset"))
        if show_subset:
            provs = sorted(sample_any.get("Province_farsi_name", pd.Series([], dtype=str)).dropna().unique().tolist()) \
                    if "Province_farsi_name" in sample_any.columns else []
            # intersect stored with current options
            default_provs = [p for p in st.session_state.get(k("filter_provs"), provs) if p in provs]
            filter_provs = st.multiselect("Provinces", provs, default=default_provs, key=k("filter_provs"))

            cnts = []
            if filter_provs and "Province_farsi_name" in sample_any.columns and "County_farsi_name" in sample_any.columns:
                cnts = sorted(sample_any[sample_any["Province_farsi_name"].isin(filter_provs)]
                              ["County_farsi_name"].dropna().unique().tolist())
            default_cnts = [c for c in st.session_state.get(k("filter_counties"), cnts) if c in cnts]
            filter_counties = st.multiselect("Counties (optional)", cnts, default=default_cnts, key=k("filter_counties"))

        _color_default = st.session_state.get(k("color_by"), "Settlement")
        color_by = st.selectbox("Color households by", ["None", "Settlement", "Province", "County"],
                                index=["None","Settlement","Province","County"].index(_color_default),
                                key=k("color_by"))
        if color_by == "None":
            color_by = None

        opacity_default = st.session_state.get(k("point_opacity"), 0.30)
        point_opacity = st.slider("Point opacity", 0.05, 1.0, opacity_default, 0.05, key=k("point_opacity"))

        randomize_order = st.checkbox("Randomize category draw order",
                                      value=st.session_state.get(k("rand_order"), True),
                                      key=k("rand_order"))

    elif point_level == "Province":
        provs = sorted(sample_any.get("Province_farsi_name", pd.Series([], dtype=str)).dropna().unique().tolist()) \
                if "Province_farsi_name" in sample_any.columns else []
        choose = st.checkbox("Show only a subset of provinces",
                             value=st.session_state.get(k("prov_subset"), False), key=k("prov_subset"))
        default_provs = [p for p in st.session_state.get(k("filter_provs"), provs) if p in provs] if choose else provs
        filter_provs = st.multiselect("Provinces", provs, default=default_provs, key=k("filter_provs"))
        color_by = None

    else:  # County
        cnts = sorted(sample_any.get("County_farsi_name", pd.Series([], dtype=str)).dropna().unique().tolist()) \
               if "County_farsi_name" in sample_any.columns else []
        choose = st.checkbox("Show only a subset of counties",
                             value=st.session_state.get(k("county_subset"), False), key=k("county_subset"))
        default_cnts = [c for c in st.session_state.get(k("filter_counties"), cnts) if c in cnts] if choose else cnts
        filter_counties = st.multiselect("Counties", cnts, default=default_cnts, key=k("filter_counties"))
        color_by = None

# Loadings filter
loading_filter = None
if plot_kind in ("Loadings", "Biplot"):
    st.subheader("Loadings filter")

    res_cur = st.session_state.pca_results[cur_year]
    feat_meta = res_cur.get("feature_meta")

    # Prefer saved depth; fall back to app selection
    saved_level = res_cur.get("level")
    inferred_level = (len([c for c in (pd.DataFrame(feat_meta).columns if feat_meta is not None else [])
                           if str(c).startswith("label_")]) + 1) if feat_meta is not None else None
    max_level = int(saved_level or inferred_level or state.level or 1)

    base_key = "plot_all" if apply_all else f"plot_y{cur_year}"

    def _get_state(key, default):
        return st.session_state.get(key, default)

    if feat_meta is None or len(feat_meta) == 0:
        # No hierarchy -> flat feature picker
        features_all = sorted(pd.DataFrame(res_cur["load_df_all"])["feature"].astype(str).unique().tolist())
        default_feats = _get_state(f"{base_key}_features_lvl{max_level}", features_all)
        # Intersect defaults with what's available this year
        default_feats = [f for f in default_feats if f in features_all]
        loading_filter = st.multiselect(
            "Features (filtered by parents)",
            options=features_all,
            default=default_feats,
            key=f"{base_key}_features_lvl{max_level}",
        )
    else:
        df_cur = pd.DataFrame(feat_meta).copy()
        # Ensure label_1..label_{max_level-1} exist as strings
        for lev in range(1, max_level):
            col = f"label_{lev}"
            if col not in df_cur.columns:
                df_cur[col] = "(unknown)"
            df_cur[col] = df_cur[col].fillna("(unknown)").astype(str)

        # Parent pickers: Level 1..(max_level-1)
        for lev in range(1, max_level):
            col = f"label_{lev}"
            opts = sorted(df_cur[col].dropna().astype(str).unique().tolist())

            # Use stored selection if present; else default to all options
            stored = _get_state(f"{base_key}_l{lev}_lvl{max_level}", opts)
            # Intersect stored with current options (in case year differs)
            stored = [o for o in stored if o in opts]

            picked = st.multiselect(
                f"Level {lev} groups",
                options=opts,
                default=stored if stored else opts,
                key=f"{base_key}_l{lev}_lvl{max_level}",
            )
            if picked:
                df_cur = df_cur[df_cur[col].isin(picked)]

        # Leaf features after parent filters
        features_all = sorted(df_cur["feature"].astype(str).unique().tolist())
        stored_feats = _get_state(f"{base_key}_features_lvl{max_level}", features_all)
        stored_feats = [f for f in stored_feats if f in features_all]

        loading_filter = st.multiselect(
            "Features (filtered by parents)",
            options=features_all,
            default=stored_feats if stored_feats else features_all,
            key=f"{base_key}_features_lvl{max_level}",
        )


# Build data for current year
res = st.session_state.pca_results[cur_year]

# Collapsible cumulative-variance section
if "var_df" in res and res["var_df"] is not None and len(res["var_df"]) > 0:
    with st.expander("Cumulative variance", expanded=False):
        fig_var = plot_cumulative_variance(
            pd.DataFrame(res["var_df"]),
            title=f"Cumulative variance — Year {cur_year}"
        )
        st.plotly_chart(fig_var, use_container_width=True)
        st.dataframe(pd.DataFrame(res["var_df"]), use_container_width=True)
else:
    st.info("No variance data for this year. Run PCA or load saved results.")


scores_with_meta = res["scores_with_meta"].copy()
pc_cols = [f"PC{i}" for i in range(1, len([c for c in res["scores_df"].columns if c.startswith("PC")]) + 1)]

scores_plot = scores_with_meta[["Year", "ID"] + pc_cols].copy()
if "Province_farsi_name" in scores_with_meta.columns:
    scores_plot["Province"] = scores_with_meta["Province_farsi_name"]
if "County_farsi_name" in scores_with_meta.columns:
    scores_plot["County"] = scores_with_meta["County_farsi_name"]
urc = "Urban_Rural_name" if "Urban_Rural_name" in scores_with_meta.columns else ("Urban_Rural" if "Urban_Rural" in scores_with_meta.columns else None)
if urc:
    scores_plot["Settlement"] = scores_with_meta[urc]

if plot_kind in ("Scores", "Biplot"):
    if point_level == "Household":
        if filter_provs:
            scores_plot = scores_plot[scores_plot.get("Province").isin(filter_provs)]
        if filter_counties:
            scores_plot = scores_plot[scores_plot.get("County").isin(filter_counties)]
    elif point_level == "Province":
        if filter_provs:
            scores_plot = scores_plot[scores_plot.get("Province").isin(filter_provs)]
        keep = ["Year", "Province"] + pc_cols
        scores_plot = scores_plot[keep].groupby(["Year", "Province"], as_index=False)[pc_cols].mean()
        color_by = None
    else:
        if filter_counties:
            scores_plot = scores_plot[scores_plot.get("County").isin(filter_counties)]
        keep = ["Year", "County"] + pc_cols
        scores_plot = scores_plot[keep].groupby(["Year", "County"], as_index=False)[pc_cols].mean()
        color_by = None

if point_level == "Household":
    scores_plot["point_name"] = scores_plot["ID"].astype(str)
elif point_level == "Province":
    scores_plot["point_name"] = scores_plot["Province"].fillna("(unnamed)").astype(str)
else:
    scores_plot["point_name"] = scores_plot["County"].fillna("(unnamed)").astype(str)

if color_by and color_by in scores_plot.columns:
    scores_plot[color_by] = scores_plot[color_by].where(scores_plot[color_by].notna(), "(missing)")

load_df_plot = pd.DataFrame(res["load_df_all"]).copy()
if plot_kind in ("Loadings", "Biplot") and loading_filter:
    load_df_plot = load_df_plot[load_df_plot["feature"].astype(str).isin(loading_filter)].copy()
if plot_kind == "Biplot" and loading_scale != 1.0:
    for c in load_df_plot.columns:
        if c.startswith("PC"):
            load_df_plot[c] = load_df_plot[c] * loading_scale

# Draw
max_pc_year = len([c for c in res["scores_df"].columns if c.startswith("PC")])

plot_dim = st.radio("Plot dimensionality", ["2D", "3D"], horizontal=True, key="plot_dim_inline")

pc1 = st.number_input("PC for X-axis", 1, max_pc_year, value=st.session_state.get("pc1_inline", 1),
                      step=1, key="pc1_inline")
pc2 = st.number_input("PC for Y-axis", 1, max_pc_year, value=st.session_state.get("pc2_inline", 2),
                      step=1, key="pc2_inline")
pcs = (int(pc1), int(pc2))
if plot_dim == "3D":
    pc3 = st.number_input("PC for Z-axis", 1, max_pc_year, value=st.session_state.get("pc3_inline", 3),
                          step=1, key="pc3_inline")
    pcs = (int(pc1), int(pc2), int(pc3))

if plot_kind == "Scores":
    if plot_dim == "2D":
        fig = make_scores_scatter(scores_plot, pcs=(pcs[0], pcs[1]), color_by=color_by,
                                  title=f"Scores — Year {cur_year} ({point_level})",
                                  opacity=point_opacity, randomize_trace_order=randomize_order)
        st.plotly_chart(fig, use_container_width=True)
    else:
        xlab, ylab, zlab = f"PC{pcs[0]}", f"PC{pcs[1]}", f"PC{pcs[2]}"
        missing = [c for c in (xlab, ylab, zlab) if c not in scores_plot.columns]
        if missing:
            st.warning(f"Selected PCs not available for {cur_year}: {missing}")
        else:
            fig = go.Figure()
            df3 = scores_plot.copy()
            if color_by and color_by in df3.columns:
                groups = list(df3.groupby(color_by, dropna=False))
                if randomize_order and len(groups) > 1:
                    idx = np.random.permutation(len(groups))
                    groups = [groups[i] for i in idx]
                for key, grp in groups:
                    fig.add_trace(go.Scatter3d(x=grp[xlab], y=grp[ylab], z=grp[zlab],
                                               mode="markers", name=str(key),
                                               marker=dict(size=3, opacity=point_opacity),
                                               text=grp.get("point_name"),
                                               hovertemplate="<b>%{text}</b><br>"
                                                             f"{xlab}=%{{x:.3f}}<br>"
                                                             f"{ylab}=%{{y:.3f}}<br>"
                                                             f"{zlab}=%{{z:.3f}}"
                                                             "<extra>%{fullData.name}</extra>"))
            else:
                fig.add_trace(go.Scatter3d(x=df3[xlab], y=df3[ylab], z=df3[zlab],
                                           mode="markers",
                                           marker=dict(size=3, opacity=point_opacity),
                                           text=df3.get("point_name"),
                                           hovertemplate="<b>%{text}</b><br>"
                                                         f"{xlab}=%{{x:.3f}}<br>"
                                                         f"{ylab}=%{{y:.3f}}<br>"
                                                         f"{zlab}=%{{z:.3f}}"
                                                         "<extra></extra>"))
            fig.update_layout(title=f"Scores — Year {cur_year} ({point_level})",
                              scene=dict(xaxis_title=xlab, yaxis_title=ylab, zaxis_title=zlab),
                              template="plotly_white", height=700)
            st.plotly_chart(fig, use_container_width=True)

elif plot_kind == "Loadings":
    # --- Build/trim loadings for the current year ---
    load_df_plot = pd.DataFrame(res["load_df_all"]).copy()

    # Apply the user's feature subset (if any)
    if loading_filter:
        load_df_plot = load_df_plot[
            load_df_plot["feature"].astype(str).isin(loading_filter)
        ].copy()

    # Merge parent labels for the table (optional, helpful for inspection)
    feat_meta = res.get("feature_meta")
    if feat_meta is not None and len(feat_meta):
        parent_cols = [c for c in feat_meta.columns if str(c).startswith("label_")]
        if parent_cols:
            load_df_plot = load_df_plot.merge(
                pd.DataFrame(feat_meta)[["feature"] + parent_cols],
                on="feature",
                how="left",
            )

    # --- Loadings plot (exactly above "Feature similarity explorer") ---
    try:
        pcs_tuple = (pcs[0], pcs[1]) if plot_dim == "2D" else (pcs[0], pcs[1], pcs[2])
        fig = make_loadings_plot(
            load_df_plot,
            pcs=pcs_tuple,
            title=f"Loadings — Year {cur_year}",
        )
        st.plotly_chart(fig, use_container_width=True)
    except KeyError as e:
        st.warning(f"Cannot draw loadings plot: {e}")

    with st.expander("Feature importance across PCs", expanded=True):
    # st.subheader("Feature importance across PCs")
        _ = render_feature_importance(
            load_df_plot=load_df_plot,
            var_df=pd.DataFrame(res.get("var_df")) if res.get("var_df") is not None else None,
            scope_key="all",   # or f"y{cur_year}" if you want per-year sticky controls
        )

    # --- Feature similarity explorer (moved to hbspca/similarity.py) ---
    # Use a per-year key prefix so selections are stable when stepping years.
    with st.expander("Feature similarity explorer", expanded=True):
        st.subheader("Feature similarity explorer")
        # Get variance info from current year
        var_df = pd.DataFrame(res["var_df"]) if "var_df" in res else None
        render_feature_similarity_explorer(
            load_df_plot=load_df_plot,
            pcs=pcs_tuple,
            scope_key="all",   
            var_df=var_df  # Add this parameter
        )


else:  # Biplot
    if plot_dim == "2D":
        fig = make_biplot(
            scores_plot, load_df_plot, pcs=(pcs[0], pcs[1]),
            title=f"Biplot — Year {cur_year} ({point_level})",
            color_by=color_by, opacity=point_opacity, randomize_trace_order=randomize_order
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        xlab, ylab, zlab = f"PC{pcs[0]}", f"PC{pcs[1]}", f"PC{pcs[2]}"
        missing_s = [c for c in (xlab, ylab, zlab) if c not in scores_plot.columns]
        missing_l = [c for c in (xlab, ylab, zlab) if c not in load_df_plot.columns]
        if missing_s or missing_l:
            st.warning(
                f"Selected PCs not available for {cur_year}. "
                f"Scores missing: {missing_s}; Loadings missing: {missing_l}"
            )
        else:
            # ✅ Single 3D biplot (includes colored scores + loadings)
            fig = make_biplot3d(
                scores_plot, load_df_plot, pcs=(pcs[0], pcs[1], pcs[2]),
                title=f"Biplot — Year {cur_year} ({point_level})",
                color_by=color_by, opacity=point_opacity, randomize_trace_order=randomize_order,
            )
            st.plotly_chart(fig, use_container_width=True)


# === Year stepper (exactly below the plot) ===
if plot_kind in ("Scores", "Loadings", "Biplot"):
    c_prev, c_year, c_next = st.columns([1, 3, 1])
    with c_prev:
        if st.button("◀ Prev", key="plot_prev_year_inline"):
            st.session_state.plot_year_idx = max(0, st.session_state.plot_year_idx - 1)
            st.rerun()  # <- force a clean rerun right away
    with c_next:
        if st.button("Next ▶", key="plot_next_year_inline"):
            st.session_state.plot_year_idx = min(len(years_ready) - 1, st.session_state.plot_year_idx + 1)
            st.rerun()  # <- force a clean rerun right away

    # re-clamp & display
    idx = max(0, min(st.session_state.plot_year_idx, len(years_ready) - 1))
    st.session_state.plot_year_idx = idx
    cur_year = years_ready[idx]
    c_year.markdown(f"**Showing year:** {cur_year}")
