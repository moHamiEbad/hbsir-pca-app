# hbspca/importance.py
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ---------- low-level helpers ----------

def _numeric_loading_matrix(load_df: pd.DataFrame, pc_cols: List[str]) -> pd.DataFrame:
    """
    Return a feature×PC matrix with numeric values, indexed by 'feature'.
    """
    if "feature" not in load_df.columns:
        raise KeyError("Expected a 'feature' column in load_df.")
    cols = ["feature"] + pc_cols
    X = (
        load_df[cols]
        .dropna(subset=["feature"])
        .drop_duplicates(subset=["feature"])
        .set_index("feature")
    )
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def eigenvalues_from_var_df(var_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    Build { 'PC1': λ1, 'PC2': λ2, ... } from a variance table.
    Supports common column names.
    """
    if var_df is None:
        return {}
    df = pd.DataFrame(var_df)
    cand = None
    for c in ["Explained variance", "Explained Variance", "lambda", "eigenvalue", "Eigenvalue"]:
        if c in df.columns:
            cand = df[c].to_numpy(dtype=float)
            break
    if cand is None:
        return {}
    return {f"PC{i+1}": float(cand[i]) for i in range(len(cand))}


# ---------- math: communality / importance ----------

def feature_communality(
    load_df: pd.DataFrame,
    pc_cols: List[str],
    lam_by_pc: Optional[Dict[str, float]] = None,   # kept for API compatibility; unused
    normalize: bool = True,                          # kept; unused for correlation path
) -> pd.Series:
    """
    Importance per feature over selected PCs, using correlation loadings:
      importance_j = sum_k (ℓ_{jk})^2
    """
    vecs = _numeric_loading_matrix(load_df, pc_cols)   # feature×PC
    L = vecs.to_numpy()
    imp = (L ** 2).sum(axis=1)                          # <-- correlation loadings path only
    return pd.Series(imp, index=vecs.index, name="importance")


def importance_table(
    load_df: pd.DataFrame,
    pc_cols: List[str],
    lam_by_pc: Optional[Dict[str, float]] = None,  # kept; unused
    normalize: bool = True,                        # kept; unused
    include_labels: bool = True,
) -> pd.DataFrame:
    vecs = _numeric_loading_matrix(load_df, pc_cols)
    L = vecs.to_numpy()

    # importance via correlation-loadings-only path
    imp = feature_communality(load_df, pc_cols, lam_by_pc=lam_by_pc, normalize=normalize)

    # primary PC/sign
    max_abs_idx = np.argmax(np.abs(L), axis=1)
    primary_pc = [pc_cols[i] for i in max_abs_idx]
    prim_vals = L[np.arange(L.shape[0]), max_abs_idx]
    prim_abs = np.abs(prim_vals)
    prim_sign = np.where(prim_vals >= 0, "+", "−")

    out = vecs.reset_index()
    out["importance"] = imp.values
    out["primary_pc"] = primary_pc
    out["primary_abs_loading"] = prim_abs
    out["primary_sign"] = prim_sign

    if include_labels:
        lbls = [c for c in load_df.columns if str(c).startswith("label_")]
        if lbls:
            meta = load_df[["feature"] + lbls].drop_duplicates("feature")
            out = out.merge(meta, on="feature", how="left")

    # sort by importance desc then by primary_abs desc
    out = out.sort_values(["importance", "primary_abs_loading"], ascending=[False, False])
    return out


# ---------- Streamlit UI renderer (optional) ----------

def render_feature_importance(
    load_df_plot: pd.DataFrame,
    var_df: Optional[pd.DataFrame],
    scope_key: str = "all",
    default_first_k: int = 3,
) -> pd.DataFrame:
    """
    Streamlit UI for "PC-aggregated importance".
    Returns the ranked table (you can also ignore the return).
    """
    pc_cols_all = [c for c in load_df_plot.columns if str(c).startswith("PC")]
    if not pc_cols_all:
        st.info("No PC columns in loadings.")
        return pd.DataFrame()

    max_m = len(pc_cols_all)
    c1, c2 = st.columns([1, 1])

    with c1:
        subset_mode = st.radio(
            "PC subset", 
            ["First k", "Variance threshold", "Custom"],
            horizontal=True, 
            key=f"imp_subset_{scope_key}"
        )

    if subset_mode == "First k":
        with c2:
            k_use = st.slider(
                "k", 1, max_m,
                min(default_first_k, max_m),
                key=f"imp_k_{scope_key}"
            )
            sel_pc_cols = pc_cols_all[:k_use]
    elif subset_mode == "Variance threshold":
        with c2:
            var_threshold = st.number_input(
                "Variance to capture (%)",
                min_value=1,
                max_value=100,
                value=80,
                step=1,
                key=f"imp_var_pct_{scope_key}",
            ) / 100.0
            
            if var_df is not None and "Cumulative EVR" in var_df:
                cum_var = var_df["Cumulative EVR"].values
                k_var = np.searchsorted(cum_var, var_threshold) + 1
                k_use = min(max_m, max(1, k_var))
                sel_pc_cols = pc_cols_all[:k_use]
                st.caption(f"Using first {k_use} PCs")
            else:
                st.warning("Variance data not available")
                sel_pc_cols = pc_cols_all[:default_first_k]
    else:  # Custom
        with c2:
            sel_pc_cols = st.multiselect(
                "Choose PCs", 
                pc_cols_all,
                default=pc_cols_all[:min(default_first_k, max_m)],
                key=f"imp_custompcs_{scope_key}"
            )

    tbl = importance_table(
        load_df=load_df_plot,
        pc_cols=sel_pc_cols,
        normalize=True,
        include_labels=True,
    )

    topN = st.slider("Show top-N features", 5, max(5, len(tbl)), min(20, len(tbl)), key=f"imp_topn_{scope_key}")
    preview = tbl.head(topN)

    st.markdown("**Top features by importance (communality)**")
    st.dataframe(
        preview[["feature", "importance", "primary_pc", "primary_sign", "primary_abs_loading"] + sel_pc_cols +
                [c for c in tbl.columns if str(c).startswith("label_")]],
        use_container_width=True,
    )

    # quick bar chart (optional)
    try:
        import plotly.express as px
        fig = px.bar(preview.sort_values("importance", ascending=True),
                     x="importance", y="feature", orientation="h",
                     title="Importance over selected PCs")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    return tbl
