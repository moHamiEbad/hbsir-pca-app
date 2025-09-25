from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st


def _numeric_loading_matrix(load_df: pd.DataFrame, pc_cols: List[str]) -> pd.DataFrame:
    """Return a feature×PC matrix with numeric values, indexed by 'feature'."""
    if "feature" not in load_df.columns:
        raise KeyError("Expected a 'feature' column in load_df_plot.")
    cols = ["feature"] + pc_cols
    vecs = (
        load_df[cols]
        .dropna(subset=["feature"])
        .drop_duplicates(subset=["feature"])
        .set_index("feature")
    )
    vecs = vecs.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return vecs


def _top_k_similar(
    vecs: pd.DataFrame,
    picked_feature: str,
    metric: str,
    k: int,
    use_abs_corr: bool = True,
) -> pd.DataFrame:
    """Compute top-k neighbors for picked_feature in vecs using a metric."""
    if picked_feature not in vecs.index:
        raise KeyError(f"Feature '{picked_feature}' not found in loading matrix.")

    target = vecs.loc[picked_feature].values
    target_norm = np.linalg.norm(target)

    rows = []
    for feat, row in vecs.iterrows():
        if feat == picked_feature:
            continue
        v = row.values
        if metric == "Correlation":
            # guard zero-variance vectors
            if np.allclose(v, v.mean()) or np.allclose(target, target.mean()):
                continue
            r = float(np.corrcoef(target, v)[0, 1])
            rows.append((feat, abs(r) if use_abs_corr else r))
        elif metric == "Cosine similarity":
            denom = (np.linalg.norm(v) * target_norm)
            cs = 0.0 if denom == 0 else float(np.dot(target, v) / denom)
            rows.append((feat, cs))
        elif metric == "Euclidean distance":
            d = float(np.linalg.norm(target - v))
            rows.append((feat, d))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    out = pd.DataFrame(rows, columns=["feature", "score"])
    if out.empty:
        return out

    if metric in ("Correlation", "Cosine similarity"):
        out = out.sort_values("score", ascending=False).head(k)
    else:  # Euclidean distance
        out = out.sort_values("score", ascending=True).head(k)
    return out.reset_index(drop=True)


def render_feature_similarity_explorer(
    load_df_plot: pd.DataFrame,
    pcs: Tuple[int, int] | Tuple[int, int, int],
    scope_key: str = "all",
    var_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Streamlit UI for "Feature similarity explorer":
    - keeps selections stable across years if scope_key="all"
    - otherwise stores per-year using scope_key=f"y{year}"
    - uses variance info to suggest default PC count
    """

    pc_cols_all = [c for c in load_df_plot.columns if str(c).startswith("PC")]
    if not pc_cols_all:
        st.info("No loading columns (PCs) to compare yet.")
        return

    max_m = len(pc_cols_all)
    
    # Variance percentage selector - using number input instead of slider
    var_pct_key = f"sim_var_pct__{scope_key}"
    var_threshold = st.number_input(
        "Variance threshold (%)",
        min_value=1,
        max_value=100,
        value=80,  # Default 80%
        step=1,
        key=var_pct_key,
    ) / 100.0

    # Auto-PC selection based on variance
    auto_pc_key = f"sim_auto_pc__{scope_key}"
    use_auto_pc = st.checkbox(
        f"Automatically select number of PCs to capture {var_threshold:.0%} variance",
        value=st.session_state.get(auto_pc_key, True),
        key=auto_pc_key
    )

    default_m = max_m
    if use_auto_pc and var_df is not None and "Cumulative EVR" in var_df:
        # Find minimum PCs needed for user-selected variance threshold
        cum_var = var_df["Cumulative EVR"].values
        min_pcs = np.searchsorted(cum_var, var_threshold) + 1
        default_m = min(max_m, max(1, min_pcs))
    else:
        # Fall back to previous behavior
        m_default = max(pcs) if isinstance(pcs, tuple) else 2
        default_m = max(1, min(int(m_default), max_m))

    # Store the slider value in session state
    slider_key = f"sim_use_top_m__{scope_key}"
    if slider_key not in st.session_state or use_auto_pc:
        st.session_state[slider_key] = default_m

    use_top_m = st.slider(
        "Use top M PCs for similarity",
        min_value=1,
        max_value=max_m,
        value=st.session_state[slider_key],
        step=1,
        key=slider_key,
    )
    pc_cols = pc_cols_all[:use_top_m]

    # Add variance info to UI if available
    if var_df is not None and "Cumulative EVR" in var_df and use_top_m > 0:
        cum_var = var_df["Cumulative EVR"].values[use_top_m - 1]
        st.caption(f"Selected PCs capture {cum_var:.1%} of total variance")

    features_all = sorted(load_df_plot["feature"].astype(str).unique().tolist())
    if not features_all:
        st.info("No features available (after filters).")
        return

    # --- Sticky feature selection across reruns/years ---
    feat_key = f"sim_pick_feat__{scope_key}"
    prev_feat = st.session_state.get(feat_key)

    # If previous selection no longer exists in options, clear it
    if prev_feat is not None and prev_feat not in features_all:
        del st.session_state[feat_key]
        prev_feat = None

    # Compute default index
    if prev_feat is not None:
        default_index = features_all.index(prev_feat)
    else:
        default_index = 0  # safe fallback

    picked_feature = st.selectbox(
        "Pick a feature (lowest level)",
        options=features_all,
        index=default_index,
        key=feat_key,
    )

    k_max = max(1, min(50, max(0, len(features_all) - 1)))
    top_k = st.number_input(
        "Top K closest features",
        min_value=1,
        max_value=k_max,
        value=min(5, k_max),
        step=1,
        key=f"sim_topk__{scope_key}",
    )

    metric = st.selectbox(
        "Relation metric",
        options=["Correlation", "Cosine similarity", "Euclidean distance"],
        key=f"sim_metric__{scope_key}",
    )
    use_abs_corr = False
    if metric == "Correlation":
        use_abs_corr = st.checkbox(
            "Use absolute correlation (treat +/- same)",
            value=True,
            key=f"sim_abs__{scope_key}",
        )

    vecs = _numeric_loading_matrix(load_df_plot, pc_cols)

    try:
        neighbors = _top_k_similar(
            vecs=vecs,
            picked_feature=picked_feature,
            metric=metric,
            k=int(top_k),
            use_abs_corr=use_abs_corr,
        )
    except KeyError as e:
        st.warning(str(e))
        return

    if neighbors.empty:
        st.info("No comparable features found.")
        return

    label_cols = [c for c in load_df_plot.columns if str(c).startswith("label_")]
    if label_cols:
        uniq_meta = load_df_plot[["feature"] + label_cols].drop_duplicates("feature")
        neighbors = neighbors.merge(uniq_meta, on="feature", how="left")

    display_metric = {
        "Correlation": "Correlation (higher=closer)",
        "Cosine similarity": "Cosine similarity (higher=closer)",
        "Euclidean distance": "Euclidean distance (lower=closer)",
    }[metric]
    st.markdown(
        f"**Closest features to `{picked_feature}`**  \n"
        f"*Metric:* {display_metric}  \n"
        f"*PCs used:* {len(pc_cols)}"
    )
    st.dataframe(neighbors, use_container_width=True)
