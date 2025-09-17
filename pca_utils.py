# pca_utils.py
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import os


# -----------------------------
# Preprocessing
# -----------------------------
def normalize_features(wide: pd.DataFrame, mode: str) -> pd.DataFrame:
    X = wide.copy()
    if mode == "Budget shares":
        rs = X.sum(axis=1)
        X = X.div(rs.where(rs != 0, 1), axis=0)
    elif mode == "Log(1+x)":
        X = np.log1p(X)
    elif mode == "Budget shares + Log":
        rs = X.sum(axis=1)
        X = X.div(rs.where(rs != 0, 1), axis=0)
        X = np.log1p(X)
    return X


def weighted_standardize(X: np.ndarray, w: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if w is None:
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=0)
    else:
        w = w.astype(float)
        w = w / np.sum(w)
        mu = np.average(X, axis=0, weights=w)
        xc = X - mu
        var = np.average(xc**2, axis=0, weights=w)
        sd = np.sqrt(np.maximum(var, 1e-16))
    sd_safe = np.where(sd == 0, 1.0, sd)
    Z = (X - mu) / sd_safe
    return Z, mu, sd_safe


# -----------------------------
# PCA (weighted by household weights)
# -----------------------------
def weighted_pca(X: np.ndarray, w: Optional[np.ndarray], n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Eigen-decompose weighted covariance. Returns (components p×k, eigenvalues k, scores n×k, mean p)."""
    n, p = X.shape
    if w is None:
        w = np.ones(n)
    w = w.astype(float)
    w = w / np.sum(w)

    mu = np.average(X, axis=0, weights=w)
    Xc = X - mu
    Xw = Xc * np.sqrt(w)[:, None]
    C  = Xw.T @ Xw
    eigvals, eigvecs = np.linalg.eigh(C)
    order   = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    k = min(n_components, p)
    comps = eigvecs[:, :k]
    scores = Xc @ comps
    return comps, eigvals[:k], scores[:, :k], mu


# -----------------------------
# Alignment (sign stability across years)
# -----------------------------
def align_signs_to_reference(V_ref: np.ndarray,
                             V_cur: np.ndarray,
                             scores_cur: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flip each PC j independently so that <v_cur[:,j], v_ref[:,j]> >= 0.
    (No rotation—just sign consistency across years.)
    """
    Vc = V_cur.copy()
    Sc = scores_cur.copy()
    r = min(V_ref.shape[1], V_cur.shape[1])
    for j in range(r):
        s = np.dot(V_ref[:, j], V_cur[:, j])
        if s < 0:
            Vc[:, j] *= -1.0
            Sc[:, j] *= -1.0
    return Vc, Sc


# -----------------------------
# Variance helpers
# -----------------------------
def variance_table(eigvals: np.ndarray) -> pd.DataFrame:
    lam = np.array(eigvals, dtype=float)
    total = lam.sum() if lam.size else 0.0
    evr = lam / total if total > 0 else np.zeros_like(lam)
    cum = np.cumsum(evr)
    return pd.DataFrame({
        "Component": [f"PC{i}" for i in range(1, len(lam)+1)],
        "Explained variance": lam,
        "Explained variance ratio": evr,
        "Cumulative EVR": cum
    })




def full_loadings_df(
    components: np.ndarray,      # shape: p x k
    explained_var: np.ndarray,   # length: k
    feature_names: List[str],
    scale: float = 1.0           # keep 1.0 for saving; scale only for plotting if you want
) -> pd.DataFrame:
    """
    Build a DataFrame with loadings for ALL PCs (PC1..PCk) using Gabriel scaling:
      loadings = components * sqrt(explained_var).
    """
    k = components.shape[1]
    # Gabriel scaling
    L = components * np.sqrt(explained_var[:k])[None, :]
    df = pd.DataFrame(L, columns=[f"PC{i}" for i in range(1, k + 1)])
    df.insert(0, "feature", feature_names)
    if scale != 1.0:
        pc_cols = [c for c in df.columns if c.startswith("PC")]
        df.loc[:, pc_cols] = df.loc[:, pc_cols] * scale
    return df

# -----------------------------
# Loadings for biplot
# -----------------------------
def biplot_loadings(components: np.ndarray,
                    explained_var: np.ndarray,
                    feature_names: List[str],
                    pc_idx: List[int],
                    scale: float = 1.0) -> pd.DataFrame:
    V = components  # p×k
    lam = explained_var
    take = [i for i in pc_idx if i < V.shape[1]]
    L = V[:, take] * np.sqrt(lam[take])
    cols = [f"PC{i+1}" for i in take]
    df = pd.DataFrame(L, columns=cols)
    df["feature"] = feature_names
    df[cols] = df[cols] * scale
    return df


# -----------------------------
# Saving results
# -----------------------------
def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_year_results(output_dir: str,
                      year: int,
                      scores_with_meta: pd.DataFrame,
                      loadings_df: pd.DataFrame,
                      var_df: pd.DataFrame,
                      as_excel: bool = True) -> Dict[str, str]:
    """Save one year’s results to disk. Returns dict of saved file paths."""
    ensure_dir(output_dir)
    out = {}

    if as_excel:
        fname = os.path.join(output_dir, f"pca_year_{year}.xlsx")
        with pd.ExcelWriter(fname, engine="xlsxwriter") as xw:
            scores_with_meta.to_excel(xw, sheet_name="scores", index=False)
            loadings_df.to_excel(xw, sheet_name="loadings", index=False)
            var_df.to_excel(xw, sheet_name="variance", index=False)
        out["excel"] = fname
    else:
        base = os.path.join(output_dir, f"pca_year_{year}")
        scores_with_meta.to_csv(base + "_scores.csv", index=False)
        loadings_df.to_csv(base + "_loadings.csv", index=False)
        var_df.to_csv(base + "_variance.csv", index=False)
        out["scores_csv"] = base + "_scores.csv"
        out["loadings_csv"] = base + "_loadings.csv"
        out["variance_csv"] = base + "_variance.csv"

    return out


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_cumulative_variance(var_df: pd.DataFrame, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(var_df)+1)),
        y=var_df["Cumulative EVR"],
        mode="lines+markers",
        name="Cumulative variance"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Number of PCs",
        yaxis_title="Cumulative explained variance ratio",
        template="plotly_white",
        height=420
    )
    return fig


def _customdata_and_template(df: pd.DataFrame, p1: int, p2: int):
    """
    Build customdata array and a matching hovertemplate.
    Uses df['point_name'] if present, plus Year if present.
    """
    cols = []
    if "point_name" in df.columns:
        cols.append("point_name")
    if "Year" in df.columns:
        cols.append("Year")

    customdata = df[cols].values if cols else None

    # Construct a hovertemplate that matches the number/order of cols
    pieces = []
    idx = 0
    if "point_name" in cols:
        pieces.append("<b>%{customdata[" + str(idx) + "]}</b>")
        idx += 1
    if "Year" in cols:
        pieces.append("Year=%{customdata[" + str(idx) + "]}")
        idx += 1

    pieces.append(f"PC{p1}=%{{x:.3f}}")
    pieces.append(f"PC{p2}=%{{y:.3f}}")
    hovertemplate = "<br>".join(pieces)

    return customdata, hovertemplate


def make_scores_scatter(
    scores_df: pd.DataFrame,
    pcs: Tuple[int, int],
    color_by: Optional[str] = None,
    title: str = ""
) -> go.Figure:
    p1, p2 = pcs
    xlab, ylab = f"PC{p1}", f"PC{p2}"

    fig = go.Figure()

    if color_by and color_by in scores_df.columns:
        for key, grp in scores_df.groupby(color_by, sort=False):
            customdata, hovertemplate = _customdata_and_template(grp, p1, p2)
            fig.add_trace(go.Scatter(
                x=grp[xlab],
                y=grp[ylab],
                mode="markers",
                name=str(key),
                marker=dict(size=5, opacity=0.75),
                customdata=customdata,
                hovertemplate=hovertemplate,
            ))
    else:
        customdata, hovertemplate = _customdata_and_template(scores_df, p1, p2)
        fig.add_trace(go.Scatter(
            x=scores_df[xlab],
            y=scores_df[ylab],
            mode="markers",
            marker=dict(size=5, opacity=0.75),
            customdata=customdata,
            hovertemplate=hovertemplate,
            name="Scores",
        ))

    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        template="plotly_white",
        height=600,
    )
    return fig


def make_biplot(
    scores_df: pd.DataFrame,
    load_df: pd.DataFrame,
    pcs: Tuple[int, int],
    title: str = ""
) -> go.Figure:
    p1, p2 = pcs
    xlab, ylab = f"PC{p1}", f"PC{p2}"

    # Scores layer (no color grouping here)
    fig = make_scores_scatter(scores_df, pcs, color_by=None, title=title)

    # Loadings as arrows with feature labels at the arrow tips
    for _, r in load_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[0, r.get(xlab, 0.0)],
            y=[0, r.get(ylab, 0.0)],
            mode="lines+markers+text",
            text=[None, r.get("feature", "")],
            textposition="top center",
            marker=dict(size=[0, 4]),
            line=dict(width=1),
            showlegend=False,
            hoverinfo="text",
        ))

    return fig