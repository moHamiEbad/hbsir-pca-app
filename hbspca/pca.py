from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import streamlit as st

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

def weighted_pca(X: np.ndarray, w: Optional[np.ndarray], n_components: int):
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

def align_signs_to_reference(V_ref: np.ndarray, V_cur: np.ndarray, scores_cur: np.ndarray, feat_names_ref: List[str] = None, feat_names_cur: List[str] = None):
    """Align signs of current components to reference components, handling different feature sets."""
    Vc = V_cur.copy()
    Sc = scores_cur.copy()

    # If no feature names provided, try to handle different shapes safely
    if feat_names_ref is None or feat_names_cur is None:
        # Get common dimension (minimum number of features)
        min_feats = min(V_ref.shape[0], V_cur.shape[0])
        # Get common number of components
        r = min(V_ref.shape[1], V_cur.shape[1])
        # Use only the common features for alignment
        for j in range(r):
            s = np.dot(V_ref[:min_feats, j], V_cur[:min_feats, j])
            if s < 0:
                Vc[:, j] *= -1.0
                Sc[:, j] *= -1.0
        return Vc, Sc

    # Create common feature index mapping
    common_feats = list(set(feat_names_ref) & set(feat_names_cur))
    if not common_feats:
        # If no common features, return unmodified
        return Vc, Sc
        
    # Get indices for common features in both matrices
    idx_ref = [feat_names_ref.index(f) for f in common_feats]
    idx_cur = [feat_names_cur.index(f) for f in common_feats]
    
    # Extract common features only
    V_ref_common = V_ref[idx_ref, :]
    V_cur_common = Vc[idx_cur, :]

    # Align using only common features
    r = min(V_ref_common.shape[1], V_cur_common.shape[1])
    for j in range(r):
        s = np.dot(V_ref_common[:, j], V_cur_common[:, j])
        if s < 0:
            Vc[:, j] *= -1.0
            Sc[:, j] *= -1.0

    return Vc, Sc

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

def full_loadings_df(components: np.ndarray, explained_var: np.ndarray, feature_names: List[str], scale: float = 1.0) -> pd.DataFrame:
    k = components.shape[1]
    L = components * np.sqrt(explained_var[:k])[None, :]
    df = pd.DataFrame(L, columns=[f"PC{i}" for i in range(1, k + 1)])
    df.insert(0, "feature", feature_names)
    if scale != 1.0:
        pc_cols = [c for c in df.columns if c.startswith("PC")]
        df.loc[:, pc_cols] = df.loc[:, pc_cols] * scale
    return df
