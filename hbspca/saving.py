from __future__ import annotations
from typing import Dict
import os
import pandas as pd

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_year_results(output_dir: str, year: int,
                      scores_with_meta: pd.DataFrame,
                      loadings_df: pd.DataFrame,
                      var_df: pd.DataFrame,
                      as_excel: bool = True) -> Dict[str, str]:
    ensure_dir(output_dir)
    out: Dict[str, str] = {}
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


def load_saved_results(input_dir: str) -> Dict[int, "YearResult"]:
    """
    Load per-year results saved by `save_year_results` (pca_year_YYYY.xlsx).
    Returns {year: YearResult}. Components are not stored in Excel, so `comps=None`.
    """
    import glob
    import re
    import pandas as pd
    from .types import YearResult

    results: Dict[int, YearResult] = {}
    pattern = os.path.join(input_dir, "pca_year_*.xlsx")
    for path in glob.glob(pattern):
        m = re.search(r"pca_year_(\d{3,4})\.xlsx$", os.path.basename(path))
        if not m:
            continue
        year = int(m.group(1))
        try:
            scores_with_meta = pd.read_excel(path, sheet_name="scores")
            load_df_all      = pd.read_excel(path, sheet_name="loadings")
            var_df           = pd.read_excel(path, sheet_name="variance")
        except Exception:
            continue

        # Build minimal fields
        pc_cols = [c for c in scores_with_meta.columns if str(c).startswith("PC")]
        scores_df = scores_with_meta[["Year", "ID"] + pc_cols].copy()

        lam = var_df.get("Explained variance", pd.Series([], dtype=float)).to_numpy()
        X_cols = load_df_all.get("feature", pd.Series([], dtype=str)).astype(str).tolist()

        # Try to recover a feature_meta subset from loadings (the label_* columns)
        feature_meta_cols = ["feature"] + [c for c in load_df_all.columns if str(c).startswith("label_")]
        feature_meta = (load_df_all[feature_meta_cols].drop_duplicates("feature")
                        if set(feature_meta_cols).issubset(load_df_all.columns) else None)

        results[year] = YearResult(
            X_cols=X_cols,
            comps=None,               # not stored; fine for plotting/saving
            lam=lam,
            scores_df=scores_df,
            scores_with_meta=scores_with_meta,
            load_df_all=load_df_all,
            var_df=var_df,
            level=None,
            feature_meta=feature_meta,
            saved={"excel": path},
        )
    return results
