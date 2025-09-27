# hbspca/id_filter.py
from __future__ import annotations
from typing import Optional, List, Tuple, Dict
import io
import numpy as np
import pandas as pd

# --- Candidate column names (case-insensitive) ---
ID_COL_CANDIDATES   = ["ID", "Id", "id", "household_id", "HouseholdID", "Household_ID", "HHID", "hh_id"]
YEAR_COL_CANDIDATES = ["Year", "year", "YEAR"]

# ---------- small helpers ----------

def _first_hit(candidates: List[str], columns: List[str]) -> Optional[str]:
    """case-insensitive column match; returns the actual column name."""
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        hit = lower_map.get(cand.lower())
        if hit:
            return hit
    return None

def _to_int64(series: pd.Series) -> pd.Series:
    """Coerce to pandas nullable integer (Int64), dropping non-numeric."""
    x = pd.to_numeric(series, errors="coerce")
    return x.astype("Int64")

def _dedupe_keep_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return df.dropna(subset=cols)[cols].drop_duplicates().reset_index(drop=True)

# ---------- parsers ----------

def parse_ids_from_text(text: str) -> pd.DataFrame:
    """
    Input: arbitrary text containing IDs (separated by commas/newlines/space).
    Output: DataFrame with column 'ID' (Int64).
    """
    if not text or not text.strip():
        return pd.DataFrame(columns=["ID"])
    # keep digits, turn everything else into whitespace
    clean = "".join(ch if ch.isdigit() else " " for ch in text)
    ids = [i for i in clean.split() if i.isdigit()]
    out = pd.DataFrame({"ID": _to_int64(pd.Series(ids))})
    return _dedupe_keep_cols(out, ["ID"])

def parse_ids_from_file(
    file_obj,
    cluster_col: Optional[str] = None,
    keep_clusters: Optional[List[str]] = None,
    ignore_year: bool = True,
) -> pd.DataFrame:
    """
    Read CSV/XLSX and return a tidy IDs table.
    - Requires an ID-like column (auto-detected).
    - If cluster_col & keep_clusters are provided, filter rows by cluster first.
    - If ignore_year=True (default), do NOT include 'Year' in the output even if present.

    Returns columns: ['ID'] or ['Year','ID'] (nullable Int64).
    """
    # Try CSV, then Excel
    try:
        df = pd.read_csv(file_obj)
    except Exception:
        file_obj.seek(0)
        df = pd.read_excel(file_obj)

    if df is None or df.empty:
        return pd.DataFrame(columns=["ID"])

    id_col   = _first_hit(ID_COL_CANDIDATES, list(df.columns))
    year_col = _first_hit(YEAR_COL_CANDIDATES, list(df.columns))

    if id_col is None:
        raise ValueError("No ID-like column found (tried: ID / id / household_id / HouseholdID / HHID).")

    # Optional: filter by cluster BEFORE extracting IDs/Year
    if cluster_col and (cluster_col in df.columns) and keep_clusters:
        mask = df[cluster_col].astype(str).isin([str(x) for x in keep_clusters])
        df = df.loc[mask].copy()

    # Build output
    out = pd.DataFrame({"ID": _to_int64(df[id_col])})
    if (not ignore_year) and (year_col is not None):
        out["Year"] = _to_int64(df[year_col])

    # Clean & dedupe
    keep_cols = ["ID"] + (["Year"] if "Year" in out.columns else [])
    return _dedupe_keep_cols(out, keep_cols)

# ---------- apply to a data frame ----------

def apply_household_id_filter(
    df: pd.DataFrame,
    ids_df: Optional[pd.DataFrame],
    ignore_year: bool = True,
) -> pd.DataFrame:
    """
    Filter `df` by the provided IDs table.
    - If ids_df is None/empty: return df unchanged.
    - If ignore_year=True OR 'Year' not in ids_df: filter on ID only.
    - Else: filter on (Year, ID).
    - Any IDs not found in `df` are silently dropped (inner join behavior).

    Returns the filtered DataFrame.
    """
    if ids_df is None or ids_df.empty:
        return df

    # ensure types
    keep = ids_df.copy()
    if "ID" not in keep.columns:
        raise ValueError("ids_df must contain an 'ID' column.")
    keep["ID"] = _to_int64(keep["ID"])

    out = df.copy()
    out["ID"] = _to_int64(out["ID"])

    if (not ignore_year) and ("Year" in keep.columns) and ("Year" in out.columns):
        keep["Year"] = _to_int64(keep["Year"])
        merged = out.merge(
            _dedupe_keep_cols(keep, ["Year", "ID"]),
            on=["Year", "ID"],
            how="inner",
            validate="m:1",
        )
    else:
        merged = out.merge(
            _dedupe_keep_cols(keep, ["ID"]),
            on=["ID"],
            how="inner",
            validate="m:1",
        )
    return merged

# ---------- optional: small utility to report counts ----------

def filter_with_report(
    df: pd.DataFrame,
    ids_df: Optional[pd.DataFrame],
    ignore_year: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Like apply_household_id_filter, but also returns a small report dict with counts.
    """
    if ids_df is None or ids_df.empty:
        return df, {"kept_ids": df["ID"].nunique() if "ID" in df.columns else 0, "dropped_ids": 0}

    before = df["ID"].nunique() if "ID" in df.columns else 0
    filtered = apply_household_id_filter(df, ids_df, ignore_year=ignore_year)
    after = filtered["ID"].nunique() if "ID" in filtered.columns else 0
    return filtered, {"kept_ids": after, "dropped_ids": max(0, before - after)}
