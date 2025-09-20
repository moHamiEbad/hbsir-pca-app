from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

@dataclass
class YearResult:
    X_cols: List[str]
    comps: Optional[np.ndarray]
    lam: np.ndarray
    scores_df: pd.DataFrame
    scores_with_meta: pd.DataFrame
    load_df_all: pd.DataFrame
    var_df: pd.DataFrame
    level: Optional[int]
    feature_meta: Optional[pd.DataFrame]
    saved: Dict[str, str] = field(default_factory=dict)

@dataclass
class SidebarState:
    use_existing: bool
    load_dir: Optional[str]
    years: List[int]
    area_kind: str  # "Whole country" | "Subareas"
    selected_provinces: List[str]
    selected_counties: List[str]
    settlement: str  # Both | Urban | Rural
    level: int
    weighted: bool
    norm_mode: str
    exp_measure: str
    save_dir: str
    save_as_excel: bool
