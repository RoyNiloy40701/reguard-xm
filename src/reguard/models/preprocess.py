"""Feature preprocessing fitted on training rows only (blueprint integrity rule 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TrainScaler:
    """Winsorise at training quantiles, standardise with training mean/std, impute 0 (= training mean).

    fit() sees training rows only; transform() never refits.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X: pd.DataFrame) -> "TrainScaler":
        self.cols_ = list(X.columns)
        self.lo_ = X.quantile(self.lower)
        self.hi_ = X.quantile(self.upper)
        Xc = X.clip(self.lo_, self.hi_, axis=1)
        self.mu_ = Xc.mean()
        self.sd_ = Xc.std().replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Xc = X[self.cols_].clip(self.lo_, self.hi_, axis=1)
        Z = (Xc - self.mu_) / self.sd_
        return Z.fillna(0.0).to_numpy(dtype=np.float32)


def cs_rank(df: pd.DataFrame, cols: list[str], by: str = "date") -> pd.DataFrame:
    """Cross-sectional rank transform per date to [-0.5, 0.5] (scale-free; NaN stays NaN).

    Uses only same-date values, so it adds no look-ahead.
    """
    out = df[cols].groupby(df[by]).rank(pct=True) - 0.5
    return out
