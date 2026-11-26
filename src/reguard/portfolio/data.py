"""Market data for E4: daily return matrix, availability, out-of-sample expected returns, covariances.

Everything used at decision date t is known at t: features are past-only, forecasts come from annual refits
trained with embargo on data before each year, covariances use returns up to t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from reguard.data.splits import annual_refit_plan, train_mask
from reguard.models.baselines import GBM


class Market:
    def __init__(self, panel: pd.DataFrame, min_history: int):
        p = panel.sort_values(["date", "asset"])
        self.assets = sorted(p["asset"].unique())
        self.dates = pd.DatetimeIndex(sorted(p["date"].unique()))
        self.logret = p.pivot(index="date", columns="asset", values="ret").reindex(self.dates)[self.assets]
        self.ret = np.expm1(self.logret)                                   # daily simple returns
        rf = p.groupby("date")["rf_daily"].first().reindex(self.dates)
        self.rf = np.expm1(rf.ffill().fillna(0.0))                         # daily simple risk-free
        hist = self.logret.notna().cumsum()
        self.available = (hist >= min_history) & self.logret.notna()
        self.panel = p

    def feature(self, col: str) -> pd.DataFrame:
        return self.panel.pivot(index="date", columns="asset", values=col).reindex(self.dates)[self.assets]

    def cov(self, t: int, window: int, min_obs: int, cols: list[str]) -> np.ndarray:
        """Ledoit-Wolf covariance of daily simple returns over the `window` days ending at t (inclusive)."""
        R = self.ret.iloc[max(0, t - window + 1): t + 1][cols]
        R = R.loc[:, R.notna().sum() >= min_obs].fillna(0.0)
        S = np.zeros((len(cols), len(cols)))
        if R.shape[1] == 0:
            return S
        lw = LedoitWolf().fit(R.to_numpy())
        idx = [cols.index(c) for c in R.columns]
        S[np.ix_(idx, idx)] = lw.covariance_
        return S


def pooled_forecasts(panel: pd.DataFrame, splits: pd.DataFrame, cfg: dict, embargo: int, log=print) -> pd.DataFrame:
    """Out-of-sample pooled GBM forecasts of the 20-day excess return (annual refits, as E1 'pooled')."""
    fc = cfg["forecasts"]
    df = panel.sort_values(["date", "asset"]).reset_index(drop=True)
    cal = pd.DatetimeIndex(sorted(df["date"].unique()))
    out = []
    for yp in annual_refit_plan(splits, cal, fc["calibration_days"], embargo):
        fit = train_mask(df, yp["w0"], [fc["horizon"]]) & df[fc["target"]].notna()
        ev = (df["date"] >= yp["eval_start"]) & (df["date"] <= yp["eval_end"])
        m = GBM(seed=cfg["seed"], **fc["gbm"]).fit(df.loc[fit, fc["features"]], df.loc[fit, fc["target"]].to_numpy(), df.loc[fit])
        out.append(pd.DataFrame({"date": df.loc[ev, "date"], "asset": df.loc[ev, "asset"],
                                 "mu20": m.predict(df.loc[ev, fc["features"]])}))
        log(f"    forecasts {yp['year']}: trees={m.best_iter_} rows={int(ev.sum())}")
    return pd.concat(out, ignore_index=True)
