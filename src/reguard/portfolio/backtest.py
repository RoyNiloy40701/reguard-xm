"""Weekly-rebalanced long-only backtest with optional governance (E4).

Conventions (documented simplifications):
  * target weights are held as a constant mix between rebalances (no intra-week drift); costs are charged
    on the one-way traded weight at each rebalance;
  * cash earns the daily T-bill rate known at t;
  * the budget score is computed daily for the held weights: s = (L - m_hat) / sigma_hat with
    L = -(w'r + cash*rf), m_hat = -(w'mu + cash*rf), sigma_hat = sqrt(w' Sigma w) (Sigma, mu from the last rebalance);
  * q_t: ACI on the realised scores, quantile of lam * pool + (1 - lam) * own scores, lam = n0 / (n0 + n);
    q is applied by the governance projection at each rebalance; exceedance err = 1[s > q] is recorded daily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reguard.conformal.core import SortedPool, mixture_quantile
from reguard.portfolio.governance import project


def run_backtest(mkt, start: str, end: str, proposer, mu20: pd.DataFrame, cfg: dict, governed: bool,
                 pool: SortedPool | None, cost_bps: float | None = None, b_max: float | None = None) -> pd.DataFrame:
    g = cfg["governance"]
    cost = (cfg["cost_bps"] if cost_bps is None else cost_bps) / 1e4
    bmax = g["b_max"] if b_max is None else b_max
    idx = np.flatnonzero((mkt.dates >= pd.Timestamp(start)) & (mkt.dates <= pd.Timestamp(end)))
    idx = idx[idx < len(mkt.dates) - 1]
    R = mkt.ret.to_numpy()
    rf = mkt.rf.to_numpy()
    A = mkt.available.to_numpy()
    MU = mu20.reindex(index=mkt.dates, columns=mkt.assets).to_numpy() / 20.0
    cols = mkt.assets
    n = len(cols)
    w = np.zeros(n)
    mu_d, Sig = np.zeros(n), np.zeros((n, n))
    alpha_t, own = g["alpha"], []
    pool = pool if pool is not None else SortedPool(np.array([]))
    q = pool.quantile(1 - g["alpha"]) if len(pool.v) else 1.6449
    rows = []
    for k, t in enumerate(idx):
        reb = k % cfg["rebalance_every"] == 0
        traded, modified, status = 0.0, False, ""
        if reb:
            avail = A[t]
            mu_d = np.where(avail, np.nan_to_num(MU[t]), 0.0)
            Sig = mkt.cov(t, cfg["covariance"]["window"], cfg["covariance"]["min_obs"], cols)
            w_prop = np.where(avail, proposer(t, w, mu_d, Sig, avail), 0.0)
            if governed:
                w_new, info = project(w_prop, w, mu_d, Sig, q, avail, g["max_weight"], g["max_turnover"], bmax)
                modified, status = info["modified"], info["status"]
            else:
                w_new = w_prop
            traded = float(np.abs(w_new - w).sum())
            w = w_new
        r_next = np.nan_to_num(R[t + 1])
        cash = max(0.0, 1.0 - w.sum())
        gross = float(w @ r_next + cash * rf[t + 1])
        net = gross - cost * traded
        sig = float(np.sqrt(max(w @ Sig @ w, 0.0)))
        s, err, q_used = np.nan, np.nan, q
        if sig > 1e-8:
            L = -(w @ r_next + cash * rf[t + 1])
            m_hat = -(w @ mu_d + cash * rf[t + 1])
            s = (L - m_hat) / sig
            err = float(s > q)
            own.append(s)
            alpha_t = alpha_t + g["gamma"] * (g["alpha"] - err)
            lam = g["pseudo_count"] / (g["pseudo_count"] + len(own))
            q = mixture_quantile(pool, lam, np.asarray(own), 1 - alpha_t)
        rows.append({"date": mkt.dates[t + 1], "gross": gross, "net": net, "rf": rf[t + 1], "cash": cash,
                     "traded": traded, "rebalance": reb, "modified": modified, "status": status,
                     "score": s, "q_used": q_used, "err": err,
                     "sigma_hat": sig})
    return pd.DataFrame(rows)


def reference_pool(mkt, mu20, cfg, proposer) -> SortedPool:
    """Scores of an ungoverned reference portfolio over the pool period (initialises the governance budget)."""
    a, b = cfg["periods"]["pool"]
    bt = run_backtest(mkt, a, b, proposer, mu20, cfg, governed=False, pool=None)
    return SortedPool(bt["score"].dropna().to_numpy())
