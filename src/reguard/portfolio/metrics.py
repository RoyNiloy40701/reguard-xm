"""Portfolio performance statistics and tests (daily net returns; never treated as independent)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cvar(x: np.ndarray, level: float = 0.95) -> float:
    """Expected shortfall of returns as a positive loss: mean of the worst (1 - level) share."""
    x = np.sort(np.asarray(x, float))
    k = max(1, int(np.ceil(len(x) * (1 - level) - 1e-9)))   # tolerance: 100 * 0.05 is 5.000000000000004 in floating point
    return float(-x[:k].mean())


def max_drawdown(x: np.ndarray) -> float:
    w = np.cumprod(1 + np.asarray(x, float))
    return float(np.max(1 - w / np.maximum.accumulate(w)))


def summary(bt: pd.DataFrame) -> dict:
    r, ex = bt["net"].to_numpy(), (bt["net"] - bt["rf"]).to_numpy()
    sd = ex.std(ddof=1)
    down = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    reb = bt[bt["rebalance"]]
    return {"ann_return": float((1 + r).prod() ** (252 / len(r)) - 1), "ann_vol": float(r.std(ddof=1) * np.sqrt(252)),
            "sharpe": float(ex.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
            "sortino": float(ex.mean() / down * np.sqrt(252)) if down > 0 else np.nan,
            "max_dd": max_drawdown(r), "cvar95": cvar(r), "turnover": float(reb["traded"].mean()),
            "cash_share": float(bt["cash"].mean()), "share_modified": float(reb["modified"].mean()),
            "bef": float(bt["err"].mean()), "days": len(bt)}


def stationary_bootstrap_idx(n: int, mean_block: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices."""
    idx = np.empty(n, dtype=int)
    p = 1.0 / mean_block
    i = rng.integers(n)
    for t in range(n):
        idx[t] = i
        i = rng.integers(n) if rng.random() < p else (i + 1) % n
    return idx


def paired_bootstrap(a: np.ndarray, b: np.ndarray, stat, n_boot=1000, mean_block=20, seed=0):
    """CI of stat(a) - stat(b) for two aligned daily series (same resampled days)."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        ix = stationary_bootstrap_idx(len(a), mean_block, rng)
        d.append(stat(a[ix]) - stat(b[ix]))
    d = np.array(d)
    return float(stat(a) - stat(b)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def sharpe_diff_test(x: np.ndarray, y: np.ndarray, lag: int | None = None) -> dict:
    """Ledoit-Wolf (2008) HAC test of equal Sharpe ratios (delta method, Bartlett kernel). x, y: excess returns."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    T = len(x)
    lag = lag if lag is not None else int(np.floor(4 * (T / 100) ** (2 / 9)))
    m1, m2 = x.mean(), y.mean()
    g1, g2 = (x ** 2).mean(), (y ** 2).mean()
    s1, s2 = np.sqrt(g1 - m1 ** 2), np.sqrt(g2 - m2 ** 2)
    diff = m1 / s1 - m2 / s2
    V = np.column_stack([x - m1, y - m2, x ** 2 - g1, y ** 2 - g2])
    Psi = V.T @ V / T
    for k in range(1, lag + 1):
        G = V[k:].T @ V[:-k] / T
        Psi += (1 - k / (lag + 1)) * (G + G.T)
    grad = np.array([g1 / s1 ** 3, -g2 / s2 ** 3, -m1 / (2 * s1 ** 3), m2 / (2 * s2 ** 3)])
    se = np.sqrt(max(grad @ Psi @ grad / T, 1e-18))
    z = diff / se
    return {"sharpe_diff_daily": float(diff), "sharpe_diff_ann": float(diff * np.sqrt(252)), "z": float(z),
            "p": float(2 * stats.norm.sf(abs(z)))}


def deflated_sharpe(ex: np.ndarray, n_trials: int, sr_trials_var: float) -> float:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio (probability the true SR > the expected max of trials)."""
    ex = np.asarray(ex, float)
    T = len(ex)
    sr = ex.mean() / ex.std(ddof=1)
    g3, g4 = stats.skew(ex), stats.kurtosis(ex, fisher=False)
    emc = 0.5772156649
    sr0 = np.sqrt(max(sr_trials_var, 0)) * ((1 - emc) * stats.norm.ppf(1 - 1 / max(n_trials, 2))
                                           + emc * stats.norm.ppf(1 - 1 / (max(n_trials, 2) * np.e)))
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / den))
