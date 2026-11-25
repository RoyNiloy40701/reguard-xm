"""Predictive metrics and tests that respect overlapping (autocorrelated) targets.

h-day forward targets overlap, so daily errors are autocorrelated: every t-statistic uses
Newey-West (HAC) standard errors with lag >= h-1, and confidence intervals use a moving-block
bootstrap over dates. Daily observations are never treated as independent.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.spearmanr(a[ok], b[ok]).statistic)


def oos_r2(y, pred, bench) -> float:
    """Campbell-Thompson out-of-sample R^2 against a benchmark forecast (historical mean)."""
    y, pred, bench = (np.asarray(v, float) for v in (y, pred, bench))
    ok = np.isfinite(y) & np.isfinite(pred) & np.isfinite(bench)
    if ok.sum() < 10:
        return np.nan
    return float(1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum((y[ok] - bench[ok]) ** 2))


def dir_auc(y, pred) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    lab = (y[ok] > 0).astype(int)
    if ok.sum() < 10 or lab.min() == lab.max():
        return np.nan
    return float(roc_auc_score(lab, pred[ok]))


def pinball(y, q_pred, q: float) -> float:
    y, q_pred = np.asarray(y, float), np.asarray(q_pred, float)
    ok = np.isfinite(y) & np.isfinite(q_pred)
    d = y[ok] - q_pred[ok]
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def nw_tstat(x, lag: int) -> tuple[float, float]:
    """Mean of x and its Newey-West t-statistic (Bartlett kernel)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * lag + 5:
        return (float(np.mean(x)) if n else np.nan), np.nan
    u = x - x.mean()
    s = u @ u / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    se = np.sqrt(max(s, 1e-18) / n)
    return float(x.mean()), float(x.mean() / se)


def dm_test(loss_a, loss_b, lag: int) -> tuple[float, float, float]:
    """Diebold-Mariano on d = loss_a - loss_b with HAC variance. Returns (mean d, DM stat, two-sided p).
    Negative mean => model A has lower loss."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    m, t = nw_tstat(d, lag)
    p = float(2 * stats.norm.sf(abs(t))) if np.isfinite(t) else np.nan
    return m, t, p


def block_bootstrap_ci(values_by_date: pd.Series, stat=np.mean, block: int = 21, n_boot: int = 1000,
                       alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Moving-block bootstrap CI for a statistic of a date-ordered series."""
    x = values_by_date.dropna().to_numpy(float)
    n = len(x)
    if n < 2 * block:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    bs = np.apply_along_axis(stat, 1, x[idx])
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def daily_cs_ic(df: pd.DataFrame, pred: str, y: str, min_n: int = 10) -> pd.Series:
    """Cross-sectional Spearman rank IC per date (dates with >= min_n assets)."""
    g = df[["date", pred, y]].dropna().groupby("date")
    n = g.size()
    ic = g.apply(lambda d: spearman(d[pred], d[y]) if len(d) >= min_n else np.nan, include_groups=False)
    return ic[n >= min_n].dropna()
