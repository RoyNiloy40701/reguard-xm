"""Classical proposers. Each returns risky-asset weights (sum <= 1; the rest is cash) for the available assets.

Inputs at decision index t are past-only (features at t, forecasts made at t, covariance up to t).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def equal_weight(avail: np.ndarray, **_) -> np.ndarray:
    w = avail.astype(float)
    return w / w.sum() if w.sum() > 0 else w


def momentum(avail: np.ndarray, mom: np.ndarray, top_frac: float, **_) -> np.ndarray:
    ok = avail & np.isfinite(mom)
    w = np.zeros(len(avail))
    if ok.sum() == 0:
        return w
    k = max(1, int(round(ok.sum() * top_frac)))
    idx = np.flatnonzero(ok)[np.argsort(-mom[ok])[:k]]
    w[idx] = 1.0 / k
    return w


def risk_parity(avail: np.ndarray, vol: np.ndarray, **_) -> np.ndarray:
    ok = avail & np.isfinite(vol) & (vol > 0)
    w = np.zeros(len(avail))
    w[ok] = 1.0 / vol[ok]
    return w / w.sum() if w.sum() > 0 else w


def mean_variance(avail: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, risk_aversion: float,
                  max_weight: float = 1.0, **_) -> np.ndarray:
    """max w'mu - (lambda/2) w'Sigma w, long-only, fully invested (mu, Sigma daily)."""
    ok = avail & np.isfinite(mu) & (np.diag(Sigma) > 0)
    w = np.zeros(len(avail))
    n = int(ok.sum())
    if n == 0:
        return w
    m, S = mu[ok], Sigma[np.ix_(ok, ok)]
    x0 = np.full(n, 1.0 / n)
    res = minimize(lambda x: -(x @ m) + 0.5 * risk_aversion * x @ S @ x, x0,
                   jac=lambda x: -m + risk_aversion * S @ x, method="SLSQP",
                   bounds=[(0.0, max_weight)] * n, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0}],
                   options={"maxiter": 300, "ftol": 1e-12})
    x = np.clip(res.x if res.success else x0, 0, None)
    w[ok] = x / x.sum()
    return w
