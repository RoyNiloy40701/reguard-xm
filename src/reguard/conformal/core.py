"""Online conformal risk budgets for one market-entry stream (E3).

Every method returns q[t], the budget used at decision time t, computed ONLY from information
available at t: the (fixed) calibration pool and the target scores s[0..t-1] already realised.
err[t] = 1[s[t] > q[t]] (budget exceedance). q = +inf means "no admissible risk" (abstain / cash).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
class SortedPool:
    """Pre-sorted weighted calibration pool: fast weighted CDF."""

    def __init__(self, values: np.ndarray, weights: np.ndarray | None = None):
        v = np.asarray(values, float)
        w = np.ones_like(v) if weights is None else np.asarray(weights, float)
        ok = np.isfinite(v) & (w > 0)
        o = np.argsort(v[ok])
        self.v = v[ok][o]
        cw = np.cumsum(w[ok][o])
        self.total = cw[-1] if len(cw) else 0.0
        self.cdf_at = cw / self.total if self.total > 0 else cw

    def cdf(self, x: float) -> float:
        """Weighted share of pool values <= x."""
        if len(self.v) == 0:
            return 0.0
        i = np.searchsorted(self.v, x, side="right")
        return float(self.cdf_at[i - 1]) if i > 0 else 0.0

    def quantile(self, level: float) -> float:
        if len(self.v) == 0:
            return np.nan
        if level >= 1:
            return np.inf
        i = int(np.searchsorted(self.cdf_at, level, side="left"))
        return float(self.v[min(i, len(self.v) - 1)])


def weighted_quantile(values, weights, level: float, inf_mass: float = 0.0) -> float:
    """Smallest v with weighted CDF >= level; optional extra mass at +inf (conformal correction)."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    ok = np.isfinite(v) & (w > 0)
    v, w = v[ok], w[ok]
    if len(v) == 0 or level >= 1:
        return np.inf
    if level <= 0:
        return -np.inf
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / (w.sum() + inf_mass)
    i = int(np.searchsorted(cw, level, side="left"))
    return float(v[o][i]) if i < len(v) else np.inf


def mixture_quantile(pool: SortedPool, lam: float, target: np.ndarray, level: float) -> float:
    """Quantile of lam * pool + (1 - lam) * empirical(target) at `level` (bisection on the CDF)."""
    if level >= 1:
        return np.inf
    if level <= 0:
        return -np.inf
    t = np.sort(target[np.isfinite(target)]) if len(target) else np.array([])
    if len(t) == 0 or lam >= 1:
        return pool.quantile(level)
    if lam <= 0 or len(pool.v) == 0:
        return weighted_quantile(t, np.ones(len(t)), level)

    def F(x):
        return lam * pool.cdf(x) + (1 - lam) * np.searchsorted(t, x, side="right") / len(t)

    lo = min(pool.v[0], t[0]) - 1e-9
    hi = max(pool.v[-1], t[-1])
    if F(hi) < level:
        return np.inf
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if F(mid) >= level:
            hi = mid
        else:
            lo = mid
    return hi


def _gauss(level: float) -> float:
    if level >= 1:
        return np.inf
    if level <= 0:
        return -np.inf
    return float(stats.norm.ppf(level))


# ---------------------------------------------------------------------------
# methods: each returns q (len H) for target scores s (len H); q[t] uses s[:t] only
# ---------------------------------------------------------------------------
def run_constant(s: np.ndarray, q0: float) -> np.ndarray:
    return np.full(len(s), q0, dtype=float)


def run_time_weighted(s, alpha, warmup, z, halflife):
    q = np.empty(len(s))
    for t in range(len(s)):
        if t < warmup:
            q[t] = z
            continue
        age = np.arange(t - 1, -1, -1, dtype=float)
        w = 0.5 ** (age / halflife)
        q[t] = weighted_quantile(s[:t], w, 1 - alpha, inf_mass=1.0)
    return q


def run_regime_weighted(s, sig, alpha, warmup, z, bandwidth):
    q = np.empty(len(s))
    ls = np.log(sig)
    for t in range(len(s)):
        if t < warmup:
            q[t] = z
            continue
        w = np.exp(-0.5 * ((ls[:t] - ls[t]) / bandwidth) ** 2)
        q[t] = weighted_quantile(s[:t], w, 1 - alpha, inf_mass=1.0)
    return q


def run_tcp(s, alpha, z, eta):
    """TCP-style online quantile tracking: q_{t+1} = q_t + eta * (err_t - alpha)."""
    q = np.empty(len(s))
    qt = z
    for t in range(len(s)):
        q[t] = qt
        qt = qt + eta * (float(s[t] > qt) - alpha)
    return q


def run_aci_target(s, alpha, warmup, gamma):
    """ACI on the target's own scores; Gaussian quantile at level 1 - alpha_t during warm-up."""
    q = np.empty(len(s))
    a = alpha
    for t in range(len(s)):
        lev = 1 - a
        q[t] = _gauss(lev) if t < warmup else weighted_quantile(s[:t], np.ones(t), lev, inf_mass=1.0)
        a = a + gamma * (alpha - float(s[t] > q[t]))
    return q


def run_dtaci(s, alpha, warmup, gammas, eta):
    """Dynamically-tuned ACI (Gibbs & Candes 2022, simplified): ACI experts with different steps,
    combined by exponential weights on the pinball loss of each expert's threshold."""
    K = len(gammas)
    a = np.full(K, alpha)
    logw = np.zeros(K)
    q = np.empty(len(s))
    for t in range(len(s)):
        qk = np.array([_gauss(1 - ak) if t < warmup else weighted_quantile(s[:t], np.ones(t), 1 - ak, 1.0) for ak in a])
        p = np.exp(logw - logw.max()); p /= p.sum()
        a_comb = float(p @ a)
        q[t] = _gauss(1 - a_comb) if t < warmup else weighted_quantile(s[:t], np.ones(t), 1 - a_comb, 1.0)
        d = s[t] - np.where(np.isfinite(qk), qk, 1e6)
        loss = np.maximum((1 - alpha) * d, -alpha * d) / alpha
        logw = logw - eta * np.minimum(loss, 50)
        logw = np.log(0.95 * np.exp(logw - logw.max()) / np.exp(logw - logw.max()).sum() + 0.05 / K)
        a = a + np.array(gammas) * (alpha - (s[t] > qk).astype(float))
    return q


def run_pool_aci(s, pool: SortedPool, alpha, gamma, pseudo_count, online=True):
    """Transfer-pool cold start + ACI. q_t = Quantile_{1-alpha_t}(lam_t * pool + (1 - lam_t) * target scores),
    lam_t = n0 / (n0 + t). online=False gives the static entry threshold q0 (used for the T1 diagnostic)."""
    q = np.empty(len(s))
    a = alpha
    q0 = pool.quantile(1 - alpha)
    for t in range(len(s)):
        if not online:
            q[t] = q0
            continue
        lam = pseudo_count / (pseudo_count + t)
        q[t] = mixture_quantile(pool, lam, s[:t], 1 - a)
        a = a + gamma * (alpha - float(s[t] > q[t]))
    return q


# ---------------------------------------------------------------------------
def stream_metrics(s, q, sig, alpha, horizons) -> dict:
    err = (s > q).astype(float)
    out = {"n": len(s)}
    for h in list(horizons) + [len(s)]:
        k = min(h, len(s))
        key = "full" if h == len(s) else str(h)
        out[f"bef_{key}"] = float(err[:k].mean()) if k else np.nan
    sev = (s - q)[err.astype(bool)]
    out["severity_sigma"] = float(sev.mean()) if len(sev) else 0.0
    out["severity_loss"] = float(((s - q) * sig)[err.astype(bool)].mean()) if len(sev) else 0.0
    fin = np.isfinite(q)
    out["mean_q"] = float(q[fin].mean()) if fin.any() else np.nan
    out["abstain_share"] = float((~fin).mean())
    return out


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10:
        return np.nan
    return float(stats.ks_2samp(a, b).statistic)
