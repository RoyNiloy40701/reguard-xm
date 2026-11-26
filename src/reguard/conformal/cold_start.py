"""Cold-start conformal risk budget (CSCRB): transfer-residual pool, nested stratifier choice,
Mondrian strata, localised weights and the T1 coverage-gap diagnostic.

Strict time discipline: for an entry in year Y, everything (pool, stratifier choice, strata edges,
gap estimates) uses only episodes that ended before 1 January Y and never the entered market itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from reguard.conformal.core import SortedPool, ks_distance


def build_scores(rows: pd.DataFrame, panel: pd.DataFrame, sigma_floor: float) -> pd.DataFrame:
    """Daily standardised loss scores for every episode row (zero-shot and in-domain forecasts)."""
    vol = panel[["date", "asset", "vol_20d"]]
    r = rows.merge(vol, on=["date", "asset"], how="left")
    sig = np.maximum(r["vol_20d"].to_numpy(float) / np.sqrt(252), sigma_floor)
    r["sigma"] = sig
    r["s_zs"] = -(r["next_ret"] - r["zs_mean"] / 5.0) / sig
    r["s_in"] = -(r["next_ret"] - r["in_mean"] / 5.0) / sig
    r["year"] = pd.to_datetime(r["date"]).dt.year
    return r[np.isfinite(r["s_zs"]) & np.isfinite(r["s_in"])].sort_values(["asset", "date"]).reset_index(drop=True)


def choose_stratifier(ep_pool: pd.DataFrame, sc_pool: pd.DataFrame, candidates: list[str], alpha: float) -> tuple[str | None, dict]:
    """Nested rule: candidate with the largest Spearman correlation between its value and the episode's
    exceedance rate at the unstratified pool (1 - alpha) quantile, on pool episodes only."""
    q_u = float(np.quantile(sc_pool["s_zs"], 1 - alpha))
    rate = sc_pool.assign(e=(sc_pool["s_zs"] > q_u).astype(float)).groupby("episode_id")["e"].mean()
    rhos = {}
    for c in candidates:
        if c not in ep_pool:
            continue
        x = ep_pool.set_index("episode_id")[c].reindex(rate.index)
        ok = x.notna()
        if ok.sum() >= 50:
            rhos[c] = float(stats.spearmanr(x[ok], rate[ok]).statistic)
    if not rhos:
        return None, rhos
    best = max(rhos, key=rhos.get)
    return (best if rhos[best] > 0 else None), rhos


def strata_edges(values: np.ndarray, k: int) -> np.ndarray:
    return np.quantile(values[np.isfinite(values)], np.linspace(0, 1, k + 1)[1:-1])


def assign(value: float, edges: np.ndarray) -> int:
    return int(np.searchsorted(edges, value, side="right"))


def kernel_weights(ep_pool: pd.DataFrame, target_sig: pd.Series, cols: list[str]) -> pd.Series:
    """Localised weights exp(-d^2/h^2), d = distance between standardised signal vectors, h = median distance."""
    X = ep_pool[cols].astype(float)
    mu, sd = X.mean(), X.std().replace(0, 1.0)
    Z = ((X - mu) / sd).fillna(0.0).to_numpy()
    zt = ((target_sig[cols].astype(float) - mu) / sd).fillna(0.0).to_numpy()
    d2 = ((Z - zt) ** 2).sum(1)
    h2 = np.median(d2) if np.median(d2) > 0 else 1.0
    return pd.Series(np.exp(-d2 / h2), index=ep_pool["episode_id"].to_numpy())


def gap_estimate(sc_stratum: pd.DataFrame, max_rest: int = 20000, seed: int = 0) -> float:
    """T1 diagnostic: estimated stratum mismatch = mean over markets of the Kolmogorov distance between that
    market's post-transfer scores and the rest of the stratum (leave-one-market-out)."""
    rng = np.random.default_rng(seed)
    ds = []
    for m, g in sc_stratum.groupby("market"):
        rest = sc_stratum.loc[sc_stratum["market"] != m, "s_zs"].to_numpy()
        if len(rest) > max_rest:
            rest = rng.choice(rest, max_rest, replace=False)
        d = ks_distance(g["s_zs"].to_numpy(), rest)
        if np.isfinite(d):
            ds.append(d)
    return float(np.mean(ds)) if ds else np.nan


class EntryContext:
    """Everything an entry into market j in year Y may know: episodes of OTHER markets that ended before
    1 January Y, the stratifier chosen on them, strata edges, and the stratum gap estimate."""

    def __init__(self, Y: int, market: str, ep_y: pd.DataFrame, sc_y: pd.DataFrame, cfg: dict):
        c = cfg["cscrb"]
        self.Y, self.market = Y, market
        self.ep = ep_y[ep_y["market"] != market]
        self.sc = sc_y[sc_y["market"] != market].copy()
        self.stratifier, self.strat_rhos = choose_stratifier(self.ep, self.sc, c["stratifier_candidates"], cfg["alpha"])
        self.edges = None
        if self.stratifier is not None:
            vals = self.ep[self.stratifier].to_numpy(float)
            self.edges = strata_edges(vals, c["strata"])
            b = pd.Series([assign(v, self.edges) if np.isfinite(v) else -1 for v in vals], index=self.ep["episode_id"])
            self.sc["stratum"] = self.sc["episode_id"].map(b).to_numpy()

    def gap(self, stratum: int) -> float:
        return gap_estimate(self.sc[self.sc["stratum"] == stratum]) if "stratum" in self.sc else np.nan

    def pool(self, stratum: int | None = None, weights: pd.Series | None = None, col: str = "s_zs") -> SortedPool:
        p = self.sc
        if stratum is not None and "stratum" in p:
            p = p[p["stratum"] == stratum]
        w = None if weights is None else p["episode_id"].map(weights).fillna(0.0).to_numpy()
        return SortedPool(p[col].to_numpy(), w)
