"""E8 / C6: which market characteristics drive transfer quality? (H8, pre-registered design)

Outcome: episode-level calibration degradation g_cal. Drivers (pre-declared groups) are measured at the episode
start, label-free. Model: gradient boosting regressor with monotone constraints where an orientation is declared.
Importance: grouped permutation importance (increase in out-of-fold MSE when a group's columns are permuted
jointly), folds grouped by market. Stability: share of quarter-block bootstrap resamples with the modal top-3 set.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

USD_PEGGED = {"US", "HK", "SA", "QA", "AE"}     # USD itself or a hard USD peg (pre-declared)


def driver_table(ep: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Episode drivers. Target features are taken from the last trading day BEFORE tau (strictly past)."""
    f = panel[["date", "asset", "vol_60d", "dd_60d", "logdv_20d"]].sort_values("date")
    e = ep.sort_values("tau").copy()
    e["tau_m1"] = pd.to_datetime(e["tau"]) - pd.Timedelta(days=1)
    e = pd.merge_asof(e, f.rename(columns={"date": "fdate"}), left_on="tau_m1", right_on="fdate", by="asset",
                      direction="backward")
    out = pd.DataFrame({
        "episode_id": e["episode_id"], "market": e["market"], "tau": pd.to_datetime(e["tau"]), "g_cal": e["g_cal"],
        "vol_level": e["vol_60d"], "vol_ratio": e["abs_log_vol_ratio"],
        "liquidity": e["logdv_20d"], "liq_gap": e["liq_gap"],
        "corr_src": e["corr_src"],
        "atypicality": e["xs_atypicality"], "ks_mean": e["ks_mean"],
        "em": e["em_flag"],
        "usd_pegged": e["market"].isin(USD_PEGGED).astype(float),
        "drawdown": e["dd_60d"],
    })
    for r in sorted(e["region"].dropna().unique()):
        out[f"region_{r}"] = (e["region"] == r).astype(float).to_numpy()
    return out.reset_index(drop=True)


def groups(cols: list[str]) -> dict[str, list[str]]:
    g = {"volatility": ["vol_level", "vol_ratio"], "liquidity": ["liquidity", "liq_gap"],
         "correlation_with_sources": ["corr_src"], "atypicality": ["atypicality", "ks_mean"],
         "em_status": ["em"], "currency_regime": ["usd_pegged"], "drawdown_state": ["drawdown"],
         "region": [c for c in cols if c.startswith("region_")]}
    return {k: [c for c in v if c in cols] for k, v in g.items()}


# declared orientation (+1: larger -> worse transfer expected, -1: better, 0: none)
ORIENT = {"vol_ratio": 1, "atypicality": 1, "ks_mean": 1, "corr_src": -1, "em": 1}


def _model(cols, seed, cfg):
    return HistGradientBoostingRegressor(max_iter=cfg["max_iter"], learning_rate=cfg["lr"], max_leaf_nodes=cfg["max_leaf_nodes"],
                                         min_samples_leaf=cfg["min_samples_leaf"], early_stopping=False, random_state=seed,
                                         monotonic_cst=[ORIENT.get(c, 0) for c in cols])


def grouped_importance(D: pd.DataFrame, cfg: dict, seed: int = 0, n_folds: int = 5, n_perm: int = 3) -> tuple[pd.Series, float]:
    """Out-of-fold grouped permutation importance (MSE increase) and the out-of-fold R^2 of the driver model."""
    cols = [c for c in D.columns if c not in ("episode_id", "market", "tau", "g_cal")]
    G = groups(cols)
    X, y = D[cols], D["g_cal"].to_numpy()
    rng = np.random.default_rng(seed)
    imp = {k: 0.0 for k in G}
    pred = np.full(len(y), np.nan)
    k = min(n_folds, D["market"].nunique())
    for tr, te in GroupKFold(k).split(X, y, D["market"]):
        m = _model(cols, seed, cfg).fit(X.iloc[tr], y[tr])
        Xt = X.iloc[te].copy()
        p0 = m.predict(Xt)
        pred[te] = p0
        base = np.mean((y[te] - p0) ** 2)
        for name, gc in G.items():
            if not gc:
                continue
            for _ in range(n_perm):
                Xp = Xt.copy()
                perm = rng.permutation(len(Xp))
                Xp[gc] = Xt[gc].to_numpy()[perm]
                imp[name] += (np.mean((y[te] - m.predict(Xp)) ** 2) - base) * len(te) / (len(y) * n_perm)
    r2 = 1 - np.nansum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return pd.Series(imp).sort_values(ascending=False), float(r2)


def bootstrap_top3(D: pd.DataFrame, cfg: dict, n_boot: int, seed: int = 0, log=None) -> tuple[Counter, list[frozenset]]:
    """Resample whole quarters (all markets' episodes in a quarter move together); top-3 set per resample."""
    rng = np.random.default_rng(seed)
    qs = D["tau"].dt.to_period("Q").to_numpy()
    uq = np.unique(qs)
    by_q = {q: np.flatnonzero(qs == q) for q in uq}
    sets = []
    for b in range(n_boot):
        idx = np.concatenate([by_q[q] for q in rng.choice(uq, len(uq), replace=True)])
        imp, _ = grouped_importance(D.iloc[idx].reset_index(drop=True), cfg, seed=b, n_perm=1)
        sets.append(frozenset(imp.index[:3]))
        if log and (b + 1) % 100 == 0:
            log(f"    bootstrap {b+1}/{n_boot}")
    return Counter(sets), sets
