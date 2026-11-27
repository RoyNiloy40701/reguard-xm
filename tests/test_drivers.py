"""E8 unit tests: grouped permutation importance finds a planted driver; drivers use past-only features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reguard.tre import drivers as DR  # noqa: E402

CFG = {"max_iter": 60, "lr": 0.1, "max_leaf_nodes": 7, "min_samples_leaf": 20}


def _toy(n_mk=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_mk):
        for q in range(24):
            r = {"episode_id": f"M{m}|{q}", "market": f"M{m}", "tau": pd.Timestamp("2010-01-01") + pd.DateOffset(months=3 * q)}
            for c in ["vol_level", "vol_ratio", "liquidity", "liq_gap", "corr_src", "atypicality", "ks_mean", "em",
                      "usd_pegged", "drawdown", "region_A", "region_B"]:
                r[c] = rng.normal()
            r["g_cal"] = 0.8 * r["atypicality"] + 0.1 * rng.normal()       # planted driver
            rows.append(r)
    return pd.DataFrame(rows)


def test_planted_driver_ranks_first():
    imp, r2 = DR.grouped_importance(_toy(), CFG, seed=0, n_folds=4, n_perm=2)
    assert imp.index[0] == "atypicality" and r2 > 0.5


def test_bootstrap_counts_sum_to_n():
    cnt, sets = DR.bootstrap_top3(_toy(n_mk=8), CFG, n_boot=5)
    assert sum(cnt.values()) == 5 and all(len(s) == 3 for s in sets)


def test_driver_table_uses_features_strictly_before_tau():
    tau = pd.Timestamp("2015-03-02")
    panel = pd.DataFrame({"date": [pd.Timestamp("2015-02-27"), tau], "asset": ["X", "X"],
                          "vol_60d": [0.1, 9.9], "dd_60d": [-0.1, -9.9], "logdv_20d": [1.0, 99.0]})
    ep = pd.DataFrame({"episode_id": ["X|1"], "asset": ["X"], "market": ["XX"], "tau": [tau], "g_cal": [0.0],
                       "abs_log_vol_ratio": [0.1], "liq_gap": [0.0], "corr_src": [0.5], "xs_atypicality": [1.0],
                       "ks_mean": [0.2], "em_flag": [0.0], "region": ["A"]})
    D = DR.driver_table(ep, panel)
    assert D["vol_level"].iloc[0] == 0.1 and D["drawdown"].iloc[0] == -0.1
