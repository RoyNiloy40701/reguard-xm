"""Phase 3 unit tests on synthetic data (no raw files needed).  Run:  pytest -q"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reguard.data import store  # noqa: E402
from reguard.data.dse import compute_universe  # noqa: E402
from reguard.data.features import asset_features, forward_targets  # noqa: E402
from reguard.data.splits import make_windows, train_mask  # noqa: E402

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "data" / "processing.yaml").read_text())


def _series(n=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    df = pd.DataFrame({"ret": rng.normal(0, 0.01, n), "dollar_vol": rng.lognormal(10, 1, n)}, index=idx)
    bench = pd.Series(rng.normal(0, 0.01, n), index=idx)
    return df, bench


def test_features_ignore_the_future():
    df, bench = _series()
    t = 500
    f0 = asset_features(df, CFG, bench)
    df2, bench2 = df.copy(), bench.copy()
    df2.iloc[t + 1:] = df2.iloc[t + 1:] * 50 + 1.0          # wreck everything after t
    bench2.iloc[t + 1:] = 9.9
    f1 = asset_features(df2, CFG, bench2)
    pd.testing.assert_frame_equal(f0.iloc[: t + 1], f1.iloc[: t + 1])


def test_targets_ignore_the_past_and_present():
    df, bench = _series()
    t = 400
    y0 = forward_targets(df["ret"], [5, 20], bench_ret=bench)
    r2 = df["ret"].copy()
    r2.iloc[: t + 1] = 0.5                                   # change everything up to and including t
    y1 = forward_targets(r2, [5, 20], bench_ret=bench)
    pd.testing.assert_frame_equal(y0.iloc[t:t + 1], y1.iloc[t:t + 1])


def test_target_values_and_missing_returns():
    r = pd.Series([0.01, 0.02, np.nan, 0.03, 0.04, 0.05], index=pd.bdate_range("2020-01-01", periods=6))
    y = forward_targets(r, [2])
    assert y["y_ret_2d"].iloc[3] == pytest.approx(0.09)     # 0.04 + 0.05
    assert np.isnan(y["y_ret_2d"].iloc[0])                  # horizon contains a missing return
    assert np.isnan(y["y_ret_2d"].iloc[5])                  # horizon past the end, no delisting rule
    assert y["tgt_end_2d"].iloc[3] == r.index[5]


def test_delisting_terminal_return():
    r = pd.Series([0.01, 0.02, 0.03], index=pd.bdate_range("2020-01-01", periods=3))
    y = forward_targets(r, [5], end_value=-1.0)
    assert np.isfinite(y["y_ret_5d"]).all()
    assert y["y_ret_5d"].iloc[1] == pytest.approx(0.03 + np.log(0.01))   # -100% floored at -99%
    assert y["y_ret_5d"].iloc[2] == pytest.approx(np.log(0.01))


def test_windows_embargo_and_purge():
    cal = pd.bdate_range("2012-01-02", "2021-12-31")
    w = make_windows(cal, "2015-01-01", "2022-01-01", embargo=20, min_train_days=504)
    assert (w["eval_end"] < pd.Timestamp("2022-01-01")).all()
    for _, row in w.iterrows():
        gap = ((cal > row["train_end"]) & (cal < row["eval_start"])).sum()
        assert gap == 20
    # an illiquid asset whose 20-day target spans beyond eval_start must be purged
    df = pd.DataFrame({"date": cal})
    df["tgt_end_20d"] = df["date"] + pd.Timedelta(days=200)
    m = train_mask(df, w.iloc[3], [20])
    assert (df.loc[m, "tgt_end_20d"] < w.iloc[3]["eval_start"]).all()


def test_universe_is_point_in_time():
    rng = np.random.default_rng(1)
    cal = pd.bdate_range("2015-01-01", periods=400)
    tv = pd.DataFrame(rng.lognormal(10, 1, (400, 80)), index=cal, columns=[f"S{i}" for i in range(80)])
    tv = tv.mask(rng.random(tv.shape) < 0.05)
    cfg = {"dse": {"universe": dict(CFG["dse"]["universe"], top_n=10)}}
    u0 = compute_universe(tv, cal, cfg)
    d = u0["decision_date"].unique()[3]
    tv2 = tv.copy()
    tv2.loc[tv2.index > d] = tv2.loc[tv2.index > d] * 1000 * rng.random((int((cal > d).sum()), 80))
    u1 = compute_universe(tv2, cal, cfg)
    a = u0[u0["decision_date"] <= d].reset_index(drop=True)
    b = u1[u1["decision_date"] <= d].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert (u0["effective_from"] > u0["decision_date"]).all()


def test_test_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "PROCESSED_ROOT", tmp_path)
    monkeypatch.setattr(store, "LOCKED_ROOT", tmp_path / "locked")
    monkeypatch.setattr(store, "DOCS_ROOT", tmp_path / "docs")
    monkeypatch.setattr(store, "ACCESS_LOG", tmp_path / "docs" / "log.csv")
    monkeypatch.delenv("REGUARD_UNLOCK_TEST", raising=False)
    dates = pd.bdate_range("2021-11-01", "2022-02-28")
    df = pd.DataFrame({"date": dates, "asset": "X", "y_ret_20d": 0.1})
    df["tgt_end_20d"] = df["date"] + pd.Timedelta(days=28)
    info = store.save_panel(df, "p", "2022-01-01", [20])
    dev = store.load_panel("p")
    assert dev["date"].max() < pd.Timestamp("2022-01-01")
    assert dev.loc[dev["date"] > pd.Timestamp("2021-12-05"), "y_ret_20d"].isna().all()   # boundary purge
    assert info["boundary_purged_targets"] > 0
    with pytest.raises(store.TestLockedError):
        store.load_panel("p", part="test")
    monkeypatch.setenv("REGUARD_UNLOCK_TEST", "1")
    with pytest.raises(store.TestLockedError):
        store.load_panel("p", part="test")                  # unlock flag alone is not enough
    t = store.load_panel("p", part="test", unlock_reason="unit test")
    assert t["date"].min() >= pd.Timestamp("2022-01-01")
    assert len(pd.read_csv(tmp_path / "docs" / "log.csv")) == 3
