"""Phase 4 unit tests: models, preprocessing and metrics (synthetic data).  Run:  pytest -q"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from reguard.evaluation import metrics as M  # noqa: E402
from reguard.models.baselines import AR1, GBM, HistMean, MLPEnsemble, Ridge  # noqa: E402
from reguard.models.preprocess import TrainScaler, cs_rank  # noqa: E402


def _xy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    y = 0.5 * X["a"].to_numpy() - 0.3 * X["b"].to_numpy() + rng.normal(0, 1, n)
    meta = pd.DataFrame({"date": pd.bdate_range("2010-01-01", periods=n), "market": "XX",
                         "ret": rng.normal(0, 0.01, n)})
    return X, y, meta


def test_scaler_uses_training_rows_only():
    X, _, _ = _xy()
    sc = TrainScaler().fit(X.iloc[:1000])
    Z1 = sc.transform(X.iloc[1000:])
    X2 = X.copy()
    X2.iloc[1000:] *= 100                                   # change only non-training rows
    sc2 = TrainScaler().fit(X2.iloc[:1000])
    assert np.allclose(sc.mu_, sc2.mu_) and np.allclose(sc.sd_, sc2.sd_)
    assert np.isfinite(Z1).all()


@pytest.mark.parametrize("cls,kw", [(GBM, {"seed": 3, "n_estimators": 50}), (MLPEnsemble, {"seed": 3, "members": 2})])
def test_seed_reproducibility(cls, kw):
    X, y, meta = _xy()
    p1 = cls(**kw).fit(X, y, meta).predict(X.iloc[:100])
    p2 = cls(**kw).fit(X, y, meta).predict(X.iloc[:100])
    assert np.allclose(p1, p2)


def test_models_learn_signal():
    X, y, meta = _xy(seed=1)
    Xt, yt, mt = _xy(seed=2)
    for mdl in (Ridge(), GBM(seed=0, n_estimators=100, min_leaf=50), MLPEnsemble(members=2)):
        p = mdl.fit(X, y, meta).predict(Xt, mt)
        assert M.spearman(p, yt) > 0.2
    assert np.allclose(HistMean().fit(X, y).predict(Xt), y.mean())


def test_gbm_tolerates_all_missing_feature():
    X, y, meta = _xy()
    X["e"] = np.nan
    p = GBM(seed=0, n_estimators=20).fit(X, y, meta).predict(X.iloc[:10])
    assert np.isfinite(p).all()


def test_ar1_forecast_uses_current_return_only():
    _, _, meta = _xy()
    m = AR1(horizon=5, rf_col=None).fit(None, None, meta)
    a = m.predict(None, meta.iloc[[10]])
    meta2 = meta.copy()
    meta2.loc[meta2.index[11:], "ret"] = 5.0                # future returns must not matter at predict time
    b = m.predict(None, meta2.iloc[[10]])
    assert np.allclose(a, b)


def test_cs_rank_is_per_date():
    df = pd.DataFrame({"date": ["d1"] * 3 + ["d2"] * 3, "x": [1, 2, 3, 100, 200, 300]})
    r = cs_rank(df, ["x"])
    assert np.allclose(r["x"].iloc[:3].to_numpy(), r["x"].iloc[3:].to_numpy())


def test_metrics():
    y = np.array([1.0, -1, 2, -2, 0.5] * 10)
    assert M.oos_r2(y, y, np.zeros_like(y)) == pytest.approx(1.0)
    assert M.oos_r2(y, np.zeros_like(y), np.zeros_like(y)) == pytest.approx(0.0)
    assert M.dir_auc(y, y) == pytest.approx(1.0)
    assert M.pinball(np.array([1.0]), np.array([0.0]), 0.9) == pytest.approx(0.9)
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1, 2000)
    mu, t = M.nw_tstat(x, lag=10)
    assert mu == pytest.approx(0.5, abs=0.1) and t > 5
    _, t0, p0 = M.dm_test(rng.normal(size=500), rng.normal(size=500), lag=4)
    assert p0 > 0.001
    lo, hi = M.block_bootstrap_ci(pd.Series(x), block=20, n_boot=200)
    assert lo < 0.5 < hi


def test_year_plan_is_purged():
    import run_e1
    cal = pd.bdate_range("2000-01-03", "2021-12-31")
    from reguard.data.splits import make_windows
    w = make_windows(cal, "2005-01-01", "2022-01-01", embargo=20, min_train_days=504)
    plan = run_e1.year_plan(w, cal, cal_days=252, embargo=20)
    for p in plan:
        assert p["fit_window"]["train_end"] < p["cal_start"] <= p["train_end"] < p["eval_start"]
        gap = ((cal > p["fit_window"]["train_end"]) & (cal < p["cal_start"])).sum()
        assert gap == 20
