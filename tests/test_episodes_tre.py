"""Phase 5 unit tests: transfer signals, episode labels and TRE evaluation discipline.  Run:  pytest -q"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reguard.episodes.pseudo_transfer import cvar, window_labels  # noqa: E402
from reguard.models.baselines import GBMEnsemble  # noqa: E402
from reguard.tre import estimator as T  # noqa: E402
from reguard.tre import features as F  # noqa: E402

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "experiment" / "e2.yaml").read_text())


def _two(shift: float, n=600, m=80, d=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.normal(size=(m, d)) + shift


def test_shift_signals_grow_with_shift():
    A0, B0 = _two(0.0)
    A1, B1 = _two(1.0)
    s0 = F.shift_signals(A0, A0[:300], B0)
    s1 = F.shift_signals(A1, A1[:300], B1)
    for k in ("mmd", "mahalanobis", "dc_auc", "ks_mean"):
        assert s1[k] > s0[k], k
    assert abs(s0["dc_auc"] - 0.5) < 0.15
    assert F.gaussian_w2(*_two(0.0, m=5000)) < 0.3


def test_xs_atypicality():
    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(30), 40)
    C = rng.normal(size=(1200, 4))
    typical = rng.normal(size=(30, 4))
    assert abs(F.xs_atypicality(C, dates, typical, np.arange(30)) - 1) < 0.4
    assert F.xs_atypicality(C, dates, typical + 3, np.arange(30)) > 5


def test_cka_identical_covariance_is_one():
    A, _ = _two(0.0)
    assert F.cka_similarity(A, A) == pytest.approx(1.0)


def test_model_signals_use_no_target_labels():
    rng = np.random.default_rng(0)
    src = rng.normal(0, 1, (300, 5))
    tgt = rng.normal(0, 3, (60, 5))                    # more disagreement on the target
    s = F.model_signals(src, tgt, rng.normal(size=300))
    assert s["disagree_ratio"] > 1 and 0 <= s["pred_entropy"] <= np.log(2)
    # the signature has no target-label argument at all
    import inspect
    assert list(inspect.signature(F.model_signals).parameters) == ["src_members", "tgt_members", "src_y"]


def test_window_labels_and_cvar():
    n = 60
    r = pd.DataFrame({"y": np.linspace(-1, 1, n), "next_ret": np.r_[-np.ones(6), np.zeros(n - 6)]})
    for k in ("zs", "in"):
        r[f"{k}_mean"] = r["y"]
    r["zs_lo"], r["zs_hi"] = r["y"] + 0.5, r["y"] + 1.0         # zero-shot interval misses everything
    r["in_lo"], r["in_hi"] = r["y"] - 1.0, r["y"] + 1.0         # in-domain covers everything
    lab = window_labels(r, alpha=0.10)
    assert lab["cov_zs"] == 0 and lab["cov_in"] == 1
    assert lab["g_cal"] == pytest.approx(0.9 - 0.1)
    assert cvar(np.r_[np.ones(10), np.zeros(90)], 0.90) == pytest.approx(1.0)


def test_gbm_ensemble_members_differ_and_reproduce():
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = X["a"].to_numpy() + rng.normal(0, 1, n)
    meta = pd.DataFrame({"date": pd.bdate_range("2010-01-01", periods=n)})
    kw = dict(members=3, seed=1, block=63, gbm={"n_estimators": 60, "min_leaf": 50})
    m1 = GBMEnsemble(**kw).fit(X, y, meta)
    m2 = GBMEnsemble(**kw).fit(X, y, meta)
    P1, P2 = m1.predict_members(X.iloc[:50]), m2.predict_members(X.iloc[:50])
    assert np.allclose(P1, P2)
    assert P1.std(axis=1).mean() > 0


def _synthetic_episodes(n_markets=9, years=range(2005, 2010), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        for q in range(4):
            tau = pd.Timestamp(f"{y}-{3*q+1:02d}-01")
            for m in range(n_markets):
                sig = {c: rng.normal() for c in CFG["tre"]["signals"]}
                risk = sig["mmd"] + 0.5 * sig["dc_auc"] - 0.5 * sig["cka_sim"]
                rows.append({"episode_id": f"M{m}|{tau.date()}", "asset": f"M{m}", "market": f"M{m}", "year": y,
                             "tau": tau, "window_end": tau + pd.offsets.QuarterEnd(0),
                             "g_cal": risk + rng.normal(0, 1), "g_ic": rng.normal(), "g_cvar": rng.normal(), **sig})
    return pd.DataFrame(rows)


def test_tre_evaluation_is_time_ordered_and_market_held_out(monkeypatch):
    ep = _synthetic_episodes()
    cfg = {**CFG, "tre": {**CFG["tre"], "min_train_episodes": 60, "market_folds": 3, "calib_folds": 3,
                          "gbm": {**CFG["tre"]["gbm"], "max_iter": 30}}}
    seen = []
    real_fit = T.fit_tre

    def spy(X, y, groups, kind, orient, tc, seed):
        seen.append(X.index.copy())
        return real_fit(X, y, groups, kind, orient, tc, seed)

    monkeypatch.setattr(T, "fit_tre", spy)
    R = T.run_evaluation(ep, cfg, ["g_cal"], seed=0, log=lambda s: None)
    assert len(R) > 0
    folds = T.market_folds(ep["market"].unique().tolist(), 3, 0)
    # reconstruct: every scored episode's training set ended before its year and excludes its fold
    k = 0
    for Y in sorted(R["year"].unique()):
        for f in range(3):
            te = R[(R["year"] == Y) & (R["fold"] == f)]
            if te.empty:
                continue
            for _ in ("lr", "gbm"):
                tr = ep.loc[seen[k]]
                assert (tr["window_end"] < pd.Timestamp(f"{Y}-01-01")).all()
                assert not tr["market"].map(folds).eq(f).any()
                k += 1
    # the synthetic risk signal must be detected well above chance
    sc = T.score_table(R, CFG["tre"]["signals"])
    assert sc[(sc["scorer"] == "TRE-LR")]["auroc"].iloc[0] > 0.7
    h = T.h1_test(R, "g_cal", "rho_lr", CFG["tre"]["signals"], n_boot=100, seed=0)
    assert h["ci_lo"] <= h["diff"] <= h["ci_hi"]
