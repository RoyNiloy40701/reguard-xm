"""Phase 8 unit tests: governance projection (Prop. 3), proposers, RL environment, statistics.  Run: pytest -q"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reguard.portfolio import metrics as PM  # noqa: E402
from reguard.portfolio import proposers as P  # noqa: E402
from reguard.portfolio.governance import _budget, _chol, project  # noqa: E402
from reguard.portfolio.rl_env import masked_softmax  # noqa: E402


def _problem(n=12, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.normal(0, 0.01, (n, n))
    S = B @ B.T + np.eye(n) * 1e-4
    mu = rng.normal(0, 3e-4, n)
    avail = np.ones(n, bool); avail[-2:] = False
    return mu, S, avail, rng


@pytest.mark.parametrize("q", [1.0, 1.65, 3.0, 8.0])
def test_projection_is_always_feasible(q):
    mu, S, avail, rng = _problem()
    for _ in range(5):
        w_prop = rng.dirichlet(np.ones(len(mu)))
        w_prev = rng.dirichlet(np.ones(len(mu))) * 0.8
        w, info = project(w_prop, w_prev, mu, S, q, avail, w_max=0.15, tau=0.5, b_max=0.01)
        ok = avail
        L = _chol(S[np.ix_(ok, ok)])
        assert (w >= -1e-7).all() and w.sum() <= 1 + 1e-6 and (w <= 0.15 + 1e-6).all()
        assert np.allclose(w[~avail], 0)
        assert _budget(w[ok], mu[ok], L, q) <= 0.01 + 1e-5
        if not info["relaxed_turnover"]:       # the cap applies to available assets (unavailable ones must be sold)
            assert np.abs(w[avail] - w_prev[avail]).sum() <= 0.5 + 1e-5


def test_projection_keeps_feasible_proposals_and_goes_to_cash_when_q_infinite():
    mu, S, avail, _ = _problem()
    w_prop = np.where(avail, 0.01, 0.0)
    w, info = project(w_prop, w_prop, mu, S, 1.65, avail, 0.15, 0.5, b_max=0.05)
    assert np.allclose(w, w_prop) and not info["modified"]
    w, info = project(w_prop, w_prop, mu, S, np.inf, avail, 0.15, 0.5, 0.05)
    assert np.allclose(w, 0) and info["cash_forced"]


def test_proposers_long_only_and_respect_availability():
    mu, S, avail, rng = _problem()
    mom, vol = rng.normal(size=len(mu)), np.abs(rng.normal(0.2, 0.05, len(mu)))
    for w in (P.equal_weight(avail), P.momentum(avail, mom, 1 / 3), P.risk_parity(avail, vol),
              P.mean_variance(avail, mu, S, 5.0)):
        assert (w >= -1e-9).all() and w.sum() <= 1 + 1e-6 and np.allclose(w[~avail], 0)


def test_masked_softmax():
    avail = np.array([True, False, True])
    w = masked_softmax(np.array([5.0, 50.0, 1.0, 0.0]), avail)
    assert w[1] == 0 and 0 < w.sum() < 1 and np.isfinite(w).all()


def test_sharpe_diff_and_dsr():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0005, 0.01, 2000)
    t = PM.sharpe_diff_test(x, x)
    assert abs(t["sharpe_diff_daily"]) < 1e-12
    y = rng.normal(0.0020, 0.01, 2000)
    assert PM.sharpe_diff_test(y, x)["p"] < 0.05
    d = PM.deflated_sharpe(x, 10, 0.0001)
    assert 0 <= d <= 1
    assert PM.cvar(np.r_[-np.ones(5), np.zeros(95)]) == pytest.approx(1.0)
    est, lo, hi = PM.paired_bootstrap(x * 0.5, x, PM.cvar, n_boot=200)
    assert hi < 0 and lo <= est <= hi
