"""Phase 6 unit tests: conformal budgets and the cold-start context.  Run:  pytest -q"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reguard.conformal import core as C  # noqa: E402
from reguard.conformal.cold_start import EntryContext, gap_estimate  # noqa: E402

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "experiment" / "e3.yaml").read_text())


def test_sorted_pool_matches_numpy_quantile():
    rng = np.random.default_rng(0)
    v = rng.normal(size=5000)
    p = C.SortedPool(v)
    assert p.quantile(0.95) == pytest.approx(np.quantile(v, 0.95, method="inverted_cdf"))
    w = np.r_[np.ones(2500), np.zeros(2500)]
    assert C.SortedPool(v, w).quantile(0.5) == pytest.approx(np.quantile(v[:2500], 0.5, method="inverted_cdf"))


def test_mixture_quantile_limits():
    rng = np.random.default_rng(1)
    pool = C.SortedPool(rng.normal(size=3000))
    tgt = rng.normal(5, 1, size=200)
    assert C.mixture_quantile(pool, 1.0, tgt, 0.9) == pytest.approx(pool.quantile(0.9))
    assert C.mixture_quantile(pool, 0.0, tgt, 0.9) == pytest.approx(C.weighted_quantile(tgt, np.ones(200), 0.9))
    m = C.mixture_quantile(pool, 0.5, tgt, 0.9)
    assert pool.quantile(0.9) < m < C.weighted_quantile(tgt, np.ones(200), 0.9) + 1e-9


def test_exchangeable_pool_gives_nominal_coverage():
    rng = np.random.default_rng(2)
    pool = C.SortedPool(rng.standard_t(4, size=20000))
    s = rng.standard_t(4, size=20000)
    q = C.run_pool_aci(s, pool, 0.05, 0.005, 60, online=False)
    assert abs((s > q).mean() - 0.05) < 0.006


def test_aci_long_run_bound_on_adversarial_sequence():
    """Proposition 2: |mean err - alpha| <= (max(a1, 1 - a1) + gamma) / (gamma N) for ANY sequence."""
    rng = np.random.default_rng(3)
    N, alpha, gamma = 3000, 0.05, 0.01
    s = np.r_[rng.normal(0, 1, 1000), rng.normal(0, 5, 1000), rng.normal(3, 0.5, 1000)]   # regime breaks
    pool = C.SortedPool(rng.normal(size=2000))
    q = C.run_pool_aci(s, pool, alpha, gamma, pseudo_count=60)
    bound = (max(alpha, 1 - alpha) + gamma) / (gamma * N)
    assert abs((s > q).mean() - alpha) <= bound + 1e-9


def test_budgets_never_use_the_current_outcome():
    rng = np.random.default_rng(4)
    s = rng.normal(size=120)
    sig = np.exp(rng.normal(-4, 0.2, size=120))
    pool = C.SortedPool(rng.normal(size=1000))
    runners = {
        "tw": lambda x: C.run_time_weighted(x, 0.05, 20, 1.645, 60),
        "rw": lambda x: C.run_regime_weighted(x, sig, 0.05, 20, 1.645, 0.5),
        "tcp": lambda x: C.run_tcp(x, 0.05, 1.645, 0.05),
        "aci": lambda x: C.run_aci_target(x, 0.05, 20, 0.005),
        "dtaci": lambda x: C.run_dtaci(x, 0.05, 20, [0.001, 0.01], 2.0),
        "pool": lambda x: C.run_pool_aci(x, pool, 0.05, 0.005, 60),
    }
    t = 70
    for name, f in runners.items():
        q1 = f(s)
        s2 = s.copy(); s2[t:] = 99.0               # change today's and future outcomes
        q2 = f(s2)
        assert np.allclose(q1[: t + 1], q2[: t + 1]), name


def test_gap_estimate_detects_mismatch():
    rng = np.random.default_rng(5)
    same = pd.DataFrame({"market": np.repeat(list("ABCD"), 500), "s_zs": rng.normal(size=2000)})
    diff = same.copy(); diff.loc[diff["market"] == "D", "s_zs"] += 2
    assert gap_estimate(diff) > gap_estimate(same) + 0.1


def _toy(seed=6):
    rng = np.random.default_rng(seed)
    eps, rows = [], []
    for m in ["A", "B", "C", "D", "E", "F"]:
        for y in range(2005, 2010):
            for qq in range(4):
                tau = pd.Timestamp(f"{y}-{3*qq+1:02d}-01")
                eid = f"{m}|{tau.date()}"
                risk = rng.normal()
                eps.append({"episode_id": eid, "market": m, "asset": m, "tau": tau,
                            "window_end": tau + pd.offsets.QuarterEnd(0), "xs_atypicality": risk,
                            "ks_mean": rng.normal(), "mmd": rng.normal(), "rho_lr": np.nan, "rho_gbm": np.nan,
                            "abs_log_vol_ratio": rng.normal(), "corr_src": rng.normal()})
                for k in range(50):
                    rows.append({"episode_id": eid, "market": m, "s_zs": rng.normal(0, np.exp(0.5 * risk)), "s_in": rng.normal()})
    return pd.DataFrame(eps), pd.DataFrame(rows)


def test_entry_context_excludes_target_and_future():
    ep, sc = _toy()
    cut = pd.Timestamp("2008-01-01")
    ep_y = ep[ep["window_end"] < cut]
    sc_y = sc[sc["episode_id"].isin(ep_y["episode_id"])]
    ctx = EntryContext(2008, "C", ep_y, sc_y, CFG)
    assert not (ctx.ep["market"] == "C").any() and not (ctx.sc["market"] == "C").any()
    assert (ctx.ep["window_end"] < cut).all()
    assert ctx.stratifier == "xs_atypicality"           # the only candidate that drives score dispersion
