"""Weekly portfolio environment for PPO (Gymnasium API) and the matching backtest proposer.

Observation (decision day t, past-only): for each ETF the configured features, z-scored across the ETFs available
at t (clipped to +-5, missing -> 0), an availability flag, and the current weights (ETFs + cash).
Action: n + 1 logits -> masked softmax -> weights over available ETFs and cash (long-only, fully allocated).
Reward: 100 x log(1 + net weekly return) (costs on the traded weight at the rebalance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:          # allows testing the logic without gymnasium
    gym, spaces, _Base = None, None, object


class ObsBuilder:
    def __init__(self, mkt, features: list[str]):
        self.mkt = mkt
        self.F = np.stack([mkt.feature(f).to_numpy(dtype=float) for f in features], axis=-1)   # T x n x k
        self.A = mkt.available.to_numpy()
        self.n, self.k = self.F.shape[1], self.F.shape[2]

    def obs(self, t: int, w_risky: np.ndarray) -> np.ndarray:
        f = self.F[t].copy()
        a = self.A[t]
        if a.any():
            mu = np.nanmean(np.where(a[:, None], f, np.nan), axis=0)
            sd = np.nanstd(np.where(a[:, None], f, np.nan), axis=0)
            sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
            f = np.clip((f - mu) / sd, -5, 5)
        f = np.where(a[:, None] & np.isfinite(f), f, 0.0)
        cash = max(0.0, 1.0 - float(w_risky.sum()))
        return np.concatenate([f.ravel(), a.astype(float), w_risky, [cash]]).astype(np.float32)

    @property
    def dim(self) -> int:
        return self.n * self.k + self.n + self.n + 1


def masked_softmax(logits: np.ndarray, avail: np.ndarray) -> np.ndarray:
    """Weights over n ETFs + cash; unavailable ETFs get 0. Returns risky weights (cash = 1 - sum)."""
    z = np.asarray(logits, float).copy()
    z[:-1] = np.where(avail, z[:-1], -np.inf)
    z = z - np.max(z)
    e = np.exp(np.clip(z, -50, 50)) * np.r_[avail, True]
    p = e / e.sum()
    return p[:-1]


class PortfolioEnv(_Base):
    metadata = {"render_modes": []}

    def __init__(self, mkt, ob: ObsBuilder, start: str, end: str, episode_weeks: int, every: int, cost_bps: float,
                 seed: int = 0):
        idx = np.flatnonzero((mkt.dates >= pd.Timestamp(start)) & (mkt.dates <= pd.Timestamp(end)))
        self.decision = idx[::every]
        self.decision = self.decision[self.decision + every < len(mkt.dates)]
        self.mkt, self.ob, self.every = mkt, ob, every
        self.R = np.nan_to_num(mkt.ret.to_numpy())
        self.rf = mkt.rf.to_numpy()
        self.cost = cost_bps / 1e4
        self.ep_len = episode_weeks
        self.rng = np.random.default_rng(seed)
        if spaces is not None:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(ob.dim,), dtype=np.float32)
            self.action_space = spaces.Box(-10.0, 10.0, shape=(ob.n + 1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        hi = max(1, len(self.decision) - self.ep_len)
        self.i0 = int(self.rng.integers(0, hi))
        self.i = self.i0
        self.w = np.zeros(self.ob.n)
        return self.ob.obs(self.decision[self.i], self.w), {}

    def step(self, action):
        t = self.decision[self.i]
        w_new = masked_softmax(action, self.ob.A[t])
        traded = float(np.abs(w_new - self.w).sum())
        cash = 1.0 - float(w_new.sum())
        growth = 1.0
        for d in range(t + 1, t + 1 + self.every):
            growth *= 1.0 + float(w_new @ self.R[d] + cash * self.rf[d])
        ret = growth - 1.0 - self.cost * traded
        self.w = w_new
        self.i += 1
        done = (self.i - self.i0) >= self.ep_len or self.i >= len(self.decision)
        obs = self.ob.obs(self.decision[min(self.i, len(self.decision) - 1)], self.w)
        return obs, float(100.0 * np.log(max(1.0 + ret, 1e-6))), bool(done), False, {"ret": ret, "traded": traded}


def ppo_proposer(model, ob: ObsBuilder):
    """Backtest proposer from a trained SB3 model (deterministic action)."""
    def f(t, w_prev, mu, Sigma, avail):
        a, _ = model.predict(ob.obs(t, w_prev), deterministic=True)
        return masked_softmax(a, avail)
    return f
