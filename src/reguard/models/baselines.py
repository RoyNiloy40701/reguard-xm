"""E1 baseline forecasters with one interface: fit(X_df, y, meta) / predict(X_df, meta).

X_df: feature DataFrame; y: target array; meta: DataFrame with at least `market`, `date`, `ret`.
Models are deterministic given `seed`. GBM uses LightGBM when installed, otherwise scikit-learn's
HistGradientBoostingRegressor (the backend actually used is recorded in `backend`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from reguard.models.preprocess import TrainScaler


def time_split(meta, n: int, val_frac: float = 0.15, gap: int = 20):
    """Purged, time-ordered split of training rows: (train_idx, val_idx) as integer positions.

    Validation = rows on the last `val_frac` of unique training dates; rows on the `gap` dates
    before it are dropped. Random or leave-one-out validation is NOT valid with overlapping
    h-day targets (it is almost in-sample and selects too little regularisation).
    """
    dates = meta["date"].to_numpy() if meta is not None and "date" in meta else np.arange(n)
    ud = np.unique(dates)
    v0 = ud[int(len(ud) * (1 - val_frac))]
    cut = ud[max(0, np.searchsorted(ud, v0) - gap)]
    return np.where(dates < cut)[0], np.where(dates >= v0)[0]


class HistMean:
    """Training-pool mean of the target (the 'historical mean' benchmark of OOS R^2)."""

    name = "hist_mean"
    backend = "numpy"

    def fit(self, X, y, meta=None):
        self.mu_ = float(np.nanmean(y))
        return self

    def predict(self, X, meta=None):
        return np.full(len(X), self.mu_)


class AR1:
    """Per-market AR(1) on daily log returns; forecast of the h-day sum (in-domain only)."""

    name = "ar1"
    backend = "numpy"

    def __init__(self, horizon: int, rf_col: str | None = "rf_daily"):
        self.h, self.rf_col = horizon, rf_col

    def fit(self, X, y, meta):
        self.par_ = {}
        for m, g in meta.groupby("market"):
            r = g.sort_values("date")["ret"].to_numpy(dtype=float)
            r = r[np.isfinite(r)]
            if len(r) < 100:
                continue
            x0, x1 = r[:-1], r[1:]
            mu = r.mean()
            phi = np.clip(np.cov(x0, x1)[0, 1] / np.var(x0), -0.5, 0.5)
            self.par_[m] = (mu, phi)
        return self

    def predict(self, X, meta):
        out = np.full(len(meta), np.nan)
        k = np.arange(1, self.h + 1)
        for m, (mu, phi) in self.par_.items():
            idx = np.where(meta["market"].to_numpy() == m)[0]
            r_t = np.nan_to_num(meta["ret"].to_numpy(dtype=float)[idx])
            s = (phi ** k).sum()
            pred = self.h * mu + (r_t - mu) * s
            if self.rf_col and self.rf_col in meta:
                pred = pred - self.h * np.nan_to_num(meta[self.rf_col].to_numpy(dtype=float)[idx])
            out[idx] = pred
        return out


class Ridge:
    """Ridge with the penalty chosen on a purged time-ordered validation block, then refit on all rows."""

    name = "ridge"
    backend = "sklearn"

    def __init__(self, alphas=None):
        self.alphas = np.logspace(-1, 6, 15) if alphas is None else alphas

    def fit(self, X, y, meta=None):
        from sklearn.linear_model import Ridge as _R
        y = np.asarray(y, float)
        tr, va = time_split(meta, len(y))
        sc = TrainScaler().fit(X.iloc[tr])
        Ztr, Zva = sc.transform(X.iloc[tr]), sc.transform(X.iloc[va])
        errs = [np.mean((_R(alpha=a).fit(Ztr, y[tr]).predict(Zva) - y[va]) ** 2) for a in self.alphas]
        self.alpha_ = float(self.alphas[int(np.argmin(errs))])
        self.sc_ = TrainScaler().fit(X)
        self.m_ = _R(alpha=self.alpha_).fit(self.sc_.transform(X), y)
        return self

    def predict(self, X, meta=None):
        return self.m_.predict(self.sc_.transform(X))


class GBM:
    name = "gbm"

    def __init__(self, seed: int = 0, n_estimators: int = 300, lr: float = 0.03, leaves: int = 15,
                 min_leaf: int = 200, n_jobs: int = -1):
        self.p = dict(seed=seed, n=n_estimators, lr=lr, leaves=leaves, min_leaf=min_leaf, n_jobs=n_jobs)
        try:
            import lightgbm  # noqa: F401
            self.backend = "lightgbm"
        except ImportError:
            self.backend = "sklearn-histgb"

    def _new(self, n_trees):
        p = self.p
        if self.backend == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor(n_estimators=n_trees, learning_rate=p["lr"], num_leaves=p["leaves"],
                                     min_child_samples=p["min_leaf"], subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, reg_lambda=1.0, random_state=p["seed"],
                                     n_jobs=p["n_jobs"], verbose=-1, deterministic=True, force_row_wise=True)
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=n_trees, learning_rate=p["lr"], max_leaf_nodes=p["leaves"],
                                             min_samples_leaf=p["min_leaf"], l2_regularization=1.0,
                                             early_stopping=False, random_state=p["seed"])

    def fit(self, X, y, meta=None, sample_weight=None, n_trees: int | None = None):
        """Number of trees chosen by early stopping on a purged time-ordered validation block
        (max n_estimators), then the model is refit on all rows with that number of trees."""
        y = np.asarray(y, float)
        self.cols_ = [c for c in X.columns if X[c].notna().sum() >= 20]   # drop (almost) all-missing features
        Xn = X[self.cols_].to_numpy(dtype=np.float32)
        if n_trees is not None:                          # fixed size (used by ensemble members)
            self.best_iter_ = int(n_trees)
            self.m_ = self._new(self.best_iter_).fit(Xn, y, sample_weight=sample_weight)
            return self
        tr, va = time_split(meta, len(y))
        nmax = self.p["n"]
        if self.backend == "lightgbm":
            import lightgbm as lgb
            m = self._new(nmax).fit(Xn[tr], y[tr], eval_set=[(Xn[va], y[va])],
                                    callbacks=[lgb.early_stopping(50, verbose=False)])
            best = int(m.best_iteration_ or nmax)
        else:
            m = self._new(25)
            m.set_params(warm_start=True)
            best, best_err, n = 1, np.inf, 25
            while n <= nmax:
                m.set_params(max_iter=n).fit(Xn[tr], y[tr])
                err = np.mean((m.predict(Xn[va]) - y[va]) ** 2)
                if err < best_err - 1e-12:
                    best, best_err = n, err
                elif n - best >= 100:
                    break
                n += 25
        self.best_iter_ = max(best, 1)
        self.m_ = self._new(self.best_iter_).fit(Xn, y)
        return self

    def predict(self, X, meta=None):
        return self.m_.predict(X[self.cols_].to_numpy(dtype=np.float32))


class MLPEnsemble:
    """Deep ensemble: `members` small MLPs differing only in seed; prediction = member mean.

    Early stopping uses a TIME-ORDERED validation block (last `val_frac` of training dates, after a
    `gap`-date purge), never a random split: with overlapping h-day targets a random hold-out is almost
    in-sample and stops nothing. Weights from the best validation epoch are kept.
    Member spread is available via predict_members (used later as a disagreement signal in E2).
    """

    name = "mlp_ens"
    backend = "sklearn"

    def __init__(self, members: int = 5, seed: int = 0, hidden=(32, 16), max_rows: int = 60000,
                 val_frac: float = 0.15, gap: int = 20, max_epochs: int = 40, patience: int = 5, alpha: float = 1e-3):
        self.members, self.seed, self.hidden, self.max_rows = members, seed, tuple(hidden), max_rows
        self.val_frac, self.gap, self.max_epochs, self.patience, self.alpha = val_frac, gap, max_epochs, patience, alpha

    def fit(self, X, y, meta=None):
        import copy
        y = np.asarray(y, dtype=float)
        dates = (meta["date"].to_numpy() if meta is not None and "date" in meta
                 else np.arange(len(y)))
        order = np.argsort(dates, kind="stable")
        if len(order) > self.max_rows:                   # cap cost: keep the most recent rows
            order = order[-self.max_rows:]
        X, y, dates = X.iloc[order], y[order], dates[order]
        ud = np.unique(dates)
        v0 = ud[int(len(ud) * (1 - self.val_frac))]
        cut = ud[max(0, np.searchsorted(ud, v0) - self.gap)]
        tr, va = dates < cut, dates >= v0
        self.sc_ = TrainScaler().fit(X[tr])
        Ztr, Zva = self.sc_.transform(X[tr]), self.sc_.transform(X[va])
        self.ymu_, self.ysd_ = float(np.mean(y[tr])), float(np.std(y[tr])) or 1.0
        ytr, yva = (y[tr] - self.ymu_) / self.ysd_, (y[va] - self.ymu_) / self.ysd_
        rng = np.random.default_rng(self.seed)
        self.m_, self.best_epochs_ = [], []
        for _ in range(self.members):
            m = MLPRegressor(hidden_layer_sizes=self.hidden, alpha=self.alpha, learning_rate_init=1e-3,
                             batch_size=512, shuffle=True, random_state=int(rng.integers(1 << 31)))
            best, best_m, wait, be = np.inf, None, 0, 0
            for ep in range(self.max_epochs):
                m.partial_fit(Ztr, ytr)
                loss = float(np.mean((m.predict(Zva) - yva) ** 2)) if len(yva) else 0.0
                if loss < best - 1e-6:
                    best, best_m, wait, be = loss, copy.deepcopy(m), 0, ep + 1
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            self.m_.append(best_m if best_m is not None else m)
            self.best_epochs_.append(be)
        return self

    def predict_members(self, X, meta=None):
        Z = self.sc_.transform(X)
        return np.column_stack([m.predict(Z) * self.ysd_ + self.ymu_ for m in self.m_])

    def predict(self, X, meta=None):
        return self.predict_members(X).mean(axis=1)


class GBMEnsemble:
    """Bootstrap ensemble of GBMs (deep-ensemble analogue for trees).

    The number of trees is chosen once by purged time-ordered early stopping; each member is then
    fit on a moving-block bootstrap of training DATES (block length `block`, implemented as sample
    weights), so members differ the way models trained on other histories would.
    """

    name = "gbm_ens"

    def __init__(self, members: int = 5, seed: int = 0, block: int = 63, gbm: dict | None = None):
        self.members, self.seed, self.block, self.gbm = members, seed, block, dict(gbm or {})

    def fit(self, X, y, meta):
        base = GBM(seed=self.seed, **self.gbm).fit(X, y, meta)
        self.backend, self.best_iter_ = base.backend, base.best_iter_
        dates = meta["date"].to_numpy()
        ud, inv = np.unique(dates, return_inverse=True)
        rng = np.random.default_rng(self.seed)
        nb = int(np.ceil(len(ud) / self.block))
        self.m_ = []
        for k in range(self.members):
            starts = rng.integers(0, max(1, len(ud) - self.block + 1), size=nb)
            cnt = np.bincount((starts[:, None] + np.arange(self.block)).ravel().clip(max=len(ud) - 1), minlength=len(ud))
            w = cnt[inv].astype(float)
            keep = w > 0
            m = GBM(seed=self.seed + k + 1, **self.gbm)
            m.fit(X[keep], np.asarray(y)[keep], None, sample_weight=w[keep], n_trees=self.best_iter_)
            self.m_.append(m)
        return self

    def predict_members(self, X, meta=None):
        return np.column_stack([m.predict(X) for m in self.m_])

    def predict(self, X, meta=None):
        return self.predict_members(X).mean(axis=1)


def make_model(name: str, horizon: int, seed: int, cfg: dict):
    if name == "hist_mean":
        return HistMean()
    if name == "ar1":
        return AR1(horizon)
    if name == "ridge":
        return Ridge()
    if name == "gbm":
        return GBM(seed=seed, **cfg.get("gbm", {}))
    if name == "mlp_ens":
        return MLPEnsemble(seed=seed, **cfg.get("mlp_ens", {}))
    raise ValueError(name)
