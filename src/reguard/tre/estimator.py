"""Transfer Reliability Estimator (TRE) and its honest evaluation (E2, H1).

TRE: rho = calibrated P(transfer failure | label-free signals).
  * 'lr' : standardised logistic regression (primary; simplicity is deliberate, episodes are few and dependent)
  * 'gbm': monotone-constrained gradient boosting (constraints = pre-registered signal orientation)
  Both are calibrated with isotonic regression on out-of-fold scores (GroupKFold by market).

Evaluation is time-ordered AND market-held-out: an episode of market j in year Y is scored by a TRE
trained only on episodes that (a) ended before year Y and (b) belong to markets outside j's fold.
The failure threshold g* (upper tercile) is computed on those training episodes only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def market_folds(markets: list[str], k: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    ms = sorted(markets)
    perm = rng.permutation(len(ms))
    return {ms[i]: int(r % k) for r, i in enumerate(perm)}


def _base(kind: str, cols: list[str], orient: dict, cfg: dict, seed: int):
    if kind == "lr":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=cfg["lr_C"], max_iter=2000))
    g = cfg["gbm"]
    return HistGradientBoostingClassifier(max_iter=g["max_iter"], learning_rate=g["lr"], max_leaf_nodes=g["max_leaf_nodes"],
                                          min_samples_leaf=g["min_samples_leaf"], monotonic_cst=[int(orient[c]) for c in cols],
                                          early_stopping=False, random_state=seed)


def fit_tre(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, kind: str, orient: dict, cfg: dict, seed: int):
    """Fit base model + isotonic calibrator (on out-of-fold scores grouped by market)."""
    cols = list(X.columns)
    oof = np.full(len(y), np.nan)
    k = min(cfg["calib_folds"], len(np.unique(groups)))
    for tr, te in GroupKFold(k).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = _base(kind, cols, orient, cfg, seed).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    ok = np.isfinite(oof)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof[ok], y[ok])
    base = _base(kind, cols, orient, cfg, seed).fit(X, y)
    return base, iso


def predict_tre(model, X: pd.DataFrame) -> np.ndarray:
    base, iso = model
    return iso.predict(base.predict_proba(X)[:, 1])


def oriented(ep: pd.DataFrame, orient: dict) -> pd.DataFrame:
    """Single-signal risk scores: larger = likelier failure (sign fixed in the config, never from data)."""
    return pd.DataFrame({c: ep[c] * s for c, s in orient.items()}, index=ep.index)


def run_evaluation(ep: pd.DataFrame, cfg: dict, labels: list[str], seed: int, log=print) -> pd.DataFrame:
    """Out-of-sample TRE scores for every test episode and label. Returns long DataFrame."""
    tc = cfg["tre"]
    orient = tc["signals"]
    sig_cols = list(orient)
    folds = market_folds(ep["market"].unique().tolist(), tc["market_folds"], seed)
    ep = ep.assign(fold=ep["market"].map(folds))
    years = sorted(ep["year"].unique())
    out = []
    for Y in years:
        pool = ep[ep["window_end"] < pd.Timestamp(f"{Y}-01-01")]
        if len(pool) < tc["min_train_episodes"]:
            continue
        for f in range(tc["market_folds"]):
            test = ep[(ep["year"] == Y) & (ep["fold"] == f)]
            train = pool[pool["fold"] != f]
            if test.empty:
                continue
            for lab in labels:
                tr = train[train[lab].notna()]
                te = test[test[lab].notna()]
                if te.empty or len(tr) < 50:
                    continue
                g_star = float(np.quantile(tr[lab], cfg["labels"]["failure_quantile"]))
                ytr = (tr[lab] > g_star).to_numpy(int)
                rec = te[["episode_id", "asset", "market", "year", "tau", "fold", lab]].rename(columns={lab: "g"})
                rec["label"], rec["g_star"], rec["y_fail"] = lab, g_star, (te[lab] > g_star).to_numpy(int)
                for kind in ("lr", "gbm"):
                    mdl = fit_tre(tr[sig_cols], ytr, tr["market"].to_numpy(), kind, orient, tc, seed)
                    rec[f"rho_{kind}"] = predict_tre(mdl, te[sig_cols])
                    if kind == "lr" and lab == labels[0] and f == 0:
                        coefs = mdl[0][-1].coef_[0]
                        log(f"    {Y} LR coefs (std units): " + ", ".join(f"{c}={v:+.2f}" for c, v in zip(sig_cols, coefs)))
                rec = pd.concat([rec, oriented(te, orient).add_prefix("sig_")], axis=1)
                out.append(rec)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _auc(y, s):
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], s[ok]))


def score_table(R: pd.DataFrame, orient: dict) -> pd.DataFrame:
    rows = []
    for lab, g in R.groupby("label"):
        y = g["y_fail"].to_numpy()
        cand = {"TRE-LR": g["rho_lr"], "TRE-GBM": g["rho_gbm"], **{c: g[f"sig_{c}"] for c in orient}}
        for name, s in cand.items():
            s = s.to_numpy(float)
            ok = np.isfinite(s)
            rows.append({"label": lab, "scorer": name, "n": int(ok.sum()), "fail_rate": float(y[ok].mean()),
                         "auroc": _auc(y, s), "auprc": float(average_precision_score(y[ok], s[ok])) if ok.sum() > 20 else np.nan,
                         "spearman_g": float(stats.spearmanr(s[ok], g["g"].to_numpy()[ok]).statistic) if ok.sum() > 20 else np.nan})
    return pd.DataFrame(rows)


def h1_test(R: pd.DataFrame, label: str, tre_col: str, orient: dict, n_boot: int, seed: int) -> dict:
    """AUROC(TRE) - AUROC(best single signal), best chosen on the same test episodes (conservative),
    95% CI by resampling whole quarters (all markets' episodes in a quarter move together)."""
    g = R[R["label"] == label].reset_index(drop=True)
    y = g["y_fail"].to_numpy()
    singles = {c: _auc(y, g[f"sig_{c}"].to_numpy(float)) for c in orient}
    best = max((k for k in singles if np.isfinite(singles[k])), key=lambda k: singles[k])
    t = g[tre_col].to_numpy(float)
    b = g[f"sig_{best}"].to_numpy(float)
    q = g["tau"].to_numpy()
    uq = np.unique(q)
    idx_by_q = {u: np.flatnonzero(q == u) for u in uq}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_q[u] for u in rng.choice(uq, len(uq), replace=True)])
        d = _auc(y[idx], t[idx]) - _auc(y[idx], b[idx])
        if np.isfinite(d):
            diffs.append(d)
    diffs = np.array(diffs)
    return {"label": label, "tre": tre_col, "auroc_tre": _auc(y, t), "best_single": best, "auroc_best_single": singles[best],
            "diff": _auc(y, t) - singles[best], "ci_lo": float(np.quantile(diffs, 0.025)),
            "ci_hi": float(np.quantile(diffs, 0.975)), "n_episodes": len(g), "n_quarters": len(uq)}


def reliability_bins(R: pd.DataFrame, label: str, col: str, bins: int = 10) -> pd.DataFrame:
    g = R[R["label"] == label]
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(g[col], edges) - 1, 0, bins - 1)
    return g.assign(bin=b).groupby("bin").agg(mean_rho=(col, "mean"), fail_rate=("y_fail", "mean"), n=("y_fail", "size")).reset_index()
