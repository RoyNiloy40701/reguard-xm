"""Label-free transfer signals for one pseudo-transfer episode (N1 features and single-signal baselines).

Inputs are only: source-pool covariates (training calibration block), the target's covariates in the
pre-window before tau, source-model outputs on both, and source labels (allowed: they are source data).
No target forward return (label) is used anywhere in this module.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, stats
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def mmd_rbf(A: np.ndarray, B: np.ndarray, max_rows: int = 1000, seed: int = 0) -> float:
    """Unbiased squared MMD with an RBF kernel (median-distance bandwidth)."""
    rng = np.random.default_rng(seed)
    if len(A) > max_rows:
        A = A[rng.choice(len(A), max_rows, replace=False)]
    if len(B) > max_rows:
        B = B[rng.choice(len(B), max_rows, replace=False)]
    Z = np.vstack([A, B])
    sq = np.sum(Z**2, 1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)
    med = np.median(D[np.triu_indices_from(D, 1)])
    K = np.exp(-D / max(med, 1e-12))
    n, m = len(A), len(B)
    Kaa, Kbb, Kab = K[:n, :n], K[n:, n:], K[:n, n:]
    return float((Kaa.sum() - np.trace(Kaa)) / (n * (n - 1)) + (Kbb.sum() - np.trace(Kbb)) / (m * (m - 1))
                 - 2 * Kab.mean())


def mahalanobis_mean(A: np.ndarray, B: np.ndarray) -> float:
    """Mean squared Mahalanobis distance of target rows B to the source distribution A, divided by d."""
    lw = LedoitWolf().fit(A)
    P = lw.precision_
    D = B - lw.location_
    return float(np.mean(np.einsum("ij,jk,ik->i", D, P, D)) / A.shape[1])


def domain_auc(A: np.ndarray, B: np.ndarray, seed: int = 0) -> float:
    """Cross-validated AUC of a logistic source-vs-target classifier (0.5 = indistinguishable)."""
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    k = int(min(5, len(B)))
    if k < 2:
        return np.nan
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        m = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced").fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, p))


def ks_stats(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    ks = [stats.ks_2samp(A[:, i], B[:, i]).statistic for i in range(A.shape[1])]
    return float(np.mean(ks)), float(np.max(ks))


def gaussian_w2(A: np.ndarray, B: np.ndarray) -> float:
    """2-Wasserstein distance between Gaussian fits of A and B (multivariate or 1-D)."""
    A, B = np.atleast_2d(A.T).T, np.atleast_2d(B.T).T
    m1, m2 = A.mean(0), B.mean(0)
    if A.shape[1] == 1:
        return float(np.sqrt((m1 - m2) @ (m1 - m2) + (A.std() - B.std()) ** 2))
    S1 = LedoitWolf().fit(A).covariance_
    S2 = LedoitWolf().fit(B).covariance_
    r = linalg.sqrtm(S2)
    cross = linalg.sqrtm(r @ S1 @ r)
    val = (m1 - m2) @ (m1 - m2) + np.trace(S1 + S2 - 2 * np.real(cross))
    return float(np.sqrt(max(val, 0.0)))


def transfer_risk(A: np.ndarray, B: np.ndarray, ret_src: np.ndarray, ret_tgt: np.ndarray) -> float:
    """Cao et al. (2023)-style transfer risk = input transport + output transport (Gaussian W2),
    computed with PRE-DEPLOYMENT data only: target covariates and the target's past daily returns
    (annualised) instead of the target's test-period returns used in the original paper."""
    w_in = gaussian_w2(A, B) / np.sqrt(A.shape[1])
    rs = ret_src[np.isfinite(ret_src)]
    rt = ret_tgt[np.isfinite(ret_tgt)]
    if len(rs) < 20 or len(rt) < 20:
        return np.nan
    w_out = np.sqrt((252 * (rs.mean() - rt.mean())) ** 2 + (np.sqrt(252) * (rs.std() - rt.std())) ** 2)
    return float(w_in + w_out)


def cka_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Label-free CKA-style similarity: alignment of source and target feature covariance matrices.
    (Choi et al. compare hidden representations of a US and a LOCAL model on the same inputs; a local
    model needs target labels, so we use the label-free feature-space analogue.)"""
    Ca = np.cov(A, rowvar=False)
    Cb = np.cov(B, rowvar=False)
    den = np.linalg.norm(Ca) * np.linalg.norm(Cb)
    return float(np.sum(Ca * Cb) / den) if den > 0 else np.nan


def model_signals(src_members: np.ndarray, tgt_members: np.ndarray, src_y: np.ndarray) -> dict:
    """Ensemble-based signals. members: rows x K predictions. src_y: source labels on the source rows."""
    s_mu, s_sd = src_members.mean(1), src_members.std(1)
    t_mu, t_sd = tgt_members.mean(1), tgt_members.std(1)
    eps = 1e-12
    s_conf, t_conf = np.abs(s_mu) / (s_sd + eps), np.abs(t_mu) / (t_sd + eps)
    p_up = (tgt_members > 0).mean(1)
    ent = -(p_up * np.log(np.clip(p_up, eps, 1)) + (1 - p_up) * np.log(np.clip(1 - p_up, eps, 1)))
    # ATC (Garg et al. 2022): threshold t so that P_src(conf > t) equals source directional accuracy
    ok = np.isfinite(src_y)
    acc_s = float(np.mean(np.sign(s_mu[ok]) == np.sign(src_y[ok]))) if ok.any() else np.nan
    t = np.quantile(s_conf, 1 - acc_s) if np.isfinite(acc_s) else np.nan
    atc_tgt = float(np.mean(t_conf > t)) if np.isfinite(t) else np.nan
    return {"ens_disagree": float(t_sd.mean()),
            "disagree_ratio": float(t_sd.mean() / (s_sd.mean() + eps)),
            "pred_entropy": float(ent.mean()),
            "atc_gap": acc_s - atc_tgt if np.isfinite(atc_tgt) else np.nan,
            "doc": float(s_conf.mean() - t_conf.mean())}


def xs_atypicality(C_now: np.ndarray, c_dates: np.ndarray, B: np.ndarray, b_dates: np.ndarray) -> float:
    """Mean squared z-score of the target relative to the SAME-DATE cross-section of source markets
    (per feature, per date), averaged over the pre-window. 1 = as typical as a random source market."""
    import pandas as pd
    Cd = pd.DataFrame(C_now).groupby(c_dates)
    mu, sd = Cd.mean(), Cd.std().replace(0, np.nan)
    Bd = pd.DataFrame(B, index=b_dates)
    common = Bd.index.intersection(mu.index)
    if len(common) < 10:
        return np.nan
    z = (Bd.loc[common] - mu.loc[common]) / sd.loc[common]
    return float(np.nanmean(z.to_numpy() ** 2))


def shift_signals(A_train: np.ndarray, C_now: np.ndarray, B: np.ndarray, seed: int = 0) -> dict:
    """A_train: source rows from the training calibration block; C_now: source rows on the same dates as
    the target pre-window; B: target pre-window rows. All standardized with the training scaler."""
    ks_mean, ks_max = ks_stats(C_now, B)
    return {"mmd": mmd_rbf(C_now, B, seed=seed), "mahalanobis": mahalanobis_mean(A_train, B),
            "dc_auc": domain_auc(C_now, B, seed=seed), "ks_mean": ks_mean, "ks_max": ks_max,
            "cka_sim": cka_similarity(C_now, B)}
