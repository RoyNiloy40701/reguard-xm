"""Nested leave-one-market-out pseudo-transfer episodes (blueprint 7.2, 7.5).

Episode e = (asset a of held-out market j, quarterly window tau):
  * zero-shot model: bootstrap GBM ensemble trained on all markets except j, data before the year's
    first window (annual refit; embargo before the calibration block; purged targets);
  * in-domain comparator: the same model trained on j's own history (same cut-off);
  * intervals: split-conformal on each model's own calibration block (source pool for zero-shot);
  * labels: degradation of zero-shot vs in-domain on window tau (rows whose target ends after tau are
    dropped, so no label looks past its window);
  * signals: label-free, computed at tau from the target's covariates in the pre-window and the
    source reference distribution.
The true target is never touched: everything here uses development data of source markets only.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from reguard.data.splits import train_mask
from reguard.evaluation.metrics import spearman
from reguard.models.baselines import GBMEnsemble
from reguard.models.preprocess import TrainScaler
from reguard.tre import features as F

EM_CLASSES = {"EM", "Frontier", "Standalone"}


def cvar(losses: np.ndarray, q: float = 0.90) -> float:
    x = np.sort(losses[np.isfinite(losses)])[::-1]
    k = max(1, int(np.ceil(len(x) * (1 - q))))
    return float(x[:k].mean()) if len(x) else np.nan


def window_labels(r: pd.DataFrame, alpha: float) -> dict:
    """Degradation labels for one episode from its window rows."""
    y = r["y"].to_numpy()
    out = {}
    for k in ("zs", "in"):
        cov = float(np.mean((y >= r[f"{k}_lo"]) & (y <= r[f"{k}_hi"])))
        out[f"cov_{k}"] = cov
        out[f"ic_{k}"] = spearman(r[f"{k}_mean"], y)
        strat = np.where(r[f"{k}_mean"].to_numpy() > 0, 1.0, 0.0) * r["next_ret"].to_numpy()
        out[f"cvar_{k}"] = cvar(-strat)
        out[f"mse_{k}"] = float(np.mean((y - r[f"{k}_mean"]) ** 2))
    out["g_cal"] = abs(out["cov_zs"] - (1 - alpha)) - abs(out["cov_in"] - (1 - alpha))
    out["g_ic"] = out["ic_in"] - out["ic_zs"] if np.isfinite(out["ic_in"]) and np.isfinite(out["ic_zs"]) else np.nan
    out["g_cvar"] = out["cvar_zs"] - out["cvar_in"]
    out["g_mse"] = float(np.log(out["mse_zs"] / out["mse_in"])) if out["mse_in"] > 0 else np.nan
    return out


def _conformal(model, Xc, yc, alpha):
    res = yc - model.predict(Xc)
    res = res[np.isfinite(res)]
    if len(res) < 20:
        return np.nan, np.nan
    return float(np.quantile(res, alpha / 2)), float(np.quantile(res, 1 - alpha / 2))


def generate(df: pd.DataFrame, plan: list[dict], cfg: dict, log=print, markets: list[str] | None = None):
    """Returns (episodes DataFrame, episode-rows DataFrame)."""
    ec, feats, target, h = cfg["episodes"], cfg["features"], cfg["target"], cfg["horizon"]
    alpha = ec["alpha"]
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    df["next_ret"] = df.groupby("asset")["ret"].shift(-1)
    cal = pd.DatetimeIndex(sorted(df["date"].unique()))
    pos = pd.Series(np.arange(len(cal)), index=cal)
    sfeat = ec["shift_features"]
    ret_sum_all = df.groupby("date")["ret"].sum()
    ret_cnt_all = df.groupby("date")["ret"].count()
    ret_sum_mk = df.groupby(["market", "date"])["ret"].sum()
    ret_cnt_mk = df.groupby(["market", "date"])["ret"].count()
    tgt_end = f"tgt_end_{h}d"
    eps_rows, ep_list = [], []
    mk_all = sorted(df["market"].unique()) if markets is None else markets
    for yp in plan:
        t_year = time.time()
        fit_m = train_mask(df, yp["fit_window"], [h]) & df[target].notna()
        cal_m = (df["date"] >= yp["cal_start"]) & train_mask(df, yp["w0"], [h]) & df[target].notna()
        yr_m = (df["date"] >= yp["eval_start"]) & (df["date"] <= yp["eval_end"])
        n_ep = 0
        for j in mk_all:
            is_j = df["market"] == j
            if not (yr_m & is_j).any() or (fit_m & is_j).sum() < ec["min_indomain_rows"]:
                continue
            mc = ec["model"]
            src_fit, src_cal = fit_m & ~is_j, cal_m & ~is_j
            zs = GBMEnsemble(mc["members"], cfg["seed"], mc["block"], mc["gbm"]).fit(
                df.loc[src_fit, feats], df.loc[src_fit, target].to_numpy(), df.loc[src_fit])
            ind = GBMEnsemble(mc["members"], cfg["seed"], mc["block"], mc["gbm"]).fit(
                df.loc[fit_m & is_j, feats], df.loc[fit_m & is_j, target].to_numpy(), df.loc[fit_m & is_j])
            zlo, zhi = _conformal(zs, df.loc[src_cal, feats], df.loc[src_cal, target].to_numpy(), alpha)
            ilo, ihi = _conformal(ind, df.loc[cal_m & is_j, feats], df.loc[cal_m & is_j, target].to_numpy(), alpha)
            if not np.isfinite([zlo, zhi, ilo, ihi]).all():
                continue
            # source reference distribution (fixed per year and held-out market)
            ref_idx = np.flatnonzero(src_cal.to_numpy())
            rng = np.random.default_rng(cfg["seed"] + yp["year"])
            if len(ref_idx) > ec["source_ref_rows"]:
                ref_idx = np.sort(rng.choice(ref_idx, ec["source_ref_rows"], replace=False))
            ref = df.iloc[ref_idx]
            sc = TrainScaler().fit(ref[sfeat])
            A = sc.transform(ref[sfeat]).astype(float)
            # source-pool average daily return EXCLUDING market j (for corr_src)
            sj = ret_sum_mk.loc[j].reindex(ret_sum_all.index).fillna(0)
            cj = ret_cnt_mk.loc[j].reindex(ret_cnt_all.index).fillna(0)
            src_avg_ret = (ret_sum_all - sj) / (ret_cnt_all - cj).replace(0, np.nan)
            src_members = zs.predict_members(ref[feats])
            src_y = ref[target].to_numpy()
            src_ret = df.loc[src_cal, "ret"].to_numpy()
            for _, w in yp["windows"].iterrows():
                tau = w["eval_start"]
                p_tau = pos[tau]
                pre_start = cal[max(0, p_tau - ec["pre_window_days"])]
                for a in df.loc[is_j, "asset"].unique():
                    is_a = df["asset"] == a
                    win = is_a & (df["date"] >= tau) & (df["date"] <= w["eval_end"]) & df[target].notna() \
                        & (df[tgt_end] <= w["eval_end"])
                    pre = is_a & (df["date"] >= pre_start) & (df["date"] < tau)
                    if win.sum() < ec["min_window_rows"] or pre.sum() < 20:
                        continue
                    W = df.loc[win]
                    zm, im = zs.predict_members(W[feats]), ind.predict_members(W[feats])
                    r = pd.DataFrame({"date": W["date"].to_numpy(), "asset": a, "market": j, "y": W[target].to_numpy(),
                                      "next_ret": W["next_ret"].to_numpy(),
                                      "zs_mean": zm.mean(1), "zs_sd": zm.std(1), "in_mean": im.mean(1), "in_sd": im.std(1)})
                    r["zs_lo"], r["zs_hi"] = r["zs_mean"] + zlo, r["zs_mean"] + zhi
                    r["in_lo"], r["in_hi"] = r["in_mean"] + ilo, r["in_mean"] + ihi
                    eid = f"{a}|{tau.date()}"
                    r["episode_id"] = eid
                    lab = window_labels(r, alpha)
                    # ---- label-free signals at tau
                    P = df.loc[pre]
                    B = sc.transform(P[sfeat]).astype(float)
                    now = np.flatnonzero((~is_j & (df["date"] >= pre_start) & (df["date"] < tau)).to_numpy())
                    if len(now) > ec["source_ref_rows"]:
                        now = np.sort(rng.choice(now, ec["source_ref_rows"], replace=False))
                    C = sc.transform(df.iloc[now][sfeat]).astype(float)
                    sig = F.shift_signals(A, C, B, seed=cfg["seed"])
                    allnow = ~is_j & (df["date"] >= pre_start) & (df["date"] < tau)
                    sig["xs_atypicality"] = F.xs_atypicality(sc.transform(df.loc[allnow, sfeat]).astype(float),
                                                             df.loc[allnow, "date"].to_numpy(), B, P["date"].to_numpy())
                    sig.update(F.model_signals(src_members, zs.predict_members(P[feats]), src_y))
                    hist = is_a & (df["date"] < tau) & (df["date"] >= cal[max(0, p_tau - 252)])
                    tr_ret = df.loc[hist, "ret"].to_numpy()
                    sig["transfer_risk"] = F.transfer_risk(A, B, src_ret, tr_ret)
                    sa = src_avg_ret.reindex(df.loc[hist, "date"]).to_numpy()
                    ok = np.isfinite(sa) & np.isfinite(tr_ret)
                    sig["corr_src"] = float(np.corrcoef(sa[ok], tr_ret[ok])[0, 1]) if ok.sum() > 30 else np.nan
                    vt, vs = P["vol_60d"].mean(), ref["vol_60d"].mean()
                    sig["abs_log_vol_ratio"] = float(abs(np.log(vt / vs))) if vt > 0 and vs > 0 else np.nan
                    sig["liq_gap"] = float(P["logdv_20d"].mean() - ref["logdv_20d"].mean())
                    sig["em_flag"] = float(P["msci_class"].iloc[0] in EM_CLASSES)
                    ep_list.append({"episode_id": eid, "asset": a, "market": j, "year": yp["year"],
                                    "window_id": w["window_id"], "tau": tau, "window_end": w["eval_end"],
                                    "region": P["region"].iloc[0], "msci_class": P["msci_class"].iloc[0],
                                    "n_rows": int(win.sum()), "zs_trees": zs.best_iter_, "in_trees": ind.best_iter_,
                                    "backend": zs.backend, **lab, **sig})
                    eps_rows.append(r)
                    n_ep += 1
        log(f"  episodes {yp['year']}: {n_ep:4d}  ({time.time()-t_year:5.1f}s)")
    return pd.DataFrame(ep_list), (pd.concat(eps_rows, ignore_index=True) if eps_rows else pd.DataFrame())
