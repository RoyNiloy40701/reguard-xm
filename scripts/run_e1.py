"""E1: forecasting baselines and cross-market transfer (development period only; test stays locked).

Usage (from the code folder):
    python scripts/run_e1.py                 # full run
    python scripts/run_e1.py --quick         # 3 years, fast check (~minutes)
Writes results/e1/{report.md, summary_ts.csv, metrics_ts_by_asset.csv, dm_tests.csv, transfer_matrix.csv,
cs_ic.csv, figures/, log.txt, env_info.txt} and results/e1/handoff_e1.zip.
Predictions are saved to data/interim/e1_preds/ (not zipped; reused in Phase 5).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from reguard.data.splits import train_mask  # noqa: E402
from reguard.data.store import PROCESSED_ROOT, load_panel, load_yaml, write_table  # noqa: E402
from reguard.evaluation import metrics as M  # noqa: E402
from reguard.models.baselines import make_model  # noqa: E402
from reguard.models.preprocess import cs_rank  # noqa: E402
from reguard.utils.handoff import CODE_ROOT, PROJECT_ROOT, Tee, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "e1"
PRED_DIR = PROJECT_ROOT / "data" / "interim" / "e1_preds"
WCOLS = ["train_start", "train_end", "embargo_start", "eval_start", "eval_end"]


def load_cfg() -> dict:
    with open(CODE_ROOT / "configs" / "experiment" / "e1.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def year_plan(splits: pd.DataFrame, cal: pd.DatetimeIndex, cal_days: int, embargo: int, first_year=None, years=None):
    """One refit per year: fit block, calibration block, evaluation span (all purged/embargoed)."""
    splits = splits.copy()
    splits["year"] = splits["eval_start"].dt.year
    plan = []
    for y, g in splits.groupby("year"):
        if first_year and y < first_year:
            continue
        w0 = g.iloc[0]
        te = cal.get_loc(w0["train_end"])
        cs = te - cal_days + 1                       # calibration block = last cal_days training dates
        fit_end = cs - embargo - 1                   # embargo between fit and calibration blocks
        if fit_end < 250:
            continue
        plan.append({"year": y, "w0": w0, "fit_window": pd.Series({"train_end": cal[fit_end], "eval_start": cal[cs]}),
                     "cal_start": cal[cs], "train_end": w0["train_end"],
                     "eval_start": g["eval_start"].min(), "eval_end": g["eval_end"].max()})
    return plan[-years:] if years else plan


# ---------------------------------------------------------------------------
# Task TS
# ---------------------------------------------------------------------------
def pools(meta: pd.DataFrame, setting: str, target_market: str | None):
    m = meta["market"]
    if setting == "indomain":
        return m == target_market
    if setting == "lomo":
        return m != target_market
    if setting == "pooled":
        return pd.Series(True, index=meta.index)
    if setting == "us_only":
        return m == "US"
    if setting.startswith("group:"):
        gname = setting.split(":", 1)[1]
        if gname in ("DM", "EM"):
            return meta["msci_class"] == gname
        return meta["region"] == gname
    raise ValueError(setting)


def fit_predict(df, feats, target, model_name, setting, tmask_fit, tmask_cal, emask, cfg, horizon, q):
    """Returns DataFrame (index = eval rows) with pred and quantile columns for one setting/model."""
    per_target = setting in ("indomain", "lomo")
    groups = df.loc[emask, "market"].unique() if per_target else [None]
    out = []
    for tm in groups:
        pool = pools(df, setting, tm)
        fit_rows = tmask_fit & pool & df[target].notna()
        cal_rows = tmask_cal & pool & df[target].notna()
        ev = emask & ((df["market"] == tm) if per_target else True)
        if model_name == "ar1" and setting != "indomain":
            continue
        if fit_rows.sum() < (cfg["min_train_rows_indomain"] if setting == "indomain" else 250) or ev.sum() == 0:
            continue
        mdl = make_model(model_name, horizon, cfg["seed"], cfg)
        mdl.fit(df.loc[fit_rows, feats], df.loc[fit_rows, target].to_numpy(), df.loc[fit_rows])
        res = df.loc[cal_rows, target].to_numpy() - mdl.predict(df.loc[cal_rows, feats], df.loc[cal_rows])
        res = res[np.isfinite(res)]
        p = mdl.predict(df.loc[ev, feats], df.loc[ev])
        o = pd.DataFrame({"pred": p}, index=df.index[ev])
        for qq in q:
            o[f"q{int(qq*100):02d}"] = p + (np.quantile(res, qq) if len(res) > 20 else np.nan)
        o["backend"] = getattr(mdl, "backend", "")
        out.append(o)
    return pd.concat(out) if out else None


def run_ts(cfg, quick: bool, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    tc = cfg["ts"]
    pcfg = load_yaml("processing.yaml")
    df = load_panel(tc["panel"]).sort_values(["date", "asset"]).reset_index(drop=True)
    splits = pd.read_csv(PROCESSED_ROOT / "splits_tierb.csv", parse_dates=WCOLS)
    cal = pd.DatetimeIndex(sorted(df["date"].unique()))
    plan = year_plan(splits, cal, cfg["calibration_days"], pcfg["embargo_days"], years=3 if quick else None)
    feats, target, h, q = tc["features"], tc["target"], tc["horizon"], tc["quantiles"]
    settings = list(tc["settings"]) + [f"group:{g}" for g in tc["groups"]]
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    allp = []
    for setting in settings:
        models = (tc["matrix_models"] if setting.startswith("group:")
                  else tc.get("lomo_models", tc["models"]) if setting == "lomo" else tc["models"])
        for mname in models:
            t0 = time.time()
            parts = []
            for yp in plan:
                fit_m = train_mask(df, yp["fit_window"], [h])
                cal_m = (df["date"] >= yp["cal_start"]) & train_mask(df, yp["w0"], [h])
                ev_m = (df["date"] >= yp["eval_start"]) & (df["date"] <= yp["eval_end"]) & df[target].notna()
                r = fit_predict(df, feats, target, mname, setting, fit_m, cal_m, ev_m, cfg, h, q)
                if r is not None:
                    r["year"] = yp["year"]
                    parts.append(r)
            if not parts:
                continue
            p = pd.concat(parts)
            p["setting"], p["model"] = setting, mname
            p = p.join(df[["date", "asset", "market", "region", "msci_class", target]])
            for c in ("pred", "q05", "q50", "q95", target):
                p[c] = p[c].astype("float32")
            allp.append(p)
            log(f"  TS {setting:24s} {mname:9s} rows={len(p):7d}  {time.time()-t0:6.1f}s  backend={p['backend'].iloc[0]}")
    P = pd.concat(allp, ignore_index=True).rename(columns={target: "y"})
    for c in ("setting", "model", "backend", "asset", "market", "region", "msci_class"):
        P[c] = P[c].astype("category")
    write_table(P.drop(columns=["backend"]), PRED_DIR, "ts_predictions")
    return P, df


def evaluate_ts(P: pd.DataFrame, cfg) -> dict:
    h = cfg["ts"]["horizon"]
    # common sample: rows where the in-domain historical-mean benchmark exists
    bench = P[(P["setting"] == "indomain") & (P["model"] == "hist_mean")][["date", "asset", "pred"]].rename(columns={"pred": "bench"})
    bench["asset"] = pd.Categorical(bench["asset"].astype(str), categories=P["asset"].cat.categories)
    P = P.merge(bench, on=["date", "asset"], how="inner")
    P["se"] = (P["y"] - P["pred"]) ** 2
    P["se_b"] = (P["y"] - P["bench"]) ** 2
    daily = P.groupby(["setting", "model", "date"], observed=True)[["se", "se_b"]].sum().rename(columns={"se": "a", "se_b": "b"})
    daily_reg = P.groupby(["setting", "model", "region", "date"], observed=True)[["se", "se_b"]].sum().rename(columns={"se": "a", "se_b": "b"})
    rows = []
    for (s, m, a), g in P.groupby(["setting", "model", "asset"], observed=True):
        g = g.sort_values("date")
        rows.append({"setting": s, "model": m, "asset": a, "market": g["market"].iloc[0], "region": g["region"].iloc[0],
                     "n": len(g), "ic": M.spearman(g["pred"], g["y"]), "r2_oos": M.oos_r2(g["y"], g["pred"], g["bench"]),
                     "auc": M.dir_auc(g["y"], g["pred"]), "mse": g["se"].mean(),
                     "pinball": np.mean([M.pinball(g["y"], g[c], qq) for c, qq in (("q05", .05), ("q50", .5), ("q95", .95))]),
                     "cov90": float(((g["y"] >= g["q05"]) & (g["y"] <= g["q95"])).mean()),
                     "width90": float((g["q95"] - g["q05"]).mean())})
    by_asset = pd.DataFrame(rows)

    # cross-country rank IC per date (country rotation view)
    cs_rows, summ = [], []
    for (s, m), g in P.groupby(["setting", "model"], observed=True):
        if str(s).startswith("group:"):
            continue
        ic = M.daily_cs_ic(g, "pred", "y", min_n=10)
        mu, t = M.nw_tstat(ic.to_numpy(), lag=h)
        cs_rows.append({"setting": s, "model": m, "cs_ic_mean": mu, "cs_ic_nw_t": t, "days": len(ic)})
        b = by_asset[(by_asset["setting"] == s) & (by_asset["model"] == m)]
        se_day = daily.loc[(s, m)].sort_index()
        r2 = 1 - se_day["a"].sum() / se_day["b"].sum()
        lo, hi = ratio_ci(se_day)
        summ.append({"setting": s, "model": m, "markets": b["market"].nunique(), "pooled_r2_oos": r2,
                     "r2_ci_lo": lo, "r2_ci_hi": hi, "median_ic": b["ic"].median(), "median_r2_oos": b["r2_oos"].median(),
                     "share_r2_pos": float((b["r2_oos"] > 0).mean()), "median_auc": b["auc"].median(),
                     "mean_pinball": b["pinball"].mean(), "mean_cov90": b["cov90"].mean(), "mean_width90": b["width90"].mean()})
    summary = pd.DataFrame(summ).merge(pd.DataFrame(cs_rows), on=["setting", "model"], how="left")

    # Diebold-Mariano: each setting vs in-domain, same model, per asset (HAC lag h-1)
    dm = []
    ind = P[P["setting"] == "indomain"][["date", "asset", "model", "se"]].rename(columns={"se": "se_in"})
    for (s, m), g in P[P["setting"] != "indomain"].groupby(["setting", "model"], observed=True):
        j = g.merge(ind[ind["model"] == m], on=["date", "asset", "model"])
        if j.empty:
            continue
        for a, ga in j.groupby("asset", observed=True):
            d_mean, stat, p = M.dm_test(ga.sort_values("date")["se"], ga.sort_values("date")["se_in"], lag=h - 1)
            dm.append({"setting": s, "model": m, "asset": a, "mean_loss_diff": d_mean, "dm_stat": stat, "p": p})
    dm = pd.DataFrame(dm)

    # transfer matrix: source group -> target region (pooled R^2 and IC with block-bootstrap CI)
    mat = []
    for (s, m), g in P[P["setting"].astype(str).str.startswith("group:")].groupby(["setting", "model"], observed=True):
        for reg, gr in g.groupby("region", observed=True):
            se_day = daily_reg.loc[(s, m, reg)].sort_index()
            lo, hi = ratio_ci(se_day)
            mat.append({"source": s.split(":", 1)[1], "model": m, "target_region": reg, "markets": gr["market"].nunique(),
                        "pooled_r2_oos": 1 - se_day["a"].sum() / se_day["b"].sum(), "ci_lo": lo, "ci_hi": hi,
                        "median_ic": gr.groupby("asset", observed=True).apply(lambda d: M.spearman(d["pred"], d["y"]), include_groups=False).median()})
    return {"by_asset": by_asset, "summary": summary, "dm": dm, "matrix": pd.DataFrame(mat)}


def ratio_ci(se_day: pd.DataFrame, block: int = 63, n_boot: int = 500, seed: int = 0):
    a, b = se_day["a"].to_numpy(), se_day["b"].to_numpy()
    n = len(a)
    if n < 2 * block:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    st = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (st[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    r2 = 1 - a[idx].sum(1) / b[idx].sum(1)
    return float(np.quantile(r2, 0.025)), float(np.quantile(r2, 0.975))


# ---------------------------------------------------------------------------
# Task CS
# ---------------------------------------------------------------------------
def prep_cs(df: pd.DataFrame, feats: list[str], target: str, universe_only: bool) -> pd.DataFrame:
    if universe_only:
        df = df[df["in_universe"]]
    df = df[df[target].notna()].copy()
    R = cs_rank(df, feats)
    R.columns = feats
    R["y_rank"] = df.groupby("date")[target].rank(pct=True) - 0.5
    for c in ["date", "asset", "tgt_end_20d", target]:
        R[c] = df[c]
    return R.rename(columns={target: "y_raw"})


def run_cs(cfg, quick: bool, log) -> pd.DataFrame:
    c = cfg["cs"]
    pcfg = load_yaml("processing.yaml")
    feats = c["features"]
    tb = prep_cs(load_panel("tierb_etf"), feats, c["tierb_target"], False)
    ds = prep_cs(load_panel("dse_equity"), feats, c["dse_target"], True)
    sp_tb = pd.read_csv(PROCESSED_ROOT / "splits_tierb.csv", parse_dates=WCOLS)
    sp_ds = pd.read_csv(PROCESSED_ROOT / "splits_dse.csv", parse_dates=WCOLS)
    pl_tb = {p["year"]: p for p in year_plan(sp_tb, pd.DatetimeIndex(sorted(tb["date"].unique())), 0, pcfg["embargo_days"], c["first_year"])}
    pl_ds = {p["year"]: p for p in year_plan(sp_ds, pd.DatetimeIndex(sorted(ds["date"].unique())), 0, pcfg["embargo_days"], c["first_year"])}
    years = sorted(set(pl_tb) & set(pl_ds))
    if quick:
        years = years[-2:]
    out = []
    for mname in c["models"]:
        t0 = time.time()
        for y in years:
            tr_tb = train_mask(tb, pl_tb[y]["w0"], [20])
            tr_ds = train_mask(ds, pl_ds[y]["w0"], [20])
            ev_tb = (tb["date"] >= pl_tb[y]["eval_start"]) & (tb["date"] <= pl_tb[y]["eval_end"])
            ev_ds = (ds["date"] >= pl_ds[y]["eval_start"]) & (ds["date"] <= pl_ds[y]["eval_end"])
            fits = {"tierb": tb[tr_tb], "dse": ds[tr_ds], "pooled": pd.concat([tb[tr_tb], ds[tr_ds]])}
            for src, F in fits.items():
                mdl = make_model(mname, 20, cfg["seed"], cfg).fit(F[feats], F["y_rank"].to_numpy(), F)
                for tgt, E in (("tierb", tb[ev_tb]), ("dse", ds[ev_ds])):
                    o = E[["date", "asset", "y_rank"]].copy()
                    o["pred"] = mdl.predict(E[feats], E)
                    o["source"], o["target"], o["model"], o["year"] = src, tgt, mname, y
                    out.append(o)
        log(f"  CS {mname:9s} years={len(years)}  {time.time()-t0:6.1f}s")
    P = pd.concat(out, ignore_index=True)
    write_table(P, PRED_DIR, "cs_predictions")
    rows = []
    for (s, t, m), g in P.groupby(["source", "target", "model"]):
        ic = M.daily_cs_ic(g, "pred", "y_rank", min_n=c["min_assets"])
        mu, tt = M.nw_tstat(ic.to_numpy(), lag=20)
        lo, hi = M.block_bootstrap_ci(ic, block=63)
        rows.append({"source": s, "target": t, "model": m, "setting": "in-domain" if s == t else ("pooled" if s == "pooled" else "zero-shot"),
                     "days": len(ic), "rank_ic_mean": mu, "ci_lo": lo, "ci_hi": hi, "nw_t": tt, "share_pos": float((ic > 0).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def figures(ev: dict, cs: pd.DataFrame, figdir: Path) -> None:
    s = ev["summary"][~ev["summary"]["setting"].str.startswith("group:")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, lab in ((axes[0], "pooled_r2_oos", "Pooled OOS $R^2$ vs own historical mean"), (axes[1], "median_ic", "Median time-series IC")):
        piv = s.pivot(index="model", columns="setting", values=col)
        piv.plot.bar(ax=ax, rot=0, width=0.8)
        ax.axhline(0, color="k", lw=0.8); ax.set_title(lab, fontsize=10); ax.set_xlabel("")
    axes[1].get_legend().remove()
    fig.tight_layout(); fig.savefig(figdir / "e1_settings.png", dpi=150); fig.savefig(figdir / "e1_settings.pdf"); plt.close(fig)

    mat = ev["matrix"]
    for m in mat["model"].unique():
        piv = mat[mat["model"] == m].pivot(index="source", columns="target_region", values="pooled_r2_oos")
        fig, ax = plt.subplots(figsize=(8, 5))
        v = np.nanmax(np.abs(piv.to_numpy())) or 1
        im = ax.imshow(piv.to_numpy(), cmap="RdBu", vmin=-v, vmax=v)
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.iat[i, j]:.3f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"Transfer matrix ({m}): pooled OOS $R^2$, source group → target region", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
        fig.savefig(figdir / f"e1_matrix_{m}.png", dpi=150); fig.savefig(figdir / f"e1_matrix_{m}.pdf"); plt.close(fig)

    ba = ev["by_asset"]
    for m in ("ridge", "gbm"):
        a = ba[(ba["model"] == m) & (ba["setting"] == "indomain")].set_index("asset")["r2_oos"]
        b = ba[(ba["model"] == m) & (ba["setting"] == "lomo")].set_index("asset")["r2_oos"]
        j = pd.concat([a, b], axis=1, keys=["in", "zs"]).dropna()
        if j.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(j["in"], j["zs"], s=14)
        for k, r in j.iterrows():
            ax.annotate(k, (r["in"], r["zs"]), fontsize=6)
        lim = [min(j.min().min(), -0.05), max(j.max().max(), 0.05)]
        ax.plot(lim, lim, "k--", lw=0.8); ax.axhline(0, lw=0.5, color="grey"); ax.axvline(0, lw=0.5, color="grey")
        ax.set_xlabel("in-domain OOS $R^2$"); ax.set_ylabel("zero-shot (leave-market-out) OOS $R^2$")
        ax.set_title(f"Per-market transfer ({m})", fontsize=10); fig.tight_layout()
        fig.savefig(figdir / f"e1_inv_zs_{m}.png", dpi=150); plt.close(fig)


def md(df: pd.DataFrame, fmt: str = "{:.4f}") -> str:
    if df.empty:
        return "(none)"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 years TS / 2 years CS for a fast check")
    ap.add_argument("--skip-cs", action="store_true")
    args = ap.parse_args()
    d = step_dir(STEP)
    tee = Tee(d / "log.txt"); sys.stdout = tee
    cfg = load_cfg()
    write_env_info(d, seed=cfg["seed"])
    T0 = time.time()

    def log(s):
        print(s, flush=True)

    log(f"E1 start (quick={args.quick})")
    P, _ = run_ts(cfg, args.quick, log)
    ev = evaluate_ts(P, cfg)
    ev["summary"].to_csv(d / "summary_ts.csv", index=False)
    ev["by_asset"].to_csv(d / "metrics_ts_by_asset.csv", index=False)
    ev["dm"].to_csv(d / "dm_tests.csv", index=False)
    ev["matrix"].to_csv(d / "transfer_matrix.csv", index=False)
    cs = pd.DataFrame() if args.skip_cs else run_cs(cfg, args.quick, log)
    cs.to_csv(d / "cs_ic.csv", index=False)
    figures(ev, cs, d / "figures")

    s = ev["summary"]
    main_s = s[~s["setting"].str.startswith("group:")].sort_values(["model", "setting"])
    dm = ev["dm"]
    dm_s = dm.groupby(["setting", "model"]).agg(assets=("asset", "nunique"),
                                                 worse_sig=("dm_stat", lambda x: int((x > 1.96).sum())),
                                                 better_sig=("dm_stat", lambda x: int((x < -1.96).sum()))).reset_index() if len(dm) else pd.DataFrame()
    ind = main_s[main_s["setting"] == "indomain"].set_index("model")["pooled_r2_oos"]
    zs = main_s[main_s["setting"] == "lomo"].set_index("model")["pooled_r2_oos"]
    plaus = (ind - zs).dropna()
    lines = [
        "# E1 — forecasting baselines and cross-market transfer", "",
        f"Quick mode: {args.quick}. Development period only (test locked). Runtime {time.time()-T0:.0f}s.", "",
        "Target: 20-day excess log return of each country ETF. OOS $R^2$ is against the market's own historical mean "
        "(Campbell-Thompson). Intervals: split-conformal on the last 252 training days of the setting's own source pool.", "",
        "## Main settings (Tier B, time-series task)", "",
        md(main_s[["setting", "model", "markets", "pooled_r2_oos", "r2_ci_lo", "r2_ci_hi", "median_ic", "share_r2_pos",
                   "median_auc", "cs_ic_mean", "cs_ic_nw_t", "mean_cov90", "mean_width90"]]), "",
        "## Diebold-Mariano vs in-domain (per country, HAC lag 19): number significantly worse / better at 5%", "",
        md(dm_s), "",
        "## Plausibility: in-domain minus zero-shot pooled OOS $R^2$ (by model)", "",
        md(plaus.rename("indomain_minus_lomo").reset_index()), "",
        "## Transfer matrix (source group → target region)", "",
        md(ev["matrix"].sort_values(["model", "source", "target_region"])), "",
        "## Cross-sectional task: Tier B countries ↔ DSE stocks (daily rank IC, NW t lag 20, block-bootstrap 95% CI)", "",
        md(cs) if len(cs) else "(skipped)", "",
        "Figures: e1_settings, e1_matrix_<model>, e1_inv_zs_<model>.",
    ]
    (d / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"E1 done in {time.time()-T0:.0f}s")
    sys.stdout = tee.stdout; tee.close()
    print(f"Handoff: {make_zip(d, STEP)}")


if __name__ == "__main__":
    main()
