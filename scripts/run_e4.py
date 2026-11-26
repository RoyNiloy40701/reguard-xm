"""E4: governance across proposers on the 43-country-ETF portfolio (H5, H6, H7). Development data only.

Usage (from the code folder):
    python scripts/run_e4.py --quick      # classical proposers, 2020-2021 only (pipeline check)
    python scripts/run_e4.py              # full development evaluation 2018-2021 (+ PPO if trained)
Run scripts/train_ppo.py first to include PPO (otherwise PPO is reported as missing).
Writes results/e4/{report.md, summary.csv, h5.csv, h6.csv, h7.csv, sensitivity.csv, figures/, log.txt,
env_info.txt} and results/e4/handoff_e4.zip.
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

from reguard.data.store import PROCESSED_ROOT, load_panel, load_yaml, read_table, write_table  # noqa: E402
from reguard.portfolio import metrics as PM  # noqa: E402
from reguard.portfolio import proposers as P  # noqa: E402
from reguard.portfolio.backtest import reference_pool, run_backtest  # noqa: E402
from reguard.portfolio.data import Market, pooled_forecasts  # noqa: E402
from reguard.utils.handoff import CODE_ROOT, PROJECT_ROOT, Tee, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "e4"
CACHE = PROJECT_ROOT / "data" / "interim" / "e4"
WCOLS = ["train_start", "train_end", "embargo_start", "eval_start", "eval_end"]


def md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def classical(mkt, cfg):
    mom = mkt.feature("mom_12_1").to_numpy()
    vol = mkt.feature("vol_60d").to_numpy()
    pc = cfg["proposers"]
    return {
        "equal_weight": lambda t, w, mu, S, av: P.equal_weight(av),
        "momentum": lambda t, w, mu, S, av: P.momentum(av, mom[t], pc["momentum"]["top_frac"]),
        "mean_variance": lambda t, w, mu, S, av: P.mean_variance(av, mu, S, pc["mean_variance"]["risk_aversion"],
                                                               pc["mean_variance"]["max_weight"]),
        "risk_parity": lambda t, w, mu, S, av: P.risk_parity(av, vol[t]),
    }


def ppo_models(mkt, cfg, log):
    d = CACHE / "ppo"
    files = sorted(d.glob("final_seed*.zip")) if d.exists() else []
    if not files:
        log("  PPO: no trained models found (run scripts/train_ppo.py) -> PPO skipped")
        return {}
    try:
        from stable_baselines3 import PPO
    except ImportError:
        log("  PPO: stable-baselines3 not installed -> PPO skipped")
        return {}
    from reguard.portfolio.rl_env import ObsBuilder, ppo_proposer
    ob = ObsBuilder(mkt, cfg["ppo"]["obs_features"])
    smoke = "smoke: true" in (d / "chosen.yaml").read_text() if (d / "chosen.yaml").exists() else False
    log(f"  PPO: {len(files)} trained seed(s){' (SMOKE models: not for results)' if smoke else ''}")
    return {f.stem.replace("final_", "ppo_"): ppo_proposer(PPO.load(f, device="cpu"), ob) for f in files}


def in_stress(dates: pd.Series, windows) -> np.ndarray:
    m = np.zeros(len(dates), bool)
    for a, b in windows:
        m |= (dates >= pd.Timestamp(a)).to_numpy() & (dates <= pd.Timestamp(b)).to_numpy()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    d = step_dir(STEP)
    tee = Tee(d / "log.txt"); sys.stdout = tee
    with open(CODE_ROOT / "configs" / "experiment" / "e4.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pcfg = load_yaml("processing.yaml")
    write_env_info(d, seed=cfg["seed"])
    log = lambda s: print(s, flush=True)  # noqa: E731
    T0 = time.time()
    if args.quick:
        cfg["periods"]["eval"] = ["2020-01-01", "2021-12-31"]
        cfg["periods"]["pool"] = ["2016-01-01", "2019-12-31"]
    panel = load_panel(cfg["panel"])
    mkt = Market(panel, cfg["min_history"])
    try:
        fc = read_table(CACHE, "forecasts")
        log(f"  forecasts loaded from cache ({len(fc)} rows)")
    except FileNotFoundError:
        log("  forecasts: fitting pooled GBM with annual refits (one-off, cached)")
        splits = pd.read_csv(PROCESSED_ROOT / "splits_tierb.csv", parse_dates=WCOLS)
        fc = pooled_forecasts(panel, splits, cfg, pcfg["embargo_days"], log)
        write_table(fc, CACHE, "forecasts")
    mu20 = fc.pivot(index="date", columns="asset", values="mu20")
    props = classical(mkt, cfg)
    pool = reference_pool(mkt, mu20, cfg, props["equal_weight"])
    log(f"  governance pool: {len(pool.v)} reference scores; initial q = {pool.quantile(1 - cfg['governance']['alpha']):.3f}")
    ppo = ppo_models(mkt, cfg, log)
    ea, eb = cfg["periods"]["eval"]
    res, series = [], {}
    for name, fn in {**props, **ppo}.items():
        t0 = time.time()
        for gov in (False, True):
            bt = run_backtest(mkt, ea, eb, fn, mu20, cfg, governed=gov, pool=pool)
            series[(name, gov)] = bt
            res.append({"proposer": name, "governed": gov, **PM.summary(bt)})
        log(f"  {name}: {time.time()-t0:.0f}s")
    # PPO ensemble = mean of seeds' daily net returns (pre-registered for H7)
    seeds = [k for k in ppo]
    if seeds:
        for gov in (False, True):
            base = series[(seeds[0], gov)].copy()
            for c in ("net", "gross", "cash", "traded", "err"):
                base[c] = np.mean([series[(s, gov)][c].to_numpy(dtype=float) for s in seeds], axis=0)
            base["modified"] = np.mean([series[(s, gov)]["modified"].to_numpy(dtype=float) for s in seeds], axis=0) > 0.5
            series[("ppo_ensemble", gov)] = base
            res.append({"proposer": "ppo_ensemble", "governed": gov, **PM.summary(base)})
    S = pd.DataFrame(res)
    S.to_csv(d / "summary.csv", index=False)

    # H5 / H6 per proposer (governed vs ungoverned)
    stress = cfg["stress_windows"]
    names = sorted({k[0] for k in series})
    h5, h6 = [], []
    for nm in names:
        g, u = series[(nm, True)], series[(nm, False)]
        dcv, lo, hi = PM.paired_bootstrap(g["net"].to_numpy(), u["net"].to_numpy(), PM.cvar, n_boot=500 if args.quick else 1000)
        dret = PM.summary(g)["ann_return"] - PM.summary(u)["ann_return"]
        h5.append({"proposer": nm, "delta_cvar95": dcv, "ci_lo": lo, "ci_hi": hi, "supported": hi < 0,
                   "delta_ann_return": dret, "governance_gain": (-dcv / -dret) if dret < 0 else np.inf,
                   "delta_max_dd": PM.summary(g)["max_dd"] - PM.summary(u)["max_dd"]})
        ns = ~in_stress(g["date"], stress)
        h6.append({"proposer": nm, "sharpe_loss": PM.summary(u)["sharpe"] - PM.summary(g)["sharpe"],
                   "cash_outside_stress": float(g.loc[ns, "cash"].mean()),
                   "supported": (PM.summary(u)["sharpe"] - PM.summary(g)["sharpe"] <= 0.15) and g.loc[ns, "cash"].mean() <= 0.35})
    H5, H6 = pd.DataFrame(h5), pd.DataFrame(h6)
    H5.to_csv(d / "h5.csv", index=False); H6.to_csv(d / "h6.csv", index=False)
    if ("ppo_ensemble", True) in series:
        x = series[("ppo_ensemble", True)]; y = series[("mean_variance", True)]
        t7 = PM.sharpe_diff_test((x["net"] - x["rf"]).to_numpy(), (y["net"] - y["rf"]).to_numpy())
        H7 = pd.DataFrame([{"comparison": "governed PPO ensemble - governed mean-variance", **t7, "supported": t7["p"] < 0.05}])
    else:
        H7 = pd.DataFrame([{"comparison": "PPO not available", "supported": None}])
    H7.to_csv(d / "h7.csv", index=False)

    # DSR for each strategy (trials = all proposer x governance variants evaluated)
    ex = {k: (v["net"] - v["rf"]).to_numpy() for k, v in series.items() if not str(k[0]).startswith("ppo_seed")}
    srs = np.array([e.mean() / e.std(ddof=1) for e in ex.values()])
    S["dsr"] = [PM.deflated_sharpe(ex[(r.proposer, r.governed)], len(ex), srs.var()) if (r.proposer, r.governed) in ex else np.nan
                for r in S.itertuples()]
    S.to_csv(d / "summary.csv", index=False)

    # sensitivity: costs and B_max (classical proposers)
    sens = []
    if not args.quick:
        for nm, fn in props.items():
            for c in cfg["cost_sensitivity"]:
                for gov in (False, True):
                    sens.append({"proposer": nm, "governed": gov, "cost_bps": c, "b_max": cfg["governance"]["b_max"],
                                 **PM.summary(run_backtest(mkt, ea, eb, fn, mu20, cfg, gov, pool, cost_bps=c))})
            for bm in cfg["governance"]["b_max_sensitivity"]:
                sens.append({"proposer": nm, "governed": True, "cost_bps": cfg["cost_bps"], "b_max": bm,
                             **PM.summary(run_backtest(mkt, ea, eb, fn, mu20, cfg, True, pool, b_max=bm))})
    pd.DataFrame(sens).to_csv(d / "sensitivity.csv", index=False)

    # figures
    figs = d / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for nm in names:
        if str(nm).startswith("ppo_seed"):
            continue
        for gov, ls in ((False, "--"), (True, "-")):
            b = series[(nm, gov)]
            axes[0].plot(b["date"], np.cumprod(1 + b["net"]), ls=ls, lw=1, label=f"{nm}{' (gov)' if gov else ''}")
    axes[0].set_title("Development evaluation: wealth (dashed = ungoverned)", fontsize=9); axes[0].legend(fontsize=6)
    for r in H5.itertuples():
        if str(r.proposer).startswith("ppo_seed"):
            continue
        axes[1].scatter(r.delta_ann_return, r.delta_cvar95); axes[1].annotate(r.proposer, (r.delta_ann_return, r.delta_cvar95), fontsize=7)
    axes[1].axhline(0, color="grey", lw=0.6); axes[1].axvline(0, color="grey", lw=0.6)
    axes[1].set_xlabel("Δ annual return (governed − ungoverned)"); axes[1].set_ylabel("Δ CVaR95 (governed − ungoverned)")
    axes[1].set_title("Governance-gain plane", fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "e4_wealth_gg.png", dpi=150); fig.savefig(figs / "e4_wealth_gg.pdf"); plt.close(fig)

    show = S[~S["proposer"].astype(str).str.startswith("ppo_seed")]
    lines = ["# E4 — governance across proposers (development evaluation)", "",
             f"Quick: {args.quick}. Period {ea} to {eb}. Runtime {time.time()-T0:.0f}s. Locked test untouched.",
             f"Budget: B_max = {cfg['governance']['b_max']:.3f} one-day loss at alpha = {cfg['governance']['alpha']}; "
             f"costs {cfg['cost_bps']} bps one-way; weekly rebalancing; position cap {cfg['governance']['max_weight']}.",
             "NOTE: PPO hyper-parameters were chosen on this same development period, so PPO is optimistically "
             "evaluated here; the locked test (Phase 9) is the clean comparison.", "",
             "## Performance", "", md(show[["proposer", "governed", "ann_return", "ann_vol", "sharpe", "sortino", "max_dd",
                                            "cvar95", "turnover", "cash_share", "share_modified", "bef", "dsr"]]), "",
             "## H5 — governance reduces CVaR95 for every proposer (stationary bootstrap, paired)", "", md(H5), "",
             "## H6 — cost of caution (Sharpe loss <= 0.15, cash outside stress <= 35%)", "", md(H6), "",
             "## H7 — governed PPO vs governed mean-variance (Ledoit-Wolf HAC)", "", md(H7), "",
             "Figures: e4_wealth_gg."]
    (d / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"E4 done in {time.time()-T0:.0f}s")
    sys.stdout = tee.stdout; tee.close()
    print(f"Handoff: {make_zip(d, STEP)}")


if __name__ == "__main__":
    main()
