"""Phase 3: validate processed panels and write the data-quality handoff.

Usage (from the code folder, after build_dataset.py):
    python scripts/validate_dataset.py
Writes results/data_quality/{report.md, metrics.csv, checks.csv, log.txt, env_info.txt, figures/}
and results/data_quality/handoff_data_quality.zip  -> upload this zip to Claude.
Only development data is read. The locked test files are checked for existence and date range
by file metadata only (they are never loaded here); the lock itself is self-tested.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reguard.data.features import asset_features, forward_targets  # noqa: E402
from reguard.data.splits import train_mask  # noqa: E402
from reguard.data.store import (LOCKED_ROOT, PROCESSED_ROOT, TestLockedError, load_panel,  # noqa: E402
                                load_yaml)
from reguard.utils.handoff import Tee, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "data_quality"
RNG = np.random.default_rng(20261125)
CHECKS: list[dict] = []


def md(df: pd.DataFrame, index: bool = False) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    if index:
        df = df.reset_index()
    if df.empty:
        return "(none)"
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False) -> None:
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    CHECKS.append({"check": name, "status": status, "detail": detail})
    print(f"[{status}] {name}  {detail}")


def feature_cols(df: pd.DataFrame) -> list[str]:
    skip = {"date", "market", "asset", "region", "msci_class", "currency", "in_universe", "univ_rank",
            "stopped_trading", "flag_ca_adjusted", "flag_break", "close_unadj", "traded_value",
            "rf_daily", "delisting_return", "ret_usd"}
    return [c for c in df.columns if c not in skip and not c.startswith(("y_", "tgt_", "usd_y_"))]


def basic_checks(name: str, df: pd.DataFrame, cfg: dict) -> dict:
    ts = pd.Timestamp(cfg["test_start"])
    check(f"{name}: no rows on/after test_start in dev file", bool((df["date"] < ts).all()),
          f"max date {df['date'].max().date()}")
    dups = int(df.duplicated(["asset", "date"]).sum())
    check(f"{name}: no duplicate (asset, date)", dups == 0, f"{dups} duplicates")
    for h in cfg["horizons"]:
        c = f"tgt_end_{h}d"
        if c in df:
            leak = int((df[c] >= ts).sum())
            check(f"{name}: no {h}d target reaches the test period", leak == 0, f"{leak} rows")
    finite = np.isfinite(df.select_dtypes("number").to_numpy(dtype=float, na_value=np.nan)[:, :]).sum()
    inf = int(np.isinf(df.select_dtypes("number").to_numpy(dtype=float, na_value=np.nan)).sum())
    check(f"{name}: no infinite values", inf == 0, f"{inf} inf cells", warn_only=True)
    if name.startswith("dse"):          # DSE break rule is on simple returns
        lim, x, kind = cfg["dse"]["break_abs_ret"], np.expm1(df["ret"]).abs(), "simple"
    else:
        lim, x, kind = cfg["tierb"]["max_abs_daily_logret"], df["ret"].abs(), "log"
    big = int((x > lim + 1e-12).sum())
    check(f"{name}: no |daily {kind} return| above the break threshold ({lim:.2f}) left", big == 0, f"{big} rows")
    lf = sorted(LOCKED_ROOT.glob(f"{name}__TEST.*"))
    check(f"{name}: locked test file exists", bool(lf), lf[0].name if lf else "missing")
    miss = df[feature_cols(df)].isna().mean().sort_values(ascending=False)
    return {"panel": name, "rows": len(df), "assets": df["asset"].nunique(), "markets": df["market"].nunique(),
            "first_date": str(df["date"].min().date()), "last_date": str(df["date"].max().date()),
            "finite_cells": int(finite), "feature_missing_max": round(float(miss.iloc[0]), 4),
            "feature_missing_max_col": miss.index[0],
            "target_y_ret_20d_coverage": round(float(df["y_ret_20d"].notna().mean()), 4)}


def lookahead_recompute(name: str, df: pd.DataFrame, cfg: dict, bench: pd.Series | None, n: int = 6) -> None:
    """Truncate a real asset at a random date t, recompute features from data <= t only, compare."""
    assets = df["asset"].unique()
    worst, cols = 0.0, None
    for a in RNG.choice(assets, size=min(n, len(assets)), replace=False):
        s = df[df["asset"] == a].set_index("date").sort_index()
        if len(s) < 400:
            continue
        t = s.index[RNG.integers(300, len(s) - 30)]
        past = s.loc[:t]
        x = pd.DataFrame({"ret": past["ret"]})
        if "logdv_20d" in s and "dollar_vol" not in s:
            dv = past["traded_value"] if "traded_value" in past else None
            if dv is not None:
                x["dollar_vol"] = dv
        f = asset_features(x, cfg, bench_ret=bench if ("beta_60d" in s and s["beta_60d"].notna().any()) else None)
        cols = [c for c in f.columns if c in s.columns and not c.startswith(("logdv", "dv_", "amihud"))]
        diff = (f.loc[t, cols] - s.loc[t, cols]).abs().max()
        worst = max(worst, float(np.nan_to_num(diff)))
    check(f"{name}: features at t recomputed from data <= t match the panel", worst < 1e-8,
          f"max abs diff {worst:.2e} over {len(cols or [])} price features")


def target_recompute(name: str, df: pd.DataFrame, cfg: dict) -> None:
    worst = 0.0
    for a in RNG.choice(df["asset"].unique(), size=5, replace=False):
        s = df[df["asset"] == a].set_index("date").sort_index()
        if len(s) < 100:
            continue
        i = int(RNG.integers(10, len(s) - 30))
        for h in cfg["horizons"]:
            manual = s["ret"].iloc[i + 1:i + 1 + h].sum(min_count=h)
            got = s[f"y_ret_{h}d"].iloc[i]
            if np.isfinite(manual) and np.isfinite(got):
                worst = max(worst, abs(manual - got))
    check(f"{name}: targets equal the sum of the next h returns", worst < 1e-10, f"max abs diff {worst:.2e}")


def split_checks(name: str, df: pd.DataFrame, splits: pd.DataFrame, cfg: dict) -> dict:
    bad, sizes = 0, []
    for _, w in splits.iterrows():
        m = train_mask(df, w, cfg["horizons"])
        tr = df.loc[m]
        for h in cfg["horizons"]:
            bad += int((tr[f"tgt_end_{h}d"] >= w["eval_start"]).sum())
        bad += int((tr["date"] > w["train_end"]).sum())
        sizes.append(int(m.sum()))
    gap_ok = bool(((splits["eval_start"] - splits["train_end"]).dt.days >= cfg["embargo_days"]).all())
    check(f"{name}: walk-forward windows purged (no train target reaches eval)", bad == 0,
          f"{len(splits)} windows, {bad} violations")
    check(f"{name}: embargo of {cfg['embargo_days']} market days before every window", gap_ok)
    check(f"{name}: last window ends before test_start",
          bool((splits["eval_end"] < pd.Timestamp(cfg["test_start"])).all()), str(splits["eval_end"].max().date()))
    return {"windows": len(splits), "first_window": str(splits["eval_start"].min().date()),
            "median_train_rows": int(np.median(sizes))}


def lock_selftest() -> None:
    try:
        load_panel("tierb_etf", part="test", unlock_reason=None)
        check("Test lock refuses access without unlock", False, "loaded test data!")
    except TestLockedError:
        check("Test lock refuses access without unlock", True, "refused and logged in docs/test_access_log.csv")
    except FileNotFoundError as e:
        check("Test lock refuses access without unlock", False, str(e))


def fig_coverage(etf: pd.DataFrame, out: Path) -> None:
    c = etf.assign(year=etf["date"].dt.year).groupby(["asset", "year"])["ret"].count().unstack(fill_value=0)
    c = c.loc[c.gt(0).idxmax(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow((c > 200).to_numpy(), aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index, fontsize=6)
    ax.set_xticks(range(0, len(c.columns), 3)); ax.set_xticklabels(c.columns[::3], rotation=90, fontsize=7)
    ax.set_title("Tier B country ETFs: years with >200 return observations (development period)")
    fig.tight_layout(); fig.savefig(out / "tierb_coverage.png", dpi=150); fig.savefig(out / "tierb_coverage.pdf"); plt.close(fig)


def fig_dse(dsep: pd.DataFrame, uni: pd.DataFrame, out: Path) -> dict:
    uni = uni.sort_values("decision_date")
    sets = uni.groupby("decision_date")["ticker"].apply(set)
    turn = [len(b - a) / max(len(b), 1) for a, b in zip(sets.iloc[:-1], sets.iloc[1:])]
    n_assets = dsep.groupby("date")["asset"].nunique()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(n_assets.index, n_assets.values, lw=1); axes[0].set_ylabel("equities traded")
    axes[0].set_title("DSE: equities with a record per day; monthly universe turnover (top-N by liquidity)")
    axes[1].plot(sets.index[1:], turn, lw=1); axes[1].set_ylabel("share replaced")
    fp = dsep.groupby("date")["regime_floor_price"].max()
    for ax in axes:
        ax.fill_between(fp.index, 0, 1, where=fp.values > 0, transform=ax.get_xaxis_transform(), alpha=0.15, color="grey")
    fig.tight_layout(); fig.savefig(out / "dse_universe.png", dpi=150); fig.savefig(out / "dse_universe.pdf"); plt.close(fig)
    return {"universe_turnover_median": round(float(np.median(turn)), 3),
            "equities_per_day_median": int(n_assets.median())}


def main() -> None:
    d = step_dir(STEP)
    tee = Tee(d / "log.txt"); sys.stdout = tee
    cfg = load_yaml("processing.yaml")
    write_env_info(d, seed=20261125)
    summary = json.loads((PROCESSED_ROOT / "build_summary.json").read_text(encoding="utf-8"))
    metrics: list[dict] = []
    figs = d / "figures"

    etf = load_panel("tierb_etf")
    spy = etf[etf["asset"] == "SPY"].set_index("date")["ret"]
    metrics.append(basic_checks("tierb_etf", etf, cfg))
    lookahead_recompute("tierb_etf", etf, cfg, spy)
    target_recompute("tierb_etf", etf, cfg)
    sp = pd.read_csv(PROCESSED_ROOT / "splits_tierb.csv", parse_dates=["train_start", "train_end", "embargo_start", "eval_start", "eval_end"])
    metrics[-1].update(split_checks("tierb_etf", etf, sp, cfg))
    chk = pd.read_csv(PROCESSED_ROOT / "tierb_etf_checks.csv")
    check("tierb_etf: Adj Close returns agree with (Close+Dividend)/prevClose (median |diff| < 1e-5)",
          bool((chk["tr_check_median_abs"] < 1e-5).all()), f"worst ticker share >tol: {chk['tr_check_share_above_tol'].max():.4f}")
    shutil.copy(PROCESSED_ROOT / "tierb_etf_checks.csv", d / "tierb_etf_checks.csv")
    fig_coverage(etf, figs)

    idx = load_panel("tierb_index")
    metrics.append(basic_checks("tierb_index", idx, cfg))
    lookahead_recompute("tierb_index", idx, cfg, None)
    target_recompute("tierb_index", idx, cfg)
    shutil.copy(PROCESSED_ROOT / "tierb_index_checks.csv", d / "tierb_index_checks.csv")

    dsep = load_panel("dse_equity")
    mkt = pd.read_csv(PROCESSED_ROOT / "dse_market.csv", parse_dates=["date"]).set_index("date")["mkt_ret"]
    metrics.append(basic_checks("dse_equity", dsep, cfg))
    lookahead_recompute("dse_equity", dsep, cfg, mkt)
    target_recompute("dse_equity", dsep, cfg)
    sp = pd.read_csv(PROCESSED_ROOT / "splits_dse.csv", parse_dates=["train_start", "train_end", "embargo_start", "eval_start", "eval_end"])
    metrics[-1].update(split_checks("dse_equity", dsep, sp, cfg))
    uni = pd.read_csv(PROCESSED_ROOT / "dse_universe.csv", parse_dates=["decision_date", "effective_from", "effective_to"])
    per = uni.groupby("decision_date").size()
    check("dse: universe has top_n members every month", bool((per == cfg["dse"]["universe"]["top_n"]).all()),
          f"min {per.min()} max {per.max()} over {len(per)} months")
    check("dse: universe effective only after its decision date", bool((uni["effective_from"] > uni["decision_date"]).all()))
    metrics[-1].update(fig_dse(dsep, uni[uni["decision_date"] < pd.Timestamp(cfg["test_start"])], figs))
    life = pd.read_csv(PROCESSED_ROOT / "dse_lifetimes.csv")
    aud = pd.read_csv(PROCESSED_ROOT / "dse_survivorship_audit.csv")
    for f in ("dse_lifetimes.csv", "dse_survivorship_audit.csv", "dse_quality.csv"):
        shutil.copy(PROCESSED_ROOT / f, d / f)
    lock_selftest()

    pd.DataFrame(metrics).to_csv(d / "metrics.csv", index=False)
    ck = pd.DataFrame(CHECKS); ck.to_csv(d / "checks.csv", index=False)
    n_fail, n_warn = int((ck["status"] == "FAIL").sum()), int((ck["status"] == "WARN").sum())
    q = pd.read_csv(PROCESSED_ROOT / "dse_quality.csv")
    lines = [
        "# Phase 3 data-quality report", "",
        f"**Result: {'PASS' if n_fail == 0 else 'FAIL'}** — {len(ck)} checks, {n_fail} failed, {n_warn} warnings.", "",
        f"Locked final test: dates >= {cfg['test_start']} (stored separately; never loaded here).", "",
        "## Panels (development part)", "", md(pd.DataFrame(metrics)), "",
        "## Checks", "", md(ck), "",
        "## Build summary", "", "```json", json.dumps(summary["panels"], indent=2), "```", "",
        "## DSE return construction", "", md(q), "",
        "## DSE stopped/delisted equities in the main source", "",
        md(life[life["stopped_trading"]]), "",
        "## DSE survivorship audit (source B vs A)", "",
        md(aud["classification"].str.replace(r"rename_of_.*", "rename (already in A)", regex=True).value_counts().to_frame("count"), index=True), "",
        md(aud[~aud["classification"].str.startswith(("rename", "non_equity"))]), "",
        "Figures: figures/tierb_coverage.png, figures/dse_universe.png (grey = floor-price regimes).",
    ]
    (d / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRESULT: {'PASS' if n_fail == 0 else 'FAIL'} ({n_fail} failed, {n_warn} warnings)")
    sys.stdout = tee.stdout; tee.close()
    z = make_zip(d, STEP)
    print(f"Handoff: {z}")


if __name__ == "__main__":
    main()
