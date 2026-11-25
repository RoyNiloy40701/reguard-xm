"""Check coverage and basic quality of the downloaded raw data (Tier B).

Usage (from the code folder):
    python scripts/check_raw_data.py
    python scripts/check_raw_data.py --day 2026-11-25    # check a specific download folder

Expected runtime: under 1 minute. Writes results/raw_check/ and handoff_raw_check.zip.
Raw files are only read, never changed.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reguard.data.download.common import LOG_PATH, RAW_ROOT, load_config, safe_name  # noqa: E402
from reguard.utils.handoff import Tee, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "raw_check"
BIG_MOVE = 0.25  # flag daily moves larger than 25%


def latest_day(source: str, day: str | None) -> Path | None:
    base = RAW_ROOT / source
    if not base.exists():
        return None
    if day:
        return base / day if (base / day).exists() else None
    days = sorted(p for p in base.iterdir() if p.is_dir())
    return days[-1] if days else None


def check_yahoo(folder: Path, items: list[dict], group: str) -> pd.DataFrame:
    rows = []
    today = pd.Timestamp.today().normalize()
    for it in items:
        t = it["ticker"]
        p = folder / group / f"{safe_name(t)}.csv"
        r = {"group": group, "ticker": t, "country": it.get("country", ""), "file_found": p.exists()}
        if not p.exists():
            rows.append(r)
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        px = pd.to_numeric(df[price_col], errors="coerce")
        ret = px.pct_change(fill_method=None).dropna()
        bdays = pd.bdate_range(df.index.min(), df.index.max())
        vol = pd.to_numeric(df["Volume"], errors="coerce") if "Volume" in df.columns else pd.Series(dtype=float)
        r.update(
            rows=len(df),
            first=df.index.min().date(),
            last=df.index.max().date(),
            years=round((df.index.max() - df.index.min()).days / 365.25, 1),
            price_col=price_col,
            has_adj_close="Adj Close" in df.columns,
            has_dividends="Dividends" in df.columns and (pd.to_numeric(df["Dividends"], errors="coerce").fillna(0) > 0).any(),
            missing_price_pct=round(100 * px.isna().mean(), 2),
            zero_volume_pct=round(100 * (vol == 0).mean(), 2) if len(vol) else np.nan,
            coverage_vs_weekdays_pct=round(100 * len(df) / max(len(bdays), 1), 1),
            n_big_moves=int((ret.abs() > BIG_MOVE).sum()),
            max_abs_daily_ret=round(float(ret.abs().max()), 4) if len(ret) else np.nan,
            ann_vol=round(float(ret.std() * np.sqrt(252)), 4) if len(ret) > 20 else np.nan,
            days_since_last=int((today - df.index.max()).days),
        )
        rows.append(r)
    return pd.DataFrame(rows)


def check_fred(folder: Path, cfg: dict) -> pd.DataFrame:
    rows = []
    for grp in ("rates", "fx"):
        for s in cfg[grp]:
            p = folder / f"{s['id']}.csv" if folder else None
            r = {"group": grp, "series": s["id"], "currency": s.get("currency", ""), "file_found": bool(p and p.exists())}
            if p and p.exists():
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                v = pd.to_numeric(df.iloc[:, 0], errors="coerce")
                r.update(rows=len(v), non_missing=int(v.notna().sum()), first=v.dropna().index.min().date(),
                         last=v.dropna().index.max().date(), missing_pct=round(100 * v.isna().mean(), 2),
                         min=round(float(v.min()), 4), max=round(float(v.max()), 4))
            rows.append(r)
    return pd.DataFrame(rows)


def check_french(folder: Path | None, names: list[str]) -> pd.DataFrame:
    rows = []
    for n in names:
        p = folder / f"{n}_CSV.zip" if folder else None
        r = {"dataset": n, "file_found": bool(p and p.exists())}
        if p and p.exists():
            try:
                with zipfile.ZipFile(p) as z:
                    infos = z.infolist()
                    r.update(bytes=p.stat().st_size, files_inside=";".join(i.filename for i in infos),
                             uncompressed_bytes=sum(i.file_size for i in infos), valid_zip=True)
            except zipfile.BadZipFile:
                r.update(valid_zip=False)
        rows.append(r)
    return pd.DataFrame(rows)


def coverage_figure(yh: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = yh[yh.get("file_found", False) == True].copy()  # noqa: E712
    if d.empty:
        return
    d["first"] = pd.to_datetime(d["first"])
    d["last"] = pd.to_datetime(d["last"])
    d = d.sort_values(["group", "first"])
    fig, ax = plt.subplots(figsize=(8, max(4, 0.16 * len(d))))
    colors = {"etfs": "#1f77b4", "indices": "#ff7f0e"}
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r["first"], r["last"]], [i, i], lw=3, color=colors.get(r["group"], "grey"))
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{r.ticker} ({r.country})" for r in d.itertuples()], fontsize=5)
    ax.axvline(pd.Timestamp("2022-01-01"), color="k", ls="--", lw=0.8)
    ax.text(pd.Timestamp("2022-03-01"), len(d) - 1, "proposed\ntest lock", fontsize=6, va="top")
    ax.set_title("Tier B coverage (blue = USD country ETFs, orange = local indices)", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / f"coverage_tierB.{ext}", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None)
    args = ap.parse_args()
    out = step_dir(STEP)
    tee = Tee(out / "log.txt")
    sys.stdout = tee
    t0 = time.time()
    try:
        write_env_info(out)
        cfg = load_config("tier_b.yaml")
        yfold, ffold, frfold = latest_day("yahoo", args.day), latest_day("fred", args.day), latest_day("french", args.day)
        print(f"Folders: yahoo={yfold} fred={ffold} french={frfold}")

        yh = pd.concat([
            check_yahoo(yfold, cfg["country_etfs"], "etfs") if yfold else pd.DataFrame(),
            check_yahoo(yfold, cfg["local_indices"], "indices") if yfold else pd.DataFrame(),
        ], ignore_index=True)
        fr = check_fred(ffold, cfg["fred_series"]) if ffold else pd.DataFrame()
        kf = check_french(frfold, cfg["french_datasets"])
        yh.to_csv(out / "coverage_yahoo.csv", index=False)
        fr.to_csv(out / "coverage_fred.csv", index=False)
        kf.to_csv(out / "french_files.csv", index=False)
        coverage_figure(yh, out)

        log = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()
        fails = log[log["status"] == "failed"][["source", "item", "note"]].drop_duplicates("item", keep="last") if len(log) else pd.DataFrame()
        if len(log):
            ok_items = set(log.loc[log["status"] == "ok", "item"])
            fails = fails[~fails["item"].isin(ok_items)]

        etf = yh[(yh.get("group") == "etfs") & (yh.get("file_found") == True)] if len(yh) else pd.DataFrame()  # noqa: E712
        n_etf = len(etf)
        n_10y = int((etf["years"] >= 10).sum()) if n_etf else 0
        n_pre2012 = int((pd.to_datetime(etf["first"]) <= "2012-12-31").sum()) if n_etf else 0
        metrics = {
            "etfs_found": n_etf, "etfs_expected": len(cfg["country_etfs"]),
            "etfs_with_10y_history": n_10y, "etfs_starting_by_2012": n_pre2012,
            "indices_found": int(((yh.get("group") == "indices") & (yh.get("file_found") == True)).sum()) if len(yh) else 0,  # noqa: E712
            "indices_expected": len(cfg["local_indices"]),
            "fred_found": int(fr["file_found"].sum()) if len(fr) else 0,
            "fred_expected": len(cfg["fred_series"]["rates"]) + len(cfg["fred_series"]["fx"]),
            "french_found": int(kf["file_found"].sum()), "french_expected": len(cfg["french_datasets"]),
            "unresolved_failures": len(fails),
            "series_with_big_moves": int((yh.get("n_big_moves", pd.Series(dtype=float)).fillna(0) > 0).sum()) if len(yh) else 0,
        }
        with open(out / "metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerows(metrics.items())

        lines = ["# Raw data check — Tier B", "",
                 "| Item | Found / expected |", "|---|---|",
                 f"| Country ETFs | {metrics['etfs_found']} / {metrics['etfs_expected']} |",
                 f"| ETFs with ≥ 10 years of history | {metrics['etfs_with_10y_history']} |",
                 f"| ETFs starting by end-2012 | {metrics['etfs_starting_by_2012']} |",
                 f"| Local indices | {metrics['indices_found']} / {metrics['indices_expected']} |",
                 f"| FRED series | {metrics['fred_found']} / {metrics['fred_expected']} |",
                 f"| French datasets | {metrics['french_found']} / {metrics['french_expected']} |",
                 f"| Unresolved download failures | {metrics['unresolved_failures']} |", ""]
        if len(fails):
            lines += ["## Failed items", "", "| Source | Item | Note |", "|---|---|---|"] + \
                     [f"| {r.source} | {r.item} | {str(r.note)[:120]} |" for r in fails.itertuples()] + [""]
        if len(yh) and "n_big_moves" in yh:
            flag = yh[(yh["n_big_moves"].fillna(0) > 0) | (yh["missing_price_pct"].fillna(0) > 1)]
            if len(flag):
                lines += ["## Series to inspect (daily moves > 25% or > 1% missing prices)", "",
                          "| Ticker | Country | Big moves | Max abs return | Missing % |", "|---|---|---|---|---|"] + \
                         [f"| {r.ticker} | {r.country} | {r.n_big_moves} | {r.max_abs_daily_ret} | {r.missing_price_pct} |"
                          for r in flag.itertuples()] + [""]
        lines += ["Full tables: coverage_yahoo.csv, coverage_fred.csv, french_files.csv. Figure: figures/coverage_tierB.pdf"]
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
    finally:
        sys.stdout = tee.stdout
        tee.close()
    z = make_zip(out, STEP)
    print(f"\nDone in {time.time() - t0:.0f}s. Upload this file to Claude:\n  {z}")


if __name__ == "__main__":
    main()
