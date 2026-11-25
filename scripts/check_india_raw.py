"""Check the downloaded NSE (India) bhavcopy files: coverage, formats, universe size, survivorship.

Usage (from the code folder):
    python scripts/check_india_raw.py

Expected runtime: 2-6 minutes for ~5,000 files. Writes results/raw_check_india/ and its handoff zip.
Raw files are only read, never changed.
"""

from __future__ import annotations

import csv
import io
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from reguard.data.download.common import RAW_ROOT  # noqa: E402
from reguard.utils.handoff import Tee, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "raw_check_india"
OLD = {"symbol": "SYMBOL", "series": "SERIES", "close": "CLOSE", "prev": "PREVCLOSE",
       "volume": "TOTTRDQTY", "value": "TOTTRDVAL", "isin": "ISIN", "date": "TIMESTAMP"}
NEW = {"symbol": "TckrSymb", "series": "SctySrs", "close": "ClsPric", "prev": "PrvsClsgPric",
       "volume": "TtlTradgVol", "value": "TtlTrfVal", "isin": "ISIN", "date": "TradDt"}


def read_one(path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        df = pd.read_csv(io.BytesIO(z.read(name)), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    cols = list(df.columns)
    if all(c in cols for c in ("SYMBOL", "SERIES", "CLOSE")):
        m, fmt = OLD, "old"
    elif all(c in cols for c in ("TckrSymb", "SctySrs", "ClsPric")):
        m, fmt = NEW, "udiff"
    else:
        return pd.DataFrame(), "unknown", cols
    out = pd.DataFrame({k: df[v] for k, v in m.items() if v in df.columns})
    for k in ("symbol", "series", "isin"):
        if k in out:
            out[k] = out[k].astype(str).str.strip()
    return out, fmt, cols


def main():
    out_dir = step_dir(STEP)
    tee = Tee(out_dir / "log.txt")
    sys.stdout = tee
    t0 = time.time()
    try:
        write_env_info(out_dir)
        files = sorted((RAW_ROOT / "nse" / "bhavcopy").rglob("*.zip"))
        print(f"Found {len(files)} bhavcopy files")
        manifest = RAW_ROOT / "nse" / "manifest.csv"
        man = pd.read_csv(manifest) if manifest.exists() else pd.DataFrame()

        daily, formats, unknown_cols = [], {}, {}
        first_seen, last_seen = {}, {}
        latest_eq = None
        for k, f in enumerate(files, 1):
            try:
                df, fmt, cols = read_one(f)
            except Exception as exc:  # noqa: BLE001
                daily.append({"file": f.name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            formats[fmt] = formats.get(fmt, 0) + 1
            if fmt == "unknown":
                unknown_cols[f.name] = cols[:40]
                continue
            d = pd.to_datetime(df["date"].iloc[0], dayfirst=(fmt == "old"), errors="coerce") if "date" in df else pd.NaT
            eq = df[df["series"] == "EQ"]
            daily.append({"file": f.name, "date": d.date() if pd.notna(d) else None, "format": fmt,
                          "rows": len(df), "eq_rows": len(eq),
                          "eq_traded_value_bn": round(pd.to_numeric(eq.get("value"), errors="coerce").sum() / 1e9, 3),
                          "eq_zero_volume": int((pd.to_numeric(eq.get("volume"), errors="coerce") == 0).sum())})
            key = eq["isin"] if "isin" in eq else eq["symbol"]
            for s in key:
                first_seen.setdefault(s, d)
                last_seen[s] = d
            latest_eq = eq
            if k % 500 == 0:
                print(f"  read {k}/{len(files)} files")

        dd = pd.DataFrame(daily)
        dd.to_csv(out_dir / "daily_summary.csv", index=False)
        ok = dd.dropna(subset=["date"]) if "date" in dd else pd.DataFrame()
        ok = ok.assign(year=pd.to_datetime(ok["date"]).dt.year) if len(ok) else ok
        yearly = ok.groupby("year").agg(trading_days=("date", "count"), median_eq_stocks=("eq_rows", "median"),
                                        min_eq_stocks=("eq_rows", "min"), median_value_bn=("eq_traded_value_bn", "median")
                                        ).reset_index() if len(ok) else pd.DataFrame()
        yearly.to_csv(out_dir / "yearly_summary.csv", index=False)

        fs = pd.Series(first_seen)
        ls = pd.Series(last_seen)
        last_day = ls.max() if len(ls) else pd.NaT
        n_total = len(fs)
        n_gone = int((ls < last_day - pd.Timedelta(days=30)).sum()) if len(ls) else 0

        top = pd.DataFrame()
        if latest_eq is not None and len(latest_eq):
            top = latest_eq.assign(value=pd.to_numeric(latest_eq["value"], errors="coerce")).nlargest(20, "value")[["symbol", "isin", "close", "value"]]
            top.to_csv(out_dir / "latest_top20_by_value.csv", index=False)

        status_counts = man["status"].value_counts().to_dict() if len(man) else {}
        metrics = {
            "files": len(files), "first_date": ok["date"].min() if len(ok) else "", "last_date": ok["date"].max() if len(ok) else "",
            "format_old": formats.get("old", 0), "format_udiff": formats.get("udiff", 0), "format_unknown": formats.get("unknown", 0),
            "read_errors": int(dd["error"].notna().sum()) if "error" in dd else 0,
            "manifest_ok": status_counts.get("ok", 0), "manifest_holiday": status_counts.get("holiday", 0),
            "manifest_failed": status_counts.get("failed", 0),
            "unique_eq_securities": n_total, "securities_no_longer_traded": n_gone,
        }
        with open(out_dir / "metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerows(metrics.items())

        lines = ["# Raw data check — India (NSE bhavcopy)", "", "| Metric | Value |", "|---|---|"] + \
                [f"| {k} | {v} |" for k, v in metrics.items()] + [""]
        if len(yearly):
            lines += ["## By year", "", "| Year | Trading days | Median EQ stocks | Min EQ stocks | Median traded value (bn INR) |",
                      "|---|---|---|---|---|"] + \
                     [f"| {r.year} | {r.trading_days} | {r.median_eq_stocks:.0f} | {r.min_eq_stocks:.0f} | {r.median_value_bn:.1f} |"
                      for r in yearly.itertuples()] + [""]
        if unknown_cols:
            lines += ["## Files with unknown columns (first 5)", ""] + \
                     [f"- {k}: {', '.join(v)}" for k, v in list(unknown_cols.items())[:5]] + [""]
        if len(top):
            lines += ["## Latest day: top 10 EQ stocks by traded value", "", "| Symbol | Close | Value |", "|---|---|---|"] + \
                     [f"| {r.symbol} | {r.close} | {r.value:,.0f} |" for r in top.head(10).itertuples()] + [""]
        lines += ["Survivorship check: securities_no_longer_traded counts EQ securities that stopped appearing "
                  "more than 30 days before the last file (delistings, mergers, series changes). "
                  "A large number is expected and confirms the universe is not survivorship-biased."]
        (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
    finally:
        sys.stdout = tee.stdout
        tee.close()
    z = make_zip(out_dir, STEP)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Upload this file to Claude:\n  {z}")


if __name__ == "__main__":
    main()
