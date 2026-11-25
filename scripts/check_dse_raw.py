"""Inspect the two Mendeley Dhaka Stock Exchange (DSE) datasets after manual download.

Expected folders (unzip each Mendeley download into its folder; any sub-folder layout is fine):
    data/raw/dse_mendeley/23553sm4tn_v4/   (EOD with coverage metadata, 2012-2026, raw + adjusted)
    data/raw/dse_mendeley/5mww8rb9td_v1/   (historical 1999-2025)

Usage (from the code folder):
    python scripts/check_dse_raw.py
Writes results/raw_check_dse/ and handoff_raw_check_dse.zip. Raw files are only read.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from reguard.data.download.common import RAW_ROOT  # noqa: E402
from reguard.utils.handoff import Tee, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "raw_check_dse"
BASE = RAW_ROOT / "dse_mendeley"
DATASETS = {"23553sm4tn_v4": "EOD with coverage metadata (2012-2026)", "5mww8rb9td_v1": "Historical 1999-2025"}
DATE_NAMES = ("date", "trading date", "tradedate")
CODE_NAMES = ("trading code", "tradingcode", "symbol", "ticker", "code", "instrument")
CLOSE_NAMES = ("close", "closing price", "close price", "closep", "ltp")


def find_col(cols: list[str], names: tuple[str, ...]) -> str | None:
    low = {c.lower().strip(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    for c in cols:
        if any(n in c.lower() for n in names):
            return c
    return None


def scan_dataset(folder: Path, label: str, out: Path) -> tuple[dict, list[str]]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls"))
    lines = [f"## {label}", f"Folder: `{folder}`", f"Data files found: {len(files)}", ""]
    if not files:
        return {"files": 0}, lines + ["No CSV/XLSX files found. Did you unzip the download into this folder?", ""]

    # group by sub-folder to show the structure (e.g., raw/ vs adjusted/ vs metadata)
    groups: dict[str, list[Path]] = {}
    for f in files:
        groups.setdefault(str(f.parent.relative_to(folder)) or ".", []).append(f)
    lines += ["| Sub-folder | Files | Example file |", "|---|---|---|"] + \
             [f"| {g} | {len(v)} | {v[0].name} |" for g, v in groups.items()] + [""]

    # preview the first file of each sub-folder
    for g, v in groups.items():
        try:
            df = pd.read_csv(v[0], nrows=5) if v[0].suffix.lower() == ".csv" else pd.read_excel(v[0], nrows=5)
            lines += [f"**Preview `{g}/{v[0].name}`** — columns: {', '.join(map(str, df.columns))}", "",
                      "```", df.head(3).to_string()[:1200], "```", ""]
        except Exception as exc:  # noqa: BLE001
            lines.append(f"Could not read {v[0].name}: {type(exc).__name__}: {exc}")

    # instrument-level summary: read all price files (skip obvious metadata files)
    meta_words = ("metadata", "matrix", "coverage", "availability", "readme", "summary")
    price_files = [f for f in files if not any(w in f.name.lower() for w in meta_words) and f.suffix.lower() == ".csv"]
    rows = []
    for k, f in enumerate(price_files, 1):
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            rows.append({"file": str(f.relative_to(folder)), "error": f"{type(exc).__name__}"})
            continue
        dcol, ccol, pcol = find_col(list(df.columns), DATE_NAMES), find_col(list(df.columns), CODE_NAMES), \
            find_col(list(df.columns), CLOSE_NAMES)
        if dcol is None:
            rows.append({"file": str(f.relative_to(folder)), "error": "no date column"})
            continue
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce", dayfirst=False)
        keys = df[ccol].astype(str).str.strip() if ccol and df[ccol].nunique() > 1 else pd.Series([f.stem] * len(df))
        for code, g in df.groupby(keys):
            px = pd.to_numeric(g[pcol], errors="coerce") if pcol else pd.Series(dtype=float)
            ret = px.sort_index().pct_change(fill_method=None).abs() if len(px) else px
            rows.append({"file": str(f.relative_to(folder)), "instrument": code,
                         "rows": len(g), "first": g[dcol].min(), "last": g[dcol].max(),
                         "missing_close_pct": round(100 * px.isna().mean(), 2) if len(px) else None,
                         "n_moves_gt_50pct": int((ret > 0.5).sum()) if len(ret) else None})
        if k % 200 == 0:
            print(f"    {label}: read {k}/{len(price_files)} files")
    inst = pd.DataFrame(rows)
    inst.to_csv(out / f"instruments_{folder.name}.csv", index=False)
    ok = inst.dropna(subset=["first"]) if "first" in inst else pd.DataFrame()
    summary = {"files": len(files), "price_files": len(price_files), "instruments": int(ok["instrument"].nunique()) if len(ok) else 0}
    if len(ok):
        last_all = ok["last"].max()
        summary.update(first=str(ok["first"].min().date()), last=str(last_all.date()),
                       stopped_trading_before_end=int((ok["last"] < last_all - pd.Timedelta(days=60)).sum()),
                       instruments_with_big_moves=int((ok["n_moves_gt_50pct"].fillna(0) > 0).sum()))
        years = range(ok["first"].min().year, last_all.year + 1)
        active = [{"year": y, "active_instruments": int(((ok["first"].dt.year <= y) & (ok["last"].dt.year >= y)).sum())}
                  for y in years]
        lines += ["**Active instruments per year**", "", "| Year | Active |", "|---|---|"] + \
                 [f"| {a['year']} | {a['active_instruments']} |" for a in active] + [""]
    lines += ["**Summary:** " + "; ".join(f"{k}={v}" for k, v in summary.items()), ""]
    return summary, lines


def main():
    out = step_dir(STEP)
    tee = Tee(out / "log.txt")
    sys.stdout = tee
    t0 = time.time()
    try:
        write_env_info(out)
        lines = ["# Raw data check — Dhaka Stock Exchange (Mendeley datasets)", ""]
        metrics = {}
        for key, label in DATASETS.items():
            folder = BASE / key
            if not folder.exists():
                lines += [f"## {label}", f"Folder missing: `{folder}`", ""]
                continue
            print(f"Scanning {key} ...")
            s, l = scan_dataset(folder, label, out)
            lines += l
            metrics.update({f"{key}_{k}": v for k, v in s.items()})
        with open(out / "metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerows(metrics.items())
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
    finally:
        sys.stdout = tee.stdout
        tee.close()
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Upload this file to Claude:\n  {make_zip(out, STEP)}")


if __name__ == "__main__":
    main()
