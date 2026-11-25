"""Download raw data for a data tier.

Usage (from the code folder):
    python scripts/download_all.py --tier B
    python scripts/download_all.py --tier A --markets IN --probe   # 6 test dates, ~1 minute
    python scripts/download_all.py --tier A --markets IN           # full India history, ~45-70 minutes
    python scripts/download_all.py --tier B --sources yahoo      # only one source
    python scripts/download_all.py --tier B --day 2026-11-25     # resume into an earlier folder

Expected runtime for Tier B: 5-15 minutes (about 75 Yahoo tickers, 25 FRED series, 12 French files).
Files already downloaded on the same day are skipped, so you can safely re-run after an interruption.
Afterwards run:  python scripts/check_raw_data.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datetime import date  # noqa: E402

from reguard.data.download import french, fred, nse, yahoo  # noqa: E402
from reguard.data.download.common import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="B", choices=["A", "B", "C"])
    ap.add_argument("--sources", default="yahoo,fred,french")
    ap.add_argument("--day", default=None, help="download folder date (default: today)")
    ap.add_argument("--markets", default="IN", help="Tier A markets (currently: IN)")
    ap.add_argument("--probe", action="store_true", help="Tier A: download only a few test dates")
    ap.add_argument("--start", default=None, help="Tier A: override start date (YYYY-MM-DD)")
    args = ap.parse_args()

    if args.tier == "A":
        run_tier_a(args)
        return
    if args.tier != "B":
        print(f"Tier {args.tier} downloaders are not written yet.")
        sys.exit(1)

    cfg = load_config("tier_b.yaml")
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    t0 = time.time()
    summary = {}
    print("ReGuard-XM Tier B download — expected 5-15 minutes.\n")

    if "yahoo" in sources:
        print("== Yahoo Finance: country ETFs")
        summary["yahoo_etfs"] = yahoo.download(cfg["country_etfs"], cfg["start"], "etfs", args.day)
        print("== Yahoo Finance: local indices")
        summary["yahoo_indices"] = yahoo.download(cfg["local_indices"], cfg["start"], "indices", args.day)
    if "fred" in sources:
        print("== FRED")
        try:
            summary["fred"] = fred.download(cfg["fred_series"], cfg["start"], args.day)
        except RuntimeError as exc:
            print(f"  FRED skipped: {exc}")
            summary["fred"] = {"ok": [], "failed": ["ALL (no API key)"], "skipped": []}
    if "french" in sources:
        print("== Kenneth R. French Data Library")
        summary["french"] = french.download(cfg["french_datasets"], args.day)

    print("\n" + "=" * 70)
    print(f"Finished in {(time.time() - t0) / 60:.1f} min")
    any_fail = False
    for k, v in summary.items():
        print(f"  {k:15s} ok={len(v['ok']):3d}  skipped(existing)={len(v['skipped']):3d}  failed={len(v['failed']):3d}")
        if v["failed"]:
            any_fail = True
            print(f"      failed: {', '.join(v['failed'])}")
    if any_fail:
        print("\nSome items failed. Re-run the same command once (it skips finished files).")
        print("If they still fail, that is OK - the check script will report them and Claude will decide.")
    print("\nNext: python scripts/check_raw_data.py")
    print("=" * 70)


def run_tier_a(args):
    markets = [m.strip().upper() for m in args.markets.split(",")]
    if markets != ["IN"]:
        print("Only India (IN) is implemented so far. Japan and the US follow after the India check.")
        sys.exit(1)
    cfg = load_config("tier_a_india.yaml")
    if args.probe:
        dates = [date.fromisoformat(d) for d in cfg["probe_dates"]]
        print(f"India NSE probe: {len(dates)} test dates (about 1 minute)\n")
    else:
        dates = nse.weekdays(args.start or cfg["start"], cfg.get("end"))
        print(f"India NSE full download: {len(dates)} weekdays from {dates[0]} to {dates[-1]}.")
        print("Expected 45-70 minutes. Safe to stop (Ctrl+C) and re-run: finished dates are skipped.\n")
    res = nse.download(cfg, dates, pause=cfg.get("pause_seconds", 0.35))
    c = res["counts"]
    print("\n" + "=" * 70)
    print(f"India NSE: ok={c['ok']}  holiday/no-file={c['holiday']}  failed={c['failed']}  skipped(existing)={c['skipped']}")
    if res["failed_dates"]:
        print(f"Failed dates (first 20): {', '.join(res['failed_dates'][:20])}")
        print("Re-run the same command once; if dates still fail, send Claude the output.")
    if args.probe:
        if c["ok"] >= 4:
            print("Probe looks good. Next: run the full download without --probe.")
        else:
            print("Probe found problems. Send Claude this output before running the full download.")
    else:
        print("Next: python scripts/check_india_raw.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
