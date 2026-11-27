"""Revision (audit A-05): ETF inventory table from the DEVELOPMENT panel only (no test-period access).

For each ETF in configs/data/tier_b.yaml: ticker, market, fund name, first and last date in the development panel,
first date with a non-missing 1-day return, number of return days, and the first year the ETF is investable in E4
(after 252 return days). Test-period presence is not read here (the test panel stays locked); all 43 ETFs were
listed at the data snapshot date.

Usage:  python scripts/etf_inventory.py   ->  results/etf_inventory/etf_inventory.csv (+ handoff zip)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from reguard.data.store import load_panel  # noqa: E402
from reguard.utils.handoff import CODE_ROOT, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "etf_inventory"


def main() -> None:
    d = step_dir(STEP)
    write_env_info(d, seed=0)
    cfg = yaml.safe_load(open(CODE_ROOT / "configs" / "data" / "tier_b.yaml", encoding="utf-8"))
    etfs = pd.DataFrame(cfg["country_etfs"]).rename(columns={"ticker": "asset", "country": "market", "name": "fund"})
    p = load_panel("tierb_etf")                       # development part only
    p["date"] = pd.to_datetime(p["date"])
    rows = []
    for a in etfs["asset"]:
        x = p[p["asset"] == a].sort_values("date")
        r = x[x["ret"].notna()] if "ret" in x else x
        n = len(r)
        investable = r["date"].iloc[252] if n > 252 else pd.NaT
        rows.append({"asset": a, "first_date": x["date"].min(), "last_dev_date": x["date"].max(),
                     "first_return_date": r["date"].min(), "return_days_dev": n,
                     "investable_from": investable})
    out = etfs.merge(pd.DataFrame(rows), on="asset", how="left")
    out.to_csv(d / "etf_inventory.csv", index=False)
    (d / "report.md").write_text("# ETF inventory (development panel only)\n\n" + "```\n" + out.to_string(index=False) + "\n```",
                                 encoding="utf-8")
    print(out.to_string(index=False))
    print(f"Handoff: {make_zip(d, STEP)}")


if __name__ == "__main__":
    main()
