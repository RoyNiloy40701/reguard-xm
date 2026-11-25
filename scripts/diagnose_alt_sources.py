"""Check whether alternative stock-level sources are reachable from this computer.

Tests BSE India (old and new daily files) and the Dhaka Stock Exchange. About 1 minute.

Usage (from the code folder):
    python scripts/diagnose_alt_sources.py
Then upload results/alt_sources_diagnose/handoff_alt_sources_diagnose.zip to Claude.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests  # noqa: E402

from reguard.utils.handoff import Tee, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "alt_sources_diagnose"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.bseindia.com/markets/MarketInfo/BhavCopy.aspx",
}
TESTS = [
    ("bse_home", "https://www.bseindia.com/"),
    ("bse_bhavcopy_page", "https://www.bseindia.com/markets/MarketInfo/BhavCopy.aspx"),
    ("bse_old_2018", "https://www.bseindia.com/download/BhavCopy/Equity/EQ040518_CSV.ZIP"),
    ("bse_old_2012", "https://www.bseindia.com/download/BhavCopy/Equity/EQ140612_CSV.ZIP"),
    ("bse_udiff_2024", "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_20241114_F_0000.CSV"),
    ("dse_home", "https://www.dsebd.org/"),
    ("dse_day_end_archive", "https://www.dsebd.org/day_end_archive.php"),
    ("jquants_home", "https://jpx-jquants.com/"),
    ("eodhd_home", "https://eodhd.com/"),
]


def describe(r: requests.Response) -> str:
    head = r.content[:60]
    kind = "ZIP" if head[:2] == b"PK" else ("HTML" if b"<" in head[:40] else "CSV/text" if b"," in head else "other")
    return f"HTTP {r.status_code}; type={r.headers.get('Content-Type')}; bytes={len(r.content)}; looks like {kind}; start={head!r}"


def main():
    out = step_dir(STEP)
    tee = Tee(out / "log.txt")
    sys.stdout = tee
    lines = ["# Alternative source reachability", ""]
    try:
        write_env_info(out)
        s = requests.Session()
        for name, url in TESTS:
            try:
                r = s.get(url, headers=HEADERS, timeout=30)
                lines.append(f"- {name}: {describe(r)}\n  {url}")
            except Exception as exc:  # noqa: BLE001
                lines.append(f"- {name}: EXCEPTION {type(exc).__name__}: {str(exc)[:200]}\n  {url}")
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
    finally:
        sys.stdout = tee.stdout
        tee.close()
    print(f"\nUpload this file to Claude:\n  {make_zip(out, STEP)}")


if __name__ == "__main__":
    main()
