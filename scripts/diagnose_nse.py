"""Diagnose why NSE bhavcopy downloads fail on this computer.

Tries several connection methods on two dates (old and new file format) and records exactly
what NSE returns (status code, content type, first bytes, errors). Takes about 1-2 minutes.

Usage (from the code folder):
    python scripts/diagnose_nse.py
Then upload results/nse_diagnose/handoff_nse_diagnose.zip to Claude.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests  # noqa: E402

from reguard.utils.handoff import Tee, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "nse_diagnose"
URLS = {
    "old_nsearchives": "https://nsearchives.nseindia.com/content/historical/EQUITIES/2019/NOV/cm14NOV2019bhav.csv.zip",
    "old_archives": "https://archives.nseindia.com/content/historical/EQUITIES/2019/NOV/cm14NOV2019bhav.csv.zip",
    "new_udiff": "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20241114_F_0000.csv.zip",
}
BROWSER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/all-reports",
}


def describe(r: requests.Response) -> str:
    head = r.content[:120]
    kind = "ZIP (good)" if head[:2] == b"PK" else ("HTML page" if b"<" in head[:20] else "other")
    return (f"HTTP {r.status_code}; content-type={r.headers.get('Content-Type')}; bytes={len(r.content)}; "
            f"looks like: {kind}; first bytes: {head[:80]!r}")


def try_method(name: str, url: str, session: requests.Session, headers: dict | None) -> str:
    try:
        r = session.get(url, headers=headers, timeout=30, allow_redirects=True)
        return describe(r)
    except Exception as exc:  # noqa: BLE001
        return f"EXCEPTION {type(exc).__name__}: {str(exc)[:300]}"


def main():
    out = step_dir(STEP)
    tee = Tee(out / "log.txt")
    sys.stdout = tee
    lines = ["# NSE download diagnosis", ""]
    try:
        write_env_info(out)
        # 0. basic internet and NSE home page
        for label, url in [("google", "https://www.google.com"), ("nse_home", "https://www.nseindia.com"),
                           ("yahoo", "https://query1.finance.yahoo.com")]:
            lines.append(f"- reach {label}: {try_method(label, url, requests.Session(), BROWSER)}")
        lines.append("")
        for key, url in URLS.items():
            lines.append(f"## {key}\n{url}\n")
            # A: plain request
            lines.append(f"- A plain requests: {try_method('A', url, requests.Session(), None)}")
            # B: browser headers
            lines.append(f"- B browser headers: {try_method('B', url, requests.Session(), BROWSER)}")
            # C: visit nseindia.com first to get cookies, then download
            s = requests.Session()
            try:
                s.get("https://www.nseindia.com", headers=BROWSER, timeout=30)
                s.get("https://www.nseindia.com/all-reports", headers=BROWSER, timeout=30)
                cookies = ",".join(s.cookies.keys())
            except Exception as exc:  # noqa: BLE001
                cookies = f"cookie step failed: {type(exc).__name__}"
            lines.append(f"- C cookies first ({cookies}): {try_method('C', url, s, BROWSER)}")
            # D: Windows curl.exe
            curl = shutil.which("curl")
            if curl:
                tmp = out / f"curl_{key}.bin"
                cmd = [curl, "-sS", "-L", "-o", str(tmp), "-w", "%{http_code}", "-A", BROWSER["User-Agent"],
                       "-H", "Referer: https://www.nseindia.com/", url]
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    head = tmp.read_bytes()[:2] if tmp.exists() else b""
                    lines.append(f"- D curl: HTTP {res.stdout.strip()}; zip={head == b'PK'}; stderr={res.stderr.strip()[:200]}")
                    if tmp.exists():
                        tmp.unlink()
                except Exception as exc:  # noqa: BLE001
                    lines.append(f"- D curl: EXCEPTION {type(exc).__name__}: {exc}")
            else:
                lines.append("- D curl: not available")
            lines.append("")
            time.sleep(1)
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
    finally:
        sys.stdout = tee.stdout
        tee.close()
    z = make_zip(out, STEP)
    print(f"\nUpload this file to Claude:\n  {z}")


if __name__ == "__main__":
    main()
