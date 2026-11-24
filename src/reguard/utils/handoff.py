"""Helpers every experiment script uses to write results and the handoff zip.

Each step writes, inside results/<step>/:
    report.md      human-readable summary
    metrics.csv    key numbers
    log.txt        full console log
    env_info.txt   Python / package versions, platform, seed, git commit
    figures/       PDF + PNG figures
    handoff_<step>.zip   everything above, ready to upload to Claude
"""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]   # .../ReGuard-XM
CODE_ROOT = PROJECT_ROOT / "code"
RESULTS_ROOT = PROJECT_ROOT / "results"

TRACKED_PACKAGES = [
    "numpy", "pandas", "pyarrow", "scipy", "sklearn", "lightgbm", "statsmodels", "arch",
    "torch", "gymnasium", "stable_baselines3", "cvxpy", "clarabel", "exchange_calendars",
    "yfinance", "fredapi", "pandas_datareader", "hydra", "yaml", "dotenv", "tqdm",
    "matplotlib", "pytest",
]


def step_dir(step: str) -> Path:
    d = RESULTS_ROOT / step
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


class Tee:
    """Write console output to both the screen and log.txt."""

    def __init__(self, path: Path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def package_versions() -> dict[str, str]:
    out = {}
    for name in TRACKED_PACKAGES:
        try:
            mod = importlib.import_module(name)
            out[name] = str(getattr(mod, "__version__", "installed"))
        except Exception as exc:  # noqa: BLE001
            out[name] = f"MISSING ({type(exc).__name__})"
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=CODE_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "no-git-repo-yet"


def write_env_info(d: Path, seed: int | None = None) -> dict[str, str]:
    versions = package_versions()
    lines = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"processor: {platform.processor()}",
        f"seed: {seed}",
        f"git_commit: {git_commit()}",
        "",
        "packages:",
    ] + [f"  {k}: {v}" for k, v in versions.items()]
    (d / "env_info.txt").write_text("\n".join(lines), encoding="utf-8")
    return versions


def make_zip(d: Path, step: str) -> Path:
    zpath = d / f"handoff_{step}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in d.rglob("*"):
            if p.is_file() and p != zpath and p.suffix != ".zip":
                z.write(p, p.relative_to(d))
    return zpath
