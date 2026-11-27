"""Phase 10: build the reproducibility package ReGuard-XM_v1.0.zip (for Zenodo / journal supplementary material).

Usage (from code/):  python scripts/make_release.py [--version v1.0-paper] [--allow-dirty]

Contents
  code/        every file tracked by git (exactly what `git ls-files` lists; .env is never tracked)
  docs/        research log, frozen analysis plan, test-access log, data dictionary, registers
  literature/  protocol, search log, screening, comparison matrix, PRISMA (no third-party PDFs)
  paper/       LaTeX sources, references, generated figures and tables
  results/     the paper's result steps: csv / md / txt / pdf / png only (no zips, no data)
  README.md, LICENSE-docs.md (CC BY 4.0; code/LICENSE is MIT), MANIFEST.sha256 (SHA-256 of every file), RELEASE_INFO.txt (commit, plan hash, date)
Not included: data/ (licensed vendor data; see docs/licence_register.md and README for how to rebuild it).

Safety: the build stops if any file would contain the FRED key from code/.env or an API-key URL parameter,
and if the git working tree has uncommitted changes (unless --allow-dirty).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reguard.utils.handoff import CODE_ROOT, PROJECT_ROOT, RESULTS_ROOT  # noqa: E402

RESULT_STEPS = ["data_quality", "e1", "e2", "e3", "e3_sensitivity", "e4", "e4_ppo", "e8",
                "test_e1", "test_e1_stage", "test_e1_floor", "test_e2_stage", "test_e3_stage", "test_e4",
                "test_e4_stage", "test_e8_stage", "test_summary_stage",
                "test_e4_cash_matched", "etf_inventory"]   # last two: exploratory / descriptive, reported in the paper
RESULT_EXT = {".csv", ".md", ".txt", ".pdf", ".png"}
BINARY_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".parquet", ".pkl", ".pt", ".zst", ".gz"}
TEXT_EXT = {".py", ".md", ".csv", ".txt", ".yaml", ".yml", ".tex", ".bib", ".cfg", ".toml", ".ini", ""}
# Old Phase 10/11 drafts kept in paper/ for reference; superseded by paper/sections and not part of the release.
PAPER_EXCLUDE = {"appendix_theory.tex", "results_e1.tex", "results_e2.tex", "results_e3.tex", "results_e4.tex",
                 "results_e8.tex", "sec2_related.tex", "sec2_related_preview.pdf", "sn-mathphys-ay.bst"}
# Generic secret patterns: KEY/TOKEN/SECRET assignments with a long value, and FRED-style 32-hex keys on such lines.
SECRET_RE = re.compile(rb"(?i)(api[_-]?key|token|secret|password)[\"']?\s*[=:]\s*[\"']?([A-Za-z0-9_\-]{16,})")
PLACEHOLDER = re.compile(rb"(?i)put_your|your_|xxxx|example|placeholder")
RENAME = {"docs/preregistration.md": "docs/frozen_analysis_plan.md"}   # content (and its SHA-256) unchanged
PLAN = PROJECT_ROOT / "docs" / "preregistration.md"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=CODE_ROOT, text=True).strip()


def secrets() -> list[bytes]:
    pats = [b"api_key" + b"="]  # split so this file does not match itself
    env = CODE_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if len(v) >= 12 and "put_your" not in v:
                    pats.append(v.encode())
    return pats


def collect() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for rel in git("ls-files").splitlines():
        files.append((CODE_ROOT / rel, f"code/{rel}"))
    for sub, keep in (("docs", lambda p: True),
                      ("literature", lambda p: p.suffix != ".pdf" or p.name.startswith("prisma")),
                      ("paper", lambda p: p.suffix not in {".aux", ".log", ".out", ".synctex", ".fls",
                                                           ".fdb_latexmk", ".gz", ".bbl", ".blg"}
                       and not (p.parent.name == "paper" and p.name in PAPER_EXCLUDE)
                       and not (p.parent.name == "submission" and p.name in PAPER_EXCLUDE))):
        root = PROJECT_ROOT / sub
        if root.exists():
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.name not in {".gitkeep", ".env"} and keep(p):
                    files.append((p, p.relative_to(PROJECT_ROOT).as_posix()))
    for step in RESULT_STEPS:
        d = RESULTS_ROOT / step
        if not d.exists():
            print(f"  note: results/{step} not found (skipped)")
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in RESULT_EXT:
                files.append((p, p.relative_to(PROJECT_ROOT).as_posix()))
    ho = RESULTS_ROOT / "hypothesis_outcomes.md"
    if ho.exists():
        files.append((ho, "results/hypothesis_outcomes.md"))
    for top in ("README.md", "LICENSE-docs.md"):
        if (PROJECT_ROOT / top).exists():
            files.append((PROJECT_ROOT / top, top))
    return [(p, RENAME.get(a, a)) for p, a in files]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-paper")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    dirty = git("status", "--porcelain")
    if dirty and not a.allow_dirty:
        sys.exit("git working tree has uncommitted changes — commit first (or pass --allow-dirty):\n" + dirty)
    commit = git("rev-parse", "HEAD")
    files = collect()
    pats = secrets()
    bad = []
    for p, arc in files:
        if p.suffix.lower() not in BINARY_EXT:   # scan every non-binary file (incl. .env.example)
            data = p.read_bytes()
            if any(s in data for s in pats):
                bad.append(arc)
                continue
            for m in SECRET_RE.finditer(data):
                if not PLACEHOLDER.search(m.group(2)) and not m.group(2).startswith(b"os"):
                    bad.append(arc + " (" + m.group(1).decode(errors="ignore") + "=...)")
                    break
    if bad:
        sys.exit("STOPPED: possible API key found in: " + ", ".join(bad) + "\nRemove it and run again.")
    plan_hash = sha256(PLAN) if PLAN.exists() else "missing"
    info = "\n".join([f"package: ReGuard-XM {a.version}", f"built: {datetime.now().isoformat(timespec='seconds')}",
                      f"git_commit: {commit}" + (" (working tree had uncommitted changes)" if dirty else ""),
                      f"frozen_analysis_plan_sha256: {plan_hash}",
                      "note: docs/frozen_analysis_plan.md is docs/preregistration.md renamed (file name only; content "
                      "and SHA-256 unchanged); the plan was fixed before test access, its hash is embedded in "
                      "scripts/run_final_test.py, and it was not posted on a public registry.",
                      f"files: {len(files)}"])
    out_dir = PROJECT_ROOT / "release"
    out_dir.mkdir(exist_ok=True)
    name = f"ReGuard-XM_{a.version}"
    zpath = out_dir / f"{name}.zip"
    manifest = [f"{sha256(p)}  {arc}" for p, arc in files]
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in files:
            z.write(p, f"{name}/{arc}")
        z.writestr(f"{name}/MANIFEST.sha256", "\n".join(manifest) + "\n")
        z.writestr(f"{name}/RELEASE_INFO.txt", info + "\n")
    size = zpath.stat().st_size / 1e6
    print(info)
    print(f"\nRelease written: {zpath}  ({size:.1f} MB, {len(files)} files + MANIFEST + RELEASE_INFO)")
    # wording guard: the plan was not publicly registered, so the paper must not claim "pre-registered"
    for t in sorted((PROJECT_ROOT / "paper").rglob("*.tex")):
        for i, line in enumerate(t.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if re.search(r"pre-?regist", line, re.I) and not line.lstrip().startswith("%"):
                print(f"  WARNING wording: {t.name}:{i} mentions pre-registration — use 'analysis plan fixed "
                      f"before test access'")


if __name__ == "__main__":
    main()
