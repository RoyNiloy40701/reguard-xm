"""EXPLORATORY (designed after the confirmatory test, during the manuscript revision; not part of the frozen plan).

Question (referee-type): does the governance layer lower tail risk more than simply holding the same average amount of
cash? For each proposer, the UNGOVERNED daily net return is mixed daily with the T-bill so that the mix's average cash
share equals the GOVERNED portfolio's average cash share over the test period:
    k = (c_gov - c_ungov) / (1 - c_ungov),   r_mix = (1 - k) r_ungov + k r_f   (no extra trading costs charged).
The mix uses the governed portfolio's realised average cash share (hindsight), so it is a benchmark, not a rule.

Reads only results/test_e4/daily_series.csv saved by run_final_test.py (no panel is loaded; no new test access).
Same CVaR / max-drawdown / paired stationary bootstrap (mean block 20, 1,000 resamples, seed 0) and Ledoit-Wolf
Sharpe test as E4. Output: results/test_e4_cash_matched/{cash_matched_comparison.csv, report.md, handoff zip}.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reguard.portfolio import metrics as PM  # noqa: E402
from reguard.utils.handoff import RESULTS_ROOT, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "test_e4_cash_matched"
PROPOSERS = ["equal_weight", "momentum", "mean_variance", "risk_parity", "ppo_ensemble"]


def _stats(r: np.ndarray, rf: np.ndarray) -> dict:
    ex = r - rf
    return {"ann_return": float((1 + r).prod() ** (252 / len(r)) - 1),
            "sharpe": float(ex.mean() / ex.std(ddof=1) * np.sqrt(252)),
            "max_dd": PM.max_drawdown(r), "cvar95": PM.cvar(r)}


def main(src: Path = RESULTS_ROOT / "test_e4" / "daily_series.csv") -> pd.DataFrame:
    d = step_dir(STEP)
    write_env_info(d, seed=0)
    ds = pd.read_csv(src, parse_dates=["date"])
    rows = []
    for p in PROPOSERS:
        u = ds[(ds["proposer"] == p) & (~ds["governed"])].set_index("date")
        g = ds[(ds["proposer"] == p) & (ds["governed"])].set_index("date")
        assert u.index.equals(g.index), p
        cu, cg = float(u["cash"].mean()), float(g["cash"].mean())
        k = (cg - cu) / (1 - cu)
        rf = u["rf"].to_numpy()
        mix = (1 - k) * u["net"].to_numpy() + k * rf
        gov = g["net"].to_numpy()
        sg, sm = _stats(gov, rf), _stats(mix, rf)
        dcv, lo, hi = PM.paired_bootstrap(gov, mix, PM.cvar, n_boot=1000, mean_block=20, seed=0)
        t = PM.sharpe_diff_test(gov - rf, mix - rf)
        rows.append({"proposer": p, "cash_governed": cg, "cash_overlay_k": k,
                     "cvar95_governed": sg["cvar95"], "cvar95_cash_matched": sm["cvar95"],
                     "delta_cvar95": dcv, "ci_lo": lo, "ci_hi": hi,
                     "sharpe_governed": sg["sharpe"], "sharpe_cash_matched": sm["sharpe"],
                     "sharpe_diff_ann": t["sharpe_diff_ann"], "p_two_sided": t["p"],
                     "max_dd_governed": sg["max_dd"], "max_dd_cash_matched": sm["max_dd"],
                     "ann_return_governed": sg["ann_return"], "ann_return_cash_matched": sm["ann_return"]})
    out = pd.DataFrame(rows)
    out.to_csv(d / "cash_matched_comparison.csv", index=False)
    (d / "report.md").write_text("# E4 exploratory: governed vs cash-matched (test period)\n\n```\n"
                                 + out.round(4).to_string(index=False) + "\n```\n", encoding="utf-8")
    print(out.round(4).to_string(index=False))
    print(f"Handoff: {make_zip(d, STEP)}")
    return out


if __name__ == "__main__":
    main()
