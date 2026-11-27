"""EXPLORATORY (designed after the confirmatory test; not part of the frozen analysis plan).

Question: is the high test-period DSE in-domain rank IC (0.18 vs 0.085 in development) driven by the
2022-07-28 to 2024-01-21 floor-price regime, when stale prices made ranks mechanically persistent?

Reads the E1 test-stage cross-sectional predictions already saved by run_final_test.py
(data/interim/test_e1_preds/cs_predictions.*). It does NOT load any panel, so no new test-data access
occurs and nothing is re-fitted. Uses the same rank-IC, Newey-West and block-bootstrap code as E1.

Subsets of test evaluation dates (DSE target):
  all            every test date with a prediction
  outside floor  dates whose 20-trading-day target window does not touch the floor regime
                 (date < floor_start - 31 calendar days, or date > floor_end)
  floor only     dates inside the floor regime
Output: results/test_e1_floor/{floor_sensitivity.csv, report.md, handoff_test_e1_floor.zip}
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from reguard.data.store import load_yaml, read_table  # noqa: E402
from reguard.evaluation import metrics as M  # noqa: E402
from reguard.utils.handoff import CODE_ROOT, PROJECT_ROOT, Tee, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "test_e1_floor"
PRED = PROJECT_ROOT / "data" / "interim" / "test_e1_preds"
NAMES = {"dse": "DSE", "tierb": "Country ETFs", "pooled": "Pooled"}


def main() -> None:
    d = step_dir(STEP)
    sys.stdout = Tee(d / "log.txt")
    write_env_info(d, seed=0)
    t0 = time.time()
    print("EXPLORATORY: E1 DSE floor-price sensitivity on saved test predictions (no panel loaded)")
    P = read_table(PRED, "cs_predictions")
    P["date"] = pd.to_datetime(P["date"])
    P = P[P["target"] == "dse"]
    f0, f1 = (pd.Timestamp(x) for x in load_yaml("processing.yaml")["dse"]["regimes"]["floor_price_2"])
    with open(CODE_ROOT / "configs" / "experiment" / "e1.yaml", encoding="utf-8") as fh:
        min_n = yaml.safe_load(fh)["cs"]["min_assets"]
    subsets = {"all": P["date"].notna(),
               "outside floor": (P["date"] < f0 - pd.Timedelta(days=31)) | (P["date"] > f1),
               "floor only": (P["date"] >= f0) & (P["date"] <= f1)}
    rows = []
    for sub, mask in subsets.items():
        for (src, m), g in P[mask].groupby(["source", "model"], observed=True):
            ic = M.daily_cs_ic(g, "pred", "y_rank", min_n=min_n)
            if len(ic) < 30:
                continue
            mu, tt = M.nw_tstat(ic.to_numpy(), lag=20)
            lo, hi = M.block_bootstrap_ci(ic, block=63)
            rows.append({"subset": sub, "source": src, "model": m, "days": len(ic), "rank_ic_mean": mu,
                         "ci_lo": lo, "ci_hi": hi, "nw_t": tt, "share_pos": float((ic > 0).mean())})
    S = pd.DataFrame(rows).sort_values(["source", "model", "subset"])
    S.to_csv(d / "floor_sensitivity.csv", index=False)
    lines = ["# E1 DSE floor-price sensitivity (EXPLORATORY, designed after the test results)", "",
             f"Floor regime {f0.date()} to {f1.date()}; 'outside floor' also drops dates whose 20-day target "
             "window would reach into it.", "",
             "| subset | source | model | days | rank IC | 95% CI | NW t |", "|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        lines.append(f"| {r['subset']} | {NAMES[r['source']]} | {r['model']} | {r['days']} | {r['rank_ic_mean']:.3f} "
                     f"| [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {r['nw_t']:.1f} |")
    (d / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"done in {time.time() - t0:.0f}s\nHandoff: {make_zip(d, STEP)}")


if __name__ == "__main__":
    main()
