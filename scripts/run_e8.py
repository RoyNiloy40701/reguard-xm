"""E8 / C6: drivers of transfer quality (H8). Needs the E2 episode pool (data/interim/e2/). Development data only.

Usage (from the code folder):
    python scripts/run_e8.py            # full (1,000 bootstrap resamples; ~15-30 min)
    python scripts/run_e8.py --quick    # e2_quick pool, 30 resamples
Writes results/e8/{report.md, importance.csv, bootstrap_top3.csv, market_map.csv, e1_link.csv, figures/, log.txt,
env_info.txt} and results/e8/handoff_e8.zip.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

from reguard.data.store import load_panel, read_table  # noqa: E402
from reguard.tre import drivers as DR  # noqa: E402
from reguard.utils.handoff import CODE_ROOT, PROJECT_ROOT, RESULTS_ROOT, Tee, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "e8"


def md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    d = step_dir(STEP)
    tee = Tee(d / "log.txt"); sys.stdout = tee
    with open(CODE_ROOT / "configs" / "experiment" / "e8.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    write_env_info(d, seed=cfg["seed"])
    log = lambda s: print(s, flush=True)  # noqa: E731
    T0 = time.time()
    ep = read_table(PROJECT_ROOT / "data" / "interim" / ("e2_quick" if args.quick else "e2"), "episodes")
    D = DR.driver_table(ep, load_panel("tierb_etf"))
    D = D[D["g_cal"].notna()].reset_index(drop=True)
    log(f"E8: {len(D)} episodes, {D['market'].nunique()} markets")
    imp, r2 = DR.grouped_importance(D, cfg["model"], cfg["seed"], cfg["n_folds"], cfg["n_perm"])
    imp_df = imp.rename("mse_increase").reset_index().rename(columns={"index": "driver_group"})
    imp_df["share_of_total"] = imp_df["mse_increase"] / imp_df["mse_increase"].clip(lower=0).sum()
    imp_df.to_csv(d / "importance.csv", index=False)
    log(f"  out-of-fold R2 of the driver model: {r2:.4f}")
    nb = 30 if args.quick else cfg["n_boot"]
    cnt, sets = DR.bootstrap_top3(D, cfg["model"], nb, cfg["seed"], log)
    top = pd.DataFrame([{"top3": " + ".join(sorted(s)), "share": c / nb} for s, c in cnt.most_common()])
    top.to_csv(d / "bootstrap_top3.csv", index=False)
    full_top3 = frozenset(imp.index[:3])
    share_full = cnt.get(full_top3, 0) / nb
    member = pd.Series([g for s in sets for g in s]).value_counts() / nb
    h8 = {"full_sample_top3": " + ".join(sorted(full_top3)), "share_resamples_with_same_top3": share_full,
          "modal_top3": top["top3"].iloc[0], "modal_share": float(top["share"].iloc[0]),
          "criterion": f">= {cfg['h8_share']}", "supported_dev": share_full >= cfg["h8_share"], "oof_r2": r2,
          "model_has_signal": r2 > 0}

    # transfer-risk map (descriptive)
    thr = D["g_cal"].quantile(2 / 3)
    mp = D.groupby("market").agg(episodes=("g_cal", "size"), mean_g_cal=("g_cal", "mean"),
                                 failure_rate=("g_cal", lambda x: float((x > thr).mean())),
                                 vol_level=("vol_level", "mean"), corr_src=("corr_src", "mean"),
                                 atypicality=("atypicality", "mean"), em=("em", "max")).reset_index()
    mp = mp.sort_values("failure_rate", ascending=False)
    mp.to_csv(d / "market_map.csv", index=False)
    # link to E1: per-market zero-shot minus in-domain OOS R2 (GBM), if E1 results exist
    link = pd.DataFrame()
    e1 = RESULTS_ROOT / "e1" / "metrics_ts_by_asset.csv"
    if e1.exists():
        b = pd.read_csv(e1)
        b = b[b["model"] == "gbm"].pivot_table(index="market", columns="setting", values="r2_oos")
        if {"lomo", "indomain"} <= set(b.columns):
            gap = (b["lomo"] - b["indomain"]).rename("zs_minus_in_r2")
            j = mp.set_index("market").join(gap, how="inner")
            link = pd.DataFrame([{"driver": c, "spearman_with_zs_minus_in": float(stats.spearmanr(j[c], j["zs_minus_in_r2"]).statistic),
                                  "p": float(stats.spearmanr(j[c], j["zs_minus_in_r2"]).pvalue), "markets": len(j)}
                                 for c in ["vol_level", "corr_src", "atypicality", "em", "failure_rate"]])
    link.to_csv(d / "e1_link.csv", index=False)

    figs = d / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(imp_df["driver_group"][::-1], imp_df["mse_increase"][::-1])
    axes[0].set_title(f"Grouped permutation importance for g_cal (OOF R² = {r2:.3f})", fontsize=9)
    axes[1].bar(mp["market"], mp["failure_rate"], color=["C3" if e else "C0" for e in mp["em"]])
    axes[1].axhline(1 / 3, color="k", ls=":", lw=0.8)
    axes[1].set_title("Transfer-risk map: share of failed episodes by market (red = EM/frontier)", fontsize=9)
    axes[1].tick_params(axis="x", rotation=90, labelsize=6)
    fig.tight_layout(); fig.savefig(figs / "e8_drivers_map.png", dpi=150); fig.savefig(figs / "e8_drivers_map.pdf"); plt.close(fig)

    lines = ["# E8 — drivers of transfer quality (C6, H8; development)", "",
             f"Quick: {args.quick}. Episodes {len(D)}, markets {D['market'].nunique()}, bootstrap resamples {nb}. "
             f"Runtime {time.time()-T0:.0f}s.", "",
             "## H8 (development check; the confirmatory test also requires the same top-3 on the locked test)", "",
             md(pd.DataFrame([h8])), "",
             ("**Caution: the driver model has no out-of-sample skill (R² <= 0), so the importance ranking below is not "
              "interpretable; H8 should be read as 'no stable, identifiable drivers'.**" if r2 <= 0 else
              "The driver model has positive out-of-sample skill; importances are interpretable."), "",
             "## Grouped permutation importance (out-of-fold MSE increase)", "", md(imp_df), "",
             "## How often each group is in the top 3 across resamples", "",
             md(member.rename("share_in_top3").reset_index().rename(columns={"index": "driver_group"})), "",
             "## Most frequent top-3 sets", "", md(top.head(10)), "",
             "## Transfer-risk map (descriptive)", "", md(mp), "",
             "## Link to E1 (per-market zero-shot minus in-domain OOS R², GBM)", "", md(link), "",
             "Figure: e8_drivers_map."]
    (d / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"E8 done in {time.time()-T0:.0f}s")
    sys.stdout = tee.stdout; tee.close()
    print(f"Handoff: {make_zip(d, STEP)}")


if __name__ == "__main__":
    main()
