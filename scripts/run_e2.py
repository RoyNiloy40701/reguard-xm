"""E2: pseudo-transfer episodes + Transfer Reliability Estimator (N1, H1). Development data only.

Usage (from the code folder):
    python scripts/run_e2.py --quick        # 4 years x 12 markets, pipeline check (~10-20 min)
    python scripts/run_e2.py                # full run (episodes are cached in data/interim/e2/)
    python scripts/run_e2.py --reuse        # re-evaluate the TRE on cached episodes (minutes)
Writes results/e2/{report.md, scores.csv, h1.csv, episodes_summary.csv, reliability.csv, lr_coefficients.csv,
figures/, log.txt, env_info.txt} and results/e2/handoff_e2.zip.
The episode pool (with out-of-sample rho) is saved for Phase 6 in data/interim/e2/.
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
from sklearn.metrics import roc_curve  # noqa: E402

from reguard.data.splits import annual_refit_plan  # noqa: E402
from reguard.data.store import PROCESSED_ROOT, load_panel, load_yaml, read_table, write_table  # noqa: E402
from reguard.episodes.pseudo_transfer import generate  # noqa: E402
from reguard.tre import estimator as T  # noqa: E402
from reguard.utils.handoff import CODE_ROOT, PROJECT_ROOT, Tee, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "e2"
WCOLS = ["train_start", "train_end", "embargo_start", "eval_start", "eval_end"]


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
    ap.add_argument("--reuse", action="store_true", help="reuse cached episodes")
    args = ap.parse_args()
    d = step_dir(STEP)
    tee = Tee(d / "log.txt"); sys.stdout = tee
    with open(CODE_ROOT / "configs" / "experiment" / "e2.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pcfg = load_yaml("processing.yaml")
    write_env_info(d, seed=cfg["seed"])
    T0 = time.time()
    cache = PROJECT_ROOT / "data" / "interim" / ("e2_quick" if args.quick else "e2")
    labels = [cfg["labels"]["primary"]] + cfg["labels"]["secondary"]
    print(f"E2 start (quick={args.quick}, reuse={args.reuse})", flush=True)

    if args.reuse:
        ep, rows = read_table(cache, "episodes_raw"), None
        print(f"  loaded {len(ep)} cached episodes", flush=True)
    else:
        df = load_panel(cfg["panel"])
        splits = pd.read_csv(PROCESSED_ROOT / "splits_tierb.csv", parse_dates=WCOLS)
        cal = pd.DatetimeIndex(sorted(df["date"].unique()))
        plan = annual_refit_plan(splits, cal, cfg["episodes"]["calibration_days"], pcfg["embargo_days"])
        markets = None
        if args.quick:
            plan = [p for p in plan if p["year"] in (2012, 2013, 2014, 2015)]
            markets = ["US", "JP", "GB", "DE", "FR", "CA", "AU", "BR", "MX", "KR", "TW", "ZA"]
            cfg["tre"]["min_train_episodes"] = 60
            cfg["tre"]["market_folds"] = 4
        print(f"  plan years {plan[0]['year']}-{plan[-1]['year']}", flush=True)
        ep, rows = generate(df, plan, cfg, log=lambda s: print(s, flush=True), markets=markets)
        write_table(ep, cache, "episodes_raw")
        write_table(rows, cache, "episode_rows")
    if args.quick:
        cfg["tre"]["min_train_episodes"] = 60
        cfg["tre"]["market_folds"] = 4
    print(f"  episodes: {len(ep)} over {ep['market'].nunique()} markets, {ep['year'].nunique()} years "
          f"({time.time()-T0:.0f}s)", flush=True)

    R = T.run_evaluation(ep, cfg, labels, cfg["seed"], log=lambda s: print(s, flush=True))
    orient = cfg["tre"]["signals"]
    scores = T.score_table(R, orient)
    h1 = pd.DataFrame([T.h1_test(R, lab, f"rho_{k}", orient, cfg["tre"]["bootstrap"]["n_boot"], cfg["seed"])
                       for lab in labels for k in ("lr", "gbm")])
    rel = pd.concat([T.reliability_bins(R, labels[0], f"rho_{k}").assign(tre=k) for k in ("lr", "gbm")])

    # descriptive (in-sample) logistic coefficients on all episodes with a primary label, for interpretation only
    lab0 = labels[0]
    e0 = ep[ep[lab0].notna()]
    y0 = (e0[lab0] > e0[lab0].quantile(cfg["labels"]["failure_quantile"])).to_numpy(int)
    base = T._base("lr", list(orient), orient, cfg["tre"], cfg["seed"]).fit(e0[list(orient)], y0)
    coefs = pd.DataFrame({"signal": list(orient), "coef_std_units": base[-1].coef_[0]}).sort_values("coef_std_units", key=np.abs, ascending=False)

    # save the episode pool (with out-of-sample rho for the primary label) for Phase 6
    r0 = R[R["label"] == lab0][["episode_id", "rho_lr", "rho_gbm", "g_star", "y_fail"]]
    ep_out = ep.merge(r0, on="episode_id", how="left")
    write_table(ep_out, cache, "episodes")

    # per-year AUROC of the primary TRE and the best single signal
    best = h1[(h1["label"] == lab0) & (h1["tre"] == "rho_lr")]["best_single"].iloc[0]
    py = []
    for Y, g in R[R["label"] == lab0].groupby("year"):
        y = g["y_fail"].to_numpy()
        py.append({"year": Y, "n": len(g), "auroc_tre_lr": T._auc(y, g["rho_lr"].to_numpy(float)),
                   "auroc_tre_gbm": T._auc(y, g["rho_gbm"].to_numpy(float)),
                   f"auroc_{best}": T._auc(y, g[f"sig_{best}"].to_numpy(float))})
    py = pd.DataFrame(py)

    summ = ep.groupby("year").agg(episodes=("episode_id", "size"), markets=("market", "nunique"),
                                  cov_zs=("cov_zs", "mean"), cov_in=("cov_in", "mean"), g_cal=("g_cal", "mean"),
                                  g_ic=("g_ic", "mean"), g_cvar=("g_cvar", "mean")).reset_index()
    scores.to_csv(d / "scores.csv", index=False); h1.to_csv(d / "h1.csv", index=False)
    rel.to_csv(d / "reliability.csv", index=False); coefs.to_csv(d / "lr_coefficients.csv", index=False)
    summ.to_csv(d / "episodes_summary.csv", index=False); py.to_csv(d / "per_year_auroc.csv", index=False)

    # figures
    figs = d / "figures"
    g = R[R["label"] == lab0]
    y = g["y_fail"].to_numpy()
    top = scores[(scores["label"] == lab0) & ~scores["scorer"].str.startswith("TRE")].nlargest(3, "auroc")["scorer"].tolist()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, col in [("TRE-LR", "rho_lr"), ("TRE-GBM", "rho_gbm")] + [(s, f"sig_{s}") for s in top]:
        s = g[col].to_numpy(float); ok = np.isfinite(s)
        fpr, tpr, _ = roc_curve(y[ok], s[ok])
        ax.plot(fpr, tpr, lw=1.6 if name.startswith("TRE") else 1, ls="-" if name.startswith("TRE") else "--",
                label=f"{name} ({T._auc(y, s):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title(f"Transfer-failure detection ({lab0}), out-of-sample episodes", fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figs / "e2_roc.png", dpi=150); fig.savefig(figs / "e2_roc.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    for k, gr in rel.groupby("tre"):
        ax.plot(gr["mean_rho"], gr["fail_rate"], "o-", label=f"TRE-{k.upper()}")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_xlabel("predicted failure probability ρ"); ax.set_ylabel("observed failure rate")
    ax.set_title("Reliability diagram (primary label)", fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "e2_reliability.png", dpi=150); fig.savefig(figs / "e2_reliability.pdf"); plt.close(fig)

    h1p = h1[(h1["label"] == lab0) & (h1["tre"] == "rho_lr")].iloc[0]
    passed = bool(h1p["diff"] >= cfg["tre"]["h1_margin"] and h1p["ci_lo"] > 0)
    lines = [
        "# E2 — transfer-failure detection (N1, H1)", "",
        f"Quick mode: {args.quick}. Runtime {time.time()-T0:.0f}s. Development period only; locked test untouched.", "",
        f"Episodes: {len(ep)} (asset x quarter), {ep['market'].nunique()} held-out markets, years {ep['year'].min()}-{ep['year'].max()}. "
        f"Scored out-of-sample: {R[R['label']==lab0]['episode_id'].nunique()} (time-ordered, market-fold held out).", "",
        f"## H1 (primary label `{lab0}`, primary TRE = logistic): **{'SUPPORTED' if passed else 'NOT SUPPORTED'}** on validation episodes", "",
        f"Criterion: AUROC(TRE) − AUROC(best single signal) ≥ {cfg['tre']['h1_margin']} and quarter-block bootstrap 95% CI > 0.", "",
        md(h1), "",
        "## Detection scores (all scorers, all labels)", "",
        md(scores.sort_values(["label", "auroc"], ascending=[True, False])), "",
        "## Per-year AUROC (primary label)", "", md(py), "",
        "## Episode summary by year (coverage of 90% intervals: zero-shot vs in-domain)", "", md(summ), "",
        "## Logistic TRE coefficients (descriptive, fitted on all episodes; standardised units)", "", md(coefs), "",
        "## Reliability (primary label)", "", md(rel), "",
        "Figures: e2_roc, e2_reliability.",
    ]
    (d / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"H1 ({lab0}, TRE-LR): diff={h1p['diff']:.3f} CI=[{h1p['ci_lo']:.3f}, {h1p['ci_hi']:.3f}] -> {'SUPPORTED' if passed else 'NOT SUPPORTED'}")
    print(f"E2 done in {time.time()-T0:.0f}s", flush=True)
    sys.stdout = tee.stdout; tee.close()
    print(f"Handoff: {make_zip(d, STEP)}")


if __name__ == "__main__":
    main()
