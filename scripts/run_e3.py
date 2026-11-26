"""E3: cold-start conformal risk budget (N2) vs own-history and source baselines; T1 diagnostic (H2-H4).

Needs the E2 episode pool (data/interim/e2/). Development data only; the locked test stays untouched.
Usage (from the code folder):
    python scripts/run_e3.py            # full run on the E2 pool (minutes)
    python scripts/run_e3.py --quick    # uses data/interim/e2_quick (pipeline check)
Writes results/e3/{report.md, summary.csv, entries.csv, per_market.csv, hypotheses.csv, stratifiers.csv,
curves.csv, figures/, log.txt, env_info.txt} and results/e3/handoff_e3.zip.
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

from reguard.conformal import core as C  # noqa: E402
from reguard.conformal.cold_start import EntryContext, assign, build_scores, kernel_weights  # noqa: E402
from reguard.data.store import load_panel, read_table  # noqa: E402
from reguard.utils.handoff import CODE_ROOT, PROJECT_ROOT, Tee, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "e3"
BASELINES = ["fixed", "heuristic", "source_indomain_split", "time_weighted", "regime_weighted", "tcp", "aci_target", "dtaci"]
POOLED = ["pool_static", "pool_aci", "pool_strata_aci", "cscrb", "cscrb_static"]


def md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def run_entry(s, sig, ctx: EntryContext, tgt_ep: pd.Series, cfg: dict) -> tuple[dict, dict]:
    a, b, cc = cfg["alpha"], cfg["baselines"], cfg["cscrb"]
    z = b["fixed_z"]
    qs, info = {}, {"stratifier": ctx.stratifier, "stratum": None, "gap_est": np.nan, "strat_value": np.nan}
    qs["fixed"] = C.run_constant(s, z)
    # stratum of the target at entry (label-free signal value at tau)
    stratum, val = None, np.nan
    if ctx.stratifier is not None:
        val = float(tgt_ep.get(ctx.stratifier, np.nan))
        if np.isfinite(val):
            stratum = assign(val, ctx.edges)
    info.update(stratum=stratum, strat_value=val)
    if ctx.stratifier is not None and np.isfinite(val):
        pv = ctx.ep[ctx.stratifier].astype(float)
        scale = (pv.quantile(0.75) - pv.quantile(0.25)) or 1.0
        qs["heuristic"] = C.run_constant(s, z * np.exp(b["heuristic_eta"] * (val - pv.median()) / scale))
    else:
        qs["heuristic"] = C.run_constant(s, z)
    qs["source_indomain_split"] = C.run_constant(s, ctx.pool(col="s_in").quantile(1 - a))
    qs["time_weighted"] = C.run_time_weighted(s, a, b["warmup"], z, b["time_halflife"])
    qs["regime_weighted"] = C.run_regime_weighted(s, sig, a, b["warmup"], z, b["regime_bandwidth"])
    qs["tcp"] = C.run_tcp(s, a, z, b["tcp_eta"])
    qs["aci_target"] = C.run_aci_target(s, a, b["warmup"], cc["gamma"])
    qs["dtaci"] = C.run_dtaci(s, a, b["warmup"], b["dtaci_gammas"], b["dtaci_eta"])
    p_all = ctx.pool()
    qs["pool_static"] = C.run_pool_aci(s, p_all, a, cc["gamma"], cc["pseudo_count"], online=False)
    qs["pool_aci"] = C.run_pool_aci(s, p_all, a, cc["gamma"], cc["pseudo_count"])
    if stratum is not None:
        p_b = ctx.pool(stratum=stratum)
        w = kernel_weights(ctx.ep, tgt_ep, cc["kernel_signals"]) if cc["kernel_weights"] else None
        p_bw = ctx.pool(stratum=stratum, weights=w)
        info["gap_est"] = ctx.gap(stratum)
    else:
        p_b = p_bw = p_all
    qs["pool_strata_aci"] = C.run_pool_aci(s, p_b, a, cc["gamma"], cc["pseudo_count"])
    qs["cscrb"] = C.run_pool_aci(s, p_bw, a, cc["gamma"], cc["pseudo_count"])
    qs["cscrb_static"] = C.run_pool_aci(s, p_bw, a, cc["gamma"], cc["pseudo_count"], online=False)
    return qs, info


def bootstrap_gap(E: pd.DataFrame, col: str, alpha: float, n_boot=1000, seed=0):
    """Pooled BEF and its gap with a 95% CI resampling entry YEARS (entries in a year are dependent)."""
    years = E["year"].unique()
    rng = np.random.default_rng(seed)
    by = {y: E.loc[E["year"] == y, [col, "w"]].to_numpy() for y in years}
    vals = []
    for _ in range(n_boot):
        X = np.vstack([by[y] for y in rng.choice(years, len(years), replace=True)])
        vals.append(np.sum(X[:, 0] * X[:, 1]) / np.sum(X[:, 1]))
    vals = np.abs(np.array(vals) - alpha)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    d = step_dir(STEP)
    tee = Tee(d / "log.txt"); sys.stdout = tee
    with open(CODE_ROOT / "configs" / "experiment" / "e3.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    write_env_info(d, seed=cfg["seed"])
    T0 = time.time()
    cache = PROJECT_ROOT / "data" / "interim" / ("e2_quick" if args.quick else "e2")
    if args.quick:
        cfg["entry"]["min_pool_episodes"] = 40
    ep = read_table(cache, "episodes")
    rows = read_table(cache, "episode_rows")
    panel = load_panel("tierb_etf")
    sc = build_scores(rows, panel, cfg["sigma_floor"])
    ep["tau"] = pd.to_datetime(ep["tau"]); ep["window_end"] = pd.to_datetime(ep["window_end"])
    print(f"E3 start: {len(ep)} episodes, {len(sc)} daily scores ({time.time()-T0:.0f}s)", flush=True)
    a, H = cfg["alpha"], cfg["entry"]["horizons"]
    methods = BASELINES + POOLED
    ent_rows, err_store, strat_rows = [], {m: [] for m in methods}, []
    for Y in sorted(sc["year"].unique()):
        cut = pd.Timestamp(f"{Y}-01-01")
        ep_y = ep[ep["window_end"] < cut]
        if len(ep_y) < cfg["entry"]["min_pool_episodes"]:
            continue
        sc_y = sc[sc["episode_id"].isin(ep_y["episode_id"])]
        t_year, n_ent = time.time(), 0
        for asset, g in sc[sc["year"] == Y].groupby("asset"):
            g = g.sort_values("date")
            j = g["market"].iloc[0]
            te = ep[(ep["asset"] == asset) & (ep["tau"].dt.year == Y)].sort_values("tau")
            if te.empty or len(g) < 60:
                continue
            ctx = EntryContext(Y, j, ep_y, sc_y, cfg)
            s, sig = g["s_zs"].to_numpy(), g["sigma"].to_numpy()
            qs, info = run_entry(s, sig, ctx, te.iloc[0], cfg)
            for m in methods:
                met = C.stream_metrics(s, qs[m], sig, a, H)
                ent_rows.append({"year": Y, "asset": asset, "market": j, "method": m, **met, **info})
                err_store[m].append((s > qs[m]).astype(float))
            strat_rows.append({"year": Y, "market": j, "stratifier": ctx.stratifier, **{f"rho_{k}": v for k, v in ctx.strat_rhos.items()}})
            n_ent += 1
        print(f"  {Y}: {n_ent} entries, pool {len(ep_y)} episodes ({time.time()-t_year:.1f}s)", flush=True)
    E = pd.DataFrame(ent_rows)
    E.to_csv(d / "entries.csv", index=False)
    strat = pd.DataFrame(strat_rows).drop_duplicates(["year", "market"])
    strat.to_csv(d / "stratifiers.csv", index=False)

    # ---- pooled summary with year-block bootstrap CIs
    summ = []
    for m in methods:
        e = E[E["method"] == m]
        row = {"method": m, "entries": len(e)}
        for h in [str(x) for x in H] + ["full"]:
            w = e["n"].clip(upper=int(h)) if h != "full" else e["n"]
            bef = float(np.sum(e[f"bef_{h}"] * w) / np.sum(w))
            lo, hi = bootstrap_gap(e.assign(w=w), f"bef_{h}", a)
            row.update({f"bef_{h}": bef, f"gap_{h}": abs(bef - a), f"gap_{h}_ci": f"[{lo:.4f}, {hi:.4f}]"})
        row.update(severity_sigma=e["severity_sigma"].mean(), mean_q=e["mean_q"].mean(), abstain=e["abstain_share"].mean())
        summ.append(row)
    S = pd.DataFrame(summ)
    S.to_csv(d / "summary.csv", index=False)

    # ---- curves: pooled exceedance rate by day since entry
    L = max(len(x) for x in err_store["fixed"])
    curves = {}
    for m in methods:
        M = np.full((len(err_store[m]), L), np.nan)
        for i, x in enumerate(err_store[m]):
            M[i, :len(x)] = x
        curves[m] = np.nanmean(M, axis=0)
    cv = pd.DataFrame(curves); cv.index.name = "day"; cv.to_csv(d / "curves.csv")

    # ---- hypotheses (development check)
    w60 = cfg["h2"]["window"]
    pm = E.assign(w=E["n"].clip(upper=w60)).groupby(["market", "method"]).apply(
        lambda x: abs(np.sum(x[f"bef_{w60}"] * x["w"]) / np.sum(x["w"]) - a), include_groups=False).unstack()
    best_base = S[S["method"].isin(BASELINES)].sort_values(f"gap_{w60}")["method"].iloc[0]
    pm["best_baseline"] = pm[best_base]
    pm["cscrb_halves_gap"] = pm["cscrb"] <= cfg["h2"]["reduction"] * pm["best_baseline"]
    pm.to_csv(d / "per_market.csv")
    share = float(pm["cscrb_halves_gap"].mean())
    full_gap = float(S.loc[S["method"] == "cscrb", "gap_full"].iloc[0])
    st = E[(E["method"] == "cscrb_static")].dropna(subset=["gap_est"])
    h4 = stats.spearmanr(st["gap_est"], st[f"bef_{w60}"] - a) if len(st) > 10 else None
    hyp = pd.DataFrame([
        {"hypothesis": "H2", "metric": f"share of markets where CSCRB {w60}-day gap <= {cfg['h2']['reduction']} x best baseline ({best_base})",
         "value": share, "criterion": f">= {cfg['h2']['share_markets']:.3f}", "supported_dev": share >= cfg["h2"]["share_markets"]},
        {"hypothesis": "H3", "metric": "CSCRB full-year pooled BEF gap", "value": full_gap,
         "criterion": f"<= {cfg['h3']['max_gap']}", "supported_dev": full_gap <= cfg["h3"]["max_gap"]},
        {"hypothesis": "H4", "metric": f"Spearman(estimated stratum gap, realised {w60}-day excess) [static q0]",
         "value": float(h4.statistic) if h4 else np.nan, "criterion": "> 0 and p < 0.05",
         "supported_dev": bool(h4 and h4.statistic > 0 and h4.pvalue < 0.05)},
    ])
    if h4:
        hyp.loc[2, "metric"] += f" p={h4.pvalue:.4f}, n={len(st)}"
    hyp.to_csv(d / "hypotheses.csv", index=False)

    # ---- does stratification separate risk? exceedance of the unstratified static q0 by target stratum
    ps = E[E["method"] == "pool_static"].dropna(subset=["stratum"]).groupby("stratum").agg(
        entries=("asset", "size"), bef_60=(f"bef_{w60}", "mean"), bef_full=("bef_full", "mean")).reset_index()
    ps["stratum"] = ps["stratum"].astype(int)

    # ---- figures
    figs = d / "figures"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    show = ["fixed", "source_indomain_split", "time_weighted", "aci_target", "dtaci", "pool_aci", "cscrb"]
    for m in show:
        y = pd.Series(curves[m]).rolling(10, min_periods=1).mean()
        ax.plot(y.index[:min(L, 230)], y.values[:min(L, 230)], lw=2 if m == "cscrb" else 1, label=m)
    ax.axhline(a, color="k", ls=":", lw=1)
    ax.set_xlabel("trading days since entry into the held-out market"); ax.set_ylabel("budget-exceedance rate (10-day mean)")
    ax.set_title("Cold start: exceedance rate after entering an unseen market (pooled over entries)", fontsize=9)
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(figs / "e3_entry_curves.png", dpi=150); fig.savefig(figs / "e3_entry_curves.pdf"); plt.close(fig)
    if len(st) > 10:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.scatter(st["gap_est"], st[f"bef_{w60}"] - a, s=8, alpha=0.5)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlabel("estimated stratum mismatch δ̂ (KS)"); ax.set_ylabel(f"realised {w60}-day excess exceedance")
        ax.set_title("T1 diagnostic", fontsize=9); fig.tight_layout()
        fig.savefig(figs / "e3_t1_diagnostic.png", dpi=150); fig.savefig(figs / "e3_t1_diagnostic.pdf"); plt.close(fig)

    show_cols = ["method", "entries", "bef_20", "gap_20", "bef_60", "gap_60", "gap_60_ci", "bef_120", "gap_120",
                 "bef_full", "gap_full", "gap_full_ci", "severity_sigma", "mean_q", "abstain"]
    lines = [
        "# E3 — cold-start conformal risk budget (N2) and T1 diagnostic", "",
        f"Quick mode: {args.quick}. Runtime {time.time()-T0:.0f}s. Development period only; locked test untouched.", "",
        f"Entries: {E[['year','asset']].drop_duplicates().shape[0]} (held-out asset x year, zero target history at entry), "
        f"years {E['year'].min()}-{E['year'].max()}, {E['market'].nunique()} markets. alpha = {a}.", "",
        "## Pooled budget-exceedance frequency (BEF) and gap |BEF - alpha| (95% CI: bootstrap over entry years)", "",
        md(S[show_cols]), "",
        "## Hypotheses (development check; final verdict only on the locked test)", "", md(hyp), "",
        "## Does the stratifier separate risk? (exceedance of the UNSTRATIFIED static pool threshold, by target stratum)", "",
        md(ps), "",
        "## Stratifier chosen by the nested rule (per entry year; share of entries)", "",
        md(strat["stratifier"].fillna("none").value_counts(normalize=True).rename("share").reset_index()), "",
        "## Per-market 60-day gap (CSCRB vs best baseline)", "",
        md(pm[["cscrb", "best_baseline", "cscrb_halves_gap"]].reset_index()), "",
        "Figures: e3_entry_curves (headline), e3_t1_diagnostic.",
    ]
    (d / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"E3 done in {time.time()-T0:.0f}s", flush=True)
    sys.stdout = tee.stdout; tee.close()
    print(f"Handoff: {make_zip(d, STEP)}")


if __name__ == "__main__":
    main()
