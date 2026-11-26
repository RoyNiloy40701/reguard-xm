"""E3 sensitivity checks — EXPLORATORY (not pre-registered; designed after seeing the E3 results).

Checks
  S1 entry timing     : entries at every quarter start (Q1..Q4), not only on the first trading day of the year
                        (tests the year-end low-volatility artefact seen in E3)
  S2 volatility scale : sigma_hat = vol_60d instead of vol_20d (Q1 entries)
  S3 hyper-parameters : ACI step gamma, pool pseudo-count n0, number of strata (Q1 entries)
  S4 nominal level    : alpha = 0.10 and 0.01 (Q1 entries)
Usage (from the code folder):  python scripts/run_e3_sensitivity.py   [--quick]
Writes results/e3_sensitivity/{report.md, s1_timing.csv, s2_vol.csv, s3_hyper.csv, s4_alpha.csv, log.txt,
env_info.txt, figures/} and results/e3_sensitivity/handoff_e3_sensitivity.zip.
"""

from __future__ import annotations

import argparse
import copy
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

from reguard.conformal import core as C  # noqa: E402
from reguard.conformal.cold_start import EntryContext, assign, build_scores, kernel_weights  # noqa: E402
from reguard.data.store import load_panel, read_table  # noqa: E402
from reguard.utils.handoff import CODE_ROOT, PROJECT_ROOT, Tee, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "e3_sensitivity"
STREAM_LEN = 230          # ~ one year of available score days after entry


def md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def entries(ep: pd.DataFrame, sc: pd.DataFrame, quarters: set[int]):
    """Entries at the given quarter starts, in date order: (tau, asset, market, target-episode, stream)."""
    first_tau = ep.groupby(["asset", ep["tau"].dt.year.rename("yr"), ep["tau"].dt.quarter.rename("qtr")])["tau"].min().reset_index(name="t0")
    first_tau = first_tau.sort_values("t0")
    by_asset = {a: g.sort_values("date") for a, g in sc.groupby("asset")}
    ep_idx = ep.set_index(["asset", "tau"])
    for r in first_tau.itertuples(index=False):
        if r[2] not in quarters:
            continue
        asset, tau = r[0], r[3]
        g = by_asset.get(asset)
        if g is None or (asset, tau) not in ep_idx.index:
            continue
        te = ep_idx.loc[(asset, tau)]
        te = te.iloc[0] if isinstance(te, pd.DataFrame) else te
        te = te.copy(); te["asset"] = asset
        stream = g[g["date"] >= tau].head(STREAM_LEN)
        if len(stream) < 60:
            continue
        yield tau, asset, te["market"], te, stream


def run_methods(s, sig, ctx, tgt, cfg, which):
    a, cc, b = cfg["alpha"], cfg["cscrb"], cfg["baselines"]
    qs = {}
    if "fixed" in which:
        from scipy import stats
        qs["fixed"] = C.run_constant(s, float(stats.norm.ppf(1 - a)))
    if "aci_target" in which:
        qs["aci_target"] = C.run_aci_target(s, a, b["warmup"], cc["gamma"])
    if "time_weighted" in which:
        from scipy import stats
        qs["time_weighted"] = C.run_time_weighted(s, a, b["warmup"], float(stats.norm.ppf(1 - a)), b["time_halflife"])
    if "source_indomain_split" in which:
        qs["source_indomain_split"] = C.run_constant(s, ctx.pool(col="s_in").quantile(1 - a))
    if "pool_aci" in which:
        qs["pool_aci"] = C.run_pool_aci(s, ctx.pool(), a, cc["gamma"], cc["pseudo_count"])
    if "cscrb" in which:
        stratum = None
        if ctx.stratifier is not None and np.isfinite(float(tgt.get(ctx.stratifier, np.nan))):
            stratum = assign(float(tgt[ctx.stratifier]), ctx.edges)
        if stratum is not None:
            w = kernel_weights(ctx.ep, tgt, cc["kernel_signals"]) if cc["kernel_weights"] else None
            pool = ctx.pool(stratum=stratum, weights=w)
        else:
            pool = ctx.pool()
        qs["cscrb"] = C.run_pool_aci(s, pool, a, cc["gamma"], cc["pseudo_count"])
    return qs


def evaluate(ep, sc, variants, quarters, which, min_pool, log):
    """variants: list of (label, cfg) that share alpha and strata (so they share one EntryContext per entry)."""
    rows, cur_tau, ep_t, sc_t = [], None, None, None
    t0 = time.time()
    base_cfg = variants[0][1]
    for tau, asset, market, tgt, stream in entries(ep, sc, quarters):
        if tau != cur_tau:
            cur_tau = tau
            ep_t = ep[ep["window_end"] < tau]
            sc_t = sc[sc["episode_id"].isin(ep_t["episode_id"])]
        if len(ep_t) < min_pool:
            continue
        ctx = EntryContext(tau.year, market, ep_t, sc_t, base_cfg)
        s, sig = stream["s_zs"].to_numpy(), stream["sigma"].to_numpy()
        for label, cfg in variants:
            for m, q in run_methods(s, sig, ctx, tgt, cfg, which).items():
                met = C.stream_metrics(s, q, sig, cfg["alpha"], [20, 60, 120])
                rows.append({"variant": label, "quarter": tau.quarter, "tau": tau, "asset": asset, "market": market,
                             "method": m, **met})
    log(f"  {[v[0] for v in variants]}: {len({(r['tau'], r['asset']) for r in rows})} entries ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def pooled(E: pd.DataFrame, alpha: float, by: list[str]) -> pd.DataFrame:
    out = []
    for k, g in E.groupby(by + ["method"]):
        row = dict(zip(by + ["method"], k if isinstance(k, tuple) else (k,)))
        row["entries"] = len(g)
        for h in ("20", "60", "120", "full"):
            w = g["n"].clip(upper=int(h)) if h != "full" else g["n"]
            bef = float(np.sum(g[f"bef_{h}"] * w) / np.sum(w))
            row[f"bef_{h}"], row[f"gap_{h}"] = bef, abs(bef - alpha)
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    d = step_dir(STEP)
    tee = Tee(d / "log.txt"); sys.stdout = tee
    with open(CODE_ROOT / "configs" / "experiment" / "e3.yaml", encoding="utf-8") as f:
        cfg0 = yaml.safe_load(f)
    write_env_info(d, seed=cfg0["seed"])
    T0 = time.time()
    cache = PROJECT_ROOT / "data" / "interim" / ("e2_quick" if args.quick else "e2")
    min_pool = 40 if args.quick else cfg0["entry"]["min_pool_episodes"]
    ep = read_table(cache, "episodes")
    ep["tau"] = pd.to_datetime(ep["tau"]); ep["window_end"] = pd.to_datetime(ep["window_end"])
    rows = read_table(cache, "episode_rows")
    panel = load_panel("tierb_etf")
    sc20 = build_scores(rows, panel, cfg0["sigma_floor"], "vol_20d")
    log = lambda s: print(s, flush=True)  # noqa: E731
    log("E3 sensitivity (EXPLORATORY) start")
    core = ["fixed", "aci_target", "time_weighted", "source_indomain_split", "pool_aci", "cscrb"]

    # S1 entry timing (its Q1 entries are also the default for S2 and S3)
    S1 = evaluate(ep, sc20, [("timing", cfg0)], {1, 2, 3, 4}, core, min_pool, log)
    t1 = pooled(S1, cfg0["alpha"], ["quarter"]); t1.to_csv(d / "s1_timing.csv", index=False)
    q1 = S1[S1["quarter"] == 1]
    # S2 volatility scale (Q1 entries)
    sc60 = build_scores(rows, panel, cfg0["sigma_floor"], "vol_60d")
    S2 = pd.concat([q1.assign(variant="vol_20d"), evaluate(ep, sc60, [("vol_60d", cfg0)], {1}, core, min_pool, log)])
    t2 = pooled(S2, cfg0["alpha"], ["variant"]); t2.to_csv(d / "s2_vol.csv", index=False)
    # S3 hyper-parameters (pool-based methods; Q1 entries)
    shared = []
    for name, v in [("gamma", 0.002), ("gamma", 0.01), ("gamma", 0.02), ("pseudo_count", 20), ("pseudo_count", 120)]:
        c = copy.deepcopy(cfg0); c["cscrb"][name] = v
        shared.append((f"{name}={v}", c))
    S3 = [evaluate(ep, sc20, shared, {1}, ["pool_aci", "cscrb"], min_pool, log)]
    for v in (2, 4):
        c = copy.deepcopy(cfg0); c["cscrb"]["strata"] = v
        S3.append(evaluate(ep, sc20, [(f"strata={v}", c)], {1}, ["cscrb"], min_pool, log))
    S3.append(q1[q1["method"].isin(["pool_aci", "cscrb"])].assign(variant="default"))
    t3 = pooled(pd.concat(S3), cfg0["alpha"], ["variant"]); t3.to_csv(d / "s3_hyper.csv", index=False)
    # S4 nominal level
    S4 = []
    for al in (0.10, 0.01):
        c = copy.deepcopy(cfg0); c["alpha"] = al
        S4.append(pooled(evaluate(ep, sc20, [(f"alpha={al}", c)], {1}, core, min_pool, log), al, ["variant"]))
    t4 = pd.concat(S4); t4.to_csv(d / "s4_alpha.csv", index=False)

    # figure: 60-day gap by entry quarter
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, g in t1.groupby("method"):
        ax.plot(g["quarter"], g["gap_20"], "o-", label=m, lw=2 if m == "cscrb" else 1)
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("entry quarter"); ax.set_ylabel("pooled 20-day gap |BEF - alpha|")
    ax.set_title("EXPLORATORY: entry-timing sensitivity", fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(d / "figures" / "s1_timing.png", dpi=150); plt.close(fig)

    cols = ["entries", "bef_20", "gap_20", "bef_60", "gap_60", "bef_full", "gap_full"]
    lines = ["# E3 sensitivity checks — EXPLORATORY (not pre-registered)", "",
             f"Quick: {args.quick}. Runtime {time.time()-T0:.0f}s. Designed after seeing E3; report as exploratory.", "",
             "## S1 Entry timing (entries at each quarter start)", "", md(t1[["quarter", "method"] + cols]), "",
             "## S2 Volatility scale (Q1 entries): vol_20d vs vol_60d", "", md(t2[["variant", "method"] + cols]), "",
             "## S3 Hyper-parameters (Q1 entries; pool-based methods)", "", md(t3[["variant", "method"] + cols]), "",
             "## S4 Nominal level (Q1 entries)", "", md(t4[["variant", "method"] + cols]), ""]
    (d / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"done in {time.time()-T0:.0f}s")
    sys.stdout = tee.stdout; tee.close()
    print(f"Handoff: {make_zip(d, STEP)}")


if __name__ == "__main__":
    main()
