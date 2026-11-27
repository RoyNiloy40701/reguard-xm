"""Phase 9: the ONE-TIME confirmatory analysis of the locked test period, exactly as in docs/preregistration.md
(frozen analysis plan v1.0, SHA-256 a4fd6922...e184).

REAL TEST (run once, in this order; each stage logs its unlock in docs/test_access_log.csv):
    set REGUARD_UNLOCK_TEST=1
    python scripts/run_final_test.py e2        # test-period episodes + H1          (~30-60 min)
    python scripts/run_final_test.py e3        # cold-start budgets: H2a, H2b, H3, H2-orig, H4
    python scripts/run_final_test.py e4        # governance: H5, H6, H7             (~10 min)
    python scripts/run_final_test.py e8        # drivers: H8                        (~15 min)
    python scripts/run_final_test.py e1        # descriptive forecasting tables     (~20-40 min)
    python scripts/run_final_test.py summary   # all hypotheses + Holm correction -> results/test_summary/handoff zip
PSEUDO-TEST (checks the code on development data only; never touches the test):
    python scripts/run_final_test.py <stage> --pseudo [--quick]
Outputs: results/test_<stage>/ (real) or results/pseudo_<stage>/ (pseudo), each with a handoff zip.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

from reguard.data import store  # noqa: E402
from reguard.data.splits import annual_refit_plan, make_windows, test_windows  # noqa: E402
from reguard.utils.handoff import CODE_ROOT, PROJECT_ROOT, RESULTS_ROOT, Tee, make_zip, step_dir, write_env_info  # noqa: E402

PLAN_HASH = "a4fd692275cf481af1bf30b45714c14fc6cd6418c8a6497d4c86166ea757e184"
REASON = f"Phase 9 confirmatory test, frozen analysis plan v1.0 sha256={PLAN_HASH[:12]}"
REAL = {"start": "2022-01-01", "end": "2026-11-24", "entry_last": "2025-12-31"}
TEST_STRESS = [["2022-01-03", "2022-10-14"], ["2023-03-08", "2023-03-31"], ["2024-07-24", "2024-08-16"], ["2025-04-02", "2025-04-30"]]
DEV_TOP3_DEFAULT = None
INTERIM = PROJECT_ROOT / "data" / "interim"
WCOLS = ["train_start", "train_end", "embargo_start", "eval_start", "eval_end"]


def cfg_yaml(name):
    with open(CODE_ROOT / "configs" / "experiment" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


class Ctx:
    def __init__(self, args):
        self.pseudo, self.quick = args.pseudo, args.quick
        if args.pseudo:
            self.start = args.start or ("2015-01-01" if args.quick else "2021-01-01")
            self.end = args.end or ("2015-12-31" if args.quick else "2021-12-31")
            self.entry_last = self.end
            self.stress = [["2020-02-19", "2020-04-30"]]
        else:
            self.start, self.end, self.entry_last, self.stress = REAL["start"], REAL["end"], REAL["entry_last"], TEST_STRESS
        self.prefix = "pseudo_" if self.pseudo else "test_"
        self.dev_cache = INTERIM / ("e2_quick" if self.quick else "e2")
        self.test_cache = INTERIM / f"{self.prefix}e2"

    def panel(self, name):
        return store.full_panel(name, None if self.pseudo else REASON, pseudo_end=self.end if self.pseudo else None)


def boot_years(df: pd.DataFrame, fn, n_boot=1000, seed=0):
    """Resample entry years (blocks) with replacement; returns array of fn(resampled df)."""
    yrs = df["year"].unique()
    by = {y: df[df["year"] == y] for y in yrs}
    rng = np.random.default_rng(seed)
    return np.array([fn(pd.concat([by[y] for y in rng.choice(yrs, len(yrs))])) for _ in range(n_boot)])


# ---------------------------------------------------------------------------
def stage_e2(c: Ctx, log, d):
    from reguard.episodes.pseudo_transfer import generate
    from reguard.tre import estimator as T
    cfg = cfg_yaml("e2")
    pcfg = store.load_yaml("processing.yaml")
    df = c.panel(cfg["panel"])
    cal = pd.DatetimeIndex(sorted(df["date"].unique()))
    w = test_windows(cal, c.start, c.end, pcfg["embargo_days"], 504)
    plan = annual_refit_plan(w, cal, cfg["episodes"]["calibration_days"], pcfg["embargo_days"])
    markets = ["US", "JP", "GB", "DE", "FR", "CA", "AU", "BR", "MX", "KR", "TW", "ZA"] if c.quick else None
    log(f"  test episodes: years {[p['year'] for p in plan]}")
    ep, rows = generate(df, plan, cfg, log=log, markets=markets)
    store.write_table(ep, c.test_cache, "episodes_raw"); store.write_table(rows, c.test_cache, "episode_rows")
    dev = store.read_table(c.dev_cache, "episodes_raw")
    dev = dev[pd.to_datetime(dev["window_end"]) < pd.Timestamp(c.start)]
    orient = cfg["tre"]["signals"]; sig = list(orient)
    labels = [cfg["labels"]["primary"]] + cfg["labels"]["secondary"]
    recs = []
    for lab in labels:
        tr, te = dev[dev[lab].notna()], ep[ep[lab].notna()]
        g_star = float(np.quantile(tr[lab], cfg["labels"]["failure_quantile"]))
        ytr = (tr[lab] > g_star).to_numpy(int)
        r = te[["episode_id", "asset", "market", "year", "tau", lab]].rename(columns={lab: "g"})
        r["label"], r["g_star"], r["y_fail"], r["fold"] = lab, g_star, (te[lab] > g_star).to_numpy(int), 0
        for kind in ("lr", "gbm"):
            r[f"rho_{kind}"] = T.predict_tre(T.fit_tre(tr[sig], ytr, tr["market"].to_numpy(), kind, orient, cfg["tre"], cfg["seed"]), te[sig])
        recs.append(pd.concat([r, T.oriented(te, orient).add_prefix("sig_")], axis=1))
    R = pd.concat(recs, ignore_index=True)
    r0 = R[R["label"] == labels[0]][["episode_id", "rho_lr", "rho_gbm"]]
    store.write_table(ep.merge(r0, on="episode_id", how="left"), c.test_cache, "episodes")
    scores = T.score_table(R, orient)
    h1 = pd.DataFrame([T.h1_test(R, lab, f"rho_{k}", orient, 1000, cfg["seed"]) for lab in labels for k in ("lr", "gbm")])
    # one-sided bootstrap p for the primary H1 (P[diff <= 0]) by quarter blocks
    g = R[R["label"] == labels[0]].reset_index(drop=True)
    best = h1[(h1["label"] == labels[0]) & (h1["tre"] == "rho_lr")]["best_single"].iloc[0]
    q = g["tau"].to_numpy(); uq = np.unique(q); ix = {u: np.flatnonzero(q == u) for u in uq}
    rng = np.random.default_rng(0); y = g["y_fail"].to_numpy()
    diffs = []
    for _ in range(1000):
        i = np.concatenate([ix[u] for u in rng.choice(uq, len(uq))])
        a, b = T._auc(y[i], g["rho_lr"].to_numpy(float)[i]), T._auc(y[i], g[f"sig_{best}"].to_numpy(float)[i])
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(a - b)
    p1 = float(np.mean(np.array(diffs) <= 0))
    row = h1[(h1["label"] == labels[0]) & (h1["tre"] == "rho_lr")].iloc[0]
    hyp = pd.DataFrame([{"hypothesis": "H1", "estimate": row["diff"], "ci_lo": row["ci_lo"], "ci_hi": row["ci_hi"], "p_one_sided": p1,
                         "rule_met": bool(row["diff"] >= cfg["tre"]["h1_margin"] and row["ci_lo"] > 0),
                         "detail": f"TRE-LR {row['auroc_tre']:.3f} vs {best} {row['auroc_best_single']:.3f}; {len(g)} test episodes"}])
    scores.to_csv(d / "scores.csv", index=False); h1.to_csv(d / "h1.csv", index=False); hyp.to_csv(d / "hypotheses.csv", index=False)
    return ["## H1 (test)", "", md(hyp), "", "## H1 detail", "", md(h1), "", "## Scores", "",
            md(scores.sort_values(["label", "auroc"], ascending=[True, False]))]


def stage_e3(c: Ctx, log, d):
    import run_e3
    from run_e3_sensitivity import entries
    from reguard.conformal import core as C
    from reguard.conformal.cold_start import EntryContext, build_scores
    cfg0 = cfg_yaml("e3")
    dev_ep, dev_rows = store.read_table(c.dev_cache, "episodes"), store.read_table(c.dev_cache, "episode_rows")
    te_ep, te_rows = store.read_table(c.test_cache, "episodes"), store.read_table(c.test_cache, "episode_rows")
    dev_ep = dev_ep[pd.to_datetime(dev_ep["window_end"]) < pd.Timestamp(c.start)]
    dev_rows = dev_rows[dev_rows["episode_id"].isin(dev_ep["episode_id"])]
    ep = pd.concat([dev_ep, te_ep], ignore_index=True)
    ep["tau"] = pd.to_datetime(ep["tau"]); ep["window_end"] = pd.to_datetime(ep["window_end"])
    rows = pd.concat([dev_rows, te_rows], ignore_index=True)
    panel = c.panel("tierb_etf")
    te_ids = set(te_ep["episode_id"])
    ep_entries = ep[ep["episode_id"].isin(te_ids) & (ep["tau"] <= pd.Timestamp(c.entry_last))]
    out, daily = [], []
    for variant, vol, alpha in (("primary", "vol_60d", 0.05), ("alpha_0.01", "vol_60d", 0.01), ("vol_20d", "vol_20d", 0.05)):
        cfg = copy.deepcopy(cfg0); cfg["alpha"] = alpha
        cfg["baselines"]["fixed_z"] = float(stats.norm.ppf(1 - alpha))
        sc = build_scores(rows, panel, cfg["sigma_floor"], vol)
        t0, n, cur, ep_t, sc_t = time.time(), 0, None, None, None
        for tau, asset, market, tgt, stream in entries(ep_entries, sc, {1, 2, 3, 4}):
            if tau != cur:
                cur = tau
                ep_t = ep[ep["window_end"] < tau]
                sc_t = sc[sc["episode_id"].isin(ep_t["episode_id"])]
            ctx = EntryContext(tau.year, market, ep_t, sc_t, cfg)
            s, sig = stream["s_zs"].to_numpy(), stream["sigma"].to_numpy()
            qs, info = run_e3.run_entry(s, sig, ctx, tgt, cfg)
            for m, qv in qs.items():
                out.append({"variant": variant, "alpha": alpha, "year": tau.year, "quarter": tau.quarter, "tau": tau, "asset": asset,
                            "market": market, "method": m, **C.stream_metrics(s, qv, sig, alpha, [20, 60, 120]), **info})
                if variant == "primary" and m in ("fixed", "aci_target", "pool_aci", "cscrb", "time_weighted"):
                    daily.append(pd.DataFrame({"date": stream["date"].to_numpy(), "method": m, "err": (s > qv).astype(float)}))
            n += 1
        log(f"  E3 {variant}: {n} entries ({time.time()-t0:.0f}s)")
    E = pd.DataFrame(out); E.to_csv(d / "entries.csv", index=False)

    def pooled_gap(e, h, a):
        w = e["n"].clip(upper=h) if h != "full" else e["n"]
        return abs(np.sum(e[f"bef_{h}"] * w) / np.sum(w) - a)

    P = E[E["variant"] == "primary"]; A1 = E[E["variant"] == "alpha_0.01"]
    summ = (E.groupby(["variant", "method"]).apply(lambda e: pd.Series({h: pooled_gap(e, h if h == "full" else int(h), e["alpha"].iloc[0])
                                                                         for h in ("20", "60", "120", "full")}), include_groups=False)
            .rename(columns=lambda c: f"gap_{c}").reset_index())
    summ.to_csv(d / "summary.csv", index=False)
    # H2a
    f2a = lambda e: pooled_gap(e[e["method"] == "pool_aci"], 20, 0.01) - pooled_gap(e[e["method"] == "aci_target"], 20, 0.01)
    est2a, b2a = f2a(A1), boot_years(A1, f2a)
    # H2b
    def worst(e, m):
        return max(pooled_gap(g, 20, 0.05) for _, g in e[e["method"] == m].groupby("quarter"))
    f2b = lambda e: worst(e, "pool_aci") - worst(e, "aci_target")
    est2b, b2b = f2b(P), boot_years(P, f2b)
    # H3, H2-orig, H4
    h3 = pooled_gap(P[P["method"] == "pool_aci"], "full", 0.05)
    base = ["fixed", "heuristic", "source_indomain_split", "time_weighted", "regime_weighted", "tcp", "aci_target", "dtaci"]
    bb = min(base, key=lambda m: pooled_gap(P[P["method"] == m], 60, 0.05))
    pm = P.groupby("market").apply(lambda e: pooled_gap(e[e["method"] == "cscrb"], 60, 0.05) <= 0.5 * pooled_gap(e[e["method"] == bb], 60, 0.05),
                                   include_groups=False)
    st = P[P["method"] == "cscrb_static"].dropna(subset=["gap_est"])
    h4 = stats.spearmanr(st["gap_est"], st["bef_60"] - 0.05) if len(st) > 10 else None
    hyp = pd.DataFrame([
        {"hypothesis": "H2a", "estimate": est2a, "ci_lo": float(np.quantile(b2a, .025)), "ci_hi": float(np.quantile(b2a, .975)),
         "p_one_sided": float(np.mean(b2a >= 0)), "rule_met": bool(est2a < 0 and np.quantile(b2a, .975) < 0),
         "detail": "alpha 0.01, 20-day pooled gap, pool_aci - aci_target"},
        {"hypothesis": "H2b", "estimate": est2b, "ci_lo": float(np.quantile(b2b, .025)), "ci_hi": float(np.quantile(b2b, .975)),
         "p_one_sided": float(np.mean(b2b >= 0)), "rule_met": bool(est2b < 0 and np.quantile(b2b, .975) < 0),
         "detail": "alpha 0.05, worst-entry-quarter 20-day gap, pool_aci - aci_target"},
        {"hypothesis": "H3", "estimate": h3, "ci_lo": np.nan, "ci_hi": np.nan, "p_one_sided": np.nan, "rule_met": bool(h3 <= 0.01),
         "detail": "pool_aci full-period pooled gap, alpha 0.05 (threshold rule)"},
        {"hypothesis": "H2-orig", "estimate": float(pm.mean()), "ci_lo": np.nan, "ci_hi": np.nan, "p_one_sided": np.nan,
         "rule_met": bool(pm.mean() >= 2 / 3), "detail": f"share of markets, best baseline {bb} (reported only)"},
        {"hypothesis": "H4", "estimate": float(h4.statistic) if h4 else np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
         "p_one_sided": float(h4.pvalue) if h4 else np.nan, "rule_met": bool(h4 and h4.statistic > 0 and h4.pvalue < 0.05),
         "detail": "Spearman(estimated gap, realised 60-day excess) (reported only)"}])
    hyp.to_csv(d / "hypotheses.csv", index=False)
    D = pd.concat(daily); D["date"] = pd.to_datetime(D["date"])
    sw = []
    for a_, b_ in c.stress:
        x = D[(D["date"] >= a_) & (D["date"] <= b_)]
        sw.append({"window": f"{a_}..{b_}", **x.groupby("method")["err"].mean().to_dict()})
    SW = pd.DataFrame(sw); SW.to_csv(d / "stress.csv", index=False)
    return ["## Hypotheses (test)", "", md(hyp), "", "## Pooled gaps", "", md(summ), "", "## Exceedance rate within stress windows (alpha 0.05)", "", md(SW)]


def stage_e4(c: Ctx, log, d):
    import run_e4
    cfg = cfg_yaml("e4")
    pcfg = store.load_yaml("processing.yaml")
    panel = c.panel(cfg["panel"])
    tmp = Path(tempfile.mkdtemp())
    cal = pd.DatetimeIndex(sorted(panel["date"].unique()))
    make_windows(cal, "2003-01-01", (pd.Timestamp(c.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                 pcfg["embargo_days"], 504).to_csv(tmp / "splits_tierb.csv", index=False)
    (tmp / "configs" / "experiment").mkdir(parents=True)
    tcfg = copy.deepcopy(cfg)
    tcfg["periods"]["eval"] = [c.start, (pd.Timestamp(c.end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")]
    tcfg["periods"]["pool"] = ["2008-01-01", (pd.Timestamp(c.start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")]
    tcfg["stress_windows"] = c.stress
    (tmp / "configs" / "experiment" / "e4.yaml").write_text(yaml.safe_dump(tcfg), encoding="utf-8")
    cache = INTERIM / f"{c.prefix}e4"
    if (INTERIM / "e4" / "ppo").exists() and not (cache / "ppo").exists():
        shutil.copytree(INTERIM / "e4" / "ppo", cache / "ppo")
    run_e4.load_panel = lambda name: panel
    run_e4.PROCESSED_ROOT, run_e4.CODE_ROOT, run_e4.CACHE, run_e4.STEP = tmp, tmp, cache, f"{c.prefix}e4"
    sys.argv = ["run_e4.py"]
    run_e4.main()
    sys.stdout = TEE_HOLDER[0]
    src = RESULTS_ROOT / f"{c.prefix}e4"
    h5, h6, h7 = pd.read_csv(src / "h5.csv"), pd.read_csv(src / "h6.csv"), pd.read_csv(src / "h7.csv")
    ser = pd.read_csv(src / "daily_series.csv")
    from reguard.portfolio import metrics as PM
    ps = []
    for nm, g in ser.groupby("proposer"):
        a = g[g["governed"]]["net"].to_numpy(); b = g[~g["governed"]]["net"].to_numpy()
        rng = np.random.default_rng(0)
        dd = [PM.cvar(a[i]) - PM.cvar(b[i]) for i in (PM.stationary_bootstrap_idx(len(a), 20, rng) for _ in range(1000))]
        ps.append(float(np.mean(np.array(dd) >= 0)))
    h5r = h5[~h5["proposer"].astype(str).str.startswith("ppo_seed")]
    hyp = pd.DataFrame([
        {"hypothesis": "H5", "estimate": float(h5r["delta_cvar95"].max()), "ci_lo": np.nan, "ci_hi": float(h5r["ci_hi"].max()),
         "p_one_sided": max(ps), "rule_met": bool((h5r["ci_hi"] < 0).all()), "detail": "intersection-union over proposers (max p)"},
        {"hypothesis": "H6", "estimate": float(h6[~h6["proposer"].astype(str).str.startswith("ppo_seed")]["sharpe_loss"].max()),
         "ci_lo": np.nan, "ci_hi": np.nan, "p_one_sided": np.nan,
         "rule_met": bool(h6[~h6["proposer"].astype(str).str.startswith("ppo_seed")]["supported"].all()), "detail": "threshold rule"},
        {"hypothesis": "H7", "estimate": float(h7.get("sharpe_diff_ann", pd.Series([np.nan])).iloc[0]), "ci_lo": np.nan, "ci_hi": np.nan,
         "p_one_sided": float(h7["p"].iloc[0] / 2) if "p" in h7 and h7.get("sharpe_diff_daily", pd.Series([0])).iloc[0] > 0
         else (1 - float(h7["p"].iloc[0]) / 2 if "p" in h7 else np.nan),
         "rule_met": bool(h7["supported"].iloc[0] is True or str(h7["supported"].iloc[0]) == "True") if "supported" in h7 else False,
         "detail": "governed PPO - governed MV (LW HAC)" if "p" in h7 else "PPO models missing: H7 not testable"}])
    hyp.to_csv(d / "hypotheses.csv", index=False)
    return ["## Hypotheses (test)", "", md(hyp), "", f"Full E4 report: results/{c.prefix}e4/report.md (and its handoff zip)."]


def stage_e8(c: Ctx, log, d):
    from reguard.tre import drivers as DR
    cfg = cfg_yaml("e8")
    ep = store.read_table(c.test_cache, "episodes_raw")
    D = DR.driver_table(ep, c.panel("tierb_etf"))
    D = D[D["g_cal"].notna()].reset_index(drop=True)
    imp, r2 = DR.grouped_importance(D, cfg["model"], cfg["seed"], cfg["n_folds"], cfg["n_perm"])
    nb = 30 if c.pseudo else cfg["n_boot"]
    cnt, _ = DR.bootstrap_top3(D, cfg["model"], nb, cfg["seed"], log)
    top3 = frozenset(imp.index[:3])
    share = cnt.get(top3, 0) / nb
    dev_imp = RESULTS_ROOT / "e8" / "importance.csv"
    dev_top3 = frozenset(pd.read_csv(dev_imp)["driver_group"].iloc[:3]) if dev_imp.exists() else frozenset()
    same = top3 == dev_top3
    hyp = pd.DataFrame([{"hypothesis": "H8", "estimate": share, "ci_lo": np.nan, "ci_hi": np.nan, "p_one_sided": np.nan,
                         "rule_met": bool(share >= cfg["h8_share"] and same),
                         "detail": f"test top3 {sorted(top3)}; dev top3 {sorted(dev_top3)}; same={same}; OOF R2={r2:.3f}"}])
    hyp.to_csv(d / "hypotheses.csv", index=False)
    imp.rename("mse_increase").to_csv(d / "importance.csv")
    return ["## H8 (test)", "", md(hyp), "", "## Importance (test)", "", md(imp.rename("mse_increase").reset_index())]


def stage_e1(c: Ctx, log, d):
    import run_e1
    pcfg = store.load_yaml("processing.yaml")
    tmp = Path(tempfile.mkdtemp())
    tb, ds = c.panel("tierb_etf"), c.panel("dse_equity")
    for nm, p in (("tierb", tb), ("dse", ds)):
        cal = pd.DatetimeIndex(sorted(p["date"].unique()))
        test_windows(cal, c.start, c.end, pcfg["embargo_days"], 504).to_csv(tmp / f"splits_{nm}.csv", index=False)
    run_e1.load_panel = lambda name: tb if name == "tierb_etf" else ds
    run_e1.PROCESSED_ROOT, run_e1.STEP = tmp, f"{c.prefix}e1"
    run_e1.PRED_DIR = INTERIM / f"{c.prefix}e1_preds"
    sys.argv = ["run_e1.py"] + (["--quick"] if c.pseudo else [])
    run_e1.main()
    sys.stdout = TEE_HOLDER[0]
    return [f"E1 is descriptive (no hypothesis). Full report: results/{c.prefix}e1/report.md (and its handoff zip)."]


def stage_summary(c: Ctx, log, d):
    rows = []
    for st in ("e2", "e3", "e4", "e8"):
        f = RESULTS_ROOT / f"{c.prefix}{st}_stage" / "hypotheses.csv"
        if f.exists():
            rows.append(pd.read_csv(f).assign(stage=st))
    H = pd.concat(rows, ignore_index=True)
    fam = H["hypothesis"].isin(["H1", "H2a", "H2b", "H5", "H7"]) & H["p_one_sided"].notna()
    p = H.loc[fam, "p_one_sided"].to_numpy()
    order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    H["p_holm"] = np.nan; H.loc[fam, "p_holm"] = adj
    H["verdict"] = np.where(H["hypothesis"].isin(["H2-orig", "H4"]), np.where(H["rule_met"], "met (reported only)", "not met (reported only)"),
                   np.where(fam, np.where(H["rule_met"] & (H["p_holm"] < 0.05), "SUPPORTED", "NOT SUPPORTED"),
                            np.where(H["rule_met"], "SUPPORTED", "NOT SUPPORTED")))
    H.to_csv(d / "final_hypotheses.csv", index=False)
    return ["## Final hypothesis table" + (" (PSEUDO-TEST on development data — NOT the result)" if c.pseudo else ""), "",
            md(H[["hypothesis", "stage", "estimate", "ci_lo", "ci_hi", "p_one_sided", "p_holm", "rule_met", "verdict", "detail"]]), "",
            "Holm correction is applied to the hypotheses tested with a p-value (H1, H2a, H2b, H5, H7); H3, H6 and H8 are "
            "threshold rules judged as pre-specified; H2-orig and H4 are reported only."]


TEE_HOLDER = [None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["e1", "e2", "e3", "e4", "e8", "summary"])
    ap.add_argument("--pseudo", action="store_true", help="check the code on development data only")
    ap.add_argument("--quick", action="store_true", help="(pseudo) use the quick episode pool")
    ap.add_argument("--start"); ap.add_argument("--end")
    args = ap.parse_args()
    c = Ctx(args)
    d = step_dir(f"{c.prefix}{args.stage}_stage")
    tee = Tee(d / "log.txt"); sys.stdout = tee; TEE_HOLDER[0] = tee
    write_env_info(d, seed=0)
    log = lambda s: print(s, flush=True)  # noqa: E731
    T0 = time.time()
    log(f"{'PSEUDO-TEST (development data only)' if c.pseudo else 'CONFIRMATORY TEST'} stage {args.stage}: {c.start}..{c.end}")
    lines = globals()[f"stage_{args.stage}"](c, log, d)
    hdr = [f"# Phase 9 — {args.stage} ({'PSEUDO-TEST on development data' if c.pseudo else 'confirmatory test'})", "",
           f"Period {c.start} to {c.end}. Frozen plan sha256 {PLAN_HASH}. Runtime {time.time()-T0:.0f}s.", ""]
    (d / "report.md").write_text("\n".join(hdr + lines), encoding="utf-8")
    log(f"done in {time.time()-T0:.0f}s")
    sys.stdout = tee.stdout; tee.close()
    print(f"Handoff: {make_zip(d, f'{c.prefix}{args.stage}_stage')}")


if __name__ == "__main__":
    main()
