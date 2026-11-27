"""Shared loaders and style for the paper figures and tables (Phase 10).

Every number in the paper's figures and tables is read from results/<step>/ by these helpers;
nothing is typed by hand. Development results live in results/{e1,e2,e3,e3_sensitivity,e4,e8}/,
confirmatory-test results in results/test_*/ (written by scripts/run_final_test.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Okabe-Ito order, checked with the dataviz validator (light surface): all checks pass; the one adjacent pair in
# the 6-8 CVD band (green/purple) is always paired with distinct markers and direct labels.
PALETTE = ["#0072B2", "#D55E00", "#CC79A7", "#009E73", "#B8860B"]
DEV, TEST = PALETTE[0], PALETTE[1]
MARKERS = ["o", "s", "^", "D", "v"]
INK, GRID = "#222222", "#DDDDDD"

PROPOSERS = ["equal_weight", "momentum", "mean_variance", "risk_parity", "ppo_ensemble"]
PROPOSER_LABEL = {"equal_weight": "Equal weight", "momentum": "Momentum", "mean_variance": "Mean-variance",
                  "risk_parity": "Risk parity", "ppo_ensemble": "PPO ensemble"}
SETTING_LABEL = {"indomain": "In-domain", "lomo": "Leave-one-market-out", "pooled": "Pooled", "us_only": "US only"}
METHOD_LABEL = {"fixed": "Fixed (Gaussian)", "heuristic": "Heuristic", "source_indomain_split": "Source split",
                "time_weighted": "Time-weighted", "regime_weighted": "Regime-weighted", "tcp": "TCP",
                "aci_target": "Own-history ACI", "dtaci": "DtACI", "pool_static": "Pool (static)",
                "pool_aci": "Pool + ACI", "pool_strata_aci": "Pool strata + ACI", "cscrb": "CSCRB",
                "cscrb_static": "CSCRB (static)"}
DRIVER_LABEL = {"volatility": "Volatility", "drawdown_state": "Drawdown state", "atypicality": "Atypicality",
                "correlation_with_sources": "Correlation with sources", "region": "Region",
                "em_status": "EM status", "liquidity": "Liquidity", "currency_regime": "Currency regime"}
SCORER_LABEL = {"TRE-LR": "TRE (logistic)", "TRE-GBM": "TRE (monotone GBM)", "abs_log_vol_ratio": "Volatility ratio",
                "atc_gap": "ATC", "cka_sim": "CKA similarity", "corr_src": "Correlation with sources",
                "dc_auc": "Domain-classifier AUC", "disagree_ratio": "Disagreement ratio", "doc": "DoC",
                "em_flag": "EM flag", "ens_disagree": "Ensemble disagreement", "ks_max": "KS (max)",
                "ks_mean": "KS (mean)", "liq_gap": "Liquidity gap", "mahalanobis": "Mahalanobis", "mmd": "MMD",
                "pred_entropy": "Predictive entropy", "transfer_risk": "Transfer risk (Cao)",
                "xs_atypicality": "Cross-sectional atypicality"}


class Results:
    """Reads result tables; raises a clear error naming the missing file and the command that creates it."""

    MAKER = {"e1": "run_e1.py", "e2": "run_e2.py", "e3": "run_e3.py", "e3_sensitivity": "run_e3_sensitivity.py",
             "e4": "run_e4.py", "e8": "run_e8.py"}

    def __init__(self, root: Path):
        self.root = Path(root)

    def csv(self, step: str, name: str, **kw) -> pd.DataFrame:
        f = self.root / step / name
        if not f.exists():
            how = self.MAKER.get(step, f"run_final_test.py {step.replace('test_', '').replace('_stage', '')}")
            raise FileNotFoundError(f"missing {f} (created by scripts/{how})")
        return pd.read_csv(f, **kw)

    # ---- E1
    def e1_summary(self, test: bool) -> pd.DataFrame:
        return self.csv("test_e1" if test else "e1", "summary_ts.csv")

    def e1_cs(self, test: bool) -> pd.DataFrame:
        return self.csv("test_e1" if test else "e1", "cs_ic.csv")

    # ---- E2
    def e2_scores(self, test: bool, label: str = "g_cal") -> pd.DataFrame:
        s = self.csv("test_e2_stage" if test else "e2", "scores.csv")
        return s[s["label"] == label].set_index("scorer")

    def e2_h1(self, test: bool) -> pd.Series:
        h = self.csv("test_e2_stage" if test else "e2", "h1.csv")
        return h[(h["label"] == "g_cal") & (h["tre"] == "rho_lr")].iloc[0]

    # ---- E3
    def e3_gaps(self, test: bool, variant: str = "primary") -> pd.DataFrame:
        if test:
            s = self.csv("test_e3_stage", "summary.csv")
            return s[s["variant"] == variant].set_index("method")
        return self.csv("e3", "summary.csv").set_index("method")

    # ---- E4
    def e4(self, test: bool, name: str) -> pd.DataFrame:
        d = self.csv("test_e4" if test else "e4", name)
        return d[d["proposer"].isin(PROPOSERS)] if "proposer" in d else d

    # ---- E8
    def e8_importance(self, test: bool) -> pd.Series:
        if test:
            s = self.csv("test_e8_stage", "importance.csv", index_col=0)["mse_increase"]
        else:
            s = self.csv("e8", "importance.csv").set_index("driver_group")["mse_increase"]
        pos = s.clip(lower=0)
        return pos / pos.sum()          # share of total positive importance (negatives = no importance)

    # ---- hypotheses
    def final_hypotheses(self) -> pd.DataFrame:
        return self.csv("test_summary_stage", "final_hypotheses.csv").set_index("hypothesis")

    def dev_hypotheses(self) -> pd.DataFrame:
        """Development-period values of the frozen-plan hypotheses (H2a/H2b were defined after development,
        so their development values come from the exploratory sensitivity analysis)."""
        rows = {}
        h1 = self.e2_h1(False)
        rows["H1"] = dict(estimate=h1["diff"], ci_lo=h1["ci_lo"], ci_hi=h1["ci_hi"], p=np.nan,
                          note=f"TRE-LR {h1['auroc_tre']:.3f} vs {h1['best_single']} {h1['auroc_best_single']:.3f}")
        s4 = self.csv("e3_sensitivity", "s4_alpha.csv")
        s4 = s4[s4["variant"] == "alpha=0.01"].set_index("method")
        rows["H2a"] = dict(estimate=s4.at["pool_aci", "gap_20"] - s4.at["aci_target", "gap_20"], ci_lo=np.nan,
                           ci_hi=np.nan, p=np.nan, note="exploratory: Q1 entries, S4")
        s1 = self.csv("e3_sensitivity", "s1_timing.csv")
        w = s1.pivot(index="quarter", columns="method", values="gap_20")
        rows["H2b"] = dict(estimate=w["pool_aci"].max() - w["aci_target"].max(), ci_lo=np.nan, ci_hi=np.nan,
                           p=np.nan, note="exploratory: quarter starts, S1")
        g = self.e3_gaps(False)
        rows["H3"] = dict(estimate=g.at["pool_aci", "gap_full"], ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
                          note="pool + ACI full-period gap")
        h5 = self.e4(False, "h5.csv")
        rows["H5"] = dict(estimate=h5["delta_cvar95"].max(), ci_lo=np.nan, ci_hi=h5["ci_hi"].max(), p=np.nan,
                          note="largest (least negative) change in CVaR95")
        h6 = self.e4(False, "h6.csv")
        rows["H6"] = dict(estimate=h6["sharpe_loss"].max(), ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
                          note="largest Sharpe loss")
        h7 = self.csv("e4", "h7.csv").iloc[0]
        rows["H7"] = dict(estimate=h7["sharpe_diff_ann"], ci_lo=np.nan, ci_hi=np.nan, p=h7["p"],
                          note=f"z = {h7['z']:.2f} (two-sided p)")
        b = self.csv("e8", "bootstrap_top3.csv")
        rows["H8"] = dict(estimate=b["share"].max(), ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
                          note="modal top-3 share: " + b.iloc[b["share"].argmax()]["top3"])
        return pd.DataFrame(rows).T


def style(plt) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False, "legend.numpoints": 1,
        "axes.edgecolor": "#888888", "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": "#555555", "ytick.color": "#555555", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.5, "lines.markersize": 5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
