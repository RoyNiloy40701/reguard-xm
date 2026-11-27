"""Phase 10: generate every paper table (LaTeX, booktabs) and results/hypothesis_outcomes.md from results/.

Usage:  python scripts/make_tables.py [--results DIR] [--out DIR]
Defaults: results/ and paper/tables/ under the project root. No number is typed by hand.
The LaTeX preamble needs \\usepackage{booktabs,array}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reguard.utils.handoff import PROJECT_ROOT, RESULTS_ROOT  # noqa: E402
from reguard.viz.assets import (DRIVER_LABEL, METHOD_LABEL, PROPOSER_LABEL, PROPOSERS, SCORER_LABEL,  # noqa: E402
                                SETTING_LABEL, Results)

DASH = "--"
MODEL_LABEL = {"hist_mean": "Hist.\\ mean", "ar1": "AR(1)", "ridge": "Ridge", "gbm": "LightGBM",
               "mlp_ens": "MLP ens."}


def f(x, d=3, sign=False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return DASH
    if round(float(x), d) == 0:
        x = 0.0
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") or "-" in s else s


def ci(lo, hi, d=3) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return DASH
    return f"[{f(lo, d)}, {f(hi, d)}]"


def tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def table(cols: str, header: list[str], rows: list[list[str]], caption: str, label: str,
          groups: dict[int, str] | None = None, note: str = "", size: str = r"\footnotesize\setlength{\tabcolsep}{4pt}") -> str:
    out = [r"\begin{table}[t]", r"\centering", size, rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           r"\begin{fitwidth}", rf"\begin{{fittabular}}{{{cols}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if groups and i in groups:
            if i:
                out.append(r"\midrule")
            out.append(rf"\multicolumn{{{len(header)}}}{{l}}{{\emph{{{groups[i]}}}}} \\")
        out.append(" & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{fittabular}", r"\end{fitwidth}"]
    if note:
        out.append(rf"\par\smallskip\parbox{{\linewidth}}{{\footnotesize {note}}}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- hypotheses
H_TEXT = {
    "H1": "Learned TRE beats the best single shift signal (AUROC, $g_{cal}$)",
    "H2a": r"Pooled budget beats own-history ACI at entry ($\alpha=0.01$, 20 days)",
    "H2b": r"Pooled budget beats own-history ACI in the worst entry quarter ($\alpha=0.05$)",
    "H3": r"Pooled budget full-period calibration gap $\le 0.01$",
    "H2-orig": "Cold start halves the best baseline's 60-day gap in $\\ge$ 2/3 of markets",
    "H4": "Estimated stratum gap predicts realised entry excess",
    "H5": "Governance lowers CVaR$_{95}$ for every proposer",
    "H6": r"Cost of caution bounded (Sharpe loss $\le 0.15$, cash $\le 35\%$)",
    "H7": "Governed PPO beats governed mean--variance (Sharpe)",
    "H8": r"Top-3 associated characteristics stable ($\ge 80\%$ of resamples)",
}
ORDER = ["H1", "H2a", "H2b", "H3", "H5", "H6", "H7", "H8", "H2-orig", "H4"]


def tab_hypotheses(R: Results) -> tuple[str, str]:
    T, D = R.final_hypotheses(), R.dev_hypotheses()
    rows, md = [], ["| Hypothesis | Development (2003–2021) | Test (2022–2026) | Test 95% CI | Holm p | Verdict |",
                    "|---|---|---|---|---|---|"]
    for h in ORDER:
        t = T.loc[h]
        d = D.loc[h] if h in D.index else None
        dev = f(d["estimate"], 4) if d is not None else DASH
        verdict = t["verdict"].replace("(reported only)", "").strip()
        verdict_tex = r"\textbf{Supported}" if verdict == "SUPPORTED" else (
            "Not supported" if verdict == "NOT SUPPORTED" else verdict.capitalize() + " (reported)")
        rows.append([h, H_TEXT[h], dev, f(t["estimate"], 4), ci(t["ci_lo"], t["ci_hi"], 4),
                     f(t["p_holm"], 3), verdict_tex])
        txt = H_TEXT[h].replace(r"\le", "≤").replace(r"\ge", "≥").replace("--", "–").replace("$", "").replace(chr(92), "")
        md.append(f"| {h} {txt} | "
                  f"{f(d['estimate'], 4).replace('$', '') if d is not None else '–'}"
                  f"{'; ' + d['note'] if d is not None else ''} | {t['estimate']:.4f} | "
                  f"{'–' if not np.isfinite(t['ci_lo']) else f'[{t.ci_lo:.4f}, {t.ci_hi:.4f}]'} | "
                  f"{'–' if not np.isfinite(t['p_holm']) else f'{t.p_holm:.3f}'} | {t['verdict']} — {t['detail']} |")
    note = (r"Development values for H2a and H2b come from the exploratory sensitivity analysis (the hypotheses were "
            r"defined afterwards). Units: H1 AUROC difference; H2a, H2b, H3 exceedance-rate gaps; H5 largest (least "
            r"negative) change in daily CVaR$_{95}$ (fraction of portfolio value); H6 largest Sharpe loss; H7 annualised "
            r"Sharpe difference; H8 share of resamples; H2-orig share of markets; H4 Spearman $\rho$. H7 development: HAC $z$ and two-sided $p$ given in Section~\ref{sec:res-e4}. Holm correction over H1, H2a, H2b, H5 "
            r"(intersection--union, largest $p$) and H7; H3, H6 and H8 are threshold rules; H2-orig and H4 are "
            r"reported only. Analysis plan fixed before test access (SHA-256 hash recorded).")
    tex = table(r"@{}l>{\raggedright\arraybackslash}p{4.3cm}rrlrl@{}", ["", "Hypothesis", "Dev.", "Test", "Test 95\\% CI", "Holm $p$", "Verdict"], rows,
                "Confirmatory hypotheses: development and locked-test results.", "tab:hypotheses",
                groups={0: "Frozen-plan family", 8: "Reported only"}, note=note, size=r"\scriptsize\setlength{\tabcolsep}{2.5pt}")
    return tex, "\n".join(md)


# ---------------------------------------------------------------- E1
def tab_e1(R: Results) -> str:
    rows, groups = [], {}
    for k, test in enumerate((False, True)):
        s = R.e1_summary(test)
        groups[len(rows)] = "Test (2022--2026)" if test else "Development (2003--2021)"
        for st in ["indomain", "lomo", "pooled", "us_only"]:
            for m in ["hist_mean", "ar1", "ridge", "gbm", "mlp_ens"]:
                r = s[(s["setting"] == st) & (s["model"] == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append([SETTING_LABEL[st], MODEL_LABEL[m], f(r["pooled_r2_oos"], 3),
                             ci(r["r2_ci_lo"], r["r2_ci_hi"]), f(r["share_r2_pos"], 2), f(r["mean_cov90"], 3),
                             f(r["mean_width90"], 3)])
    return table("@{}llrlrrr@{}", ["Setting", "Model", "OOS $R^2$", "95\\% CI", "Share $R^2>0$", "Cov.\\ 90\\%",
                                   "Width"], rows,
                 "E1: index-level forecasting and transfer (20-day excess return; 43 ETFs covering 42 markets).", "tab:e1",
                 groups=groups, note="OOS $R^2$ against each ETF's own historical mean (Campbell--Thompson), pooled "
                 "over ETFs; block-bootstrap 95\\% CI; share of ETFs with $R^2>0$. Coverage and width of split-conformal "
                 "90\\% intervals calibrated on each setting's source pool. The historical mean is the benchmark (in-domain "
                 "$R^2=0$ by construction). AR(1) needs the target's own history and is run in-domain only; the MLP ensemble "
                 "was not run leave-one-market-out (compute).", size=r"\scriptsize\setlength{\tabcolsep}{3pt}")


def tab_e1_cs(R: Results) -> str:
    rows, groups = [], {}
    for test in (False, True):
        c = R.e1_cs(test)
        groups[len(rows)] = "Test (2022--2026)" if test else "Development (2003--2021)"
        for (src, tgt) in [("dse", "dse"), ("pooled", "dse"), ("tierb", "dse"), ("tierb", "tierb"), ("dse", "tierb")]:
            r = c[(c["source"] == src) & (c["target"] == tgt) & (c["model"] == "ridge")]
            if r.empty:
                continue
            r = r.iloc[0]
            nm = {"dse": "DSE", "tierb": "Country ETFs", "pooled": "Pooled"}
            rows.append([nm[src], nm[tgt], f(r["rank_ic_mean"], 3), ci(r["ci_lo"], r["ci_hi"]), f(r["nw_t"], 1),
                         str(int(r["days"]))])
    return table("@{}llrlrr@{}", ["Source", "Target", "Rank IC", "95\\% CI", "NW $t$", "Days"], rows,
                 "E1: cross-sectional ranking transfer between country ETFs and Dhaka stocks (ridge).", "tab:e1cs",
                 groups=groups, note="The test-period DSE level includes the 2022--2024 floor-price regime "
                 "(see Table~\\ref{tab:e1floor}).")


# ---------------------------------------------------------------- E2
def tab_e2(R: Results) -> str:
    d, t = R.e2_scores(False), R.e2_scores(True)
    keep = [k for k in d.sort_values("auroc", ascending=False).index if k in t.index]
    rows = [[tex_escape(SCORER_LABEL.get(k, k)), f(d.at[k, "auroc"]), f(t.at[k, "auroc"]),
             f(d.at[k, "spearman_g"], 2), f(t.at[k, "spearman_g"], 2)] for k in keep]
    hd, ht = R.e2_h1(False), R.e2_h1(True)
    note = (f"Calibration-failure label $g_{{cal}}$; development {int(hd['n_episodes'])} scored episodes, test "
            f"{int(ht['n_episodes'])}. H1 difference (TRE-LR minus best single signal): development "
            f"{f(hd['diff'])} {ci(hd['ci_lo'], hd['ci_hi'])}, test {f(ht['diff'])} {ci(ht['ci_lo'], ht['ci_hi'])} "
            "(quarter-block bootstrap). Confidence intervals are available only for this pre-specified difference.")
    return table("@{}lrrrr@{}", ["Scorer", "AUROC dev.", "AUROC test", "$\\rho_g$ dev.", "$\\rho_g$ test"], rows,
                 "E2: out-of-sample detection of transfer calibration failure, sorted by development AUROC. $\\rho_g$: Spearman "
                 "correlation of the scorer with the continuous label $g_{cal}$.",
                 "tab:e2", note=note, size=r"\footnotesize")


# ---------------------------------------------------------------- E3
def tab_e3(R: Results) -> str:
    methods = ["fixed", "heuristic", "time_weighted", "tcp", "aci_target", "dtaci", "source_indomain_split",
               "pool_static", "pool_aci", "pool_strata_aci", "cscrb"]
    d, t5, t1 = R.e3_gaps(False), R.e3_gaps(True, "primary"), R.e3_gaps(True, "alpha_0.01")
    rows = []
    for m in methods:
        rows.append([METHOD_LABEL[m]] + [f(d.at[m, c], 4) for c in ("gap_20", "gap_full")]
                    + [f(t5.at[m, c], 4) if m in t5.index else DASH for c in ("gap_20", "gap_60", "gap_full")]
                    + [f(t1.at[m, c], 4) if m in t1.index else DASH for c in ("gap_20", "gap_full")])
    hdr = ["Method", "Dev.\\ 20d", "Dev.\\ full", "Test 20d", "Test 60d", "Test full", "Test 20d", "Test full"]
    tex = table("@{}lrrrrrrr@{}", hdr, rows,
                "E3: pooled calibration gap $|\\mathrm{BEF}-\\alpha|$ after entry into an unseen market.", "tab:e3",
                note="Development: $\\alpha=0.05$, entries at January starts, 20-day volatility scale. Test (frozen plan): entries "
                "at every quarter start, 60-day volatility scale, "
                "$\\alpha=0.05$ (columns 4--6) and $\\alpha=0.01$ (columns 7--8). Lower is better.",
                size=r"\footnotesize")
    return tex.replace(r"\toprule", r"\toprule" + "\n" + r" & \multicolumn{2}{c}{$\alpha=0.05$} & "
                       r"\multicolumn{3}{c}{$\alpha=0.05$} & \multicolumn{2}{c}{$\alpha=0.01$} \\"
                       + "\n" + r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}", 1)


def tab_e3_stress(R: Results) -> str:
    s = R.csv("test_e3_stage", "stress.csv")
    cols = [c for c in ["fixed", "aci_target", "time_weighted", "pool_aci", "cscrb"] if c in s]
    rows = [[r["window"].replace("..", "--")] + [f(r[c], 3) for c in cols] for _, r in s.iterrows()]
    return table("@{}l" + "r" * len(cols) + "@{}", ["Window"] + [METHOD_LABEL[c] for c in cols], rows,
                 "E3: budget-exceedance frequency in pre-declared test stress windows ($\\alpha=0.05$).",
                 "tab:e3stress", size=r"\footnotesize")


# ---------------------------------------------------------------- E4
def tab_e4(R: Results, test: bool) -> str:
    s = R.e4(test, "summary.csv").set_index(["proposer", "governed"])
    rows = []
    for p in PROPOSERS:
        for g in (False, True):
            r = s.loc[(p, g)]
            rows.append([PROPOSER_LABEL[p] if not g else "", "Governed" if g else "Ungoverned",
                         f(100 * r["ann_return"], 1), f(100 * r["ann_vol"], 1), f(r["sharpe"], 2),
                         f(100 * r["max_dd"], 1), f(100 * r["cvar95"], 2), f(r["turnover"], 2),
                         f(100 * r["cash_share"], 1), f(r["dsr"], 2)])
    per = "test period 2022--2026" if test else "development period 2018--2021"
    return table("@{}llrrrrrrrr@{}", ["Proposer", "", "Ret.\\ \\%", "Vol.\\ \\%", "Sharpe", "MDD \\%",
                                      "CVaR$_{95}$ \\%", "Turnover", "Cash \\%", "DSR"], rows,
                 f"E4: weekly country-ETF portfolios with and without the governance layer, {per}.",
                 "tab:e4" + ("" if test else "dev"),
                 note="Annualised return and volatility; daily CVaR$_{95}$; costs 10 bps; risk budget "
                 "$B_{\\max}=2\\%$. DSR = deflated Sharpe ratio.", size=r"\scriptsize\setlength{\tabcolsep}{3pt}")


# ---------------------------------------------------------------- E8
def tab_e8(R: Results) -> str:
    d, t = R.e8_importance(False), R.e8_importance(True)
    rows = [[DRIVER_LABEL.get(k, k), f(d[k], 3), f(t.get(k, np.nan), 3)] for k in d.sort_values(ascending=False).index]
    b = R.csv("e8", "bootstrap_top3.csv")
    T = R.final_hypotheses().loc["H8"]
    return table("@{}lrr@{}", ["Characteristic group", "Share dev.", "Share test"], rows,
                 "E8: share of grouped permutation importance of market characteristics associated with calibration failure.", "tab:e8",
                 note=f"Modal top-3 set in {100 * b['share'].max():.1f}\\% of development and "
                 f"{100 * T['estimate']:.1f}\\% of test quarter-block resamples (criterion 80\\%).")


# ---------------------------------------------------------------- E1 floor sensitivity (exploratory)
def tab_e1_floor(R: Results) -> str | None:
    f_ = R.root / "test_e1_floor" / "floor_sensitivity.csv"
    if not f_.exists():
        return None
    s = pd.read_csv(f_)
    rows = [[r["subset"], {"dse": "DSE", "tierb": "Country ETFs", "pooled": "Pooled"}[r["source"]],
             r["model"], f(r["rank_ic_mean"], 3), ci(r["ci_lo"], r["ci_hi"]), f(r["nw_t"], 1), str(int(r["days"]))]
            for _, r in s.iterrows()]
    return table("@{}lllrlrr@{}", ["Test days", "Source", "Model", "Rank IC", "95\\% CI", "NW $t$", "Days"], rows,
                 "Exploratory: DSE in-domain and transferred rank IC on the test period, with and without the "
                 "2022--2024 floor-price regime.", "tab:e1floor", size=r"\footnotesize",
                 note="Exploratory analysis designed after the test results; not part of the frozen plan.")


# ---------------------------------------------------------------- revision tables (all values read from result files)
def tab_sample_flow(R: Results) -> str:
    epi = R.csv("e2", "episodes_summary.csv")
    h1d, h1t = R.e2_h1(False), R.e2_h1(True)
    e3d = R.csv("e3", "entries.csv"); e3d = e3d[e3d["method"] == "fixed"]
    e3t = R.csv("test_e3_stage", "entries.csv"); e3t = e3t[(e3t["variant"] == "primary") & (e3t["method"] == "fixed")]
    e4d = R.e4(False, "summary.csv"); e4t = R.e4(True, "summary.csv")
    dm = R.csv("test_e1", "dm_tests.csv")
    mm = R.csv("e8", "market_map.csv")
    rows = [
        ["E1", "ETF $\\times$ day", f"{dm['asset'].nunique()} ETFs ({R.e1_summary(False)['markets'].max()} markets), 2003--2021",
         f"same ETFs, 2022--2026-11-24", "annual refits, quarterly evaluation"],
        ["E2", "ETF $\\times$ quarter", f"{int(epi['episodes'].sum()):,} episodes ({int(epi['markets'].min())}--{int(epi['markets'].max())} markets); "
         f"{int(h1d['n_episodes']):,} scored", f"{int(h1t['n_episodes'])} episodes",
         f"first {int(epi['episodes'].sum()) - int(h1d['n_episodes'])} dev.\\ episodes train only"],
        ["E3", "ETF $\\times$ entry", f"{len(e3d)} entries ({e3d['year'].min()}--{e3d['year'].max()}, January)",
         f"{len(e3t)} entries ({e3t['asset'].nunique()} ETFs $\\times$ 16 quarter starts)", "entries repeat by ETF"],
        ["E4", "trading day", f"{int(e4d['days'].iloc[0]):,} days (2018--2021)", f"{int(e4t['days'].iloc[0]):,} days (2022--2026)",
         "PPO trained 2003--2017"],
        ["E8", "episode", f"{int(mm['episodes'].sum()):,} episodes, {len(mm)} markets", f"{int(h1t['n_episodes'])} episodes",
         "1,000 quarter-block resamples"],
    ]
    return table("@{}llp{3.8cm}p{3.3cm}p{2.6cm}@{}", ["Exp.", "Unit", "Development", "Test", "Note"], rows,
                 "Sample flow by experiment (counts read from the result files).", "tab:sampleflow",
                 note="43 ETFs cover 42 markets; China is represented by two ETFs (FXI, MCHI). Episodes and entries are "
                 "defined per ETF and repeat over time; they are not independent markets.",
                 size=r"\footnotesize\setlength{\tabcolsep}{3pt}")


def tab_e1_matrix(R: Results) -> str:
    d, t = (R.csv(s, "transfer_matrix.csv") for s in ("e1", "test_e1"))
    d, t = d[d["model"] == "ridge"], t[t["model"] == "ridge"]
    regions = [c for c in ["Asia-Pacific", "Europe", "EMEA", "Middle East", "Latin America", "North America"]
               if c in set(d["target_region"])]
    rows = []
    for src in d["source"].unique():
        row = [src]
        for reg in regions:
            a = d[(d["source"] == src) & (d["target_region"] == reg)]["pooled_r2_oos"]
            b = t[(t["source"] == src) & (t["target_region"] == reg)]["pooled_r2_oos"]
            row.append(f"{f(a.iloc[0], 2) if len(a) else DASH}/{f(b.iloc[0], 2) if len(b) else DASH}")
        rows.append(row)
    n = d.groupby("target_region")["markets"].first()
    hdr = ["Source group"] + [f"{r} ({int(n[r])})" for r in regions]
    return table("@{}l" + "c" * len(regions) + "@{}", hdr, rows,
                 "E1: source-group $\\to$ target-region transfer matrix (ridge, pooled OOS $R^2$, development/test).",
                 "tab:e1matrix", size=r"\scriptsize\setlength{\tabcolsep}{2.5pt}",
                 note="Rows: training pool; columns: evaluated region (number of markets). Diagonal-type cells include "
                 "the target markets in the source pool. Block-bootstrap CIs in transfer\\_matrix.csv.")


def tab_e4_cash_matched(R: Results) -> str | None:
    f_ = R.root / "test_e4_cash_matched" / "cash_matched_comparison.csv"
    if not f_.exists():
        return None
    c = pd.read_csv(f_).set_index("proposer").loc[PROPOSERS]
    rows = [[PROPOSER_LABEL[p], f(100 * r["cash_governed"], 1), f(100 * r["cvar95_governed"], 2),
             f(100 * r["cvar95_cash_matched"], 2), f(100 * r["delta_cvar95"], 2),
             ci(100 * r["ci_lo"], 100 * r["ci_hi"], 2), f(r["sharpe_governed"], 2), f(r["sharpe_cash_matched"], 2),
             f(100 * r["max_dd_governed"], 1), f(100 * r["max_dd_cash_matched"], 1)] for p, r in c.iterrows()]
    return table("@{}lrrrrlrrrr@{}", ["Proposer", "Cash \\%", "CVaR gov.", "CVaR mix", "$\\Delta$ (pp)", "95\\% CI",
                                      "SR gov.", "SR mix", "MDD gov.", "MDD mix"], rows,
                 "Exploratory (post hoc): governed portfolio versus the ungoverned portfolio mixed with T-bills to the "
                 "same average cash share, test period 2022--2026.", "tab:e4cash",
                 size=r"\scriptsize\setlength{\tabcolsep}{3pt}",
                 note="Designed after the test results; computed from saved daily returns (no new data access). The mix "
                 "uses the governed portfolio's realised average cash share (hindsight) and pays no extra costs. Daily "
                 "CVaR$_{95}$ and MDD in \\%; $\\Delta$ = governed $-$ mix, paired stationary bootstrap (mean block 20).")


def tab_ppo_seeds(R: Results) -> str:
    out = {}
    for test in (False, True):
        s = R.csv("test_e4" if test else "e4", "summary.csv")
        s = s[s["proposer"].str.startswith("ppo")]
        out[test] = s.pivot(index="proposer", columns="governed", values="sharpe")
    idx = [f"ppo_seed{i}" for i in range(10)] + ["ppo_ensemble"]
    rows = [[("Seed " + p[8:]) if p != "ppo_ensemble" else "Ensemble (mean returns)",
             f(out[False].at[p, False], 2), f(out[False].at[p, True], 2), f(out[True].at[p, False], 2),
             f(out[True].at[p, True], 2)] for p in idx if p in out[False].index]
    return table("@{}lrrrr@{}", ["PPO policy", "Dev.\\ ungov.", "Dev.\\ gov.", "Test ungov.", "Test gov."], rows,
                 "E4: annualised Sharpe ratio of each PPO seed and of the ten-seed ensemble.", "tab:pposeeds",
                 note="Seeds share data and hyper-parameters; their spread measures optimisation variability, not "
                 "independent samples. Development 2018--2021 (hyper-parameters chosen here), test 2022--2026.")




COUNTRY = {"AE": "United Arab Emirates", "AR": "Argentina", "AT": "Austria", "AU": "Australia", "BE": "Belgium",
           "BR": "Brazil", "CA": "Canada", "CH": "Switzerland", "CL": "Chile", "CN": "China", "DE": "Germany",
           "DK": "Denmark", "ES": "Spain", "FI": "Finland", "FR": "France", "GB": "United Kingdom",
           "HK": "Hong Kong", "ID": "Indonesia", "IE": "Ireland", "IL": "Israel", "IN": "India", "IT": "Italy",
           "JP": "Japan", "KR": "South Korea", "MX": "Mexico", "MY": "Malaysia", "NL": "Netherlands", "NO": "Norway",
           "NZ": "New Zealand", "PE": "Peru", "PH": "Philippines", "PL": "Poland", "QA": "Qatar",
           "SA": "Saudi Arabia", "SE": "Sweden", "SG": "Singapore", "TH": "Thailand", "TR": "Turkey",
           "TW": "Taiwan", "US": "United States", "VN": "Vietnam", "ZA": "South Africa"}


def tab_etf_inventory(R: Results) -> str | None:
    f_ = R.root / "etf_inventory" / "etf_inventory.csv"
    if not f_.exists():
        return None
    d = pd.read_csv(f_)
    d = d.assign(_name=d["market"].map(COUNTRY).fillna(d["market"])).sort_values(["_name", "asset"])
    rows = [[tex_escape(COUNTRY.get(r["market"], r["market"])), tex_escape(r["asset"]), tex_escape(r["fund"]),
             str(r["first_date"]), str(r["investable_from"])] for _, r in d.iterrows()]
    return table("@{}lllll@{}", ["Market", "Ticker", "Fund", "First price", "Investable from"], rows,
                 f"Country ETFs used in the study ({len(d)} ETFs, {d['market'].nunique()} markets).", "tab:etfs",
                 size=r"\scriptsize\setlength{\tabcolsep}{4pt}",
                 note="First daily adjusted close in the development panel and the date after which the ETF has 252 "
                 "return days and enters the investable universe. Features and evaluation start in 2003. Generated "
                 "by scripts/etf\\_inventory.py from development data only.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS_ROOT))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "paper" / "tables"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = Results(Path(a.results))
    hyp_tex, hyp_md = tab_hypotheses(R)
    tabs = {"tab_hypotheses": hyp_tex, "tab_e1": tab_e1(R), "tab_e1_cs": tab_e1_cs(R), "tab_e2": tab_e2(R),
            "tab_e3": tab_e3(R), "tab_e3_stress": tab_e3_stress(R), "tab_e4_test": tab_e4(R, True),
            "tab_e4_dev": tab_e4(R, False), "tab_e8": tab_e8(R)}
    tabs.update({"tab_sample_flow": tab_sample_flow(R), "tab_e1_matrix": tab_e1_matrix(R),
                 "tab_ppo_seeds": tab_ppo_seeds(R)})
    cm = tab_e4_cash_matched(R)
    if cm:
        tabs["tab_e4_cash_matched"] = cm
        inv = tab_etf_inventory(R)
    if inv:
        tabs["tab_etf_inventory"] = inv
    fl = tab_e1_floor(R)
    if fl:
        tabs["tab_e1_floor"] = fl
    for k, v in tabs.items():
        (out / f"{k}.tex").write_text(v, encoding="utf-8")
    md = ["# Hypothesis outcomes (generated by scripts/make_tables.py — do not edit by hand)", "",
          "Development 2003–2021; confirmatory test 2022-01-01 to 2026-11-24, analysis plan fixed before "
          "test access (frozen plan sha256 a4fd6922…). Negative results are findings.", "",
          hyp_md, ""]
    (R.root / "hypothesis_outcomes.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{len(tabs)} tables written to {out}: " + ", ".join(tabs))
    print(f"hypothesis outcomes: {R.root / 'hypothesis_outcomes.md'}")


if __name__ == "__main__":
    main()
