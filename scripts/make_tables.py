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


def f(x, d=3, sign=False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return DASH
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
           rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if groups and i in groups:
            if i:
                out.append(r"\midrule")
            out.append(rf"\multicolumn{{{len(header)}}}{{l}}{{\emph{{{groups[i]}}}}} \\")
        out.append(" & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
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
    "H8": r"Top-3 failure drivers stable ($\ge 80\%$ of resamples)",
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
            r"defined afterwards); H7 development: HAC $z$ and two-sided $p$ given in Section~\ref{sec:res-e4}. Holm correction over H1, H2a, H2b, H5 "
            r"(intersection--union, largest $p$) and H7; H3, H6 and H8 are threshold rules; H2-orig and H4 are "
            r"reported only. Analysis plan fixed and version-tagged before test access.")
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
            for m in ["ridge", "gbm"]:
                r = s[(s["setting"] == st) & (s["model"] == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append([SETTING_LABEL[st], {"ridge": "Ridge", "gbm": "LightGBM"}[m], f(r["pooled_r2_oos"], 3),
                             ci(r["r2_ci_lo"], r["r2_ci_hi"]), f(r["share_r2_pos"], 2), f(r["mean_cov90"], 3),
                             f(r["mean_width90"], 3)])
    return table("@{}llrlrrr@{}", ["Setting", "Model", "OOS $R^2$", "95\\% CI", "Share $R^2>0$", "Cov.\\ 90\\%",
                                   "Width"], rows,
                 "E1: index-level forecasting and transfer (20-day excess return, 42 country ETFs).", "tab:e1",
                 groups=groups, note="OOS $R^2$ against each market's own historical mean (Campbell--Thompson), "
                 "pooled over markets; block-bootstrap CI. Coverage and width of split-conformal 90\\% intervals.")


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
            "(quarter-block bootstrap).")
    return table("@{}lrrrr@{}", ["Scorer", "AUROC dev.", "AUROC test", "$\\rho_g$ dev.", "$\\rho_g$ test"], rows,
                 "E2: out-of-sample detection of transfer calibration failure, sorted by development AUROC.",
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
    return table("@{}lrr@{}", ["Driver group", "Share dev.", "Share test"], rows,
                 "E8: share of grouped permutation importance for calibration failure.", "tab:e8",
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
    fl = tab_e1_floor(R)
    if fl:
        tabs["tab_e1_floor"] = fl
    for k, v in tabs.items():
        (out / f"{k}.tex").write_text(v, encoding="utf-8")
    md = ["# Hypothesis outcomes (generated by scripts/make_tables.py — do not edit by hand)", "",
          "Development 2003–2021; confirmatory test 2022-01-01 to 2026-11-24, analysis plan fixed and "
          "version-tagged before test access (frozen plan sha256 a4fd6922…). Negative results are findings.", "",
          hyp_md, ""]
    (R.root / "hypothesis_outcomes.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{len(tabs)} tables written to {out}: " + ", ".join(tabs))
    print(f"hypothesis outcomes: {R.root / 'hypothesis_outcomes.md'}")


if __name__ == "__main__":
    main()
