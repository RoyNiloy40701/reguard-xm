"""Phase 10: generate every paper figure from results/ (vector PDF + 300-dpi PNG preview).

Usage:  python scripts/make_figures.py [--results DIR] [--out DIR]
Defaults: results/ and paper/figures/ under the project root. No number is typed by hand.
Colour: Okabe-Ito palette (colour-blind safe); every series also has its own marker or line style,
so the figures stay readable in greyscale.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reguard.utils.handoff import PROJECT_ROOT, RESULTS_ROOT  # noqa: E402
from reguard.viz.assets import (DEV, DRIVER_LABEL, GRID, INK, MARKERS, METHOD_LABEL, PALETTE,  # noqa: E402
                                PROPOSER_LABEL, PROPOSERS, SCORER_LABEL, SETTING_LABEL, TEST, Results, style)

W = 5.15  # Springer Nature sn-jnl text width (372 pt)
LEG_DEV = dict(color=DEV, marker="o", mfc="white", mew=1.3, ls="none", label="Development (2003–2021)")
LEG_TEST = dict(color=TEST, marker="s", ls="none", label="Test (2022–2026)")


def save(fig, out: Path, name: str, made: list) -> None:
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
    made.append(name)


def dev_test_dots(ax, y, dev, test, dev_ci=None, test_ci=None, off=0.17):
    """Development = hollow circle, test = filled square, offset vertically; optional CI whiskers."""
    for vals, ci, dy, kw in ((dev, dev_ci, -off, LEG_DEV), (test, test_ci, off, LEG_TEST)):
        yy = np.asarray(y, float) + dy
        if ci is not None:
            lo, hi = ci
            ax.hlines(yy, lo, hi, color=kw["color"], lw=1.0)
        ax.plot(vals, yy, **{k: v for k, v in kw.items() if k != "label"}, ms=5)


# ---------------------------------------------------------------- E1
def fig_e1(R: Results, out: Path, made: list) -> None:
    d, t = R.e1_summary(False), R.e1_summary(True)
    settings = ["indomain", "lomo", "pooled", "us_only"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.15), sharey=True, gridspec_kw=dict(wspace=0.16))
    y = np.arange(len(settings))[::-1]
    for ax, (model, title) in zip(axes[:2], (("ridge", "(a) Ridge"), ("gbm", "(b) LightGBM"))):
        def get(df, col):
            s = df[df["model"] == model].set_index("setting")
            return np.array([s[col].get(k, np.nan) for k in settings])
        dev_test_dots(ax, y, get(d, "pooled_r2_oos"), get(t, "pooled_r2_oos"),
                      (get(d, "r2_ci_lo"), get(d, "r2_ci_hi")), (get(t, "r2_ci_lo"), get(t, "r2_ci_hi")))
        ax.axvline(0, color="#777777", lw=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Out-of-sample $R^2$")
    ax = axes[2]
    def cov(df):
        s = df[df["model"] == "ridge"].set_index("setting")
        return np.array([s["mean_cov90"].get(k, np.nan) for k in settings])
    dev_test_dots(ax, y, cov(d), cov(t))
    ax.axvline(0.90, color="#777777", lw=0.8, ls="--")
    ax.text(0.895, y.max() + 0.5, "0.90", fontsize=7, color="#555555", va="center", ha="right")
    ax.set_title("(c) Ridge intervals", loc="left")
    ax.set_xlabel("Coverage (nominal 0.90)")
    axes[0].set_yticks(y, [SETTING_LABEL[s] for s in settings])
    for a in axes:
        a.grid(axis="y", visible=False)
    axes[0].set_ylim(-0.6, len(settings) - 0.2)
    h = [plt.Line2D([], [], **LEG_DEV), plt.Line2D([], [], **LEG_TEST)]
    fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    save(fig, out, "fig_e1_transfer", made)


# ---------------------------------------------------------------- E2
def fig_e2(R: Results, out: Path, made: list) -> None:
    d, t = R.e2_scores(False), R.e2_scores(True)
    keep = [k for k in d.index if k in t.index]
    order = sorted(keep, key=lambda k: d.at[k, "auroc"])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(W * 0.78, 3.4))
    dev_test_dots(ax, y, d.loc[order, "auroc"].to_numpy(), t.loc[order, "auroc"].to_numpy(), off=0.0)
    for i, k in enumerate(order):
        lo, hi = sorted((d.at[k, "auroc"], t.at[k, "auroc"]))
        ax.hlines(i, lo, hi, color="#BBBBBB", lw=0.8, zorder=0)
    ax.axvline(0.5, color="#777777", lw=0.8, ls="--")
    labels = [SCORER_LABEL.get(k, k) for k in order]
    ax.set_yticks(y, labels)
    for lab in ax.get_yticklabels():
        if lab.get_text().startswith("TRE"):
            lab.set_fontweight("bold")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUROC for calibration failure ($g_{cal}$), out of sample")
    ax.legend(handles=[plt.Line2D([], [], **LEG_DEV), plt.Line2D([], [], **LEG_TEST)], loc="lower right")
    save(fig, out, "fig_e2_auroc", made)


# ---------------------------------------------------------------- E3
def fig_e3(R: Results, out: Path, made: list) -> None:
    methods = ["fixed", "aci_target", "source_indomain_split", "pool_aci", "cscrb"]
    styles = ["-", "--", ":", "-", "-."]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.3), gridspec_kw=dict(width_ratios=[1.3, 1, 1], wspace=0.5))
    # (a) development entry curves
    c = R.csv("e3", "curves.csv").set_index("day")
    ax = axes[0]
    for i, m in enumerate(methods):
        s = c[m].rolling(20, min_periods=1).mean()
        ax.plot(s.index + 1, s.to_numpy(), color=PALETTE[i], ls=styles[i], lw=1.4 if m != "pool_aci" else 2.0,
                label=METHOD_LABEL[m])
    ax.axhline(0.05, color="#777777", lw=0.8)
    ax.set_xlabel("Trading days since entry")
    ax.set_ylabel("Budget-exceedance frequency")
    ax.set_title("(a) Development (20-day mean)", loc="left", fontsize=7.5)
    ax.set_xlim(1, c.index.max() + 1)
    # (b, c) test pooled gaps by horizon
    hz = ["gap_20", "gap_60", "gap_120", "gap_full"]
    hl = ["20 d", "60 d", "120 d", "Full"]
    for ax, (variant, title) in zip(axes[1:], (("primary", r"(b) Test, $\alpha=0.05$"),
                                               ("alpha_0.01", r"(c) Test, $\alpha=0.01$"))):
        g = R.e3_gaps(True, variant)
        x = np.arange(len(hz))
        for i, m in enumerate(["fixed", "aci_target", "pool_aci"]):
            j = methods.index(m)
            ax.plot(x, g.loc[m, hz].to_numpy(), color=PALETTE[j], ls=styles[j], marker=MARKERS[j], ms=4.5,
                    label=METHOD_LABEL[m])
        ax.set_xticks(x, hl)
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Horizon after entry")
    axes[1].set_ylabel("Pooled gap $|\\mathrm{BEF}-\\alpha|$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.2),
               handlelength=2.4)
    save(fig, out, "fig_e3_coldstart", made)


# ---------------------------------------------------------------- E4
def fig_e4(R: Results, out: Path, made: list) -> None:
    y = np.arange(len(PROPOSERS))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.1), sharey=True, gridspec_kw=dict(wspace=0.1))
    h5d, h5t = (R.e4(t, "h5.csv").set_index("proposer").loc[PROPOSERS] for t in (False, True))
    dev_test_dots(axes[0], y, 100 * h5d["delta_cvar95"], 100 * h5t["delta_cvar95"],
                  (100 * h5d["ci_lo"], 100 * h5d["ci_hi"]), (100 * h5t["ci_lo"], 100 * h5t["ci_hi"]))
    axes[0].axvline(0, color="#777777", lw=0.8)
    axes[0].set_xlabel("Change in daily CVaR$_{95}$ (percentage points)")
    axes[0].set_title("(a) H5: governance effect on tail loss", loc="left")
    h6d, h6t = (R.e4(t, "h6.csv").set_index("proposer").loc[PROPOSERS] for t in (False, True))
    dev_test_dots(axes[1], y, h6d["sharpe_loss"], h6t["sharpe_loss"])
    axes[1].axvline(0.15, color="#777777", lw=0.8, ls="--")
    axes[1].text(0.153, y.max() + 0.42, "limit 0.15", fontsize=7, color="#555555", va="center")
    axes[1].axvline(0, color="#777777", lw=0.8)
    axes[1].set_xlabel("Annualised Sharpe loss from governance")
    axes[1].set_title("(b) H6: cost of caution", loc="left")
    axes[0].set_yticks(y, [PROPOSER_LABEL[p] for p in PROPOSERS])
    axes[0].set_ylim(-0.6, len(PROPOSERS) - 0.2)
    for a in axes:
        a.grid(axis="y", visible=False)
    fig.legend(handles=[plt.Line2D([], [], **LEG_DEV), plt.Line2D([], [], **LEG_TEST)], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.24))
    save(fig, out, "fig_e4_governance", made)

    # test wealth, small multiples (one panel per proposer: governed solid, ungoverned dashed)
    ds = R.csv("test_e4", "daily_series.csv", parse_dates=["date"])
    fig, axes = plt.subplots(1, 5, figsize=(W, 1.75), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, p in zip(axes, PROPOSERS):
        for gov, ls, col, lab in ((False, "--", "#777777", "Ungoverned"), (True, "-", TEST, "Governed")):
            s = ds[(ds["proposer"] == p) & (ds["governed"] == gov)].set_index("date")["net"]
            ax.plot(s.index, np.exp(np.log1p(s).cumsum()), ls=ls, color=col, lw=1.2, label=lab)
        ax.set_title(PROPOSER_LABEL[p], fontsize=7.5)
        ax.axhline(1, color=GRID, lw=0.6)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    axes[0].set_ylabel("Wealth (start = 1)")
    axes[-1].legend(loc="upper left", fontsize=6.5)
    save(fig, out, "fig_e4_wealth_test", made)


# ---------------------------------------------------------------- E8
def fig_e8(R: Results, out: Path, made: list) -> None:
    d, t = R.e8_importance(False), R.e8_importance(True)
    order = d.sort_values().index.tolist()
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.55))
    ax = axes[0]
    dev_test_dots(ax, y, d.loc[order].to_numpy(), t.reindex(order).to_numpy())
    ax.set_yticks(y, [DRIVER_LABEL.get(k, k) for k in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of grouped permutation importance")
    ax.set_title("(a) Drivers of calibration failure", loc="left")
    ax.legend(handles=[plt.Line2D([], [], **LEG_DEV), plt.Line2D([], [], **LEG_TEST)], loc="lower right",
              fontsize=6.5)
    # (b) transfer-risk map (development): failure rate by market vs volatility level
    m = R.csv("e8", "market_map.csv")
    ax = axes[1]
    for em, mk, col, lab in ((0, "o", "#555555", "Developed"), (1, "^", PALETTE[3], "Emerging")):
        s = m[m["em"] == em]
        ax.scatter(s["vol_level"], s["failure_rate"], marker=mk, s=18, facecolor="white" if em == 0 else col,
                   edgecolor=col, linewidth=1.0, label=lab, zorder=3)
    base = m["failure_rate"].mul(m["episodes"]).sum() / m["episodes"].sum()
    ax.axhline(base, color="#777777", lw=0.8, ls="--")
    ax.text(m["vol_level"].max(), base, "base rate", fontsize=6.5, color="#555555", ha="right", va="bottom")
    for _, r in m.nlargest(5, "failure_rate").iterrows():
        ax.annotate(r["market"], (r["vol_level"], r["failure_rate"]), xytext=(3, 2), textcoords="offset points",
                    fontsize=6.5, color=INK)
    ax.set_xlabel("Mean target volatility (development)")
    ax.set_ylabel("Calibration-failure rate")
    ax.set_title("(b) Transfer-risk map by market", loc="left")
    ax.legend(loc="lower right", fontsize=6.5)
    save(fig, out, "fig_e8_drivers", made)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS_ROOT))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "paper" / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style(plt)
    R, made = Results(Path(a.results)), []
    for f in (fig_e1, fig_e2, fig_e3, fig_e4, fig_e8):
        f(R, out, made)
    print(f"{len(made)} figures written to {out}: " + ", ".join(made))


if __name__ == "__main__":
    main()
