"""Phase 0 smoke test for ReGuard-XM.

Runs a miniature version of the whole pipeline on SYNTHETIC markets to prove that the
environment works before any real data is used:

  1. package check            6. cold-start conformal budget + ACI (N2)
  2. synthetic markets        7. governance projection with cvxpy (N3)
  3. forecaster ensemble      8. tiny PPO training run (torch + stable-baselines3)
  4. pseudo-transfer episodes 9. other libraries (lightgbm, statsmodels, arch, calendars)
  5. TRE (N1)                10. internet check for data downloads (warning only)

Expected runtime: 2-6 minutes on a normal laptop CPU. No GPU needed.
The numbers produced are from fake data: they test the code, they are NOT research results.

Run from the "code" folder:
    python scripts/run_smoke_test.py
Then upload  results/smoke_test/handoff_smoke_test.zip  to Claude.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from reguard.utils.handoff import Tee, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "smoke_test"
SEED = 42
ALPHA = 0.05            # nominal budget-exceedance level
EMBARGO = 5             # days between training data and evaluation window

RESULTS: list[dict] = []   # one row per stage
METRICS: dict[str, float] = {}


# ----------------------------------------------------------------------------- helpers
def stage(name: str, required: bool = True):
    """Decorator: run a stage, record PASS / FAIL / WARN and time, never crash the script."""

    def wrap(fn):
        def run(*args, **kwargs):
            t0 = time.time()
            print(f"\n=== {name} ===")
            try:
                out = fn(*args, **kwargs)
                status, note = "PASS", ""
            except Exception as exc:  # noqa: BLE001
                out = None
                status = "FAIL" if required else "WARN"
                note = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            dt = time.time() - t0
            print(f"--> {status} ({dt:.1f}s) {note}")
            RESULTS.append({"stage": name, "status": status, "seconds": round(dt, 1), "note": note})
            return out

        return run

    return wrap


def spearman_ic(pred: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if np.std(pred) == 0 or np.std(y) == 0:
        return 0.0
    return float(spearmanr(pred, y).statistic)


# ----------------------------------------------------------------------------- 1. packages
@stage("1. Package check")
def check_packages(out_dir: Path):
    versions = write_env_info(out_dir, seed=SEED)
    missing = [k for k, v in versions.items() if v.startswith("MISSING")]
    for k, v in versions.items():
        print(f"  {k:20s} {v}")
    py = sys.version_info
    if (py.major, py.minor) != (3, 11):
        print(f"  NOTE: Python {py.major}.{py.minor} detected; the project expects 3.11.")
    if missing:
        raise RuntimeError(f"missing packages: {missing}")


# ----------------------------------------------------------------------------- 2. data
@stage("2. Synthetic markets")
def make_markets(n_markets: int = 6, n_assets: int = 30, n_days: int = 1500):
    """Six fake markets. Markets close to the typical feature distribution share a positive
    signal; markets whose features are shifted have weaker or reversed signals. Market 5
    (the 'unseen target') is the most shifted, so transfer into it should fail and be
    detectable from label-free shift signals."""
    rng = np.random.default_rng(SEED)
    # signal strength falls as a market's features drift away from the "typical" market:
    # far-away markets (large shift) are where transferred knowledge fails
    shifts = [0.0, 0.3, -0.3, 0.9, 1.3, 1.7]
    vols = [0.010, 0.012, 0.011, 0.013, 0.016, 0.020]
    params = [dict(a=0.07 - 0.07 * abs(sh), vol=v, shift=sh) for sh, v in zip(shifts, vols)][:n_markets]
    g = rng.normal(0, 0.008, n_days)                      # global factor
    markets = []
    for k, p in enumerate(params):
        beta = rng.uniform(0.6, 1.4, n_assets)
        mkt = rng.normal(0, p["vol"] * 0.6, n_days)
        # mild regimes: in some 250-day blocks features drift further and the signal weakens
        regime = np.repeat(rng.random(n_days // 250 + 1) < 0.3, 250)[:n_days]
        shift_t = p["shift"] + 0.5 * regime
        a_t = p["a"] - 0.035 * regime
        x = rng.normal(shift_t[:, None], 1.0, (n_days, n_assets))  # observed signal feature
        noise = rng.normal(0, p["vol"], (n_days, n_assets))
        r = np.zeros((n_days, n_assets))
        r[1:] = (beta * g[1:, None] + mkt[1:, None]
                 + (a_t[:-1] * p["vol"])[:, None] * (x[:-1] - shift_t[:-1, None]) + noise[1:])
        # features at day t use information up to day t only
        ret5 = np.full_like(r, np.nan)
        vol20 = np.full_like(r, np.nan)
        for t in range(20, n_days):
            ret5[t] = r[t - 4 : t + 1].sum(0)
            vol20[t] = r[t - 19 : t + 1].std(0)
        y = np.full_like(r, np.nan)
        y[:-1] = r[1:]                                     # next-day return = target
        feats = np.stack([x, ret5, vol20], axis=-1)        # (days, assets, 3)
        markets.append(dict(k=k, feats=feats, y=y, r=r, params=p))
    print(f"  built {len(markets)} markets x {n_assets} assets x {n_days} days")
    METRICS["n_markets"] = len(markets)
    return markets


def flat(market: dict, t0: int, t1: int):
    X = market["feats"][t0:t1].reshape(-1, 3)
    y = market["y"][t0:t1].reshape(-1)
    ok = ~np.isnan(X).any(1) & ~np.isnan(y)
    return X[ok], y[ok]


# ----------------------------------------------------------------------------- 3. forecaster
def fit_ensemble(X, y, n_members: int = 5, seed: int = SEED):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    members = []
    for _ in range(n_members):
        idx = rng.integers(0, len(Xs), len(Xs))
        members.append(Ridge(alpha=1.0).fit(Xs[idx], y[idx]))
    return sc, members


def predict_ensemble(model, X):
    sc, members = model
    P = np.stack([m.predict(sc.transform(X)) for m in members])
    return P.mean(0), P.std(0)


@stage("3. Forecaster ensemble")
def test_forecaster(markets):
    X, y = flat(markets[0], 20, 1000)
    Xv, yv = flat(markets[0], 1000, 1499)
    model = fit_ensemble(X, y)
    mu, sd = predict_ensemble(model, Xv)
    ic = spearman_ic(mu, yv)
    print(f"  in-domain rank IC (market 0): {ic:.3f}   mean ensemble sd: {sd.mean():.2e}")
    METRICS["forecaster_ic_market0"] = round(ic, 4)


# ----------------------------------------------------------------------------- 4. episodes
def mmd_rbf(A, B, n: int = 400, seed: int = SEED) -> float:
    rng = np.random.default_rng(seed)
    A = A[rng.choice(len(A), min(n, len(A)), replace=False)]
    B = B[rng.choice(len(B), min(n, len(B)), replace=False)]
    Z = np.vstack([A, B])
    mu, sd = Z.mean(0), Z.std(0) + 1e-12
    A, B = (A - mu) / sd, (B - mu) / sd
    d = lambda P, Q: ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)  # noqa: E731
    h = np.median(d(A, B))
    k = lambda P, Q: np.exp(-d(P, Q) / h)  # noqa: E731
    return float(k(A, A).mean() + k(B, B).mean() - 2 * k(A, B).mean())


def domain_auc(A, B, n: int = 2000, seed: int = SEED) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(seed)
    A = A[rng.choice(len(A), min(n, len(A)), replace=False)]
    B = B[rng.choice(len(B), min(n, len(B)), replace=False)]
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    return float(cross_val_score(LogisticRegression(max_iter=500), X, y, cv=3, scoring="roc_auc").mean())


def portfolio_scores(market, model, t0, t1, top_frac=0.2):
    """Daily standardised loss scores s=(L-m_hat)/sigma_hat for a top-quintile long portfolio."""
    scores = []
    n_assets = market["feats"].shape[1]
    k = max(1, int(top_frac * n_assets))
    for t in range(t0, t1):
        X = market["feats"][t]
        if np.isnan(X).any() or np.isnan(market["y"][t]).any():
            continue
        mu, _ = predict_ensemble(model, X)
        sel = np.argsort(mu)[-k:]
        L = -market["y"][t, sel].mean()
        m_hat = -mu[sel].mean()
        sig_i = market["r"][max(0, t - 59) : t + 1, sel].std(0)
        rho = 0.3
        s_hat = np.sqrt((sig_i**2).mean() / k * (1 - rho) + rho * sig_i.mean() ** 2)
        scores.append((L - m_hat) / max(s_hat, 1e-8))
    return np.array(scores)


@stage("4. Pseudo-transfer episodes")
def build_episodes(markets, sources=(0, 1, 2, 3, 4), windows=(500, 750, 1000, 1250), win_len=250):
    eps = []
    for j in sources:
        for w0 in windows:
            others = [m for m in sources if m != j]
            Xtr = np.vstack([flat(markets[m], 20, w0 - EMBARGO)[0] for m in others])
            ytr = np.concatenate([flat(markets[m], 20, w0 - EMBARGO)[1] for m in others])
            Xin, yin = flat(markets[j], 20, w0 - EMBARGO)
            Xte, yte = flat(markets[j], w0, w0 + win_len)
            transfer = fit_ensemble(Xtr, ytr)
            indomain = fit_ensemble(Xin, yin)
            mu_tr, sd_tr = predict_ensemble(transfer, Xte)
            mu_in, _ = predict_ensemble(indomain, Xte)
            G = spearman_ic(mu_in, yte) - spearman_ic(mu_tr, yte)   # transfer degradation
            eps.append(dict(
                market=j, window=w0, G=G,
                mmd=mmd_rbf(Xtr, Xte), dom_auc=domain_auc(Xtr, Xte), ens_sd=float(sd_tr.mean()),
                scores=portfolio_scores(markets[j], transfer, w0, w0 + win_len),
            ))
    G = np.array([e["G"] for e in eps])
    thr = np.quantile(G, 2 / 3)
    for e in eps:
        e["Y"] = int(e["G"] > thr)
    print(f"  {len(eps)} episodes; failure threshold G > {thr:.3f}; failures: {sum(e['Y'] for e in eps)}")
    METRICS["n_episodes"] = len(eps)
    return eps


# ----------------------------------------------------------------------------- 5. TRE
@stage("5. Transfer Reliability Estimator (N1)")
def train_tre(eps):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    F = np.array([[e["mmd"], e["dom_auc"], e["ens_sd"]] for e in eps])
    Y = np.array([e["Y"] for e in eps])
    groups = np.array([e["market"] for e in eps])
    oof = np.full(len(eps), np.nan)
    for g in np.unique(groups):                       # leave-one-market-out
        tr, te = groups != g, groups == g
        if len(np.unique(Y[tr])) < 2:
            continue
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(max_iter=500).fit(sc.transform(F[tr]), Y[tr])
        oof[te] = clf.predict_proba(sc.transform(F[te]))[:, 1]
    ok = ~np.isnan(oof)
    auc_tre = roc_auc_score(Y[ok], oof[ok]) if len(np.unique(Y[ok])) == 2 else float("nan")
    single = {n: roc_auc_score(Y, F[:, i]) for i, n in enumerate(["mmd", "dom_auc", "ens_sd"])}
    print(f"  TRE leave-one-market-out AUROC: {auc_tre:.3f}")
    print("  (only 20 synthetic episodes: AUROC values are noisy here; this stage only tests the code)")
    for n, a in single.items():
        print(f"  single signal {n:8s} AUROC: {a:.3f}")
    METRICS["tre_auroc_synthetic"] = round(float(auc_tre), 4)
    for n, a in single.items():
        METRICS[f"single_auroc_{n}"] = round(float(a), 4)
    sc = StandardScaler().fit(F)
    final = LogisticRegression(max_iter=500).fit(sc.transform(F), Y)
    for e, p in zip(eps, final.predict_proba(sc.transform(F))[:, 1]):
        e["rho"] = float(p)
    return sc, final


# ----------------------------------------------------------------------------- 6. conformal
def weighted_quantile(x: np.ndarray, level: float) -> float:
    if level >= 1:
        return float("inf")
    if level <= 0:
        return float(np.min(x))
    return float(np.quantile(x, level, method="higher"))


def run_aci(pool: np.ndarray, target_scores: np.ndarray, gamma: float = 0.01, warm: int = 250):
    """Adaptive conformal inference; pool is gradually replaced by target scores."""
    a_t = ALPHA
    errs, qs = [], []
    seen: list[float] = []
    for s in target_scores:
        lam = max(0.0, 1 - len(seen) / warm)
        n_pool = int(round(lam * len(pool)))
        cal = np.concatenate([pool[:n_pool] if n_pool else np.array([]), np.array(seen)])
        q = weighted_quantile(cal, 1 - a_t) if len(cal) else float("inf")
        err = int(s > q)
        errs.append(err)
        qs.append(q)
        seen.append(s)
        a_t = a_t + gamma * (ALPHA - err)
    return np.array(errs), np.array(qs)


@stage("6. Cold-start conformal budget + ACI (N2)")
def test_conformal(markets, eps, tre, out_dir: Path):
    sc, clf = tre
    target = markets[5]
    sources = [0, 1, 2, 3, 4]
    t_start = 1000
    Xtr = np.vstack([flat(markets[m], 20, t_start - EMBARGO)[0] for m in sources])
    ytr = np.concatenate([flat(markets[m], 20, t_start - EMBARGO)[1] for m in sources])
    model = fit_ensemble(Xtr, ytr)
    Xte, _ = flat(target, t_start, t_start + 250)
    phi = np.array([[mmd_rbf(Xtr, Xte), domain_auc(Xtr, Xte), predict_ensemble(model, Xte)[1].mean()]])
    rho_T = float(clf.predict_proba(sc.transform(phi))[:, 1][0])
    print(f"  target market reliability score rho_T = {rho_T:.3f} (high = transfer likely to fail)")
    METRICS["target_rho"] = round(rho_T, 4)

    tgt_scores = portfolio_scores(target, model, t_start, 1499)

    # (a) in-sample calibration: source scores of the pooled model on its own training markets
    insample = np.concatenate([portfolio_scores(markets[m], model, 600, t_start - EMBARGO) for m in sources])
    # (b) transfer-residual pool, stratified by reliability (Mondrian, 2 strata)
    rhos = np.array([e["rho"] for e in eps])
    med = np.median(rhos)
    stratum = [e for e in eps if (e["rho"] > med) == (rho_T > med)]
    pool = np.concatenate([e["scores"] for e in stratum])
    np.random.default_rng(SEED).shuffle(pool)

    q_in = weighted_quantile(insample, 1 - ALPHA)
    q_pool = weighted_quantile(pool, 1 - ALPHA)
    err_in = (tgt_scores > q_in).astype(int)
    err_pool = (tgt_scores > q_pool).astype(int)
    err_aci, _ = run_aci(pool, tgt_scores)

    rows = {
        "in_sample_static": err_in,
        "transfer_pool_static": err_pool,
        "CSCRB_pool_plus_ACI": err_aci,
    }
    for name, e in rows.items():
        METRICS[f"BEF_first60_{name}"] = round(float(e[:60].mean()), 4)
        METRICS[f"BEF_all_{name}"] = round(float(e.mean()), 4)
        print(f"  {name:24s} BEF first 60 days: {e[:60].mean():.3f}   all: {e.mean():.3f}   (nominal {ALPHA})")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.2))
    for name, e in rows.items():
        roll = np.convolve(e, np.ones(40) / 40, mode="valid")
        ax.plot(roll, label=name)
    ax.axhline(ALPHA, ls="--", c="k", lw=1, label="nominal α")
    ax.set_xlabel("days in unseen target market")
    ax.set_ylabel("rolling exceedance rate (40d)")
    ax.set_title("Smoke test (synthetic data): budget exceedance")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / "figures" / f"smoke_bef.{ext}", dpi=150)
    plt.close(fig)
    return q_pool


# ----------------------------------------------------------------------------- 7. governance
@stage("7. Governance projection with cvxpy (N3)")
def test_governance(markets, q: float):
    import cvxpy as cp

    rng = np.random.default_rng(SEED)
    n = markets[0]["feats"].shape[1]
    t = 1200
    R = markets[0]["r"][t - 120 : t]
    Sigma = np.cov(R.T) + 1e-8 * np.eye(n)
    Lc = np.linalg.cholesky(Sigma)
    mu = R.mean(0)
    w_prop = np.zeros(n)
    w_prop[rng.choice(n, 6, replace=False)] = 1 / 6        # "proposer" puts everything in 6 stocks

    def project(B_max):
        w = cp.Variable(n)
        cash = cp.Variable()
        q_eff = q if np.isfinite(q) else 1e6
        cons = [cp.sum(w) + cash == 1, w >= 0, cash >= 0, w <= 0.10,
                -mu @ w + q_eff * cp.norm(Lc.T @ w, 2) <= B_max]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(w - w_prop)), cons)
        prob.solve()
        return prob.status, (w.value if w.value is not None else np.zeros(n)), cash.value

    for label, B in [("normal budget", 0.03), ("very tight budget", 0.002)]:
        status, w, cash = project(B)
        print(f"  {label:18s} solver={status:10s} equity={w.sum():.3f} cash={float(cash):.3f} max_w={w.max():.3f}")
        if status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"projection infeasible for {label}")
        if w.max() > 0.10 + 1e-4:
            raise RuntimeError("position cap violated")
        METRICS[f"governance_cash_{label.replace(' ', '_')}"] = round(float(cash), 4)


# ----------------------------------------------------------------------------- 8. RL
@stage("8. Tiny PPO run (torch + stable-baselines3)")
def test_rl():
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO

    class ToyPortfolio(gym.Env):
        """2 assets + cash; the agent should learn to hold the asset with positive drift."""

        def __init__(self):
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)
            self.action_space = gym.spaces.Box(-1, 1, (3,), np.float32)
            self.rng = np.random.default_rng(SEED)
            self.t = 0

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            return np.zeros(2, np.float32), {}

        def step(self, action):
            w = np.exp(action) / np.exp(action).sum()
            r = np.array([0.002, -0.001]) + self.rng.normal(0, 0.01, 2)
            reward = float(w[:2] @ r)
            self.t += 1
            return r.astype(np.float32), reward, self.t >= 64, False, {}

    torch.manual_seed(SEED)
    model = PPO("MlpPolicy", ToyPortfolio(), n_steps=256, batch_size=64, seed=SEED, verbose=0, device="cpu")
    model.learn(total_timesteps=2048)
    print(f"  PPO trained 2048 steps on CPU; torch {torch.__version__}; cuda available: {torch.cuda.is_available()}")
    METRICS["cuda_available"] = int(torch.cuda.is_available())


# ----------------------------------------------------------------------------- 9. other libs
@stage("9. Other libraries (lightgbm, statsmodels, arch, calendars)")
def test_other_libs(markets):
    import exchange_calendars as xcals
    import lightgbm as lgb
    from arch.bootstrap import StationaryBootstrap
    from statsmodels.tsa.arima.model import ARIMA

    X, y = flat(markets[0], 20, 800)
    lgb.LGBMRegressor(n_estimators=50, verbose=-1).fit(X, y)
    print("  lightgbm OK")
    ARIMA(markets[0]["r"][:500, 0], order=(1, 0, 0)).fit()
    print("  statsmodels ARIMA OK")
    bs = StationaryBootstrap(20, markets[0]["r"][:500, 0], seed=SEED)
    ci = bs.conf_int(np.mean, reps=200)
    print(f"  arch stationary bootstrap OK (CI of mean: {ci.ravel().round(5)})")
    for code in ["XNYS", "XTKS", "XBOM", "XLON", "XETR"]:
        cal = xcals.get_calendar(code)
        n = len(cal.sessions_in_range("2024-01-01", "2024-12-31"))
        print(f"  calendar {code}: {n} trading days in 2024")


# ----------------------------------------------------------------------------- 10. internet
@stage("10. Internet check for data downloads", required=False)
def test_internet():
    import yfinance as yf

    df = yf.download("^GSPC", period="5d", progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        raise RuntimeError("yfinance returned no data (internet or Yahoo issue)")
    print(f"  yfinance OK: downloaded {len(df)} rows of S&P 500 data")


# ----------------------------------------------------------------------------- report
def write_report(out_dir: Path, total_s: float):
    import csv

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in METRICS.items():
            w.writerow([k, v])
    n_fail = sum(r["status"] == "FAIL" for r in RESULTS)
    n_warn = sum(r["status"] == "WARN" for r in RESULTS)
    verdict = "ALL REQUIRED STAGES PASSED" if n_fail == 0 else f"{n_fail} REQUIRED STAGE(S) FAILED"
    lines = [
        "# Smoke test report",
        "",
        f"**Verdict:** {verdict} ({n_warn} warning(s)). Total time {total_s:.0f} s.",
        "",
        "Synthetic data only: numbers below test the code and are not research results.",
        "With only 20 synthetic episodes, AUROC and exceedance values are noisy by design.",
        "",
        "| Stage | Status | Seconds | Note |",
        "|---|---|---|---|",
    ] + [f"| {r['stage']} | {r['status']} | {r['seconds']} | {r['note'][:150]} |" for r in RESULTS] + [
        "",
        "## Key synthetic metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in METRICS.items()]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return n_fail


def main():
    out_dir = step_dir(STEP)
    tee = Tee(out_dir / "log.txt")
    sys.stdout = tee
    t0 = time.time()
    print("ReGuard-XM smoke test — expected runtime 2-6 minutes on CPU.")
    print(f"Results folder: {out_dir}")
    try:
        check_packages(out_dir)
        markets = make_markets()
        if markets is not None:
            test_forecaster(markets)
            eps = build_episodes(markets)
            tre = train_tre(eps) if eps else None
            q = test_conformal(markets, eps, tre, out_dir) if tre else None
            test_governance(markets, q if q is not None else 2.0)
        test_rl()
        if markets is not None:
            test_other_libs(markets)
        test_internet()
    finally:
        n_fail = write_report(out_dir, time.time() - t0)
        sys.stdout = tee.stdout
        tee.close()
    zpath = make_zip(out_dir, STEP)
    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED" if n_fail == 0 else f"SMOKE TEST: {n_fail} required stage(s) FAILED")
    print(f"Upload this file to Claude:\n  {zpath}")
    print("=" * 70)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
