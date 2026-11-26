"""Policy-agnostic governance layer: Euclidean projection of a proposed portfolio onto the admissible set

    C_t = { w >= 0, 1'w <= 1 (rest cash), w <= w_max, ||w - w_prev||_1 <= tau,
            -w'mu + q * ||chol(Sigma)' w||_2 <= B_max }        (second-order-cone budget; Prop. 3)

Uses cvxpy (Clarabel/ECOS/SCS) when installed, otherwise SciPy SLSQP on the same convex problem.
All-cash (w = 0) is always feasible; if the turnover cap makes the set empty it is dropped (documented
relaxation order) and risk constraints are never relaxed. q = +inf means "no admissible risk" -> cash.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _chol(Sigma: np.ndarray) -> np.ndarray:
    S = 0.5 * (Sigma + Sigma.T) + 1e-12 * np.eye(len(Sigma))
    return np.linalg.cholesky(S)


def _budget(w, mu, L, q):
    return -(w @ mu) + q * np.linalg.norm(L.T @ w)


def project(w_prop, w_prev, mu, Sigma, q, avail, w_max, tau, b_max):
    """Returns (w, info). Works on the available assets only; unavailable assets get 0."""
    n = len(w_prop)
    info = {"status": "ok", "relaxed_turnover": False, "modified": False, "cash_forced": False}
    if not np.isfinite(q):
        info.update(status="cash_q_inf", modified=bool(np.any(w_prop > 0)), cash_forced=True)
        return np.zeros(n), info
    ok = avail & np.isfinite(mu)
    if ok.sum() == 0:
        info.update(status="cash_no_assets", cash_forced=True, modified=bool(np.any(w_prop > 0)))
        return np.zeros(n), info
    x0 = np.where(ok, w_prop, 0.0)
    m, S, prev = mu[ok], Sigma[np.ix_(ok, ok)], w_prev[ok]
    L = _chol(S)
    a = x0[ok]
    feasible = ((a >= -1e-12).all() and a.sum() <= 1 + 1e-9 and (a <= w_max + 1e-9).all()
                and np.abs(a - prev).sum() <= tau + 1e-9 and _budget(a, m, L, q) <= b_max + 1e-12)
    if feasible and np.allclose(x0, w_prop):
        return w_prop.copy(), info
    for use_tau in (True, False):
        x = _solve(a, prev, m, L, q, w_max, tau if use_tau else None, b_max)
        if x is not None:
            info["relaxed_turnover"] = not use_tau
            w = np.zeros(n)
            w[ok] = np.clip(x, 0, w_max)
            info["modified"] = bool(np.abs(w - w_prop).sum() > 1e-6)
            return w, info
    info.update(status="cash_infeasible", cash_forced=True, modified=True)
    return np.zeros(n), info


def _solve(a, prev, m, L, q, w_max, tau, b_max):
    k = len(a)
    try:
        import cvxpy as cp
        x = cp.Variable(k)
        cons = [x >= 0, cp.sum(x) <= 1, x <= w_max, -(m @ x) + q * cp.norm(L.T @ x, 2) <= b_max]
        if tau is not None:
            cons.append(cp.norm1(x - prev) <= tau)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - a)), cons)
        for solver in ("CLARABEL", "ECOS", "SCS"):
            try:
                prob.solve(solver=solver)
                if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
                    return np.asarray(x.value).ravel()
            except Exception:  # noqa: BLE001
                continue
        return None
    except ImportError:
        pass
    # SciPy fallback (turnover via split variables u >= |x - prev|)
    if tau is None:
        z0 = np.clip(a, 0, w_max) * 0.5
        cons = [{"type": "ineq", "fun": lambda x: 1 - x.sum()},
                {"type": "ineq", "fun": lambda x: b_max - _budget(x, m, L, q)}]
        res = minimize(lambda x: np.sum((x - a) ** 2), z0, jac=lambda x: 2 * (x - a), method="SLSQP",
                       bounds=[(0, w_max)] * k, constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
        x = res.x
    else:
        z0 = np.r_[np.clip(prev, 0, w_max), np.zeros(k)]
        cons = [{"type": "ineq", "fun": lambda z: 1 - z[:k].sum()},
                {"type": "ineq", "fun": lambda z: b_max - _budget(z[:k], m, L, q)},
                {"type": "ineq", "fun": lambda z: tau - z[k:].sum()},
                {"type": "ineq", "fun": lambda z: z[k:] - (z[:k] - prev)},
                {"type": "ineq", "fun": lambda z: z[k:] + (z[:k] - prev)}]
        res = minimize(lambda z: np.sum((z[:k] - a) ** 2), z0, method="SLSQP",
                       bounds=[(0, w_max)] * k + [(0, None)] * k, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12})
        x = res.x[:k]
    feas = (x >= -1e-7).all() and x.sum() <= 1 + 1e-6 and _budget(x, m, L, q) <= b_max + 1e-6 and \
        (tau is None or np.abs(x - prev).sum() <= tau + 1e-6)
    return x if feas else None
