"""ODME solvers specialised to the TrafficFlowBench corridor structure.

Every measured link is a screenline (paths o <= k < d), so the count operator
collapses to S (n_seg x n_paths) with per-segment multiplicity n_k:

    ||A f - c||^2  ==  sum_k n_k (S_k f - c_k)^2      (counts are identical within a segment)

All dual problems therefore live in n_seg (<= 100) dimensions and are solved by
(semismooth) Newton, which is exact and fast.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


# ----------------------------------------------------------------------------- exact fits (S f = c)

def kl_fit(S: np.ndarray, c: np.ndarray, b: np.ndarray, iters: int = 200, tol: float = 1e-10,
           return_dual: bool = False):
    """Max-entropy / KL projection: argmin sum f log(f/b) - f + b  s.t.  S f = c.

    Solution f = b * exp(S^T mu) (multiplicative, path-level Furness on screenlines).
    """
    b = np.maximum(b, 1e-12)
    mu = np.zeros(S.shape[0])
    for _ in range(iters):
        t = np.clip(S.T @ mu, -50, 50)
        f = b * np.exp(t)
        g = S @ f - c
        if np.max(np.abs(g) / np.maximum(c, 1e-9)) < tol:
            break
        H = (S * f) @ S.T + 1e-12 * np.eye(S.shape[0])
        step = np.linalg.solve(H, g)
        # backtracking on the convex dual  phi(mu) = sum b exp(S^T mu) - mu.c
        phi0 = f.sum() - mu @ c
        a = 1.0
        while a > 1e-8:
            mu_n = mu - a * step
            fn = b * np.exp(np.clip(S.T @ mu_n, -50, 50))
            if fn.sum() - mu_n @ c <= phi0 - 1e-4 * a * (g @ step):
                break
            a *= 0.5
        mu = mu_n
    f = b * np.exp(np.clip(S.T @ mu, -50, 50))
    return (f, mu) if return_dual else f


def l2_fit(S: np.ndarray, c: np.ndarray, b: np.ndarray, w: np.ndarray | None = None,
           iters: int = 200, tol: float = 1e-10, return_dual: bool = False):
    """argmin sum (f-b)^2 / (2 w)  s.t.  S f = c,  f >= 0   (w = 1: Euclidean projection;
    w = b: chi-square / relative projection).  f = max(0, b + w * S^T mu)."""
    w = np.ones_like(b) if w is None else np.maximum(w, 1e-12)
    mu = np.zeros(S.shape[0])

    def primal(m):
        return np.maximum(0.0, b + w * (S.T @ m))

    def dual(m):
        f = primal(m)
        return np.sum((f - b) ** 2 / (2 * w)) - (S.T @ m) @ f + m @ c

    for _ in range(iters):
        f = primal(mu)
        g = S @ f - c
        if np.max(np.abs(g) / np.maximum(c, 1e-9)) < tol:
            break
        act = (f > 0).astype(float)
        H = (S * (w * act)) @ S.T + 1e-9 * np.eye(S.shape[0])
        step = np.linalg.solve(H, g)
        # we MAXIMISE dual q(mu); gradient of q is  c - S f  = -g ; Newton step mu - step
        q0 = dual(mu)
        a = 1.0
        while a > 1e-10:
            mu_n = mu - a * step
            if dual(mu_n) >= q0 + 1e-4 * a * (g @ step):
                break
            a *= 0.5
        mu = mu_n
    f = primal(mu)
    return (f, mu) if return_dual else f


# ----------------------------------------------------------------------------- penalised fits

def ridge_nn(S: np.ndarray, nk: np.ndarray, c: np.ndarray, b: np.ndarray, lam: float,
             x0: np.ndarray | None = None) -> np.ndarray:
    """Official baseline objective on the collapsed operator:
       min sum_k nk (S_k f - c_k)^2 + lam ||f - b||^2,  f >= 0.
    Solved exactly through the dual (n_seg dims):  f = max(0, b + S^T mu / lam)."""
    # KKT: 2 S^T N (c - S f) = 2 lam (f - b) on the free set -> f = b + S^T mu/lam, mu = N(c - S f)
    mu = np.zeros(S.shape[0])
    for _ in range(300):
        f = np.maximum(0.0, b + (S.T @ mu) / lam)
        r = mu - nk * (c - S @ f)            # root of r(mu) = 0
        if np.max(np.abs(r)) < 1e-9 * max(1.0, np.max(np.abs(mu))):
            break
        act = (f > 0).astype(float)
        J = np.eye(S.shape[0]) + (nk[:, None] * (S * act) @ S.T) / lam
        step = np.linalg.solve(J, r)
        # damped Newton on ||r||
        n0 = r @ r
        a = 1.0
        while a > 1e-8:
            mu_n = mu - a * step
            fn = np.maximum(0.0, b + (S.T @ mu_n) / lam)
            rn = mu_n - nk * (c - S @ fn)
            if rn @ rn <= (1 - 1e-4 * a) * n0:
                break
            a *= 0.5
        mu = mu_n
    return np.maximum(0.0, b + (S.T @ mu) / lam)


def kl_pen(S: np.ndarray, nk: np.ndarray, c: np.ndarray, b: np.ndarray, rho: float) -> np.ndarray:
    """min  sum_k nk (S_k f - c_k)^2 / (2 c_k)  + rho * KL(f||b)   ->  partially fitted max-entropy."""
    b = np.maximum(b, 1e-12)
    w = nk / np.maximum(c, 1e-9)

    def fun(mu):
        t = np.clip(S.T @ mu, -50, 50)
        f = b * np.exp(t / rho)
        # dual of the problem (mu = w (c - S f))
        val = rho * f.sum() - mu @ c + 0.5 * np.sum(mu ** 2 / w)
        grad = S @ f - c + mu / w
        return val, grad

    res = minimize(fun, np.zeros(S.shape[0]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-10, "ftol": 1e-15})
    return b * np.exp(np.clip(S.T @ res.x, -50, 50) / rho)
