"""Fast Frank-Wolfe (FFW) of Li, Liang, Zhou & Xie, Signal Processing 239 (2026) 110333.

This module follows the **final journal paper**, Algorithm 1 (vector sampling) and
Algorithm 2 (tensor sampling), DOI 10.1016/j.sigpro.2025.110333.

For one mode ``r`` with factor matrix Psi_r in R^{N_r x K_r}, define

    m_ij^(r) = (p_i^(r))^T p_j^(r),
    f^(r)    = sum_{i,j} (m_ij^(r))^2 = FP(Psi_r),

where p_i^(r) denotes column i of Psi_r.  The derivative of the mode-wise
multilinear extension at x^(r)=0.5*1 is

    g_n^(r) = sum_{i,j} m_ij^(r) p_ni^(r) p_nj^(r).

The **paper-original tensor FFW score** is not merely ``g_n/f``.  Algorithm 2
uses the explicit cross-mode balancing weight

    w_r = N_r / f^(r),
    d_n^(r) = w_r * g_n^(r).

Hence (1/N_r) * sum_n d_n^(r) = 1 for every mode.  The factor N_r is essential
when N_r differs across modes; dropping it changes cross-mode ordering.  The
paper validates this weighting against FFW-A (w_r=1/f_r), FFW-B (w_r=1), and
random weights.

The journal multilinear extension contains the Bernoulli diagonal-correction
term sum_n (x_n-x_n^2)(p_ni p_nj)^2.  At x=0.5*1 its derivative contribution
from that correction is zero, yielding the closed-form derivative above (Eq. 6).

``ffw_original_total_budget`` implements final-journal Algorithm 2 with the
paper feasible set: total domain-sensor budget sum_r L_r=L and lower bounds
L_r>=K_r. This is also the revised VPM-RFW structured feasible family.
``ffw_fixed_quotas`` is retained only as a legacy compatibility helper; it is
not used in the revised total-budget main experiments.
"""
from __future__ import annotations
import numpy as np


def ffw_mode_partial_derivative(U: np.ndarray) -> np.ndarray:
    """Eq. (6): d/dx_n \tilde F_r(x) at x=0.5*1, before tensor-mode weighting."""
    U = np.asarray(U, dtype=float)
    if U.ndim != 2:
        raise ValueError("FFW factor matrix must be 2-D")
    M = U.T @ U
    return np.einsum("ni,ij,nj->n", U, M, U, optimize=True)


def ffw_mode_frame_potential(U: np.ndarray) -> float:
    """Full-set mode frame potential f^(r)=sum_ij m_ij^2."""
    U = np.asarray(U, dtype=float)
    M = U.T @ U
    f = float(np.sum(M * M))
    if not np.isfinite(f) or f <= 0.0:
        raise ValueError("FFW requires a nonzero finite mode frame potential")
    return f


def ffw_mode_scores(U: np.ndarray) -> np.ndarray:
    """Final journal Algorithm 2 tensor score d_n^(r)=N_r/f^(r) * Eq.-(6) derivative."""
    U = np.asarray(U, dtype=float)
    n = U.shape[0]
    f = ffw_mode_frame_potential(U)
    return (float(n) / f) * ffw_mode_partial_derivative(U)


def ffw_mode_scores_a(U: np.ndarray) -> np.ndarray:
    """Paper ablation FFW-A: w_r=1/f^(r) (i.e. omit the N_r balancing factor)."""
    U = np.asarray(U, dtype=float)
    return ffw_mode_partial_derivative(U) / ffw_mode_frame_potential(U)


def ffw_mode_scores_b(U: np.ndarray) -> np.ndarray:
    """Paper ablation FFW-B: no cross-mode weight, w_r=1."""
    return ffw_mode_partial_derivative(np.asarray(U, dtype=float))


def _constrained_smallest(scores: list[np.ndarray], total: int, lower: list[int]) -> list[np.ndarray]:
    """Minimize additive scores subject to total count and per-mode lower bounds.

    Because all decisions are binary with unit cardinality cost, the exact solution
    is: take the `lower[r]` smallest scores in every mode, then fill the remaining
    budget with the globally smallest unselected scores.
    """
    if len(scores) != len(lower):
        raise ValueError("scores/lower length mismatch")
    if total < sum(lower):
        raise ValueError("total budget violates L_r >= K_r")
    if total > sum(len(s) for s in scores):
        raise ValueError("budget exceeds candidates")
    selected = []
    for s, lb in zip(scores, lower):
        lb = int(lb)
        if lb < 0 or lb > len(s):
            raise ValueError("invalid per-mode lower bound")
        selected.append(set(np.argsort(s, kind="stable")[:lb].tolist()))
    remaining = int(total) - sum(int(v) for v in lower)
    pool = []
    for r, s in enumerate(scores):
        for i, val in enumerate(s):
            if i not in selected[r]:
                pool.append((float(val), r, i))
    pool.sort(key=lambda q: (q[0], q[1], q[2]))
    for _, r, i in pool[:remaining]:
        selected[r].add(i)
    return [np.asarray(sorted(v), dtype=int) for v in selected]


def ffw_original_total_budget(
    factors: list[np.ndarray], total_retained: int, lower_bounds: list[int] | None = None
) -> list[np.ndarray]:
    """Final-journal Algorithm-2 scores under a total-budget feasible family.

    With ``lower_bounds=None`` this is exactly the paper-original constraint
    ``L_r >= K_r``.  Experiments may pass a stricter common operational floor
    (still >= K_r) to keep all structured methods on the same numerically stable
    subset of the total-budget polytope; the FFW scoring rule itself is unchanged.
    """
    scores = [ffw_mode_scores(U) for U in factors]
    lower = [int(U.shape[1]) for U in factors] if lower_bounds is None else list(map(int,lower_bounds))
    if len(lower) != len(factors) or any(lb < U.shape[1] for lb,U in zip(lower,factors)):
        raise ValueError('FFW lower bounds must match modes and satisfy L_r >= K_r')
    return _constrained_smallest(scores, int(total_retained), lower)


def ffw_original_total_budget_variant(
    factors: list[np.ndarray], total_retained: int, variant: str, rng: np.random.Generator | None = None
) -> list[np.ndarray]:
    """FFW weighting ablations from the final paper's Section 7.1.

    variant:
      - ``paper``: w_r=N_r/f_r (FFW)
      - ``a``:     w_r=1/f_r (FFW-A)
      - ``b``:     w_r=1     (FFW-B)
      - ``random``: one independent Uniform[0,1] mode weight per mode (FFW-Random)
    """
    v = str(variant).lower()
    if v == "paper":
        scores = [ffw_mode_scores(U) for U in factors]
    elif v == "a":
        scores = [ffw_mode_scores_a(U) for U in factors]
    elif v == "b":
        scores = [ffw_mode_scores_b(U) for U in factors]
    elif v == "random":
        if rng is None:
            raise ValueError("rng is required for FFW-Random")
        weights = rng.uniform(0.0, 1.0, size=len(factors))
        scores = [float(w) * ffw_mode_partial_derivative(U) for w, U in zip(weights, factors)]
    else:
        raise ValueError("variant must be one of: paper, a, b, random")
    lower = [int(U.shape[1]) for U in factors]
    return _constrained_smallest(scores, int(total_retained), lower)


def ffw_fixed_quotas(factors: list[np.ndarray], quotas: list[int]) -> list[np.ndarray]:
    """Quota-adapted FFW using final-journal row score; NOT original Algorithm 2 feasible set."""
    if len(factors) != len(quotas):
        raise ValueError("factors/quotas length mismatch")
    out = []
    for U, L in zip(factors, quotas):
        s = ffw_mode_scores(U)
        L = int(L)
        if L < U.shape[1]:
            raise ValueError("quota violates identifiability L_r>=K_r")
        if L > U.shape[0]:
            raise ValueError("quota exceeds mode size")
        out.append(np.sort(np.argsort(s, kind="stable")[:L]))
    return out
