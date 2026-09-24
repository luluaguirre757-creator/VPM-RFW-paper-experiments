from __future__ import annotations
import numpy as np


def row_gram_sq(U: np.ndarray) -> np.ndarray:
    """H_ij=(u_i^T u_j)^2 used by FP set functions."""
    G = U @ U.T
    return G * G


def fp_discrete(U: np.ndarray, idx) -> float:
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return 0.0
    H = row_gram_sq(U[idx])
    return float(H.sum())


def fp_mode_terms(U: np.ndarray):
    """Paper notation for multilinear FP: d_i=||u_i||^4, Q_ii=0,Q_ik=(u_i^T u_k)^2."""
    H = row_gram_sq(U)
    d = np.diag(H).copy()
    Q = H.copy()
    np.fill_diagonal(Q, 0.0)
    return d, Q


def fp_multilinear(U: np.ndarray, x: np.ndarray) -> float:
    d, Q = fp_mode_terms(U)
    return float(d @ x + x @ Q @ x)


def fp_multilinear_grad(U: np.ndarray, x: np.ndarray) -> np.ndarray:
    d, Q = fp_mode_terms(U)
    return d + 2.0 * (Q @ x)


def tensor_fp_multilinear(factors: list[np.ndarray], xs: list[np.ndarray]) -> float:
    vals = [fp_multilinear(U, x) for U, x in zip(factors, xs)]
    return float(np.prod(vals))


def tensor_fp_grad(factors: list[np.ndarray], xs: list[np.ndarray]) -> list[np.ndarray]:
    vals = np.asarray([fp_multilinear(U, x) for U, x in zip(factors, xs)], dtype=float)
    out=[]
    for r,(U,x) in enumerate(zip(factors,xs)):
        other = float(np.prod(np.delete(vals,r))) if len(vals)>1 else 1.0
        out.append(other * fp_multilinear_grad(U,x))
    return out


def tensor_fp_discrete(factors: list[np.ndarray], sets: list[np.ndarray]) -> float:
    return float(np.prod([fp_discrete(U,idx) for U,idx in zip(factors,sets)]))


def finite_difference_grad(U, x, eps=1e-6):
    g=np.zeros_like(x,dtype=float)
    for i in range(len(x)):
        xp=x.copy(); xm=x.copy(); xp[i]+=eps; xm[i]-=eps
        g[i]=(fp_multilinear(U,xp)-fp_multilinear(U,xm))/(2*eps)
    return g
