"""Ortiz-Jimenez et al., IEEE TSP 2019, dense-core Greedy-FP.

Paper-original: greedily build the removal set subject to a total retained
mode-index budget and per-mode identifiability (partition matroid).  The
implementation below uses exact incremental frame-potential marginals; it is
algebraically identical to recomputing every retained Gram submatrix at every
greedy deletion, but is much faster on KSC/Salinas.

The fixed-quota variant is retained only as a legacy adaptation; revised main
experiments use the paper total-budget form.
"""
from __future__ import annotations
import numpy as np
from vpmrfw.core.fp import row_gram_sq


def _validate_budget(factors, total_retained, alpha):
    R=len(factors); Ns=[int(U.shape[0]) for U in factors]; Ks=[int(U.shape[1]) for U in factors]
    if alpha is None: alpha=[0]*R
    if len(alpha)!=R: raise ValueError("alpha must have one entry per mode")
    lower=[int(k+a) for k,a in zip(Ks,alpha)]
    if any(lo<0 or lo>N for lo,N in zip(lower,Ns)):
        raise ValueError("invalid per-mode lower bound")
    total_retained=int(total_retained)
    if total_retained < sum(lower): raise ValueError("total budget violates identifiability/slack")
    if total_retained > sum(Ns): raise ValueError("total budget exceeds available mode indices")
    return Ns,lower,total_retained


def greedy_fp_original(factors, total_retained:int, alpha=None):
    """Paper total-budget dense-core Greedy-FP with exact incremental marginals.

    At every deletion, this selects exactly the same candidate as the direct
    submatrix-recomputation rule.  For mode r and retained set I_r,

        FP_r(I_r) = sum_{i,j in I_r} H_r[i,j],
        Delta_r(t) = 2 sum_{j in I_r} H_r[t,j] - H_r[t,t].

    Removing t changes only FP_r, so all candidate objectives can be evaluated
    from maintained row sums in O(sum_r N_r) per deletion instead of rebuilding
    O(N_r^2) submatrices.
    """
    Ns,lower,total_retained=_validate_budget(factors,total_retained,alpha)
    R=len(factors); Hs=[row_gram_sq(U) for U in factors]
    active=[np.ones(N,dtype=bool) for N in Ns]
    counts=np.array(Ns,dtype=int)
    # H is symmetric and nonnegative. row_sum[r][i] is sum over currently active j.
    row_sum=[H.sum(axis=1).astype(float,copy=True) for H in Hs]
    fps=np.array([float(H.sum()) for H in Hs],dtype=float)

    n_remove=int(sum(Ns)-total_retained)
    for _ in range(n_remove):
        best=None
        total=float(np.prod(fps))
        for r,H in enumerate(Hs):
            if counts[r] <= lower[r]:
                continue
            idx=np.flatnonzero(active[r])
            dec=2.0*row_sum[r][idx]-np.diag(H)[idx]
            new_fr=fps[r]-dec
            if fps[r] > 0.0:
                vals=(total/fps[r])*new_fr
            else:
                # Degenerate zero-FP mode: evaluate the product directly to keep
                # deterministic exact semantics without division by zero.
                other=float(np.prod(np.delete(fps,r))) if R>1 else 1.0
                vals=other*new_fr
            k=int(np.argmin(vals)); t=int(idx[k]); val=float(vals[k])
            cand=(val,r,t)
            if best is None or cand[0] < best[0]-1e-15 or (abs(cand[0]-best[0])<=1e-15 and cand[1:]<best[1:]):
                best=cand
        if best is None: raise RuntimeError("no feasible removal")
        _,r,t=best; H=Hs[r]
        # Compute the exact decrement before mutating the active set/row sums.
        dec_t=float(2.0*row_sum[r][t]-H[t,t])
        active[r][t]=False; counts[r]-=1; fps[r]-=dec_t
        # Removing t deletes H[i,t] from every surviving active row sum.
        idx=np.flatnonzero(active[r])
        row_sum[r][idx]-=H[idx,t]
        row_sum[r][t]=0.0
        # Roundoff can produce a tiny negative FP although the exact quantity is >=0.
        if fps[r] < 0.0 and fps[r] > -1e-10*max(1.0,abs(dec_t)):
            fps[r]=0.0
    return [np.flatnonzero(a).astype(int) for a in active]


def greedy_fp_fixed_quotas(factors, quotas):
    """Same exact marginal rule, but fixed L_r. This is NOT the published feasible set."""
    quotas=[int(q) for q in quotas]
    if len(quotas)!=len(factors): raise ValueError("quotas must have one entry per mode")
    if any(q<0 or q>U.shape[0] for q,U in zip(quotas,factors)): raise ValueError("invalid quota")
    # A fixed-quota run is equivalent to the same removal rule with mode-specific
    # deletion floors.  Keep a dedicated implementation for backwards compatibility.
    R=len(factors); Hs=[row_gram_sq(U) for U in factors]
    active=[np.ones(U.shape[0],dtype=bool) for U in factors]
    counts=np.array([U.shape[0] for U in factors],dtype=int)
    row_sum=[H.sum(axis=1).astype(float,copy=True) for H in Hs]
    fps=np.array([float(H.sum()) for H in Hs],dtype=float)
    while np.any(counts>np.asarray(quotas,dtype=int)):
        total=float(np.prod(fps)); best=None
        for r,H in enumerate(Hs):
            if counts[r] <= quotas[r]: continue
            idx=np.flatnonzero(active[r]); dec=2.0*row_sum[r][idx]-np.diag(H)[idx]
            new_fr=fps[r]-dec
            vals=(total/fps[r])*new_fr if fps[r]>0 else (float(np.prod(np.delete(fps,r))) if R>1 else 1.0)*new_fr
            k=int(np.argmin(vals)); cand=(float(vals[k]),r,int(idx[k]))
            if best is None or cand[0] < best[0]-1e-15 or (abs(cand[0]-best[0])<=1e-15 and cand[1:]<best[1:]): best=cand
        if best is None: raise RuntimeError("no feasible removal")
        _,r,t=best; H=Hs[r]; dec_t=float(2.0*row_sum[r][t]-H[t,t])
        active[r][t]=False; counts[r]-=1; fps[r]-=dec_t
        idx=np.flatnonzero(active[r]); row_sum[r][idx]-=H[idx,t]; row_sum[r][t]=0.0
        if fps[r] < 0.0 and fps[r] > -1e-10*max(1.0,abs(dec_t)): fps[r]=0.0
    return [np.flatnonzero(a).astype(int) for a in active]
