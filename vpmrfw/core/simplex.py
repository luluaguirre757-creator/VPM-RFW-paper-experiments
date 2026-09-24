from __future__ import annotations
from typing import Sequence
import numpy as np


def project_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Euclidean projection onto {x >= 0, sum(x)=z} (Duchi et al. threshold form)."""
    v=np.asarray(v,dtype=float)
    if v.ndim != 1:
        raise ValueError("v must be 1-D")
    if z <= 0:
        return np.zeros_like(v)
    u=np.sort(v)[::-1]
    cssv=np.cumsum(u)-z
    j=np.arange(1,len(v)+1,dtype=float)
    rho=np.nonzero(u-cssv/j>0)[0]
    if rho.size==0:
        return np.full_like(v,z/len(v))
    r=int(rho[-1]); theta=cssv[r]/(r+1.0)
    return np.maximum(v-theta,0.0)


def project_capped_simplex(v: np.ndarray, total: float, lo=0.0, hi=1.0, tol=1e-12) -> np.ndarray:
    """Projection onto lo<=x<=hi, sum x=total via scalar bisection."""
    v=np.asarray(v,dtype=float); n=v.size
    if not (n*lo-tol <= total <= n*hi+tol):
        raise ValueError("infeasible capped simplex")
    a=float(v.min()-hi-1); b=float(v.max()-lo+1)
    for _ in range(120):
        lam=(a+b)/2; x=np.clip(v-lam,lo,hi)
        if x.sum()>total:
            a=lam
        else:
            b=lam
    return np.clip(v-(a+b)/2,lo,hi)


def lmo_capped_simplex(grad: np.ndarray, total: int) -> np.ndarray:
    """Exact LMO for {0<=x<=1, sum x=L}: ones at L smallest gradient entries."""
    grad=np.asarray(grad,dtype=float); L=int(total)
    if L<0 or L>grad.size:
        raise ValueError("invalid total")
    out=np.zeros_like(grad)
    if L==0:
        return out
    idx=np.argsort(grad,kind="stable")[:L]
    out[idx]=1.0
    return out


def top_l_round(x: np.ndarray, L: int) -> np.ndarray:
    x=np.asarray(x); L=int(L)
    if L<0 or L>x.size:
        raise ValueError("invalid L")
    if L==0:
        return np.array([],dtype=int)
    idx=np.argsort(-x,kind="stable")[:L]
    return np.sort(idx.astype(int))


def validate_total_budget(Ns: Sequence[int], lower: Sequence[int], total: int) -> tuple[list[int],list[int],int]:
    """Validate the paper's total mode-index budget constraints.

    Feasible binary/continuous designs satisfy, for each mode r,
      0 <= x_r <= 1,  1^T x_r >= lower[r],  and  sum_r 1^T x_r = total.
    """
    Ns=[int(n) for n in Ns]; lower=[int(k) for k in lower]; total=int(total)
    if len(Ns)==0 or len(Ns)!=len(lower):
        raise ValueError("Ns/lower length mismatch")
    if any(n<=0 for n in Ns):
        raise ValueError("all mode sizes must be positive")
    if any(k<0 or k>n for k,n in zip(lower,Ns)):
        raise ValueError("invalid per-mode lower bound")
    if total<sum(lower) or total>sum(Ns):
        raise ValueError("total budget is infeasible for the per-mode lower bounds")
    return Ns,lower,total


def mode_budget_bounds(Ns: Sequence[int], lower: Sequence[int], total: int) -> tuple[list[int],list[int]]:
    """Exact projection limits (underline ell_r, bar ell_r) from the paper."""
    Ns,lower,total=validate_total_budget(Ns,lower,total)
    lo=[]; hi=[]
    for r in range(len(Ns)):
        lo.append(max(lower[r], total-sum(Ns[q] for q in range(len(Ns)) if q!=r)))
        hi.append(min(Ns[r], total-sum(lower[q] for q in range(len(Ns)) if q!=r)))
    return lo,hi


def balanced_total_budget_point(Ns: Sequence[int], lower: Sequence[int], total: int) -> np.ndarray:
    """A deterministic feasible fractional cold point for the total-budget polytope.

    The compulsory lower-bound mass is assigned first. Remaining mass is distributed
    proportionally to each mode's residual capacity N_r-lower_r, then uniformly inside
    each mode. This point is used only as a neutral cold initialization/reference scale;
    the FW LMOs are free to revise the allocation immediately.
    """
    Ns,lower,total=validate_total_budget(Ns,lower,total)
    residual=float(total-sum(lower)); cap=np.asarray([n-k for n,k in zip(Ns,lower)],float)
    if residual<=0:
        ell=np.asarray(lower,float)
    elif cap.sum()<=0:
        ell=np.asarray(lower,float)
    else:
        ell=np.asarray(lower,float)+residual*cap/cap.sum()
    # Numerical normalization while preserving bounds.
    delta=float(total-ell.sum())
    if abs(delta)>1e-12:
        for r in np.argsort(-(cap-(ell-np.asarray(lower,float)))):
            room=(Ns[r]-ell[r]) if delta>0 else (ell[r]-lower[r])
            step=np.sign(delta)*min(abs(delta),max(0.0,float(room)))
            ell[r]+=step; delta-=step
            if abs(delta)<=1e-12:
                break
    blocks=[np.full(n,float(e)/n,dtype=float) for n,e in zip(Ns,ell)]
    x=np.concatenate(blocks)
    if not np.isclose(x.sum(),total,atol=1e-9):
        raise RuntimeError("failed to construct total-budget cold point")
    return x


def _split_blocks(v: np.ndarray, Ns: Sequence[int]) -> list[np.ndarray]:
    v=np.asarray(v,dtype=float); Ns=[int(n) for n in Ns]
    if v.ndim!=1 or v.size!=sum(Ns):
        raise ValueError("vector size does not match mode sizes")
    out=[]; k=0
    for n in Ns:
        out.append(v[k:k+n]); k+=n
    return out


def constrained_binary_selection(scores: Sequence[np.ndarray], total: int, lower: Sequence[int], *, largest: bool=False) -> list[np.ndarray]:
    """Exact total-budget binary selection used by the LMO and rounding.

    For minimization (largest=False), select lower[r] smallest scores in each mode,
    then the globally smallest remaining ``total-sum(lower)`` scores. For nearest-
    vertex rounding (largest=True), apply the same rule to -x, equivalently selecting
    the lower[r] largest x entries and then the globally largest remaining entries.
    Stable tie breaking by (score, mode, index) makes runs deterministic.
    """
    scores=[np.asarray(s,dtype=float).reshape(-1) for s in scores]
    Ns=[len(s) for s in scores]; Ns,lower,total=validate_total_budget(Ns,lower,total)
    selected=[]
    for s,k in zip(scores,lower):
        order=np.argsort(-s if largest else s,kind="stable")
        selected.append(set(map(int,order[:k])))
    remaining=total-sum(lower)
    pool=[]
    for r,s in enumerate(scores):
        for i,val in enumerate(s):
            if i in selected[r]:
                continue
            key=-float(val) if largest else float(val)
            pool.append((key,r,int(i)))
    pool.sort(key=lambda q:(q[0],q[1],q[2]))
    for _,r,i in pool[:remaining]:
        selected[r].add(i)
    out=[np.asarray(sorted(v),dtype=int) for v in selected]
    if sum(len(v) for v in out)!=total or any(len(v)<k for v,k in zip(out,lower)):
        raise RuntimeError("internal total-budget selection error")
    return out


def lmo_total_budget(grad: np.ndarray, Ns: Sequence[int], lower: Sequence[int], total: int) -> np.ndarray:
    """Exact LMO for the paper's total-budget polytope, returned as a binary vector."""
    blocks=_split_blocks(np.asarray(grad,float),Ns)
    sets=constrained_binary_selection(blocks,total,lower,largest=False)
    out=np.zeros(sum(map(int,Ns)),dtype=float); k=0
    for n,idx in zip(Ns,sets):
        out[k+idx]=1.0; k+=int(n)
    return out


def round_total_budget(x: np.ndarray, Ns: Sequence[int], lower: Sequence[int], total: int) -> list[np.ndarray]:
    """Nearest feasible binary total-budget design R_L(x), returned as local mode indices."""
    blocks=_split_blocks(np.asarray(x,float),Ns)
    return constrained_binary_selection(blocks,total,lower,largest=True)


def allocation_from_vector(x: np.ndarray, Ns: Sequence[int]) -> np.ndarray:
    """Fractional mode allocation ell_r(x)=1^T x_r."""
    return np.asarray([float(b.sum()) for b in _split_blocks(np.asarray(x,float),Ns)],dtype=float)


def allocation_from_sets(sets: Sequence[np.ndarray]) -> np.ndarray:
    return np.asarray([len(np.asarray(s)) for s in sets],dtype=int)
