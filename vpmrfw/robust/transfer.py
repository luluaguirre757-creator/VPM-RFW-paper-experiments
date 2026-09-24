from __future__ import annotations
import numpy as np
from vpmrfw.core.fp import fp_mode_terms
from vpmrfw.robust.certificate import geometry_bounds


def _assert_common_outer_geometry(task_a,task_b):
    if task_a.J!=task_b.J or task_a.R!=task_b.R:
        raise ValueError('task dimensions must match')
    if task_a.Ns!=task_b.Ns or task_a.lower_bounds!=task_b.lower_bounds or task_a.total_budget!=task_b.total_budget:
        raise ValueError('transfer theory requires the same (N_r, lower bounds, total budget) across tasks')


def _scaled_terms(task,j,r):
    """Represent F_j/scale as a product of mode terms by assigning 1/scale to mode 0."""
    d,Q=fp_mode_terms(task.scenarios[j][r])
    if r==0:
        d=d/task.scale; Q=Q/task.scale
    return d,Q


def _quadratic_budget_bound(Q: np.ndarray, Lbar: float) -> float:
    """Certified bound for |x^T Q x| on 0<=x<=1, ||x||_1<=Lbar.

    Two independent inequalities are available:
      |x^T Q x| <= Lbar^2 ||Q||_max,
      |x^T Q x| <= ||Q||_2 ||x||_2^2 <= Lbar ||Q||_F.
    Their minimum is therefore still a valid upper bound.  The Frobenius form
    is much tighter for the experiment Gram-difference matrices and costs only
    O(N^2), unlike an explicit spectral-norm decomposition.
    """
    Q=np.asarray(Q,float); Lbar=float(Lbar)
    if Lbar<=0 or Q.size==0:
        return 0.0
    entry=(Lbar**2)*float(np.max(np.abs(Q),initial=0.0))
    fro=Lbar*float(np.linalg.norm(Q,'fro'))
    return min(entry,fro)


def mode_variation(task_a,task_b,j,r,Lbar):
    da,Qa=_scaled_terms(task_a,j,r); db,Qb=_scaled_terms(task_b,j,r)
    Lbar=float(Lbar)
    return float(Lbar*np.max(np.abs(da-db))+_quadratic_budget_bound(Qa-Qb,Lbar))


def mode_sup(task,j,r,Lbar):
    d,Q=_scaled_terms(task,j,r); Lbar=float(Lbar)
    return float(Lbar*np.max(np.abs(d))+_quadratic_budget_bound(Q,Lbar))


def tensor_value_variation(task_a,task_b):
    _assert_common_outer_geometry(task_a,task_b)
    worst=0.0
    for j in range(task_a.J):
        ds=[mode_variation(task_a,task_b,j,r,Lbar) for r,Lbar in enumerate(task_a.bar_ells)]
        Ms=[max(mode_sup(task_a,j,r,Lbar),mode_sup(task_b,j,r,Lbar),1e-15)
            for r,Lbar in enumerate(task_a.bar_ells)]
        # Telescoping-product bound from the revised appendix.
        v=sum(ds[r]*float(np.prod([Ms[q] for q in range(task_a.R) if q!=r])) for r in range(task_a.R))
        worst=max(worst,float(v))
    return worst


def constraint_variation(task_a,task_b):
    _assert_common_outer_geometry(task_a,task_b)
    # Paper explicit bound: R_Y||dA|| + R_P||dB|| + ||db||; A=I for all experiment tasks.
    _,RPa,_=geometry_bounds(task_a); _,RPb,_=geometry_bounds(task_b); RP=max(RPa,RPb)
    return float(RP*np.linalg.norm(task_a.B-task_b.B,2)+np.linalg.norm(task_a.b-task_b.b))


def fvec_variation_bound(task_a,task_b):
    return float(np.sqrt(task_a.J)*tensor_value_variation(task_a,task_b))


def paper_transfer_bounds(cur,old,C_pair):
    _assert_common_outer_geometry(cur,old)
    dten=tensor_value_variation(cur,old); dG=constraint_variation(cur,old)
    # Same mu,y0,A in this experiment => Delta psi=Delta gradpsi=||A_s-A_i||=0.
    V=dten+C_pair*dG
    Omega=np.sqrt(cur.J)*dten+dG
    return {'delta_ten':dten,'delta_G':dG,'V_hat':float(V),'Omega_hat':float(Omega)}


def paper_stationarity_transfer_bound(cur,old,cert,item):
    """Conservative experiment-specialized bound for paper Eq. (17).

    The Jacobian-difference term is bounded by the sum of the two certified
    Jacobian norms.  This is intentionally conservative but globally valid on
    the shared polytope and therefore cannot create a false screen hit.
    """
    tr=paper_transfer_bounds(cur,old,max(cert.C_hat,item.C_hat))
    bcur=float(np.linalg.norm(cur.B,2)); bold=float(np.linalg.norm(old.B,2))
    mj_old=max(0.0,float(item.Lgz)-bold)
    same_factors=(cur.scale==old.scale and all(
        np.array_equal(cur.scenarios[j][r],old.scenarios[j][r])
        for j in range(cur.J) for r in range(cur.R)))
    delta_j=0.0 if same_factors else float(cert.M_J)+mj_old
    gamma=(delta_j+max(cert.C_hat,item.C_hat)*float(np.linalg.norm(cur.B-old.B,2))+
           (max(float(cert.M_J),mj_old)+max(bcur,bold))*cert.caps.bar_kappa_cross*tr['Omega_hat'])
    return float(gamma),{**tr,'Delta_J_hat':delta_j,'Gamma_hat':float(gamma)}


def active_set_kappa(task,x,z,active_tol=1e-7):
    """Diagnostic only; never promoted to theorem-level certificate automatically."""
    J=task.J; y=z[:J]; lam=z[J:]; g=task.G(x,y)
    act_c=np.where((np.abs(g)<=active_tol)|(np.abs(lam)>active_tol))[0]
    act_y=np.where(y<=active_tol)[0]; rows=[np.ones(J)]
    for k in act_y:
        e=np.zeros(J); e[k]=1; rows.append(e)
    for k in act_c:
        e=np.zeros(J); e[k]=1; rows.append(e)
    M=np.asarray(rows,float); W=task.mu*np.eye(J)
    K=np.block([[W,M.T],[M,np.zeros((M.shape[0],M.shape[0]))]])
    s=np.linalg.svd(K,compute_uv=False)
    if s[-1]<=1e-12*s[0]: return float('inf'),0.0
    inactive_c=[k for k in range(J) if k not in set(act_c)]; inactive_y=[k for k in range(J) if k not in set(act_y)]
    margins=[]
    if inactive_c: margins += (-g[inactive_c]).tolist()
    if inactive_y: margins += y[inactive_y].tolist()
    if len(act_c): margins += (-lam[act_c]).tolist()
    return float(1/s[-1]),float(max(0.0,min(margins)) if margins else 1.0)
