from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from vpmrfw.core.simplex import project_simplex

@dataclass
class InnerInfo:
    z:np.ndarray
    residual:float
    fo:int
    ppa:int
    certified_converged:bool
    distance_certificate:float
    natural_residual_initial:float
    distance_cert_initial:float
    epsilon_z_target:float
    hit_ppa_cap:bool=False
    hit_resolvent_cap:bool=False
    reached_ppa_limit:bool=False
    reached_resolvent_limit:bool=False
    termination_reason:str=''
    practical_converged:bool=False

    @property
    def converged(self):
        """Backward-compatible name: always means distance-certified convergence."""
        return self.certified_converged

    @property
    def distance_cert_ratio(self):
        return (self.distance_certificate/self.epsilon_z_target
                if self.epsilon_z_target > 0 else float('inf'))

def project_Z(z,J):
    y=project_simplex(np.asarray(z[:J],float),1.0)
    # Paper convention K=R_- in this experiment => dual K*=R_- under <lambda,k>>=0 convention.
    lam=np.minimum(np.asarray(z[J:],float),0.0)
    return np.concatenate([y,lam])

def natural_residual(task,x,z,tau=1.0):
    return float(np.linalg.norm(z-project_Z(z-tau*task.H(x,z),task.J))/tau)

def solve_inner_paper(task,x,z0,cert,d0_bound:float,oracle_guard=None):
    """Inexact-PPA realization used in the paper, with its exact ceiling schedule.

    K_hat and N_res_hat are computed from the same `d0_bound`, epsilon_z, chi and rho
    used by the history-cost formula. We allow *safe early termination* when the natural
    residual plus the declared residual-to-distance cap already certifies epsilon_z; this
    can only reduce the realized count relative to the theorem bound.
    """
    z=project_Z(np.asarray(z0,float),task.J); fo=0; ppa=0
    K,N,_=cert.inner_components(float(d0_bound)); alpha=cert.caps.alpha_ppa
    target_res=cert.eps_z/cert.kappa_R_hat
    nr0=natural_residual(task,x,z); dist0=cert.kappa_R_hat*nr0
    for k in range(K):
        nr=natural_residual(task,x,z)
        if nr<=target_res:
            return InnerInfo(z,nr,fo,ppa,True,cert.kappa_R_hat*nr,nr0,dist0,cert.eps_z,
                             termination_reason='certified_tolerance',practical_converged=True)
        w=z.copy()
        for _ in range(N):
            if oracle_guard is not None: oracle_guard('fo')
            F=task.H(x,w)+(w-z)/alpha
            w=project_Z(w-cert.eta_alpha*F,task.J); fo+=1
        z=w; ppa+=1
    nr=natural_residual(task,x,z); dist=cert.kappa_R_hat*nr
    ok=bool(dist<=cert.eps_z*(1+1e-10))
    return InnerInfo(z,nr,fo,ppa,ok,dist,nr0,dist0,cert.eps_z,
                     hit_ppa_cap=not ok,reached_ppa_limit=bool(ppa>=K),
                     termination_reason=('certified_tolerance' if ok else 'ppa_cap_before_certification'),
                     practical_converged=ok)


def solve_inner_practical(task,x,z0,cert,residual_tol:float=1e-5,max_ppa:int=100,max_resolvent:int=500):
    """Residual-stopped inexact PPA used only by the explicitly labelled practical experiments.

    This is the same PPA/projected-gradient realization as the paper but replaces the theorem's
    worst-case precomputed iteration ceilings by computable residual stopping.  It must not be
    reported as a theorem-certified execution.
    """
    z=project_Z(np.asarray(z0,float),task.J); fo=0; ppa=0
    alpha=cert.caps.alpha_ppa
    nr0=natural_residual(task,x,z); dist0=cert.kappa_R_hat*nr0
    hit_resolvent=False; reached_resolvent=False
    for _ in range(int(max_ppa)):
        nr=natural_residual(task,x,z)
        dist=cert.kappa_R_hat*nr
        if dist<=cert.eps_z*(1+1e-10) and not hit_resolvent:
            return InnerInfo(z,nr,fo,ppa,True,dist,nr0,dist0,cert.eps_z,
                             termination_reason='certified_tolerance',practical_converged=True)
        if nr<=residual_tol:
            return InnerInfo(z,nr,fo,ppa,False,dist,nr0,dist0,cert.eps_z,
                             hit_resolvent_cap=hit_resolvent,
                             termination_reason=('resolvent_cap_before_certification' if hit_resolvent
                                                 else 'practical_tolerance'),
                             practical_converged=True)
        w=z.copy()
        # Solve one resolvent VI until its natural fixed-point residual is small enough.
        target=max(residual_tol*0.25,1e-10)
        resolvent_converged=False
        resolvent_iterations=0
        for _ in range(int(max_resolvent)):
            resolvent_iterations+=1
            F=task.H(x,w)+(w-z)/alpha
            wn=project_Z(w-cert.eta_alpha*F,task.J); fo+=1
            # fixed-point residual for the resolvent map with the same projected-gradient step
            if np.linalg.norm(wn-w) <= target*max(1.0,cert.eta_alpha):
                w=wn; resolvent_converged=True; break
            w=wn
        reached_resolvent |= bool(resolvent_iterations>=int(max_resolvent))
        hit_resolvent |= not resolvent_converged
        z=w; ppa+=1
    nr=natural_residual(task,x,z)
    dist=cert.kappa_R_hat*nr
    practical_ok=bool(nr<=residual_tol)
    certified_ok=bool(dist<=cert.eps_z*(1+1e-10) and not hit_resolvent)
    reached=bool(ppa>=int(max_ppa))
    hit_ppa=bool(reached and not certified_ok)
    if certified_ok:
        reason='certified_tolerance'
    elif hit_resolvent:
        reason='resolvent_cap_before_certification'
    elif hit_ppa:
        reason='ppa_cap_before_certification'
    else:
        reason='practical_tolerance' if practical_ok else 'practical_budget_exhausted'
    return InnerInfo(z,nr,fo,ppa,certified_ok,dist,nr0,dist0,cert.eps_z,
                     hit_ppa_cap=hit_ppa,hit_resolvent_cap=hit_resolvent,
                     reached_ppa_limit=reached,reached_resolvent_limit=reached_resolvent,
                     termination_reason=reason,
                     practical_converged=practical_ok)
