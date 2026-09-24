from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from vpmrfw.core.fp import tensor_fp_multilinear, tensor_fp_grad
from vpmrfw.core.simplex import (
    validate_total_budget, mode_budget_bounds, balanced_total_budget_point,
    allocation_from_vector,
)

@dataclass
class RobustTask:
    scenarios:list[list[np.ndarray]]
    total_budget:int
    lower_bounds:list[int]|None=None
    mu:float=0.1
    u0:float=0.45
    eta_c:float=0.20
    sensitivities:list[list[np.ndarray]]|None=None
    scale:float=1.0

    def __post_init__(self):
        if not self.scenarios or not self.scenarios[0]:
            raise ValueError('scenarios cannot be empty')
        self.J=len(self.scenarios); self.R=len(self.scenarios[0])
        self.Ns=[int(u.shape[0]) for u in self.scenarios[0]]
        Ks=[int(u.shape[1]) for u in self.scenarios[0]]
        for fac in self.scenarios:
            if len(fac)!=self.R:
                raise ValueError('all scenarios must have the same number of modes')
            for r,U in enumerate(fac):
                if U.ndim!=2 or U.shape!=(self.Ns[r],Ks[r]):
                    raise ValueError('all scenario factors must share mode dimensions')
        if self.lower_bounds is None:
            self.lower_bounds=Ks
        self.Ns,self.lower_bounds,self.total_budget=validate_total_budget(self.Ns,self.lower_bounds,self.total_budget)
        # The paper uses the latent dimensions as cardinality floors in the experiment model.
        if any(lb<k for lb,k in zip(self.lower_bounds,Ks)):
            raise ValueError('per-mode cardinality floors must be at least the latent dimensions K_r')
        self.Ks=Ks
        # Paper Eq. (2) uses K_r as the theoretical rank floor. Experiments may
        # impose a stricter operational floor for numerical rank stability.
        self.theoretical_rank_floors=list(Ks)
        self.operational_floors=list(self.lower_bounds)
        self.underline_ells,self.bar_ells=mode_budget_bounds(self.Ns,self.lower_bounds,self.total_budget)
        self.slices=[]; k=0
        for n in self.Ns:
            self.slices.append(slice(k,k+n)); k+=n
        self.nx=k
        self.y0=np.full(self.J,1/self.J)
        if self.sensitivities is None:
            # Model discrepancy to the first (nominal/reference) scenario, normalized into [0,1].
            sens=[]; nom=self.scenarios[0]
            for fac in self.scenarios:
                row=[]
                for U,U0 in zip(fac,nom):
                    d=np.sum((U-U0)**2,axis=1); mx=float(d.max(initial=0.0))
                    row.append(d/mx if mx>0 else np.zeros_like(d))
                sens.append(row)
            self.sensitivities=sens
        if len(self.sensitivities)!=self.J or any(len(row)!=self.R for row in self.sensitivities):
            raise ValueError('sensitivities must match scenario/mode dimensions')
        # Experimental affine coupling for the new total-budget model:
        #   h_j(x) = (1/L) sum_r <c_j^(r), x_r> in [0,1],
        #   y_j + eta_c h_j(x) <= u0 + eta_c.
        # In this implementation G=y+B x-b <= 0 and lambda<=0.
        self.B=np.zeros((self.J,self.nx))
        denom=float(self.total_budget)
        for j in range(self.J):
            for r,sl in enumerate(self.slices):
                c=np.asarray(self.sensitivities[j][r],float)
                if c.shape!=(self.Ns[r],):
                    raise ValueError('sensitivity vector has wrong shape')
                if np.min(c)<-1e-12 or np.max(c)>1+1e-12:
                    raise ValueError('experiment sensitivities must lie in [0,1]')
                self.B[j,sl]=(self.eta_c/denom)*c
        self.b=np.full(self.J,self.u0+self.eta_c)

    def split_x(self,x):
        x=np.asarray(x,float)
        if x.shape!=(self.nx,):
            raise ValueError('x has wrong shape')
        return [x[sl] for sl in self.slices]
    def join_x(self,xs): return np.concatenate(xs)
    def allocation(self,x): return allocation_from_vector(np.asarray(x,float),self.Ns)
    def uniform_x(self): return balanced_total_budget_point(self.Ns,self.lower_bounds,self.total_budget)
    def cold_z(self): return np.concatenate([self.y0, np.zeros(self.J)])

    def is_outer_feasible(self,x,tol=1e-8):
        x=np.asarray(x,float)
        if x.shape!=(self.nx,) or np.min(x)<-tol or np.max(x)>1+tol:
            return False
        a=self.allocation(x)
        return bool(abs(float(a.sum())-self.total_budget)<=tol and np.all(a+tol>=np.asarray(self.lower_bounds)))

    def Fvec(self,x):
        xs=self.split_x(x)
        return np.asarray([tensor_fp_multilinear(fac,xs) for fac in self.scenarios])/self.scale
    def Jmat(self,x):
        xs=self.split_x(x); rows=[]
        for fac in self.scenarios:
            rows.append(np.concatenate(tensor_fp_grad(fac,xs))/self.scale)
        return np.asarray(rows)
    def psi(self,y): return 0.5*self.mu*np.sum((y-self.y0)**2)
    def grad_psi(self,y): return self.mu*(y-self.y0)
    def G(self,x,y): return y + self.B@x - self.b
    def lagrangian(self,x,lam,y): return float(y@self.Fvec(x)-self.psi(y)+lam@self.G(x,y))
    def H(self,x,z):
        y=z[:self.J]; lam=z[self.J:]
        return np.concatenate([self.grad_psi(y)-self.Fvec(x)-lam, self.G(x,y)])
    def robust_grad(self,x,z):
        y=z[:self.J]; lam=z[self.J:]
        return self.Jmat(x).T@y + self.B.T@lam


def build_task_from_synthetic(stask,total_budget,lower_bounds=None,mu=.1,u0=.45,eta_c=.2):
    if lower_bounds is None:
        lower_bounds=[int(U.shape[1]) for U in stask.nominal]
    tmp=RobustTask(stask.design_scenarios,total_budget,lower_bounds,mu,u0,eta_c)
    # Pre-run scaling at a deterministic feasible total-budget fractional design; task-fixed before solve.
    ref=max(1.0,float(np.max(tmp.Fvec(tmp.uniform_x()))))
    return RobustTask(stask.design_scenarios,total_budget,lower_bounds,mu,u0,eta_c,scale=ref)
