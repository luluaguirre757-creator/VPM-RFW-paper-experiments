"""Exact VPM-RFW paper score formulas with explicitly declared regularity caps.

The algebra in this module follows the paper exactly.  The *status* of the caps matters:
`theorem_certified` may only be used when interval/rational/local-chart verification was actually
performed; the provided experiment configs use `numerical_proxy`, so plots/tables must not
call those scores theorem-level certificates.
"""
from __future__ import annotations
from dataclasses import dataclass
import math, numpy as np
from vpmrfw.core.fp import fp_mode_terms


def _ceil_log_pos(x:float, base_abs_log:float)->int:
    if x<=1.0: return 0
    return int(math.ceil(math.log(x)/base_abs_log))


def geometry_bounds(task):
    # Total-budget geometry from the revised paper:
    # diam(P)<=sqrt(2 min{L,N-L}) and sup_{x in P} ||x||<=sqrt(L).
    N=int(sum(task.Ns)); L=int(task.total_budget)
    D=math.sqrt(max(0.0,2.0*min(L,N-L)))
    RP=math.sqrt(float(L))
    DY=math.sqrt(2.0) if task.J>=2 else 0.0
    return D,RP,DY


def scenario_mode_bounds(U,L):
    d,Q=fp_mode_terms(U); L=int(L)
    # Certified capped-simplex bounds, much tighter than replacing x by the full box [0,1]^N.
    # d^T x <= sum of the L largest diagonal coefficients.  For every i,
    # (Qx)_i <= sum of the L largest entries in row i, hence
    # x^T Q x <= ||x||_1 max_i (Qx)_i = L max_i topLsum(Q_i,:).
    top_d=float(np.sort(d)[-L:].sum()) if L else 0.0
    if L:
        top_row=np.partition(Q, Q.shape[1]-L, axis=1)[:,-L:].sum(axis=1)
    else:
        top_row=np.zeros(Q.shape[0])
    M=top_d + L*float(np.max(top_row,initial=0.0))
    # Componentwise positive upper bound for grad=d+2Qx, then Euclidean norm.
    grad_bound=d+2.0*top_row
    G=float(np.linalg.norm(grad_bound))
    H=float(np.linalg.norm(2.0*Q,2))
    return M,G,H


def task_smoothness_bounds(task):
    ten_G=[]; ten_L=[]; Fmax=[]
    for fac in task.scenarios:
        mode=[scenario_mode_bounds(U,Lbar) for U,Lbar in zip(fac,task.bar_ells)]
        Ms=[q[0] for q in mode]; Gs=[q[1] for q in mode]; Hs=[q[2] for q in mode]
        # task.Fvec divides tensor FP by task.scale
        fmax=float(np.prod(Ms))/task.scale
        gblocks=[Gs[r]*float(np.prod([Ms[q] for q in range(task.R) if q!=r]))/task.scale for r in range(task.R)]
        gten=float(np.linalg.norm(gblocks))
        lten=(sum(Hs[r]*float(np.prod([Ms[q] for q in range(task.R) if q!=r])) for r in range(task.R))
              +2.0*sum(Gs[r]*Gs[q]*float(np.prod([Ms[p] for p in range(task.R) if p not in (r,q)]))
                       for r in range(task.R) for q in range(r+1,task.R)))/task.scale
        Fmax.append(fmax); ten_G.append(gten); ten_L.append(float(lten))
    MJ=float(np.linalg.norm(ten_G)); LJ=float(np.linalg.norm(ten_L))
    # f=y^T F - psi. F>=0. max psi on simplex relative to uniform y0.
    psi_max=0.5*task.mu*(1.0-1.0/task.J) if task.J>0 else 0.0
    Mplus=max(Fmax); Mminus=-psi_max
    return {'M_J':MJ,'L_J':LJ,'M_plus':float(Mplus),'M_minus':float(Mminus),'Fmax':Fmax}


def uniform_rcq_lower_bound(task):
    """Analytic uniform Slater/RCQ margin for this experiment's G=y+Bx-b<=0.

    In the total-budget experiment h_j(x)=(1/L)sum_r<c_j^(r),x_r> with c in [0,1].
    Since sum_r 1^T x_r=L, B_j x=eta_c h_j(x)<=eta_c; with y0=1/J and
    b=u0+eta_c, G_j(x,y0)<=1/J-u0.
    """
    return max(0.0,float(task.u0-1.0/task.J))

@dataclass(frozen=True)
class RegularityCaps:
    bar_kappa:float=8.0
    bar_kappa_ms:float=8.0
    bar_kappa_cross:float=8.0
    r_ppa:float=25.0
    alpha_ppa:float=1.0
    status:str='numerical_proxy'
    enforce_local_basin:bool=True
    global_cold_error_bound_verified:bool=False
    trajectory_cover_verified:bool=False

    @property
    def theorem_certified(self)->bool:
        return (self.status == 'theorem_certified' and
                (self.trajectory_cover_verified or self.enforce_local_basin))

@dataclass
class TaskCertificate:
    task:object; epsilon:float; w_lmo:float; w_fo:float; caps:RegularityCaps
    D:float; RP:float; DY:float; M_J:float; L_J:float; M_plus:float; M_minus:float
    sigma:float; C_hat:float; L_Hz:float; L_Hx:float; Lz_hat:float; LPhi_hat:float
    Lgz:float; eps_z:float; gamma_basin:float; gamma:float; d_cont:float
    chi:float; rho:float; eta_alpha:float; kappa_R_hat:float

    def T(self,delta:float)->int:
        return max(1,int(math.ceil(4.0*max(0.0,float(delta))/(self.gamma*self.epsilon))))

    def inner_components(self,d:float):
        d=max(0.0,float(d)); ez=self.eps_z
        K=_ceil_log_pos(2.0*d/ez,abs(math.log(self.chi)))
        ratio=2.0*(1.0+self.chi)*(d+ez/2.0)/((1.0-self.chi)*ez)
        N=_ceil_log_pos(ratio,abs(math.log(self.rho)))
        return K,N,K*N

    def inner_fo(self,d:float)->int: return self.inner_components(d)[2]

    def cost(self,delta:float,d0:float,*,allow_global_error_bound:bool=False):
        # The theorem-certified path enforces the declared local PPA basin unless an
        # independently verified global-on-trajectory error bound is available.  The
        # default experiment path uses the same algebra as a numerical score proxy but
        # never labels it a theorem certificate.
        basin_ok = d0 <= self.caps.r_ppa + 1e-12
        global_ok = bool(allow_global_error_bound and self.caps.global_cold_error_bound_verified)
        if self.caps.enforce_local_basin and not (basin_ok or global_ok):
            return float('inf'), {'eligible':False,'reason':'ppa_basin','basin_ok':False,
                                  'score_status':self.caps.status}
        T=self.T(delta); n0=self.inner_fo(d0); nc=self.inner_fo(self.d_cont)
        C=self.w_lmo*T+self.w_fo*(n0+(T-1)*nc)
        return float(C),{'eligible':True,'T_hat':T,'N_first_hat':n0,'N_cont_hat':nc,
                         'delta_hat':float(delta),'d0_hat':float(d0),'d_cont_hat':self.d_cont,
                         'gamma_hat':self.gamma,'epsilon_z':self.eps_z,'cert_status':self.caps.status,
                         'score_status':self.caps.status,'basin_ok':bool(basin_ok),
                         'global_error_bound_used':bool(global_ok)}

    def cold(self):
        d=math.sqrt(self.DY**2+self.C_hat**2); delta=self.M_plus-self.M_minus
        return self.cost(delta,d,allow_global_error_bound=True)


def build_task_certificate(task,epsilon:float,w_lmo=1.0,w_fo=1.0,caps:RegularityCaps|None=None):
    caps=caps or RegularityCaps(); epsilon=float(epsilon)
    if epsilon<=0: raise ValueError('epsilon must be positive')
    D,RP,DY=geometry_bounds(task); sb=task_smoothness_bounds(task)
    sigma=uniform_rcq_lower_bound(task)
    if sigma<=0: raise ValueError('experiment coupling parameters do not provide positive uniform RCQ margin')
    C_hat=(sb['M_plus']-sb['M_minus'])/sigma
    Bnorm=float(np.linalg.norm(task.B,2))
    Lgz=sb['M_J']+Bnorm
    eps_z=epsilon/(4.0*D*Lgz) if D*Lgz>0 else epsilon
    # H_z = [[mu I,-I],[I,0]]
    J=task.J
    Hz=np.block([[task.mu*np.eye(J),-np.eye(J)],[np.eye(J),np.zeros((J,J))]])
    LHz=float(np.linalg.norm(Hz,2))
    LHx=math.sqrt(sb['M_J']**2+Bnorm**2)
    Lz=caps.bar_kappa*LHx
    LPhi=sb['L_J']+(sb['M_J']+Bnorm)*Lz
    if not caps.enforce_local_basin:
        gb=1.0
    elif Lz*D==0:
        gb=1.0
    else:
        gb=min(1.0,max(0.0,(caps.r_ppa-eps_z)/(Lz*D)))
    smooth_term=float('inf') if LPhi*D*D==0 else epsilon/(2.0*LPhi*D*D)
    gamma=min(1.0,smooth_term,gb)
    if gamma<=0: raise ValueError('declared PPA basin is too small for configured epsilon')
    dcont=eps_z+Lz*gamma*D
    alpha=caps.alpha_ppa; chi=(1.0+alpha*alpha/(caps.bar_kappa_ms**2))**-0.5
    m=1.0/alpha; Lalpha=LHz+1.0/alpha; eta=m/(Lalpha*Lalpha)
    rho=math.sqrt(max(0.0,1.0-2.0*eta*m+eta*eta*Lalpha*Lalpha))
    kR=1.0+caps.bar_kappa_ms*(1.0+LHz)  # tau=1
    return TaskCertificate(task,epsilon,float(w_lmo),float(w_fo),caps,D,RP,DY,sb['M_J'],sb['L_J'],
        sb['M_plus'],sb['M_minus'],sigma,C_hat,LHz,LHx,Lz,LPhi,Lgz,eps_z,gb,gamma,dcont,chi,rho,eta,kR)
