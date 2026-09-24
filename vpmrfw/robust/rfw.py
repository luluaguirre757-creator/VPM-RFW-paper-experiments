from __future__ import annotations
from dataclasses import dataclass
import time, numpy as np
from vpmrfw.core.simplex import lmo_total_budget, round_total_budget
from vpmrfw.robust.ppa import solve_inner_paper
from vpmrfw.robust.trajectory import CertificationFailure

@dataclass
class RFWResult:
    x:np.ndarray; z:np.ndarray; sets:list[np.ndarray]; lmo:int; fo:int; outer:int
    fw_gap_cert:float; fw_gap_approx:float; runtime:float; history:list; inner_converged:bool
    epsilon_z:float; configured_T:int; gamma:float
    ppa:int=0; resolvent:int=0; converged:bool=False
    hit_outer_cap:bool=False; hit_ppa_cap:bool=False
    certified_converged:bool=False; theorem_certified:bool=False
    hit_resolvent_cap:bool=False; reached_ppa_limit:bool=False
    reached_resolvent_limit:bool=False
    termination_reason:str=''; natural_residual_initial:float=float('nan')
    natural_residual_final:float=float('nan'); distance_cert_initial:float=float('nan')
    distance_cert_final:float=float('nan'); epsilon_z_target:float=float('nan')
    distance_cert_ratio:float=float('nan')
    solver_execution:str='practical'; practical_converged:bool=False
    probe_weighted_cost:float=0.0


def lmo_total_budget_task(task,g):
    """Exact LMO from the revised paper: mandatory lower bounds + global residual budget."""
    return lmo_total_budget(np.asarray(g,float),task.Ns,task.lower_bounds,task.total_budget)


def round_task(task,x):
    """Nearest feasible binary total-budget design R_L(x)."""
    return round_total_budget(np.asarray(x,float),task.Ns,task.lower_bounds,task.total_budget)


def solve_rfw_paper(task,cert,delta_hat:float,d0_hat:float,x0=None,z0=None,store_history=False,
                    hard_iteration_guard:int|None=None, trajectory_guard=None,oracle_guard=None):
    """Paper Algorithm 1 outer/inner execution for one chosen action.

    This version uses the revised total-budget feasible set. The exact LMO can revise
    the cross-mode allocation at every FW step, even when the warm start came from a
    historical allocation profile.
    """
    if cert.caps.status != 'theorem_certified':
        raise CertificationFailure(
            "solve_rfw_paper requires status='theorem_certified'; numerical condition estimates "
            "must use solve_rfw_practical and cannot be relabelled as theorem-certified."
        )
    if trajectory_guard is None:
        raise CertificationFailure("theorem-certified execution requires an explicit trajectory guard")
    x=task.uniform_x() if x0 is None else np.asarray(x0,float).copy()
    if not task.is_outer_feasible(x):
        raise ValueError('outer warm start is not feasible for the common total-budget polytope')
    z=task.cold_z() if z0 is None else np.asarray(z0,float).copy()
    T=cert.T(delta_hat); gamma=cert.gamma
    if hard_iteration_guard is not None and T>int(hard_iteration_guard):
        raise RuntimeError(f"paper-configured T_hat={T} exceeds explicit hard_iteration_guard={hard_iteration_guard}; "
                           "increase the guard or tighten declared numerical/certified constants")
    best=(float('inf'),float('inf'),x.copy(),z.copy()); lmo=fo=ppa=resolvent=0; hist=[]; allconv=True
    d_bound=float(d0_hat); t0=time.perf_counter(); completed=0
    for t in range(T):
        ii=solve_inner_paper(task,x,z,cert,d_bound,oracle_guard=oracle_guard); z=ii.z; fo+=ii.fo; ppa+=ii.ppa; resolvent+=ii.fo; allconv &= ii.converged
        if not ii.converged:
            raise CertificationFailure(
                f"inner certified-distance target failed at outer iteration {t}: "
                f"distance_certificate={ii.distance_certificate:.6g} > epsilon_z={cert.eps_z:.6g}"
            )
        g=task.robust_grad(x,z)
        if oracle_guard is not None: oracle_guard('lmo')
        v=lmo_total_budget_task(task,g); lmo+=1
        gap_approx=float((x-v)@g)
        grad_err=cert.Lgz*ii.distance_certificate
        gap_cert=gap_approx+cert.D*grad_err
        if gap_cert<best[0]: best=(gap_cert,gap_approx,x.copy(),z.copy())
        if store_history:
            hist.append({'t':t,'gap_cert':gap_cert,'gap_approx':gap_approx,'inner_res':ii.residual,
                         'inner_distance_cert':ii.distance_certificate,'inner_fo':ii.fo,
                         'allocation':task.allocation(x).tolist()})
        completed=t+1
        if gap_cert<=cert.epsilon: break
        x_plus=(1.0-gamma)*x+gamma*v
        if not trajectory_guard.accept_segment(x,x_plus):
            raise CertificationFailure(f"trajectory guard rejected FW segment at outer iteration {t}")
        x=x_plus
        d_bound=cert.d_cont
    runtime=time.perf_counter()-t0
    gap_cert,gap_approx,x,z=best
    sets=round_task(task,x)
    converged=bool(allconv and gap_cert<=cert.epsilon)
    hit_outer=bool(completed>=T and gap_cert>cert.epsilon)
    return RFWResult(x,z,sets,lmo,fo,completed,gap_cert,gap_approx,runtime,hist,allconv,
                     cert.eps_z,T,gamma,ppa,resolvent,converged,hit_outer,False,
                     certified_converged=allconv,theorem_certified=bool(allconv and cert.caps.theorem_certified),
                     hit_resolvent_cap=False,reached_ppa_limit=ii.reached_ppa_limit,
                     reached_resolvent_limit=False,
                     termination_reason=('certified_tolerance' if converged else 'outer_cap_before_certification'),
                     natural_residual_initial=ii.natural_residual_initial,natural_residual_final=ii.residual,
                     distance_cert_initial=ii.distance_cert_initial,distance_cert_final=ii.distance_certificate,
                     epsilon_z_target=cert.eps_z,distance_cert_ratio=ii.distance_cert_ratio,
                     solver_execution='certified',practical_converged=allconv)


def solve_rfw_practical(task,cert,x0=None,z0=None,outer_gap_tol:float=1e-2,inner_residual_tol:float=1e-5,
                        max_outer:int=500,max_ppa:int=100,max_resolvent:int=500,store_history=False):
    """Explicitly non-certified empirical RFW execution on the total-budget polytope."""
    from vpmrfw.robust.ppa import solve_inner_practical
    x=task.uniform_x() if x0 is None else np.asarray(x0,float).copy()
    if not task.is_outer_feasible(x):
        raise ValueError('outer warm start is not feasible for the common total-budget polytope')
    z=task.cold_z() if z0 is None else np.asarray(z0,float).copy()
    best=(float('inf'),x.copy(),z.copy()); lmo=fo=ppa=resolvent=0; hist=[]
    allcert=True; allpractical=True; hit_ppa=False; hit_resolvent=False; reached_ppa=False
    reached_resolvent=False; first_ii=None
    t0=time.perf_counter(); completed=0
    for t in range(int(max_outer)):
        ii=solve_inner_practical(task,x,z,cert,inner_residual_tol,max_ppa,max_resolvent)
        if first_ii is None: first_ii=ii
        z=ii.z; fo+=ii.fo; ppa+=ii.ppa; resolvent+=ii.fo
        allcert &= ii.certified_converged; allpractical &= ii.practical_converged
        hit_ppa |= ii.hit_ppa_cap; hit_resolvent |= ii.hit_resolvent_cap
        reached_ppa |= ii.reached_ppa_limit
        reached_resolvent |= ii.reached_resolvent_limit
        g=task.robust_grad(x,z); v=lmo_total_budget_task(task,g); lmo+=1
        gap=float((x-v)@g)
        if gap<best[0]: best=(gap,x.copy(),z.copy())
        if store_history:
            hist.append({'t':t,'gap_approx':gap,'inner_res':ii.residual,'inner_fo':ii.fo,
                         'allocation':task.allocation(x).tolist()})
        completed=t+1
        if gap<=outer_gap_tol: break
        gamma=2.0/(t+2.0)
        x=(1.0-gamma)*x+gamma*v
    runtime=time.perf_counter()-t0
    gap,x,z=best
    sets=round_task(task,x)
    converged=bool(allpractical and gap<=outer_gap_tol)
    hit_outer=bool(completed>=int(max_outer) and gap>outer_gap_tol)
    reason=('resolvent_cap_before_certification' if hit_resolvent else
            'ppa_cap_before_certification' if hit_ppa else
            'practical_tolerance' if converged else 'practical_budget_exhausted')
    return RFWResult(x,z,sets,lmo,fo,completed,float('nan'),gap,runtime,hist,allpractical,
                     float('nan'),int(max_outer),float('nan'),ppa,resolvent,converged,hit_outer,hit_ppa,
                     certified_converged=allcert,theorem_certified=False,
                     hit_resolvent_cap=hit_resolvent,reached_ppa_limit=reached_ppa,
                     reached_resolvent_limit=reached_resolvent,
                     termination_reason=reason,
                     natural_residual_initial=first_ii.natural_residual_initial,
                     natural_residual_final=ii.residual,distance_cert_initial=first_ii.distance_cert_initial,
                     distance_cert_final=ii.distance_certificate,epsilon_z_target=cert.eps_z,
                     distance_cert_ratio=ii.distance_cert_ratio,
                     solver_execution='practical',practical_converged=allpractical)
