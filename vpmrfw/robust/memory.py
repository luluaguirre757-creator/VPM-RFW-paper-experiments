from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from vpmrfw.robust.transfer import paper_transfer_bounds, paper_stationarity_transfer_bound
from vpmrfw.robust.ppa import natural_residual

@dataclass
class MemoryItem:
    task_id:int; task:object; x:np.ndarray; z:np.ndarray; Ubar:float; e:float; realized_cost:float
    C_hat:float; epsilon_z:float; Lgz:float; allocation:tuple[float,...]; certified:bool=False
    terminal_fw_gap:float=float('nan'); solver_execution:str='practical'; theorem_certified:bool=False


def terminal_memory_item(task_id,task,res,cert,certified_execution=True):
    x=res.x; z=res.z; J=task.J; y=z[:J]; lam=z[J:]
    nr=natural_residual(task,x,z)
    if certified_execution:
        e=cert.kappa_R_hat*nr
        if not (res.theorem_certified and res.certified_converged and e<=cert.eps_z*(1+1e-10)):
            raise ValueError('uncertified terminal state cannot enter theorem-certified VPM memory')
    else:
        e=cert.kappa_R_hat*nr
    gy=task.Fvec(x)-task.grad_psi(y)+lam
    Ubar=float(task.lagrangian(x,lam,y)+max(0.0,float(np.max(gy)-gy@y)))
    allocation=tuple(map(float,task.allocation(x)))
    return MemoryItem(task_id,task,x.copy(),z.copy(),Ubar,float(e),float(res.lmo+res.fo),
                      float(cert.C_hat),float(cert.eps_z),float(cert.Lgz),allocation,bool(certified_execution),
                      float(res.fw_gap_cert if res.theorem_certified else res.fw_gap_approx),
                      str(res.solver_execution),bool(res.theorem_certified))


def _state_upper_value(task,x,z):
    """Computable robust-value upper state used by the practical VPM score.

    This is the same primal-dual upper-state expression stored at terminal tasks,
    but evaluated under the *current* task model at a candidate historical state.
    It uses no current-task truth/reconstruction samples.
    """
    z=np.asarray(z,float); y=z[:task.J]; lam=z[task.J:]
    gy=task.Fvec(x)-task.grad_psi(y)+lam
    return float(task.lagrangian(x,lam,y)+max(0.0,float(np.max(gy)-gy@y)))


def state_conditioned_score(cur,x,z,cert):
    """Tight numerical V/KKT score for practical experiments.

    The theorem path uses the task-uniform transfer certificate.  Practical
    experiments already carry ``caps.status == numerical_proxy`` and execute a
    residual-stopped solver, so selection should be based on the actually
    available current-task state rather than a very loose global sup bound.

    ``delta_state`` is the current robust value upper-state gap and ``d0_state``
    is the residual-to-distance proxy at the candidate warm start.  Feeding
    these into the *same* cost algebra preserves the value/KKT interpretation
    and makes cold/history scores directly comparable.
    """
    x=np.asarray(x,float); z=np.asarray(z,float)
    U=_state_upper_value(cur,x,z)
    delta=max(0.0,U-cert.M_minus)
    nr=natural_residual(cur,x,z)
    d0=cert.kappa_R_hat*nr
    cost,meta=cert.cost(delta,d0)
    return cost,{**meta,'selection_score':cost,
                 'selection_score_status':'state_conditioned_numerical_proxy',
                 'selection_mode':'practical_proxy','theoretical_certified_cost':np.nan,
                 'state_value_upper':float(U),'state_natural_residual':float(nr),
                 'state_delta_hat':float(delta),'state_d0_hat':float(d0)}


def practical_history_score(cur,item,cert):
    if cur.Ns!=item.task.Ns or cur.lower_bounds!=item.task.lower_bounds or cur.total_budget!=item.task.total_budget:
        return float('inf'),{'eligible':False,'reason':'incompatible_total_budget_geometry'}
    # Keep the paper task-uniform transfer score as a diagnostic, but do not use
    # that deliberately worst-case quantity to rank warm starts in practical mode.
    Cpair=max(cert.C_hat,item.C_hat)
    tr=paper_transfer_bounds(cur,item.task,Cpair)
    global_delta=max(0.0,item.Ubar+tr['V_hat']-cert.M_minus)
    global_d0=item.e+cert.caps.bar_kappa_cross*tr['Omega_hat']
    global_cost,_=cert.cost(global_delta,global_d0)
    cost,meta=state_conditioned_score(cur,item.x,item.z,cert)
    return cost,{**tr,**meta,'history_allocation':list(item.allocation),
                 'global_transfer_score_proxy':float(global_cost)}


def practical_cold_score(cur,cert):
    global_cost,_=cert.cold()
    cost,meta=state_conditioned_score(cur,cur.uniform_x(),cur.cold_z(),cert)
    return cost,{**meta,'action':'cold','history_allocation':None,
                 'global_transfer_score_proxy':float(global_cost)}


def history_certificate(cur,item,cert):
    # Revised theory assumes a common (N_r,K_r,L) total-budget polytope; this guard prevents
    # accidental reuse across incompatible outer feasible sets.
    if not item.certified:
        return float('inf'),{'eligible':False,'reason':'uncertified_memory_item'}
    if cur.Ns!=item.task.Ns or cur.lower_bounds!=item.task.lower_bounds or cur.total_budget!=item.task.total_budget:
        return float('inf'),{'eligible':False,'reason':'incompatible_total_budget_geometry'}
    Cpair=max(cert.C_hat,item.C_hat)
    tr=paper_transfer_bounds(cur,item.task,Cpair)
    delta=max(0.0,item.Ubar+tr['V_hat']-cert.M_minus)
    d0=item.e+cert.caps.bar_kappa_cross*tr['Omega_hat']
    cost,meta=cert.cost(delta,d0)
    return cost,{**tr,**meta,'cert_cost':cost,'selection_mode':'certified',
                 'history_allocation':list(item.allocation)}


def cold_certificate(cert):
    cost,meta=cert.cold(); return cost,{**meta,'cert_cost':cost,'selection_mode':'certified',
                                       'action':'cold','history_allocation':None}


def stationarity_screen(cur,memory,cert):
    """Paper Theorem 2 screen; returns the best certified reusable outer point."""
    if not cert.caps.theorem_certified:
        return None,{'stationarity_screened':False,'reason':'current_task_not_theorem_certified'}
    rows=[]
    for item in memory:
        if not item.theorem_certified or not np.isfinite(item.terminal_fw_gap):
            continue
        if cur.Ns!=item.task.Ns or cur.lower_bounds!=item.task.lower_bounds or cur.total_budget!=item.task.total_budget:
            continue
        gamma,meta=paper_stationarity_transfer_bound(cur,item.task,cert,item)
        bound=float(item.terminal_fw_gap+cert.D*gamma)
        rows.append((bound,item,meta))
    if not rows:
        return None,{'stationarity_screened':False,'reason':'no_certified_screen_history'}
    bound,item,meta=min(rows,key=lambda q:(q[0],q[1].task_id))
    if bound>cert.epsilon:
        return None,{'stationarity_screened':False,'reason':'screen_bound_above_target',
                     'best_stationarity_bound':bound,**meta}
    return item,{'stationarity_screened':True,'action':'stationarity_screen','selection_mode':'certified',
                 'transferred_fw_gap_certified':bound,**meta}


def select_action(policy,cur,memory,cert,rng=None):
    """Select a cold/history action under the allocation-aware total-budget memory model.

    A selected history carries its prior allocation implicitly through x. That allocation is
    *not* frozen: subsequent total-budget LMOs may move budget across modes.
    """
    practical = cert.caps.status != 'theorem_certified'
    cold_cost,cold_meta=(practical_cold_score(cur,cert) if practical else cold_certificate(cert))
    def _with_relative(meta,cost):
        out=dict(meta); out['cold_selection_score']=float(cold_cost)
        out['selection_score_ratio_to_cold']=float(cost/cold_cost) if np.isfinite(cost) and np.isfinite(cold_cost) and cold_cost>0 else np.nan
        return out
    cold_meta=_with_relative(cold_meta,cold_cost)
    if policy=='cold' or not memory: return None,cold_meta
    rows=[]
    for it in memory:
        c,m=(practical_history_score(cur,it,cert) if practical else history_certificate(cur,it,cert)); rows.append((it,c,m))
    finite=[q for q in rows if np.isfinite(q[1])]
    if policy=='vpm':
        if not finite or cold_cost<=min(q[1] for q in finite): return None,cold_meta
        best=min(finite,key=lambda q:(q[1],q[0].task_id)); return best[0],{'action':'history','policy':'vpm',**_with_relative(best[2],best[1])}
    if policy=='prev':
        it=memory[-1]; c,m=(practical_history_score(cur,it,cert) if practical else history_certificate(cur,it,cert))
        return (None,cold_meta) if not np.isfinite(c) else (it,{'action':'history','policy':'prev',**_with_relative(m,c)})
    if policy=='random':
        candidates=finite
        if not candidates: return None,cold_meta
        q=candidates[int(rng.integers(len(candidates)))]; return q[0],{'action':'history','policy':'random',**_with_relative(q[2],q[1])}
    if policy=='nearest': key=lambda q:q[2]['V_hat']+q[2]['Omega_hat']
    elif policy=='v_only': key=lambda q:q[2]['V_hat']
    elif policy=='omega_only': key=lambda q:q[2]['d0_hat']
    else: raise ValueError(policy)
    if not finite: return None,cold_meta
    best=min(finite,key=key); return best[0],{'action':'history','policy':policy,**_with_relative(best[2],best[1])}
