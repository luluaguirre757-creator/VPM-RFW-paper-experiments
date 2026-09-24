from __future__ import annotations
import json, time
import numpy as np, pandas as pd
from scipy.linalg import svdvals

from vpmrfw.data.synthetic import make_sequence
from vpmrfw.robust.problem import build_task_from_synthetic, RobustTask
from vpmrfw.robust.certificate import build_task_certificate
from vpmrfw.robust.rfw import solve_rfw_paper, solve_rfw_practical, RFWResult, round_task
from vpmrfw.robust.trajectory import FullCoverTrajectoryGuard
from vpmrfw.robust.trajectory import CertificationFailure
from vpmrfw.robust.memory import (terminal_memory_item, select_action, history_certificate, cold_certificate,
    practical_history_score, practical_cold_score, stationarity_screen)
from vpmrfw.robust.ppa import natural_residual
from vpmrfw.utils.config import caps_from_cfg
from vpmrfw.utils.timing import median_runtime
from vpmrfw.baselines.greedy_fp_ortiz2019 import greedy_fp_original
from vpmrfw.baselines.ffw_li2026 import ffw_original_total_budget
from vpmrfw.core.fp import tensor_fp_discrete
from vpmrfw.core.metrics import structured_theoretical_mse, structured_full_rank
from vpmrfw.core.reconstruction import empirical_structured_nmse
from vpmrfw.core.simplex import allocation_from_sets

POLICY_NAMES={
    'cold':'Cold-RFW','prev':'Prev-RFW','nearest':'Nearest-RFW',
    'v_only':'V-only','omega_only':'Omega-only','vpm':'VPM-RFW'
}


def _solve(task, cert, meta, cfg, x0=None, z0=None, store_history=False, quick=False):
    mode=str(cfg.get('solver_execution','practical')).lower()
    if mode in ('paper_exact','certified'):
        if cert.caps.status != 'theorem_certified' or not cert.caps.trajectory_cover_verified:
            raise RuntimeError(
                "paper_exact is disabled unless theorem-certified constants AND an independently verified "
                "full trajectory cover are declared. Floating-point diagnostics are not promoted to certificates."
            )
        guard=FullCoverTrajectoryGuard(True, provenance='user-supplied verified full cover')
        action=str(meta.get('action','history' if x0 is not None else 'cold'))
        cold_cost,cold_meta=cert.cold()
        eta=float(cfg.get('fail_safe_eta',1.0))
        if action=='history' and (not np.isfinite(cold_cost) or float(meta.get('cert_cost',np.inf))>eta*cold_cost):
            x0=z0=None; meta=cold_meta; action='cold'
        if action=='cold' and (not np.isfinite(cold_cost) or not meta.get('eligible',True)):
            raise CertificationFailure('UNCERTIFIED: no finite independently certified cold branch')
        spent=[0.0]
        probe_budget=(eta*cold_cost if action=='history' else float('inf'))
        def oracle_guard(kind):
            unit=cert.w_lmo if kind=='lmo' else cert.w_fo
            if spent[0]+unit>probe_budget+1e-12:
                raise CertificationFailure('historical probe budget exhausted before indivisible oracle unit')
            spent[0]+=unit
        try:
            res=solve_rfw_paper(task,cert,meta['delta_hat'],meta['d0_hat'],x0,z0,store_history,
                                hard_iteration_guard=cfg.get('hard_iteration_guard'),trajectory_guard=guard,
                                oracle_guard=(oracle_guard if action=='history' else None))
            res.probe_weighted_cost=float(spent[0])
            return res,'theorem_certified_execution'
        except CertificationFailure as warm_error:
            # A failed history probe is not reusable evidence.  Retry from the independently
            # certified cold action; if that also fails, expose UNCERTIFIED instead of silently
            # continuing with a truncated inner state.
            if x0 is None and z0 is None:
                raise CertificationFailure(f'UNCERTIFIED: cold certified branch failed: {warm_error}') from warm_error
            if not cold_meta.get('eligible',True):
                raise CertificationFailure('UNCERTIFIED: no eligible independently certified cold branch') from warm_error
            try:
                res=solve_rfw_paper(task,cert,cold_meta['delta_hat'],cold_meta['d0_hat'],None,None,store_history,
                                    hard_iteration_guard=cfg.get('hard_iteration_guard'),trajectory_guard=guard)
                res.probe_weighted_cost=float(spent[0])
                return res,'theorem_certified_execution_cold_fallback'
            except CertificationFailure as cold_error:
                raise CertificationFailure(f'UNCERTIFIED: warm and cold certified branches failed: {cold_error}') from cold_error
    if mode=='practical':
        pcfg=cfg.get('practical_solver',{})
        res=solve_rfw_practical(
            task,cert,x0,z0,
            outer_gap_tol=float(pcfg.get('outer_gap_tol',1e-2)),
            inner_residual_tol=float(pcfg.get('inner_residual_tol',1e-5)),
            max_outer=int(pcfg.get('quick_max_outer',20) if quick else pcfg.get('max_outer',500)),
            max_ppa=int(pcfg.get('quick_max_ppa',10) if quick else pcfg.get('max_ppa',100)),
            max_resolvent=int(pcfg.get('quick_max_resolvent',60) if quick else pcfg.get('max_resolvent',500)),
            store_history=store_history,
        )
        return res,'practical_noncertified_execution'
    raise ValueError(f"unknown solver_execution={mode}")


def _screened_result(task,item,cert,meta):
    """Return a Theorem-2-certified point without claiming a current-task KKT solve."""
    x=np.asarray(item.x,float).copy(); z=np.asarray(item.z,float).copy()
    nr=natural_residual(task,x,z); dist=cert.kappa_R_hat*nr
    gap=float(meta['transferred_fw_gap_certified'])
    return RFWResult(x,z,round_task(task,x),0,0,0,gap,float('nan'),0.0,[],False,
                     cert.eps_z,0,cert.gamma,converged=True,certified_converged=False,
                     theorem_certified=True,termination_reason='stationarity_transfer_screen',
                     natural_residual_initial=nr,natural_residual_final=nr,
                     distance_cert_initial=dist,distance_cert_final=dist,
                     epsilon_z_target=cert.eps_z,distance_cert_ratio=dist/cert.eps_z,
                     solver_execution='certified_stationarity_screen',practical_converged=False)


def _budget(cfg):
    return int(cfg['total_budget'])


def _lower(cfg):
    ranks=list(map(int,cfg['K']))
    floors=list(map(int,cfg.get('operational_floors',ranks)))
    assert len(floors)==len(ranks)
    assert sum(floors)<=_budget(cfg)
    assert all(f>=k for f,k in zip(floors,ranks))
    return floors


def _sets_json(sets):
    return json.dumps([np.asarray(s,dtype=int).tolist() for s in sets],separators=(',',':'))


def _alloc_json(vals):
    return json.dumps([float(v) for v in vals],separators=(',',':'))


def solver_diagnostics(res):
    """Serializable iteration/cap diagnostics; does not affect solver execution.

    ``inner_certified_converged`` reports only the distance-certificate state of
    every executed inner solve.  ``overall_certified_converged`` additionally
    requires a theorem-certified execution and the outer FW stopping condition.
    The legacy ``certified_converged`` field is retained for backward-readable
    result files but must not be interpreted as whole-algorithm certification.
    """
    overall_cert = bool(res.theorem_certified and res.certified_converged and res.converged
                        and str(res.solver_execution).startswith('certified'))
    return {
        'converged':int(bool(res.converged)),
        'solver_execution':str(res.solver_execution),
        'certified_converged':int(bool(res.certified_converged)),
        'inner_certified_converged':int(bool(res.certified_converged)),
        'overall_certified_converged':int(overall_cert),
        'practical_converged':int(bool(res.practical_converged)),
        'theorem_certified':int(bool(res.theorem_certified)),
        'hit_outer_cap':int(bool(res.hit_outer_cap)),
        'hit_ppa_cap':int(bool(res.hit_ppa_cap)),
        'ppa_cap_failure':int(bool(res.hit_ppa_cap)),
        'hit_resolvent_cap':int(bool(res.hit_resolvent_cap)),
        'resolvent_cap_failure':int(bool(res.hit_resolvent_cap)),
        'reached_ppa_limit':int(bool(res.reached_ppa_limit)),
        'reached_resolvent_limit':int(bool(res.reached_resolvent_limit)),
        'termination_reason':str(res.termination_reason),
        'natural_residual_initial':float(res.natural_residual_initial),
        'natural_residual_final':float(res.natural_residual_final),
        'distance_cert_initial':float(res.distance_cert_initial),
        'distance_cert_final':float(res.distance_cert_final),
        'epsilon_z_target':float(res.epsilon_z_target),
        'distance_cert_ratio':float(res.distance_cert_ratio),
        'fw_gap_empirical':float(res.fw_gap_approx),
        'fw_gap_certified':float(res.fw_gap_cert),
        'probe_weighted_cost':float(res.probe_weighted_cost),
        'num_outer':int(res.outer),
        'num_ppa':int(res.ppa),
        'num_resolvent':int(res.resolvent),
    }


def design_metadata(sets):
    a=allocation_from_sets(sets)
    out={
        'mode_cardinalities':json.dumps(a.astype(int).tolist(),separators=(',',':')),
        'total_mode_budget':int(a.sum()),
        'cartesian_observations':int(np.prod(a,dtype=np.int64)),
        # Backward-readable aliases used by old plotting scripts.
        'domain_budget':int(a.sum()),
        'tensor_samples':int(np.prod(a,dtype=np.int64)),
    }
    out.update({f'mode_budget_r{r+1}':int(v) for r,v in enumerate(a)})
    return out


def eval_structured(stask, sets, noise_var=1.0, robust_task=None):
    ms=np.asarray([structured_theoretical_mse(f,sets,noise_var) for f in stask.test_scenarios],float)
    finite=np.isfinite(ms)
    robust_discrete=np.nan
    if robust_task is not None:
        vals=[tensor_fp_discrete(fac,sets)/float(robust_task.scale) for fac in robust_task.scenarios]
        robust_discrete=float(np.max(vals))
    return {
        'wc_mse':float(np.max(ms)), 'p95_mse':float(np.quantile(ms,.95)) if np.all(finite) else float('inf'),
        'avg_mse':float(np.mean(ms)), 'median_mse':float(np.median(ms)),
        'nominal_fp':float(tensor_fp_discrete(stask.nominal,sets)),
        'robust_design_objective':robust_discrete,
        'singular':int(not np.all(finite)),
        **design_metadata(sets),
    }


def design_min_relative_singular_value(stask, sets):
    """Worst mode-wise relative singular value over design scenarios only."""
    values=[]
    for factors in stask.design_scenarios:
        for U,idx in zip(factors,sets):
            s=svdvals(np.asarray(U,float)[np.asarray(idx,dtype=int),:])
            values.append(0.0 if not len(s) or s[0]<=0 else float(s[-1]/s[0]))
    return float(min(values)) if values else 0.0


def _nominal_task(st,total_budget,lower,cfg):
    t=RobustTask([st.nominal],total_budget,lower,mu=float(cfg['mu']),u0=1.1,eta_c=0.0)
    ref=max(1.0,float(np.max(t.Fvec(t.uniform_x()))))
    return RobustTask([st.nominal],total_budget,lower,mu=float(cfg['mu']),u0=1.1,eta_c=0.0,scale=ref)


def _maybe_empirical(row, st, sets, cfg, seed, task_index, quick):
    reps=set(map(int,cfg.get('quick_empirical_tasks',[]) if quick else cfg.get('empirical_tasks',[])))
    trials=int(cfg.get('quick_empirical_trials',2) if quick else cfg.get('empirical_trials',0))
    if trials<=0 or int(task_index) not in reps:
        return row
    # Cardinality floors are not rank guarantees. Never hide a singular selected factor with pinv.
    if not structured_full_rank(st.nominal,sets):
        row['empirical_nmse_db_mean']=np.nan; row['empirical_nmse_db_std']=np.nan
        row['empirical_trials']=0; row['empirical_rank_feasible']=0
        row['reconstruction_runtime_median5']=np.nan
        return row
    row['empirical_rank_feasible']=1
    paired_seed=int(seed)*10_000_000+int(task_index)
    vals=empirical_structured_nmse(st.nominal,sets,np.random.default_rng(paired_seed),float(cfg.get('snr_db',20.0)),trials)
    row['empirical_nmse_db_mean']=float(np.mean(vals))
    row['empirical_nmse_db_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0
    row['empirical_trials']=int(trials)
    warm=int(cfg.get('runtime_warmups',1)); reps_rt=int(cfg.get('runtime_repeats',5))
    rt,_=median_runtime(lambda: empirical_structured_nmse(
        st.nominal,sets,np.random.default_rng(paired_seed),float(cfg.get('snr_db',20.0)),trials),
        warmups=warm,repeats=reps_rt)
    row['reconstruction_runtime_median5']=rt
    return row


def run_sequence(cfg,seed,row_sigma,sequence='recurrent',quick=False,policies=None):
    Ns=list(map(int,cfg['N'])); Ks=list(map(int,cfg['K'])); L=_budget(cfg); lower=_lower(cfg)
    S=int(cfg.get('quick_tasks',4) if quick else cfg['tasks'])
    Jtest=int(cfg.get('quick_j_test',8) if quick else cfg['j_test'])
    tasks=make_sequence(Ns,Ks,S,seed,float(row_sigma),sequence,float(cfg['drift']),
                        cfg['design_scales'],Jtest,float(cfg['test_max']))
    caps=caps_from_cfg(cfg['regularity_caps']); rows=[]
    sampling_stride=int(cfg.get('sampling_stride',1))
    runtime_tasks=set(map(int,cfg.get('quick_runtime_tasks',[0]) if quick else cfg.get('runtime_tasks',[9,49,89])))
    rt_warm=int(cfg.get('runtime_warmups',1)); rt_repeats=int(cfg.get('runtime_repeats',5))

    # Sampling-quality track: Greedy-FP, FFW, and RFW share exactly the same
    # additive total mode-index budget and shared operational cardinality floors.
    for s,st in enumerate(tasks):
        if s % sampling_stride:
            continue
        quality_task=build_task_from_synthetic(st,L,lower,float(cfg['mu']),float(cfg['u0']),float(cfg['eta_c']))
        methods=[
            ('Greedy-FP-2019',lambda:greedy_fp_original(st.nominal,L,[lower[r]-Ks[r] for r in range(len(Ks))]),lambda:greedy_fp_original(st.nominal,L,[lower[r]-Ks[r] for r in range(len(Ks))]),'total_budget_shared'),
            ('FFW-2026',lambda:ffw_original_total_budget(st.nominal,L,lower),lambda:ffw_original_total_budget(st.nominal,L,lower),'total_budget_shared'),
        ]
        for name,fn,bfn,feasible_set in methods:
            t=time.perf_counter(); sets=fn(); dt=time.perf_counter()-t
            med=np.nan
            if s in runtime_tasks:
                med,_=median_runtime(bfn,warmups=rt_warm,repeats=rt_repeats)
            row={'kind':'sampling','seed':seed,'task':s,'row_sigma':float(row_sigma),'sequence':sequence,
                 'method':name,'runtime':dt,'optimization_runtime_median5':med,
                 'feasible_set':feasible_set,'sets':_sets_json(sets),**eval_structured(st,sets,robust_task=quality_task)}
            rows.append(_maybe_empirical(row,st,sets,cfg,seed,s,quick))

        # Nominal-RFW isolates robustification from memory while retaining adaptive allocation.
        nt=_nominal_task(st,L,lower,cfg)
        certn=build_task_certificate(nt,float(cfg['epsilon_abs']),float(cfg['w_lmo']),float(cfg['w_fo']),caps)
        _,nmeta=certn.cold(); nr,exec_status=_solve(nt,certn,nmeta,cfg,quick=quick)
        nmed=np.nan
        if s in runtime_tasks:
            nmed,_=median_runtime(lambda:_solve(nt,certn,nmeta,cfg,quick=quick),warmups=rt_warm,repeats=rt_repeats)
        row={'kind':'sampling','seed':seed,'task':s,'row_sigma':float(row_sigma),'sequence':sequence,
             'method':'Nominal-RFW','runtime':nr.runtime,'optimization_runtime_median5':nmed,
             'total_runtime_median5':nmed,'execution_status':exec_status,'feasible_set':'total_budget_shared',
             'fractional_allocation':_alloc_json(nt.allocation(nr.x)),'sets':_sets_json(nr.sets),
             'lmo':nr.lmo,'fo':nr.fo,
             'realized_weighted_cost':float(cfg['w_lmo'])*nr.lmo+float(cfg['w_fo'])*nr.fo,
             **solver_diagnostics(nr),**eval_structured(st,nr.sets,robust_task=quality_task)}
        rows.append(_maybe_empirical(row,st,nr.sets,cfg,seed,s,quick))

    # Solver-efficiency track: all policies solve the same total-budget robust task sequence.
    policies=list(policies or cfg.get('policies',['cold','prev','nearest','vpm']))
    for policy in policies:
        memory=[]; rng=np.random.default_rng(int(seed)+777)
        cumulative=0.0; cumulative_lmo=0; cumulative_fo=0; cumulative_solver_runtime=0.0; cumulative_total_runtime=0.0
        for s,st in enumerate(tasks):
            task=build_task_from_synthetic(st,L,lower,float(cfg['mu']),float(cfg['u0']),float(cfg['eta_c']))
            c0=time.perf_counter()
            cert=build_task_certificate(task,float(cfg['epsilon_abs']),float(cfg['w_lmo']),float(cfg['w_fo']),caps)
            cert_build_time=time.perf_counter()-c0
            s0=time.perf_counter(); screened=None
            if str(cfg.get('solver_execution','practical')).lower() in ('certified','paper_exact') and policy=='vpm':
                screened,meta=stationarity_screen(task,memory,cert)
            if screened is None:
                item,meta=select_action(policy,task,memory,cert,rng)
            else:
                item=screened
            selection_time=time.perf_counter()-s0
            if not meta.get('eligible',True):
                raise RuntimeError(f"No finite action for task {s}, policy {policy}: {meta}")
            x0=z0=None
            if item is not None: x0,z0=item.x,item.z
            initial_alloc=task.allocation(task.uniform_x() if x0 is None else x0)
            if meta.get('stationarity_screened',False):
                res=_screened_result(task,item,cert,meta); exec_status='theorem_certified_stationarity_screen'
            else:
                res,exec_status=_solve(task,cert,meta,cfg,x0,z0,quick=quick)
            pre_fallback_sets=res.sets
            pre_fallback_stability=design_min_relative_singular_value(st,pre_fallback_sets)
            stability_fallback=0; fallback_stability=np.nan
            threshold=float(cfg.get('min_relative_singular_value',0.01))
            if policy=='vpm' and bool(cfg.get('stability_guard',False)) and pre_fallback_stability<threshold:
                cold_item,cold_meta=select_action('cold',task,[],cert,rng)
                cold_res,cold_status=_solve(task,cert,cold_meta,cfg,None,None,quick=quick)
                fallback_stability=design_min_relative_singular_value(st,cold_res.sets)
                # The choice is made only from design scenarios; held-out test metrics
                # are never inspected. Even when neither candidate reaches the absolute
                # threshold, retain the numerically safer of the two.
                if fallback_stability>pre_fallback_stability:
                    res,exec_status,meta,item=cold_res,cold_status,cold_meta,cold_item
                    initial_alloc=task.allocation(task.uniform_x())
                    stability_fallback=1
            final_alloc=task.allocation(res.x); rounded_alloc=allocation_from_sets(res.sets).astype(float)
            cert_med=sel_med=solve_med=total_med=np.nan
            if s in runtime_tasks:
                bench_rng=np.random.default_rng(int(seed)*1_000_003+s+31)
                cert_med,_=median_runtime(lambda:build_task_certificate(task,float(cfg['epsilon_abs']),float(cfg['w_lmo']),float(cfg['w_fo']),caps),warmups=rt_warm,repeats=rt_repeats)
                sel_med,_=median_runtime(lambda:select_action(policy,task,memory,cert,bench_rng),warmups=rt_warm,repeats=rt_repeats)
                solve_med,_=median_runtime(lambda:_solve(task,cert,meta,cfg,x0,z0,quick=quick),warmups=rt_warm,repeats=rt_repeats)
                total_med=cert_med+sel_med+solve_med
            realized=float(cfg['w_lmo'])*res.lmo+float(cfg['w_fo'])*res.fo
            cumulative+=realized; cumulative_lmo+=int(res.lmo); cumulative_fo+=int(res.fo)
            cumulative_solver_runtime+=float(res.runtime); cumulative_total_runtime+=float(res.runtime)+float(cert_build_time)+float(selection_time)
            score=float(meta.get('cert_cost',np.nan))
            row={
                'kind':'solver','seed':seed,'task':s,'row_sigma':float(row_sigma),'sequence':sequence,
                'method':POLICY_NAMES[policy],'lmo':res.lmo,'fo':res.fo,'outer':res.outer,
                **solver_diagnostics(res),
                'fw_gap_cert':res.fw_gap_cert,'fw_gap_approx':res.fw_gap_approx,'runtime':res.runtime,
                'certificate_time':cert_build_time,'certificate_build_time':cert_build_time,
                'selection_time':selection_time,'selection_overhead_time':selection_time,
                'certificate_build_runtime_median5':cert_med,'selection_certificate_runtime_median5':sel_med,
                'optimization_runtime_median5':solve_med,'total_runtime_median5':total_med,
                'solver_end_to_end_runtime':float(cert_build_time+selection_time+res.runtime),
                'realized_weighted_cost':realized,'cumulative_cost':cumulative,
                'cumulative_lmo':cumulative_lmo,'cumulative_fo':cumulative_fo,
                'cumulative_solver_runtime':cumulative_solver_runtime,'cumulative_total_runtime':cumulative_total_runtime,
                'inner_converged':int(res.inner_converged),'configured_T':res.configured_T,'gamma':res.gamma,
                'execution_status':exec_status,'memory_size':len(memory),'selected_history':(-1 if item is None else item.task_id),
                'initial_allocation':_alloc_json(initial_alloc),'fractional_allocation':_alloc_json(final_alloc),
                'rounded_allocation':_alloc_json(rounded_alloc),
                'allocation_revision_l1':float(np.sum(np.abs(final_alloc-initial_alloc))),
                'stability_guard_enabled':int(bool(cfg.get('stability_guard',False))),
                'stability_threshold':threshold,
                'pre_fallback_design_min_relative_singular':pre_fallback_stability,
                'fallback_design_min_relative_singular':fallback_stability,
                'design_min_relative_singular':design_min_relative_singular_value(st,res.sets),
                'stability_fallback':stability_fallback,
                'pre_fallback_sets':_sets_json(pre_fallback_sets),
                'score_or_cert_cost':score,'score_minus_realized':score-realized if np.isfinite(score) else np.nan,
                'score_upper_bound_holds':(int(score+1e-12>=realized) if (np.isfinite(score) and cert.caps.theorem_certified) else np.nan),
                'robust_objective':float(np.max(task.Fvec(res.x))),
                'sets':_sets_json(res.sets),**meta,**eval_structured(st,res.sets,robust_task=task),
            }
            rows.append(_maybe_empirical(row,st,res.sets,cfg,seed,s,quick))
            memory.append(terminal_memory_item(s,task,res,cert,
                certified_execution=exec_status.startswith('theorem_certified_execution')))
    return pd.DataFrame(rows)


def run_hindsight_diagnostic(cfg,seed,row_sigma=0.75,quick=False):
    Ns=list(map(int,cfg['N'])); Ks=list(map(int,cfg['K'])); L=_budget(cfg); lower=_lower(cfg)
    S=min(int(cfg.get('quick_tasks',4) if quick else cfg.get('hindsight_tasks',20)),int(cfg['tasks']))
    tasks=make_sequence(Ns,Ks,S,seed,float(row_sigma),'recurrent',float(cfg['drift']),cfg['design_scales'],
                        int(cfg.get('quick_j_test',8) if quick else cfg['j_test']),float(cfg['test_max']))
    caps=caps_from_cfg(cfg['regularity_caps']); memory=[]; rows=[]
    for s,st in enumerate(tasks):
        task=build_task_from_synthetic(st,L,lower,float(cfg['mu']),float(cfg['u0']),float(cfg['eta_c']))
        cert=build_task_certificate(task,float(cfg['epsilon_abs']),float(cfg['w_lmo']),float(cfg['w_fo']),caps)
        practical=cert.caps.status!='theorem_certified'
        cold_c,cold_m=(practical_cold_score(task,cert) if practical else cold_certificate(cert)); candidates=[(None,cold_c,cold_m)]
        for it in memory:
            c,m=(practical_history_score(task,it,cert) if practical else history_certificate(task,it,cert)); candidates.append((it,c,m))
        realized=[]
        for it,c,m in candidates:
            if not np.isfinite(c): continue
            res,_=_solve(task,cert,m,cfg,None if it is None else it.x,None if it is None else it.z,quick=quick)
            rc=float(cfg['w_lmo'])*res.lmo+float(cfg['w_fo'])*res.fo
            realized.append((it,c,m,res,rc))
            rows.append({'seed':seed,'task':s,'history':-1 if it is None else it.task_id,
                         'history_allocation':None if it is None else _alloc_json(it.allocation),
                         'score_or_cert_cost':c,'realized_cost':rc,'is_cold':int(it is None)})
        if not realized: raise RuntimeError(f'no eligible action at task {s}')
        it,c,m,res,rc=min(realized,key=lambda q:(q[1],-1 if q[0] is None else q[0].task_id))
        memory.append(terminal_memory_item(s,task,res,cert,certified_execution=False))
        oracle=min(q[4] for q in realized); vpm_real=rc
        for row in rows:
            if row['task']==s:
                row['realized_oracle']=oracle; row['vpm_selected_realized']=vpm_real
                row['vpm_to_oracle_ratio']=vpm_real/oracle if oracle>0 else np.nan
    return pd.DataFrame(rows)
