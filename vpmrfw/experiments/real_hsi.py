from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from skimage.metrics import structural_similarity

from vpmrfw.data.hsi import load_hsi, rank_factors_from_band, validate_no_leakage
from vpmrfw.robust.problem import RobustTask
from vpmrfw.robust.certificate import build_task_certificate
from vpmrfw.robust.memory import terminal_memory_item, select_action, stationarity_screen
from vpmrfw.utils.config import caps_from_cfg
from vpmrfw.baselines.greedy_fp_ortiz2019 import greedy_fp_original
from vpmrfw.baselines.ffw_li2026 import ffw_original_total_budget
from vpmrfw.core.metrics import nmse_db, psnr, structured_theoretical_mse, structured_full_rank
from vpmrfw.core.fp import tensor_fp_discrete
from vpmrfw.core.simplex import allocation_from_sets
from vpmrfw.experiments.synthetic_main import _solve, _screened_result, POLICY_NAMES, solver_diagnostics


_CUBE_CACHE = {}
_TRUE_FACTOR_CACHE = {}


def _load_hsi_cached(name, data_dir):
    key=(str(name),str(Path(data_dir).resolve()))
    if key not in _CUBE_CACHE:
        _CUBE_CACHE[key]=load_hsi(name,data_dir)
    return _CUBE_CACHE[key]


def _lowrank_factors(U, G, V, rank):
    """Exact rank factors of U G V^T using only thin QR + a small SVD."""
    Qu,Ru=np.linalg.qr(np.asarray(U,dtype=np.float64),mode='reduced')
    Qv,Rv=np.linalg.qr(np.asarray(V,dtype=np.float64),mode='reduced')
    M=Ru@np.asarray(G,dtype=np.float64)@Rv.T
    Pu,s,Pvt=np.linalg.svd(M,full_matrices=False)
    k=min(int(rank),len(s)); root=np.sqrt(np.maximum(s[:k],0.0))
    return [Qu@Pu[:,:k]*root[None,:], Qv@Pvt[:k,:].T*root[None,:]], s


def _scaled_true_factors(name,data_dir,cube,band_index,rank,scale):
    key=(str(name),str(Path(data_dir).resolve()),int(band_index),int(rank),float(scale))
    if key not in _TRUE_FACTOR_CACHE:
        _TRUE_FACTOR_CACHE[key]=rank_factors_from_band(
            np.asarray(cube[:,:,band_index],dtype=np.float64)/scale,rank
        )[0]
    return _TRUE_FACTOR_CACHE[key]


def _ridge_core(A, B, obs, lam, prior_core):
    """Solve min_G ||A G B^T-obs||_F^2 + lam ||G-prior_core||_F^2.

    The normal equation is diagonalized with two rank-by-rank eigendecompositions,
    so this is much cheaper and more stable than forming the Kronecker design.
    """
    A=np.asarray(A,float); B=np.asarray(B,float); obs=np.asarray(obs,float)
    prior_core=np.asarray(prior_core,float)
    C=A.T@A; D=B.T@B; rhs=A.T@obs@B
    ac,P=np.linalg.eigh(C); bd,Q=np.linalg.eigh(D)
    ac=np.maximum(ac,0.0); bd=np.maximum(bd,0.0)
    lam=float(max(lam,0.0))
    den=ac[:,None]*bd[None,:]+lam
    # lam>0 in the regularized branch. The guard is only for roundoff.
    den=np.where(den>1e-30,den,np.inf)
    R=P.T@(rhs+lam*prior_core)@Q
    G=P@(R/den)@Q.T
    pmin=float(np.min(ac)*np.min(bd)); pmax=float(np.max(ac)*np.max(bd))
    return G,pmin,pmax


def reconstruct_matrix_from_field(
    band, factors, sets, noise_field, *, trust_radius=None, rank_tol=1e-10, return_core=False, return_info=False
):
    """Leakage-free stabilized reconstruction from Cartesian samples.

    The historical nominal model is U I V^T because the singular values are
    absorbed into U and V.  We first compute the ordinary LS core.  If its
    update from I is larger than the *history-calibrated* trust radius, we find
    the smallest Tikhonov parameter lambda such that

        ||G_lambda - I||_F <= trust_radius.

    This prevents ill-conditioned sample submatrices from producing arbitrary
    extrapolation while still allowing the current sampled pixels to update the
    historical prediction.  The same rule is used for every sampling method and
    uses no unobserved current-band pixels.
    """
    r,c=sets; U,V=[np.asarray(x,float) for x in factors]
    A=U[np.asarray(r,dtype=int),:]; B=V[np.asarray(c,dtype=int),:]
    if not structured_full_rank(factors,sets,rank_tol=float(rank_tol)):
        raise np.linalg.LinAlgError('selected HSI factor is rank deficient under the configured relative tolerance')
    obs=np.asarray(band,float)[np.ix_(r,c)]+np.asarray(noise_field,float)[np.ix_(r,c)]
    prior=np.eye(U.shape[1],dtype=float)

    G_ls=np.linalg.pinv(A)@obs@np.linalg.pinv(B.T)
    update_ls=float(np.linalg.norm(G_ls-prior,'fro'))
    lam=0.0; G=G_ls
    pmin=pmax=np.nan

    if trust_radius is not None:
        radius=float(trust_radius)
        if not np.isfinite(radius) or radius<=0:
            raise ValueError('trust_radius must be finite and positive')
        if (not np.isfinite(update_ls)) or update_ls>radius:
            # Find the least amount of prior regularization needed to satisfy
            # the history-calibrated core-change trust region.
            _,pmin,pmax=_ridge_core(A,B,obs,1.0,prior)
            hi=max(float(pmax),1e-12)
            Ghi,_,_=_ridge_core(A,B,obs,hi,prior)
            for _ in range(80):
                if float(np.linalg.norm(Ghi-prior,'fro'))<=radius:
                    break
                hi*=10.0
                Ghi,_,_=_ridge_core(A,B,obs,hi,prior)
            else:
                raise np.linalg.LinAlgError('failed to regularize the HSI core into its trust region')
            lo=0.0
            for _ in range(70):
                mid=0.5*(lo+hi)
                Gmid,_,_=_ridge_core(A,B,obs,mid,prior)
                if float(np.linalg.norm(Gmid-prior,'fro'))>radius:
                    lo=mid
                else:
                    hi=mid; Ghi=Gmid
            lam=hi; G=Ghi
        else:
            # Diagnostics for an unregularized but admissible update.
            C=A.T@A; D=B.T@B
            ac=np.linalg.eigvalsh(C); bd=np.linalg.eigvalsh(D)
            pmin=float(max(np.min(ac),0.0)*max(np.min(bd),0.0))
            pmax=float(max(np.max(ac),0.0)*max(np.max(bd),0.0))

    rec=U@G@V.T
    info={
        'reconstruction_lambda':float(lam),
        'core_update_norm':float(np.linalg.norm(G-prior,'fro')),
        'unregularized_core_update_norm':update_ls,
        'sample_factor_cond_r1':float(np.linalg.cond(A)),
        'sample_factor_cond_r2':float(np.linalg.cond(B)),
        'regularized_normal_condition':float((pmax+lam)/(pmin+lam)) if np.isfinite(pmax) and np.isfinite(pmin) and (pmin+lam)>0 else np.nan,
    }
    if return_core and return_info:
        return rec,G,info
    if return_core:
        return rec,G
    if return_info:
        return rec,info
    return rec


def _make_task(history_factors,total_budget,lower,cfg):
    recent=list(history_factors[-int(cfg['history_scenarios']):])
    if not recent: raise ValueError('history_factors cannot be empty')
    # Scenario 0 is the most recent available model (task s-1).
    scenarios=[recent[-1]]+list(reversed(recent[:-1]))
    while len(scenarios)<int(cfg['history_scenarios']): scenarios.append(scenarios[-1])
    tmp=RobustTask(scenarios,total_budget,lower,float(cfg['mu']),float(cfg['u0']),float(cfg['eta_c']))
    scale=max(1.0,float(np.max(tmp.Fvec(tmp.uniform_x()))))
    return RobustTask(scenarios,total_budget,lower,float(cfg['mu']),float(cfg['u0']),float(cfg['eta_c']),scale=scale)


def _nominal_task(nom,total_budget,lower,cfg):
    t=RobustTask([nom],total_budget,lower,float(cfg['mu']),1.1,0.0)
    ref=max(1.0,float(np.max(t.Fvec(t.uniform_x()))))
    return RobustTask([nom],total_budget,lower,float(cfg['mu']),1.1,0.0,scale=ref)


def _sets_json(sets):
    return json.dumps([np.asarray(s,dtype=int).tolist() for s in sets],separators=(',',':'))


def _alloc_json(vals):
    return json.dumps([float(v) for v in vals],separators=(',',':'))


def _design_meta(sets):
    a=allocation_from_sets(sets)
    out={'mode_cardinalities':json.dumps(a.astype(int).tolist(),separators=(',',':')),
         'total_mode_budget':int(a.sum()),'cartesian_observations':int(np.prod(a,dtype=np.int64))}
    out.update({f'mode_budget_r{r+1}':int(v) for r,v in enumerate(a)})
    return out


def _split_count(total, parts):
    total=int(total); parts=int(parts)
    q,r=divmod(total,parts)
    return [q+(1 if i<r else 0) for i in range(parts)]


def _evaluation_band_windows(total_bands, warmup, cfg, dataset, quick=False):
    """Return independent contiguous spectral windows.

    Each window consists of `warmup` fully historical bands immediately followed
    by contiguous evaluation bands.  Window histories/memories are reset, so
    widely separated spectral regions are never treated as adjacent tasks.
    """
    B=int(total_bands); warm=int(warmup)
    if warm<2 or warm>=B:
        raise ValueError('HSI warmup must satisfy 2 <= warmup < number of bands')
    if quick:
        count=int(cfg.get('quick_bands',2)); nwin=int(cfg.get('quick_windows',1))
    else:
        by=cfg.get('evaluation_bands_by_dataset',{}) or {}
        count=int(by.get(dataset,cfg.get('evaluation_bands',B-warm)))
        wb=cfg.get('evaluation_windows_by_dataset',{}) or {}
        nwin=int(wb.get(dataset,cfg.get('evaluation_windows',1)))
    count=max(1,min(count,B-warm)); nwin=max(1,min(nwin,count))
    mode=str(cfg.get('band_selection','contiguous_windows')).lower()
    if mode!='contiguous_windows':
        raise ValueError(f"unsupported band_selection={mode!r}; use 'contiguous_windows'")
    lengths=_split_count(count,nwin)
    max_len=max(lengths)
    latest=B-max_len
    if latest<warm:
        raise ValueError('not enough spectral bands for requested contiguous-window evaluation')
    frac_map=cfg.get('evaluation_window_fractions_by_dataset',{}) or {}
    fractions=frac_map.get(dataset) if not quick else None
    if fractions is not None:
        fractions=[float(x) for x in fractions]
        if len(fractions)!=nwin or any((x<0.0 or x>1.0) for x in fractions):
            raise ValueError('evaluation window fractions must match the number of windows and lie in [0,1]')
        targets=np.asarray([warm+x*(latest-warm) for x in fractions],dtype=float)
    else:
        targets=np.linspace(warm,latest,num=nwin)
    windows=[]; occupied=[]
    for wid,(t,length) in enumerate(zip(targets,lengths)):
        start=int(round(float(t)))
        start=max(warm,min(start,B-int(length)))
        hist=list(range(start-warm,start)); ev=list(range(start,start+int(length)))
        span=set(hist+ev)
        if any(span.intersection(old) for old in occupied):
            raise ValueError('requested HSI evaluation windows overlap; reduce bands/windows or warmup')
        occupied.append(span)
        windows.append({'window_id':wid,'warmup_bands':hist,'eval_bands':ev})
    return windows


def _evaluation_band_indices(total_bands, warmup, cfg, dataset, quick=False):
    """Flattened compatibility view of `_evaluation_band_windows`."""
    return [s for w in _evaluation_band_windows(total_bands,warmup,cfg,dataset,quick=quick) for s in w['eval_bands']]


def _calibrate_core_trust_radius(cube, warmup_bands, rank, scale, cfg):
    """Calibrate a method-independent core-change radius from historical bands only."""
    vals=[]; warmup_bands=list(map(int,warmup_bands))
    for a,b in zip(warmup_bands[:-1],warmup_bands[1:]):
        fac,_=rank_factors_from_band(np.asarray(cube[:,:,a],dtype=np.float64)/scale,rank)
        U,V=fac; X=np.asarray(cube[:,:,b],dtype=np.float64)/scale
        G=np.linalg.pinv(U)@X@np.linalg.pinv(V.T)
        d=float(np.linalg.norm(G-np.eye(rank),'fro'))
        if np.isfinite(d): vals.append(d)
    if not vals:
        raise ValueError('at least two warmup bands are required to calibrate HSI reconstruction')
    q=float(cfg.get('reconstruction_trust_quantile',0.5))
    if not (0.0<=q<=1.0): raise ValueError('reconstruction_trust_quantile must lie in [0,1]')
    radius=float(np.quantile(np.asarray(vals,float),q))
    lo=float(cfg.get('reconstruction_trust_min',1e-3)); hi=float(cfg.get('reconstruction_trust_max',np.inf))
    radius=float(np.clip(radius,lo,hi))
    return radius,vals


def _initialize_window_history(cube, warmup_bands, rank, max_rank, scale, ranks, window_id):
    rank_rows=[]; history=[]; band_history=[]
    for s0 in warmup_bands:
        fac_max,sv=rank_factors_from_band(np.asarray(cube[:,:,s0],dtype=np.float64)/scale,max_rank)
        den=float(np.sum(sv*sv))
        for k0 in ranks:
            rank_rows.append({'window_id':int(window_id),'warmup_band':int(s0),'rank':int(k0),
                              'retained_energy':float(np.sum(sv[:int(k0)]**2)/max(den,1e-30))})
        history.append([fac_max[0][:,:rank].copy(),fac_max[1][:,:rank].copy()]); band_history.append(int(s0))
    return history,band_history,rank_rows


def run_dataset(name,cfg,data_dir,quick=False,history_mode='closed_loop',history_driver_method='VPM-RFW',evaluation_methods=None):
    """Leakage-free sequential HSI experiment with adaptive cross-mode allocation."""
    cube=_load_hsi_cached(name,data_dir); H,W,B=cube.shape
    rank=int(cfg['rank']); floor=int(cfg.get('mode_budget_floor',rank))
    lower=[max(rank,floor),max(rank,floor)]; total_budget=int(cfg['total_budget']); warm=int(cfg['warmup'])
    if total_budget<sum(lower) or total_budget>H+W:
        raise ValueError('HSI total budget is infeasible for the rank/cardinality floors')
    caps=caps_from_cfg(cfg['regularity_caps'])
    # One dataset-wide physical scale; evaluation-window choice never changes it.
    scale=float(np.percentile(np.abs(cube[:,:,:warm]),99.5)); scale=max(scale,1e-12)
    ranks=[int(k) for k in cfg.get('rank_sweep',[rank])]; max_rank=max([rank,*ranks])
    windows=_evaluation_band_windows(B,warm,cfg,name,quick=quick)

    rows=[]; rank_rows=[]
    policies=list(cfg.get('policies',['cold','prev','nearest','vpm']))
    all_method_names=['Greedy-FP-2019','FFW-2026','Nominal-RFW']+[POLICY_NAMES[p] for p in policies]
    if evaluation_methods is None:
        method_names=list(all_method_names)
    else:
        requested=list(dict.fromkeys(map(str,evaluation_methods)))
        unknown=[m for m in requested if m not in all_method_names]
        if unknown:
            raise ValueError(f'unknown HSI evaluation methods: {unknown}')
        method_names=requested
    if history_mode=='closed_loop' and history_driver_method not in method_names:
        raise ValueError(f'history_driver_method={history_driver_method!r} is not among evaluated methods')
    active_policies=[p for p in policies if POLICY_NAMES[p] in method_names]
    rank_tol=float(cfg.get('rank_tol',1e-10))
    true2=np.zeros((H,W),dtype=np.float64)
    sam_dot={m:np.zeros((H,W),dtype=np.float64) for m in method_names}
    sam_rec2={m:np.zeros((H,W),dtype=np.float64) for m in method_names}
    all_eval_bands=[]

    for win in windows:
        wid=int(win['window_id']); warm_ids=list(win['warmup_bands']); eval_ids=list(win['eval_bands'])
        all_eval_bands.extend(eval_ids)
        model_history,model_band_history,rr=_initialize_window_history(cube,warm_ids,rank,max_rank,scale,ranks,wid)
        rank_rows.extend(rr)
        memories={p:[] for p in active_policies}
        trust_radius,trust_samples=_calibrate_core_trust_radius(cube,warm_ids,rank,scale,cfg)

        for s in eval_ids:
            design_ids=list(model_band_history[-int(cfg['history_scenarios']):]); validate_no_leakage(design_ids,s)
            task=_make_task(model_history,total_budget,lower,cfg); nom=model_history[-1]
            tc=time.perf_counter()
            cert=build_task_certificate(task,float(cfg['epsilon_abs']),float(cfg['w_lmo']),float(cfg['w_fo']),caps)
            cert_build_time=time.perf_counter()-tc
            designs={}
            if 'Greedy-FP-2019' in method_names:
                t=time.perf_counter(); ss=greedy_fp_original(nom,total_budget,alpha=[lower[r]-nom[r].shape[1] for r in range(len(lower))])
                designs['Greedy-FP-2019']=(ss,time.perf_counter()-t,None)
            if 'FFW-2026' in method_names:
                t=time.perf_counter(); ss=ffw_original_total_budget(nom,total_budget,lower)
                designs['FFW-2026']=(ss,time.perf_counter()-t,None)
            if 'Nominal-RFW' in method_names:
                nt=_nominal_task(nom,total_budget,lower,cfg)
                tn=time.perf_counter(); certn=build_task_certificate(nt,float(cfg['epsilon_abs']),float(cfg['w_lmo']),float(cfg['w_fo']),caps)
                nominal_cert_build_time=time.perf_counter()-tn
                _,nm=certn.cold(); nr,nes=_solve(nt,certn,nm,cfg,quick=quick)
                designs['Nominal-RFW']=(nr.sets,nr.runtime,{'lmo':nr.lmo,'fo':nr.fo,'execution_status':nes,
                                                            'realized_weighted_cost':float(cfg['w_lmo'])*nr.lmo+float(cfg['w_fo'])*nr.fo,
                                                            'certificate_build_time':nominal_cert_build_time,'selection_time':0.0,
                                                            'fractional_allocation':_alloc_json(nt.allocation(nr.x)),
                                                            **solver_diagnostics(nr)})
            for p in active_policies:
                c0=time.perf_counter(); screened=None
                if str(cfg.get('solver_execution','practical')).lower() in ('certified','paper_exact') and p=='vpm':
                    screened,meta=stationarity_screen(task,memories[p],cert)
                if screened is None:
                    item,meta=select_action(p,task,memories[p],cert,np.random.default_rng(s+int(cfg['seed'])))
                else:
                    item=screened
                ctime=time.perf_counter()-c0
                if not meta.get('eligible',True): raise RuntimeError(f'No finite HSI action: band={s}, policy={p}, meta={meta}')
                x0=None if item is None else item.x; z0=None if item is None else item.z
                initial_alloc=task.allocation(task.uniform_x() if x0 is None else x0)
                if meta.get('stationarity_screened',False):
                    res=_screened_result(task,item,cert,meta); es='theorem_certified_stationarity_screen'
                else:
                    res,es=_solve(task,cert,meta,cfg,x0,z0,quick=quick)
                method=POLICY_NAMES[p]; score=float(meta.get('cert_cost',np.nan)); realized=float(cfg['w_lmo'])*res.lmo+float(cfg['w_fo'])*res.fo
                final_alloc=task.allocation(res.x)
                designs[method]=(res.sets,res.runtime,{'lmo':res.lmo,'fo':res.fo,'fw_gap_cert':res.fw_gap_cert,
                    'fw_gap_approx':res.fw_gap_approx,'certificate_time':cert_build_time,
                    'certificate_build_time':cert_build_time,'selection_time':ctime,'selection_overhead_time':ctime,
                    'memory_size':len(memories[p]),'selected_history':-1 if item is None else item.task_id,
                    'execution_status':es,'score_or_cert_cost':score,'realized_weighted_cost':realized,
                    **solver_diagnostics(res),
                    'score_upper_bound_holds':(int(score+1e-12>=realized) if (np.isfinite(score) and cert.caps.theorem_certified) else np.nan),
                    'initial_allocation':_alloc_json(initial_alloc),'fractional_allocation':_alloc_json(final_alloc),
                    'allocation_revision_l1':float(np.sum(np.abs(final_alloc-initial_alloc))),**meta})
                memories[p].append(terminal_memory_item(s,task,res,cert,certified_execution=es.startswith('theorem_certified_execution')))

            # Current truth enters only after every sampling decision has been frozen.
            band=np.asarray(cube[:,:,s],dtype=np.float64)/scale; true2 += band*band
            rng_noise=np.random.default_rng(int(cfg['seed'])*1000000+s)
            noise_var=float(np.mean(band*band))/(10.0**(float(cfg['snr_db'])/10.0))
            noise_field=rng_noise.normal(0,np.sqrt(max(noise_var,1e-20)),size=band.shape)
            truefac=_scaled_true_factors(name,data_dir,cube,s,rank,scale)
            driver_rec=None; driver_core=None; driver_nom=None
            for method,(sets,rt,extra) in designs.items():
                nominal_rank_ok=structured_full_rank(nom,sets,rank_tol=rank_tol)
                rec=None; rec_rt=np.nan; rinfo={}
                if nominal_rank_ok:
                    tr0=time.perf_counter()
                    try:
                        rec,G,rinfo=reconstruct_matrix_from_field(
                            band,nom,sets,noise_field,trust_radius=trust_radius,rank_tol=rank_tol,
                            return_core=True,return_info=True
                        )
                    except np.linalg.LinAlgError:
                        rec=None; G=None
                    rec_rt=time.perf_counter()-tr0
                dr=max(float(band.max()-band.min()),1e-12)
                if rec is None:
                    nd=ps=ssim=np.nan
                else:
                    sam_dot[method]+=band*rec; sam_rec2[method]+=rec*rec
                    nd=nmse_db(rec,band); ps=psnr(rec,band,dr); ssim=float(structural_similarity(band,rec,data_range=dr))
                robust_obj=float(np.max([tensor_fp_discrete(fac,sets)/float(task.scale) for fac in task.scenarios]))
                row={'dataset':name,'band':s,'window_id':wid,'history_mode':history_mode,
                     'history_driver_method':history_driver_method if history_mode=='closed_loop' else 'oracle_history',
                     'window_warmup_start':int(warm_ids[0]),'window_warmup_end':int(warm_ids[-1]),
                     'core_trust_radius':float(trust_radius),'core_trust_calibration_n':int(len(trust_samples)),
                     'kind':'solver' if method in [POLICY_NAMES[p] for p in active_policies] else 'sampling',
                     'method':method,'nmse_db':nd,'psnr':ps,'ssim':ssim,'runtime':rt,'optimization_runtime':rt,
                     'reconstruction_runtime':rec_rt,'post_design_mse':structured_theoretical_mse(truefac,sets,1.0,rank_tol=rank_tol),
                     'robust_design_objective':robust_obj,'nominal_rank_feasible':int(nominal_rank_ok),
                     'sets':_sets_json(sets),'snr_db':float(cfg['snr_db']),'rank':rank,'rank_tol':rank_tol,
                     'configured_total_budget':total_budget,**_design_meta(sets),**rinfo}
                if extra: row.update(extra)
                row.setdefault('certificate_build_time',0.0); row.setdefault('selection_time',0.0); row.setdefault('selection_overhead_time',row['selection_time'])
                row['solver_end_to_end_runtime']=float(row['certificate_build_time']+row['selection_time']+rt)
                row['end_to_end_runtime']=float(row['solver_end_to_end_runtime']+(rec_rt if np.isfinite(rec_rt) else 0.0))
                rows.append(row)
                if method==history_driver_method:
                    driver_rec=rec; driver_core=(G if rec is not None else None); driver_nom=nom
            if history_mode=='closed_loop':
                if driver_rec is None:
                    raise RuntimeError(f'{history_driver_method} produced an inadmissible HSI design; closed-loop history cannot proceed')
                if driver_core is None or driver_nom is None: raise RuntimeError(f'missing {history_driver_method} low-rank core')
                fac_next,_=_lowrank_factors(driver_nom[0],driver_core,driver_nom[1],rank)
            elif history_mode=='oracle_history':
                fac_next=truefac
            else:
                raise ValueError("history_mode must be 'closed_loop' or 'oracle_history'")
            model_history.append(fac_next); model_band_history.append(int(s))

    rank_energy=pd.DataFrame(rank_rows)
    summary=[]
    for method in method_names:
        den=np.sqrt(np.maximum(true2*sam_rec2[method],1e-30)); cs=np.clip(sam_dot[method]/den,-1.0,1.0)
        valid=(true2>1e-20)&(sam_rec2[method]>1e-20)
        sam=float(np.mean(np.arccos(cs[valid]))) if np.any(valid) else float('nan')
        summary.append({'dataset':name,'history_mode':history_mode,'history_driver_method':history_driver_method if history_mode=='closed_loop' else 'oracle_history','method':method,'sam_rad':sam,
                        'evaluated_bands':int(len(all_eval_bands)),'evaluation_windows':int(len(windows)),
                        'first_eval_band':int(all_eval_bands[0]) if all_eval_bands else -1,
                        'last_eval_band':int(all_eval_bands[-1]) if all_eval_bands else -1})
    return pd.DataFrame(rows),pd.DataFrame(summary),rank_energy
