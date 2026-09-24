#!/usr/bin/env python3
"""Reproducible entry point for the experiments reported in the paper.

Examples
--------
Quick smoke run::
    python run_experiments.py --suite synthetic --quick

Run all paper experiments with ``python RUN_PAPER.py``.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse, copy, gc, json, os, shutil, subprocess, sys, time
# Word protocol requires identical single-thread BLAS/OpenMP settings for runtime tests.
for _k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[_k]="1"
import numpy as np, pandas as pd, yaml

from vpmrfw.experiments import synthetic_main, synthetic_budget, synthetic_drift, real_hsi, diagnostics
from vpmrfw.data.hsi import write_data_audit
from vpmrfw.plotting.figures import line_by_method, scatter_score_realized, point_by_method, pretty_method_names
from vpmrfw.plotting.style import (MAIN_SOLVER_METHODS, MAIN_TASK_QUALITY_METHODS,
    ALL_SOLVER_METHODS, MAIN_SAMPLING_METHODS, ALL_SAMPLING_METHODS)
from vpmrfw.utils.config import load_config
from vpmrfw.utils.io import environment, source_manifest
from vpmrfw.utils.stats import bootstrap_mean_ci, contiguous_block_bootstrap
from vpmrfw.utils.timing import pin_threads


PROJECT_ROOT = Path(__file__).resolve().parent


def _csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False)


def _solver_convergence_summary(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    methods=['Cold-RFW','Prev-RFW','Nearest-RFW','VPM-RFW']
    if df.empty:
        return pd.DataFrame()
    g=df[df['method'].isin(methods)].copy()
    # Backward compatibility for old result tables/tests.  New experiment rows
    # carry the explicit split fields below; legacy ``certified_converged`` only
    # represented the inner state and is copied to both slots solely so old
    # artifacts remain readable.
    if 'inner_certified_converged' not in g and 'certified_converged' in g:
        g['inner_certified_converged']=g['certified_converged']
    if 'overall_certified_converged' not in g and 'certified_converged' in g:
        g['overall_certified_converged']=g['certified_converged']
    if 'practical_converged' not in g:
        g['practical_converged']=np.nan
    needed={'method','inner_certified_converged','overall_certified_converged','practical_converged',
            'hit_outer_cap','hit_ppa_cap','hit_resolvent_cap','reached_ppa_limit','distance_cert_ratio',
            'num_outer','num_ppa','lmo','fo','realized_weighted_cost'}
    if not needed.issubset(g.columns):
        return pd.DataFrame()
    if g.empty:
        return pd.DataFrame()
    out=(g.groupby('method',sort=False)
         .agg(inner_certified_convergence_rate=('inner_certified_converged','mean'),
              overall_certified_convergence_rate=('overall_certified_converged','mean'),
              practical_convergence_rate=('practical_converged','mean'),
              outer_cap_rate=('hit_outer_cap','mean'),
              ppa_cap_failure_rate=('hit_ppa_cap','mean'),
              resolvent_cap_failure_rate=('hit_resolvent_cap','mean'),
              reached_ppa_limit_rate=('reached_ppa_limit','mean'),
              mean_final_distance_cert_ratio=('distance_cert_ratio','mean'),
              median_final_distance_cert_ratio=('distance_cert_ratio','median'),
              p95_final_distance_cert_ratio=('distance_cert_ratio',lambda x: x.quantile(.95)),
              mean_outer=('num_outer','mean'), mean_ppa=('num_ppa','mean'),
              mean_lmo=('lmo','mean'), mean_fo=('fo','mean'),
              mean_weighted_cost=('realized_weighted_cost','mean'))
         .reset_index())
    # Backward-readable alias now means whole-algorithm certification, not merely inner convergence.
    out['certified_convergence_rate']=out['overall_certified_convergence_rate']
    out.insert(0,'experiment',str(experiment))
    return out


def _paired_bootstrap_mean_ci(df, value_col, unit_col, n_boot=2000, seed=0):
    units=pd.unique(df[unit_col].dropna())
    if len(units)==0:
        return np.nan,np.nan
    rng=np.random.default_rng(int(seed)); vals=[]
    grouped={u:df.loc[df[unit_col]==u,value_col].to_numpy(float) for u in units}
    for _ in range(int(n_boot)):
        sampled=rng.choice(units,size=len(units),replace=True)
        x=np.concatenate([grouped[u] for u in sampled])
        vals.append(float(np.mean(x)))
    return tuple(map(float,np.quantile(vals,[.025,.975])))


def _synthetic_paired_advantage(sol, n_boot=2000):
    rows=[]; keys=['seed','row_sigma','sequence','task']
    v=sol[sol.method=='VPM-RFW'][keys+['realized_weighted_cost','wc_mse']]
    for sigma in sorted(pd.unique(sol['row_sigma'])):
        vs=v[np.isclose(v.row_sigma.astype(float),float(sigma))]
        for baseline in ('Cold-RFW','Prev-RFW','Nearest-RFW'):
            b=sol[(sol.method==baseline)&np.isclose(sol.row_sigma.astype(float),float(sigma))]
            p=vs.merge(b[keys+['realized_weighted_cost','wc_mse']],on=keys,suffixes=('_vpm','_baseline'),validate='one_to_one')
            if p.empty: continue
            p['cost_delta']=p.realized_weighted_cost_vpm-p.realized_weighted_cost_baseline
            p['mse_delta']=p.wc_mse_vpm-p.wc_mse_baseline
            lo,hi=_paired_bootstrap_mean_ci(p,'cost_delta','seed',n_boot=n_boot,seed=0)
            rows.append({'row_sigma':float(sigma),'baseline':baseline,'n_pairs':len(p),
                         'mean_cost_delta':float(p.cost_delta.mean()),
                         'median_cost_delta':float(p.cost_delta.median()),
                         'relative_cost_reduction':float(1-p.realized_weighted_cost_vpm.mean()/max(p.realized_weighted_cost_baseline.mean(),1e-12)),
                         'cost_delta_ci_lo':lo,'cost_delta_ci_hi':hi,
                         'fraction_vpm_lower_cost':float((p.cost_delta<0).mean()),
                         'mean_wc_mse_delta':float(p.mse_delta.mean()),
                         'relative_wc_mse_delta':float(p.mse_delta.mean()/max(abs(p.wc_mse_baseline.mean()),1e-12)),
                         'fraction_vpm_lower_mse':float((p.mse_delta<0).mean()),
                         'fraction_vpm_pareto_better':float(((p.cost_delta<=0)&(p.mse_delta<=0)).mean())})
    return pd.DataFrame(rows)


def _hsi_paired_advantage(closed):
    rows=[]; group=['dataset','history_mode','window_id','method']
    w=(closed[closed['kind']=='solver'].groupby(group,as_index=False)
       .agg(realized_weighted_cost=('realized_weighted_cost','mean'),nmse_db=('nmse_db','mean')))
    v=w[w.method=='VPM-RFW']
    keys=['dataset','history_mode','window_id']
    for baseline in ('Cold-RFW','Prev-RFW','Nearest-RFW'):
        b=w[w.method==baseline]
        p=v.merge(b,on=keys,suffixes=('_vpm','_baseline'),validate='one_to_one')
        if p.empty: continue
        cd=p.realized_weighted_cost_vpm-p.realized_weighted_cost_baseline
        nd=p.nmse_db_vpm-p.nmse_db_baseline
        rows.append({'baseline':baseline,'n_pairs':len(p),
                     'relative_cost_reduction':float(1-p.realized_weighted_cost_vpm.mean()/max(p.realized_weighted_cost_baseline.mean(),1e-12)),
                     'mean_nmse_delta_db':float(nd.mean()),
                     'fraction_vpm_lower_cost':float((cd<0).mean()),
                     'fraction_vpm_better_or_equal_nmse':float((nd<=0).mean()),
                     'fraction_vpm_pareto_better':float(((cd<=0)&(nd<=0)).mean())})
    return pd.DataFrame(rows)


def _history_selection_summary(df, experiment, group_cols, task_col):
    methods=('VPM-RFW','Prev-RFW','Nearest-RFW')
    d=df[df.method.isin(methods)].copy()
    keys=[*group_cols,task_col]
    wide=d.pivot_table(index=keys,columns='method',values=['selected_history','realized_weighted_cost'],aggfunc='first')
    if wide.empty: return pd.DataFrame()
    wide.columns=[f'{a}_{b}' for a,b in wide.columns]; wide=wide.reset_index()
    first=wide.groupby(group_cols)[task_col].transform('min'); eligible=wide[task_col]!=first
    vsel=wide['selected_history_VPM-RFW']; prev=wide['selected_history_Prev-RFW']; near=wide['selected_history_Nearest-RFW']
    dp=eligible&(vsel!=prev); dn=eligible&(vsel!=near)
    return pd.DataFrame([{
        'experiment':experiment,'n_tasks':len(wide),'n_history_eligible':int(eligible.sum()),
        'n_history_selected':int((eligible&(vsel>=0)).sum()),
        'history_selection_rate':float((vsel[eligible]>=0).mean()) if eligible.any() else np.nan,
        'fraction_same_as_prev':float((vsel[eligible]==prev[eligible]).mean()) if eligible.any() else np.nan,
        'fraction_same_as_nearest':float((vsel[eligible]==near[eligible]).mean()) if eligible.any() else np.nan,
        'mean_cost_when_diff_from_prev':float(wide.loc[dp,'realized_weighted_cost_VPM-RFW'].mean()) if dp.any() else np.nan,
        'mean_cost_prev_when_diff':float(wide.loc[dp,'realized_weighted_cost_Prev-RFW'].mean()) if dp.any() else np.nan,
        'mean_cost_when_diff_from_nearest':float(wide.loc[dn,'realized_weighted_cost_VPM-RFW'].mean()) if dn.any() else np.nan,
        'mean_cost_nearest_when_diff':float(wide.loc[dn,'realized_weighted_cost_Nearest-RFW'].mean()) if dn.any() else np.nan,
    }])


def _outdir(cfg,args):
    if args.output: return Path(args.output)
    root=Path(cfg.get('output_dir','results'))
    tag=datetime.now().strftime('%Y%m%d_%H%M%S')
    return root/tag


def _synthetic_main(cfg,out,quick):
    scfg=cfg['synthetic']; seq=str(scfg.get('main_sequence','recurrent'))
    sigmas=list(map(float,scfg.get('main_row_sigmas',[0.0,0.75])))
    nseq=int(scfg.get('quick_sequences',1) if quick else scfg['sequences'])
    frames=[]
    nrt=min(nseq,int(scfg.get('runtime_benchmark_sequences',nseq)))
    for k in range(nseq):
        seed=int(cfg['seed'])+k
        for sig in sigmas:
            print(f'[synthetic-main] seed={seed} sigma={sig:g} sequence={seq}')
            run_cfg=copy.deepcopy(scfg)
            # Runtime medians are a measurement diagnostic, not an independent
            # Monte Carlo outcome. Benchmark only a representative subset of
            # sequences while retaining all sequences for the scientific metrics.
            if not quick and k>=nrt:
                run_cfg['runtime_tasks']=[]
            frames.append(synthetic_main.run_sequence(run_cfg,seed,sig,seq,quick=quick))
    df=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    _csv(df,out/'raw'/'synthetic_main.csv')
    alloc_cols=[c for c in df.columns if c.startswith('mode_budget_r')]
    summary=bootstrap_mean_ci(df,['kind','method','row_sigma','sequence'],
        ['wc_mse','p95_mse','avg_mse','nominal_fp','robust_design_objective','cartesian_observations',*alloc_cols,'runtime','certificate_build_time','selection_time','solver_end_to_end_runtime','optimization_runtime_median5','total_runtime_median5','reconstruction_runtime_median5','lmo','fo','realized_weighted_cost','allocation_revision_l1'],unit_col='seed',n_boot=200 if quick else 2000)
    _csv(pretty_method_names(summary),out/'tables'/'synthetic_main_summary.csv')
    # Explicit sampling-quality comparison.  Sampling baselines are evaluated on
    # a stride for runtime reasons, so restrict every method to the same task
    # indices before comparing MSE.  Independent sequences, not tasks, are the
    # bootstrap units.
    sampled_tasks=set(df.loc[df.method=='FFW-2026','task'].dropna().astype(int).unique().tolist())
    quality=df[df['task'].isin(sampled_tasks)].copy() if sampled_tasks else df.iloc[0:0].copy()
    quality_methods=list(ALL_SAMPLING_METHODS)
    quality=quality[quality.method.isin(quality_methods)]
    if not quality.empty:
        qtab=bootstrap_mean_ci(quality,['method','row_sigma','sequence'],
                               ['wc_mse','p95_mse','avg_mse','nominal_fp','robust_design_objective','cartesian_observations'],
                               unit_col='seed',n_boot=200 if quick else 2000)
        _csv(pretty_method_names(qtab),out/'tables'/'synthetic_sampling_quality.csv')
        _csv(pretty_method_names(qtab[qtab.method.isin(MAIN_SAMPLING_METHODS)]),out/'tables'/'synthetic_sampling_quality_main.csv')
        task_quality=quality[quality.method.isin(MAIN_TASK_QUALITY_METHODS)].copy()
        for sig,g in task_quality.groupby('row_sigma'):
            stag=str(f'{sig:g}').replace('.', 'p')
            line_by_method(g,'task','wc_mse',out/'figures'/f'synthetic_task_mse_sigma_{stag}',
                           ylabel='Worst-case theoretical MSE',xlabel='Task',
                           methods=MAIN_TASK_QUALITY_METHODS,ci=True,logy=True)
        qplot=quality.groupby(['seed','method','row_sigma'],as_index=False)['wc_mse'].mean()
        for sig,g in qplot.groupby('row_sigma'):
            stag=str(f'{sig:g}').replace('.', 'p')
            # Main-paper view: only scientifically necessary sampling comparators.
            point_by_method(g,'wc_mse',out/'figures'/f"synthetic_sampling_quality_sigma_{stag}",
                            ylabel='Worst-case theoretical MSE',logy=True,methods=MAIN_SAMPLING_METHODS)
            # Full evidence view: all methods retained for appendix/audit; never filtered by performance.
            point_by_method(g,'wc_mse',out/'figures'/f"synthetic_sampling_quality_all_baselines_sigma_{stag}",
                            ylabel='Worst-case theoretical MSE',logy=True,methods=ALL_SAMPLING_METHODS)
    sol=df[df.kind=='solver']
    if not sol.empty:
        _csv(_solver_convergence_summary(sol,'synthetic'),out/'tables'/'synthetic_solver_convergence_summary.csv')
        _csv(_synthetic_paired_advantage(sol,n_boot=200 if quick else 2000),out/'tables'/'vpm_paired_advantage_synthetic.csv')
        hsel=_history_selection_summary(sol,'Synthetic',['seed','row_sigma','sequence'],'task')
        _csv(hsel,out/'tables'/'vpm_history_selection_summary_synthetic.csv')
        _csv(hsel,out/'tables'/'vpm_history_selection_summary.csv')
        rtcols=[c for c in ['runtime','certificate_build_time','selection_time','solver_end_to_end_runtime','total_runtime_median5'] if c in sol.columns]
        if rtcols:
            rt=(sol.groupby(['method','row_sigma'],as_index=False)[rtcols].mean())
            _csv(rt,out/'tables'/'runtime_overhead_decomposition_synthetic.csv')
        if {'memory_size','selection_time'}.issubset(sol.columns):
            ms=(sol[sol.method=='VPM-RFW'].groupby(['row_sigma','task','memory_size'],as_index=False)
                .agg(mean_selection_time=('selection_time','mean'),mean_solver_time=('runtime','mean'),
                     mean_end_to_end_runtime=('solver_end_to_end_runtime','mean')))
            _csv(ms,out/'tables'/'vpm_selection_scaling_synthetic.csv')
        # Never average normalized and row-heterogeneous controls into one trajectory.
        # The first/main sigma (normally 0) also keeps the legacy short filename used
        # by the paper; every setting additionally gets an explicit sigma-tagged file.
        solver_specs=[
            ('cumulative_lmo','solver_cumulative_lmo','Cumulative LMO calls'),
            ('cumulative_fo','solver_cumulative_fo','Cumulative inner FO evaluations'),
            ('cumulative_total_runtime','solver_cumulative_wall_time','Cumulative wall-clock time (s)'),
            ('cumulative_cost','solver_cumulative_unit_oracle_count','Cumulative oracle count'),
        ]
        solver_sigmas=list(dict.fromkeys(sol['row_sigma'].astype(float).tolist())) if 'row_sigma' in sol else [None]
        for si,sig in enumerate(solver_sigmas):
            sg=sol if sig is None else sol[np.isclose(sol['row_sigma'].astype(float),float(sig))]
            stag='all' if sig is None else str(f'{float(sig):g}').replace('.', 'p')
            for yy,stem,lab in solver_specs:
                line_by_method(sg,'task',yy,out/'figures'/f'{stem}_sigma_{stag}',ylabel=lab,methods=MAIN_SOLVER_METHODS)
                line_by_method(sg,'task',yy,out/'figures'/f'{stem}_all_baselines_sigma_{stag}',ylabel=lab,methods=ALL_SOLVER_METHODS)
                if si==0:
                    line_by_method(sg,'task',yy,out/'figures'/stem,ylabel=lab,methods=MAIN_SOLVER_METHODS)
        scatter_score_realized(sol[sol.method=='VPM-RFW'],out/'figures'/'score_vs_realized',
                               certified=bool((sol.get('cert_status','')=='theorem_certified').all()) if 'cert_status' in sol else False)
        # In practical mode the absolute score inherits deliberately conservative
        # theory constants.  The per-task ratio to cold is the interpretable
        # selection quantity; compare it with the realized cost ratio to cold.
        v=sol[sol.method=='VPM-RFW'].copy(); c=sol[sol.method=='Cold-RFW'][['seed','task','row_sigma','realized_weighted_cost']].copy()
        c=c.rename(columns={'realized_weighted_cost':'cold_realized_cost'})
        if 'selection_score_ratio_to_cold' in v.columns and not v.empty and not c.empty:
            v=v.merge(c,on=['seed','task','row_sigma'],how='left')
            v['realized_cost_ratio_to_cold']=v['realized_weighted_cost']/v['cold_realized_cost'].replace(0,np.nan)
            scatter_score_realized(v,out/'figures'/'score_ratio_vs_realized_ratio',
                                   score_col='selection_score_ratio_to_cold',realized_col='realized_cost_ratio_to_cold',certified=False,
                                   xlabel='Selection score / cold score',ylabel='Realized cost / cold cost')
    return df


def _tag_frame(df, sweep, value):
    d=df.copy(); d['sweep']=str(sweep); d['sweep_value']=value; return d


def _synthetic_appendix(cfg,out,quick):
    scfg=copy.deepcopy(cfg['synthetic']); frames=[]
    if quick:
        # Smoke mode touches every appendix code path but intentionally uses a tiny solver/data
        # configuration. It is never used for reported paper statistics.
        scfg['quick_tasks']=2; scfg['quick_j_test']=2; scfg['quick_empirical_trials']=0
        scfg['sampling_stride']=2
        scfg['practical_solver']['quick_max_outer']=2
        scfg['practical_solver']['quick_max_ppa']=1
        scfg['practical_solver']['quick_max_resolvent']=3
    # Main experiments carry the statistical replication. Appendix sweeps use a
    # small, configurable number of independent sequences per point to retain
    # trend robustness without multiplying runtime.
    if not quick:
        scfg['tasks']=int(scfg.get('appendix_tasks', scfg['tasks']))
    nseed=1 if quick else min(int(scfg.get('appendix_sequences_per_point',2)),int(scfg['sequences']))
    seeds=[int(cfg['seed'])+i for i in range(nseed)]
    default_sigma=0.75

    # Sequence-shape controls: smooth drift and abrupt negative-transfer shift.
    for seq in scfg.get('appendix_sequences',['smooth','abrupt']):
        for seed in seeds:
            d=synthetic_main.run_sequence(scfg,seed,default_sigma,seq,quick=quick,
                    policies=scfg.get('policies',['cold','prev','nearest','vpm'])+scfg.get('appendix_policies',[]))
            frames.append(_tag_frame(d,'sequence',seq))

    # Row-norm heterogeneity sweep.
    sigma_vals=list(scfg.get('row_sigma_sweep',[0,.25,.5,.75,1.0]))
    if quick and len(sigma_vals)>2: sigma_vals=[sigma_vals[0],sigma_vals[-1]]
    for sig in sigma_vals:
        for seed in seeds[:1 if quick else min(3,len(seeds))]:
            d=synthetic_main.run_sequence(scfg,seed,float(sig),'recurrent',quick=quick,
                    policies=scfg.get('policies',['cold','prev','nearest','vpm']))
            frames.append(_tag_frame(d,'row_sigma',float(sig)))

    # Target-accuracy sensitivity. Total-budget and drift sensitivity are
    # separately replicated paired components, so the general appendix suite
    # must not rerun or mix their old low-replication sweeps.
    sweep_specs=[
        ('epsilon_abs','epsilon_sweep'),
    ]
    for field,key in sweep_specs:
        vals=list(scfg.get(key,[]))
        if quick and len(vals)>1: vals=[vals[-1]]
        for val in vals:
            c=copy.deepcopy(scfg); c[field]=int(val) if field=='total_budget' else float(val)
            # These sweeps target solver behavior; one seed in quick, up to three in full.
            for seed in seeds[:1 if quick else min(3,len(seeds))]:
                d=synthetic_main.run_sequence(c,seed,default_sigma,'recurrent',quick=quick,
                        policies=scfg.get('policies',['cold','prev','nearest','vpm']))
                frames.append(_tag_frame(d,field,int(val) if field=='total_budget' else float(val)))

    # LMO/FO artificial-weight sensitivity is optional: the compact default
    # already reports LMO, FO, and wall-clock separately, so this sweep is
    # redundant unless explicitly requested.
    if bool(scfg.get('run_cost_ratio_sweep',False)):
        ratios=list(scfg.get('cost_ratio_sweep',[0.1,1.0,10.0]))
        if quick and len(ratios)>1: ratios=[ratios[-1]]
        for ratio in ratios:
            c=copy.deepcopy(scfg); c['w_lmo']=float(ratio); c['w_fo']=1.0
            for seed in seeds[:1 if quick else len(seeds)]:
                d=synthetic_main.run_sequence(c,seed,default_sigma,'recurrent',quick=quick,
                        policies=scfg.get('policies',['cold','prev','nearest','vpm']))
                frames.append(_tag_frame(d,'w_lmo_over_w_fo',float(ratio)))

    df=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(); _csv(df,out/'raw'/'synthetic_appendix.csv')
    if not df.empty:
        summary=bootstrap_mean_ci(df,['sweep','sweep_value','kind','method'],
            ['wc_mse','avg_mse','cartesian_observations','runtime','lmo','fo','realized_weighted_cost','allocation_revision_l1'],
            unit_col='seed',n_boot=200 if quick else 1000)
        _csv(pretty_method_names(summary),out/'tables'/'synthetic_appendix_summary.csv')
        sol=df[(df.kind=='solver') & np.isfinite(df.get('score_or_cert_cost',np.nan))].copy()
        if not sol.empty:
            # Ranking/tightness diagnostics for the *numerical score proxy*. These are not
            # theorem-certificate validity claims unless execution_status says theorem certified.
            diag=[]
            for keys,g in sol.groupby(['sweep','sweep_value','method'],dropna=False):
                ok=g[['score_or_cert_cost','realized_weighted_cost']].dropna()
                rho=ok.corr(method='spearman').iloc[0,1] if len(ok)>2 else np.nan
                rat=(ok.score_or_cert_cost/ok.realized_weighted_cost.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
                diag.append({'sweep':keys[0],'sweep_value':keys[1],'method':keys[2],
                             'n':len(ok),'score_ge_realized_rate':float((ok.score_or_cert_cost>=ok.realized_weighted_cost).mean()) if len(ok) else np.nan,
                             'score_to_realized_median':float(rat.median()) if len(rat) else np.nan,
                             'score_to_realized_p90':float(rat.quantile(.9)) if len(rat) else np.nan,
                             'spearman_score_realized':float(rho) if np.isfinite(rho) else np.nan})
            _csv(pd.DataFrame(diag),out/'tables'/'score_proxy_tightness.csv')

    # Reduced O(S^2) hindsight realized-oracle diagnostic.
    hds=[]
    if bool(scfg.get('run_hindsight_diagnostic',True)):
        nh=1 if quick else int(scfg.get('hindsight_sequences',2))
        for j in range(nh):
            hds.append(synthetic_main.run_hindsight_diagnostic(scfg,int(cfg['seed'])+j,0.75,quick=quick))
    hd=pd.concat(hds,ignore_index=True) if hds else pd.DataFrame()
    _csv(hd,out/'raw'/'hindsight_diagnostic.csv')
    return df

def _hsi_one_set(cfg,out,quick,names,modes,suffix,data_dir=None):
    hcfg=cfg['hsi']; ddir=data_dir or hcfg.get('data_dir','data')
    rows=[]; summaries=[]; energies=[]
    for mode in modes:
        for name in names:
            print(f'[hsi] dataset={name} history={mode}')
            r,s,e=real_hsi.run_dataset(name,hcfg,ddir,quick=quick,history_mode=mode)
            rows.append(r); summaries.append(s); e=e.copy(); e['dataset']=name; e['history_mode']=mode; energies.append(e)
    rdf=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(); sdf=pd.concat(summaries,ignore_index=True) if summaries else pd.DataFrame(); edf=pd.concat(energies,ignore_index=True) if energies else pd.DataFrame()
    _csv(rdf,out/'raw'/f'hsi_{suffix}.csv'); _csv(pretty_method_names(sdf),out/'tables'/f'hsi_{suffix}_sam.csv'); _csv(edf,out/'tables'/f'hsi_{suffix}_rank_energy.csv')
    if not rdf.empty:
        closed=rdf[rdf.history_mode=='closed_loop']
        if not closed.empty:
            conv=_solver_convergence_summary(closed[closed['kind']=='solver'],'hsi')
            if not conv.empty:
                _csv(conv,out/'tables'/f'hsi_{suffix}_solver_convergence_summary.csv')
                if suffix=='main':
                    _csv(conv,out/'tables'/'solver_convergence_summary.csv')
            if suffix=='main':
                paired=_hsi_paired_advantage(closed)
                if not paired.empty:
                    _csv(paired,out/'tables'/'vpm_paired_advantage_hsi.csv')
                hsel=_history_selection_summary(closed[closed['kind']=='solver'],'KSC',
                                                ['dataset','history_mode','window_id'],'band')
                if not hsel.empty:
                    _csv(hsel,out/'tables'/'vpm_history_selection_summary_hsi.csv')
                    _csv(hsel,out/'tables'/'vpm_history_selection_summary.csv')
            hsi_values=['nmse_db','psnr','ssim','runtime','optimization_runtime','certificate_build_time','selection_time',
                        'solver_end_to_end_runtime','reconstruction_runtime','end_to_end_runtime','robust_design_objective',
                        'cartesian_observations','lmo','fo','realized_weighted_cost','allocation_revision_l1']+[c for c in closed.columns if c.startswith('mode_budget_r')]
            present=[c for c in hsi_values if c in closed.columns]
            hsi_values=present
            rtcols=[c for c in ['runtime','certificate_build_time','selection_time','solver_end_to_end_runtime','reconstruction_runtime','end_to_end_runtime'] if c in closed.columns]
            if rtcols:
                _csv(closed.groupby('method',as_index=False)[rtcols].mean(),out/'tables'/f'hsi_{suffix}_runtime_overhead.csv')
            block_len=int(hcfg.get('block_bootstrap_length', 2 if quick else 4))
            nboot=int(hcfg.get('bootstrap_resamples', 200 if quick else 1000))
            if 'window_id' in closed.columns and int(closed['window_id'].nunique())>1:
                # Independent local spectral episodes are the statistical units;
                # tasks inside one window remain correlated and are averaged first.
                bs=bootstrap_mean_ci(closed,['method'],hsi_values,unit_col='window_id',
                                     n_boot=nboot,seed=int(cfg['seed']))
                _csv(pretty_method_names(bs),out/'tables'/f'hsi_{suffix}_window_bootstrap.csv')
                _csv(pretty_method_names(bs),out/'tables'/f'hsi_{suffix}_comparison.csv')
                _csv(pretty_method_names(bs[bs.method.isin(MAIN_SAMPLING_METHODS)]),out/'tables'/f'hsi_{suffix}_comparison_main.csv')
                hp=closed.groupby(['window_id','method'],as_index=False)['nmse_db'].mean()
                point_by_method(hp,'nmse_db',out/'figures'/f'hsi_{suffix}_nmse_by_method',
                                ylabel='NMSE (dB)',methods=MAIN_SAMPLING_METHODS)
                point_by_method(hp,'nmse_db',out/'figures'/f'hsi_{suffix}_nmse_by_method_all_baselines',
                                ylabel='NMSE (dB)',methods=ALL_SAMPLING_METHODS)
            else:
                bs=contiguous_block_bootstrap(closed,group_col='method',value_cols=tuple(hsi_values),
                                              block_length=block_len,n_boot=nboot,seed=int(cfg['seed']))
                _csv(pretty_method_names(bs),out/'tables'/f'hsi_{suffix}_block_bootstrap.csv')
                _csv(pretty_method_names(bs),out/'tables'/f'hsi_{suffix}_comparison.csv')
                _csv(pretty_method_names(bs[bs.method.isin(MAIN_SAMPLING_METHODS)]),out/'tables'/f'hsi_{suffix}_comparison_main.csv')
                point_by_method(closed,'nmse_db',out/'figures'/f'hsi_{suffix}_nmse_by_method',
                                ylabel='NMSE (dB)',methods=MAIN_SAMPLING_METHODS)
                point_by_method(closed,'nmse_db',out/'figures'/f'hsi_{suffix}_nmse_by_method_all_baselines',
                                ylabel='NMSE (dB)',methods=ALL_SAMPLING_METHODS)
    return rdf


def _hsi(cfg,out,quick,appendix=False,data_dir=None):
    hcfg=cfg['hsi']
    if not appendix:
        names=hcfg.get('quick_datasets',['KSC']) if quick else hcfg.get('main_datasets',['KSC'])
        return _hsi_one_set(cfg,out,quick,names,['closed_loop'],'main',data_dir)

    # Additional real dataset is a closed-loop appendix result.
    names=hcfg.get('quick_datasets',['KSC']) if quick else hcfg.get('appendix_datasets',['Salinas'])
    main=_hsi_one_set(cfg,out,quick,names,['closed_loop'],'appendix',data_dir)
    # `--quick` is a smoke test, not a paper-statistics run. The oracle-history and
    # rank/budget studies are intentionally full-mode only because the oracle-history
    # task can be substantially more expensive even for a few bands.
    if quick:
        return main

    # Oracle-history is an explicitly labelled leakage-free diagnostic: band s enters only
    # after its sampling decision and can then be used for task s+1.
    if bool(hcfg.get('run_oracle_history',False)):
        onames=hcfg.get('oracle_history_datasets',['KSC'])
        _hsi_one_set(cfg,out,False,onames,['oracle_history'],'oracle_history',data_dir)

    # Rank/total-budget sensitivity is retained in the compact paper suite but may
    # be disabled by the fast trend-check profile because it repeats several KSC
    # passes and is not needed to decide whether the main effect is promising.
    if not bool(hcfg.get('run_rank_budget_sensitivity',True)):
        return main

    sens=[]
    rank_vals=list(map(int,hcfg.get('rank_sweep',[hcfg['rank']])))
    bud_vals=list(map(int,hcfg.get('total_budget_sweep',[hcfg['total_budget']])))
    if quick:
        rank_vals=[rank_vals[0],rank_vals[-1]] if len(rank_vals)>1 else rank_vals
        bud_vals=[bud_vals[0],bud_vals[-1]] if len(bud_vals)>1 else bud_vals
    for k in rank_vals:
        c=copy.deepcopy(cfg); c['hsi']['rank']=int(k)
        # Sensitivity curves use a smaller, common spectral subset; this preserves
        # the parameter trend while avoiding six additional full KSC passes.
        sb=int(hcfg.get('sensitivity_evaluation_bands',hcfg.get('evaluation_bands',8)))
        c['hsi']['evaluation_bands']=sb; c['hsi']['evaluation_bands_by_dataset']={'KSC':sb}; c['hsi']['evaluation_windows_by_dataset']={'KSC':1}; c['hsi']['evaluation_window_fractions_by_dataset']={'KSC':[float(hcfg.get('sensitivity_window_fraction',0.25))]}
        sw=int(hcfg.get('sensitivity_warmup',hcfg.get('warmup',1)))
        c['hsi']['warmup']=sw; c['hsi']['history_scenarios']=min(sw,int(hcfg.get('sensitivity_history_scenarios',sw)))
        # Sensitivity studies test the rank/budget trend, not the memory-policy
        # ablation.  Keep VPM plus the shared sampling baselines and avoid
        # rerunning Cold/Prev/Nearest for every sensitivity point.
        c['hsi']['policies']=list(hcfg.get('sensitivity_policies',['vpm']))
        # Keep the common total budget valid under the new two-mode rank floors.
        c['hsi']['total_budget']=max(int(hcfg['total_budget']),2*int(k))
        r=_hsi_one_set(c,out,quick,['KSC'],['closed_loop'],f'sensitivity_rank_{k}',data_dir)
        r=r.copy(); r['sweep']='rank'; r['sweep_value']=k; sens.append(r)
    for L in bud_vals:
        if int(L)<2*int(hcfg['rank']):
            continue
        c=copy.deepcopy(cfg); c['hsi']['total_budget']=int(L)
        sb=int(hcfg.get('sensitivity_evaluation_bands',hcfg.get('evaluation_bands',8)))
        c['hsi']['evaluation_bands']=sb; c['hsi']['evaluation_bands_by_dataset']={'KSC':sb}; c['hsi']['evaluation_windows_by_dataset']={'KSC':1}; c['hsi']['evaluation_window_fractions_by_dataset']={'KSC':[float(hcfg.get('sensitivity_window_fraction',0.25))]}
        sw=int(hcfg.get('sensitivity_warmup',hcfg.get('warmup',1)))
        c['hsi']['warmup']=sw; c['hsi']['history_scenarios']=min(sw,int(hcfg.get('sensitivity_history_scenarios',sw)))
        c['hsi']['policies']=list(hcfg.get('sensitivity_policies',['vpm']))
        r=_hsi_one_set(c,out,quick,['KSC'],['closed_loop'],f'sensitivity_budget_{L}',data_dir)
        r=r.copy(); r['sweep']='total_mode_budget'; r['sweep_value']=L; sens.append(r)
    if sens:
        _csv(pd.concat(sens,ignore_index=True),out/'raw'/'hsi_sensitivity.csv')
    return main

def _retrieval(cfg,out,quick):
    print('[retrieval] train/calibration/test + history-size runtime scaling')
    model_path=out/'models'/'retrieval.pkl'
    train,cal,test=retrieval_main.run(cfg,quick=quick,model_out=model_path)
    _csv(train,out/'raw'/'retrieval_train.csv'); _csv(cal,out/'raw'/'retrieval_calibration.csv'); _csv(test,out/'raw'/'retrieval_test.csv')
    if not test.empty:
        tab=(test.groupby(['retrieval_policy','topk'],dropna=False)
             .agg(regret_mean=('retrieval_score_regret','mean'),regret_median=('retrieval_score_regret','median'),
                  zero_regret_rate=('retrieval_score_regret',lambda x: float((np.asarray(x)<=1e-12).mean())),
                  coverage=('covered','mean'),candidate_fraction=('candidate_fraction','mean'),
                  certified_stop_rate=('adaptive_certified_stop','mean'),fallback_full_scan_rate=('fallback_full_scan','mean'),
                  mean_verified_score_eval_time=('verified_score_eval_time','mean'),
                  mean_full_scan_score_eval_time=('full_scan_score_eval_time','mean'))
             .reset_index())
        _csv(tab,out/'tables'/'retrieval_summary.csv')
    scaling_raw,scaling_summary=retrieval_main.run_scaling(cfg,model_path,quick=quick)
    _csv(scaling_raw,out/'raw'/'retrieval_runtime_scaling.csv')
    _csv(scaling_summary,out/'tables'/'retrieval_runtime_scaling_summary.csv')
    return test


def _certified_synthetic(cfg,out,quick):
    print('[certified] analytic small synthetic theorem-path benchmark')
    raw,summary,proof=certified_synthetic.run(cfg,quick=quick)
    _csv(raw,out/'raw'/'certified_synthetic_benchmark.csv')
    _csv(summary,out/'tables'/'certified_synthetic_benchmark_summary.csv')
    (out/'tables'/'certified_synthetic_analytic_proof.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    return raw


def _convergence_sensitivity(cfg,out,quick):
    print('[diagnostic] solver cap/convergence sensitivity')
    raw,summary=diagnostics.run_solver_cap_sensitivity(cfg,quick=quick)
    _csv(raw,out/'raw'/'solver_cap_sensitivity.csv'); _csv(summary,out/'tables'/'solver_cap_sensitivity_summary.csv')
    return raw


def _robustness(cfg,out,quick):
    print('[diagnostic] clean-to-uncertain robustness sweep')
    raw,summary=diagnostics.run_robustness_sweep(cfg,quick=quick)
    _csv(raw,out/'raw'/'robustness_sweep.csv'); _csv(summary,out/'tables'/'robustness_sweep_summary.csv')
    if not raw.empty:
        q=raw.groupby(['seed','task','uncertainty_multiplier','method'],as_index=False)['wc_mse'].mean()
        for mult,g in q.groupby('uncertainty_multiplier'):
            tag=str(f'{float(mult):g}').replace('.','p')
            point_by_method(g,'wc_mse',out/'figures'/f'robustness_wc_mse_u_{tag}',ylabel='Worst-case theoretical MSE',logy=True,methods=['Greedy-FP-2019','FFW-2026','Nominal-RFW','VPM-RFW'])
    return raw


def _hsi_independent(cfg,out,quick,data_dir=None):
    print('[diagnostic] independent closed-loop HSI sanity check')
    ddir=data_dir or cfg['hsi'].get('data_dir','data')
    raw,summary=diagnostics.run_hsi_independent_closed_loop(cfg,ddir,quick=quick)
    _csv(raw,out/'raw'/'hsi_independent_closed_loop.csv'); _csv(pretty_method_names(summary.rename(columns={'trajectory_method':'method'})),out/'tables'/'hsi_independent_closed_loop_summary.csv')
    return raw


def _certified_audit(out):
    print('[audit] fail-closed sequential certified path')
    import run_certified_audit as audit_runner
    return audit_runner.run(PROJECT_ROOT/'configs'/'certified_audit.yaml',out)


def _wang(cfg,out,quick):
    print('[baseline] Wang et al. 2022 paper-faithful Algorithm 3 reproduction')
    df=wang_reproduction.run(cfg['wang_reproduction'],quick=quick); _csv(df,out/'raw'/'wang2022_reproduction.csv')
    return df


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suite',default='synthetic',choices=['synthetic','synthetic-budget','synthetic-drift','hsi','convergence'])
    ap.add_argument('--quick',action='store_true',help='small smoke run; preserves code paths but not paper statistics')
    ap.add_argument('--config',default=str(PROJECT_ROOT/'configs'/'full.yaml'))
    ap.add_argument('--data-dir',default=None)
    ap.add_argument('--output',default=None)
    ap.add_argument('--no-archive',action='store_true')
    ap.add_argument('--_child',action='store_true',help=argparse.SUPPRESS)
    args=ap.parse_args(); pin_threads(1)
    cfg=load_config(args.config)
    # Make the one-click VSCode run independent of the terminal's current working directory.
    if not Path(cfg.get('output_dir','results')).is_absolute() and not args.output:
        cfg['output_dir']=str(PROJECT_ROOT/cfg.get('output_dir','results'))
    if not Path(cfg['hsi'].get('data_dir','data')).is_absolute() and args.data_dir is None:
        cfg['hsi']['data_dir']=str(PROJECT_ROOT/cfg['hsi'].get('data_dir','data'))
    out=_outdir(cfg,args); out.mkdir(parents=True,exist_ok=True)
    (out/'models').mkdir(exist_ok=True); (out/'figures').mkdir(exist_ok=True); (out/'tables').mkdir(exist_ok=True); (out/'raw').mkdir(exist_ok=True)
    (out/'effective_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    environment(out/'environment.json'); source_manifest(PROJECT_ROOT,out/'SOURCE_MANIFEST.sha256')
    write_data_audit(cfg['hsi'].get('datasets',['KSC']),args.data_dir or cfg['hsi'].get('data_dir','data'),out/'data_audit.json')
    t0=time.perf_counter()

    # Composite suites are delegated via os.execv to the lightweight standard-library
    # RUN_ALL.py orchestrator.  execv replaces this already-imported scientific Python
    # process, so the parent does not retain NumPy/Pandas/Matplotlib memory while KSC loads.
    composite=set()
    if args.suite in composite and not args._child:
        cmd=[sys.executable,str(PROJECT_ROOT/'RUN_ALL.py'),'--suite',args.suite,
             '--config',str(Path(args.config).resolve()),'--output',str(out.resolve())]
        if args.quick: cmd.append('--quick')
        if args.data_dir: cmd += ['--data-dir',str(Path(args.data_dir).resolve())]
        if args.no_archive: cmd.append('--no-archive')
        os.execv(sys.executable,cmd)

    try:
        if args.suite == 'synthetic':
            _synthetic_main(cfg,out,args.quick); gc.collect()
        elif args.suite == 'synthetic-budget':
            synthetic_budget.run(cfg,out,args.quick); gc.collect()
        elif args.suite == 'synthetic-drift':
            synthetic_drift.run(cfg,out,args.quick); gc.collect()
        elif args.suite == 'hsi':
            _hsi(cfg,out,args.quick,appendix=False,data_dir=args.data_dir); gc.collect()
        elif args.suite == 'convergence':
            _convergence_sensitivity(cfg,out,args.quick); gc.collect()
        else:
            raise ValueError(f'unhandled suite {args.suite}')
    finally:
        # Child RUN_COMPLETE is overwritten by the parent for composite suites.
        (out/'RUN_COMPLETE.json').write_text(json.dumps({'elapsed_seconds':time.perf_counter()-t0,'suite':args.suite,'quick':args.quick},indent=2))
    if not args.no_archive:
        archive=shutil.make_archive(str(out),'zip',root_dir=out)
        print(f'[done] results: {out}\n[done] archive: {archive}')
    else:
        print(f'[done] results: {out}')

if __name__=='__main__': main()
