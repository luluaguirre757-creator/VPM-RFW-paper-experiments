from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from vpmrfw.experiments import synthetic_main
from vpmrfw.plotting.figures import (numeric_sensitivity_plot,
    sensitivity_series_plot, pretty_method_names)


DRIFT_GRID=[0.0,0.005,0.01,0.02,0.035,0.05,0.075,0.10]
PAPER_METHODS=['VPM-RFW','FFW-2026','Greedy-FP-2019']
SOLVER_METHODS=['Cold-RFW','Prev-RFW','Nearest-RFW','VPM-RFW']
BASELINES=['FFW-2026','Greedy-FP-2019']
METRICS=['wc_mse','avg_mse','median_mse','p95_mse']


def validate_drift_grid(scfg):
    values=list(map(float,scfg.get('drift_sweep',[])))
    if values != DRIFT_GRID:
        raise ValueError(f'formal synthetic drift grid must be exactly {DRIFT_GRID}, got {values}')
    if any(v<0.0 or v>0.10 for v in values):
        raise ValueError('synthetic drift values must lie in [0,0.10]')
    return values


def _bootstrap(x,n_boot,rng):
    x=np.asarray(x,float)
    if not len(x): return np.nan,np.nan
    if len(x)==1: return float(x[0]),float(x[0])
    b=np.mean(rng.choice(x,size=(int(n_boot),len(x)),replace=True),axis=1)
    lo,hi=np.quantile(b,[.025,.975]); return float(lo),float(hi)


def _sequence_aggregate(raw, methods, value_cols):
    d=raw[raw.method.isin(methods)].copy(); rows=[]
    for (drift,seed,method),g in d.groupby(['drift','seed','method'],sort=True):
        row={'drift':float(drift),'seed':int(seed),'method':method,
             'n_tasks':int(g.task.nunique())}
        for col in value_cols:
            row[col]=float(g[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def sequence_reconstruction_metrics(raw):
    return _sequence_aggregate(raw,PAPER_METHODS,METRICS)


def sequence_solver_metrics(raw):
    d=raw[raw.kind=='solver']
    return _sequence_aggregate(d,SOLVER_METHODS,
        ['realized_weighted_cost','runtime','allocation_revision_l1'])


def summarize(seq, value_cols, n_boot=2000, seed=0):
    rng=np.random.default_rng(seed); rows=[]
    for (drift,method),g in seq.groupby(['drift','method'],sort=True):
        row={'drift':float(drift),'method':method,'n_sequences':int(g.seed.nunique())}
        for col in value_cols:
            x=g[col].to_numpy(float); row[col]=float(x.mean())
            row[f'{col}_ci_lo'],row[f'{col}_ci_hi']=_bootstrap(x,n_boot,rng)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(seq,n_boot=2000,seed=0):
    rng=np.random.default_rng(seed); rows=[]
    for drift in DRIFT_GRID:
        p=seq[np.isclose(seq.drift,drift)].pivot(index='seed',columns='method',values='wc_mse')
        for baseline in BASELINES:
            pair=p[['VPM-RFW',baseline]].dropna(); v=pair['VPM-RFW'].to_numpy(float)
            b=pair[baseline].to_numpy(float); diff=b-v; rel=diff/b
            dlo,dhi=_bootstrap(diff,n_boot,rng); rlo,rhi=_bootstrap(rel,n_boot,rng)
            rows.append({'drift':drift,'baseline':baseline,'n_sequences':len(pair),
                         'vpm_wc_mse_mean':float(v.mean()),'baseline_wc_mse_mean':float(b.mean()),
                         'paired_mean_difference':float(diff.mean()),
                         'paired_mean_difference_ci_lo':dlo,'paired_mean_difference_ci_hi':dhi,
                         'relative_reduction':float(rel.mean()),
                         'relative_reduction_ci_lo':rlo,'relative_reduction_ci_hi':rhi,
                         'win_rate':float(np.mean(v<b))})
    return pd.DataFrame(rows)


def history_selection(raw,n_boot=2000,seed=0):
    v=raw[(raw.kind=='solver')&(raw.method=='VPM-RFW')&(raw.task>0)].copy()
    nearest=raw[(raw.kind=='solver')&(raw.method=='Nearest-RFW')][
        ['drift','seed','task','selected_history']].rename(columns={'selected_history':'nearest_history'})
    v=v.merge(nearest,on=['drift','seed','task'],how='left')
    v['history_selection_rate']=(v.selected_history>=0).astype(float)
    v['same_as_prev_rate']=(v.selected_history==v.task-1).astype(float)
    v['same_as_nearest_rate']=(v.selected_history==v.nearest_history).astype(float)
    v['older_history_selection_rate']=((v.selected_history>=0)&(v.selected_history!=v.task-1)).astype(float)
    v['older_not_nearest_rate']=((v.selected_history>=0)&(v.selected_history!=v.task-1)&
                                 (v.selected_history!=v.nearest_history)).astype(float)
    cols=['history_selection_rate','same_as_prev_rate','same_as_nearest_rate',
          'older_history_selection_rate','older_not_nearest_rate']
    per=v.groupby(['drift','seed'],as_index=False)[cols].mean()
    rng=np.random.default_rng(seed); rows=[]
    for drift,g in per.groupby('drift',sort=True):
        row={'drift':float(drift),'n_sequences':int(g.seed.nunique())}
        for col in cols:
            x=g[col].to_numpy(float); row[col]=float(x.mean())
            row[f'{col}_ci_lo'],row[f'{col}_ci_hi']=_bootstrap(x,n_boot,rng)
        rows.append(row)
    return per,pd.DataFrame(rows)


def paper_table(recon,solver):
    rlabels={'VPM-RFW':'VPM','FFW-2026':'FFW','Greedy-FP-2019':'Greedy'}
    slabels={'VPM-RFW':'VPM','Prev-RFW':'Prev','Nearest-RFW':'Nearest','Cold-RFW':'Cold'}
    rows=[]
    for drift in DRIFT_GRID:
        rg=recon[np.isclose(recon.drift,drift)].set_index('method')
        sg=solver[np.isclose(solver.drift,drift)].set_index('method')
        row={'drift':drift}
        for method,label in rlabels.items():
            row[f'{label} wc_mse']=float(rg.loc[method,'wc_mse'])
            row[f'{label} mean_mse']=float(rg.loc[method,'avg_mse'])
            row[f'{label} p95_mse']=float(rg.loc[method,'p95_mse'])
        for baseline,label in [('FFW-2026','FFW'),('Greedy-FP-2019','Greedy')]:
            row[f'VPM-vs-{label} wc reduction']=1-row['VPM wc_mse']/row[f'{label} wc_mse']
            row[f'VPM-vs-{label} mean reduction']=1-row['VPM mean_mse']/row[f'{label} mean_mse']
        for method,label in slabels.items():
            row[f'{label} solver cost']=float(sg.loc[method,'realized_weighted_cost'])
        rows.append(row)
    return pd.DataFrame(rows)


def _trend_label(x,y):
    y=np.asarray(y,float); differences=np.diff(y)
    tol=max(np.max(np.abs(y)),1e-30)*1e-6
    if np.all(differences>=-tol): return 'increases'
    if np.all(differences<=tol): return 'decreases'
    slope=np.polyfit(np.asarray(x,float),y,1)[0]
    if abs(slope)<=max(np.mean(np.abs(y)),1e-30)*.05: return 'approximately stable'
    return 'non-monotonic'


def report(recon,paired,solver,history,p95_figure):
    wc=recon.pivot(index='drift',columns='method',values='wc_mse')
    best=wc.idxmin(axis=1); vbest=best[best=='VPM-RFW'].index.tolist()
    f=paired[paired.baseline=='FFW-2026']; g=paired[paired.baseline=='Greedy-FP-2019']
    allp=pd.concat([f.assign(label='FFW'),g.assign(label='Greedy-FP')],ignore_index=True)
    weakest=allp.loc[allp.relative_reduction.idxmin()]; strongest=allp.loc[allp.relative_reduction.idxmax()]
    worse=allp[allp.relative_reduction_ci_hi<0][['drift','label']].to_dict('records')
    vrec=recon[recon.method=='VPM-RFW'].sort_values('drift')
    metric_trends={m:_trend_label(vrec.drift,vrec[m]) for m in METRICS}
    vf=solver[solver.method=='VPM-RFW'].sort_values('drift')
    cold=solver[solver.method=='Cold-RFW'].sort_values('drift')
    saving=1-vf.realized_weighted_cost.to_numpy()/cold.realized_weighted_cost.to_numpy()
    regimes=[('Near-stationary',0,.01),('Moderate',.02,.05),('Strong',.075,.10)]
    region_lines=[]
    for name,lo,hi in regimes:
        sub=wc[(wc.index>=lo)&(wc.index<=hi)]
        region_lines.append(f'- {name} ({lo:g}–{hi:g}): VPM ranks first at {int((sub.idxmin(axis=1)=="VPM-RFW").sum())}/{len(sub)} points.')
    lines=['# Synthetic Drift Sweep Report','',
        '- Protocol: 8 fixed drift values × 10 paired independent sequences × 8 tasks; bootstrap unit is the sequence.',
        '- Only drift changes. Total budget is fixed at 48; seeds and random-number streams are shared across drift values.',
        '- Worst-case MSE is evaluated per task and averaged within each sequence before cross-sequence inference.',
        f'- VPM-RFW has the lowest mean worst-case MSE at {len(vbest)}/8 drift values: {vbest}.',
        f'- VPM-RFW beats FFW at {int((wc["VPM-RFW"]<wc["FFW-2026"]).sum())}/8 points and Greedy-FP at {int((wc["VPM-RFW"]<wc["Greedy-FP-2019"]).sum())}/8 points.',
        f'- Paired relative-reduction CI is strictly favorable versus FFW at {int((f.relative_reduction_ci_lo>0).sum())}/8 points and versus Greedy-FP at {int((g.relative_reduction_ci_lo>0).sum())}/8 points.',
        f'- VPM metric trends: {metric_trends}; P95 figure generated={p95_figure}.',
        f'- Paired VPM advantage versus FFW is {_trend_label(f.drift,f.relative_reduction)} with drift; versus Greedy-FP it is {_trend_label(g.drift,g.relative_reduction)}.',
        f'- Strongest paired advantage: {strongest.label} at drift={strongest.drift:g} ({strongest.relative_reduction:.2%}); weakest: {weakest.label} at drift={weakest.drift:g} ({weakest.relative_reduction:.2%}).',
        f'- Drift points where VPM is clearly worse by paired 95% CI: {worse if worse else "none"}.',
        f'- VPM solver-cost saving versus Cold is {_trend_label(DRIFT_GRID,saving)}; values={dict(zip(DRIFT_GRID,np.round(saving,4)))}.',
        f'- History-selection rate is {_trend_label(history.drift,history.history_selection_rate)}; older-history rate is {_trend_label(history.drift,history.older_history_selection_rate)}.','',
        '## Drift regimes','',*region_lines,'',
        '- All drift points are retained regardless of method performance. Exact CIs and selection rates are in the generated CSV tables.','']
    return '\n'.join(lines)


def run(cfg,out:Path,quick=False):
    scfg=copy.deepcopy(cfg['synthetic']); validate_drift_grid(scfg)
    if int(scfg['total_budget'])!=48:
        raise ValueError(f'drift sweep requires the frozen default total_budget=48, got {scfg["total_budget"]}')
    scfg['tasks']=int(scfg.get('appendix_tasks',scfg['tasks']))
    scfg['sampling_stride']=1; scfg['empirical_trials']=0; scfg['quick_empirical_trials']=0
    scfg['runtime_tasks']=[]; scfg['quick_runtime_tasks']=[]
    nseed=int(scfg['drift_sequences_per_point_quick' if quick else 'drift_sequences_per_point'])
    nboot=int(scfg['drift_bootstrap_resamples_quick' if quick else 'drift_bootstrap_resamples'])
    seeds=[int(cfg['seed'])+i for i in range(nseed)]; frames=[]
    frozen={k:copy.deepcopy(v) for k,v in scfg.items() if k!='drift'}
    for drift in DRIFT_GRID:
        c=copy.deepcopy(scfg); c['drift']=drift
        if {k:v for k,v in c.items() if k!='drift'} != frozen:
            raise AssertionError('a non-drift configuration field changed inside drift sweep')
        for seed in seeds:
            d=synthetic_main.run_sequence(c,seed,.75,'recurrent',quick=False,policies=['cold','prev','nearest','vpm'])
            d['drift']=drift; d['paired_task_id']=d.seed.astype(str)+':'+d.task.astype(str)+':recurrent:0.75'
            frames.append(d)
    raw=pd.concat(frames,ignore_index=True)
    if not raw.total_mode_budget.astype(int).eq(48).all():
        raise AssertionError('drift sweep changed or violated the frozen total budget')
    recon_seq=sequence_reconstruction_metrics(raw); solver_seq=sequence_solver_metrics(raw)
    expected={(d,s,m) for d in DRIFT_GRID for s in seeds for m in PAPER_METHODS}
    observed=set(map(tuple,recon_seq[['drift','seed','method']].itertuples(index=False,name=None)))
    if observed!=expected: raise AssertionError(f'incomplete paired drift design: {len(expected-observed)} cells missing')
    recon=summarize(recon_seq,METRICS,nboot,int(cfg['seed']))
    solver=summarize(solver_seq,['realized_weighted_cost','runtime','allocation_revision_l1'],nboot,int(cfg['seed'])+1)
    paired=paired_comparisons(recon_seq,nboot,int(cfg['seed'])+2)
    history_seq,history=history_selection(raw,nboot,int(cfg['seed'])+3)
    paper=paper_table(recon,solver)
    out=Path(out); (out/'raw').mkdir(parents=True,exist_ok=True); (out/'tables').mkdir(parents=True,exist_ok=True)
    raw.to_csv(out/'raw'/'synthetic_drift.csv',index=False)
    recon_seq.to_csv(out/'tables'/'synthetic_drift_sequence_metrics.csv',index=False)
    solver_seq.to_csv(out/'tables'/'synthetic_drift_solver_sequence_metrics.csv',index=False)
    pretty_method_names(recon).to_csv(out/'tables'/'synthetic_drift_summary.csv',index=False)
    pretty_method_names(solver).to_csv(out/'tables'/'synthetic_drift_solver_cost.csv',index=False)
    paired.assign(baseline=paired.baseline.replace({'FFW-2026':'FFW','Greedy-FP-2019':'Greedy-FP'})).to_csv(out/'tables'/'synthetic_drift_paired_comparison.csv',index=False)
    history_seq.to_csv(out/'tables'/'synthetic_drift_history_selection_by_sequence.csv',index=False)
    history.to_csv(out/'tables'/'synthetic_drift_history_selection.csv',index=False)
    paper.to_csv(out/'tables'/'synthetic_drift_summary_paper.csv',index=False)
    numeric_sensitivity_plot(recon,'drift','wc_mse',out/'figures'/'synthetic_drift_wc_mse','Task drift','Worst-case MSE',PAPER_METHODS,logy=True)
    numeric_sensitivity_plot(recon,'drift','avg_mse',out/'figures'/'synthetic_drift_mean_mse','Task drift','Mean MSE',PAPER_METHODS,logy=True)
    numeric_sensitivity_plot(solver,'drift','realized_weighted_cost',out/'figures'/'synthetic_drift_solver_cost','Task drift','Realized weighted solver cost',SOLVER_METHODS,logy='auto')
    sensitivity_series_plot(history,'drift',[
        ('history_selection_rate','Any history'),('same_as_prev_rate','Same as Prev'),
        ('same_as_nearest_rate','Same as Nearest'),('older_history_selection_rate','Older history')],
        out/'figures'/'synthetic_drift_history_selection','Task drift','Selection rate')
    correlations=[]
    for method in PAPER_METHODS:
        g=recon[recon.method==method].sort_values('drift')
        correlations.extend([g.p95_mse.corr(g.wc_mse),g.p95_mse.corr(g.avg_mse)])
    p95_figure=bool(min(correlations)<.995)
    if p95_figure:
        numeric_sensitivity_plot(recon,'drift','p95_mse',out/'figures'/'synthetic_drift_p95_mse','Task drift','P95 MSE',PAPER_METHODS,logy=True)
    (out/'DRIFT_SWEEP_REPORT.md').write_text(report(recon,paired,solver,history,p95_figure),encoding='utf-8')
    return raw,recon,paired
