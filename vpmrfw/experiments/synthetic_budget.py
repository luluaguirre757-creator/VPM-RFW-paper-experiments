from __future__ import annotations

import copy
from pathlib import Path
import secrets

import numpy as np
import pandas as pd

from vpmrfw.experiments import synthetic_main
from vpmrfw.plotting.figures import budget_sensitivity_plot, pretty_method_names


BUDGET_GRID = [36, 42, 48, 60, 72, 84, 96, 108, 120]
PAPER_METHODS = ['VPM-RFW', 'FFW-2026', 'Greedy-FP-2019']
BASELINES = ['FFW-2026', 'Greedy-FP-2019']
METRICS = ['wc_mse', 'avg_mse', 'median_mse', 'p95_mse']


def feasible_allocation(budget: int, floors, capacities):
    """Construct an integer allocation or fail; never clip or substitute a budget."""
    budget=int(budget); floors=np.asarray(floors,dtype=int); capacities=np.asarray(capacities,dtype=int)
    if floors.shape != capacities.shape or np.any(floors>capacities):
        raise ValueError('invalid floor/capacity geometry')
    if not int(floors.sum()) <= budget <= int(capacities.sum()):
        raise ValueError(f'budget {budget} is infeasible; require {floors.sum()} <= B <= {capacities.sum()}')
    allocation=floors.copy(); remaining=budget-int(allocation.sum())
    for j in range(len(allocation)):
        add=min(remaining,int(capacities[j]-allocation[j])); allocation[j]+=add; remaining-=add
    if remaining or int(allocation.sum()) != budget:
        raise ValueError(f'no integer allocation found for budget {budget}')
    return allocation.tolist()


def validate_budget_grid(scfg):
    budgets=list(map(int,scfg.get('total_budget_sweep',[])))
    if budgets != BUDGET_GRID:
        raise ValueError(f'formal synthetic budget grid must be exactly {BUDGET_GRID}, got {budgets}')
    floors=list(map(int,scfg.get('operational_floors',scfg['K'])))
    capacities=list(map(int,scfg['N']))
    return {b:feasible_allocation(b,floors,capacities) for b in budgets}


def _bootstrap(x, n_boot, rng):
    x=np.asarray(x,dtype=float)
    if not len(x):
        return np.nan,np.nan
    if len(x)==1:
        return float(x[0]),float(x[0])
    vals=np.mean(rng.choice(x,size=(int(n_boot),len(x)),replace=True),axis=1)
    lo,hi=np.quantile(vals,[.025,.975]); return float(lo),float(hi)


def sequence_metrics(raw):
    """Aggregate tasks inside each independent sequence before any inference."""
    d=raw[raw['method'].isin(PAPER_METHODS)].copy()
    rows=[]
    for (budget,seed,method),g in d.groupby(['budget','seed','method'],sort=True):
        row={'budget':int(budget),'seed':int(seed),'method':method,'n_tasks':int(g['task'].nunique())}
        for metric in METRICS:
            row[metric]=float(g[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_sequences(seq, n_boot=2000, seed=0):
    rng=np.random.default_rng(seed); rows=[]
    for (budget,method),g in seq.groupby(['budget','method'],sort=True):
        row={'budget':int(budget),'method':method,'n_sequences':int(g['seed'].nunique())}
        for metric in METRICS:
            x=g[metric].to_numpy(float); row[metric]=float(x.mean())
            row[f'{metric}_ci_lo'],row[f'{metric}_ci_hi']=_bootstrap(x,n_boot,rng)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(seq, n_boot=2000, seed=0):
    rng=np.random.default_rng(seed); rows=[]
    for budget in sorted(seq['budget'].unique()):
        pivot=seq[seq.budget==budget].pivot(index='seed',columns='method',values='wc_mse')
        for baseline in BASELINES:
            pair=pivot[['VPM-RFW',baseline]].dropna()
            v=pair['VPM-RFW'].to_numpy(float); b=pair[baseline].to_numpy(float)
            diff=b-v; rel=diff/b
            dlo,dhi=_bootstrap(diff,n_boot,rng); rlo,rhi=_bootstrap(rel,n_boot,rng)
            rows.append({'budget':int(budget),'baseline':baseline,'n_sequences':len(pair),
                         'vpm_wc_mse_mean':float(v.mean()),'baseline_wc_mse_mean':float(b.mean()),
                         'paired_mean_difference':float(diff.mean()),
                         'paired_mean_difference_ci_lo':dlo,'paired_mean_difference_ci_hi':dhi,
                         'relative_reduction':float(rel.mean()),
                         'relative_reduction_ci_lo':rlo,'relative_reduction_ci_hi':rhi,
                         'win_rate':float(np.mean(v<b))})
    return pd.DataFrame(rows)


def improvement_vs_low_budget(seq, n_boot=2000, seed=0):
    rng=np.random.default_rng(seed); rows=[]
    for method,g in seq.groupby('method',sort=False):
        base=g[g.budget==BUDGET_GRID[0]].set_index('seed')['wc_mse']
        for budget,bg in g.groupby('budget',sort=True):
            cur=bg.set_index('seed')['wc_mse']; pair=pd.concat([base.rename('base'),cur.rename('cur')],axis=1).dropna()
            vals=(pair['base']-pair['cur'])/pair['base']; lo,hi=_bootstrap(vals.to_numpy(float),n_boot,rng)
            rows.append({'budget':int(budget),'method':method,'n_sequences':len(pair),
                         'wc_relative_improvement_vs_budget_36':float(vals.mean()),
                         'ci_lo':lo,'ci_hi':hi})
    return pd.DataFrame(rows)


def paper_table(summary):
    labels={'VPM-RFW':'VPM','FFW-2026':'FFW','Greedy-FP-2019':'Greedy'}
    rows=[]
    for budget in BUDGET_GRID:
        g=summary[summary.budget==budget].set_index('method')
        row={'budget':budget}
        for method,label in labels.items():
            row[f'{label} wc_mse']=float(g.loc[method,'wc_mse'])
            row[f'{label} mean_mse']=float(g.loc[method,'avg_mse'])
        for baseline,label in [('FFW-2026','FFW'),('Greedy-FP-2019','Greedy')]:
            row[f'VPM-vs-{label} wc reduction']=1-row['VPM wc_mse']/row[f'{label} wc_mse']
            row[f'VPM-vs-{label} mean reduction']=1-row['VPM mean_mse']/row[f'{label} mean_mse']
        rows.append(row)
    return pd.DataFrame(rows)


def allocation_summary(raw):
    rows=[]
    for (budget,method),g in raw[raw.method.isin(PAPER_METHODS)].groupby(['budget','method'],sort=True):
        cards=sorted(g['mode_cardinalities'].astype(str).unique().tolist())
        rows.append({'budget':int(budget),'method':method,'n_task_rows':len(g),
                     'all_allocations_feasible':bool(g['total_mode_budget'].astype(int).eq(int(budget)).all()),
                     'total_mode_budget_min':int(g['total_mode_budget'].min()),
                     'total_mode_budget_max':int(g['total_mode_budget'].max()),
                     'mode_cardinalities_observed':' | '.join(cards),
                     'cartesian_observations_min':int(g['cartesian_observations'].min()),
                     'cartesian_observations_mean':float(g['cartesian_observations'].mean()),
                     'cartesian_observations_max':int(g['cartesian_observations'].max())})
    return pd.DataFrame(rows)


def _pretty_baselines(df):
    out=df.copy()
    if 'baseline' in out:
        out['baseline']=out['baseline'].replace({'FFW-2026':'FFW','Greedy-FP-2019':'Greedy-FP'})
    return out


def _report(summary, paired, seed_manifest=None):
    wc=summary.pivot(index='budget',columns='method',values='wc_mse')
    best=wc.idxmin(axis=1); vpm_best=best[best=='VPM-RFW'].index.tolist()
    ffw=paired[paired.baseline=='FFW-2026']; greedy=paired[paired.baseline=='Greedy-FP-2019']
    sig_ffw=ffw.loc[ffw.relative_reduction_ci_lo>0,'budget'].tolist()
    sig_greedy=greedy.loc[greedy.relative_reduction_ci_lo>0,'budget'].tolist()
    weakest=pd.concat([ffw.assign(label='FFW'),greedy.assign(label='Greedy-FP')]).sort_values('relative_reduction').iloc[0]
    worse=pd.concat([ffw.assign(label='FFW'),greedy.assign(label='Greedy-FP')])
    worse=worse[worse.relative_reduction_ci_hi<0]
    trend_signs={}
    for method,g in summary.groupby('method'):
        trend_signs[method]={metric:bool(np.corrcoef(g.budget,g[metric])[0,1]<0)
                             for metric in ['wc_mse','avg_mse','p95_mse']}
    trends_consistent=all(len(set(signs.values()))==1 for signs in trend_signs.values())
    region_lines=[]
    for name,budgets in [('Low (36–48)',[36,42,48]),('Medium (60–84)',[60,72,84]),
                         ('High (96–120)',[96,108,120])]:
        sub=wc.loc[budgets]
        rf=1-sub['VPM-RFW'].mean()/sub['FFW-2026'].mean()
        rg=1-sub['VPM-RFW'].mean()/sub['Greedy-FP-2019'].mean()
        wins=int((sub.idxmin(axis=1)=='VPM-RFW').sum())
        region_lines.append(f'- {name}: VPM ranks first at {wins}/3 budgets; aggregate-mean reduction versus FFW={rf:.2%}, versus Greedy-FP={rg:.2%}.')
    lines=[
        '# Synthetic Total-Budget Sweep Report','',
        f'- Protocol: {len(BUDGET_GRID)} budgets × {int(summary.n_sequences.min())} independent sequences; tasks are aggregated within each sequence before bootstrap.',
        f'- Random seed screening used design stability only: {int(seed_manifest.accepted.sum()) if seed_manifest is not None else 0} accepted, {int((~seed_manifest.accepted).sum()) if seed_manifest is not None else 0} rejected; held-out MSE was not used for acceptance.',
        '- Worst-case MSE is the project metric evaluated per task and then averaged within each sequence; it is not the global maximum over pooled tasks.',
        f'- VPM-RFW is best in {len(vpm_best)}/9 budgets: {vpm_best}.',
        f'- VPM-RFW beats FFW in mean worst-case MSE at {(wc["VPM-RFW"]<wc["FFW-2026"]).sum()}/9 budgets.',
        f'- VPM-RFW beats Greedy-FP in mean worst-case MSE at {(wc["VPM-RFW"]<wc["Greedy-FP-2019"]).sum()}/9 budgets.',
        f'- Paired 95% CI is strictly favorable versus FFW at {len(sig_ffw)}/9 budgets: {sig_ffw}.',
        f'- Paired 95% CI is strictly favorable versus Greedy-FP at {len(sig_greedy)}/9 budgets: {sig_greedy}.',
        f'- Weakest observed VPM advantage is versus {weakest.label} at B={int(weakest.budget)} (paired relative reduction {weakest.relative_reduction:.2%}).',
        f'- Budgets with a paired 95% CI showing VPM clearly worse: {"none" if worse.empty else worse[["budget","label"]].to_dict("records")}.',
        f'- Worst-case, mean, and P95 trend directions are consistent within every method: {trends_consistent}; details={trend_signs}.','',
        '## Budget regions','',
        *region_lines,
        '- Low, medium, and high results are retained in full; no point was filtered by performance.',
        '- Consult `tables/synthetic_budget_summary.csv` and `tables/synthetic_budget_paired_comparison.csv` for the exact region-wise values and uncertainty.',''
    ]
    return '\n'.join(lines)


def run(cfg, out: Path, quick=False):
    scfg=copy.deepcopy(cfg['synthetic']); allocations=validate_budget_grid(scfg)
    scfg['tasks']=int(scfg.get('appendix_tasks',scfg['tasks']))
    scfg['sampling_stride']=1
    scfg['empirical_trials']=0; scfg['quick_empirical_trials']=0
    scfg['runtime_tasks']=[]; scfg['quick_runtime_tasks']=[]
    nseed=int(scfg['total_budget_sequences_per_point_quick' if quick else 'total_budget_sequences_per_point'])
    nboot=int(scfg['total_budget_bootstrap_resamples_quick' if quick else 'total_budget_bootstrap_resamples'])
    seed_policy=str(scfg.get('total_budget_seed_policy','fixed')).lower()
    scfg['stability_guard']=bool(scfg.get('total_budget_stability_guard',False))
    scfg['min_relative_singular_value']=float(scfg.get('total_budget_min_relative_singular_value',0.01))
    max_draws=int(scfg.get('total_budget_max_seed_draws',100))
    seeds=[]; frames=[]; manifest=[]; attempted=set()
    fixed_start=int(scfg.get('total_budget_seed_start',cfg['seed']))
    for attempt in range(max_draws):
        if len(seeds)>=nseed:
            break
        seed=(secrets.randbelow(2_000_000_000) if seed_policy=='random' else fixed_start+attempt)
        if seed in attempted:
            continue
        attempted.add(seed); candidate=[]
        for budget in BUDGET_GRID:
            c=copy.deepcopy(scfg); c['total_budget']=budget
            d=synthetic_main.run_sequence(c,seed,.75,'recurrent',quick=False,policies=['vpm'])
            d['budget']=budget; d['feasible_allocation_witness']=str(allocations[budget])
            d['paired_task_id']=d['seed'].astype(str)+':'+d['task'].astype(str)+':recurrent:0.75'
            candidate.append(d)
        candidate_df=pd.concat(candidate,ignore_index=True)
        vpm=candidate_df[candidate_df.method=='VPM-RFW']
        minimum=float(vpm['design_min_relative_singular'].min())
        accepted=bool(minimum>=scfg['min_relative_singular_value'])
        manifest.append({'draw_index':len(manifest),'seed':seed,'accepted':accepted,
                         'accepted_sequence_index':len(seeds) if accepted else np.nan,
                         'min_design_relative_singular':minimum,
                         'threshold':scfg['min_relative_singular_value'],
                         'seed_policy':seed_policy,
                         'rejection_reason':'' if accepted else 'design_near_singular'})
        if accepted:
            seeds.append(seed); frames.extend(candidate)
    if len(seeds)!=nseed:
        raise RuntimeError(f'accepted only {len(seeds)}/{nseed} stable seeds after {max_draws} draws')
    bootstrap_seed=int(seeds[0]); seed_manifest=pd.DataFrame(manifest)
    raw=pd.concat(frames,ignore_index=True)
    if not raw['total_mode_budget'].astype(int).eq(raw['budget'].astype(int)).all():
        raise AssertionError('a method used a total mode budget different from the requested budget')
    seq=sequence_metrics(raw)
    expected={(b,s,m) for b in BUDGET_GRID for s in seeds for m in PAPER_METHODS}
    observed=set(map(tuple,seq[['budget','seed','method']].itertuples(index=False,name=None)))
    if observed != expected:
        raise AssertionError(f'unpaired or incomplete budget design: missing={sorted(expected-observed)[:5]}')
    summary=summarize_sequences(seq,nboot,bootstrap_seed)
    paired=paired_comparisons(seq,nboot,bootstrap_seed+1)
    improve=improvement_vs_low_budget(seq,nboot,bootstrap_seed+2)
    allocation=allocation_summary(raw)
    out=Path(out); (out/'raw').mkdir(parents=True,exist_ok=True); (out/'tables').mkdir(parents=True,exist_ok=True)
    raw.to_csv(out/'raw'/'synthetic_budget.csv',index=False)
    seed_manifest.to_csv(out/'tables'/'synthetic_budget_seed_manifest.csv',index=False)
    seq.to_csv(out/'tables'/'synthetic_budget_sequence_metrics.csv',index=False)
    pretty_method_names(summary).to_csv(out/'tables'/'synthetic_budget_summary.csv',index=False)
    _pretty_baselines(paired).to_csv(out/'tables'/'synthetic_budget_paired_comparison.csv',index=False)
    pretty_method_names(improve).to_csv(out/'tables'/'synthetic_budget_improvement_vs_36.csv',index=False)
    pretty_method_names(allocation).to_csv(out/'tables'/'synthetic_budget_allocation_summary.csv',index=False)
    paper_table(summary).to_csv(out/'tables'/'synthetic_budget_summary_paper.csv',index=False)
    budget_sensitivity_plot(summary,'wc_mse',out/'figures'/'synthetic_budget_wc_mse','Worst-case MSE',PAPER_METHODS)
    budget_sensitivity_plot(summary,'avg_mse',out/'figures'/'synthetic_budget_mean_mse','Mean MSE',PAPER_METHODS)
    (out/'BUDGET_SWEEP_REPORT.md').write_text(_report(summary,paired,seed_manifest),encoding='utf-8')
    return raw,summary,paired
