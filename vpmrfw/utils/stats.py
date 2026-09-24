from __future__ import annotations
import numpy as np, pandas as pd


def bootstrap_mean_ci(df: pd.DataFrame, group_cols, value_cols, unit_col='seed', n_boot=1000, seed=0):
    """Mean and percentile 95% CI by independent experimental unit.

    The function first averages repeated rows within each independent unit, then bootstraps
    units. This avoids treating tasks from the same synthetic sequence as independent samples.
    """
    if df.empty: return pd.DataFrame()
    rng=np.random.default_rng(seed); out=[]
    group_cols=list(group_cols); value_cols=[v for v in list(value_cols) if v in df.columns]
    if not value_cols:
        return df[group_cols].drop_duplicates().reset_index(drop=True)
    for keys,g in df.groupby(group_cols,dropna=False):
        if not isinstance(keys,tuple): keys=(keys,)
        unit=g.groupby(unit_col)[value_cols].mean(numeric_only=True).dropna(how='all')
        row=dict(zip(group_cols,keys)); row['n_units']=len(unit)
        for v in value_cols:
            x=unit[v].dropna().to_numpy(float) if v in unit else np.array([])
            if not len(x):
                row[f'{v}_mean']=row[f'{v}_ci_lo']=row[f'{v}_ci_hi']=np.nan; continue
            row[f'{v}_mean']=float(np.mean(x))
            if len(x)==1:
                lo=hi=float(x[0])
            else:
                boot=np.mean(rng.choice(x,size=(int(n_boot),len(x)),replace=True),axis=1)
                lo,hi=np.quantile(boot,[0.025,0.975])
            row[f'{v}_ci_lo']=float(lo); row[f'{v}_ci_hi']=float(hi)
        out.append(row)
    return pd.DataFrame(out)


def contiguous_block_bootstrap(df: pd.DataFrame, group_col='method', index_col='band', value_cols=('nmse_db',),
                               block_length=8, n_boot=1000, seed=0):
    """Contiguous-block bootstrap for correlated spectral-band sequences."""
    if df.empty: return pd.DataFrame()
    rng=np.random.default_rng(seed); out=[]; L=max(1,int(block_length))
    for key,g in df.sort_values(index_col).groupby(group_col):
        row={group_col:key,'n_bands':len(g)}
        for v in value_cols:
            x=g[v].to_numpy(float); n=len(x)
            if not n: row[f'{v}_mean']=row[f'{v}_ci_lo']=row[f'{v}_ci_hi']=np.nan; continue
            row[f'{v}_mean']=float(np.mean(x))
            if n==1: lo=hi=float(x[0])
            else:
                vals=[]
                starts=np.arange(max(1,n-L+1))
                for _ in range(int(n_boot)):
                    sample=[]
                    while len(sample)<n:
                        st=int(rng.choice(starts)); sample.extend(x[st:min(st+L,n)].tolist())
                    vals.append(float(np.mean(sample[:n])))
                lo,hi=np.quantile(vals,[0.025,0.975])
            row[f'{v}_ci_lo']=float(lo); row[f'{v}_ci_hi']=float(hi)
        out.append(row)
    return pd.DataFrame(out)
