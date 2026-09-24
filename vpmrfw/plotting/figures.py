from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
from .style import context, display_name, compact_display_name, method_style


def _save(fig,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',pad_inches=0.025)
    fig.savefig(path.with_suffix('.png'),bbox_inches='tight',pad_inches=0.025)
    plt.close(fig)


def pretty_method_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a presentation copy with compact display names; raw data stay unchanged."""
    out=df.copy()
    if 'method' in out.columns:
        out['method']=out['method'].map(display_name)
    return out


def _ordered_methods(df, methods=None):
    present=list(dict.fromkeys(df['method'].astype(str).tolist())) if 'method' in df else []
    if methods is None:
        return present
    return [m for m in methods if m in set(present)]


def line_by_method(df,x,y,path,ylabel=None,xlabel=None,methods=None,ci=True,logy=False):
    """Conference-style task trajectory: point-line mean with a light 95% CI ribbon."""
    if df.empty or y not in df.columns: return
    methods=_ordered_methods(df,methods)
    if not methods: return
    with context():
        if len(methods)<=3:
            figsize=(3.15,2.30)
        elif len(methods)==4:
            figsize=(3.35,2.30)
        elif len(methods)==5:
            figsize=(3.65,2.35)
        else:
            figsize=(3.95,2.40)
        fig,ax=plt.subplots(figsize=figsize)
        for m in methods:
            g=df[df.method==m]
            if g.empty: continue
            gg=g.groupby(x)[y].agg(['mean','sem']).reset_index()
            st=method_style(m)
            markevery=max(1,int(np.ceil(max(len(gg),1)/7)))
            ax.plot(gg[x],gg['mean'],label=display_name(m),
                    color=st['color'],marker=st['marker'],linestyle=st['linestyle'],
                    linewidth=st['linewidth'],markersize=st['markersize'],zorder=st['zorder'],
                    markevery=markevery,markeredgewidth=0.75,markerfacecolor='white')
            if ci and len(gg)>1 and gg['sem'].notna().any():
                err=1.96*gg['sem'].fillna(0).to_numpy(float)
                mu=gg['mean'].to_numpy(float)
                ax.fill_between(gg[x],mu-err,mu+err,color=st['color'],alpha=.09,linewidth=0,zorder=1)
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        if logy:
            ax.set_yscale('log')
        xv=np.sort(pd.unique(df[x].dropna())) if x in df.columns else np.asarray([])
        xv=np.asarray(xv)
        is_integer_x=(xv.size and np.issubdtype(xv.dtype,np.number)
                      and np.allclose(xv,np.round(xv)))
        if str(x).lower()=='task' and is_integer_x and len(xv)>=8:
            tick_idx=np.unique(np.rint(np.linspace(0,len(xv)-1,5)).astype(int))
            ax.set_xticks(xv[tick_idx])
        elif is_integer_x:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))
        ax.margins(x=0.015)
        ncol=1 if len(methods)<=4 else 2
        ax.legend(ncol=ncol,loc='best',frameon=False,fontsize=7.7,
                  borderaxespad=0.20,handlelength=1.85,handletextpad=0.40,
                  columnspacing=0.70,labelspacing=0.27)
        _save(fig,path)


def budget_sensitivity_plot(summary, metric, path, ylabel, methods):
    """Plot numeric total budgets with sequence-bootstrap confidence intervals."""
    numeric_sensitivity_plot(summary,'budget',metric,path,'Total sampling budget',ylabel,methods)


def numeric_sensitivity_plot(summary, xcol, metric, path, xlabel, ylabel, methods, logy='auto'):
    """Paper-style numeric sensitivity curve with precomputed bootstrap CIs."""
    if summary.empty or metric not in summary.columns:
        return
    methods=_ordered_methods(summary,methods)
    if not methods:
        return
    with context():
        fig,ax=plt.subplots(figsize=(3.35,2.35))
        for m in methods:
            g=summary[summary.method==m].sort_values(xcol)
            if g.empty:
                continue
            x=g[xcol].to_numpy(float); y=g[metric].to_numpy(float)
            lo=g[f'{metric}_ci_lo'].to_numpy(float); hi=g[f'{metric}_ci_hi'].to_numpy(float)
            st=method_style(m)
            ax.plot(x,y,label=display_name(m),color=st['color'],marker=st['marker'],
                    linestyle=st['linestyle'],linewidth=st['linewidth'],
                    markersize=st['markersize'],markeredgewidth=.75,
                    markerfacecolor='white',zorder=st['zorder'])
            ax.fill_between(x,lo,hi,color=st['color'],alpha=.09,linewidth=0,zorder=1)
        xvalues=np.sort(summary[xcol].dropna().unique().astype(float))
        ax.set_xticks(xvalues)
        ax.tick_params(axis='x',labelrotation=35 if len(xvalues)>7 else 0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        finite=summary[metric].replace([np.inf,-np.inf],np.nan).dropna()
        if logy is True or (logy=='auto' and len(finite) and
                            finite.max()/max(finite.min(),np.finfo(float).tiny) >= 100):
            ax.set_yscale('log')
        ax.margins(x=.015)
        ax.legend(loc='best',frameon=False)
        _save(fig,path)


def sensitivity_series_plot(summary, xcol, series, path, xlabel, ylabel):
    """Numeric sensitivity plot for non-method explanatory series."""
    colors=['#4C78A8','#E45756','#59A14F','#B279A2']
    markers=['o','s','^','D']
    with context():
        fig,ax=plt.subplots(figsize=(3.35,2.35))
        for i,(metric,label) in enumerate(series):
            if metric not in summary:
                continue
            g=summary.sort_values(xcol); x=g[xcol].to_numpy(float); y=g[metric].to_numpy(float)
            lo=g.get(f'{metric}_ci_lo',pd.Series(y)).to_numpy(float)
            hi=g.get(f'{metric}_ci_hi',pd.Series(y)).to_numpy(float)
            ax.plot(x,y,label=label,color=colors[i%len(colors)],marker=markers[i%len(markers)],
                    linewidth=1.45,markersize=4.3,markerfacecolor='white',markeredgewidth=.75)
            ax.fill_between(x,lo,hi,color=colors[i%len(colors)],alpha=.09,linewidth=0)
        xv=np.sort(summary[xcol].dropna().unique().astype(float)); ax.set_xticks(xv)
        ax.tick_params(axis='x',labelrotation=35 if len(xv)>7 else 0)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_ylim(-.03,1.03)
        ax.legend(loc='best',frameon=False); ax.margins(x=.015); _save(fig,path)


def scatter_score_realized(df,path,score_col='score_or_cert_cost',realized_col='realized_weighted_cost',certified=False,xlabel=None,ylabel=None):
    g=df[np.isfinite(df[score_col]) & np.isfinite(df[realized_col])]
    if g.empty: return
    with context():
        fig,ax=plt.subplots(figsize=(3.20,2.28))
        ax.scatter(g[score_col],g[realized_col],s=17,alpha=.52,facecolors='none',edgecolors='#4C78A8',linewidths=.65)
        lo=min(float(g[score_col].min()),float(g[realized_col].min()),0.0)
        hi=max(float(g[score_col].max()),float(g[realized_col].max()))
        ax.plot([lo,hi],[lo,hi],color='#666666',ls='--',lw=1.0,label=r'$y=x$')
        ax.set_xlabel(xlabel or ('Certified cost' if certified else 'VPM score proxy'))
        ax.set_ylabel(ylabel or 'Realized solver cost')
        ax.legend(loc='best')
        _save(fig,path)


def point_by_method(df,y,path,ylabel=None,logy=False,order=None,methods=None):
    """Vertical categorical dot-whisker with mean and 95% confidence interval."""
    if df.empty or y not in df.columns: return
    g=df[['method',y]].copy(); g=g[np.isfinite(g[y])]
    if g.empty: return
    preferred=methods if methods is not None else order
    meth=_ordered_methods(g,preferred)
    if not meth: return
    rows=[]
    for m in meth:
        v=g.loc[g.method==m,y].astype(float)
        rows.append((m,float(v.mean()),float(v.sem()) if len(v)>1 else 0.0))
    with context():
        if len(rows)<=3:
            figsize=(2.95,2.25)
        elif len(rows)==4:
            figsize=(3.30,2.30)
        elif len(rows)==5:
            figsize=(3.70,2.35)
        elif len(rows)==6:
            figsize=(4.10,2.40)
        else:
            figsize=(4.45,2.45)
        fig,ax=plt.subplots(figsize=figsize,constrained_layout=False)
        spacing=0.82
        xpos=np.arange(len(rows),dtype=float)*spacing
        for pos,(m,mu,se) in zip(xpos,rows):
            st=method_style(m); err=1.96*(0.0 if not np.isfinite(se) else se)
            ax.errorbar(pos,mu,yerr=err,fmt='none',color=st['color'],
                        elinewidth=1.55,capsize=3.0,capthick=1.35,zorder=2)
            point_size=5.3 if m=='VPM-RFW' else 4.8
            ax.plot(pos,mu,marker=st['marker'],linestyle='none',color=st['color'],
                    markerfacecolor='white',markeredgewidth=.8,
                    markersize=point_size,zorder=3)
        ax.set_xticks(xpos)
        label_fn=display_name if len(rows)<=3 else compact_display_name
        ax.set_xticklabels([label_fn(r[0]) for r in rows],
                           rotation=0,ha='center',va='top',fontsize=7.7)
        for label in ax.get_xticklabels():
            label.set_linespacing(0.85)
        ax.set_xlabel('')
        ax.set_ylabel(ylabel or y,fontsize=9)
        ax.tick_params(axis='y',labelsize=8)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.grid(False)
        if logy and all(r[1]>0 for r in rows):
            ax.set_yscale('log')
            ax.yaxis.set_minor_locator(NullLocator())
        ax.margins(x=.07 if len(rows)<=3 else .055)
        if len(rows)>=6:
            fig.subplots_adjust(left=0.14,right=0.99,top=0.975,bottom=0.215)
        elif len(rows)<=3:
            fig.subplots_adjust(left=0.15,right=0.985,top=0.975,bottom=0.17)
        else:
            fig.subplots_adjust(left=0.15,right=0.985,top=0.975,bottom=0.205)
        _save(fig,path)
