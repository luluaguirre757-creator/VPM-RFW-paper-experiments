from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .figures import _save
from .style import ALL_SOLVER_METHODS, MAIN_SAMPLING_METHODS, context, display_name, method_style

RFW_METHODS = list(ALL_SOLVER_METHODS)
SIGMAS = (0.0, 0.5)
KEY = ["seed", "task", "row_sigma"]


def _tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _require(df: pd.DataFrame, columns, source: Path) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _validate_solver_data(raw: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = KEY + ["sequence", "kind", "method", "wc_mse", "realized_weighted_cost",
                      "selected_history", "cumulative_cost", "cumulative_total_runtime"]
    _require(raw, required, source)
    sol = raw[raw["kind"].eq("solver") & raw["method"].isin(RFW_METHODS)].copy()
    sol = sol[sol["row_sigma"].astype(float).isin(SIGMAS)]
    if sol.empty:
        raise ValueError(f"{source} contains no RFW solver rows for sigma 0 and 0.5")
    dup = sol.duplicated(KEY + ["method"], keep=False)
    if dup.any():
        raise ValueError("duplicate solver task keys: " + str(sol.loc[dup, KEY + ["method"]].head().to_dict("records")))
    for sigma, group in sol.groupby("row_sigma"):
        reference = None
        for method in RFW_METHODS:
            keys = set(map(tuple, group.loc[group.method.eq(method), ["seed", "task"]].to_numpy()))
            if reference is None:
                reference = keys
            elif keys != reference:
                raise ValueError(f"unaligned task IDs for {method} at sigma={sigma}: "
                                 f"missing={len(reference-keys)}, extra={len(keys-reference)}")
    numeric = ["wc_mse", "realized_weighted_cost", "selected_history",
               "cumulative_cost", "cumulative_total_runtime"]
    for col in numeric:
        sol[col] = pd.to_numeric(sol[col], errors="coerce")
    bad_mse = ~np.isfinite(sol["wc_mse"])
    if bad_mse.any():
        raise ValueError(f"non-finite wc_mse rows: {int(bad_mse.sum())}")
    if "execution_status" in sol:
        failed = sol["execution_status"].fillna("").str.lower().str.contains("fail|error")
        if failed.any():
            raise ValueError(f"failed solver/reconstruction rows: {int(failed.sum())}")
    return sol.sort_values(["row_sigma", "method", "seed", "task"]).reset_index(drop=True)


def _bootstrap_mean(values: np.ndarray, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap input must be non-empty and finite")
    mean = values.mean(axis=0)
    if len(values) == 1:
        return mean, mean.copy(), mean.copy()
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(int(n_boot), len(values)))].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    return mean, lo, hi


def cumulative_mse_trajectories(sol: pd.DataFrame, n_boot=2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cumulative MSE per seed, then a seed-level percentile bootstrap."""
    d = sol.copy().sort_values(["row_sigma", "method", "seed", "task"])
    d["cumulative_wc_mse"] = d.groupby(["row_sigma", "method", "seed"])["wc_mse"].cumsum()
    rows = []
    finals = []
    for (sigma, method), g in d.groupby(["row_sigma", "method"], sort=False):
        pivot = g.pivot(index="seed", columns="task", values="cumulative_wc_mse").sort_index(axis=1)
        if pivot.isna().any().any():
            raise ValueError(f"sequence boundary/task alignment failure for {method}, sigma={sigma}")
        mean, lo, hi = _bootstrap_mean(pivot.to_numpy(), n_boot=n_boot, seed=1701 + int(float(sigma)*100))
        for task, m, l, h in zip(pivot.columns, mean, lo, hi):
            rows.append({"row_sigma": float(sigma), "method": method, "task": int(task),
                         "mean": m, "ci_lo": l, "ci_hi": h, "n_sequences": len(pivot)})
        for seed, value in pivot.iloc[:, -1].items():
            finals.append({"row_sigma": float(sigma), "method": method, "seed": seed,
                           "final_cumulative_wc_mse": float(value)})
    traj = pd.DataFrame(rows)
    final = pd.DataFrame(finals)
    summary = []
    for sigma in SIGMAS:
        sg = final[np.isclose(final.row_sigma, sigma)]
        means = sg.groupby("method")["final_cumulative_wc_mse"].mean()
        vpm = float(means["VPM-RFW"])
        for method in RFW_METHODS:
            value = float(means[method])
            summary.append({"row_sigma": sigma, "method": method,
                            "final_cumulative_worst_case_mse": value,
                            "vpm_reduction_percent": np.nan if method == "VPM-RFW" else 100.0*(value-vpm)/value,
                            "sequence_count": int(sg.loc[sg.method.eq(method), "seed"].nunique())})
    return traj, pd.DataFrame(summary)


def _plot_trajectory(summary, sigma, path, y, logy=False):
    g = summary[np.isclose(summary.row_sigma, sigma)]
    with context():
        fig, ax = plt.subplots(figsize=(3.35, 2.30))
        for method in RFW_METHODS:
            m = g[g.method.eq(method)].sort_values("task")
            st = method_style(method)
            markevery = max(1, int(np.ceil(len(m) / 7)))
            ax.plot(m.task, m["mean"], label=display_name(method), markevery=markevery,
                    markerfacecolor="white", markeredgewidth=.75, **st)
            ax.fill_between(m.task, m.ci_lo, m.ci_hi, color=st["color"], alpha=.09, linewidth=0, zorder=1)
        ax.set_xlabel("Task index")
        ax.set_ylabel(y)
        if logy:
            ax.set_yscale("log")
        ticks = np.unique(np.rint(np.linspace(g.task.min(), g.task.max(), 5)).astype(int))
        ax.set_xticks(ticks)
        ax.margins(x=.015)
        ax.legend(ncol=1, loc="best", borderaxespad=.2, handlelength=1.85,
                  handletextpad=.4, columnspacing=.7, labelspacing=.27)
        _save(fig, path)


def _paired_wide(sol, baseline):
    pair = sol[sol.method.isin([baseline, "VPM-RFW"])].pivot(
        index=KEY, columns="method", values=["selected_history", "realized_weighted_cost"])
    if pair.isna().any().any():
        raise ValueError(f"missing paired task for {baseline}")
    eligible = ((pair[("selected_history", baseline)] >= 0) &
                (pair[("selected_history", "VPM-RFW")] >= 0))
    disagree = eligible & (pair[("selected_history", baseline)] != pair[("selected_history", "VPM-RFW")])
    return pair.loc[disagree].reset_index()


def history_disagreement_summary(sol: pd.DataFrame, n_boot=2000):
    detail = {}
    rows = []
    for baseline in ["Prev-RFW", "Nearest-RFW"]:
        d = _paired_wide(sol, baseline)
        detail[baseline] = d
        for sigma in SIGMAS:
            g = d[np.isclose(d.row_sigma, sigma)]
            if g.empty:
                rows.append({"baseline": baseline, "row_sigma": sigma,
                             "disagreement_task_count": 0, "sequence_count": 0,
                             "baseline_mean_cost": np.nan, "baseline_ci_lo": np.nan, "baseline_ci_hi": np.nan,
                             "vpm_mean_cost": np.nan, "vpm_ci_lo": np.nan, "vpm_ci_hi": np.nan,
                             "relative_reduction_percent": np.nan,
                             "reduction_ci_lo": np.nan, "reduction_ci_hi": np.nan})
                continue
            seq = g.groupby("seed")[[
                ("realized_weighted_cost", baseline), ("realized_weighted_cost", "VPM-RFW")]].mean()
            vals = seq.to_numpy(float)
            bmean, blo, bhi = _bootstrap_mean(vals[:, 0], n_boot, 3100 + int(sigma*10))
            vmean, vlo, vhi = _bootstrap_mean(vals[:, 1], n_boot, 3200 + int(sigma*10))
            rng = np.random.default_rng(3300 + int(sigma*10))
            idx = rng.integers(0, len(vals), size=(int(n_boot), len(vals)))
            bm, vm = vals[idx, 0].mean(axis=1), vals[idx, 1].mean(axis=1)
            reductions = 100.0 * (bm-vm) / bm
            rlo, rhi = np.quantile(reductions, [.025, .975])
            rows.append({"baseline": baseline, "row_sigma": sigma,
                         "disagreement_task_count": len(g), "sequence_count": len(seq),
                         "baseline_mean_cost": float(bmean[0]), "baseline_ci_lo": float(blo[0]), "baseline_ci_hi": float(bhi[0]),
                         "vpm_mean_cost": float(vmean[0]), "vpm_ci_lo": float(vlo[0]), "vpm_ci_hi": float(vhi[0]),
                         "relative_reduction_percent": 100.0*(float(bmean[0])-float(vmean[0]))/float(bmean[0]),
                         "reduction_ci_lo": float(rlo), "reduction_ci_hi": float(rhi)})
    return pd.DataFrame(rows), detail


def _plot_disagreement(summary, baseline, path):
    g = summary[summary.baseline.eq(baseline)].sort_values("row_sigma")
    x = np.arange(len(g)); width = .34
    with context():
        fig, ax = plt.subplots(figsize=(3.15, 2.30))
        for offset, method, prefix in [(-width/2, baseline, "baseline"), (width/2, "VPM-RFW", "vpm")]:
            st = method_style(method); means = g[f"{prefix}_mean_cost"].to_numpy(float)
            err = np.vstack([means-g[f"{prefix}_ci_lo"], g[f"{prefix}_ci_hi"]-means])
            ax.bar(x+offset, means, width, label=display_name(method), color=st["color"], alpha=.82,
                   edgecolor="white", linewidth=.55, yerr=err, capsize=2.5,
                   error_kw={"elinewidth": 1.0, "capthick": 1.0})
        ax.set_xticks(x, [rf"$\sigma_a={s:g}$" for s in g.row_sigma])
        ax.set_ylabel("Mean realized solver cost")
        ax.legend(loc="best")
        for pos, count in zip(x, g.disagreement_task_count):
            if int(count) == 0:
                ax.text(pos, .04, "no disagreements", rotation=90, ha="center", va="bottom",
                        transform=ax.get_xaxis_transform(), fontsize=6.5, color="#666666")
        _save(fig, path)


def pareto_summary(sol: pd.DataFrame, n_boot=2000):
    seq = sol.groupby(["row_sigma", "method", "seed"], as_index=False).agg(
        mean_realized_solver_cost=("realized_weighted_cost", "mean"),
        cumulative_worst_case_mse=("wc_mse", "sum"))
    rows = []
    for (sigma, method), g in seq.groupby(["row_sigma", "method"], sort=False):
        x, xl, xh = _bootstrap_mean(g.mean_realized_solver_cost.to_numpy(), n_boot, 4100+int(float(sigma)*10))
        y, yl, yh = _bootstrap_mean(g.cumulative_worst_case_mse.to_numpy(), n_boot, 4200+int(float(sigma)*10))
        rows.append({"row_sigma": float(sigma), "method": method, "sequence_count": len(g),
                     "mean_realized_solver_cost": float(x[0]), "cost_ci_lo": float(xl[0]), "cost_ci_hi": float(xh[0]),
                     "mean_cumulative_worst_case_mse": float(y[0]), "mse_ci_lo": float(yl[0]), "mse_ci_hi": float(yh[0])})
    return pd.DataFrame(rows)


def _plot_pareto(summary, sigma, path):
    g = summary[np.isclose(summary.row_sigma, sigma)]
    with context():
        fig, ax = plt.subplots(figsize=(3.35, 2.30))
        for _, r in g.iterrows():
            st = method_style(r.method)
            ax.errorbar(r.mean_realized_solver_cost, r.mean_cumulative_worst_case_mse,
                        xerr=[[r.mean_realized_solver_cost-r.cost_ci_lo], [r.cost_ci_hi-r.mean_realized_solver_cost]],
                        yerr=[[r.mean_cumulative_worst_case_mse-r.mse_ci_lo], [r.mse_ci_hi-r.mean_cumulative_worst_case_mse]],
                        fmt=st["marker"], color=st["color"], markerfacecolor="white",
                        markeredgewidth=.8, markersize=st["markersize"], capsize=2.5, label=display_name(r.method))
        if (g.mean_cumulative_worst_case_mse.max()/g.mean_cumulative_worst_case_mse.min()) > 20:
            ax.set_yscale("log")
        ax.set_xlabel("Mean realized solver cost")
        ax.set_ylabel("Mean cumulative worst-case MSE")
        ax.legend(loc="best")
        _save(fig, path)


def build_sampling_quality_summary(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw[raw.method.isin(MAIN_SAMPLING_METHODS)].copy()
    keys = None
    for method in MAIN_SAMPLING_METHODS:
        mk = set(map(tuple, d.loc[d.method.eq(method), KEY].to_numpy()))
        keys = mk if keys is None else keys & mk
    if not keys:
        raise ValueError("sampling methods have no aligned task IDs")
    aligned = pd.MultiIndex.from_tuples(keys, names=KEY)
    d = d.set_index(KEY).loc[lambda x: x.index.isin(aligned)].reset_index()
    rows = []
    for (sigma, method), g in d.groupby(["row_sigma", "method"]):
        values = pd.to_numeric(g.wc_mse, errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite sampling MSE for {method}, sigma={sigma}")
        row = {"row_sigma": float(sigma), "method": display_name(method),
               "task_count": len(g), "sequence_count": g.seed.nunique(),
               "mean_worst_case_mse": values.mean(), "median_worst_case_mse": np.median(values),
               "p95_worst_case_mse": np.quantile(values, .95)}
        if "avg_mse" in g:
            row["mean_mse"] = pd.to_numeric(g.avg_mse, errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def build_ksc_main_summary(raw: pd.DataFrame, n_boot=2000) -> pd.DataFrame:
    required = ["method", "band", "nmse_db", "psnr", "ssim", "realized_weighted_cost"]
    _require(raw, required, Path("raw/hsi_main.csv"))
    rows=[]
    for method, g in raw.groupby("method", sort=False):
        row={"method": display_name(method), "band_count": g.band.nunique()}
        for col in ["nmse_db", "psnr", "ssim", "realized_weighted_cost"]:
            vals=g.groupby("band")[col].mean().to_numpy(float)
            finite=np.isfinite(vals)
            if not finite.any():
                # Sampling-only methods have no solver-work definition. Preserve
                # that structural absence as NA rather than inventing zero work.
                row.update({f"{col}_mean":np.nan,f"{col}_ci_lo":np.nan,f"{col}_ci_hi":np.nan})
                continue
            if not finite.all():
                raise ValueError(f"partially missing KSC metric {col} for {method}")
            mean, lo, hi=_bootstrap_mean(vals,n_boot,5100)
            row.update({f"{col}_mean":float(mean[0]),f"{col}_ci_lo":float(lo[0]),f"{col}_ci_hi":float(hi[0])})
        rows.append(row)
    return pd.DataFrame(rows)


def _dominance_text(pareto, sigma):
    g=pareto[np.isclose(pareto.row_sigma,sigma)].set_index("method")
    v=g.loc["VPM-RFW"]
    dominated=[]
    for method in RFW_METHODS:
        if method=="VPM-RFW": continue
        r=g.loc[method]
        if (v.mean_realized_solver_cost <= r.mean_realized_solver_cost and
                v.mean_cumulative_worst_case_mse <= r.mean_cumulative_worst_case_mse and
                (v.mean_realized_solver_cost < r.mean_realized_solver_cost or
                 v.mean_cumulative_worst_case_mse < r.mean_cumulative_worst_case_mse)):
            dominated.append(display_name(method))
    return ", ".join(dominated) if dominated else "none"


def generate_paper_figures(results_dir, n_boot=2000):
    results_dir=Path(results_dir).resolve(); source=results_dir/"raw"/"synthetic_main.csv"
    if not source.is_file():
        raise FileNotFoundError(f"required full-run input not found: {source}")
    raw=pd.read_csv(source); sol=_validate_solver_data(raw,source)
    out=results_dir/"paper_figures"; out.mkdir(parents=True,exist_ok=True)
    tables=results_dir/"tables"; tables.mkdir(parents=True,exist_ok=True)
    traj,cumulative=cumulative_mse_trajectories(sol,n_boot)
    cumulative.to_csv(tables/"cumulative_worst_case_mse_rfw_summary.csv",index=False)
    for sigma in SIGMAS:
        tag=_tag(sigma)
        _plot_trajectory(traj,sigma,out/f"cumulative_worst_case_mse_rfw_sigma_{tag}","Cumulative worst-case MSE",logy=True)
    disagreement,_=history_disagreement_summary(sol,n_boot)
    disagreement.to_csv(tables/"history_disagreement_effectiveness.csv",index=False)
    _plot_disagreement(disagreement,"Prev-RFW",out/"history_disagreement_cost_prev")
    _plot_disagreement(disagreement,"Nearest-RFW",out/"history_disagreement_cost_nearest")
    pareto=pareto_summary(sol,n_boot); pareto.to_csv(tables/"pareto_solver_cost_vs_cumulative_mse_summary.csv",index=False)
    for sigma in SIGMAS:
        tag=_tag(sigma); _plot_pareto(pareto,sigma,out/f"pareto_solver_cost_vs_cumulative_mse_sigma_{tag}")
        for col,stem,label in [("cumulative_cost","solver_cumulative_unit_oracle_count","Cumulative oracle count"),
                               ("cumulative_total_runtime","synthetic_wall_time","Cumulative wall-clock time (s)")]:
            rows=[]
            for method in RFW_METHODS:
                p=sol[np.isclose(sol.row_sigma,sigma)&sol.method.eq(method)].pivot(index="seed",columns="task",values=col)
                mean,lo,hi=_bootstrap_mean(p.to_numpy(float),n_boot,6100)
                rows += [{"row_sigma":sigma,"method":method,"task":int(t),"mean":m,"ci_lo":l,"ci_hi":h}
                         for t,m,l,h in zip(p.columns,mean,lo,hi)]
            _plot_trajectory(pd.DataFrame(rows),sigma,out/f"{stem}_sigma_{tag}",label)
    sampling=build_sampling_quality_summary(raw)
    sampling.to_csv(tables/"sampling_quality_summary.csv",index=False)
    ksc_source=results_dir/"raw"/"hsi_main.csv"
    if ksc_source.is_file():
        build_ksc_main_summary(pd.read_csv(ksc_source),n_boot).to_csv(tables/"ksc_main_summary.csv",index=False)
    manifest=[]
    roles={
        "cumulative_worst_case_mse_rfw":"main", "history_disagreement_cost":"main",
        "solver_cumulative_unit_oracle_count":"main", "synthetic_wall_time":"main",
        "pareto_solver_cost_vs_cumulative_mse":"candidate"}
    for pdf in sorted(out.glob("*.pdf")):
        stem=pdf.stem; role=next((v for k,v in roles.items() if stem.startswith(k)),"appendix")
        manifest.append({"figure_name":pdf.name,"source_result_file":str(source.relative_to(results_dir)),
                         "plotting_script/function":"generate_paper_figures.py / vpmrfw.plotting.paper_figures.generate_paper_figures",
                         "description":stem.replace("_"," "),"paper_role":role})
    pd.DataFrame(manifest).to_csv(out/"paper_figure_manifest.csv",index=False)
    lines=["# Paper Figure Report","",f"Source: `{source.relative_to(results_dir)}`",
           f"Independent sequences: {sol.seed.nunique()}; tasks per method/setting: {sol.task.nunique()}",
           "Bootstrap: percentile bootstrap over seed/sequence (never over task rows).",
           "Missing/failed tasks: none (strict alignment and finite-value checks passed).","",
           "## Generated figure inventory","",
           "Every PDF below has a same-stem PNG preview; all are generated from `raw/synthetic_main.csv`.",""]
    for item in manifest:
        lines.append(f"- `paper_figures/{item['figure_name']}` — {item['paper_role']}; "
                     f"{item['plotting_script/function']}.")
    lines += ["",
           "## Cumulative MSE (log y-axis; Main paper)",""]
    for sigma in SIGMAS:
        g=cumulative[np.isclose(cumulative.row_sigma,sigma)].set_index("method")
        lines.append(f"- sigma_a={sigma:g}: VPM final cumulative MSE {g.loc['VPM-RFW','final_cumulative_worst_case_mse']:.6g}; "
                     f"reduction vs Cold/Prev/Nearest = {g.loc['Cold-RFW','vpm_reduction_percent']:.3f}% / "
                     f"{g.loc['Prev-RFW','vpm_reduction_percent']:.3f}% / {g.loc['Nearest-RFW','vpm_reduction_percent']:.3f}%.")
    lines += ["","## History disagreement (linear axis; Main paper)",""]
    for _,r in disagreement.iterrows():
        lines.append(f"- sigma_a={r.row_sigma:g}, VPM vs {display_name(r.baseline)}: {int(r.disagreement_task_count)} tasks, "
                     f"{int(r.sequence_count)} sequences, cost reduction {r.relative_reduction_percent:.3f}% "
                     f"(95% CI {r.reduction_ci_lo:.3f}% to {r.reduction_ci_hi:.3f}%).")
    lines += ["","## Cost-quality Pareto (Candidate only)",""]
    for sigma in SIGMAS:
        pg=pareto[np.isclose(pareto.row_sigma,sigma)]
        scale="log y-axis" if pg.mean_cumulative_worst_case_mse.max()/pg.mean_cumulative_worst_case_mse.min()>20 else "linear axes"
        lines.append(f"- sigma_a={sigma:g} ({scale}): VPM Pareto-dominates: {_dominance_text(pareto,sigma)}.")
    lines += ["","## Other generated figures","",
              "- Cumulative oracle count and synthetic wall-clock: Main paper; wall-clock uses cumulative_total_runtime, including selection/certificate overhead.",
              "- Pareto plots: Candidate only.","- PDF is paper-facing; matching PNG files are previews."]
    (results_dir/"PAPER_FIGURE_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return {"output_dir":out,"cumulative":cumulative,"disagreement":disagreement,"pareto":pareto,"manifest":pd.DataFrame(manifest)}
