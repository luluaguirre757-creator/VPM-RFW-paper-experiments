#!/usr/bin/env python3
"""Redraw only the figures used by the paper from existing experiment results.

This script never runs an experiment.  It reads the frozen CSV outputs, applies
the requested paper-facing typography/labels, and writes PDF figures only.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _same_executable(a: Path, b: Path) -> bool:
    try:
        return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))
    except OSError:
        return False


if VENV_PYTHON.is_file() and not _same_executable(Path(sys.executable), VENV_PYTHON):
    completed = subprocess.run(
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=str(ROOT),
        check=False,
    )
    raise SystemExit(completed.returncode)


MPL_CACHE = ROOT / "tmp" / "matplotlib_redraw"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, ScalarFormatter  # noqa: E402

from vpmrfw.plotting import figures as plot_helpers  # noqa: E402
from vpmrfw.plotting import paper_figures as paper  # noqa: E402
from vpmrfw.plotting import style  # noqa: E402


MAIN_RUN = ROOT / "results" / "synthetic"
BUDGET_RUN = ROOT / "results" / "synthetic-budget"
DRIFT_RUN = ROOT / "results" / "synthetic-drift"
EXPECTED_SCORE_POINTS: int | None = 600
SIGMAS = (0.0, 0.5)


def _configure_paper_style() -> None:
    """Use Times New Roman and increase standard text by two points from the original."""
    style.ICLR_RC.update({
        "font.size": 10.2,
        "axes.labelsize": 10.3,
        "axes.titlesize": 10.6,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "legend.fontsize": 9.1,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    })
    style.METHOD_LABELS["Nearest-RFW"] = "TD-RFW"
    style.COMPACT_METHOD_LABELS["Nearest-RFW"] = "TD-\nRFW"


def _check_paper_font(*, allow_fallback: bool) -> None:
    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
    except ValueError as exc:
        if not allow_fallback:
            raise RuntimeError(
                "Times New Roman is required for exact paper typography. "
                "Install the font or use --allow-font-fallback for a non-final smoke run."
            ) from exc
        print("[warning] Times New Roman unavailable; using a fallback font.")


def _scientific_if_large(ax, axis: str = "y", threshold: float = 1e3) -> None:
    """Use scientific notation on large linear axes without rotating labels."""
    if (axis == "y" and ax.get_yscale() != "linear") or (
        axis == "x" and ax.get_xscale() != "linear"
    ):
        return
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    if max(abs(lo), abs(hi)) < threshold:
        return
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def _expand_top(ax, factor: float = 1.16) -> None:
    """Reserve a little top-corner space for a legend without changing data."""
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log" and lo > 0:
        ax.set_ylim(lo, hi * factor)
    else:
        ax.set_ylim(lo, lo + (hi - lo) * factor)


def _save_pdf_only(fig, path) -> None:
    path = Path(path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required existing result is missing: {path}")
    return path


def _trajectory_summary(sol: pd.DataFrame, value: str, sigma: float, n_boot: int) -> pd.DataFrame:
    rows = []
    for method in paper.RFW_METHODS:
        pivot = sol[
            np.isclose(sol.row_sigma, sigma) & sol.method.eq(method)
        ].pivot(index="seed", columns="task", values=value)
        if pivot.empty or pivot.isna().any().any():
            raise ValueError(f"unaligned {value} data for {method}, sigma={sigma}")
        mean, lo, hi = paper._bootstrap_mean(pivot.to_numpy(float), n_boot, 6100)
        rows.extend(
            {"row_sigma": sigma, "method": method, "task": int(task),
             "mean": m, "ci_lo": l, "ci_hi": h}
            for task, m, l, h in zip(pivot.columns, mean, lo, hi)
        )
    return pd.DataFrame(rows)


def _plot_pareto_with_requested_label(summary: pd.DataFrame, sigma: float, path: Path) -> None:
    g = summary[np.isclose(summary.row_sigma, sigma)]
    with style.context():
        fig, ax = plt.subplots(figsize=(3.35, 2.30))
        for _, row in g.iterrows():
            st = style.method_style(row.method)
            ax.errorbar(
                row.mean_realized_solver_cost,
                row.mean_cumulative_worst_case_mse,
                xerr=[[row.mean_realized_solver_cost - row.cost_ci_lo],
                      [row.cost_ci_hi - row.mean_realized_solver_cost]],
                yerr=[[row.mean_cumulative_worst_case_mse - row.mse_ci_lo],
                      [row.mse_ci_hi - row.mean_cumulative_worst_case_mse]],
                fmt=st["marker"], color=st["color"], markerfacecolor="white",
                markeredgewidth=.8, markersize=st["markersize"], capsize=2.5,
                label=style.display_name(row.method),
            )
        if g.mean_cumulative_worst_case_mse.max() / g.mean_cumulative_worst_case_mse.min() > 20:
            ax.set_yscale("log")
        ax.set_xlabel("Mean realized solver cost")
        ax.set_ylabel("Worst-case analytical LS MSE")
        ax.tick_params(axis="x", labelrotation=0)
        _scientific_if_large(ax, "y")
        _expand_top(ax, 1.16 if np.isclose(sigma, 0.0) else 20.0)
        ax.legend(loc="upper left" if np.isclose(sigma, 0.0) else "upper right")
        _save_pdf_only(fig, path)


def _plot_disagreement(summary: pd.DataFrame, baseline: str, path: Path) -> None:
    """Plot one Figure 6 comparison using the color-blind-safe palette."""
    g = summary[summary.baseline.eq(baseline)].sort_values("row_sigma")
    x = np.arange(len(g))
    width = .34
    with style.context():
        fig, ax = plt.subplots(figsize=(3.15, 2.30))
        colors = {baseline: "#4C78A8", "VPM-RFW": "#E69F00"}
        for offset, method, prefix in [
            (-width / 2, baseline, "baseline"),
            (width / 2, "VPM-RFW", "vpm"),
        ]:
            means = g[f"{prefix}_mean_cost"].to_numpy(float)
            err = np.vstack([
                means - g[f"{prefix}_ci_lo"],
                g[f"{prefix}_ci_hi"] - means,
            ])
            ax.bar(
                x + offset, means, width, label=style.display_name(method),
                color=colors[method], alpha=.88, edgecolor="white", linewidth=.55,
                yerr=err, capsize=2.5,
                error_kw={"elinewidth": 1.0, "capthick": 1.0},
            )
        ax.set_xticks(x, [rf"$\sigma_a={s:g}$" for s in g.row_sigma])
        ax.tick_params(axis="x", labelrotation=0)
        ax.set_ylabel("Mean realized solver cost")
        _expand_top(ax, 1.20)
        ax.legend(loc="upper right")
        for pos, count in zip(x, g.disagreement_task_count):
            if int(count) == 0:
                ax.text(
                    pos, .04, "no disagreements", rotation=90,
                    ha="center", va="bottom", transform=ax.get_xaxis_transform(),
                    fontsize=8.5, color="#666666",
                )
        _save_pdf_only(fig, path)


def _plot_score_ratio(main_raw: pd.DataFrame, path: Path) -> None:
    """Reproduce the paper's selected-action score/cost ratios from synthetic_main."""
    source = MAIN_RUN / "raw" / "synthetic_main.csv"
    data = main_raw[
        main_raw.kind.eq("solver") & main_raw.method.eq("VPM-RFW")
    ].copy()
    cold = main_raw[
        main_raw.kind.eq("solver") & main_raw.method.eq("Cold-RFW")
    ][["seed", "task", "row_sigma", "realized_weighted_cost"]].rename(
        columns={"realized_weighted_cost": "cold_realized_cost"}
    )
    data = data.merge(
        cold, on=["seed", "task", "row_sigma"], how="left", validate="one_to_one"
    )
    required = {"selection_score_ratio_to_cold", "realized_weighted_cost", "cold_realized_cost"}
    if missing := required - set(data.columns):
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    data["realized_cost_ratio_to_cold"] = (
        data.realized_weighted_cost / data.cold_realized_cost.replace(0, np.nan)
    )
    points = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["selection_score_ratio_to_cold", "realized_cost_ratio_to_cold"]
    )
    if EXPECTED_SCORE_POINTS is not None and len(points) != EXPECTED_SCORE_POINTS:
        raise ValueError(
            f"expected {EXPECTED_SCORE_POINTS} paper score-ratio points in {source}, "
            f"found {len(points)}"
        )
    with style.context():
        fig, ax = plt.subplots(figsize=(3.20, 2.28))
        ax.scatter(
            points.selection_score_ratio_to_cold,
            points.realized_cost_ratio_to_cold,
            s=17, alpha=.52,
            facecolors="none", edgecolors="#4C78A8", linewidths=.65,
        )
        lo = min(
            float(points.selection_score_ratio_to_cold.min()),
            float(points.realized_cost_ratio_to_cold.min()), 0.0,
        )
        hi = max(
            float(points.selection_score_ratio_to_cold.max()),
            float(points.realized_cost_ratio_to_cold.max()),
        )
        ax.plot([lo, hi], [lo, hi], color="#666666", ls="--", lw=1.0, label=r"$y=x$")
        ax.set_xlabel("Selection score / cold score")
        ax.set_ylabel("Realized cost / cold cost")
        ax.tick_params(axis="x", labelrotation=0)
        _expand_top(ax)
        ax.legend(loc="upper left")
        _save_pdf_only(fig, path)


def _plot_trajectory(
    summary: pd.DataFrame, sigma: float, path: Path, ylabel: str
) -> None:
    """Paper trajectory with a fixed, non-overlapping upper-left legend."""
    g = summary[np.isclose(summary.row_sigma, sigma)]
    with style.context():
        fig, ax = plt.subplots(figsize=(3.35, 2.30))
        for method in paper.RFW_METHODS:
            m = g[g.method.eq(method)].sort_values("task")
            st = style.method_style(method)
            markevery = max(1, int(np.ceil(len(m) / 7)))
            ax.plot(
                m.task, m["mean"], label=style.display_name(method),
                markevery=markevery, markerfacecolor="white",
                markeredgewidth=.75, **st,
            )
            ax.fill_between(
                m.task, m.ci_lo, m.ci_hi,
                color=st["color"], alpha=.09, linewidth=0, zorder=1,
            )
        ax.set_xlabel("Task index")
        ax.set_ylabel(ylabel)
        ticks = np.unique(np.rint(np.linspace(g.task.min(), g.task.max(), 5)).astype(int))
        ax.set_xticks(ticks)
        ax.tick_params(axis="x", labelrotation=0)
        ax.margins(x=.015)
        _scientific_if_large(ax, "y")
        _expand_top(ax, 1.20)
        ax.legend(
            ncol=1, loc="upper left", borderaxespad=.2, handlelength=1.85,
            handletextpad=.4, columnspacing=.7, labelspacing=.27,
        )
        _save_pdf_only(fig, path)


def _plot_sensitivity(
    summary: pd.DataFrame, xcol: str, metric: str, path: Path,
    xlabel: str, ylabel: str, methods: list[str], logy: bool,
) -> None:
    """Sensitivity curve with horizontal ticks and a top-corner legend."""
    with style.context():
        fig, ax = plt.subplots(figsize=(3.35, 2.35))
        for method in methods:
            g = summary[summary.method.eq(method)].sort_values(xcol)
            st = style.method_style(method)
            x = g[xcol].to_numpy(float)
            y = g[metric].to_numpy(float)
            lo = g[f"{metric}_ci_lo"].to_numpy(float)
            hi = g[f"{metric}_ci_hi"].to_numpy(float)
            ax.plot(
                x, y, label=style.display_name(method), color=st["color"],
                marker=st["marker"], linestyle=st["linestyle"],
                linewidth=st["linewidth"], markersize=st["markersize"],
                markeredgewidth=.75, markerfacecolor="white", zorder=st["zorder"],
            )
            ax.fill_between(x, lo, hi, color=st["color"], alpha=.09, linewidth=0, zorder=1)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=0)
        if xcol == "budget":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        _scientific_if_large(ax, "y")
        ax.margins(x=.015)
        _expand_top(ax, 1.18)
        ax.legend(loc="upper right")
        _save_pdf_only(fig, path)


def _restore_internal_method_names(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["method"] = out["method"].replace({
        "FFW": "FFW-2026",
        "Greedy-FP": "Greedy-FP-2019",
        "TD-RFW": "Nearest-RFW",
    })
    return out


def redraw(
    results_dir: Path,
    output_dir: Path,
    n_boot: int = 2000,
    *,
    allow_partial: bool = False,
    allow_font_fallback: bool = False,
) -> list[Path]:
    global MAIN_RUN, BUDGET_RUN, DRIFT_RUN, EXPECTED_SCORE_POINTS
    results_dir = results_dir.resolve()
    MAIN_RUN = results_dir / "synthetic"
    BUDGET_RUN = results_dir / "synthetic-budget"
    DRIFT_RUN = results_dir / "synthetic-drift"
    EXPECTED_SCORE_POINTS = None if allow_partial else 600
    _check_paper_font(allow_fallback=allow_font_fallback)
    _configure_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure imported plotting helpers emit only the paper-facing PDFs.
    paper._save = _save_pdf_only
    plot_helpers._save = _save_pdf_only

    main_source = _require_file(MAIN_RUN / "raw" / "synthetic_main.csv")
    main_raw = pd.read_csv(main_source)
    sol = paper._validate_solver_data(main_raw, main_source)

    # Figure 1: cumulative solver-oracle cost, two row-norm regimes.
    for sigma in SIGMAS:
        tag = paper._tag(sigma)
        summary = _trajectory_summary(sol, "cumulative_cost", sigma, n_boot)
        _plot_trajectory(
            summary, sigma,
            output_dir / f"solver_cost_sigma{'0' if np.isclose(sigma, 0.0) else '05'}",
            "Cumulative oracle count",
        )

    # Figure 2: requested analytical-LS wording, otherwise unchanged.
    pareto = paper.pareto_summary(sol, n_boot)
    _plot_pareto_with_requested_label(pareto, 0.0, output_dir / "pareto_quality_sigma0")
    _plot_pareto_with_requested_label(pareto, 0.5, output_dir / "pareto_quality_sigma05")

    # Figure 3: explicitly use the frozen 20260921_164309 budget-sweep results.
    budget = _restore_internal_method_names(pd.read_csv(_require_file(
        BUDGET_RUN / "tables" / "synthetic_budget_summary.csv"
    )))
    _plot_sensitivity(
        budget, "budget", "wc_mse", output_dir / "budget_sweep_wc_mse",
        "Total sampling budget", "Worst-case analytical LS MSE",
        ["VPM-RFW", "FFW-2026", "Greedy-FP-2019"], logy=True,
    )

    # Figure 4: cumulative wall-clock time, two row-norm regimes.
    for sigma in SIGMAS:
        tag = paper._tag(sigma)
        summary = _trajectory_summary(sol, "cumulative_total_runtime", sigma, n_boot)
        _plot_trajectory(
            summary, sigma,
            output_dir / f"solver_walltime_sigma{'0' if np.isclose(sigma, 0.0) else '05'}",
            "Cumulative wall-clock time (s)",
        )

    # Figure 5: reproduce the paper's 600 selected-action ratios exactly.
    _plot_score_ratio(main_raw, output_dir / "selection_score_vs_cost")

    # Figure 6: retain the two separate paper subfigures.
    disagreement, _ = paper.history_disagreement_summary(sol, n_boot)
    _plot_disagreement(
        disagreement, "Prev-RFW", output_dir / "history_effect_prev"
    )
    _plot_disagreement(
        disagreement, "Nearest-RFW", output_dir / "history_effect_td"
    )
    (output_dir / "history_disagreement_cost.pdf").unlink(missing_ok=True)

    # Figure 7: requested analytical-LS wording, otherwise unchanged.
    drift = _restore_internal_method_names(pd.read_csv(_require_file(
        DRIFT_RUN / "tables" / "synthetic_drift_summary.csv"
    )))
    methods = ["VPM-RFW", "FFW-2026", "Greedy-FP-2019"]
    _plot_sensitivity(
        drift, "drift", "wc_mse", output_dir / "drift_sweep_wc_mse",
        "Task drift", "Worst-case analytical LS MSE", methods, logy=True,
    )
    _plot_sensitivity(
        drift, "drift", "avg_mse", output_dir / "drift_mean_mse",
        "Task drift", "Mean analytical LS MSE", methods, logy=True,
    )

    outputs = sorted(output_dir.glob("*.pdf"))
    expected = 12
    if len(outputs) != expected:
        raise RuntimeError(f"expected {expected} paper figure PDFs, found {len(outputs)}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redraw only paper-used figures from frozen CSV results; never rerun experiments."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-font-fallback", action="store_true")
    args = parser.parse_args()
    output_dir = (args.output_dir or args.results_dir / "paper_figures").resolve()
    outputs = redraw(
        args.results_dir,
        output_dir,
        args.bootstrap_resamples,
        allow_partial=args.allow_partial,
        allow_font_fallback=args.allow_font_fallback,
    )
    print(f"Generated {len(outputs)} paper figure PDFs in {output_dir}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
