#!/usr/bin/env python3
"""Fail-closed structural verification for a completed paper reproduction."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
KSC_SHA256 = "b1ad011cfdb65c853e4f9f6108ca4774467d87f90a5c23b74ff3a2984a3b4786"
PINNED_PACKAGES = {
    "numpy": "2.5.3",
    "scipy": "1.18.1",
    "pandas": "3.0.6",
    "matplotlib": "3.11.2",
    "PyYAML": "6.0.3",
    "scikit-image": "0.26.0",
}
RAW_OUTPUTS = {
    "synthetic/raw/synthetic_main.csv": 3300,
    "synthetic-budget/raw/synthetic_budget.csv": 2880,
    "synthetic-drift/raw/synthetic_drift.csv": 4480,
    "hsi/raw/hsi_main.csv": 252,
    "convergence/raw/solver_cap_sensitivity.csv": 72,
    "certified-transfer/certified_bridge_raw_v2.csv": 1512,
}
PAPER_FIGURES = (
    "solver_cost_sigma0.pdf",
    "solver_cost_sigma05.pdf",
    "pareto_quality_sigma0.pdf",
    "pareto_quality_sigma05.pdf",
    "budget_sweep_wc_mse.pdf",
    "solver_walltime_sigma0.pdf",
    "solver_walltime_sigma05.pdf",
    "selection_score_vs_cost.pdf",
    "history_effect_prev.pdf",
    "history_effect_td.pdf",
    "drift_sweep_wc_mse.pdf",
    "drift_mean_mse.pdf",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_config(errors: list[str]) -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "paper.yaml").read_text(encoding="utf-8"))
    checks = {
        "synthetic.tasks": (cfg["synthetic"]["tasks"], 20),
        "synthetic.sequences": (cfg["synthetic"]["sequences"], 15),
        "synthetic.main_row_sigmas": (cfg["synthetic"]["main_row_sigmas"], [0.0, 0.5]),
        "synthetic.total_budget": (cfg["synthetic"]["total_budget"], 48),
        "synthetic.total_budget_sweep": (
            cfg["synthetic"]["total_budget_sweep"], [36, 42, 48, 60, 72, 84, 96, 108, 120]
        ),
        "synthetic.drift_sweep": (
            cfg["synthetic"]["drift_sweep"], [0.0, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10]
        ),
        "hsi.datasets": (cfg["hsi"]["datasets"], ["KSC"]),
        "hsi.rank": (cfg["hsi"]["rank"], 20),
        "hsi.total_budget": (cfg["hsi"]["total_budget"], 80),
        "hsi.evaluation_bands": (cfg["hsi"]["evaluation_bands"], 36),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"{name}: expected {expected!r}, found {actual!r}")


def verify(results_dir: Path, *, quick: bool) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    _check_config(errors)

    data_path = ROOT / "data" / "KSC.mat"
    if not data_path.is_file() or _sha256(data_path) != KSC_SHA256:
        errors.append("KSC.mat is missing or its SHA-256 does not match the paper data")

    expected_python = (3, 12, 10)
    actual_python = sys.version_info[:3]
    if actual_python != expected_python:
        message = f"paper Python was {expected_python}; current Python is {actual_python}"
        (warnings if quick else errors).append(message)
    for package, expected in PINNED_PACKAGES.items():
        actual = version(package)
        if actual != expected:
            message = f"{package}: expected {expected}, found {actual}"
            (warnings if quick else errors).append(message)

    checked_rows: dict[str, int] = {}
    for relative, expected_rows in RAW_OUTPUTS.items():
        path = results_dir / relative
        if not path.is_file():
            errors.append(f"missing output: {relative}")
            continue
        frame = pd.read_csv(path)
        checked_rows[relative] = len(frame)
        if frame.empty:
            errors.append(f"empty output: {relative}")
        if not quick and len(frame) != expected_rows:
            errors.append(f"{relative}: expected {expected_rows} rows, found {len(frame)}")

    figure_dir = results_dir / "paper_figures"
    missing_figures = [name for name in PAPER_FIGURES if not (figure_dir / name).is_file()]
    if missing_figures:
        errors.append(f"missing paper figures: {missing_figures}")

    bridge_summary = results_dir / "certified-transfer" / "certified_bridge_summary_v2.csv"
    if bridge_summary.is_file():
        bridge = pd.read_csv(bridge_summary)
        expected_drifts = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]
        observed_drifts = sorted(bridge["drift"].round(7).unique().tolist())
        if observed_drifts != expected_drifts:
            errors.append("certified-transfer drift grid differs from the paper")
        if not quick:
            vpm = bridge[bridge["method"].eq("vpm")]
            prev_nonzero = bridge[bridge["method"].eq("prev") & bridge["drift"].gt(0)]
            if not (
                vpm["screen_rate_mean"].round(12).eq(1.0).all()
                and vpm["total_lmo_mean"].round(12).eq(1.0).all()
                and vpm["cost_ratio_to_cold_mean"].round(12).eq(round(1 / 12, 12)).all()
            ):
                errors.append("certified VPM summary does not match the published validation table")
            if not (
                prev_nonzero["screen_rate_mean"].round(12).eq(round(56 / 66, 12)).all()
                and prev_nonzero["total_lmo_mean"].round(12).eq(round(8 / 3, 12)).all()
                and prev_nonzero["cost_ratio_to_cold_mean"].round(12).eq(round(2 / 9, 12)).all()
            ):
                errors.append("Prev-RFW summary does not match the published validation table")

    if not quick:
        hsi_summary = results_dir / "hsi" / "tables" / "hsi_main_comparison.csv"
        if not hsi_summary.is_file():
            errors.append("missing KSC comparison table")
        else:
            hsi = pd.read_csv(hsi_summary).set_index("method")
            published = {
                "Greedy-FP": (0.643, 25.523, 0.689, None),
                "FFW": (0.750, 25.415, 0.681, None),
                "Cold-RFW": (1.059, 25.107, 0.668, 56.0),
                "VPM-RFW": (0.636, 25.530, 0.677, 42.6),
            }
            for method, expected in published.items():
                if method not in hsi.index:
                    errors.append(f"KSC table is missing {method}")
                    continue
                row = hsi.loc[method]
                actual = (
                    round(float(row["nmse_db_mean"]), 3),
                    round(float(row["psnr_mean"]), 3),
                    round(float(row["ssim_mean"]), 3),
                    None if expected[3] is None else round(float(row["realized_weighted_cost_mean"]), 1),
                )
                if actual != expected:
                    errors.append(f"KSC published row mismatch for {method}: {actual} != {expected}")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "quick": quick,
        "results_dir": str(results_dir.resolve()),
        "python": sys.version,
        "package_versions": {name: version(name) for name in PINNED_PACKAGES},
        "ksc_sha256": _sha256(data_path) if data_path.is_file() else None,
        "checked_rows": checked_rows,
        "paper_figures": list(PAPER_FIGURES),
        "warnings": warnings,
        "errors": errors,
        "scope": (
            "Structural, environment, data, row-count, and artifact verification. "
            "Wall-clock values remain hardware dependent."
        ),
    }
    (results_dir / "REPRODUCTION_CHECK.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    report = verify(args.results_dir, quick=args.quick)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
