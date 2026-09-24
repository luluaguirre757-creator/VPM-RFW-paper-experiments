"""Paired, audited theorem-certified bridge experiment."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import importlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentConfig:
    drifts: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)
    n_sequences_per_drift: int = 6
    n_tasks: int = 12
    epsilon: float = 0.02
    w_lmo: float = 1.0
    w_fo: float = 1.0
    base_seed: int = 20260923
    bootstrap_resamples: int = 2000
    schedule: tuple[str, ...] = (
        "A", "A", "B", "B", "C", "C", "A", "B", "A", "C", "B", "A")
    backend_module: str = "vpmrfw.experiments.certified_bridge_backend"


METHODS = ("cold", "prev", "vpm")
LABELS = {"cold": "Cold-RFW", "prev": "Prev-RFW", "vpm": "Certified VPM-RFW"}
REQUIRED = {
    "selection_mode", "epsilon_solver", "epsilon_selection", "certified",
    "terminal_memory_certified", "memory_eligible", "stationarity_screened",
    "selected_action", "selected_history_index", "selected_history_certified",
    "regularity_pass", "localization_pass", "ppa_basin_pass",
    "theoretical_certified_cost", "cold_certified_cost", "realized_lmo",
    "realized_fo", "realized_cost", "fw_gap_cert", "distance_ratio",
    "value_cert", "kkt_distance_cert", "gradient_cert",
    "selected_cold_initially", "fallback_after_warm_probe",
    "n_history_candidates", "n_value_certified", "n_localization_certified",
    "n_kkt_certified", "n_gradient_certified", "n_cost_certified",
    "stored_fw_gap_bound", "D_times_Gamma", "stationarity_screen_lhs",
    "T_hat", "N_first", "N_cont", "cold_T_hat", "cold_N_first", "cold_N_cont",
}


def _backend(name: str):
    module = importlib.import_module(name)
    for attr in ("build_sequence", "run_method", "to_memory_record"):
        if not callable(getattr(module, attr, None)):
            raise AttributeError(f"backend missing callable {attr}")
    return module


def _assert_result(result: Mapping[str, Any], *, epsilon: float,
                   method: str, task_index: int, screen_eligible: bool) -> None:
    missing = sorted(REQUIRED - result.keys())
    if missing:
        raise RuntimeError(f"task {task_index}, {method}: missing required fields {missing}")
    mode = str(result["selection_mode"]).lower()
    if mode != "theorem_certified" or any(x in mode for x in
            ("practical_proxy", "numerical_proxy", "state_conditioned_numerical_proxy")):
        raise RuntimeError(f"task {task_index}, {method}: invalid selection mode {mode!r}")
    for key in ("epsilon_solver", "epsilon_selection"):
        if not math.isclose(float(result[key]), epsilon, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"task {task_index}, {method}: {key} mismatch")
    if bool(result["stationarity_screened"]):
        checks = (
            screen_eligible,
            bool(result["selected_history_certified"]),
            bool(result["localization_pass"]),
            np.isfinite(float(result["stored_fw_gap_bound"])),
            np.isfinite(float(result["gradient_cert"])),
            float(result["stationarity_screen_lhs"]) <= epsilon + 1e-12,
            int(result["realized_lmo"]) == 0,
        )
        if not all(checks):
            raise RuntimeError(f"task {task_index}, {method}: invalid stationarity screen")
    if str(result["selected_action"]) == "history":
        if not bool(result["selected_history_certified"]):
            raise RuntimeError("warm history selected without full certificates")
        if not np.isfinite(float(result["theoretical_certified_cost"])):
            raise RuntimeError("warm history selected with non-finite C^epsilon")


def _run_sequence(backend, cfg: ExperimentConfig, drift: float,
                  sequence_id: int, seed: int) -> list[dict[str, Any]]:
    tasks = backend.build_sequence(
        sequence_seed=seed, n_tasks=cfg.n_tasks, drift=drift, schedule=cfg.schedule)
    if len(tasks) != cfg.n_tasks:
        raise ValueError("backend returned wrong task count")
    memories = {method: [] for method in METHODS}
    rows = []
    for task_index, source_task in enumerate(tasks):
        for method in METHODS:
            task = deepcopy(source_task)
            screen_eligible = bool(memories[method]) if method in {"prev", "vpm"} else False
            result = backend.run_method(
                method=method, task=task, memory=memories[method],
                epsilon=cfg.epsilon, w_lmo=cfg.w_lmo, w_fo=cfg.w_fo)
            _assert_result(result, epsilon=cfg.epsilon, method=method,
                           task_index=task_index, screen_eligible=screen_eligible)
            lmo, fo = int(result["realized_lmo"]), int(result["realized_fo"])
            checked_cost = cfg.w_lmo * lmo + cfg.w_fo * fo
            if abs(float(result["realized_cost"]) - checked_cost) >= 1e-12:
                raise RuntimeError(f"task {task_index}, {method}: backend cost mismatch")
            memory_stored = False
            if bool(result["terminal_memory_certified"]) and bool(result["memory_eligible"]):
                record = backend.to_memory_record(task, result)
                if not (record.certified and record.theorem_certified):
                    raise RuntimeError("backend returned uncertified memory record")
                memories[method].append(record)
                memory_stored = True
            elif result.get("_memory_record") is not None:
                raise RuntimeError("ineligible result exposed a memory record")
            row = {
                "drift": drift, "sequence_id": sequence_id, "seed": seed,
                "task_index": task_index, "regime": cfg.schedule[task_index],
                "method": method, "method_label": LABELS[method],
                "base_task_hash": task.base_task_hash,
                "perturbation_direction_hash": task.perturbation_direction_hash,
                "screen_eligible": screen_eligible,
                "memory_stored": memory_stored,
                "realized_cost": float(checked_cost),
            }
            for key, value in result.items():
                if not key.startswith("_") and key not in {"returned_x", "returned_z", "realized_cost"}:
                    row[key] = value
            rows.append(row)
    return rows


def _safe_ratio(num, den):
    return float(num / den) if den else np.nan


def _sequence_level(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in raw.groupby(["drift", "sequence_id", "method", "method_label"], sort=True):
        drift, sequence_id, method, label = keys
        hist_n = int(group.n_history_candidates.sum())
        eligible = int(group.screen_eligible.sum())
        rows.append({
            "drift": drift, "sequence_id": sequence_id, "method": method, "method_label": label,
            "total_cost": float(group.realized_cost.sum()),
            "total_lmo": int(group.realized_lmo.sum()), "total_fo": int(group.realized_fo.sum()),
            "n_history_candidates": hist_n, "n_cost_certified": int(group.n_cost_certified.sum()),
            "history_certificate_acceptance_rate": _safe_ratio(group.n_cost_certified.sum(), hist_n),
            "n_screen_eligible": eligible, "n_screened": int(group.stationarity_screened.sum()),
            "screen_rate": _safe_ratio(group.stationarity_screened.sum(), eligible),
            "warm_continuation_rate": float((group.selected_action == "history").mean()),
            "initial_cold_selection_rate": float(group.selected_cold_initially.mean()),
            "fallback_after_warm_probe_rate": float(group.fallback_after_warm_probe.mean()),
            "execution_certified_rate": float(group.certified.mean()),
            "runtime_sec": float(group.runtime_sec.sum()),
            "median_D_z": float(group.kkt_distance_cert.median()),
            "max_D_z": float(group.kkt_distance_cert.max()),
            "median_Gamma": float(group.gradient_cert.median()),
            "max_Gamma": float(group.gradient_cert.max()),
            "median_distance_ratio": float(group.distance_ratio.median()),
            "max_distance_ratio": float(group.distance_ratio.max()),
        })
    out = pd.DataFrame(rows)
    cold = out[out.method == "cold"][["drift", "sequence_id", "total_cost"]].rename(
        columns={"total_cost": "cold_total_cost"})
    out = out.merge(cold, on=["drift", "sequence_id"], validate="many_to_one")
    out["cost_ratio_to_cold"] = out.total_cost / out.cold_total_cost
    return out


def _bootstrap(values: Sequence[float], rng, n: int):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return x[0], x[0], x[0]
    means = np.asarray([np.mean(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _summary(seq: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.base_seed + 99173)
    metrics = (
        "cost_ratio_to_cold", "history_certificate_acceptance_rate", "screen_rate",
        "warm_continuation_rate", "initial_cold_selection_rate",
        "fallback_after_warm_probe_rate", "total_lmo", "total_fo", "runtime_sec")
    rows = []
    for keys, group in seq.groupby(["drift", "method", "method_label"], sort=True):
        row = {"drift": keys[0], "method": keys[1], "method_label": keys[2],
               "n_sequences": int(group.sequence_id.nunique()),
               "n_screen_eligible": int(group.n_screen_eligible.sum()),
               "n_screened": int(group.n_screened.sum())}
        for metric in metrics:
            mean, low, high = _bootstrap(group[metric], rng, cfg.bootstrap_resamples)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _diagnostics(raw: pd.DataFrame, seq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drift, group in raw[raw.method == "vpm"].groupby("drift", sort=True):
        sg = seq[(seq.method == "vpm") & (seq.drift == drift)]
        hist = int(group.n_history_candidates.sum())
        eligible = int(group.screen_eligible.sum())
        rows.append({
            "drift": drift,
            "history_certificate_acceptance_rate": _safe_ratio(group.n_cost_certified.sum(), hist),
            "screen_rate": _safe_ratio(group.stationarity_screened.sum(), eligible),
            "warm_continuation_rate": float((group.selected_action == "history").mean()),
            "initial_cold_selection_rate": float(group.selected_cold_initially.mean()),
            "fallback_after_warm_probe_rate": float(group.fallback_after_warm_probe.mean()),
            "median_D_z": float(group.kkt_distance_cert.median()),
            "max_D_z": float(group.kkt_distance_cert.max()),
            "median_Gamma": float(group.gradient_cert.median()),
            "max_Gamma": float(group.gradient_cert.max()),
            "median_distance_ratio": float(group.distance_ratio.median()),
            "max_distance_ratio": float(group.distance_ratio.max()),
            "mean_lmo": float(group.realized_lmo.mean()), "mean_fo": float(group.realized_fo.mean()),
            "realized_cost_ratio_to_cold": float(sg.cost_ratio_to_cold.mean()),
        })
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, diagnostics: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for method in ("prev", "vpm"):
        d = summary[summary.method == method].sort_values("drift")
        y = d.cost_ratio_to_cold_mean.to_numpy(float)
        ax.errorbar(d.drift, y,
                    yerr=np.vstack([y - d.cost_ratio_to_cold_ci_low,
                                    d.cost_ratio_to_cold_ci_high - y]),
                    marker="o", capsize=2.5, label=LABELS[method])
    ax.set_xlabel("Task drift")
    ax.set_ylabel("Realized solver-cost ratio to Cold-RFW")
    ax.grid(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / "certified_bridge_cost_ratio_v2.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    curves = (
        ("history_certificate_acceptance_rate", "Historical certificate acceptance"),
        ("screen_rate", "Conditional stationarity screen"),
        ("warm_continuation_rate", "Warm continuation"),
        ("initial_cold_selection_rate", "Initial cold selection"),
        ("fallback_after_warm_probe_rate", "Fallback after warm probe"),
    )
    for column, label in curves:
        ax.plot(diagnostics.drift, diagnostics[column], marker="o", label=label)
    ax.set_xlabel("Task drift")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(False)
    ax.legend(frameon=False, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "certified_bridge_mechanism_v2.pdf", bbox_inches="tight")
    plt.close(fig)


def _audit(raw: pd.DataFrame, cfg: ExperimentConfig):
    expected = len(cfg.drifts) * cfg.n_sequences_per_drift * cfg.n_tasks * len(METHODS)
    proxy = int(raw.selection_mode.str.lower().str.contains("proxy").sum())
    eps_bad = int(((raw.epsilon_solver - cfg.epsilon).abs() > 1e-12).sum() +
                  ((raw.epsilon_selection - cfg.epsilon).abs() > 1e-12).sum())
    uncert_mem = int((raw.memory_stored &
                      (~raw.terminal_memory_certified | ~raw.memory_eligible)).sum())
    screened = raw.stationarity_screened
    invalid_screen = int((screened & (
        ~raw.screen_eligible | ~raw.selected_history_certified | ~raw.localization_pass |
        (raw.stationarity_screen_lhs > cfg.epsilon + 1e-12) | (raw.realized_lmo != 0))).sum())
    expected_cost = cfg.w_lmo * raw.realized_lmo + cfg.w_fo * raw.realized_fo
    cost_bad = int(((raw.realized_cost - expected_cost).abs() >= 1e-12).sum())
    hash_counts = raw.groupby(["sequence_id", "task_index", "regime"])[
        ["base_task_hash", "perturbation_direction_hash"]].nunique()
    paired_bad = int(((hash_counts.base_task_hash != 1) |
                      (hash_counts.perturbation_direction_hash != 1)).sum())
    warm = raw.selected_action == "history"
    warm_bad = int((warm & (~raw.selected_history_certified |
                            ~np.isfinite(raw.theoretical_certified_cost))).sum())
    dz = raw.kkt_distance_cert.dropna().to_numpy(float)
    gamma = raw.gradient_cert.dropna().to_numpy(float)
    nonzero_dz = int(np.count_nonzero(np.abs(dz) > 1e-15))
    nonzero_gamma = int(np.count_nonzero(np.abs(gamma) > 1e-15))
    failures = {
        "row_count_mismatch": int(len(raw) != expected), "proxy_mode_count": proxy,
        "epsilon_mismatch_count": eps_bad, "uncertified_memory_count": uncert_mem,
        "invalid_screen_count": invalid_screen, "cost_mismatch_count": cost_bad,
        "paired_seed_hash_mismatch_count": paired_bad,
        "invalid_warm_selection_count": warm_bad,
    }
    status = "PASS" if not any(failures.values()) else "FAIL"
    stats = {
        "raw_row_count": len(raw), "expected_raw_row_count": expected, **failures,
        "number_nonzero_D_z": nonzero_dz, "number_nonzero_Gamma": nonzero_gamma,
        "min_D_z": float(np.min(dz)), "median_D_z": float(np.median(dz)),
        "max_D_z": float(np.max(dz)), "min_Gamma": float(np.min(gamma)),
        "median_Gamma": float(np.median(gamma)), "max_Gamma": float(np.max(gamma)),
        "number_warm_continuations": int(warm.sum()),
        "number_cold_fallbacks_after_warm_probes": int(raw.fallback_after_warm_probe.sum()),
        "zero_certificate_reason": (
            "D_z/Gamma are zero only for exact same-task recurrences; rows without an evaluated "
            "history candidate are NaN. Positive drift enters task.scale and is recomputed by "
            "paper transfer and stationarity-certificate routines."),
        "FINAL STATUS": status,
    }
    return status, stats


def run(cfg: ExperimentConfig, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    backend = _backend(cfg.backend_module)
    rows = []
    for drift in cfg.drifts:
        for sequence_id in range(cfg.n_sequences_per_drift):
            seed = cfg.base_seed + sequence_id
            rows.extend(_run_sequence(backend, cfg, drift, sequence_id, seed))
    raw = pd.DataFrame(rows)
    seq = _sequence_level(raw)
    summary = _summary(seq, cfg)
    diagnostics = _diagnostics(raw, seq)
    status, audit = _audit(raw, cfg)

    raw.to_csv(out / "certified_bridge_raw_v2.csv", index=False)
    seq.to_csv(out / "certified_bridge_sequence_v2.csv", index=False)
    summary.to_csv(out / "certified_bridge_summary_v2.csv", index=False)
    diagnostics.to_csv(out / "certificate_diagnostics.csv", index=False)
    with open(out / "config_v2.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2, ensure_ascii=False)
    with open(out / "audit_v2.txt", "w", encoding="utf-8") as handle:
        for key, value in audit.items():
            handle.write(f"{key}: {value}\n")
    _plot(summary, diagnostics, out)
    print(diagnostics.to_string(index=False))
    print(f"FINAL STATUS: {status}")
    print(f"Saved results to: {out}")
    return raw, seq, summary, diagnostics, audit


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/certified_bridge_v2_20260923"))
    parser.add_argument("--backend", default="vpmrfw.experiments.certified_bridge_backend")
    parser.add_argument("--sequences", type=int, default=6)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--epsilon", type=float, default=.02)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser.parse_args()


def main():
    args = _args()
    if args.tasks != 12:
        raise ValueError("the explicit recurrence schedule requires 12 tasks")
    cfg = ExperimentConfig(
        n_sequences_per_drift=args.sequences, n_tasks=args.tasks,
        epsilon=args.epsilon, bootstrap_resamples=args.bootstrap,
        base_seed=args.seed, backend_module=args.backend)
    run(cfg, args.out)


if __name__ == "__main__":
    main()
