"""Strict theorem-certified backend for the paired bridge experiment.

This adapter only orchestrates the theorem path already used by the Table-6
analytic sanity experiment. Cross-task quantities come from the project's real
transfer/certificate implementation, never from proxy formulas.
"""
from __future__ import annotations

import hashlib
import struct
import time
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from vpmrfw.experiments.certified_synthetic import _analytic_checks, _build_task, _certified_caps
from vpmrfw.robust.certificate import build_task_certificate
from vpmrfw.robust.memory import cold_certificate, history_certificate, select_action, stationarity_screen, terminal_memory_item
from vpmrfw.robust.rfw import solve_rfw_paper
from vpmrfw.robust.trajectory import FullCoverTrajectoryGuard
from vpmrfw.robust.transfer import paper_stationarity_transfer_bound


_CFG = {
    "N": 4, "K": 1, "total_budget": 2, "lower_bound": 1,
    "mu": 0.1, "u0": 1.25, "objective_scale": 1.0e4,
    "analytic_tolerance": 1.0e-12, "hard_iteration_guard": 4,
}


def _hash_parts(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(struct.pack("<Q", len(part)))
        h.update(part)
    return h.hexdigest()


def _float_bytes(value: float) -> bytes:
    return struct.pack("<d", float(value))


def _task_at_scale(scale: float, *, label: str, task_index: int,
                   base_scale: float, direction: float):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("certified bridge objective scale must be positive")
    task = _build_task({**_CFG, "objective_scale": float(scale)})
    factors = b"".join(np.ascontiguousarray(u, dtype=np.float64).tobytes()
                       for scenario in task.scenarios for u in scenario)
    task.bridge_regime = str(label)
    task.bridge_task_index = int(task_index)
    task.bridge_base_scale = float(base_scale)
    task.bridge_direction = float(direction)
    task.base_task_hash = _hash_parts(
        label.encode(), factors, _float_bytes(base_scale),
        repr((task.Ns, task.lower_bounds, task.total_budget, task.mu,
              task.u0, task.eta_c)).encode(),
    )
    task.perturbation_direction_hash = _hash_parts(label.encode(), _float_bytes(direction))
    return task


def build_sequence(*, sequence_seed: int, n_tasks: int, drift: float,
                   schedule: Sequence[str]) -> list[Any]:
    """Build base + drift * direction tasks paired across drift levels."""
    if n_tasks != len(schedule):
        raise ValueError("n_tasks must match the explicit recurrence schedule")
    if not 0 <= drift < 1:
        raise ValueError("drift must be in [0, 1)")
    rng = np.random.default_rng(sequence_seed)
    labels = tuple(dict.fromkeys(schedule))
    prototypes = {label: 1.0e4 * float(rng.uniform(0.90, 1.10)) for label in labels}
    directions = rng.uniform(-1.0, 1.0, size=n_tasks)
    tasks = []
    for t, label in enumerate(schedule):
        base = prototypes[label]
        direction = base * float(directions[t])
        tasks.append(_task_at_scale(
            base + float(drift) * direction, label=label, task_index=t,
            base_scale=base, direction=direction,
        ))
    return tasks


def _certificate(task, epsilon: float, w_lmo: float, w_fo: float):
    cert = build_task_certificate(task, epsilon, w_lmo, w_fo, _certified_caps(_CFG))
    checks = _analytic_checks(task, cert, _CFG["analytic_tolerance"])
    if not checks["all_passed"]:
        failed = [k for k, v in checks.items()
                  if isinstance(v, (bool, np.bool_)) and not bool(v)]
        raise RuntimeError(f"certified bridge analytic checks failed: {failed}")
    return cert, checks


def _candidate(task, item, cert) -> dict[str, Any]:
    out = {
        "item": item, "value_ok": False, "localization_ok": False,
        "kkt_ok": False, "gradient_ok": False, "cost_ok": False,
        "cost": float("inf"), "meta": {}, "gamma": float("nan"),
        "screen_lhs": float("inf"),
    }
    if not item.certified or not item.theorem_certified:
        return out
    try:
        cost, meta = history_certificate(task, item, cert)
        gamma, gamma_meta = paper_stationarity_transfer_bound(task, item.task, cert, item)
    except (ValueError, FloatingPointError):
        return out
    meta = {**meta, **gamma_meta}
    value_ok = bool(np.isfinite(meta.get("V_hat", np.nan)))
    localization_ok = bool(cert.caps.theorem_certified and cert.caps.trajectory_cover_verified)
    d_z = float(meta.get("d0_hat", np.inf))
    kkt_ok = bool(localization_ok and np.isfinite(d_z))
    gradient_ok = bool(kkt_ok and np.isfinite(gamma) and gamma >= 0)
    cost_ok = bool(value_ok and gradient_ok and np.isfinite(cost) and meta.get("eligible", False))
    out.update(
        value_ok=value_ok, localization_ok=localization_ok, kkt_ok=kkt_ok,
        gradient_ok=gradient_ok, cost_ok=cost_ok,
        cost=float(cost) if cost_ok else float("inf"), meta=meta,
        gamma=float(gamma), screen_lhs=float(item.terminal_fw_gap + cert.D * gamma),
    )
    return out


def _counts(candidates):
    return {
        "n_history_candidates": len(candidates),
        "n_value_certified": sum(c["value_ok"] for c in candidates),
        "n_localization_certified": sum(c["localization_ok"] for c in candidates),
        "n_kkt_certified": sum(c["kkt_ok"] for c in candidates),
        "n_gradient_certified": sum(c["gradient_ok"] for c in candidates),
        "n_cost_certified": sum(c["cost_ok"] for c in candidates),
    }


def _screen_record(task, item, cert, gap: float, task_id: int):
    proxy = SimpleNamespace(
        x=np.asarray(item.x, float).copy(), z=np.asarray(item.z, float).copy(),
        lmo=0, fo=0, fw_gap_cert=float(gap), theorem_certified=True,
        certified_converged=True, solver_execution="certified_stationarity_transfer",
    )
    return terminal_memory_item(task_id, task, proxy, cert, certified_execution=True)


def _diagnostics(candidate, cert):
    if candidate is None:
        return {k: np.nan for k in (
            "value_cert", "kkt_distance_cert", "gradient_cert",
            "stored_fw_gap_bound", "D_times_Gamma", "stationarity_screen_lhs")}
    item, gamma, meta = candidate["item"], float(candidate["gamma"]), candidate["meta"]
    return {
        "value_cert": float(meta.get("V_hat", np.nan)),
        "kkt_distance_cert": float(meta.get("d0_hat", np.nan)),
        "gradient_cert": gamma,
        "stored_fw_gap_bound": float(item.terminal_fw_gap),
        "D_times_Gamma": float(cert.D * gamma),
        "stationarity_screen_lhs": float(candidate["screen_lhs"]),
    }


def run_method(*, method: str, task: Any, memory: list[Any], epsilon: float,
               w_lmo: float, w_fo: float) -> dict:
    if method not in {"cold", "prev", "vpm"}:
        raise ValueError(f"unknown method: {method}")
    started = time.perf_counter()
    cert, checks = _certificate(task, epsilon, w_lmo, w_fo)
    cold_cost, cold_meta = cold_certificate(cert)
    admissible = [] if method == "cold" else (memory[-1:] if method == "prev" else list(memory))
    candidates = [_candidate(task, item, cert) for item in admissible]
    counts = _counts(candidates)
    best_cost = min(candidates, key=lambda c: (c["cost"], c["item"].task_id), default=None)
    task_id = len(memory)

    screened_item, screen_meta = (stationarity_screen(task, admissible, cert)
                                  if admissible else (None, {"stationarity_screened": False}))
    if screened_item is not None:
        chosen = next(c for c in candidates if c["item"] is screened_item)
        if not (chosen["gradient_ok"] and chosen["localization_ok"] and
                chosen["screen_lhs"] <= epsilon + 1e-12):
            raise RuntimeError("stationarity screen failed independent certificate audit")
        gap = float(screen_meta["transferred_fw_gap_certified"])
        if not np.isclose(gap, chosen["screen_lhs"], rtol=0, atol=1e-12):
            raise RuntimeError("screen bound disagrees with transfer certificate")
        record = _screen_record(task, screened_item, cert, gap, task_id)
        return {
            "returned_x": record.x, "returned_z": record.z,
            "certified": True, "terminal_memory_certified": True,
            "memory_eligible": True, "stationarity_screened": True,
            "selected_action": "stationarity_screen",
            "selected_history_index": int(screened_item.task_id),
            "selected_history_certified": True,
            "selection_mode": "theorem_certified",
            "epsilon_solver": float(epsilon), "epsilon_selection": float(epsilon),
            "regularity_pass": bool(checks["all_passed"]),
            "localization_pass": bool(chosen["localization_ok"]),
            "ppa_basin_pass": bool(chosen["meta"].get("basin_ok", False)),
            "theoretical_certified_cost": 0.0, "cold_certified_cost": float(cold_cost),
            "T_hat": 0, "N_first": 0, "N_cont": 0,
            "cold_T_hat": int(cold_meta["T_hat"]),
            "cold_N_first": int(cold_meta["N_first_hat"]),
            "cold_N_cont": int(cold_meta["N_cont_hat"]),
            "realized_lmo": 0, "realized_fo": 0, "realized_cost": 0.0,
            "fw_gap_cert": gap, "distance_ratio": float(record.e / cert.eps_z),
            "selected_cold_initially": False, "fallback_after_warm_probe": False,
            "runtime_sec": time.perf_counter() - started, "_memory_record": record,
            **counts, **_diagnostics(chosen, cert),
        }

    if method == "cold":
        selected_item, selection_meta, selected_cost = None, cold_meta, cold_cost
    else:
        selected_item, selection_meta = select_action(method, task, admissible, cert)
        selected_cost = float(selection_meta["cert_cost"])
    chosen = (next((c for c in candidates if c["item"] is selected_item), None)
              if selected_item is not None else best_cost)
    if selected_item is not None and (chosen is None or not chosen["cost_ok"]):
        raise RuntimeError("warm history selected without complete theorem certificates")

    solve = solve_rfw_paper(
        task, cert, selection_meta["delta_hat"], selection_meta["d0_hat"],
        x0=None if selected_item is None else selected_item.x,
        z0=None if selected_item is None else selected_item.z,
        store_history=False, hard_iteration_guard=_CFG["hard_iteration_guard"],
        trajectory_guard=FullCoverTrajectoryGuard(
            True, provenance="Table-6 analytic stationary singleton; scale drift preserves proof"),
    )
    certified = bool(
        solve.theorem_certified and solve.certified_converged and solve.converged
        and solve.fw_gap_cert <= epsilon + 1e-12
        and solve.distance_cert_final <= cert.eps_z + 1e-12)
    if not certified:
        raise RuntimeError("bridge solver did not produce a certified terminal state")
    record = terminal_memory_item(task_id, task, solve, cert, certified_execution=True)
    memory_eligible = bool(record.certified and record.theorem_certified and
                           np.isfinite(record.terminal_fw_gap) and record.e <= cert.eps_z + 1e-12)
    selected_cold = selected_item is None
    return {
        "returned_x": solve.x, "returned_z": solve.z,
        "certified": certified, "terminal_memory_certified": bool(record.certified),
        "memory_eligible": memory_eligible, "stationarity_screened": False,
        "selected_action": "cold" if selected_cold else "history",
        "selected_history_index": np.nan if selected_item is None else int(selected_item.task_id),
        "selected_history_certified": bool(selected_item is not None and chosen["cost_ok"]),
        "selection_mode": "theorem_certified",
        "epsilon_solver": float(epsilon), "epsilon_selection": float(epsilon),
        "regularity_pass": bool(checks["all_passed"]),
        "localization_pass": bool(chosen["localization_ok"]) if chosen else True,
        "ppa_basin_pass": bool(selection_meta.get("basin_ok", False)),
        "theoretical_certified_cost": float(selected_cost),
        "cold_certified_cost": float(cold_cost),
        "T_hat": int(selection_meta["T_hat"]),
        "N_first": int(selection_meta["N_first_hat"]),
        "N_cont": int(selection_meta["N_cont_hat"]),
        "cold_T_hat": int(cold_meta["T_hat"]),
        "cold_N_first": int(cold_meta["N_first_hat"]),
        "cold_N_cont": int(cold_meta["N_cont_hat"]),
        "realized_lmo": int(solve.lmo), "realized_fo": int(solve.fo),
        "realized_cost": float(w_lmo * solve.lmo + w_fo * solve.fo),
        "fw_gap_cert": float(solve.fw_gap_cert),
        "distance_ratio": float(solve.distance_cert_ratio),
        "selected_cold_initially": selected_cold, "fallback_after_warm_probe": False,
        "runtime_sec": time.perf_counter() - started, "_memory_record": record,
        **counts, **_diagnostics(chosen, cert),
    }


def to_memory_record(task: Any, result: dict) -> Any:
    del task
    if not (result.get("terminal_memory_certified") is True and result.get("memory_eligible") is True):
        raise ValueError("uncertified or ineligible state cannot enter memory")
    record = result.get("_memory_record")
    if record is None or not record.certified or not record.theorem_certified:
        raise ValueError("missing theorem-certified memory record")
    return record
