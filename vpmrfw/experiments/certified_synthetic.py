from __future__ import annotations

"""Analytically certified tiny synthetic benchmark for the theorem path.

This benchmark is intentionally separate from the large practical experiments.  It is
constructed so the required stationarity statements can be checked analytically rather
than inferred from floating-point condition numbers:

* one mode and one robust scenario (the y-simplex is the singleton {1});
* eta_c=0 and u0>1, so the cold dual point lambda=0 is strictly feasible and the
  projected natural residual is exactly zero for every outer x;
* all factor rows are identical and the balanced cold point has a constant gradient,
  hence its Frank-Wolfe gap is exactly zero because every feasible point has the same
  total mass.

The first task therefore executes the theorem-certified solver and terminates in one
outer iteration with zero inner work.  Repeated identical tasks are then certified by
the VPM stationarity-transfer screen (all transfer discrepancies are exactly zero).
"""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vpmrfw.robust.problem import RobustTask
from vpmrfw.robust.certificate import RegularityCaps, build_task_certificate
from vpmrfw.robust.ppa import natural_residual
from vpmrfw.robust.rfw import lmo_total_budget_task, solve_rfw_paper
from vpmrfw.robust.trajectory import FullCoverTrajectoryGuard
from vpmrfw.robust.memory import terminal_memory_item, stationarity_screen


def _build_task(cfg: dict) -> RobustTask:
    n = int(cfg.get("N", 4))
    k = int(cfg.get("K", 1))
    if k != 1:
        raise ValueError("certified synthetic construction currently requires K=1")
    if n < 2:
        raise ValueError("certified synthetic construction requires N>=2")
    total_budget = int(cfg.get("total_budget", 2))
    lower = int(cfg.get("lower_bound", 1))
    if not (1 <= lower <= total_budget <= n):
        raise ValueError("invalid certified synthetic budget geometry")
    # Identical positive rows give a constant gradient at the balanced point.
    U = np.ones((n, 1), dtype=float)
    return RobustTask(
        [[U]],
        total_budget=total_budget,
        lower_bounds=[lower],
        mu=float(cfg.get("mu", 0.1)),
        u0=float(cfg.get("u0", 1.25)),
        eta_c=0.0,
        scale=float(cfg.get("objective_scale", 1.0e4)),
    )


def _certified_caps(cfg: dict) -> RegularityCaps:
    # The trajectory is analytically a singleton: the certified first iterate has
    # exact zero inner residual and exact zero FW gap, hence neither a PPA move nor
    # an outer segment is taken.  The unit caps below only cover that visited
    # singleton and are not estimated from numerical conditioning.
    return RegularityCaps(
        bar_kappa=float(cfg.get("bar_kappa", 1.0)),
        bar_kappa_ms=float(cfg.get("bar_kappa_ms", 1.0)),
        bar_kappa_cross=float(cfg.get("bar_kappa_cross", 1.0)),
        r_ppa=float(cfg.get("r_ppa", 1.0)),
        alpha_ppa=float(cfg.get("alpha_ppa", 1.0)),
        status="theorem_certified",
        enforce_local_basin=False,
        global_cold_error_bound_verified=False,
        trajectory_cover_verified=True,
    )


def _analytic_checks(task: RobustTask, cert, tol: float) -> dict:
    x0 = task.uniform_x()
    z0 = task.cold_z()
    g = task.robust_grad(x0, z0)
    v = lmo_total_budget_task(task, g)
    fw_gap = float((x0 - v) @ g)
    nr = float(natural_residual(task, x0, z0))
    G0 = task.G(x0, task.y0)
    cold_cost, cold_meta = cert.cold()
    grad_constant_error = float(np.max(np.abs(g - np.mean(g))))
    checks = {
        "single_scenario": bool(task.J == 1),
        "single_mode": bool(task.R == 1),
        "zero_coupling_matrix": bool(np.max(np.abs(task.B)) <= tol),
        "strictly_inactive_coupling": bool(np.max(G0) < -tol),
        "cold_natural_residual_zero": bool(nr <= tol),
        "balanced_gradient_constant": bool(grad_constant_error <= tol),
        "balanced_fw_gap_zero": bool(abs(fw_gap) <= tol),
        "configured_one_outer_step": bool(int(cold_meta["T_hat"]) == 1),
        "configured_zero_first_inner_steps": bool(int(cold_meta["N_first_hat"]) == 0),
        "certificate_caps_theorem_status": bool(cert.caps.theorem_certified),
    }
    checks["all_passed"] = bool(all(checks.values()))
    return {
        **checks,
        "natural_residual": nr,
        "fw_gap_direct": fw_gap,
        "gradient_constant_error": grad_constant_error,
        "cold_certified_cost": float(cold_cost),
        "cold_T_hat": int(cold_meta["T_hat"]),
        "cold_N_first_hat": int(cold_meta["N_first_hat"]),
        "epsilon_z": float(cert.eps_z),
        "epsilon": float(cert.epsilon),
    }


def run(cfg: dict, quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ccfg = cfg.get("certified_synthetic", {}) or {}
    task = _build_task(ccfg)
    caps = _certified_caps(ccfg)
    epsilon = float(ccfg.get("epsilon_abs", 0.005))
    cert = build_task_certificate(task, epsilon, 1.0, 1.0, caps)
    tol = float(ccfg.get("analytic_tolerance", 1.0e-12))
    checks = _analytic_checks(task, cert, tol)
    if not checks["all_passed"]:
        failed = [k for k, v in checks.items() if isinstance(v, (bool, np.bool_)) and not bool(v)]
        raise RuntimeError(f"analytic certified-synthetic construction failed checks: {failed}")

    ntasks = int(ccfg.get("quick_tasks", 2) if quick else ccfg.get("tasks", 4))
    if ntasks < 1:
        raise ValueError("certified_synthetic tasks must be positive")
    cold_cost, cold_meta = cert.cold()
    guard = FullCoverTrajectoryGuard(
        True,
        provenance="analytic-by-construction stationary singleton trajectory; no outer segment is traversed",
    )

    rows: list[dict] = []
    # First task: execute the actual theorem-certified RFW path.
    res = solve_rfw_paper(
        task,
        cert,
        cold_meta["delta_hat"],
        cold_meta["d0_hat"],
        x0=None,
        z0=None,
        store_history=True,
        hard_iteration_guard=int(ccfg.get("hard_iteration_guard", 4)),
        trajectory_guard=guard,
    )
    first_ok = bool(
        res.theorem_certified
        and res.certified_converged
        and res.converged
        and res.fw_gap_cert <= epsilon + tol
        and res.distance_cert_final <= cert.eps_z + tol
    )
    if not first_ok:
        raise RuntimeError("constructed certified synthetic first task did not certify")
    memory_item = terminal_memory_item(0, task, res, cert, certified_execution=True)
    rows.append({
        "task": 0,
        "method": "VPM-RFW",
        "certification_mode": "full_theorem_solve",
        "certified_task_success": 1,
        "theorem_certified": int(res.theorem_certified),
        "inner_certified_converged": int(res.certified_converged),
        "outer_certified_converged": int(res.converged),
        "overall_certified_converged": 1,
        "stationarity_screened": 0,
        "memory_size": 0,
        "selected_history": -1,
        "fw_gap_certified": float(res.fw_gap_cert),
        "distance_cert_final": float(res.distance_cert_final),
        "epsilon_z_target": float(cert.eps_z),
        "num_outer": int(res.outer),
        "num_ppa": int(res.ppa),
        "num_fo": int(res.fo),
        "num_lmo": int(res.lmo),
        "configured_T": int(res.configured_T),
        "cold_certified_cost": float(cold_cost),
    })

    # Subsequent tasks are exactly identical; Theorem-2 transfer discrepancies are
    # zero and the already-certified terminal point must pass the stationarity screen.
    for s in range(1, ntasks):
        item, meta = stationarity_screen(task, [memory_item], cert)
        screened = bool(item is not None and meta.get("stationarity_screened", False))
        transferred_gap = float(meta.get("transferred_fw_gap_certified", np.inf))
        ok = bool(screened and transferred_gap <= epsilon + tol)
        if not ok:
            raise RuntimeError(f"constructed certified synthetic transfer screen failed at task {s}: {meta}")
        rows.append({
            "task": s,
            "method": "VPM-RFW",
            "certification_mode": "stationarity_transfer_screen",
            "certified_task_success": 1,
            "theorem_certified": 1,
            "inner_certified_converged": 1,  # reused terminal state was already certified
            "outer_certified_converged": 1,
            "overall_certified_converged": 1,
            "stationarity_screened": 1,
            "memory_size": 1,
            "selected_history": int(item.task_id),
            "fw_gap_certified": transferred_gap,
            "distance_cert_final": float(memory_item.e),
            "epsilon_z_target": float(cert.eps_z),
            "num_outer": 0,
            "num_ppa": 0,
            "num_fo": 0,
            "num_lmo": 0,
            "configured_T": 0,
            "cold_certified_cost": float(cold_cost),
        })

    raw = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "tasks": int(len(raw)),
        "certified_task_success_rate": float(raw.certified_task_success.mean()),
        "overall_certified_convergence_rate": float(raw.overall_certified_converged.mean()),
        "stationarity_screen_rate_after_first": float(raw.loc[raw.task > 0, "stationarity_screened"].mean()) if len(raw) > 1 else np.nan,
        "total_lmo": int(raw.num_lmo.sum()),
        "total_fo": int(raw.num_fo.sum()),
        "max_fw_gap_certified": float(raw.fw_gap_certified.max()),
        "max_distance_cert_ratio": float((raw.distance_cert_final / raw.epsilon_z_target).max()),
        "certificate_scope": "analytic_by_construction_stationary_instance",
    }])
    proof = {
        "construction": {
            "N": task.Ns,
            "K": task.Ks,
            "total_budget": task.total_budget,
            "lower_bounds": task.lower_bounds,
            "J": task.J,
            "R": task.R,
            "mu": task.mu,
            "u0": task.u0,
            "eta_c": task.eta_c,
            "scale": task.scale,
        },
        "caps": asdict(caps),
        "analytic_checks": checks,
        "proof_note": (
            "For J=1 the y-simplex is {1}. With eta_c=0 and u0>1, lambda=0 at the cold point has exact projected natural residual 0. "
            "With one mode, identical rows, and fixed total mass, the balanced point has a constant gradient, so its FW gap is exactly 0. "
            "The certified first task therefore terminates before any PPA or outer trajectory move; repeated identical tasks have zero transfer discrepancies and pass the stationarity screen."
        ),
    }
    return raw, summary, proof
