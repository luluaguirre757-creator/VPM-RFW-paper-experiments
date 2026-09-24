from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from vpmrfw.robust.certificate import RegularityCaps


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    validate_config(cfg)
    return cfg


def caps_from_cfg(c: dict[str, Any]) -> RegularityCaps:
    return RegularityCaps(
        bar_kappa=float(c["bar_kappa"]),
        bar_kappa_ms=float(c["bar_kappa_ms"]),
        bar_kappa_cross=float(c["bar_kappa_cross"]),
        r_ppa=float(c["r_ppa"]),
        alpha_ppa=float(c.get("alpha_ppa", 1.0)),
        status=str(c.get("status", "numerical_proxy")),
        enforce_local_basin=bool(c.get("enforce_local_basin", True)),
        global_cold_error_bound_verified=bool(c.get("global_cold_error_bound_verified", False)),
        trajectory_cover_verified=bool(c.get("trajectory_cover_verified", False)),
    )


def _need(d: dict[str, Any], keys: list[str], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"Missing config keys in {where}: {missing}")


def validate_config(cfg: dict[str, Any]) -> None:
    _need(cfg, ["seed", "synthetic", "hsi", "convergence_sensitivity"], "root")
    s = cfg["synthetic"]
    _need(s, ["N", "K", "tasks", "sequences", "total_budget", "mu", "u0", "eta_c",
              "design_scales", "j_test", "test_max", "main_row_sigmas", "epsilon_abs",
              "w_lmo", "w_fo", "regularity_caps", "practical_solver"], "synthetic")
    if len(s["N"]) != len(s["K"]):
        raise ValueError("synthetic N/K length mismatch")
    Ns=list(map(int,s['N'])); Ks=list(map(int,s['K'])); L=int(s['total_budget'])
    if any(n <= k for n,k in zip(Ns,Ks)):
        raise ValueError("synthetic requires N_r > K_r")
    if L < sum(Ks) or L > sum(Ns):
        raise ValueError("synthetic total_budget must satisfy sum K_r <= L <= sum N_r")
    floors=list(map(int,s.get('operational_floors',Ks)))
    if len(floors)!=len(Ks):
        raise ValueError('synthetic operational_floors length must match K')
    if any(f<k or f>n for f,k,n in zip(floors,Ks,Ns)):
        raise ValueError('synthetic operational floors must satisfy K_r <= floor_r <= N_r')
    if sum(floors)>L:
        raise ValueError('synthetic operational floors exceed total_budget')
    margin=s.get('operational_floor_margin')
    if margin is not None and floors != [k+int(margin) for k in Ks]:
        raise ValueError('synthetic operational_floors must equal K + operational_floor_margin')
    if float(s["epsilon_abs"]) <= 0:
        raise ValueError("synthetic epsilon_abs must be positive")
    if int(s['tasks']) <= 0 or int(s['sequences']) <= 0:
        raise ValueError('synthetic tasks/sequences must be positive')
    sweep=list(map(int,s.get('total_budget_sweep',[])))
    if len(sweep) != len(set(sweep)):
        raise ValueError('synthetic total_budget_sweep values must be unique')
    for v in sweep:
        if v<sum(floors) or v>sum(Ns):
            raise ValueError(
                f"invalid synthetic total_budget_sweep value {v}; require "
                f"{sum(floors)} <= budget <= {sum(Ns)} under operational floors"
            )
    for key in ('total_budget_sequences_per_point','total_budget_sequences_per_point_quick',
                'total_budget_bootstrap_resamples','total_budget_bootstrap_resamples_quick'):
        if key in s and int(s[key]) <= 0:
            raise ValueError(f'synthetic {key} must be positive')
    seed_policy=str(s.get('total_budget_seed_policy','fixed')).lower()
    if seed_policy not in {'fixed','random'}:
        raise ValueError("synthetic total_budget_seed_policy must be 'fixed' or 'random'")
    if 'total_budget_seed_start' in s and int(s['total_budget_seed_start']) < 0:
        raise ValueError('synthetic total_budget_seed_start must be nonnegative')
    stability_threshold=float(s.get('total_budget_min_relative_singular_value',0.01))
    if not 0.0 < stability_threshold < 1.0:
        raise ValueError('synthetic total_budget_min_relative_singular_value must lie in (0,1)')
    if int(s.get('total_budget_max_seed_draws',100)) <= 0:
        raise ValueError('synthetic total_budget_max_seed_draws must be positive')
    drift_values=list(map(float,s.get('drift_sweep',[])))
    if len(drift_values) != len(set(drift_values)) or any(v<0.0 or v>0.10 for v in drift_values):
        raise ValueError('synthetic drift_sweep values must be unique and lie in [0,0.10]')
    for key in ('drift_sequences_per_point','drift_sequences_per_point_quick',
                'drift_bootstrap_resamples','drift_bootstrap_resamples_quick'):
        if key in s and int(s[key]) <= 0:
            raise ValueError(f'synthetic {key} must be positive')

    h = cfg["hsi"]
    _need(h, ["datasets", "rank", "total_budget", "warmup", "history_scenarios", "snr_db",
              "epsilon_abs", "w_lmo", "w_fo", "regularity_caps", "practical_solver"], "hsi")
    rank=int(h['rank']); Lh=int(h['total_budget']); floor=int(h.get('mode_budget_floor',rank))
    if floor < rank:
        raise ValueError('HSI mode_budget_floor must be at least rank')
    if Lh < 2*floor:
        raise ValueError('HSI total_budget must be at least 2*mode_budget_floor')
    if int(h.get('evaluation_bands',1)) <= 0:
        raise ValueError('HSI evaluation_bands must be positive')
    for _,v in (h.get('evaluation_bands_by_dataset',{}) or {}).items():
        if int(v) <= 0:
            raise ValueError('HSI evaluation_bands_by_dataset values must be positive')
    if str(h.get('band_selection','contiguous_windows')).lower() != 'contiguous_windows':
        raise ValueError("HSI band_selection must be 'contiguous_windows'")
    for _,v in (h.get('evaluation_windows_by_dataset',{}) or {}).items():
        if int(v) <= 0:
            raise ValueError('HSI evaluation_windows_by_dataset values must be positive')
    for ds,vals in (h.get('evaluation_window_fractions_by_dataset',{}) or {}).items():
        nwin=int((h.get('evaluation_windows_by_dataset',{}) or {}).get(ds,len(vals)))
        if len(vals)!=nwin or any(float(x)<0.0 or float(x)>1.0 for x in vals):
            raise ValueError('HSI evaluation window fractions must match window counts and lie in [0,1]')
    if int(h.get('quick_windows',1)) <= 0:
        raise ValueError('HSI quick_windows must be positive')
    if int(h.get('warmup',1)) < int(h.get('history_scenarios',1)):
        raise ValueError('HSI warmup must be at least history_scenarios')
    tq=float(h.get('reconstruction_trust_quantile',0.5))
    if not 0.0 <= tq <= 1.0:
        raise ValueError('HSI reconstruction_trust_quantile must lie in [0,1]')
    tmin=float(h.get('reconstruction_trust_min',1e-3)); tmax=float(h.get('reconstruction_trust_max',float('inf')))
    if tmin <= 0 or tmax < tmin:
        raise ValueError('HSI reconstruction trust bounds must satisfy 0 < min <= max')
    if int(h.get('sensitivity_evaluation_bands',1)) <= 0:
        raise ValueError('HSI sensitivity_evaluation_bands must be positive')
    # The experiment coupling uses the uniform simplex point y0=1/J and has
    # analytic RCQ margin u0-1/J.  Validate both the main and sensitivity
    # scenario counts before a long run so invalid profiles fail immediately.
    for label, jval in (('history_scenarios', h.get('history_scenarios',1)),
                        ('sensitivity_history_scenarios', h.get('sensitivity_history_scenarios',h.get('history_scenarios',1)))):
        J=int(jval)
        if J <= 0:
            raise ValueError(f'HSI {label} must be positive')
        if float(h.get('eta_c',0.0)) != 0.0 and float(h.get('u0',0.0)) - 1.0/J <= 0.0:
            raise ValueError(f'HSI {label}={J} does not provide a positive uniform RCQ margin: require u0 > 1/J')
    if int(h.get('sensitivity_warmup',h.get('warmup',1))) < int(h.get('sensitivity_history_scenarios',h.get('history_scenarios',1))):
        raise ValueError('HSI sensitivity_warmup must be at least sensitivity_history_scenarios')
    allowed_policies={'cold','prev','nearest','vpm','random','v_only','omega_only'}
    sp=list(h.get('sensitivity_policies',['vpm']))
    if not sp or any(str(p) not in allowed_policies for p in sp):
        raise ValueError('HSI sensitivity_policies contains an unsupported policy')
    for k in h.get('rank_sweep',[]):
        if int(k) > floor:
            raise ValueError(f'HSI rank_sweep value {k} exceeds mode_budget_floor {floor}; increase the floor or budget')
    for v in h.get('total_budget_sweep',[]):
        if int(v)<2*floor:
            raise ValueError(f"HSI total_budget_sweep value {v} violates the operational mode floors")
