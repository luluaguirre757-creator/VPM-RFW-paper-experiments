from __future__ import annotations
import copy
import json
import numpy as np
import pandas as pd

from vpmrfw.experiments import synthetic_main, real_hsi


def _scale_solver_caps(scfg, multiplier: int):
    c = copy.deepcopy(scfg)
    pcfg = c['practical_solver']
    m = int(multiplier)
    for key in ('max_outer', 'max_ppa', 'max_resolvent', 'quick_max_outer', 'quick_max_ppa', 'quick_max_resolvent'):
        if key in pcfg:
            pcfg[key] = max(1, int(pcfg[key]) * m)
    return c


def run_solver_cap_sensitivity(cfg, quick=False):
    """Check whether the reported numerical conclusions are stable to solver caps.

    This is deliberately a small diagnostic.  It does not try to turn practical
    execution into theorem certification; it tests whether increasing numerical
    budgets changes selected designs, errors, residuals, or method ordering.
    """
    base = copy.deepcopy(cfg['synthetic'])
    ccfg = cfg.get('convergence_sensitivity', {}) or {}
    multipliers = list(map(int, ccfg.get('cap_multipliers', [1, 2, 4])))
    if quick:
        multipliers = multipliers[:2]
    tasks = int(ccfg.get('quick_tasks', 2) if quick else ccfg.get('tasks', 6))
    seeds_n = int(ccfg.get('quick_sequences', 1) if quick else ccfg.get('sequences', 2))
    row_sigma = float(ccfg.get('row_sigma', 0.5))
    sequence = str(ccfg.get('sequence', 'recurrent'))

    frames = []
    for mult in multipliers:
        scfg = _scale_solver_caps(base, mult)
        scfg['tasks'] = tasks
        scfg['quick_tasks'] = tasks
        scfg['sampling_stride'] = max(tasks + 1, 9999)
        scfg['runtime_tasks'] = []
        scfg['quick_runtime_tasks'] = []
        scfg['empirical_trials'] = 0
        scfg['quick_empirical_trials'] = 0
        for j in range(seeds_n):
            seed = int(cfg['seed']) + j
            d = synthetic_main.run_sequence(scfg, seed, row_sigma, sequence, quick=quick, policies=['cold', 'vpm'])
            d = d[d.kind == 'solver'].copy()
            d['cap_multiplier'] = int(mult)
            d['configured_max_outer'] = int(scfg['practical_solver']['quick_max_outer'] if quick else scfg['practical_solver']['max_outer'])
            d['configured_max_ppa'] = int(scfg['practical_solver']['quick_max_ppa'] if quick else scfg['practical_solver']['max_ppa'])
            d['configured_max_resolvent'] = int(scfg['practical_solver']['quick_max_resolvent'] if quick else scfg['practical_solver']['max_resolvent'])
            frames.append(d)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        return raw, pd.DataFrame()

    # Exact set stability relative to the smallest-cap run for each paired task.
    ref_mult = min(multipliers)
    ref = raw[raw.cap_multiplier == ref_mult][['seed', 'task', 'method', 'sets', 'wc_mse', 'robust_design_objective']].rename(
        columns={'sets': 'sets_ref', 'wc_mse': 'wc_mse_ref', 'robust_design_objective': 'robust_design_objective_ref'}
    )
    raw = raw.merge(ref, on=['seed', 'task', 'method'], how='left', validate='many_to_one')
    raw['same_rounded_design_as_base_cap'] = (raw['sets'] == raw['sets_ref']).astype(int)
    raw['wc_mse_ratio_to_base_cap'] = raw['wc_mse'] / raw['wc_mse_ref'].replace(0, np.nan)
    raw['robust_objective_ratio_to_base_cap'] = raw['robust_design_objective'] / raw['robust_design_objective_ref'].replace(0, np.nan)

    summary = (raw.groupby(['cap_multiplier', 'method'], as_index=False)
               .agg(n=('task', 'size'),
                    practical_convergence_rate=('practical_converged', 'mean'),
                    inner_certificate_rate=('inner_certified_converged', 'mean'),
                    overall_certificate_rate=('overall_certified_converged', 'mean'),
                    outer_cap_rate=('hit_outer_cap', 'mean'),
                    ppa_cap_rate=('hit_ppa_cap', 'mean'),
                    resolvent_cap_rate=('hit_resolvent_cap', 'mean'),
                    median_distance_cert_ratio=('distance_cert_ratio', 'median'),
                    mean_wc_mse=('wc_mse', 'mean'),
                    mean_robust_design_objective=('robust_design_objective', 'mean'),
                    design_stability_rate=('same_rounded_design_as_base_cap', 'mean'),
                    mean_realized_cost=('realized_weighted_cost', 'mean'),
                    mean_end_to_end_runtime=('solver_end_to_end_runtime', 'mean')))
    return raw, summary


def run_robustness_sweep(cfg, quick=False):
    """Clean-to-uncertain sweep aligned with the robust objective in the paper.

    The same multiplier scales both design-scenario perturbations and held-out
    perturbations.  Baselines are unchanged and receive the same total budget.
    This exposes the price of robustness at multiplier 0 and robustness/tail
    behavior as uncertainty grows without tuning any method per setting.
    """
    base = copy.deepcopy(cfg['synthetic'])
    rcfg = cfg.get('robustness_sweep', {}) or {}
    multipliers = list(map(float, rcfg.get('uncertainty_multipliers', [0.0, 0.5, 1.0, 1.5, 2.0])))
    if quick and len(multipliers) > 2:
        multipliers = [multipliers[0], multipliers[-1]]
    tasks = int(rcfg.get('quick_tasks', 2) if quick else rcfg.get('tasks', 8))
    nseq = int(rcfg.get('quick_sequences', 1) if quick else rcfg.get('sequences', 3))
    row_sigma = float(rcfg.get('row_sigma', 0.5))
    sequence = str(rcfg.get('sequence', 'recurrent'))
    base_scales = np.asarray(base['design_scales'], float)
    base_test = float(base['test_max'])

    frames = []
    for mult in multipliers:
        scfg = copy.deepcopy(base)
        scfg['tasks'] = tasks; scfg['quick_tasks'] = tasks
        scfg['sampling_stride'] = 1
        scfg['design_scales'] = (base_scales * mult).tolist()
        scfg['test_max'] = base_test * mult
        scfg['runtime_tasks'] = []; scfg['quick_runtime_tasks'] = []
        scfg['empirical_trials'] = 0; scfg['quick_empirical_trials'] = 0
        for j in range(nseq):
            seed = int(cfg['seed']) + j
            d = synthetic_main.run_sequence(scfg, seed, row_sigma, sequence, quick=quick, policies=['vpm'])
            keep = d.method.isin(['Greedy-FP-2019', 'FFW-2026', 'Nominal-RFW', 'VPM-RFW'])
            d = d[keep].copy()
            d['uncertainty_multiplier'] = float(mult)
            d['design_test_max'] = float(scfg['test_max'])
            frames.append(d)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        return raw, pd.DataFrame()

    # Direct task-level paired comparison across methods.  All methods use the
    # same generated task and held-out perturbations at a given seed/task/multiplier.
    summary = (raw.groupby(['uncertainty_multiplier', 'method'], as_index=False)
               .agg(n=('task', 'size'),
                    mean_wc_mse=('wc_mse', 'mean'),
                    median_wc_mse=('wc_mse', 'median'),
                    mean_p95_mse=('p95_mse', 'mean'),
                    mean_avg_mse=('avg_mse', 'mean'),
                    mean_robust_design_objective=('robust_design_objective', 'mean'),
                    singular_rate=('singular', 'mean'),
                    mean_cartesian_observations=('cartesian_observations', 'mean')))
    clean = summary[np.isclose(summary.uncertainty_multiplier.astype(float), 0.0)][['method', 'mean_wc_mse', 'mean_p95_mse']].rename(
        columns={'mean_wc_mse': 'clean_wc_mse', 'mean_p95_mse': 'clean_p95_mse'}
    )
    summary = summary.merge(clean, on='method', how='left')
    summary['wc_mse_degradation_vs_clean'] = summary['mean_wc_mse'] / summary['clean_wc_mse'].replace(0, np.nan)
    summary['p95_mse_degradation_vs_clean'] = summary['mean_p95_mse'] / summary['clean_p95_mse'].replace(0, np.nan)
    return raw, summary


def run_hsi_independent_closed_loop(cfg, data_dir, quick=False):
    """Small method-specific closed-loop sanity check on KSC.

    The main HSI experiment intentionally uses a shared VPM-driven history to
    isolate the current sampling design.  Here each reported method drives its
    *own* future low-rank history.  We obtain this without duplicating solver code
    by rerunning the common evaluation with a different ``history_driver_method``
    and retaining only the driver's own trajectory.
    """
    base = copy.deepcopy(cfg['hsi'])
    icfg = cfg.get('independent_closed_loop', {}) or {}
    dataset = str(icfg.get('dataset', 'KSC'))
    drivers = list(icfg.get('methods', ['Greedy-FP-2019', 'FFW-2026', 'Nominal-RFW', 'VPM-RFW']))
    bands = int(icfg.get('quick_bands', 2) if quick else icfg.get('bands', 8))
    windows = int(icfg.get('quick_windows', 1) if quick else icfg.get('windows', 2))
    base['evaluation_bands'] = bands
    base['evaluation_bands_by_dataset'] = {dataset: bands}
    base['evaluation_windows_by_dataset'] = {dataset: windows}
    if quick:
        base['quick_bands'] = bands
        base['quick_windows'] = windows
    # VPM must be available when it is the driver; other solver policies are not
    # needed for this sanity check and would only multiply runtime.
    base['policies'] = ['vpm']

    frames = []
    for driver in drivers:
        r, _, _ = real_hsi.run_dataset(
            dataset, base, data_dir, quick=quick, history_mode='closed_loop',
            history_driver_method=driver, evaluation_methods=[driver]
        )
        own = r[r.method == driver].copy()
        own['trajectory_method'] = driver
        own['comparison_mode'] = 'independent_closed_loop'
        frames.append(own)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        return raw, pd.DataFrame()
    summary = (raw.groupby(['trajectory_method'], as_index=False)
               .agg(n=('band', 'size'), mean_nmse_db=('nmse_db', 'mean'), mean_psnr=('psnr', 'mean'),
                    mean_ssim=('ssim', 'mean'), mean_post_design_mse=('post_design_mse', 'mean'),
                    mean_robust_design_objective=('robust_design_objective', 'mean'),
                    mean_end_to_end_runtime=('end_to_end_runtime', 'mean')))
    return raw, summary
