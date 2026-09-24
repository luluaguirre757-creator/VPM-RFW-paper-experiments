# VPM-RFW paper experiments

This repository contains only the code and KSC data needed to reproduce the experiments reported in the final paper:

- recurrent synthetic experiments under `sigma_a = 0` and `0.5`;
- total-budget and drift sweeps;
- the KSC hyperspectral experiment;
- numerical-cap sensitivity;
- certified transfer validation.

## Setup

The paper runs used **Python 3.12.10 on 64-bit Windows 11**. The dependency
versions in `requirements.txt` are exact pins from that environment. Python
3.12 is required by `pyproject.toml`.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment in the usual way for your operating system.

## Reproduce the paper experiments

Run all reported experiments:

```bash
python RUN_PAPER.py
```

This command runs all six experiment families, creates the 12 final paper PDFs
under `results/<timestamp>/paper_figures/`, and finishes with a fail-closed
verification report at `results/<timestamp>/REPRODUCTION_CHECK.json`.

Run a short code-path check (not suitable for reporting numerical results):

```bash
python RUN_PAPER.py --quick
```

The quick run verifies every code path and artifact but does not reproduce the
paper's sample sizes or reported numbers.

Run one experiment family:

```bash
python RUN_PAPER.py --component synthetic
python RUN_PAPER.py --component synthetic-budget
python RUN_PAPER.py --component synthetic-drift
python RUN_PAPER.py --component hsi
python RUN_PAPER.py --component convergence
python RUN_PAPER.py --component certified-transfer
```

The frozen paper parameters are in `configs/paper.yaml`. Results are written
below `results/` by default. Use `verify_reproduction.py --results-dir PATH`
to recheck an existing full run.

## Paper artifact mapping

The unified run creates these experiment directories:

- `synthetic/`: solver-cost, Pareto, wall-clock, score-alignment, and history-selection evidence;
- `synthetic-budget/`: total-budget sweep;
- `synthetic-drift/`: drift sweep;
- `hsi/`: KSC results and summary tables;
- `convergence/`: numerical-cap sensitivity;
- `certified-transfer/`: certified transfer validation.

The exact paper-facing figure names are generated automatically:

```text
solver_cost_sigma0.pdf             solver_cost_sigma05.pdf
pareto_quality_sigma0.pdf          pareto_quality_sigma05.pdf
budget_sweep_wc_mse.pdf            solver_walltime_sigma0.pdf
solver_walltime_sigma05.pdf        selection_score_vs_cost.pdf
history_effect_prev.pdf            history_effect_td.pdf
drift_sweep_wc_mse.pdf             drift_mean_mse.pdf
```

Exact typography requires Times New Roman. The full run fails with a clear
message if that font is unavailable; the quick run permits a fallback font.
Wall-clock measurements are inherently hardware dependent, so the verifier
checks their structure and completeness rather than requiring byte-identical
timings across machines.

Existing result artifacts, caches, virtual environments, manuscript files, and
experiments not used in the final paper are intentionally excluded.
