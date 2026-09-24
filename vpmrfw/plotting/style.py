from __future__ import annotations
import matplotlib as mpl

# Compact single-column conference style (ICLR/ICML/NeurIPS-like): restrained,
# color-blind friendly, no decorative grid, vector-font friendly PDF output.
ICLR_RC = {
    'figure.dpi': 160,
    'savefig.dpi': 320,
    'font.size': 8.2,
    'axes.labelsize': 8.3,
    'axes.titlesize': 8.6,
    'xtick.labelsize': 7.2,
    'ytick.labelsize': 7.2,
    'legend.fontsize': 7.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.72,
    'axes.grid': False,
    'axes.axisbelow': True,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.68,
    'ytick.major.width': 0.68,
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'lines.linewidth': 1.45,
    'lines.markersize': 4.3,
    'legend.frameon': False,
    'legend.handlelength': 2.2,
    'legend.handletextpad': 0.45,
    'legend.columnspacing': 0.9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'sans-serif',
    'mathtext.fontset': 'dejavusans',
    'figure.constrained_layout.use': True,
}

# Internal method names are left untouched for provenance/reproducibility.
# Only human-facing figures/tables use these compact labels.
METHOD_LABELS = {
    'Greedy-FP-2019': 'Greedy-FP',
    'Greedy-FP-2019-original': 'Greedy-FP',
    'FFW-2026': 'FFW',
    'UB-FMBS-Wang2022-original': 'UB-FMBS',
    'Nominal-RFW': 'Nominal-RFW',
    'Cold-RFW': 'Cold-RFW',
    'Prev-RFW': 'Prev-RFW',
    'Nearest-RFW': 'Nearest-RFW',
    'VPM-RFW': 'VPM-RFW',
}

COMPACT_METHOD_LABELS = {
    'Greedy-FP-2019': 'Greedy-\nFP',
    'Greedy-FP-2019-original': 'Greedy-\nFP',
    'FFW-2026': 'FFW',
    'UB-FMBS-Wang2022-original': 'UB-\nFMBS',
    'Nominal-RFW': 'Nominal-\nRFW',
    'Cold-RFW': 'Cold-\nRFW',
    'Prev-RFW': 'Prev-\nRFW',
    'Nearest-RFW': 'Nearest-\nRFW',
    'VPM-RFW': 'VPM-\nRFW',
}

# Stable styling by scientific role.  Never style by observed rank/performance.
METHOD_STYLES = {
    'Greedy-FP-2019': dict(color='#6E6E6E', marker='s', linestyle='--', linewidth=1.45, markersize=4.3),
    'FFW-2026': dict(color='#4C78A8', marker='^', linestyle='-.', linewidth=1.45, markersize=4.3),
    'Nominal-RFW': dict(color='#A0A0A0', marker='P', linestyle=':', linewidth=1.35, markersize=4.1),
    'Cold-RFW': dict(color='#7F7F7F', marker='o', linestyle='--', linewidth=1.45, markersize=4.3),
    'Prev-RFW': dict(color='#59A14F', marker='s', linestyle='-.', linewidth=1.45, markersize=4.3),
    'Nearest-RFW': dict(color='#B279A2', marker='D', linestyle=':', linewidth=1.45, markersize=4.3),
    'VPM-RFW': dict(color='#E45756', marker='*', linestyle='-', linewidth=1.45, markersize=5.0),
}

# Main-paper views are fixed by experimental role, not by who wins a run.
MAIN_SOLVER_METHODS = ['Cold-RFW', 'Prev-RFW', 'VPM-RFW']
MAIN_TASK_QUALITY_METHODS = [
    'Greedy-FP-2019', 'FFW-2026', 'Cold-RFW', 'Prev-RFW', 'VPM-RFW',
]
ALL_SOLVER_METHODS = ['Cold-RFW', 'Prev-RFW', 'Nearest-RFW', 'VPM-RFW']
MAIN_SAMPLING_METHODS = ['Greedy-FP-2019', 'FFW-2026', 'VPM-RFW']
ALL_SAMPLING_METHODS = [
    'Greedy-FP-2019', 'FFW-2026', 'Nominal-RFW',
    'Cold-RFW', 'Prev-RFW', 'Nearest-RFW', 'VPM-RFW',
]


def context():
    return mpl.rc_context(ICLR_RC)


def display_name(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method))


def compact_display_name(method: str) -> str:
    return COMPACT_METHOD_LABELS.get(str(method), display_name(method))


def method_style(method: str) -> dict:
    base = METHOD_STYLES.get(str(method), dict(
        color='#666666', marker='o', linestyle='-', linewidth=1.45, markersize=4.3
    ))
    out = dict(base)
    out['zorder'] = 3
    return out
