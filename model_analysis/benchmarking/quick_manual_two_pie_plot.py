"""
Quick manual plot generator for the same 2-pie benchmark visualization used in full_benchmark.

Edit values in the USER CONFIG section, then run:
    python model_analysis/benchmarking/quick_manual_two_pie_plot.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Thesis Plot Style Configuration (same as full_benchmark)
# =============================================================================
THESIS_TEXTWIDTH_INCHES = 14 / 2.54

plt.rcParams.update({
    'figure.figsize': (THESIS_TEXTWIDTH_INCHES, 3.5),
    'figure.constrained_layout.use': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'text.usetex': False,
})

ITERATION_COLORS = ['#E07A5F', '#3D405B', '#81B29A', '#F2CC8F']
STAGE_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#95190C', '#610345', '#044B7F']


# =============================================================================
# USER CONFIG (edit these numbers)
# =============================================================================
ITER_TIMES = {
    'load_mean': 1565,
    'transfer_mean': 62,
    'forward_mean': 88,
    'backward_mean': 44,
    # If None, this will be computed as sum of the 4 means above.
    'iteration_mean': None,
}

STAGE_TIMES = {
    'Backbone': 30,
    'FPS': 5,
    'Labels': 31,
    'Crop': 9,
    'Other': 8
}

# If None, total is sum(STAGE_TIMES.values()).
# Set this bigger than sum(STAGE_TIMES) to visualize "unaccounted" slice.
FORWARD_TOTAL_MS = None

OUTPUT_PATH = Path(__file__).parent / 'plots' / 'manual_profiling_breakdown.png'


def group_small_items(names, values, total, threshold_pct=1.0):
    filtered_names = []
    filtered_values = []
    other_total = 0

    for name, val in zip(names, values):
        pct = (val / total) * 100 if total > 0 else 0
        if pct >= threshold_pct:
            filtered_names.append(name)
            filtered_values.append(val)
        else:
            other_total += val

    if other_total > 0:
        filtered_names.append('other')
        filtered_values.append(other_total)

    return filtered_names, filtered_values


def make_autopct_ms(values, threshold_pct=0):
    def autopct(pct):
        if pct < threshold_pct:
            return ''
        total = sum(values)
        val = pct * total / 100.0
        return f'{val:.0f}ms'

    return autopct


def add_outside_labels(ax, wedges, values, label_distance=1.15, x_offsets=None, y_offsets=None):
    """Add ms labels outside the pie chart, next to each wedge.
    
    x_offsets: dict mapping index -> x offset to apply (e.g., {0: 0.1} to move first label right)
    y_offsets: dict mapping index -> y offset to apply (e.g., {0: 0.1} to move first label up)
    """
    if x_offsets is None:
        x_offsets = {}
    if y_offsets is None:
        y_offsets = {}
    for i, (wedge, val) in enumerate(zip(wedges, values)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        angle_rad = np.deg2rad(angle)
        x = label_distance * np.cos(angle_rad) + x_offsets.get(i, 0)
        y = label_distance * np.sin(angle_rad) + y_offsets.get(i, 0)
        ha = 'left' if x >= 0 else 'right'
        ax.annotate(
            f'{val:.0f}',
            xy=(x, y),
            ha=ha,
            va='center',
            fontsize=12,
        )


def plot_breakdown_manual(iter_times, stage_times, total_time, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(THESIS_TEXTWIDTH_INCHES * 1.3, 3.0))

    # Left: full training iteration
    ax = axes[0]
    components = ['Data Loading', 'GPU Transfer', 'Forward', 'Backward']
    times_iter = [
        iter_times['load_mean'],
        iter_times['transfer_mean'],
        iter_times['forward_mean'],
        iter_times['backward_mean'],
    ]

    wedges, texts = ax.pie(
        times_iter,
        colors=ITERATION_COLORS,
        startangle=90,
        wedgeprops={'linewidth': 0.8, 'edgecolor': 'white'},
    )
    add_outside_labels(ax, wedges, times_iter)

    ax.legend(
        wedges,
        components,
        loc='center left',
        bbox_to_anchor=(-0.80, 0.5),
        frameon=False,
    )
    ax.set_title(f'Training Iteration (ms)')

    # Right: forward pass stage breakdown
    ax = axes[1]
    stages = list(stage_times.keys())
    times_stage = list(stage_times.values())

    accounted = sum(times_stage)
    unaccounted = total_time - accounted
    if unaccounted > 0:
        stages.append('unaccounted')
        times_stage.append(unaccounted)

    stages, times_stage = group_small_items(stages, times_stage, total_time, threshold_pct=1.0)

    n_stages = len(stages)
    colors = STAGE_COLORS[:n_stages] if n_stages <= len(STAGE_COLORS) else (
        STAGE_COLORS + list(plt.cm.Set3(np.linspace(0, 1, n_stages - len(STAGE_COLORS))))
    )

    wedges, texts = ax.pie(
        times_stage,
        colors=colors,
        startangle=90,
        wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'},
    )
    add_outside_labels(ax, wedges, times_stage, x_offsets={0: 0.2}, y_offsets={0: 0.3})

    ax.legend(
        wedges,
        stages,
        loc='center left',
        bbox_to_anchor=(-0.65, 0.5),
        frameon=False,
    )
    ax.set_title('Forward Pass Breakdown (ms)')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_base = str(output_path).rsplit('.', 1)[0]
    fig.savefig(f'{output_base}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{output_base}.png', bbox_inches='tight', dpi=300)
    print(f'Plot saved to: {output_base}.pdf and .png')
    plt.close()


def main():
    iter_times = dict(ITER_TIMES)
    if iter_times['iteration_mean'] is None:
        iter_times['iteration_mean'] = (
            iter_times['load_mean']
            + iter_times['transfer_mean']
            + iter_times['forward_mean']
            + iter_times['backward_mean']
        )

    total_time = FORWARD_TOTAL_MS if FORWARD_TOTAL_MS is not None else float(sum(STAGE_TIMES.values()))
    plot_breakdown_manual(iter_times, STAGE_TIMES, total_time, OUTPUT_PATH)


if __name__ == '__main__':
    main()
