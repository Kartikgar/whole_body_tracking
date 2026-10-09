#!/usr/bin/env python3
"""Plot paired delta replay comparisons produced by compare_delta_replay_results.py."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

METRICS = {
    'joint_pos_rmse_rad': ('Joint position RMSE', 'rad'),
    'joint_vel_rmse_rad_s': ('Joint velocity RMSE', 'rad/s'),
    'body_pos_error_m': ('Body position error', 'm'),
    'root_pos_error_m': ('Root position error', 'm'),
    'body_orientation_error_rad': ('Body orientation error', 'rad'),
    'body_lin_vel_rmse_m_s': ('Body linear velocity RMSE', 'm/s'),
    'body_ang_vel_rmse_rad_s': ('Body angular velocity RMSE', 'rad/s'),
}


def paired_trajectory_means(comparison, steps):
    """Match windows before averaging within each trajectory, then weight trajectories equally."""
    rows = {}
    for label, report in comparison.items():
        records = json.loads((Path(report['run_dir']) / 'partial_results.json').read_text())
        rows[label] = {(r['trajectory'], r['start']): r for r in records
                       if r['status'] == 'complete' and r['steps'] == steps}
    common = sorted(set.intersection(*(set(r) for r in rows.values())))
    if not common:
        raise ValueError('No common completed windows for selected horizon')
    trajectories = sorted({t for t, _ in common})
    keys = [f'{prefix}_{m}' for prefix in ('mean', 'endpoint') for m in METRICS]
    keys += ['force_mean_n', 'torque_mean_nm']
    values = {label: {key: np.array([
        np.mean([records[w][key] for w in common if w[0] == t])
        for t in trajectories]) for key in keys} for label, records in rows.items()}
    return values, trajectories, len(common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--steps', type=int, help='Required if comparison has multiple horizons')
    parser.add_argument('--output_dir', type=Path, help='Defaults to comparison file directory')
    parser.add_argument('--bootstrap_samples', type=int, default=5000)
    parser.add_argument('--title', help='Optional prefix for figure titles; omitted for compact metric titles')
    parser.add_argument('--highlight_models', nargs='+', default=[], metavar='LABEL',
                        help='Enclose adjacent model tick labels in a red outline')
    parser.add_argument('--oracle_model', help='Label to annotate as Oracle')
    parser.add_argument('--oracle_style', choices=('pointer', 'badge', 'inline'), default='pointer',
                        help='Presentation of the optional Oracle annotation')
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    if args.baseline not in comparison:
        parser.error('Baseline must name a model in comparison.json')
    horizons = list(next(iter(comparison.values()))['matched_summary'])
    if args.steps is None and len(horizons) != 1:
        parser.error('Select --steps for comparisons with multiple horizons')
    steps = args.steps if args.steps is not None else int(horizons[0])
    if str(steps) not in horizons or args.bootstrap_samples < 1:
        parser.error('Invalid horizon or bootstrap sample count')
    values, trajectories, window_count = paired_trajectory_means(comparison, steps)
    labels = list(values)
    if any(label not in labels for label in args.highlight_models):
        parser.error('--highlight_models must name models in the comparison')
    if args.oracle_model is not None and args.oracle_model not in labels:
        parser.error('--oracle_model must name a model in the comparison')
    learned = [label for label in labels if label != args.baseline]
    if not learned:
        parser.error('At least one non-baseline model is required')
    output = args.output_dir or args.comparison.parent
    output.mkdir(parents=True, exist_ok=True)
    first = json.loads((Path(comparison[args.baseline]['run_dir']) / 'partial_results.json').read_text())
    horizon_s = next(r['horizon_s'] for r in first if r['steps'] == steps)
    title_prefix = args.title + ': ' if args.title else ''
    rng = np.random.default_rng(0)
    samples = rng.integers(0, len(trajectories), (args.bootstrap_samples, len(trajectories)))
    statistics = []
    for label in labels:
        for key, array in values[label].items():
            base = values[args.baseline][key]
            boot = array[samples].mean(axis=1)
            low, high = np.percentile(boot, [2.5, 97.5])
            record = dict(model=label, metric=key, trajectory_mean=float(array.mean()),
                          ci95_low=float(low), ci95_high=float(high))
            if key not in ('force_mean_n', 'torque_mean_nm'):
                improvement = 100 * (1 - array.mean() / base.mean())
                paired_boot = 100 * (1 - boot / base[samples].mean(axis=1))
                lo, hi = np.percentile(paired_boot, [2.5, 97.5])
                record.update(improvement_pct=float(improvement), improvement_ci95_low=float(lo),
                              improvement_ci95_high=float(hi),
                              improved_trajectory_fraction=float(np.mean(array < base)))
            statistics.append(record)
    (output / 'paired_statistics.json').write_text(json.dumps(dict(
        baseline=args.baseline, steps=steps, horizon_s=horizon_s, trajectories=len(trajectories),
        matched_windows=window_count, bootstrap_samples=args.bootstrap_samples,
        method='Equal trajectory weighting; paired trajectory bootstrap, seed 0; pointwise 95% intervals',
        statistics=statistics), indent=2))
    with (output / 'paired_statistics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in statistics for k in r)))
        writer.writeheader()
        writer.writerows(statistics)
    lookup = {(r['model'], r['metric']): r for r in statistics}
    heatmap_labels = {
        label: re.sub(r'(\+\s+\D+?)(\d+)$', r'\1\n\2', label.replace(' + ', '\n+ '))
        for label in learned
    }
    tick_model_names = {display: label for label, display in heatmap_labels.items()}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.size': 17, 'axes.titlesize': 20, 'axes.labelsize': 17,
                         'xtick.labelsize': 17, 'ytick.labelsize': 17,
                         'axes.titlepad': 12, 'axes.labelpad': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})

    def save(fig, name):
        # Measure the final text layout, so outlines never rely on guessed label widths.
        # Figure coordinates keep the geometry identical in PNG and vector PDF exports.
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Bbox
        if args.oracle_model and args.oracle_style == 'inline':
            for ax in fig.axes:
                for axis in (ax.xaxis, ax.yaxis):
                    ticks = axis.get_ticklabels()
                    if any(t.get_text() == args.oracle_model for t in ticks):
                        axis.set_ticks(axis.get_ticklocs(), labels=[
                            t.get_text() + '\n(Oracle)' if t.get_text() == args.oracle_model
                            else t.get_text() for t in ticks])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
                selected = [(i, tick) for i, tick in enumerate(ticks) if tick.get_visible()
                            and tick_model_names.get(tick.get_text(), tick.get_text()) in args.highlight_models]
                # Separate nonadjacent selections instead of boxing intervening models.
                groups = []
                for i, tick in selected:
                    if not groups or i != groups[-1][-1][0] + 1:
                        groups.append([])
                    groups[-1].append((i, tick))
                for group in groups:
                    bounds = Bbox.union([tick.get_window_extent(renderer) for _, tick in group])
                    pad = fig.dpi * 2.5 / 72
                    bounds = Bbox.from_extents(bounds.x0-pad, bounds.y0-pad,
                                               bounds.x1+pad, bounds.y1+pad)
                    bounds = bounds.transformed(fig.transFigure.inverted())
                    fig.add_artist(Rectangle((bounds.x0, bounds.y0), bounds.width, bounds.height,
                                             transform=fig.transFigure, fill=False,
                                             edgecolor='#b62828', linewidth=1.3,
                                             clip_on=False, zorder=20))
            if args.oracle_model and args.oracle_style != 'inline':
                for direction, ticks in (('x', ax.get_xticklabels()), ('y', ax.get_yticklabels())):
                    for tick in ticks:
                        if not tick.get_visible() or tick.get_text() != args.oracle_model:
                            continue
                        bounds = tick.get_window_extent(renderer)
                        unit = fig.dpi / 72
                        if direction == 'y':
                            target = (bounds.x0-3*unit, (bounds.y0+bounds.y1)/2)
                            position = (bounds.x0-27*unit, target[1])
                            align = 'right'
                        else:
                            target = (bounds.x1+3*unit, (bounds.y0+bounds.y1)/2)
                            position = (bounds.x1+27*unit, target[1])
                            align = 'left'
                        target, position = fig.transFigure.inverted().transform([target, position])
                        ax.annotate('Oracle', xy=target, xytext=position,
                                    xycoords=fig.transFigure, textcoords=fig.transFigure,
                                    ha=align, va='center', fontsize=17, fontweight='semibold',
                                    color='#7b2cbf', annotation_clip=False, clip_on=False,
                                    arrowprops=dict(arrowstyle='->', color='#7b2cbf', lw=1,
                                                    shrinkA=4, shrinkB=1) if args.oracle_style == 'pointer' else None,
                                    bbox=dict(boxstyle='round,pad=.2', fc='#f4f4f4', ec='#777777', lw=.7)
                                    if args.oracle_style == 'badge' else None)
        fig.savefig(output / f'{name}.png', dpi=300, bbox_inches='tight')
        fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)

    # Separate physical units rather than averaging incompatible error metrics.
    fig, axes = plt.subplots(2, 3, figsize=(16, max(9, 1.1 * len(labels))), sharey=True)
    colors = ['#777777' if label == args.baseline else '#33789b' for label in labels]
    for row, prefix in enumerate(('mean', 'endpoint')):
        for col, metric in enumerate(('joint_pos_rmse_rad', 'body_pos_error_m', 'root_pos_error_m')):
            ax = axes[row, col]
            key = f'{prefix}_{metric}'
            stats = [lookup[label, key] for label in labels]
            means = np.array([r['trajectory_mean'] for r in stats])
            errors = np.array([[r['trajectory_mean'] - r['ci95_low'] for r in stats],
                               [r['ci95_high'] - r['trajectory_mean'] for r in stats]])
            ax.barh(labels, means, color=colors, xerr=errors, capsize=3)
            ax.set_title(f'{"Window mean" if prefix == "mean" else "Endpoint"}\n{METRICS[metric][0]}')
            ax.set_xlabel(f'Error ({METRICS[metric][1]})\nLower is better')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.grid(axis='x', alpha=.2)
            ax.set_axisbelow(True)
    axes[0, 0].invert_yaxis()
    fig.suptitle(f'{title_prefix}Tracking errors', fontsize=24)
    fig.tight_layout(rect=(0, 0, 1, .97))
    save(fig, 'tracking_errors')

    keys = [f'{prefix}_{metric}' for prefix in ('mean', 'endpoint') for metric in METRICS]
    matrix = np.array([[lookup[label, key]['improvement_pct'] for label in learned] for key in keys])
    limit = max(1, np.abs(matrix).max())
    fig, ax = plt.subplots(figsize=(max(14, 5 + 1.5 * len(learned)), 11))
    im = ax.imshow(matrix, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_xticks(range(len(learned)), [heatmap_labels[label] for label in learned])
    ax.set_yticks(range(len(keys)), [f'{"Mean" if key.startswith("mean_") else "Endpoint"} · '
                                   f'{METRICS[key.split("_", 1)[1]][0]}' for key in keys])
    ax.set_xlabel('Delta model')
    ax.set_ylabel('Tracking metric')
    for i, key in enumerate(keys):
        for j, label in enumerate(learned):
            r = lookup[label, key]
            significant = r['improvement_ci95_low'] > 0 or r['improvement_ci95_high'] < 0
            ax.text(j, i, f'{matrix[i,j]:+.1f}%' + (' *' if significant else ''), ha='center', va='center',
                    color='white' if abs(matrix[i,j]) > .6 * limit else 'black', fontsize=17)
    fig.colorbar(im, ax=ax, label='Error reduction (%) · positive is better', shrink=.7,
                 fraction=.035, pad=.035)
    ax.set_title(f'{title_prefix}Error reduction relative to {args.baseline}', pad=14)
    fig.tight_layout()
    save(fig, 'relative_improvement')

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, metric in zip(axes, ('joint_pos_rmse_rad', 'body_pos_error_m')):
        key = f'endpoint_{metric}'
        differences = [values[args.baseline][key] - values[label][key] for label in learned]
        ax.boxplot(differences, vert=False, tick_labels=learned, showfliers=False, widths=.45)
        for i, diff in enumerate(differences):
            jitter = rng.uniform(-.13, .13, len(diff))
            ax.scatter(diff, i + 1 + jitter, alpha=.4, s=12, color='#33789b')
        ax.axvline(0, color='#777777', linestyle='--')
        ax.invert_yaxis()
        ax.set_title(f'Endpoint\n{METRICS[metric][0]}')
        ax.set_xlabel(f'Baseline − model error ({METRICS[metric][1]})\nPositive = better than {args.baseline}')
        ax.grid(axis='x', alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    fig.suptitle(f'{title_prefix}Paired endpoint error differences', fontsize=24)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(fig, 'paired_trajectory_transfer')

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, key, title, unit in zip(axes, ('force_mean_n', 'torque_mean_nm'),
                                   ('Applied pelvis force', 'Applied pelvis torque'), ('N', 'N m')):
        stats = [lookup[label, key] for label in labels]
        means = np.array([r['trajectory_mean'] for r in stats])
        errors = np.array([[r['trajectory_mean']-r['ci95_low'] for r in stats],
                           [r['ci95_high']-r['trajectory_mean'] for r in stats]])
        ax.barh(labels, means, color=colors, xerr=errors, capsize=3)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(f'Mean wrench magnitude ({unit})')
        ax.grid(axis='x', alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    fig.tight_layout()
    save(fig, 'applied_wrenches')
    print(f'Plots and paired statistics: {output.resolve()}')


if __name__ == '__main__':
    main()
