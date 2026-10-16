#!/usr/bin/env python3
"""Plot replay error curves recorded with ``eval_delta_replay_isaac.py --measurement_interval_s``.

Subcommands:
  single       error growth of one run, bootstrap CI, p90 and sustained threshold crossings
  compare      matched runs on identical windows, with paired differences against a reference
  uncertainty  a per-step uncertainty signal against the run's true error, next to a matched baseline

All statistics average overlapping windows within a trajectory, weight trajectories equally and use
2000 trajectory bootstrap resamples (seed 0) for pointwise 95% intervals.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (bootstrap_ci, bootstrap_indices, grouped_curve_values, load_run, matched_run_problem,  # noqa: E402
                    save_figure, trajectory_time_matrix, use_agg)

METRICS = {
    'joint_pos_rmse_rad': ('Joint position RMSE', 'rad'),
    'body_pos_error_m': ('Body position error', 'm'),
    'root_pos_error_m': ('Root position error', 'm'),
    'root_orientation_error_rad': ('Root orientation error', 'rad'),
    'root_height_error_m': ('Root height error', 'm'),
    'body_lin_vel_rmse_m_s': ('Body linear velocity RMSE', 'm/s'),
}
# Signals the replay evaluator records per window in ``disagreement_curve`` (ensemble runs).
ENSEMBLE_SIGNALS = {
    'normalized_disagreement': ('Raw output disagreement', 'Normalized units'),
    'force_disagreement_n': ('Force disagreement after clipping', 'N (component RMS std)'),
    'torque_disagreement_nm': ('Torque disagreement after clipping', 'N m (component RMS std)'),
    'saturation_fraction': ('Member output saturation', 'Fraction of member/components'),
}


def write_csv(path, rows):
    with Path(path).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def departure_time(curve, key, threshold, persistence_s):
    """First observed crossing sustained over a sampled span; None is right-censored."""
    start = None
    for point in curve:
        if point[key] > threshold:
            if start is None:
                start = point['requested_time_s']
            if point['requested_time_s'] - start >= persistence_s - 1e-9:
                return start
        else:
            start = None
    return None


# ---------------------------------------------------------------------------- single

def run_single(args, parser):
    if min(args.body_threshold_m, args.joint_threshold_rad) <= 0 or args.persistence_s < 0:
        parser.error('Thresholds must be positive and persistence nonnegative')
    run = load_run(args.run_dir)
    rows, manifest = run['rows'], run['manifest']
    if not rows or any('error_curve' not in row for row in rows):
        parser.error('Run requires --measurement_interval_s recordings')
    out = args.output_dir or args.run_dir / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    trajectories = sorted({r['trajectory'] for r in rows})
    times = sorted({p['requested_time_s'] for r in rows for p in r['error_curve']})
    if not times:
        parser.error('No valid sampled states')
    samples = [dict(trajectory=r['trajectory'], start=r['start'], status=r['status'], **point)
               for r in rows for point in r['error_curve']]
    write_csv(out / 'error_samples.csv', samples)
    grouped = grouped_curve_values(rows)
    matrices = {key: trajectory_time_matrix(grouped, key, trajectories, times) for key in METRICS}
    indices = bootstrap_indices(len(trajectories))
    stats = {}
    for key, matrix in matrices.items():
        low, high = bootstrap_ci(matrix, indices, ignore_nan=True)
        stats[key] = dict(mean=np.nanmean(matrix, axis=0), p90=np.nanpercentile(matrix, 90, axis=0), low=low, high=high)
    aggregate = [dict(requested_time_s=time, metric=key, trajectory_mean=stats[key]['mean'][i],
                      trajectory_p90=stats[key]['p90'][i], ci95_low=stats[key]['low'][i],
                      ci95_high=stats[key]['high'][i], available_trajectories=int(np.isfinite(matrices[key][:, i]).sum()),
                      available_windows=sum(len(grouped[trajectory, time]) for trajectory in trajectories))
                 for key in METRICS for i, time in enumerate(times)]
    write_csv(out / 'error_growth.csv', aggregate)
    thresholds = {'body_pos_error_m': args.body_threshold_m, 'joint_pos_rmse_rad': args.joint_threshold_rad}
    departures = [dict(trajectory=r['trajectory'], start=r['start'], status=r['status'],
                       **{key: departure_time(r['error_curve'], key, threshold, args.persistence_s)
                          for key, threshold in thresholds.items()}) for r in rows]
    (out / 'departure_times.json').write_text(json.dumps(departures, indent=2))
    report = dict(windows=len(rows), trajectories=len(trajectories),
                  complete=sum(r['status'] == 'complete' for r in rows),
                  failed=sum(r['status'] != 'complete' for r in rows),
                  requested_times_s=times,
                  actual_times_s=sorted({p['elapsed_s'] for p in samples}),
                  aggregation='Window means within trajectory, equal trajectory weighting; 2000 trajectory bootstrap resamples, seed 0',
                  thresholds=thresholds, persistence_s=args.persistence_s,
                  threshold_interpretation='Exploratory reference-departure markers, not falls or training-distribution boundaries',
                  final_metrics={key: {k: float(v[-1]) for k, v in value.items()} for key, value in stats.items()},
                  departures={key: dict(crossed=sum(r[key] is not None for r in departures),
                                        total=len(departures),
                                        median_crossing_time_s=float(np.median([r[key] for r in departures if r[key] is not None]))
                                        if any(r[key] is not None for r in departures) else None)
                              for key in thresholds})
    (out / 'analysis_summary.json').write_text(json.dumps(report, indent=2))

    plt = use_agg()
    plt.rcParams.update({'font.size': 15, 'axes.titlesize': 18, 'axes.labelsize': 15,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    label = 'Zero delta' if manifest['zero_delta'] else 'Delta replay'
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for ax, (key, (title, unit)) in zip(axes.flat, METRICS.items()):
        s = stats[key]
        ax.plot(times, s['mean'], color='#237898', lw=2.3, label='Trajectory mean')
        ax.fill_between(times, s['low'], s['high'], color='#237898', alpha=.18, label='95% bootstrap CI')
        ax.plot(times, s['p90'], color='#d48722', ls='--', lw=2, label='Trajectory p90')
        ax.set_title(title); ax.set_ylabel(unit); ax.set_xlabel('Elapsed replay time (s)'); ax.grid(alpha=.2)
        ax.set_ylim(bottom=0)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle(f'{label}: error accumulation', fontsize=23)
    fig.tight_layout(rect=(0, .06, 1, .95)); save_figure(fig, out, 'error_growth')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, key in zip(axes, thresholds):
        for trace in matrices[key]:
            ax.plot(times, trace, color='#667788', alpha=.23, lw=.8)
        ax.plot(times, stats[key]['mean'], color='#237898', lw=2.8)
        ax.axhline(thresholds[key], color='#b53d3d', ls='--', lw=1.5)
        title, unit = METRICS[key]
        ax.set_title(title); ax.set_xlabel('Elapsed replay time (s)'); ax.set_ylabel(unit)
        ax.grid(alpha=.2); ax.set_ylim(bottom=0)
    fig.suptitle(f'{label}: individual trajectory error growth', fontsize=23)
    fig.text(.5, .015, 'Thin lines: trajectory averages · Blue: overall mean · Red: exploratory departure threshold',
             ha='center', fontsize=13)
    fig.tight_layout(rect=(0, .07, 1, .94)); save_figure(fig, out, 'trajectory_error_growth')
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, threshold in thresholds.items():
        fractions = [np.mean([np.mean([r[key] is not None and r[key] <= t for r in departures if r['trajectory'] == trajectory])
                              for trajectory in trajectories]) for t in times]
        ax.step(times, 100 * np.array(fractions), where='post', lw=2.3,
                label=f'{METRICS[key][0]} > {threshold:g} {METRICS[key][1]}')
    ax.set_xlabel('Departure onset time (s)'); ax.set_ylabel('Windows with sustained crossing (%)')
    ax.set_ylim(0, 100); ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=12)
    ax.set_title(f'Sustained reference departure (≥ {args.persistence_s:g} s)')
    fig.tight_layout(); save_figure(fig, out, 'reference_departure')
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------- compare

def paired_rows(model, reference, model_matrices, reference_matrices, indices, times):
    rows = []
    for key, values in model_matrices.items():
        diff = reference_matrices[key] - values
        lo, hi = np.percentile(diff[indices].mean(axis=1), [2.5, 97.5], axis=0)
        for i, t in enumerate(times):
            rows.append(dict(model=model, reference=reference, metric=key, time_s=t,
                             reference_minus_model=float(diff.mean(axis=0)[i]),
                             ci95_low=float(lo[i]), ci95_high=float(hi[i])))
    return rows


def run_compare(args, parser):
    metrics_to_plot = {key: METRICS[key] for key in args.metrics}
    paths = {}
    for item in args.run:
        label, sep, path = item.partition('=')
        if not sep or not label or label in paths:
            parser.error('Each run needs a unique LABEL=directory')
        paths[label] = Path(path)
    if args.reference not in paths or len(paths) < 2:
        parser.error('Need reference and at least two runs')
    runs = {label: load_run(path) for label, path in paths.items()}
    problem = matched_run_problem(runs, require_complete=True)
    if problem:
        parser.error(problem)
    if any(r['status'] != 'complete' for run in runs.values() for r in run['rows']):
        parser.error('Failed windows require an explicit censoring protocol')
    first = next(iter(runs))
    trajectories = sorted({r['trajectory'] for r in runs[first]['rows']})
    times = sorted({point['requested_time_s'] for r in runs[first]['rows'] for point in r['error_curve']})
    matrices = {}
    for label, run in runs.items():
        grouped = grouped_curve_values(run['rows'])
        if len(grouped) != len(trajectories) * len(times):
            parser.error(f'{label} missing sampled times')
        matrices[label] = {key: trajectory_time_matrix(grouped, key, trajectories, times) for key in metrics_to_plot}
    indices = bootstrap_indices(len(trajectories))
    stats, paired = {}, []
    for label, metrics in matrices.items():
        stats[label] = {}
        for key, values in metrics.items():
            if not np.isfinite(values).all():
                parser.error('Nonfinite error measurements')
            low, high = bootstrap_ci(values, indices)
            stats[label][key] = dict(mean=values.mean(axis=0), low=low, high=high)
        if label != args.reference:
            paired += paired_rows(label, args.reference, metrics, matrices[args.reference], indices, times)
    for pair in args.pair:
        label, sep, reference = pair.partition('=')
        if not sep or label not in runs or reference not in runs or label == reference:
            parser.error('Additional pairs must be distinct known MODEL=REFERENCE labels')
        if reference != args.reference:
            paired += paired_rows(label, reference, matrices[label], matrices[reference], indices, times)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'paired_error_differences.csv', paired)
    report = dict(windows=len(runs[first]['rows']), trajectories=len(trajectories),
                  runs={k: str(v.resolve()) for k, v in paths.items()},
                  method='Window averages within trajectory; equal trajectory weights; paired bootstrap 2000 samples, seed 0; pointwise 95% intervals',
                  final_metrics={label: {key: {k: float(v[-1]) for k, v in s.items()} for key, s in metrics.items()}
                                 for label, metrics in stats.items()},
                  final_paired_differences=[r for r in paired if r['time_s'] == times[-1]])
    (out / 'comparison.json').write_text(json.dumps(report, indent=2))

    plt = use_agg()
    plt.rcParams.update({'font.size': 17, 'axes.titlesize': 20, 'axes.labelsize': 17,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    columns = min(3, len(metrics_to_plot)); nrows = (len(metrics_to_plot) + columns - 1) // columns
    fig, axes = plt.subplots(nrows, columns, figsize=(17, 5 * nrows + 1), squeeze=False)
    colors = ['#777777', '#237898', '#d48722', '#7b2cbf']
    for ax, (key, (title, unit)) in zip(axes.flat, metrics_to_plot.items()):
        for i, label in enumerate(runs):
            s = stats[label][key]; color = colors[i % len(colors)]
            ax.plot(times, s['mean'], color=color, lw=2.5, label=label)
            ax.fill_between(times, s['low'], s['high'], color=color, alpha=.13)
        ax.set_title(title); ax.set_xlabel('Elapsed replay time (s)'); ax.set_ylabel(unit)
        ax.grid(alpha=.2); ax.set_ylim(bottom=0)
    for ax in list(axes.flat)[len(metrics_to_plot):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(runs), frameon=False)
    fig.tight_layout(rect=(0, .07 if nrows > 1 else .14, 1, 1))
    save_figure(fig, out, 'error_growth_comparison')
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------- uncertainty

def run_uncertainty(args, parser):
    runs = {'run': load_run(args.run_dir), 'baseline': load_run(args.baseline_dir)}
    problem = matched_run_problem(runs)
    if problem:
        parser.error(f'Comparison mismatch: {problem}')
    rows, zero = runs['run']['rows'], runs['baseline']['rows']
    if any(r['status'] != 'complete' for r in rows + zero):
        parser.error('Incomplete windows require explicit censoring analysis before comparison')
    if any(args.signal_field not in r for r in rows):
        parser.error(f'Run has no per-window {args.signal_field!r}')
    signals = dict(ENSEMBLE_SIGNALS) if args.signal_field == 'disagreement_curve' else {}
    signals.setdefault(args.signal, (args.signal, 'signal units'))
    trajectories = sorted({r['trajectory'] for r in rows})
    points = [dict(trajectory=r['trajectory'], start=r['start'], **p) for r in rows for p in r[args.signal_field]]
    times = sorted({p['requested_time_s'] for p in points})
    out = args.output_dir or args.run_dir / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'disagreement_samples.csv', points)
    signal_grouped = grouped_curve_values(rows, args.signal_field)
    indices = bootstrap_indices(len(trajectories))
    extra = ('clipped_normalized_disagreement',) if args.signal_field == 'disagreement_curve' else ()
    metrics = {key: trajectory_time_matrix(signal_grouped, key, trajectories, times) for key in (*signals, *extra)}
    stats = {}
    for key, values in metrics.items():
        if not np.isfinite(values).all():
            parser.error('Missing/nonfinite uncertainty samples')
        low, high = bootstrap_ci(values, indices)
        stats[key] = dict(mean=values.mean(axis=0), p90=np.percentile(values, 90, axis=0), low=low, high=high)
    aggregates = [dict(metric=key, requested_time_s=t, mean=s['mean'][i], p90=s['p90'][i],
                       ci95_low=s['low'][i], ci95_high=s['high'][i])
                  for key, s in stats.items() for i, t in enumerate(times)]
    write_csv(out / 'disagreement_growth.csv', aggregates)
    from scipy.stats import spearmanr
    associations = []
    uncertainty = {(r['trajectory'], r['start']): {p['requested_time_s']: p[args.signal] for p in r[args.signal_field]}
                   for r in rows}
    for time in times[1:]:
        u, e = [], []
        for r in rows:
            point = next(p for p in r['error_curve'] if p['requested_time_s'] == time)
            u.append(uncertainty[r['trajectory'], r['start']][time]); e.append(point[args.error_metric])
        associations.append(dict(time_s=time, spearman=float(spearmanr(u, e).statistic)))
    manifest = runs['run']['manifest']
    report = dict(windows=len(rows), trajectories=len(trajectories), signal=args.signal,
                  signal_field=args.signal_field, error_metric=args.error_metric,
                  aggregation='Average windows within trajectory, then equal trajectory weighting; 2000 trajectory bootstrap resamples',
                  final={key: {k: float(v[-1]) for k, v in s.items()} for key, s in stats.items()},
                  same_time_error_correlations=associations,
                  correlation_note='Descriptive within-horizon window correlations; overlapping windows are dependent; no significance claim')
    if 'ensemble_members' in manifest:
        report.update(members=len(manifest['ensemble_members']), protocol=manifest.get('ensemble_protocol'),
                      definition='Population std across members per output component; RMS across six raw components for normalized uncertainty; RMS across three clipped force/torque components for physical disagreement')
    (out / 'ensemble_summary.json').write_text(json.dumps(report, indent=2))

    plt = use_agg()
    plt.rcParams.update({'font.size': 15, 'axes.titlesize': 18, 'axes.labelsize': 15,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    panels = list(signals.items())
    ncols = 2 if len(panels) > 1 else 1
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), sharex=True, squeeze=False)
    for ax, (key, (title, unit)) in zip(axes.flat, panels):
        s = stats[key]; ax.plot(times, s['mean'], color='#237898', lw=2.5, label='Trajectory mean')
        ax.fill_between(times, s['low'], s['high'], color='#237898', alpha=.18, label='95% bootstrap CI')
        ax.plot(times, s['p90'], color='#d48722', lw=2, ls='--', label='Trajectory p90')
        ax.set_title(title); ax.set_ylabel(unit); ax.set_xlabel('Elapsed replay time (s)'); ax.grid(alpha=.2)
        ax.set_ylim(bottom=0)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle(args.title, fontsize=23)
    fig.tight_layout(rect=(0, .06, 1, .95)); save_figure(fig, out, 'ensemble_disagreement')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, key, title, unit in zip(axes, ('body_pos_error_m', 'joint_pos_rmse_rad', 'root_height_error_m'),
                                    ('Body position error', 'Joint position RMSE', 'Root height error'), ('m', 'rad', 'm')):
        for records, label, color in ((zero, args.baseline_label, '#777777'), (rows, args.run_label, '#237898')):
            values = trajectory_time_matrix(grouped_curve_values(records), key, trajectories, times)
            low, high = bootstrap_ci(values, indices)
            ax.plot(times, values.mean(axis=0), color=color, lw=2.5, label=label)
            ax.fill_between(times, low, high, color=color, alpha=.15)
        ax.set_title(title); ax.set_xlabel('Elapsed replay time (s)'); ax.set_ylabel(unit); ax.grid(alpha=.2)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .13, 1, 1)); save_figure(fig, out, 'ensemble_vs_zero_error_growth')
    fig, ax = plt.subplots(figsize=(10, 7))
    heat = metrics[args.signal]
    image = ax.imshow(heat, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=np.percentile(heat, 95),
                      extent=(-.125, max(times) + .125, -.5, len(trajectories) - .5))
    ax.set_yticks(np.arange(0, len(trajectories), 5), [str(t) for t in trajectories[::5]])
    ax.set_xlabel('Elapsed replay time (s)'); ax.set_ylabel('Validation trajectory ID')
    ax.set_title(f'{signals[args.signal][0]} by trajectory')
    fig.colorbar(image, ax=ax, label=f'{args.signal} (color clipped at p95)')
    fig.tight_layout(); save_figure(fig, out, 'trajectory_disagreement')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('single', help='Error growth of one run')
    p.add_argument('--run_dir', type=Path, required=True)
    p.add_argument('--output_dir', type=Path)
    p.add_argument('--body_threshold_m', type=float, default=.25)
    p.add_argument('--joint_threshold_rad', type=float, default=.30)
    p.add_argument('--persistence_s', type=float, default=.5)
    p.set_defaults(handler=run_single)

    p = sub.add_parser('compare', help='Matched runs with paired differences')
    p.add_argument('--run', action='append', required=True, help='LABEL=run directory; repeat')
    p.add_argument('--reference', required=True, help='Reference label for paired error differences')
    p.add_argument('--pair', action='append', default=[], help='Additional MODEL=REFERENCE paired comparison')
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--metrics', nargs='+', choices=list(METRICS),
                   default=['body_pos_error_m', 'joint_pos_rmse_rad', 'root_height_error_m'])
    p.set_defaults(handler=run_compare)

    p = sub.add_parser('uncertainty', help='Uncertainty signal vs true error, with a matched baseline')
    p.add_argument('--run_dir', type=Path, required=True)
    p.add_argument('--baseline_dir', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, help='Defaults to <run_dir>/analysis')
    p.add_argument('--signal_field', default='disagreement_curve', help='Per-window curve holding the signal')
    p.add_argument('--signal', default='normalized_disagreement', help='Signal key within each curve point')
    p.add_argument('--error_metric', default='body_pos_error_m', choices=list(METRICS))
    p.add_argument('--run_label', default='Ungated ensemble mean')
    p.add_argument('--baseline_label', default='Zero delta')
    p.add_argument('--title', default='Ensemble disagreement', help='Main figure title')
    p.set_defaults(handler=run_uncertainty)

    args = parser.parse_args()
    args.handler(args, parser)


if __name__ == '__main__':
    main()
