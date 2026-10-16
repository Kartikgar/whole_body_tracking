#!/usr/bin/env python3
"""Plot instantaneous replay error growth and sustained threshold crossings."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = {
    'joint_pos_rmse_rad': ('Joint position RMSE', 'rad'),
    'body_pos_error_m': ('Body position error', 'm'),
    'root_pos_error_m': ('Root position error', 'm'),
    'root_orientation_error_rad': ('Root orientation error', 'rad'),
    'root_height_error_m': ('Root height error', 'm'),
    'body_lin_vel_rmse_m_s': ('Body linear velocity RMSE', 'm/s'),
}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--body_threshold_m', type=float, default=.25)
    parser.add_argument('--joint_threshold_rad', type=float, default=.30)
    parser.add_argument('--persistence_s', type=float, default=.5)
    args = parser.parse_args()
    if min(args.body_threshold_m, args.joint_threshold_rad) <= 0 or args.persistence_s < 0:
        parser.error('Thresholds must be positive and persistence nonnegative')
    rows = json.loads((args.run_dir / 'partial_results.json').read_text())
    if not rows or any('error_curve' not in row for row in rows):
        parser.error('Run requires --measurement_interval_s recordings')
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    out = args.output_dir or args.run_dir / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    trajectories = sorted({r['trajectory'] for r in rows})
    times = sorted({p['requested_time_s'] for r in rows for p in r['error_curve']})
    if not times:
        parser.error('No valid sampled states')
    samples = [dict(trajectory=r['trajectory'], start=r['start'], status=r['status'], **point)
               for r in rows for point in r['error_curve']]
    with (out / 'error_samples.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader(); writer.writerows(samples)
    # Average overlapping windows within each trajectory; trajectories then carry equal weight.
    matrices = {}
    grouped = defaultdict(list)
    for point in samples:
        grouped[point['trajectory'], point['requested_time_s']].append(point)
    for key in METRICS:
        matrices[key] = np.array([[np.mean([p[key] for p in grouped[trajectory, time]])
                                  if grouped[trajectory, time] else np.nan
                                  for time in times] for trajectory in trajectories])
    rng = np.random.default_rng(0)
    indices = rng.integers(len(trajectories), size=(2000, len(trajectories)))
    stats = {}
    for key, matrix in matrices.items():
        means = np.nanmean(matrix, axis=0)
        bounds = np.nanpercentile(np.nanmean(matrix[indices], axis=1), [2.5, 97.5], axis=0)
        stats[key] = dict(mean=means, p90=np.nanpercentile(matrix, 90, axis=0), low=bounds[0], high=bounds[1])
    aggregate = [dict(requested_time_s=time, metric=key, trajectory_mean=stats[key]['mean'][i],
                      trajectory_p90=stats[key]['p90'][i], ci95_low=stats[key]['low'][i],
                      ci95_high=stats[key]['high'][i], available_trajectories=int(np.isfinite(matrices[key][:, i]).sum()),
                      available_windows=sum(len(grouped[trajectory, time]) for trajectory in trajectories))
                 for key in METRICS for i, time in enumerate(times)]
    with (out / 'error_growth.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader(); writer.writerows(aggregate)
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
                  final_metrics={key: {k: float(v[-1]) for k,v in value.items()} for key,value in stats.items()},
                  departures={key: dict(crossed=sum(r[key] is not None for r in departures),
                                        total=len(departures),
                                        median_crossing_time_s=float(np.median([r[key] for r in departures if r[key] is not None]))
                                        if any(r[key] is not None for r in departures) else None)
                              for key in thresholds})
    (out / 'analysis_summary.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 15, 'axes.titlesize': 18, 'axes.labelsize': 15,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    label = 'Zero delta' if manifest['zero_delta'] else 'Delta replay'
    def save(fig, name):
        fig.savefig(out / (name + '.png'), dpi=300, bbox_inches='tight')
        fig.savefig(out / (name + '.pdf'), bbox_inches='tight'); plt.close(fig)
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
    fig.tight_layout(rect=(0, .06, 1, .95)); save(fig, 'error_growth')
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
    fig.text(.5, .015, 'Thin lines: trajectory averages · Blue: overall mean · Red: exploratory departure threshold', ha='center', fontsize=13)
    fig.tight_layout(rect=(0, .07, 1, .94)); save(fig, 'trajectory_error_growth')
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, threshold in thresholds.items():
        fractions = [np.mean([np.mean([r[key] is not None and r[key] <= t for r in departures if r['trajectory'] == trajectory])
                              for trajectory in trajectories]) for t in times]
        ax.step(times, 100 * np.array(fractions), where='post', lw=2.3,
                label=f'{METRICS[key][0]} > {threshold:g} {METRICS[key][1]}')
    ax.set_xlabel('Departure onset time (s)'); ax.set_ylabel('Windows with sustained crossing (%)')
    ax.set_ylim(0, 100); ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=12)
    ax.set_title(f'Sustained reference departure (≥ {args.persistence_s:g} s)')
    fig.tight_layout(); save(fig, 'reference_departure')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
