#!/usr/bin/env python3
"""Compare independently recorded delta replay evaluations on identical schedules."""
import argparse
import csv
import json
from pathlib import Path
from eval_delta_replay_isaac import summarize


def compare_runs(runs, baseline=None):
    reports, rows, signatures = {}, {}, []
    for label, folder in runs.items():
        folder = Path(folder)
        manifest = json.loads((folder/'manifest.json').read_text())
        if manifest['status'] != 'complete':
            raise ValueError(f'{label} is not a completed replay')
        report = json.loads((folder/'summary.json').read_text())
        windows = json.loads((folder/'windows.json').read_text())
        signatures.append((manifest['dataset_sha256'], manifest['num_envs'], manifest['seed'],
                           report['configuration_contract'], windows))
        reports[label] = report
        rows[label] = json.loads((folder/'partial_results.json').read_text())
    if not reports or any(s != signatures[0] for s in signatures[1:]):
        raise ValueError('Runs must use the same dataset, schedule, batching, seed and dynamics configuration')
    if baseline is not None and baseline not in reports:
        raise ValueError('Baseline must name one of the supplied runs')
    def identity(row):
        return row['trajectory'], row['start'], row['steps']
    common = set.intersection(*(set(identity(r) for r in group if r['status']=='complete') for group in rows.values()))
    if not common:
        raise ValueError('No successfully evaluated windows are shared across runs')
    result = {}
    for label, report in reports.items():
        result[label] = dict(run_dir=str(Path(runs[label]).resolve()), scheduled_summary=report['summary'],
                             matched_window_count=len(common),
                             matched_summary=summarize([r for r in rows[label] if identity(r) in common]))
    if baseline:
        for label, report in result.items():
            for horizon, group in report['matched_summary'].items():
                for key, values in group['metrics'].items():
                    base = result[baseline]['matched_summary'][horizon]['metrics'][key]['trajectory_equal_weight_mean']
                    values['relative_improvement'] = 1-values['trajectory_equal_weight_mean']/base if base else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, metavar='LABEL=RUN_DIR')
    parser.add_argument('--baseline', help='Optional label for relative improvements')
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for spec in args.run:
        label, sep, folder = spec.partition('=')
        if not sep or not label or label in runs:
            parser.error('Use unique LABEL=RUN_DIR specifications')
        runs[label] = Path(folder)
    result = compare_runs(runs, args.baseline)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'comparison.json').write_text(json.dumps(result, indent=2))
    records = [dict(model=label, steps=int(horizon), metric=metric, **values)
               for label, report in result.items() for horizon, group in report['matched_summary'].items()
               for metric, values in group['metrics'].items()]
    with (args.output_dir/'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in records))))
        writer.writeheader()
        writer.writerows(records)
    print(f'Comparison: {args.output_dir}')


if __name__ == '__main__':
    main()
