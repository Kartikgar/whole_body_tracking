#!/usr/bin/env python3
"""Summarize matched multi-condition replay screens without pooling datasets."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = {
    'body_pos_error_m': ('World body position error', 'm'),
    'joint_pos_rmse_rad': ('Joint position RMSE', 'rad'),
    'root_height_error_m': ('Root height error', 'm'),
    'root_orientation_error_rad': ('Root orientation error', 'rad'),
}


def paired_arrays(rows, key):
    conditions = list(dict.fromkeys(r['condition'] for r in rows))
    lookup = {c:{(r['trajectory'], r['start']):r for r in rows if r['condition']==c} for c in conditions}
    common = set.intersection(*(set(v) for v in lookup.values()))
    if any(set(v)!=common for v in lookup.values()):
        raise ValueError('Conditions must have identical windows')
    windows = sorted(common)
    trajectories = sorted({t for t, _ in windows})
    arrays = {}
    raw = {}
    for condition in conditions:
        raw[condition] = np.array([lookup[condition][w][key] for w in windows])
        arrays[condition] = np.array([raw[condition][[i for i,w in enumerate(windows) if w[0]==t]].mean()
                                      for t in trajectories])
    return arrays, raw, trajectories


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True, help='LABEL=run_dir')
    p.add_argument('--output_dir', type=Path, required=True)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':14, 'axes.titlesize':17, 'axes.labelsize':15,
                         'xtick.labelsize':13, 'ytick.labelsize':13, 'pdf.fonttype':42})
    runs = {}
    reports = []
    curves = {}
    conditions = None
    for item in args.runs:
        label, path = item.split('=',1)
        path = Path(path)
        manifest = json.loads((path/'manifest.json').read_text())
        if manifest['status']!='complete':
            raise ValueError(f'Incomplete run: {path}')
        rows = json.loads((path/'partial_results.json').read_text())
        if any(r['status']!='complete' for r in rows):
            raise ValueError('Failed windows need explicit handling; refusing silent exclusion')
        current_conditions = list(dict.fromkeys(r['condition'] for r in rows))
        if conditions is not None and current_conditions!=conditions:
            raise ValueError('Different condition schedules')
        conditions = current_conditions
        rng = np.random.default_rng(0)
        _, _, trajectories = paired_arrays(rows, 'endpoint_body_pos_error_m')
        indices = rng.integers(len(trajectories), size=(2000,len(trajectories)))
        runs[label] = dict(rows=rows, trajectories=len(trajectories), windows=len(rows)//len(conditions),
                           dataset=manifest['dataset'], source=str(path.resolve()))
        keys = [f'{prefix}_{m}' for prefix in ('endpoint','mean') for m in
                ('body_pos_error_m','joint_pos_rmse_rad','root_pos_error_m')]
        keys += ['gate_mean','force_mean_n','torque_mean_nm']
        for key in keys:
            values, raw, _ = paired_arrays(rows,key)
            for condition in conditions:
                diff = values['zero_delta']-values[condition]
                ci = np.percentile(diff[indices].mean(axis=1),[2.5,97.5])
                absolute_ci = np.percentile(values[condition][indices].mean(axis=1),[2.5,97.5])
                raw_diff = raw['zero_delta']-raw[condition]
                reports.append(dict(dataset=label,condition=condition,metric=key,
                                    trajectory_mean=float(values[condition].mean()),
                                    ci95_low=float(absolute_ci[0]),ci95_high=float(absolute_ci[1]),
                                    baseline_minus_model=float(diff.mean()),
                                    improvement_ci95_low=float(ci[0]),improvement_ci95_high=float(ci[1]),
                                    improvement_pct=float(100*diff.mean()/values['zero_delta'].mean())
                                    if values['zero_delta'].mean()!=0 else None,
                                    windows_improved_fraction=float(np.mean(raw_diff>0)),
                                    trajectories_improved_fraction=float(np.mean(diff>0)),
                                    window_p90_error_increase=float(np.percentile(-raw_diff,90))))
        times = sorted({point['requested_time_s'] for r in rows for point in r['error_curve']})
        for metric in METRICS:
            for condition in conditions:
                selected = [r for r in rows if r['condition']==condition]
                values = np.array([[np.mean([next(point[metric] for point in r['error_curve']
                                                  if point['requested_time_s']==time)
                                              for r in selected if r['trajectory']==t])
                                    for time in times] for t in trajectories])
                ci = np.percentile(values[indices].mean(axis=1),[2.5,97.5],axis=0)
                curves[label,condition,metric] = (times,values.mean(axis=0),ci)
    out = args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    with (out/'screening_statistics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(reports[0]));writer.writeheader();writer.writerows(reports)
    summary = dict(protocol='1 s fixed-action replay; up to 3 starts/trajectory from 0.5 s grid; independently evolved gate conditions',
                   gate='exp(-lambda * RMS population std of six raw member outputs); gate mean before component clipping',
                   aggregation='Window means within trajectories, then equal trajectory weights; paired trajectory bootstrap, 2000 draws, seed 0',
                   ci='Pointwise percentile 95% intervals, not simultaneous; interpret training/validation split from supplied datasets',
                   datasets={k:{key:value for key,value in v.items() if key!='rows'} for k,v in runs.items()},
                   statistics=reports)
    (out/'screening_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    colors = {'zero_delta':'#666666',**{c:plt.get_cmap('viridis')(i/(len(conditions)-2))
                                      for i,c in enumerate(conditions[1:])}}
    names = {c:('Zero delta' if c=='zero_delta' else 'Ungated (λ=0)' if c=='lambda_0'
                else 'λ='+c.removeprefix('lambda_')) for c in conditions}
    fig, axes = plt.subplots(len(runs),len(METRICS),figsize=(20,4*len(runs)),squeeze=False)
    for row,label in enumerate(runs):
        for col,(metric,(title,unit)) in enumerate(METRICS.items()):
            ax=axes[row,col]
            for condition in conditions:
                times,mean,ci=curves[label,condition,metric]
                ax.plot(times,mean,color=colors[condition],label=names[condition],lw=2,
                        linestyle='--' if condition=='zero_delta' else '-')
                ax.fill_between(times,ci[0],ci[1],color=colors[condition],alpha=.08)
            ax.set_title(title);ax.set_xlabel('Elapsed replay time (s)');ax.set_ylabel(f'{label}\n{unit}')
            ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=7,frameon=False)
    fig.tight_layout(rect=(0,.075,1,1))
    for suffix in ('png','pdf'):fig.savefig(out/f'error_growth.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(len(runs),3,figsize=(16,4.5*len(runs)),squeeze=False)
    for row,label in enumerate(runs):
        for col,key in enumerate(('endpoint_body_pos_error_m','endpoint_joint_pos_rmse_rad','gate_mean')):
            ax=axes[row,col]
            records=[next(r for r in reports if r['dataset']==label and r['condition']==c and r['metric']==key)
                     for c in conditions[1:]]
            if key=='gate_mean':
                value=np.array([r['trajectory_mean'] for r in records])
                lo=np.array([r['ci95_low'] for r in records]);hi=np.array([r['ci95_high'] for r in records])
                ax.set_title('Mean gate strength');unit='Fraction of raw ensemble mean'
            else:
                value=np.array([r['baseline_minus_model'] for r in records])
                lo=np.array([r['improvement_ci95_low'] for r in records]);hi=np.array([r['improvement_ci95_high'] for r in records])
                metric=key.removeprefix('endpoint_');title,unit=METRICS[metric]
                ax.set_title('Endpoint '+title);unit=f'Zero − gated error ({unit})\nPositive = improvement'
            ax.errorbar(range(len(value)),value,yerr=np.maximum(0,np.array([value-lo,hi-value])),fmt='o',capsize=4,color='#33789b')
            ax.axhline(0,color='#777777',ls='--',lw=1)
            ax.set_xticks(range(len(value)),[c.removeprefix('lambda_') for c in conditions[1:]])
            ax.set_xlabel('λ (0 = ungated)');ax.set_ylabel(f'{label}\n{unit}');ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(out/f'gate_screen.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(len(runs),2,figsize=(12,4.5*len(runs)),squeeze=False)
    for row,label in enumerate(runs):
        for col,metric in enumerate(('body_pos_error_m','joint_pos_rmse_rad')):
            ax=axes[row,col]
            records=[next(r for r in reports if r['dataset']==label and r['condition']==c
                          and r['metric']=='mean_'+metric) for c in conditions[1:]]
            value=np.array([r['baseline_minus_model'] for r in records])
            lo=np.array([r['improvement_ci95_low'] for r in records])
            hi=np.array([r['improvement_ci95_high'] for r in records])
            ax.errorbar(range(len(value)),value,yerr=np.maximum(0,np.array([value-lo,hi-value])),
                        fmt='o',capsize=4,color='#33789b')
            ax.axhline(0,color='#777777',ls='--',lw=1)
            ax.set_xticks(range(len(value)),[c.removeprefix('lambda_') for c in conditions[1:]])
            title,unit=METRICS[metric]
            ax.set_title('Window mean '+title);ax.set_xlabel('λ (0 = ungated)')
            ax.set_ylabel(f'{label}\nZero − gated error ({unit})\nPositive = improvement');ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(out/f'window_mean_screen.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(f'Plots and statistics: {out}')


if __name__=='__main__':
    main()
