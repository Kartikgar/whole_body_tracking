#!/usr/bin/env python3
"""Compare interval error curves from runs sharing the same replay protocol."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

METRICS = {'body_pos_error_m': ('Body position error', 'm'),
           'joint_pos_rmse_rad': ('Joint position RMSE', 'rad'),
           'root_height_error_m': ('Root height error', 'm'),
           'root_pos_error_m': ('Root position error', 'm'),
           'root_orientation_error_rad': ('Root orientation error', 'rad'),
           'body_lin_vel_rmse_m_s': ('Body linear velocity RMSE', 'm/s')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',required=True,help='LABEL=run directory; repeat')
    p.add_argument('--reference',required=True,help='Reference label for paired error differences')
    p.add_argument('--pair', action='append', default=[], help='Additional MODEL=REFERENCE paired comparison')
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--metrics', nargs='+', choices=list(METRICS),
                   default=['body_pos_error_m', 'joint_pos_rmse_rad', 'root_height_error_m'])
    args=p.parse_args()
    metrics_to_plot={key:METRICS[key] for key in args.metrics}
    runs={}
    for item in args.run:
        label, sep, path=item.partition('=')
        if not sep or not label or label in runs:p.error('Each run needs a unique LABEL=directory')
        runs[label]=Path(path)
    if args.reference not in runs or len(runs)<2:p.error('Need reference and at least two runs')
    manifests={label:json.loads((path/'manifest.json').read_text()) for label,path in runs.items()}
    summaries={label:json.loads((path/'summary.json').read_text()) for label,path in runs.items()}
    schedules={label:json.loads((path/'windows.json').read_text()) for label,path in runs.items()}
    first=next(iter(runs))
    for label in runs:
        if manifests[label]['status']!='complete':p.error(f'Incomplete run: {label}')
        for key in ('dataset_sha256','seed','num_envs','replay_steps','measurement_interval_s'):
            if manifests[label][key]!=manifests[first][key]:p.error(f'{label} differs in {key}')
        if schedules[label]!=schedules[first]:p.error(f'{label} has a different schedule')
        if summaries[label]['configuration_contract']!=summaries[first]['configuration_contract']:
            p.error(f'{label} has different physical/observation/action settings')
    rows={label:json.loads((path/'partial_results.json').read_text()) for label,path in runs.items()}
    if any(r['status']!='complete' for group in rows.values() for r in group):
        p.error('Failed windows require an explicit censoring protocol')
    trajectories=sorted({r['trajectory'] for r in rows[first]})
    times=sorted({point['requested_time_s'] for r in rows[first] for point in r['error_curve']})
    matrices={}
    for label,group in rows.items():
        grouped=defaultdict(list)
        for r in group:
            for point in r['error_curve']:grouped[r['trajectory'],point['requested_time_s']].append(point)
        if len(grouped)!=len(trajectories)*len(times):p.error(f'{label} missing sampled times')
        matrices[label]={key:np.array([[np.mean([point[key] for point in grouped[traj,time]]) for time in times]
                                       for traj in trajectories]) for key in metrics_to_plot}
    rng=np.random.default_rng(0);indices=rng.integers(len(trajectories),size=(2000,len(trajectories)))
    stats={};paired=[]
    for label,metrics in matrices.items():
        stats[label]={}
        for key,values in metrics.items():
            if not np.isfinite(values).all():p.error('Nonfinite error measurements')
            low,high=np.percentile(values[indices].mean(axis=1),[2.5,97.5],axis=0)
            stats[label][key]=dict(mean=values.mean(axis=0),low=low,high=high)
            if label!=args.reference:
                diff=matrices[args.reference][key]-values
                lo,hi=np.percentile(diff[indices].mean(axis=1),[2.5,97.5],axis=0)
                for i,t in enumerate(times):
                    paired.append(dict(model=label,reference=args.reference,metric=key,time_s=t,
                                       reference_minus_model=float(diff.mean(axis=0)[i]),
                                       ci95_low=float(lo[i]),ci95_high=float(hi[i])))
    for pair in args.pair:
        label, sep, reference=pair.partition('=')
        if not sep or label not in runs or reference not in runs or label==reference:
            p.error('Additional pairs must be distinct known MODEL=REFERENCE labels')
        if reference==args.reference:
            continue
        for key,values in matrices[label].items():
            diff=matrices[reference][key]-values
            lo,hi=np.percentile(diff[indices].mean(axis=1),[2.5,97.5],axis=0)
            for i,t in enumerate(times):
                paired.append(dict(model=label,reference=reference,metric=key,time_s=t,
                                   reference_minus_model=float(diff.mean(axis=0)[i]),
                                   ci95_low=float(lo[i]),ci95_high=float(hi[i])))
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    with (out/'paired_error_differences.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    report=dict(windows=len(rows[first]),trajectories=len(trajectories),runs={k:str(v.resolve()) for k,v in runs.items()},
                method='Window averages within trajectory; equal trajectory weights; paired bootstrap 2000 samples, seed 0; pointwise 95% intervals',
                final_metrics={label:{key:{k:float(v[-1]) for k,v in s.items()} for key,s in metrics.items()} for label,metrics in stats.items()},
                final_paired_differences=[r for r in paired if r['time_s']==times[-1]])
    (out/'comparison.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':17,'axes.titlesize':20,'axes.labelsize':17,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    columns=min(3,len(metrics_to_plot));nrows=(len(metrics_to_plot)+columns-1)//columns
    fig,axes=plt.subplots(nrows,columns,figsize=(17,5*nrows+1),squeeze=False)
    colors=['#777777','#237898','#d48722','#7b2cbf']
    for ax,(key,(title,unit)) in zip(axes.flat,metrics_to_plot.items()):
        for i,label in enumerate(runs):
            s=stats[label][key];color=colors[i%len(colors)]
            ax.plot(times,s['mean'],color=color,lw=2.5,label=label)
            ax.fill_between(times,s['low'],s['high'],color=color,alpha=.13)
        ax.set_title(title);ax.set_xlabel('Elapsed replay time (s)');ax.set_ylabel(unit)
        ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    for ax in list(axes.flat)[len(metrics_to_plot):]:ax.set_visible(False)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=len(runs),frameon=False)
    fig.tight_layout(rect=(0,.07 if nrows>1 else .14,1,1))
    fig.savefig(out/'error_growth_comparison.png',dpi=300,bbox_inches='tight')
    fig.savefig(out/'error_growth_comparison.pdf',bbox_inches='tight');plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
