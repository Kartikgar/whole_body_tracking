#!/usr/bin/env python3
"""Plot shared-input ensemble disagreement and compare matched zero-delta errors."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

KEYS = {
    'normalized_disagreement': ('Raw output disagreement', 'Normalized units'),
    'force_disagreement_n': ('Force disagreement after clipping', 'N (component RMS std)'),
    'torque_disagreement_nm': ('Torque disagreement after clipping', 'N m (component RMS std)'),
    'saturation_fraction': ('Member output saturation', 'Fraction of member/components'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--baseline_dir', type=Path, required=True)
    parser.add_argument('--title', default='Ensemble disagreement', help='Main figure title')
    args = parser.parse_args()
    run, baseline = args.run_dir, args.baseline_dir
    manifests = [json.loads((p/'manifest.json').read_text()) for p in (run, baseline)]
    for key in ('dataset_sha256', 'seed', 'num_envs', 'replay_steps', 'measurement_interval_s'):
        if manifests[0][key] != manifests[1][key]:
            parser.error(f'Comparison mismatch: {key}')
    if json.loads((run/'windows.json').read_text()) != json.loads((baseline/'windows.json').read_text()):
        parser.error('Window schedules differ')
    summaries = [json.loads((p/'summary.json').read_text()) for p in (run, baseline)]
    if summaries[0]['configuration_contract'] != summaries[1]['configuration_contract']:
        parser.error('Physics/observations/action contracts differ')
    rows = json.loads((run/'partial_results.json').read_text())
    zero = json.loads((baseline/'partial_results.json').read_text())
    if any(r['status'] != 'complete' for r in rows + zero):
        parser.error('Incomplete windows require explicit censoring analysis before comparison')
    trajectories = sorted({r['trajectory'] for r in rows})
    points = [dict(trajectory=r['trajectory'], start=r['start'], **p)
              for r in rows for p in r['disagreement_curve']]
    times = sorted({p['requested_time_s'] for p in points})
    out = run/'analysis';out.mkdir(exist_ok=True)
    with (out/'disagreement_samples.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    def matrix(records, field, key):
        grouped=defaultdict(list)
        for r in records:
            for p in r[field]:grouped[r['trajectory'],p['requested_time_s']].append(p[key])
        return np.array([[np.mean(grouped[t,time]) if grouped[t,time] else np.nan for time in times] for t in trajectories])
    rng=np.random.default_rng(0);indices=rng.integers(len(trajectories),size=(2000,len(trajectories)))
    metrics={key:matrix(rows,'disagreement_curve',key) for key in (*KEYS,'clipped_normalized_disagreement')}
    stats={}
    for key, values in metrics.items():
        if not np.isfinite(values).all():parser.error('Missing/nonfinite disagreement samples')
        ci=np.percentile(values[indices].mean(axis=1),[2.5,97.5],axis=0)
        stats[key]=dict(mean=values.mean(axis=0),p90=np.percentile(values,90,axis=0),low=ci[0],high=ci[1])
    aggregates=[dict(metric=key,requested_time_s=t,mean=s['mean'][i],p90=s['p90'][i],ci95_low=s['low'][i],ci95_high=s['high'][i])
                for key,s in stats.items() for i,t in enumerate(times)]
    with (out/'disagreement_growth.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(aggregates[0]));writer.writeheader();writer.writerows(aggregates)
    from scipy.stats import spearmanr
    associations=[]
    uncertainty={(r['trajectory'],r['start']):{p['requested_time_s']:p['normalized_disagreement'] for p in r['disagreement_curve']} for r in rows}
    for time in times[1:]:
        u=[]; e=[]
        for r in rows:
            point=next(p for p in r['error_curve'] if p['requested_time_s']==time)
            u.append(uncertainty[r['trajectory'],r['start']][time]);e.append(point['body_pos_error_m'])
        associations.append(dict(time_s=time,spearman=float(spearmanr(u,e).statistic)))
    report=dict(members=len(manifests[0]['ensemble_members']),windows=len(rows),trajectories=len(trajectories),
                protocol=manifests[0]['ensemble_protocol'],
                definition='Population std across members per output component; RMS across six raw components for normalized uncertainty; RMS across three clipped force/torque components for physical disagreement',
                aggregation='Average windows within trajectory, then equal trajectory weighting; 2000 trajectory bootstrap resamples',
                final={key:{k:float(v[-1]) for k,v in s.items()} for key,s in stats.items()},
                same_time_error_correlations=associations,
                correlation_note='Descriptive within-horizon window correlations; overlapping windows are dependent; no significance claim')
    (out/'ensemble_summary.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':15,'axes.titlesize':18,'axes.labelsize':15,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
    for ax,(key,(title,unit)) in zip(axes.flat,KEYS.items()):
        s=stats[key];ax.plot(times,s['mean'],color='#237898',lw=2.5,label='Trajectory mean')
        ax.fill_between(times,s['low'],s['high'],color='#237898',alpha=.18,label='95% bootstrap CI')
        ax.plot(times,s['p90'],color='#d48722',lw=2,ls='--',label='Trajectory p90')
        ax.set_title(title);ax.set_ylabel(unit);ax.set_xlabel('Elapsed replay time (s)');ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.suptitle(args.title,fontsize=23)
    fig.tight_layout(rect=(0,.06,1,.95));save(fig,'ensemble_disagreement')
    fig,axes=plt.subplots(1,3,figsize=(16,5.5))
    for ax,key,title,unit in zip(axes,('body_pos_error_m','joint_pos_rmse_rad','root_height_error_m'),
                                ('Body position error','Joint position RMSE','Root height error'),('m','rad','m')):
        for records,label,color in ((zero,'Zero delta','#777777'),(rows,'Ungated ensemble mean','#237898')):
            values=matrix(records,'error_curve',key);mean=values.mean(axis=0)
            ci=np.percentile(values[indices].mean(axis=1),[2.5,97.5],axis=0)
            ax.plot(times,mean,color=color,lw=2.5,label=label);ax.fill_between(times,ci[0],ci[1],color=color,alpha=.15)
        ax.set_title(title);ax.set_xlabel('Elapsed replay time (s)');ax.set_ylabel(unit);ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.13,1,1));save(fig,'ensemble_vs_zero_error_growth')
    fig,ax=plt.subplots(figsize=(10,7))
    vmax=np.percentile(metrics['normalized_disagreement'],95)
    image=ax.imshow(metrics['normalized_disagreement'],aspect='auto',origin='lower',cmap='magma',vmin=0,vmax=vmax,
                    extent=(-.125,max(times)+.125,-.5,len(trajectories)-.5))
    ax.set_yticks(np.arange(0,len(trajectories),5),[str(t) for t in trajectories[::5]])
    ax.set_xlabel('Elapsed replay time (s)');ax.set_ylabel('Validation trajectory ID')
    ax.set_title('Raw disagreement by trajectory')
    fig.colorbar(image,ax=ax,label='Normalized disagreement (color clipped at p95)')
    fig.tight_layout();save(fig,'trajectory_disagreement')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
