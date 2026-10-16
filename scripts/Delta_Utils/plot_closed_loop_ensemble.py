#!/usr/bin/env python3
"""Compare frozen base-policy Isaac ensemble conditions with paired Genesis rollouts."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


def enrich_root_errors(rows, archive, reference, origins):
    """Recorded Isaac states are post-action; sample step s pairs with reference s-1."""
    root=list(map(str,archive['body_names'])).index('pelvis')
    positions=archive['body_pos_w'][:,:,root]-origins[:,None,:]
    quats=archive['body_quat_w'][:,:,root].astype(np.float64)
    ref_positions=reference['body_pos_w'][:,0]
    ref_quats=reference['body_quat_w'][:,0].astype(np.float64)
    quats/=np.linalg.norm(quats,axis=-1,keepdims=True)
    ref_quats/=np.linalg.norm(ref_quats,axis=-1,keepdims=True)
    for row in rows:
        for point in row['diagnostic_curve']:
            i=row['env_id'];frame=point['step']-1
            point.setdefault('root_pos_error_m',float(np.linalg.norm(positions[i,frame]-ref_positions[frame])))
            dot=np.clip(abs(np.dot(quats[i,frame],ref_quats[frame])),0,1)
            point.setdefault('root_orientation_error_rad',float(2*np.arccos(dot)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment_dir',type=Path,required=True)
    p.add_argument('--exclude_ungated',action='store_true')
    p.add_argument('--lambdas',type=int,nargs='+',default=[2,4,6,8],
                   help='Ensemble gate coefficients to include, in plot order.')
    p.add_argument('--genesis_dir',type=Path)
    p.add_argument('--output_dir',type=Path)
    args=p.parse_args();root=args.experiment_dir
    labels={'zero_delta':'Isaac zero delta'}
    if not args.exclude_ungated:labels['ungated']='Isaac ungated'
    labels.update({f'lambda_{lam}':f'Isaac λ={lam}' for lam in args.lambdas if (root/f'lambda_{lam}').exists()})
    labels['genesis']='Genesis'
    colors={'zero_delta':'#777777','ungated':'#cc7722','lambda_2':'#237baf','lambda_4':'#3a9860',
            'lambda_6':'#dc8a23','lambda_8':'#c34f71','lambda_10':'#008c95',
            'lambda_12':'#a57500','lambda_14':'#315a97','genesis':'#7b2cbf'}
    for i,lam in enumerate(args.lambdas):
        colors.setdefault(f'lambda_{lam}',f'C{i % 10}')
    results={};initial=None;summary={}
    for key in labels:
        path=args.genesis_dir if key=='genesis' and args.genesis_dir else root/key
        file=next(path.glob('metrics*.json'))
        data=json.loads(file.read_text())
        if key=='genesis':
            rows=[dict(r,steps=r['survived_steps'],terminated=not r['completed']) for r in data['rollout_metrics']]
        else:
            rows=data['rollouts']
            with np.load(path/'rollout.npz',allow_pickle=True) as d:
                current={name:d[name] for name in d.files if name.startswith('initial_')}
                if initial is None:
                    initial=current
                    joint_names=list(map(str,d['joint_names']))
                elif any(not np.allclose(initial[name],current[name],atol=1e-6,rtol=0) for name in initial):
                    raise ValueError(f'Isaac initial states differ: {key}')
                if any('root_pos_error_m' not in point or 'root_orientation_error_rad' not in point
                       for row in rows for point in row['diagnostic_curve']):
                    origins=d['env_origins'] if 'env_origins' in d else np.load(root/'isaac_env_origins.npy')
                    with np.load(data['motion_file']) as reference:
                        enrich_root_errors(rows,d,reference,origins)
        if len(rows)!=100 or any(r['env_id']!=i for i,r in enumerate(rows)):
            raise ValueError('Expected 100 paired rollouts ordered by env_id')
        results[key]=rows
        summary[key]=dict(completion_pct=100*np.mean([not r['terminated'] for r in rows]),
                          survival_pct=100*np.mean([r['steps']/1000 for r in rows]),
                          observed_duration_mean_s=float(np.mean([r['steps']*.02 for r in rows])),
                          failure_reasons=dict(Counter(r['termination_reason'] for r in rows)),
                          result=str(file.resolve()))
        scalar_names=('mpjpe','mpjpe_l','vel_dist','accel_dist','error_anchor_pos','error_anchor_rot')
        summary[key]['tracking_means'] = {
            name: float(data[name+'_mean']) if key=='genesis' else float(np.mean([r['metrics'][name] for r in rows]))
            for name in scalar_names
        }
        if 'termination_definition' in data:
            summary[key]['termination_definition']=data['termination_definition']
        if key=='genesis':
            # Genesis state arrays are in exported joint order; Isaac archives expose the same joint order.
            # Avoid loading a deployment session: ONNX metadata contains the joint order.
            import onnx
            policy_path=Path(data['policy_path'])
            if not policy_path.exists():policy_path=root/'policy.onnx'
            meta={m.key:m.value for m in onnx.load(policy_path).metadata_props}
            def names(value):
                try:return json.loads(value)
                except json.JSONDecodeError:return value.split(',')
            genesis_names=names(meta['joint_names'])
            order=[genesis_names.index(n) for n in joint_names]
            reset_q=np.array([s['joint_pos'] for s in data['initial_states']])[:,order]
            max_q_error=float(np.max(np.abs(reset_q-initial['initial_joint_pos'])))
            initial_path=Path(data['initial_state_npz'])
            if not initial_path.exists():initial_path=root/'paired_initial_states.npz'
            with np.load(initial_path,allow_pickle=True) as logged:
                expected_root=logged['initial_body_pos_w'][:,0].copy()
                if 'env_origins' in logged:expected_root-=logged['env_origins']
                reset_velocity_errors={key:float(np.max(np.abs(np.array([s[key][0] for s in data['initial_states']])-logged['initial_'+key][:,0])))
                                       for key in ('body_lin_vel_w','body_ang_vel_w')}
            max_root_error=float(np.max(np.abs(np.array([s['body_pos_w'][0] for s in data['initial_states']])-expected_root)))
            max_qvel_error=float(np.max(np.abs(np.array([s['joint_vel'] for s in data['initial_states']])[:,order]-initial['initial_joint_vel'])))
            if max_q_error>1e-4 or max_root_error>1e-4 or max_qvel_error>1e-4 or max(reset_velocity_errors.values())>1e-4:
                raise ValueError(f'Genesis reset differs: q {max_q_error}, root {max_root_error}')
            summary['initial_state_verification']=dict(max_joint_position_error_rad=max_q_error,max_joint_velocity_error_rad_s=max_qvel_error,
                                                       max_root_position_error_m=max_root_error,root_velocity_errors=reset_velocity_errors)
    genesis_steps=np.array([r['steps'] for r in results['genesis']])
    for key in labels:
        steps=np.array([r['steps'] for r in results[key]])
        summary[key]['paired_duration_mae_vs_genesis_s']=float(np.abs(steps-genesis_steps).mean()*.02)
        summary[key]['within_1s_of_genesis_fraction']=float(np.mean(np.abs(steps-genesis_steps)<=50))
        if key!='genesis':
            failed=[r for r in results[key] if r['terminated'] and r['diagnostic_curve']]
            summary[key]['failure_diagnostics_median']={name:float(np.median([r['diagnostic_curve'][-1][name] for r in failed]))
                                                        for name in ('uncertainty','gate','force_n','torque_nm','member_saturation')
                                                        if all(name in r['diagnostic_curve'][-1] for r in failed)} if failed else {}
    snapshot={}
    for key,rows in results.items():
        points=[p for r in rows for p in r['diagnostic_curve'] if p['step']==125]
        snapshot_metrics=['mpjpe_m','joint_rmse_rad']
        if all('root_pos_error_m' in p and 'root_orientation_error_rad' in p for p in points):
            snapshot_metrics+=['root_pos_error_m','root_orientation_error_rad']
        snapshot[key]=dict(count=len(points), **{metric:float(np.mean([p[metric] for p in points]))
                                                for metric in snapshot_metrics})
        if key!='genesis':
            snapshot[key].update({metric:float(np.mean([p[metric] for p in points]))
                                  for metric in ('uncertainty','gate','force_n')
                                  if points and all(metric in p for p in points)})
    summary['matched_tracking_at_2p5s']=snapshot
    out=args.output_dir or root/'analysis';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'comparison.csv').open('w',newline='') as handle:
        fields=['condition','completion_pct','survival_pct','mpjpe_mm','mpjpe_l_mm',
                'root_position_error_m','root_orientation_error_rad','velocity_distance','acceleration_distance']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for key in labels:
            m=summary[key]['tracking_means']
            writer.writerow(dict(condition=key,completion_pct=summary[key]['completion_pct'],survival_pct=summary[key]['survival_pct'],
                                 mpjpe_mm=m['mpjpe'],mpjpe_l_mm=m['mpjpe_l'],root_position_error_m=m['error_anchor_pos'],
                                 root_orientation_error_rad=m['error_anchor_rot'],velocity_distance=m['vel_dist'],acceleration_distance=m['accel_dist']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':14,'axes.titlesize':17,'axes.labelsize':15,'pdf.fonttype':42})
    keys=list(labels)
    has_ensemble_diagnostics=any('gate' in p for key in keys if key!='genesis'
                                 for row in results[key] for p in row['diagnostic_curve'])
    fig,axes=plt.subplots(2,2,figsize=(15,11))
    x=np.arange(len(keys))
    for ax,metric,title in [(axes[0,0],'completion_pct','Full-clip completion'),(axes[0,1],'survival_pct','Mean clip survival')]:
        values=[summary[k][metric] for k in keys]
        ax.bar(x,values,color=[colors[k] for k in keys],width=.65)
        ax.set_xticks(x,[labels[k].replace(' ','\n',1) for k in keys],fontsize=12)
        ax.set_ylim(0,112);ax.set_ylabel('%');ax.set_title(title)
        for i,v in enumerate(values):ax.text(i,v+2,f'{v:.1f}%',ha='center',fontsize=12)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax=axes[1,0];times=np.arange(0,1001)*.02
    for k in keys:
        rows=results[k]
        live=np.array([np.mean([not r['terminated'] or r['steps']*.02>t for r in rows]) for t in times])*100
        ax.plot(times,live,color=colors[k],label=labels[k],lw=2)
    ax.set_title('Rollouts without failure');ax.set_xlabel('Clip time (s)');ax.set_ylabel('%');ax.grid(alpha=.2)
    ax.legend(fontsize=11,frameon=False)
    ax=axes[1,1]
    for k in keys[:-1]:
        ax.scatter(genesis_steps*.02,np.array([r['steps'] for r in results[k]])*.02,
                   label=labels[k],color=colors[k],alpha=.45,s=23)
    ax.plot([0,20],[0,20],'k--',lw=1);ax.set_xlim(0,20.5);ax.set_ylim(0,20.5)
    ax.set_title('Paired observed durations');ax.set_xlabel('Genesis duration (s)');ax.set_ylabel('Isaac duration (s)');ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(out/f'closed_loop_performance.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    # A common at-risk subset prevents comparing different survivor populations.
    step_sets={k:{r['env_id']:{p['step']:p for p in r['diagnostic_curve']} for r in rows} for k,rows in results.items()}
    sample_steps=sorted({p['step'] for r in results['zero_delta'] for p in r['diagnostic_curve']})
    curves={k:{m:[] for m in ('mpjpe_m','joint_rmse_rad')} for k in keys};valid_times=[];counts=[]
    for step in sample_steps:
        common=[i for i in range(100) if all(step in step_sets[k][i] for k in keys)]
        if not common:continue
        valid_times.append(step*.02);counts.append(len(common))
        for k in keys:
            for metric in curves[k]:curves[k][metric].append(np.mean([step_sets[k][i][step][metric] for i in common]))
    fig,axes=plt.subplots(1,3,figsize=(17,4.8))
    for ax,metric,title,unit in [(axes[0],'mpjpe_m','World body-position error','m'),(axes[1],'joint_rmse_rad','Joint-position RMSE','rad')]:
        for k in keys:ax.plot(valid_times,curves[k][metric],color=colors[k],label=labels[k],lw=2)
        ax.set_title(title);ax.set_xlabel('Clip time (s)');ax.set_ylabel(unit);ax.grid(alpha=.2)
    axes[2].plot(valid_times,counts,color='black');axes[2].set_title('Common at-risk rollouts')
    axes[2].set_xlabel('Clip time (s)');axes[2].set_ylabel('Count');axes[2].grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=min(3,len(keys)),frameon=False,fontsize=12)
    fig.tight_layout(rect=(0,.12,1,1))
    for suffix in ('png','pdf'):fig.savefig(out/f'paired_pre_failure_tracking.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    (out/'common_at_risk_counts.json').write_text(json.dumps(dict(times_s=valid_times,counts=counts),indent=2)+'\n')
    # Each condition now keeps its own observed population through the full horizon.
    # Use regular sample times plus its final recorded failure/completion, rather
    # than mixing asynchronous individual failure samples into the mean curve.
    full_rows=[]
    root_metrics=all('root_pos_error_m' in p and 'root_orientation_error_rad' in p
                     for rows in results.values() for row in rows for p in row['diagnostic_curve'])
    metrics=['mpjpe_m','joint_rmse_rad']+(['root_pos_error_m','root_orientation_error_rad'] if root_metrics else [])
    if root_metrics and not has_ensemble_diagnostics:
        fig=plt.figure(figsize=(17,9))
        grid=fig.add_gridspec(2,3)
        axes=[fig.add_subplot(grid[0,i]) for i in range(3)]
        axes.extend([fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1:])])
    else:
        fig,axes=plt.subplots(2,3,figsize=(17,9)) if root_metrics else plt.subplots(1,3,figsize=(17,4.8))
        axes=axes.ravel()
    for k in keys:
        final_step=max(r['steps'] for r in results[k])
        steps=sorted({s for s in sample_steps if s<=final_step} | {final_step})
        points=[]
        for step in steps:
            observed=[step_sets[k][i][step] for i in range(100) if step in step_sets[k][i]]
            if not observed:continue
            point=dict(condition=k,step=step,elapsed_s=step*.02,count=len(observed),
                       **{metric:float(np.mean([p[metric] for p in observed]))
                          for metric in metrics})
            points.append(point);full_rows.append(point)
        for ax,metric in zip(axes,[*metrics,'count']):
            ax.plot([p['elapsed_s'] for p in points],[p[metric] for p in points],
                    color=colors[k],label=labels[k],lw=2,
                    linestyle='--' if k=='zero_delta' else '-')
    titles=['World body-position error','Joint-position RMSE']+(['Root-position error','Root-orientation error'] if root_metrics else [])+['Contributing rollouts']
    units=['m','rad']+(['m','rad'] if root_metrics else [])+['Count']
    for ax,title,unit in zip(axes,titles,units):
        ax.set_title(title);ax.set_xlabel('Clip time (s)');ax.set_ylabel(unit)
        ax.set_xlim(0,20);ax.grid(alpha=.2)
    axes[len(metrics)].set_ylim(0,105)
    if root_metrics and has_ensemble_diagnostics:
        ax=axes[-1]
        for k in keys[:-1]:
            ts=[];values=[]
            for step in sample_steps:
                observed=[step_sets[k][i][step]['gate'] for i in range(100)
                          if step in step_sets[k][i] and 'gate' in step_sets[k][i][step]]
                if observed:ts.append(step*.02);values.append(np.mean(observed))
            ax.plot(ts,values,color=colors[k],lw=2,ls='--' if k=='zero_delta' else '-')
        ax.set_title('Mean applied gate');ax.set_xlabel('Clip time (s)');ax.set_ylabel('Fraction')
        ax.set_xlim(0,20);ax.set_ylim(0,1.05);ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=min(3,len(keys)),frameon=False,fontsize=12)
    fig.tight_layout(rect=(0,.12,1,1))
    for suffix in ('png','pdf'):fig.savefig(out/f'full_horizon_tracking.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    (out/'full_horizon_tracking.json').write_text(json.dumps(dict(
        method='Per-condition mean over observed rollouts at each sampled time; individual contributions stop at first failure, inclusive; no imputation after failure. Different curves can have different contributor sets.',
        rows=full_rows),indent=2)+'\n')
    if has_ensemble_diagnostics:
        fig,axes=plt.subplots(1,3,figsize=(17,4.8))
        for ax,metric,title in zip(axes,('uncertainty','gate','force_n'),('Survivor mean disagreement','Survivor mean applied gate','Survivor mean force norm (N)')):
            for k in keys[:-1]:
                means=[];ts=[]
                for step in sample_steps:
                    values=[step_sets[k][i][step][metric] for i in range(100) if step in step_sets[k][i] and metric in step_sets[k][i][step]]
                    if values:ts.append(step*.02);means.append(np.mean(values))
                ax.plot(ts,means,color=colors[k],label=labels[k],lw=2)
            ax.set_title(title);ax.set_xlabel('Clip time (s)');ax.grid(alpha=.2)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,fontsize=12)
        fig.tight_layout(rect=(0,.12,1,1))
        for suffix in ('png','pdf'):fig.savefig(out/f'ensemble_diagnostics.{suffix}',dpi=220,bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
