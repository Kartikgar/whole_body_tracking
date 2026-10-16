#!/usr/bin/env python3
"""Closed-loop comparison of Isaac conditions against paired Genesis rollouts.

Expects an experiment directory with one folder per Isaac condition (``play.py --batch_eval`` output:
``metrics.json`` + ``rollout.npz``), a ``genesis`` folder (``eval_sim2sim_genesis.py`` metrics JSON, and for
``states`` a ``--record_motion`` ``states*.npz``) and ``policy.onnx``. Condition folders are named
``zero_delta``, ``ungated``, ``lambda_<x>``, ``single_<tag>`` or any other name (labelled verbatim).

Subcommands:
  metrics  completion/survival, paired durations, tracking-error curves, Genesis reset verification
  states   root/joint/body state evolution, body errors and paired distance to Genesis
"""
import argparse
import csv
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import euler_wxyz, load_genesis_rollout, load_isaac_rollout, quat_angle, use_agg  # noqa: E402

plt = use_agg()

# Validated categorical slots; Genesis always uses slot 5, Isaac conditions take the others in order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#008300", "#6250d6", "#e34948"]
GENESIS_COLOR = "#e87ba4"
INK, MUTED, GRID, SURFACE = "#3d3d3a", "#6b6a63", "#e4e3dd", "#fcfcfb"
T_MAX = None


def label(key):
    """Display name for a condition folder."""
    if key == "genesis":
        return "Genesis"
    if key == "zero_delta":
        return "Isaac zero delta"
    if key == "ungated":
        return "Isaac ungated"
    if key.startswith("lambda_"):
        return f"Isaac λ={key.split('_', 1)[1]}"
    if key.startswith("single_"):
        return f"Isaac single delta {key.split('_', 1)[1].upper()}"
    return f"Isaac {key}"


def condition_colors(keys):
    isaac = [k for k in keys if k != "genesis"]
    if len(isaac) > len(PALETTE):
        raise ValueError(f"At most {len(PALETTE)} Isaac conditions can be colored distinctly; plot fewer at once")
    colors = dict(zip(isaac, PALETTE))
    colors["genesis"] = GENESIS_COLOR
    return colors


def discover_conditions(root):
    """Default order: zero delta, ungated, λ ascending, single deltas, then any other arm."""
    arms = [d.name for d in sorted(root.iterdir()) if d.is_dir() and (d / "rollout.npz").exists()
            and any(d.glob("metrics*.json"))]
    def rank(name):
        if name == "zero_delta":
            return (0, 0, name)
        if name == "ungated":
            return (1, 0, name)
        if name.startswith("lambda_"):
            try:
                return (2, float(name.split("_", 1)[1]), name)
            except ValueError:
                pass
        if name.startswith("single_"):
            return (3, 0, name)
        return (4, 0, name)
    return sorted(arms, key=rank)


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



def run_metrics(args):
    root = args.experiment_dir
    keys_in = args.conditions or discover_conditions(root)
    labels = {k: label(k) for k in keys_in}
    labels['genesis'] = 'Genesis'
    colors = condition_colors(list(labels))
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



def load_reference(onnx_path, joint_names, body_names, frames):
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    jn, bn = meta["joint_names"].split(","), meta["body_names"].split(",")
    ji, bi = [jn.index(n) for n in joint_names], [bn.index(n) for n in body_names]
    obs = np.zeros((1, session.get_inputs()[0].shape[1]), np.float32)
    rows = [session.run(["joint_pos", "body_pos_w", "body_quat_w"], {"obs": obs, "time_step": np.array([[t]], np.float32)})
            for t in range(frames)]
    return dict(joint_pos=np.stack([r[0][0] for r in rows])[:, ji], body_pos=np.stack([r[1][0] for r in rows])[:, bi],
                body_quat=np.stack([r[2][0] for r in rows])[:, bi])


def unwrap_alive(angle, alive):
    out = np.full(angle.shape, np.nan)
    for i in range(angle.shape[0]):
        n = int(alive[i].sum())
        out[i, :n] = np.unwrap(angle[i, :n])
    return out


def masked(values, alive):
    shape = alive.shape + (1,) * (values.ndim - 2)
    return np.where(alive.reshape(shape), values, np.nan)


def band(values, alive, min_alive):
    """Median and IQR over alive rollouts at each time; NaN where fewer than min_alive remain."""
    v = masked(values, alive)
    enough = alive.sum(0) >= min_alive
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        q = np.nanpercentile(v, [25, 50, 75], axis=0)
    q[:, ~enough] = np.nan
    return q


def finish(fig, axes, keys, colors, path, title, reference=True):
    handles = [plt.Line2D([], [], color=colors[k], lw=2.2 if k == "genesis" else 1.4,
                          ls=(0, (1.2, 1.2)) if k == "genesis" else "-", label=label(k)) for k in keys]
    if reference:
        handles.append(plt.Line2D([], [], color=INK, lw=1.0, ls="--", label="Reference"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, y=1.035, fontsize=11)
    fig.subplots_adjust(top=1 - 0.55 / fig.get_figheight())
    for ax in np.ravel(axes):
        if T_MAX is not None:
            ax.set_xlim(0, T_MAX)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    if T_MAX is not None:
        path = path.with_name(f"{path.name}_0-{T_MAX:g}s")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw(ax, t, q, color, key, show_band):
    genesis = key == "genesis"
    if show_band:
        ax.fill_between(t, q[0], q[2], color=color, alpha=0.14, lw=0)
    ax.plot(t, q[1], color=color, lw=2.0 if genesis else 1.2, ls=(0, (1.2, 1.2)) if genesis else "-")



def run_states(args):
    global T_MAX
    T_MAX = args.t_max
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 9,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "font.size": 8})
    root = args.experiment_dir
    out = args.output_dir or root / "analysis_states"
    out.mkdir(parents=True, exist_ok=True)

    data = {k: load_isaac_rollout(root / k / "rollout.npz") for k in args.conditions}
    first = data[args.conditions[0]]
    joint_names, body_names, fps = first["joint_names"], first["body_names"], first["fps"]
    genesis_path = args.genesis_states or next((root / "genesis").glob("states*.npz"))
    data["genesis"] = load_genesis_rollout(genesis_path, joint_names, body_names)
    keys = [*args.conditions, "genesis"]
    colors = condition_colors(keys)
    T = max(d["joint_pos"].shape[1] for d in data.values())
    for d in data.values():  # pad to a common horizon (Genesis has no post-action state at the final frame)
        pad = T - d["joint_pos"].shape[1]
        for name in ("joint_pos", "body_pos", "body_quat"):
            d[name] = np.concatenate([d[name], np.repeat(d[name][:, -1:], pad, 1)], 1) if pad else d[name]
        d["alive"] = np.concatenate([d["alive"], np.zeros((d["alive"].shape[0], pad), bool)], 1) if pad else d["alive"]
    ref = load_reference(args.policy_onnx or root / "policy.onnx", joint_names, body_names, T - 1)
    for name in ref:  # reference frame k for k < T-1; hold the last frame for the final post-action state
        ref[name] = np.concatenate([ref[name], ref[name][-1:]], 0)
    t = np.arange(T) / fps
    root_i = body_names.index("pelvis")

    checks = {}
    for k in keys:
        checks[k] = dict(
            initial_joint_max_abs_diff_vs_first=float(np.abs(data[k]["joint_pos"][:, 0] - first["joint_pos"][:, 0]).max()),
            initial_body_pos_max_abs_diff_vs_first=float(np.abs(data[k]["body_pos"][:, 0] - first["body_pos"][:, 0]).max()))
    if checks["genesis"]["initial_body_pos_max_abs_diff_vs_first"] > 1e-3:
        raise RuntimeError(f"Genesis initial states do not match Isaac after origin removal: {checks['genesis']}")

    derived = {}
    for k, d in data.items():
        roll, pitch, yaw = euler_wxyz(d["body_quat"][:, :, root_i])
        derived[k] = dict(
            root_pos=d["body_pos"][:, :, root_i], roll=unwrap_alive(roll, d["alive"]), pitch=pitch, yaw=unwrap_alive(yaw, d["alive"]),
            root_pos_err=np.linalg.norm(d["body_pos"][:, :, root_i] - ref["body_pos"][None, :, root_i], axis=-1),
            root_ori_err=quat_angle(d["body_quat"][:, :, root_i], ref["body_quat"][None, :, root_i]),
            body_err=np.linalg.norm(d["body_pos"] - ref["body_pos"][None], axis=-1),
            joint_rmse=np.sqrt(np.mean((d["joint_pos"] - ref["joint_pos"][None]) ** 2, axis=-1)))
    r_roll, r_pitch, r_yaw = euler_wxyz(ref["body_quat"][:, root_i])
    r_roll, r_yaw = np.unwrap(r_roll), np.unwrap(r_yaw)

    # Root
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.5), sharex=True)
    panels = [("Pelvis x (m)", lambda k: derived[k]["root_pos"][..., 0], ref["body_pos"][:, root_i, 0]),
              ("Pelvis y (m)", lambda k: derived[k]["root_pos"][..., 1], ref["body_pos"][:, root_i, 1]),
              ("Pelvis height z (m)", lambda k: derived[k]["root_pos"][..., 2], ref["body_pos"][:, root_i, 2]),
              ("Pelvis roll, unwrapped (rad)", lambda k: derived[k]["roll"], r_roll),
              ("Pelvis pitch (rad)", lambda k: derived[k]["pitch"], r_pitch),
              ("Pelvis yaw, unwrapped (rad)", lambda k: derived[k]["yaw"], r_yaw),
              ("Pelvis position error (m)", lambda k: derived[k]["root_pos_err"], None),
              ("Pelvis orientation error (rad)", lambda k: derived[k]["root_ori_err"], None)]
    for ax, (title, fn, reference) in zip(axes.flat, panels):
        for k in keys:
            draw(ax, t, band(fn(k), data[k]["alive"], args.min_alive), colors[k], k, show_band=True)
        if reference is not None:
            ax.plot(t, reference, color=INK, lw=1.0, ls="--")
        ax.set_title(title)
    ax = axes.flat[-1]
    for k in keys:  # Genesis records no post-action state for the final step
        n = T - 1 if k == "genesis" else T
        ax.plot(t[:n], data[k]["alive"].sum(0)[:n], color=colors[k], lw=2.0 if k == "genesis" else 1.2,
                ls=(0, (1.2, 1.2)) if k == "genesis" else "-")
    ax.set_title("Rollouts still alive (of 100)")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    finish(fig, axes, keys, colors, out / "root_states",
           "Root (pelvis) state: median over alive rollouts, shaded 25–75%")

    # Joints
    rows, cols = 6, 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), sharex=True)
    for j, ax in enumerate(axes.flat):
        if j >= len(joint_names):
            ax.axis("off")
            continue
        for k in keys:
            draw(ax, t, band(data[k]["joint_pos"][..., j], data[k]["alive"], args.min_alive), colors[k], k, False)
        ax.plot(t, ref["joint_pos"][:, j], color=INK, lw=0.9, ls="--")
        ax.set_title(joint_names[j].removesuffix("_joint"), fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    finish(fig, axes, keys, colors, out / "joint_states", "Joint positions (rad): median over alive rollouts")

    # Bodies: height and position error
    for name, fn, ref_fn, title in (
            ("body_heights", lambda k, b: data[k]["body_pos"][..., b, 2], lambda b: ref["body_pos"][:, b, 2],
             "Body height z (m): median over alive rollouts"),
            ("body_position_errors", lambda k, b: derived[k]["body_err"][..., b], None,
             "Body world-position error vs reference (m): median over alive rollouts, shaded 25–75%")):
        fig, axes = plt.subplots(4, 4, figsize=(14, 11), sharex=True)
        for b, ax in enumerate(axes.flat):
            if b >= len(body_names):
                ax.axis("off")
                continue
            for k in keys:
                draw(ax, t, band(fn(k, b), data[k]["alive"], args.min_alive), colors[k], k, ref_fn is None)
            if ref_fn is not None:
                ax.plot(t, ref_fn(b), color=INK, lw=0.9, ls="--")
            ax.set_title(body_names[b].removesuffix("_link"), fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Time (s)")
        finish(fig, axes, keys, colors, out / name, title, reference=ref_fn is not None)

    # Paired distance from each Isaac condition to Genesis (same initial state, both alive)
    g = data["genesis"]
    measures = [("Joint-position RMS difference (rad)",
                 lambda d: np.sqrt(np.mean((d["joint_pos"] - g["joint_pos"]) ** 2, -1))),
                ("Pelvis position difference (m)",
                 lambda d: np.linalg.norm(d["body_pos"][:, :, root_i] - g["body_pos"][:, :, root_i], axis=-1)),
                ("Pelvis height difference, Isaac − Genesis (m)",
                 lambda d: d["body_pos"][:, :, root_i, 2] - g["body_pos"][:, :, root_i, 2]),
                ("Mean body position difference (m)",
                 lambda d: np.linalg.norm(d["body_pos"] - g["body_pos"], axis=-1).mean(-1))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    paired_summary = {}
    for ax, (title, fn) in zip(axes.flat, measures):
        for k in args.conditions:
            both = data[k]["alive"] & g["alive"]
            draw(ax, t, band(fn(data[k]), both, args.min_alive), colors[k], k, True)
            values = masked(fn(data[k]), both)
            for window in (1.0, 2.0, 5.0, 20.0):
                n = int(window * fps) + 1
                with np.errstate(all="ignore"):
                    paired_summary.setdefault(k, {}).setdefault(title, {})[f"mean_0_to_{window:g}s"] = float(
                        np.nanmean(values[:, :n]))
        if "height" in title:
            ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    finish(fig, axes, args.conditions, colors, out / "paired_distance_to_genesis",
           "Paired distance to Genesis from the same initial state (both rollouts alive): median, shaded 25–75%",
           reference=False)

    summary = dict(
        conditions=keys, fps=fps, min_alive=args.min_alive, genesis_states=str(genesis_path),
        time_alignment="index k = state at k/fps vs reference frame k; Isaac post-action arrays shifted by one, "
                       "prefixed with the initial state; Genesis pre-action recording used directly",
        initial_state_checks=checks,
        completion={k: int(data[k]["alive"][:, -1].sum()) if k != "genesis" else int(data[k]["alive"][:, T - 2].sum())
                    for k in keys},
        median_failure_time_s={k: float(np.median(data[k]["alive"].sum(1) / fps)) for k in keys},
        paired_distance_to_genesis=paired_summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("initial_state_checks", "completion", "median_failure_time_s")}, indent=1))



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("metrics", help="Completion, survival and tracking-error curves")
    p.add_argument("--experiment_dir", type=Path, required=True)
    p.add_argument("--conditions", nargs="+", help="Isaac condition folders in plot order (default: discovered)")
    p.add_argument("--genesis_dir", type=Path)
    p.add_argument("--output_dir", type=Path)
    p.set_defaults(handler=run_metrics)
    p = sub.add_parser("states", help="State evolution and paired distance to Genesis")
    p.add_argument("--experiment_dir", type=Path, required=True)
    p.add_argument("--conditions", nargs="+", help="Isaac condition folders in plot order (default: discovered)")
    p.add_argument("--genesis_states", type=Path, help="Defaults to the single states*.npz under <experiment>/genesis.")
    p.add_argument("--policy_onnx", type=Path, help="Defaults to <experiment>/policy.onnx.")
    p.add_argument("--output_dir", type=Path, help="Defaults to <experiment>/analysis_states.")
    p.add_argument("--min_alive", type=int, default=5, help="Hide statistics when fewer rollouts remain alive.")
    p.add_argument("--t_max", type=float, help="Limit the time axis; figures get a _0-<t_max>s suffix.")
    p.set_defaults(handler=run_states)
    args = parser.parse_args()
    if args.handler is run_states and not args.conditions:
        args.conditions = discover_conditions(args.experiment_dir)
    args.handler(args)


if __name__ == "__main__":
    main()
