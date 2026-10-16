# Genesis simulation and trajectory recording

[Documentation index](README.md) · [Dataset contracts](datasets.md)

Run these commands in the Genesis environment from `whole_body_tracking/`. Export the base tracking ONNX using Isaac playback or [segment preparation](segment_artifacts.md).

## Evaluate a base policy

```bash
python scripts/eval_sim2sim_genesis.py \
  --policy_path path/to/policy.onnx --num_envs 100 \
  --no-domain_randomization --no-add_noise --seed 0 \
  --compute_metrics --metric_num_envs 100
```

The BeyondMimic policy reads its reference and controller metadata from ONNX. Explicitly set `--metric_num_envs` when all parallel environments should contribute tracking metrics; its default is one. `--record_video` captures execution, and `--viewer --show_reference` supports interactive inspection.

The modular Genesis evaluator now uses the same failure checks as Isaac's
`checkpoint_batch_eval.py`: nonfinite robot state, absolute pelvis-height error
greater than 0.25 m, absolute projected-gravity z-component error greater than
0.8, or absolute world-height error greater than 0.25 m for any ankle-roll or
wrist-yaw link. The saved `termination_definition` records this protocol.
Historical Genesis results used a 0.5 m pelvis-height threshold and no endpoint
height checks; re-evaluate older policies before directly comparing rates.
`full_clip_completion_pct` requires no failure through the entire horizon;
`clip_survival_pct` averages the observed fraction of the horizon across scored
rollouts, including the failing step. The older `success` field allows a five-step
tolerance and is distinct from strict full-clip completion.

## Record target state/action trajectories

```bash
python scripts/eval_sim2sim_genesis.py \
  --policy_path path/to/policy.onnx --num_envs 100 \
  --no-domain_randomization --add_noise --randomize_startup_qpos \
  --record_motion --target_trajectories 100 --seed 0 \
  --output_motion_npz path/to/recorded_dataset.npz
```

Choose noise, startup randomization and physical overrides to match the experiment. The example records varied initial states with observation noise; the preceding evaluation command disables it. The CLI seed default is `None`; set a seed explicitly for reproducibility. Preserve trajectories' valid lengths and action metadata, including early terminations.

Use [physical-property configs](genesis_experiments.md) for mass, passive stiffness and friction changes. Record the resolved overrides with data. Model/reference sampling and numerical timestep choices are part of the experiment; current CLI defaults use 0.02-second control steps and 0.001-second physics steps.

## SONIC

```bash
python scripts/sonic/setup_sonic.py --model_dir artifacts/sonic/default
python scripts/eval_sim2sim_genesis.py \
  --policy_type sonic --sonic_model_dir artifacts/sonic/default \
  --motion_file path/to/reference.npz --num_envs 1 \
  --no-domain_randomization --no-add_noise --seed 0 --compute_metrics
```

SONIC's reference is supplied as NPZ, with `--trajectory_index` for stacked data. It requires compatible 29-DoF G1 motion fields and resamples to 50 Hz. Setup downloads model/config/assets/licenses into the local artifact directory.

## What this evaluation measures

Closed-loop Genesis evaluation recomputes base policy joint actions from the evolving target state. In contrast, [delta replay](delta_replay_evaluation.md) fixes the recorded joint-action sequence and recomputes only the source correction. Compare these outcomes according to their different questions.

The separate `scripts/rsl_rl/evaluate_sim2sim_genesis.py` is a legacy monolithic evaluator. Use the current modular `scripts/eval_sim2sim_genesis.py` entrypoint for the workflows above. `scripts/sim2sim_genesis/` contains the modular scene, controller, metrics, trajectory I/O and runner code; see the [architecture notes](archive/genesis_sim2sim_modular_refactor.md).

## Pair Genesis with Isaac startup states

Pass `--initial_state_npz path/to/isaac/rollout.npz` to initialize each Genesis
environment from the archive's `initial_*` joint/root state fields. The archive
must provide one state per environment plus joint/body names. This mode supports
one parallel rollout batch; no additional startup noise is added, since the
recorded Isaac states already contain the chosen randomization. Keep the noise
and DR flags explicit, and preserve the input archive with the experiment.
When `env_origins` is present, Isaac's grid-placement translations are removed
before loading states into Genesis. For older archives without that field,
provide positions already expressed in the reference motion's local world frame.
This path also restores recorded velocities; native Genesis reference reset uses
zero velocities, so paired-state runs are a different initialization protocol.
Logged world angular velocity is converted to the body-local free-root rotational
DOFs before writing the Genesis state. Verify the actual saved reset states when
making paired comparisons, including both linear and angular velocities.
Saved JSON includes actual `initial_states`, per-rollout failure reasons, and
tracking samples every 0.25 seconds and at failure for checking the pairing.
