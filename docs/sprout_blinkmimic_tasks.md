# Sprout tasks with a BlinkMimic base policy

[Documentation index](README.md) · [Training and evaluation commands](training_and_evaluation_commands.md)

The Sprout general tracker is trained outside the BeyondMimic pipeline, with the BlinkMimic world-model recipe
(`scripts/blinkmimic/train_isaac.py`, research log [2026.11.11](research_log/2026.11.11.md)). The tasks below load
such a checkpoint into the usual BeyondMimic workflow: play, open-loop joint-delta training, and PPO finetuning with a
frozen delta.

| Task | Use | Runner config |
|---|---|---|
| `Tracking-Flat-Sprout-BlinkMimic-v0` | play / evaluate / plain PPO finetune of the BlinkMimic policy | `SproutBlinkMimicPPORunnerCfg` |
| `Tracking-Flat-Sprout-DeltaA-OpenLoop-v0` | open-loop joint-delta model on recorded Sprout states/actions | `SproutDeltaActPPORunnerCfg` |
| `Tracking-Flat-Sprout-BlinkMimic-DeltaA-Finetune-v0` | PPO finetune of the BlinkMimic policy with a frozen joint delta | `SproutBlinkMimicDeltaAFineTunePPORunnerCfg` |

Code: `source/.../tasks/tracking/config/sprout/`, observation/termination terms in `tasks/tracking/mdp/blinkmimic.py`,
the actor-critic in `source/.../blinkmimic/rsl_actor_critic.py`.

## Contract

- **Motions.** Convert frax retargets with `scripts/blinkmimic/frax_to_motion_npz.py`. It resamples through
  BlinkMimic's `MotionLib` (50 Hz, central-difference velocities), so the reference is frame-identical to training;
  body poses come from Isaac FK (agrees with BlinkMimic's MJCF FK to 30 µm). The 40 benchmark clips are in
  `data/lafan1_frax/sprout_bm40_motion/`.
- **Policy observation** (`policy` group, 1620 values): 10 steps (oldest first) of `[q - q0, qd, root angular
  velocity (root frame), projected gravity, previous action, 6D R^T R_ref]` (93 each), then 10 future reference
  frames of `[dp, rot6d (current reference heading frame), v, w (future root frame), q, 0.1 qd]` (69 each). Uniform
  noise uses BlinkMimic's per-dimension half-widths.
- **Actions** use BlinkMimic's MJCF joint order (`sprout_spec.MJCF_JOINT_ORDER`, `preserve_order=True`) and its action
  scale `0.25 * 80 Nm / kp`; target = default pose + scale * action.
- **Termination** (base and finetune tasks): BlinkMimic's root height error > 0.25 m or up-axis error > 0.8. The delta
  open-loop task uses the end-effector height termination, as the G1 delta task.
- **Actor-critic.** `BlinkMimicActorCritic` wraps BlinkMimic's `Policy` (EMA weights, frozen input normalizers,
  bf16 autocast as trained) as the RSL-RL actor, with a fresh critic MLP and exploration std 0.05 (action units).
  `empirical_normalization` is off. The architecture comes from `agent.policy.blinkmimic_ckpt` (default
  `logs/blinkmimic_isaac/sprout_10k_s0/ckpt_010000.pt`).
- **Checkpoints.** `--checkpoint` accepts a raw BlinkMimic `ckpt_*.pt` (actor weights only; critic, std and optimizer
  start fresh, iteration 0) or an RSL-RL `model_*.pt` written by a Sprout finetuning run.

## Parity with BlinkMimic

`scripts/blinkmimic/check_task_parity.py` plays a clip from frame 0 (no randomization or noise) in
`Tracking-Flat-Sprout-BlinkMimic-v0` and recomputes BlinkMimic's inputs from the same robot state with BlinkMimic's
code every step. Checkpoint `sprout_10k_s0/ckpt_010000.pt`, full clips:

| Clip | Task: normalized return / failed | `evaluate_isaac.py` | Max obs-history diff | Max goal diff |
|---|---|---|---|---|
| walk1_subject1 | 0.893 / no | 0.888 | 6.3e-7 | 2.1e-4 |
| jumps1_subject1 | 0.841 / no | 0.848 | 6.0e-7 | 1.9e-4 |
| fallAndGetUp1_subject1 | 0.842 / no | 0.858 | 6.6e-7 | 2.8e-4 |
| dance1_subject2 | 0.831 / no | 0.830 | 6.1e-7 | 1.8e-4 |

The goal differences are float32 rounding of the stored reference (Isaac FK and finite differences). The return
differences (≤ 0.016) come from the scene, not the policy interface: ground material combine mode and the reset
writing the reference root velocity as the CoM velocity.

## Commands

```bash
# play
python scripts/rsl_rl/play.py --task Tracking-Flat-Sprout-BlinkMimic-v0 \
  --checkpoint logs/blinkmimic_isaac/sprout_10k_s0/ckpt_010000.pt \
  --motion_file data/lafan1_frax/sprout_bm40_motion/walk1_subject1.npz --num_envs 1

# parity against BlinkMimic's own MDP (one clip, frame 0, no randomization)
python scripts/blinkmimic/check_task_parity.py --headless --clip walk1_subject1

# record states/actions (here in Isaac, as a stand-in for a target simulator)
python scripts/rsl_rl/play.py --task Tracking-Flat-Sprout-BlinkMimic-v0 --headless \
  --checkpoint logs/blinkmimic_isaac/sprout_10k_s0/ckpt_010000.pt \
  --motion_file data/lafan1_frax/sprout_bm40_motion/walk1_subject1.npz --num_envs 64 \
  --record_state_action_trajectories --state_action_target_trajectories 256 --output_state_action_npz path/to/recorded.npz

# open-loop joint delta
python scripts/rsl_rl/train.py --task Tracking-Flat-Sprout-DeltaA-OpenLoop-v0 \
  --motion_file path/to/recorded.npz --num_envs 4096 --headless --disable_dr

# finetune the BlinkMimic policy with the frozen delta
python scripts/rsl_rl/train.py --task Tracking-Flat-Sprout-BlinkMimic-DeltaA-Finetune-v0 \
  --resume True --checkpoint logs/blinkmimic_isaac/sprout_10k_s0/ckpt_010000.pt \
  --delta_policy_checkpoints path/to/delta/model.pt \
  --motion_file data/lafan1_frax/sprout_bm40_motion/walk1_subject1.npz --num_envs 4096 --headless
```

The whole chain (record → delta open-loop → finetune → play the finetuned `model_*.pt`) was smoke-tested on
walk1_subject1 with an Isaac recording (`logs/blinkmimic_isaac/smoke_tasks/`); those models are pipeline checks, not
results.

## Before real runs

Status (2026-11-15): paused; only the smoke chain has run. Resolve these before training deltas or finetunes for
results.

- **Reset noise in recordings.** `--disable_dr` removes events and observation noise but keeps the motion command's
  reset perturbation (pose ±5 cm / ±0.1–0.2 rad, velocity up to ±0.5 m/s, joints ±0.1 rad). With
  `--sample_start_frames`, many recorded 10 s episodes then end early (shortest 42 steps). Zero those ranges
  (`env.commands.motion.pose_range` etc.) when a recording should start on the reference.
- **Delta exploration on Sprout.** Sprout's action scale is 0.62 rad (kp 32.5 joints) and 2.0 rad (kp 10 joints) per
  unit, so the G1 delta runner's initial std 1.0 is large: replaying recorded actions with std-1 delta noise
  survives ~32 steps against ~139 with zero delta, and the 30-iteration smoke run drifted to ~4-step episodes. A
  smaller `agent.policy.init_noise_std` for the Sprout delta is worth trying before long runs.
- **Delta replay tools.** `scripts/Delta_Utils/eval_delta_replay_isaac.py` and `build_gap_dataset_isaac.py` index
  `action_scale` by `joint_names`; for these tasks actions follow `action_joint_names` (MJCF order, now written by
  `play.py`), so those tools need updating (prefer `action_joint_names` when present) before use on Sprout.
- **First step after `env.reset()`.** As in the G1 tasks, the end-effector termination reads the relative body
  targets before the command updates them, so every env terminates on the first step after a full reset; steady-state
  episodes are unaffected.

## Not yet supported

- ONNX deployment export of the BlinkMimic actor (`play.py` skips it with a message).
- `--batch_eval` / `checkpoint_batch_eval.py` (G1 end-effector names, velocity limits and labels).
- Pelvis-wrench delta tasks for Sprout.
- Sprout recordings in Genesis (target-simulator data for the delta).
- Training the BlinkMimic tracker itself stays in `scripts/blinkmimic/train_isaac.py`.
