# Delta Action Training: Changes and ASAP Correspondence

This document summarizes the delta-action training implementation added in this codebase, and maps each part to the corresponding behavior in ASAP.

It is intended to make code review straightforward: what changed, where it changed, and whether it is parity or an intentional deviation.

## Scope

This covers:

- Open-loop delta-action pretraining.
- Finetuning with a frozen delta policy.
- Data plumbing for motion files, including multi-trajectory motion files.

It does not cover unrelated motion retargeting or visualization scripts.

## ASAP Reference Points

Primary ASAP references used for parity:

- Training flow and commands:
  - `/home/kartikgarg/ASAP/README.md` (`Train delta action model`, `Use delta action model for policy finetuning`)
- Open-loop delta observations/scales/noise:
  - `/home/kartikgarg/ASAP/humanoidverse/config/obs/delta_a/open_loop.yaml`
- Delta action control composition in open-loop:
  - `/home/kartikgarg/ASAP/humanoidverse/envs/delta_a/delta_a_open_loop.py` (`_compute_torques`)
- Finetune obs split (base actor vs closed-loop actor used by delta policy):
  - `/home/kartikgarg/ASAP/humanoidverse/config/obs/delta_a/train_policy_with_delta_a.yaml`

## What Changed in This Repo

## 1) Motion Data Plumbing

Files:

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`

Changes:

- `MotionLoader` now accepts motion action arrays from either `action` or `actions`.
- Open-loop delta action can require these keys (`require_motion_action=True`).
- Motion loader now supports both:
  - Single trajectory format:
    - `joint_pos/joint_vel`: `[T, D]`
    - `body_*`: `[T, B, dim]`
    - `action/actions`: `[T, D]`
  - Multi-trajectory format:
    - `joint_pos/joint_vel`: `[N_traj, T, D]`
    - `body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w`: `[N_traj, T, B, dim]`
    - `action/actions`: `[N_traj, T, D]`
- Added consistency checks across trajectory/time dimensions.

ASAP correspondence:

- ASAP requires motion files with an extra `action` key for open-loop delta training.
- This implementation keeps that requirement while extending to multi-trajectory files.

## 2) Equal Random Trajectory Sampling (Open-Loop Only)

Files:

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`

Changes:

- Added per-env `trajectory_ids` in `MotionCommand`.
- Added trajectory sampling controls in `MotionCommandCfg`:
  - `sample_trajectories`
  - `equal_trajectory_sampling`
- In open-loop env config:
  - `self.commands.motion.sample_trajectories = True`
  - `self.commands.motion.equal_trajectory_sampling = True`
- Sampling behavior when enabled:
  - Balanced random assignment per resample batch so trajectories are used approximately equally (floor/ceil counts), with random order.

ASAP correspondence:

- ASAP supports sampling from motion sets; this implementation explicitly guarantees equal random coverage for open-loop training.

## 3) Delta Action Control Terms

Files:

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/delta_actions.py`
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/__init__.py`

Changes:

- Added `DeltaJointPositionAction` and config:
  - Open-loop action becomes `delta + motion_action` before scale/offset.
- Added `ExternalDeltaJointPositionAction` and config:
  - Finetune action becomes `base_policy_action + external_delta` (external delta provided by frozen policy runner).
- Exported these terms in `mdp/__init__.py`.

ASAP correspondence:

- Matches ASAP open-loop concept where delta action is applied relative to motion action.
- Matches ASAP finetune concept where closed-loop policy is combined with a frozen/open-loop-style delta policy output.

## 4) Observation Parity for Open-Loop Delta

Files:

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`

Changes:

- Added open-loop-specific obs groups:
  - `DeltaOpenLoopPolicyObsCfg`
  - `DeltaOpenLoopCriticObsCfg`
- Added observation terms needed by ASAP-style open-loop:
  - `base_pos_z`
  - `feet_contact_force`
  - `projected_gravity`
  - `motion_joint_action`
- Added explicit scale/noise setup in open-loop groups to mirror ASAP config intent.
- Added helper obs terms:
  - `motion_joint_action`
  - `external_delta_action`
  - `feet_contact_force`

ASAP correspondence:

- Mirrors ASAP `obs/delta_a/open_loop.yaml` structure with the extra terms beyond base tracking obs.

## 5) Rewards

Files:

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- Base reward terms inherited from:
  - `source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`

Changes:

- Added `penalty_minimal_action_norm` term and wired it for open-loop with weight `-0.1`.
- Open-loop retains base tracking rewards and adds this term.

ASAP correspondence:

- ASAP open-loop config includes `penalty_minimal_action_norm=-0.1`.
- Formula note:
  - Current code uses `exp(-||a_delta||) - 1` (per current repo state).
  - ASAP has both normalized and penalty variants in code; this implementation follows current project choice.

## 6) Task Registration and Environment Configs

Files:

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/__init__.py`
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`

New tasks:

- `Tracking-Flat-G1-DeltaA-OpenLoop-v0`
- `Tracking-Flat-G1-DeltaA-Finetune-v0`

Environment configs added:

- `G1FlatDeltaAOpenLoopEnvCfg`
- `G1FlatDeltaAFineTuneEnvCfg`
- `DeltaPolicyObsCfg` (frozen delta-policy input group during finetuning)

ASAP correspondence:

- Open-loop pretraining and closed-loop finetune split matches ASAP training stages.

## 7) PPO Runner Integration for Frozen Delta Policy (Finetune)

Files:

- `source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py`
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `scripts/rsl_rl/cli_args.py`

Changes:

- Extended runner to optionally load a frozen delta policy checkpoint.
- Runner computes frozen delta actions from `delta_policy` observation group each rollout step.
- Runner writes those actions to an env buffer consumed by `ExternalDeltaJointPositionAction`.
- Added fine-tune runner cfg fields:
  - `delta_policy_checkpoint`
  - `delta_policy_obs_group`
  - `delta_policy_critic_obs_group`
  - `delta_policy_action_buffer_name`
  - `delta_policy_require`
  - `delta_policy_clip_actions`
- Added CLI arg `--delta_policy_checkpoint`.

ASAP correspondence:

- Equivalent to using pretrained/open-loop delta model during policy finetuning.

## 8) Training CLI: Local Motion File Without Registry

Files:

- `scripts/rsl_rl/train.py`

Changes:

- Added `--motion_file` support (local NPZ path).
- Made `--registry_name` optional when `--motion_file` is provided.
- Keeps wandb artifact path as fallback when `--motion_file` is not provided.

ASAP correspondence:

- ASAP uses direct motion file path in training commands; this now supports the same workflow.

## Parity Matrix (ASAP vs This Repo)

- Motion file must include action for open-loop delta:
  - ASAP: Yes
  - This repo: Yes (`require_motion_action=True`)
  - Status: Matched

- Open-loop extra obs (`base_pos_z`, `feet_contact_force`, `projected_gravity`, motion action):
  - ASAP: Yes
  - This repo: Yes
  - Status: Matched

- Open-loop obs scales/noise:
  - ASAP: Explicit `obs_scales` and `noise_scales`
  - This repo: Explicit `scale=` and selective `noise=` in open-loop obs group
  - Status: Matched in intent

- Closed-loop finetune with frozen delta policy:
  - ASAP: Yes
  - This repo: Yes (`MotionOnPolicyRunner` + `ExternalDeltaJointPositionAction`)
  - Status: Matched

- Multi-trajectory motion file support:
  - ASAP: supported through motion library datasets
  - This repo: explicit `[N_traj, T, ...]` loader support and indexing
  - Status: Matched/extended

- Equal random trajectory usage:
  - ASAP: sampling over motion set
  - This repo: explicit equal-balanced random sampling toggle (open-loop enabled)
  - Status: Explicitly enforced here

- Minimal action penalty exact formula:
  - ASAP: multiple related variants exist
  - This repo: currently `exp(-||a||) - 1`
  - Status: Project-specific choice (documented)

## Open-Loop and Finetune Run Commands (Current Repo)

Open-loop delta training:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaA-OpenLoop-v0 \
  --motion_file /abs/path/to/motion_with_action.npz \
  --num_envs 4096 \
  --max_iterations 20000 \
  agent.save_interval=100 \
  --headless
```

Finetune with frozen delta policy:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaA-Finetune-v0 \
  --motion_file /abs/path/to/motion.npz \
  --delta_policy_checkpoint /abs/path/to/open_loop_model.pt \
  --num_envs 4096 \
  --headless
```

## Notes for Reviewers

- This implementation intentionally keeps delta logic as project-side extensions without forking `rsl_rl`.
- Open-loop-only trajectory equalization is configured in env cfg, not hardcoded globally.
- Existing non-delta tracking tasks remain available and unchanged in task registry.
