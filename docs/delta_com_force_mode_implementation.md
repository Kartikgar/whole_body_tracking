# Delta COM-Force Mode: Implementation and Debugging Guide

This document explains the **delta-action COM-force mode** for both open-loop training and delta-policy finetuning.

Goal:

- Keep the existing motion-tracking setup.
- Replace delta joint-action outputs with **3D force outputs**: `Fx, Fy, Fz`.
- Apply those forces at the robot body COM every step.


## 1) What This Mode Does

When `--delta_action_space com_force` is used:

1. The delta-action channel becomes **3D** (`Fx, Fy, Fz`).
2. The external force is applied each step on one body (default: `torso_link`, using motion anchor body name).
3. Open-loop task behavior:
   - the policy itself outputs the 3D force action,
   - joint targets are still replayed from motion NPZ action (`motion_joint_action`),
   - action-penalty reward terms are disabled in the launcher (`action_rate_l2`, `penalty_minimal_action_norm`).
4. Finetune task behavior:
   - the trainable base policy remains in joint space,
   - the frozen delta policy outputs the 3D force action,
   - the action term applies both the base joint target and the frozen COM force in the same step.


## 2) Files and Symbols

Main implementation points:

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/delta_actions.py`
  - `DeltaComForceAction`
  - `DeltaComForceActionCfg`
  - `ExternalDeltaComForceAction`
  - `ExternalDeltaComForceActionCfg`
- `scripts/rsl_rl/train.py`
  - `--delta_action_space com_force`
  - `--delta_com_force_scale`
  - `--delta_com_force_clip`
  - `_configure_delta_action_space(...)`
- `scripts/rsl_rl/play.py`
  - Same CLI/config override logic as training.


## 3) Runtime Flow

### 3.1 CLI to Env Config

In `train.py` / `play.py`, when:

- `--delta_action_space com_force`

the code replaces `env_cfg.actions.joint_pos` with one of:

- `mdp.DeltaComForceActionCfg(...)` for open-loop tasks
- `mdp.ExternalDeltaComForceActionCfg(...)` for finetune tasks

It also copies affine fields from the previous action term (`scale`, `offset`, `clip`, etc.) so joint-target behavior remains consistent.

For open-loop tasks, it also disables action-penalty rewards in the env cfg.


### 3.2 Action Processing in `DeltaComForceAction`

For each environment step:

1. `process_actions(actions)` receives policy actions with shape `[num_envs, 3]`.
2. Optional action-space clipping (`force_clip`) is applied per-axis.
3. A gain (`force_scale`) is applied.
4. Joint replay target is built from motion action:
   - `joint_target = motion_action * joint_scale + joint_offset`
   - optional joint clip is applied.

Then `apply_actions()` does:

1. `set_joint_position_target(joint_target, joint_ids=...)`
2. `set_external_force_and_torque(forces=[Fx,Fy,Fz], torques=[0,0,0], body_ids=[selected_body])`


### 3.3 Action Processing in `ExternalDeltaComForceAction`

For finetuning:

1. `process_actions(actions)` receives the **base policy** joint action with shape `[num_envs, num_controlled_joints]`.
2. That base joint action is processed with the usual joint `scale`, `offset`, and optional joint clip.
3. The frozen delta-policy output is read from the external buffer (`delta_external_actions`) with shape `[num_envs, 3]`.
4. Optional action-space clipping (`force_clip`) is applied per-axis to the external force action.
5. A gain (`force_scale`) is applied.

Then `apply_actions()` does:

1. `set_joint_position_target(base_joint_target, joint_ids=...)`
2. `set_external_force_and_torque(forces=[Fx,Fy,Fz], torques=[0,0,0], body_ids=[selected_body])`


### 3.4 Where the Force Is Actually Applied

IsaacLab applies the buffered wrench inside articulation `write_data_to_sim()`.

- API: `apply_forces_and_torques_at_position(...)`
- `position_data=None` -> applied at body COM.
- `is_global=False` -> force is interpreted in the body local frame.

So the `com_force` actions are body-local COM forces.


## 4) Shapes and Frames (Quick Reference)

- Open-loop policy action: `[N, 3]`
- Finetune base-policy action: `[N, num_controlled_joints]`
- Finetune frozen delta-policy action: `[N, 3]`
- Internal force tensor sent to asset: `[N, 1, 3]`
- Torques: `[N, 1, 3]` zeros
- Open-loop motion replay action: `[N, num_controlled_joints]`
- Joint target: `[N, num_controlled_joints]`

Frame:

- Force is **local to selected body** (not world-frame).


## 5) Training Commands

Open-loop COM-force delta training:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaA-OpenLoop-v0 \
  --motion_file /abs/path/to/motion.npz \
  --delta_action_space com_force \
  --delta_com_force_scale 1.0 \
  --delta_com_force_clip 1.0
```

Finetune with a frozen COM-force delta policy:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaA-Finetune-v0 \
  --motion_file /abs/path/to/motion.npz \
  --delta_policy_checkpoint /abs/path/to/open_loop_com_force_model.pt \
  --delta_action_space com_force \
  --delta_com_force_scale 1.0 \
  --delta_com_force_clip 1.0
```

Notes:

- `delta_com_force_clip` clips action commands **before** scaling.
- `delta_com_force_scale` is applied after that clip.
- Set `--delta_com_force_clip <= 0` to disable clipping.


## 6) What to Check First When Debugging

### A) Confirm mode is active

Look for launcher log:

- `Using COM-force delta action space with 3D force actions ...`
- or `Using COM-force delta action space for finetuning ...`

For open-loop mode you should also see:

- `Disabled action-penalty rewards for COM-force mode.`

If missing, config override did not trigger.


### B) Confirm action dimensions match the task

At runtime:

- open-loop: `env.action_manager.get_term("joint_pos").action_dim` should be `3`
- finetune: `env.action_manager.get_term("joint_pos").action_dim` should stay `num_controlled_joints`


### C) Confirm force body resolves correctly

`DeltaComForceAction` resolves `force_body_name` to exactly one body.

If it does not match exactly one, it raises:

- `force_body_name must resolve to exactly one body`

Fix by using a valid unique body name/pattern.


### D) Confirm the motion file / delta checkpoint pair matches the mode

Open-loop mode still requires motion joint-action replay. If missing:

- `DeltaComForceAction requires motion files with action/actions but none was found`

Finetune mode requires a frozen delta checkpoint whose actor output dim is `3`.


### E) Confirm reward handling is what you expect

For open-loop mode, inspect dumped env config (`logs/.../params/env.yaml`) and verify:

- `rewards.action_rate_l2: null`
- `rewards.penalty_minimal_action_norm: null`

For finetune mode, the base-policy joint-action reward terms stay on their normal finetune settings.


### F) Check force saturation

Because clipping is in action space, a tight clip plus large scale can saturate behavior quickly.

Symptoms:

- `processed_actions` often exactly `-1` or `1`.
- jerky or bang-bang behavior.

If needed, lower `--delta_com_force_scale`.


## 7) Current Scope and Limitation

- `com_force` override is enabled for both **open-loop** and **delta-policy finetune** G1 delta-action tasks via launcher-side checks.
- Finetune COM-force mode keeps the trainable policy in joint space; only the frozen delta-policy branch becomes 3D force output.


## 8) Extension Points

If you want to extend this mode:

1. Add world-frame force mode:
   - requires changing how force is applied (`is_global=True` path in asset API).
2. Add torque outputs:
   - increase action dim from 3 to 6 and map `[Fx,Fy,Fz,Tx,Ty,Tz]`.
3. Remove motion joint replay:
   - in `process_actions`, replace motion replay with other baseline behavior.


## 9) Minimal Code Trace

1. `scripts/rsl_rl/train.py` parses `--delta_action_space com_force`.
2. `_configure_delta_action_space(...)` swaps action cfg to:
   - `DeltaComForceActionCfg` for open-loop, or
   - `ExternalDeltaComForceActionCfg` for finetune.
3. Env builds the matching action term.
4. Open-loop:
   - PPO outputs 3D force action,
   - action term replays motion joint targets,
   - action term applies COM force.
5. Finetune:
   - PPO base policy outputs joint action,
   - frozen delta policy outputs 3D force action,
   - action term applies both in the same step.
