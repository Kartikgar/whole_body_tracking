# Delta COM-Force Mode: Implementation and Debugging Guide

This document explains the new **delta-action COM-force mode** for open-loop training.

Goal:

- Keep the existing motion-tracking setup.
- Replace delta joint-action outputs with **3D force outputs**: `Fx, Fy, Fz`.
- Apply those forces at the robot body COM every step.


## 1) What This Mode Does

When `--delta_action_space com_force` is used:

1. Policy action space becomes **3D** (`Fx, Fy, Fz`).
2. Joint targets are still replayed from motion NPZ action (`motion_joint_action`) as the baseline.
3. External force is applied each step on one body (default: `torso_link`, using motion anchor body name).
4. Action-penalty reward terms are disabled in the launcher (`train.py` / `play.py`):
   - `action_rate_l2`
   - `penalty_minimal_action_norm`


## 2) Files and Symbols

Main implementation points:

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/delta_actions.py`
  - `DeltaComForceAction`
  - `DeltaComForceActionCfg`
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

the code replaces `env_cfg.actions.joint_pos` with `mdp.DeltaComForceActionCfg(...)` and copies affine fields from the previous action term (`scale`, `offset`, `clip`, etc.) so baseline joint replay behavior remains consistent.

It also disables action-penalty rewards in the env cfg.


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


### 3.3 Where the Force Is Actually Applied

IsaacLab applies the buffered wrench inside articulation `write_data_to_sim()`.

- API: `apply_forces_and_torques_at_position(...)`
- `position_data=None` -> applied at body COM.
- `is_global=False` -> force is interpreted in the body local frame.

So the `com_force` actions are body-local COM forces.


## 4) Shapes and Frames (Quick Reference)

- Policy action: `[N, 3]`
- Internal force tensor sent to asset: `[N, 1, 3]`
- Torques: `[N, 1, 3]` zeros
- Motion replay action: `[N, num_controlled_joints]`
- Joint target: `[N, num_controlled_joints]`

Frame:

- Force is **local to selected body** (not world-frame).


## 5) Training Command

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaA-OpenLoop-v0 \
  --motion_file /abs/path/to/motion.npz \
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
- `Disabled action-penalty rewards for COM-force mode.`

If missing, config override did not trigger.


### B) Confirm action dimension is 3

At runtime, `env.action_manager.get_term("joint_pos").action_dim` should be `3` in `com_force` mode.


### C) Confirm force body resolves correctly

`DeltaComForceAction` resolves `force_body_name` to exactly one body.

If it does not match exactly one, it raises:

- `force_body_name must resolve to exactly one body`

Fix by using a valid unique body name/pattern.


### D) Confirm motion file still contains `action`/`actions`

This mode still requires motion joint-action replay (open-loop baseline). If missing:

- `DeltaComForceAction requires motion files with action/actions but none was found`


### E) Confirm penalties are disabled

Inspect dumped env config (`logs/.../params/env.yaml`) and verify:

- `rewards.action_rate_l2: null`
- `rewards.penalty_minimal_action_norm: null`


### F) Check force saturation

Because clipping is in action space, a tight clip plus large scale can saturate behavior quickly.

Symptoms:

- `processed_actions` often exactly `-1` or `1`.
- jerky or bang-bang behavior.

If needed, lower `--delta_com_force_scale`.


## 7) Current Scope and Limitation

- `com_force` override is currently enabled for **open-loop delta-action tasks** via launcher-side checks.
- Finetune-specific external-delta composition path is not converted to COM-force mode by this change.


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
2. `_configure_delta_action_space(...)` swaps action cfg to `DeltaComForceActionCfg`.
3. Env builds action term `DeltaComForceAction`.
4. PPO outputs 3D action.
5. Action term:
   - computes force command,
   - replays motion joint targets,
   - applies both each step.
