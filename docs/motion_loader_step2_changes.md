# Motion Loader Revamp (Step 2)

This document summarizes the Step 2 implementation for motion dataset loading:

- Support the new per-motion NPZ format (`motion0`, `motion1`, ...).
- Keep compatibility with the legacy stacked NPZ format.
- Correctly handle variable trajectory lengths during open-loop delta-action training and replay.

## Goal

Previously, motion loading assumed one global temporal length `T` for all trajectories (stacked arrays), which made mixed-length datasets impractical.

Step 2 introduces a loader path that accepts per-motion entries and preserves each motion's native length while still exposing tensorized data to the rest of the pipeline.

## Supported Input Formats

## 1) Legacy stacked format (unchanged)

Top-level keys:

- `fps`
- `joint_pos`, `joint_vel`
- `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`
- optional `action` or `actions`

Shapes:

- `joint_*`, `action(s)`: `[N, T, D]` or `[T, D]`
- `body_*`: `[N, T, B, dim]` or `[T, B, dim]`

## 2) New per-motion format (new)

Top-level keys:

- `motion0`, `motion1`, ...
- optional `motion_keys`, optional top-level `fps`

Each `motion{i}` is an object-dict containing:

- required: `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`
- optional: `action` or `actions`
- optional: per-motion `fps`

Shapes inside each motion dict:

- `joint_*`, `action(s)`: `[T_i, D]`
- `body_*`: `[T_i, B, dim]`

## Code Changes

## Motion loader parsing and validation

File:

- [commands.py](/home/kartikgarg/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py)

Main updates:

- Added format dispatch:
  - legacy path if top-level `joint_pos` exists
  - per-motion path otherwise
- Added strict required-key checks for per-motion dicts.
- Added shape checks for both formats.
- Added body-count checks against requested body indices.
- Added action coverage consistency checks:
  - either all motions provide `action/actions`, or none do
- Added `trajectory_time_step_total` tensor storing each trajectory's true length.

Implementation references:

- format dispatch and outputs: `MotionLoader.__init__` ([commands.py](/home/kartikgarg/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py#L41))
- legacy parsing: `_parse_stacked_format` ([commands.py](/home/kartikgarg/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py#L100))
- per-motion parsing: `_parse_per_motion_format` ([commands.py](/home/kartikgarg/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py#L228))

## No Padding (Ragged Storage)

Per-motion trajectories are now stored without temporal padding.

The loader concatenates all trajectories into flat tensors and resolves `(trajectory_id, time_step)` through:

- trajectory start offsets
- per-trajectory lengths (`trajectory_time_step_total`)

Why this is used:

- preserves true motion length exactly
- removes synthetic repeated terminal frames
- keeps per-env wrap/resample behavior length-accurate

Implementation references:

- frame-index resolver and gather methods in `MotionLoader` ([commands.py](/home/kartikgarg/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py#L382))

## MotionCommand updates for variable lengths

File:

- [commands.py](/home/kartikgarg/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py)

Main updates:

- Adaptive bin mapping now uses each env's selected trajectory length.
- Sampled timestep generation now scales by each selected trajectory length.
- Episode wrap/reset now checks each env against its own trajectory length.

Implementation references:

- adaptive sampling bin index and sampled step scaling ([commands.py](/home/kartikgarg/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py#L569))
- per-env wrap condition in `_update_command` ([commands.py](/home/kartikgarg/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py#L667))

## Replay script update

File:

- [replay_npz.py](/home/kartikgarg/whole_body_tracking/scripts/replay_npz.py)

Change:

- Replay loop reset condition now uses selected trajectory length (`trajectory_time_step_total`) instead of global `time_step_total`.

Implementation reference:

- [replay_npz.py](/home/kartikgarg/whole_body_tracking/scripts/replay_npz.py#L124)

## Compatibility

- Existing stacked NPZ files continue to work.
- New per-motion files are now supported.
- `MotionCommand` call-site behavior remains unchanged.
- `MotionLoader` now uses explicit gather methods (`get_joint_pos`, `get_body_pos_w`, etc.) for variable-length trajectories.

## Validation Performed

Static validation:

```bash
python -m py_compile \
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py \
  scripts/replay_npz.py \
  source/whole_body_tracking/whole_body_tracking/utils/exporter.py \
  scripts/convert_motion_npz_to_per_motion.py
```

Runtime note:

- Full runtime instantiation was not executed in this shell because `omni.kit` is unavailable outside Isaac Sim.

## Related Files

- converter from step 1:
  - [convert_motion_npz_to_per_motion.py](/home/kartikgarg/whole_body_tracking/scripts/convert_motion_npz_to_per_motion.py)
- structured change dictionary:
  - [motion_loader_step2_change_dict.json](/home/kartikgarg/whole_body_tracking/docs/motion_loader_step2_change_dict.json)
