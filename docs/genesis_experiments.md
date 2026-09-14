# Genesis physical-property experiments

Run from `whole_body_tracking`:

```bash
python scripts/eval_sim2sim_genesis.py \
  --policy_path /path/to/policy.onnx \
  --experiment_config configs/genesis/example.yaml \
  --no-domain_randomization --no-add_noise --seed 42 --compute_metrics
```

Copy the example YAML for each experiment. Every field is optional; omit a
field to preserve its existing setting. An empty mapping (`{}`) changes nothing.
Named maps accept multiple exact robot names. Unknown fields/names, invalid
numbers, and non-single-DoF spring joints fail explicitly.

- `link_masses_kg`: absolute link mass in kg, strictly positive. Uses Genesis
  `RigidLink.set_mass`. In the installed Genesis 0.4.6 implementation this also
  scales the link inertia proportionally while keeping the COM location.
- `joint_stiffness`: absolute passive spring stiffness, non-negative. Uses
  `set_dofs_stiffness`, independently of ONNX PD gains. Revolute units are
  N m/rad; prismatic units are N/m. The spring acts toward the model's reference
  coordinate (`qpos0`), normally zero for URDF joints, not the policy target or
  the pose used to initialize the rollout. Zero removes passive stiffness.
- `robot_friction`: absolute friction coefficient for all robot collision
  geometries, in `[0.01, 5.0]` (Genesis's supported setter range). Nominal G1
  friction is `1.0`. This override also resets all robot friction ratios to
  `1.0`, overriding startup friction randomization in every environment.
- `ground_friction`: ground geometry friction coefficient in `[0.01, 5.0]`.
  This is not necessarily the effective contact coefficient: the installed
  Genesis contact solver uses `max(robot_friction, ground_friction, 0.01)`.
  Lowering ground friction below robot friction therefore may not reduce
  contact friction. Set both `robot_friction` and `ground_friction` to `0.01`
  for low-friction contact. Omitting `robot_friction` preserves robot friction
  and its optional randomization.

Overrides apply after scene construction and startup randomization, to every
environment equally, and persist across rollout resets. Randomization and
observation noise keep their existing CLI defaults; the command above disables
both for controlled comparisons. Config values are printed and included as
`experiment_config_json` in evaluation summaries and recorded motion NPZ metadata.

This config is currently wired into `eval_sim2sim_genesis.py`; the separate
open-loop replay launcher does not consume it.
