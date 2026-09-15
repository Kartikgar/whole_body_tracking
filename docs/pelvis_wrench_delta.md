# Pelvis-wrench dynamics emulation

Train a wrench in nominal Isaac to reproduce loaded-robot trajectories recorded
in Genesis. The target mass override is an absolute mass: use nominal pelvis mass
plus payload mass. Genesis's installed mass setter also scales inertia.

The new tasks preserve the corresponding Delta-A tracking rewards, observations,
terminations, and 1-second horizon. Open-loop action magnitude and smoothness
penalties are disabled. Fine-tuning retains base joint-action regularization.
Existing joint-delta and COM-force tasks are unchanged.

## Representation

The six dimensionless actor outputs are ordered `[Fx, Fy, Fz, Tx, Ty, Tz]`.
Each is clipped to `[-1, 1]`, then force components are scaled by `300 N` and
torque components by `60 N m`. Both are applied in the **pelvis body-local frame
at its CoM**, on each physics substep. The torso remains the tracking anchor.
These limits are per component, not vector-norm limits.

For payloads in the 10--15 kg range, the force limit is roughly twice the
additional static weight at the upper end. This leaves authority for both
gravity and roughly 1 g of payload acceleration. The torque limit is an initial
tuning value, scaled proportionally from the earlier 5 kg setting: added inertia and angular
acceleration determine the required torque. Clipping can prevent exact dynamics
matching, especially during impacts. The wrench emulates the load in Isaac;
it is not applied in Genesis deployment.

Tune through Hydra overrides, identically in training, playback and fine-tuning:

```bash
env.actions.joint_pos.force_scale=300.0 \
env.actions.joint_pos.torque_scale=60.0 \
env.actions.joint_pos.action_clip=1.0
```

Scales and clipping must be finite positive scalars. The task fixes the body to
`pelvis`. No `--delta_action_space` option is needed; incompatible modes fail.
Do not use the legacy COM-force clipping option for this representation.

## Commands

From `whole_body_tracking`, in the Isaac Lab environment:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaWrench-OpenLoop-v0 \
  --motion_file /path/to/genesis_dataset.npz \
  --num_envs 4096 --max_iterations 2000 --headless --logger tensorboard
```

The dataset must include recorded full-body joint actions. These are replayed
with the usual joint scaling/offset while PPO learns the six wrench outputs.
Add `--disable_dr` to disable source randomization and observation noise.

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-DeltaWrench-OpenLoop-v0 \
  --checkpoint /path/to/wrench/model.pt \
  --motion_file /path/to/genesis_dataset.npz --num_envs 1
```

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaWrench-Finetune-v0 \
  --resume True --checkpoint /path/to/base/model.pt \
  --delta_policy_checkpoints /path/to/wrench/model.pt \
  --motion_file /path/to/reference_motion.npz \
  --num_envs 4096 --max_iterations 1000 --headless --logger tensorboard
```

The frozen wrench actor receives the current state and same-step base joint
action. Only the base actor/critic are updated. Fine-tuning playback uses the
same task, motion, and frozen checkpoint arguments with `play.py` and the
fine-tuned checkpoint (omit `--resume` and training-only iteration arguments).

## Checkpoints, export and diagnostics

Wrench checkpoints store body/frame/order, scales/clipping, and the ordered
actor observation layout under `infos.pelvis_wrench`. Loading, playback, and
frozen ensembles reject incompatible contracts. Nominal base checkpoints remain
valid for initializing fine-tuning. Changing the wrench mapping requires a new
training run; pass training overrides again when loading its checkpoint.
Keep the accompanying `params/agent.yaml` for frozen-policy architecture loading.

Open-loop wrench policies are training artifacts and skip deployment ONNX export.
Fine-tuned export contains only the base joint-action policy, usable by the
existing Genesis evaluator without a wrench. State-action recordings keep joint
actions separate from normalized delta outputs and label wrench representation.

`DeltaWrench` TensorBoard metrics report mean applied force/torque components,
mean vector norms, and per-axis saturation fractions. Repeated saturation warrants
examining trajectory mismatch and increasing the affected scale before retraining.
Force and torque arrow visualization is not included.

## Validation performed

CPU unit tests cover scaling/clipping, joint replay, shape validation, reset
isolation, logging statistics, and checkpoint compatibility. GPU smoke tests
completed one PPO update for each new task, loaded the frozen wrench model,
played both checkpoints, and recorded separate 29D joint / 6D delta actions.
Fine-tuning playback reached the 50-step timeout. The exported base-only ONNX
passed the ONNX checker with 29 action outputs. These short tests validate
integration, not learned payload matching or transfer quality.
