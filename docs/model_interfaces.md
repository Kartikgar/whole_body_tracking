# Model interfaces and observations

[Documentation index](README.md)

## Delta actor: 103 dimensions

Current full-body G1 joint-delta and pelvis-wrench delta tasks use the same actor observation, ordered as follows:

1. Base height: **1**, scale 1.
2. World-frame contact-force XYZ for both ankle links: **6**, scale 0.01.
3. Body-frame base linear velocity: **3**, scale 2.
4. Body-frame base angular velocity: **3**, scale 0.25.
5. Projected gravity in the base frame: **3**, scale 1.
6. Joint positions relative to default pose: **29**, scale 1.
7. Joint velocities relative to default velocity: **29**, scale 0.05.
8. Current base joint action: **29**, scale 1.

Total: `1 + 6 + 3 + 3 + 3 + 29 + 29 + 29 = 103`.

During open-loop training the last term is `motion_joint_action` from the recorded dataset. During finetuning it is `current_action`, injected by the runner from the same-step base actor. No previous delta-action channel or temporal history is enabled in the current configuration. Any trained observation normalizer must accompany inference.

The current open-loop critic has **440 dimensions**: actor inputs plus reference joint positions/velocities (58), reference anchor position/orientation (3 + 6), robot body positions (30 × 3), and body orientations (30 × 6). Critic inputs are training-only.

Definitions live in `tasks/tracking/config/g1/flat_env_cfg.py` and `tasks/tracking/mdp/observations.py`. Saved configs describe older checkpoints, which can have different layouts.

## Actions

- **Base policy:** 29 raw joint-position actions; joint-specific scaling and default offsets map them to PD targets.
- **Joint delta:** normally 29 corrections composed with the recorded or current base action. Optional configured joint subsets change the output dimension; see task settings.
- **Pelvis wrench:** 6 normalized outputs `[Fx, Fy, Fz, Tx, Ty, Tz]`, clipped per component and converted to force/torque in the pelvis-local frame at its center of mass. See [wrench mapping](pelvis_wrench_delta.md).
- **Legacy COM-force mode:** separate force representation described in the [historical implementation guide](archive/delta_com_force_mode_implementation.md). It is not interchangeable with a six-component wrench checkpoint.

## References and export

Base tracking consumes motion references through the motion command. ONNX export includes the loaded reference arrays and deployment metadata. The embedded reference is the motion data, not a concatenation of playback episodes or reset-generated segments. Video length and RL episode caps do not truncate or extend those arrays. Export each selected segment separately when its reference should be segment-specific.

Open-loop wrench policies are helpers and skip deployment ONNX export. Finetuned base export contains the deployable joint-action actor, without the frozen wrench branch. [Segment preparation](segment_artifacts.md) validates parent checkpoint identity and segment-reference export.

Training references, target state/action recordings and delta replay predictions have different roles. Read [dataset contracts](datasets.md) before reusing one as another.
