# Modular Genesis Sim2Sim Evaluator

This document describes the new modular Genesis sim2sim pipeline introduced alongside the legacy evaluator.

## Entry Points

- Legacy evaluator: `scripts/rsl_rl/evaluate_sim2sim_genesis.py`
- New modular evaluator: `scripts/evaluate_sim2sim_genesis.py`

The legacy script remains untouched and still acts as the behavioral reference implementation.

## Module Map

- `scripts/sim2sim_genesis/config.py`
  - runtime configuration dataclasses
- `scripts/sim2sim_genesis/onnx_policy.py`
  - ONNX session setup, metadata parsing, reference-length inference, batched policy calls
- `scripts/sim2sim_genesis/scene.py`
  - Genesis initialization, robot loading, reset-to-reference, state extraction, markers, camera
- `scripts/sim2sim_genesis/observations.py`
  - supported observation terms, history buffers, optional observation noise, motion-action loading
- `scripts/sim2sim_genesis/control.py`
  - action post-processing and explicit PD stepping
- `scripts/sim2sim_genesis/metrics.py`
  - tracking metrics, lite MPJPE metrics, and termination logic
- `scripts/sim2sim_genesis/trajectory_io.py`
  - trajectory recording built on `scripts/rsl_rl/utils.py`
- `scripts/sim2sim_genesis/runner.py`
  - rollout orchestration and summary aggregation

## Current Scope

The modular evaluator intentionally ports only the core sim2sim path:

- ONNX motion policy loading
- reference-motion fetch from ONNX
- Genesis scene build
- observation reconstruction
- action-to-joint-target conversion
- explicit PD stepping
- rollout metrics
- optional trajectory export
- startup and interval domain randomization

The following legacy behaviors are deliberately not migrated yet:

- every legacy compatibility shim beyond what Genesis currently exposes

The new script now supports the legacy evaluator's startup and interval domain randomization path
(joint-default offsets, friction buckets, torso COM shift, and root pushes), and applies the same
fixed-seed Kp/Kd perturbation (`seed=913571`, scale `0.27`) to ONNX metadata gains at load time.
This perturbation is independent of `--seed`.

## Verification Workflow

Recommended smoke-test command:

```bash
python scripts/evaluate_sim2sim_genesis.py \
  --policy_path /path/to/base_policy.onnx \
  --backend cpu \
  --policy_device cpu \
  --num_envs 1 \
  --max_steps 5 \
  --no-add_noise \
  --no-domain_randomization \
  --compute_metrics \
  --metric_num_envs 1
```

Recommended parity checklist:

1. Confirm the ONNX metadata is accepted by both evaluators.
2. Confirm the modular evaluator builds the same observation width.
3. Run a short CPU rollout with noise and domain randomization disabled.
4. Compare summary schema and motion NPZ layout.
5. Compare against the legacy evaluator on short deterministic runs to confirm parity.
