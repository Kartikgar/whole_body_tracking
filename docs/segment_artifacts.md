# Segment demo and Genesis policy artifacts

[Documentation index](README.md)

Run from `whole_body_tracking` using the Isaac Lab Python environment:

```bash
python scripts/prepare_segment_artifacts.py \
  --batch_dir logs/batch_eval/beyondMimic/20261002_214719 \
  --motion jumps1_subject2__040.00s-060.00s \
  --motion run1_subject5__200.00s-220.00s
```

`--motion` accepts a dataset segment stem, a batch motion ID, or its NPZ/JSON filename. Repeat it to prepare multiple segments. `--validate_only` checks selection and file paths without starting Isaac. The launcher resolves each checkpoint from the batch's `config.json` and verifies it against the saved evaluation report. It uses the original segment seed and joint-position noise, with one environment by default. `--num_envs`, `--device`, `--python`, `--no-headless`, and `--timeout_s` customize execution.

Artifacts are stored in `<batch_dir>/segment_artifacts/<dataset-segment-stem>/`:

- `demo.mp4`: newly recorded parent-checkpoint execution of the segment.
- `policy.onnx`: actor, observation normalizer, embedded segment reference, and Genesis deployment metadata.
- Original segment NPZ: local copy of the exact exported reference.
- `rollout.npz` and `rollout.json`: fresh bounded rollout and metrics.
- `manifest.json`: checkpoint/config/motion hashes, exact command, output validation, and completion status.
- `isaac.log`: simulator and policy-loading output.

Existing artifact folders are refused to prevent mixing files across invocations. Use a new batch output directory or move the existing segment folder before repeating a preparation.

The exporter compares ONNX actions with the loaded PyTorch checkpoint. The launcher then loads the ONNX through the Genesis policy loader, checks its reference length, and compares joint references at the first, middle, and last frames with the segment. It checks that the demo completed and contains all rollout frames. Simulator shutdown is bounded; owned child processes are cleaned up on interruption or timeout.

Use the resulting policy in Genesis:

```bash
python scripts/eval_sim2sim_genesis.py \
  --policy_path logs/batch_eval/beyondMimic/20261002_214719/segment_artifacts/jumps1_subject2__040.00s-060.00s/policy.onnx \
  --no-domain_randomization --no-add_noise --seed 85 --compute_metrics
```

Genesis reads the segment reference from the ONNX. Each segment needs its own ONNX even when segments share the same parent checkpoint. The artifact validation checks the deployment interface; it does not establish tracking performance in Genesis.
