# Training and evaluation commands

[Documentation index](README.md) · [Model interfaces](model_interfaces.md)

Run from `whole_body_tracking/` with the Isaac Lab interpreter for train/play. Replace placeholder paths. Use explicit checkpoints for reproducible experiments; keep their saved `params/` directory.

## 1. Base tracking

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 --motion_file path/to/reference.npz \
  --num_envs 4096 --headless --logger tensorboard

python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-v0 --checkpoint path/to/base/model.pt \
  --motion_file path/to/reference.npz --num_envs 1
```

`--registry_name` can select a WandB reference instead of `--motion_file`. `play.py` exports compatible base policies with reference arrays and metadata. Use [segment preparation](segment_artifacts.md) for a bounded segment demo and validated ONNX.

## 2. Record target trajectories

Run exported base policies in the Genesis environment, as described in [Genesis workflow](genesis_workflow.md). Delta training requires recorded joint actions and corresponding states, not reference poses alone.

## 3. Train a delta model

Pelvis-wrench model:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaWrench-OpenLoop-v0 \
  --motion_file path/to/recorded_dataset.npz \
  --num_envs 4096 --max_iterations 5000 --headless --logger tensorboard \
  --disable_dr
```

Joint-action model uses `Tracking-Flat-G1-DeltaA-OpenLoop-v0` with the same recorded state/action prerequisite. Wrench and joint checkpoints have different output contracts. The current default episode length is 10 seconds; override intentionally through `env.episode_length_s=...` and preserve that setting with results. [Episode caps](episode_length_resets_motion_sampling.md) and replay horizons are independent.

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-DeltaWrench-OpenLoop-v0 \
  --checkpoint path/to/delta/model.pt \
  --motion_file path/to/recorded_dataset.npz --num_envs 1
```

## 4. Evaluate transfer before finetuning

```bash
python scripts/Delta_Utils/eval_delta_replay_isaac.py \
  --dataset path/to/validation.npz --checkpoint path/to/delta/model.pt \
  --replay_length_s 2.0 --start_stride_s 0.5 --num_envs 1459 --seed 0
```

Run each compatible wrench checkpoint independently. Add `--zero_delta` to run the baseline under a checkpoint's environment settings. Keep the dataset, starts, seed, replay length and batching identical. The [delta replay guide](delta_replay_evaluation.md) explains reset handling, aggregation and plotting.

## 5. Optional finetuning

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaWrench-Finetune-v0 \
  --resume True --checkpoint path/to/base/model.pt \
  --delta_policy_checkpoints path/to/delta/model.pt \
  --motion_file path/to/reference.npz \
  --num_envs 4096 --max_iterations 1000 --headless --logger tensorboard
```

Only the base actor/critic train; the delta is frozen. Use `Tracking-Flat-G1-DeltaA-Finetune-v0` for joint-delta finetuning. Finetuning playback takes the same task/reference/frozen-delta arguments and the finetuned base checkpoint, without training-only resume/iteration options. Export the base actor and evaluate it in Genesis without applying delta corrections there.

## Saved runs and reproducibility

Training writes `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/model_*.pt` and `params/` configs. Date-specific experiment names are organization choices, not required inputs to the generic evaluators. Specify checkpoint files rather than assuming the numerically latest file is best.

`--disable_dr` is a training/playback convenience; delta replay separately disables randomization and observation corruption while preserving the saved dynamics. Genesis has its own noise/randomization flags. Always record these separately.

[Historical extended command recipes](archive/training_and_evaluation_commands_legacy.md) retain earlier ensemble, replay and debugging examples. Current workflows above and saved configs take precedence over old defaults.
