# Training and evaluation commands (G1 Delta-A pipeline)

Reference for training and evaluating the three policy stages in this repo:

| Stage | Task ID | Purpose |
|-------|---------|---------|
| 1. Base tracking | `Tracking-Flat-G1-v0` | Standard whole-body motion tracking |
| 2. Open-loop delta | `Tracking-Flat-G1-DeltaA-OpenLoop-v0` | Learn residual dynamics on top of motion `action`/`actions` |
| 3. Finetune base + frozen delta | `Tracking-Flat-G1-DeltaA-Finetune-v0` | Train base policy with frozen open-loop delta injected at rollout |

All commands below assume you are in the **`whole_body_tracking/`** repo root, with Isaac Lab installed and the package editable-installed:

```bash
cd /path/to/whole_body_tracking
python -m pip install -e source/whole_body_tracking
```

Use your Isaac Lab conda/env when launching scripts (see [README](../README.md)).

**Log layout:** checkpoints go under `logs/rsl_rl/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>_<run_name>/model_*.pt`, with `params/agent.yaml` and `params/env.yaml` saved per run. Default experiment folders (from `rsl_rl_ppo_cfg.py`):

- Base: `2026.06.03/g1_base_policies`
- Delta open-loop: `2026.06.03/g1_delta_policies`
- Finetune: `2026.06.03/g1_finetuned_policies`

---

## Pipeline order

```text
Motion NPZ  -->  (1) Train base tracking
                      |
                      v
              (2) Train open-loop delta  (motion NPZ must include action/actions)
                      |
                      v
              (3) Finetune base with --delta_policy_checkpoint=<open-loop model.pt>
                      |
                      v
              Deploy: play.py exports finetuned base ONNX; Genesis eval uses base only
```

---

## Motion data (shared prerequisite)

### Convert CSV retarget to NPZ (optional upload to WandB registry)

```bash
python scripts/csv_to_npz.py \
  --input_file /path/to/motion.csv \
  --input_fps 30 \
  --output_name my_motion \
  --headless
```

### Replay motion in Isaac (sanity check)

```bash
# From WandB registry
python scripts/replay_npz.py \
  --registry_name your-org/wandb-registry-motions/my_motion

# Or local file (if your replay script supports it)
python scripts/replay_npz.py --motion_file /path/to/motion.npz
```

### Motion file requirements by stage

| Stage | Motion NPZ |
|-------|------------|
| Base tracking | Standard motion fields (`joint_pos`, `body_pos_w`, etc.) |
| Delta open-loop | Same **plus** per-frame `action` or `actions` (reference joint commands used as `motion_joint_action`) |
| Finetune | Standard motion NPZ (no motion `action` required in obs; base uses motion-tracking obs) |

For training, provide motion via **`--motion_file`** (local) or **`--registry_name`** (WandB artifact containing `motion.npz`). Local path is preferred for delta work.

---

## 1. Base tracking policy

**Task:** `Tracking-Flat-G1-v0`  
**Runner cfg:** `G1FlatPPORunnerCfg` (`max_iterations=30000`, `save_interval=500`)

### Train

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --registry_name your-org/wandb-registry-motions/my_motion \
  --num_envs 4096 \
  --headless \
  --logger wandb \
  --log_project_name whole_body_tracking \
  --run_name g1_flat_my_motion
```

Local motion file instead of registry:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /abs/path/to/motion.npz \
  --num_envs 4096 \
  --headless \
  --run_name g1_flat_local_motion
```

### Train without domain randomization

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /abs/path/to/motion.npz \
  --disable_dr \
  --num_envs 4096 \
  --headless \
  --run_name g1_flat_no_dr
```

### Resume training

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /abs/path/to/motion.npz \
  --resume True \
  --checkpoint /abs/path/to/logs/rsl_rl/.../model_5000.pt \
  --num_envs 4096 \
  --headless
```

Or resume from a WandB run (downloads latest `model_*.pt` unless path includes a specific checkpoint name):

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /abs/path/to/motion.npz \
  --resume True \
  --wandb_path your-entity/your-project/run_id \
  --num_envs 4096 \
  --headless
```

### Evaluate in Isaac Lab (`play.py`)

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-v0 \
  --checkpoint /abs/path/to/model_30000.pt \
  --motion_file /abs/path/to/motion.npz \
  --num_envs 4
```

```bash
# WandB checkpoint (optionally append /model_30000.pt to pin a file)
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-v0 \
  --wandb_path your-entity/your-project/run_id \
  --motion_file /abs/path/to/motion.npz \
  --num_envs 4
```

```bash
# Latest checkpoint under logs/rsl_rl/<experiment_name>/ (uses agent load_run settings)
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /abs/path/to/motion.npz \
  --num_envs 4
```

**Notes:**

- `play.py` runs `env.reset()` + motion bootstrap, disables adaptive reference sampling, and exports **base policy ONNX** to `<checkpoint_dir>/exported/<checkpoint_stem>.onnx`.
- Add `--video` for viewport recording; `--disable_dr` to match no-DR training.

### Record state–action rollouts (Isaac, for sim2sim comparison)

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-v0 \
  --checkpoint /abs/path/to/model_30000.pt \
  --motion_file /abs/path/to/motion.npz \
  --num_envs 64 \
  --disable_dr \
  --record_state_action_trajectories \
  --state_action_target_trajectories 100 \
  --output_state_action_npz /abs/path/to/out/base_state_action.npz \
  --headless
```

### Genesis sim2sim (deployed base policy only)

After `play.py`, use the exported ONNX:

```bash
python scripts/rsl_rl/evaluate_sim2sim_genesis.py \
  --policy_path /abs/path/to/checkpoint_dir/exported/model_30000.onnx \
  --urdf_file source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf \
  --motion_file /abs/path/to/motion.npz \
  --num_envs 64 \
  --backend cpu \
  --policy_device cpu \
  --output_csv logs/sim2sim_eval/g1_base_eval.csv
```

---

## 2. Open-loop delta action policy

**Task:** `Tracking-Flat-G1-DeltaA-OpenLoop-v0`  
**Runner cfg:** `G1FlatDeltaActPPORunnerCfg` (`max_iterations=10000`)  
**Action:** policy output is **delta**; env combines `delta + motion_joint_action` before scaling (see `DeltaJointPositionAction`).

### Train (joint delta, default `whole_body`)

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaA-OpenLoop-v0 \
  --motion_file /abs/path/to/motion_with_actions.npz \
  --num_envs 4096 \
  --max_iterations 10000 \
  --headless \
  --run_name walk1_sub1_delta_openloop
```

Hydra-style overrides (optional):

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaA-OpenLoop-v0 \
  --motion_file /abs/path/to/motion_with_actions.npz \
  --num_envs 4096 \
  --headless \
  agent.save_interval=100 \
  agent.max_iterations=20000
```

### Train without DR

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaA-OpenLoop-v0 \
  --motion_file /abs/path/to/motion_with_actions.npz \
  --disable_dr \
  --num_envs 4096 \
  --headless \
  --run_name delta_openloop_no_dr
```

### Train COM-force delta (optional mode)

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaA-OpenLoop-v0 \
  --motion_file /abs/path/to/motion.npz \
  --delta_action_space com_force \
  --delta_com_force_scale 1.0 \
  --delta_com_force_clip 1.0 \
  --num_envs 4096 \
  --headless
```

### Resume

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaA-OpenLoop-v0 \
  --motion_file /abs/path/to/motion_with_actions.npz \
  --resume True \
  --checkpoint /abs/path/to/logs/rsl_rl/.../g1_delta_policies/.../model_5000.pt \
  --num_envs 4096 \
  --headless
```

### Evaluate in Isaac Lab

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-DeltaA-OpenLoop-v0 \
  --checkpoint /abs/path/to/delta_model_9999.pt \
  --motion_file /abs/path/to/motion_with_actions.npz \
  --num_envs 4 \
  --disable_dr
```

Replay **only** motion NPZ actions (zero learned delta):

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-DeltaA-OpenLoop-v0 \
  --checkpoint /abs/path/to/delta_model_9999.pt \
  --motion_file /abs/path/to/motion_with_actions.npz \
  --replay_motion_actions_only \
  --num_envs 4 \
  --headless
```

### Record delta open-loop dataset (for analysis / plots)

Records **policy delta** observations and actions (not combined joint targets):

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-DeltaA-OpenLoop-v0 \
  --checkpoint /abs/path/to/delta_model_9999.pt \
  --motion_file /abs/path/to/motion_with_actions.npz \
  --num_envs 64 \
  --disable_dr \
  --record_delta_model_dataset \
  --delta_dataset_target_trajectories 100 \
  --output_delta_model_npz /abs/path/to/out/delta_openloop_rollout.npz \
  --headless
```

Default output if `--output_delta_model_npz` is omitted:

`<checkpoint_dir>/delta_model_datasets/<checkpoint_stem>_<timestamp>.npz`

### Plot open-loop rollouts (scratch scripts)

```bash
python scripts/scratch/plot_delta_openloop_rollout.py \
  /abs/path/to/delta_model_datasets/model_9999_....npz

python scripts/scratch/plot_state_action_trajectories.py \
  /abs/path/to/state_action_datasets/model_9999_state_action_....npz
```

---

## 3. Finetune base policy with frozen delta

**Task:** `Tracking-Flat-G1-DeltaA-Finetune-v0`  
**Runner cfg:** `G1FlatDeltaAFineTunePPORunnerCfg`  
**Requires:** `--delta_policy_checkpoint` pointing to a trained **open-loop** delta checkpoint (`model_*.pt`).

At each env step during training: base policy acts; frozen delta reads `delta_policy` obs (previous delta + current base action); env applies `base + delta` via `ExternalDeltaJointPositionAction`. PPO updates **only the base policy**.

### Train

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaA-Finetune-v0 \
  --motion_file /abs/path/to/motion.npz \
  --delta_policy_checkpoint /abs/path/to/open_loop_delta/model_9999.pt \
  --num_envs 4096 \
  --headless \
  --run_name finetune_walk1_frozen_delta
```

Optional: initialize finetune from a **base** checkpoint via resume (base weights); still pass frozen delta:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaA-Finetune-v0 \
  --motion_file /abs/path/to/motion.npz \
  --delta_policy_checkpoint /abs/path/to/open_loop_delta/model_9999.pt \
  --resume True \
  --checkpoint /abs/path/to/base_tracking/model_30000.pt \
  --num_envs 4096 \
  --headless
```

### Finetune with COM-force frozen delta

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-DeltaA-Finetune-v0 \
  --motion_file /abs/path/to/motion.npz \
  --delta_policy_checkpoint /abs/path/to/open_loop_com_force_model.pt \
  --delta_action_space com_force \
  --delta_com_force_scale 1.0 \
  --delta_com_force_clip 1.0 \
  --num_envs 4096 \
  --headless
```

### Evaluate in Isaac Lab (base + frozen delta)

**Important:** `play.py` does **not** auto-load `delta_policy_checkpoint` from the finetune run’s `agent.yaml` — pass it explicitly.

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-DeltaA-Finetune-v0 \
  --checkpoint /abs/path/to/finetuned_model_5000.pt \
  --delta_policy_checkpoint /abs/path/to/open_loop_delta/model_9999.pt \
  --motion_file /abs/path/to/motion.npz \
  --num_envs 4 \
  --disable_dr
```

With video:

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-DeltaA-Finetune-v0 \
  --checkpoint /abs/path/to/finetuned_model_5000.pt \
  --delta_policy_checkpoint /abs/path/to/open_loop_delta/model_9999.pt \
  --motion_file /abs/path/to/motion.npz \
  --num_envs 4 \
  --disable_dr \
  --video
```

### Record finetune rollout dataset

With frozen delta loaded, logging records **delta branch** obs/actions (not base-only):

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-DeltaA-Finetune-v0 \
  --checkpoint /abs/path/to/finetuned_model_5000.pt \
  --delta_policy_checkpoint /abs/path/to/open_loop_delta/model_9999.pt \
  --motion_file /abs/path/to/motion.npz \
  --num_envs 64 \
  --disable_dr \
  --record_delta_model_dataset \
  --delta_dataset_target_trajectories 100 \
  --headless
```

Plot finetune action breakdown:

```bash
python scripts/scratch/plot_delta_finetune_actions.py \
  /abs/path/to/delta_model_datasets/finetuned_....npz
```

### Deploy / Genesis eval (finetuned **base** only)

`play.py` exports ONNX from the **finetuned base actor** only (frozen delta is not in the ONNX). Genesis evaluation matches deployment:

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-DeltaA-Finetune-v0 \
  --checkpoint /abs/path/to/finetuned_model_5000.pt \
  --delta_policy_checkpoint /abs/path/to/open_loop_delta/model_9999.pt \
  --motion_file /abs/path/to/motion.npz \
  --num_envs 1 \
  --headless
# ONNX written to: <checkpoint_dir>/exported/<stem>.onnx

python scripts/rsl_rl/evaluate_sim2sim_genesis.py \
  --policy_path /abs/path/to/finetuned/exported/finetuned_model_5000.onnx \
  --urdf_file source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf \
  --motion_file /abs/path/to/motion.npz \
  --num_envs 64 \
  --backend cpu \
  --policy_device cpu \
  --output_csv logs/sim2sim_eval/g1_finetuned_base_eval.csv
```

---

## Common CLI flags (train & play)

| Flag | Scripts | Meaning |
|------|---------|---------|
| `--task` | both | Gym task ID (see table at top) |
| `--motion_file` | both | Override `commands.motion.motion_file` |
| `--registry_name` | train | WandB motion artifact (if no `--motion_file`) |
| `--num_envs` | both | Parallel env count |
| `--headless` | both | No GUI (via AppLauncher) |
| `--device cuda:0` | both | Simulation device |
| `--disable_dr` | both | Turn off event DR and obs corruption/noise |
| `--max_iterations` | train | Override PPO iteration budget |
| `--run_name` | train | Suffix on log directory name |
| `--logger wandb` | train | WandB logging |
| `--log_project_name` | train | WandB project |
| `--resume True` | train | Resume from checkpoint |
| `--checkpoint` | both | Absolute path to `model_*.pt` |
| `--wandb_path` | both | `entity/project/run` or `.../run/model_X.pt` |
| `--delta_policy_checkpoint` | train, play | Frozen open-loop delta for finetune |
| `--delta_action_space` | both | `whole_body` (default), `ankles`, `lower_body`, `com_force` |
| `--record_delta_model_dataset` | play | NPZ delta obs/actions |
| `--record_state_action_trajectories` | play | NPZ joint/body state + applied actions |
| `--video` | both | Record rollout video |

---

## Related documentation

- [play_evaluation_workflows.html](play_evaluation_workflows.html) — detailed `play.py` behavior per policy type, ONNX export, finetune step loop
- [delta_action_finetuning_implementation.md](delta_action_finetuning_implementation.md) — frozen delta wiring in `MotionOnPolicyRunner`
- [delta_action_training_changes.md](delta_action_training_changes.md) — env/obs/reward differences for delta tasks
- [delta_com_force_mode_implementation.md](delta_com_force_mode_implementation.md) — COM-force delta variant
- [README](../README.md) — install, WandB motion registry, original BeyondMimic commands
