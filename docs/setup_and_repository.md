# Setup and repository map

[Documentation index](README.md)

## Runtime environments

- **Isaac Lab environment:** base/delta training, Isaac playback, batch rollouts, segment export and delta replay. The upstream baseline targets Isaac Sim 4.5 / Isaac Lab 2.1 and Python 3.10. This workspace also contains an `IsaacLab/` checkout; use the interpreter/environment configured for that installation.
- **Genesis environment:** `scripts/eval_sim2sim_genesis.py`, with Genesis, PyTorch and ONNX Runtime installed. Keep simulator-specific dependencies in their respective environments.
- **Analysis environment:** NumPy, SciPy and Matplotlib for rankings/relevance/comparison plots; no simulator launch is needed for offline analysis.
- **Captioning environment:** CUDA-enabled PyTorch and the VLM dependencies described in [motion captioning](motion_captioning.md).

From this repository directory, install the extension using the Isaac Lab interpreter:

```bash
python -m pip install -e source/whole_body_tracking
```

Robot assets must be available under the extension's assets directory. Preserve upstream asset/model licenses. See the upstream setup instructions in [the README](../README.md#installation-and-upstream-provenance).

## Directory map

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/`: manager-based environments, observations, rewards, sampling, actions and termination logic.
- `source/whole_body_tracking/whole_body_tracking/robots/g1.py`: G1 dynamics, gains, action scaling and actuator limits.
- `source/whole_body_tracking/whole_body_tracking/safety_metrics.py`: shared batch safety accumulator and definitions.
- `source/whole_body_tracking/whole_body_tracking/sonic/`: SONIC observation and inference integration.
- `scripts/rsl_rl/`: training, playback and checkpoint rollout worker.
- `scripts/Delta_Utils/`: single-checkpoint delta replay evaluation, result aggregation and Matplotlib comparison plots.
- `scripts/NPZ_utils/`: conversion, replay, extraction, merge and upload utilities.
- `scripts/sim2sim_genesis/`: Genesis scene, control, observations, metrics, recording and runner modules.
- `scripts/sim2sim_mujoco/`: separate MuJoCo evaluation implementation.
- `configs/batch/`: SONIC and checkpoint-specific batch motion lists.
- `configs/genesis/`: explicit target physics experiments.
- `data/`: reference motion datasets, including PgS2R-mini.
- `artifacts/checkpoints/`: parent-motion tracking checkpoints.
- `artifacts/genesis/`: recorded trajectory datasets and compositions.
- `logs/rsl_rl/`: trained models and saved environment/agent settings.
- `logs/delta_eval/`: replay runs and comparisons.
- `docs/`: current guides; `docs/archive/`: historical implementation notes.

## Main entrypoints

- Train/play: `scripts/rsl_rl/train.py`, `scripts/rsl_rl/play.py`.
- Batch launchers: `scripts/eval_sonic_batch_isaac.py`, `scripts/eval_checkpoint_batch_isaac.py`.
- Checkpoint batch worker: `scripts/rsl_rl/checkpoint_batch_eval.py`, invoked through `play.py`; use the launcher to schedule a batch.
- Rank/relevance: `scripts/rank_motions.py`, `scripts/find_motion_relevance.py`.
- Segment export: `scripts/prepare_segment_artifacts.py`.
- Target simulation: `scripts/eval_sim2sim_genesis.py`.
- Delta replay: `scripts/Delta_Utils/eval_delta_replay_isaac.py`.
- Aggregate/plot replay: `scripts/Delta_Utils/compare_delta_replay_results.py`, `scripts/Delta_Utils/plot_delta_replay_comparison.py`.

Use `--help` on lightweight launchers for complete options. Training and playback import Isaac and require its runtime even for some help paths. Historical filenames in old experiments do not necessarily match current script locations.
