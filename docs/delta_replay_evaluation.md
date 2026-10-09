# Single-checkpoint delta replay evaluation

[Documentation index](README.md)

The delta test utilities live in `scripts/Delta_Utils/`: the replay evaluator, separate result aggregator, and Matplotlib plotting script.

Run from `whole_body_tracking` using the Isaac Lab Python environment:

```bash
python scripts/Delta_Utils/eval_delta_replay_isaac.py \
  --dataset path/to/recorded_rollouts.npz \
  --checkpoint path/to/delta/model_3600.pt \
  --replay_length_s 2.0 \
  --start_stride_s 0.5 \
  --num_envs 100
```

Each invocation evaluates **one supplied pelvis-wrench delta checkpoint on one recorded state/action dataset**. It does not classify the dataset as source/target, select other checkpoints, run an automatic baseline, compare models, or plot comparisons.

## Starting points and replay length

- `--replay_length_s`: a single fixed replay duration; defaults to 1 second.
- `--replay_steps`: alternative duration in control steps. Use `--replay_steps 1` for one-step evaluation. Mutually exclusive with seconds.
- `--start_stride_s`: uniformly spaced starts within each trajectory; defaults to 0.5 seconds.
- `--start_frames 0 25 50`: explicit starts instead of the uniform grid, applied to every selected trajectory.
- `--trajectory_indices 0 1 2`: optional trajectory subset; defaults to all trajectories.
- Windows whose successor states would exceed `valid_lengths` are excluded. `windows.json` records the exact schedule.
- `--num_envs`: runs distinct trajectory/start-frame windows in parallel and batches additional windows until the schedule is exhausted.
- `--max_windows`: evenly subsamples eligible windows for quick checks.
- `--validate_only`: validates the dataset, checkpoint files, and schedule without launching Isaac.

Keep the same dataset, starts, replay length, seed, and number of environments when comparing independent checkpoint runs. Longer contact-rich rollouts can change when batching changes.

## Model, dataset, and reset contract

The checkpoint requires its saved `params/env.pkl` and `params/agent.pkl`. These are trusted local Python pickles. The evaluator preserves the trained robot dynamics, observation layout/normalizer, and pelvis-wrench scaling. It disables Isaac randomization, pushes, observation noise, and reset perturbations. Ordinary episode resets and reference resampling are bypassed during a scheduled replay.

The dataset must contain stacked joint/body states, raw recorded joint actions (`action` or `actions`, with `action_mode=base_policy_raw`), joint/body names, fps, `valid_lengths`, and `initial_*` fields. State samples must be **pre-action**: applying `action[t]` to `state[t]` is compared with `state[t+1]`. Frame zero must agree with the initial-state fields. The final recorded action cannot be tested without a successor state.

Joint and body arrays are mapped by name. Recorded link-origin velocities are restored through Isaac's root-link writer and compared against Isaac's link-origin velocity accessors. Initial joint positions are not clipped to soft limits. Every window reports reset errors; invalid resets and nonfinite predictions are counted separately from successful outcomes.

Recorded ONNX action metadata may be rounded to three decimals. The affine transform must agree with training within `0.00051`; the evaluator applies the exact recorded transform.

The policy recomputes a wrench from the evolving Isaac state at every step, while replaying the dataset's fixed joint-action sequence. `--zero_delta` is an explicit alternative condition that uses the same checkpoint's environment configuration but applies zero wrench. It does not create an additional run automatically.

Contact-solver caches and measured contact-force history are not recorded and cannot be restored exactly. Each window starts with cleared sensor/history buffers and a reconstructed physical state. This evaluates dynamics replay accuracy, not closed-loop tracking success. Recurrent policies and temporal observation histories require another reset/history protocol and are rejected.

## Outputs

By default, each invocation creates a timestamped folder under `logs/delta_eval/`. `--output_dir` chooses a folder that must not already exist. It contains:

- `manifest.json`: dataset/checkpoint hashes, replay settings, and execution status.
- `windows.json`: trajectory, starting frame, and replay horizon for each scheduled window.
- `windows.csv`, `partial_results.json`: per-window outcomes, reset errors, metrics, and force/torque magnitudes.
- `summary.json`: per-window and per-trajectory mean/p90 metrics and failure counts.
- `isaac.log`: worker diagnostics.
- Optional `rollouts.npz` with `--record_rollouts`: post-action predicted states, corresponding recorded joint actions, and physical wrenches, indexed by scheduled window.

Metrics include endpoint and within-window mean joint position/velocity RMSE, body/root position error, sign-invariant quaternion angular error, and body linear/angular velocity RMSE. Simulator/environment cleanup is bounded; the launcher terminates only its owned process group on interruption or `--timeout_s` (default one hour). Partial results are saved after each batch.

## Separate result aggregation

After running the desired checkpoints independently:

```bash
python scripts/Delta_Utils/compare_delta_replay_results.py \
  --run model_a=logs/delta_eval/run_a \
  --run model_b=logs/delta_eval/run_b \
  --output_dir logs/delta_eval/comparison
```

The comparison utility validates matching data, schedules, batching, seed, and dynamics configurations, then aggregates common successful windows into JSON/CSV. Individual-run failure counts are retained. Optional `--baseline LABEL` computes relative improvements against that supplied run. It never launches Isaac or loads a checkpoint.

Generate Matplotlib PNG/PDF plots from that comparison:

```bash
python scripts/Delta_Utils/plot_delta_replay_comparison.py \
  --comparison logs/delta_eval/comparison/comparison.json \
  --baseline model_a --title "Validation replay"
```

Plots show absolute mean/endpoint errors, percentage improvements across all state metrics,
paired per-trajectory endpoint differences, and applied force/torque magnitudes.
Windows are matched before averaging within each trajectory, and trajectories receive equal
weight. Whiskers and improvement intervals use 5,000 paired trajectory bootstrap resamples
(seed 0). These pointwise intervals describe this collection of trajectories; they are not
corrected for multiple comparisons and do not establish generalization to other motions.
The script saves `paired_statistics.json` and `.csv` alongside the plots. For comparisons
with multiple horizons, select one with `--steps`. Optional `--output_dir` and `--title`
control the destination and display title. This utility does not launch simulation.

Use `--highlight_models 'LABEL A' 'LABEL B'` to enclose selected model labels in
thin red outlines across all plots. Adjacent labels share one outline; separated
labels get separate outlines. Bounds are measured after layout, and heatmap
composite labels wrap across three lines to avoid overlap. Plot text uses 17–24 pt
fonts with additional spacing for labels and annotations. PNGs use 300 dpi and PDFs retain
vector text/graphics.
Use `--oracle_model 'LABEL'` for a compact Oracle pointer to a model label.
Alternative presentation styles are `--oracle_style badge` and `inline`.

## Single-environment play videos

To visually inspect a learned pelvis-wrench delta on any recorded dataset with
uniformly sampled reference start frames:

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-DeltaWrench-OpenLoop-v0 \
  --checkpoint path/to/model_4999.pt --use_checkpoint_config \
  --motion_file path/to/recorded_rollouts.npz \
  --sample_start_frames --num_envs 1 --seed 0 --disable_dr \
  --headless --video --video_length 500 --video_folder path/to/video_output
```

At 50 Hz, 500 steps produces approximately 10 seconds of video. Add
`--replay_motion_actions_only` for zero delta. `--use_checkpoint_config` loads
the trusted saved environment and agent pickles next to the checkpoint, then
applies the CLI dataset, seed, device and environment-count overrides.
Adaptive sampling remains disabled; `--sample_start_frames` enables uniform
sampling at resets. Without this flag, play retains its frame-zero default.
These demos use ordinary play terminations/resets and the saved episode length;
they are not fixed-horizon evaluation windows, and reset schedules can differ
between models even when their first sampled start matches.
