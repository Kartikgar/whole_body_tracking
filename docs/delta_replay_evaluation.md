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


### Model-family brackets

Repeat `--model_group` with an annotation followed by adjacent model labels:

```bash
--model_group 'jump sets' 'Jump100' 'Jump100 + Dance20' 'Jump100 + Dance40' 'Jump100 + Dance60' \
--model_group 'walk sets' 'Walk100' 'Walk100 + Dance20' 'Walk100 + Dance40' 'Walk100 + Dance60'
```

Brackets appear beside model rows and below model columns. Their bounds follow
rendered tick labels and can be combined with red highlights and the Oracle cue.


## Error accumulation at fixed intervals

Add `--measurement_interval_s 0.25` to save instantaneous errors and root-height/orientation diagnostics from reset through the replay endpoint. For example:

```bash
python scripts/Delta_Utils/eval_delta_replay_isaac.py \
  --dataset path/to/validation.npz --checkpoint path/to/delta/model.pt \
  --zero_delta --replay_length_s 5 --start_stride_s 0.5 \
  --measurement_interval_s 0.25 --num_envs 818 --seed 0 \
  --output_dir logs/delta_eval/my_zero_delta_growth

python scripts/Delta_Utils/plot_delta_error_growth.py \
  --run_dir logs/delta_eval/my_zero_delta_growth
```

The checkpoint supplies the saved dynamics/action contract; `--zero_delta` applies no wrench. Each starting window runs continuously to 5 seconds without ordinary episode resets or fall terminations. Windows require the entire horizon plus a recorded successor, so the included trajectory population is smaller than for short-horizon experiments. Use the saved schedule for later ensemble comparisons.

Sampling chooses the nearest control frame, rounding half-frame ties forward. At 50 Hz, requested 0.25/0.75-second points are measured at 0.26/0.76 seconds. Both requested and actual times are stored in each window's `error_curve` in `partial_results.json`; the reset sample and endpoint are included. Curves retain observed prefixes on reset/numerical failures; inspect available counts rather than treating missing points as zero error.

The plotting utility writes CSV/JSON analysis and three PNG/PDF figures under `analysis/`: aggregate error growth with trajectory-bootstrap confidence intervals and trajectory p90, individual trajectory-average curves, and cumulative sustained departure onsets. At each time it averages windows within trajectory, then weights trajectories equally. These are instantaneous errors, not the cumulative mean over elapsed replay time.

Default exploratory departure thresholds are 0.25 m mean world body-position error and 0.30 rad joint-position RMSE, sustained over at least 0.5 seconds of sampled time. Adjust with `--body_threshold_m`, `--joint_threshold_rad`, and `--persistence_s`. The onset is retrospectively assigned to the first sample of the sustained crossing. It is neither an exact fall detector nor proof of departure from the training distribution. Windows without a confirmed crossing are right-censored at their last observation. The displayed crossing fractions weight trajectories equally; raw crossing counts and medians in the summary describe windows.


## Shared-rollout ensemble disagreement

The same evaluator can load an explicit list of compatible bootstrap checkpoints:

```bash
python scripts/Delta_Utils/eval_delta_replay_isaac.py \
  --dataset path/to/validation.npz --checkpoint path/to/member0/model.pt \
  --ensemble_checkpoints path/to/member0/model.pt path/to/member1/model.pt path/to/member2/model.pt \
  --replay_length_s 5 --start_stride_s 0.5 --measurement_interval_s 0.25 \
  --num_envs 818 --seed 0 --output_dir logs/delta_eval/my_ensemble_growth

python scripts/Delta_Utils/plot_delta_ensemble_growth.py \
  --run_dir logs/delta_eval/my_ensemble_growth \
  --baseline_dir logs/delta_eval/my_zero_delta_growth
```

The `--checkpoint` supplies the environment configuration; `--ensemble_checkpoints` lists all members, including that checkpoint if desired. Members must share robot, physics, observation and action contracts. Each member loads its own normalizer. No uncertainty gate is applied: the mean raw output drives the shared rollout, followed by the action term's clipping/scaling. The recorded joint-action sequence stays fixed. This measures wrench disagreement on common evolving states, not divergence between independent member trajectories.

At reset and each sampled post-action state, all members receive the same raw observation and the base action at that state/frame. `disagreement_curve` stores timing, member raw/clipped outputs, saturation and disagreement. Normalized uncertainty is the RMS across six component population standard deviations of raw outputs (the runner's uncertainty definition). Physical force/torque disagreement is the RMS across three component standard deviations after clipping, multiplied by the force/torque scale. Force and torque are reported separately, not averaged in mixed units. Clipped normalized disagreement is also saved.

The plotting utility requires a matching zero-delta schedule, seed, environment count, dataset hash, interval and physical contract. It saves aggregate/trajectory disagreement figures and matched error-growth comparisons, with sample CSVs and `ensemble_summary.json`. Its within-horizon error correlations are descriptive because overlapping windows are dependent; disagreement is not a calibrated probability of failure or distribution shift.


### Compare ensemble and single-model error growth

Run a single model with the same dataset, checkpoint environment contract, seed, schedule, batching and measurement interval as the ensemble. Then compare any named runs with:

```bash
python scripts/Delta_Utils/plot_delta_error_growth_comparison.py \
  --run 'Zero delta=logs/delta_eval/my_zero_delta_growth' \
  --run 'Ensemble mean=logs/delta_eval/my_ensemble_growth' \
  --run 'Single delta=logs/delta_eval/my_single_growth' \
  --reference 'Single delta' \
  --output_dir logs/delta_eval/my_growth_comparison
```

The plot shows trajectory-weighted instantaneous body position, joint position and root-height errors with 95% trajectory-bootstrap intervals. CSV/JSON outputs include paired `reference minus model` differences at each sample, using the same trajectory resamples for both conditions. Positive differences favor the tested model. These pointwise intervals do not adjust for testing multiple times/metrics. Complete runs and matching schedules/physical contracts are required.

Comparing a bootstrap ensemble with a model trained on the original dataset measures the practical ensemble pipeline, including bootstrap and training-seed effects. To isolate the benefit of averaging, also compare with each individual bootstrap member on the same schedule.

### Short ensemble gate screening

Add these flags to the ensemble replay command to test several coefficients in
one simulator launch:

```bash
--ensemble_gate_scales 0 0.5 1 2 4 8 --include_zero_baseline \
--windows_per_trajectory 3 --replay_length_s 1 \
--start_stride_s 0.5 --measurement_interval_s 0.25
```

Each selected starting state is duplicated across conditions. Each copy evolves
independently and recomputes its own member predictions and gate every control
step. The gate is `exp(-lambda * u)`, where `u` is the RMS population standard
deviation across six raw output components. Multiply the raw ensemble mean by
the gate before clipping/scaling, matching finetuning. Lambda zero is ungated;
the separately named zero-delta control applies no wrench. The optional window
cap selects up to three evenly spread starts per trajectory from the eligible
stride grid. This is a short screening subset, not exhaustive coverage.

`windows.json` and each result row contain `condition`, `gate_scale` and
`zero_delta`. `gate_mean` records the time-average applied gate; physical wrench
norms are also saved. `summary.json` includes `per_condition` summaries. Do not
interpret its pooled top-level summary as performance of one coefficient.

```bash
python scripts/Delta_Utils/plot_ensemble_gate_screen.py \
  --runs 'Jump100=path/to/jump_run' 'Dance100 train=path/to/dance_run' \
  --output_dir path/to/screen_analysis
```

This writes error-growth and gate-screen PNG/PDF figures plus statistics CSV/JSON.
Windows are averaged within trajectories, then trajectories receive equal weight.
Paired 95% intervals resample trajectories 2,000 times with replacement, seed 0.
Window improvement fractions and p90 model-minus-zero error increases expose
local regressions; overlapping windows are not treated as independent samples.
Use the endpoint and window-mean body/joint errors to shortlist coefficients;
replay screening does not replace finetuning and closed-loop target evaluation.

## Frozen closed-loop evaluation in the finetune environment

The existing playback batch mode also supports
`--task Tracking-Flat-G1-DeltaWrench-Finetune-v0`. Supply the base checkpoint,
single reference clip, `--delta_policy_checkpoints`,
`--delta_policy_uncertainty_gate_scale`, and
`--delta_policy_injection_probability` (zero for the no-wrench control, one for
ensemble assistance), together with `--batch_eval --batch_allow_failures`.
This performs inference only, including same-step base actions in the delta
observations using the production finetuning runner. No optimizer updates run.

Use `--batch_spawn_joint_noise_rad 0.1 --seed 0 --num_envs 100` for uniform
startup joint noise on this experiment. Batch mode disables DR/observation noise,
starts at frame zero, extends the episode to cover the full clip, disables
automatic tracking resets, and records the first matched failure per robot.
Output paths are `--batch_output_json` and `--batch_output_npz`.
Expected tracking failures are successful evaluations when `--batch_allow_failures`
is set; simulator errors still fail the process. A bounded Kit close timeout
preserves the evaluation exit status after results are saved.

Each rollout JSON includes a `diagnostic_curve`: post-action tracking errors and
the disagreement/gate used for that step's pre-action prediction, actual applied
force/torque norms and raw member saturation. Samples occur every 0.25 seconds
(nearest control frame) and at first failure. Compare the saved `initial_*`
arrays to verify exact matching across source conditions; use Genesis's
`--initial_state_npz` to pair the target run with those same joint/root states.

Closed-loop diagnostic curves also include pelvis root-position error (3D world
Euclidean distance in metres) and root-orientation error (shortest full 3D
quaternion angle, including yaw, in radians). Older Isaac recordings can recover
these values from saved post-action poses, reference frame `step - 1`, and the
recorded environment-placement offsets. This recovery does not change rollout
outcomes. Genesis needs these measurements recorded during evaluation.

`plot_closed_loop_ensemble.py` accepts `--exclude_ungated`, `--genesis_dir` and
`--output_dir` to compare an extended coefficient set without overwriting older
plots. Available `lambda_2`, `lambda_4`, `lambda_6` and `lambda_8` folders are
included automatically. The full-horizon figure shows body/joint/root tracking,
contributor counts and applied gate; each condition stops contributing after its
own first failures.
