# Ensemble-gated base-policy finetuning sweep

[Documentation index](README.md)

This experiment selects the uncertainty gate from **closed-loop base-policy performance in Genesis after finetuning**. Delta replay error is a separate diagnostic. The source task is `Tracking-Flat-G1-DeltaWrench-Finetune-v0`; the frozen ensemble assists PPO in Isaac, and only the learned base actor is exported to Genesis.

## Configurable single run with the matched evaluation protocol

From `whole_body_tracking/`, launch a run with configurable gate and injection probability:

```bash
python scripts/Delta_Utils/sweep_ensemble_finetune.py \
  --config configs/finetune/dance29999_matched.json \
  --lambda_value 8 --injection_prob 0.5 \
  --iterations 200 --eval_iterations 50 75 100 150 200 \
  --train_seed 0 \
  --output_dir logs/ensemble_finetune/dance_lambda8_prob05_new \
  --execute
```

Change `--lambda_value` (alias `--lambda`) and `--injection_prob` for each experiment. Use a new output directory for each configuration. Omit `--execute` to write and inspect `plan.json` without running simulators; append `--resume` to reuse completed, validated stages of an identical plan. A failed training stage must be retried in a fresh output directory, not resumed mid-PPO.

The matched config enables **source DR and policy observation noise**, enforced by `--require_dr`. It uses the original dance checkpoint, three Jump100+Dance40 bootstrap ensemble members, 4096 training environments, and RNG-preserving frozen-model loading. Genesis evaluates 100 matched initial states with DR/noise OFF, randomized startup joints ON, and endpoint vertical-error tolerance **1.0 m**. The fixed-state archive is the one from the recent original-policy baseline; edit its path in the config or pass `--initial_state_npz` to use another archive. `--endpoint_tolerance`, `--num_envs` and `--genesis_num_envs` are also configurable; the archive must match the requested evaluation batch.

**`--eval_iterations` counts completed PPO iterations, starting at 1.** Thus 50, 75, 100, 150 and 200 map to checkpoint filename indices 0049, 0074, 0099, 0149 and 0199. The config's legacy `export_iterations` field remains zero-based. Do not confuse the two conventions. `training/update_checkpoints.json` records global checkpoint iterations.

Output contains `plan.json`, `progress.json`, per-stage commands/logs, `lambda_8/training/`, and `lambda_8/iteration_0049/` etc. Each snapshot has a verified `policy.onnx` and `genesis_seed_0/` metrics. Only the finetuned actor is deployed in Genesis. Generate plots after completion with:

```bash
python scripts/Delta_Utils/summarize_ensemble_finetune.py \
  --sweep_dir logs/ensemble_finetune/dance_lambda8_prob05_new
```

## Configuration and iteration convention

Edit `configs/finetune/dance_ensemble_gate.json`. Paths in this config are relative to the project root; interpreter paths are explicit. The supplied configuration uses:

- Dance1 S1 100–120 s reference and its parent `model_29999.pt` base checkpoint.
- Three bootstrap Jump100+Dance40 wrench members, each at `model_4999.pt`, with their own saved normalizers.
- λ = 0, 0.5, 1, 2, 4, 8 and **100 PPO iterations** per finetune, numbered **0–99**.
- Export at **finetune iteration 50, 75, 99**, after the update for that iteration. Each PPO iteration contains its normal multiple optimizer/minibatch updates; these are not capped at 100.
- Source domain randomization **on**: material, joint-default-position, base COM, pushes and policy observation corruption. The launcher passes `--require_dr`, checks these terms, and rejects a config that disables source DR.
- 4096 source environments and source seed 0, same initialization for every λ. Resume restores the base optimizer and normalizer as well as actor/critic weights.
- 100 Genesis environments for each evaluation seed 0, 1, 2. Randomized startup joint positions are enabled; observation noise and target physics DR are off. These are independently controlled by `genesis_randomize_startup_qpos`, `genesis_add_noise` and `genesis_domain_randomization`. Source training DR remains on.
- Original-base control and a no-delta finetuning control at iteration 99. The latter uses the same source task and ensemble with injection probability zero, keeping the source settings identical.

The gate is `exp(-lambda * uncertainty)`, where uncertainty is the RMS of the six population standard deviations across raw member outputs. The ensemble mean is gated before clipping/scaling to wrench units. λ=0 gives full ensemble assistance; it is **not** the no-delta control. TensorBoard records uncertainty, gate, member count, applied wrench norms and saturation.

Finetune checkpoints have names such as `model_finetune_iter_0050.pt`. The checkpoint's internal RSL-RL iteration retains the resumed global counter; `training/update_checkpoints.json` maps it to the zero-based finetune iteration. From parent iteration 29999, the requested local iterations map to stored iterations 30049, 30074 and 30098. This avoids interpreting the parent's iteration number as finetuning progress.

## Plan without launching simulators

From `whole_body_tracking/`:

```bash
python scripts/Delta_Utils/sweep_ensemble_finetune.py \
  --config configs/finetune/dance_ensemble_gate.json \
  --output_dir logs/ensemble_finetune/dance_gate
```

This writes a reviewable `plan.json` containing all commands, resolved configuration and SHA-256 hashes of input checkpoints/reference. It launches nothing. This legacy default plan has **7 training runs**, **20 ONNX policies** and **60 Genesis evaluations** (87 sequential stages), with 100 PPO iterations per run. The driver also supports larger explicit budgets. Use the same output directory for execution after approval:

```bash
python scripts/Delta_Utils/sweep_ensemble_finetune.py \
  --config configs/finetune/dance_ensemble_gate.json \
  --output_dir logs/ensemble_finetune/dance_gate --execute
```

**Obtain experiment approval before launching the full sweep.** `--execute` is the explicit full-run switch; this is not an automatic approval mechanism.

## Short pipeline smoke test

```bash
python scripts/Delta_Utils/sweep_ensemble_finetune.py \
  --output_dir logs/ensemble_finetune/dance_gate_smoke --smoke
```

This uses only λ=1, 16 Isaac environments, **two PPO iterations (0 and 1)**, two ONNX exports and two 25-step Genesis evaluations with two environments each. Controls are skipped. Use a separate directory: these results do not measure full-clip performance.

## Artifacts, cleanup and resume

Each λ has its own `training/` directory and one `iteration_0050/`, `iteration_0075/`, `iteration_0099/` folder, each containing `policy.onnx` and per-seed Genesis summaries. `stages/` stores worker logs and completion records. Export uses `play.py --export_only` in the **base tracking environment**, explicitly accepts a finetuned base actor, verifies ONNX actions against the loaded actor and embeds the full 1000-frame reference. The delta ensemble is absent from deployment.

The original base is evaluated first. Each finetuning run is then followed by its exports and Genesis evaluations before starting the next λ. `progress.json` records the active stage, completed stage counts, failure details and a live leaderboard; the same best-so-far summary is printed to the launcher log. Candidates are ordered by highest strict completion, highest survival, then lowest completed-rollout world MPJPE. Comparisons remain provisional until all configured evaluation seeds complete; the leaderboard includes the original-base and no-delta controls. For a detached launch, save the process ID and redirect output to `sweep.log` so progress remains inspectable after the terminal closes.

Workers execute sequentially with separate process groups and a configurable `--worker_timeout` (default two hours per worker). Kit cleanup is bounded; descendants of the owned worker group are terminated when it exits or times out. No unrelated simulator process is killed. A completion record requires successful execution and validated outputs.

Use `--resume` to reuse completed stages with matching config and checkpoint/reference fingerprints. Incomplete training directories are deliberately not overwritten: preserve the failed logs and use a new output directory to retry that training. Other incomplete stages can be retried after removing their partial outputs. This is stage-level resume, not training continuation.

## Aggregation and plots

```bash
python scripts/Delta_Utils/summarize_ensemble_finetune.py \
  --sweep_dir logs/ensemble_finetune/dance_gate
```

Produces `analysis/results.csv` and Matplotlib PNG/PDF panels for completion, survival, world/local MPJPE, velocity distance and acceleration distance. Curves compare iterations 50, 75, 99 against λ; horizontal lines show controls. Whiskers show seed-level standard deviation, **not** a confidence interval.

- Strict full-clip completion: no termination through the entire reference.
- Clip survival: rollout steps observed before termination divided by the evaluated horizon, averaged across environments.
- Existing Genesis `success`: tolerates failure in the final five steps. The strict completion field is recorded separately.
- Tracking panels: means over **strictly completed rollouts only**, with missing values/gaps if none completed. All-rollout metrics remain in CSV.
- MPJPE is in **mm**; velocity/acceleration distances use finite frame differences in **mm/frame** and **mm/frame²**, consistent with the existing Genesis evaluator. They are not time-scaled m/s quantities.
- Per-environment outcomes and errors are preserved in `rollout_metrics`. Wall time includes simulator startup/compilation, so it is pipeline cost rather than pure policy throughput.

A single source seed is an initial tuning sweep. Multiple target seeds measure evaluation variability, not finetuning seed variability. Replicate promising λ values with additional source seeds before drawing a strong conclusion about gating effectiveness.

## Dance39998: coefficients 4, 6, 8 with Genesis videos

`configs/finetune/dance39998_gate_4_6_8.json` starts from the longer-trained
`model_39998.pt`, caps each run at 100 PPO iterations, and exports/evaluates only
iterations 50 and 75. Source DR and observation corruption remain enabled, with
4096 environments and training seed 0. The three delta ensemble members stay frozen.

This configuration uses one batch of 100 fixed paired startup states, including
the recorded reference velocities, so all exported policies face the same
Genesis initial conditions. Target DR and observation noise are off. This differs
from native Genesis startup, which sets velocities to zero. With a fixed state
batch and no target noise/DR, multiple evaluation seeds would repeat the same
states; the launcher therefore requires one evaluation seed for this mode.

The optional `include_original_base` enables the unchanged-base reference
independently of `include_controls`, which controls the extra no-delta training
run. `genesis_initial_state_npz` supplies the paired archive.
`genesis_record_video: true` saves `rollout.mp4` beside each evaluation's metrics,
showing env 0 while all 100 environments are scored. Video output is validated as
a required stage artifact. Only the base actor is deployed in Genesis; no delta
correction is applied in the target simulator.

```bash
python scripts/Delta_Utils/sweep_ensemble_finetune.py \
  --config configs/finetune/dance39998_gate_4_6_8.json \
  --output_dir logs/ensemble_finetune/2026.10.21/dance39998_gate_4_6_8_50_75 \
  --execute
```

This produces 3 training stages, 7 exports and 7 Genesis video/evaluation stages,
including the original-base reference. Use `--resume` only for that exact plan.

## Re-evaluate exported policies without finetuning

To reuse every ONNX from an existing sweep under corrected target evaluation settings:

```bash
python scripts/Delta_Utils/reevaluate_ensemble_finetune.py \
  --source_dir logs/ensemble_finetune/dance_gate_full \
  --output_dir logs/ensemble_finetune/dance_gate_genesis_no_noise --execute
```

Defaults are `--no-domain_randomization --no-add_noise --randomize_startup_qpos`. All evaluation seeds, environment counts, reference horizons and physics settings are inherited from the source plan. No checkpoints are trained or exported again. The plan fingerprints the reused ONNX files and writes results separately; each completed result must confirm the requested three Boolean settings. Omit `--execute` to inspect the plan, or add `--resume` to reuse completed stages. The final summary CSV and Matplotlib plots are generated automatically. `progress.json` and `sweep.log` provide live progress and best-so-far results.

The initial `dance_gate_full` evaluation used observation noise **on**, with physics DR off and randomized startup joints on. Its saved results remain unchanged. The re-evaluation in `dance_gate_genesis_no_noise` turns observation noise **off**, keeping the same policies and seeds.
