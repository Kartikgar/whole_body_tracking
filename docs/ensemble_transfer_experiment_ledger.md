# Dance ensemble transfer: experiment ledger

This ledger accompanies the [research interpretation](ensemble_transfer_research.md). It records the experiment state as of 2026-10-16. Paths under `logs/` and `artifacts/` are local, usually Git-ignored evidence; saved plans and summaries are the authority for exact commands and hashes.

## Shared question and inputs

- **Target motion:** `data/LAFAN1_Retargeting_Dataset/g1/PgS2R-mini/dance1_subject1__100.00s-120.00s.npz` (20 s).
- **Starting actor:** `artifacts/checkpoints/lafan_dance1_subject1/model_29999.pt`.
- **Delta ensemble:** three frozen Jump100+Dance40 bootstrap members, `btstrap000_sd0`, `btstrap001_sd1`, `btstrap002_sd2`, all `model_4999.pt` under `logs/rsl_rl/2026.10.14/g1_delta_wrench_policies/`. Exact paths and SHA-256 values are in each run's `protocol.json` or `plan.json`.
- **Training:** 200 completed PPO iterations, 4,096 parallel Isaac environments, seed 0 for the main comparison, source domain randomization and observation corruption **on**. No-delta uses `Tracking-Flat-G1-v0`; ensemble uses `Tracking-Flat-G1-DeltaWrench-Finetune-v0`. The frozen-model loader preserves Torch RNG. Optimizer resume is enabled; the recorded starting learning rate is 1e-5. Saved checkpoint indices 49/74/99/149/199 correspond to **50/75/100/150/200 completed iterations**.
- **Gate:** uncertainty is RMS population standard deviation over the six dimensionless member outputs. Applied mean is multiplied by `exp(-λ × uncertainty)` before clipping each component to [-1,1] and scaling by 300 N for force or 60 Nm for torque. Injection probability is an episode-level Bernoulli choice in the source simulator.
- **Target evaluation:** exported base-actor ONNX, 100 matched archived Genesis initial states, Genesis domain randomization **off**, observation noise **off**, randomized startup joint positions **on**, endpoint height-error tolerance **1.0 m**. The archive is `logs/sim2sim_eval/2026.10.21/dance29999_baseline_endpoint1m_20261016/paired_initial_states.npz`. Delta ensemble is absent during Genesis deployment.

## Short replay evidence (different evaluation protocol)

- [2-second comparison](../logs/delta_eval/2026.10.14/dance_validation_2s/comparison/README.md): **single** Jump100+Dance40 delta versus zero, 1,160 windows / 78 trajectories. Endpoint joint RMSE **0.3046 vs 0.3592 rad**; body error **0.8466 vs 1.4263 m**. The 100-trajectory archive has 22 trajectories too short for eligible 2-second windows.
- [1-second gate validation](../logs/delta_eval/2026.10.21/ensemble_gate_validation_dance_1s_20261015/README.md): **three-member gated ensemble**, 277 selected windows / 100 trajectories. λ=2 endpoint body error **0.2958 vs 0.4278 m** and joint RMSE **0.1959 vs 0.2153 rad** relative to zero. λ=4 also improves both endpoints. Recorded actions, not actor feedback; ordinary resets bypassed within a window.
- [5-second single-versus-ensemble report](../logs/delta_eval/2026.10.14/dance_validation_jump100_dance40_single_vs_ensemble_5s/README.md): ensemble gains over the single delta on several metrics, but zero delta has lower final world body-position error. This is not a 20-second closed-loop test.

## Closed-loop Genesis completion and survival

The entries below give **completed PPO iterations: full-clip completion % / mean clip survival %**. Each Genesis checkpoint uses 100 matched rollouts. A dash means that iteration was not evaluated. Use the linked JSON for MPJPE, local MPJPE, per-rollout pairing, failure reasons and initialization checks. Survival is the mean fraction of the reference clip observed before termination or completion.

- **Original actor, 0:** **24 / 32.924**. [Baseline evaluation](../logs/sim2sim_eval/2026.10.21/dance29999_baseline_endpoint1m_20261016/).
- **No delta, seed 0:** 50: **12 / 19.851**; 100: **31 / 38.424**; 150: **98 / 98.256**; 200: **44 / 48.997**. [Summary](../logs/ensemble_finetune/2026.10.21/dance29999_base_env_no_delta_200_20261016/analysis/summary.json).
- **No delta, seed 1:** 150: **88 / 90.682**; 200: **67 / 71.719**. [Summary](../logs/ensemble_finetune/2026.10.21/dance29999_base_env_no_delta_200_seed1_20261016/analysis/summary.json).
- **Ensemble λ=8, injection=1.0, seed 0:** 50: **8 / 16.337**; 75: **9 / 17.760**; 100: **7 / 15.433**; 150: **1 / 12.291**; 200: **0 / 9.467**. [Summary](../logs/ensemble_finetune/2026.10.21/dance29999_ensemble_lambda8_200_seed0_20261016/analysis/summary.json).
- **Ensemble λ=8, injection=0.5, seed 0:** 50: **2 / 17.200**; 75: **10 / 18.089**; 100: **6 / 14.766**; 150: **1 / 10.200**; 200: **1 / 9.992**. [Summary](../logs/ensemble_finetune/2026.10.21/dance29999_ensemble_lambda8_prob05_200_seed0_20261016/analysis/summary.json).
- **Ensemble λ=14, injection=0.5, seed 0:** 50: **6 / 14.844**; 75: **4 / 12.279**; 100: **8 / 16.142**; 150: **5 / 13.869**; 200: **6 / 15.038**. [Plan](../logs/ensemble_finetune/dance_lambda14_prob05_new/plan.json), [progress and leaderboard](../logs/ensemble_finetune/dance_lambda14_prob05_new/progress.json).
- **Ensemble λ=14, injection=1.0, seed 0:** 50: **4 / 12.260**; 75: **1 / 9.529**; 100: **0 / 8.608**; 150: **0 / 8.563**; 200: **1 / 9.790**. [Plan](../logs/ensemble_finetune/dance_lambda14_prob10_new/plan.json), [progress and leaderboard](../logs/ensemble_finetune/dance_lambda14_prob10_new/progress.json).

The no-delta seed-0 checkpoint at 150 iterations has observed local MPJPE **46.75 mm** with 98% completion. The original actor has **74.66 mm** with 24% completion. Ensemble λ=8/injection=1.0 at 150 iterations has **82.80 mm** with 1% completion. These observed errors cover very different survived portions; they support the completion result but must not be read as like-for-like 20-second tracking errors. See the source summaries for successful-rollout-only metrics.

At 150 and 200 iterations, the λ=8/injection=0.5 run records 99 `anchor_pos_z` terminations among 100 rollouts at each checkpoint. The zero-delta seed-0 150-iteration run has 2 such terminations. These are termination labels, not a diagnosis of the physical mechanism. [Control and training verification](../logs/ensemble_finetune/2026.10.21/dance29999_ensemble_lambda8_prob05_200_seed0_20261016/CONTROL_COMPARISON.md).

## Comparison integrity and limits

The [no-delta control audit](../logs/ensemble_finetune/2026.10.21/dance29999_base_env_no_delta_200_20261016/CONTROL_AUDIT.md) found that an earlier zero-injection run was confounded by Torch RNG consumed when frozen networks were constructed. With RNG preserved, the standard no-delta task and zero-injection custom task matched exactly through 200 iterations. The valid comparator above is the standard-task run. The saved λ=8 summaries verify zero maximum initial-state differences against the original baseline for each evaluated checkpoint and matching export metadata. λ=14 `plan.json` files identify the same input hashes and archived initial states. Both λ=14 drivers report all 11 stages complete.

One training seed for the ensemble and two for no delta do not establish population-level effects. Genesis rolls out one archived initial-state set; bootstrapping that set does not measure training-seed variability. The trial also changed dynamics during PPO but exported only the actor. None of the present logs isolates whether a harmful correction, weak gate calibration, longer-horizon model error, or PPO adaptation is the dominant cause. [Research note](ensemble_transfer_research.md) lists targeted tests.
