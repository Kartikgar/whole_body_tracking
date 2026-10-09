# Batch evaluation, difficulty and safety

[Documentation index](README.md) · [Motion relevance](motion_relevance.md)

## Run a batch

```bash
python scripts/eval_checkpoint_batch_isaac.py \
  --config configs/batch/beyondmimic.yaml --validate_only
python scripts/eval_checkpoint_batch_isaac.py \
  --config configs/batch/beyondmimic.yaml --headless --device cuda:0

python scripts/eval_sonic_batch_isaac.py \
  --config configs/batch/sonic.yaml --headless --device cuda:0
```

The checkpoint YAML maps each PgS2R-mini segment to its parent-motion checkpoint. SONIC uses one pretrained model directory across its listed motions. YAML paths are relative to the config file. `output_dir` in that file controls storage; do not assume all experiments use the same log prefix.

Both workflows start each environment at reference frame zero, apply configured initial joint noise, and record the valid prefix until first failure or clip completion. Each parallel robot contributes one rollout. Ordinary training episode caps do not define the full-clip evaluation horizon.

The checkpoint launcher orchestrates `scripts/rsl_rl/play.py`; `scripts/rsl_rl/checkpoint_batch_eval.py` performs the bounded rollout and tracking accumulation. Shared safety calculations live in `source/whole_body_tracking/whole_body_tracking/safety_metrics.py`. SONIC has its own rollout integration with that shared accumulator. Child shutdown is bounded so a finished motion cannot indefinitely stall the batch.

## Results

A timestamped batch contains a resolved `config.json`, `summary.csv`, per-motion JSON/process logs, padded rollout NPZ files with `valid_lengths` and initial states, and videos when enabled. Preserve JSON definitions with results: changing today's source does not recompute yesterday's measurements.

Checkpoint tracking metrics use 14 reference bodies; SONIC uses 30. Their body-error values are not directly comparable without matching body sets and conventions. See [tracking metric details](tracking_metrics.html).

## Plot two independent orders

```bash
python scripts/rank_motions.py \
  --run_dir logs/batch_eval/beyondMimic/<run> --min_duration_s 10
```

Outputs default to `<run>/analysis/rankings/`, including difficulty/safety CSVs, all-motion figures and paginated PNG/PDF figures. `--page_size` controls page density. Clips below 10 seconds are excluded by default.

### Difficulty: easy to hard

Sort lexicographically by:

1. Full-clip completion rate, descending.
2. Clip survival, descending: mean over rollouts of `min(observed duration / reference duration, 1)`.
3. Successful-rollout mean errors, ascending, in order: `mpjpe`, `mpjpe_l`, `vel_dist`, `accel_dist`.
4. Motion ID as the final deterministic tie-breaker.

Tracking panels use only full-clip successful rollouts. With zero successes they show a dash. Errors already average over observed samples; dividing again by rollout length would change their meaning. Successful-only reporting prevents short failures from looking artificially accurate.

### Safety measurements

Each rollout contributes measurements over its observed valid prefix. For each clip, take **p90 across rollouts** of each metric. P90 emphasizes the worse part of the rollout distribution without relying on one extreme run; it is not a formal safety guarantee.

- **Non-support contact fraction:** control-step time with contact on any non-support link, divided by observed rollout duration. Simultaneous contacts count once for this time fraction. Only left/right ankle-roll links are exempt: wrists are included in current evaluations. Contact is net force magnitude above 1 N, using the peak over sensor physics-step history within the control step.
- **Soft joint-limit fraction:** violating joint/time samples divided by observed time samples × number of joints with valid soft limits. G1 soft position limits retain 90% of the hard interval around its midpoint. Exceeding this buffer need not exceed the hard mechanical limit.
- **Effort near-limit fraction:** joint/time samples with absolute applied effort at least 90% of the effort limit, divided by all observed joint/time samples. Effort is the implicit actuator's estimated clipped output.
- **Foot slip fraction:** slipping foot/time samples divided by stance foot/time samples. Stance requires force above 10 N; horizontal ankle speed above 0.1 m/s counts as slip. This is a link-speed proxy, not a sole contact-point measurement.
- Additional diagnostics include hard joint violations, minimum margins, maximum speed/effort utilization, contact events and peak forces. These are recorded but do not all enter the sorting score.

A joint/time sample means one joint at one control step. Contact time treats any affected link as one robot-level event at that step. At fixed control rate a robot-level time fraction equals a step fraction; averaging across joints introduces a different denominator.

Older saved runs may exempt wrists. Inspect their `safety_definitions.allowed_contact_bodies`; plotting cannot retroactively recover measurements absent from a saved run. Intentional rolling/floor contact can rank as risky under this operational definition without being unintended behavior.

### Final safety ordering: safer to riskier

Let `c`, `j`, `e` be the clip p90 values for contact fraction, soft-limit fraction and effort near-limit fraction. For each metric compute its percentile rank across included clips:

`P(x) = (number below x + 0.5 × number equal to x) / number of finite values`.

The score is:

`S = 0.50 P(c) + 0.30 P(j) + 0.20 P(e)`.

Lower is safer relative to this evaluated collection. Percentiles normalize unlike measurement distributions; scores change when the included collection changes.

Sustained contact has priority over this score:

1. Clips with `c < 0.10` precede clips with `c >= 0.10` (default threshold; configurable).
2. Below the threshold, sort primarily by `S`, then contact fraction.
3. At/above the threshold, sort primarily by contact fraction, then `S`.
4. Break remaining ties by soft-limit fraction, effort fraction and motion ID.

Missing contact is classified as sustained; missing score components receive worst percentile rank. Thus high sustained contact cannot be canceled out by low effort/joint-limit percentiles. Slip and hard-limit diagnostics are displayed, but excluded from `S`. Safety plots omit completion and a dedicated score panel, show the three score components first, and explain the ordering in the figure footer.
