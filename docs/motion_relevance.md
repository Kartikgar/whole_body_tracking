# Recorded motion relevance

[Documentation index](README.md)

`scripts/find_motion_relevance.py` accepts SONIC and BeyondMimic checkpoint batch evaluation folders containing `summary.csv`, per-motion JSON reports, and rollout NPZ recordings. It compares executed behavior using continuous mean nearest-window distance. Lower values indicate greater similarity; rows are sources and columns are targets.

Run from the `whole_body_tracking` directory with Python providing NumPy, SciPy and Matplotlib:

```bash
python scripts/find_motion_relevance.py logs/batch_eval/sonic/<run>
python scripts/find_motion_relevance.py logs/batch_eval/beyondMimic/<run> --min_duration_s 10
```

To use the safety order on both axes, add:

```bash
--safety_ranking_csv logs/batch_eval/beyondMimic/<run>/analysis/rankings/safety_ranking.csv
```

The safety CSV must contain exactly the included motion IDs. Parent axes use median clip safety rank. Without that option, axes use completion, survival and tracking error. Safety order reflects the measurements saved in that evaluation.

Motion labels are generated from arbitrary motion IDs. Numeric batch prefixes are removed, subject IDs and `__<start>s-<end>s` segment suffixes are formatted, and trajectory IDs are retained. No fixed motion list is needed. The behavior name is inferred from JSON `policy_type`; `--behavior_name "My behavior"` overrides the display name.

Recordings must contain `fps`, `joint_names`, `body_names`, `valid_lengths`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w` (wxyz), and `body_lin_vel_w`. Joint columns are aligned by name and must have the same joint set across recordings. The features still use the G1 pelvis, torso, knee, ankle, elbow and wrist body names; this generalizes motions and evaluation workflows, not arbitrary robot morphologies.

Outputs default to `<run>/analysis/relevance_rollout`: clip distance CSV/PNG/PDF, pair scores with original motion IDs, and method metadata with axis identities and feature names. A parent distance matrix is also produced when multiple clips share a parent. No coverage threshold or pre-failure analysis is performed.


## Distance construction and rollout aggregation

Defaults are 50 Hz sampling, 2-second windows, 0.5-second stride and four temporal bins. The script selects up to eight rollouts per clip, balancing completed and failed runs when available. It uses valid observed frames; it does not discard failed executions wholesale.

At each aligned window start, it computes features for the selected rollouts. At least three supporting rollouts are required by default. It takes the component-wise median and selects the actual rollout window nearest to that median as the representative. This gives one consensus window per supported start, up to 600 evenly selected windows per clip. It does not concatenate all rollouts or average their pair distances directly.

Feature groups and their fixed `(weight, scale)` are:

- Joint pose: `(0.22, 0.50 rad)`.
- Joint speed: `(0.16, 2.00 rad/s)`.
- Body pose: `(0.26, 0.25 m)`.
- Root speed: `(0.12, 0.60 m/s)`.
- Pelvis height: `(0.05, 0.20 m)`.
- Foot height: `(0.05, 0.15 m)`.
- Foot speed: `(0.05, 0.50 m/s)`.
- Local path: `(0.07, 0.60 m)`.
- Yaw change: `(0.02, 0.50 rad)`.

A group is a feature family, including its temporal-bin entries. For group g with n_g scalar entries, the transformed vector is `x_g / scale_g × sqrt(weight_g / n_g)`. Concatenate the transformed groups, then use Euclidean distance. Body positions are pelvis-relative and heading-aligned; local path/yaw are relative to the window start. Fixed scales reflect physical feature ranges; they are not fitted percentile ranks or a learned similarity threshold.

For a source library A and target library B:

`D(A → B) = mean over b in B of min over a in A ||b - a||₂`.

Rows are sources; columns are targets. This distance is nonnegative, has no fixed upper bound and need not be symmetric. Lower means the source contains closer examples of the target's executed windows. Zero indicates identical feature windows. The heatmap's 95th-percentile color limit affects display only, not the saved distances. There is no binary relevance threshold, coverage plot or pre-failure plot in the current workflow.

Use the safety CSV to order both axes from safer to riskier. The upper triangle then selects safer-source/riskier-target directions, but proximity alone does not establish delta transferability. Inspect captions and executed videos, especially where partial survival means only an early prefix contributes windows.
