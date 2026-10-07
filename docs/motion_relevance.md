# Recorded motion relevance

`scripts/analyze_sonic_motion_relevance.py` accepts SONIC and BeyondMimic checkpoint batch evaluation folders containing `summary.csv`, per-motion JSON reports, and rollout NPZ recordings. It compares executed behavior using continuous mean nearest-window distance. Lower values indicate greater similarity; rows are sources and columns are targets.

Run from the `whole_body_tracking` directory with Python providing NumPy, SciPy and Matplotlib:

```bash
python scripts/analyze_sonic_motion_relevance.py logs/batch_eval/sonic/<run>
python scripts/analyze_sonic_motion_relevance.py logs/batch_eval/beyondMimic/<run> --min_duration_s 10
```

To use the safety order on both axes, add:

```bash
--safety_ranking_csv logs/batch_eval/beyondMimic/<run>/analysis/rankings/safety_ranking.csv
```

The safety CSV must contain exactly the included motion IDs. Parent axes use median clip safety rank. Without that option, axes use completion, survival and tracking error. Safety order reflects the measurements saved in that evaluation.

Motion labels are generated from arbitrary motion IDs. Numeric batch prefixes are removed, subject IDs and `__<start>s-<end>s` segment suffixes are formatted, and trajectory IDs are retained. No fixed motion list is needed. The behavior name is inferred from JSON `policy_type`; `--behavior_name "My behavior"` overrides the display name.

Recordings must contain `fps`, `joint_names`, `body_names`, `valid_lengths`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w` (wxyz), and `body_lin_vel_w`. Joint columns are aligned by name and must have the same joint set across recordings. The features still use the G1 pelvis, torso, knee, ankle, elbow and wrist body names; this generalizes motions and evaluation workflows, not arbitrary robot morphologies.

Outputs default to `<run>/analysis/relevance_rollout`: clip distance CSV/PNG/PDF, pair scores with original motion IDs, and method metadata with axis identities and feature names. A parent distance matrix is also produced when multiple clips share a parent. No coverage threshold or pre-failure analysis is performed.
