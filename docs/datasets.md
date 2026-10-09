# Motion and trajectory datasets

[Documentation index](README.md)

## Reference motion versus executed recording

- **Reference NPZ:** desired motion joint/body states for tracking.
- **Executed trajectory NPZ:** actual simulator states plus the base policy's raw joint actions, used for residual dynamics training and replay evaluation.
- **Replay prediction:** corrected Isaac states generated from a recorded initial state and fixed action sequence.

These can share array names but serve different purposes. Validate names, timing, action representation and valid lengths before using a recording for training or replay.

## PgS2R-mini segments

`data/LAFAN1_Retargeting_Dataset/g1/PgS2R-mini/` contains approximately 20-second clips derived from longer retargeted LAFAN1 parent motions. A stem such as `dance1_subject1__100.00s-120.00s` identifies its parent and original time bounds. Tail clips may be shorter. Dataset spelling such as `jumps1` is retained in file IDs, even when plot labels display `Jump1`.

A parent checkpoint can be assigned to its segments because the poses come from that parent reference. Training with sampled start frames supports this use, but does not guarantee identical performance under a segment's new initialization. Measure it through batch evaluation.

## Core recording contract

Typical stacked recordings have trajectory, time and joint/body axes. Key fields include `fps`, `joint_names`, `body_names`, `joint_pos`, `joint_vel`, body world positions/quaternions/velocities, recorded `action` or `actions`, `valid_lengths`, and `initial_*` state fields. Quaternions use wxyz in these workflows.

Padded array dimensions are not the number of usable samples. Sum `valid_lengths` for valid state counts. For replay, states are pre-action: apply `action[t]` at `state[t]` and compare against `state[t+1]`. A window of H steps needs H+1 states. The last action cannot be scored without a successor. Short trajectories can therefore be excluded from longer-horizon evaluation.

The dedicated [replay evaluator](delta_replay_evaluation.md) additionally requires raw base-policy action metadata (`action_mode=base_policy_raw`), validates reset states and aligns joint/body arrays by name. A dataset name ending `traj100` indicates collected trajectories, not necessarily 100 trajectories long enough for every replay horizon.

## Utilities

```bash
python scripts/NPZ_utils/csv_to_npz.py \
  --input_file path/to/retargeted.csv --input_fps 30 --output_name my_motion --headless

python scripts/NPZ_utils/replay_npz.py --motion_file path/to/reference.npz

python scripts/NPZ_utils/merge_motion_npz.py \
  path/to/jump100.npz path/to/dance20.npz --output path/to/composite.npz
```

The merge combines trajectories, rather than joining motions end-to-end. Select the intended trajectory subsets before merging. Shared metadata is preserved when identical; incompatible metadata may be dropped and is reported. Preserve source membership and train/validation separation outside the filename alone.

`extract_motion_segment.py` and `replay_npz_batch_videos.py` in `scripts/NPZ_utils/` support segment preparation and inspection; use their help for exact options. [VLM captions](motion_captioning.md) describe reference videos and are interpretive labels, not ground-truth executed behavior.

Historical loader-format details remain in [the archive](archive/motion_loader_step2_changes.md). Current loaders and saved dataset metadata determine compatibility.
