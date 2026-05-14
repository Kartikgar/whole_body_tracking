# Episode Length, Resets, and Motion Sampling

This note summarizes how **RL episode length** relates to **motion clip length**, when **commands and motion references reset**, and how **starting frames and trajectories** are chosen in the tracking tasks. It reflects the behavior implemented in Isaac Lab’s command manager plus this repository’s `MotionCommand`.

## Two Different Horizons

### Environment episode length (RL rollout cap)

In `ManagerBasedRLEnv`, the maximum duration of an RL episode is set by configuration:

- `episode_length_s` — maximum episode time in **seconds**.
- `max_episode_length` — same quantity in **control steps** (after decimation):

  \[
  \texttt{max\_episode\_length} = \lceil \texttt{episode\_length\_s} / \texttt{step\_dt} \rceil
  \]

  where `step_dt` is the control period (typically `decimation * sim.dt`).

**Meaning:** This is the **training/evaluation rollout horizon** for the MDP: timeouts, episode buffers, and “end of episode” logging are tied to this clock, not to the motion file’s number of frames.

Tracking configs in this repo (for example G1 flat) inherit defaults such as `episode_length_s = 10.0` from `tracking_env_cfg.py` and the robot-specific `flat_env_cfg.py`. If your motion clip is **longer** than one episode in wall-clock terms, a single episode will **not** necessarily play the entire clip; the episode can end earlier due to timeout or terminations.

### Motion clip length (reference data)

The motion loader concatenates one or more trajectories and exposes per-trajectory lengths (`trajectory_time_step_total`). The command term keeps a per-environment **`time_steps`** index into the **currently selected** trajectory.

**Meaning:** The NPZ (or other motion source) defines **how much reference motion exists**. The RL episode length defines **how long the agent keeps interacting before the env declares the episode over**. Those are orthogonal unless you configure them to align (for example short clips and long episodes, or the opposite).

## When Does the Motion Command Resample?

`MotionCommand._resample_command` runs whenever the generic `CommandTerm` path decides to resample. Concretely, that includes:

1. **Episode reset** — Isaac Lab’s `CommandTerm.reset` zeros the command counter and calls `_resample` → `_resample_command` for the reset environment indices. So at the start of each new episode (per env), trajectory and time index are redrawn together with state perturbations (pose/velocity/joint noise per config).

2. **End of the current motion trajectory** — In `MotionCommand._update_command`, `time_steps` is incremented each control step. If `time_steps` reaches the length of the current trajectory, those envs call `_resample_command` **even if the RL episode has not timed out**. Playback of the reference then “jumps” to a new sampled trajectory segment and start frame.

So an agent can see **multiple** motion resamples inside one RL episode if the sampled motion segment is shorter than the remaining episode time.

## How the Next Trajectory and Start Frame Are Chosen

All of this logic lives in:

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py` — class `MotionCommand`.

On each resample, the order is:

1. **`_sample_trajectory_ids`** — If the motion file contains multiple trajectories and `sample_trajectories` is true, pick which trajectory each env follows (uniform random, or **equal** coverage with shuffled order when `equal_trajectory_sampling` is true). Single-trajectory files always use index `0`.

2. **`_adaptive_sampling`** — Sets the **starting frame** `time_steps` for those envs.

### `sample_time_steps` (start frame)

- **`sample_time_steps=True` (default):** Start frames are **not** fixed at zero. The implementation uses **binned adaptive sampling** along the motion timeline: bins are sized from total motion length vs control step (`bin_count` in `__init__`). Sampling probabilities combine **failure statistics** (which bins led to recent terminations) with a **uniform floor** (`adaptive_uniform_ratio`), then optional temporal smoothing (`adaptive_kernel_size`, `adaptive_lambda`). A bin is drawn with `torch.multinomial`, then a **continuous** offset in `[0, 1)` is added and mapped to an integer frame in **`[0, T-1]`** for trajectory length `T`. So starts are **always inside** the valid frame range for the selected trajectory, but the distribution is **curriculum-like**, not necessarily uniform over all frames.

- **`sample_time_steps=False`:** **Deterministic replay:** each resample sets `time_steps` to **0** for the affected envs (always start at the first frame of the selected trajectory).

### Relation to “arbitrary start anywhere in the clip”

- **Within bounds:** Any sampled start is an integer index along **that** trajectory only; it does not point outside the clip.

- **Not i.i.d. uniform over frames** when adaptive sampling is active: hard segments get revisited more often once failures accumulate in their bins, unless the uniform component dominates.

## Quick Mental Model

| Concept | What it controls |
|--------|-------------------|
| `episode_length_s` / `max_episode_length` | RL episode timeout and rollout length (Isaac Lab env). |
| `time_steps` + trajectory length | How long the **current** reference segment plays before a **motion** resample. |
| `_resample_command` | New trajectory id (if applicable) + new start frame + robot state perturbations. |
| Terminations / timeouts | Can end the RL episode; motion resample also happens on trajectory exhaustion independently. |

## Primary Code References

- Motion command: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py` — `MotionCommand._resample_command`, `_sample_trajectory_ids`, `_adaptive_sampling`, `_update_command`.
- Motion command config: same file — `MotionCommandCfg` (`sample_time_steps`, `sample_trajectories`, `equal_trajectory_sampling`, adaptive hyperparameters).
- RL episode length: `IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py` — `max_episode_length_s`, `max_episode_length`.
- Command reset / resample orchestration: `IsaacLab/source/isaaclab/isaaclab/managers/command_manager.py` — `CommandTerm.reset`, `compute`, `_resample`.
- Task defaults: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py` and robot env cfgs such as `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py` for `episode_length_s` and motion-related overrides.
