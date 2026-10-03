# BeyondMimic Motion Tracking Code

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)

## Overview

BeyondMimic is a versatile humanoid control framework that provides highly dynamic motion tracking with the
state-of-the-art motion quality on real-world deployment and steerable test-time control with guided diffusion-based
controllers.

This repo covers the motion tracking training in BeyondMimic. **You should be able to
train any sim-to-real-ready motion in the LAFAN1 dataset, without tuning any parameters**.

For sim-to-sim and sim-to-real deployment, please refer to
the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

### Alternative Implementations

- There is an alternative reproduction of BeyondMimic in [mjlab](https://github.com/mujocolab/mjlab), a new Isaac Lab-style manager API powered by MuJoCo-Warp for RL and robotics research. See the implementation [here](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/tracking_env_cfg.py).

## Installation

- Install Isaac Lab v2.1.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
# Option 1: SSH
git clone git@github.com:HybridRobotics/whole_body_tracking.git

# Option 2: HTTPS
git clone https://github.com/HybridRobotics/whole_body_tracking.git
```

- Pull the robot description files from GCS

```bash
# Enter the repository
cd whole_body_tracking
# Rename all occurrences of whole_body_tracking (in files/directories) to your_fancy_extension_name
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/whole_body_tracking
```

## Motion Tracking

### Motion Preprocessing & Registry Setup

In order to manage the large set of motions we used in this work, we leverage the WandB registry to store and load
reference motions automatically.
Note: The reference motion should be retargeted and use generalized coordinates only.

- Gather the reference motion datasets (please follow the original licenses), we use the same convention as .csv of
  Unitree's dataset

    - Unitree-retargeted LAFAN1 Dataset is available
      on [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
    - Sidekicks are from [KungfuBot](https://kungfu-bot.github.io/)
    - Christiano Ronaldo celebration is from [ASAP](https://github.com/LeCAR-Lab/ASAP).
    - Balance motions are from [HuB](https://hub-robot.github.io/)


- Log in to your WandB account; access Registry under Core on the left. Create a new registry collection with the name "
  Motions" and artifact type "All Types".


- Convert retargeted motions to include the maximum coordinates information (body pose, body velocity, and body
  acceleration) via forward kinematics,

```bash
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
```

This will automatically upload the processed motion file to the WandB registry with output name {motion_name}.

- Test if the WandB registry works properly by replaying the motion in Isaac Sim:

```bash
python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
```

- Debugging
    - Make sure to export WANDB_ENTITY to your organization name, not your personal username.
    - If /tmp folder is not accessible, modify csv_to_npz.py L319 & L326 to a temporary folder of your choice.

### Policy Training

- Train policy by the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

- Play the trained policy by the following command:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

The WandB run path can be located in the run overview. It follows the format {your_organization}/{project_name}/ along
with a unique 8-character identifier. Note that run_name is different from run_path.

## Code Structure

Below is an overview of the code structure for this repository:

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  This directory contains the atomic functions to define the MDP for BeyondMimic. Below is a breakdown of the functions:

    - **`commands.py`**
      Command library to compute relevant variables from the reference motion, current robot state, and error
      computations. This includes pose and velocity error calculation, initial state randomization, and adaptive
      sampling.

    - **`rewards.py`**
      Implements the DeepMimic reward functions and smoothing terms.

    - **`events.py`**
      Implements domain randomization terms.

    - **`observations.py`**
      Implements observation terms for motion tracking and data collection.

    - **`terminations.py`**
      Implements early terminations and timeouts.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  Contains the environment (MDP) hyperparameters configuration for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  Contains the PPO hyperparameters for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  Contains robot-specific settings, including armature parameters, joint stiffness/damping calculation, and action scale
  calculation.

- **`scripts`**
  Includes utility scripts for preprocessing motion data, training policies, and evaluating trained policies.

This structure is designed to ensure modularity and ease of navigation for developers expanding the project.

## NVIDIA SONIC evaluation

The default pretrained NVIDIA SONIC G1 actor can track compatible 29-DoF G1
motion NPZ files from `data/` in Isaac Lab or Genesis. Download the pinned ONNX
pair, observation configuration, licenses, and G1 asset into an ignored local
directory:

```bash
python scripts/setup_sonic.py --model_dir artifacts/sonic/default
```

Run it in Isaac Lab:

```bash
python scripts/run_sonic.py \
  --sonic_model_dir artifacts/sonic/default \
  --motion_file data/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.npz
```

For a batch of motions, edit `configs/sonic/isaac_batch_example.yaml` and run:

```bash
python scripts/eval_sonic_batch_isaac.py \
  --config configs/sonic/isaac_batch_example.yaml --headless --device cuda:0
```

The YAML lists motion NPZ paths, the SONIC model directory, an output directory,
the number of parallel Isaac environments, a seed, and uniform initial joint
noise in radians. Paths are relative to the YAML file. Each motion starts at
frame zero and runs through its full valid length, with one reset per environment.
Each environment contributes one rollout. A timestamped output folder contains
one common `summary.csv` row per motion, a detailed JSON and a log per motion,
and a padded NPZ of policy actions and robot states with `valid_lengths` and
initial states. Each motion also gets an MP4 with a camera following env 0;
set `record_video: false` in the YAML to skip capture. The MP4 path appears in
the CSV. Failed rollouts are preserved in the CSV and NPZ. The command
exits nonzero if any rollout ends early. The `body_position_error` and
`body_velocity_error` columns are in meters and meters per second;
`body_acceleration_error` is in meters per second squared.

Run the same actor and motion in Genesis:

```bash
python scripts/eval_sim2sim_genesis.py \
  --policy_type sonic \
  --sonic_model_dir artifacts/sonic/default \
  --motion_file data/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.npz \
  --viewer --show_reference --compute_metrics
```

Use `--trajectory_index` for stacked NPZ files. SONIC inputs are resampled to
50 Hz and must contain `joint_pos`, `joint_vel`, and all six body state arrays.
Legacy unnamed files are accepted only when their dimensions match this
repository's G1 exporter. CSV and H1 inputs are not supported by this entrypoint.

`eval_sim2sim_genesis.py` still defaults to `--policy_type beyondmimic`; its
existing `--policy_path` workflow is unchanged. SONIC uses nominal dynamics by
default. Pass `--experiment_config` for explicit Genesis physics changes.

The model weights are subject to the NVIDIA Open Model License. Upstream source
and asset attribution is recorded in `whole_body_tracking/sonic/NOTICE.md` and
the setup command downloads the upstream license files beside the artifacts.

## Motion-specific G1 checkpoint batch

`artifacts/checkpoints/` contains 11 LAFAN1 G1 tracking checkpoints with matching
motion NPZ files. The separate checkpoint batch YAML lists every pair, including
the full `walk1_subject1.npz` clip. Validate or run it from this repository root:

```bash
python scripts/eval_checkpoint_batch_isaac.py \
  --config configs/sonic/isaac_checkpoint_batch_example.yaml --validate_only
python scripts/eval_checkpoint_batch_isaac.py \
  --config configs/sonic/isaac_checkpoint_batch_example.yaml --headless --device cuda:0
```

The batch uses `play.py` checkpoint inference with 100 environments per motion,
frame-zero starts, uniform initial joint noise of ±0.01 rad, and no other domain
randomization. Each environment contributes one full-clip rollout, ending at its
first tracking failure or the last reference frame. A timestamped folder under
`logs/checkpoint_batch/` contains the resolved config, `summary.csv`, per-motion
JSON and process logs, a compressed padded NPZ with `valid_lengths` and initial
states, and an env-0 MP4. Set `record_video: false` to skip MP4 capture.
Failures remain in the summary and make the batch exit nonzero after all motions
have been attempted. Metrics use the G1 task's 14 reference bodies and should
not be compared numerically with SONIC's 30-body metrics.
After a result is saved, the Isaac child gets 10 seconds to close gracefully;
the batch launcher terminates it if it is still alive after 30 seconds and
continues to the next motion. A shutdown timeout is recorded in `summary.csv`.

Both Isaac batch evaluators also report `safety_*` metrics per rollout in JSON
and mean/worst values in `summary.csv`. These cover stance foot-link horizontal
motion, non-support contact events/time/peak force magnitude, hard and soft joint
position-limit margins and violations, and actuator speed/effort limit
utilization. The JSON
lists the bodies with non-support contacts and the worst joint for each limit
measure. Foot contact uses a 10 N net-force threshold; contact on body links
other than the two ankle and two wrist links uses 1 N, matching the tracking
task's undesired-contact convention. Stance foot speed above 0.1 m/s is counted
as slip time. This is an ankle-link motion proxy, not measured sole contact-point
slip. Applied effort is the implicit actuator's estimated, clipped effort. The
speed reference comes from the G1 actuator configuration. Interpret contact
counts for `fallAndGetUp` clips with their intended contact sequence in mind.
