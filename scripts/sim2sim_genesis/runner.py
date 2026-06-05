"""Rollout orchestration for the modular Genesis sim2sim evaluator."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from sim2sim_genesis.config import EvalConfig
from sim2sim_genesis.constants import DEFAULT_NEXT_LAB_DATE
from sim2sim_genesis.domain_randomization import DomainRandomizer
from sim2sim_genesis.metrics import TrackingMetricsEvaluator, compute_metrics_lite
from sim2sim_genesis.observations import ObservationBuilder
from sim2sim_genesis.onnx_policy import OnnxMotionPolicy
from sim2sim_genesis.scene import GenesisSceneAdapter
from sim2sim_genesis.trajectory_io import TrajectoryRecorder

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class Sim2SimRunner:
    """Coordinate policy, scene, observations, control, metrics, and outputs."""

    def __init__(
        self,
        config: EvalConfig,
        policy: OnnxMotionPolicy,
        scene: GenesisSceneAdapter,
        observation_builder: ObservationBuilder,
        controller: Any,
        metrics_evaluator: TrackingMetricsEvaluator,
        domain_randomizer: DomainRandomizer,
        trajectory_recorder: TrajectoryRecorder | None = None,
        rng: np.random.Generator | None = None,
    ):
        """Create the rollout runner for the modular evaluator."""

        self.config = config
        self.policy = policy
        self.scene = scene
        self.observation_builder = observation_builder
        self.controller = controller
        self.metrics_evaluator = metrics_evaluator
        self.domain_randomizer = domain_randomizer
        self.trajectory_recorder = trajectory_recorder
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)

        self.obs_dim_expected = self.observation_builder.compute_obs_dim(self.policy.get_obs_input_dim())
        self.start_timestep = int(config.start_timestep)
        self.reference_motion_length_steps = self.policy.reference_motion_length_steps
        if self.start_timestep < 0:
            raise ValueError(f"--start_timestep must be >= 0. Got {self.start_timestep}.")
        if self.start_timestep >= self.reference_motion_length_steps:
            raise ValueError(
                f"--start_timestep={self.start_timestep} is out of range for reference length "
                f"{self.reference_motion_length_steps}."
            )

        self.max_steps = self.reference_motion_length_steps - self.start_timestep
        if config.max_steps is not None and int(config.max_steps) != self.max_steps:
            self.max_steps = int(config.max_steps)
            print(f"max_steps: {self.max_steps}")

    def reference_for_env(self, reference_batch: dict[str, np.ndarray], env_id: int) -> dict[str, np.ndarray]:
        """Extract a single environment slice from a batched reference payload."""

        return {key: value[env_id].astype(np.float32) for key, value in reference_batch.items()}

    def build_summary_metrics(self, values: list[float]) -> tuple[float, float]:
        """Compute a finite-only mean and standard deviation pair."""

        array = np.asarray(values, dtype=np.float32)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return float("nan"), float("nan")
        return float(array.mean()), float(array.std())

    def evaluate(self, run_timestamp: str) -> dict[str, Any]:
        """Run one or more rollouts and return a single summary dictionary."""

        self.scene.start_recording()

        metric_sums: dict[str, float] = defaultdict(float)
        rollout_metric_values_all: dict[str, list[float]] = defaultdict(list)
        rollout_metric_values_succ: dict[str, list[float]] = defaultdict(list)
        termination_reason = "completed"
        total_steps = 0
        rollout_count = 0
        metrics_enabled = self.config.compute_metrics
        metric_target_envs = self.config.metric_num_envs if metrics_enabled else 0
        metric_envs_done = 0
        success_metric_envs = 0
        record_motion_enabled = self.trajectory_recorder is not None and self.config.record_motion
        target_for_npz = self.config.target_trajectories if record_motion_enabled else 0

        traj_pbar = None
        metrics_pbar = None
        step_pbar = None
        target_rollouts = max(target_for_npz, metric_target_envs)
        estimated_rollouts = (target_rollouts + self.config.num_envs - 1) // self.config.num_envs if target_rollouts > 0 else 0
        if record_motion_enabled and target_for_npz > 0 and tqdm is not None:
            traj_pbar = tqdm(total=target_for_npz, desc="Collect trajectories", unit="traj", leave=True, dynamic_ncols=True)
        if metrics_enabled and metric_target_envs > 0 and tqdm is not None:
            metrics_pbar = tqdm(total=metric_target_envs, desc="Compute metrics", unit="env", leave=True, dynamic_ncols=True)
        if tqdm is not None:
            step_total = estimated_rollouts * self.max_steps if estimated_rollouts > 0 else None
            step_pbar = tqdm(
                total=step_total,
                desc="Simulate rollouts",
                unit="step",
                leave=True,
                dynamic_ncols=True,
                mininterval=0.5,
            )

        while True:
            active_rollout_index = rollout_count + 1
            reference0_batch = self.policy.reference_at(self.start_timestep, self.config.num_envs, self.obs_dim_expected)
            reference0 = self.reference_for_env(reference0_batch, env_id=0)
            joint_qpos_noise_range = (
                self.config.startup_qpos_joint_range if self.config.randomize_startup_qpos else None
            )
            self.scene.reset_to_reference(
                reference0,
                rng=self.rng if self.config.randomize_startup_qpos else None,
                joint_qpos_noise_range=joint_qpos_noise_range,
            )
            self.observation_builder.reset()
            self.domain_randomizer.reset_rollout()
            self.scene.update_reference_markers(reference0)
            state_batch = self.scene.extract_state_batch()
            observation = self.observation_builder.build(
                state_batch,
                reference0_batch,
                self.scene.anchor_idx,
                self.start_timestep,
                self.obs_dim_expected,
            )

            steps_this_rollout = 0
            rollout_body_pos_pred: list[np.ndarray] = []
            rollout_body_pos_gt: list[np.ndarray] = []
            env_done = np.zeros(self.config.num_envs, dtype=bool)
            env_end_step = np.full(self.config.num_envs, self.max_steps, dtype=np.int32)

            for t_step in range(self.max_steps):
                reference_step = self.start_timestep + t_step
                policy_output = self.policy.run(observation, reference_step)
                action_batch = policy_output["actions"].astype(np.float32)
                reference_batch = {
                    "joint_pos": policy_output["joint_pos"].astype(np.float32),
                    "joint_vel": policy_output["joint_vel"].astype(np.float32),
                    "body_pos_w": policy_output["body_pos_w"].astype(np.float32),
                    "body_quat_w": policy_output["body_quat_w"].astype(np.float32),
                    "body_lin_vel_w": policy_output["body_lin_vel_w"].astype(np.float32),
                    "body_ang_vel_w": policy_output["body_ang_vel_w"].astype(np.float32),
                }
                reference = self.reference_for_env(reference_batch, env_id=0)
                self.scene.update_reference_markers(reference)

                joint_target = self.controller.compute_joint_target(action_batch, reference_step)
                self.domain_randomizer.apply_interval_push(t_step)
                self.controller.step(joint_target)
                state_batch = self.scene.extract_state_batch()
                if self.trajectory_recorder is not None:
                    self.trajectory_recorder.append_step(state_batch, action_batch)
                state = self.scene.state_for_env(state_batch, env_id=0)

                if metrics_enabled:
                    rollout_body_pos_pred.append(state_batch["body_pos_w"].astype(np.float32).copy())
                    rollout_body_pos_gt.append(reference_batch["body_pos_w"].astype(np.float32).copy())
                    for env_id in range(self.config.num_envs):
                        if env_done[env_id]:
                            continue
                        env_state = self.scene.state_for_env(state_batch, env_id=env_id)
                        env_reference = self.reference_for_env(reference_batch, env_id=env_id)
                        body_pos_relative, _ = self.metrics_evaluator.compute_body_relative_targets(
                            env_state, env_reference
                        )
                        env_terminated, env_reason = self.metrics_evaluator.termination_check(
                            env_state, env_reference, body_pos_relative
                        )
                        if env_terminated:
                            env_done[env_id] = True
                            env_end_step[env_id] = t_step + 1
                            if env_id == 0:
                                termination_reason = env_reason

                step_metrics = self.metrics_evaluator.tracking_metrics(state, reference)
                terminated = bool(step_metrics.pop("terminated"))
                step_reason = str(step_metrics.pop("termination_reason"))
                if not metrics_enabled or terminated:
                    termination_reason = step_reason
                for key, value in step_metrics.items():
                    metric_sums[key] += float(value)

                self.observation_builder.update_last_action(action_batch)
                total_steps += 1
                steps_this_rollout += 1
                if step_pbar is not None:
                    step_pbar.update(1)
                    if (
                        t_step == 0
                        or (t_step + 1) % 10 == 0
                        or (not metrics_enabled and terminated)
                        or (metrics_enabled and bool(np.all(env_done)))
                        or t_step + 1 >= self.max_steps
                    ):
                        step_pbar.set_postfix(
                            rollout=active_rollout_index,
                            rollout_step=f"{t_step + 1}/{self.max_steps}",
                            traj_saved=self.trajectory_recorder.collected_count if self.trajectory_recorder else 0,
                            done_envs=int(env_done.sum()),
                            refresh=False,
                        )

                self.scene.render_camera(state["body_pos_w"][self.scene.root_idx])

                if not metrics_enabled and terminated:
                    break
                if t_step + 1 >= self.max_steps:
                    break
                if metrics_enabled and bool(np.all(env_done)):
                    break

                next_reference_step = reference_step + 1
                next_reference_batch = self.policy.reference_at(
                    next_reference_step, self.config.num_envs, self.obs_dim_expected
                )
                observation = self.observation_builder.build(
                    state_batch,
                    next_reference_batch,
                    self.scene.anchor_idx,
                    next_reference_step,
                    self.obs_dim_expected,
                )

            rollout_id = rollout_count
            rollout_count += 1

            if metrics_enabled and len(rollout_body_pos_pred) > 0 and metric_envs_done < metric_target_envs:
                pred_pos_rollout = np.stack(rollout_body_pos_pred, axis=0)
                gt_pos_rollout = np.stack(rollout_body_pos_gt, axis=0)
                for env_id in range(self.config.num_envs):
                    if metric_envs_done >= metric_target_envs:
                        break
                    traj_len = int(np.clip(env_end_step[env_id], 1, pred_pos_rollout.shape[0]))
                    pred_pos = pred_pos_rollout[:traj_len, env_id]
                    gt_pos = gt_pos_rollout[:traj_len, env_id]
                    spacing_offset = pred_pos[0, self.scene.root_idx] - gt_pos[0, self.scene.root_idx]
                    pred_pos_aligned = pred_pos - spacing_offset[None, None, :]
                    lite_metrics = compute_metrics_lite(pred_pos_aligned, gt_pos, root_idx=self.scene.root_idx)
                    is_success = int(env_end_step[env_id]) >= max(self.max_steps - 5, 1)
                    env_metrics = {
                        "mpjpe": float(lite_metrics["mpjpe_g"].mean()) if lite_metrics["mpjpe_g"].size > 0 else float("nan"),
                        "mpjpe_l": float(lite_metrics["mpjpe_l"].mean()) if lite_metrics["mpjpe_l"].size > 0 else float("nan"),
                        "accel_dist": float(lite_metrics["accel_dist"].mean())
                        if lite_metrics["accel_dist"].size > 0
                        else float("nan"),
                        "vel_dist": float(lite_metrics["vel_dist"].mean()) if lite_metrics["vel_dist"].size > 0 else float("nan"),
                    }
                    for key, value in env_metrics.items():
                        rollout_metric_values_all[key].append(value)
                        if is_success:
                            rollout_metric_values_succ[key].append(value)
                    if is_success:
                        success_metric_envs += 1
                    metric_envs_done += 1
                    if metrics_pbar is not None:
                        metrics_pbar.update(1)

            if self.trajectory_recorder is not None:
                gained = self.trajectory_recorder.finalize_rollout()
                if gained > 0 and traj_pbar is not None and target_for_npz > 0:
                    remaining = max(target_for_npz - int(traj_pbar.n), 0)
                    traj_pbar.update(min(gained, remaining))

            if step_pbar is not None:
                step_pbar.set_postfix(
                    rollout=rollout_id + 1,
                    rollout_step=f"{steps_this_rollout}/{self.max_steps}",
                    traj_saved=self.trajectory_recorder.collected_count if self.trajectory_recorder else 0,
                    done_envs=int(env_done.sum()),
                    refresh=False,
                )

            if metrics_enabled and metric_envs_done >= metric_target_envs:
                break
            if self.trajectory_recorder is None and not metrics_enabled:
                break
            if self.trajectory_recorder is not None and self.trajectory_recorder.has_reached_target():
                break
            if steps_this_rollout <= 0:
                break

        if self.scene.camera is not None:
            filename = self.config.video_name or f"logs/sim2sim_eval/{DEFAULT_NEXT_LAB_DATE}/{run_timestamp}_genesis_eval.mp4"
            self.scene.stop_recording(filename=filename, fps=round(1.0 / self.config.control_dt))
        if step_pbar is not None:
            step_pbar.close()
        if traj_pbar is not None:
            traj_pbar.close()
        if metrics_pbar is not None:
            metrics_pbar.close()

        motion_npz_path = self.trajectory_recorder.save() if self.trajectory_recorder is not None else None
        output = {
            "evaluation_seed": self.config.seed,
            "policy_path": self.config.policy_path,
            "motion_file": self.config.motion_file,
            "num_envs": self.config.num_envs,
            "domain_randomization": int(self.config.domain_randomization),
            "randomize_startup_qpos": int(self.config.randomize_startup_qpos),
            "startup_qpos_joint_range_low": self.config.startup_qpos_joint_range[0],
            "startup_qpos_joint_range_high": self.config.startup_qpos_joint_range[1],
            "steps": total_steps,
            "max_steps": self.max_steps,
            "start_timestep": self.start_timestep,
            "reference_motion_length_steps": self.reference_motion_length_steps,
            "rollouts": rollout_count,
            "motion_completion_pct": (
                100.0 * total_steps / (self.max_steps * rollout_count)
                if self.max_steps > 0 and rollout_count > 0
                else 0.0
            ),
            "terminated": 1 if termination_reason != "completed" else 0,
            "termination_reason": termination_reason,
            "collected_trajectories": self.trajectory_recorder.collected_count if self.trajectory_recorder else 0,
            "target_trajectories": target_for_npz,
            "compute_metrics": int(metrics_enabled),
        }
        if motion_npz_path is not None:
            output["output_motion_npz"] = motion_npz_path
        if total_steps > 0:
            for key, value in metric_sums.items():
                output[f"{key}_mean"] = value / total_steps
        if metrics_enabled:
            output["metric_num_envs_target"] = metric_target_envs
            output["metric_num_envs_done"] = metric_envs_done
            output["success"] = (
                100.0 * float(success_metric_envs) / float(metric_envs_done) if metric_envs_done > 0 else 0.0
            )
            output["success_envs"] = success_metric_envs
            for metric_name in ("mpjpe", "mpjpe_l", "accel_dist", "vel_dist"):
                mean_all, std_all = self.build_summary_metrics(rollout_metric_values_all[metric_name])
                mean_succ, std_succ = self.build_summary_metrics(rollout_metric_values_succ[metric_name])
                output[f"{metric_name}_mean"] = mean_all
                output[f"{metric_name}_std"] = std_all
                output[f"{metric_name}_succ_mean"] = mean_succ
                output[f"{metric_name}_succ_std"] = std_succ
        return output
