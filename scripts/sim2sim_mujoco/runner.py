"""Single-environment MuJoCo sim2sim rollout runner."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from sim2sim_genesis.metrics import TrackingMetricsEvaluator, compute_metrics_lite
from sim2sim_genesis.observations import ObservationBuilder
from sim2sim_genesis.onnx_policy import OnnxMotionPolicy
from sim2sim_mujoco.config import MujocoEvalConfig, default_video_path
from sim2sim_mujoco.control import MujocoPdController
from sim2sim_mujoco.scene import MujocoSceneAdapter


class MujocoSim2SimRunner:
    """Coordinate ONNX inference, MuJoCo stepping, metrics, and video."""

    def __init__(
        self,
        *,
        config: MujocoEvalConfig,
        policy: OnnxMotionPolicy,
        scene: MujocoSceneAdapter,
        observation_builder: ObservationBuilder,
        controller: MujocoPdController,
        metrics_evaluator: TrackingMetricsEvaluator,
    ):
        self.config = config
        self.policy = policy
        self.scene = scene
        self.observation_builder = observation_builder
        self.controller = controller
        self.metrics_evaluator = metrics_evaluator
        self.obs_dim_expected = self.observation_builder.compute_obs_dim(self.policy.get_obs_input_dim())

        self.start_timestep = int(config.start_timestep)
        self.reference_motion_length_steps = int(policy.reference_motion_length_steps)
        if self.start_timestep < 0 or self.start_timestep >= self.reference_motion_length_steps:
            raise ValueError(
                f"start_timestep={self.start_timestep} out of range for reference length "
                f"{self.reference_motion_length_steps}."
            )
        self.max_steps = self.reference_motion_length_steps - self.start_timestep
        if config.max_steps is not None:
            self.max_steps = min(int(config.max_steps), self.max_steps)
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}.")

    @staticmethod
    def _reference_for_env(reference_batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: value[0].astype(np.float32) for key, value in reference_batch.items()}

    @staticmethod
    def _finite_mean_std(values: list[float]) -> tuple[float, float]:
        array = np.asarray(values, dtype=np.float32)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return float("nan"), float("nan")
        return float(array.mean()), float(array.std())

    def evaluate(self, run_timestamp: str) -> dict[str, Any]:
        """Run one rollout and return a metrics summary."""

        reference0_batch = self.policy.reference_at(self.start_timestep, 1, self.obs_dim_expected)
        reference0 = self._reference_for_env(reference0_batch)
        self.scene.reset_to_reference(reference0)
        self.observation_builder.reset()
        state_batch = self.scene.extract_state_batch()
        observation = self.observation_builder.build(
            state_batch,
            reference0_batch,
            self.scene.anchor_idx,
            self.start_timestep,
            self.obs_dim_expected,
        )

        metric_sums: dict[str, float] = defaultdict(float)
        pred_body_pos: list[np.ndarray] = []
        ref_body_pos: list[np.ndarray] = []
        termination_reason = "completed"
        total_steps = 0

        for step_idx in range(self.max_steps):
            reference_step = self.start_timestep + step_idx
            policy_output = self.policy.run(observation, reference_step)
            action = policy_output["actions"].astype(np.float32)
            reference_batch = {
                "joint_pos": policy_output["joint_pos"].astype(np.float32),
                "joint_vel": policy_output["joint_vel"].astype(np.float32),
                "body_pos_w": policy_output["body_pos_w"].astype(np.float32),
                "body_quat_w": policy_output["body_quat_w"].astype(np.float32),
                "body_lin_vel_w": policy_output["body_lin_vel_w"].astype(np.float32),
                "body_ang_vel_w": policy_output["body_ang_vel_w"].astype(np.float32),
            }
            reference = self._reference_for_env(reference_batch)

            joint_target = self.controller.compute_joint_target(action)
            self.controller.step(joint_target)
            state_batch = self.scene.extract_state_batch()
            state = self.scene.state_for_env(state_batch)
            self.scene.render_frame(reference)

            pred_body_pos.append(state["body_pos_w"].astype(np.float32).copy())
            ref_body_pos.append(reference["body_pos_w"].astype(np.float32).copy())

            step_metrics = self.metrics_evaluator.tracking_metrics(state, reference)
            terminated = bool(step_metrics.pop("terminated"))
            step_reason = str(step_metrics.pop("termination_reason"))
            for key, value in step_metrics.items():
                metric_sums[key] += float(value)
            total_steps += 1

            self.observation_builder.update_last_action(action)
            if terminated:
                termination_reason = step_reason
                break
            if step_idx + 1 >= self.max_steps:
                break

            next_reference_step = reference_step + 1
            next_reference_batch = self.policy.reference_at(next_reference_step, 1, self.obs_dim_expected)
            observation = self.observation_builder.build(
                state_batch,
                next_reference_batch,
                self.scene.anchor_idx,
                next_reference_step,
                self.obs_dim_expected,
            )

        video_path = None
        if self.config.record_video:
            video_path = self.config.video_name or default_video_path(run_timestamp, self.config.policy_path)
            self.scene.save_video(video_path, fps=round(1.0 / self.config.control_dt))

        output: dict[str, Any] = {
            "evaluation_seed": self.config.seed,
            "policy_path": self.config.policy_path,
            "xml_file": self.config.xml_file,
            "steps": total_steps,
            "max_steps": self.max_steps,
            "start_timestep": self.start_timestep,
            "reference_motion_length_steps": self.reference_motion_length_steps,
            "terminated": 1 if termination_reason != "completed" else 0,
            "termination_reason": termination_reason,
            "compute_metrics": int(self.config.compute_metrics),
            "record_video": int(self.config.record_video),
        }
        if video_path is not None:
            output["video_path"] = video_path
        if total_steps > 0:
            for key, value in metric_sums.items():
                output[f"{key}_mean"] = value / total_steps

        if self.config.compute_metrics and pred_body_pos:
            pred = np.stack(pred_body_pos, axis=0)
            ref = np.stack(ref_body_pos, axis=0)
            spacing_offset = pred[0, self.scene.root_idx] - ref[0, self.scene.root_idx]
            pred_aligned = pred - spacing_offset[None, None, :]
            lite_metrics = compute_metrics_lite(pred_aligned, ref, root_idx=self.scene.root_idx)
            metric_name_map = {
                "mpjpe": "mpjpe_g",
                "mpjpe_l": "mpjpe_l",
                "vel_dist": "vel_dist",
                "accel_dist": "accel_dist",
            }
            for output_name, source_name in metric_name_map.items():
                values = lite_metrics[source_name].reshape(-1).astype(np.float32)
                mean, std = self._finite_mean_std(values.tolist())
                output[f"{output_name}_mean"] = mean
                output[f"{output_name}_std"] = std
        return output

