"""Batched MuJoCo sim2sim rollout runner (one rollout per env, Genesis-compatible per-rollout metrics)."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim2sim_genesis.metrics import TrackingMetricsEvaluator, compute_metrics_lite
from sim2sim_genesis.observations import ObservationBuilder
from sim2sim_genesis.onnx_policy import OnnxMotionPolicy
from sim2sim_mujoco.config import MujocoEvalConfig, default_video_path
from sim2sim_mujoco.control import MujocoPdController
from sim2sim_mujoco.domain_randomization import MujocoDomainRandomizer
from sim2sim_mujoco.scene import MujocoSceneAdapter

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

REFERENCE_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


class MujocoSim2SimRunner:
    """Coordinate batched ONNX inference, parallel MuJoCo stepping, DR, termination and metrics."""

    def __init__(
        self,
        *,
        config: MujocoEvalConfig,
        policy: OnnxMotionPolicy,
        scene: MujocoSceneAdapter,
        observation_builder: ObservationBuilder,
        controller: MujocoPdController,
        metrics_evaluator: TrackingMetricsEvaluator,
        domain_randomizer: MujocoDomainRandomizer | None,
        rng: np.random.Generator,
    ):
        self.config, self.policy, self.scene = config, policy, scene
        self.observation_builder, self.controller = observation_builder, controller
        self.metrics_evaluator, self.domain_randomizer, self.rng = metrics_evaluator, domain_randomizer, rng
        self.num_envs = scene.num_envs
        self.obs_dim_expected = observation_builder.compute_obs_dim(policy.get_obs_input_dim())
        self.start_timestep = int(config.start_timestep)
        self.reference_motion_length_steps = int(policy.reference_motion_length_steps)
        if not 0 <= self.start_timestep < self.reference_motion_length_steps:
            raise ValueError(f"start_timestep={self.start_timestep} out of range ({self.reference_motion_length_steps}).")
        self.max_steps = self.reference_motion_length_steps - self.start_timestep
        if config.max_steps is not None:
            self.max_steps = min(int(config.max_steps), self.max_steps)
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}.")

    def evaluate(self, run_timestamp: str) -> dict[str, Any]:
        n = self.num_envs
        dr_params = self.domain_randomizer.setup() if self.domain_randomizer is not None else [{}] * n

        reference0 = self.policy.reference_at(self.start_timestep, n, self.obs_dim_expected)
        self.scene.reset_to_reference(
            {key: value[0] for key, value in reference0.items()},
            rng=self.rng if self.config.randomize_startup_qpos else None,
            joint_qpos_noise_range=self.config.startup_qpos_joint_range if self.config.randomize_startup_qpos else None,
        )
        self.observation_builder.reset()
        if self.domain_randomizer is not None:
            self.domain_randomizer.reset_rollout()
        state_batch = self.scene.extract_state_batch()
        observation = self.observation_builder.build(
            state_batch, reference0, self.scene.anchor_idx, self.start_timestep, self.obs_dim_expected)

        done = np.zeros(n, dtype=bool)
        end_step = np.full(n, self.max_steps, dtype=np.int32)
        reasons = ["completed"] * n
        pred_pos, ref_pos = [], []
        pbar = tqdm(total=self.max_steps, desc="MuJoCo rollout", unit="step", dynamic_ncols=True) if tqdm else None

        for t_step in range(self.max_steps):
            reference_step = self.start_timestep + t_step
            output = self.policy.run(observation, reference_step)
            action = output["actions"].astype(np.float32)
            reference_batch = {key: output[key].astype(np.float32) for key in REFERENCE_KEYS}

            targets = self.controller.compute_joint_target(action)
            if self.domain_randomizer is not None:
                self.domain_randomizer.apply_interval_push(t_step, ~done)
            self.scene.step(targets, active=~done)
            state_batch = self.scene.extract_state_batch()
            self.scene.render_frame(reference_batch)
            if self.config.compute_metrics:
                pred_pos.append(state_batch["body_pos_w"].copy())
                ref_pos.append(reference_batch["body_pos_w"].copy())

            for env_id in np.flatnonzero(~done):
                env_state = self.scene.state_for_env(state_batch, env_id)
                env_reference = {key: value[env_id] for key, value in reference_batch.items()}
                rel_pos, _ = self.metrics_evaluator.compute_body_relative_targets(env_state, env_reference)
                terminated, reason = self.metrics_evaluator.termination_check(env_state, env_reference, rel_pos)
                if terminated:
                    done[env_id], end_step[env_id], reasons[env_id] = True, t_step + 1, reason

            self.observation_builder.update_last_action(action)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(done_envs=int(done.sum()), refresh=False)
            if done.all() or t_step + 1 >= self.max_steps:
                break
            next_reference = self.policy.reference_at(reference_step + 1, n, self.obs_dim_expected)
            observation = self.observation_builder.build(
                state_batch, next_reference, self.scene.anchor_idx, reference_step + 1, self.obs_dim_expected)
        if pbar is not None:
            pbar.close()

        rollouts = []
        for env_id in range(n):
            row: dict[str, Any] = dict(env_id=env_id, survived_steps=int(end_step[env_id]),
                                       completed=bool(not done[env_id]), termination_reason=reasons[env_id],
                                       dr_params=dr_params[env_id])
            if self.config.compute_metrics and pred_pos:
                length = int(np.clip(end_step[env_id], 1, len(pred_pos)))
                pred = np.stack([p[env_id] for p in pred_pos[:length]])
                ref = np.stack([r[env_id] for r in ref_pos[:length]])
                pred = pred - (pred[0, self.scene.root_idx] - ref[0, self.scene.root_idx])[None, None, :]
                lite = compute_metrics_lite(pred, ref, root_idx=self.scene.root_idx)
                for name, source in (("mpjpe", "mpjpe_g"), ("mpjpe_l", "mpjpe_l"), ("vel_dist", "vel_dist"),
                                     ("accel_dist", "accel_dist")):
                    row[name] = float(lite[source].mean()) if lite[source].size else float("nan")
            rollouts.append(row)

        video_path = None
        if self.config.record_video:
            video_path = self.config.video_name or default_video_path(run_timestamp, self.config.policy_path)
            video_path = self.scene.save_video(video_path, fps=round(1.0 / self.config.control_dt))

        survived = np.array([r["survived_steps"] for r in rollouts])
        result: dict[str, Any] = {
            "simulator": "mujoco",
            "evaluation_seed": self.config.seed,
            "policy_path": self.config.policy_path,
            "xml_file": self.config.xml_file,
            "num_envs": n,
            "max_steps": self.max_steps,
            "start_timestep": self.start_timestep,
            "control_dt": self.config.control_dt,
            "sim_dt": self.config.sim_dt,
            "completion_rate": float(np.mean([r["completed"] for r in rollouts])),
            "clip_survival_pct": float(100.0 * survived.sum() / (n * self.max_steps)),
            "domain_randomization": int(self.config.domain_randomization),
            "domain_randomization_config": self.config.dr.to_dict() if self.config.domain_randomization else None,
            "randomize_startup_qpos": int(self.config.randomize_startup_qpos),
            "startup_qpos_joint_range": list(self.config.startup_qpos_joint_range),
            "termination_definition": self.metrics_evaluator.termination_definition(),
            "rollout_metrics": rollouts,
        }
        if self.config.compute_metrics:
            for name in ("mpjpe", "mpjpe_l", "vel_dist", "accel_dist"):
                values = np.array([r[name] for r in rollouts], dtype=np.float64)
                result[f"{name}_mean"] = float(np.nanmean(values))
        if video_path is not None:
            result["video_path"] = video_path
        return result
