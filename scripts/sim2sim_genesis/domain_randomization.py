"""Domain-randomization helpers for the modular Genesis sim2sim evaluator."""

from __future__ import annotations

import numpy as np
import torch

from sim2sim_genesis.constants import (
    DOMAIN_RAND_BASE_COM_RANGE,
    DOMAIN_RAND_FRICTION_NUM_BUCKETS,
    DOMAIN_RAND_FRICTION_RANGE,
    DOMAIN_RAND_JOINT_DEFAULT_POS_RANGE,
    DOMAIN_RAND_PUSH_INTERVAL_RANGE_S,
    DOMAIN_RAND_PUSH_VELOCITY_RANGE,
    bucketize_uniform,
)
from sim2sim_genesis.observations import ObservationBuilder
from sim2sim_genesis.scene import GenesisSceneAdapter


class DomainRandomizer:
    """Mirror the legacy Genesis evaluator's startup and interval randomization."""

    def __init__(
        self,
        scene: GenesisSceneAdapter,
        observation_builder: ObservationBuilder,
        rng: np.random.Generator,
        control_dt: float,
        enabled: bool,
    ):
        """Create the domain-randomization helper."""

        self.scene = scene
        self.observation_builder = observation_builder
        self.rng = rng
        self.control_dt = float(control_dt)
        self.enabled = bool(enabled)
        self.push_next_step_by_env = np.full(self.scene.num_envs, np.iinfo(np.int32).max, dtype=np.int32)
        self.push_root_dofs: list[int] = []

    def call_link_randomization(self, method_name: str, values: np.ndarray, link_indices: list[int]) -> bool:
        """Call a batched Genesis link-randomization method if it exists."""

        method = getattr(self.scene.robot, method_name, None)
        if method is None:
            return False

        candidates = [np.asarray(values, dtype=np.float32)]
        if self.scene.num_envs == 1 and candidates[0].ndim >= 2:
            candidates.append(candidates[0][0])

        for candidate in candidates:
            candidate_f32 = np.asarray(candidate, dtype=np.float32)
            for payload in (candidate_f32, torch.from_numpy(candidate_f32)):
                try:
                    method(payload, link_indices)
                    return True
                except Exception:
                    continue
        return False

    def resolve_push_root_dofs(self) -> list[int]:
        """Resolve the six floating-base DoFs used for interval pushes."""

        for root_joint_name in ("root_joint", "floating_base", "base_joint"):
            try:
                joint = self.scene.robot.get_joint(root_joint_name)
                dof_ids = [int(index) for index in joint.dofs_idx_local]
                if len(dof_ids) >= 6:
                    return dof_ids[:6]
            except Exception:
                continue

        n_dofs = int(getattr(self.scene.robot, "n_dofs", len(self.scene.joint_dof_indices) + 6))
        actuated = {int(index) for index in self.scene.joint_dof_indices}
        free_like = [index for index in range(n_dofs) if index not in actuated]
        if len(free_like) >= 6:
            return free_like[:6]
        return []

    def schedule_next_push(self, env_ids: np.ndarray, current_step: int) -> None:
        """Schedule the next random push time for a subset of environments."""

        if env_ids.size == 0:
            return
        low_seconds, high_seconds = DOMAIN_RAND_PUSH_INTERVAL_RANGE_S
        interval_seconds = self.rng.uniform(low_seconds, high_seconds, size=env_ids.size)
        interval_steps = np.maximum(1, np.round(interval_seconds / self.control_dt).astype(np.int32))
        self.push_next_step_by_env[env_ids] = current_step + interval_steps

    def sample_push_velocity(self, count: int) -> np.ndarray:
        """Sample root linear/angular velocities for interval pushes."""

        ranges = DOMAIN_RAND_PUSH_VELOCITY_RANGE
        return np.stack(
            [
                self.rng.uniform(ranges["x"][0], ranges["x"][1], size=count),
                self.rng.uniform(ranges["y"][0], ranges["y"][1], size=count),
                self.rng.uniform(ranges["z"][0], ranges["z"][1], size=count),
                self.rng.uniform(ranges["roll"][0], ranges["roll"][1], size=count),
                self.rng.uniform(ranges["pitch"][0], ranges["pitch"][1], size=count),
                self.rng.uniform(ranges["yaw"][0], ranges["yaw"][1], size=count),
            ],
            axis=1,
        ).astype(np.float32)

    def apply_interval_push(self, rollout_step: int) -> None:
        """Apply scheduled interval pushes to the scene root DoFs."""

        if (not self.enabled) or len(self.push_root_dofs) < 6:
            return

        due_env_ids = np.nonzero(rollout_step >= self.push_next_step_by_env)[0].astype(np.int32)
        if due_env_ids.size == 0:
            return

        push_velocity = self.sample_push_velocity(int(due_env_ids.size))
        try:
            if self.scene.num_envs > 1:
                self.scene.robot.set_dofs_velocity(push_velocity, self.push_root_dofs, envs_idx=due_env_ids)
            else:
                self.scene.robot.set_dofs_velocity(push_velocity[0], self.push_root_dofs)
        except TypeError:
            if self.scene.num_envs > 1:
                self.scene.robot.set_dofs_velocity(push_velocity, self.push_root_dofs, due_env_ids)
            else:
                self.scene.robot.set_dofs_velocity(push_velocity[0], self.push_root_dofs)
        except Exception as exc:
            print(f"[WARN] Failed to apply push randomization. Disabling pushes for this run. Error: {exc}")
            self.push_next_step_by_env[:] = np.iinfo(np.int32).max
            return

        self.schedule_next_push(due_env_ids, current_step=rollout_step)

    def reset_rollout(self) -> None:
        """Reset per-rollout randomization state."""

        if not self.enabled:
            return
        env_ids = np.arange(self.scene.num_envs, dtype=np.int32)
        self.schedule_next_push(env_ids, current_step=0)

    def setup(self) -> None:
        """Apply startup randomization effects that persist across rollouts."""

        if not self.enabled:
            return

        low_joint, high_joint = DOMAIN_RAND_JOINT_DEFAULT_POS_RANGE
        joint_offset = self.rng.uniform(
            low_joint,
            high_joint,
            size=self.observation_builder.default_joint_pos_by_env.shape,
        ).astype(np.float32)
        self.observation_builder.default_joint_pos_by_env = (
            self.observation_builder.default_joint_pos_by_env + joint_offset
        ).astype(np.float32)

        n_links = int(getattr(self.scene.robot, "n_links", 0))
        link_ids = list(range(n_links))
        if n_links > 0:
            low_friction, high_friction = DOMAIN_RAND_FRICTION_RANGE
            friction = self.rng.uniform(
                low_friction, high_friction, size=(self.scene.num_envs, n_links)
            ).astype(np.float32)
            friction = bucketize_uniform(friction, low_friction, high_friction, DOMAIN_RAND_FRICTION_NUM_BUCKETS)
            friction_ok = self.call_link_randomization("set_friction_ratio", friction, link_ids)
            if not friction_ok:
                print(
                    "[WARN] Genesis API does not expose set_friction_ratio for this build; "
                    "skipping physics-material friction randomization."
                )

        try:
            torso_link = self.scene.robot.get_link("torso_link")
            torso_idx_local_raw = getattr(torso_link, "idx_local", None)
            if torso_idx_local_raw is None:
                torso_idx_raw = getattr(torso_link, "idx", None)
                link_start = int(getattr(self.scene.robot, "link_start", 0))
                torso_idx_local_raw = None if torso_idx_raw is None else int(torso_idx_raw) - link_start
            torso_idx_local = -1 if torso_idx_local_raw is None else int(torso_idx_local_raw)
        except Exception:
            torso_idx_local = -1

        if torso_idx_local >= 0:
            com_shift = np.zeros((self.scene.num_envs, 1, 3), dtype=np.float32)
            com_shift[:, 0, 0] = self.rng.uniform(
                DOMAIN_RAND_BASE_COM_RANGE["x"][0], DOMAIN_RAND_BASE_COM_RANGE["x"][1], size=self.scene.num_envs
            )
            com_shift[:, 0, 1] = self.rng.uniform(
                DOMAIN_RAND_BASE_COM_RANGE["y"][0], DOMAIN_RAND_BASE_COM_RANGE["y"][1], size=self.scene.num_envs
            )
            com_shift[:, 0, 2] = self.rng.uniform(
                DOMAIN_RAND_BASE_COM_RANGE["z"][0], DOMAIN_RAND_BASE_COM_RANGE["z"][1], size=self.scene.num_envs
            )
            com_ok = self.call_link_randomization("set_COM_shift", com_shift, [torso_idx_local])
            if not com_ok:
                print(
                    "[WARN] Genesis API does not expose set_COM_shift for this build; "
                    "skipping base COM randomization."
                )
        else:
            print("[WARN] Could not resolve torso_link for COM randomization.")

        self.push_root_dofs = self.resolve_push_root_dofs()
        if len(self.push_root_dofs) < 6:
            print("[WARN] Could not resolve 6 root DoFs for push randomization; interval pushes are disabled.")
        else:
            print("[INFO] Genesis domain randomization enabled (friction, joint defaults, torso COM, interval pushes).")
