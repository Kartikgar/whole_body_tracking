"""Open-loop replay of Isaac-logged base actions in Genesis."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim2sim_genesis.control import PdController
from sim2sim_genesis.observations import ObservationBuilder
from sim2sim_genesis.onnx_policy import PolicyMeta
from sim2sim_genesis.scene import GenesisSceneAdapter
from sim2sim_genesis.state_action_io import StateActionDataset, load_state_action_npz, select_trajectory_indices


@dataclass(frozen=True)
class ReplayMetrics:
    """Per-trajectory error against Isaac source states."""

    traj_index: int
    valid_length: int
    rmse_abs: float
    rmse_rel: float
    max_abs: float
    max_rel: float
    body_rmse: float | None = None
    body_max: float | None = None


class OpenLoopReplayRunner:
    """Replay logged base actions in Genesis from Isaac initial states."""

    def __init__(
        self,
        dataset: StateActionDataset,
        scene: GenesisSceneAdapter,
        controller: PdController,
        meta: PolicyMeta,
        *,
        trajectory_indices: list[int] | None = None,
        torque_limit: float | None = None,
    ):
        self.dataset = dataset
        self.scene = scene
        self.controller = controller
        self.meta = meta
        self.torque_limit = torque_limit
        self.trajectory_indices = trajectory_indices or list(range(dataset.num_traj))
        self.body_names_for_log = list(dataset.body_names) if dataset.body_names else list(scene.log_body_names)
        self.reset_body_names = list(dataset.body_names) if dataset.body_names else list(scene.log_body_names)
        self.reset_root_body_name = self._resolve_reset_root_body_name()
        self.reset_root_body_index = (
            self.reset_body_names.index(self.reset_root_body_name) if self.reset_body_names else 0
        )

        if len(self.meta.action_scale) != len(self.meta.joint_names):
            raise ValueError("Policy metadata action_scale/joint_names length mismatch.")
        if len(self.meta.default_joint_pos) != len(self.meta.joint_names):
            raise ValueError("Policy metadata default_joint_pos/joint_names length mismatch.")

    def _resolve_reset_root_body_name(self) -> str:
        """Resolve the root body name within the logged body ordering used by the dataset."""

        candidate_lists = [
            list(getattr(self.scene, "log_body_names", [])),
            list(getattr(self.scene, "meta_body_names", [])),
            list(self.dataset.body_names),
        ]
        for names in candidate_lists:
            if 0 <= self.scene.root_idx < len(names):
                candidate = names[self.scene.root_idx]
                if candidate in self.reset_body_names:
                    return candidate
        if self.reset_body_names:
            return self.reset_body_names[0]
        raise ValueError("Could not resolve a root body name for replay reset.")

    def _compute_joint_target(self, raw_action: np.ndarray) -> np.ndarray:
        return self._compute_joint_target_batch(raw_action)

    def _compute_joint_target_batch(self, raw_actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(raw_actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[1] != len(self.meta.joint_names):
            raise ValueError(
                f"Replay action dim {actions.shape[1]} does not match policy joint count {len(self.meta.joint_names)}."
            )
        joint_target = self.meta.default_joint_pos[None, :] + self.meta.action_scale[None, :] * actions
        if self.scene.num_envs == 1 and joint_target.shape[0] == 1:
            return joint_target[0].astype(np.float32)
        return joint_target.astype(np.float32)

    def _reset_trajectory_batch(self, batch_indices: list[int]) -> None:
        initial = self.dataset.initial_states
        self.scene.reset_from_logged_state_batch(
            logged_joint_names=self.dataset.joint_names,
            initial_joint_pos=initial["initial_joint_pos"][batch_indices],
            initial_joint_vel=initial["initial_joint_vel"][batch_indices],
            initial_body_pos_w=initial["initial_body_pos_w"][batch_indices],
            initial_body_quat_w=initial["initial_body_quat_w"][batch_indices],
            initial_body_lin_vel_w=initial["initial_body_lin_vel_w"][batch_indices],
            initial_body_ang_vel_w=initial["initial_body_ang_vel_w"][batch_indices],
            root_body_index=self.reset_root_body_index,
            batch_size=len(batch_indices),
        )

    @staticmethod
    def _max_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
        actual_arr = np.asarray(actual, dtype=np.float32)
        expected_arr = np.asarray(expected, dtype=np.float32)
        return float(np.max(np.abs(actual_arr - expected_arr)))

    def _warn_on_reset_mismatch(self, batch_indices: list[int], state_batch: dict[str, np.ndarray]) -> None:
        """Check whether the reset reproduces the loaded initial state closely enough."""

        active = len(batch_indices)
        if active == 0:
            return

        initial = self.dataset.initial_states
        body_batch = self.scene.extract_body_states_batch(self.reset_body_names, state_batch=state_batch)

        error_summary = {
            "joint_pos": self._max_abs_error(state_batch["joint_pos"][:active], initial["initial_joint_pos"][batch_indices]),
            "joint_vel": self._max_abs_error(state_batch["joint_vel"][:active], initial["initial_joint_vel"][batch_indices]),
            "body_pos_w": self._max_abs_error(body_batch["body_pos_w"][:active], initial["initial_body_pos_w"][batch_indices]),
            "body_quat_w": self._max_abs_error(body_batch["body_quat_w"][:active], initial["initial_body_quat_w"][batch_indices]),
            "body_lin_vel_w": self._max_abs_error(
                body_batch["body_lin_vel_w"][:active], initial["initial_body_lin_vel_w"][batch_indices]
            ),
            "body_ang_vel_w": self._max_abs_error(
                body_batch["body_ang_vel_w"][:active], initial["initial_body_ang_vel_w"][batch_indices]
            ),
        }

        tolerance = 1.0e-4
        failed = {key: value for key, value in error_summary.items() if value > tolerance}
        if failed:
            formatted = ", ".join(f"{key}={value:.6g}" for key, value in failed.items())
            print(
                "[WARN] Replay reset mismatch against loaded initial state: "
                f"{formatted}. root_body='{self.reset_root_body_name}', "
                f"root_body_index={self.reset_root_body_index}."
            )

    def replay_trajectory(self, traj_index: int) -> tuple[dict[str, np.ndarray], ReplayMetrics]:
        """Replay one trajectory and compare against Isaac source joint positions."""

        stacked_list, metrics_list = self.replay_batch([traj_index])
        return stacked_list[0], metrics_list[0]

    def replay_batch(self, batch_indices: list[int]) -> tuple[list[dict[str, np.ndarray]], list[ReplayMetrics]]:
        """Replay a batch of trajectories in parallel across Genesis environments."""

        if len(batch_indices) == 0:
            return [], []

        active = len(batch_indices)
        num_envs = self.scene.num_envs
        if active > num_envs:
            raise ValueError(f"Batch size {active} exceeds scene num_envs={num_envs}.")

        valid_lengths = [int(self.dataset.valid_lengths[traj_index]) for traj_index in batch_indices]
        max_steps = max(valid_lengths)
        self._reset_trajectory_batch(batch_indices)
        self._warn_on_reset_mismatch(batch_indices, self.scene.extract_state_batch())

        replay_keys = (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "actions",
        )
        replay_states = [{key: [] for key in replay_keys} for _ in range(active)]
        last_actions = np.zeros((num_envs, len(self.meta.joint_names)), dtype=np.float32)
        state_batch = self.scene.extract_state_batch()

        for step_idx in range(max_steps):
            actions = np.zeros((num_envs, len(self.meta.joint_names)), dtype=np.float32)
            for env_id, traj_index in enumerate(batch_indices):
                if step_idx < valid_lengths[env_id]:
                    action = self.dataset.replay_actions[traj_index, step_idx].astype(np.float32)
                    actions[env_id] = action
                    last_actions[env_id] = action
                else:
                    actions[env_id] = last_actions[env_id]

            if active < num_envs:
                actions[active:] = last_actions[active - 1 : active]

            if self.body_names_for_log:
                body_states = self.scene.extract_body_states_batch(self.body_names_for_log, state_batch=state_batch)
            else:
                body_states = {
                    "body_pos_w": state_batch["log_body_pos_w"],
                    "body_quat_w": state_batch["log_body_quat_w"],
                    "body_lin_vel_w": state_batch["log_body_lin_vel_w"],
                    "body_ang_vel_w": state_batch["log_body_ang_vel_w"],
                }

            for env_id in range(active):
                if step_idx >= valid_lengths[env_id]:
                    continue
                replay_states[env_id]["joint_pos"].append(state_batch["joint_pos"][env_id])
                replay_states[env_id]["joint_vel"].append(state_batch["joint_vel"][env_id])
                replay_states[env_id]["body_pos_w"].append(body_states["body_pos_w"][env_id])
                replay_states[env_id]["body_quat_w"].append(body_states["body_quat_w"][env_id])
                replay_states[env_id]["body_lin_vel_w"].append(body_states["body_lin_vel_w"][env_id])
                replay_states[env_id]["body_ang_vel_w"].append(body_states["body_ang_vel_w"][env_id])
                replay_states[env_id]["actions"].append(
                    self.dataset.replay_actions[batch_indices[env_id], step_idx].astype(np.float32)
                )

            joint_target = self._compute_joint_target_batch(actions)
            self.controller.step(joint_target)
            state_batch = self.scene.extract_state_batch()

        stacked_list: list[dict[str, np.ndarray]] = []
        metrics_list: list[ReplayMetrics] = []
        for env_id, traj_index in enumerate(batch_indices):
            stacked = {key: np.stack(values, axis=0).astype(np.float32) for key, values in replay_states[env_id].items()}
            traj_metrics = self._compute_metrics(
                traj_index,
                stacked["joint_pos"],
                stacked.get("body_pos_w"),
            )
            stacked_list.append(stacked)
            metrics_list.append(traj_metrics)

        return stacked_list, metrics_list

    def _compute_metrics(
        self,
        traj_index: int,
        replay_joint_pos: np.ndarray,
        replay_body_pos: np.ndarray | None = None,
    ) -> ReplayMetrics:
        valid_length = int(replay_joint_pos.shape[0])
        source_joint_pos = self.dataset.source_states["joint_pos"][traj_index, :valid_length]

        logged_default = self.dataset.default_joint_pos.reshape(1, 1, -1)
        policy_default = self.meta.default_joint_pos.reshape(1, 1, -1)

        if source_joint_pos.shape[-1] == logged_default.shape[-1]:
            source_rel = source_joint_pos - logged_default
            replay_rel = replay_joint_pos - policy_default
        else:
            source_rel = source_joint_pos
            replay_rel = replay_joint_pos

        abs_error = replay_joint_pos - source_joint_pos
        rel_error = replay_rel - source_rel

        body_rmse = None
        body_max = None
        if replay_body_pos is not None and "body_pos_w" in self.dataset.source_states:
            source_body_pos = self.dataset.source_states["body_pos_w"][traj_index, :valid_length]
            if source_body_pos.shape == replay_body_pos.shape:
                body_error = replay_body_pos - source_body_pos
                body_rmse = float(np.sqrt(np.mean(np.square(body_error))))
                body_max = float(np.max(np.abs(body_error)))

        return ReplayMetrics(
            traj_index=traj_index,
            valid_length=valid_length,
            rmse_abs=float(np.sqrt(np.mean(np.square(abs_error)))),
            rmse_rel=float(np.sqrt(np.mean(np.square(rel_error)))),
            max_abs=float(np.max(np.abs(abs_error))),
            max_rel=float(np.max(np.abs(rel_error))),
            body_rmse=body_rmse,
            body_max=body_max,
        )

    def run(
        self,
        *,
        output_npz: str | None = None,
        report_csv: str | None = None,
    ) -> dict[str, Any]:
        """Replay selected trajectories and optionally save outputs."""

        replay_trajectories: list[dict[str, np.ndarray]] = []
        metrics: list[ReplayMetrics] = []

        print(f"[INFO] Replaying {len(self.trajectory_indices)} trajectories")
        num_envs = self.scene.num_envs
        for batch_start in range(0, len(self.trajectory_indices), num_envs):
            print(f"[INFO] Batch start: {batch_start}")
            batch_indices = self.trajectory_indices[batch_start : batch_start + num_envs]
            batch_replays, batch_metrics = self.replay_batch(batch_indices)
            replay_trajectories.extend(batch_replays)
            metrics.extend(batch_metrics)
            for traj_metrics in batch_metrics:
                print(
                    f"[INFO] Traj {traj_metrics.traj_index}: len={traj_metrics.valid_length} "
                    f"rmse_abs={traj_metrics.rmse_abs:.5f} rmse_rel={traj_metrics.rmse_rel:.5f}"
                    + (
                        f" body_rmse={traj_metrics.body_rmse:.5f}"
                        if traj_metrics.body_rmse is not None
                        else ""
                    )
                )

        saved_npz = None
        if output_npz is not None and replay_trajectories:
            saved_npz = self._save_replay_npz(output_npz, replay_trajectories)
            print(f"[INFO] Saved Genesis replay NPZ to {saved_npz}")

        if report_csv is not None and metrics:
            self._write_report_csv(report_csv, metrics)
            print(f"[INFO] Saved replay metrics CSV to {report_csv}")

        summary = {
            "num_trajectories": len(metrics),
            "mean_rmse_abs": float(np.mean([item.rmse_abs for item in metrics])) if metrics else float("nan"),
            "mean_rmse_rel": float(np.mean([item.rmse_rel for item in metrics])) if metrics else float("nan"),
            "output_npz": saved_npz,
            "report_csv": report_csv,
        }
        return summary

    def _save_replay_npz(self, output_path: str, replay_trajectories: list[dict[str, np.ndarray]]) -> str:
        motion_length = max(int(traj["actions"].shape[0]) for traj in replay_trajectories)
        payload: dict[str, np.ndarray] = {
            "fps": np.array([self.dataset.fps], dtype=np.float32),
            "valid_lengths": np.asarray([traj["actions"].shape[0] for traj in replay_trajectories], dtype=np.int32),
            "action_mode": np.array("genesis_base_openloop_replay", dtype=np.object_),
            "joint_names": np.asarray(self.meta.joint_names, dtype=np.object_),
            "default_joint_pos": self.meta.default_joint_pos.astype(np.float32),
            "body_names": np.asarray(self.body_names_for_log, dtype=np.object_),
        }

        keys = (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "actions",
        )
        for key in keys:
            padded = []
            feature_shape = tuple(replay_trajectories[0][key].shape[1:])
            for traj in replay_trajectories:
                seq = traj[key]
                if seq.shape[0] < motion_length:
                    pad = np.repeat(seq[-1:, ...], motion_length - seq.shape[0], axis=0)
                    seq = np.concatenate([seq, pad], axis=0)
                padded.append(seq.astype(np.float32))
            payload[key] = np.stack(padded, axis=0)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savez(output_path, **payload)
        return output_path

    @staticmethod
    def _write_report_csv(output_path: str, metrics: list[ReplayMetrics]) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "traj_index",
                    "valid_length",
                    "rmse_abs",
                    "rmse_rel",
                    "max_abs",
                    "max_rel",
                    "body_rmse",
                    "body_max",
                ],
            )
            writer.writeheader()
            for item in metrics:
                writer.writerow(
                    {
                        "traj_index": item.traj_index,
                        "valid_length": item.valid_length,
                        "rmse_abs": item.rmse_abs,
                        "rmse_rel": item.rmse_rel,
                        "max_abs": item.max_abs,
                        "max_rel": item.max_rel,
                        "body_rmse": item.body_rmse,
                        "body_max": item.body_max,
                    }
                )


def build_runner_from_paths(
    *,
    state_action_npz: str,
    policy_path: str,
    urdf_file: str | None,
    xml_file: str | None,
    backend: str,
    sim_dt: float,
    control_dt: float,
    trajectory_indices_text: str | None,
    seed: int | None,
    torque_limit: float | None,
    num_envs: int = 1,
) -> OpenLoopReplayRunner:
    """Construct an open-loop replay runner from dataset and policy metadata paths."""

    dataset = load_state_action_npz(state_action_npz)
    meta = parse_policy_meta_from_path(policy_path)
    indices = select_trajectory_indices(dataset.num_traj, trajectory_indices_text)
    num_envs = max(int(num_envs), 1)

    scene = GenesisSceneAdapter(
        backend=backend,
        sim_dt=sim_dt,
        num_envs=num_envs,
        viewer=False,
        record_video=False,
        show_reference=False,
        reference_marker_radius=0.03,
        urdf_file=urdf_file,
        xml_file=xml_file,
        meta_body_names=meta.body_names,
        joint_names=meta.joint_names,
        anchor_body_name=meta.anchor_body_name,
        domain_randomization=False,
        seed=seed,
    )
    controller = PdController(
        scene=scene,
        meta=meta,
        observation_builder=ObservationBuilder(
            meta=meta,
            num_envs=num_envs,
            add_noise=False,
            rng=np.random.default_rng(seed),
            motion_file=None,
        ),
        control_dt=control_dt,
        sim_dt=sim_dt,
        torque_limit=torque_limit,
    )
    return OpenLoopReplayRunner(
        dataset=dataset,
        scene=scene,
        controller=controller,
        meta=meta,
        trajectory_indices=indices,
        torque_limit=torque_limit,
    )


def parse_policy_meta_from_path(policy_path: str) -> PolicyMeta:
    from sim2sim_genesis.onnx_policy import load_onnx_metadata, parse_policy_meta

    return parse_policy_meta(load_onnx_metadata(policy_path))
