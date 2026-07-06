"""MuJoCo scene adapter for single-environment sim2sim evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim2sim_mujoco.constants import DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH


class MujocoSceneAdapter:
    """Own MuJoCo model/data, state extraction, reset, and optional video rendering."""

    num_envs = 1

    def __init__(
        self,
        *,
        xml_file: str,
        sim_dt: float,
        meta_body_names: list[str],
        joint_names: list[str],
        anchor_body_name: str,
        record_video: bool,
        show_reference: bool,
        reference_marker_radius: float,
        width: int = DEFAULT_VIDEO_WIDTH,
        height: int = DEFAULT_VIDEO_HEIGHT,
    ):
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError(
                "MuJoCo is required for this evaluator. Install it with `python -m pip install mujoco`."
            ) from exc

        self.mujoco = mujoco
        self.xml_file = xml_file
        self.sim_dt = float(sim_dt)
        self.meta_body_names = list(meta_body_names)
        self.joint_names = list(joint_names)
        self.anchor_body_name = anchor_body_name
        self.record_video = bool(record_video)
        self.show_reference = bool(show_reference)
        self.reference_marker_radius = float(reference_marker_radius)
        self.width = int(width)
        self.height = int(height)

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)

        self.joint_qpos_indices: list[int] = []
        self.joint_dof_indices: list[int] = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise KeyError(f"Joint '{joint_name}' from ONNX metadata is missing in MuJoCo model.")
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise RuntimeError(f"Only 1-DoF hinge policy joints are supported. Got '{joint_name}'.")
            self.joint_qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            self.joint_dof_indices.append(int(self.model.jnt_dofadr[joint_id]))

        self.body_ids: list[int] = []
        for body_name in self.meta_body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise KeyError(f"Body '{body_name}' from ONNX metadata is missing in MuJoCo model.")
            self.body_ids.append(int(body_id))

        if anchor_body_name not in self.meta_body_names:
            raise KeyError(f"Anchor body '{anchor_body_name}' is not present in ONNX body_names metadata.")
        self.anchor_idx = self.meta_body_names.index(anchor_body_name)
        self.root_idx = 0
        self.root_body_id = self.body_ids[self.root_idx]
        self.free_joint_qpos_addr, self.free_joint_dof_addr = self._resolve_free_joint_addresses()

        self.renderer = None
        self.camera = None
        self.frames: list[np.ndarray] = []
        if self.record_video:
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            # Match the Genesis evaluator's follow-camera geometry:
            # camera pos ~= root + (2.5, 0.0, 1.6), lookat ~= root + (0.0, 0.0, 0.3).
            self.camera.azimuth = 90.0
            self.camera.elevation = -27.5
            self.camera.distance = 2.82
            self.camera.lookat[:] = np.array([0.0, 0.0, 0.8], dtype=np.float32)

    def _resolve_free_joint_addresses(self) -> tuple[int, int]:
        free_joint_type = self.mujoco.mjtJoint.mjJNT_FREE
        free_joint_ids = np.where(self.model.jnt_type == free_joint_type)[0]
        if free_joint_ids.size == 0:
            raise RuntimeError("MuJoCo model must contain a freejoint for the floating base.")
        joint_id = int(free_joint_ids[0])
        return int(self.model.jnt_qposadr[joint_id]), int(self.model.jnt_dofadr[joint_id])

    def reset_to_reference(self, reference: dict[str, np.ndarray]) -> None:
        """Teleport the robot to the first reference pose and zero velocities."""

        root_position = np.asarray(reference["body_pos_w"][self.root_idx], dtype=np.float64)
        root_quaternion = np.asarray(reference["body_quat_w"][self.root_idx], dtype=np.float64)
        joint_positions = np.asarray(reference["joint_pos"], dtype=np.float64).reshape(-1)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        root_qpos = self.free_joint_qpos_addr
        self.data.qpos[root_qpos : root_qpos + 3] = root_position
        self.data.qpos[root_qpos + 3 : root_qpos + 7] = root_quaternion
        for joint_index, qpos_index in enumerate(self.joint_qpos_indices):
            self.data.qpos[qpos_index] = joint_positions[joint_index]
        self.mujoco.mj_forward(self.model, self.data)

    def extract_state_batch(self) -> dict[str, np.ndarray]:
        """Read a Genesis-compatible single-env state batch from MuJoCo."""

        joint_pos = self.data.qpos[self.joint_qpos_indices].astype(np.float32)[None, :]
        joint_vel = self.data.qvel[self.joint_dof_indices].astype(np.float32)[None, :]

        body_pos = self.data.xpos[self.body_ids].astype(np.float32)
        body_quat = self.data.xquat[self.body_ids].astype(np.float32)
        body_lin_vel, body_ang_vel = self._extract_body_velocities()

        root_quat = body_quat[self.root_idx]
        root_lin_vel = body_lin_vel[self.root_idx]
        root_ang_vel = body_ang_vel[self.root_idx]

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_pos_w": body_pos[None, :, :],
            "body_quat_w": body_quat[None, :, :],
            "body_lin_vel_w": body_lin_vel[None, :, :],
            "body_ang_vel_w": body_ang_vel[None, :, :],
            "root_quat_w": root_quat[None, :],
            "root_lin_vel_w": root_lin_vel[None, :],
            "root_ang_vel_w": root_ang_vel[None, :],
        }

    def _extract_body_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        body_lin_vel = []
        body_ang_vel = []
        for body_id in self.body_ids:
            spatial_velocity = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_objectVelocity(
                self.model,
                self.data,
                self.mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                spatial_velocity,
                0,
            )
            body_ang_vel.append(spatial_velocity[:3].astype(np.float32))
            body_lin_vel.append(spatial_velocity[3:].astype(np.float32))
        return np.stack(body_lin_vel, axis=0), np.stack(body_ang_vel, axis=0)

    def state_for_env(self, state_batch: dict[str, np.ndarray], env_id: int = 0) -> dict[str, np.ndarray]:
        """Extract a single-env state dictionary."""

        if env_id != 0:
            raise ValueError("MuJoCo v0 supports only env_id=0.")
        return {key: value[0] for key, value in state_batch.items()}

    def render_frame(self, reference: dict[str, np.ndarray] | None = None) -> None:
        """Render one video frame with optional reference body markers."""

        if self.renderer is None:
            return
        root_position = np.asarray(self.data.xpos[self.root_body_id], dtype=np.float32)
        self.camera.lookat[:] = root_position + np.array([0.0, 0.0, 0.3], dtype=np.float32)
        self.camera.azimuth = 90.0
        self.camera.elevation = -27.5
        self.camera.distance = 2.82
        self.renderer.update_scene(self.data, camera=self.camera)
        if self.show_reference and reference is not None:
            self._add_reference_markers(reference)
        self.frames.append(self.renderer.render().copy())

    def _add_reference_markers(self, reference: dict[str, np.ndarray]) -> None:
        positions = np.asarray(reference["body_pos_w"], dtype=np.float32).reshape(-1, 3)
        scene = self.renderer.scene
        for position in positions:
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            self.mujoco.mjv_initGeom(
                geom,
                self.mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([self.reference_marker_radius, 0.0, 0.0], dtype=np.float64),
                position.astype(np.float64),
                np.eye(3, dtype=np.float64).reshape(-1),
                np.array([1.0, 1.0, 1.0, 0.85], dtype=np.float32),
            )
            scene.ngeom += 1

    def save_video(self, filename: str, fps: int) -> str | None:
        """Save recorded frames as MP4."""

        if not self.frames:
            return None
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise ImportError(
                "Video recording requires imageio with ffmpeg support. Install with `python -m pip install imageio[ffmpeg]`."
            ) from exc

        import os

        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        iio.imwrite(filename, np.asarray(self.frames, dtype=np.uint8), fps=fps)
        return filename

    def close(self) -> None:
        """Release MuJoCo renderer resources."""

        if self.renderer is not None:
            close = getattr(self.renderer, "close", None)
            if close is not None:
                close()
