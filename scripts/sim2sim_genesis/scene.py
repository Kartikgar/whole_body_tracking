"""Genesis scene management for the modular sim2sim evaluator."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sim2sim_genesis.constants import G1_ALL_BODY_JOINT_NAMES, resolve_urdf_child_link_names


def add_robot_entity(gs_module: Any, scene: Any, urdf_file: str | None, xml_file: str | None):
    """Create a Genesis robot entity, preferring URDF and falling back to MJCF."""

    if urdf_file is not None:
        urdf_constructor = getattr(gs_module.morphs, "URDF", None)
        if urdf_constructor is None:
            raise RuntimeError("Genesis build does not expose gs.morphs.URDF. Provide --xml_file instead.")

        candidate_kwargs = [
            {"file": urdf_file, "pos": (0.0, 0.0, 0.0)},
            {"file": urdf_file},
            {"urdf_file": urdf_file, "pos": (0.0, 0.0, 0.0)},
            {"urdf_file": urdf_file},
            {"urdf_path": urdf_file, "pos": (0.0, 0.0, 0.0)},
            {"urdf_path": urdf_file},
        ]
        last_error = None
        for kwargs in candidate_kwargs:
            try:
                return scene.add_entity(urdf_constructor(**kwargs))
            except TypeError as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Could not construct Genesis URDF morph for '{urdf_file}': {last_error}")

    if xml_file is not None:
        return scene.add_entity(
            gs_module.morphs.MJCF(
                file=xml_file,
                pos=(0.0, 0.0, 0.0),
                default_armature=0.0,
            )
        )

    raise ValueError("Provide either --urdf_file or --xml_file.")


class GenesisSceneAdapter:
    """Own Genesis scene setup, robot reset, state extraction, and visualization."""

    def __init__(
        self,
        backend: str,
        sim_dt: float,
        num_envs: int,
        viewer: bool,
        record_video: bool,
        show_reference: bool,
        reference_marker_radius: float,
        urdf_file: str | None,
        xml_file: str | None,
        meta_body_names: list[str],
        joint_names: list[str],
        anchor_body_name: str,
        domain_randomization: bool,
        seed: int | None = None,
    ):
        """Initialize Genesis and build the G1 scene for evaluation."""

        try:
            import genesis as gs
        except ImportError as exc:
            raise ImportError(
                "Genesis is required to run the modular sim2sim evaluator. "
                "Install Genesis in the current Python environment."
            ) from exc

        self.gs = gs
        self.backend = backend
        self.sim_dt = sim_dt
        self.num_envs = max(int(num_envs), 1)
        self.viewer = bool(viewer)
        self.record_video = bool(record_video)
        self.show_reference = bool(show_reference)
        self.reference_marker_radius = reference_marker_radius
        self.meta_body_names = list(meta_body_names)
        self.joint_names = list(joint_names)
        self.anchor_body_name = anchor_body_name
        self.domain_randomization = bool(domain_randomization)
        self.seed = seed

        gs_backend = gs.gpu if backend == "gpu" else gs.cpu
        if seed is not None:
            try:
                gs.init(backend=gs_backend, precision="32", logging_level="warning", seed=seed)
            except TypeError:
                gs.init(backend=gs_backend, precision="32", logging_level="warning")
        else:
            gs.init(backend=gs_backend, precision="32", logging_level="warning")

        rigid_options_kwargs: dict[str, Any] = {"enable_self_collision": False}
        if self.domain_randomization:
            rigid_options_kwargs.update({"batch_dofs_info": True, "batch_links_info": True})
        try:
            rigid_options = gs.options.RigidOptions(**rigid_options_kwargs)
        except TypeError:
            rigid_options_kwargs.pop("batch_dofs_info", None)
            rigid_options_kwargs.pop("batch_links_info", None)
            rigid_options = gs.options.RigidOptions(**rigid_options_kwargs)
            if self.domain_randomization:
                print(
                    "[WARN] Current Genesis build does not accept batch_dofs_info/batch_links_info; "
                    "domain randomization coverage may be reduced."
                )

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=sim_dt, substeps=1, gravity=(0.0, 0.0, -9.81)),
            rigid_options=rigid_options,
            show_viewer=self.viewer,
            renderer=gs.renderers.Rasterizer(),
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = add_robot_entity(gs, self.scene, urdf_file=urdf_file, xml_file=xml_file)

        self.reference_markers: list[Any] = []
        if self.show_reference:
            for _ in self.meta_body_names:
                marker = self.scene.add_entity(
                    gs.morphs.Sphere(radius=self.reference_marker_radius, collision=False, fixed=True),
                    surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0)),
                )
                self.reference_markers.append(marker)

        self.camera = None
        if self.record_video:
            self.camera = self.scene.add_camera(
                res=(1280, 720),
                pos=(3.0, 0.0, 2.0),
                lookat=(0.0, 0.0, 0.8),
                fov=45,
                GUI=False,
            )

        if self.num_envs > 1:
            self.scene.build(n_envs=self.num_envs, env_spacing=(2.5, 2.5))
        else:
            self.scene.build()

        self.joint_qs_indices: list[int] = []
        self.joint_dof_indices: list[int] = []
        for joint_name in self.joint_names:
            joint = self.robot.get_joint(joint_name)
            if len(joint.qs_idx_local) != 1 or len(joint.dofs_idx_local) != 1:
                raise RuntimeError(f"Only 1-DoF joints are supported. Joint '{joint_name}' has multiple DoFs.")
            self.joint_qs_indices.append(int(joint.qs_idx_local[0]))
            self.joint_dof_indices.append(int(joint.dofs_idx_local[0]))

        self.body_links = [self.robot.get_link(name) for name in self.meta_body_names]
        self.log_body_names = list(self.meta_body_names)
        self.log_body_links = list(self.body_links)
        if urdf_file is not None:
            all_body_names = resolve_urdf_child_link_names(urdf_file, G1_ALL_BODY_JOINT_NAMES)
            if all_body_names is not None:
                try:
                    self.log_body_links = [self.robot.get_link(name) for name in all_body_names]
                    self.log_body_names = list(all_body_names)
                except Exception:
                    self.log_body_links = list(self.body_links)
                    self.log_body_names = list(self.meta_body_names)

        self.anchor_idx = self.meta_body_names.index(anchor_body_name)
        self.root_idx = 0

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert Genesis or torch tensors to squeezed NumPy arrays."""

        if hasattr(tensor, "cpu"):
            return tensor.cpu().numpy().squeeze().astype(np.float32)
        return np.asarray(tensor, dtype=np.float32).squeeze()

    def call_genesis(self, fn: Any, values: np.ndarray, *args, **kwargs):
        """Call a Genesis function with 1D or batched tensors as needed."""

        tensor = torch.from_numpy(values.astype(np.float32))
        try:
            return fn(tensor, *args, **kwargs)
        except Exception:
            if tensor.ndim == 1:
                return fn(tensor.unsqueeze(0), *args, **kwargs)
            raise

    def update_reference_markers(self, reference: dict[str, np.ndarray]) -> None:
        """Move optional visual reference markers to the current target positions."""

        if not self.reference_markers:
            return

        reference_positions = reference["body_pos_w"].astype(np.float32)
        for marker_index, marker in enumerate(self.reference_markers):
            position = reference_positions[marker_index]
            if self.num_envs > 1:
                position = np.repeat(position[None, :], self.num_envs, axis=0)
            self.call_genesis(marker.set_pos, position, zero_velocity=False)

    def start_recording(self) -> None:
        """Start video recording when a camera is configured."""

        if self.camera is not None:
            self.camera.start_recording()

    def stop_recording(self, filename: str, fps: int) -> None:
        """Stop video recording and save the rendered file."""

        if self.camera is not None:
            self.camera.stop_recording(save_to_filename=filename, fps=fps)

    def render_camera(self, root_position: np.ndarray) -> None:
        """Render a follow camera frame for the current root position."""

        if self.camera is None:
            return

        self.camera.set_pose(
            pos=(float(root_position[0] + 2.5), float(root_position[1]), 1.6),
            lookat=(float(root_position[0]), float(root_position[1]), float(root_position[2] + 0.3)),
        )
        self.camera.render()

    def reset_to_reference(self, reference: dict[str, np.ndarray]) -> None:
        """Teleport the robot to the reference root pose and joint positions."""

        root_position = reference["body_pos_w"][self.root_idx]
        root_quaternion = reference["body_quat_w"][self.root_idx]
        joint_positions = reference["joint_pos"]

        qpos_size = max(7, max(self.joint_qs_indices) + 1)
        qpos = np.zeros(qpos_size, dtype=np.float32)
        qpos[:3] = root_position
        qpos[3:7] = root_quaternion
        for joint_index, qs_index in enumerate(self.joint_qs_indices):
            qpos[qs_index] = joint_positions[joint_index]

        qpos_command = np.repeat(qpos[None, :], self.num_envs, axis=0) if self.num_envs > 1 else qpos
        self.call_genesis(self.robot.set_qpos, qpos_command)

        qvel_size = len(self.joint_dof_indices) + 6
        qvel = np.zeros(qvel_size, dtype=np.float32)
        qvel_command = np.repeat(qvel[None, :], self.num_envs, axis=0) if self.num_envs > 1 else qvel
        self.call_genesis(self.robot.set_dofs_velocity, qvel_command)

    def extract_state_batch(self) -> dict[str, np.ndarray]:
        """Read joint and body state tensors from Genesis for all active environments."""

        joint_pos = self.to_numpy(self.robot.get_dofs_position(self.joint_dof_indices))
        joint_vel = self.to_numpy(self.robot.get_dofs_velocity(self.joint_dof_indices))
        if joint_pos.ndim == 1:
            joint_pos = joint_pos[None, :]
            joint_vel = joint_vel[None, :]

        def extract_link_states(links: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            body_pos = []
            body_quat = []
            body_lin_vel = []
            body_ang_vel = []
            for link in links:
                pos = self.to_numpy(link.get_pos())
                quat = self.to_numpy(link.get_quat())
                lin_vel = self.to_numpy(link.get_vel())
                ang_vel = self.to_numpy(link.get_ang())
                if pos.ndim == 1:
                    pos = pos[None, :]
                    quat = quat[None, :]
                    lin_vel = lin_vel[None, :]
                    ang_vel = ang_vel[None, :]
                body_pos.append(pos)
                body_quat.append(quat)
                body_lin_vel.append(lin_vel)
                body_ang_vel.append(ang_vel)
            return (
                np.stack(body_pos, axis=1),
                np.stack(body_quat, axis=1),
                np.stack(body_lin_vel, axis=1),
                np.stack(body_ang_vel, axis=1),
            )

        body_pos, body_quat, body_lin_vel, body_ang_vel = extract_link_states(self.body_links)
        log_body_pos, log_body_quat, log_body_lin_vel, log_body_ang_vel = extract_link_states(self.log_body_links)

        root_quat = body_quat[:, self.root_idx]
        root_lin_vel = body_lin_vel[:, self.root_idx]
        root_ang_vel = body_ang_vel[:, self.root_idx]

        return {
            "joint_pos": joint_pos.astype(np.float32),
            "joint_vel": joint_vel.astype(np.float32),
            "body_pos_w": body_pos.astype(np.float32),
            "body_quat_w": body_quat.astype(np.float32),
            "body_lin_vel_w": body_lin_vel.astype(np.float32),
            "body_ang_vel_w": body_ang_vel.astype(np.float32),
            "log_body_pos_w": log_body_pos.astype(np.float32),
            "log_body_quat_w": log_body_quat.astype(np.float32),
            "log_body_lin_vel_w": log_body_lin_vel.astype(np.float32),
            "log_body_ang_vel_w": log_body_ang_vel.astype(np.float32),
            "root_quat_w": root_quat.astype(np.float32),
            "root_lin_vel_w": root_lin_vel.astype(np.float32),
            "root_ang_vel_w": root_ang_vel.astype(np.float32),
        }

    def state_for_env(self, state_batch: dict[str, np.ndarray], env_id: int) -> dict[str, np.ndarray]:
        """Extract a single environment slice from a batched state dictionary."""

        return {key: value[env_id] for key, value in state_batch.items()}
