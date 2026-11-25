"""Batched MuJoCo scene: one MjModel/MjData per env, stepped in parallel threads.

PD control is compiled into the model as affine joint actuators (force = kp*(target - q) - kd*qd), so each env can
carry its own randomized gains and a whole control step runs in one GIL-free ``mj_step(nstep=...)`` call. Joint
``actuatorfrcrange`` limits in the MJCF clamp the PD torque, matching Genesis' effort-limit clipping.
"""

from __future__ import annotations

import copy
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from sim2sim_mujoco.constants import DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH


class MujocoSceneAdapter:
    """Own per-env MuJoCo models/data, reset, parallel stepping, state extraction and env-0 video."""

    def __init__(
        self,
        *,
        xml_file: str,
        sim_dt: float,
        control_dt: float,
        meta_body_names: list[str],
        joint_names: list[str],
        anchor_body_name: str,
        joint_stiffness: np.ndarray,
        joint_damping: np.ndarray,
        num_envs: int = 1,
        num_threads: int | None = None,
        torque_limit: float | None = None,
        record_video: bool = False,
        video_env_ids: list[int] | None = None,
        show_reference: bool = False,
        reference_marker_radius: float = 0.035,
        width: int = DEFAULT_VIDEO_WIDTH,
        height: int = DEFAULT_VIDEO_HEIGHT,
    ):
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError("MuJoCo is required: `python -m pip install mujoco`.") from exc

        ratio = float(control_dt) / float(sim_dt)
        if abs(ratio - round(ratio)) > 1.0e-6:
            raise ValueError(f"control_dt/sim_dt must be an integer. Got {control_dt=} {sim_dt=}.")
        self.mujoco = mujoco
        self.sim_decimation = int(round(ratio))
        self.num_envs = int(num_envs)
        self.meta_body_names = list(meta_body_names)
        self.joint_names = list(joint_names)
        self.record_video = bool(record_video)
        self.show_reference = bool(show_reference)
        self.reference_marker_radius = float(reference_marker_radius)
        self.width, self.height = int(width), int(height)

        spec = mujoco.MjSpec.from_file(xml_file)
        for name, kp, kd in zip(self.joint_names, joint_stiffness, joint_damping, strict=True):
            act = spec.add_actuator(name=f"pd_{name}", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            gain, bias = np.zeros(10), np.zeros(10)
            gain[0], bias[1], bias[2] = float(kp), -float(kp), -float(kd)
            act.gainprm, act.biasprm = gain, bias
            if torque_limit is not None:
                act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
                act.forcerange = np.array([-torque_limit, torque_limit], dtype=np.float64)
        base_model = spec.compile()
        base_model.opt.timestep = float(sim_dt)
        self.models = [copy.deepcopy(base_model) for _ in range(self.num_envs)]
        self.datas = [mujoco.MjData(model) for model in self.models]
        self.model, self.data = self.models[0], self.datas[0]

        self.joint_qpos_indices, self.joint_dof_indices = [], []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise KeyError(f"Joint '{joint_name}' from ONNX metadata is missing in MuJoCo model.")
            if base_model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise RuntimeError(f"Only 1-DoF hinge policy joints are supported. Got '{joint_name}'.")
            self.joint_qpos_indices.append(int(base_model.jnt_qposadr[joint_id]))
            self.joint_dof_indices.append(int(base_model.jnt_dofadr[joint_id]))
        self.joint_qpos_indices = np.array(self.joint_qpos_indices)
        self.joint_dof_indices = np.array(self.joint_dof_indices)

        self.body_ids = []
        for body_name in self.meta_body_names:
            body_id = mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise KeyError(f"Body '{body_name}' from ONNX metadata is missing in MuJoCo model.")
            self.body_ids.append(int(body_id))
        self.body_ids = np.array(self.body_ids)
        self.body_root_ids = base_model.body_rootid[self.body_ids]
        if anchor_body_name not in self.meta_body_names:
            raise KeyError(f"Anchor body '{anchor_body_name}' is not present in ONNX body_names metadata.")
        self.anchor_idx = self.meta_body_names.index(anchor_body_name)
        self.root_idx = 0
        self.root_body_id = int(self.body_ids[self.root_idx])

        free_ids = np.where(base_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
        if free_ids.size == 0:
            raise RuntimeError("MuJoCo model must contain a freejoint for the floating base.")
        self.free_joint_qpos_addr = int(base_model.jnt_qposadr[free_ids[0]])
        self.free_joint_dof_addr = int(base_model.jnt_dofadr[free_ids[0]])

        threads = num_threads or min(self.num_envs, os.cpu_count() or 1)
        self.pool = ThreadPoolExecutor(max_workers=threads) if self.num_envs > 1 and threads > 1 else None

        # One renderer per recorded env (default: env 0).
        self.video_env_ids = list(video_env_ids or [0]) if self.record_video else []
        for env_id in self.video_env_ids:
            if not 0 <= env_id < self.num_envs:
                raise ValueError(f"video env id {env_id} out of range for {self.num_envs} envs.")
        self.renderers, self.frames = {}, {env_id: [] for env_id in self.video_env_ids}
        self.camera = None
        if self.video_env_ids:
            for e in self.video_env_ids:
                self.models[e].vis.global_.offwidth = max(self.width, self.models[e].vis.global_.offwidth)
                self.models[e].vis.global_.offheight = max(self.height, self.models[e].vis.global_.offheight)
            self.renderers = {e: mujoco.Renderer(self.models[e], height=self.height, width=self.width)
                              for e in self.video_env_ids}
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    # ------------------------------------------------------------------ reset / step
    def reset_to_reference(
        self,
        reference: dict[str, np.ndarray],
        rng: np.random.Generator | None = None,
        joint_qpos_noise_range: tuple[float, float] | None = None,
    ) -> None:
        """Place every env at the first reference pose (optional per-env uniform joint noise), zero velocity."""

        root_pos = np.asarray(reference["body_pos_w"][self.root_idx], dtype=np.float64)
        root_quat = np.asarray(reference["body_quat_w"][self.root_idx], dtype=np.float64)
        joint_pos = np.asarray(reference["joint_pos"], dtype=np.float64).reshape(-1)
        a = self.free_joint_qpos_addr
        for model, data in zip(self.models, self.datas):
            self.mujoco.mj_resetData(model, data)
            data.qpos[a : a + 3] = root_pos
            data.qpos[a + 3 : a + 7] = root_quat
            noise = 0.0
            if rng is not None and joint_qpos_noise_range is not None:
                noise = rng.uniform(joint_qpos_noise_range[0], joint_qpos_noise_range[1], size=joint_pos.shape)
            data.qpos[self.joint_qpos_indices] = joint_pos + noise
            self.mujoco.mj_forward(model, data)

    def _step_env(self, env_id: int, target: np.ndarray) -> None:
        model, data = self.models[env_id], self.datas[env_id]
        data.ctrl[:] = target
        self.mujoco.mj_step(model, data, nstep=self.sim_decimation)
        self.mujoco.mj_forward(model, data)

    def step(self, joint_targets: np.ndarray, active: np.ndarray | None = None) -> None:
        """Advance active envs by one control step with joint position targets [N, J]."""

        targets = np.asarray(joint_targets, dtype=np.float64).reshape(self.num_envs, -1)
        env_ids = range(self.num_envs) if active is None else np.flatnonzero(active).tolist()
        if self.pool is None:
            for env_id in env_ids:
                self._step_env(env_id, targets[env_id])
        else:
            list(self.pool.map(lambda e: self._step_env(e, targets[e]), env_ids))

    def set_root_velocity(self, env_ids: np.ndarray, velocity: np.ndarray) -> None:
        """Set root [lin(3) world, ang(3)] velocity of selected envs (interval pushes)."""

        d = self.free_joint_dof_addr
        for env_id, vel in zip(env_ids, velocity):
            self.datas[env_id].qvel[d : d + 6] = vel

    # ------------------------------------------------------------------ state
    def extract_state_batch(self) -> dict[str, np.ndarray]:
        """Read a Genesis-compatible [N, ...] state batch (body velocities at body CoM, world frame)."""

        jp, jv, bp, bq, blv, bav = [], [], [], [], [], []
        for data in self.datas:
            jp.append(data.qpos[self.joint_qpos_indices])
            jv.append(data.qvel[self.joint_dof_indices])
            bp.append(data.xpos[self.body_ids])
            bq.append(data.xquat[self.body_ids])
            cvel = data.cvel[self.body_ids]
            offset = data.xipos[self.body_ids] - data.subtree_com[self.body_root_ids]
            bav.append(cvel[:, :3])
            blv.append(cvel[:, 3:] + np.cross(cvel[:, :3], offset))
        out = {
            "joint_pos": np.stack(jp), "joint_vel": np.stack(jv), "body_pos_w": np.stack(bp),
            "body_quat_w": np.stack(bq), "body_lin_vel_w": np.stack(blv), "body_ang_vel_w": np.stack(bav),
        }
        out = {key: value.astype(np.float32) for key, value in out.items()}
        out["root_quat_w"] = out["body_quat_w"][:, self.root_idx]
        out["root_lin_vel_w"] = out["body_lin_vel_w"][:, self.root_idx]
        out["root_ang_vel_w"] = out["body_ang_vel_w"][:, self.root_idx]
        return out

    @staticmethod
    def state_for_env(state_batch: dict[str, np.ndarray], env_id: int = 0) -> dict[str, np.ndarray]:
        return {key: value[env_id] for key, value in state_batch.items()}

    # ------------------------------------------------------------------ video (env 0)
    def render_frame(self, reference_batch: dict[str, np.ndarray] | None = None) -> None:
        """Render one frame for each recorded env; ``reference_batch`` is [N, ...] (markers per env)."""

        for env_id, renderer in self.renderers.items():
            data = self.datas[env_id]
            # Genesis follow camera: pos = (root_x, root_y + 2.5, 1.6), lookat = root + (0, 0, 0.2), fov 45.
            lookat = np.asarray(data.xpos[self.root_body_id]) + np.array([0.0, 0.0, 0.2])
            dz = 1.6 - lookat[2]
            self.camera.lookat[:] = lookat
            self.camera.azimuth = -90.0
            self.camera.elevation = -float(np.degrees(np.arctan2(dz, 2.5)))
            self.camera.distance = float(np.hypot(2.5, dz))
            renderer.update_scene(data, camera=self.camera)
            if self.show_reference and reference_batch is not None:
                scene = renderer.scene
                for position in np.asarray(reference_batch["body_pos_w"][env_id], dtype=np.float64).reshape(-1, 3):
                    if scene.ngeom >= scene.maxgeom:
                        break
                    self.mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom], self.mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([self.reference_marker_radius, 0.0, 0.0]), position, np.eye(3).reshape(-1),
                        np.array([1.0, 1.0, 1.0, 0.85], dtype=np.float32))
                    scene.ngeom += 1
            self.frames[env_id].append(renderer.render().copy())

    def save_video(self, filename: str, fps: int) -> list[str]:
        """Save one MP4 per recorded env (``<stem>_env<id>.mp4`` when several envs are recorded)."""

        import imageio.v3 as iio

        paths = []
        stem, ext = os.path.splitext(filename)
        for env_id, frames in self.frames.items():
            if not frames:
                continue
            path = filename if len(self.frames) == 1 else f"{stem}_env{env_id}{ext or '.mp4'}"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            iio.imwrite(path, np.asarray(frames, dtype=np.uint8), fps=fps)
            paths.append(path)
        return paths

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
        for renderer in self.renderers.values():
            if hasattr(renderer, "close"):
                renderer.close()
