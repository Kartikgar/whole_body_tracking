"""Default SONIC ONNX pair and its simulator-independent G1 observations."""
from collections import deque
import hashlib
from pathlib import Path

import numpy as np
import yaml

from .math_utils import quat_conjugate_wxyz, quat_mul_wxyz, quat_rotate_inverse_wxyz, quat_to_matrix_wxyz
from .spec import SonicSpec, MODEL_REVISION, SOURCE_REVISION

ENCODER_DIMS = {
    "encoder_mode_4": 4, "motion_joint_positions_10frame_step5": 290,
    "motion_joint_velocities_10frame_step5": 290, "motion_root_z_position_10frame_step5": 10,
    "motion_root_z_position": 1, "motion_anchor_orientation": 6,
    "motion_anchor_orientation_10frame_step5": 60, "motion_joint_positions_lowerbody_10frame_step5": 120,
    "motion_joint_velocities_lowerbody_10frame_step5": 120, "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12, "smpl_joints_10frame_step1": 720,
    "smpl_anchor_orientation_10frame_step1": 60, "motion_joint_positions_wrists_10frame_step1": 60,
}
HISTORY_DIMS = {"base_angular_velocity": 3, "body_joint_positions": 29, "body_joint_velocities": 29,
                "last_actions": 29, "gravity_dir": 3}
G1_REQUIRED = {"encoder_mode_4", "motion_joint_positions_10frame_step5",
               "motion_joint_velocities_10frame_step5", "motion_anchor_orientation_10frame_step5"}


class SonicObservations:
    """One history update per control tick, oldest first, zero-padded at reset."""
    requires_motion_action = False

    def __init__(self, config, motion, num_envs=1, spec=None):
        self.meta = spec or SonicSpec()
        self.motion = motion
        self.num_envs = num_envs
        self.encoder_names = [x["name"] for x in config["encoder"]["encoder_observations"] if x.get("enabled", True)]
        self.policy_names = [x["name"] for x in config["observations"] if x.get("enabled", True)]
        required = next((set(m["required_observations"]) for m in config["encoder"]["encoder_modes"]
                         if m["name"] == "g1" and m["mode_id"] == 0), None)
        supported_policy = {f"his_{k}_10frame_step1" for k in HISTORY_DIMS} | {"token_state"}
        if (required != G1_REQUIRED or not required.issubset(self.encoder_names)
                or set(self.encoder_names) - ENCODER_DIMS.keys()
                or set(self.policy_names) != supported_policy
                or len(set(self.policy_names)) != len(self.policy_names)
                or len(set(self.encoder_names)) != len(self.encoder_names)
                or config["encoder"]["dimension"] != 64):
            raise ValueError("Unsupported SONIC configuration: use the matched default release ONNX pair and YAML")
        self.encoder_dim = sum(ENCODER_DIMS[n] for n in self.encoder_names)
        self.decoder_dim = 64 + 10 * sum(HISTORY_DIMS.values())
        self.default_joint_pos_by_env = np.repeat(self.meta.default_joint_pos[None], num_envs, axis=0)
        self.reset()

    def reset(self):
        self.last_action = np.zeros((self.num_envs, 29), dtype=np.float32)
        self.history = {key: deque(maxlen=10) for key in HISTORY_DIMS}

    def default_joint_pos_for_batch(self, batch_size):
        return self.default_joint_pos_by_env[:batch_size]

    def compute_obs_dim(self, model_dim):
        if isinstance(model_dim, int) and model_dim != self.encoder_dim:
            raise ValueError(f"SONIC encoder dimension: YAML={self.encoder_dim}, model={model_dim}")
        return self.encoder_dim

    def build(self, state, reference, anchor_idx, step, obs_dim=None):
        del reference, anchor_idx, obs_dim
        q = state["root_quat_w"]
        current = {
            "body_joint_positions": state["joint_pos"] - self.default_joint_pos_by_env,
            "body_joint_velocities": state["joint_vel"], "last_actions": self.last_action.copy(),
            "base_angular_velocity": quat_rotate_inverse_wxyz(q, state["body_ang_vel_w"][:, 0]),
            "gravity_dir": quat_rotate_inverse_wxyz(q, np.broadcast_to([0., 0., -1.], (self.num_envs, 3))),
        }
        for key, value in current.items():
            entry = np.asarray(value, dtype=np.float32).copy()
            # Isaac Lab's CircularBuffer fills every history slot with the
            # first observation after reset, rather than zero-padding it.
            if not self.history[key]:
                self.history[key].extend(entry.copy() for _ in range(10))
            else:
                self.history[key].append(entry)
        # Reset starts at the reference pose, hence upstream initial heading correction is identity.
        relative = quat_mul_wxyz(quat_conjugate_wxyz(q)[:, None], self.motion.future("body_quat_w", step)[None, :, 0])
        values = {
            "encoder_mode_4": np.zeros((self.num_envs, 4), dtype=np.float32),
            "motion_joint_positions_10frame_step5": np.broadcast_to(self.motion.future("joint_pos", step).reshape(1, -1), (self.num_envs, 290)),
            "motion_joint_velocities_10frame_step5": np.broadcast_to(self.motion.future("joint_vel", step).reshape(1, -1), (self.num_envs, 290)),
            "motion_anchor_orientation_10frame_step5": quat_to_matrix_wxyz(relative)[..., :2].reshape(self.num_envs, 60),
        }
        return np.concatenate([values[n] if n in G1_REQUIRED else np.zeros((self.num_envs, ENCODER_DIMS[n]))
                               for n in self.encoder_names], axis=1).astype(np.float32)

    def decoder_observation(self, token):
        fields = {f"his_{k}_10frame_step1": np.stack(list(v), axis=1).reshape(self.num_envs, -1)
                  for k, v in self.history.items()}
        fields["token_state"] = token
        return np.concatenate([fields[n] for n in self.policy_names], axis=1).astype(np.float32)

    def update_last_action(self, action):
        self.last_action[:] = action


class SonicPolicy:
    """Policy/reference adapter compatible with the Genesis rollout runner."""
    reference_time_offset = 1

    def __init__(self, model_dir, motion, num_envs=1, policy_device="cuda", seed=None):
        import onnxruntime as ort

        self.meta = SonicSpec()
        self.motion = motion
        # One control tick is available per sample. The final tick holds the
        # final reference pose, allowing an N-sample 50 Hz clip to run N / 50 s.
        self.reference_motion_length_steps = max(motion.length, 1)
        self.policy_path = str(Path(model_dir) / "model_decoder.onnx")
        paths = [Path(model_dir) / name for name in ("model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml")]
        for p in paths:
            if not p.is_file():
                raise FileNotFoundError(f"Missing SONIC artifact {p}; run scripts/setup_sonic.py")
        with paths[2].open() as f:
            config = yaml.safe_load(f)
        self.observations = SonicObservations(config, motion, num_envs, self.meta)
        providers = ["CPUExecutionProvider"]
        if policy_device.startswith("cuda"):
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            else:
                print("[WARN] SONIC ONNX Runtime CUDA unavailable; using CPU inference.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.encoder = ort.InferenceSession(str(paths[0]), sess_options=options, providers=providers)
        self.decoder = ort.InferenceSession(str(paths[1]), sess_options=options, providers=providers)
        for label, session, width, output in (("encoder", self.encoder, self.observations.encoder_dim, 64),
                                               ("decoder", self.decoder, self.observations.decoder_dim, 29)):
            inputs, outputs = session.get_inputs(), session.get_outputs()
            if (len(inputs) != 1 or len(outputs) != 1 or len(inputs[0].shape) != 2
                    or inputs[0].shape[1] != width or outputs[0].shape[-1] != output
                    or inputs[0].type != "tensor(float)" or inputs[0].shape[0] not in (1, None, "batch_size", "batch")):
                raise ValueError(f"Unsupported SONIC {label} contract: {[(x.name, x.shape) for x in inputs + outputs]}")
        self.artifact_metadata = {"sonic_source_revision": SOURCE_REVISION, "sonic_model_revision": MODEL_REVISION,
                                  "trajectory_index": motion.trajectory_index,
                                  "onnx_providers": self.decoder.get_providers()}
        self.artifact_metadata.update({p.name + "_sha256": hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

    @staticmethod
    def _run(session, array):
        name = session.get_inputs()[0].name
        if session.get_inputs()[0].shape[0] == 1 and len(array) > 1:
            return np.concatenate([session.run(None, {name: row[None]})[0] for row in array])
        return session.run(None, {name: array})[0]

    def get_obs_input_dim(self):
        return self.observations.encoder_dim

    def reference_at(self, time_step, batch_size, obs_dim=None):
        return self.motion.reference_at(time_step, batch_size)

    def run(self, observation, time_step):
        token = self._run(self.encoder, observation)
        action = self._run(self.decoder, self.observations.decoder_observation(token)).astype(np.float32)
        if not np.all(np.isfinite(action)):
            raise RuntimeError("SONIC produced non-finite actions")
        return {"actions": action, **self.reference_at(int(time_step) + self.reference_time_offset, len(action))}
