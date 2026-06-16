"""ONNX policy loading and metadata handling for modular sim2sim evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from sim2sim_genesis.constants import EVAL_KP_KD_PERTURB_RNG_SEED, KP_KD_PERTURB_SCALE


def parse_csv_list(value: str | None, cast_type: Any = str) -> list[Any]:
    """Parse an ONNX metadata CSV field into a typed Python list."""

    if value is None or value == "":
        return []

    items = [item for item in value.split(",") if item != ""]
    if cast_type is str:
        return items
    if cast_type is int:
        return [int(round(float(item))) for item in items]
    return [cast_type(item) for item in items]


def load_onnx_metadata(path: str) -> dict[str, str]:
    """Load metadata properties from an ONNX file."""

    model = onnx.load(path)
    return {entry.key: entry.value for entry in model.metadata_props}


def infer_reference_length_from_policy(path: str) -> int | None:
    """Infer the embedded reference motion length from the ONNX graph."""

    model = onnx.load(path)
    graph = model.graph
    input_names = {input_tensor.name for input_tensor in graph.input}
    if "time_step" not in input_names:
        return None

    producer_by_output: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for output_name in node.output:
            producer_by_output[output_name] = node

    scalar_constants: dict[str, float] = {}
    for initializer in graph.initializer:
        array = np.asarray(numpy_helper.to_array(initializer))
        if array.size == 1:
            scalar_constants[initializer.name] = float(array.reshape(()))

    for node in graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attribute in node.attribute:
            if attribute.name != "value" or not attribute.HasField("t"):
                continue
            array = np.asarray(numpy_helper.to_array(attribute.t))
            if array.size == 1:
                scalar_constants[node.output[0]] = float(array.reshape(()))

    depends_cache: dict[str, bool] = {}

    def depends_on_time_step(tensor_name: str) -> bool:
        if tensor_name == "time_step":
            return True
        if tensor_name in depends_cache:
            return depends_cache[tensor_name]

        producer = producer_by_output.get(tensor_name)
        if producer is None:
            depends_cache[tensor_name] = False
            return False

        depends = any(depends_on_time_step(input_name) for input_name in producer.input if input_name != "")
        depends_cache[tensor_name] = depends
        return depends

    candidates: list[int] = []

    def append_candidate(value: float) -> None:
        rounded = int(round(value))
        if abs(value - rounded) > 1e-5 or rounded < 0:
            return
        candidates.append(rounded + 1)

    for node in graph.node:
        if node.op_type == "Clip" and len(node.input) >= 3:
            x_name, _, max_name = node.input[:3]
            if x_name and depends_on_time_step(x_name) and max_name in scalar_constants:
                append_candidate(scalar_constants[max_name])
            continue

        if node.op_type == "Min" and len(node.input) == 2:
            first_name, second_name = node.input
            first_depends = first_name != "" and depends_on_time_step(first_name)
            second_depends = second_name != "" and depends_on_time_step(second_name)
            if first_depends == second_depends:
                continue
            constant_name = second_name if first_depends else first_name
            if constant_name in scalar_constants:
                append_candidate(scalar_constants[constant_name])

    if not candidates:
        return None
    return max(candidates)


@dataclass(slots=True)
class PolicyMeta:
    """Metadata required to rebuild policy observations and control semantics."""

    joint_names: list[str]
    default_joint_pos: np.ndarray
    joint_stiffness: np.ndarray
    joint_damping: np.ndarray
    action_scale: np.ndarray
    observation_names: list[str]
    observation_history_lengths: list[int]
    body_names: list[str]
    anchor_body_name: str


def apply_kp_kd_perturbation(
    joint_stiffness: np.ndarray,
    joint_damping: np.ndarray,
    rng_seed: int = EVAL_KP_KD_PERTURB_RNG_SEED,
    scale: float = KP_KD_PERTURB_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the legacy fixed-seed Gaussian Kp/Kd perturbation used for target-env sim2sim."""

    stiffness = np.asarray(joint_stiffness, dtype=np.float32)
    damping = np.asarray(joint_damping, dtype=np.float32)
    if stiffness.shape != damping.shape:
        raise ValueError(
            f"joint_stiffness/joint_damping shape mismatch: {stiffness.shape} vs {damping.shape}"
        )

    rng = np.random.default_rng(rng_seed)
    noise = rng.standard_normal(stiffness.shape).astype(np.float32)
    multiplier = 1.0 + scale * noise
    return (stiffness * multiplier).astype(np.float32), (damping * multiplier).astype(np.float32)


def parse_policy_meta(raw_meta: dict[str, str]) -> PolicyMeta:
    """Parse the ONNX metadata contract exported by `attach_onnx_metadata`."""

    required = [
        "joint_names",
        "default_joint_pos",
        "joint_stiffness",
        "joint_damping",
        "action_scale",
        "observation_names",
        "body_names",
        "anchor_body_name",
    ]
    missing = [key for key in required if key not in raw_meta]
    if missing:
        raise RuntimeError(
            "Missing ONNX metadata keys: "
            f"{missing}. Re-export policy with `attach_onnx_metadata` from this codebase."
        )

    observation_names = parse_csv_list(raw_meta["observation_names"], str)
    history_lengths = parse_csv_list(raw_meta.get("observation_history_lengths"), int)
    if len(history_lengths) != len(observation_names):
        history_lengths = [1] * len(observation_names)

    return PolicyMeta(
        joint_names=parse_csv_list(raw_meta["joint_names"], str),
        default_joint_pos=np.array(parse_csv_list(raw_meta["default_joint_pos"], float), dtype=np.float32),
        joint_stiffness=np.array(parse_csv_list(raw_meta["joint_stiffness"], float), dtype=np.float32),
        joint_damping=np.array(parse_csv_list(raw_meta["joint_damping"], float), dtype=np.float32),
        action_scale=np.array(parse_csv_list(raw_meta["action_scale"], float), dtype=np.float32),
        observation_names=observation_names,
        observation_history_lengths=[max(int(value), 1) for value in history_lengths],
        body_names=parse_csv_list(raw_meta["body_names"], str),
        anchor_body_name=raw_meta["anchor_body_name"],
    )


class OnnxMotionPolicy:
    """Wrap an exported ONNX motion-tracking policy and its embedded reference motion."""

    def __init__(self, policy_path: str, policy_device: str, seed: int | None = None):
        """Create the ONNX Runtime session and validate the exported model contract."""

        if not policy_path.endswith(".onnx"):
            raise ValueError("This evaluator expects an ONNX policy exported by this codebase.")

        self.policy_path = policy_path
        self.seed = seed
        self.meta = parse_policy_meta(load_onnx_metadata(policy_path))
        self.reference_motion_length_steps = infer_reference_length_from_policy(policy_path)
        if self.reference_motion_length_steps is None or self.reference_motion_length_steps <= 0:
            raise RuntimeError(
                "Failed to infer reference motion length from exported ONNX policy. "
                "Re-export the policy with this codebase's motion exporter."
            )

        self.num_joints = len(self.meta.joint_names)
        self.num_actions = len(self.meta.action_scale)
        if self.num_actions != self.num_joints:
            raise RuntimeError(
                f"Unsupported action/joint mismatch: action_dim={self.num_actions}, joint_dim={self.num_joints}."
            )
        if len(self.meta.default_joint_pos) != self.num_joints:
            raise RuntimeError("default_joint_pos length does not match joint_names.")

        available_providers = ort.get_available_providers()
        use_cuda = policy_device.startswith("cuda") and "CUDAExecutionProvider" in available_providers
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]

        if seed is not None:
            session_options = ort.SessionOptions()
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.enable_mem_pattern = False
            session_options.enable_cpu_mem_arena = False
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(policy_path, sess_options=session_options, providers=providers)
        else:
            self.session = ort.InferenceSession(policy_path, providers=providers)

        input_names = [input_tensor.name for input_tensor in self.session.get_inputs()]
        if "obs" not in input_names or "time_step" not in input_names:
            raise RuntimeError(f"ONNX inputs must include `obs` and `time_step`. Found={input_names}")
        self.obs_input_name = "obs"
        self.time_input_name = "time_step"

        output_names = [output_tensor.name for output_tensor in self.session.get_outputs()]
        required_outputs = {
            "actions",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }
        if not required_outputs.issubset(set(output_names)):
            raise RuntimeError(f"ONNX outputs missing required names. Found={output_names}")
        self.output_names = output_names

    def get_obs_input_dim(self) -> int | str | None:
        """Return the ONNX model observation input width."""

        input_tensor = next(input_tensor for input_tensor in self.session.get_inputs() if input_tensor.name == "obs")
        return input_tensor.shape[-1]

    def run(self, observation: np.ndarray, time_step: int | np.ndarray) -> dict[str, np.ndarray]:
        """Run policy inference for a batch of observations and timesteps."""

        observation_batch = np.asarray(observation, dtype=np.float32)
        if observation_batch.ndim == 1:
            observation_batch = observation_batch.reshape(1, -1)
        if observation_batch.ndim != 2:
            raise ValueError(f"Expected obs shape [B, D], got {observation_batch.shape}")

        batch_size = int(observation_batch.shape[0])
        if np.isscalar(time_step):
            time_input = np.full((batch_size, 1), float(time_step), dtype=np.float32)
        else:
            time_array = np.asarray(time_step, dtype=np.float32).reshape(-1)
            if time_array.size == 1 and batch_size > 1:
                time_array = np.full((batch_size,), float(time_array.item()), dtype=np.float32)
            if time_array.size != batch_size:
                raise ValueError(
                    f"time_step batch size mismatch: obs_batch={batch_size}, time_step={time_array.size}"
                )
            time_input = time_array.reshape(batch_size, 1)

        try:
            outputs = self.session.run(
                self.output_names,
                {
                    self.obs_input_name: observation_batch,
                    self.time_input_name: time_input,
                },
            )
        except Exception:
            if batch_size == 1:
                raise

            per_env_outputs = []
            for env_id in range(batch_size):
                env_outputs = self.session.run(
                    self.output_names,
                    {
                        self.obs_input_name: observation_batch[env_id : env_id + 1],
                        self.time_input_name: time_input[env_id : env_id + 1],
                    },
                )
                per_env_outputs.append(env_outputs)

            outputs = []
            for output_index in range(len(self.output_names)):
                outputs.append(
                    np.concatenate(
                        [np.asarray(per_env_outputs[env_id][output_index], dtype=np.float32) for env_id in range(batch_size)],
                        axis=0,
                    )
                )

        parsed: dict[str, np.ndarray] = {}
        for name, value in zip(self.output_names, outputs, strict=True):
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 0:
                array = array.reshape(batch_size, 1)
            elif array.shape[0] != batch_size:
                if batch_size == 1:
                    array = array.reshape((1, *array.shape))
                else:
                    raise RuntimeError(
                        f"Policy output '{name}' has unexpected batch dim {array.shape[0]} (expected {batch_size})."
                    )
            parsed[name] = array.reshape(batch_size, -1) if name == "actions" else array
        return parsed

    def reference_at(self, time_step: int, batch_size: int, obs_dim: int) -> dict[str, np.ndarray]:
        """Fetch the embedded reference tensors for a timestep by running zero observations."""

        zero_obs = np.zeros((batch_size, obs_dim), dtype=np.float32)
        output = self.run(zero_obs, time_step)
        return {
            "joint_pos": output["joint_pos"].astype(np.float32),
            "joint_vel": output["joint_vel"].astype(np.float32),
            "body_pos_w": output["body_pos_w"].astype(np.float32),
            "body_quat_w": output["body_quat_w"].astype(np.float32),
            "body_lin_vel_w": output["body_lin_vel_w"].astype(np.float32),
            "body_ang_vel_w": output["body_ang_vel_w"].astype(np.float32),
        }
