"""Load Isaac state-action NPZ datasets for Genesis open-loop replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_STATE_KEYS: tuple[str, ...] = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

INITIAL_STATE_KEYS: tuple[str, ...] = tuple(f"initial_{key}" for key in DEFAULT_STATE_KEYS)

REPLAY_ACTION_KEYS: tuple[str, ...] = ("base_actions", "actions", "action")
SUPPORTED_ACTION_MODES: tuple[str, ...] = (
    "base_plus_delta_states",
    "base_policy_raw",
    "motion_joint_action_states",
)


@dataclass(frozen=True)
class StateActionDataset:
    """Parsed Isaac state-action transfer dataset."""

    path: str
    fps: float
    num_traj: int
    motion_length: int
    valid_lengths: np.ndarray
    replay_actions: np.ndarray
    source_states: dict[str, np.ndarray]
    initial_states: dict[str, np.ndarray]
    joint_names: list[str]
    body_names: list[str]
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    action_mode: str
    delta_actions: np.ndarray | None


def _read_string_array(value: np.ndarray) -> list[str]:
    array = np.asarray(value)
    if array.ndim == 0:
        return [str(array.item())]
    return [str(item) for item in array.reshape(-1).tolist()]


def _infer_valid_lengths(states: np.ndarray, *, atol: float = 1.0e-8, rtol: float = 1.0e-8) -> np.ndarray:
    if states.ndim != 3:
        raise ValueError(f"Expected 3D state array [num_traj, T, D], got {states.shape}.")
    num_traj, total_steps, _ = states.shape
    lengths = np.ones(num_traj, dtype=np.int32)
    for traj_idx in range(num_traj):
        traj = states[traj_idx]
        last_change = 0
        for step_idx in range(1, total_steps):
            if not np.allclose(traj[step_idx], traj[step_idx - 1], atol=atol, rtol=rtol):
                last_change = step_idx
        lengths[traj_idx] = last_change + 1
    return lengths


def _resolve_replay_action_key(files: set[str]) -> str:
    for key in REPLAY_ACTION_KEYS:
        if key in files:
            return key
    raise KeyError(f"No replay action key found. Expected one of {REPLAY_ACTION_KEYS}. Available: {sorted(files)}")


def load_state_action_npz(path: str) -> StateActionDataset:
    """Load and validate an Isaac state-action NPZ for Genesis base-only replay."""

    with np.load(path, allow_pickle=True) as data:
        files = set(data.files)
        if "fps" not in files:
            raise KeyError(f"Missing 'fps' in {path}.")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if fps <= 0.0:
            raise ValueError(f"Invalid fps={fps} in {path}.")

        action_key = _resolve_replay_action_key(files)
        replay_actions = np.asarray(data[action_key], dtype=np.float32)
        if replay_actions.ndim != 3:
            raise ValueError(f"Expected replay actions shape [num_traj, T, A]; got {replay_actions.shape}.")

        source_states: dict[str, np.ndarray] = {}
        for key in DEFAULT_STATE_KEYS:
            if key not in files:
                raise KeyError(f"Missing source state key '{key}' in {path}.")
            source_states[key] = np.asarray(data[key], dtype=np.float32)

        initial_states: dict[str, np.ndarray] = {}
        for key in INITIAL_STATE_KEYS:
            if key not in files:
                raise KeyError(
                    f"Missing initial state key '{key}' in {path}. "
                    "Re-record with updated play.py state-action logging."
                )
            initial_states[key] = np.asarray(data[key], dtype=np.float32)

        if "valid_lengths" in files:
            valid_lengths = np.asarray(data["valid_lengths"], dtype=np.int32).reshape(-1)
        else:
            valid_lengths = _infer_valid_lengths(source_states["joint_pos"])

        if "joint_names" not in files:
            raise KeyError(f"Missing 'joint_names' metadata in {path}.")
        joint_names = _read_string_array(data["joint_names"])

        body_names = _read_string_array(data["body_names"]) if "body_names" in files else []

        if "default_joint_pos" not in files:
            raise KeyError(f"Missing 'default_joint_pos' metadata in {path}.")
        default_joint_pos = np.asarray(data["default_joint_pos"], dtype=np.float32).reshape(-1)

        if "action_scale" not in files:
            raise KeyError(f"Missing 'action_scale' metadata in {path}.")
        action_scale = np.asarray(data["action_scale"], dtype=np.float32).reshape(-1)

        if "action_mode" in files:
            action_mode = str(np.asarray(data["action_mode"]).reshape(-1)[0])
        else:
            action_mode = "base_policy_raw"

        delta_actions = None
        if "delta_actions" in files:
            delta_actions = np.asarray(data["delta_actions"], dtype=np.float32)

    if action_mode not in SUPPORTED_ACTION_MODES:
        raise ValueError(
            f"Unsupported action_mode={action_mode!r} in {path}. Expected one of {SUPPORTED_ACTION_MODES}."
        )

    num_traj = int(replay_actions.shape[0])
    motion_length = int(replay_actions.shape[1])
    if valid_lengths.shape[0] != num_traj:
        raise ValueError(f"valid_lengths length {valid_lengths.shape[0]} != num_traj {num_traj}.")
    if np.any(valid_lengths <= 0) or np.any(valid_lengths > motion_length):
        raise ValueError(f"Invalid valid_lengths in {path}: {valid_lengths}")

    return StateActionDataset(
        path=path,
        fps=fps,
        num_traj=num_traj,
        motion_length=motion_length,
        valid_lengths=valid_lengths,
        replay_actions=replay_actions,
        source_states=source_states,
        initial_states=initial_states,
        joint_names=joint_names,
        body_names=body_names,
        default_joint_pos=default_joint_pos,
        action_scale=action_scale,
        action_mode=action_mode,
        delta_actions=delta_actions,
    )


def select_trajectory_indices(num_traj: int, text: str | None) -> list[int]:
    """Parse a trajectory slice like ``0:10`` or ``3,5,7``."""

    if text is None or text.strip() == "":
        return list(range(num_traj))

    text = text.strip()
    if "," in text:
        indices = [int(item.strip()) for item in text.split(",") if item.strip()]
    elif ":" in text:
        start_text, end_text = text.split(":", 1)
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else num_traj
        indices = list(range(start, end))
    else:
        indices = [int(text)]

    for index in indices:
        if index < 0 or index >= num_traj:
            raise ValueError(f"Trajectory index {index} out of range [0, {num_traj}).")
    return indices
