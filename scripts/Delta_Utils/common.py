"""Shared helpers for Delta_Utils analysis scripts.

Replay runs are directories written by ``eval_delta_replay_isaac.py`` (``manifest.json``, ``summary.json``,
``windows.json``, ``partial_results.json``). Closed-loop helpers cover the two rollout conventions:
Isaac ``play.py --batch_eval`` arrays are post-action and include scene origins; Genesis ``--record_motion``
arrays are pre-action with frame 0 = the restored initial state.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MATCHED_RUN_KEYS = ('dataset_sha256', 'seed', 'num_envs', 'replay_steps', 'measurement_interval_s')
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0


# ---------------------------------------------------------------------------- replay runs

def load_run(path: Path) -> dict:
    """Read one replay run directory."""
    path = Path(path)
    return dict(path=path,
                manifest=json.loads((path / 'manifest.json').read_text()),
                summary=json.loads((path / 'summary.json').read_text()),
                windows=json.loads((path / 'windows.json').read_text()),
                rows=json.loads((path / 'partial_results.json').read_text()))


def matched_run_problem(runs: dict[str, dict], require_complete: bool = False) -> str | None:
    """Return why runs cannot be compared window-for-window, or None if they can."""
    labels = list(runs)
    first = runs[labels[0]]
    for label in labels:
        run = runs[label]
        if require_complete and run['manifest']['status'] != 'complete':
            return f'Incomplete run: {label}'
        for key in MATCHED_RUN_KEYS:
            if run['manifest'][key] != first['manifest'][key]:
                return f'{label} differs in {key}'
        if run['windows'] != first['windows']:
            return f'{label} has a different schedule'
        if run['summary']['configuration_contract'] != first['summary']['configuration_contract']:
            return f'{label} has different physical/observation/action settings'
    return None


def grouped_curve_values(rows, field: str = 'error_curve') -> dict:
    """Map (trajectory, requested_time_s) to the curve points of every window at that time."""
    grouped = defaultdict(list)
    for row in rows:
        for point in row[field]:
            grouped[row['trajectory'], point['requested_time_s']].append(point)
    return grouped


def trajectory_time_matrix(grouped: dict, key: str, trajectories, times) -> np.ndarray:
    """Average overlapping windows within each trajectory; NaN where a trajectory has no sample."""
    return np.array([[np.mean([p[key] for p in grouped[trajectory, time]]) if grouped[trajectory, time] else np.nan
                      for time in times] for trajectory in trajectories])


def bootstrap_indices(num_trajectories: int) -> np.ndarray:
    """Trajectory resampling indices shared by all metrics of one analysis."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(num_trajectories, size=(BOOTSTRAP_SAMPLES, num_trajectories))


def bootstrap_ci(matrix: np.ndarray, indices: np.ndarray, ignore_nan: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise 95% interval of the trajectory-equal-weight mean."""
    if ignore_nan:
        low, high = np.nanpercentile(np.nanmean(matrix[indices], axis=1), [2.5, 97.5], axis=0)
    else:
        low, high = np.percentile(matrix[indices].mean(axis=1), [2.5, 97.5], axis=0)
    return low, high


# ---------------------------------------------------------------------------- closed-loop rollouts

def quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation angle between wxyz quaternions, sign-invariant."""
    return 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1))


def euler_wxyz(q: np.ndarray):
    """Roll, pitch, yaw from wxyz quaternions."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def load_isaac_rollout(path: Path) -> dict:
    """Isaac batch rollout on a frame axis: index k is the state at k/fps (k=0 initial), origins removed.

    ``alive[i, k]`` is True for k <= valid_length (the recorded post-action states, including a failing one).
    """
    with np.load(path, allow_pickle=True) as d:
        origins = d['env_origins'][:, None, None, :]
        lengths = d['valid_lengths'].astype(int)

        def series(key):
            return np.concatenate([d['initial_' + key][:, None], d[key]], axis=1)

        out = dict(joint_pos=series('joint_pos'), body_pos=series('body_pos_w') - origins,
                   body_quat=series('body_quat_w'),
                   joint_names=list(map(str, d['joint_names'])), body_names=list(map(str, d['body_names'])),
                   fps=float(d['fps'][0]))
    out['alive'] = np.arange(out['joint_pos'].shape[1])[None, :] <= lengths[:, None]
    return out


def load_genesis_rollout(path: Path, joint_names, body_names) -> dict:
    """Genesis ``--record_motion`` rollout reordered to the given names; pre-action, so frame k = state at k/fps.

    ``alive[i, k]`` is True for k < valid_length (the failing post-action state is not recorded).
    """
    with np.load(path, allow_pickle=True) as d:
        jn, bn = list(map(str, d['joint_names'])), list(map(str, d['body_names']))
        ji, bi = [jn.index(n) for n in joint_names], [bn.index(n) for n in body_names]
        lengths = d['valid_lengths'].astype(int)
        out = dict(joint_pos=d['joint_pos'][:, :, ji], body_pos=d['body_pos_w'][:, :, bi],
                   body_quat=d['body_quat_w'][:, :, bi])
    out['alive'] = np.arange(out['joint_pos'].shape[1])[None, :] < lengths[:, None]
    return out


# ---------------------------------------------------------------------------- figures

def use_agg():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def save_figure(fig, out: Path, name: str, dpi: int = 300):
    """Save PNG and PDF side by side and close the figure."""
    import matplotlib.pyplot as plt
    fig.savefig(Path(out) / (name + '.png'), dpi=dpi, bbox_inches='tight')
    fig.savefig(Path(out) / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
