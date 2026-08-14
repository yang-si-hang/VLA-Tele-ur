"""Provide shared UR action interpolation and RTDE recording helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from lerobot_robot_ur.ur import GRIPPER_FEATURE, TCP_FEATURES
from lerobot_robot_ur.ur_control import rot6d_to_rotation, rotation_to_rot6d
from scipy.spatial.transform import Rotation, Slerp

UR_ACTION_FEATURES = (*TCP_FEATURES, GRIPPER_FEATURE)
UR_ACTION_DIM = len(UR_ACTION_FEATURES)


def action_dict_to_array(action: Mapping[str, float]) -> np.ndarray:
    """Convert a named UR action to the canonical feature-ordered array."""

    try:
        values = np.asarray([action[name] for name in UR_ACTION_FEATURES], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"Action is missing required feature: {exc.args[0]}") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("Action must contain only finite values")
    return values


def action_array_to_dict(action: Sequence[float]) -> dict[str, float]:
    """Convert an ordered UR action array to a feature dictionary."""

    values = np.asarray(action, dtype=np.float64)
    if values.shape != (UR_ACTION_DIM,):
        raise ValueError(f"Action must have shape ({UR_ACTION_DIM},), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Action must contain only finite values")
    return dict(zip(UR_ACTION_FEATURES, values.tolist(), strict=True))


def interpolate_actions(previous_action: Sequence[float], next_action: Sequence[float], sample_count: int) -> np.ndarray:
    """Interpolate two UR actions linearly, using SLERP for orientation."""

    start_values = np.asarray(previous_action, dtype=np.float64)
    target_values = np.asarray(next_action, dtype=np.float64)
    if start_values.shape != (UR_ACTION_DIM,) or target_values.shape != (UR_ACTION_DIM,):
        raise ValueError(f"Actions must have shape ({UR_ACTION_DIM},)")
    if not np.all(np.isfinite(start_values)) or not np.all(np.isfinite(target_values)):
        raise ValueError("Actions must contain only finite values")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    fractions = np.arange(1, sample_count + 1, dtype=np.float64) / sample_count
    samples = start_values + fractions[:, None] * (target_values - start_values)
    rotations = Rotation.concatenate((rot6d_to_rotation(start_values[3:9]), rot6d_to_rotation(target_values[3:9])))
    for index, rotation in enumerate(Slerp([0.0, 1.0], rotations)(fractions)):
        samples[index, 3:9] = rotation_to_rot6d(rotation)
    return samples.astype(np.float32)


def action_record_header() -> tuple[str, ...]:
    """Return columns for one sampled-action RTDE record."""

    state_columns = tuple(f"state.{name}" for name in UR_ACTION_FEATURES)
    requested_columns = tuple(f"requested_action.{name}" for name in UR_ACTION_FEATURES)
    sent_columns = tuple(f"sent_action.{name}" for name in UR_ACTION_FEATURES)
    return ("controller_timestamp", "monotonic_time", "episode_frame", "sample_index", *state_columns, *requested_columns, *sent_columns)


def action_record_row(
    *,
    controller_timestamp: float,
    monotonic_time: float,
    frame_index: int,
    sample_index: int,
    state: Mapping[str, float],
    requested_action: Mapping[str, float],
    sent_action: Mapping[str, float],
) -> tuple[float | int, ...]:
    """Build one row aligned to a sampled send_action call."""

    return (
        controller_timestamp,
        monotonic_time,
        frame_index,
        sample_index,
        *(float(state[name]) for name in UR_ACTION_FEATURES),
        *(float(requested_action[name]) for name in UR_ACTION_FEATURES),
        *(float(sent_action[name]) for name in UR_ACTION_FEATURES),
    )
